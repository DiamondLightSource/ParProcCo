from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from .data_slicer_interface import DataSlicerInterface
from .job_scheduler import JobScheduler
from .job_scheduling_information import JobResources, JobSchedulingInformation
from .program_wrapper import ProgramWrapper
from .utils import check_jobscript_is_readable, check_location, get_absolute_path

AGGREGATION_TIME = 60  # timeout per single file, in seconds


class JobController:
    def __init__(
        self,
        url: str,
        program_wrapper: ProgramWrapper,
        output_dir_or_file: Path,
        partition: str,
        user_name: str | None = None,
        user_token: str | None = None,
        timeout: timedelta = timedelta(hours=2),
    ) -> None:
        """JobController is used to coordinate cluster job submissions with
        JobScheduler"""
        self.url = url
        self.program_wrapper = program_wrapper
        self.partition = partition
        self.output_file: Path | None = None
        self.cluster_output_dir: Path | None = None

        if output_dir_or_file is not None:
            logging.debug("JC output: %s", output_dir_or_file)
            if output_dir_or_file.is_dir():
                output_dir = output_dir_or_file
            else:
                output_dir = output_dir_or_file.parent
                self.output_file = output_dir_or_file
            self.cluster_output_dir = check_location(output_dir)
            logging.debug(
                "JC cluster output: %s; file %s",
                self.cluster_output_dir,
                self.output_file,
            )
        try:
            self.working_directory: Path | None = check_location(os.getcwd())
        except Exception:
            logging.warning(
                "Could not use %s as working directory on cluster so using %s",
                os.getcwd(),
                self.cluster_output_dir,
            )
            self.working_directory = self.cluster_output_dir
        logging.debug("JC working dir: %s", self.working_directory)
        self.data_slicer: DataSlicerInterface
        self.user_name = user_name
        self.user_token = user_token
        self.timeout = timeout
        self.sliced_results: tuple[Path, ...] | None = None
        self.aggregated_result: Path | None = None

    def run(
        self,
        number_jobs: int,
        jobscript_args: list[str] | None = None,
        job_name: str = "ParProcCo",
        processing_job_resources: JobResources | None = None,
        aggregation_job_resources: JobResources | None = None,
    ) -> None:
        self.cluster_runner = self.program_wrapper.get_process_script()
        if self.cluster_runner is None:
            raise ValueError("Processing script must be defined")
        if processing_job_resources is None:
            processing_job_resources = JobResources()
        if aggregation_job_resources is None:
            aggregation_job_resources = JobResources(
                memory=processing_job_resources.memory,
                cpu_cores=1,
                extra_properties=processing_job_resources.extra_properties,
            )
        self.cluster_env = self.program_wrapper.get_environment()
        logging.debug("Cluster environment is %s", self.cluster_env)

        timestamp = datetime.now()
        if jobscript_args:
            jobscript_args[0] = str(
                check_jobscript_is_readable(
                    check_location(get_absolute_path(jobscript_args[0]))
                )
            )
        sliced_jobs_success = self._submit_sliced_jobs(
            number_jobs,
            jobscript_args,
            processing_job_resources,
            job_name,
            timestamp,
        )

        if sliced_jobs_success and self.sliced_results:
            logging.info("Sliced jobs ran successfully.")
            out_file: Path | None = None
            if number_jobs == 1:
                out_file = self.sliced_results[0] if self.sliced_results else None
            else:
                self._submit_aggregation_job(aggregation_job_resources, timestamp)
                out_file = self.aggregated_result

            if (
                out_file is not None
                and out_file.is_file()
                and self.output_file is not None
            ):
                renamed_file = out_file.rename(self.output_file)
                logging.debug(
                    "Rename %s to %s: %s", out_file, renamed_file, renamed_file.exists()
                )
        else:
            slice_params = self.program_wrapper.create_slices(number_jobs=number_jobs)
            logging.error(
                f"Sliced jobs failed with slice_params: {slice_params},"
                f" jobscript_args: {jobscript_args}, job_name: {job_name}"
            )
            raise RuntimeError("Sliced jobs failed\n")

    def _submit_sliced_jobs(
        self,
        number_of_jobs: int,
        jobscript_args: list[str] | None,
        job_resources: JobResources,
        job_name: str,
        timestamp: datetime,
    ) -> bool:
        if jobscript_args is None:
            jobscript_args = []

        assert self.cluster_runner
        jsi = JobSchedulingInformation(
            job_name=job_name,
            job_script_path=self.cluster_runner,
            job_resources=job_resources,
            timeout=self.timeout,
            job_script_arguments=tuple(jobscript_args),
            working_directory=self.working_directory,
            output_dir=self.output_file.parent if self.output_file else None,
            output_filename=self.output_file.name if self.output_file else None,
            log_directory=self.cluster_output_dir,
            timestamp=timestamp,
        )
        jsi.set_job_env(self.cluster_env)

        job_scheduler = JobScheduler(
            url=self.url,
            partition=self.partition,
            user_name=self.user_name,
            user_token=self.user_token,
        )

        processing_jobs = self.program_wrapper.processing_slicer.create_slice_jobs(
            jsi, self.program_wrapper.create_slices(number_jobs=number_of_jobs)
        )

        sliced_jobs_success = job_scheduler.run(processing_jobs)

        if not sliced_jobs_success:
            sliced_jobs_success = job_scheduler.resubmit_killed_jobs()

        self.sliced_results = (
            job_scheduler.get_output_paths(processing_jobs)
            if sliced_jobs_success
            else None
        )
        return sliced_jobs_success

    def _submit_aggregation_job(
        self, job_resources: JobResources, timestamp: datetime
    ) -> None:
        aggregator_path = self.program_wrapper.get_aggregate_script()
        aggregating_slicer = self.program_wrapper.aggregating_slicer
        if aggregating_slicer is None or self.sliced_results is None:
            return

        aggregation_args = []
        if aggregator_path is not None:
            aggregation_args.append(str(aggregator_path))

        assert self.sliced_results is not None and self.cluster_runner
        jsi = JobSchedulingInformation(
            job_name=aggregating_slicer.__class__.__name__,
            job_script_path=self.cluster_runner,
            job_resources=job_resources,
            job_script_arguments=tuple(aggregation_args),
            working_directory=self.working_directory,
            timeout=timedelta(seconds=AGGREGATION_TIME * len(self.sliced_results)),
            output_dir=self.output_file.parent if self.output_file else None,
            output_filename=self.output_file.name if self.output_file else None,
            log_directory=self.cluster_output_dir,
            timestamp=timestamp,
        )
        jsi.set_job_env(self.cluster_env)

        aggregation_scheduler = JobScheduler(
            url=self.url,
            partition=self.partition,
            user_name=self.user_name,
            user_token=self.user_token,
        )

        aggregation_jobs = aggregating_slicer.create_slice_jobs(
            jsi, list(self.sliced_results)
        )

        aggregation_success = aggregation_scheduler.run(aggregation_jobs)

        if not aggregation_success:
            aggregation_scheduler.resubmit_killed_jobs(allow_all_failed=True)

        if aggregation_success:
            self.aggregated_result = aggregation_scheduler.get_output_paths(
                aggregation_jobs
            )[0]
            for result in self.sliced_results:
                os.remove(str(result))
        else:
            logging.warning(
                "Aggregated job was unsuccessful with aggregating_slicer:"
                f" {aggregating_slicer}, cluster_runner: {self.cluster_runner},"
                f" cluster_env: {self.cluster_env}, aggregator_path: {aggregator_path},"
                f" aggregation_args: {aggregation_args}"
            )
            self.aggregated_result = None
