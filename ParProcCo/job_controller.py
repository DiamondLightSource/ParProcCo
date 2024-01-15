from __future__ import annotations

import logging
import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Optional

from .job_scheduler import JobScheduler
from .slicer_interface import SlicerInterface
from .utils import check_location, get_absolute_path
from .program_wrapper import ProgramWrapper
from .job_scheduling_information import JobSchedulingInformation, JobResources

AGGREGATION_TIME = 60  # timeout per single file, in seconds


class JobController:
    def __init__(
        self,
        url: str,
        program_wrapper: ProgramWrapper,
        output_dir_or_file: Path,
        partition: str,
        extra_properties: Optional[dict[str, str]] = None,
        user_name: Optional[str] = None,
        user_token: Optional[str] = None,
        timeout: timedelta = timedelta(hours=2),
    ) -> None:
        """JobController is used to coordinate cluster job submissions with JobScheduler"""
        self.url = url
        self.program_wrapper = program_wrapper
        self.partition = partition
        self.extra_properties = extra_properties
        self.output_file: Optional[Path] = None
        self.cluster_output_dir: Optional[Path] = None

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
            self.working_directory: Optional[Path] = check_location(os.getcwd())
        except Exception:
            logging.warning(
                "Could not use %s as working directory on cluster so using %s",
                os.getcwd(),
                self.cluster_output_dir,
            )
            self.working_directory = self.cluster_output_dir
        logging.debug("JC working dir: %s", self.working_directory)
        self.data_slicer: SlicerInterface
        self.user_name = user_name
        self.user_token = user_token
        self.timeout = timeout
        self.sliced_results: Optional[List[Path]] = None
        self.aggregated_result: Optional[Path] = None

    def run(
        self,
        number_jobs: int,
        processing_job_resources: JobResources,
        aggregation_job_resources: JobResources,
        jobscript_args: Optional[List] = None,
        job_name: str = "ParProcCo",
    ) -> None:
        self.cluster_runner = check_location(
            get_absolute_path(self.program_wrapper.get_cluster_runner_script())
        )
        self.cluster_env = self.program_wrapper.get_environment()
        logging.debug("Cluster environment is %s", self.cluster_env)

        timestamp = datetime.now()
        sliced_jobs_success = self._submit_sliced_jobs(
            number_jobs,
            jobscript_args,
            processing_job_resources,
            job_name,
            timestamp=timestamp,
        )

        if sliced_jobs_success and self.sliced_results:
            logging.info("Sliced jobs ran successfully.")
            if number_jobs == 1:
                out_file = (
                    self.sliced_results[0] if len(self.sliced_results) > 0 else None
                )
            else:
                self._submit_aggregation_job(
                    aggregation_job_resources, timestamp=timestamp
                )
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
                f"Sliced jobs failed with slice_params: {slice_params}, jobscript_args: {jobscript_args},"
                f" job_name: {job_name}"
            )
            raise RuntimeError("Sliced jobs failed\n")

    def _submit_sliced_jobs(
        self,
        number_of_jobs: int,
        jobscript_args: Optional[List],
        memory: int,
        job_name: str,
        timestamp: datetime,
    ) -> bool:
        if jobscript_args is None:
            jobscript_args = []

        processing_mode = self.program_wrapper.processing_mode

        job_resources = JobResources(
            memory=memory,
            cpu_cores=processing_mode.cores,
            gpus=0,
            extra_properties=self.extra_properties,
        )

        jsi = JobSchedulingInformation(
            job_name=job_name,
            job_script_path=self.cluster_runner,
            job_resources=job_resources,
            timeout=self.timeout,
            job_script_arguments=jobscript_args,
            job_env=self.cluster_env,
            working_directory=self.working_directory,
            output_dir=self.output_file.parent if self.output_file else None,
            output_filename=self.output_file.name if self.output_file else None,
            log_directory=self.cluster_output_dir,
            timestamp=timestamp,
        )

        job_scheduler = JobScheduler(
            url=self.url,
            partition=self.partition,
            user_name=self.user_name,
            user_token=self.user_token,
            cluster_output_dir=self.cluster_output_dir,
        )

        start_time = datetime.now()
        processing_jobs = self.program_wrapper.create_sliced_processing_jobs(
            job_scheduling_information=jsi,
            t=start_time,
            slice_params=self.program_wrapper.create_slices(number_jobs=number_of_jobs),
        )

        sliced_jobs_success = job_scheduler.run(processing_jobs, start_time=start_time)

        if not sliced_jobs_success:
            sliced_jobs_success = job_scheduler.resubmit_killed_jobs()

        self.sliced_results = (
            job_scheduler.get_output_paths() if sliced_jobs_success else None
        )
        return sliced_jobs_success

    def _submit_aggregation_job(self, memory: int, timestamp: datetime) -> None:
        aggregator_path = self.program_wrapper.get_aggregate_script()
        aggregating_mode = self.program_wrapper.aggregating_mode
        if aggregating_mode is None or self.sliced_results is None:
            return

        aggregating_mode.set_parameters(self.sliced_results)

        aggregation_args = []
        if aggregator_path is not None:
            aggregator_path = check_location(get_absolute_path(aggregator_path))
            aggregation_args.append(aggregator_path)

        job_resources = JobResources(
            memory=memory,
            cpu_cores=aggregating_mode.cores,
            gpus=0,
            extra_properties=self.extra_properties,
        )

        jsi = JobSchedulingInformation(
            job_name=aggregating_mode.__class__.__name__,
            job_script_path=self.cluster_runner,
            job_resources=job_resources,
            job_script_arguments=aggregation_args,
            job_env=self.cluster_env,
            working_directory=self.working_directory,
            timeout=timedelta(seconds=AGGREGATION_TIME * len(self.sliced_results)),
            output_dir=self.output_file.parent if self.output_file else None,
            output_filename=self.output_file.name if self.output_file else None,
            log_directory=self.cluster_output_dir,
            timestamp=timestamp,
        )

        aggregation_scheduler = JobScheduler(
            url=self.url,
            partition=self.partition,
            user_name=self.user_name,
            user_token=self.user_token,
            cluster_output_dir=self.cluster_output_dir,
        )

        aggregation_jobs = self.program_wrapper.create_sliced_aggregating_jobs(
            job_scheduling_information=jsi,
            slice_params=self.sliced_results,
        )

        aggregation_success = aggregation_scheduler.run(aggregation_jobs)

        if not aggregation_success:
            aggregation_scheduler.resubmit_killed_jobs(allow_all_failed=True)

        if aggregation_success:
            self.aggregated_result = aggregation_scheduler.get_output_paths()[0]
            for result in self.sliced_results:
                os.remove(str(result))
        else:
            logging.warning(
                f"Aggregated job was unsuccessful with aggregating_mode: {aggregating_mode},"
                f" cluster_runner: {self.cluster_runner}, cluster_env: {self.cluster_env},"
                f" aggregator_path: {aggregator_path}, aggregation_args: {aggregation_args}"
            )
            self.aggregated_result = None
