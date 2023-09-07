from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from .job_scheduler import JobScheduler
from .slicer_interface import SlicerInterface
from .utils import check_location, get_absolute_path, get_slurm_token, get_user
from .program_wrapper import ProgramWrapper


class JobController:
    def __init__(
        self,
        url: str,
        program_wrapper: ProgramWrapper,
        output_dir_or_file: Path,
        partition: str,
        version: str = "v0.0.38",
        user_name: Optional[str] = None,
        user_token: Optional[str] = None,
        timeout: timedelta = timedelta(hours=2),
    ) -> None:
        """JobController is used to coordinate cluster job submissions with JobScheduler"""
        self.url = url
        self.program_wrapper = program_wrapper
        self.partition = partition
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
        self.version = version
        self.user_name = user_name if user_name else get_user()
        self.user_token = user_token if user_token else get_slurm_token()
        self.timeout = timeout
        self.sliced_results: Optional[List[Path]] = None
        self.aggregated_result: Optional[Path] = None

    def run(
        self,
        number_jobs: int,
        jobscript_args: Optional[List] = None,
        memory: int = 4000,
        job_name: str = "ParProcCo",
    ) -> None:
        self.cluster_runner = check_location(
            get_absolute_path(self.program_wrapper.get_cluster_runner_script())
        )
        self.cluster_env = self.program_wrapper.get_environment()
        logging.debug("Cluster environment is %s", self.cluster_env)
        slice_params = self.program_wrapper.create_slices(number_jobs)

        sliced_jobs_success = self._submit_sliced_jobs(
            slice_params, jobscript_args, memory, job_name
        )

        if sliced_jobs_success and self.sliced_results:
            logging.info("Sliced jobs ran successfully.")
            if number_jobs == 1:
                out_file = (
                    self.sliced_results[0] if len(self.sliced_results) > 0 else None
                )
            else:
                self._submit_aggregation_job(memory)
                out_file = self.aggregated_result

            if (
                out_file is not None
                and out_file.is_file()
                and self.output_file is not None
            ):
                out_file.rename(self.output_file)
        else:
            logging.error(
                f"Sliced jobs failed with slice_params: {slice_params}, jobscript_args: {jobscript_args},"
                f" memory: {memory}, job_name: {job_name}"
            )
            raise RuntimeError("Sliced jobs failed\n")

    def _submit_sliced_jobs(
        self,
        slice_params: List[Optional[slice]],
        jobscript_args: Optional[List],
        memory: int,
        job_name: str,
    ) -> bool:
        if jobscript_args is None:
            jobscript_args = []

        processing_mode = self.program_wrapper.processing_mode
        processing_mode.set_parameters(slice_params)

        job_scheduler = JobScheduler(
            self.url,
            self.working_directory,
            self.cluster_output_dir,
            self.partition,
            self.timeout,
            self.version,
            self.user_name,
            self.user_token,
        )
        sliced_jobs_success = job_scheduler.run(
            processing_mode,
            self.cluster_runner,
            self.cluster_env,
            memory,
            processing_mode.cores,
            jobscript_args,
            job_name,
        )

        if not sliced_jobs_success:
            sliced_jobs_success = job_scheduler.resubmit_killed_jobs()

        self.sliced_results = (
            job_scheduler.get_output_paths() if sliced_jobs_success else None
        )
        return sliced_jobs_success

    def _submit_aggregation_job(self, memory: int) -> None:
        aggregator_path = self.program_wrapper.get_aggregate_script()
        aggregating_mode = self.program_wrapper.aggregating_mode
        if aggregating_mode is None or self.sliced_results is None:
            return

        aggregating_mode.set_parameters(self.sliced_results)

        aggregation_args = []
        if aggregator_path is not None:
            aggregator_path = check_location(get_absolute_path(aggregator_path))
            aggregation_args.append(aggregator_path)

        aggregation_scheduler = JobScheduler(
            self.url,
            self.working_directory,
            self.cluster_output_dir,
            self.partition,
            self.timeout,
            self.version,
            self.user_name,
            self.user_token,
        )
        aggregation_success = aggregation_scheduler.run(
            aggregating_mode,
            self.cluster_runner,
            self.cluster_env,
            memory,
            aggregating_mode.cores,
            aggregation_args,
            aggregating_mode.__class__.__name__,
        )
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
