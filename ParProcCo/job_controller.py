from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from .job_scheduler import JobScheduler
from .slicer_interface import SlicerInterface
from .utils import check_location, get_absolute_path
from .program_wrapper import ProgramWrapper

class JobController:

    def __init__(self, program_wrapper: ProgramWrapper, output_dir_or_file: Path, project: str, queue: str, cluster_resources: Optional[dict[str,str]] = None, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""
        self.program_wrapper = program_wrapper
        self.output_file: Optional[Path] = None
        if output_dir_or_file.is_dir():
            output_dir = output_dir_or_file
        else:
            output_dir = output_dir_or_file.parent
            self.output_file = output_dir_or_file
        self.cluster_output_dir = check_location(output_dir)
        try:
            self.working_directory: Path = check_location(os.getcwd())
        except Exception:
            logging.warning(f"Could not use %s as working directory on cluster so using %s", os.getcwd(), self.cluster_output_dir, exc_info=True)
            self.working_directory = self.cluster_output_dir
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.cluster_resources = cluster_resources
        self.timeout = timeout
        self.sliced_results: Optional[List[Path]] = None
        self.aggregated_result: Optional[Path] = None

    def run(self, number_jobs: int, jobscript_args: Optional[List] = None,
            memory: str = "4G", job_name: str = "ParProcCo") -> None:

        self.cluster_runner = check_location(get_absolute_path(self.program_wrapper.get_cluster_runner_script()))
        self.cluster_env = self.program_wrapper.get_environment()
        slice_params = self.program_wrapper.create_slices(number_jobs)

        sliced_jobs_success = self._run_sliced_jobs(slice_params, jobscript_args, memory, job_name)

        if sliced_jobs_success:
            if number_jobs == 1:
                out_file = self.sliced_results[0]
            else:
                self._run_aggregation_job(memory)
                out_file = self.aggregated_result

            if self.output_file is not None:
                os.rename(out_file, self.output_file)
        else:
            raise RuntimeError("Sliced jobs failed\n")

    def _run_sliced_jobs(self, slice_params: List[slice],
                        jobscript_args: Optional[List], memory: str, job_name: str):
        if jobscript_args is None:
            jobscript_args = []

        processing_mode = self.program_wrapper.processing_mode
        processing_mode.set_parameters(slice_params)

        job_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                          self.cluster_resources, self.timeout)
        sliced_jobs_success = job_scheduler.run(processing_mode, self.cluster_runner, self.cluster_env, memory, processing_mode.cores, jobscript_args,
                                                     job_name)

        if not sliced_jobs_success:
            sliced_jobs_success = job_scheduler.rerun_killed_jobs()

        self.sliced_results = job_scheduler.get_output_paths() if sliced_jobs_success else None
        return sliced_jobs_success

    def _run_aggregation_job(self, memory: str) -> None:

        aggregator_path = self.program_wrapper.get_aggregate_script()
        aggregating_mode = self.program_wrapper.aggregating_mode
        if aggregating_mode is None:
            return

        aggregating_mode.set_parameters(self.sliced_results)

        aggregation_args = []
        if aggregator_path is not None:
            aggregator_path = check_location(get_absolute_path(aggregator_path))
            aggregation_args.append(aggregator_path)

        aggregation_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project,
                                                  self.queue, self.cluster_resources, self.timeout)
        aggregation_success = aggregation_scheduler.run(aggregating_mode, self.cluster_runner, self.cluster_env, memory, aggregating_mode.cores,
                                                             aggregation_args, aggregating_mode.__class__.__name__)

        if not aggregation_success:
            aggregation_scheduler.rerun_killed_jobs(allow_all_failed=True)

        if aggregation_success:
            self.aggregated_result = aggregation_scheduler.get_output_paths()[0]
            for result in self.sliced_results:
                os.remove(str(result))
        else:
            self.aggregated_result = None
