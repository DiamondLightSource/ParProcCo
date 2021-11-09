from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from .job_scheduler import JobScheduler
from .slicer_interface import SlicerInterface
from .utils import check_location, get_absolute_path
from .program_wrapper import ProgramWrapper

class JobController:

    def __init__(self, program_wrapper: ProgramWrapper, cluster_output_dir_name: str, project: str, queue: str, cluster_resources: Optional[dict[str,str]] = None, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""
        self.program_wrapper = program_wrapper
        self.working_directory: Path = check_location(os.getcwd())
        self.cluster_output_dir: Path = self.working_directory / cluster_output_dir_name
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.cluster_resources = cluster_resources
        self.timeout = timeout
        self.sliced_results: Optional[List[Path]] = None

    def run(self, number_jobs: int, cluster_runner: Path,
            jobscript_args: Optional[List] = None, aggregator_path: Optional[Path] = None,
            memory: str = "4G", cores: int = 6, job_name: str = "ParProcCo") -> None:

        slice_params = self.program_wrapper.create_slices(number_jobs)

        sliced_jobs_success = self.run_sliced_jobs(slice_params, cluster_runner, jobscript_args, memory, cores, job_name)

        if sliced_jobs_success:
            if number_jobs > 1:
                self.run_aggregation_job(cluster_runner, aggregator_path, memory, cores)
            # TODO rename if given filename
        else:
            raise RuntimeError("Sliced jobs failed\n")

    def run_sliced_jobs(self, slice_params: List[slice], processing_runner: Path,
                        jobscript_args: Optional[List], memory: str, cores: int, job_name: str):
        processing_runner = check_location(get_absolute_path(processing_runner))
        if jobscript_args is None:
            jobscript_args = []

        processing_mode = self.program_wrapper.processing_mode
        processing_mode.set_parameters(slice_params)

        job_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                          self.cluster_resources, self.timeout)
        sliced_jobs_success = job_scheduler.run(processing_mode, processing_runner, memory, cores, jobscript_args,
                                                     job_name)

        if not sliced_jobs_success:
            sliced_jobs_success = job_scheduler.rerun_killed_jobs()

        self.sliced_results = job_scheduler.get_output_paths() if sliced_jobs_success else None
        return sliced_jobs_success

    def run_aggregation_job(self, aggregation_runner: Path,
                            aggregator_path: Optional[Path], memory: str, cores: int) -> None:

        aggregating_mode = self.program_wrapper.aggregating_mode
        if aggregating_mode is None:
            return

        aggregating_mode.set_parameters(self.sliced_results)

        aggregation_runner = check_location(get_absolute_path(aggregation_runner))
        aggregation_args = []
        if aggregator_path is not None:
            aggregator_path = check_location(get_absolute_path(aggregator_path))
            aggregation_args.append(aggregator_path)

        aggregation_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project,
                                                  self.queue, self.cluster_resources, self.timeout)
        aggregation_success = aggregation_scheduler.run(aggregating_mode, aggregation_runner, memory, cores,
                                                             aggregation_args, aggregating_mode.__class__.__name__)

        if not aggregation_success:
            aggregation_scheduler.rerun_killed_jobs(allow_all_failed=True)

        if aggregation_success:
            for result in self.sliced_results:
                os.remove(str(result))
