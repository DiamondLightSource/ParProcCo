from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.job_scheduler import JobScheduler
from ParProcCo.slicer_interface import SlicerInterface
from ParProcCo.utils import check_location, get_absolute_path


class JobController:

    def __init__(self, cluster_output_dir_name: str, project: str, queue: str, cluster_resources: Optional[dict[str,str]] = None, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""

        self.working_directory: Path = check_location(os.getcwd())
        self.aggregation_scheduler: JobScheduler
        self.cluster_output_dir: Path = self.working_directory / cluster_output_dir_name
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.cluster_resources = cluster_resources
        self.job_scheduler: JobScheduler
        self.timeout = timeout
        self.scheduler: JobScheduler

    def run(self, data_slicer: SlicerInterface, number_jobs: int, processing_mode: SchedulerModeInterface,
            aggregating_mode: SchedulerModeInterface, processing_script: Path, aggregation_script: Path,
            jobscript_args: Optional[List] = None, aggregation_args: Optional[List] = None, memory: str = "4G", cores: int = 6,
            job_name: str = "ParProcCo") -> None:

        slice_params = self.create_slices(data_slicer, number_jobs)

        self.run_sliced_jobs(processing_mode, number_jobs, slice_params, processing_script, jobscript_args, memory, cores, job_name)

        self.run_aggregation_job(aggregating_mode, aggregation_script, aggregation_args, number_jobs, memory, cores, job_name)

    def create_slices(self, data_slicer: SlicerInterface, number_jobs: int) -> List[slice]:
        self.data_slicer = data_slicer
        slice_params = self.data_slicer.slice(number_jobs)
        return slice_params

    def run_sliced_jobs(self, processing_mode, number_jobs: int, slice_params: List[slice], processing_script: Path,
                        jobscript_args: Optional[List], memory: str, cores: int, job_name: str):
        processing_script = check_location(get_absolute_path(processing_script))
        if jobscript_args is None:
            jobscript_args = []

        processing_mode.set_parameters(slice_params, number_jobs)

        self.job_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                          self.cluster_resources, self.timeout)
        sliced_jobs_success = self.job_scheduler.run(processing_mode, processing_script, memory, cores, jobscript_args,
                                                     job_name)

        if not sliced_jobs_success:
            self.job_scheduler.rerun_killed_jobs(processing_mode, processing_script, memory, cores, jobscript_args,
                                                 job_name)

    def run_aggregation_job(self, aggregating_mode: SchedulerModeInterface, aggregation_script: Path,
                            aggregation_args: Optional[List], number_jobs: int, memory: str, cores: int,
                            job_name: str) -> None:

        aggregation_script = check_location(get_absolute_path(aggregation_script))
        if aggregation_args is None:
            aggregation_args = []
        sliced_results = self.job_scheduler.get_output_paths()

        aggregating_mode.set_parameters(sliced_results, number_jobs)

        self.aggregation_scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project,
                                                  self.queue, self.timeout)
        aggregation_success = self.aggregation_scheduler.run(aggregating_mode, aggregation_script, memory, cores,
                                                             aggregation_args, job_name)

        if not aggregation_success:
            self.aggregation_scheduler.rerun_killed_jobs(aggregating_mode, aggregation_script, memory, cores,
                                                         aggregation_args, job_name, allow_all_failed=True)
