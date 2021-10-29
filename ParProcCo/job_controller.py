from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from ParProcCo.aggregator_interface import AggregatorInterface
from ParProcCo.job_scheduler import JobScheduler
from ParProcCo.slicer_interface import SlicerInterface
from ParProcCo.utils import check_location, get_absolute_path


class JobController:

    def __init__(self, cluster_output_dir_name: str, project: str, queue: str, cluster_resources: Optional[dict[str,str]] = None, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""

        self.working_directory: Path = check_location(os.getcwd())

        self.cluster_output_dir: Path = self.working_directory / cluster_output_dir_name
        self.data_aggregator: AggregatorInterface
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.cluster_resources = cluster_resources
        self.timeout = timeout
        self.scheduler: JobScheduler

    def run(self, data_slicer: SlicerInterface, data_aggregator: AggregatorInterface, number_jobs: int,
            processing_script: Path, memory: str = "4G", cores: int = 6, jobscript_args: Optional[List] = None,
            job_name: str = "ParProcCo") -> Path:
        self.data_slicer = data_slicer
        self.data_aggregator = data_aggregator
        processing_script = check_location(get_absolute_path(processing_script))
        slice_params = self.data_slicer.slice(number_jobs)
        if jobscript_args is None:
            jobscript_args = []
        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                      self.cluster_resources, self.timeout)
        success = self.scheduler.run(processing_script, slice_params, memory, cores, jobscript_args, job_name)
        if not success:
            self.scheduler.rerun_killed_jobs(processing_script)
        aggregated_file_path = self.aggregate_data(number_jobs)
        return aggregated_file_path

    def aggregate_data(self, number_jobs: int) -> Path:
        if self.scheduler.get_success():
            sliced_results = self.scheduler.get_output_paths()
            aggregated_file_path = self.data_aggregator.aggregate(number_jobs, self.cluster_output_dir, sliced_results)
            return aggregated_file_path
        else:
            raise RuntimeError(f"Not all jobs were successful. Aggregation not performed\n")
