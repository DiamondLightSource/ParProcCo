from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from ParProcCo.job_slicer_interface import JobSlicerInterface
from ParProcCo.utils import (
    slice_to_string,
    check_jobscript_is_readable,
    check_location,
    format_timestamp,
    get_absolute_path,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from ParProcCo.job_scheduling_information import JobSchedulingInformation


class SimpleProcessingMode(JobSlicerInterface):
    def __init__(self, job_script: Path) -> None:
        self.job_script = check_jobscript_is_readable(
            check_location(get_absolute_path(job_script))
        )
        self.allowed_modules = ("python",)

    def create_slice_jobs(
        self,
        slice_params: list[Any] | None,
        job_scheduling_information: JobSchedulingInformation,
    ) -> list[JobSchedulingInformation]:
        """A basic implementation of create_slice_jobs"""
        if slice_params is None:
            return [deepcopy(job_scheduling_information)]
        number_of_jobs = len(slice_params)
        return [
            self.create_slice_job(
                i=i,
                slice_params=slice_params,
                job_scheduling_information=deepcopy(job_scheduling_information),
            )
            for i in range(number_of_jobs)
        ]

    def create_slice_job(
        self,
        i: int,
        slice_params: list[slice],
        job_scheduling_information: JobSchedulingInformation,
    ) -> JobSchedulingInformation:
        # Output paths:
        timestamp = format_timestamp(job_scheduling_information.timestamp)
        job_scheduling_information.output_filename = f"out_{i}"
        job_scheduling_information.stdout_filename = f"out_{timestamp}_{i}"
        job_scheduling_information.sterr_filename = f"err_{timestamp}_{i}"

        # Arguments:
        slice_param = slice_to_string(slice_params[i])
        job_script = check_jobscript_is_readable(
            check_location(
                get_absolute_path(job_scheduling_information.job_script_path)
            )
        )
        job_scheduling_information.job_script_arguments = tuple(
            [
                str(job_script),
                "--memory",
                str(job_scheduling_information.job_resources.memory),
                "--cores",
                str(job_scheduling_information.job_resources.cpu_cores),
                "--output",
                str(job_scheduling_information.get_output_path()),
                "--images",
                slice_param,
            ]
            + job_scheduling_information.job_script_arguments
        )

        job_scheduling_information.job_script_path = self.job_script

        return job_scheduling_information
