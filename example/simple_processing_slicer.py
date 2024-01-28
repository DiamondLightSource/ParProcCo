from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ParProcCo.job_slicer_interface import JobSlicerInterface
from ParProcCo.utils import format_timestamp, slice_to_string

if TYPE_CHECKING:
    from ParProcCo.job_scheduling_information import JobSchedulingInformation


class SimpleProcessingSlicer(JobSlicerInterface):
    def __init__(self, job_script: Path) -> None:
        super().__init__(job_script)
        self.allowed_modules = ("python",)

    def create_slice_job(
        self,
        i: int,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[slice],
    ) -> JobSchedulingInformation:
        # Output paths:
        assert job_scheduling_information.timestamp is not None
        timestamp = format_timestamp(job_scheduling_information.timestamp)
        job_scheduling_information.output_filename = f"out_{i}"
        job_scheduling_information.stdout_filename = f"out_{timestamp}_{i}"
        job_scheduling_information.stderr_filename = f"err_{timestamp}_{i}"

        # Arguments:
        slice_param = slice_to_string(slice_params[i])
        old_args = job_scheduling_information.job_script_arguments

        job_scheduling_information.job_script_arguments = (
            (
                old_args[0],
                "--memory",
                f"{job_scheduling_information.job_resources.memory}M",
                "--cores",
                str(job_scheduling_information.job_resources.cpu_cores),
                "--output",
                str(job_scheduling_information.get_output_path()),
                "--images",
                slice_param,
            )
            + old_args[1:]
            if len(old_args) > 0
            else ()
        )

        return job_scheduling_information
