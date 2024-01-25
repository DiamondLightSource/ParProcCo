from __future__ import annotations

from typing import TYPE_CHECKING

from .job_slicer_interface import JobSlicerInterface
from .program_wrapper import ProgramWrapper
from .utils import (check_jobscript_is_readable, check_location,
                    format_timestamp, get_absolute_path)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from .job_scheduling_information import JobSchedulingInformation


class PassThruProcessingSlicer(JobSlicerInterface):
    def create_slice_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any] | None,
    ) -> list[JobSchedulingInformation]:
        """Overrides JobSlicerInterface.create_slice_jobs"""

        timestamp = format_timestamp(job_scheduling_information.timestamp)
        job_scheduling_information.stdout_filename = f"out_{timestamp}"
        job_scheduling_information.stderr_filename = f"err_{timestamp}"
        job_script = str(
            check_jobscript_is_readable(
                check_location(
                    get_absolute_path(
                        job_scheduling_information.job_script_arguments[0]
                    )
                )
            )
        )

        args = [
            job_script,
            "--memory",
            str(job_scheduling_information.job_resources.memory),
            "--cores",
            str(job_scheduling_information.job_resources.cpu_cores),
        ]
        if job_scheduling_information.output_filename:
            args += ("--output", job_scheduling_information.output_filename)
        args += job_scheduling_information.job_script_arguments[1:]
        job_scheduling_information.job_script_arguments = args
        return [job_scheduling_information]


class PassThruWrapper(ProgramWrapper):
    def __init__(self, original_wrapper: ProgramWrapper):
        super().__init__(processing_slicer=PassThruProcessingSlicer())
        self.original_wrapper = original_wrapper
        self.processing_mode.allowed_modules = (
            original_wrapper.processing_mode.allowed_modules
        )

    def get_args(self, args: list[str], debug: bool = False):
        return self.original_wrapper.get_args(args, debug)

    def get_output(
        self, output: str | None, program_args: list[str] | None
    ) -> Path | None:
        return self.original_wrapper.get_output(output, program_args)

    def create_slices(self, number_jobs: int, stop: int | None = None) -> None:
        return None
