from __future__ import annotations

from typing import TYPE_CHECKING

from .job_slicer_interface import JobSlicerInterface
from .program_wrapper import ProgramWrapper
from .utils import (
    check_jobscript_is_readable,
    check_location,
    format_timestamp,
    get_absolute_path,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from .job_scheduling_information import JobSchedulingInformation


class PassThruProcessingSlicer(JobSlicerInterface):
    def __init__(self):
        super().__init__("ppc_cluster_runner")

    def create_slice_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any] | None,
    ) -> list[JobSchedulingInformation]:
        """Overrides JobSlicerInterface.create_slice_jobs"""

        assert job_scheduling_information.timestamp is not None
        timestamp = format_timestamp(job_scheduling_information.timestamp)
        job_scheduling_information.stdout_filename = f"out_{timestamp}"
        job_scheduling_information.stderr_filename = f"err_{timestamp}"
        old_args = job_scheduling_information.job_script_arguments
        job_script = str(
            check_jobscript_is_readable(check_location(get_absolute_path(old_args[0])))
        )

        args: tuple[str, ...] = (
            job_script,
            "--memory",
            f"{job_scheduling_information.job_resources.memory}M",
            "--cores",
            str(job_scheduling_information.job_resources.cpu_cores),
        )
        output_path = job_scheduling_information.get_output_path()
        if output_path is not None:
            args += ("--output", str(output_path))
        if len(old_args) > 1:
            args += old_args[1:]
        job_scheduling_information.job_script_arguments = args
        return [job_scheduling_information]


class PassThruWrapper(ProgramWrapper):
    def __init__(self, original_wrapper: ProgramWrapper):
        super().__init__(processing_slicer=PassThruProcessingSlicer())
        self.original_wrapper = original_wrapper
        self.processing_slicer.allowed_modules = (
            original_wrapper.processing_slicer.allowed_modules
        )

    def get_args(self, args: list[str], debug: bool = False):
        return self.original_wrapper.get_args(args, debug)

    def get_output(
        self, output: str | None, program_args: list[str] | None
    ) -> Path | None:
        return self.original_wrapper.get_output(output, program_args)

    def create_slices(self, number_jobs: int, stop: int | None = None) -> None:
        return None
