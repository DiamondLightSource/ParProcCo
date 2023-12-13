from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .program_wrapper import ProgramWrapper
from .scheduler_mode_interface import SchedulerModeInterface
from .utils import (
    check_jobscript_is_readable,
    check_location,
    format_timestamp,
    get_absolute_path,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from .job_scheduling_information import JobSchedulingInformation


class PassThruProcessingMode(SchedulerModeInterface):
    def create_slice_jobs(
        self,
        slice_params: list[Any] | None,
        job_scheduling_information: JobSchedulingInformation,
    ) -> list[JobSchedulingInformation]:
        """Overrides SchedulerModeInterface.create_slice_jobs"""

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
            str(job_scheduling_information.job_resources.cores),
        ]
        if job_scheduling_information.output_filename:
            args += ("--output", job_scheduling_information.output_filename)
        args += job_scheduling_information.job_script_arguments[1:]
        job_scheduling_information.job_script_arguments = args


class PassThruWrapper(ProgramWrapper):
    def __init__(self, original_wrapper: ProgramWrapper):
        super().__init__(PassThruProcessingMode())
        self.original_wrapper = original_wrapper
        self.processing_mode.allowed_modules = (
            original_wrapper.processing_mode.allowed_modules
        )

    def create_slices(self, number_jobs: int, stop: int | None = None) -> None:
        return None

    def create_sliced_processing_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[slice] | None,
    ) -> list[JobSchedulingInformation]:
        return self.processing_mode.create_slice_jobs(
            slice_params=slice_params,
            job_scheduling_information=job_scheduling_information,
        )

    def create_sliced_aggregating_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Path] | None,
    ) -> list[JobSchedulingInformation]:
        return []
