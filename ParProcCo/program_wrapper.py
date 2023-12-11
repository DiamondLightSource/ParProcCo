from __future__ import annotations

from .job_scheduling_information import JobSchedulingInformation
from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


class ProgramWrapper:
    def __init__(
        self,
        processing_mode: SchedulerModeInterface | None = None,
        slicer: SlicerInterface | None = None,
        aggregating_mode: SchedulerModeInterface | None = None,
    ):
        self.processing_mode = processing_mode
        self.slicer = slicer
        self.aggregating_mode = aggregating_mode

    def create_slices(
        self, number_jobs: int, stop: int | None = None
    ) -> list[slice] | None:
        if number_jobs == 1 or self.slicer is None:
            return None
        return self.slicer.slice(number_jobs, stop)

    def create_sliced_processing_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        t: datetime,
        slice_params: list[slice] | None,
    ) -> list[JobSchedulingInformation]:
        if self.processing_mode is None or slice_params is None:
            return [job_scheduling_information]

        return self.processing_mode.create_slice_jobs(
            slice_params=slice_params,
            job_scheduling_information=job_scheduling_information,
            t=t,
        )

    def create_sliced_aggregating_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        t: datetime,
        slice_params: list[Path] | None,
    ) -> list[JobSchedulingInformation] | None:
        if self.aggregating_mode is None or slice_params is None:
            return None

        return self.aggregating_mode.create_slice_jobs(
            slice_params=slice_params,
            job_scheduling_information=job_scheduling_information,
            t=t,
        )
