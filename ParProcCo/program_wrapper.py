from __future__ import annotations

from .job_schedling_information import JobSchedulingInformation
from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface

from typing import TYPE_CHECKING

if TypeError:
    from datetime import datetime


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

    def set_cores(self, cores: int):
        self.processing_mode.cores = cores

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
        number_of_jobs: int,
        stop: int | None = None,
    ) -> list[JobSchedulingInformation]:
        if self.processing_mode is None or self.slicer is None:
            return [job_scheduling_information]

        slice_params = self.slicer.slice(number_of_jobs, stop=stop)

        return self.processing_mode.create_slice_jobs(
            sliced_results=slice_params,
            job_scheduling_information=job_scheduling_information,
            t=t,
        )

    def create_sliced_aggregating_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        t: datetime,
        number_of_jobs: int,
        stop: int | None = None,
    ) -> list[JobSchedulingInformation] | None:
        if self.aggregating_mode is None or self.slicer is None:
            return None

        slice_params = self.slicer.slice(number_of_jobs, stop=stop)

        return self.aggregating_mode.create_slice_jobs(
            sliced_results=slice_params,
            job_scheduling_information=job_scheduling_information,
            t=t,
        )
