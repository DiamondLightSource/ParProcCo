from __future__ import annotations

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from .job_schedling_information import JobSchedulingInformation


class SchedulerModeInterface:
    def __init__(self) -> None:
        self.allowed_modules: tuple[str, ...] | None = None

    def create_slice_jobs(
        self,
        sliced_results: list[slice] | None,
        job_scheduling_information: JobSchedulingInformation,
        t: datetime,
    ) -> list[JobSchedulingInformation]:
        if sliced_results is None:
            return [deepcopy(job_scheduling_information)]
        return [
            self.create_slice_job(
                i=i,
                slice_result=res,
                job_scheduling_information=deepcopy(job_scheduling_information),
                t=t,
            )
            for i, res in enumerate(sliced_results)
        ]

    def create_slice_job(
        self,
        i: int,
        slice_result: slice,
        job_scheduling_information: JobSchedulingInformation,
        t: datetime,
    ) -> JobSchedulingInformation:
        raise NotImplementedError
