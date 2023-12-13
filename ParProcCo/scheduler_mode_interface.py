from __future__ import annotations

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from datetime import datetime
    from .job_scheduling_information import JobSchedulingInformation


class SchedulerModeInterface:
    def __init__(self) -> None:
        self.allowed_modules: tuple[str, ...] | None = None

    def create_slice_jobs(
        self,
        slice_params: list[Any] | None,
        job_scheduling_information: JobSchedulingInformation,
    ) -> list[JobSchedulingInformation]:
        """For creating a list of new `JobSchedulingInformation`s based on the `slice_params` given"""
        raise NotImplementedError
