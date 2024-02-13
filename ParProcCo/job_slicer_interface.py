from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from ParProcCo.utils import (
    check_jobscript_is_readable,
    check_location,
    get_absolute_path,
)

if TYPE_CHECKING:
    from typing import Any

    from .job_scheduling_information import JobSchedulingInformation


class JobSlicerInterface:
    def __init__(self, job_script: Path | None = None) -> None:
        if job_script is not None:
            self.job_script = check_jobscript_is_readable(
                check_location(get_absolute_path(job_script))
            )
        else:
            self.job_script = Path()  # set this to the current working directory
        self.allowed_modules: tuple[str, ...] | None = None

    def create_slice_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any] | None,
    ) -> list[JobSchedulingInformation]:
        """For creating a list of new `JobSchedulingInformation`s based on
        given `slice_params`"""
        if slice_params is None:
            slice_params = [None]

        number_of_jobs = len(slice_params)
        return [
            self.create_slice_job(
                i=i,
                job_scheduling_information=deepcopy(job_scheduling_information),
                slice_params=slice_params,
            )
            for i in range(number_of_jobs)
        ]

    def create_slice_job(
        self,
        i: int,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any],
    ) -> JobSchedulingInformation:
        """This mutates the JSI according to the job number and slice parameters.

        This must be implemented if create_slice_jobs has not been overridden."""
        raise NotImplementedError
