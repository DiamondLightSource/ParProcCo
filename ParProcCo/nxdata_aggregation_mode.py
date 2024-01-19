from __future__ import annotations

from copy import deepcopy
from ParProcCo.job_scheduling_information import JobSchedulingInformation

from ParProcCo.job_slicer_interface import JobSlicerInterface
from ParProcCo.utils import (
    check_jobscript_is_readable,
    check_location,
    format_timestamp,
    get_absolute_path,
)


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Any


class NXdataAggregationMode(JobSlicerInterface):
    def __init__(self):
        self.allowed_modules = ("python",)

    def create_slice_jobs(
        self,
        slice_params: list[Any] | None,
        job_scheduling_information: JobSchedulingInformation,
    ) -> list[JobSchedulingInformation]:
        """Overrides JobSlicerInterface.create_slice_jobs"""
        if slice_params is None:
            return []
        return [
            self.create_slice_job(
                slice_params=slice_params,
                job_scheduling_information=deepcopy(job_scheduling_information),
            )
        ]

    def create_slice_job(
        self,
        slice_params: list[Any],
        job_scheduling_information: JobSchedulingInformation,
    ) -> JobSchedulingInformation:
        timestamp = format_timestamp(job_scheduling_information.timestamp)
        job_scheduling_information.output_filename = (
            f"aggregated_results_{timestamp}.nxs"
        )
        job_scheduling_information.stdout_filename = f"out_{timestamp}_aggregated"
        job_scheduling_information.stderr_filename = f"err_{timestamp}_aggregated"

        job_script = str(
            check_jobscript_is_readable(
                check_location(
                    get_absolute_path(
                        job_scheduling_information.job_script_arguments[0]
                    )
                )
            )
        )
        job_scheduling_information.job_script_arguments = tuple(
            [job_script, "--output", job_scheduling_information.get_output_path()]
            + [str(x) for x in slice_params]
        )
        return job_scheduling_information
