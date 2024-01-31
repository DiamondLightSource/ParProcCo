from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ParProcCo.job_scheduling_information import JobSchedulingInformation
from ParProcCo.job_slicer_interface import JobSlicerInterface
from ParProcCo.utils import (
    check_jobscript_is_readable,
    check_location,
    format_timestamp,
    get_absolute_path,
)

if TYPE_CHECKING:
    from typing import Any


class NXdataAggregationSlicer(JobSlicerInterface):
    def __init__(self):
        super().__init__("nxdata_aggregate")
        self.allowed_modules = ("python",)

    def create_slice_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any] | None,
    ) -> list[JobSchedulingInformation]:
        """Overrides JobSlicerInterface.create_slice_jobs"""
        if slice_params is None:
            return []

        jsi = deepcopy(job_scheduling_information)
        assert jsi.timestamp is not None
        timestamp = format_timestamp(jsi.timestamp)

        jsi.output_filename = f"aggregated_results_{timestamp}.nxs"
        jsi.stdout_filename = f"out_{timestamp}_aggregated"
        jsi.stderr_filename = f"err_{timestamp}_aggregated"

        job_script = str(
            check_jobscript_is_readable(
                check_location(get_absolute_path(jsi.job_script_arguments[0]))
            )
        )
        jsi.job_script_arguments = tuple(
            [job_script, "--output", str(jsi.get_output_path())]
            + [str(x) for x in slice_params]
        )
        return [jsi]
