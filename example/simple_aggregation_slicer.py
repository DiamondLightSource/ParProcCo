from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from ParProcCo.job_slicer_interface import JobSlicerInterface
from ParProcCo.utils import format_timestamp

if TYPE_CHECKING:
    from typing import Any

    from ParProcCo.job_scheduling_information import JobSchedulingInformation


class SimpleAggregationSlicer(JobSlicerInterface):
    def __init__(self, job_script: Path) -> None:
        super().__init__(job_script)

    def create_slice_jobs(
        self,
        job_scheduling_information: JobSchedulingInformation,
        slice_params: list[Any] | None,
    ) -> list[JobSchedulingInformation]:
        """A basic implementation of create_slice_jobs"""
        if slice_params is None:
            return []

        jsi = deepcopy(job_scheduling_information)
        assert jsi.timestamp is not None
        timestamp = format_timestamp(jsi.timestamp)

        jsi.output_filename = f"aggregated_results_{timestamp}.nxs"
        jsi.stdout_filename = f"out_{timestamp}_aggregated"
        jsi.stderr_filename = f"err_{timestamp}_aggregated"
        jsi.job_script_arguments = tuple(
            [str(self.job_script), "--output", str(jsi.get_output_path())]
            + [str(x) for x in slice_params]
        )

        return [jsi]
