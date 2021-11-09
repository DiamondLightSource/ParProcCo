from __future__ import annotations

from typing import List, Optional

from pathlib import Path

from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface

class ProgramWrapper:

    def __init__(self, processing_mode: SchedulerModeInterface, slicer : Optional[SlicerInterface] = None, aggregating_mode: Optional[SchedulerModeInterface] = None):
        self.processing_mode = processing_mode
        self.slicer = slicer
        self.aggregating_mode = aggregating_mode
        import os
        current_script_dir = Path(os.path.realpath(__file__)).parent.parent / "scripts"
        self.cluster_runner_path = current_script_dir / "msm_cluster_runner"
        self.agg_script_path = current_script_dir / "nxdata_aggregate"

    def create_slices(self, number_jobs: int, stop: Optional[int] = None) -> List[slice]:
        if number_jobs == 1 or self.slicer is None:
            return [None]
        return self.slicer.slice(number_jobs, stop)

    def get_output(self, output: str, _program_args: Optional[List[str]]) -> Path:
        return output

    def get_aggregate_script(self) -> str:
        return str(self.agg_script_path)

    def get_cluster_runner_script(self) -> str:
        return str(self.cluster_runner_path)
