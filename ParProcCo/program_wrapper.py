from __future__ import annotations

from typing import Dict, List, Optional

from pathlib import Path

from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface

import os

class ProgramWrapper:

    def __init__(self, processing_mode: SchedulerModeInterface, slicer : Optional[SlicerInterface] = None, aggregating_mode: Optional[SchedulerModeInterface] = None):
        self.processing_mode = processing_mode
        self.slicer = slicer
        self.aggregating_mode = aggregating_mode

    def set_cores(self, cores: int):
        self.processing_mode.cores = cores

    def create_slices(self, number_jobs: int, stop: Optional[int] = None) -> List[slice]:
        if number_jobs == 1 or self.slicer is None:
            return [None]
        return self.slicer.slice(number_jobs, stop)

    def get_output(self, output: str, _program_args: Optional[List[str]]) -> Path:
        return Path(output)

    def get_aggregate_script(self) -> Optional[Path]:
        return self.aggregating_mode.program_path if self.aggregating_mode else None

    def get_cluster_runner_script(self) -> Path:
        return self.processing_mode.program_path

    def get_environment(self) -> Dict[str,str]:
        test_modules = os.getenv('TEST_PPC_MODULES')
        if test_modules:
            return {"PPC_MODULES":test_modules}
        return self.processing_mode.environment
