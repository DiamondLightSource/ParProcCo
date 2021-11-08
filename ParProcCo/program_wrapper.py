from __future__ import annotations

from typing import List, Optional

from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface

class ProgramWrapper:

    def __init__(self, slicer : SlicerInterface, processing_mode: SchedulerModeInterface, aggregating_mode: Optional[SchedulerModeInterface] = None):
        self.slicer = slicer
        self.processing_mode = processing_mode
        self.aggregating_mode = aggregating_mode

    def create_slices(self, number_jobs: int, stop: Optional[int] = None) -> List[slice]:
        return self.slicer.slice(number_jobs, stop)

