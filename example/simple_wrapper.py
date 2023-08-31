from __future__ import annotations

from pathlib import Path

from ParProcCo.program_wrapper import ProgramWrapper
from ParProcCo.simple_data_slicer import SimpleDataSlicer

from .simple_aggregation_mode import SimpleAggregationMode
from .simple_processing_mode import SimpleProcessingMode


class SimpleWrapper(ProgramWrapper):
    def __init__(self, processing_script: Path, aggregating_script: Path):
        super().__init__(
            SimpleProcessingMode(processing_script), SimpleDataSlicer(), SimpleAggregationMode(aggregating_script)
        )
