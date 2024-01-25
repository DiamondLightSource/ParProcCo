from __future__ import annotations

from pathlib import Path

from ParProcCo.program_wrapper import ProgramWrapper
from ParProcCo.simple_data_slicer import SimpleDataSlicer

from .simple_aggregation_slicer import SimpleAggregationSlicer
from .simple_processing_slicer import SimpleProcessingSlicer


class SimpleWrapper(ProgramWrapper):
    def __init__(self, processing_script: Path, aggregating_script: Path):
        super().__init__(
            SimpleProcessingSlicer(processing_script),
            SimpleDataSlicer(),
            SimpleAggregationSlicer(aggregating_script),
        )
