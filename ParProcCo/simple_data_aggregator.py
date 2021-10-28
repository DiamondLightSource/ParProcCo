from __future__ import annotations

from pathlib import Path
from typing import List

from ParProcCo.aggregator_interface import AggregatorInterface


class SimpleDataAggregator(AggregatorInterface):

    def __init__(self) -> None:
        pass

    def aggregate(self, aggregation_output: Path, output_data_files: List[Path]) -> Path:
        """Overrides AggregatorInterface.aggregate"""
        aggregated_lines = []
        for output_file in output_data_files:
            with open(output_file) as f:
                for line in f.readlines():
                    aggregated_lines.append(line)

        with open(aggregation_output, "a") as af:
            af.writelines(aggregated_lines)

        return aggregation_output
