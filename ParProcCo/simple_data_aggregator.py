from __future__ import annotations

from pathlib import Path
from typing import List

from ParProcCo.aggregator_interface import AggregatorInterface


class SimpleDataAggregator(AggregatorInterface):

    def __init__(self) -> None:
        pass

    def aggregate(self, total_slices: int, aggregation_output: Path, output_data_files: List[Path]) -> Path:
        """Overrides AggregatorInterface.aggregate"""
        if type(total_slices) is int:
            self.total_slices = total_slices
        else:
            raise TypeError(f"total_slices is {type(total_slices)}, should be int\n")

        if len(output_data_files) != self.total_slices:
            raise ValueError(
                f"Number of output files {len(output_data_files)} must equal total_slices {self.total_slices}")

        aggregated_lines = []
        for output_file in output_data_files:
            with open(output_file) as f:
                for line in f.readlines():
                    aggregated_lines.append(line)

        with open(aggregation_output, "a") as af:
            af.writelines(aggregated_lines)

        return aggregation_output
