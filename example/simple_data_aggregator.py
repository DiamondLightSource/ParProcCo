from __future__ import annotations

from pathlib import Path

from ParProcCo.aggregator_interface import AggregatorInterface


class SimpleDataAggregator(AggregatorInterface):
    def __init__(self) -> None:
        pass

    def aggregate(self, aggregation_output: Path, data_files: list[Path]) -> Path:
        """Overrides AggregatorInterface.aggregate"""
        aggregated_lines = []
        for data_file in data_files:
            with open(data_file) as f:
                for line in f.readlines():
                    aggregated_lines.append(line)

        with open(aggregation_output, "a") as af:
            af.writelines(aggregated_lines)

        return aggregation_output
