from __future__ import annotations

from pathlib import Path


class AggregatorInterface:
    def aggregate(self, aggregation_output_dir: Path, data_files: list[Path]) -> Path:
        """Aggregates data from multiple output files into one"""
        raise NotImplementedError
