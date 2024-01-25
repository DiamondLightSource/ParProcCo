from __future__ import annotations

import logging
import unittest
from pathlib import Path

from example.simple_data_aggregator import SimpleDataAggregator
from ParProcCo.test import TemporaryDirectory

from .utils import setup_aggregator_data_files


class TestDataAggregator(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

    def test_aggregate_data(self) -> None:
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.is_dir():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            aggregation_output = cluster_output_dir / "aggregated_results.txt"
            sliced_data_files = setup_aggregator_data_files(cluster_output_dir)
            written_data = []
            for data_file in sliced_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(
                written_data,
                [["0\n", "8\n"], ["2\n", "10\n"], ["4\n", "12\n"], ["6\n", "14\n"]],
            )

            aggregator = SimpleDataAggregator()
            agg_data_path = aggregator.aggregate(aggregation_output, sliced_data_files)
            self.assertEqual(agg_data_path, aggregation_output)
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(
                agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"]
            )


if __name__ == "__main__":
    unittest.main()
