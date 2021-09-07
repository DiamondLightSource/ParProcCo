from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import h5py
import numpy as np
from msmapper_aggregator import MSMAggregator


def setup_data_files(working_directory: Path) -> List[Path]:
    # create test files
    file_paths = [Path(working_directory) / f"file_0{i}.txt" for i in range(4)]
    file_contents = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    for file_path, content in zip(file_paths, file_contents):
        with open(file_path, "w") as f:
            f.write(content)
    return file_paths


class TestDataSlicer(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_renormalise(self) -> None:
        output_file_paths = [Path("/scratch/victoria/i07-394487-applied-halfa.nxs"),
                             Path("/scratch/victoria/i07-394487-applied-halfb.nxs")]
        aggregator = MSMAggregator()
        aggregator._check_total_slices(2, output_file_paths)
        aggregator._renormalise(output_file_paths)
        total_volume = aggregator.total_volume
        total_weights = aggregator.accumulator_weights
        with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
            weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
        np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
        np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

    def test_check_total_slices_not_int(self) -> None:
        total_slices = "1"
        output_data_files = [Path("file/path/a"), Path("file/path/b")]
        aggregator = MSMAggregator()

        with self.assertRaises(TypeError) as context:
            aggregator._check_total_slices(total_slices, output_data_files)
        self.assertTrue("total_slices is <class 'str'>, should be int" in str(context.exception))
        self.assertRaises(AttributeError, lambda: aggregator.total_slices)

    def test_check_total_slices_length_wrong(self) -> None:
        total_slices = 2
        output_data_files = [Path("file/path/a")]
        aggregator = MSMAggregator()

        with self.assertRaises(ValueError) as context:
            aggregator._check_total_slices(total_slices, output_data_files)
        self.assertTrue("Number of output files 1 must equal total_slices 2" in str(context.exception))
        self.assertEqual(total_slices, aggregator.total_slices)

    def test_check_total_slices(self) -> None:
        total_slices = 2
        output_data_files = [Path("file/path/a"), Path("file/path/b")]
        aggregator = MSMAggregator()
        aggregator._check_total_slices(total_slices, output_data_files)
        self.assertEqual(total_slices, aggregator.total_slices)

    def test_write_aggregation_file(self) -> None:
        output_file_paths = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        sliced_data_files = [Path(x) for x in output_file_paths]
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)

            aggregator = MSMAggregator()
            aggregator_filepath = aggregator.aggregate(2, cluster_output_dir, sliced_data_files)
            total_volume = aggregator.total_volume
            total_weights = aggregator.accumulator_weights
            with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
                volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
                weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
            np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
            np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

            self.assertEqual(aggregator_filepath, cluster_output_dir / "aggregated_results.nxs")
            with h5py.File(aggregator_filepath, "r") as af:
                aggregated_volumes = np.array(af["processed"]["reciprocal_space"]["volume"])
                aggregated_weights = np.array(af["processed"]["reciprocal_space"]["weight"])
            np.testing.assert_allclose(volumes_array, aggregated_volumes, rtol=0.001)
            np.testing.assert_allclose(weights_array, aggregated_weights, rtol=0.001)


if __name__ == '__main__':
    unittest.main()