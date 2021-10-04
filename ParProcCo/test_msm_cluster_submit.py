from __future__ import annotations

import getpass
import logging
import os.path
import subprocess
import unittest
from parameterized import parameterized
from pathlib import Path
from tempfile import TemporaryDirectory

from test_job_controller import setup_jobscript, setup_data_file


class TestClusterSubmit(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)
        self.current_dir = os.getcwd()

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_end_to_end(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            current_script_dir = Path(os.path.realpath(__file__)).parent
            runner_script_path = str(current_script_dir / "msm_cluster_submit.py")
            cluster_output_name = "cluster_output"

            input_file_path = "/scratch/victoria/i07-394487-applied-whole.nxs"

            args = ["python", runner_script_path, "-o", cluster_output_name, "-p", "b24", "-q", "medium.q", "-n", "4",
                    "-f", input_file_path]

            proc = subprocess.Popen(args)
            proc.communicate()
            cluster_output_dir = Path(working_directory) / cluster_output_name
            self.assertTrue(cluster_output_dir.is_dir())
            output_files = [cluster_output_dir / f"out_i07-394487-applied-whole_{i}.nxs" for i in range(4)]

            for output_file in output_files:
                self.assertTrue(output_file.is_file())

            aggregated_file = cluster_output_dir / "aggregated_results.txt"
            self.assertTrue(aggregated_file.is_file())

            print("job finished")


if __name__ == '__main__':
    unittest.main()