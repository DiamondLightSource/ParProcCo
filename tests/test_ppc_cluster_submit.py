from __future__ import annotations

import getpass
import logging
import os.path
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_job_scheduler import CLUSTER_PROJ

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
            submit_script_path = str(current_script_dir.parent / "scripts" / f"{CLUSTER_PROJ}_cluster_submit")
            cluster_output_name = "cluster_output"

            input_file_path = "/dls/science/groups/das/ExampleData/i07/i07-394487-applied.nxs"

            repo_dir = str(current_script_dir.parent)
            # these are inherited by sub-processes
            os.environ["TEST_PPC_DIR"] = repo_dir
            os.environ["PYTHONPATH"] = f"{repo_dir}:{os.environ['PYTHONPATH']}" if 'PYTHONPATH' in os.environ else repo_dir
            os.environ["PATH"] = f"{repo_dir}/scripts:{os.environ['PATH']}"

            args = [submit_script_path, "rs_map", "--jobs", "4", "-s", "0.01",
                    "--output", cluster_output_name, "--cores", "6", "--memory", "4G", input_file_path]
            proc = subprocess.Popen(args)
            proc.communicate()
            cluster_output_dir = Path(working_directory) / cluster_output_name
            self.assertTrue(cluster_output_dir.is_dir())
            output_files = [cluster_output_dir / f"out_{i}.nxs" for i in range(4)]

            for output_file in output_files:
                self.assertFalse(output_file.is_file())

            aggregated_file = cluster_output_dir / "aggregated_results.nxs"
            self.assertTrue(aggregated_file.is_file())


if __name__ == '__main__':
    unittest.main()