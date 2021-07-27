import os
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import timedelta
import unittest

from job_controller import JobController
from simple_data_chunker import SimpleDataChunker


def setup_data_file(working_directory):
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("3\n4\n11\n30\n")
    return input_file_path


class TestJobController(unittest.TestCase):

    def test_end_to_end(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_script.sh")

            input_file_path = setup_data_file(working_directory)

            jc = JobController(working_directory, cluster_output_dir, project="b24", priority="medium.q")
            agg_data_path = jc.run(SimpleDataChunker, [4], input_file_path, jobscript)

            self.assertEqual(agg_data_path, Path(cluster_output_dir) / "aggregated_results.txt")
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["96"])

    def test_all_jobs_fail(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_sleeper_script.sh")

            input_file_path = setup_data_file(working_directory)

            jc = JobController(working_directory, cluster_output_dir, project="b24", priority="medium.q",
                               timeout=timedelta(seconds=1))
            with self.assertRaises(RuntimeError) as context:
                jc.run(SimpleDataChunker, [4], input_file_path, jobscript)
            self.assertTrue(f"All jobs failed\n" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
