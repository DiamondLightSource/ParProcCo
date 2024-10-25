from __future__ import annotations

import logging
import os
import unittest
from datetime import timedelta
from pathlib import Path

import pytest

from example.simple_wrapper import SimpleWrapper
from ParProcCo.job_controller import JobController
from ParProcCo.test import TemporaryDirectory
from tests.utils import (
    PARTITION,
    get_slurm_rest_url,
    get_tmp_base_dir,
    setup_aggregation_script,
    setup_data_file,
    setup_jobscript,
    setup_runner_script,
)

slurm_rest_url = get_slurm_rest_url()
gh_testing = slurm_rest_url is None

if not gh_testing and not PARTITION:
    raise ValueError("Need to define default partition in SLURM_PARTITION")


@pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
class TestJobController(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        self.base_dir: str = get_tmp_base_dir()
        self.current_dir: str = os.getcwd()
        self.starting_path = os.environ["PATH"]
        assert slurm_rest_url and PARTITION
        self.url = slurm_rest_url
        self.partition = PARTITION

    def tearDown(self) -> None:
        os.environ["PATH"] = self.starting_path
        final_path = os.environ["PATH"]
        self.assertTrue(final_path == self.starting_path)
        os.chdir(self.current_dir)
        if gh_testing:
            os.rmdir(self.base_dir)

    def test_all_jobs_fail(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)
            with open(jobscript, "a+") as f:
                f.write("import time\ntime.sleep(60)\n")

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            os.environ["PATH"] = ":".join(
                [str(runner_script.parent), self.starting_path]
            )

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            jc = JobController(
                self.url,
                wrapper,
                Path(cluster_output_name),
                self.partition,
                timeout=timedelta(seconds=1),
            )
            with self.assertRaises(RuntimeError) as context:
                jc.run(4, runner_script_args)
            self.assertTrue("All jobs failed. job_history: " in str(context.exception))

    def test_end_to_end(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)
            os.environ["PATH"] = ":".join(
                [str(runner_script.parent), self.starting_path]
            )

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            jc = JobController(
                self.url,
                wrapper,
                Path(cluster_output_name),
                self.partition,
            )
            jc.run(4, runner_script_args)

            assert jc.aggregated_result
            with open(jc.aggregated_result, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(
                agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"]
            )
            assert jc.sliced_results
            for result in jc.sliced_results:
                self.assertFalse(result.is_file())

    def test_single_job_does_not_aggregate(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)
            os.environ["PATH"] = ":".join(
                [str(runner_script.parent), self.starting_path]
            )

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            aggregated_file = (
                Path(working_directory) / cluster_output_name / "aggregated_results.txt"
            )

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            jc = JobController(
                self.url,
                wrapper,
                Path(cluster_output_name),
                self.partition,
            )
            jc.run(1, runner_script_args)

            assert jc.sliced_results
            self.assertEqual(len(jc.sliced_results), 1)
            self.assertFalse(aggregated_file.is_file())
            self.assertTrue(jc.sliced_results[0].is_file())
            with open(jc.sliced_results[0], "r") as af:
                agg_data = af.readlines()

            self.assertEqual(
                agg_data, ["0\n", "2\n", "4\n", "6\n", "8\n", "10\n", "12\n", "14\n"]
            )


if __name__ == "__main__":
    unittest.main()
