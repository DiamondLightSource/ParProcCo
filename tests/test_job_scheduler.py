from __future__ import annotations

import logging
import os
import pytest
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from parameterized import parameterized

from example.simple_processing_mode import SimpleProcessingMode
from models.slurm_rest import JobProperties, JobSubmission, JobsResponse
from ParProcCo.job_scheduler import JobScheduler, SLURMSTATE, StatusInfo
from .utils import (
    get_slurm_rest_url,
    get_tmp_base_dir,
    setup_data_files,
    setup_jobscript,
    setup_runner_script,
    TemporaryDirectory,
    PARTITION,
)

slurm_rest_url = get_slurm_rest_url()
gh_testing = slurm_rest_url is None


def create_js(work_dir, out_dir, timeout=timedelta(hours=2)) -> JobScheduler:
    return JobScheduler(slurm_rest_url, work_dir, out_dir, PARTITION, timeout)


@pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
class TestJobScheduler(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        self.base_dir: str = get_tmp_base_dir()

    def tearDown(self) -> None:
        if gh_testing:
            os.rmdir(self.base_dir)

    def test_create_job_scheduler(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output_dir"
            js = create_js(working_directory, cluster_output_dir)
        self.assertTrue(
            js._session.headers["X-SLURM-USER-NAME"] == os.environ["USER"],
            msg="User name not set correctly\n",
        )

    def test_create_job_submission(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            input_path = Path("path/to/file.extension")
            cluster_output_dir = Path(working_directory) / "cluster_output_dir"
            scheduler = create_js(working_directory, cluster_output_dir)
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters([slice(0, None, 2), slice(1, None, 2)])
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            scheduler.jobscript_path = runner_script
            scheduler.jobscript_args = runner_script_args
            scheduler.jobscript_command = " ".join(
                [
                    "#!/bin/bash\n",
                    str(scheduler.jobscript_path),
                    *scheduler.jobscript_args,
                ]
            )
            expected_command = (
                f"#!/bin/bash\n{runner_script} {jobscript} --memory 4000 --cores 5"
                f" --output {cluster_output_dir}/out_0 --images 0::2"
                f" --input-path {input_path}"
            )

            scheduler.memory = 4000
            scheduler.cores = 5
            scheduler.job_name = "create_template_test"
            scheduler.scheduler_mode = processing_mode
            scheduler.job_env = {"ParProcCo": "0"}
            (
                _,
                stdout_fp,
                stderr_fp,
            ) = scheduler.scheduler_mode.generate_output_paths(
                cluster_output_dir,
                cluster_output_dir / "cluster_logs",
                0,
                scheduler.start_time,
            )
            job_submission = scheduler.make_job_submission(0)
            cluster_output_dir_exists = cluster_output_dir.is_dir()

        expected = JobSubmission(
            script=expected_command,
            job=JobProperties(
                name="create_template_test",
                partition=PARTITION,
                cpus_per_task=5,
                environment={"ParProcCo": "0"},
                memory_per_cpu=4000,
                current_working_directory=str(working_directory),
                standard_output=stdout_fp,
                standard_error=stderr_fp,
                get_user_environment="10L",
            ),
            jobs=None,
        )

        self.assertTrue(
            cluster_output_dir_exists,
            msg="Cluster output directory was not created\n",
        )

        self.assertEqual(
            job_submission,
            expected,
            msg="JobSubmission has incorrect parameter values\n",
        )

    def test_job_scheduler_runs(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            input_path, output_paths, out_nums, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            # submit jobs
            scheduler = create_js(working_directory, cluster_output_dir)
            scheduler.run(
                processing_mode,
                runner_script,
                jobscript_args=runner_script_args,
            )

            # check output files
            for output_file, expected_nums in zip(output_paths, out_nums):
                with open(output_file, "r") as f:
                    file_content = f.read()
                self.assertTrue(
                    output_file.is_file(),
                    msg=f"Output file {output_file} was not created\n",
                )
                self.assertEqual(
                    expected_nums,
                    file_content,
                    msg=f"Output file {output_file} content was incorrect\n",
                )

    def test_old_output_timestamps(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)

            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            js = create_js(working_directory, cluster_output_dir)

            # submit jobs
            js.jobscript_path = runner_script
            js.job_env = {}
            job_indices = list(range(processing_mode.number_jobs))
            js.jobscript_args = runner_script_args
            js.job_history[js.batch_number] = {}
            js.job_completion_status = {str(i): False for i in range(4)}
            js.memory = 4000
            js.cores = 6
            js.job_name = "old_output_test"
            js.scheduler_mode = processing_mode

            # _submit_and_monitor
            js._submit_jobs(job_indices)
            js._wait_for_jobs()
            t = datetime.now()
            js.start_time = t

            with self.assertLogs(level="WARNING") as context:
                js._report_job_info()
                self.assertEqual(len(context.output), 4)
                for i, err_msg in enumerate(context.output):
                    test_msg = (
                        f"with args ['{working_directory + '/test_script'}', '--input-path',"
                        f" '{working_directory + '/test_raw_data.txt'}'] has not created a new output file"
                    )
                    self.assertTrue(test_msg in err_msg)
            js._report_job_info()

            job_stats = js.job_completion_status
            # check failure list
            self.assertFalse(
                js.get_success(), msg="JobScheduler.success is not False\n"
            )
            self.assertFalse(
                any(job_stats.values()),
                msg=f"All jobs not failed:" f"{js.job_completion_status.values()}\n",
            )
            self.assertEqual(
                len(job_stats),
                4,
                msg=f"len(js.job_completion_status) is not 4. js.job_completion_status: {job_stats}\n",
            )

    def test_job_times_out(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            with open(jobscript, "a+") as f:
                f.write("    import time\n    time.sleep(60)\n")

            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            # submit jobs
            js = create_js(
                working_directory, cluster_output_dir, timeout=timedelta(seconds=1)
            )

            with self.assertLogs(level="WARNING") as context:
                js.run(
                    processing_mode,
                    runner_script,
                    jobscript_args=runner_script_args,
                )
                self.assertEqual(len(context.output), 8)
                for warn_msg in context.output[:4]:
                    self.assertTrue(
                        warn_msg.endswith(" timed out. Terminating job now.")
                    )
                for err_msg in context.output[4:]:
                    self.assertTrue("ended with job state" in err_msg)

            jh = js.job_history
            self.assertEqual(
                len(jh),
                1,
                f"There should be one batch of jobs; job_history: {jh}\n",
            )
            returned_jobs = jh[0]
            self.assertEqual(len(returned_jobs), 4)
            for job_id in returned_jobs:
                self.assertEqual(
                    returned_jobs[job_id].final_state, SLURMSTATE.CANCELLED
                )

    @parameterized.expand(
        [
            (
                "bad_name",
                "bad_jobscript_name",
                False,
                None,
                FileNotFoundError,
                "bad_jobscript_name does not exist",
            ),
            (
                "insufficient_permissions",
                "test_bad_permissions",
                True,
                0o666,
                PermissionError,
                "must be readable and executable by user",
            ),
            (
                "cannot_be_opened",
                "test_bad_read_permissions",
                True,
                0o333,
                PermissionError,
                "must be readable and executable by user",
            ),
        ]
    )
    def test_script(
        self, name, rs_name, open_rs, permissions, error_name, error_msg
    ) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            jobscript = Path(working_directory) / "test_jobscript"
            runner_script = Path(working_directory) / rs_name
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)
            if open_rs:
                f = open(runner_script, "x")
                f.close()
                os.chmod(runner_script, permissions)

            with self.assertRaises(error_name) as context:
                js.run(
                    processing_mode,
                    runner_script,
                    jobscript_args=runner_script_args,
                )
            self.assertTrue(error_msg in str(context.exception))

    def test_get_output_paths(self) -> None:
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            js.output_paths = [
                cluster_output_dir / "out1.nxs",
                cluster_output_dir / "out2.nxs",
            ]
            self.assertEqual(
                js.get_output_paths(),
                [cluster_output_dir / "out1.nxs", cluster_output_dir / "out2.nxs"],
            )

    @parameterized.expand(
        [
            ("all_true", True, True, True),
            ("all_false", False, False, False),
            ("true_false", True, False, False),
        ]
    )
    def test_get_success(self, name, stat_0, stat_1, success) -> None:
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            js = create_js(working_directory, cluster_output_dir)
            js.job_completion_status = {"0": stat_0, "1": stat_1}
            self.assertEqual(js.get_success(), success)

    @parameterized.expand([("true", True), ("false", False)])
    def test_timestamp_ok_true(self, name, run_scheduler_last) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            filepath = cluster_output_dir / "out_0.nxs"
            if run_scheduler_last:
                js = create_js(working_directory, cluster_output_dir)
                time.sleep(2)
            f = open(filepath, "x")
            f.close()
            if not run_scheduler_last:
                time.sleep(2)
                js = create_js(working_directory, cluster_output_dir)
            self.assertEqual(js.timestamp_ok(filepath), run_scheduler_last)

    def test_get_jobs_response(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output_dir"
            js = create_js(working_directory, cluster_output_dir)
            jobs = js.get_jobs_response()
        self.assertTrue(
            isinstance(jobs, JobsResponse),
            msg="jobs is not instance of JobsResponse\n",
        )

    @parameterized.expand(
        [
            (
                "all_killed",
                [
                    StatusInfo(
                        output_path=Path(f"to/somewhere_{i}"),
                        i=i,
                        start_time=0,
                        current_state=SLURMSTATE.CANCELLED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.FAILED,
                    )
                    for i in range(2)
                ],
                [0, 1],
            ),
            (
                "none_killed",
                [
                    StatusInfo(
                        output_path=Path(f"to/somewhere_{i}"),
                        i=i,
                        start_time=0,
                        current_state=SLURMSTATE.BOOT_FAIL,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.FAILED,
                    )
                    for i in range(2)
                ],
                [],
            ),
            (
                "one_killed",
                [
                    StatusInfo(
                        output_path=Path("to/somewhere_0"),
                        i=0,
                        start_time=0,
                        current_state=SLURMSTATE.CANCELLED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.FAILED,
                    ),
                    StatusInfo(
                        output_path=Path("to/somewhere_1"),
                        i=1,
                        start_time=0,
                        current_state=SLURMSTATE.OUT_OF_MEMORY,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.OUT_OF_MEMORY,
                    ),
                ],
                [0],
            ),
        ]
    )
    def test_filter_killed_jobs(self, name, failed_jobs, result) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            killed_jobs_indices = js.filter_killed_jobs(failed_jobs)
            self.assertEqual(killed_jobs_indices, result)

    def test_resubmit_jobs(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            input_path, output_paths, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters(slices)
            jobscript = setup_runner_script(working_directory)

            js = create_js(working_directory, cluster_output_dir)
            js.jobscript_path = jobscript
            js.jobscript_args = [
                str(setup_jobscript(working_directory)),
                "--input-path",
                str(input_path),
            ]
            js.memory = 4000
            js.cores = 6
            js.job_name = "test_resubmit_jobs"
            js.scheduler_mode = processing_mode
            js.output_paths = output_paths
            js.job_env = {"ParProcCo": "0"}
            js.job_history = {
                0: {
                    0: StatusInfo(
                        output_path=Path("to/somewhere_0"),
                        i=0,
                        start_time=0,
                        current_state=SLURMSTATE.CANCELLED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.FAILED,
                    ),
                    1: StatusInfo(
                        output_path=Path("to/somewhere_1"),
                        i=1,
                        start_time=0,
                        current_state=SLURMSTATE.COMPLETED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.COMPLETED,
                    ),
                    2: StatusInfo(
                        output_path=Path("to/somewhere_2"),
                        i=2,
                        start_time=0,
                        current_state=SLURMSTATE.CANCELLED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.FAILED,
                    ),
                    3: StatusInfo(
                        output_path=Path("to/somewhere_3"),
                        i=3,
                        start_time=0,
                        current_state=SLURMSTATE.COMPLETED,
                        slots=0,
                        time_to_dispatch=0,
                        wall_time=0,
                        final_state=SLURMSTATE.COMPLETED,
                    ),
                }
            }

            js.job_completion_status = {
                "0": False,
                "1": True,
                "2": False,
                "3": True,
            }

            success = js.resubmit_jobs([0, 2])
            self.assertTrue(success)
            resubmitted_output_paths = [output_paths[i] for i in [0, 2]]
            for output in resubmitted_output_paths:
                self.assertTrue(output.is_file())

    @parameterized.expand(
        [
            (
                "all_success",
                False,
                {
                    0: {
                        i: StatusInfo(
                            output_path=Path(f"to/somewhere_{i}"),
                            i=i,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.COMPLETED,
                        )
                        for i in range(4)
                    }
                },
                {str(i): True for i in range(4)},
                False,
                None,
                True,
                False,
            ),
            (
                "all_failed_do_not_allow",
                False,
                {
                    0: {
                        i: StatusInfo(
                            output_path=Path(f"to/somewhere_{i}"),
                            i=i,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        )
                        for i in range(4)
                    }
                },
                {str(i): False for i in range(4)},
                False,
                None,
                False,
                True,
            ),
            (
                "all_failed_do_allow",
                True,
                {
                    0: {
                        i: StatusInfo(
                            output_path=Path(f"to/somewhere_{i}"),
                            i=i,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        )
                        for i in range(4)
                    }
                },
                {str(i): False for i in range(4)},
                True,
                [0, 1, 2, 3],
                True,
                False,
            ),
            (
                "some_failed_do_allow",
                True,
                {
                    0: {
                        0: StatusInfo(
                            output_path=Path("to/somewhere_0"),
                            i=0,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        1: StatusInfo(
                            output_path=Path("to/somewhere_1"),
                            i=1,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                        2: StatusInfo(
                            output_path=Path("to/somewhere_2"),
                            i=2,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        3: StatusInfo(
                            output_path=Path("to/somewhere_3"),
                            i=3,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                    }
                },
                {"0": False, "1": True, "2": False, "3": True},
                True,
                [0, 2],
                True,
                False,
            ),
            (
                "some_failed_do_not_allow",
                False,
                {
                    0: {
                        0: StatusInfo(
                            output_path=Path("to/somewhere_0"),
                            i=0,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        1: StatusInfo(
                            output_path=Path("to/somewhere_1"),
                            i=1,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                        2: StatusInfo(
                            output_path=Path("to/somewhere_2"),
                            i=2,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        3: StatusInfo(
                            output_path=Path("to/somewhere_3"),
                            i=3,
                            start_time=0,
                            current_state=SLURMSTATE.CANCELLED,
                            slots=0,
                            time_to_dispatch=0,
                            wall_time=0,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                    }
                },
                {"0": False, "1": True, "2": False, "3": True},
                True,
                [0, 2],
                True,
                False,
            ),
        ]
    )
    def test_resubmit_killed_jobs(
        self,
        name,
        allow_all_failed,
        job_history,
        job_completion_status,
        runs,
        indices,
        expected_success,
        raises_error,
    ) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            input_path, output_paths, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters(slices)
            js = create_js(working_directory, cluster_output_dir)
            jobscript = setup_runner_script(working_directory)
            js.jobscript_path = jobscript
            js.jobscript_args = [
                str(setup_jobscript(working_directory)),
                "--input-path",
                str(input_path),
            ]
            js.memory = 4000
            js.cores = 6
            js.job_name = "test_resubmit_jobs"
            js.scheduler_mode = processing_mode
            for output_path, status_info in zip(output_paths, job_history[0].values()):
                status_info.output = output_path
            js.job_env = {"ParProcCo": "0"}
            js.job_history = job_history
            js.job_completion_status = job_completion_status
            js.output_paths = output_paths

            if raises_error:
                with self.assertRaises(RuntimeError) as context:
                    js.resubmit_killed_jobs(allow_all_failed)
                self.assertTrue(
                    "All jobs failed. job_history: " in str(context.exception)
                )
                self.assertEqual(js.batch_number, 0)
                return

            success = js.resubmit_killed_jobs(allow_all_failed)
            self.assertEqual(success, expected_success)
            self.assertEqual(js.output_paths, output_paths)
            if runs:
                self.assertEqual(js.batch_number, 1)
                resubmitted_output_paths = [output_paths[i] for i in indices]
                for output in resubmitted_output_paths:
                    self.assertTrue(output.is_file())
            else:
                self.assertEqual(js.batch_number, 0)


if __name__ == "__main__":
    unittest.main()
