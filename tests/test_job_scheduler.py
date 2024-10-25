# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import getpass
import logging
import os
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from parameterized import parameterized  # type: ignore

from example.simple_processing_slicer import SimpleProcessingSlicer
from ParProcCo.job_scheduler import SLURMSTATE, JobScheduler, StatusInfo
from ParProcCo.job_scheduling_information import JobResources, JobSchedulingInformation
from ParProcCo.slurm.slurm_rest import (
    JobDescMsg,
    JobInfo,
    JobSubmitReq,
    StringArray,
    Uint32NoVal,
    Uint64NoVal,
)
from ParProcCo.test import TemporaryDirectory
from tests.utils import (
    PARTITION,
    get_slurm_rest_url,
    get_tmp_base_dir,
    setup_data_files,
    setup_jobscript,
    setup_runner_script,
)

slurm_rest_url = get_slurm_rest_url()
gh_testing = slurm_rest_url is None

if not gh_testing and not PARTITION:
    raise ValueError("Need to define default partition in SLURM_PARTITION")


def create_js(timeout=timedelta(seconds=20)) -> JobScheduler:
    assert slurm_rest_url and PARTITION
    return JobScheduler(slurm_rest_url, PARTITION, wait_timeout=timeout)


@pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
class TestJobScheduler(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        self.base_dir: str = get_tmp_base_dir()

    def tearDown(self) -> None:
        if gh_testing:
            os.rmdir(self.base_dir)

    def test_create_job_scheduler(self) -> None:
        js = create_js()
        self.assertTrue(
            js.client._session.headers["X-SLURM-USER-NAME"] == os.environ["USER"],
            msg="User name not set correctly\n",
        )

    @parameterized.expand([("cpu only", 0), ("gpu", 1)])
    def test_create_job_submission(self, _name, gpus) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            working_directory = Path(working_directory)
            input_path = Path("path/to/file.extension")
            cluster_output_dir = working_directory / "cluster_output_dir"
            scheduler = create_js()
            job_script = setup_jobscript(working_directory)
            runner_script = setup_runner_script(working_directory)
            runner_script_args = (str(job_script), "--input-path", str(input_path))
            timestamp_time = datetime.now()

            jsi = JobSchedulingInformation(
                job_name="create_template_test",
                job_script_path=runner_script,
                job_script_arguments=runner_script_args,
                job_resources=JobResources(memory=4000, cpu_cores=5, gpus=gpus),
                working_directory=working_directory,
                job_env={"ParProcCo": "0"},
                timestamp=timestamp_time,
                output_dir=cluster_output_dir,
                log_directory=None,
            )
            processing_slicer = SimpleProcessingSlicer(job_script)
            slice_params = [slice(0, None, 2), slice(1, None, 2)]
            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slice_params, job_scheduling_information=jsi
            )

            expected_command = (
                f"#!/bin/bash\n{runner_script} {job_script} --memory 4000M --cores 5"
                f" --output {cluster_output_dir}/out_0 --images 0::2"
                f" --input-path {input_path}"
            )
            job_submission = scheduler.make_job_submission(jsi_list[0])

        env_list = [f"{k}={v}" for k, v in jsi.job_env.items()]
        env_list.append(f"USER={getpass.getuser()}")
        expected = JobSubmitReq(
            script=expected_command,
            job=JobDescMsg(
                name="create_template_test",
                partition=PARTITION,
                cpus_per_task=5,
                tres_per_task=f"gres/gpu:{gpus}",
                tasks=1,
                time_limit=Uint32NoVal(
                    number=int((jsi.timeout.total_seconds() + 59) // 60), set=True
                ),
                environment=StringArray(root=env_list),
                memory_per_cpu=Uint64NoVal(number=jsi.job_resources.memory, set=True),
                current_working_directory=str(working_directory),
                standard_output=str(jsi_list[0].get_stdout_path()),
                standard_error=str(jsi_list[0].get_stderr_path()),
            ),
            jobs=None,
        )

        self.assertEqual(
            job_submission,
            expected,
            msg="JobSubmission has incorrect parameter values\n",
        )

    @parameterized.expand([("cpu only", 0), ("gpu", 1)])
    def test_job_scheduler_runs(self, _name, gpus) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            working_directory = Path(working_directory)
            cluster_output_dir = working_directory / "cluster_output"

            input_path, output_paths, out_nums, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            job_script = setup_jobscript(working_directory)
            runner_script = setup_runner_script(working_directory)
            runner_script_args = (str(job_script), "--input-path", str(input_path))
            processing_slicer = SimpleProcessingSlicer(job_script)

            jsi = JobSchedulingInformation(
                job_name="scheduler_runs_test",
                job_script_path=runner_script,
                job_script_arguments=runner_script_args,
                job_resources=JobResources(memory=4000, cpu_cores=5, gpus=gpus),
                working_directory=working_directory,
                job_env={"ParProcCo": "0"},
                timestamp=datetime.now(),
                output_dir=cluster_output_dir,
                log_directory=cluster_output_dir / "cluster_logs",
                timeout=timedelta(seconds=60),
            )

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )

            # submit jobs
            js = create_js()
            js.run(jsi_list)

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
            working_directory = Path(working_directory)
            cluster_output_dir = working_directory / "cluster_output"
            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            job_script = setup_jobscript(working_directory)
            runner_script = setup_runner_script(working_directory)
            runner_script_args = (str(job_script), "--input-path", str(input_path))
            processing_slicer = SimpleProcessingSlicer(job_script)

            jsi = JobSchedulingInformation(
                job_name="old_output_test",
                job_script_path=runner_script,
                job_script_arguments=runner_script_args,
                job_resources=JobResources(memory=4000, cpu_cores=6),
                working_directory=working_directory,
                job_env={},
                timestamp=datetime.now(),
                output_dir=cluster_output_dir,
                timeout=timedelta(seconds=60),
            )

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )

            js = create_js(timeout=timedelta(seconds=120))

            # submit jobs

            # _submit_and_monitor
            js._submit_jobs(jsi_list)
            js._wait_for_jobs(jsi_list)
            for jsi in jsi_list:
                assert jsi.status_info
                jsi.status_info.start_time = datetime.now() + timedelta(seconds=60)
                logging.debug(
                    "Job %i: %s; %s; %s",
                    jsi.job_id,
                    jsi.status_info.current_state,
                    jsi.status_info.start_time,
                    jsi.get_stdout_path(),
                )

            with self.assertLogs(level="WARNING") as context:
                js._report_job_info(jsi_list)
                self.assertEqual(len(context.output), 4)
                logging.debug("Logs are: %s", context.output)
                for i, err_msg in enumerate(context.output):
                    test_msg = (
                        f"'--input-path', '{input_path}') has not created a "
                        f"new output file '{jsi_list[i].get_stdout_path()}'"
                    )
                    self.assertTrue(
                        test_msg in err_msg,
                        msg=f"'{test_msg}' was not found in '{err_msg}'",
                    )
            js._report_job_info(jsi_list)

            # check failure list
            jh = js.get_job_history_batch()
            completion_statuses = [jsi.completion_status for jsi in jh.values()]
            self.assertFalse(
                js.get_success(jsi_list), msg="JobScheduler.success is not False\n"
            )
            self.assertFalse(
                any(completion_statuses),
                msg=f"All jobs not failed:" f"{completion_statuses}\n",
            )
            self.assertEqual(
                len(completion_statuses),
                4,
                msg="Number of completion statuses for batch is not 4."
                f"Completion statuses: {completion_statuses}\n",
            )

    def test_job_times_out(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            working_directory = Path(working_directory)
            cluster_output_dir = working_directory / "cluster_output"
            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            job_script = setup_jobscript(working_directory)
            with open(job_script, "a+") as f:
                f.write("    import time\n    time.sleep(120)\n")
            runner_script = setup_runner_script(working_directory)
            runner_script_args = (str(job_script), "--input-path", str(input_path))
            processing_slicer = SimpleProcessingSlicer(job_script)

            jsi = JobSchedulingInformation(
                job_name="timeout_test",
                job_script_path=runner_script,
                job_script_arguments=runner_script_args,
                job_resources=JobResources(memory=4000, cpu_cores=6),
                working_directory=working_directory,
                job_env={},
                timestamp=datetime.now(),
                output_dir=working_directory,
                log_directory=cluster_output_dir,
                timeout=timedelta(seconds=5),
            )

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )

            # submit jobs
            js = create_js(timeout=timedelta(seconds=10))

            with self.assertLogs(level="WARNING") as context:
                js.run(jsi_list)
                self.assertEqual(len(context.output), 8, msg=f"{context.output}")
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
            for jsi in returned_jobs.values():
                assert jsi.status_info
                self.assertEqual(jsi.status_info.final_state, SLURMSTATE.CANCELLED)

    @parameterized.expand(
        [
            (
                "bad_name",
                "bad_jobscript_name",
                False,
                None,
                FileNotFoundError,
                "bad_jobscript_name not found",
            ),
            (
                "insufficient_permissions",
                "test_bad_permissions",
                True,
                0o660,
                PermissionError,
                "must be readable and executable by user",
            ),
            (
                "cannot_be_opened",
                "test_bad_read_permissions",
                True,
                0o330,
                PermissionError,
                "must be readable and executable by user",
            ),
        ]
    )
    def test_script(
        self, _name, rs_name, open_rs, permissions, error_name, error_msg
    ) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            working_directory = Path(working_directory)
            cluster_output_dir = working_directory / "cluster_output"

            js = create_js()
            input_path, _, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )
            job_script = working_directory / "test_jobscript"
            runner_script = working_directory / rs_name
            runner_script_args = (str(job_script), "--input-path", str(input_path))

            job_script.touch()
            os.chmod(job_script, 0o770)

            if open_rs:
                f = open(runner_script, "x")
                f.close()
                os.chmod(runner_script, permissions)

            with self.assertRaises(error_name) as context:
                processing_slicer = SimpleProcessingSlicer(runner_script)
            self.assertTrue(
                error_msg in str(context.exception), msg=str(context.exception)
            )

            # Now set up the processing mode bypassing the check using a valid file:
            processing_slicer = SimpleProcessingSlicer(job_script)
            processing_slicer.job_script = runner_script

            # Bypass the protection in the JobSchedulingInformation to check the protection
            # in the job_scheduler
            jsi = JobSchedulingInformation(
                job_name="bad_scripts",
                job_script_path=job_script,
                job_script_arguments=runner_script_args,
                job_resources=JobResources(memory=4000, cpu_cores=6),
                working_directory=working_directory,
                job_env={},
                timestamp=datetime.now(),
                output_dir=cluster_output_dir,
                timeout=timedelta(seconds=5),
            )
            jsi.job_script_path = runner_script

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )

            with self.assertRaises(error_name) as context:
                js.run(jsi_list)
            self.assertTrue(error_msg in str(context.exception))

    def test_get_output_paths(self) -> None:
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js()

            output_paths = (
                cluster_output_dir / "out1.nxs",
                cluster_output_dir / "out2.nxs",
            )

            jsi_list: list[JobSchedulingInformation] = []
            for i, output in enumerate(output_paths, 1):
                jsi = MagicMock(spec=JobSchedulingInformation, name=f"JSI_{i}")
                jsi.get_output_path.return_value = output
                jsi_list.append(jsi)

            self.assertEqual(
                js.get_output_paths(jsi_list),
                output_paths,
            )

    @parameterized.expand(
        [
            ("all_true", True, True, True),
            ("all_false", False, False, False),
            ("true_false", True, False, False),
        ]
    )
    def test_get_success(self, _name, stat_0, stat_1, success) -> None:
        js = create_js()

        jsi_list: list[JobSchedulingInformation] = []
        for i, complete in enumerate([stat_0, stat_1], 1):
            jsi = MagicMock(spec=JobSchedulingInformation, name=f"JSI_{i}")
            jsi.completion_status = complete
            jsi_list.append(jsi)

        self.assertEqual(js.get_success(jsi_list), success)

    @parameterized.expand([("true", True), ("false", False)])
    def test_timestamp_ok_true(self, _name, start_time_before) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            filepath = cluster_output_dir / "out_0.nxs"
            if start_time_before:
                start_time = datetime.now()
                time.sleep(2)
            f = open(filepath, "x")
            f.close()
            if not start_time_before:
                time.sleep(2)
                start_time = datetime.now()

            js = create_js()
            self.assertEqual(
                js.timestamp_ok(filepath, start_time=start_time), start_time_before
            )

    def test_get_jobs_response(self) -> None:
        js = create_js()
        jobs = js.client.get_job_response()
        self.assertTrue(
            isinstance(jobs, list),
            msg="jobs is not instance of list\n",
        )
        if jobs:
            self.assertTrue(
                isinstance(jobs[0], JobInfo),
                msg="job is not instance of JobInfo\n",
            )

    @parameterized.expand(
        [
            (
                "all_killed",
                [
                    StatusInfo(
                        submit_time=datetime.now(),
                        current_state=SLURMSTATE.CANCELLED,
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
                        submit_time=datetime.now(),
                        current_state=SLURMSTATE.BOOT_FAIL,
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
                        submit_time=datetime.now(),
                        current_state=SLURMSTATE.CANCELLED,
                        final_state=SLURMSTATE.FAILED,
                    ),
                    StatusInfo(
                        submit_time=datetime.now(),
                        current_state=SLURMSTATE.OUT_OF_MEMORY,
                        final_state=SLURMSTATE.OUT_OF_MEMORY,
                    ),
                ],
                [0],
            ),
        ]
    )
    def test_filter_killed_jobs(self, _name, job_statuses, result) -> None:
        js = create_js()
        jsi_list: list[JobSchedulingInformation] = []
        for status_info in job_statuses:
            jsi = MagicMock(
                spec=JobSchedulingInformation, name="JobSchedulingInformation"
            )
            jsi.status_info = status_info
            jsi_list.append(jsi)
        killed_jobs = js.filter_killed_jobs(jsi_list)
        self.assertEqual(killed_jobs, [jsi_list[i] for i in result])

    def test_resubmit_jobs(self) -> None:
        with TemporaryDirectory(
            prefix="test_dir_", dir=self.base_dir
        ) as working_directory:
            working_directory = Path(working_directory)
            cluster_output_dir = working_directory / "cluster_output"
            input_path, output_paths, _, slices = setup_data_files(
                working_directory, cluster_output_dir
            )

            job_ids = [0, 1, 2, 3]
            stdout_paths = [
                cluster_output_dir / "to" / f"somewhere_{i}" for i in job_ids
            ]
            status_infos = [
                StatusInfo(
                    submit_time=datetime.now(),
                    current_state=SLURMSTATE.CANCELLED,
                    final_state=SLURMSTATE.FAILED,
                ),
                StatusInfo(
                    submit_time=datetime.now(),
                    current_state=SLURMSTATE.COMPLETED,
                    final_state=SLURMSTATE.COMPLETED,
                ),
                StatusInfo(
                    submit_time=datetime.now(),
                    current_state=SLURMSTATE.CANCELLED,
                    final_state=SLURMSTATE.FAILED,
                ),
                StatusInfo(
                    submit_time=datetime.now(),
                    current_state=SLURMSTATE.COMPLETED,
                    final_state=SLURMSTATE.COMPLETED,
                ),
            ]

            runner_script = setup_runner_script(working_directory)
            job_script = setup_jobscript(working_directory)
            processing_slicer = SimpleProcessingSlicer(job_script)
            jsi = JobSchedulingInformation(
                job_name="test_resubmit_jobs",
                job_script_path=runner_script,
                job_script_arguments=(str(job_script), "--input-path", str(input_path)),
                job_resources=JobResources(memory=4000, cpu_cores=6),
                working_directory=working_directory,
                job_env={"ParProcCo": "0"},
                timestamp=datetime.now(),
                output_dir=cluster_output_dir,
                timeout=timedelta(seconds=30),
            )

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )
            job_history = {}
            completion_statuses = [False, True, False, True]
            for job_id, jsi, stdout_path, status_info, completed in zip(
                job_ids, jsi_list, stdout_paths, status_infos, completion_statuses
            ):
                jsi.log_directory = stdout_path.parent
                jsi.stdout_filename = stdout_path.name
                jsi.stderr_filename = stdout_path.name + "_err"
                jsi.set_job_id(job_id)
                jsi.update_status_info(status_info)
                jsi.set_completion_status(completed)
                job_history[job_id] = jsi

            js = create_js()
            js.job_history.append(job_history)

            success = js.resubmit_jobs(job_ids=[0, 2], batch=0)
            self.assertTrue(success)
            resubmitted_output_paths = [output_paths[i] for i in [0, 2]]
            for output in resubmitted_output_paths:
                self.assertTrue(output.is_file())

    @parameterized.expand(
        [
            (
                "all_success",
                False,
                [
                    {
                        i: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.COMPLETED,
                        )
                        for i in range(4)
                    }
                ],
                {i: Path(f"stdoud_{i}") for i in range(4)},
                {i: True for i in range(4)},
                False,
                None,
                True,
                False,
            ),
            (
                "all_failed_do_not_allow",
                False,
                [
                    {
                        i: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        )
                        for i in range(4)
                    }
                ],
                {i: Path(f"stdout_{i}") for i in range(4)},
                {i: False for i in range(4)},
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
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        )
                        for i in range(4)
                    }
                },
                {i: Path(f"stdout_{i}") for i in range(4)},
                {i: False for i in range(4)},
                True,
                [0, 1, 2, 3],
                True,
                False,
            ),
            (
                "some_failed_do_allow",
                True,
                [
                    {
                        0: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        1: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                        2: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        3: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                    }
                ],
                {i: Path(f"stdout_{i}") for i in range(4)},
                {0: False, 1: True, 2: False, 3: True},
                True,
                [0, 2],
                True,
                False,
            ),
            (
                "some_failed_do_not_allow",
                False,
                [
                    {
                        0: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        1: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                        2: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.FAILED,
                        ),
                        3: StatusInfo(
                            submit_time=datetime.now(),
                            current_state=SLURMSTATE.CANCELLED,
                            final_state=SLURMSTATE.COMPLETED,
                        ),
                    }
                ],
                {i: Path(f"stdout_{i}") for i in range(4)},
                {0: False, 1: True, 2: False, 3: True},
                True,
                [0, 2],
                True,
                False,
            ),
        ]
    )
    def test_resubmit_killed_jobs(
        self,
        _name,
        allow_all_failed,
        job_statuses,
        stdout_paths,
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

            runner_script = setup_runner_script(working_directory)
            job_script = setup_jobscript(working_directory)
            processing_slicer = SimpleProcessingSlicer(job_script)
            jsi = JobSchedulingInformation(
                job_name="test_resubmit_jobs",
                job_script_path=runner_script,
                job_script_arguments=(str(job_script), "--input-path", str(input_path)),
                job_resources=JobResources(memory=4000, cpu_cores=6),
                working_directory=working_directory,
                job_env={"ParProcCo": "0"},
                timestamp=datetime.now(),
                output_dir=cluster_output_dir,
                timeout=timedelta(seconds=30),
            )

            jsi_list = processing_slicer.create_slice_jobs(
                slice_params=slices, job_scheduling_information=jsi
            )

            job_history = {}
            for job_id, jsi in enumerate(jsi_list):
                status_info = job_statuses[0][job_id]
                stdout_path = stdout_paths[
                    job_id
                ]  # Only works because test IDs are 0, 1, 2, 3
                jsi.log_directory = cluster_output_dir / "logs"
                jsi.stdout_filename = stdout_path.name
                jsi.stderr_filename = stdout_path.name + "_err"
                jsi.output_dir = output_paths[job_id].parent
                jsi.output_filename = output_paths[job_id].name
                jsi.set_job_id(job_id)
                jsi.update_status_info(status_info)
                jsi.set_completion_status(job_completion_status[job_id])
                job_history[job_id] = jsi

            js = create_js()
            js.job_history.append(job_history)

            if raises_error:
                with self.assertRaises(RuntimeError) as context:
                    js.resubmit_killed_jobs(allow_all_failed=allow_all_failed)
                self.assertTrue(
                    "All jobs failed. job_history: " in str(context.exception)
                )
                self.assertEqual(js.get_batch_number(), 0)
                return

            success = js.resubmit_killed_jobs(allow_all_failed=allow_all_failed)
            self.assertEqual(success, expected_success)
            latest_batch = js.get_job_history_batch()

            if runs:
                self.assertEqual(js.get_batch_number(), 1)

                resubmitted_output_paths = [output_paths[i] for i in indices]
                self.assertEqual(
                    list(js.get_output_paths(latest_batch.values())),
                    resubmitted_output_paths,
                )
                for output in resubmitted_output_paths:
                    self.assertTrue(output.is_file())
            else:
                self.assertEqual(js.get_batch_number(), 0)
                self.assertEqual(
                    list(js.get_output_paths(latest_batch.values())),
                    output_paths,
                )


if __name__ == "__main__":
    unittest.main()
