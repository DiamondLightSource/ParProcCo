from __future__ import annotations

import getpass
import logging
import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import drmaa2
from parameterized import parameterized

from example.simple_processing_mode import SimpleProcessingMode
from ParProcCo.job_scheduler import JobScheduler, StatusInfo
from ParProcCo.utils import check_jobscript_is_readable
from tests.utils import setup_data_files, setup_jobscript, setup_runner_script

from .utils import CLUSTER_PROJ, CLUSTER_QUEUE, CLUSTER_RESOURCES

def create_js(work_dir, out_dir, project=CLUSTER_PROJ, queue=CLUSTER_QUEUE, cluster_resources=CLUSTER_RESOURCES,
              timeout=timedelta(hours=2)):
    return JobScheduler(work_dir, out_dir, project, queue, cluster_resources, timeout)

class TestJobScheduler(unittest.TestCase):

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.info(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_create_template_with_cluster_output_dir(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_path = Path('path/to/file.extension')
            cluster_output_dir = Path(working_directory) / 'cluster_output_dir'
            js = create_js(working_directory, cluster_output_dir)
            jobscript = setup_jobscript(working_directory)
            runner_script_args = [jobscript, "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters([slice(0, None, 2), slice(1, None, 2)])
            js.jobscript = Path("some_script.py")
            js.jobscript_args = runner_script_args

            js.memory = '4G'
            js.cores = 5
            js.job_name = "create_template_test"
            js.scheduler_mode = processing_mode
            js._create_template(1)
            cluster_output_dir_exists = cluster_output_dir.is_dir()
        self.assertTrue(cluster_output_dir_exists, msg="Cluster output directory was not created\n")

    def test_job_scheduler_runs(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            input_path, output_paths, out_nums, slices = setup_data_files(working_directory, cluster_output_dir)
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            # run jobs
            js = create_js(working_directory, cluster_output_dir)
            js.run(processing_mode, runner_script, {}, jobscript_args=runner_script_args)

            # check output files
            for output_file, expected_nums in zip(output_paths, out_nums):

                with open(output_file, "r") as f:
                    file_content = f.read()

                self.assertTrue(output_file.is_file(), msg=f"Output file {output_file} was not created\n")
                self.assertEqual(expected_nums, file_content, msg=f"Output file {output_file} content was incorrect\n")

    def test_old_output_timestamps(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)

            input_path, _, _, slices = setup_data_files(working_directory, cluster_output_dir)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            # run jobs
            js = create_js(working_directory, cluster_output_dir)

            # run jobs
            js.jobscript = check_jobscript_is_readable(runner_script)
            job_indices = list(range(processing_mode.number_jobs))
            js.jobscript_args = runner_script_args
            js.job_history[js.batch_number] = {}
            js.job_completion_status = {str(i): False for i in range(4)}

            # _run_and_monitor
            js.jobscript = check_jobscript_is_readable(js.jobscript)
            session = drmaa2.JobSession()  # Automatically destroyed when it is out of scope
            js.memory="4G"
            js.cores=6
            js.job_name="old_output_test"
            js.scheduler_mode = processing_mode
            js._run_jobs(session, job_indices)
            js._wait_for_jobs(session)
            t = datetime.now()
            js.start_time = t
            timestamp = f"{t.year}{t.month}{t.day}_{t.hour}{t.minute}"

            with self.assertLogs(level='WARNING') as context:
                js._report_job_info()
                self.assertEqual(len(context.output), 4)
                for i, err_msg in enumerate(context.output):
                    self.assertTrue(err_msg.startswith("ERROR:root:drmaa job "))
                    test_msg = f"with args ['{working_directory + '/test_script'}', '--input-path',"\
                    f" '{working_directory + '/test_raw_data.txt'}'] has not created a new output file"\
                    f" {working_directory + '/cluster_output/cluster_logs/out_'}{timestamp}_{str(i)}"\
                    f" Terminating signal: 0."

                    self.assertTrue(err_msg.endswith(test_msg))
            js._report_job_info()

            job_stats = js.job_completion_status
            # check failure list
            self.assertFalse(js.get_success(), msg=f"JobScheduler.success is not False\n")
            self.assertFalse(any(job_stats.values()), msg=f"All jobs not failed:"
                             f"{js.job_completion_status.values()}\n")
            self.assertEqual(len(job_stats), 4,
                             msg=f"len(js.job_completion_status) is not 4. js.job_completion_status: {job_stats}\n")

    def test_get_all_queues(self) -> None:
        with os.popen('qconf -sql') as q_proc:
            q_name_list = q_proc.read().split()
        ms = drmaa2.MonitoringSession('ms-01')
        qi_list = ms.get_all_queues(q_name_list)
        self.assertEqual(len(qi_list), len(q_name_list))
        for qi in qi_list:
            q_name = qi.name
            self.assertTrue(q_name in q_name_list)

    @parameterized.expand([
        ("is_none", None, "project must be non-empty string"),
        ("is_empty", "", "project must be non-empty string"),
        ("is_bad", "bad_project_name", "bad_project_name must be in list of project names"),
        ("is_good", CLUSTER_PROJ, None)
    ])
    def test_project_name(self, name, project, error_msg) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            if not error_msg:
                js = create_js(working_directory, cluster_output_dir, project=project)
                self.assertEqual(js.project, project)
                return

            with self.assertRaises(ValueError) as context:
                create_js(working_directory, cluster_output_dir, project=project)
            self.assertTrue(error_msg in str(context.exception))

    @parameterized.expand([
        ("is_lowercase", CLUSTER_QUEUE),
        ("is_uppercase", CLUSTER_QUEUE.upper())
    ])
    def test_check_queue_list(self, name, queue) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir, queue=queue)
            self.assertEqual(js.queue, CLUSTER_QUEUE)

    @parameterized.expand([
        ("is_none", None, "queue must be non-empty string"),
        ("is_empty", "", "queue must be non-empty string"),
        ("is_bad", "bad_queue_name.q", "queue bad_queue_name.q not in queue list")
    ])
    def test_queue(self, name, queue, error_msg) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                create_js(working_directory, cluster_output_dir, queue=queue)
            self.assertTrue(error_msg in str(context.exception))

    def test_job_times_out(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            jobscript = setup_jobscript(working_directory)
            runner_script = setup_runner_script(working_directory)

            with open(jobscript, "a+") as f:
                f.write("import time\ntime.sleep(60)\n")

            input_path, _, _, slices = setup_data_files(working_directory, cluster_output_dir)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            # run jobs
            js = create_js(working_directory, cluster_output_dir, timeout=timedelta(seconds=1))

            with self.assertLogs(level='WARNING') as context:
                js.run(processing_mode, runner_script, {}, jobscript_args=runner_script_args)
                self.assertEqual(len(context.output), 8)
                for warn_msg in context.output[:4]:
                    self.assertTrue(warn_msg.startswith("WARNING:root:Job "))
                    self.assertTrue(warn_msg.endswith(" timed out. Terminating job now."))
                for err_msg in context.output[4:]:
                    self.assertTrue(err_msg.startswith("ERROR:root:drmaa job "))
                    self.assertTrue(err_msg.endswith(" failed. Terminating signal: SIGKILL."))

            jh = js.job_history
            self.assertEqual(len(jh), 1, f"There should be one batch of jobs; job_history: {jh}\n")
            returned_jobs = jh[0]
            self.assertEqual(len(returned_jobs), 4)
            for job_id in returned_jobs:
                self.assertEqual(returned_jobs[job_id].info.exit_status, 137)
                self.assertEqual(returned_jobs[job_id].info.terminating_signal, "SIGKILL")
                self.assertEqual(returned_jobs[job_id].info.job_state, "FAILED")

    @parameterized.expand([
        ("bad_name", "bad_jobscript_name", False, None, FileNotFoundError, "does not exist"),
        ("insufficient_permissions", "test_bad_permissions", True, 0o666, PermissionError,
         "must be readable and executable by user"),
        ("cannot_be_opened", "test_bad_read_permissions", True, 0o333, PermissionError,
         "must be readable and executable by user")
    ])
    def test_script(self, name, rs_name, open_rs, permissions, error_name, error_msg) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            input_path, _, _, slices = setup_data_files(working_directory, cluster_output_dir)
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
                js.run(processing_mode, runner_script, {}, jobscript_args=runner_script_args)

            self.assertTrue(error_msg in str(context.exception))

    def test_check_jobscript(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)

            input_path, _, _, slices = setup_data_files(working_directory, cluster_output_dir)
            jobscript = setup_jobscript(working_directory)
            runner_script = setup_runner_script(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            processing_mode = SimpleProcessingMode(runner_script)
            processing_mode.set_parameters(slices)

            js.run(processing_mode, runner_script, {}, jobscript_args=runner_script_args)

    def test_get_output_paths(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            js.output_paths = [cluster_output_dir / "out1.nxs", cluster_output_dir / "out2.nxs"]
            self.assertEqual(js.get_output_paths(), [cluster_output_dir / "out1.nxs", cluster_output_dir / "out2.nxs"])

    @parameterized.expand([
        ("all_true", True, True, True),
        ("all_false", False, False, False),
        ("true_false", True, False, False)
    ])
    def test_get_success(self, name, stat_0, stat_1, success) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            js = create_js(working_directory, cluster_output_dir)
            js.job_completion_status = {"0": stat_0, "1": stat_1}
            self.assertEqual(js.get_success(), success)

    @parameterized.expand([
        ("true", True),
        ("false", False)
    ])
    def test_timestamp_ok_true(self, name, run_scheduler_last) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            filepath = cluster_output_dir / "out_0.nxs"

            if run_scheduler_last:
                js = create_js(working_directory, cluster_output_dir)
            f = open(filepath, "x")
            f.close()
            if not run_scheduler_last:
                js = create_js(working_directory, cluster_output_dir)
            self.assertEqual(js.timestamp_ok(filepath), run_scheduler_last)

    @parameterized.expand([
        ("all_killed", [
            StatusInfo(None, Path(f"to/somewhere_{i}"), i, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED") for i in range(2)],
            [StatusInfo(None, Path(f"to/somewhere_{i}"), i, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                        drmaa2.JobState(8), "FAILED") for i in range(2)]),
        ("none_killed", [
            StatusInfo(None, Path("to/somewhere"), i, drmaa2.JobInfo({"terminating_signal": "0"}), drmaa2.JobState(8),
                       "FAILED") for i in range(2)], []),
        ("one_killed", [
            StatusInfo(None, Path("to/somewhere_0"), 0, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED"),
            StatusInfo(None, Path("to/somewhere_1"), 1, drmaa2.JobInfo({"terminating_signal": "0"}), drmaa2.JobState(8),
                       "FAILED")],
            [StatusInfo(None, Path("to/somewhere_0"), 0, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                        drmaa2.JobState(8), "FAILED")])
    ])
    def test_filter_killed_jobs(self, name, failed_jobs, result):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = create_js(working_directory, cluster_output_dir)
            killed_jobs = js.filter_killed_jobs(failed_jobs)
            self.assertEqual(killed_jobs, result)

    def test_resubmit_jobs(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            input_path, output_paths, _, slices = setup_data_files(working_directory, cluster_output_dir)
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters(slices)

            js = create_js(working_directory, cluster_output_dir)
            js.jobscript = setup_runner_script(working_directory)
            js.jobscript_args = [str(setup_jobscript(working_directory)), "--input-path", str(input_path)]
            js.memory = "4G"
            js.cores = 6
            js.job_name = "test_resubmit_jobs"
            js.scheduler_mode = processing_mode
            js.output_paths = output_paths
            js.job_history = {0: {
                0: StatusInfo(None, output_paths[0], 0, drmaa2.JobInfo({"terminating_signal": "0"}),
                              drmaa2.JobState(8), "FAILED"),
                1: StatusInfo(None, output_paths[1], 1, drmaa2.JobInfo({"terminating_signal": "0"}),
                              drmaa2.JobState(7), "DONE"),
                2: StatusInfo(None, output_paths[2], 2, drmaa2.JobInfo({"terminating_signal": "0"}),
                              drmaa2.JobState(8), "FAILED"),
                3: StatusInfo(None, output_paths[3], 3, drmaa2.JobInfo({"terminating_signal": "0"}),
                              drmaa2.JobState(7), "DONE")}}

            js.job_completion_status = {"0": False, "1": True, "2": False, '3': True}

            success = js.resubmit_jobs([0, 2])
            self.assertTrue(success)
            resubmitted_output_paths = [output_paths[i] for i in [0, 2]]
            for output in resubmitted_output_paths:
                self.assertTrue(output.is_file())

    @parameterized.expand([
        ("all_success", False, [
            StatusInfo(None, Path(), i, drmaa2.JobInfo({"terminating_signal": "0"}), drmaa2.JobState(7),
                       "DONE") for i in range(4)], {str(i): True for i in range(4)}, False, None, True, False),
        ("all_failed_do_not_allow", False, [
            StatusInfo(None, Path(), i, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED") for i in range(4)], {str(i): False for i in range(4)},
            False, None, False, True),
        ("all_failed_do_allow", True, [
            StatusInfo(None, Path(), i, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED") for i in range(4)],
            {str(i): False for i in range(4)}, True, [0, 1, 2, 3], True, False),
        ("some_failed_do_allow", True, [
            StatusInfo(None, Path(), 0, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED"),
            StatusInfo(None, Path(), 1, drmaa2.JobInfo({"terminating_signal": "0"}), drmaa2.JobState(7),
                       "DONE"),
            StatusInfo(None, Path(), 2, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED"),
            StatusInfo(None, Path(), 3, drmaa2.JobInfo({"terminating_signal": "0"}), drmaa2.JobState(7),
                       "DONE")], {"0": False, "1": True, "2": False, "3": True}, True, [0, 2], True, False),
        ("some_failed_do_not_allow", False, [
            StatusInfo(None, Path(), 0, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED"),
            StatusInfo(None, Path(), 1, drmaa2.JobInfo({"terminating_signal": "0"}),
                       drmaa2.JobState(7), "DONE"),
            StatusInfo(None, Path(), 2, drmaa2.JobInfo({"terminating_signal": "SIGKILL"}),
                       drmaa2.JobState(8), "FAILED"),
            StatusInfo(None, Path(), 3, drmaa2.JobInfo({"terminating_signal": "0"}),
                       drmaa2.JobState(7), "DONE")],
            {"0": False, "1": True, "2": False, "3": True}, True, [0, 2], True, False)
    ])
    def test_rerun_killed_jobs(self, name, allow_all_failed, job_history, job_completion_status, runs, indices,
                               expected_success, raises_error):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            input_path, output_paths, _, slices = setup_data_files(working_directory, cluster_output_dir)
            processing_mode = SimpleProcessingMode()
            processing_mode.set_parameters(slices)
            js = create_js(working_directory, cluster_output_dir)
            js.jobscript = setup_runner_script(working_directory)
            js.jobscript_args = [str(setup_jobscript(working_directory)), "--input-path", str(input_path)]
            js.memory = "4G"
            js.cores = 6
            js.job_name = "test_resubmit_jobs"
            js.scheduler_mode = processing_mode
            for output_path, status_info in zip(output_paths, job_history):
                status_info.output = output_path
            js.job_history = {0: {i: job_history[i] for i in range(4)}}
            js.job_completion_status = job_completion_status
            js.output_paths = output_paths

            if raises_error:
                with self.assertRaises(RuntimeError) as context:
                    js.rerun_killed_jobs(allow_all_failed)
                self.assertTrue("All jobs failed. job_history: " in str(context.exception))
                self.assertEqual(js.batch_number, 0)
                return

            success = js.rerun_killed_jobs(allow_all_failed)
            self.assertEqual(success, expected_success)
            self.assertEqual(js.output_paths, output_paths)
            if runs:
                self.assertEqual(js.batch_number, 1)
                resubmitted_output_paths = [output_paths[i] for i in indices]
                for output in resubmitted_output_paths:
                    self.assertTrue(output.is_file())
            else:
                self.assertEqual(js.batch_number, 0)


if __name__ == '__main__':
    unittest.main()
