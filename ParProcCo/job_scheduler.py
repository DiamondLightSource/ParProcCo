from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import drmaa2

from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable


@dataclass
class StatusInfo:
    """Class for keeping track of job status."""

    job: drmaa2.Job
    output_path: Path
    i: int
    info: Optional[drmaa2.JobInfo] = None
    state: Optional[drmaa2.JobState] = None
    final_state: Optional[str] = None


class JobScheduler:
    def __init__(
        self,
        working_directory: Optional[Union[Path, str]],
        cluster_output_dir: Optional[Union[Path, str]],
        project: str,
        queue: str,
        cluster_resources: Optional[dict[str, str]] = None,
        timeout: timedelta = timedelta(hours=2),
    ):
        """JobScheduler can be used for cluster job submissions"""
        self.batch_number = 0
        self.cluster_output_dir: Optional[Path] = (
            Path(cluster_output_dir) if cluster_output_dir else None
        )
        self.job_completion_status: Dict[str, bool] = {}
        self.job_history: Dict[int, Dict[int, StatusInfo]] = {}
        self.jobscript: Path
        self.job_env: Optional[Dict[str, str]] = None
        self.jobscript_args: List
        self.output_paths: List[Path] = []
        self.project = self.check_project_list(project)
        self.queue = self.check_queue_list(queue)
        self.start_time = datetime.now()
        self.status_infos: List[StatusInfo]
        self.timeout = timeout
        self.working_directory = (
            Path(working_directory)
            if working_directory
            else (self.cluster_output_dir if self.cluster_output_dir else Path.home())
        )
        self.resources: Dict[str, str] = {}
        if cluster_resources:
            self.resources.update(cluster_resources)
        self.scheduler_mode: SchedulerModeInterface
        self.memory: str
        self.cores: int
        self.job_name: str

    def check_queue_list(self, queue: str) -> str:
        if not queue:
            raise ValueError("queue must be non-empty string")
        queue = queue.lower()
        with os.popen("qconf -sql") as q_proc:
            q_name_list = q_proc.read().split()
        if queue in q_name_list:
            return queue
        else:
            raise ValueError(f"queue {queue} not in queue list: {q_name_list}\n")

    def check_project_list(self, project: str) -> str:
        if not project:
            raise ValueError("project must be non-empty string")
        with os.popen("qconf -sprjl") as prj_proc:
            prj_name_list = prj_proc.read().split()
        if project in prj_name_list:
            return project
        else:
            raise ValueError(
                f"{project} must be in list of project names: {prj_name_list}\n"
            )

    def get_output_paths(self) -> List[Path]:
        return self.output_paths

    def get_success(self) -> bool:
        return all(self.job_completion_status.values())

    def timestamp_ok(self, output: Path) -> bool:
        mod_time = datetime.fromtimestamp(output.stat().st_mtime)
        if mod_time > self.start_time:
            return True
        return False

    def run(
        self,
        scheduler_mode: SchedulerModeInterface,
        jobscript: Path,
        job_env: Optional[Dict[str, str]],
        memory: str = "4G",
        cores: int = 6,
        jobscript_args: Optional[List] = None,
        job_name: str = "ParProcCo_job",
    ) -> bool:
        self.jobscript = check_jobscript_is_readable(jobscript)
        self.job_env = job_env
        self.scheduler_mode = scheduler_mode
        self.memory = memory
        self.cores = cores
        self.job_name = job_name
        job_indices = list(range(scheduler_mode.number_jobs))
        if jobscript_args is None:
            jobscript_args = []
        self.jobscript_args = jobscript_args
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {
            str(i): False for i in range(scheduler_mode.number_jobs)
        }
        self.output_paths.clear()
        return self._run_and_monitor(job_indices)

    def _run_and_monitor(self, job_indices: List[int]) -> bool:
        session = drmaa2.JobSession()  # Automatically destroyed when it is out of scope
        self._run_jobs(session, job_indices)
        self._wait_for_jobs(session)
        self._report_job_info()
        return self.get_success()

    def _run_jobs(self, session: drmaa2.JobSession, job_indices: List[int]) -> None:
        logging.debug(
            f"Running jobs on cluster for jobscript {self.jobscript} and args {self.jobscript_args}"
        )
        try:
            # Run all input paths in parallel:
            self.status_infos = []
            for i in job_indices:
                template = self._create_template(i)
                logging.debug(
                    f"Submitting drmaa job with jobscript {self.jobscript} and args {template.args}"
                )
                job = session.run_job(template)
                self.status_infos.append(StatusInfo(job, Path(template.output_path), i))
                logging.debug(
                    f"drmaa job for jobscript {self.jobscript} and args {template.args}"
                    f" has been submitted with id {job.id}"
                )
        except drmaa2.Drmaa2Exception:
            logging.error("Drmaa exception", exc_info=True)
            raise
        except Exception:
            logging.error("Unknown error occurred running drmaa job", exc_info=True)
            raise

    def _create_template(self, i: int) -> drmaa2.JobTemplate:
        if self.cluster_output_dir:
            if not self.cluster_output_dir.is_dir():
                logging.debug(f"Making directory {self.cluster_output_dir}")
                self.cluster_output_dir.mkdir(exist_ok=True, parents=True)
            else:
                logging.debug(f"Directory {self.cluster_output_dir} already exists")

            error_dir = self.cluster_output_dir / "cluster_logs"
        else:
            error_dir = self.working_directory / "cluster_logs"
        if not error_dir.is_dir():
            logging.debug(f"Making directory {error_dir}")
            error_dir.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(f"Directory {error_dir} already exists")

        output_fp, stdout_fp, stderr_fp = self.scheduler_mode.generate_output_paths(
            self.cluster_output_dir, error_dir, i, self.start_time
        )
        if output_fp and output_fp not in self.output_paths:
            self.output_paths.append(Path(output_fp))
        args = self.scheduler_mode.generate_args(
            i, self.memory, self.cores, self.jobscript_args, output_fp
        )
        logging.info(
            f"creating template with jobscript: {str(self.jobscript)} and args: {args}"
        )

        self.resources["m_mem_free"] = self.memory
        jt = drmaa2.JobTemplate(
            {
                "job_name": self.job_name,
                "job_category": self.project,
                "remote_command": str(self.jobscript),
                "min_slots": self.cores,
                "args": args,
                "resource_limits": self.resources,
                "working_directory": str(self.working_directory),
                "output_path": stdout_fp,
                "error_path": stderr_fp,
                "queue_name": self.queue,
                "implementation_specific": {
                    "uge_jt_pe": "smp",
                },
            }
        )
        job_env = dict(self.job_env) if self.job_env is not None else {}
        test_ppc_dir = os.getenv("TEST_PPC_DIR")
        if test_ppc_dir:
            logging.debug("Adding TEST_PPC_DIR=%s to environment", test_ppc_dir)
            job_env.update(TEST_PPC_DIR=test_ppc_dir)
        jt.job_environment = job_env
        return jt

    def _wait_for_jobs(self, session: drmaa2.JobSession) -> None:
        max_time = int(round(self.timeout.total_seconds()))
        check_time = min(120, max_time)  # 2 minutes or less
        start_wait = max(check_time, 60)  # wait at least 1 minute
        try:
            job_list = [status_info.job for status_info in self.status_infos]
            # Wait for jobs to start (timeout shouldn't include queue time)
            job_list_str = ", ".join([str(job.id) for job in job_list])
            logging.info(f"Waiting for jobs to start: {job_list_str}")
            jobs_left = len(job_list)
            while jobs_left > 0:
                jobs_left -= len(session.wait_all_started(job_list, start_wait))
                if jobs_left:
                    logging.info(f"Jobs left to start: {jobs_left}")
            logging.info(f"Jobs started, waiting for jobs: {job_list_str}")
            total_time = 0
            while total_time < max_time and job_list:
                session.wait_all_terminated(job_list, check_time)
                total_time += check_time
                jobs_remaining = []
                for job in job_list:
                    if job.get_state()[0] == drmaa2.JobState.RUNNING:
                        jobs_remaining.append(job)
                        logging.debug(f"Job {job.id} still running")
                job_list = jobs_remaining
                logging.info(
                    f"Jobs remaining = {len(jobs_remaining)} after {total_time}s"
                )

            jobs_remaining = []
            for job in job_list:
                if job.get_state()[0] == drmaa2.JobState.RUNNING:
                    logging.warning(f"Job {job.id} timed out. Terminating job now.")
                    jobs_remaining.append(job)
                    job.terminate()
            if jobs_remaining:
                # Termination takes some time, wait a max of 2 mins
                session.wait_all_terminated(jobs_remaining, 120)
                total_time += 120
                logging.info(
                    f"Jobs terminated = {len(jobs_remaining)} after {total_time}s"
                )
        except drmaa2.Drmaa2Exception:
            logging.error("Drmaa exception", exc_info=True)
        except Exception:
            logging.error("Unknown error occurred running drmaa job", exc_info=True)

    def _report_job_info(self) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for status_info in self.status_infos:
            logging.debug(f"Retrieving info for drmaa job {status_info.job.id}")
            try:
                status_info.state = status_info.job.get_state()[
                    0
                ]  # Returns job state and job substate (always seems to be None)
                status_info.info = status_info.job.get_info()
            except Exception:
                logging.error(
                    f"Failed to get job information for job {status_info.job.id}",
                    exc_info=True,
                )
                raise

            dispatch_time = status_info.info.dispatch_time
            if dispatch_time:
                try:
                    time_to_dispatch = dispatch_time - status_info.info.submission_time
                    wall_time = status_info.info.finish_time - dispatch_time
                except Exception:
                    logging.error(
                        f"Failed to get job submission time statistics for job {status_info.job.id}"
                    )
                    raise
            else:
                time_to_dispatch = "n/a"
                wall_time = "n/a"

            # Check job states against expected possible options:
            if status_info.state == drmaa2.JobState.UNDETERMINED:  # Lost contact?
                status_info.final_state = "UNDETERMINED"
                logging.warning(
                    f"Job state undetermined for job {status_info.job.id}. job info: {status_info.info}."
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )

            elif status_info.state == drmaa2.JobState.FAILED:
                status_info.final_state = "FAILED"
                logging.error(
                    f"drmaa job {status_info.job.id} failed. Terminating signal: {status_info.info.terminating_signal}."
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )

            elif not status_info.output_path.is_file():
                status_info.final_state = "NO_OUTPUT"
                logging.error(
                    f"drmaa job {status_info.job.id} with args {self.jobscript_args} has not created"
                    f" output file {status_info.output_path}"
                    f" Terminating signal: {status_info.info.terminating_signal}."
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )

            elif not self.timestamp_ok(status_info.output_path):
                status_info.final_state = "OLD_OUTPUT_FILE"
                logging.error(
                    f"drmaa job {status_info.job.id} with args {self.jobscript_args} has not created"
                    f" a new output file {status_info.output_path}"
                    f" Terminating signal: {status_info.info.terminating_signal}."
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )

            elif status_info.state == drmaa2.JobState.DONE:
                self.job_completion_status[str(status_info.i)] = True
                status_info.final_state = "SUCCESS"
                logging.info(
                    f"Job {status_info.job.id} with args {self.jobscript_args} completed."
                    f" CPU time: {timedelta(seconds=float(status_info.info.cpu_time))}; Slots: {status_info.info.slots}"
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )
            else:
                status_info.final_state = "UNSPECIFIED"
                logging.error(
                    f"Unexpected job state for job {status_info.job.id}"
                    f" with args {self.jobscript_args}, job info: {status_info.info}"
                    f" Dispatch time: {time_to_dispatch}; Wall time: {wall_time}."
                )

            self.job_history[self.batch_number][status_info.i] = status_info

    def resubmit_jobs(self, job_indices: List[int]) -> bool:
        self.batch_number += 1
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {str(i): False for i in job_indices}
        logging.info(f"Resubmitting jobs with job_indices: {job_indices}")
        return self._run_and_monitor(job_indices)

    def filter_killed_jobs(self, jobs: List[StatusInfo]) -> List[StatusInfo]:
        killed_jobs = [
            job
            for job in jobs
            if job.info is not None and job.info.terminating_signal == "SIGKILL"
        ]
        return killed_jobs

    def rerun_killed_jobs(self, allow_all_failed: bool = False):
        logging.info("Rerunning killed jobs")
        job_history = self.job_history
        if all(self.job_completion_status.values()):
            logging.warning("No failed jobs to rerun")
            return True
        elif allow_all_failed or any(self.job_completion_status.values()):
            failed_jobs = [
                job_info
                for job_info in job_history[0].values()
                if job_info.final_state != "SUCCESS"
            ]
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            killed_jobs_indices = [job.i for job in killed_jobs]
            logging.info(
                f"Total failed_jobs: {len(failed_jobs)}."
                f" Total killed_jobs: {len(killed_jobs)}"
            )
            if killed_jobs_indices:
                return self.resubmit_jobs(killed_jobs_indices)
            return True

        raise RuntimeError(f"All jobs failed. job_history: {job_history}\n")
