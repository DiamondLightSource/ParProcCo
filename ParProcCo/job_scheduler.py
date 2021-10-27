from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import drmaa2

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface


@dataclass
class StatusInfo:
    '''Class for keeping track of job status.'''
    job: drmaa2.Job
    output_path: Path
    jobscript_args: List
    i: int
    info: Optional[drmaa2.JobInfo] = None
    state: Optional[drmaa2.JobState] = None
    final_state: Optional[str] = None


class JobScheduler:

    def __init__(self, working_directory: Union[Path, str], cluster_output_dir: Union[Path, str], project: str,
                 queue: str, cluster_resources: Optional[dict[str,str]] = None, timeout: timedelta = timedelta(hours=2)):
        """JobScheduler can be used for cluster job submissions"""
        self.batch_number = 0
        self.cluster_output_dir = Path(cluster_output_dir)
        self.job_completion_status: Dict[str, bool] = {}
        self.job_history: Dict[int, Dict[int, StatusInfo]] = {}
        self.jobscript: Path
        self.jobscript_args: List
        self.output_paths: List[Path] = []
        self.project = self.check_project_list(project)
        self.queue = self.check_queue_list(queue)
        self.start_time = datetime.now()
        self.status_infos: List[StatusInfo]
        self.timeout = timeout
        self.working_directory = Path(working_directory)
        self.resources: Dict[str, str] = {}
        if cluster_resources:
            self.resources.update(cluster_resources)

    def check_queue_list(self, queue: str) -> str:
        if not queue:
            raise ValueError(f"queue must be non-empty string")
        queue = queue.lower()
        with os.popen('qconf -sql') as q_proc:
            q_name_list = q_proc.read().split()
        if queue in q_name_list:
            return queue
        else:
            raise ValueError(f"queue {queue} not in queue list: {q_name_list}\n")

    def check_project_list(self, project: str) -> str:
        if not project:
            raise ValueError(f"project must be non-empty string")
        with os.popen('qconf -sprjl') as prj_proc:
            prj_name_list = prj_proc.read().split()
        if project in prj_name_list:
            return project
        else:
            raise ValueError(f"{project} must be in list of project names: {prj_name_list}\n")

    def check_jobscript(self, jobscript: Path) -> Path:
        if not jobscript.is_file():
            raise FileNotFoundError(f"{jobscript} does not exist\n")

        if not (os.access(jobscript, os.R_OK) and os.access(jobscript, os.X_OK)):
            raise PermissionError(f"{jobscript} must be readable and executable by user\n")

        try:
            js = jobscript.open()
            js.close()
        except IOError:
            logging.error(f"{jobscript} cannot be opened\n")
            raise

        else:
            return jobscript

    def get_output_paths(self) -> List[Path]:
        return self.output_paths

    def get_success(self) -> bool:
        return all(self.job_completion_status.values())

    def timestamp_ok(self, output: Path) -> bool:
        mod_time = datetime.fromtimestamp(output.stat().st_mtime)
        if mod_time > self.start_time:
            return True
        return False

    def run(self, scheduler_mode: SchedulerModeInterface, jobscript: Path, memory: str = "4G", cores: int = 6,
            jobscript_args: Optional[List] = None, job_name: str = "ParProcCo_job") -> bool:
        self.jobscript = jobscript
        job_indices = list(range(scheduler_mode.number_jobs))
        if jobscript_args is None:
            jobscript_args = []
        self.jobscript_args = jobscript_args
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {str(i): False for i in range(scheduler_mode.number_jobs)}
        self.jobscript = self.check_jobscript(jobscript)
        session = drmaa2.JobSession()  # Automatically destroyed when it is out of scope
        self._run_jobs(session, scheduler_mode, jobscript, job_indices, memory, cores, jobscript_args, job_name)
        self._wait_for_jobs(session)
        self._report_job_info()
        return self.get_success()

    def _run_jobs(self, session: drmaa2.JobSession, scheduler_mode: SchedulerModeInterface, jobscript: Path,
                  job_indices: List[int], memory: str, cores: int, jobscript_args: List, job_name: str) -> None:
        logging.debug(f"Running jobs on cluster for jobscript {jobscript} and args {jobscript_args}")
        try:
            # Run all input paths in parallel:
            self.status_infos = []
            for i in job_indices:
                template = self._create_template(jobscript, scheduler_mode, i, memory, cores, jobscript_args, job_name)
                logging.debug(f"Submitting drmaa job with jobscript {jobscript} and args {template.args}")
                job = session.run_job(template)
                self.status_infos.append(StatusInfo(job, Path(template.output_path), jobscript_args, i))
                logging.debug(f"drmaa job for jobscript {jobscript} and args {template.args} has been submitted with id {job.id}")
        except drmaa2.Drmaa2Exception:
            logging.error(f"Drmaa exception", exc_info=True)
            raise
        except Exception:
            logging.error(f"Unknown error occurred running drmaa job", exc_info=True)
            raise

    def _create_template(self, jobscript: Path, scheduler_mode: SchedulerModeInterface, i: int, memory: str, cores: int,
                         jobscript_args: List, job_name: str) -> drmaa2.JobTemplate:
        if not self.cluster_output_dir.is_dir():
            logging.debug(f"Making directory {self.cluster_output_dir}")
            self.cluster_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(f"Directory {self.cluster_output_dir} already exists")

        error_dir = self.cluster_output_dir / "error_logs"
        if not error_dir.is_dir():
            logging.debug(f"Making directory {error_dir}")
            error_dir.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(f"Directory {error_dir} already exists")

        output_fp, std_out_fp, err_fp = scheduler_mode.generate_output_paths(self.cluster_output_dir, error_dir, i)
        self.output_paths.append(Path(output_fp))
        args = scheduler_mode.generate_args(i, memory, cores, jobscript_args, output_fp)
        print(f"creating template with jobscript: {str(jobscript)} and args: {args}")

        self.resources["m_mem_free"] = memory
        jt = drmaa2.JobTemplate({
            "job_name": job_name,
            "job_category": self.project,
            "remote_command": str(jobscript),
            "min_slots": cores,
            "args": args,
            "resource_limits": self.resources,
            "working_directory": str(self.working_directory),
            "output_path": std_out_fp,
            "error_path": err_fp,
            "queue_name": self.queue,
            "implementation_specific": {
                "uge_jt_pe": f"smp",
            },
        })
        return jt

    def _wait_for_jobs(self, session: drmaa2.JobSession) -> None:
        try:
            job_list = [status_info.job for status_info in self.status_infos]
            # Wait for jobs to start (timeout shouldn't include queue time)
            job_list_str = ", ".join([str(job.id) for job in job_list])
            logging.info(f"Waiting for jobs to start: {job_list_str}")
            session.wait_all_started(job_list)
            logging.info(f"Jobs started, waiting for jobs: {job_list_str}")
            session.wait_all_terminated(job_list, int(round(self.timeout.total_seconds())))
            jobs_running = False
            for job in job_list:
                if job.get_state()[0] == drmaa2.JobState.RUNNING:
                    logging.info(f"Job {job.id} timed out. Terminating job now.")
                    jobs_running = True
                    job.terminate()
                    print(f"terminating job {job.id}")
            if jobs_running:
                # Termination takes some time, wait a max of 2 mins
                session.wait_all_terminated(job_list, 120)
        except drmaa2.Drmaa2Exception:
            logging.error(f"Drmaa exception", exc_info=True)
        except Exception:
            logging.error(f"Unknown error occurred running drmaa job", exc_info=True)

    def _report_job_info(self) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for status_info in self.status_infos:
            logging.debug(f"Retrieving info for drmaa job {status_info.job.id}")
            try:
                status_info.state = status_info.job.get_state()[0]  # Returns job state and job substate (always seems to be None)
                status_info.info = status_info.job.get_info()

            except Exception:
                logging.error(f"Failed to get job information for job {status_info.job.id}", exc_info=True)
                raise

            # Check job states against expected possible options:
            if status_info.state == drmaa2.JobState.UNDETERMINED:  # Lost contact?
                status_info.final_state = "UNDETERMINED"
                logging.warning(f"Job state undetermined for job {status_info.job.id}. job info: {status_info.info}")

            elif status_info.state == drmaa2.JobState.FAILED:
                status_info.final_state = "FAILED"
                logging.error(
                    f"drmaa job {status_info.job.id} processing file filename failed."
                    f" Terminating signal: {status_info.info.terminating_signal}."
                )

            elif not status_info.output_path.is_file():
                status_info.final_state = "NO_OUTPUT"
                logging.error(
                    f"drmaa job {status_info.job.id} with args {status_info.jobscript_args} has not created"
                    f" output file {status_info.output_path}"
                    f" Terminating signal: {status_info.info.terminating_signal}."
                )

            elif not self.timestamp_ok(status_info.output_path):
                status_info.final_state = "OLD_OUTPUT_FILE"
                logging.error(
                    f"drmaa job {status_info.job.id} with args {status_info.jobscript_args} has not created"
                    f" a new output file {status_info.output_path}"
                    f"Terminating signal: {status_info.info.terminating_signal}."
                )

            elif status_info.state == drmaa2.JobState.DONE:
                self.job_completion_status[str(status_info.i)] = True
                status_info.final_state = "SUCCESS"
                logging.info(
                    f"Job {status_info.job.id} with with args {status_info.jobscript_args} completed"
                    f" successfully after {status_info.info.wallclock_time}."
                    f" CPU time={timedelta(seconds=float(status_info.info.cpu_time))}, slots={status_info.info.slots}"
                )
            else:
                status_info.final_state = "UNSPECIFIED"
                logging.error(
                    f"Unexpected job state for job {status_info.job.id} with args {status_info.jobscript_args}, job info: {status_info.info}"
                )

            self.job_history[self.batch_number][status_info.job.id] = status_info

    def resubmit_jobs(self, scheduler_mode: SchedulerModeInterface, jobscript: Path, job_indices: List[int],
                      memory: str, cores: int, jobscript_args: List, job_name: str) -> bool:
        self.batch_number += 1
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {str(i): False for i in job_indices}
        self._run_and_monitor(scheduler_mode, jobscript, job_indices, memory, cores, jobscript_args, job_name)
        return self.get_success()

    def filter_killed_jobs(self, jobs: List[drmaa2.Job]) -> List[drmaa2.Job]:
        killed_jobs = [job for job in jobs if job.info.terminating_signal == "SIGKILL"]
        return killed_jobs

    def rerun_killed_jobs(self, scheduler_mode: SchedulerModeInterface, jobscript: Path, memory: str, cores: int,
                          jobscript_args: List, job_name: str, allow_all_failed: bool = False):
        job_history = self.job_history
        if all(self.job_completion_status.values()):
            warnings.warn("No failed jobs")
            return True
        elif allow_all_failed or any(self.job_completion_status.values()):
            failed_jobs = [job_info for job_info in job_history[0].values() if job_info.final_state != "SUCCESS"]
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            if not all(job.jobscript_args == killed_jobs[0].jobscript_args for job in killed_jobs):
                raise RuntimeError(f"jobscript_args in killed_jobs must all be the same\n")
            killed_jobs_indices = [job.i for job in killed_jobs]
            success = self.resubmit_jobs(scheduler_mode, jobscript, killed_jobs_indices, memory, cores, jobscript_args, job_name)
            return success

        raise RuntimeError(f"All jobs failed\n")
