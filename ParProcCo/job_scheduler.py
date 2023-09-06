from __future__ import annotations

import logging
import requests
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable, get_slurm_token, get_user, get_ppc_dir
from models.slurm_rest import (
    JobProperties,
    JobsResponse,
    JobResponseProperties,
    JobSubmission,
    JobSubmissionResponse,
)

# WIP: Migrating from drmaa2 to slurm as in https://github.com/DiamondLightSource/python-zocalo


class SLURMSTATE(str, Enum):
    # The following are states from https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
    BOOT_FAIL = (
        "BOOT_FAIL"  # Job terminated due to launch failure, typically due to a hardware failure
        # (e.g. unable to boot the node or block and the job can not be requeued).
    )
    CANCELLED = (
        "CANCELLED"  # Job was explicitly cancelled by the user or system administrator.
        # The job may or may not have been initiated
    )
    COMPLETED = "COMPLETED"  # Job has terminated all processes on all nodes with an exit code of zero
    CONFIGURING = (
        "CONFIGURING"  # Job has been allocated resources, but are waiting for them to become ready for
        # use (e.g. booting)
    )
    COMPLETING = "COMPLETING"  # Job is in the process of completing. Some processes on some nodes may still be active
    DEADLINE = "DEADLINE"  # Job terminated on deadline
    FAILED = (
        "FAILED"  # Job terminated with non-zero exit code or other failure condition
    )
    NODE_FAIL = (
        "NODE_FAIL"  # Job terminated due to failure of one or more allocated nodes
    )
    OUT_OF_MEMORY = "OUT_OF_MEMORY"  # Job experienced out of memory error
    PENDING = "PENDING"  # Job is awaiting resource allocation
    PREEMPTED = "PREEMPTED"  # Job terminated due to preemption
    RUNNING = "RUNNING"  # Job currently has an allocation
    RESV_DEL_HOLD = "RESV_DEL_HOLD"  # Job is held
    REQUEUE_FED = "REQUEUE_FED"  # Job is being requeued by a federation
    REQUEUE_HOLD = "REQUEUE_HOLD"  # Held job is being requeued
    REQUEUED = "REQUEUED"  # Completing job is being requeued
    RESIZING = "RESIZING"  # Job is about to change size
    REVOKED = "REVOKED"  # Sibling was removed from cluster due to other cluster starting the job
    SIGNALING = "SIGNALING"  # Job is being signaled
    SPECIAL_EXIT = (
        "SPECIAL_EXIT"  # The job was requeued in a special state. This state can be set by users, typically in
        # EpilogSlurmctld, if the job has terminated with a particular exit value
    )
    STAGE_OUT = "STAGE_OUT"  # Job is staging out files
    STOPPED = (
        "STOPPED"  # Job has an allocation, but execution has been stopped with SIGSTOP signal. CPUS have been
        # retained by this job
    )
    SUSPENDED = (
        "SUSPENDED"  # Job has an allocation, but execution has been suspended and CPUs have been released for
        # other jobs
    )
    TIMEOUT = "TIMEOUT"  # Job terminated upon reaching its time limit
    NO_OUTPUT = "NO_OUTPUT"  # Custom state. No output file found
    OLD_OUTPUT_FILE = "OLD_OUTPUT_FILE"  # Custom state. Output file has not been updated since job started.


class STATEGROUP(set, Enum):
    OUTOFTIME = {SLURMSTATE.TIMEOUT, SLURMSTATE.DEADLINE}
    FINISHED = {
        SLURMSTATE.COMPLETED,
        SLURMSTATE.FAILED,
        SLURMSTATE.TIMEOUT,
        SLURMSTATE.DEADLINE,
    }
    COMPUTEISSUE = {
        SLURMSTATE.BOOT_FAIL,
        SLURMSTATE.NODE_FAIL,
        SLURMSTATE.OUT_OF_MEMORY,
    }
    ENDED = {
        SLURMSTATE.COMPLETED,
        SLURMSTATE.FAILED,
        SLURMSTATE.TIMEOUT,
        SLURMSTATE.DEADLINE,
    }
    REQUEUEABLE = {
        SLURMSTATE.CONFIGURING,
        SLURMSTATE.RUNNING,
        SLURMSTATE.STOPPED,
        SLURMSTATE.SUSPENDED,
    }
    STARTING = {
        SLURMSTATE.PENDING,
        SLURMSTATE.REQUEUED,
        SLURMSTATE.RESIZING,
        SLURMSTATE.SUSPENDED,
        SLURMSTATE.CONFIGURING,
    }


@dataclass
class StatusInfo:
    """Class for keeping track of job status."""

    output_path: Path
    i: int
    start_time: int
    current_state: Optional[SLURMSTATE] = None
    slots: Optional[int] = None
    time_to_dispatch: Optional[int] = None
    wall_time: Optional[int] = None
    final_state: Optional[SLURMSTATE] = None


class JobScheduler:
    def __init__(
        self,
        url: str,
        working_directory: Optional[Union[Path, str]],
        cluster_output_dir: Optional[Union[Path, str]],
        partition: str,
        timeout: timedelta = timedelta(hours=2),
        version: str = "v0.0.38",
        user_name: Optional[str] = None,
        user_token: Optional[str] = None,
    ):
        """JobScheduler can be used for cluster job submissions"""
        self.batch_number = 0
        self.cluster_output_dir: Optional[Path] = (
            Path(cluster_output_dir) if cluster_output_dir else None
        )
        self.job_completion_status: Dict[str, bool] = {}
        self.job_history: Dict[int, Dict[int, StatusInfo]] = {}
        self.jobscript_path: Path
        self.jobscript_command: str
        self.job_env: Dict[str, str]
        self.jobscript_args: List
        self.output_paths: List[Path] = []
        self.start_time = datetime.now()
        self.status_infos: Dict[int, StatusInfo]
        self.timeout = timeout
        self.working_directory = (
            Path(working_directory)
            if working_directory
            else (self.cluster_output_dir if self.cluster_output_dir else Path.home())
        )
        self.partition = partition
        self.scheduler_mode: SchedulerModeInterface
        self.memory: int
        self.cores: int
        self.job_name: str
        self._url = url
        self._version = version
        self._slurm_endpoint_prefix = f"slurm/{self._version}"
        self._session = requests.Session()
        self._session.headers["X-SLURM-USER-NAME"] = (
            user_name if user_name else get_user()
        )
        self.user = user_name if user_name else get_user()
        self.token = user_token if user_token else get_slurm_token()
        self._session.headers["X-SLURM-USER-TOKEN"] = (
            user_token if user_token else get_slurm_token()
        )

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        response = self._session.get(
            f"{self._url}/{endpoint}", params=params, timeout=timeout
        )
        response.raise_for_status()
        return response

    def _prepare_request(
        self, data: BaseModel
    ) -> tuple[str, dict[str, str]] | tuple[None, None]:
        if data is None:
            return None, None
        return data.model_dump_json(exclude_defaults=True), {
            "X-SLURM-USER-NAME": self.user,
            "X-SLURM-USER-TOKEN": self.token,
            "Content-Type": "application/json",
        }

    def _post(self, data: BaseModel, endpoint):
        url = self._url + "/" + endpoint
        jdata, headers = self._prepare_request(data)
        resp = requests.post(url, data=jdata, headers=headers)
        return resp

    def delete(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        response = self._session.delete(
            f"{self._url}/{endpoint}", params=params, timeout=timeout
        )
        response.raise_for_status()
        return response

    def get_jobs(self) -> JobsResponse:
        endpoint = f"{self._slurm_endpoint_prefix}/jobs"
        response = self.get(endpoint)
        return JobsResponse.model_validate(response.json())

    def get_job_response(self, job_id: int) -> JobsResponse:
        endpoint = f"{self._slurm_endpoint_prefix}/job/{job_id}"
        response = self.get(endpoint)
        return JobsResponse.model_validate(response.json())

    def get_job(self, job_id: int) -> JobResponseProperties:
        ji = self.get_job_response(job_id)
        if ji.jobs:
            n = len(ji.jobs)
            if n == 1:
                return ji.jobs[0]
            if n > 1:
                raise ValueError("Multiple jobs returned {ji.jobs}")
        raise ValueError("No job info found for job id {job_id}")

    def update_status_infos(
        self, job_id: int, job_info: JobResponseProperties, state: SLURMSTATE | None
    ) -> None:
        try:
            tres_alloc_str = job_info.tres_alloc_str
            slots = (
                int((tres_alloc_str.split(",")[1]).split("=")[1])
                if tres_alloc_str
                else None
            )

        except Exception:
            logging.warning(
                f"Failed to get slots for job {job_id}; setting slots to 0. Job info: {job_info}"
            )
            slots = None

        dispatch_time = job_info.start_time
        submit_time = job_info.submit_time
        end_time = job_info.end_time
        if dispatch_time and submit_time and end_time:
            time_to_dispatch = dispatch_time - submit_time
            wall_time = end_time - dispatch_time

        else:
            time_to_dispatch = None
            wall_time = None
        self.status_infos[job_id].slots = slots
        self.status_infos[job_id].time_to_dispatch = time_to_dispatch
        self.status_infos[job_id].wall_time = wall_time
        self.status_infos[job_id].current_state = state
        logging.info(f"Updating current state of {job_id} to {state}")

    def submit_job(self, job_submission: JobSubmission) -> JobSubmissionResponse:
        endpoint = f"{self._slurm_endpoint_prefix}/job/submit"
        response = self._post(data=job_submission, endpoint=endpoint)
        return JobSubmissionResponse.model_validate(response.json())

    def cancel_job(self, job_id: int) -> JobsResponse:
        endpoint = f"{self._slurm_endpoint_prefix}/job/{job_id}"
        response = self.delete(endpoint)
        return JobsResponse.model_validate(response.json())

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
        jobscript_path: Path,
        job_env: Optional[Dict[str, str]] = None,
        memory: int = 4000,
        cores: int = 6,
        jobscript_args: Optional[List] = None,
        job_name: str = "ParProcCo_job",
    ) -> bool:
        self.jobscript_path = check_jobscript_is_readable(jobscript_path)
        self.scheduler_mode = scheduler_mode
        self.job_env = job_env if job_env else {"ParProcCo": "0"}
        test_ppc_dir = get_ppc_dir()
        if test_ppc_dir:
            self.job_env.update(TEST_PPC_DIR=test_ppc_dir)
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
        self._run_jobs(job_indices)
        self._wait_for_jobs()
        self._report_job_info()
        return self.get_success()

    def _run_jobs(self, job_indices: List[int]) -> None:
        logging.debug(
            f"Running jobs on cluster for jobscript {self.jobscript_path} and args {self.jobscript_args}"
        )
        try:
            self.status_infos = {}
            for i in job_indices:
                submission = self.make_job_submission(i)
                resp = self.submit_job(submission)
                if resp.job_id is None:
                    raise ValueError("Job submission failed", resp.errors)
                self.status_infos[resp.job_id] = StatusInfo(
                    Path(submission.job.standard_output),
                    i,
                    int(time.time()),
                )
                logging.debug(
                    f"job array for jobscript {self.jobscript_path} and args {submission.job.argv}"
                    f" has been submitted with id {resp.job_id}"
                )
        except Exception:
            logging.error("Unknown error occurred running job", exc_info=True)
            raise

    def make_job_submission(self, i: int, job=None, jobs=None) -> JobSubmission:
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
        self.jobscript_command = " ".join(
            [f"#!/bin/bash\n{self.jobscript_path}", *args]
        )
        logging.info(f"creating submission with command: {self.jobscript_command}")
        job = JobProperties(
            name=self.job_name,
            partition=self.partition,
            cpus_per_task=self.cores,
            environment=self.job_env,
            memory_per_cpu=self.memory,
            current_working_directory=str(self.working_directory),
            standard_output=stdout_fp,
            standard_error=stderr_fp,
            get_user_environment="10L",
        )

        return JobSubmission(script=self.jobscript_command, job=job)

    def fetch_and_update_state(
        self, job_id: int
    ) -> Tuple[JobResponseProperties, SLURMSTATE | None]:
        ji = self.get_job(job_id)
        state = ji.job_state
        self.update_status_infos(job_id, ji, state)
        return ji, state

    def wait_all_jobs_terminated(self, job_ids: List[int], check_time: int):
        wait_until = time.time() + check_time
        remaining_jobs = job_ids.copy()
        terminated_jobs = []
        while len(remaining_jobs) > 0 and time.time() <= wait_until:
            for job_id in remaining_jobs:
                ji, state = self.fetch_and_update_state(job_id)
                if state in STATEGROUP.ENDED:
                    remaining_jobs.remove(job_id)
                    terminated_jobs.append((job_id, ji, state))
            time.sleep(min(60, check_time))

    def wait_all_jobs_started(self, job_ids: List[int], check_time: int):
        wait_until = time.time() + check_time
        remaining_jobs = job_ids.copy()
        started_jobs = []
        while len(remaining_jobs) > 0 and time.time() <= wait_until:
            for job_id in remaining_jobs:
                _, state = self.fetch_and_update_state(job_id)
                if state not in STATEGROUP.STARTING:
                    started_jobs.append(job_id)
                    remaining_jobs.remove(job_id)
        return started_jobs

    def _wait_for_jobs(
        self,
    ) -> None:
        max_time = int(round(self.timeout.total_seconds()))
        check_time = min(120, max_time)  # 2 minutes or less
        start_wait = max(check_time, 60)  # wait at least 1 minute
        try:
            job_list = list(self.status_infos)
            # Wait for jobs to start (timeout shouldn't include queue time)
            logging.info(f"Waiting for jobs to start: {job_list}")
            jobs_left = len(job_list)
            while jobs_left > 0:
                jobs_left -= len(self.wait_all_jobs_started(job_list, start_wait))
                if jobs_left:
                    logging.info(f"Jobs left to start: {jobs_left}")
            logging.info(f"Jobs started, waiting for jobs: {job_list}")
            total_time = 0
            while total_time < max_time and job_list:
                self.wait_all_jobs_terminated(job_list, check_time)
                total_time += check_time
                jobs_remaining = []
                for job_id in job_list:
                    ji, state = self.fetch_and_update_state(job_id)
                    if ji.job_state == SLURMSTATE.RUNNING:
                        jobs_remaining.append(job_id)
                        logging.debug(f"Job {job_id} still running")
                job_list = jobs_remaining
                logging.info(
                    f"Jobs remaining = {len(jobs_remaining)} after {total_time}s"
                )
            jobs_remaining = []
            for job_id in job_list:
                ji, state = self.fetch_and_update_state(job_id)
                if state == SLURMSTATE.RUNNING:
                    logging.warning(f"Job {job_id} timed out. Terminating job now.")
                    jobs_remaining.append(job_id)
                    self.cancel_job(job_id)
            if jobs_remaining:
                # Termination takes some time, wait a max of 2 mins
                self.wait_all_jobs_terminated(jobs_remaining, 120)
                total_time += 120
                logging.info(
                    f"Jobs terminated = {len(jobs_remaining)} after {total_time}s"
                )
        except Exception:
            logging.error("Unknown error occurred running slurm job", exc_info=True)

    def _report_job_info(self) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for job_id, status_info in self.status_infos.items():
            logging.debug(f"Retrieving info for slurm job {job_id}")

            # Check job states against expected possible options:
            state = status_info.current_state
            if state == SLURMSTATE.FAILED:
                status_info.final_state = SLURMSTATE.FAILED
                logging.error(
                    f"Slurm job {job_id} failed."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not status_info.output_path.is_file():
                status_info.final_state = SLURMSTATE.NO_OUTPUT
                logging.error(
                    f"Slurm job {job_id} with args {self.jobscript_args} has not created"
                    f" output file {status_info.output_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not self.timestamp_ok(status_info.output_path):
                status_info.final_state = SLURMSTATE.OLD_OUTPUT_FILE
                logging.error(
                    f"Slurm job {job_id} with args {self.jobscript_args} has not created"
                    f" a new output file {status_info.output_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif state == SLURMSTATE.COMPLETED:
                self.job_completion_status[str(status_info.i)] = True
                status_info.final_state = SLURMSTATE.COMPLETED
                if status_info.slots and status_info.wall_time:
                    cpu_time = timedelta(
                        seconds=float(status_info.wall_time * status_info.slots)
                    )
                else:
                    cpu_time = "n/a"
                logging.info(
                    f"Job {job_id} with args {self.jobscript_args} completed."
                    f" CPU time: {cpu_time}; Slots: {status_info.slots}"
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )
            else:
                status_info.final_state = state
                logging.error(
                    f"Job {job_id} ended with job state {status_info.final_state}"
                    f" Args {self.jobscript_args};"
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            self.job_history[self.batch_number][job_id] = status_info

    def resubmit_jobs(self, job_indices: List[int]) -> bool:
        self.batch_number += 1
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {str(i): False for i in job_indices}
        logging.info(f"Resubmitting jobs with job_indices: {job_indices}")
        return self._run_and_monitor(job_indices)

    def filter_killed_jobs(self, jobs: Dict[int, StatusInfo]) -> Dict[int, StatusInfo]:
        killed_jobs = {
            job_id: status_info
            for job_id, status_info in jobs.items()
            if status_info.current_state == SLURMSTATE.CANCELLED
        }
        return killed_jobs

    def rerun_killed_jobs(self, allow_all_failed: bool = False):
        logging.info("Rerunning killed jobs")
        job_history = self.job_history
        if all(self.job_completion_status.values()):
            logging.warning("No failed jobs to rerun")
            return True
        elif allow_all_failed or any(self.job_completion_status.values()):
            failed_jobs = {
                job_id: job_info
                for job_id, job_info in job_history[0].items()
                if job_info.final_state != SLURMSTATE.COMPLETED
            }
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            killed_jobs_indices = [job.i for job in killed_jobs.values()]
            logging.info(
                f"Total failed_jobs: {len(failed_jobs)}. Total killed_jobs: {len(killed_jobs)}"
            )
            if killed_jobs_indices:
                return self.resubmit_jobs(killed_jobs_indices)
            return True
        raise RuntimeError(f"All jobs failed. job_history: {job_history}\n")
