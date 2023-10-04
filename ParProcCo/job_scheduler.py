from __future__ import annotations

import logging
import requests
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel

from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable, get_slurm_token, get_user, get_ppc_dir
from .models.slurm_rest import (
    JobProperties,
    JobsResponse,
    JobResponseProperties,
    JobSubmission,
    JobSubmissionResponse,
)


_SLURM_VERSION = "v0.0.38"


class SLURMSTATE(Enum):
    # The following are states from https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
    BOOT_FAIL = auto()
    """Job terminated due to launch failure, typically due to a hardware failure
    (e.g. unable to boot the node or block and the job can not be requeued)."""
    CANCELLED = auto()
    """Job was explicitly cancelled by the user or system administrator.
    The job may or may not have been initiated"""
    COMPLETED = auto()
    "Job has terminated all processes on all nodes with an exit code of zero"
    CONFIGURING = auto()
    """Job has been allocated resources, but are waiting for them to become
    ready for use (e.g. booting)"""
    COMPLETING = auto()
    "Job is in the process of completing. Some processes on some nodes may still be active"
    DEADLINE = auto()
    "Job terminated on deadline"
    FAILED = auto()
    "Job terminated with non-zero exit code or other failure condition"
    NODE_FAIL = auto()
    "Job terminated due to failure of one or more allocated nodes"
    OUT_OF_MEMORY = auto()
    "Job experienced out of memory error"
    PENDING = auto()
    "Job is awaiting resource allocation"
    PREEMPTED = auto()
    "Job terminated due to preemption"
    RUNNING = auto()
    "Job currently has an allocation"
    RESV_DEL_HOLD = auto()
    "Job is held"
    REQUEUE_FED = auto()
    "Job is being requeued by a federation"
    REQUEUE_HOLD = auto()
    "Held job is being requeued"
    REQUEUED = auto()
    "Completing job is being requeued"
    RESIZING = auto()
    "Job is about to change size"
    REVOKED = auto()
    "Sibling was removed from cluster due to other cluster starting the job"
    SIGNALING = auto()
    "Job is being signaled"
    SPECIAL_EXIT = auto()
    """The job was requeued in a special state. This state can be set by users, typically in
        EpilogSlurmctld, if the job has terminated with a particular exit value"""
    STAGE_OUT = auto()
    "Job is staging out files"
    STOPPED = auto()
    """Job has an allocation, but execution has been stopped with SIGSTOP signal.
    CPUS have been retained by this job"""
    SUSPENDED = auto()
    "Job has an allocation, but execution has been suspended and CPUs have been released for other jobs"
    TIMEOUT = auto()
    "Job terminated upon reaching its time limit"
    NO_OUTPUT = auto()
    "Custom state. No output file found"
    OLD_OUTPUT_FILE = auto()
    "Custom state. Output file has not been updated since job started."


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
        extra_properties: Optional[dict[str, str]] = None,
        timeout: timedelta = timedelta(hours=2),
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
        self.extra_properties = extra_properties
        self.scheduler_mode: SchedulerModeInterface
        self.memory: int
        self.cores: int
        self.job_name: str
        self._slurm_endpoint_url = f"{url}/slurm/{_SLURM_VERSION}"
        self._session = requests.Session()

        self.user = user_name if user_name else get_user()
        self.token = user_token if user_token else get_slurm_token()
        self._session.headers["X-SLURM-USER-NAME"] = self.user
        self._session.headers["X-SLURM-USER-TOKEN"] = self.token
        self._session.headers["Content-Type"] = "application/json"

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        return self._session.get(
            f"{self._slurm_endpoint_url}/{endpoint}", params=params, timeout=timeout
        )

    def _post(self, endpoint: str, data: BaseModel) -> requests.Response:
        return self._session.post(
            f"{self._slurm_endpoint_url}/{endpoint}",
            data.model_dump_json(exclude_defaults=True),
        )

    def _delete(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        return self._session.delete(
            f"{self._slurm_endpoint_url}/{endpoint}", params=params, timeout=timeout
        )

    def _get_response_json(self, response: requests.Response) -> dict:
        response.raise_for_status()
        try:
            return response.json()
        except:
            logging.error("Response not json: %s", response.content, exc_info=True)
            raise

    def get_jobs_response(self, job_id: int | None = None) -> JobsResponse:
        endpoint = f"job/{job_id}" if job_id is not None else "jobs"
        response = self._get(endpoint)
        return JobsResponse.model_validate(self._get_response_json(response))

    def get_job(self, job_id: int) -> JobResponseProperties:
        ji = self.get_jobs_response(job_id)
        if ji.jobs:
            n = len(ji.jobs)
            if n == 1:
                return ji.jobs[0]
            if n > 1:
                raise ValueError("Multiple jobs returned {ji.jobs}")
        raise ValueError("No job info found for job id {job_id}")

    def update_status_infos(self, job_info: JobResponseProperties) -> None:
        job_id = job_info.job_id
        if job_id is None:
            raise ValueError(f"Job info has no job id: {job_info}")
        state = job_info.job_state
        slurm_state = SLURMSTATE[state] if state else None
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
        self.status_infos[job_id].current_state = slurm_state
        logging.info(f"Updating current state of {job_id} to {state}")

    def submit_job(self, job_submission: JobSubmission) -> JobSubmissionResponse:
        response = self._post("job/submit", job_submission)
        return JobSubmissionResponse.model_validate(self._get_response_json(response))

    def cancel_job(self, job_id: int) -> JobsResponse:
        response = self._delete(f"job/{job_id}")
        return JobsResponse.model_validate(self._get_response_json(response))

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
        self.job_env = (
            job_env if job_env else {"ParProcCo": "0"}
        )  # job_env cannot be empty dict
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
        return self._submit_and_monitor(job_indices)

    def _submit_and_monitor(self, job_indices: List[int]) -> bool:
        self._submit_jobs(job_indices)
        self._wait_for_jobs()
        self._report_job_info()
        return self.get_success()

    def _submit_jobs(self, job_indices: List[int]) -> None:
        logging.debug(
            f"Submitting jobs on cluster for jobscript {self.jobscript_path} and args {self.jobscript_args}"
        )
        try:
            self.status_infos = {}
            for i in job_indices:
                submission = self.make_job_submission(i)
                resp = self.submit_job(submission)
                if resp.job_id is None:
                    resp = self.submit_job(submission)
                    if resp.job_id is None:
                        raise ValueError("Job submission failed", resp.errors)
                self.status_infos[resp.job_id] = StatusInfo(
                    Path(submission.job.standard_output),
                    i,
                    int(time.time()),
                )
                logging.debug(
                    f"job for jobscript {self.jobscript_path} and args {submission.job.argv}"
                    f" has been submitted with id {resp.job_id}"
                )
        except Exception:
            logging.error("Unknown error occurred during job submission", exc_info=True)
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
        if self.extra_properties:
            for k, v in self.extra_properties.items():
                setattr(job, k, v)

        return JobSubmission(script=self.jobscript_command, job=job)

    def fetch_and_update_state(self, job_id: int) -> JobResponseProperties:
        ji = self.get_job(job_id)
        self.update_status_infos(ji)
        return ji

    def wait_all_jobs(
        self,
        job_ids: List[int],
        required_state: Literal["ENDED"] | Literal["STARTING"],
        timeout: int,
        sleep_time: int,
    ) -> list[int]:
        deadline = time.time() + timeout
        remaining_jobs = job_ids.copy()
        state_group = STATEGROUP[required_state]
        while len(remaining_jobs) > 0 and time.time() <= deadline:
            for job_id in list(remaining_jobs):
                self.fetch_and_update_state(job_id)
                if self.status_infos[job_id].current_state in state_group:
                    remaining_jobs.remove(job_id)
            if len(remaining_jobs) > 0:
                time.sleep(sleep_time)
        return remaining_jobs

    def _wait_for_jobs(
        self,
    ) -> None:
        wait_begin_time = time.time()
        max_time = int(round(self.timeout.total_seconds()))
        check_time = min(int(round(max_time / 2)), 60)  # smaller of half of max_time or one minute
        logging.info("Jobs have check_time=%d and max_time=%d", check_time, max_time)
        try:
            remaining_jobs = list(self.status_infos)
            # Wait for jobs to start (timeout shouldn't include queue time)
            while len(remaining_jobs) > 0:
                remaining_jobs = self.wait_all_jobs(
                    remaining_jobs, STATEGROUP.STARTING.name, 60, 5
                )
                if len(remaining_jobs) > 0:
                    logging.info("Jobs left to start: %d", len(remaining_jobs))

            jobs_started_time = time.time()
            running_jobs = [
                job_id
                for job_id, status_info in self.status_infos.items()
                if status_info.current_state not in STATEGROUP.ENDED
            ]
            logging.info("All jobs started: %d still running", len(running_jobs))

            deadline = wait_begin_time + max_time
            while time.time() < deadline and len(running_jobs) > 0:
                # Wait for jobs to complete
                running_jobs = self.wait_all_jobs(
                    running_jobs, STATEGROUP.ENDED.name, max_time, check_time
                )
                for job_id in list(running_jobs):
                    self.fetch_and_update_state(job_id)
                    if self.status_infos[job_id].current_state in STATEGROUP.ENDED:
                        logging.debug("Removing(1) ended %d", job_id)
                        running_jobs.remove(job_id)
                logging.info(
                    "Jobs remaining = %d after %.3fs", len(running_jobs), time.time() - jobs_started_time
                )
            for job_id in list(running_jobs):
                # Check state of remaining jobs immediately before cancelling
                self.fetch_and_update_state(job_id)
                if self.status_infos[job_id].current_state in STATEGROUP.ENDED:
                    logging.debug("Removing(2) ended %d", job_id)
                    running_jobs.remove(job_id)
                else:
                    logging.warning("Job %d timed out. Terminating job now.", job_id)
                    self.cancel_job(job_id)
            if len(running_jobs) > 0:
                # Termination takes some time, wait a max of 2 mins
                self.wait_all_jobs(running_jobs, STATEGROUP.ENDED.name, 120, 60)
                logging.info(
                    "Jobs terminated = %d after %.3fs", len(running_jobs), time.time() - jobs_started_time
                )
        except Exception:
            logging.error("Unknown error occurred running Slurm job", exc_info=True)

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
        return self._submit_and_monitor(job_indices)

    def filter_killed_jobs(self, jobs: List[StatusInfo]) -> List[int]:
        return [
            status_info.i
            for status_info in jobs
            if status_info.current_state == SLURMSTATE.CANCELLED
        ]

    def resubmit_killed_jobs(self, allow_all_failed: bool = False) -> bool:
        logging.info("Resubmitting killed jobs")
        job_history = self.job_history
        if all(self.job_completion_status.values()):
            logging.warning("No failed jobs to resubmit")
            return True
        elif allow_all_failed or any(self.job_completion_status.values()):
            failed_jobs = [
                status_info
                for status_info in job_history[self.batch_number].values()
                if status_info.final_state != SLURMSTATE.COMPLETED
            ]
            killed_jobs_indices = self.filter_killed_jobs(failed_jobs)
            logging.info(
                f"Total failed_jobs: {len(failed_jobs)}. Total killed_jobs: {len(killed_jobs_indices)}"
            )
            if killed_jobs_indices:
                return self.resubmit_jobs(killed_jobs_indices)
            return True
        raise RuntimeError(f"All jobs failed. job_history: {job_history}\n")
