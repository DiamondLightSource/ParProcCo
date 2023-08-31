from __future__ import annotations

import logging
import os
import requests
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable
from models.slurmdb_rest import DbJob, DbJobInfo
from models.slurm_rest import JobProperties, JobsResponse, JobResponseProperties, JobSubmission, JobSubmissionResponse

# WIP: Migrating from drmaa2 to slurm as in https://github.com/DiamondLightSource/python-zocalo


class SLURMSTATE(str, Enum):
    # The following are states from https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
    BF = (
        "BOOT_FAIL"  # Job terminated due to launch failure, typically due to a hardware failure
        # (e.g. unable to boot the node or block and the job can not be requeued).
    )
    CA = (
        "CANCELLED"  # Job was explicitly cancelled by the user or system administrator.
        # The job may or may not have been initiated
    )
    CD = "COMPLETED"  # Job has terminated all processes on all nodes with an exit code of zero
    CF = (
        "CONFIGURING"  # Job has been allocated resources, but are waiting for them to become ready for
        # use (e.g. booting)
    )
    CG = "COMPLETING"  # Job is in the process of completing. Some processes on some nodes may still be active
    DL = "DEADLINE"  # Job terminated on deadline
    F = "FAILED"  # Job terminated with non-zero exit code or other failure condition
    NF = "NODE_FAIL"  # Job terminated due to failure of one or more allocated nodes
    OOM = "OUT_OF_MEMORY"  # Job experienced out of memory error
    PD = "PENDING"  # Job is awaiting resource allocation
    PR = "PREEMPTED"  # Job terminated due to preemption
    R = "RUNNING"  # Job currently has an allocation
    RD = "RESV_DEL_HOLD"  # Job is held
    RF = "REQUEUE_FED"  # Job is being requeued by a federation
    RH = "REQUEUE_HOLD"  # Held job is being requeued
    RQ = "REQUEUED"  # Completing job is being requeued
    RS = "RESIZING"  # Job is about to change size
    RV = "REVOKED"  # Sibling was removed from cluster due to other cluster starting the job
    SI = "SIGNALING"  # Job is being signaled
    SE = (
        "SPECIAL_EXIT"  # The job was requeued in a special state. This state can be set by users, typically in
        # EpilogSlurmctld, if the job has terminated with a particular exit value
    )
    SO = "STAGE_OUT"  # Job is staging out files
    ST = (
        "STOPPED"  # Job has an allocation, but execution has been stopped with SIGSTOP signal. CPUS have been
        # retained by this job
    )
    S = (
        "SUSPENDED"  # Job has an allocation, but execution has been suspended and CPUs have been released for
        # other jobs
    )
    TO = "TIMEOUT"  # Job terminated upon reaching its time limit


class STATEGROUP(set, Enum):
    OUTOFTIME = {SLURMSTATE.TO, SLURMSTATE.DL}
    FINISHED = {SLURMSTATE.CD, SLURMSTATE.F, SLURMSTATE.TO, SLURMSTATE.DL}
    COMPUTEISSUE = {SLURMSTATE.BF, SLURMSTATE.NF, SLURMSTATE.OOM}
    ENDED = {SLURMSTATE.CD, SLURMSTATE.F, SLURMSTATE.TO, SLURMSTATE.DL}
    REQUEUEABLE = {SLURMSTATE.CF, SLURMSTATE.R, SLURMSTATE.ST, SLURMSTATE.S}
    STARTING = {
        SLURMSTATE.PD,
        SLURMSTATE.RQ,
        SLURMSTATE.RS,
        SLURMSTATE.S,
        SLURMSTATE.CF,
    }


@dataclass
class StatusInfo:
    """Class for keeping track of job status."""

    output_path: Path
    i: int
    start_time: int
    current_state: str = ""
    slots: int = 0
    time_to_dispatch: int = 0
    wall_time: int = 0
    final_state: str = ""


class JobScheduler:
    def __init__(
        self,
        url: str,
        working_directory: Optional[Union[Path, str]],
        cluster_output_dir: Optional[Union[Path, str]],
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
        self.job_env: Optional[Dict[str, str]] = None
        self.jobscript_args: List
        self.output_paths: List[Path] = []
        self.start_time = datetime.now()
        self.status_infos: Dict[str:StatusInfo]
        self.timeout = timeout
        self.working_directory = (
            Path(working_directory)
            if working_directory
            else (self.cluster_output_dir if self.cluster_output_dir else Path.home())
        )
        self.scheduler_mode: SchedulerModeInterface
        self.memory: int
        self.cores: int
        self.job_name: str
        self._url = url
        self._version = version
        self._session = requests.Session()
        self._session.headers["X-SLURM-USER-NAME"] = user_name if user_name else os.environ["USER"]
        self.user = user_name if user_name else os.environ["USER"]
        self.token = user_token if user_token else os.environ["SLURM_JWT"]
        self._session.headers["X-SLURM-USER-TOKEN"] = user_token if user_token else os.environ["SLURM_JWT"]

    def get(self, endpoint: str, params: dict[str, Any] = None, timeout: float | None = None) -> requests.Response:
        response = self._session.get(f"{self._url}/{endpoint}", params=params, timeout=timeout)
        response.raise_for_status()
        return response

    def put(
        self,
        endpoint: str,
        params: dict[str, Any] = None,
        json: dict[str, Any] = None,
        timeout: float | None = None,
    ) -> requests.Response:
        response = self._session.put(f"{self._url}/{endpoint}", params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    def _prepare_request(self, data: BaseModel) -> tuple[str, dict[str, str]] | tuple[None, None]:
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

    def delete(self, endpoint: str, params: dict[str, Any] = None, timeout: float | None = None) -> requests.Response:
        response = self._session.delete(f"{self._url}/{endpoint}", params=params, timeout=timeout)
        response.raise_for_status()
        return response

    def get_jobs(self) -> JobsResponse:
        endpoint = f"slurm/{self._version}/jobs"
        response = self.get(endpoint)
        return JobsResponse.model_validate(response.json())

    def get_job_response(self, job_id: int) -> JobsResponse:
        endpoint = f"slurm/{self._version}/job/{job_id}"
        response = self.get(endpoint)
        return JobsResponse(**response.json())

    def get_job_info(self, job_id: int) -> DbJobInfo:
        endpoint = f"slurmdb/{self._version}/job/{job_id}"
        response = self.get(endpoint)
        return DbJobInfo(**response.json())

    def get_job(self, job_id: int) -> DbJob | JobResponseProperties:
        ji = self.get_job_response(job_id)
        if len(ji.jobs) == 1:
            return ji.jobs[0]
        elif ji.jobs == []:
            ji = self.get_job_info(job_id)
            if len(ji.jobs) == 1:
                return ji.jobs[0]
            elif ji.jobs == []:
                raise ValueError("No job info found for job id {job_id}")
        raise ValueError("Multiple jobs returned {ji.jobs}")

    def get_job_state(self, job: DbJob | JobResponseProperties) -> str:
        if isinstance(job, JobResponseProperties):
            return job.job_state
        elif isinstance(job, DbJob):
            return job.state.current

    def update_status_infos(self, job_id, job_info: DbJob | JobResponseProperties, state: str):
        if isinstance(job_info, JobResponseProperties):
            try:
                slots = int((job_info.tres_alloc_str.split(",")[1]).split("=")[1])
                dispatch_time = job_info.start_time
                time_to_dispatch = dispatch_time - job_info.submit_time
                wall_time = job_info.end_time - dispatch_time
            except Exception:
                logging.error(f"Failed to get job submission time statistics for job {job_info}")
                raise
        else:
            try:
                slots = job_info.allocation_nodes
                job_time = job_info.time
                dispatch_time = job_time.start
                time_to_dispatch = dispatch_time - job_time.submission
                wall_time = job_time.end - dispatch_time
            except Exception:
                logging.error(f"Failed to get job submission time statistics for job {job_info}")
                raise

        self.status_infos[job_id].slots = slots
        self.status_infos[job_id].time_to_dispatch = time_to_dispatch
        self.status_infos[job_id].wall_time = wall_time
        self.status_infos[job_id].current_state = state

    def submit_job(self, job_submission: JobSubmission) -> JobSubmissionResponse:
        endpoint = f"slurm/{self._version}/job/submit"
        response = self._post(job_submission, endpoint)
        return JobSubmissionResponse(**response.json())

    def cancel_job(self, job_id: int) -> JobsResponse:
        endpoint = f"slurm/{self._version}/job/{job_id}"
        response = self.delete(endpoint)
        return JobsResponse(**response.json())

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
        job_env: Optional[Dict[str, str]],
        memory: int = 4000,
        cores: int = 6,
        jobscript_args: Optional[List] = None,
        job_name: str = "ParProcCo_job",
    ) -> bool:
        self.jobscript_path = check_jobscript_is_readable(jobscript_path)
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
        self._run_jobs(job_indices)
        self._wait_for_jobs()
        self._report_job_info()
        return self.get_success()

    def _run_jobs(self, job_indices: List[int]) -> None:
        logging.debug(f"Running jobs on cluster for jobscript {self.jobscript_path} and args {self.jobscript_args}")
        try:
            self.status_infos = {}
            for i in job_indices:
                template = self.make_job_submission(i)
                resp = self.submit_job(template)
                self.status_infos[resp.job_id] = StatusInfo(
                    Path(template.job.standard_output),
                    i,
                    int(time.time()),
                )
                logging.debug(
                    f"job array for jobscript {self.jobscript_path} and args {template.job.argv}"
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
        args = self.scheduler_mode.generate_args(i, self.memory, self.cores, self.jobscript_args, output_fp)
        self.jobscript_command = " ".join([f"#!/bin/bash\n{self.jobscript_path}", *args])
        logging.info(f"creating template with command: {self.jobscript_command}")
        job = JobProperties(
            name=self.job_name,
            partition="cs05r",
            cpus_per_task=self.cores,
            environment={"ParProcCo": "0"},
            memory_per_cpu=self.memory,
            current_working_directory=str(self.working_directory),
            standard_output=stdout_fp,
            standard_error=stderr_fp,
            get_user_environment="10L",
        )
        return JobSubmission(script=self.jobscript_command, job=job, jobs=jobs)

    def wait_all_jobs_terminated(self, job_ids: List[int], check_time: int):
        wait_until = time.time() + check_time
        remaining_jobs = job_ids.copy()
        while len(remaining_jobs) > 0 and time.time() <= wait_until:
            terminated_jobs = []
            for j in remaining_jobs:
                ji = self.get_job(j)
                state = self.get_job_state(ji)
                if state in STATEGROUP.ENDED:
                    remaining_jobs.remove(j)
                    terminated_jobs.append((j, ji, state))
            time.sleep(min(60, check_time))
        for x in terminated_jobs:
            self.update_status_infos(*x)
        for job in remaining_jobs:
            ji = self.get_job(job)
            state = self.get_job_state(ji)
            self.update_status_infos(job, ji, state)

    def wait_all_jobs_started(self, job_ids: List[int], check_time: int):
        wait_until = time.time() + check_time
        remaining_jobs = job_ids.copy()
        started_jobs = []
        while len(remaining_jobs) > 0 and time.time() <= wait_until:
            for j in remaining_jobs:
                ji = self.get_job(j)
                state = self.get_job_state(ji)
                if state not in STATEGROUP.STARTING:
                    started_jobs.append(j)
                    remaining_jobs.remove(j)
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
                    ji = self.get_job(job_id)
                    if self.get_job_state(ji) == "RUNNING":
                        jobs_remaining.append(job_id)
                        logging.debug(f"Job {job_id} still running")
                job_list = jobs_remaining
                logging.info(
                    f"Jobs remaining = {len(jobs_remaining)} after {total_time}s"
                )

            jobs_remaining = []
            for job_id in job_list:
                ji = self.get_job(job_id)
                if self.get_job_state(ji) == "RUNNING":
                    logging.warning(f"Job {job_id} timed out. Terminating job now.")
                    jobs_remaining.append(job_id)
                    self.cancel_job(job_id)
            if jobs_remaining:
                # Termination takes some time, wait a max of 2 mins
                self.wait_all_jobs_terminated(jobs_remaining, 120)
                total_time += 120
                logging.info(f"Jobs terminated = {len(jobs_remaining)} after {total_time}s")
        except Exception:
            logging.error("Unknown error occurred running slurm job", exc_info=True)

    def _report_job_info(self) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for job_id, status_info in self.status_infos.items():
            logging.debug(f"Retrieving info for slurm job {job_id}")

            # Check job states against expected possible options:
            state = status_info.current_state
            if state == "FAILED":
                status_info.final_state = "FAILED"
                logging.error(
                    f"Slurm job {job_id} failed."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not status_info.output_path.is_file():
                status_info.final_state = "NO_OUTPUT"
                logging.error(
                    f"Slurm job {job_id} with args {self.jobscript_args} has not created"
                    f" output file {status_info.output_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not self.timestamp_ok(status_info.output_path):
                status_info.final_state = "OLD_OUTPUT_FILE"
                logging.error(
                    f"Slurm job {job_id} with args {self.jobscript_args} has not created"
                    f" a new output file {status_info.output_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif state == "COMPLETED":
                self.job_completion_status[str(status_info.i)] = True
                status_info.final_state = "COMPLETED"
                slots = status_info.slots
                cpu_time = float(status_info.wall_time * slots)
                logging.info(
                    f"Job {job_id} with args {self.jobscript_args} completed."
                    f" CPU time: {timedelta(seconds=cpu_time)}; Slots: {status_info.slots}"
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

    def filter_killed_jobs(self, jobs: Dict[str:StatusInfo]) -> Dict[str:StatusInfo]:
        killed_jobs = {
            job_id: status_info for job_id, status_info in jobs.items() if status_info.current_state == "CANCELLED"
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
                job_id: job_info for job_id, job_info in job_history[0].items() if job_info.final_state != "COMPLETED"
            }
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            killed_jobs_indices = [job.i for job in killed_jobs.values()]
            logging.info(f"Total failed_jobs: {len(failed_jobs)}. Total killed_jobs: {len(killed_jobs)}")
            if killed_jobs_indices:
                return self.resubmit_jobs(killed_jobs_indices)
            return True

        raise RuntimeError(f"All jobs failed. job_history: {job_history}\n")
