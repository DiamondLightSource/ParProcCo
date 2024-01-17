from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, Sequence
from copy import deepcopy

from .job_scheduling_information import JobSchedulingInformation
from .utils import check_jobscript_is_readable
from .slurm.slurm_rest import (
    JobProperties,
    JobSubmission,
)
from .slurm.slurm_client import SlurmClient


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

    submit_time: datetime
    start_time: Optional[datetime] = None
    current_state: Optional[SLURMSTATE] = None
    slots: Optional[int] = None
    time_to_dispatch: Optional[timedelta] = None
    wall_time: Optional[timedelta] = None
    final_state: Optional[SLURMSTATE] = None


class JobScheduler:
    def __init__(
        self,
        url: str,
        partition: str,
        cluster_output_dir: Optional[Union[Path, str]],
        user_name: Optional[str] = None,
        user_token: Optional[str] = None,
        wait_timeout: timedelta = timedelta(hours=2),
        terminate_after_wait: bool = True,
    ):
        """JobScheduler can be used for cluster job submissions"""
        self.job_history: list[dict[int, JobSchedulingInformation]] = []
        self.client = SlurmClient(url, user_name, user_token)
        self.partition = partition
        self.cluster_output_dir: Optional[Path] = (
            Path(cluster_output_dir) if cluster_output_dir else None
        )
        self.wait_timeout = wait_timeout
        self.terminate_after_wait = terminate_after_wait

    def fetch_and_update_state(
        self, job_scheduling_info: JobSchedulingInformation
    ) -> Optional[SLURMSTATE]:
        job_info = self.client.get_job(job_scheduling_info.job_id)
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

        start_time = job_info.start_time
        submit_time = job_info.submit_time
        end_time = job_info.end_time

        if start_time and submit_time and end_time:
            time_to_dispatch = timedelta(seconds=start_time - submit_time)
            wall_time = timedelta(seconds=end_time - start_time)
        else:
            time_to_dispatch = None
            wall_time = None

        status_info = job_scheduling_info.status_info

        if submit_time:
            # Don't overwrite unless a more specific value is given by the scheduler
            status_info.submit_time = datetime.fromtimestamp(submit_time)
        if start_time:
            status_info.start_time = datetime.fromtimestamp(start_time)
        status_info.slots = slots
        status_info.time_to_dispatch = time_to_dispatch
        status_info.wall_time = wall_time
        status_info.current_state = slurm_state
        logging.debug(f"Updating current state of {job_id} to {state}")
        return slurm_state

    def get_output_paths(
        self, job_scheduling_info_list: list[JobSchedulingInformation]
    ) -> tuple[Path]:
        return tuple(
            jsi.get_output_path()
            for jsi in job_scheduling_info_list
            if jsi.get_output_path() is not None
        )

    def get_success(
        self, job_scheduling_info_list: list[JobSchedulingInformation]
    ) -> bool:
        return all((info.completion_status for info in job_scheduling_info_list))

    def timestamp_ok(self, output: Path, start_time: datetime) -> bool:
        mod_time = datetime.fromtimestamp(output.stat().st_mtime)
        return mod_time > start_time

    def run(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation],
    ) -> bool:
        return self._submit_and_monitor(job_scheduling_info_list)

    def _submit_and_monitor(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation],
        wait_timeout: Optional[timedelta] = None,
        terminate_after_wait: Optional[bool] = None,
    ) -> bool:
        # Use scheduler settings if not given here
        if wait_timeout is None:
            wait_timeout = self.wait_timeout
        if terminate_after_wait is None:
            terminate_after_wait = self.terminate_after_wait

        self._submit_jobs(job_scheduling_info_list)
        self._wait_for_jobs(
            job_scheduling_info_list,
            wait_timeout=wait_timeout,
            terminate_after_wait=terminate_after_wait,
        )
        self._report_job_info(job_scheduling_info_list)
        return self.get_success(job_scheduling_info_list)

    def _submit_jobs(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation],
    ) -> None:
        try:
            for job_scheduling_info in job_scheduling_info_list:
                logging.debug(
                    f"Submitting job on cluster for job script {job_scheduling_info.job_script_path} and args {job_scheduling_info.job_script_arguments}"
                )
                submission = self.make_job_submission(job_scheduling_info)
                resp = self.client.submit_job(submission)
                if resp.job_id is None:
                    resp = self.client.submit_job(submission)
                    if resp.job_id is None:
                        raise ValueError("Job submission failed", resp.errors)
                job_scheduling_info.set_job_id(resp.job_id)
                job_scheduling_info.update_status_info(
                    StatusInfo(
                        submit_time=datetime.now(),
                    )
                )
                logging.debug(
                    f"Job for job script {job_scheduling_info.job_script_path} and args {submission.job.argv}"
                    f" has been submitted with id {resp.job_id}"
                )
        except Exception:
            logging.error("Unknown error occurred during job submission", exc_info=True)
            raise

    def make_job_submission(
        self, job_scheduling_info: JobSchedulingInformation
    ) -> JobSubmission:
        if job_scheduling_info.log_directory is None:
            if self.cluster_output_dir:
                if not self.cluster_output_dir.is_dir():
                    logging.debug(f"Making directory {self.cluster_output_dir}")
                    self.cluster_output_dir.mkdir(exist_ok=True, parents=True)
                else:
                    logging.debug(f"Directory {self.cluster_output_dir} already exists")

                error_dir = self.cluster_output_dir / "cluster_logs"
            else:
                error_dir = job_scheduling_info.working_directory / "cluster_logs"
            job_scheduling_info.log_directory = error_dir

        if not job_scheduling_info.log_directory.is_dir():
            logging.debug(f"Making directory {job_scheduling_info.log_directory}")
            job_scheduling_info.log_directory.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(
                f"Directory {job_scheduling_info.log_directory} already exists"
            )
        job_script_path = check_jobscript_is_readable(
            job_scheduling_info.job_script_path
        )
        job_script_command = " ".join(
            [
                f"#!/bin/bash\n{job_script_path}",
                *job_scheduling_info.job_script_arguments,
            ]
        )
        logging.info(f"creating submission with command: {job_script_command}")
        job = JobProperties(
            name=job_scheduling_info.job_name,
            partition=self.partition,
            cpus_per_task=job_scheduling_info.job_resources.cpu_cores,
            gpus_per_task=str(job_scheduling_info.job_resources.gpus),
            environment=job_scheduling_info.job_env,
            memory_per_cpu=job_scheduling_info.job_resources.memory,
            current_working_directory=str(job_scheduling_info.working_directory),
            standard_output=str(job_scheduling_info.get_stdout_path()),
            standard_error=str(job_scheduling_info.get_stderr_path()),
            get_user_environment="10L",
        )
        if job_scheduling_info.job_resources.extra_properties:
            for k, v in job_scheduling_info.job_resources.extra_properties.items():
                setattr(job, k, v)

        return JobSubmission(script=job_script_command, job=job)

    def wait_all_jobs(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation],
        state_group: STATEGROUP,
        deadline: datetime,
        sleep_time: int,
    ) -> list[JobSchedulingInformation]:
        remaining_jobs = list(job_scheduling_info_list)
        while len(remaining_jobs) > 0 and datetime.now() <= deadline:
            for jsi in list(remaining_jobs):
                current_state = self.fetch_and_update_state(jsi)
                if current_state in state_group:
                    remaining_jobs.remove(jsi)
            if len(remaining_jobs) > 0:
                time.sleep(sleep_time)
        return remaining_jobs

    def _wait_for_jobs(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation],
        wait_timeout: timedelta = timedelta(hours=2),
        terminate_after_wait: bool = False,
    ) -> None:
        wait_begin_time = datetime.now()
        wait_deadline = wait_begin_time + wait_timeout

        def get_deadline(
            job_scheduling_info: JobSchedulingInformation,
            allow_from_submission: bool = False,
        ) -> Optional[datetime]:
            # Timeout shouldn't include queue time
            if job_scheduling_info.status_info is None:
                return None
            elif job_scheduling_info.status_info.start_time is None:
                if allow_from_submission:
                    return (
                        job_scheduling_info.status_info.submit_time
                        + job_scheduling_info.timeout
                    )
                return None
            return (
                job_scheduling_info.status_info.start_time + job_scheduling_info.timeout
            )

        def handle_not_started(
            job_scheduling_info_list: List[JobSchedulingInformation],
            check_time: timedelta,
        ) -> List[JobSchedulingInformation]:
            # Wait for jobs to start (timeout shouldn't include queue time)
            starting_jobs = list(job_scheduling_info_list)
            timeout = datetime.now() + check_time
            while len(job_scheduling_info_list) > 0 and datetime.now() < timeout:
                for jsi in self.wait_all_jobs(
                    starting_jobs,
                    STATEGROUP.STARTING,
                    timeout,
                    0,
                ):
                    starting_jobs.remove(jsi)
                if len(starting_jobs) > 0:
                    # We want ot sleep only if there are jobs waiting to start
                    time.sleep(5)
                    logging.info("Jobs left to start: %d", len(starting_jobs))
            return starting_jobs

        def wait_for_ended(
            job_scheduling_info_list: Sequence[JobSchedulingInformation],
            deadline: datetime,
            check_time: timedelta,
        ) -> List[JobSchedulingInformation]:
            # Wait for jobs to complete
            self.wait_all_jobs(
                job_scheduling_info_list,
                STATEGROUP.ENDED,
                deadline,
                int(round(check_time.total_seconds())),
            )
            ended_jobs = handle_ended_jobs(job_scheduling_info_list)
            logging.info(
                "Jobs remaining = %d after %.3fs",
                len(job_scheduling_info_list) - len(ended_jobs),
                (datetime.now() - wait_begin_time).total_seconds(),
            )
            return ended_jobs

        def handle_ended_jobs(
            job_scheduling_info_list: Sequence[JobSchedulingInformation],
        ) -> List[JobSchedulingInformation]:
            ended_jobs = []
            for job_scheduling_info in job_scheduling_info_list:
                self.fetch_and_update_state(job_scheduling_info)
                if job_scheduling_info.status_info.current_state in STATEGROUP.ENDED:
                    logging.debug("Removing ended %d", job_scheduling_info.job_id)
                    ended_jobs.append(job_scheduling_info)
            return ended_jobs

        def handle_timeouts(
            job_scheduling_info_list: Sequence[JobSchedulingInformation],
        ) -> List[JobSchedulingInformation]:
            deadlines = (
                (jsi, get_deadline(jsi, allow_from_submission=False))
                for jsi in job_scheduling_info_list
            )
            timed_out_jobs = [
                jsi
                for jsi, deadline in deadlines
                if deadline is not None and deadline < datetime.now()
            ]
            for job_scheduling_info in timed_out_jobs:
                logging.warning(
                    "Job %d timed out. Terminating job now.",
                    job_scheduling_info.job_id,
                )
                self.client.cancel_job(job_scheduling_info.job_id)
            return timed_out_jobs

        ended_jobs = {
            jsi.job_id: jsi
            for jsi in handle_ended_jobs(
                job_scheduling_info_list=job_scheduling_info_list
            )  # Returns ended jobs
        }
        job_scheduling_info_dict = {jsi.job_id: jsi for jsi in job_scheduling_info_list}
        unfinished_jobs = {
            k: job_scheduling_info_dict[k]
            for k in set(job_scheduling_info_dict.keys()) - ended_jobs.keys()
        }

        timed_out_jobs = {
            jsi.job_id: jsi
            for jsi in handle_timeouts(
                job_scheduling_info_list=job_scheduling_info_list
            )  # Returns timed out jobs
        }

        running_jobs = {
            k: unfinished_jobs[k]
            for k in set(unfinished_jobs.keys()) - timed_out_jobs.keys()
        }

        # Check for any jobs that ended while waiting for jobs to start
        for jsi in handle_ended_jobs(job_scheduling_info_list=running_jobs.values()):
            ended_jobs[jsi.job_id] = jsi
            running_jobs.pop(jsi.job_id, None)

        if not running_jobs:
            logging.warning("All jobs ended before wait began")
            handle_ended_jobs(ended_jobs.values())
            return
        try:
            while datetime.now() < wait_deadline and len(running_jobs) > 0:
                # Handle none started (empty deadline list)
                next_deadline = min(
                    [
                        deadline
                        for deadline in (
                            get_deadline(jsi, allow_from_submission=True)
                            for jsi in running_jobs.values()
                        )
                    ]
                )
                check_time = min(
                    ((next_deadline - datetime.now()) / 2), timedelta(minutes=1)
                )  # smaller of half of between now and the nearest deadline or one minute

                not_started = handle_not_started(
                    running_jobs.values(), check_time=check_time
                )
                for jsi in wait_for_ended(
                    [v for k, v in running_jobs.items() if k not in not_started],
                    deadline=next_deadline,
                    check_time=check_time,
                ):
                    ended_jobs[jsi.job_id] = jsi
                    running_jobs.pop(jsi.job_id, None)

                for jsi in handle_timeouts(  # Returns timed out jobs
                    running_jobs.values()
                ):  # Update timed out jobs
                    timed_out_jobs[jsi.job_id] = jsi
                    running_jobs.pop(jsi.job_id, None)

            logging.debug("_wait_for_jobs loop ending, starting clear-up")

            if terminate_after_wait:
                for jsi in list(running_jobs):
                    try:
                        logging.info(
                            "Waiting for jobs timed out. Terminating job %d now.",
                            jsi.job_id,
                        )
                        self.client.cancel_job(jsi.job_id)
                        timed_out_jobs[jsi.job_id] = jsi
                    except Exception:
                        logging.error(
                            "Unknown error occurred terminating job %d",
                            jsi.job_id,
                            exc_info=True,
                        )

            # Finally wait for all timed_out_jobs to be terminated
            wait_for_ended(
                timed_out_jobs.values(),
                deadline=datetime.now() + timedelta(minutes=2),
                check_time=timedelta(minutes=1),
            )

        except Exception:
            logging.error("Unknown error occurred running job", exc_info=True)

    def _report_job_info(
        self, job_scheduling_info_list: list[JobSchedulingInformation]
    ) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for job_scheduling_info in job_scheduling_info_list:
            job_id = job_scheduling_info.job_id
            status_info = job_scheduling_info.status_info
            stdout_path = job_scheduling_info.get_stdout_path()
            logging.debug(f"Retrieving info for job {job_id}")

            # Check job states against expected possible options:
            state = status_info.current_state
            if state == SLURMSTATE.FAILED:
                status_info.final_state = SLURMSTATE.FAILED
                logging.error(
                    f"Job {job_id} failed."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not stdout_path.is_file():
                status_info.final_state = SLURMSTATE.NO_OUTPUT
                logging.error(
                    f"Job {job_id} with args {job_scheduling_info.job_script_arguments} has not created"
                    f" output file {stdout_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif not self.timestamp_ok(
                stdout_path,
                start_time=job_scheduling_info.status_info.start_time,
            ):
                status_info.final_state = SLURMSTATE.OLD_OUTPUT_FILE
                logging.error(
                    f"Job {job_id} with args {job_scheduling_info.job_script_arguments} has not created"
                    f" a new output file {stdout_path}"
                    f" State: {state}."
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

            elif state == SLURMSTATE.COMPLETED:
                job_scheduling_info.set_completion_status(True)
                status_info.final_state = SLURMSTATE.COMPLETED
                if status_info.slots and status_info.wall_time:
                    cpu_time = status_info.wall_time * status_info.slots
                else:
                    cpu_time = "n/a"
                logging.info(
                    f"Job {job_id} with args {job_scheduling_info.job_script_arguments} completed."
                    f" CPU time: {cpu_time}; Slots: {status_info.slots}"
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )
            else:
                status_info.final_state = state
                logging.error(
                    f"Job {job_id} ended with job state {status_info.final_state}"
                    f" Args {job_scheduling_info.job_script_arguments};"
                    f" Dispatch time: {status_info.time_to_dispatch}; Wall time: {status_info.wall_time}."
                )

        self.job_history.append({jsi.job_id: jsi for jsi in job_scheduling_info_list})

    def resubmit_jobs(
        self, job_ids: Optional[List[int]] = None, batch: int | None = None
    ) -> bool:
        old_job_scheduling_info_dict = self.get_job_history_batch(batch_number=batch)
        new_job_scheduling_info_list = []
        for job_id, old_job_scheduling_info in old_job_scheduling_info_dict.items():
            if job_ids is None or job_id in job_ids:
                new_job_scheduling_info = deepcopy(old_job_scheduling_info)
                new_job_scheduling_info.set_completion_status(False)
                new_job_scheduling_info.status_info = None
                new_job_scheduling_info.job_id = None
                new_job_scheduling_info_list.append(new_job_scheduling_info)
        logging.info(f"Resubmitting jobs from batch {batch} with job_ids: {job_ids}")
        return self._submit_and_monitor(new_job_scheduling_info_list)

    def filter_killed_jobs(
        self, job_scheduling_information_list: List[JobSchedulingInformation]
    ) -> List[JobSchedulingInformation]:
        return [
            jsi
            for jsi in job_scheduling_information_list
            if jsi.status_info.current_state == SLURMSTATE.CANCELLED
        ]

    def resubmit_killed_jobs(
        self, batch_number: int | None = None, allow_all_failed: bool = False
    ) -> bool:
        logging.info("Resubmitting killed jobs")
        job_scheduling_info_dict = self.get_job_history_batch(batch_number=batch_number)
        batch_completion_status = tuple(
            jsi.completion_status for jsi in job_scheduling_info_dict.values()
        )
        if all(batch_completion_status):
            logging.warning("No failed jobs to resubmit")
            return True
        elif allow_all_failed or any(batch_completion_status):
            failed_jobs = [
                jsi
                for jsi in job_scheduling_info_dict.values()
                if jsi.status_info.final_state != SLURMSTATE.COMPLETED
            ]
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            logging.info(
                f"Total failed_jobs: {len(failed_jobs)}. Total killed_jobs: {len(killed_jobs)}"
            )
            if killed_jobs:
                return self.resubmit_jobs(
                    job_ids=[jsi.job_id for jsi in killed_jobs], batch=batch_number
                )
            return True
        pretty_format_job_history = "\n".join(
            [
                f"Batch {i} - {', '.join(f'{jsi.job_id}: {jsi.status_info}' for jsi in batch.values())}"
                for i, batch in enumerate(self.job_history, 0)
            ]
        )
        raise RuntimeError(
            f"All jobs failed. job_history: {pretty_format_job_history}\n"
        )

    def clear_job_history(self) -> None:
        self.job_history.clear()

    def get_job_history_batch(
        self, batch_number: Optional[int] = None
    ) -> Dict[int, JobSchedulingInformation]:
        if batch_number is None:
            batch_number = self.get_batch_number()
            if batch_number < 0:
                raise IndexError("Job history is empty")
        elif batch_number >= len(self.job_history):
            raise IndexError("Batch %i does not exist in the job history")
        logging.debug("Getting batch %i from job history", batch_number)
        return self.job_history[batch_number]

    def get_batch_number(self) -> int:
        return len(self.job_history) - 1
