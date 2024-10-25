# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import logging
import re
import time
import shlex
from collections.abc import Sequence, ValuesView
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .job_scheduling_information import JobSchedulingInformation
from .slurm.slurm_client import SlurmClient
from .slurm.slurm_rest import (
    JobDescMsg,
    JobInfo,
    JobSubmitReq,
    JobStateEnum,
    StringArray,
    Uint32NoVal,
    Uint64NoVal,
)
from .utils import check_jobscript_is_readable


SLURMSTATE = Enum(  # type: ignore
    "SLURMSTATE",
    [(js.name, js.value) for js in JobStateEnum]
    + [
        (n, n)
        for n in (
            "NO_OUTPUT",  # Custom state. No output file found
            "OLD_OUTPUT_FILE",  # Custom state. Output file has not been updated since job started.
        )
    ],
)

# class NewJobStateEnum(Enum):
#    CANCELLED = "CANCELLED" #
#    LAUNCH_FAILED = "LAUNCH_FAILED" #
#    UPDATE_DB = "UPDATE_DB" #
#    RECONFIG_FAIL = "RECONFIG_FAIL" #
#    POWER_UP_NODE = "POWER_UP_NODE" #


class STATEGROUP(tuple[SLURMSTATE], Enum):  # type: ignore
    OUTOFTIME = (SLURMSTATE.TIMEOUT, SLURMSTATE.DEADLINE)
    FINISHED = (
        SLURMSTATE.COMPLETED,
        SLURMSTATE.FAILED,
        SLURMSTATE.TIMEOUT,
        SLURMSTATE.DEADLINE,
    )
    COMPUTEISSUE = (
        SLURMSTATE.BOOT_FAIL,
        SLURMSTATE.NODE_FAIL,
        SLURMSTATE.OUT_OF_MEMORY,
    )
    ENDED = (
        SLURMSTATE.COMPLETED,
        SLURMSTATE.FAILED,
        SLURMSTATE.TIMEOUT,
        SLURMSTATE.DEADLINE,
        SLURMSTATE.CANCELLED,
        SLURMSTATE.LAUNCH_FAILED,
    )
    REQUEUEABLE = (
        SLURMSTATE.CONFIGURING,
        SLURMSTATE.RUNNING,
        SLURMSTATE.STOPPED,
        SLURMSTATE.SUSPENDED,
    )
    STARTING = (
        SLURMSTATE.PENDING,
        SLURMSTATE.REQUEUED,
        SLURMSTATE.RESIZING,
        SLURMSTATE.SUSPENDED,
        SLURMSTATE.CONFIGURING,
    )


@dataclass
class StatusInfo:
    """Class for keeping track of job status."""

    submit_time: datetime
    start_time: datetime | None = None
    current_state: SLURMSTATE | None = None
    cpus: int | None = None
    gpus: int | None = None
    time_to_dispatch: timedelta | None = None
    wall_time: timedelta | None = None
    final_state: SLURMSTATE | None = None


class JobScheduler:
    RE_CPU = re.compile(r"cpu=(\d+)")
    RE_GPU = re.compile(r"gpu=(\d+)")

    def __init__(
        self,
        url: str,
        partition: str,
        user_name: str | None = None,
        user_token: str | None = None,
        wait_timeout: timedelta = timedelta(hours=2),
        terminate_after_wait: bool = True,
    ):
        """JobScheduler can be used for cluster job submissions"""
        self.job_history: list[dict[int, JobSchedulingInformation]] = []
        self.client = SlurmClient(url, user_name, user_token)
        self.partition = partition
        self.wait_timeout = wait_timeout
        self.terminate_after_wait = terminate_after_wait

    def _update_processors(
        self, status_info: StatusInfo, allocated: str, job_info: JobInfo
    ):
        if status_info.cpus is None:
            try:
                cpu_match = JobScheduler.RE_CPU.search(allocated)
                status_info.cpus = int(cpu_match.group(1)) if cpu_match else None
            except Exception as e:
                logging.warning(
                    "Failed to get cpus for job %i; setting cpus to 0. Job info: %s",
                    job_info.job_id,
                    job_info,
                    exc_info=e,
                )
                status_info.cpus = 0
        if status_info.gpus is None:
            try:
                gpu_match = JobScheduler.RE_GPU.search(allocated)
                status_info.gpus = int(gpu_match.group(1)) if gpu_match else None
            except Exception as e:
                logging.warning(
                    "Failed to get gpus for job %i; setting gpus to 0. Job info: %s",
                    job_info.job_id,
                    job_info,
                    exc_info=e,
                )
                status_info.gpus = 0

    def fetch_and_update_state(
        self, job_scheduling_info: JobSchedulingInformation
    ) -> SLURMSTATE | None:
        job_info = self.client.get_job(job_scheduling_info.job_id)
        job_id = job_info.job_id
        if job_id is None or job_id < 0:
            raise ValueError(f"Job info has invalid job id: {job_info}")
        state = job_info.job_state
        slurm_state = SLURMSTATE[state[0].value] if state else None

        start_time = (
            job_info.start_time.number if job_info.start_time is not None else 0
        )
        submit_time = (
            job_info.submit_time.number if job_info.submit_time is not None else 0
        )
        end_time = job_info.end_time.number if job_info.end_time is not None else 0
        logging.debug(
            "Update %i: submit %s; start %s; end %s",
            job_id,
            submit_time,
            start_time,
            end_time,
        )

        if start_time and submit_time:
            time_to_dispatch = timedelta(seconds=start_time - submit_time)
            now = datetime.now().timestamp()
            if end_time and end_time > now:
                end_time = now
            wall_time = timedelta(seconds=end_time - start_time)
        else:
            time_to_dispatch = None
            wall_time = None

        status_info = job_scheduling_info.status_info
        assert status_info
        if submit_time:
            # Don't overwrite unless a more specific value is given by the scheduler
            status_info.submit_time = datetime.fromtimestamp(submit_time)
        if start_time:
            status_info.start_time = datetime.fromtimestamp(start_time)
        tres_alloc_str = job_info.tres_alloc_str
        if tres_alloc_str:
            self._update_processors(status_info, tres_alloc_str, job_info)

        status_info.time_to_dispatch = time_to_dispatch
        status_info.wall_time = wall_time
        status_info.current_state = slurm_state
        logging.debug(
            "Updating current state of %i to %s: %s", job_id, slurm_state, status_info
        )
        logging.info(
            "Job %i: %s %ss", job_id, slurm_state, wall_time if wall_time else 0
        )
        return slurm_state

    def get_output_paths(
        self,
        job_scheduling_info_list: list[JobSchedulingInformation]
        | ValuesView[JobSchedulingInformation],
    ) -> tuple[Path, ...]:
        return tuple(
            p for p in (jsi.get_output_path() for jsi in job_scheduling_info_list) if p
        )

    def get_success(
        self, job_scheduling_info_list: list[JobSchedulingInformation]
    ) -> bool:
        return all((info.completion_status for info in job_scheduling_info_list))

    def timestamp_ok(self, output: Path, start_time: datetime | None) -> bool:
        if start_time is None:
            return False
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
        wait_timeout: timedelta | None = None,
        terminate_after_wait: bool | None = None,
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
                    "Submitting job on cluster for job script '%s' and args '%s'",
                    job_scheduling_info.job_script_path,
                    job_scheduling_info.job_script_arguments,
                )
                submission = self.make_job_submission(job_scheduling_info)
                assert submission.job is not None
                resp = None
                try:
                    resp = self.client.submit_job(submission)
                except Exception:
                    logging.error("Job submission failed. Trying again", exc_info=True)

                if resp is None or resp.job_id is None:
                    try:
                        resp = self.client.submit_job(submission)
                    except Exception:
                        logging.error("Job submission failed", exc_info=True)
                        raise ValueError("Job submission failed")
                assert resp.job_id
                job_scheduling_info.set_job_id(resp.job_id)
                job_scheduling_info.update_status_info(
                    StatusInfo(
                        submit_time=datetime.now(),
                    )
                )
                logging.info(
                    "Submitted job '%s' with id %d",
                    job_scheduling_info.job_name,
                    resp.job_id,
                )
        except Exception:
            logging.error("Unknown error occurred during job submission", exc_info=True)
            raise

    def make_job_submission(
        self, job_scheduling_info: JobSchedulingInformation
    ) -> JobSubmitReq:
        if job_scheduling_info.log_directory is None:
            assert job_scheduling_info.working_directory
            job_scheduling_info.log_directory = (
                job_scheduling_info.working_directory / "cluster_logs"
            )

        if not job_scheduling_info.log_directory.is_dir():
            logging.debug("Making directory '%s'", job_scheduling_info.log_directory)
            job_scheduling_info.log_directory.mkdir(exist_ok=True, parents=True)
        else:
            assert job_scheduling_info.log_directory
            logging.debug(
                "Directory '%s' already exists", job_scheduling_info.log_directory
            )
        assert job_scheduling_info.job_script_path
        job_script_path = check_jobscript_is_readable(
            job_scheduling_info.job_script_path
        )
        job_script_command = "#!/bin/bash\n" + shlex.join(
            [
                str(job_script_path),
                *job_scheduling_info.job_script_arguments,
            ]
        )
        logging.info("Creating submission with command: %s", job_script_command)
        env_list = [f"{k}={v}" for k, v in job_scheduling_info.job_env.items()]
        if "USER" not in job_scheduling_info.job_env:
            env_list.append(f"USER={self.client.user}")

        job = JobDescMsg(
            name=job_scheduling_info.job_name,
            partition=self.partition,
            cpus_per_task=job_scheduling_info.job_resources.cpu_cores,
            tres_per_task=f"gres/gpu:{job_scheduling_info.job_resources.gpus}",
            tasks=1,
            time_limit=Uint32NoVal(
                number=int((job_scheduling_info.timeout.total_seconds() + 59) // 60),
                set=True,
            ),
            environment=StringArray(root=env_list),
            memory_per_cpu=Uint64NoVal(
                number=job_scheduling_info.job_resources.memory, set=True
            ),
            current_working_directory=str(job_scheduling_info.working_directory),
            standard_output=str(job_scheduling_info.get_stdout_path()),
            standard_error=str(job_scheduling_info.get_stderr_path()),
        )
        if job_scheduling_info.job_resources.extra_properties:
            for k, v in job_scheduling_info.job_resources.extra_properties.items():
                setattr(job, k, v)

        return JobSubmitReq(script=job_script_command, job=job)

    def wait_all_jobs(
        self,
        job_scheduling_info_list: Sequence[JobSchedulingInformation]
        | ValuesView[JobSchedulingInformation],
        in_group: bool,
        state_group: STATEGROUP,
        deadline: datetime,
        sleep_time: int,
    ) -> list[JobSchedulingInformation]:
        remaining_jobs = list(job_scheduling_info_list)
        while len(remaining_jobs) > 0 and datetime.now() <= deadline:
            for jsi in list(remaining_jobs):
                current_state = self.fetch_and_update_state(jsi)
                if (current_state in state_group) == in_group:
                    logging.debug(
                        "Removing %i as %s in %s", jsi.job_id, in_group, state_group
                    )
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
        ) -> datetime | None:
            # Timeout shouldn't include queue time
            status_info = job_scheduling_info.status_info
            if status_info is None:
                return None
            elif status_info.start_time is None:
                if allow_from_submission:
                    return status_info.submit_time + job_scheduling_info.timeout
                return None
            return status_info.start_time + job_scheduling_info.timeout

        def handle_not_started(
            job_scheduling_info_list: Sequence[JobSchedulingInformation]
            | ValuesView[JobSchedulingInformation],
            check_time: timedelta,
        ) -> list[JobSchedulingInformation]:
            # Wait for jobs to start (timeout shouldn't include queue time)
            starting_jobs = list(job_scheduling_info_list)
            timeout = datetime.now() + check_time
            logging.debug(
                "Wait for jobs (%d) to start up to %s", len(starting_jobs), timeout
            )
            while len(starting_jobs) > 0 and datetime.now() < timeout:
                starting_jobs = self.wait_all_jobs(
                    starting_jobs,
                    False,
                    STATEGROUP.STARTING,
                    timeout,
                    5,
                )
                if len(starting_jobs) > 0:
                    # We want to sleep only if there are jobs waiting to start
                    time.sleep(5)
                    logging.info("Jobs left to start: %d", len(starting_jobs))
            return starting_jobs

        def wait_for_ended(
            job_scheduling_info_list: Sequence[JobSchedulingInformation]
            | ValuesView[JobSchedulingInformation],
            deadline: datetime,
            check_time: timedelta,
        ) -> list[JobSchedulingInformation]:
            sleep_time = int(round(check_time.total_seconds()))
            logging.debug(
                "Wait for ending in %d jobs up to %s with sleeps of %ss",
                len(job_scheduling_info_list),
                deadline,
                sleep_time,
            )
            # Wait for jobs to complete
            self.wait_all_jobs(
                job_scheduling_info_list,
                True,
                STATEGROUP.ENDED,
                deadline,
                sleep_time,
            )
            ended_jobs = handle_ended_jobs(False, job_scheduling_info_list)
            logging.info(
                "Jobs remaining = %d after %.3fs",
                len(job_scheduling_info_list) - len(ended_jobs),
                (datetime.now() - wait_begin_time).total_seconds(),
            )
            return ended_jobs

        def handle_ended_jobs(
            update: bool,
            job_scheduling_info_list: Sequence[JobSchedulingInformation]
            | ValuesView[JobSchedulingInformation],
        ) -> list[JobSchedulingInformation]:
            ended_jobs = []
            for job_scheduling_info in job_scheduling_info_list:
                if update:
                    self.fetch_and_update_state(job_scheduling_info)
                assert job_scheduling_info.status_info
                if job_scheduling_info.status_info.current_state in STATEGROUP.ENDED:
                    logging.debug("Removing ended %i", job_scheduling_info.job_id)
                    ended_jobs.append(job_scheduling_info)
            return ended_jobs

        def handle_timeouts(
            job_scheduling_info_list: Sequence[JobSchedulingInformation]
            | ValuesView[JobSchedulingInformation],
        ) -> list[JobSchedulingInformation]:
            deadlines = ((jsi, get_deadline(jsi)) for jsi in job_scheduling_info_list)
            timed_out_jobs = [
                jsi
                for jsi, deadline in deadlines
                if deadline is not None and deadline < datetime.now()
            ]
            for job_scheduling_info in timed_out_jobs:
                logging.warning(
                    "Job %i timed out. Terminating job now.",
                    job_scheduling_info.job_id,
                )
                self.client.cancel_job(job_scheduling_info.job_id)
            return timed_out_jobs

        time.sleep(10)
        ended_jobs = {
            jsi.job_id: jsi
            for jsi in handle_ended_jobs(
                True, job_scheduling_info_list=job_scheduling_info_list
            )
        }
        not_started = [
            jsi
            for jsi in job_scheduling_info_list
            if jsi.status_info.current_state in STATEGROUP.STARTING
        ]
        logging.debug("Ended jobs: %s", ended_jobs)
        job_scheduling_info_dict = {jsi.job_id: jsi for jsi in job_scheduling_info_list}
        unfinished_jobs = {
            k: job_scheduling_info_dict[k]
            for k in set(job_scheduling_info_dict.keys()) - ended_jobs.keys()
        }
        logging.debug("Unfinished jobs: %s", len(unfinished_jobs))

        timed_out_jobs = {
            jsi.job_id: jsi
            for jsi in handle_timeouts(
                job_scheduling_info_list=job_scheduling_info_list
            )
        }
        logging.debug("Timed out jobs: %s", timed_out_jobs)

        running_jobs = {
            k: unfinished_jobs[k]
            for k in set(unfinished_jobs.keys()) - timed_out_jobs.keys()
        }

        if not running_jobs:
            logging.info("All jobs ended before wait began")
            for jsi in job_scheduling_info_list:
                self.fetch_and_update_state(jsi)
            return

        logging.info("Jobs running: %i", len(running_jobs))
        time.sleep(10)

        try:
            while datetime.now() < wait_deadline and len(running_jobs) > 0:
                # Handle none started (empty deadline list)
                next_deadline = min(
                    [
                        deadline
                        for deadline in (
                            get_deadline(jsi, True) for jsi in running_jobs.values()
                        )
                        if deadline is not None
                    ]
                )
                check_time = min(
                    ((next_deadline - datetime.now()) / 2), timedelta(minutes=1)
                )

                if not_started:
                    not_started = handle_not_started(not_started, check_time=check_time)
                    logging.info("Not started: %i", len(not_started))

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
                for jsi in running_jobs.values():
                    try:
                        logging.info(
                            "Waiting for jobs timed out. Terminating job %i now.",
                            jsi.job_id,
                        )
                        self.client.cancel_job(jsi.job_id)
                        timed_out_jobs[jsi.job_id] = jsi
                    except Exception:
                        logging.error(
                            "Unknown error occurred terminating job %i",
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
            assert status_info
            stdout_path = job_scheduling_info.get_stdout_path()
            logging.debug("Retrieving info for job %i", job_id)

            # Check job states against expected possible options:
            state = status_info.current_state
            if state == SLURMSTATE.FAILED:
                status_info.final_state = SLURMSTATE.FAILED
                logging.error(
                    "Job %i failed. Dispatch time: %s; Wall time: %s.",
                    job_id,
                    status_info.time_to_dispatch,
                    status_info.wall_time,
                )

            elif not stdout_path.is_file():
                status_info.final_state = SLURMSTATE.NO_OUTPUT
                logging.error(
                    "Job %i with args %s has not created output file '%s'. State: %s. Dispatch time: %s; Wall time: %s.",
                    job_id,
                    job_scheduling_info.job_script_arguments,
                    stdout_path,
                    state,
                    status_info.time_to_dispatch,
                    status_info.wall_time,
                )

            elif not self.timestamp_ok(
                stdout_path,
                start_time=status_info.start_time,
            ):
                status_info.final_state = SLURMSTATE.OLD_OUTPUT_FILE
                logging.error(
                    "Job %i with args %s has not created a new output file '%s'. State: %s. Dispatch time: %s; Wall time: %s.",
                    job_id,
                    job_scheduling_info.job_script_arguments,
                    stdout_path,
                    state,
                    status_info.time_to_dispatch,
                    status_info.wall_time,
                )

            elif state == SLURMSTATE.COMPLETED:
                job_scheduling_info.set_completion_status(True)
                status_info.final_state = SLURMSTATE.COMPLETED
                if status_info.cpus and status_info.wall_time:
                    cpu_time = str(status_info.wall_time * status_info.cpus)
                else:
                    cpu_time = "n/a"
                logging.info(
                    "Job %i with args %s completed. CPU time: %s; Slots: %i. Dispatch time: %s; Wall time: %s.",
                    job_id,
                    job_scheduling_info.job_script_arguments,
                    cpu_time,
                    status_info.cpus,
                    status_info.time_to_dispatch,
                    status_info.wall_time,
                )
            else:
                status_info.final_state = state
                logging.error(
                    "Job %i ended with job state %s. Args %s. Dispatch time: %s; Wall time: %s.",
                    job_id,
                    status_info.final_state,
                    job_scheduling_info.job_script_arguments,
                    status_info.time_to_dispatch,
                    status_info.wall_time,
                )

        self.job_history.append({jsi.job_id: jsi for jsi in job_scheduling_info_list})

    def resubmit_jobs(
        self, job_ids: list[int] | None = None, batch: int | None = None
    ) -> bool:
        old_job_scheduling_info_dict = self.get_job_history_batch(batch_number=batch)
        new_job_scheduling_info_list = []
        for job_id, old_job_scheduling_info in old_job_scheduling_info_dict.items():
            if job_ids is None or job_id in job_ids:
                new_job_scheduling_info = deepcopy(old_job_scheduling_info)
                new_job_scheduling_info.set_completion_status(False)
                new_job_scheduling_info.status_info = None
                new_job_scheduling_info.job_id = -1
                new_job_scheduling_info_list.append(new_job_scheduling_info)
        logging.info("Resubmitting jobs from batch %s with job_ids: %s", batch, job_ids)
        return self._submit_and_monitor(new_job_scheduling_info_list)

    def filter_killed_jobs(
        self, job_scheduling_information_list: list[JobSchedulingInformation]
    ) -> list[JobSchedulingInformation]:
        return [
            jsi
            for jsi in job_scheduling_information_list
            if jsi.status_info and jsi.status_info.current_state == SLURMSTATE.CANCELLED
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
                if jsi.status_info
                and jsi.status_info.final_state != SLURMSTATE.COMPLETED
            ]
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            logging.info(
                f"Total failed_jobs: {len(failed_jobs)}."
                f" Total killed_jobs: {len(killed_jobs)}"
            )
            if killed_jobs:
                return self.resubmit_jobs(
                    job_ids=[jsi.job_id for jsi in killed_jobs], batch=batch_number
                )
            return True
        pretty_format_job_history = "\n".join(
            f"Batch {i} - {', '.join(f'{jsi.job_id}: {jsi.status_info}' for jsi in batch.values())}"  # noqa: E501
            for i, batch in enumerate(self.job_history, 0)
        )
        raise RuntimeError(
            f"All jobs failed. job_history: {pretty_format_job_history}\n"
        )

    def clear_job_history(self) -> None:
        self.job_history.clear()

    def get_job_history_batch(
        self, batch_number: int | None = None
    ) -> dict[int, JobSchedulingInformation]:
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
