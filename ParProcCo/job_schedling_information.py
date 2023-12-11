from __future__ import annotations
from pydantic import BaseModel

from pathlib import Path
from datetime import timedelta, datetime
from dataclasses import dataclass, field

from .program_wrapper import ProgramWrapper
from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable, get_ppc_dir, format_timestamp


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .job_scheduler import StatusInfo

# TODO: Perhaps JobSchedulingInformation should be passed to a ProgramWrapper, with a default one that just reads from the values given, and complex ones that do slicing etc?


@dataclass
class JobResources:
    memory: int = 4000
    cpu_cores: int = 6
    gpus: int = 0
    extra_properties: dict[str, str] = field(default_factory=dict)


@dataclass
class JobSchedulingInformation(BaseModel):
    job_name: str
    job_script_path: Path | None
    job_resources: JobResources
    timeout: timedelta = timedelta(hours=2)
    job_script_arguments: tuple[str] = field(default_factory=tuple)
    job_env: dict[str, str] = field(default_factory=dict)
    log_directory: Path | None = None
    stderr_filename: str | None = None
    stdout_filename: str | None = None
    working_directory: Path | None = None
    output_path: Path | None = None

    def __post_init__(self):
        self.job_script_path = check_jobscript_is_readable(self.job_script_path)
        self.job_env = (
            self.job_env if self.job_env else {"ParProcCo": "0"}
        )  # job_env cannot be empty dict
        test_ppc_dir = get_ppc_dir()
        if test_ppc_dir:
            self.job_env.update(TEST_PPC_DIR=test_ppc_dir)

        # To be updated when submitted, not on creation
        self.start_time: datetime | None = None
        self.job_id: int | None = None
        self.status_info: StatusInfo | None = None
        self.completion_status: bool = False

    def update_start_time(self, start_time: datetime | None = None) -> None:
        if start_time is None:
            start_time = datetime.now()
        self.start_time = start_time

    def set_job_id(self, job_id: int) -> None:
        self.job_id = job_id

    def update_status_info(self, status_info: StatusInfo) -> None:
        self.status_info = status_info

    def set_completion_status(self, completion_status: bool) -> None:
        self.completion_status = completion_status

    # TODO: make this a part of a wrapper that can be set on a per-job basis?
    def get_stdout_path(self) -> str:
        if self.log_directory is None:
            raise ValueError(
                "The log directory must be set before getting the stdout path"
            )
        if self.stdout_filename is None:
            self.stdout_filename = self._generate_log_filename(suffix="_stdout.txt")
        return self.log_directory / self.stdout_filename

    def get_stderr_path(self) -> str:
        if self.log_directory is None:
            raise ValueError(
                "The log directory must be set before getting the stderr path"
            )
        if self.stderr_filename is None:
            self.stderr_filename = self._generate_log_filename(suffix="_stderr.txt")
        return self.log_directory / self.stderr_filename

    def _generate_log_filename(self, suffix: str) -> str:
        log_filename = self.job_name
        if self.start_time is not None:
            log_filename += f"_{format_timestamp(self.start_time)}"
        log_filename += f"_{suffix}"
        return log_filename
