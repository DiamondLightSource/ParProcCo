from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import check_jobscript_is_readable, format_timestamp, get_ppc_dir

if TYPE_CHECKING:
    from .job_scheduler import StatusInfo


@dataclass
class JobResources:
    memory: int = 4000
    "Memory per cpu in MB"
    cpu_cores: int = 6
    "CPU cores per task"
    gpus: int = 0
    "GPU per task"
    extra_properties: dict[str, str] = field(default_factory=dict)


@dataclass
class JobSchedulingInformation:
    job_name: str
    job_script_path: Path
    job_resources: JobResources
    timeout: timedelta = timedelta(hours=2)
    job_script_arguments: tuple[str, ...] = field(default_factory=tuple)
    job_env: dict[str, str] = field(default_factory=dict)
    log_directory: Path | None = None
    stderr_filename: str | None = None
    stdout_filename: str | None = None
    working_directory: Path | None = None
    output_dir: Path | None = None
    output_filename: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        self.set_job_script_path(self.job_script_path)  # For validation
        self.set_job_env(self.job_env)  # For validation
        # To be updated when submitted, not on creation
        self.job_id: int = -1
        self.status_info: StatusInfo | None = None
        self.completion_status: bool = False

    def set_job_script_path(self, path: Path) -> None:
        self.job_script_path = check_jobscript_is_readable(path)

    def set_job_env(self, job_env: dict[str, str] | None) -> None:
        self.job_env = (
            job_env if job_env else {"ParProcCo": "0"}
        )  # job_env cannot be empty dict
        test_ppc_dir = get_ppc_dir()
        if test_ppc_dir:
            self.job_env.update(TEST_PPC_DIR=test_ppc_dir)

    def set_job_id(self, job_id: int) -> None:
        self.job_id = job_id

    def update_status_info(self, status_info: StatusInfo) -> None:
        self.status_info = status_info

    def set_completion_status(self, completion_status: bool) -> None:
        self.completion_status = completion_status

    def get_output_path(self) -> Path | None:
        if self.output_filename is None:
            return None
        if self.output_dir is None:
            return Path(self.output_filename)
        return self.output_dir / self.output_filename

    def get_stdout_path(self) -> Path:
        if self.log_directory is None:
            raise ValueError(
                "The log directory must be set before getting the stdout path"
            )
        if self.stdout_filename is None:
            self.stdout_filename = self._generate_log_filename(suffix="_stdout.txt")
        return self.log_directory / self.stdout_filename

    def get_stderr_path(self) -> Path:
        if self.log_directory is None:
            raise ValueError(
                "The log directory must be set before getting the stderr path"
            )
        if self.stderr_filename is None:
            self.stderr_filename = self._generate_log_filename(suffix="_stderr.txt")
        return self.log_directory / self.stderr_filename

    def _generate_log_filename(self, suffix: str) -> str:
        log_filename = self.job_name
        if self.timestamp is not None:
            log_filename += f"_{format_timestamp(self.timestamp)}"
        log_filename += f"_{suffix}"
        return log_filename
