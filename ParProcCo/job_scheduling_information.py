from __future__ import annotations
from pydantic import BaseModel

from pathlib import Path
from datetime import timedelta, datetime
from dataclasses import dataclass, field

from .utils import check_jobscript_is_readable, get_ppc_dir, format_timestamp


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Union, Dict, List, Tuple
    from .job_scheduler import StatusInfo


@dataclass
class JobResources:
    memory: int = 4000
    cpu_cores: int = 6
    gpus: int = 0
    extra_properties: dict[str, str] = field(default_factory=dict)


@dataclass
class JobSchedulingInformation:
    job_name: str
    job_script_path: Optional[Path]
    job_resources: JobResources
    timeout: timedelta = timedelta(hours=2)
    job_script_arguments: Tuple[str] = field(default_factory=tuple)
    job_env: Dict[str, str] = field(default_factory=dict)
    log_directory: Optional[Path] = None
    stderr_filename: Optional[str] = None
    stdout_filename: Optional[str] = None
    working_directory: Optional[Path] = None
    output_dir: Optional[Path] = None
    output_filename: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.set_job_script_path(self.job_script_path)  # For validation
        self.set_job_env(self.job_env)  # For validation
        # To be updated when submitted, not on creation
        self.job_id: Optional[int] = None
        self.status_info: Optional[StatusInfo] = None
        self.completion_status: bool = False

    def set_job_script_path(self, path: Path) -> None:
        self.job_script_path = check_jobscript_is_readable(path)

    def set_job_env(self, job_env: Optional[Dict[str, str]]) -> None:
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

    # TODO: make this a part of a wrapper that can be set on a per-job basis?
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
