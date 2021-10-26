from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


class SchedulerModeInterface:

    def __init__(self) -> None:
        pass

    def set_parameters(self, *args, **kwargs) -> None:
        """Sets parameters for generating jobscript args for use within JobScheduler"""
        raise NotImplementedError

    def generate_output_paths(self, output_dir: Path, error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Generates output, std_out and error file paths for job template within JobScheduler"""
        raise NotImplementedError

    def generate_args(self, job_number: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Generates jobscript args for use within JobScheduler"""
        raise NotImplementedError
