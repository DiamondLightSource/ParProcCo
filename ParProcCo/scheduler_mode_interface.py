from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


class SchedulerModeInterface:

    def __init__(self) -> None:
        self.number_jobs: int
        self.cores: int
        self.program_path: Optional[Path]

    def set_parameters(self, sliced_results: List) -> None:
        """Sets parameters for generating jobscript args for use within JobScheduler"""
        raise NotImplementedError

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Generates output, stdout and stderr file paths for job template within JobScheduler"""
        raise NotImplementedError

    def generate_args(self, job_number: int, memory: str, cores: int, jobscript_args: List[str],
                      output_fp: str) -> Tuple[str, ...]:
        """Generates jobscript args for use within JobScheduler"""
        raise NotImplementedError
