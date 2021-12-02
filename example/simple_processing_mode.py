from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import slice_to_string, check_jobscript_is_readable, check_location, get_absolute_path


class SimpleProcessingMode(SchedulerModeInterface):
    def __init__(self, program: Optional[Path] = None) -> None:
        self.program_path: Optional[Path] = program
        self.cores = 1
        self.allowed_modules = ('python',)

    def set_parameters(self, slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.slice_params = slice_params
        self.number_jobs = len(slice_params)

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        output_file = f"out_{i}"
        output_fp = str(output_dir / output_file) if output_dir else output_file
        stdout_fp = str(error_dir / f"out_{i}")
        stderr_fp = str(error_dir / f"err_{i}")
        return output_fp, stdout_fp, stderr_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        slice_param = slice_to_string(self.slice_params[i])
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = tuple([jobscript, "--memory", memory, "--cores", str(cores), "--output", output_fp, "--images", slice_param] + jobscript_args[1:])
        return args
