from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import slice_to_string, check_script_is_readable, check_location, format_timestamp, get_absolute_path


class SimpleProcessingMode(SchedulerModeInterface):
    def __init__(self, program: Optional[Path] = None) -> None:
        self.program_name: Optional[str] = program
        self.cores = 1
        self.allowed_modules = ('python',)

    def set_parameters(self, slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.slice_params = slice_params
        self.number_jobs = len(slice_params)

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int, t: datetime) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        timestamp = format_timestamp(t)
        output_file = f"out_{i}"
        output_fp = str(output_dir / output_file) if output_dir else output_file
        stdout_fp = str(error_dir / f"out_{timestamp}_{i}")
        stderr_fp = str(error_dir / f"err_{timestamp}_{i}")
        return output_fp, stdout_fp, stderr_fp

    def generate_args(self, i: int, memory: str, cores: int, script_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        slice_param = slice_to_string(self.slice_params[i])
        script = str(check_script_is_readable(check_location(get_absolute_path(script_args[0]))))
        args = tuple([script, "--memory", memory, "--cores", str(cores), "--output", output_fp, "--images", slice_param] + script_args[1:])
        return args
