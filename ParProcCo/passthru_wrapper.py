from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .program_wrapper import ProgramWrapper
from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable, check_location, format_timestamp, get_absolute_path


class PassThruProcessingMode(SchedulerModeInterface):
    def __init__(self):
        super().__init__()
        self.cores = 6
        self.program_name = "ppc_cluster_runner"

    def set_parameters(self, _slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.number_jobs = 1

    def generate_output_paths(
        self, output_dir: Optional[Path], error_dir: Path, i: int, t: datetime
    ) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        timestamp = format_timestamp(t)
        stdout_fp = str(error_dir / f"out_{timestamp}_{i}")
        stderr_fp = str(error_dir / f"err_{timestamp}_{i}")
        return str(output_dir) if output_dir else "", stdout_fp, stderr_fp

    def generate_args(
        self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str
    ) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert i < self.number_jobs
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = [jobscript, "--memory", memory, "--cores", str(cores)]
        if output_fp:
            args += ("--output", output_fp)
        args += jobscript_args[1:]
        return tuple(args)


class PassThruWrapper(ProgramWrapper):
    def __init__(self, original_wrapper: ProgramWrapper):
        super().__init__(PassThruProcessingMode())
        self.original_wrapper = original_wrapper
        self.processing_mode.allowed_modules = original_wrapper.processing_mode.allowed_modules

    def get_args(self, args: List[str], debug: bool = False):
        return self.original_wrapper.get_args(args, debug)

    def get_output(self, output: Optional[str] = None, program_args: Optional[List[str]] = None) -> Path:
        return self.original_wrapper.get_output(output, program_args)
