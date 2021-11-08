from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import check_jobscript_is_readable, check_location, get_absolute_path


class PassThruProcessingMode(SchedulerModeInterface):

    def set_parameters(self, slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.number_jobs = 1

    def generate_output_paths(self, output_dir: Path, error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        std_out_fp = str(error_dir / f"std_out_{i}")
        err_fp = str(error_dir / f"err_{i}")
        return output_dir, std_out_fp, err_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = [jobscript, "--memory", memory, "--cores", str(cores), "--output", output_fp] + jobscript_args[1:]
        return tuple(args)
