from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import slice_to_string


class SimpleProcessingModeInterface(SchedulerModeInterface):

    def set_parameters(self, slice_params: List[slice], number_jobs: int) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        assert(len(slice_params) == number_jobs)
        self.slice_params = slice_params
        self.number_jobs = number_jobs

    def generate_output_paths(self, output_dir: Path, error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        output_file = f"out_{i}"
        std_out_file = f"std_out_{i}"
        err_file = f"err_{i}"
        output_fp = str(output_dir / output_file)
        std_out_fp = str(error_dir / std_out_file)
        err_fp = str(error_dir / err_file)
        return output_fp, std_out_fp, err_fp

    def generate_args(self, i: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        slice_param = slice_to_string(self.slice_params[i])
        args = tuple([jobscript_args[0], "--output", output_fp, "--images", slice_param] + jobscript_args[1:])
        return args
