from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import check_jobscript_is_readable, check_location, get_absolute_path


class SimpleAggregationMode(SchedulerModeInterface):

    def __init__(self, program: Path) -> None:
        self.program_path = program
        self.cores = 1

    def set_parameters(self, sliced_results: List[Path]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.sliced_results = [str(res) for res in sliced_results]
        self.number_jobs: int = 1

    def generate_output_paths(self, output_dir: Path, error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        output_file = f"aggregated_results.txt"
        std_out_file = f"std_out_aggregated"
        err_file = f"err_aggregated"
        output_fp = str(output_dir / output_file)
        std_out_fp = str(error_dir / std_out_file)
        err_fp = str(error_dir / err_file)
        return output_fp, std_out_fp, err_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i == 0)
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = tuple([jobscript, "--output", output_fp] + self.sliced_results)
        return args
