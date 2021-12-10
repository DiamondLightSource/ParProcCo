from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import check_jobscript_is_readable, check_location, get_absolute_path


class SimpleAggregationMode(SchedulerModeInterface):

    def __init__(self, program: Path) -> None:
        self.program_path = program
        self.cores = 1

    def set_parameters(self, sliced_results: List) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.sliced_results = [str(res) for res in sliced_results]
        self.number_jobs: int = 1

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int, t: datetime.datetime) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        timestamp = f"{t.year}{t.month}{t.day}_{t.hour}{t.minute}"
        output_file = "aggregated_results.txt"
        output_fp = str(output_dir / output_file) if output_dir else output_file
        stdout_fp = str(error_dir / f"out_{timestamp}_aggregated")
        stderr_fp = str(error_dir / f"err_{timestamp}_aggregated")
        return output_fp, stdout_fp, stderr_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i == 0)
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = tuple([jobscript, "--output", output_fp] + self.sliced_results)
        return args
