from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .program_wrapper import ProgramWrapper
from .scheduler_mode_interface import SchedulerModeInterface
from .utils import check_jobscript_is_readable, check_location, get_absolute_path

import os

class PassThruProcessingMode(SchedulerModeInterface):
    PPC_Modules = "python/3.9"

    def __init__(self):
        self.cores = 6
        current_script_dir = Path(os.path.realpath(__file__)).parent.parent / "scripts"
        self.program_path = current_script_dir / "ppc_cluster_runner"
        self.environment = {"PPC_MODULES":PassThruProcessingMode.PPC_Modules}

    def set_parameters(self, _slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.number_jobs = 1

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        stdout_fp = str(error_dir / f"out_{i}")
        stderr_fp = str(error_dir / f"err_{i}")
        return str(output_dir) if output_dir else '', stdout_fp, stderr_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str],
                      output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = [jobscript, "--memory", memory, "--cores", str(cores)]
        if output_fp:
            args += ("--output", output_fp)
        args += jobscript_args[1:]
        return tuple(args)

class PassThruWrapper(ProgramWrapper):

    def __init__(self):
        super().__init__(PassThruProcessingMode())
