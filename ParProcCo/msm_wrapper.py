from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from .nxdata_aggregation_mode import NXdataAggregationMode
from .program_wrapper import ProgramWrapper
from .scheduler_mode_interface import SchedulerModeInterface
from .simple_data_slicer import SimpleDataSlicer
from .utils import slice_to_string, check_jobscript_is_readable, check_location, get_absolute_path


class MSMProcessingMode(SchedulerModeInterface):
    PPC_Modules = "python/3.9:msmapper/1.4"

    def __init__(self):
        current_script_dir = Path(os.path.realpath(__file__)).parent.parent / "scripts"
        self.program_path = current_script_dir / "ppc_cluster_runner"
        self.cores = 6
        self.environment = {"PPC_MODULES":MSMProcessingMode.PPC_Modules}

    def set_parameters(self, slice_params: List[slice]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.slice_params = slice_params
        self.number_jobs = len(slice_params)
        assert(self.number_jobs > 1)

    def generate_output_paths(self, output_dir: Path, error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        output_file = f"out_{i}.nxs"
        std_out_file = f"std_out_{i}"
        err_file = f"err_{i}"
        output_fp = str(output_dir / output_file)
        std_out_fp = str(error_dir / std_out_file)
        err_fp = str(error_dir / err_file)
        return output_fp, std_out_fp, err_fp

    def generate_args(self, i: int, memory: str, cores: int, jobscript_args: List[str], output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i < self.number_jobs)
        # TODO move images argument to wrapper
        # refactor job_scheduler too
        # or move this to msmapper-utils
        slice_param = slice_to_string(self.slice_params[i])
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = tuple([jobscript, "--memory", memory, "--cores", str(cores), "--output", output_fp, "--images", slice_param] + jobscript_args[1:])
        return args

class MSMWrapper(ProgramWrapper):
    def __init__(self):
        super().__init__(MSMProcessingMode(), SimpleDataSlicer(), NXdataAggregationMode())

# TODO
# from msmapper_utils.msm_run import add_options, parse_dls_mode
# def get_output(self, output: str, program_args: Optional[List[str]]) -> Path:
#      parser = argparse.ArgumentParser(description='Internal parser')
#      add_options(parser)
#      args = parser.parse_args(program_args)
#      dls_mode, auto = parse_dls_mode(args)
#      return Path('/')
