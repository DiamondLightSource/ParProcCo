from __future__ import annotations

from typing import Dict, List, Optional

from pathlib import Path

from .slicer_interface import SlicerInterface
from .scheduler_mode_interface import SchedulerModeInterface
from .utils import get_filepath_on_path

import logging
import os

class ProgramWrapper:

    def __init__(self, processing_mode: SchedulerModeInterface, slicer : Optional[SlicerInterface] = None,
                 aggregating_mode: Optional[SchedulerModeInterface] = None):
        self.processing_mode = processing_mode
        self.slicer = slicer
        self.aggregating_mode = aggregating_mode
        self.cluster_module: Optional[str] = None

    def set_module(self, module: str):
        self.cluster_module = module

    def set_cores(self, cores: int):
        self.processing_mode.cores = cores

    def get_args(self, args: List[str], debug: bool = False):
        '''
        Get arguments given passed-in arguments
        args  -- given arguments
        debug -- if True, add debug option to arguments if available for wrapped program
        '''
        return args

    def create_slices(self, number_jobs: int, stop: Optional[int] = None) -> List[Optional[slice]]:
        if number_jobs == 1 or self.slicer is None:
            return [None]
        return self.slicer.slice(number_jobs, stop)

    def get_output(self, output: Optional[str], _program_args: Optional[List[str]]) -> Optional[Path]:
        return Path(output) if output else None

    def get_aggregate_script(self) -> Optional[Path]:
        return get_filepath_on_path(self.aggregating_mode.program_name) if self.aggregating_mode else None

    def get_cluster_runner_script(self) -> Optional[Path]:
        return get_filepath_on_path(self.processing_mode.program_name)

    def get_environment(self) -> Optional[Dict[str,str]]:
        test_modules = os.getenv('TEST_PPC_MODULES')
        if test_modules:
            return {"PPC_MODULES": test_modules}

        loaded_modules = os.getenv('LOADEDMODULES', '').split(':')
        logging.debug('Modules are %s', loaded_modules)
        allowed = self.processing_mode.allowed_modules
        logging.debug('Allowed include %s from %s', allowed, type(self.processing_mode))
        ppc_modules = []
        if allowed:
            for m in loaded_modules:
                if m and m.split('/')[0] in allowed:
                    ppc_modules.append(m)
        else:
            for m in reversed(loaded_modules):
                if m and m != self.cluster_module:
                    ppc_modules.append(m)
                    break

        logging.debug('Passing through %s', ppc_modules)
        if ppc_modules:
            return {'PPC_MODULES': ':'.join(ppc_modules)}

        return None
