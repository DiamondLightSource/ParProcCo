from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .data_slicer_interface import DataSlicerInterface
from .job_slicer_interface import JobSlicerInterface

if TYPE_CHECKING:
    from pathlib import Path


class ProgramWrapper:
    def __init__(
        self,
        processing_slicer: JobSlicerInterface,
        slicer: DataSlicerInterface | None = None,
        aggregating_slicer: JobSlicerInterface | None = None,
    ):
        self.processing_slicer = processing_slicer
        self.slicer = slicer
        self.aggregating_slicer = aggregating_slicer

    def get_args(self, args: list[str], debug: bool = False) -> list[str]:
        """
        Get arguments given passed-in arguments
        args  -- given arguments
        debug -- if True, add debug option to arguments if available for wrapped program
        """
        return args

    def get_output(
        self, output: str | None, _program_args: list[str] | None
    ) -> Path | None:
        return Path(output) if output else None

    def create_slices(
        self, number_jobs: int, stop: int | None = None
    ) -> list[slice] | None:
        if number_jobs == 1 or self.slicer is None:
            return None
        return self.slicer.slice(number_jobs, stop)

    def get_process_script(self) -> Path | None:
        return self.processing_slicer.job_script if self.processing_slicer else None

    def get_aggregate_script(self) -> Path | None:
        return self.aggregating_slicer.job_script if self.aggregating_slicer else None

    def get_environment(self) -> dict[str, str] | None:
        test_modules = os.getenv("TEST_PPC_MODULES")
        if test_modules:
            return {"PPC_MODULES": test_modules}

        loaded_modules = os.getenv("LOADEDMODULES", "").split(":")
        logging.debug("Modules are %s", loaded_modules)
        allowed = self.processing_slicer.allowed_modules
        logging.debug(
            "Allowed include %s from %s", allowed, type(self.processing_slicer)
        )
        ppc_modules = []
        if allowed:
            for m in loaded_modules:
                if m and m.split("/")[0] in allowed:
                    ppc_modules.append(m)
        else:
            for m in reversed(loaded_modules):
                if m:
                    ppc_modules.append(m)
                    break

        logging.debug("Passing through %s", ppc_modules)
        if ppc_modules:
            return {"PPC_MODULES": ":".join(ppc_modules)}

        return None
