from __future__ import annotations

from pathlib import Path
from typing import List

from job_controller import SlicerInterface


class SimpleDataSlicer(SlicerInterface):

    def __init__(self):
        pass

    def slice(self, number_jobs: int, stop: int = None) -> List[slice]:
        """Overrides SlicerInterface.slice"""
        if type(number_jobs) is not int:
            raise TypeError(f"number_jobs is {type(number_jobs)}, should be int\n")

        if (stop is not None) and (type(stop) is not int):
            raise TypeError(f"stop is {type(stop)}, should be int or None\n")

        if stop:
            number_jobs = min(stop, number_jobs)
        slices = [slice(i, stop, number_jobs) for i in range(number_jobs)]
        return slices
