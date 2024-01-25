from __future__ import annotations

from ParProcCo.data_slicer_interface import DataSlicerInterface


class SimpleDataSlicer(DataSlicerInterface):
    def __init__(self):
        pass

    def slice(self, number_jobs: int, stop: int | None = None) -> list[slice] | None:
        """Overrides DataSlicerInterface.slice"""
        if not isinstance(number_jobs, int):
            raise TypeError(f"number_jobs is {type(number_jobs)}, should be int\n")

        if (stop is not None) and not isinstance(stop, int):
            raise TypeError(f"stop is {type(stop)}, should be int or None\n")

        if stop:
            number_jobs = min(stop, number_jobs)
        if number_jobs < 1:
            return None
        slices: list[slice] = [slice(i, stop, number_jobs) for i in range(number_jobs)]
        return slices
