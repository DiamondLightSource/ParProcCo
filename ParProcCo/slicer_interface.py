from __future__ import annotations


class SlicerInterface:
    def slice(self, number_jobs: int, stop: int | None = None) -> list[slice] | None:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError
