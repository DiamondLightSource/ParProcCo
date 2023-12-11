from __future__ import annotations

from typing import List, Optional


class SlicerInterface:
    def slice(self, number_jobs: int, stop: Optional[int] = None) -> list[slice] | None:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError
