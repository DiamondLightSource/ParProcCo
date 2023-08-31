from __future__ import annotations

from typing import List, Optional


class SlicerInterface:
    def slice(
        self, number_jobs: int, stop: Optional[int] = None
    ) -> List[Optional[slice]]:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError
