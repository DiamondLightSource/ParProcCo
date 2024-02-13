from __future__ import annotations

import logging
import unittest
from typing import TYPE_CHECKING

from parameterized import parameterized  # type: ignore

from ParProcCo.simple_data_slicer import SimpleDataSlicer

if TYPE_CHECKING:
    from typing import Any


class TestDataSlicer(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

    @parameterized.expand(
        [
            (
                "all_ok",
                [4],
                {"stop": 8},
                None,
                4,
                [slice(0, 8, 4), slice(1, 8, 4), slice(2, 8, 4), slice(3, 8, 4)],
            ),
            (
                "no_stop",
                [4],
                {},
                None,
                4,
                [
                    slice(0, None, 4),
                    slice(1, None, 4),
                    slice(2, None, 4),
                    slice(3, None, 4),
                ],
            ),
            (
                "stop_not_int",
                [4],
                {"stop": "8"},
                "stop is <class 'str'>, should be int",
                None,
                None,
            ),
            (
                "number_jobs_not_int",
                ["4"],
                {},
                "number_jobs is <class 'str'>, should be int",
                None,
                None,
            ),
            (
                "too_many_slices",
                [20],
                {"stop": 11},
                None,
                11,
                [
                    slice(0, 11, 11),
                    slice(1, 11, 11),
                    slice(2, 11, 11),
                    slice(3, 11, 11),
                    slice(4, 11, 11),
                    slice(5, 11, 11),
                    slice(6, 11, 11),
                    slice(7, 11, 11),
                    slice(8, 11, 11),
                    slice(9, 11, 11),
                    slice(10, 11, 11),
                ],
            ),
        ]
    )
    def test_slices(
        self,
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        error_msg: str | None,
        expected_length: int,
        expected_slices: list[slice],
    ) -> None:
        slicer = SimpleDataSlicer()

        if error_msg:
            with self.assertRaises(TypeError) as context:
                slicer.slice(*args, **kwargs)
            self.assertTrue(error_msg in str(context.exception))
            return

        slices = slicer.slice(*args, **kwargs)
        assert slices is not None
        self.assertEqual(len(slices), expected_length)
        self.assertEqual(slices, expected_slices)


if __name__ == "__main__":
    unittest.main()
