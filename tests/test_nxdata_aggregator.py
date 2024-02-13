from __future__ import annotations

import logging
import unittest
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import pytest
from parameterized import parameterized  # type: ignore

from ParProcCo.nxdata_aggregator import NXdataAggregator
from ParProcCo.test import TemporaryDirectory
from ParProcCo.utils import decode_to_string

from .utils import get_slurm_rest_url

slurm_rest_url = get_slurm_rest_url()
gh_testing = slurm_rest_url is None


class TestNXdataAggregator(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

    def create_basic_nexus_file(self, file_path: Path, has_weight: bool) -> None:
        with h5py.File(file_path, "w") as f:
            default_entry = f.create_group("default_entry")
            default_entry.attrs["NX_class"] = "NXentry"
            f.attrs["default"] = "default_entry"
            default_data = default_entry.create_group("default_data")
            default_data.attrs["NX_class"] = "NXdata"
            default_entry.attrs["default"] = "default_data"
            default_data.attrs["axes"] = ["a-axis", "b-axis", "c-axis"]
            default_data.attrs["signal"] = "volume"

            if has_weight:
                default_data.attrs["auxiliary_signals"] = ["weight"]

    def test_decode_to_string_input_is_string(self):
        name = "name"
        name = decode_to_string(name)
        self.assertEqual(name, "name")

    def test_decode_to_string_input_is_bytes(self):
        name = b"name"
        name = decode_to_string(name)
        self.assertEqual(name, "name")

    @pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
    def test_renormalise(self) -> None:
        output_file_paths = [
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"
            ),
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs"
            ),
        ]
        aggregator = NXdataAggregator()
        aggregator._renormalise(output_file_paths)
        with h5py.File(
            "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs", "r"
        ) as f:
            volumes_array = NXdataAggregator._require_dataset(
                f, "processed/reciprocal_space/volume"
            )[...]
            weights_array = NXdataAggregator._require_dataset(
                f, "processed/reciprocal_space/weight"
            )[...]
        np.testing.assert_allclose(
            aggregator.accumulator_volume, volumes_array, rtol=1e-12
        )
        np.testing.assert_allclose(
            aggregator.accumulator_weights, weights_array, rtol=2.1e-14
        )

    @pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
    def test_initialise_arrays_applied_data(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_dimensions = 3
        aggregator.data_files = [
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"
            ),
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs"
            ),
        ]
        aggregator.nxentry_name = "processed"
        aggregator.nxdata_name = "reciprocal_space"
        aggregator.nxdata_path_name = "processed/reciprocal_space"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["h-axis", "k-axis", "l-axis"]
        aggregator.axes_spacing = [0.02, 0.02, 0.02]
        aggregator.renormalisation = True
        aggregator.aux_signal_names = ["weight"]
        aggregator.non_weight_aux_signal_names = []

        aggregator._initialise_arrays()

        self.assertEqual(aggregator.axes_mins, [-0.2, -0.08, 0.86])
        np.testing.assert_allclose(aggregator.axes_maxs, [1.44, 1.44, 1.1])

        self.assertEqual(aggregator.accumulator_axis_lengths, [83, 77, 13])
        self.assertEqual(len(aggregator.accumulator_axis_ranges), 3)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[0]), 83)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[1]), 77)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[2]), 13)
        self.assertEqual(aggregator.accumulator_axis_ranges[0][0], -0.2)
        self.assertAlmostEqual(
            aggregator.accumulator_axis_ranges[0][-1], 1.44, places=14
        )
        self.assertEqual(aggregator.accumulator_axis_ranges[1][0], -0.08)
        self.assertEqual(aggregator.accumulator_axis_ranges[1][-1], 1.44)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][0], 0.86)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][-1], 1.1)
        self.assertTrue(
            np.array_equal(aggregator.accumulator_volume, np.zeros([83, 77, 13]))
        )
        self.assertTrue(
            np.array_equal(aggregator.accumulator_weights, np.zeros([83, 77, 13]))
        )
        self.assertEqual(
            aggregator.all_slices,
            [
                (slice(0, 83), slice(0, 77), slice(0, 13)),
                (slice(0, 83), slice(0, 77), slice(0, 13)),
            ],
        )

    @parameterized.expand(
        [
            ("normal", (2, 3, 4), True, ["weight"], [], None, None),
            ("no_aux", (2, 3, 4), False, None, [], None, None),
            (
                "axes_wrong",
                (2, 4, 3),
                True,
                ["weight"],
                [],
                AssertionError,
                "axes_lengths must equal volumes_array.shape",
            ),
            (
                "non_weight_signals",
                (2, 3, 4),
                False,
                ["other_0", "other_1"],
                ["other_0", "other_1"],
                None,
                None,
            ),
            (
                "non_weight_signals_plus_weight",
                (2, 3, 4),
                True,
                ["weight", "other_0", "other_1"],
                ["other_0", "other_1"],
                None,
                None,
            ),
        ]
    )
    def test_initialise_arrays(
        self,
        name,
        shape,
        has_weight,
        aux_signal_names,
        non_weight_names,
        error_name,
        error_msg,
    ) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_dimensions = 3
        aggregator.nxentry_name = "default_entry"
        aggregator.nxdata_name = "default_data"
        aggregator.nxdata_path_name = "default_entry/default_data"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["a-axis", "b-axis", "c-axis"]
        aggregator.axes_spacing = [0.2, 0.2, 0.2]
        aggregator.renormalisation = has_weight
        aggregator.aux_signal_names = aux_signal_names
        aggregator.non_weight_aux_signal_names = non_weight_names

        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, has_weight)

            with h5py.File(file_path_0, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), shape)
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    aux_data = np.reshape(
                        np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4)
                    )
                    for name in aux_signal_names:
                        nxdata_group.create_dataset(name, data=aux_data)

            with h5py.File(file_path_1, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2, 0.4])
                nxdata_group.create_dataset("b-axis", data=[1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2, 0.4])
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    aux_data = np.reshape(
                        np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5)
                    )
                    for name in aux_signal_names:
                        nxdata_group.create_dataset(name, data=aux_data)

            if error_name:
                with self.assertRaises(error_name) as context:
                    aggregator._initialise_arrays()
                    self.assertTrue(error_msg in str(context.exception))
                return

            aggregator._initialise_arrays()

            self.assertEqual(aggregator.axes_mins, [0.0, 1.0, -0.4])
            self.assertEqual(aggregator.axes_maxs, [0.4, 1.4, 0.4])
            self.assertEqual(
                aggregator.all_slices,
                [
                    (slice(0, 2), slice(0, 3), slice(0, 4)),
                    (slice(0, 3), slice(1, 3), slice(0, 5)),
                ],
            )
            self.assertEqual(aggregator.accumulator_axis_lengths, [3, 3, 5])
            for la, e in zip(
                aggregator.accumulator_axis_ranges,
                [[0.0, 0.2, 0.4], [1.0, 1.2, 1.4], [-0.4, -0.2, 0.0, 0.2, 0.4]],
            ):
                np.testing.assert_allclose(np.array(la), e, rtol=1e-14)
            self.assertTrue(
                np.array_equal(aggregator.accumulator_volume, np.zeros([3, 3, 5]))
            )
            self.assertEqual(aggregator.non_weight_aux_signal_names, non_weight_names)
            self.assertEqual(aggregator.aux_signal_names, aux_signal_names)
            if has_weight:
                self.assertTrue(
                    np.array_equal(aggregator.accumulator_weights, np.zeros([3, 3, 5]))
                )
            else:
                self.assertFalse(hasattr(aggregator, "accumulator_weights"))

            if non_weight_names:
                self.assertEqual(len(aggregator.accumulator_aux_signals), 2)
                for signal in aggregator.accumulator_aux_signals:
                    np.testing.assert_array_equal(signal, np.zeros((3, 3, 5)))
            else:
                self.assertEqual(aggregator.accumulator_aux_signals, [])

    @pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
    def test_get_nxdata_applied_data(self) -> None:
        output_data_files = [
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"
            ),
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs"
            ),
        ]
        aggregator = NXdataAggregator()
        aggregator.data_files = output_data_files
        aggregator._get_nxdata()
        self.assertEqual(aggregator.nxentry_name, "processed")
        self.assertEqual(aggregator.nxdata_name, "reciprocal_space")
        self.assertEqual(aggregator.aux_signal_names, ["weight"])
        self.assertEqual(aggregator.non_weight_aux_signal_names, [])
        self.assertEqual(aggregator.renormalisation, True)
        self.assertEqual(aggregator.signal_name, "volume")
        self.assertEqual(aggregator.axes_names, ["h-axis", "k-axis", "l-axis"])
        self.assertEqual(aggregator.data_dimensions, 3)

    @parameterized.expand(
        [
            ("normal", (2, 3, 4), True, None, None),
            ("no_aux", (2, 3, 4), False, None, None),
            (
                "shape_wrong",
                (2, 4, 3),
                True,
                AssertionError,
                "signal and weight shapes must match",
            ),
            (
                "dims_wrong",
                (2, 12),
                True,
                AssertionError,
                "signal and weight dimensions must match",
            ),
        ]
    )
    def test_get_nx_data_param(
        self, name, shape, has_weight, error_name, error_msg
    ) -> None:
        aggregator = NXdataAggregator()
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            aggregator.data_files = [file_path]

            self.create_basic_nexus_file(file_path, has_weight)
            with h5py.File(file_path, "r+") as f:
                nxdata_group = f.require_group("default_entry/default_data")
                nxdata_group.attrs["a-axis_indices"] = 0
                nxdata_group.attrs["b-axis_indices"] = 1
                nxdata_group.attrs["c-axis_indices"] = 2
                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if has_weight:
                    weight_data = np.reshape(
                        np.array([i * 2 + 3 for i in range(24)]), shape
                    )
                    nxdata_group.create_dataset("weight", data=weight_data)

            if error_name:
                with self.assertRaises(error_name) as context:
                    aggregator._get_nxdata()
                self.assertTrue(error_msg in str(context.exception))
                return

            aggregator._get_nxdata()

        self.assertEqual(aggregator.nxentry_name, "default_entry")
        self.assertEqual(aggregator.nxdata_name, "default_data")
        self.assertEqual(aggregator.signal_name, "volume")
        self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])
        self.assertEqual(aggregator.data_dimensions, 3)
        self.assertEqual(aggregator.renormalisation, has_weight)
        if has_weight:
            self.assertEqual(aggregator.aux_signal_names, ["weight"])
            self.assertEqual(aggregator.non_weight_aux_signal_names, [])
        else:
            self.assertEqual(aggregator.aux_signal_names, None)
            self.assertEqual(aggregator.non_weight_aux_signal_names, [])

    @parameterized.expand(
        [
            ("normal", "NXentry", "default_entry", True, None, None),
            (
                "wrong_class",
                "NXprocess",
                "default_entry",
                True,
                AssertionError,
                "default_entry class_name must be NXentry",
            ),
            ("bytes", "NXentry", b"default_entry", True, None, None),
            ("no_default", "NXentry", "default_entry", False, None, None),
            (
                "no_default_no_class",
                None,
                "default_entry",
                False,
                ValueError,
                "no NXentry group found",
            ),
            (
                "no_default_wrong_class",
                "NXprocess",
                "default_entry",
                False,
                ValueError,
                "no NXentry group found",
            ),
            ("no_groups", None, None, False, ValueError, "no NXentry group found"),
        ]
    )
    def test_get_default_nxgroup(
        self, name, default_class, default_name, has_default, error_name, error_msg
    ) -> None:
        aggregator = NXdataAggregator()
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, "w") as f:
                default_group = f.create_group("default_entry")
                if default_class:
                    default_group.attrs["NX_class"] = default_class
                if has_default:
                    f.attrs["default"] = default_name

                if error_name:
                    with self.assertRaises(error_name) as context:
                        aggregator._get_default_nxgroup(f, "NXentry")
                    self.assertTrue(error_msg in str(context.exception))
                    return

                nxentry_name = aggregator._get_default_nxgroup(f, "NXentry")

        self.assertEqual(nxentry_name, "default_entry")

    def test_get_nxgroup_missing_external_file(self) -> None:
        aggregator = NXdataAggregator()
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            linked_file_path = Path(working_directory) / "linked_file.nxs"
            with h5py.File(file_path, "w") as f:
                f["entry0"] = h5py.ExternalLink(str(linked_file_path), "missing_group")
                entry_group = f.create_group("entry1")
                entry_group.attrs["NX_class"] = "NXentry"

                with self.assertLogs(level="WARNING") as cm:
                    aggregator._get_default_nxgroup(f, "NXentry")
                    self.assertEqual(
                        cm.output,
                        [
                            'WARNING:root:KeyError: entry0 could not be accessed in <HDF5 file "output.nxs" (mode r+)>'
                        ],
                    )

    @parameterized.expand(
        [
            ("normal", "volume", ["weight"], True, [], None),
            ("no_aux_signals", "volume", None, False, [], None),
            ("no_weight_signal", "volume", ["other"], False, ["other"], None),
            ("extra_aux_signals", "volume", ["weight", "other"], True, ["other"], None),
            ("data_as_signal", "data", ["weight"], True, [], None),
            ("no_axes", "volume", ["weight"], True, [], None),
            ("no_signal_or_data", None, ["weight"], True, [], KeyError),
            ("other_signal", "other", ["weight"], True, [], KeyError),
        ]
    )
    def test_get_default_signals_and_axes(
        self, name, signal, aux_signals, renormalisation, non_weight_signals, error_name
    ) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxentry_name = "entry_group"
        aggregator.nxdata_name = "data_group"
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, "w") as f:
                entry_group = f.create_group("entry_group")
                data_group = entry_group.create_group("data_group")
                data_group.attrs["NX_class"] = "NXdata"
                data_group.attrs["axes"] = ["a-axis", "b-axis", "c-axis"]
                if signal:
                    if signal == "volume":
                        data_group.attrs["signal"] = signal
                    elif signal == "data":
                        data_group.create_dataset(signal, (3,))
                if aux_signals:
                    data_group.attrs["auxiliary_signals"] = aux_signals
                if error_name:
                    with self.assertRaises(error_name):
                        aggregator._get_default_signals_and_axes(data_group)
                else:
                    aggregator._get_default_signals_and_axes(data_group)

        self.assertEqual(aggregator.non_weight_aux_signal_names, non_weight_signals)
        self.assertEqual(aggregator.aux_signal_names, aux_signals)
        self.assertEqual(aggregator.renormalisation, renormalisation)
        if not error_name:
            self.assertEqual(aggregator.signal_name, signal)
            self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])

    def test_get_default_signals_and_axes_no_axes(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_files = []
        aggregator.nxentry_name = "entry_group"
        aggregator.nxdata_name = "data_group"
        aggregator.nxdata_path_name = "entry_group/data_group"

        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            aggregator.data_files = [
                Path(working_directory) / f"output{i}.nxs" for i in range(3)
            ]
            for file_path in aggregator.data_files:
                with h5py.File(file_path, "w") as f:
                    entry_group = f.create_group("entry_group")
                    data_group = entry_group.create_group("data_group")
                    data_group.attrs["NX_class"] = "NXdata"
                    data_group.attrs["signal"] = "volume"
                    data_group.attrs["auxiliary_signals"] = ["weight"]

                    volume_data = np.reshape(
                        np.array([i for i in range(24)]), (2, 3, 4)
                    )
                    data_group.create_dataset("volume", data=volume_data)

            with h5py.File(aggregator.data_files[0], "r") as f:
                data_group = f.require_group(aggregator.nxdata_path_name)
                aggregator._get_default_signals_and_axes(data_group)

            self.assertEqual(aggregator.non_weight_aux_signal_names, [])
            self.assertEqual(aggregator.aux_signal_names, ["weight"])
            self.assertEqual(aggregator.renormalisation, True)
            self.assertEqual(aggregator.signal_name, "volume")
            self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])

    def test_generate_axes_names(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.signal_name = "volume"
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, "w") as f:
                entry_group = f.create_group("entry_group")
                data_group = entry_group.create_group("data_group")
                data_group.attrs["NX_class"] = "NXdata"
                data_group.attrs["signal"] = "volume"
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                data_group.create_dataset("volume", data=volume_data)

                aggregator._generate_axes_names(data_group)

            self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])
            self.assertEqual(aggregator.use_default_axes, True)

    @parameterized.expand(
        [
            ("normal", ["weight"], True),
            ("two_aux_signals", ["weight", "other"], True),
            ("no_aux_signals", None, True),
            ("no_aux_signals_or_axes", None, False),
            ("no_axes", ["weight"], False),
        ]
    )
    def test_get_all_axes(self, name, aux_signal_names, use_default_axes) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxdata_path_name = "default_entry/default_data"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["a-axis", "b-axis", "c-axis"]
        aggregator.use_default_axes = use_default_axes
        aggregator.data_dimensions = 3
        aggregator.aux_signal_names = aux_signal_names
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, True)

            with h5py.File(file_path_0, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                if not use_default_axes:
                    nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                    nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                    nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    for name in aux_signal_names:
                        data = np.reshape(
                            np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4)
                        )
                        nxdata_group.create_dataset(name, data=data)

            with h5py.File(file_path_1, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                if not use_default_axes:
                    nxdata_group.create_dataset("a-axis", data=[0.0, 0.2, 0.4])
                    nxdata_group.create_dataset("b-axis", data=[1.2, 1.4])
                    nxdata_group.create_dataset(
                        "c-axis", data=[-0.4, -0.2, 0.0, 0.2, 0.4]
                    )
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    for name in aux_signal_names:
                        data = np.reshape(
                            np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5)
                        )
                        nxdata_group.create_dataset(name, data=data)

            aggregator._get_all_axes()

            self.assertEqual(aggregator.signal_shapes, [(2, 3, 4), (3, 2, 5)])
            all_axes_flat = np.hstack(
                [item for axis in aggregator.all_axes for item in axis]
            )
            if use_default_axes:
                np.testing.assert_allclose(
                    all_axes_flat,
                    np.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1, 2, 3, 4]),
                )
                self.assertEqual(aggregator.axes_spacing, [1, 1, 1])
            else:
                np.testing.assert_allclose(
                    all_axes_flat,
                    np.array(
                        [
                            0.0,
                            0.2,
                            1.0,
                            1.2,
                            1.4,
                            -0.4,
                            -0.2,
                            0.0,
                            0.2,
                            0.0,
                            0.2,
                            0.4,
                            1.2,
                            1.4,
                            -0.4,
                            -0.2,
                            0.0,
                            0.2,
                            0.4,
                        ]
                    ),
                )
                np.testing.assert_allclose(
                    aggregator.axes_spacing, [0.2, 0.2, 0.2], rtol=1e-14
                )

    @parameterized.expand(
        [
            ("renormalised_no_aux", True, ["weight"], []),
            (
                "renormalised_aux",
                True,
                ["weight", "aux_signal_0", "aux_signal_1"],
                ["aux_signal_0", "aux_signal_1"],
            ),
            ("no_weight_no_aux", False, None, []),
            (
                "no_weight_aux",
                False,
                ["aux_signal_0", "aux_signal_1"],
                ["aux_signal_0", "aux_signal_1"],
            ),
        ]
    )
    def test_accumulate_volumes(
        self, name, renormalisation, aux_signal_names, non_weight_aux_signal_names
    ) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxdata_path_name = "default_entry/default_data"
        aggregator.signal_name = "volume"
        aggregator.renormalisation = renormalisation
        aggregator.axes_names = ["a-axis", "b-axis", "c-axis"]
        aggregator.use_default_axes = True
        aggregator.data_dimensions = 3
        aggregator.aux_signal_names = aux_signal_names
        aggregator.non_weight_aux_signal_names = non_weight_aux_signal_names
        aggregator.all_slices = [
            (slice(0, 2, None), slice(0, 3, None), slice(0, 4, None)),
            (slice(0, 3, None), slice(0, 2, None), slice(0, 5, None)),
        ]
        aggregator.accumulator_volume = np.zeros((3, 3, 5))
        if renormalisation:
            aggregator.accumulator_weights = np.zeros((3, 3, 5))
        aggregator.accumulator_aux_signals = [np.zeros((3, 3, 5))] * len(
            non_weight_aux_signal_names
        )
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, True)

            with h5py.File(file_path_0, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if renormalisation:
                    weight_data = np.reshape(
                        np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4)
                    )
                    nxdata_group.create_dataset("weight", data=weight_data)
                if non_weight_aux_signal_names:
                    for count, name in enumerate(non_weight_aux_signal_names):
                        signal = np.reshape(
                            np.array([i * 3 + (count + 5) for i in range(24)]),
                            (2, 3, 4),
                        )
                        nxdata_group.create_dataset(name, data=signal)

            with h5py.File(file_path_1, "r+") as f:
                nxdata_group = f.require_group(aggregator.nxdata_path_name)
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if renormalisation:
                    weight_data = np.reshape(
                        np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5)
                    )
                    nxdata_group.create_dataset("weight", data=weight_data)
                if non_weight_aux_signal_names:
                    for count, name in enumerate(non_weight_aux_signal_names):
                        signal = np.reshape(
                            np.array([i * 4 + (count + 2) for i in range(30)]),
                            (3, 2, 5),
                        )
                        nxdata_group.create_dataset(name, data=signal)

            aggregator._accumulate_volumes()
            if renormalisation:
                self.assertEqual(aggregator.accumulator_weights.shape, (3, 3, 5))
                volume = np.array(
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        4.56,
                        5.55172414,
                        6.54545455,
                        7.54054054,
                        9.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        0.0,
                        11.05882353,
                        12.05454545,
                        13.05084746,
                        14.04761905,
                        14.0,
                        15.50724638,
                        16.50684932,
                        17.50649351,
                        18.50617284,
                        19.0,
                        20.0,
                        21.0,
                        22.0,
                        23.0,
                        0.0,
                        20.0,
                        21.0,
                        22.0,
                        23.0,
                        24.0,
                        25.0,
                        26.0,
                        27.0,
                        28.0,
                        29.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                self.assertEqual(aggregator.accumulator_volume.shape, (3, 3, 5))
                np.testing.assert_allclose(
                    aggregator.accumulator_volume,
                    volume.reshape(3, 3, 5),
                    rtol=6.9e-9,
                )
                weight = np.array(
                    [
                        7.0,
                        11.0,
                        15.0,
                        19.0,
                        12.0,
                        25.0,
                        29.0,
                        33.0,
                        37.0,
                        22.0,
                        19.0,
                        21.0,
                        23.0,
                        25.0,
                        0.0,
                        51.0,
                        55.0,
                        59.0,
                        63.0,
                        32.0,
                        69.0,
                        73.0,
                        77.0,
                        81.0,
                        42.0,
                        43.0,
                        45.0,
                        47.0,
                        49.0,
                        0.0,
                        44.0,
                        46.0,
                        48.0,
                        50.0,
                        52.0,
                        54.0,
                        56.0,
                        58.0,
                        60.0,
                        62.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                np.testing.assert_allclose(
                    aggregator.accumulator_weights,
                    weight.reshape(3, 3, 5),
                    rtol=1e-14,
                )
            else:
                volume = np.array(
                    [
                        0.0,
                        2.0,
                        4.0,
                        6.0,
                        4.0,
                        9.0,
                        11.0,
                        13.0,
                        15.0,
                        9.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        0.0,
                        22.0,
                        24.0,
                        26.0,
                        28.0,
                        14.0,
                        31.0,
                        33.0,
                        35.0,
                        37.0,
                        19.0,
                        20.0,
                        21.0,
                        22.0,
                        23.0,
                        0.0,
                        20.0,
                        21.0,
                        22.0,
                        23.0,
                        24.0,
                        25.0,
                        26.0,
                        27.0,
                        28.0,
                        29.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                self.assertEqual(aggregator.accumulator_volume.shape, (3, 3, 5))
                np.testing.assert_allclose(
                    aggregator.accumulator_volume,
                    volume.reshape(3, 3, 5),
                    rtol=1e-14,
                )
                self.assertFalse(hasattr(aggregator, "accumulator_weights"))

            for aux_signal in aggregator.accumulator_aux_signals:
                self.assertEqual(aux_signal.shape, (3, 3, 5))
                if renormalisation:
                    signal = np.array(
                        [
                            53.0,
                            163.0,
                            329.0,
                            551.0,
                            444.0,
                            1015.0,
                            1381.0,
                            1803.0,
                            2281.0,
                            1694.0,
                            1121.0,
                            1365.0,
                            1633.0,
                            1925.0,
                            0.0,
                            4281.0,
                            4999.0,
                            5773.0,
                            6603.0,
                            3744.0,
                            7995.0,
                            8969.0,
                            9999.0,
                            11085.0,
                            6594.0,
                            5633.0,
                            6165.0,
                            6721.0,
                            7301.0,
                            0.0,
                            7260.0,
                            7958.0,
                            8688.0,
                            9450.0,
                            10244.0,
                            11070.0,
                            11928.0,
                            12818.0,
                            13740.0,
                            14694.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                else:
                    signal = np.array(
                        [
                            16.0,
                            30.0,
                            44.0,
                            58.0,
                            37.0,
                            80.0,
                            94.0,
                            108.0,
                            122.0,
                            77.0,
                            59.0,
                            65.0,
                            71.0,
                            77.0,
                            0.0,
                            168.0,
                            182.0,
                            196.0,
                            210.0,
                            117.0,
                            232.0,
                            246.0,
                            260.0,
                            274.0,
                            157.0,
                            131.0,
                            137.0,
                            143.0,
                            149.0,
                            0.0,
                            165.0,
                            173.0,
                            181.0,
                            189.0,
                            197.0,
                            205.0,
                            213.0,
                            221.0,
                            229.0,
                            237.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                np.testing.assert_allclose(
                    aux_signal, signal.reshape(3, 3, 5), rtol=1e-14
                )

    @pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
    def test_accumulate_volumes_applied_data(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_files = [
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"
            ),
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs"
            ),
        ]
        aggregator.nxentry_name = "processed"
        aggregator.nxdata_name = "reciprocal_space"
        aggregator.nxdata_path_name = "processed/reciprocal_space"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["h-axis", "k-axis", "l-axis"]
        aggregator.renormalisation = True
        aggregator.data_dimensions = 3
        aggregator.non_weight_aux_signal_names = []
        aggregator.accumulator_weights = np.zeros([83, 77, 13])
        aggregator.accumulator_volume = np.zeros([83, 77, 13])
        aggregator.accumulator_aux_signals = []
        aggregator.axes_mins = [-0.2, -0.08, 0.86]
        aggregator.axes_spacing = [0.02, 0.02, 0.02]
        aggregator.accumulator_axis_lengths = [83, 77, 13]
        aggregator.accumulator_axis_ranges = [
            [
                x * aggregator.axes_spacing[i] + aggregator.axes_mins[i]
                for x in range(aggregator.accumulator_axis_lengths[i])
            ]
            for i in range(3)
        ]
        aggregator.all_slices = [
            (slice(0, 83), slice(0, 77), slice(0, 13)),
            (slice(0, 83), slice(0, 77), slice(0, 13)),
        ]
        aggregator._accumulate_volumes()

        with h5py.File(
            "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs", "r"
        ) as f:
            volumes_array = NXdataAggregator._require_dataset(
                f, "processed/reciprocal_space/volume"
            )[...]
            weights_array = NXdataAggregator._require_dataset(
                f, "processed/reciprocal_space/weight"
            )[...]
        self.assertEqual(aggregator.accumulator_volume.shape, (83, 77, 13))
        np.testing.assert_allclose(
            aggregator.accumulator_volume, volumes_array, rtol=1e-12
        )
        np.testing.assert_allclose(
            aggregator.accumulator_weights, weights_array, rtol=2.1e-14
        )

    @pytest.mark.skipif(gh_testing, reason="running GitHub workflow")
    def test_write_aggregation_file(self) -> None:
        sliced_data_files = [
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"
            ),
            Path(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs"
            ),
        ]
        with TemporaryDirectory(prefix="test_dir_") as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.is_dir():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            aggregation_file = cluster_output_dir / "aggregated_results.nxs"

            aggregator = NXdataAggregator()
            aggregation_results = aggregator.aggregate(
                aggregation_file, sliced_data_files
            )
            with h5py.File(
                "/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs",
                "r",
            ) as f:
                volumes_array = NXdataAggregator._require_dataset(
                    f, "processed/reciprocal_space/volume"
                )[...]
                weights_array = NXdataAggregator._require_dataset(
                    f, "processed/reciprocal_space/weight"
                )[...]
            np.testing.assert_allclose(
                aggregator.accumulator_volume, volumes_array, rtol=1e-12
            )
            np.testing.assert_allclose(
                aggregator.accumulator_weights, weights_array, rtol=2.1e-14
            )

            self.assertEqual(aggregation_results, aggregation_file)
            with h5py.File(aggregation_results, "r") as af:
                self.assertTrue("old_processed" in af)
                self.assertTrue("old_processed/process0.0" in af)
                self.assertTrue("old_processed/process1.0" in af)
                self.assertTrue("processed" in af)
                self.assertTrue("processed/process" in af)
                aggregated_volumes = NXdataAggregator._require_dataset(
                    af, "processed/reciprocal_space/volume"
                )[...]
                aggregated_weights = NXdataAggregator._require_dataset(
                    af, "processed/reciprocal_space/weight"
                )[...]
            np.testing.assert_allclose(volumes_array, aggregated_volumes, rtol=1e-12)
            np.testing.assert_allclose(weights_array, aggregated_weights, rtol=2.1e-14)


if __name__ == "__main__":
    unittest.main()
