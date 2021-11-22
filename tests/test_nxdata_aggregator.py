from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from parameterized import parameterized

from ParProcCo.nxdata_aggregator import NXdataAggregator
from ParProcCo.utils import decode_to_string


class TestNXdataAggregator(unittest.TestCase):

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def create_basic_nexus_file(self, file_path: Path, has_weight: bool) -> None:
        with h5py.File(file_path, 'w') as f:
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
        name = b'name'
        name = decode_to_string(name)
        self.assertEqual(name, "name")

    def test_renormalise(self) -> None:
        output_file_paths = [Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"),
                             Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs")]
        aggregator = NXdataAggregator()
        aggregator._renormalise(output_file_paths)
        with h5py.File("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed/reciprocal_space/volume"])
            weights_array = np.array(f["processed/reciprocal_space/weight"])
        np.testing.assert_allclose(aggregator.accumulator_volume, volumes_array, rtol=1e-12)
        np.testing.assert_allclose(aggregator.accumulator_weights, weights_array, rtol=2.1e-14)

    def test_initialise_arrays_applied_data(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_dimensions = 3
        aggregator.data_files = [Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"),
                                        Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs")]
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
        self.assertAlmostEqual(aggregator.accumulator_axis_ranges[0][-1], 1.44, places=14)
        self.assertEqual(aggregator.accumulator_axis_ranges[1][0], -0.08)
        self.assertEqual(aggregator.accumulator_axis_ranges[1][-1], 1.44)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][0], 0.86)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][-1], 1.1)
        self.assertTrue(np.array_equal(aggregator.accumulator_volume, np.zeros([83, 77, 13])))
        self.assertTrue(np.array_equal(aggregator.accumulator_weights, np.zeros([83, 77, 13])))
        self.assertEqual(aggregator.all_slices, [(slice(0, 83), slice(0, 77), slice(0, 13)),
                                                 (slice(0, 83), slice(0, 77), slice(0, 13))])

    @parameterized.expand([
        ("normal", (2, 3, 4), True, ["weight"], [], None, None),
        ("no_aux", (2, 3, 4), False, None, [], None, None),
        ("axes_wrong", (2, 4, 3), True, ["weight"], [], AssertionError, "axes_lengths must equal volumes_array.shape"),
        ("non_weight_signals", (2, 3, 4), False, ["other_0", "other_1"], ["other_0", "other_1"], None, None),
        ("non_weight_signals_plus_weight", (2, 3, 4), True, ["weight", "other_0", "other_1"], ["other_0", "other_1"], None, None)
    ])
    def test_initialise_arrays(self, name, shape, has_weight, aux_signal_names, non_weight_names, error_name, error_msg) -> None:
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

        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, has_weight)

            with h5py.File(file_path_0, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]
                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), shape)
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    aux_data = np.reshape(np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4))
                    for name in aux_signal_names:
                        nxdata_group.create_dataset(name, data=aux_data)

            with h5py.File(file_path_1, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]

                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2, 0.4])
                nxdata_group.create_dataset("b-axis", data=[1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2, 0.4])
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    aux_data = np.reshape(np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5))
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
            self.assertEqual(aggregator.all_slices, [(slice(0, 2), slice(0, 3), slice(0, 4)),
                                                     (slice(0, 3), slice(1, 3), slice(0, 5))])
            self.assertEqual(aggregator.accumulator_axis_lengths, [3, 3, 5])
            for l, e in zip(aggregator.accumulator_axis_ranges,
                            [[0.0, 0.2, 0.4], [1.0, 1.2, 1.4], [-0.4, -0.2, 0.0, 0.2, 0.4]]):
                np.testing.assert_allclose(np.array(l), e, rtol=1e-14)
            self.assertTrue(np.array_equal(aggregator.accumulator_volume, np.zeros([3, 3, 5])))
            self.assertEqual(aggregator.non_weight_aux_signal_names, non_weight_names)
            self.assertEqual(aggregator.aux_signal_names, aux_signal_names)
            if has_weight:
                self.assertTrue(np.array_equal(aggregator.accumulator_weights, np.zeros([3, 3, 5])))
            else:
                self.assertFalse(hasattr(aggregator, "accumulator_weights"))

            if non_weight_names:
                self.assertEqual(len(aggregator.accumulator_aux_signals), 2)
                for signal in aggregator.accumulator_aux_signals:
                    np.testing.assert_array_equal(signal, np.zeros((3, 3, 5)))
            else:
                self.assertEqual(aggregator.accumulator_aux_signals, [])

    def test_get_nxdata_applied_data(self) -> None:
        output_data_files = [Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"),
                             Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs")]
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

    @parameterized.expand([
        ("normal", (2, 3, 4), True, None, None),
        ("no_aux", (2, 3, 4), False, None, None),
        ("shape_wrong", (2, 4, 3), True, AssertionError, "signal and weight shapes must match"),
        ("dims_wrong", (2, 12), True, AssertionError, "signal and weight dimensions must match")
    ])
    def test_get_nx_data_param(self, name, shape, has_weight, error_name, error_msg) -> None:
        aggregator = NXdataAggregator()
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            aggregator.data_files = [file_path]

            self.create_basic_nexus_file(file_path, has_weight)
            with h5py.File(file_path, 'r+') as f:
                nxdata_group = f["default_entry/default_data"]
                nxdata_group.attrs[f"a-axis_indices"] = 0
                nxdata_group.attrs[f"b-axis_indices"] = 1
                nxdata_group.attrs[f"c-axis_indices"] = 2
                nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if has_weight:
                    weight_data = np.reshape(np.array([i * 2 + 3 for i in range(24)]), shape)
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

    @parameterized.expand([
        ("normal", "NXentry", "default_entry", True, None, None),
        ("wrong_class", "NXprocess", "default_entry", True, AssertionError, "default_entry class_name must be NXentry"),
        ("bytes", "NXentry", b'default_entry', True, None, None),
        ("no_default", "NXentry", "default_entry", False, None, None),
        ("no_default_no_class", None, "default_entry", False, ValueError, "no NXentry group found"),
        ("no_default_wrong_class", "NXprocess", "default_entry", False, ValueError, "no NXentry group found"),
        ("no_groups", None, None, False, ValueError, "no NXentry group found")
    ])
    def test_get_default_nxgroup(self, name, default_class, default_name, has_default, error_name, error_msg) -> None:
        aggregator = NXdataAggregator()
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, 'w') as f:
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
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            linked_file_path = Path(working_directory) / "linked_file.nxs"
            with h5py.File(file_path, 'w') as f:
                f["entry0"] = h5py.ExternalLink(str(linked_file_path), "missing_group")
                entry_group = f.create_group("entry1")
                entry_group.attrs["NX_class"] = "NXentry"

                with self.assertLogs(level='WARNING') as cm:
                    aggregator._get_default_nxgroup(f, "NXentry")
                    self.assertEqual(cm.output, ['WARNING:root:KeyError: entry0 could not be accessed in <HDF5 file "output.nxs" (mode r+)>'])

    @parameterized.expand([
        ("normal", "volume", ["weight"], True, [], None),
        ("no_aux_signals", "volume", None, False, [], None),
        ("no_weight_signal", "volume", ["other"], False, ["other"], None),
        ("extra_aux_signals", "volume", ["weight", "other"], True, ["other"], None),
        ("data_as_signal", "data", ["weight"], True, [], None),
        ("no_axes", "volume", ["weight"], True, [], None),
        ("no_signal_or_data", None, ["weight"], True, [], KeyError),
        ("other_signal", "other", ["weight"], True, [], KeyError)

    ])
    def test_get_default_signals_and_axes(
            self, name, signal, aux_signals, renormalisation, non_weight_signals, error_name) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxentry_name = "entry_group"
        aggregator.nxdata_name = "data_group"
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, 'w') as f:
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

        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            aggregator.data_files = [Path(working_directory) / f"output{i}.nxs" for i in range(3)]
            for file_path in aggregator.data_files:
                with h5py.File(file_path, 'w') as f:
                    entry_group = f.create_group("entry_group")
                    data_group = entry_group.create_group("data_group")
                    data_group.attrs["NX_class"] = "NXdata"
                    data_group.attrs["signal"] = "volume"
                    data_group.attrs["auxiliary_signals"] = ["weight"]

                    volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                    data_group.create_dataset("volume", data=volume_data)

            with h5py.File(aggregator.data_files[0], 'r') as f:
                data_group = f[aggregator.nxdata_path_name]
                aggregator._get_default_signals_and_axes(data_group)

            self.assertEqual(aggregator.non_weight_aux_signal_names, [])
            self.assertEqual(aggregator.aux_signal_names, ["weight"])
            self.assertEqual(aggregator.renormalisation, True)
            self.assertEqual(aggregator.signal_name, "volume")
            self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])

    def test_generate_axes_names(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.signal_name = "volume"
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path = Path(working_directory) / "output.nxs"
            with h5py.File(file_path, 'w') as f:
                entry_group = f.create_group("entry_group")
                data_group = entry_group.create_group("data_group")
                data_group.attrs["NX_class"] = "NXdata"
                data_group.attrs["signal"] = "volume"
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                data_group.create_dataset("volume", data=volume_data)

                aggregator._generate_axes_names(data_group)

            self.assertEqual(aggregator.axes_names, ["a-axis", "b-axis", "c-axis"])
            self.assertEqual(aggregator.use_default_axes, True)

    @parameterized.expand([
        ("normal", ["weight"], True),
        ("two_aux_signals", ["weight", "other"], True),
        ("no_aux_signals", None, True),
        ("no_aux_signals_or_axes", None, False),
        ("no_axes", ["weight"], False)
    ])
    def test_get_all_axes(self, name, aux_signal_names, use_default_axes) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxdata_path_name = "default_entry/default_data"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["a-axis", "b-axis", "c-axis"]
        aggregator.use_default_axes = use_default_axes
        aggregator.data_dimensions = 3
        aggregator.aux_signal_names = aux_signal_names
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, True)

            with h5py.File(file_path_0, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]
                if not use_default_axes:
                    nxdata_group.create_dataset("a-axis", data=[0.0, 0.2])
                    nxdata_group.create_dataset("b-axis", data=[1.0, 1.2, 1.4])
                    nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2])
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    for name in aux_signal_names:
                        data = np.reshape(np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4))
                        nxdata_group.create_dataset(name, data=data)

            with h5py.File(file_path_1, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]
                if not use_default_axes:
                    nxdata_group.create_dataset("a-axis", data=[0.0, 0.2, 0.4])
                    nxdata_group.create_dataset("b-axis", data=[1.2, 1.4])
                    nxdata_group.create_dataset("c-axis", data=[-0.4, -0.2, 0.0, 0.2, 0.4])
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if aux_signal_names:
                    for name in aux_signal_names:
                        data = np.reshape(np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5))
                        nxdata_group.create_dataset(name, data=data)

            aggregator._get_all_axes()

            self.assertEqual(aggregator.signal_shapes, [(2, 3, 4), (3, 2, 5)])
            if use_default_axes:
                self.assertEqual(aggregator.all_axes,
                                 [[[0, 1], [0, 1, 2], [0, 1, 2, 3]], [[0, 1, 2], [0, 1], [0, 1, 2, 3, 4]]])
                self.assertEqual(aggregator.axes_spacing, [1, 1, 1])
            else:
                self.assertEqual(aggregator.all_axes,
                                 [[[0.0, 0.2], [1.0, 1.2, 1.4], [-0.4, -0.2, 0.0, 0.2]],
                                  [[0.0, 0.2, 0.4], [1.2, 1.4], [-0.4, -0.2, 0.0, 0.2, 0.4]]])
                np.testing.assert_allclose(aggregator.axes_spacing, [0.2, 0.2, 0.2], rtol=1e-14)

    @parameterized.expand([
        ("renormalised_no_aux", True, ["weight"], []),
        ("renormalised_aux", True, ["weight", "aux_signal_0", "aux_signal_1"], ["aux_signal_0", "aux_signal_1"]),
        ("no_weight_no_aux", False, None, []),
        ("no_weight_aux", False, ["aux_signal_0", "aux_signal_1"], ["aux_signal_0", "aux_signal_1"])
    ])
    def test_accumulate_volumes(self, name, renormalisation, aux_signal_names, non_weight_aux_signal_names) -> None:
        aggregator = NXdataAggregator()
        aggregator.nxdata_path_name = "default_entry/default_data"
        aggregator.signal_name = "volume"
        aggregator.renormalisation = renormalisation
        aggregator.axes_names = ["a-axis", "b-axis", "c-axis"]
        aggregator.use_default_axes = True
        aggregator.data_dimensions = 3
        aggregator.aux_signal_names = aux_signal_names
        aggregator.non_weight_aux_signal_names = non_weight_aux_signal_names
        aggregator.all_slices = [(slice(0, 2, None), slice(0, 3, None), slice(0, 4, None)),
                                 (slice(0, 3, None), slice(0, 2, None), slice(0, 5, None))]
        aggregator.accumulator_volume = np.zeros((3, 3, 5))
        if renormalisation:
            aggregator.accumulator_weights = np.zeros((3, 3, 5))
        aggregator.accumulator_aux_signals = [np.zeros((3, 3, 5))] * len(non_weight_aux_signal_names)
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            file_path_0 = Path(working_directory) / "output0.nxs"
            file_path_1 = Path(working_directory) / "output1.nxs"

            aggregator.data_files = [file_path_0, file_path_1]

            for file_path in aggregator.data_files:
                self.create_basic_nexus_file(file_path, True)

            with h5py.File(file_path_0, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]
                volume_data = np.reshape(np.array([i for i in range(24)]), (2, 3, 4))
                nxdata_group.create_dataset("volume", data=volume_data)
                if renormalisation:
                    weight_data = np.reshape(np.array([i * 2 + 3 for i in range(24)]), (2, 3, 4))
                    nxdata_group.create_dataset("weight", data=weight_data)
                if non_weight_aux_signal_names:
                    for count, name in enumerate(non_weight_aux_signal_names):
                        signal = np.reshape(np.array([i * 3 + (count + 5) for i in range(24)]), (2, 3, 4))
                        nxdata_group.create_dataset(name, data=signal)

            with h5py.File(file_path_1, 'r+') as f:
                nxdata_group = f[aggregator.nxdata_path_name]
                volume_data = np.reshape(np.array([i for i in range(30)]), (3, 2, 5))
                nxdata_group.create_dataset("volume", data=volume_data)
                if renormalisation:
                    weight_data = np.reshape(np.array([i * 2 + 4 for i in range(30)]), (3, 2, 5))
                    nxdata_group.create_dataset("weight", data=weight_data)
                if non_weight_aux_signal_names:
                    for count, name in enumerate(non_weight_aux_signal_names):
                        signal = np.reshape(np.array([i * 4 + (count + 2) for i in range(30)]), (3, 2, 5))
                        nxdata_group.create_dataset(name, data=signal)

            aggregator._accumulate_volumes()
            if renormalisation:
                self.assertEqual(aggregator.accumulator_weights.shape, (3, 3, 5))
                volume = [
                    0., 1., 2., 3., 4., 4.56, 5.55172414, 6.54545455, 7.54054054, 9., 8., 9., 10., 11., 0., 11.05882353,
                    12.05454545, 13.05084746, 14.04761905, 14., 15.50724638, 16.50684932, 17.50649351, 18.50617284, 19.,
                    20., 21., 22., 23., 0., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 0., 0., 0., 0., 0.
                ]
                self.assertEqual(aggregator.accumulator_volume.shape, (3, 3, 5))
                np.testing.assert_allclose(aggregator.accumulator_volume,
                                           np.array(volume).reshape(3, 3, 5), rtol=6.9e-9)
                weight = [
                    7., 11., 15., 19., 12., 25., 29., 33., 37., 22., 19., 21., 23., 25., 0., 51., 55., 59., 63., 32.,
                    69., 73., 77., 81., 42., 43., 45., 47., 49., 0., 44., 46., 48., 50., 52., 54., 56., 58., 60., 62.,
                    0., 0., 0., 0., 0.
                ]
                np.testing.assert_allclose(aggregator.accumulator_weights,
                                           np.array(weight).reshape(3, 3, 5), rtol=1e-14)
            else:
                volume = [
                    0., 2., 4., 6., 4., 9., 11., 13., 15., 9., 8., 9., 10., 11., 0., 22., 24., 26., 28., 14., 31., 33.,
                    35., 37., 19., 20., 21., 22., 23., 0., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 0., 0., 0.,
                    0., 0.
                ]
                self.assertEqual(aggregator.accumulator_volume.shape, (3, 3, 5))
                np.testing.assert_allclose(aggregator.accumulator_volume, np.array(volume).reshape(3, 3, 5), rtol=1e-14)
                self.assertFalse(hasattr(aggregator, "accumulator_weights"))

            for aux_signal in aggregator.accumulator_aux_signals:
                self.assertEqual(aux_signal.shape, (3, 3, 5))
                if renormalisation:
                    signal = [
                        53., 163., 329., 551., 444., 1015., 1381., 1803., 2281., 1694., 1121., 1365., 1633., 1925., 0.,
                        4281., 4999., 5773., 6603., 3744., 7995., 8969., 9999., 11085., 6594., 5633., 6165., 6721.,
                        7301., 0., 7260., 7958., 8688., 9450., 10244., 11070., 11928., 12818., 13740., 14694., 0., 0.,
                        0., 0., 0.
                    ]
                else:
                    signal = [
                        16., 30., 44., 58., 37., 80., 94., 108., 122., 77., 59., 65., 71., 77., 0., 168., 182., 196.,
                        210., 117., 232., 246., 260., 274., 157., 131., 137., 143., 149., 0., 165., 173., 181., 189.,
                        197., 205., 213., 221., 229., 237., 0., 0., 0., 0., 0.
                    ]
                np.testing.assert_allclose(aux_signal, np.array(signal).reshape(3, 3, 5), rtol=1e-14)

    def test_accumulate_volumes_applied_data(self) -> None:
        aggregator = NXdataAggregator()
        aggregator.data_files = [Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"),
                                        Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs")]
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
        aggregator.accumulator_axis_ranges = [[x * aggregator.axes_spacing[i] + aggregator.axes_mins[i]
                                               for x in range(aggregator.accumulator_axis_lengths[i])]
                                              for i in range(3)]
        aggregator.all_slices = [(slice(0, 83), slice(0, 77), slice(0, 13)), (slice(0, 83), slice(0, 77), slice(0, 13))]
        aggregator._accumulate_volumes()

        with h5py.File("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed/reciprocal_space/volume"])
            weights_array = np.array(f["processed/reciprocal_space/weight"])
        self.assertEqual(aggregator.accumulator_volume.shape, (83, 77, 13))
        np.testing.assert_allclose(aggregator.accumulator_volume, volumes_array, rtol=1e-12)
        np.testing.assert_allclose(aggregator.accumulator_weights, weights_array, rtol=2.1e-14)

    def test_write_aggregation_file(self) -> None:
        sliced_data_files = [Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfa.nxs"),
                             Path("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-halfb.nxs")]
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.is_dir():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            aggregation_file = cluster_output_dir / "aggregated_results.nxs"

            aggregator = NXdataAggregator()
            aggregation_results = aggregator.aggregate(aggregation_file, sliced_data_files)
            with h5py.File("/dls/science/groups/das/ExampleData/i07/i07-394487-applied-whole.nxs", "r") as f:
                volumes_array = np.array(f["processed/reciprocal_space/volume"])
                weights_array = np.array(f["processed/reciprocal_space/weight"])
            np.testing.assert_allclose(aggregator.accumulator_volume, volumes_array, rtol=1e-12)
            np.testing.assert_allclose(aggregator.accumulator_weights, weights_array, rtol=2.1e-14)

            self.assertEqual(aggregation_results, aggregation_file)
            with h5py.File(aggregation_results, "r") as af:
                aggregated_volumes = np.array(af["processed/reciprocal_space/volume"])
                aggregated_weights = np.array(af["processed/reciprocal_space/weight"])
            np.testing.assert_allclose(volumes_array, aggregated_volumes, rtol=1e-12)
            np.testing.assert_allclose(weights_array, aggregated_weights, rtol=2.1e-14)


if __name__ == '__main__':
    unittest.main()
