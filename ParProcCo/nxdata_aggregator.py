from __future__ import annotations

import logging
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import h5py  # type: ignore
import numpy as np

from ParProcCo.aggregator_interface import AggregatorInterface
from ParProcCo.utils import decode_to_string

from . import __version__


class NXdataAggregator(AggregatorInterface):
    def __init__(self) -> None:
        self.accumulator_aux_signals: list[np.ndarray]
        self.accumulator_axis_lengths: list
        self.accumulator_axis_ranges: list
        self.accumulator_volume: np.ndarray
        self.accumulator_weights: np.ndarray
        self.all_axes: list[list]
        self.all_slices: list[tuple[slice, ...]]
        self.aux_signal_names: list[str] | None
        self.axes_maxs: list
        self.axes_mins: list
        self.axes_names: list[str]
        self.axes_spacing: list
        self.data_dimensions: int
        self.non_weight_aux_signal_names: list[str]
        self.nxdata_name: str
        self.nxdata_path_name: str
        self.nxentry_name: str
        self.data_files: list[Path]
        self.renormalisation: bool
        self.signal_name: str
        self.signal_shapes: list[tuple]
        self.use_default_axes: bool = False
        self.is_binoculars: bool = False

    def aggregate(self, aggregation_output: Path, data_files: list[Path]) -> Path:
        """Overrides AggregatorInterface.aggregate"""
        self._renormalise(data_files)
        aggregated_data_file = self._write_aggregation_file(aggregation_output)
        return aggregated_data_file

    def _renormalise(self, data_files: list[Path]) -> None:
        start = datetime.now()
        self.data_files = data_files
        self._get_nxdata()
        self._initialise_arrays()
        self._accumulate_volumes()
        aggregation_time = datetime.now() - start
        logging.info(
            f"Aggregation completed in {aggregation_time.total_seconds():.3f}s. Sliced file paths: {data_files}."
        )

    @staticmethod
    def _require_dataset(
        group: h5py.File | h5py.Group | h5py.AttributeManager, name: str
    ) -> h5py.Dataset | np.ndarray:
        if name in group:
            data = group[name]
            if isinstance(data, (h5py.Dataset, np.ndarray)):
                return data
            raise ValueError(f"{name} in {group} must be a dataset (is a {type(data)})")
        raise ValueError(f"{name} is not found in {group}")

    def _initialise_arrays(self) -> None:
        self._get_all_axes()

        self.axes_mins = [np.inf] * self.data_dimensions
        self.axes_maxs = [-np.inf] * self.data_dimensions

        for axis_set in self.all_axes:
            for j, axis in enumerate(axis_set):
                self.axes_mins[j] = min([min(axis), self.axes_mins[j]])
                self.axes_maxs[j] = max([max(axis), self.axes_maxs[j]])
        logging.debug(
            f"Calculated axes_mins: {self.axes_mins} and axes_maxs: {self.axes_maxs}"
        )

        self.all_slices = []
        for axes, signal_shape in zip(self.all_axes, self.signal_shapes):
            axes_lengths = tuple(len(axis) for axis in axes)
            assert (
                axes_lengths == signal_shape
            ), "axes_lengths must equal volumes_array.shape"
            a_slices = []
            for j, axis in enumerate(axes):
                start = int(round((axis[0] - self.axes_mins[j]) / self.axes_spacing[j]))
                stop = axes_lengths[j] + start
                a_slices.append(slice(start, stop))

            self.all_slices.append(tuple(a_slices))

        self.accumulator_axis_lengths = []
        self.accumulator_axis_ranges = []

        for i in range(self.data_dimensions):
            length = (
                int(
                    round(
                        (self.axes_maxs[i] - self.axes_mins[i]) / self.axes_spacing[i]
                    )
                )
                + 1
            )
            self.accumulator_axis_lengths.append(length)
            ranges = [
                x * self.axes_spacing[i] + self.axes_mins[i]
                for x in np.arange(self.accumulator_axis_lengths[i])
            ]
            self.accumulator_axis_ranges.append(ranges)
        logging.debug(
            f"Calculated accumulator_axis_lengths: {self.accumulator_axis_lengths} and accumulator_axis_ranges:"
            f" {self.accumulator_axis_ranges}"
        )

        for axes, slices in zip(self.all_axes, self.all_slices):
            for axis, axis_range, single_slice in zip(
                axes, self.accumulator_axis_ranges, slices
            ):
                if not np.allclose(axis, axis_range[single_slice]):
                    raise RuntimeError(
                        f"axis does not match slice {single_slice} of accumulator_axis_range"
                    )

        self.accumulator_volume = np.zeros(self.accumulator_axis_lengths)
        logging.debug(
            f"Accumulator volume array initialised with shape: {self.accumulator_volume.shape}"
        )
        self.accumulator_aux_signals = [np.zeros(self.accumulator_axis_lengths)] * len(
            self.non_weight_aux_signal_names
        )
        if self.renormalisation:
            self.accumulator_weights = np.zeros(self.accumulator_axis_lengths)
            logging.debug(
                f"Accumulator weight array initialised with shape: {self.accumulator_weights.shape}"
            )

    def _get_nxdata(self):
        """sets self.nxentry_name, self.nxdata_name and self.axes_names"""
        data_file = self.data_files[0]
        self.is_binoculars = False
        with h5py.File(data_file, "r") as root:
            if not self.is_binoculars:
                self.is_binoculars = "binoculars" in root
            self.nxentry_name = self._get_default_nxgroup(root, "NXentry")
            nxentry = root.require_group(self.nxentry_name)
            self.nxdata_name = self._get_default_nxgroup(nxentry, "NXdata")
            self.nxdata_path_name = "/".join([self.nxentry_name, self.nxdata_name])
            nxdata = root.require_group(self.nxdata_path_name)
            self._get_default_signals_and_axes(nxdata)

            signal = nxdata[self.signal_name]
            assert isinstance(signal, h5py.Dataset)
            signal_shape = signal.shape
            self.data_dimensions = len(signal_shape)

            if self.renormalisation:
                weights = nxdata["weight"]
                assert isinstance(weights, h5py.Dataset)
                assert (
                    len(weights.shape) == self.data_dimensions
                ), "signal and weight dimensions must match"
                assert (
                    weights.shape == signal_shape
                ), "signal and weight shapes must match"

    def _get_default_nxgroup(self, f: h5py.File | h5py.Group, class_name: str) -> str:
        if "default" in f.attrs:
            group_name = f.attrs["default"]
            assert isinstance(group_name, (str, bytes))  # XXX
            group_name = decode_to_string(group_name)
            class_type = f[group_name].attrs.get("NX_class", "")
            class_type = decode_to_string(class_type)
            assert (
                class_type == class_name
            ), f"{group_name} class_name must be {class_name}"
            return group_name

        group_name = self._get_group_name(f, class_name)
        try:
            return next(group_name)
        except StopIteration:
            raise ValueError(f"no {class_name} group found")

    def _get_group_name(
        self, group: h5py.File | h5py.Group, class_name: str
    ) -> Iterator[str]:
        for group_name in group.keys():
            try:
                class_type = group[group_name].attrs.get("NX_class", "")
                class_type = decode_to_string(class_type)
                if class_type == class_name:
                    group_name = decode_to_string(group_name)
                    yield group_name
            except KeyError:
                logging.warning(
                    f"KeyError: {group_name} could not be accessed in {group}"
                )

    def _get_default_signals_and_axes(self, nxdata: h5py.Group) -> None:
        self.renormalisation = False
        self.non_weight_aux_signal_names = []

        if "auxiliary_signals" in nxdata.attrs:
            self.aux_signal_names = [
                decode_to_string(name)
                for name in NXdataAggregator._require_dataset(
                    nxdata.attrs, "auxiliary_signals"
                )
            ]
            self.non_weight_aux_signal_names = [
                name for name in self.aux_signal_names if name != "weight"
            ]
            logging.info(f"Auxiliary signals found: {self.aux_signal_names}")
            if "weight" in self.aux_signal_names:
                self.renormalisation = True
        else:
            self.aux_signal_names = None

        if "signal" in nxdata.attrs:
            signal_name = nxdata.attrs["signal"]
            assert isinstance(signal_name, (str, bytes))  # XXX
            self.signal_name = decode_to_string(signal_name)
        elif "data" in nxdata.keys():
            self.signal_name = "data"

        if hasattr(self, "signal_name"):
            if "axes" in nxdata.attrs:
                self.axes_names = [
                    decode_to_string(name)
                    for name in NXdataAggregator._require_dataset(nxdata.attrs, "axes")
                ]
            else:
                self._generate_axes_names(nxdata)
        else:
            raise KeyError

    def _generate_axes_names(self, nxdata: h5py.Group) -> None:
        self.use_default_axes = True
        signal_shape = NXdataAggregator._require_dataset(nxdata, self.signal_name).shape
        self.axes_names = [
            f"{letter}-axis" for letter in string.ascii_lowercase[: len(signal_shape)]
        ]

    def _get_all_axes(self) -> None:
        self.signal_shapes = []
        self.all_axes = []
        for data_file in self.data_files:
            with h5py.File(data_file, "r") as f:
                nxdata = f.require_group(self.nxdata_path_name)
                signal = nxdata[self.signal_name]
                assert isinstance(signal, h5py.Dataset)
                signal_shape = signal.shape
                logging.info(
                    f"Signal '{'/'.join([self.nxdata_path_name, self.signal_name])}' read from {data_file}."
                    f" Shape: {signal_shape}"
                )
                assert len(signal_shape) == self.data_dimensions
                self.signal_shapes.append(signal_shape)
                if self.aux_signal_names:
                    for aux_signal_name in self.aux_signal_names:
                        aux_signal_shape = NXdataAggregator._require_dataset(
                            nxdata, aux_signal_name
                        ).shape
                        logging.debug(
                            f"Auxiliary signal '{'/'.join([self.nxdata_path_name, aux_signal_name])}' read from"
                            f" {data_file}. Shape: {aux_signal_shape}"
                        )
                        assert (
                            signal_shape == aux_signal_shape
                        ), f"{aux_signal_name} shape must equal signal_shape"
                if self.use_default_axes:
                    axes = [np.arange(length) for length in signal_shape]
                else:
                    axes = [
                        NXdataAggregator._require_dataset(nxdata, axis_name)[...]
                        for axis_name in self.axes_names
                    ]
                self.all_axes.append(axes)
        self.axes_spacing = [
            np.mean([np.mean(np.diff(axis)) for axis in axis_set])
            for axis_set in zip(*self.all_axes)
        ]
        logging.debug(f"Calculated axes spacings: {self.axes_spacing}")

    def _accumulate_volumes(self) -> None:
        logging.info(
            f"Accumulating volume with shape {self.accumulator_volume.shape} and axes {self.axes_names}"
        )
        for data_file, slices in zip(self.data_files, self.all_slices):
            weights = None
            with h5py.File(data_file, "r") as f:
                aux_signals = []
                nxdata = f.require_group(self.nxdata_path_name)
                volume = NXdataAggregator._require_dataset(nxdata, self.signal_name)[
                    ...
                ]
                logging.debug(
                    f"Reading volume from {'/'.join([self.nxdata_path_name, self.signal_name])} in {data_file}."
                    f" Shape is {volume.shape}"
                )
                if self.renormalisation:
                    weights = NXdataAggregator._require_dataset(nxdata, "weight")[...]
                for name in self.non_weight_aux_signal_names:
                    aux_signals.append(
                        NXdataAggregator._require_dataset(nxdata, name)[...]
                    )
                    logging.debug(
                        f"Reading auxiliary signal from {'/'.join([self.nxdata_path_name, name])} in {data_file}"
                    )

            if self.renormalisation and weights is not None:
                volume = np.multiply(volume, weights)
                self.accumulator_weights[slices] += weights
                aux_signals = [
                    np.multiply(aux_signal, weights) for aux_signal in aux_signals
                ]

            self.accumulator_volume[slices] += volume

            for signal, accumulator_signal in zip(
                aux_signals, self.accumulator_aux_signals
            ):
                accumulator_signal[slices] += signal

        if self.renormalisation:
            logging.info(
                f"Renormalising accumulator_volume with weights {'/'.join([self.nxdata_path_name, 'weight'])}"
            )
            self.accumulator_volume = self.accumulator_volume / self.accumulator_weights
            self.accumulator_volume[np.isnan(self.accumulator_volume)] = 0
            for aux_signal in self.accumulator_aux_signals:
                logging.info(
                    f"Renormalising aux_signal with weights in {'/'.join([self.nxdata_path_name, 'weight'])}"
                )
                aux_signal = aux_signal / self.accumulator_weights
                aux_signal[np.isnan(aux_signal)] = 0

    def _write_aggregation_file(self, aggregation_output: Path) -> Path:
        start = datetime.now()
        logging.info(f"Writing aggregated data to file: {aggregation_output}")
        with h5py.File(aggregation_output, "w") as f:
            processed = f.create_group(self.nxentry_name)
            processed.attrs["NX_class"] = "NXentry"
            processed.attrs["default"] = self.nxdata_name

            process = processed.create_group("process")
            process.attrs["NX_class"] = "NXprocess"
            process.create_dataset("date", data=str(datetime.now(timezone.utc)))
            process.create_dataset(
                "parameters",
                data=f"inputs: {self.data_files}, output: {aggregation_output}",
            )
            process.create_dataset("program", data="ParProcCo")
            process.create_dataset("version", data=__version__)

            data_group = processed.create_group(self.nxdata_name)
            data_group.attrs["NX_class"] = "NXdata"
            if self.aux_signal_names:
                data_group.attrs["auxiliary_signals"] = self.aux_signal_names
            data_group.attrs["axes"] = self.axes_names
            data_group.attrs["signal"] = self.signal_name
            for i, axis in enumerate(self.axes_names):
                data_group.attrs[f"{axis}_indices"] = i
                data_group.create_dataset(
                    f"{axis}", data=self.accumulator_axis_ranges[i]
                )
            data_group.create_dataset(self.signal_name, data=self.accumulator_volume)
            if self.renormalisation:
                data_group.create_dataset("weight", data=self.accumulator_weights)
            for name, dataset in zip(
                self.non_weight_aux_signal_names, self.accumulator_aux_signals
            ):
                data_group.create_dataset(name, data=dataset)

            f.attrs["default"] = self.nxentry_name

            old_processed = None
            for i, filepath in enumerate(self.data_files):
                with h5py.File(filepath, "r") as df:
                    data_nxentry_group = df.require_group(self.nxentry_name)
                    group_name = self._get_group_name(data_nxentry_group, "NXprocess")
                    for j, name in enumerate(group_name):
                        if "old_processed" not in f:
                            old_processed = f.create_group("old_processed")
                            old_processed.attrs["NX_class"] = "NXentry"
                            logging.info(
                                f"Created 'old_processed' group in {aggregation_output}"
                            )
                        data_nxentry_group.copy(
                            name, old_processed, name=f"process{i}.{j}"
                        )
                        logging.info(
                            f"Copied '{'/'.join([data_nxentry_group.name, name])}' group in {filepath} to"  # type: ignore
                            f" '{'/'.join(['old_processed', f'process{i}.{j}'])}' group in {aggregation_output}"
                        )

            if self.is_binoculars:
                logging.info("Writing BINoculars group")
                binoculars = f.create_group("binoculars")
                binoculars.attrs["type"] = "space"
                f.create_group("binoculars/axes")
                binocular_axes = [
                    axis.split("-axis")[0].capitalize() for axis in self.axes_names
                ]
                for i, axis in enumerate(binocular_axes):
                    axis_min = self.axes_mins[i]
                    axis_max = self.axes_maxs[i]
                    scaling = (self.accumulator_axis_lengths[i] - 1) / (
                        axis_max - axis_min
                    )
                    axis_dataset = [
                        i,
                        axis_min,
                        axis_max,
                        self.axes_spacing[i],
                        axis_min * scaling,
                        axis_max * scaling,
                    ]
                    f.create_dataset(f"binoculars/axes/{axis}", data=axis_dataset)
                binoculars["counts"] = data_group[self.signal_name]
                if self.renormalisation:
                    binoculars["contributions"] = data_group["weight"]

        elapsed_time = datetime.now() - start
        logging.info(
            f"Aggregated data written in {elapsed_time.total_seconds():.3f}s. Aggregation file: {aggregation_output}"
        )
        return aggregation_output
