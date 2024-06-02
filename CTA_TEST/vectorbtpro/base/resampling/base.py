# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base classes and functions for resampling."""

import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.datetime_ import freq_to_timedelta64, try_to_datetime_index, infer_index_freq
from vectorbtpro.utils.decorators import cached_property, class_or_instancemethod
from vectorbtpro.base.resampling import nb
from vectorbtpro.base.indexes import repeat_index
from vectorbtpro.registries.jit_registry import jit_reg

__all__ = [
    "Resampler",
]


ResamplerT = tp.TypeVar("ResamplerT", bound="Resampler")


class Resampler(Configured):
    """Class that exposes methods to resample index.

    Args:
        source_index (index_like): Index being resampled.
        target_index (index_like): Index resulted from resampling.
        source_freq (frequency_like or bool): Frequency or date offset of the source index.

            Set to False to force-set the frequency to None.
        target_freq (frequency_like or bool): Frequency or date offset of the target index.

            Set to False to force-set the frequency to None.
        silence_warnings (bool): Whether to silence all warnings."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "source_index",
        "target_index",
        "source_freq",
        "target_freq",
        "silence_warnings",
    }

    def __init__(
        self,
        source_index: tp.IndexLike,
        target_index: tp.IndexLike,
        source_freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        target_freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> None:
        source_index = try_to_datetime_index(source_index)
        target_index = try_to_datetime_index(target_index)
        infer_source_freq = True
        if isinstance(source_freq, bool):
            if not source_freq:
                infer_source_freq = False
            source_freq = None
        infer_target_freq = True
        if isinstance(target_freq, bool):
            if not target_freq:
                infer_target_freq = False
            target_freq = None
        if infer_source_freq:
            source_freq = infer_index_freq(source_index, freq=source_freq)
        if infer_target_freq:
            target_freq = infer_index_freq(target_index, freq=target_freq)

        self._source_index = source_index
        self._target_index = target_index
        self._source_freq = source_freq
        self._target_freq = target_freq
        self._silence_warnings = silence_warnings

        Configured.__init__(
            self,
            source_index=source_index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=target_freq,
            silence_warnings=silence_warnings,
        )

    @classmethod
    def from_pd_resampler(
        cls: tp.Type[ResamplerT],
        pd_resampler: tp.PandasResampler,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: bool = True,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.core.resample.Resampler](https://pandas.pydata.org/docs/reference/resampling.html).
        """
        target_index = pd_resampler.count().index
        return cls(
            source_index=pd_resampler.obj.index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=None,
            silence_warnings=silence_warnings,
        )

    @classmethod
    def from_pd_resample(
        cls: tp.Type[ResamplerT],
        source_index: tp.IndexLike,
        *args,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: bool = True,
        **kwargs,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.DataFrame.resample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
        """
        pd_resampler = pd.Series(index=source_index, dtype=object).resample(*args, **kwargs)
        return cls.from_pd_resampler(pd_resampler, source_freq=source_freq, silence_warnings=silence_warnings)

    @classmethod
    def from_pd_date_range(
        cls: tp.Type[ResamplerT],
        source_index: tp.IndexLike,
        *args,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.date_range](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html).
        """
        target_index = pd.date_range(*args, **kwargs)
        return cls(
            source_index=source_index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=None,
            silence_warnings=silence_warnings,
        )

    @property
    def source_index(self) -> tp.Index:
        """Index being resampled."""
        return self._source_index

    @property
    def target_index(self) -> tp.Index:
        """Index resulted from resampling."""
        return self._target_index

    @property
    def source_freq(self) -> tp.AnyPandasFrequency:
        """Frequency or date offset of the source index."""
        return self._source_freq

    @property
    def target_freq(self) -> tp.AnyPandasFrequency:
        """Frequency or date offset of the target index."""
        return self._target_freq

    @property
    def silence_warnings(self) -> bool:
        """Frequency or date offset of the target index."""
        from vectorbtpro._settings import settings

        resampling_cfg = settings["resampling"]

        silence_warnings = self._silence_warnings
        if silence_warnings is None:
            silence_warnings = resampling_cfg["silence_warnings"]
        return silence_warnings

    def get_np_source_freq(self, silence_warnings: tp.Optional[bool] = None) -> tp.AnyPandasFrequency:
        """Frequency or date offset of the source index in NumPy format."""
        if silence_warnings is None:
            silence_warnings = self.silence_warnings

        warned = False
        source_freq = self.source_freq
        if source_freq is not None:
            if not isinstance(source_freq, (int, float)):
                try:
                    source_freq = freq_to_timedelta64(source_freq)
                except ValueError as e:
                    if not silence_warnings:
                        warnings.warn(f"Cannot convert {source_freq} to np.timedelta64. Setting to None.", stacklevel=2)
                        warned = True
                    source_freq = None
        if source_freq is None:
            if not warned and not silence_warnings:
                warnings.warn("Using right bound of source index without frequency. Set source_freq.", stacklevel=2)
        return source_freq

    def get_np_target_freq(self, silence_warnings: tp.Optional[bool] = None) -> tp.AnyPandasFrequency:
        """Frequency or date offset of the target index in NumPy format."""
        if silence_warnings is None:
            silence_warnings = self.silence_warnings

        warned = False
        target_freq = self.target_freq
        if target_freq is not None:
            if not isinstance(target_freq, (int, float)):
                try:
                    target_freq = freq_to_timedelta64(target_freq)
                except ValueError as e:
                    if not silence_warnings:
                        warnings.warn(f"Cannot convert {target_freq} to np.timedelta64. Setting to None.", stacklevel=2)
                        warned = True
                    target_freq = None
        if target_freq is None:
            if not warned and not silence_warnings:
                warnings.warn("Using right bound of target index without frequency. Set target_freq.", stacklevel=2)
        return target_freq

    @classmethod
    def get_lbound_index(cls, index: pd.Index, freq: tp.AnyPandasFrequency = None) -> tp.Index:
        """Get the left bound of a datetime index.

        If `freq` is None, calculates the leftmost bound."""
        index = try_to_datetime_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(-1, freq=freq) + pd.Timedelta(1, "ns")
        min_ts = pd.DatetimeIndex([pd.Timestamp.min.tz_localize(index.tzinfo)])
        return (index[:-1] + pd.Timedelta(1, "ns")).append(min_ts)

    @classmethod
    def get_rbound_index(cls, index: pd.Index, freq: tp.AnyPandasFrequency = None) -> tp.Index:
        """Get the right bound of a datetime index.

        If `freq` is None, calculates the rightmost bound."""
        index = try_to_datetime_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(1, freq=freq) - pd.Timedelta(1, "ns")
        max_ts = pd.DatetimeIndex([pd.Timestamp.max.tz_localize(index.tzinfo)])
        return (index[1:] - pd.Timedelta(1, "ns")).append(max_ts)

    @cached_property
    def source_lbound_index(self) -> tp.Index:
        """Get the left bound of the source datetime index."""
        return self.get_lbound_index(self.source_index, freq=self.source_freq)

    @cached_property
    def source_rbound_index(self) -> tp.Index:
        """Get the right bound of the source datetime index."""
        return self.get_rbound_index(self.source_index, freq=self.source_freq)

    @cached_property
    def target_lbound_index(self) -> tp.Index:
        """Get the left bound of the target datetime index."""
        return self.get_lbound_index(self.target_index, freq=self.target_freq)

    @cached_property
    def target_rbound_index(self) -> tp.Index:
        """Get the right bound of the target datetime index."""
        return self.get_rbound_index(self.target_index, freq=self.target_freq)

    def map_to_target_index(
        self,
        before: bool = False,
        raise_missing: bool = True,
        return_index: bool = True,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[tp.Array1d, tp.Index]:
        """See `vectorbtpro.base.resampling.nb.map_to_target_index_nb`."""
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)
        func = jit_reg.resolve_option(nb.map_to_target_index_nb, jitted)
        mapped_arr = func(
            self.source_index.values,
            self.target_index.values,
            target_freq=target_freq,
            before=before,
            raise_missing=raise_missing,
        )
        if return_index:
            nan_mask = mapped_arr == -1
            if nan_mask.any():
                mapped_index = self.source_index.to_series().copy()
                mapped_index[nan_mask] = np.nan
                mapped_index[~nan_mask] = self.target_index[mapped_arr]
                mapped_index = pd.Index(mapped_index)
            else:
                mapped_index = self.target_index[mapped_arr]
            return mapped_index
        return mapped_arr

    def index_difference(
        self,
        reverse: bool = False,
        return_index: bool = True,
        jitted: tp.JittedOption = None,
    ) -> tp.Union[tp.Array1d, tp.Index]:
        """See `vectorbtpro.base.resampling.nb.index_difference_nb`."""
        func = jit_reg.resolve_option(nb.index_difference_nb, jitted)
        if reverse:
            mapped_arr = func(self.target_index.values, self.source_index.values)
        else:
            mapped_arr = func(self.source_index.values, self.target_index.values)
        if return_index:
            return self.target_index[mapped_arr]
        return mapped_arr

    def map_index_to_source_ranges(
        self,
        before: bool = False,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """See `vectorbtpro.base.resampling.nb.map_index_to_source_ranges_nb`.

        If `Resampler.target_freq` is a date offset, sets is to None and gives a warning.
        Raises another warning is `target_freq` is None."""
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)
        func = jit_reg.resolve_option(nb.map_index_to_source_ranges_nb, jitted)
        return func(
            self.source_index.values,
            self.target_index.values,
            target_freq=target_freq,
            before=before,
        )

    @class_or_instancemethod
    def map_bounds_to_source_ranges(
        cls_or_self,
        source_index: tp.Optional[tp.IndexLike] = None,
        target_lbound_index: tp.Optional[tp.IndexLike] = None,
        target_rbound_index: tp.Optional[tp.IndexLike] = None,
        closed_lbound: bool = True,
        closed_rbound: bool = False,
        skip_minus_one: bool = False,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """See `vectorbtpro.base.resampling.nb.map_bounds_to_source_ranges_nb`.

        Either `target_lbound_index` or `target_rbound_index` must be set.
        Set `target_lbound_index` and `target_rbound_index` to 'pandas' to use
        `Resampler.get_lbound_index` and `Resampler.get_rbound_index` respectively.
        Also, both allow providing a single datetime string and will automatically broadcast
        to the `Resampler.target_index`."""

        if not isinstance(cls_or_self, type):
            if target_lbound_index is None and target_rbound_index is None:
                raise ValueError("Either target_lbound_index or target_rbound_index must be set")
            if target_lbound_index is not None:
                if isinstance(target_lbound_index, str) and target_lbound_index.lower() == "pandas":
                    target_lbound_index = cls_or_self.target_lbound_index
                else:
                    target_lbound_index = try_to_datetime_index(target_lbound_index)
                target_rbound_index = cls_or_self.target_index
            if target_rbound_index is not None:
                target_lbound_index = cls_or_self.target_index
                if isinstance(target_rbound_index, str) and target_rbound_index.lower() == "pandas":
                    target_rbound_index = cls_or_self.target_rbound_index
                else:
                    target_rbound_index = try_to_datetime_index(target_rbound_index)
            if len(target_lbound_index) == 1 and len(target_rbound_index) > 1:
                target_lbound_index = repeat_index(target_lbound_index, len(target_rbound_index))
            elif len(target_lbound_index) > 1 and len(target_rbound_index) == 1:
                target_rbound_index = repeat_index(target_rbound_index, len(target_lbound_index))
        else:
            source_index = try_to_datetime_index(source_index)
            target_lbound_index = try_to_datetime_index(target_lbound_index)
            target_rbound_index = try_to_datetime_index(target_rbound_index)

        checks.assert_len_equal(target_rbound_index, target_lbound_index)
        func = jit_reg.resolve_option(nb.map_bounds_to_source_ranges_nb, jitted)
        return func(
            source_index.values,
            target_lbound_index.values,
            target_rbound_index.values,
            closed_lbound=closed_lbound,
            closed_rbound=closed_rbound,
            skip_minus_one=skip_minus_one,
        )

    def resample_source_mask(
        self,
        source_mask: tp.ArrayLike,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Array1d:
        """See `vectorbtpro.base.resampling.nb.resample_source_mask_nb`."""
        from vectorbtpro.base.reshaping import broadcast_array_to

        if silence_warnings is None:
            silence_warnings = self.silence_warnings
        source_mask = broadcast_array_to(source_mask, len(self.source_index))
        source_freq = self.get_np_source_freq(silence_warnings=silence_warnings)
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)

        func = jit_reg.resolve_option(nb.resample_source_mask_nb, jitted)
        return func(
            source_mask,
            self.source_index.values,
            self.target_index.values,
            source_freq,
            target_freq,
        )
