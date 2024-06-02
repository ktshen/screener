# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with range records.

Range records capture information on ranges. They are useful for analyzing duration of processes,
such as drawdowns, trades, and positions. They also come in handy when analyzing distance between events,
such as entry and exit signals.

Each range has a starting point and an ending point. For example, the points for `range(20)`
are 0 and 20 (not 19!) respectively.

!!! note
    Be aware that if a range hasn't ended in a column, its `end_idx` will point at the latest index.
    Make sure to account for this when computing custom metrics involving duration.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbtpro as vbt

>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> price = vbt.YFData.fetch('BTC-USD', start=start, end=end).get('Close')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> fast_ma = vbt.MA.run(price, 10)
>>> slow_ma = vbt.MA.run(price, 50)
>>> fast_below_slow = fast_ma.ma_above(slow_ma)

>>> ranges = vbt.Ranges.from_array(fast_below_slow, wrapper_kwargs=dict(freq='d'))

>>> ranges.records_readable
   Range Id  Column           Start Timestamp             End Timestamp  \\
0         0       0 2019-02-19 00:00:00+00:00 2019-07-25 00:00:00+00:00
1         1       0 2019-08-08 00:00:00+00:00 2019-08-19 00:00:00+00:00
2         2       0 2019-11-01 00:00:00+00:00 2019-11-20 00:00:00+00:00

   Status
0  Closed
1  Closed
2  Closed

>>> ranges.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('156 days 00:00:00')
```

## From accessors

Moreover, all generic accessors have a property `ranges` and a method `get_ranges`:

```pycon
>>> # vectorbtpro.generic.accessors.GenericAccessor.ranges.coverage
>>> fast_below_slow.vbt.ranges.coverage
0.5081967213114754
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Ranges.metrics`.

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, np.nan, np.nan, 5, 6],
...     'b': [np.nan, 2, np.nan, 4, np.nan, 6]
... })
>>> ranges = df.vbt(freq='d').ranges

>>> ranges['a'].stats()
Start                             0
End                               5
Period              6 days 00:00:00
Total Records                     2
Coverage                   0.666667
Overlap Coverage                0.0
Duration: Min       2 days 00:00:00
Duration: Median    2 days 00:00:00
Duration: Max       2 days 00:00:00
Name: a, dtype: object
```

`Ranges.stats` also supports (re-)grouping:

```pycon
>>> ranges.stats(group_by=True)
Start                                       0
End                                         5
Period                        6 days 00:00:00
Total Records                               5
Coverage                             0.416667
Overlap Coverage                          0.4
Duration: Min                 1 days 00:00:00
Duration: Median              1 days 00:00:00
Duration: Max                 2 days 00:00:00
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Ranges.subplots`.

`Ranges` class has a single subplot based on `Ranges.plot`:

```pycon
>>> ranges['a'].plots().show()
```

![](/assets/images/api/ranges_plots.svg){: .iimg loading=lazy }
"""

import attr
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_pd_array, to_1d_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexes import stack_indexes, combine_indexes, tile_index
from vectorbtpro.generic import nb
from vectorbtpro.generic.enums import (
    RangeStatus,
    range_dt,
    pattern_range_dt,
    InterpMode,
    RescaleMode,
    ErrorType,
    DistanceMeasure,
    OverlapMode,
)
from vectorbtpro.generic.price_records import PriceRecords
from vectorbtpro.records.base import Records
from vectorbtpro.records.decorators import override_field_config, attach_fields, attach_shortcut_properties
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.datetime_ import freq_to_timedelta, freq_to_timedelta64, to_ns
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.params import combine_params, Param
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "Ranges",
    "PatternRanges",
    "PSC",
]

__pdoc__ = {}

# ############# Ranges ############# #

ranges_field_config = ReadonlyConfig(
    dict(
        dtype=range_dt,
        settings=dict(
            id=dict(title="Range Id"),
            idx=dict(name="end_idx"),  # remap field of Records
            start_idx=dict(title="Start Index", mapping="index"),
            end_idx=dict(title="End Index", mapping="index"),
            status=dict(title="Status", mapping=RangeStatus),
        ),
    )
)
"""_"""

__pdoc__[
    "ranges_field_config"
] = f"""Field config for `Ranges`.

```python
{ranges_field_config.prettify()}
```
"""

ranges_attach_field_config = ReadonlyConfig(dict(status=dict(attach_filters=True)))
"""_"""

__pdoc__[
    "ranges_attach_field_config"
] = f"""Config of fields to be attached to `Ranges`.

```python
{ranges_attach_field_config.prettify()}
```
"""

ranges_shortcut_config = ReadonlyConfig(
    dict(
        valid=dict(),
        invalid=dict(),
        first_pd_mask=dict(obj_type="array"),
        last_pd_mask=dict(obj_type="array"),
        ranges_pd_mask=dict(obj_type="array"),
        first_idx=dict(obj_type="mapped_array"),
        last_idx=dict(obj_type="mapped_array"),
        duration=dict(obj_type="mapped_array"),
        real_duration=dict(obj_type="mapped_array"),
        avg_duration=dict(obj_type="red_array"),
        max_duration=dict(obj_type="red_array"),
        coverage=dict(obj_type="red_array"),
        overlap_coverage=dict(method_name="get_coverage", obj_type="red_array", method_kwargs=dict(overlapping=True)),
        projections=dict(obj_type="array"),
    )
)
"""_"""

__pdoc__[
    "ranges_shortcut_config"
] = f"""Config of shortcut properties to be attached to `Ranges`.

```python
{ranges_shortcut_config.prettify()}
```
"""

RangesT = tp.TypeVar("RangesT", bound="Ranges")


@attach_shortcut_properties(ranges_shortcut_config)
@attach_fields(ranges_attach_field_config)
@override_field_config(ranges_field_config)
class Ranges(PriceRecords):
    """Extends `vectorbtpro.generic.price_records.PriceRecords` for working with range records.

    Requires `records_arr` to have all fields defined in `vectorbtpro.generic.enums.range_dt`."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_array(
        cls: tp.Type[RangesT],
        arr: tp.ArrayLike,
        gap_value: tp.Optional[tp.Scalar] = None,
        attach_as_close: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RangesT:
        """Build `Ranges` from an array.

        Searches for sequences of

        * True values in boolean data (False acts as a gap),
        * positive values in integer data (-1 acts as a gap), and
        * non-NaN values in any other data (NaN acts as a gap).

        If `attach_as_close` is True, will attach `arr` as `close`.

        `**kwargs` will be passed to `Ranges.__init__`."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper = ArrayWrapper.from_obj(arr, **wrapper_kwargs)

        arr = to_2d_array(arr)
        if gap_value is None:
            if np.issubdtype(arr.dtype, np.bool_):
                gap_value = False
            elif np.issubdtype(arr.dtype, np.integer):
                gap_value = -1
            else:
                gap_value = np.nan
        func = jit_reg.resolve_option(nb.get_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        records_arr = func(arr, gap_value)
        if attach_as_close and "close" not in kwargs:
            kwargs["close"] = arr
        return cls(wrapper, records_arr, **kwargs)

    @classmethod
    def from_delta(
        cls: tp.Type[RangesT],
        records_or_mapped: tp.Union[Records, MappedArray],
        delta: tp.Union[str, int, tp.FrequencyLike],
        shift: tp.Optional[int] = None,
        idx_field_or_arr: tp.Union[None, str, tp.Array1d] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RangesT:
        """Build `Ranges` from a record/mapped array with a timedelta applied on its index field.

        See `vectorbtpro.generic.nb.records.get_ranges_from_delta_nb`.

        Set `delta` to an integer to wait a certain amount of rows. Set it to anything else to
        wait a timedelta. The conversion is done using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
        The second option requires the index to be datetime-like, or at least the frequency to be set.

        `**kwargs` will be passed to `Ranges.__init__`."""
        if idx_field_or_arr is None:
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr("idx")
            else:
                idx_field_or_arr = records_or_mapped.idx_arr
        if isinstance(idx_field_or_arr, str):
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr(idx_field_or_arr)
            else:
                raise ValueError("Providing an index field is allowed for records only")
        if isinstance(records_or_mapped, Records):
            id_arr = records_or_mapped.get_field_arr("id")
        else:
            id_arr = records_or_mapped.id_arr
        if isinstance(delta, int):
            delta_use_index = False
            index = None
        else:
            delta = to_ns(freq_to_timedelta64(delta))
            if isinstance(records_or_mapped.wrapper.index, pd.DatetimeIndex):
                index = to_ns(records_or_mapped.wrapper.index)
            else:
                freq = to_ns(freq_to_timedelta64(records_or_mapped.wrapper.freq))
                index = np.arange(records_or_mapped.wrapper.shape[0]) * freq
            delta_use_index = True
        if shift is None:
            shift = 0
        col_map = records_or_mapped.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.get_ranges_from_delta_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_records_arr = func(
            records_or_mapped.wrapper.shape[0],
            idx_field_or_arr,
            id_arr,
            col_map,
            index=index,
            delta=delta,
            delta_use_index=delta_use_index,
            shift=shift,
        )
        if isinstance(records_or_mapped, PriceRecords):
            kwargs = merge_dicts(
                dict(
                    open=records_or_mapped._open,
                    high=records_or_mapped._high,
                    low=records_or_mapped._low,
                    close=records_or_mapped._close,
                ),
                kwargs,
            )
        return Ranges.from_records(records_or_mapped.wrapper, new_records_arr, **kwargs)

    def with_delta(self, *args, **kwargs):
        """Pass self to `Ranges.from_delta`."""
        return Ranges.from_delta(self, *args, **kwargs)

    def crop(self) -> RangesT:
        """Remove any data outside the minimum start index and the maximum end index."""
        min_start_idx = np.min(self.get_field_arr("start_idx"))
        max_start_idx = np.max(self.get_field_arr("end_idx")) + 1
        return self.iloc[min_start_idx:max_start_idx]

    # ############# Filtering ############# #

    def filter_min_duration(
        self: RangesT,
        min_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges that last less than a minimum duration."""
        if isinstance(min_duration, int):
            return self.apply_mask(self.duration.values >= min_duration, **kwargs)
        min_duration = freq_to_timedelta64(min_duration)
        if real:
            return self.apply_mask(self.real_duration.values >= min_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq >= min_duration, **kwargs)

    def filter_max_duration(
        self: RangesT,
        max_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges that last more than a maximum duration."""
        if isinstance(max_duration, int):
            return self.apply_mask(self.duration.values <= max_duration, **kwargs)
        max_duration = freq_to_timedelta64(max_duration)
        if real:
            return self.apply_mask(self.real_duration.values <= max_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq <= max_duration, **kwargs)

    # ############# Masking ############# #

    def get_first_pd_mask(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Get mask from `Ranges.get_first_idx`."""
        return self.get_pd_mask(idx_arr=self.first_idx.values, group_by=group_by, **kwargs)

    def get_last_pd_mask(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Get mask from `Ranges.get_last_idx`."""
        out = self.get_pd_mask(idx_arr=self.last_idx.values, group_by=group_by, **kwargs)
        return out

    def get_ranges_pd_mask(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get mask from ranges.

        See `vectorbtpro.generic.nb.records.ranges_to_mask_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.ranges_to_mask_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mask = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            len(self.wrapper.index),
        )
        return self.wrapper.wrap(mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Stats ############# #

    def get_valid(self: RangesT, **kwargs) -> RangesT:
        """Get valid ranges.

        A valid range doesn't have the start and end index set to -1."""
        filter_mask = (self.get_field_arr("start_idx") != -1) & (self.get_field_arr("end_idx") != -1)
        return self.apply_mask(filter_mask, **kwargs)

    def get_invalid(self: RangesT, **kwargs) -> RangesT:
        """Get invalid ranges.

        An invalid range has the start and/or end index set to -1."""
        filter_mask = (self.get_field_arr("start_idx") == -1) | (self.get_field_arr("end_idx") == -1)
        return self.apply_mask(filter_mask, **kwargs)

    def get_first_idx(self, **kwargs):
        """Get the first index in each range."""
        return self.map_field("start_idx", **kwargs)

    def get_last_idx(self, **kwargs):
        """Get the last index in each range."""
        last_idx = self.get_field_arr("end_idx", copy=True)
        status = self.get_field_arr("status")
        last_idx[status == RangeStatus.Closed] -= 1
        return self.map_array(last_idx, **kwargs)

    def get_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get the effective duration of each range in integer format."""
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            freq=1,
        )
        return self.map_array(duration, **kwargs)

    def get_real_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get the real duration of each range in timedelta format."""
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            to_ns(self.get_map_field_to_index("start_idx")),
            to_ns(self.get_map_field_to_index("end_idx")),
            self.get_field_arr("status"),
            freq=to_ns(freq_to_timedelta64(self.wrapper.freq)),
        ).astype("timedelta64[ns]")
        return self.map_array(duration, **kwargs)

    def get_avg_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get average range duration (as timedelta)."""
        if real:
            duration = self.real_duration
            duration = duration.replace(mapped_arr=to_ns(duration.mapped_arr))
            wrap_kwargs = merge_dicts(dict(name_or_index="avg_real_duration", dtype="timedelta64[ns]"), wrap_kwargs)
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index="avg_duration"), wrap_kwargs)
        return duration.mean(group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs)

    def get_max_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get maximum range duration (as timedelta)."""
        if real:
            duration = self.real_duration
            duration = duration.replace(mapped_arr=to_ns(duration.mapped_arr))
            wrap_kwargs = merge_dicts(dict(name_or_index="max_real_duration", dtype="timedelta64[ns]"), wrap_kwargs)
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index="max_duration"), wrap_kwargs)
        return duration.max(group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs)

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get coverage, that is, the number of steps that are covered by all ranges.

        See `vectorbtpro.generic.nb.records.range_coverage_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        index_lens = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.range_coverage_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        coverage = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            index_lens,
            overlapping=overlapping,
            normalize=normalize,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="coverage"), wrap_kwargs)
        return self.wrapper.wrap_reduced(coverage, group_by=group_by, **wrap_kwargs)

    def get_projections(
        self,
        close: tp.Optional[tp.ArrayLike] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = None,
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = None,
        incl_end_idx: bool = True,
        extend: bool = False,
        rebase: bool = True,
        start_value: tp.ArrayLike = 1.0,
        ffill: bool = False,
        remove_empty: bool = True,
        return_raw: bool = False,
        start_index: tp.Optional[pd.Timestamp] = None,
        id_level: tp.Union[None, str, tp.IndexLike] = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        index_stack_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.Tuple[tp.Array1d, tp.Array2d], tp.Frame]:
        """Generate a projection for each range record.

        See `vectorbtpro.generic.nb.records.map_ranges_to_projections_nb`.

        Set `proj_start` to an integer to generate a projection after a certain row
        after the start row. Set it to anything else to wait a timedelta.
        The conversion is done using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
        The second option requires the index to be datetime-like, or at least the frequency to be set.

        Set `proj_period` the same way as `proj_start` to generate a projection of a certain length.
        Unless `extend` is True, it still respects the duration of the range.

        Set `extend` to True to extend the projection even after the end of the range.
        The extending period is taken from the longest range duration if `proj_period` is None,
        and from the longest `proj_period` if it's not None.

        Set `rebase` to True to make each projection start with 1, otherwise, each projection
        will consist of original `close` values during the projected period. Use `start_value`
        to replace 1 with another start value. It can also be a flexible array with elements per column.
        If `start_value` is -1, will set it to the latest row in `close`.

        Set `ffill` to True to forward fill NaN values, even if they are NaN in `close` itself.

        Set `remove_empty` to True to remove projections that are either NaN or with only one element.
        The index of each projection is still being tracked and will appear in the multi-index of the
        returned DataFrame.

        !!! note
            As opposed to the Numba-compiled function, the returned DataFrame will have
            projections stacked along columns rather than rows. Set `return_raw` to True
            to return them in the original format.
        """
        if close is None:
            close = self.close
            checks.assert_not_none(close)
        else:
            close = self.wrapper.wrap(close, group_by=False)
        if proj_start is None:
            proj_start = 0
        if isinstance(proj_start, int):
            proj_start_use_index = False
            index = None
        else:
            proj_start = to_ns(freq_to_timedelta64(proj_start))
            if isinstance(self.wrapper.index, pd.DatetimeIndex):
                index = to_ns(self.wrapper.index)
            else:
                freq = to_ns(freq_to_timedelta64(self.wrapper.freq))
                index = np.arange(self.wrapper.shape[0]) * freq
            proj_start_use_index = True
        if proj_period is not None:
            if isinstance(proj_period, int):
                proj_period_use_index = False
            else:
                proj_period = to_ns(freq_to_timedelta64(proj_period))
                if index is None:
                    if isinstance(self.wrapper.index, pd.DatetimeIndex):
                        index = to_ns(self.wrapper.index)
                    else:
                        freq = to_ns(freq_to_timedelta64(self.wrapper.freq))
                        index = np.arange(self.wrapper.shape[0]) * freq
                proj_period_use_index = True
        else:
            proj_period_use_index = False

        func = jit_reg.resolve_option(nb.map_ranges_to_projections_nb, jitted)
        ridxs, projections = func(
            to_2d_array(close),
            self.get_field_arr("col"),
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            index=index,
            proj_start=proj_start,
            proj_start_use_index=proj_start_use_index,
            proj_period=proj_period,
            proj_period_use_index=proj_period_use_index,
            incl_end_idx=incl_end_idx,
            extend=extend,
            rebase=rebase,
            start_value=to_1d_array(start_value),
            ffill=ffill,
            remove_empty=remove_empty,
        )
        if return_raw:
            return ridxs, projections
        projections = projections.T
        freq = self.wrapper.get_freq(allow_numeric=False)
        wrapper = ArrayWrapper.from_obj(projections, freq=freq)
        if id_level is None:
            id_level = pd.Index(self.id_arr, name="range_id")
        elif isinstance(id_level, str):
            mapping = self.get_field_mapping(id_level)
            if isinstance(mapping, str) and mapping == "index":
                id_level = self.get_map_field_to_index(id_level).rename(id_level)
            else:
                id_level = pd.Index(self.get_apply_mapping_arr(id_level), name=id_level)
        else:
            if not isinstance(id_level, pd.Index):
                id_level = pd.Index(id_level, name="range_id")
        if start_index is None:
            start_index = close.index[-1]
        wrap_kwargs = merge_dicts(
            dict(
                index=pd.date_range(
                    start=start_index,
                    periods=projections.shape[0],
                    freq=freq,
                ),
                columns=stack_indexes(
                    self.wrapper.columns[self.col_arr[ridxs]],
                    id_level[ridxs],
                    **resolve_dict(index_stack_kwargs),
                ),
            ),
            wrap_kwargs,
        )
        return wrapper.wrap(projections, **wrap_kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Ranges.stats`.

        Merges `vectorbtpro.records.base.Records.stats_defaults` and
        `stats` from `vectorbtpro._settings.ranges`."""
        from vectorbtpro._settings import settings

        ranges_stats_cfg = settings["ranges"]["stats"]

        return merge_dicts(Records.stats_defaults.__get__(self), ranges_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(title="Start", calc_func=lambda self: self.wrapper.index[0], agg_func=None, tags="wrapper"),
            end=dict(title="End", calc_func=lambda self: self.wrapper.index[-1], agg_func=None, tags="wrapper"),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                tags=["ranges", "coverage"],
            ),
            duration=dict(
                title="Duration",
                calc_func="duration.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                apply_to_timedelta=True,
                tags=["ranges", "duration"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_projections(
        self,
        column: tp.Optional[tp.Label] = None,
        min_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        max_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        last_n: tp.Optional[int] = None,
        top_n: tp.Optional[int] = None,
        random_n: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_0",
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = "max",
        incl_end_idx: bool = True,
        extend: bool = False,
        ffill: bool = False,
        plot_past_period: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_proj_period",
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_projections: bool = True,
        plot_bands: bool = True,
        plot_lower: tp.Union[bool, str, tp.Callable] = True,
        plot_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_upper: tp.Union[bool, str, tp.Callable] = True,
        plot_aux_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_fill: bool = True,
        colorize: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        projection_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        aux_middle_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot projections.

        Combines generation of projections using `Ranges.get_projections` and
        their plotting using `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.

        Args:
            column (str): Name of the column to plot.
            min_duration (str, int, or frequency_like): Filter range records by minimum duration.
            max_duration (str, int, or frequency_like): Filter range records by maximum duration.
            last_n (int): Select last N range records.
            top_n (int): Select top N range records by maximum duration.
            random_n (int): Select N range records randomly.
            seed (int): Set seed to make output deterministic.
            proj_start (str, int, or frequency_like): See `Ranges.get_projections`.

                Allows an additional option "current_or_{value}", which sets `proj_start` to
                the duration of the current open range, and to the specified value if there is no open range.
            proj_period (str, int, or frequency_like): See `Ranges.get_projections`.

                Allows additional options "current_or_{option}", "mean", "min", "max", "median", or
                a percentage such as "50%" representing a quantile. All of those options are based
                on the duration of all the closed ranges filtered by the arguments above.
            incl_end_idx (bool): See `Ranges.get_projections`.
            extend (bool): See `Ranges.get_projections`.
            ffill (bool): See `Ranges.get_projections`.
            plot_past_period (str, int, or frequency_like): Past period to plot.

                Allows the same options as `proj_period` plus "proj_period" and "current_or_proj_period".
            plot_ohlc (bool or DataFrame): Whether to plot OHLC.
            plot_close (bool or Series): Whether to plot close.
            plot_projections (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_bands (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_lower (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_middle (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_upper (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_aux_middle (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_fill (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            colorize (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Ranges.close`.
            projection_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for projections.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.plotly.graph_objects.Scatter` for lower band.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for middle band.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for upper band.
            aux_middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for auxiliary middle band.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> price = pd.Series(
            ...     [11, 12, 13, 14, 11, 12, 13, 12, 11, 12],
            ...     index=pd.date_range("2020", periods=10),
            ... )
            >>> vbt.Ranges.from_array(
            ...     price >= 12,
            ...     attach_as_close=False,
            ...     close=price,
            ... ).plot_projections(
            ...     proj_start=0,
            ...     proj_period=4,
            ...     extend=True,
            ...     plot_past_period=None
            ... ).show()
            ```

            ![](/assets/images/api/ranges_plot_projections.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        self_col_open = self_col.status_open
        self_col = self_col.status_closed
        if proj_start is not None:
            if isinstance(proj_start, str) and proj_start.startswith("current_or_"):
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    proj_start = int(self_col_open.duration.values[0])
                else:
                    proj_start = proj_start.replace("current_or_", "")
                    if proj_start.isnumeric():
                        proj_start = int(proj_start)
            if proj_start != 0:
                self_col = self_col.filter_min_duration(proj_start, real=True)
        if min_duration is not None:
            self_col = self_col.filter_min_duration(min_duration, real=True)
        if max_duration is not None:
            self_col = self_col.filter_max_duration(max_duration, real=True)
        if last_n is not None:
            self_col = self_col.last_n(last_n)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))
        if random_n is not None:
            self_col = self_col.random_n(random_n, seed=seed)
        if self_col.count() == 0:
            warnings.warn("No ranges to plot. Relax the requirements.")

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True
        if close is None:
            raise ValueError("Close cannot be None")

        # Resolve windows
        def _resolve_period(period):
            if self_col.count() == 0:
                period = None
            if period is not None:
                if isinstance(period, str):
                    period = period.lower().replace(" ", "")
                    if period == "median":
                        period = "50%"
                    if "%" in period:
                        period = int(
                            np.quantile(
                                self_col.duration.values,
                                float(period.replace("%", "")) / 100,
                            )
                        )
                    elif period.startswith("current_or_"):
                        if self_col_open.count() > 0:
                            if self_col_open.count() > 1:
                                raise ValueError("Only one open range is allowed")
                            period = int(self_col_open.duration.values[0])
                        else:
                            period = period.replace("current_or_", "")
                            return _resolve_period(period)
                    elif period == "mean":
                        period = int(np.mean(self_col.duration.values))
                    elif period == "min":
                        period = int(np.min(self_col.duration.values))
                    elif period == "max":
                        period = int(np.max(self_col.duration.values))
            return period

        proj_period = _resolve_period(proj_period)
        if isinstance(proj_period, int) and proj_period == 0:
            warnings.warn("Projection period is zero. Setting to maximum.")
            proj_period = int(np.max(self_col.duration.values))
        if plot_past_period is not None and isinstance(plot_past_period, str):
            plot_past_period = plot_past_period.lower().replace(" ", "")
            if plot_past_period == "proj_period":
                plot_past_period = proj_period
            elif plot_past_period == "current_or_proj_period":
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    plot_past_period = int(self_col_open.duration.values[0])
                else:
                    plot_past_period = proj_period
        plot_past_period = _resolve_period(plot_past_period)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot OHLC/close
        if plot_ohlc and ohlc is not None:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _ohlc = ohlc.iloc[-plot_past_period:]
                else:
                    plot_past_period = freq_to_timedelta(plot_past_period)
                    _ohlc = ohlc[ohlc.index > ohlc.index[-1] - plot_past_period]
            else:
                _ohlc = ohlc
            if _ohlc.size > 0:
                if "opacity" not in ohlc_trace_kwargs:
                    ohlc_trace_kwargs["opacity"] = 0.5
                fig = _ohlc.vbt.ohlcv.plot(
                    ohlc_type=ohlc_type,
                    plot_volume=False,
                    ohlc_trace_kwargs=ohlc_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
        elif plot_close:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _close = close.iloc[-plot_past_period:]
                else:
                    plot_past_period = freq_to_timedelta(plot_past_period)
                    _close = close[close.index > close.index[-1] - plot_past_period]
            else:
                _close = close
            if _close.size > 0:
                fig = _close.vbt.lineplot(
                    trace_kwargs=close_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        if self_col.count() > 0:
            # Get projections
            projections = self_col.get_projections(
                close=close,
                proj_start=proj_start,
                proj_period=proj_period,
                incl_end_idx=incl_end_idx,
                extend=extend,
                rebase=True,
                start_value=close.iloc[-1],
                ffill=ffill,
                remove_empty=True,
                return_raw=False,
            )

            if len(projections.columns) > 0:
                # Plot projections
                rename_levels = dict(range_id=self_col.get_field_title("id"))
                fig = projections.vbt.plot_projections(
                    plot_projections=plot_projections,
                    plot_bands=plot_bands,
                    plot_lower=plot_lower,
                    plot_middle=plot_middle,
                    plot_upper=plot_upper,
                    plot_aux_middle=plot_aux_middle,
                    plot_fill=plot_fill,
                    colorize=colorize,
                    rename_levels=rename_levels,
                    projection_trace_kwargs=projection_trace_kwargs,
                    upper_trace_kwargs=upper_trace_kwargs,
                    middle_trace_kwargs=middle_trace_kwargs,
                    lower_trace_kwargs=lower_trace_kwargs,
                    aux_middle_trace_kwargs=aux_middle_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        return fig

    def plot_shapes(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot range shapes.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool or DataFrame): Whether to plot OHLC.
            plot_close (bool or Series): Whether to plot close.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Ranges.close`.
            shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for shapes.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            * Plot zones colored by duration:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> price = pd.Series(
            ...     [1, 2, 1, 2, 3, 2, 1, 2, 3],
            ...     index=pd.date_range("2020", periods=9),
            ... )

            >>> def get_opacity(self_col, i):
            ...     real_duration = self_col.get_real_duration().values
            ...     return real_duration[i] / real_duration.max() * 0.5

            >>> vbt.Ranges.from_array(price >= 2).plot_shapes(
            ...     shape_kwargs=dict(fillcolor="teal", opacity=vbt.RepFunc(get_opacity))
            ... ).show()
            ```

            ![](/assets/images/api/ranges_plot_shapes.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if shape_kwargs is None:
            shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(yref, fig)
        y_domain = get_domain(yref, fig)

        # Plot OHLC/close
        if plot_ohlc and ohlc is not None:
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and close is not None:
            fig = close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            end_idx = self_col.get_map_field_to_index("end_idx")
            for i in range(len(self_col.values)):
                start_index = start_idx[i]
                end_index = end_idx[i]
                _shape_kwargs = substitute_templates(
                    shape_kwargs,
                    context=dict(
                        self_col=self_col,
                        i=i,
                        record=self_col.values[i],
                        start_index=start_index,
                        end_index=end_index,
                        xref=xref,
                        yref=yref,
                        x_domain=x_domain,
                        y_domain=y_domain,
                        close=close,
                        ohlc=ohlc,
                    ),
                    sub_id="shape_kwargs",
                )
                _shape_kwargs = merge_dicts(
                    dict(
                        type="rect",
                        xref=xref,
                        yref="paper",
                        x0=start_index,
                        y0=y_domain[0],
                        x1=end_index,
                        y1=y_domain[1],
                        fillcolor="gray",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    ),
                    _shape_kwargs,
                )
                fig.add_shape(**_shape_kwargs)

        return fig

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        top_n: tp.Optional[int] = None,
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        start_trace_kwargs: tp.KwargsLike = None,
        end_trace_kwargs: tp.KwargsLike = None,
        open_shape_kwargs: tp.KwargsLike = None,
        closed_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        return_close: bool = False,
        **layout_kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.Tuple[tp.BaseFigure, tp.Series]]:
        """Plot ranges.

        Args:
            column (str): Name of the column to plot.
            top_n (int): Filter top N range records by maximum duration.
            plot_ohlc (bool or DataFrame): Whether to plot OHLC.
            plot_close (bool or Series): Whether to plot close.
            plot_markers (bool): Whether to plot markers.
            plot_zones (bool): Whether to plot zones.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Ranges.close`.
            start_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for start values.
            end_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for end values.
            open_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for open zones.
            closed_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for closed zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            return_close (bool): Whether to return the close series along with the figure.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> price = pd.Series(
            ...     [1, 2, 1, 2, 3, 2, 1, 2, 3],
            ...     index=pd.date_range("2020", periods=9),
            ... )
            >>> vbt.Ranges.from_array(price >= 2).plot().show()
            ```

            ![](/assets/images/api/ranges_plot.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if start_trace_kwargs is None:
            start_trace_kwargs = {}
        if end_trace_kwargs is None:
            end_trace_kwargs = {}
        if open_shape_kwargs is None:
            open_shape_kwargs = {}
        if closed_shape_kwargs is None:
            closed_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        y_domain = get_domain(yref, fig)

        # Plot OHLC/close
        plotting_ohlc = False
        if plot_ohlc and ohlc is not None:
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            plotting_ohlc = True
        elif plot_close and close is not None:
            fig = close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            if plotting_ohlc and self_col.open is not None:
                start_val = self_col.open.loc[start_idx]
            elif close is not None:
                start_val = close.loc[start_idx]
            else:
                start_val = np.full(len(start_idx), 0)
            end_idx = self_col.get_map_field_to_index("end_idx")
            if close is not None:
                end_val = close.loc[end_idx]
            else:
                end_val = np.full(len(end_idx), 0)
            status = self_col.get_field_arr("status")

            if plot_markers:
                # Plot start markers
                start_customdata, start_hovertemplate = self_col.prepare_customdata(incl_fields=["id", "start_idx"])
                _start_trace_kwargs = merge_dicts(
                    dict(
                        x=start_idx,
                        y=start_val,
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            color=plotting_cfg["contrast_color_schema"]["blue"],
                            size=7,
                            line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["blue"])),
                        ),
                        name="Start",
                        customdata=start_customdata,
                        hovertemplate=start_hovertemplate,
                    ),
                    start_trace_kwargs,
                )
                start_scatter = go.Scatter(**_start_trace_kwargs)
                fig.add_trace(start_scatter, **add_trace_kwargs)

            closed_mask = status == RangeStatus.Closed
            if closed_mask.any():
                if plot_markers:
                    # Plot end markers
                    closed_end_customdata, closed_end_hovertemplate = self_col.prepare_customdata(mask=closed_mask)
                    _end_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[closed_mask],
                            y=end_val[closed_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["green"],
                                size=7,
                                line=dict(
                                    width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["green"])
                                ),
                            ),
                            name="Closed",
                            customdata=closed_end_customdata,
                            hovertemplate=closed_end_hovertemplate,
                        ),
                        end_trace_kwargs,
                    )
                    closed_end_scatter = go.Scatter(**_end_trace_kwargs)
                    fig.add_trace(closed_end_scatter, **add_trace_kwargs)

            open_mask = status == RangeStatus.Open
            if open_mask.any():
                if plot_markers:
                    # Plot end markers
                    open_end_customdata, open_end_hovertemplate = self_col.prepare_customdata(
                        excl_fields=["end_idx"], mask=open_mask
                    )
                    _end_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[open_mask],
                            y=end_val[open_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["orange"],
                                size=7,
                                line=dict(
                                    width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["orange"])
                                ),
                            ),
                            name="Open",
                            customdata=open_end_customdata,
                            hovertemplate=open_end_hovertemplate,
                        ),
                        end_trace_kwargs,
                    )
                    open_end_scatter = go.Scatter(**_end_trace_kwargs)
                    fig.add_trace(open_end_scatter, **add_trace_kwargs)

            if plot_zones:
                # Plot closed range zones
                self_col.status_closed.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    shape_kwargs=merge_dicts(
                        dict(fillcolor=plotting_cfg["contrast_color_schema"]["green"]),
                        closed_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot open range zones
                self_col.status_open.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    shape_kwargs=merge_dicts(
                        dict(fillcolor=plotting_cfg["contrast_color_schema"]["orange"]),
                        open_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

        if return_close:
            return fig, close
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Ranges.plots`.

        Merges `vectorbtpro.records.base.Records.plots_defaults` and
        `plots` from `vectorbtpro._settings.ranges`."""
        from vectorbtpro._settings import settings

        ranges_plots_cfg = settings["ranges"]["plots"]

        return merge_dicts(Records.plots_defaults.__get__(self), ranges_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(plot=dict(title="Ranges", check_is_not_grouped=True, plot_func="plot", tags="ranges")),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Ranges.override_field_config_doc(__pdoc__)
Ranges.override_metrics_doc(__pdoc__)
Ranges.override_subplots_doc(__pdoc__)


# ############# Pattern ranges ############# #


PatternRangesT = tp.TypeVar("PatternRangesT", bound="PatternRanges")


_DEF = object()
"""Default value for internal purposes."""


@attr.s(frozen=True, eq=False)
class PSC:
    """Class that represents a pattern search config.

    Every field will be resolved into the format suitable for Numba."""

    pattern: tp.Union[tp.ArrayLike] = attr.ib(default=_DEF)
    """Flexible pattern array.
    
    Can be smaller or bigger than the source array; in such a case, the values of the smaller array
    will be "stretched" by interpolation of the type in `PSC.interp_mode`."""

    window: tp.Optional[int] = attr.ib(default=_DEF)
    """Minimum window.
    
    Defaults to the length of `PSC.pattern`."""

    max_window: tp.Optional[int] = attr.ib(default=_DEF)
    """Maximum window (including)."""

    row_select_prob: tp.Union[float] = attr.ib(default=_DEF)
    """Row selection probability."""

    window_select_prob: tp.Union[float] = attr.ib(default=_DEF)
    """Window selection probability."""

    roll_forward: tp.Union[bool] = attr.ib(default=_DEF)
    """Whether to roll windows to the left of the current row, otherwise to the right."""

    interp_mode: tp.Union[int, str] = attr.ib(default=_DEF)
    """Interpolation mode. See `vectorbtpro.generic.enums.InterpMode`."""

    rescale_mode: tp.Union[int, str] = attr.ib(default=_DEF)
    """Rescaling mode. See `vectorbtpro.generic.enums.RescaleMode`."""

    vmin: tp.Union[float] = attr.ib(default=_DEF)
    """Minimum value of any window. Should only be used when the array has fixed bounds.
    
    Used in rescaling using `RescaleMode.MinMax` and checking against `PSC.min_pct_change` and `PSC.max_pct_change`.
    
    If `np.nan`, gets calculated dynamically."""

    vmax: tp.Union[float] = attr.ib(default=_DEF)
    """Maximum value of any window. Should only be used when the array has fixed bounds.
    
    Used in rescaling using `RescaleMode.MinMax` and checking against `PSC.min_pct_change` and `PSC.max_pct_change`.
    
    If `np.nan`, gets calculated dynamically."""

    pmin: tp.Union[float] = attr.ib(default=_DEF)
    """Value to be considered as the minimum of `PSC.pattern`.
    
    Used in rescaling using `RescaleMode.MinMax` and calculating the maximum distance at each point 
    if `PSC.max_error_as_maxdist` is disabled.
    
    If `np.nan`, gets calculated dynamically."""

    pmax: tp.Union[float] = attr.ib(default=_DEF)
    """Value to be considered as the maximum of `PSC.pattern`.
    
    Used in rescaling using `RescaleMode.MinMax` and calculating the maximum distance at each point 
    if `PSC.max_error_as_maxdist` is disabled.
    
    If `np.nan`, gets calculated dynamically."""

    invert: tp.Union[bool] = attr.ib(default=_DEF)
    """Whether to invert the pattern vertically."""

    error_type: tp.Union[int, str] = attr.ib(default=_DEF)
    """Error type. See `vectorbtpro.generic.enums.ErrorType`."""

    distance_measure: tp.Union[int, str] = attr.ib(default=_DEF)
    """Distance measure. See `vectorbtpro.generic.enums.DistanceMeasure`."""

    max_error: tp.Union[tp.ArrayLike] = attr.ib(default=_DEF)
    """Maximum error at each point. Can be provided as a flexible array.
    
    If `max_error` is an array, it must be of the same size as the pattern array.
    It also should be provided within the same scale as the pattern."""

    max_error_interp_mode: tp.Union[None, int, str] = attr.ib(default=_DEF)
    """Interpolation mode for `PSC.max_error`. See `vectorbtpro.generic.enums.InterpMode`.
    
    If None, defaults to `PSC.interp_mode`."""

    max_error_as_maxdist: tp.Union[bool] = attr.ib(default=_DEF)
    """Whether `PSC.max_error` should be used as the maximum distance at each point.
    
    If False, crossing `PSC.max_error` will set the distance to the maximum distance
    based on `PSC.pmin`, `PSC.pmax`, and the pattern value at that point.
    
    If True and any of the points in a window is `np.nan`, the point will be skipped."""

    max_error_strict: tp.Union[bool] = attr.ib(default=_DEF)
    """Whether crossing `PSC.max_error` even once should yield the similarity of `np.nan`."""

    min_pct_change: tp.Union[float] = attr.ib(default=_DEF)
    """Minimum percentage change of the window to stay a candidate for search.

    If any window doesn't cross this mark, its similarity becomes `np.nan`."""

    max_pct_change: tp.Union[float] = attr.ib(default=_DEF)
    """Maximum percentage change of the window to stay a candidate for search.

    If any window crosses this mark, its similarity becomes `np.nan`."""

    min_similarity: tp.Union[float] = attr.ib(default=_DEF)
    """Minimum similarity.
    
    If any window doesn't cross this mark, its similarity becomes `np.nan`."""

    minp: tp.Optional[int] = attr.ib(default=_DEF)
    """Minimum number of observations in price window required to have a value."""

    overlap_mode: tp.Union[int, str] = attr.ib(default=_DEF)
    """Overlapping mode. See `vectorbtpro.generic.enums.OverlapMode`."""

    max_records: tp.Optional[int] = attr.ib(default=_DEF)
    """Maximum number of records expected to be filled.
    
    Set to avoid creating empty arrays larger than needed."""

    name: tp.Optional[str] = attr.ib(default=None)
    """Name of the config."""

    def __eq__(self, other):
        return checks.is_deep_equal(self, other)

    def __hash__(self):
        dct = attr.asdict(self)
        if isinstance(dct["pattern"], np.ndarray):
            dct["pattern"] = tuple(dct["pattern"])
        else:
            dct["pattern"] = (dct["pattern"],)
        if isinstance(dct["max_error"], np.ndarray):
            dct["max_error"] = tuple(dct["max_error"])
        else:
            dct["max_error"] = (dct["max_error"],)
        return hash(tuple(dct.items()))


pattern_ranges_field_config = ReadonlyConfig(
    dict(
        dtype=pattern_range_dt,
        settings=dict(
            id=dict(title="Pattern Range Id"),
            similarity=dict(title="Similarity"),
        ),
    )
)
"""_"""

__pdoc__[
    "pattern_ranges_field_config"
] = f"""Field config for `PatternRanges`.

```python
{pattern_ranges_field_config.prettify()}
```
"""


@attach_fields
@override_field_config(pattern_ranges_field_config)
class PatternRanges(Ranges):
    """Extends `Ranges` for working with range records generated from pattern search."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def resolve_search_config(cls, search_config: tp.Union[None, dict, PSC] = None, **kwargs) -> PSC:
        """Resolve search config for `PatternRanges.from_pattern_search`.

        Converts array-like objects into arrays and enums into integers."""
        if search_config is None:
            search_config = dict()
        if isinstance(search_config, dict):
            search_config = PSC(**search_config)
        search_config = attr.asdict(search_config)
        defaults = {}
        for k, v in get_func_kwargs(cls.from_pattern_search).items():
            if k in search_config:
                defaults[k] = v
        defaults = merge_dicts(defaults, kwargs)
        for k, v in search_config.items():
            if v is _DEF:
                v = defaults[k]
            if k == "pattern":
                if v is None:
                    raise ValueError("Pattern must be provided")
                v = to_1d_array(v)
            elif k == "max_error":
                v = to_1d_array(v)
            elif k == "interp_mode":
                v = map_enum_fields(v, InterpMode)
            elif k == "rescale_mode":
                v = map_enum_fields(v, RescaleMode)
            elif k == "error_type":
                v = map_enum_fields(v, ErrorType)
            elif k == "distance_measure":
                v = map_enum_fields(v, DistanceMeasure)
            elif k == "max_error_interp_mode":
                if v is None:
                    v = search_config["interp_mode"]
                else:
                    v = map_enum_fields(v, InterpMode)
            elif k == "overlap_mode":
                v = map_enum_fields(v, OverlapMode)
            search_config[k] = v
        return PSC(**search_config)

    @classmethod
    def from_pattern_search(
        cls: tp.Type[PatternRangesT],
        arr: tp.ArrayLike,
        pattern: tp.Union[Param, tp.ArrayLike] = None,
        window: tp.Union[Param, None, int] = None,
        max_window: tp.Union[Param, None, int] = None,
        row_select_prob: tp.Union[Param, float] = 1.0,
        window_select_prob: tp.Union[Param, float] = 1.0,
        roll_forward: tp.Union[Param, bool] = False,
        interp_mode: tp.Union[Param, int, str] = "mixed",
        rescale_mode: tp.Union[Param, int, str] = "minmax",
        vmin: tp.Union[Param, float] = np.nan,
        vmax: tp.Union[Param, float] = np.nan,
        pmin: tp.Union[Param, float] = np.nan,
        pmax: tp.Union[Param, float] = np.nan,
        invert: bool = False,
        error_type: tp.Union[Param, int, str] = "absolute",
        distance_measure: tp.Union[Param, int, str] = "mae",
        max_error: tp.Union[Param, tp.ArrayLike] = np.nan,
        max_error_interp_mode: tp.Union[Param, None, int, str] = None,
        max_error_as_maxdist: tp.Union[Param, bool] = False,
        max_error_strict: tp.Union[Param, bool] = False,
        min_pct_change: tp.Union[Param, float] = np.nan,
        max_pct_change: tp.Union[Param, float] = np.nan,
        min_similarity: tp.Union[Param, float] = 0.85,
        minp: tp.Union[Param, None, int] = None,
        overlap_mode: tp.Union[Param, int, str] = "disallow",
        max_records: tp.Union[Param, None, int] = None,
        random_subset: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        search_configs: tp.Optional[tp.Sequence[tp.MaybeSequence[PSC]]] = None,
        jitted: tp.JittedOption = None,
        execute_kwargs: tp.KwargsLike = None,
        attach_as_close: bool = True,
        index_stack_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PatternRangesT:
        """Build `PatternRanges` from all occurrences of a pattern in an array.

        Searches for parameters of the type `vectorbtpro.utils.params.Param`, and if found, broadcasts
        and combines them using `vectorbtpro.utils.params.combine_params`. Then, converts them
        into a list of search configurations. If none of such parameters was found among the passed arguments,
        builds one search configuration using the passed arguments. If `search_configs` is not None, uses it
        instead. In all cases, it uses the defaults defined in the signature of this method to augment
        search configurations. For example, passing `min_similarity` of 95% will use it in all search
        configurations except where it was explicitly overridden.

        Argument `search_configs` must be provided as a sequence of `PSC` instances.
        If any element is a list of `PSC` instances itself, it will be used per column in `arr`,
        otherwise per entire `arr`. Each configuration will be resolved using `PatternRanges.resolve_search_config`
        to prepare arguments for the use in Numba.

        After all the search configurations have been resolved, uses `vectorbtpro.utils.execution.execute`
        to loop over each configuration and execute it using `vectorbtpro.generic.nb.records.find_pattern_1d_nb`.
        The results are then concatenated into a single records array and wrapped with `PatternRanges`.

        If `attach_as_close` is True, will attach `arr` as `close`.

        `**kwargs` will be passed to `PatternRanges.__init__`."""
        if seed is not None:
            set_seed(seed)
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        arr = to_pd_array(arr)
        arr_2d = to_2d_array(arr)
        arr_wrapper = ArrayWrapper.from_obj(arr)
        psc_keys = [a.name for a in PSC.__attrs_attrs__ if a.name != "name"]
        method_locals = {k: v for k, v in locals().items() if k in psc_keys}

        # Flatten search configs
        flat_search_configs = []
        psc_names = []
        psc_names_none = True
        n_configs = 0
        if search_configs is not None:
            for maybe_search_config in search_configs:
                if isinstance(maybe_search_config, dict):
                    maybe_search_config = PSC(**maybe_search_config)
                if isinstance(maybe_search_config, PSC):
                    for col in range(arr_2d.shape[1]):
                        flat_search_configs.append(maybe_search_config)
                        if maybe_search_config.name is not None:
                            psc_names.append(maybe_search_config.name)
                            psc_names_none = False
                        else:
                            psc_names.append(n_configs)
                    n_configs += 1
                else:
                    if len(maybe_search_config) != arr_2d.shape[1]:
                        raise ValueError("Sub-list with PSC instances must match the number of columns")
                    for col, search_config in enumerate(maybe_search_config):
                        if isinstance(search_config, dict):
                            search_config = PSC(**search_config)
                        flat_search_configs.append(search_config)
                        if search_config.name is not None:
                            psc_names.append(search_config.name)
                            psc_names_none = False
                        else:
                            psc_names.append(n_configs)
                        n_configs += 1

        # Combine parameters
        param_dct = {}
        for k, v in method_locals.items():
            if k in psc_keys and isinstance(v, Param):
                param_dct[k] = v
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                index_stack_kwargs=index_stack_kwargs,
            )
            if len(flat_search_configs) == 0:
                flat_search_configs = []
                for i in range(len(param_columns)):
                    search_config = dict()
                    for k, v in param_product.items():
                        search_config[k] = v[i]
                    for col in range(arr_2d.shape[1]):
                        flat_search_configs.append(PSC(**search_config))
            else:
                new_flat_search_configs = []
                for i in range(len(param_columns)):
                    for search_config in flat_search_configs:
                        new_search_config = dict()
                        for k, v in attr.asdict(search_config).items():
                            if v is not _DEF:
                                if k in param_product:
                                    raise ValueError(f"Parameter '{k}' is re-defined in a search configuration")
                                new_search_config[k] = v
                            if k in param_product:
                                new_search_config[k] = param_product[k][i]
                        new_flat_search_configs.append(PSC(**new_search_config))
                flat_search_configs = new_flat_search_configs

        # Create config from arguments if empty
        if len(flat_search_configs) == 0:
            for col in range(arr_2d.shape[1]):
                flat_search_configs.append(PSC())

        # Prepare function and arguments
        funcs_args = []
        func = jit_reg.resolve_option(nb.find_pattern_1d_nb, jitted)
        def_func_kwargs = get_func_kwargs(func)
        new_search_configs = []
        for c in range(len(flat_search_configs)):
            func_kwargs = {
                "col": c,
                "arr": arr_2d[:, c % arr_2d.shape[1]],
            }
            new_search_config = cls.resolve_search_config(flat_search_configs[c], **method_locals)
            for k, v in attr.asdict(new_search_config).items():
                if k == "name":
                    continue
                if isinstance(v, Param):
                    raise TypeError(f"Cannot use Param inside search configs")
                if k in def_func_kwargs:
                    if v is not def_func_kwargs[k]:
                        func_kwargs[k] = v
                else:
                    func_kwargs[k] = v
            funcs_args.append((func, (), func_kwargs))
            new_search_configs.append(new_search_config)

        # Execute each configuration
        execute_kwargs = merge_dicts(
            dict(show_progress=len(flat_search_configs) > 1),
            execute_kwargs,
        )
        result_list = execute(funcs_args, **execute_kwargs)
        records_arr = np.concatenate(result_list)

        # Build column hierarchy
        n_config_params = len(psc_names) // arr_2d.shape[1]
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and psc_names_none):
                new_columns = combine_indexes((param_columns, arr_wrapper.columns), **index_stack_kwargs)
            else:
                search_config_index = pd.Index(psc_names, name="search_config")
                base_columns = stack_indexes(
                    (search_config_index, tile_index(arr_wrapper.columns, n_config_params)), **index_stack_kwargs
                )
                new_columns = combine_indexes((param_columns, base_columns), **index_stack_kwargs)
        else:
            if n_config_params == 0 or (n_config_params == 1 and psc_names_none):
                new_columns = arr_wrapper.columns
            else:
                search_config_index = pd.Index(psc_names, name="search_config")
                new_columns = stack_indexes(
                    (search_config_index, tile_index(arr_wrapper.columns, n_config_params)),
                    **index_stack_kwargs,
                )

        # Wrap with class
        wrapper = ArrayWrapper(
            **merge_dicts(
                dict(
                    index=arr_wrapper.index,
                    columns=new_columns,
                ),
                wrapper_kwargs,
            )
        )
        if attach_as_close and "close" not in kwargs:
            kwargs["close"] = arr
        return cls(wrapper, records_arr, new_search_configs, **kwargs)

    def with_delta(self, *args, **kwargs):
        """Pass self to `Ranges.from_delta` but with the index set to the last index."""
        if "idx_field_or_arr" not in kwargs:
            kwargs["idx_field_or_arr"] = self.last_idx.values
        return Ranges.from_delta(self, *args, **kwargs)

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[PatternRangesT],
        *objs: tp.MaybeTuple[PatternRangesT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PatternRanges` after stacking along columns."""
        kwargs = Ranges.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PatternRanges):
                raise TypeError("Each object to be merged must be an instance of PatternRanges")
        new_search_configs = []
        for obj in objs:
            if len(obj.search_configs) == 1:
                new_search_configs.append(obj.search_configs * len(kwargs["wrapper"].columns))
            else:
                new_search_configs.append(obj.search_configs)
            if len(new_search_configs) >= 2:
                if new_search_configs[-1] != new_search_configs[0]:
                    raise ValueError(f"Objects to be merged must have compatible PSC instances. Pass to override.")
        kwargs["search_configs"] = new_search_configs[0]
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[PatternRangesT],
        *objs: tp.MaybeTuple[PatternRangesT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PatternRanges` after stacking along columns."""
        kwargs = Ranges.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PatternRanges):
                raise TypeError("Each object to be merged must be an instance of PatternRanges")
        kwargs["search_configs"] = [search_config for obj in objs for search_config in obj.search_configs]
        return kwargs

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Ranges._expected_keys or set()) | {
        "search_configs",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        search_configs: tp.List[PSC],
        **kwargs,
    ) -> None:
        Ranges.__init__(
            self,
            wrapper,
            records_arr,
            search_configs=search_configs,
            **kwargs,
        )

        self._search_configs = search_configs

    def indexing_func(self: PatternRangesT, *args, ranges_meta: tp.DictLike = None, **kwargs) -> PatternRangesT:
        """Perform indexing on `PatternRanges`."""
        if ranges_meta is None:
            ranges_meta = Ranges.indexing_func_meta(self, *args, **kwargs)
        col_idxs = ranges_meta["wrapper_meta"]["col_idxs"]
        if not isinstance(col_idxs, slice):
            col_idxs = to_1d_array(col_idxs)
        col_idxs = np.arange(self.wrapper.shape_2d[1])[col_idxs]
        new_search_configs = []
        for i in col_idxs:
            new_search_configs.append(self.search_configs[i])
        return self.replace(
            wrapper=ranges_meta["wrapper_meta"]["new_wrapper"],
            records_arr=ranges_meta["new_records_arr"],
            search_configs=new_search_configs,
            open=ranges_meta["open"],
            high=ranges_meta["high"],
            low=ranges_meta["low"],
            close=ranges_meta["close"],
        )

    @property
    def search_configs(self) -> tp.List[PSC]:
        """List of `PSC` instances, one per column."""
        return self._search_configs

    # ############# Stats ############# #

    _metrics: tp.ClassVar[Config] = HybridConfig(
        {
            **Ranges.metrics,
            "similarity": dict(
                title="Similarity",
                calc_func="similarity.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                tags=["pattern_ranges", "similarity"],
            ),
        }
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plots ############# #

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        top_n: tp.Optional[int] = None,
        plot_patterns: bool = True,
        plot_max_error: bool = False,
        fill_distance: bool = True,
        pattern_trace_kwargs: tp.KwargsLike = None,
        lower_max_error_trace_kwargs: tp.KwargsLike = None,
        upper_max_error_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot pattern ranges.

        Based on `Ranges.plot` and `vectorbtpro.generic.accessors.GenericSRAccessor.plot_pattern`.

        Args:
            column (str): Name of the column to plot.
            top_n (int): Filter top N range records by maximum duration.
            plot_patterns (bool or array_like): Whether to plot `PSC.pattern`.
            plot_max_error (array_like): Whether to plot `PSC.max_error`.
            fill_distance (bool): Whether to fill the space between close and pattern.

                Visible for every interpolation mode except discrete.
            pattern_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for pattern.
            lower_max_error_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for lower max error.
            upper_max_error_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for upper max error.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **kwargs: Keyword arguments passed to `Ranges.plot`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))
        search_config = self_col.search_configs[0]

        if pattern_trace_kwargs is None:
            pattern_trace_kwargs = {}
        if lower_max_error_trace_kwargs is None:
            lower_max_error_trace_kwargs = {}
        if upper_max_error_trace_kwargs is None:
            upper_max_error_trace_kwargs = {}

        open_shape_kwargs = merge_dicts(
            dict(fillcolor=plotting_cfg["contrast_color_schema"]["blue"]),
            kwargs.pop("open_shape_kwargs", None),
        )
        closed_shape_kwargs = merge_dicts(
            dict(fillcolor=plotting_cfg["contrast_color_schema"]["blue"]),
            kwargs.pop("closed_shape_kwargs", None),
        )
        fig, close = Ranges.plot(
            self_col,
            return_close=True,
            open_shape_kwargs=open_shape_kwargs,
            closed_shape_kwargs=closed_shape_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            xref=xref,
            yref=yref,
            fig=fig,
            **kwargs,
        )

        if self_col.count() > 0:
            # Extract information
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            end_idx = self_col.get_map_field_to_index("end_idx")
            status = self_col.get_field_arr("status")

            if plot_patterns:
                # Plot pattern
                for r in range(len(start_idx)):
                    _start_idx = start_idx[r]
                    _end_idx = end_idx[r]
                    if close is None:
                        raise ValueError("Must provide close to overlay patterns")
                    arr_sr = close.loc[_start_idx:_end_idx]
                    if status[r] == RangeStatus.Closed:
                        arr_sr = arr_sr.iloc[:-1]
                    if fill_distance:
                        obj_trace_kwargs = dict(
                            line=dict(color="rgba(0, 0, 0, 0)", width=0),
                            opacity=0,
                            hoverinfo="skip",
                            showlegend=False,
                            name=None,
                        )
                    else:
                        obj_trace_kwargs = None
                    _pattern_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="pattern",
                            showlegend=r == 0,
                        ),
                        pattern_trace_kwargs,
                    )
                    _lower_max_error_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="max_error",
                            showlegend=r == 0,
                        ),
                        lower_max_error_trace_kwargs,
                    )
                    _upper_max_error_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="max_error",
                            showlegend=False,
                        ),
                        upper_max_error_trace_kwargs,
                    )

                    fig = arr_sr.vbt.plot_pattern(
                        pattern=search_config.pattern,
                        interp_mode=search_config.interp_mode,
                        rescale_mode=search_config.rescale_mode,
                        vmin=search_config.vmin,
                        vmax=search_config.vmax,
                        pmin=search_config.pmin,
                        pmax=search_config.pmax,
                        invert=search_config.invert,
                        error_type=search_config.error_type,
                        max_error=search_config.max_error if plot_max_error else np.nan,
                        max_error_interp_mode=search_config.max_error_interp_mode,
                        plot_obj=fill_distance,
                        fill_distance=fill_distance,
                        obj_trace_kwargs=obj_trace_kwargs,
                        pattern_trace_kwargs=_pattern_trace_kwargs,
                        lower_max_error_trace_kwargs=_lower_max_error_trace_kwargs,
                        upper_max_error_trace_kwargs=_upper_max_error_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        fig=fig,
                    )

        return fig


PatternRanges.override_field_config_doc(__pdoc__)
PatternRanges.override_metrics_doc(__pdoc__)
