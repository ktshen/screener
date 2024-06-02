# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for splitting."""

import attr
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import is_range
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.template import CustomTemplate, Rep, RepFunc, substitute_templates
from vectorbtpro.utils.decorators import class_or_instancemethod
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.datetime_ import (
    try_to_datetime_index,
    try_align_dt_to_index,
    try_align_to_dt_index,
    parse_timedelta,
)
from vectorbtpro.utils.execution import execute
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexing import hslice, PandasIndexer, get_index_ranges
from vectorbtpro.base.indexes import combine_indexes, stack_indexes
from vectorbtpro.base.reshaping import to_dict
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.merging import row_stack_merge, column_stack_merge, resolve_merge_func
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.splitting import nb

if tp.TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator as BaseCrossValidatorT
else:
    BaseCrossValidatorT = tp.Any

__all__ = [
    "FixRange",
    "RelRange",
    "Takeable",
    "Splitter",
    "SKLSplitter",
]

__pdoc__ = {}


SplitterT = tp.TypeVar("SplitterT", bound="Splitter")


@attr.s(frozen=True)
class FixRange:
    """Class that represents a fixed range."""

    range_: tp.FixRangeLike = attr.ib()
    """Range."""


@attr.s(frozen=True)
class RelRange:
    """Class that represents a relative range."""

    offset: tp.Union[int, float, tp.TimedeltaLike] = attr.ib(default=0)
    """Offset.
    
    Floating values between 0 and 1 are considered relative.
    
    Can be negative."""

    offset_anchor: str = attr.ib(default="prev_end")
    """Offset anchor.
    
    Supported are
    
    * 'start': Start of the range
    * 'end': End of the range
    * 'prev_start': Start of the previous range
    * 'prev_end': End of the previous range
    """

    offset_space: str = attr.ib(default="free")
    """Offset space.

    Supported are

    * 'all': All space
    * 'free': Remaining space after the offset anchor
    * 'prev': Length of the previous range
    
    Applied only when `RelRange.offset` is a relative number."""

    length: tp.Union[int, float, tp.TimedeltaLike] = attr.ib(default=1.0)
    """Length.
    
    Floating values between 0 and 1 are considered relative.
    
    Can be negative."""

    length_space: str = attr.ib(default="free")
    """Length space.
    
    Supported are
    
    * 'all': All space
    * 'free': Remaining space after the offset
    * 'free_or_prev': Remaining space after the offset or the start/end of the previous range,
    depending what comes first in the direction of `RelRange.length`
    
    Applied only when `RelRange.length` is a relative number."""

    out_of_bounds: str = attr.ib(default="warn")
    """Check if start and stop are within bounds.
    
    Supported are
    
    * 'keep': Keep out-of-bounds values
    * 'ignore': Ignore if out-of-bounds
    * 'warn': Emit a warning if out-of-bounds
    * 'raise": Raise an error if out-of-bounds
    """

    is_gap: bool = attr.ib(default=False)
    """Whether the range acts as a gap."""

    def __attrs_post_init__(self):
        object.__setattr__(self, "offset_anchor", self.offset_anchor.lower())
        if self.offset_anchor not in ("start", "end", "prev_start", "prev_end", "next_start", "next_end"):
            raise ValueError(f"Invalid option offset_anchor='{self.offset_anchor}'")
        object.__setattr__(self, "offset_space", self.offset_space.lower())
        if self.offset_space not in ("all", "free", "prev"):
            raise ValueError(f"Invalid option offset_space='{self.offset_space}'")
        object.__setattr__(self, "length_space", self.length_space.lower())
        if self.length_space not in ("all", "free", "free_or_prev"):
            raise ValueError(f"Invalid option length_space='{self.length_space}'")
        object.__setattr__(self, "out_of_bounds", self.out_of_bounds.lower())
        if self.out_of_bounds not in ("keep", "ignore", "warn", "raise"):
            raise ValueError(f"Invalid option out_of_bounds='{self.out_of_bounds}'")

    def to_slice(
        self,
        total_len: int,
        prev_start: int = 0,
        prev_end: int = 0,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> slice:
        """Convert the relative range into a slice."""
        if index is not None:
            index = try_to_datetime_index(index)
            freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        offset_anchor = self.offset_anchor
        offset = self.offset
        length = self.length
        if not checks.is_number(offset) or not checks.is_number(length):
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(f"Index must be of type pandas.DatetimeIndex, not {index.dtype}")

        if offset_anchor == "start":
            if checks.is_number(offset):
                offset_anchor = 0
            else:
                offset_anchor = index[0]
        elif offset_anchor == "end":
            if checks.is_number(offset):
                offset_anchor = total_len
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                offset_anchor = index[-1] + freq
        elif offset_anchor == "prev_start":
            if checks.is_number(offset):
                offset_anchor = prev_start
            else:
                offset_anchor = index[prev_start]
        else:
            if checks.is_number(offset):
                offset_anchor = prev_end
            else:
                if prev_end < total_len:
                    offset_anchor = index[prev_end]
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset_anchor = index[-1] + freq

        if checks.is_float(offset) and 0 <= abs(offset) <= 1:
            if self.offset_space == "all":
                offset = offset_anchor + int(offset * total_len)
            elif self.offset_space == "free":
                if offset < 0:
                    offset = int((1 + offset) * offset_anchor)
                else:
                    offset = offset_anchor + int(offset * (total_len - offset_anchor))
            else:
                offset = offset_anchor + int(offset * (prev_end - prev_start))
        else:
            if checks.is_float(offset):
                if not offset.is_integer():
                    raise TypeError(f"Floating number for offset ({offset}) must be between 0 and 1")
                offset = offset_anchor + int(offset)
            elif not checks.is_int(offset):
                offset = offset_anchor + parse_timedelta(offset)
                if index[0] <= offset <= index[-1]:
                    offset = index.get_indexer([offset], method="ffill")[0]
                elif offset < index[0]:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = -int((index[0] - offset) / freq)
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = total_len - 1 + int((offset - index[-1]) / freq)
            else:
                offset = offset_anchor + offset

        if checks.is_float(length) and 0 <= abs(length) <= 1:
            if self.length_space == "all":
                length = int(length * total_len)
            elif self.length_space == "free":
                if length < 0:
                    length = int(length * offset)
                else:
                    length = int(length * (total_len - offset))
            else:
                if length < 0:
                    if offset > prev_end:
                        length = int(length * (offset - prev_end))
                    else:
                        length = int(length * offset)
                else:
                    if offset < prev_start:
                        length = int(length * (prev_start - offset))
                    else:
                        length = int(length * (total_len - offset))
        else:
            if checks.is_float(length):
                if not length.is_integer():
                    raise TypeError(f"Floating number for length ({length}) must be between 0 and 1")
                length = int(length)
            elif not checks.is_int(length):
                length = parse_timedelta(length)

        start = offset
        if checks.is_int(length):
            stop = start + length
        else:
            if 0 <= start < total_len:
                stop = index[start] + length
            elif start < 0:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[0] + start * freq + length
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[-1] + (start - total_len + 1) * freq + length
            if stop <= index[-1]:
                stop = index.get_indexer([stop], method="bfill")[0]
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = total_len - 1 + int((stop - index[-1]) / freq)
        if checks.is_int(length):
            if length < 0:
                start, stop = stop, start
        else:
            if length < pd.Timedelta(0):
                start, stop = stop, start
        if start < 0:
            if self.out_of_bounds == "ignore":
                start = 0
            elif self.out_of_bounds == "warn":
                warnings.warn(f"Range start ({start}) is out of bounds", stacklevel=2)
                start = 0
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range start ({start}) is out of bounds")
        if stop > total_len:
            if self.out_of_bounds == "ignore":
                stop = total_len
            elif self.out_of_bounds == "warn":
                warnings.warn(f"Range stop ({stop}) is out of bounds", stacklevel=2)
                stop = total_len
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range stop ({stop}) is out of bounds")
        if stop - start <= 0:
            raise ValueError("Range length is negative or zero")
        return slice(start, stop)


_DEF = object()
"""Default value for internal purposes."""


@attr.s(frozen=True)
class Takeable:
    """Class that represents an object from which a range can be taken."""

    obj: tp.Any = attr.ib()
    """Takeable object."""

    remap_to_obj: bool = attr.ib(default=_DEF)
    """Whether to remap `Splitter.index` to the index of `Takeable.obj`.
    
    Otherwise, will assume that the object has the same index."""

    index: tp.Optional[tp.IndexLike] = attr.ib(default=_DEF)
    """Index of the object.
    
    If not present, will be accessed using `Splitter.get_obj_index`."""

    freq: tp.Optional[tp.FrequencyLike] = attr.ib(default=_DEF)
    """Frequency of `Takeable.index`."""

    point_wise: bool = attr.ib(default=_DEF)
    """Whether to select one range point at a time and return a tuple."""


class Splitter(Analyzable):
    """Base class for splitting."""

    @classmethod
    def from_splits(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        splits: tp.Splits,
        squeeze: bool = False,
        fix_ranges: bool = True,
        wrap_with_fixrange: bool = False,
        split_range_kwargs: tp.KwargsLike = None,
        split_check_template: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from an iterable of splits.

        Argument `splits` supports both absolute and relative ranges.
        To transform relative ranges into the absolute format, enable `fix_ranges`.
        Arguments `split_range_kwargs` are then passed to `Splitter.split_range`.

        Enable `wrap_with_fixrange` to wrap any fixed range with `FixRange`. If the range
        is an array, it will be wrapped regardless of this argument to avoid building a 3d array.

        Pass a template via `split_check_template` to discard splits that do not fulfill certain criteria.
        The current split will be available as `split`. Should return a boolean (`False` to discard).

        Labels for splits and sets can be provided via `split_labels` and `set_labels` respectively.
        Both arguments can be provided as templates. The split array will be available as `splits`."""
        index = try_to_datetime_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}

        new_splits = []
        removed_indices = []
        for i, split in enumerate(splits):
            already_fixed = False
            if checks.is_number(split) or checks.is_td_like(split):
                split = cls.split_range(
                    slice(None),
                    split,
                    template_context=template_context,
                    index=index,
                    wrap_with_fixrange=False,
                    **split_range_kwargs,
                )
                already_fixed = True
                new_split = split
                ndim = 2
            elif cls.is_range_relative(split):
                new_split = [split]
                ndim = 1
            elif not checks.is_sequence(split):
                new_split = [split]
                ndim = 1
            elif isinstance(split, np.ndarray):
                new_split = [split]
                ndim = 1
            else:
                new_split = split
                ndim = 2
            if fix_ranges and not already_fixed:
                new_split = cls.split_range(
                    slice(None),
                    new_split,
                    template_context=template_context,
                    index=index,
                    wrap_with_fixrange=False,
                    **split_range_kwargs,
                )
            _new_split = []
            for range_ in new_split:
                if checks.is_number(range_) or checks.is_td_like(range_):
                    range_ = RelRange(length=range_)
                if not isinstance(range_, (FixRange, RelRange)):
                    if wrap_with_fixrange or checks.is_sequence(range_):
                        _new_split.append(FixRange(range_))
                    else:
                        _new_split.append(range_)
                else:
                    _new_split.append(range_)
            if split_check_template is not None:
                _template_context = merge_dicts(dict(index=index, i=i, split=_new_split), template_context)
                split_ok = substitute_templates(split_check_template, _template_context, sub_id="split_check_template")
                if not split_ok:
                    removed_indices.append(i)
                    continue
            new_splits.append(_new_split)
        if len(new_splits) == 0:
            raise ValueError("Must provide at least one range")
        new_splits_arr = np.asarray(new_splits, dtype=object)
        if squeeze and new_splits_arr.shape[1] == 1:
            ndim = 1

        if split_labels is None:
            split_labels = pd.RangeIndex(stop=new_splits_arr.shape[0], name="split")
        else:
            if isinstance(split_labels, CustomTemplate):
                _template_context = merge_dicts(dict(index=index, splits_arr=new_splits_arr), template_context)
                split_labels = substitute_templates(split_labels, _template_context, sub_id=split_labels)
                if not isinstance(split_labels, pd.Index):
                    split_labels = pd.Index(split_labels, name="split")
            else:
                if not isinstance(split_labels, pd.Index):
                    split_labels = pd.Index(split_labels, name="split")
                if len(removed_indices) > 0:
                    split_labels = split_labels.delete(removed_indices)
        if set_labels is None:
            set_labels = pd.Index(["set_%d" % i for i in range(new_splits_arr.shape[1])], name="set")
        else:
            if isinstance(split_labels, CustomTemplate):
                _template_context = merge_dicts(dict(index=index, splits_arr=new_splits_arr), template_context)
                set_labels = substitute_templates(set_labels, _template_context, sub_id=set_labels)
            if not isinstance(set_labels, pd.Index):
                set_labels = pd.Index(set_labels, name="set")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper = ArrayWrapper(index=split_labels, columns=set_labels, ndim=ndim, **wrapper_kwargs)
        return cls(wrapper, index, new_splits_arr, **kwargs)

    @classmethod
    def from_single(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split: tp.Optional[tp.SplitLike],
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a single split."""
        if split_range_kwargs is None:
            split_range_kwargs = {}
        new_split = cls.split_range(
            slice(None),
            split,
            template_context=template_context,
            index=index,
            **split_range_kwargs,
        )
        splits = [new_split]

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        length: tp.Union[int, float, tp.TimedeltaLike],
        offset: tp.Union[int, float, tp.TimedeltaLike] = 0,
        offset_anchor: str = "prev_end",
        offset_anchor_set: tp.Optional[int] = 0,
        offset_space: str = "prev",
        backwards: tp.Union[bool, str] = False,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a rolling range of a fixed length.

        Uses `Splitter.from_splits` to prepare the splits array and labels, and to build the instance.

        Args:
            index (index_like): Index.
            length (int, float, or timedelta_like): See `RelRange.length`.
            offset (int, float, or timedelta_like): See `RelRange.offset`.
            offset_anchor (str): See `RelRange.offset_anchor`.
            offset_anchor_set (int): Offset anchor set.

                Selects the set from the previous range to be used as an offset anchor.
                If None, the whole previous split is considered as a single range.
                By default, it's the first set.
            offset_space (str): See `RelRange.offset_space`.
            backwards (bool or str): Whether to roll backwards.

                If 'sorted', will roll backwards and sort the resulting splits by the start index.
            split (any): Ranges to split the range into.

                If None, will produce the entire range as a single range.
                Otherwise, will use `Splitter.split_range` to split the range into multiple ranges.
            split_range_kwargs (dict): Keyword arguments passed to `Splitter.split_range`.
            range_bounds_kwargs (dict): Keyword arguments passed to `Splitter.get_range_bounds`.
            template_context (dict): Mapping used to substitute templates in ranges.
            freq (any): Index frequency in case it cannot be parsed from `index`.

                If None, will be parsed using `vectorbtpro.base.accessors.BaseIDXAccessor.get_freq`.
            **kwargs: Keyword arguments passed to the constructor of `Splitter`.

        Usage:
            * Divide a range into a set of non-overlapping ranges:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_rolling(index, 30)
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_1.svg){: .iimg loading=lazy }

            * Divide a range into ranges, each split into 1/2:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_2.svg){: .iimg loading=lazy }

            * Make the ranges above non-overlapping by using the right bound of the last
            set as an offset anchor:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     offset_anchor_set=-1,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_3.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if isinstance(backwards, str):
            if backwards.lower() == "sorted":
                sort_backwards = True
            else:
                raise ValueError(f"Invalid option backwards='{backwards}'")
            backwards = True
        else:
            sort_backwards = False
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = RelRange(
                    length=-length if backwards else length,
                    offset_anchor="end" if backwards else "start",
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), index=index, freq=freq)
            else:
                if offset_anchor_set is None:
                    prev_start, prev_end = bounds[-1][0][0], bounds[-1][-1][1]
                else:
                    prev_start, prev_end = bounds[-1][offset_anchor_set]
                new_split = RelRange(
                    offset=offset,
                    offset_anchor=offset_anchor,
                    offset_space=offset_space,
                    length=-length if backwards else length,
                    length_space="all",
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), prev_start=prev_start, prev_end=prev_end, index=index, freq=freq)
                if backwards:
                    if new_split.stop >= bounds[-1][-1][1]:
                        raise ValueError("Infinite loop detected. Provide a positive offset.")
                else:
                    if new_split.start <= bounds[-1][0][0]:
                        raise ValueError("Infinite loop detected. Provide a positive offset.")
            if backwards:
                if new_split.start < 0:
                    break
                if new_split.stop > len(index):
                    raise ValueError("Range stop cannot exceed index length")
            else:
                if new_split.start < 0:
                    raise ValueError("Range start cannot be negative")
                if new_split.stop > len(index):
                    break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits[::-1] if sort_backwards else splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        length: tp.Union[None, int, float, tp.TimedeltaLike] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of rolling ranges of the same length.

        If `length` is None, splits the index evenly into `n` non-overlapping ranges
        using `Splitter.from_rolling`. Otherwise, picks `n` evenly-spaced, potentially overlapping
        ranges of a fixed length. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll 10 ranges with 100 elements, and split it into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_rolling(
            ...     index,
            ...     10,
            ...     length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_rolling.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if length is None:
            return cls.from_rolling(
                index,
                length=len(index) // n,
                offset=0,
                offset_anchor="prev_end",
                offset_anchor_set=None,
                split=split,
                split_range_kwargs=split_range_kwargs,
                template_context=template_context,
                **kwargs,
            )

        if checks.is_float(length):
            if 0 <= abs(length) <= 1:
                length = len(index) * length
            elif not length.is_integer():
                raise TypeError("Floating number for length must be between 0 and 1")
            length = int(length)
        if checks.is_int(length):
            if length < 1 or length > len(index):
                raise TypeError(f"Length must be within [{1}, {len(index)}]")
            offsets = np.arange(len(index))
            offsets = offsets[offsets + length <= len(index)]
        else:
            length = parse_timedelta(length)
            if freq is None:
                raise ValueError("Must provide freq")
            if length < freq or length > index[-1] + freq - index[0]:
                raise TypeError(f"Length must be within [{freq}, {index[-1] + freq - index[0]}]")
            offsets = index[index + length <= index[-1] + freq] - index[0]
        if n > len(offsets):
            n = len(offsets)
        rows = np.round(np.linspace(0, len(offsets) - 1, n)).astype(int)
        offsets = offsets[rows]

        splits = []
        for offset in offsets:
            new_split = RelRange(
                offset=offset,
                length=length,
            ).to_slice(len(index), index=index, freq=freq)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        min_length: tp.Union[int, float, tp.TimedeltaLike],
        offset: tp.Union[int, float, tp.TimedeltaLike],
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from an expanding range.

        Argument `min_length` is the minimum length of the expanding range. Provide it as
        a float between 0 and 1 to make it relative to the length of the index. Argument `offset` is
        an offset after the right bound of the previous range from which the next range should start.
        It can also be a float relative to the index length. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll an expanding range with a length of 10 and an offset of 10, and split it into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_expanding(
            ...     index,
            ...     10,
            ...     10,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_expanding.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = RelRange(
                    length=min_length,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), index=index, freq=freq)
            else:
                prev_end = bounds[-1][-1][-1]
                new_split = RelRange(
                    offset=offset,
                    offset_anchor="prev_end",
                    offset_space="all",
                    length=-1.0,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), prev_end=prev_end, index=index, freq=freq)
                if new_split.stop <= prev_end:
                    raise ValueError("Infinite loop detected. Provide a positive offset.")
            if new_split.start < 0:
                raise ValueError("Range start cannot be negative")
            if new_split.stop > len(index):
                break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[None, int, float, tp.TimedeltaLike] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of expanding ranges.

        Picks `n` evenly-spaced, expanding ranges. Argument `min_length` defines the minimum
        length for each range. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll 10 expanding ranges with a minimum length of 100, while reserving 50 elements for test:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_expanding(
            ...     index,
            ...     10,
            ...     min_length=100,
            ...     split=-50,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_expanding.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if min_length is None:
            min_length = len(index) // n
        if checks.is_float(min_length):
            if 0 <= abs(min_length) <= 1:
                min_length = len(index) * min_length
            elif not min_length.is_integer():
                raise TypeError("Floating number for minimum length must be between 0 and 1")
        if checks.is_int(min_length):
            min_length = int(min_length)
            if min_length < 1 or min_length > len(index):
                raise TypeError(f"Minimum length must be within [{1}, {len(index)}]")
            lengths = np.arange(1, len(index) + 1)
            lengths = lengths[lengths >= min_length]
        else:
            min_length = parse_timedelta(min_length)
            if freq is None:
                raise ValueError("Must provide freq")
            if min_length < freq or min_length > index[-1] + freq - index[0]:
                raise TypeError(f"Minimum length must be within [{freq}, {index[-1] + freq - index[0]}]")
            lengths = index[1:].append(index[[-1]] + freq) - index[0]
            lengths = lengths[lengths >= min_length]
        if n > len(lengths):
            n = len(lengths)
        rows = np.round(np.linspace(0, len(lengths) - 1, n)).astype(int)
        lengths = lengths[rows]

        splits = []
        for length in lengths:
            new_split = RelRange(length=length).to_slice(len(index), index=index, freq=freq)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_ranges(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        *args,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from ranges.

        Uses `vectorbtpro.base.indexing.get_index_ranges` to generate start and end indices.
        Other keyword arguments will be passed to `Splitter.from_splits`. For details on
        `split` and `split_range_kwargs`, see `Splitter.from_rolling`.

        Usage:
            * Translate each quarter into a range:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_ranges(index, every="QS")
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_ranges_1.svg){: .iimg loading=lazy }

            * In addition to the above, reserve the last month for testing purposes:

            ```pycon
            >>> splitter = vbt.Splitter.from_ranges(
            ...     index,
            ...     every="QS",
            ...     split=(1.0, lambda index: index.month == index.month[-1]),
            ...     split_range_kwargs=dict(backwards=True)
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_ranges_2.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        func_arg_names = get_func_arg_names(get_index_ranges)
        ranges_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in func_arg_names:
                ranges_kwargs[k] = kwargs.pop(k)

        start_idxs, stop_idxs = get_index_ranges(index, *args, skip_minus_one=True, **ranges_kwargs)
        splits = []
        for start, stop in zip(start_idxs, stop_idxs):
            new_split = slice(start, stop)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_grouper(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        by: tp.AnyGroupByLike,
        groupby_kwargs: tp.KwargsLike = None,
        grouper_kwargs: tp.KwargsLike = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a grouper.

        See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Uses `Splitter.from_splits` to prepare the splits array and labels, and to build the instance.

        Usage:
            * Map each month into a range:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> def is_month_end(index, split):
            ...     last_range = split[-1]
            ...     return index[last_range][-1].is_month_end

            >>> splitter = vbt.Splitter.from_grouper(
            ...     index,
            ...     "M",
            ...     split_check_template=vbt.RepFunc(is_month_end)
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_grouper.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if grouper_kwargs is None:
            grouper_kwargs = {}

        if isinstance(by, CustomTemplate):
            _template_context = merge_dicts(dict(index=index), template_context)
            by = substitute_templates(by, _template_context, sub_id="by")
        grouper = BaseIDXAccessor(index).get_grouper(by, groupby_kwargs=groupby_kwargs, **grouper_kwargs)
        splits = []
        indices = []
        for i, new_split in enumerate(grouper.yield_group_idxs()):
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            else:
                new_split = [new_split]
            splits.append(new_split)
            indices.append(i)

        if split_labels is None:
            split_labels = grouper.get_index()[indices]
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            split_labels=split_labels,
            **kwargs,
        )

    @classmethod
    def from_n_random(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[int, float, tp.TimedeltaLike],
        max_length: tp.Union[None, int, float, tp.TimedeltaLike] = None,
        min_start: tp.Union[None, int, float, tp.DatetimeLike] = None,
        max_end: tp.Union[None, int, float, tp.DatetimeLike] = None,
        length_choice_func: tp.Optional[tp.Callable] = None,
        start_choice_func: tp.Optional[tp.Callable] = None,
        length_p_func: tp.Optional[tp.Callable] = None,
        start_p_func: tp.Optional[tp.Callable] = None,
        seed: tp.Optional[int] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of random ranges.

        Randomly picks the length of a range between `min_length` and `max_length` (including) using
        `length_choice_func`, which receives an array of possible values and selects one. It defaults to
        `numpy.random.Generator.choice`. Optional function `length_p_func` takes the same as
        `length_choice_func` and must return either None or probabilities.

        Randomly picks the start position of a range starting at `min_start` and ending at `max_end`
        (excluding) minus the chosen length using `start_choice_func`, which receives an array of possible
        values and selects one. It defaults to `numpy.random.Generator.choice`. Optional function
        `start_p_func` takes the same as `start_choice_func` and must return either None or probabilities.

        !!! note
            Each function must take two arguments: the iteration index and the array with possible values.

        For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Generate 20 random ranges with a length from [40, 100], and split each into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_random(
            ...     index,
            ...     20,
            ...     min_length=40,
            ...     max_length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_random.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if min_start is None:
            min_start = 0
        if min_start is not None:
            if checks.is_float(min_start):
                if 0 <= abs(min_start) <= 1:
                    min_start = len(index) * min_start
                elif not min_start.is_integer():
                    raise TypeError("Floating number for minimum start must be between 0 and 1")
            if checks.is_float(min_start):
                min_start = int(min_start)
            if checks.is_int(min_start):
                if min_start < 0 or min_start > len(index) - 1:
                    raise TypeError(f"Minimum start must be within [{0}, {len(index) - 1}]")
            else:
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type pandas.DatetimeIndex, not {index.dtype}")
                min_start = try_align_dt_to_index(min_start, index)
                if not isinstance(min_start, pd.Timestamp):
                    raise ValueError(f"Minimum start ({min_start}) could not be parsed")
                if min_start < index[0] or min_start > index[-1]:
                    raise TypeError(f"Minimum start must be within [{index[0]}, {index[-1]}]")
                min_start = index.get_indexer([min_start], method="bfill")[0]
        if max_end is None:
            max_end = len(index)
        if checks.is_float(max_end):
            if 0 <= abs(max_end) <= 1:
                max_end = len(index) * max_end
            elif not max_end.is_integer():
                raise TypeError("Floating number for maximum end must be between 0 and 1")
        if checks.is_float(max_end):
            max_end = int(max_end)
        if checks.is_int(max_end):
            if max_end < 1 or max_end > len(index):
                raise TypeError(f"Maximum end must be within [{1}, {len(index)}]")
        else:
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(f"Index must be of type pandas.DatetimeIndex, not {index.dtype}")
            max_end = try_align_dt_to_index(max_end, index)
            if not isinstance(max_end, pd.Timestamp):
                raise ValueError(f"Maximum end ({max_end}) could not be parsed")
            if freq is None:
                raise ValueError("Must provide freq")
            if max_end < index[0] + freq or max_end > index[-1] + freq:
                raise TypeError(f"Maximum end must be within [{index[0] + freq}, {index[-1] + freq}]")
            if max_end > index[-1]:
                max_end = len(index)
            else:
                max_end = index.get_indexer([max_end], method="bfill")[0]
        space_len = max_end - min_start
        if not checks.is_number(min_length):
            index_min_start = index[min_start]
            if max_end < len(index):
                index_max_end = index[max_end]
            else:
                if freq is None:
                    raise ValueError("Must provide freq")
                index_max_end = index[-1] + freq
            index_space_len = index_max_end - index_min_start
        else:
            index_min_start = None
            index_max_end = None
            index_space_len = None

        if checks.is_float(min_length):
            if 0 <= abs(min_length) <= 1:
                min_length = space_len * min_length
            elif not min_length.is_integer():
                raise TypeError("Floating number for minimum length must be between 0 and 1")
            min_length = int(min_length)
        if checks.is_int(min_length):
            if min_length < 1 or min_length > space_len:
                raise TypeError(f"Minimum length must be within [{1}, {space_len}]")
        else:
            min_length = parse_timedelta(min_length)
            if freq is None:
                raise ValueError("Must provide freq")
            if min_length < freq or min_length > index_space_len:
                raise TypeError(f"Minimum length must be within [{freq}, {index_space_len}]")
        if max_length is not None:
            if checks.is_float(max_length):
                if 0 <= abs(max_length) <= 1:
                    max_length = space_len * max_length
                elif not max_length.is_integer():
                    raise TypeError("Floating number for maximum length must be between 0 and 1")
                max_length = int(max_length)
            if checks.is_int(max_length):
                if max_length < min_length or max_length > space_len:
                    raise TypeError(f"Maximum length must be within [{min_length}, {space_len}]")
            else:
                max_length = parse_timedelta(max_length)
                if freq is None:
                    raise ValueError("Must provide freq")
                if max_length < min_length or max_length > index_space_len:
                    raise TypeError(f"Maximum length must be within [{min_length}, {index_space_len}]")
        else:
            max_length = min_length

        rng = np.random.default_rng(seed=seed)
        if length_p_func is None:
            length_p_func = lambda i, x: None
        if start_p_func is None:
            start_p_func = lambda i, x: None
        if length_choice_func is None:
            length_choice_func = lambda i, x: rng.choice(x, p=length_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if start_choice_func is None:
            start_choice_func = lambda i, x: rng.choice(x, p=start_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if checks.is_int(min_length):
            length_space = np.arange(min_length, max_length + 1)
        else:
            if freq is None:
                raise ValueError("Must provide freq")
            length_space = np.arange(min_length // freq, max_length // freq + 1) * freq
        index_space = np.arange(len(index))

        splits = []
        for i in range(n):
            length = length_choice_func(i, length_space)
            if checks.is_int(length):
                start = start_choice_func(i, index_space[min_start : max_end - length + 1])
            else:
                from_dt = index_min_start.to_datetime64()
                to_dt = index_max_end.to_datetime64() - length
                start = start_choice_func(i, index_space[(index.values >= from_dt) & (index.values <= to_dt)])
            new_split = RelRange(offset=start, length=length).to_slice(len(index), index=index, freq=freq)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_sklearn(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        splitter: BaseCrossValidatorT,
        groups: tp.Optional[tp.ArrayLike] = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a scikit-learn's splitter.

        The splitter must be an instance of `sklearn.model_selection.BaseCrossValidator`.

        Uses `Splitter.from_splits` to prepare the splits array and labels, and to build the instance."""
        from sklearn.model_selection import BaseCrossValidator

        index = try_to_datetime_index(index)
        checks.assert_instance_of(splitter, BaseCrossValidator)
        if set_labels is None:
            set_labels = ["train", "test"]

        indices_generator = splitter.split(np.arange(len(index))[:, None], groups=groups)
        return cls.from_splits(
            index,
            list(indices_generator),
            split_labels=split_labels,
            set_labels=set_labels,
            **kwargs,
        )

    @classmethod
    def from_split_func(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split_func: tp.Callable,
        split_args: tp.ArgsLike = None,
        split_kwargs: tp.KwargsLike = None,
        fix_ranges: bool = True,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a custom split function.

        In a while-loop, substitutes templates in `split_args` and `split_kwargs` and passes
        them to `split_func`, which should return either a split (see `new_split` in `Splitter.split_range`,
        also supports a single range if it's not an iterable) or None to abrupt the while-loop.
        If `fix_ranges` is True, the returned split is then converted into a fixed split using
        `Splitter.split_range` and the bounds of its sets are measured using `Splitter.get_range_bounds`.

        Each template substitution has the following information:

        * `split_idx`: Current split index, starting at 0
        * `splits`: Nested list of splits appended up to this point
        * `bounds`: Nested list of bounds appended up to this point
        * `prev_start`: Left bound of the previous split
        * `prev_end`: Right bound of the previous split
        * Arguments and keyword arguments passed to `Splitter.from_split_func`

        Usage:
            * Rolling window of 30 elements, 20 for train and 10 for test:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> def split_func(splits, bounds, index):
            ...     if len(splits) == 0:
            ...         new_split = (slice(0, 20), slice(20, 30))
            ...     else:
            ...         # Previous split, first set, right bound
            ...         prev_end = bounds[-1][0][1]
            ...         new_split = (
            ...             slice(prev_end, prev_end + 20),
            ...             slice(prev_end + 20, prev_end + 30)
            ...         )
            ...     if new_split[-1].stop > len(index):
            ...         return None
            ...     return new_split

            >>> splitter = vbt.Splitter.from_split_func(
            ...     index,
            ...     split_func,
            ...     split_args=(
            ...         vbt.Rep("splits"),
            ...         vbt.Rep("bounds"),
            ...         vbt.Rep("index"),
            ...     ),
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_split_func.svg){: .iimg loading=lazy }
        """
        index = try_to_datetime_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        if split_args is None:
            split_args = ()
        if split_kwargs is None:
            split_kwargs = {}

        splits = []
        bounds = []
        split_idx = 0
        n_sets = None
        while True:
            _template_context = merge_dicts(
                dict(
                    split_idx=split_idx,
                    splits=splits,
                    bounds=bounds,
                    prev_start=bounds[-1][0][0] if len(bounds) > 0 else None,
                    prev_end=bounds[-1][-1][1] if len(bounds) > 0 else None,
                    index=index,
                    freq=freq,
                    fix_ranges=fix_ranges,
                    split_args=split_args,
                    split_kwargs=split_kwargs,
                    split_range_kwargs=split_range_kwargs,
                    range_bounds_kwargs=range_bounds_kwargs,
                    **kwargs,
                ),
                template_context,
            )
            _split_func = substitute_templates(split_func, _template_context, sub_id="split_func")
            _split_args = substitute_templates(split_args, _template_context, sub_id="split_args")
            _split_kwargs = substitute_templates(split_kwargs, _template_context, sub_id="split_kwargs")
            new_split = _split_func(*_split_args, **_split_kwargs)
            if new_split is None:
                break
            if not checks.is_iterable(new_split):
                new_split = (new_split,)
            if fix_ranges or split is not None:
                new_split = cls.split_range(
                    slice(None),
                    new_split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if split is not None:
                if len(new_split) > 1:
                    raise ValueError("Split function must return only one range if split is already provided")
                new_split = cls.split_range(
                    new_split[0],
                    split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if n_sets is None:
                n_sets = len(new_split)
            elif n_sets != len(new_split):
                raise ValueError("All splits must have the same number of sets")
            splits.append(new_split)
            if fix_ranges:
                split_bounds = tuple(
                    map(
                        lambda x: cls.get_range_bounds(
                            x,
                            template_context=_template_context,
                            index=index,
                            **range_bounds_kwargs,
                        ),
                        new_split,
                    )
                )
                bounds.append(split_bounds)
            split_idx += 1

        return cls.from_splits(
            index,
            splits,
            fix_ranges=fix_ranges,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Splitter` after stacking along rows."""
        if "splits_arr" not in kwargs:
            kwargs["splits_arr"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.splits for obj in objs],
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Splitter` after stacking along columns."""
        if "splits_arr" not in kwargs:
            kwargs["splits_arr"] = kwargs["wrapper"].column_stack_arrs(
                *[obj.splits for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs],
                stack_columns=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                union_index=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "index",
        "splits_arr",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        index: tp.Index,
        splits_arr: tp.SplitsArray,
        **kwargs,
    ) -> None:
        if wrapper.grouper.is_grouped():
            raise ValueError("Splitter cannot be grouped")
        index = try_to_datetime_index(index)
        if splits_arr.shape[0] != wrapper.shape_2d[0]:
            raise ValueError("Number of splits must match wrapper index")
        if splits_arr.shape[1] != wrapper.shape_2d[1]:
            raise ValueError("Number of sets must match wrapper columns")

        Analyzable.__init__(
            self,
            wrapper,
            index=index,
            splits_arr=splits_arr,
            **kwargs,
        )

        self._index = index
        self._splits_arr = splits_arr

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `Splitter` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        if wrapper_meta["rows_changed"] or wrapper_meta["columns_changed"]:
            new_splits_arr = ArrayWrapper.select_from_flex_array(
                self.splits_arr,
                row_idxs=wrapper_meta["row_idxs"],
                col_idxs=wrapper_meta["col_idxs"],
                rows_changed=wrapper_meta["rows_changed"],
                columns_changed=wrapper_meta["columns_changed"],
            )
        else:
            new_splits_arr = self.splits_arr
        return dict(
            wrapper_meta=wrapper_meta,
            new_splits_arr=new_splits_arr,
        )

    def indexing_func(self: SplitterT, *args, splitter_meta: tp.DictLike = None, **kwargs) -> SplitterT:
        """Perform indexing on `Splitter`."""
        if splitter_meta is None:
            splitter_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=splitter_meta["wrapper_meta"]["new_wrapper"],
            splits_arr=splitter_meta["new_splits_arr"],
        )

    @property
    def index(self) -> tp.Index:
        """Index."""
        return self._index

    @property
    def splits_arr(self) -> tp.SplitsArray:
        """Two-dimensional, object-dtype DataFrame with splits.

        First axis represents splits. Second axis represents sets. Elements represent ranges.
        Range must be either a slice, a sequence of indices, a mask, or a callable that returns such."""
        return self._splits_arr

    @property
    def splits(self) -> tp.Frame:
        """`Splitter.splits_arr` as a DataFrame."""
        return self.wrapper.wrap(self._splits_arr, group_by=False)

    @property
    def split_labels(self) -> tp.Index:
        """Split labels."""
        return self.wrapper.index

    @property
    def set_labels(self) -> tp.Index:
        """Set labels."""
        return self.wrapper.columns

    @property
    def n_splits(self) -> int:
        """Number of splits."""
        return self.splits_arr.shape[0]

    @property
    def n_sets(self) -> int:
        """Number of sets."""
        return self.splits_arr.shape[1]

    def get_split_grouper(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Get split grouper."""
        if split_group_by is None:
            return None
        if isinstance(split_group_by, Grouper):
            return split_group_by
        return BaseIDXAccessor(self.split_labels).get_grouper(split_group_by, def_lvl_name="split_group")

    def get_set_grouper(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Get set grouper."""
        if set_group_by is None:
            return None
        if isinstance(set_group_by, Grouper):
            return set_group_by
        return BaseIDXAccessor(self.set_labels).get_grouper(set_group_by, def_lvl_name="set_group")

    def get_n_splits(self, split_group_by: tp.AnyGroupByLike = None) -> int:
        """Get number of splits while considering the grouper."""
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_group_count()
        return self.n_splits

    def get_n_sets(self, set_group_by: tp.AnyGroupByLike = None) -> int:
        """Get number of sets while considering the grouper."""
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_group_count()
        return self.n_sets

    def get_split_labels(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Get split labels while considering the grouper."""
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_index()
        return self.split_labels

    def get_set_labels(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Get set labels while considering the grouper."""
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_index()
        return self.set_labels

    # ############# Conversion ############# #

    def to_fixed(self: SplitterT, split_range_kwargs: tp.KwargsLike = None, **kwargs) -> SplitterT:
        """Convert relative ranges into fixed ones and return a new `Splitter` instance.

        Keyword arguments `split_range_kwargs` are passed to `Splitter.split_range`."""
        if split_range_kwargs is None:
            split_range_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange
        new_splits_arr = []
        for split in self.splits_arr:
            new_split = self.split_range(slice(None), split, **split_range_kwargs)
            new_splits_arr.append(new_split)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)
        return self.replace(splits_arr=new_splits_arr, **kwargs)

    def to_grouped(
        self: SplitterT,
        split: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        set_: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        split_as_indices: bool = False,
        set_as_indices: bool = False,
        merge_split_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Merge all ranges within the same group and return a new `Splitter` instance."""
        if merge_split_kwargs is None:
            merge_split_kwargs = {}
        merge_split_kwargs = dict(merge_split_kwargs)
        wrap_with_fixrange = merge_split_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        merge_split_kwargs["wrap_with_fixrange"] = wrap_with_fixrange
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            split_as_indices=split_as_indices,
            set_as_indices=set_as_indices,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]

        new_splits_arr = []
        for i in split_group_indices:
            new_splits_arr.append([])
            for j in set_group_indices:
                new_range = self.select_range(
                    split=i,
                    set_=j,
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    split_as_indices=True,
                    set_as_indices=True,
                    merge_split_kwargs=merge_split_kwargs,
                )
                new_splits_arr[-1].append(new_range)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if set_group_by is None or not set_group_by.is_grouped():
            ndim = self.wrapper.ndim
        else:
            ndim = 1 if new_splits_arr.shape[1] == 1 else 2
        wrapper = self.wrapper.replace(index=split_labels, columns=set_labels, ndim=ndim)
        return self.replace(wrapper=wrapper, splits_arr=new_splits_arr, **kwargs)

    # ############# Ranges ############# #

    @classmethod
    def is_range_relative(cls, range_: tp.RangeLike) -> bool:
        """Return whether a range is relative."""
        return checks.is_number(range_) or checks.is_td_like(range_) or isinstance(range_, RelRange)

    @class_or_instancemethod
    def get_ready_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        allow_relative: bool = False,
        allow_zero_len: bool = False,
        range_format: str = "slice_or_any",
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        return_meta: bool = False,
    ) -> tp.Union[tp.RelRangeLike, tp.ReadyRangeLike, dict]:
        """Get a range that can be directly used in array indexing.

        Such a range is either an integer or datetime-like slice (right bound is always exclusive!),
        a one-dimensional NumPy array with integer indices or datetime-like objects,
        or a one-dimensional NumPy mask of the same length as the index.

        Argument `range_format` accepts the following options:

        * 'any': Return any format
        * 'indices': Return indices
        * 'mask': Return mask of the same length as index
        * 'slice': Return slice
        * 'slice_or_indices': If slice fails, return indices
        * 'slice_or_mask': If slice fails, return mask
        * 'slice_or_any': If slice fails, return any format
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        if range_format.lower() not in (
            "any",
            "indices",
            "mask",
            "slice",
            "slice_or_indices",
            "slice_or_mask",
            "slice_or_any",
        ):
            raise ValueError(f"Invalid option range_format='{range_format}'")

        meta = dict()
        meta["was_fixed"] = False
        meta["was_template"] = False
        meta["was_callable"] = False
        meta["was_relative"] = False
        meta["was_hslice"] = False
        meta["was_slice"] = False
        meta["was_neg_slice"] = False
        meta["was_datetime"] = False
        meta["was_mask"] = False
        meta["was_indices"] = False
        meta["is_constant"] = False
        meta["start"] = None
        meta["stop"] = None
        meta["length"] = None
        if isinstance(range_, FixRange):
            meta["was_fixed"] = True
            range_ = range_.range_
        if isinstance(range_, CustomTemplate):
            meta["was_template"] = True
            if template_context is None:
                template_context = {}
            if "index" not in template_context:
                template_context["index"] = index
            range_ = range_.substitute(context=template_context, sub_id="range")
        if callable(range_):
            meta["was_callable"] = True
            range_ = range_(index)
        if cls_or_self.is_range_relative(range_):
            meta["was_relative"] = True
            if allow_relative:
                if return_meta:
                    meta["range_"] = range_
                    return meta
                return range_
            raise TypeError("Relative ranges must be converted to fixed")
        if isinstance(range_, hslice):
            meta["was_hslice"] = True
            range_ = range_.to_slice()
        if isinstance(range_, slice):
            meta["was_slice"] = True
            meta["is_constant"] = True
            start = range_.start
            stop = range_.stop
            if range_.step is not None and range_.step > 1:
                raise ValueError("Step must be either None or 1")
            if start is not None and checks.is_int(start) and start < 0:
                if stop is not None and checks.is_int(stop) and stop > 0:
                    raise ValueError("Slices must be either strictly negative or positive")
                meta["was_neg_slice"] = True
                start = len(index) + start
                if stop is not None and checks.is_int(stop):
                    stop = len(index) + stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(index)
            if not checks.is_int(start):
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type pandas.DatetimeIndex, not {index.dtype}")
                start = try_align_dt_to_index(start, index)
                if not isinstance(start, pd.Timestamp):
                    raise ValueError(f"Range start ({start}) could not be parsed")
                meta["was_datetime"] = True
            if not checks.is_int(stop):
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type pandas.DatetimeIndex, not {index.dtype}")
                stop = try_align_dt_to_index(stop, index)
                if not isinstance(stop, pd.Timestamp):
                    raise ValueError(f"Range start ({stop}) could not be parsed")
                meta["was_datetime"] = True
            if checks.is_int(start):
                if start < 0:
                    start = 0
            else:
                if start < index[0]:
                    start = 0
                else:
                    start = index.get_indexer([start], method="bfill")[0]
                    if start == -1:
                        raise ValueError(f"Range start ({start}) is out of bounds")
            if checks.is_int(stop):
                if stop > len(index):
                    stop = len(index)
            else:
                if stop > index[-1]:
                    stop = len(index)
                else:
                    stop = index.get_indexer([stop], method="bfill")[0]
                    if stop == -1:
                        raise ValueError(f"Range stop ({stop}) is out of bounds")
            range_ = slice(start, stop)
            meta["start"] = start
            meta["stop"] = stop
            meta["length"] = stop - start
            if not allow_zero_len and meta["length"] == 0:
                raise ValueError("Range has zero length")
            if range_format.lower() == "indices":
                range_ = np.arange(*range_.indices(len(index)))
            elif range_format.lower() == "mask":
                mask = np.full(len(index), False)
                mask[range_] = True
                range_ = mask
        else:
            range_ = np.asarray(range_)
            if np.issubdtype(range_.dtype, np.bool_):
                if len(range_) != len(index):
                    raise ValueError("Mask must have the same length as index")
                meta["was_mask"] = True
                indices = np.flatnonzero(range_)
                if len(indices) == 0:
                    if not allow_zero_len:
                        raise ValueError("Range has zero length")
                    meta["is_constant"] = True
                    meta["start"] = 0
                    meta["stop"] = 0
                    meta["length"] = 0
                else:
                    meta["is_constant"] = is_range(indices)
                    meta["start"] = indices[0]
                    meta["stop"] = indices[-1] + 1
                    meta["length"] = len(indices)
                if range_format.lower() == "indices":
                    range_ = indices
                elif range_format.lower().startswith("slice"):
                    if not meta["is_constant"]:
                        if range_format.lower() == "slice":
                            raise ValueError("Cannot convert to slice: range is not constant")
                        if range_format.lower() == "slice_or_indices":
                            range_ = indices
                    else:
                        range_ = slice(meta["start"], meta["stop"])
            else:
                if not np.issubdtype(range_.dtype, np.integer):
                    range_ = try_align_to_dt_index(range_, index)
                    if not isinstance(range_, pd.DatetimeIndex):
                        raise ValueError("Range array could not be parsed")
                    range_ = index.get_indexer(range_, method=None)
                    if -1 in range_:
                        raise ValueError(f"Range array has values that cannot be found in index")
                if np.issubdtype(range_.dtype, np.integer):
                    meta["was_indices"] = True
                    if len(range_) == 0:
                        if not allow_zero_len:
                            raise ValueError("Range has zero length")
                        meta["is_constant"] = True
                        meta["start"] = 0
                        meta["stop"] = 0
                        meta["length"] = 0
                    else:
                        meta["is_constant"] = is_range(range_)
                        if meta["is_constant"]:
                            meta["start"] = range_[0]
                            meta["stop"] = range_[-1] + 1
                        else:
                            meta["start"] = np.min(range_)
                            meta["stop"] = np.max(range_) + 1
                        meta["length"] = len(range_)
                    if range_format.lower() == "mask":
                        mask = np.full(len(index), False)
                        mask[range_] = True
                        range_ = mask
                    elif range_format.lower().startswith("slice"):
                        if not meta["is_constant"]:
                            if range_format.lower() == "slice":
                                raise ValueError("Cannot convert to slice: range is not constant")
                            if range_format.lower() == "slice_or_mask":
                                mask = np.full(len(index), False)
                                mask[range_] = True
                                range_ = mask
                        else:
                            range_ = slice(meta["start"], meta["stop"])
                else:
                    raise TypeError(f"Range array has invalid data type ({range_.dtype})")
        if meta["start"] != meta["stop"]:
            if meta["start"] > meta["stop"]:
                raise ValueError(f"Range start ({meta['start']}) is higher than range stop ({meta['stop']})")
            if meta["start"] < 0 or meta["start"] >= len(index):
                raise ValueError(f"Range start ({meta['start']}) is out of bounds")
            if meta["stop"] < 0 or meta["stop"] > len(index):
                raise ValueError(f"Range stop ({meta['stop']}) is out of bounds")
        if return_meta:
            meta["range_"] = range_
            return meta
        return range_

    @class_or_instancemethod
    def split_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        new_split: tp.SplitLike,
        backwards: bool = False,
        allow_zero_len: bool = False,
        range_format: tp.Optional[str] = None,
        wrap_with_template: bool = False,
        wrap_with_fixrange: tp.Optional[bool] = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.FixSplit:
        """Split a fixed range into a split of multiple fixed ranges.

        Range must be either a template, a callable, a tuple (start and stop), a slice, a sequence
        of indices, or a mask. This range will then be re-mapped into the index.

        Each sub-range in `new_split` can be either a fixed or relative range, that is, an instance
        of `RelRange` or a number that will be used as a length to create an `RelRange`.
        Each sub-range will then be re-mapped into the main range. Argument `new_split` can also
        be provided as an integer or a float indicating the length; in such a case the second part
        (or the first one depending on `backwards`) will stretch. If `new_split` is a string,
        the following options are supported:

        * 'by_gap': Split `range_` by gap using `vectorbtpro.generic.splitting.nb.split_range_by_gap_nb`.

        New ranges are returned relative to the index and in the same order as passed.

        For `range_format`, see `Splitter.get_ready_range`. Enable `wrap_with_template` to wrap the
        resulting ranges with a template of the type `vectorbtpro.utils.template.Rep`."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)

        # Prepare source range
        range_meta = cls_or_self.get_ready_range(
            range_,
            allow_zero_len=allow_zero_len,
            range_format="slice_or_indices",
            template_context=template_context,
            index=index,
            return_meta=True,
        )
        range_ = range_meta["range_"]
        range_was_hslice = range_meta["was_hslice"]
        range_was_indices = range_meta["was_indices"]
        range_was_mask = range_meta["was_mask"]
        range_length = range_meta["length"]
        if range_format is None:
            if range_was_indices:
                range_format = "slice_or_indices"
            elif range_was_mask:
                range_format = "slice_or_mask"
            else:
                range_format = "slice_or_any"

        # Substitute template
        if isinstance(new_split, CustomTemplate):
            _template_context = merge_dicts(dict(index=index[range_]), template_context)
            new_split = substitute_templates(new_split, _template_context, sub_id="new_split")

        # Split by gap
        if isinstance(new_split, str) and new_split.lower() == "by_gap":
            if isinstance(range_, np.ndarray) and np.issubdtype(range_.dtype, np.integer):
                range_arr = range_
            else:
                range_arr = np.arange(len(index))[range_]
            start_idxs, stop_idxs = nb.split_range_by_gap_nb(range_arr)
            new_split = list(map(lambda x: slice(x[0], x[1]), zip(start_idxs, stop_idxs)))

        # Prepare target ranges
        if checks.is_number(new_split):
            if new_split < 0:
                backwards = not backwards
                new_split = abs(new_split)
            if not backwards:
                new_split = (new_split, 1.0)
            else:
                new_split = (1.0, new_split)
        elif checks.is_td_like(new_split):
            new_split = parse_timedelta(new_split)
            if new_split < pd.Timedelta(0):
                backwards = not backwards
                new_split = abs(new_split)
            if not backwards:
                new_split = (new_split, 1.0)
            else:
                new_split = (1.0, new_split)
        elif not checks.is_iterable(new_split):
            new_split = (new_split,)

        # Perform split
        new_ranges = []
        if backwards:
            new_split = new_split[::-1]
            prev_start = range_length
            prev_end = range_length
        else:
            prev_start = 0
            prev_end = 0
        for new_range in new_split:
            # Resolve new range
            new_range_meta = cls_or_self.get_ready_range(
                new_range,
                allow_relative=True,
                allow_zero_len=allow_zero_len,
                range_format="slice_or_any",
                template_context=template_context,
                index=index[range_],
                return_meta=True,
            )
            new_range = new_range_meta["range_"]
            if checks.is_number(new_range) or checks.is_td_like(new_range):
                new_range = RelRange(length=new_range)
            if isinstance(new_range, RelRange):
                new_range_is_gap = new_range.is_gap
                new_range = new_range.to_slice(
                    range_length,
                    prev_start=range_length - prev_end if backwards else prev_start,
                    prev_end=range_length - prev_start if backwards else prev_end,
                    index=index,
                    freq=freq,
                )
                if backwards:
                    new_range = slice(range_length - new_range.stop, range_length - new_range.start)
            else:
                new_range_is_gap = False

            # Update previous bounds
            if isinstance(new_range, slice):
                prev_start = new_range.start
                prev_end = new_range.stop
            else:
                prev_start = new_range_meta["start"]
                prev_end = new_range_meta["stop"]

            # Remap new range to index
            if new_range_is_gap:
                continue
            if isinstance(range_, slice) and isinstance(new_range, slice):
                new_range = slice(
                    range_.start + new_range.start,
                    range_.start + new_range.stop,
                )
            else:
                if isinstance(range_, slice):
                    new_range = np.arange(range_.start, range_.stop)[new_range]
                else:
                    new_range = range_[new_range]
            new_range = cls_or_self.get_ready_range(
                new_range,
                allow_zero_len=allow_zero_len,
                range_format=range_format,
                index=index,
            )
            if isinstance(new_range, slice) and range_was_hslice:
                new_range = hslice.from_slice(new_range)
            if wrap_with_template:
                new_range = Rep("range_", context=dict(range_=new_range))
            if wrap_with_fixrange is None:
                _wrap_with_fixrange = checks.is_sequence(new_range)
            else:
                _wrap_with_fixrange = False
            if _wrap_with_fixrange:
                new_range = FixRange(new_range)
            new_ranges.append(new_range)

        if backwards:
            return tuple(new_ranges)[::-1]
        return tuple(new_ranges)

    @class_or_instancemethod
    def merge_split(
        cls_or_self,
        split: tp.FixSplit,
        range_format: tp.Optional[str] = None,
        wrap_with_template: bool = False,
        wrap_with_fixrange: tp.Optional[bool] = False,
        wrap_with_hslice: tp.Optional[bool] = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.FixRangeLike:
        """Merge a split of multiple fixed ranges into a fixed range.

        Creates one mask and sets True for each range. If all input ranges are masks,
        returns that mask. If all input ranges are slices, returns a slice if possible.
        Otherwise, returns integer indices.

        For `range_format`, see `Splitter.get_ready_range`. Enable `wrap_with_template` to wrap the
        resulting range with a template of the type `vectorbtpro.utils.template.Rep`."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        all_hslices = True
        all_masks = True
        new_ranges = []
        if len(split) == 1:
            raise ValueError("Two or more ranges are required to be merged")
        for range_ in split:
            range_meta = cls_or_self.get_ready_range(
                range_,
                allow_zero_len=True,
                range_format="any",
                template_context=template_context,
                index=index,
                return_meta=True,
            )
            if not range_meta["was_hslice"]:
                all_hslices = False
            if not range_meta["was_mask"]:
                all_masks = False
            new_ranges.append(range_meta["range_"])
        ranges = new_ranges
        if range_format is None:
            if all_masks:
                range_format = "slice_or_mask"
            else:
                range_format = "slice_or_indices"

        new_range = np.full(len(index), False)
        for range_ in ranges:
            new_range[range_] = True
        new_range = cls_or_self.get_ready_range(
            new_range,
            range_format=range_format,
            index=index,
        )
        if isinstance(new_range, slice) and all_hslices:
            if wrap_with_hslice is None:
                wrap_with_hslice = True
            if wrap_with_hslice:
                new_range = hslice.from_slice(new_range)
        if wrap_with_template:
            new_range = Rep("range_", context=dict(range_=new_range))
        if wrap_with_fixrange is None:
            _wrap_with_fixrange = checks.is_sequence(new_range)
        else:
            _wrap_with_fixrange = False
        if _wrap_with_fixrange:
            new_range = FixRange(new_range)
        return new_range

    # ############# Taking ############# #

    def select_indices(
        self,
        split: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        set_: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        split_as_indices: bool = False,
        set_as_indices: bool = False,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
        """Get indices corresponding to selected splits and sets.

        Arguments `split` and `set_` can be either integers and labels. Also, multiple
        values are accepted; in such a case, the corresponding ranges are merged.
        If split/set labels are of the integer data type, treats the provided values as labels
        rather than indices, unless `split_as_indices`/`set_as_indices` is enabled.

        If `split_group_by` and/or `set_group_by` are provided, their groupers get
        created using `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper` and
        arguments `split` and `set_` become relative to the groups.

        If `split`/`set_` is not provided, selects all indices.

        Returns four arrays: split group indices, set group indices, split indices, and set indices."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        if split is None:
            split_group_indices = np.arange(self.get_n_splits(split_group_by=split_group_by))
            split_indices = np.arange(self.n_splits)
        else:
            if checks.is_hashable(split):
                split = [split]
            if split_group_by is not None:
                split_group_indices = []
                groups, group_index = split_group_by.get_groups_and_index()
                mask = None
                for g in split:
                    if checks.is_int(g) and (split_as_indices or not group_index.is_integer()):
                        i = g
                    else:
                        i = group_index.get_indexer([g])[0]
                        if i == -1:
                            raise ValueError(f"Split group '{g}' not found")
                    if mask is None:
                        mask = groups == i
                    else:
                        mask |= groups == i
                    split_group_indices.append(i)
                split_group_indices = np.asarray(split_group_indices)
                split_indices = np.arange(self.n_splits)[mask]
            else:
                split_indices = []
                for s in split:
                    if checks.is_int(s) and (split_as_indices or not self.split_labels.is_integer()):
                        i = s
                    else:
                        i = self.split_labels.get_indexer([s])[0]
                        if i == -1:
                            raise ValueError(f"Split '{s}' not found")
                    split_indices.append(i)
                split_group_indices = split_indices = np.asarray(split_indices)
        if set_ is None:
            set_group_indices = np.arange(self.get_n_sets(set_group_by=set_group_by))
            set_indices = np.arange(self.n_sets)
        else:
            if checks.is_hashable(set_):
                set_ = [set_]
            if set_group_by is not None:
                set_group_indices = []
                groups, group_index = set_group_by.get_groups_and_index()
                mask = None
                for g in set_:
                    if checks.is_int(g) and (set_as_indices or not group_index.is_integer()):
                        i = g
                    else:
                        i = group_index.get_indexer([g])[0]
                        if i == -1:
                            raise ValueError(f"Set group '{g}' not found")
                    if mask is None:
                        mask = groups == i
                    else:
                        mask |= groups == i
                    set_group_indices.append(i)
                set_group_indices = np.asarray(set_group_indices)
                set_indices = np.arange(self.n_sets)[mask]
            else:
                set_indices = []
                for s in set_:
                    if checks.is_int(s) and (set_as_indices or not self.set_labels.is_integer()):
                        i = s
                    else:
                        i = self.set_labels.get_indexer([s])[0]
                        if i == -1:
                            raise ValueError(f"Set '{s}' not found")
                    set_indices.append(i)
                set_group_indices = set_indices = np.asarray(set_indices)
        return split_group_indices, set_group_indices, split_indices, set_indices

    def select_range(self, merge_split_kwargs: tp.KwargsLike = None, **select_indices_kwargs) -> tp.RangeLike:
        """Select a range.

        Passes `**select_indices_kwargs` to `Splitter.select_indices` to get the indices for selected
        splits and sets. If multiple ranges correspond to those indices, merges them using
        `Splitter.merge_split`."""
        _, _, split_indices, set_indices = self.select_indices(**select_indices_kwargs)
        ranges = []
        for i in split_indices:
            for j in set_indices:
                ranges.append(self.splits_arr[i, j])
        if len(ranges) == 1:
            return ranges[0]
        if merge_split_kwargs is None:
            merge_split_kwargs = {}
        return self.merge_split(ranges, **merge_split_kwargs)

    @class_or_instancemethod
    def remap_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        target_index: tp.IndexLike,
        target_freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.FixRangeLike:
        """Remap a range to a target index.

        If `index` and `target_index` are the same, returns the range. Otherwise,
        uses `vectorbtpro.base.resampling.base.Resampler.resample_source_mask` to resample
        the range into the target index. In such a case, `freq` and `target_freq` must be provided."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        if target_index is None:
            raise ValueError("Must provide target index")
        target_index = try_to_datetime_index(target_index)
        if index.equals(target_index):
            return range_

        mask = cls_or_self.get_range_mask(range_, template_context=template_context, index=index)
        resampler = Resampler(
            source_index=index,
            target_index=target_index,
            source_freq=freq,
            target_freq=target_freq,
        )
        target_mask = resampler.resample_source_mask(mask, jitted=jitted, silence_warnings=silence_warnings)
        return target_mask

    @classmethod
    def get_obj_index(cls, obj: tp.Any) -> tp.Index:
        """Get index from an object."""
        if isinstance(obj, pd.Index):
            return obj
        if hasattr(obj, "index"):
            return obj.index
        if hasattr(obj, "wrapper"):
            return obj.wrapper.index
        raise ValueError("Must provide object index")

    @class_or_instancemethod
    def get_ready_obj_range(
        cls_or_self,
        obj: tp.Any,
        range_: tp.FixRangeLike,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        return_obj_meta: bool = False,
        **ready_range_kwargs,
    ) -> tp.Any:
        """Get a range that is ready to be mapped into an array-like object.

        If the object is Pandas-like and `obj_index` is not None, searches for an index in the object
        using `Splitter.get_obj_index`. Once found, uses `Splitter.remap_range` to get the range
        that maps to the object index. Finally, uses `Splitter.get_ready_range` to convert the range
        into the one that can be used directly in indexing."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        if remap_to_obj and (
            isinstance(obj, (pd.Index, pd.Series, pd.DataFrame, PandasIndexer)) or obj_index is not None
        ):
            if obj_index is None:
                obj_index = cls_or_self.get_obj_index(obj)
            target_range = cls_or_self.remap_range(
                range_,
                target_index=obj_index,
                target_freq=obj_freq,
                template_context=template_context,
                jitted=jitted,
                silence_warnings=silence_warnings,
                index=index,
                freq=freq,
            )
        else:
            obj_index = index
            obj_freq = freq
            target_range = range_
        ready_range_or_meta = cls_or_self.get_ready_range(
            target_range,
            template_context=template_context,
            index=obj_index,
            **ready_range_kwargs,
        )
        if return_obj_meta:
            obj_meta = dict(index=obj_index, freq=obj_freq)
            return obj_meta, ready_range_or_meta
        return ready_range_or_meta

    @classmethod
    def take_range(cls, obj: tp.Any, ready_range: tp.ReadyRangeLike, point_wise: bool = False) -> tp.Any:
        """Take a ready range from an array-like object.

        Set `point_wise` to True to select one range point at a time and return a tuple."""
        if isinstance(obj, (pd.Series, pd.DataFrame, PandasIndexer)):
            if point_wise:
                return tuple(obj.iloc[i] for i in np.arange(len(obj))[ready_range])
            return obj.iloc[ready_range]
        if point_wise:
            return tuple(obj[i] for i in np.arange(len(obj))[ready_range])
        return obj[ready_range]

    def take(
        self,
        obj: tp.Any,
        split: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        set_: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        split_as_indices: bool = False,
        set_as_indices: bool = False,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        into: tp.Optional[str] = None,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        range_format: str = "slice_or_any",
        point_wise: bool = False,
        attach_bounds: tp.Union[bool, str] = False,
        right_inclusive: bool = False,
        template_context: tp.KwargsLike = None,
        silence_warnings: bool = False,
        index_combine_kwargs: tp.KwargsLike = None,
        stack_axis: int = 1,
        stack_kwargs: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Any:
        """Take all ranges from an array-like object and optionally column-stack them.

        Uses `Splitter.select_indices` to get the indices for selected splits and sets.
        Arguments `split_group_by` and `set_group_by` can be used to group splits and sets respectively.
        Ranges belonging to the same split and set group will be merged.

        For each index pair, resolves the source range using `Splitter.select_range` and
        `Splitter.get_ready_range`. Then, remaps this range into the object index using
        `Splitter.get_ready_obj_range` and takes the slice from the object using `Splitter.take_range`.
        If the object is a custom template, substitutes its instead of calling `Splitter.take_range`.
        Finally, uses `vectorbtpro.base.merging.column_stack_merge` (`stack_axis=1`) or
        `vectorbtpro.base.merging.row_stack_merge` (`stack_axis=0`) with `stack_kwargs` to merge the taken slices.

        If `attach_bounds` is enabled, measures the bounds of each range and makes it an additional
        level in the final index hierarchy. The argument supports the following options:

        * True, 'index', 'source', or 'source_index': Attach source (index) bounds
        * 'target' or 'target_index': Attach target (index) bounds
        * False: Do not attach

        Argument `into` supports the following options:

        * None: Series of range slices
        * 'stacked': Stack all slices into a single object
        * 'stacked_by_split': Stack set slices in each split and return a Series of objects
        * 'stacked_by_set': Stack split slices in each set and return a Series of objects
        * 'split_major_meta': Generator with ranges processed lazily in split-major order.
            Returns meta with indices and labels, and the generator.
        * 'set_major_meta': Generator with ranges processed lazily in set-major order.
            Returns meta with indices and labels, and the generator.

        Prepend any stacked option with "from_start_" (also "reset_") or "from_end_" to reset the index
        from start and from end respectively.

        Usage:
            * Roll a window and stack it along columns by keeping the index:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np
            >>> import pandas as pd

            >>> data = vbt.YFData.fetch(
            ...     "BTC-USD",
            ...     start="2020-01-01 UTC",
            ...     end="2021-01-01 UTC"
            ... )
            >>> splitter = vbt.Splitter.from_n_rolling(
            ...     data.wrapper.index,
            ...     3,
            ...     length=5
            ... )
            >>> splitter.take(data.close, into="stacked")
            split                                0            1             2
            Date
            2020-01-01 00:00:00+00:00  7200.174316          NaN           NaN
            2020-01-02 00:00:00+00:00  6985.470215          NaN           NaN
            2020-01-03 00:00:00+00:00  7344.884277          NaN           NaN
            2020-01-04 00:00:00+00:00  7410.656738          NaN           NaN
            2020-01-05 00:00:00+00:00  7411.317383          NaN           NaN
            2020-06-29 00:00:00+00:00          NaN  9190.854492           NaN
            2020-06-30 00:00:00+00:00          NaN  9137.993164           NaN
            2020-07-01 00:00:00+00:00          NaN  9228.325195           NaN
            2020-07-02 00:00:00+00:00          NaN  9123.410156           NaN
            2020-07-03 00:00:00+00:00          NaN  9087.303711           NaN
            2020-12-27 00:00:00+00:00          NaN          NaN  26272.294922
            2020-12-28 00:00:00+00:00          NaN          NaN  27084.808594
            2020-12-29 00:00:00+00:00          NaN          NaN  27362.437500
            2020-12-30 00:00:00+00:00          NaN          NaN  28840.953125
            2020-12-31 00:00:00+00:00          NaN          NaN  29001.720703
            ```

            * Disgard the index and attach index bounds to the column hierarchy:

            ```pycon
            >>> splitter.take(
            ...     data.close,
            ...     into="reset_stacked",
            ...     attach_bounds="index"
            ... )
            split                         0                         1  \\
            start 2020-01-01 00:00:00+00:00 2020-06-29 00:00:00+00:00
            end   2020-01-06 00:00:00+00:00 2020-07-04 00:00:00+00:00
            0                   7200.174316               9190.854492
            1                   6985.470215               9137.993164
            2                   7344.884277               9228.325195
            3                   7410.656738               9123.410156
            4                   7411.317383               9087.303711

            split                         2
            start 2020-12-27 00:00:00+00:00
            end   2021-01-01 00:00:00+00:00
            0                  26272.294922
            1                  27084.808594
            2                  27362.437500
            3                  28840.953125
            4                  29001.720703
            ```
        """
        if isinstance(attach_bounds, bool):
            if attach_bounds:
                attach_bounds = "source"
            else:
                attach_bounds = None
        index_bounds = False
        if attach_bounds is not None:
            if attach_bounds.lower() == "index":
                attach_bounds = "source"
                index_bounds = True
            if attach_bounds.lower() in ("source_index", "target_index"):
                attach_bounds = attach_bounds.split("_")[0]
                index_bounds = True
            if attach_bounds.lower() not in ("source", "target"):
                raise ValueError(f"Invalid option attach_bounds='{attach_bounds}'")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        if stack_axis not in (0, 1):
            raise ValueError("Axis for stacking must be either 0 or 1")
        if stack_kwargs is None:
            stack_kwargs = {}

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            split_as_indices=split_as_indices,
            set_as_indices=set_as_indices,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]
        n_splits = len(split_group_indices)
        n_sets = len(set_group_indices)
        one_split = n_splits == 1 and squeeze_one_split
        one_set = n_sets == 1 and squeeze_one_set
        one_range = one_split and one_set

        def _get_bounds(range_meta, obj_meta, obj_range_meta):
            if attach_bounds is not None:
                if attach_bounds.lower() == "source":
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            range_meta["start"],
                            range_meta["stop"],
                            right_inclusive=right_inclusive,
                            freq=freq,
                        )
                    else:
                        if right_inclusive:
                            bounds = (range_meta["start"], range_meta["stop"] - 1)
                        else:
                            bounds = (range_meta["start"], range_meta["stop"])
                else:
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            obj_range_meta["start"],
                            obj_range_meta["stop"],
                            right_inclusive=right_inclusive,
                            index=obj_meta["index"],
                            freq=obj_meta["freq"],
                        )
                    else:
                        if right_inclusive:
                            bounds = (obj_range_meta["start"], obj_range_meta["stop"] - 1)
                        else:
                            bounds = (obj_range_meta["start"], obj_range_meta["stop"])
            else:
                bounds = (None, None)
            return bounds

        def _get_range_meta(i, j):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            range_ = self.select_range(
                split=split_idx,
                set_=set_idx,
                split_group_by=split_group_by,
                set_group_by=set_group_by,
                split_as_indices=True,
                set_as_indices=True,
                merge_split_kwargs=dict(template_context=template_context),
            )
            range_meta = self.get_ready_range(
                range_,
                range_format=range_format,
                template_context=template_context,
                return_meta=True,
            )
            obj_meta, obj_range_meta = self.get_ready_obj_range(
                obj,
                range_meta["range_"],
                remap_to_obj=remap_to_obj,
                obj_index=obj_index,
                obj_freq=obj_freq,
                range_format=range_format,
                template_context=template_context,
                silence_warnings=silence_warnings,
                freq=freq,
                return_obj_meta=True,
                return_meta=True,
            )
            if isinstance(obj, CustomTemplate):
                _template_context = merge_dicts(
                    dict(
                        split_idx=split_idx,
                        set_idx=set_idx,
                        range_=obj_range_meta["range_"],
                        range_meta=obj_range_meta,
                        point_wise=point_wise,
                    ),
                    template_context,
                )
                obj_slice = substitute_templates(obj, _template_context, sub_id="take_range")
            else:
                obj_slice = self.take_range(obj, obj_range_meta["range_"], point_wise=point_wise)
            bounds = _get_bounds(range_meta, obj_meta, obj_range_meta)
            return dict(
                split_idx=split_idx,
                set_idx=set_idx,
                range_meta=range_meta,
                obj_range_meta=obj_range_meta,
                obj_slice=obj_slice,
                bounds=bounds,
            )

        def _attach_bounds(keys, range_bounds):
            range_bounds = pd.MultiIndex.from_tuples(range_bounds, names=["start", "end"])
            if keys is None:
                return range_bounds
            index_stack_kwargs = dict(index_combine_kwargs)
            index_stack_kwargs.pop("ignore_ranges", None)
            return stack_indexes((keys, range_bounds), **index_stack_kwargs)

        if into is None:
            range_objs = []
            range_bounds = []
            for i in range(n_splits):
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
            if one_range:
                return range_objs[0]
            if one_set:
                keys = split_labels
            elif one_split:
                keys = set_labels
            else:
                keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
            if attach_bounds is not None:
                keys = _attach_bounds(keys, range_bounds)
            return pd.Series(range_objs, index=keys, dtype=object)
        if isinstance(into, str) and into.lower().startswith("reset_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_start"
            into = into.lower().replace("reset_", "")
        if isinstance(into, str) and into.lower().startswith("from_start_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_start"
            into = into.lower().replace("from_start_", "")
        if isinstance(into, str) and into.lower().startswith("from_end_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_end"
            into = into.lower().replace("from_end_", "")
        if isinstance(into, str) and into.lower() in ("split_major_meta", "set_major_meta"):
            meta = {
                "split_group_indices": split_group_indices,
                "set_group_indices": set_group_indices,
                "split_indices": split_indices,
                "set_indices": set_indices,
                "n_splits": n_splits,
                "n_sets": n_sets,
                "split_labels": split_labels,
                "set_labels": set_labels,
            }
            if isinstance(into, str) and into.lower() == "split_major_meta":

                def _get_generator():
                    for i in range(n_splits):
                        for j in range(n_sets):
                            yield _get_range_meta(i, j)

                return meta, _get_generator()
            if isinstance(into, str) and into.lower() == "set_major_meta":

                def _get_generator():
                    for j in range(n_sets):
                        for i in range(n_splits):
                            yield _get_range_meta(i, j)

                return meta, _get_generator()
        if isinstance(into, str) and into.lower() == "stacked":
            range_objs = []
            range_bounds = []
            for i in range(n_splits):
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
            if one_range:
                return range_objs[0]
            if one_set:
                keys = split_labels
            elif one_split:
                keys = set_labels
            else:
                keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
            if attach_bounds is not None:
                keys = _attach_bounds(keys, range_bounds)
            _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
            if stack_axis == 0:
                return row_stack_merge(range_objs, **_stack_kwargs)
            return column_stack_merge(range_objs, **_stack_kwargs)
        if isinstance(into, str) and into.lower() == "stacked_by_split":
            new_split_objs = []
            one_set_bounds = []
            for i in range(n_splits):
                range_objs = []
                range_bounds = []
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
                if one_set and squeeze_one_set:
                    new_split_objs.append(range_objs[0])
                    one_set_bounds.append(range_bounds[0])
                else:
                    keys = set_labels
                    if attach_bounds is not None:
                        keys = _attach_bounds(keys, range_bounds)
                    _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
                    if stack_axis == 0:
                        new_split_objs.append(row_stack_merge(range_objs, **_stack_kwargs))
                    else:
                        new_split_objs.append(column_stack_merge(range_objs, **_stack_kwargs))
            if one_split and squeeze_one_split:
                return new_split_objs[0]
            if one_set and squeeze_one_set:
                if attach_bounds is not None:
                    return pd.Series(new_split_objs, index=_attach_bounds(split_labels, one_set_bounds), dtype=object)
            return pd.Series(new_split_objs, index=split_labels, dtype=object)
        if isinstance(into, str) and into.lower() == "stacked_by_set":
            new_set_objs = []
            one_split_bounds = []
            for j in range(n_sets):
                range_objs = []
                range_bounds = []
                for i in range(n_splits):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
                if one_split and squeeze_one_split:
                    new_set_objs.append(range_objs[0])
                    one_split_bounds.append(range_bounds[0])
                else:
                    keys = split_labels
                    if attach_bounds:
                        keys = _attach_bounds(keys, range_bounds)
                    _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
                    if stack_axis == 0:
                        new_set_objs.append(row_stack_merge(range_objs, **_stack_kwargs))
                    else:
                        new_set_objs.append(column_stack_merge(range_objs, **_stack_kwargs))
            if one_set and squeeze_one_set:
                return new_set_objs[0]
            if one_split and squeeze_one_split:
                if attach_bounds is not None:
                    return pd.Series(new_set_objs, index=_attach_bounds(set_labels, one_split_bounds), dtype=object)
            return pd.Series(new_set_objs, index=set_labels, dtype=object)
        raise ValueError(f"Invalid option into='{into}'")

    # ############# Applying ############# #

    def apply(
        self,
        apply_func: tp.Callable,
        *apply_args,
        split: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        set_: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        split_as_indices: bool = False,
        set_as_indices: bool = False,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        range_format: str = "slice_or_any",
        point_wise: bool = False,
        attach_bounds: tp.Union[bool, str] = False,
        right_inclusive: bool = False,
        template_context: tp.KwargsLike = None,
        silence_warnings: bool = False,
        index_combine_kwargs: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        iteration: str = "split_wise",
        execute_kwargs: tp.KwargsLike = None,
        merge_func: tp.Union[None, str, tuple, tp.Callable] = None,
        merge_kwargs: tp.KwargsLike = None,
        merge_all: bool = True,
        wrap_results: bool = True,
        **apply_kwargs,
    ) -> tp.Any:
        """Apply a function on each range.

        Uses `Splitter.select_indices` to get the indices for selected splits and sets.
        Arguments `split_group_by` and `set_group_by` can be used to group splits and sets respectively.
        Ranges belonging to the same split and set group will be merged.

        For each index pair, in a lazily manner, resolves the source range using `Splitter.select_range`
        and `Splitter.get_ready_range`. Then, takes each argument from `args` and `kwargs`
        wrapped with `Takeable`, remaps the range into each object's index using `Splitter.get_ready_obj_range`,
        and takes the slice from that object using `Splitter.take_range`. The original object will
        be substituted by this slice. At the end, substitutes any templates in the prepared
        `args` and `kwargs` and saves the function and arguments for execution.

        For substitution, the following information is available:

        * `split/set_group_indices`: Indices corresponding to the selected row/column groups
        * `split/set_indices`: Indices corresponding to the selected rows/columns
        * `n_splits/sets`: Number of the selected rows/columns
        * `split/set_labels`: Labels corresponding to the selected row/column groups
        * `split/set_idx`: Index of the selected row/column
        * `split/set_label`: Label of the selected row/column
        * `range_`: Selected range ready for indexing (see `Splitter.get_ready_range`)
        * `range_meta`: Various information on the selected range
        * `obj_range_meta`: Various information on the range taken from each takeable argument.
            Positional arguments are denoted by position, keyword arguments are denoted by keys.
        * `args`: Positional arguments with ranges already selected
        * `kwargs`: Keyword arguments with ranges already selected
        * `bounds`: A tuple of either integer or index bounds. Can be source or target depending on `attach_bounds`.
        * `template_context`: Passed template context

        Since each range is processed lazily (that is, upon request), there are multiple iteration
        modes controlled by the argument `iteration`:

        * 'split_major': Flatten all ranges in split-major order and iterate over them
        * 'set_major': Flatten all ranges in set-major order and iterate over them
        * 'split_wise': Iterate over splits, while ranges in each split are processed sequentially
        * 'set_wise': Iterate over sets, while ranges in each set are processed sequentially

        The execution is done using `vectorbtpro.utils.execution.execute` with `execute_kwargs`.
        Once all results have been obtained, attempts to merge them using `merge_func` with `merge_kwargs`
        (all templates in it will be substituted as well), which can also be a string or a tuple of
        strings resolved using `vectorbtpro.base.merging.resolve_merge_func`. If `wrap_results` is enabled,
        packs the results into a Pandas object. If `apply_func` returns something complex, the resulting
        Pandas object will be of object data type. If `apply_func` returns a tuple (detected by the first
        returned result), a Pandas object is built for each element of that tuple.

        If `merge_all` is True, will merge all results in a flattened manner irrespective of the
        iteration mode. Otherwise, will merge by split/set.

        Usage:
            * Get the return of each data range:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np
            >>> import pandas as pd

            >>> data = vbt.YFData.fetch(
            ...     "BTC-USD",
            ...     start="2020-01-01 UTC",
            ...     end="2021-01-01 UTC"
            ... )
            >>> splitter = vbt.Splitter.from_n_rolling(data.wrapper.index, 5)

            >>> def apply_func(data):
            ...     return data.close.iloc[-1] - data.close.iloc[0]

            >>> splitter.apply(apply_func, vbt.Takeable(data))
            split
            0    -1636.467285
            1     3706.568359
            2     2944.720703
            3     -118.113281
            4    17098.916016
            dtype: float64
            ```

            * The same but by indexing manually:

            ```pycon
            >>> def apply_func(range_, data):
            ...     data = data.iloc[range_]
            ...     return data.close.iloc[-1] - data.close.iloc[0]

            >>> splitter.apply(apply_func, vbt.Rep("range_"), data)
            split
            0    -1636.467285
            1     3706.568359
            2     2944.720703
            3     -118.113281
            4    17098.916016
            dtype: float64
            ```

            * Divide into two windows, each consisting of 50% train and 50% test, compute SMA for
            each range, and row-stack the outputs of each set upon merging:

            ```pycon
            >>> splitter = vbt.Splitter.from_n_rolling(data.wrapper.index, 2, split=0.5)

            >>> def apply_func(data):
            ...     return data.run("SMA", 10).real

            >>> splitter.apply(
            ...     apply_func,
            ...     vbt.Takeable(data),
            ...     iteration="set_wise",
            ...     merge_func="row_stack",
            ...     merge_all=False,
            ... ).T.vbt.drop_levels("split", axis=0).vbt.plot().show()
            ```

            ![](/assets/images/api/Splitter_apply.svg){: .iimg loading=lazy }
        """
        if isinstance(attach_bounds, bool):
            if attach_bounds:
                attach_bounds = "source"
            else:
                attach_bounds = None
        index_bounds = False
        if attach_bounds is not None:
            if attach_bounds.lower() == "index":
                attach_bounds = "source"
                index_bounds = True
            if attach_bounds.lower() in ("source_index", "target_index"):
                attach_bounds = attach_bounds.split("_")[0]
                index_bounds = True
            if attach_bounds.lower() not in ("source", "target"):
                raise ValueError(f"Invalid option attach_bounds='{attach_bounds}'")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        if execute_kwargs is None:
            execute_kwargs = {}
        if merge_kwargs is None:
            merge_kwargs = {}

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            split_as_indices=split_as_indices,
            set_as_indices=set_as_indices,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]
        n_splits = len(split_group_indices)
        n_sets = len(set_group_indices)
        one_split = n_splits == 1 and squeeze_one_split
        one_set = n_sets == 1 and squeeze_one_set
        one_range = one_split and one_set
        template_context = merge_dicts(
            {
                "splitter": self,
                "index": self.index,
                "split_group_indices": split_group_indices,
                "set_group_indices": set_group_indices,
                "split_indices": split_indices,
                "set_indices": set_indices,
                "n_splits": n_splits,
                "n_sets": n_sets,
                "split_labels": split_labels,
                "set_labels": set_labels,
                "one_split": one_split,
                "one_set": one_set,
                "one_range": one_range,
            },
            template_context,
        )

        def _get_range_meta(i, j, _template_context):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            range_ = self.select_range(
                split=split_idx,
                set_=set_idx,
                split_group_by=split_group_by,
                set_group_by=set_group_by,
                split_as_indices=True,
                set_as_indices=True,
                merge_split_kwargs=dict(template_context=_template_context),
            )
            range_meta = self.get_ready_range(
                range_,
                range_format=range_format,
                template_context=_template_context,
                return_meta=True,
            )
            return range_meta

        def _take_range(takeable, range_, _template_context):
            obj_meta, obj_range_meta = self.get_ready_obj_range(
                takeable.obj,
                range_,
                remap_to_obj=takeable.remap_to_obj if takeable.remap_to_obj is not _DEF else remap_to_obj,
                obj_index=takeable.index if takeable.index is not _DEF else obj_index,
                obj_freq=takeable.freq if takeable.freq is not _DEF else obj_freq,
                range_format=range_format,
                template_context=_template_context,
                silence_warnings=silence_warnings,
                freq=freq,
                return_obj_meta=True,
                return_meta=True,
            )
            if isinstance(takeable.obj, CustomTemplate):
                _template_context = merge_dicts(
                    dict(
                        range_=obj_range_meta["range_"],
                        range_meta=obj_range_meta,
                        point_wise=takeable.point_wise if takeable.point_wise is not _DEF else point_wise,
                    ),
                    _template_context,
                )
                obj_slice = substitute_templates(takeable.obj, _template_context, sub_id="take_range")
            else:
                obj_slice = self.take_range(
                    takeable.obj,
                    obj_range_meta["range_"],
                    point_wise=takeable.point_wise if takeable.point_wise is not _DEF else point_wise,
                )
            return obj_meta, obj_range_meta, obj_slice

        def _take_args(args, range_, _template_context):
            obj_meta = {}
            obj_range_meta = {}
            new_args = ()
            if args is not None:
                for i, arg in enumerate(args):
                    if isinstance(arg, Takeable):
                        _obj_meta, _obj_range_meta, obj_slice = _take_range(arg, range_, _template_context)
                        new_args += (obj_slice,)
                        obj_meta[i] = _obj_meta
                        obj_range_meta[i] = _obj_range_meta
                    else:
                        new_args += (arg,)
            return obj_meta, obj_range_meta, new_args

        def _take_kwargs(kwargs, range_, _template_context):
            obj_meta = {}
            obj_range_meta = {}
            new_kwargs = {}
            if kwargs is not None:
                for k, v in kwargs.items():
                    if isinstance(v, Takeable):
                        _obj_meta, _obj_range_meta, obj_slice = _take_range(v, range_, _template_context)
                        new_kwargs[k] = obj_slice
                        obj_meta[k] = _obj_meta
                        obj_range_meta[k] = _obj_range_meta
                    else:
                        new_kwargs[k] = v
            return obj_meta, obj_range_meta, new_kwargs

        def _get_bounds(range_meta, _template_context):
            if attach_bounds is not None:
                if isinstance(attach_bounds, str) and attach_bounds.lower() == "source":
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            range_meta["start"],
                            range_meta["stop"],
                            right_inclusive=right_inclusive,
                            freq=freq,
                        )
                    else:
                        if right_inclusive:
                            bounds = (range_meta["start"], range_meta["stop"] - 1)
                        else:
                            bounds = (range_meta["start"], range_meta["stop"])
                else:
                    obj_meta, obj_range_meta = self.get_ready_obj_range(
                        self.index,
                        range_meta["range_"],
                        remap_to_obj=remap_to_obj,
                        obj_index=obj_index,
                        obj_freq=obj_freq,
                        range_format=range_format,
                        template_context=_template_context,
                        silence_warnings=silence_warnings,
                        freq=freq,
                        return_obj_meta=True,
                        return_meta=True,
                    )
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            obj_range_meta["start"],
                            obj_range_meta["stop"],
                            right_inclusive=right_inclusive,
                            index=obj_meta["index"],
                            freq=obj_meta["freq"],
                        )
                    else:
                        if right_inclusive:
                            bounds = (
                                obj_range_meta["start"],
                                obj_range_meta["stop"] - 1,
                            )
                        else:
                            bounds = (
                                obj_range_meta["start"],
                                obj_range_meta["stop"],
                            )
            else:
                bounds = (None, None)
            return bounds

        bounds = {}

        def _get_func_args(i, j, _bounds=bounds):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            _template_context = merge_dicts(
                {
                    "split_idx": split_idx,
                    "split_label": split_labels[i],
                    "set_idx": set_idx,
                    "set_label": set_labels[j],
                },
                template_context,
            )
            range_meta = _get_range_meta(i, j, _template_context)
            _template_context = merge_dicts(
                dict(range_=range_meta["range_"], range_meta=range_meta),
                _template_context,
            )
            obj_meta1, obj_range_meta1, _apply_args = _take_args(apply_args, range_meta["range_"], _template_context)
            obj_meta2, obj_range_meta2, _apply_kwargs = _take_kwargs(
                apply_kwargs, range_meta["range_"], _template_context
            )
            obj_meta = {**obj_meta1, **obj_meta2}
            obj_range_meta = {**obj_range_meta1, **obj_range_meta2}
            _bounds[(i, j)] = _get_bounds(range_meta, _template_context)
            _template_context = merge_dicts(
                dict(
                    obj_meta=obj_meta,
                    obj_range_meta=obj_range_meta,
                    apply_args=_apply_args,
                    apply_kwargs=_apply_kwargs,
                    bounds=_bounds[(i, j)],
                ),
                _template_context,
            )
            _apply_func = substitute_templates(apply_func, _template_context, sub_id="apply_func")
            _apply_args = substitute_templates(_apply_args, _template_context, sub_id="apply_args")
            _apply_kwargs = substitute_templates(_apply_kwargs, _template_context, sub_id="apply_kwargs")
            return _apply_func, _apply_args, _apply_kwargs

        def _attach_bounds(keys, range_bounds):
            range_bounds = pd.MultiIndex.from_tuples(range_bounds, names=["start", "end"])
            if keys is None:
                return range_bounds
            index_stack_kwargs = dict(index_combine_kwargs)
            index_stack_kwargs.pop("ignore_ranges", None)
            return stack_indexes((keys, range_bounds), **index_stack_kwargs)

        if iteration.lower() == "split_major":

            def _get_generator():
                for i in range(n_splits):
                    for j in range(n_sets):
                        yield _get_func_args(i, j)

            funcs_args = _get_generator()
            results = execute(funcs_args, n_calls=n_splits * n_sets, **execute_kwargs)
        elif iteration.lower() == "set_major":

            def _get_generator():
                for j in range(n_sets):
                    for i in range(n_splits):
                        yield _get_func_args(i, j)

            funcs_args = _get_generator()
            results = execute(funcs_args, n_calls=n_splits * n_sets, **execute_kwargs)
        elif iteration.lower() == "split_wise":

            def _process_chunk(chunk):
                results = []
                for func, args, kwargs in chunk:
                    results.append(func(*args, **kwargs))
                return results

            def _get_generator():
                for i in range(n_splits):
                    chunk = []
                    for j in range(n_sets):
                        chunk.append(_get_func_args(i, j))
                    yield _process_chunk, (chunk,), {}

            funcs_args = _get_generator()
            results = execute(funcs_args, n_calls=n_splits, **execute_kwargs)
        elif iteration.lower() == "set_wise":

            def _process_chunk(chunk):
                results = []
                for func, args, kwargs in chunk:
                    results.append(func(*args, **kwargs))
                return results

            def _get_generator():
                for j in range(n_sets):
                    chunk = []
                    for i in range(n_splits):
                        chunk.append(_get_func_args(i, j))
                    yield _process_chunk, (chunk,), {}

            funcs_args = _get_generator()
            results = execute(funcs_args, n_calls=n_sets, **execute_kwargs)
        else:
            raise ValueError(f"Invalid option iteration='{iteration}'")

        if merge_all:
            if iteration.lower() in ("split_wise", "set_wise"):
                results = [result for _results in results for result in _results]
            if one_range:
                return results[0]
            if iteration.lower() in ("split_major", "split_wise"):
                if one_set:
                    keys = split_labels
                elif one_split:
                    keys = set_labels
                else:
                    keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
                if attach_bounds is not None:
                    range_bounds = []
                    for i in range(n_splits):
                        for j in range(n_sets):
                            range_bounds.append(bounds[(i, j)])
                    keys = _attach_bounds(keys, range_bounds)
            else:
                if one_set:
                    keys = split_labels
                elif one_split:
                    keys = set_labels
                else:
                    keys = combine_indexes((set_labels, split_labels), **index_combine_kwargs)
                if attach_bounds is not None:
                    range_bounds = []
                    for j in range(n_sets):
                        for i in range(n_splits):
                            range_bounds.append(bounds[(i, j)])
                    keys = _attach_bounds(keys, range_bounds)

            def _wrap_output(_results):
                try:
                    return pd.Series(_results, index=keys)
                except Exception as e:
                    return pd.Series(_results, index=keys, dtype=object)

            if merge_func is not None:
                template_context["funcs_args"] = funcs_args
                template_context["keys"] = keys
                if isinstance(merge_func, (str, tuple)):
                    merge_func = resolve_merge_func(merge_func)
                    merge_kwargs = {**dict(keys=keys), **merge_kwargs}
                merge_kwargs = substitute_templates(merge_kwargs, template_context, sub_id="merge_kwargs")
                return merge_func(results, **merge_kwargs)
            if wrap_results:
                if isinstance(results[0], tuple):
                    return tuple(map(_wrap_output, zip(*results)))
                return _wrap_output(results)
            return results

        if iteration.lower() == "split_major":
            new_results = []
            for i in range(n_splits):
                new_results.append(results[i * n_sets : (i + 1) * n_sets])
            results = new_results
        elif iteration.lower() == "set_major":
            new_results = []
            for i in range(n_sets):
                new_results.append(results[i * n_splits : (i + 1) * n_splits])
            results = new_results
        if one_range:
            return results[0][0]
        split_bounds = []
        if attach_bounds is not None:
            for i in range(n_splits):
                split_bounds.append([])
                for j in range(n_sets):
                    split_bounds[-1].append(bounds[(i, j)])
        set_bounds = []
        if attach_bounds is not None:
            for j in range(n_sets):
                set_bounds.append([])
                for i in range(n_splits):
                    set_bounds[-1].append(bounds[(i, j)])
        if iteration.lower() in ("split_major", "split_wise"):
            major_keys = split_labels
            minor_keys = set_labels
            major_bounds = split_bounds
            minor_bounds = set_bounds
            one_major = one_split
            one_minor = one_set
        else:
            major_keys = set_labels
            minor_keys = split_labels
            major_bounds = set_bounds
            minor_bounds = split_bounds
            one_major = one_set
            one_minor = one_split

        if merge_func is not None:
            merged_results = []
            for i, _results in enumerate(results):
                if one_minor:
                    merged_results.append(_results[0])
                else:
                    _template_context = dict(template_context)
                    _template_context["funcs_args"] = funcs_args
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[i])
                    else:
                        minor_keys_wbounds = minor_keys
                    _template_context["keys"] = minor_keys_wbounds
                    if isinstance(merge_func, (str, tuple)):
                        _merge_func = resolve_merge_func(merge_func)
                        _merge_kwargs = {**dict(keys=minor_keys_wbounds), **merge_kwargs}
                    else:
                        _merge_func = merge_func
                        _merge_kwargs = merge_kwargs
                    _merge_kwargs = substitute_templates(_merge_kwargs, _template_context, sub_id="merge_kwargs")
                    merged_results.append(_merge_func(_results, **_merge_kwargs))
            if one_major:
                return merged_results[0]
            if wrap_results:

                def _wrap_output(_results):
                    try:
                        return pd.Series(_results, index=major_keys)
                    except Exception as e:
                        return pd.Series(_results, index=major_keys, dtype=object)

                if isinstance(merged_results[0], tuple):
                    return tuple(map(_wrap_output, zip(*merged_results)))
                return _wrap_output(merged_results)
            return merged_results

        if one_major:
            results = results[0]
        elif one_minor:
            results = [_results[0] for _results in results]
        if wrap_results:

            def _wrap_output(_results):
                if one_minor:
                    if attach_bounds is not None:
                        major_keys_wbounds = _attach_bounds(major_keys, minor_bounds[0])
                    else:
                        major_keys_wbounds = major_keys
                    try:
                        return pd.Series(_results, index=major_keys_wbounds)
                    except Exception as e:
                        return pd.Series(_results, index=major_keys_wbounds, dtype=object)
                if one_major:
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[0])
                    else:
                        minor_keys_wbounds = minor_keys
                    try:
                        return pd.Series(_results, index=minor_keys_wbounds)
                    except Exception as e:
                        return pd.Series(_results, index=minor_keys_wbounds, dtype=object)
                new_results = []
                for i, r in enumerate(_results):
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[i])
                    else:
                        minor_keys_wbounds = minor_keys
                    try:
                        new_r = pd.Series(r, index=minor_keys_wbounds)
                    except Exception as e:
                        new_r = pd.Series(r, index=minor_keys_wbounds, dtype=object)
                    new_results.append(new_r)
                try:
                    return pd.Series(new_results, index=major_keys)
                except Exception as e:
                    return pd.Series(new_results, index=major_keys, dtype=object)

            if one_major or one_minor:
                if isinstance(results[0], tuple):
                    new_results = []
                    for k in range(len(results[0])):
                        new_results.append([])
                        for i in range(len(results)):
                            new_results[-1].append(results[i][k])
                    return tuple(map(_wrap_output, new_results))
            else:
                if isinstance(results[0][0], tuple):
                    new_results = []
                    for k in range(len(results[0][0])):
                        new_results.append([])
                        for i in range(len(results)):
                            new_results[-1].append([])
                            for j in range(len(results[0])):
                                new_results[-1][-1].append(results[i][j][k])
                    return tuple(map(_wrap_output, new_results))
            return _wrap_output(results)
        return results

    # ############# Splits ############# #

    def shuffle_splits(
        self: SplitterT,
        size: tp.Union[None, str, int] = None,
        replace: bool = False,
        p: tp.Optional[tp.Array1d] = None,
        seed: tp.Optional[int] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **init_kwargs,
    ) -> SplitterT:
        """Shuffle splits."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        rng = np.random.default_rng(seed=seed)
        if size is None:
            size = self.n_splits
        new_split_indices = rng.choice(np.arange(self.n_splits), size=size, replace=replace, p=p)
        new_splits_arr = self.splits_arr[new_split_indices]
        new_index = self.wrapper.index[new_split_indices]
        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = new_index
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    def break_up_splits(
        self: SplitterT,
        new_split: tp.SplitLike,
        sort: bool = False,
        template_context: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **split_range_kwargs,
    ) -> SplitterT:
        """Split each split into multiple splits.

        If there are multiple sets, make sure to merge them into one beforehand.

        Arguments `new_split` and `**split_range_kwargs` are passed to `Splitter.split_range`."""
        if self.n_sets > 1:
            raise ValueError("Cannot break up splits with more than one set. Merge sets first.")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        new_index = []
        range_starts = []
        for i, split in enumerate(self.splits_arr):
            new_ranges = self.split_range(split[0], new_split, template_context=template_context, **split_range_kwargs)
            for j, range_ in enumerate(new_ranges):
                if sort:
                    range_starts.append(self.get_range_bounds(range_, template_context=template_context)[0])
                new_splits_arr.append([range_])
                if isinstance(self.split_labels, pd.MultiIndex):
                    new_index.append((*self.split_labels[i], j))
                else:
                    new_index.append((self.split_labels[i], j))
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)
        new_index = pd.MultiIndex.from_tuples(new_index, names=[*self.split_labels.names, "split_part"])
        if sort:
            sorted_indices = np.argsort(range_starts)
            new_splits_arr = new_splits_arr[sorted_indices]
            new_index = new_index[sorted_indices]

        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = new_index
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    # ############# Sets ############# #

    def split_set(
        self: SplitterT,
        new_split: tp.SplitLike,
        column: tp.Optional[tp.Hashable] = None,
        new_set_labels: tp.Optional[tp.Sequence[tp.Hashable]] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **split_range_kwargs,
    ) -> SplitterT:
        """Split a set (column) into multiple sets (columns).

        Arguments `new_split` and `**split_range_kwargs` are passed to `Splitter.split_range`.

        Column must be provided if there are two or more sets.

        Use `new_set_labels` to specify the labels of the new sets; it must have the same length
        as there are new ranges in the new split. To provide final labels, define `columns` in
        `wrapper_kwargs`."""
        if self.n_sets == 0:
            raise ValueError("There are no sets to split")
        if self.n_sets > 1:
            if column is None:
                raise ValueError("Must provide column for multiple sets")
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
        else:
            column = 0
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        for split in self.splits_arr:
            new_ranges = self.split_range(split[column], new_split, **split_range_kwargs)
            new_splits_arr.append([*split[:column], *new_ranges, *split[column + 1 :]])
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            n_new_sets = new_splits_arr.shape[1] - self.n_sets + 1
            if new_set_labels is None:
                old_set_label = self.set_labels[column]
                if isinstance(old_set_label, str) and len(old_set_label.split("+")) == n_new_sets:
                    new_set_labels = old_set_label.split("+")
                else:
                    new_set_labels = [str(old_set_label) + "/" + str(i) for i in range(n_new_sets)]
            if len(new_set_labels) != n_new_sets:
                raise ValueError(f"Argument new_set_labels must have length {n_new_sets}, not {len(new_set_labels)}")
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(column)
            new_columns = new_columns.insert(column, new_set_labels)
            wrapper_kwargs["columns"] = new_columns
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    def merge_sets(
        self: SplitterT,
        columns: tp.Optional[tp.Iterable[tp.Hashable]] = None,
        new_set_label: tp.Optional[tp.Hashable] = None,
        insert_at_last: bool = False,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **merge_split_kwargs,
    ) -> SplitterT:
        """Merge multiple sets (columns) into a set (column).

        Arguments `**merge_split_kwargs` are passed to `Splitter.merge_split`.

        If columns are not provided, merges all columns. If provided and `insert_at_last` is True,
        a new column is inserted at the position of the last column.

        Use `new_set_label` to specify the label of the new set. To provide final labels,
        define `columns` in `wrapper_kwargs`."""
        if self.n_sets < 2:
            raise ValueError("There are no sets to merge")
        if columns is None:
            columns = range(len(self.set_labels))
        new_columns = []
        for column in columns:
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
            new_columns.append(column)
        columns = sorted(new_columns)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        merge_split_kwargs = dict(merge_split_kwargs)
        wrap_with_fixrange = merge_split_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        merge_split_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        for split in self.splits_arr:
            split_to_merge = []
            for j, range_ in enumerate(split):
                if j in columns:
                    split_to_merge.append(range_)
            new_range = self.merge_split(split_to_merge, **merge_split_kwargs)
            new_split = []
            for j in range(self.n_sets):
                if j not in columns:
                    new_split.append(split[j])
                else:
                    if insert_at_last:
                        if j == columns[-1]:
                            new_split.append(new_range)
                    else:
                        if j == columns[0]:
                            new_split.append(new_range)
            new_splits_arr.append(new_split)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            if new_set_label is None:
                old_set_labels = self.set_labels[columns]
                can_aggregate = True
                prefix = None
                suffix = None
                for i, old_set_label in enumerate(old_set_labels):
                    if not isinstance(old_set_label, str):
                        can_aggregate = False
                        break
                    _prefix = "/".join(old_set_label.split("/")[:-1])
                    _suffix = old_set_label.split("/")[-1]
                    if not _suffix.isdigit():
                        can_aggregate = False
                        break
                    _suffix = int(_suffix)
                    if prefix is None:
                        prefix = _prefix
                        suffix = _suffix
                        continue
                    if suffix != 0:
                        can_aggregate = False
                        break
                    if not _prefix == prefix or _suffix != i:
                        can_aggregate = False
                        break
                if can_aggregate and prefix + "/%d" % len(old_set_labels) not in self.set_labels:
                    new_set_label = prefix
                else:
                    new_set_label = "+".join(map(str, old_set_labels))
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(columns)
            if insert_at_last:
                new_columns = new_columns.insert(columns[-1] - len(columns) + 1, new_set_label)
            else:
                new_columns = new_columns.insert(columns[0], new_set_label)
            wrapper_kwargs["columns"] = new_columns
        if "ndim" not in wrapper_kwargs:
            if len(wrapper_kwargs["columns"]) == 1:
                wrapper_kwargs["ndim"] = 1
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    # ############# Bounds ############# #

    @class_or_instancemethod
    def map_bounds_to_index(
        cls_or_self,
        start: int,
        stop: int,
        right_inclusive: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Tuple[tp.Any, tp.Any]:
        """Map bounds to index."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        if right_inclusive:
            return index[start], index[stop - 1]
        if stop == len(index):
            freq = BaseIDXAccessor(index, freq=freq).any_freq
            if freq is None:
                raise ValueError("Must provide freq")
            return index[start], index[stop - 1] + freq
        return index[start], index[stop]

    @class_or_instancemethod
    def get_range_bounds(
        cls_or_self,
        range_: tp.FixRangeLike,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        check_constant: bool = True,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Tuple[tp.Any, tp.Any]:
        """Get the left (inclusive) and right (exclusive) bound of a range.

        !!! note
            Even when mapped to the index, the right bound is always exclusive."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        range_meta = cls_or_self.get_ready_range(
            range_,
            template_context=template_context,
            index=index,
            return_meta=True,
        )
        if check_constant and not range_meta["is_constant"]:
            raise ValueError("Range is not constant")
        if index_bounds:
            range_meta["start"], range_meta["stop"] = cls_or_self.map_bounds_to_index(
                range_meta["start"],
                range_meta["stop"],
                right_inclusive=right_inclusive,
                index=index,
                freq=freq,
            )
        else:
            if right_inclusive:
                range_meta["stop"] = range_meta["stop"] - 1
        return range_meta["start"], range_meta["stop"]

    def get_bounds_arr(
        self,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **range_bounds_kwargs,
    ) -> tp.BoundsArray:
        """Three-dimensional integer array with bounds.

        First axis represents splits. Second axis represents sets. Third axis represents bounds.

        Each range is getting selected using `Splitter.select_range` and then measured using
        `Splitter.get_range_bounds`. Keyword arguments `**kwargs` are passed to
        `Splitter.get_range_bounds`."""
        if index_bounds:
            dtype = self.index.dtype
        else:
            dtype = np.int_
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)

        try:
            bounds = np.empty((n_splits, n_sets, 2), dtype=dtype)
        except TypeError as e:
            bounds = np.empty((n_splits, n_sets, 2), dtype=object)
        for i in range(n_splits):
            for j in range(n_sets):
                range_ = self.select_range(
                    split=i,
                    set_=j,
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    split_as_indices=True,
                    set_as_indices=True,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                bounds[i, j, :] = self.get_range_bounds(
                    range_,
                    index_bounds=index_bounds,
                    right_inclusive=right_inclusive,
                    template_context=template_context,
                    **range_bounds_kwargs,
                )
        return bounds

    @property
    def bounds_arr(self) -> tp.BoundsArray:
        """`Splitter.get_bounds_arr` with default arguments."""
        return self.get_bounds_arr()

    def get_bounds(
        self,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Boolean Series/DataFrame where index are bounds and columns are splits stacked together.

        Keyword arguments `**kwargs` are passed to `Splitter.get_bounds_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        bounds_arr = self.get_bounds_arr(
            index_bounds=index_bounds,
            right_inclusive=right_inclusive,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        )
        out = bounds_arr.reshape((-1, 2))
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        new_columns = pd.Index(["start", "end"], name="bound")
        if one_split and one_set:
            return pd.Series(out[0], index=new_columns)
        if one_split:
            return pd.DataFrame(out, index=set_labels, columns=new_columns)
        if one_set:
            return pd.DataFrame(out, index=split_labels, columns=new_columns)
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=new_index, columns=new_columns)

    @property
    def bounds(self) -> tp.Frame:
        """`Splitter.get_bounds` with default arguments."""
        return self.get_bounds()

    @property
    def index_bounds(self) -> tp.Frame:
        """`Splitter.get_bounds` with `index_bounds=True`."""
        return self.get_bounds(index_bounds=True)

    def get_duration(self, **kwargs) -> tp.Series:
        """Get duration."""
        bounds = self.get_bounds(right_inclusive=False, **kwargs)
        return (bounds["end"] - bounds["start"]).rename("duration")

    @property
    def duration(self) -> tp.Series:
        """`Splitter.get_duration` with default arguments."""
        return self.get_duration()

    @property
    def index_duration(self) -> tp.Series:
        """`Splitter.get_duration` with `index_bounds=True`."""
        return self.get_duration(index_bounds=True)

    # ############# Masks ############# #

    @class_or_instancemethod
    def get_range_mask(
        cls_or_self,
        range_: tp.FixRangeLike,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Array1d:
        """Get the mask of a range."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = try_to_datetime_index(index)
        range_ = cls_or_self.get_ready_range(
            range_,
            allow_zero_len=True,
            template_context=template_context,
            index=index,
        )
        if isinstance(range_, np.ndarray) and range_.dtype == np.bool_:
            return range_
        mask = np.full(len(index), False)
        mask[range_] = True
        return mask

    def get_iter_split_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Array2d, None, None]:
        """Generator of two-dimensional boolean arrays, one per split.

        First axis represents sets. Second axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_range_mask`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)
        for i in range(n_splits):
            out = np.full((n_sets, len(self.index)), False)
            for j in range(n_sets):
                range_ = self.select_range(
                    split=i,
                    set_=j,
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    split_as_indices=True,
                    set_as_indices=True,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                out[j, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
            yield out

    @property
    def iter_split_mask_arrs(self) -> tp.Generator[tp.Array2d, None, None]:
        """`Splitter.get_iter_split_mask_arrs` with default arguments."""
        return self.get_iter_split_mask_arrs()

    def get_iter_set_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Array2d, None, None]:
        """Generator of two-dimensional boolean arrays, one per set.

        First axis represents splits. Second axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_range_mask`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)
        for j in range(n_sets):
            out = np.full((n_splits, len(self.index)), False)
            for i in range(n_splits):
                range_ = self.select_range(
                    split=i,
                    set_=j,
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    split_as_indices=True,
                    set_as_indices=True,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                out[i, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
            yield out

    @property
    def iter_set_mask_arrs(self) -> tp.Generator[tp.Array2d, None, None]:
        """`Splitter.get_iter_set_mask_arrs` with default arguments."""
        return self.get_iter_set_mask_arrs()

    def get_iter_split_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Frame, None, None]:
        """Generator of boolean DataFrames, one per split.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_split_mask_arrs`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        for mask in self.get_iter_split_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=set_labels)

    @property
    def iter_split_masks(self) -> tp.Generator[tp.Frame, None, None]:
        """`Splitter.get_iter_split_masks` with default arguments."""
        return self.get_iter_split_masks()

    def get_iter_set_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Frame, None, None]:
        """Generator of boolean DataFrames, one per set.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_set_mask_arrs`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        for mask in self.get_iter_set_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=split_labels)

    @property
    def iter_set_masks(self) -> tp.Generator[tp.Frame, None, None]:
        """`Splitter.get_iter_set_masks` with default arguments."""
        return self.get_iter_set_masks()

    def get_mask_arr(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SplitsMask:
        """Three-dimensional boolean array with splits.

        First axis represents splits. Second axis represents sets. Third axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_split_mask_arrs`."""
        return np.array(
            list(
                self.get_iter_split_mask_arrs(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    template_context=template_context,
                    **kwargs,
                )
            )
        )

    @property
    def mask_arr(self) -> tp.SplitsMask:
        """`Splitter.get_mask_arr` with default arguments."""
        return self.get_mask_arr()

    def get_mask(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Boolean Series/DataFrame where index is `Splitter.index` and columns are splits stacked together.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`.

        !!! warning
            Boolean arrays for a big number of splits may take a considerable amount of memory."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        out = np.moveaxis(mask_arr, -1, 0).reshape((len(self.index), -1))
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_split and one_set:
            return pd.Series(out[:, 0], index=self.index)
        if one_split:
            return pd.DataFrame(out, index=self.index, columns=set_labels)
        if one_set:
            return pd.DataFrame(out, index=self.index, columns=split_labels)
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_columns = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=self.index, columns=new_columns)

    @property
    def mask(self) -> tp.Frame:
        """`Splitter.get_mask` with default arguments."""
        return self.get_mask()

    def get_split_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get the coverage of each split mask.

        If `overlapping` is True, returns the number of overlapping True values between sets in each split.
        If `normalize` is True, returns the number of True values in each split relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each split relative to the total number of True values across all splits.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                coverage = (mask_arr.sum(axis=1) > 1).sum(axis=1) / mask_arr.any(axis=1).sum(axis=1)
            else:
                coverage = (mask_arr.sum(axis=1) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask_arr.any(axis=1).sum(axis=1) / mask_arr.any(axis=(0, 1)).sum()
                else:
                    coverage = mask_arr.any(axis=1).mean(axis=1)
            else:
                coverage = mask_arr.any(axis=1).sum(axis=1)
        one_split = len(split_labels) == 1 and squeeze_one_split
        if one_split:
            return coverage[0]
        return pd.Series(coverage, index=split_labels, name="split_coverage")

    @property
    def split_coverage(self) -> tp.Series:
        """`Splitter.get_split_coverage` with default arguments."""
        return self.get_split_coverage()

    def get_set_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_set: bool = True,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get the coverage of each set mask.

        If `overlapping` is True, returns the number of overlapping True values between splits in each set.
        If `normalize` is True, returns the number of True values in each set relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each set relative to the total number of True values across all sets.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                coverage = (mask_arr.sum(axis=0) > 1).sum(axis=1) / mask_arr.any(axis=0).sum(axis=1)
            else:
                coverage = (mask_arr.sum(axis=0) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask_arr.any(axis=0).sum(axis=1) / mask_arr.any(axis=(0, 1)).sum()
                else:
                    coverage = mask_arr.any(axis=0).mean(axis=1)
            else:
                coverage = mask_arr.any(axis=0).sum(axis=1)
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_set:
            return coverage[0]
        return pd.Series(coverage, index=set_labels, name="set_coverage")

    @property
    def set_coverage(self) -> tp.Series:
        """`Splitter.get_set_coverage` with default arguments."""
        return self.get_set_coverage()

    def get_range_coverage(
        self,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get the coverage of each range mask.

        If `normalize` is True, returns the number of True values in each range relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each range relative to the total number of True values in its split.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if normalize:
            if relative:
                coverage = (mask_arr.sum(axis=2) / mask_arr.any(axis=1).sum(axis=1)[:, None]).flatten()
            else:
                coverage = (mask_arr.sum(axis=2) / mask_arr.shape[2]).flatten()
        else:
            coverage = mask_arr.sum(axis=2).flatten()
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_split and one_set:
            return coverage[0]
        if one_split:
            return pd.Series(coverage, index=set_labels, name="range_coverage")
        if one_set:
            return pd.Series(coverage, index=split_labels, name="range_coverage")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.Series(coverage, index=index, name="range_coverage")

    @property
    def range_coverage(self) -> tp.Series:
        """`Splitter.get_range_coverage` with default arguments."""
        return self.get_range_coverage()

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> float:
        """Get the coverage of the entire mask.

        If `overlapping` is True, returns the number of overlapping True values.
        If `normalize` is True, returns the number of True values relative to the length of the index.
        If `overlapping` and `normalize` are True, returns the number of overlapping True values relative
        to the total number of True values.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                return (mask_arr.sum(axis=(0, 1)) > 1).sum() / mask_arr.any(axis=(0, 1)).sum()
            return (mask_arr.sum(axis=(0, 1)) > 1).sum()
        if normalize:
            return mask_arr.any(axis=(0, 1)).mean()
        return mask_arr.any(axis=(0, 1)).sum()

    @property
    def coverage(self) -> float:
        """`Splitter.get_coverage` with default arguments."""
        return self.get_coverage()

    def get_overlap_matrix(
        self,
        by: str = "split",
        normalize: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        jitted: tp.JittedOption = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Frame:
        """Get the overlap between each pair of ranges.

        The argument `by` can be one of 'split', 'set', and 'range'.

        If `normalize` is True, returns the number of True values in each overlap relative
        to the total number of True values in both ranges.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if by.lower() == "split":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_split_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.split_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_split:
                return overlap_matrix[0, 0]
            index = split_labels
        elif by.lower() == "set":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_set_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.set_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_set:
                return overlap_matrix[0, 0]
            index = set_labels
        elif by.lower() == "range":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_range_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.range_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_split and one_set:
                return overlap_matrix[0, 0]
            if one_split:
                index = set_labels
            elif one_set:
                index = split_labels
            else:
                if index_combine_kwargs is None:
                    index_combine_kwargs = {}
                index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        else:
            raise ValueError(f"Invalid option by='{by}'")
        return pd.DataFrame(overlap_matrix, index=index, columns=index)

    @property
    def split_overlap_matrix(self) -> tp.Frame:
        """`Splitter.get_overlap_matrix` with `by="split"`."""
        return self.get_overlap_matrix(by="split")

    @property
    def set_overlap_matrix(self) -> tp.Frame:
        """`Splitter.get_overlap_matrix` with `by="set"`."""
        return self.get_overlap_matrix(by="set")

    @property
    def range_overlap_matrix(self) -> tp.Frame:
        """`Splitter.get_overlap_matrix` with `by="range"`."""
        return self.get_overlap_matrix(by="range")

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Splitter.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.splitter`."""
        from vectorbtpro._settings import settings

        splitter_stats_cfg = settings["splitter"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), splitter_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Index Start",
                calc_func=lambda self: self.index[0],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            end=dict(
                title="Index End",
                calc_func=lambda self: self.index[-1],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            period=dict(
                title="Index Length",
                calc_func=lambda self: len(self.index),
                agg_func=None,
                tags=["splitter", "index"],
            ),
            split_count=dict(
                title="Splits",
                calc_func="n_splits",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            set_count=dict(
                title="Sets",
                calc_func="n_sets",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                calc_func="coverage",
                overlapping=False,
                post_calc_func=lambda self, out, settings: out * 100 if settings["normalize"] else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                check_has_multiple_sets=True,
                calc_func="set_coverage",
                overlapping=False,
                relative=False,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_mean_rel_coverage=dict(
                title="Mean Rel Coverage [%]",
                check_has_multiple_sets=True,
                check_normalize=True,
                calc_func="range_coverage",
                relative=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out.groupby(self.get_set_labels(set_group_by=settings.get("set_group_by", None)).names).mean()[
                        self.get_set_labels(set_group_by=settings.get("set_group_by", None))
                    ]
                    * 100,
                    orient="index_series",
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            overlap_coverage=dict(
                title=RepFunc(lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"),
                calc_func="coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: out * 100 if settings["normalize"] else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_overlap_coverage=dict(
                title=RepFunc(lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"),
                check_has_multiple_sets=True,
                calc_func="set_coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot splits as rows and sets as colors.

        Args:
            split_group_by (any): Split groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (any): Set groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (dict): Keyword arguments passed to `Splitter.get_iter_set_masks`.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap`.

                Can be a sequence, one per set.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd
        >>> from sklearn.model_selection import TimeSeriesSplit

        >>> index = pd.date_range("2020", "2021", freq="D")
        >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
        >>> splitter.plot().show()
        ```

        ![](/assets/images/api/Splitter.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure
        import plotly.express as px

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        if fig.layout.colorway is not None:
            colorway = fig.layout.colorway
        else:
            colorway = fig.layout.template.layout.colorway
        if len(set_labels) > len(colorway):
            colorway = px.colors.qualitative.Alphabet

        if self.get_n_splits(split_group_by=split_group_by) > 0:
            if self.get_n_sets(set_group_by=set_group_by) > 0:
                if mask_kwargs is None:
                    mask_kwargs = {}
                for i, mask in enumerate(
                    self.get_iter_set_masks(
                        split_group_by=split_group_by,
                        set_group_by=set_group_by,
                        **mask_kwargs,
                    )
                ):
                    df = mask.vbt.wrapper.fill()
                    df[mask] = i
                    color = adjust_opacity(colorway[i % len(colorway)], 0.8)
                    trace_name = str(set_labels[i])
                    _trace_kwargs = merge_dicts(
                        dict(
                            showscale=False,
                            showlegend=True,
                            legendgroup=str(set_labels[i]),
                            name=trace_name,
                            colorscale=[color, color],
                            hovertemplate="%{x}<br>Split: %{y}<br>Set: " + trace_name,
                        ),
                        resolve_dict(trace_kwargs, i=i),
                    )
                    fig = df.vbt.ts_heatmap(
                        trace_kwargs=_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        is_y_category=True,
                        fig=fig,
                    )
        return fig

    def plot_coverage(
        self,
        stacked: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot index as rows and sets as lines.

        Args:
            stacked (bool): Whether to plot as an area plot.
            split_group_by (any): Split groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (any): Set groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (dict): Keyword arguments passed to `Splitter.get_iter_set_masks`.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.

                Can be a sequence, one per set.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            * Area plot:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd
            >>> from sklearn.model_selection import TimeSeriesSplit

            >>> index = pd.date_range("2020", "2021", freq="D")
            >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
            >>> splitter.plot_coverage().show()
            ```

            ![](/assets/images/api/Splitter_coverage_area.svg){: .iimg loading=lazy }

            * Line plot:

            ```pycon
            >>> splitter.plot_coverage(stacked=False).show()
            ```

            ![](/assets/images/api/Splitter_coverage_line.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure
        import plotly.express as px

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        if fig.layout.colorway is not None:
            colorway = fig.layout.colorway
        else:
            colorway = fig.layout.template.layout.colorway
        if len(set_labels) > len(colorway):
            colorway = px.colors.qualitative.Alphabet

        if self.get_n_splits(split_group_by=split_group_by) > 0:
            if self.get_n_sets(set_group_by=set_group_by) > 0:
                if mask_kwargs is None:
                    mask_kwargs = {}
                for i, mask in enumerate(
                    self.get_iter_set_masks(
                        split_group_by=split_group_by,
                        set_group_by=set_group_by,
                        **mask_kwargs,
                    )
                ):
                    _trace_kwargs = merge_dicts(
                        dict(
                            stackgroup="coverage" if stacked else None,
                            legendgroup=str(set_labels[i]),
                            name=str(set_labels[i]),
                            line=dict(color=colorway[i % len(colorway)], shape="hv"),
                        ),
                        resolve_dict(trace_kwargs, i=i),
                    )
                    fig = mask.sum(axis=1).vbt.lineplot(
                        trace_kwargs=_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        fig=fig,
                    )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Splitter.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.splitter`."""
        from vectorbtpro._settings import settings

        splitter_plots_cfg = settings["splitter"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), splitter_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Splits",
                yaxis_kwargs=dict(title="Split"),
                plot_func="plot",
                tags="splitter",
            ),
            plot_coverage=dict(
                title="Coverage",
                yaxis_kwargs=dict(title="Count"),
                plot_func="plot_coverage",
                tags="splitter",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Splitter.override_metrics_doc(__pdoc__)
Splitter.override_subplots_doc(__pdoc__)

if settings["importing"]["sklearn"]:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.utils.validation import indexable

    class SKLSplitter(BaseCrossValidator):
        """Split iterator based on `Splitter`.

        Args:
            method (str or callable): Method that returns an instance of `Splitter`.
            *method_args: Positional arguments passed to `method`.
            splitter_cls (type): Splitter class.
            split_group_by (any): Split groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

                Not passed to `method`.
            set_group_by (any): Set groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

                Not passed to `method`.
            template_context (dict): Mapping used to substitute templates in ranges.

                Passed to `method`.
            **method_kwargs: Keyword arguments passed to `method`.

        Usage:
            * Replicate `TimeSeriesSplit` from scikit-learn:

            ```pycon
            >>> import numpy as np
            >>> import vectorbtpro as vbt

            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([1, 2, 3, 4])

            >>> tscv = vbt.SKLSplitter(
            ...     "from_expanding",
            ...     min_length=2,
            ...     offset=1,
            ...     split=-1
            ... )
            >>> for i, (train_indices, test_indices) in enumerate(tscv.split(X)):
            ...     print("Split %d:" % i)
            ...     X_train, X_test = X[train_indices], X[test_indices]
            ...     print("  X:", X_train.tolist(), X_test.tolist())
            ...     y_train, y_test = y[train_indices], y[test_indices]
            ...     print("  y:", y_train.tolist(), y_test.tolist())
            Split 0:
              X: [[1, 2]] [[3, 4]]
              y: [1] [2]
            Split 1:
              X: [[1, 2], [3, 4]] [[5, 6]]
              y: [1, 2] [3]
            Split 2:
              X: [[1, 2], [3, 4], [5, 6]] [[7, 8]]
              y: [1, 2, 3] [4]
            ```
        """

        def __init__(
            self,
            method: tp.Union[str, tp.Callable],
            *method_args,
            splitter_cls: tp.Type[Splitter] = Splitter,
            split_group_by: tp.AnyGroupByLike = None,
            set_group_by: tp.AnyGroupByLike = None,
            template_context: tp.KwargsLike = None,
            **method_kwargs,
        ) -> None:
            self.method = method
            self.method_args = method_args
            self.method_kwargs = method_kwargs
            self.splitter_cls = splitter_cls
            self.split_group_by = split_group_by
            self.set_group_by = set_group_by
            self.template_context = template_context

        def get_splitter(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> Splitter:
            """Get splitter of type `Splitter`."""
            X, y, groups = indexable(X, y, groups)
            try:
                index = self.splitter_cls.get_obj_index(X)
            except ValueError as e:
                index = pd.RangeIndex(stop=len(X))
            if isinstance(self.method, str):
                method = getattr(self.splitter_cls, self.method)
            else:
                method = self.method
            splitter = method(
                index,
                *self.method_args,
                template_context=self.template_context,
                **self.method_kwargs,
            )
            if splitter.get_n_sets(set_group_by=self.set_group_by) != 2:
                raise ValueError("Number of sets in the splitter must be 2: train and test")
            return splitter

        def _iter_masks(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Tuple[tp.Array1d, tp.Array1d], None, None]:
            """Generates boolean masks corresponding to train and test sets."""
            splitter = self.get_splitter(X=X, y=y, groups=groups)
            for mask_arr in splitter.get_iter_split_mask_arrs(
                split_group_by=self.split_group_by,
                set_group_by=self.set_group_by,
                template_context=self.template_context,
            ):
                yield mask_arr[0], mask_arr[1]

        def _iter_train_masks(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Array1d, None, None]:
            """Generates boolean masks corresponding to train sets."""
            for train_mask_arr, _ in self._iter_masks(X=X, y=y, groups=groups):
                yield train_mask_arr

        def _iter_test_masks(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Array1d, None, None]:
            """Generates boolean masks corresponding to test sets."""
            for _, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
                yield test_mask_arr

        def _iter_indices(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Tuple[tp.Array1d, tp.Array1d], None, None]:
            """Generates integer indices corresponding to train and test sets."""
            for train_mask_arr, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
                yield np.flatnonzero(train_mask_arr), np.flatnonzero(test_mask_arr)

        def _iter_train_indices(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Array1d, None, None]:
            """Generates integer indices corresponding to train sets."""
            for train_indices, _ in self._iter_indices(X=X, y=y, groups=groups):
                yield train_indices

        def _iter_test_indices(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Array1d, None, None]:
            """Generates integer indices corresponding to test sets."""
            for _, test_indices in self._iter_indices(X=X, y=y, groups=groups):
                yield test_indices

        def get_n_splits(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> int:
            """Returns the number of splitting iterations in the cross-validator."""
            splitter = self.get_splitter(X=X, y=y, groups=groups)
            return splitter.get_n_splits(split_group_by=self.split_group_by)

        def split(
            self,
            X: tp.Any = None,
            y: tp.Any = None,
            groups: tp.Any = None,
        ) -> tp.Generator[tp.Tuple[tp.Array1d, tp.Array1d], None, None]:
            """Generate indices to split data into training and test set."""
            return self._iter_indices(X=X, y=y, groups=groups)

else:
    SKLSplitter = None
