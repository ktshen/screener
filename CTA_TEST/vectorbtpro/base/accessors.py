# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom Pandas accessors for base operations with Pandas objects."""

import inspect
import ast
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler
from pandas.tseries.frequencies import to_offset

from vectorbtpro import _typing as tp
from vectorbtpro.base import combining, reshaping, indexes
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.indexing import point_idxr_defaults, range_idxr_defaults, get_index_points, get_index_ranges
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, resolve_dict, Configured
from vectorbtpro.utils.decorators import class_or_instanceproperty, class_or_instancemethod
from vectorbtpro.utils.magic_decorators import attach_binary_magic_methods, attach_unary_magic_methods
from vectorbtpro.utils.parsing import get_context_vars
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.utils.datetime_ import (
    infer_index_freq,
    freq_to_timedelta64,
    parse_timedelta,
    try_to_datetime_index,
    to_ns,
)
from vectorbtpro.utils.eval_ import multiline_eval

__all__ = ["BaseIDXAccessor", "BaseAccessor", "BaseSRAccessor", "BaseDFAccessor"]


class BaseIDXAccessor(Configured):
    """Accessor on top of Index.

    Accessible via `pd.Index.vbt` and all child accessors."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "obj",
        "freq",
    }

    def __init__(self, obj: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None, **kwargs) -> None:
        checks.assert_instance_of(obj, pd.Index)

        Configured.__init__(self, obj=obj, freq=freq, **kwargs)

        self._obj = obj
        self._freq = freq

    @property
    def obj(self):
        """Pandas object."""
        return self._obj

    # ############# Index ############# #

    def to_ns(self) -> tp.Array1d:
        """Convert index to an 64-bit integer array.

        Timestamps will be converted to nanoseconds."""
        return to_ns(self.obj)

    def to_period_ns(self, freq: tp.FrequencyLike, shift: bool = True) -> tp.Array1d:
        """Convert index to period and then to an 64-bit integer array.

        Timestamps will be converted to nanoseconds."""
        index = self.obj
        if isinstance(index, pd.DatetimeIndex):
            index = index.tz_localize(None).to_period(freq)
            if shift:
                index = index.shift()
        return to_ns(index)

    @classmethod
    def from_values(cls, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.index_from_values`."""
        return indexes.index_from_values(*args, **kwargs)

    def repeat(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.repeat_index`."""
        return indexes.repeat_index(self.obj, *args, **kwargs)

    def tile(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.tile_index`."""
        return indexes.tile_index(self.obj, *args, **kwargs)

    @class_or_instancemethod
    def stack(cls_or_self, *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"], **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.stack_indexes`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return indexes.stack_indexes(*others, **kwargs)
        return indexes.stack_indexes(cls_or_self.obj, *others, **kwargs)

    @class_or_instancemethod
    def combine(cls_or_self, *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"], **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.combine_indexes`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return indexes.combine_indexes(*others, **kwargs)
        return indexes.combine_indexes(cls_or_self.obj, *others, **kwargs)

    @class_or_instancemethod
    def concat(cls_or_self, *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"], **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.concat_indexes`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return indexes.concat_indexes(*others, **kwargs)
        return indexes.concat_indexes(cls_or_self.obj, *others, **kwargs)

    def drop_levels(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.drop_levels`."""
        return indexes.drop_levels(self.obj, *args, **kwargs)

    def rename_levels(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.rename_levels`."""
        return indexes.rename_levels(self.obj, *args, **kwargs)

    def select_levels(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.select_levels`."""
        return indexes.select_levels(self.obj, *args, **kwargs)

    def drop_redundant_levels(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.drop_redundant_levels`."""
        return indexes.drop_redundant_levels(self.obj, *args, **kwargs)

    def drop_duplicate_levels(self, *args, **kwargs) -> tp.Index:
        """See `vectorbtpro.base.indexes.drop_duplicate_levels`."""
        return indexes.drop_duplicate_levels(self.obj, *args, **kwargs)

    def align_to(self, *args, **kwargs) -> tp.IndexSlice:
        """See `vectorbtpro.base.indexes.align_index_to`."""
        return indexes.align_index_to(self.obj, *args, **kwargs)

    def find_first_occurrence(self, *args, **kwargs) -> int:
        """See `vectorbtpro.base.indexes.find_first_occurrence`."""
        return indexes.find_first_occurrence(self.obj, *args, **kwargs)

    # ############# Frequency ############# #

    def get_freq(
        self,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> tp.Union[None, float, tp.PandasFrequency]:
        """Index frequency as `pd.Timedelta` or None if it cannot be converted."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if freq is None:
            freq = self._freq
        if freq is None:
            freq = wrapping_cfg["freq"]
        try:
            return infer_index_freq(self.obj, freq=freq, **kwargs)
        except Exception as e:
            return None

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """`BaseIDXAccessor.get_freq` with date offsets and integer frequencies not allowed."""
        return self.get_freq(allow_date_offset=False, allow_numeric=False)

    @property
    def ns_freq(self) -> tp.Optional[int]:
        """Convert frequency to a 64-bit integer.

        Timedelta will be converted to nanoseconds."""
        freq = self.get_freq(allow_date_offset=False, allow_numeric=True)
        if freq is not None:
            freq = to_ns(freq_to_timedelta64(freq))
        return freq

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """Index frequency of any type."""
        return self.get_freq()

    @property
    def period(self) -> int:
        """Get the period of the index, without taking into account its datetime-like properties."""
        return len(self.obj)

    @property
    def dt_period(self) -> float:
        """Get the period of the index, taking into account its datetime-like properties."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if isinstance(self.obj, pd.DatetimeIndex):
            if self.freq is not None:
                return (self.obj[-1] - self.obj[0]) / self.freq + 1
            if not wrapping_cfg["silence_warnings"]:
                warnings.warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`.",
                    stacklevel=2,
                )
        if isinstance(self.obj[0], int) and isinstance(self.obj[-1], int):
            return self.obj[-1] - self.obj[0] + 1
        if not wrapping_cfg["silence_warnings"]:
            warnings.warn("Index is neither datetime-like nor integer", stacklevel=2)
        return self.period

    def arr_to_timedelta(
        self,
        a: tp.MaybeArray,
        to_pd: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[pd.Index, tp.MaybeArray]:
        """Convert array to duration using `BaseIDXAccessor.freq`."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        freq = self.freq
        if freq is None:
            if not silence_warnings:
                warnings.warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`.",
                    stacklevel=2,
                )
            return a

        if to_pd:
            out = pd.to_timedelta(a * freq)
        else:
            out = a * freq
        return out

    # ############# Grouping ############# #

    def get_grouper(self, by: tp.AnyGroupByLike, groupby_kwargs: tp.KwargsLike = None, **kwargs) -> Grouper:
        """Get an index grouper of type `vectorbtpro.base.grouping.base.Grouper`.

        Argument `by` can be a grouper itself, an instance of Pandas `GroupBy`,
        an instance of Pandas `Resampler`, but also any supported input to any of them
        such as a frequency or an array of indices.

        Keyword arguments `groupby_kwargs` are passed to the Pandas methods `groupby` and `resample`,
        while `**kwargs` are passed to initialize `vectorbtpro.base.grouping.base.Grouper`."""
        if groupby_kwargs is None:
            groupby_kwargs = {}
        if isinstance(by, Grouper):
            if len(kwargs) > 0:
                return by.replace(**kwargs)
            return by
        if isinstance(by, (PandasGroupBy, PandasResampler)):
            return Grouper.from_pd_group_by(by, **kwargs)
        try:
            return Grouper(index=self.obj, group_by=by, **kwargs)
        except Exception as e:
            pass
        if isinstance(self.obj, pd.DatetimeIndex):
            try:
                by = to_offset(parse_timedelta(by))
                if by.n == 1:
                    return Grouper(index=self.obj, group_by=self.obj.tz_localize(None).to_period(by), **kwargs)
            except Exception as e:
                pass
            try:
                pd_group_by = pd.Series(index=self.obj, dtype=object).resample(parse_timedelta(by), **groupby_kwargs)
                return Grouper.from_pd_group_by(pd_group_by, **kwargs)
            except Exception as e:
                pass
        pd_group_by = pd.Series(index=self.obj, dtype=object).groupby(by, axis=0, **groupby_kwargs)
        return Grouper.from_pd_group_by(pd_group_by, **kwargs)

    def get_resampler(
        self,
        rule: tp.AnyRuleLike,
        freq: tp.Optional[tp.FrequencyLike] = None,
        resample_kwargs: tp.KwargsLike = None,
        return_pd_resampler: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[Resampler, tp.PandasResampler]:
        """Get an index resampler of type `vectorbtpro.base.resampling.base.Resampler`."""
        if checks.is_td_like(rule):
            try:
                rule = parse_timedelta(rule)
                is_td = True
            except Exception as e:
                is_td = False
            if is_td:
                resample_kwargs = merge_dicts(
                    dict(closed="left", label="left"),
                    resample_kwargs,
                )
                rule = pd.Series(index=self.obj, dtype=object).resample(rule, **resolve_dict(resample_kwargs))
        if isinstance(rule, PandasResampler):
            if return_pd_resampler:
                return rule
            if silence_warnings is None:
                silence_warnings = True
            rule = Resampler.from_pd_resampler(rule, source_freq=self.freq, silence_warnings=silence_warnings)
        if return_pd_resampler:
            raise TypeError("Cannot convert Resampler to Pandas Resampler")
        if checks.is_dt_like(rule) or checks.is_iterable(rule):
            rule = try_to_datetime_index(rule)
            rule = Resampler(
                source_index=self.obj,
                target_index=rule,
                source_freq=self.freq,
                target_freq=freq,
                silence_warnings=silence_warnings,
            )
        if isinstance(rule, Resampler):
            if freq is not None:
                rule = rule.replace(target_freq=freq)
            return rule
        raise ValueError(f"Cannot build Resampler from {rule}")

    # ############# Points and ranges ############# #

    def get_index_points(self, *args, **kwargs) -> tp.Array1d:
        """See `vectorbtpro.base.indexing.get_index_points`."""
        return get_index_points(self.obj, *args, **kwargs)

    def get_index_ranges(self, *args, **kwargs) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """See `vectorbtpro.base.indexing.get_index_ranges`."""
        return get_index_ranges(self.obj, self.any_freq, *args, **kwargs)


BaseAccessorT = tp.TypeVar("BaseAccessorT", bound="BaseAccessor")


@attach_binary_magic_methods(lambda self, other, np_func: self.combine(other, combine_func=np_func))
@attach_unary_magic_methods(lambda self, np_func: self.apply(apply_func=np_func))
class BaseAccessor(Wrapping):
    """Accessor on top of Series and DataFrames.

    Accessible via `pd.Series.vbt` and `pd.DataFrame.vbt`, and all child accessors.

    Series is just a DataFrame with one column, hence to avoid defining methods exclusively for 1-dim data,
    we will convert any Series to a DataFrame and perform matrix computation on it. Afterwards,
    by using `BaseAccessor.wrapper`, we will convert the 2-dim output back to a Series.

    `**kwargs` will be passed to `vectorbtpro.base.wrapping.ArrayWrapper`.

    !!! note
        When using magic methods, ensure that `.vbt` is called on the operand on the left
        if the other operand is an array.

        Accessors do not utilize caching.

        Grouping is only supported by the methods that accept the `group_by` argument.

    Usage:
        * Build a symmetric matrix:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> import vectorbtpro as vbt

        >>> # vectorbtpro.base.accessors.BaseAccessor.make_symmetric
        >>> pd.Series([1, 2, 3]).vbt.make_symmetric()
             0    1    2
        0  1.0  2.0  3.0
        1  2.0  NaN  NaN
        2  3.0  NaN  NaN
        ```

        * Broadcast pandas objects:

        ```pycon
        >>> sr = pd.Series([1])
        >>> df = pd.DataFrame([1, 2, 3])

        >>> vbt.base.reshaping.broadcast_to(sr, df)
           0
        0  1
        1  1
        2  1

        >>> sr.vbt.broadcast_to(df)
           0
        0  1
        1  1
        2  1
        ```

        * Many methods such as `BaseAccessor.broadcast` are both class and instance methods:

        ```pycon
        >>> from vectorbtpro.base.accessors import BaseAccessor

        >>> # Same as sr.vbt.broadcast(df)
        >>> new_sr, new_df = BaseAccessor.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        * Instead of explicitly importing `BaseAccessor` or any other accessor, we can use `pd_acc` instead:

        ```pycon
        >>> vbt.pd_acc.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        * `BaseAccessor` implements arithmetic (such as `+`), comparison (such as `>`) and
        logical operators (such as `&`) by forwarding the operation to `BaseAccessor.combine`:

        ```pycon
        >>> sr.vbt + df
           0
        0  2
        1  3
        2  4
        ```

        Many interesting use cases can be implemented this way.

        * For example, let's compare an array with 3 different thresholds:

        ```pycon
        >>> df.vbt > vbt.Param(np.arange(3), name='threshold')
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```

        * The same using the broadcasting mechanism:

        ```pycon
        >>> df.vbt > vbt.Param(np.arange(3), name='threshold')
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```
    """

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `BaseAccessor` after stacking along rows."""
        if "obj" not in kwargs:
            kwargs["obj"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.obj for obj in objs],
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `BaseAccessor` after stacking along columns."""
        if "obj" not in kwargs:
            kwargs["obj"] = kwargs["wrapper"].column_stack_arrs(
                *[obj.obj for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        if kwargs["wrapper"].ndim == 1:
            return cls.sr_accessor_cls(**kwargs)
        return cls.df_accessor_cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        wrapper_kwargs: tp.KwargsLike = None,
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.column_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls.df_accessor_cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Wrapping._expected_keys or set()) | {
        "obj",
    }

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        if len(kwargs) > 0:
            wrapper_kwargs, kwargs = ArrayWrapper.extract_init_kwargs(**kwargs)
        else:
            wrapper_kwargs, kwargs = {}, {}
        if not isinstance(wrapper, ArrayWrapper):
            if obj is not None:
                raise ValueError("Must either provide wrapper and object, or only object")
            wrapper, obj = ArrayWrapper.from_obj(wrapper, **wrapper_kwargs), wrapper
        else:
            if obj is None:
                raise ValueError("Must either provide wrapper and object, or only object")
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)

        Wrapping.__init__(self, wrapper, obj=obj, **kwargs)

        self._obj = obj

    def __call__(self: BaseAccessorT, **kwargs) -> BaseAccessorT:
        """Allows passing arguments to the initializer."""

        return self.replace(**kwargs)

    @class_or_instanceproperty
    def sr_accessor_cls(cls_or_self) -> tp.Type["BaseSRAccessor"]:
        """Accessor class for `pd.Series`."""
        return BaseSRAccessor

    @class_or_instanceproperty
    def df_accessor_cls(cls_or_self) -> tp.Type["BaseDFAccessor"]:
        """Accessor class for `pd.DataFrame`."""
        return BaseDFAccessor

    def indexing_func(self: BaseAccessorT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> BaseAccessorT:
        """Perform indexing on `BaseAccessor`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_obj = ArrayWrapper.select_from_flex_array(
            self._obj,
            row_idxs=wrapper_meta["row_idxs"],
            col_idxs=wrapper_meta["col_idxs"],
            rows_changed=wrapper_meta["rows_changed"],
            columns_changed=wrapper_meta["columns_changed"],
        )
        if checks.is_series(new_obj):
            return self.replace(cls_=self.sr_accessor_cls, wrapper=wrapper_meta["new_wrapper"], obj=new_obj)
        return self.replace(cls_=self.df_accessor_cls, wrapper=wrapper_meta["new_wrapper"], obj=new_obj)

    @property
    def obj(self) -> tp.SeriesFrame:
        """Pandas object."""
        if isinstance(self._obj, (pd.Series, pd.DataFrame)):
            if self._obj.shape == self.wrapper.shape:
                if self._obj.index is self.wrapper.index:
                    if isinstance(self._obj, pd.Series) and self._obj.name == self.wrapper.name:
                        return self._obj
                    if isinstance(self._obj, pd.DataFrame) and self._obj.columns is self.wrapper.columns:
                        return self._obj
        return self.wrapper.wrap(self._obj, group_by=False)

    @class_or_instanceproperty
    def ndim(cls_or_self) -> tp.Optional[int]:
        if isinstance(cls_or_self, type):
            return None
        return cls_or_self.obj.ndim

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.Series)

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.DataFrame)

    @classmethod
    def resolve_shape(cls, shape: tp.ShapeLike) -> tp.Shape:
        """Resolve shape."""
        shape_2d = reshaping.to_2d_shape(shape)
        try:
            if cls.is_series() and shape_2d[1] > 1:
                raise ValueError("Use DataFrame accessor")
        except NotImplementedError:
            pass
        return shape_2d

    # ############# Creation ############# #

    @classmethod
    def empty(cls, shape: tp.Shape, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """Generate an empty Series/DataFrame of shape `shape` and fill with `fill_value`."""
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            return pd.Series(np.full(shape, fill_value), **kwargs)
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(cls, other: tp.SeriesFrame, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """Generate an empty Series/DataFrame like `other` and fill with `fill_value`."""
        if checks.is_series(other):
            return cls.empty(other.shape, fill_value=fill_value, index=other.index, name=other.name, **kwargs)
        return cls.empty(other.shape, fill_value=fill_value, index=other.index, columns=other.columns, **kwargs)

    # ############# Indexes ############# #

    def apply_on_index(
        self,
        apply_func: tp.Callable,
        *args,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Apply function `apply_func` on index of the pandas object.

        Set `axis` to 1 for columns, 0 for index, and None to determine automatically.
        Set `copy_data` to True to make a deep copy of the data."""
        if axis is None:
            axis = 0 if self.is_series() else 1
        if self.is_series() and axis == 1:
            raise TypeError("Axis 1 is not supported in Series")
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            obj_index = self.wrapper.columns
        else:
            obj_index = self.wrapper.index
        obj_index = apply_func(obj_index, *args, **kwargs)
        obj = self.obj.values
        if copy_data:
            obj = obj.copy()
        if axis == 1:
            return self.wrapper.wrap(obj, group_by=False, columns=obj_index)
        return self.wrapper.wrap(obj, group_by=False, index=obj_index)

    def stack_index(
        self,
        index: tp.Index,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        on_top: bool = True,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.stack_indexes`.

        Set `on_top` to False to stack at bottom.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            if on_top:
                return indexes.stack_indexes([index, obj_index], **kwargs)
            return indexes.stack_indexes([obj_index, index], **kwargs)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_levels(
        self,
        levels: tp.MaybeLevelSequence,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        strict: bool = True,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_levels(obj_index, levels, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def rename_levels(
        self,
        name_dict: tp.Dict[str, tp.Any],
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        strict: bool = True,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.rename_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.rename_levels(obj_index, name_dict, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def select_levels(
        self,
        level_names: tp.MaybeLevelSequence,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.select_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.select_levels(obj_index, level_names)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_redundant_levels(
        self,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_redundant_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_redundant_levels(obj_index)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_duplicate_levels(
        self,
        keep: tp.Optional[str] = None,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_duplicate_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_duplicate_levels(obj_index, keep=keep)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    # ############# Setting ############# #

    def set(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set value at each index point using `vectorbtpro.base.indexing.get_index_points`.

        If `value_or_func` is a function, selects all keyword arguments that were not passed
        to the `get_index_points` method, substitutes any templates, and passes everything to the function.
        As context uses `kwargs`, `template_context`, and various variables such as `i` (iteration index),
        `index_point` (absolute position in the index), `wrapper`, and `obj`."""
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_points = get_index_points(self.wrapper.index, **kwargs)

        if callable(value_or_func):
            func_kwargs = {k: v for k, v in kwargs.items() if k not in point_idxr_defaults}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        if callable(value_or_func):
            for i in range(len(index_points)):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_point=index_points[i],
                        index_points=index_points,
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = substitute_templates(args, _template_context, sub_id="func_args")
                _func_kwargs = substitute_templates(func_kwargs, _template_context, sub_id="func_kwargs")
                v = value_or_func(*_func_args, **_func_kwargs)
                if self.is_series() or columns is None:
                    obj.iloc[index_points[i]] = v
                elif is_scalar(columns):
                    obj.iloc[index_points[i], obj.columns.get_indexer([columns])[0]] = v
                else:
                    obj.iloc[index_points[i], obj.columns.get_indexer(columns)] = v
        elif checks.is_sequence(value_or_func) and not is_scalar(value_or_func):
            if self.is_series():
                obj.iloc[index_points] = reshaping.to_1d_array(value_or_func)
            elif columns is None:
                obj.iloc[index_points] = reshaping.to_2d_array(value_or_func)
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = reshaping.to_1d_array(value_or_func)
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = reshaping.to_2d_array(value_or_func)
        else:
            if self.is_series() or columns is None:
                obj.iloc[index_points] = value_or_func
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = value_or_func
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = value_or_func
        if inplace:
            return None
        return obj

    def set_between(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set value at each index range using `vectorbtpro.base.indexing.get_index_ranges`.

        If `value_or_func` is a function, selects all keyword arguments that were not passed
        to the `get_index_points` method, substitutes any templates, and passes everything to the function.
        As context uses `kwargs`, `template_context`, and various variables such as `i` (iteration index),
        `index_slice` (absolute slice of the index), `wrapper`, and `obj`."""
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_ranges = get_index_ranges(self.wrapper.index, **kwargs)

        if callable(value_or_func):
            func_kwargs = {k: v for k, v in kwargs.items() if k not in range_idxr_defaults}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        for i in range(len(index_ranges[0])):
            if callable(value_or_func):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_slice=slice(index_ranges[0][i], index_ranges[1][i]),
                        range_starts=index_ranges[0],
                        range_ends=index_ranges[1],
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = substitute_templates(args, _template_context, sub_id="func_args")
                _func_kwargs = substitute_templates(func_kwargs, _template_context, sub_id="func_kwargs")
                v = value_or_func(*_func_args, **_func_kwargs)
            elif checks.is_sequence(value_or_func) and not isinstance(value_or_func, str):
                v = value_or_func[i]
            else:
                v = value_or_func
            if self.is_series() or columns is None:
                obj.iloc[index_ranges[0][i] : index_ranges[1][i]] = v
            elif is_scalar(columns):
                obj.iloc[index_ranges[0][i] : index_ranges[1][i], obj.columns.get_indexer([columns])[0]] = v
            else:
                obj.iloc[index_ranges[0][i] : index_ranges[1][i], obj.columns.get_indexer(columns)] = v
        if inplace:
            return None
        return obj

    # ############# Reshaping ############# #

    def to_1d_array(self) -> tp.Array1d:
        """See `vectorbtpro.base.reshaping.to_1d` with `raw` set to True."""
        return reshaping.to_1d_array(self.obj)

    def to_2d_array(self) -> tp.Array2d:
        """See `vectorbtpro.base.reshaping.to_2d` with `raw` set to True."""
        return reshaping.to_2d_array(self.obj)

    def tile(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.base.reshaping.tile`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        tiled = reshaping.tile(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([keys, self.wrapper.columns])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([keys, self.wrapper.index])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return tiled

    def repeat(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.base.reshaping.repeat`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        repeated = reshaping.repeat(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([self.wrapper.columns, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([self.wrapper.index, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return repeated

    def align_to(self, other: tp.SeriesFrame, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Align to `other` on their axes.

        Usage:
            ```pycon
            >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['x', 'y'], columns=['a', 'b'])
            >>> df1
               a  b
            x  1  2
            y  3  4

            >>> df2 = pd.DataFrame([[5, 6, 7, 8], [9, 10, 11, 12]], index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']]))
            >>> df2
                   1       2
               a   b   a   b
            x  5   6   7   8
            y  9  10  11  12

            >>> df1.vbt.align_to(df2)
                  1     2
               a  b  a  b
            x  1  2  1  2
            y  3  4  3  4
            ```
        """
        checks.assert_instance_of(other, (pd.Series, pd.DataFrame))
        obj = reshaping.to_2d(self.obj)
        other = reshaping.to_2d(other)

        aligned_index = indexes.align_index_to(obj.index, other.index)
        aligned_columns = indexes.align_index_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrapper.wrap(
            obj.values,
            group_by=False,
            **merge_dicts(dict(index=other.index, columns=other.columns), wrap_kwargs),
        )

    @class_or_instancemethod
    def broadcast(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return reshaping.broadcast(*others, **kwargs)
        return reshaping.broadcast(cls_or_self.obj, *others, **kwargs)

    def broadcast_to(self, other: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast_to`."""
        if isinstance(other, BaseAccessor):
            other = other.obj
        return reshaping.broadcast_to(self.obj, other, **kwargs)

    @class_or_instancemethod
    def broadcast_combs(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast_combs`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return reshaping.broadcast_combs(*others, **kwargs)
        return reshaping.broadcast_combs(cls_or_self.obj, *others, **kwargs)

    def make_symmetric(self, *args, **kwargs) -> tp.Frame:
        """See `vectorbtpro.base.reshaping.make_symmetric`."""
        return reshaping.make_symmetric(self.obj, *args, **kwargs)

    def unstack_to_array(self, *args, **kwargs) -> tp.Array:
        """See `vectorbtpro.base.reshaping.unstack_to_array`."""
        return reshaping.unstack_to_array(self.obj, *args, **kwargs)

    def unstack_to_df(self, *args, **kwargs) -> tp.Frame:
        """See `vectorbtpro.base.reshaping.unstack_to_df`."""
        return reshaping.unstack_to_df(self.obj, *args, **kwargs)

    def to_dict(self, *args, **kwargs) -> tp.Mapping:
        """See `vectorbtpro.base.reshaping.to_dict`."""
        return reshaping.to_dict(self.obj, *args, **kwargs)

    # ############# Combining ############# #

    def apply(
        self,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Apply a function `apply_func`.

        Set `keep_pd` to True to keep inputs as pandas objects, otherwise convert to NumPy arrays.

        Set `to_2d` to True to reshape inputs to 2-dim arrays, otherwise keep as-is.

        `*args` and `**kwargs` are passed to `apply_func`.

        !!! note
            The resulted array must have the same shape as the original array.

        Usage:
            * Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> sr.vbt.apply(lambda x: x ** 2)
            x    1
            y    4
            dtype: int64
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr.vbt.apply(
            ...     lambda x, y: x + y,
            ...     vbt.Rep('y'),
            ...     broadcast_named_args=dict(
            ...         y=pd.DataFrame([[3, 4]], columns=['a', 'b'])
            ...     )
            ... )
               a  b
            x  4  5
            y  5  6
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = substitute_templates(args, template_context, sub_id="args")
        kwargs = substitute_templates(kwargs, template_context, sub_id="kwargs")
        out = apply_func(broadcast_named_args["obj"], *args, **kwargs)
        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def concat(
        cls_or_self,
        *others: tp.ArrayLike,
        broadcast_kwargs: tp.KwargsLike = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Frame:
        """Concatenate with `others` along columns.

        Usage:
            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> sr.vbt.concat(df, keys=['c', 'd'])
                  c     d
               a  b  a  b
            x  1  1  3  4
            y  2  2  5  6
            ```
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj,) + others
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        broadcasted = reshaping.broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(reshaping.to_2d, broadcasted))
        out = pd.concat(broadcasted, axis=1, keys=keys)
        if not isinstance(out.columns, pd.MultiIndex) and np.all(out.columns == 0):
            out.columns = pd.RangeIndex(start=0, stop=len(out.columns), step=1)
        return out

    def apply_and_concat(
        self,
        ntimes: int,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.Frame]:
        """Apply `apply_func` `ntimes` times and concatenate the results along columns.

        See `vectorbtpro.base.combining.apply_and_concat`.

        `ntimes` is the number of times to call `apply_func`, while `n_outputs` is the number of outputs to expect.

        `*args` and `**kwargs` are passed to `vectorbtpro.base.combining.apply_and_concat`.

        !!! note
            The resulted arrays to be concatenated must have the same shape as broadcast input arrays.

        Usage:
            * Using instance method:

            ```pycon
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> df.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b[i],
            ...     [1, 2, 3],
            ...     keys=['c', 'd', 'e']
            ... )
                  c       d       e
               a  b   a   b   a   b
            x  3  4   6   8   9  12
            y  5  6  10  12  15  18
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b + i,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            apply_idx        0         1         2
                       a  b  c  a  b   c  a  b   c
            x          1  2  3  2  3   4  3  4   5
            y          2  4  6  3  5   7  4  6   8
            z          3  6  9  4  7  10  5  8  11
            ```

            * To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(i, a):
            ...     time.sleep(1)
            ...     return a

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func)
            3.02 s ± 3.76 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 927 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        checks.assert_not_none(apply_func)
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, dict(ntimes=ntimes), template_context)
        args = substitute_templates(args, template_context, sub_id="args")
        kwargs = substitute_templates(kwargs, template_context, sub_id="kwargs")
        out = combining.apply_and_concat(ntimes, apply_func, broadcast_named_args["obj"], *args, **kwargs)
        if keys is not None:
            new_columns = indexes.combine_indexes([keys, wrapper.columns])
        else:
            top_columns = pd.Index(np.arange(ntimes), name="apply_idx")
            new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
        if out is None:
            return None
        wrap_kwargs = merge_dicts(dict(columns=new_columns), wrap_kwargs)
        if isinstance(out, list):
            return tuple(map(lambda x: wrapper.wrap(x, group_by=False, **wrap_kwargs), out))
        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def combine(
        cls_or_self,
        obj: tp.MaybeTupleList[tp.Union[tp.ArrayLike, "BaseAccessor"]],
        combine_func: tp.Callable,
        *args,
        allow_multiple: bool = True,
        keep_pd: bool = False,
        to_2d: bool = False,
        concat: tp.Optional[bool] = None,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Combine with `other` using `combine_func`.

        Args:
            obj (array_like): Object(s) to combine this array with.
            combine_func (callable): Function to combine two arrays.

                Can be Numba-compiled.
            *args: Variable arguments passed to `combine_func`.
            allow_multiple (bool): Whether a tuple/list/Index will be considered as multiple objects in `other`.

                Takes effect only when using the instance method.
            keep_pd (bool): Whether to keep inputs as pandas objects, otherwise convert to NumPy arrays.
            to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
            concat (bool): Whether to concatenate the results along the column axis.
                Otherwise, pairwise combine into a Series/DataFrame of the same shape.

                If True, see `vectorbtpro.base.combining.combine_and_concat`.
                If False, see `vectorbtpro.base.combining.combine_multiple`.
                If None, becomes True if there are more than two arguments to combine.

                Can only concatenate using the instance method.
            keys (index_like): Outermost column level.
            broadcast_named_args (dict): Dictionary with arguments to broadcast against each other.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            template_context (dict): Mapping used to substitute templates in `args` and `kwargs`.
            wrap_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments passed to `combine_func`.

        !!! note
            If `combine_func` is Numba-compiled, will broadcast using `WRITEABLE` and `C_CONTIGUOUS`
            flags, which can lead to an expensive computation overhead if passed objects are large and
            have different shape/memory order. You also must ensure that all objects have the same data type.

            Also remember to bring each in `*args` to a Numba-compatible format.

        Usage:
            * Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> # using instance method
            >>> sr.vbt.combine(df, np.add)
               a  b
            x  4  5
            y  7  8

            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20

            >>> sr.vbt.combine([df, df * 2], np.add)
            combine_idx     0       1
                         a  b   a   b
            x            4  5   7   9
            y            7  8  12  14

            >>> sr.vbt.combine([df, df * 2], np.add, keys=['c', 'd'])
                  c       d
               a  b   a   b
            x  4  5   7   9
            y  7  8  12  14

            >>> sr.vbt.combine(vbt.Param([1, 2], name='param'), np.add)
            param  1  2
            x      2  3
            y      3  4

            >>> # using class method
            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.combine(
            ...     [1, 2, 3],
            ...     lambda x, y, z: x + y + z,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            combine_idx        0        1        2
                         a  b  c  a  b  c  a  b  c
            x            3  4  5  4  5  6  5  6  7
            y            4  5  6  5  6  7  6  7  8
            z            5  6  7  6  7  8  7  8  9
            ```

            * To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def combine_func(a, b):
            ...     time.sleep(1)
            ...     return a + b

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func)
            3.01 s ± 2.98 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(cls_or_self, type):
            objs = obj
        else:
            if allow_multiple and isinstance(obj, (tuple, list)):
                objs = obj
            else:
                objs = (obj,)
        objs = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, objs))
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self.obj,) + objs
        if checks.is_numba_func(combine_func):
            # Numba requires writeable arrays and in the same order
            broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements=["W", "C"])), broadcast_kwargs)

        # Broadcast and substitute templates
        broadcast_named_args = {**{"obj_" + str(i): obj for i, obj in enumerate(objs)}, **broadcast_named_args}
        broadcast_named_args, wrapper = reshaping.broadcast(
            broadcast_named_args,
            return_wrapper=True,
            **broadcast_kwargs,
        )
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = substitute_templates(args, template_context, sub_id="args")
        kwargs = substitute_templates(kwargs, template_context, sub_id="kwargs")
        inputs = [broadcast_named_args["obj_" + str(i)] for i in range(len(objs))]

        if concat is None:
            concat = len(inputs) > 2
        if concat:
            # Concat the results horizontally
            if isinstance(cls_or_self, type):
                raise TypeError("Use instance method to concatenate")
            out = combining.combine_and_concat(inputs[0], inputs[1:], combine_func, *args, **kwargs)
            if keys is not None:
                new_columns = indexes.combine_indexes([keys, wrapper.columns])
            else:
                top_columns = pd.Index(np.arange(len(objs) - 1), name="combine_idx")
                new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
            return wrapper.wrap(out, **merge_dicts(dict(columns=new_columns), wrap_kwargs))
        else:
            # Combine arguments pairwise into one object
            out = combining.combine_multiple(inputs, combine_func, *args, **kwargs)
            return wrapper.wrap(out, **resolve_dict(wrap_kwargs))

    @classmethod
    def eval(
        cls,
        expr: str,
        frames_back: int = 1,
        use_numexpr: bool = False,
        numexpr_kwargs: tp.KwargsLike = None,
        local_dict: tp.Optional[tp.Mapping] = None,
        global_dict: tp.Optional[tp.Mapping] = None,
        broadcast_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ):
        """Evaluate a simple array expression element-wise using NumExpr or NumPy.

        If NumExpr is enables, only one-line statements are supported. Otherwise, uses
        `vectorbtpro.utils.eval_.multiline_eval`.

        !!! note
            All required variables will broadcast against each other prior to the evaluation.

        Usage:
            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> df = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a', 'b', 'c'])
            >>> vbt.pd_acc.eval('sr + df')
               a  b  c
            x  5  6  7
            y  6  7  8
            z  7  8  9
            ```
        """
        if numexpr_kwargs is None:
            numexpr_kwargs = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        expr = inspect.cleandoc(expr)
        parsed = ast.parse(expr)
        body_nodes = list(parsed.body)

        load_vars = set()
        store_vars = set()
        for body_node in body_nodes:
            for child_node in ast.walk(body_node):
                if type(child_node) is ast.Name:
                    if isinstance(child_node.ctx, ast.Load):
                        if child_node.id not in store_vars:
                            load_vars.add(child_node.id)
                    if isinstance(child_node.ctx, ast.Store):
                        store_vars.add(child_node.id)
        load_vars = list(load_vars)
        objs = get_context_vars(load_vars, frames_back=frames_back, local_dict=local_dict, global_dict=global_dict)
        objs = dict(zip(load_vars, objs))
        objs, wrapper = reshaping.broadcast(objs, return_wrapper=True, **broadcast_kwargs)
        objs = {k: np.asarray(v) for k, v in objs.items()}

        if use_numexpr:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("numexpr")
            import numexpr

            out = numexpr.evaluate(expr, local_dict=objs, **numexpr_kwargs)
        else:
            out = multiline_eval(expr, context=objs)
        return wrapper.wrap(out, **wrap_kwargs)


class BaseSRAccessor(BaseAccessor):
    """Accessor on top of Series.

    Accessible via `pd.Series.vbt` and all child accessors."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        if wrapper.ndim == 2:
            if wrapper.shape[1] == 1:
                wrapper = wrapper.replace(ndim=1)
            else:
                raise TypeError("Series accessors work only one one-dimensional data")

        BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)

    @class_or_instanceproperty
    def ndim(cls_or_self) -> int:
        return 1

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        return True

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        return False


class BaseDFAccessor(BaseAccessor):
    """Accessor on top of DataFrames.

    Accessible via `pd.DataFrame.vbt` and all child accessors."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        if wrapper.ndim == 1:
            wrapper = wrapper.replace(ndim=2)

        BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)

    @class_or_instanceproperty
    def ndim(cls_or_self) -> int:
        return 2

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        return False

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        return True
