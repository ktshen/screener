# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with mapped arrays.

This class takes the mapped array and the corresponding column and (optionally) index arrays,
and offers features to directly process the mapped array without converting it to pandas;
for example, to compute various statistics by column, such as standard deviation.

Consider the following example:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> import vectorbtpro as vbt

>>> a = np.array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
>>> col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
>>> idx_arr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> ma = vbt.MappedArray(wrapper, a, col_arr, idx_arr=idx_arr)
```

## Reducing

Using `MappedArray`, we can then reduce by column as follows:

* Use already provided reducers such as `MappedArray.mean`:

```pycon
>>> ma.mean()
a    11.0
b    14.0
c    17.0
dtype: float64
```

* Use `MappedArray.to_pd` to map to pandas and then reduce manually (expensive):

```pycon
>>> ma.to_pd().mean()
a    11.0
b    14.0
c    17.0
dtype: float64
```

* Use `MappedArray.reduce` to reduce using a custom function:

```pycon
>>> # Reduce to a scalar

>>> @njit
... def pow_mean_reduce_nb(a, pow):
...     return np.mean(a ** pow)

>>> ma.reduce(pow_mean_reduce_nb, 2)
a    121.666667
b    196.666667
c    289.666667
dtype: float64

>>> # Reduce to an array

>>> @njit
... def min_max_reduce_nb(a):
...     return np.array([np.min(a), np.max(a)])

>>> ma.reduce(min_max_reduce_nb, returns_array=True,
...     wrap_kwargs=dict(name_or_index=['min', 'max']))
        a     b     c
min  10.0  13.0  16.0
max  12.0  15.0  18.0

>>> # Reduce to an array of indices

>>> @njit
... def idxmin_idxmax_reduce_nb(a):
...     return np.array([np.argmin(a), np.argmax(a)])

>>> ma.reduce(idxmin_idxmax_reduce_nb, returns_array=True,
...     returns_idx=True, wrap_kwargs=dict(name_or_index=['idxmin', 'idxmax']))
        a  b  c
idxmin  x  x  x
idxmax  z  z  z

>>> # Reduce using a meta function to combine multiple mapped arrays

>>> @njit
... def mean_ratio_reduce_meta_nb(idxs, col, a, b):
...     return np.mean(a[idxs]) / np.mean(b[idxs])

>>> vbt.MappedArray.reduce(mean_ratio_reduce_meta_nb,
...     ma.values - 1, ma.values + 1, col_mapper=ma.col_mapper)
a    0.833333
b    0.866667
c    0.888889
Name: reduce, dtype: float64
```

## Mapping

Use `MappedArray.apply` to apply a function on each column/group:

```pycon
>>> @njit
... def cumsum_apply_nb(a):
...     return np.cumsum(a)

>>> ma.apply(cumsum_apply_nb)
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff061382198>

>>> ma.apply(cumsum_apply_nb).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])

>>> group_by = np.array(['first', 'first', 'second'])
>>> ma.apply(cumsum_apply_nb, group_by=group_by, apply_per_group=True).values
array([10., 21., 33., 46., 60., 75., 16., 33., 51.])

>>> # Apply using a meta function

>>> @njit
... def cumsum_apply_meta_nb(ridxs, col, a):
...     return np.cumsum(a[ridxs])

>>> vbt.MappedArray.apply(cumsum_apply_meta_nb, ma.values, col_mapper=ma.col_mapper).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])
```

Notice how cumsum resets at each column in the first example and at each group in the second example.

## Conversion

We can unstack any `MappedArray` instance to pandas:

* Given `idx_arr` was provided:

```pycon
>>> ma.to_pd()
      a     b     c
x  10.0  13.0  16.0
y  11.0  14.0  17.0
z  12.0  15.0  18.0
```

!!! note
    Will throw a warning if there are multiple values pointing to the same position.

* In case `group_by` was provided, index can be ignored, or there are position conflicts:

```pycon
>>> ma.to_pd(group_by=np.array(['first', 'first', 'second']), ignore_index=True)
   first  second
0   10.0    16.0
1   11.0    17.0
2   12.0    18.0
3   13.0     NaN
4   14.0     NaN
5   15.0     NaN
```

## Resolving conflicts

Sometimes, we may encounter multiple values for each index and column combination.
In such case, we can use `MappedArray.reduce_segments` to aggregate "duplicate" elements.
For example, let's sum up duplicate values per each index and column combination:

```pycon
>>> ma_conf = ma.replace(idx_arr=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
>>> ma_conf.to_pd()
UserWarning: Multiple values are pointing to the same position. Only the latest value is used.
      a     b     c
x  12.0   NaN   NaN
y   NaN  15.0   NaN
z   NaN   NaN  18.0

>>> @njit
... def sum_reduce_nb(a):
...     return np.sum(a)

>>> ma_no_conf = ma_conf.reduce_segments(
...     (ma_conf.idx_arr, ma_conf.col_arr),
...     sum_reduce_nb
... )
>>> ma_no_conf.to_pd()
      a     b     c
x  33.0   NaN   NaN
y   NaN  42.0   NaN
z   NaN   NaN  51.0
```

## Filtering

Use `MappedArray.apply_mask` to filter elements per column/group:

```pycon
>>> mask = [True, False, True, False, True, False, True, False, True]
>>> filtered_ma = ma.apply_mask(mask)
>>> filtered_ma.count()
a    2
b    1
c    2
dtype: int64

>>> filtered_ma.id_arr
array([0, 2, 4, 6, 8])
```

## Grouping

One of the key features of `MappedArray` is that we can perform reducing operations on a group
of columns as if they were a single column. Groups can be specified by `group_by`, which
can be anything from positions or names of column levels, to a NumPy array with actual groups.

There are multiple ways of define grouping:

* When creating `MappedArray`, pass `group_by` to `vectorbtpro.base.wrapping.ArrayWrapper`:

```pycon
>>> group_by = np.array(['first', 'first', 'second'])
>>> grouped_wrapper = wrapper.replace(group_by=group_by)
>>> grouped_ma = vbt.MappedArray(grouped_wrapper, a, col_arr, idx_arr=idx_arr)

>>> grouped_ma.mean()
first     12.5
second    17.0
dtype: float64
```

* Regroup an existing `MappedArray`:

```pycon
>>> ma.regroup(group_by).mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the reducing method:

```pycon
>>> ma.mean(group_by=group_by)
first     12.5
second    17.0
dtype: float64
```

By the same way we can disable or modify any existing grouping:

```pycon
>>> grouped_ma.mean(group_by=False)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations, there is no change to the arrays.

## Operators

`MappedArray` implements arithmetic, comparison, and logical operators. We can perform basic
operations (such as addition) on mapped arrays as if they were NumPy arrays.

```pycon
>>> ma ** 2
<vectorbtpro.records.mapped_array.MappedArray at 0x7f97bfc49358>

>>> ma * np.array([1, 2, 3, 4, 5, 6])
<vectorbtpro.records.mapped_array.MappedArray at 0x7f97bfc65e80>

>>> ma + ma
<vectorbtpro.records.mapped_array.MappedArray at 0x7fd638004d30>
```

!!! note
    Ensure that your `MappedArray` operand is on the left if the other operand is an array.

    If two `MappedArray` operands have different metadata, will copy metadata from the first one,
    but at least their `id_arr` and `col_arr` must match.

## Indexing

Like any other class subclassing `vectorbtpro.base.wrapping.Wrapping`, we can do pandas indexing
on a `MappedArray` instance, which forwards indexing operation to each object with columns:

```pycon
>>> ma['a'].values
array([10., 11., 12.])

>>> grouped_ma['first'].values
array([10., 11., 12., 13., 14., 15.])
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`
    to get the first column.

    Indexing behavior depends solely upon `vectorbtpro.base.wrapping.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns.

## Caching

`MappedArray` supports caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbtpro.utils.decorators.cached_method` and `vectorbtpro.utils.decorators.cached_property`
respectively. Caching can be disabled globally in `vectorbtpro._settings.caching`.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `MappedArray.replace` method and pass changes as keyword arguments.

## Saving and loading

Like any other class subclassing `vectorbtpro.utils.pickling.Pickleable`, we can save a `MappedArray`
instance to the disk with `MappedArray.save` and load it with `MappedArray.load`.

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `MappedArray.metrics`.

Metric for mapped arrays are similar to that for `vectorbtpro.generic.accessors.GenericAccessor`.

```pycon
>>> ma.stats(column='a')
Start                      x
End                        z
Period       3 days 00:00:00
Count                      3
Mean                    11.0
Std                      1.0
Min                     10.0
Median                  11.0
Max                     12.0
Min Index                  x
Max Index                  z
Name: a, dtype: object
```

The main difference unfolds once the mapped array has a mapping:
values are then considered as categorical and usual statistics are meaningless to compute.
For this case, `MappedArray.stats` returns the value counts:

```pycon
>>> mapping = {v: "test_" + str(v) for v in np.unique(ma.values)}
>>> ma.stats(column='a', settings=dict(mapping=mapping))
Start                                    x
End                                      z
Period                     3 days 00:00:00
Count                                    3
Value Counts: test_10.0                  1
Value Counts: test_11.0                  1
Value Counts: test_12.0                  1
Value Counts: test_13.0                  0
Value Counts: test_14.0                  0
Value Counts: test_15.0                  0
Value Counts: test_16.0                  0
Value Counts: test_17.0                  0
Value Counts: test_18.0                  0
Name: a, dtype: object

`MappedArray.stats` also supports (re-)grouping:

```pycon
>>> grouped_ma.stats(column='first')
Start                      x
End                        z
Period       3 days 00:00:00
Count                      6
Mean                    12.5
Std                 1.870829
Min                     10.0
Median                  12.5
Max                     15.0
Min Index                  x
Max Index                  z
Name: first, dtype: object
```

## Plots

We can build histograms and boxplots of `MappedArray` directly:

```pycon
>>> ma.boxplot().show()
```

![](/assets/images/api/mapped_boxplot.svg){: .iimg loading=lazy }

To use scatterplots or any other plots that require index, convert to pandas first:

```pycon
>>> ma.to_pd().vbt.plot().show()
```

![](/assets/images/api/mapped_to_pd_plot.svg){: .iimg loading=lazy }

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `MappedArray.subplots`.

`MappedArray` class has a single subplot based on `MappedArray.to_pd` and
`vectorbtpro.generic.accessors.GenericAccessor.plot`.
"""

import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_dict
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.records import nb
from vectorbtpro.records.col_mapper import ColumnMapper
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import index_repeating_rows_nb
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.decorators import class_or_instancemethod, cached_method
from vectorbtpro.utils.magic_decorators import attach_binary_magic_methods, attach_unary_magic_methods
from vectorbtpro.utils.mapping import to_value_mapping, apply_mapping

__all__ = [
    "MappedArray",
]

MappedArrayT = tp.TypeVar("MappedArrayT", bound="MappedArray")


def combine_mapped_with_other(
    self: MappedArrayT,
    other: tp.Union["MappedArray", tp.ArrayLike],
    np_func: tp.Callable[[tp.ArrayLike, tp.ArrayLike], tp.Array1d],
) -> MappedArrayT:
    """Combine `MappedArray` with other compatible object.

    If other object is also `MappedArray`, their `id_arr` and `col_arr` must match."""
    if isinstance(other, MappedArray):
        checks.assert_array_equal(self.id_arr, other.id_arr)
        checks.assert_array_equal(self.col_arr, other.col_arr)
        other = other.values
    return self.replace(mapped_arr=np_func(self.values, other))


@attach_binary_magic_methods(combine_mapped_with_other)
@attach_unary_magic_methods(lambda self, np_func: self.replace(mapped_arr=np_func(self.values)))
class MappedArray(Analyzable):
    """Exposes methods for reducing, converting, and plotting arrays mapped by
    `vectorbtpro.records.base.Records` class.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        mapped_arr (array_like): A one-dimensional array of mapped record values.
        col_arr (array_like): A one-dimensional column array.

            Must be of the same size as `mapped_arr`.
        id_arr (array_like): A one-dimensional id array. Defaults to simple range.

            Must be of the same size as `mapped_arr`.
        idx_arr (array_like): A one-dimensional index array. Optional.

            Must be of the same size as `mapped_arr`.
        mapping (namedtuple, dict or callable): Mapping.
        col_mapper (ColumnMapper): Column mapper if already known.

            !!! note
                It depends upon `wrapper` and `col_arr`, so make sure to invalidate `col_mapper` upon creating
                a `MappedArray` instance with a modified `wrapper` or `col_arr.

                `MappedArray.replace` does it automatically.
        **kwargs: Custom keyword arguments passed to the config.

            Useful if any subclass wants to extend the config.
    """

    @classmethod
    def row_stack(
        cls: tp.Type[MappedArrayT],
        *objs: tp.MaybeTuple[MappedArrayT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MappedArrayT:
        """Stack multiple `MappedArray` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, MappedArray):
                raise TypeError("Each object to be merged must be an instance of MappedArray")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        if "col_mapper" not in kwargs:
            kwargs["col_mapper"] = ColumnMapper.row_stack(
                *[obj.col_mapper for obj in objs],
                wrapper=kwargs["wrapper"],
            )
        if "mapped_arr" not in kwargs:
            mapped_arrs = []
            for col in range(kwargs["wrapper"].shape_2d[1]):
                for obj in objs:
                    col_idxs, col_lens = obj.col_mapper.col_map
                    if len(col_idxs) > 0:
                        if col > 0 and obj.wrapper.shape_2d[1] == 1:
                            mapped_arrs.append(obj.mapped_arr[col_idxs])
                        elif col_lens[col] > 0:
                            col_end_idxs = np.cumsum(col_lens)
                            col_start_idxs = col_end_idxs - col_lens
                            mapped_arrs.append(obj.mapped_arr[col_idxs[col_start_idxs[col] : col_end_idxs[col]]])
            kwargs["mapped_arr"] = np.concatenate(mapped_arrs)
        if "col_arr" not in kwargs:
            kwargs["col_arr"] = kwargs["col_mapper"].col_arr
        if "idx_arr" not in kwargs:
            stack_idx_arrs = True
            for obj in objs:
                if obj.idx_arr is None:
                    stack_idx_arrs = False
                    break
            if stack_idx_arrs:
                idx_arrs = []
                for col in range(kwargs["wrapper"].shape_2d[1]):
                    n_rows_sum = 0
                    for obj in objs:
                        col_idxs, col_lens = obj.col_mapper.col_map
                        if len(col_idxs) > 0:
                            if col > 0 and obj.wrapper.shape_2d[1] == 1:
                                idx_arrs.append(obj.idx_arr[col_idxs] + n_rows_sum)
                            elif col_lens[col] > 0:
                                col_end_idxs = np.cumsum(col_lens)
                                col_start_idxs = col_end_idxs - col_lens
                                col_idx_arr = obj.idx_arr[col_idxs[col_start_idxs[col] : col_end_idxs[col]]]
                                idx_arrs.append(col_idx_arr + n_rows_sum)
                        n_rows_sum += obj.wrapper.shape_2d[0]
                kwargs["idx_arr"] = np.concatenate(idx_arrs)
        if "id_arr" not in kwargs:
            id_arrs = []
            for col in range(kwargs["wrapper"].shape_2d[1]):
                from_id = 0
                for obj in objs:
                    col_idxs, col_lens = obj.col_mapper.col_map
                    if len(col_idxs) > 0:
                        if col > 0 and obj.wrapper.shape_2d[1] == 1:
                            id_arrs.append(obj.id_arr[col_idxs] + from_id)
                        elif col_lens[col] > 0:
                            col_end_idxs = np.cumsum(col_lens)
                            col_start_idxs = col_end_idxs - col_lens
                            id_arrs.append(obj.id_arr[col_idxs[col_start_idxs[col] : col_end_idxs[col]]] + from_id)
                        if len(id_arrs) > 0 and len(id_arrs[-1]) > 0:
                            from_id = id_arrs[-1].max() + 1
            kwargs["id_arr"] = np.concatenate(id_arrs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[MappedArrayT],
        *objs: tp.MaybeTuple[MappedArrayT],
        wrapper_kwargs: tp.KwargsLike = None,
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MappedArrayT:
        """Stack multiple `MappedArray` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        `get_indexer_kwargs` are passed to
        [pandas.Index.get_indexer](https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html)
        to translate old indices to new ones after the reindexing operation.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, MappedArray):
                raise TypeError("Each object to be merged must be an instance of MappedArray")
        if get_indexer_kwargs is None:
            get_indexer_kwargs = {}
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "col_mapper" not in kwargs:
            kwargs["col_mapper"] = ColumnMapper.column_stack(
                *[obj.col_mapper for obj in objs],
                wrapper=kwargs["wrapper"],
            )
        if "mapped_arr" not in kwargs:
            mapped_arrs = []
            for obj in objs:
                col_idxs, col_lens = obj.col_mapper.col_map
                if len(col_idxs) > 0:
                    mapped_arrs.append(obj.mapped_arr[col_idxs])
            kwargs["mapped_arr"] = np.concatenate(mapped_arrs)
        if "col_arr" not in kwargs:
            kwargs["col_arr"] = kwargs["col_mapper"].col_arr
        if "idx_arr" not in kwargs:
            stack_idx_arrs = True
            for obj in objs:
                if obj.idx_arr is None:
                    stack_idx_arrs = False
                    break
            if stack_idx_arrs:
                idx_arrs = []
                for obj in objs:
                    col_idxs, col_lens = obj.col_mapper.col_map
                    if len(col_idxs) > 0:
                        old_idxs = obj.idx_arr[col_idxs]
                        if not obj.wrapper.index.equals(kwargs["wrapper"].index):
                            new_idxs = kwargs["wrapper"].index.get_indexer(
                                obj.wrapper.index[old_idxs],
                                **get_indexer_kwargs,
                            )
                        else:
                            new_idxs = old_idxs
                        idx_arrs.append(new_idxs)
                kwargs["idx_arr"] = np.concatenate(idx_arrs)
        if "id_arr" not in kwargs:
            id_arrs = []
            for obj in objs:
                col_idxs, col_lens = obj.col_mapper.col_map
                if len(col_idxs) > 0:
                    id_arrs.append(obj.id_arr[col_idxs])
            kwargs["id_arr"] = np.concatenate(id_arrs)

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "mapped_arr",
        "col_arr",
        "idx_arr",
        "id_arr",
        "mapping",
        "col_mapper",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        mapped_arr: tp.ArrayLike,
        col_arr: tp.ArrayLike,
        idx_arr: tp.Optional[tp.ArrayLike] = None,
        id_arr: tp.Optional[tp.ArrayLike] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> None:

        mapped_arr = np.asarray(mapped_arr)
        col_arr = np.asarray(col_arr)
        checks.assert_shape_equal(mapped_arr, col_arr, axis=0)
        if idx_arr is not None:
            idx_arr = np.asarray(idx_arr)
            checks.assert_shape_equal(mapped_arr, idx_arr, axis=0)
        if col_mapper is None:
            col_mapper = ColumnMapper(wrapper, col_arr)
        if id_arr is None:
            id_arr = col_mapper.new_id_arr
        else:
            id_arr = np.asarray(id_arr)
            checks.assert_shape_equal(mapped_arr, id_arr, axis=0)

        Analyzable.__init__(
            self,
            wrapper,
            mapped_arr=mapped_arr,
            col_arr=col_arr,
            idx_arr=idx_arr,
            id_arr=id_arr,
            mapping=mapping,
            col_mapper=col_mapper,
            **kwargs,
        )

        self._mapped_arr = mapped_arr
        self._col_arr = col_arr
        self._idx_arr = idx_arr
        self._id_arr = id_arr
        self._mapping = mapping
        self._col_mapper = col_mapper

        # Only slices of rows can be selected
        self._range_only_select = True

    def replace(self: MappedArrayT, **kwargs) -> MappedArrayT:
        """See `vectorbtpro.utils.config.Configured.replace`.

        Also, makes sure that `MappedArray.col_mapper` is not passed to the new instance."""
        if self.config.get("col_mapper", None) is not None:
            if "wrapper" in kwargs:
                if self.wrapper is not kwargs.get("wrapper"):
                    kwargs["col_mapper"] = None
            if "col_arr" in kwargs:
                if self.col_arr is not kwargs.get("col_arr"):
                    kwargs["col_mapper"] = None
        return Analyzable.replace(self, **kwargs)

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `MappedArray` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                range_only_select=self.range_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        new_indices, new_col_arr = self.col_mapper.select_cols(wrapper_meta["col_idxs"])
        new_mapped_arr = self.values[new_indices]
        if self.idx_arr is not None:
            new_idx_arr = self.idx_arr[new_indices]
        else:
            new_idx_arr = None
        new_id_arr = self.id_arr[new_indices]
        if wrapper_meta["rows_changed"] and new_idx_arr is not None:
            row_idxs = wrapper_meta["row_idxs"]
            mask = (new_idx_arr >= row_idxs.start) & (new_idx_arr < row_idxs.stop)
            new_indices = new_indices[mask]
            new_mapped_arr = new_mapped_arr[mask]
            new_col_arr = new_col_arr[mask]
            if new_idx_arr is not None:
                new_idx_arr = new_idx_arr[mask] - row_idxs.start
            new_id_arr = new_id_arr[mask]
        return dict(
            wrapper_meta=wrapper_meta,
            new_indices=new_indices,
            new_mapped_arr=new_mapped_arr,
            new_col_arr=new_col_arr,
            new_idx_arr=new_idx_arr,
            new_id_arr=new_id_arr,
        )

    def indexing_func(self: MappedArrayT, *args, mapped_meta: tp.DictLike = None, **kwargs) -> MappedArrayT:
        """Perform indexing on `MappedArray`."""
        if mapped_meta is None:
            mapped_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=mapped_meta["wrapper_meta"]["new_wrapper"],
            mapped_arr=mapped_meta["new_mapped_arr"],
            col_arr=mapped_meta["new_col_arr"],
            id_arr=mapped_meta["new_id_arr"],
            idx_arr=mapped_meta["new_idx_arr"],
        )

    def resample_meta(self: MappedArrayT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform resampling on `MappedArray` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        if isinstance(wrapper_meta["resampler"], Resampler):
            _resampler = wrapper_meta["resampler"]
        else:
            _resampler = Resampler.from_pd_resampler(wrapper_meta["resampler"])
        if self.idx_arr is not None:
            index_map = _resampler.map_to_target_index(return_index=False)
            new_idx_arr = index_map[self.idx_arr]
        else:
            new_idx_arr = None
        return dict(wrapper_meta=wrapper_meta, new_idx_arr=new_idx_arr)

    def resample(self: MappedArrayT, *args, mapped_meta: tp.DictLike = None, **kwargs) -> MappedArrayT:
        """Perform resampling on `MappedArray`."""
        if mapped_meta is None:
            mapped_meta = self.resample_meta(*args, **kwargs)
        return self.replace(
            wrapper=mapped_meta["wrapper_meta"]["new_wrapper"],
            idx_arr=mapped_meta["new_idx_arr"],
        )

    @property
    def mapped_arr(self) -> tp.Array1d:
        """Mapped array."""
        return self._mapped_arr

    @property
    def values(self) -> tp.Array1d:
        """Mapped array."""
        return self.mapped_arr

    def to_readable(self, title: str = "Value", only_values: bool = False, **kwargs) -> tp.SeriesFrame:
        """Get values in a human-readable format."""
        values = pd.Series(self.apply_mapping(**kwargs).values, name=title)
        if only_values:
            return pd.Series(values, name=title)
        columns = list()
        columns.append(pd.Series(self.id_arr, name="Id"))
        columns.append(pd.Series(self.wrapper.columns[self.col_arr], name="Column"))
        if self.idx_arr is not None:
            columns.append(pd.Series(self.wrapper.index[self.idx_arr], name="Index"))
        columns.append(values)
        return pd.concat(columns, axis=1)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array."""
        return self._col_arr

    @property
    def col_mapper(self) -> ColumnMapper:
        """Column mapper.

        See `vectorbtpro.records.col_mapper.ColumnMapper`."""
        return self._col_mapper

    @property
    def idx_arr(self) -> tp.Optional[tp.Array1d]:
        """Index array."""
        return self._idx_arr

    @property
    def id_arr(self) -> tp.Array1d:
        """Id array."""
        return self._id_arr

    @property
    def mapping(self) -> tp.Optional[tp.MappingLike]:
        """Mapping."""
        return self._mapping

    # ############# Sorting ############# #

    @cached_method
    def is_sorted(self, incl_id: bool = False, jitted: tp.JittedOption = None) -> bool:
        """Check whether mapped array is sorted."""
        if incl_id:
            func = jit_reg.resolve_option(nb.is_col_id_sorted_nb, jitted)
            return func(self.col_arr, self.id_arr)
        func = jit_reg.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)

    def sort(
        self: MappedArrayT,
        incl_id: bool = False,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArrayT:
        """Sort mapped array by column array (primary) and id array (secondary, optional).

        `**kwargs` are passed to `MappedArray.replace`."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        if self.is_sorted(incl_id=incl_id):
            return self.replace(idx_arr=idx_arr, **kwargs).regroup(group_by)
        if incl_id:
            ind = np.lexsort((self.id_arr, self.col_arr))  # expensive!
        else:
            ind = np.argsort(self.col_arr)
        return self.replace(
            mapped_arr=self.values[ind],
            col_arr=self.col_arr[ind],
            id_arr=self.id_arr[ind],
            idx_arr=idx_arr[ind] if idx_arr is not None else None,
            **kwargs,
        ).regroup(group_by)

    # ############# Filtering ############# #

    def apply_mask(
        self: MappedArrayT,
        mask: tp.Array1d,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArrayT:
        """Return a new class instance, filtered by mask.

        `**kwargs` are passed to `MappedArray.replace`."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        mask_indices = np.flatnonzero(mask)
        return self.replace(
            mapped_arr=np.take(self.values, mask_indices),
            col_arr=np.take(self.col_arr, mask_indices),
            id_arr=np.take(self.id_arr, mask_indices),
            idx_arr=np.take(idx_arr, mask_indices) if idx_arr is not None else None,
            **kwargs,
        ).regroup(group_by)

    def top_n_mask(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
    ) -> tp.Array1d:
        """Return mask of top N elements in each column/group."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.top_n_mapped_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return func(self.values, col_map, n)

    def bottom_n_mask(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
    ) -> tp.Array1d:
        """Return mask of bottom N elements in each column/group."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.bottom_n_mapped_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return func(self.values, col_map, n)

    def top_n(
        self: MappedArrayT,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArrayT:
        """Filter top N elements from each column/group."""
        return self.apply_mask(self.top_n_mask(n, group_by=group_by, jitted=jitted, chunked=chunked), **kwargs)

    def bottom_n(
        self: MappedArrayT,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArrayT:
        """Filter bottom N elements from each column/group."""
        return self.apply_mask(self.bottom_n_mask(n, group_by=group_by, jitted=jitted, chunked=chunked), **kwargs)

    # ############# Mapping ############# #

    def resolve_mapping(self, mapping: tp.Union[None, bool, tp.MappingLike] = None) -> tp.Optional[tp.Mapping]:
        """Resolve mapping.

        Set `mapping` to False to disable mapping completely."""
        if mapping is None or mapping is True:
            mapping = self.mapping
        if isinstance(mapping, bool):
            if not mapping:
                return None
        if isinstance(mapping, str):
            if mapping.lower() == "index":
                mapping = self.wrapper.index
            elif mapping.lower() == "columns":
                mapping = self.wrapper.columns
            elif mapping.lower() == "groups":
                mapping = self.wrapper.get_columns()
            mapping = to_value_mapping(mapping)
        return mapping

    def apply_mapping(
        self: MappedArrayT,
        mapping: tp.Union[None, bool, tp.MappingLike] = None,
        mapping_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MappedArrayT:
        """Apply mapping on each element."""
        mapping = self.resolve_mapping(mapping)
        new_mapped_arr = apply_mapping(self.values, mapping, **resolve_dict(mapping_kwargs))
        return self.replace(mapped_arr=new_mapped_arr, **kwargs)

    def to_index(self, minus_one_to_zero: bool = False) -> tp.Index:
        """Convert to index.

        If `minus_one_to_zero` is True, index -1 will automatically become 0.
        Otherwise, will throw an error."""
        if np.isin(-1, self.values):
            nan_mask = self.values == -1
            values = self.values.copy()
            values[nan_mask] = 0
            if minus_one_to_zero:
                return self.wrapper.index[values]
            if self.wrapper.index.is_integer():
                new_values = self.wrapper.index.values[values]
                new_values[nan_mask] = -1
                return pd.Index(new_values, name=self.wrapper.index.name)
            if isinstance(self.wrapper.index, pd.DatetimeIndex):
                new_values = self.wrapper.index.values[values]
                new_values[nan_mask] = np.datetime64("NaT")
                return pd.Index(new_values, name=self.wrapper.index.name)
            new_values = self.wrapper.index.values[values]
            new_values[nan_mask] = np.nan
            return pd.Index(new_values, name=self.wrapper.index.name)
        return self.wrapper.index[self.values]

    def to_columns(self) -> tp.Index:
        """Convert to columns."""
        if np.isin(-1, self.values):
            raise ValueError("Cannot get index at position -1")
        return self.wrapper.columns[self.values]

    @class_or_instancemethod
    def apply(
        cls_or_self: tp.Union[tp.Type[MappedArrayT], MappedArrayT],
        apply_func_nb: tp.Union[tp.ApplyFunc, tp.ApplyMetaFunc],
        *args,
        group_by: tp.GroupByLike = None,
        apply_per_group: bool = False,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> MappedArrayT:
        """Apply function on mapped array per column/group. Returns a new mapped array.

        Applies per group of columns if `apply_per_group` is True.

        See `vectorbtpro.records.nb.apply_nb`.

        For details on the meta version, see `vectorbtpro.records.nb.apply_meta_nb`.

        `**kwargs` are passed to `MappedArray.replace`."""
        if isinstance(cls_or_self, type):
            checks.assert_not_none(col_mapper)
            col_map = col_mapper.get_col_map(group_by=group_by if apply_per_group else False)
            func = jit_reg.resolve_option(nb.apply_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(len(col_mapper.col_arr), col_map, apply_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return MappedArray(col_mapper.wrapper, mapped_arr, col_mapper.col_arr, col_mapper=col_mapper, **kwargs)
        else:
            col_map = cls_or_self.col_mapper.get_col_map(group_by=group_by if apply_per_group else False)
            func = jit_reg.resolve_option(nb.apply_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(cls_or_self.values, col_map, apply_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return cls_or_self.replace(mapped_arr=mapped_arr, **kwargs).regroup(group_by)

    # ############# Reducing ############# #

    def reduce_segments(
        self: MappedArrayT,
        segment_arr: tp.Union[str, tp.MaybeTuple[tp.Array1d]],
        reduce_func_nb: tp.Union[str, tp.ReduceFunc],
        *args,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        apply_per_group: bool = False,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArrayT:
        """Reduce each segment of values in mapped array. Returns a new mapped array.

        `segment_arr` must be an array of integers increasing per column, each indicating a segment.
        It must have the same length as the mapped array. You can also pass a list of such arrays.
        In this case, each unique combination of values will be considered a single segment.
        Can also pass the string "idx" to use the index array.

        `reduce_func_nb` can be a string denoting the suffix of a reducing function
        from `vectorbtpro.generic.nb`. For example, "sum" will refer to "sum_reduce_nb".

        !!! warning
            Each segment or combination of segments in `segment_arr` is assumed to be coherent and non-repeating.
            That is, `np.array([0, 1, 0])` for a single column annotates three different segments, not two.
            See `vectorbtpro.utils.array_.index_repeating_rows_nb`.

        !!! hint
            Use `MappedArray.sort` to bring the mapped array to the desired order, if required.

        Applies per group of columns if `apply_per_group` is True.

        See `vectorbtpro.records.nb.reduce_mapped_segments_nb`.

        `**kwargs` are passed to `MappedArray.replace`."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        col_map = self.col_mapper.get_col_map(group_by=group_by if apply_per_group else False)
        if isinstance(segment_arr, str):
            if segment_arr.lower() == "idx":
                segment_arr = idx_arr
            else:
                raise ValueError(f"Invalid option segment_arr='{segment_arr}'")
        if isinstance(segment_arr, tuple):
            stacked_segment_arr = np.column_stack(segment_arr)
            segment_arr = index_repeating_rows_nb(stacked_segment_arr)
        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(generic_nb, reduce_func_nb + "_reduce_nb")

        func = jit_reg.resolve_option(nb.reduce_mapped_segments_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_mapped_arr, new_col_arr, new_idx_arr, new_id_arr = func(
            self.values,
            idx_arr,
            self.id_arr,
            col_map,
            segment_arr,
            reduce_func_nb,
            *args,
        )
        new_mapped_arr = np.asarray(new_mapped_arr, dtype=dtype)
        return self.replace(
            mapped_arr=new_mapped_arr,
            col_arr=new_col_arr,
            idx_arr=new_idx_arr,
            id_arr=new_id_arr,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def reduce(
        cls_or_self,
        reduce_func_nb: tp.Union[
            tp.ReduceFunc, tp.MappedReduceMetaFunc, tp.ReduceToArrayFunc, tp.MappedReduceToArrayMetaFunc
        ],
        *args,
        idx_arr: tp.Optional[tp.Array1d] = None,
        returns_array: bool = False,
        returns_idx: bool = False,
        to_index: bool = True,
        fill_value: tp.Scalar = np.nan,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeriesFrame:
        """Reduce mapped array by column/group.

        Set `returns_array` to True if `reduce_func_nb` returns an array.

        Set `returns_idx` to True if `reduce_func_nb` returns row index/position. Must pass `idx_arr`.

        Set `to_index` to True to return labels instead of positions.

        Use `fill_value` to set the default value.

        For implementation details, see

        * `vectorbtpro.records.nb.reduce_mapped_nb` if `returns_array` is False and `returns_idx` is False
        * `vectorbtpro.records.nb.reduce_mapped_to_idx_nb` if `returns_array` is False and `returns_idx` is True
        * `vectorbtpro.records.nb.reduce_mapped_to_array_nb` if `returns_array` is True and `returns_idx` is False
        * `vectorbtpro.records.nb.reduce_mapped_to_idx_array_nb` if `returns_array` is True and `returns_idx` is True

        For implementation details on the meta versions, see

        * `vectorbtpro.records.nb.reduce_mapped_meta_nb` if `returns_array` is False and `returns_idx` is False
        * `vectorbtpro.records.nb.reduce_mapped_to_idx_meta_nb` if `returns_array` is False and `returns_idx` is True
        * `vectorbtpro.records.nb.reduce_mapped_to_array_meta_nb` if `returns_array` is True and `returns_idx` is False
        * `vectorbtpro.records.nb.reduce_mapped_to_idx_array_meta_nb` if `returns_array` is True and `returns_idx` is True
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(col_mapper)
            col_map = col_mapper.get_col_map(group_by=group_by)
            if not returns_array:
                if not returns_idx:
                    func = jit_reg.resolve_option(nb.reduce_mapped_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(col_map, fill_value, reduce_func_nb, *args)
                else:
                    checks.assert_not_none(idx_arr)
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_idx_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(col_map, idx_arr, fill_value, reduce_func_nb, *args)
            else:
                if not returns_idx:
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_array_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(col_map, fill_value, reduce_func_nb, *args)
                else:
                    checks.assert_not_none(idx_arr)
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_idx_array_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(col_map, idx_arr, fill_value, reduce_func_nb, *args)
            wrapper = col_mapper.wrapper
        else:
            if idx_arr is None:
                if cls_or_self.idx_arr is None:
                    if returns_idx:
                        raise ValueError("Must pass idx_arr")
                idx_arr = cls_or_self.idx_arr
            col_map = cls_or_self.col_mapper.get_col_map(group_by=group_by)
            if not returns_array:
                if not returns_idx:
                    func = jit_reg.resolve_option(nb.reduce_mapped_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(cls_or_self.values, col_map, fill_value, reduce_func_nb, *args)
                else:
                    checks.assert_not_none(idx_arr)
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_idx_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(cls_or_self.values, col_map, idx_arr, fill_value, reduce_func_nb, *args)
            else:
                if not returns_idx:
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_array_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(cls_or_self.values, col_map, fill_value, reduce_func_nb, *args)
                else:
                    checks.assert_not_none(idx_arr)
                    func = jit_reg.resolve_option(nb.reduce_mapped_to_idx_array_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(cls_or_self.values, col_map, idx_arr, fill_value, reduce_func_nb, *args)
            wrapper = cls_or_self.wrapper

        wrap_kwargs = merge_dicts(
            dict(
                name_or_index="reduce" if not returns_array else None,
                to_index=returns_idx and to_index,
                fillna=-1 if returns_idx else None,
                dtype=np.int_ if returns_idx else None,
            ),
            wrap_kwargs,
        )
        return wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    @cached_method
    def nth(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return n-th element of each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="nth"), wrap_kwargs)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.reduce(
            jit_reg.resolve_option(generic_nb.nth_reduce_nb, jitted),
            n,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def nth_index(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return index of n-th element of each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="nth_index"), wrap_kwargs)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.reduce(
            jit_reg.resolve_option(generic_nb.nth_index_reduce_nb, jitted),
            n,
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def min(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return min by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="min"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.min_reduce_nb, jitted),
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def max(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return max by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="max"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.max_reduce_nb, jitted),
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def mean(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return mean by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="mean"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.mean_reduce_nb, jitted),
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def median(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return median by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="median"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.median_reduce_nb, jitted),
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def std(
        self,
        ddof: int = 1,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return std by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="std"), wrap_kwargs)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.reduce(
            jit_reg.resolve_option(generic_nb.std_reduce_nb, jitted),
            ddof,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def sum(
        self,
        fill_value: tp.Scalar = 0.0,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return sum by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="sum"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.sum_reduce_nb, jitted),
            fill_value=fill_value,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def idxmin(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return index of min by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="idxmin"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.argmin_reduce_nb, jitted),
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def idxmax(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return index of max by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="idxmax"), wrap_kwargs)
        return self.reduce(
            jit_reg.resolve_option(generic_nb.argmax_reduce_nb, jitted),
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @cached_method
    def describe(
        self,
        percentiles: tp.Optional[tp.ArrayLike] = None,
        ddof: int = 1,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return statistics by column/group."""
        if percentiles is not None:
            percentiles = to_1d_array(percentiles)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        percentiles = percentiles.tolist()
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.unique(percentiles)
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        index = pd.Index(["count", "mean", "std", "min", *perc_formatted, "max"])
        wrap_kwargs = merge_dicts(dict(name_or_index=index), wrap_kwargs)
        chunked = ch.specialize_chunked_option(chunked, arg_take_spec=dict(args=ch.ArgsTaker(None, None)))
        out = self.reduce(
            jit_reg.resolve_option(generic_nb.describe_reduce_nb, jitted),
            percentiles,
            ddof,
            returns_array=True,
            returns_idx=False,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )
        if wrap_kwargs.get("to_timedelta", False):
            out.drop("count", axis=0, inplace=True)
        else:
            if isinstance(out, pd.DataFrame):
                out.loc["count"].fillna(0.0, inplace=True)
            else:
                if np.isnan(out.loc["count"]):
                    out.loc["count"] = 0.0
        return out

    @cached_method
    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Return number of values by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="count"), wrap_kwargs)
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by,
            **wrap_kwargs,
        )

    # ############# Value counts ############# #

    @cached_method
    def value_counts(
        self,
        axis: int = 1,
        idx_arr: tp.Optional[tp.Array1d] = None,
        normalize: bool = False,
        sort_uniques: bool = True,
        sort: bool = False,
        ascending: bool = False,
        dropna: bool = False,
        group_by: tp.GroupByLike = None,
        mapping: tp.Union[None, bool, tp.MappingLike] = None,
        incl_all_keys: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.generic.accessors.GenericAccessor.value_counts`."""
        checks.assert_in(axis, (-1, 0, 1))

        mapping = self.resolve_mapping(mapping)
        mapped_codes, mapped_uniques = pd.factorize(self.values, sort=False, use_na_sentinel=False)
        if axis == 0:
            if idx_arr is None:
                idx_arr = self.idx_arr
            checks.assert_not_none(idx_arr)
            func = jit_reg.resolve_option(nb.mapped_value_counts_per_row_nb, jitted)
            value_counts = func(mapped_codes, len(mapped_uniques), idx_arr, self.wrapper.shape[0])
        elif axis == 1:
            col_map = self.col_mapper.get_col_map(group_by=group_by)
            func = jit_reg.resolve_option(nb.mapped_value_counts_per_col_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            value_counts = func(mapped_codes, len(mapped_uniques), col_map)
        else:
            func = jit_reg.resolve_option(nb.mapped_value_counts_nb, jitted)
            value_counts = func(mapped_codes, len(mapped_uniques))
        if incl_all_keys and mapping is not None:
            missing_keys = []
            for x in mapping:
                if pd.isnull(x) and pd.isnull(mapped_uniques).any():
                    continue
                if x not in mapped_uniques:
                    missing_keys.append(x)
            if axis == 0 or axis == 1:
                value_counts = np.vstack((value_counts, np.full((len(missing_keys), value_counts.shape[1]), 0)))
            else:
                value_counts = np.concatenate((value_counts, np.full(len(missing_keys), 0)))
            mapped_uniques = np.concatenate((mapped_uniques, np.array(missing_keys)))
        nan_mask = np.isnan(mapped_uniques)
        if dropna:
            value_counts = value_counts[~nan_mask]
            mapped_uniques = mapped_uniques[~nan_mask]
        if sort_uniques:
            new_indices = mapped_uniques.argsort()
            value_counts = value_counts[new_indices]
            mapped_uniques = mapped_uniques[new_indices]
        if axis == 0 or axis == 1:
            value_counts_sum = value_counts.sum(axis=1)
        else:
            value_counts_sum = value_counts
        if normalize:
            value_counts = value_counts / value_counts_sum.sum()
        if sort:
            if ascending:
                new_indices = value_counts_sum.argsort()
            else:
                new_indices = (-value_counts_sum).argsort()
            value_counts = value_counts[new_indices]
            mapped_uniques = mapped_uniques[new_indices]
        if axis == 0:
            wrapper = ArrayWrapper.from_obj(value_counts)
            value_counts_pd = wrapper.wrap(
                value_counts,
                index=mapped_uniques,
                columns=self.wrapper.index,
                **resolve_dict(wrap_kwargs),
            )
        elif axis == 1:
            value_counts_pd = self.wrapper.wrap(
                value_counts,
                index=mapped_uniques,
                group_by=group_by,
                **resolve_dict(wrap_kwargs),
            )
        else:
            wrapper = ArrayWrapper.from_obj(value_counts)
            value_counts_pd = wrapper.wrap(
                value_counts,
                index=mapped_uniques,
                **merge_dicts(dict(columns=["value_counts"]), wrap_kwargs),
            )
        if mapping is not None:
            value_counts_pd.index = apply_mapping(value_counts_pd.index, mapping, **kwargs)
        return value_counts_pd

    # ############# Conflicts ############# #

    @cached_method
    def has_conflicts(
        self,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
    ) -> bool:
        """See `vectorbtpro.records.nb.mapped_has_conflicts_nb`."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        func = jit_reg.resolve_option(nb.mapped_has_conflicts_nb, jitted)
        return func(col_arr, idx_arr, target_shape)

    def coverage_map(
        self,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.records.nb.mapped_coverage_map_nb`."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        func = jit_reg.resolve_option(nb.mapped_coverage_map_nb, jitted)
        out = func(col_arr, idx_arr, target_shape)
        return self.wrapper.wrap(out, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Unstacking ############# #

    def to_pd(
        self,
        idx_arr: tp.Optional[tp.Array1d] = None,
        reduce_func_nb: tp.Union[None, str, tp.ReduceFunc] = None,
        reduce_args: tp.ArgsLike = None,
        dtype: tp.Optional[tp.DTypeLike] = None,
        ignore_index: bool = False,
        repeat_index: bool = False,
        fill_value: float = np.nan,
        mapping: tp.Union[None, bool, tp.MappingLike] = False,
        mapping_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        silence_warnings: bool = False,
    ) -> tp.SeriesFrame:
        """Unstack mapped array to a Series/DataFrame.

        If `reduce_func_nb` is not None, will use it to reduce conflicting index segments
        using `MappedArray.reduce_segments`.

        * If `ignore_index`, will ignore the index and place values on top of each other in every column/group.
            See `vectorbtpro.records.nb.ignore_unstack_mapped_nb`.
        * If `repeat_index`, will repeat any index pointed from multiple values.
            Otherwise, in case of positional conflicts, will throw a warning and use the latest value.
            See `vectorbtpro.records.nb.repeat_unstack_mapped_nb`.
        * Otherwise, see `vectorbtpro.records.nb.unstack_mapped_nb`.

        !!! note
            Will raise an error if there are multiple values pointing to the same position.
            Set `ignore_index` to True in this case.

        !!! warning
            Mapped arrays represent information in the most memory-friendly format.
            Mapping back to pandas may occupy lots of memory if records are sparse."""
        if ignore_index:
            if self.wrapper.ndim == 1:
                return self.wrapper.wrap(
                    self.values,
                    index=np.arange(len(self.values)),
                    group_by=group_by,
                    **resolve_dict(wrap_kwargs),
                )
            col_map = self.col_mapper.get_col_map(group_by=group_by)
            func = jit_reg.resolve_option(nb.ignore_unstack_mapped_nb, jitted)
            out = func(self.values, col_map, fill_value)
            mapping = self.resolve_mapping(mapping)
            out = apply_mapping(out, mapping, **resolve_dict(mapping_kwargs))
            return self.wrapper.wrap(out, index=np.arange(out.shape[0]), group_by=group_by, **resolve_dict(wrap_kwargs))
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        has_conflicts = self.has_conflicts(idx_arr=idx_arr, group_by=group_by)
        if has_conflicts and repeat_index:
            col_arr = self.col_mapper.get_col_arr(group_by=group_by)
            target_shape = self.wrapper.get_shape_2d(group_by=group_by)
            func = jit_reg.resolve_option(nb.mapped_coverage_map_nb, jitted)
            coverage_map = func(col_arr, idx_arr, target_shape)
            repeat_cnt_arr = np.max(coverage_map, axis=1)
            func = jit_reg.resolve_option(nb.unstack_index_nb, jitted)
            unstacked_index = self.wrapper.index[func(repeat_cnt_arr)]
            func = jit_reg.resolve_option(nb.repeat_unstack_mapped_nb, jitted)
            out = func(self.values, col_arr, idx_arr, repeat_cnt_arr, target_shape[1], fill_value)
            mapping = self.resolve_mapping(mapping)
            out = apply_mapping(out, mapping, **resolve_dict(mapping_kwargs))
            wrap_kwargs = merge_dicts(dict(index=unstacked_index), wrap_kwargs)
            return self.wrapper.wrap(out, group_by=group_by, **wrap_kwargs)
        else:
            if has_conflicts:
                if reduce_func_nb is not None:
                    if reduce_args is None:
                        reduce_args = ()
                    self_ = self.reduce_segments(
                        "idx",
                        reduce_func_nb,
                        *reduce_args,
                        idx_arr=idx_arr,
                        group_by=group_by,
                        dtype=dtype,
                        jitted=jitted,
                        chunked=chunked,
                    )
                    idx_arr = self_.idx_arr
                else:
                    if not silence_warnings:
                        warnings.warn(
                            "Multiple values are pointing to the same position. Only the latest value is used.",
                            stacklevel=2,
                        )
                    self_ = self
            else:
                self_ = self
            col_arr = self_.col_mapper.get_col_arr(group_by=group_by)
            target_shape = self_.wrapper.get_shape_2d(group_by=group_by)
            func = jit_reg.resolve_option(nb.unstack_mapped_nb, jitted)
            out = func(self_.values, col_arr, idx_arr, target_shape, fill_value)
            mapping = self_.resolve_mapping(mapping)
            out = apply_mapping(out, mapping, **resolve_dict(mapping_kwargs))
            return self_.wrapper.wrap(out, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Masking ############# #

    def get_pd_mask(
        self,
        idx_arr: tp.Optional[tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get mask in form of a Series/DataFrame from row and column indices."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        out_arr = np.full(target_shape, False)
        out_arr[idx_arr, col_arr] = True
        return self.wrapper.wrap(out_arr, group_by=group_by, **resolve_dict(wrap_kwargs))

    @property
    def mask(self) -> tp.SeriesFrame:
        """`MappedArray.get_pd_mask` with default arguments."""
        return self.get_pd_mask()

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `MappedArray.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.mapped_array`."""
        from vectorbtpro._settings import settings

        mapped_array_stats_cfg = settings["mapped_array"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), mapped_array_stats_cfg)

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
            count=dict(title="Count", calc_func="count", tags="mapped_array"),
            mean=dict(title="Mean", calc_func="mean", inv_check_has_mapping=True, tags=["mapped_array", "describe"]),
            std=dict(title="Std", calc_func="std", inv_check_has_mapping=True, tags=["mapped_array", "describe"]),
            min=dict(title="Min", calc_func="min", inv_check_has_mapping=True, tags=["mapped_array", "describe"]),
            median=dict(
                title="Median",
                calc_func="median",
                inv_check_has_mapping=True,
                tags=["mapped_array", "describe"],
            ),
            max=dict(title="Max", calc_func="max", inv_check_has_mapping=True, tags=["mapped_array", "describe"]),
            idx_min=dict(
                title="Min Index",
                calc_func="idxmin",
                inv_check_has_mapping=True,
                agg_func=None,
                tags=["mapped_array", "index"],
            ),
            idx_max=dict(
                title="Max Index",
                calc_func="idxmax",
                inv_check_has_mapping=True,
                agg_func=None,
                tags=["mapped_array", "index"],
            ),
            value_counts=dict(
                title="Value Counts",
                calc_func=lambda value_counts: to_dict(value_counts, orient="index_series"),
                resolve_value_counts=True,
                check_has_mapping=True,
                tags=["mapped_array", "value_counts"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def histplot(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.BaseFigure:
        """Plot histogram by column/group."""
        return self.to_pd(group_by=group_by, ignore_index=True).vbt.histplot(**kwargs)

    def boxplot(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.BaseFigure:
        """Plot box plot by column/group."""
        return self.to_pd(group_by=group_by, ignore_index=True).vbt.boxplot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `MappedArray.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.mapped_array`."""
        from vectorbtpro._settings import settings

        mapped_array_plots_cfg = settings["mapped_array"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), mapped_array_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            to_pd_plot=dict(
                check_is_not_grouped=True,
                plot_func="to_pd.vbt.plot",
                pass_trace_names=False,
                tags="mapped_array",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


__pdoc__ = dict()
MappedArray.override_metrics_doc(__pdoc__)
MappedArray.override_subplots_doc(__pdoc__)
