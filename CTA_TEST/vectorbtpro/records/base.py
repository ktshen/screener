# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with records.

vectorbt works with two different representations of data: matrices and records.

A matrix, in this context, is just an array of one-dimensional arrays, each corresponding
to a separate feature. The matrix itself holds only one kind of information (one attribute).
For example, one can create a matrix for entry signals, with columns being different strategy
configurations. But what if the matrix is huge and sparse? What if there is more
information we would like to represent by each element? Creating multiple matrices would be
a waste of memory.

Records make possible representing complex, sparse information in a dense format. They are just
an array of one-dimensional arrays of a fixed schema, where each element holds a different
kind of information. You can imagine records being a DataFrame, where each row represents a record
and each column represents a specific attribute. Read more on structured arrays
[here](https://numpy.org/doc/stable/user/basics.rec.html).

For example, let's represent two DataFrames as a single record array:

```plaintext
               a     b
         0   1.0   5.0
attr1 =  1   2.0   NaN
         2   NaN   7.0
         3   4.0   8.0
               a     b
         0   9.0  13.0
attr2 =  1  10.0   NaN
         2   NaN  15.0
         3  12.0  16.0
            |
            v
      id  col  idx  attr1  attr2
0      0    0    0      1      9
1      1    0    1      2     10
2      2    0    3      4     12
3      0    1    0      5     13
4      1    1    2      7     15
5      2    1    3      8     16
```

Another advantage of records is that they are not constrained by size. Multiple records can map
to a single element in a matrix. For example, one can define multiple orders at the same timestamp,
which is impossible to represent in a matrix form without duplicating index entries or using complex data types.

Consider the following example:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from collections import namedtuple
>>> import vectorbtpro as vbt

>>> example_dt = np.dtype([
...     ('id', np.int_),
...     ('col', np.int_),
...     ('idx', np.int_),
...     ('some_field', np.float_)
... ])
>>> records_arr = np.array([
...     (0, 0, 0, 10.),
...     (1, 0, 1, 11.),
...     (2, 0, 2, 12.),
...     (0, 1, 0, 13.),
...     (1, 1, 1, 14.),
...     (2, 1, 2, 15.),
...     (0, 2, 0, 16.),
...     (1, 2, 1, 17.),
...     (2, 2, 2, 18.)
... ], dtype=example_dt)
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> records = vbt.Records(wrapper, records_arr)
```

## Printing

There are two ways to print records:

* Raw dataframe that preserves field names and data types:

```pycon
>>> records.records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0
3   0    1    0        13.0
4   1    1    1        14.0
5   2    1    2        15.0
6   0    2    0        16.0
7   1    2    1        17.0
8   2    2    2        18.0
```

* Readable dataframe that takes into consideration `Records.field_config`:

```pycon
>>> records.records_readable
   Id Column Timestamp  some_field
0   0      a         x        10.0
1   1      a         y        11.0
2   2      a         z        12.0
3   0      b         x        13.0
4   1      b         y        14.0
5   2      b         z        15.0
6   0      c         x        16.0
7   1      c         y        17.0
8   2      c         z        18.0
```

## Mapping

`Records` are just [structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) with a bunch
of methods and properties for processing them. Their main feature is to map the records array and
to reduce it by column (similar to the MapReduce paradigm). The main advantage is that it all happens
without conversion to the matrix form and wasting memory resources.

`Records` can be mapped to `vectorbtpro.records.mapped_array.MappedArray` in several ways:

* Use `Records.map_field` to map a record field:

```pycon
>>> records.map_field('some_field')
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49bd31a58>

>>> records.map_field('some_field').values
array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
```

* Use `Records.map` to map records using a custom function.

```pycon
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> records.map(power_map_nb, 2)
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49c990cf8>

>>> records.map(power_map_nb, 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])

>>> # Map using a meta function

>>> @njit
... def power_map_meta_nb(ridx, records, pow):
...     return records[ridx].some_field ** pow

>>> vbt.Records.map(power_map_meta_nb, records.values, 2, col_mapper=records.col_mapper).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

* Use `Records.map_array` to convert an array to `vectorbtpro.records.mapped_array.MappedArray`.

```pycon
>>> records.map_array(records_arr['some_field'] ** 2)
<vectorbtpro.records.mapped_array.MappedArray object at 0x7fe9bccf2978>

>>> records.map_array(records_arr['some_field'] ** 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

* Use `Records.apply` to apply a function on each column/group:

```pycon
>>> @njit
... def cumsum_apply_nb(records):
...     return np.cumsum(records.some_field)

>>> records.apply(cumsum_apply_nb)
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49c990cf8>

>>> records.apply(cumsum_apply_nb).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])

>>> group_by = np.array(['first', 'first', 'second'])
>>> records.apply(cumsum_apply_nb, group_by=group_by, apply_per_group=True).values
array([10., 21., 33., 46., 60., 75., 16., 33., 51.])

>>> # Apply using a meta function

>>> @njit
... def cumsum_apply_meta_nb(idxs, col, records):
...     return np.cumsum(records[idxs].some_field)

>>> vbt.Records.apply(cumsum_apply_meta_nb, records.values, col_mapper=records.col_mapper).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])
```

Notice how cumsum resets at each column in the first example and at each group in the second example.

## Filtering

Use `Records.apply_mask` to filter elements per column/group:

```pycon
>>> mask = [True, False, True, False, True, False, True, False, True]
>>> filtered_records = records.apply_mask(mask)
>>> filtered_records.records
   id  col  idx  some_field
0   0    0    0        10.0
1   2    0    2        12.0
2   1    1    1        14.0
3   0    2    0        16.0
4   2    2    2        18.0
```

## Grouping

One of the key features of `Records` is that you can perform reducing operations on a group
of columns as if they were a single column. Groups can be specified by `group_by`, which
can be anything from positions or names of column levels, to a NumPy array with actual groups.

There are multiple ways of define grouping:

* When creating `Records`, pass `group_by` to `vectorbtpro.base.wrapping.ArrayWrapper`:

```pycon
>>> group_by = np.array(['first', 'first', 'second'])
>>> grouped_wrapper = wrapper.replace(group_by=group_by)
>>> grouped_records = vbt.Records(grouped_wrapper, records_arr)

>>> grouped_records.map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

* Regroup an existing `Records`:

```pycon
>>> records.regroup(group_by).map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the mapping method:

```pycon
>>> records.map_field('some_field', group_by=group_by).mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the reducing method:

```pycon
>>> records.map_field('some_field').mean(group_by=group_by)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations, there is no change to the arrays.

## Indexing

Like any other class subclassing `vectorbtpro.base.wrapping.Wrapping`, we can do pandas indexing
on a `Records` instance, which forwards indexing operation to each object with columns:

```pycon
>>> records['a'].records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0

>>> grouped_records['first'].records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0
3   0    1    0        13.0
4   1    1    1        14.0
5   2    1    2        15.0
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`
    to get the first column.

    Indexing behavior depends solely upon `vectorbtpro.base.wrapping.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups when grouped,
    otherwise on single columns.

## Caching

`Records` supports caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbtpro.utils.decorators.cached_method` and `vectorbtpro.utils.decorators.cached_property`
respectively. Caching can be disabled globally via `vectorbtpro._settings.caching`.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `Records.replace` method and pass changes as keyword arguments.

## Saving and loading

Like any other class subclassing `vectorbtpro.utils.pickling.Pickleable`, we can save a `Records`
instance to the disk with `Records.save` and load it with `Records.load`.

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Records.metrics`.

```pycon
>>> records.stats(column='a')
Start                          x
End                            z
Period           3 days 00:00:00
Total Records                  3
Name: a, dtype: object
```

`Records.stats` also supports (re-)grouping:

```pycon
>>> grouped_records.stats(column='first')
Start                          x
End                            z
Period           3 days 00:00:00
Total Records                  6
Name: first, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Records.subplots`.

This class is too generic to have any subplots, but feel free to add custom subplots to your subclass.

## Extending

`Records` class can be extended by subclassing.

In case some of our fields have the same meaning but different naming (such as the base field `idx`)
or other properties, we can override `field_config` using `vectorbtpro.records.decorators.override_field_config`.
It will look for configs of all base classes and merge our config on top of them. This preserves
any base class property that is not explicitly listed in our config.

```pycon
>>> from vectorbtpro.records.decorators import override_field_config

>>> my_dt = np.dtype([
...     ('my_id', np.int_),
...     ('my_col', np.int_),
...     ('my_idx', np.int_)
... ])

>>> my_fields_config = dict(
...     dtype=my_dt,
...     settings=dict(
...         id=dict(name='my_id'),
...         col=dict(name='my_col'),
...         idx=dict(name='my_idx')
...     )
... )
>>> @override_field_config(my_fields_config)
... class MyRecords(vbt.Records):
...     pass

>>> records_arr = np.array([
...     (0, 0, 0),
...     (1, 0, 1),
...     (0, 1, 0),
...     (1, 1, 1)
... ], dtype=my_dt)
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y'],
...     columns=['a', 'b'], ndim=2, freq='1 day')
>>> my_records = MyRecords(wrapper, records_arr)

>>> my_records.id_arr
array([0, 1, 0, 1])

>>> my_records.col_arr
array([0, 0, 1, 1])

>>> my_records.idx_arr
array([0, 1, 0, 1])
```

Alternatively, we can override the `_field_config` class attribute.

```pycon
>>> @override_field_config
... class MyRecords(vbt.Records):
...     _field_config = dict(
...         dtype=my_dt,
...         settings=dict(
...             id=dict(name='my_id'),
...             idx=dict(name='my_idx'),
...             col=dict(name='my_col')
...         )
...     )
```

!!! note
    Don't forget to decorate the class with `@override_field_config` to inherit configs from base classes.

    You can stop inheritance by not decorating or passing `merge_configs=False` to the decorator.
"""

import inspect
import string
from collections import defaultdict

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.records import nb
from vectorbtpro.records.col_mapper import ColumnMapper
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.decorators import cached_method, class_or_instancemethod
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.template import Sub

__all__ = [
    "Records",
]

__pdoc__ = {}

RecordsT = tp.TypeVar("RecordsT", bound="Records")


class MetaFields(type):
    """Meta class that exposes a read-only class property `MetaFields.field_config`."""

    @property
    def field_config(cls) -> Config:
        """Field config."""
        return cls._field_config


class RecordsWithFields(metaclass=MetaFields):
    """Class exposes a read-only class property `RecordsWithFields.field_config`."""

    @property
    def field_config(self) -> Config:
        """Field config of `${cls_name}`.

        ```python
        ${field_config}
        ```
        """
        return self._field_config


class MetaRecords(type(Analyzable), type(RecordsWithFields)):
    pass


class Records(Analyzable, RecordsWithFields, metaclass=MetaRecords):
    """Wraps the actual records array (such as trades) and exposes methods for mapping
    it to some array of values (such as PnL of each trade).

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        records_arr (array_like): A structured NumPy array of records.

            Must have the fields `id` (record index) and `col` (column index).
        col_mapper (ColumnMapper): Column mapper if already known.

            !!! note
                It depends on `records_arr`, so make sure to invalidate `col_mapper` upon creating
                a `Records` instance with a modified `records_arr`.

                `Records.replace` does it automatically.
        **kwargs: Custom keyword arguments passed to the config.

            Useful if any subclass wants to extend the config.
    """

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_field_config"}

    _field_config: tp.ClassVar[Config] = HybridConfig(
        dict(
            dtype=None,
            settings=dict(
                id=dict(name="id", title="Id", mapping="ids"),
                col=dict(name="col", title="Column", mapping="columns", as_customdata=False),
                idx=dict(name="idx", title="Index", mapping="index"),
            ),
        )
    )

    @property
    def field_config(self) -> Config:
        """Field config of `${cls_name}`.

        ```python
        ${field_config}
        ```

        Returns `${cls_name}._field_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change fields, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._field_config`.
        """
        return self._field_config

    @classmethod
    def row_stack_records_arrs(cls, *objs: tp.MaybeTuple[tp.RecordArray], **kwargs):
        """Stack multiple record arrays along rows."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        records_arrs = []
        for col in range(kwargs["wrapper"].shape_2d[1]):
            n_rows_sum = 0
            from_id = defaultdict(int)
            for i, obj in enumerate(objs):
                col_idxs, col_lens = obj.col_mapper.col_map
                if len(col_idxs) > 0:
                    col_records = None
                    set_columns = False
                    if col > 0 and obj.wrapper.shape_2d[1] == 1:
                        col_records = obj.records_arr[col_idxs]
                        set_columns = True
                    elif col_lens[col] > 0:
                        col_end_idxs = np.cumsum(col_lens)
                        col_start_idxs = col_end_idxs - col_lens
                        col_records = obj.records_arr[col_idxs[col_start_idxs[col] : col_end_idxs[col]]]
                    if col_records is not None:
                        col_records = col_records.copy()
                        for field in obj.values.dtype.names:
                            field_mapping = cls.field_config.get("settings", {}).get(field, {}).get("mapping", None)
                            if isinstance(field_mapping, str) and field_mapping == "columns" and set_columns:
                                col_records[field][:] = col
                            elif isinstance(field_mapping, str) and field_mapping == "index":
                                col_records[field][:] += n_rows_sum
                            elif isinstance(field_mapping, str) and field_mapping == "ids":
                                col_records[field][:] += from_id[field]
                                from_id[field] = col_records[field].max() + 1
                        records_arrs.append(col_records)
                n_rows_sum += obj.wrapper.shape_2d[0]
        if len(records_arrs) == 0:
            return np.array([], dtype=objs[0].values.dtype)
        return np.concatenate(records_arrs)

    @classmethod
    def row_stack(
        cls: tp.Type[RecordsT],
        *objs: tp.MaybeTuple[RecordsT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RecordsT:
        """Stack multiple `Records` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers
        and `Records.row_stack_records_arrs` to stack the record arrays.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Records):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        if "col_mapper" not in kwargs:
            kwargs["col_mapper"] = ColumnMapper.row_stack(
                *[obj.col_mapper for obj in objs],
                wrapper=kwargs["wrapper"],
            )
        if "records_arr" not in kwargs:
            kwargs["records_arr"] = cls.row_stack_records_arrs(*objs, **kwargs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack_records_arrs(
        cls,
        *objs: tp.MaybeTuple[tp.RecordArray],
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ):
        """Stack multiple record arrays along columns."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        if get_indexer_kwargs is None:
            get_indexer_kwargs = {}
        records_arrs = []
        col_sum = 0
        for i, obj in enumerate(objs):
            col_idxs, col_lens = obj.col_mapper.col_map
            if len(col_idxs) > 0:
                col_end_idxs = np.cumsum(col_lens)
                col_start_idxs = col_end_idxs - col_lens
                for obj_col in range(len(col_lens)):
                    if col_lens[obj_col] > 0:
                        col_records = obj.records_arr[col_idxs[col_start_idxs[obj_col] : col_end_idxs[obj_col]]]
                        col_records = col_records.copy()
                        for field in obj.values.dtype.names:
                            field_mapping = cls.field_config.get("settings", {}).get(field, {}).get("mapping", None)
                            if isinstance(field_mapping, str) and field_mapping == "columns":
                                col_records[field][:] += col_sum
                            elif isinstance(field_mapping, str) and field_mapping == "index":
                                old_idxs = col_records[field]
                                if not obj.wrapper.index.equals(kwargs["wrapper"].index):
                                    new_idxs = kwargs["wrapper"].index.get_indexer(
                                        obj.wrapper.index[old_idxs],
                                        **get_indexer_kwargs,
                                    )
                                else:
                                    new_idxs = old_idxs
                                col_records[field][:] = new_idxs
                        records_arrs.append(col_records)
            col_sum += obj.wrapper.shape_2d[1]
        if len(records_arrs) == 0:
            return np.array([], dtype=objs[0].values.dtype)
        return np.concatenate(records_arrs)

    @classmethod
    def column_stack(
        cls: tp.Type[RecordsT],
        *objs: tp.MaybeTuple[RecordsT],
        wrapper_kwargs: tp.KwargsLike = None,
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RecordsT:
        """Stack multiple `Records` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers
        and `Records.column_stack_records_arrs` to stack the record arrays.

        `get_indexer_kwargs` are passed to
        [pandas.Index.get_indexer](https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html)
        to translate old indices to new ones after the reindexing operation.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Records):
                raise TypeError("Each object to be merged must be an instance of Records")
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
        if "records_arr" not in kwargs:
            kwargs["records_arr"] = cls.column_stack_records_arrs(
                *objs,
                get_indexer_kwargs=get_indexer_kwargs,
                **kwargs,
            )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "records_arr",
        "col_mapper",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> None:

        # Check fields
        records_arr = np.asarray(records_arr)
        checks.assert_not_none(records_arr.dtype.fields)
        field_names = {dct.get("name", field_name) for field_name, dct in self.field_config.get("settings", {}).items()}
        dtype = self.field_config.get("dtype", None)
        if dtype is not None:
            for field in dtype.names:
                if field not in records_arr.dtype.names:
                    if field not in field_names:
                        raise TypeError(f"Field '{field}' from {dtype} cannot be found in records or config")
        if col_mapper is None:
            col_mapper = ColumnMapper(wrapper, records_arr[self.get_field_name("col")])

        Analyzable.__init__(self, wrapper, records_arr=records_arr, col_mapper=col_mapper, **kwargs)

        self._records_arr = records_arr
        self._col_mapper = col_mapper

        # Only slices of rows can be selected
        self._range_only_select = True

        # Copy writeable attrs
        self._field_config = type(self)._field_config.copy()

    def replace(self: RecordsT, **kwargs) -> RecordsT:
        """See `vectorbtpro.utils.config.Configured.replace`.

        Also, makes sure that `Records.col_mapper` is not passed to the new instance."""
        if self.config.get("col_mapper", None) is not None:
            if "wrapper" in kwargs:
                if self.wrapper is not kwargs.get("wrapper"):
                    kwargs["col_mapper"] = None
            if "records_arr" in kwargs:
                if self.records_arr is not kwargs.get("records_arr"):
                    kwargs["col_mapper"] = None
        return Analyzable.replace(self, **kwargs)

    def select_cols(
        self,
        col_idxs: tp.MaybeIndexArray,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
        """Select columns.

        Returns indices and new record array. Automatically decides whether to use column lengths or column map."""
        if len(self.values) == 0:
            return np.arange(len(self.values)), self.values
        if isinstance(col_idxs, slice):
            if col_idxs.start is None and col_idxs.stop is None:
                return np.arange(len(self.values)), self.values
            col_idxs = np.arange(col_idxs.start, col_idxs.stop)
        if self.col_mapper.is_sorted():
            func = jit_reg.resolve_option(nb.record_col_lens_select_nb, jitted)
            new_indices, new_records_arr = func(self.values, self.col_mapper.col_lens, to_1d_array(col_idxs))  # faster
        else:
            func = jit_reg.resolve_option(nb.record_col_map_select_nb, jitted)
            new_indices, new_records_arr = func(
                self.values, self.col_mapper.col_map, to_1d_array(col_idxs)
            )  # more flexible
        return new_indices, new_records_arr

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `Records` and return metadata.

        By default, all fields that are mapped to index are indexed.
        To avoid indexing on some fields, set their setting `noindex` to True."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                range_only_select=self.range_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        if self.get_field_setting("col", "group_indexing", False):
            new_indices, new_records_arr = self.select_cols(wrapper_meta["group_idxs"])
        else:
            new_indices, new_records_arr = self.select_cols(wrapper_meta["col_idxs"])
        if wrapper_meta["rows_changed"]:
            row_idxs = wrapper_meta["row_idxs"]
            index_fields = []
            all_index_fields = []
            for field in new_records_arr.dtype.names:
                field_mapping = self.get_field_mapping(field)
                noindex = self.get_field_setting(field, "noindex", False)
                if isinstance(field_mapping, str) and field_mapping == "index":
                    all_index_fields.append(field)
                    if not noindex:
                        index_fields.append(field)
            if len(index_fields) > 0:
                masks = []
                for field in index_fields:
                    field_arr = new_records_arr[field]
                    masks.append((field_arr >= row_idxs.start) & (field_arr < row_idxs.stop))
                mask = np.array(masks).all(axis=0)
                new_indices = new_indices[mask]
                new_records_arr = new_records_arr[mask]
                for field in all_index_fields:
                    new_records_arr[field] = new_records_arr[field] - row_idxs.start
        return dict(
            wrapper_meta=wrapper_meta,
            new_indices=new_indices,
            new_records_arr=new_records_arr,
        )

    def indexing_func(self: RecordsT, *args, records_meta: tp.DictLike = None, **kwargs) -> RecordsT:
        """Perform indexing on `Records`."""
        if records_meta is None:
            records_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
        )

    def resample_records_arr(self, resampler: tp.Union[Resampler, tp.PandasResampler]) -> tp.RecordArray:
        """Perform resampling on the record array."""
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        new_records_arr = self.records_arr.copy()
        for field_name in self.values.dtype.names:
            field_mapping = self.get_field_mapping(field_name)
            if isinstance(field_mapping, str) and field_mapping == "index":
                index_map = _resampler.map_to_target_index(return_index=False)
                new_records_arr[field_name] = index_map[new_records_arr[field_name]]
        return new_records_arr

    def resample_meta(self: RecordsT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform resampling on `Records` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        new_records_arr = self.resample_records_arr(wrapper_meta["resampler"])
        return dict(wrapper_meta=wrapper_meta, new_records_arr=new_records_arr)

    def resample(self: RecordsT, *args, records_meta: tp.DictLike = None, **kwargs) -> RecordsT:
        """Perform resampling on `Records`."""
        if records_meta is None:
            records_meta = self.resample_meta(*args, **kwargs)
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
        )

    @property
    def records_arr(self) -> tp.RecordArray:
        """Records array."""
        return self._records_arr

    @property
    def values(self) -> tp.RecordArray:
        """Records array."""
        return self.records_arr

    def __len__(self) -> int:
        return len(self.values)

    @property
    def records(self) -> tp.Frame:
        """Records."""
        return pd.DataFrame.from_records(self.values)

    @property
    def recarray(self) -> tp.RecArray:
        return self.values.view(np.recarray)

    @property
    def col_mapper(self) -> ColumnMapper:
        """Column mapper.

        See `vectorbtpro.records.col_mapper.ColumnMapper`."""
        return self._col_mapper

    @property
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        df = self.records.copy()
        field_settings = self.field_config.get("settings", {})
        for col_name in df.columns:
            if col_name in field_settings:
                dct = field_settings[col_name]
                if dct.get("ignore", False):
                    df = df.drop(columns=col_name)
                    continue
                field_name = dct.get("name", col_name)
                if "title" in dct:
                    title = dct["title"]
                    new_columns = dict()
                    new_columns[field_name] = title
                    df.rename(columns=new_columns, inplace=True)
                else:
                    title = field_name
                if "mapping" in dct:
                    if isinstance(dct["mapping"], str) and dct["mapping"] == "index":
                        df[title] = self.get_map_field_to_index(col_name)
                    else:
                        df[title] = self.get_apply_mapping_arr(col_name)
        if all([isinstance(col, tuple) for col in df.columns]):
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def get_field_setting(self, field: str, setting: str, default: tp.Any = None) -> tp.Any:
        """Get any setting of the field. Uses `Records.field_config`."""
        return self.field_config.get("settings", {}).get(field, {}).get(setting, default)

    def get_field_name(self, field: str) -> str:
        """Get the name of the field. Uses `Records.field_config`.."""
        return self.get_field_setting(field, "name", field)

    def get_field_title(self, field: str) -> str:
        """Get the title of the field. Uses `Records.field_config`."""
        return self.get_field_setting(field, "title", field)

    def get_field_mapping(self, field: str) -> tp.Optional[tp.MappingLike]:
        """Get the mapping of the field. Uses `Records.field_config`."""
        return self.get_field_setting(field, "mapping", None)

    def get_field_arr(self, field: str, copy: bool = False) -> tp.Array1d:
        """Get the array of the field. Uses `Records.field_config`."""
        out = self.values[self.get_field_name(field)]
        if copy:
            out = out.copy()
        return out

    def get_map_field(self, field: str, **kwargs) -> MappedArray:
        """Get the mapped array of the field. Uses `Records.field_config`."""
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "ids":
            mapping = None
        return self.map_field(self.get_field_name(field), mapping=mapping, **kwargs)

    def get_map_field_to_index(self, field: str, minus_one_to_zero: bool = False, **kwargs) -> tp.Index:
        """Get the mapped array on the field, with index applied. Uses `Records.field_config`."""
        return self.get_map_field(field, **kwargs).to_index(minus_one_to_zero=minus_one_to_zero)

    def get_map_field_to_columns(self, field: str, **kwargs) -> tp.Index:
        """Get the mapped array on the field, with columns applied. Uses `Records.field_config`."""
        return self.get_map_field(field, **kwargs).to_columns()

    def get_apply_mapping_arr(self, field: str, mapping_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Array1d:
        """Get the mapped array on the field, with mapping applied. Uses `Records.field_config`."""
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "index":
            return self.get_map_field_to_index(field, **kwargs).values
        if isinstance(mapping, str) and mapping == "columns":
            return self.get_map_field_to_columns(field, **kwargs).values
        return self.get_map_field(field, **kwargs).apply_mapping(mapping_kwargs=mapping_kwargs).values

    def get_apply_mapping_str_arr(self, field: str, mapping_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Array1d:
        """Get the mapped array on the field, with mapping applied and stringified. Uses `Records.field_config`."""
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "index":
            return self.get_map_field_to_index(field, **kwargs).astype(str).values
        if isinstance(mapping, str) and mapping == "columns":
            return self.get_map_field_to_columns(field, **kwargs).astype(str).values
        return self.get_map_field(field, **kwargs).apply_mapping(mapping_kwargs=mapping_kwargs).values.astype(str)

    @property
    def id_arr(self) -> tp.Array1d:
        """Get id array."""
        return self.values[self.get_field_name("id")]

    @property
    def col_arr(self) -> tp.Array1d:
        """Get column array."""
        return self.values[self.get_field_name("col")]

    @property
    def idx_arr(self) -> tp.Optional[tp.Array1d]:
        """Get index array."""
        idx_field_name = self.get_field_name("idx")
        if idx_field_name is None:
            return None
        return self.values[idx_field_name]

    # ############# Sorting ############# #

    @cached_method
    def is_sorted(self, incl_id: bool = False, jitted: tp.JittedOption = None) -> bool:
        """Check whether records are sorted."""
        if incl_id:
            func = jit_reg.resolve_option(nb.is_col_id_sorted_nb, jitted)
            return func(self.col_arr, self.id_arr)
        func = jit_reg.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)

    def sort(self: RecordsT, incl_id: bool = False, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """Sort records by columns (primary) and ids (secondary, optional).

        !!! note
            Sorting is expensive. A better approach is to append records already in the correct order."""
        if self.is_sorted(incl_id=incl_id):
            return self.replace(**kwargs).regroup(group_by)
        if incl_id:
            ind = np.lexsort((self.id_arr, self.col_arr))  # expensive!
        else:
            ind = np.argsort(self.col_arr)
        return self.replace(records_arr=self.values[ind], **kwargs).regroup(group_by)

    # ############# Filtering ############# #

    def apply_mask(self: RecordsT, mask: tp.Array1d, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """Return a new class instance, filtered by mask."""
        mask_indices = np.flatnonzero(mask)
        return self.replace(records_arr=np.take(self.values, mask_indices), **kwargs).regroup(group_by)

    def first_n(
        self: RecordsT,
        n: int,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return the first N records in each column."""
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.first_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    def last_n(
        self: RecordsT,
        n: int,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return the last N records in each column."""
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.last_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    def random_n(
        self: RecordsT,
        n: int,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return random N records in each column."""
        if seed is not None:
            set_seed_nb(seed)
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.random_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    # ############# Mapping ############# #

    def map_array(
        self,
        a: tp.ArrayLike,
        idx_arr: tp.Optional[tp.ArrayLike] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArray:
        """Convert array to mapped array.

        The length of the array must match that of the records."""
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        checks.assert_shape_equal(a, self.values)
        if idx_arr is None:
            idx_arr = self.idx_arr
        return MappedArray(
            self.wrapper,
            a,
            self.col_arr,
            id_arr=self.id_arr,
            idx_arr=idx_arr,
            mapping=mapping,
            col_mapper=self.col_mapper,
            **kwargs,
        ).regroup(group_by)

    def map_field(self, field: str, **kwargs) -> MappedArray:
        """Convert field to mapped array.

        `**kwargs` are passed to `Records.map_array`."""
        mapped_arr = self.values[field]
        return self.map_array(mapped_arr, **kwargs)

    @class_or_instancemethod
    def map(
        cls_or_self,
        map_func_nb: tp.Union[tp.RecordsMapFunc, tp.RecordsMapMetaFunc],
        *args,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> MappedArray:
        """Map each record to a scalar value. Returns mapped array.

        See `vectorbtpro.records.nb.map_records_nb`.

        For details on the meta version, see `vectorbtpro.records.nb.map_records_meta_nb`.

        `**kwargs` are passed to `Records.map_array`."""
        if isinstance(cls_or_self, type):
            checks.assert_not_none(col_mapper)
            func = jit_reg.resolve_option(nb.map_records_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(len(col_mapper.col_arr), map_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return MappedArray(col_mapper.wrapper, mapped_arr, col_mapper.col_arr, col_mapper=col_mapper, **kwargs)
        else:
            func = jit_reg.resolve_option(nb.map_records_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(cls_or_self.values, map_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return cls_or_self.map_array(mapped_arr, **kwargs)

    @class_or_instancemethod
    def apply(
        cls_or_self,
        apply_func_nb: tp.Union[tp.ApplyFunc, tp.ApplyMetaFunc],
        *args,
        group_by: tp.GroupByLike = None,
        apply_per_group: bool = False,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> MappedArray:
        """Apply function on records per column/group. Returns mapped array.

        Applies per group if `apply_per_group` is True.

        See `vectorbtpro.records.nb.apply_nb`.

        For details on the meta version, see `vectorbtpro.records.nb.apply_meta_nb`.

        `**kwargs` are passed to `Records.map_array`."""
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
            return cls_or_self.map_array(mapped_arr, group_by=group_by, **kwargs)

    # ############# Masking ############# #

    def get_pd_mask(
        self,
        idx_arr: tp.Union[None, str, tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get mask in form of a Series/DataFrame from row and column indices."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        elif isinstance(idx_arr, str):
            idx_arr = self.get_field_arr(idx_arr)
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        out_arr = np.full(target_shape, False)
        out_arr[idx_arr, col_arr] = True
        return self.wrapper.wrap(out_arr, group_by=group_by, **resolve_dict(wrap_kwargs))

    @property
    def mask(self) -> tp.SeriesFrame:
        """`MappedArray.get_pd_mask` with default arguments."""
        return self.get_pd_mask()

    # ############# Reducing ############# #

    @cached_method
    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Get count by column."""
        wrap_kwargs = merge_dicts(dict(name_or_index="count"), wrap_kwargs)
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by,
            **wrap_kwargs,
        )

    # ############# Conflicts ############# #

    @cached_method
    def has_conflicts(self, **kwargs) -> bool:
        """See `vectorbtpro.records.mapped_array.MappedArray.has_conflicts`."""
        return self.get_map_field("col").has_conflicts(**kwargs)

    def coverage_map(self, **kwargs) -> tp.SeriesFrame:
        """See `vectorbtpro.records.mapped_array.MappedArray.coverage_map`."""
        return self.get_map_field("col").coverage_map(**kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Records.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.records`."""
        from vectorbtpro._settings import settings

        records_stats_cfg = settings["records"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), records_stats_cfg)

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
            count=dict(title="Count", calc_func="count", tags="records"),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def prepare_customdata(
        self,
        incl_fields: tp.Optional[tp.Sequence[str]] = None,
        excl_fields: tp.Optional[tp.Sequence[str]] = None,
        append_info: tp.Optional[tp.Sequence[tp.Tuple]] = None,
        mask: tp.Optional[tp.Array1d] = None,
    ) -> tp.Tuple[tp.Array2d, str]:
        """Prepare customdata and hoverinfo for Plotly.

        Will display all fields in the data type or only those in `incl_fields`, unless any of them has
        the field config setting `as_customdata` disabled, or it's listed in `excl_fields`.
        Additionally, you can define `hovertemplate` in the field config such as by using
        `vectorbtpro.utils.template.Sub` where `title` is substituted by the title and `index` is
        substituted by (final) index in the customdata. If provided as a string, will be wrapped with
        `vectorbtpro.utils.template.Sub`. Defaults to "$title: %{{customdata[$index]}}". Mapped fields
        will be stringified automatically.

        To append one or more custom arrays, provide `append_info` as a list of tuples, each consisting
        of a 1-dim NumPy array, title, and optionally hoverinfo. If the array's data type is `object`,
        will treat it as strings, otherwise as numbers."""
        customdata_info = []
        if incl_fields is not None:
            iterate_over_names = incl_fields
        else:
            iterate_over_names = self.field_config.get("dtype").names
        for field in iterate_over_names:
            if excl_fields is not None and field in excl_fields:
                continue
            field_as_customdata = self.get_field_setting(field, "as_customdata", True)
            if field_as_customdata:
                numeric_customdata = self.get_field_setting(field, "mapping", None)
                if numeric_customdata is not None:
                    field_arr = self.get_apply_mapping_str_arr(field)
                    field_hovertemplate = self.get_field_setting(
                        field,
                        "hovertemplate",
                        "$title: %{customdata[$index]}",
                    )
                else:
                    field_arr = self.get_apply_mapping_arr(field)
                    field_hovertemplate = self.get_field_setting(
                        field,
                        "hovertemplate",
                        "$title: %{customdata[$index]:,}",
                    )
                if isinstance(field_hovertemplate, str):
                    field_hovertemplate = Sub(field_hovertemplate)
                field_title = self.get_field_title(field)
                customdata_info.append((field_arr, field_title, field_hovertemplate))
        if append_info is not None:
            for info in append_info:
                checks.assert_instance_of(info, tuple)
                if len(info) == 2:
                    if info[0].dtype == object:
                        info += ("$title: %{customdata[$index]}",)
                    else:
                        info += ("$title: %{customdata[$index]:,}",)
                if isinstance(info[2], str):
                    info = (info[0], info[1], Sub(info[2]))
                customdata_info.append(info)
        customdata = []
        hovertemplate = []
        for i in range(len(customdata_info)):
            if mask is not None:
                customdata.append(customdata_info[i][0][mask])
            else:
                customdata.append(customdata_info[i][0])
            _hovertemplate = customdata_info[i][2].substitute(dict(title=customdata_info[i][1], index=i))
            if not _hovertemplate.startswith("<br>"):
                _hovertemplate = "<br>" + _hovertemplate
            hovertemplate.append(_hovertemplate)
        return np.stack(customdata, axis=1), "\n".join(hovertemplate)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Records.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.records`."""
        from vectorbtpro._settings import settings

        records_plots_cfg = settings["records"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), records_plots_cfg)

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_field_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build field config documentation."""
        if source_cls is None:
            source_cls = Records
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "field_config").__doc__)).substitute(
            {"field_config": cls.field_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_field_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Records.field_config`."""
        __pdoc__[cls.__name__ + ".field_config"] = cls.build_field_config_doc(source_cls=source_cls)


Records.override_field_config_doc(__pdoc__)
Records.override_metrics_doc(__pdoc__)
Records.override_subplots_doc(__pdoc__)
