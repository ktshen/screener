# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for wrapping NumPy arrays into Series/DataFrames."""

import warnings
from functools import partial

import numpy as np
import pandas as pd
from pandas.core.groupby import GroupBy as PandasGroupBy

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.indexing import IndexingError, PandasIndexer, index_dict, IdxSetter, IdxSetterFactory, IdxDict
from vectorbtpro.base.indexes import stack_indexes, concat_indexes
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import AttrResolverMixin, AttrResolverMixinT
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.datetime_ import infer_index_freq, try_to_datetime_index
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.decorators import class_or_instancemethod, cached_method, cached_property
from vectorbtpro.utils.array_ import is_range, cast_to_min_precision, cast_to_max_precision
from vectorbtpro.utils.template import CustomTemplate

if tp.TYPE_CHECKING:
    from vectorbtpro.base.accessors import BaseIDXAccessor as BaseIDXAccessorT
    from vectorbtpro.generic.splitting.base import Splitter as SplitterT
else:
    BaseIDXAccessorT = tp.Any
    SplitterT = tp.Any

__all__ = [
    "ArrayWrapper",
    "Wrapping",
]

ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")


class ArrayWrapper(Configured, PandasIndexer):
    """Class that stores index, columns, and shape metadata for wrapping NumPy arrays.
    Tightly integrated with `vectorbtpro.base.grouping.base.Grouper` for grouping columns.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    `**kwargs` are passed to `vectorbtpro.base.grouping.base.Grouper`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `ArrayWrapper.replace`.

        Use methods that begin with `get_` to get group-aware results."""

    @classmethod
    def from_obj(cls: tp.Type[ArrayWrapperT], obj: tp.ArrayLike, *args, **kwargs) -> ArrayWrapperT:
        """Derive metadata from an object."""
        from vectorbtpro.base.reshaping import to_pd_array

        pd_obj = to_pd_array(obj)
        index = indexes.get_index(pd_obj, 0)
        columns = indexes.get_index(pd_obj, 1)
        ndim = pd_obj.ndim
        kwargs.pop("index", None)
        kwargs.pop("columns", None)
        kwargs.pop("ndim", None)
        return cls(index, columns, ndim, *args, **kwargs)

    @classmethod
    def from_shape(
        cls: tp.Type[ArrayWrapperT],
        shape: tp.ShapeLike,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        ndim: tp.Optional[int] = None,
        *args,
        **kwargs,
    ) -> ArrayWrapperT:
        """Derive metadata from shape."""
        shape = reshaping.to_tuple_shape(shape)
        if index is None:
            index = pd.RangeIndex(stop=shape[0])
        if columns is None:
            columns = pd.RangeIndex(stop=shape[1] if len(shape) > 1 else 1)
        if ndim is None:
            ndim = len(shape)
        return cls(index, columns, ndim, *args, **kwargs)

    @staticmethod
    def extract_init_kwargs(**kwargs) -> tp.Tuple[tp.Kwargs, tp.Kwargs]:
        """Extract keyword arguments that can be passed to `ArrayWrapper` or `Grouper`."""
        wrapper_arg_names = get_func_arg_names(ArrayWrapper.__init__)
        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        init_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in wrapper_arg_names or k in grouper_arg_names:
                init_kwargs[k] = kwargs.pop(k)
        return init_kwargs, kwargs

    @classmethod
    def resolve_stack_kwargs(cls, *wrappers: tp.MaybeTuple[ArrayWrapperT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `ArrayWrapper` after stacking."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)

        common_keys = set()
        for wrapper in wrappers:
            common_keys = common_keys.union(set(wrapper.config.keys()))
            if "grouper" not in kwargs:
                common_keys = common_keys.union(set(wrapper.grouper.config.keys()))
        common_keys.remove("grouper")
        init_wrapper = wrappers[0]
        for i in range(1, len(wrappers)):
            wrapper = wrappers[i]
            for k in common_keys:
                if k not in kwargs:
                    same_k = True
                    try:
                        if k in wrapper.config:
                            if not checks.is_deep_equal(init_wrapper.config[k], wrapper.config[k]):
                                same_k = False
                        elif "grouper" not in kwargs and k in wrapper.grouper.config:
                            if not checks.is_deep_equal(init_wrapper.grouper.config[k], wrapper.grouper.config[k]):
                                same_k = False
                        else:
                            same_k = False
                    except KeyError as e:
                        same_k = False
                    if not same_k:
                        raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        for k in common_keys:
            if k not in kwargs:
                if k in init_wrapper.config:
                    kwargs[k] = init_wrapper.config[k]
                elif "grouper" not in kwargs and k in init_wrapper.grouper.config:
                    kwargs[k] = init_wrapper.grouper.config[k]
                else:
                    raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[ArrayWrapperT],
        *wrappers: tp.MaybeTuple[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        stack_columns: bool = True,
        index_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
        keys: tp.Optional[tp.IndexLike] = None,
        index_stack_kwargs: tp.KwargsLike = None,
        verify_integrity: bool = True,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along rows.

        Concatenates indexes using `vectorbtpro.base.indexes.concat_indexes`.

        Frequency must be the same across all indexes. A custom frequency can be provided via `freq`.

        If column levels in some instances differ, they will be stacked upon each other.
        Custom columns can be provided via `columns`.

        If `group_by` is None, all instances must be either grouped or not, and they must
        contain the same group values and labels.

        All instances must contain the same keys and values in their configs and configs of their
        grouper instances, apart from those arguments provided explicitly via `kwargs`."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")
        if keys is not None and not isinstance(keys, pd.Index):
            keys = pd.Index(keys)

        if index is None:
            index = concat_indexes(
                [wrapper.index for wrapper in wrappers],
                index_concat_method=index_concat_method,
                keys=keys,
                index_stack_kwargs=index_stack_kwargs,
                verify_integrity=verify_integrity,
                axis=0,
            )
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            freq = infer_index_freq(index)
            if freq is None:
                new_freq = None
                for wrapper in wrappers:
                    if new_freq is None:
                        new_freq = wrapper.freq
                    else:
                        if new_freq is not None and wrapper.freq is not None and new_freq != wrapper.freq:
                            raise ValueError("Objects to be merged must have the same frequency")
                freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            new_columns = None
            for wrapper in wrappers:
                if new_columns is None:
                    new_columns = wrapper.columns
                else:
                    if not checks.is_index_equal(new_columns, wrapper.columns):
                        if not stack_columns:
                            raise ValueError("Objects to be merged must have the same columns")
                        new_columns = stack_indexes((new_columns, wrapper.columns), **resolve_dict(index_stack_kwargs))
            columns = new_columns
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                grouped = None
                for wrapper in wrappers:
                    wrapper_grouped = wrapper.grouper.is_grouped()
                    if grouped is None:
                        grouped = wrapper_grouped
                    else:
                        if grouped is not wrapper_grouped:
                            raise ValueError("Objects to be merged must be either grouped or not")
                if grouped:
                    new_group_by = None
                    for wrapper in wrappers:
                        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
                        wrapper_group_by = wrapper_grouped_index[wrapper_groups]
                        if new_group_by is None:
                            new_group_by = wrapper_group_by
                        else:
                            if not checks.is_index_equal(new_group_by, wrapper_group_by):
                                raise ValueError("Objects to be merged must have the same groups")
                    group_by = new_group_by
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            ndim = None
            for wrapper in wrappers:
                if ndim is None or wrapper.ndim > 1:
                    ndim = wrapper.ndim
            kwargs["ndim"] = ndim

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    @classmethod
    def column_stack(
        cls: tp.Type[ArrayWrapperT],
        *wrappers: tp.MaybeTuple[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        union_index: bool = True,
        col_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
        group_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = ("append", "factorize_each"),
        keys: tp.Optional[tp.IndexLike] = None,
        index_stack_kwargs: tp.KwargsLike = None,
        verify_integrity: bool = True,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along columns.

        If indexes are the same in each wrapper index, will use that index. If indexes differ and
        `union_index` is True, they will be merged into a single one by the set union operation.
        Otherwise, an error will be raised. The merged index must have no duplicates or mixed data,
        and must be monotonically increasing. A custom index can be provided via `index`.

        Frequency must be the same across all indexes. A custom frequency can be provided via `freq`.

        Concatenates columns and groups using `vectorbtpro.base.indexes.concat_indexes`.

        If any of the instances has `column_only_select` being enabled, the final wrapper will also enable it.
        If any of the instances has `group_select` or other grouping-related flags being disabled, the final
        wrapper will also disable them.

        All instances must contain the same keys and values in their configs and configs of their
        grouper instances, apart from those arguments provided explicitly via `kwargs`."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")
        if keys is not None and not isinstance(keys, pd.Index):
            keys = pd.Index(keys)

        for wrapper in wrappers:
            if wrapper.index.has_duplicates:
                raise ValueError("Index of some objects to be merged contains duplicates")
        if index is None:
            new_index = None
            for wrapper in wrappers:
                if new_index is None:
                    new_index = wrapper.index
                else:
                    if not checks.is_index_equal(new_index, wrapper.index):
                        if not union_index:
                            raise ValueError(
                                "Objects to be merged must have the same index. "
                                "Use union_index=True to merge index as well."
                            )
                        else:
                            if new_index.dtype != wrapper.index.dtype:
                                raise ValueError("Indexes to be merged must have the same data type")
                            new_index = new_index.union(wrapper.index)
            if not new_index.is_monotonic_increasing:
                raise ValueError("Merged index must be monotonically increasing")
            index = new_index
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            freq = infer_index_freq(index)
            if freq is None:
                new_freq = None
                for wrapper in wrappers:
                    if new_freq is None:
                        new_freq = wrapper.freq
                    else:
                        if new_freq is not None and wrapper.freq is not None and new_freq != wrapper.freq:
                            raise ValueError("Objects to be merged must have the same frequency")
                freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            columns = concat_indexes(
                [wrapper.columns for wrapper in wrappers],
                index_concat_method=col_concat_method,
                keys=keys,
                index_stack_kwargs=index_stack_kwargs,
                verify_integrity=verify_integrity,
                axis=1,
            )
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                any_grouped = False
                for wrapper in wrappers:
                    if wrapper.grouper.is_grouped():
                        any_grouped = True
                        break
                if any_grouped:
                    group_by = concat_indexes(
                        [wrapper.grouper.get_stretched_index() for wrapper in wrappers],
                        index_concat_method=group_concat_method,
                        keys=keys,
                        index_stack_kwargs=index_stack_kwargs,
                        verify_integrity=verify_integrity,
                        axis=2,
                    )
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            kwargs["ndim"] = 2
        if "grouped_ndim" not in kwargs:
            kwargs["grouped_ndim"] = None
        if "column_only_select" not in kwargs:
            column_only_select = None
            for wrapper in wrappers:
                if column_only_select is None or wrapper.column_only_select:
                    column_only_select = wrapper.column_only_select
            kwargs["column_only_select"] = column_only_select
        if "range_only_select" not in kwargs:
            range_only_select = None
            for wrapper in wrappers:
                if range_only_select is None or wrapper.range_only_select:
                    range_only_select = wrapper.range_only_select
            kwargs["range_only_select"] = range_only_select
        if "group_select" not in kwargs:
            group_select = None
            for wrapper in wrappers:
                if group_select is None or not wrapper.group_select:
                    group_select = wrapper.group_select
            kwargs["group_select"] = group_select
        if "grouper" not in kwargs:
            if "allow_enable" not in kwargs:
                allow_enable = None
                for wrapper in wrappers:
                    if allow_enable is None or not wrapper.grouper.allow_enable:
                        allow_enable = wrapper.grouper.allow_enable
                kwargs["allow_enable"] = allow_enable
            if "allow_disable" not in kwargs:
                allow_disable = None
                for wrapper in wrappers:
                    if allow_disable is None or not wrapper.grouper.allow_disable:
                        allow_disable = wrapper.grouper.allow_disable
                kwargs["allow_disable"] = allow_disable
            if "allow_modify" not in kwargs:
                allow_modify = None
                for wrapper in wrappers:
                    if allow_modify is None or not wrapper.grouper.allow_modify:
                        allow_modify = wrapper.grouper.allow_modify
                kwargs["allow_modify"] = allow_modify

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "index",
        "columns",
        "ndim",
        "freq",
        "column_only_select",
        "range_only_select",
        "group_select",
        "grouped_ndim",
        "grouper",
    }

    def __init__(
        self,
        index: tp.IndexLike,
        columns: tp.IndexLike,
        ndim: tp.Optional[int] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        grouped_ndim: tp.Optional[int] = None,
        grouper: tp.Optional[Grouper] = None,
        **kwargs,
    ) -> None:

        checks.assert_not_none(index)
        checks.assert_not_none(columns)
        index = try_to_datetime_index(index)
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if ndim is None:
            if len(columns) == 1 and not isinstance(columns, pd.MultiIndex):
                ndim = 1
            else:
                ndim = 2
        else:
            if len(columns) > 1:
                ndim = 2

        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        grouper_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in grouper_arg_names:
                grouper_kwargs[k] = kwargs.pop(k)
        if grouper is None:
            grouper = Grouper(columns, **grouper_kwargs)
        elif not checks.is_index_equal(columns, grouper.index) or len(grouper_kwargs) > 0:
            grouper = grouper.replace(index=columns, **grouper_kwargs)

        PandasIndexer.__init__(self)
        Configured.__init__(
            self,
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            column_only_select=column_only_select,
            range_only_select=range_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
            grouper=grouper,
            **kwargs,
        )

        self._index = index
        self._columns = columns
        self._ndim = ndim
        self._freq = freq
        self._column_only_select = column_only_select
        self._range_only_select = range_only_select
        self._group_select = group_select
        self._grouper = grouper
        self._grouped_ndim = grouped_ndim

    def indexing_func_meta(
        self: ArrayWrapperT,
        pd_indexing_func: tp.PandasIndexingFunc,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        return_slices: bool = True,
        return_none_slices: bool = True,
        return_scalars: bool = True,
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
    ) -> dict:
        """Perform indexing on `ArrayWrapper` and also return metadata.

        Takes into account column grouping.

        Flipping rows and columns is not allowed. If one row is selected, the result will still be
        a Series when indexing a Series and a DataFrame when indexing a DataFrame.

        Set `column_only_select` to True to index the array wrapper as a Series of columns/groups.
        This way, selection of index (axis 0) can be avoided. Set `range_only_select` to True to
        allow selection of rows only using slices. Set `group_select` to True to allow selection of groups.
        Otherwise, indexing is performed on columns, even if grouping is enabled. Takes effect only if
        grouping is enabled.

        Returns the new array wrapper, row indices, column indices, and group indices.
        If `return_slices` is True (default), indices will be returned as a slice if they were
        identified as a range. If `return_none_slices` is True (default), indices will be returned as a slice
        `(None, None, None)` if the axis hasn't been changed.

        !!! note
            If `column_only_select` is True, make sure to index the array wrapper
            as a Series of columns rather than a DataFrame. For example, the operation
            `.iloc[:, :2]` should become `.iloc[:2]`. Operations are not allowed if the
            object is already a Series and thus has only one column/group."""
        if column_only_select is None:
            column_only_select = self.column_only_select
        if range_only_select is None:
            range_only_select = self.range_only_select
        if group_select is None:
            group_select = self.group_select
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        _self = self.regroup(group_by)
        group_select = group_select and _self.grouper.is_grouped()
        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            if group_select:
                columns = _self.get_columns()
            else:
                columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if group_select:
            # Groups as columns
            i_wrapper = ArrayWrapper(index, columns, _self.get_ndim())
        else:
            # Columns as columns
            i_wrapper = ArrayWrapper(index, columns, _self.ndim)
        n_rows = len(index)
        n_cols = len(columns)

        def _resolve_arr(arr, n):
            if checks.is_np_array(arr) and is_range(arr):
                if arr[0] == 0 and arr[-1] == n - 1:
                    if return_none_slices:
                        return slice(None, None, None), False
                    return arr, False
                if return_slices:
                    return slice(arr[0], arr[-1] + 1, None), True
                return arr, True
            if isinstance(arr, np.integer):
                arr = arr.item()
            columns_changed = True
            if isinstance(arr, int):
                if arr == 0 and n == 1:
                    columns_changed = False
                if not return_scalars:
                    arr = np.array([arr])
            return arr, columns_changed

        if column_only_select:
            if i_wrapper.ndim == 1:
                raise IndexingError("Columns only: This object already contains one column of data")
            try:
                col_mapper = pd_indexing_func(i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns))
            except pd.core.indexing.IndexingError as e:
                warnings.warn(
                    "Columns only: Make sure to treat this object as a Series of columns rather than a DataFrame",
                    stacklevel=2,
                )
                raise e
            if checks.is_series(col_mapper):
                new_columns = col_mapper.index
                col_idxs = col_mapper.values
                new_ndim = 2
            else:
                new_columns = columns[[col_mapper]]
                col_idxs = col_mapper
                new_ndim = 1
            new_index = index
            row_idxs = np.arange(len(index))
        else:
            init_row_mapper_values = reshaping.broadcast_array_to(np.arange(n_rows)[:, None], (n_rows, n_cols))
            init_row_mapper = i_wrapper.wrap(init_row_mapper_values, index=index, columns=columns)
            row_mapper = pd_indexing_func(init_row_mapper)
            if i_wrapper.ndim == 1:
                if not checks.is_series(row_mapper):
                    row_idxs = np.array([row_mapper])
                    new_index = index[row_idxs]
                else:
                    row_idxs = row_mapper.values
                    new_index = indexes.get_index(row_mapper, 0)
                col_idxs = 0
                new_columns = columns
                new_ndim = 1
            else:
                init_col_mapper_values = reshaping.broadcast_array_to(np.arange(n_cols)[None], (n_rows, n_cols))
                init_col_mapper = i_wrapper.wrap(init_col_mapper_values, index=index, columns=columns)
                col_mapper = pd_indexing_func(init_col_mapper)

                if checks.is_frame(col_mapper):
                    # Multiple rows and columns selected
                    row_idxs = row_mapper.values[:, 0]
                    col_idxs = col_mapper.values[0]
                    new_index = indexes.get_index(row_mapper, 0)
                    new_columns = indexes.get_index(col_mapper, 1)
                    new_ndim = 2
                elif checks.is_series(col_mapper):
                    multi_index = isinstance(index, pd.MultiIndex)
                    multi_columns = isinstance(columns, pd.MultiIndex)
                    multi_name = isinstance(col_mapper.name, tuple)
                    if multi_index and multi_name and col_mapper.name in index:
                        one_row = True
                    elif not multi_index and not multi_name and col_mapper.name in index:
                        one_row = True
                    else:
                        one_row = False
                    if multi_columns and multi_name and col_mapper.name in columns:
                        one_col = True
                    elif not multi_columns and not multi_name and col_mapper.name in columns:
                        one_col = True
                    else:
                        one_col = False
                    if (one_row and one_col) or (not one_row and not one_col):
                        one_row = np.all(row_mapper.values == row_mapper.values.item(0))
                        one_col = np.all(col_mapper.values == col_mapper.values.item(0))
                    if (one_row and one_col) or (not one_row and not one_col):
                        raise IndexingError("Could not parse indexing operation")
                    if one_row:
                        # One row selected
                        row_idxs = row_mapper.values[[0]]
                        col_idxs = col_mapper.values
                        new_index = index[row_idxs]
                        new_columns = indexes.get_index(col_mapper, 0)
                        new_ndim = 2
                    else:
                        # One column selected
                        row_idxs = row_mapper.values
                        col_idxs = col_mapper.values[0]
                        new_index = indexes.get_index(row_mapper, 0)
                        new_columns = columns[[col_idxs]]
                        new_ndim = 1
                else:
                    # One row and column selected
                    row_idxs = np.array([row_mapper])
                    col_idxs = col_mapper
                    new_index = index[row_idxs]
                    new_columns = columns[[col_idxs]]
                    new_ndim = 1

        if _self.grouper.is_grouped():
            # Grouping enabled
            if np.asarray(row_idxs).ndim == 0:
                raise IndexingError("Flipping index and columns is not allowed")

            if group_select:
                # Selection based on groups
                # Get indices of columns corresponding to selected groups
                group_idxs = col_idxs
                col_idxs, new_groups = _self.grouper.select_groups(group_idxs)
                ungrouped_columns = _self.columns[col_idxs]
                if new_ndim == 1 and len(ungrouped_columns) == 1:
                    ungrouped_ndim = 1
                    col_idxs = col_idxs[0]
                else:
                    ungrouped_ndim = 2

                row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
                if range_only_select and rows_changed:
                    if not isinstance(row_idxs, slice):
                        raise ValueError("Rows can be selected only by slicing")
                    if row_idxs.step not in (1, None):
                        raise ValueError("Slice for selecting rows must have a step of 1 or None")
                col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
                group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
                return dict(
                    new_wrapper=_self.replace(
                        **merge_dicts(
                            dict(
                                index=new_index,
                                columns=ungrouped_columns,
                                ndim=ungrouped_ndim,
                                grouped_ndim=new_ndim,
                                group_by=new_columns[new_groups],
                            ),
                            wrapper_kwargs,
                        )
                    ),
                    row_idxs=row_idxs,
                    rows_changed=rows_changed,
                    col_idxs=col_idxs,
                    columns_changed=columns_changed,
                    group_idxs=group_idxs,
                    groups_changed=groups_changed,
                )

            # Selection based on columns
            group_idxs = _self.grouper.get_groups()[col_idxs]
            new_group_by = _self.grouper.group_by[reshaping.to_1d_array(col_idxs)]
            row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
            if range_only_select and rows_changed:
                if not isinstance(row_idxs, slice):
                    raise ValueError("Rows can be selected only by slicing")
                if row_idxs.step not in (1, None):
                    raise ValueError("Slice for selecting rows must have a step of 1 or None")
            col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
            group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
            return dict(
                new_wrapper=_self.replace(
                    **merge_dicts(
                        dict(
                            index=new_index,
                            columns=new_columns,
                            ndim=new_ndim,
                            grouped_ndim=None,
                            group_by=new_group_by,
                        ),
                        wrapper_kwargs,
                    )
                ),
                row_idxs=row_idxs,
                rows_changed=rows_changed,
                col_idxs=col_idxs,
                columns_changed=columns_changed,
                group_idxs=group_idxs,
                groups_changed=groups_changed,
            )

        # Grouping disabled
        row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
        if range_only_select and rows_changed:
            if not isinstance(row_idxs, slice):
                raise ValueError("Rows can be selected only by slicing")
            if row_idxs.step not in (1, None):
                raise ValueError("Slice for selecting rows must have a step of 1 or None")
        col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
        return dict(
            new_wrapper=_self.replace(
                **merge_dicts(
                    dict(
                        index=new_index,
                        columns=new_columns,
                        ndim=new_ndim,
                        grouped_ndim=None,
                        group_by=None,
                    ),
                    wrapper_kwargs,
                )
            ),
            row_idxs=row_idxs,
            rows_changed=rows_changed,
            col_idxs=col_idxs,
            columns_changed=columns_changed,
            group_idxs=col_idxs,
            groups_changed=columns_changed,
        )

    def indexing_func(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform indexing on `ArrayWrapper`"""
        return self.indexing_func_meta(*args, **kwargs)["new_wrapper"]

    @staticmethod
    def select_from_flex_array(
        arr: tp.ArrayLike,
        row_idxs: tp.Union[int, tp.Array1d, slice] = None,
        col_idxs: tp.Union[int, tp.Array1d, slice] = None,
        rows_changed: bool = True,
        columns_changed: bool = True,
        rotate_rows: bool = False,
        rotate_cols: bool = True,
    ) -> tp.Array2d:
        """Select rows and columns from a flexible array.

        Always returns a 2-dim NumPy array."""
        new_arr = arr_2d = reshaping.to_2d_array(arr)
        if row_idxs is not None and rows_changed:
            if arr_2d.shape[0] > 1:
                if isinstance(row_idxs, slice):
                    max_idx = row_idxs.stop - 1
                else:
                    row_idxs = reshaping.to_1d_array(row_idxs)
                    max_idx = np.max(row_idxs)
                if arr_2d.shape[0] <= max_idx:
                    if rotate_rows:
                        new_arr = new_arr[row_idxs % arr_2d.shape[0], :]
                    else:
                        new_arr = new_arr[row_idxs, :]
                else:
                    new_arr = new_arr[row_idxs, :]
        if col_idxs is not None and columns_changed:
            if arr_2d.shape[1] > 1:
                if isinstance(col_idxs, slice):
                    max_idx = col_idxs.stop - 1
                else:
                    col_idxs = reshaping.to_1d_array(col_idxs)
                    max_idx = np.max(col_idxs)
                if arr_2d.shape[1] <= max_idx:
                    if rotate_cols:
                        new_arr = new_arr[:, col_idxs % arr_2d.shape[1]]
                    else:
                        new_arr = new_arr[:, col_idxs]
                else:
                    new_arr = new_arr[:, col_idxs]
        return new_arr

    def get_resampler(self, *args, **kwargs) -> tp.Union[Resampler, tp.PandasResampler]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.get_resampler`."""
        return self.index_acc.get_resampler(*args, **kwargs)

    def resample_meta(self: ArrayWrapperT, *args, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> dict:
        """Perform resampling on `ArrayWrapper` and also return metadata.

        `*args` and `**kwargs` are passed to `ArrayWrapper.get_resampler`."""
        resampler = self.get_resampler(*args, **kwargs)
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = _resampler.target_index
        if "freq" not in wrapper_kwargs:
            wrapper_kwargs["freq"] = infer_index_freq(wrapper_kwargs["index"], freq=_resampler.target_freq)
        new_wrapper = self.replace(**wrapper_kwargs)
        return dict(resampler=resampler, new_wrapper=new_wrapper)

    def resample(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform resampling on `ArrayWrapper`.

        Uses `ArrayWrapper.resample_meta`."""
        return self.resample_meta(*args, **kwargs)["new_wrapper"]

    @property
    def index(self) -> tp.Index:
        """Index."""
        return self._index

    @cached_property(whitelist=True)
    def index_acc(self) -> BaseIDXAccessorT:
        """Get index accessor of the type `vectorbtpro.base.accessors.BaseIDXAccessor`."""
        from vectorbtpro.base.accessors import BaseIDXAccessor

        return BaseIDXAccessor(self.index, freq=self._freq)

    @property
    def ns_index(self) -> tp.Array1d:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.to_ns`."""
        return self.index_acc.to_ns()

    def get_period_ns_index(self, *args, **kwargs) -> tp.Array1d:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.to_period_ns`."""
        return self.index_acc.to_period_ns(*args, **kwargs)

    @property
    def columns(self) -> tp.Index:
        """Columns."""
        return self._columns

    def get_columns(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """Get group-aware `ArrayWrapper.columns`."""
        return self.resolve(group_by=group_by).columns

    @property
    def name(self) -> tp.Any:
        """Name."""
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

    def get_name(self, group_by: tp.GroupByLike = None) -> tp.Any:
        """Get group-aware `ArrayWrapper.name`."""
        return self.resolve(group_by=group_by).name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    def get_ndim(self, group_by: tp.GroupByLike = None) -> int:
        """Get group-aware `ArrayWrapper.ndim`."""
        return self.resolve(group_by=group_by).ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape."""
        if self.ndim == 1:
            return (len(self.index),)
        return len(self.index), len(self.columns)

    def get_shape(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape`."""
        return self.resolve(group_by=group_by).shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object was two-dimensional."""
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    def get_shape_2d(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape_2d`."""
        return self.resolve(group_by=group_by).shape_2d

    def get_freq(self, *args, **kwargs) -> tp.Union[None, float, tp.PandasFrequency]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.get_freq`."""
        return self.index_acc.get_freq(*args, **kwargs)

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.freq`."""
        return self.index_acc.freq

    @property
    def ns_freq(self) -> tp.Optional[int]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.ns_freq`."""
        return self.index_acc.ns_freq

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.any_freq`."""
        return self.index_acc.any_freq

    @property
    def period(self) -> int:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.period`."""
        return self.index_acc.period

    @property
    def dt_period(self) -> float:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.dt_period`."""
        return self.index_acc.dt_period

    def arr_to_timedelta(self, *args, **kwargs) -> tp.Union[pd.Index, tp.MaybeArray]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.arr_to_timedelta`."""
        return self.index_acc.arr_to_timedelta(*args, **kwargs)

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on columns only."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        column_only_select = self._column_only_select
        if column_only_select is None:
            column_only_select = wrapping_cfg["column_only_select"]
        return column_only_select

    @property
    def range_only_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on rows using slices only."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        range_only_select = self._range_only_select
        if range_only_select is None:
            range_only_select = wrapping_cfg["range_only_select"]
        return range_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Whether to allow indexing on groups."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        group_select = self._group_select
        if group_select is None:
            group_select = wrapping_cfg["group_select"]
        return group_select

    @property
    def grouper(self) -> Grouper:
        """Column grouper."""
        return self._grouper

    @property
    def grouped_ndim(self) -> int:
        """Number of dimensions under column grouping."""
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    @cached_method(whitelist=True)
    def regroup(self: ArrayWrapperT, group_by: tp.GroupByLike, **kwargs) -> ArrayWrapperT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself."""
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            grouped_ndim = None
            if self.grouper.is_grouped(group_by=group_by):
                if not self.grouper.is_group_count_changed(group_by=group_by):
                    grouped_ndim = self.grouped_ndim
            return self.replace(grouped_ndim=grouped_ndim, group_by=group_by, **kwargs)
        if len(kwargs) > 0:
            return self.replace(**kwargs)
        return self  # important for keeping cache

    def flip(self: ArrayWrapperT, **kwargs) -> ArrayWrapperT:
        """Flip index and columns."""
        if "grouper" not in kwargs:
            kwargs["grouper"] = None
        return self.replace(index=self.columns, columns=self.index, **kwargs)

    @cached_method(whitelist=True)
    def resolve(self: ArrayWrapperT, group_by: tp.GroupByLike = None, **kwargs) -> ArrayWrapperT:
        """Resolve this object.

        Replaces columns and other metadata with groups."""
        _self = self.regroup(group_by=group_by, **kwargs)
        if _self.grouper.is_grouped():
            return _self.replace(
                columns=_self.grouper.get_index(),
                ndim=_self.grouped_ndim,
                grouped_ndim=None,
                group_by=None,
            )
        return _self  # important for keeping cache

    def get_index_grouper(self, *args, **kwargs) -> Grouper:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`."""
        return self.index_acc.get_grouper(*args, **kwargs)

    def wrap(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        zero_to_none: tp.Optional[bool] = None,
        force_2d: bool = False,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        min_precision: tp.Union[None, int, str] = None,
        max_precision: tp.Union[None, int, str] = None,
        prec_float_only: tp.Optional[bool] = None,
        prec_check_bounds: tp.Optional[bool] = None,
        prec_strict: tp.Optional[bool] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SeriesFrame:
        """Wrap a NumPy array using the stored metadata.

        Runs the following pipeline:

        1) Converts to NumPy array
        2) Fills NaN (optional)
        3) Wraps using index, columns, and dtype (optional)
        4) Converts to index (optional)
        5) Converts to timedelta using `ArrayWrapper.arr_to_timedelta` (optional)"""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if zero_to_none is None:
            zero_to_none = wrapping_cfg["zero_to_none"]
        if min_precision is None:
            min_precision = wrapping_cfg["min_precision"]
        if max_precision is None:
            max_precision = wrapping_cfg["max_precision"]
        if prec_float_only is None:
            prec_float_only = wrapping_cfg["prec_float_only"]
        if prec_check_bounds is None:
            prec_check_bounds = wrapping_cfg["prec_check_bounds"]
        if prec_strict is None:
            prec_strict = wrapping_cfg["prec_strict"]
        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        _self = self.resolve(group_by=group_by)

        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if len(columns) == 1:
            name = columns[0]
            if zero_to_none and name == 0:  # was a Series before
                name = None
        else:
            name = None

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap(arr):
            orig_arr = arr
            arr = np.asarray(arr)
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            shape_2d = (arr.shape[0] if arr.ndim > 0 else 1, arr.shape[1] if arr.ndim > 1 else 1)
            target_shape_2d = (len(index), len(columns))
            if shape_2d != target_shape_2d:
                if isinstance(orig_arr, (pd.Series, pd.DataFrame)):
                    arr = reshaping.align_pd_arrays(orig_arr, to_index=index, to_columns=columns).values
                arr = reshaping.broadcast_array_to(arr, target_shape_2d)
            arr = reshaping.soft_to_ndim(arr, self.ndim)
            if min_precision is not None:
                arr = cast_to_min_precision(
                    arr,
                    min_precision,
                    float_only=prec_float_only,
                )
            if max_precision is not None:
                arr = cast_to_max_precision(
                    arr,
                    max_precision,
                    float_only=prec_float_only,
                    check_bounds=prec_check_bounds,
                    strict=prec_strict,
                )
            if arr.ndim == 1:
                if force_2d:
                    return _apply_dtype(pd.DataFrame(arr[:, None], index=index, columns=columns))
                return _apply_dtype(pd.Series(arr, index=index, name=name))
            if arr.ndim == 2:
                if not force_2d and arr.shape[1] == 1 and _self.ndim == 1:
                    return _apply_dtype(pd.Series(arr[:, 0], index=index, name=name))
                return _apply_dtype(pd.DataFrame(arr, index=index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
        if to_timedelta:
            # Convert to timedelta
            out = self.arr_to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def wrap_reduced(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        name_or_index: tp.NameIndex = None,
        columns: tp.Optional[tp.IndexLike] = None,
        force_1d: bool = False,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.MaybeSeriesFrame:
        """Wrap result of reduction.

        `name_or_index` can be the name of the resulting series if reducing to a scalar per column,
        or the index of the resulting series/dataframe if reducing to an array per column.
        `columns` can be set to override object's default columns.

        See `ArrayWrapper.wrap` for the pipeline."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        checks.assert_not_none(self.ndim)
        _self = self.resolve(group_by=group_by)

        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        if to_index:
            if dtype is None:
                dtype = np.int_
            if fillna is None:
                fillna = -1

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap_reduced(arr):
            nonlocal name_or_index

            arr = np.asarray(arr)
            if force_1d and arr.ndim == 0:
                arr = arr[None]
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            if arr.ndim == 0:
                # Scalar per Series/DataFrame
                return _apply_dtype(pd.Series(arr))[0]
            if arr.ndim == 1:
                if not force_1d and _self.ndim == 1:
                    if arr.shape[0] == 1:
                        # Scalar per Series/DataFrame with one column
                        return _apply_dtype(pd.Series(arr))[0]
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Scalar per column in DataFrame
                return _apply_dtype(pd.Series(arr, index=columns, name=name_or_index))
            if arr.ndim == 2:
                if arr.shape[1] == 1 and _self.ndim == 1:
                    arr = reshaping.soft_to_ndim(arr, 1)
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Array per column in DataFrame
                if isinstance(name_or_index, str):
                    name_or_index = None
                return _apply_dtype(pd.DataFrame(arr, index=name_or_index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap_reduced(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            elif checks.is_frame(out):
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = self.index[out] if out != -1 else np.nan
        if to_timedelta:
            # Convert to timedelta
            out = self.arr_to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def concat_arrs(
        self,
        *objs: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray1d:
        """Stack reduced objects along columns and wrap the final object."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            new_objs.append(reshaping.to_1d_array(obj))

        stacked_obj = np.concatenate(new_objs)
        if wrap:
            return _self.wrap_reduced(stacked_obj, **kwargs)
        return stacked_obj

    def row_stack_arrs(
        self,
        *objs: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray:
        """Stack objects along rows and wrap the final object."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            obj = reshaping.to_2d_array(obj)
            if obj.shape[1] != _self.shape_2d[1]:
                if obj.shape[1] != 1:
                    raise ValueError(f"Cannot broadcast {obj.shape[1]} to {_self.shape_2d[1]} columns")
                obj = np.repeat(obj, _self.shape_2d[1], axis=1)
            new_objs.append(obj)

        stacked_obj = np.row_stack(new_objs)
        if wrap:
            return _self.wrap(stacked_obj, **kwargs)
        return stacked_obj

    def column_stack_arrs(
        self,
        *objs: tp.ArrayLike,
        reindex_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray2d:
        """Stack objects along columns and wrap the final object.

        `reindex_kwargs` will be passed to
        [pandas.DataFrame.reindex](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html)."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            if not checks.is_index_equal(obj.index, _self.index, check_names=False):
                was_bool = (isinstance(obj, pd.Series) and obj.dtype == "bool") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "bool").all()
                )
                obj = obj.reindex(_self.index, **resolve_dict(reindex_kwargs))
                is_object = (isinstance(obj, pd.Series) and obj.dtype == "object") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "object").all()
                )
                if was_bool and is_object:
                    obj = obj.astype(None)
            new_objs.append(reshaping.to_2d_array(obj))

        stacked_obj = np.column_stack(new_objs)
        if wrap:
            return _self.wrap(stacked_obj, **kwargs)
        return stacked_obj

    def dummy(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Create a dummy Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.empty(_self.shape), **kwargs)

    def fill(self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Fill a Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.full(_self.shape_2d, fill_value), **kwargs)

    def fill_reduced(self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Fill a reduced Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap_reduced(np.full(_self.shape_2d[1], fill_value), **kwargs)

    def get_index_points(self, *args, **kwargs) -> tp.Array1d:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.get_index_points`."""
        return self.index_acc.get_index_points(*args, **kwargs)

    def get_index_ranges(self, *args, **kwargs) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """See `vectorbtpro.base.accessors.BaseIDXAccessor.get_index_ranges`."""
        return self.index_acc.get_index_ranges(*args, **kwargs)

    def fill_and_set(
        self,
        idx_setter: tp.Union[index_dict, IdxSetter, IdxSetterFactory],
        keep_flex: bool = False,
        fill_value: tp.Scalar = np.nan,
        **kwargs,
    ) -> tp.AnyArray:
        """Fill a new array using an index object such as `vectorbtpro.base.indexing.index_dict`.

        Will be wrapped with `vectorbtpro.base.indexing.IdxSetter` if not already.

        Will call `vectorbtpro.base.indexing.IdxSetter.fill_and_set`.

        Usage:
            * Set a single row:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd
            >>> import numpy as np

            >>> index = pd.date_range("2020", periods=5)
            >>> columns = pd.Index(["a", "b", "c"])
            >>> wrapper = vbt.ArrayWrapper(index, columns)

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     1: 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     "2020-01-02": 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     "2020-01-02": [1, 2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN
            ```

            * Set multiple rows:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     (1, 3): [2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  3.0  3.0  3.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  1.0  2.0  3.0
            2020-01-05  NaN  NaN  NaN
            ```

            * Set rows using slices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.hslice(1, 3): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.hslice("2020-01-02", "2020-01-04"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1], [2]]
            ... }))
                          a    b    c
            2020-01-01  1.0  1.0  1.0
            2020-01-02  1.0  1.0  1.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  2.0  2.0  2.0

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  1.0  2.0  3.0
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  4.0  5.0  6.0
            ```

            * Set rows using index points:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.pointidx(every="2D"): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  2.0  2.0  2.0
            ```

            * Set rows using index ranges:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.rangeidx(
            ...         start=("2020-01-01", "2020-01-03"),
            ...         end=("2020-01-02", "2020-01-05")
            ...     ): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```

            * Set column indices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx("a"): 2
            ... }))
                          a   b   c
            2020-01-01  2.0 NaN NaN
            2020-01-02  2.0 NaN NaN
            2020-01-03  2.0 NaN NaN
            2020-01-04  2.0 NaN NaN
            2020-01-05  2.0 NaN NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx(("a", "b")): [1, 2]
            ... }))
                          a    b   c
            2020-01-01  1.0  2.0 NaN
            2020-01-02  1.0  2.0 NaN
            2020-01-03  1.0  2.0 NaN
            2020-01-04  1.0  2.0 NaN
            2020-01-05  1.0  2.0 NaN

            >>> multi_columns = pd.MultiIndex.from_arrays(
            ...     [["a", "a", "b", "b"], [1, 2, 1, 2]],
            ...     names=["c1", "c2"]
            ... )
            >>> multi_wrapper = vbt.ArrayWrapper(index, multi_columns)

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx(("a", 2)): 2
            ... }))
            c1           a        b
            c2           1    2   1   2
            2020-01-01 NaN  2.0 NaN NaN
            2020-01-02 NaN  2.0 NaN NaN
            2020-01-03 NaN  2.0 NaN NaN
            2020-01-04 NaN  2.0 NaN NaN
            2020-01-05 NaN  2.0 NaN NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx("b", level="c1"): [3, 4]
            ... }))
            c1           a        b
            c2           1   2    1    2
            2020-01-01 NaN NaN  3.0  4.0
            2020-01-02 NaN NaN  3.0  4.0
            2020-01-03 NaN NaN  3.0  4.0
            2020-01-04 NaN NaN  3.0  4.0
            2020-01-05 NaN NaN  3.0  4.0
            ```

            * Set row and column indices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(2, 2): 2
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  NaN
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(("2020-01-01", "2020-01-03"), 2): [1, 2]
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  1.0
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(("2020-01-01", "2020-01-03"), (0, 2)): [[1, 2], [3, 4]]
            ... }))
                          a   b    c
            2020-01-01  1.0 NaN  2.0
            2020-01-02  NaN NaN  NaN
            2020-01-03  3.0 NaN  4.0
            2020-01-04  NaN NaN  NaN
            2020-01-05  NaN NaN  NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(
            ...         vbt.pointidx(every="2d"),
            ...         vbt.colidx(1, level="c2")
            ...     ): [[1, 2]]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  2.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  1.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  1.0 NaN  2.0 NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(
            ...         vbt.pointidx(every="2d"),
            ...         vbt.colidx(1, level="c2")
            ...     ): [[1], [2], [3]]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  1.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  2.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  3.0 NaN  3.0 NaN
            ```

            * Set rows using a template:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.RepEval("index.day % 2 == 0"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```
        """
        if isinstance(idx_setter, index_dict):
            idx_setter = IdxDict(idx_setter)
        if isinstance(idx_setter, IdxSetterFactory):
            idx_setter = idx_setter.get()
            if not isinstance(idx_setter, IdxSetter):
                raise ValueError("Index setter factory must return exactly one index setter")
        checks.assert_instance_of(idx_setter, IdxSetter)
        arr = idx_setter.fill_and_set(
            self.shape,
            keep_flex=keep_flex,
            fill_value=fill_value,
            index=self.index,
            columns=self.columns,
            freq=self.freq,
            **kwargs,
        )
        if not keep_flex:
            return self.wrap(arr, group_by=False)
        return arr

    def split(
        self,
        splitter: tp.Union[str, SplitterT, tp.Callable],
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        splitter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **take_kwargs,
    ) -> tp.Any:
        """Split using `vectorbtpro.generic.splitting.base.Splitter`."""
        from vectorbtpro.generic.splitting.base import Splitter

        if splitter_cls is None:
            splitter_cls = Splitter
        if not isinstance(splitter, splitter_cls):
            if isinstance(splitter, str):
                splitter = getattr(splitter_cls, splitter)
            splitter = splitter(self.index, template_context=template_context, **splitter_kwargs)
        return splitter.take(self, template_context=template_context, **take_kwargs)


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, PandasIndexer, AttrResolverMixin):
    """Class that uses `ArrayWrapper` globally."""

    @classmethod
    def resolve_row_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along rows."""
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along columns."""
        return kwargs

    @classmethod
    def resolve_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking.

        Should be called after `Wrapping.resolve_row_stack_kwargs` or `Wrapping.resolve_column_stack_kwargs`."""
        if len(wrappings) == 1:
            wrappings = wrappings[0]
        wrappings = list(wrappings)

        common_keys = set()
        for wrapping in wrappings:
            common_keys = common_keys.union(set(wrapping.config.keys()))
        init_wrapping = wrappings[0]
        for i in range(1, len(wrappings)):
            wrapping = wrappings[i]
            for k in common_keys:
                if k not in kwargs:
                    same_k = True
                    try:
                        if k in wrapping.config:
                            if not checks.is_deep_equal(init_wrapping.config[k], wrapping.config[k]):
                                same_k = False
                        else:
                            same_k = False
                    except KeyError as e:
                        same_k = False
                    if not same_k:
                        raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        for k in common_keys:
            if k not in kwargs:
                if k in init_wrapping.config:
                    kwargs[k] = init_wrapping.config[k]
                else:
                    raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[WrappingT],
        *args: tp.MaybeTuple[WrappingT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> WrappingT:
        """Stack multiple `Wrapping` instances along rows.

        Should use `ArrayWrapper.row_stack`."""
        raise NotImplementedError

    @classmethod
    def column_stack(
        cls: tp.Type[WrappingT],
        *args: tp.MaybeTuple[WrappingT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> WrappingT:
        """Stack multiple `Wrapping` instances along columns.

        Should use `ArrayWrapper.column_stack`."""
        raise NotImplementedError

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "wrapper",
    }

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        checks.assert_instance_of(wrapper, ArrayWrapper)
        self._wrapper = wrapper

        Configured.__init__(self, wrapper=wrapper, **kwargs)
        PandasIndexer.__init__(self)
        AttrResolverMixin.__init__(self)

    def indexing_func(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform indexing on `Wrapping`."""
        new_wrapper = self.wrapper.indexing_func(
            *args,
            column_only_select=self.column_only_select,
            range_only_select=self.range_only_select,
            group_select=self.group_select,
            **kwargs,
        )
        return self.replace(wrapper=new_wrapper)

    def resample(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform resampling on `Wrapping`.

        When overriding, make sure to create a resampler by passing `*args` and `**kwargs`
        to `ArrayWrapper.get_resampler`."""
        raise NotImplementedError

    @property
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper of the type `ArrayWrapper`."""
        return self._wrapper

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.column_only_select`."""
        column_only_select = getattr(self, "_column_only_select", None)
        if column_only_select is None:
            return self.wrapper.column_only_select
        return column_only_select

    @property
    def range_only_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.range_only_select`."""
        range_only_select = getattr(self, "_range_only_select", None)
        if range_only_select is None:
            return self.wrapper.range_only_select
        return range_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.group_select`."""
        group_select = getattr(self, "_group_select", None)
        if group_select is None:
            return self.wrapper.group_select
        return group_select

    def regroup(self: WrappingT, group_by: tp.GroupByLike, **kwargs) -> WrappingT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself.

        `**kwargs` will be passed to `ArrayWrapper.regroup`."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.replace(wrapper=self.wrapper.regroup(group_by, **kwargs))
        return self  # important for keeping cache

    def resolve_self(
        self: AttrResolverMixinT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.ClassVar[tp.Optional[tp.Set[str]]] = None,
        impacts_caching: bool = True,
        silence_warnings: tp.Optional[bool] = None,
    ) -> AttrResolverMixinT:
        """Resolve self.

        Creates a copy of this instance if a different `freq` can be found in `cond_kwargs`."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()
        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        if "freq" in cond_kwargs:
            wrapper_copy = self.wrapper.replace(freq=cond_kwargs["freq"])

            if wrapper_copy.freq != self.wrapper.freq:
                if not silence_warnings:
                    warnings.warn(
                        f"Changing the frequency will create a copy of this object. "
                        f"Consider setting it upon object creation to re-use existing cache.",
                        stacklevel=2,
                    )
                self_copy = self.replace(wrapper=wrapper_copy)
                for alias in self.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs["freq"] = self_copy.wrapper.freq
                if impacts_caching:
                    cond_kwargs["use_caching"] = False
                return self_copy
        return self

    def select_col(self: WrappingT, column: tp.Any = None, group_by: tp.GroupByLike = None, **kwargs) -> WrappingT:
        """Select one column/group.

        `column` can be a label-based position as well as an integer position (if label fails)."""
        _self = self.regroup(group_by, **kwargs)

        def _check_out_dim(out: WrappingT) -> WrappingT:
            if out.wrapper.get_ndim() == 2:
                if out.wrapper.get_shape_2d()[1] == 1:
                    if out.column_only_select:
                        return out.iloc[0]
                    return out.iloc[:, 0]
                if _self.wrapper.grouper.is_grouped():
                    raise TypeError("Could not select one group: multiple groups returned")
                else:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is None:
            if _self.wrapper.get_ndim() == 2 and _self.wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if _self.wrapper.grouper.is_grouped():
                if _self.wrapper.grouped_ndim == 1:
                    raise TypeError("This object already contains one group of data")
                if column not in _self.wrapper.get_columns():
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Group '{column}' not found")
            else:
                if _self.wrapper.ndim == 1:
                    raise TypeError("This object already contains one column of data")
                if column not in _self.wrapper.columns:
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Column '{column}' not found")
            return _check_out_dim(_self[column])
        if _self.wrapper.grouper.is_grouped():
            if _self.wrapper.grouped_ndim == 1:
                return _self
            raise TypeError("Only one group is allowed. Use indexing or column argument.")
        if _self.wrapper.ndim == 1:
            return _self
        raise TypeError("Only one column is allowed. Use indexing or column argument.")

    @class_or_instancemethod
    def select_col_from_obj(
        cls_or_self,
        obj: tp.Optional[tp.SeriesFrame],
        column: tp.Any = None,
        obj_ungrouped: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> tp.MaybeSeries:
        """Select one column/group from a pandas object.

        `column` can be a label-based position as well as an integer position (if label fails)."""
        if not isinstance(cls_or_self, type) and wrapper is None:
            wrapper = cls_or_self.wrapper
        if obj is None:
            return None

        def _check_out_dim(out: tp.SeriesFrame, from_df: bool) -> tp.Series:
            bad_shape = False
            if from_df and isinstance(out, pd.DataFrame):
                if len(out.columns) == 1:
                    return out.iloc[:, 0]
                bad_shape = True
            if not from_df and isinstance(out, pd.Series):
                if len(out) == 1:
                    return out.iloc[0]
                bad_shape = True
            if bad_shape:
                if wrapper.grouper.is_grouped():
                    raise TypeError("Could not select one group: multiple groups returned")
                else:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is None:
            if wrapper.get_ndim() == 2 and wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if wrapper.grouper.is_grouped():
                if wrapper.grouped_ndim == 1:
                    raise TypeError("This object already contains one group of data")
                if obj_ungrouped:
                    mask = wrapper.grouper.group_by == column
                    if not mask.any():
                        raise KeyError(f"Group '{column}' not found")
                    if isinstance(obj, pd.DataFrame):
                        return obj.loc[:, mask]
                    return obj.loc[mask]
                else:
                    if column not in wrapper.get_columns():
                        if isinstance(column, int):
                            if isinstance(obj, pd.DataFrame):
                                return _check_out_dim(obj.iloc[:, column], True)
                            return _check_out_dim(obj.iloc[column], False)
                        raise KeyError(f"Group '{column}' not found")
            else:
                if wrapper.ndim == 1:
                    raise TypeError("This object already contains one column of data")
                if column not in wrapper.columns:
                    if isinstance(column, int):
                        if isinstance(obj, pd.DataFrame):
                            return _check_out_dim(obj.iloc[:, column], True)
                        return _check_out_dim(obj.iloc[column], False)
                    raise KeyError(f"Column '{column}' not found")
            if isinstance(obj, pd.DataFrame):
                return _check_out_dim(obj[column], True)
            return _check_out_dim(obj[column], False)
        if not wrapper.grouper.is_grouped():
            if wrapper.ndim == 1:
                return obj
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if wrapper.grouped_ndim == 1:
            return obj
        raise TypeError("Only one group is allowed. Use indexing or column argument.")

    def split(
        self,
        splitter: tp.Union[str, SplitterT, tp.Callable],
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        splitter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **take_kwargs,
    ) -> tp.Any:
        """Split using `vectorbtpro.generic.splitting.base.Splitter`."""
        from vectorbtpro.generic.splitting.base import Splitter

        if splitter_cls is None:
            splitter_cls = Splitter
        if not isinstance(splitter, splitter_cls):
            if isinstance(splitter, str):
                splitter = getattr(splitter_cls, splitter)
            splitter = splitter(self.wrapper.index, template_context=template_context, **splitter_kwargs)
        return splitter.take(self, template_context=template_context, **take_kwargs)
