# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions for reshaping arrays.

Reshape functions transform a Pandas object/NumPy array in some way."""

import attr
import functools
import itertools

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base import indexes, wrapping, indexing
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import resolve_dict, merge_dicts
from vectorbtpro.utils.params import combine_params, Param
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate

__all__ = [
    "to_1d_shape",
    "to_2d_shape",
    "repeat_shape",
    "tile_shape",
    "to_1d_array",
    "to_2d_array",
    "to_2d_pr_array",
    "to_2d_pc_array",
    "to_1d_array_nb",
    "to_2d_array_nb",
    "to_2d_pr_array_nb",
    "to_2d_pc_array_nb",
    "broadcast_shapes",
    "broadcast_array_to",
    "broadcast_arrays",
    "repeat",
    "tile",
    "align_pd_arrays",
    "BCO",
    "Default",
    "Ref",
    "broadcast",
    "broadcast_to",
]


def to_tuple_shape(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a tuple."""
    if checks.is_int(shape):
        return (int(shape),)
    return tuple(shape)


def to_1d_shape(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a 1-dim shape."""
    shape = to_tuple_shape(shape)
    if len(shape) == 0:
        return (1,)
    if len(shape) == 1:
        return shape
    if len(shape) == 2 and shape[1] == 1:
        return (shape[0],)
    raise ValueError(f"Cannot reshape a {len(shape)}-dimensional shape to 1 dimension")


def to_2d_shape(shape: tp.ShapeLike, expand_axis: int = 1) -> tp.Shape:
    """Convert a shape-like object to a 2-dim shape."""
    shape = to_tuple_shape(shape)
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        if expand_axis == 1:
            return shape[0], 1
        else:
            return shape[0], 0
    if len(shape) == 2:
        return shape
    raise ValueError(f"Cannot reshape a {len(shape)}-dimensional shape to 2 dimensions")


def repeat_shape(shape: tp.ShapeLike, n: int, axis: int = 1) -> tp.Shape:
    """Repeat shape `n` times along the specified axis."""
    shape = to_tuple_shape(shape)
    if len(shape) <= axis:
        shape = tuple([shape[i] if i < len(shape) else 1 for i in range(axis + 1)])
    return *shape[:axis], shape[axis] * n, *shape[axis + 1 :]


def tile_shape(shape: tp.ShapeLike, n: int, axis: int = 1) -> tp.Shape:
    """Tile shape `n` times along the specified axis.

    Identical to `repeat_shape`. Exists purely for naming consistency."""
    return repeat_shape(shape, n, axis=axis)


def index_to_series(arg: tp.Index) -> tp.Series:
    """Convert Index to Series."""
    return arg.to_series()


def mapping_to_series(arg: tp.MappingLike) -> tp.Series:
    """Convert a mapping-like object to Series."""
    if checks.is_namedtuple(arg):
        arg = arg._asdict()
    return pd.Series(arg)


def to_any_array(arg: tp.ArrayLike, raw: bool = False, convert_index: bool = True) -> tp.AnyArray:
    """Convert any array-like object to an array.

    Pandas objects are kept as-is unless `raw` is True."""
    if not raw:
        if checks.is_any_array(arg):
            if convert_index and checks.is_index(arg):
                return index_to_series(arg)
            return arg
        if checks.is_mapping_like(arg):
            return mapping_to_series(arg)
    return np.asarray(arg)


def to_pd_array(arg: tp.ArrayLike, convert_index: bool = True) -> tp.PandasArray:
    """Convert any array-like object to a Pandas object."""
    if checks.is_pandas(arg):
        if convert_index and checks.is_index(arg):
            return index_to_series(arg)
        return arg
    if checks.is_mapping_like(arg):
        return mapping_to_series(arg)

    arg = np.asarray(arg)
    if arg.ndim == 0:
        arg = arg[None]
    if arg.ndim == 1:
        return pd.Series(arg)
    if arg.ndim == 2:
        return pd.DataFrame(arg)
    raise ValueError("Wrong number of dimensions: cannot convert to Series or DataFrame")


def soft_to_ndim(arg: tp.ArrayLike, ndim: int, raw: bool = False) -> tp.AnyArray:
    """Try to softly bring `arg` to the specified number of dimensions `ndim` (max 2)."""
    arg = to_any_array(arg, raw=raw)
    if ndim == 1:
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if checks.is_frame(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]  # downgrade
    if ndim == 2:
        if arg.ndim == 1:
            if checks.is_series(arg):
                return arg.to_frame()
            return arg[:, None]  # upgrade
    return arg  # do nothing


def to_1d(arg: tp.ArrayLike, raw: bool = False) -> tp.AnyArray1d:
    """Reshape argument to one dimension.

    If `raw` is True, returns NumPy array.
    If 2-dim, will collapse along axis 1 (i.e., DataFrame with one column to Series)."""
    arg = to_any_array(arg, raw=raw)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if checks.is_frame(arg):
                return arg.iloc[:, 0]
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


to_1d_array = functools.partial(to_1d, raw=True)
"""`to_1d` with `raw` enabled."""


def to_2d(arg: tp.ArrayLike, raw: bool = False, expand_axis: int = 1) -> tp.AnyArray2d:
    """Reshape argument to two dimensions.

    If `raw` is True, returns NumPy array.
    If 1-dim, will expand along axis 1 (i.e., Series to DataFrame with one column)."""
    arg = to_any_array(arg, raw=raw)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if checks.is_series(arg):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


to_2d_array = functools.partial(to_2d, raw=True)
"""`to_2d` with `raw` enabled."""

to_2d_pr_array = functools.partial(to_2d_array, expand_axis=1)
"""`to_2d_array` with `expand_axis=1`."""

to_2d_pc_array = functools.partial(to_2d_array, expand_axis=0)
"""`to_2d_array` with `expand_axis=0`."""


@register_jitted(cache=True)
def to_1d_array_nb(arg: tp.Array) -> tp.Array1d:
    """Resize array to one dimension."""
    if arg.ndim == 0:
        return np.expand_dims(arg, axis=0)
    if arg.ndim == 1:
        return arg
    if arg.ndim == 2 and arg.shape[1] == 1:
        return arg[:, 0]
    raise ValueError("Array cannot be resized to one dimension")


@register_jitted(cache=True)
def to_2d_array_nb(arg: tp.Array, expand_axis: int = 1) -> tp.Array2d:
    """Resize array to two dimensions."""
    if arg.ndim == 0:
        return np.expand_dims(np.expand_dims(arg, axis=0), axis=0)
    if arg.ndim == 1:
        return np.expand_dims(arg, axis=expand_axis)
    if arg.ndim == 2:
        return arg
    raise ValueError("Array cannot be resized to two dimensions")


@register_jitted(cache=True)
def to_2d_pr_array_nb(arg: tp.Array) -> tp.Array2d:
    """`to_2d_array_nb` with `expand_axis=1`."""
    return to_2d_array_nb(arg, expand_axis=1)


@register_jitted(cache=True)
def to_2d_pc_array_nb(arg: tp.Array) -> tp.Array2d:
    """`to_2d_array_nb` with `expand_axis=0`."""
    return to_2d_array_nb(arg, expand_axis=0)


def to_dict(arg: tp.ArrayLike, orient: str = "dict") -> dict:
    """Convert object to dict."""
    arg = to_pd_array(arg)
    if orient == "index_series":
        return {arg.index[i]: arg.iloc[i] for i in range(len(arg.index))}
    return arg.to_dict(orient)


def repeat(
    arg: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Repeat `arg` `n` times along the specified axis."""
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        if checks.is_pandas(arg):
            new_index = indexes.repeat_index(arg.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.repeat(arg.values, n, axis=0), index=new_index)
        return np.repeat(arg, n, axis=0)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            new_columns = indexes.repeat_index(arg.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.repeat(arg.values, n, axis=1), columns=new_columns)
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")


def tile(
    arg: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Tile `arg` `n` times along the specified axis."""
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        if arg.ndim == 2:
            if checks.is_pandas(arg):
                new_index = indexes.tile_index(arg.index, n, ignore_ranges=ignore_ranges)
                return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, (n, 1)), index=new_index)
            return np.tile(arg, (n, 1))
        if checks.is_pandas(arg):
            new_index = indexes.tile_index(arg.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, n), index=new_index)
        return np.tile(arg, n)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            new_columns = indexes.tile_index(arg.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, (1, n)), columns=new_columns)
        return np.tile(arg, (1, n))
    else:
        raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")


def column_stack(*arrs: tp.MaybeSequence[tp.AnyArray]) -> tp.Array2d:
    """Stack arrays along columns."""
    if len(arrs) == 1:
        arrs = arrs[0]
    arrs = list(arrs)

    first_arr = arrs[0]
    if not hasattr(first_arr, "ndim"):
        first_arr = np.asarray(first_arr)
    if first_arr.ndim == 1 or (first_arr.ndim == 2 and first_arr.shape[1] == 1):
        return np.concatenate(arrs).reshape((len(arrs), len(first_arr))).T
    return np.column_stack(arrs)


def broadcast_shapes(
    *shapes: tp.ArrayLike,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    expand_axis: tp.Optional[tp.MaybeSequence[int]] = None,
) -> tp.Tuple[tp.Shape, ...]:
    """Broadcast shape-like objects using vectorbt's broadcasting rules."""
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if expand_axis is None:
        expand_axis = broadcasting_cfg["expand_axis"]

    is_2d = False
    for i, shape in enumerate(shapes):
        shape = to_tuple_shape(shape)
        if len(shape) == 2:
            is_2d = True
            break

    new_shapes = []
    for i, shape in enumerate(shapes):
        shape = to_tuple_shape(shape)
        if is_2d:
            if checks.is_sequence(expand_axis):
                _expand_axis = expand_axis[i]
            else:
                _expand_axis = expand_axis
            new_shape = to_2d_shape(shape, expand_axis=_expand_axis)
        else:
            new_shape = to_1d_shape(shape)
        if axis is not None:
            if checks.is_sequence(axis):
                _axis = axis[i]
            else:
                _axis = axis
            if _axis is not None:
                if _axis == 0:
                    if is_2d:
                        new_shape = (new_shape[0], 1)
                    else:
                        new_shape = (new_shape[0],)
                elif _axis == 1:
                    if is_2d:
                        new_shape = (1, new_shape[1])
                    else:
                        new_shape = (1,)
                else:
                    raise ValueError(f"Only axes 0 and 1 are supported, not {_axis}")
        new_shapes.append(new_shape)
    return tuple(np.broadcast_shapes(*new_shapes))


def broadcast_array_to(
    arr: tp.ArrayLike,
    target_shape: tp.ShapeLike,
    axis: tp.Optional[int] = None,
    expand_axis: tp.Optional[int] = None,
) -> tp.Array:
    """Broadcast an array-like object to a target shape using vectorbt's broadcasting rules."""
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if expand_axis is None:
        expand_axis = broadcasting_cfg["expand_axis"]

    arr = np.asarray(arr)
    target_shape = to_tuple_shape(target_shape)
    if len(target_shape) not in (1, 2):
        raise ValueError(f"Target shape must have either 1 or 2 dimensions, not {len(target_shape)}")
    if len(target_shape) == 2:
        new_arr = to_2d_array(arr, expand_axis=expand_axis)
    else:
        new_arr = to_1d_array(arr)
    if axis is not None:
        if axis == 0:
            if len(target_shape) == 2:
                target_shape = (target_shape[0], new_arr.shape[1])
            else:
                target_shape = (target_shape[0],)
        elif axis == 1:
            target_shape = (new_arr.shape[0], target_shape[1])
        else:
            raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")
    return np.broadcast_to(new_arr, target_shape)


def broadcast_arrays(
    *arrs: tp.ArrayLike,
    target_shape: tp.Optional[tp.ShapeLike] = None,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    expand_axis: tp.Optional[tp.MaybeSequence[int]] = None,
) -> tp.Tuple[tp.Array, ...]:
    """Broadcast array-like objects using vectorbt's broadcasting rules.

    Optionally to a target shape."""
    if target_shape is None:
        shapes = []
        for arr in arrs:
            shapes.append(np.asarray(arr).shape)
        target_shape = broadcast_shapes(*shapes, axis=axis, expand_axis=expand_axis)
    new_arrs = []
    for i, arr in enumerate(arrs):
        if axis is not None:
            if checks.is_sequence(axis):
                _axis = axis[i]
            else:
                _axis = axis
        else:
            _axis = None
        if expand_axis is not None:
            if checks.is_sequence(expand_axis):
                _expand_axis = expand_axis[i]
            else:
                _expand_axis = expand_axis
        else:
            _expand_axis = None
        new_arr = broadcast_array_to(arr, target_shape, axis=_axis, expand_axis=_expand_axis)
        new_arrs.append(new_arr)
    return tuple(new_arrs)


IndexFromLike = tp.Union[None, str, int, tp.Any]
"""Any object that can be coerced into a `index_from` argument."""


def broadcast_index(
    args: tp.Sequence[tp.AnyArray],
    to_shape: tp.Shape,
    index_from: IndexFromLike = None,
    axis: int = 0,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    **index_stack_kwargs,
) -> tp.Optional[tp.Index]:
    """Produce a broadcast index/columns.

    Args:
        args (iterable of array_like): Array-like objects.
        to_shape (tuple of int): Target shape.
        index_from (any): Broadcasting rule for this index/these columns.

            Accepts the following values:

            * 'keep' or None - keep the original index/columns of the objects in `args`
            * 'stack' - stack different indexes/columns using `vectorbtpro.base.indexes.stack_indexes`
            * 'strict' - ensure that all Pandas objects have the same index/columns
            * 'reset' - reset any index/columns (they become a simple range)
            * integer - use the index/columns of the i-th object in `args`
            * everything else will be converted to `pd.Index`
        axis (int): Set to 0 for index and 1 for columns.
        ignore_sr_names (bool): Whether to ignore Series names if they are in conflict.

            Conflicting Series names are those that are different but not None.
        ignore_ranges (bool): Whether to ignore indexes of type `pd.RangeIndex`.
        check_index_names (bool): See `vectorbtpro.utils.checks.is_index_equal`.
        **index_stack_kwargs: Keyword arguments passed to `vectorbtpro.base.indexes.stack_indexes`.

    For defaults, see `vectorbtpro._settings.broadcasting`.

    !!! note
        Series names are treated as columns with a single element but without a name.
        If a column level without a name loses its meaning, better to convert Series to DataFrames
        with one column prior to broadcasting. If the name of a Series is not that important,
        better to drop it altogether by setting it to None.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_sr_names is None:
        ignore_sr_names = broadcasting_cfg["ignore_sr_names"]
    if check_index_names is None:
        check_index_names = broadcasting_cfg["check_index_names"]

    index_str = "columns" if axis == 1 else "index"
    to_shape_2d = (to_shape[0], 1) if len(to_shape) == 1 else to_shape
    maxlen = to_shape_2d[1] if axis == 1 else to_shape_2d[0]
    new_index = None
    args = list(args)

    if index_from is None or (isinstance(index_from, str) and index_from.lower() == "keep"):
        return None
    if isinstance(index_from, int):
        if not checks.is_pandas(args[index_from]):
            raise TypeError(f"Argument under index {index_from} must be a pandas object")
        new_index = indexes.get_index(args[index_from], axis)
    elif isinstance(index_from, str):
        if index_from.lower() == "reset":
            new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
        elif index_from.lower() in ("stack", "strict"):
            last_index = None
            index_conflict = False
            for arg in args:
                if checks.is_pandas(arg):
                    index = indexes.get_index(arg, axis)
                    if last_index is not None:
                        if not checks.is_index_equal(index, last_index, check_names=check_index_names):
                            index_conflict = True
                    last_index = index
                    continue
            if not index_conflict:
                new_index = last_index
            else:
                for arg in args:
                    if checks.is_pandas(arg):
                        index = indexes.get_index(arg, axis)
                        if axis == 1 and checks.is_series(arg) and ignore_sr_names:
                            continue
                        if checks.is_default_index(index):
                            continue
                        if new_index is None:
                            new_index = index
                        else:
                            if checks.is_index_equal(index, new_index, check_names=check_index_names):
                                continue
                            if index_from.lower() == "strict":
                                raise ValueError(
                                    f"Arrays have different index. Broadcasting {index_str} "
                                    f"is not allowed when {index_str}_from=strict"
                                )
                            if len(index) != len(new_index):
                                if len(index) > 1 and len(new_index) > 1:
                                    raise ValueError("Indexes could not be broadcast together")
                                if len(index) > len(new_index):
                                    new_index = indexes.repeat_index(new_index, len(index), ignore_ranges=ignore_ranges)
                                elif len(index) < len(new_index):
                                    index = indexes.repeat_index(index, len(new_index), ignore_ranges=ignore_ranges)
                            new_index = indexes.stack_indexes([new_index, index], **index_stack_kwargs)
        else:
            raise ValueError(f"Invalid value '{index_from}' for {'columns' if axis == 1 else 'index'}_from")
    else:
        if not isinstance(index_from, pd.Index):
            index_from = pd.Index(index_from)
        new_index = index_from
    if new_index is not None:
        if maxlen > len(new_index):
            if isinstance(index_from, str) and index_from.lower() == "strict":
                raise ValueError(f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
            if maxlen > 1 and len(new_index) > 1:
                raise ValueError("Indexes could not be broadcast together")
            new_index = indexes.repeat_index(new_index, maxlen, ignore_ranges=ignore_ranges)
    else:
        new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
    return new_index


def wrap_broadcasted(
    new_obj: tp.Array,
    old_obj: tp.Optional[tp.AnyArray] = None,
    axis: tp.Optional[int] = None,
    is_pd: bool = False,
    new_index: tp.Optional[tp.Index] = None,
    new_columns: tp.Optional[tp.Index] = None,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """If the newly brodcasted array was originally a Pandas object, make it Pandas object again
    and assign it the newly broadcast index/columns."""
    if is_pd:
        if axis == 0:
            new_columns = None
        elif axis == 1:
            new_index = None
        if old_obj is not None and checks.is_pandas(old_obj):
            if new_index is None:
                old_index = indexes.get_index(old_obj, 0)
                if old_obj.shape[0] == new_obj.shape[0]:
                    new_index = old_index
                else:
                    new_index = indexes.repeat_index(old_index, new_obj.shape[0], ignore_ranges=ignore_ranges)
            if new_columns is None:
                old_columns = indexes.get_index(old_obj, 1)
                new_ncols = new_obj.shape[1] if new_obj.ndim == 2 else 1
                if len(old_columns) == new_ncols:
                    new_columns = old_columns
                else:
                    new_columns = indexes.repeat_index(old_columns, new_ncols, ignore_ranges=ignore_ranges)
        if new_obj.ndim == 2:
            return pd.DataFrame(new_obj, index=new_index, columns=new_columns)
        if new_columns is not None and len(new_columns) == 1:
            name = new_columns[0]
            if name == 0:
                name = None
        else:
            name = None
        return pd.Series(new_obj, index=new_index, name=name)
    return new_obj


def align_pd_arrays(
    *args: tp.AnyArray,
    align_index: bool = True,
    align_columns: bool = True,
    to_index: tp.Optional[tp.Index] = None,
    to_columns: tp.Optional[tp.Index] = None,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    reindex_kwargs: tp.KwargsLikeSequence = None,
) -> tp.MaybeTuple[tp.ArrayLike]:
    """Align Pandas arrays against common index and/or column levels using reindexing
    and `vectorbtpro.base.indexes.align_indexes` respectively."""
    args = list(args)
    if align_index:
        indexes_to_align = []
        for i in range(len(args)):
            if axis is not None:
                if checks.is_sequence(axis):
                    _axis = axis[i]
                else:
                    _axis = axis
            else:
                _axis = None
            if _axis in (None, 0):
                if checks.is_pandas(args[i]):
                    if not checks.is_default_index(args[i].index):
                        indexes_to_align.append(i)
        if (len(indexes_to_align) > 0 and to_index is not None) or len(indexes_to_align) > 1:
            if to_index is None:
                new_index = None
                index_changed = False
                for i in indexes_to_align:
                    arg_index = args[i].index
                    if new_index is None:
                        new_index = arg_index
                    else:
                        if not checks.is_index_equal(new_index, arg_index):
                            if new_index.dtype != arg_index.dtype:
                                raise ValueError("Indexes to be aligned must have the same data type")
                            new_index = new_index.union(arg_index)
                            index_changed = True
            else:
                new_index = to_index
                index_changed = True
            if index_changed:
                for i in indexes_to_align:
                    if to_index is None or not checks.is_index_equal(args[i].index, to_index):
                        if args[i].index.has_duplicates:
                            raise ValueError(f"Index at position {i} contains duplicates")
                        if not args[i].index.is_monotonic_increasing:
                            raise ValueError(f"Index at position {i} is not monotonically increasing")
                        _reindex_kwargs = resolve_dict(reindex_kwargs, i=i)
                        was_bool = (isinstance(args[i], pd.Series) and args[i].dtype == "bool") or (
                            isinstance(args[i], pd.DataFrame) and (args[i].dtypes == "bool").all()
                        )
                        args[i] = args[i].reindex(new_index, **_reindex_kwargs)
                        is_object = (isinstance(args[i], pd.Series) and args[i].dtype == "object") or (
                            isinstance(args[i], pd.DataFrame) and (args[i].dtypes == "object").all()
                        )
                        if was_bool and is_object:
                            args[i] = args[i].astype(None)
    if align_columns:
        columns_to_align = []
        for i in range(len(args)):
            if axis is not None:
                if checks.is_sequence(axis):
                    _axis = axis[i]
                else:
                    _axis = axis
            else:
                _axis = None
            if _axis in (None, 1):
                if checks.is_frame(args[i]) and len(args[i].columns) > 1:
                    if not checks.is_default_index(args[i].columns):
                        columns_to_align.append(i)
        if (len(columns_to_align) > 0 and to_columns is not None) or len(columns_to_align) > 1:
            indexes_ = [args[i].columns for i in columns_to_align]
            if to_columns is not None:
                indexes_.append(to_columns)
            if len(set(map(len, indexes_))) > 1:
                col_indices = indexes.align_indexes(*indexes_)
                for i in columns_to_align:
                    args[i] = args[i].iloc[:, col_indices[columns_to_align.index(i)]]
    if len(args) == 1:
        return args[0]
    return tuple(args)


@attr.s(frozen=True)
class BCO:
    """Class that represents an object passed to `broadcast`.

    If any value is None, mostly defaults to the global value passed to `broadcast`."""

    value: tp.Any = attr.ib()
    """Value of the object."""

    axis: tp.Optional[int] = attr.ib(default=None)
    """Axis to broadcast.
    
    Set to None to broadcast all axes."""

    to_pd: tp.Optional[bool] = attr.ib(default=None)
    """Whether to convert the output array to a Pandas object."""

    keep_flex: tp.Optional[bool] = attr.ib(default=None)
    """Whether to keep the raw version of the output for flexible indexing.
    
    Only makes sure that the array can broadcast to the target shape."""

    min_ndim: tp.Optional[int] = attr.ib(default=None)
    """Minimum number of dimensions."""

    expand_axis: tp.Optional[int] = attr.ib(default=None)
    """Axis to expand if the array is 1-dim but the target shape is 2-dim."""

    post_func: tp.Optional[tp.Callable] = attr.ib(default=None)
    """Function to post-process the output array."""

    require_kwargs: tp.Optional[tp.Kwargs] = attr.ib(default=None)
    """Keyword arguments passed to `np.require`."""

    reindex_kwargs: tp.Optional[tp.Kwargs] = attr.ib(default=None)
    """Keyword arguments passed to `pd.DataFrame.reindex`."""

    merge_kwargs: tp.Optional[int] = attr.ib(default=None)
    """Keyword arguments passed to `vectorbtpro.base.merging.column_stack_merge`."""


@attr.s(frozen=True)
class Default:
    """Class for wrapping default values."""

    value: tp.Any = attr.ib()
    """Default value."""


@attr.s(frozen=True)
class Ref:
    """Class for wrapping references to other values."""

    key: tp.Hashable = attr.ib()
    """Reference to another key."""


def resolve_ref(dct: dict, k: tp.Hashable, inside_bco: bool = False, keep_wrap_default: bool = False) -> tp.Any:
    """Resolve a potential reference."""
    v = dct[k]
    is_default = False
    if isinstance(v, Default):
        v = v.value
        is_default = True
    if isinstance(v, Ref):
        new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
        if keep_wrap_default and is_default:
            return Default(new_v)
        return new_v
    if isinstance(v, BCO) and inside_bco:
        v = v.value
        is_default = False
        if isinstance(v, Default):
            v = v.value
            is_default = True
        if isinstance(v, Ref):
            new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
            if keep_wrap_default and is_default:
                return Default(new_v)
            return new_v
    return v


def broadcast(
    *args,
    to_shape: tp.Optional[tp.ShapeLike] = None,
    align_index: tp.Optional[bool] = None,
    align_columns: tp.Optional[bool] = None,
    index_from: tp.Optional[IndexFromLike] = None,
    columns_from: tp.Optional[IndexFromLike] = None,
    to_frame: tp.Optional[bool] = None,
    axis: tp.Optional[tp.MaybeMappingSequence[int]] = None,
    to_pd: tp.Optional[tp.MaybeMappingSequence[bool]] = None,
    keep_flex: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    min_ndim: tp.MaybeMappingSequence[tp.Optional[int]] = None,
    expand_axis: tp.MaybeMappingSequence[tp.Optional[int]] = None,
    post_func: tp.MaybeMappingSequence[tp.Optional[tp.Callable]] = None,
    require_kwargs: tp.MaybeMappingSequence[tp.Optional[tp.Kwargs]] = None,
    reindex_kwargs: tp.MaybeMappingSequence[tp.Optional[tp.Kwargs]] = None,
    merge_kwargs: tp.MaybeMappingSequence[tp.Optional[tp.Kwargs]] = None,
    tile: tp.Union[None, int, tp.IndexLike] = None,
    random_subset: tp.Optional[int] = None,
    seed: tp.Optional[int] = None,
    keep_wrap_default: tp.Optional[bool] = None,
    return_wrapper: bool = False,
    wrapper_kwargs: tp.KwargsLike = None,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    **index_stack_kwargs,
) -> tp.Any:
    """Bring any array-like object in `args` to the same shape by using NumPy-like broadcasting.

    See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

    !!! important
        The major difference to NumPy is that one-dimensional arrays will always broadcast against the row axis!

    Can broadcast Pandas objects by broadcasting their index/columns with `broadcast_index`.

    Args:
        *args: Objects to broadcast.

            If the first and only argument is a mapping, will return a dict.

            Allows using `BCO`, `Ref`, `Default`, `vectorbtpro.utils.params.Param`,
            `vectorbtpro.base.indexing.index_dict`, `vectorbtpro.base.indexing.IdxSetter`,
            `vectorbtpro.base.indexing.IdxSetterFactory`, and templates.
            If an index dictionary, fills using `vectorbtpro.base.wrapping.ArrayWrapper.fill_and_set`.
        to_shape (tuple of int): Target shape. If set, will broadcast every object in `args` to `to_shape`.
        align_index (bool): Whether to align index of Pandas objects using union.

            Pass None to use the default.
        align_columns (bool): Whether to align columns of Pandas objects using multi-index.

            Pass None to use the default.
        index_from (any): Broadcasting rule for index.

            Pass None to use the default.
        columns_from (any): Broadcasting rule for columns.

            Pass None to use the default.
        to_frame (bool): Whether to convert all Series to DataFrames.
        axis (int, sequence or mapping): See `BCO.axis`.
        to_pd (bool, sequence or mapping): See `BCO.to_pd`.

            If None, converts only if there is at least one Pandas object among them.
        keep_flex (bool, sequence or mapping): See `BCO.keep_flex`.
        min_ndim (int, sequence or mapping): See `BCO.min_ndim`.

            If None, becomes 2 if `keep_flex` is True, otherwise 1.
        expand_axis (int, sequence or mapping): See `BCO.expand_axis`.
        post_func (callable, sequence or mapping): See `BCO.post_func`.

            Applied only when `keep_flex` is False.
        require_kwargs (dict, sequence or mapping): See `BCO.require_kwargs`.

            This key will be merged with any argument-specific dict. If the mapping contains all keys in
            `np.require`, it will be applied on all objects.
        reindex_kwargs (dict, sequence or mapping): See `BCO.reindex_kwargs`.

            This key will be merged with any argument-specific dict. If the mapping contains all keys in
            `pd.DataFrame.reindex`, it will be applied on all objects.
        merge_kwargs (dict, sequence or mapping): See `BCO.merge_kwargs`.

            This key will be merged with any argument-specific dict. If the mapping contains all keys in
            `pd.DataFrame.reindex`, it will be applied on all objects.
        tile (int or index_like): Tile the final object by the number of times or index.
        random_subset (int): Select a random subset of parameter values.

            Seed can be set using NumPy before calling this function.
        seed (int): Set seed to make output deterministic.
        keep_wrap_default (bool): Whether to keep wrapping with `vectorbtpro.base.reshaping.Default`.
        return_wrapper (bool): Whether to also return the wrapper associated with the operation.
        wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
        ignore_sr_names (bool): See `broadcast_index`.
        ignore_ranges (bool): See `broadcast_index`.
        check_index_names (bool): See `broadcast_index`.
        **index_stack_kwargs: Keyword arguments passed to `vectorbtpro.base.indexes.stack_indexes`.

    For defaults, see `vectorbtpro._settings.broadcasting`.

    Any keyword argument that can be associated with an object can be passed as

    * a const that is applied on all objects,
    * a sequence with value per object, and
    * a mapping with value per object name and the special key `_def` denoting the default value.

    Additionally, any object can be passed wrapped with `BCO`, which attributes will override
    any of the above arguments if not None.

    Usage:
        * Without broadcasting index and columns:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> import vectorbtpro as vbt

        >>> v = 0
        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([1, 2, 3], index=pd.Index(['x', 'y', 'z']), name='a')
        >>> df = pd.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     index=pd.Index(['x2', 'y2', 'z2']),
        ...     columns=pd.Index(['a2', 'b2', 'c2']),
        ... )

        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='keep',
        ...     columns_from='keep',
        ...     align_index=False
        ... ): print(i)
           0  1  2
        0  0  0  0
        1  0  0  0
        2  0  0  0
           0  1  2
        0  1  2  3
        1  1  2  3
        2  1  2  3
           a  a  a
        x  1  1  1
        y  2  2  2
        z  3  3  3
            a2  b2  c2
        x2   1   2   3
        y2   4   5   6
        z2   7   8   9
        ```

        * Take index and columns from the argument at specific position:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=2,
        ...     columns_from=3,
        ...     align_index=False
        ... ): print(i)
           a2  b2  c2
        x   0   0   0
        y   0   0   0
        z   0   0   0
           a2  b2  c2
        x   1   2   3
        y   1   2   3
        z   1   2   3
           a2  b2  c2
        x   1   1   1
        y   2   2   2
        z   3   3   3
           a2  b2  c2
        x   1   2   3
        y   4   5   6
        z   7   8   9
        ```

        * Broadcast index and columns through stacking:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='stack',
        ...     columns_from='stack',
        ...     align_index=False
        ... ): print(i)
              a2  b2  c2
        x x2   0   0   0
        y y2   0   0   0
        z z2   0   0   0
              a2  b2  c2
        x x2   1   2   3
        y y2   1   2   3
        z z2   1   2   3
              a2  b2  c2
        x x2   1   1   1
        y y2   2   2   2
        z z2   3   3   3
              a2  b2  c2
        x x2   1   2   3
        y y2   4   5   6
        z z2   7   8   9
        ```

        * Set index and columns manually:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=['a', 'b', 'c'],
        ...     columns_from=['d', 'e', 'f'],
        ...     align_index=False
        ... ): print(i)
           d  e  f
        a  0  0  0
        b  0  0  0
        c  0  0  0
           d  e  f
        a  1  2  3
        b  1  2  3
        c  1  2  3
           d  e  f
        a  1  1  1
        b  2  2  2
        c  3  3  3
           d  e  f
        a  1  2  3
        b  4  5  6
        c  7  8  9
        ```

        * Pass arguments as a mapping returns a mapping:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack',
        ...     align_index=False
        ... )
        {'v':       a2  b2  c2
              x x2   0   0   0
              y y2   0   0   0
              z z2   0   0   0,
         'a':       a2  b2  c2
              x x2   1   2   3
              y y2   1   2   3
              z z2   1   2   3,
         'sr':       a2  b2  c2
               x x2   1   1   1
               y y2   2   2   2
               z z2   3   3   3,
         'df':       a2  b2  c2
               x x2   1   2   3
               y y2   4   5   6
               z z2   7   8   9}
        ```

        * Keep all results in a format suitable for flexible indexing apart from one:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack',
        ...     keep_flex=dict(_def=True, df=False),
        ...     require_kwargs=dict(df=dict(dtype=float)),
        ...     align_index=False
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        * Specify arguments per object using `BCO`:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco),
        ...     index_from='stack',
        ...     keep_flex=True,
        ...     align_index=False
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        * Introduce a parameter that should build a Cartesian product of its values and other objects:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> p_bco = vbt.BCO(pd.Param([1, 2, 3], name='my_p'))
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco, p=p_bco),
        ...     index_from='stack',
        ...     keep_flex=True,
        ...     align_index=False
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
         'sr': array([[1],
                [2],
                [3]]),
         'df': my_p        1              2              3
                a2   b2   c2   a2   b2   c2   a2   b2   c2
         x x2  1.0  2.0  3.0  1.0  2.0  3.0  1.0  2.0  3.0
         y y2  4.0  5.0  6.0  4.0  5.0  6.0  4.0  5.0  6.0
         z z2  7.0  8.0  9.0  7.0  8.0  9.0  7.0  8.0  9.0,
         'p': array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3]])}
        ```

        * Build a Cartesian product of all parameters:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3]),
        ...         b=vbt.Param(['x', 'y']),
        ...         c=vbt.Param([False, True])
        ...     )
        ... )
        {'a': array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]),
         'b': array([['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'y']], dtype='<U1'),
         'c': array([[False, True, False, True, False, True, False, True, False, True, False, True]])}
        ```

        * Build a Cartesian product of two groups of parameters - (a, d) and (b, c):

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3], level=0),
        ...         b=vbt.Param(['x', 'y'], level=1),
        ...         d=vbt.Param([100., 200., 300.], level=0),
        ...         c=vbt.Param([False, True], level=1)
        ...     )
        ... )
        {'a': array([[1, 1, 2, 2, 3, 3]]),
         'b': array([['x', 'y', 'x', 'y', 'x', 'y']], dtype='<U1'),
         'd': array([[100., 100., 200., 200., 300., 300.]]),
         'c': array([[False,  True, False,  True, False,  True]])}
        ```

        * Select a random subset of parameter combinations:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3]),
        ...         b=vbt.Param(['x', 'y']),
        ...         c=vbt.Param([False, True])
        ...     ),
        ...     random_subset=5,
        ...     seed=42
        ... )
        {'a': array([[1, 2, 3, 3, 3]]),
         'b': array([['x', 'x', 'x', 'x', 'y']], dtype='<U1'),
         'c': array([[False,  True, False,  True, False]])}
        ```
    """
    # Get defaults
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if align_index is None:
        align_index = broadcasting_cfg["align_index"]
    if align_columns is None:
        align_columns = broadcasting_cfg["align_columns"]
    if index_from is None:
        index_from = broadcasting_cfg["index_from"]
    if columns_from is None:
        columns_from = broadcasting_cfg["columns_from"]
    if keep_wrap_default is None:
        keep_wrap_default = broadcasting_cfg["keep_wrap_default"]
    require_kwargs_per_obj = True
    if require_kwargs is not None and checks.is_mapping(require_kwargs):
        require_arg_names = get_func_arg_names(np.require)
        if set(require_kwargs) <= set(require_arg_names):
            require_kwargs_per_obj = False
    reindex_kwargs_per_obj = True
    if reindex_kwargs is not None and checks.is_mapping(reindex_kwargs):
        reindex_arg_names = get_func_arg_names(pd.DataFrame.reindex)
        if set(reindex_kwargs) <= set(reindex_arg_names):
            reindex_kwargs_per_obj = False
    merge_kwargs_per_obj = True
    if merge_kwargs is not None and checks.is_mapping(merge_kwargs):
        merge_arg_names = get_func_arg_names(pd.DataFrame.merge)
        if set(merge_kwargs) <= set(merge_arg_names):
            merge_kwargs_per_obj = False
    if checks.is_mapping(args[0]) and not isinstance(args[0], indexing.index_dict):
        if len(args) > 1:
            raise ValueError("Only one argument is allowed when passing a mapping")
        all_keys = list(dict(args[0]).keys())
        objs = list(args[0].values())
        return_dict = True
    else:
        objs = list(args)
        all_keys = list(range(len(args)))
        return_dict = False

    def _resolve_arg(obj: tp.Any, arg_name: str, global_value: tp.Any, default_value: tp.Any) -> tp.Any:
        if isinstance(obj, BCO) and getattr(obj, arg_name) is not None:
            return getattr(obj, arg_name)
        if checks.is_mapping(global_value):
            return global_value.get(k, global_value.get("_def", default_value))
        if checks.is_sequence(global_value):
            return global_value[i]
        return global_value

    # Build BCO instances
    none_keys = set()
    default_keys = set()
    param_keys = set()
    special_keys = set()
    bco_instances = {}
    pool = dict(zip(all_keys, objs))
    for i, k in enumerate(all_keys):
        obj = objs[i]

        if isinstance(obj, Default):
            obj = obj.value
            default_keys.add(k)
        if isinstance(obj, Ref):
            obj = resolve_ref(pool, k)
        if isinstance(obj, BCO):
            value = obj.value
        else:
            value = obj
        if isinstance(value, Default):
            value = value.value
            default_keys.add(k)
        if isinstance(value, Ref):
            value = resolve_ref(pool, k, inside_bco=True)
        if value is None:
            none_keys.add(k)
            continue

        _axis = _resolve_arg(obj, "axis", axis, None)
        _to_pd = _resolve_arg(obj, "to_pd", to_pd, None)

        _keep_flex = _resolve_arg(obj, "keep_flex", keep_flex, None)
        if _keep_flex is None:
            _keep_flex = broadcasting_cfg["keep_flex"]

        _min_ndim = _resolve_arg(obj, "min_ndim", min_ndim, None)
        if _min_ndim is None:
            _min_ndim = broadcasting_cfg["min_ndim"]

        _expand_axis = _resolve_arg(obj, "expand_axis", expand_axis, None)
        if _expand_axis is None:
            _expand_axis = broadcasting_cfg["expand_axis"]

        _post_func = _resolve_arg(obj, "post_func", post_func, None)

        if isinstance(obj, BCO) and obj.require_kwargs is not None:
            _require_kwargs = obj.require_kwargs
        else:
            _require_kwargs = None
        if checks.is_mapping(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(
                require_kwargs.get("_def", None),
                require_kwargs.get(k, None),
                _require_kwargs,
            )
        elif checks.is_sequence(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(require_kwargs[i], _require_kwargs)
        else:
            _require_kwargs = merge_dicts(require_kwargs, _require_kwargs)

        if isinstance(obj, BCO) and obj.reindex_kwargs is not None:
            _reindex_kwargs = obj.reindex_kwargs
        else:
            _reindex_kwargs = None
        if checks.is_mapping(reindex_kwargs) and reindex_kwargs_per_obj:
            _reindex_kwargs = merge_dicts(
                reindex_kwargs.get("_def", None),
                reindex_kwargs.get(k, None),
                _reindex_kwargs,
            )
        elif checks.is_sequence(reindex_kwargs) and reindex_kwargs_per_obj:
            _reindex_kwargs = merge_dicts(reindex_kwargs[i], _reindex_kwargs)
        else:
            _reindex_kwargs = merge_dicts(reindex_kwargs, _reindex_kwargs)

        if isinstance(obj, BCO) and obj.merge_kwargs is not None:
            _merge_kwargs = obj.merge_kwargs
        else:
            _merge_kwargs = None
        if checks.is_mapping(merge_kwargs) and merge_kwargs_per_obj:
            _merge_kwargs = merge_dicts(
                merge_kwargs.get("_def", None),
                merge_kwargs.get(k, None),
                _merge_kwargs,
            )
        elif checks.is_sequence(merge_kwargs) and merge_kwargs_per_obj:
            _merge_kwargs = merge_dicts(merge_kwargs[i], _merge_kwargs)
        else:
            _merge_kwargs = merge_dicts(merge_kwargs, _merge_kwargs)

        if isinstance(value, Param):
            param_keys.add(k)
        elif isinstance(value, (indexing.index_dict, indexing.IdxSetter, indexing.IdxSetterFactory, CustomTemplate)):
            special_keys.add(k)
        else:
            value = to_any_array(value)

        bco_instances[k] = BCO(
            value,
            axis=_axis,
            to_pd=_to_pd,
            keep_flex=_keep_flex,
            min_ndim=_min_ndim,
            expand_axis=_expand_axis,
            post_func=_post_func,
            require_kwargs=_require_kwargs,
            reindex_kwargs=_reindex_kwargs,
            merge_kwargs=_merge_kwargs,
        )

    # Check whether we should broadcast Pandas metadata and work on 2-dim data
    is_pd = False
    is_2d = False

    old_objs = {}
    obj_axis = {}
    obj_reindex_kwargs = {}
    for k, bco_obj in bco_instances.items():
        if k in none_keys or k in param_keys or k in special_keys:
            continue

        obj = bco_obj.value
        if obj.ndim > 1:
            is_2d = True
        if checks.is_pandas(obj):
            is_pd = True
        if bco_obj.to_pd is not None and bco_obj.to_pd:
            is_pd = True
        old_objs[k] = obj
        obj_axis[k] = bco_obj.axis
        obj_reindex_kwargs[k] = bco_obj.reindex_kwargs

    if to_shape is not None:
        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        if len(to_shape) > 1:
            is_2d = True

    if to_frame is not None:
        is_2d = to_frame

    if to_pd is not None:
        is_pd = to_pd or (return_wrapper and is_pd)

    # Align pandas arrays
    if index_from is not None and not isinstance(index_from, (int, str, pd.Index)):
        index_from = pd.Index(index_from)
    if columns_from is not None and not isinstance(columns_from, (int, str, pd.Index)):
        columns_from = pd.Index(columns_from)
    aligned_objs = align_pd_arrays(
        *old_objs.values(),
        align_index=align_index,
        align_columns=align_columns,
        to_index=index_from if isinstance(index_from, pd.Index) else None,
        to_columns=columns_from if isinstance(columns_from, pd.Index) else None,
        axis=list(obj_axis.values()),
        reindex_kwargs=list(obj_reindex_kwargs.values()),
    )
    if not isinstance(aligned_objs, tuple):
        aligned_objs = (aligned_objs,)
    aligned_objs = dict(zip(old_objs.keys(), aligned_objs))

    # Convert to NumPy
    ready_objs = {}
    for k, obj in aligned_objs.items():
        _expand_axis = bco_instances[k].expand_axis

        new_obj = np.asarray(obj)
        if is_2d and new_obj.ndim == 1:
            if isinstance(obj, pd.Series):
                new_obj = new_obj[:, None]
            else:
                new_obj = np.expand_dims(new_obj, _expand_axis)
        ready_objs[k] = new_obj

    # Get final shape
    if to_shape is None:
        try:
            to_shape = broadcast_shapes(
                *map(lambda x: x.shape, ready_objs.values()),
                axis=list(obj_axis.values()),
            )
        except ValueError:
            arr_shapes = {}
            for i, k in enumerate(bco_instances):
                if k in none_keys or k in param_keys or k in special_keys:
                    continue

                if len(ready_objs[k].shape) > 0:
                    arr_shapes[k] = ready_objs[k].shape
            raise ValueError("Could not broadcast shapes: %s" % str(arr_shapes))
    if not isinstance(to_shape, tuple):
        to_shape = (to_shape,)
    if len(to_shape) == 0:
        to_shape = (1,)
    to_shape_2d = to_shape if len(to_shape) > 1 else (*to_shape, 1)

    if is_pd:
        # Decide on index and columns
        # NOTE: Important to pass aligned_objs, not ready_objs, to preserve original shape info
        new_index = broadcast_index(
            [v for k, v in aligned_objs.items() if obj_axis[k] in (None, 0)],
            to_shape,
            index_from=index_from,
            axis=0,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **index_stack_kwargs,
        )
        new_columns = broadcast_index(
            [v for k, v in aligned_objs.items() if obj_axis[k] in (None, 1)],
            to_shape,
            index_from=columns_from,
            axis=1,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **index_stack_kwargs,
        )
    else:
        new_index = pd.RangeIndex(stop=to_shape_2d[0])
        new_columns = pd.RangeIndex(stop=to_shape_2d[1])

    # Build a product
    param_product = None
    param_columns = None
    n_params = 0
    if len(param_keys) > 0:
        # Combine parameters
        param_dct = {}
        for k, bco_obj in bco_instances.items():
            if k not in param_keys:
                continue
            param_dct[k] = bco_obj.value
        param_product, param_columns = combine_params(
            param_dct,
            random_subset=random_subset,
            seed=seed,
            index_stack_kwargs=index_stack_kwargs,
        )
        n_params = len(param_columns)

        # Combine parameter columns with new columns
        if param_columns is not None and new_columns is not None:
            new_columns = indexes.combine_indexes([param_columns, new_columns], **index_stack_kwargs)

    # Tile
    if tile is not None:
        if isinstance(tile, int):
            if new_columns is not None:
                new_columns = indexes.tile_index(new_columns, tile)
        else:
            if new_columns is not None:
                new_columns = indexes.combine_indexes([tile, new_columns], **index_stack_kwargs)
            tile = len(tile)
        n_params = max(n_params, 1) * tile

    # Build wrapper
    if n_params == 0:
        new_shape = to_shape
    else:
        new_shape = (to_shape_2d[0], to_shape_2d[1] * n_params)
    wrapper = wrapping.ArrayWrapper.from_shape(
        new_shape,
        **merge_dicts(
            dict(
                index=new_index,
                columns=new_columns,
            ),
            wrapper_kwargs,
        ),
    )

    def _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis):
        if _min_ndim is None:
            if _keep_flex:
                _min_ndim = 2
            else:
                _min_ndim = 1
        if _min_ndim not in (1, 2):
            raise ValueError("Argument min_ndim must be either 1 or 2")
        if _min_ndim in (1, 2) and new_obj.ndim == 0:
            new_obj = new_obj[None]
        if _min_ndim == 2 and new_obj.ndim == 1:
            if len(to_shape) == 1:
                new_obj = new_obj[:, None]
            else:
                new_obj = np.expand_dims(new_obj, _expand_axis)
        return new_obj

    # Perform broadcasting
    aligned_objs2 = {}
    new_objs = {}
    for i, k in enumerate(all_keys):
        if k in none_keys or k in special_keys:
            continue
        _keep_flex = bco_instances[k].keep_flex
        _min_ndim = bco_instances[k].min_ndim
        _axis = bco_instances[k].axis
        _expand_axis = bco_instances[k].expand_axis
        _merge_kwargs = bco_instances[k].merge_kwargs

        if k in param_keys:
            # Broadcast parameters
            from vectorbtpro.base.merging import column_stack_merge

            if _axis == 0:
                raise ValueError("Parameters do not support broadcasting with axis=0")
            obj = param_product[k]
            new_obj = []
            any_needs_broadcasting = False
            all_forced_broadcast = True
            for o in obj:
                o = to_2d_array(o)
                if not _keep_flex:
                    needs_broadcasting = True
                elif o.shape[0] > 1:
                    needs_broadcasting = True
                elif o.shape[1] > 1 and o.shape[1] != to_shape_2d[1]:
                    needs_broadcasting = True
                else:
                    needs_broadcasting = False
                if needs_broadcasting:
                    any_needs_broadcasting = True
                    o = broadcast_array_to(o, to_shape_2d, axis=_axis)
                elif o.size == 1:
                    all_forced_broadcast = False
                    o = np.repeat(o, to_shape_2d[1], axis=1)
                else:
                    all_forced_broadcast = False
                new_obj.append(o)
            if any_needs_broadcasting and not all_forced_broadcast:
                new_obj2 = []
                for o in new_obj:
                    if o.shape[1] != to_shape_2d[1]:
                        o = broadcast_array_to(o, to_shape_2d, axis=_axis)
                    new_obj2.append(o)
                new_obj = new_obj2
            obj = column_stack_merge(new_obj, **_merge_kwargs)
            if tile is not None:
                obj = np.tile(obj, (1, tile))
            old_obj = obj
            new_obj = obj
        else:
            # Broadcast regular objects
            old_obj = aligned_objs[k]
            new_obj = ready_objs[k]
            if _keep_flex:
                if n_params > 0 and _axis in (None, 1):
                    if len(to_shape) == 1:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = new_obj[:, None]  # product changes is_2d behavior
                    else:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = np.tile(new_obj, n_params)
                        elif new_obj.ndim == 2 and new_obj.shape[1] > 1:
                            new_obj = np.tile(new_obj, (1, n_params))
            else:
                new_obj = broadcast_array_to(new_obj, to_shape, axis=_axis)
                if n_params > 0 and _axis in (None, 1):
                    if new_obj.ndim == 1:
                        new_obj = new_obj[:, None]  # product changes is_2d behavior
                    new_obj = np.tile(new_obj, (1, n_params))

        new_obj = _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis)
        aligned_objs2[k] = old_obj
        new_objs[k] = new_obj

    # Resolve special objects
    new_objs2 = {}
    for i, k in enumerate(all_keys):
        if k in none_keys:
            continue
        if k in special_keys:
            bco = bco_instances[k]
            if isinstance(bco.value, (indexing.index_dict, indexing.IdxSetter, indexing.IdxSetterFactory)):
                # Index dict
                _is_pd = bco.to_pd
                if _is_pd is None:
                    _is_pd = is_pd
                _keep_flex = bco.keep_flex
                _min_ndim = bco.min_ndim
                _expand_axis = bco.expand_axis
                _reindex_kwargs = resolve_dict(bco.reindex_kwargs)
                _fill_value = _reindex_kwargs.get("fill_value", np.nan)
                new_obj = wrapper.fill_and_set(
                    bco.value,
                    fill_value=_fill_value,
                    keep_flex=_keep_flex,
                )
                if not _is_pd and not _keep_flex:
                    new_obj = new_obj.values
                new_obj = _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis)
            elif isinstance(bco.value, CustomTemplate):
                # Template
                context = dict(
                    bco_instances=bco_instances,
                    new_objs=new_objs,
                    wrapper=wrapper,
                    obj_name=k,
                    bco=bco,
                )
                new_obj = bco.value.substitute(context, sub_id="broadcast")
            else:
                raise TypeError(f"Special type {type(bco.value)} is not supported")
        else:
            new_obj = new_objs[k]

        # Force to match requirements
        new_obj = np.require(new_obj, **resolve_dict(bco_instances[k].require_kwargs))
        new_objs2[k] = new_obj

    # Perform wrapping and post-processing
    new_objs3 = {}
    for i, k in enumerate(all_keys):
        if k in none_keys:
            continue
        new_obj = new_objs2[k]
        _axis = bco_instances[k].axis
        _keep_flex = bco_instances[k].keep_flex

        if not _keep_flex:
            # Wrap array
            _is_pd = bco_instances[k].to_pd
            if _is_pd is None:
                _is_pd = is_pd
            new_obj = wrap_broadcasted(
                new_obj,
                old_obj=aligned_objs2[k] if k not in special_keys else None,
                axis=_axis,
                is_pd=_is_pd,
                new_index=new_index,
                new_columns=new_columns,
                ignore_ranges=ignore_ranges,
            )

        # Post-process array
        _post_func = bco_instances[k].post_func
        if _post_func is not None:
            new_obj = _post_func(new_obj)
        new_objs3[k] = new_obj

    # Prepare outputs
    return_objs = []
    for k in all_keys:
        if k not in none_keys:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(new_objs3[k]))
            else:
                return_objs.append(new_objs3[k])
        else:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(None))
            else:
                return_objs.append(None)
    if return_dict:
        return_objs = dict(zip(all_keys, return_objs))
    else:
        return_objs = tuple(return_objs)
    if len(return_objs) > 1 or return_dict:
        if return_wrapper:
            return return_objs, wrapper
        return return_objs
    if return_wrapper:
        return return_objs[0], wrapper
    return return_objs[0]


def broadcast_to(
    arg1: tp.ArrayLike,
    arg2: tp.Union[tp.ArrayLike, tp.ShapeLike, wrapping.ArrayWrapper],
    to_pd: tp.Optional[bool] = None,
    index_from: tp.Optional[IndexFromLike] = None,
    columns_from: tp.Optional[IndexFromLike] = None,
    **kwargs,
) -> tp.Any:
    """Broadcast `arg1` to `arg2`.

    Argument `arg2` can be a shape, an instance of `vectorbtpro.base.wrapping.ArrayWrapper`,
    or any array-like object.

    Pass None to `index_from`/`columns_from` to use index/columns of the second argument.

    Keyword arguments `**kwargs` are passed to `broadcast`.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import broadcast_to

        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')

        >>> broadcast_to(a, sr)
        x    1
        y    2
        z    3
        Name: a, dtype: int64

        >>> broadcast_to(sr, a)
        array([4, 5, 6])
        ```
    """
    if checks.is_int(arg2) or isinstance(arg2, tuple):
        arg2 = to_tuple_shape(arg2)
    if isinstance(arg2, tuple):
        to_shape = arg2
    elif isinstance(arg2, wrapping.ArrayWrapper):
        to_pd = True
        if index_from is None:
            index_from = arg2.index
        if columns_from is None:
            columns_from = arg2.columns
        to_shape = arg2.shape
    else:
        arg2 = to_any_array(arg2)
        if to_pd is None:
            to_pd = checks.is_pandas(arg2)
        if to_pd:
            # Take index and columns from arg2
            if index_from is None:
                index_from = indexes.get_index(arg2, 0)
            if columns_from is None:
                columns_from = indexes.get_index(arg2, 1)
        to_shape = arg2.shape
    return broadcast(
        arg1,
        to_shape=to_shape,
        to_pd=to_pd,
        index_from=index_from,
        columns_from=columns_from,
        **kwargs,
    )


def broadcast_to_array_of(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> tp.Array:
    """Broadcast `arg1` to the shape `(1, *arg2.shape)`.

    `arg1` must be either a scalar, a 1-dim array, or have 1 dimension more than `arg2`.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.base.reshaping import broadcast_to_array_of

        >>> broadcast_to_array_of([0.1, 0.2], np.empty((2, 2)))
        [[[0.1 0.1]
          [0.1 0.1]]

         [[0.2 0.2]
          [0.2 0.2]]]
        ```
    """
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
    # From here on arg1 can be only a 1-dim array
    if arg1.ndim == 0:
        arg1 = to_1d(arg1)
    checks.assert_ndim(arg1, 1)

    if arg2.ndim == 0:
        return arg1
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
    return np.tile(arg1, (1, *arg2.shape))


def broadcast_to_axis_of(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    axis: int,
    require_kwargs: tp.KwargsLike = None,
) -> tp.Array:
    """Broadcast `arg1` to an axis of `arg2`.

    If `arg2` has less dimensions than requested, will broadcast `arg1` to a single number.

    For other keyword arguments, see `broadcast`."""
    if require_kwargs is None:
        require_kwargs = {}
    arg2 = to_any_array(arg2)
    if arg2.ndim < axis + 1:
        return broadcast_array_to(arg1, (1,))[0]  # to a single number
    arg1 = broadcast_array_to(arg1, (arg2.shape[axis],))
    arg1 = np.require(arg1, **require_kwargs)
    return arg1


def broadcast_combs(
    *args: tp.ArrayLike,
    axis: int = 1,
    comb_func: tp.Callable = itertools.product,
    **broadcast_kwargs,
) -> tp.Any:
    """Align an axis of each array using a combinatoric function and broadcast their indexes.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.base.reshaping import broadcast_combs

        >>> df = pd.DataFrame([[1, 2, 3], [3, 4, 5]], columns=pd.Index(['a', 'b', 'c'], name='df_param'))
        >>> df2 = pd.DataFrame([[6, 7], [8, 9]], columns=pd.Index(['d', 'e'], name='df2_param'))
        >>> sr = pd.Series([10, 11], name='f')

        >>> new_df, new_df2, new_sr = broadcast_combs((df, df2, sr))

        >>> new_df
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          1  1  2  2  3  3
        1          3  3  4  4  5  5

        >>> new_df2
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          6  7  6  7  6  7
        1          8  9  8  9  8  9

        >>> new_sr
        df_param    a       b       c
        df2_param   d   e   d   e   d   e
        0          10  10  10  10  10  10
        1          11  11  11  11  11  11
        ```
    """
    if broadcast_kwargs is None:
        broadcast_kwargs = {}

    args = list(args)
    if len(args) < 2:
        raise ValueError("At least two arguments are required")
    for i in range(len(args)):
        arg = to_any_array(args[i])
        if axis == 1:
            arg = to_2d(arg)
        args[i] = arg
    indices = []
    for arg in args:
        indices.append(np.arange(len(indexes.get_index(to_pd_array(arg), axis))))
    new_indices = list(map(list, zip(*list(comb_func(*indices)))))
    results = []
    for i, arg in enumerate(args):
        if axis == 1:
            if checks.is_pandas(arg):
                results.append(arg.iloc[:, new_indices[i]])
            else:
                results.append(arg[:, new_indices[i]])
        else:
            if checks.is_pandas(arg):
                results.append(arg.iloc[new_indices[i]])
            else:
                results.append(arg[new_indices[i]])
    if axis == 1:
        broadcast_kwargs = merge_dicts(dict(columns_from="stack"), broadcast_kwargs)
    else:
        broadcast_kwargs = merge_dicts(dict(index_from="stack"), broadcast_kwargs)
    return broadcast(*results, **broadcast_kwargs)


def get_multiindex_series(arg: tp.SeriesFrame) -> tp.Series:
    """Get Series with a multi-index.

    If DataFrame has been passed, must at maximum have one row or column."""
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    if checks.is_frame(arg):
        if arg.shape[0] == 1:
            arg = arg.iloc[0, :]
        elif arg.shape[1] == 1:
            arg = arg.iloc[:, 0]
        else:
            raise ValueError("Supported are either Series or DataFrame with one column/row")
    checks.assert_instance_of(arg.index, pd.MultiIndex)
    return arg


def unstack_to_array(
    arg: tp.SeriesFrame,
    levels: tp.Optional[tp.MaybeLevelSequence] = None,
    sort: bool = True,
    return_indexes: bool = False,
) -> tp.Union[tp.Array, tp.Tuple[tp.Array, tp.List[tp.Index]]]:
    """Reshape `arg` based on its multi-index into a multi-dimensional array.

    Use `levels` to specify what index levels to unstack and in which order.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import unstack_to_array

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_array(sr).shape
        (2, 2, 4)

        >>> unstack_to_array(sr)
        [[[ 1. nan nan nan]
         [nan  2. nan nan]]

         [[nan nan  3. nan]
        [nan nan nan  4.]]]

        >>> unstack_to_array(sr, levels=(2, 0))
        [[ 1. nan]
         [ 2. nan]
         [nan  3.]
         [nan  4.]]
        ```
    """
    sr = get_multiindex_series(arg)
    if sr.index.duplicated().any():
        raise ValueError("Index contains duplicate entries, cannot reshape")

    new_index_list = []
    value_indices_list = []
    if levels is None:
        levels = range(sr.index.nlevels)
    if isinstance(levels, (int, str)):
        levels = (levels,)
    for level in levels:
        level_values = indexes.select_levels(sr.index, level)
        new_index = level_values.unique()
        if sort:
            new_index = new_index.sort_values()
        new_index_list.append(new_index)
        index_map = pd.Series(range(len(new_index)), index=new_index)
        value_indices = index_map.loc[level_values]
        value_indices_list.append(value_indices)

    a = np.full(list(map(len, new_index_list)), np.nan)
    a[tuple(zip(value_indices_list))] = sr.values
    if return_indexes:
        return a, new_index_list
    return a


def make_symmetric(arg: tp.SeriesFrame, sort: bool = True) -> tp.Frame:
    """Make `arg` symmetric.

    The index and columns of the resulting DataFrame will be identical.

    Requires the index and columns to have the same number of levels.

    Pass `sort=False` if index and columns should not be sorted, but concatenated
    and get duplicates removed.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import make_symmetric

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])

        >>> make_symmetric(df)
             a    b    c    d
        a  NaN  NaN  1.0  2.0
        b  NaN  NaN  3.0  4.0
        c  1.0  3.0  NaN  NaN
        d  2.0  4.0  NaN  NaN
        ```
    """
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    df = to_2d(arg)
    if isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        checks.assert_instance_of(df.index, pd.MultiIndex)
        checks.assert_instance_of(df.columns, pd.MultiIndex)
        checks.assert_array_equal(df.index.nlevels, df.columns.nlevels)
        names1, names2 = tuple(df.index.names), tuple(df.columns.names)
    else:
        names1, names2 = df.index.name, df.columns.name

    if names1 == names2:
        new_name = names1
    else:
        if isinstance(df.index, pd.MultiIndex):
            new_name = tuple(zip(*[names1, names2]))
        else:
            new_name = (names1, names2)
    if sort:
        idx_vals = np.unique(np.concatenate((df.index, df.columns))).tolist()
    else:
        idx_vals = list(dict.fromkeys(np.concatenate((df.index, df.columns))))
    df_index = df.index.copy()
    df_columns = df.columns.copy()
    if isinstance(df.index, pd.MultiIndex):
        unique_index = pd.MultiIndex.from_tuples(idx_vals, names=new_name)
        df_index.names = new_name
        df_columns.names = new_name
    else:
        unique_index = pd.Index(idx_vals, name=new_name)
        df_index.name = new_name
        df_columns.name = new_name
    df = df.copy(deep=False)
    df.index = df_index
    df.columns = df_columns
    df_out_dtype = np.promote_types(df.values.dtype, np.min_scalar_type(np.nan))
    df_out = pd.DataFrame(index=unique_index, columns=unique_index, dtype=df_out_dtype)
    df_out.loc[:, :] = df
    df_out[df_out.isnull()] = df.transpose()
    return df_out


def unstack_to_df(
    arg: tp.SeriesFrame,
    index_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    column_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    symmetric: bool = False,
    sort: bool = True,
) -> tp.Frame:
    """Reshape `arg` based on its multi-index into a DataFrame.

    Use `index_levels` to specify what index levels will form new index, and `column_levels`
    for new columns. Set `symmetric` to True to make DataFrame symmetric.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import unstack_to_df

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']],
        ...     names=['x', 'y', 'z'])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_df(sr, index_levels=(0, 1), column_levels=2)
        z      a    b    c    d
        x y
        1 3  1.0  NaN  NaN  NaN
        1 4  NaN  2.0  NaN  NaN
        2 3  NaN  NaN  3.0  NaN
        2 4  NaN  NaN  NaN  4.0
        ```
    """
    sr = get_multiindex_series(arg)
    if sr.index.nlevels > 2:
        if index_levels is None:
            raise ValueError("index_levels must be specified")
        if column_levels is None:
            raise ValueError("column_levels must be specified")
    else:
        if index_levels is None:
            index_levels = 0
        if column_levels is None:
            column_levels = 1

    unstacked, (new_index, new_columns) = unstack_to_array(
        sr,
        levels=(index_levels, column_levels),
        sort=sort,
        return_indexes=True,
    )
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    if symmetric:
        return make_symmetric(df, sort=sort)
    return df
