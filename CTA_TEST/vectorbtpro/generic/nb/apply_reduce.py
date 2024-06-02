# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for mapping, applying, and reducing."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic.nb.base import nancov_1d_nb, nanstd_1d_nb, nancorr_1d_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


# ############# Map, apply, and reduce ############# #


@register_jitted
def map_1d_nb(arr: tp.Array1d, map_func_nb: tp.MapFunc, *args) -> tp.Array1d:
    """Map elements element-wise using `map_func_nb`.

    `map_func_nb` must accept the element and `*args`. Must return a single value."""
    i_0_out = map_func_nb(arr[0], *args)
    out = np.empty_like(arr, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, arr.shape[0]):
        out[i] = map_func_nb(arr[i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def map_nb(arr: tp.Array2d, map_func_nb: tp.MapFunc, *args) -> tp.Array2d:
    """2-dim version of `map_1d_nb`."""
    col_0_out = map_1d_nb(arr[:, 0], map_func_nb, *args)
    out = np.empty_like(arr, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_nb(arr[:, col], map_func_nb, *args)
    return out


@register_jitted
def map_1d_meta_nb(n: int, col: int, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array1d:
    """Meta version of `map_1d_nb`.

    `map_func_nb` must accept the row index, the column index, and `*args`.
    Must return a single value."""
    i_0_out = map_func_nb(0, col, *args)
    out = np.empty(n, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, n):
        out[i] = map_func_nb(i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(target_shape=ch.ShapeSlicer(axis=1), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def map_meta_nb(target_shape: tp.Shape, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array2d:
    """2-dim version of `map_1d_meta_nb`."""
    col_0_out = map_1d_meta_nb(target_shape[0], 0, map_func_nb, *args)
    out = np.empty(target_shape, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_meta_nb(target_shape[0], col, map_func_nb, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), apply_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """Apply function on each column of an object.

    `apply_func_nb` must accept the array and `*args`.
    Must return a single value or an array of shape `a.shape[1]`."""
    col_0_out = apply_func_nb(arr[:, 0], *args)
    out = np.empty_like(arr, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = apply_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(target_shape=ch.ShapeSlicer(axis=1), apply_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Meta version of `apply_nb` that prepends the column index to the arguments of `apply_func_nb`."""
    col_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, target_shape[1]):
        out[:, col] = apply_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=0),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=0), apply_func_nb=None, args=ch.ArgsTaker()),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def row_apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """`apply_nb` but applied on rows rather than columns."""
    row_0_out = apply_func_nb(arr[0, :], *args)
    out = np.empty_like(arr, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, arr.shape[0]):
        out[i, :] = apply_func_nb(arr[i, :], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=0),
    arg_take_spec=dict(target_shape=ch.ShapeSlicer(axis=0), apply_func_nb=None, args=ch.ArgsTaker()),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def row_apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Meta version of `row_apply_nb` that prepends the row index to the arguments of `apply_func_nb`."""
    row_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, target_shape[0]):
        out[i, :] = apply_func_nb(i, *args)
    return out


@register_jitted
def rolling_reduce_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Provide rolling window calculations.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    if minp is None:
        minp = window
    out = np.empty_like(arr, dtype=np.float_)
    nancnt_arr = np.empty(arr.shape[0], dtype=np.int_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            valid_cnt = window - (nancnt - nancnt_arr[i - window])
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            window_a = arr[from_i:to_i]
            out[i] = reduce_func_nb(window_a, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_reduce_nb(
    arr: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `rolling_reduce_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_reduce_1d_nb(arr[:, col], window, minp, reduce_func_nb, *args)
    return out


@register_jitted
def rolling_reduce_1d_meta_nb(
    n: int,
    col: int,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `rolling_reduce_1d_nb`.

    `reduce_func_nb` must accept the start row index, the end row index, the column, and `*args`.
    Must return a single value."""
    if minp is None:
        minp = window
    out = np.empty(n, dtype=np.float_)
    for i in range(n):
        valid_cnt = min(i + 1, window)
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            out[i] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        window=None,
        minp=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_reduce_meta_nb(
    target_shape: tp.Shape,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `rolling_reduce_1d_meta_nb`."""
    out = np.empty(target_shape, dtype=np.float_)
    for col in prange(target_shape[1]):
        out[:, col] = rolling_reduce_1d_meta_nb(target_shape[0], col, window, minp, reduce_func_nb, *args)
    return out


@register_jitted
def rolling_freq_reduce_1d_nb(
    index: tp.Array1d,
    arr: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Provide rolling, frequency-based window calculations.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    out = np.empty_like(arr, dtype=np.float_)
    from_i = 0
    for i in range(arr.shape[0]):
        if index[from_i] <= index[i] - freq:
            for j in range(from_i + 1, index.shape[0]):
                if index[j] > index[i] - freq:
                    from_i = j
                    break
        to_i = i + 1
        window_a = arr[from_i:to_i]
        out[i] = reduce_func_nb(window_a, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(index=None, arr=ch.ArraySlicer(axis=1), freq=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_freq_reduce_nb(
    index: tp.Array1d,
    arr: tp.Array2d,
    freq: np.timedelta64,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `rolling_reduce_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_freq_reduce_1d_nb(index, arr[:, col], freq, reduce_func_nb, *args)
    return out


@register_jitted
def rolling_freq_reduce_1d_meta_nb(
    col: int,
    index: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `rolling_freq_reduce_1d_nb`.

    `reduce_func_nb` must accept the start row index, the end row index, the column, and `*args`.
    Must return a single value."""
    out = np.empty(index.shape[0], dtype=np.float_)
    from_i = 0
    for i in range(index.shape[0]):
        if index[from_i] <= index[i] - freq:
            for j in range(from_i + 1, index.shape[0]):
                if index[j] > index[i] - freq:
                    from_i = j
                    break
        to_i = i + 1
        out[i] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(n_cols=ch.CountAdapter(), index=None, freq=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_freq_reduce_meta_nb(
    n_cols: int,
    index: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `rolling_freq_reduce_1d_meta_nb`."""
    out = np.empty((index.shape[0], n_cols), dtype=np.float_)
    for col in prange(n_cols):
        out[:, col] = rolling_freq_reduce_1d_meta_nb(col, index, freq, reduce_func_nb, *args)
    return out


@register_jitted
def groupby_reduce_1d_nb(arr: tp.Array1d, group_map: tp.GroupMap, reduce_func_nb: tp.ReduceFunc, *args) -> tp.Array1d:
    """Provide group-by reduce calculations.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[group_0_idxs], *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(arr[idxs], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), group_map=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def groupby_reduce_nb(arr: tp.Array2d, group_map: tp.GroupMap, reduce_func_nb: tp.ReduceFunc, *args) -> tp.Array2d:
    """2-dim version of `groupby_reduce_1d_nb`."""
    col_0_out = groupby_reduce_1d_nb(arr[:, 0], group_map, reduce_func_nb, *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = groupby_reduce_1d_nb(arr[:, col], group_map, reduce_func_nb, *args)
    return out


@register_jitted
def groupby_reduce_1d_meta_nb(
    col: int,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.GroupByReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `groupby_reduce_1d_nb`.

    `reduce_func_nb` must accept the array of indices in the group, the group index, the column index,
    and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, col, *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(idxs, group, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(n_cols=ch.CountAdapter(), group_map=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def groupby_reduce_meta_nb(
    n_cols: int,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.GroupByReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `groupby_reduce_1d_meta_nb`."""
    col_0_out = groupby_reduce_1d_meta_nb(0, group_map, reduce_func_nb, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = groupby_reduce_1d_meta_nb(col, group_map, reduce_func_nb, *args)
    return out


@register_jitted(tags={"can_parallel"})
def groupby_transform_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    transform_func_nb: tp.GroupByTransformFunc,
    *args,
) -> tp.Array2d:
    """Provide group-by transform calculations.

    `transform_func_nb` must accept the 2-dim array of the group and `*args`. Must return a scalar
    or an array that broadcasts against the group array's shape."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]: group_start_idxs[0] + group_lens[0]]
    group_0_out = transform_func_nb(arr[group_0_idxs], *args)
    out = np.empty(arr.shape, dtype=np.asarray(group_0_out).dtype)
    out[group_0_idxs] = group_0_out

    for group in prange(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx: start_idx + group_len]
        out[idxs] = transform_func_nb(arr[idxs], *args)
    return out


@register_jitted(tags={"can_parallel"})
def groupby_transform_meta_nb(
    target_shape: tp.Shape,
    group_map: tp.GroupMap,
    transform_func_nb: tp.GroupByTransformMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `groupby_transform_nb`.

    `transform_func_nb` must accept the array of indices in the group, the group index, and `*args`.
    Must return a scalar or an array that broadcasts against the group's shape."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]: group_start_idxs[0] + group_lens[0]]
    group_0_out = transform_func_nb(group_0_idxs, 0, *args)
    out = np.empty(target_shape, dtype=np.asarray(group_0_out).dtype)
    out[group_0_idxs] = group_0_out

    for group in prange(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx: start_idx + group_len]
        out[idxs] = transform_func_nb(idxs, group, *args)
    return out


@register_jitted
def reduce_index_ranges_1d_nb(
    arr: tp.Array1d,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce each index range.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    out = np.empty(range_starts.shape[0], dtype=np.float_)

    for k in range(len(range_starts)):
        from_i = range_starts[k]
        to_i = range_ends[k]
        if from_i == -1 or to_i == -1:
            out[k] = np.nan
        else:
            out[k] = reduce_func_nb(arr[from_i:to_i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        range_starts=None,
        range_ends=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_index_ranges_nb(
    arr: tp.Array2d,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `reduce_index_ranges_1d_nb`."""
    out = np.empty((range_starts.shape[0], arr.shape[1]), dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = reduce_index_ranges_1d_nb(arr[:, col], range_starts, range_ends, reduce_func_nb, *args)
    return out


@register_jitted
def reduce_index_ranges_1d_meta_nb(
    col: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `reduce_index_ranges_1d_nb`.

    `reduce_func_nb` must accept the start row index, the end row index, the column,
    and `*args`. Must return a single value."""
    out = np.empty(range_starts.shape[0], dtype=np.float_)

    for k in range(len(range_starts)):
        from_i = range_starts[k]
        to_i = range_ends[k]
        if from_i == -1 or to_i == -1:
            out[k] = np.nan
        else:
            out[k] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        range_starts=None,
        range_ends=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_index_ranges_meta_nb(
    n_cols: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `reduce_index_ranges_1d_meta_nb`."""
    out = np.empty((range_starts.shape[0], n_cols), dtype=np.float_)
    for col in prange(n_cols):
        out[:, col] = reduce_index_ranges_1d_meta_nb(col, range_starts, range_ends, reduce_func_nb, *args)
    return out


@register_jitted
def apply_and_reduce_1d_nb(
    arr: tp.Array1d,
    apply_func_nb: tp.ApplyFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceFunc,
    reduce_args: tuple,
) -> tp.Scalar:
    """Apply `apply_func_nb` and reduce into a single value using `reduce_func_nb`.

    `apply_func_nb` must accept the array and `*apply_args`.
    Must return an array.

    `reduce_func_nb` must accept the array of results from `apply_func_nb` and `*reduce_args`.
    Must return a single value."""
    temp = apply_func_nb(arr, *apply_args)
    return reduce_func_nb(temp, *reduce_args)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_and_reduce_nb(
    arr: tp.Array2d,
    apply_func_nb: tp.ApplyFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceFunc,
    reduce_args: tuple,
) -> tp.Array1d:
    """2-dim version of `apply_and_reduce_1d_nb`."""
    col_0_out = apply_and_reduce_1d_nb(arr[:, 0], apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = apply_and_reduce_1d_nb(arr[:, col], apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    return out


@register_jitted
def apply_and_reduce_1d_meta_nb(
    col: int,
    apply_func_nb: tp.ApplyMetaFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceMetaFunc,
    reduce_args: tuple,
) -> tp.Scalar:
    """Meta version of `apply_and_reduce_1d_nb`.

    `apply_func_nb` must accept the column index, the array, and `*apply_args`.
    Must return an array.

    `reduce_func_nb` must accept the column index, the array of results from `apply_func_nb`, and `*reduce_args`.
    Must return a single value."""
    temp = apply_func_nb(col, *apply_args)
    return reduce_func_nb(col, temp, *reduce_args)


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_and_reduce_meta_nb(
    n_cols: int,
    apply_func_nb: tp.ApplyMetaFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceMetaFunc,
    reduce_args: tuple,
) -> tp.Array1d:
    """2-dim version of `apply_and_reduce_1d_meta_nb`."""
    col_0_out = apply_and_reduce_1d_meta_nb(0, apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = apply_and_reduce_1d_meta_nb(col, apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceFunc, *args) -> tp.Array1d:
    """Reduce each column into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(n_cols=ch.CountAdapter(), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_meta_nb(n_cols: int, reduce_func_nb: tp.ReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_nb`.

    `reduce_func_nb` must accept the column index and `*args`. Must return a single value."""
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_to_array_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceToArrayFunc, *args) -> tp.Array2d:
    """Same as `reduce_nb` but `reduce_func_nb` must return an array."""
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(n_cols=ch.CountAdapter(), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_to_array_meta_nb(n_cols: int, reduce_func_nb: tp.ReduceToArrayMetaFunc, *args) -> tp.Array2d:
    """Same as `reduce_meta_nb` but `reduce_func_nb` must return an array."""
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedFunc,
    *args,
) -> tp.Array1d:
    """Reduce each group of columns into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept the 2-dim array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(group_map=base_ch.GroupMapSlicer(), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_meta_nb(group_map: tp.GroupMap, reduce_func_nb: tp.ReduceGroupedMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_grouped_nb`.

    `reduce_func_nb` must accept the column indices of the group, the group index, and `*args`.
    Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def flatten_forder_nb(arr: tp.Array2d) -> tp.Array1d:
    """Flatten the array in F order."""
    out = np.empty(arr.shape[0] * arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col * arr.shape[0] : (col + 1) * arr.shape[0]] = arr[:, col]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_flat_grouped_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    in_c_order: bool,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array1d:
    """Same as `reduce_grouped_nb` but passes flattened array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        if in_c_order:
            out[group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_to_array_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedToArrayFunc,
    *args,
) -> tp.Array2d:
    """Same as `reduce_grouped_nb` but `reduce_func_nb` must return an array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[:, group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(group_map=base_ch.GroupMapSlicer(), reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_to_array_meta_nb(
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Same as `reduce_grouped_meta_nb` but `reduce_func_nb` must return an array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[:, group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_flat_grouped_to_array_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    in_c_order: bool,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Same as `reduce_grouped_to_array_nb` but passes flattened array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        if in_c_order:
            out[:, group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[:, group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        squeeze_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def squeeze_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, squeeze_func_nb: tp.ReduceFunc, *args) -> tp.Array2d:
    """Squeeze each group of columns into a single column using `squeeze_func_nb`.

    `squeeze_func_nb` must accept index the array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(arr[0][group_0_idxs], *args)
    out = np.empty((arr.shape[0], len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(arr.shape[0]):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(arr[i][col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(n_rows=None, group_map=base_ch.GroupMapSlicer(), squeeze_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def squeeze_grouped_meta_nb(
    n_rows: int,
    group_map: tp.GroupMap,
    squeeze_func_nb: tp.GroupSqueezeMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `squeeze_grouped_nb`.

    `squeeze_func_nb` must accept the row index, the column indices of the group,
    the group index, and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(0, group_0_idxs, 0, *args)
    out = np.empty((n_rows, len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(n_rows):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(i, col_idxs, group, *args)
    return out


# ############# Flattening ############# #


@register_jitted(cache=True)
def flatten_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool) -> tp.Array2d:
    """Flatten each group of columns."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((arr.shape[0] * np.max(group_lens), len(group_lens)), np.nan, dtype=np.float_)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0] : (k + 1) * arr.shape[0], group] = arr[:, col]
    return out


@register_jitted(cache=True)
def flatten_uniform_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool) -> tp.Array2d:
    """Flatten each group of columns of the same length."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty((arr.shape[0] * np.max(group_lens), len(group_lens)), dtype=arr.dtype)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0] : (k + 1) * arr.shape[0], group] = arr[:, col]
    return out


# ############# Proximity ############# #

@register_jitted(tags={"can_parallel"})
def proximity_reduce_nb(
    arr: tp.Array2d,
    window: int,
    reduce_func_nb: tp.ProximityReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """Flatten `window` surrounding rows and columns and reduce them into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    out = np.empty_like(arr, dtype=np.float_)
    for i in prange(arr.shape[0]):
        for col in range(arr.shape[1]):
            from_i = max(0, i - window)
            to_i = min(i + window + 1, arr.shape[0])
            from_col = max(0, col - window)
            to_col = min(col + window + 1, arr.shape[1])
            stride_arr = arr[from_i:to_i, from_col:to_col]
            out[i, col] = reduce_func_nb(stride_arr.flatten(), *args)
    return out


@register_jitted(tags={"can_parallel"})
def proximity_reduce_meta_nb(
    target_shape: tp.Shape,
    window: int,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `proximity_reduce_nb`.

    `reduce_func_nb` must accept the start row index, the end row index, the start column index,
    the end column index, and `*args`. Must return a single value."""
    out = np.empty(target_shape, dtype=np.float_)
    for i in prange(target_shape[0]):
        for col in range(target_shape[1]):
            from_i = max(0, i - window)
            to_i = min(i + window + 1, target_shape[0])
            from_col = max(0, col - window)
            to_col = min(col + window + 1, target_shape[1])
            out[i, col] = reduce_func_nb(from_i, to_i, from_col, to_col, *args)
    return out


# ############# Reducers ############# #


@register_jitted(cache=True)
def nth_reduce_nb(arr: tp.Array1d, n: int) -> float:
    """Get n-th element."""
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    return arr[n]


@register_jitted(cache=True)
def first_reduce_nb(arr: tp.Array1d) -> float:
    """Get first non-NA element."""
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            return arr[i]
    return np.nan


@register_jitted(cache=True)
def last_reduce_nb(arr: tp.Array1d) -> float:
    """Get last non-NA element."""
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr) - 1, -1, -1):
        if not np.isnan(arr[i]):
            return arr[i]
    return np.nan


@register_jitted(cache=True)
def first_index_reduce_nb(arr: tp.Array1d) -> int:
    """Get index of first non-NA element."""
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def last_index_reduce_nb(arr: tp.Array1d) -> int:
    """Get index of last non-NA element."""
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr) - 1, -1, -1):
        if not np.isnan(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def nth_index_reduce_nb(arr: tp.Array1d, n: int) -> int:
    """Get index of n-th element including NA elements."""
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    if n >= 0:
        return n
    return arr.shape[0] + n


@register_jitted(cache=True)
def any_reduce_nb(arr: tp.Array1d) -> bool:
    """Get whether any of the elements are True."""
    return np.any(arr)


@register_jitted(cache=True)
def all_reduce_nb(arr: tp.Array1d) -> bool:
    """Get whether all of the elements are True."""
    return np.all(arr)


@register_jitted(cache=True)
def min_reduce_nb(arr: tp.Array1d) -> float:
    """Get min. Ignores NaN."""
    return np.nanmin(arr)


@register_jitted(cache=True)
def max_reduce_nb(arr: tp.Array1d) -> float:
    """Get max. Ignores NaN."""
    return np.nanmax(arr)


@register_jitted(cache=True)
def mean_reduce_nb(arr: tp.Array1d) -> float:
    """Get mean. Ignores NaN."""
    return np.nanmean(arr)


@register_jitted(cache=True)
def median_reduce_nb(arr: tp.Array1d) -> float:
    """Get median. Ignores NaN."""
    return np.nanmedian(arr)


@register_jitted(cache=True)
def std_reduce_nb(arr: tp.Array1d, ddof) -> float:
    """Get std. Ignores NaN."""
    return nanstd_1d_nb(arr, ddof=ddof)


@register_jitted(cache=True)
def sum_reduce_nb(arr: tp.Array1d) -> float:
    """Get sum. Ignores NaN."""
    return np.nansum(arr)


@register_jitted(cache=True)
def prod_reduce_nb(arr: tp.Array1d) -> float:
    """Get product. Ignores NaN."""
    return np.nanprod(arr)


@register_jitted(cache=True)
def nonzero_prod_reduce_nb(arr: tp.Array1d) -> float:
    """Get product. Ignores zero and NaN. Default value is zero."""
    prod = 0.0
    for i in range(len(arr)):
        if not np.isnan(arr[i]) and arr[i] != 0:
            if prod == 0:
                prod = 1.0
            prod *= arr[i]
    return prod


@register_jitted(cache=True)
def count_reduce_nb(arr: tp.Array1d) -> int:
    """Get count. Ignores NaN."""
    return np.sum(~np.isnan(arr))


@register_jitted(cache=True)
def argmin_reduce_nb(arr: tp.Array1d) -> int:
    """Get position of min."""
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = np.inf
    return np.argmin(arr)


@register_jitted(cache=True)
def argmax_reduce_nb(arr: tp.Array1d) -> int:
    """Get position of max."""
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = -np.inf
    return np.argmax(arr)


@register_jitted(cache=True)
def describe_reduce_nb(arr: tp.Array1d, perc: tp.Array1d, ddof: int) -> tp.Array1d:
    """Get descriptive statistics. Ignores NaN.

    Numba equivalent to `pd.Series(arr).describe(perc)`."""
    arr = arr[~np.isnan(arr)]
    out = np.empty(5 + len(perc), dtype=np.float_)
    out[0] = len(arr)
    if len(arr) > 0:
        out[1] = np.mean(arr)
        out[2] = nanstd_1d_nb(arr, ddof=ddof)
        out[3] = np.min(arr)
        out[4:-1] = np.percentile(arr, perc * 100)
        out[4 + len(perc)] = np.max(arr)
    else:
        out[1:] = np.nan
    return out


@register_jitted(cache=True)
def cov_reduce_grouped_meta_nb(
    group_idxs: tp.GroupIdxs,
    group: int,
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    ddof: int,
) -> float:
    """Get correlation coefficient. Ignores NaN."""
    return nancov_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten(), ddof=ddof)


@register_jitted(cache=True)
def corr_reduce_grouped_meta_nb(group_idxs: tp.GroupIdxs, group: int, arr1: tp.Array2d, arr2: tp.Array2d) -> float:
    """Get correlation coefficient. Ignores NaN."""
    return nancorr_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten())


@register_jitted(cache=True)
def wmean_range_reduce_meta_nb(from_i: int, to_i: int, col: int, arr1: tp.Array2d, arr2: tp.Array2d) -> float:
    """Get the weighted average."""
    nom_cumsum = 0
    denum_cumsum = 0
    for i in range(from_i, to_i):
        nom_cumsum += arr1[i, col] * arr2[i, col]
        denum_cumsum += arr2[i, col]
    if denum_cumsum == 0:
        return np.nan
    return nom_cumsum / denum_cumsum
