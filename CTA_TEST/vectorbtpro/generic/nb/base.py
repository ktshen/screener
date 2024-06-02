# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for base operations."""

import numpy as np
from numba import prange
from numba.core.types import Omitted
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


@register_jitted(cache=True, is_generated_jit=True)
def select_indices_1d_nb(arr: tp.Array1d, indices: tp.Array1d, fill_value: tp.Scalar) -> tp.Array1d:
    """Set each element to a value by boolean mask."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _select_indices_1d_nb(arr, indices, fill_value):
        out = np.empty(indices.shape, dtype=dtype)
        for i in range(indices.shape[0]):
            if 0 <= indices[i] <= arr.shape[0] - 1:
                out[i] = arr[indices[i]]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _select_indices_1d_nb(arr, indices, fill_value)

    return _select_indices_1d_nb


@register_jitted(cache=True, is_generated_jit=True)
def select_indices_nb(arr: tp.Array2d, indices: tp.Array2d, fill_value: tp.Scalar) -> tp.Array2d:
    """Set each element to a value by boolean mask."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _select_indices_nb(arr, indices, fill_value):
        out = np.empty(indices.shape, dtype=dtype)
        for col in range(indices.shape[1]):
            for i in range(indices.shape[0]):
                if 0 <= indices[i, col] <= arr.shape[0] - 1:
                    out[i, col] = arr[indices[i, col], col]
                else:
                    out[i, col] = fill_value
        return out

    if not nb_enabled:
        return _select_indices_nb(arr, indices, fill_value)

    return _select_indices_nb


@register_jitted(cache=True)
def shuffle_1d_nb(arr: tp.Array1d, seed: tp.Optional[int] = None) -> tp.Array1d:
    """Shuffle each column in the array.

    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(arr)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), seed=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def shuffle_nb(arr: tp.Array2d, seed: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `shuffle_1d_nb`."""
    if seed is not None:
        np.random.seed(seed)
    out = np.empty_like(arr, dtype=arr.dtype)

    for col in range(arr.shape[1]):
        out[:, col] = np.random.permutation(arr[:, col])
    return out


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_1d_nb(arr: tp.Array1d, mask: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Set each element to a value by boolean mask."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_1d_nb(arr, mask, value):
        out = arr.astype(dtype)
        out[mask] = value
        return out

    if not nb_enabled:
        return _set_by_mask_1d_nb(arr, mask, value)

    return _set_by_mask_1d_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_nb(arr: tp.Array2d, mask: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """2-dim version of `set_by_mask_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_nb(arr, mask, value):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = value
        return out

    if not nb_enabled:
        return _set_by_mask_nb(arr, mask, value)

    return _set_by_mask_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_mult_1d_nb(arr: tp.Array1d, mask: tp.Array1d, values: tp.Array1d) -> tp.Array1d:
    """Set each element in one array to the corresponding element in another by boolean mask.

    `values` must be of the same shape as in the array."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_mult_1d_nb(arr, mask, values):
        out = arr.astype(dtype)
        out[mask] = values[mask]
        return out

    if not nb_enabled:
        return _set_by_mask_mult_1d_nb(arr, mask, values)

    return _set_by_mask_mult_1d_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_mult_nb(arr: tp.Array2d, mask: tp.Array2d, values: tp.Array2d) -> tp.Array2d:
    """2-dim version of `set_by_mask_mult_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_mult_nb(arr, mask, values):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = values[mask[:, col], col]
        return out

    if not nb_enabled:
        return _set_by_mask_mult_nb(arr, mask, values)

    return _set_by_mask_mult_nb


@register_jitted(cache=True)
def first_valid_index_1d_nb(arr: tp.Array1d) -> int:
    """Get the index of the first valid value."""
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]) and not np.isinf(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def first_valid_index_nb(arr):
    """2-dim version of `first_valid_index_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.int_)
    for col in range(arr.shape[1]):
        out[col] = first_valid_index_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def last_valid_index_1d_nb(arr: tp.Array1d) -> int:
    """Get the index of the last valid value."""
    for i in range(arr.shape[0] - 1, -1, -1):
        if not np.isnan(arr[i]) and not np.isinf(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def last_valid_index_nb(arr):
    """2-dim version of `last_valid_index_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.int_)
    for col in range(arr.shape[1]):
        out[col] = last_valid_index_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def fillna_1d_nb(arr: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Replace NaNs with value.

    Numba equivalent to `pd.Series(arr).fillna(value)`."""
    return set_by_mask_1d_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def fillna_nb(arr: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """2-dim version of `fillna_1d_nb`."""
    return set_by_mask_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def fbfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """Forward and backward fill NaN values.

    !!! note
        If there are no NaN (or any) values, will return `arr`."""
    if arr.size == 0:
        return arr
    need_fbfill = False
    for col in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if np.isnan(arr[i, col]):
                need_fbfill = True
                break
        if need_fbfill:
            break
    if not need_fbfill:
        return arr

    out = np.empty_like(arr)
    for col in prange(arr.shape[1]):
        last_valid = np.nan
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i, col]):
                last_valid = arr[i, col]
            out[i, col] = last_valid
        if np.isnan(out[0, col]):
            last_valid = np.nan
            for i in range(arr.shape[0] - 1, -1, -1):
                if not np.isnan(arr[i, col]):
                    last_valid = arr[i, col]
                if np.isnan(out[i, col]):
                    out[i, col] = last_valid
    return out


@register_jitted(cache=True, is_generated_jit=True)
def bshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Shift backward by `n` positions.

    Numba equivalent to `pd.Series(arr).shift(-n)`.

    !!! warning
        This operation looks ahead."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _bshift_1d_nb(arr, n, fill_value):
        out = np.empty(arr.shape[0], dtype=dtype)
        for i in range(out.shape[0]):
            if i + n <= out.shape[0] - 1:
                out[i] = arr[i + n]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _bshift_1d_nb(arr, n, fill_value)

    return _bshift_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None, fill_value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, is_generated_jit=True)
def bshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """2-dim version of `bshift_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _bshift_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = bshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _bshift_nb(arr, n, fill_value)

    return _bshift_nb


@register_jitted(cache=True, is_generated_jit=True)
def fshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Shift forward by `n` positions.

    Numba equivalent to `pd.Series(arr).shift(n)`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _fshift_1d_nb(arr, n, fill_value):
        out = np.empty(arr.shape[0], dtype=dtype)
        for i in range(out.shape[0]):
            if i - n >= 0:
                out[i] = arr[i - n]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _fshift_1d_nb(arr, n, fill_value)

    return _fshift_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None, fill_value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, is_generated_jit=True)
def fshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """2-dim version of `fshift_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _fshift_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = fshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _fshift_nb(arr, n, fill_value)

    return _fshift_nb


@register_jitted(cache=True)
def diff_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the 1-th discrete difference.

    Numba equivalent to `pd.Series(arr).diff()`."""
    out = np.empty(arr.shape[0], dtype=np.float_)
    for i in range(out.shape[0]):
        if i - n >= 0:
            out[i] = arr[i] - arr[i - n]
        else:
            out[i] = np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def diff_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """2-dim version of `diff_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = diff_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def pct_change_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the percentage change.

    Numba equivalent to `pd.Series(arr).pct_change()`."""
    out = np.empty(arr.shape[0], dtype=np.float_)
    for i in range(out.shape[0]):
        if i - n >= 0:
            out[i] = (arr[i] - arr[i - n]) / arr[i - n]
        else:
            out[i] = np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pct_change_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """2-dim version of `pct_change_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = pct_change_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def bfill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaNs by propagating first valid observation backward.

    Numba equivalent to `pd.Series(arr).fillna(method='bfill')`.

    !!! warning
        This operation looks ahead."""
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[-1]
    for i in range(arr.shape[0] - 1, -1, -1):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `bfill_1d_nb`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = bfill_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def ffill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaNs by propagating last valid observation forward.

    Numba equivalent to `pd.Series(arr).fillna(method='ffill')`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ffill_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `ffill_1d_nb`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = ffill_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, is_generated_jit=True, tags={"can_parallel"})
def nanprod_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanprod` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nanprod_nb(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nanprod(arr[:, col])
        return out

    if not nb_enabled:
        return _nanprod_nb(arr)

    return _nanprod_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, is_generated_jit=True, tags={"can_parallel"})
def nancumsum_nb(arr: tp.Array2d) -> tp.Array2d:
    """Numba equivalent of `np.nancumsum` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nancumsum_nb(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumsum(arr[:, col])
        return out

    if not nb_enabled:
        return _nancumsum_nb(arr)

    return _nancumsum_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, is_generated_jit=True, tags={"can_parallel"})
def nancumprod_nb(arr: tp.Array2d) -> tp.Array2d:
    """Numba equivalent of `np.nancumprod` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nancumprod_nb(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumprod(arr[:, col])
        return out

    if not nb_enabled:
        return _nancumprod_nb(arr)

    return _nancumprod_nb


@register_jitted(cache=True)
def nancnt_1d_nb(arr: tp.Array1d) -> int:
    """Compute count while ignoring NaNs and not allocating any arrays."""
    cnt = 0
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            cnt += 1
    return cnt


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancnt_nb(arr: tp.Array2d) -> tp.Array1d:
    """2-dim version of `nancnt_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[col] = nancnt_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, is_generated_jit=True, tags={"can_parallel"})
def nansum_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nansum` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nansum_nb(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nansum(arr[:, col])
        return out

    if not nb_enabled:
        return _nansum_nb(arr)

    return _nansum_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmin_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmin` along axis 0."""
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmin(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmax_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmax` along axis 0."""
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmax(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmean_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmean` along axis 0."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmean(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmedian_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmedian` along axis 0."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmedian(arr[:, col])
    return out


@register_jitted(cache=True)
def nanpercentile_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Numba equivalent of `np.nanpercentile` that does not allocate any arrays.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.nanpercentile`,
        but still faster if used in rolling calculations, especially for `q` near 0 and 100."""
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    do_min = q < 50
    if not do_min:
        q = 100 - q
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth_float = q / 100 * (cnt - 1)
    if nth_float % 1 == 0:
        nth1 = nth2 = int(nth_float)
    else:
        nth1 = int(nth_float)
        nth2 = nth1 + 1
    found1 = np.nan
    found2 = np.nan
    k = 0
    if do_min:
        prev_val = -np.inf
    else:
        prev_val = np.inf
    while True:
        n_same = 0
        if do_min:
            curr_val = np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] > prev_val:
                        if arr[i] < curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        else:
            curr_val = -np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] < prev_val:
                        if arr[i] > curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        prev_val = curr_val
        k += n_same
        if np.isnan(found1) and k >= nth1 + 1:
            found1 = curr_val
        if np.isnan(found2) and k >= nth2 + 1:
            found2 = curr_val
            break
    if found1 == found2:
        return found1
    factor = (nth_float - nth1) / (nth2 - nth1)
    return factor * (found2 - found1) + found1


@register_jitted(cache=True)
def nanpartition_mean_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Average of `np.partition` that ignores NaN values and does not allocate any arrays.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.partition`,
        but still faster if used in rolling calculations, especially for `q` near 0."""
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth = int(q / 100 * (cnt - 1))
    prev_val = -np.inf
    partition_sum = 0.0
    partition_cnt = 0
    k = 0
    while True:
        n_same = 0
        curr_val = np.inf
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i]):
                if arr[i] > prev_val:
                    if arr[i] < curr_val:
                        curr_val = arr[i]
                        n_same = 0
                    if arr[i] == curr_val:
                        n_same += 1
        if k + n_same >= nth + 1:
            partition_sum += (nth + 1 - k) * curr_val
            partition_cnt += nth + 1 - k
            break
        else:
            partition_sum += n_same * curr_val
            partition_cnt += n_same
        prev_val = curr_val
        k += n_same
    return partition_sum / partition_cnt


@register_jitted(cache=True)
def nanvar_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.nanvar` that does not allocate any arrays."""
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    rcount = max(cnt - ddof, 0)
    if rcount == 0:
        return np.nan
    out = 0.0
    a_mean = np.nanmean(arr)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            out += abs(arr[i] - a_mean) ** 2
    return out / rcount


@register_jitted(cache=True)
def nanstd_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.nanstd`."""
    return np.sqrt(nanvar_1d_nb(arr, ddof=ddof))


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanstd_nb(arr: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `nanstd_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = nanstd_1d_nb(arr[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancov_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.cov` that ignores NaN values."""
    arr1_sum = 0.0
    arr2_sum = 0.0
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            k += 1
    if k - ddof <= 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    num = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            num += (arr1[i] - arr1_mean) * (arr2[i] - arr2_mean)
    return num / (k - ddof)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancov_nb(arr1: tp.Array2d, arr2: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `nancov_1d_nb`."""
    out = np.empty(arr1.shape[1], dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancov_1d_nb(arr1[:, col], arr2[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancorr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d) -> float:
    """Numba equivalent of `np.corrcoef` that ignores NaN values.

    Numerically stable."""
    arr1_sum = 0.0
    arr2_sum = 0.0
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            k += 1
    if k == 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    num = 0
    denom1 = 0
    denom2 = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            num += (arr1[i] - arr1_mean) * (arr2[i] - arr2_mean)
            denom1 += (arr1[i] - arr1_mean) ** 2
            denom2 += (arr2[i] - arr2_mean) ** 2
    denom = np.sqrt(denom1 * denom2)
    if denom == 0:
        return np.nan
    return num / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancorr_nb(arr1: tp.Array2d, arr2: tp.Array2d) -> tp.Array1d:
    """2-dim version of `nancorr_1d_nb`."""
    out = np.empty(arr1.shape[1], dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancorr_1d_nb(arr1[:, col], arr2[:, col])
    return out


@register_jitted(cache=True)
def rank_1d_nb(arr: tp.Array1d, argsorted: tp.Optional[tp.Array1d] = None, pct: bool = False) -> tp.Array1d:
    """Compute numerical data ranks.

    Numba equivalent to `pd.Series(arr).rank(pct=pct)`."""
    if argsorted is None:
        argsorted = np.argsort(arr)
    out = np.empty_like(arr, dtype=np.float_)
    rank_sum = 0
    rank_cnt = 0
    nan_cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nan_cnt += 1
    if nan_cnt == arr.shape[0]:
        out[:] = np.nan
        return out
    valid_cnt = out.shape[0] - nan_cnt
    for i in range(argsorted.shape[0]):
        rank = i + 1
        if np.isnan(arr[argsorted[i]]):
            out[argsorted[i]] = np.nan
        elif i < out.shape[0] - 1 and arr[argsorted[i]] == arr[argsorted[i + 1]]:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
        elif rank_sum > 0:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank_sum / rank_cnt / valid_cnt
            else:
                v = rank_sum / rank_cnt
            out[argsorted[i - rank_cnt + 1 : i + 1]] = v
            rank_sum = 0
            rank_cnt = 0
        else:
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), argsorted=ch.ArraySlicer(axis=1), pct=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rank_nb(arr: tp.Array2d, argsorted: tp.Optional[tp.Array2d] = None, pct: bool = False) -> tp.Array2d:
    """2-dim version of `rank_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        if argsorted is None:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=None, pct=pct)
        else:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=argsorted[:, col], pct=pct)
    return out


# ############# Value counts ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="codes", axis=1),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        n_uniques=None,
        group_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_counts_nb(codes: tp.Array2d, n_uniques: int, group_map: tp.GroupMap) -> tp.Array2d:
    """Compute value counts per column/group."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((n_uniques, group_lens.shape[0]), 0, dtype=np.int_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            for i in range(codes.shape[0]):
                out[codes[i, col], group] += 1
    return out


@register_jitted(cache=True)
def value_counts_1d_nb(codes: tp.Array1d, n_uniques: int) -> tp.Array1d:
    """Compute value counts."""
    out = np.full(n_uniques, 0, dtype=np.int_)

    for i in range(codes.shape[0]):
        out[codes[i]] += 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="codes", axis=0),
    arg_take_spec=dict(codes=ch.ArraySlicer(axis=0), n_uniques=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_counts_per_row_nb(codes: tp.Array2d, n_uniques: int) -> tp.Array2d:
    """Compute value counts per row."""
    out = np.empty((n_uniques, codes.shape[0]), dtype=np.int_)

    for i in prange(codes.shape[0]):
        out[:, i] = value_counts_1d_nb(codes[i, :], n_uniques)
    return out


# ############# Repartitioning ############# #


@register_jitted(cache=True)
def repartition_nb(arr: tp.Array2d, counts: tp.Array1d) -> tp.Array1d:
    """Repartition a 2-dimensional array into a 1-dimensional by removing empty elements."""
    if arr.shape[0] == 0:
        return arr.flatten()
    out = np.empty(np.sum(counts), dtype=arr.dtype)
    j = 0
    for col in range(counts.shape[0]):
        out[j : j + counts[col]] = arr[: counts[col], col]
        j += counts[col]
    return out


# ############# Crossover ############# #


@register_jitted(cache=True)
def crossed_above_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, wait: int = 0, dropna: bool = False) -> tp.Array1d:
    """Get the crossover of the first array going above the second array.

    If `dropna` is True, produces the same results as if all rows with at least one NaN were dropped."""
    out = np.empty(arr1.shape, dtype=np.bool_)
    was_below = False
    confirmed = 0

    for i in range(arr1.shape[0]):
        if np.isnan(arr1[i]) or np.isnan(arr2[i]):
            if not dropna:
                was_below = False
                confirmed = 0
            out[i] = False
        elif arr1[i] > arr2[i]:
            if was_below:
                confirmed += 1
                out[i] = confirmed == wait + 1
            else:
                out[i] = False
        elif arr1[i] == arr2[i]:
            if confirmed > 0:
                was_below = False
            confirmed = 0
            out[i] = False
        elif arr1[i] < arr2[i]:
            confirmed = 0
            was_below = True
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        wait=None,
        dropna=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def crossed_above_nb(arr1: tp.Array2d, arr2: tp.Array2d, wait: int = 0, dropna: bool = False) -> tp.Array2d:
    """2-dim version of `crossed_above_1d_nb`."""
    out = np.empty(arr1.shape, dtype=np.bool_)
    for col in prange(arr1.shape[1]):
        out[:, col] = crossed_above_1d_nb(arr1[:, col], arr2[:, col], wait=wait, dropna=dropna)
    return out


@register_jitted(cache=True)
def crossed_below_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, wait: int = 0, dropna: bool = False) -> tp.Array1d:
    """Get the crossover of the first array going below the second array.

    Calls `crossed_above_1d_nb` but with the arguments switched."""
    return crossed_above_1d_nb(arr2, arr1, wait=wait, dropna=dropna)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        wait=None,
        dropna=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def crossed_below_nb(arr1: tp.Array2d, arr2: tp.Array2d, wait: int = 0, dropna: bool = False) -> tp.Array2d:
    """2-dim version of `crossed_below_1d_nb`."""
    out = np.empty(arr1.shape, dtype=np.bool_)
    for col in prange(arr1.shape[1]):
        out[:, col] = crossed_below_1d_nb(arr1[:, col], arr2[:, col], wait=wait, dropna=dropna)
    return out


# ############# Transformation ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def demean_nb(arr: tp.Array2d, group_map: tp.GroupMap) -> tp.Array2d:
    """Demean each value within its group."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty_like(arr, dtype=np.float_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(arr.shape[0]):
            group_sum = 0
            group_cnt = 0
            for k in range(group_len):
                col = col_idxs[k]
                if not np.isnan(arr[i, col]):
                    group_sum += arr[i, col]
                    group_cnt += 1
            for k in range(group_len):
                col = col_idxs[k]
                if np.isnan(arr[i, col]) or group_cnt == 0:
                    out[i, col] = np.nan
                else:
                    out[i, col] = arr[i, col] - group_sum / group_cnt
    return out


@register_jitted(cache=True)
def to_renko_1d_nb(
    arr: tp.Array1d,
    brick_size: tp.FlexArray1dLike,
    relative: tp.FlexArray1dLike = False,
    start_value: tp.Optional[float] = None,
    max_out_len: tp.Optional[int] = None
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Convert to Renko format."""
    brick_size_ = to_1d_array_nb(np.asarray(brick_size))
    relative_ = to_1d_array_nb(np.asarray(relative))

    if max_out_len is None:
        out_n = arr.shape[0]
    else:
        out_n = max_out_len
    arr_out = np.empty(out_n, dtype=np.float_)
    idx_out = np.empty(out_n, dtype=np.int_)
    uptrend_out = np.empty(out_n, dtype=np.bool_)
    prev_value = np.nan
    k = 0
    trend = 0

    for i in range(arr.shape[0]):
        _brick_size = abs(flex_select_1d_nb(brick_size_, i))
        _relative = flex_select_1d_nb(relative_, i)
        curr_value = arr[i]
        if np.isnan(curr_value):
            continue
        if np.isnan(prev_value):
            if start_value is None:
                if not _relative:
                    prev_value = curr_value - curr_value % _brick_size
                else:
                    prev_value = curr_value
            else:
                prev_value = start_value
            continue
        if _relative:
            diff = (curr_value - prev_value) / prev_value
        else:
            diff = curr_value - prev_value
        while abs(diff) >= _brick_size:
            prev_trend = trend
            if diff >= 0:
                if _relative:
                    prev_value *= 1 + _brick_size
                else:
                    prev_value += _brick_size
                trend = 1
            else:
                if _relative:
                    prev_value *= 1 - _brick_size
                else:
                    prev_value -= _brick_size
                trend = -1
            if _relative:
                diff = (curr_value - prev_value) / prev_value
            else:
                diff = curr_value - prev_value
            if trend == -prev_trend:
                continue
            if k >= len(arr_out):
                raise IndexError("Index out of range. Set a higher max_out_len.")
            arr_out[k] = prev_value
            idx_out[k] = i
            uptrend_out[k] = trend == 1
            k += 1

    return arr_out[:k], idx_out[:k], uptrend_out[:k]


@register_jitted(cache=True)
def to_renko_ohlc_1d_nb(
    arr: tp.Array1d,
    brick_size: tp.FlexArray1dLike,
    relative: tp.FlexArray1dLike = False,
    start_value: tp.Optional[float] = None,
    max_out_len: tp.Optional[int] = None
) -> tp.Tuple[tp.Array2d, tp.Array1d]:
    """Convert to Renko OHLC format."""
    brick_size_ = to_1d_array_nb(np.asarray(brick_size))
    relative_ = to_1d_array_nb(np.asarray(relative))

    if max_out_len is None:
        out_n = arr.shape[0]
    else:
        out_n = max_out_len
    arr_out = np.empty((out_n, 4), dtype=np.float_)
    idx_out = np.empty(out_n, dtype=np.int_)
    prev_value = np.nan
    k = 0
    trend = 0

    for i in range(arr.shape[0]):
        _brick_size = abs(flex_select_1d_nb(brick_size_, i))
        _relative = flex_select_1d_nb(relative_, i)
        curr_value = arr[i]
        if np.isnan(curr_value):
            continue
        if np.isnan(prev_value):
            if start_value is None:
                if not _relative:
                    prev_value = curr_value - curr_value % _brick_size
                else:
                    prev_value = curr_value
            else:
                prev_value = start_value
            continue
        if _relative:
            diff = (curr_value - prev_value) / prev_value
        else:
            diff = curr_value - prev_value
        while abs(diff) >= _brick_size:
            open_value = prev_value
            prev_trend = trend
            if diff >= 0:
                if _relative:
                    prev_value *= 1 + _brick_size
                else:
                    prev_value += _brick_size
                trend = 1
            else:
                if _relative:
                    prev_value *= 1 - _brick_size
                else:
                    prev_value -= _brick_size
                trend = -1
            if _relative:
                diff = (curr_value - prev_value) / prev_value
            else:
                diff = curr_value - prev_value
            if trend == -prev_trend:
                continue
            if k >= len(arr_out):
                raise IndexError("Index out of range. Set a higher max_out_len.")
            if trend == 1:
                high_value = prev_value
                low_value = open_value
            else:
                high_value = open_value
                low_value = prev_value
            close_value = prev_value
            arr_out[k, 0] = open_value
            arr_out[k, 1] = high_value
            arr_out[k, 2] = low_value
            arr_out[k, 3] = close_value
            idx_out[k] = i
            k += 1

    return arr_out[:k], idx_out[:k]


# ############# Resampling ############# #

@register_jitted(cache=True, is_generated_jit=True)
def latest_at_index_1d_nb(
    arr: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = None,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array1d:
    """Get the latest in `arr` at each index in `target_index` based on `source_index`.

    If `source_rbound` is True, then each element in `source_index` is effectively located at
    the right bound, which is the frequency or the next element (excluding) if the frequency is None.
    The same for `target_rbound` and `target_index`.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed.

        If `arr` contains bar data, both indexes must represent the opening time."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _latest_at_index_1d_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty(target_index.shape[0], dtype=dtype)
        curr_j = -1
        last_j = curr_j
        last_valid = np.nan
        for i in range(len(target_index)):
            if i > 0 and target_index[i] < target_index[i - 1]:
                raise ValueError("Target index must be increasing")
            target_bound_inf = target_rbound and i == len(target_index) - 1 and target_freq is None

            last_valid_at_i = np.nan
            for j in range(curr_j + 1, source_index.shape[0]):
                if j > 0 and source_index[j] < source_index[j - 1]:
                    raise ValueError("Array index must be increasing")
                source_bound_inf = source_rbound and j == len(source_index) - 1 and source_freq is None

                if source_bound_inf and target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    break
                if source_bound_inf:
                    break
                if target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    continue

                if source_rbound and target_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_val > target_val:
                        break
                elif source_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if source_val > target_index[i]:
                        break
                elif target_rbound:
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_index[j] >= target_val:
                        break
                else:
                    if source_index[j] > target_index[i]:
                        break
                curr_j = j
                if not np.isnan(arr[curr_j]):
                    last_valid_at_i = arr[curr_j]

            if ffill and not np.isnan(last_valid_at_i):
                last_valid = last_valid_at_i
            if curr_j == -1 or (not ffill and curr_j == last_j):
                out[i] = nan_value
            else:
                if ffill:
                    if np.isnan(last_valid):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid
                else:
                    if np.isnan(last_valid_at_i):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid_at_i
                last_j = curr_j

        return out

    if not nb_enabled:
        return _latest_at_index_1d_nb(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _latest_at_index_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        source_index=None,
        target_index=None,
        source_freq=None,
        target_freq=None,
        source_rbound=None,
        target_rbound=None,
        nan_value=None,
        ffill=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"}, is_generated_jit=True)
def latest_at_index_nb(
    arr: tp.Array2d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = False,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array2d:
    """2-dim version of `latest_at_index_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _latest_at_index_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty((target_index.shape[0], arr.shape[1]), dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = latest_at_index_1d_nb(
                arr[:, col],
                source_index,
                target_index,
                source_freq=source_freq,
                target_freq=target_freq,
                source_rbound=source_rbound,
                target_rbound=target_rbound,
                nan_value=nan_value,
                ffill=ffill,
            )
        return out

    if not nb_enabled:
        return _latest_at_index_nb(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _latest_at_index_nb