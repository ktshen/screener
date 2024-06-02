# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


def is_sorted(arr: tp.Array1d) -> bool:
    """Checks if array is sorted."""
    return np.all(arr[:-1] <= arr[1:])


@register_jitted(cache=True)
def is_sorted_nb(arr: tp.Array1d) -> bool:
    """Numba-compiled version of `is_sorted`."""
    for i in range(arr.size - 1):
        if arr[i + 1] < arr[i]:
            return False
    return True


def is_range(arr: tp.Array1d) -> bool:
    """Checks if array is arr range."""
    return np.all(np.diff(arr) == 1)


@register_jitted(cache=True)
def is_range_nb(arr: tp.Array1d) -> bool:
    """Numba-compiled version of `is_range`."""
    for i in range(arr.size):
        if arr[i] != arr[0] + i:
            return False
    return True


@register_jitted(cache=True)
def insert_argsort_nb(A: tp.Array1d, I: tp.Array1d) -> None:
    """Perform argsort using insertion sort.

    In-memory and without recursion -> very fast for smaller arrays."""
    for j in range(1, len(A)):
        A_j = A[j]
        I_j = I[j]
        i = j - 1
        while i >= 0 and (A[i] > A_j or np.isnan(A[i])):
            A[i + 1] = A[i]
            I[i + 1] = I[i]
            i = i - 1
        A[i + 1] = A_j
        I[i + 1] = I_j


def get_ranges_arr(starts: tp.ArrayLike, ends: tp.ArrayLike) -> tp.Array1d:
    """Build array from start and end indices.

    Based on https://stackoverflow.com/a/37626057"""
    starts_arr = np.asarray(starts)
    if starts_arr.ndim == 0:
        starts_arr = np.array([starts_arr])
    ends_arr = np.asarray(ends)
    if ends_arr.ndim == 0:
        ends_arr = np.array([ends_arr])
    starts_arr, end = np.broadcast_arrays(starts_arr, ends_arr)
    counts = ends_arr - starts_arr
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1], dtype=int)
    id_arr[0] = starts_arr[0]
    id_arr[counts_csum[:-1]] = starts_arr[1:] - ends_arr[:-1] + 1
    return id_arr.cumsum()


@register_jitted(cache=True)
def uniform_summing_to_one_nb(n: int) -> tp.Array1d:
    """Generate random floats summing to one.

    See # https://stackoverflow.com/a/2640067/8141780"""
    rand_floats = np.empty(n + 1, dtype=np.float_)
    rand_floats[0] = 0.0
    rand_floats[1] = 1.0
    rand_floats[2:] = np.random.uniform(0, 1, n - 1)
    rand_floats = np.sort(rand_floats)
    rand_floats = rand_floats[1:] - rand_floats[:-1]
    return rand_floats


def rescale(
    arr: tp.MaybeArray,
    from_range: tp.Tuple[float, float],
    to_range: tp.Tuple[float, float],
) -> tp.MaybeArray:
    """Renormalize `arr` from one range to another."""
    from_min, from_max = from_range
    to_min, to_max = to_range
    from_delta = from_max - from_min
    to_delta = to_max - to_min
    return (to_delta * (arr - from_min) / from_delta) + to_min


@register_jitted(cache=True)
def rescale_nb(
    arr: tp.MaybeArray,
    from_range: tp.Tuple[float, float],
    to_range: tp.Tuple[float, float],
) -> tp.MaybeArray:
    """Numba-compiled version of `rescale`."""
    from_min, from_max = from_range
    to_min, to_max = to_range
    from_delta = from_max - from_min
    to_delta = to_max - to_min
    return (to_delta * (arr - from_min) / from_delta) + to_min


def min_rel_rescale(arr: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    """Rescale elements in `arr` relatively to minimum."""
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max - a_min == 0:
        return np.full(arr.shape, to_range[0])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[0], to_range[0] * from_range_ratio)
    return rescale(arr, from_range, to_range)


def max_rel_rescale(arr: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    """Rescale elements in `arr` relatively to maximum."""
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max - a_min == 0:
        return np.full(arr.shape, to_range[1])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[1] / from_range_ratio, to_range[1])
    return rescale(arr, from_range, to_range)


@register_jitted(cache=True)
def rescale_float_to_int_nb(floats: tp.Array, int_range: tp.Tuple[float, float], total: float) -> tp.Array:
    """Rescale a float array into an int array."""
    ints = np.floor(rescale_nb(floats, (0.0, 1.0), int_range))
    leftover = int(total - ints.sum())
    for i in range(leftover):
        ints[np.random.choice(len(ints))] += 1
    return ints


@register_jitted(cache=True)
def int_digit_count_nb(number: int) -> int:
    """Get the digit count in a number."""
    out = 0
    while number != 0:
        number //= 10
        out += 1
    return out


@register_jitted(cache=True)
def hash_int_rows_nb(arr: tp.Array2d) -> tp.Array1d:
    """Hash rows in a 2-dim array.

    First digits of each hash correspond to the left-most column, the last digits to the right-most column.
    Thus, the resulting hashes are not suitable for sorting by value."""
    out = np.full(arr.shape[0], 0, dtype=np.int_)
    prefix = 1
    for col in range(arr.shape[1]):
        vmax = np.nan
        for i in range(arr.shape[0]):
            if np.isnan(vmax) or arr[i, col] > vmax:
                vmax = arr[i, col]
            out[i] += arr[i, col] * prefix
        prefix *= 10 ** int_digit_count_nb(vmax)
    return out


@register_jitted(cache=True)
def index_repeating_rows_nb(arr: tp.Array2d) -> tp.Array1d:
    """Index repeating rows using monotonically increasing numbers."""
    out = np.empty(arr.shape[0], dtype=np.int_)
    temp = np.copy(arr[0])

    k = 0
    for i in range(arr.shape[0]):
        new_unique = False
        for col in range(arr.shape[1]):
            if arr[i, col] != temp[col]:
                if not new_unique:
                    k += 1
                    new_unique = True
                temp[col] = arr[i, col]
        out[i] = k
    return out


def build_nan_mask(*arrs: tp.Array) -> tp.Optional[tp.Array]:
    """Build NaN mask out of one to multiple arrays via OR rule."""
    nan_mask = None
    for arr in arrs:
        if nan_mask is None:
            nan_mask = np.isnan(arr)
        else:
            nan_mask |= nan_mask
    return nan_mask


def squeeze_nan(*arrs: tp.Array, nan_mask: tp.Optional[tp.Array1d] = None) -> tp.Tuple[tp.Array, ...]:
    """Squeeze NaN values using a mask."""
    if nan_mask is None or not np.any(nan_mask, axis=-1):
        return arrs

    new_arrs = ()
    for arr in arrs:
        new_arrs += (arr[~nan_mask],)
    return new_arrs


def unsqueeze_nan(*arrs: tp.Array, nan_mask: tp.Optional[tp.Array1d] = None) -> tp.Tuple[tp.Array, ...]:
    """Un-squeeze NaN values using a mask."""
    if nan_mask is None or not np.any(nan_mask, axis=-1):
        return arrs

    new_arrs = ()
    for arr in arrs:
        new_arr = np.full(len(nan_mask), np.nan, dtype=np.float_)
        new_arr[~nan_mask] = arr
        new_arrs += (new_arr,)
    return new_arrs


def cast_to_min_precision(
    arr: tp.Array,
    min_precision: tp.Union[int, str],
    float_only: bool = True,
) -> tp.Array:
    """Cast an array to a minimum integer/floating precision.

    Argument must be either an integer denoting the number of bits,
    or one of 'half', 'single', and 'double'."""
    if min_precision is None:
        return arr
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return arr
    if float_only and np.issubdtype(arr.dtype, np.integer):
        return arr
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        if isinstance(min_precision, str):
            if min_precision == "half":
                min_precision = 16
            elif min_precision == "single":
                min_precision = 32
            elif min_precision == "double":
                min_precision = 64
            else:
                raise ValueError(f"Only 'half', 'single', and 'double' max precisions are supported")
        if isinstance(min_precision, int):
            if np.issubdtype(arr.dtype, np.integer):
                prefix = "int"
            else:
                prefix = "float"
            target_dtype = np.dtype(prefix + str(min_precision))
        else:
            raise TypeError("Minimum precision must be either integer or string")
        if arr.dtype < target_dtype:
            return arr.astype(target_dtype)
    return arr


def cast_to_max_precision(
    arr: tp.Array,
    max_precision: tp.Union[int, str],
    float_only: bool = True,
    check_bounds: bool = True,
    strict: bool = True,
) -> tp.Array:
    """Cast an array to a maximum integer/floating precision.

    Argument must be either an integer denoting the number of bits,
    or one of 'half', 'single', and 'double'."""
    if max_precision is None:
        return arr
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return arr
    if float_only and np.issubdtype(arr.dtype, np.integer):
        return arr
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        if isinstance(max_precision, str):
            if max_precision == "half":
                max_precision = 16
            elif max_precision == "single":
                max_precision = 32
            elif max_precision == "double":
                max_precision = 64
            else:
                raise ValueError(f"Only 'half', 'single', and 'double' max precisions are supported")
        if isinstance(max_precision, int):
            if np.issubdtype(arr.dtype, np.integer):
                prefix = "int"
                dtype_info = np.iinfo
            else:
                prefix = "float"
                dtype_info = np.finfo
            target_dtype = np.dtype(prefix + str(max_precision))
        else:
            raise TypeError("Maximum precision must be either integer or string")
        if arr.dtype > target_dtype:
            if check_bounds:
                min_overflow = np.min(arr) < dtype_info(target_dtype).min
                max_overflow = np.max(arr) > dtype_info(target_dtype).max
                if min_overflow or max_overflow:
                    if strict:
                        raise ValueError(f"Cannot lower dtype to {target_dtype}: values out of bounds")
                    return arr
            return arr.astype(target_dtype)
    return arr
