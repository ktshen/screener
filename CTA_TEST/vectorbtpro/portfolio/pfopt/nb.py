# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio optimization."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.portfolio.enums import Direction, alloc_point_dt, alloc_range_dt

__all__ = []


@register_jitted(cache=True)
def get_alloc_points_nb(
    filled_allocations: tp.Array2d,
    valid_only: bool = True,
    nonzero_only: bool = True,
    unique_only: bool = True,
) -> tp.Array1d:
    """Get allocation points from filled allocations.

    If `valid_only` is True, does not register a new allocation when all points are NaN.v
    If `nonzero_only` is True, does not register a new allocation when all points are zero.
    If `unique_only` is True, does not register a new allocation when it's the same as the last one."""
    out = np.empty(len(filled_allocations), dtype=np.int_)
    k = 0
    for i in range(filled_allocations.shape[0]):
        all_nan = True
        all_zeros = True
        all_same = True
        for col in range(filled_allocations.shape[1]):
            if not np.isnan(filled_allocations[i, col]):
                all_nan = False
            if abs(filled_allocations[i, col]) > 0:
                all_zeros = False
            if k == 0 or (k > 0 and filled_allocations[i, col] != filled_allocations[out[k - 1], col]):
                all_same = False
        if valid_only and all_nan:
            continue
        if nonzero_only and all_zeros:
            continue
        if unique_only and all_same:
            continue
        out[k] = i
        k += 1
    return out[:k]


@register_chunkable(
    size=ch.ArraySizer(arg_query="range_starts", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        range_starts=ch.ArraySlicer(axis=0),
        range_ends=ch.ArraySlicer(axis=0),
        optimize_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def optimize_meta_nb(
    n_cols: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    optimize_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Optimize by reducing each index range.

    `reduce_func_nb` must take the range index, the range start, the range end, and `*args`.
    Must return a 1-dim array with the same size as `n_cols`."""
    out = np.empty((range_starts.shape[0], n_cols), dtype=np.float_)
    for i in prange(len(range_starts)):
        out[i] = optimize_func_nb(i, range_starts[i], range_ends[i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="index_points", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        index_points=ch.ArraySlicer(axis=0),
        allocate_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def allocate_meta_nb(
    n_cols: int,
    index_points: tp.Array1d,
    allocate_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Allocate by mapping each index point.

    `map_func_nb` must take the point index, the index point, and `*args`.
    Must return a 1-dim array with the same size as `n_cols`."""
    out = np.empty((index_points.shape[0], n_cols), dtype=np.float_)
    for i in prange(len(index_points)):
        out[i] = allocate_func_nb(i, index_points[i], *args)
    return out


@register_jitted(cache=True)
def pick_idx_allocate_func_nb(i: int, index_point: int, allocations: tp.Array2d) -> tp.Array1d:
    """Pick the allocation at an absolute position in an array."""
    return allocations[i]


@register_jitted(cache=True)
def pick_point_allocate_func_nb(i: int, index_point: int, allocations: tp.Array2d) -> tp.Array1d:
    """Pick the allocation at an index point in an array."""
    return allocations[index_point]


@register_jitted(cache=True)
def random_allocate_func_nb(
    i: int,
    index_point: int,
    n_cols: int,
    direction: int = Direction.LongOnly,
    n: tp.Optional[int] = None,
) -> tp.Array1d:
    """Generate a random allocation."""
    weights = np.full(n_cols, np.nan, dtype=np.float_)
    pos_sum = 0
    neg_sum = 0
    if n is None:
        for c in range(n_cols):
            w = np.random.uniform(0, 1)
            if direction == Direction.ShortOnly:
                w = -w
            elif direction == Direction.Both:
                if np.random.randint(0, 2) == 0:
                    w = -w
            if w >= 0:
                pos_sum += w
            else:
                neg_sum += abs(w)
            weights[c] = w
    else:
        rand_indices = np.random.choice(n_cols, size=n, replace=False)
        for k in range(len(rand_indices)):
            w = np.random.uniform(0, 1)
            if direction == Direction.ShortOnly:
                w = -w
            elif direction == Direction.Both:
                if np.random.randint(0, 2) == 0:
                    w = -w
            if w >= 0:
                pos_sum += w
            else:
                neg_sum += abs(w)
            weights[rand_indices[k]] = w
    for c in range(n_cols):
        if not np.isnan(weights[c]):
            if weights[c] >= 0:
                if pos_sum > 0:
                    weights[c] = weights[c] / pos_sum
            else:
                if neg_sum > 0:
                    weights[c] = weights[c] / neg_sum
        else:
            weights[c] = 0.0
    return weights


@register_jitted(cache=True)
def prepare_alloc_points_nb(
    index_points: tp.Array1d,
    allocations: tp.Array2d,
    group: int,
) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
    """Prepare allocation points."""
    alloc_points = np.empty_like(index_points, dtype=alloc_point_dt)
    new_allocations = np.empty_like(allocations)
    k = 0
    for i in range(allocations.shape[0]):
        all_nan = True
        for col in range(allocations.shape[1]):
            if not np.isnan(allocations[i, col]):
                all_nan = False
                break
        if all_nan:
            continue
        if k > 0 and alloc_points["alloc_idx"][k - 1] == index_points[i]:
            new_allocations[k - 1] = allocations[i]
        else:
            alloc_points["id"][k] = k
            alloc_points["col"][k] = group
            alloc_points["alloc_idx"][k] = index_points[i]
            new_allocations[k] = allocations[i]
            k += 1
    return alloc_points[:k], new_allocations[:k]


@register_jitted(cache=True)
def prepare_alloc_ranges_nb(
    start_idx: tp.Array1d,
    end_idx: tp.Array1d,
    alloc_idx: tp.Array1d,
    status: tp.Array1d,
    allocations: tp.Array2d,
    group: int,
) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
    """Prepare allocation ranges."""
    alloc_ranges = np.empty_like(alloc_idx, dtype=alloc_range_dt)
    new_allocations = np.empty_like(allocations)
    k = 0
    for i in range(allocations.shape[0]):
        all_nan = True
        for col in range(allocations.shape[1]):
            if not np.isnan(allocations[i, col]):
                all_nan = False
                break
        if all_nan:
            continue
        if k > 0 and alloc_ranges["alloc_idx"][k - 1] == alloc_idx[i]:
            new_allocations[k - 1] = allocations[i]
        else:
            alloc_ranges["id"][k] = k
            alloc_ranges["col"][k] = group
            alloc_ranges["start_idx"][k] = start_idx[i]
            alloc_ranges["end_idx"][k] = end_idx[i]
            alloc_ranges["alloc_idx"][k] = alloc_idx[i]
            alloc_ranges["status"][k] = status[i]
            new_allocations[k] = allocations[i]
            k += 1
    return alloc_ranges[:k], new_allocations[:k]
