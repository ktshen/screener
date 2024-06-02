# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for splitting."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


@register_jitted(cache=True, tags={"can_parallel"})
def split_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for splits."""
    out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=np.int_)
    temp_mask = np.empty((mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
    for i in range(mask_arr.shape[0]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[i, :, m].any():
                temp_mask[i, m] = True
            else:
                temp_mask[i, m] = False
    for i1 in prange(mask_arr.shape[0]):
        for i2 in range(mask_arr.shape[0]):
            intersection = (temp_mask[i1] & temp_mask[i2]).sum()
            out[i1, i2] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_split_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for splits."""
    out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=np.float_)
    temp_mask = np.empty((mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
    for i in range(mask_arr.shape[0]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[i, :, m].any():
                temp_mask[i, m] = True
            else:
                temp_mask[i, m] = False
    for i1 in prange(mask_arr.shape[0]):
        for i2 in range(mask_arr.shape[0]):
            intersection = (temp_mask[i1] & temp_mask[i2]).sum()
            union = (temp_mask[i1] | temp_mask[i2]).sum()
            out[i1, i2] = intersection / union
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def set_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for sets."""
    out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=np.int_)
    temp_mask = np.empty((mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
    for j in range(mask_arr.shape[1]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[:, j, m].any():
                temp_mask[j, m] = True
            else:
                temp_mask[j, m] = False
    for j1 in prange(mask_arr.shape[1]):
        for j2 in range(mask_arr.shape[1]):
            intersection = (temp_mask[j1] & temp_mask[j2]).sum()
            out[j1, j2] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_set_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for sets."""
    out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=np.float_)
    temp_mask = np.empty((mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
    for j in range(mask_arr.shape[1]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[:, j, m].any():
                temp_mask[j, m] = True
            else:
                temp_mask[j, m] = False
    for j1 in prange(mask_arr.shape[1]):
        for j2 in range(mask_arr.shape[1]):
            intersection = (temp_mask[j1] & temp_mask[j2]).sum()
            union = (temp_mask[j1] | temp_mask[j2]).sum()
            out[j1, j2] = intersection / union
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def range_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for ranges."""
    out = np.empty((mask_arr.shape[0] * mask_arr.shape[1], mask_arr.shape[0] * mask_arr.shape[1]), dtype=np.int_)
    for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
        i1 = k // mask_arr.shape[1]
        j1 = k % mask_arr.shape[1]
        for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
            i2 = l // mask_arr.shape[1]
            j2 = l % mask_arr.shape[1]
            intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
            out[k, l] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_range_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for ranges."""
    out = np.empty((mask_arr.shape[0] * mask_arr.shape[1], mask_arr.shape[0] * mask_arr.shape[1]), dtype=np.float_)
    for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
        i1 = k // mask_arr.shape[1]
        j1 = k % mask_arr.shape[1]
        for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
            i2 = l // mask_arr.shape[1]
            j2 = l % mask_arr.shape[1]
            intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
            union = (mask_arr[i1, j1] | mask_arr[i2, j2]).sum()
            out[k, l] = intersection / union
    return out


@register_jitted(cache=True)
def split_range_by_gap_nb(range_: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Split a range with gaps into start and end indices."""
    if len(range_) == 0:
        raise ValueError("Range is empty")
    start_idxs_out = np.empty(len(range_), dtype=np.int_)
    stop_idxs_out = np.empty(len(range_), dtype=np.int_)
    start_idxs_out[0] = 0
    k = 0
    for i in range(1, len(range_)):
        if range_[i] - range_[i - 1] != 1:
            stop_idxs_out[k] = i
            k += 1
            start_idxs_out[k] = i
    stop_idxs_out[k] = len(range_)
    return start_idxs_out[:k + 1], stop_idxs_out[:k + 1]
