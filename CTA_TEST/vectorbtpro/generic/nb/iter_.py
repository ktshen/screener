# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for iterative use."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base.flex_indexing import flex_select_nb


@register_jitted(cache=True)
def iter_above_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Check whether `arr1` is above `arr2` at specific row and column."""
    if i < 0:
        return False
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now > arr2_now


@register_jitted(cache=True)
def iter_below_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Check whether `arr1` is below `arr2` at specific row and column."""
    if i < 0:
        return False
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now < arr2_now


@register_jitted(cache=True)
def iter_crossed_above_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Check whether `arr1` crossed above `arr2` at specific row and column."""
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_nb(arr1, i - 1, col)
    arr2_prev = flex_select_nb(arr2, i - 1, col)
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev < arr2_prev and arr1_now > arr2_now


@register_jitted(cache=True)
def iter_crossed_below_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Check whether `arr1` crossed below `arr2` at specific row and column."""
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_nb(arr1, i - 1, col)
    arr2_prev = flex_select_nb(arr2, i - 1, col)
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev > arr2_prev and arr1_now < arr2_now
