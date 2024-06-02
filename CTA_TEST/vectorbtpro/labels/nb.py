# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for label generation.

!!! note
    Set `wait` to 1 to exclude the current value from calculation of future values.

!!! warning
    Do not attempt to use these functions for building features as they
    may introduce look-ahead bias to your model."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.labels.enums import TrendLabelMode
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), window=None, wtype=None, wait=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def future_mean_nb(close: tp.Array2d, window: int, wtype: int, wait: int = 1, adjust: bool = False) -> tp.Array2d:
    """Get the mean of the next period."""
    out = generic_nb.ma_nb(close[::-1], window, wtype=wtype, minp=window, adjust=adjust)[::-1]
    if wait > 0:
        return generic_nb.bshift_nb(out, wait)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), window=None, wtype=None, wait=None, adjust=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def future_std_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    wait: int = 1,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """Get the standard deviation of the next period."""
    out = generic_nb.msd_nb(close[::-1], window, wtype=wtype, minp=window, adjust=adjust, ddof=ddof)[::-1]
    if wait > 0:
        return generic_nb.bshift_nb(out, wait)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), window=None, wait=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def future_min_nb(close: tp.Array2d, window: int, wait: int = 1) -> tp.Array2d:
    """Get the minimum of the next period."""
    out = generic_nb.rolling_min_nb(close[::-1], window, minp=window)[::-1]
    if wait > 0:
        return generic_nb.bshift_nb(out, wait)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), window=None, wait=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def future_max_nb(close: tp.Array2d, window: int, wait: int = 1) -> tp.Array2d:
    """Get the maximum of the next period."""
    out = generic_nb.rolling_max_nb(close[::-1], window, minp=window)[::-1]
    if wait > 0:
        return generic_nb.bshift_nb(out, wait)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), n=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def fixed_labels_nb(close: tp.Array2d, n: int) -> tp.Array2d:
    """Get percentage change from the current value to a future value."""
    return (generic_nb.bshift_nb(close, n) - close) / close


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), window=None, wtype=None, wait=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def mean_labels_nb(close: tp.Array2d, window: int, wtype: int, wait: int = 1, adjust: bool = False) -> tp.Array2d:
    """Get the percentage change from the current value to the average of the next period."""
    return (future_mean_nb(close, window, wtype, wait, adjust) - close) / close


@register_jitted(cache=True)
def get_symmetric_up_th_nb(down_th: tp.FlexArray2d) -> tp.FlexArray2d:
    """Compute the positive return that is symmetric to a negative one.

    For example, 50% down requires 100% to go up to the initial level."""
    return down_th / (1 - down_th)


@register_jitted(cache=True)
def get_symmetric_down_th_nb(up_th: tp.FlexArray2d) -> tp.FlexArray2d:
    """Compute the negative return that is symmetric to a positive one."""
    return up_th / (1 + up_th)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def local_extrema_nb(
    close: tp.Array2d,
    up_th: tp.FlexArray2d,
    down_th: tp.FlexArray2d,
) -> tp.Array2d:
    """Get array of local extrema denoted by 1 (peak) or -1 (trough), otherwise 0.

    Two adjacent peak and trough points should exceed the given threshold parameters.

    If any threshold is given element-wise, it will be applied per new/updated extremum.

    Inspired by https://www.mdpi.com/1099-4300/22/10/1162/pdf"""
    up_th = np.asarray(up_th)
    down_th = np.asarray(down_th)
    out = np.full(close.shape, 0, dtype=np.int_)

    for col in prange(close.shape[1]):
        prev_i = 0
        direction = 0

        for i in range(1, close.shape[0]):
            _up_th = abs(flex_select_nb(up_th, prev_i, col))
            _down_th = abs(flex_select_nb(down_th, prev_i, col))
            if _up_th == 0:
                raise ValueError("Positive threshold cannot be 0")
            if _down_th == 0:
                raise ValueError("Negative threshold cannot be 0")

            if direction == 1:
                # Find next high while updating current lows
                if close[i, col] < close[prev_i, col]:
                    prev_i = i
                elif close[i, col] >= close[prev_i, col] * (1 + _up_th):
                    out[prev_i, col] = -1
                    prev_i = i
                    direction = -1
            elif direction == -1:
                # Find next low while updating current highs
                if close[i, col] > close[prev_i, col]:
                    prev_i = i
                elif close[i, col] <= close[prev_i, col] * (1 - _down_th):
                    out[prev_i, col] = 1
                    prev_i = i
                    direction = 1
            else:
                # Find first high/low
                if close[i, col] >= close[prev_i, col] * (1 + _up_th):
                    out[prev_i, col] = -1
                    prev_i = i
                    direction = -1
                elif close[i, col] <= close[prev_i, col] * (1 - _down_th):
                    out[prev_i, col] = 1
                    prev_i = i
                    direction = 1

            if i == close.shape[0] - 1:
                # Find last high/low
                if direction != 0:
                    out[prev_i, col] = -direction
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), local_extrema=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bn_trend_labels_nb(close: tp.Array2d, local_extrema: tp.Array2d) -> tp.Array2d:
    """Return 0 for H-L and 1 for L-H."""
    out = np.full_like(close, np.nan, dtype=np.float_)

    for col in prange(close.shape[1]):
        idxs = np.flatnonzero(local_extrema[:, col])
        if idxs.shape[0] == 0:
            continue

        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]
            next_i = idxs[k]

            if close[next_i, col] > close[prev_i, col]:
                out[prev_i:next_i, col] = 1
            else:
                out[prev_i:next_i, col] = 0

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), local_extrema=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bn_cont_trend_labels_nb(close: tp.Array2d, local_extrema: tp.Array2d) -> tp.Array2d:
    """Normalize each range between two extrema between 0 (will go up) and 1 (will go down)."""
    out = np.full_like(close, np.nan, dtype=np.float_)

    for col in prange(close.shape[1]):
        idxs = np.flatnonzero(local_extrema[:, col])
        if idxs.shape[0] == 0:
            continue

        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]
            next_i = idxs[k]

            _min = np.min(close[prev_i : next_i + 1, col])
            _max = np.max(close[prev_i : next_i + 1, col])
            out[prev_i:next_i, col] = 1 - (close[prev_i:next_i, col] - _min) / (_max - _min)

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        local_extrema=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bn_cont_sat_trend_labels_nb(
    close: tp.Array2d,
    local_extrema: tp.Array2d,
    up_th: tp.FlexArray2d,
    down_th: tp.FlexArray2d,
) -> tp.Array2d:
    """Similar to `bn_cont_trend_labels_nb` but sets each close value to 0 or 1
    if the percentage change to the next extremum exceeds the threshold set for this range.
    """
    up_th = np.asarray(up_th)
    down_th = np.asarray(down_th)
    out = np.full_like(close, np.nan, dtype=np.float_)

    for col in prange(close.shape[1]):
        idxs = np.flatnonzero(local_extrema[:, col])
        if idxs.shape[0] == 0:
            continue

        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]
            next_i = idxs[k]

            _up_th = abs(flex_select_nb(up_th, prev_i, col))
            _down_th = abs(flex_select_nb(down_th, prev_i, col))
            if _up_th == 0:
                raise ValueError("Positive threshold cannot be 0")
            if _down_th == 0:
                raise ValueError("Negative threshold cannot be 0")
            _min = np.min(close[prev_i : next_i + 1, col])
            _max = np.max(close[prev_i : next_i + 1, col])

            for i in range(prev_i, next_i):
                if close[next_i, col] > close[prev_i, col]:
                    _start = _max / (1 + _up_th)
                    _end = _min * (1 + _up_th)
                    if _max >= _end and close[i, col] <= _start:
                        out[i, col] = 1
                    else:
                        out[i, col] = 1 - (close[i, col] - _start) / (_max - _start)
                else:
                    _start = _min / (1 - _down_th)
                    _end = _max * (1 - _down_th)
                    if _min <= _end and close[i, col] >= _start:
                        out[i, col] = 0
                    else:
                        out[i, col] = 1 - (close[i, col] - _min) / (_start - _min)

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), local_extrema=ch.ArraySlicer(axis=1), normalize=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pct_trend_labels_nb(close: tp.Array2d, local_extrema: tp.Array2d, normalize: bool) -> tp.Array2d:
    """Compute the percentage change of the current value to the next extremum."""
    out = np.full_like(close, np.nan, dtype=np.float_)

    for col in prange(close.shape[1]):
        idxs = np.flatnonzero(local_extrema[:, col])
        if idxs.shape[0] == 0:
            continue

        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]
            next_i = idxs[k]

            for i in range(prev_i, next_i):
                if close[next_i, col] > close[prev_i, col] and normalize:
                    out[i, col] = (close[next_i, col] - close[i, col]) / close[next_i, col]
                else:
                    out[i, col] = (close[next_i, col] - close[i, col]) / close[i, col]

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def trend_labels_nb(
    close: tp.Array2d,
    up_th: tp.FlexArray2d,
    down_th: tp.FlexArray2d,
    mode: int,
) -> tp.Array2d:
    """Apply a trend labeling function based on `TrendLabelMode`."""
    local_extrema = local_extrema_nb(close, up_th, down_th)
    if mode == TrendLabelMode.Binary:
        return bn_trend_labels_nb(close, local_extrema)
    if mode == TrendLabelMode.BinaryCont:
        return bn_cont_trend_labels_nb(close, local_extrema)
    if mode == TrendLabelMode.BinaryContSat:
        return bn_cont_sat_trend_labels_nb(close, local_extrema, up_th, down_th)
    if mode == TrendLabelMode.PctChange:
        return pct_trend_labels_nb(close, local_extrema, False)
    if mode == TrendLabelMode.PctChangeNorm:
        return pct_trend_labels_nb(close, local_extrema, True)
    raise ValueError("Invalid trend mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=None,
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def breakout_labels_nb(
    close: tp.Array2d,
    window: int,
    up_th: tp.FlexArray2d,
    down_th: tp.FlexArray2d,
    wait: int = 1,
) -> tp.Array2d:
    """For each value, return 1 if any value in the next period is greater than the
    positive threshold (in %), -1 if less than the negative threshold, and 0 otherwise.

    First hit wins."""
    up_th = np.asarray(up_th)
    down_th = np.asarray(down_th)
    out = np.full_like(close, 0, dtype=np.float_)

    for col in prange(close.shape[1]):
        for i in range(close.shape[0]):
            _up_th = abs(flex_select_nb(up_th, i, col))
            _down_th = abs(flex_select_nb(down_th, i, col))

            for j in range(i + wait, min(i + window + wait, close.shape[0])):
                if _up_th > 0 and close[j, col] >= close[i, col] * (1 + _up_th):
                    out[i, col] = 1
                    break
                if _down_th > 0 and close[j, col] <= close[i, col] * (1 - _down_th):
                    out[i, col] = -1
                    break

    return out
