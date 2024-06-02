# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for custom indicators.

Provides an arsenal of Numba-compiled functions that are used by indicator
classes. These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.indicators.enums import Pivot, SuperTrendAIS, SuperTrendAOS
from vectorbtpro.utils import chunking as ch

__all__ = []


@register_jitted(cache=True)
def ma_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Cache function for `vectorbtpro.indicators.custom.MA`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.ma_nb(close, windows[i], wtypes[i], adjust=adjust, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def ma_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MA`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return cache_dict[h]
    return generic_nb.ma_nb(close, window, wtype, adjust=adjust, minp=minp)


@register_jitted(cache=True)
def msd_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Cache function for `vectorbtpro.indicators.custom.MSD`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.msd_nb(close, windows[i], wtypes[i], adjust=adjust, ddof=ddof, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def msd_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MSD`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return cache_dict[h]
    return generic_nb.msd_nb(close, window, wtype, adjust=adjust, ddof=ddof, minp=minp)


@register_jitted(cache=True)
def bbands_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    alphas: tp.List[float],
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Dict[int, tp.Array2d]], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Cache function for `vectorbtpro.indicators.custom.BBANDS`."""
    if per_column:
        return None, None

    ma_cache_dict = ma_cache_nb(close, windows, wtypes, adjust=adjust, minp=minp)
    msd_cache_dict = msd_cache_nb(close, windows, wtypes, adjust=adjust, ddof=ddof, minp=minp)
    return ma_cache_dict, msd_cache_dict


@register_jitted(cache=True)
def bbands_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    alpha: float,
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    ma_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
    msd_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.BBANDS`."""
    if ma_cache_dict is not None and msd_cache_dict is not None:
        h = hash((window, wtype))
        ma = np.copy(ma_cache_dict[h])
        msd = np.copy(msd_cache_dict[h])
    else:
        ma = generic_nb.ma_nb(close, window, wtype, adjust=adjust, minp=minp)
        msd = generic_nb.msd_nb(close, window, wtype, adjust=adjust, ddof=ddof, minp=minp)
    return ma, ma + alpha * msd, ma - alpha * msd


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rsi_up_down_nb(close: tp.Array2d) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Calculate the `up` and `down` arrays for RSI."""
    up = np.empty_like(close, dtype=np.float_)
    down = np.empty_like(close, dtype=np.float_)
    for col in prange(close.shape[1]):
        for i in range(close.shape[0]):
            if i == 0:
                up[i, col] = np.nan
                down[i, col] = np.nan
            else:
                delta = close[i, col] - close[i - 1, col]
                if delta < 0:
                    up[i, col] = 0.0
                    down[i, col] = abs(delta)
                else:
                    up[i, col] = delta
                    down[i, col] = 0.0
    return up, down


@register_jitted(cache=True)
def rsi_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Cache function for `vectorbtpro.indicators.custom.RSI`."""
    up, down = rsi_up_down_nb(close)
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            roll_up = generic_nb.ma_nb(up, windows[i], wtypes[i], adjust=adjust, minp=minp)
            roll_down = generic_nb.ma_nb(down, windows[i], wtypes[i], adjust=adjust, minp=minp)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@register_jitted(cache=True)
def rsi_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.RSI`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        roll_up, roll_down = cache_dict[h]
    else:
        up, down = rsi_up_down_nb(close)
        roll_up = generic_nb.ma_nb(up, window, wtype, adjust=adjust, minp=minp)
        roll_down = generic_nb.ma_nb(down, window, wtype, adjust=adjust, minp=minp)
    return 100 * roll_up / (roll_up + roll_down)


@register_jitted(cache=True)
def stoch_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    fast_k_windows: tp.List[int],
    slow_k_windows: tp.List[int],
    slow_d_windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Cache function for `vectorbtpro.indicators.custom.STOCH`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(fast_k_windows)):
        h = hash(fast_k_windows[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low, fast_k_windows[i], minp=minp)
            roll_max = generic_nb.rolling_max_nb(high, fast_k_windows[i], minp=minp)
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@register_jitted(cache=True)
def stoch_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    fast_k_window: int,
    slow_k_window: int,
    slow_d_window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.STOCH`."""
    if cache_dict is not None:
        h = hash(fast_k_window)
        roll_min, roll_max = cache_dict[h]
    else:
        roll_min = generic_nb.rolling_min_nb(low, fast_k_window, minp=minp)
        roll_max = generic_nb.rolling_max_nb(high, fast_k_window, minp=minp)
    fast_k = 100 * (close - roll_min) / (roll_max - roll_min)
    slow_k = generic_nb.ma_nb(fast_k, slow_k_window, wtype, adjust=adjust, minp=minp)
    slow_d = generic_nb.ma_nb(slow_k, slow_d_window, wtype, adjust=adjust, minp=minp)
    return fast_k, slow_k, slow_d


@register_jitted(cache=True)
def macd_cache_nb(
    close: tp.Array2d,
    fast_windows: tp.List[int],
    slow_windows: tp.List[int],
    signal_windows: tp.List[int],
    macd_wtypes: tp.List[int],
    signal_wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Cache function for `vectorbtpro.indicators.custom.MACD`."""
    if per_column:
        return None

    windows = fast_windows.copy()
    windows.extend(slow_windows)
    wtypes = macd_wtypes.copy()
    wtypes.extend(macd_wtypes)
    return ma_cache_nb(close, windows, wtypes, adjust=adjust, minp=minp)


@register_jitted(cache=True)
def macd_nb(
    close: tp.Array2d,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    macd_wtype: int,
    signal_wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.MACD`."""
    if cache_dict is not None:
        fast_h = hash((fast_window, macd_wtype))
        slow_h = hash((slow_window, macd_wtype))
        fast_ma = cache_dict[fast_h]
        slow_ma = cache_dict[slow_h]
    else:
        fast_ma = generic_nb.ma_nb(close, fast_window, macd_wtype, adjust=adjust, minp=minp)
        slow_ma = generic_nb.ma_nb(close, slow_window, macd_wtype, adjust=adjust, minp=minp)
    macd_ts = fast_ma - slow_ma
    signal_ts = generic_nb.ma_nb(macd_ts, signal_window, signal_wtype, adjust=adjust, minp=minp)
    return macd_ts, signal_ts


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def tr_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Calculate true range."""
    tr = np.empty(high.shape, dtype=np.float_)
    for col in prange(high.shape[1]):
        for i in range(high.shape[0]):
            prev_close = close[i - 1, col] if i > 0 else np.nan
            tr1 = high[i, col] - low[i, col]
            tr2 = abs(high[i, col] - prev_close)
            tr3 = abs(low[i, col] - prev_close)
            tr[i, col] = max(tr1, tr2, tr3)
    return tr


@register_jitted(cache=True)
def atr_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Array2d], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Cache function for `vectorbtpro.indicators.custom.ATR`."""
    tr = tr_nb(high, low, close)
    if per_column:
        return None, None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.ma_nb(tr, windows[i], wtypes[i], adjust=adjust, minp=minp)
    return tr, cache_dict


@register_jitted(cache=True)
def atr_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    tr: tp.Optional[tp.Array2d] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.ATR`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return tr, cache_dict[h]
    if tr is None:
        _tr = tr_nb(high, low, close)
    else:
        _tr = tr
    return _tr, generic_nb.ma_nb(_tr, window, wtype, adjust=adjust, minp=minp)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        volume=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def obv_nb(close: tp.Array2d, volume: tp.Array2d) -> tp.Array2d:
    """Custom calculation function for `vectorbtpro.indicators.custom.OBV`."""
    obv = np.empty_like(close, dtype=np.float_)
    for col in prange(close.shape[1]):
        cumsum = 0.0
        for i in range(close.shape[0]):
            prev_close = close[i - 1, col] if i > 0 else np.nan
            if close[i, col] < prev_close:
                value = -volume[i, col]
            else:
                value = volume[i, col]
            if not np.isnan(value):
                cumsum += value
            obv[i, col] = cumsum
    return obv


@register_jitted(cache=True)
def ols_cache_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    windows: tp.List[int],
    with_zscore: bool = True,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]]]:
    """Cache function for `vectorbtpro.indicators.custom.OLS`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash(windows[i])
        if h not in cache_dict:
            cache_dict[h] = ols_nb(x, y, windows[i], with_zscore=with_zscore, ddof=ddof, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def ols_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    window: int,
    with_zscore: bool = True,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.OLS`."""
    if cache_dict is not None:
        h = hash(window)
        return cache_dict[h]
    slope, intercept = generic_nb.rolling_ols_nb(x, y, window, minp=minp)
    if with_zscore:
        pred = intercept + slope * x
        error = y - pred
        error_mean = generic_nb.rolling_mean_nb(error, window, minp=minp)
        error_std = generic_nb.rolling_std_nb(error, window, ddof=ddof, minp=minp)
        zscore = (error - error_mean) / error_std
    else:
        zscore = np.full(x.shape, np.nan, dtype=np.float_)
    return slope, intercept, zscore


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        volume=ch.ArraySlicer(axis=1),
        group_lens=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def vwap_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    volume: tp.Array2d,
    group_lens: tp.GroupLens,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.VWAP`."""
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    out = np.empty_like(volume, dtype=np.float_)

    for col in prange(volume.shape[1]):
        for group in range(len(group_lens)):
            from_i = group_start_idxs[group]
            to_i = group_end_idxs[group]
            nom_cumsum = 0
            denum_cumsum = 0
            for i in range(from_i, to_i):
                typical_price = (high[i, col] + low[i, col] + close[i, col]) / 3
                nom_cumsum += volume[i, col] * typical_price
                denum_cumsum += volume[i, col]
                out[i, col] = nom_cumsum / denum_cumsum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivot_info_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2d,
    down_th: tp.FlexArray2d,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.PIVOTINFO`."""
    conf_pivot = np.empty(high.shape, dtype=np.int_)
    conf_idx = np.empty(high.shape, dtype=np.int_)
    last_pivot = np.empty(high.shape, dtype=np.int_)
    last_idx = np.empty(high.shape, dtype=np.int_)

    for col in prange(high.shape[1]):
        _conf_pivot = 0
        _conf_idx = -1
        _conf_value = np.nan
        _last_pivot = 0
        _last_idx = -1
        _last_value = np.nan
        first_valid = -1

        for i in range(high.shape[0]):
            if not np.isnan(high[i, col]) and not np.isnan(low[i, col]):
                if first_valid == -1:
                    first_valid = i
                _up_th = 1 + abs(flex_select_nb(up_th, i, col))
                _down_th = 1 - abs(flex_select_nb(down_th, i, col))

                if _last_pivot == Pivot.Valley:
                    if not np.isnan(_last_value) and not np.isnan(_up_th) and high[i, col] >= _last_value * _up_th:
                        _conf_pivot = _last_pivot
                        _conf_idx = _last_idx
                        _conf_value = _last_value
                        _last_pivot = Pivot.Peak
                        _last_idx = i
                        _last_value = high[i, col]
                    elif np.isnan(_last_value) or low[i, col] < _last_value:
                        _last_idx = i
                        _last_value = low[i, col]
                elif _last_pivot == Pivot.Peak:
                    if not np.isnan(_last_value) and not np.isnan(_down_th) and low[i, col] <= _last_value * _down_th:
                        _conf_pivot = _last_pivot
                        _conf_idx = _last_idx
                        _conf_value = _last_value
                        _last_pivot = Pivot.Valley
                        _last_idx = i
                        _last_value = low[i, col]
                    elif np.isnan(_last_value) or high[i, col] > _last_value:
                        _last_idx = i
                        _last_value = high[i, col]
                else:
                    if not np.isnan(_up_th) and high[i, col] >= low[first_valid, col] * _up_th:
                        _conf_pivot = Pivot.Valley
                        _conf_idx = first_valid
                        _conf_value = low[first_valid, col]
                        _last_pivot = Pivot.Peak
                        _last_idx = i
                        _last_value = high[i, col]
                    if not np.isnan(_down_th) and low[i, col] <= high[first_valid, col] * _down_th:
                        _conf_pivot = Pivot.Peak
                        _conf_idx = first_valid
                        _conf_value = high[first_valid, col]
                        _last_pivot = Pivot.Valley
                        _last_idx = i
                        _last_value = low[i, col]

            conf_pivot[i, col] = _conf_pivot
            conf_idx[i, col] = _conf_idx
            last_pivot[i, col] = _last_pivot
            last_idx[i, col] = _last_idx

    return conf_pivot, conf_idx, last_pivot, last_idx


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        last_pivot=ch.ArraySlicer(axis=1),
        last_idx=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivot_value_nb(high: tp.Array2d, low: tp.Array2d, last_pivot: tp.Array2d, last_idx: tp.Array2d) -> tp.Array2d:
    """Get pivot value."""
    pivot_value = np.empty(high.shape, dtype=np.float_)
    for col in prange(high.shape[1]):
        for i in range(high.shape[0]):
            if last_pivot[i, col] == Pivot.Peak:
                pivot_value[i, col] = high[last_idx[i, col], col]
            elif last_pivot[i, col] == Pivot.Valley:
                pivot_value[i, col] = low[last_idx[i, col], col]
            else:
                pivot_value[i, col] = np.nan
    return pivot_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="conf_pivot", axis=1),
    arg_take_spec=dict(
        conf_pivot=ch.ArraySlicer(axis=1),
        conf_idx=ch.ArraySlicer(axis=1),
        last_pivot=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivots_nb(
    conf_pivot: tp.Array2d,
    conf_idx: tp.Array2d,
    last_pivot: tp.Array2d,
) -> tp.Array2d:
    """Get pivots.

    !!! warning
        To be used in plotting. Do not use it as an indicator!"""
    pivots = np.zeros(conf_pivot.shape, dtype=np.int_)
    for col in prange(conf_pivot.shape[1]):
        for i in range(conf_pivot.shape[0] - 1):
            pivots[conf_idx[i, col], col] = conf_pivot[i, col]
        pivots[-1, col] = last_pivot[-1, col]
    return pivots


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(pivots=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivots_to_modes_nb(pivots: tp.Array2d) -> tp.Array2d:
    """Translate pivots into trend modes.

    !!! warning
        To be used in plotting. Do not use it as an indicator!"""
    modes = np.empty(pivots.shape, dtype=np.int_)
    for col in prange(pivots.shape[1]):
        mode = 0
        for i in range(pivots.shape[0]):
            if pivots[i, col] != 0:
                mode = -pivots[i, col]
            modes[i, col] = mode
    return modes


@register_jitted(cache=True)
def get_tr_iter_nb(high: float, low: float, prev_close: float) -> float:
    """Get TR at one iteration."""
    tr0 = abs(high - low)
    tr1 = abs(high - prev_close)
    tr2 = abs(low - prev_close)
    if np.isnan(tr0) or np.isnan(tr1) or np.isnan(tr2):
        tr = np.nan
    else:
        tr = max(tr0, tr1, tr2)
    return tr


@register_jitted(cache=True)
def get_med_price_iter_nb(high: float, low: float) -> float:
    """Get median price at one iteration."""
    return (high + low) / 2


@register_jitted(cache=True)
def get_basic_bands_iter_nb(high: float, low: float, atr: float, multiplier: float) -> tp.Tuple[float, float]:
    """Get upper and lower bands at one iteration."""
    med_price = get_med_price_iter_nb(high, low)
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower


@register_jitted(cache=True)
def get_final_bands_iter_nb(
    close: float,
    upper: float,
    lower: float,
    prev_upper: float,
    prev_lower: float,
    prev_dir_: int,
) -> tp.Tuple[float, float, float, int, float, float]:
    """Get final bands at one iteration."""
    if close > prev_upper:
        dir_ = 1
    elif close < prev_lower:
        dir_ = -1
    else:
        dir_ = prev_dir_
        if dir_ > 0 and lower < prev_lower:
            lower = prev_lower
        if dir_ < 0 and upper > prev_upper:
            upper = prev_upper

    if dir_ > 0:
        trend = long = lower
        short = np.nan
    else:
        trend = short = upper
        long = np.nan
    return upper, lower, trend, dir_, long, short


@register_jitted(cache=True)
def supertrend_acc_nb(in_state: SuperTrendAIS) -> SuperTrendAOS:
    """Accumulator of `supertrend_nb`.

    Takes a state of type `vectorbtpro.indicators.enums.SuperTrendAIS` and returns
    a state of type `vectorbtpro.indicators.enums.SuperTrendAOS`."""
    i = in_state.i
    high = in_state.high
    low = in_state.low
    close = in_state.close
    prev_close = in_state.prev_close
    prev_upper = in_state.prev_upper
    prev_lower = in_state.prev_lower
    prev_dir_ = in_state.prev_dir_
    nobs = in_state.nobs
    weighted_avg = in_state.weighted_avg
    old_wt = in_state.old_wt
    period = in_state.period
    multiplier = in_state.multiplier

    tr = get_tr_iter_nb(high, low, prev_close)
    alpha = generic_nb.alpha_from_wilder_nb(period)
    ewm_mean_in_state = generic_enums.EWMMeanAIS(
        i=i,
        value=tr,
        old_wt=old_wt,
        weighted_avg=weighted_avg,
        nobs=nobs,
        alpha=alpha,
        minp=period,
        adjust=False,
    )
    ewm_mean_out_state = generic_nb.ewm_mean_acc_nb(ewm_mean_in_state)
    atr = ewm_mean_out_state.value
    upper, lower = get_basic_bands_iter_nb(high, low, atr, multiplier)
    if i == 0:
        trend, dir_, long, short = np.nan, 1, np.nan, np.nan
    else:
        upper, lower, trend, dir_, long, short = get_final_bands_iter_nb(
            close,
            upper,
            lower,
            prev_upper,
            prev_lower,
            prev_dir_,
        )
    return SuperTrendAOS(
        nobs=ewm_mean_out_state.nobs,
        weighted_avg=ewm_mean_out_state.weighted_avg,
        old_wt=ewm_mean_out_state.old_wt,
        upper=upper,
        lower=lower,
        trend=trend,
        dir_=dir_,
        long=long,
        short=short,
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        period=None,
        multiplier=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def supertrend_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    period: int,
    multiplier: float,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.SUPERTREND`."""
    trend = np.empty(close.shape, dtype=np.float_)
    dir_ = np.empty(close.shape, dtype=np.int_)
    long = np.empty(close.shape, dtype=np.float_)
    short = np.empty(close.shape, dtype=np.float_)

    if close.shape[0] == 0:
        return trend, dir_, long, short

    for col in prange(close.shape[1]):
        nobs = 0
        old_wt = 1.0
        weighted_avg = np.nan
        prev_upper = np.nan
        prev_lower = np.nan

        for i in range(close.shape[0]):
            in_state = SuperTrendAIS(
                i=i,
                high=high[i, col],
                low=low[i, col],
                close=close[i, col],
                prev_close=close[i - 1, col] if i > 0 else np.nan,
                prev_upper=prev_upper,
                prev_lower=prev_lower,
                prev_dir_=dir_[i - 1, col] if i > 0 else 1,
                nobs=nobs,
                weighted_avg=weighted_avg,
                old_wt=old_wt,
                period=period,
                multiplier=multiplier,
            )

            out_state = supertrend_acc_nb(in_state)

            nobs = out_state.nobs
            weighted_avg = out_state.weighted_avg
            old_wt = out_state.old_wt
            prev_upper = out_state.upper
            prev_lower = out_state.lower
            trend[i, col] = out_state.trend
            dir_[i, col] = out_state.dir_
            long[i, col] = out_state.long
            short[i, col] = out_state.short

    return trend, dir_, long, short


@register_chunkable(
    size=ch.ArraySizer(arg_query="y", axis=1),
    arg_take_spec=dict(
        y=ch.ArraySlicer(axis=1),
        lag=None,
        factor=base_ch.FlexArraySlicer(axis=1),
        influence=base_ch.FlexArraySlicer(axis=1),
        down_factor=base_ch.FlexArraySlicer(axis=1),
        std_influence=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def signal_detection_nb(
    close: tp.Array2d,
    lag: int,
    factor: tp.FlexArray2d,
    influence: tp.FlexArray2d,
    down_factor: tp.Optional[tp.FlexArray2d] = None,
    std_influence: tp.Optional[tp.FlexArray2d] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.SIGDET`."""
    signal = np.full(close.shape, 0, dtype=np.int_)
    close_mean_filter = close.astype(np.float_)
    close_std_filter = close.astype(np.float_)
    mean_filter = np.full(close.shape, np.nan, dtype=np.float_)
    std_filter = np.full(close.shape, np.nan, dtype=np.float_)
    upper_band = np.full(close.shape, np.nan, dtype=np.float_)
    lower_band = np.full(close.shape, np.nan, dtype=np.float_)
    if lag == 0:
        raise ValueError("Lag cannot be zero")
    if lag - 1 >= close.shape[0]:
        raise ValueError("Lag must be smaller than close")

    for col in prange(close.shape[1]):
        mean_filter[lag - 1, col] = np.nanmean(close[:lag, col])
        std_filter[lag - 1, col] = np.nanstd(close[:lag, col])

        for i in range(lag, close.shape[0]):
            _factor = abs(flex_select_nb(factor, i, col))
            if down_factor is None:
                _down_factor = _factor
            else:
                _down_factor = abs(flex_select_nb(down_factor, i, col))
            _influence = abs(flex_select_nb(influence, i, col))
            if std_influence is None:
                _std_influence = _influence
            else:
                _std_influence = abs(flex_select_nb(std_influence, i, col))

            up_crossed = close[i, col] - mean_filter[i - 1, col] >= _factor * std_filter[i - 1, col]
            down_crossed = close[i, col] - mean_filter[i - 1, col] <= -_down_factor * std_filter[i - 1, col]
            if up_crossed or down_crossed:
                if up_crossed:
                    signal[i, col] = 1
                else:
                    signal[i, col] = -1

                close_mean_filter[i, col] = (
                    _influence * close[i, col] + (1 - _influence) * close_mean_filter[i - 1, col]
                )
                close_std_filter[i, col] = (
                    _std_influence * close[i, col] + (1 - _std_influence) * close_std_filter[i - 1, col]
                )
            else:
                signal[i, col] = 0
                close_mean_filter[i, col] = close[i, col]
                close_std_filter[i, col] = close[i, col]

            mean_filter[i, col] = np.nanmean(close_mean_filter[(i - lag + 1) : i + 1, col])
            std_filter[i, col] = np.nanstd(close_std_filter[(i - lag + 1) : i + 1, col])
            upper_band[i, col] = mean_filter[i, col] + _factor * std_filter[i - 1, col]
            lower_band[i, col] = mean_filter[i, col] - _down_factor * std_filter[i - 1, col]

    return signal, upper_band, lower_band
