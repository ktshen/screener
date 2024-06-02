# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for rolling and expanding windows."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import rank_1d_nb
from vectorbtpro.generic.nb.patterns import pattern_similarity_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

# ############# Rolling functions ############# #


@register_jitted(cache=True)
def rolling_sum_acc_nb(in_state: RollSumAIS) -> RollSumAOS:
    """Accumulator of `rolling_sum_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollSumAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollSumAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum

    return RollSumAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_sum_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling sum.

    Uses `rolling_sum_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).sum()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollSumAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_sum_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sum_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_sum_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_sum_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_prod_acc_nb(in_state: RollProdAIS) -> RollProdAOS:
    """Accumulator of `rolling_prod_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollProdAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollProdAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumprod = in_state.cumprod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumprod = cumprod * value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumprod = cumprod
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumprod = cumprod / pre_window_value
        window_len = window - nancnt
        window_cumprod = cumprod
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumprod

    return RollProdAOS(cumprod=cumprod, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_prod_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling product.

    Uses `rolling_prod_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).apply(np.prod)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumprod = 1.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollProdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumprod=cumprod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_prod_acc_nb(in_state)
        cumprod = out_state.cumprod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_prod_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_prod_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_prod_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_mean_acc_nb(in_state: RollMeanAIS) -> RollMeanAOS:
    """Accumulator of `rolling_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollMeanAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum / window_len

    return RollMeanAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling mean.

    Uses `rolling_mean_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_std_acc_nb(in_state: RollStdAIS) -> RollStdAOS:
    """Accumulator of `rolling_std_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollStdAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollStdAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    cumsum_sq = in_state.cumsum_sq
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
        cumsum_sq = cumsum_sq + value**2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
            cumsum_sq = cumsum_sq - pre_window_value**2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        mean = cumsum / window_len
        value = np.sqrt(np.abs(cumsum_sq - 2 * cumsum * mean + window_len * mean**2) / (window_len - ddof))

    return RollStdAOS(cumsum=cumsum, cumsum_sq=cumsum_sq, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_std_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array1d:
    """Compute rolling standard deviation.

    Uses `rolling_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).std(ddof=ddof)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    cumsum_sq = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollStdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            cumsum_sq=cumsum_sq,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_std_acc_nb(in_state)
        cumsum = out_state.cumsum
        cumsum_sq = out_state.cumsum_sq
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_std_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_std_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_zscore_acc_nb(in_state: RollZScoreAIS) -> RollZScoreAOS:
    """Accumulator of `rolling_zscore_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollZScoreAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollZScoreAOS`."""
    mean_in_state = RollMeanAIS(
        i=in_state.i,
        value=in_state.value,
        pre_window_value=in_state.pre_window_value,
        cumsum=in_state.cumsum,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
    )
    std_in_state = RollStdAIS(
        i=in_state.i,
        value=in_state.value,
        pre_window_value=in_state.pre_window_value,
        cumsum=in_state.cumsum,
        cumsum_sq=in_state.cumsum_sq,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
        ddof=in_state.ddof,
    )
    mean_out_state = rolling_mean_acc_nb(mean_in_state)
    std_out_state = rolling_std_acc_nb(std_in_state)
    if std_out_state.value == 0:
        value = np.nan
    else:
        value = (in_state.value - mean_out_state.value) / std_out_state.value

    return RollZScoreAOS(
        cumsum=std_out_state.cumsum,
        cumsum_sq=std_out_state.cumsum_sq,
        nancnt=std_out_state.nancnt,
        window_len=std_out_state.window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_zscore_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array1d:
    """Compute rolling z-score.

    Uses `rolling_zscore_acc_nb` at each iteration."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    cumsum_sq = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollZScoreAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            cumsum_sq=cumsum_sq,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_zscore_acc_nb(in_state)
        cumsum = out_state.cumsum
        cumsum_sq = out_state.cumsum_sq
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_zscore_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_zscore_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_zscore_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def wm_mean_acc_nb(in_state: WMMeanAIS) -> WMMeanAOS:
    """Accumulator of `wm_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.WMMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.WMMeanAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    wcumsum = in_state.wcumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if i >= window and not np.isnan(pre_window_value):
        wcumsum = wcumsum - cumsum
    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
    if not np.isnan(value):
        wcumsum = wcumsum + value * window_len
    if window_len < minp:
        value = np.nan
    else:
        value = wcumsum * 2 / (window_len + 1) / window_len

    return WMMeanAOS(cumsum=cumsum, wcumsum=wcumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def wm_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute weighted moving average.

    Uses `wm_mean_acc_nb` at each iteration."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    wcumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = WMMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            wcumsum=wcumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = wm_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        wcumsum = out_state.wcumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wm_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `wm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wm_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def alpha_from_com_nb(com: float) -> float:
    """Get the smoothing factor `alpha` from a center of mass."""
    return 1.0 / (1.0 + com)


@register_jitted(cache=True)
def alpha_from_span_nb(span: float) -> float:
    """Get the smoothing factor `alpha` from a span."""
    com = (span - 1) / 2.0
    return alpha_from_com_nb(com)


@register_jitted(cache=True)
def alpha_from_halflife_nb(halflife: float) -> float:
    """Get the smoothing factor `alpha` from a half-life."""
    return 1 - np.exp(-np.log(2) / halflife)


@register_jitted(cache=True)
def alpha_from_wilder_nb(period: int) -> float:
    """Get the smoothing factor `alpha` from a Wilder's period."""
    return 1 / period


@register_jitted(cache=True)
def ewm_mean_acc_nb(in_state: EWMMeanAIS) -> EWMMeanAOS:
    """Accumulator of `ewm_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.EWMMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.EWMMeanAOS`."""
    i = in_state.i
    value = in_state.value
    old_wt = in_state.old_wt
    weighted_avg = in_state.weighted_avg
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    if i > 0:
        is_observation = not np.isnan(value)
        nobs += is_observation
        if not np.isnan(weighted_avg):
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != value:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * value)) / (old_wt + new_wt)
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.0
        elif is_observation:
            weighted_avg = value
    else:
        is_observation = not np.isnan(weighted_avg)
        nobs += int(is_observation)
    value = weighted_avg if (nobs >= minp) else np.nan

    return EWMMeanAOS(old_wt=old_wt, weighted_avg=weighted_avg, nobs=nobs, value=value)


@register_jitted(cache=True)
def ewm_mean_1d_nb(arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving average.

    Uses `ewm_mean_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp, adjust=adjust).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=np.float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    weighted_avg = float(arr[0]) + 0.0  # cast to np.float_
    nobs = 0
    old_wt = 1.0

    for i in range(len(arr)):
        in_state = EWMMeanAIS(
            i=i,
            value=arr[i],
            old_wt=old_wt,
            weighted_avg=weighted_avg,
            nobs=nobs,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_mean_acc_nb(in_state)
        old_wt = out_state.old_wt
        weighted_avg = out_state.weighted_avg
        nobs = out_state.nobs
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_mean_nb(arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_mean_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def ewm_std_acc_nb(in_state: EWMStdAIS) -> EWMStdAOS:
    """Accumulator of `ewm_std_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.EWMStdAIS` and returns
    a state of type `vectorbtpro.generic.enums.EWMStdAOS`."""
    i = in_state.i
    value = in_state.value
    mean_x = in_state.mean_x
    mean_y = in_state.mean_y
    cov = in_state.cov
    sum_wt = in_state.sum_wt
    sum_wt2 = in_state.sum_wt2
    old_wt = in_state.old_wt
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    cur_x = value
    cur_y = value
    is_observation = not np.isnan(cur_x) and not np.isnan(cur_y)
    nobs += is_observation
    if i > 0:
        if not np.isnan(mean_x):
            sum_wt *= old_wt_factor
            sum_wt2 *= old_wt_factor * old_wt_factor
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt)
                cov = (
                    (old_wt * (cov + ((old_mean_x - mean_x) * (old_mean_y - mean_y))))
                    + (new_wt * ((cur_x - mean_x) * (cur_y - mean_y)))
                ) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += new_wt * new_wt
                old_wt += new_wt
                if not adjust:
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = 1.0
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y
    else:
        if not is_observation:
            mean_x = np.nan
            mean_y = np.nan

    if nobs >= minp:
        numerator = sum_wt * sum_wt
        denominator = numerator - sum_wt2
        if denominator > 0.0:
            value = (numerator / denominator) * cov
        else:
            value = np.nan
    else:
        value = np.nan

    return EWMStdAOS(
        mean_x=mean_x,
        mean_y=mean_y,
        cov=cov,
        sum_wt=sum_wt,
        sum_wt2=sum_wt2,
        old_wt=old_wt,
        nobs=nobs,
        value=value,
    )


@register_jitted(cache=True)
def ewm_std_1d_nb(arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving standard deviation.

    Uses `ewm_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp).std()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=np.float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    mean_x = float(arr[0]) + 0.0  # cast to np.float_
    mean_y = float(arr[0]) + 0.0  # cast to np.float_
    nobs = 0
    cov = 0.0
    sum_wt = 1.0
    sum_wt2 = 1.0
    old_wt = 1.0

    for i in range(len(arr)):
        in_state = EWMStdAIS(
            i=i,
            value=arr[i],
            mean_x=mean_x,
            mean_y=mean_y,
            cov=cov,
            sum_wt=sum_wt,
            sum_wt2=sum_wt2,
            old_wt=old_wt,
            nobs=nobs,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_std_acc_nb(in_state)
        mean_x = out_state.mean_x
        mean_y = out_state.mean_y
        cov = out_state.cov
        sum_wt = out_state.sum_wt
        sum_wt2 = out_state.sum_wt2
        old_wt = out_state.old_wt
        nobs = out_state.nobs
        out[i] = out_state.value

    return np.sqrt(out)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_std_nb(arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_std_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def wwm_mean_1d_nb(arr: tp.Array1d, period: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving average."""
    if minp is None:
        minp = period
    return ewm_mean_1d_nb(arr, 2 * period - 1, minp=minp, adjust=adjust)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_mean_nb(arr: tp.Array2d, period: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `wwm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_mean_1d_nb(arr[:, col], period, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def wwm_std_1d_nb(arr: tp.Array1d, period: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving standard deviation."""
    if minp is None:
        minp = period
    return ewm_std_1d_nb(arr, 2 * period - 1, minp=minp, adjust=adjust)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_std_nb(arr: tp.Array2d, period: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `wwm_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_std_1d_nb(arr[:, col], period, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def vidya_acc_nb(in_state: VidyaAIS) -> VidyaAOS:
    """Accumulator of `vidya_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.VidyaAIS` and returns
    a state of type `vectorbtpro.generic.enums.VidyaAOS`."""
    i = in_state.i
    prev_value = in_state.prev_value
    value = in_state.value
    pre_window_prev_value = in_state.pre_window_prev_value
    pre_window_value = in_state.pre_window_value
    pos_cumsum = in_state.pos_cumsum
    neg_cumsum = in_state.neg_cumsum
    prev_vidya = in_state.prev_vidya
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    alpha = 2 / (window + 1)

    diff = value - prev_value
    if np.isnan(diff):
        nancnt = nancnt + 1
    else:
        if diff > 0:
            pos_value = diff
            neg_value = 0.0
        else:
            pos_value = 0.0
            neg_value = abs(diff)
        pos_cumsum = pos_cumsum + pos_value
        neg_cumsum = neg_cumsum + neg_value
    if i < window:
        window_len = i + 1 - nancnt
    else:
        pre_window_diff = pre_window_value - pre_window_prev_value
        if np.isnan(pre_window_diff):
            nancnt = nancnt - 1
        else:
            if pre_window_diff > 0:
                pre_window_pos_value = pre_window_diff
                pre_window_neg_value = 0.0
            else:
                pre_window_pos_value = 0.0
                pre_window_neg_value = abs(pre_window_diff)
            pos_cumsum = pos_cumsum - pre_window_pos_value
            neg_cumsum = neg_cumsum - pre_window_neg_value
        window_len = window - nancnt
    window_pos_cumsum = pos_cumsum
    window_neg_cumsum = neg_cumsum
    if window_len < minp:
        cmo = np.nan
        vidya = np.nan
    else:
        sh = window_pos_cumsum
        sl = window_neg_cumsum
        if sh + sl == 0:
            cmo = 0.0
        else:
            cmo = np.abs((sh - sl) / (sh + sl))
        if np.isnan(prev_vidya):
            prev_vidya = 0.0
        vidya = alpha * cmo * value + prev_vidya * (1 - alpha * cmo)

    return VidyaAOS(
        pos_cumsum=pos_cumsum,
        neg_cumsum=neg_cumsum,
        nancnt=nancnt,
        window_len=window_len,
        cmo=cmo,
        vidya=vidya,
    )


@register_jitted(cache=True)
def vidya_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute VIDYA.

    Uses `vidya_acc_nb` at each iteration."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    pos_cumsum = 0.0
    neg_cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = VidyaAIS(
            i=i,
            prev_value=arr[i - 1] if i - 1 >= 0 else np.nan,
            value=arr[i],
            pre_window_prev_value=arr[i - window - 1] if i - window - 1 >= 0 else np.nan,
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            pos_cumsum=pos_cumsum,
            neg_cumsum=neg_cumsum,
            prev_vidya=out[i - 1] if i - 1 >= 0 else np.nan,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = vidya_acc_nb(in_state)
        pos_cumsum = out_state.pos_cumsum
        neg_cumsum = out_state.neg_cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.vidya

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def vidya_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `vidya_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = vidya_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def ma_1d_nb(
    arr: tp.Array1d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Compute a moving average based on the mode of the type `vectorbtpro.generic.enums.WType`."""
    if wtype == WType.Simple:
        return rolling_mean_1d_nb(arr, window, minp=minp)
    if wtype == WType.Weighted:
        return wm_mean_1d_nb(arr, window, minp=minp)
    if wtype == WType.Exp:
        return ewm_mean_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Wilder:
        return wwm_mean_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Vidya:
        return vidya_1d_nb(arr, window, minp=minp)
    raise ValueError("Invalid rolling mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, wtype=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ma_nb(
    arr: tp.Array2d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """2-dim version of `ma_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ma_1d_nb(arr[:, col], window, wtype=wtype, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def msd_1d_nb(
    arr: tp.Array1d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array1d:
    """Compute a moving standard deviation based on the mode of the type `vectorbtpro.generic.enums.WType`."""
    if wtype == WType.Simple:
        return rolling_std_1d_nb(arr, window, minp=minp, ddof=ddof)
    if wtype == WType.Weighted:
        raise ValueError("Weighted mode is not supported for standard deviations")
    if wtype == WType.Exp:
        return ewm_std_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Wilder:
        return wwm_std_1d_nb(arr, window, minp=minp, adjust=adjust)
    raise ValueError("Invalid rolling mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, wtype=None, minp=None, adjust=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def msd_nb(
    arr: tp.Array2d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """2-dim version of `msd_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = msd_1d_nb(arr[:, col], window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_cov_acc_nb(in_state: RollCovAIS) -> RollCovAOS:
    """Accumulator of `rolling_cov_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollCovAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollCovAOS`."""
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        window_prod_mean = cumsum_prod / (window_len - ddof)
        window_mean1 = cumsum1 / window_len
        window_mean2 = cumsum2 / window_len
        window_mean_prod = window_mean1 * window_mean2 * window_len / (window_len - ddof)
        value = window_prod_mean - window_mean_prod

    return RollCovAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_cov_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array1d:
    """Compute rolling covariance.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).cov(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCovAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_cov_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_cov_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array2d:
    """2-dim version of `rolling_cov_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_cov_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_corr_acc_nb(in_state: RollCorrAIS) -> RollCorrAOS:
    """Accumulator of `rolling_corr_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollCorrAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollCorrAOS`."""
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_sq1 = in_state.cumsum_sq1
    cumsum_sq2 = in_state.cumsum_sq2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_sq1 = cumsum_sq1 + value1**2
        cumsum_sq2 = cumsum_sq2 + value2**2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_sq1 = cumsum_sq1 - pre_window_value1**2
            cumsum_sq2 = cumsum_sq2 - pre_window_value2**2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp:
        value = np.nan
    else:
        nom = window_len * cumsum_prod - cumsum1 * cumsum2
        denom1 = np.sqrt(window_len * cumsum_sq1 - cumsum1**2)
        denom2 = np.sqrt(window_len * cumsum_sq2 - cumsum2**2)
        denom = denom1 * denom2
        if denom == 0:
            value = np.nan
        else:
            value = nom / denom

    return RollCorrAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_sq1=cumsum_sq1,
        cumsum_sq2=cumsum_sq2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_corr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling correlation coefficient.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).corr(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_sq1 = 0.0
    cumsum_sq2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCorrAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_sq1=cumsum_sq1,
            cumsum_sq2=cumsum_sq2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_corr_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_sq1 = out_state.cumsum_sq1
        cumsum_sq2 = out_state.cumsum_sq2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_corr_nb(arr1: tp.Array2d, arr2: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_corr_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_corr_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_ols_acc_nb(in_state: RollOLSAIS) -> RollOLSAOS:
    """Accumulator of `rolling_ols_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollOLSAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollOLSAOS`."""
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    validcnt = in_state.validcnt
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_sq1 = in_state.cumsum_sq1
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        validcnt = validcnt + 1
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_sq1 = cumsum_sq1 + value1**2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            validcnt = validcnt - 1
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_sq1 = cumsum_sq1 - pre_window_value1**2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp:
        slope_value = np.nan
        intercept_value = np.nan
    else:
        slope_num = validcnt * cumsum_prod - cumsum1 * cumsum2
        slope_denom = validcnt * cumsum_sq1 - cumsum1**2
        if slope_denom != 0:
            slope_value = slope_num / slope_denom
        else:
            slope_value = np.nan
        intercept_num = cumsum2 - slope_value * cumsum1
        intercept_denom = validcnt
        if intercept_denom != 0:
            intercept_value = intercept_num / intercept_denom
        else:
            intercept_value = np.nan

    return RollOLSAOS(
        validcnt=validcnt,
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_sq1=cumsum_sq1,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        slope_value=slope_value,
        intercept_value=intercept_value,
    )


@register_jitted(cache=True)
def rolling_ols_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Compute rolling linear regression."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    slope_out = np.empty_like(arr1, dtype=np.float_)
    intercept_out = np.empty_like(arr1, dtype=np.float_)
    validcnt = 0
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_sq1 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollOLSAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            validcnt=validcnt,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_sq1=cumsum_sq1,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_ols_acc_nb(in_state)
        validcnt = out_state.validcnt
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_sq1 = out_state.cumsum_sq1
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        slope_out[i] = out_state.slope_value
        intercept_out[i] = out_state.intercept_value

    return slope_out, intercept_out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_ols_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """2-dim version of `rolling_ols_1d_nb`."""
    slope_out = np.empty_like(arr1, dtype=np.float_)
    intercept_out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        slope_out[:, col], intercept_out[:, col] = rolling_ols_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp)
    return slope_out, intercept_out


@register_jitted(cache=True)
def rolling_rank_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, pct: bool = False) -> tp.Array1d:
    """Rolling version of `rank_1d_nb`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            valid_cnt = window - nancnt
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            arr_window = arr[from_i:to_i]
            out[i] = rank_1d_nb(arr_window, pct=pct)[-1]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, pct=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_rank_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, pct: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_rank_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_rank_1d_nb(arr[:, col], window, minp=minp, pct=pct)
    return out


@register_jitted(cache=True)
def rolling_min_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling min.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_min_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_min_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_max_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling max.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_max_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_max_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_argmin_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Compute rolling min index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        if local:
            mini = 0
        else:
            mini = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
                if local:
                    mini = k
                else:
                    mini = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = mini
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmin_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmin_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmin_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_argmax_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Compute rolling max index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        if local:
            maxi = 0
        else:
            maxi = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
                if local:
                    maxi = k
                else:
                    maxi = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = maxi
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmax_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmax_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmax_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_any_1d_nb(arr: tp.Array1d, window: int) -> tp.Array1d:
    """Compute rolling any."""
    out = np.empty_like(arr, dtype=np.bool_)
    last_true_i = -1
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]) and arr[i]:
            last_true_i = i
        from_i = max(0, i + 1 - window)
        if last_true_i >= from_i:
            out[i] = True
        else:
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_any_nb(arr: tp.Array2d, window: int) -> tp.Array2d:
    """2-dim version of `rolling_any_1d_nb`."""
    out = np.empty_like(arr, dtype=np.bool_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_any_1d_nb(arr[:, col], window)
    return out


@register_jitted(cache=True)
def rolling_all_1d_nb(arr: tp.Array1d, window: int) -> tp.Array1d:
    """Compute rolling all."""
    out = np.empty_like(arr, dtype=np.bool_)
    last_false_i = -1
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]) or not arr[i]:
            last_false_i = i
        from_i = max(0, i + 1 - window)
        if last_false_i >= from_i:
            out[i] = False
        else:
            out[i] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_all_nb(arr: tp.Array2d, window: int) -> tp.Array2d:
    """2-dim version of `rolling_all_1d_nb`."""
    out = np.empty_like(arr, dtype=np.bool_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_all_1d_nb(arr[:, col], window)
    return out


@register_jitted(cache=True)
def rolling_pattern_similarity_1d_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Compute rolling pattern similarity.

    Uses `vectorbtpro.generic.nb.patterns.pattern_similarity_nb`."""
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    out = np.full(arr.shape, np.nan, dtype=np.float_)
    min_max_required = False
    if rescale_mode == RescaleMode.MinMax:
        min_max_required = True
    if not np.isnan(min_pct_change):
        min_max_required = True
    if not np.isnan(max_pct_change):
        min_max_required = True
    if not max_error_as_maxdist:
        min_max_required = True
    if min_max_required:
        if np.isnan(pmin):
            pmin = np.nanmin(pattern)
        if np.isnan(pmax):
            pmax = np.nanmax(pattern)

    for i in range(arr.shape[0]):
        from_i = i - window + 1
        to_i = i + 1
        if from_i < 0:
            continue

        if np.random.uniform(0, 1) < row_select_prob:
            _vmin = vmin
            _vmax = vmax
            if min_max_required:
                if np.isnan(_vmin) or np.isnan(_vmax):
                    for j in range(from_i, to_i):
                        if np.isnan(_vmin) or arr[j] < _vmin:
                            _vmin = arr[j]
                        if np.isnan(_vmax) or arr[j] > _vmax:
                            _vmax = arr[j]

            for w in range(window, max_window + 1):
                from_i = i - w + 1
                to_i = i + 1
                if from_i < 0:
                    continue
                if min_max_required:
                    if w > window:
                        if arr[from_i] < _vmin:
                            _vmin = arr[from_i]
                        if arr[from_i] > _vmax:
                            _vmax = arr[from_i]

                if np.random.uniform(0, 1) < window_select_prob:
                    arr_window = arr[from_i:to_i]
                    similarity = pattern_similarity_nb(
                        arr_window,
                        pattern,
                        interp_mode=interp_mode,
                        rescale_mode=rescale_mode,
                        vmin=_vmin,
                        vmax=_vmax,
                        pmin=pmin,
                        pmax=pmax,
                        invert=invert,
                        error_type=error_type,
                        distance_measure=distance_measure,
                        max_error=max_error_,
                        max_error_interp_mode=max_error_interp_mode,
                        max_error_as_maxdist=max_error_as_maxdist,
                        max_error_strict=max_error_strict,
                        min_pct_change=min_pct_change,
                        max_pct_change=max_pct_change,
                        min_similarity=min_similarity,
                        minp=minp,
                    )
                    if not np.isnan(similarity):
                        if not np.isnan(out[i]):
                            if similarity > out[i]:
                                out[i] = similarity
                        else:
                            out[i] = similarity

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        pattern=None,
        window=None,
        max_window=None,
        row_select_prob=None,
        window_select_prob=None,
        interp_mode=None,
        rescale_mode=None,
        vmin=None,
        vmax=None,
        pmin=None,
        pmax=None,
        invert=None,
        error_type=None,
        distance_measure=None,
        max_error=None,
        max_error_interp_mode=None,
        max_error_as_maxdist=None,
        max_error_strict=None,
        min_pct_change=None,
        max_pct_change=None,
        min_similarity=None,
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_pattern_similarity_nb(
    arr: tp.Array2d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """2-dim version of `rolling_pattern_similarity_1d_nb`."""
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    out = np.full(arr.shape, np.nan, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_pattern_similarity_1d_nb(
            arr[:, col],
            pattern,
            window=window,
            max_window=max_window,
            row_select_prob=row_select_prob,
            window_select_prob=window_select_prob,
            interp_mode=interp_mode,
            rescale_mode=rescale_mode,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            invert=invert,
            error_type=error_type,
            distance_measure=distance_measure,
            max_error=max_error_,
            max_error_interp_mode=max_error_interp_mode,
            max_error_as_maxdist=max_error_as_maxdist,
            max_error_strict=max_error_strict,
            min_pct_change=min_pct_change,
            max_pct_change=max_pct_change,
            min_similarity=min_similarity,
            minp=minp,
        )
    return out


# ############# Expanding functions ############# #


@register_jitted(cache=True)
def expanding_min_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding min.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).min()`."""
    out = np.empty_like(arr, dtype=np.float_)
    minv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(minv) or arr[i] < minv:
            minv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_min_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_min_1d_nb(arr[:, col], minp=minp)
    return out


@register_jitted(cache=True)
def expanding_max_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding max.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).max()`."""
    out = np.empty_like(arr, dtype=np.float_)
    maxv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(maxv) or arr[i] > maxv:
            maxv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_max_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_max_1d_nb(arr[:, col], minp=minp)
    return out
