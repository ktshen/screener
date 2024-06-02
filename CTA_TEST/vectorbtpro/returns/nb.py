# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for returns.

Provides an arsenal of Numba-compiled functions that are used by accessors and for measuring
portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.returns.enums import RollSharpeAIS, RollSharpeAOS
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb

__all__ = []


# ############# Metrics ############# #


@register_jitted(cache=True)
def get_return_nb(input_value: float, output_value: float, log_returns: bool = False) -> float:
    """Calculate return from input and output value."""
    if input_value == 0:
        if output_value == 0:
            return 0.0
        return np.inf * np.sign(output_value)
    return_value = add_nb(output_value, -input_value) / input_value
    if log_returns:
        return np.log1p(return_value)
    return return_value


@register_jitted(cache=True)
def returns_1d_nb(arr: tp.Array1d, init_value: float = np.nan, log_returns: bool = False) -> tp.Array1d:
    """Calculate returns."""
    out = np.empty(arr.shape, dtype=np.float_)
    if np.isnan(init_value) and arr.shape[0] > 0:
        input_value = arr[0]
    else:
        input_value = init_value
    for i in range(arr.shape[0]):
        output_value = arr[i]
        out[i] = get_return_nb(input_value, output_value, log_returns=log_returns)
        input_value = output_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        log_returns=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    arr: tp.Array2d,
    init_value: tp.FlexArray1dLike = np.nan,
    log_returns: bool = False,
) -> tp.Array2d:
    """2-dim version of `returns_1d_nb`."""
    init_value_ = to_1d_array_nb(np.asarray(init_value))

    out = np.empty(arr.shape, dtype=np.float_)
    for col in prange(out.shape[1]):
        _init_value = flex_select_1d_pc_nb(init_value_, col)
        out[:, col] = returns_1d_nb(arr[:, col], init_value=_init_value, log_returns=log_returns)
    return out


@register_jitted(cache=True)
def cum_returns_1d_nb(rets: tp.Array1d, start_value: float = 0.0, log_returns: bool = False) -> tp.Array1d:
    """Cumulative returns."""
    out = np.empty_like(rets, dtype=np.float_)
    if log_returns:
        cumsum = 0
        for i in range(rets.shape[0]):
            if not np.isnan(rets[i]):
                cumsum += rets[i]
            if start_value == 0:
                out[i] = cumsum
            else:
                out[i] = np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(rets.shape[0]):
            if not np.isnan(rets[i]):
                cumprod *= 1 + rets[i]
            if start_value == 0:
                out[i] = cumprod - 1
            else:
                out[i] = cumprod * start_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), start_value=None, log_returns=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cum_returns_nb(rets: tp.Array2d, start_value: float = 0.0, log_returns: bool = False) -> tp.Array2d:
    """2-dim version of `cum_returns_1d_nb`."""
    out = np.empty_like(rets, dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[:, col] = cum_returns_1d_nb(rets[:, col], start_value=start_value, log_returns=log_returns)
    return out


@register_jitted(cache=True)
def cum_returns_final_1d_nb(rets: tp.Array1d, start_value: float = 0.0, log_returns: bool = False) -> float:
    """Total return."""
    if log_returns:
        cumsum = 0
        for i in range(rets.shape[0]):
            if not np.isnan(rets[i]):
                cumsum += rets[i]
        if start_value == 0:
            return cumsum
        return np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(rets.shape[0]):
            if not np.isnan(rets[i]):
                cumprod *= 1 + rets[i]
        if start_value == 0:
            return cumprod - 1
        return cumprod * start_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), start_value=None, log_returns=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cum_returns_final_nb(rets: tp.Array2d, start_value: float = 0.0, log_returns: bool = False) -> tp.Array1d:
    """2-dim version of `cum_returns_final_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = cum_returns_final_1d_nb(rets[:, col], start_value=start_value, log_returns=log_returns)
    return out


@register_jitted(cache=True)
def annualized_return_1d_nb(
    rets: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Annualized total return.

    This is equivalent to the compound annual growth rate (CAGR)."""
    if period is None:
        period = rets.shape[0]
    cum_return = cum_returns_final_1d_nb(rets, start_value=1.0, log_returns=log_returns)
    return cum_return ** (ann_factor / period) - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), ann_factor=None, period=None, log_returns=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_return_nb(
    rets: tp.Array2d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> tp.Array1d:
    """2-dim version of `annualized_return_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = annualized_return_1d_nb(rets[:, col], ann_factor, period=period, log_returns=log_returns)
    return out


@register_jitted(cache=True)
def annualized_volatility_1d_nb(rets: tp.Array1d, ann_factor: float, levy_alpha: float = 2.0, ddof: int = 0) -> float:
    """Annualized volatility of a strategy."""
    return generic_nb.nanstd_1d_nb(rets, ddof) * ann_factor ** (1.0 / levy_alpha)


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), ann_factor=None, levy_alpha=None, ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_volatility_nb(rets: tp.Array2d, ann_factor: float, levy_alpha: float = 2.0, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `annualized_volatility_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = annualized_volatility_1d_nb(rets[:, col], ann_factor, levy_alpha, ddof)
    return out


@register_jitted(cache=True)
def max_drawdown_1d_nb(rets: tp.Array1d, log_returns: bool = False) -> float:
    """Total maximum drawdown (MDD)."""
    cum_ret = np.nan
    value_max = 1.0
    out = 0.0
    for i in range(rets.shape[0]):
        if not np.isnan(rets[i]):
            if np.isnan(cum_ret):
                cum_ret = 1.0
            if log_returns:
                ret = np.exp(rets[i]) - 1
            else:
                ret = rets[i]
            cum_ret *= ret + 1.0
        if cum_ret > value_max:
            value_max = cum_ret
        elif cum_ret < value_max:
            dd = cum_ret / value_max - 1
            if dd < out:
                out = dd
    if np.isnan(cum_ret):
        return np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), log_returns=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def max_drawdown_nb(rets: tp.Array2d, log_returns: bool = False) -> tp.Array1d:
    """2-dim version of `max_drawdown_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = max_drawdown_1d_nb(rets[:, col], log_returns=log_returns)
    return out


@register_jitted(cache=True)
def calmar_ratio_1d_nb(
    rets: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Calmar ratio, or drawdown ratio, of a strategy."""
    max_drawdown = max_drawdown_1d_nb(rets, log_returns=log_returns)
    if max_drawdown == 0:
        return np.nan
    annualized_return = annualized_return_1d_nb(rets, ann_factor, period=period, log_returns=log_returns)
    if max_drawdown == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / np.abs(max_drawdown)


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), ann_factor=None, period=None, log_returns=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def calmar_ratio_nb(
    rets: tp.Array2d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> tp.Array1d:
    """2-dim version of `calmar_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = calmar_ratio_1d_nb(rets[:, col], ann_factor, period=period, log_returns=log_returns)
    return out


@register_jitted(cache=True)
def deannualized_return_nb(ret: float, ann_factor: float) -> float:
    """Deannualized return."""
    if ann_factor == 1:
        return ret
    if ann_factor <= -1:
        return np.nan
    return (1 + ret) ** (1.0 / ann_factor) - 1


@register_jitted(cache=True)
def omega_ratio_1d_nb(adj_rets: tp.Array1d) -> float:
    """Omega ratio of a strategy."""
    numer = 0.0
    denom = 0.0
    for i in range(adj_rets.shape[0]):
        ret = adj_rets[i]
        if ret > 0:
            numer += ret
        elif ret < 0:
            denom -= ret
    if denom == 0:
        if numer == 0:
            return np.nan
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def omega_ratio_nb(adj_rets: tp.Array2d) -> tp.Array1d:
    """2-dim version of `omega_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = omega_ratio_1d_nb(adj_rets[:, col])
    return out


@register_jitted(cache=True)
def sharpe_ratio_1d_nb(adj_rets: tp.Array1d, ann_factor: float, ddof: int = 0) -> float:
    """Sharpe ratio of a strategy."""
    mean = np.nanmean(adj_rets)
    std = generic_nb.nanstd_1d_nb(adj_rets, ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1), ann_factor=None, ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sharpe_ratio_nb(adj_rets: tp.Array2d, ann_factor: float, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `sharpe_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = sharpe_ratio_1d_nb(adj_rets[:, col], ann_factor, ddof)
    return out


@register_jitted(cache=True)
def downside_risk_1d_nb(adj_rets: tp.Array1d, ann_factor: float) -> float:
    """Downside deviation below a threshold."""
    cnt = 0
    adj_ret_sqrd_sum = np.nan
    for i in range(adj_rets.shape[0]):
        if not np.isnan(adj_rets[i]):
            cnt += 1
            if np.isnan(adj_ret_sqrd_sum):
                adj_ret_sqrd_sum = 0.0
            if adj_rets[i] <= 0:
                adj_ret_sqrd_sum += adj_rets[i] ** 2
    adj_ret_sqrd_mean = adj_ret_sqrd_sum / cnt
    return np.sqrt(adj_ret_sqrd_mean) * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1), ann_factor=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def downside_risk_nb(adj_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `downside_risk_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = downside_risk_1d_nb(adj_rets[:, col], ann_factor)
    return out


@register_jitted(cache=True)
def sortino_ratio_1d_nb(adj_rets: tp.Array1d, ann_factor: float) -> float:
    """Sortino ratio of a strategy."""
    avg_annualized_return = np.nanmean(adj_rets) * ann_factor
    downside_risk = downside_risk_1d_nb(adj_rets, ann_factor)
    if downside_risk == 0:
        if avg_annualized_return == 0:
            return np.nan
        return np.inf
    return avg_annualized_return / downside_risk


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1), ann_factor=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sortino_ratio_nb(adj_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `sortino_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = sortino_ratio_1d_nb(adj_rets[:, col], ann_factor)
    return out


@register_jitted(cache=True)
def information_ratio_1d_nb(adj_rets: tp.Array1d, ddof: int = 0) -> float:
    """Information ratio of a strategy."""
    mean = np.nanmean(adj_rets)
    std = generic_nb.nanstd_1d_nb(adj_rets, ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def information_ratio_nb(adj_rets: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `information_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = information_ratio_1d_nb(adj_rets[:, col], ddof)
    return out


@register_jitted(cache=True)
def beta_1d_nb(rets: tp.Array1d, bm_returns: tp.Array1d, ddof: int = 0) -> float:
    """Beta."""
    cov = generic_nb.nancov_1d_nb(rets, bm_returns, ddof=ddof)
    var = generic_nb.nanvar_1d_nb(bm_returns, ddof=ddof)
    if var == 0:
        if cov == 0:
            return np.nan
        return np.inf
    return cov / var


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), bm_returns=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def beta_nb(rets: tp.Array2d, bm_returns: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `beta_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = beta_1d_nb(rets[:, col], bm_returns[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def beta_rollmeta_nb(
    from_i: int,
    to_i: int,
    col: int,
    rets: tp.Array2d,
    bm_returns: tp.Array1d,
    ddof: int = 0,
) -> float:
    """Rolling apply meta function based on `beta_1d_nb`."""
    return beta_1d_nb(rets[from_i:to_i, col], bm_returns[from_i:to_i, col], ddof)


@register_jitted(cache=True)
def alpha_1d_nb(adj_rets: tp.Array1d, adj_bm_returns: tp.Array1d, ann_factor: float) -> float:
    """Annualized alpha."""
    beta = beta_1d_nb(adj_rets, adj_bm_returns)
    return (np.nanmean(adj_rets) - beta * np.nanmean(adj_bm_returns) + 1) ** ann_factor - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="adj_rets", axis=1),
    arg_take_spec=dict(adj_rets=ch.ArraySlicer(axis=1), adj_bm_returns=ch.ArraySlicer(axis=1), ann_factor=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def alpha_nb(adj_rets: tp.Array2d, adj_bm_returns: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `alpha_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = alpha_1d_nb(adj_rets[:, col], adj_bm_returns[:, col], ann_factor)
    return out


@register_jitted(cache=True)
def alpha_rollmeta_nb(
    from_i: int,
    to_i: int,
    col: int,
    adj_rets: tp.Array2d,
    adj_bm_returns: tp.Array1d,
    ann_factor: float,
) -> float:
    """Rolling apply meta function based on `alpha_1d_nb`."""
    return alpha_1d_nb(adj_rets[from_i:to_i, col], adj_bm_returns[from_i:to_i, col], ann_factor)


@register_jitted(cache=True)
def tail_ratio_1d_nb(rets: tp.Array1d) -> float:
    """Ratio between the right (95%) and left tail (5%)."""
    perc_95 = np.abs(np.nanpercentile(rets, 95))
    perc_5 = np.abs(np.nanpercentile(rets, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_jitted(cache=True)
def tail_ratio_noarr_1d_nb(rets: tp.Array1d) -> float:
    """`tail_ratio_1d_nb` that does not allocate any arrays."""
    perc_95 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(rets, 95))
    perc_5 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(rets, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def tail_ratio_nb(rets: tp.Array2d) -> tp.Array1d:
    """2-dim version of `tail_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = tail_ratio_1d_nb(rets[:, col])
    return out


@register_jitted(cache=True)
def value_at_risk_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """Value at risk (VaR) of a returns stream."""
    return np.nanpercentile(rets, 100 * cutoff)


@register_jitted(cache=True)
def value_at_risk_noarr_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """`value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpercentile_noarr_1d_nb(rets, 100 * cutoff)


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), cutoff=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_at_risk_nb(rets: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """2-dim version of `value_at_risk_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = value_at_risk_1d_nb(rets[:, col], cutoff)
    return out


@register_jitted(cache=True)
def cond_value_at_risk_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """Conditional value at risk (CVaR) of a returns stream."""
    cutoff_index = int((len(rets) - 1) * cutoff)
    return np.mean(np.partition(rets, cutoff_index)[: cutoff_index + 1])


@register_jitted(cache=True)
def cond_value_at_risk_noarr_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """`cond_value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpartition_mean_noarr_1d_nb(rets, cutoff * 100)


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(rets=ch.ArraySlicer(axis=1), cutoff=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cond_value_at_risk_nb(rets: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """2-dim version of `cond_value_at_risk_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = cond_value_at_risk_1d_nb(rets[:, col], cutoff)
    return out


@register_jitted(cache=True)
def capture_ratio_1d_nb(
    rets: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Capture ratio."""
    annualized_return1 = annualized_return_1d_nb(rets, ann_factor, period=period, log_returns=log_returns)
    annualized_return2 = annualized_return_1d_nb(bm_returns, ann_factor, period=period, log_returns=log_returns)
    if annualized_return2 == 0:
        if annualized_return1 == 0:
            return np.nan
        return np.inf
    return annualized_return1 / annualized_return2


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None,
        log_returns=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def capture_ratio_nb(
    rets: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> tp.Array1d:
    """2-dim version of `capture_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = capture_ratio_1d_nb(
            rets[:, col],
            bm_returns[:, col],
            ann_factor,
            period=period,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def capture_ratio_rollmeta_nb(
    from_i: int,
    to_i: int,
    col: int,
    rets: tp.Array2d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Rolling apply meta function based on `capture_ratio_1d_nb`."""
    return capture_ratio_1d_nb(
        rets[from_i:to_i, col],
        bm_returns[from_i:to_i, col],
        ann_factor,
        period=period,
        log_returns=log_returns,
    )


@register_jitted(cache=True)
def up_capture_ratio_1d_nb(
    rets: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Capture ratio for periods when the benchmark return is positive."""
    if period is None:
        period = rets.shape[0]

    def _annualized_pos_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a > 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_pos_return(rets)
    annualized_bm_return = _annualized_pos_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None,
        log_returns=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def up_capture_ratio_nb(
    rets: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> tp.Array1d:
    """2-dim version of `up_capture_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = up_capture_ratio_1d_nb(
            rets[:, col],
            bm_returns[:, col],
            ann_factor,
            period=period,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def up_capture_ratio_rollmeta_nb(
    from_i: int,
    to_i: int,
    col: int,
    rets: tp.Array2d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Rolling apply meta function based on `up_capture_ratio_1d_nb`."""
    return up_capture_ratio_1d_nb(
        rets[from_i:to_i, col],
        bm_returns[from_i:to_i, col],
        ann_factor,
        period=period,
        log_returns=log_returns,
    )


@register_jitted(cache=True)
def down_capture_ratio_1d_nb(
    rets: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Capture ratio for periods when the benchmark return is negative."""
    if period is None:
        period = rets.shape[0]

    def _annualized_neg_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a < 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_neg_return(rets)
    annualized_bm_return = _annualized_neg_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="rets", axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None,
        log_returns=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def down_capture_ratio_nb(
    rets: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> tp.Array1d:
    """2-dim version of `down_capture_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = down_capture_ratio_1d_nb(
            rets[:, col],
            bm_returns[:, col],
            ann_factor,
            period=period,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def down_capture_ratio_rollmeta_nb(
    from_i: int,
    to_i: int,
    col: int,
    rets: tp.Array2d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    period: tp.Optional[float] = None,
    log_returns: bool = False,
) -> float:
    """Rolling apply meta function based on `down_capture_ratio_1d_nb`."""
    return down_capture_ratio_1d_nb(
        rets[from_i:to_i, col],
        bm_returns[from_i:to_i, col],
        ann_factor,
        period=period,
        log_returns=log_returns,
    )


# ############# Accumulators ############# #


@register_jitted(cache=True)
def rolling_sharpe_acc_nb(in_state: RollSharpeAIS) -> RollSharpeAOS:
    """Accumulator of `rolling_sharpe_nb`.

    Takes a state of type `vectorbtpro.returns.enums.RollSharpeAIS` and returns
    a state of type `vectorbtpro.returns.enums.RollSharpeAOS`."""
    mean_in_state = generic_enums.RollMeanAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
    )
    mean_out_state = generic_nb.rolling_mean_acc_nb(mean_in_state)
    std_in_state = generic_enums.RollStdAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        cumsum_sq=in_state.cumsum_sq,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
        ddof=in_state.ddof,
    )
    std_out_state = generic_nb.rolling_std_acc_nb(std_in_state)
    mean = mean_out_state.value
    std = std_out_state.value
    if std == 0:
        sharpe = np.nan
    else:
        sharpe = mean / std * np.sqrt(in_state.ann_factor)

    return RollSharpeAOS(
        cumsum=std_out_state.cumsum, cumsum_sq=std_out_state.cumsum_sq, nancnt=std_out_state.nancnt, value=sharpe
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(returns=ch.ArraySlicer(axis=1), window=None, ann_factor=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sharpe_ratio_nb(
    returns: tp.Array2d, window: int, ann_factor: float, minp: tp.Optional[int] = None, ddof: int = 0
) -> tp.Array2d:
    """Calculate rolling Sharpe ratio.

    Uses `rolling_sharpe_acc_nb` at each iteration."""
    if window is None:
        window = returns.shape[0]
    if minp is None:
        minp = window
    out = np.empty(returns.shape, dtype=np.float_)
    if returns.shape[0] == 0:
        return out
    for col in prange(returns.shape[1]):
        cumsum = 0.0
        cumsum_sq = 0.0
        nancnt = 0
        for i in range(returns.shape[0]):
            in_state = RollSharpeAIS(
                i=i,
                ret=returns[i, col],
                pre_window_ret=returns[i - window, col] if i - window >= 0 else np.nan,
                cumsum=cumsum,
                cumsum_sq=cumsum_sq,
                nancnt=nancnt,
                window=window,
                minp=minp,
                ddof=ddof,
                ann_factor=ann_factor,
            )
            out_state = rolling_sharpe_acc_nb(in_state)
            cumsum = out_state.cumsum
            cumsum_sq = out_state.cumsum_sq
            nancnt = out_state.nancnt
            out[i, col] = out_state.value

    return out
