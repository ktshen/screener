# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions and config for evaluating indicator expressions."""

import math

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.generic.nb import (
    fshift_nb,
    diff_nb,
    rank_nb,
    rolling_sum_nb,
    rolling_mean_nb,
    rolling_std_nb,
    wm_mean_nb,
    rolling_rank_nb,
    rolling_prod_nb,
    rolling_min_nb,
    rolling_max_nb,
    rolling_argmin_nb,
    rolling_argmax_nb,
    rolling_cov_nb,
    rolling_corr_nb,
    demean_nb,
)
from vectorbtpro.indicators.nb import vwap_nb
from vectorbtpro.returns.nb import returns_nb
from vectorbtpro.utils.config import HybridConfig

__all__ = []


# ############# Delay ############# #


def delay(x: tp.Array2d, d: float) -> tp.Array2d:
    """Value of `x` `d` days ago."""
    return fshift_nb(x, math.floor(d))


def delta(x: tp.Array2d, d: float) -> tp.Array2d:
    """Today’s value of `x` minus the value of `x` `d` days ago."""
    return diff_nb(x, math.floor(d))


# ############# Cross-section ############# #


def cs_rescale(x: tp.Array2d) -> tp.Array2d:
    """Rescale `x` such that `sum(abs(x)) = 1`."""
    return (x.T / np.abs(x).sum(axis=1)).T


def cs_rank(x: tp.Array2d) -> tp.Array2d:
    """Rank cross-sectionally."""
    return rank_nb(x.T, pct=True).T


def cs_demean(x: tp.Array2d, g: tp.GroupByLike, context: tp.KwargsLike = None) -> tp.Array2d:
    """Demean `x` against groups `g` cross-sectionally."""
    group_map = Grouper(context["wrapper"].columns, g).get_group_map()
    return demean_nb(x, group_map)


# ############# Rolling ############# #


def ts_min(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling min."""
    return rolling_min_nb(x, math.floor(d))


def ts_max(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling max."""
    return rolling_max_nb(x, math.floor(d))


def ts_argmin(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling argmin."""
    return np.add(rolling_argmin_nb(x, math.floor(d), local=True), 1)


def ts_argmax(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling argmax."""
    return np.add(rolling_argmax_nb(x, math.floor(d), local=True), 1)


def ts_rank(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling rank."""
    return rolling_rank_nb(x, math.floor(d), pct=True)


def ts_sum(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling sum."""
    return rolling_sum_nb(x, math.floor(d))


def ts_product(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling product."""
    return rolling_prod_nb(x, math.floor(d))


def ts_mean(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling mean."""
    return rolling_mean_nb(x, math.floor(d))


def ts_wmean(x: tp.Array2d, d: float) -> tp.Array2d:
    """Weighted moving average over the past `d` days with linearly decaying weight."""
    return wm_mean_nb(x, math.floor(d))


def ts_std(x: tp.Array2d, d: float) -> tp.Array2d:
    """Return the rolling standard deviation."""
    return rolling_std_nb(x, math.floor(d))


def ts_corr(x: tp.Array2d, y: tp.Array2d, d: float) -> tp.Array2d:
    """Time-serial correlation of `x` and `y` for the past `d` days."""
    return rolling_corr_nb(x, y, math.floor(d))


def ts_cov(x: tp.Array2d, y: tp.Array2d, d: float) -> tp.Array2d:
    """Time-serial covariance of `x` and `y` for the past `d` days."""
    return rolling_cov_nb(x, y, math.floor(d))


def adv(d: float, context: tp.KwargsLike = None) -> tp.Array2d:
    """Average daily dollar volume for the past `d` days."""
    return ts_mean(context["volume"], math.floor(d))


# ############# Substitutions ############# #


def returns(context: tp.KwargsLike = None) -> tp.Array2d:
    """Daily close-to-close returns."""
    return returns_nb(context["close"])


def vwap(context: tp.KwargsLike = None) -> tp.Array2d:
    """VWAP."""
    if isinstance(context["wrapper"].index, pd.DatetimeIndex):
        group_lens = context["wrapper"].get_index_grouper("D").get_group_lens()
    else:
        group_lens = np.array([context["wrapper"].shape[0]])
    return vwap_nb(context["high"], context["low"], context["close"], context["volume"], group_lens)


def cap(context: tp.KwargsLike = None) -> tp.Array2d:
    """Market capitalization."""
    return context["close"] * context["volume"]


# ############# Configs ############# #

__pdoc__ = {}

expr_func_config = HybridConfig(
    dict(
        delay=dict(func=delay),
        delta=dict(func=delta),
        cs_rescale=dict(func=cs_rescale),
        cs_rank=dict(func=cs_rank),
        cs_demean=dict(func=cs_demean),
        ts_min=dict(func=ts_min),
        ts_max=dict(func=ts_max),
        ts_argmin=dict(func=ts_argmin),
        ts_argmax=dict(func=ts_argmax),
        ts_rank=dict(func=ts_rank),
        ts_sum=dict(func=ts_sum),
        ts_product=dict(func=ts_product),
        ts_mean=dict(func=ts_mean),
        ts_wmean=dict(func=ts_wmean),
        ts_std=dict(func=ts_std),
        ts_corr=dict(func=ts_corr),
        ts_cov=dict(func=ts_cov),
        adv=dict(func=adv, magnet_inputs=["volume"]),
    )
)
"""_"""

__pdoc__[
    "expr_func_config"
] = f"""Config for functions used in indicator expressions.

Can be modified.

```python
{expr_func_config.prettify()}
```
"""

expr_res_func_config = HybridConfig(
    dict(
        returns=dict(func=returns, magnet_inputs=["close"]),
        vwap=dict(func=vwap, magnet_inputs=["high", "low", "close", "volume"]),
        cap=dict(func=cap, magnet_inputs=["close", "volume"]),
    )
)
"""_"""

__pdoc__[
    "expr_res_func_config"
] = f"""Config for resolvable functions used in indicator expressions.

Can be modified.

```python
{expr_res_func_config.prettify()}
```
"""

wqa101_expr_config = HybridConfig(
    {
        1: "cs_rank(ts_argmax(power(where(returns < 0, ts_std(returns, 20), close), 2.), 5)) - 0.5",
        2: "-ts_corr(cs_rank(delta(log(volume), 2)), cs_rank((close - open) / open), 6)",
        3: "-ts_corr(cs_rank(open), cs_rank(volume), 10)",
        4: "-ts_rank(cs_rank(low), 9)",
        5: "cs_rank(open - (ts_sum(vwap, 10) / 10)) * (-abs(cs_rank(close - vwap)))",
        6: "-ts_corr(open, volume, 10)",
        7: "where(adv(20) < volume, (-ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)), -1)",
        8: "-cs_rank((ts_sum(open, 5) * ts_sum(returns, 5)) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))",
        9: (
            "where(0 < ts_min(delta(close, 1), 5), delta(close, 1), where(ts_max(delta(close, 1), 5) < 0, delta(close,"
            " 1), -delta(close, 1)))"
        ),
        10: (
            "cs_rank(where(0 < ts_min(delta(close, 1), 4), delta(close, 1), where(ts_max(delta(close, 1), 4) < 0,"
            " delta(close, 1), -delta(close, 1))))"
        ),
        11: "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(delta(volume, 3))",
        12: "sign(delta(volume, 1)) * (-delta(close, 1))",
        13: "-cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))",
        14: "(-cs_rank(delta(returns, 3))) * ts_corr(open, volume, 10)",
        15: "-ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)",
        16: "-cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))",
        17: (
            "((-cs_rank(ts_rank(close, 10))) * cs_rank(delta(delta(close, 1), 1))) * cs_rank(ts_rank(volume /"
            " adv(20), 5))"
        ),
        18: "-cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))",
        19: "(-sign((close - delay(close, 7)) + delta(close, 7))) * (1 + cs_rank(1 + ts_sum(returns, 250)))",
        20: "((-cs_rank(open - delay(high, 1))) * cs_rank(open - delay(close, 1))) * cs_rank(open - delay(low, 1))",
        21: (
            "where(((ts_sum(close, 8) / 8) + ts_std(close, 8)) < (ts_sum(close, 2) / 2), -1, where((ts_sum(close, 2) /"
            " 2) < ((ts_sum(close, 8) / 8) - ts_std(close, 8)), 1, where(volume / adv(20) >= 1, 1, -1)))"
        ),
        22: "-(delta(ts_corr(high, volume, 5), 5) * cs_rank(ts_std(close, 20)))",
        23: "where((ts_sum(high, 20) / 20) < high, -delta(high, 2), 0)",
        24: (
            "where((delta(ts_sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05, (-(close - ts_min(close, 100))),"
            " -delta(close, 3))"
        ),
        25: "cs_rank((((-returns) * adv(20)) * vwap) * (high - close))",
        26: "-ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
        27: "where(0.5 < cs_rank(ts_sum(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2) / 2.0), -1, 1)",
        28: "cs_rescale((ts_corr(adv(20), low, 5) + ((high + low) / 2)) - close)",
        29: (
            "ts_min(ts_product(cs_rank(cs_rank(cs_rescale(log(ts_sum(ts_min(cs_rank(cs_rank(-cs_rank(delta(close - 1,"
            " 5)))), 2), 1))))), 1), 5) + ts_rank(delay(-returns, 6), 5)"
        ),
        30: (
            "((1.0 - cs_rank((sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2))) +"
            " sign(delay(close, 2) - delay(close, 3)))) * ts_sum(volume, 5)) / ts_sum(volume, 20)"
        ),
        31: (
            "(cs_rank(cs_rank(cs_rank(ts_wmean(-cs_rank(cs_rank(delta(close, 10))), 10)))) + cs_rank(-delta(close, 3)))"
            " + sign(cs_rescale(ts_corr(adv(20), low, 12)))"
        ),
        32: "cs_rescale((ts_sum(close, 7) / 7) - close) + (20 * cs_rescale(ts_corr(vwap, delay(close, 5), 230)))",
        33: "cs_rank(-(1 - (open / close)))",
        34: "cs_rank((1 - cs_rank(ts_std(returns, 2) / ts_std(returns, 5))) + (1 - cs_rank(delta(close, 1))))",
        35: "(ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16))) * (1 - ts_rank(returns, 32))",
        36: (
            "((((2.21 * cs_rank(ts_corr(close - open, delay(volume, 1), 15))) + (0.7 * cs_rank(open - close))) + (0.73"
            " * cs_rank(ts_rank(delay(-returns, 6), 5)))) + cs_rank(abs(ts_corr(vwap, adv(20), 6)))) + (0.6 *"
            " cs_rank(((ts_sum(close, 200) / 200) - open) * (close - open)))"
        ),
        37: "cs_rank(ts_corr(delay(open - close, 1), close, 200)) + cs_rank(open - close)",
        38: "(-cs_rank(ts_rank(close, 10))) * cs_rank(close / open)",
        39: (
            "(-cs_rank(delta(close, 7) * (1 - cs_rank(ts_wmean(volume / adv(20), 9))))) * (1 + cs_rank(ts_sum(returns,"
            " 250)))"
        ),
        40: "(-cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
        41: "((high * low) ** 0.5) - vwap",
        42: "cs_rank(vwap - close) / cs_rank(vwap + close)",
        43: "ts_rank(volume / adv(20), 20) * ts_rank(-delta(close, 7), 8)",
        44: "-ts_corr(high, cs_rank(volume), 5)",
        45: (
            "-((cs_rank(ts_sum(delay(close, 5), 20) / 20) * ts_corr(close, volume, 2)) * cs_rank(ts_corr(ts_sum(close,"
            " 5), ts_sum(close, 20), 2)))"
        ),
        46: (
            "where(0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)), -1,"
            " where((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0, 1, -(close"
            " - delay(close, 1))))"
        ),
        47: (
            "(((cs_rank(1 / close) * volume) / adv(20)) * ((high * cs_rank(high - close)) / (ts_sum(high, 5) / 5))) -"
            " cs_rank(vwap - delay(vwap, 5))"
        ),
        48: (
            "cs_demean((ts_corr(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close,"
            " 'subindustry') / ts_sum((delta(close, 1) / delay(close, 1)) ** 2, 250)"
        ),
        49: (
            "where((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-0.1), 1,"
            " -(close - delay(close, 1)))"
        ),
        50: "-ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)",
        51: (
            "where((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-0.05), 1,"
            " -(close - delay(close, 1)))"
        ),
        52: (
            "(((-ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * cs_rank((ts_sum(returns, 240) - ts_sum(returns, 20)) /"
            " 220)) * ts_rank(volume, 5)"
        ),
        53: "-delta(((close - low) - (high - close)) / (close - low), 9)",
        54: "(-((low - close) * (open ** 5))) / ((low - high) * (close ** 5))",
        55: "-ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)",
        56: "0 - (1 * (cs_rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * cs_rank(returns * cap)))",
        57: "0 - (1 * ((close - vwap) / ts_wmean(cs_rank(ts_argmax(close, 30)), 2)))",
        58: "-ts_rank(ts_wmean(ts_corr(cs_demean(vwap, 'sector'), volume, 3.92795), 7.89291), 5.50322)",
        59: (
            "-ts_rank(ts_wmean(ts_corr(cs_demean((vwap * 0.728317) + (vwap * (1 - 0.728317)), 'industry'), volume,"
            " 4.25197), 16.2289), 8.19648)"
        ),
        60: (
            "0 - (1 * ((2 * cs_rescale(cs_rank((((close - low) - (high - close)) / (high - low)) * volume))) -"
            " cs_rescale(cs_rank(ts_argmax(close, 10)))))"
        ),
        61: "cs_rank(vwap - ts_min(vwap, 16.1219)) < cs_rank(ts_corr(vwap, adv(180), 17.9282))",
        62: (
            "(cs_rank(ts_corr(vwap, ts_sum(adv(20), 22.4101), 9.91009)) < cs_rank((cs_rank(open) + cs_rank(open)) <"
            " (cs_rank((high + low) / 2) + cs_rank(high)))) * (-1)"
        ),
        63: (
            "(cs_rank(ts_wmean(delta(cs_demean(close, 'industry'), 2.25164), 8.22237)) - cs_rank(ts_wmean(ts_corr((vwap"
            " * 0.318108) + (open * (1 - 0.318108)), ts_sum(adv(180), 37.2467), 13.557), 12.2883))) * (-1)"
        ),
        64: (
            "(cs_rank(ts_corr(ts_sum((open * 0.178404) + (low * (1 - 0.178404)), 12.7054), ts_sum(adv(120), 12.7054),"
            " 16.6208)) < cs_rank(delta((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741))) * (-1)"
        ),
        65: (
            "(cs_rank(ts_corr((open * 0.00817205) + (vwap * (1 - 0.00817205)), ts_sum(adv(60), 8.6911), 6.40374)) <"
            " cs_rank(open - ts_min(open, 13.635))) * (-1)"
        ),
        66: (
            "(cs_rank(ts_wmean(delta(vwap, 3.51013), 7.23052)) + ts_rank(ts_wmean((((low * 0.96633) + (low * (1 -"
            " 0.96633))) - vwap) / (open - ((high + low) / 2)), 11.4157), 6.72611)) * (-1)"
        ),
        67: (
            "(cs_rank(high - ts_min(high, 2.14593)) ** cs_rank(ts_corr(cs_demean(vwap, 'sector'), cs_demean(adv(20),"
            " 'subindustry'), 6.02936))) * (-1)"
        ),
        68: (
            "(ts_rank(ts_corr(cs_rank(high), cs_rank(adv(15)), 8.91644), 13.9333) < cs_rank(delta((close * 0.518371) +"
            " (low * (1 - 0.518371)), 1.06157))) * (-1)"
        ),
        69: (
            "(cs_rank(ts_max(delta(cs_demean(vwap, 'industry'), 2.72412), 4.79344)) ** ts_rank(ts_corr((close *"
            " 0.490655) + (vwap * (1 - 0.490655)), adv(20), 4.92416), 9.0615)) * (-1)"
        ),
        70: (
            "(cs_rank(delta(vwap, 1.29456)) ** ts_rank(ts_corr(cs_demean(close, 'industry'), adv(50), 17.8256),"
            " 17.9171)) * (-1)"
        ),
        71: (
            "maximum(ts_rank(ts_wmean(ts_corr(ts_rank(close, 3.43976), ts_rank(adv(180), 12.0647), 18.0175), 4.20501),"
            " 15.6948), ts_rank(ts_wmean(cs_rank((low + open) - (vwap + vwap)) ** 2, 16.4662), 4.4388))"
        ),
        72: (
            "cs_rank(ts_wmean(ts_corr((high + low) / 2, adv(40), 8.93345), 10.1519)) /"
            " cs_rank(ts_wmean(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011))"
        ),
        73: (
            "maximum(cs_rank(ts_wmean(delta(vwap, 4.72775), 2.91864)), ts_rank(ts_wmean((delta((open * 0.147155) + (low"
            " * (1 - 0.147155)), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * (-1), 3.33829), 16.7411)) *"
            " (-1)"
        ),
        74: (
            "(cs_rank(ts_corr(close, ts_sum(adv(30), 37.4843), 15.1365)) < cs_rank(ts_corr(cs_rank((high * 0.0261661) +"
            " (vwap * (1 - 0.0261661))), cs_rank(volume), 11.4791))) * (-1)"
        ),
        75: "cs_rank(ts_corr(vwap, volume, 4.24304)) < cs_rank(ts_corr(cs_rank(low), cs_rank(adv(50)), 12.4413))",
        76: (
            "maximum(cs_rank(ts_wmean(delta(vwap, 1.24383), 11.8259)), ts_rank(ts_wmean(ts_rank(ts_corr(cs_demean(low,"
            " 'sector'), adv(81), 8.14941), 19.569), 17.1543), 19.383)) * (-1)"
        ),
        77: (
            "minimum(cs_rank(ts_wmean((((high + low) / 2) + high) - (vwap + high), 20.0451)),"
            " cs_rank(ts_wmean(ts_corr((high + low) / 2, adv(40), 3.1614), 5.64125)))"
        ),
        78: (
            "cs_rank(ts_corr(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 19.7428), ts_sum(adv(40), 19.7428),"
            " 6.83313)) ** cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5.77492))"
        ),
        79: (
            "cs_rank(delta(cs_demean((close * 0.60733) + (open * (1 - 0.60733)), 'sector'), 1.23438)) <"
            " cs_rank(ts_corr(ts_rank(vwap, 3.60973), ts_rank(adv(150), 9.18637), 14.6644))"
        ),
        80: (
            "(cs_rank(sign(delta(cs_demean((open * 0.868128) + (high * (1 - 0.868128)), 'industry'), 4.04545))) **"
            " ts_rank(ts_corr(high, adv(10), 5.11456), 5.53756)) * (-1)"
        ),
        81: (
            "(cs_rank(log(ts_product(cs_rank(cs_rank(ts_corr(vwap, ts_sum(adv(10), 49.6054), 8.47743)) ** 4),"
            " 14.9655))) < cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5.07914))) * (-1)"
        ),
        82: (
            "minimum(cs_rank(ts_wmean(delta(open, 1.46063), 14.8717)), ts_rank(ts_wmean(ts_corr(cs_demean(volume,"
            " 'sector'), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * (-1)"
        ),
        83: (
            "(cs_rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) / (((high - low) /"
            " (ts_sum(close, 5) / 5)) / (vwap - close))"
        ),
        84: "power(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796))",
        85: (
            "cs_rank(ts_corr((high * 0.876703) + (close * (1 - 0.876703)), adv(30), 9.61331)) **"
            " cs_rank(ts_corr(ts_rank((high + low) / 2, 3.70596), ts_rank(volume, 10.1595), 7.11408))"
        ),
        86: (
            "(ts_rank(ts_corr(close, ts_sum(adv(20), 14.7444), 6.00049), 20.4195) < cs_rank((open + close) - (vwap +"
            " open))) * (-1)"
        ),
        87: (
            "maximum(cs_rank(ts_wmean(delta((close * 0.369701) + (vwap * (1 - 0.369701)), 1.91233), 2.65461)),"
            " ts_rank(ts_wmean(abs(ts_corr(cs_demean(adv(81), 'industry'), close, 13.4132)), 4.89768), 14.4535)) * (-1)"
        ),
        88: (
            "minimum(cs_rank(ts_wmean((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8.06882)),"
            " ts_rank(ts_wmean(ts_corr(ts_rank(close, 8.44728), ts_rank(adv(60), 20.6966), 8.01266), 6.65053),"
            " 2.61957))"
        ),
        89: (
            "ts_rank(ts_wmean(ts_corr((low * 0.967285) + (low * (1 - 0.967285)), adv(10), 6.94279), 5.51607), 3.79744)"
            " - ts_rank(ts_wmean(delta(cs_demean(vwap, 'industry'), 3.48158), 10.1466), 15.3012)"
        ),
        90: (
            "(cs_rank(close - ts_max(close, 4.66719)) ** ts_rank(ts_corr(cs_demean(adv(40), 'subindustry'), low,"
            " 5.38375), 3.21856)) * (-1)"
        ),
        91: (
            "(ts_rank(ts_wmean(ts_wmean(ts_corr(cs_demean(close, 'industry'), volume, 9.74928), 16.398), 3.83219),"
            " 4.8667) - cs_rank(ts_wmean(ts_corr(vwap, adv(30), 4.01303), 2.6809))) * (-1)"
        ),
        92: (
            "minimum(ts_rank(ts_wmean((((high + low) / 2) + close) < (low + open), 14.7221), 18.8683),"
            " ts_rank(ts_wmean(ts_corr(cs_rank(low), cs_rank(adv(30)), 7.58555), 6.94024), 6.80584))"
        ),
        93: (
            "ts_rank(ts_wmean(ts_corr(cs_demean(vwap, 'industry'), adv(81), 17.4193), 19.848), 7.54455) /"
            " cs_rank(ts_wmean(delta((close * 0.524434) + (vwap * (1 - 0.524434)), 2.77377), 16.2664))"
        ),
        94: (
            "(cs_rank(vwap - ts_min(vwap, 11.5783)) ** ts_rank(ts_corr(ts_rank(vwap, 19.6462), ts_rank(adv(60),"
            " 4.02992), 18.0926), 2.70756)) * (-1)"
        ),
        95: (
            "cs_rank(open - ts_min(open, 12.4105)) < ts_rank(cs_rank(ts_corr(ts_sum((high + low) / 2, 19.1351),"
            " ts_sum(adv(40), 19.1351), 12.8742)) ** 5, 11.7584)"
        ),
        96: (
            "maximum(ts_rank(ts_wmean(ts_corr(cs_rank(vwap), cs_rank(volume), 3.83878), 4.16783), 8.38151),"
            " ts_rank(ts_wmean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv(60), 4.13242), 3.65459),"
            " 12.6556), 14.0365), 13.4143)) * (-1)"
        ),
        97: (
            "(cs_rank(ts_wmean(delta(cs_demean((low * 0.721001) + (vwap * (1 - 0.721001)), 'industry'), 3.3705),"
            " 20.4523)) - ts_rank(ts_wmean(ts_rank(ts_corr(ts_rank(low, 7.87871), ts_rank(adv(60), 17.255), 4.97547),"
            " 18.5925), 15.7152), 6.71659)) * (-1)"
        ),
        98: (
            "cs_rank(ts_wmean(ts_corr(vwap, ts_sum(adv(5), 26.4719), 4.58418), 7.18088)) -"
            " cs_rank(ts_wmean(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(adv(15)), 20.8187), 8.62571), 6.95668),"
            " 8.07206))"
        ),
        99: (
            "(cs_rank(ts_corr(ts_sum((high + low) / 2, 19.8975), ts_sum(adv(60), 19.8975), 8.8136)) <"
            " cs_rank(ts_corr(low, volume, 6.28259))) * (-1)"
        ),
        100: (
            "0 - (1 * (((1.5 * cs_rescale(cs_demean(cs_demean(cs_rank((((close - low) - (high - close)) / (high - low))"
            " * volume), 'subindustry'), 'subindustry'))) - cs_rescale(cs_demean(ts_corr(close, cs_rank(adv(20)), 5) -"
            " cs_rank(ts_argmin(close, 30)), 'subindustry'))) * (volume / adv(20))))"
        ),
        101: "(close - open) / ((high - low) + .001)",
    }
)
"""_"""

__pdoc__[
    "wqa101_expr_config"
] = f"""Config with WorldQuant's 101 alpha expressions.

See [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991).

Can be modified.

```python
{wqa101_expr_config.prettify()}
```
"""
