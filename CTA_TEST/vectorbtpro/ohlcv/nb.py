# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for OHLCV.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0)."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_pr_nb

__all__ = []


@register_jitted(cache=True)
def ohlc_every_1d_nb(price: tp.Array1d, n: tp.FlexArray1dLike) -> tp.Array2d:
    """Aggregate every `n` price points into an OHLC point."""
    n_ = to_1d_array_nb(np.asarray(n))
    out = np.empty((price.shape[0], 4), dtype=np.float_)
    vmin = np.inf
    vmax = -np.inf
    k = 0
    start_i = 0
    for i in range(price.shape[0]):
        _n = flex_select_1d_pr_nb(n_, k)
        if _n <= 0:
            out[k, 0] = np.nan
            out[k, 1] = np.nan
            out[k, 2] = np.nan
            out[k, 3] = np.nan
            vmin = np.inf
            vmax = -np.inf
            if i < price.shape[0] - 1:
                k = k + 1
            continue
        if price[i] < vmin:
            vmin = price[i]
        if price[i] > vmax:
            vmax = price[i]
        if i == start_i:
            out[k, 0] = price[i]
        if i == start_i + _n - 1 or i == price.shape[0] - 1:
            out[k, 1] = vmax
            out[k, 2] = vmin
            out[k, 3] = price[i]
            vmin = np.inf
            vmax = -np.inf
            if i < price.shape[0] - 1:
                k = k + 1
                start_i = start_i + _n
    return out[: k + 1]
