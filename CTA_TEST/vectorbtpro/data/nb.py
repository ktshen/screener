# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for generating data.

Provides an arsenal of Numba-compiled functions that are used to generate data.
These only accept NumPy arrays and other Numba-compatible types."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb

__all__ = []


@register_jitted(cache=True)
def generate_random_data_1d_nb(
    n_rows: int,
    start_value: float = 100.0,
    mean: float = 0.0,
    std: float = 0.01,
    symmetric: bool = False,
) -> tp.Array1d:
    """Generate data using cumulative product of returns drawn from normal (Gaussian) distribution.

    Turn on `symmetric` to diminish negative returns and make them symmetric to positive ones.
    Otherwise, the majority of generated paths will go downward."""
    out = np.empty(n_rows, dtype=np.float_)

    for i in range(n_rows):
        if i == 0:
            prev_value = start_value
        else:
            prev_value = out[i - 1]
        return_ = np.random.normal(mean, std)
        if symmetric and return_ < 0:
            return_ = -abs(return_) / (1 + abs(return_))
        out[i] = prev_value * (1 + return_)

    return out


@register_jitted(cache=True, tags={"can_parallel"})
def generate_random_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    symmetric: tp.FlexArray1dLike = False,
) -> tp.Array2d:
    """2-dim version of `generate_random_data_1d_nb`.

    Each argument can be provided per column thanks to flexible indexing."""
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    symmetric_ = to_1d_array_nb(np.asarray(symmetric))

    out = np.empty(shape, dtype=np.float_)

    for col in prange(shape[1]):
        out[:, col] = generate_random_data_1d_nb(
            shape[0],
            start_value=flex_select_1d_pc_nb(start_value_, col),
            mean=flex_select_1d_pc_nb(mean_, col),
            std=flex_select_1d_pc_nb(std_, col),
            symmetric=flex_select_1d_pc_nb(symmetric_, col),
        )

    return out


@register_jitted(cache=True)
def generate_gbm_data_1d_nb(
    n_rows: int,
    start_value: float = 100.0,
    mean: float = 0.0,
    std: float = 0.01,
    dt: float = 1.0,
) -> tp.Array2d:
    """Generate data using Geometric Brownian Motion (GBM)."""
    out = np.empty(n_rows, dtype=np.float_)

    for i in range(n_rows):
        if i == 0:
            prev_value = start_value
        else:
            prev_value = out[i - 1]
        rand = np.random.standard_normal()
        out[i] = prev_value * np.exp((mean - 0.5 * std ** 2) * dt + std * np.sqrt(dt) * rand)

    return out


@register_jitted(cache=True, tags={"can_parallel"})
def generate_gbm_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    dt: tp.FlexArray1dLike = 1.0,
) -> tp.Array2d:
    """2-dim version of `generate_gbm_data_1d_nb`.

    Each argument can be provided per column thanks to flexible indexing."""
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    dt_ = to_1d_array_nb(np.asarray(dt))

    out = np.empty(shape, dtype=np.float_)

    for col in prange(shape[1]):
        out[:, col] = generate_gbm_data_1d_nb(
            shape[0],
            start_value=flex_select_1d_pc_nb(start_value_, col),
            mean=flex_select_1d_pc_nb(mean_, col),
            std=flex_select_1d_pc_nb(std_, col),
            dt=flex_select_1d_pc_nb(dt_, col),
        )

    return out
