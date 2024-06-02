# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for generic data.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of a backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    Rolling functions with `minp=None` have `min_periods` set to the window size.

    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in.

!!! warning
    Make sure to use `parallel=True` only if your columns are independent.
"""

from vectorbtpro.generic.nb.apply_reduce import *
from vectorbtpro.generic.nb.base import *
from vectorbtpro.generic.nb.iter_ import *
from vectorbtpro.generic.nb.patterns import *
from vectorbtpro.generic.nb.records import *
from vectorbtpro.generic.nb.rolling import *

__all__ = []
