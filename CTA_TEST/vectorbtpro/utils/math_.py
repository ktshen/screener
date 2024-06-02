# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Math utilities."""

import numpy as np

from vectorbtpro._settings import settings
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []

_use_tol = settings["math"]["use_tol"]
_rel_tol = settings["math"]["rel_tol"]
_abs_tol = settings["math"]["abs_tol"]
_use_round = settings["math"]["use_round"]
_decimals = settings["math"]["decimals"]


@register_jitted(cache=True)
def is_close_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Tell whether two values are approximately equal."""
    if np.isnan(a) or np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return False
    if a == b:
        return True
    return use_tol and abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@register_jitted(cache=True)
def is_close_or_less_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Tell whether the first value is approximately less than or equal to the second value."""
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a < b


@register_jitted(cache=True)
def is_less_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Tell whether the first value is approximately less than the second value."""
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b


@register_jitted(cache=True)
def is_addition_zero_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Tell whether addition of two values yields zero."""
    if use_tol:
        if np.sign(a) != np.sign(b):
            return is_close_nb(abs(a), abs(b), rel_tol=rel_tol, abs_tol=abs_tol)
        return is_close_nb(a + b, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == -b


@register_jitted(cache=True)
def add_nb(a: float, b: float, use_tol: bool = _use_tol, rel_tol: float = _rel_tol, abs_tol: float = _abs_tol) -> float:
    """Add two floats."""
    if use_tol and is_addition_zero_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return 0.0
    return a + b


@register_jitted(cache=True)
def round_nb(a: float, use_round: bool = _use_round, decimals: int = _decimals) -> float:
    """Round a float to a number of decimals."""
    if use_round:
        return round(a, decimals)
    return a
