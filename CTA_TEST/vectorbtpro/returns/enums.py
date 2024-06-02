# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for returns."""

from vectorbtpro import _typing as tp

__pdoc__all__ = __all__ = [
    "RollSharpeAIS",
    "RollSharpeAOS",
]

__pdoc__ = {}


# ############# States ############# #


class RollSharpeAIS(tp.NamedTuple):
    i: int
    ret: float
    pre_window_ret: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int
    ann_factor: float


__pdoc__[
    "RollSharpeAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.returns.nb.rolling_sharpe_acc_nb`."""


class RollSharpeAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    value: float


__pdoc__[
    "RollSharpeAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.returns.nb.rolling_sharpe_acc_nb`."""
