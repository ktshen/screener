# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for generic data.

Defines enums and other schemas for `vectorbtpro.generic`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__pdoc__all__ = __all__ = [
    "BarZone",
    "WType",
    "RangeStatus",
    "InterpMode",
    "RescaleMode",
    "ErrorType",
    "DistanceMeasure",
    "OverlapMode",
    "DrawdownStatus",
    "range_dt",
    "pattern_range_dt",
    "drawdown_dt",
    "RollSumAIS",
    "RollSumAOS",
    "RollProdAIS",
    "RollProdAOS",
    "RollMeanAIS",
    "RollMeanAOS",
    "RollStdAIS",
    "RollStdAOS",
    "RollZScoreAIS",
    "RollZScoreAOS",
    "WMMeanAIS",
    "WMMeanAOS",
    "EWMMeanAIS",
    "EWMMeanAOS",
    "EWMStdAIS",
    "EWMStdAOS",
    "VidyaAIS",
    "VidyaAOS",
    "RollCovAIS",
    "RollCovAOS",
    "RollCorrAIS",
    "RollCorrAOS",
    "RollOLSAIS",
    "RollOLSAOS",
]

__pdoc__ = {}


# ############# Enums ############# #


class BarZoneT(tp.NamedTuple):
    Open: int = 0
    Middle: int = 1
    Close: int = 2


BarZone = BarZoneT()
"""_"""

__pdoc__[
    "BarZone"
] = f"""Bar zone.

```python
{prettify(BarZone)}
```
"""


class WTypeT(tp.NamedTuple):
    Simple: int = 0
    Weighted: int = 1
    Exp: int = 2
    Wilder: int = 3
    Vidya: int = 4


WType = WTypeT()
"""_"""

__pdoc__[
    "WType"
] = f"""Rolling window type.

```python
{prettify(WType)}
```
"""


class RangeStatusT(tp.NamedTuple):
    Open: int = 0
    Closed: int = 1


RangeStatus = RangeStatusT()
"""_"""

__pdoc__[
    "RangeStatus"
] = f"""Range status.

```python
{prettify(RangeStatus)}
```
"""


class InterpModeT(tp.NamedTuple):
    Linear: int = 0
    Nearest: int = 1
    Discrete: int = 2
    Mixed: int = 3


InterpMode = InterpModeT()
"""_"""

__pdoc__[
    "InterpMode"
] = f"""Interpolation mode.

```python
{prettify(InterpMode)}
```

Attributes:
    Line: Linear interpolation.

        For example: `[1.0, 2.0, 3.0]` -> `[1.0, 1.5, 2.0, 2.5, 3.0]`
    Nearest: Nearest-neighbor interpolation.

        For example: `[1.0, 2.0, 3.0]` -> `[1.0, 1.0, 2.0, 3.0, 3.0]`
    Discrete: Discrete interpolation.

        For example: `[1.0, 2.0, 3.0]` -> `[1.0, np.nan, 2.0, np.nan, 3.0]`
    Mixed: Mixed interpolation.

        For example: `[1.0, 2.0, 3.0]` -> `[1.0, 1.5, 2.0, 2.5, 3.0]`
"""


class RescaleModeT(tp.NamedTuple):
    MinMax: int = 0
    Rebase: int = 1
    Disable: int = 2


RescaleMode = RescaleModeT()
"""_"""

__pdoc__[
    "RescaleMode"
] = f"""Rescaling mode.

```python
{prettify(RescaleMode)}
```

Attributes:
    MinMax: Array is rescaled from its min-max range to the min-max range of another array.
    
        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` -> `[12.0, 11.0, 10.0]`
    
        Use this to search for patterns irrespective of their vertical scale.
    Rebase: Array is rebased to the first value in another array.
    
        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` -> `[10.0, 6.6, 3.3]`
    
        Use this to search for percentage changes.
    Disable: Disable any rescaling.
    
        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` -> `[3.0, 2.0, 1.0]`
    
        Use this to search for particular numbers.
"""


class ErrorTypeT(tp.NamedTuple):
    Absolute: int = 0
    Relative: int = 1


ErrorType = ErrorTypeT()
"""_"""

__pdoc__[
    "ErrorType"
] = f"""Error type.

```python
{prettify(ErrorType)}
```

Attributes:
    Absolute: Absolute error, that is, `x1 - x0`.
    Relative: Relative error, that is, `(x1 - x0) / x0`.
"""


class DistanceMeasureT(tp.NamedTuple):
    MAE: int = 0
    MSE: int = 1
    RMSE: int = 2


DistanceMeasure = DistanceMeasureT()
"""_"""

__pdoc__[
    "DistanceMeasure"
] = f"""Distance measure.

```python
{prettify(DistanceMeasure)}
```

Attributes:
    MAE: Mean absolute error.
    MSE: Mean squared error.
    RMSE: Root mean squared error.
"""


class OverlapModeT(tp.NamedTuple):
    AllowAll: int = -2
    Allow: int = -1
    Disallow: int = 0


OverlapMode = OverlapModeT()
"""_"""

__pdoc__[
    "OverlapMode"
] = f"""Overlapping mode.

```python
{prettify(OverlapMode)}
```

Attributes:
    AllowAll: Allow any overlapping ranges, even if they start at the same row.
    Allow: Allow overlapping ranges, but only if they do not start at the same row.
    Disallow: Disallow any overlapping ranges.
    
Any other positive number will check whether the intersection of each two consecutive ranges is 
bigger than that number of rows, and if so, the range with the highest similarity will be selected.
"""


class DrawdownStatusT(tp.NamedTuple):
    Active: int = 0
    Recovered: int = 1


DrawdownStatus = DrawdownStatusT()
"""_"""

__pdoc__[
    "DrawdownStatus"
] = f"""Drawdown status.

```python
{prettify(DrawdownStatus)}
```
"""

# ############# Records ############# #

range_dt = np.dtype(
    [("id", np.int_), ("col", np.int_), ("start_idx", np.int_), ("end_idx", np.int_), ("status", np.int_)],
    align=True,
)
"""_"""

__pdoc__[
    "range_dt"
] = f"""`np.dtype` of range records.

```python
{prettify(range_dt)}
```
"""

pattern_range_dt = np.dtype(
    [
        ("id", np.int_),
        ("col", np.int_),
        ("start_idx", np.int_),
        ("end_idx", np.int_),
        ("status", np.int_),
        ("similarity", np.float_),
    ],
    align=True,
)
"""_"""

__pdoc__[
    "pattern_range_dt"
] = f"""`np.dtype` of pattern range records.

```python
{prettify(pattern_range_dt)}
```
"""

drawdown_dt = np.dtype(
    [
        ("id", np.int_),
        ("col", np.int_),
        ("peak_idx", np.int_),
        ("start_idx", np.int_),
        ("valley_idx", np.int_),
        ("end_idx", np.int_),
        ("peak_val", np.float_),
        ("valley_val", np.float_),
        ("end_val", np.float_),
        ("status", np.int_),
    ],
    align=True,
)
"""_"""

__pdoc__[
    "drawdown_dt"
] = f"""`np.dtype` of drawdown records.

```python
{prettify(drawdown_dt)}
```
"""


# ############# States ############# #


class RollSumAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollSumAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""


class RollSumAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollSumAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""


class RollProdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumprod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollProdAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""


class RollProdAOS(tp.NamedTuple):
    cumprod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollProdAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""


class RollMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollMeanAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""


class RollMeanAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollMeanAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""


class RollStdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__[
    "RollStdAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""


class RollStdAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollStdAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""


class RollZScoreAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__[
    "RollZScoreAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_zscore_acc_nb`."""


class RollZScoreAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollZScoreAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_zscore_acc_nb`."""


class WMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    wcumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "WMMeanAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""


class WMMeanAOS(tp.NamedTuple):
    cumsum: float
    wcumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "WMMeanAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""


class EWMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    old_wt: float
    weighted_avg: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMMeanAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`.

To get `alpha`, use one of the following:

* `vectorbtpro.generic.nb.rolling.alpha_from_com_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_span_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_halflife_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_wilder_nb`"""


class EWMMeanAOS(tp.NamedTuple):
    old_wt: float
    weighted_avg: float
    nobs: int
    value: float


__pdoc__[
    "EWMMeanAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`."""


class EWMStdAIS(tp.NamedTuple):
    i: int
    value: float
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMStdAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`.

For tips on `alpha`, see `EWMMeanAIS`."""


class EWMStdAOS(tp.NamedTuple):
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    value: float


__pdoc__[
    "EWMStdAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`."""


class VidyaAIS(tp.NamedTuple):
    i: int
    prev_value: float
    value: float
    pre_window_prev_value: float
    pre_window_value: float
    pos_cumsum: float
    neg_cumsum: float
    prev_vidya: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "VidyaAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.vidya_acc_nb`."""


class VidyaAOS(tp.NamedTuple):
    pos_cumsum: float
    neg_cumsum: float
    nancnt: int
    window_len: int
    cmo: float
    vidya: float


__pdoc__[
    "VidyaAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.vidya_acc_nb`."""


class RollCovAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__[
    "RollCovAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""


class RollCovAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollCovAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""


class RollCorrAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollCorrAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""


class RollCorrAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollCorrAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""


class RollOLSAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    validcnt: int
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollOLSAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.generic.nb.rolling.rolling_ols_acc_nb`."""


class RollOLSAOS(tp.NamedTuple):
    validcnt: int
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    slope_value: float
    intercept_value: float


__pdoc__[
    "RollOLSAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.generic.nb.rolling.rolling_ols_acc_nb`."""
