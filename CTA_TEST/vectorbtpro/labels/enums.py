# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for label generation.

Defines enums and other schemas for `vectorbtpro.labels`."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__pdoc__all__ = __all__ = ["TrendLabelMode"]

__pdoc__ = {}


class TrendLabelModeT(tp.NamedTuple):
    Binary: int = 0
    BinaryCont: int = 1
    BinaryContSat: int = 2
    PctChange: int = 3
    PctChangeNorm: int = 4


TrendLabelMode = TrendLabelModeT()
"""_"""

__pdoc__[
    "TrendLabelMode"
] = f"""Trend label mode.

```python
{prettify(TrendLabelMode)}
```

Attributes:
    Binary: See `vectorbtpro.labels.nb.bn_trend_labels_nb`.
    BinaryCont: See `vectorbtpro.labels.nb.bn_cont_trend_labels_nb`.
    BinaryContSat: See `vectorbtpro.labels.nb.bn_cont_sat_trend_labels_nb`.
    PctChange: See `vectorbtpro.labels.nb.pct_trend_labels_nb`.
    PctChangeNorm: See `vectorbtpro.labels.nb.pct_trend_labels_nb` with `normalize` set to True.
"""
