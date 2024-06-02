# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for working with allocation records."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import ReadonlyConfig, Config
from vectorbtpro.records.base import Records
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.records.decorators import override_field_config
from vectorbtpro.portfolio.enums import alloc_range_dt, alloc_point_dt

__all__ = [
    "AllocRanges",
    "AllocPoints",
]

__pdoc__ = {}

# ############# AllocRanges ############# #

alloc_ranges_field_config = ReadonlyConfig(
    dict(
        dtype=alloc_range_dt,
        settings={
            "idx": dict(name="alloc_idx"),  # remap field of Records
            "col": dict(title="Group", mapping="groups", group_indexing=True),  # remap field of Records
            "alloc_idx": dict(title="Allocation Index", mapping="index"),
        },
    )
)
"""_"""

__pdoc__[
    "alloc_ranges_field_config"
] = f"""Field config for `AllocRanges`.

```python
{alloc_ranges_field_config.prettify()}
```
"""

AllocRangesT = tp.TypeVar("AllocRangesT", bound="AllocRanges")


@override_field_config(alloc_ranges_field_config)
class AllocRanges(Ranges):
    """Extends `vectorbtpro.records.base.Records` for working with allocation point records."""

    @property
    def field_config(self) -> Config:
        return self._field_config


AllocRanges.override_field_config_doc(__pdoc__)


# ############# AllocPoints ############# #

alloc_points_field_config = ReadonlyConfig(
    dict(
        dtype=alloc_point_dt,
        settings={
            "idx": dict(name="alloc_idx"),  # remap field of Records
            "col": dict(title="Group", mapping="groups", group_indexing=True),  # remap field of Records
            "alloc_idx": dict(title="Allocation Index", mapping="index"),
        },
    )
)
"""_"""

__pdoc__[
    "alloc_points_field_config"
] = f"""Field config for `AllocRanges`.

```python
{alloc_points_field_config.prettify()}
```
"""

AllocPointsT = tp.TypeVar("AllocPointsT", bound="AllocPoints")


@override_field_config(alloc_points_field_config)
class AllocPoints(Records):
    """Extends `vectorbtpro.generic.ranges.Ranges` for working with allocation range records."""

    @property
    def field_config(self) -> Config:
        return self._field_config


AllocPoints.override_field_config_doc(__pdoc__)
