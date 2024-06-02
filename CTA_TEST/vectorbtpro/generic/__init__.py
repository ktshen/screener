# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, focuses on the data itself."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.generic.nb import *
    from vectorbtpro.generic.splitting import *
    from vectorbtpro.generic.accessors import *
    from vectorbtpro.generic.analyzable import *
    from vectorbtpro.generic.decorators import *
    from vectorbtpro.generic.drawdowns import *
    from vectorbtpro.generic.plots_builder import *
    from vectorbtpro.generic.plotting import *
    from vectorbtpro.generic.price_records import *
    from vectorbtpro.generic.ranges import *
    from vectorbtpro.generic.stats_builder import *

__exclude_from__all__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["plotting"] = "plotly"
