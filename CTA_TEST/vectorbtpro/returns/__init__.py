# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules for working with returns.

Offers common financial risk and performance metrics as found in [empyrical](https://github.com/quantopian/empyrical),
an adapter for quantstats, and other features based on returns."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.returns.accessors import *
    from vectorbtpro.returns.nb import *
    from vectorbtpro.returns.qs_adapter import *

__exclude_from__all__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["qs_adapter"] = "quantstats"
