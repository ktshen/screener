# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules for plotting with Plotly Express."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.px.accessors import *

__import_if_installed__ = dict()
__import_if_installed__["accessors"] = "plotly"
