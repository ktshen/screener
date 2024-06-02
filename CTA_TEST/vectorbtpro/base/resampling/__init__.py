# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules with classes and utilities for resampling."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.base.resampling.base import *
    from vectorbtpro.base.resampling.nb import *
