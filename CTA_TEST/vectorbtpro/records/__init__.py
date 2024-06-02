# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules for working with records.

Records are the second form of data representation in vectorbtpro. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.records.base import *
    from vectorbtpro.records.chunking import *
    from vectorbtpro.records.col_mapper import *
    from vectorbtpro.records.mapped_array import *
    from vectorbtpro.records.nb import *
