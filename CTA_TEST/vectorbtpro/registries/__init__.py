# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules that register objects across vectorbtpro."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.registries.ca_registry import *
    from vectorbtpro.registries.ch_registry import *
    from vectorbtpro.registries.jit_registry import *
