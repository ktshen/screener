# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules with utilities that are used throughout the package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.utils.array_ import *
    from vectorbtpro.utils.attr_ import *
    from vectorbtpro.utils.caching import *
    from vectorbtpro.utils.checks import *
    from vectorbtpro.utils.chunking import *
    from vectorbtpro.utils.colors import *
    from vectorbtpro.utils.config import *
    from vectorbtpro.utils.cutting import *
    from vectorbtpro.utils.datetime_ import *
    from vectorbtpro.utils.datetime_nb import *
    from vectorbtpro.utils.decorators import *
    from vectorbtpro.utils.enum_ import *
    from vectorbtpro.utils.eval_ import *
    from vectorbtpro.utils.execution import *
    from vectorbtpro.utils.figure import *
    from vectorbtpro.utils.formatting import *
    from vectorbtpro.utils.hashing import *
    from vectorbtpro.utils.image_ import *
    from vectorbtpro.utils.jitting import *
    from vectorbtpro.utils.magic_decorators import *
    from vectorbtpro.utils.mapping import *
    from vectorbtpro.utils.math_ import *
    from vectorbtpro.utils.module_ import *
    from vectorbtpro.utils.params import *
    from vectorbtpro.utils.parsing import *
    from vectorbtpro.utils.path_ import *
    from vectorbtpro.utils.pbar import *
    from vectorbtpro.utils.pickling import *
    from vectorbtpro.utils.profiling import *
    from vectorbtpro.utils.random_ import *
    from vectorbtpro.utils.requests_ import *
    from vectorbtpro.utils.schedule_ import *
    from vectorbtpro.utils.tagging import *
    from vectorbtpro.utils.template import *

__import_if_installed__ = dict()
__import_if_installed__["figure"] = "plotly"
