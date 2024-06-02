# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class for analyzing data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.stats_builder import StatsBuilderMixin

__all__ = [
    "Analyzable",
]


class MetaAnalyzable(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    pass


AnalyzableT = tp.TypeVar("AnalyzableT", bound="Analyzable")


class Analyzable(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaAnalyzable):
    """Class that can be analyzed by computing and plotting attributes of any kind."""

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, **kwargs)
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)
