# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Basic look-ahead indicators and label generators.

You can access all the indicators either by `vbt.*` or `vbt.labels.*`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.labels.enums import TrendLabelMode
from vectorbtpro.generic import enums as generic_enums

__all__ = [
    "FMEAN",
    "FSTD",
    "FMIN",
    "FMAX",
    "FIXLB",
    "MEANLB",
    "LEXLB",
    "TRENDLB",
    "BOLB",
]

# ############# Look-ahead indicators ############# #

FMEAN = IndicatorFactory(
    class_name="FMEAN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["fmean"],
).with_apply_func(
    nb.future_mean_nb,
    kwargs_as_args=["wait", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    wtype="simple",
    wait=1,
    adjust=False,
)

FMEAN.__doc__ = """Look-ahead indicator based on `vectorbtpro.labels.nb.future_mean_nb`."""

FSTD = IndicatorFactory(
    class_name="FSTD",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["fstd"],
).with_apply_func(
    nb.future_std_nb,
    kwargs_as_args=["wait", "adjust", "ddof"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    wtype="simple",
    wait=1,
    adjust=False,
    ddof=0,
)

FSTD.__doc__ = """Look-ahead indicator based on `vectorbtpro.labels.nb.future_std_nb`."""

FMIN = IndicatorFactory(
    class_name="FMIN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window"],
    output_names=["fmin"],
).with_apply_func(
    nb.future_min_nb,
    kwargs_as_args=["wait"],
    wait=1,
)

FMIN.__doc__ = """Look-ahead indicator based on `vectorbtpro.labels.nb.future_min_nb`."""

FMAX = IndicatorFactory(
    class_name="FMAX",
    module_name=__name__,
    input_names=["close"],
    param_names=["window"],
    output_names=["fmax"],
).with_apply_func(
    nb.future_max_nb,
    kwargs_as_args=["wait"],
    wait=1,
)

FMAX.__doc__ = """Look-ahead indicator based on `vectorbtpro.labels.nb.future_max_nb`."""


# ############# Label generators ############# #


def _plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
    """Plot `close` and overlay it with the heatmap of `labels`.

    `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.overlay_with_heatmap`."""
    self_col = self.select_col(column=column, group_by=False)

    return self_col.close.rename("close").vbt.overlay_with_heatmap(self_col.labels.rename("labels"), **kwargs)


FIXLB = IndicatorFactory(
    class_name="FIXLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["n"],
    output_names=["labels"],
).with_apply_func(
    nb.fixed_labels_nb,
)


class _FIXLB(FIXLB):
    """Label generator based on `vectorbtpro.labels.nb.fixed_labels_nb`."""

    plot = _plot


setattr(FIXLB, "__doc__", _FIXLB.__doc__)
setattr(FIXLB, "plot", _FIXLB.plot)

MEANLB = IndicatorFactory(
    class_name="MEANLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["labels"],
).with_apply_func(
    nb.mean_labels_nb,
    kwargs_as_args=["wait", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    wtype="simple",
    wait=1,
    adjust=False,
)


class _MEANLB(MEANLB):
    """Label generator based on `vectorbtpro.labels.nb.mean_labels_nb`."""

    plot = _plot


setattr(MEANLB, "__doc__", _MEANLB.__doc__)
setattr(MEANLB, "plot", _MEANLB.plot)

LEXLB = IndicatorFactory(
    class_name="LEXLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["up_th", "down_th"],
    output_names=["labels"],
).with_apply_func(
    nb.local_extrema_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
)


class _LEXLB(LEXLB):
    """Label generator based on `vectorbtpro.labels.nb.local_extrema_nb`."""

    plot = _plot


setattr(LEXLB, "__doc__", _LEXLB.__doc__)
setattr(LEXLB, "plot", _LEXLB.plot)

TRENDLB = IndicatorFactory(
    class_name="TRENDLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["up_th", "down_th", "mode"],
    output_names=["labels"],
).with_apply_func(
    nb.trend_labels_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
        mode=dict(
            dtype=TrendLabelMode,
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    mode=TrendLabelMode.Binary,
)


class _TRENDLB(TRENDLB):
    """Label generator based on `vectorbtpro.labels.nb.trend_labels_nb`."""

    plot = _plot


setattr(TRENDLB, "__doc__", _TRENDLB.__doc__)
setattr(TRENDLB, "plot", _TRENDLB.plot)

BOLB = IndicatorFactory(
    class_name="BOLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "up_th", "down_th"],
    output_names=["labels"],
).with_apply_func(
    nb.breakout_labels_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
    kwargs_as_args=["wait"],
    up_th=0.0,
    down_th=0.0,
    wait=1,
)


class _BOLB(BOLB):
    """Label generator based on `vectorbtpro.labels.nb.breakout_labels_nb`."""

    plot = _plot


setattr(BOLB, "__doc__", _BOLB.__doc__)
setattr(BOLB, "plot", _BOLB.plot)
