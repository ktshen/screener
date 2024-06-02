# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom Pandas accessors for OHLC(V) data.

Methods can be accessed as follows:

* `OHLCVDFAccessor` -> `pd.DataFrame.vbt.ohlcv.*`

The accessors inherit `vectorbtpro.generic.accessors`.

!!! note
    Accessors do not utilize caching.

## Column names

By default, vectorbt searches for columns with names 'open', 'high', 'low', 'close', and 'volume'
(case doesn't matter). You can change the naming either using `column_names` in
`vectorbtpro._settings.ohlcv`, or by providing `column_names` directly to the accessor.

```pycon
>>> import pandas as pd
>>> import vectorbtpro as vbt

>>> df = pd.DataFrame({
...     'my_open1': [2, 3, 4, 3.5, 2.5],
...     'my_high2': [3, 4, 4.5, 4, 3],
...     'my_low3': [1.5, 2.5, 3.5, 2.5, 1.5],
...     'my_close4': [2.5, 3.5, 4, 3, 2],
...     'my_volume5': [10, 11, 10, 9, 10]
... })

>>> # vectorbt can't find columns
>>> df.vbt.ohlcv.get_column('open')
None

>>> my_column_names = dict(
...     open='my_open1',
...     high='my_high2',
...     low='my_low3',
...     close='my_close4',
...     volume='my_volume5',
... )
>>> ohlcv_acc = df.vbt.ohlcv(freq='d', column_names=my_column_names)
>>> ohlcv_acc.get_column('open')
0    2.0
1    3.0
2    4.0
3    3.5
4    2.5
Name: my_open1, dtype: float64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `OHLCVDFAccessor.metrics`.

```pycon
>>> ohlcv_acc.stats()
Start                           0
End                             4
Period            5 days 00:00:00
First Price                   2.0
Lowest Price                  1.5
Highest Price                 4.5
Last Price                    2.0
First Volume                   10
Lowest Volume                   9
Highest Volume                 11
Last Volume                    10
Name: agg_stats, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `OHLCVDFAccessor.subplots`.

`OHLCVDFAccessor` class has a single subplot based on `OHLCVDFAccessor.plot` (without volume):

```pycon
>>> ohlcv_acc.plots(settings=dict(ohlc_type='candlestick')).show()
```

![](/assets/images/api/ohlcv_plots.svg){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.accessors import register_df_vbt_accessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig

__all__ = [
    "OHLCVDFAccessor",
]

__pdoc__ = {}


OHLCVDFAccessorT = tp.TypeVar("OHLCVDFAccessorT", bound="OHLCVDFAccessor")


@register_df_vbt_accessor("ohlcv")
class OHLCVDFAccessor(GenericDFAccessor):
    """Accessor on top of OHLCV data. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.ohlcv`."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (GenericDFAccessor._expected_keys or set()) | {
        "column_names",
    }

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        column_names: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, column_names=column_names, **kwargs)

        self._column_names = column_names

    @property
    def column_names(self) -> tp.Kwargs:
        """Column names."""
        from vectorbtpro._settings import settings

        ohlcv_cfg = settings["ohlcv"]

        return merge_dicts(ohlcv_cfg["column_names"], self._column_names)

    def get_column(self, col_name: str) -> tp.Optional[tp.Series]:
        """Get column from `OHLCVDFAccessor.column_names`."""
        df_column_names = self.obj.columns.str.lower().tolist()
        col_name = self.column_names[col_name].lower()
        if col_name not in df_column_names:
            return None
        return self.obj.iloc[:, df_column_names.index(col_name)]

    @property
    def open(self) -> tp.Optional[tp.Series]:
        """Open."""
        return self.get_column("open")

    @property
    def high(self) -> tp.Optional[tp.Series]:
        """High."""
        return self.get_column("high")

    @property
    def low(self) -> tp.Optional[tp.Series]:
        """Low."""
        return self.get_column("low")

    @property
    def close(self) -> tp.Optional[tp.Series]:
        """Close."""
        return self.get_column("close")

    @property
    def volume(self) -> tp.Optional[tp.Series]:
        """Volume."""
        return self.get_column("volume")

    @property
    def hlc3(self) -> tp.Optional[tp.Series]:
        """HLC/3."""
        return (self.high + self.low + self.close) / 3

    @property
    def ohlc4(self) -> tp.Optional[tp.Series]:
        """OHLC/4."""
        return (self.open + self.high + self.low + self.close) / 4

    @property
    def ohlc(self) -> tp.Optional[tp.Frame]:
        """Open, high, low, and close series."""
        to_concat = []
        if self.open is not None:
            to_concat.append(self.open)
        if self.high is not None:
            to_concat.append(self.high)
        if self.low is not None:
            to_concat.append(self.low)
        if self.close is not None:
            to_concat.append(self.close)
        if len(to_concat) == 0:
            return None
        return pd.concat(to_concat, axis=1)

    # ############# Resampling ############# #

    def resample(self: OHLCVDFAccessorT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> OHLCVDFAccessorT:
        """Perform resampling on `OHLCVDFAccessor`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        sr_dct = {}
        for column in self.obj.columns:
            found = False
            for k, v in self.column_names.items():
                if column.lower() == v.lower():
                    if k == "open":
                        sr_dct[column] = self.obj[column].vbt.resample_apply(
                            wrapper_meta["resampler"],
                            generic_nb.first_reduce_nb,
                        )
                    elif k == "high":
                        sr_dct[column] = self.obj[column].vbt.resample_apply(
                            wrapper_meta["resampler"],
                            generic_nb.max_reduce_nb,
                        )
                    elif k == "low":
                        sr_dct[column] = self.obj[column].vbt.resample_apply(
                            wrapper_meta["resampler"],
                            generic_nb.min_reduce_nb,
                        )
                    elif k == "close":
                        sr_dct[column] = self.obj[column].vbt.resample_apply(
                            wrapper_meta["resampler"],
                            generic_nb.last_reduce_nb,
                        )
                    elif k == "volume":
                        sr_dct[column] = self.obj[column].vbt.resample_apply(
                            wrapper_meta["resampler"],
                            generic_nb.sum_reduce_nb,
                        )
                    else:
                        raise ValueError(f"Unknown key '{k}' in column_names")
                    found = True
                    break
            if not found:
                raise ValueError(f"Cannot match column '{column}'")
        new_obj = pd.DataFrame(sr_dct)
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            obj=new_obj,
        )

    @property
    def drawdowns(self) -> Drawdowns:
        """`OHLCVDFAccessor.get_drawdowns` with default arguments."""
        return self.get_drawdowns()

    def get_drawdowns(self, **kwargs) -> Drawdowns:
        """Generate drawdown records.

        See `vectorbtpro.generic.drawdowns.Drawdowns`."""
        return Drawdowns.from_price(
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            **kwargs,
        )

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `OHLCVDFAccessor.stats`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults` and
        `stats` from `vectorbtpro._settings.ohlcv`."""
        from vectorbtpro._settings import settings

        ohlcv_stats_cfg = settings["ohlcv"]["stats"]

        return merge_dicts(GenericAccessor.stats_defaults.__get__(self), ohlcv_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Start",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end=dict(
                title="End",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            first_price=dict(
                title="First Price",
                calc_func=lambda ohlc: generic_nb.bfill_1d_nb(ohlc.values.flatten())[0],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            lowest_price=dict(
                title="Lowest Price",
                calc_func=lambda ohlc: ohlc.values.min(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            highest_price=dict(
                title="Highest Price",
                calc_func=lambda ohlc: ohlc.values.max(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            last_price=dict(
                title="Last Price",
                calc_func=lambda ohlc: generic_nb.ffill_1d_nb(ohlc.values.flatten())[-1],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            first_volume=dict(
                title="First Volume",
                calc_func=lambda volume: generic_nb.bfill_1d_nb(volume.values)[0],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            lowest_volume=dict(
                title="Lowest Volume",
                calc_func=lambda volume: volume.values.min(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            highest_volume=dict(
                title="Highest Volume",
                calc_func=lambda volume: volume.values.max(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            last_volume=dict(
                title="Last Volume",
                calc_func=lambda volume: generic_nb.ffill_1d_nb(volume.values)[-1],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        plot_volume: tp.Optional[bool] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        volume_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        volume_add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot OHLCV data.

        Args:
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            plot_volume (bool): Whether to plot volume beneath.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            volume_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace` for OHLC.
            volume_add_trace_kwargs (dict): Keyword arguments passed to `add_trace` for volume.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> vbt.YFData.fetch("BTC-USD").get().vbt.ohlcv.plot().show()
            ```

            [=100% "100%"]{: .candystripe}

            ![](/assets/images/api/ohlcv_plot.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, make_subplots
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]
        ohlcv_cfg = settings["ohlcv"]

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if volume_trace_kwargs is None:
            volume_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if volume_add_trace_kwargs is None:
            volume_add_trace_kwargs = {}
        if plot_volume is None:
            plot_volume = self.volume is not None
        if plot_volume:
            add_trace_kwargs = merge_dicts(dict(row=1, col=1), add_trace_kwargs)
            volume_add_trace_kwargs = merge_dicts(dict(row=2, col=1), volume_add_trace_kwargs)

        # Set up figure
        if fig is None:
            if plot_volume:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0,
                    row_heights=[0.7, 0.3],
                )
            else:
                fig = make_figure()
            fig.update_layout(
                showlegend=True,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
            )
            if plot_volume:
                fig.update_layout(
                    xaxis2=dict(showgrid=True),
                    yaxis2=dict(showgrid=True),
                )
        fig.update_layout(**layout_kwargs)
        if ohlc_type is None:
            ohlc_type = ohlcv_cfg["ohlc_type"]
        if isinstance(ohlc_type, str):
            if ohlc_type.lower() == "ohlc":
                plot_obj = go.Ohlc
            elif ohlc_type.lower() == "candlestick":
                plot_obj = go.Candlestick
            else:
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")
        else:
            plot_obj = ohlc_type
        def_ohlc_trace_kwargs = dict(
            x=self.wrapper.index,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            name="OHLC",
            increasing=dict(
                fillcolor=plotting_cfg["color_schema"]["increasing"],
                line=dict(color=plotting_cfg["color_schema"]["increasing"]),
            ),
            decreasing=dict(
                fillcolor=plotting_cfg["color_schema"]["decreasing"],
                line=dict(color=plotting_cfg["color_schema"]["decreasing"]),
            ),
            opacity=0.75,
        )
        if plot_obj is go.Ohlc:
            del def_ohlc_trace_kwargs["increasing"]["fillcolor"]
            del def_ohlc_trace_kwargs["decreasing"]["fillcolor"]
        _ohlc_trace_kwargs = merge_dicts(def_ohlc_trace_kwargs, ohlc_trace_kwargs)
        ohlc = plot_obj(**_ohlc_trace_kwargs)
        fig.add_trace(ohlc, **add_trace_kwargs)
        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        if "rangeslider_visible" not in layout_kwargs.get(xaxis.replace("x", "xaxis"), {}):
            fig.update_layout({xaxis.replace("x", "xaxis"): dict(rangeslider_visible=False)})

        if plot_volume:
            marker_colors = np.empty(self.volume.shape, dtype=object)
            mask_greater = (self.close.values - self.open.values) > 0
            mask_less = (self.close.values - self.open.values) < 0
            marker_colors[mask_greater] = plotting_cfg["color_schema"]["increasing"]
            marker_colors[mask_less] = plotting_cfg["color_schema"]["decreasing"]
            marker_colors[~(mask_greater | mask_less)] = plotting_cfg["color_schema"]["gray"]
            _volume_trace_kwargs = merge_dicts(
                dict(
                    x=self.wrapper.index,
                    y=self.volume,
                    marker=dict(color=marker_colors, line_width=0),
                    opacity=0.5,
                    name="Volume",
                ),
                volume_trace_kwargs,
            )
            volume_bar = go.Bar(**_volume_trace_kwargs)
            fig.add_trace(volume_bar, **volume_add_trace_kwargs)

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `OHLCVDFAccessor.plots`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults` and
        `plots` from `vectorbtpro._settings.ohlcv`."""
        from vectorbtpro._settings import settings

        ohlcv_plots_cfg = settings["ohlcv"]["plots"]

        return merge_dicts(GenericAccessor.plots_defaults.__get__(self), ohlcv_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="OHLC",
                xaxis_kwargs=dict(showgrid=True, rangeslider_visible=False),
                yaxis_kwargs=dict(showgrid=True),
                check_is_not_grouped=True,
                plot_func="plot",
                plot_volume=False,
                tags="ohlcv",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


OHLCVDFAccessor.override_metrics_doc(__pdoc__)
OHLCVDFAccessor.override_subplots_doc(__pdoc__)
