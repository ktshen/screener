# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom indicators built with the indicator factory.

You can access all the indicators either by `vbt.*` or `vbt.indicators.*`.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbtpro as vbt

>>> # vectorbtpro.indicators.custom.MA
>>> vbt.MA.run(pd.Series([1, 2, 3]), [2, 3]).ma
ma_window    2    3
0          NaN  NaN
1          1.5  NaN
2          2.5  2.0
```

The advantage of these indicators over TA-Lib's is that they work primarily on 2-dimensional arrays
and utilize caching, which makes them faster for matrices with huge number of (repeating) columns.
They also have plotting methods.

Run for the examples below:

```pycon
>>> start = '2019-03-01 UTC'  # crypto is in UTC
>>> end = '2019-09-01 UTC'
>>> cols = ['Open', 'High', 'Low', 'Close', 'Volume']
>>> ohlcv = vbt.YFData.fetch("BTC-USD", start=start, end=end).get(cols)
```

[=100% "100%"]{: .candystripe}

```pycon
>>> ohlcv
                                   Open          High          Low  \\
Date
2019-03-01 00:00:00+00:00   3853.757080   3907.795410  3851.692383
2019-03-02 00:00:00+00:00   3855.318115   3874.607422  3832.127930
2019-03-03 00:00:00+00:00   3862.266113   3875.483643  3836.905762
...                                 ...           ...          ...
2019-08-30 00:00:00+00:00   9514.844727   9656.124023  9428.302734
2019-08-31 00:00:00+00:00   9597.539062   9673.220703  9531.799805
2019-09-01 00:00:00+00:00   9630.592773   9796.755859  9582.944336

                                 Close       Volume
Date
2019-03-01 00:00:00+00:00  3859.583740   7661247975
2019-03-02 00:00:00+00:00  3864.415039   7578786076
2019-03-03 00:00:00+00:00  3847.175781   7253558152
...                                ...          ...
2019-08-30 00:00:00+00:00  9598.173828  13595263986
2019-08-31 00:00:00+00:00  9630.664062  11454806419
2019-09-01 00:00:00+00:00  9757.970703  11445355859

[185 rows x 5 columns]

>>> ohlcv.vbt.ohlcv.plot().show()
```

![](/assets/images/api/custom_price.svg){: .iimg loading=lazy }"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.enums import Pivot, TrendMode
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.template import RepFunc

__all__ = [
    "MA",
    "MSD",
    "BBANDS",
    "RSI",
    "STOCH",
    "MACD",
    "ATR",
    "OBV",
    "OLS",
    "PATSIM",
    "VWAP",
    "PIVOTINFO",
    "SUPERTREND",
    "SIGDET",
]

# ############# MA ############# #


MA = IndicatorFactory(
    class_name="MA",
    module_name=__name__,
    short_name="ma",
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["ma"],
).with_apply_func(
    nb.ma_nb,
    cache_func=nb.ma_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=10,
    wtype="simple",
    adjust=False,
    minp=None,
)


class _MA(MA):
    """Moving Average (MA).

    A moving average is a widely used indicator in technical analysis that helps smooth out
    price action by filtering out the “noise” from random short-term price fluctuations.

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        ma_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `MA.ma` against `MA.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `MA.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.close`.
            ma_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ma`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.MA.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MA.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        ma_trace_kwargs = merge_dicts(
            dict(name="MA", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            ma_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.ma.vbt.lineplot(
            trace_kwargs=ma_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(MA, "__doc__", _MA.__doc__)
setattr(MA, "plot", _MA.plot)

# ############# MSD ############# #


MSD = IndicatorFactory(
    class_name="MSD",
    module_name=__name__,
    short_name="msd",
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["msd"],
).with_apply_func(
    nb.msd_nb,
    cache_func=nb.msd_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "ddof", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=10,
    wtype="simple",
    adjust=False,
    ddof=0,
    minp=None,
)


class _MSD(MSD):
    """Moving Standard Deviation (MSD).

    Standard deviation is an indicator that measures the size of an assets recent price moves
    in order to predict how volatile the price may be in the future."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        msd_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `MSD.msd`.

        Args:
            column (str): Name of the column to plot.
            msd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MSD.msd`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.MSD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MSD.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if msd_trace_kwargs is None:
            msd_trace_kwargs = {}
        msd_trace_kwargs = merge_dicts(
            dict(name="MSD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            msd_trace_kwargs,
        )

        fig = self_col.msd.vbt.lineplot(
            trace_kwargs=msd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


setattr(MSD, "__doc__", _MSD.__doc__)
setattr(MSD, "plot", _MSD.plot)

# ############# BBANDS ############# #


BBANDS = IndicatorFactory(
    class_name="BBANDS",
    module_name=__name__,
    short_name="bb",
    input_names=["close"],
    param_names=["window", "wtype", "alpha"],
    output_names=["middle", "upper", "lower"],
    lazy_outputs=dict(
        percent_b=lambda self: self.wrapper.wrap(
            (self.close.values - self.lower.values) / (self.upper.values - self.lower.values)
        ),
        bandwidth=lambda self: self.wrapper.wrap((self.upper.values - self.lower.values) / self.middle.values),
    ),
).with_apply_func(
    nb.bbands_nb,
    cache_func=nb.bbands_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "ddof", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=20,
    wtype="simple",
    alpha=2,
    adjust=False,
    ddof=0,
    minp=None,
)


class _BBANDS(BBANDS):
    """Bollinger Bands (BBANDS).

    A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `BBANDS.middle`, `BBANDS.upper` and `BBANDS.lower` against
        `BBANDS.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `BBANDS.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.close`.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.middle`.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.upper`.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.lower`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.BBANDS.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/BBANDS.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        lower_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_trace_kwargs,
        )
        upper_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_trace_kwargs,
        )  # default kwargs
        middle_trace_kwargs = merge_dicts(
            dict(name="Middle band", line=dict(color=plotting_cfg["color_schema"]["lightblue"])), middle_trace_kwargs
        )
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower.vbt.lineplot(
            trace_kwargs=lower_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper.vbt.lineplot(
            trace_kwargs=upper_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.middle.vbt.lineplot(
            trace_kwargs=middle_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        return fig


setattr(BBANDS, "__doc__", _BBANDS.__doc__)
setattr(BBANDS, "plot", _BBANDS.plot)

# ############# RSI ############# #


RSI = IndicatorFactory(
    class_name="RSI",
    module_name=__name__,
    short_name="rsi",
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["rsi"],
).with_apply_func(
    nb.rsi_nb,
    cache_func=nb.rsi_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="wilder",
    adjust=False,
    minp=None,
)


class _RSI(RSI):
    """Relative Strength Index (RSI).

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Tuple[float, float] = (30, 70),
        rsi_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `RSI.rsi`.

        Args:
            column (str): Name of the column to plot.
            limits (tuple of float): Tuple of the lower and upper limit.
            rsi_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `RSI.rsi`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.RSI.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/RSI.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_dicts(
            dict(name="RSI", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            rsi_trace_kwargs,
        )

        fig = self_col.rsi.vbt.lineplot(
            trace_kwargs=rsi_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        default_layout = dict()
        default_layout[yaxis.replace("y", "yaxis")] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # Fill void between limits
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xaxis,
                yref=yaxis,
                x0=self_col.wrapper.index[0],
                y0=limits[0],
                x1=self_col.wrapper.index[-1],
                y1=limits[1],
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(RSI, "__doc__", _RSI.__doc__)
setattr(RSI, "plot", _RSI.plot)

# ############# STOCH ############# #


STOCH = IndicatorFactory(
    class_name="STOCH",
    module_name=__name__,
    short_name="stoch",
    input_names=["high", "low", "close"],
    param_names=["fast_k_window", "slow_k_window", "slow_d_window", "wtype"],
    output_names=["fast_k", "slow_k", "slow_d"],
).with_apply_func(
    nb.stoch_nb,
    cache_func=nb.stoch_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    fast_k_window=14,
    slow_k_window=3,
    slow_d_window=3,
    wtype="simple",
    adjust=False,
    minp=None,
)


class _STOCH(STOCH):
    """Stochastic Oscillator (STOCH).

    A stochastic oscillator is a momentum indicator comparing a particular closing price
    of a security to a range of its prices over a certain period of time. It is used to
    generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.

    See [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Tuple[float, float] = (20, 80),
        fast_k_trace_kwargs: tp.KwargsLike = None,
        slow_k_trace_kwargs: tp.KwargsLike = None,
        slow_d_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `STOCH.slow_k` and `STOCH.slow_d`.

        Args:
            column (str): Name of the column to plot.
            limits (tuple of float): Tuple of the lower and upper limit.
            fast_k_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.fast_k`.
            slow_k_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.slow_k`.
            slow_d_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.slow_d`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.STOCH.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/STOCH.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fast_k_trace_kwargs is None:
            fast_k_trace_kwargs = {}
        if slow_k_trace_kwargs is None:
            slow_k_trace_kwargs = {}
        if slow_d_trace_kwargs is None:
            slow_d_trace_kwargs = {}
        fast_k_trace_kwargs = merge_dicts(
            dict(name="Fast %K", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fast_k_trace_kwargs,
        )
        slow_k_trace_kwargs = merge_dicts(
            dict(name="Slow %K", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            slow_k_trace_kwargs,
        )
        slow_d_trace_kwargs = merge_dicts(
            dict(name="Slow %D", line=dict(color=plotting_cfg["color_schema"]["lightpink"])),
            slow_d_trace_kwargs,
        )

        fig = self_col.fast_k.vbt.lineplot(
            trace_kwargs=fast_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.slow_k.vbt.lineplot(
            trace_kwargs=slow_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.slow_d.vbt.lineplot(
            trace_kwargs=slow_d_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        default_layout = dict()
        default_layout[yaxis.replace("y", "yaxis")] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # Fill void between limits
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xaxis,
                yref=yaxis,
                x0=self_col.wrapper.index[0],
                y0=limits[0],
                x1=self_col.wrapper.index[-1],
                y1=limits[1],
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(STOCH, "__doc__", _STOCH.__doc__)
setattr(STOCH, "plot", _STOCH.plot)

# ############# MACD ############# #


MACD = IndicatorFactory(
    class_name="MACD",
    module_name=__name__,
    short_name="macd",
    input_names=["close"],
    param_names=["fast_window", "slow_window", "signal_window", "macd_wtype", "signal_wtype"],
    output_names=["macd", "signal"],
    lazy_outputs=dict(
        hist=lambda self: self.wrapper.wrap(self.macd.values - self.signal.values),
    ),
).with_apply_func(
    nb.macd_nb,
    cache_func=nb.macd_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "minp"],
    param_settings=dict(
        macd_wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        ),
        signal_wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    fast_window=12,
    slow_window=26,
    signal_window=9,
    macd_wtype="exp",
    signal_wtype="exp",
    adjust=False,
    minp=None,
)


class _MACD(MACD):
    """Moving Average Convergence Divergence (MACD).

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        macd_trace_kwargs: tp.KwargsLike = None,
        signal_trace_kwargs: tp.KwargsLike = None,
        hist_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `MACD.macd`, `MACD.signal` and `MACD.hist`.

        Args:
            column (str): Name of the column to plot.
            macd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.macd`.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.signal`.
            hist_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar` for `MACD.hist`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.MACD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MACD.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
            fig.update_layout(bargap=0)
        fig.update_layout(**layout_kwargs)

        if macd_trace_kwargs is None:
            macd_trace_kwargs = {}
        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        if hist_trace_kwargs is None:
            hist_trace_kwargs = {}
        macd_trace_kwargs = merge_dicts(
            dict(name="MACD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])), macd_trace_kwargs
        )
        signal_trace_kwargs = merge_dicts(
            dict(name="Signal", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])), signal_trace_kwargs
        )

        fig = self_col.macd.vbt.lineplot(
            trace_kwargs=macd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.signal.vbt.lineplot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Plot hist
        hist = self_col.hist.values
        hist_diff = generic_nb.diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, adjust_opacity("silver", 0.75), dtype=object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = adjust_opacity("green", 0.75)
        marker_colors[(hist > 0) & (hist_diff <= 0)] = adjust_opacity("lightgreen", 0.75)
        marker_colors[(hist < 0) & (hist_diff < 0)] = adjust_opacity("red", 0.75)
        marker_colors[(hist < 0) & (hist_diff >= 0)] = adjust_opacity("lightcoral", 0.75)

        _hist_trace_kwargs = merge_dicts(
            dict(
                name="Histogram",
                x=self_col.hist.index,
                y=self_col.hist.values,
                marker_color=marker_colors,
                marker_line_width=0,
            ),
            hist_trace_kwargs,
        )
        hist_bar = go.Bar(**_hist_trace_kwargs)
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        fig.add_trace(hist_bar, **add_trace_kwargs)

        return fig


setattr(MACD, "__doc__", _MACD.__doc__)
setattr(MACD, "plot", _MACD.plot)

# ############# ATR ############# #


ATR = IndicatorFactory(
    class_name="ATR",
    module_name=__name__,
    short_name="atr",
    input_names=["high", "low", "close"],
    param_names=["window", "wtype"],
    output_names=["tr", "atr"],
).with_apply_func(
    nb.atr_nb,
    cache_func=nb.atr_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["adjust", "minp"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="wilder",
    adjust=False,
    minp=None,
)


class _ATR(ATR):
    """Average True Range (ATR).

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp).
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        tr_trace_kwargs: tp.KwargsLike = None,
        atr_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            column (str): Name of the column to plot.
            tr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.atr`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.ATR.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/ATR.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if tr_trace_kwargs is None:
            tr_trace_kwargs = {}
        if atr_trace_kwargs is None:
            atr_trace_kwargs = {}
        tr_trace_kwargs = merge_dicts(
            dict(name="TR", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            tr_trace_kwargs,
        )
        atr_trace_kwargs = merge_dicts(
            dict(name="ATR", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            atr_trace_kwargs,
        )

        fig = self_col.tr.vbt.lineplot(
            trace_kwargs=tr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.atr.vbt.lineplot(
            trace_kwargs=atr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(ATR, "__doc__", _ATR.__doc__)
setattr(ATR, "plot", _ATR.plot)

# ############# OBV ############# #


OBV = IndicatorFactory(
    class_name="OBV",
    module_name=__name__,
    short_name="obv",
    input_names=["close", "volume"],
    param_names=[],
    output_names=["obv"],
).with_custom_func(nb.obv_nb)


class _OBV(OBV):
    """On-balance volume (OBV).

    It relates price and volume in the stock market. OBV is based on a cumulative total volume.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        obv_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `OBV.obv`.

        Args:
            column (str): Name of the column to plot.
            obv_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OBV.obv`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```py
            >>> vbt.OBV.run(ohlcv['Close'], ohlcv['Volume']).plot().show()
            ```

            ![](/assets/images/api/OBV.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_dicts(
            dict(name="OBV", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            obv_trace_kwargs,
        )

        fig = self_col.obv.vbt.lineplot(
            trace_kwargs=obv_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(OBV, "__doc__", _OBV.__doc__)
setattr(OBV, "plot", _OBV.plot)


# ############# OLS ############# #


OLS = IndicatorFactory(
    class_name="OLS",
    module_name=__name__,
    short_name="ols",
    input_names=["x", "y"],
    param_names=["window"],
    output_names=["slope", "intercept", "zscore"],
    lazy_outputs=dict(
        pred=lambda self: self.wrapper.wrap(self.intercept.values + self.slope.values * self.x.values),
        error=lambda self: self.wrapper.wrap(self.y.values - self.pred.values),
        angle=lambda self: self.wrapper.wrap(np.arctan(self.slope.values) * 180 / np.pi),
    ),
).with_apply_func(
    nb.ols_nb,
    cache_func=nb.ols_cache_nb,
    cache_pass_per_column=True,
    kwargs_as_args=["with_zscore", "ddof", "minp"],
    window=14,
    with_zscore=True,
    ddof=0,
    minp=None,
)


class _OLS(OLS):
    """Rolling Ordinary Least Squares (OLS).

    The indicator can be used to detect changes in the behavior of the stocks against the market or each other.

    See [The Linear Regression of Time and Price](https://www.investopedia.com/articles/trading/09/linear-regression-time-price.asp).
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_y: bool = True,
        y_trace_kwargs: tp.KwargsLike = None,
        pred_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `OLS.pred` against `OLS.y`.

        Args:
            column (str): Name of the column to plot.
            plot_y (bool): Whether to plot `OLS.y`.
            y_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.y`.
            pred_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.pred`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/OLS.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if y_trace_kwargs is None:
            y_trace_kwargs = {}
        if pred_trace_kwargs is None:
            pred_trace_kwargs = {}
        y_trace_kwargs = merge_dicts(
            dict(name="Y", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            y_trace_kwargs,
        )
        pred_trace_kwargs = merge_dicts(
            dict(name="Pred", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            pred_trace_kwargs,
        )

        if plot_y:
            fig = self_col.y.vbt.lineplot(
                trace_kwargs=y_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.pred.vbt.lineplot(
            trace_kwargs=pred_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig

    def plot_zscore(
        self,
        column: tp.Optional[tp.Label] = None,
        alpha: float = 0.05,
        zscore_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `OLS.zscore` with confidence intervals.

        Args:
            column (str): Name of the column to plot.
            alpha (float): The alpha level for the confidence interval.

                The default alpha = .05 returns a 95% confidence interval.
            zscore_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.zscore`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape`
                when adding the range between both confidence intervals.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/OLS_zscore.svg){: .iimg loading=lazy }
        """
        import scipy.stats as st
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        zscore_trace_kwargs = merge_dicts(
            dict(name="Z-score", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            zscore_trace_kwargs,
        )
        fig = self_col.zscore.vbt.lineplot(
            trace_kwargs=zscore_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Fill void between limits
        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xaxis,
                yref=yaxis,
                x0=self_col.wrapper.index[0],
                y0=st.norm.ppf(1 - alpha / 2),
                x1=self_col.wrapper.index[-1],
                y1=st.norm.ppf(alpha / 2),
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(OLS, "__doc__", _OLS.__doc__)
setattr(OLS, "plot", _OLS.plot)
setattr(OLS, "plot_zscore", _OLS.plot_zscore)


# ############# PATSIM ############# #


PATSIM = IndicatorFactory(
    class_name="PATSIM",
    module_name=__name__,
    short_name="patsim",
    input_names=["close"],
    param_names=[
        "pattern",
        "window",
        "max_window",
        "row_select_prob",
        "window_select_prob",
        "interp_mode",
        "rescale_mode",
        "vmin",
        "vmax",
        "pmin",
        "pmax",
        "invert",
        "error_type",
        "distance_measure",
        "max_error",
        "max_error_interp_mode",
        "max_error_as_maxdist",
        "max_error_strict",
        "min_pct_change",
        "max_pct_change",
        "min_similarity",
    ],
    output_names=["sim"],
).with_apply_func(
    generic_nb.rolling_pattern_similarity_nb,
    param_settings=dict(
        pattern=dict(is_array_like=True, min_one_dim=True),
        interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
        ),
        rescale_mode=dict(
            dtype=generic_enums.RescaleMode,
            post_index_func=lambda index: index.str.lower(),
        ),
        error_type=dict(
            dtype=generic_enums.ErrorType,
            post_index_func=lambda index: index.str.lower(),
        ),
        distance_measure=dict(
            dtype=generic_enums.DistanceMeasure,
            post_index_func=lambda index: index.str.lower(),
        ),
        max_error=dict(is_array_like=True, min_one_dim=True),
        max_error_interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    window=None,
    max_window=None,
    row_select_prob=1.0,
    window_select_prob=1.0,
    interp_mode="mixed",
    rescale_mode="minmax",
    vmin=np.nan,
    vmax=np.nan,
    pmin=np.nan,
    pmax=np.nan,
    invert=False,
    error_type="absolute",
    distance_measure="mae",
    max_error=np.nan,
    max_error_interp_mode=None,
    max_error_as_maxdist=False,
    max_error_strict=False,
    min_pct_change=np.nan,
    max_pct_change=np.nan,
    min_similarity=np.nan,
)


class _PATSIM(PATSIM):
    """Rolling pattern similarity.

    Based on `vectorbtpro.generic.nb.rolling.rolling_pattern_similarity_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        sim_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `PATSIM.sim` against `PATSIM.close`.

        Args:
            column (str): Name of the column to plot.
            sim_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PATSIM.sim`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).plot().show()
            ```

            ![](/assets/images/api/PATSIM.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        sim_trace_kwargs = merge_dicts(
            dict(name="Similarity", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            sim_trace_kwargs,
        )
        fig = self_col.sim.vbt.lineplot(
            trace_kwargs=sim_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        default_layout = dict()
        default_layout[yaxis.replace("y", "yaxis")] = dict(tickformat=",.0%")
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        return fig

    def overlay_with_heatmap(
        self,
        column: tp.Optional[tp.Label] = None,
        close_trace_kwargs: tp.KwargsLike = None,
        sim_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Overlay `PATSIM.sim` as a heatmap on top of `PATSIM.close`.

        Args:
            column (str): Name of the column to plot.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PATSIM.close`.
            sim_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap` for `PATSIM.sim`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).overlay_with_heatmap().show()
            ```

            ![](/assets/images/api/PATSIM_heatmap.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if sim_trace_kwargs is None:
            sim_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        sim_trace_kwargs = merge_dicts(
            dict(
                colorbar=dict(tickformat=",.0%"),
                colorscale=[
                    [0.0, "rgba(0, 0, 0, 0)"],
                    [1.0, plotting_cfg["color_schema"]["lightpurple"]],
                ],
                zmin=0,
                zmax=1,
            ),
            sim_trace_kwargs,
        )
        fig = self_col.close.vbt.overlay_with_heatmap(
            self_col.sim,
            trace_kwargs=close_trace_kwargs,
            heatmap_kwargs=dict(y_labels=["Similarity"], trace_kwargs=sim_trace_kwargs),
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


setattr(PATSIM, "__doc__", _PATSIM.__doc__)
setattr(PATSIM, "plot", _PATSIM.plot)
setattr(PATSIM, "overlay_with_heatmap", _PATSIM.overlay_with_heatmap)


# ############# VWAP ############# #


def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute reset frequency by group lens."""
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()


VWAP = IndicatorFactory(
    class_name="VWAP",
    module_name=__name__,
    short_name="vwap",
    input_names=["high", "low", "close", "volume"],
    param_names=["anchor"],
    output_names=["vwap"],
).with_apply_func(
    nb.vwap_nb,
    param_settings=dict(
        anchor=dict(template=RepFunc(substitute_anchor)),
    ),
    anchor="D",
)


class _VWAP(VWAP):
    """Volume-Weighted Average Price (VWAP).

    VWAP is a technical analysis indicator used on intraday charts that resets at the start
    of every new trading session.

    See [Volume-Weighted Average Price (VWAP)](https://www.investopedia.com/terms/v/vwap.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        vwap_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `VWAP.vwap` against `VWAP.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `VWAP.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `VWAP.close`.
            vwap_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `VWAP.vwap`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.VWAP.run(
            ...    ohlcv['High'],
            ...    ohlcv['Low'],
            ...    ohlcv['Close'],
            ...    ohlcv['Volume'],
            ...    anchor="W"
            ... ).plot().show()
            ```

            ![](/assets/images/api/VWAP.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if vwap_trace_kwargs is None:
            vwap_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        vwap_trace_kwargs = merge_dicts(
            dict(name="VWAP", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            vwap_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.vwap.vbt.lineplot(
            trace_kwargs=vwap_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(VWAP, "__doc__", _VWAP.__doc__)
setattr(VWAP, "plot", _VWAP.plot)


# ############# PIVOTINFO ############# #


PIVOTINFO = IndicatorFactory(
    class_name="PIVOTINFO",
    module_name=__name__,
    short_name="pivotinfo",
    input_names=["high", "low"],
    param_names=["up_th", "down_th"],
    output_names=["conf_pivot", "conf_idx", "last_pivot", "last_idx"],
    lazy_outputs=dict(
        conf_value=lambda self: self.wrapper.wrap(
            nb.pivot_value_nb(
                to_2d_array(self.high),
                to_2d_array(self.low),
                to_2d_array(self.conf_pivot),
                to_2d_array(self.conf_idx),
            )
        ),
        last_value=lambda self: self.wrapper.wrap(
            nb.pivot_value_nb(
                to_2d_array(self.high),
                to_2d_array(self.low),
                to_2d_array(self.last_pivot),
                to_2d_array(self.last_idx),
            )
        ),
        pivots=lambda self: self.wrapper.wrap(
            nb.pivots_nb(
                to_2d_array(self.conf_pivot),
                to_2d_array(self.conf_idx),
                to_2d_array(self.last_pivot),
            )
        ),
        modes=lambda self: self.wrapper.wrap(
            nb.pivots_to_modes_nb(
                to_2d_array(self.pivots),
            )
        ),
    ),
    attr_settings=dict(
        conf_pivot=dict(dtype=Pivot, enum_unkval=0),
        last_pivot=dict(dtype=Pivot, enum_unkval=0),
        pivots=dict(dtype=Pivot, enum_unkval=0),
        modes=dict(dtype=TrendMode, enum_unkval=0),
    ),
).with_apply_func(
    nb.pivot_info_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
)


class _PIVOTINFO(PIVOTINFO):
    """Indicator that returns various information on pivots identified based on thresholds.

    * `conf_pivot`: the type of the latest confirmed pivot (running)
    * `conf_idx`: the index of the latest confirmed pivot (running)
    * `conf_value`: the high/low value under the latest confirmed pivot (running)
    * `last_pivot`: the type of the latest pivot (running)
    * `last_idx`: the index of the latest pivot (running)
    * `last_value`: the high/low value under the latest pivot (running)
    * `pivots`: confirmed pivots stored under their indices (looking ahead - use only for plotting!)
    * `modes`: modes between confirmed pivot points (looking ahead - use only for plotting!)
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        conf_value_trace_kwargs: tp.KwargsLike = None,
        last_value_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `PIVOTINFO.conf_value` and `PIVOTINFO.last_value`.

        Args:
            column (str): Name of the column to plot.
            conf_value_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PIVOTINFO.conf_value` line.
            last_value_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PIVOTINFO.last_value` line.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> fig = ohlcv.vbt.ohlcv.plot()
            >>> vbt.PIVOTINFO.run(ohlcv['High'], ohlcv['Low'], 0.1, 0.1).plot(fig=fig).show()
            ```

            ![](/assets/images/api/PIVOTINFO.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if conf_value_trace_kwargs is None:
            conf_value_trace_kwargs = {}
        if last_value_trace_kwargs is None:
            last_value_trace_kwargs = {}
        conf_value_trace_kwargs = merge_dicts(
            dict(name="Confirmed value", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            conf_value_trace_kwargs,
        )
        last_value_trace_kwargs = merge_dicts(
            dict(name="Last value", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            last_value_trace_kwargs,
        )

        fig = self_col.conf_value.vbt.lineplot(
            trace_kwargs=conf_value_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.last_value.vbt.lineplot(
            trace_kwargs=last_value_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig

    def plot_zigzag(
        self,
        column: tp.Optional[tp.Label] = None,
        zigzag_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot zig-zag line.

        Args:
            column (str): Name of the column to plot.
            zigzag_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for zig-zag line.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> fig = ohlcv.vbt.ohlcv.plot()
            >>> vbt.PIVOTINFO.run(ohlcv['High'], ohlcv['Low'], 0.1, 0.1).plot_zigzag(fig=fig).show()
            ```

            ![](/assets/images/api/PIVOTINFO_zigzag.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if zigzag_trace_kwargs is None:
            zigzag_trace_kwargs = {}
        zigzag_trace_kwargs = merge_dicts(
            dict(name="ZigZag", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            zigzag_trace_kwargs,
        )

        pivots = self_col.pivots
        highs = self_col.high[pivots == Pivot.Peak]
        lows = self_col.low[pivots == Pivot.Valley]
        fig = (
            pd.concat((highs, lows))
            .sort_index()
            .vbt.lineplot(
                trace_kwargs=zigzag_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        )

        return fig


setattr(PIVOTINFO, "__doc__", _PIVOTINFO.__doc__)
setattr(PIVOTINFO, "plot", _PIVOTINFO.plot)
setattr(PIVOTINFO, "plot_zigzag", _PIVOTINFO.plot_zigzag)


# ############# SUPERTREND ############# #


SUPERTREND = IndicatorFactory(
    class_name="SUPERTREND",
    module_name=__name__,
    short_name="supertrend",
    input_names=["high", "low", "close"],
    param_names=["period", "multiplier"],
    output_names=["supert", "superd", "superl", "supers"],
).with_apply_func(nb.supertrend_nb, period=7, multiplier=3)


class _SUPERTREND(SUPERTREND):
    """Supertrend indicator."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        superl_trace_kwargs: tp.KwargsLike = None,
        supers_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `SUPERTREND.superl` and `SUPERTREND.supers` against `SUPERTREND.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `SUPERTREND.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.close`.
            superl_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.superl`.
            supers_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.supers`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.SUPERTREND.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SUPERTREND.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if superl_trace_kwargs is None:
            superl_trace_kwargs = {}
        if supers_trace_kwargs is None:
            supers_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        superl_trace_kwargs = merge_dicts(
            dict(name="Long", line=dict(color=plotting_cfg["color_schema"]["green"])),
            superl_trace_kwargs,
        )
        supers_trace_kwargs = merge_dicts(
            dict(name="Short", line=dict(color=plotting_cfg["color_schema"]["red"])),
            supers_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.superl.vbt.lineplot(
            trace_kwargs=superl_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.supers.vbt.lineplot(
            trace_kwargs=supers_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(SUPERTREND, "__doc__", _SUPERTREND.__doc__)
setattr(SUPERTREND, "plot", _SUPERTREND.plot)


# ############# SIGDET ############# #


SIGDET = IndicatorFactory(
    class_name="SIGDET",
    module_name=__name__,
    short_name="sigdet",
    input_names=["close"],
    param_names=["lag", "factor", "influence", "down_factor", "std_influence"],
    output_names=["signal", "upper_band", "lower_band"],
).with_apply_func(
    nb.signal_detection_nb,
    param_settings=dict(
        factor=flex_elem_param_config,
        influence=flex_elem_param_config,
        down_factor=flex_elem_param_config,
        std_influence=flex_elem_param_config,
    ),
    lag=14,
    factor=1.0,
    influence=1.0,
    down_factor=None,
    std_influence=None,
)


class _SIGDET(SIGDET):
    """Robust peak detection algorithm (using z-scores).

    See https://stackoverflow.com/a/22640362"""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        signal_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> tp.BaseFigure:
        """Plot `SIGDET.signal` against `SIGDET.close`.

        Args:
            column (str): Name of the column to plot.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.signal`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SIGDET.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        signal_trace_kwargs = merge_dicts(
            dict(name="Signal", line=dict(color=plotting_cfg["color_schema"]["lightblue"], shape="hv")),
            signal_trace_kwargs,
        )
        fig = self_col.signal.vbt.lineplot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig

    def plot_bands(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        upper_band_trace_kwargs: tp.KwargsLike = None,
        lower_band_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `SIGDET.upper_band` and `SIGDET.lower_band` against `SIGDET.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `SIGDET.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.close`.
            upper_band_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.upper_band`.
            lower_band_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.lower_band`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot_bands().show()
            ```

            ![](/assets/images/api/SIGDET_plot_bands.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if upper_band_trace_kwargs is None:
            upper_band_trace_kwargs = {}
        if lower_band_trace_kwargs is None:
            lower_band_trace_kwargs = {}
        lower_band_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_band_trace_kwargs,
        )
        upper_band_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_band_trace_kwargs,
        )  # default kwargs
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower_band.vbt.lineplot(
            trace_kwargs=lower_band_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper_band.vbt.lineplot(
            trace_kwargs=upper_band_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        return fig


setattr(SIGDET, "__doc__", _SIGDET.__doc__)
setattr(SIGDET, "plot", _SIGDET.plot)
setattr(SIGDET, "plot_bands", _SIGDET.plot_bands)
