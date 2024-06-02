# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base plotting functions.

Provides functions for visualizing data in an efficient and convenient way.
Each creates a figure widget that is compatible with ipywidgets and enables interactive
data visualization in Jupyter Notebook and JupyterLab environments. For more details
on using Plotly, see [Getting Started with Plotly in Python](https://plotly.com/python/getting-started/).

!!! warning
    Errors related to plotting in Jupyter environment usually appear in the logs, not under the cell."""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from vectorbtpro import _typing as tp
from vectorbtpro.base import reshaping
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import rescale
from vectorbtpro.utils.colors import map_value_to_cmap
from vectorbtpro.utils.config import Configured, resolve_dict, merge_dicts
from vectorbtpro.utils.figure import make_figure

__all__ = [
    "TraceUpdater",
    "Gauge",
    "Bar",
    "Scatter",
    "Histogram",
    "Box",
    "Heatmap",
    "Volume",
]


def clean_labels(labels: tp.Labels) -> tp.Labels:
    """Clean labels.

    Plotly doesn't support multi-indexes."""
    if isinstance(labels, pd.MultiIndex):
        labels = labels.to_flat_index()
    if isinstance(labels, pd.PeriodIndex):
        labels = labels.map(str)
    if len(labels) > 0 and isinstance(labels[0], tuple):
        labels = list(map(str, labels))
    return labels


class TraceUpdater:
    def __init__(self, fig: tp.BaseFigure, traces: tp.Tuple[BaseTraceType, ...]) -> None:
        """Base trace updating class."""
        self._fig = fig
        self._traces = traces

    @property
    def fig(self) -> tp.BaseFigure:
        """Figure."""
        return self._fig

    @property
    def traces(self) -> tp.Tuple[BaseTraceType, ...]:
        """Traces to update."""
        return self._traces

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, *args, **kwargs) -> None:
        """Update one trace."""
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        """Update all traces using new data."""
        raise NotImplementedError


class Gauge(Configured, TraceUpdater):
    def __init__(
        self,
        value: tp.Optional[float] = None,
        label: tp.Optional[str] = None,
        value_range: tp.Optional[tp.Tuple[float, float]] = None,
        cmap_name: str = "Spectral",
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a gauge plot.

        Args:
            value (float): The value to be displayed.
            label (str): The label to be displayed.
            value_range (tuple of float): The value range of the gauge.
            cmap_name (str): A matplotlib-compatible colormap name.

                See the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
            trace_kwargs (dict): Keyword arguments passed to the `plotly.graph_objects.Indicator`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> gauge = vbt.Gauge(
            ...     value=2,
            ...     value_range=(1, 3),
            ...     label='My Gauge'
            ... )
            >>> gauge.fig.show()
            ```

            ![](/assets/images/api/Gauge.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            value=value,
            label=label,
            value_range=value_range,
            cmap_name=cmap_name,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                fig.update_layout(width=layout_cfg["width"] * 0.7, height=layout_cfg["width"] * 0.5, margin=dict(t=80))
        fig.update_layout(**layout_kwargs)

        _trace_kwargs = merge_dicts(
            dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                mode="gauge+number+delta",
                title=dict(text=label),
            ),
            trace_kwargs,
        )
        trace = go.Indicator(**_trace_kwargs)
        if value is not None:
            self.update_trace(trace, value, value_range=value_range, cmap_name=cmap_name)
        fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))
        self._value_range = value_range
        self._cmap_name = cmap_name

    @property
    def value_range(self) -> tp.Tuple[float, float]:
        """The value range of the gauge."""
        return self._value_range

    @property
    def cmap_name(self) -> str:
        """A matplotlib-compatible colormap name."""
        return self._cmap_name

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        value: float,
        value_range: tp.Optional[tp.Tuple[float, float]] = None,
        cmap_name: str = "Spectral",
    ) -> None:
        if value_range is not None:
            trace.gauge.axis.range = value_range
            if cmap_name is not None:
                trace.gauge.bar.color = map_value_to_cmap(value, cmap_name, vmin=value_range[0], vmax=value_range[1])
        trace.delta.reference = trace.value
        trace.value = value

    def update(self, value: float) -> None:
        if self.value_range is None:
            self._value_range = value, value
        else:
            self._value_range = min(self.value_range[0], value), max(self.value_range[1], value)

        with self.fig.batch_update():
            self.update_trace(
                self.traces[0],
                value=value,
                value_range=self.value_range,
                cmap_name=self.cmap_name,
            )


class Bar(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a bar plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            x_labels (array_like): X-axis labels, corresponding to index in pandas.
            trace_kwargs (dict or list of dict): Keyword arguments passed to `plotly.graph_objects.Bar`.

                Can be specified per trace as a sequence of dicts.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> bar = vbt.Bar(
            ...     data=[[1, 2], [3, 4]],
            ...     trace_names=['a', 'b'],
            ...     x_labels=['x', 'y']
            ... )
            >>> bar.fig.show()
            ```

            ![](/assets/images/api/Bar.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]
        if x_labels is not None:
            x_labels = clean_labels(x_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(x=x_labels, name=trace_name, showlegend=trace_name is not None),
                _trace_kwargs,
            )
            trace = go.Bar(**_trace_kwargs)
            if data is not None:
                self.update_trace(trace, data, i)
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, i: int) -> None:
        data = reshaping.to_2d_array(data)

        trace.y = data[:, i]
        if trace.marker.colorscale is not None:
            trace.marker.color = data[:, i]

    def update(self, data: tp.ArrayLike) -> None:
        data = reshaping.to_2d_array(data)

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(trace, data, i)


class Scatter(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        use_gl: tp.Optional[bool] = None,
        **layout_kwargs
    ) -> None:
        """Create a scatter plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            x_labels (array_like): X-axis labels, corresponding to index in pandas.
            trace_kwargs (dict or list of dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.

                Can be specified per trace as a sequence of dicts.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            use_gl (bool): Whether to use `plotly.graph_objects.Scattergl`.

                Defaults to the global setting. If the global setting is None, becomes True
                if there are more than 10,000 data points.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> scatter = vbt.Scatter(
            ...     data=[[1, 2], [3, 4]],
            ...     trace_names=['a', 'b'],
            ...     x_labels=['x', 'y']
            ... )
            >>> scatter.fig.show()
            ```

            ![](/assets/images/api/Scatter.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]
        if x_labels is not None:
            x_labels = clean_labels(x_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            _use_gl = _trace_kwargs.pop("use_gl", use_gl)
            if _use_gl is None:
                _use_gl = plotting_cfg["use_gl"]
            if _use_gl is None:
                _use_gl = _use_gl is None and data is not None and data.size >= 10000
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            if _use_gl:
                scatter_obj = go.Scattergl
            else:
                scatter_obj = go.Scatter
            try:
                from plotly_resampler.aggregation import AbstractFigureAggregator

                if isinstance(fig, AbstractFigureAggregator):
                    use_resampler = True
                else:
                    use_resampler = False
            except ImportError:
                use_resampler = False
            if use_resampler:
                if data is None:
                    raise ValueError("Cannot create empty scatter traces when using plotly-resampler")
                _trace_kwargs = merge_dicts(
                    dict(name=trace_name, showlegend=trace_name is not None),
                    _trace_kwargs,
                )
                trace = scatter_obj(**_trace_kwargs)
                fig.add_trace(trace, hf_x=x_labels, hf_y=data[:, i], **add_trace_kwargs)
            else:
                _trace_kwargs = merge_dicts(
                    dict(x=x_labels, name=trace_name, showlegend=trace_name is not None),
                    _trace_kwargs,
                )
                trace = scatter_obj(**_trace_kwargs)
                if data is not None:
                    self.update_trace(trace, data, i)
                fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, i: int) -> None:
        data = reshaping.to_2d_array(data)

        trace.y = data[:, i]

    def update(self, data: tp.ArrayLike) -> None:
        data = reshaping.to_2d_array(data)

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(trace, data, i)


class Histogram(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a histogram plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (any, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            horizontal (bool): Whether to plot horizontally.
            remove_nan (bool): Whether to remove NaN values.
            from_quantile (float): Filter out data points before this quantile.

                Must be in range `[0, 1]`.
            to_quantile (float): Filter out data points after this quantile.

                Must be in range `[0, 1]`.
            trace_kwargs (dict or list of dict): Keyword arguments passed to `plotly.graph_objects.Histogram`.

                Can be specified per trace as a sequence of dicts.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> hist = vbt.Histogram(
            ...     data=[[1, 2], [3, 4], [2, 1]],
            ...     trace_names=['a', 'b']
            ... )
            >>> hist.fig.show()
            ```

            ![](/assets/images/api/Histogram.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            fig.update_layout(barmode="overlay")
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(
                    opacity=0.75 if len(trace_names) > 1 else 1,
                    name=trace_name,
                    showlegend=trace_name is not None,
                ),
                _trace_kwargs,
            )
            trace = go.Histogram(**_trace_kwargs)
            if data is not None:
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=horizontal,
                    remove_nan=remove_nan,
                    from_quantile=from_quantile,
                    to_quantile=to_quantile,
                )
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])
        self._horizontal = horizontal
        self._remove_nan = remove_nan
        self._from_quantile = from_quantile
        self._to_quantile = to_quantile

    @property
    def horizontal(self) -> bool:
        """Whether to plot horizontally."""
        return self._horizontal

    @property
    def remove_nan(self) -> bool:
        """Whether to remove NaN values."""
        return self._remove_nan

    @property
    def from_quantile(self) -> float:
        """Filter out data points before this quantile."""
        return self._from_quantile

    @property
    def to_quantile(self) -> float:
        """Filter out data points after this quantile."""
        return self._to_quantile

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: tp.ArrayLike,
        i: int,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
    ) -> None:
        data = reshaping.to_2d_array(data)

        d = data[:, i]
        if remove_nan:
            d = d[~np.isnan(d)]
        mask = np.full(d.shape, True)
        if from_quantile is not None:
            mask &= d >= np.quantile(d, from_quantile)
        if to_quantile is not None:
            mask &= d <= np.quantile(d, to_quantile)
        d = d[mask]
        if horizontal:
            trace.x = None
            trace.y = d
        else:
            trace.x = d
            trace.y = None

    def update(self, data: tp.ArrayLike) -> None:
        data = reshaping.to_2d_array(data)

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=self.horizontal,
                    remove_nan=self.remove_nan,
                    from_quantile=self.from_quantile,
                    to_quantile=self.to_quantile,
                )


class Box(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a box plot.

        For keyword arguments, see `Histogram`.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> box = vbt.Box(
            ...     data=[[1, 2], [3, 4], [2, 1]],
            ...     trace_names=['a', 'b']
            ... )
            >>> box.fig.show()
            ```

            ![](/assets/images/api/Box.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(name=trace_name, showlegend=trace_name is not None, boxmean="sd"),
                _trace_kwargs,
            )
            trace = go.Box(**_trace_kwargs)
            if data is not None:
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=horizontal,
                    remove_nan=remove_nan,
                    from_quantile=from_quantile,
                    to_quantile=to_quantile,
                )
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])
        self._horizontal = horizontal
        self._remove_nan = remove_nan
        self._from_quantile = from_quantile
        self._to_quantile = to_quantile

    @property
    def horizontal(self) -> bool:
        """Whether to plot horizontally."""
        return self._horizontal

    @property
    def remove_nan(self) -> bool:
        """Whether to remove NaN values."""
        return self._remove_nan

    @property
    def from_quantile(self) -> float:
        """Filter out data points before this quantile."""
        return self._from_quantile

    @property
    def to_quantile(self) -> float:
        """Filter out data points after this quantile."""
        return self._to_quantile

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: tp.ArrayLike,
        i: int,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
    ) -> None:
        data = reshaping.to_2d_array(data)

        d = data[:, i]
        if remove_nan:
            d = d[~np.isnan(d)]
        mask = np.full(d.shape, True)
        if from_quantile is not None:
            mask &= d >= np.quantile(d, from_quantile)
        if to_quantile is not None:
            mask &= d <= np.quantile(d, to_quantile)
        d = d[mask]
        if horizontal:
            trace.x = d
            trace.y = None
        else:
            trace.x = None
            trace.y = d

    def update(self, data: tp.ArrayLike) -> None:
        data = reshaping.to_2d_array(data)

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=self.horizontal,
                    remove_nan=self.remove_nan,
                    from_quantile=self.from_quantile,
                    to_quantile=self.to_quantile,
                )


class Heatmap(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        is_x_category: bool = False,
        is_y_category: bool = False,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a heatmap plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`y_labels`, `x_labels`).
            x_labels (array_like): X-axis labels, corresponding to columns in pandas.
            y_labels (array_like): Y-axis labels, corresponding to index in pandas.
            is_x_category (bool): Whether X-axis is a categorical axis.
            is_y_category (bool): Whether Y-axis is a categorical axis.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> heatmap = vbt.Heatmap(
            ...     data=[[1, 2], [3, 4]],
            ...     x_labels=['a', 'b'],
            ...     y_labels=['x', 'y']
            ... )
            >>> heatmap.fig.show()
            ```

            ![](/assets/images/api/Heatmap.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if x_labels is not None:
                checks.assert_shape_equal(data, x_labels, (1, 0))
            if y_labels is not None:
                checks.assert_shape_equal(data, y_labels, (0, 0))
        else:
            if x_labels is None or y_labels is None:
                raise ValueError("At least data, or x_labels and y_labels must be passed")
        if x_labels is not None:
            x_labels = clean_labels(x_labels)
        if y_labels is not None:
            y_labels = clean_labels(y_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                max_width = layout_cfg["width"]
                if data is not None:
                    x_len = data.shape[1]
                    y_len = data.shape[0]
                else:
                    x_len = len(x_labels)
                    y_len = len(y_labels)
                width = math.ceil(rescale(x_len / (x_len + y_len), (0, 1), (0.3 * max_width, max_width)))
                width = min(width + 150, max_width)  # account for colorbar
                height = math.ceil(rescale(y_len / (x_len + y_len), (0, 1), (0.3 * max_width, max_width)))
                height = min(height, max_width * 0.7)  # limit height
                fig.update_layout(width=width, height=height)

        _trace_kwargs = merge_dicts(
            dict(hoverongaps=False, colorscale="Plasma", x=x_labels, y=y_labels),
            trace_kwargs,
        )
        trace = go.Heatmap(**_trace_kwargs)
        if data is not None:
            self.update_trace(trace, data)
        fig.add_trace(trace, **add_trace_kwargs)

        axis_kwargs = dict()
        if is_x_category:
            if fig.data[-1]["xaxis"] is not None:
                axis_kwargs["xaxis" + fig.data[-1]["xaxis"][1:]] = dict(type="category")
            else:
                axis_kwargs["xaxis"] = dict(type="category")
        if is_y_category:
            if fig.data[-1]["yaxis"] is not None:
                axis_kwargs["yaxis" + fig.data[-1]["yaxis"][1:]] = dict(type="category")
            else:
                axis_kwargs["yaxis"] = dict(type="category")
        fig.update_layout(**axis_kwargs)
        fig.update_layout(**layout_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, *args, **kwargs) -> None:
        trace.z = reshaping.to_2d_array(data)

    def update(self, data: tp.ArrayLike) -> None:
        with self.fig.batch_update():
            self.update_trace(self.traces[0], data)


class Volume(Configured, TraceUpdater):
    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        z_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        scene_name: str = "scene",
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs
    ) -> None:
        """Create a volume plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be a 3-dim array.
            x_labels (array_like): X-axis labels.
            y_labels (array_like): Y-axis labels.
            z_labels (array_like): Z-axis labels.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Volume`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            scene_name (str): Reference to the 3D scene.
            make_figure_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.figure.make_figure`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        !!! note
            Figure widgets have currently problems displaying NaNs.
            Use `.show()` method for rendering.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np

            >>> volume = vbt.Volume(
            ...     data=np.random.randint(1, 10, size=(3, 3, 3)),
            ...     x_labels=['a', 'b', 'c'],
            ...     y_labels=['d', 'e', 'f'],
            ...     z_labels=['g', 'h', 'i']
            ... )
            >>> volume.fig.show()
            ```

            ![](/assets/images/api/Volume.svg){: .iimg loading=lazy }
        """
        Configured.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            z_labels=z_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            scene_name=scene_name,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            checks.assert_ndim(data, 3)
            data = np.asarray(data)
            x_len, y_len, z_len = data.shape
            if x_labels is not None:
                checks.assert_shape_equal(data, x_labels, (0, 0))
            if y_labels is not None:
                checks.assert_shape_equal(data, y_labels, (1, 0))
            if z_labels is not None:
                checks.assert_shape_equal(data, z_labels, (2, 0))
        else:
            if x_labels is None or y_labels is None or z_labels is None:
                raise ValueError("At least data, or x_labels, y_labels and z_labels must be passed")
            x_len = len(x_labels)
            y_len = len(y_labels)
            z_len = len(z_labels)
        if x_labels is None:
            x_labels = np.arange(x_len)
        else:
            x_labels = clean_labels(x_labels)
        if y_labels is None:
            y_labels = np.arange(y_len)
        else:
            y_labels = clean_labels(y_labels)
        if z_labels is None:
            z_labels = np.arange(z_len)
        else:
            z_labels = clean_labels(z_labels)
        x_labels = np.asarray(x_labels)
        y_labels = np.asarray(y_labels)
        z_labels = np.asarray(z_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                fig.update_layout(width=layout_cfg["width"], height=0.7 * layout_cfg["width"])

        # Non-numeric data types are not supported by go.Volume, so use ticktext
        # Note: Currently plotly displays the entire tick array, in future versions it will be more sensible
        more_layout = dict()
        more_layout[scene_name] = dict()
        if not np.issubdtype(x_labels.dtype, np.number):
            x_ticktext = x_labels
            x_labels = np.arange(x_len)
            more_layout[scene_name]["xaxis"] = dict(ticktext=x_ticktext, tickvals=x_labels, tickmode="array")
        if not np.issubdtype(y_labels.dtype, np.number):
            y_ticktext = y_labels
            y_labels = np.arange(y_len)
            more_layout[scene_name]["yaxis"] = dict(ticktext=y_ticktext, tickvals=y_labels, tickmode="array")
        if not np.issubdtype(z_labels.dtype, np.number):
            z_ticktext = z_labels
            z_labels = np.arange(z_len)
            more_layout[scene_name]["zaxis"] = dict(ticktext=z_ticktext, tickvals=z_labels, tickmode="array")
        fig.update_layout(**more_layout)
        fig.update_layout(**layout_kwargs)

        # Arrays must have the same length as the flattened data array
        x = np.repeat(x_labels, len(y_labels) * len(z_labels))
        y = np.tile(np.repeat(y_labels, len(z_labels)), len(x_labels))
        z = np.tile(z_labels, len(x_labels) * len(y_labels))

        _trace_kwargs = merge_dicts(
            dict(x=x, y=y, z=z, opacity=0.2, surface_count=15, colorscale="Plasma"),
            trace_kwargs,
        )
        trace = go.Volume(**_trace_kwargs)
        if data is not None:
            self.update_trace(trace, data)
        fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, *args, **kwargs) -> None:
        trace.value = np.asarray(data).flatten()

    def update(self, data: tp.ArrayLike) -> None:
        with self.fig.batch_update():
            self.update_trace(self.traces[0], data)
