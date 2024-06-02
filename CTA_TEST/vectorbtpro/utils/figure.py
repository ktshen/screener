# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for constructing and displaying figures."""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

from plotly.graph_objects import Figure as _Figure, FigureWidget as _FigureWidget
from plotly.subplots import make_subplots as _make_subplots

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "Figure",
    "FigureWidget",
    "make_figure",
    "make_subplots",
]


def get_domain(ref: str, fig: tp.BaseFigure) -> tp.Tuple[int, int]:
    """Get domain of a coordinate axis."""
    axis = ref[0] + "axis" + ref[1:]
    if axis in fig.layout:
        if "domain" in fig.layout[axis]:
            if fig.layout[axis]["domain"] is not None:
                return fig.layout[axis]["domain"]
    return 0, 1


class FigureMixin:
    def show(self, *args, **kwargs) -> None:
        """Display the figure in PNG format."""
        raise NotImplementedError

    def show_png(self, **kwargs) -> None:
        """Display the figure in PNG format."""
        self.show(renderer="png", **kwargs)

    def show_svg(self, **kwargs) -> None:
        """Display the figure in SVG format."""
        self.show(renderer="svg", **kwargs)


class Figure(_Figure, FigureMixin):
    """Figure."""

    def __init__(self, *args, **kwargs) -> None:
        """Extends `plotly.graph_objects.Figure`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        """Show the figure."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
        _Figure.show(self, *args, **show_kwargs)


class FigureWidget(_FigureWidget, FigureMixin):
    """Figure widget."""

    def __init__(self, *args, **kwargs) -> None:
        """Extends `plotly.graph_objects.FigureWidget`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        """Show the figure."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
        _Figure.show(self, *args, **show_kwargs)


try:
    from plotly_resampler import FigureResampler as _FigureResampler, FigureWidgetResampler as _FigureWidgetResampler

    class FigureResampler(_FigureResampler, FigureMixin):
        """Figure resampler."""

        def __init__(self, *args, **kwargs) -> None:
            """Extends `plotly.graph_objects.Figure`."""
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            layout = kwargs.pop("layout", {})
            super().__init__(*args, **kwargs)
            self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

        def show(self, *args, **kwargs) -> None:
            """Show the figure."""
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
            show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
            _Figure.show(self, *args, **show_kwargs)

    class FigureWidgetResampler(_FigureWidgetResampler, FigureMixin):
        """Figure widget resampler."""

        def __init__(self, *args, **kwargs) -> None:
            """Extends `plotly.graph_objects.FigureWidget`."""
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            layout = kwargs.pop("layout", {})
            super().__init__(*args, **kwargs)
            self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

        def show(self, *args, **kwargs) -> None:
            """Show the figure."""
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
            show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
            _Figure.show(self, *args, **show_kwargs)

except ImportError:
    FigureResampler = Figure
    FigureWidgetResampler = FigureWidget


def make_figure(
    *args,
    use_widgets: tp.Optional[bool] = None,
    use_resampler: tp.Optional[bool] = None,
    **kwargs,
) -> tp.BaseFigure:
    """Make a new Plotly figure.

    If `use_widgets` is True, returns `FigureWidget`, otherwise `Figure`.

    If `use_resampler` is True, additionally wraps the class using `plotly_resampler`.

    Defaults are defined under `vectorbtpro._settings.plotting`."""
    from vectorbtpro._settings import settings

    plotting_cfg = settings["plotting"]

    if use_widgets is None:
        use_widgets = plotting_cfg["use_widgets"]
    if use_resampler is None:
        use_resampler = plotting_cfg["use_resampler"]

    if use_widgets:
        if use_resampler is None:
            return FigureWidgetResampler(*args, **kwargs)
        if use_resampler:
            assert_can_import("plotly_resampler")
            return FigureWidgetResampler(*args, **kwargs)
        return FigureWidget(*args, **kwargs)
    if use_resampler is None:
        return FigureResampler(*args, **kwargs)
    if use_resampler:
        assert_can_import("plotly_resampler")
        return FigureResampler(*args, **kwargs)
    return Figure(*args, **kwargs)


def make_subplots(
    *args,
    use_widgets: tp.Optional[bool] = None,
    use_resampler: tp.Optional[bool] = None,
    **kwargs,
) -> tp.BaseFigure:
    """Make Plotly subplots using `make_figure`."""
    return make_figure(_make_subplots(*args, **kwargs), use_widgets=use_widgets, use_resampler=use_resampler)
