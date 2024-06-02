# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with colors."""

import numpy as np

from vectorbtpro import _typing as tp

__all__ = []


def map_value_to_cmap(
    value: tp.MaybeSequence[float],
    cmap: tp.Any,
    vmin: tp.Optional[float] = None,
    vcenter: tp.Optional[float] = None,
    vmax: tp.Optional[float] = None,
) -> tp.MaybeSequence[str]:
    """Get RGB of `value` from the colormap."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    value_is_scalar = np.isscalar(value)
    if value_is_scalar:
        value = np.array([value])
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    elif isinstance(cmap, (tuple, list)):
        cmap = mcolors.LinearSegmentedColormap.from_list("", cmap)
    if vmin is not None and vcenter is not None and vmin > vcenter:
        vmin = vcenter
    if vmin is not None and vcenter is not None and vmin == vcenter:
        vcenter = None
    if vmax is not None and vcenter is not None and vmax < vcenter:
        vmax = vcenter
    if vmax is not None and vcenter is not None and vmax == vcenter:
        vcenter = None
    if vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        value = norm(value)
    elif vmin is not None or vmax is not None:
        if vmin == vmax:
            value = value * 0 + 0.5
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            value = norm(value)
    rgbs = list(map(lambda x: "rgb(%d,%d,%d)" % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), cmap(value)))
    if value_is_scalar:
        return rgbs[0]
    return rgbs


def parse_rgba_tuple(color: str) -> tp.Tuple[float, float, float, float]:
    """Parse floating RGBA tuple from string."""
    rgba = color.replace("rgba", "").replace("(", "").replace(")", "").split(",")
    return int(rgba[0]) / 255, int(rgba[1]) / 255, int(rgba[2]) / 255, float(rgba[3])


def parse_rgb_tuple(color: str) -> tp.Tuple[float, float, float]:
    """Parse floating RGB tuple from string."""
    rgb = color.replace("rgb", "").replace("(", "").replace(")", "").split(",")
    return int(rgb[0]) / 255, int(rgb[1]) / 255, int(rgb[2]) / 255


def adjust_opacity(color: tp.Any, opacity: float) -> str:
    """Adjust opacity of color."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    rgb = mc.to_rgb(color)
    return "rgba(%d,%d,%d,%.4f)" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), opacity)


def adjust_lightness(color: tp.Any, amount: float = 0.7) -> str:
    """Lightens the given color by multiplying (1-luminosity) by the given amount.

    Input can be matplotlib color string, hex string, or RGB tuple.
    Output will be an RGB string."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc
    import colorsys

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return "rgb(%d,%d,%d)" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
