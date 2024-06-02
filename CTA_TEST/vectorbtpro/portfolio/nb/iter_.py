# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for iterative portfolio simulation."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.generic.nb.iter_ import (
    iter_above_nb as _iter_above_nb,
    iter_below_nb as _iter_below_nb,
    iter_crossed_above_nb as _iter_crossed_above_nb,
    iter_crossed_below_nb as _iter_crossed_below_nb,
)
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted
def select_nb(
    ctx: tp.NamedTuple,
    arr: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> tp.Scalar:
    """Get the current element using flexible indexing.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    if col is None:
        _col = ctx.col
    else:
        _col = col
    return flex_select_nb(arr, _i, _col)


@register_jitted
def select_from_col_nb(
    ctx: tp.NamedTuple,
    col: int,
    arr: tp.FlexArray2d,
    i: tp.Optional[int] = None,
) -> tp.Scalar:
    """Get the current element from a specific column using flexible indexing.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    return flex_select_nb(arr, _i, col)


@register_jitted
def iter_above_nb(
    ctx: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_above_nb` on the context.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    if col is None:
        _col = ctx.col
    else:
        _col = col
    return _iter_above_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_below_nb(
    ctx: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_below_nb` on the context.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    if col is None:
        _col = ctx.col
    else:
        _col = col
    return _iter_below_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_crossed_above_nb(
    ctx: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_above_nb` on the context.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    if col is None:
        _col = ctx.col
    else:
        _col = col
    return _iter_crossed_above_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_crossed_below_nb(
    ctx: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_below_nb` on the context.

    If any of the arguments are None, will use the respective value from the context."""
    if i is None:
        _i = ctx.i
    else:
        _i = i
    if col is None:
        _col = ctx.col
    else:
        _col = col
    return _iter_crossed_below_nb(arr1, arr2, _i, _col)
