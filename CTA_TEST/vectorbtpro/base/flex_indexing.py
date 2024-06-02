# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes and functions for flexible indexing."""

from vectorbtpro._settings import settings
from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = [
    "flex_select_1d_nb",
    "flex_select_1d_pr_nb",
    "flex_select_1d_pc_nb",
    "flex_select_nb",
]


_rotate_rows = settings["indexing"]["rotate_rows"]
_rotate_cols = settings["indexing"]["rotate_cols"]


@register_jitted(cache=True)
def flex_choose_i_1d_nb(arr: tp.Array2d, i: int) -> int:
    """Choose a position in an array as if it has been broadcast against rows or columns.

    !!! note
        Array must be one-dimensional."""
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    return int(flex_i)


@register_jitted(cache=True)
def flex_select_1d_nb(arr: tp.Array2d, i: int) -> tp.Scalar:
    """Select an element of an array as if it has been broadcast against rows or columns.

    !!! note
        Array must be one-dimensional."""
    flex_i = flex_choose_i_1d_nb(arr, i)
    return arr[flex_i]


@register_jitted(cache=True)
def flex_choose_i_pr_1d_nb(arr: tp.Array2d, i: int, rotate_rows: bool = _rotate_rows) -> int:
    """Choose a position in an array as if it has been broadcast against rows.

    Can use rotational indexing along rows.

    !!! note
        Array must be one-dimensional."""
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    if rotate_rows:
        return int(flex_i) % arr.shape[0]
    return int(flex_i)


@register_jitted(cache=True)
def flex_select_1d_pr_nb(arr: tp.Array2d, i: int, rotate_rows: bool = _rotate_rows) -> tp.Scalar:
    """Select an element of an array as if it has been broadcast against rows.

    Can use rotational indexing along rows.

    !!! note
        Array must be one-dimensional."""
    flex_i = flex_choose_i_pr_1d_nb(arr, i, rotate_rows=rotate_rows)
    return arr[flex_i]


@register_jitted(cache=True)
def flex_choose_i_pc_1d_nb(arr: tp.Array2d, col: int, rotate_cols: bool = _rotate_cols) -> int:
    """Choose a position in an array as if it has been broadcast against columns.

    Can use rotational indexing along columns.

    !!! note
        Array must be one-dimensional."""
    if arr.shape[0] == 1:
        flex_col = 0
    else:
        flex_col = col
    if rotate_cols:
        return int(flex_col) % arr.shape[0]
    return int(flex_col)


@register_jitted(cache=True)
def flex_select_1d_pc_nb(arr: tp.Array2d, col: int, rotate_cols: bool = _rotate_cols) -> tp.Scalar:
    """Select an element of an array as if it has been broadcast against columns.

    Can use rotational indexing along columns.

    !!! note
        Array must be one-dimensional."""
    flex_col = flex_choose_i_pc_1d_nb(arr, col, rotate_cols=rotate_cols)
    return arr[flex_col]


@register_jitted(cache=True)
def flex_choose_i_and_col_nb(
    arr: tp.Array2d,
    i: int,
    col: int,
    rotate_rows: bool = _rotate_rows,
    rotate_cols: bool = _rotate_cols,
) -> tp.Tuple[int, int]:
    """Choose a position in an array as if it has been broadcast rows and columns.

    Can use rotational indexing along rows and columns.

    !!! note
        Array must be two-dimensional."""
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    if arr.shape[1] == 1:
        flex_col = 0
    else:
        flex_col = col
    if rotate_rows and rotate_cols:
        return int(flex_i) % arr.shape[0], int(flex_col) % arr.shape[1]
    if rotate_rows:
        return int(flex_i) % arr.shape[0], int(flex_col)
    if rotate_cols:
        return int(flex_i), int(flex_col) % arr.shape[1]
    return int(flex_i), int(flex_col)


@register_jitted(cache=True)
def flex_select_nb(
    arr: tp.Array2d,
    i: int,
    col: int,
    rotate_rows: bool = _rotate_rows,
    rotate_cols: bool = _rotate_cols,
) -> tp.Scalar:
    """Select element of an array as if it has been  broadcast rows and columns.

    Can use rotational indexing along rows and columns.

    !!! note
        Array must be two-dimensional."""
    flex_i, flex_col = flex_choose_i_and_col_nb(
        arr,
        i,
        col,
        rotate_rows=rotate_rows,
        rotate_cols=rotate_cols,
    )
    return arr[flex_i, flex_col]
