# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions for merging arrays."""

from functools import partial

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import resolve_dict, merge_dicts
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.reshaping import column_stack, to_2d_array

__all__ = [
    "concat_merge",
    "row_stack_merge",
    "column_stack_merge",
]


def concat_merge(
    *objs,
    keys: tp.Optional[tp.Index] = None,
    wrap: tp.Optional[bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like objects through concatenation.

    Supports a sequence of tuples.

    If `wrap` is None, it will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None.
    If `wrap` is True, each array will be wrapped with Pandas Series and merged using `pd.concat`.
    Otherwise, arrays will be kept as-is and merged using `np.concatenate`.
    `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

    Keyword arguments `**kwargs` are passed to `pd.concat` only.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                concat_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: concat_merge(
                    x,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )
    if isinstance(objs[0], Wrapping):
        raise TypeError("Concatenating Wrapping instances is not supported")

    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if not checks.is_iterable(objs[0]) or isinstance(objs[0], str):
        if wrap:
            wrap_kwargs = merge_dicts(dict(index=keys), wrap_kwargs)
            return pd.Series(objs, **wrap_kwargs)
        return np.asarray(objs)
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))
    if not isinstance(objs[0], pd.Series):
        if isinstance(objs[0], pd.DataFrame):
            raise ValueError("Use row stacking for concatenating DataFrames")
        if wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    if "force_1d" not in _wrap_kwargs:
                        _wrap_kwargs["force_1d"] = True
                    new_objs.append(wrapper.wrap_reduced(obj, **_wrap_kwargs))
                else:
                    new_objs.append(pd.Series(obj, **_wrap_kwargs))
            objs = new_objs
        else:
            return np.concatenate(objs)
    return pd.concat(objs, axis=0, keys=keys, **kwargs)


def row_stack_merge(
    *objs,
    keys: tp.Optional[tp.Index] = None,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects through row stacking.

    Supports a sequence of tuples.

    Argument `wrap` supports the following options:

    * None: will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None
    * True: each array will be wrapped with Pandas Series/DataFrame (depending on dimensions)
    * 'sr', 'series': each array will be wrapped with Pandas Series
    * 'df', 'frame', 'dataframe': each array will be wrapped with Pandas DataFrame

    Argument `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

    Keyword arguments `**kwargs` are passed to `pd.concat` and
    `vectorbtpro.base.wrapping.Wrapping.row_stack` only.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                row_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: row_stack_merge(
                    x,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )

    if isinstance(objs[0], Wrapping):
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).row_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option '{wrap}'")
            objs = new_objs
        else:
            return np.row_stack(objs)
    return pd.concat(objs, axis=0, keys=keys, **kwargs)


def column_stack_merge(
    *objs,
    reset_index: tp.Union[None, bool, str] = None,
    fill_value: tp.Scalar = np.nan,
    keys: tp.Optional[tp.Index] = None,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects through column stacking.

    Supports a sequence of tuples.

    Argument `wrap` supports the following options:

    * None: will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None
    * True: each array will be wrapped with Pandas Series/DataFrame (depending on dimensions)
    * 'sr', 'series': each array will be wrapped with Pandas Series
    * 'df', 'frame', 'dataframe': each array will be wrapped with Pandas DataFrame

    Argument `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

    Keyword arguments `**kwargs` are passed to `pd.concat` and
    `vectorbtpro.base.wrapping.Wrapping.column_stack` only.

    Argument `reset_index` supports the following options:

    * False or None: Keep original index of each object
    * True or 'from_start': Reset index of each object and align them at start
    * 'from_end': Reset index of each object and align them at end

    Options above work on Pandas, NumPy, and `vectorbtpro.base.wrapping.Wrapping` instances.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if isinstance(reset_index, bool):
        if reset_index:
            reset_index = "from_start"
        else:
            reset_index = None

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                column_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    reset_index=reset_index,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: column_stack_merge(
                    x,
                    reset_index=reset_index,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )

    if isinstance(objs[0], Wrapping):
        if reset_index is not None:
            max_length = max(map(lambda x: x.wrapper.shape[0], objs))
            new_objs = []
            for obj in objs:
                if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                    new_index = pd.RangeIndex(stop=obj.wrapper.shape[0])
                    new_obj = obj.replace(wrapper=obj.wrapper.replace(index=new_index))
                elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                    new_index = pd.RangeIndex(start=max_length - obj.wrapper.shape[0], stop=max_length)
                    new_obj = obj.replace(wrapper=obj.wrapper.replace(index=new_index))
                else:
                    raise ValueError(f"Invalid index resetting option '{reset_index}'")
                new_objs.append(new_obj)
            objs = new_objs
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).column_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option '{wrap}'")
            objs = new_objs
        else:
            if reset_index is not None:
                min_n_rows = None
                max_n_rows = None
                n_cols = 0
                new_objs = []
                for obj in objs:
                    new_obj = to_2d_array(obj)
                    new_objs.append(new_obj)
                    if min_n_rows is None or new_obj.shape[0] < min_n_rows:
                        min_n_rows = new_obj.shape[0]
                    if max_n_rows is None or new_obj.shape[0] > min_n_rows:
                        max_n_rows = new_obj.shape[0]
                    n_cols += new_obj.shape[1]
                if min_n_rows == max_n_rows:
                    return column_stack(new_objs)
                new_obj = np.full((max_n_rows, n_cols), fill_value)
                start_col = 0
                for obj in new_objs:
                    end_col = start_col + obj.shape[1]
                    if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                        new_obj[:len(obj), start_col:end_col] = obj
                    elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                        new_obj[-len(obj):, start_col:end_col] = obj
                    else:
                        raise ValueError(f"Invalid index resetting option '{reset_index}'")
                    start_col = end_col
                return new_obj
            return column_stack(objs)
    if reset_index is not None:
        max_length = max(map(len, objs))
        new_objs = []
        for obj in objs:
            new_obj = obj.copy(deep=False)
            if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                new_obj.index = pd.RangeIndex(stop=len(new_obj))
            elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                new_obj.index = pd.RangeIndex(start=max_length - len(new_obj), stop=max_length)
            else:
                raise ValueError(f"Invalid index resetting option '{reset_index}'")
            new_objs.append(new_obj)
        objs = new_objs
        kwargs = merge_dicts(dict(sort=True), kwargs)
    return pd.concat(objs, axis=1, keys=keys, **kwargs)


def mixed_merge(
    *objs,
    func_names: tp.Optional[tp.Tuple[str, ...]] = None,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    keys: tp.Optional[tp.Index] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge objects of mixed types."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if func_names is None:
        raise ValueError("Merging function names are required")
    if not isinstance(objs[0], tuple):
        raise ValueError("Mixed merging must be applied on tuples")

    outputs = []
    for i, obj_kind in enumerate(zip(*objs)):
        func_name = func_names[i]
        merge_func = resolve_merge_func(func_name)
        output = merge_func(
            obj_kind,
            keys=keys,
            wrap=wrap,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )
        outputs.append(output)
    return tuple(outputs)


def resolve_merge_func(func_name: tp.MaybeTuple[str]) -> tp.Callable:
    """Resolve merging function based on name.

    * Tuple of strings: `mixed_merge` with `func_names=func_name`
    * String "concat": `concat_merge`
    * String "row_stack": `column_stack_merge`
    * String "column_stack": `column_stack_merge`
    * String "reset_column_stack": `column_stack_merge` with `reset_index=True`
    * String "from_start_column_stack": `column_stack_merge` with `reset_index="from_start"`
    * String "from_end_column_stack": `column_stack_merge` with `reset_index="from_end"`
    """
    if isinstance(func_name, tuple):
        return partial(mixed_merge, func_names=func_name)
    if func_name.lower() == "concat":
        return concat_merge
    if func_name.lower() == "row_stack":
        return row_stack_merge
    if func_name.lower() == "column_stack":
        return column_stack_merge
    if func_name.lower() == "reset_column_stack":
        return partial(column_stack_merge, reset_index=True)
    if func_name.lower() == "from_start_column_stack":
        return partial(column_stack_merge, reset_index="from_start")
    if func_name.lower() == "from_end_column_stack":
        return partial(column_stack_merge, reset_index="from_end")
    raise ValueError(f"Invalid merging function name '{func_name}'")
