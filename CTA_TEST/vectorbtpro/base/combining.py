# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions for combining arrays.

Combine functions combine two or more NumPy arrays using a custom function. The emphasis here is
done upon stacking the results into one NumPy array - since vectorbt is all about brute-forcing
large spaces of hyper-parameters, concatenating the results of each hyper-parameter combination into
a single DataFrame is important. All functions are available in both Python and Numba-compiled form."""

import numpy as np
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.template import RepFunc
from vectorbtpro.base.reshaping import column_stack

__all__ = []


@register_jitted
def custom_apply_and_concat_none_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.Callable,
    *args,
) -> None:
    """Run `apply_func_nb` that returns nothing for each index.

    Meant for in-place outputs."""
    for i in indices:
        apply_func_nb(i, *args)


@register_jitted
def apply_and_concat_none_nb(
    ntimes: int,
    apply_func_nb: tp.Callable,
    *args,
) -> None:
    """Run `apply_func_nb` that returns nothing number of times.

    Uses `custom_apply_and_concat_none_nb`."""
    custom_apply_and_concat_none_nb(np.arange(ntimes), apply_func_nb, *args)


@register_jitted
def to_2d_one_nb(a: tp.Array) -> tp.Array2d:
    """Expand the dimensions of the array along the axis 1."""
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@register_jitted
def custom_apply_and_concat_one_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Run `apply_func_nb` that returns one array for each index."""
    output_0 = to_2d_one_nb(apply_func_nb(indices[0], *args))
    output = np.empty((output_0.shape[0], len(indices) * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(len(indices)):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(indices[i], *args))
        output[:, i * outputs_i.shape[1] : (i + 1) * outputs_i.shape[1]] = outputs_i
    return output


@register_jitted
def apply_and_concat_one_nb(
    ntimes: int,
    apply_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Run `apply_func_nb` that returns one array number of times.

    Uses `custom_apply_and_concat_one_nb`."""
    return custom_apply_and_concat_one_nb(np.arange(ntimes), apply_func_nb, *args)


@register_jitted
def to_2d_multiple_nb(a: tp.Iterable[tp.Array]) -> tp.List[tp.Array2d]:
    """Expand the dimensions of each array in `a` along axis 1."""
    lst = list()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@register_jitted
def custom_apply_and_concat_multiple_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.Callable,
    *args,
) -> tp.List[tp.Array2d]:
    """Run `apply_func_nb` that returns multiple arrays for each index."""
    outputs = list()
    outputs_0 = to_2d_multiple_nb(apply_func_nb(indices[0], *args))
    for j in range(len(outputs_0)):
        outputs.append(
            np.empty((outputs_0[j].shape[0], len(indices) * outputs_0[j].shape[1]), dtype=outputs_0[j].dtype)
        )
    for i in range(len(indices)):
        if i == 0:
            outputs_i = outputs_0
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(indices[i], *args))
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1] : (i + 1) * outputs_i[j].shape[1]] = outputs_i[j]
    return outputs


@register_jitted
def apply_and_concat_multiple_nb(
    ntimes: int,
    apply_func_nb: tp.Callable,
    *args,
) -> tp.List[tp.Array2d]:
    """Run `apply_func_nb` that returns multiple arrays number of times.

    Uses `custom_apply_and_concat_multiple_nb`."""
    return custom_apply_and_concat_multiple_nb(np.arange(ntimes), apply_func_nb, *args)


def apply_and_concat_each(
    funcs_args: tp.FuncsArgs,
    n_outputs: tp.Optional[int] = None,
    execute_kwargs: tp.KwargsLike = None,
) -> tp.Union[None, tp.Array2d, tp.List[tp.Array2d]]:
    """Apply each function on its own set of positional and keyword arguments.

    Executes the function using `vectorbtpro.utils.execution.execute`."""
    if execute_kwargs is None:
        execute_kwargs = {}

    out = execute(funcs_args, **execute_kwargs)
    if n_outputs is None:
        if out[0] is None:
            n_outputs = 0
        elif isinstance(out[0], (tuple, list, List)):
            n_outputs = len(out[0])
        else:
            n_outputs = 1
    if n_outputs == 0:
        return None
    if n_outputs == 1:
        if isinstance(out[0], (tuple, list, List)) and len(out[0]) == 1:
            out = list(map(lambda x: x[0], out))
        return column_stack(out)
    return list(map(column_stack, zip(*out)))


def apply_and_concat(
    ntimes: int,
    apply_func: tp.Callable,
    *args,
    n_outputs: tp.Optional[int] = None,
    jitted_loop: bool = False,
    jitted_warmup: bool = False,
    execute_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Union[None, tp.Array2d, tp.List[tp.Array2d]]:
    """Run `apply_func` function a number of times and concatenate the results depending upon how
    many array-like objects it generates.

    `apply_func` must accept arguments `i`, `*args`, and `**kwargs`.

    Set `jitted_loop` to True to use the JIT-compiled version.

    All jitted iteration functions are resolved using `vectorbtpro.registries.jit_registry.JITRegistry.resolve`.

    !!! note
        `n_outputs` must be set when `jitted_loop` is True.

        Numba doesn't support variable keyword arguments."""
    if jitted_loop:
        if n_outputs is None:
            raise ValueError("Jitted iteration requires n_outputs")
        if n_outputs == 0:
            func = jit_reg.resolve(custom_apply_and_concat_none_nb)
        elif n_outputs == 1:
            func = jit_reg.resolve(custom_apply_and_concat_one_nb)
        else:
            func = jit_reg.resolve(custom_apply_and_concat_multiple_nb)
        if jitted_warmup:
            func(np.array([0]), apply_func, *args, **kwargs)

        def _funcs_args_template(chunk_meta):
            funcs_args = []
            for _chunk_meta in chunk_meta:
                if _chunk_meta.indices is not None:
                    chunk_indices = np.asarray(_chunk_meta.indices)
                else:
                    if _chunk_meta.start is None or _chunk_meta.end is None:
                        raise ValueError("Each chunk must have a start and an end index")
                    chunk_indices = np.arange(_chunk_meta.start, _chunk_meta.end)
                funcs_args.append((func, (chunk_indices, apply_func, *args), kwargs))
            return funcs_args

        funcs_args = RepFunc(_funcs_args_template)
    else:
        funcs_args = [(apply_func, (i, *args), kwargs) for i in range(ntimes)]
    if execute_kwargs is None:
        execute_kwargs = {}
    execute_kwargs["n_calls"] = ntimes
    return apply_and_concat_each(
        funcs_args,
        n_outputs=n_outputs,
        execute_kwargs=execute_kwargs,
    )


@register_jitted
def select_and_combine_nb(
    i: int,
    obj: tp.Any,
    others: tp.Sequence,
    combine_func_nb: tp.Callable,
    *args,
) -> tp.AnyArray:
    """Numba-compiled version of `select_and_combine`."""
    return combine_func_nb(obj, others[i], *args)


@register_jitted
def combine_and_concat_nb(
    obj: tp.Any,
    others: tp.Sequence,
    combine_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Numba-compiled version of `combine_and_concat`."""
    return apply_and_concat_one_nb(len(others), select_and_combine_nb, obj, others, combine_func_nb, *args)


def select_and_combine(
    i: int,
    obj: tp.Any,
    others: tp.Sequence,
    combine_func: tp.Callable,
    *args,
    **kwargs,
) -> tp.AnyArray:
    """Combine `obj` with an array at position `i` in `others` using `combine_func`."""
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(
    obj: tp.Any,
    others: tp.Sequence,
    combine_func: tp.Callable,
    *args,
    jitted_loop: bool = False,
    **kwargs,
) -> tp.Array2d:
    """Combine `obj` with each in `others` using `combine_func` and concatenate.

    `select_and_combine_nb` is resolved using `vectorbtpro.registries.jit_registry.JITRegistry.resolve`."""
    if jitted_loop:
        apply_func = jit_reg.resolve(select_and_combine_nb)
    else:
        apply_func = select_and_combine
    return apply_and_concat(
        len(others),
        apply_func,
        obj,
        others,
        combine_func,
        *args,
        n_outputs=1,
        jitted_loop=jitted_loop,
        **kwargs,
    )


@register_jitted
def combine_multiple_nb(
    objs: tp.Sequence,
    combine_func_nb: tp.Callable,
    *args,
) -> tp.Any:
    """Numba-compiled version of `combine_multiple`."""
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func_nb(result, objs[i], *args)
    return result


def combine_multiple(
    objs: tp.Sequence,
    combine_func: tp.Callable,
    *args,
    jitted_loop: bool = False,
    **kwargs,
) -> tp.Any:
    """Combine `objs` pairwise into a single object.

    Set `jitted_loop` to True to use the JIT-compiled version.

    `combine_multiple_nb` is resolved using `vectorbtpro.registries.jit_registry.JITRegistry.resolve`.

    !!! note
        Numba doesn't support variable keyword arguments."""
    if jitted_loop:
        func = jit_reg.resolve(combine_multiple_nb)
        return func(objs, combine_func, *args)
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func(result, objs[i], *args, **kwargs)
    return result
