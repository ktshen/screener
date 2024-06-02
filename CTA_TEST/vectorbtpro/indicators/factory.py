# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Factory for building indicators.

Run for the examples below:

```pycon
>>> import vectorbtpro as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> price = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5),
... ])).astype(float)
>>> price
            a    b
2020-01-01  1.0  5.0
2020-01-02  2.0  4.0
2020-01-03  3.0  3.0
2020-01-04  4.0  2.0
2020-01-05  5.0  1.0
```"""

import functools
import inspect
import itertools
import re
import warnings
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from types import ModuleType

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, reshaping, combining
from vectorbtpro.base.indexing import build_param_indexer
from vectorbtpro.base.reshaping import broadcast_array_to, broadcast_arrays, Default, resolve_ref, column_stack
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import BaseAccessor
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.indicators.expr import expr_func_config, expr_res_func_config, wqa101_expr_config
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import build_nan_mask, squeeze_nan, unsqueeze_nan
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts, resolve_dict, Config, Configured
from vectorbtpro.utils.decorators import classproperty, cacheable_property, class_or_instancemethod
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.eval_ import multiline_eval
from vectorbtpro.utils.formatting import prettify
from vectorbtpro.utils.mapping import to_value_mapping, apply_mapping
from vectorbtpro.utils.params import to_typed_list, broadcast_params, create_param_product, params_to_list
from vectorbtpro.utils.parsing import glob2re, get_expr_var_names, get_func_arg_names, get_func_kwargs, supress_stdout
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import has_templates, substitute_templates
from vectorbtpro.utils.datetime_ import freq_to_timedelta64, infer_index_freq
from vectorbtpro.utils.module_ import search_package_for_funcs

__all__ = [
    "IndicatorBase",
    "IndicatorFactory",
    "IF",
    "indicator",
    "talib",
    "pandas_ta",
    "ta",
    "wqa101",
    "technical",
    "techcon",
]

__pdoc__ = {}

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from ta.utils import IndicatorMixin as IndicatorMixinT
except ImportError:
    IndicatorMixinT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from technical.consensus import Consensus as ConsensusT
except ImportError:
    ConsensusT = tp.Any


def prepare_params(
    param_list: tp.Sequence[tp.Params],
    param_names: tp.Sequence[str],
    param_settings: tp.Sequence[tp.KwargsLike],
    input_shape: tp.Optional[tp.Shape] = None,
    to_2d: bool = False,
    context: tp.KwargsLike = None,
) -> tp.List[tp.Params]:
    """Prepare parameters.

    Resolves references and performs broadcasting to the input shape."""
    # Resolve references
    if context is None:
        context = {}
    pool = dict(zip(param_names, param_list))
    for k in pool:
        pool[k] = resolve_ref(pool, k)
    param_list = [pool[k] for k in param_names]

    new_param_list = []
    for i, p_values in enumerate(param_list):
        # Resolve settings
        _param_settings = resolve_dict(param_settings[i])
        is_tuple = _param_settings.get("is_tuple", False)
        dtype = _param_settings.get("dtype", None)
        if checks.is_mapping_like(dtype):
            if checks.is_namedtuple(dtype):
                p_values = map_enum_fields(p_values, dtype)
            else:
                p_values = apply_mapping(p_values, dtype)
        is_array_like = _param_settings.get("is_array_like", False)
        min_one_dim = _param_settings.get("min_one_dim", False)
        bc_to_input = _param_settings.get("bc_to_input", False)
        broadcast_kwargs = merge_dicts(
            dict(require_kwargs=dict(requirements="W")),
            _param_settings.get("broadcast_kwargs", None),
        )
        template = _param_settings.get("template", None)

        new_p_values = params_to_list(p_values, is_tuple, is_array_like)
        if template is not None:
            new_p_values = [
                template.substitute(context={param_names[i]: new_p_values[j], **context})
                for j in range(len(new_p_values))
            ]
        if not bc_to_input:
            if is_array_like:
                if min_one_dim:
                    new_p_values = list(map(reshaping.to_1d_array, new_p_values))
                else:
                    new_p_values = list(map(np.asarray, new_p_values))
        else:
            # Broadcast to input or its axis
            if is_tuple:
                raise ValueError("Cannot broadcast to input if tuple")
            if input_shape is None:
                raise ValueError("Cannot broadcast to input if input shape is unknown. Pass input_shape.")
            if bc_to_input is True:
                to_shape = input_shape
            else:
                checks.assert_in(bc_to_input, (0, 1))
                # Note that input_shape can be 1D
                if bc_to_input == 0:
                    to_shape = (input_shape[0],)
                else:
                    to_shape = (input_shape[1],) if len(input_shape) > 1 else (1,)
            _new_p_values = reshaping.broadcast(*new_p_values, to_shape=to_shape, **broadcast_kwargs)
            if len(new_p_values) == 1:
                _new_p_values = [_new_p_values]
            else:
                _new_p_values = list(_new_p_values)
            if to_2d and bc_to_input is True:
                # If inputs are meant to reshape to 2D, do the same to parameters
                # But only to those that fully resemble inputs (= not raw)
                __new_p_values = _new_p_values.copy()
                for j, param in enumerate(__new_p_values):
                    keep_flex = broadcast_kwargs.get("keep_flex", False)
                    if keep_flex is False or (isinstance(keep_flex, (tuple, list)) and not keep_flex[j]):
                        __new_p_values[j] = reshaping.to_2d(param)
                new_p_values = __new_p_values
            else:
                new_p_values = _new_p_values
        new_param_list.append(new_p_values)
    return new_param_list


def build_columns(
    param_list: tp.Sequence[tp.Params],
    input_columns: tp.IndexLike,
    level_names: tp.Optional[tp.Sequence[str]] = None,
    hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
    single_value: tp.Optional[tp.Sequence[bool]] = None,
    param_settings: tp.KwargsLikeSequence = None,
    per_column: bool = False,
    ignore_ranges: bool = False,
    **kwargs,
) -> tp.Tuple[tp.List[tp.Index], tp.Index]:
    """For each parameter in `param_list`, create a new column level with parameter values
    and stack it on top of `input_columns`.

    Returns a list of parameter indexes and new columns."""
    if level_names is not None:
        checks.assert_len_equal(param_list, level_names)
    if hide_levels is None:
        hide_levels = []
    input_columns = indexes.to_any_index(input_columns)

    param_indexes = []
    shown_param_indexes = []
    for i in range(len(param_list)):
        p_values = param_list[i]
        level_name = None
        if level_names is not None:
            level_name = level_names[i]
        _single_value = False
        if single_value is not None:
            _single_value = single_value[i]
        _param_settings = resolve_dict(param_settings, i=i)
        dtype = _param_settings.get("dtype", None)
        if checks.is_mapping_like(dtype):
            if checks.is_namedtuple(dtype):
                dtype = to_value_mapping(dtype, reverse=False)
            else:
                dtype = to_value_mapping(dtype, reverse=True)
            p_values = apply_mapping(p_values, dtype)
        _per_column = _param_settings.get("per_column", False)
        _post_index_func = _param_settings.get("post_index_func", None)
        if per_column:
            param_index = indexes.index_from_values(p_values, single_value=_single_value, name=level_name)
        else:
            if _per_column:
                param_index = None
                for p in p_values:
                    bc_param = broadcast_array_to(p, len(input_columns))
                    _param_index = indexes.index_from_values(bc_param, single_value=False, name=level_name)
                    if param_index is None:
                        param_index = _param_index
                    else:
                        param_index = param_index.append(_param_index)
                if len(param_index) == 1 and len(input_columns) > 1:
                    # When using flexible column-wise parameters
                    param_index = indexes.repeat_index(param_index, len(input_columns), ignore_ranges=ignore_ranges)
            else:
                param_index = indexes.index_from_values(p_values, single_value=_single_value, name=level_name)
                param_index = indexes.repeat_index(param_index, len(input_columns), ignore_ranges=ignore_ranges)
        if _post_index_func is not None:
            param_index = _post_index_func(param_index)
        param_indexes.append(param_index)
        if i not in hide_levels and (level_names is None or level_names[i] not in hide_levels):
            shown_param_indexes.append(param_index)
    if not per_column:
        n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
        input_columns = indexes.tile_index(input_columns, n_param_values, ignore_ranges=ignore_ranges)
    if len(shown_param_indexes) > 0:
        stacked_columns = indexes.stack_indexes([*shown_param_indexes, input_columns], **kwargs)
    else:
        stacked_columns = input_columns
    return param_indexes, stacked_columns


def combine_objs(
    obj: tp.SeriesFrame,
    other: tp.MaybeTupleList[tp.Union[tp.ArrayLike, BaseAccessor]],
    combine_func: tp.Callable,
    *args,
    level_name: tp.Optional[str] = None,
    keys: tp.Optional[tp.IndexLike] = None,
    allow_multiple: bool = True,
    **kwargs,
) -> tp.SeriesFrame:
    """Combines/compares `obj` to `other`, for example, to generate signals.

    Both will broadcast together.
    Pass `other` as a tuple or a list to compare with multiple arguments.
    In this case, a new column level will be created with the name `level_name`.

    See `vectorbtpro.base.accessors.BaseAccessor.combine`."""
    if allow_multiple and isinstance(other, (tuple, list)):
        if keys is None:
            keys = indexes.index_from_values(other, name=level_name)
    return obj.vbt.combine(other, combine_func, *args, keys=keys, allow_multiple=allow_multiple, **kwargs)


IndicatorBaseT = tp.TypeVar("IndicatorBaseT", bound="IndicatorBase")
CacheOutputT = tp.Any
RawOutputT = tp.Tuple[
    tp.List[tp.Array2d],
    tp.List[tp.Tuple[tp.Param, ...]],
    int,
    tp.List[tp.Any],
]
InputListT = tp.List[tp.Array2d]
InputMapperT = tp.Optional[tp.Array1d]
InOutputListT = tp.List[tp.Array2d]
OutputListT = tp.List[tp.Array2d]
ParamListT = tp.List[tp.List[tp.Param]]
MapperListT = tp.List[tp.Index]
OtherListT = tp.List[tp.Any]
PipelineOutputT = tp.Tuple[
    ArrayWrapper,
    InputListT,
    InputMapperT,
    InOutputListT,
    OutputListT,
    ParamListT,
    MapperListT,
    OtherListT,
]
RunOutputT = tp.Union[IndicatorBaseT, tp.Tuple[tp.Any, ...], RawOutputT, CacheOutputT]
RunCombsOutputT = tp.Tuple[IndicatorBaseT, ...]


class IndicatorBase(Analyzable):
    """Indicator base class.

    Properties should be set before instantiation."""

    _short_name: tp.ClassVar[str]
    _input_names: tp.ClassVar[tp.Tuple[str, ...]]
    _param_names: tp.ClassVar[tp.Tuple[str, ...]]
    _in_output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _lazy_output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _output_flags: tp.ClassVar[tp.Kwargs]
    _level_names: tp.Tuple[str, ...]

    @classmethod
    def run_pipeline(
        cls,
        num_ret_outputs: int,
        custom_func: tp.Callable,
        *args,
        require_input_shape: bool = False,
        input_shape: tp.Optional[tp.ShapeLike] = None,
        input_index: tp.Optional[tp.IndexLike] = None,
        input_columns: tp.Optional[tp.IndexLike] = None,
        inputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_outputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_output_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        params: tp.Optional[tp.MappingSequence[tp.Params]] = None,
        param_product: bool = False,
        random_subset: tp.Optional[int] = None,
        param_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        run_unique: bool = False,
        silence_warnings: bool = False,
        per_column: tp.Optional[bool] = None,
        keep_pd: bool = False,
        to_2d: bool = True,
        pass_packed: bool = False,
        pass_input_shape: tp.Optional[bool] = None,
        pass_wrapper: bool = False,
        level_names: tp.Optional[tp.Sequence[str]] = None,
        hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
        build_col_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        use_raw: tp.Optional[RawOutputT] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> tp.Union[CacheOutputT, RawOutputT, PipelineOutputT]:
        """A pipeline for running an indicator, used by `IndicatorFactory`.

        Args:
            num_ret_outputs (int): The number of output arrays returned by `custom_func`.
            custom_func (callable): A custom calculation function.

                See `IndicatorFactory.with_custom_func`.
            *args: Arguments passed to the `custom_func`.
            require_input_shape (bool): Whether to input shape is required.

                Will set `pass_input_shape` to True and raise an error if `input_shape` is None.
            input_shape (tuple): Shape to broadcast each input to.

                Can be passed to `custom_func`. See `pass_input_shape`.
            input_index (index_like): Sets index of each input.

                Can be used to label index if no inputs passed.
            input_columns (index_like): Sets columns of each input.

                Can be used to label columns if no inputs passed.
            inputs (mapping or sequence of array_like): A mapping or sequence of input arrays.

                Use mapping to also supply names. If sequence, will convert to a mapping using `input_{i}` key.
            in_outputs (mapping or sequence of array_like): A mapping or sequence of in-place output arrays.

                Use mapping to also supply names. If sequence, will convert to a mapping using `in_output_{i}` key.
            in_output_settings (dict or sequence of dict): Settings corresponding to each in-place output.

                If mapping, should contain keys from `in_outputs`.

                Following keys are accepted:

                * `dtype`: Create this array using this data type and `np.empty`. Default is None.
            broadcast_named_args (dict): Dictionary with named arguments to broadcast together with inputs.

                You can then pass argument names wrapped with `vectorbtpro.utils.template.Rep`
                and this method will substitute them by their corresponding broadcasted objects.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`
                to broadcast inputs.
            template_context (dict): Mapping used to substitute templates in `args` and `kwargs`.
            params (mapping or sequence of any): A mapping or sequence of parameters.

                Use mapping to also supply names. If sequence, will convert to a mapping using `param_{i}` key.

                Each element is either an array-like object or a single value of any type.
            param_product (bool): Whether to build a Cartesian product out of all parameters.
            random_subset (int): Number of parameter combinations to pick randomly.
            param_settings (dict or sequence of dict): Settings corresponding to each parameter.

                If mapping, should contain keys from `params`.

                Following keys are accepted:

                * `dtype`: If data type is an enumerated type or other mapping, and a string as parameter
                    value was passed, will convert it first.
                * `is_tuple`: If tuple was passed, it will be considered as a single value.
                    To treat it as multiple values, pack it into a list.
                * `is_array_like`: If array-like object was passed, it will be considered as a single value.
                    To treat it as multiple values, pack it into a list.
                * `template`: Template to substitute each parameter value with, before broadcasting to input.
                * `min_one_dim`: Whether to convert any scalar into a one-dimensional array.
                    Works only if `bc_to_input` is False.
                * `bc_to_input`: Whether to broadcast parameter to input size. You can also broadcast
                    parameter to an axis by passing an integer.
                * `broadcast_kwargs`: Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
                * `per_column`: Whether each parameter value can be split by columns such that it can
                    be better reflected in a multi-index. Does not affect broadcasting.
                * `post_index_func`: Function to convert the final index level of the parameter. Defaults to None.
            run_unique (bool): Whether to run only on unique parameter combinations.

                Disable if two identical parameter combinations can lead to different results
                (e.g., due to randomness) or if inputs are large and `custom_func` is fast.

                !!! note
                    Cache, raw output, and output objects outside of `num_ret_outputs` will be returned
                    for unique parameter combinations only.
            silence_warnings (bool): Whether to hide warnings such as coming from `run_unique`.
            per_column (bool): Whether the values of each parameter should be split by columns.

                Defaults to False. Will pass `per_column` if it's not None.

                Each list of parameter values will broadcast to the number of columns and
                each parameter value will be applied per column rather than per whole input.
                Input shape must be known beforehand.

                Each from inputs, in-outputs, and parameters will be passed to `custom_func`
                with the full shape. Expects the outputs be of the same shape as inputs.
            keep_pd (bool): Whether to keep inputs as pandas objects, otherwise convert to NumPy arrays.
            to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
            pass_packed (bool): Whether to pass inputs and parameters to `custom_func` as lists.

                If `custom_func` is Numba-compiled, passes tuples.
            pass_input_shape (bool): Whether to pass `input_shape` to `custom_func` as keyword argument.

                Defaults to True if `require_input_shape` is True, otherwise to False.
            pass_wrapper (bool): Whether to pass the input wrapper to `custom_func` as keyword argument.
            level_names (list of str): A list of column level names corresponding to each parameter.

                Must have the same length as `param_list`.
            hide_levels (list of int or str): A list of level names or indices of parameter levels to hide.
            build_col_kwargs (dict): Keyword arguments passed to `build_columns`.
            return_raw (bool): Whether to return raw output without post-processing and hashed parameter tuples.
            use_raw (bool): Takes the raw results and uses them instead of running `custom_func`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            seed (int): Set seed to make output deterministic.
            **kwargs: Keyword arguments passed to the `custom_func`.

                Some common arguments include `return_cache` to return cache and `use_cache` to use cache.
                If `use_cache` is False, disables caching completely. Those are only applicable to `custom_func`
                that supports it (`custom_func` created using `IndicatorFactory.with_apply_func` are supported by default).

        Returns:
            Array wrapper, list of inputs (`np.ndarray`), input mapper (`np.ndarray`), list of outputs
            (`np.ndarray`), list of parameter arrays (`np.ndarray`), list of parameter mappers (`np.ndarray`),
            list of outputs that are outside of `num_ret_outputs`.
        """
        pass_per_column = per_column is not None
        if per_column is None:
            per_column = False
        if len(params) == 0 and per_column:
            raise ValueError("per_column cannot be enabled without parameters")
        if require_input_shape:
            checks.assert_not_none(input_shape)
            if pass_input_shape is None:
                pass_input_shape = True
        if pass_input_shape is None:
            pass_input_shape = False
        if input_index is not None:
            input_index = indexes.to_any_index(input_index)
        if input_columns is not None:
            input_columns = indexes.to_any_index(input_columns)
        if inputs is None:
            inputs = {}
        if not checks.is_mapping(inputs):
            inputs = {"input_" + str(i): input for i, input in enumerate(inputs)}
        input_names = list(inputs.keys())
        input_list = list(inputs.values())
        if in_outputs is None:
            in_outputs = {}
        if not checks.is_mapping(in_outputs):
            in_outputs = {"in_output_" + str(i): in_output for i, in_output in enumerate(in_outputs)}
        in_output_names = list(in_outputs.keys())
        in_output_list = list(in_outputs.values())
        if in_output_settings is None:
            in_output_settings = {}
        if checks.is_mapping(in_output_settings):
            checks.assert_dict_valid(in_output_settings, [in_output_names, "dtype"])
            in_output_settings = [in_output_settings.get(k, None) for k in in_output_names]
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}
        if params is None:
            params = {}
        if not checks.is_mapping(params):
            params = {"param_" + str(i): param for i, param in enumerate(params)}
        param_names = list(params.keys())
        param_list = list(params.values())
        if param_settings is None:
            param_settings = {}
        if checks.is_mapping(param_settings):
            checks.assert_dict_valid(
                param_settings,
                [
                    param_names,
                    [
                        "dtype",
                        "is_tuple",
                        "is_array_like",
                        "template",
                        "min_one_dim",
                        "bc_to_input",
                        "broadcast_kwargs",
                        "per_column",
                        "post_index_func",
                    ],
                ],
            )
            param_settings = [param_settings.get(k, None) for k in param_names]
        if hide_levels is None:
            hide_levels = []
        if build_col_kwargs is None:
            build_col_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if keep_pd and checks.is_numba_func(custom_func):
            raise ValueError("Cannot pass pandas objects to a Numba-compiled custom_func. Set keep_pd to False.")

        # Set seed
        if seed is not None:
            set_seed(seed)

        if input_shape is not None:
            input_shape = reshaping.to_tuple_shape(input_shape)
        if len(inputs) > 0 or len(in_outputs) > 0 or len(broadcast_named_args) > 0:
            # Broadcast inputs, in-outputs, and named args
            # If input_shape is provided, will broadcast all inputs to this shape
            broadcast_args = merge_dicts(inputs, in_outputs, broadcast_named_args)
            broadcast_kwargs = merge_dicts(
                dict(
                    to_shape=input_shape,
                    index_from=input_index,
                    columns_from=input_columns,
                    require_kwargs=dict(requirements="W"),
                    post_func=None if keep_pd else np.asarray,
                    to_pd=True,
                ),
                broadcast_kwargs,
            )
            broadcast_args, wrapper = reshaping.broadcast(broadcast_args, return_wrapper=True, **broadcast_kwargs)
            input_shape, input_index, input_columns = wrapper.shape, wrapper.index, wrapper.columns
            if input_index is None:
                input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
            if input_columns is None:
                input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)
            input_list = [broadcast_args[input_name] for input_name in input_names]
            in_output_list = [broadcast_args[in_output_name] for in_output_name in in_output_names]
            broadcast_named_args = {arg_name: broadcast_args[arg_name] for arg_name in broadcast_named_args}
        else:
            wrapper = None

        # Reshape input shape
        input_shape_ready = input_shape
        input_shape_2d = input_shape
        if input_shape is not None:
            input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
        if to_2d:
            if input_shape is not None:
                input_shape_ready = input_shape_2d  # ready for custom_func
        if wrapper is not None:
            wrapper_ready = wrapper
        elif input_index is not None and input_columns is not None and input_shape_ready is not None:
            wrapper_ready = ArrayWrapper(input_index, input_columns, len(input_shape_ready))
        else:
            wrapper_ready = None

        # Prepare inputs
        input_list_ready = []
        for input in input_list:
            new_input = input
            if to_2d:
                new_input = reshaping.to_2d(input)
            if keep_pd and isinstance(new_input, np.ndarray):
                # Keep as pandas object
                new_input = ArrayWrapper(input_index, input_columns, new_input.ndim).wrap(new_input)
            input_list_ready.append(new_input)

        # Prepare parameters
        # NOTE: input_shape instead of input_shape_ready since parameters should
        # broadcast by the same rules as inputs
        param_context = merge_dicts(
            broadcast_named_args,
            dict(
                input_shape=input_shape_ready,
                wrapper=wrapper_ready,
                **dict(zip(input_names, input_list_ready)),
                pre_sub_args=args,
                pre_sub_kwargs=kwargs,
            ),
            template_context,
        )
        param_list = prepare_params(
            param_list,
            param_names,
            param_settings,
            input_shape=input_shape,
            to_2d=to_2d,
            context=param_context,
        )
        single_value = list(map(lambda x: len(x) == 1, param_list))
        if len(param_list) > 1:
            if level_names is not None:
                # Check level names
                checks.assert_len_equal(param_list, level_names)
                # Columns should be free of the specified level names
                if input_columns is not None:
                    for level_name in level_names:
                        if level_name is not None:
                            checks.assert_level_not_exists(input_columns, level_name)
            if param_product:
                # Make Cartesian product out of all params
                param_list = create_param_product(param_list)
        if len(param_list) > 0:
            # Broadcast such that each array has the same length
            if per_column:
                # The number of parameters should match the number of columns before split
                param_list = broadcast_params(param_list, to_n=input_shape_2d[1])
            else:
                param_list = broadcast_params(param_list)
        if random_subset is not None:
            # Pick combinations randomly
            if per_column:
                raise ValueError("Cannot select random subset when per_column=True")
            random_indices = np.sort(np.random.permutation(np.arange(len(param_list[0])))[:random_subset])
            param_list = [[params[i] for i in random_indices] for params in param_list]
        n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
        use_run_unique = False
        param_list_unique = param_list
        if not per_column and run_unique:
            try:
                # Try to get all unique parameter combinations
                param_tuples = list(zip(*param_list))
                unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
                if len(unique_param_tuples) < len(param_tuples):
                    param_list_unique = list(map(list, zip(*unique_param_tuples)))
                    use_run_unique = True
            except:
                pass
        if checks.is_numba_func(custom_func):
            # Numba can't stand untyped lists
            param_list_ready = [to_typed_list(params) for params in param_list_unique]
        else:
            param_list_ready = param_list_unique
        n_unique_param_values = len(param_list_unique[0]) if len(param_list_unique) > 0 else 1

        # Prepare in-place outputs
        in_output_list_ready = []
        for i in range(len(in_output_list)):
            if input_shape_2d is None:
                raise ValueError("input_shape is required when using in-place outputs")
            if in_output_list[i] is not None:
                # This in-place output has been already broadcast with inputs
                in_output_wide = in_output_list[i]
                if isinstance(in_output_list[i], np.ndarray):
                    in_output_wide = np.require(in_output_wide, requirements="W")
                if not per_column:
                    # One per parameter combination
                    in_output_wide = reshaping.tile(in_output_wide, n_unique_param_values, axis=1)
            else:
                # This in-place output hasn't been provided, so create empty
                _in_output_settings = resolve_dict(in_output_settings[i])
                dtype = _in_output_settings.get("dtype", None)
                if per_column:
                    in_output_shape = input_shape_ready
                else:
                    in_output_shape = (input_shape_2d[0], input_shape_2d[1] * n_unique_param_values)
                in_output_wide = np.empty(in_output_shape, dtype=dtype)
            in_output_list[i] = in_output_wide
            # Split each in-place output into chunks, each of input shape, and append to a list
            in_outputs = []
            if per_column:
                in_outputs.append(in_output_wide)
            else:
                for p in range(n_unique_param_values):
                    if isinstance(in_output_wide, pd.DataFrame):
                        in_output = in_output_wide.iloc[:, p * input_shape_2d[1] : (p + 1) * input_shape_2d[1]]
                        if len(input_shape_ready) == 1:
                            in_output = in_output.iloc[:, 0]
                    else:
                        in_output = in_output_wide[:, p * input_shape_2d[1] : (p + 1) * input_shape_2d[1]]
                        if len(input_shape_ready) == 1:
                            in_output = in_output[:, 0]
                    if keep_pd and isinstance(in_output, np.ndarray):
                        in_output = ArrayWrapper(input_index, input_columns, in_output.ndim).wrap(in_output)
                    in_outputs.append(in_output)
            in_output_list_ready.append(in_outputs)
        if checks.is_numba_func(custom_func):
            # Numba can't stand untyped lists
            in_output_list_ready = [to_typed_list(in_outputs) for in_outputs in in_output_list_ready]

        def _use_raw(_raw):
            # Use raw results of previous run to build outputs
            _output_list, _param_map, _n_input_cols, _other_list = _raw
            idxs = np.array([_param_map.index(param_tuple) for param_tuple in zip(*param_list)])
            _output_list = [
                np.hstack([o[:, idx * _n_input_cols : (idx + 1) * _n_input_cols] for idx in idxs]) for o in _output_list
            ]
            return _output_list, _param_map, _n_input_cols, _other_list

        # Get raw results
        if use_raw is not None:
            # Use raw results of previous run to build outputs
            output_list, param_map, n_input_cols, other_list = _use_raw(use_raw)
        else:
            # Prepare other arguments
            func_args = args
            func_kwargs = dict(kwargs)
            if pass_input_shape:
                func_kwargs["input_shape"] = input_shape_ready
            if pass_wrapper:
                func_kwargs["wrapper"] = wrapper_ready
            if pass_per_column:
                func_kwargs["per_column"] = per_column

            # Substitute templates
            if has_templates(func_args) or has_templates(func_kwargs):
                template_context = merge_dicts(
                    broadcast_named_args,
                    dict(
                        input_shape=input_shape_ready,
                        wrapper=wrapper_ready,
                        **dict(zip(input_names, input_list_ready)),
                        **dict(zip(in_output_names, in_output_list_ready)),
                        **dict(zip(param_names, param_list_ready)),
                        pre_sub_args=func_args,
                        pre_sub_kwargs=func_kwargs,
                    ),
                    template_context,
                )
                func_args = substitute_templates(func_args, template_context, sub_id="custom_func_args")
                func_kwargs = substitute_templates(func_kwargs, template_context, sub_id="custom_func_kwargs")

            # Run the custom function
            if checks.is_numba_func(custom_func):
                func_args += tuple(func_kwargs.values())
                func_kwargs = {}
            if pass_packed:
                output = custom_func(
                    tuple(input_list_ready),
                    tuple(in_output_list_ready),
                    tuple(param_list_ready),
                    *func_args,
                    **func_kwargs,
                )
            else:
                output = custom_func(
                    *input_list_ready, *in_output_list_ready, *param_list_ready, *func_args, **func_kwargs
                )

            # Return cache
            if kwargs.get("return_cache", False):
                if use_run_unique and not silence_warnings:
                    warnings.warn(
                        "Cache is produced by unique parameter combinations when run_unique=True",
                        stacklevel=2,
                    )
                return output

            # Post-process results
            if output is None:
                output_list = []
                other_list = []
            else:
                if isinstance(output, (tuple, list, List)):
                    output_list = list(output)
                else:
                    output_list = [output]
                # Other outputs should be returned without post-processing (for example cache_dict)
                if len(output_list) > num_ret_outputs:
                    other_list = output_list[num_ret_outputs:]
                    if use_run_unique and not silence_warnings:
                        warnings.warn(
                            "Additional output objects are produced by unique parameter combinations when"
                            " run_unique=True",
                            stacklevel=2,
                        )
                else:
                    other_list = []
                # Process only the num_ret_outputs outputs
                output_list = output_list[:num_ret_outputs]
            if len(output_list) != num_ret_outputs:
                raise ValueError("Number of returned outputs other than expected")
            output_list = list(map(lambda x: reshaping.to_2d_array(x), output_list))

            # In-place outputs are treated as outputs from here
            output_list = in_output_list + output_list

            # Prepare raw
            param_map = list(zip(*param_list_unique))  # account for use_run_unique
            output_shape = output_list[0].shape
            for output in output_list:
                if output.shape != output_shape:
                    raise ValueError("All outputs must have the same shape")
            if per_column:
                n_input_cols = output_shape[1]
            else:
                n_input_cols = output_shape[1] // n_unique_param_values
            if input_shape_2d is not None:
                if n_input_cols != input_shape_2d[1]:
                    if per_column:
                        raise ValueError(
                            "All outputs must have the same number of columns as inputs when per_column=True"
                        )
                    else:
                        raise ValueError("All outputs must have the number of columns = #input columns x #parameters")
            raw = output_list, param_map, n_input_cols, other_list
            if return_raw:
                if use_run_unique and not silence_warnings:
                    warnings.warn(
                        "Raw output is produced by unique parameter combinations when run_unique=True",
                        stacklevel=2,
                    )
                return raw
            if use_run_unique:
                output_list, param_map, n_input_cols, other_list = _use_raw(raw)

        # Update shape and other meta if no inputs
        if input_shape is None:
            if n_input_cols == 1:
                input_shape = (output_list[0].shape[0],)
            else:
                input_shape = (output_list[0].shape[0], n_input_cols)
        if input_index is None:
            input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
        if input_columns is None:
            input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)

        # Build column hierarchy and create mappers
        if len(param_list) > 0:
            # Build new column levels on top of input levels
            param_indexes, new_columns = build_columns(
                param_list,
                input_columns,
                level_names=level_names,
                hide_levels=hide_levels,
                single_value=single_value,
                param_settings=param_settings,
                per_column=per_column,
                **build_col_kwargs,
            )
            # Build a mapper that maps old columns in inputs to new columns
            # Instead of tiling all inputs to the shape of outputs and wasting memory,
            # we just keep a mapper and perform the tiling when needed
            input_mapper = None
            if len(input_list) > 0:
                if per_column:
                    input_mapper = np.arange(len(input_columns))
                else:
                    input_mapper = np.tile(np.arange(len(input_columns)), n_param_values)
            # Build mappers to easily map between parameters and columns
            mapper_list = [param_indexes[i] for i in range(len(param_list))]
        else:
            # Some indicators don't have any params
            new_columns = input_columns
            input_mapper = None
            mapper_list = []

        # Return artifacts: no pandas objects, just a wrapper and NumPy arrays
        new_ndim = len(input_shape) if output_list[0].shape[1] == 1 else output_list[0].ndim
        wrapper = ArrayWrapper(input_index, new_columns, new_ndim, **wrapper_kwargs)

        return (
            wrapper,
            input_list,
            input_mapper,
            output_list[: len(in_output_list)],
            output_list[len(in_output_list) :],
            param_list,
            mapper_list,
            other_list,
        )

    @classmethod
    def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """Private run method."""
        raise NotImplementedError

    @classmethod
    def run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """Public run method."""
        return cls._run(*args, **kwargs)

    @classmethod
    def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """Private run combinations method."""
        raise NotImplementedError

    @classmethod
    def run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """Public run combinations method."""
        return cls._run_combs(*args, **kwargs)

    @classmethod
    def row_stack(
        cls: tp.Type[IndicatorBaseT],
        *objs: tp.MaybeTuple[IndicatorBaseT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> IndicatorBaseT:
        """Stack multiple `IndicatorBase` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        All objects to be merged must have the same columns x parameters."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, IndicatorBase):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs], stack_columns=False, **wrapper_kwargs
            )

        if "input_list" not in kwargs:
            new_input_list = []
            for input_name in cls.input_names:
                new_input_list.append(np.row_stack([getattr(obj, f"_{input_name}") for obj in objs]))
            kwargs["input_list"] = new_input_list
        if "in_output_list" not in kwargs:
            new_in_output_list = []
            for in_output_name in cls.in_output_names:
                new_in_output_list.append(np.row_stack([getattr(obj, f"_{in_output_name}") for obj in objs]))
            kwargs["in_output_list"] = new_in_output_list
        if "output_list" not in kwargs:
            new_output_list = []
            for output_name in cls.output_names:
                new_output_list.append(np.row_stack([getattr(obj, f"_{output_name}") for obj in objs]))
            kwargs["output_list"] = new_output_list

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[IndicatorBaseT],
        *objs: tp.MaybeTuple[IndicatorBaseT],
        wrapper_kwargs: tp.KwargsLike = None,
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> IndicatorBaseT:
        """Stack multiple `IndicatorBase` instances along columns x parameters.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        All objects to be merged must have the same index."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, IndicatorBase):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "input_mapper" not in kwargs:
            stack_input_mapper_objs = True
            for obj in objs:
                if getattr(obj, "_input_mapper", None) is None:
                    stack_input_mapper_objs = False
                    break
            if stack_input_mapper_objs:
                kwargs["input_mapper"] = np.concatenate([getattr(obj, "_input_mapper") for obj in objs])
        if "in_output_list" not in kwargs:
            new_in_output_list = []
            for in_output_name in cls.in_output_names:
                new_in_output_list.append(np.column_stack([getattr(obj, f"_{in_output_name}") for obj in objs]))
            kwargs["in_output_list"] = new_in_output_list
        if "output_list" not in kwargs:
            new_output_list = []
            for output_name in cls.output_names:
                new_output_list.append(np.column_stack([getattr(obj, f"_{output_name}") for obj in objs]))
            kwargs["output_list"] = new_output_list
        if "param_list" not in kwargs:
            new_param_list = []
            for param_name in cls.param_names:
                param_objs = []
                for obj in objs:
                    param_objs.extend(getattr(obj, f"_{param_name}_list"))
                new_param_list.append(param_objs)
            kwargs["param_list"] = new_param_list
        if "mapper_list" not in kwargs:
            new_mapper_list = []
            for param_name in cls.param_names:
                new_mapper = None
                for obj in objs:
                    obj_mapper = getattr(obj, f"_{param_name}_mapper")
                    if new_mapper is None:
                        new_mapper = obj_mapper
                    else:
                        new_mapper = new_mapper.append(obj_mapper)
                new_mapper_list.append(new_mapper)
            kwargs["mapper_list"] = new_mapper_list

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "input_list",
        "input_mapper",
        "in_output_list",
        "output_list",
        "param_list",
        "mapper_list",
        "short_name",
        "level_names",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        input_list: InputListT,
        input_mapper: InputMapperT,
        in_output_list: InOutputListT,
        output_list: OutputListT,
        param_list: ParamListT,
        mapper_list: MapperListT,
        short_name: str,
        level_names: tp.Tuple[str, ...],
        **kwargs,
    ) -> None:
        if input_mapper is not None:
            checks.assert_equal(input_mapper.shape[0], wrapper.shape_2d[1])
        for ts in input_list:
            checks.assert_equal(ts.shape[0], wrapper.shape_2d[0])
        for ts in in_output_list + output_list:
            checks.assert_equal(ts.shape, wrapper.shape_2d)
        for params in param_list:
            checks.assert_len_equal(param_list[0], params)
        for mapper in mapper_list:
            checks.assert_equal(len(mapper), wrapper.shape_2d[1])
        checks.assert_instance_of(short_name, str)
        checks.assert_len_equal(level_names, param_list)

        Analyzable.__init__(
            self,
            wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
            short_name=short_name,
            level_names=level_names,
            **kwargs,
        )

        setattr(self, "_short_name", short_name)
        setattr(self, "_level_names", level_names)

        for i, ts_name in enumerate(self.input_names):
            setattr(self, f"_{ts_name}", input_list[i])
        setattr(self, "_input_mapper", input_mapper)
        for i, in_output_name in enumerate(self.in_output_names):
            setattr(self, f"_{in_output_name}", in_output_list[i])
        for i, output_name in enumerate(self.output_names):
            setattr(self, f"_{output_name}", output_list[i])
        for i, param_name in enumerate(self.param_names):
            setattr(self, f"_{param_name}_list", param_list[i])
            setattr(self, f"_{param_name}_mapper", mapper_list[i])
        if len(self.param_names) > 1:
            tuple_mapper = list(zip(*list(mapper_list)))
            setattr(self, "_tuple_mapper", tuple_mapper)

    def indexing_func(self: IndicatorBaseT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> IndicatorBaseT:
        """Perform indexing on `IndicatorBase`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        row_idxs = wrapper_meta["row_idxs"]
        col_idxs = wrapper_meta["col_idxs"]
        rows_changed = wrapper_meta["rows_changed"]
        columns_changed = wrapper_meta["columns_changed"]
        if not isinstance(row_idxs, slice):
            row_idxs = reshaping.to_1d_array(row_idxs)
        if not isinstance(col_idxs, slice):
            col_idxs = reshaping.to_1d_array(col_idxs)

        input_mapper = getattr(self, "_input_mapper", None)
        if input_mapper is not None:
            input_mapper = input_mapper[col_idxs]
        input_list = []
        for input_name in self.input_names:
            new_input = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{input_name}"),
                row_idxs=row_idxs,
                rows_changed=rows_changed,
            )
            input_list.append(new_input)
        in_output_list = []
        for in_output_name in self.in_output_names:
            new_in_output = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{in_output_name}"),
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
            in_output_list.append(new_in_output)
        output_list = []
        for output_name in self.output_names:
            new_output = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{output_name}"),
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
            output_list.append(new_output)
        param_list = []
        for param_name in self.param_names:
            param_list.append(getattr(self, f"_{param_name}_list"))
        mapper_list = []
        for param_name in self.param_names:
            # Tuple mapper is a list because of its complex data type
            mapper_list.append(getattr(self, f"_{param_name}_mapper")[col_idxs])

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
        )

    @classproperty
    def short_name(cls_or_self) -> str:
        """Name of the indicator."""
        return cls_or_self._short_name

    @classproperty
    def input_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the input arrays."""
        return cls_or_self._input_names

    @classproperty
    def param_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the parameters."""
        return cls_or_self._param_names

    @classproperty
    def in_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the in-place output arrays."""
        return cls_or_self._in_output_names

    @classproperty
    def output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the regular output arrays."""
        return cls_or_self._output_names

    @classproperty
    def lazy_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the lazy output arrays."""
        return cls_or_self._lazy_output_names

    @classproperty
    def output_flags(cls_or_self) -> tp.Kwargs:
        """Dictionary of output flags."""
        return cls_or_self._output_flags

    @property
    def level_names(self) -> tp.Tuple[str, ...]:
        """Column level names corresponding to each parameter."""
        return self._level_names

    @classproperty
    def param_defaults(cls_or_self) -> tp.Dict[str, tp.Any]:
        """Parameter defaults extracted from the signature of `IndicatorBase.run`."""
        func_kwargs = get_func_kwargs(cls_or_self.run)
        out = {}
        for k, v in func_kwargs.items():
            if k in cls_or_self.param_names:
                if isinstance(v, Default):
                    out[k] = v.value
                else:
                    out[k] = v
        return out

    def unpack(self) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Return outputs, either one output or a tuple if there are multiple."""
        out = tuple([getattr(self, name) for name in self.output_names])
        if len(out) == 1:
            out = out[0]
        return out

    def to_dict(self, include_all: bool = True) -> tp.Dict[str, tp.SeriesFrame]:
        """Return outputs as a dict."""
        if include_all:
            output_names = self.output_names + self.in_output_names + self.lazy_output_names
        else:
            output_names = self.output_names
        return {name: getattr(self, name) for name in output_names}

    def to_frame(self, include_all: bool = True) -> tp.Frame:
        """Return outputs as a DataFrame."""
        out = self.to_dict(include_all=include_all)
        return pd.concat(list(out.values()), axis=1, keys=pd.Index(list(out.keys()), name="output"))


class IndicatorFactory(Configured):
    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "class_name",
        "class_docstring",
        "module_name",
        "short_name",
        "prepend_name",
        "input_names",
        "param_names",
        "in_output_names",
        "output_names",
        "output_flags",
        "lazy_outputs",
        "attr_settings",
        "metrics",
        "stats_defaults",
        "subplots",
        "plots_defaults",
    }

    def __init__(
        self,
        class_name: tp.Optional[str] = None,
        class_docstring: tp.Optional[str] = None,
        module_name: tp.Optional[str] = __name__,
        short_name: tp.Optional[str] = None,
        prepend_name: bool = True,
        input_names: tp.Optional[tp.Sequence[str]] = None,
        param_names: tp.Optional[tp.Sequence[str]] = None,
        in_output_names: tp.Optional[tp.Sequence[str]] = None,
        output_names: tp.Optional[tp.Sequence[str]] = None,
        output_flags: tp.KwargsLike = None,
        lazy_outputs: tp.KwargsLike = None,
        attr_settings: tp.KwargsLike = None,
        metrics: tp.Optional[tp.Kwargs] = None,
        stats_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
        subplots: tp.Optional[tp.Kwargs] = None,
        plots_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
        **kwargs,
    ) -> None:
        """A factory for creating new indicators.

        Initialize `IndicatorFactory` to create a skeleton and then use a class method
        such as `IndicatorFactory.with_custom_func` to bind a calculation function to the skeleton.

        Args:
            class_name (str): Name for the created indicator class.
            class_docstring (str): Docstring for the created indicator class.
            module_name (str): Name of the module the class originates from.
            short_name (str): Short name of the indicator.

                Defaults to lower-case `class_name`.
            prepend_name (bool): Whether to prepend `short_name` to each parameter level.
            input_names (list of str): List with input names.
            param_names (list of str): List with parameter names.
            in_output_names (list of str): List with in-output names.

                An in-place output is an output that is not returned but modified in-place.
                Some advantages of such outputs include:

                1) they don't need to be returned,
                2) they can be passed between functions as easily as inputs,
                3) they can be provided with already allocated data to safe memory,
                4) if data or default value are not provided, they are created empty to not occupy memory.
            output_names (list of str): List with output names.
            output_flags (dict): Dictionary of in-place and regular output flags.
            lazy_outputs (dict): Dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `property` if not already wrapped.
            attr_settings (dict): Dictionary with attribute settings.

                Attributes can be `input_names`, `in_output_names`, `output_names`, and `lazy_outputs`.

                Following keys are accepted:

                * `dtype`: Data type used to determine which methods to generate around this attribute.
                    Set to None to disable. Default is `np.float_`. Can be set to instance of
                    `collections.namedtuple` acting as enumerated type, or any other mapping;
                    It will then create a property with suffix `readable` that contains data in a string format.
                * `enum_unkval`: Value to be considered as unknown. Applies to enumerated data types only.
                * `make_cacheable`: Whether to make the property cacheable. Applies to inputs only.
            metrics (dict): Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                If dict, will be converted to `vectorbtpro.utils.config.Config`.
            stats_defaults (callable or dict): Defaults for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                If dict, will be converted into a property.
            subplots (dict): Subplots supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

                If dict, will be converted to `vectorbtpro.utils.config.Config`.
            plots_defaults (callable or dict): Defaults for `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

                If dict, will be converted into a property.
            **kwargs: Custom keyword arguments passed to the config.

        !!! note
            The `__init__` method is not used for running the indicator, for this use `run`.
            The reason for this is indexing, which requires a clean `__init__` method for creating
            a new indicator object with newly indexed attributes.
        """
        Configured.__init__(
            self,
            class_name=class_name,
            class_docstring=class_docstring,
            module_name=module_name,
            short_name=short_name,
            prepend_name=prepend_name,
            input_names=input_names,
            param_names=param_names,
            in_output_names=in_output_names,
            output_names=output_names,
            output_flags=output_flags,
            lazy_outputs=lazy_outputs,
            attr_settings=attr_settings,
            metrics=metrics,
            stats_defaults=stats_defaults,
            subplots=subplots,
            plots_defaults=plots_defaults,
            **kwargs,
        )

        # Check parameters
        if class_name is None:
            class_name = "Indicator"
        checks.assert_instance_of(class_name, str)
        if class_docstring is None:
            class_docstring = ""
        checks.assert_instance_of(class_docstring, str)
        if module_name is not None:
            checks.assert_instance_of(module_name, str)
        if short_name is None:
            if class_name == "Indicator":
                short_name = "custom"
            else:
                short_name = class_name.lower()
        checks.assert_instance_of(short_name, str)
        checks.assert_instance_of(prepend_name, bool)
        if input_names is None:
            input_names = []
        else:
            checks.assert_sequence(input_names)
            input_names = list(input_names)
        if param_names is None:
            param_names = []
        else:
            checks.assert_sequence(param_names)
            param_names = list(param_names)
        if in_output_names is None:
            in_output_names = []
        else:
            checks.assert_sequence(in_output_names)
            in_output_names = list(in_output_names)
        if output_names is None:
            output_names = []
        else:
            checks.assert_sequence(output_names)
            output_names = list(output_names)
        all_output_names = in_output_names + output_names
        if len(all_output_names) == 0:
            raise ValueError("Must have at least one in-place or regular output")
        if len(set.intersection(set(input_names), set(in_output_names), set(output_names))) > 0:
            raise ValueError("Inputs, in-outputs, and parameters must all have unique names")
        if output_flags is None:
            output_flags = {}
        checks.assert_instance_of(output_flags, dict)
        if len(output_flags) > 0:
            checks.assert_dict_valid(output_flags, all_output_names)
        if lazy_outputs is None:
            lazy_outputs = {}
        checks.assert_instance_of(lazy_outputs, dict)
        if attr_settings is None:
            attr_settings = {}
        checks.assert_instance_of(attr_settings, dict)
        all_attr_names = input_names + all_output_names + list(lazy_outputs.keys())
        if len(attr_settings) > 0:
            checks.assert_dict_valid(
                attr_settings,
                [
                    all_attr_names,
                    ["dtype", "enum_unkval", "make_cacheable"],
                ],
            )

        # Set up class
        ParamIndexer = build_param_indexer(
            param_names + (["tuple"] if len(param_names) > 1 else []),
            module_name=module_name,
        )
        Indicator = type(class_name, (IndicatorBase, ParamIndexer), {})
        Indicator.__doc__ = class_docstring
        if module_name is not None:
            Indicator.__module__ = module_name

        # Create read-only properties
        setattr(Indicator, "_short_name", short_name)
        setattr(Indicator, "_input_names", tuple(input_names))
        setattr(Indicator, "_param_names", tuple(param_names))
        setattr(Indicator, "_in_output_names", tuple(in_output_names))
        setattr(Indicator, "_output_names", tuple(output_names))
        setattr(Indicator, "_lazy_output_names", tuple(lazy_outputs.keys()))
        setattr(Indicator, "_output_flags", output_flags)

        for param_name in param_names:

            def param_list_prop(self, _param_name=param_name) -> tp.List[tp.Param]:
                return getattr(self, f"_{_param_name}_list")

            param_list_prop.__doc__ = f"List of `{param_name}` values."
            setattr(Indicator, f"{param_name}_list", property(param_list_prop))

        for input_name in input_names:
            _attr_settings = attr_settings.get(input_name, {})
            make_cacheable = _attr_settings.get("make_cacheable", False)

            def input_prop(self, _input_name: str = input_name) -> tp.SeriesFrame:
                """Input array."""
                old_input = reshaping.to_2d_array(getattr(self, "_" + _input_name))
                input_mapper = getattr(self, "_input_mapper")
                if input_mapper is None:
                    return self.wrapper.wrap(old_input)
                return self.wrapper.wrap(old_input[:, input_mapper])

            input_prop.__name__ = input_name
            if make_cacheable:
                setattr(Indicator, input_name, cacheable_property(input_prop))
            else:
                setattr(Indicator, input_name, property(input_prop))

        for output_name in all_output_names:

            def output_prop(self, _output_name: str = output_name) -> tp.SeriesFrame:
                return self.wrapper.wrap(getattr(self, "_" + _output_name))

            if output_name in in_output_names:
                output_prop.__doc__ = """In-place output array."""
            else:
                output_prop.__doc__ = """Output array."""

            output_prop.__name__ = output_name
            if output_name in output_flags:
                _output_flags = output_flags[output_name]
                if isinstance(_output_flags, (tuple, list)):
                    _output_flags = ", ".join(_output_flags)
                output_prop.__doc__ += "\n\n" + _output_flags
            setattr(Indicator, output_name, property(output_prop))

        # Add __init__ method
        def __init__(
            self,
            wrapper: ArrayWrapper,
            input_list: InputListT,
            input_mapper: InputMapperT,
            in_output_list: InOutputListT,
            output_list: OutputListT,
            param_list: ParamListT,
            mapper_list: MapperListT,
            short_name: str,
            level_names: tp.Tuple[str, ...],
        ) -> None:
            IndicatorBase.__init__(
                self,
                wrapper,
                input_list,
                input_mapper,
                in_output_list,
                output_list,
                param_list,
                mapper_list,
                short_name,
                level_names,
            )
            if len(param_names) > 1:
                tuple_mapper = list(zip(*list(mapper_list)))
            else:
                tuple_mapper = None

            # Initialize indexers
            mapper_sr_list = []
            for i, m in enumerate(mapper_list):
                mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
            if tuple_mapper is not None:
                mapper_sr_list.append(pd.Series(tuple_mapper, index=wrapper.columns))
            ParamIndexer.__init__(self, mapper_sr_list, level_names=[*level_names, level_names])

        setattr(Indicator, "__init__", __init__)

        # Add user-defined outputs
        for prop_name, prop in lazy_outputs.items():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""
            prop.__name__ = prop_name
            if not isinstance(prop, property):
                prop = property(prop)
            setattr(Indicator, prop_name, prop)

        # Add comparison & combination methods for all inputs, outputs, and user-defined properties
        def assign_combine_method(
            func_name: str,
            combine_func: tp.Callable,
            def_kwargs: tp.Kwargs,
            attr_name: str,
            docstring: str,
        ) -> None:
            def combine_method(
                self: IndicatorBaseT,
                other: tp.MaybeTupleList[tp.Union[IndicatorBaseT, tp.ArrayLike, BaseAccessor]],
                level_name: tp.Optional[str] = None,
                allow_multiple: bool = True,
                _prepend_name: bool = prepend_name,
                **kwargs,
            ) -> tp.SeriesFrame:
                if allow_multiple and isinstance(other, (tuple, list)):
                    other = list(other)
                    for i in range(len(other)):
                        if isinstance(other[i], IndicatorBase):
                            other[i] = getattr(other[i], attr_name)
                else:
                    if isinstance(other, IndicatorBase):
                        other = getattr(other, attr_name)
                if level_name is None:
                    if _prepend_name:
                        if attr_name == self.short_name:
                            level_name = f"{self.short_name}_{func_name}"
                        else:
                            level_name = f"{self.short_name}_{attr_name}_{func_name}"
                    else:
                        level_name = f"{attr_name}_{func_name}"
                out = combine_objs(
                    getattr(self, attr_name),
                    other,
                    combine_func,
                    level_name=level_name,
                    allow_multiple=allow_multiple,
                    **merge_dicts(def_kwargs, kwargs),
                )
                return out

            combine_method.__qualname__ = f"{Indicator.__name__}.{attr_name}_{func_name}"
            combine_method.__doc__ = docstring
            setattr(Indicator, f"{attr_name}_{func_name}", combine_method)

        for attr_name in all_attr_names:
            _attr_settings = attr_settings.get(attr_name, {})
            dtype = _attr_settings.get("dtype", np.float_)
            enum_unkval = _attr_settings.get("enum_unkval", -1)

            if checks.is_mapping_like(dtype):

                def attr_readable(
                    self,
                    _attr_name: str = attr_name,
                    _mapping: tp.MappingLike = dtype,
                    _enum_unkval: tp.Any = enum_unkval,
                ) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).apply_mapping(enum_unkval=_enum_unkval)

                attr_readable.__qualname__ = f"{Indicator.__name__}.{attr_name}_readable"
                attr_readable.__doc__ = inspect.cleandoc(
                    """`{attr_name}` in readable format based on the following mapping: 
                                
                    ```python
                    {dtype}
                    ```"""
                ).format(attr_name=attr_name, dtype=prettify(to_value_mapping(dtype, enum_unkval=enum_unkval)))
                setattr(Indicator, f"{attr_name}_readable", property(attr_readable))

                def attr_stats(
                    self,
                    *args,
                    _attr_name: str = attr_name,
                    _mapping: tp.MappingLike = dtype,
                    **kwargs,
                ) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).stats(*args, **kwargs)

                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_name}_stats"
                attr_stats.__doc__ = inspect.cleandoc(
                    """Stats of `{attr_name}` based on the following mapping: 

                    ```python
                    {dtype}
                    ```"""
                ).format(attr_name=attr_name, dtype=prettify(to_value_mapping(dtype)))
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

            elif np.issubdtype(dtype, np.number):
                func_info = [
                    ("above", np.greater, dict()),
                    ("below", np.less, dict()),
                    ("equal", np.equal, dict()),
                    (
                        "crossed_above",
                        lambda x, y, wait=0, dropna=False: jit_reg.resolve(generic_nb.crossed_above_nb)(
                            x,
                            y,
                            wait=wait,
                            dropna=dropna,
                        ),
                        dict(to_2d=True),
                    ),
                    (
                        "crossed_below",
                        lambda x, y, wait=0, dropna=False: jit_reg.resolve(generic_nb.crossed_above_nb)(
                            y,
                            x,
                            wait=wait,
                            dropna=dropna,
                        ),
                        dict(to_2d=True),
                    ),
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return True for each element where `{attr_name}` is {func_name} `other`. 
                
                    See `vectorbtpro.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.stats(*args, **kwargs)

                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_name}_stats"
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as generic."""
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

            elif np.issubdtype(dtype, np.bool_):
                func_info = [
                    ("and", np.logical_and, dict()),
                    ("or", np.logical_or, dict()),
                    ("xor", np.logical_xor, dict()),
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return `{attr_name} {func_name.upper()} other`. 

                    See `vectorbtpro.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.signals.stats(*args, **kwargs)

                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_name}_stats"
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as signals."""
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

        # Prepare stats
        if metrics is not None:
            if not isinstance(metrics, Config):
                metrics = Config(metrics, options_=dict(copy_kwargs=dict(copy_mode="deep")))
            setattr(Indicator, "_metrics", metrics.copy())

        if stats_defaults is not None:
            if isinstance(stats_defaults, dict):

                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return _stats_defaults

            else:

                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return stats_defaults(self)

            stats_defaults_prop.__name__ = "stats_defaults"
            setattr(Indicator, "stats_defaults", property(stats_defaults_prop))

        # Prepare plots
        if subplots is not None:
            if not isinstance(subplots, Config):
                subplots = Config(subplots, options_=dict(copy_kwargs=dict(copy_mode="deep")))
            setattr(Indicator, "_subplots", subplots.copy())

        if plots_defaults is not None:
            if isinstance(plots_defaults, dict):

                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return _plots_defaults

            else:

                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return plots_defaults(self)

            plots_defaults_prop.__name__ = "plots_defaults"
            setattr(Indicator, "plots_defaults", property(plots_defaults_prop))

        # Store arguments
        self._class_name = class_name
        self._class_docstring = class_docstring
        self._module_name = module_name
        self._short_name = short_name
        self._prepend_name = prepend_name
        self._input_names = input_names
        self._param_names = param_names
        self._in_output_names = in_output_names
        self._output_names = output_names
        self._output_flags = output_flags
        self._lazy_outputs = lazy_outputs
        self._attr_settings = attr_settings
        self._metrics = metrics
        self._stats_defaults = stats_defaults
        self._subplots = subplots
        self._plots_defaults = plots_defaults

        # Store indicator class
        self._Indicator = Indicator

    @property
    def class_name(self) -> str:
        """Name for the created indicator class."""
        return self._class_name

    @property
    def class_docstring(self) -> str:
        """Docstring for the created indicator class."""
        return self._class_docstring

    @property
    def module_name(self) -> str:
        """Name of the module the class originates from."""
        return self._module_name

    @property
    def short_name(self) -> str:
        """Short name of the indicator."""
        return self._short_name

    @property
    def prepend_name(self) -> bool:
        """Whether to prepend `IndicatorFactory.short_name` to each parameter level."""
        return self._prepend_name

    @property
    def input_names(self) -> tp.List[str]:
        """List with input names."""
        return self._input_names

    @property
    def param_names(self) -> tp.List[str]:
        """List with parameter names."""
        return self._param_names

    @property
    def in_output_names(self) -> tp.List[str]:
        """List with in-output names."""
        return self._in_output_names

    @property
    def output_names(self) -> tp.List[str]:
        """List with output names."""
        return self._output_names

    @property
    def output_flags(self) -> tp.Kwargs:
        """Dictionary of in-place and regular output flags."""
        return self._output_flags

    @property
    def lazy_outputs(self) -> tp.Kwargs:
        """Dictionary with user-defined functions that will become properties."""
        return self._lazy_outputs

    @property
    def attr_settings(self) -> tp.Kwargs:
        """Dictionary with attribute settings."""
        return self._attr_settings

    @property
    def metrics(self) -> Config:
        """Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`."""
        return self._metrics

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`."""
        return self._stats_defaults

    @property
    def subplots(self) -> Config:
        """Subplots supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."""
        return self._subplots

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."""
        return self._plots_defaults

    @property
    def Indicator(self) -> tp.Type[IndicatorBase]:
        """Built indicator class."""
        return self._Indicator

    def with_custom_func(
        self,
        custom_func: tp.Callable,
        require_input_shape: bool = False,
        param_settings: tp.KwargsLike = None,
        in_output_settings: tp.KwargsLike = None,
        hide_params: tp.Union[None, bool, tp.Sequence[str]] = None,
        hide_default: bool = True,
        var_args: bool = False,
        keyword_only_args: bool = False,
        **pipeline_kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build indicator class around a custom calculation function.

        In contrast to `IndicatorFactory.with_apply_func`, this method offers full flexibility.
        It's up to the user to handle caching and concatenate columns for each parameter (for example,
        by using `vectorbtpro.base.combining.apply_and_concat`). Also, you must ensure that
        each output array has an appropriate number of columns, which is the number of columns in
        input arrays multiplied by the number of parameter combinations.

        Args:
            custom_func (callable): A function that takes broadcast arrays corresponding
                to `input_names`, broadcast in-place output arrays corresponding to `in_output_names`,
                broadcast parameter arrays corresponding to `param_names`, and other arguments and
                keyword arguments, and returns outputs corresponding to `output_names` and other objects
                that are then returned with the indicator instance.

                Can be Numba-compiled.

                !!! note
                    Shape of each output must be the same and match the shape of each input stacked
                    n times (= the number of parameter values) along the column axis.
            require_input_shape (bool): Whether to input shape is required.
            param_settings (dict): A dictionary of parameter settings keyed by name.
                See `IndicatorBase.run_pipeline` for keys.

                Can be overwritten by any run method.
            in_output_settings (dict): A dictionary of in-place output settings keyed by name.
                See `IndicatorBase.run_pipeline` for keys.

                Can be overwritten by any run method.
            hide_params (bool or list of str): Parameter names to hide column levels for,
                or whether to hide all parameters.

                Can be overwritten by any run method.
            hide_default (bool): Whether to hide column levels of parameters with default value.

                Can be overwritten by any run method.
            var_args (bool): Whether run methods should accept variable arguments (`*args`).

                Set to True if `custom_func` accepts positional agruments that are not listed in the config.
            keyword_only_args (bool): Whether run methods should accept keyword-only arguments (`*`).

                Set to True to force the user to use keyword arguments (e.g., to avoid misplacing arguments).
            **pipeline_kwargs: Keyword arguments passed to `IndicatorBase.run_pipeline`.

                Can be overwritten by any run method.

                Can contain default values and also references to other arguments wrapped
                with `vectorbtpro.base.reshaping.Ref`.

        Returns:
            `Indicator`, and optionally other objects that are returned by `custom_func`
            and exceed `output_names`.

        Usage:
            * The following example produces the same indicator as the `IndicatorFactory.with_apply_func` example.

            ```pycon
            >>> @njit
            >>> def apply_func_nb(i, ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1[i] + arg1, ts2 * p2[i] + arg2

            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     return vbt.base.combining.apply_and_concat_multiple_nb(
            ...         len(p1), apply_func_nb, ts1, ts2, p1, p2, arg1, arg2)

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).with_custom_func(custom_func, var_args=True, arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.o1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.o2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            The difference between `apply_func_nb` here and in `IndicatorFactory.with_apply_func` is that
            here it takes the index of the current parameter combination that can be used for parameter selection.

            * You can also remove the entire `apply_func_nb` and define your logic in `custom_func`
            (which shouldn't necessarily be Numba-compiled):

            ```pycon
            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     input_shape = ts1.shape
            ...     n_params = len(p1)
            ...     out1 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=np.float_)
            ...     out2 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=np.float_)
            ...     for k in range(n_params):
            ...         for col in range(input_shape[1]):
            ...             for i in range(input_shape[0]):
            ...                 out1[i, input_shape[1] * k + col] = ts1[i, col] * p1[k] + arg1
            ...                 out2[i, input_shape[1] * k + col] = ts2[i, col] * p2[k] + arg2
            ...     return out1, out2
            ```
        """
        Indicator = self.Indicator

        short_name = self.short_name
        prepend_name = self.prepend_name
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names
        output_names = self.output_names

        all_input_names = input_names + param_names + in_output_names

        setattr(Indicator, "custom_func", custom_func)

        def _split_args(
            args: tp.Sequence,
        ) -> tp.Tuple[tp.Dict[str, tp.ArrayLike], tp.Dict[str, tp.ArrayLike], tp.Dict[str, tp.Params], tp.Args]:
            inputs = dict(zip(input_names, args[: len(input_names)]))
            checks.assert_len_equal(inputs, input_names)
            args = args[len(input_names) :]

            params = dict(zip(param_names, args[: len(param_names)]))
            checks.assert_len_equal(params, param_names)
            args = args[len(param_names) :]

            in_outputs = dict(zip(in_output_names, args[: len(in_output_names)]))
            checks.assert_len_equal(in_outputs, in_output_names)
            args = args[len(in_output_names) :]
            if not var_args and len(args) > 0:
                raise TypeError(
                    "Variable length arguments are not supported by this function (var_args is set to False)"
                )

            return inputs, in_outputs, params, args

        for k, v in pipeline_kwargs.items():
            if k in param_names and not isinstance(v, Default):
                pipeline_kwargs[k] = Default(v)  # track default params
        pipeline_kwargs = merge_dicts({k: None for k in in_output_names}, pipeline_kwargs)

        # Display default parameters and in-place outputs in the signature
        default_kwargs = {}
        for k in list(pipeline_kwargs.keys()):
            if k in input_names or k in param_names or k in in_output_names:
                default_kwargs[k] = pipeline_kwargs.pop(k)

        if var_args and keyword_only_args:
            raise ValueError("var_args and keyword_only_args cannot be used together")

        # Add private run method
        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=hide_params,
            hide_default=hide_default,
            **default_kwargs,
        )

        def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
            _short_name = kwargs.pop("short_name", def_run_kwargs["short_name"])
            _hide_params = kwargs.pop("hide_params", def_run_kwargs["hide_params"])
            _hide_default = kwargs.pop("hide_default", def_run_kwargs["hide_default"])
            _param_settings = merge_dicts(param_settings, kwargs.pop("param_settings", {}))
            _in_output_settings = merge_dicts(in_output_settings, kwargs.pop("in_output_settings", {}))

            if isinstance(_hide_params, bool):
                if not _hide_params:
                    _hide_params = None
                else:
                    _hide_params = param_names
            if _hide_params is None:
                _hide_params = []

            args = list(args)

            # Split arguments
            inputs, in_outputs, params, args = _split_args(args)

            # Prepare column levels
            level_names = []
            hide_levels = []
            for pname in param_names:
                level_name = _short_name + "_" + pname if prepend_name else pname
                level_names.append(level_name)
                if pname in _hide_params or (_hide_default and isinstance(params[pname], Default)):
                    hide_levels.append(level_name)
            for k, v in params.items():
                if isinstance(v, Default):
                    params[k] = v.value

            # Run the pipeline
            results = Indicator.run_pipeline(
                len(output_names),  # number of returned outputs
                custom_func,
                *args,
                require_input_shape=require_input_shape,
                inputs=inputs,
                in_outputs=in_outputs,
                params=params,
                level_names=level_names,
                hide_levels=hide_levels,
                param_settings=_param_settings,
                in_output_settings=_in_output_settings,
                **merge_dicts(pipeline_kwargs, kwargs),
            )

            # Return the raw result if any of the flags are set
            if kwargs.get("return_raw", False) or kwargs.get("return_cache", False):
                return results

            # Unpack the result
            (
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                other_list,
            ) = results

            # Create a new instance
            obj = cls(
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                short_name,
                tuple(level_names),
            )
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        setattr(Indicator, "_run", classmethod(_run))

        # Add public run method
        # Create function dynamically to provide user with a proper signature
        def compile_run_function(func_name: str, docstring: str, _default_kwargs: tp.KwargsLike = None) -> tp.Callable:
            pos_names = []
            main_kw_names = []
            other_kw_names = []
            if _default_kwargs is None:
                _default_kwargs = {}
            for k in input_names + param_names:
                if k in _default_kwargs:
                    main_kw_names.append(k)
                else:
                    pos_names.append(k)
            main_kw_names.extend(in_output_names)  # in_output_names are keyword-only
            for k, v in _default_kwargs.items():
                if k not in pos_names and k not in main_kw_names:
                    other_kw_names.append(k)

            _0 = func_name
            _1 = "*, " if keyword_only_args else ""
            _2 = []
            if require_input_shape:
                _2.append("input_shape")
            _2.extend(pos_names)
            _2 = ", ".join(_2) + ", " if len(_2) > 0 else ""
            _3 = "*args, " if var_args else ""
            _4 = ["{}={}".format(k, k) for k in main_kw_names + other_kw_names]
            if require_input_shape:
                _4 += ["input_index=None", "input_columns=None"]
            _4 = ", ".join(_4) + ", " if len(_4) > 0 else ""
            _5 = docstring
            _6 = all_input_names
            _6 = ", ".join(_6) + ", " if len(_6) > 0 else ""
            _7 = []
            if require_input_shape:
                _7.append("input_shape")
            _7.extend(other_kw_names)
            _7 = ["{}={}".format(k, k) for k in _7]
            if require_input_shape:
                _7 += ["input_index=input_index", "input_columns=input_columns"]
            _7 = ", ".join(_7) + ", " if len(_7) > 0 else ""
            func_str = (
                "@classmethod\n"
                "def {0}(cls, {1}{2}{3}{4}**kwargs):\n"
                '    """{5}"""\n'
                "    return cls._{0}({6}{3}{7}**kwargs)".format(_0, _1, _2, _3, _4, _5, _6, _7)
            )
            scope = {**dict(Default=Default), **_default_kwargs}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, "single")
            exec(code, scope)
            return scope[func_name]

        _0 = self.class_name
        _1 = ""
        if len(self.input_names) > 0:
            _1 += "\n* Inputs: " + ", ".join(map(lambda x: f"`{x}`", self.input_names))
        if len(self.in_output_names) > 0:
            _1 += "\n* In-place outputs: " + ", ".join(map(lambda x: f"`{x}`", self.in_output_names))
        if len(self.param_names) > 0:
            _1 += "\n* Parameters: " + ", ".join(map(lambda x: f"`{x}`", self.param_names))
        if len(self.output_names) > 0:
            _1 += "\n* Outputs: " + ", ".join(map(lambda x: f"`{x}`", self.output_names))
        if len(self.lazy_outputs) > 0:
            _1 += "\n* Lazy outputs: " + ", ".join(map(lambda x: f"`{x}`", list(self.lazy_outputs.keys())))
        run_docstring = """Run `{0}` indicator.
{1}

Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
Set `hide_default` to False to show the column levels of the parameters with a default value.

Other keyword arguments are passed to `{0}.run_pipeline`.""".format(
            _0,
            _1,
        )
        run = compile_run_function("run", run_docstring, def_run_kwargs)
        run.__qualname__ = f"{Indicator.__name__}.run"
        setattr(Indicator, "run", run)

        if len(param_names) > 0:
            # Add private run_combs method
            def_run_combs_kwargs = dict(
                r=2,
                param_product=False,
                comb_func=itertools.combinations,
                run_unique=True,
                short_names=None,
                hide_params=hide_params,
                hide_default=hide_default,
                **default_kwargs,
            )

            def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
                _r = kwargs.pop("r", def_run_combs_kwargs["r"])
                _param_product = kwargs.pop("param_product", def_run_combs_kwargs["param_product"])
                _comb_func = kwargs.pop("comb_func", def_run_combs_kwargs["comb_func"])
                _run_unique = kwargs.pop("run_unique", def_run_combs_kwargs["run_unique"])
                _short_names = kwargs.pop("short_names", def_run_combs_kwargs["short_names"])
                _hide_params = kwargs.pop("hide_params", def_run_kwargs["hide_params"])
                _hide_default = kwargs.pop("hide_default", def_run_kwargs["hide_default"])
                _param_settings = merge_dicts(param_settings, kwargs.get("param_settings", {}))

                if isinstance(_hide_params, bool):
                    if not _hide_params:
                        _hide_params = None
                    else:
                        _hide_params = param_names
                if _hide_params is None:
                    _hide_params = []
                if _short_names is None:
                    _short_names = [f"{short_name}_{str(i + 1)}" for i in range(_r)]

                args = list(args)

                # Split arguments
                inputs, in_outputs, params, args = _split_args(args)

                # Hide params
                for pname in param_names:
                    if _hide_default and isinstance(params[pname], Default):
                        params[pname] = params[pname].value
                        if pname not in _hide_params:
                            _hide_params.append(pname)
                checks.assert_len_equal(params, param_names)

                # Bring argument to list format
                input_list = list(inputs.values())
                in_output_list = list(in_outputs.values())
                param_list = list(params.values())

                # Prepare params
                for i, pname in enumerate(param_names):
                    is_tuple = _param_settings.get(pname, {}).get("is_tuple", False)
                    is_array_like = _param_settings.get(pname, {}).get("is_array_like", False)
                    param_list[i] = params_to_list(params[pname], is_tuple, is_array_like)
                if _param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)

                # Speed up by pre-calculating raw outputs
                if _run_unique:
                    raw_results = cls._run(
                        *input_list,
                        *param_list,
                        *in_output_list,
                        *args,
                        return_raw=True,
                        run_unique=False,
                        **kwargs,
                    )
                    kwargs["use_raw"] = raw_results  # use them next time

                # Generate indicator instances
                instances = []
                if _comb_func == itertools.product:
                    param_lists = zip(*_comb_func(zip(*param_list), repeat=_r))
                else:
                    param_lists = zip(*_comb_func(zip(*param_list), _r))
                for i, param_list in enumerate(param_lists):
                    instances.append(
                        cls._run(
                            *input_list,
                            *zip(*param_list),
                            *in_output_list,
                            *args,
                            short_name=_short_names[i],
                            hide_params=_hide_params,
                            hide_default=_hide_default,
                            run_unique=False,
                            **kwargs,
                        )
                    )
                return tuple(instances)

            setattr(Indicator, "_run_combs", classmethod(_run_combs))

            # Add public run_combs method
            _0 = self.class_name
            _1 = ""
            if len(self.input_names) > 0:
                _1 += "\n* Inputs: " + ", ".join(map(lambda x: f"`{x}`", self.input_names))
            if len(self.in_output_names) > 0:
                _1 += "\n* In-place outputs: " + ", ".join(map(lambda x: f"`{x}`", self.in_output_names))
            if len(self.param_names) > 0:
                _1 += "\n* Parameters: " + ", ".join(map(lambda x: f"`{x}`", self.param_names))
            if len(self.output_names) > 0:
                _1 += "\n* Outputs: " + ", ".join(map(lambda x: f"`{x}`", self.output_names))
            if len(self.lazy_outputs) > 0:
                _1 += "\n* Lazy outputs: " + ", ".join(map(lambda x: f"`{x}`", list(self.lazy_outputs.keys())))
            run_combs_docstring = """Create a combination of multiple `{0}` indicators using function `comb_func`.
{1}

`comb_func` must accept an iterable of parameter tuples and `r`. 
Also accepts all combinatoric iterators from itertools such as `itertools.combinations`.
Pass `r` to specify how many indicators to run. 
Pass `short_names` to specify the short name for each indicator. 
Set `run_unique` to True to first compute raw outputs for all parameters, 
and then use them to build each indicator (faster).

Other keyword arguments are passed to `{0}.run`.

!!! note
    This method should only be used when multiple indicators are needed. 
    To test multiple parameters, pass them as lists to `{0}.run`.
""".format(
                _0,
                _1,
            )
            run_combs = compile_run_function("run_combs", run_combs_docstring, def_run_combs_kwargs)
            run_combs.__qualname__ = f"{Indicator.__name__}.run_combs"
            setattr(Indicator, "run_combs", run_combs)

        return Indicator

    def with_apply_func(
        self,
        apply_func: tp.Callable,
        cache_func: tp.Optional[tp.Callable] = None,
        takes_1d: bool = False,
        select_params: bool = True,
        pass_packed: bool = False,
        cache_pass_packed: tp.Optional[bool] = None,
        pass_per_column: bool = False,
        cache_pass_per_column: tp.Optional[bool] = None,
        kwargs_as_args: tp.Optional[tp.Iterable[str]] = None,
        jit_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[str, tp.Type[IndicatorBase]]:
        """Build indicator class around a custom apply function.

        In contrast to `IndicatorFactory.with_custom_func`, this method handles a lot of things for you,
        such as caching, parameter selection, and concatenation. Your part is writing a function `apply_func`
        that accepts a selection of parameters (single values as opposed to multiple values in
        `IndicatorFactory.with_custom_func`) and does the calculation. It then automatically concatenates
        the resulting arrays into a single array per output.

        While this approach is simpler, it's also less flexible, since we can only work with
        one parameter selection at a time and can't view all parameters.

        The execution and concatenation is performed using `vectorbtpro.base.combining.apply_and_concat`.

        !!! note
            If `apply_func` is a Numba-compiled function:

            * All inputs are automatically converted to NumPy arrays
            * Each argument in `*args` must be of a Numba-compatible type
            * You cannot pass keyword arguments
            * Your outputs must be arrays of the same shape, data type and data order

        !!! note
            Reserved arguments such as  `per_column` (in this order) get passed as positional
            arguments if `jitted_loop` is True, otherwise as keyword arguments.

        Args:
            apply_func (callable): A function that takes inputs, selection of parameters, and
                other arguments, and does calculations to produce outputs.

                Arguments are passed to `apply_func` in the following order:

                * `i` (index of the parameter combination) if `select_params` is set to False
                * `input_shape` if `pass_input_shape` is set to True and `input_shape` not in `kwargs_as_args`
                * Input arrays corresponding to `input_names`. Passed as a tuple if `pass_packed`, otherwise unpacked.
                    If `select_params` is True, each argument is a list composed of multiple arrays -
                    one per parameter combination. When `per_column` is True, each of those arrays
                    corresponds to a column. Otherwise, they all refer to the same array. If `takes_1d`,
                    each array gets additionally split into multiple column arrays. Still passed
                    as a single array to the caching function.
                * In-output arrays corresponding to `in_output_names`. Passed as a tuple if `pass_packed`, otherwise unpacked.
                    If `select_params` is True, each argument is a list composed of multiple arrays -
                    one per parameter combination. When `per_column` is True, each of those arrays
                    corresponds to a column. If `takes_1d`, each array gets additionally split into
                    multiple column arrays. Still passed as a single array to the caching function.
                * Parameters corresponding to `param_names`. Passed as a tuple if `pass_packed`, otherwise unpacked.
                    If `select_params` is True, each argument is a list composed of multiple values -
                    one per parameter combination.  When `per_column` is True, each of those values
                    corresponds to a column. If `takes_1d`, each value gets additionally repeated by
                    the number of columns in the input arrays.
                * Variable arguments if `var_args` is set to True
                * `per_column` if `pass_per_column` is set to True and `per_column` not in
                    `kwargs_as_args` and `jitted_loop` is set to True
                * Arguments listed in `kwargs_as_args` passed as positional. Can include `takes_1d` and `per_column`.
                * Other keyword arguments if `jitted_loop` is False. Also includes `takes_1d` and `per_column`
                    if they must be passed and not in `kwargs_as_args`.

                Can be Numba-compiled (but doesn't have to).

                !!! note
                    Shape of each output must be the same and match the shape of each input.
            cache_func (callable): A caching function to preprocess data beforehand.

                Takes the same arguments as `apply_func`. Must return a single object or a tuple of objects.
                All returned objects will be passed unpacked as last arguments to `apply_func`.

                Can be Numba-compiled (but doesn't have to).
            takes_1d (bool): Whether to split 2-dim arrays into multiple 1-dim arrays along the column axis.

                Gets applied on inputs and in-outputs, while parameters get repeated by the number of columns.
            select_params (bool): Whether to automatically select in-outputs and parameters.

                If False, prepends the current iteration index to the arguments.
            pass_packed (bool): Whether to pass packed tuples for inputs, in-place outputs, and parameters.
            cache_pass_packed (bool): Overrides `pass_packed` for the caching function.
            pass_per_column (bool): Whether to pass `per_column`.
            cache_pass_per_column (bool): Overrides `pass_per_column` for the caching function.
            kwargs_as_args (iterable of str): Keyword arguments from `kwargs` dict to pass as
                positional arguments to the apply function.

                Should be used together with `jitted_loop` set to True since Numba doesn't support
                variable keyword arguments.

                Defaults to []. Order matters.
            jit_kwargs (dict): Keyword arguments passed to `@njit` decorator of the parameter selection function.

                By default, has `nogil` set to True.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_custom_func`, all the way down
                to `vectorbtpro.base.combining.apply_and_concat`.

        Returns:
            Indicator

        Usage:
            * The following example produces the same indicator as the `IndicatorFactory.with_custom_func` example.

            ```pycon
            >>> @njit
            ... def apply_func_nb(ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1 + arg1, ts2 * p2 + arg2

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['out1', 'out2']
            ... ).with_apply_func(
            ...     apply_func_nb, var_args=True,
            ...     kwargs_as_args=['arg2'], arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.out1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.out2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            * To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(ts, p):
            ...     time.sleep(1)
            ...     return ts * p

            >>> MyInd = vbt.IF(
            ...     input_names=['ts'],
            ...     param_names=['p'],
            ...     output_names=['out']
            ... ).with_apply_func(apply_func)

            >>> %timeit MyInd.run(price, [1, 2, 3])
            3.02 s ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit MyInd.run(price, [1, 2, 3], execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        Indicator = self.Indicator

        setattr(Indicator, "apply_func", apply_func)
        setattr(Indicator, "cache_func", cache_func)

        module_name = self.module_name
        input_names = self.input_names
        output_names = self.output_names
        in_output_names = self.in_output_names
        param_names = self.param_names

        num_ret_outputs = len(output_names)

        if kwargs_as_args is None:
            kwargs_as_args = []

        if checks.is_numba_func(apply_func):
            # Build a function that selects a parameter tuple
            # Do it here to avoid compilation with Numba every time custom_func is run
            _0 = "i"
            _0 += ", args_before"
            if len(input_names) > 0:
                _0 += ", " + ", ".join(input_names)
            if len(in_output_names) > 0:
                _0 += ", " + ", ".join(in_output_names)
            if len(param_names) > 0:
                _0 += ", " + ", ".join(param_names)
            _0 += ", *args"
            if select_params:
                _1 = "*args_before"
            else:
                _1 = "i, *args_before"
            if pass_packed:
                if len(input_names) > 0:
                    _1 += ", (" + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), input_names)) + ",)"
                else:
                    _1 += ", ()"
                if len(in_output_names) > 0:
                    _1 += ", (" + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), in_output_names)) + ",)"
                else:
                    _1 += ", ()"
                if len(param_names) > 0:
                    _1 += ", (" + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), param_names)) + ",)"
                else:
                    _1 += ", ()"
            else:
                if len(input_names) > 0:
                    _1 += ", " + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), input_names))
                if len(in_output_names) > 0:
                    _1 += ", " + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), in_output_names))
                if len(param_names) > 0:
                    _1 += ", " + ", ".join(map(lambda x: x + ("[i]" if select_params else ""), param_names))
            _1 += ", *args"
            func_str = "def param_select_func_nb({0}):\n   return apply_func({1})".format(_0, _1)
            scope = {"apply_func": apply_func}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, "single")
            exec(code, scope)
            param_select_func_nb = scope["param_select_func_nb"]
            if module_name is not None:
                param_select_func_nb.__module__ = module_name
            jit_kwargs = merge_dicts(dict(nogil=True), jit_kwargs)
            param_select_func_nb = njit(param_select_func_nb, **jit_kwargs)

            setattr(Indicator, "param_select_func_nb", param_select_func_nb)

        def custom_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.List[tp.AnyArray], ...],
            param_tuple: tp.Tuple[tp.List[tp.Param], ...],
            *_args,
            input_shape: tp.Optional[tp.Shape] = None,
            per_column: tp.Optional[bool] = None,
            return_cache: bool = False,
            use_cache: tp.Union[bool, CacheOutputT] = True,
            jitted_loop: bool = False,
            jitted_warmup: bool = False,
            execute_kwargs: tp.KwargsLike = None,
            **_kwargs,
        ) -> tp.Union[None, CacheOutputT, tp.Array2d, tp.List[tp.Array2d]]:
            """Custom function that forwards inputs and parameters to `apply_func`."""
            if jitted_loop and not checks.is_numba_func(apply_func):
                raise ValueError("Apply function must be Numba-compiled for jitted_loop=True")

            _cache_pass_packed = cache_pass_packed
            _cache_pass_per_column = cache_pass_per_column

            # Prepend positional arguments
            args_before = ()
            if input_shape is not None and "input_shape" not in kwargs_as_args:
                if per_column:
                    args_before += (input_shape[0],)
                else:
                    args_before += (input_shape,)

            # Append positional arguments
            more_args = ()
            for key in kwargs_as_args:
                if key == "per_column":
                    value = per_column
                elif key == "takes_1d":
                    value = per_column
                else:
                    value = _kwargs.pop(key)  # important: remove from kwargs
                more_args += (value,)

            # Resolve the number of parameters
            if len(input_tuple) > 0:
                if input_tuple[0].ndim == 1:
                    n_cols = 1
                else:
                    n_cols = input_tuple[0].shape[1]
            elif input_shape is not None:
                if len(input_shape) == 1:
                    n_cols = 1
                else:
                    n_cols = input_shape[1]
            else:
                n_cols = None
            if per_column:
                n_params = n_cols
            else:
                n_params = len(param_tuple[0]) if len(param_tuple) > 0 else 1

            # Caching
            cache = use_cache
            if isinstance(cache, bool):
                if cache and cache_func is not None:
                    _input_tuple = input_tuple
                    _in_output_tuple = ()
                    for in_outputs in in_output_tuple:
                        if checks.is_numba_func(cache_func):
                            _in_outputs = to_typed_list(in_outputs)
                        else:
                            _in_outputs = in_outputs
                        _in_output_tuple += (_in_outputs,)
                    _param_tuple = ()
                    for params in param_tuple:
                        if checks.is_numba_func(cache_func):
                            _params = to_typed_list(params)
                        else:
                            _params = params
                        _param_tuple += (_params,)

                    if _cache_pass_packed is None:
                        _cache_pass_packed = pass_packed
                    if _cache_pass_per_column is None and per_column:
                        _cache_pass_per_column = True
                    if _cache_pass_per_column is None:
                        _cache_pass_per_column = pass_per_column
                    cache_more_args = tuple(more_args)
                    cache_kwargs = dict(_kwargs)
                    if _cache_pass_per_column:
                        if "per_column" not in kwargs_as_args:
                            if jitted_loop:
                                cache_more_args += (per_column,)
                            else:
                                cache_kwargs["per_column"] = per_column

                    if _cache_pass_packed:
                        cache = cache_func(
                            *args_before,
                            _input_tuple,
                            _in_output_tuple,
                            _param_tuple,
                            *_args,
                            *cache_more_args,
                            **cache_kwargs,
                        )
                    else:
                        cache = cache_func(
                            *args_before,
                            *_input_tuple,
                            *_in_output_tuple,
                            *_param_tuple,
                            *_args,
                            *cache_more_args,
                            **cache_kwargs,
                        )
                else:
                    cache = None
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            # Prepare inputs
            def _expand_input(input: tp.MaybeList[tp.AnyArray], multiple: bool = False) -> tp.List[tp.AnyArray]:
                if jitted_loop:
                    _inputs = List()
                else:
                    _inputs = []
                if per_column:
                    if multiple:
                        _input = input[0]
                    else:
                        _input = input
                    if _input.ndim == 2:
                        for i in range(_input.shape[1]):
                            if takes_1d:
                                if isinstance(_input, pd.DataFrame):
                                    _inputs.append(_input.iloc[:, i])
                                else:
                                    _inputs.append(_input[:, i])
                            else:
                                if isinstance(_input, pd.DataFrame):
                                    _inputs.append(_input.iloc[:, i : i + 1])
                                else:
                                    _inputs.append(_input[:, i : i + 1])
                    else:
                        _inputs.append(_input)
                else:
                    for p in range(n_params):
                        if multiple:
                            _input = input[p]
                        else:
                            _input = input
                        if takes_1d:
                            if isinstance(_input, pd.DataFrame):
                                for i in range(_input.shape[1]):
                                    _inputs.append(_input.iloc[:, i])
                            elif _input.ndim == 2:
                                for i in range(_input.shape[1]):
                                    _inputs.append(_input[:, i])
                            else:
                                _inputs.append(_input)
                        else:
                            _inputs.append(_input)
                return _inputs

            _input_tuple = ()
            for input in input_tuple:
                _inputs = _expand_input(input)
                _input_tuple += (_inputs,)
            _in_output_tuple = ()
            for in_outputs in in_output_tuple:
                _in_outputs = _expand_input(in_outputs, multiple=True)
                _in_output_tuple += (_in_outputs,)
            _param_tuple = ()
            for params in param_tuple:
                if takes_1d and not per_column:
                    _params = [params[p] for p in range(len(params)) for i in range(n_cols)]
                else:
                    _params = params
                if jitted_loop:
                    if len(_params) > 0 and np.isscalar(_params[0]):
                        _params = np.asarray(_params)
                    else:
                        _params = to_typed_list(_params)
                _param_tuple += (_params,)
            if takes_1d and not per_column:
                _n_params = n_params * n_cols
            else:
                _n_params = n_params

            if pass_per_column:
                if "per_column" not in kwargs_as_args:
                    if jitted_loop:
                        more_args += (per_column,)
                    else:
                        _kwargs["per_column"] = per_column

            # Apply function and concatenate outputs
            if jitted_loop:
                return combining.apply_and_concat(
                    _n_params,
                    param_select_func_nb,
                    args_before,
                    *_input_tuple,
                    *_in_output_tuple,
                    *_param_tuple,
                    *_args,
                    *more_args,
                    *cache,
                    **_kwargs,
                    n_outputs=num_ret_outputs,
                    jitted_loop=True,
                    jitted_warmup=jitted_warmup,
                    execute_kwargs=execute_kwargs,
                )

            funcs_args = []
            for i in range(_n_params):
                if select_params:
                    _inputs = tuple(_inputs[i] for _inputs in _input_tuple)
                    _in_outputs = tuple(_in_outputs[i] for _in_outputs in _in_output_tuple)
                    _params = tuple(_params[i] for _params in _param_tuple)
                else:
                    _inputs = _input_tuple
                    _in_outputs = _in_output_tuple
                    _params = _param_tuple
                funcs_args.append(
                    (
                        apply_func,
                        (
                            *((i,) if not select_params else ()),
                            *args_before,
                            *((_inputs,) if pass_packed else _inputs),
                            *((_in_outputs,) if pass_packed else _in_outputs),
                            *((_params,) if pass_packed else _params),
                            *_args,
                            *more_args,
                            *cache,
                        ),
                        _kwargs,
                    )
                )
            return combining.apply_and_concat_each(
                funcs_args,
                n_outputs=num_ret_outputs,
                execute_kwargs=execute_kwargs,
            )

        return self.with_custom_func(custom_func, pass_packed=True, **kwargs)

    @classmethod
    def list_vbt_indicators(cls) -> tp.List[str]:
        """List all vectorbt indicators."""
        import vectorbtpro as vbt

        return sorted(
            [
                attr
                for attr in dir(vbt)
                if not attr.startswith("_")
                   and isinstance(getattr(vbt, attr), type)
                   and issubclass(getattr(vbt, attr), IndicatorBase)
            ]
        )

    @classmethod
    def list_locations(cls) -> tp.List[str]:
        """List supported locations."""
        return [
            "vbt",
            "talib",
            "pandas_ta",
            "ta",
            "technical",
            "techcon",
            "wqa101",
        ]

    @classmethod
    def list_indicators(
        cls,
        pattern: tp.Optional[str] = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        location: tp.Optional[str] = None,
        prepend_location: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """List indicators, optionally matching a pattern.

        Pattern can also be a location, in such a case all indicators from that location will be returned.
        For supported locations, see `IndicatorFactory.list_locations`."""
        if pattern is not None:
            if not case_sensitive:
                pattern = pattern.lower()
            if location is None and pattern.lower() in cls.list_locations():
                location = pattern
                pattern = None
        if prepend_location is None:
            if location is not None:
                prepend_location = False
            else:
                prepend_location = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if location is not None:
                location = location.lower()
                all_indicators = getattr(cls, f"list_{location.lower()}_indicators")()
            else:
                all_indicators = [
                    *map(lambda x: "vbt:" + x if prepend_location else x, cls.list_vbt_indicators()),
                    *map(lambda x: "talib:" + x if prepend_location else x, cls.list_talib_indicators()),
                    *map(lambda x: "pandas_ta:" + x if prepend_location else x, cls.list_pandas_ta_indicators()),
                    *map(lambda x: "ta:" + x if prepend_location else x, cls.list_ta_indicators()),
                    *map(lambda x: "technical:" + x if prepend_location else x, cls.list_technical_indicators()),
                    *map(lambda x: "techcon:" + x if prepend_location else x, cls.list_techcon_indicators()),
                    *map(lambda x: "wqa101:" + str(x) if prepend_location else str(x), range(1, 102)),
                ]
        found_indicators = []
        for indicator in all_indicators:
            if prepend_location and location is not None:
                indicator = location + ":" + indicator
            if case_sensitive:
                indicator_name = indicator
            else:
                indicator_name = indicator.lower()
            if pattern is not None:
                if use_regex:
                    if location is not None:
                        if not re.match(pattern, indicator_name):
                            continue
                    else:
                        if not re.match(pattern, indicator_name.split(":")[1]):
                            continue
                else:
                    if location is not None:
                        if not re.match(glob2re(pattern), indicator_name):
                            continue
                    else:
                        if not re.match(glob2re(pattern), indicator_name.split(":")[1]):
                            continue
            found_indicators.append(indicator)
        return found_indicators

    @classmethod
    def get_indicator(cls, name: str) -> tp.Type[IndicatorBase]:
        """Get the indicator class by its name.

        The name can contain a location suffix followed by a colon. For example, "talib:sma"
        or "talib_sma" will return the TA-Lib's SMA. Without a location, the indicator will be
        searched throughout all indicators, including the vectorbt's ones."""
        locations = cls.list_locations()

        if ":" in name:
            location = name.split(":")[0].lower().strip()
            name = name.split(":")[1].upper().strip()
        else:
            location = None
            name = name.lower().strip()
            found_location = False
            if "_" in name:
                for location in locations:
                    if name.startswith(location + "_"):
                        found_location = True
                        break
            if found_location:
                name = name[len(location) + 1:].upper()
            else:
                location = None
                name = name.upper()

        if location is not None:
            if location == "vbt":
                import vectorbtpro as vbt

                return getattr(vbt, name.upper())
            if location == "talib":
                return cls.from_talib(name)
            if location == "pandas_ta":
                return cls.from_pandas_ta(name)
            if location == "ta":
                return cls.from_ta(name)
            if location == "technical":
                return cls.from_technical(name)
            if location == "techcon":
                return cls.from_techcon(name)
            if location == "wqa101":
                return cls.from_wqa101(int(name))
            raise ValueError(f"Location '{location}' not found")
        else:
            import vectorbtpro as vbt
            from vectorbtpro.utils.module_ import check_installed

            if hasattr(vbt, name):
                return getattr(vbt, name)
            if str(name).isnumeric():
                return cls.from_wqa101(int(name))
            if check_installed("technical") and name in IndicatorFactory.list_techcon_indicators():
                return cls.from_techcon(name)
            if check_installed("talib") and name in IndicatorFactory.list_talib_indicators():
                return cls.from_talib(name)
            if check_installed("ta") and name in IndicatorFactory.list_ta_indicators(uppercase=True):
                return cls.from_ta(name)
            if check_installed("pandas_ta") and name in IndicatorFactory.list_pandas_ta_indicators():
                return cls.from_pandas_ta(name)
            if check_installed("technical") and name in IndicatorFactory.list_technical_indicators():
                return cls.from_technical(name)
        raise ValueError(f"Indicator '{name}' not found")

    # ############# Third party ############# #

    @classmethod
    def list_talib_indicators(cls) -> tp.List[str]:
        """List all parseable indicators in `talib`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("talib")
        import talib

        return sorted(talib.get_functions())

    @classmethod
    def from_talib(cls, func_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a `talib` function.

        Requires [TA-Lib](https://github.com/mrjbq7/ta-lib) installed.

        For input, parameter and output names, see [docs](https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md).

        Args:
            func_name (str): Function name.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMA = vbt.IF.from_talib('SMA')

            >>> sma = SMA.run(price, timeperiod=[2, 3])
            >>> sma.real
            sma_timeperiod         2         3
                              a    b    a    b
            2020-01-01      NaN  NaN  NaN  NaN
            2020-01-02      1.5  4.5  NaN  NaN
            2020-01-03      2.5  3.5  2.0  4.0
            2020-01-04      3.5  2.5  3.0  3.0
            2020-01-05      4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use `vectorbtpro.utils.formatting.phelp`:

            ```pycon
            >>> vbt.phelp(SMA.run)
            SMA.run(
                close,
                timeperiod=Default(value=30),
                timeframe=Default(value=None),
                short_name='sma',
                hide_params=None,
                hide_default=True,
                **kwargs
            ):
                Run `SMA` indicator.

                * Inputs: `close`
                * Parameters: `timeperiod`, `timeframe`
                * Outputs: `real`

                Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `SMA.run_pipeline`.
            ```

            * To plot an indicator:

            ```pycon
            >>> sma.plot(column=(2, 'a')).show()
            ```

            ![](/assets/images/api/talib_plot.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("talib")
        import talib
        from talib import abstract

        func_name = func_name.upper()
        talib_func = getattr(talib, func_name)
        info = abstract.Function(func_name).info
        input_names = []
        for in_names in info["input_names"].values():
            if isinstance(in_names, (list, tuple)):
                input_names.extend(list(in_names))
            else:
                input_names.append(in_names)
        class_name = info["name"]
        class_docstring = "{}, {}".format(info["display_name"], info["group"])
        param_names = list(info["parameters"].keys()) + ["timeframe"]
        output_names = info["output_names"]
        output_flags = info["output_flags"]

        def apply_func_1d(
            input_tuple: tp.Tuple[tp.Array1d, ...],
            in_output_tuple: tp.Tuple[tp.Array1d, ...],
            param_tuple: tp.Tuple[tp.Param, ...],
            timeframe: tp.Optional[tp.FrequencyLike] = None,
            wrapper: tp.Optional[ArrayWrapper] = None,
            skipna: bool = False,
            silence_warnings: bool = False,
            **_kwargs,
        ) -> tp.Union[tp.Array1d, tp.Tuple[tp.Array1d]]:
            """1-dim apply function wrapping a TA-Lib function.

            Set `skipna` to True to run the TA-Lib function on non-NA values only."""
            if len(param_tuple) == len(param_names):
                if timeframe is not None:
                    raise ValueError("Time frame is set both as a parameter and as a keyword argument")
                timeframe = param_tuple[-1]
                param_tuple = param_tuple[:-1]
            elif len(param_tuple) > len(param_names):
                raise ValueError("Provided more parameters than registered")

            new_index = None
            if timeframe is not None:
                if wrapper is None:
                    raise ValueError("Resampling requires a wrapper")
                if wrapper.freq is None:
                    if not silence_warnings:
                        warnings.warn(
                            "Couldn't parse the frequency of index. Set freq in broadcast_kwargs or globally.",
                            stacklevel=2,
                        )
                new_input_tuple = ()
                for i, input in enumerate(input_tuple):
                    if input_names[i] == "open":
                        new_input = pd.Series(input, index=wrapper.index).resample(timeframe).first()
                    elif input_names[i] == "high":
                        new_input = pd.Series(input, index=wrapper.index).resample(timeframe).max()
                    elif input_names[i] == "low":
                        new_input = pd.Series(input, index=wrapper.index).resample(timeframe).min()
                    elif input_names[i] == "close":
                        new_input = pd.Series(input, index=wrapper.index).resample(timeframe).last()
                    elif input_names[i] == "volume":
                        new_input = pd.Series(input, index=wrapper.index).resample(timeframe).sum()
                    else:
                        raise ValueError(f"Can't resample '{input_names[i]}'")
                    new_index = new_input.index
                    new_input_tuple += (new_input.values,)
                input_tuple = new_input_tuple

            if skipna:
                nan_mask = build_nan_mask(*input_tuple)
                input_tuple = squeeze_nan(*input_tuple, nan_mask=nan_mask)
            else:
                nan_mask = None

            input_tuple = tuple([arr.astype(np.double) for arr in input_tuple])
            outputs = talib_func(*input_tuple, *param_tuple, **_kwargs)
            one_output = not isinstance(outputs, tuple)
            if one_output:
                outputs = unsqueeze_nan(outputs, nan_mask=nan_mask)
            else:
                outputs = unsqueeze_nan(*outputs, nan_mask=nan_mask)
            if timeframe is not None:
                new_outputs = ()
                for output in outputs:
                    source_freq = infer_index_freq(new_index, allow_date_offset=False, allow_numeric=False)
                    source_freq = freq_to_timedelta64(source_freq) if source_freq is not None else None
                    target_freq = freq_to_timedelta64(wrapper.freq) if wrapper.freq is not None else None
                    new_output = generic_nb.latest_at_index_1d_nb(
                        output,
                        new_index.values,
                        wrapper.index.values,
                        source_freq=source_freq,
                        target_freq=target_freq,
                        source_rbound=True,
                        target_rbound=True,
                        nan_value=np.nan,
                        ffill=True,
                    )
                    new_outputs += (new_output,)
                outputs = new_outputs
            if one_output:
                return outputs[0]
            return outputs

        kwargs = merge_dicts({k: Default(v) for k, v in info["parameters"].items()}, dict(timeframe=None), kwargs)
        Indicator = cls(
            **merge_dicts(
                dict(
                    class_name=class_name,
                    class_docstring=class_docstring,
                    module_name=__name__ + ".talib",
                    input_names=input_names,
                    param_names=param_names,
                    output_names=output_names,
                    output_flags=output_flags,
                ),
                factory_kwargs,
            )
        ).with_apply_func(apply_func_1d, pass_packed=True, takes_1d=True, pass_wrapper=True, **kwargs)

        def plot(
            self,
            column: tp.Optional[tp.Label] = None,
            limits: tp.Optional[tp.Tuple[float, float]] = None,
            add_shape_kwargs: tp.KwargsLike = None,
            add_trace_kwargs: tp.KwargsLike = None,
            fig: tp.Optional[tp.BaseFigure] = None,
            **kwargs,
        ) -> tp.BaseFigure:
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            self_col = self.select_col(column=column)

            output_trace_kwargs = {}
            for output_name in output_names:
                output_trace_kwargs[output_name] = kwargs.pop(output_name + "_trace_kwargs", {})
            priority_outputs = []
            other_outputs = []
            for output_name in output_names:
                flags = set(output_flags.get(output_name))
                found_priority = False
                if talib.abstract.TA_OUTPUT_FLAGS[2048] in flags:
                    priority_outputs = priority_outputs + [output_name]
                    found_priority = True
                if talib.abstract.TA_OUTPUT_FLAGS[4096] in flags:
                    priority_outputs = [output_name] + priority_outputs
                    found_priority = True
                if not found_priority:
                    other_outputs.append(output_name)

            for output_name in priority_outputs + other_outputs:
                output = getattr(self_col, output_name).rename(output_name)
                flags = set(output_flags.get(output_name))
                trace_kwargs = {}
                plot_func_name = "lineplot"

                if talib.abstract.TA_OUTPUT_FLAGS[2] in flags:
                    # Dotted Line
                    if "line" not in trace_kwargs:
                        trace_kwargs["line"] = dict()
                    trace_kwargs["line"]["dash"] = "dashdot"
                if talib.abstract.TA_OUTPUT_FLAGS[4] in flags:
                    # Dashed Line
                    if "line" not in trace_kwargs:
                        trace_kwargs["line"] = dict()
                    trace_kwargs["line"]["dash"] = "dash"
                if talib.abstract.TA_OUTPUT_FLAGS[8] in flags:
                    # Dot
                    if "line" not in trace_kwargs:
                        trace_kwargs["line"] = dict()
                    trace_kwargs["line"]["dash"] = "dot"
                if talib.abstract.TA_OUTPUT_FLAGS[16] in flags:
                    # Histogram
                    hist = np.asarray(output)
                    hist_diff = generic_nb.diff_1d_nb(hist)
                    marker_colors = np.full(hist.shape, adjust_opacity("silver", 0.75), dtype=object)
                    marker_colors[(hist > 0) & (hist_diff > 0)] = adjust_opacity("green", 0.75)
                    marker_colors[(hist > 0) & (hist_diff <= 0)] = adjust_opacity("lightgreen", 0.75)
                    marker_colors[(hist < 0) & (hist_diff < 0)] = adjust_opacity("red", 0.75)
                    marker_colors[(hist < 0) & (hist_diff >= 0)] = adjust_opacity("lightcoral", 0.75)
                    if "marker" not in trace_kwargs:
                        trace_kwargs["marker"] = {}
                    trace_kwargs["marker"]["color"] = marker_colors
                    if "line" not in trace_kwargs["marker"]:
                        trace_kwargs["marker"]["line"] = {}
                    trace_kwargs["marker"]["line"]["width"] = 0
                    kwargs["bargap"] = 0
                    plot_func_name = "barplot"
                if talib.abstract.TA_OUTPUT_FLAGS[2048] in flags:
                    # Values represent an upper limit
                    if "line" not in trace_kwargs:
                        trace_kwargs["line"] = {}
                    trace_kwargs["line"]["color"] = adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.75)
                    trace_kwargs["fill"] = "tonexty"
                    trace_kwargs["fillcolor"] = "rgba(128, 128, 128, 0.2)"
                if talib.abstract.TA_OUTPUT_FLAGS[4096] in flags:
                    # Values represent a lower limit
                    if "line" not in trace_kwargs:
                        trace_kwargs["line"] = {}
                    trace_kwargs["line"]["color"] = adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.75)

                trace_kwargs = merge_dicts(trace_kwargs, output_trace_kwargs[output_name])
                plot_func = getattr(output.vbt, plot_func_name)
                fig = plot_func(trace_kwargs=trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig, **kwargs)

            if limits is not None:
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

        signature = inspect.signature(plot)
        new_parameters = list(signature.parameters.values())[:-1]
        for output_name in output_names:
            new_parameters.append(
                inspect.Parameter(
                    output_name + "_trace_kwargs",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=tp.KwargsLike,
                )
            )
        new_parameters.append(inspect.Parameter("layout_kwargs", inspect.Parameter.VAR_KEYWORD))
        plot.__signature__ = signature.replace(parameters=new_parameters)
        output_trace_kwargs_docstring = "\n    ".join(
            [
                f"{output_name}_trace_kwargs (dict): Keyword arguments passed to the trace of `{output_name}`."
                for output_name in output_names
            ]
        )
        plot.__doc__ = f"""Plot the outputs of the indicator based on their flags.
        
Args:
    column (str): Name of the column to plot.
    limits (tuple of float): Tuple of the lower and upper limit.
    add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
    add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
    {output_trace_kwargs_docstring}
    fig (Figure or FigureWidget): Figure to add the traces to.
    **layout_kwargs: Keyword arguments passed to `fig.update_layout`."""
        setattr(Indicator, "plot", plot)

        return Indicator

    @classmethod
    def parse_pandas_ta_config(
        cls,
        func: tp.Callable,
        test_input_names: tp.Optional[tp.Sequence[str]] = None,
        test_index_len: int = 100,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Kwargs:
        """Get the config of a `pandas_ta` indicator."""
        if test_input_names is None:
            test_input_names = {"open_", "open", "high", "low", "close", "adj_close", "volume", "dividends", "split"}

        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # Parse the function signature of the indicator to get input names
        sig = inspect.signature(func)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation != inspect.Parameter.empty and v.annotation == pd.Series:
                    input_names.append(k)
                elif k in test_input_names:
                    input_names.append(k)
                elif v.default == inspect.Parameter.empty:
                    # Any positional argument is considered input
                    input_names.append(k)
                else:
                    param_names.append(k)
                    defaults[k] = v.default

        # To get output names, we need to run the indicator
        test_df = pd.DataFrame(
            {c: np.random.uniform(1, 10, size=(test_index_len,)) for c in input_names},
            index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(test_index_len)],
        )
        new_args = merge_dicts({c: test_df[c] for c in input_names}, kwargs)
        try:
            result = supress_stdout(func)(**new_args)
        except Exception as e:
            raise ValueError("Couldn't parse the indicator: " + str(e))

        # Concatenate Series/DataFrames if the result is a tuple
        if isinstance(result, tuple):
            results = []
            for i, r in enumerate(result):
                if len(r.index) != len(test_df.index):
                    if not silence_warnings:
                        warnings.warn(f"Couldn't parse the output at index {i}: mismatching index", stacklevel=2)
                else:
                    results.append(r)
            if len(results) > 1:
                result = pd.concat(results, axis=1)
            elif len(results) == 1:
                result = results[0]
            else:
                raise ValueError("Couldn't parse the output")

        # Test if the produced array has the same index length
        if len(result.index) != len(test_df.index):
            raise ValueError("Couldn't parse the output: mismatching index")

        # Standardize output names: remove numbers, remove hyphens, and bring to lower case
        output_cols = result.columns.tolist() if isinstance(result, pd.DataFrame) else [result.name]
        new_output_cols = []
        for i in range(len(output_cols)):
            name_parts = []
            for name_part in output_cols[i].split("_"):
                try:
                    float(name_part)
                    continue
                except:
                    name_parts.append(name_part.replace("-", "_").lower())
            output_col = "_".join(name_parts)
            new_output_cols.append(output_col)

        # Add numbers to duplicates
        for k, v in Counter(new_output_cols).items():
            if v == 1:
                output_names.append(k)
            else:
                for i in range(v):
                    output_names.append(k + str(i))

        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults,
        )

    @classmethod
    def list_pandas_ta_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.List[str]:
        """List all parseable indicators in `pandas_ta`.

        !!! note
            Returns only the indicators that have been successfully parsed."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pandas_ta")
        import pandas_ta

        indicators = set()
        for func_name in [_k for k, v in pandas_ta.Category.items() for _k in v]:
            try:
                cls.parse_pandas_ta_config(getattr(pandas_ta, func_name), silence_warnings=silence_warnings, **kwargs)
                indicators.add(func_name.upper())
            except Exception as e:
                if not silence_warnings:
                    warnings.warn(f"Function {func_name}: " + str(e), stacklevel=2)
        return sorted(indicators)

    @classmethod
    def from_pandas_ta(
        cls,
        func_name: str,
        parse_kwargs: tp.KwargsLike = None,
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a `pandas_ta` function.

        Requires [pandas-ta](https://github.com/twopirllc/pandas-ta) installed.

        Args:
            func_name (str): Function name.
            parse_kwargs (dict): Keyword arguments passed to `IndicatorFactory.parse_pandas_ta_config`.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMA = vbt.IF.from_pandas_ta('SMA')

            >>> sma = SMA.run(price, length=[2, 3])
            >>> sma.sma
            sma_length         2         3
                          a    b    a    b
            2020-01-01  NaN  NaN  NaN  NaN
            2020-01-02  1.5  4.5  NaN  NaN
            2020-01-03  2.5  3.5  2.0  4.0
            2020-01-04  3.5  2.5  3.0  3.0
            2020-01-05  4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use `vectorbtpro.utils.formatting.phelp`:

            ```pycon
            >>> vbt.phelp(SMA.run)
            SMA.run(
                close,
                length=Default(value=None),
                talib=Default(value=None),
                offset=Default(value=None),
                short_name='sma',
                hide_params=None,
                hide_default=True,
                **kwargs
            ):
                Run `SMA` indicator.

                * Inputs: `close`
                * Parameters: `length`, `talib`, `offset`
                * Outputs: `sma`

                Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `SMA.run_pipeline`.
            ```

            * To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMA.__doc__)
            Simple Moving Average (SMA)

            The Simple Moving Average is the classic moving average that is the equally
            weighted average over n periods.

            Sources:
                https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

            Calculation:
                Default Inputs:
                    length=10
                SMA = SUM(close, length) / length

            Args:
                close (pd.Series): Series of 'close's
                length (int): It's period. Default: 10
                offset (int): How many periods to offset the result. Default: 0

            Kwargs:
                adjust (bool): Default: True
                presma (bool, optional): If True, uses SMA for initial value.
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method

            Returns:
                pd.Series: New feature generated.
            ```
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pandas_ta")
        import pandas_ta

        func_name = func_name.lower()
        func = getattr(pandas_ta, func_name)

        if parse_kwargs is None:
            parse_kwargs = {}
        config = cls.parse_pandas_ta_config(func, **parse_kwargs)

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.Param, ...],
            **_kwargs,
        ) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                output = supress_stdout(func)(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config["input_names"])
                    },
                    **{name: param_tuple[i] for i, name in enumerate(config["param_names"])},
                    **_kwargs,
                )
                if isinstance(output, tuple):
                    _outputs = []
                    for o in output:
                        if len(input_tuple[0].index) == len(o.index):
                            _outputs.append(o)
                    if len(_outputs) > 1:
                        output = pd.concat(_outputs, axis=1)
                    elif len(_outputs) == 1:
                        output = _outputs[0]
                    else:
                        raise ValueError("No valid outputs were returned")
                if isinstance(output, pd.DataFrame):
                    output = tuple([output.iloc[:, i] for i in range(len(output.columns))])
                outputs.append(output)
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(column_stack, outputs))
            return column_stack(outputs)

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".pandas_ta"), config, factory_kwargs),
        ).with_apply_func(apply_func, pass_packed=True, keep_pd=True, to_2d=False, **kwargs)
        return Indicator

    @classmethod
    def list_ta_indicators(cls, uppercase: bool = False) -> tp.List[str]:
        """List all parseable indicators in `ta`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        indicators = set()
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and obj != ta.utils.IndicatorMixin
                    and issubclass(obj, ta.utils.IndicatorMixin)
                ):
                    if uppercase:
                        indicators.add(obj.__name__.upper())
                    else:
                        indicators.add(obj.__name__)
        return sorted(indicators)

    @classmethod
    def find_ta_indicator(cls, cls_name: str) -> IndicatorMixinT:
        """Get `ta` indicator class by its name."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            for attr in dir(module):
                if cls_name.upper() == attr.upper():
                    return getattr(module, attr)
        raise AttributeError(f"Indicator '{cls_name}' not found")

    @classmethod
    def parse_ta_config(cls, ind_cls: IndicatorMixinT) -> tp.Kwargs:
        """Get the config of a `ta` indicator."""
        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # Parse the __init__ signature of the indicator class to get input names
        sig = inspect.signature(ind_cls)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation == inspect.Parameter.empty:
                    raise ValueError(f'Argument "{k}" has no annotation')
                if v.annotation == pd.Series:
                    input_names.append(k)
                else:
                    param_names.append(k)
                    if v.default != inspect.Parameter.empty:
                        defaults[k] = v.default

        # Get output names by looking into instance methods
        for attr in dir(ind_cls):
            if not attr.startswith("_"):
                if inspect.signature(getattr(ind_cls, attr)).return_annotation == pd.Series:
                    output_names.append(attr)
                elif "Returns:\n            pandas.Series" in getattr(ind_cls, attr).__doc__:
                    output_names.append(attr)

        return dict(
            class_name=ind_cls.__name__,
            class_docstring=ind_cls.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults,
        )

    @classmethod
    def from_ta(cls, cls_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a `ta` class.

        Requires [ta](https://github.com/bukosabino/ta) installed.

        Args:
            cls_name (str): Class name.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMAIndicator = vbt.IF.from_ta('SMAIndicator')

            >>> sma = SMAIndicator.run(price, window=[2, 3])
            >>> sma.sma_indicator
            smaindicator_window    2         3
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           1.5  4.5  NaN  NaN
            2020-01-03           2.5  3.5  2.0  4.0
            2020-01-04           3.5  2.5  3.0  3.0
            2020-01-05           4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use `vectorbtpro.utils.formatting.phelp`:

            ```pycon
            >>> vbt.phelp(SMAIndicator.run)
            SMAIndicator.run(
                close,
                window,
                fillna=Default(value=False),
                short_name='smaindicator',
                hide_params=None,
                hide_default=True,
                **kwargs
            ):
                Run `SMAIndicator` indicator.

                * Inputs: `close`
                * Parameters: `window`, `fillna`
                * Outputs: `sma_indicator`

                Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `SMAIndicator.run_pipeline`.
            ```

            * To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMAIndicator.__doc__)
            SMA - Simple Moving Average

                Args:
                    close(pandas.Series): dataset 'Close' column.
                    window(int): n period.
                    fillna(bool): if True, fill nan values.
            ```
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")

        ind_cls = cls.find_ta_indicator(cls_name)
        config = cls.parse_ta_config(ind_cls)

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.Param, ...],
            **_kwargs,
        ) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                ind = ind_cls(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config["input_names"])
                    },
                    **{name: param_tuple[i] for i, name in enumerate(config["param_names"])},
                    **_kwargs,
                )
                output = []
                for output_name in config["output_names"]:
                    output.append(getattr(ind, output_name)())
                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)
                outputs.append(output)
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(column_stack, outputs))
            return column_stack(outputs)

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(**merge_dicts(dict(module_name=__name__ + ".ta"), config, factory_kwargs)).with_apply_func(
            apply_func,
            pass_packed=True,
            keep_pd=True,
            to_2d=False,
            **kwargs,
        )
        return Indicator

    @classmethod
    def parse_technical_config(cls, func: tp.Callable, test_index_len: int = 100) -> tp.Kwargs:
        """Get the config of a `technical` indicator."""
        df = pd.DataFrame(
            np.random.randint(1, 10, size=(test_index_len, 5)),
            index=pd.date_range("2020", periods=test_index_len),
            columns=["open", "high", "low", "close", "volume"],
        )

        func_arg_names = get_func_arg_names(func)
        func_kwargs = get_func_kwargs(func)
        args = ()
        input_names = []
        param_names = []
        output_names = []
        defaults = {}

        for arg_name in func_arg_names:
            if arg_name == "field":
                continue
            if arg_name in ("dataframe", "df", "bars"):
                args += (df,)
                if "field" in func_kwargs:
                    input_names.append(func_kwargs["field"])
                else:
                    input_names.extend(["open", "high", "low", "close", "volume"])
            elif arg_name in ("series", "sr"):
                args += (df["close"],)
                input_names.append("close")
            elif arg_name in ("open", "high", "low", "close", "volume"):
                args += (df["close"],)
                input_names.append(arg_name)
            else:
                if arg_name not in func_kwargs:
                    args += (5,)
                else:
                    defaults[arg_name] = func_kwargs[arg_name]
                param_names.append(arg_name)
        if len(input_names) == 0:
            raise ValueError("Couldn't parse the output: unknown input arguments")

        def _validate_series(sr, name: tp.Optional[str] = None):
            if not isinstance(sr, pd.Series):
                raise TypeError("Couldn't parse the output: wrong output type")
            if len(sr.index) != len(df.index):
                raise ValueError("Couldn't parse the output: mismatching index")
            if np.issubdtype(sr.dtype, object):
                raise ValueError("Couldn't parse the output: wrong output data type")
            if name is None and sr.name is None:
                raise ValueError("Couldn't parse the output: missing output name")

        out = supress_stdout(func)(*args)
        if isinstance(out, list):
            out = np.asarray(out)
        if isinstance(out, np.ndarray):
            out = pd.Series(out)
        if isinstance(out, dict):
            out = pd.DataFrame(out)
        if isinstance(out, tuple):
            out = pd.concat(out, axis=1)
        if isinstance(out, (pd.Series, pd.DataFrame)):
            if isinstance(out, pd.DataFrame):
                for c in out.columns:
                    _validate_series(out[c], name=c)
                    output_names.append(c)
            else:
                if out.name is not None:
                    out_name = out.name
                else:
                    out_name = func.__name__.lower()
                _validate_series(out, name=out_name)
                output_names.append(out_name)
        else:
            raise TypeError("Couldn't parse the output: wrong output type")

        new_output_names = []
        for name in output_names:
            name = name.replace(" ", "").lower()
            if len(output_names) == 1 and name == "close":
                new_output_names.append(func.__name__.lower())
                continue
            if name in ("open", "high", "low", "close", "volume", "data"):
                continue
            new_output_names.append(name)
        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=new_output_names,
            defaults=defaults,
        )

    @classmethod
    def list_technical_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.List[str]:
        """List all parseable indicators in `technical`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        import technical

        funcs = search_package_for_funcs(technical, blacklist=["technical.util"])
        indicators = set()
        for func_name, func in funcs.items():
            try:
                cls.parse_technical_config(func, **kwargs)
                indicators.add(func_name.upper())
            except Exception as e:
                if not silence_warnings:
                    warnings.warn(f"Function {func_name}: " + str(e), stacklevel=2)
        return sorted(indicators)

    @classmethod
    def find_technical_indicator(cls, func_name: str) -> IndicatorMixinT:
        """Get `technical` indicator function by its name."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        import technical

        funcs = search_package_for_funcs(technical, blacklist=["technical.util"])
        for k, v in funcs.items():
            if func_name.upper() == k.upper():
                return v
        raise AttributeError(f"Indicator '{func_name}' not found")

    @classmethod
    def from_technical(
        cls,
        func_name: str,
        parse_kwargs: tp.KwargsLike = None,
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a `technical` function.

        Requires [technical](https://github.com/freqtrade/technical) installed.

        Args:
            func_name (str): Function name.
            parse_kwargs (dict): Keyword arguments passed to `IndicatorFactory.parse_technical_config`.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> ROLLING_MEAN = vbt.IF.from_technical("ROLLING_MEAN")

            >>> rolling_mean = ROLLING_MEAN.run(price, window=[3, 4])
            >>> rolling_mean.rolling_mean
            rolling_mean_window         3         4
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           NaN  NaN  NaN  NaN
            2020-01-03           2.0  4.0  NaN  NaN
            2020-01-04           3.0  3.0  2.5  3.5
            2020-01-05           4.0  2.0  3.5  2.5
            ```

            * To get help on running the indicator, use `vectorbtpro.utils.formatting.phelp`:

            ```pycon
            >>> vbt.phelp(ROLLING_MEAN.run)
            ROLLING_MEAN.run(
                close,
                window=Default(value=200),
                min_periods=Default(value=None),
                short_name='rolling_mean',
                hide_params=None,
                hide_default=True,
                **kwargs
            ):
                Run `ROLLING_MEAN` indicator.

                * Inputs: `close`
                * Parameters: `window`, `min_periods`
                * Outputs: `rolling_mean`

                Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `ROLLING_MEAN.run_pipeline`.
            ```
        """
        func = cls.find_technical_indicator(func_name)
        func_arg_names = get_func_arg_names(func)

        if parse_kwargs is None:
            parse_kwargs = {}
        config = cls.parse_technical_config(func, **parse_kwargs)

        def apply_func(
            input_tuple: tp.Tuple[tp.Series, ...],
            in_output_tuple: tp.Tuple[tp.Series, ...],
            param_tuple: tp.Tuple[tp.Param, ...],
            *_args,
            **_kwargs,
        ) -> tp.Union[tp.Array1d, tp.List[tp.Array1d]]:
            input_series = {name: input_tuple[i] for i, name in enumerate(config["input_names"])}
            _kwargs = {**{name: param_tuple[i] for i, name in enumerate(config["param_names"])}, **_kwargs}
            __args = ()
            for arg_name in func_arg_names:
                if arg_name in ("dataframe", "df", "bars"):
                    __args += (pd.DataFrame(input_series),)
                elif arg_name in ("series", "sr"):
                    __args += (input_series["close"],)
                elif arg_name in ("open", "high", "low", "close", "volume"):
                    __args += (input_series["close"],)
                else:
                    break

            out = supress_stdout(func)(*__args, *_args, **_kwargs)
            if isinstance(out, list):
                out = np.asarray(out)
            if isinstance(out, np.ndarray):
                out = pd.Series(out)
            if isinstance(out, dict):
                out = pd.DataFrame(out)
            if isinstance(out, tuple):
                out = pd.concat(out, axis=1)
            if isinstance(out, pd.DataFrame):
                outputs = []
                for c in out.columns:
                    if len(out.columns) == len(config["output_names"]):
                        outputs.append(out[c].values)
                    elif c not in ("open", "high", "low", "close", "volume", "data"):
                        outputs.append(out[c].values)
                return outputs
            return out.values

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".technical"), config, factory_kwargs),
        ).with_apply_func(apply_func, pass_packed=True, keep_pd=True, takes_1d=True, **kwargs)
        return Indicator

    @classmethod
    def from_custom_techcon(
        cls,
        consensus_cls: tp.Type[ConsensusT],
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Create an indicator based on a technical consensus class subclassing
        `technical.consensus.consensus.Consensus`.

        Requires Technical library: https://github.com/freqtrade/technical"""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        from technical.consensus.consensus import Consensus

        checks.assert_subclass_of(consensus_cls, Consensus)

        def apply_func(
            open: tp.Series,
            high: tp.Series,
            low: tp.Series,
            close: tp.Series,
            volume: tp.Series,
            smooth: tp.Optional[int] = None,
            _consensus_cls: tp.Type[ConsensusT] = consensus_cls,
        ) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
            """Apply function for `technical.consensus.movingaverage.MovingAverageConsensus`."""
            dataframe = pd.DataFrame(
                {
                    "open": open,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            consensus = _consensus_cls(dataframe)
            score = consensus.score(smooth=smooth)
            return (
                score["buy"].values,
                score["sell"].values,
                score["buy_agreement"].values,
                score["sell_agreement"].values,
                score["buy_disagreement"].values,
                score["sell_disagreement"].values,
            )

        if factory_kwargs is None:
            factory_kwargs = {}
        factory_kwargs = merge_dicts(
            dict(
                class_name="CON",
                module_name=__name__ + ".custom_techcon",
                short_name=None,
                input_names=["open", "high", "low", "close", "volume"],
                param_names=["smooth"],
                output_names=[
                    "buy",
                    "sell",
                    "buy_agreement",
                    "sell_agreement",
                    "buy_disagreement",
                    "sell_disagreement",
                ],
            ),
            factory_kwargs,
        )
        Indicator = cls(**factory_kwargs).with_apply_func(
            apply_func,
            takes_1d=True,
            keep_pd=True,
            smooth=None,
            **kwargs,
        )

        def plot(
            self,
            column: tp.Optional[tp.Label] = None,
            buy_trace_kwargs: tp.KwargsLike = None,
            sell_trace_kwargs: tp.KwargsLike = None,
            add_trace_kwargs: tp.KwargsLike = None,
            fig: tp.Optional[tp.BaseFigure] = None,
            **layout_kwargs,
        ) -> tp.BaseFigure:
            """Plot `MA.ma` against `MA.close`.

            Args:
                column (str): Name of the column to plot.
                buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `buy`.
                sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `sell`.
                add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
                fig (Figure or FigureWidget): Figure to add traces to.
                **layout_kwargs: Keyword arguments passed to `fig.update_layout`.
            """
            from vectorbtpro.utils.figure import make_figure
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            self_col = self.select_col(column=column)

            if fig is None:
                fig = make_figure()
            fig.update_layout(**layout_kwargs)

            if buy_trace_kwargs is None:
                buy_trace_kwargs = {}
            if sell_trace_kwargs is None:
                sell_trace_kwargs = {}
            buy_trace_kwargs = merge_dicts(
                dict(name="Buy", line=dict(color=plotting_cfg["color_schema"]["green"])),
                buy_trace_kwargs,
            )
            sell_trace_kwargs = merge_dicts(
                dict(name="Sell", line=dict(color=plotting_cfg["color_schema"]["red"])),
                sell_trace_kwargs,
            )

            fig = self_col.buy.vbt.lineplot(
                trace_kwargs=buy_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            fig = self_col.sell.vbt.lineplot(
                trace_kwargs=sell_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

            return fig

        Indicator.plot = plot
        return Indicator

    @classmethod
    def from_techcon(cls, cls_name: str, **kwargs) -> tp.Type[IndicatorBase]:
        """Create an indicator from a preset technical consensus.

        Supported are case-insensitive values `MACON` (or `MovingAverageConsensus`),
        `OSCCON` (or `OscillatorConsensus`), and `SUMCON` (or `SummaryConsensus`)."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")

        if cls_name.lower() in ("MACON".lower(), "MovingAverageConsensus".lower()):
            from technical.consensus.movingaverage import MovingAverageConsensus

            return IndicatorFactory.from_custom_techcon(
                MovingAverageConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="MACON"),
                **kwargs,
            )
        if cls_name.lower() in ("OSCCON".lower(), "OscillatorConsensus".lower()):
            from technical.consensus.oscillator import OscillatorConsensus

            return IndicatorFactory.from_custom_techcon(
                OscillatorConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="OSCCON"),
                **kwargs,
            )
        if cls_name.lower() in ("SUMCON".lower(), "SummaryConsensus".lower()):
            from technical.consensus.summary import SummaryConsensus

            return IndicatorFactory.from_custom_techcon(
                SummaryConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="SUMCON"),
                **kwargs,
            )
        raise ValueError(f"Unknown technical consensus class '{cls_name}'")

    @classmethod
    def list_techcon_indicators(cls) -> tp.List[str]:
        """List all consensus indicators in `technical`."""
        return sorted({"MACON", "OSCCON", "SUMCON"})

    # ############# Expressions ############# #

    @class_or_instancemethod
    def from_expr(
        cls_or_self,
        expr: str,
        parse_annotations: bool = True,
        factory_kwargs: tp.KwargsLike = None,
        magnet_inputs: tp.Iterable[str] = None,
        magnet_in_outputs: tp.Iterable[str] = None,
        magnet_params: tp.Iterable[str] = None,
        func_mapping: tp.KwargsLike = None,
        res_func_mapping: tp.KwargsLike = None,
        use_pd_eval: tp.Optional[bool] = None,
        pd_eval_kwargs: tp.KwargsLike = None,
        return_clean_expr: bool = False,
        **kwargs,
    ) -> tp.Union[str, tp.Type[IndicatorBase]]:
        """Build an indicator class from an indicator expression.

        Args:
            expr (str): Expression.

                Expression must be a string with a valid Python code.
                Supported are both single-line and multi-line expressions.
            parse_annotations (bool): Whether to parse annotations starting with `@`.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.

                Only applied when calling the class method.
            magnet_inputs (iterable of str): Names recognized as input names.

                Defaults to `open`, `high`, `low`, `close`, and `volume`.
            magnet_in_outputs (iterable of str): Names recognized as in-output names.

                Defaults to an empty list.
            magnet_params (iterable of str): Names recognized as params names.

                Defaults to an empty list.
            func_mapping (mapping): Mapping merged over `vectorbtpro.indicators.expr.expr_func_config`.

                Each key must be a function name and each value must be a dict with
                `func` and optionally `magnet_inputs`, `magnet_in_outputs`, and `magnet_params`.
            res_func_mapping (mapping): Mapping merged over `vectorbtpro.indicators.expr.expr_res_func_config`.

                Each key must be a function name and each value must be a dict with
                `func` and optionally `magnet_inputs`, `magnet_in_outputs`, and `magnet_params`.
            use_pd_eval (bool): Whether to use `pd.eval`.

                Defaults to False.

                Otherwise, uses `vectorbtpro.utils.eval_.multiline_eval`.

                !!! hint
                    By default, operates on NumPy objects using NumExpr.
                    If you want to operate on Pandas objects, set `keep_pd` to True.
            pd_eval_kwargs (dict): Keyword arguments passed to `pd.eval`.
            return_clean_expr (bool): Whether to return a cleaned expression.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator

        Searches each variable name parsed from `expr` in

        * `vectorbtpro.indicators.expr.expr_res_func_config` (calls right away)
        * `vectorbtpro.indicators.expr.expr_func_config`
        * inputs, in-outputs, and params
        * keyword arguments
        * attributes of `np`
        * attributes of `vectorbtpro.generic.nb` (with and without `_nb` suffix)
        * attributes of `vbt`

        `vectorbtpro.indicators.expr.expr_func_config` and `vectorbtpro.indicators.expr.expr_res_func_config`
        can be overridden with `func_mapping` and `res_func_mapping` respectively.

        !!! note
            Each variable name is case-sensitive.

        When using the class method, all names are parsed from the expression itself.
        If any of `open`, `high`, `low`, `close`, and `volume` appear in the expression or
        in `magnet_inputs` in either `vectorbtpro.indicators.expr.expr_func_config` or
        `vectorbtpro.indicators.expr.expr_res_func_config`, they are automatically added to `input_names`.
        Set `magnet_inputs` to an empty list to disable this logic.

        If the expression begins with a valid variable name and a colon (`:`), the variable name
        will be used as the name of the generated class. Provide another variable in the square brackets
        after this one and before the colon to specify the indicator's short name.

        If `parse_annotations` is True, variables that start with `@` have a special meaning:

        * `@in_*`: input variable
        * `@inout_*`: in-output variable
        * `@p_*`: parameter variable
        * `@out_*`: output variable
        * `@out_*:`: indicates that the next part until a comma is an output
        * `@talib_*`: name of a TA-Lib function. Uses the indicator's `apply_func`.
        * `@res_*`: name of the indicator to resolve automatically. Input names can overlap with
            those of other indicators, while all other information gets a prefix with the indicator's short name.
        * `@settings(*)`: settings to be merged with the current `IndicatorFactory.from_expr` settings.
            Everything within the parentheses gets evaluated using the Pythons `eval` command
            and must be a dictionary. Overrides defaults but gets overridden by any argument
            passed to this method. Arguments `expr` and `parse_annotations` cannot be overridden.

        !!! note
            The parsed names come in the same order they appear in the expression, not in the execution order,
            apart from the magnet input names, which are added in the same order they appear in the list.

        The number of outputs is derived based on the number of commas outside of any bracket pair.
        If there is only one output, the output name is `out`. If more - `out1`, `out2`, etc.

        Any information can be overridden using `factory_kwargs`.

        Usage:
            ```pycon
            >>> WMA = vbt.IF(
            ...     class_name='WMA',
            ...     input_names=['close'],
            ...     param_names=['window'],
            ...     output_names=['wma']
            ... ).from_expr("wm_mean_nb(close, window)")

            >>> wma = WMA.run(price, window=[2, 3])
            >>> wma.wma
            wma_window                   2                   3
                               a         b         a         b
            2020-01-01       NaN       NaN       NaN       NaN
            2020-01-02  1.666667  4.333333       NaN       NaN
            2020-01-03  2.666667  3.333333  2.333333  3.666667
            2020-01-04  3.666667  2.333333  3.333333  2.666667
            2020-01-05  4.666667  1.333333  4.333333  1.666667
            ```

            * The same can be achieved by calling the class method and providing prefixes
            to the variable names to indicate their type:

            ```pycon
            >>> expr = "WMA: @out_wma:wm_mean_nb((@in_high + @in_low) / 2, @p_window)"
            >>> WMA = vbt.IF.from_expr(expr)
            >>> wma = WMA.run(price + 1, price, window=[2, 3])
            >>> wma.wma
            wma_window                   2                   3
                               a         b         a         b
            2020-01-01       NaN       NaN       NaN       NaN
            2020-01-02  2.166667  4.833333       NaN       NaN
            2020-01-03  3.166667  3.833333  2.833333  4.166667
            2020-01-04  4.166667  2.833333  3.833333  3.166667
            2020-01-05  5.166667  1.833333  4.833333  2.166667
            ```

            * Magnet names are recognized automatically:

            ```pycon
            >>> expr = "WMA: @out_wma:wm_mean_nb((high + low) / 2, @p_window)"
            ```

            * Most settings of this method can be overriden from within the expression:

            ```pycon
            >>> expr = \"\"\"
            ... @settings({factory_kwargs={'class_name': 'WMA', 'param_names': ['window']}})
            ... @out_wma:wm_mean_nb((high + low) / 2, window)
            ... \"\"\"
            ```
        """

        def _clean_expr(expr: str) -> str:
            # Clean the expression from redundant brackets and commas
            expr = inspect.cleandoc(expr).strip()
            if expr.endswith(","):
                expr = expr[:-1]
            if expr.startswith("(") and expr.endswith(")"):
                n_open_brackets = 0
                remove_brackets = True
                for i, s in enumerate(expr):
                    if s == "(":
                        n_open_brackets += 1
                    elif s == ")":
                        n_open_brackets -= 1
                        if n_open_brackets == 0 and i < len(expr) - 1:
                            remove_brackets = False
                            break
                if remove_brackets:
                    expr = expr[1:-1]
            if expr.endswith(","):
                expr = expr[:-1]  # again
            return expr

        if isinstance(cls_or_self, type):
            settings = dict(
                factory_kwargs=dict(
                    class_name=None,
                    input_names=[],
                    in_output_names=[],
                    param_names=[],
                    output_names=[],
                )
            )

            # Parse the class name
            match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[([a-zA-Z_][a-zA-Z0-9_]*)\])?\s*:\s*", expr)
            if match:
                settings["factory_kwargs"]["class_name"] = match.group(1)
                if match.group(2):
                    settings["factory_kwargs"]["short_name"] = match.group(2)
                expr = expr[len(match.group(0)) :]

            # Parse the settings dictionary
            if "@settings" in expr:
                remove_chars = set()
                for m in re.finditer("@settings", expr):
                    n_open_brackets = 0
                    from_i = None
                    to_i = None
                    for i in range(m.start(), m.end()):
                        remove_chars.add(i)
                    for i in range(m.end(), len(expr)):
                        remove_chars.add(i)
                        s = expr[i]
                        if s in "(":
                            if n_open_brackets == 0:
                                from_i = i + 1
                            n_open_brackets += 1
                        elif s in ")":
                            n_open_brackets -= 1
                            if n_open_brackets == 0:
                                to_i = i
                                break
                    if n_open_brackets != 0:
                        raise ValueError("Couldn't parse the settings: mismatching brackets")
                    settings = merge_dicts(settings, eval(_clean_expr(expr[from_i:to_i])))
                expr = "".join([expr[i] for i in range(len(expr)) if i not in remove_chars])

            expr = _clean_expr(expr)

            # Merge info
            parsed_factory_kwargs = settings.pop("factory_kwargs")
            magnet_inputs = settings.pop("magnet_inputs", magnet_inputs)
            magnet_in_outputs = settings.pop("magnet_in_outputs", magnet_in_outputs)
            magnet_params = settings.pop("magnet_params", magnet_params)
            func_mapping = merge_dicts(expr_func_config, settings.pop("func_mapping", None), func_mapping)
            res_func_mapping = merge_dicts(
                expr_res_func_config,
                settings.pop("res_func_mapping", None),
                res_func_mapping,
            )
            use_pd_eval = settings.pop("use_pd_eval", use_pd_eval)
            pd_eval_kwargs = merge_dicts(settings.pop("pd_eval_kwargs", None), pd_eval_kwargs)

            # Resolve defaults
            if use_pd_eval is None:
                use_pd_eval = False
            if magnet_inputs is None:
                magnet_inputs = ["open", "high", "low", "close", "volume"]
            if magnet_in_outputs is None:
                magnet_in_outputs = []
            if magnet_params is None:
                magnet_params = []
            found_magnet_inputs = []
            found_magnet_in_outputs = []
            found_magnet_params = []
            found_defaults = {}
            remove_defaults = set()

            # Parse annotated variables
            if parse_annotations:
                # Parse input, in-output, parameter, and TA-Lib function names
                for var_name in re.findall(r"@[a-z]+_[a-zA-Z_][a-zA-Z0-9_]*", expr):
                    var_name = var_name.replace("@", "")
                    if var_name.startswith("in_"):
                        var_name = var_name[3:]
                        if var_name in magnet_inputs:
                            if var_name not in found_magnet_inputs:
                                found_magnet_inputs.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["input_names"]:
                                parsed_factory_kwargs["input_names"].append(var_name)
                    elif var_name.startswith("inout_"):
                        var_name = var_name[6:]
                        if var_name in magnet_in_outputs:
                            if var_name not in found_magnet_in_outputs:
                                found_magnet_in_outputs.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["in_output_names"]:
                                parsed_factory_kwargs["in_output_names"].append(var_name)
                    elif var_name.startswith("p_"):
                        var_name = var_name[2:]
                        if var_name in magnet_params:
                            if var_name not in found_magnet_params:
                                found_magnet_params.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["param_names"]:
                                parsed_factory_kwargs["param_names"].append(var_name)
                    elif var_name.startswith("res_"):
                        ind_name = var_name[4:]
                        if ind_name.startswith("talib_"):
                            ind_name = ind_name[6:]
                            I = IndicatorFactory.from_talib(ind_name)
                        else:
                            I = kwargs[ind_name]
                        if not issubclass(I, IndicatorBase):
                            raise TypeError(f"Indicator class '{ind_name}' must subclass IndicatorBase")

                        def _ind_func(context: tp.Kwargs, _I: IndicatorBase = I) -> tp.Any:
                            _args = ()
                            _kwargs = {}
                            signature = inspect.signature(_I.run)
                            for p in signature.parameters.values():
                                if p.name in _I.input_names:
                                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                                        _args += (context[p.name],)
                                    else:
                                        _kwargs[p.name] = context[p.name]
                                else:
                                    ind_p_name = _I.short_name + "_" + p.name
                                    if ind_p_name in context:
                                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                                            _args += (context[ind_p_name],)
                                        elif p.kind == p.VAR_POSITIONAL:
                                            _args += context[ind_p_name]
                                        elif p.kind == p.VAR_KEYWORD:
                                            for k, v in context[ind_p_name].items():
                                                _kwargs[k] = v
                                        else:
                                            _kwargs[p.name] = context[ind_p_name]
                            return_raw = _kwargs.pop("return_raw", True)
                            ind = _I.run(*_args, return_raw=return_raw, **_kwargs)
                            if return_raw:
                                raw_outputs = ind[0]
                                if len(raw_outputs) == 1:
                                    return raw_outputs[0]
                                return raw_outputs
                            return ind

                        res_func_mapping["__" + var_name] = dict(
                            func=_ind_func,
                            magnet_inputs=I.input_names,
                            magnet_in_outputs=[I.short_name + "_" + name for name in I.in_output_names],
                            magnet_params=[I.short_name + "_" + name for name in I.param_names],
                        )
                        run_kwargs = get_func_kwargs(I.run)

                        def _add_defaults(names, prefix=None):
                            for k in names:
                                if prefix is None:
                                    k_prefixed = k
                                else:
                                    k_prefixed = prefix + "_" + k
                                if k in run_kwargs:
                                    if k_prefixed in found_defaults:
                                        if not checks.is_deep_equal(found_defaults[k_prefixed], run_kwargs[k]):
                                            remove_defaults.add(k_prefixed)
                                    else:
                                        found_defaults[k_prefixed] = run_kwargs[k]

                        _add_defaults(I.input_names)
                        _add_defaults(I.in_output_names, I.short_name)
                        _add_defaults(I.param_names, I.short_name)

                expr = expr.replace("@in_", "__in_")
                expr = expr.replace("@inout_", "__inout_")
                expr = expr.replace("@p_", "__p_")
                expr = expr.replace("@talib_", "__talib_")
                expr = expr.replace("@res_", "__res_")

                # Parse output names
                to_replace = []
                for var_name in re.findall(r"@out_[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*", expr):
                    to_replace.append(var_name)
                    var_name = var_name.split(":")[0].strip()[5:]
                    if var_name not in parsed_factory_kwargs["output_names"]:
                        parsed_factory_kwargs["output_names"].append(var_name)
                for s in to_replace:
                    expr = expr.replace(s, "")

                for var_name in re.findall(r"@out_[a-zA-Z_][a-zA-Z0-9_]*", expr):
                    var_name = var_name.replace("@", "")
                    if var_name.startswith("out_"):
                        var_name = var_name[4:]
                        if var_name not in parsed_factory_kwargs["output_names"]:
                            parsed_factory_kwargs["output_names"].append(var_name)

                expr = expr.replace("@out_", "__out_")

                if len(parsed_factory_kwargs["output_names"]) == 0:
                    lines = expr.split("\n")
                    if len(lines) > 1:
                        last_line = _clean_expr(lines[-1])
                        valid_output_names = []
                        found_not_valid = False
                        for i, out in enumerate(last_line.split(",")):
                            out = out.strip()
                            if not out.startswith("__") and out.isidentifier():
                                valid_output_names.append(out)
                            else:
                                found_not_valid = True
                                break
                        if not found_not_valid:
                            parsed_factory_kwargs["output_names"] = valid_output_names

            # Parse magnet names
            var_names = get_expr_var_names(expr)

            def _find_magnets(magnet_type, magnet_names, magnet_lst, found_magnet_lst):
                for var_name in var_names:
                    if var_name in magnet_lst:
                        if var_name not in found_magnet_lst:
                            found_magnet_lst.append(var_name)
                    if var_name in func_mapping:
                        for magnet_name in func_mapping[var_name].get(magnet_type, []):
                            if magnet_name not in found_magnet_lst:
                                found_magnet_lst.append(magnet_name)
                    if var_name in res_func_mapping:
                        for magnet_name in res_func_mapping[var_name].get(magnet_type, []):
                            if magnet_name not in found_magnet_lst:
                                found_magnet_lst.append(magnet_name)
                for magnet_name in magnet_lst:
                    if magnet_name in found_magnet_lst and magnet_name not in magnet_names:
                        magnet_names.append(magnet_name)
                for magnet_name in found_magnet_lst:
                    if magnet_name not in magnet_names and magnet_name not in magnet_names:
                        magnet_names.append(magnet_name)

            _find_magnets("magnet_inputs", parsed_factory_kwargs["input_names"], magnet_inputs, found_magnet_inputs)
            _find_magnets(
                "magnet_in_outputs",
                parsed_factory_kwargs["in_output_names"],
                magnet_in_outputs,
                found_magnet_in_outputs,
            )
            _find_magnets("magnet_params", parsed_factory_kwargs["param_names"], magnet_params, found_magnet_params)

            # Prepare defaults
            for k in remove_defaults:
                found_defaults.pop(k, None)

            def _sort_names(names_name):
                new_names = []
                for k in parsed_factory_kwargs[names_name]:
                    if k not in found_defaults:
                        new_names.append(k)
                for k in parsed_factory_kwargs[names_name]:
                    if k in found_defaults:
                        new_names.append(k)
                parsed_factory_kwargs[names_name] = new_names

            _sort_names("input_names")
            _sort_names("in_output_names")
            _sort_names("param_names")

            # Parse the number of outputs
            if len(parsed_factory_kwargs["output_names"]) == 0:
                lines = expr.split("\n")
                last_line = _clean_expr(lines[-1])
                n_open_brackets = 0
                n_outputs = 1
                for i, s in enumerate(last_line):
                    if s == "," and n_open_brackets == 0:
                        n_outputs += 1
                    elif s in "([{":
                        n_open_brackets += 1
                    elif s in ")]}":
                        n_open_brackets -= 1
                if n_open_brackets != 0:
                    raise ValueError("Couldn't parse the number of outputs: mismatching brackets")
                elif len(parsed_factory_kwargs["output_names"]) == 0:
                    if n_outputs == 1:
                        parsed_factory_kwargs["output_names"] = ["out"]
                    else:
                        parsed_factory_kwargs["output_names"] = ["out%d" % (i + 1) for i in range(n_outputs)]

            factory = cls_or_self(**merge_dicts(parsed_factory_kwargs, factory_kwargs))
            kwargs = merge_dicts(settings, found_defaults, kwargs)
        else:
            func_mapping = merge_dicts(expr_func_config, func_mapping)
            res_func_mapping = merge_dicts(expr_res_func_config, res_func_mapping)

            var_names = get_expr_var_names(expr)

            factory = cls_or_self

        if return_clean_expr:
            # For debugging purposes
            return expr

        input_names = factory.input_names
        in_output_names = factory.in_output_names
        param_names = factory.param_names

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.Param, ...],
            **_kwargs,
        ) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            import vectorbtpro as vbt

            input_context = dict(np=np, pd=pd, vbt=vbt)
            for i, input in enumerate(input_tuple):
                input_context[input_names[i]] = input
            for i, in_output in enumerate(in_output_tuple):
                input_context[in_output_names[i]] = in_output
            for i, param in enumerate(param_tuple):
                input_context[param_names[i]] = param
            merged_context = merge_dicts(input_context, _kwargs)
            context = {}

            # Resolve each variable in the expression
            for var_name in var_names:
                if var_name in context:
                    continue
                if var_name.startswith("__in_"):
                    var = merged_context[var_name[5:]]
                elif var_name.startswith("__inout_"):
                    var = merged_context[var_name[8:]]
                elif var_name.startswith("__p_"):
                    var = merged_context[var_name[4:]]
                elif var_name.startswith("__talib_"):
                    from vectorbtpro.utils.module_ import assert_can_import

                    assert_can_import("talib")
                    import talib
                    from talib import abstract

                    talib_func_name = var_name[8:].upper()
                    talib_ind = cls_or_self.from_talib(talib_func_name)

                    def _talib_func(*__args, _talib_ind=talib_ind, wrapper=_kwargs["wrapper"], **__kwargs) -> tp.Any:
                        inputs = {}
                        other_args = []
                        input_names = _talib_ind.input_names
                        for k in range(len(__args)):
                            if k < len(input_names) and len(inputs) < len(input_names):
                                inputs[input_names[k]] = __args[k]
                            else:
                                other_args.append(__args[k])
                        if len(inputs) < len(input_names):
                            for k in __kwargs:
                                if k in input_names:
                                    inputs[k] = __kwargs.pop(k)

                        bc_inputs = broadcast_arrays(*inputs.values())
                        if bc_inputs[0].ndim == 1:
                            return _talib_ind.apply_func(bc_inputs, (), other_args, wrapper=wrapper, **__kwargs)
                        outputs = []
                        for col in range(bc_inputs[0].shape[1]):
                            col_inputs = [input[:, col] for input in bc_inputs]
                            output = _talib_ind.apply_func(col_inputs, (), other_args, wrapper=wrapper, **__kwargs)
                            outputs.append(output)
                        if isinstance(outputs[0], tuple):  # multiple outputs
                            outputs = list(zip(*outputs))
                            return list(map(column_stack, outputs))
                        return column_stack(outputs)

                    var = _talib_func
                elif var_name in res_func_mapping:
                    var = res_func_mapping[var_name]["func"]
                elif var_name in func_mapping:
                    var = func_mapping[var_name]["func"]
                elif var_name in merged_context:
                    var = merged_context[var_name]
                elif hasattr(np, var_name):
                    var = getattr(np, var_name)
                elif hasattr(generic_nb, var_name):
                    var = getattr(generic_nb, var_name)
                elif hasattr(generic_nb, var_name + "_nb"):
                    var = getattr(generic_nb, var_name + "_nb")
                elif hasattr(vbt, var_name):
                    var = getattr(vbt, var_name)
                else:
                    continue
                try:
                    if callable(var) and "context" in get_func_arg_names(var):
                        var = functools.partial(var, context=merged_context)
                except:
                    pass
                if var_name in res_func_mapping:
                    var = var()
                context[var_name] = var

            # Evaluate the expression using resolved variables as a context
            if use_pd_eval:
                return pd.eval(expr, local_dict=context, **resolve_dict(pd_eval_kwargs))
            return multiline_eval(expr, context=context)

        return factory.with_apply_func(apply_func, pass_packed=True, pass_wrapper=True, **kwargs)

    @classmethod
    def from_wqa101(cls, alpha_idx: tp.Union[str, int], **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class from one of the WorldQuant's 101 alpha expressions.

        See `vectorbtpro.indicators.expr.wqa101_expr_config`.

        !!! note
            Some expressions that utilize cross-sectional operations require columns to be
            a multi-index with a level `sector`, `subindustry`, or `industry`.

        Usage:
            ```pycon
            >>> data = vbt.YFData.fetch(['BTC-USD', 'ETH-USD'])

            >>> WQA1 = vbt.IF.from_wqa101(1)
            >>> wqa1 = WQA1.run(data.get('Close'))
            >>> wqa1.out
            symbol                     BTC-USD  ETH-USD
            Date
            2014-09-17 00:00:00+00:00     0.25     0.25
            2014-09-18 00:00:00+00:00     0.25     0.25
            2014-09-19 00:00:00+00:00     0.25     0.25
            2014-09-20 00:00:00+00:00     0.25     0.25
            2014-09-21 00:00:00+00:00     0.25     0.25
            ...                            ...      ...
            2022-01-21 00:00:00+00:00     0.00     0.50
            2022-01-22 00:00:00+00:00     0.00     0.50
            2022-01-23 00:00:00+00:00     0.25     0.25
            2022-01-24 00:00:00+00:00     0.50     0.00
            2022-01-25 00:00:00+00:00     0.50     0.00

            [2688 rows x 2 columns]
            ```

            * To get help on running the indicator, use `vectorbtpro.utils.formatting.phelp`:

            ```pycon
            >>> vbt.phelp(WQA1.run)
            WQA1.run(
                close,
                short_name='wqa1',
                hide_params=None,
                hide_default=True,
                **kwargs
            ):
                Run `WQA1` indicator.

                * Inputs: `close`
                * Outputs: `out`

                Pass a list of parameter names as `hide_params` to hide their column levels, or True to hide all.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `WQA1.run_pipeline`.
            ```
        """
        if isinstance(alpha_idx, str):
            alpha_idx = int(alpha_idx.upper().replace("WQA", ""))
        return cls.from_expr(
            wqa101_expr_config[alpha_idx],
            factory_kwargs=dict(class_name="WQA%d" % alpha_idx, module_name=__name__ + ".wqa101"),
            **kwargs,
        )


IF = IndicatorFactory
"""Shortcut for `IndicatorFactory`."""

__pdoc__["IF"] = False


def indicator(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.get_indicator`."""
    return IndicatorFactory.get_indicator(*args, **kwargs)


def talib(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_talib`."""
    return IndicatorFactory.from_talib(*args, **kwargs)


def pandas_ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_pandas_ta`."""
    return IndicatorFactory.from_pandas_ta(*args, **kwargs)


def ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_ta`."""
    return IndicatorFactory.from_ta(*args, **kwargs)


def wqa101(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_wqa101`."""
    return IndicatorFactory.from_wqa101(*args, **kwargs)


def technical(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_technical`."""
    return IndicatorFactory.from_technical(*args, **kwargs)


def techcon(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_techcon`."""
    return IndicatorFactory.from_techcon(*args, **kwargs)
