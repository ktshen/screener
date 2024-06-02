# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with parameters."""

import attr
import inspect
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, merge_dicts
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.template import CustomTemplate, substitute_templates
from vectorbtpro.utils.parsing import annotate_args, ann_args_to_args

__all__ = [
    "generate_param_combs",
    "Param",
    "combine_params",
    "parameterized",
]


def to_typed_list(lst: list) -> List:
    """Cast Python list to typed list.

    Direct construction is flawed in Numba 0.52.0.
    See https://github.com/numba/numba/issues/6651"""
    nb_lst = List()
    for elem in lst:
        nb_lst.append(elem)
    return nb_lst


def flatten_param_tuples(param_tuples: tp.Sequence) -> tp.List[tp.List]:
    """Flattens a nested list of iterables using unzipping."""
    param_list = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = list(unzipped)
        if isinstance(unzipped[0], tuple):
            param_list.extend(flatten_param_tuples(unzipped))
        else:
            param_list.append(unzipped)
    return param_list


def generate_param_combs(op_tree: tp.Tuple, depth: int = 0) -> tp.List[tp.List]:
    """Generate arbitrary parameter combinations from the operation tree `op_tree`.

    `op_tree` is a tuple with nested instructions to generate parameters.
    The first element of the tuple must be either the name of a callale from `itertools` or the
    callable itself that takes remaining elements as arguments. If one of the elements is a tuple
    itself and its first argument is a callable, it will be unfolded in the same way as above.

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.generate_param_combs(("product", ("combinations", [0, 1, 2, 3], 2), [4, 5]))
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
         [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
         [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]]

        >>> vbt.generate_param_combs(("product", (zip, [0, 1, 2, 3], [4, 5, 6, 7]), [8, 9]))
        [[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 7, 7], [8, 9, 8, 9, 8, 9, 8, 9]]
        ```
    """
    checks.assert_instance_of(op_tree, tuple)
    checks.assert_instance_of(op_tree[0], (Callable, str))
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple) and isinstance(elem[0], (Callable, str)):
            new_op_tree += (generate_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    if isinstance(new_op_tree[0], Callable):
        out = list(new_op_tree[0](*new_op_tree[1:]))
    else:
        import itertools

        out = list(getattr(itertools, new_op_tree[0])(*new_op_tree[1:]))
    if depth == 0:
        # do something
        return flatten_param_tuples(out)
    return out


def broadcast_params(param_list: tp.Sequence[tp.Params], to_n: tp.Optional[int] = None) -> tp.List[tp.List]:
    """Broadcast parameters in `param_list`."""
    if to_n is None:
        to_n = max(list(map(len, param_list)))
    new_param_list = []
    for i in range(len(param_list)):
        params = param_list[i]
        if len(params) in [1, to_n]:
            if len(params) < to_n:
                new_param_list.append([p for _ in range(to_n) for p in params])
            else:
                new_param_list.append(list(params))
        else:
            raise ValueError(f"Parameters at index {i} have length {len(params)} that cannot be broadcast to {to_n}")
    return new_param_list


def create_param_product(param_list: tp.Sequence[tp.Params]) -> tp.List[tp.List]:
    """Make Cartesian product out of all params in `param_list`."""
    import itertools

    return list(map(list, zip(*itertools.product(*param_list))))


def params_to_list(params: tp.Params, is_tuple: bool, is_array_like: bool) -> list:
    """Cast parameters to a list."""
    check_against = [list, List]
    if not is_tuple:
        check_against.append(tuple)
    if not is_array_like:
        check_against.append(np.ndarray)
    if isinstance(params, tuple(check_against)):
        new_params = list(params)
    else:
        new_params = [params]
    return new_params


@attr.s(frozen=True)
class Param:
    """Class that represents a parameter."""

    value: tp.Union[tp.Param, tp.Dict[tp.Hashable, tp.Param], tp.Sequence[tp.Param]] = attr.ib()
    """One or more parameter values."""

    is_tuple: bool = attr.ib(default=False)
    """Whether `Param.value` is a tuple.
    
    If so, providing a tuple will be considered as a single value."""

    is_array_like: bool = attr.ib(default=False)
    """Whether `Param.value` is array-like.
    
    If so, providing a NumPy array will be considered as a single value."""

    map_template: tp.Optional[CustomTemplate] = attr.ib(default=None)
    """Template to map `Param.value` before building parameter combinations."""

    random_subset: tp.Union[None, int, float] = attr.ib(default=None)
    """Random subset of values to select."""

    level: tp.Optional[int] = attr.ib(default=None)
    """Level of the product the parameter takes part in.

    Parameters with the same level are stacked together, while parameters with different levels
    are combined as usual.
    
    Parameters are processed based on their level: a lower-level parameter is processed before 
    (and thus displayed above) a higher-level parameter. If two parameters share the same level, 
    they are processed in the order they were passed to the function.
    
    Levels must come in a strict order starting with 0 and without gaps. If any of the parameters
    have a level specified, all parameters must specify their level."""

    condition: tp.Optional[str] = attr.ib(default=None)
    """Keep a parameter combination only if the condition is met.
    
    Condition should be an expression where `x` denotes this parameter and any other variable
    denotes the name of other parameter(s)."""

    context: tp.KwargsLike = attr.ib(default=None)
    """Context used in evaluation of `Param.condition`."""

    keys: tp.Optional[tp.IndexLike] = attr.ib(default=None)
    """Keys acting as an index level.

    If None, converts `Param.value` to an index using 
    `vectorbtpro.base.indexes.index_from_values`."""

    hide: bool = attr.ib(default=False)
    """Whether to hide the parameter from the parameter index.
    
    At least one parameter must be displayed per level."""

    name: tp.Optional[tp.Hashable] = attr.ib(default=None)
    """Name of the parameter.
    
    If None, defaults to the name of the index in `Param.keys`, or to the key in 
    `param_dct` passed to `combine_params`."""


def combine_params(
    param_dct: tp.Dict[tp.Hashable, Param],
    random_subset: tp.Optional[int] = None,
    seed: tp.Optional[int] = None,
    index_stack_kwargs: tp.KwargsLike = None,
    name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
    build_index: bool = True,
) -> tp.Union[dict, tp.Tuple[dict, pd.Index]]:
    """Combine a dictionary with parameters of the type `Param`.

    Returns a dictionary with combined parameters and an index."""
    from vectorbtpro._settings import settings
    from vectorbtpro.base import indexes

    params_cfg = settings["params"]

    if random_subset is None:
        random_subset = params_cfg["random_subset"]
    if seed is None:
        seed = params_cfg["seed"]
    rng = np.random.default_rng(seed=seed)
    index_stack_kwargs = merge_dicts(params_cfg["index_stack_kwargs"], index_stack_kwargs)
    if name_tuple_to_str is None:
        name_tuple_to_str = params_cfg["name_tuple_to_str"]
    if index_stack_kwargs is None:
        index_stack_kwargs = {}

    # Build a product
    level_values = defaultdict(OrderedDict)
    product_indexes = OrderedDict()
    level_seen = False
    curr_idx = 0
    max_idx = 0
    conditions = {}
    contexts = {}
    names = {}

    for k, p in param_dct.items():
        if p.condition is not None:
            conditions[k] = p.condition
            if p.context is not None:
                contexts[k] = p.context
            else:
                contexts[k] = {}
        if p.level is None:
            if level_seen:
                raise ValueError("Please provide level for all product parameters")
            level = curr_idx
        else:
            if curr_idx > 0 and not level_seen:
                raise ValueError("Please provide level for all product parameters")
            level_seen = True
            level = p.level
        if level > max_idx:
            max_idx = level

        keys_name = None
        p_name = p.name
        sr_name = None
        index_name = None

        if not p.hide:
            keys = p.keys
            if keys is not None:
                if not isinstance(keys, pd.Index):
                    keys = pd.Index(keys)
                keys_name = keys.name
        else:
            keys = None

        value = p.value
        if isinstance(value, dict):
            if not p.hide and keys is None:
                keys = pd.Index(value.keys())
            value = list(value.values())
        elif isinstance(value, pd.Index):
            if not p.hide and keys is None:
                keys = value
            index_name = value.name
            value = value.tolist()
        elif isinstance(value, pd.Series):
            if not checks.is_default_index(value.index):
                if not p.hide and keys is None:
                    keys = value.index
                index_name = value.index.name
            sr_name = value.name
            value = value.values.tolist()
        values = params_to_list(value, is_tuple=p.is_tuple, is_array_like=p.is_array_like)

        if not p.hide:
            if keys_name is None:
                if p_name is not None:
                    keys_name = p_name
                elif sr_name is not None:
                    keys_name = sr_name
                elif index_name is not None:
                    keys_name = index_name
                else:
                    keys_name = k
            if keys is None:
                keys = indexes.index_from_values(values, name=keys_name)
            else:
                keys = keys.rename(keys_name)

        if p.random_subset is not None:
            if checks.is_float(p.random_subset):
                _random_subset = int(p.random_subset * len(values))
            else:
                _random_subset = p.random_subset
            random_indices = np.sort(rng.permutation(np.arange(len(values)))[:_random_subset])
        else:
            random_indices = None
        if random_indices is not None:
            values = [values[i] for i in random_indices]
            if keys is not None:
                keys = keys[random_indices]

        if p.map_template is not None:
            param_context = merge_dicts(
                dict(
                    param=p,
                    values=values,
                    keys=keys,
                    random_indices=random_indices,
                ),
                p.context,
            )
            values = p.map_template.substitute(param_context, sub_id="map_template")

        level_values[level][k] = values
        product_indexes[k] = keys
        names[k] = keys_name
        curr_idx += 1

    # Build an operation tree and parameter index
    op_tree_operands = []
    param_keys = []
    new_product_indexes = []
    for level in range(max_idx + 1):
        if level not in level_values:
            raise ValueError("Levels must come in a strict order starting with 0 and without gaps")
        for k in level_values[level].keys():
            param_keys.append(k)

        # Broadcast parameter arrays
        param_lists = tuple(level_values[level].values())
        if len(param_lists) > 1:
            op_tree_operands.append((zip, *broadcast_params(param_lists)))
        else:
            op_tree_operands.append(param_lists[0])

        # Stack or combine parameter indexes together
        if build_index:
            levels = []
            for k in level_values[level].keys():
                if product_indexes[k] is not None:
                    levels.append(product_indexes[k])
            if len(levels) > 1:
                _param_index = indexes.stack_indexes(levels, **index_stack_kwargs)
            elif len(levels) == 1:
                _param_index = levels[0]
            else:
                raise ValueError("At least one parameter must be displayed per level")
            new_product_indexes.append(_param_index)
    if build_index:
        if len(new_product_indexes) > 1:
            param_index = indexes.combine_indexes(new_product_indexes, **index_stack_kwargs)
        else:
            param_index = new_product_indexes[0]
    else:
        param_index = None

    # Generate parameter combinations using the operation tree
    if len(op_tree_operands) > 1:
        param_product = dict(zip(param_keys, generate_param_combs(("product", *op_tree_operands))))
    elif isinstance(op_tree_operands[0], tuple):
        param_product = dict(zip(param_keys, generate_param_combs(op_tree_operands[0])))
    else:
        param_product = dict(zip(param_keys, op_tree_operands))
    ncombs = len(list(param_product.values())[0])

    # Filter by condition
    if len(conditions) > 0:
        indices = np.arange(ncombs)
        pre_random_subset = random_subset is not None and not checks.is_float(random_subset)
        if pre_random_subset:
            indices = rng.permutation(indices)
        keep_indices = []
        condition_funcs = {
            k: eval(
                f"lambda {', '.join({'x'} | set(names.keys()) | set(names.values()) | set(contexts[k].keys()))}: {expr}"
            )
            for k, expr in conditions.items()
        }
        any_discarded = False
        for i in indices:
            param_values = {}
            for k in param_product:
                param_values[k] = param_product[k][i]
                param_values[names[k]] = param_product[k][i]
            conditions_met = True
            for k, condition_func in condition_funcs.items():
                param_context = {"x": param_values[k], **param_values, **contexts[k]}
                if not condition_func(**param_context):
                    conditions_met = False
                    break
            if conditions_met:
                keep_indices.append(i)
                if pre_random_subset:
                    if len(keep_indices) == random_subset:
                        break
            else:
                any_discarded = True
        if any_discarded:
            if len(keep_indices) == 0:
                raise ValueError("No parameters left")
            if pre_random_subset:
                keep_indices = np.sort(keep_indices)
            param_product = {k: [v[i] for i in keep_indices] for k, v in param_product.items()}
            ncombs = len(keep_indices)
            if build_index:
                param_index = param_index[keep_indices]
    else:
        pre_random_subset = False

    # Select a random subset
    if random_subset is not None and not pre_random_subset:
        if checks.is_float(random_subset):
            random_subset = int(random_subset * ncombs)
        random_indices = np.sort(rng.permutation(np.arange(ncombs))[:random_subset])
        param_product = {k: [v[i] for i in random_indices] for k, v in param_product.items()}
        if build_index:
            param_index = param_index[random_indices]

    # Stringify index names
    if build_index:
        if isinstance(name_tuple_to_str, bool):
            if name_tuple_to_str:
                name_tuple_to_str = lambda name_tuple: "_".join(map(lambda x: str(x).strip().lower(), name_tuple))
            else:
                name_tuple_to_str = None
        if name_tuple_to_str is not None:
            found_tuple = False
            new_names = []
            for name in param_index.names:
                if isinstance(name, tuple):
                    name = name_tuple_to_str(name)
                    found_tuple = True
                new_names.append(name)
            if found_tuple:
                if isinstance(param_index, pd.MultiIndex):
                    param_index.rename(new_names, inplace=True)
                else:
                    param_index.rename(new_names[0], inplace=True)
    if build_index:
        return param_product, param_index
    return param_product


def find_params_in_obj(
    obj: tp.Any,
    key: tp.Optional[tp.Hashable] = None,
    search_except_types: tp.Optional[tp.Sequence[type]] = None,
    search_max_len: tp.Optional[int] = None,
    search_max_depth: tp.Optional[int] = None,
    _depth: int = 0,
) -> dict:
    """Find values wrapped with `Param` in a recursive manner.

    If a value is a dictionary or a tuple, applies `find_params_in_obj` on each element,
    unless its length exceeds `search_max_len` or the current depth exceeds `search_max_depth`."""
    from vectorbtpro._settings import settings

    params_cfg = settings["params"]

    if search_except_types is None:
        search_except_types = params_cfg["search_except_types"]
    if search_max_len is None:
        search_max_len = params_cfg["search_max_len"]
    if search_max_depth is None:
        search_max_depth = params_cfg["search_max_depth"]

    if isinstance(obj, Param):
        return {key: obj}
    if search_max_depth is None or _depth < search_max_depth:
        if search_except_types is not None and checks.is_instance_of(obj, search_except_types):
            return obj
        if isinstance(obj, dict):
            if search_max_len is None or len(obj) <= search_max_len:
                found_dct = {}
                for k, v in obj.items():
                    new_key = k if key is None else (*key, k) if isinstance(key, tuple) else (key, k)
                    found_dct.update(
                        find_params_in_obj(
                            v,
                            key=new_key,
                            search_except_types=search_except_types,
                            search_max_len=search_max_len,
                            search_max_depth=search_max_depth,
                            _depth=_depth + 1,
                        )
                    )
                return found_dct
        if isinstance(obj, (tuple, list)):
            if search_max_len is None or len(obj) <= search_max_len:
                found_dct = {}
                for i in range(len(obj)):
                    new_key = i if key is None else (*key, i) if isinstance(key, tuple) else (key, i)
                    found_dct.update(
                        find_params_in_obj(
                            obj[i],
                            key=new_key,
                            search_except_types=search_except_types,
                            search_max_len=search_max_len,
                            search_max_depth=search_max_depth,
                            _depth=_depth + 1,
                        )
                    )
                return found_dct
    return {}


def replace_param_set_in_obj(obj: tp.Any, param_dct: dict, key: tp.Optional[tp.Hashable] = None) -> tp.Any:
    """Replace a single parameter set in an object in a recursive manner."""
    if len(param_dct) == 0:
        return obj
    if key in param_dct:
        return param_dct[key]
    if isinstance(obj, dict):
        new_obj = {}
        for k in obj:
            if k in param_dct:
                new_obj[k] = param_dct[k]
                del param_dct[k]
            else:
                replaced = False
                for k2 in param_dct:
                    if isinstance(k2, tuple) and k2[0] == k:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        param_dct[new_k2] = param_dct[k2]
                        del param_dct[k2]
                        new_key = k if key is None else (*key, k) if isinstance(key, tuple) else (key, k)
                        new_obj[k] = replace_param_set_in_obj(obj[k], param_dct, new_key)
                        replaced = True
                        break
                if not replaced:
                    new_obj[k] = obj[k]
        return new_obj
    if isinstance(obj, (tuple, list)):
        new_obj = []
        for i in range(len(obj)):
            if i in param_dct:
                new_obj.append(param_dct[i])
                del param_dct[i]
            else:
                replaced = False
                for i2 in param_dct:
                    if isinstance(i2, tuple) and i2[0] == i:
                        new_i2 = i2[1:] if len(i2) > 2 else i2[1]
                        param_dct[new_i2] = param_dct[i2]
                        del param_dct[i2]
                        new_key = i if key is None else (*key, i) if isinstance(key, tuple) else (key, i)
                        new_obj.append(replace_param_set_in_obj(obj[i], param_dct, new_key))
                        replaced = True
                        break
                if not replaced:
                    new_obj.append(obj[i])
        if isinstance(obj, tuple):
            return tuple(new_obj)
        return new_obj
    return obj


def param_product_to_objs(obj: tp.Any, param_product: dict) -> tp.List[dict]:
    """Resolve parameter product into a list of objects based on the original object."""
    if len(param_product) == 0:
        return []
    param_product_items = list(param_product.items())
    n_values = len(param_product_items[0][1])
    new_objs = []
    for i in range(n_values):
        param_dct = {k: v[i] for k, v in param_product.items()}
        new_objs.append(replace_param_set_in_obj(obj, param_dct))
    return new_objs


def parameterized(
    *args,
    search_except_types: tp.Optional[tp.Sequence[type]] = None,
    search_max_len: tp.Optional[int] = None,
    search_max_depth: tp.Optional[int] = None,
    skip_single_param: tp.Optional[bool] = None,
    template_context: tp.Optional[tp.Mapping] = None,
    random_subset: tp.Optional[int] = None,
    seed: tp.Optional[int] = None,
    index_stack_kwargs: tp.KwargsLike = None,
    name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
    merge_func: tp.Union[None, str, tuple, tp.Callable] = None,
    merge_kwargs: tp.KwargsLike = None,
    return_meta: bool = False,
    use_meta: tp.KwargsLike = None,
    selection: tp.Union[None, tp.MaybeIterable[tp.Hashable]] = None,
    forward_kwargs_as: tp.KwargsLike = None,
    execute_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that parameterizes a function. Engine-agnostic.
    Returns a new function with the same signature as the passed one.

    Does the following:

    1. Searches for values wrapped with the class `Param` in any nested dicts and tuples using `find_params_in_obj`
    2. Uses `combine_params` to build parameter combinations
    3. Maps parameter combinations to configs using `param_product_to_objs`
    4. Generates and resolves parameter configs by combining combinations from the step above with
    `param_configs` that is optionally passed by the user. User-defined `param_configs` have more priority.
    5. Extracts arguments and keyword arguments from each parameter config and substitutes any templates (lazily)
    6. If `return_meta` is True, returns all the objects generated above as a dictionary
    7. If `selection` is not None, substitutes it as a template, translates it into indices that
    can be mapped to `param_index`, and selects them from all the objects generated above
    8. Passes each set of the function and its arguments to `vectorbtpro.utils.execution.execute` for execution
    9. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`

    Argument `param_configs` will be added as an extra argument to the function's signature.
    It accepts either a list of dictionaries with arguments named by their names in the signature,
    or a dictionary of dictionaries, where keys are config names. If a list is passed, each dictionary
    can also contain the key `_name` to give the config a name. Variable arguments can be passed
    either in the rolled (`args=(...), kwargs={...}`) or unrolled (`args_0=..., args_1=..., some_kwarg=...`) format.

    !!! important
        Defining a parameter and listing the same argument in `param_configs` will prioritize
        the config over the parameter, even though the parameter will still be visible in the final columns.
        There are no checks implemented to raise an error when this happens!

    Any template in both `execute_kwargs` and `merge_kwargs` will be substituted. You can use
    the keys `param_configs`, `param_index`, all keys in `template_context`, and all arguments as found
    in the signature of the function.

    If `skip_single_param` is True, won't use the execution engine, but will execute and
    return the result right away.

    Argument `merge_func` also accepts one of the following strings:

    * 'concat': uses `vectorbtpro.base.merging.concat_merge`
    * 'row_stack': uses `vectorbtpro.base.merging.row_stack_merge`
    * 'column_stack': uses `vectorbtpro.base.merging.column_stack_merge`

    When defining a custom merging function, make sure to use `param_index` (via templates) to build the final
    index/column hierarchy.

    Keyword arguments `**execute_kwargs` are passed directly to `vectorbtpro.utils.execution.execute`.

    Usage:
        * No parameters, no parameter configs:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd

        >>> @vbt.parameterized(merge_func="column_stack")
        ... def my_ma(sr_or_df, window, wtype="simple", minp=0, adjust=False):
        ...     return sr_or_df.vbt.ma(window, wtype=wtype, minp=minp, adjust=adjust)

        >>> sr = pd.Series([1, 2, 3, 4, 3, 2, 1])
        >>> my_ma(sr, 3)
        0    1.000000
        1    1.500000
        2    2.000000
        3    3.000000
        4    3.333333
        5    3.000000
        6    2.000000
        dtype: float64
        ```

        * One parameter, no parameter configs:

        ```pycon
        >>> my_ma(sr, vbt.Param([3, 4, 5]))
        window         3    4    5
        0       1.000000  1.0  1.0
        1       1.500000  1.5  1.5
        2       2.000000  2.0  2.0
        3       3.000000  2.5  2.5
        4       3.333333  3.0  2.6
        5       3.000000  3.0  2.8
        6       2.000000  2.5  2.6
        ```

        * Product of two parameters, no parameter configs:

        ```pycon
        >>> my_ma(
        ...     sr,
        ...     vbt.Param([3, 4, 5]),
        ...     wtype=vbt.Param(["simple", "exp"])
        ... )
        window         3                4                5
        wtype     simple       exp simple       exp simple       exp
        0       1.000000  1.000000    1.0  1.000000    1.0  1.000000
        1       1.500000  1.500000    1.5  1.400000    1.5  1.333333
        2       2.000000  2.250000    2.0  2.040000    2.0  1.888889
        3       3.000000  3.125000    2.5  2.824000    2.5  2.592593
        4       3.333333  3.062500    3.0  2.894400    2.6  2.728395
        5       3.000000  2.531250    3.0  2.536640    2.8  2.485597
        6       2.000000  1.765625    2.5  1.921984    2.6  1.990398
        ```

        * No parameters, one partial parameter config:

        ```pycon
        >>> my_ma(sr, param_configs=[dict(window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        * No parameters, one full parameter config:

        ```pycon
        >>> my_ma(param_configs=[dict(sr_or_df=sr, window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        * No parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, window=2),
        ...     dict(sr_or_df=sr - 1, window=3)
        ... ], minp=None)
        param_config    0         1
        0             NaN       NaN
        1             2.5       NaN
        2             3.5  1.000000
        3             4.5  2.000000
        4             4.5  2.333333
        5             3.5  2.000000
        6             2.5  1.000000
        ```

        * Multiple parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, minp=0),
        ...     dict(sr_or_df=sr - 1, minp=None)
        ... ], window=vbt.Param([2, 3]))
        window          2              3
        param_config    0    1         0         1
        0             2.0  NaN  2.000000       NaN
        1             2.5  0.5  2.500000       NaN
        2             3.5  1.5  3.000000  1.000000
        3             4.5  2.5  4.000000  2.000000
        4             4.5  2.5  4.333333  2.333333
        5             3.5  1.5  4.000000  2.000000
        6             2.5  0.5  3.000000  1.000000
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            search_except_types = kwargs.pop("_search_except_types", wrapper.options["search_except_types"])
            if search_except_types is None:
                search_except_types = params_cfg["search_except_types"]
            search_max_len = kwargs.pop("_search_max_len", wrapper.options["search_max_len"])
            if search_max_len is None:
                search_max_len = params_cfg["search_max_len"]
            search_max_depth = kwargs.pop("_search_max_depth", wrapper.options["search_max_depth"])
            if search_max_depth is None:
                search_max_depth = params_cfg["search_max_depth"]
            skip_single_param = kwargs.pop("_skip_single_param", wrapper.options["skip_single_param"])
            if skip_single_param is None:
                skip_single_param = params_cfg["skip_single_param"]
            template_context = merge_dicts(
                params_cfg["template_context"], wrapper.options["template_context"], kwargs.pop("_template_context", {})
            )
            random_subset = kwargs.pop("_random_subset", wrapper.options["random_subset"])
            if random_subset is None:
                random_subset = params_cfg["random_subset"]
            seed = kwargs.pop("_seed", wrapper.options["seed"])
            if seed is None:
                seed = params_cfg["seed"]
            index_stack_kwargs = merge_dicts(
                params_cfg["index_stack_kwargs"],
                wrapper.options["index_stack_kwargs"],
                kwargs.pop("_index_stack_kwargs", {}),
            )
            name_tuple_to_str = kwargs.pop("_name_tuple_to_str", wrapper.options["name_tuple_to_str"])
            if name_tuple_to_str is None:
                name_tuple_to_str = params_cfg["name_tuple_to_str"]
            merge_func = kwargs.pop("_merge_func", wrapper.options["merge_func"])
            merge_kwargs = merge_dicts(wrapper.options["merge_kwargs"], kwargs.pop("_merge_kwargs", {}))
            return_meta = kwargs.pop("_return_meta", wrapper.options["return_meta"])
            use_meta = kwargs.pop("_use_meta", wrapper.options["use_meta"])
            selection = kwargs.pop("_selection", wrapper.options["selection"])
            execute_kwargs = merge_dicts(
                params_cfg["execute_kwargs"], wrapper.options["execute_kwargs"], kwargs.pop("_execute_kwargs", {})
            )
            forward_kwargs_as = merge_dicts(wrapper.options["forward_kwargs_as"], kwargs.pop("_forward_kwargs_as", {}))
            if len(forward_kwargs_as) > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in forward_kwargs_as:
                        new_kwargs[forward_kwargs_as.pop(k)] = v
                    else:
                        new_kwargs[k] = v
                kwargs = new_kwargs
            if len(forward_kwargs_as) > 0:
                for k, v in forward_kwargs_as.items():
                    kwargs[v] = locals()[k]

            param_configs = kwargs.pop("param_configs", None)
            if param_configs is None:
                param_configs = []

            if use_meta is None:
                # Annotate arguments
                ann_args = annotate_args(func, args, kwargs, allow_partial=True)
                var_args_name = None
                var_kwargs_name = None
                for k, v in ann_args.items():
                    if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                        var_args_name = k
                    if v["kind"] == inspect.Parameter.VAR_KEYWORD:
                        var_kwargs_name = k

                # Unroll parameter configs
                pc_names = []
                pc_names_none = True
                n_param_configs = 0
                if isinstance(param_configs, dict):
                    new_param_configs = []
                    for k, v in param_configs.items():
                        v = dict(v)
                        v["_name"] = k
                        new_param_configs.append(v)
                    param_configs = new_param_configs
                else:
                    param_configs = list(param_configs)
                for i, param_config in enumerate(param_configs):
                    param_config = dict(param_config)
                    if var_args_name is not None and var_args_name in param_config:
                        for k, arg in enumerate(param_config.pop(var_args_name)):
                            param_config[f"{var_args_name}_{k}"] = arg
                    if var_kwargs_name is not None and var_kwargs_name in param_config:
                        for k, v in param_config.pop(var_kwargs_name).items():
                            param_config[k] = v
                    if "_name" in param_config and param_config["_name"] is not None:
                        pc_names.append(param_config.pop("_name"))
                        pc_names_none = False
                    else:
                        pc_names.append(n_param_configs)
                    param_configs[i] = param_config
                    n_param_configs += 1

                # Combine parameters
                paramable_kwargs = {}
                for k, v in ann_args.items():
                    if "value" in v:
                        if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                            for i, arg in enumerate(v["value"]):
                                paramable_kwargs[f"{var_args_name}_{i}"] = arg
                        elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                            for k2, v2 in v["value"].items():
                                paramable_kwargs[k2] = v2
                        else:
                            paramable_kwargs[k] = v["value"]
                param_dct = find_params_in_obj(
                    paramable_kwargs,
                    search_except_types=search_except_types,
                    search_max_len=search_max_len,
                    search_max_depth=search_max_depth,
                )
                param_columns = None
                if len(param_dct) > 0:
                    param_product, param_columns = combine_params(
                        param_dct,
                        random_subset=random_subset,
                        seed=seed,
                        index_stack_kwargs=index_stack_kwargs,
                        name_tuple_to_str=name_tuple_to_str,
                    )
                    product_param_configs = param_product_to_objs(paramable_kwargs, param_product)
                    if len(param_configs) == 0:
                        param_configs = product_param_configs
                    else:
                        new_param_configs = []
                        for i in range(len(product_param_configs)):
                            for param_config in param_configs:
                                new_param_config = merge_dicts(product_param_configs[i], param_config)
                                new_param_configs.append(new_param_config)
                        param_configs = new_param_configs

                # Build param index
                n_config_params = len(pc_names)
                if param_columns is not None:
                    if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                        param_index = param_columns
                    else:
                        from vectorbtpro.base.indexes import combine_indexes

                        param_index = combine_indexes(
                            (
                                param_columns,
                                pd.Index(pc_names, name="param_config"),
                            ),
                            **index_stack_kwargs,
                        )
                else:
                    if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                        param_index = pd.Index([0], name="param_config")
                    else:
                        param_index = pd.Index(pc_names, name="param_config")

                # Create parameter config from arguments if empty
                if len(param_configs) == 0:
                    single_param = True
                    param_configs.append(dict())
                else:
                    single_param = False
                template_context["single_param"] = single_param

                # Roll parameter configs
                new_param_configs = []
                for param_config in param_configs:
                    new_param_config = merge_dicts(paramable_kwargs, param_config)
                    if var_args_name is not None:
                        _args = ()
                        while True:
                            if f"{var_args_name}_{len(_args)}" in new_param_config:
                                _args += (new_param_config.pop(f"{var_args_name}_{len(_args)}"),)
                            else:
                                break
                        new_param_config[var_args_name] = _args
                    if var_kwargs_name is not None:
                        new_param_config[var_kwargs_name] = {}
                        for k in list(new_param_config.keys()):
                            if k not in ann_args:
                                new_param_config[var_kwargs_name][k] = new_param_config.pop(k)
                    new_param_configs.append(new_param_config)
                param_configs = new_param_configs
                template_context["param_configs"] = param_configs
                template_context["param_index"] = param_index

                # Prepare function and arguments

                def _prepare_args(
                    _ann_args=ann_args,
                    _param_configs=param_configs,
                    _template_context=template_context,
                ):
                    for p, param_config in enumerate(_param_configs):
                        __template_context = dict(_template_context)
                        __template_context["config_idx"] = p
                        __ann_args = dict()
                        for k, v in _ann_args.items():
                            v = dict(v)
                            v["value"] = param_config[k]
                            __ann_args[k] = v
                        _args, _kwargs = ann_args_to_args(__ann_args)
                        _args = substitute_templates(_args, __template_context, sub_id="args")
                        _kwargs = substitute_templates(_kwargs, __template_context, sub_id="kwargs")
                        yield func, _args, _kwargs

                funcs_args = _prepare_args()
                template_context["funcs_args"] = funcs_args
                use_meta = dict(
                    single_param=single_param,
                    param_configs=param_configs,
                    param_index=param_index,
                    funcs_args=funcs_args,
                )
            else:
                template_context["single_param"] = use_meta["single_param"]
                template_context["param_configs"] = use_meta["param_configs"]
                template_context["param_index"] = use_meta["param_index"]
                template_context["funcs_args"] = use_meta["funcs_args"]
            del single_param
            del param_configs
            del param_index
            del funcs_args
            if return_meta:
                return use_meta

            if selection is not None:
                selection = substitute_templates(selection, template_context, sub_id="selection")
                found_param = False
                if checks.is_hashable(selection):
                    if checks.is_int(selection):
                        selection = {selection}
                        found_param = True
                        template_context["single_param"] = True
                    elif selection in template_context["param_index"]:
                        selection = {template_context["param_index"].get_loc(selection)}
                        found_param = True
                        template_context["single_param"] = True
                if not found_param:
                    if checks.is_iterable(selection):
                        new_selection = set()
                        for s in selection:
                            if checks.is_int(s):
                                new_selection.add(s)
                            elif s in template_context["param_index"]:
                                new_selection.add(template_context["param_index"].get_loc(s))
                            else:
                                raise ValueError(f"Selection {selection} couldn't be matched with parameter index")
                        selection = new_selection
                    else:
                        raise ValueError(f"Selection {selection} couldn't be matched with parameter index")
                template_context["param_index"] = template_context["param_index"][list(selection)]
                new_param_configs = []
                _selection = selection.copy()
                for i, x in enumerate(template_context["param_configs"]):
                    if i in _selection:
                        new_param_configs.append(x)
                        _selection.remove(i)
                        if len(_selection) == 0:
                            break
                template_context["param_configs"] = new_param_configs
                new_funcs_args = []
                _selection = selection.copy()
                for i, x in enumerate(template_context["funcs_args"]):
                    if i in _selection:
                        new_funcs_args.append(x)
                        _selection.remove(i)
                        if len(_selection) == 0:
                            break
                template_context["funcs_args"] = new_funcs_args

            if skip_single_param and template_context["single_param"]:
                funcs_args = list(template_context["funcs_args"])
                return funcs_args[0][0](*funcs_args[0][1], **funcs_args[0][2])

            # Execute function on each parameter combination
            execute_kwargs = substitute_templates(execute_kwargs, template_context, sub_id="execute_kwargs")
            results = execute(
                template_context["funcs_args"],
                n_calls=len(template_context["param_configs"]),
                **execute_kwargs,
            )

            # Merge the results
            if merge_func is not None:
                if isinstance(merge_func, (str, tuple)):
                    from vectorbtpro.base.merging import resolve_merge_func

                    merge_func = resolve_merge_func(merge_func)
                    merge_kwargs = {**dict(keys=template_context["param_index"]), **merge_kwargs}
                merge_kwargs = substitute_templates(merge_kwargs, template_context, sub_id="merge_kwargs")
                return merge_func(results, **merge_kwargs)
            return results

        wrapper.is_parameterized = True
        wrapper.options = Config(
            dict(
                search_except_types=search_except_types,
                search_max_len=search_max_len,
                search_max_depth=search_max_depth,
                skip_single_param=skip_single_param,
                template_context=template_context,
                random_subset=random_subset,
                seed=seed,
                index_stack_kwargs=index_stack_kwargs,
                name_tuple_to_str=name_tuple_to_str,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                return_meta=return_meta,
                use_meta=use_meta,
                selection=selection,
                forward_kwargs_as=forward_kwargs_as,
                execute_kwargs=merge_dicts(kwargs, execute_kwargs),
            ),
            options_=dict(
                frozen_keys=True,
                as_attrs=True,
            ),
        )
        signature = inspect.signature(wrapper)
        lists_var_kwargs = False
        for k, v in signature.parameters.items():
            if v.kind == v.VAR_KEYWORD:
                lists_var_kwargs = True
                break
        if not lists_var_kwargs:
            var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
            wrapper.__signature__ = signature.replace(parameters=new_parameters)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
