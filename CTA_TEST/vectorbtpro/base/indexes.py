# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions for working with indexes: index and columns.

They perform operations on index objects, such as stacking, combining, and cleansing MultiIndex levels.

!!! note
    "Index" in pandas context is referred to both index and columns."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.utils import checks

__all__ = [
    "repeat_index",
    "tile_index",
    "stack_indexes",
    "combine_indexes",
]


def to_any_index(index_like: tp.IndexLike) -> tp.Index:
    """Convert any index-like object to an index.

    Index objects are kept as-is."""
    if checks.is_np_array(index_like) and index_like.ndim == 0:
        index_like = index_like[None]
    if not checks.is_index(index_like):
        return pd.Index(index_like)
    return index_like


def get_index(arg: tp.SeriesFrame, axis: int) -> tp.Index:
    """Get index of `arg` by `axis`."""
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    checks.assert_in(axis, (0, 1))

    if axis == 0:
        return arg.index
    else:
        if checks.is_series(arg):
            if arg.name is not None:
                return pd.Index([arg.name])
            return pd.Index([0])  # same as how pandas does it
        else:
            return arg.columns


def index_from_values(
    values: tp.Sequence,
    single_value: bool = False,
    name: tp.Optional[tp.Hashable] = None,
) -> tp.Index:
    """Create a new `pd.Index` with `name` by parsing an iterable `values`.

    Each in `values` will correspond to an element in the new index."""
    scalar_types = (int, float, complex, str, bool, datetime, timedelta, np.generic)
    type_id_number = {}
    value_names = []
    if len(values) == 1:
        single_value = True
    for i in range(len(values)):
        if i > 0 and single_value:
            break
        v = values[i]
        if v is None or isinstance(v, scalar_types):
            value_names.append(v)
        elif isinstance(v, np.ndarray):
            all_same = False
            if np.issubdtype(v.dtype, np.floating):
                if np.isclose(v, v.item(0), equal_nan=True).all():
                    all_same = True
            elif v.dtype.names is not None:
                all_same = False
            else:
                if np.equal(v, v.item(0)).all():
                    all_same = True
            if all_same:
                value_names.append(v.item(0))
            else:
                if single_value:
                    value_names.append("array")
                else:
                    if "array" not in type_id_number:
                        type_id_number["array"] = {}
                    if id(v) not in type_id_number["array"]:
                        type_id_number["array"][id(v)] = len(type_id_number["array"])
                    value_names.append("array_%d" % (type_id_number["array"][id(v)]))
        else:
            type_name = str(type(v).__name__)
            if single_value:
                value_names.append("%s" % type_name)
            else:
                if type_name not in type_id_number:
                    type_id_number[type_name] = {}
                if id(v) not in type_id_number[type_name]:
                    type_id_number[type_name][id(v)] = len(type_id_number[type_name])
                value_names.append("%s_%d" % (type_name, type_id_number[type_name][id(v)]))
    if single_value and len(values) > 1:
        value_names *= len(values)
    return pd.Index(value_names, name=name)


def repeat_index(index: tp.IndexLike, n: int, ignore_ranges: tp.Optional[bool] = None) -> tp.Index:
    """Repeat each element in `index` `n` times.

    Set `ignore_ranges` to True to ignore indexes of type `pd.RangeIndex`."""
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_ranges is None:
        ignore_ranges = broadcasting_cfg["ignore_ranges"]

    index = to_any_index(index)
    if n == 1:
        return index
    if checks.is_default_index(index) and ignore_ranges:  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    return index.repeat(n)


def tile_index(index: tp.IndexLike, n: int, ignore_ranges: tp.Optional[bool] = None) -> tp.Index:
    """Tile the whole `index` `n` times.

    Set `ignore_ranges` to True to ignore indexes of type `pd.RangeIndex`."""
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_ranges is None:
        ignore_ranges = broadcasting_cfg["ignore_ranges"]

    index = to_any_index(index)
    if n == 1:
        return index
    if checks.is_default_index(index) and ignore_ranges:  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def stack_indexes(
    *indexes: tp.MaybeTuple[tp.IndexLike],
    drop_duplicates: tp.Optional[bool] = None,
    keep: tp.Optional[str] = None,
    drop_redundant: tp.Optional[bool] = None,
) -> tp.Index:
    """Stack each index in `indexes` on top of each other, from top to bottom.

    Set `drop_duplicates` to True to remove duplicate levels.

    For details on `keep`, see `drop_duplicate_levels`.

    Set `drop_redundant` to True to use `drop_redundant_levels`."""
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if drop_duplicates is None:
        drop_duplicates = broadcasting_cfg["drop_duplicates"]
    if keep is None:
        keep = broadcasting_cfg["keep"]
    if drop_redundant is None:
        drop_redundant = broadcasting_cfg["drop_redundant"]

    levels = []
    for i in range(len(indexes)):
        index = indexes[i]
        if not isinstance(index, pd.MultiIndex):
            levels.append(to_any_index(index))
        else:
            for j in range(index.nlevels):
                levels.append(index.get_level_values(j))

    max_len = max(map(len, levels))
    for i in range(len(levels)):
        if len(levels[i]) < max_len:
            if len(levels[i]) != 1:
                raise ValueError(f"Index at level {i} could not be broadcast to shape ({max_len},) ")
            levels[i] = repeat_index(levels[i], max_len, ignore_ranges=False)
    new_index = pd.MultiIndex.from_arrays(levels)
    if drop_duplicates:
        new_index = drop_duplicate_levels(new_index, keep=keep)
    if drop_redundant:
        new_index = drop_redundant_levels(new_index)
    return new_index


def combine_indexes(*indexes: tp.MaybeTuple[tp.IndexLike], **kwargs) -> tp.Index:
    """Combine each index in `indexes` using Cartesian product.

    Keyword arguments will be passed to `stack_indexes`."""
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    new_index = to_any_index(indexes[0])
    for i in range(1, len(indexes)):
        index1, index2 = new_index, to_any_index(indexes[i])
        new_index1 = repeat_index(index1, len(index2), ignore_ranges=False)
        new_index2 = tile_index(index2, len(index1), ignore_ranges=False)
        new_index = stack_indexes([new_index1, new_index2], **kwargs)
    return new_index


def combine_index_with_keys(index: tp.IndexLike, keys: tp.IndexLike, lens: tp.Sequence[int], **kwargs) -> tp.Index:
    """Build keys based on index lengths."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if not isinstance(keys, pd.Index):
        keys = pd.Index(keys)
    new_index = None
    new_keys = None
    start_idx = 0
    for i in range(len(keys)):
        _index = index[start_idx:start_idx + lens[i]]
        if new_index is None:
            new_index = _index
        else:
            new_index = new_index.append(_index)
        start_idx += lens[i]
        new_key = keys[[i]].repeat(lens[i])
        if new_keys is None:
            new_keys = new_key
        else:
            new_keys = new_keys.append(new_key)
    return stack_indexes([new_keys, new_index], **kwargs)


def concat_indexes(
    *indexes: tp.MaybeTuple[tp.IndexLike],
    index_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
    keys: tp.Optional[tp.IndexLike] = None,
    index_stack_kwargs: tp.KwargsLike = None,
    verify_integrity: bool = True,
    axis: int = 1,
) -> tp.Index:
    """Concatenate indexes.
    
    The following index concatenation methods are supported:

    * 'append': append one index to another
    * 'union': build a union of indexes
    * 'pd_concat': convert indexes to Pandas Series or DataFrames and use `pd.concat`
    * 'factorize': factorize the concatenated index
    * 'factorize_each': factorize each index and concatenate while keeping numbers unique
    * 'reset': reset the concatenated index without applying `keys`
    * Callable: a custom callable that takes the indexes and returns the concatenated index

    Argument `index_concat_method` also accepts a tuple of two options: the second option gets applied
    if the first one fails.

    Use `keys` as an index with the same number of elements as there are indexes to add
    another index level on top of the concatenated indexes.

    If `verify_integrity` is True and `keys` is None, performs various checks depending on the axis."""
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)
    if keys is not None and not isinstance(keys, pd.Index):
        keys = pd.Index(keys)
    if index_stack_kwargs is None:
        index_stack_kwargs = {}
    if axis == 0:
        factorized_name = "row_idx"
    elif axis == 1:
        factorized_name = "col_idx"
    else:
        factorized_name = "group_idx"

    if keys is None:
        all_ranges = True
        for index in indexes:
            if not checks.is_default_index(index):
                all_ranges = False
                break
        if all_ranges:
            return pd.RangeIndex(stop=sum(map(len, indexes)))
    if isinstance(index_concat_method, tuple):
        try:
            return concat_indexes(
                *indexes,
                index_concat_method=index_concat_method[0],
                keys=keys,
                index_stack_kwargs=index_stack_kwargs,
                verify_integrity=verify_integrity,
                axis=axis,
            )
        except Exception as e:
            return concat_indexes(
                *indexes,
                index_concat_method=index_concat_method[1],
                keys=keys,
                index_stack_kwargs=index_stack_kwargs,
                verify_integrity=verify_integrity,
                axis=axis,
            )
    if not isinstance(index_concat_method, str):
        new_index = index_concat_method(indexes)
    elif index_concat_method.lower() == "append":
        new_index = None
        for index in indexes:
            if new_index is None:
                new_index = index
            else:
                new_index = new_index.append(index)
    elif index_concat_method.lower() == "union":
        if keys is not None:
            raise ValueError("Cannot apply keys after concatenating indexes through union")
        new_index = None
        for index in indexes:
            if new_index is None:
                new_index = index
            else:
                new_index = new_index.union(index)
    elif index_concat_method.lower() == "pd_concat":
        new_index = None
        for index in indexes:
            if isinstance(index, pd.MultiIndex):
                index = index.to_frame().reset_index(drop=True)
            else:
                index = index.to_series().reset_index(drop=True)
            if new_index is None:
                new_index = index
            else:
                if isinstance(new_index, pd.DataFrame):
                    if isinstance(index, pd.Series):
                        index = index.to_frame()
                elif isinstance(index, pd.Series):
                    if isinstance(new_index, pd.DataFrame):
                        new_index = new_index.to_frame()
                new_index = pd.concat((new_index, index), ignore_index=True)
        if isinstance(new_index, pd.Series):
            new_index = pd.Index(new_index)
        else:
            new_index = pd.MultiIndex.from_frame(new_index)
    elif index_concat_method.lower() == "factorize":
        new_index = concat_indexes(
            *indexes,
            index_concat_method="append",
            verify_integrity=False,
            axis=axis,
        )
        new_index = pd.Index(pd.factorize(new_index)[0], name=factorized_name)
    elif index_concat_method.lower() == "factorize_each":
        new_index = None
        for index in indexes:
            index = pd.Index(pd.factorize(index)[0], name=factorized_name)
            if new_index is None:
                new_index = index
                next_min = index.max() + 1
            else:
                new_index = new_index.append(index + next_min)
                next_min = index.max() + 1 + next_min
    elif index_concat_method.lower() == "reset":
        return pd.RangeIndex(stop=sum(map(len, indexes)))
    else:
        if axis == 0:
            raise ValueError(f"Invalid index concatenation method '{index_concat_method}'")
        elif axis == 1:
            raise ValueError(f"Invalid column concatenation method '{index_concat_method}'")
        else:
            raise ValueError(f"Invalid group concatenation method '{index_concat_method}'")
    if keys is not None:
        top_index = None
        for i, index in enumerate(indexes):
            repeated_index = repeat_index(keys[[i]], len(index))
            if top_index is None:
                top_index = repeated_index
            else:
                top_index = top_index.append(repeated_index)
        new_index = stack_indexes((top_index, new_index), **index_stack_kwargs)
    if verify_integrity:
        if keys is None:
            if axis == 0:
                if not new_index.is_monotonic_increasing:
                    raise ValueError("Concatenated index is not monotonically increasing")
                if "mixed" in new_index.inferred_type:
                    raise ValueError("Concatenated index is mixed")
                if new_index.has_duplicates:
                    raise ValueError("Concatenated index contains duplicates")
            if axis == 1:
                if new_index.has_duplicates:
                    raise ValueError("Concatenated columns contain duplicates")
            if axis == 2:
                if new_index.has_duplicates:
                    len_sum = 0
                    for index in indexes:
                        if len_sum > 0:
                            prev_index = new_index[:len_sum]
                            this_index = new_index[len_sum:len_sum + len(index)]
                            if len(prev_index.intersection(this_index)) > 0:
                                raise ValueError("Concatenated groups contain duplicates")
                        len_sum += len(index)
    return new_index


def drop_levels(index: tp.Index, levels: tp.MaybeLevelSequence, strict: bool = True) -> tp.Index:
    """Drop `levels` in `index` by their name/position."""
    if not isinstance(index, pd.MultiIndex):
        return index
    if strict:
        return index.droplevel(levels)

    levels_to_drop = set()
    if isinstance(levels, (int, str)):
        levels = (levels,)
    for level in levels:
        if level in index.names:
            levels_to_drop.add(level)
        elif isinstance(level, int) and 0 <= level < index.nlevels or level == -1:
            levels_to_drop.add(level)
    if len(levels_to_drop) < index.nlevels:
        # Drop only if there will be some indexes left
        return index.droplevel(list(levels_to_drop))
    return index


def rename_levels(index: tp.Index, name_dict: tp.Dict[str, tp.Any], strict: bool = True) -> tp.Index:
    """Rename levels in `index` by `name_dict`."""
    for k, v in name_dict.items():
        if isinstance(index, pd.MultiIndex):
            if k in index.names:
                index = index.rename(v, level=k)
            elif strict:
                raise KeyError(f"Level '{k}' not found")
        else:
            if index.name == k:
                index.name = v
            elif strict:
                raise KeyError(f"Level '{k}' not found")
    return index


def select_levels(index: tp.Index, level_names: tp.MaybeLevelSequence) -> tp.Index:
    """Build a new index by selecting one or multiple `level_names` from `index`."""
    if isinstance(level_names, (int, str)):
        return index.get_level_values(level_names)
    levels = [index.get_level_values(level_name) for level_name in level_names]
    return pd.MultiIndex.from_arrays(levels)


def drop_redundant_levels(index: tp.Index) -> tp.Index:
    """Drop levels in `index` that either have a single unnamed value or a range from 0 to n."""
    if not isinstance(index, pd.MultiIndex):
        return index

    levels_to_drop = []
    for i in range(index.nlevels):
        if len(index) > 1 and len(index.levels[i]) == 1 and index.levels[i].name is None:
            levels_to_drop.append(i)
        elif checks.is_default_index(index.get_level_values(i)):
            levels_to_drop.append(i)
    # Remove redundant levels only if there are some non-redundant levels left
    if len(levels_to_drop) < index.nlevels:
        return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index: tp.Index, keep: tp.Optional[str] = None) -> tp.Index:
    """Drop levels in `index` with the same name and values.

    Set `keep` to 'last' to keep last levels, otherwise 'first'.

    Set `keep` to None to use the default."""
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if keep is None:
        keep = broadcasting_cfg["keep"]
    if not isinstance(index, pd.MultiIndex):
        return index
    checks.assert_in(keep.lower(), ["first", "last"])

    levels = []
    levels_to_drop = []
    if keep == "first":
        r = range(0, index.nlevels)
    else:
        r = range(index.nlevels - 1, -1, -1)  # loop backwards
    for i in r:
        level = (index.levels[i].name, tuple(index.get_level_values(i).to_numpy().tolist()))
        if level not in levels:
            levels.append(level)
        else:
            levels_to_drop.append(i)
    return index.droplevel(levels_to_drop)


@register_jitted(cache=True)
def align_arr_indices_nb(a: tp.Array1d, b: tp.Array1d) -> tp.Array1d:
    """Return indices required to align `a` to `b`."""
    idxs = np.empty(b.shape[0], dtype=np.int_)
    g = 0
    for i in range(b.shape[0]):
        for j in range(a.shape[0]):
            if b[i] == a[j]:
                idxs[g] = j
                g += 1
                break
    return idxs


def align_index_to(index1: tp.Index, index2: tp.Index, jitted: tp.JittedOption = None) -> tp.IndexSlice:
    """Align `index1` to have the same shape as `index2` if they have any levels in common.

    Returns index slice for the aligning."""
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if pd.Index.equals(index1, index2):
        return pd.IndexSlice[:]

    # Build map between levels in first and second index
    mapper = {}
    for i in range(index1.nlevels):
        for j in range(index2.nlevels):
            name1 = index1.names[i]
            name2 = index2.names[j]
            if name1 == name2:
                if set(index2.levels[j]).issubset(set(index1.levels[i])):
                    if i in mapper:
                        raise ValueError(f"There are multiple candidate levels with name {name1} in second index")
                    mapper[i] = j
                    continue
                if name1 is not None:
                    raise ValueError(f"Level {name1} in second index contains values not in first index")
    if len(mapper) == 0:
        raise ValueError("Can't find common levels to align both indexes")

    # Factorize first to be accepted by Numba
    factorized = []
    for k, v in mapper.items():
        factorized.append(
            pd.factorize(pd.concat((index1.get_level_values(k).to_series(), index2.get_level_values(v).to_series())))[
                0
            ],
        )
    stacked = np.transpose(np.stack(factorized))
    indices1 = stacked[: len(index1)]
    indices2 = stacked[len(index1) :]
    if len(np.unique(indices1, axis=0)) != len(indices1):
        raise ValueError("Duplicated values in first index are not allowed")

    # Try to tile
    if len(index2) % len(index1) == 0:
        tile_times = len(index2) // len(index1)
        index1_tiled = np.tile(indices1, (tile_times, 1))
        if np.array_equal(index1_tiled, indices2):
            return pd.IndexSlice[np.tile(np.arange(len(index1)), tile_times)]

    # Do element-wise comparison
    unique_indices = np.unique(stacked, axis=0, return_inverse=True)[1]
    unique1 = unique_indices[: len(index1)]
    unique2 = unique_indices[len(index1) :]
    func = jit_reg.resolve_option(align_arr_indices_nb, jitted)
    return pd.IndexSlice[func(unique1, unique2)]


def align_indexes(*indexes: tp.MaybeTuple[tp.Index]) -> tp.List[tp.IndexSlice]:
    """Align multiple indexes to each other."""
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    max_len = max(map(len, indexes))
    indices = []
    for i in range(len(indexes)):
        index_i = indexes[i]
        if len(index_i) == max_len:
            indices.append(pd.IndexSlice[:])
        else:
            for j in range(len(indexes)):
                index_j = indexes[j]
                if len(index_j) == max_len:
                    try:
                        indices.append(align_index_to(index_i, index_j))
                        break
                    except ValueError:
                        pass
            if len(indices) < i + 1:
                raise ValueError(f"Index at position {i} could not be aligned")
    return indices


OptionalLevelSequence = tp.Optional[tp.Sequence[tp.Union[None, tp.Level]]]


def pick_levels(
    index: tp.Index,
    required_levels: OptionalLevelSequence = None,
    optional_levels: OptionalLevelSequence = None,
) -> tp.Tuple[tp.List[int], tp.List[int]]:
    """Pick optional and required levels and return their indices.

    Raises an exception if index has less or more levels than expected."""
    if required_levels is None:
        required_levels = []
    if optional_levels is None:
        optional_levels = []
    checks.assert_instance_of(index, pd.MultiIndex)

    n_opt_set = len(list(filter(lambda x: x is not None, optional_levels)))
    n_req_set = len(list(filter(lambda x: x is not None, required_levels)))
    n_levels_left = index.nlevels - n_opt_set
    if n_req_set < len(required_levels):
        if n_levels_left != len(required_levels):
            n_expected = len(required_levels) + n_opt_set
            raise ValueError(f"Expected {n_expected} levels, found {index.nlevels}")

    levels_left = list(range(index.nlevels))

    # Pick optional levels
    _optional_levels = []
    for level in optional_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _optional_levels.append(level_pos)

    # Pick required levels
    _required_levels = []
    for level in required_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _required_levels.append(level_pos)
    for i, level in enumerate(_required_levels):
        if level is None:
            _required_levels[i] = levels_left.pop(0)

    return _required_levels, _optional_levels


def find_first_occurrence(index_value: tp.Any, index: tp.Index) -> int:
    """Return index of the first occurrence in `index`."""
    loc = index.get_loc(index_value)
    if isinstance(loc, slice):
        return loc.start
    elif isinstance(loc, list):
        return loc[0]
    elif isinstance(loc, np.ndarray):
        return np.flatnonzero(loc)[0]
    return loc
