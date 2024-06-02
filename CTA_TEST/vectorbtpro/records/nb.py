# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for records and mapped arrays.

Provides an arsenal of Numba-compiled functions for records and mapped arrays.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in."""

import numpy as np
from numba import prange
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


# ############# Generation ############# #


@register_jitted(cache=True)
def generate_ids_nb(col_arr: tp.Array1d, n_cols: int) -> tp.Array1d:
    """Generate the monotonically increasing id array based on the column index array."""
    col_idxs = np.full(n_cols, 0, dtype=np.int_)
    out = np.empty_like(col_arr)
    for c in range(len(col_arr)):
        out[c] = col_idxs[col_arr[c]]
        col_idxs[col_arr[c]] += 1
    return out


# ############# Indexing ############# #


@register_jitted(cache=True)
def col_lens_nb(col_arr: tp.Array1d, n_cols: int) -> tp.GroupLens:
    """Get column lengths from sorted column array.

    !!! note
        Requires `col_arr` to be in ascending order. This can be done by sorting."""
    col_lens = np.full(n_cols, 0, dtype=np.int_)
    last_col = -1

    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        if col < last_col:
            raise ValueError("col_arr must come in ascending order")
        last_col = col
        col_lens[col] += 1
    return col_lens


@register_jitted(cache=True)
def record_col_lens_select_nb(
    records: tp.RecordArray,
    col_lens: tp.GroupLens,
    new_cols: tp.Array1d,
) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
    """Perform indexing on sorted records using column lengths.

    Returns new records."""
    col_end_idxs = np.cumsum(col_lens)
    col_start_idxs = col_end_idxs - col_lens
    n_values = np.sum(col_lens[new_cols])
    indices_out = np.empty(n_values, dtype=np.int_)
    records_arr_out = np.empty(n_values, dtype=records.dtype)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_start_idxs[new_cols[c]]
        to_r = col_end_idxs[new_cols[c]]
        if from_r == to_r:
            continue
        col_records = np.copy(records[from_r:to_r])
        col_records["col"][:] = c  # don't forget to assign new column indices
        rang = np.arange(from_r, to_r)
        indices_out[j: j + rang.shape[0]] = rang
        records_arr_out[j : j + rang.shape[0]] = col_records
        j += col_records.shape[0]
    return indices_out, records_arr_out


@register_jitted(cache=True)
def col_map_nb(col_arr: tp.Array1d, n_cols: int) -> tp.GroupMap:
    """Build a map between columns and value indices.

    Returns an array with indices segmented by column and an array with column lengths.

    Works well for unsorted column arrays."""
    col_lens_out = np.full(n_cols, 0, dtype=np.int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_lens_out[col] += 1

    col_start_idxs = np.cumsum(col_lens_out) - col_lens_out
    col_idxs_out = np.empty((col_arr.shape[0],), dtype=np.int_)
    col_i = np.full(n_cols, 0, dtype=np.int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_idxs_out[col_start_idxs[col] + col_i[col]] = c
        col_i[col] += 1

    return col_idxs_out, col_lens_out


@register_jitted(cache=True)
def record_col_map_select_nb(
    records: tp.RecordArray,
    col_map: tp.GroupMap,
    new_cols: tp.Array1d,
) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
    """Same as `record_col_lens_select_nb` but using column map `col_map`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    total_count = np.sum(col_lens[new_cols])
    indices_out = np.empty(total_count, dtype=np.int_)
    records_arr_out = np.empty(total_count, dtype=records.dtype)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        col_len = col_lens[new_col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[new_col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_records = np.copy(records[idxs])
        col_records["col"][:] = new_col_i
        indices_out[j: j + col_len] = idxs
        records_arr_out[j : j + col_len] = col_records
        j += col_len
    return indices_out, records_arr_out


# ############# Sorting ############# #


@register_jitted(cache=True)
def is_col_sorted_nb(col_arr: tp.Array1d) -> bool:
    """Check whether the column array is sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
    return True


@register_jitted(cache=True)
def is_col_id_sorted_nb(col_arr: tp.Array1d, id_arr: tp.Array1d) -> bool:
    """Check whether the column and id arrays are sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
        if col_arr[i + 1] == col_arr[i] and id_arr[i + 1] < id_arr[i]:
            return False
    return True


# ############# Filtering ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def first_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Returns the mask of the first N elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[:n]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def last_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Returns the mask of the last N elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[-n:]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def random_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Returns the mask of random N elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[np.random.choice(idxs, n, replace=False)] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def top_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Returns the mask of the top N mapped elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[-n:]]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bottom_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Returns the mask of the bottom N mapped elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[:n]]] = True
    return out


# ############# Mapping ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="records", axis=0),
    arg_take_spec=dict(records=ch.ArraySlicer(axis=0), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def map_records_nb(records: tp.RecordArray, map_func_nb: tp.RecordsMapFunc, *args) -> tp.Array1d:
    """Map each record to a single value.

    `map_func_nb` must accept a single record and `*args`. Must return a single value."""
    out = np.empty(records.shape[0], dtype=np.float_)

    for ridx in prange(records.shape[0]):
        out[ridx] = map_func_nb(records[ridx], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_values"),
    arg_take_spec=dict(n_values=ch.CountAdapter(), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def map_records_meta_nb(n_values: int, map_func_nb: tp.MappedReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `map_records_nb`.

    `map_func_nb` must accept the record index and `*args`. Must return a single value."""
    out = np.empty(n_values, dtype=np.float_)

    for ridx in prange(n_values):
        out[ridx] = map_func_nb(ridx, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_nb(arr: tp.Array1d, col_map: tp.GroupMap, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array1d:
    """Apply function on mapped array or records per column.

    Returns the same shape as `arr`.

    `apply_func_nb` must accept the values of the column and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(arr.shape[0], dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs] = apply_func_nb(arr[idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        n_values=ch.CountAdapter(mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_meta_nb(n_values: int, col_map: tp.GroupMap, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array1d:
    """Meta version of `apply_nb`.

    `apply_func_nb` must accept the indices, the column index, and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(n_values, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs] = apply_func_nb(idxs, col, *args)
    return out


# ############# Reducing ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        id_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        segment_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted
def reduce_mapped_segments_nb(
    mapped_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    id_arr: tp.Array1d,
    col_map: tp.GroupMap,
    segment_arr: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Reduce each segment of values in mapped array.

    Uses the last column, index, and id of each segment for the new value.

    `reduce_func_nb` must accept the values in the segment and `*args`. Must return a single value.

    !!! note
        Groups must come in ascending order per column, and `idx_arr` and `id_arr`
        must come in ascending order per segment of values."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(len(mapped_arr), dtype=mapped_arr.dtype)
    col_arr_out = np.empty(len(mapped_arr), dtype=np.int_)
    idx_arr_out = np.empty(len(mapped_arr), dtype=np.int_)
    id_arr_out = np.empty(len(mapped_arr), dtype=np.int_)

    k = 0
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue

        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]

        segment_start_i = 0
        for i in range(len(idxs)):
            r = idxs[i]
            if i == 0:
                prev_r = -1
            else:
                prev_r = idxs[i - 1]
            if i < len(idxs) - 1:
                next_r = idxs[i + 1]
            else:
                next_r = -1

            if prev_r != -1:
                if segment_arr[r] < segment_arr[prev_r]:
                    raise ValueError("segment_arr must come in ascending order per column")
                elif segment_arr[r] == segment_arr[prev_r]:
                    if idx_arr[r] < idx_arr[prev_r]:
                        raise ValueError("idx_arr must come in ascending order per segment")
                    if id_arr[r] < id_arr[prev_r]:
                        raise ValueError("id_arr must come in ascending order per segment")
                else:
                    segment_start_i = i
            if next_r == -1 or segment_arr[r] != segment_arr[next_r]:
                n_values = i - segment_start_i + 1
                if n_values > 1:
                    out[k] = reduce_func_nb(mapped_arr[idxs[segment_start_i : i + 1]], *args)
                else:
                    out[k] = mapped_arr[r]
                col_arr_out[k] = col
                idx_arr_out[k] = idx_arr[r]
                id_arr_out[k] = id_arr[r]
                k += 1
    return out[:k], col_arr_out[:k], idx_arr_out[:k], id_arr_out[:k]


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce mapped array by column to a single value.

    Faster than `unstack_mapped_nb` and `vbt.*` used together, and also
    requires less memory. But does not take advantage of caching.

    `reduce_func_nb` must accept the mapped array and `*args`.
    Must return a single value."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[col] = reduce_func_nb(mapped_arr[idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(col_map=base_ch.GroupMapSlicer(), fill_value=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_meta_nb(
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `reduce_mapped_nb`.

    `reduce_func_nb` must accept the mapped indices, the column index, and `*args`.
    Must return a single value."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[col] = reduce_func_nb(idxs, col, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce mapped array by column to an index.

    Same as `reduce_mapped_nb` except `idx_arr` must be passed.

    !!! note
        Must return integers or raise an exception."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_meta_nb(
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `reduce_mapped_to_idx_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_array_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Reduce mapped array by column to an array.

    `reduce_func_nb` same as for `reduce_mapped_nb` but must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(col_map=base_ch.GroupMapSlicer(), fill_value=None, reduce_func_nb=None, args=ch.ArgsTaker()),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_array_meta_nb(
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_array_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_array_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Reduce mapped array by column to an index array.

    Same as `reduce_mapped_to_array_nb` except `idx_arr` must be passed.

    !!! note
        Must return integers or raise an exception."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_array_meta_nb(
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_idx_array_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


# ############# Value counts ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        n_uniques=None,
        col_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mapped_value_counts_per_col_nb(codes: tp.Array1d, n_uniques: int, col_map: tp.GroupMap) -> tp.Array2d:
    """Get value counts per column/group of an already factorized mapped array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((n_uniques, col_lens.shape[0]), 0, dtype=np.int_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[:, col] = generic_nb.value_counts_1d_nb(codes[idxs], n_uniques)
    return out


@register_jitted(cache=True)
def mapped_value_counts_per_row_nb(
    mapped_arr: tp.Array1d,
    n_uniques: int,
    idx_arr: tp.Array1d,
    n_rows: int,
) -> tp.Array2d:
    """Get value counts per row of an already factorized mapped array."""
    out = np.full((n_uniques, n_rows), 0, dtype=np.int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c], idx_arr[c]] += 1
    return out


@register_jitted(cache=True)
def mapped_value_counts_nb(mapped_arr: tp.Array1d, n_uniques: int) -> tp.Array2d:
    """Get value counts globally of an already factorized mapped array."""
    out = np.full(n_uniques, 0, dtype=np.int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c]] += 1
    return out


# ############# Coverage ############# #


@register_jitted(cache=True)
def mapped_has_conflicts_nb(col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape) -> bool:
    """Check whether mapped array has positional conflicts."""
    temp = np.zeros(target_shape)

    for i in range(len(col_arr)):
        if temp[idx_arr[i], col_arr[i]] > 0:
            return True
        temp[idx_arr[i], col_arr[i]] = 1
    return False


@register_jitted(cache=True)
def mapped_coverage_map_nb(col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape) -> tp.Array2d:
    """Get the coverage map of a mapped array.

    Each element corresponds to the number of times it was referenced (= duplicates of `col_arr` and `idx_arr`).
    More than one depicts a positional conflict."""
    out = np.zeros(target_shape, dtype=np.int_)

    for i in range(len(col_arr)):
        out[idx_arr[i], col_arr[i]] += 1
    return out


# ############# Unstacking ############# #


@register_jitted(cache=True, is_generated_jit=True)
def unstack_mapped_nb(
    mapped_arr: tp.Array1d,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    target_shape: tp.Shape,
    fill_value: float,
) -> tp.Array2d:
    """Unstack mapped array using index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _unstack_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value):
        out = np.full(target_shape, fill_value, dtype=dtype)

        for r in range(mapped_arr.shape[0]):
            out[idx_arr[r], col_arr[r]] = mapped_arr[r]
        return out

    if not nb_enabled:
        return _unstack_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value)

    return _unstack_mapped_nb


@register_jitted(cache=True, is_generated_jit=True)
def ignore_unstack_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.GroupMap, fill_value: float) -> tp.Array2d:
    """Unstack mapped array by ignoring index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value):
        col_idxs, col_lens = col_map
        col_start_idxs = np.cumsum(col_lens) - col_lens
        out = np.full((np.max(col_lens), col_lens.shape[0]), fill_value, dtype=dtype)

        for col in range(col_lens.shape[0]):
            col_len = col_lens[col]
            if col_len == 0:
                continue
            col_start_idx = col_start_idxs[col]
            idxs = col_idxs[col_start_idx : col_start_idx + col_len]
            out[:col_len, col] = mapped_arr[idxs]

        return out

    if not nb_enabled:
        return _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value)

    return _ignore_unstack_mapped_nb


@register_jitted(cache=True)
def unstack_index_nb(repeat_cnt_arr: tp.Array1d) -> tp.Array1d:
    """Unstack index using the number of times each element must repeat.

    `repeat_cnt_arr` can be created from the coverage map."""
    out = np.empty(np.sum(repeat_cnt_arr), dtype=np.int_)

    k = 0
    for i in range(len(repeat_cnt_arr)):
        out[k : k + repeat_cnt_arr[i]] = i
        k += repeat_cnt_arr[i]
    return out


@register_jitted(cache=True, is_generated_jit=True)
def repeat_unstack_mapped_nb(
    mapped_arr: tp.Array1d,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    repeat_cnt_arr: tp.Array1d,
    n_cols: int,
    fill_value: float,
) -> tp.Array2d:
    """Unstack mapped array using repeated index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _repeat_unstack_mapped_nb(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value):
        index_start_arr = np.cumsum(repeat_cnt_arr) - repeat_cnt_arr
        out = np.full((np.sum(repeat_cnt_arr), n_cols), fill_value, dtype=dtype)
        temp = np.zeros((len(repeat_cnt_arr), n_cols), dtype=np.int_)

        for i in range(len(col_arr)):
            out[index_start_arr[idx_arr[i]] + temp[idx_arr[i], col_arr[i]], col_arr[i]] = mapped_arr[i]
            temp[idx_arr[i], col_arr[i]] += 1
        return out

    if not nb_enabled:
        return _repeat_unstack_mapped_nb(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value)

    return _repeat_unstack_mapped_nb
