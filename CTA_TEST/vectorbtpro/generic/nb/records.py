# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for records."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import repartition_nb
from vectorbtpro.generic.nb.patterns import pattern_similarity_nb
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.template import Rep

# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), gap_value=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_nb(arr: tp.Array2d, gap_value: tp.Scalar) -> tp.RecordArray:
    """Fill range records between gaps.

    Usage:
        * Find ranges in time series:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_ranges_nb

        >>> a = np.array([
        ...     [np.nan, np.nan, np.nan, np.nan],
        ...     [     2, np.nan, np.nan, np.nan],
        ...     [     3,      3, np.nan, np.nan],
        ...     [np.nan,      4,      4, np.nan],
        ...     [     5, np.nan,      5,      5],
        ...     [     6,      6, np.nan,      6]
        ... ])
        >>> records = get_ranges_nb(a, np.nan)

        >>> pd.DataFrame.from_records(records)
           id  col  start_idx  end_idx  status
        0   0    0          1        3       1
        1   1    0          4        5       0
        2   0    1          2        4       1
        3   1    1          5        5       0
        4   0    2          3        5       1
        5   0    3          4        5       0
        ```
    """
    new_records = np.empty(arr.shape, dtype=range_dt)
    counts = np.full(arr.shape[1], 0, dtype=np.int_)

    for col in prange(arr.shape[1]):
        range_started = False
        start_idx = -1
        end_idx = -1
        store_record = False
        status = -1

        for i in range(arr.shape[0]):
            cur_val = arr[i, col]

            if cur_val == gap_value or np.isnan(cur_val) and np.isnan(gap_value):
                if range_started:
                    # If stopped, save the current range
                    end_idx = i
                    range_started = False
                    store_record = True
                    status = RangeStatus.Closed
            else:
                if not range_started:
                    # If started, register a new range
                    start_idx = i
                    range_started = True

            if i == arr.shape[0] - 1 and range_started:
                # If still running, mark for save
                end_idx = arr.shape[0] - 1
                range_started = False
                store_record = True
                status = RangeStatus.Open

            if store_record:
                # Save range to the records
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = start_idx
                new_records["end_idx"][r, col] = end_idx
                new_records["status"][r, col] = status
                counts[col] += 1

                # Reset running vars for a new range
                store_record = False

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        n_rows=None,
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        id_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index=None,
        delta=None,
        delta_use_index=None,
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_from_delta_nb(
    n_rows: int,
    idx_arr: tp.Array1d,
    id_arr: tp.Array1d,
    col_map: tp.GroupMap,
    index: tp.Optional[tp.Array1d] = None,
    delta: int = 0,
    delta_use_index: bool = False,
    shift: int = 0,
) -> tp.RecordArray:
    """Build delta ranges."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(idx_arr.shape[0], dtype=range_dt)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]

        for r in ridxs:
            r_idx = idx_arr[r] + shift
            if r_idx < 0:
                r_idx = 0
            if r_idx > n_rows - 1:
                r_idx = n_rows - 1
            if delta >= 0:
                start_idx = r_idx
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    end_idx = len(index) - 1
                    status = RangeStatus.Open
                    for i in range(start_idx, index.shape[0]):
                        if index[i] >= index[start_idx] + delta:
                            end_idx = i
                            status = RangeStatus.Closed
                            break
                else:
                    if start_idx + delta < n_rows:
                        end_idx = start_idx + delta
                        status = RangeStatus.Closed
                    else:
                        end_idx = n_rows - 1
                        status = RangeStatus.Open
            else:
                end_idx = r_idx
                status = RangeStatus.Closed
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    start_idx = 0
                    for i in range(end_idx, -1, -1):
                        if index[i] <= index[end_idx] + delta:
                            start_idx = i
                            break
                else:
                    if end_idx + delta >= 0:
                        start_idx = end_idx + delta
                    else:
                        start_idx = 0

            out["id"][r] = id_arr[r]
            out["col"][r] = col
            out["start_idx"][r] = start_idx
            out["end_idx"][r] = end_idx
            out["status"][r] = status

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
        status_arr=ch.ArraySlicer(axis=0),
        freq=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_duration_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    freq: int = 1,
) -> tp.Array1d:
    """Get duration of each range record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.int_)
    for r in prange(start_idx_arr.shape[0]):
        if status_arr[r] == RangeStatus.Open:
            out[r] = end_idx_arr[r] - start_idx_arr[r] + freq
        else:
            out[r] = end_idx_arr[r] - start_idx_arr[r]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_lens=ch.ArraySlicer(axis=0),
        overlapping=None,
        normalize=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_coverage_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_lens: tp.Array1d,
    overlapping: bool = False,
    normalize: bool = False,
) -> tp.Array1d:
    """Get coverage of range records.

    Set `overlapping` to True to get the number of overlapping steps.
    Set `normalize` to True to get the number of steps in relation either to the total number of steps
    (when `overlapping=False`) or to the number of covered steps (when `overlapping=True`).
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], np.nan, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        temp = np.full(index_lens[col], 0, dtype=np.int_)
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                temp[start_idx_arr[r] : end_idx_arr[r] + 1] += 1
            else:
                temp[start_idx_arr[r] : end_idx_arr[r]] += 1
        if overlapping:
            if normalize:
                pos_temp_sum = np.sum(temp > 0)
                if pos_temp_sum == 0:
                    out[col] = np.nan
                else:
                    out[col] = np.sum(temp > 1) / pos_temp_sum
            else:
                out[col] = np.sum(temp > 1)
        else:
            if normalize:
                if index_lens[col] == 0:
                    out[col] = np.nan
                else:
                    out[col] = np.sum(temp > 0) / index_lens[col]
            else:
                out[col] = np.sum(temp > 0)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_len=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ranges_to_mask_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_len: int,
) -> tp.Array2d:
    """Convert ranges to 2-dim mask."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((index_len, col_lens.shape[0]), False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                out[start_idx_arr[r] : end_idx_arr[r] + 1, col] = True
            else:
                out[start_idx_arr[r] : end_idx_arr[r], col] = True

    return out


@register_jitted(cache=True)
def map_ranges_to_projections_nb(
    close: tp.Array2d,
    col_arr: tp.Array1d,
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array1d,
    index: tp.Optional[tp.Array1d] = None,
    proj_start: int = 0,
    proj_start_use_index: bool = False,
    proj_period: tp.Optional[int] = None,
    proj_period_use_index: bool = False,
    incl_end_idx: bool = True,
    extend: bool = False,
    rebase: bool = True,
    start_value: tp.FlexArray1dLike = np.nan,
    ffill: bool = False,
    remove_empty: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array2d]:
    """Map each range into a projection.

    Returns two arrays:

    1. One-dimensional array where elements are record indices
    2. Two-dimensional array where rows are projections"""
    start_value_ = to_1d_array_nb(np.asarray(start_value))

    index_ranges_temp = np.empty((start_idx_arr.shape[0], 2), dtype=np.int_)

    max_duration = 0
    for r in range(start_idx_arr.shape[0]):
        if proj_start_use_index:
            if index is None:
                raise ValueError("Index is required")
            r_proj_start = len(index) - start_idx_arr[r]
            for i in range(start_idx_arr[r], index.shape[0]):
                if index[i] >= index[start_idx_arr[r]] + proj_start:
                    r_proj_start = i - start_idx_arr[r]
                    break
            r_start_idx = start_idx_arr[r] + r_proj_start
        else:
            r_start_idx = start_idx_arr[r] + proj_start
        if status_arr[r] == RangeStatus.Open:
            if incl_end_idx:
                r_duration = end_idx_arr[r] - start_idx_arr[r] + 1
            else:
                r_duration = end_idx_arr[r] - start_idx_arr[r]
        else:
            if incl_end_idx:
                r_duration = end_idx_arr[r] - start_idx_arr[r]
            else:
                r_duration = end_idx_arr[r] - start_idx_arr[r] - 1
        if proj_period is None:
            r_end_idx = start_idx_arr[r] + r_duration
        else:
            if proj_period_use_index:
                if index is None:
                    raise ValueError("Index is required")
                r_proj_period = -1
                for i in range(r_start_idx, index.shape[0]):
                    if index[i] <= index[r_start_idx] + proj_period:
                        r_proj_period = i - r_start_idx
                    else:
                        break
            else:
                r_proj_period = proj_period
            if extend:
                r_end_idx = r_start_idx + r_proj_period
            else:
                r_end_idx = min(start_idx_arr[r] + r_duration, r_start_idx + r_proj_period)
        r_end_idx = r_end_idx + 1
        if r_end_idx > close.shape[0]:
            r_end_idx = close.shape[0]
        if r_start_idx > r_end_idx:
            r_start_idx = r_end_idx
        if r_end_idx - r_start_idx > max_duration:
            max_duration = r_end_idx - r_start_idx
        index_ranges_temp[r, 0] = r_start_idx
        index_ranges_temp[r, 1] = r_end_idx

    ridx_out = np.empty((start_idx_arr.shape[0],), dtype=np.int_)
    proj_out = np.empty((start_idx_arr.shape[0], max_duration), dtype=np.float_)

    k = 0
    for r in range(start_idx_arr.shape[0]):
        if extend:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 0] + proj_out.shape[1]
        else:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 1]
        r_close = close[r_start_idx:r_end_idx, col_arr[r]]
        any_set = False
        for i in range(proj_out.shape[1]):
            if i >= r_close.shape[0]:
                proj_out[k, i] = np.nan
            else:
                if rebase:
                    if i == 0:
                        _start_value = flex_select_1d_pc_nb(start_value_, col_arr[r])
                        if _start_value == -1:
                            proj_out[k, i] = close[-1, col_arr[r]]
                        else:
                            proj_out[k, i] = _start_value
                    else:
                        if r_close[i - 1] == 0:
                            proj_out[k, i] = np.nan
                        else:
                            proj_out[k, i] = proj_out[k, i - 1] * r_close[i] / r_close[i - 1]
                else:
                    proj_out[k, i] = r_close[i]
            if not np.isnan(proj_out[k, i]) and i > 0:
                any_set = True
            if ffill and np.isnan(proj_out[k, i]) and i > 0:
                proj_out[k, i] = proj_out[k, i - 1]
        if any_set or not remove_empty:
            ridx_out[k] = r
            k += 1
    if remove_empty:
        return ridx_out[:k], proj_out[:k]
    return ridx_out, proj_out


@register_jitted(cache=True)
def find_pattern_1d_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    roll_forward: bool = False,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
    overlap_mode: int = OverlapMode.Disallow,
    max_records: tp.Optional[int] = None,
    col: int = 0,
) -> tp.RecordArray:
    """Find all occurrences of a pattern in an array.

    Uses `vectorbtpro.generic.nb.patterns.pattern_similarity_nb` to fill records of the type
    `vectorbtpro.generic.enums.pattern_range_dt`.

    Goes through the array, and for each window selected between `window` and `max_window` (including),
    checks whether the window of array values is similar enough to the pattern. If so, writes a new
    range to the output array. If `window_select_prob` is set, decides whether to test a window based on
    the given probability. The same for `row_select_prob` but on rows.

    If `roll_forward`, windows are rolled forward (`start_idx` is guaranteed to be sorted), otherwise
    backward (`end_idx` is guaranteed to be sorted).

    By default, creates an empty record array of the same size as the number of rows in `arr`.
    This can be increased or decreased using `max_records`."""
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    if max_records is None:
        records_out = np.empty(arr.shape[0], dtype=pattern_range_dt)
    else:
        records_out = np.empty(max_records, dtype=pattern_range_dt)
    r = 0
    min_max_required = False
    if rescale_mode == RescaleMode.MinMax:
        min_max_required = True
    if not np.isnan(min_pct_change):
        min_max_required = True
    if not np.isnan(max_pct_change):
        min_max_required = True
    if not max_error_as_maxdist:
        min_max_required = True
    if min_max_required:
        if np.isnan(pmin):
            pmin = np.nanmin(pattern)
        if np.isnan(pmax):
            pmax = np.nanmax(pattern)

    for i in range(arr.shape[0]):
        if roll_forward:
            from_i = i
            to_i = i + window
            if to_i > arr.shape[0]:
                break
        else:
            from_i = i - window + 1
            to_i = i + 1
            if from_i < 0:
                continue

        if np.random.uniform(0, 1) < row_select_prob:
            _vmin = vmin
            _vmax = vmax
            if min_max_required:
                if np.isnan(_vmin) or np.isnan(_vmax):
                    for j in range(from_i, to_i):
                        if np.isnan(_vmin) or arr[j] < _vmin:
                            _vmin = arr[j]
                        if np.isnan(_vmax) or arr[j] > _vmax:
                            _vmax = arr[j]

            for w in range(window, max_window + 1):
                if roll_forward:
                    from_i = i
                    to_i = i + w
                    if to_i > arr.shape[0]:
                        break
                    if min_max_required:
                        if w > window:
                            if arr[to_i - 1] < _vmin:
                                _vmin = arr[to_i - 1]
                            if arr[to_i - 1] > _vmax:
                                _vmax = arr[to_i - 1]
                else:
                    from_i = i - w + 1
                    to_i = i + 1
                    if from_i < 0:
                        continue
                    if min_max_required:
                        if w > window:
                            if arr[from_i] < _vmin:
                                _vmin = arr[from_i]
                            if arr[from_i] > _vmax:
                                _vmax = arr[from_i]

                if np.random.uniform(0, 1) < window_select_prob:
                    arr_window = arr[from_i:to_i]
                    similarity = pattern_similarity_nb(
                        arr_window,
                        pattern,
                        interp_mode=interp_mode,
                        rescale_mode=rescale_mode,
                        vmin=_vmin,
                        vmax=_vmax,
                        pmin=pmin,
                        pmax=pmax,
                        invert=invert,
                        error_type=error_type,
                        distance_measure=distance_measure,
                        max_error=max_error_,
                        max_error_interp_mode=max_error_interp_mode,
                        max_error_as_maxdist=max_error_as_maxdist,
                        max_error_strict=max_error_strict,
                        min_pct_change=min_pct_change,
                        max_pct_change=max_pct_change,
                        min_similarity=min_similarity,
                        minp=minp,
                    )
                    if not np.isnan(similarity):
                        skip = False
                        while True:
                            if r > 0:
                                if roll_forward:
                                    prev_same_row = records_out["start_idx"][r - 1] == from_i
                                else:
                                    prev_same_row = records_out["end_idx"][r - 1] == to_i
                                if overlap_mode != OverlapMode.AllowAll and prev_same_row:
                                    if similarity > records_out["similarity"][r - 1]:
                                        r -= 1
                                        continue
                                    else:
                                        skip = True
                                        break
                                elif overlap_mode >= 0:
                                    overlap = records_out["end_idx"][r - 1] - from_i
                                    if overlap > overlap_mode:
                                        if similarity > records_out["similarity"][r - 1]:
                                            r -= 1
                                            continue
                                        else:
                                            skip = True
                                            break
                            break
                        if skip:
                            continue
                        if r >= records_out.shape[0]:
                            raise IndexError("Records index out of range. Set a higher max_records.")
                        records_out["id"][r] = r
                        records_out["col"][r] = col
                        records_out["start_idx"][r] = from_i
                        if to_i <= arr.shape[0] - 1:
                            records_out["end_idx"][r] = to_i
                            records_out["status"][r] = RangeStatus.Closed
                        else:
                            records_out["end_idx"][r] = arr.shape[0] - 1
                            records_out["status"][r] = RangeStatus.Open
                        records_out["similarity"][r] = similarity
                        r += 1

    return records_out[:r]


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        pattern=None,
        window=None,
        max_window=None,
        row_select_prob=None,
        window_select_prob=None,
        roll_forward=None,
        interp_mode=None,
        rescale_mode=None,
        vmin=None,
        vmax=None,
        pmin=None,
        pmax=None,
        invert=None,
        error_type=None,
        distance_measure=None,
        max_error=None,
        max_error_interp_mode=None,
        max_error_as_maxdist=None,
        max_error_strict=None,
        min_pct_change=None,
        max_pct_change=None,
        min_similarity=None,
        minp=None,
        overlap_mode=None,
        max_records=None,
    ),
    merge_func=records_ch.merge_records,
)
@register_jitted(cache=True, tags={"can_parallel"})
def find_pattern_nb(
    arr: tp.Array2d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    roll_forward: bool = False,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
    overlap_mode: int = OverlapMode.Disallow,
    max_records: tp.Optional[int] = None,
) -> tp.RecordArray:
    """2-dim version of `find_pattern_1d_nb`."""
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    if max_records is None:
        records_out = np.empty((arr.shape[0], arr.shape[1]), dtype=pattern_range_dt)
    else:
        records_out = np.empty((max_records, arr.shape[1]), dtype=pattern_range_dt)
    record_counts = np.full(arr.shape[1], 0, dtype=np.int_)
    for col in prange(arr.shape[1]):
        records = find_pattern_1d_nb(
            arr[:, col],
            pattern,
            window=window,
            max_window=max_window,
            row_select_prob=row_select_prob,
            window_select_prob=window_select_prob,
            roll_forward=roll_forward,
            interp_mode=interp_mode,
            rescale_mode=rescale_mode,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            invert=invert,
            error_type=error_type,
            distance_measure=distance_measure,
            max_error=max_error_,
            max_error_interp_mode=max_error_interp_mode,
            max_error_as_maxdist=max_error_as_maxdist,
            max_error_strict=max_error_strict,
            min_pct_change=min_pct_change,
            max_pct_change=max_pct_change,
            min_similarity=min_similarity,
            minp=minp,
            overlap_mode=overlap_mode,
            max_records=max_records,
            col=col,
        )
        record_counts[col] = records.shape[0]
        records_out[: records.shape[0], col] = records
    return repartition_nb(records_out, record_counts)


# ############# Drawdowns ############# #


@register_jitted(cache=True)
def drawdown_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Compute drawdown."""
    out = np.empty_like(arr, dtype=np.float_)
    max_val = np.nan
    for i in range(arr.shape[0]):
        if np.isnan(max_val) or arr[i] > max_val:
            max_val = arr[i]
        if max_val == 0:
            out[i] = np.nan
        else:
            out[i] = arr[i] / max_val - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def drawdown_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `drawdown_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = drawdown_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def fill_drawdown_record_nb(
    new_records: tp.RecordArray2d,
    counts: tp.Array2d,
    i: int,
    col: int,
    peak_idx: int,
    valley_idx: int,
    peak_val: float,
    valley_val: float,
    end_val: float,
    status: int,
):
    """Fill a drawdown record."""
    r = counts[col]
    new_records["id"][r, col] = r
    new_records["col"][r, col] = col
    new_records["peak_idx"][r, col] = peak_idx
    new_records["start_idx"][r, col] = peak_idx + 1
    new_records["valley_idx"][r, col] = valley_idx
    new_records["end_idx"][r, col] = i
    new_records["peak_val"][r, col] = peak_val
    new_records["valley_val"][r, col] = valley_val
    new_records["end_val"][r, col] = end_val
    new_records["status"][r, col] = status
    counts[col] += 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        open=ch.ArraySlicer(axis=1),
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_drawdowns_nb(
    open: tp.Optional[tp.Array2d],
    high: tp.Optional[tp.Array2d],
    low: tp.Optional[tp.Array2d],
    close: tp.Array2d,
) -> tp.RecordArray:
    """Fill drawdown records by analyzing a time series.

    Only `close` must be provided, other time series are optional.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_drawdowns_nb

        >>> close = np.array([
        ...     [1, 5, 1, 3],
        ...     [2, 4, 2, 2],
        ...     [3, 3, 3, 1],
        ...     [4, 2, 2, 2],
        ...     [5, 1, 1, 3]
        ... ])
        >>> records = get_drawdowns_nb(None, None, None, close)

        >>> pd.DataFrame.from_records(records)
           id  col  peak_idx  start_idx  valley_idx  end_idx  peak_val  valley_val  \\
        0   0    1         0          1           4        4       5.0         1.0
        1   0    2         2          3           4        4       3.0         1.0
        2   0    3         0          1           2        4       3.0         1.0

           end_val  status
        0      1.0       0
        1      1.0       0
        2      3.0       1
        ```
    """
    new_records = np.empty(close.shape, dtype=drawdown_dt)
    counts = np.full(close.shape[1], 0, dtype=np.int_)

    for col in prange(close.shape[1]):
        drawdown_started = False
        _close = close[0, col]
        if open is None:
            _open = np.nan
        else:
            _open = open[0, col]
        peak_idx = 0
        valley_idx = 0
        peak_val = _open
        valley_val = _open

        for i in range(close.shape[0]):
            _close = close[i, col]
            if open is None:
                _open = np.nan
            else:
                _open = open[i, col]
            if high is None:
                _high = np.nan
            else:
                _high = high[i, col]
            if low is None:
                _low = np.nan
            else:
                _low = low[i, col]
            if np.isnan(_high):
                if np.isnan(_open):
                    _high = _close
                elif np.isnan(_close):
                    _high = _open
                else:
                    _high = max(_open, _close)
            if np.isnan(_low):
                if np.isnan(_open):
                    _low = _close
                elif np.isnan(_close):
                    _low = _open
                else:
                    _low = min(_open, _close)

            if drawdown_started:
                if _open >= peak_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_open,
                        status=DrawdownStatus.Recovered,
                    )
                    peak_idx = i
                    valley_idx = i
                    peak_val = _open
                    valley_val = _open

            if drawdown_started:
                if _low < valley_val:
                    valley_idx = i
                    valley_val = _low
                if _high >= peak_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_high,
                        status=DrawdownStatus.Recovered,
                    )
                    peak_idx = i
                    valley_idx = i
                    peak_val = _high
                    valley_val = _high
            else:
                if np.isnan(peak_val) or _high >= peak_val:
                    peak_idx = i
                    valley_idx = i
                    peak_val = _high
                    valley_val = _high
                elif _low < valley_val:
                    if not np.isnan(valley_val):
                        drawdown_started = True
                    valley_idx = i
                    valley_val = _low

            if drawdown_started:
                if i == close.shape[0] - 1:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_close,
                        status=DrawdownStatus.Active,
                    )

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="peak_val_arr", axis=0),
    arg_take_spec=dict(peak_val_arr=ch.ArraySlicer(axis=0), valley_val_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_drawdown_nb(peak_val_arr: tp.Array1d, valley_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the drawdown of each drawdown record."""
    out = np.empty(valley_val_arr.shape[0], dtype=np.float_)
    for r in prange(valley_val_arr.shape[0]):
        if peak_val_arr[r] == 0:
            out[r] = np.nan
        else:
            out[r] = (valley_val_arr[r] - peak_val_arr[r]) / peak_val_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(start_idx_arr=ch.ArraySlicer(axis=0), valley_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_decline_duration_nb(start_idx_arr: tp.Array1d, valley_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the peak-to-valley phase of each drawdown record."""
    out = np.empty(valley_idx_arr.shape[0], dtype=np.float_)
    for r in prange(valley_idx_arr.shape[0]):
        out[r] = valley_idx_arr[r] - start_idx_arr[r] + 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_idx_arr", axis=0),
    arg_take_spec=dict(valley_idx_arr=ch.ArraySlicer(axis=0), end_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_nb(valley_idx_arr: tp.Array1d, end_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the valley-to-recovery phase of each drawdown record."""
    out = np.empty(end_idx_arr.shape[0], dtype=np.float_)
    for r in prange(end_idx_arr.shape[0]):
        out[r] = end_idx_arr[r] - valley_idx_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        valley_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_ratio_nb(
    start_idx_arr: tp.Array1d,
    valley_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
) -> tp.Array1d:
    """Compute the ratio of the recovery duration to the decline duration of each drawdown record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.float_)
    for r in prange(start_idx_arr.shape[0]):
        if valley_idx_arr[r] - start_idx_arr[r] + 1 == 0:
            out[r] = np.nan
        else:
            out[r] = (end_idx_arr[r] - valley_idx_arr[r]) / (valley_idx_arr[r] - start_idx_arr[r] + 1)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_val_arr", axis=0),
    arg_take_spec=dict(valley_val_arr=ch.ArraySlicer(axis=0), end_val_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_return_nb(valley_val_arr: tp.Array1d, end_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the recovery return of each drawdown record."""
    out = np.empty(end_val_arr.shape[0], dtype=np.float_)
    for r in prange(end_val_arr.shape[0]):
        if valley_val_arr[r] == 0:
            out[r] = np.nan
        else:
            out[r] = (end_val_arr[r] - valley_val_arr[r]) / valley_val_arr[r]
    return out


@register_jitted(cache=True)
def bar_price_nb(records: tp.RecordArray, price: tp.Optional[tp.FlexArray2d]) -> tp.Array1d:
    """Return the bar price."""
    out = np.empty(len(records), dtype=np.float_)
    for i in range(len(records)):
        record = records[i]
        if price is not None:
            out[i] = float(flex_select_nb(price, record["idx"], record["col"]))
        else:
            out[i] = np.nan
    return out