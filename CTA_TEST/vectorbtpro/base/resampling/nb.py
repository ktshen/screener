# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for resampling."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.datetime_nb import d_td

__all__ = []


@register_jitted(cache=True)
def date_range_nb(
    start: np.datetime64,
    end: np.datetime64,
    freq: np.timedelta64 = d_td,
    incl_left: bool = True,
    incl_right: bool = True,
) -> tp.Array1d:
    """Generate a datetime index with nanosecond precision from a date range.

    Inspired by [pandas.date_range](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html)."""
    values_len = int(np.floor((end - start) / freq)) + 1
    values = np.empty(values_len, dtype="datetime64[ns]")
    for i in range(values_len):
        values[i] = start + i * freq
    if start == end:
        if not incl_left and not incl_right:
            values = values[1:-1]
    else:
        if not incl_left or not incl_right:
            if not incl_left and len(values) and values[0] == start:
                values = values[1:]
            if not incl_right and len(values) and values[-1] == end:
                values = values[:-1]
    return values


@register_jitted(cache=True)
def map_to_target_index_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
    raise_missing: bool = True,
) -> tp.Array1d:
    """Get the index of each from `source_index` in `target_index`.

    If `before` is True, applied on elements that come before and including that index.
    Otherwise, applied on elements that come after and including that index.

    If `raise_missing` is True, will throw an error if an index cannot be mapped.
    Otherwise, the element for that index becomes -1."""
    out = np.empty(len(source_index), dtype=np.int_)
    from_j = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] <= source_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")

        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if target_freq is None:
                if before and source_index[i] <= target_index[j]:
                    if j == 0 or target_index[j - 1] < source_index[i]:
                        out[i] = from_j = j
                        found = True
                        break
                if not before and target_index[j] <= source_index[i]:
                    if j == len(target_index) - 1 or source_index[i] < target_index[j + 1]:
                        out[i] = from_j = j
                        found = True
                        break
            else:
                if before and target_index[j] - target_freq < source_index[i] <= target_index[j]:
                    out[i] = from_j = j
                    found = True
                    break
                if not before and target_index[j] <= source_index[i] < target_index[j] + target_freq:
                    out[i] = from_j = j
                    found = True
                    break

        if not found:
            if raise_missing:
                raise ValueError("Resampling failed: cannot map some indices")
            out[i] = -1
    return out


@register_jitted(cache=True)
def index_difference_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
) -> tp.Array1d:
    """Get the elements in `source_index` not present in `target_index`."""
    out = np.empty(len(source_index), dtype=np.int_)
    from_j = 0
    k = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] <= source_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if source_index[i] < target_index[j]:
                break
            if source_index[i] == target_index[j]:
                from_j = j
                found = True
                break
            from_j = j
        if not found:
            out[k] = i
            k += 1
    return out[:k]


@register_jitted(cache=True)
def map_index_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Get the source bounds that correspond to each target index.

    If `target_freq` is not None, the right bound is limited by the frequency in `target_freq`.
    Otherwise, the right bound is the next index in `target_index`.

    Returns a 2-dim array where the first column is the absolute start index (including) and
    the second column is the absolute end index (excluding).

    If an element cannot be mapped, the start and end of the range becomes -1.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed."""
    range_starts_out = np.empty(len(target_index), dtype=np.int_)
    range_ends_out = np.empty(len(target_index), dtype=np.int_)

    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")

        from_j = -1
        for j in range(to_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if target_freq is None:
                if before:
                    if i == 0 and source_index[j] <= target_index[i]:
                        found = True
                    elif i > 0 and target_index[i - 1] < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if i == len(target_index) - 1 and target_index[i] <= source_index[j]:
                        found = True
                    elif i < len(target_index) - 1 and target_index[i] <= source_index[j] < target_index[i + 1]:
                        found = True
                    elif i < len(target_index) - 1 and source_index[j] >= target_index[i + 1]:
                        break
            else:
                if before:
                    if target_index[i] - target_freq < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if target_index[i] <= source_index[j] < target_index[i] + target_freq:
                        found = True
                    elif source_index[j] >= target_index[i] + target_freq:
                        break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            range_starts_out[i] = -1
            range_ends_out[i] = -1
        else:
            range_starts_out[i] = from_j
            range_ends_out[i] = to_j

    return range_starts_out, range_ends_out


@register_jitted(cache=True)
def map_bounds_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool = True,
    closed_rbound: bool = False,
    skip_minus_one: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Get the source bounds that correspond to the target bounds.

    Returns a 2-dim array where the first column is the absolute start index (including) nad
    the second column is the absolute end index (excluding).

    If an element cannot be mapped, the start and end of the range becomes -1.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed."""
    range_starts_out = np.empty(len(target_lbound_index), dtype=np.int_)
    range_ends_out = np.empty(len(target_lbound_index), dtype=np.int_)
    k = 0

    to_j = 0
    for i in range(len(target_lbound_index)):
        if i > 0 and target_lbound_index[i] < target_lbound_index[i - 1]:
            raise ValueError("Target left-bound index must be increasing")
        if i > 0 and target_rbound_index[i] < target_rbound_index[i - 1]:
            raise ValueError("Target right-bound index must be increasing")

        from_j = -1
        for j in range(len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if closed_lbound and closed_rbound:
                if target_lbound_index[i] <= source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            elif closed_lbound:
                if target_lbound_index[i] <= source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            elif closed_rbound:
                if target_lbound_index[i] < source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            else:
                if target_lbound_index[i] < source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if skip_minus_one:
            if from_j != -1:
                range_starts_out[k] = from_j
                range_ends_out[k] = to_j
                k += 1
        else:
            if from_j == -1:
                range_starts_out[i] = -1
                range_ends_out[i] = -1
            else:
                range_starts_out[i] = from_j
                range_ends_out[i] = to_j

    if skip_minus_one:
        return range_starts_out[:k], range_ends_out[:k]
    return range_starts_out, range_ends_out


@register_jitted(cache=True)
def resample_source_mask_nb(
    source_mask: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
) -> tp.Array1d:
    """Resample a source mask to the target index.

    Becomes True only if the target bar is fully contained in the source bar. The source bar
    is represented by a non-interrupting sequence of True values in the source mask."""
    out = np.full(len(target_index), False, dtype=np.bool_)

    from_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")
        target_lbound = target_index[i]
        if target_freq is None:
            if i + 1 < len(target_index):
                target_rbound = target_index[i + 1]
            else:
                target_rbound = None
        else:
            target_rbound = target_index[i] + target_freq

        found_start = False
        for j in range(from_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Source index must be increasing")
            source_lbound = source_index[j]
            if source_freq is None:
                if j + 1 < len(source_index):
                    source_rbound = source_index[j + 1]
                else:
                    source_rbound = None
            else:
                source_rbound = source_index[j] + source_freq

            if target_rbound is not None and target_rbound <= source_lbound:
                break
            if found_start or (
                target_lbound >= source_lbound and (source_rbound is None or target_lbound < source_rbound)
            ):
                if not found_start:
                    from_j = j
                    found_start = True
                if not source_mask[j]:
                    break
                if source_rbound is None or (target_rbound is not None and target_rbound <= source_rbound):
                    out[i] = True
                    break

    return out
