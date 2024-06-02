# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for signals.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```pycon
>>> import numpy as np
>>> import vectorbtpro as vbt

>>> # vectorbtpro.signals.nb.pos_rank_nb
>>> vbt.signals.nb.pos_rank_nb(np.array([False, True, True, True, False])[:, None])[:, 0]
[-1  0  1  2 -1]
```

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.enums import range_dt, RangeStatus
from vectorbtpro.signals.enums import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import uniform_summing_to_one_nb, rescale_float_to_int_nb, rescale_nb
from vectorbtpro.utils.template import Rep

__all__ = []


# ############# Generation ############# #


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        only_once=None,
        wait=None,
        place_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_nb(target_shape: tp.Shape, only_once: bool, wait: int, place_func_nb: tp.PlaceFunc, *args) -> tp.Array2d:
    """Create a boolean matrix of `target_shape` and place signals using `place_func_nb`.

    Args:
        target_shape (array): Target shape.
        only_once (bool): Whether to run the placement function only once.
        wait (int): Number of ticks to wait before placing the next entry.
        place_func_nb (callable): Signal placement function.

            `place_func_nb` must accept a context of type `vectorbtpro.signals.enums.GenEnContext`,
            and return the index of the last signal (-1 to break the loop).
        *args: Arguments passed to `place_func_nb`.

    !!! note
        The first argument is always a 1-dimensional boolean array that contains only those
        elements where signals can be placed. The range and column indices only describe which
        range this array maps to.
    """
    if wait < 0:
        raise ValueError("wait must be zero or greater")
    out = np.full(target_shape, False, dtype=np.bool_)

    for col in prange(target_shape[1]):
        from_i = 0
        while from_i <= target_shape[0] - 1:
            c = GenEnContext(
                target_shape=target_shape,
                only_once=only_once,
                wait=wait,
                entries_out=out,
                out=out[from_i:, col],
                from_i=from_i,
                to_i=target_shape[0],
                col=col,
            )
            _last_i = place_func_nb(c, *args)
            if _last_i == -1:
                break
            last_i = from_i + _last_i
            if last_i < from_i or last_i >= target_shape[0]:
                raise ValueError("Last index is out of bounds")
            if not out[last_i, col]:
                out[last_i, col] = True
            if only_once:
                break
            from_i = last_i + wait
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="entries", axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        wait=None,
        until_next=None,
        skip_until_exit=None,
        exit_place_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_ex_nb(
    entries: tp.Array2d,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    exit_place_func_nb: tp.PlaceFunc,
    *args,
) -> tp.Array2d:
    """Place exit signals using `exit_place_func_nb` after each signal in `entries`.

    Args:
        entries (array): Boolean array with entry signals.
        wait (int): Number of ticks to wait before placing exits.

            !!! note
                Setting `wait` to 0 or False may result in two signals at one bar.
        until_next (int): Whether to place signals up to the next entry signal.

            !!! note
                Setting it to False makes it difficult to tell which exit belongs to which entry.
        skip_until_exit (bool): Whether to skip processing entry signals until the next exit.

            Has only effect when `until_next` is disabled.

            !!! note
                Setting it to True makes it impossible to tell which exit belongs to which entry.
        exit_place_func_nb (callable): Exit place function.

            `exit_place_func_nb` must accept a context of type `vectorbtpro.signals.enums.GenExContext`,
            and return the index of the last signal (-1 to break the loop).
        *args (callable): Arguments passed to `exit_place_func_nb`.
    """
    if wait < 0:
        raise ValueError("wait must be zero or greater")
    out = np.full_like(entries, False)

    def _place_exits(from_i, to_i, col, last_exit_i):
        if from_i > -1:
            if skip_until_exit and from_i <= last_exit_i:
                return last_exit_i
            from_i += wait
            if not until_next:
                to_i = entries.shape[0]
            if to_i > from_i:
                c = GenExContext(
                    entries=out,
                    until_next=until_next,
                    skip_until_exit=skip_until_exit,
                    exits_out=out,
                    out=out[from_i:to_i, col],
                    wait=wait,
                    from_i=from_i,
                    to_i=to_i,
                    col=col,
                )
                _last_exit_i = exit_place_func_nb(c, *args)
                if _last_exit_i != -1:
                    last_exit_i = from_i + _last_exit_i
                    if last_exit_i < from_i or last_exit_i >= entries.shape[0]:
                        raise ValueError("Last index is out of bounds")
                    if not out[last_exit_i, col]:
                        out[last_exit_i, col] = True
                elif skip_until_exit:
                    last_exit_i = -1
        return last_exit_i

    for col in prange(entries.shape[1]):
        from_i = -1
        last_exit_i = -1
        should_stop = False
        for i in range(entries.shape[0]):
            if entries[i, col]:
                last_exit_i = _place_exits(from_i, i, col, last_exit_i)
                if skip_until_exit and last_exit_i == -1 and from_i != -1:
                    should_stop = True
                    break
                from_i = i
        if should_stop:
            continue
        last_exit_i = _place_exits(from_i, entries.shape[0], col, last_exit_i)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        entry_wait=None,
        exit_wait=None,
        entry_place_func_nb=None,
        entry_args=ch.ArgsTaker(),
        exit_place_func_nb=None,
        exit_args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted
def generate_enex_nb(
    target_shape: tp.Shape,
    entry_wait: int,
    exit_wait: int,
    entry_place_func_nb: tp.PlaceFunc,
    entry_args: tp.Args,
    exit_place_func_nb: tp.PlaceFunc,
    exit_args: tp.Args,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Place entry signals using `entry_place_func_nb` and exit signals using
    `exit_place_func_nb` one after another.

    Args:
        target_shape (array): Target shape.
        entry_wait (int): Number of ticks to wait before placing entries.

            !!! note
                Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and exit can be processed before entry.
        exit_wait (int): Number of ticks to wait before placing exits.

            !!! note
                Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and entry can be processed before exit.
        entry_place_func_nb (callable): Entry place function.

            `entry_place_func_nb` must accept a context of type `vectorbtpro.signals.enums.GenEnExContext`,
            and return the index of the last signal (-1 to break the loop).
        entry_args (tuple): Arguments unpacked and passed to `entry_place_func_nb`.
        exit_place_func_nb (callable): Exit place function.

            `exit_place_func_nb` must accept a context of type `vectorbtpro.signals.enums.GenEnExContext`,
            and return the index of the last signal (-1 to break the loop).
        exit_args (tuple): Arguments unpacked and passed to `exit_place_func_nb`.
    """
    if entry_wait < 0:
        raise ValueError("entry_wait must be zero or greater")
    if exit_wait < 0:
        raise ValueError("exit_wait must be zero or greater")
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)

    def _place_signals(entries_turn, out, from_i, col, wait, place_func_nb, args):
        to_i = target_shape[0]
        if to_i > from_i:
            c = GenEnExContext(
                target_shape=target_shape,
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                entries_out=entries,
                exits_out=exits,
                entries_turn=entries_turn,
                out=out[from_i:to_i, col],
                wait=wait if from_i > 0 else 0,
                from_i=from_i,
                to_i=to_i,
                col=col,
            )
            _last_i = place_func_nb(c, *args)
            if _last_i == -1:
                return -1
            last_i = from_i + _last_i
            if last_i < from_i or last_i >= target_shape[0]:
                raise ValueError("Last index is out of bounds")
            if not out[last_i, col]:
                out[last_i, col] = True
            return last_i
        return -1

    for col in range(target_shape[1]):
        from_i = 0
        entries_turn = True
        first_signal = True
        while from_i != -1:
            if entries_turn:
                if not first_signal:
                    from_i += entry_wait
                from_i = _place_signals(
                    entries_turn,
                    entries,
                    from_i,
                    col,
                    entry_wait,
                    entry_place_func_nb,
                    entry_args,
                )
                entries_turn = False
            else:
                from_i += exit_wait
                from_i = _place_signals(
                    entries_turn,
                    exits,
                    from_i,
                    col,
                    exit_wait,
                    exit_place_func_nb,
                    exit_args,
                )
                entries_turn = True
            first_signal = False

    return entries, exits


# ############# Random signals ############# #


@register_jitted
def rand_place_nb(c: tp.Union[GenEnContext, GenExContext, GenEnExContext], n: tp.FlexArray1d) -> int:
    """`place_func_nb` to randomly pick `n` values.

    `n` uses flexible indexing."""
    size = min(c.to_i - c.from_i, flex_select_1d_pc_nb(n, c.col))
    k = 0
    last_i = -1
    while k < size:
        i = np.random.choice(len(c.out))
        if not c.out[i]:
            c.out[i] = True
            k += 1
        if i > last_i:
            last_i = i
    return last_i


@register_jitted
def rand_by_prob_place_nb(
    c: tp.Union[GenEnContext, GenExContext, GenEnExContext],
    prob: tp.FlexArray2d,
    pick_first: bool,
) -> int:
    """`place_func_nb` to randomly place signals with probability `prob`.

    `prob` uses flexible indexing."""
    last_i = -1
    for i in range(c.from_i, c.to_i):
        if np.random.uniform(0, 1) < flex_select_nb(prob, i, c.col):
            c.out[i - c.from_i] = True
            last_i = i - c.from_i
            if pick_first:
                break
    return last_i


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        n=base_ch.FlexArraySlicer(),
        entry_wait=None,
        exit_wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_rand_enex_nb(
    target_shape: tp.Shape,
    n: tp.FlexArray1d,
    entry_wait: int,
    exit_wait: int,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick a number of entries and the same number of exits one after another.

    Respects `entry_wait` and `exit_wait` constraints through a number of tricks.
    Tries to mimic a uniform distribution as much as possible.

    The idea is the following: with constraints, there is some fixed amount of total
    space required between first entry and last exit. Upscale this space in a way that
    distribution of entries and exit is similar to a uniform distribution. This means
    randomizing the position of first entry, last exit, and all signals between them.

    `n` uses flexible indexing and thus must be at least a 0-dim array."""
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")

    if entry_wait == 1 and exit_wait == 1:
        # Basic case
        both = generate_nb(target_shape, True, 1, rand_place_nb, n * 2)
        for col in prange(both.shape[1]):
            both_idxs = np.flatnonzero(both[:, col])
            entries[both_idxs[0::2], col] = True
            exits[both_idxs[1::2], col] = True
    else:
        for col in prange(target_shape[1]):
            _n = flex_select_1d_pc_nb(n, col)
            if _n == 1:
                entry_idx = np.random.randint(0, target_shape[0] - exit_wait)
                entries[entry_idx, col] = True
            else:
                # Minimum range between two entries
                min_range = entry_wait + exit_wait

                # Minimum total range between first and last entry
                min_total_range = min_range * (_n - 1)
                if target_shape[0] < min_total_range + exit_wait + 1:
                    raise ValueError("Cannot take a larger sample than population")

                # We should decide how much space should be allocate before first and after last entry
                # Maximum space outside of min_total_range
                max_free_space = target_shape[0] - min_total_range - 1

                # If min_total_range is tiny compared to max_free_space, limit it
                # otherwise we would have huge space before first and after last entry
                # Limit it such as distribution of entries mimics uniform
                free_space = min(max_free_space, 3 * target_shape[0] // (_n + 1))

                # What about last exit? it requires exit_wait space
                free_space -= exit_wait

                # Now we need to distribute free space among three ranges:
                # 1) before first, 2) between first and last added to min_total_range, 3) after last
                # We do 2) such that min_total_range can freely expand to maximum
                # We allocate twice as much for 3) as for 1) because an exit is missing
                rand_floats = uniform_summing_to_one_nb(6)
                chosen_spaces = rescale_float_to_int_nb(rand_floats, (0, free_space), free_space)
                first_idx = chosen_spaces[0]
                last_idx = target_shape[0] - np.sum(chosen_spaces[-2:]) - exit_wait - 1

                # Selected range between first and last entry
                total_range = last_idx - first_idx

                # Maximum range between two entries within total_range
                max_range = total_range - (_n - 2) * min_range

                # Select random ranges within total_range
                rand_floats = uniform_summing_to_one_nb(_n - 1)
                chosen_ranges = rescale_float_to_int_nb(rand_floats, (min_range, max_range), total_range)

                # Translate them into entries
                entry_idxs = np.empty(_n, dtype=np.int_)
                entry_idxs[0] = first_idx
                entry_idxs[1:] = chosen_ranges
                entry_idxs = np.cumsum(entry_idxs)
                entries[entry_idxs, col] = True

        # Generate exits
        for col in range(target_shape[1]):
            entry_idxs = np.flatnonzero(entries[:, col])
            for j in range(len(entry_idxs)):
                entry_i = entry_idxs[j] + exit_wait
                if j < len(entry_idxs) - 1:
                    exit_i = entry_idxs[j + 1] - entry_wait
                else:
                    exit_i = entries.shape[0] - 1
                i = np.random.randint(exit_i - entry_i + 1)
                exits[entry_i + i, col] = True
    return entries, exits


def rand_enex_apply_nb(
    target_shape: tp.Shape,
    n: tp.FlexArray1d,
    entry_wait: int,
    exit_wait: int,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """`apply_func_nb` that calls `generate_rand_enex_nb`."""
    return generate_rand_enex_nb(target_shape, n, entry_wait, exit_wait)


# ############# Stop signals ############# #


@register_jitted
def first_place_nb(c: tp.Union[GenEnContext, GenExContext, GenEnExContext], mask: tp.Array2d) -> int:
    """`place_func_nb` that keeps only the first signal in `mask`."""
    last_i = -1
    for i in range(c.from_i, c.to_i):
        if mask[i, c.col]:
            c.out[i - c.from_i] = True
            last_i = i - c.from_i
            break
    return last_i


@register_jitted
def stop_place_nb(
    c: tp.Union[GenExContext, GenEnExContext],
    entry_ts: tp.FlexArray2d,
    ts: tp.FlexArray2d,
    follow_ts: tp.FlexArray2d,
    stop_ts_out: tp.Array2d,
    stop: tp.FlexArray2d,
    trailing: tp.FlexArray2d,
) -> int:
    """`place_func_nb` that places an exit signal whenever a threshold is being hit.

    !!! note
        Waiting time cannot be higher than 1.

        If waiting time is 0, `entry_ts` should be the first value in the bar.
        If waiting time is 1, `entry_ts` should be the last value in the bar.

    Args:
        c (GenExContext or GenEnExContext): Signal context.
        entry_ts (array of float): Entry price.

            Utilizes flexible indexing.
        ts (array of float): Price to compare the stop value against.

            Utilizes flexible indexing. If NaN, defaults to `entry_ts`.
        follow_ts (array of float): Following price.

            Utilizes flexible indexing. If NaN, defaults to `ts`. Applied only if the stop is trailing.
        stop_ts_out (array of float): Array where hit price of each exit will be stored.

            Must be of the full shape.
        stop (array of float): Stop value.

            Utilizes flexible indexing. Set an element to `np.nan` to disable it.
        trailing (array of bool): Whether the stop is trailing.

            Utilizes flexible indexing. Set an element to False to disable it.
    """
    if c.wait > 1:
        raise ValueError("Wait must be either 0 or 1")
    init_i = c.from_i - c.wait
    init_entry_ts = flex_select_nb(entry_ts, init_i, c.col)
    init_stop = flex_select_nb(stop, init_i, c.col)
    if init_stop == 0:
        init_stop = np.nan
    init_trailing = flex_select_nb(trailing, init_i, c.col)
    max_high = min_low = init_entry_ts

    last_i = -1
    for i in range(c.from_i, c.to_i):
        curr_entry_ts = flex_select_nb(entry_ts, i, c.col)
        curr_ts = flex_select_nb(ts, i, c.col)
        curr_follow_ts = flex_select_nb(follow_ts, i, c.col)
        if np.isnan(curr_ts):
            curr_ts = curr_entry_ts
        if np.isnan(curr_follow_ts):
            if not np.isnan(curr_entry_ts):
                if init_stop >= 0:
                    curr_follow_ts = min(curr_entry_ts, curr_ts)
                else:
                    curr_follow_ts = max(curr_entry_ts, curr_ts)
            else:
                curr_follow_ts = curr_ts

        if not np.isnan(init_stop):
            if init_trailing:
                if init_stop >= 0:
                    # Trailing stop buy
                    curr_stop_price = min_low * (1 + abs(init_stop))
                else:
                    # Trailing stop sell
                    curr_stop_price = max_high * (1 - abs(init_stop))
            else:
                curr_stop_price = init_entry_ts * (1 + init_stop)

        # Check if stop price is within bar
        if not np.isnan(init_stop):
            if init_stop >= 0:
                exit_signal = curr_ts >= curr_stop_price
            else:
                exit_signal = curr_ts <= curr_stop_price
            if exit_signal:
                stop_ts_out[i, c.col] = curr_stop_price
                c.out[i - c.from_i] = True
                last_i = i - c.from_i
                break

        # Keep track of lowest low and highest high if trailing
        if init_trailing:
            if curr_follow_ts < min_low:
                min_low = curr_follow_ts
            elif curr_follow_ts > max_high:
                max_high = curr_follow_ts

    return last_i


@register_jitted
def ohlc_stop_place_nb(
    c: tp.Union[GenExContext, GenEnExContext],
    entry_price: tp.FlexArray2d,
    open: tp.FlexArray2d,
    high: tp.FlexArray2d,
    low: tp.FlexArray2d,
    close: tp.FlexArray2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.FlexArray2d,
    tsl_th: tp.FlexArray2d,
    tsl_stop: tp.FlexArray2d,
    tp_stop: tp.FlexArray2d,
    reverse: tp.FlexArray2d,
    is_entry_open: bool = False,
) -> int:
    """`place_func_nb` that places an exit signal whenever a threshold is being hit using OHLC.

    Compared to `stop_place_nb`, takes into account the whole bar, can check for both
    (trailing) stop loss and take profit simultaneously, and tracks hit price and stop type.

    !!! note
        Waiting time cannot be higher than 1.

    Args:
        c (GenExContext or GenEnExContext): Signal context.
        entry_price (array of float): Entry price.

            Utilizes flexible indexing.
        open (array of float): Open price.

            Utilizes flexible indexing. If Nan and `is_entry_open` is True, defaults to entry price.
        high (array of float): High price.

            Utilizes flexible indexing. If NaN, gets calculated from open and close.
        low (array of float): Low price.

            Utilizes flexible indexing. If NaN, gets calculated from open and close.
        close (array of float): Close price.

            Utilizes flexible indexing. If Nan and `is_entry_open` is False, defaults to entry price.
        stop_price_out (array of float): Array where hit price of each exit will be stored.

            Must be of the full shape.
        stop_type_out (array of int): Array where stop type of each exit will be stored.

            Must be of the full shape. 0 for stop loss, 1 for take profit.
        sl_stop (array of float): Stop loss as a percentage.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tsl_th (array of float): Take profit threshold as a percentage for the trailing stop loss.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tsl_stop (array of float): Trailing stop loss as a percentage for the trailing stop loss.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tp_stop (array of float): Take profit as a percentage.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        reverse (array of float): Whether to do the opposite, i.e.: prices are followed downwards.

            Utilizes flexible indexing.
        is_entry_open (bool): Whether entry price comes right at or before open.

            If True, uses high and low of the entry bar. Otherwise, uses only close.
    """
    if c.wait > 1:
        raise ValueError("Wait must be either 0 or 1")
    init_i = c.from_i - c.wait
    init_entry_price = flex_select_nb(entry_price, init_i, c.col)
    init_sl_stop = abs(flex_select_nb(sl_stop, init_i, c.col))
    init_tp_stop = abs(flex_select_nb(tp_stop, init_i, c.col))
    init_tsl_th = abs(flex_select_nb(tsl_th, init_i, c.col))
    init_tsl_stop = abs(flex_select_nb(tsl_stop, init_i, c.col))
    init_reverse = flex_select_nb(reverse, init_i, c.col)
    last_high = last_low = init_entry_price

    last_i = -1
    for i in range(c.from_i - c.wait, c.to_i):
        # Resolve current bar
        _entry_price = flex_select_nb(entry_price, i, c.col)
        _open = flex_select_nb(open, i, c.col)
        _high = flex_select_nb(high, i, c.col)
        _low = flex_select_nb(low, i, c.col)
        _close = flex_select_nb(close, i, c.col)
        if np.isnan(_open) and not np.isnan(_entry_price) and is_entry_open:
            _open = _entry_price
        if np.isnan(_close) and not np.isnan(_entry_price) and not is_entry_open:
            _close = _entry_price
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
        if i > init_i or is_entry_open:
            curr_high = _high
            curr_low = _low
        else:
            curr_high = curr_low = _close

        if i >= c.from_i:
            # Calculate stop prices
            if not np.isnan(init_sl_stop):
                if init_reverse:
                    curr_sl_stop_price = init_entry_price * (1 + init_sl_stop)
                else:
                    curr_sl_stop_price = init_entry_price * (1 - init_sl_stop)
            if not np.isnan(init_tsl_stop):
                if np.isnan(init_tsl_th):
                    if init_reverse:
                        curr_tsl_stop_price = last_low * (1 + init_tsl_stop)
                    else:
                        curr_tsl_stop_price = last_high * (1 - init_tsl_stop)
                else:
                    if init_reverse:
                        if last_low <= init_entry_price * (1 - init_tsl_th):
                            curr_tsl_stop_price = last_low * (1 + init_tsl_stop)
                        else:
                            curr_tsl_stop_price = np.nan
                    else:
                        if last_high >= init_entry_price * (1 + init_tsl_th):
                            curr_tsl_stop_price = last_high * (1 - init_tsl_stop)
                        else:
                            curr_tsl_stop_price = np.nan
            if not np.isnan(init_tp_stop):
                if init_reverse:
                    curr_tp_stop_price = init_entry_price * (1 - init_tp_stop)
                else:
                    curr_tp_stop_price = init_entry_price * (1 + init_tp_stop)

            # Check if stop price is within bar
            exit_signal = False
            if not np.isnan(init_sl_stop):
                # SL hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open <= curr_sl_stop_price:
                        stop_price = _open
                    if curr_low <= curr_sl_stop_price:
                        stop_price = curr_sl_stop_price
                else:
                    if _open >= curr_sl_stop_price:
                        stop_price = _open
                    if curr_high >= curr_sl_stop_price:
                        stop_price = curr_sl_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    stop_type_out[i, c.col] = StopType.SL
                    exit_signal = True

            if not exit_signal and not np.isnan(init_tsl_stop):
                # TSL/TTP hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open <= curr_tsl_stop_price:
                        stop_price = _open
                    if curr_low <= curr_tsl_stop_price:
                        stop_price = curr_tsl_stop_price
                else:
                    if _open >= curr_tsl_stop_price:
                        stop_price = _open
                    if curr_high >= curr_tsl_stop_price:
                        stop_price = curr_tsl_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    if np.isnan(init_tsl_th):
                        stop_type_out[i, c.col] = StopType.TSL
                    else:
                        stop_type_out[i, c.col] = StopType.TTP
                    exit_signal = True

            if not exit_signal and not np.isnan(init_tp_stop):
                # TP hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open >= curr_tp_stop_price:
                        stop_price = _open
                    if curr_high >= curr_tp_stop_price:
                        stop_price = curr_tp_stop_price
                else:
                    if _open <= curr_tp_stop_price:
                        stop_price = _open
                    if curr_low <= curr_tp_stop_price:
                        stop_price = curr_tp_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    stop_type_out[i, c.col] = StopType.TP
                    exit_signal = True

            if exit_signal:
                c.out[i - c.from_i] = True
                last_i = i - c.from_i
                break

        if i > init_i or is_entry_open:
            # Keep track of the lowest low and the highest high
            if curr_low < last_low:
                last_low = curr_low
            if curr_high > last_high:
                last_high = curr_high

    return last_i


# ############# Ranking ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1),
        reset_by_mask=None,
        after_false=None,
        rank_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rank_nb(
    mask: tp.Array2d,
    reset_by_mask: tp.Optional[tp.Array2d],
    after_false: bool,
    after_reset: bool,
    reset_wait: int,
    rank_func_nb: tp.RankFunc,
    *args,
) -> tp.Array2d:
    """Rank each signal using `rank_func_nb`.

    Applies `rank_func_nb` on each True value. Must accept a context of type
    `vectorbtpro.signals.enums.RankContext`. Must return -1 for no rank, otherwise 0 or greater.

    Setting `after_false` to True will disregard the first partition of True values
    if there is no False value before them. Setting `after_reset` to True will disregard
    the first partition of True values coming before the first reset signal. Setting `reset_wait`
    to 0 will treat the signal at the same position as the reset signal as the first signal in
    the next partition. Setting it to 1 will treat it as the last signal in the previous partition."""
    out = np.full(mask.shape, -1, dtype=np.int_)

    for col in prange(mask.shape[1]):
        in_partition = False
        false_seen = not after_false
        reset_seen = reset_by_mask is None
        last_false_i = -1
        last_reset_i = -1
        all_sig_cnt = 0
        all_part_cnt = 0
        all_sig_in_part_cnt = 0
        nonres_sig_cnt = 0
        nonres_part_cnt = 0
        nonres_sig_in_part_cnt = 0
        sig_cnt = 0
        part_cnt = 0
        sig_in_part_cnt = 0

        for i in range(mask.shape[0]):
            if reset_by_mask is not None and reset_by_mask[i, col]:
                last_reset_i = i
            if last_reset_i > -1 and i - last_reset_i == reset_wait:
                reset_seen = True
                sig_cnt = 0
                part_cnt = 0
                sig_in_part_cnt = 0
            if mask[i, col]:
                all_sig_cnt += 1
                if i == 0 or not mask[i - 1, col]:
                    all_part_cnt += 1
                all_sig_in_part_cnt += 1
                if not (after_false and not false_seen) and not (after_reset and not reset_seen):
                    nonres_sig_cnt += 1
                    sig_cnt += 1
                    if not in_partition:
                        nonres_part_cnt += 1
                        part_cnt += 1
                    elif last_reset_i > -1 and i - last_reset_i == reset_wait:
                        part_cnt += 1
                    nonres_sig_in_part_cnt += 1
                    sig_in_part_cnt += 1
                    in_partition = True
                    c = RankContext(
                        mask=mask,
                        reset_by_mask=reset_by_mask,
                        after_false=after_false,
                        after_reset=after_reset,
                        reset_wait=reset_wait,
                        col=col,
                        i=i,
                        last_false_i=last_false_i,
                        last_reset_i=last_reset_i,
                        all_sig_cnt=all_sig_cnt,
                        all_part_cnt=all_part_cnt,
                        all_sig_in_part_cnt=all_sig_in_part_cnt,
                        nonres_sig_cnt=nonres_sig_cnt,
                        nonres_part_cnt=nonres_part_cnt,
                        nonres_sig_in_part_cnt=nonres_sig_in_part_cnt,
                        sig_cnt=sig_cnt,
                        part_cnt=part_cnt,
                        sig_in_part_cnt=sig_in_part_cnt,
                    )
                    out[i, col] = rank_func_nb(c, *args)
            else:
                all_sig_in_part_cnt = 0
                nonres_sig_in_part_cnt = 0
                sig_in_part_cnt = 0
                last_false_i = i
                in_partition = False
                false_seen = True

    return out


@register_jitted
def sig_pos_rank_nb(c: RankContext, allow_gaps: bool) -> int:
    """`rank_func_nb` that returns the rank of each signal by its position in the partition
    if `allow_gaps` is False, otherwise globally.

    Resets at each reset signal."""
    if allow_gaps:
        return c.sig_cnt - 1
    return c.sig_in_part_cnt - 1


@register_jitted
def part_pos_rank_nb(c: RankContext) -> int:
    """`rank_func_nb` that returns the rank of each partition by its position in the series.

    Resets at each reset signal."""
    return c.part_cnt - 1


# ############# Cleaning ############# #


@register_jitted(cache=True)
def clean_enex_1d_nb(
    entries: tp.Array1d,
    exits: tp.Array1d,
    force_first: bool = True,
    keep_conflicts: bool = False,
    reverse_order: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Clean entry and exit arrays by picking the first signal out of each.

    Set `force_first` to True to force placing the first entry/exit before the first exit/entry.
    Set `keep_conflicts` to True to process signals at the same timestamp sequentially instead of removing them.
    Set `reverse_order` to True to reverse the order of signals."""
    entries_out = np.full(entries.shape, False, dtype=np.bool_)
    exits_out = np.full(exits.shape, False, dtype=np.bool_)

    def _process_entry(i, phase):
        if ((not force_first or not reverse_order) and phase == -1) or phase == 1:
            phase = 0
            entries_out[i] = True
        return phase

    def _process_exit(i, phase):
        if ((not force_first or reverse_order) and phase == -1) or phase == 0:
            phase = 1
            exits_out[i] = True
        return phase

    phase = -1
    for i in range(entries.shape[0]):
        if entries[i] and exits[i]:
            if keep_conflicts:
                if not reverse_order:
                    phase = _process_entry(i, phase)
                    phase = _process_exit(i, phase)
                else:
                    phase = _process_exit(i, phase)
                    phase = _process_entry(i, phase)
        elif entries[i]:
            phase = _process_entry(i, phase)
        elif exits[i]:
            phase = _process_exit(i, phase)

    return entries_out, exits_out


@register_chunkable(
    size=ch.ArraySizer(arg_query="entries", axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        exits=ch.ArraySlicer(axis=1),
        force_first=None,
        keep_conflicts=None,
        reverse_order=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def clean_enex_nb(
    entries: tp.Array2d,
    exits: tp.Array2d,
    force_first: bool = True,
    keep_conflicts: bool = False,
    reverse_order: bool = False,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """2-dim version of `clean_enex_1d_nb`."""
    entries_out = np.empty(entries.shape, dtype=np.bool_)
    exits_out = np.empty(exits.shape, dtype=np.bool_)

    for col in prange(entries.shape[1]):
        entries_out[:, col], exits_out[:, col] = clean_enex_1d_nb(
            entries[:, col],
            exits[:, col],
            force_first=force_first,
            keep_conflicts=keep_conflicts,
            reverse_order=reverse_order,
        )
    return entries_out, exits_out


# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), incl_open=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_ranges_nb(mask: tp.Array2d, incl_open: bool = False) -> tp.RecordArray:
    """Create a record of type `vectorbtpro.generic.enums.range_dt` for each range between two signals in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        from_i = -1
        to_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if from_i > -1:
                    to_i = i
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Closed
                    counts[col] += 1
                from_i = i
        if incl_open and from_i < mask.shape[0] - 1:
            r = counts[col]
            new_records["id"][r, col] = r
            new_records["col"][r, col] = col
            new_records["start_idx"][r, col] = from_i
            new_records["end_idx"][r, col] = mask.shape[0] - 1
            new_records["status"][r, col] = RangeStatus.Open
            counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), other_mask=ch.ArraySlicer(axis=1), from_other=None, incl_open=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_two_ranges_nb(
    mask: tp.Array2d,
    other_mask: tp.Array2d,
    from_other: bool = False,
    incl_open: bool = False,
) -> tp.RecordArray:
    """Create a record of type `vectorbtpro.generic.enums.range_dt` for each range between two
    signals in `mask` and `other_mask`.

    If `from_other` is False, returns ranges from each in `mask` to the succeeding in `other_mask`.
    Otherwise, returns ranges from each in `other_mask` to the preceding in `mask`.

    When `mask` and `other_mask` overlap (two signals at the same time), the distance between overlapping
    signals is still considered and `from_i` would match `to_i`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        from_i = -1
        to_i = -1
        if from_other:
            for i in range(mask.shape[0] - 1, -1, -1):
                if other_mask[i, col]:
                    to_i = i
                if mask[i, col]:
                    if to_i != -1:
                        from_i = i
                        r = counts[col]
                        new_records["id"][r, col] = r
                        new_records["col"][r, col] = col
                        new_records["start_idx"][r, col] = from_i
                        new_records["end_idx"][r, col] = to_i
                        new_records["status"][r, col] = RangeStatus.Closed
                        counts[col] += 1
                    elif incl_open:
                        r = counts[col]
                        new_records["id"][r, col] = r
                        new_records["col"][r, col] = col
                        new_records["start_idx"][r, col] = from_i
                        new_records["end_idx"][r, col] = mask.shape[0] - 1
                        new_records["status"][r, col] = RangeStatus.Open
                        counts[col] += 1
        else:
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    from_i = i
                if other_mask[i, col] and from_i != -1:
                    to_i = i
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Closed
                    counts[col] += 1
            if incl_open and to_i < from_i:
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = from_i
                new_records["end_idx"][r, col] = mask.shape[0] - 1
                new_records["status"][r, col] = RangeStatus.Open
                counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record of type `vectorbtpro.generic.enums.range_dt` for each partition of signals in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition:
                    from_i = i
                is_partition = True
            elif is_partition:
                to_i = i
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = from_i
                new_records["end_idx"][r, col] = to_i
                new_records["status"][r, col] = RangeStatus.Closed
                counts[col] += 1
                is_partition = False
            if i == mask.shape[0] - 1:
                if is_partition:
                    to_i = mask.shape[0] - 1
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Open
                    counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record of type `vectorbtpro.generic.enums.range_dt` for each range between two partitions in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition and from_i != -1:
                    to_i = i
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Closed
                    counts[col] += 1
                is_partition = True
                from_i = i
            else:
                is_partition = False

    return generic_nb.repartition_nb(new_records, counts)


# ############# Index ############# #


@register_jitted(cache=True)
def nth_index_1d_nb(mask: tp.Array1d, n: int) -> int:
    """Get the index of the n-th True value.

    !!! note
        `n` starts with 0 and can be negative."""
    if n >= 0:
        found = -1
        for i in range(mask.shape[0]):
            if mask[i]:
                found += 1
                if found == n:
                    return i
    else:
        found = 0
        for i in range(mask.shape[0] - 1, -1, -1):
            if mask[i]:
                found -= 1
                if found == n:
                    return i
    return -1


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), n=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nth_index_nb(mask: tp.Array2d, n: int) -> tp.Array1d:
    """2-dim version of `nth_index_1d_nb`."""
    out = np.empty(mask.shape[1], dtype=np.int_)
    for col in prange(mask.shape[1]):
        out[col] = nth_index_1d_nb(mask[:, col], n)
    return out


@register_jitted(cache=True)
def norm_avg_index_1d_nb(mask: tp.Array1d) -> float:
    """Get mean index normalized to (-1, 1)."""
    mean_index = np.mean(np.flatnonzero(mask))
    return rescale_nb(mean_index, (0, len(mask) - 1), (-1, 1))


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def norm_avg_index_nb(mask: tp.Array2d) -> tp.Array1d:
    """2-dim version of `norm_avg_index_1d_nb`."""
    out = np.empty(mask.shape[1], dtype=np.float_)
    for col in prange(mask.shape[1]):
        out[col] = norm_avg_index_1d_nb(mask[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        mask=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def norm_avg_index_grouped_nb(mask, group_lens):
    """Grouped version of `norm_avg_index_nb`."""
    out = np.empty(len(group_lens), dtype=np.float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        temp_sum = 0
        temp_cnt = 0
        for col in range(from_col, to_col):
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    temp_sum += i
                    temp_cnt += 1
        out[group] = rescale_nb(temp_sum / temp_cnt, (0, mask.shape[0] - 1), (-1, 1))
    return out
