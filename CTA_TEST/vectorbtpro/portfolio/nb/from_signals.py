# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio simulation based on signals."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.generic.enums import BarZone
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.from_order_func import no_post_func_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.signals.enums import StopType
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb
from vectorbtpro.utils.math_ import is_less_nb
from vectorbtpro.utils.template import RepFunc


@register_jitted(cache=True)
def resolve_pending_conflict_nb(
    is_pending_long: bool,
    is_user_long: bool,
    upon_adj_conflict: int,
    upon_opp_conflict: int,
) -> tp.Tuple[bool, bool]:
    """Resolve any conflict between a pending signal and a user-defined signal.

    Returns whether to keep the pending signal and execute the user signal."""
    if (is_pending_long and is_user_long) or (not is_pending_long and not is_user_long):
        if upon_adj_conflict == PendingConflictMode.KeepIgnore:
            return True, False
        if upon_adj_conflict == PendingConflictMode.KeepExecute:
            return True, True
        if upon_adj_conflict == PendingConflictMode.CancelIgnore:
            return False, False
        if upon_adj_conflict == PendingConflictMode.CancelExecute:
            return False, True
        raise ValueError("Invalid PendingConflictMode option")
    if upon_opp_conflict == PendingConflictMode.KeepIgnore:
        return True, False
    if upon_opp_conflict == PendingConflictMode.KeepExecute:
        return True, True
    if upon_opp_conflict == PendingConflictMode.CancelIgnore:
        return False, False
    if upon_opp_conflict == PendingConflictMode.CancelExecute:
        return False, True
    raise ValueError("Invalid PendingConflictMode option")


@register_jitted(cache=True)
def generate_stop_signal_nb(
    position_now: float,
    stop_exit_type: int,
    accumulate: int = False,
) -> tp.Tuple[bool, bool, bool, bool, int]:
    """Generate stop signal and change accumulation if needed."""
    is_long_entry = False
    is_long_exit = False
    is_short_entry = False
    is_short_exit = False
    if position_now > 0:
        if stop_exit_type == StopExitType.Close:
            is_long_exit = True
            accumulate = AccumulationMode.Disabled
        elif stop_exit_type == StopExitType.CloseReduce:
            is_long_exit = True
        elif stop_exit_type == StopExitType.Reverse:
            is_short_entry = True
            accumulate = AccumulationMode.Disabled
        elif stop_exit_type == StopExitType.ReverseReduce:
            is_short_entry = True
        else:
            raise ValueError("Invalid StopExitType option")
    elif position_now < 0:
        if stop_exit_type == StopExitType.Close:
            is_short_exit = True
            accumulate = AccumulationMode.Disabled
        elif stop_exit_type == StopExitType.CloseReduce:
            is_short_exit = True
        elif stop_exit_type == StopExitType.Reverse:
            is_long_entry = True
            accumulate = AccumulationMode.Disabled
        elif stop_exit_type == StopExitType.ReverseReduce:
            is_long_entry = True
        else:
            raise ValueError("Invalid StopExitType option")
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@register_jitted(cache=True)
def resolve_stop_exit_price_nb(
    stop_price: float,
    close: float,
    stop_exit_price: float,
) -> float:
    """Resolve the exit price of a stop order."""
    if stop_exit_price == StopExitPrice.Stop:
        return float(stop_price)
    elif stop_exit_price == StopExitPrice.Close:
        return float(close)
    elif stop_exit_price < 0:
        raise ValueError("Invalid StopExitPrice option")
    return float(stop_exit_price)


@register_jitted(cache=True)
def resolve_signal_conflict_nb(
    position_now: float,
    is_entry: bool,
    is_exit: bool,
    direction: int,
    conflict_mode: int,
) -> tp.Tuple[bool, bool]:
    """Resolve any conflict between an entry and an exit."""
    if is_entry and is_exit:
        # Conflict
        if conflict_mode == ConflictMode.Entry:
            # Ignore exit signal
            is_exit = False
        elif conflict_mode == ConflictMode.Exit:
            # Ignore entry signal
            is_entry = False
        elif conflict_mode == ConflictMode.Adjacent:
            # Take the signal adjacent to the position we are in
            if position_now == 0:
                # Cannot decide -> ignore
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.Both:
                    if position_now > 0:
                        is_exit = False
                    elif position_now < 0:
                        is_entry = False
                else:
                    is_exit = False
        elif conflict_mode == ConflictMode.Opposite:
            # Take the signal opposite to the position we are in
            if position_now == 0:
                if direction == Direction.Both:
                    # Cannot decide -> ignore
                    is_entry = False
                    is_exit = False
                else:
                    is_exit = False
            else:
                if direction == Direction.Both:
                    if position_now > 0:
                        is_entry = False
                    elif position_now < 0:
                        is_exit = False
                else:
                    is_entry = False
        elif conflict_mode == ConflictMode.Ignore:
            is_entry = False
            is_exit = False
        else:
            raise ValueError("Invalid ConflictMode option")
    return is_entry, is_exit


@register_jitted(cache=True)
def resolve_dir_conflict_nb(
    position_now: float,
    is_long_entry: bool,
    is_short_entry: bool,
    upon_dir_conflict: int,
) -> tp.Tuple[bool, bool]:
    """Resolve any direction conflict between a long entry and a short entry."""
    if is_long_entry and is_short_entry:
        if upon_dir_conflict == DirectionConflictMode.Long:
            is_short_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Short:
            is_long_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Adjacent:
            if position_now > 0:
                is_short_entry = False
            elif position_now < 0:
                is_long_entry = False
            else:
                is_long_entry = False
                is_short_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Opposite:
            if position_now > 0:
                is_long_entry = False
            elif position_now < 0:
                is_short_entry = False
            else:
                is_long_entry = False
                is_short_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Ignore:
            is_long_entry = False
            is_short_entry = False
        else:
            raise ValueError("Invalid DirectionConflictMode option")
    return is_long_entry, is_short_entry


@register_jitted(cache=True)
def resolve_opposite_entry_nb(
    position_now: float,
    is_long_entry: bool,
    is_long_exit: bool,
    is_short_entry: bool,
    is_short_exit: bool,
    upon_opposite_entry: int,
    accumulate: int,
) -> tp.Tuple[bool, bool, bool, bool, int]:
    """Resolve opposite entry."""
    if position_now > 0 and is_short_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            is_short_entry = False
        elif upon_opposite_entry == OppositeEntryMode.Close:
            is_short_entry = False
            is_long_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            is_short_entry = False
            is_long_exit = True
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.ReverseReduce:
            pass
        else:
            raise ValueError("Invalid OppositeEntryMode option")
    if position_now < 0 and is_long_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            is_long_entry = False
        elif upon_opposite_entry == OppositeEntryMode.Close:
            is_long_entry = False
            is_short_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            is_long_entry = False
            is_short_exit = True
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.ReverseReduce:
            pass
        else:
            raise ValueError("Invalid OppositeEntryMode option")
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@register_jitted(cache=True)
def signal_to_size_nb(
    position_now: float,
    val_price_now: float,
    value_now: float,
    is_long_entry: bool,
    is_long_exit: bool,
    is_short_entry: bool,
    is_short_exit: bool,
    size: float,
    size_type: int,
    accumulate: int,
) -> tp.Tuple[float, int, int]:
    """Translate direction-aware signals into size, size type, and direction."""

    if (
        accumulate != AccumulationMode.Disabled
        and accumulate != AccumulationMode.Both
        and accumulate != AccumulationMode.AddOnly
        and accumulate != AccumulationMode.RemoveOnly
    ):
        raise ValueError("Invalid AccumulationMode option")

    def _check_size_type(_size_type):
        if (
            _size_type == SizeType.TargetAmount
            or _size_type == SizeType.TargetValue
            or _size_type == SizeType.TargetPercent
            or _size_type == SizeType.TargetPercent100
        ):
            raise ValueError("Target size types are not supported")

    if is_less_nb(size, 0):
        raise ValueError("Negative size is not allowed. Please express direction using signals.")
    if size_type == SizeType.Percent100:
        size /= 100
        size_type = SizeType.Percent
    if size_type == SizeType.ValuePercent100:
        size /= 100
        size_type = SizeType.ValuePercent
    if size_type == SizeType.ValuePercent:
        size *= value_now
        size_type = SizeType.Value
    order_size = np.nan
    direction = Direction.Both
    abs_position_now = abs(position_now)

    if position_now > 0:
        # We're in a long position
        if is_short_entry:
            _check_size_type(size_type)
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = -size
            else:
                # Reverse the position
                if not np.isnan(size):
                    if size_type == SizeType.Percent:
                        order_size = -size
                    else:
                        order_size = -abs_position_now
                        if size_type == SizeType.Value:
                            order_size -= size / val_price_now
                        else:
                            order_size -= size
                        size_type = SizeType.Amount
        elif is_long_exit:
            direction = Direction.LongOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                _check_size_type(size_type)
                order_size = -size
            else:
                # Close the position
                order_size = -abs_position_now
                size_type = SizeType.Amount
        elif is_long_entry:
            _check_size_type(size_type)
            direction = Direction.LongOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # Increase the position
                order_size = size
    elif position_now < 0:
        # We're in a short position
        if is_long_entry:
            _check_size_type(size_type)
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = size
            else:
                # Reverse the position
                if not np.isnan(size):
                    if size_type == SizeType.Percent:
                        order_size = size
                    else:
                        order_size = abs_position_now
                        if size_type == SizeType.Value:
                            order_size += size / val_price_now
                        else:
                            order_size += size
                        size_type = SizeType.Amount
        elif is_short_exit:
            direction = Direction.ShortOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                _check_size_type(size_type)
                order_size = size
            else:
                # Close the position
                order_size = abs_position_now
                size_type = SizeType.Amount
        elif is_short_entry:
            _check_size_type(size_type)
            direction = Direction.ShortOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # Increase the position
                order_size = -size
    else:
        _check_size_type(size_type)
        if is_long_entry:
            # Open long position
            order_size = size
        elif is_short_entry:
            # Open short position
            order_size = -size

    if direction == Direction.ShortOnly:
        order_size = -order_size
    return order_size, size_type, direction


@register_jitted(cache=True)
def is_limit_active_nb(init_idx: int, init_price: float) -> bool:
    """Check whether a limit order is active."""
    return init_idx != -1 and not np.isnan(init_price)


@register_jitted(cache=True)
def is_stop_active_nb(init_idx: int, stop: float) -> bool:
    """Check whether a stop order is active."""
    return init_idx != -1 and not np.isnan(stop)


@register_jitted(cache=True)
def is_time_stop_active_nb(init_idx: int, stop: int) -> bool:
    """Check whether a time stop order is active."""
    return init_idx != -1 and stop != -1


@register_jitted(cache=True)
def should_update_stop_nb(new_stop: float, upon_stop_update: int) -> bool:
    """Whether to update stop."""
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if not np.isnan(new_stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def should_update_time_stop_nb(new_stop: int, upon_stop_update: int) -> bool:
    """Whether to update time stop."""
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if new_stop != -1 or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def check_limit_expired_nb(
    creation_idx: int,
    i: int,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether limit is expired by comparing the current index with the creation index.

    Returns whether the limit expires already on open, and whether the limit expires during this bar."""
    if tif == -1 and expiry == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if creation_idx + tif <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < creation_idx + tif < i + 1:
                is_expired = True
        if expiry != -1:
            if expiry <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < expiry < i + 1:
                is_expired = True
        return is_expired_on_open, is_expired
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Index must be provided for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Frequency must be provided for TimeDeltaFormat.Index")
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if index[creation_idx] + tif <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < index[creation_idx] + tif < index[i] + freq:
                is_expired = True
        if expiry != -1:
            if expiry <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < expiry < index[i] + freq:
                is_expired = True
        return is_expired_on_open, is_expired
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def resolve_limit_price_nb(
    price: float,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the limit price."""
    if delta_format == DeltaFormat.Percent100:
        limit_delta /= 100
        delta_format = DeltaFormat.Percent
    if not np.isnan(limit_delta):
        if hit_below:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta > 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = price - limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = price * (1 - limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
        else:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta < 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = price + limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = price * (1 + limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
    else:
        limit_price = price
    return limit_price


@register_jitted(cache=True)
def check_limit_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    size: float,
    direction: int = Direction.Both,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    limit_reverse: bool = False,
    can_use_ohlc: bool = True,
    check_open: bool = True,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the limit price using `resolve_limit_price_nb` and check whether it was hit.

    Returns the limit price, whether it was hit before open, and whether it was hit during this bar.

    If `can_use_ohlc` and `check_open` is True and the stop is hit before open, returns open."""
    if size == 0:
        raise ValueError("Limit order size cannot be zero")
    _size = get_diraware_size_nb(size, direction)
    hit_below = (_size > 0 and not limit_reverse) or (_size < 0 and limit_reverse)
    limit_price = resolve_limit_price_nb(
        price=price,
        limit_delta=limit_delta,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    hit_on_open = False

    if can_use_ohlc:
        high, low = resolve_hl_nb(
            open=open,
            high=high,
            low=low,
            close=close,
        )
        if hit_below:
            if check_open and open <= limit_price:
                hit_on_open = True
                hit = True
                limit_price = open
            else:
                hit = low <= limit_price
                if hit and np.isinf(limit_price):
                    limit_price = low
        else:
            if check_open and open >= limit_price:
                hit_on_open = True
                hit = True
                limit_price = open
            else:
                hit = high >= limit_price
                if hit and np.isinf(limit_price):
                    limit_price = high
    else:
        if hit_below:
            hit = close <= limit_price
        else:
            hit = close >= limit_price
        if hit and np.isinf(limit_price):
            limit_price = close
    return limit_price, hit_on_open, hit


@register_jitted(cache=True)
def resolve_stop_price_nb(
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the stop price."""
    if delta_format == DeltaFormat.Percent100:
        stop /= 100
        delta_format = DeltaFormat.Percent
    if hit_below:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price - abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 - abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    else:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price + abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 + abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    return stop_price


@register_jitted(cache=True)
def check_stop_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    is_position_long: bool,
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the stop price using `resolve_stop_price_nb` and check whether it was hit.

    If `can_use_ohlc` and `check_open` is True and the stop is hit before open, returns open.

    Returns the stop price, whether it was hit before open, and whether it was hit during this bar."""
    hit_below = (is_position_long and hit_below) or (not is_position_long and not hit_below)
    stop_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    return check_price_hit_nb(
        open=open,
        high=high,
        low=low,
        close=close,
        price=stop_price,
        hit_below=hit_below,
        can_use_ohlc=can_use_ohlc,
        check_open=check_open,
    )


@register_jitted(cache=True)
def check_td_stop_hit_nb(
    init_idx: int,
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether TD stop was hit by comparing the current index with the initial index.

    Returns whether the stop was hit already on open, and whether the stop was hit during this bar.

    Identical mechanics to `check_limit_hit_nb`."""
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if init_idx + stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < init_idx + stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Index must be provided for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Frequency must be provided for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if index[init_idx] + stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < index[init_idx] + stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_dt_stop_hit_nb(
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether DT stop was hit by comparing the current index with the initial index.

    Returns whether the stop was hit already on open, and whether the stop was hit during this bar.

    Identical mechanics to `check_limit_hit_nb`."""
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Index must be provided for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Frequency must be provided for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_tsl_th_hit_nb(
    is_position_long: bool,
    init_price: float,
    peak_price: float,
    threshold: float,
    delta_format: int = DeltaFormat.Percent,
) -> bool:
    """Resolve the TSL threshold price using `resolve_stop_price_nb` and check whether it was hit."""
    hit_below = not is_position_long
    tsl_th_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=threshold,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if hit_below:
        return peak_price <= tsl_th_price
    else:
        return peak_price >= tsl_th_price


@register_jitted(cache=True)
def resolve_dyn_limit_price_nb(val_price: float, price: float, limit_price: float) -> float:
    """Resolve price dynamically.

    Uses the valuation price as the left bound and order price as the right bound."""
    if np.isinf(limit_price):
        if limit_price < 0:
            return float(val_price)
        return float(price)
    return float(limit_price)


@register_jitted(cache=True)
def resolve_dyn_stop_entry_price_nb(val_price: float, price: float, stop_entry_price: float) -> float:
    """Resolve stop entry price dynamically.

    Uses the valuation/open price as the left bound and order price as the right bound."""
    if stop_entry_price < 0:
        if stop_entry_price == StopEntryPrice.ValPrice:
            return float(val_price)
        if stop_entry_price == StopEntryPrice.Price:
            return float(price)
        raise ValueError("Only valuation and order price are supported when setting stop entry price dynamically")
    if np.isinf(stop_entry_price):
        if stop_entry_price < 0:
            return float(val_price)
        return float(price)
    return float(stop_entry_price)


@register_jitted(cache=True)
def get_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_price: float,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Get the exit size corresponding to the current step in the ladder."""
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    stop = flex_select_nb(stop_, step, col)
    if np.isnan(stop):
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if not np.isnan(flex_select_nb(stop_, i, col)):
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    hit_below = (init_position >= 0 and hit_below) or (init_position < 0 and not hit_below)
    price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    last_stop = flex_select_nb(stop_, last_step, col)
    last_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=last_stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if step == 0:
        prev_price = init_price
    else:
        prev_stop = flex_select_nb(stop_, step - 1, col)
        prev_price = resolve_stop_price_nb(
            init_price=init_price,
            stop=prev_stop,
            delta_format=delta_format,
            hit_below=hit_below,
        )
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (price - prev_price) / (last_price - init_price)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (price - prev_price) / (last_price - prev_price)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_jitted(cache=True)
def get_time_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_idx: int,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
) -> float:
    """Get the exit size corresponding to the current step in the ladder."""
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    if init_idx == -1:
        raise ValueError("Initial index of the ladder must be known")
    if time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Index must be provided for TimeDeltaFormat.Index")
        init_idx = index[init_idx]
    idx = flex_select_nb(stop_, step, col)
    if idx == -1:
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if flex_select_nb(stop_, i, col) != -1:
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    last_idx = flex_select_nb(stop_, last_step, col)
    if step == 0:
        prev_idx = init_idx
    else:
        prev_idx = flex_select_nb(stop_, step - 1, col)
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (idx - prev_idx) / (last_idx - init_idx)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (idx - prev_idx) / (last_idx - prev_idx)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        init_cash=base_ch.FlexArraySlicer(),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_earnings=base_ch.flex_array_gl_slicer,
        cash_dividends=base_ch.flex_array_gl_slicer,
        long_entries=base_ch.flex_array_gl_slicer,
        long_exits=base_ch.flex_array_gl_slicer,
        short_entries=base_ch.flex_array_gl_slicer,
        short_exits=base_ch.flex_array_gl_slicer,
        size=base_ch.flex_array_gl_slicer,
        price=base_ch.flex_array_gl_slicer,
        size_type=base_ch.flex_array_gl_slicer,
        fees=base_ch.flex_array_gl_slicer,
        fixed_fees=base_ch.flex_array_gl_slicer,
        slippage=base_ch.flex_array_gl_slicer,
        min_size=base_ch.flex_array_gl_slicer,
        max_size=base_ch.flex_array_gl_slicer,
        size_granularity=base_ch.flex_array_gl_slicer,
        leverage=base_ch.flex_array_gl_slicer,
        leverage_mode=base_ch.flex_array_gl_slicer,
        reject_prob=base_ch.flex_array_gl_slicer,
        price_area_vio_mode=base_ch.flex_array_gl_slicer,
        allow_partial=base_ch.flex_array_gl_slicer,
        raise_reject=base_ch.flex_array_gl_slicer,
        log=base_ch.flex_array_gl_slicer,
        val_price=base_ch.flex_array_gl_slicer,
        accumulate=base_ch.flex_array_gl_slicer,
        upon_long_conflict=base_ch.flex_array_gl_slicer,
        upon_short_conflict=base_ch.flex_array_gl_slicer,
        upon_dir_conflict=base_ch.flex_array_gl_slicer,
        upon_opposite_entry=base_ch.flex_array_gl_slicer,
        order_type=base_ch.flex_array_gl_slicer,
        limit_delta=base_ch.flex_array_gl_slicer,
        limit_tif=base_ch.flex_array_gl_slicer,
        limit_expiry=base_ch.flex_array_gl_slicer,
        limit_reverse=base_ch.flex_array_gl_slicer,
        upon_adj_limit_conflict=base_ch.flex_array_gl_slicer,
        upon_opp_limit_conflict=base_ch.flex_array_gl_slicer,
        use_stops=None,
        stop_ladder=None,
        sl_stop=base_ch.flex_array_gl_slicer,
        tsl_stop=base_ch.flex_array_gl_slicer,
        tsl_th=base_ch.flex_array_gl_slicer,
        tp_stop=base_ch.flex_array_gl_slicer,
        td_stop=base_ch.flex_array_gl_slicer,
        dt_stop=base_ch.flex_array_gl_slicer,
        stop_entry_price=base_ch.flex_array_gl_slicer,
        stop_exit_price=base_ch.flex_array_gl_slicer,
        stop_exit_type=base_ch.flex_array_gl_slicer,
        stop_order_type=base_ch.flex_array_gl_slicer,
        stop_limit_delta=base_ch.flex_array_gl_slicer,
        upon_stop_update=base_ch.flex_array_gl_slicer,
        upon_adj_stop_conflict=base_ch.flex_array_gl_slicer,
        upon_opp_stop_conflict=base_ch.flex_array_gl_slicer,
        delta_format=base_ch.flex_array_gl_slicer,
        time_delta_format=base_ch.flex_array_gl_slicer,
        from_ago=base_ch.flex_array_gl_slicer,
        call_seq=base_ch.array_gl_slicer,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        save_state=None,
        save_value=None,
        save_returns=None,
        max_orders=None,
        max_logs=None,
    ),
    **portfolio_ch.merge_sim_outs_config,
)
@register_jitted(cache=True, tags={"can_parallel"})
def from_signals_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    cash_dividends: tp.FlexArray2dLike = 0.0,
    long_entries: tp.FlexArray2dLike = False,
    long_exits: tp.FlexArray2dLike = False,
    short_entries: tp.FlexArray2dLike = False,
    short_exits: tp.FlexArray2dLike = False,
    size: tp.FlexArray2dLike = np.inf,
    price: tp.FlexArray2dLike = np.inf,
    size_type: tp.FlexArray2dLike = SizeType.Amount,
    fees: tp.FlexArray2dLike = 0.0,
    fixed_fees: tp.FlexArray2dLike = 0.0,
    slippage: tp.FlexArray2dLike = 0.0,
    min_size: tp.FlexArray2dLike = np.nan,
    max_size: tp.FlexArray2dLike = np.nan,
    size_granularity: tp.FlexArray2dLike = np.nan,
    leverage: tp.FlexArray2dLike = 1.0,
    leverage_mode: tp.FlexArray2dLike = LeverageMode.Lazy,
    reject_prob: tp.FlexArray2dLike = 0.0,
    price_area_vio_mode: tp.FlexArray2dLike = PriceAreaVioMode.Ignore,
    allow_partial: tp.FlexArray2dLike = True,
    raise_reject: tp.FlexArray2dLike = False,
    log: tp.FlexArray2dLike = False,
    val_price: tp.FlexArray2dLike = np.inf,
    accumulate: tp.FlexArray2dLike = AccumulationMode.Disabled,
    upon_long_conflict: tp.FlexArray2dLike = ConflictMode.Ignore,
    upon_short_conflict: tp.FlexArray2dLike = ConflictMode.Ignore,
    upon_dir_conflict: tp.FlexArray2dLike = DirectionConflictMode.Ignore,
    upon_opposite_entry: tp.FlexArray2dLike = OppositeEntryMode.ReverseReduce,
    order_type: tp.FlexArray2dLike = OrderType.Market,
    limit_delta: tp.FlexArray2dLike = np.nan,
    limit_tif: tp.FlexArray2dLike = -1,
    limit_expiry: tp.FlexArray2dLike = -1,
    limit_reverse: tp.FlexArray2dLike = False,
    upon_adj_limit_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepIgnore,
    upon_opp_limit_conflict: tp.FlexArray2dLike = PendingConflictMode.CancelExecute,
    use_stops: bool = True,
    stop_ladder: int = StopLadderMode.Disabled,
    sl_stop: tp.FlexArray2dLike = np.nan,
    tsl_stop: tp.FlexArray2dLike = np.nan,
    tsl_th: tp.FlexArray2dLike = np.nan,
    tp_stop: tp.FlexArray2dLike = np.nan,
    td_stop: tp.FlexArray2dLike = -1,
    dt_stop: tp.FlexArray2dLike = -1,
    stop_entry_price: tp.FlexArray2dLike = StopEntryPrice.Close,
    stop_exit_price: tp.FlexArray2dLike = StopExitPrice.Stop,
    stop_exit_type: tp.FlexArray2dLike = StopExitType.Close,
    stop_order_type: tp.FlexArray2dLike = OrderType.Market,
    stop_limit_delta: tp.FlexArray2dLike = np.nan,
    upon_stop_update: tp.FlexArray2dLike = StopUpdateMode.Keep,
    upon_adj_stop_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepExecute,
    upon_opp_stop_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepExecute,
    delta_format: tp.FlexArray2dLike = DeltaFormat.Percent,
    time_delta_format: tp.FlexArray2dLike = TimeDeltaFormat.Index,
    from_ago: tp.FlexArray2dLike = 0,
    call_seq: tp.Optional[tp.Array2d] = None,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    save_state: bool = False,
    save_value: bool = False,
    save_returns: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
) -> SimulationOutput:
    """Simulate given signals.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).
    """
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    cash_dividends_ = to_2d_array_nb(np.asarray(cash_dividends))
    long_entries_ = to_2d_array_nb(np.asarray(long_entries))
    long_exits_ = to_2d_array_nb(np.asarray(long_exits))
    short_entries_ = to_2d_array_nb(np.asarray(short_entries))
    short_exits_ = to_2d_array_nb(np.asarray(short_exits))
    size_ = to_2d_array_nb(np.asarray(size))
    price_ = to_2d_array_nb(np.asarray(price))
    size_type_ = to_2d_array_nb(np.asarray(size_type))
    fees_ = to_2d_array_nb(np.asarray(fees))
    fixed_fees_ = to_2d_array_nb(np.asarray(fixed_fees))
    slippage_ = to_2d_array_nb(np.asarray(slippage))
    min_size_ = to_2d_array_nb(np.asarray(min_size))
    max_size_ = to_2d_array_nb(np.asarray(max_size))
    size_granularity_ = to_2d_array_nb(np.asarray(size_granularity))
    leverage_ = to_2d_array_nb(np.asarray(leverage))
    leverage_mode_ = to_2d_array_nb(np.asarray(leverage_mode))
    reject_prob_ = to_2d_array_nb(np.asarray(reject_prob))
    price_area_vio_mode_ = to_2d_array_nb(np.asarray(price_area_vio_mode))
    allow_partial_ = to_2d_array_nb(np.asarray(allow_partial))
    raise_reject_ = to_2d_array_nb(np.asarray(raise_reject))
    log_ = to_2d_array_nb(np.asarray(log))
    val_price_ = to_2d_array_nb(np.asarray(val_price))
    accumulate_ = to_2d_array_nb(np.asarray(accumulate))
    upon_long_conflict_ = to_2d_array_nb(np.asarray(upon_long_conflict))
    upon_short_conflict_ = to_2d_array_nb(np.asarray(upon_short_conflict))
    upon_dir_conflict_ = to_2d_array_nb(np.asarray(upon_dir_conflict))
    upon_opposite_entry_ = to_2d_array_nb(np.asarray(upon_opposite_entry))
    order_type_ = to_2d_array_nb(np.asarray(order_type))
    limit_delta_ = to_2d_array_nb(np.asarray(limit_delta))
    limit_tif_ = to_2d_array_nb(np.asarray(limit_tif))
    limit_expiry_ = to_2d_array_nb(np.asarray(limit_expiry))
    limit_reverse_ = to_2d_array_nb(np.asarray(limit_reverse))
    upon_adj_limit_conflict_ = to_2d_array_nb(np.asarray(upon_adj_limit_conflict))
    upon_opp_limit_conflict_ = to_2d_array_nb(np.asarray(upon_opp_limit_conflict))
    sl_stop_ = to_2d_array_nb(np.asarray(sl_stop))
    tsl_stop_ = to_2d_array_nb(np.asarray(tsl_stop))
    tsl_th_ = to_2d_array_nb(np.asarray(tsl_th))
    tp_stop_ = to_2d_array_nb(np.asarray(tp_stop))
    td_stop_ = to_2d_array_nb(np.asarray(td_stop))
    dt_stop_ = to_2d_array_nb(np.asarray(dt_stop))
    stop_entry_price_ = to_2d_array_nb(np.asarray(stop_entry_price))
    stop_exit_price_ = to_2d_array_nb(np.asarray(stop_exit_price))
    stop_exit_type_ = to_2d_array_nb(np.asarray(stop_exit_type))
    stop_order_type_ = to_2d_array_nb(np.asarray(stop_order_type))
    stop_limit_delta_ = to_2d_array_nb(np.asarray(stop_limit_delta))
    upon_stop_update_ = to_2d_array_nb(np.asarray(upon_stop_update))
    upon_adj_stop_conflict_ = to_2d_array_nb(np.asarray(upon_adj_stop_conflict))
    upon_opp_stop_conflict_ = to_2d_array_nb(np.asarray(upon_opp_stop_conflict))
    delta_format_ = to_2d_array_nb(np.asarray(delta_format))
    time_delta_format_ = to_2d_array_nb(np.asarray(time_delta_format))
    from_ago_ = to_2d_array_nb(np.asarray(from_ago))

    n_sl_steps = sl_stop_.shape[0]
    n_tsl_steps = tsl_stop_.shape[0]
    n_tp_steps = tp_stop_.shape[0]
    n_td_steps = td_stop_.shape[0]
    n_dt_steps = dt_stop_.shape[0]

    if max_orders is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=fs_order_dt)
    else:
        order_records = np.empty((max_orders, target_shape[1]), dtype=fs_order_dt)
    if max_logs is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_logs, target_shape[1]), dtype=log_dt)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    last_cash = prepare_last_cash_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
    )
    last_position = prepare_last_position_nb(
        target_shape=target_shape,
        init_position=init_position_,
    )
    last_value = prepare_last_value_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
    )
    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_locked_cash = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    track_cash_deposits = np.any(cash_deposits_)
    if track_cash_deposits:
        cash_deposits_out = np.full((target_shape[0], len(group_lens)), 0.0, dtype=np.float_)
    else:
        cash_deposits_out = np.full((1, 1), 0.0, dtype=np.float_)
    track_cash_earnings = np.any(cash_earnings_) or np.any(cash_dividends_)
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0.0, dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=np.float_)
    if save_state:
        cash = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)
        position = np.full(target_shape, np.nan, dtype=np.float_)
        debt = np.full(target_shape, np.nan, dtype=np.float_)
        locked_cash = np.full(target_shape, np.nan, dtype=np.float_)
        free_cash = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)
    else:
        cash = np.full((0, 0), np.nan, dtype=np.float_)
        position = np.full((0, 0), np.nan, dtype=np.float_)
        debt = np.full((0, 0), np.nan, dtype=np.float_)
        locked_cash = np.full((0, 0), np.nan, dtype=np.float_)
        free_cash = np.full((0, 0), np.nan, dtype=np.float_)
    if save_value:
        value = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)
    else:
        value = np.full((0, 0), np.nan, dtype=np.float_)
    if save_returns:
        returns = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)
    else:
        returns = np.full((0, 0), np.nan, dtype=np.float_)
    in_outputs = FOInOutputs(
        cash=cash,
        position=position,
        debt=debt,
        locked_cash=locked_cash,
        free_cash=free_cash,
        value=value,
        returns=returns,
    )

    last_limit_info = np.empty(target_shape[1], dtype=limit_info_dt)
    last_limit_info["signal_idx"][:] = -1
    last_limit_info["creation_idx"][:] = -1
    last_limit_info["init_idx"][:] = -1
    last_limit_info["init_price"][:] = np.nan
    last_limit_info["init_size"][:] = np.nan
    last_limit_info["init_size_type"][:] = -1
    last_limit_info["init_direction"][:] = -1
    last_limit_info["init_stop_type"][:] = -1
    last_limit_info["delta"][:] = np.nan
    last_limit_info["delta_format"][:] = -1
    last_limit_info["tif"][:] = -1
    last_limit_info["expiry"][:] = -1
    last_limit_info["time_delta_format"][:] = -1
    last_limit_info["reverse"][:] = False

    if use_stops:
        last_sl_info = np.empty(target_shape[1], dtype=sl_info_dt)
        last_sl_info["init_idx"][:] = -1
        last_sl_info["init_price"][:] = np.nan
        last_sl_info["init_position"][:] = np.nan
        last_sl_info["stop"][:] = np.nan
        last_sl_info["exit_price"][:] = -1
        last_sl_info["exit_size"][:] = np.nan
        last_sl_info["exit_size_type"][:] = -1
        last_sl_info["exit_type"][:] = -1
        last_sl_info["order_type"][:] = -1
        last_sl_info["limit_delta"][:] = np.nan
        last_sl_info["delta_format"][:] = -1
        last_sl_info["ladder"][:] = -1
        last_sl_info["step"][:] = -1
        last_sl_info["step_idx"][:] = -1

        last_tsl_info = np.empty(target_shape[1], dtype=tsl_info_dt)
        last_tsl_info["init_idx"][:] = -1
        last_tsl_info["init_price"][:] = np.nan
        last_tsl_info["init_position"][:] = np.nan
        last_tsl_info["peak_idx"][:] = -1
        last_tsl_info["peak_price"][:] = np.nan
        last_tsl_info["stop"][:] = np.nan
        last_tsl_info["th"][:] = np.nan
        last_tsl_info["exit_price"][:] = -1
        last_tsl_info["exit_size"][:] = np.nan
        last_tsl_info["exit_size_type"][:] = -1
        last_tsl_info["exit_type"][:] = -1
        last_tsl_info["order_type"][:] = -1
        last_tsl_info["limit_delta"][:] = np.nan
        last_tsl_info["delta_format"][:] = -1
        last_tsl_info["ladder"][:] = -1
        last_tsl_info["step"][:] = -1
        last_tsl_info["step_idx"][:] = -1

        last_tp_info = np.empty(target_shape[1], dtype=tp_info_dt)
        last_tp_info["init_idx"][:] = -1
        last_tp_info["init_price"][:] = np.nan
        last_tp_info["init_position"][:] = np.nan
        last_tp_info["stop"][:] = np.nan
        last_tp_info["exit_price"][:] = -1
        last_tp_info["exit_size"][:] = np.nan
        last_tp_info["exit_size_type"][:] = -1
        last_tp_info["exit_type"][:] = -1
        last_tp_info["order_type"][:] = -1
        last_tp_info["limit_delta"][:] = np.nan
        last_tp_info["delta_format"][:] = -1
        last_tp_info["ladder"][:] = -1
        last_tp_info["step"][:] = -1
        last_tp_info["step_idx"][:] = -1

        last_td_info = np.empty(target_shape[1], dtype=time_info_dt)
        last_td_info["init_idx"][:] = -1
        last_td_info["init_position"][:] = np.nan
        last_td_info["stop"][:] = -1
        last_td_info["exit_price"][:] = -1
        last_td_info["exit_size"][:] = np.nan
        last_td_info["exit_size_type"][:] = -1
        last_td_info["exit_type"][:] = -1
        last_td_info["order_type"][:] = -1
        last_td_info["limit_delta"][:] = np.nan
        last_td_info["delta_format"][:] = -1
        last_td_info["time_delta_format"][:] = -1
        last_td_info["ladder"][:] = -1
        last_td_info["step"][:] = -1
        last_td_info["step_idx"][:] = -1

        last_dt_info = np.empty(target_shape[1], dtype=time_info_dt)
        last_dt_info["init_idx"][:] = -1
        last_dt_info["init_position"][:] = np.nan
        last_dt_info["stop"][:] = -1
        last_dt_info["exit_price"][:] = -1
        last_dt_info["exit_size"][:] = np.nan
        last_dt_info["exit_size_type"][:] = -1
        last_dt_info["exit_type"][:] = -1
        last_dt_info["order_type"][:] = -1
        last_dt_info["limit_delta"][:] = np.nan
        last_dt_info["delta_format"][:] = -1
        last_dt_info["time_delta_format"][:] = -1
        last_dt_info["ladder"][:] = -1
        last_dt_info["step"][:] = -1
        last_dt_info["step_idx"][:] = -1
    else:
        last_sl_info = np.empty(0, dtype=sl_info_dt)
        last_tsl_info = np.empty(0, dtype=tsl_info_dt)
        last_tp_info = np.empty(0, dtype=tp_info_dt)
        last_td_info = np.empty(0, dtype=time_info_dt)
        last_dt_info = np.empty(0, dtype=time_info_dt)

    main_info = np.empty(target_shape[1], dtype=main_info_dt)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)
    temp_sort_by = np.empty(target_shape[1], dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        for i in range(target_shape[0]):
            # Add cash
            _cash_deposits = flex_select_nb(cash_deposits_, i, group)
            if _cash_deposits < 0:
                _cash_deposits = max(_cash_deposits, -last_cash[group])
            last_cash[group] += _cash_deposits
            last_free_cash[group] += _cash_deposits
            last_cash_deposits[group] = _cash_deposits
            if track_cash_deposits:
                cash_deposits_out[i, group] += _cash_deposits

            for c in range(group_len):
                col = from_col + c

                # Update valuation price using current open
                _open = flex_select_nb(open_, i, col)
                if not np.isnan(_open) or not ffill_val_price:
                    last_val_price[col] = _open

            # Update value and return
            group_value = last_cash[group]
            for col in range(from_col, to_col):
                if last_position[col] != 0:
                    group_value += last_position[col] * last_val_price[col]
            last_value[group] = group_value
            last_return[group] = returns_nb_.get_return_nb(
                input_value=prev_close_value[group],
                output_value=last_value[group] - _cash_deposits,
            )

            # Get size and value of each order
            for c in range(group_len):
                col = from_col + c  # order doesn't matter

                # Get signals
                _i = i - abs(flex_select_nb(from_ago_, i, col))
                if _i < 0:
                    is_long_entry = False
                    is_long_exit = False
                    is_short_entry = False
                    is_short_exit = False
                else:
                    is_long_entry = flex_select_nb(long_entries_, _i, col)
                    is_long_exit = flex_select_nb(long_exits_, _i, col)
                    is_short_entry = flex_select_nb(short_entries_, _i, col)
                    is_short_exit = flex_select_nb(short_exits_, _i, col)

                # Set defaults
                main_info["bar_zone"][col] = -1
                main_info["signal_idx"][col] = -1
                main_info["creation_idx"][col] = -1
                main_info["idx"][col] = i
                main_info["val_price"][col] = np.nan
                main_info["price"][col] = np.nan
                main_info["size"][col] = np.nan
                main_info["size_type"][col] = -1
                main_info["direction"][col] = -1
                main_info["type"][col] = -1
                main_info["stop_type"][col] = -1
                temp_sort_by[col] = 0.0

                any_limit_signal = is_limit_active_nb(
                    init_idx=last_limit_info["init_idx"][col],
                    init_price=last_limit_info["init_price"][col],
                )
                if use_stops:
                    sl_stop_signal = is_stop_active_nb(
                        init_idx=last_sl_info["init_idx"][col],
                        stop=last_sl_info["stop"][col],
                    )
                    tsl_stop_signal = is_stop_active_nb(
                        init_idx=last_tsl_info["init_idx"][col],
                        stop=last_tsl_info["stop"][col],
                    )
                    tp_stop_signal = is_stop_active_nb(
                        init_idx=last_tp_info["init_idx"][col],
                        stop=last_tp_info["stop"][col],
                    )
                    td_stop_signal = is_time_stop_active_nb(
                        init_idx=last_td_info["init_idx"][col],
                        stop=last_td_info["stop"][col],
                    )
                    dt_stop_signal = is_time_stop_active_nb(
                        init_idx=last_dt_info["init_idx"][col],
                        stop=last_dt_info["stop"][col],
                    )
                    any_stop_signal = (
                        sl_stop_signal or tsl_stop_signal or tp_stop_signal or td_stop_signal or dt_stop_signal
                    )
                else:
                    sl_stop_signal = False
                    tsl_stop_signal = False
                    tp_stop_signal = False
                    td_stop_signal = False
                    dt_stop_signal = False
                    any_stop_signal = False
                any_user_signal = is_long_entry or is_long_exit or is_short_entry or is_short_exit
                if not any_limit_signal and not any_stop_signal and not any_user_signal:  # shortcut
                    continue

                # Set initial info
                exec_limit_set = False
                exec_limit_set_on_open = False
                exec_limit_signal_i = -1
                exec_limit_creation_i = -1
                exec_limit_init_i = -1
                exec_limit_val_price = np.nan
                exec_limit_price = np.nan
                exec_limit_size = np.nan
                exec_limit_size_type = -1
                exec_limit_direction = -1
                exec_limit_stop_type = -1
                exec_limit_bar_zone = -1

                exec_stop_set = False
                exec_stop_set_on_open = False
                exec_stop_set_on_close = False
                exec_stop_init_i = -1
                exec_stop_val_price = np.nan
                exec_stop_price = np.nan
                exec_stop_size = np.nan
                exec_stop_size_type = -1
                exec_stop_direction = -1
                exec_stop_type = -1
                exec_stop_stop_type = -1
                exec_stop_delta = np.nan
                exec_stop_delta_format = -1
                exec_stop_make_limit = False
                exec_stop_bar_zone = -1

                user_on_open = False
                user_on_close = False
                exec_user_set = False
                exec_user_val_price = np.nan
                exec_user_price = np.nan
                exec_user_size = np.nan
                exec_user_size_type = -1
                exec_user_direction = -1
                exec_user_type = -1
                exec_user_stop_type = -1
                exec_user_make_limit = False
                exec_user_bar_zone = -1

                # Resolve the current bar
                _open = flex_select_nb(open_, i, col)
                _high = flex_select_nb(high_, i, col)
                _low = flex_select_nb(low_, i, col)
                _close = flex_select_nb(close_, i, col)
                _high, _low = resolve_hl_nb(
                    open=_open,
                    high=_high,
                    low=_low,
                    close=_close,
                )

                # Process the limit signal
                if any_limit_signal:
                    # Check whether the limit price was hit
                    _signal_i = last_limit_info["signal_idx"][col]
                    _creation_i = last_limit_info["creation_idx"][col]
                    _init_i = last_limit_info["init_idx"][col]
                    _price = last_limit_info["init_price"][col]
                    _size = last_limit_info["init_size"][col]
                    _size_type = last_limit_info["init_size_type"][col]
                    _direction = last_limit_info["init_direction"][col]
                    _stop_type = last_limit_info["init_stop_type"][col]
                    _delta = last_limit_info["delta"][col]
                    _delta_format = last_limit_info["delta_format"][col]
                    _tif = last_limit_info["tif"][col]
                    _expiry = last_limit_info["expiry"][col]
                    _time_delta_format = last_limit_info["time_delta_format"][col]
                    _reverse = last_limit_info["reverse"][col]

                    limit_expired_on_open, limit_expired = check_limit_expired_nb(
                        creation_idx=_creation_i,
                        i=i,
                        tif=_tif,
                        expiry=_expiry,
                        time_delta_format=_time_delta_format,
                        index=index,
                        freq=freq,
                    )
                    limit_price, limit_hit_on_open, limit_hit = check_limit_hit_nb(
                        open=_open,
                        high=_high,
                        low=_low,
                        close=_close,
                        price=_price,
                        size=_size,
                        direction=_direction,
                        limit_delta=_delta,
                        delta_format=_delta_format,
                        limit_reverse=_reverse,
                        can_use_ohlc=True,
                        check_open=True,
                    )
                    if limit_expired_on_open or (not limit_hit_on_open and limit_expired):
                        # Expired limit signal
                        any_limit_signal = False

                        last_limit_info["signal_idx"][col] = -1
                        last_limit_info["creation_idx"][col] = -1
                        last_limit_info["init_idx"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1
                        last_limit_info["reverse"][col] = False
                    else:
                        # Save info
                        if limit_hit:
                            # Executable limit signal
                            exec_limit_set = True
                            exec_limit_set_on_open = limit_hit_on_open
                            exec_limit_signal_i = _signal_i
                            exec_limit_creation_i = _creation_i
                            exec_limit_init_i = _init_i
                            if np.isinf(limit_price) and limit_price > 0:
                                exec_limit_val_price = _close
                            elif np.isinf(limit_price) and limit_price < 0:
                                exec_limit_val_price = _open
                            else:
                                exec_limit_val_price = limit_price
                            exec_limit_price = limit_price
                            exec_limit_size = _size
                            exec_limit_size_type = _size_type
                            exec_limit_direction = _direction
                            exec_limit_stop_type = _stop_type

                # Process the stop signal
                if any_stop_signal:
                    # Check SL
                    sl_stop_price, sl_stop_hit_on_open, sl_stop_hit = np.nan, False, False
                    if sl_stop_signal:
                        # Check against high and low
                        sl_stop_price, sl_stop_hit_on_open, sl_stop_hit = check_stop_hit_nb(
                            open=_open,
                            high=_high,
                            low=_low,
                            close=_close,
                            is_position_long=last_position[col] > 0,
                            init_price=last_sl_info["init_price"][col],
                            stop=last_sl_info["stop"][col],
                            delta_format=last_sl_info["delta_format"][col],
                            hit_below=True,
                        )

                    # Check TSL and TTP
                    tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = np.nan, False, False
                    if tsl_stop_signal:
                        # Update peak price using open
                        if last_position[col] > 0:
                            if _open > last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _open - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _open
                        else:
                            if _open < last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _open - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _open
                        # Check threshold against previous bars and open
                        if np.isnan(last_tsl_info["th"][col]):
                            th_hit = True
                        else:
                            th_hit = check_tsl_th_hit_nb(
                                is_position_long=last_position[col] > 0,
                                init_price=last_tsl_info["init_price"][col],
                                peak_price=last_tsl_info["peak_price"][col],
                                threshold=last_tsl_info["th"][col],
                                delta_format=last_tsl_info["delta_format"][col],
                            )
                        if th_hit:
                            tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = check_stop_hit_nb(
                                open=_open,
                                high=_high,
                                low=_low,
                                close=_close,
                                is_position_long=last_position[col] > 0,
                                init_price=last_tsl_info["peak_price"][col],
                                stop=last_tsl_info["stop"][col],
                                delta_format=last_tsl_info["delta_format"][col],
                                hit_below=True,
                            )
                        # Update peak price using full bar
                        if last_position[col] > 0:
                            if _high > last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _high - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _high
                        else:
                            if _low < last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _low - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _low
                        if not tsl_stop_hit:
                            # Check threshold against full bar
                            if not th_hit:
                                if np.isnan(last_tsl_info["th"][col]):
                                    th_hit = True
                                else:
                                    th_hit = check_tsl_th_hit_nb(
                                        is_position_long=last_position[col] > 0,
                                        init_price=last_tsl_info["init_price"][col],
                                        peak_price=last_tsl_info["peak_price"][col],
                                        threshold=last_tsl_info["th"][col],
                                        delta_format=last_tsl_info["delta_format"][col],
                                    )
                            if th_hit:
                                # Check threshold against close
                                tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = check_stop_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    is_position_long=last_position[col] > 0,
                                    init_price=last_tsl_info["peak_price"][col],
                                    stop=last_tsl_info["stop"][col],
                                    delta_format=last_tsl_info["delta_format"][col],
                                    hit_below=True,
                                    can_use_ohlc=False,
                                )

                    # Check TP
                    tp_stop_price, tp_stop_hit_on_open, tp_stop_hit = np.nan, False, False
                    if tp_stop_signal:
                        tp_stop_price, tp_stop_hit_on_open, tp_stop_hit = check_stop_hit_nb(
                            open=_open,
                            high=_high,
                            low=_low,
                            close=_close,
                            is_position_long=last_position[col] > 0,
                            init_price=last_tp_info["init_price"][col],
                            stop=last_tp_info["stop"][col],
                            delta_format=last_tp_info["delta_format"][col],
                            hit_below=False,
                        )

                    # Check TD
                    td_stop_price, td_stop_hit_on_open, td_stop_hit = np.nan, False, False
                    if td_stop_signal:
                        td_stop_hit_on_open, td_stop_hit = check_td_stop_hit_nb(
                            init_idx=last_td_info["init_idx"][col],
                            i=i,
                            stop=last_td_info["stop"][col],
                            time_delta_format=last_td_info["time_delta_format"][col],
                            index=index,
                            freq=freq,
                        )
                        if np.isnan(_open):
                            td_stop_hit_on_open = False
                        if td_stop_hit_on_open:
                            td_stop_price = _open
                        else:
                            td_stop_price = _close

                    # Check DT
                    dt_stop_price, dt_stop_hit_on_open, dt_stop_hit = np.nan, False, False
                    if dt_stop_signal:
                        dt_stop_hit_on_open, dt_stop_hit = check_dt_stop_hit_nb(
                            i=i,
                            stop=last_dt_info["stop"][col],
                            time_delta_format=last_dt_info["time_delta_format"][col],
                            index=index,
                            freq=freq,
                        )
                        if np.isnan(_open):
                            dt_stop_hit_on_open = False
                        if dt_stop_hit_on_open:
                            dt_stop_price = _open
                        else:
                            dt_stop_price = _close

                    # Resolve the stop signal
                    sl_hit = False
                    tsl_hit = False
                    tp_hit = False
                    td_hit = False
                    dt_hit = False
                    if sl_stop_hit_on_open:
                        sl_hit = True
                    elif tsl_stop_hit_on_open:
                        tsl_hit = True
                    elif tp_stop_hit_on_open:
                        tp_hit = True
                    elif td_stop_hit_on_open:
                        td_hit = True
                    elif dt_stop_hit_on_open:
                        dt_hit = True
                    elif sl_stop_hit:
                        sl_hit = True
                    elif tsl_stop_hit:
                        tsl_hit = True
                    elif tp_stop_hit:
                        tp_hit = True
                    elif td_stop_hit:
                        td_hit = True
                    elif dt_stop_hit:
                        dt_hit = True

                    if sl_hit:
                        stop_price, stop_hit_on_open, stop_hit = sl_stop_price, sl_stop_hit_on_open, sl_stop_hit
                        _stop_type = StopType.SL
                        _init_i = last_sl_info["init_idx"][col]
                        _stop_exit_price = last_sl_info["exit_price"][col]
                        _stop_exit_size = last_sl_info["exit_size"][col]
                        _stop_exit_size_type = last_sl_info["exit_size_type"][col]
                        _stop_exit_type = last_sl_info["exit_type"][col]
                        _stop_order_type = last_sl_info["order_type"][col]
                        _limit_delta = last_sl_info["limit_delta"][col]
                        _delta_format = last_sl_info["delta_format"][col]
                        _ladder = last_sl_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_sl_info["step"][col]
                                if step < n_sl_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=sl_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_sl_info["init_price"][col],
                                        init_position=last_sl_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_sl_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif tsl_hit:
                        stop_price, stop_hit_on_open, stop_hit = tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit
                        if np.isnan(last_tsl_info["th"][col]):
                            _stop_type = StopType.TSL
                        else:
                            _stop_type = StopType.TTP
                        _init_i = last_tsl_info["init_idx"][col]
                        _stop_exit_price = last_tsl_info["exit_price"][col]
                        _stop_exit_size = last_tsl_info["exit_size"][col]
                        _stop_exit_size_type = last_tsl_info["exit_size_type"][col]
                        _stop_exit_type = last_tsl_info["exit_type"][col]
                        _stop_order_type = last_tsl_info["order_type"][col]
                        _limit_delta = last_tsl_info["limit_delta"][col]
                        _delta_format = last_tsl_info["delta_format"][col]
                        _ladder = last_tsl_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_tsl_info["step"][col]
                                if step < n_tsl_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=tsl_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_tsl_info["init_price"][col],
                                        init_position=last_tsl_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_tsl_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif tp_hit:
                        stop_price, stop_hit_on_open, stop_hit = tp_stop_price, tp_stop_hit_on_open, tp_stop_hit
                        _stop_type = StopType.TP
                        _init_i = last_tp_info["init_idx"][col]
                        _stop_exit_price = last_tp_info["exit_price"][col]
                        _stop_exit_size = last_tp_info["exit_size"][col]
                        _stop_exit_size_type = last_tp_info["exit_size_type"][col]
                        _stop_exit_type = last_tp_info["exit_type"][col]
                        _stop_order_type = last_tp_info["order_type"][col]
                        _limit_delta = last_tp_info["limit_delta"][col]
                        _delta_format = last_tp_info["delta_format"][col]
                        _ladder = last_tp_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_tp_info["step"][col]
                                if step < n_tp_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=tp_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_tp_info["init_price"][col],
                                        init_position=last_tp_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_tp_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif td_hit:
                        stop_price, stop_hit_on_open, stop_hit = td_stop_price, td_stop_hit_on_open, td_stop_hit
                        _stop_type = StopType.TD
                        _init_i = last_td_info["init_idx"][col]
                        _stop_exit_price = last_td_info["exit_price"][col]
                        _stop_exit_size = last_td_info["exit_size"][col]
                        _stop_exit_size_type = last_td_info["exit_size_type"][col]
                        _stop_exit_type = last_td_info["exit_type"][col]
                        _stop_order_type = last_td_info["order_type"][col]
                        _limit_delta = last_td_info["limit_delta"][col]
                        _delta_format = last_td_info["delta_format"][col]
                        _ladder = last_td_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_td_info["step"][col]
                                if step < n_td_steps:
                                    _stop_exit_size = get_time_stop_ladder_exit_size_nb(
                                        stop_=td_stop_,
                                        step=step,
                                        col=col,
                                        init_idx=last_td_info["init_idx"][col],
                                        init_position=last_td_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        time_delta_format=last_td_info["time_delta_format"][col],
                                        index=index,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif dt_hit:
                        stop_price, stop_hit_on_open, stop_hit = dt_stop_price, dt_stop_hit_on_open, dt_stop_hit
                        _stop_type = StopType.DT
                        _init_i = last_dt_info["init_idx"][col]
                        _stop_exit_price = last_dt_info["exit_price"][col]
                        _stop_exit_size = last_dt_info["exit_size"][col]
                        _stop_exit_size_type = last_dt_info["exit_size_type"][col]
                        _stop_exit_type = last_dt_info["exit_type"][col]
                        _stop_order_type = last_dt_info["order_type"][col]
                        _limit_delta = last_dt_info["limit_delta"][col]
                        _delta_format = last_dt_info["delta_format"][col]
                        _ladder = last_dt_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_dt_info["step"][col]
                                if step < n_dt_steps:
                                    _stop_exit_size = get_time_stop_ladder_exit_size_nb(
                                        stop_=dt_stop_,
                                        step=step,
                                        col=col,
                                        init_idx=last_dt_info["init_idx"][col],
                                        init_position=last_dt_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        time_delta_format=last_dt_info["time_delta_format"][col],
                                        index=index,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    else:
                        stop_price, stop_hit_on_open, stop_hit = np.nan, False, False

                    if stop_hit:
                        # Stop price was hit
                        # Resolve the final stop signal
                        _accumulate = flex_select_nb(accumulate_, i, col)
                        _size = flex_select_nb(size_, i, col)
                        _size_type = flex_select_nb(size_type_, i, col)
                        if not np.isnan(_stop_exit_size):
                            _accumulate = True
                            if _stop_exit_type == StopExitType.Close:
                                _stop_exit_type = StopExitType.CloseReduce
                            _size = _stop_exit_size
                        if _stop_exit_size_type != -1:
                            _size_type = _stop_exit_size_type

                        (
                            stop_is_long_entry,
                            stop_is_long_exit,
                            stop_is_short_entry,
                            stop_is_short_exit,
                            _accumulate,
                        ) = generate_stop_signal_nb(
                            position_now=last_position[col],
                            stop_exit_type=_stop_exit_type,
                            accumulate=_accumulate,
                        )

                        # Resolve the price
                        _price = resolve_stop_exit_price_nb(
                            stop_price=stop_price,
                            close=_close,
                            stop_exit_price=_stop_exit_price,
                        )

                        # Convert both signals to size (direction-aware), size type, and direction
                        _size, _size_type, _direction = signal_to_size_nb(
                            position_now=last_position[col],
                            val_price_now=_price,
                            value_now=last_value[group],
                            is_long_entry=stop_is_long_entry,
                            is_long_exit=stop_is_long_exit,
                            is_short_entry=stop_is_short_entry,
                            is_short_exit=stop_is_short_exit,
                            size=_size,
                            size_type=_size_type,
                            accumulate=_accumulate,
                        )

                        if not np.isnan(_size):
                            # Executable stop signal
                            can_execute = True
                            if _stop_order_type == OrderType.Limit:
                                # Use close to check whether the limit price was hit
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    price=_price,
                                    size=_size,
                                    direction=_direction,
                                    limit_delta=_limit_delta,
                                    delta_format=_delta_format,
                                    limit_reverse=False,
                                    can_use_ohlc=stop_hit_on_open,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                            # Save info
                            exec_stop_set = True
                            exec_stop_set_on_open = stop_hit_on_open
                            exec_stop_set_on_close = _stop_exit_price == StopExitPrice.Close
                            exec_stop_init_i = _init_i
                            if np.isinf(_price) and _price > 0:
                                exec_stop_val_price = _close
                            elif np.isinf(_price) and _price < 0:
                                exec_stop_val_price = _open
                            else:
                                exec_stop_val_price = _price
                            exec_stop_price = _price
                            exec_stop_size = _size
                            exec_stop_size_type = _size_type
                            exec_stop_direction = _direction
                            exec_stop_type = _stop_order_type
                            exec_stop_stop_type = _stop_type
                            exec_stop_delta = _limit_delta
                            exec_stop_delta_format = _delta_format
                            exec_stop_make_limit = not can_execute

                # Process user signal
                if any_user_signal:
                    if _i < 0:
                        _price = np.nan
                        _size = np.nan
                        _size_type = -1
                        _direction = -1
                    else:
                        _accumulate = flex_select_nb(accumulate_, _i, col)
                        if is_long_entry or is_short_entry:
                            # Resolve any single-direction conflicts
                            _upon_long_conflict = flex_select_nb(upon_long_conflict_, _i, col)
                            is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                                position_now=last_position[col],
                                is_entry=is_long_entry,
                                is_exit=is_long_exit,
                                direction=Direction.LongOnly,
                                conflict_mode=_upon_long_conflict,
                            )
                            _upon_short_conflict = flex_select_nb(upon_short_conflict_, _i, col)
                            is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                                position_now=last_position[col],
                                is_entry=is_short_entry,
                                is_exit=is_short_exit,
                                direction=Direction.ShortOnly,
                                conflict_mode=_upon_short_conflict,
                            )

                            # Resolve any multi-direction conflicts
                            _upon_dir_conflict = flex_select_nb(upon_dir_conflict_, _i, col)
                            is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                                position_now=last_position[col],
                                is_long_entry=is_long_entry,
                                is_short_entry=is_short_entry,
                                upon_dir_conflict=_upon_dir_conflict,
                            )

                            # Resolve an opposite entry
                            _upon_opposite_entry = flex_select_nb(upon_opposite_entry_, _i, col)
                            (
                                is_long_entry,
                                is_long_exit,
                                is_short_entry,
                                is_short_exit,
                                _accumulate,
                            ) = resolve_opposite_entry_nb(
                                position_now=last_position[col],
                                is_long_entry=is_long_entry,
                                is_long_exit=is_long_exit,
                                is_short_entry=is_short_entry,
                                is_short_exit=is_short_exit,
                                upon_opposite_entry=_upon_opposite_entry,
                                accumulate=_accumulate,
                            )

                        # Resolve the price
                        _price = flex_select_nb(price_, _i, col)

                        # Convert both signals to size (direction-aware), size type, and direction
                        _val_price = flex_select_nb(val_price_, i, col)
                        if np.isinf(_val_price) and _val_price > 0:
                            if np.isinf(_price) and _price > 0:
                                _val_price = _close
                            elif np.isinf(_price) and _price < 0:
                                _val_price = _open
                            else:
                                _val_price = _price
                        elif np.isnan(_val_price) or (np.isinf(_val_price) and _val_price < 0):
                            _val_price = last_val_price[col]
                        _size, _size_type, _direction = signal_to_size_nb(
                            position_now=last_position[col],
                            val_price_now=_val_price,
                            value_now=last_value[group],
                            is_long_entry=is_long_entry,
                            is_long_exit=is_long_exit,
                            is_short_entry=is_short_entry,
                            is_short_exit=is_short_exit,
                            size=flex_select_nb(size_, _i, col),
                            size_type=flex_select_nb(size_type_, _i, col),
                            accumulate=_accumulate,
                        )

                    if np.isinf(_price):
                        if _price > 0:
                            user_on_close = True
                        else:
                            user_on_open = True
                    if not np.isnan(_size):
                        # Executable user signal
                        can_execute = True
                        _order_type = flex_select_nb(order_type_, _i, col)
                        if _order_type == OrderType.Limit:
                            # Use close to check whether the limit price was hit
                            can_use_ohlc = False
                            if np.isinf(_price):
                                if _price > 0:
                                    # Cannot place a limit order at the close price and execute right away
                                    _price = _close
                                    can_execute = False
                                else:
                                    can_use_ohlc = True
                                    _price = _open
                            if can_execute:
                                _limit_delta = flex_select_nb(limit_delta_, _i, col)
                                _delta_format = flex_select_nb(delta_format_, _i, col)
                                _limit_reverse = flex_select_nb(limit_reverse_, _i, col)
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    price=_price,
                                    size=_size,
                                    direction=_direction,
                                    limit_delta=_limit_delta,
                                    delta_format=_delta_format,
                                    limit_reverse=_limit_reverse,
                                    can_use_ohlc=can_use_ohlc,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                        # Save info
                        exec_user_set = True
                        exec_user_val_price = _val_price
                        exec_user_price = _price
                        exec_user_size = _size
                        exec_user_size_type = _size_type
                        exec_user_direction = _direction
                        exec_user_type = _order_type
                        exec_user_stop_type = -1
                        exec_user_make_limit = not can_execute

                if (
                    exec_limit_set
                    or exec_stop_set
                    or exec_user_set
                    or ((any_limit_signal or any_stop_signal) and any_user_signal)
                ):
                    # Choose the main executable signal
                    # Priority: limit -> stop -> user

                    # Check whether the main signal comes on open
                    keep_limit = True
                    keep_stop = True
                    execute_limit = False
                    execute_stop = False
                    execute_user = False
                    if exec_limit_set_on_open:
                        keep_limit = False
                        keep_stop = False
                        execute_limit = True
                        exec_limit_bar_zone = BarZone.Open
                    elif exec_stop_set_on_open:
                        keep_limit = False
                        keep_stop = _ladder
                        execute_stop = True
                        if exec_stop_set_on_close:
                            exec_stop_bar_zone = BarZone.Close
                        else:
                            exec_stop_bar_zone = BarZone.Open
                    elif any_user_signal and user_on_open:
                        execute_user = True
                        if any_limit_signal and (execute_user or not exec_user_set):
                            stop_size = get_diraware_size_nb(
                                size=last_limit_info["init_size"][col],
                                direction=last_limit_info["init_direction"][col],
                            )
                            keep_limit, execute_user = resolve_pending_conflict_nb(
                                is_pending_long=stop_size >= 0,
                                is_user_long=is_long_entry or is_short_exit,
                                upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                            )
                        if any_stop_signal and (execute_user or not exec_user_set):
                            keep_stop, execute_user = resolve_pending_conflict_nb(
                                is_pending_long=last_position[col] < 0,
                                is_user_long=is_long_entry or is_short_exit,
                                upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                            )
                        if not exec_user_set:
                            execute_user = False
                        if execute_user:
                            exec_user_bar_zone = BarZone.Open
                    if not execute_limit and not execute_stop and not execute_user:
                        # Check whether the main signal comes in the middle of the bar
                        if exec_limit_set and not exec_limit_set_on_open and keep_limit:
                            keep_limit = False
                            keep_stop = False
                            execute_limit = True
                            exec_limit_bar_zone = BarZone.Middle
                        elif exec_stop_set and not exec_stop_set_on_open and not exec_stop_set_on_close and keep_stop:
                            keep_limit = False
                            keep_stop = _ladder
                            execute_stop = True
                            exec_stop_bar_zone = BarZone.Middle
                        elif any_user_signal and not user_on_open and not user_on_close:
                            execute_user = True
                            if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                stop_size = get_diraware_size_nb(
                                    size=last_limit_info["init_size"][col],
                                    direction=last_limit_info["init_direction"][col],
                                )
                                keep_limit, execute_user = resolve_pending_conflict_nb(
                                    is_pending_long=stop_size >= 0,
                                    is_user_long=is_long_entry or is_short_exit,
                                    upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                    upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                                )
                            if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                keep_stop, execute_user = resolve_pending_conflict_nb(
                                    is_pending_long=last_position[col] < 0,
                                    is_user_long=is_long_entry or is_short_exit,
                                    upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                    upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                                )
                            if not exec_user_set:
                                execute_user = False
                            if execute_user:
                                exec_user_bar_zone = BarZone.Middle
                        if not execute_limit and not execute_stop and not execute_user:
                            # Check whether the main signal comes on close
                            if exec_stop_set_on_close and keep_stop:
                                keep_limit = False
                                keep_stop = _ladder
                                execute_stop = True
                                exec_stop_bar_zone = BarZone.Close
                            elif any_user_signal and user_on_close:
                                execute_user = True
                                if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                    stop_size = get_diraware_size_nb(
                                        size=last_limit_info["init_size"][col],
                                        direction=last_limit_info["init_direction"][col],
                                    )
                                    keep_limit, execute_user = resolve_pending_conflict_nb(
                                        is_pending_long=stop_size >= 0,
                                        is_user_long=is_long_entry or is_short_exit,
                                        upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                        upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                                    )
                                if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                    keep_stop, execute_user = resolve_pending_conflict_nb(
                                        is_pending_long=last_position[col] < 0,
                                        is_user_long=is_long_entry or is_short_exit,
                                        upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                        upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                                    )
                                if not exec_user_set:
                                    execute_user = False
                                if execute_user:
                                    exec_user_bar_zone = BarZone.Close

                    # Process the limit signal
                    if execute_limit:
                        # Execute the signal
                        main_info["bar_zone"][col] = exec_limit_bar_zone
                        main_info["signal_idx"][col] = exec_limit_signal_i
                        main_info["creation_idx"][col] = exec_limit_creation_i
                        main_info["idx"][col] = exec_limit_init_i
                        main_info["val_price"][col] = exec_limit_val_price
                        main_info["price"][col] = exec_limit_price
                        main_info["size"][col] = exec_limit_size
                        main_info["size_type"][col] = exec_limit_size_type
                        main_info["direction"][col] = exec_limit_direction
                        main_info["type"][col] = OrderType.Limit
                        main_info["stop_type"][col] = exec_limit_stop_type
                    if execute_limit or (any_limit_signal and not keep_limit):
                        # Clear the pending info
                        any_limit_signal = False

                        last_limit_info["signal_idx"][col] = -1
                        last_limit_info["creation_idx"][col] = -1
                        last_limit_info["init_idx"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["init_stop_type"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1
                        last_limit_info["reverse"][col] = False

                    # Process the stop signal
                    if execute_stop:
                        # Execute the signal
                        if exec_stop_make_limit:
                            if any_limit_signal:
                                raise ValueError("Only one active limit signal is allowed at a time")

                            _limit_tif = flex_select_nb(limit_tif_, i, col)
                            _limit_expiry = flex_select_nb(limit_expiry_, i, col)
                            _time_delta_format = flex_select_nb(time_delta_format_, i, col)
                            last_limit_info["signal_idx"][col] = exec_stop_init_i
                            last_limit_info["creation_idx"][col] = i
                            last_limit_info["init_idx"][col] = i
                            last_limit_info["init_price"][col] = exec_stop_price
                            last_limit_info["init_size"][col] = exec_stop_size
                            last_limit_info["init_size_type"][col] = exec_stop_size_type
                            last_limit_info["init_direction"][col] = exec_stop_direction
                            last_limit_info["init_stop_type"][col] = exec_stop_stop_type
                            last_limit_info["delta"][col] = exec_stop_delta
                            last_limit_info["delta_format"][col] = exec_stop_delta_format
                            last_limit_info["tif"][col] = _limit_tif
                            last_limit_info["expiry"][col] = _limit_expiry
                            last_limit_info["time_delta_format"][col] = _time_delta_format
                            last_limit_info["reverse"][col] = False
                        else:
                            main_info["bar_zone"][col] = exec_stop_bar_zone
                            main_info["signal_idx"][col] = exec_stop_init_i
                            main_info["creation_idx"][col] = i
                            main_info["idx"][col] = i
                            main_info["val_price"][col] = exec_stop_val_price
                            main_info["price"][col] = exec_stop_price
                            main_info["size"][col] = exec_stop_size
                            main_info["size_type"][col] = exec_stop_size_type
                            main_info["direction"][col] = exec_stop_direction
                            main_info["type"][col] = exec_stop_type
                            main_info["stop_type"][col] = exec_stop_stop_type

                    if any_stop_signal and not keep_stop:
                        # Clear the pending info
                        any_stop_signal = False

                        last_sl_info["init_idx"][col] = -1
                        last_sl_info["init_price"][col] = np.nan
                        last_sl_info["init_position"][col] = np.nan
                        last_sl_info["stop"][col] = np.nan
                        last_sl_info["exit_price"][col] = -1
                        last_sl_info["exit_size"][col] = np.nan
                        last_sl_info["exit_size_type"][col] = -1
                        last_sl_info["exit_type"][col] = -1
                        last_sl_info["order_type"][col] = -1
                        last_sl_info["limit_delta"][col] = np.nan
                        last_sl_info["delta_format"][col] = -1
                        last_sl_info["ladder"][col] = -1
                        last_sl_info["step"][col] = -1
                        last_sl_info["step_idx"][col] = -1

                        last_tsl_info["init_idx"][col] = -1
                        last_tsl_info["init_price"][col] = np.nan
                        last_tsl_info["init_position"][col] = np.nan
                        last_tsl_info["peak_idx"][col] = -1
                        last_tsl_info["peak_price"][col] = np.nan
                        last_tsl_info["stop"][col] = np.nan
                        last_tsl_info["th"][col] = np.nan
                        last_tsl_info["exit_price"][col] = -1
                        last_tsl_info["exit_size"][col] = np.nan
                        last_tsl_info["exit_size_type"][col] = -1
                        last_tsl_info["exit_type"][col] = -1
                        last_tsl_info["order_type"][col] = -1
                        last_tsl_info["limit_delta"][col] = np.nan
                        last_tsl_info["delta_format"][col] = -1
                        last_tsl_info["ladder"][col] = -1
                        last_tsl_info["step"][col] = -1
                        last_tsl_info["step_idx"][col] = -1

                        last_tp_info["init_idx"][col] = -1
                        last_tp_info["init_price"][col] = np.nan
                        last_tp_info["init_position"][col] = np.nan
                        last_tp_info["stop"][col] = np.nan
                        last_tp_info["exit_price"][col] = -1
                        last_tp_info["exit_size"][col] = np.nan
                        last_tp_info["exit_size_type"][col] = -1
                        last_tp_info["exit_type"][col] = -1
                        last_tp_info["order_type"][col] = -1
                        last_tp_info["limit_delta"][col] = np.nan
                        last_tp_info["delta_format"][col] = -1
                        last_tp_info["ladder"][col] = -1
                        last_tp_info["step"][col] = -1
                        last_tp_info["step_idx"][col] = -1

                        last_td_info["init_idx"][col] = -1
                        last_td_info["init_position"][col] = np.nan
                        last_td_info["stop"][col] = -1
                        last_td_info["exit_price"][col] = -1
                        last_td_info["exit_size"][col] = np.nan
                        last_td_info["exit_size_type"][col] = -1
                        last_td_info["exit_type"][col] = -1
                        last_td_info["order_type"][col] = -1
                        last_td_info["limit_delta"][col] = np.nan
                        last_td_info["delta_format"][col] = -1
                        last_td_info["time_delta_format"][col] = -1
                        last_td_info["ladder"][col] = -1
                        last_td_info["step"][col] = -1
                        last_td_info["step_idx"][col] = -1

                        last_dt_info["init_idx"][col] = -1
                        last_dt_info["init_position"][col] = np.nan
                        last_dt_info["stop"][col] = -1
                        last_dt_info["exit_price"][col] = -1
                        last_dt_info["exit_size"][col] = np.nan
                        last_dt_info["exit_size_type"][col] = -1
                        last_dt_info["exit_type"][col] = -1
                        last_dt_info["order_type"][col] = -1
                        last_dt_info["limit_delta"][col] = np.nan
                        last_dt_info["delta_format"][col] = -1
                        last_dt_info["time_delta_format"][col] = -1
                        last_dt_info["ladder"][col] = -1
                        last_dt_info["step"][col] = -1
                        last_dt_info["step_idx"][col] = -1

                    # Process the user signal
                    if execute_user:
                        # Execute the signal
                        if _i >= 0:
                            if exec_user_make_limit:
                                if any_limit_signal:
                                    raise ValueError("Only one active limit signal is allowed at a time")

                                _limit_delta = flex_select_nb(limit_delta_, _i, col)
                                _delta_format = flex_select_nb(delta_format_, _i, col)
                                _limit_tif = flex_select_nb(limit_tif_, _i, col)
                                _limit_expiry = flex_select_nb(limit_expiry_, _i, col)
                                _time_delta_format = flex_select_nb(time_delta_format_, _i, col)
                                _limit_reverse = flex_select_nb(limit_reverse_, _i, col)
                                last_limit_info["signal_idx"][col] = _i
                                last_limit_info["creation_idx"][col] = i
                                last_limit_info["init_idx"][col] = _i
                                last_limit_info["init_price"][col] = exec_user_price
                                last_limit_info["init_size"][col] = exec_user_size
                                last_limit_info["init_size_type"][col] = exec_user_size_type
                                last_limit_info["init_direction"][col] = exec_user_direction
                                last_limit_info["init_stop_type"][col] = -1
                                last_limit_info["delta"][col] = _limit_delta
                                last_limit_info["delta_format"][col] = _delta_format
                                last_limit_info["tif"][col] = _limit_tif
                                last_limit_info["expiry"][col] = _limit_expiry
                                last_limit_info["time_delta_format"][col] = _time_delta_format
                                last_limit_info["reverse"][col] = _limit_reverse
                            else:
                                main_info["bar_zone"][col] = exec_user_bar_zone
                                main_info["signal_idx"][col] = _i
                                main_info["creation_idx"][col] = i
                                main_info["idx"][col] = _i
                                main_info["val_price"][col] = exec_user_val_price
                                main_info["price"][col] = exec_user_price
                                main_info["size"][col] = exec_user_size
                                main_info["size_type"][col] = exec_user_size_type
                                main_info["direction"][col] = exec_user_direction
                                main_info["type"][col] = exec_user_type
                                main_info["stop_type"][col] = exec_user_stop_type

            any_signal_set = False
            for col in range(from_col, to_col):
                if not np.isnan(main_info["size"][col]):
                    any_signal_set = True
                    break
            if any_signal_set:
                # Check bar zone and update valuation price
                bar_zone = -1
                same_bar_zone = True
                same_timing = True
                for c in range(group_len):
                    col = from_col + c
                    if np.isnan(main_info["size"][col]):
                        continue
                    if bar_zone == -1:
                        bar_zone = main_info["bar_zone"][col]
                    if main_info["bar_zone"][col] != bar_zone:
                        same_bar_zone = False
                        same_timing = False
                    if main_info["bar_zone"][col] == BarZone.Middle:
                        same_timing = False
                    _val_price = main_info["val_price"][col]
                    if not np.isnan(_val_price) or not ffill_val_price:
                        last_val_price[col] = _val_price

                if cash_sharing:
                    # Dynamically sort by order value -> selling comes first to release funds early
                    if call_seq is None:
                        for c in range(group_len):
                            temp_call_seq[c] = c
                        call_seq_now = temp_call_seq[:group_len]
                    else:
                        call_seq_now = call_seq[i, from_col:to_col]
                    if auto_call_seq:
                        # Sort by order value
                        if not same_timing:
                            raise ValueError("Cannot sort orders by value if they are executed at different times")
                        for c in range(group_len):
                            if call_seq_now[c] != c:
                                raise ValueError("Call sequence must follow CallSeqType.Default")
                            col = from_col + c
                            if np.isnan(main_info["size"][col]):
                                continue
                            # Approximate order value
                            exec_state = ExecState(
                                cash=last_cash[group] if cash_sharing else last_cash[col],
                                position=last_position[col],
                                debt=last_debt[col],
                                locked_cash=last_locked_cash[col],
                                free_cash=last_free_cash[group] if cash_sharing else last_free_cash[col],
                                val_price=last_val_price[col],
                                value=last_value[group] if cash_sharing else last_value[col],
                            )
                            temp_sort_by[c] = approx_order_value_nb(
                                exec_state=exec_state,
                                size=main_info["size"][col],
                                size_type=main_info["size_type"][col],
                                direction=main_info["direction"][col],
                            )
                        insert_argsort_nb(temp_sort_by[:group_len], call_seq_now)
                    else:
                        if not same_bar_zone:
                            # Sort by bar zone
                            for c in range(group_len):
                                if call_seq_now[c] != c:
                                    raise ValueError("Call sequence must follow CallSeqType.Default")
                                col = from_col + c
                                if np.isnan(main_info["size"][col]):
                                    continue
                                temp_sort_by[c] = main_info["bar_zone"][col]
                            insert_argsort_nb(temp_sort_by[:group_len], call_seq_now)

                for k in range(group_len):
                    if cash_sharing:
                        c = call_seq_now[k]
                        if c >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        c = k
                    col = from_col + c
                    if np.isnan(main_info["size"][col]):  # shortcut
                        continue

                    # Get current values per column
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    cash_now = last_cash[group]
                    free_cash_now = last_free_cash[group]
                    value_now = last_value[group]
                    return_now = last_return[group]

                    # Generate the next order
                    _i = main_info["idx"][col]
                    if main_info["type"][col] == OrderType.Limit:
                        _slippage = 0.0
                    else:
                        _slippage = flex_select_nb(slippage_, _i, col)
                    order = order_nb(
                        size=main_info["size"][col],
                        price=main_info["price"][col],
                        size_type=main_info["size_type"][col],
                        direction=main_info["direction"][col],
                        fees=flex_select_nb(fees_, _i, col),
                        fixed_fees=flex_select_nb(fixed_fees_, _i, col),
                        slippage=_slippage,
                        min_size=flex_select_nb(min_size_, _i, col),
                        max_size=flex_select_nb(max_size_, _i, col),
                        size_granularity=flex_select_nb(size_granularity_, _i, col),
                        leverage=flex_select_nb(leverage_, _i, col),
                        leverage_mode=flex_select_nb(leverage_mode_, _i, col),
                        reject_prob=flex_select_nb(reject_prob_, _i, col),
                        price_area_vio_mode=flex_select_nb(price_area_vio_mode_, _i, col),
                        allow_partial=flex_select_nb(allow_partial_, _i, col),
                        raise_reject=flex_select_nb(raise_reject_, _i, col),
                        log=flex_select_nb(log_, _i, col),
                    )

                    # Process the order
                    price_area = PriceArea(
                        open=flex_select_nb(open_, i, col),
                        high=flex_select_nb(high_, i, col),
                        low=flex_select_nb(low_, i, col),
                        close=flex_select_nb(close_, i, col),
                    )
                    exec_state = ExecState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        locked_cash=locked_cash_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                    )
                    order_result, new_exec_state = process_order_nb(
                        group=group,
                        col=col,
                        i=i,
                        exec_state=exec_state,
                        order=order,
                        price_area=price_area,
                        update_value=update_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                    )

                    # Append more order information
                    if order_result.status == OrderStatus.Filled:
                        if order_counts[col] >= 1:
                            order_records["signal_idx"][order_counts[col] - 1, col] = main_info["signal_idx"][col]
                            order_records["creation_idx"][order_counts[col] - 1, col] = main_info["creation_idx"][col]
                            order_records["type"][order_counts[col] - 1, col] = main_info["type"][col]
                            order_records["stop_type"][order_counts[col] - 1, col] = main_info["stop_type"][col]

                    # Update execution state
                    cash_now = new_exec_state.cash
                    position_now = new_exec_state.position
                    debt_now = new_exec_state.debt
                    locked_cash_now = new_exec_state.locked_cash
                    free_cash_now = new_exec_state.free_cash
                    val_price_now = new_exec_state.val_price
                    value_now = new_exec_state.value

                    if use_stops:
                        # Update stop price
                        if position_now == 0:
                            # Not in position anymore -> clear stops (irrespective of order success)
                            last_sl_info["init_idx"][col] = -1
                            last_sl_info["init_price"][col] = np.nan
                            last_sl_info["init_position"][col] = np.nan
                            last_sl_info["stop"][col] = np.nan
                            last_sl_info["exit_price"][col] = -1
                            last_sl_info["exit_size"][col] = np.nan
                            last_sl_info["exit_size_type"][col] = -1
                            last_sl_info["exit_type"][col] = -1
                            last_sl_info["order_type"][col] = -1
                            last_sl_info["limit_delta"][col] = np.nan
                            last_sl_info["delta_format"][col] = -1
                            last_sl_info["ladder"][col] = -1
                            last_sl_info["step"][col] = -1
                            last_sl_info["step_idx"][col] = -1

                            last_tsl_info["init_idx"][col] = -1
                            last_tsl_info["init_price"][col] = np.nan
                            last_tsl_info["init_position"][col] = np.nan
                            last_tsl_info["peak_idx"][col] = -1
                            last_tsl_info["peak_price"][col] = np.nan
                            last_tsl_info["stop"][col] = np.nan
                            last_tsl_info["th"][col] = np.nan
                            last_tsl_info["exit_price"][col] = -1
                            last_tsl_info["exit_size"][col] = np.nan
                            last_tsl_info["exit_size_type"][col] = -1
                            last_tsl_info["exit_type"][col] = -1
                            last_tsl_info["order_type"][col] = -1
                            last_tsl_info["limit_delta"][col] = np.nan
                            last_tsl_info["delta_format"][col] = -1
                            last_tsl_info["ladder"][col] = -1
                            last_tsl_info["step"][col] = -1
                            last_tsl_info["step_idx"][col] = -1

                            last_tp_info["init_idx"][col] = -1
                            last_tp_info["init_price"][col] = np.nan
                            last_tp_info["init_position"][col] = np.nan
                            last_tp_info["stop"][col] = np.nan
                            last_tp_info["exit_price"][col] = -1
                            last_tp_info["exit_size"][col] = np.nan
                            last_tp_info["exit_size_type"][col] = -1
                            last_tp_info["exit_type"][col] = -1
                            last_tp_info["order_type"][col] = -1
                            last_tp_info["limit_delta"][col] = np.nan
                            last_tp_info["delta_format"][col] = -1
                            last_tp_info["ladder"][col] = -1
                            last_tp_info["step"][col] = -1
                            last_tp_info["step_idx"][col] = -1

                            last_td_info["init_idx"][col] = -1
                            last_td_info["init_position"][col] = np.nan
                            last_td_info["stop"][col] = -1
                            last_td_info["exit_price"][col] = -1
                            last_td_info["exit_size"][col] = np.nan
                            last_td_info["exit_size_type"][col] = -1
                            last_td_info["exit_type"][col] = -1
                            last_td_info["order_type"][col] = -1
                            last_td_info["limit_delta"][col] = np.nan
                            last_td_info["delta_format"][col] = -1
                            last_td_info["time_delta_format"][col] = -1
                            last_td_info["ladder"][col] = -1
                            last_td_info["step"][col] = -1
                            last_td_info["step_idx"][col] = -1

                            last_dt_info["init_idx"][col] = -1
                            last_dt_info["init_position"][col] = np.nan
                            last_dt_info["stop"][col] = -1
                            last_dt_info["exit_price"][col] = -1
                            last_dt_info["exit_size"][col] = np.nan
                            last_dt_info["exit_size_type"][col] = -1
                            last_dt_info["exit_type"][col] = -1
                            last_dt_info["order_type"][col] = -1
                            last_dt_info["limit_delta"][col] = np.nan
                            last_dt_info["delta_format"][col] = -1
                            last_dt_info["time_delta_format"][col] = -1
                            last_dt_info["ladder"][col] = -1
                            last_dt_info["step"][col] = -1
                            last_dt_info["step_idx"][col] = -1
                        else:
                            if main_info["stop_type"][col] == StopType.SL:
                                if last_sl_info["ladder"][col]:
                                    step = last_sl_info["step"][col] + 1
                                    last_sl_info["exit_size"][col] = np.nan
                                    last_sl_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_sl_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_sl_steps:
                                            last_sl_info["stop"][col] = flex_select_nb(sl_stop_, step, col)
                                            last_sl_info["step"][col] = step
                                            last_sl_info["step_idx"][col] = i
                                        else:
                                            last_sl_info["stop"][col] = np.nan
                                            last_sl_info["step"][col] = -1
                                            last_sl_info["step_idx"][col] = -1
                                    else:
                                        last_sl_info["stop"][col] = np.nan
                                        last_sl_info["step"][col] = step
                                        last_sl_info["step_idx"][col] = i
                            elif (
                                main_info["stop_type"][col] == StopType.TSL
                                or main_info["stop_type"][col] == StopType.TTP
                            ):
                                if last_tsl_info["ladder"][col]:
                                    step = last_tsl_info["step"][col] + 1
                                    last_tsl_info["step"][col] = step
                                    last_tsl_info["step_idx"][col] = i
                                    last_tsl_info["exit_size"][col] = np.nan
                                    last_tsl_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_tsl_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_tsl_steps:
                                            last_tsl_info["stop"][col] = flex_select_nb(tsl_stop_, step, col)
                                            last_tsl_info["step"][col] = step
                                            last_tsl_info["step_idx"][col] = i
                                        else:
                                            last_tsl_info["stop"][col] = np.nan
                                            last_tsl_info["step"][col] = -1
                                            last_tsl_info["step_idx"][col] = -1
                                    else:
                                        last_tsl_info["stop"][col] = np.nan
                                        last_tsl_info["step"][col] = step
                                        last_tsl_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.TP:
                                if last_tp_info["ladder"][col]:
                                    step = last_tp_info["step"][col] + 1
                                    last_tp_info["step"][col] = step
                                    last_tp_info["step_idx"][col] = i
                                    last_tp_info["exit_size"][col] = np.nan
                                    last_tp_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_tp_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_tp_steps:
                                            last_tp_info["stop"][col] = flex_select_nb(tp_stop_, step, col)
                                            last_tp_info["step"][col] = step
                                            last_tp_info["step_idx"][col] = i
                                        else:
                                            last_tp_info["stop"][col] = np.nan
                                            last_tp_info["step"][col] = -1
                                            last_tp_info["step_idx"][col] = -1
                                    else:
                                        last_tp_info["stop"][col] = np.nan
                                        last_tp_info["step"][col] = step
                                        last_tp_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.TD:
                                if last_td_info["ladder"][col]:
                                    step = last_td_info["step"][col] + 1
                                    last_td_info["step"][col] = step
                                    last_td_info["step_idx"][col] = i
                                    last_td_info["exit_size"][col] = np.nan
                                    last_td_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_td_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_td_steps:
                                            last_td_info["stop"][col] = flex_select_nb(td_stop_, step, col)
                                            last_td_info["step"][col] = step
                                            last_td_info["step_idx"][col] = i
                                        else:
                                            last_td_info["stop"][col] = -1
                                            last_td_info["step"][col] = -1
                                            last_td_info["step_idx"][col] = -1
                                    else:
                                        last_td_info["stop"][col] = -1
                                        last_td_info["step"][col] = step
                                        last_td_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.DT:
                                if last_dt_info["ladder"][col]:
                                    step = last_dt_info["step"][col] + 1
                                    last_dt_info["step"][col] = step
                                    last_dt_info["step_idx"][col] = i
                                    last_dt_info["exit_size"][col] = np.nan
                                    last_dt_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_dt_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_dt_steps:
                                            last_dt_info["stop"][col] = flex_select_nb(dt_stop_, step, col)
                                            last_dt_info["step"][col] = step
                                            last_dt_info["step_idx"][col] = i
                                        else:
                                            last_dt_info["stop"][col] = -1
                                            last_dt_info["step"][col] = -1
                                            last_dt_info["step_idx"][col] = -1
                                    else:
                                        last_dt_info["stop"][col] = -1
                                        last_dt_info["step"][col] = step
                                        last_dt_info["step_idx"][col] = i

                        if order_result.status == OrderStatus.Filled and position_now != 0:
                            # Order filled and in position -> possibly set stops
                            _price = main_info["price"][col]
                            _stop_entry_price = flex_select_nb(stop_entry_price_, i, col)
                            if _stop_entry_price < 0:
                                if _stop_entry_price == StopEntryPrice.ValPrice:
                                    new_init_price = val_price_now
                                    can_use_ohlc = False
                                elif _stop_entry_price == StopEntryPrice.Price:
                                    new_init_price = order.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                    if np.isinf(new_init_price):
                                        if new_init_price > 0:
                                            new_init_price = flex_select_nb(close_, i, col)
                                        else:
                                            new_init_price = flex_select_nb(open_, i, col)
                                elif _stop_entry_price == StopEntryPrice.FillPrice:
                                    new_init_price = order_result.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                elif _stop_entry_price == StopEntryPrice.Open:
                                    new_init_price = flex_select_nb(open_, i, col)
                                    can_use_ohlc = True
                                elif _stop_entry_price == StopEntryPrice.Close:
                                    new_init_price = flex_select_nb(close_, i, col)
                                    can_use_ohlc = False
                                else:
                                    raise ValueError("Invalid StopEntryPrice option")
                            else:
                                new_init_price = _stop_entry_price
                                can_use_ohlc = False

                            if stop_ladder:
                                _sl_stop = flex_select_nb(sl_stop_, 0, col)
                                _tsl_stop = flex_select_nb(tsl_stop_, 0, col)
                                _tp_stop = flex_select_nb(tp_stop_, 0, col)
                                _td_stop = flex_select_nb(td_stop_, 0, col)
                                _dt_stop = flex_select_nb(dt_stop_, 0, col)
                            else:
                                _sl_stop = flex_select_nb(sl_stop_, i, col)
                                _tsl_stop = flex_select_nb(tsl_stop_, i, col)
                                _tp_stop = flex_select_nb(tp_stop_, i, col)
                                _td_stop = flex_select_nb(td_stop_, i, col)
                                _dt_stop = flex_select_nb(dt_stop_, i, col)
                            _tsl_th = flex_select_nb(tsl_th_, i, col)
                            _stop_exit_price = flex_select_nb(stop_exit_price_, i, col)
                            _stop_exit_type = flex_select_nb(stop_exit_type_, i, col)
                            _stop_order_type = flex_select_nb(stop_order_type_, i, col)
                            _stop_limit_delta = flex_select_nb(stop_limit_delta_, i, col)
                            _delta_format = flex_select_nb(delta_format_, i, col)
                            _time_delta_format = flex_select_nb(time_delta_format_, i, col)

                            tsl_updated = False
                            if exec_state.position == 0 or np.sign(position_now) != np.sign(exec_state.position):
                                # Position opened/reversed -> set stops
                                last_sl_info["init_idx"][col] = i
                                last_sl_info["init_price"][col] = new_init_price
                                last_sl_info["init_position"][col] = position_now
                                last_sl_info["stop"][col] = _sl_stop
                                last_sl_info["exit_price"][col] = _stop_exit_price
                                last_sl_info["exit_size"][col] = np.nan
                                last_sl_info["exit_size_type"][col] = -1
                                last_sl_info["exit_type"][col] = _stop_exit_type
                                last_sl_info["order_type"][col] = _stop_order_type
                                last_sl_info["limit_delta"][col] = _stop_limit_delta
                                last_sl_info["delta_format"][col] = _delta_format
                                last_sl_info["ladder"][col] = stop_ladder
                                last_sl_info["step"][col] = 0
                                last_sl_info["step_idx"][col] = i

                                tsl_updated = True
                                last_tsl_info["init_idx"][col] = i
                                last_tsl_info["init_price"][col] = new_init_price
                                last_tsl_info["init_position"][col] = position_now
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = new_init_price
                                last_tsl_info["stop"][col] = _tsl_stop
                                last_tsl_info["th"][col] = _tsl_th
                                last_tsl_info["exit_price"][col] = _stop_exit_price
                                last_tsl_info["exit_size"][col] = np.nan
                                last_tsl_info["exit_size_type"][col] = -1
                                last_tsl_info["exit_type"][col] = _stop_exit_type
                                last_tsl_info["order_type"][col] = _stop_order_type
                                last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                last_tsl_info["delta_format"][col] = _delta_format
                                last_tsl_info["ladder"][col] = stop_ladder
                                last_tsl_info["step"][col] = 0
                                last_tsl_info["step_idx"][col] = i

                                last_tp_info["init_idx"][col] = i
                                last_tp_info["init_price"][col] = new_init_price
                                last_tp_info["init_position"][col] = position_now
                                last_tp_info["stop"][col] = _tp_stop
                                last_tp_info["exit_price"][col] = _stop_exit_price
                                last_tp_info["exit_size"][col] = np.nan
                                last_tp_info["exit_size_type"][col] = -1
                                last_tp_info["exit_type"][col] = _stop_exit_type
                                last_tp_info["order_type"][col] = _stop_order_type
                                last_tp_info["limit_delta"][col] = _stop_limit_delta
                                last_tp_info["delta_format"][col] = _delta_format
                                last_tp_info["ladder"][col] = stop_ladder
                                last_tp_info["step"][col] = 0
                                last_tp_info["step_idx"][col] = i

                                last_td_info["init_idx"][col] = i
                                last_td_info["init_position"][col] = position_now
                                last_td_info["stop"][col] = _td_stop
                                last_td_info["exit_price"][col] = _stop_exit_price
                                last_td_info["exit_size"][col] = np.nan
                                last_td_info["exit_size_type"][col] = -1
                                last_td_info["exit_type"][col] = _stop_exit_type
                                last_td_info["order_type"][col] = _stop_order_type
                                last_td_info["limit_delta"][col] = _stop_limit_delta
                                last_td_info["delta_format"][col] = _delta_format
                                last_td_info["time_delta_format"][col] = _time_delta_format
                                last_td_info["ladder"][col] = stop_ladder
                                last_td_info["step"][col] = 0
                                last_td_info["step_idx"][col] = i

                                last_dt_info["init_idx"][col] = i
                                last_dt_info["init_position"][col] = position_now
                                last_dt_info["stop"][col] = _dt_stop
                                last_dt_info["exit_price"][col] = _stop_exit_price
                                last_dt_info["exit_size"][col] = np.nan
                                last_dt_info["exit_size_type"][col] = -1
                                last_dt_info["exit_type"][col] = _stop_exit_type
                                last_dt_info["order_type"][col] = _stop_order_type
                                last_dt_info["limit_delta"][col] = _stop_limit_delta
                                last_dt_info["delta_format"][col] = _delta_format
                                last_dt_info["time_delta_format"][col] = _time_delta_format
                                last_dt_info["ladder"][col] = stop_ladder
                                last_dt_info["step"][col] = 0
                                last_dt_info["step_idx"][col] = i

                            elif abs(position_now) > abs(exec_state.position):
                                # Position increased -> keep/override stops
                                _upon_stop_update = flex_select_nb(upon_stop_update_, i, col)
                                if should_update_stop_nb(new_stop=_sl_stop, upon_stop_update=_upon_stop_update):
                                    last_sl_info["init_idx"][col] = i
                                    last_sl_info["init_price"][col] = new_init_price
                                    last_sl_info["init_position"][col] = position_now
                                    last_sl_info["stop"][col] = _sl_stop
                                    last_sl_info["exit_price"][col] = _stop_exit_price
                                    last_sl_info["exit_size"][col] = np.nan
                                    last_sl_info["exit_size_type"][col] = -1
                                    last_sl_info["exit_type"][col] = _stop_exit_type
                                    last_sl_info["order_type"][col] = _stop_order_type
                                    last_sl_info["limit_delta"][col] = _stop_limit_delta
                                    last_sl_info["delta_format"][col] = _delta_format
                                    last_sl_info["ladder"][col] = stop_ladder
                                    last_sl_info["step"][col] = 0
                                    last_sl_info["step_idx"][col] = i
                                if should_update_stop_nb(new_stop=_tsl_stop, upon_stop_update=_upon_stop_update):
                                    tsl_updated = True
                                    last_tsl_info["init_idx"][col] = i
                                    last_tsl_info["init_price"][col] = new_init_price
                                    last_tsl_info["init_position"][col] = position_now
                                    last_tsl_info["peak_idx"][col] = i
                                    last_tsl_info["peak_price"][col] = new_init_price
                                    last_tsl_info["stop"][col] = _tsl_stop
                                    last_tsl_info["th"][col] = _tsl_th
                                    last_tsl_info["exit_price"][col] = _stop_exit_price
                                    last_tsl_info["exit_size"][col] = np.nan
                                    last_tsl_info["exit_size_type"][col] = -1
                                    last_tsl_info["exit_type"][col] = _stop_exit_type
                                    last_tsl_info["order_type"][col] = _stop_order_type
                                    last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                    last_tsl_info["delta_format"][col] = _delta_format
                                    last_tsl_info["ladder"][col] = stop_ladder
                                    last_tsl_info["step"][col] = 0
                                    last_tsl_info["step_idx"][col] = i
                                if should_update_stop_nb(new_stop=_tp_stop, upon_stop_update=_upon_stop_update):
                                    last_tp_info["init_idx"][col] = i
                                    last_tp_info["init_price"][col] = new_init_price
                                    last_tp_info["init_position"][col] = position_now
                                    last_tp_info["stop"][col] = _tp_stop
                                    last_tp_info["exit_price"][col] = _stop_exit_price
                                    last_tp_info["exit_size"][col] = np.nan
                                    last_tp_info["exit_size_type"][col] = -1
                                    last_tp_info["exit_type"][col] = _stop_exit_type
                                    last_tp_info["order_type"][col] = _stop_order_type
                                    last_tp_info["limit_delta"][col] = _stop_limit_delta
                                    last_tp_info["delta_format"][col] = _delta_format
                                    last_tp_info["ladder"][col] = stop_ladder
                                    last_tp_info["step"][col] = 0
                                    last_tp_info["step_idx"][col] = i
                                if should_update_time_stop_nb(new_stop=_td_stop, upon_stop_update=_upon_stop_update):
                                    last_td_info["init_idx"][col] = i
                                    last_td_info["init_position"][col] = position_now
                                    last_td_info["stop"][col] = _td_stop
                                    last_td_info["exit_price"][col] = _stop_exit_price
                                    last_td_info["exit_size"][col] = np.nan
                                    last_td_info["exit_size_type"][col] = -1
                                    last_td_info["exit_type"][col] = _stop_exit_type
                                    last_td_info["order_type"][col] = _stop_order_type
                                    last_td_info["limit_delta"][col] = _stop_limit_delta
                                    last_td_info["delta_format"][col] = _delta_format
                                    last_td_info["time_delta_format"][col] = _time_delta_format
                                    last_td_info["ladder"][col] = stop_ladder
                                    last_td_info["step"][col] = 0
                                    last_td_info["step_idx"][col] = i
                                if should_update_time_stop_nb(new_stop=_dt_stop, upon_stop_update=_upon_stop_update):
                                    last_dt_info["init_idx"][col] = i
                                    last_dt_info["init_position"][col] = position_now
                                    last_dt_info["stop"][col] = _dt_stop
                                    last_dt_info["exit_price"][col] = _stop_exit_price
                                    last_dt_info["exit_size"][col] = np.nan
                                    last_dt_info["exit_size_type"][col] = -1
                                    last_dt_info["exit_type"][col] = _stop_exit_type
                                    last_dt_info["order_type"][col] = _stop_order_type
                                    last_dt_info["limit_delta"][col] = _stop_limit_delta
                                    last_dt_info["delta_format"][col] = _delta_format
                                    last_dt_info["time_delta_format"][col] = _time_delta_format
                                    last_dt_info["ladder"][col] = stop_ladder
                                    last_dt_info["step"][col] = 0
                                    last_dt_info["step_idx"][col] = i

                            if tsl_updated:
                                # Update highest/lowest price
                                if can_use_ohlc:
                                    _open = flex_select_nb(open_, i, col)
                                    _high = flex_select_nb(high_, i, col)
                                    _low = flex_select_nb(low_, i, col)
                                    _close = flex_select_nb(close_, i, col)
                                    _high, _low = resolve_hl_nb(
                                        open=_open,
                                        high=_high,
                                        low=_low,
                                        close=_close,
                                    )
                                else:
                                    _open = np.nan
                                    _high = _low = _close = flex_select_nb(close_, i, col)
                                if tsl_updated:
                                    if position_now > 0:
                                        if _high > last_tsl_info["peak_price"][col]:
                                            if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                                last_tsl_info["stop"][col] = (
                                                    last_tsl_info["stop"][col]
                                                    + _high
                                                    - last_tsl_info["peak_price"][col]
                                                )
                                            last_tsl_info["peak_idx"][col] = i
                                            last_tsl_info["peak_price"][col] = _high
                                    elif position_now < 0:
                                        if _low < last_tsl_info["peak_price"][col]:
                                            if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                                last_tsl_info["stop"][col] = (
                                                    last_tsl_info["stop"][col] + _low - last_tsl_info["peak_price"][col]
                                                )
                                            last_tsl_info["peak_idx"][col] = i
                                            last_tsl_info["peak_price"][col] = _low

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    last_cash[group] = cash_now
                    last_free_cash[group] = free_cash_now
                    last_value[group] = value_now
                    last_return[group] = return_now

            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_nb(close_, i, col)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                _cash_dividends = flex_select_nb(cash_dividends_, i, col)
                _cash_earnings += _cash_dividends * last_position[col]
                if _cash_earnings < 0:
                    _cash_earnings = max(_cash_earnings, -last_cash[group])
                last_cash[group] += _cash_earnings
                last_free_cash[group] += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings
                if save_state:
                    position[i, col] = last_position[col]
                    debt[i, col] = last_debt[col]
                    locked_cash[i, col] = last_locked_cash[col]
                    cash[i, group] = last_cash[group]
                    free_cash[i, group] = last_free_cash[group]

            # Update value and return
            group_value = last_cash[group]
            for col in range(from_col, to_col):
                if last_position[col] != 0:
                    group_value += last_position[col] * last_val_price[col]
            last_value[group] = group_value
            last_return[group] = returns_nb_.get_return_nb(
                input_value=prev_close_value[group],
                output_value=last_value[group] - _cash_deposits,
            )
            prev_close_value[group] = last_value[group]
            if save_value:
                in_outputs.value[i, group] = last_value[group]
            if save_returns:
                in_outputs.returns[i, group] = last_return[group]

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits_out,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


@register_jitted(cache=True)
def is_limit_info_active_nb(limit_info: tp.Record) -> bool:
    """Check whether information record for a limit order is active."""
    return is_limit_active_nb(limit_info["init_idx"], limit_info["init_price"])


@register_jitted(cache=True)
def is_stop_info_active_nb(stop_info: tp.Record) -> bool:
    """Check whether information record for a stop order is active."""
    return is_stop_active_nb(stop_info["init_idx"], stop_info["stop"])


@register_jitted(cache=True)
def is_time_stop_info_active_nb(time_stop_info: tp.Record) -> bool:
    """Check whether information record for a time stop order is active."""
    return is_time_stop_active_nb(time_stop_info["init_idx"], time_stop_info["stop"])


@register_jitted(cache=True)
def is_stop_info_ladder_active_nb(info: tp.Record) -> bool:
    """Check whether information record for a stop ladder is active."""
    return info["step"] != -1


@register_jitted(cache=True)
def set_limit_info_nb(
    limit_info: tp.Record,
    signal_idx: int,
    creation_idx: tp.Optional[int] = None,
    init_idx: tp.Optional[int] = None,
    init_price: float = -np.inf,
    init_size: float = np.inf,
    init_size_type: int = SizeType.Amount,
    init_direction: int = Direction.Both,
    init_stop_type: int = -1,
    delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    reverse: bool = False,
) -> None:
    """Set limit order information.

    See `vectorbtpro.portfolio.enums.limit_info_dt`."""
    limit_info["signal_idx"] = signal_idx
    limit_info["creation_idx"] = creation_idx if creation_idx is not None else signal_idx
    limit_info["init_idx"] = init_idx if init_idx is not None else signal_idx
    limit_info["init_price"] = init_price
    limit_info["init_size"] = init_size
    limit_info["init_size_type"] = init_size_type
    limit_info["init_direction"] = init_direction
    limit_info["init_stop_type"] = init_stop_type
    limit_info["delta"] = delta
    limit_info["delta_format"] = delta_format
    limit_info["tif"] = tif
    limit_info["expiry"] = expiry
    limit_info["time_delta_format"] = time_delta_format
    limit_info["reverse"] = reverse


@register_jitted(cache=True)
def clear_limit_info_nb(limit_info: tp.Record) -> None:
    """Clear limit order information."""
    limit_info["signal_idx"] = -1
    limit_info["creation_idx"] = -1
    limit_info["init_idx"] = -1
    limit_info["init_price"] = np.nan
    limit_info["init_size"] = np.nan
    limit_info["init_size_type"] = -1
    limit_info["init_direction"] = -1
    limit_info["init_stop_type"] = -1
    limit_info["delta"] = np.nan
    limit_info["delta_format"] = -1
    limit_info["tif"] = -1
    limit_info["expiry"] = -1
    limit_info["time_delta_format"] = -1
    limit_info["reverse"] = False


@register_jitted(cache=True)
def set_sl_info_nb(
    sl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set SL order information.

    See `vectorbtpro.portfolio.enums.sl_info_dt`."""
    sl_info["init_idx"] = init_idx
    sl_info["init_price"] = init_price
    sl_info["init_position"] = init_position
    sl_info["stop"] = stop
    sl_info["exit_price"] = exit_price
    sl_info["exit_size"] = exit_size
    sl_info["exit_size_type"] = exit_size_type
    sl_info["exit_type"] = exit_type
    sl_info["order_type"] = order_type
    sl_info["limit_delta"] = limit_delta
    sl_info["delta_format"] = delta_format
    sl_info["ladder"] = ladder
    sl_info["step"] = step
    sl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_sl_info_nb(sl_info: tp.Record) -> None:
    """Clear SL order information."""
    sl_info["init_idx"] = -1
    sl_info["init_price"] = np.nan
    sl_info["init_position"] = np.nan
    sl_info["stop"] = np.nan
    sl_info["exit_price"] = -1
    sl_info["exit_size"] = np.nan
    sl_info["exit_size_type"] = -1
    sl_info["exit_type"] = -1
    sl_info["order_type"] = -1
    sl_info["limit_delta"] = np.nan
    sl_info["delta_format"] = -1
    sl_info["ladder"] = -1
    sl_info["step"] = -1
    sl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tsl_info_nb(
    tsl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    peak_idx: tp.Optional[int] = None,
    peak_price: tp.Optional[float] = None,
    stop: float = np.nan,
    th: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TSL order information.

    See `vectorbtpro.portfolio.enums.tsl_info_dt`."""
    tsl_info["init_idx"] = init_idx
    tsl_info["init_price"] = init_price
    tsl_info["init_position"] = init_position
    tsl_info["peak_idx"] = peak_idx if peak_idx is not None else init_idx
    tsl_info["peak_price"] = peak_price if peak_price is not None else init_price
    tsl_info["stop"] = stop
    tsl_info["th"] = th
    tsl_info["exit_price"] = exit_price
    tsl_info["exit_size"] = exit_size
    tsl_info["exit_size_type"] = exit_size_type
    tsl_info["exit_type"] = exit_type
    tsl_info["order_type"] = order_type
    tsl_info["limit_delta"] = limit_delta
    tsl_info["delta_format"] = delta_format
    tsl_info["ladder"] = ladder
    tsl_info["step"] = step
    tsl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tsl_info_nb(tsl_info: tp.Record) -> None:
    """Clear TSL order information."""
    tsl_info["init_idx"] = -1
    tsl_info["init_price"] = np.nan
    tsl_info["init_position"] = np.nan
    tsl_info["peak_idx"] = -1
    tsl_info["peak_price"] = np.nan
    tsl_info["stop"] = np.nan
    tsl_info["th"] = np.nan
    tsl_info["exit_price"] = -1
    tsl_info["exit_size"] = np.nan
    tsl_info["exit_size_type"] = -1
    tsl_info["exit_type"] = -1
    tsl_info["order_type"] = -1
    tsl_info["limit_delta"] = np.nan
    tsl_info["delta_format"] = -1
    tsl_info["ladder"] = -1
    tsl_info["step"] = -1
    tsl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tp_info_nb(
    tp_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TP order information.

    See `vectorbtpro.portfolio.enums.tp_info_dt`."""
    tp_info["init_idx"] = init_idx
    tp_info["init_price"] = init_price
    tp_info["init_position"] = init_position
    tp_info["stop"] = stop
    tp_info["exit_price"] = exit_price
    tp_info["exit_size"] = exit_size
    tp_info["exit_size_type"] = exit_size_type
    tp_info["exit_type"] = exit_type
    tp_info["order_type"] = order_type
    tp_info["limit_delta"] = limit_delta
    tp_info["delta_format"] = delta_format
    tp_info["ladder"] = ladder
    tp_info["step"] = step
    tp_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tp_info_nb(tp_info: tp.Record) -> None:
    """Clear TP order information."""
    tp_info["init_idx"] = -1
    tp_info["init_price"] = np.nan
    tp_info["init_position"] = np.nan
    tp_info["stop"] = np.nan
    tp_info["exit_price"] = -1
    tp_info["exit_size"] = np.nan
    tp_info["exit_size_type"] = -1
    tp_info["exit_type"] = -1
    tp_info["order_type"] = -1
    tp_info["limit_delta"] = np.nan
    tp_info["delta_format"] = -1
    tp_info["ladder"] = -1
    tp_info["step"] = -1
    tp_info["step_idx"] = -1


@register_jitted(cache=True)
def set_time_info_nb(
    time_info: tp.Record,
    init_idx: int,
    init_position: float = np.nan,
    stop: int = -1,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    time_delta_format: int = TimeDeltaFormat.Index,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set time order information.

    See `vectorbtpro.portfolio.enums.time_info_dt`."""
    time_info["init_idx"] = init_idx
    time_info["init_position"] = init_position
    time_info["stop"] = stop
    time_info["exit_price"] = exit_price
    time_info["exit_size"] = exit_size
    time_info["exit_size_type"] = exit_size_type
    time_info["exit_type"] = exit_type
    time_info["order_type"] = order_type
    time_info["limit_delta"] = limit_delta
    time_info["delta_format"] = delta_format
    time_info["time_delta_format"] = time_delta_format
    time_info["ladder"] = ladder
    time_info["step"] = step
    time_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_time_info_nb(time_info: tp.Record) -> None:
    """Clear time order information."""
    time_info["init_idx"] = -1
    time_info["init_position"] = np.nan
    time_info["stop"] = -1
    time_info["exit_price"] = -1
    time_info["exit_size"] = np.nan
    time_info["exit_size_type"] = -1
    time_info["exit_type"] = -1
    time_info["order_type"] = -1
    time_info["limit_delta"] = np.nan
    time_info["delta_format"] = -1
    time_info["time_delta_format"] = -1
    time_info["ladder"] = -1
    time_info["step"] = -1
    time_info["step_idx"] = -1


@register_jitted
def no_signal_func_nb(c: SignalContext, *args) -> tp.Tuple[bool, bool, bool, bool]:
    """Placeholder signal function that returns no signal."""
    return False, False, False, False


SignalFuncT = tp.Callable[[SignalContext, tp.VarArg()], tp.Tuple[bool, bool, bool, bool]]
PostSegmentFuncT = tp.Callable[[SignalSegmentContext, tp.VarArg()], None]


# % <block signal_func_nb>
# % <skip? skip_func(out_lines, "signal_func_nb")>
# % <uncomment>
# @register_jitted
# def signal_func_nb(
#     c: SignalContext,
# ) -> tp.Tuple[bool, bool, bool, bool]:
#     """Custom signal function."""
#     return False, False, False, False
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_segment_func_nb>
# % <skip? skip_func(out_lines, "post_segment_func_nb")>
# % <uncomment>
# @register_jitted
# def post_segment_func_nb(
#     c: SignalSegmentContext,
# ) -> None:
#     """Custom post-segment function."""
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <section from_signal_func_nb>
# % <uncomment>
# import vectorbtpro as vbt
# from vectorbtpro.portfolio.nb.from_signals import *
# %? import_lines
#
#
# % </uncomment>
# %? blocks[signal_func_nb_block]
# % blocks["signal_func_nb"]
# %? blocks[post_segment_func_nb_block]
# % blocks["post_segment_func_nb"]
@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=base_ch.flex_array_gl_slicer,
        cash_dividends=base_ch.flex_array_gl_slicer,
        signal_func_nb=None,  # % None
        signal_args=ch.ArgsTaker(),
        post_segment_func_nb=None,  # % None
        post_segment_args=ch.ArgsTaker(),
        size=base_ch.flex_array_gl_slicer,
        price=base_ch.flex_array_gl_slicer,
        size_type=base_ch.flex_array_gl_slicer,
        fees=base_ch.flex_array_gl_slicer,
        fixed_fees=base_ch.flex_array_gl_slicer,
        slippage=base_ch.flex_array_gl_slicer,
        min_size=base_ch.flex_array_gl_slicer,
        max_size=base_ch.flex_array_gl_slicer,
        size_granularity=base_ch.flex_array_gl_slicer,
        leverage=base_ch.flex_array_gl_slicer,
        leverage_mode=base_ch.flex_array_gl_slicer,
        reject_prob=base_ch.flex_array_gl_slicer,
        price_area_vio_mode=base_ch.flex_array_gl_slicer,
        allow_partial=base_ch.flex_array_gl_slicer,
        raise_reject=base_ch.flex_array_gl_slicer,
        log=base_ch.flex_array_gl_slicer,
        val_price=base_ch.flex_array_gl_slicer,
        accumulate=base_ch.flex_array_gl_slicer,
        upon_long_conflict=base_ch.flex_array_gl_slicer,
        upon_short_conflict=base_ch.flex_array_gl_slicer,
        upon_dir_conflict=base_ch.flex_array_gl_slicer,
        upon_opposite_entry=base_ch.flex_array_gl_slicer,
        order_type=base_ch.flex_array_gl_slicer,
        limit_delta=base_ch.flex_array_gl_slicer,
        limit_tif=base_ch.flex_array_gl_slicer,
        limit_expiry=base_ch.flex_array_gl_slicer,
        limit_reverse=base_ch.flex_array_gl_slicer,
        upon_adj_limit_conflict=base_ch.flex_array_gl_slicer,
        upon_opp_limit_conflict=base_ch.flex_array_gl_slicer,
        use_stops=None,
        stop_ladder=None,
        sl_stop=base_ch.flex_array_gl_slicer,
        tsl_stop=base_ch.flex_array_gl_slicer,
        tsl_th=base_ch.flex_array_gl_slicer,
        tp_stop=base_ch.flex_array_gl_slicer,
        td_stop=base_ch.flex_array_gl_slicer,
        dt_stop=base_ch.flex_array_gl_slicer,
        stop_entry_price=base_ch.flex_array_gl_slicer,
        stop_exit_price=base_ch.flex_array_gl_slicer,
        stop_exit_type=base_ch.flex_array_gl_slicer,
        stop_order_type=base_ch.flex_array_gl_slicer,
        stop_limit_delta=base_ch.flex_array_gl_slicer,
        upon_stop_update=base_ch.flex_array_gl_slicer,
        upon_adj_stop_conflict=base_ch.flex_array_gl_slicer,
        upon_opp_stop_conflict=base_ch.flex_array_gl_slicer,
        delta_format=base_ch.flex_array_gl_slicer,
        time_delta_format=base_ch.flex_array_gl_slicer,
        from_ago=base_ch.flex_array_gl_slicer,
        call_seq=base_ch.array_gl_slicer,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        fill_pos_info=None,
        max_orders=None,
        max_logs=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config,
    setup_id=None,  # %? line.replace("None", task_id)
)
@register_jitted(
    tags={"can_parallel"},
    cache=False,  # % line.replace("False", "True")
    task_id_or_func=None,  # %? line.replace("None", task_id)
)
def from_signal_func_nb(  # %? line.replace("from_signal_func_nb", new_func_name)
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    cash_dividends: tp.FlexArray2dLike = 0.0,
    signal_func_nb: SignalFuncT = no_signal_func_nb,  # % None
    signal_args: tp.ArgsLike = (),
    post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,  # % None
    post_segment_args: tp.ArgsLike = (),
    size: tp.FlexArray2dLike = np.inf,
    price: tp.FlexArray2dLike = np.inf,
    size_type: tp.FlexArray2dLike = SizeType.Amount,
    fees: tp.FlexArray2dLike = 0.0,
    fixed_fees: tp.FlexArray2dLike = 0.0,
    slippage: tp.FlexArray2dLike = 0.0,
    min_size: tp.FlexArray2dLike = np.nan,
    max_size: tp.FlexArray2dLike = np.nan,
    size_granularity: tp.FlexArray2dLike = np.nan,
    leverage: tp.FlexArray2dLike = 1.0,
    leverage_mode: tp.FlexArray2dLike = LeverageMode.Lazy,
    reject_prob: tp.FlexArray2dLike = 0.0,
    price_area_vio_mode: tp.FlexArray2dLike = PriceAreaVioMode.Ignore,
    allow_partial: tp.FlexArray2dLike = True,
    raise_reject: tp.FlexArray2dLike = False,
    log: tp.FlexArray2dLike = False,
    val_price: tp.FlexArray2dLike = np.inf,
    accumulate: tp.FlexArray2dLike = AccumulationMode.Disabled,
    upon_long_conflict: tp.FlexArray2dLike = ConflictMode.Ignore,
    upon_short_conflict: tp.FlexArray2dLike = ConflictMode.Ignore,
    upon_dir_conflict: tp.FlexArray2dLike = DirectionConflictMode.Ignore,
    upon_opposite_entry: tp.FlexArray2dLike = OppositeEntryMode.ReverseReduce,
    order_type: tp.FlexArray2dLike = OrderType.Market,
    limit_delta: tp.FlexArray2dLike = np.nan,
    limit_tif: tp.FlexArray2dLike = -1,
    limit_expiry: tp.FlexArray2dLike = -1,
    limit_reverse: tp.FlexArray2dLike = False,
    upon_adj_limit_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepIgnore,
    upon_opp_limit_conflict: tp.FlexArray2dLike = PendingConflictMode.CancelExecute,
    use_stops: bool = True,
    stop_ladder: int = StopLadderMode.Disabled,
    sl_stop: tp.FlexArray2dLike = np.nan,
    tsl_stop: tp.FlexArray2dLike = np.nan,
    tsl_th: tp.FlexArray2dLike = np.nan,
    tp_stop: tp.FlexArray2dLike = np.nan,
    td_stop: tp.FlexArray2dLike = -1,
    dt_stop: tp.FlexArray2dLike = -1,
    stop_entry_price: tp.FlexArray2dLike = StopEntryPrice.Close,
    stop_exit_price: tp.FlexArray2dLike = StopExitPrice.Stop,
    stop_exit_type: tp.FlexArray2dLike = StopExitType.Close,
    stop_order_type: tp.FlexArray2dLike = OrderType.Market,
    stop_limit_delta: tp.FlexArray2dLike = np.nan,
    upon_stop_update: tp.FlexArray2dLike = StopUpdateMode.Keep,
    upon_adj_stop_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepExecute,
    upon_opp_stop_conflict: tp.FlexArray2dLike = PendingConflictMode.KeepExecute,
    delta_format: tp.FlexArray2dLike = DeltaFormat.Percent,
    time_delta_format: tp.FlexArray2dLike = TimeDeltaFormat.Index,
    from_ago: tp.FlexArray2dLike = 0,
    call_seq: tp.Optional[tp.Array2d] = None,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_info: bool = True,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Simulate given a signal function.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    `signal_func_nb` is a user-defined signal generation function that is called at each row and column
    (= element). It must accept the context of the type `vectorbtpro.portfolio.enums.SignalContext`
    and return 4 signals: long entry, long exit, short entry, and short exit.

    `post_segment_func_nb` is a user-defined post-segment function that is called after each row and group
    (= segment). It must accept the context of the type `vectorbtpro.portfolio.enums.SignalSegmentContext`
    and return nothing.

    !!! note
        Should be only grouped if cash sharing is enabled.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).
    """
    check_group_lens_nb(group_lens, target_shape[1])

    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    cash_dividends_ = to_2d_array_nb(np.asarray(cash_dividends))
    size_ = to_2d_array_nb(np.asarray(size))
    price_ = to_2d_array_nb(np.asarray(price))
    size_type_ = to_2d_array_nb(np.asarray(size_type))
    fees_ = to_2d_array_nb(np.asarray(fees))
    fixed_fees_ = to_2d_array_nb(np.asarray(fixed_fees))
    slippage_ = to_2d_array_nb(np.asarray(slippage))
    min_size_ = to_2d_array_nb(np.asarray(min_size))
    max_size_ = to_2d_array_nb(np.asarray(max_size))
    size_granularity_ = to_2d_array_nb(np.asarray(size_granularity))
    leverage_ = to_2d_array_nb(np.asarray(leverage))
    leverage_mode_ = to_2d_array_nb(np.asarray(leverage_mode))
    reject_prob_ = to_2d_array_nb(np.asarray(reject_prob))
    price_area_vio_mode_ = to_2d_array_nb(np.asarray(price_area_vio_mode))
    allow_partial_ = to_2d_array_nb(np.asarray(allow_partial))
    raise_reject_ = to_2d_array_nb(np.asarray(raise_reject))
    log_ = to_2d_array_nb(np.asarray(log))
    val_price_ = to_2d_array_nb(np.asarray(val_price))
    accumulate_ = to_2d_array_nb(np.asarray(accumulate))
    upon_long_conflict_ = to_2d_array_nb(np.asarray(upon_long_conflict))
    upon_short_conflict_ = to_2d_array_nb(np.asarray(upon_short_conflict))
    upon_dir_conflict_ = to_2d_array_nb(np.asarray(upon_dir_conflict))
    upon_opposite_entry_ = to_2d_array_nb(np.asarray(upon_opposite_entry))
    order_type_ = to_2d_array_nb(np.asarray(order_type))
    limit_delta_ = to_2d_array_nb(np.asarray(limit_delta))
    limit_tif_ = to_2d_array_nb(np.asarray(limit_tif))
    limit_expiry_ = to_2d_array_nb(np.asarray(limit_expiry))
    limit_reverse_ = to_2d_array_nb(np.asarray(limit_reverse))
    upon_adj_limit_conflict_ = to_2d_array_nb(np.asarray(upon_adj_limit_conflict))
    upon_opp_limit_conflict_ = to_2d_array_nb(np.asarray(upon_opp_limit_conflict))
    sl_stop_ = to_2d_array_nb(np.asarray(sl_stop))
    tsl_stop_ = to_2d_array_nb(np.asarray(tsl_stop))
    tsl_th_ = to_2d_array_nb(np.asarray(tsl_th))
    tp_stop_ = to_2d_array_nb(np.asarray(tp_stop))
    td_stop_ = to_2d_array_nb(np.asarray(td_stop))
    dt_stop_ = to_2d_array_nb(np.asarray(dt_stop))
    stop_entry_price_ = to_2d_array_nb(np.asarray(stop_entry_price))
    stop_exit_price_ = to_2d_array_nb(np.asarray(stop_exit_price))
    stop_exit_type_ = to_2d_array_nb(np.asarray(stop_exit_type))
    stop_order_type_ = to_2d_array_nb(np.asarray(stop_order_type))
    stop_limit_delta_ = to_2d_array_nb(np.asarray(stop_limit_delta))
    upon_stop_update_ = to_2d_array_nb(np.asarray(upon_stop_update))
    upon_adj_stop_conflict_ = to_2d_array_nb(np.asarray(upon_adj_stop_conflict))
    upon_opp_stop_conflict_ = to_2d_array_nb(np.asarray(upon_opp_stop_conflict))
    delta_format_ = to_2d_array_nb(np.asarray(delta_format))
    time_delta_format_ = to_2d_array_nb(np.asarray(time_delta_format))
    from_ago_ = to_2d_array_nb(np.asarray(from_ago))

    n_sl_steps = sl_stop_.shape[0]
    n_tsl_steps = tsl_stop_.shape[0]
    n_tp_steps = tp_stop_.shape[0]
    n_td_steps = td_stop_.shape[0]
    n_dt_steps = dt_stop_.shape[0]

    if max_orders is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=fs_order_dt)
    else:
        order_records = np.empty((max_orders, target_shape[1]), dtype=fs_order_dt)
    if max_logs is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_logs, target_shape[1]), dtype=log_dt)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    last_cash = prepare_last_cash_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
    )
    last_position = prepare_last_position_nb(
        target_shape=target_shape,
        init_position=init_position_,
    )
    last_value = prepare_last_value_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
    )
    last_pos_info = prepare_last_pos_info_nb(
        target_shape,
        init_position=init_position_,
        init_price=init_price_,
        fill_pos_info=fill_pos_info,
    )
    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_locked_cash = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    track_cash_deposits = np.any(cash_deposits_)
    if track_cash_deposits:
        cash_deposits_out = np.full((target_shape[0], len(group_lens)), 0.0, dtype=np.float_)
    else:
        cash_deposits_out = np.full((1, 1), 0.0, dtype=np.float_)
    track_cash_earnings = np.any(cash_earnings_) or np.any(cash_dividends_)
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0.0, dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=np.float_)

    long_entries = np.full(target_shape[1], False, dtype=np.bool_)
    long_exits = np.full(target_shape[1], False, dtype=np.bool_)
    short_entries = np.full(target_shape[1], False, dtype=np.bool_)
    short_exits = np.full(target_shape[1], False, dtype=np.bool_)

    last_limit_info = np.empty(target_shape[1], dtype=limit_info_dt)
    last_limit_info["signal_idx"][:] = -1
    last_limit_info["creation_idx"][:] = -1
    last_limit_info["init_idx"][:] = -1
    last_limit_info["init_price"][:] = np.nan
    last_limit_info["init_size"][:] = np.nan
    last_limit_info["init_size_type"][:] = -1
    last_limit_info["init_direction"][:] = -1
    last_limit_info["init_stop_type"][:] = -1
    last_limit_info["delta"][:] = np.nan
    last_limit_info["delta_format"][:] = -1
    last_limit_info["tif"][:] = -1
    last_limit_info["expiry"][:] = -1
    last_limit_info["time_delta_format"][:] = -1
    last_limit_info["reverse"][:] = False

    if use_stops:
        last_sl_info = np.empty(target_shape[1], dtype=sl_info_dt)
        last_sl_info["init_idx"][:] = -1
        last_sl_info["init_price"][:] = np.nan
        last_sl_info["init_position"][:] = np.nan
        last_sl_info["stop"][:] = np.nan
        last_sl_info["exit_price"][:] = -1
        last_sl_info["exit_size"][:] = np.nan
        last_sl_info["exit_size_type"][:] = -1
        last_sl_info["exit_type"][:] = -1
        last_sl_info["order_type"][:] = -1
        last_sl_info["limit_delta"][:] = np.nan
        last_sl_info["delta_format"][:] = -1
        last_sl_info["ladder"][:] = -1
        last_sl_info["step"][:] = -1
        last_sl_info["step_idx"][:] = -1

        last_tsl_info = np.empty(target_shape[1], dtype=tsl_info_dt)
        last_tsl_info["init_idx"][:] = -1
        last_tsl_info["init_price"][:] = np.nan
        last_tsl_info["init_position"][:] = np.nan
        last_tsl_info["peak_idx"][:] = -1
        last_tsl_info["peak_price"][:] = np.nan
        last_tsl_info["stop"][:] = np.nan
        last_tsl_info["th"][:] = np.nan
        last_tsl_info["exit_price"][:] = -1
        last_tsl_info["exit_size"][:] = np.nan
        last_tsl_info["exit_size_type"][:] = -1
        last_tsl_info["exit_type"][:] = -1
        last_tsl_info["order_type"][:] = -1
        last_tsl_info["limit_delta"][:] = np.nan
        last_tsl_info["delta_format"][:] = -1
        last_tsl_info["ladder"][:] = -1
        last_tsl_info["step"][:] = -1
        last_tsl_info["step_idx"][:] = -1

        last_tp_info = np.empty(target_shape[1], dtype=tp_info_dt)
        last_tp_info["init_idx"][:] = -1
        last_tp_info["init_price"][:] = np.nan
        last_tp_info["init_position"][:] = np.nan
        last_tp_info["stop"][:] = np.nan
        last_tp_info["exit_price"][:] = -1
        last_tp_info["exit_size"][:] = np.nan
        last_tp_info["exit_size_type"][:] = -1
        last_tp_info["exit_type"][:] = -1
        last_tp_info["order_type"][:] = -1
        last_tp_info["limit_delta"][:] = np.nan
        last_tp_info["delta_format"][:] = -1
        last_tp_info["ladder"][:] = -1
        last_tp_info["step"][:] = -1
        last_tp_info["step_idx"][:] = -1

        last_td_info = np.empty(target_shape[1], dtype=time_info_dt)
        last_td_info["init_idx"][:] = -1
        last_td_info["init_position"][:] = np.nan
        last_td_info["stop"][:] = -1
        last_td_info["exit_price"][:] = -1
        last_td_info["exit_size"][:] = np.nan
        last_td_info["exit_size_type"][:] = -1
        last_td_info["exit_type"][:] = -1
        last_td_info["order_type"][:] = -1
        last_td_info["limit_delta"][:] = np.nan
        last_td_info["delta_format"][:] = -1
        last_td_info["time_delta_format"][:] = -1
        last_td_info["ladder"][:] = -1
        last_td_info["step"][:] = -1
        last_td_info["step_idx"][:] = -1

        last_dt_info = np.empty(target_shape[1], dtype=time_info_dt)
        last_dt_info["init_idx"][:] = -1
        last_dt_info["init_position"][:] = np.nan
        last_dt_info["stop"][:] = -1
        last_dt_info["exit_price"][:] = -1
        last_dt_info["exit_size"][:] = np.nan
        last_dt_info["exit_size_type"][:] = -1
        last_dt_info["exit_type"][:] = -1
        last_dt_info["order_type"][:] = -1
        last_dt_info["limit_delta"][:] = np.nan
        last_dt_info["delta_format"][:] = -1
        last_dt_info["time_delta_format"][:] = -1
        last_dt_info["ladder"][:] = -1
        last_dt_info["step"][:] = -1
        last_dt_info["step_idx"][:] = -1
    else:
        last_sl_info = np.empty(0, dtype=sl_info_dt)
        last_tsl_info = np.empty(0, dtype=tsl_info_dt)
        last_tp_info = np.empty(0, dtype=tp_info_dt)
        last_td_info = np.empty(0, dtype=time_info_dt)
        last_dt_info = np.empty(0, dtype=time_info_dt)

    main_info = np.empty(target_shape[1], dtype=main_info_dt)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)
    temp_sort_by = np.empty(target_shape[1], dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        for i in range(target_shape[0]):
            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_nb(cash_deposits_, i, group)
                if _cash_deposits < 0:
                    _cash_deposits = max(_cash_deposits, -last_cash[group])
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
                if track_cash_deposits:
                    cash_deposits_out[i, group] += _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_nb(cash_deposits_, i, col)
                    if _cash_deposits < 0:
                        _cash_deposits = max(_cash_deposits, -last_cash[col])
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits
                    if track_cash_deposits:
                        cash_deposits_out[i, col] += _cash_deposits

            # Update valuation price using current open
            for c in range(group_len):
                col = from_col + c
                _open = flex_select_nb(open_, i, col)
                if not np.isnan(_open) or not ffill_val_price:
                    last_val_price[col] = _open

            # Update value and return
            if cash_sharing:
                group_value = last_cash[group]
                for col in range(from_col, to_col):
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                last_value[group] = group_value
                last_return[group] = returns_nb_.get_return_nb(
                    input_value=prev_close_value[group],
                    output_value=last_value[group] - last_cash_deposits[group],
                )
            else:
                for col in range(from_col, to_col):
                    group_value = last_cash[col]
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                    last_value[col] = group_value
                    last_return[col] = returns_nb_.get_return_nb(
                        input_value=prev_close_value[col], output_value=last_value[col] - last_cash_deposits[col]
                    )

            # Update open position stats
            if fill_pos_info:
                for col in range(from_col, to_col):
                    update_open_pos_info_stats_nb(last_pos_info[col], last_position[col], last_val_price[col])

            # Get signals
            for c in range(group_len):
                col = from_col + c  # order doesn't matter

                signal_ctx = SignalContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    track_cash_deposits=track_cash_deposits,
                    cash_deposits_out=cash_deposits_out,
                    track_cash_earnings=track_cash_earnings,
                    cash_earnings_out=cash_earnings_out,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    last_limit_info=last_limit_info,
                    last_sl_info=last_sl_info,
                    last_tsl_info=last_tsl_info,
                    last_tp_info=last_tp_info,
                    last_td_info=last_td_info,
                    last_dt_info=last_dt_info,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    col=col,
                )
                is_long_entry, is_long_exit, is_short_entry, is_short_exit = signal_func_nb(signal_ctx, *signal_args)
                long_entries[col] = is_long_entry
                long_exits[col] = is_long_exit
                short_entries[col] = is_short_entry
                short_exits[col] = is_short_exit

                # Update limit and stop prices
                _i = i - abs(flex_select_nb(from_ago_, i, col))
                if _i < 0:
                    _price = np.nan
                else:
                    _price = flex_select_nb(price_, _i, col)
                last_limit_info["init_price"][col] = resolve_dyn_limit_price_nb(
                    val_price=last_val_price[col],
                    price=_price,
                    limit_price=last_limit_info["init_price"][col],
                )
                last_sl_info["init_price"][col] = resolve_dyn_stop_entry_price_nb(
                    val_price=last_val_price[col],
                    price=_price,
                    stop_entry_price=last_sl_info["init_price"][col],
                )
                last_tsl_info["init_price"][col] = resolve_dyn_stop_entry_price_nb(
                    val_price=last_val_price[col],
                    price=_price,
                    stop_entry_price=last_tsl_info["init_price"][col],
                )
                last_tsl_info["peak_price"][col] = resolve_dyn_stop_entry_price_nb(
                    val_price=last_val_price[col],
                    price=_price,
                    stop_entry_price=last_tsl_info["peak_price"][col],
                )
                last_tp_info["init_price"][col] = resolve_dyn_stop_entry_price_nb(
                    val_price=last_val_price[col],
                    price=_price,
                    stop_entry_price=last_tp_info["init_price"][col],
                )

            # Update value and return
            if cash_sharing:
                group_value = last_cash[group]
                for col in range(from_col, to_col):
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                last_value[group] = group_value
                last_return[group] = returns_nb_.get_return_nb(
                    input_value=prev_close_value[group],
                    output_value=last_value[group] - last_cash_deposits[group],
                )
            else:
                for col in range(from_col, to_col):
                    group_value = last_cash[col]
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                    last_value[col] = group_value
                    last_return[col] = returns_nb_.get_return_nb(
                        input_value=prev_close_value[col], output_value=last_value[col] - last_cash_deposits[col]
                    )

            # Get size and value of each order
            for c in range(group_len):
                col = from_col + c  # order doesn't matter

                # Get signals
                is_long_entry = long_entries[col]
                is_long_exit = long_exits[col]
                is_short_entry = short_entries[col]
                is_short_exit = short_exits[col]

                # Set defaults
                main_info["bar_zone"][col] = -1
                main_info["signal_idx"][col] = -1
                main_info["creation_idx"][col] = -1
                main_info["idx"][col] = i
                main_info["val_price"][col] = np.nan
                main_info["price"][col] = np.nan
                main_info["size"][col] = np.nan
                main_info["size_type"][col] = -1
                main_info["direction"][col] = -1
                main_info["type"][col] = -1
                main_info["stop_type"][col] = -1
                temp_sort_by[col] = 0.0

                any_limit_signal = is_limit_active_nb(
                    init_idx=last_limit_info["init_idx"][col],
                    init_price=last_limit_info["init_price"][col],
                )
                if use_stops:
                    sl_stop_signal = is_stop_active_nb(
                        init_idx=last_sl_info["init_idx"][col],
                        stop=last_sl_info["stop"][col],
                    )
                    tsl_stop_signal = is_stop_active_nb(
                        init_idx=last_tsl_info["init_idx"][col],
                        stop=last_tsl_info["stop"][col],
                    )
                    tp_stop_signal = is_stop_active_nb(
                        init_idx=last_tp_info["init_idx"][col],
                        stop=last_tp_info["stop"][col],
                    )
                    td_stop_signal = is_time_stop_active_nb(
                        init_idx=last_td_info["init_idx"][col],
                        stop=last_td_info["stop"][col],
                    )
                    dt_stop_signal = is_time_stop_active_nb(
                        init_idx=last_dt_info["init_idx"][col],
                        stop=last_dt_info["stop"][col],
                    )
                    any_stop_signal = (
                        sl_stop_signal or tsl_stop_signal or tp_stop_signal or td_stop_signal or dt_stop_signal
                    )
                else:
                    sl_stop_signal = False
                    tsl_stop_signal = False
                    tp_stop_signal = False
                    td_stop_signal = False
                    dt_stop_signal = False
                    any_stop_signal = False
                any_user_signal = is_long_entry or is_long_exit or is_short_entry or is_short_exit
                if not any_limit_signal and not any_stop_signal and not any_user_signal:  # shortcut
                    continue

                # Set initial info
                exec_limit_set = False
                exec_limit_set_on_open = False
                exec_limit_signal_i = -1
                exec_limit_creation_i = -1
                exec_limit_init_i = -1
                exec_limit_val_price = np.nan
                exec_limit_price = np.nan
                exec_limit_size = np.nan
                exec_limit_size_type = -1
                exec_limit_direction = -1
                exec_limit_stop_type = -1
                exec_limit_bar_zone = -1

                exec_stop_set = False
                exec_stop_set_on_open = False
                exec_stop_set_on_close = False
                exec_stop_init_i = -1
                exec_stop_val_price = np.nan
                exec_stop_price = np.nan
                exec_stop_size = np.nan
                exec_stop_size_type = -1
                exec_stop_direction = -1
                exec_stop_type = -1
                exec_stop_stop_type = -1
                exec_stop_delta = np.nan
                exec_stop_delta_format = -1
                exec_stop_make_limit = False
                exec_stop_bar_zone = -1

                user_on_open = False
                user_on_close = False
                exec_user_set = False
                exec_user_val_price = np.nan
                exec_user_price = np.nan
                exec_user_size = np.nan
                exec_user_size_type = -1
                exec_user_direction = -1
                exec_user_type = -1
                exec_user_stop_type = -1
                exec_user_make_limit = False
                exec_user_bar_zone = -1

                # Resolve the current bar
                _open = flex_select_nb(open_, i, col)
                _high = flex_select_nb(high_, i, col)
                _low = flex_select_nb(low_, i, col)
                _close = flex_select_nb(close_, i, col)
                _high, _low = resolve_hl_nb(
                    open=_open,
                    high=_high,
                    low=_low,
                    close=_close,
                )

                # Process the limit signal
                if any_limit_signal:
                    # Check whether the limit price was hit
                    _signal_i = last_limit_info["signal_idx"][col]
                    _creation_i = last_limit_info["creation_idx"][col]
                    _init_i = last_limit_info["init_idx"][col]
                    _price = last_limit_info["init_price"][col]
                    _size = last_limit_info["init_size"][col]
                    _size_type = last_limit_info["init_size_type"][col]
                    _direction = last_limit_info["init_direction"][col]
                    _stop_type = last_limit_info["init_stop_type"][col]
                    _delta = last_limit_info["delta"][col]
                    _delta_format = last_limit_info["delta_format"][col]
                    _tif = last_limit_info["tif"][col]
                    _expiry = last_limit_info["expiry"][col]
                    _time_delta_format = last_limit_info["time_delta_format"][col]
                    _reverse = last_limit_info["reverse"][col]

                    limit_expired_on_open, limit_expired = check_limit_expired_nb(
                        creation_idx=_creation_i,
                        i=i,
                        tif=_tif,
                        expiry=_expiry,
                        time_delta_format=_time_delta_format,
                        index=index,
                        freq=freq,
                    )
                    limit_price, limit_hit_on_open, limit_hit = check_limit_hit_nb(
                        open=_open,
                        high=_high,
                        low=_low,
                        close=_close,
                        price=_price,
                        size=_size,
                        direction=_direction,
                        limit_delta=_delta,
                        delta_format=_delta_format,
                        limit_reverse=_reverse,
                        can_use_ohlc=True,
                        check_open=True,
                    )
                    if limit_expired_on_open or (not limit_hit_on_open and limit_expired):
                        # Expired limit signal
                        any_limit_signal = False

                        last_limit_info["signal_idx"][col] = -1
                        last_limit_info["creation_idx"][col] = -1
                        last_limit_info["init_idx"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1
                        last_limit_info["reverse"][col] = False
                    else:
                        # Save info
                        if limit_hit:
                            # Executable limit signal
                            exec_limit_set = True
                            exec_limit_set_on_open = limit_hit_on_open
                            exec_limit_signal_i = _signal_i
                            exec_limit_creation_i = _creation_i
                            exec_limit_init_i = _init_i
                            if np.isinf(limit_price) and limit_price > 0:
                                exec_limit_val_price = _close
                            elif np.isinf(limit_price) and limit_price < 0:
                                exec_limit_val_price = _open
                            else:
                                exec_limit_val_price = limit_price
                            exec_limit_price = limit_price
                            exec_limit_size = _size
                            exec_limit_size_type = _size_type
                            exec_limit_direction = _direction
                            exec_limit_stop_type = _stop_type

                # Process the stop signal
                if any_stop_signal:
                    # Check SL
                    sl_stop_price, sl_stop_hit_on_open, sl_stop_hit = np.nan, False, False
                    if sl_stop_signal:
                        # Check against high and low
                        sl_stop_price, sl_stop_hit_on_open, sl_stop_hit = check_stop_hit_nb(
                            open=_open,
                            high=_high,
                            low=_low,
                            close=_close,
                            is_position_long=last_position[col] > 0,
                            init_price=last_sl_info["init_price"][col],
                            stop=last_sl_info["stop"][col],
                            delta_format=last_sl_info["delta_format"][col],
                            hit_below=True,
                        )

                    # Check TSL and TTP
                    tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = np.nan, False, False
                    if tsl_stop_signal:
                        # Update peak price using open
                        if last_position[col] > 0:
                            if _open > last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _open - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _open
                        else:
                            if _open < last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _open - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _open
                        # Check threshold against previous bars and open
                        if np.isnan(last_tsl_info["th"][col]):
                            th_hit = True
                        else:
                            th_hit = check_tsl_th_hit_nb(
                                is_position_long=last_position[col] > 0,
                                init_price=last_tsl_info["init_price"][col],
                                peak_price=last_tsl_info["peak_price"][col],
                                threshold=last_tsl_info["th"][col],
                                delta_format=last_tsl_info["delta_format"][col],
                            )
                        if th_hit:
                            tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = check_stop_hit_nb(
                                open=_open,
                                high=_high,
                                low=_low,
                                close=_close,
                                is_position_long=last_position[col] > 0,
                                init_price=last_tsl_info["peak_price"][col],
                                stop=last_tsl_info["stop"][col],
                                delta_format=last_tsl_info["delta_format"][col],
                                hit_below=True,
                            )
                        # Update peak price using full bar
                        if last_position[col] > 0:
                            if _high > last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _high - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _high
                        else:
                            if _low < last_tsl_info["peak_price"][col]:
                                if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                    last_tsl_info["stop"][col] = (
                                        last_tsl_info["stop"][col] + _low - last_tsl_info["peak_price"][col]
                                    )
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = _low
                        if not tsl_stop_hit:
                            # Check threshold against full bar
                            if not th_hit:
                                if np.isnan(last_tsl_info["th"][col]):
                                    th_hit = True
                                else:
                                    th_hit = check_tsl_th_hit_nb(
                                        is_position_long=last_position[col] > 0,
                                        init_price=last_tsl_info["init_price"][col],
                                        peak_price=last_tsl_info["peak_price"][col],
                                        threshold=last_tsl_info["th"][col],
                                        delta_format=last_tsl_info["delta_format"][col],
                                    )
                            if th_hit:
                                # Check threshold against close
                                tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit = check_stop_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    is_position_long=last_position[col] > 0,
                                    init_price=last_tsl_info["peak_price"][col],
                                    stop=last_tsl_info["stop"][col],
                                    delta_format=last_tsl_info["delta_format"][col],
                                    hit_below=True,
                                    can_use_ohlc=False,
                                )

                    # Check TP
                    tp_stop_price, tp_stop_hit_on_open, tp_stop_hit = np.nan, False, False
                    if tp_stop_signal:
                        tp_stop_price, tp_stop_hit_on_open, tp_stop_hit = check_stop_hit_nb(
                            open=_open,
                            high=_high,
                            low=_low,
                            close=_close,
                            is_position_long=last_position[col] > 0,
                            init_price=last_tp_info["init_price"][col],
                            stop=last_tp_info["stop"][col],
                            delta_format=last_tp_info["delta_format"][col],
                            hit_below=False,
                        )

                    # Check TD
                    td_stop_price, td_stop_hit_on_open, td_stop_hit = np.nan, False, False
                    if td_stop_signal:
                        td_stop_hit_on_open, td_stop_hit = check_td_stop_hit_nb(
                            init_idx=last_td_info["init_idx"][col],
                            i=i,
                            stop=last_td_info["stop"][col],
                            time_delta_format=last_td_info["time_delta_format"][col],
                            index=index,
                            freq=freq,
                        )
                        if np.isnan(_open):
                            td_stop_hit_on_open = False
                        if td_stop_hit_on_open:
                            td_stop_price = _open
                        else:
                            td_stop_price = _close

                    # Check DT
                    dt_stop_price, dt_stop_hit_on_open, dt_stop_hit = np.nan, False, False
                    if dt_stop_signal:
                        dt_stop_hit_on_open, dt_stop_hit = check_dt_stop_hit_nb(
                            i=i,
                            stop=last_dt_info["stop"][col],
                            time_delta_format=last_dt_info["time_delta_format"][col],
                            index=index,
                            freq=freq,
                        )
                        if np.isnan(_open):
                            dt_stop_hit_on_open = False
                        if dt_stop_hit_on_open:
                            dt_stop_price = _open
                        else:
                            dt_stop_price = _close

                    # Resolve the stop signal
                    sl_hit = False
                    tsl_hit = False
                    tp_hit = False
                    td_hit = False
                    dt_hit = False
                    if sl_stop_hit_on_open:
                        sl_hit = True
                    elif tsl_stop_hit_on_open:
                        tsl_hit = True
                    elif tp_stop_hit_on_open:
                        tp_hit = True
                    elif td_stop_hit_on_open:
                        td_hit = True
                    elif dt_stop_hit_on_open:
                        dt_hit = True
                    elif sl_stop_hit:
                        sl_hit = True
                    elif tsl_stop_hit:
                        tsl_hit = True
                    elif tp_stop_hit:
                        tp_hit = True
                    elif td_stop_hit:
                        td_hit = True
                    elif dt_stop_hit:
                        dt_hit = True

                    if sl_hit:
                        stop_price, stop_hit_on_open, stop_hit = sl_stop_price, sl_stop_hit_on_open, sl_stop_hit
                        _stop_type = StopType.SL
                        _init_i = last_sl_info["init_idx"][col]
                        _stop_exit_price = last_sl_info["exit_price"][col]
                        _stop_exit_size = last_sl_info["exit_size"][col]
                        _stop_exit_size_type = last_sl_info["exit_size_type"][col]
                        _stop_exit_type = last_sl_info["exit_type"][col]
                        _stop_order_type = last_sl_info["order_type"][col]
                        _limit_delta = last_sl_info["limit_delta"][col]
                        _delta_format = last_sl_info["delta_format"][col]
                        _ladder = last_sl_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_sl_info["step"][col]
                                if step < n_sl_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=sl_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_sl_info["init_price"][col],
                                        init_position=last_sl_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_sl_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif tsl_hit:
                        stop_price, stop_hit_on_open, stop_hit = tsl_stop_price, tsl_stop_hit_on_open, tsl_stop_hit
                        if np.isnan(last_tsl_info["th"][col]):
                            _stop_type = StopType.TSL
                        else:
                            _stop_type = StopType.TTP
                        _init_i = last_tsl_info["init_idx"][col]
                        _stop_exit_price = last_tsl_info["exit_price"][col]
                        _stop_exit_size = last_tsl_info["exit_size"][col]
                        _stop_exit_size_type = last_tsl_info["exit_size_type"][col]
                        _stop_exit_type = last_tsl_info["exit_type"][col]
                        _stop_order_type = last_tsl_info["order_type"][col]
                        _limit_delta = last_tsl_info["limit_delta"][col]
                        _delta_format = last_tsl_info["delta_format"][col]
                        _ladder = last_tsl_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_tsl_info["step"][col]
                                if step < n_tsl_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=tsl_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_tsl_info["init_price"][col],
                                        init_position=last_tsl_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_tsl_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif tp_hit:
                        stop_price, stop_hit_on_open, stop_hit = tp_stop_price, tp_stop_hit_on_open, tp_stop_hit
                        _stop_type = StopType.TP
                        _init_i = last_tp_info["init_idx"][col]
                        _stop_exit_price = last_tp_info["exit_price"][col]
                        _stop_exit_size = last_tp_info["exit_size"][col]
                        _stop_exit_size_type = last_tp_info["exit_size_type"][col]
                        _stop_exit_type = last_tp_info["exit_type"][col]
                        _stop_order_type = last_tp_info["order_type"][col]
                        _limit_delta = last_tp_info["limit_delta"][col]
                        _delta_format = last_tp_info["delta_format"][col]
                        _ladder = last_tp_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_tp_info["step"][col]
                                if step < n_tp_steps:
                                    _stop_exit_size = get_stop_ladder_exit_size_nb(
                                        stop_=tp_stop_,
                                        step=step,
                                        col=col,
                                        init_price=last_tp_info["init_price"][col],
                                        init_position=last_tp_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        delta_format=last_tp_info["delta_format"][col],
                                        hit_below=True,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif td_hit:
                        stop_price, stop_hit_on_open, stop_hit = td_stop_price, td_stop_hit_on_open, td_stop_hit
                        _stop_type = StopType.TD
                        _init_i = last_td_info["init_idx"][col]
                        _stop_exit_price = last_td_info["exit_price"][col]
                        _stop_exit_size = last_td_info["exit_size"][col]
                        _stop_exit_size_type = last_td_info["exit_size_type"][col]
                        _stop_exit_type = last_td_info["exit_type"][col]
                        _stop_order_type = last_td_info["order_type"][col]
                        _limit_delta = last_td_info["limit_delta"][col]
                        _delta_format = last_td_info["delta_format"][col]
                        _ladder = last_td_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_td_info["step"][col]
                                if step < n_td_steps:
                                    _stop_exit_size = get_time_stop_ladder_exit_size_nb(
                                        stop_=td_stop_,
                                        step=step,
                                        col=col,
                                        init_idx=last_td_info["init_idx"][col],
                                        init_position=last_td_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        time_delta_format=last_td_info["time_delta_format"][col],
                                        index=index,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    elif dt_hit:
                        stop_price, stop_hit_on_open, stop_hit = dt_stop_price, dt_stop_hit_on_open, dt_stop_hit
                        _stop_type = StopType.DT
                        _init_i = last_dt_info["init_idx"][col]
                        _stop_exit_price = last_dt_info["exit_price"][col]
                        _stop_exit_size = last_dt_info["exit_size"][col]
                        _stop_exit_size_type = last_dt_info["exit_size_type"][col]
                        _stop_exit_type = last_dt_info["exit_type"][col]
                        _stop_order_type = last_dt_info["order_type"][col]
                        _limit_delta = last_dt_info["limit_delta"][col]
                        _delta_format = last_dt_info["delta_format"][col]
                        _ladder = last_dt_info["ladder"][col]
                        if np.isnan(_stop_exit_size):
                            if stop_ladder and _ladder and _ladder != StopLadderMode.Dynamic:
                                step = last_dt_info["step"][col]
                                if step < n_dt_steps:
                                    _stop_exit_size = get_time_stop_ladder_exit_size_nb(
                                        stop_=dt_stop_,
                                        step=step,
                                        col=col,
                                        init_idx=last_dt_info["init_idx"][col],
                                        init_position=last_dt_info["init_position"][col],
                                        position_now=last_position[col],
                                        ladder=_ladder,
                                        time_delta_format=last_dt_info["time_delta_format"][col],
                                        index=index,
                                    )
                                    _stop_exit_size_type = SizeType.Amount
                    else:
                        stop_price, stop_hit_on_open, stop_hit = np.nan, False, False

                    if stop_hit:
                        # Stop price was hit
                        # Resolve the final stop signal
                        _accumulate = flex_select_nb(accumulate_, i, col)
                        _size = flex_select_nb(size_, i, col)
                        _size_type = flex_select_nb(size_type_, i, col)
                        if not np.isnan(_stop_exit_size):
                            _accumulate = True
                            if _stop_exit_type == StopExitType.Close:
                                _stop_exit_type = StopExitType.CloseReduce
                            _size = _stop_exit_size
                        if _stop_exit_size_type != -1:
                            _size_type = _stop_exit_size_type

                        (
                            stop_is_long_entry,
                            stop_is_long_exit,
                            stop_is_short_entry,
                            stop_is_short_exit,
                            _accumulate,
                        ) = generate_stop_signal_nb(
                            position_now=last_position[col],
                            stop_exit_type=_stop_exit_type,
                            accumulate=_accumulate,
                        )

                        # Resolve the price
                        _price = resolve_stop_exit_price_nb(
                            stop_price=stop_price,
                            close=_close,
                            stop_exit_price=_stop_exit_price,
                        )

                        # Convert both signals to size (direction-aware), size type, and direction
                        _size, _size_type, _direction = signal_to_size_nb(
                            position_now=last_position[col],
                            val_price_now=_price,
                            value_now=last_value[group],
                            is_long_entry=stop_is_long_entry,
                            is_long_exit=stop_is_long_exit,
                            is_short_entry=stop_is_short_entry,
                            is_short_exit=stop_is_short_exit,
                            size=_size,
                            size_type=_size_type,
                            accumulate=_accumulate,
                        )

                        if not np.isnan(_size):
                            # Executable stop signal
                            can_execute = True
                            if _stop_order_type == OrderType.Limit:
                                # Use close to check whether the limit price was hit
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    price=_price,
                                    size=_size,
                                    direction=_direction,
                                    limit_delta=_limit_delta,
                                    delta_format=_delta_format,
                                    limit_reverse=False,
                                    can_use_ohlc=stop_hit_on_open,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                            # Save info
                            exec_stop_set = True
                            exec_stop_set_on_open = stop_hit_on_open
                            exec_stop_set_on_close = _stop_exit_price == StopExitPrice.Close
                            exec_stop_init_i = _init_i
                            if np.isinf(_price) and _price > 0:
                                exec_stop_val_price = _close
                            elif np.isinf(_price) and _price < 0:
                                exec_stop_val_price = _open
                            else:
                                exec_stop_val_price = _price
                            exec_stop_price = _price
                            exec_stop_size = _size
                            exec_stop_size_type = _size_type
                            exec_stop_direction = _direction
                            exec_stop_type = _stop_order_type
                            exec_stop_stop_type = _stop_type
                            exec_stop_delta = _limit_delta
                            exec_stop_delta_format = _delta_format
                            exec_stop_make_limit = not can_execute

                # Process user signal
                if any_user_signal:
                    if _i < 0:
                        _price = np.nan
                        _size = np.nan
                        _size_type = -1
                        _direction = -1
                    else:
                        _accumulate = flex_select_nb(accumulate_, _i, col)
                        if is_long_entry or is_short_entry:
                            # Resolve any single-direction conflicts
                            _upon_long_conflict = flex_select_nb(upon_long_conflict_, _i, col)
                            is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                                position_now=last_position[col],
                                is_entry=is_long_entry,
                                is_exit=is_long_exit,
                                direction=Direction.LongOnly,
                                conflict_mode=_upon_long_conflict,
                            )
                            _upon_short_conflict = flex_select_nb(upon_short_conflict_, _i, col)
                            is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                                position_now=last_position[col],
                                is_entry=is_short_entry,
                                is_exit=is_short_exit,
                                direction=Direction.ShortOnly,
                                conflict_mode=_upon_short_conflict,
                            )

                            # Resolve any multi-direction conflicts
                            _upon_dir_conflict = flex_select_nb(upon_dir_conflict_, _i, col)
                            is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                                position_now=last_position[col],
                                is_long_entry=is_long_entry,
                                is_short_entry=is_short_entry,
                                upon_dir_conflict=_upon_dir_conflict,
                            )

                            # Resolve an opposite entry
                            _upon_opposite_entry = flex_select_nb(upon_opposite_entry_, _i, col)
                            (
                                is_long_entry,
                                is_long_exit,
                                is_short_entry,
                                is_short_exit,
                                _accumulate,
                            ) = resolve_opposite_entry_nb(
                                position_now=last_position[col],
                                is_long_entry=is_long_entry,
                                is_long_exit=is_long_exit,
                                is_short_entry=is_short_entry,
                                is_short_exit=is_short_exit,
                                upon_opposite_entry=_upon_opposite_entry,
                                accumulate=_accumulate,
                            )

                        # Resolve the price
                        _price = flex_select_nb(price_, _i, col)

                        # Convert both signals to size (direction-aware), size type, and direction
                        _val_price = flex_select_nb(val_price_, i, col)
                        if np.isinf(_val_price) and _val_price > 0:
                            if np.isinf(_price) and _price > 0:
                                _val_price = _close
                            elif np.isinf(_price) and _price < 0:
                                _val_price = _open
                            else:
                                _val_price = _price
                        elif np.isnan(_val_price) or (np.isinf(_val_price) and _val_price < 0):
                            _val_price = last_val_price[col]
                        _size, _size_type, _direction = signal_to_size_nb(
                            position_now=last_position[col],
                            val_price_now=_val_price,
                            value_now=last_value[group],
                            is_long_entry=is_long_entry,
                            is_long_exit=is_long_exit,
                            is_short_entry=is_short_entry,
                            is_short_exit=is_short_exit,
                            size=flex_select_nb(size_, _i, col),
                            size_type=flex_select_nb(size_type_, _i, col),
                            accumulate=_accumulate,
                        )

                    if np.isinf(_price):
                        if _price > 0:
                            user_on_close = True
                        else:
                            user_on_open = True
                    if not np.isnan(_size):
                        # Executable user signal
                        can_execute = True
                        _order_type = flex_select_nb(order_type_, _i, col)
                        if _order_type == OrderType.Limit:
                            # Use close to check whether the limit price was hit
                            can_use_ohlc = False
                            if np.isinf(_price):
                                if _price > 0:
                                    # Cannot place a limit order at the close price and execute right away
                                    _price = _close
                                    can_execute = False
                                else:
                                    can_use_ohlc = True
                                    _price = _open
                            if can_execute:
                                _limit_delta = flex_select_nb(limit_delta_, _i, col)
                                _delta_format = flex_select_nb(delta_format_, _i, col)
                                _limit_reverse = flex_select_nb(limit_reverse_, _i, col)
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    open=_open,
                                    high=_high,
                                    low=_low,
                                    close=_close,
                                    price=_price,
                                    size=_size,
                                    direction=_direction,
                                    limit_delta=_limit_delta,
                                    delta_format=_delta_format,
                                    limit_reverse=_limit_reverse,
                                    can_use_ohlc=can_use_ohlc,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                        # Save info
                        exec_user_set = True
                        exec_user_val_price = _val_price
                        exec_user_price = _price
                        exec_user_size = _size
                        exec_user_size_type = _size_type
                        exec_user_direction = _direction
                        exec_user_type = _order_type
                        exec_user_stop_type = -1
                        exec_user_make_limit = not can_execute

                if (
                    exec_limit_set
                    or exec_stop_set
                    or exec_user_set
                    or ((any_limit_signal or any_stop_signal) and any_user_signal)
                ):
                    # Choose the main executable signal
                    # Priority: limit -> stop -> user

                    # Check whether the main signal comes on open
                    keep_limit = True
                    keep_stop = True
                    execute_limit = False
                    execute_stop = False
                    execute_user = False
                    if exec_limit_set_on_open:
                        keep_limit = False
                        keep_stop = False
                        execute_limit = True
                        exec_limit_bar_zone = BarZone.Open
                    elif exec_stop_set_on_open:
                        keep_limit = False
                        keep_stop = _ladder
                        execute_stop = True
                        if exec_stop_set_on_close:
                            exec_stop_bar_zone = BarZone.Close
                        else:
                            exec_stop_bar_zone = BarZone.Open
                    elif any_user_signal and user_on_open:
                        execute_user = True
                        if any_limit_signal and (execute_user or not exec_user_set):
                            stop_size = get_diraware_size_nb(
                                size=last_limit_info["init_size"][col],
                                direction=last_limit_info["init_direction"][col],
                            )
                            keep_limit, execute_user = resolve_pending_conflict_nb(
                                is_pending_long=stop_size >= 0,
                                is_user_long=is_long_entry or is_short_exit,
                                upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                            )
                        if any_stop_signal and (execute_user or not exec_user_set):
                            keep_stop, execute_user = resolve_pending_conflict_nb(
                                is_pending_long=last_position[col] < 0,
                                is_user_long=is_long_entry or is_short_exit,
                                upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                            )
                        if not exec_user_set:
                            execute_user = False
                        if execute_user:
                            exec_user_bar_zone = BarZone.Open
                    if not execute_limit and not execute_stop and not execute_user:
                        # Check whether the main signal comes in the middle of the bar
                        if exec_limit_set and not exec_limit_set_on_open and keep_limit:
                            keep_limit = False
                            keep_stop = False
                            execute_limit = True
                            exec_limit_bar_zone = BarZone.Middle
                        elif exec_stop_set and not exec_stop_set_on_open and not exec_stop_set_on_close and keep_stop:
                            keep_limit = False
                            keep_stop = _ladder
                            execute_stop = True
                            exec_stop_bar_zone = BarZone.Middle
                        elif any_user_signal and not user_on_open and not user_on_close:
                            execute_user = True
                            if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                stop_size = get_diraware_size_nb(
                                    size=last_limit_info["init_size"][col],
                                    direction=last_limit_info["init_direction"][col],
                                )
                                keep_limit, execute_user = resolve_pending_conflict_nb(
                                    is_pending_long=stop_size >= 0,
                                    is_user_long=is_long_entry or is_short_exit,
                                    upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                    upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                                )
                            if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                keep_stop, execute_user = resolve_pending_conflict_nb(
                                    is_pending_long=last_position[col] < 0,
                                    is_user_long=is_long_entry or is_short_exit,
                                    upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                    upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                                )
                            if not exec_user_set:
                                execute_user = False
                            if execute_user:
                                exec_user_bar_zone = BarZone.Middle
                        if not execute_limit and not execute_stop and not execute_user:
                            # Check whether the main signal comes on close
                            if exec_stop_set_on_close and keep_stop:
                                keep_limit = False
                                keep_stop = _ladder
                                execute_stop = True
                                exec_stop_bar_zone = BarZone.Close
                            elif any_user_signal and user_on_close:
                                execute_user = True
                                if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                    stop_size = get_diraware_size_nb(
                                        size=last_limit_info["init_size"][col],
                                        direction=last_limit_info["init_direction"][col],
                                    )
                                    keep_limit, execute_user = resolve_pending_conflict_nb(
                                        is_pending_long=stop_size >= 0,
                                        is_user_long=is_long_entry or is_short_exit,
                                        upon_adj_conflict=flex_select_nb(upon_adj_limit_conflict_, i, col),
                                        upon_opp_conflict=flex_select_nb(upon_opp_limit_conflict_, i, col),
                                    )
                                if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                    keep_stop, execute_user = resolve_pending_conflict_nb(
                                        is_pending_long=last_position[col] < 0,
                                        is_user_long=is_long_entry or is_short_exit,
                                        upon_adj_conflict=flex_select_nb(upon_adj_stop_conflict_, i, col),
                                        upon_opp_conflict=flex_select_nb(upon_opp_stop_conflict_, i, col),
                                    )
                                if not exec_user_set:
                                    execute_user = False
                                if execute_user:
                                    exec_user_bar_zone = BarZone.Close

                    # Process the limit signal
                    if execute_limit:
                        # Execute the signal
                        main_info["bar_zone"][col] = exec_limit_bar_zone
                        main_info["signal_idx"][col] = exec_limit_signal_i
                        main_info["creation_idx"][col] = exec_limit_creation_i
                        main_info["idx"][col] = exec_limit_init_i
                        main_info["val_price"][col] = exec_limit_val_price
                        main_info["price"][col] = exec_limit_price
                        main_info["size"][col] = exec_limit_size
                        main_info["size_type"][col] = exec_limit_size_type
                        main_info["direction"][col] = exec_limit_direction
                        main_info["type"][col] = OrderType.Limit
                        main_info["stop_type"][col] = exec_limit_stop_type
                    if execute_limit or (any_limit_signal and not keep_limit):
                        # Clear the pending info
                        any_limit_signal = False

                        last_limit_info["signal_idx"][col] = -1
                        last_limit_info["creation_idx"][col] = -1
                        last_limit_info["init_idx"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["init_stop_type"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1
                        last_limit_info["reverse"][col] = False

                    # Process the stop signal
                    if execute_stop:
                        # Execute the signal
                        if exec_stop_make_limit:
                            if any_limit_signal:
                                raise ValueError("Only one active limit signal is allowed at a time")

                            _limit_tif = flex_select_nb(limit_tif_, i, col)
                            _limit_expiry = flex_select_nb(limit_expiry_, i, col)
                            _time_delta_format = flex_select_nb(time_delta_format_, i, col)
                            last_limit_info["signal_idx"][col] = exec_stop_init_i
                            last_limit_info["creation_idx"][col] = i
                            last_limit_info["init_idx"][col] = i
                            last_limit_info["init_price"][col] = exec_stop_price
                            last_limit_info["init_size"][col] = exec_stop_size
                            last_limit_info["init_size_type"][col] = exec_stop_size_type
                            last_limit_info["init_direction"][col] = exec_stop_direction
                            last_limit_info["init_stop_type"][col] = exec_stop_stop_type
                            last_limit_info["delta"][col] = exec_stop_delta
                            last_limit_info["delta_format"][col] = exec_stop_delta_format
                            last_limit_info["tif"][col] = _limit_tif
                            last_limit_info["expiry"][col] = _limit_expiry
                            last_limit_info["time_delta_format"][col] = _time_delta_format
                            last_limit_info["reverse"][col] = False
                        else:
                            main_info["bar_zone"][col] = exec_stop_bar_zone
                            main_info["signal_idx"][col] = exec_stop_init_i
                            main_info["creation_idx"][col] = i
                            main_info["idx"][col] = i
                            main_info["val_price"][col] = exec_stop_val_price
                            main_info["price"][col] = exec_stop_price
                            main_info["size"][col] = exec_stop_size
                            main_info["size_type"][col] = exec_stop_size_type
                            main_info["direction"][col] = exec_stop_direction
                            main_info["type"][col] = exec_stop_type
                            main_info["stop_type"][col] = exec_stop_stop_type

                    if any_stop_signal and not keep_stop:
                        # Clear the pending info
                        any_stop_signal = False

                        last_sl_info["init_idx"][col] = -1
                        last_sl_info["init_price"][col] = np.nan
                        last_sl_info["init_position"][col] = np.nan
                        last_sl_info["stop"][col] = np.nan
                        last_sl_info["exit_price"][col] = -1
                        last_sl_info["exit_size"][col] = np.nan
                        last_sl_info["exit_size_type"][col] = -1
                        last_sl_info["exit_type"][col] = -1
                        last_sl_info["order_type"][col] = -1
                        last_sl_info["limit_delta"][col] = np.nan
                        last_sl_info["delta_format"][col] = -1
                        last_sl_info["ladder"][col] = -1
                        last_sl_info["step"][col] = -1
                        last_sl_info["step_idx"][col] = -1

                        last_tsl_info["init_idx"][col] = -1
                        last_tsl_info["init_price"][col] = np.nan
                        last_tsl_info["init_position"][col] = np.nan
                        last_tsl_info["peak_idx"][col] = -1
                        last_tsl_info["peak_price"][col] = np.nan
                        last_tsl_info["stop"][col] = np.nan
                        last_tsl_info["th"][col] = np.nan
                        last_tsl_info["exit_price"][col] = -1
                        last_tsl_info["exit_size"][col] = np.nan
                        last_tsl_info["exit_size_type"][col] = -1
                        last_tsl_info["exit_type"][col] = -1
                        last_tsl_info["order_type"][col] = -1
                        last_tsl_info["limit_delta"][col] = np.nan
                        last_tsl_info["delta_format"][col] = -1
                        last_tsl_info["ladder"][col] = -1
                        last_tsl_info["step"][col] = -1
                        last_tsl_info["step_idx"][col] = -1

                        last_tp_info["init_idx"][col] = -1
                        last_tp_info["init_price"][col] = np.nan
                        last_tp_info["init_position"][col] = np.nan
                        last_tp_info["stop"][col] = np.nan
                        last_tp_info["exit_price"][col] = -1
                        last_tp_info["exit_size"][col] = np.nan
                        last_tp_info["exit_size_type"][col] = -1
                        last_tp_info["exit_type"][col] = -1
                        last_tp_info["order_type"][col] = -1
                        last_tp_info["limit_delta"][col] = np.nan
                        last_tp_info["delta_format"][col] = -1
                        last_tp_info["ladder"][col] = -1
                        last_tp_info["step"][col] = -1
                        last_tp_info["step_idx"][col] = -1

                        last_td_info["init_idx"][col] = -1
                        last_td_info["init_position"][col] = np.nan
                        last_td_info["stop"][col] = -1
                        last_td_info["exit_price"][col] = -1
                        last_td_info["exit_size"][col] = np.nan
                        last_td_info["exit_size_type"][col] = -1
                        last_td_info["exit_type"][col] = -1
                        last_td_info["order_type"][col] = -1
                        last_td_info["limit_delta"][col] = np.nan
                        last_td_info["delta_format"][col] = -1
                        last_td_info["time_delta_format"][col] = -1
                        last_td_info["ladder"][col] = -1
                        last_td_info["step"][col] = -1
                        last_td_info["step_idx"][col] = -1

                        last_dt_info["init_idx"][col] = -1
                        last_dt_info["init_position"][col] = np.nan
                        last_dt_info["stop"][col] = -1
                        last_dt_info["exit_price"][col] = -1
                        last_dt_info["exit_size"][col] = np.nan
                        last_dt_info["exit_size_type"][col] = -1
                        last_dt_info["exit_type"][col] = -1
                        last_dt_info["order_type"][col] = -1
                        last_dt_info["limit_delta"][col] = np.nan
                        last_dt_info["delta_format"][col] = -1
                        last_dt_info["time_delta_format"][col] = -1
                        last_dt_info["ladder"][col] = -1
                        last_dt_info["step"][col] = -1
                        last_dt_info["step_idx"][col] = -1

                    # Process the user signal
                    if execute_user:
                        # Execute the signal
                        if _i >= 0:
                            if exec_user_make_limit:
                                if any_limit_signal:
                                    raise ValueError("Only one active limit signal is allowed at a time")

                                _limit_delta = flex_select_nb(limit_delta_, _i, col)
                                _delta_format = flex_select_nb(delta_format_, _i, col)
                                _limit_tif = flex_select_nb(limit_tif_, _i, col)
                                _limit_expiry = flex_select_nb(limit_expiry_, _i, col)
                                _time_delta_format = flex_select_nb(time_delta_format_, _i, col)
                                _limit_reverse = flex_select_nb(limit_reverse_, _i, col)
                                last_limit_info["signal_idx"][col] = _i
                                last_limit_info["creation_idx"][col] = i
                                last_limit_info["init_idx"][col] = _i
                                last_limit_info["init_price"][col] = exec_user_price
                                last_limit_info["init_size"][col] = exec_user_size
                                last_limit_info["init_size_type"][col] = exec_user_size_type
                                last_limit_info["init_direction"][col] = exec_user_direction
                                last_limit_info["init_stop_type"][col] = -1
                                last_limit_info["delta"][col] = _limit_delta
                                last_limit_info["delta_format"][col] = _delta_format
                                last_limit_info["tif"][col] = _limit_tif
                                last_limit_info["expiry"][col] = _limit_expiry
                                last_limit_info["time_delta_format"][col] = _time_delta_format
                                last_limit_info["reverse"][col] = _limit_reverse
                            else:
                                main_info["bar_zone"][col] = exec_user_bar_zone
                                main_info["signal_idx"][col] = _i
                                main_info["creation_idx"][col] = i
                                main_info["idx"][col] = _i
                                main_info["val_price"][col] = exec_user_val_price
                                main_info["price"][col] = exec_user_price
                                main_info["size"][col] = exec_user_size
                                main_info["size_type"][col] = exec_user_size_type
                                main_info["direction"][col] = exec_user_direction
                                main_info["type"][col] = exec_user_type
                                main_info["stop_type"][col] = exec_user_stop_type

            any_signal_set = False
            for col in range(from_col, to_col):
                if not np.isnan(main_info["size"][col]):
                    any_signal_set = True
                    break
            if any_signal_set:
                # Check bar zone and update valuation price
                bar_zone = -1
                same_bar_zone = True
                same_timing = True
                for c in range(group_len):
                    col = from_col + c
                    if np.isnan(main_info["size"][col]):
                        continue
                    if bar_zone == -1:
                        bar_zone = main_info["bar_zone"][col]
                    if main_info["bar_zone"][col] != bar_zone:
                        same_bar_zone = False
                        same_timing = False
                    if main_info["bar_zone"][col] == BarZone.Middle:
                        same_timing = False
                    _val_price = main_info["val_price"][col]
                    if not np.isnan(_val_price) or not ffill_val_price:
                        last_val_price[col] = _val_price

                if cash_sharing:
                    # Dynamically sort by order value -> selling comes first to release funds early
                    if call_seq is None:
                        for c in range(group_len):
                            temp_call_seq[c] = c
                        call_seq_now = temp_call_seq[:group_len]
                    else:
                        call_seq_now = call_seq[i, from_col:to_col]
                    if auto_call_seq:
                        # Sort by order value
                        if not same_timing:
                            raise ValueError("Cannot sort orders by value if they are executed at different times")
                        for c in range(group_len):
                            if call_seq_now[c] != c:
                                raise ValueError("Call sequence must follow CallSeqType.Default")
                            col = from_col + c
                            if np.isnan(main_info["size"][col]):
                                continue
                            # Approximate order value
                            exec_state = ExecState(
                                cash=last_cash[group] if cash_sharing else last_cash[col],
                                position=last_position[col],
                                debt=last_debt[col],
                                locked_cash=last_locked_cash[col],
                                free_cash=last_free_cash[group] if cash_sharing else last_free_cash[col],
                                val_price=last_val_price[col],
                                value=last_value[group] if cash_sharing else last_value[col],
                            )
                            temp_sort_by[c] = approx_order_value_nb(
                                exec_state=exec_state,
                                size=main_info["size"][col],
                                size_type=main_info["size_type"][col],
                                direction=main_info["direction"][col],
                            )
                        insert_argsort_nb(temp_sort_by[:group_len], call_seq_now)
                    else:
                        if not same_bar_zone:
                            # Sort by bar zone
                            for c in range(group_len):
                                if call_seq_now[c] != c:
                                    raise ValueError("Call sequence must follow CallSeqType.Default")
                                col = from_col + c
                                if np.isnan(main_info["size"][col]):
                                    continue
                                temp_sort_by[c] = main_info["bar_zone"][col]
                            insert_argsort_nb(temp_sort_by[:group_len], call_seq_now)

                for k in range(group_len):
                    if cash_sharing:
                        c = call_seq_now[k]
                        if c >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        c = k
                    col = from_col + c
                    if np.isnan(main_info["size"][col]):  # shortcut
                        continue

                    # Get current values per column
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    cash_now = last_cash[group] if cash_sharing else last_cash[col]
                    free_cash_now = last_free_cash[group] if cash_sharing else last_free_cash[col]
                    value_now = last_value[group] if cash_sharing else last_value[col]
                    return_now = last_return[group] if cash_sharing else last_return[col]

                    # Generate the next order
                    _i = main_info["idx"][col]
                    if main_info["type"][col] == OrderType.Limit:
                        _slippage = 0.0
                    else:
                        _slippage = flex_select_nb(slippage_, _i, col)
                    order = order_nb(
                        size=main_info["size"][col],
                        price=main_info["price"][col],
                        size_type=main_info["size_type"][col],
                        direction=main_info["direction"][col],
                        fees=flex_select_nb(fees_, _i, col),
                        fixed_fees=flex_select_nb(fixed_fees_, _i, col),
                        slippage=_slippage,
                        min_size=flex_select_nb(min_size_, _i, col),
                        max_size=flex_select_nb(max_size_, _i, col),
                        size_granularity=flex_select_nb(size_granularity_, _i, col),
                        leverage=flex_select_nb(leverage_, _i, col),
                        leverage_mode=flex_select_nb(leverage_mode_, _i, col),
                        reject_prob=flex_select_nb(reject_prob_, _i, col),
                        price_area_vio_mode=flex_select_nb(price_area_vio_mode_, _i, col),
                        allow_partial=flex_select_nb(allow_partial_, _i, col),
                        raise_reject=flex_select_nb(raise_reject_, _i, col),
                        log=flex_select_nb(log_, _i, col),
                    )

                    # Process the order
                    price_area = PriceArea(
                        open=flex_select_nb(open_, i, col),
                        high=flex_select_nb(high_, i, col),
                        low=flex_select_nb(low_, i, col),
                        close=flex_select_nb(close_, i, col),
                    )
                    exec_state = ExecState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        locked_cash=locked_cash_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                    )
                    order_result, new_exec_state = process_order_nb(
                        group=group,
                        col=col,
                        i=i,
                        exec_state=exec_state,
                        order=order,
                        price_area=price_area,
                        update_value=update_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                    )

                    # Append more order information
                    if order_result.status == OrderStatus.Filled:
                        if order_counts[col] >= 1:
                            order_records["signal_idx"][order_counts[col] - 1, col] = main_info["signal_idx"][col]
                            order_records["creation_idx"][order_counts[col] - 1, col] = main_info["creation_idx"][col]
                            order_records["type"][order_counts[col] - 1, col] = main_info["type"][col]
                            order_records["stop_type"][order_counts[col] - 1, col] = main_info["stop_type"][col]

                    # Update execution state
                    cash_now = new_exec_state.cash
                    position_now = new_exec_state.position
                    debt_now = new_exec_state.debt
                    locked_cash_now = new_exec_state.locked_cash
                    free_cash_now = new_exec_state.free_cash
                    val_price_now = new_exec_state.val_price
                    value_now = new_exec_state.value

                    # Update position record
                    if fill_pos_info:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_info_nb(
                                last_pos_info[col],
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    if use_stops:
                        # Update stop price
                        if position_now == 0:
                            # Not in position anymore -> clear stops (irrespective of order success)
                            last_sl_info["init_idx"][col] = -1
                            last_sl_info["init_price"][col] = np.nan
                            last_sl_info["init_position"][col] = np.nan
                            last_sl_info["stop"][col] = np.nan
                            last_sl_info["exit_price"][col] = -1
                            last_sl_info["exit_size"][col] = np.nan
                            last_sl_info["exit_size_type"][col] = -1
                            last_sl_info["exit_type"][col] = -1
                            last_sl_info["order_type"][col] = -1
                            last_sl_info["limit_delta"][col] = np.nan
                            last_sl_info["delta_format"][col] = -1
                            last_sl_info["ladder"][col] = -1
                            last_sl_info["step"][col] = -1
                            last_sl_info["step_idx"][col] = -1

                            last_tsl_info["init_idx"][col] = -1
                            last_tsl_info["init_price"][col] = np.nan
                            last_tsl_info["init_position"][col] = np.nan
                            last_tsl_info["peak_idx"][col] = -1
                            last_tsl_info["peak_price"][col] = np.nan
                            last_tsl_info["stop"][col] = np.nan
                            last_tsl_info["th"][col] = np.nan
                            last_tsl_info["exit_price"][col] = -1
                            last_tsl_info["exit_size"][col] = np.nan
                            last_tsl_info["exit_size_type"][col] = -1
                            last_tsl_info["exit_type"][col] = -1
                            last_tsl_info["order_type"][col] = -1
                            last_tsl_info["limit_delta"][col] = np.nan
                            last_tsl_info["delta_format"][col] = -1
                            last_tsl_info["ladder"][col] = -1
                            last_tsl_info["step"][col] = -1
                            last_tsl_info["step_idx"][col] = -1

                            last_tp_info["init_idx"][col] = -1
                            last_tp_info["init_price"][col] = np.nan
                            last_tp_info["init_position"][col] = np.nan
                            last_tp_info["stop"][col] = np.nan
                            last_tp_info["exit_price"][col] = -1
                            last_tp_info["exit_size"][col] = np.nan
                            last_tp_info["exit_size_type"][col] = -1
                            last_tp_info["exit_type"][col] = -1
                            last_tp_info["order_type"][col] = -1
                            last_tp_info["limit_delta"][col] = np.nan
                            last_tp_info["delta_format"][col] = -1
                            last_tp_info["ladder"][col] = -1
                            last_tp_info["step"][col] = -1
                            last_tp_info["step_idx"][col] = -1

                            last_td_info["init_idx"][col] = -1
                            last_td_info["init_position"][col] = np.nan
                            last_td_info["stop"][col] = -1
                            last_td_info["exit_price"][col] = -1
                            last_td_info["exit_size"][col] = np.nan
                            last_td_info["exit_size_type"][col] = -1
                            last_td_info["exit_type"][col] = -1
                            last_td_info["order_type"][col] = -1
                            last_td_info["limit_delta"][col] = np.nan
                            last_td_info["delta_format"][col] = -1
                            last_td_info["time_delta_format"][col] = -1
                            last_td_info["ladder"][col] = -1
                            last_td_info["step"][col] = -1
                            last_td_info["step_idx"][col] = -1

                            last_dt_info["init_idx"][col] = -1
                            last_dt_info["init_position"][col] = np.nan
                            last_dt_info["stop"][col] = -1
                            last_dt_info["exit_price"][col] = -1
                            last_dt_info["exit_size"][col] = np.nan
                            last_dt_info["exit_size_type"][col] = -1
                            last_dt_info["exit_type"][col] = -1
                            last_dt_info["order_type"][col] = -1
                            last_dt_info["limit_delta"][col] = np.nan
                            last_dt_info["delta_format"][col] = -1
                            last_dt_info["time_delta_format"][col] = -1
                            last_dt_info["ladder"][col] = -1
                            last_dt_info["step"][col] = -1
                            last_dt_info["step_idx"][col] = -1
                        else:
                            if main_info["stop_type"][col] == StopType.SL:
                                if last_sl_info["ladder"][col]:
                                    step = last_sl_info["step"][col] + 1
                                    last_sl_info["exit_size"][col] = np.nan
                                    last_sl_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_sl_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_sl_steps:
                                            last_sl_info["stop"][col] = flex_select_nb(sl_stop_, step, col)
                                            last_sl_info["step"][col] = step
                                            last_sl_info["step_idx"][col] = i
                                        else:
                                            last_sl_info["stop"][col] = np.nan
                                            last_sl_info["step"][col] = -1
                                            last_sl_info["step_idx"][col] = -1
                                    else:
                                        last_sl_info["stop"][col] = np.nan
                                        last_sl_info["step"][col] = step
                                        last_sl_info["step_idx"][col] = i
                            elif (
                                main_info["stop_type"][col] == StopType.TSL
                                or main_info["stop_type"][col] == StopType.TTP
                            ):
                                if last_tsl_info["ladder"][col]:
                                    step = last_tsl_info["step"][col] + 1
                                    last_tsl_info["step"][col] = step
                                    last_tsl_info["step_idx"][col] = i
                                    last_tsl_info["exit_size"][col] = np.nan
                                    last_tsl_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_tsl_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_tsl_steps:
                                            last_tsl_info["stop"][col] = flex_select_nb(tsl_stop_, step, col)
                                            last_tsl_info["step"][col] = step
                                            last_tsl_info["step_idx"][col] = i
                                        else:
                                            last_tsl_info["stop"][col] = np.nan
                                            last_tsl_info["step"][col] = -1
                                            last_tsl_info["step_idx"][col] = -1
                                    else:
                                        last_tsl_info["stop"][col] = np.nan
                                        last_tsl_info["step"][col] = step
                                        last_tsl_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.TP:
                                if last_tp_info["ladder"][col]:
                                    step = last_tp_info["step"][col] + 1
                                    last_tp_info["step"][col] = step
                                    last_tp_info["step_idx"][col] = i
                                    last_tp_info["exit_size"][col] = np.nan
                                    last_tp_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_tp_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_tp_steps:
                                            last_tp_info["stop"][col] = flex_select_nb(tp_stop_, step, col)
                                            last_tp_info["step"][col] = step
                                            last_tp_info["step_idx"][col] = i
                                        else:
                                            last_tp_info["stop"][col] = np.nan
                                            last_tp_info["step"][col] = -1
                                            last_tp_info["step_idx"][col] = -1
                                    else:
                                        last_tp_info["stop"][col] = np.nan
                                        last_tp_info["step"][col] = step
                                        last_tp_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.TD:
                                if last_td_info["ladder"][col]:
                                    step = last_td_info["step"][col] + 1
                                    last_td_info["step"][col] = step
                                    last_td_info["step_idx"][col] = i
                                    last_td_info["exit_size"][col] = np.nan
                                    last_td_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_td_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_td_steps:
                                            last_td_info["stop"][col] = flex_select_nb(td_stop_, step, col)
                                            last_td_info["step"][col] = step
                                            last_td_info["step_idx"][col] = i
                                        else:
                                            last_td_info["stop"][col] = -1
                                            last_td_info["step"][col] = -1
                                            last_td_info["step_idx"][col] = -1
                                    else:
                                        last_td_info["stop"][col] = -1
                                        last_td_info["step"][col] = step
                                        last_td_info["step_idx"][col] = i
                            elif main_info["stop_type"][col] == StopType.DT:
                                if last_dt_info["ladder"][col]:
                                    step = last_dt_info["step"][col] + 1
                                    last_dt_info["step"][col] = step
                                    last_dt_info["step_idx"][col] = i
                                    last_dt_info["exit_size"][col] = np.nan
                                    last_dt_info["exit_size_type"][col] = -1
                                    if stop_ladder and last_dt_info["ladder"][col] != StopLadderMode.Dynamic:
                                        if step < n_dt_steps:
                                            last_dt_info["stop"][col] = flex_select_nb(dt_stop_, step, col)
                                            last_dt_info["step"][col] = step
                                            last_dt_info["step_idx"][col] = i
                                        else:
                                            last_dt_info["stop"][col] = -1
                                            last_dt_info["step"][col] = -1
                                            last_dt_info["step_idx"][col] = -1
                                    else:
                                        last_dt_info["stop"][col] = -1
                                        last_dt_info["step"][col] = step
                                        last_dt_info["step_idx"][col] = i

                        if order_result.status == OrderStatus.Filled and position_now != 0:
                            # Order filled and in position -> possibly set stops
                            _price = main_info["price"][col]
                            _stop_entry_price = flex_select_nb(stop_entry_price_, i, col)
                            if _stop_entry_price < 0:
                                if _stop_entry_price == StopEntryPrice.ValPrice:
                                    new_init_price = val_price_now
                                    can_use_ohlc = False
                                elif _stop_entry_price == StopEntryPrice.Price:
                                    new_init_price = order.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                    if np.isinf(new_init_price):
                                        if new_init_price > 0:
                                            new_init_price = flex_select_nb(close_, i, col)
                                        else:
                                            new_init_price = flex_select_nb(open_, i, col)
                                elif _stop_entry_price == StopEntryPrice.FillPrice:
                                    new_init_price = order_result.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                elif _stop_entry_price == StopEntryPrice.Open:
                                    new_init_price = flex_select_nb(open_, i, col)
                                    can_use_ohlc = True
                                elif _stop_entry_price == StopEntryPrice.Close:
                                    new_init_price = flex_select_nb(close_, i, col)
                                    can_use_ohlc = False
                                else:
                                    raise ValueError("Invalid StopEntryPrice option")
                            else:
                                new_init_price = _stop_entry_price
                                can_use_ohlc = False

                            if stop_ladder:
                                _sl_stop = flex_select_nb(sl_stop_, 0, col)
                                _tsl_stop = flex_select_nb(tsl_stop_, 0, col)
                                _tp_stop = flex_select_nb(tp_stop_, 0, col)
                                _td_stop = flex_select_nb(td_stop_, 0, col)
                                _dt_stop = flex_select_nb(dt_stop_, 0, col)
                            else:
                                _sl_stop = flex_select_nb(sl_stop_, i, col)
                                _tsl_stop = flex_select_nb(tsl_stop_, i, col)
                                _tp_stop = flex_select_nb(tp_stop_, i, col)
                                _td_stop = flex_select_nb(td_stop_, i, col)
                                _dt_stop = flex_select_nb(dt_stop_, i, col)
                            _tsl_th = flex_select_nb(tsl_th_, i, col)
                            _stop_exit_price = flex_select_nb(stop_exit_price_, i, col)
                            _stop_exit_type = flex_select_nb(stop_exit_type_, i, col)
                            _stop_order_type = flex_select_nb(stop_order_type_, i, col)
                            _stop_limit_delta = flex_select_nb(stop_limit_delta_, i, col)
                            _delta_format = flex_select_nb(delta_format_, i, col)
                            _time_delta_format = flex_select_nb(time_delta_format_, i, col)

                            tsl_updated = False
                            if exec_state.position == 0 or np.sign(position_now) != np.sign(exec_state.position):
                                # Position opened/reversed -> set stops
                                last_sl_info["init_idx"][col] = i
                                last_sl_info["init_price"][col] = new_init_price
                                last_sl_info["init_position"][col] = position_now
                                last_sl_info["stop"][col] = _sl_stop
                                last_sl_info["exit_price"][col] = _stop_exit_price
                                last_sl_info["exit_size"][col] = np.nan
                                last_sl_info["exit_size_type"][col] = -1
                                last_sl_info["exit_type"][col] = _stop_exit_type
                                last_sl_info["order_type"][col] = _stop_order_type
                                last_sl_info["limit_delta"][col] = _stop_limit_delta
                                last_sl_info["delta_format"][col] = _delta_format
                                last_sl_info["ladder"][col] = stop_ladder
                                last_sl_info["step"][col] = 0
                                last_sl_info["step_idx"][col] = i

                                tsl_updated = True
                                last_tsl_info["init_idx"][col] = i
                                last_tsl_info["init_price"][col] = new_init_price
                                last_tsl_info["init_position"][col] = position_now
                                last_tsl_info["peak_idx"][col] = i
                                last_tsl_info["peak_price"][col] = new_init_price
                                last_tsl_info["stop"][col] = _tsl_stop
                                last_tsl_info["th"][col] = _tsl_th
                                last_tsl_info["exit_price"][col] = _stop_exit_price
                                last_tsl_info["exit_size"][col] = np.nan
                                last_tsl_info["exit_size_type"][col] = -1
                                last_tsl_info["exit_type"][col] = _stop_exit_type
                                last_tsl_info["order_type"][col] = _stop_order_type
                                last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                last_tsl_info["delta_format"][col] = _delta_format
                                last_tsl_info["ladder"][col] = stop_ladder
                                last_tsl_info["step"][col] = 0
                                last_tsl_info["step_idx"][col] = i

                                last_tp_info["init_idx"][col] = i
                                last_tp_info["init_price"][col] = new_init_price
                                last_tp_info["init_position"][col] = position_now
                                last_tp_info["stop"][col] = _tp_stop
                                last_tp_info["exit_price"][col] = _stop_exit_price
                                last_tp_info["exit_size"][col] = np.nan
                                last_tp_info["exit_size_type"][col] = -1
                                last_tp_info["exit_type"][col] = _stop_exit_type
                                last_tp_info["order_type"][col] = _stop_order_type
                                last_tp_info["limit_delta"][col] = _stop_limit_delta
                                last_tp_info["delta_format"][col] = _delta_format
                                last_tp_info["ladder"][col] = stop_ladder
                                last_tp_info["step"][col] = 0
                                last_tp_info["step_idx"][col] = i

                                last_td_info["init_idx"][col] = i
                                last_td_info["init_position"][col] = position_now
                                last_td_info["stop"][col] = _td_stop
                                last_td_info["exit_price"][col] = _stop_exit_price
                                last_td_info["exit_size"][col] = np.nan
                                last_td_info["exit_size_type"][col] = -1
                                last_td_info["exit_type"][col] = _stop_exit_type
                                last_td_info["order_type"][col] = _stop_order_type
                                last_td_info["limit_delta"][col] = _stop_limit_delta
                                last_td_info["delta_format"][col] = _delta_format
                                last_td_info["time_delta_format"][col] = _time_delta_format
                                last_td_info["ladder"][col] = stop_ladder
                                last_td_info["step"][col] = 0
                                last_td_info["step_idx"][col] = i

                                last_dt_info["init_idx"][col] = i
                                last_dt_info["init_position"][col] = position_now
                                last_dt_info["stop"][col] = _dt_stop
                                last_dt_info["exit_price"][col] = _stop_exit_price
                                last_dt_info["exit_size"][col] = np.nan
                                last_dt_info["exit_size_type"][col] = -1
                                last_dt_info["exit_type"][col] = _stop_exit_type
                                last_dt_info["order_type"][col] = _stop_order_type
                                last_dt_info["limit_delta"][col] = _stop_limit_delta
                                last_dt_info["delta_format"][col] = _delta_format
                                last_dt_info["time_delta_format"][col] = _time_delta_format
                                last_dt_info["ladder"][col] = stop_ladder
                                last_dt_info["step"][col] = 0
                                last_dt_info["step_idx"][col] = i

                            elif abs(position_now) > abs(exec_state.position):
                                # Position increased -> keep/override stops
                                _upon_stop_update = flex_select_nb(upon_stop_update_, i, col)
                                if should_update_stop_nb(new_stop=_sl_stop, upon_stop_update=_upon_stop_update):
                                    last_sl_info["init_idx"][col] = i
                                    last_sl_info["init_price"][col] = new_init_price
                                    last_sl_info["init_position"][col] = position_now
                                    last_sl_info["stop"][col] = _sl_stop
                                    last_sl_info["exit_price"][col] = _stop_exit_price
                                    last_sl_info["exit_size"][col] = np.nan
                                    last_sl_info["exit_size_type"][col] = -1
                                    last_sl_info["exit_type"][col] = _stop_exit_type
                                    last_sl_info["order_type"][col] = _stop_order_type
                                    last_sl_info["limit_delta"][col] = _stop_limit_delta
                                    last_sl_info["delta_format"][col] = _delta_format
                                    last_sl_info["ladder"][col] = stop_ladder
                                    last_sl_info["step"][col] = 0
                                    last_sl_info["step_idx"][col] = i
                                if should_update_stop_nb(new_stop=_tsl_stop, upon_stop_update=_upon_stop_update):
                                    tsl_updated = True
                                    last_tsl_info["init_idx"][col] = i
                                    last_tsl_info["init_price"][col] = new_init_price
                                    last_tsl_info["init_position"][col] = position_now
                                    last_tsl_info["peak_idx"][col] = i
                                    last_tsl_info["peak_price"][col] = new_init_price
                                    last_tsl_info["stop"][col] = _tsl_stop
                                    last_tsl_info["th"][col] = _tsl_th
                                    last_tsl_info["exit_price"][col] = _stop_exit_price
                                    last_tsl_info["exit_size"][col] = np.nan
                                    last_tsl_info["exit_size_type"][col] = -1
                                    last_tsl_info["exit_type"][col] = _stop_exit_type
                                    last_tsl_info["order_type"][col] = _stop_order_type
                                    last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                    last_tsl_info["delta_format"][col] = _delta_format
                                    last_tsl_info["ladder"][col] = stop_ladder
                                    last_tsl_info["step"][col] = 0
                                    last_tsl_info["step_idx"][col] = i
                                if should_update_stop_nb(new_stop=_tp_stop, upon_stop_update=_upon_stop_update):
                                    last_tp_info["init_idx"][col] = i
                                    last_tp_info["init_price"][col] = new_init_price
                                    last_tp_info["init_position"][col] = position_now
                                    last_tp_info["stop"][col] = _tp_stop
                                    last_tp_info["exit_price"][col] = _stop_exit_price
                                    last_tp_info["exit_size"][col] = np.nan
                                    last_tp_info["exit_size_type"][col] = -1
                                    last_tp_info["exit_type"][col] = _stop_exit_type
                                    last_tp_info["order_type"][col] = _stop_order_type
                                    last_tp_info["limit_delta"][col] = _stop_limit_delta
                                    last_tp_info["delta_format"][col] = _delta_format
                                    last_tp_info["ladder"][col] = stop_ladder
                                    last_tp_info["step"][col] = 0
                                    last_tp_info["step_idx"][col] = i
                                if should_update_time_stop_nb(new_stop=_td_stop, upon_stop_update=_upon_stop_update):
                                    last_td_info["init_idx"][col] = i
                                    last_td_info["init_position"][col] = position_now
                                    last_td_info["stop"][col] = _td_stop
                                    last_td_info["exit_price"][col] = _stop_exit_price
                                    last_td_info["exit_size"][col] = np.nan
                                    last_td_info["exit_size_type"][col] = -1
                                    last_td_info["exit_type"][col] = _stop_exit_type
                                    last_td_info["order_type"][col] = _stop_order_type
                                    last_td_info["limit_delta"][col] = _stop_limit_delta
                                    last_td_info["delta_format"][col] = _delta_format
                                    last_td_info["time_delta_format"][col] = _time_delta_format
                                    last_td_info["ladder"][col] = stop_ladder
                                    last_td_info["step"][col] = 0
                                    last_td_info["step_idx"][col] = i
                                if should_update_time_stop_nb(new_stop=_dt_stop, upon_stop_update=_upon_stop_update):
                                    last_dt_info["init_idx"][col] = i
                                    last_dt_info["init_position"][col] = position_now
                                    last_dt_info["stop"][col] = _dt_stop
                                    last_dt_info["exit_price"][col] = _stop_exit_price
                                    last_dt_info["exit_size"][col] = np.nan
                                    last_dt_info["exit_size_type"][col] = -1
                                    last_dt_info["exit_type"][col] = _stop_exit_type
                                    last_dt_info["order_type"][col] = _stop_order_type
                                    last_dt_info["limit_delta"][col] = _stop_limit_delta
                                    last_dt_info["delta_format"][col] = _delta_format
                                    last_dt_info["time_delta_format"][col] = _time_delta_format
                                    last_dt_info["ladder"][col] = stop_ladder
                                    last_dt_info["step"][col] = 0
                                    last_dt_info["step_idx"][col] = i

                            if tsl_updated:
                                # Update highest/lowest price
                                if can_use_ohlc:
                                    _open = flex_select_nb(open_, i, col)
                                    _high = flex_select_nb(high_, i, col)
                                    _low = flex_select_nb(low_, i, col)
                                    _close = flex_select_nb(close_, i, col)
                                    _high, _low = resolve_hl_nb(
                                        open=_open,
                                        high=_high,
                                        low=_low,
                                        close=_close,
                                    )
                                else:
                                    _open = np.nan
                                    _high = _low = _close = flex_select_nb(close_, i, col)
                                if tsl_updated:
                                    if position_now > 0:
                                        if _high > last_tsl_info["peak_price"][col]:
                                            if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                                last_tsl_info["stop"][col] = (
                                                    last_tsl_info["stop"][col]
                                                    + _high
                                                    - last_tsl_info["peak_price"][col]
                                                )
                                            last_tsl_info["peak_idx"][col] = i
                                            last_tsl_info["peak_price"][col] = _high
                                    elif position_now < 0:
                                        if _low < last_tsl_info["peak_price"][col]:
                                            if last_tsl_info["delta_format"][col] == DeltaFormat.Target:
                                                last_tsl_info["stop"][col] = (
                                                    last_tsl_info["stop"][col] + _low - last_tsl_info["peak_price"][col]
                                                )
                                            last_tsl_info["peak_idx"][col] = i
                                            last_tsl_info["peak_price"][col] = _low

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now

            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_nb(close_, i, col)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                _cash_dividends = flex_select_nb(cash_dividends_, i, col)
                _cash_earnings += _cash_dividends * last_position[col]
                if cash_sharing:
                    if _cash_earnings < 0:
                        _cash_earnings = max(_cash_earnings, -last_cash[group])
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    if _cash_earnings < 0:
                        _cash_earnings = max(_cash_earnings, -last_cash[col])
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings

            # Update value and return
            if cash_sharing:
                group_value = last_cash[group]
                for col in range(from_col, to_col):
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                last_value[group] = group_value
                last_return[group] = returns_nb_.get_return_nb(
                    input_value=prev_close_value[group],
                    output_value=last_value[group] - last_cash_deposits[group],
                )
                prev_close_value[group] = last_value[group]
            else:
                for col in range(from_col, to_col):
                    group_value = last_cash[col]
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                    last_value[col] = group_value
                    last_return[col] = returns_nb_.get_return_nb(
                        input_value=prev_close_value[col], output_value=last_value[col] - last_cash_deposits[col]
                    )
                    prev_close_value[col] = last_value[col]

            # Update open position stats
            if fill_pos_info:
                for col in range(from_col, to_col):
                    update_open_pos_info_stats_nb(last_pos_info[col], last_position[col], last_val_price[col])

            # Call post-segment function
            post_segment_ctx = SignalSegmentContext(
                target_shape=target_shape,
                group_lens=group_lens,
                cash_sharing=cash_sharing,
                index=index,
                freq=freq,
                open=open_,
                high=high_,
                low=low_,
                close=close_,
                init_cash=init_cash_,
                init_position=init_position_,
                init_price=init_price_,
                order_records=order_records,
                order_counts=order_counts,
                log_records=log_records,
                log_counts=log_counts,
                track_cash_deposits=track_cash_deposits,
                cash_deposits_out=cash_deposits_out,
                track_cash_earnings=track_cash_earnings,
                cash_earnings_out=cash_earnings_out,
                in_outputs=in_outputs,
                last_cash=last_cash,
                last_position=last_position,
                last_debt=last_debt,
                last_locked_cash=last_locked_cash,
                last_free_cash=last_free_cash,
                last_val_price=last_val_price,
                last_value=last_value,
                last_return=last_return,
                last_pos_info=last_pos_info,
                last_limit_info=last_limit_info,
                last_sl_info=last_sl_info,
                last_tsl_info=last_tsl_info,
                last_tp_info=last_tp_info,
                last_td_info=last_td_info,
                last_dt_info=last_dt_info,
                group=group,
                group_len=group_len,
                from_col=from_col,
                to_col=to_col,
                i=i,
            )
            post_segment_func_nb(post_segment_ctx, *post_segment_args)

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits_out,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


# % </section>


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        entries=base_ch.FlexArraySlicer(axis=1),
        exits=base_ch.FlexArraySlicer(axis=1),
        direction=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dir_to_ls_signals_nb(
    target_shape: tp.Shape,
    entries: tp.FlexArray2d,
    exits: tp.FlexArray2d,
    direction: tp.FlexArray2d,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Convert direction-unaware to direction-aware signals."""
    long_entries_out = np.empty(target_shape, dtype=np.bool_)
    long_exits_out = np.empty(target_shape, dtype=np.bool_)
    short_entries_out = np.empty(target_shape, dtype=np.bool_)
    short_exits_out = np.empty(target_shape, dtype=np.bool_)
    for col in prange(target_shape[1]):
        for i in range(target_shape[0]):
            is_entry = flex_select_nb(entries, i, col)
            is_exit = flex_select_nb(exits, i, col)
            _direction = flex_select_nb(direction, i, col)
            if _direction == Direction.LongOnly:
                long_entries_out[i, col] = is_entry
                long_exits_out[i, col] = is_exit
                short_entries_out[i, col] = False
                short_exits_out[i, col] = False
            elif _direction == Direction.ShortOnly:
                long_entries_out[i, col] = False
                long_exits_out[i, col] = False
                short_entries_out[i, col] = is_entry
                short_exits_out[i, col] = is_exit
            else:
                long_entries_out[i, col] = is_entry
                long_exits_out[i, col] = False
                short_entries_out[i, col] = is_exit
                short_exits_out[i, col] = False
    return long_entries_out, long_exits_out, short_entries_out, short_exits_out


# % <block holding_enex_signal_func_nb>
@register_jitted
def holding_enex_signal_func_nb(  # % line.replace("holding_enex_signal_func_nb", "signal_func_nb")
    c: SignalContext,
    direction: int,
    close_at_end: bool,
) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals for holding."""
    if c.last_position[c.col] == 0:
        if direction == Direction.ShortOnly:
            return False, False, True, False
        return True, False, False, False
    if close_at_end and c.i == c.target_shape[0] - 1:
        if c.last_position[c.col] < 0:
            return False, False, False, True
        return False, True, False, False
    return False, False, False, False


# % </block>


AdjustFuncT = tp.Callable[[SignalContext, tp.VarArg()], None]


@register_jitted
def no_adjust_func_nb(c: SignalContext, *args) -> None:
    """Placeholder adjustment function."""
    return None


# % <block adjust_func_nb>
# % <skip? skip_func(out_lines, "adjust_func_nb")>
# % <uncomment>
# @register_jitted
# def adjust_func_nb(
#     c: SignalContext,
# ) -> None:
#     """Custom adjustment function."""
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block dir_signal_func_nb>
# % blocks["adjust_func_nb"]
@register_jitted
def dir_signal_func_nb(  # % line.replace("dir_signal_func_nb", "signal_func_nb")
    c: SignalContext,
    entries: tp.FlexArray2d,
    exits: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    from_ago: tp.FlexArray2d,
    adjust_func_nb: AdjustFuncT = no_adjust_func_nb,  # % None
    adjust_args: tp.Args = (),
) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals out of entries, exits, and direction.

    The direction of each pair of signals is taken from `direction` argument:

    * True, True, `Direction.LongOnly` -> True, True, False, False
    * True, True, `Direction.ShortOnly` -> False, False, True, True
    * True, True, `Direction.Both` -> True, False, True, False

    Best to use when the direction doesn't change throughout time.

    Prior to returning the signals, calls user-defined `adjust_func_nb`, which can be used to adjust
    stop values in the context. Must accept `vectorbtpro.portfolio.enums.SignalContext` and `*adjust_args`,
    and return nothing."""
    adjust_func_nb(c, *adjust_args)

    _i = c.i - abs(flex_select_nb(from_ago, c.i, c.col))
    if _i < 0:
        return False, False, False, False
    is_entry = flex_select_nb(entries, _i, c.col)
    is_exit = flex_select_nb(exits, _i, c.col)
    _direction = flex_select_nb(direction, _i, c.col)
    if _direction == Direction.LongOnly:
        return is_entry, is_exit, False, False
    if _direction == Direction.ShortOnly:
        return False, False, is_entry, is_exit
    return is_entry, False, is_exit, False


# % </block>


# % <block ls_signal_func_nb>
# % blocks["adjust_func_nb"]
@register_jitted
def ls_signal_func_nb(  # % line.replace("ls_signal_func_nb", "signal_func_nb")
    c: SignalContext,
    long_entries: tp.FlexArray2d,
    long_exits: tp.FlexArray2d,
    short_entries: tp.FlexArray2d,
    short_exits: tp.FlexArray2d,
    from_ago: tp.FlexArray2d,
    adjust_func_nb: AdjustFuncT = no_adjust_func_nb,  # % None
    adjust_args: tp.Args = (),
) -> tp.Tuple[bool, bool, bool, bool]:
    """Get an element of direction-aware signals.

    The direction is already built into the arrays. Best to use when the direction changes frequently
    (for example, if you have one indicator providing long signals and one providing short signals).

    Prior to returning the signals, calls user-defined `adjust_func_nb`, which can be used to adjust
    stop values in the context. Must accept `vectorbtpro.portfolio.enums.SignalContext` and `*adjust_args`,
    and return nothing."""
    adjust_func_nb(c, *adjust_args)

    _i = c.i - abs(flex_select_nb(from_ago, c.i, c.col))
    if _i < 0:
        return False, False, False, False
    is_long_entry = flex_select_nb(long_entries, _i, c.col)
    is_long_exit = flex_select_nb(long_exits, _i, c.col)
    is_short_entry = flex_select_nb(short_entries, _i, c.col)
    is_short_exit = flex_select_nb(short_exits, _i, c.col)
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit


# % </block>


# % <block order_signal_func_nb>
# % blocks["adjust_func_nb"]
@register_jitted
def order_signal_func_nb(  # % line.replace("order_signal_func_nb", "signal_func_nb")
    c: SignalContext,
    size: tp.FlexArray2d,
    price: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    min_size: tp.FlexArray2d,
    max_size: tp.FlexArray2d,
    val_price: tp.FlexArray2d,
    from_ago: tp.FlexArray2d,
    adjust_func_nb: AdjustFuncT = no_adjust_func_nb,  # % None
    adjust_args: tp.Args = (),
) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals out of orders.

    You must ensure that `size`, `size_type`, `min_size`, and `max_size` are writeable non-flexible arrays
    and accumulation is enabled."""
    adjust_func_nb(c, *adjust_args)

    _i = c.i - abs(flex_select_nb(from_ago, c.i, c.col))
    if _i < 0:
        return False, False, False, False
    order_size = float(flex_select_nb(size, _i, c.col))
    if np.isnan(order_size):
        return False, False, False, False
    order_size_type = int(flex_select_nb(size_type, _i, c.col))
    order_direction = int(flex_select_nb(direction, _i, c.col))
    min_order_size = float(flex_select_nb(min_size, _i, c.col))
    max_order_size = float(flex_select_nb(max_size, _i, c.col))
    order_size = get_diraware_size_nb(order_size, order_direction)

    if (
        order_size_type == SizeType.TargetAmount
        or order_size_type == SizeType.TargetValue
        or order_size_type == SizeType.TargetPercent
        or order_size_type == SizeType.TargetPercent100
    ):
        order_price = flex_select_nb(price, _i, c.col)
        order_val_price = flex_select_nb(val_price, c.i, c.col)
        if np.isinf(order_val_price) and order_val_price > 0:
            if np.isinf(order_price) and order_price > 0:
                order_val_price = flex_select_nb(c.close, c.i, c.col)
            elif np.isinf(order_price) and order_price < 0:
                order_val_price = flex_select_nb(c.open, c.i, c.col)
            else:
                order_val_price = order_price
        elif np.isnan(order_val_price) or (np.isinf(order_val_price) and order_val_price < 0):
            order_val_price = c.last_val_price[c.col]
        order_size, _ = resolve_size_nb(
            size=order_size,
            size_type=order_size_type,
            position=c.last_position[c.col],
            debt=c.last_debt[c.col],
            val_price=order_val_price,
            value=c.last_value[c.group] if c.cash_sharing else c.last_value[c.col],
        )
        if not np.isnan(min_order_size):
            min_order_size, _ = resolve_size_nb(
                size=min_order_size,
                size_type=order_size_type,
                position=c.last_position[c.col],
                debt=c.last_debt[c.col],
                val_price=order_val_price,
                value=c.last_value[c.group] if c.cash_sharing else c.last_value[c.col],
                as_requirement=True,
            )
        if not np.isnan(max_order_size):
            max_order_size, _ = resolve_size_nb(
                size=max_order_size,
                size_type=order_size_type,
                position=c.last_position[c.col],
                debt=c.last_debt[c.col],
                val_price=order_val_price,
                value=c.last_value[c.group] if c.cash_sharing else c.last_value[c.col],
                as_requirement=True,
            )
        order_size_type = SizeType.Amount

        size[_i, c.col] = abs(order_size)
        size_type[_i, c.col] = order_size_type
        min_size[_i, c.col] = min_order_size
        max_size[_i, c.col] = max_order_size
    else:
        size[_i, c.col] = abs(order_size)

    if order_size > 0:
        if c.last_position[c.col] < 0 and order_direction == Direction.ShortOnly:
            return False, False, False, True
        return True, False, False, False
    if order_size < 0:
        if c.last_position[c.col] > 0 and order_direction == Direction.LongOnly:
            return False, True, False, False
        return False, False, True, False
    return False, False, False, False


# % </block>
