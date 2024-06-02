# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio analysis."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import is_close_nb, add_nb
from vectorbtpro.utils.template import RepFunc


# ############# Assets ############# #


@register_jitted(cache=True)
def get_long_size_nb(position_before: float, position_now: float) -> float:
    """Get long size."""
    if position_before <= 0 and position_now <= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_before
    if position_before < 0 and position_now >= 0:
        return position_now
    return add_nb(position_now, -position_before)


@register_jitted(cache=True)
def get_short_size_nb(position_before: float, position_now: float) -> float:
    """Get short size."""
    if position_before >= 0 and position_now >= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_now
    if position_before < 0 and position_now >= 0:
        return position_before
    return add_nb(position_before, -position_now)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        direction=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    direction: int = Direction.Both,
    init_position: tp.FlexArray1dLike = 0.0,
) -> tp.Array2d:
    """Get asset flow series per column.

    Returns the total transacted amount of assets at each time step."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(target_shape, 0.0, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = flex_select_1d_pc_nb(init_position_, col)

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if direction == Direction.LongOnly:
                asset_flow = get_long_size_nb(position_now, new_position_now)
            elif direction == Direction.ShortOnly:
                asset_flow = get_short_size_nb(position_now, new_position_now)
            else:
                asset_flow = size
            out[i, col] = add_nb(out[i, col], asset_flow)
            position_now = new_position_now
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_flow", axis=1),
    arg_take_spec=dict(asset_flow=ch.ArraySlicer(axis=1), init_position=base_ch.FlexArraySlicer()),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def assets_nb(asset_flow: tp.Array2d, init_position: tp.FlexArray1dLike = 0.0) -> tp.Array2d:
    """Get asset series per column.

    Returns the current position at each time step."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    out = np.empty_like(asset_flow)
    for col in prange(asset_flow.shape[1]):
        position_now = flex_select_1d_pc_nb(init_position_, col)
        for i in range(asset_flow.shape[0]):
            flow_value = asset_flow[i, col]
            position_now = add_nb(position_now, flow_value)
            out[i, col] = position_now
    return out


@register_jitted(cache=True)
def long_assets_nb(assets: tp.Array2d) -> tp.Array2d:
    """Get long-only assets."""
    return np.where(assets > 0, assets, 0.0)


@register_jitted(cache=True)
def short_assets_nb(assets: tp.Array2d) -> tp.Array2d:
    """Get short-only assets."""
    return np.where(assets < 0, -assets, 0.0)


# ############# Cash ############# #


@register_jitted(cache=True)
def get_free_cash_diff_nb(
    position_before: float,
    position_now: float,
    debt_now: float,
    price: float,
    fees: float,
) -> tp.Tuple[float, float]:
    """Get updated debt and free cash flow."""
    size = add_nb(position_now, -position_before)
    final_cash = -size * price - fees
    if is_close_nb(size, 0):
        new_debt = debt_now
        free_cash_diff = 0.0
    elif size > 0:
        if position_before < 0:
            if position_now < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_before)
            avg_entry_price = debt_now / abs(position_before)
            debt_diff = short_size * avg_entry_price
            new_debt = add_nb(debt_now, -debt_diff)
            free_cash_diff = add_nb(2 * debt_diff, final_cash)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    else:
        if position_now < 0:
            if position_before < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_now)
            short_value = short_size * price
            new_debt = debt_now + short_value
            free_cash_diff = add_nb(final_cash, -2 * short_value)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    return new_debt, free_cash_diff


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        free=None,
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    free: bool = False,
    cash_earnings: tp.FlexArray2dLike = 0.0,
) -> tp.Array2d:
    """Get (free) cash flow series per column."""
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(target_shape, dtype=np.float_)

    for col in prange(target_shape[1]):
        for i in range(target_shape[0]):
            out[i, col] = flex_select_nb(cash_earnings_, i, col)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.0
        debt_now = 0.0

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]
            price = order_record["price"]
            fees = order_record["fees"]

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if free:
                debt_now, cash_flow = get_free_cash_diff_nb(position_now, new_position_now, debt_now, price, fees)
            else:
                cash_flow = -size * price - fees
            out[i, col] = add_nb(out[i, col], cash_flow)
            position_now = new_position_now
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(arr=base_ch.array_gl_slicer, group_lens=ch.ArraySlicer(axis=0)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sum_grouped_nb(arr: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Squeeze each group of columns into a single column using sum operation."""
    check_group_lens_nb(group_lens, arr.shape[1])
    out = np.empty((arr.shape[0], len(group_lens)), dtype=np.float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        out[:, group] = np.sum(arr[:, from_col:to_col], axis=1)
    return out


@register_jitted(cache=True)
def cash_flow_grouped_nb(cash_flow: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get cash flow series per group."""
    return sum_grouped_nb(cash_flow, group_lens)


@register_chunkable(
    size=ch.ArraySizer(arg_query="free_cash_flow", axis=1),
    arg_take_spec=dict(
        init_cash_raw=None,
        free_cash_flow=ch.ArraySlicer(axis=1),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def align_init_cash_nb(
    init_cash_raw: int,
    free_cash_flow: tp.Array2d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
) -> tp.Array1d:
    """Align initial cash to the maximum negative free cash flow.

    Adds 1 for easier computing returns."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.empty(free_cash_flow.shape[1], dtype=np.float_)
    for col in range(free_cash_flow.shape[1]):
        free_cash = 0.0
        min_req_cash = np.inf
        for i in range(free_cash_flow.shape[0]):
            free_cash = add_nb(free_cash, free_cash_flow[i, col])
            free_cash = add_nb(free_cash, flex_select_nb(cash_deposits_, i, col))
            if free_cash < min_req_cash:
                min_req_cash = free_cash
        if min_req_cash < 0:
            out[col] = np.abs(min_req_cash)
        else:
            out[col] = 1.0
    if init_cash_raw == InitCashMode.AutoAlign:
        out = np.full(out.shape, np.max(out))
    return out


@register_jitted(cache=True)
def init_cash_grouped_nb(init_cash_raw: tp.FlexArray1d, group_lens: tp.Array1d, cash_sharing: bool) -> tp.Array1d:
    """Get initial cash per group."""
    out = np.empty(group_lens.shape, dtype=np.float_)
    if cash_sharing:
        for group in range(len(group_lens)):
            out[group] = flex_select_1d_pc_nb(init_cash_raw, group)
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            cash_sum = 0.0
            for col in range(from_col, to_col):
                cash_sum += flex_select_1d_pc_nb(init_cash_raw, col)
            out[group] = cash_sum
            from_col = to_col
    return out


@register_jitted(cache=True)
def init_cash_nb(
    init_cash_raw: tp.FlexArray1d,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    split_shared: bool = False,
) -> tp.Array1d:
    """Get initial cash per column."""
    out = np.empty(np.sum(group_lens), dtype=np.float_)
    if not cash_sharing:
        for col in range(out.shape[0]):
            out[col] = flex_select_1d_pc_nb(init_cash_raw, col)
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, group)
            for col in range(from_col, to_col):
                if split_shared:
                    out[col] = _init_cash / group_len
                else:
                    out[col] = _init_cash
            from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_grouped_nb(
    target_shape: tp.Shape,
    cash_deposits_raw: tp.FlexArray2d,
    group_lens: tp.Array1d,
    cash_sharing: bool,
) -> tp.Array2d:
    """Get cash deposit series per group."""
    out = np.empty((target_shape[0], len(group_lens)), dtype=np.float_)
    if cash_sharing:
        for group in prange(len(group_lens)):
            for i in range(target_shape[0]):
                out[i, group] = flex_select_nb(cash_deposits_raw, i, group)
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            for i in range(target_shape[0]):
                cash_sum = 0.0
                for col in range(from_col, to_col):
                    cash_sum += flex_select_nb(cash_deposits_raw, i, col)
                out[i, group] = cash_sum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        split_shared=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_nb(
    target_shape: tp.Shape,
    cash_deposits_raw: tp.FlexArray2d,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    split_shared: bool = False,
) -> tp.Array2d:
    """Get cash deposit series per column."""
    out = np.empty(target_shape, dtype=np.float_)
    if not cash_sharing:
        for col in prange(target_shape[1]):
            for i in range(target_shape[0]):
                out[i, col] = flex_select_nb(cash_deposits_raw, i, col)
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            for i in range(target_shape[0]):
                _cash_deposits = flex_select_nb(cash_deposits_raw, i, group)
                for col in range(from_col, to_col):
                    if split_shared:
                        out[i, col] = _cash_deposits / (to_col - from_col)
                    else:
                        out[i, col] = _cash_deposits
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash_flow", axis=1),
    arg_take_spec=dict(
        cash_flow=ch.ArraySlicer(axis=1),
        init_cash=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_nb(
    cash_flow: tp.Array2d,
    init_cash: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
) -> tp.Array2d:
    """Get cash series per column."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.empty_like(cash_flow)
    for col in prange(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            if i == 0:
                cash_now = flex_select_1d_pc_nb(init_cash, col)
            else:
                cash_now = out[i - 1, col]
            cash_now = add_nb(cash_now, flex_select_nb(cash_deposits_, i, col))
            cash_now = add_nb(cash_now, cash_flow[i, col])
            out[i, col] = cash_now
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        cash_flow_grouped=ch.ArraySlicer(axis=1),
        group_lens=ch.ArraySlicer(axis=0),
        init_cash_grouped=base_ch.FlexArraySlicer(),
        cash_deposits_grouped=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_grouped_nb(
    target_shape: tp.Shape,
    cash_flow_grouped: tp.Array2d,
    group_lens: tp.Array1d,
    init_cash_grouped: tp.FlexArray1d,
    cash_deposits_grouped: tp.FlexArray2dLike = 0.0,
) -> tp.Array2d:
    """Get cash series per group."""
    cash_deposits_grouped_ = to_2d_array_nb(np.asarray(cash_deposits_grouped))

    check_group_lens_nb(group_lens, target_shape[1])
    out = np.empty_like(cash_flow_grouped)

    for group in prange(len(group_lens)):
        cash_now = flex_select_1d_pc_nb(init_cash_grouped, group)
        for i in range(cash_flow_grouped.shape[0]):
            flow_value = cash_flow_grouped[i, group]
            cash_now = add_nb(cash_now, flex_select_nb(cash_deposits_grouped_, i, group))
            cash_now = add_nb(cash_now, flow_value)
            out[i, group] = cash_now
    return out


# ############# Value ############# #


@register_jitted(cache=True)
def init_position_value_nb(
    n_cols: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get initial position value per column."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.empty(n_cols, dtype=np.float_)
    for col in range(n_cols):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position == 0:
            out[col] = 0.0
        else:
            out[col] = _init_position * _init_price
    return out


@register_jitted(cache=True)
def init_value_nb(init_position_value: tp.Array1d, init_cash: tp.FlexArray1d) -> tp.Array1d:
    """Get initial value per column."""
    out = np.empty(len(init_position_value), dtype=np.float_)
    for col in range(len(init_position_value)):
        _init_cash = flex_select_1d_pc_nb(init_cash, col)
        out[col] = _init_cash + init_position_value[col]
    return out


@register_jitted(cache=True)
def init_value_grouped_nb(
    group_lens: tp.Array1d,
    init_position_value: tp.Array1d,
    init_cash_grouped: tp.FlexArray1d,
) -> tp.Array1d:
    """Get initial value per group."""
    check_group_lens_nb(group_lens, len(init_position_value))
    out = np.empty(len(group_lens), dtype=np.float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_value = flex_select_1d_pc_nb(init_cash_grouped, group)
        for col in range(from_col, to_col):
            group_value += init_position_value[col]
        out[group] = group_value
        from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(close=ch.ArraySlicer(axis=1), assets=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_value_nb(close: tp.Array2d, assets: tp.Array2d) -> tp.Array2d:
    """Get asset value series per column."""
    out = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        for i in range(close.shape[0]):
            if assets[i, col] == 0:
                out[i, col] = 0.0
            else:
                out[i, col] = close[i, col] * assets[i, col]
    return out


@register_jitted(cache=True)
def asset_value_grouped_nb(asset_value: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get asset value series per group."""
    return sum_grouped_nb(asset_value, group_lens)


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(asset_value=ch.ArraySlicer(axis=1), value=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def gross_exposure_nb(asset_value: tp.Array2d, value: tp.Array2d) -> tp.Array2d:
    """Get gross exposure per column/group."""
    out = np.empty(asset_value.shape, dtype=np.float_)
    for col in prange(asset_value.shape[1]):
        for i in range(asset_value.shape[0]):
            if value[i, col] == 0:
                out[i, col] = np.nan
            else:
                out[i, col] = abs(asset_value[i, col] / value[i, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash", axis=1),
    arg_take_spec=dict(cash=ch.ArraySlicer(axis=1), asset_value=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_nb(cash: tp.Array2d, asset_value: tp.Array2d) -> tp.Array2d:
    """Get portfolio value series per column/group."""
    out = np.empty(cash.shape, dtype=np.float_)
    for col in prange(cash.shape[1]):
        for i in range(cash.shape[0]):
            out[i, col] = cash[i, col] + asset_value[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        asset_value=base_ch.array_gl_slicer,
        value=ch.ArraySlicer(axis=1),
        group_lens=ch.ArraySlicer(axis=0),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def allocations_nb(
    asset_value: tp.Array2d,
    value: tp.Array2d,
    group_lens: tp.Array1d,
) -> tp.Array2d:
    """Get allocations per column."""
    check_group_lens_nb(group_lens, asset_value.shape[1])
    out = np.empty(asset_value.shape, dtype=np.float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for i in range(asset_value.shape[0]):
            for col in range(from_col, to_col):
                out[i, col] = asset_value[i, col] / value[i, group]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def total_profit_nb(
    target_shape: tp.Shape,
    close: tp.Array2d,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_earnings: tp.FlexArray2dLike = 0.0,
) -> tp.Array1d:
    """Get total profit per column.

    A much faster version than the one based on `value_nb`."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    assets = np.full(target_shape[1], 0.0, dtype=np.float_)
    cash = np.full(target_shape[1], 0.0, dtype=np.float_)
    zero_mask = np.full(target_shape[1], False, dtype=np.bool_)

    for col in prange(target_shape[1]):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position != 0:
            assets[col] = _init_position
            cash[col] = -_init_position * _init_price

        for i in range(target_shape[0]):
            cash[col] += flex_select_nb(cash_earnings_, i, col)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            zero_mask[col] = assets[col] == 0 and cash[col] == 0
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            # Fill assets
            if order_record["side"] == OrderSide.Buy:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], order_size)
            else:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], -order_size)

            # Fill cash balance
            if order_record["side"] == OrderSide.Buy:
                order_cash = order_record["size"] * order_record["price"] + order_record["fees"]
                cash[col] = add_nb(cash[col], -order_cash)
            else:
                order_cash = order_record["size"] * order_record["price"] - order_record["fees"]
                cash[col] = add_nb(cash[col], order_cash)

    total_profit = cash + assets * close[-1, :]
    total_profit[zero_mask] = 0.0
    return total_profit


@register_jitted(cache=True)
def total_profit_grouped_nb(total_profit: tp.Array1d, group_lens: tp.Array1d) -> tp.Array1d:
    """Get total profit per group."""
    check_group_lens_nb(group_lens, total_profit.shape[0])
    out = np.empty(len(group_lens), dtype=np.float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="value", axis=1),
    arg_take_spec=dict(
        value=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        log_returns=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    value: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    log_returns: bool = False,
) -> tp.Array2d:
    """Get return series per column/group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.empty(value.shape, dtype=np.float_)
    for col in prange(value.shape[1]):
        input_value = flex_select_1d_pc_nb(init_value, col)
        for i in range(value.shape[0]):
            output_value = value[i, col]
            adj_output_value = output_value - flex_select_nb(cash_deposits_, i, col)
            out[i, col] = returns_nb_.get_return_nb(input_value, adj_output_value, log_returns=log_returns)
            input_value = output_value
    return out


@register_jitted(cache=True)
def get_asset_pnl_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
) -> float:
    """Get asset PnL from the input and output asset value, and the cash flow."""
    return output_asset_value + cash_flow - input_asset_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="init_position_value", axis=0),
    arg_take_spec=dict(
        init_position_value=ch.ArraySlicer(axis=0),
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_pnl_nb(
    init_position_value: tp.Array1d,
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
) -> tp.Array2d:
    """Get asset (realized and unrealized) PnL series per column/group."""
    out = np.empty_like(cash_flow)
    for col in prange(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            if i == 0:
                input_asset_value = 0.0
                _cash_flow = cash_flow[i, col] - init_position_value[col]
            else:
                input_asset_value = asset_value[i - 1, col]
                _cash_flow = cash_flow[i, col]
            out[i, col] = get_asset_pnl_nb(
                input_asset_value,
                asset_value[i, col],
                _cash_flow,
            )
    return out


@register_jitted(cache=True)
def get_asset_return_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
    log_returns: bool = False,
) -> float:
    """Get asset return from the input and output asset value, and the cash flow."""
    if is_close_nb(input_asset_value, 0):
        input_value = -output_asset_value
        output_value = cash_flow
    else:
        input_value = input_asset_value
        output_value = output_asset_value + cash_flow
    if input_value < 0 and output_value < 0:
        return_value = -returns_nb_.get_return_nb(-input_value, -output_value, log_returns=False)
    else:
        return_value = returns_nb_.get_return_nb(input_value, output_value, log_returns=False)
    if log_returns:
        return np.log1p(return_value)
    return return_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="init_position_value", axis=0),
    arg_take_spec=dict(
        init_position_value=ch.ArraySlicer(axis=0),
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
        log_returns=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_returns_nb(
    init_position_value: tp.Array1d,
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
    log_returns: bool = False,
) -> tp.Array2d:
    """Get asset return series per column/group."""
    out = np.empty_like(cash_flow)
    for col in prange(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            if i == 0:
                input_asset_value = 0.0
                _cash_flow = cash_flow[i, col] - init_position_value[col]
            else:
                input_asset_value = asset_value[i - 1, col]
                _cash_flow = cash_flow[i, col]
            out[i, col] = get_asset_return_nb(
                input_asset_value,
                asset_value[i, col],
                _cash_flow,
                log_returns=log_returns,
            )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_nb(
    close: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
) -> tp.Array2d:
    """Get market value per column."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.empty_like(close)
    for col in prange(close.shape[1]):
        curr_value = flex_select_1d_pc_nb(init_value, col)
        for i in range(close.shape[0]):
            if i > 0:
                curr_value *= close[i, col] / close[i - 1, col]
            curr_value += flex_select_nb(cash_deposits_, i, col)
            out[i, col] = curr_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        close=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        init_value=base_ch.FlexArraySlicer(mapper=base_ch.group_lens_mapper),
        cash_deposits=base_ch.FlexArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_grouped_nb(
    close: tp.Array2d,
    group_lens: tp.Array1d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
) -> tp.Array2d:
    """Get market value per group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    check_group_lens_nb(group_lens, close.shape[1])
    out = np.empty((close.shape[0], len(group_lens)), dtype=np.float_)
    temp = np.empty(close.shape[1], dtype=np.float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for i in range(close.shape[0]):
            for col in range(from_col, to_col):
                if i == 0:
                    temp[col] = flex_select_1d_pc_nb(init_value, col)
                else:
                    temp[col] *= close[i, col] / close[i - 1, col]
                temp[col] += flex_select_nb(cash_deposits_, i, col)
            out[i, group] = np.sum(temp[from_col:to_col])
    return out
