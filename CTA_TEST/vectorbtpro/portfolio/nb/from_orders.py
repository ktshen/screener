# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio simulation based on orders."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns.nb import get_return_nb
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
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
        size=base_ch.flex_array_gl_slicer,
        price=base_ch.flex_array_gl_slicer,
        size_type=base_ch.flex_array_gl_slicer,
        direction=base_ch.flex_array_gl_slicer,
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
        skipna=None,
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(cache=True, tags={"can_parallel"})
def from_orders_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
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
    size: tp.FlexArray2dLike = np.inf,
    price: tp.FlexArray2dLike = np.inf,
    size_type: tp.FlexArray2dLike = SizeType.Amount,
    direction: tp.FlexArray2dLike = Direction.Both,
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
    """Creates on order out of each element.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).

    Usage:
        * Buy and hold using all cash and closing price (default):

        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.records.nb import col_map_nb
        >>> from vectorbtpro.portfolio.nb import from_orders_nb, asset_flow_nb

        >>> close = np.array([1, 2, 3, 4, 5])[:, None]
        >>> sim_out = from_orders_nb(
        ...     target_shape=close.shape,
        ...     group_lens=np.array([1]),
        ...     call_seq=np.full(close.shape, 0),
        ...     close=close
        ... )
        >>> col_map = col_map_nb(sim_out.order_records['col'], close.shape[1])
        >>> asset_flow = asset_flow_nb(close.shape, sim_out.order_records, col_map)
        >>> asset_flow
        array([[100.],
               [  0.],
               [  0.],
               [  0.],
               [  0.]])
        ```
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
    size_ = to_2d_array_nb(np.asarray(size))
    price_ = to_2d_array_nb(np.asarray(price))
    size_type_ = to_2d_array_nb(np.asarray(size_type))
    direction_ = to_2d_array_nb(np.asarray(direction))
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
    from_ago_ = to_2d_array_nb(np.asarray(from_ago))

    order_records, log_records = prepare_records_nb(target_shape, max_orders, max_logs)
    last_cash = prepare_last_cash_nb(target_shape, group_lens, cash_sharing, init_cash_)
    last_position = prepare_last_position_nb(target_shape, init_position_)
    last_value = prepare_last_value_nb(
        target_shape,
        group_lens,
        cash_sharing,
        init_cash_,
        init_position=init_position_,
        init_price=init_price_,
    )

    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_locked_cash = np.full(target_shape[1], 0.0, dtype=np.float_)
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)
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

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col
        cash_now = last_cash[group]
        free_cash_now = cash_now

        for i in range(target_shape[0]):
            skip = not ffill_val_price and not save_state and not save_value and not save_returns
            if skip:
                if flex_select_nb(cash_deposits_, i, group) != 0:
                    skip = False
            if skip:
                for c in range(group_len):
                    col = from_col + c
                    if flex_select_nb(cash_earnings_, i, col) != 0:
                        skip = False
                        break
                    if flex_select_nb(cash_dividends_, i, col) != 0:
                        skip = False
                        break
                    _i = i - abs(flex_select_nb(from_ago_, i, col))
                    if _i < 0:
                        continue
                    if not np.isnan(flex_select_nb(size_, _i, col)):
                        if not np.isnan(flex_select_nb(price_, _i, col)):
                            skip = False
                            break
            if skip:
                continue

            # Add cash
            _cash_deposits = flex_select_nb(cash_deposits_, i, group)
            if _cash_deposits < 0:
                _cash_deposits = max(_cash_deposits, -cash_now)
            cash_now += _cash_deposits
            free_cash_now += _cash_deposits
            if track_cash_deposits:
                cash_deposits_out[i, group] += _cash_deposits

            for c in range(group_len):
                col = from_col + c

                # Update valuation price using current open
                _open = flex_select_nb(open_, i, col)
                if not np.isnan(_open) or not ffill_val_price:
                    last_val_price[col] = _open

                # Resolve valuation price
                _val_price = flex_select_nb(val_price_, i, col)
                if np.isinf(_val_price):
                    if _val_price > 0:
                        _i = i - abs(flex_select_nb(from_ago_, i, col))
                        if _i < 0:
                            _price = np.nan
                        else:
                            _price = flex_select_nb(price_, _i, col)
                        if np.isinf(_price):
                            if _price > 0:
                                _price = flex_select_nb(close_, i, col)
                            else:
                                _price = _open
                        _val_price = _price
                    else:
                        _val_price = last_val_price[col]
                if not np.isnan(_val_price) or not ffill_val_price:
                    last_val_price[col] = _val_price

            # Calculate group value and rearrange if cash sharing is enabled
            if cash_sharing:
                # Same as get_ctx_group_value_nb but with flexible indexing
                value_now = cash_now
                for c in range(group_len):
                    col = from_col + c

                    if last_position[col] != 0:
                        value_now += last_position[col] * last_val_price[col]

                # Dynamically sort by order value -> selling comes first to release funds early
                if call_seq is None:
                    for c in range(group_len):
                        temp_call_seq[c] = c
                    call_seq_now = temp_call_seq[:group_len]
                else:
                    call_seq_now = call_seq[i, from_col:to_col]
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for c in range(group_len):
                        col = from_col + c
                        exec_state = ExecState(
                            cash=cash_now,
                            position=last_position[col],
                            debt=last_debt[col],
                            locked_cash=last_locked_cash[col],
                            free_cash=free_cash_now,
                            val_price=last_val_price[col],
                            value=value_now,
                        )
                        _i = i - abs(flex_select_nb(from_ago_, i, col))
                        if _i < 0:
                            temp_order_value[c] = 0.0
                        else:
                            temp_order_value[c] = approx_order_value_nb(
                                exec_state,
                                flex_select_nb(size_, _i, col),
                                flex_select_nb(size_type_, _i, col),
                                flex_select_nb(direction_, _i, col),
                            )
                        if call_seq_now[c] != c:
                            raise ValueError("Call sequence must follow CallSeqType.Default")

                    # Sort by order value
                    insert_argsort_nb(temp_order_value[:group_len], call_seq_now)

            for k in range(group_len):
                if cash_sharing:
                    c = call_seq_now[k]
                    if c >= group_len:
                        raise ValueError("Call index out of bounds of the group")
                else:
                    c = k
                col = from_col + c

                # Get current values per column
                position_now = last_position[col]
                debt_now = last_debt[col]
                locked_cash_now = last_locked_cash[col]
                val_price_now = last_val_price[col]
                if not cash_sharing:
                    value_now = cash_now
                    if position_now != 0:
                        value_now += position_now * val_price_now

                # Generate the next order
                _i = i - abs(flex_select_nb(from_ago_, i, col))
                if _i < 0:
                    continue
                order = order_nb(
                    size=flex_select_nb(size_, _i, col),
                    price=flex_select_nb(price_, _i, col),
                    size_type=flex_select_nb(size_type_, _i, col),
                    direction=flex_select_nb(direction_, _i, col),
                    fees=flex_select_nb(fees_, _i, col),
                    fixed_fees=flex_select_nb(fixed_fees_, _i, col),
                    slippage=flex_select_nb(slippage_, _i, col),
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

                # Update execution state
                cash_now = new_exec_state.cash
                position_now = new_exec_state.position
                debt_now = new_exec_state.debt
                locked_cash_now = new_exec_state.locked_cash
                free_cash_now = new_exec_state.free_cash
                val_price_now = new_exec_state.val_price
                value_now = new_exec_state.value

                # Now becomes last
                last_position[col] = position_now
                last_debt[col] = debt_now
                last_locked_cash[col] = locked_cash_now
                if not np.isnan(val_price_now) or not ffill_val_price:
                    last_val_price[col] = val_price_now

            group_value = cash_now
            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_nb(close_, i, col)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                _cash_dividends = flex_select_nb(cash_dividends_, i, col)
                _cash_earnings += _cash_dividends * last_position[col]
                if _cash_earnings < 0:
                    _cash_earnings = max(_cash_earnings, -cash_now)
                cash_now += _cash_earnings
                free_cash_now += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings
                if save_state:
                    position[i, col] = last_position[col]
                    debt[i, col] = last_debt[col]
                    locked_cash[i, col] = last_locked_cash[col]
                    if not cash_sharing:
                        cash[i, col] = cash_now
                        free_cash[i, col] = free_cash_now

                # Update previous value, current value, and return
                if save_value or save_returns:
                    if cash_sharing:
                        if last_position[col] != 0:
                            group_value += last_position[col] * last_val_price[col]
                    else:
                        if last_position[col] == 0:
                            last_value[col] = cash_now
                        else:
                            last_value[col] = cash_now + last_position[col] * last_val_price[col]
                        last_return[col] = get_return_nb(
                            prev_close_value[col],
                            last_value[col] - _cash_deposits,
                        )
                        prev_close_value[col] = last_value[col]
                        if save_value:
                            in_outputs.value[i, group] = last_value[col]
                        if save_returns:
                            in_outputs.returns[i, group] = last_return[col]

            # Fill group state and returns
            if cash_sharing:
                if save_state:
                    cash[i, group] = cash_now
                    free_cash[i, group] = free_cash_now
                if save_value or save_returns:
                    last_value[group] = group_value
                    last_return[group] = get_return_nb(
                        prev_close_value[group],
                        last_value[group] - _cash_deposits,
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
