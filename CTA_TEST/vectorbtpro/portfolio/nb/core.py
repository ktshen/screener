# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Core Numba-compiled functions for portfolio simulation."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.math_ import is_close_nb, is_close_or_less_nb, is_less_nb, add_nb


@register_jitted(cache=True)
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """Return `OrderResult` for order that hasn't been filled."""
    return OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=status, status_info=status_info)


@register_jitted(cache=True)
def check_adj_price_nb(
    adj_price: float,
    price_area: PriceArea,
    is_closing_price: bool,
    price_area_vio_mode: int,
) -> float:
    """Check whether adjusted price is within price boundaries."""
    if price_area_vio_mode == PriceAreaVioMode.Ignore:
        return adj_price
    if adj_price > price_area.high:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is above the highest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.high
    if adj_price < price_area.low:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is below the lowest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.low
    if is_closing_price and adj_price != price_area.close:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is beyond the closing price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.close
    return adj_price


@register_jitted(cache=True)
def approx_long_buy_value_nb(val_price: float, size: float) -> float:
    """Approximate value of a long-buy operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def adj_size_granularity_nb(size: float, size_granularity: float) -> bool:
    """Whether to adjust the size with the size granularity."""
    adj_size = size // size_granularity * size_granularity
    return not is_close_nb(size, adj_size) and not is_close_nb(size, adj_size + size_granularity)


@register_jitted(cache=True)
def long_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a long position."""
    # Get cash limit
    cash_limit = account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), account_state
    cash_limit = cash_limit * leverage

    # Adjust for max size
    if not np.isnan(max_size) and size > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), account_state

        size = max_size
    if np.isinf(size) and np.isinf(cash_limit):
        raise ValueError("Attempt to go in long direction infinitely")

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size, size_granularity):
        size = size // size_granularity * size_granularity

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if not np.isnan(size_granularity) and adj_size_granularity_nb(max_acq_size, size_granularity):
            final_size = max_acq_size // size_granularity * size_granularity
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check against size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), account_state

    # Create a filled order
    order_result = OrderResult(
        final_size,
        adj_price,
        fees_paid,
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(account_state.cash, -req_cash)
    new_position = add_nb(account_state.position, final_size)
    if leverage_mode == LeverageMode.Lazy:
        debt_diff = max(add_nb(req_cash, -account_state.free_cash), 0.0)
        if debt_diff > 0:
            new_debt = account_state.debt + debt_diff
            new_locked_cash = account_state.locked_cash + account_state.free_cash
            new_free_cash = 0.0
        else:
            new_debt = account_state.debt
            new_locked_cash = account_state.locked_cash
            new_free_cash = add_nb(account_state.free_cash, -req_cash)
    else:
        if leverage > 1:
            if np.isinf(leverage):
                raise ValueError("Leverage must be finite for LeverageMode.Eager")
            order_value = final_size * adj_price
            new_debt = account_state.debt + order_value * (leverage - 1) / leverage
            new_locked_cash = account_state.locked_cash + order_value / leverage
            new_free_cash = add_nb(account_state.free_cash, -order_value / leverage - fees_paid)
        else:
            new_debt = account_state.debt
            new_locked_cash = account_state.locked_cash
            new_free_cash = add_nb(account_state.free_cash, -req_cash)
    new_account_state = AccountState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        locked_cash=new_locked_cash,
        free_cash=new_free_cash,
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_long_sell_value_nb(position: float, debt: float, val_price: float, size: float) -> float:
    """Approximate value of a long-sell operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(position, abs(size))
    order_value = size_limit * val_price
    size_fraction = size_limit / position
    released_debt = size_fraction * debt
    add_free_cash = order_value - released_debt
    return -add_free_cash


@register_jitted(cache=True)
def long_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a long position."""
    # Check for open position
    if account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), account_state

    # Get size limit
    size_limit = min(account_state.position, size)
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), account_state

        size_limit = max_size

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), account_state

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get acquired cash
    acq_cash = size_limit * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(acq_cash, -fees_paid)
    if final_acq_cash < 0 and is_less_nb(account_state.free_cash, -final_acq_cash):
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state

    # Create a filled order
    order_result = OrderResult(
        size_limit,
        adj_price,
        fees_paid,
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = account_state.cash + final_acq_cash
    new_position = add_nb(account_state.position, -size_limit)
    new_pos_fraction = abs(new_position) / abs(account_state.position)
    new_debt = new_pos_fraction * account_state.debt
    new_locked_cash = new_pos_fraction * account_state.locked_cash
    size_fraction = size_limit / account_state.position
    released_debt = size_fraction * account_state.debt
    new_free_cash = add_nb(account_state.free_cash, final_acq_cash - released_debt)
    new_account_state = AccountState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        locked_cash=new_locked_cash,
        free_cash=new_free_cash,
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_sell_value_nb(val_price: float, size: float) -> float:
    """Approximate value of a short-sell operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a short position."""
    # Get cash limit
    cash_limit = account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), account_state
    cash_limit = cash_limit * leverage

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get size limit
    fees_adj_price = adj_price * (1 + fees)
    if fees_adj_price == 0:
        max_size_limit = np.inf
    else:
        max_size_limit = add_nb(cash_limit, -fixed_fees) / (adj_price * (1 + fees))
    size_limit = min(size, max_size_limit)
    if size_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), account_state

        size_limit = max_size
    if np.isinf(size_limit):
        raise ValueError("Attempt to go in short direction infinitely")

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), account_state

    # Get acquired cash
    order_value = size_limit * adj_price

    # Update fees
    fees_paid = order_value * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(order_value, -fees_paid)
    if final_acq_cash < 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state

    # Create a filled order
    order_result = OrderResult(
        size_limit,
        adj_price,
        fees_paid,
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = account_state.cash + final_acq_cash
    new_position = account_state.position - size_limit
    new_debt = account_state.debt + order_value
    if np.isinf(leverage):
        if np.isinf(account_state.free_cash):
            raise ValueError("Leverage must be finite when account_state.free_cash is infinite")
        if is_close_or_less_nb(account_state.free_cash, fees_paid):
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state
        leverage_ = order_value / (account_state.free_cash - fees_paid)
    else:
        leverage_ = float(leverage)
    new_locked_cash = account_state.locked_cash + order_value / leverage_
    new_free_cash = add_nb(account_state.free_cash, -order_value / leverage_ - fees_paid)
    new_account_state = AccountState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        locked_cash=new_locked_cash,
        free_cash=new_free_cash,
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_buy_value_nb(position: float, debt: float, locked_cash: float, val_price: float, size: float) -> float:
    """Approximate value of a short-buy operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(abs(position), abs(size))
    order_value = size_limit * val_price
    size_fraction = size_limit / abs(position)
    released_debt = size_fraction * debt
    released_cash = size_fraction * locked_cash
    add_free_cash = released_cash + released_debt - order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a short position."""
    # Check for open position
    if account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), account_state

    # Get cash limit
    cash_limit = account_state.free_cash + account_state.debt + account_state.locked_cash
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), account_state

    # Get size limit
    size_limit = min(abs(account_state.position), size)
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), account_state

        size_limit = max_size

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size_limit):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size_limit * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size_limit
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if not np.isnan(size_granularity) and adj_size_granularity_nb(max_acq_size, size_granularity):
            final_size = max_acq_size // size_granularity * size_granularity
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size_limit) and is_less_nb(final_size, size_limit) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), account_state

    # Create a filled order
    order_result = OrderResult(
        final_size,
        adj_price,
        fees_paid,
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(account_state.cash, -req_cash)
    new_position = add_nb(account_state.position, final_size)
    new_pos_fraction = abs(new_position) / abs(account_state.position)
    new_debt = new_pos_fraction * account_state.debt
    new_locked_cash = new_pos_fraction * account_state.locked_cash
    size_fraction = final_size / abs(account_state.position)
    released_debt = size_fraction * account_state.debt
    released_cash = size_fraction * account_state.locked_cash
    new_free_cash = add_nb(account_state.free_cash, released_cash + released_debt - req_cash)
    new_account_state = AccountState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        locked_cash=new_locked_cash,
        free_cash=new_free_cash,
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_buy_value_nb(
    position: float,
    debt: float,
    locked_cash: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate value of a buy operation.

    Positive value means spending (for sorting reasons)."""
    if position <= 0 and direction == Direction.ShortOnly:
        return approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    if position >= 0:
        return approx_long_buy_value_nb(val_price, size)
    value1 = approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_long_buy_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Buy."""
    if account_state.position <= 0 and direction == Direction.ShortOnly:
        return short_buy_nb(
            account_state=account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if account_state.position >= 0:
        return long_buy_nb(
            account_state=account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            leverage_mode=leverage_mode,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if not np.isnan(min_size):
        min_size1 = min(min_size, abs(account_state.position))
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, abs(account_state.position))
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = short_buy_nb(
        account_state=account_state,
        size=size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - abs(account_state.position), 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - abs(account_state.position), 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = long_buy_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def approx_sell_value_nb(
    position: float,
    debt: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate value of a sell operation.

    Positive value means spending (for sorting reasons)."""
    if position >= 0 and direction == Direction.LongOnly:
        return approx_long_sell_value_nb(position, debt, val_price, size)
    if position <= 0:
        return approx_short_sell_value_nb(val_price, size)
    value1 = approx_long_sell_value_nb(position, debt, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_short_sell_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Sell."""
    if account_state.position >= 0 and direction == Direction.LongOnly:
        return long_sell_nb(
            account_state=account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if account_state.position <= 0:
        return short_sell_nb(
            account_state=account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if not np.isnan(min_size):
        min_size1 = min(min_size, account_state.position)
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, account_state.position)
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = long_sell_nb(
        account_state=account_state,
        size=size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - account_state.position, 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - account_state.position, 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = short_sell_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def update_value_nb(
    cash_before: float,
    cash_now: float,
    position_before: float,
    position_now: float,
    val_price_before: float,
    price: float,
    value_before: float,
) -> tp.Tuple[float, float]:
    """Update valuation price and value."""
    val_price_now = price
    cash_flow = cash_now - cash_before
    if position_before != 0:
        asset_value_before = position_before * val_price_before
    else:
        asset_value_before = 0.0
    if position_now != 0:
        asset_value_now = position_now * val_price_now
    else:
        asset_value_now = 0.0
    asset_value_diff = asset_value_now - asset_value_before
    value_now = value_before + cash_flow + asset_value_diff
    return val_price_now, value_now


@register_jitted(cache=True)
def get_diraware_size_nb(size: float, direction: int) -> float:
    """Get direction-aware size."""
    if direction == Direction.ShortOnly:
        return size * -1
    return size


@register_jitted(cache=True)
def get_mn_val_price_nb(position: float, debt: float, val_price: float) -> float:
    """Get market-neutral asset valuation price."""
    if position < 0:
        avg_entry_price = debt / abs(position)
        return 2 * avg_entry_price - val_price
    return val_price


@register_jitted(cache=True)
def resolve_size_nb(
    size: float,
    size_type: int,
    position: float,
    debt: float,
    val_price: float,
    value: float,
    as_requirement: bool = False,
) -> tp.Tuple[float, float]:
    """Resolve size into an absolute amount of assets and percentage of resources.

    Percentage is only set if the option `SizeType.Percent(100)` is used."""
    market_neutral = False
    if size_type == SizeType.MNTargetPercent100:
        market_neutral = True
        size_type = SizeType.TargetPercent100
    if size_type == SizeType.MNTargetPercent:
        market_neutral = True
        size_type = SizeType.TargetPercent
    if size_type == SizeType.MNTargetValue:
        market_neutral = True
        size_type = SizeType.TargetValue
    if market_neutral:
        val_price = get_mn_val_price_nb(
            position=position,
            debt=debt,
            val_price=val_price,
        )

    if size_type == SizeType.ValuePercent100:
        size /= 100
        size_type = SizeType.ValuePercent
    if size_type == SizeType.TargetPercent100:
        size /= 100
        size_type = SizeType.TargetPercent
    if size_type == SizeType.ValuePercent or size_type == SizeType.TargetPercent:
        # Percentage or target percentage of the current value
        size *= value
        if size_type == SizeType.ValuePercent:
            size_type = SizeType.Value
        else:
            size_type = SizeType.TargetValue
    if size_type == SizeType.Value or size_type == SizeType.TargetValue:
        # Value or target value
        size /= val_price
        if size_type == SizeType.Value:
            size_type = SizeType.Amount
        else:
            size_type = SizeType.TargetAmount
    if size_type == SizeType.TargetAmount:
        # Target amount
        if not as_requirement:
            size -= position
        size_type = SizeType.Amount

    percent = np.nan
    if size_type == SizeType.Percent100:
        size /= 100
        size_type = SizeType.Percent
    if size_type == SizeType.Percent:
        # Percentage of resources
        percent = abs(size)
        size = np.sign(size) * np.inf

    if as_requirement:
        size = abs(size)
    return size, percent


@register_jitted(cache=True)
def approx_order_value_nb(
    exec_state: ExecState,
    size: float,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
) -> float:
    """Approximate the value of an order.

    Assumes that cash is infinite.

    Positive value means spending (for sorting reasons)."""
    size = get_diraware_size_nb(float(size), direction)
    amount_size, _ = resolve_size_nb(
        size=size,
        size_type=size_type,
        position=exec_state.position,
        debt=exec_state.debt,
        val_price=exec_state.val_price,
        value=exec_state.value,
    )
    if amount_size >= 0:
        order_value = approx_buy_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            locked_cash=exec_state.locked_cash,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    else:
        order_value = approx_sell_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    return order_value


@register_jitted(cache=True)
def execute_order_nb(
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
) -> tp.Tuple[OrderResult, ExecState]:
    """Execute an order given the current state.

    Args:
        exec_state (ExecState): See `vectorbtpro.portfolio.enums.ExecState`.
        order (Order): See `vectorbtpro.portfolio.enums.Order`.
        price_area (OrderPriceArea): See `vectorbtpro.portfolio.enums.PriceArea`.
        update_value (bool): Whether to update the value.

    Error is thrown if an input has value that is not expected.
    Order is ignored if its execution has no effect on the current balance.
    Order is rejected if an input goes over a limit or against a restriction.
    """
    # numerical stability
    cash = float(exec_state.cash)
    if is_close_nb(cash, 0):
        cash = 0.0
    position = float(exec_state.position)
    if is_close_nb(position, 0):
        position = 0.0
    debt = float(exec_state.debt)
    if is_close_nb(debt, 0):
        debt = 0.0
    locked_cash = float(exec_state.locked_cash)
    if is_close_nb(locked_cash, 0):
        locked_cash = 0.0
    free_cash = float(exec_state.free_cash)
    if is_close_nb(free_cash, 0):
        free_cash = 0.0
    val_price = float(exec_state.val_price)
    if is_close_nb(val_price, 0):
        val_price = 0.0
    value = float(exec_state.value)
    if is_close_nb(value, 0):
        value = 0.0

    # Pre-fill account state
    account_state = AccountState(
        cash=cash,
        position=position,
        debt=debt,
        locked_cash=locked_cash,
        free_cash=free_cash,
    )

    # Check price area
    if np.isinf(price_area.open) or price_area.open < 0:
        raise ValueError("price_area.open must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.high) or price_area.high < 0:
        raise ValueError("price_area.high must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.low) or price_area.low < 0:
        raise ValueError("price_area.low must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.close) or price_area.close < 0:
        raise ValueError("price_area.close must be either NaN, or finite and 0 or greater")

    # Resolve price
    order_price = order.price
    is_closing_price = False
    if np.isinf(order_price):
        if order_price > 0:
            order_price = price_area.close
            is_closing_price = True
        else:
            order_price = price_area.open
    elif order_price == PriceType.NextOpen:
        raise ValueError("Next open must be handled higher in the stack")
    elif order_price == PriceType.NextClose:
        raise ValueError("Next close must be handled higher in the stack")

    # Ignore order if size or price is nan
    if np.isnan(order.size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeNaN), exec_state
    if np.isnan(order_price):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.PriceNaN), exec_state

    # Check account state
    if np.isnan(cash):
        raise ValueError("exec_state.cash cannot be NaN")
    if not np.isfinite(position):
        raise ValueError("exec_state.position must be finite")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("exec_state.debt must be finite and 0 or greater")
    if not np.isfinite(locked_cash) or locked_cash < 0:
        raise ValueError("exec_state.locked_cash must be finite and 0 or greater")
    if np.isnan(free_cash):
        raise ValueError("exec_state.free_cash cannot be NaN")

    # Check order
    if not np.isfinite(order_price) or order_price < 0:
        raise ValueError("order.price must be finite and 0 or greater")
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("order.size_type is invalid")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("order.direction is invalid")
    if not np.isfinite(order.fees):
        raise ValueError("order.fees must be finite")
    if not np.isfinite(order.fixed_fees):
        raise ValueError("order.fixed_fees must be finite")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("order.slippage must be finite and 0 or greater")
    if np.isinf(order.min_size) or order.min_size < 0:
        raise ValueError("order.min_size must be either NaN, 0, or greater")
    if order.max_size <= 0:
        raise ValueError("order.max_size must be either NaN or greater than 0")
    if np.isinf(order.size_granularity) or order.size_granularity <= 0:
        raise ValueError("order.size_granularity must be either NaN, or finite and greater than 0")
    if np.isnan(order.leverage) or order.leverage <= 0:
        raise ValueError("order.leverage must be greater than 0")
    if order.leverage_mode < 0 or order.leverage_mode >= len(LeverageMode):
        raise ValueError("order.leverage_mode is invalid")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("order.reject_prob must be between 0 and 1")

    # Positive/negative size in short direction should be treated as negative/positive
    order_size = get_diraware_size_nb(order.size, order.direction)
    min_order_size = order.min_size
    max_order_size = order.max_size
    order_size_type = order.size_type

    if (
        order_size_type == SizeType.ValuePercent100
        or order_size_type == SizeType.ValuePercent
        or order_size_type == SizeType.TargetPercent100
        or order_size_type == SizeType.TargetPercent
        or order_size_type == SizeType.Value
        or order_size_type == SizeType.TargetValue
    ):
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValPriceNaN), exec_state
        if (
            order_size_type == SizeType.ValuePercent100
            or order_size_type == SizeType.ValuePercent
            or order_size_type == SizeType.TargetPercent100
            or order_size_type == SizeType.TargetPercent
        ):
            if np.isnan(value):
                return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValueNaN), exec_state
            if value <= 0:
                return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.ValueZeroNeg), exec_state

    order_size, percent = resolve_size_nb(
        size=order_size,
        size_type=order_size_type,
        position=position,
        debt=debt,
        val_price=val_price,
        value=value,
    )
    if not np.isnan(min_order_size):
        min_order_size, min_percent = resolve_size_nb(
            size=min_order_size,
            size_type=order_size_type,
            position=position,
            debt=debt,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(min_percent) and is_less_nb(percent, min_percent):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), exec_state
    if not np.isnan(max_order_size):
        max_order_size, max_percent = resolve_size_nb(
            size=max_order_size,
            size_type=order_size_type,
            position=position,
            debt=debt,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(max_percent) and is_less_nb(max_percent, percent):
            percent = max_percent

    if order_size >= 0:
        order_result, new_account_state = buy_nb(
            account_state=account_state,
            size=order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            leverage_mode=order.leverage_mode,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    else:
        order_result, new_account_state = sell_nb(
            account_state=account_state,
            size=-order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.RandomEvent), exec_state

    if order_result.status == OrderStatus.Rejected and order.raise_reject:
        raise_rejected_order_nb(order_result)

    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        new_val_price, new_value = update_value_nb(
            cash,
            new_account_state.cash,
            position,
            new_account_state.position,
            val_price,
            order_result.price,
            value,
        )
    else:
        new_val_price = val_price
        new_value = value

    new_exec_state = ExecState(
        cash=new_account_state.cash,
        position=new_account_state.position,
        debt=new_account_state.debt,
        locked_cash=new_account_state.locked_cash,
        free_cash=new_account_state.free_cash,
        val_price=new_val_price,
        value=new_value,
    )

    return order_result, new_exec_state


@register_jitted(cache=True)
def fill_log_record_nb(
    records: tp.RecordArray2d,
    r: int,
    group: int,
    col: int,
    i: int,
    price_area: PriceArea,
    exec_state: ExecState,
    order: Order,
    order_result: OrderResult,
    new_exec_state: ExecState,
    order_id: int,
) -> None:
    """Fill a log record."""

    records["id"][r, col] = r
    records["group"][r, col] = group
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["price_area_open"][r, col] = price_area.open
    records["price_area_high"][r, col] = price_area.high
    records["price_area_low"][r, col] = price_area.low
    records["price_area_close"][r, col] = price_area.close
    records["st0_cash"][r, col] = exec_state.cash
    records["st0_position"][r, col] = exec_state.position
    records["st0_debt"][r, col] = exec_state.debt
    records["st0_locked_cash"][r, col] = exec_state.locked_cash
    records["st0_free_cash"][r, col] = exec_state.free_cash
    records["st0_val_price"][r, col] = exec_state.val_price
    records["st0_value"][r, col] = exec_state.value
    records["req_size"][r, col] = order.size
    records["req_price"][r, col] = order.price
    records["req_size_type"][r, col] = order.size_type
    records["req_direction"][r, col] = order.direction
    records["req_fees"][r, col] = order.fees
    records["req_fixed_fees"][r, col] = order.fixed_fees
    records["req_slippage"][r, col] = order.slippage
    records["req_min_size"][r, col] = order.min_size
    records["req_max_size"][r, col] = order.max_size
    records["req_size_granularity"][r, col] = order.size_granularity
    records["req_leverage"][r, col] = order.leverage
    records["req_leverage_mode"][r, col] = order.leverage_mode
    records["req_reject_prob"][r, col] = order.reject_prob
    records["req_price_area_vio_mode"][r, col] = order.price_area_vio_mode
    records["req_allow_partial"][r, col] = order.allow_partial
    records["req_raise_reject"][r, col] = order.raise_reject
    records["req_log"][r, col] = order.log
    records["res_size"][r, col] = order_result.size
    records["res_price"][r, col] = order_result.price
    records["res_fees"][r, col] = order_result.fees
    records["res_side"][r, col] = order_result.side
    records["res_status"][r, col] = order_result.status
    records["res_status_info"][r, col] = order_result.status_info
    records["st1_cash"][r, col] = new_exec_state.cash
    records["st1_position"][r, col] = new_exec_state.position
    records["st1_debt"][r, col] = new_exec_state.debt
    records["st1_locked_cash"][r, col] = new_exec_state.locked_cash
    records["st1_free_cash"][r, col] = new_exec_state.free_cash
    records["st1_val_price"][r, col] = new_exec_state.val_price
    records["st1_value"][r, col] = new_exec_state.value
    records["order_id"][r, col] = order_id


@register_jitted(cache=True)
def fill_order_record_nb(records: tp.RecordArray2d, r: int, col: int, i: int, order_result: OrderResult) -> None:
    """Fill an order record."""

    records["id"][r, col] = r
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["size"][r, col] = order_result.size
    records["price"][r, col] = order_result.price
    records["fees"][r, col] = order_result.fees
    records["side"][r, col] = order_result.side


@register_jitted(cache=True)
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """Raise an `vectorbtpro.portfolio.enums.RejectedOrderError`."""

    if order_result.status_info == OrderStatusInfo.SizeNaN:
        raise RejectedOrderError("Size is NaN")
    if order_result.status_info == OrderStatusInfo.PriceNaN:
        raise RejectedOrderError("Price is NaN")
    if order_result.status_info == OrderStatusInfo.ValPriceNaN:
        raise RejectedOrderError("Asset valuation price is NaN")
    if order_result.status_info == OrderStatusInfo.ValueNaN:
        raise RejectedOrderError("Asset/group value is NaN")
    if order_result.status_info == OrderStatusInfo.ValueZeroNeg:
        raise RejectedOrderError("Asset/group value is zero or negative")
    if order_result.status_info == OrderStatusInfo.SizeZero:
        raise RejectedOrderError("Size is zero")
    if order_result.status_info == OrderStatusInfo.NoCash:
        raise RejectedOrderError("Not enough cash")
    if order_result.status_info == OrderStatusInfo.NoOpenPosition:
        raise RejectedOrderError("No open position to reduce/close")
    if order_result.status_info == OrderStatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("Size is greater than maximum allowed")
    if order_result.status_info == OrderStatusInfo.RandomEvent:
        raise RejectedOrderError("Random event happened")
    if order_result.status_info == OrderStatusInfo.CantCoverFees:
        raise RejectedOrderError("Not enough cash to cover fees")
    if order_result.status_info == OrderStatusInfo.MinSizeNotReached:
        raise RejectedOrderError("Final size is less than minimum allowed")
    if order_result.status_info == OrderStatusInfo.PartialFill:
        raise RejectedOrderError("Final size is less than requested")
    raise RejectedOrderError


@register_jitted(cache=True)
def process_order_nb(
    group: int,
    col: int,
    i: int,
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
    order_records: tp.Optional[tp.RecordArray2d] = None,
    order_counts: tp.Optional[tp.Array1d] = None,
    log_records: tp.Optional[tp.RecordArray2d] = None,
    log_counts: tp.Optional[tp.Array1d] = None,
) -> tp.Tuple[OrderResult, ExecState]:
    """Process an order by executing it, saving relevant information to the logs, and returning a new state."""
    # Execute the order
    order_result, new_exec_state = execute_order_nb(
        exec_state=exec_state,
        order=order,
        price_area=price_area,
        update_value=update_value,
    )

    is_filled = order_result.status == OrderStatus.Filled
    if order_records is not None and order_counts is not None:
        if is_filled and order_records.shape[0] > 0:
            # Fill order record
            if order_counts[col] >= order_records.shape[0]:
                raise IndexError("order_records index out of range. Set a higher max_orders.")
            fill_order_record_nb(order_records, order_counts[col], col, i, order_result)
            order_counts[col] += 1

    if log_records is not None and log_counts is not None:
        if order.log and log_records.shape[0] > 0:
            # Fill log record
            if log_counts[col] >= log_records.shape[0]:
                raise IndexError("log_records index out of range. Set a higher max_logs.")
            fill_log_record_nb(
                log_records,
                log_counts[col],
                group,
                col,
                i,
                price_area,
                exec_state,
                order,
                order_result,
                new_exec_state,
                order_counts[col] - 1 if order_counts is not None and is_filled else -1,
            )
            log_counts[col] += 1

    return order_result, new_exec_state


@register_jitted(cache=True)
def order_nb(
    size: float = np.inf,
    price: float = np.inf,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Create an order.

    See `vectorbtpro.portfolio.enums.Order` for details on arguments."""

    return Order(
        size=float(size),
        price=float(price),
        size_type=int(size_type),
        direction=int(direction),
        fees=float(fees),
        fixed_fees=float(fixed_fees),
        slippage=float(slippage),
        min_size=float(min_size),
        max_size=float(max_size),
        size_granularity=float(size_granularity),
        leverage=float(leverage),
        leverage_mode=int(leverage_mode),
        reject_prob=float(reject_prob),
        price_area_vio_mode=int(price_area_vio_mode),
        allow_partial=bool(allow_partial),
        raise_reject=bool(raise_reject),
        log=bool(log),
    )


@register_jitted(cache=True)
def close_position_nb(
    price: float = np.inf,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Close the current position."""

    return order_nb(
        size=0.0,
        price=price,
        size_type=SizeType.TargetAmount,
        direction=Direction.Both,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size,
        max_size=max_size,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        reject_prob=reject_prob,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        raise_reject=raise_reject,
        log=log,
    )


@register_jitted(cache=True)
def order_nothing_nb() -> Order:
    """Convenience function to order nothing."""
    return NoOrder


@register_jitted(cache=True)
def check_group_lens_nb(group_lens: tp.Array1d, n_cols: int) -> None:
    """Check `group_lens`."""
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@register_jitted(cache=True)
def is_grouped_nb(group_lens: tp.Array1d) -> bool:
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_lens > 1)


@register_jitted(cache=True)
def get_group_value_nb(
    from_col: int,
    to_col: int,
    cash_now: float,
    last_position: tp.Array1d,
    last_val_price: tp.Array1d,
) -> float:
    """Get group value."""
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_position[col] != 0:
            group_value += last_position[col] * last_val_price[col]
    return group_value


@register_jitted(cache=True)
def prepare_records_nb(
    target_shape: tp.Shape,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
) -> tp.Tuple[tp.RecordArray2d, tp.RecordArray2d]:
    """Prepare records."""
    if max_orders is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=order_dt)
    else:
        order_records = np.empty((max_orders, target_shape[1]), dtype=order_dt)
    if max_logs is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_logs, target_shape[1]), dtype=log_dt)
    return order_records, log_records


@register_jitted(cache=True)
def prepare_last_cash_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare `last_cash`."""
    if cash_sharing:
        last_cash = np.empty(len(group_lens), dtype=np.float_)
        for group in range(len(group_lens)):
            last_cash[group] = float(flex_select_1d_pc_nb(init_cash, group))
    else:
        last_cash = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            last_cash[col] = float(flex_select_1d_pc_nb(init_cash, col))
    return last_cash


@register_jitted(cache=True)
def prepare_last_position_nb(target_shape: tp.Shape, init_position: tp.FlexArray1d) -> tp.Array1d:
    """Prepare `last_position`."""
    last_position = np.empty(target_shape[1], dtype=np.float_)
    for col in range(target_shape[1]):
        last_position[col] = float(flex_select_1d_pc_nb(init_position, col))
    return last_position


@register_jitted(cache=True)
def prepare_last_value_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare `last_value`."""
    if cash_sharing:
        last_value = np.empty(len(group_lens), dtype=np.float_)
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = float(flex_select_1d_pc_nb(init_cash, group))
            last_value[group] = _init_cash
            for col in range(from_col, to_col):
                _init_position = float(flex_select_1d_pc_nb(init_position, col))
                _init_price = float(flex_select_1d_pc_nb(init_price, col))
                if _init_position != 0:
                    last_value[group] += _init_position * _init_price
            from_col = to_col
    else:
        last_value = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            _init_cash = float(flex_select_1d_pc_nb(init_cash, col))
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position == 0:
                last_value[col] = _init_cash
            else:
                last_value[col] = _init_cash + _init_position * _init_price
    return last_value


@register_jitted(cache=True)
def prepare_last_pos_info_nb(
    target_shape: tp.Shape,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
    fill_pos_info: bool = True,
) -> tp.RecordArray:
    """Prepare `last_pos_info`."""
    if fill_pos_info:
        last_pos_info = np.empty(target_shape[1], dtype=trade_dt)
        last_pos_info["id"][:] = -1
        last_pos_info["col"][:] = -1
        last_pos_info["size"][:] = np.nan
        last_pos_info["entry_order_id"][:] = -1
        last_pos_info["entry_idx"][:] = -1
        last_pos_info["entry_price"][:] = np.nan
        last_pos_info["entry_fees"][:] = np.nan
        last_pos_info["exit_order_id"][:] = -1
        last_pos_info["exit_idx"][:] = -1
        last_pos_info["exit_price"][:] = np.nan
        last_pos_info["exit_fees"][:] = np.nan
        last_pos_info["pnl"][:] = np.nan
        last_pos_info["return"][:] = np.nan
        last_pos_info["direction"][:] = -1
        last_pos_info["status"][:] = -1
        last_pos_info["parent_id"][:] = -1

        for col in range(target_shape[1]):
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position != 0:
                fill_init_pos_info_nb(last_pos_info[col], col, _init_position, _init_price)
    else:
        last_pos_info = np.empty(0, dtype=trade_dt)
    return last_pos_info


@register_jitted
def prepare_simout_nb(
    order_records: tp.RecordArray2d,
    order_counts: tp.Array1d,
    log_records: tp.RecordArray2d,
    log_counts: tp.Array1d,
    cash_deposits: tp.Array2d,
    cash_earnings: tp.Array2d,
    call_seq: tp.Optional[tp.Array2d] = None,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Prepare simulation output."""
    order_records_flat = generic_nb.repartition_nb(order_records, order_counts)
    log_records_flat = generic_nb.repartition_nb(log_records, log_counts)
    return SimulationOutput(
        order_records=order_records_flat,
        log_records=log_records_flat,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


@register_jitted(cache=True)
def get_trade_stats_nb(
    size: float,
    entry_price: float,
    entry_fees: float,
    exit_price: float,
    exit_fees: float,
    direction: int,
) -> tp.Tuple[float, float]:
    """Get trade statistics."""
    entry_val = size * entry_price
    exit_val = size * exit_price
    val_diff = add_nb(exit_val, -entry_val)
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    pnl = val_diff - entry_fees - exit_fees
    if is_close_nb(entry_val, 0):
        ret = np.nan
    else:
        ret = pnl / entry_val
    return pnl, ret


@register_jitted(cache=True)
def update_open_pos_info_stats_nb(record: tp.Record, position_now: float, price: float) -> None:
    """Update statistics of an open position record using custom price."""
    if record["id"] >= 0 and record["status"] == TradeStatus.Open:
        if np.isnan(record["exit_price"]):
            exit_price = price
        else:
            exit_size_sum = record["size"] - abs(position_now)
            exit_gross_sum = exit_size_sum * record["exit_price"]
            exit_gross_sum += abs(position_now) * price
            exit_price = exit_gross_sum / record["size"]
        pnl, ret = get_trade_stats_nb(
            record["size"],
            record["entry_price"],
            record["entry_fees"],
            exit_price,
            record["exit_fees"],
            record["direction"],
        )
        record["pnl"] = pnl
        record["return"] = ret


@register_jitted(cache=True)
def fill_init_pos_info_nb(record: tp.Record, col: int, position_now: float, price: float) -> None:
    """Fill position record for an initial position."""
    record["id"] = 0
    record["col"] = col
    record["size"] = abs(position_now)
    record["entry_order_id"] = -1
    record["entry_idx"] = -1
    record["entry_price"] = price
    record["entry_fees"] = 0.0
    record["exit_order_id"] = -1
    record["exit_idx"] = -1
    record["exit_price"] = np.nan
    record["exit_fees"] = 0.0
    if position_now >= 0:
        record["direction"] = TradeDirection.Long
    else:
        record["direction"] = TradeDirection.Short
    record["status"] = TradeStatus.Open
    record["parent_id"] = record["id"]

    # Update open position stats
    update_open_pos_info_stats_nb(record, position_now, np.nan)


@register_jitted(cache=True)
def update_pos_info_nb(
    record: tp.Record,
    i: int,
    col: int,
    position_before: float,
    position_now: float,
    order_result: OrderResult,
    order_id: int,
) -> None:
    """Update position record after filling an order."""
    if order_result.status == OrderStatus.Filled:
        if position_before == 0 and position_now != 0:
            # New position opened
            record["id"] += 1
            record["col"] = col
            record["size"] = order_result.size
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            record["entry_fees"] = order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        elif position_before != 0 and position_now == 0:
            # Position closed
            record["exit_order_id"] = order_id
            record["exit_idx"] = i
            if np.isnan(record["exit_price"]):
                exit_price = order_result.price
            else:
                exit_size_sum = record["size"] - abs(position_before)
                exit_gross_sum = exit_size_sum * record["exit_price"]
                exit_gross_sum += abs(position_before) * order_result.price
                exit_price = exit_gross_sum / record["size"]
            record["exit_price"] = exit_price
            record["exit_fees"] += order_result.fees
            pnl, ret = get_trade_stats_nb(
                record["size"],
                record["entry_price"],
                record["entry_fees"],
                record["exit_price"],
                record["exit_fees"],
                record["direction"],
            )
            record["pnl"] = pnl
            record["return"] = ret
            record["status"] = TradeStatus.Closed
        elif np.sign(position_before) != np.sign(position_now):
            # Position reversed
            record["id"] += 1
            record["size"] = abs(position_now)
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            new_pos_fraction = abs(position_now) / abs(position_now - position_before)
            record["entry_fees"] = new_pos_fraction * order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        else:
            # Position changed
            if abs(position_before) <= abs(position_now):
                # Position increased
                entry_gross_sum = record["size"] * record["entry_price"]
                entry_gross_sum += order_result.size * order_result.price
                entry_price = entry_gross_sum / (record["size"] + order_result.size)
                record["entry_price"] = entry_price
                record["entry_fees"] += order_result.fees
                record["size"] += order_result.size
            else:
                # Position decreased
                record["exit_order_id"] = order_id
                if np.isnan(record["exit_price"]):
                    exit_price = order_result.price
                else:
                    exit_size_sum = record["size"] - abs(position_before)
                    exit_gross_sum = exit_size_sum * record["exit_price"]
                    exit_gross_sum += order_result.size * order_result.price
                    exit_price = exit_gross_sum / (exit_size_sum + order_result.size)
                record["exit_price"] = exit_price
                record["exit_fees"] += order_result.fees

        # Update open position stats
        update_open_pos_info_stats_nb(record, position_now, order_result.price)


@register_jitted(cache=True)
def resolve_hl_nb(open, high, low, close):
    """Resolve the current high and low."""
    if np.isnan(high):
        if np.isnan(open):
            high = close
        elif np.isnan(close):
            high = open
        else:
            high = max(open, close)
    if np.isnan(low):
        if np.isnan(open):
            low = close
        elif np.isnan(close):
            low = open
        else:
            low = min(open, close)
    return high, low


@register_jitted(cache=True)
def check_price_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
) -> tp.Tuple[float, bool, bool]:
    """Check whether a target price was hit.

    If `can_use_ohlc` and `check_open` is True and the target price is hit before open, returns open.

    Returns the stop price, whether it was hit before open, and whether it was hit during this bar."""
    high, low = resolve_hl_nb(
        open=open,
        high=high,
        low=low,
        close=close,
    )
    if hit_below:
        if can_use_ohlc and check_open and open <= price:
            return open, True, True
        if close <= price or (can_use_ohlc and low <= price):
            return price, False, True
        return price, False, False
    if can_use_ohlc and check_open and open >= price:
        return open, True, True
    if close >= price or (can_use_ohlc and high >= price):
        return price, False, True
    return price, False, False
