# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for portfolio.

Defines enums and other schemas for `vectorbtpro.portfolio`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__pdoc__all__ = __all__ = [
    "RejectedOrderError",
    "PriceType",
    "ValPriceType",
    "InitCashMode",
    "CallSeqType",
    "PendingConflictMode",
    "AccumulationMode",
    "ConflictMode",
    "DirectionConflictMode",
    "OppositeEntryMode",
    "DeltaFormat",
    "TimeDeltaFormat",
    "StopLadderMode",
    "StopEntryPrice",
    "StopExitPrice",
    "StopExitType",
    "StopUpdateMode",
    "SizeType",
    "Direction",
    "LeverageMode",
    "PriceAreaVioMode",
    "OrderStatus",
    "OrderStatusInfo",
    "status_info_desc",
    "OrderSide",
    "OrderType",
    "TradeDirection",
    "TradeStatus",
    "TradesType",
    "OrderPriceStatus",
    "PriceArea",
    "NoPriceArea",
    "AccountState",
    "ExecState",
    "SimulationOutput",
    "SimulationContext",
    "GroupContext",
    "RowContext",
    "SegmentContext",
    "OrderContext",
    "PostOrderContext",
    "FlexOrderContext",
    "Order",
    "NoOrder",
    "OrderResult",
    "SignalSegmentContext",
    "SignalContext",
    "FSInOutputs",
    "FOInOutputs",
    "order_fields",
    "order_dt",
    "fs_order_fields",
    "fs_order_dt",
    "trade_fields",
    "trade_dt",
    "log_fields",
    "log_dt",
    "alloc_range_fields",
    "alloc_range_dt",
    "alloc_point_fields",
    "alloc_point_dt",
    "main_info_fields",
    "main_info_dt",
    "limit_info_fields",
    "limit_info_dt",
    "sl_info_fields",
    "sl_info_dt",
    "tsl_info_fields",
    "tsl_info_dt",
    "tp_info_fields",
    "tp_info_dt",
    "time_info_fields",
    "time_info_dt",
]

__pdoc__ = {}


# ############# Errors ############# #


class RejectedOrderError(Exception):
    """Rejected order error."""

    pass


# ############# Enums ############# #


class PriceTypeT(tp.NamedTuple):
    Open: int = -np.inf
    Close: int = np.inf
    NextOpen: int = -1
    NextClose: int = -2


PriceType = PriceTypeT()
"""_"""

__pdoc__[
    "PriceType"
] = f"""Price type.

```python
{prettify(PriceType)}
```

Attributes:
    Open: Opening price. 
    
        Will be substituted by `-np.inf`.
    Close: Closing price. 
    
        Will be substituted by `np.inf`.
    NextOpen: Next opening price. 
    
        Will be substituted by `-np.inf` and `from_ago` will be set to 1.
    NextClose: Next closing price. 
    
        Will be substituted by `np.inf` and `from_ago` will be set to 1.
"""


class ValPriceTypeT(tp.NamedTuple):
    Latest: int = -np.inf
    Price: int = np.inf


ValPriceType = ValPriceTypeT()
"""_"""

__pdoc__[
    "ValPriceType"
] = f"""Asset valuation price type.

```python
{prettify(ValPriceType)}
```

Attributes:
    Latest: Latest price. Will be substituted by `-np.inf`.
    Price: Order price. Will be substituted by `np.inf`.
"""


class InitCashModeT(tp.NamedTuple):
    Auto: int = -1
    AutoAlign: int = -2


InitCashMode = InitCashModeT()
"""_"""

__pdoc__[
    "InitCashMode"
] = f"""Initial cash mode.

```python
{prettify(InitCashMode)}
```

Attributes:
    Auto: Initial cash is infinite within simulation, and then set to the total cash spent.
    AutoAlign: Initial cash is set to the total cash spent across all columns.
"""


class CallSeqTypeT(tp.NamedTuple):
    Default: int = 0
    Reversed: int = 1
    Random: int = 2
    Auto: int = 3


CallSeqType = CallSeqTypeT()
"""_"""

__pdoc__[
    "CallSeqType"
] = f"""Call sequence type.

```python
{prettify(CallSeqType)}
```

Attributes:
    Default: Place calls from left to right.
    Reversed: Place calls from right to left.
    Random: Place calls randomly.
    Auto: Place calls dynamically based on order value.
"""


class PendingConflictModeT(tp.NamedTuple):
    KeepIgnore: int = 0
    KeepExecute: int = 1
    CancelIgnore: int = 2
    CancelExecute: int = 3


PendingConflictMode = PendingConflictModeT()
"""_"""

__pdoc__[
    "PendingConflictMode"
] = f"""Conflict mode for pending signals.

```python
{prettify(PendingConflictMode)}
```

What should happen if an executable signal occurs during a pending signal?

Attributes:
    KeepIgnore: Keep the pending signal and cancel the user-defined signal.
    KeepExecute: Keep the pending signal and execute the user-defined signal.
    CancelIgnore: Cancel the pending signal and ignore the user-defined signal.
    CancelExecute: Cancel the pending signal and execute the user-defined signal.
"""


class AccumulationModeT(tp.NamedTuple):
    Disabled: int = 0
    Both: int = 1
    AddOnly: int = 2
    RemoveOnly: int = 3


AccumulationMode = AccumulationModeT()
"""_"""

__pdoc__[
    "AccumulationMode"
] = f"""Accumulation mode.

```python
{prettify(AccumulationMode)}
```

Accumulation allows gradually increasing and decreasing positions by a size.

Attributes:
    Disabled: Disable accumulation. 
    
        Can also be provided as False.
    Both: Allow both adding to and removing from the position. 
    
        Can also be provided as True.
    AddOnly: Allow accumulation to only add to the position.
    RemoveOnly: Allow accumulation to only remove from the position.
    
!!! note
    Accumulation acts differently for exits and opposite entries: exits reduce the current position
    but won't enter the opposite one, while opposite entries reduce the position by the same amount,
    but as soon as this position is closed, they begin to increase the opposite position.

    The behavior for opposite entries can be changed by `OppositeEntryMode` and for stop orders by `StopExitType`.
"""


class ConflictModeT(tp.NamedTuple):
    Ignore: int = 0
    Entry: int = 1
    Exit: int = 2
    Adjacent: int = 3
    Opposite: int = 4


ConflictMode = ConflictModeT()
"""_"""

__pdoc__[
    "ConflictMode"
] = f"""Conflict mode.

```python
{prettify(ConflictMode)}
```

What should happen if both an entry signal and an exit signal occur simultaneously?

Attributes:
    Ignore: Ignore both signals.
    Entry: Execute the entry signal.
    Exit: Execute the exit signal.
    Adjacent: Execute the signal adjacent to the current position.
    
        Takes effect only when in position, otherwise ignores.
    Opposite: Execute the signal opposite to the current position.
    
        Takes effect only when in position, otherwise ignores.
"""


class DirectionConflictModeT(tp.NamedTuple):
    Ignore: int = 0
    Long: int = 1
    Short: int = 2
    Adjacent: int = 3
    Opposite: int = 4


DirectionConflictMode = DirectionConflictModeT()
"""_"""

__pdoc__[
    "DirectionConflictMode"
] = f"""Direction conflict mode.

```python
{prettify(DirectionConflictMode)}
```

What should happen if both a long entry signal and a short entry signals occur simultaneously?

Attributes:
    Ignore: Ignore both entry signals.
    Long: Execute the long entry signal.
    Short: Execute the short entry signal.
    Adjacent: Execute the adjacent entry signal. 
    
        Takes effect only when in position, otherwise ignores.
    Opposite: Execute the opposite entry signal. 
    
        Takes effect only when in position, otherwise ignores.
"""


class OppositeEntryModeT(tp.NamedTuple):
    Ignore: int = 0
    Close: int = 1
    CloseReduce: int = 2
    Reverse: int = 3
    ReverseReduce: int = 4


OppositeEntryMode = OppositeEntryModeT()
"""_"""

__pdoc__[
    "OppositeEntryMode"
] = f"""Opposite entry mode.

```python
{prettify(OppositeEntryMode)}
```

What should happen if an entry signal of opposite direction occurs before an exit signal?

Attributes:
    Ignore: Ignore the opposite entry signal.
    Close: Close the current position.
    CloseReduce: Close the current position or reduce it if accumulation is enabled.
    Reverse: Reverse the current position.
    ReverseReduce: Reverse the current position or reduce it if accumulation is enabled.
"""


class DeltaFormatT(tp.NamedTuple):
    Absolute: int = 0
    Percent: int = 1
    Percent100: int = 2
    Target: int = 3


DeltaFormat = DeltaFormatT()
"""_"""

__pdoc__[
    "DeltaFormat"
] = f"""Delta format.

```python
{prettify(DeltaFormat)}
```

In which format a delta is provided?

Attributes:
    Absolute: Absolute format
    Percent: Percentage format where 0.01 means 1%
    Percent100: Percentage format where 1.0 means 1%
    Target: Target format
"""


class TimeDeltaFormatT(tp.NamedTuple):
    Rows: int = 0
    Index: int = 1


TimeDeltaFormat = TimeDeltaFormatT()
"""_"""

__pdoc__[
    "TimeDeltaFormat"
] = f"""Time delta format.

```python
{prettify(TimeDeltaFormat)}
```

In which format a time delta is provided?

Attributes:
    Rows: Row format where 1 means one row (simulation step) has passed. 
    
        Doesn't require the index to be provided.
    Index: Index format where 1 means one value in index has passed. 
    
        If index is datetime-like, 1 means one nanosecond.
    
        Requires the index to be provided.
"""


class StopLadderModeT(tp.NamedTuple):
    Disabled: int = 0
    Uniform: int = 1
    Weighted: int = 2
    AdaptUniform: int = 3
    AdaptWeighted: int = 4
    Dynamic: int = 5


StopLadderMode = StopLadderModeT()
"""_"""

__pdoc__[
    "StopLadderMode"
] = f"""Stop ladder mode.

```python
{prettify(StopLadderMode)}
```

Attributes:
    Disabled: Disable the stop ladder.
    
        Can also be provided as False.
    Uniform: Enable the stop ladder with a uniform exit size.
    
        Can also be provided as True.
    Weighted: Enable the stop ladder with a stop-weighted exit size.
    AdaptUniform: Enable the stop ladder with a uniform exit size that adapts to the current position.
    AdaptWeighted: Enable the stop ladder with a stop-weighted exit size that adapts to the current position.
    Dynamic: Enable the stop ladder but do not use stop arrays.
    
!!! note
    When disabled, make sure that stop arrays broadcast against the target shape.
    
    When enabled, make sure that rows in stop arrays represent steps in the ladder.
"""


class StopEntryPriceT(tp.NamedTuple):
    ValPrice: int = -1
    Open: int = -2
    Price: int = -3
    FillPrice: int = -4
    Close: int = -5


StopEntryPrice = StopEntryPriceT()
"""_"""

__pdoc__[
    "StopEntryPrice"
] = f"""Stop entry price.

```python
{prettify(StopEntryPrice)}
```

Which price to use as an initial stop price?

Attributes:
    ValPrice: Asset valuation price.
    Open: Opening price.
    Price: Order price.
    FillPrice: Filled order price (that is, slippage is already applied).
    Close: Closing price.
    
!!! note
    Each flag is negative, thus if a positive value is provided, it's used directly as price.
"""


class StopExitPriceT(tp.NamedTuple):
    Stop: int = -1
    Close: int = -2


StopExitPrice = StopExitPriceT()
"""_"""

__pdoc__[
    "StopExitPrice"
] = f"""Stop exit price.

```python
{prettify(StopExitPrice)}
```

Which price to use when exiting a position upon a stop signal?

Attributes:
    Stop: Stop price. 
    
        If the stop was hit before, the opening price at the next bar is used.
    Close: Closing price.
    
!!! note
    Each flag is negative, thus if a positive value is provided, it's used directly as price.
"""


class StopExitTypeT(tp.NamedTuple):
    Close: int = 0
    CloseReduce: int = 1
    Reverse: int = 2
    ReverseReduce: int = 3


StopExitType = StopExitTypeT()
"""_"""

__pdoc__[
    "StopExitType"
] = f"""Stop exit type.

```python
{prettify(StopExitType)}
```

How to exit the current position upon a stop signal?

Attributes:
    Close: Close the current position.
    CloseReduce: Close the current position or reduce it if accumulation is enabled.
    Reverse: Reverse the current position.
    ReverseReduce: Reverse the current position or reduce it if accumulation is enabled.
"""


class StopUpdateModeT(tp.NamedTuple):
    Keep: int = 0
    Override: int = 1
    OverrideNaN: int = 2


StopUpdateMode = StopUpdateModeT()
"""_"""

__pdoc__[
    "StopUpdateMode"
] = f"""Stop update mode.

```python
{prettify(StopUpdateMode)}
```

What to do with the old stop upon a new entry/accumulation? 

Attributes:
    Keep: Keep the old stop.
    Override: Override the old stop, but only if the new stop is not NaN.
    OverrideNaN: Override the old stop, even if the new stop is NaN.
"""


class SizeTypeT(tp.NamedTuple):
    Amount: int = 0
    Value: int = 1
    Percent: int = 2
    Percent100: int = 3
    ValuePercent: int = 4
    ValuePercent100: int = 5
    TargetAmount: int = 6
    TargetValue: int = 7
    TargetPercent: int = 8
    TargetPercent100: int = 9
    MNTargetValue: int = 10
    MNTargetPercent: int = 11
    MNTargetPercent100: int = 12


SizeType = SizeTypeT()
"""_"""

__pdoc__[
    "SizeType"
] = f"""Size type.

```python
{prettify(SizeType)}
```

Attributes:
    Amount: Amount of assets to trade.
    Value: Asset value to trade.
    
        Gets converted into `SizeType.Amount` using `ExecState.val_price`.
    Percent: Percentage of available resources to use in either direction (not to be confused with 
        the percentage of position value!) where 0.01 means 1%
    
        * When long buying, the percentage of `ExecState.free_cash`
        * When long selling, the percentage of `ExecState.position`
        * When short selling, the percentage of `ExecState.free_cash`
        * When short buying, the percentage of `ExecState.free_cash`, `ExecState.debt`, and `ExecState.locked_cash`
        * When reversing, the percentage is getting applied on the final position
    Percent100: `SizeType.Percent` where 1.0 means 1%.
    ValuePercent: Percentage of total value.
    
        Uses `ExecState.value` to get the current total value.
        Gets converted into `SizeType.Value`.
    ValuePercent100: `SizeType.ValuePercent` where 1.0 means 1%.
    TargetAmount: Target amount of assets to hold (= target position).
    
        Uses `ExecState.position` to get the current position.
        Gets converted into `SizeType.Amount`.
    TargetValue: Target asset value. 

        Uses `ExecState.val_price` to get the current asset value. 
        Gets converted into `SizeType.TargetAmount`.
    TargetPercent: Target percentage of total value. 

        Uses `ExecState.value_now` to get the current total value.
        Gets converted into `SizeType.TargetValue`.
    TargetPercent100: `SizeType.TargetPercent` where 1.0 means 1%.
"""


class DirectionT(tp.NamedTuple):
    LongOnly: int = 0
    ShortOnly: int = 1
    Both: int = 2


Direction = DirectionT()
"""_"""

__pdoc__[
    "Direction"
] = f"""Position direction.

```python
{prettify(Direction)}
```

Attributes:
    LongOnly: Only long positions.
    ShortOnly: Only short positions.
    Both: Both long and short positions.
"""


class LeverageModeT(tp.NamedTuple):
    Lazy: int = 0
    Eager: int = 1


LeverageMode = LeverageModeT()
"""_"""

__pdoc__[
    "LeverageMode"
] = f"""Leverage mode.

```python
{prettify(LeverageMode)}
```

Attributes:
    Lazy: Applies leverage only if free cash has been exhausted.
    Eager: Applies leverage to each order.
"""


class PriceAreaVioModeT(tp.NamedTuple):
    Ignore: int = 0
    Cap: int = 1
    Error: int = 2


PriceAreaVioMode = PriceAreaVioModeT()
"""_"""

__pdoc__[
    "PriceAreaVioMode"
] = f"""Price are violation mode.

```python
{prettify(PriceAreaVioMode)}
```

Attributes:
    Ignore: Ignore any violation.
    Cap: Cap price to prevent violation.
    Error: Throw an error upon violation.
"""


class OrderStatusT(tp.NamedTuple):
    Filled: int = 0
    Ignored: int = 1
    Rejected: int = 2


OrderStatus = OrderStatusT()
"""_"""

__pdoc__[
    "OrderStatus"
] = f"""Order status.

```python
{prettify(OrderStatus)}
```

Attributes:
    Filled: Order has been filled.
    Ignored: Order has been ignored.
    Rejected: Order has been rejected.
"""


class OrderStatusInfoT(tp.NamedTuple):
    SizeNaN: int = 0
    PriceNaN: int = 1
    ValPriceNaN: int = 2
    ValueNaN: int = 3
    ValueZeroNeg: int = 4
    SizeZero: int = 5
    NoCash: int = 6
    NoOpenPosition: int = 7
    MaxSizeExceeded: int = 8
    RandomEvent: int = 9
    CantCoverFees: int = 10
    MinSizeNotReached: int = 11
    PartialFill: int = 12


OrderStatusInfo = OrderStatusInfoT()
"""_"""

__pdoc__[
    "OrderStatusInfo"
] = f"""Order status information.

```python
{prettify(OrderStatusInfo)}
```
"""

status_info_desc = [
    "Size is NaN",
    "Price is NaN",
    "Asset valuation price is NaN",
    "Asset/group value is NaN",
    "Asset/group value is zero or negative",
    "Size is zero",
    "Not enough cash",
    "No open position to reduce/close",
    "Size is greater than maximum allowed",
    "Random event happened",
    "Not enough cash to cover fees",
    "Final size is less than minimum allowed",
    "Final size is less than requested",
]
"""_"""

__pdoc__[
    "status_info_desc"
] = f"""Order status description.

```python
{prettify(status_info_desc)}
```
"""


class OrderSideT(tp.NamedTuple):
    Buy: int = 0
    Sell: int = 1


OrderSide = OrderSideT()
"""_"""

__pdoc__[
    "OrderSide"
] = f"""Order side.

```python
{prettify(OrderSide)}
```
"""


class OrderTypeT(tp.NamedTuple):
    Market: int = 0
    Limit: int = 1


OrderType = OrderTypeT()
"""_"""

__pdoc__[
    "OrderType"
] = f"""Order type.

```python
{prettify(OrderType)}
```
"""


class TradeDirectionT(tp.NamedTuple):
    Long: int = 0
    Short: int = 1


TradeDirection = TradeDirectionT()
"""_"""

__pdoc__[
    "TradeDirection"
] = f"""Event direction.

```python
{prettify(TradeDirection)}
```
"""


class TradeStatusT(tp.NamedTuple):
    Open: int = 0
    Closed: int = 1


TradeStatus = TradeStatusT()
"""_"""

__pdoc__[
    "TradeStatus"
] = f"""Event status.

```python
{prettify(TradeStatus)}
```
"""


class TradesTypeT(tp.NamedTuple):
    Trades: int = 0
    EntryTrades: int = 1
    ExitTrades: int = 2
    Positions: int = 3


TradesType = TradesTypeT()
"""_"""

__pdoc__[
    "TradesType"
] = f"""Trades type.

```python
{prettify(TradesType)}
```
"""


class OrderPriceStatusT(tp.NamedTuple):
    OK: int = 0
    AboveHigh: int = 1
    BelowLow: int = 2
    Unknown: int = 3


OrderPriceStatus = OrderPriceStatusT()
"""_"""

__pdoc__[
    "OrderPriceStatus"
] = f"""Order price error.

```python
{prettify(OrderPriceStatus)}
```

Attributes:
    OK: Order price is within OHLC.
    AboveHigh: Order price is above high.
    BelowLow: Order price is below low.
    Unknown: High and/or low are unknown.
"""


# ############# Named tuples ############# #


class PriceArea(tp.NamedTuple):
    open: float
    high: float
    low: float
    close: float


__pdoc__[
    "PriceArea"
] = """Price area defined by four boundaries.

Used together with `PriceAreaVioMode`."""
__pdoc__["PriceArea.open"] = "Opening price of the time step."
__pdoc__[
    "PriceArea.high"
] = """Highest price of the time step.

Violation takes place when adjusted price goes above this value.
"""
__pdoc__[
    "PriceArea.low"
] = """Lowest price of the time step.

Violation takes place when adjusted price goes below this value.
"""
__pdoc__[
    "PriceArea.close"
] = """Closing price of the time step.

Violation takes place when adjusted price goes beyond this value.
"""

NoPriceArea = PriceArea(open=np.nan, high=np.nan, low=np.nan, close=np.nan)
"""_"""

__pdoc__["NoPriceArea"] = "No price area."


class AccountState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    locked_cash: float
    free_cash: float


__pdoc__["AccountState"] = "State of the account."
__pdoc__["AccountState.cash"] = """Cash. 

Per group with cash sharing, otherwise per column."""
__pdoc__["AccountState.position"] = """Position. 

Per column."""
__pdoc__["AccountState.debt"] = """Debt. 

Per column."""
__pdoc__["AccountState.locked_cash"] = """Locked cash.

Per column."""
__pdoc__["AccountState.free_cash"] = """Free cash. 

Per group with cash sharing, otherwise per column."""


class ExecState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    locked_cash: float
    free_cash: float
    val_price: float
    value: float


__pdoc__["ExecState"] = "State before or after order execution."
__pdoc__["ExecState.cash"] = "See `AccountState.cash`."
__pdoc__["ExecState.position"] = "See `AccountState.position`."
__pdoc__["ExecState.debt"] = "See `AccountState.debt`."
__pdoc__["ExecState.locked_cash"] = "See `AccountState.locked_cash`."
__pdoc__["ExecState.free_cash"] = "See `AccountState.free_cash`."
__pdoc__["ExecState.val_price"] = "Valuation price in the current column."
__pdoc__["ExecState.value"] = "Value in the current column (or group with cash sharing)."


class SimulationOutput(tp.NamedTuple):
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    cash_deposits: tp.Array2d
    cash_earnings: tp.Array2d
    call_seq: tp.Optional[tp.Array2d]
    in_outputs: tp.Optional[tp.NamedTuple]


__pdoc__["SimulationOutput"] = "A named tuple representing the output of a simulation."
__pdoc__["SimulationOutput.order_records"] = "Order records (flattened)."
__pdoc__["SimulationOutput.log_records"] = "Log records (flattened)."
__pdoc__["SimulationOutput.cash_deposits"] = """Cash deposited/withdrawn at each timestamp.

If not tracked, becomes shape `(0, 0)`."""
__pdoc__["SimulationOutput.cash_earnings"] = """Cash earnings added/removed at each timestamp.

If not tracked, becomes shape `(0, 0)`."""
__pdoc__["SimulationOutput.call_seq"] = """Call sequence.

If not tracked, becomes None."""
__pdoc__["SimulationOutput.in_outputs"] = """Named tuple with in-output objects.

If not tracked, becomes None."""


class SimulationContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray


__pdoc__[
    "SimulationContext"
] = """A named tuple representing the context of a simulation.

Contains general information available to all other contexts.

Passed to `pre_sim_func_nb` and `post_sim_func_nb`."""
__pdoc__[
    "SimulationContext.target_shape"
] = """Target shape of the simulation.

A tuple with exactly two elements: the number of rows and columns.

Example:
    One day of minute data for three assets would yield a `target_shape` of `(1440, 3)`,
    where the first axis are rows (minutes) and the second axis are columns (assets).
"""
__pdoc__[
    "SimulationContext.group_lens"
] = """Number of columns in each group.

Even if columns are not grouped, `group_lens` contains ones - one column per group.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.

Example:
    In pairs trading, `group_lens` would be `np.array([2])`, while three independent
    columns would be represented by `group_lens` of `np.array([1, 1, 1])`.
"""
__pdoc__["SimulationContext.cash_sharing"] = "Whether cash sharing is enabled."
__pdoc__[
    "SimulationContext.call_seq"
] = """Default sequence of calls per segment.

Controls the sequence in which `order_func_nb` is executed within each segment.

Has shape `SimulationContext.target_shape` and each value must exist in the range `[0, group_len)`.
Can also be None if not provided.

!!! note
    To use `sort_call_seq_1d_nb`, must be generated using `CallSeqType.Default`.

    To change the call sequence dynamically, better change `SegmentContext.call_seq_now` in-place.
    
Example:
    The default call sequence for three data points and two groups with three columns each:
    
    ```python
    np.array([
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2]
    ])
    ```
"""
__pdoc__[
    "SimulationContext.init_cash"
] = """Initial capital per column (or per group with cash sharing).

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb`.

Must broadcast to shape `(group_lens.shape[0],)` with cash sharing, otherwise `(target_shape[1],)`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.

Example:
    Consider three columns, each having $100 of starting capital. If we built one group of two columns
    and one group of one column, the `init_cash` would be `np.array([200, 100])` with cash sharing
    and `np.array([100, 100, 100])` without cash sharing.
"""
__pdoc__[
    "SimulationContext.init_position"
] = """Initial position per column.

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb`.

Must broadcast to shape `(target_shape[1],)`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.init_price"
] = """Initial position price per column.

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb`.

Must broadcast to shape `(target_shape[1],)`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.cash_deposits"
] = """Cash to be deposited/withdrawn per column 
(or per group with cash sharing).

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_nb`.

Must broadcast to shape `(target_shape[0], group_lens.shape[0])`.

Cash is deposited/withdrawn right after `pre_segment_func_nb`.
You can modify this array in `pre_segment_func_nb`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.
"""
__pdoc__[
    "SimulationContext.cash_earnings"
] = """Earnings to be added per column.

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_nb`.

Must broadcast to shape `SimulationContext.target_shape`.

Earnings are added right before `post_segment_func_nb` and are already included 
in the value of each group. You can modify this array in `pre_segment_func_nb` or `post_order_func_nb`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.
"""
__pdoc__[
    "SimulationContext.segment_mask"
] = """Mask of whether a particular segment should be executed.

A segment is simply a sequence of `order_func_nb` calls under the same group and row.

If a segment is inactive, any callback function inside of it will not be executed.
You can still execute the segment's pre- and postprocessing function by enabling 
`SimulationContext.call_pre_segment` and `SimulationContext.call_post_segment` respectively.

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_nb`.

Must broadcast to shape `(target_shape[0], group_lens.shape[0])`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.

Example:
    Consider two groups with two columns each and the following activity mask:
    
    ```python
    np.array([[ True, False], 
              [False,  True]])
    ```
    
    The first group is only executed in the first row and the second group is only executed in the second row.
"""
__pdoc__[
    "SimulationContext.call_pre_segment"
] = """Whether to call `pre_segment_func_nb` regardless of 
`SimulationContext.segment_mask`."""
__pdoc__[
    "SimulationContext.call_post_segment"
] = """Whether to call `post_segment_func_nb` regardless of 
`SimulationContext.segment_mask`.

Allows, for example, to write user-defined arrays such as returns at the end of each segment."""
__pdoc__[
    "SimulationContext.index"
] = """Index in integer (nanosecond) format.

If datetime-like, assumed to have the UTC timezone. Preset simulation methods
will automatically format any index as UTC without actually converting it to UTC, that is, 
`12:00 +02:00` will become `12:00 +00:00` to avoid timezone conversion issues."""
__pdoc__[
    "SimulationContext.freq"
] = """Frequency of index in integer (nanosecond) format."""
__pdoc__[
    "SimulationContext.open"
] = """Opening price.

Replaces `Order.price` in case it's `-np.inf`.

Similar behavior to that of `SimulationContext.close`."""
__pdoc__[
    "SimulationContext.high"
] = """Highest price.

Similar behavior to that of `SimulationContext.close`."""
__pdoc__[
    "SimulationContext.low"
] = """Lowest price.

Similar behavior to that of `SimulationContext.close`."""
__pdoc__[
    "SimulationContext.close"
] = """Closing price at each time step.

Replaces `Order.price` in case it's `np.inf`.

Acts as a boundary - see `PriceArea.close`.

Utilizes flexible indexing using `vectorbtpro.base.flex_indexing.flex_select_nb`.

Must broadcast to shape `SimulationContext.target_shape`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.
"""
__pdoc__[
    "SimulationContext.bm_close"
] = """Benchmark closing price at each time step.

Must broadcast to shape `SimulationContext.target_shape`."""
__pdoc__[
    "SimulationContext.ffill_val_price"
] = """Whether to track valuation price only if it's known.

Otherwise, unknown `SimulationContext.close` will lead to NaN in valuation price at the next timestamp."""
__pdoc__[
    "SimulationContext.update_value"
] = """Whether to update group value after each filled order.

Otherwise, stays the same for all columns in the group (the value is calculated
only once, before executing any order).

The change is marginal and mostly driven by transaction costs and slippage."""
__pdoc__[
    "SimulationContext.fill_pos_info"
] = """Whether to fill position record.

Disable this to make simulation faster for simple use cases."""
__pdoc__[
    "SimulationContext.track_value"
] = """Whether to track value metrics such as 
the current valuation price, value, and return.

If False, 'SimulationContext.last_val_price', 'SimulationContext.last_value', and 
'SimulationContext.last_return' will stay NaN and the statistics of any open position won't be updated.
You won't be able to use `SizeType.Value`, `SizeType.TargetValue`, and `SizeType.TargetPercent`.

Disable this to make simulation faster for simple use cases."""
__pdoc__[
    "SimulationContext.order_records"
] = """Order records per column.

It's a 2-dimensional array with records of type `order_dt`.

The array is initialized with empty records first (they contain random data), and then 
gradually filled with order data. The number of empty records depends upon `max_orders`, 
but usually it matches the number of rows, meaning there is maximal one order record per element.
`max_orders` can be chosen lower if not every `order_func_nb` leads to a filled order, to save memory.
It can also be chosen higher if more than one order per element is expected.

You can use `SimulationContext.order_counts` to get the number of filled orders in each column.
To get all order records filled up to this point in a column, do `order_records[:order_counts[col], col]`.

Example:
    Before filling, each order record looks like this:
    
    ```python
    np.array([(-8070450532247928832, -8070450532247928832, 4, 0., 0., 0., 5764616306889786413)]
    ```
    
    After filling, it becomes like this:
    
    ```python
    np.array([(0, 0, 1, 50., 1., 0., 1)]
    ```
"""
__pdoc__[
    "SimulationContext.log_records"
] = """Log records per column.

Similar to `SimulationContext.order_records` but of type `log_dt` and count `SimulationContext.log_counts`."""
__pdoc__[
    "SimulationContext.in_outputs"
] = """Named tuple with in-output objects.

Can contain objects of arbitrary shape and type. Will be returned as part of `SimulationOutput`."""
__pdoc__[
    "SimulationContext.last_cash"
] = """Latest cash per column (or per group with cash sharing).

At the very first timestamp, contains initial capital.

Gets updated right after `order_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_position"
] = """Latest position per column.

At the very first timestamp, contains initial position.

Has shape `(target_shape[1],)`.

Gets updated right after `order_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_debt"
] = """Latest debt from leverage or shorting per column.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_locked_cash"
] = """Latest locked cash from leverage or shorting per column.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_free_cash"
] = """Latest free cash per column (or per group with cash sharing).

Free cash never goes above the initial level, because an operation always costs money.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_val_price"
] = """Latest valuation price per column.

Has shape `(target_shape[1],)`.

Enables `SizeType.Value`, `SizeType.TargetValue`, and `SizeType.TargetPercent`.

Gets multiplied by the current position to get the value of the column (see `SimulationContext.last_value`).

Gets updated right before `pre_segment_func_nb` using `SimulationContext.open`.
Then, gets updated right after `pre_segment_func_nb` - you can use `pre_segment_func_nb` to
override `last_val_price` in-place, such that `order_func_nb` can use the new group value. 
If `SimulationContext.update_value`, gets also updated right after `order_func_nb` using 
filled order price as the latest known price. Finally, gets updated right before 
`post_segment_func_nb` using `SimulationContext.close`.

If `SimulationContext.ffill_val_price`, gets updated only if the value is not NaN.
For example, close of `[1, 2, np.nan, np.nan, 5]` yields valuation price of `[1, 2, 2, 2, 5]`.


!!! note
    You are not allowed to use `-np.inf` or `np.inf` - only finite values.
    
    If `SimulationContext.open` is NaN in the first row, the `last_val_price` is also NaN.

Example:
    Consider 10 units in column 1 and 20 units in column 2. The current opening price of them is 
    $40 and $50 respectively, which is also the default valuation price in the current row,
    available as `last_val_price` in `pre_segment_func_nb`. If both columns are in the same group 
    with cash sharing, the group is valued at $1400 before any `order_func_nb` is called, and can 
    be later accessed via `OrderContext.value_now`.
"""
__pdoc__[
    "SimulationContext.last_value"
] = """Latest value per column (or per group with cash sharing).

Calculated by multiplying valuation price by the current position.
The value in each column in a group with cash sharing is summed to get the value of the entire group.

Gets updated right before `pre_segment_func_nb`. Then, gets updated right after `pre_segment_func_nb`.
If `SimulationContext.update_value`, gets also updated right after `order_func_nb` using 
filled order price as the latest known price (the difference will be minimal, 
only affected by costs). Finally, gets updated right before `post_segment_func_nb`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_return"
] = """Latest return per column (or per group with cash sharing).

Has the same shape as `SimulationContext.last_value`.

Calculated by comparing the current `SimulationContext.last_value` to the last one of the previous row.

Gets updated each time `SimulationContext.last_value` is updated.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.order_counts"
] = """Number of filled order records in each column.

Points to `SimulationContext.order_records` and has shape `(target_shape[1],)`.

Example:
    `order_counts` of `np.array([2, 100, 0])` means the latest filled order is `order_records[1, 0]` in the
    first column, `order_records[99, 1]` in the second column, and no orders have been filled yet
    in the third column (`order_records[0, 2]` is empty).
    
    !!! note
        Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.log_counts"
] = """Number of filled log records in each column.

Similar to `SimulationContext.log_counts` but for log records.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_pos_info"
] = """Latest position record in each column.

It's a 1-dimensional array with records of type `trade_dt`.

Has shape `(target_shape[1],)`.

If `SimulationContext.init_position` is not zero in a column, that column's position record
is automatically filled before the simulation with `entry_price` set to `SimulationContext.init_price` and 
`entry_idx` of -1.

The fields `entry_price` and `exit_price` are average entry and exit price respectively.
The average exit price does **not** contain open statistics, as opposed to `vectorbtpro.portfolio.trades.Positions`.
On the other hand, fields `pnl` and `return` contain statistics as if the position has been closed and are 
re-calculated using `SimulationContext.last_val_price` right before and after `pre_segment_func_nb`, 
right after `order_func_nb`, and right before `post_segment_func_nb`.

!!! note
    In an open position record, the field `exit_price` doesn't reflect the latest valuation price,
    but keeps the average price at which the position has been reduced.
"""


class GroupContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int


__pdoc__[
    "GroupContext"
] = """A named tuple representing the context of a group.

A group is a set of nearby columns that are somehow related (for example, by sharing the same capital).
In each row, the columns under the same group are bound to the same segment.

Contains all fields from `SimulationContext` plus fields describing the current group.

Passed to `pre_group_func_nb` and `post_group_func_nb`.

Example:
    Consider a group of three columns, a group of two columns, and one more column:
    
    | group | group_len | from_col | to_col |
    | ----- | --------- | -------- | ------ |
    | 0     | 3         | 0        | 3      |
    | 1     | 2         | 3        | 5      |
    | 2     | 1         | 5        | 6      |
"""
for field in GroupContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["GroupContext." + field] = f"See `SimulationContext.{field}`."
__pdoc__[
    "GroupContext.group"
] = """Index of the current group.

Has range `[0, group_lens.shape[0])`.
"""
__pdoc__[
    "GroupContext.group_len"
] = """Number of columns in the current group.

Scalar value. Same as `group_lens[group]`.
"""
__pdoc__[
    "GroupContext.from_col"
] = """Index of the first column in the current group.

Has range `[0, target_shape[1])`.
"""
__pdoc__[
    "GroupContext.to_col"
] = """Index of the last column in the current group plus one.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals to `from_col + 1`.

!!! warning
    In the last group, `to_col` points at a column that doesn't exist.
"""


class RowContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    i: int


__pdoc__[
    "RowContext"
] = """A named tuple representing the context of a row.

A row is a time step in which segments are executed.

Contains all fields from `SimulationContext` plus fields describing the current row.

Passed to `pre_row_func_nb` and `post_row_func_nb`.
"""
for field in RowContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["RowContext." + field] = f"See `SimulationContext.{field}`."
__pdoc__[
    "RowContext.i"
] = """Index of the current row.

Has range `[0, target_shape[0])`.
"""


class SegmentContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Optional[tp.Array1d]


__pdoc__[
    "SegmentContext"
] = """A named tuple representing the context of a segment.

A segment is an intersection between groups and rows. It's an entity that defines
how and in which order elements within the same group and row are processed.

Contains all fields from `SimulationContext`, `GroupContext`, and `RowContext`, plus fields 
describing the current segment.

Passed to `pre_segment_func_nb` and `post_segment_func_nb`.
"""
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `RowContext.{field}`."
__pdoc__[
    "SegmentContext.call_seq_now"
] = """Sequence of calls within the current segment.

Has shape `(group_len,)`. 

Each value in this sequence must indicate the position of column in the group to
call next. Processing goes always from left to right.

You can use `pre_segment_func_nb` to override `call_seq_now`.
    
Example:
    `[2, 0, 1]` would first call column 2, then 0, and finally 1.
"""


class OrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Optional[tp.Array1d]
    col: int
    call_idx: int
    cash_now: float
    position_now: float
    debt_now: float
    locked_cash_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float
    return_now: float
    pos_info_now: tp.Record


__pdoc__[
    "OrderContext"
] = """A named tuple representing the context of an order.

Contains all fields from `SegmentContext` plus fields describing the current state.

Passed to `order_func_nb`.
"""
for field in OrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["OrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["OrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["OrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["OrderContext." + field] = f"See `SegmentContext.{field}`."
__pdoc__[
    "OrderContext.col"
] = """Current column.

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`.
"""
__pdoc__[
    "OrderContext.call_idx"
] = """Index of the current call in `SegmentContext.call_seq_now`.

Has range `[0, group_len)`.
"""
__pdoc__["OrderContext.cash_now"] = "`SimulationContext.last_cash` for the current column/group."
__pdoc__["OrderContext.position_now"] = "`SimulationContext.last_position` for the current column."
__pdoc__["OrderContext.debt_now"] = "`SimulationContext.last_debt` for the current column."
__pdoc__["OrderContext.locked_cash_now"] = "`SimulationContext.last_locked_cash` for the current column."
__pdoc__["OrderContext.free_cash_now"] = "`SimulationContext.last_free_cash` for the current column/group."
__pdoc__["OrderContext.val_price_now"] = "`SimulationContext.last_val_price` for the current column."
__pdoc__["OrderContext.value_now"] = "`SimulationContext.last_value` for the current column/group."
__pdoc__["OrderContext.return_now"] = "`SimulationContext.last_return` for the current column/group."
__pdoc__["OrderContext.pos_info_now"] = "`SimulationContext.last_pos_info` for the current column."


class PostOrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Optional[tp.Array1d]
    col: int
    call_idx: int
    cash_before: float
    position_before: float
    debt_before: float
    locked_cash_before: float
    free_cash_before: float
    val_price_before: float
    value_before: float
    order_result: "OrderResult"
    cash_now: float
    position_now: float
    debt_now: float
    locked_cash_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float
    return_now: float
    pos_info_now: tp.Record


__pdoc__[
    "PostOrderContext"
] = """A named tuple representing the context after an order has been processed.

Contains all fields from `OrderContext` plus fields describing the order result and the previous state.

Passed to `post_order_func_nb`.
"""
for field in PostOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `SegmentContext.{field}`."
    elif field in OrderContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `OrderContext.{field}`."
__pdoc__["PostOrderContext.cash_before"] = "`OrderContext.cash_now` before execution."
__pdoc__["PostOrderContext.position_before"] = "`OrderContext.position_now` before execution."
__pdoc__["PostOrderContext.debt_before"] = "`OrderContext.debt_now` before execution."
__pdoc__["PostOrderContext.locked_cash_before"] = "`OrderContext.locked_cash_now` before execution."
__pdoc__["PostOrderContext.free_cash_before"] = "`OrderContext.free_cash_now` before execution."
__pdoc__["PostOrderContext.val_price_before"] = "`OrderContext.val_price_now` before execution."
__pdoc__["PostOrderContext.value_before"] = "`OrderContext.value_now` before execution."
__pdoc__[
    "PostOrderContext.order_result"
] = """Order result of type `OrderResult`.

Can be used to check whether the order has been filled, ignored, or rejected.
"""
__pdoc__["PostOrderContext.cash_now"] = "`OrderContext.cash_now` after execution."
__pdoc__["PostOrderContext.position_now"] = "`OrderContext.position_now` after execution."
__pdoc__["PostOrderContext.debt_now"] = "`OrderContext.debt_now` after execution."
__pdoc__["PostOrderContext.locked_cash_now"] = "`OrderContext.locked_cash_now` after execution."
__pdoc__["PostOrderContext.free_cash_now"] = "`OrderContext.free_cash_now` after execution."
__pdoc__[
    "PostOrderContext.val_price_now"
] = """`OrderContext.val_price_now` after execution.

If `SimulationContext.update_value`, gets replaced with the fill price, 
as it becomes the most recently known price. Otherwise, stays the same.
"""
__pdoc__[
    "PostOrderContext.value_now"
] = """`OrderContext.value_now` after execution.

If `SimulationContext.update_value`, gets updated with the new cash and value of the column. Otherwise, stays the same.
"""
__pdoc__["PostOrderContext.return_now"] = "`OrderContext.return_now` after execution."
__pdoc__["PostOrderContext.pos_info_now"] = "`OrderContext.pos_info_now` after execution."


class FlexOrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    in_outputs: tp.Optional[tp.NamedTuple]
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    order_counts: tp.Array1d
    log_counts: tp.Array1d
    last_pos_info: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: None
    call_idx: int


__pdoc__[
    "FlexOrderContext"
] = """A named tuple representing the context of a flexible order.

Contains all fields from `SegmentContext` plus the current call index.

Passed to `flex_order_func_nb`.
"""
for field in FlexOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `SegmentContext.{field}`."
__pdoc__["FlexOrderContext.call_idx"] = "Index of the current call."


class Order(tp.NamedTuple):
    size: float = np.inf
    price: float = np.inf
    size_type: int = SizeType.Amount
    direction: int = Direction.Both
    fees: float = 0.0
    fixed_fees: float = 0.0
    slippage: float = 0.0
    min_size: float = np.nan
    max_size: float = np.nan
    size_granularity: float = np.nan
    leverage: float = 1.0
    leverage_mode: int = LeverageMode.Lazy
    reject_prob: float = 0.0
    price_area_vio_mode: int = PriceAreaVioMode.Ignore
    allow_partial: bool = True
    raise_reject: bool = False
    log: bool = False


__pdoc__[
    "Order"
] = """A named tuple representing an order.

!!! note
    Currently, Numba has issues with using defaults when filling named tuples. 
    Use `vectorbtpro.portfolio.nb.core.order_nb` to create an order."""
__pdoc__[
    "Order.size"
] = """Size in units.

Behavior depends upon `Order.size_type` and `Order.direction`.

For any fixed size:

* Set to any number to buy/sell some fixed amount or value.
* Set to `np.inf` to buy for all cash, or `-np.inf` to sell for all free cash.
    If `Order.direction` is not `Direction.Both`, `-np.inf` will close the position.
* Set to `np.nan` or 0 to skip.

For any target size:

* Set to any number to buy/sell an amount relative to the current position or value.
* Set to 0 to close the current position.
* Set to `np.nan` to skip.
"""
__pdoc__[
    "Order.price"
] = """Price per unit. 

Final price will depend upon slippage.

* If `-np.inf`, gets replaced by the current open.
* If `np.inf`, gets replaced by the current close.

!!! note
    Make sure to use timestamps that come between (and ideally not including) the current open and close."""
__pdoc__["Order.size_type"] = "See `SizeType`."
__pdoc__["Order.direction"] = "See `Direction`."
__pdoc__[
    "Order.fees"
] = """Fees in percentage of the order value.

Negative trading fees like -0.05 mean earning 5% per trade instead of paying a fee.

!!! note
    0.01 = 1%."""
__pdoc__[
    "Order.fixed_fees"
] = """Fixed amount of fees to pay for this order.

Similar to `Order.fees`, can be negative."""
__pdoc__[
    "Order.slippage"
] = """Slippage in percentage of `Order.price`. 

Slippage is a penalty applied on the price.

!!! note
    0.01 = 1%."""
__pdoc__[
    "Order.min_size"
] = """Minimum size in both directions. 

Depends on `Order.size_type`. Lower than that will be rejected."""
__pdoc__[
    "Order.max_size"
] = """Maximum size in both directions. 

Depends on `Order.size_type`. Higher than that will be partly filled."""
__pdoc__[
    "Order.size_granularity"
] = """Granularity of the size.

For example, granularity of 1.0 makes the quantity to behave like an integer. 
Placing an order of 12.5 shares (in any direction) will order exactly 12.0 shares.

!!! note
    The filled size remains a floating number."""
__pdoc__["Order.leverage"] = "Leverage."
__pdoc__["Order.leverage_mode"] = "See `LeverageMode`."
__pdoc__[
    "Order.reject_prob"
] = """Probability of rejecting this order to simulate a random rejection event.

Not everything goes smoothly in real life. Use random rejections to test your order management for robustness."""
__pdoc__["Order.price_area_vio_mode"] = "See `PriceAreaVioMode`."
__pdoc__[
    "Order.allow_partial"
] = """Whether to allow partial fill.

Otherwise, the order gets rejected.

Does not apply when `Order.size` is `np.inf`."""
__pdoc__[
    "Order.raise_reject"
] = """Whether to raise exception if order has been rejected.

Terminates the simulation."""
__pdoc__[
    "Order.log"
] = """Whether to log this order by filling a log record. 

Remember to increase `max_logs`."""

NoOrder = Order(
    size=np.nan,
    price=np.nan,
    size_type=-1,
    direction=-1,
    fees=np.nan,
    fixed_fees=np.nan,
    slippage=np.nan,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode=LeverageMode.Lazy,
    reject_prob=np.nan,
    price_area_vio_mode=-1,
    allow_partial=False,
    raise_reject=False,
    log=False,
)
"""_"""

__pdoc__["NoOrder"] = "Order that should not be processed."


class OrderResult(tp.NamedTuple):
    size: float
    price: float
    fees: float
    side: int
    status: int
    status_info: int


__pdoc__["OrderResult"] = "A named tuple representing an order result."
__pdoc__["OrderResult.size"] = "Filled size."
__pdoc__["OrderResult.price"] = "Filled price per unit, adjusted with slippage."
__pdoc__["OrderResult.fees"] = "Total fees paid for this order."
__pdoc__["OrderResult.side"] = "See `OrderSide`."
__pdoc__["OrderResult.status"] = "See `OrderStatus`."
__pdoc__["OrderResult.status_info"] = "See `OrderStatusInfo`."


class SignalSegmentContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    track_cash_deposits: bool
    cash_deposits_out: tp.Array2d
    track_cash_earnings: bool
    cash_earnings_out: tp.Array2d
    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d

    last_pos_info: tp.Array1d
    last_limit_info: tp.Array1d
    last_sl_info: tp.Array1d
    last_tsl_info: tp.Array1d
    last_tp_info: tp.Array1d
    last_td_info: tp.Array1d
    last_dt_info: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int


__pdoc__[
    "SignalSegmentContext"
] = """A named tuple representing the context of a segment in a from-signals simulation.

Contains information related to the cascade of the simulation, such as OHLC, but also
internal information that is not passed by the user but created at the beginning of the simulation.
To make use of other information, such as order size, use templates.

Passed to `post_segment_func_nb`."""
__pdoc__["SignalSegmentContext.target_shape"] = "See `SimulationContext.target_shape`."
__pdoc__["SignalSegmentContext.group_lens"] = "See `SimulationContext.group_lens`."
__pdoc__["SignalSegmentContext.cash_sharing"] = "See `SimulationContext.cash_sharing`."
__pdoc__["SignalSegmentContext.index"] = "See `SimulationContext.index`."
__pdoc__["SignalSegmentContext.freq"] = "See `SimulationContext.freq`."
__pdoc__["SignalSegmentContext.open"] = "See `SimulationContext.open`."
__pdoc__["SignalSegmentContext.high"] = "See `SimulationContext.high`."
__pdoc__["SignalSegmentContext.low"] = "See `SimulationContext.low`."
__pdoc__["SignalSegmentContext.close"] = "See `SimulationContext.close`."
__pdoc__["SignalSegmentContext.init_cash"] = "See `SimulationContext.init_cash`."
__pdoc__["SignalSegmentContext.init_position"] = "See `SimulationContext.init_position`."
__pdoc__["SignalSegmentContext.init_price"] = "See `SimulationContext.init_price`."
__pdoc__["SignalSegmentContext.order_records"] = "See `SimulationContext.order_records`."
__pdoc__["SignalSegmentContext.order_counts"] = "See `SimulationContext.order_counts`."
__pdoc__["SignalSegmentContext.log_records"] = "See `SimulationContext.log_records`."
__pdoc__["SignalSegmentContext.log_counts"] = "See `SimulationContext.log_counts`."
__pdoc__["SignalSegmentContext.track_cash_deposits"] = """Whether to track cash deposits.

Becomes True if any value in `cash_deposits` is not zero."""
__pdoc__["SignalSegmentContext.cash_deposits_out"] = "See `SimulationOutput.cash_deposits`."
__pdoc__["SignalSegmentContext.track_cash_earnings"] = """Whether to track cash earnings.

Becomes True if any value in `cash_earnings` is not zero."""
__pdoc__["SignalSegmentContext.cash_earnings_out"] = "See `SimulationOutput.cash_earnings`."
__pdoc__["SignalSegmentContext.in_outputs"] = "See `FSInOutputs`."
__pdoc__["SignalSegmentContext.last_cash"] = "See `SimulationContext.last_cash`."
__pdoc__["SignalSegmentContext.last_position"] = "See `SimulationContext.last_position`."
__pdoc__["SignalSegmentContext.last_debt"] = "See `SimulationContext.last_debt`."
__pdoc__["SignalSegmentContext.last_free_cash"] = "See `SimulationContext.last_free_cash`."
__pdoc__["SignalSegmentContext.last_val_price"] = "See `SimulationContext.last_val_price`."
__pdoc__["SignalSegmentContext.last_value"] = "See `SimulationContext.last_value`."
__pdoc__["SignalSegmentContext.last_return"] = "See `SimulationContext.last_return`."
__pdoc__["SignalSegmentContext.last_pos_info"] = "See `SimulationContext.last_pos_info`."
__pdoc__["SignalSegmentContext.last_limit_info"] = """Record of type `limit_info_dt` per column.

Accessible via `c.limit_info_dt[field][col]`."""
__pdoc__["SignalSegmentContext.last_sl_info"] = """Record of type `sl_info_dt` per column.

Accessible via `c.last_sl_info[field][col]`."""
__pdoc__["SignalSegmentContext.last_tsl_info"] = """Record of type `tsl_info_dt` per column.

Accessible via `c.last_tsl_info[field][col]`."""
__pdoc__["SignalSegmentContext.last_tp_info"] = """Record of type `tp_info_dt` per column.

Accessible via `c.last_tp_info[field][col]`."""
__pdoc__["SignalSegmentContext.last_td_info"] = """Record of type `time_info_dt` per column.

Accessible via `c.last_td_info[field][col]`."""
__pdoc__["SignalSegmentContext.last_dt_info"] = """Record of type `time_info_dt` per column.

Accessible via `c.last_dt_info[field][col]`."""
__pdoc__["SignalSegmentContext.group"] = "See `GroupContext.group`."
__pdoc__["SignalSegmentContext.group_len"] = "See `GroupContext.group_len`."
__pdoc__["SignalSegmentContext.from_col"] = "See `GroupContext.from_col`."
__pdoc__["SignalSegmentContext.to_col"] = "See `GroupContext.to_col`."
__pdoc__["SignalSegmentContext.i"] = "See `RowContext.i`."


class SignalContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.Array1d
    cash_sharing: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    track_cash_deposits: bool
    cash_deposits_out: tp.Array2d
    track_cash_earnings: bool
    cash_earnings_out: tp.Array2d
    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d

    last_pos_info: tp.Array1d
    last_limit_info: tp.Array1d
    last_sl_info: tp.Array1d
    last_tsl_info: tp.Array1d
    last_tp_info: tp.Array1d
    last_td_info: tp.Array1d
    last_dt_info: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    col: int


__pdoc__[
    "SignalContext"
] = """A named tuple representing the context of an element in a from-signals simulation.

Contains all fields from `SignalSegmentContext` plus the column field.

Passed to `signal_func_nb` and `adjust_func_nb`.
"""
for field in SignalContext._fields:
    if field in SignalSegmentContext._fields:
        __pdoc__["SignalContext." + field] = f"See `SignalSegmentContext.{field}`."
__pdoc__["SignalContext.col"] = "See `OrderContext.col`."


# ############# In-outputs ############# #


class FOInOutputs(tp.NamedTuple):
    cash: tp.Array2d
    position: tp.Array2d
    debt: tp.Array2d
    locked_cash: tp.Array2d
    free_cash: tp.Array2d
    value: tp.Array2d
    returns: tp.Array2d


__pdoc__["FOInOutputs"] = "A named tuple representing the in-outputs for simulation based on orders."
__pdoc__[
    "FOInOutputs.cash"
] = """See `AccountState.cash`.

Follows groups if cash sharing is enabled, otherwise columns.

Gets filled if `save_state` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.position"
] = """See `AccountState.position`.

Follows columns.

Gets filled if `save_state` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.debt"
] = """See `AccountState.debt`.

Follows columns.

Gets filled if `save_state` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.locked_cash"
] = """See `AccountState.locked_cash`.

Follows columns.

Gets filled if `save_state` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.free_cash"
] = """See `AccountState.free_cash`.

Follows groups if cash sharing is enabled, otherwise columns.

Gets filled if `save_state` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.value"
] = """Value.

Follows groups if cash sharing is enabled, otherwise columns.

Gets filled if `fill_value` is True, otherwise has the shape `(0, 0)`."""
__pdoc__[
    "FOInOutputs.returns"
] = """Returns.

Follows groups if cash sharing is enabled, otherwise columns.

Gets filled if `save_returns` is True, otherwise has the shape `(0, 0)`."""


class FSInOutputs(tp.NamedTuple):
    cash: tp.Array2d
    position: tp.Array2d
    debt: tp.Array2d
    locked_cash: tp.Array2d
    free_cash: tp.Array2d
    returns: tp.Array2d


__pdoc__["FSInOutputs"] = "A named tuple representing the in-outputs for simulation based on signals."
__pdoc__["FSInOutputs.cash"] = "See `FOInOutputs.cash`."
__pdoc__["FSInOutputs.position"] = "See `FOInOutputs.position`."
__pdoc__["FSInOutputs.debt"] = "See `FOInOutputs.debt`."
__pdoc__["FSInOutputs.locked_cash"] = "See `FOInOutputs.locked_cash`."
__pdoc__["FSInOutputs.free_cash"] = "See `FOInOutputs.free_cash`."
__pdoc__["FSInOutputs.returns"] = "See `FOInOutputs.returns`."

# ############# Records ############# #

order_fields = [
    ("id", np.int_),
    ("col", np.int_),
    ("idx", np.int_),
    ("size", np.float_),
    ("price", np.float_),
    ("fees", np.float_),
    ("side", np.int_),
]
"""Fields for `order_dt`."""

order_dt = np.dtype(order_fields, align=True)
"""_"""

__pdoc__[
    "order_dt"
] = f"""`np.dtype` of order records.

```python
{prettify(order_dt)}
```
"""

fs_order_fields = [
    ("id", np.int_),
    ("col", np.int_),
    ("signal_idx", np.int_),
    ("creation_idx", np.int_),
    ("idx", np.int_),
    ("size", np.float_),
    ("price", np.float_),
    ("fees", np.float_),
    ("side", np.int_),
    ("type", np.int_),
    ("stop_type", np.int_),
]
"""Fields for `fs_order_dt`."""

fs_order_dt = np.dtype(fs_order_fields, align=True)
"""_"""

__pdoc__[
    "fs_order_dt"
] = f"""`np.dtype` of order records generated from signals.

```python
{prettify(fs_order_dt)}
```
"""

trade_fields = [
    ("id", np.int_),
    ("col", np.int_),
    ("size", np.float_),
    ("entry_order_id", np.int_),
    ("entry_idx", np.int_),
    ("entry_price", np.float_),
    ("entry_fees", np.float_),
    ("exit_order_id", np.int_),
    ("exit_idx", np.int_),
    ("exit_price", np.float_),
    ("exit_fees", np.float_),
    ("pnl", np.float_),
    ("return", np.float_),
    ("direction", np.int_),
    ("status", np.int_),
    ("parent_id", np.int_),
]
"""Fields for `trade_dt`."""

trade_dt = np.dtype(trade_fields, align=True)
"""_"""

__pdoc__[
    "trade_dt"
] = f"""`np.dtype` of trade records.

```python
{prettify(trade_dt)}
```
"""

log_fields = [
    ("id", np.int_),
    ("group", np.int_),
    ("col", np.int_),
    ("idx", np.int_),
    ("price_area_open", np.float_),
    ("price_area_high", np.float_),
    ("price_area_low", np.float_),
    ("price_area_close", np.float_),
    ("st0_cash", np.float_),
    ("st0_position", np.float_),
    ("st0_debt", np.float_),
    ("st0_locked_cash", np.float_),
    ("st0_free_cash", np.float_),
    ("st0_val_price", np.float_),
    ("st0_value", np.float_),
    ("req_size", np.float_),
    ("req_price", np.float_),
    ("req_size_type", np.int_),
    ("req_direction", np.int_),
    ("req_fees", np.float_),
    ("req_fixed_fees", np.float_),
    ("req_slippage", np.float_),
    ("req_min_size", np.float_),
    ("req_max_size", np.float_),
    ("req_size_granularity", np.float_),
    ("req_leverage", np.float_),
    ("req_leverage_mode", np.int_),
    ("req_reject_prob", np.float_),
    ("req_price_area_vio_mode", np.int_),
    ("req_allow_partial", np.bool_),
    ("req_raise_reject", np.bool_),
    ("req_log", np.bool_),
    ("res_size", np.float_),
    ("res_price", np.float_),
    ("res_fees", np.float_),
    ("res_side", np.int_),
    ("res_status", np.int_),
    ("res_status_info", np.int_),
    ("st1_cash", np.float_),
    ("st1_position", np.float_),
    ("st1_debt", np.float_),
    ("st1_locked_cash", np.float_),
    ("st1_free_cash", np.float_),
    ("st1_val_price", np.float_),
    ("st1_value", np.float_),
    ("order_id", np.int_),
]
"""Fields for `log_fields`."""

log_dt = np.dtype(log_fields, align=True)
"""_"""

__pdoc__[
    "log_dt"
] = f"""`np.dtype` of log records.

```python
{prettify(log_dt)}
```
"""

alloc_range_fields = [
    ("id", np.int_),
    ("col", np.int_),
    ("start_idx", np.int_),
    ("end_idx", np.int_),
    ("alloc_idx", np.int_),
    ("status", np.int_),
]
"""Fields for `alloc_range_dt`."""

alloc_range_dt = np.dtype(alloc_range_fields, align=True)
"""_"""

__pdoc__[
    "alloc_range_dt"
] = f"""`np.dtype` of allocation range records.

```python
{prettify(alloc_range_dt)}
```
"""

alloc_point_fields = [
    ("id", np.int_),
    ("col", np.int_),
    ("alloc_idx", np.int_),
]
"""Fields for `alloc_point_dt`."""

alloc_point_dt = np.dtype(alloc_point_fields, align=True)
"""_"""

__pdoc__[
    "alloc_point_dt"
] = f"""`np.dtype` of allocation point records.

```python
{prettify(alloc_point_dt)}
```
"""

# ############# Info records ############# #

main_info_fields = [
    ("bar_zone", np.int_),
    ("signal_idx", np.int_),
    ("creation_idx", np.int_),
    ("idx", np.int_),
    ("val_price", np.float_),
    ("price", np.float_),
    ("size", np.float_),
    ("size_type", np.int_),
    ("direction", np.int_),
    ("type", np.int_),
    ("stop_type", np.int_),
]
"""Fields for `main_info_dt`."""

main_info_dt = np.dtype(main_info_fields, align=True)
"""_"""

__pdoc__[
    "main_info_dt"
] = f"""`np.dtype` of main information records.

```python
{prettify(main_info_dt)}
```

Attributes:
    bar_zone: See `vectorbtpro.generic.enums.BarZone`.
    signal_idx: Row where signal was placed.
    creation_idx: Row where order was created.
    i: Row from where order information was taken.
    val_price: Valuation price.
    price: Requested price.
    size: Order size.
    size_type: See `SizeType`.
    direction: See `Direction`.
    type: See `OrderType`.
    stop_type: See `vectorbtpro.signals.enums.StopType`.
"""

limit_info_fields = [
    ("signal_idx", np.int_),
    ("creation_idx", np.int_),
    ("init_idx", np.int_),
    ("init_price", np.float_),
    ("init_size", np.float_),
    ("init_size_type", np.int_),
    ("init_direction", np.int_),
    ("init_stop_type", np.int_),
    ("delta", np.float_),
    ("delta_format", np.int_),
    ("tif", int),
    ("expiry", int),
    ("time_delta_format", np.int_),
    ("reverse", np.float_),
]
"""Fields for `limit_info_dt`."""

limit_info_dt = np.dtype(limit_info_fields, align=True)
"""_"""

__pdoc__[
    "limit_info_dt"
] = f"""`np.dtype` of limit information records.

```python
{prettify(limit_info_dt)}
```

Attributes:
    signal_idx: Signal row.
    creation_idx: Limit creation row.
    init_idx: Initial row from where order information is taken.
    init_price: Initial price.
    init_size: Order size.
    init_size_type: See `SizeType`.
    init_direction: See `Direction`.
    init_stop_type: See `vectorbtpro.signals.enums.StopType`.
    delta: Delta from the initial price.
    delta_format: See `DeltaFormat`.
    tif: Time in force in integer format. Set to `-1` to disable.
    expiry: Expiry time in integer format. Set to `-1` to disable.
    time_delta_format: See `TimeDeltaFormat`.
    reverse: Whether to reverse the price hit detection.
"""

sl_info_fields = [
    ("init_idx", np.int_),
    ("init_price", np.float_),
    ("init_position", np.float_),
    ("stop", np.float_),
    ("exit_price", np.float_),
    ("exit_size", np.float_),
    ("exit_size_type", np.int_),
    ("exit_type", np.int_),
    ("order_type", np.int_),
    ("limit_delta", np.float_),
    ("delta_format", np.int_),
    ("ladder", np.int_),
    ("step", np.int_),
    ("step_idx", np.int_),
]
"""Fields for `sl_info_dt`."""

sl_info_dt = np.dtype(sl_info_fields, align=True)
"""_"""

__pdoc__[
    "sl_info_dt"
] = f"""`np.dtype` of SL information records.

```python
{prettify(sl_info_dt)}
```

Attributes:
    init_idx: Initial row.
    init_price: Initial price.
    init_position: Initial position.
    stop: Latest updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price. Only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Step in the ladder (i.e., the number of times the stop was executed)
    step_idx: Step row.
"""

tsl_info_fields = [
    ("init_idx", np.int_),
    ("init_price", np.float_),
    ("init_position", np.float_),
    ("peak_idx", np.int_),
    ("peak_price", np.float_),
    ("stop", np.float_),
    ("th", np.float_),
    ("exit_price", np.float_),
    ("exit_size", np.float_),
    ("exit_size_type", np.int_),
    ("exit_type", np.int_),
    ("order_type", np.int_),
    ("limit_delta", np.float_),
    ("delta_format", np.int_),
    ("ladder", np.int_),
    ("step", np.int_),
    ("step_idx", np.int_),
]
"""Fields for `tsl_info_dt`."""

tsl_info_dt = np.dtype(tsl_info_fields, align=True)
"""_"""

__pdoc__[
    "tsl_info_dt"
] = f"""`np.dtype` of TSL information records.

```python
{prettify(tsl_info_dt)}
```

Attributes:
    init_idx: Initial row.
    init_price: Initial price.
    init_position: Initial position.
    peak_idx: Row of the highest/lowest price.
    peak_price: Highest/lowest price.
    stop: Latest updated stop value.
    th: Latest updated threshold value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price. Only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Step in the ladder (i.e., the number of times the stop was executed)
    step_idx: Step row.
"""

tp_info_fields = [
    ("init_idx", np.int_),
    ("init_price", np.float_),
    ("init_position", np.float_),
    ("stop", np.float_),
    ("exit_price", np.float_),
    ("exit_size", np.float_),
    ("exit_size_type", np.int_),
    ("exit_type", np.int_),
    ("order_type", np.int_),
    ("limit_delta", np.float_),
    ("delta_format", np.int_),
    ("ladder", np.int_),
    ("step", np.int_),
    ("step_idx", np.int_),
]
"""Fields for `tp_info_dt`."""

tp_info_dt = np.dtype(tp_info_fields, align=True)
"""_"""

__pdoc__[
    "tp_info_dt"
] = f"""`np.dtype` of TP information records.

```python
{prettify(tp_info_dt)}
```

Attributes:
    init_idx: Initial row.
    init_price: Initial price.
    init_position: Initial position.
    stop: Latest updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price. Only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Step in the ladder (i.e., the number of times the stop was executed)
    step_idx: Step row.
"""

time_info_fields = [
    ("init_idx", np.int_),
    ("init_position", np.float_),
    ("stop", np.int_),
    ("exit_price", np.float_),
    ("exit_size", np.float_),
    ("exit_size_type", np.int_),
    ("exit_type", np.int_),
    ("order_type", np.int_),
    ("limit_delta", np.float_),
    ("delta_format", np.int_),
    ("time_delta_format", np.int_),
    ("ladder", np.int_),
    ("step", np.int_),
    ("step_idx", np.int_),
]
"""Fields for `time_info_dt`."""

time_info_dt = np.dtype(time_info_fields, align=True)
"""_"""

__pdoc__[
    "time_info_dt"
] = f"""`np.dtype` of time information records.

```python
{prettify(time_info_dt)}
```

Attributes:
    init_idx: Initial row.
    init_position: Initial position.
    stop: Latest updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price. Only for `StopType.Limit`.
    delta_format: See `DeltaFormat`. Only for `StopType.Limit`.
    time_delta_format: See `TimeDeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Step in the ladder (i.e., the number of times the stop was executed)
    step_idx: Step row.
"""
