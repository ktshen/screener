# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with trade records.

Trade records capture information on trades.

In vectorbt, a trade is a sequence of orders that starts with an opening order and optionally ends
with a closing order. Every pair of opposite orders can be represented by a trade. Each trade has a PnL
info attached to quickly assess its performance. An interesting effect of this representation
is the ability to aggregate trades: if two or more trades are happening one after another in time,
they can be aggregated into a bigger trade. This way, for example, single-order trades can be aggregated
into positions; but also multiple positions can be aggregated into a single blob that reflects the performance
of the entire symbol.

!!! warning
    All classes return both closed AND open trades/positions, which may skew your performance results.
    To only consider closed trades/positions, you should explicitly query the `status_closed` attribute.

## Trade types

There are three main types of trades.

### Entry trades

An entry trade is created from each order that opens or adds to a position.

For example, if we have a single large buy order and 100 smaller sell orders, we will see
a single trade with the entry information copied from the buy order and the exit information being
a size-weighted average over the exit information of all sell orders. On the other hand,
if we have 100 smaller buy orders and a single sell order, we will see 100 trades,
each with the entry information copied from the buy order and the exit information being
a size-based fraction of the exit information of the sell order.

Use `vectorbtpro.portfolio.trades.EntryTrades.from_orders` to build entry trades from orders.
Also available as `vectorbtpro.portfolio.base.Portfolio.entry_trades`.

### Exit trades

An exit trade is created from each order that closes or removes from a position.

Use `vectorbtpro.portfolio.trades.ExitTrades.from_orders` to build exit trades from orders.
Also available as `vectorbtpro.portfolio.base.Portfolio.exit_trades`.

### Positions

A position is created from a sequence of entry or exit trades.

Use `vectorbtpro.portfolio.trades.Positions.from_trades` to build positions from entry or exit trades.
Also available as `vectorbtpro.portfolio.base.Portfolio.positions`.

## Example

* Increasing position:

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbtpro as vbt

>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 1., 1., 1., -4.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0
1               1       0   1.0               1            1              2.0
2               2       0   1.0               2            2              3.0
3               3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees   PnL  \\
0         1.0              4           4             5.0       0.25  2.75
1         1.0              4           4             5.0       0.25  1.75
2         1.0              4           4             5.0       0.25  0.75
3         1.0              4           4             5.0       0.25 -0.25

   Return Direction  Status  Position Id
0  2.7500      Long  Closed            0
1  0.8750      Long  Closed            0
2  0.2500      Long  Closed            0
3 -0.0625      Long  Closed            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   4.0               0            0              2.5

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         4.0              4           4             5.0        1.0  5.0

   Return Direction  Status  Position Id
0     0.5      Long  Closed            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   4.0               0            0              2.5

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         4.0              4           4             5.0        1.0  5.0

   Return Direction  Status
0     0.5      Long  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Decreasing position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([4., -1., -1., -1., -1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   4.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              4           4             3.5        4.0  5.0

   Return Direction  Status  Position Id
0    1.25      Long  Closed            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0
1              1       0   1.0               0            0              1.0
2              2       0   1.0               0            0              1.0
3              3       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees   PnL  \\
0        0.25              1           1             2.0        1.0 -0.25
1        0.25              2           2             3.0        1.0  0.75
2        0.25              3           3             4.0        1.0  1.75
3        0.25              4           4             5.0        1.0  2.75

   Return Direction  Status  Position Id
0   -0.25      Long  Closed            0
1    0.75      Long  Closed            0
2    1.75      Long  Closed            0
3    2.75      Long  Closed            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   4.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              4           4             3.5        4.0  5.0

   Return Direction  Status
0    1.25      Long  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Multiple reversing positions:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., -2., 2., -2., 1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0
1               1       0   1.0               1            1              2.0
2               2       0   1.0               2            2              3.0
3               3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status  Position Id
0  -0.500      Long  Closed            0
1  -1.000     Short  Closed            1
2   0.000      Long  Closed            2
3  -0.625     Short  Closed            3

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0
1              1       0   1.0               1            1              2.0
2              2       0   1.0               2            2              3.0
3              3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status  Position Id
0  -0.500      Long  Closed            0
1  -1.000     Short  Closed            1
2   0.000      Long  Closed            2
3  -0.625     Short  Closed            3

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   1.0               0            0              1.0
1            1       0   1.0               1            1              2.0
2            2       0   1.0               2            2              3.0
3            3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status
0  -0.500      Long  Closed
1  -1.000     Short  Closed
2   0.000      Long  Closed
3  -0.625     Short  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Open position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 0., 0., 0., 0.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status  Position Id
0     3.0      Long   Open            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status  Position Id
0     3.0      Long   Open            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status
0     3.0      Long   Open

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Get trade count, trade PnL, and winning trade PnL:

```pycon
>>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
>>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
>>> trades = vbt.Portfolio.from_orders(price, size).trades

>>> trades.count()
6

>>> trades.pnl.sum()
-3.0

>>> trades.winning.count()
2

>>> trades.winning.pnl.sum()
1.5
```

Get count and PnL of trades with duration of more than 2 days:

```pycon
>>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
>>> trades_filtered = trades.apply_mask(mask)
>>> trades_filtered.count()
2

>>> trades_filtered.pnl.sum()
-3.0
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Trades.metrics`.

```pycon
>>> price = vbt.RandomData.fetch(
...     ['a', 'b'],
...     start=datetime(2020, 1, 1),
...     end=datetime(2020, 3, 1),
...     seed=vbt.symbol_dict(a=42, b=43)
... ).get()
```

[=100% "100%"]{: .candystripe}

```pycon
>>> size = pd.DataFrame({
...     'a': np.random.randint(-1, 2, size=len(price.index)),
...     'b': np.random.randint(-1, 2, size=len(price.index)),
... }, index=price.index, columns=price.columns)
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d')

>>> pf.trades['a'].stats()
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                60 days 00:00:00
Overlap Coverage                        49 days 00:00:00
Total Records                                       19.0
Total Long Trades                                    2.0
Total Short Trades                                  17.0
Total Closed Trades                                 18.0
Total Open Trades                                    1.0
Open Trade PnL                                    16.063
Win Rate [%]                                   61.111111
Max Win Streak                                      11.0
Max Loss Streak                                      7.0
Best Trade [%]                                  3.526377
Worst Trade [%]                                -6.543679
Avg Winning Trade [%]                           2.225861
Avg Losing Trade [%]                           -3.601313
Avg Winning Trade Duration    32 days 19:38:10.909090909
Avg Losing Trade Duration                5 days 00:00:00
Profit Factor                                   1.022425
Expectancy                                      0.028157
SQN                                             0.039174
Name: agg_stats, dtype: object
```

Positions share almost identical metrics with trades:

```pycon
>>> pf.positions['a'].stats()
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                        0 days 00:00:00
Total Records                                       5.0
Total Long Trades                                   2.0
Total Short Trades                                  3.0
Total Closed Trades                                 4.0
Total Open Trades                                   1.0
Open Trade PnL                                38.356823
Win Rate [%]                                        0.0
Max Win Streak                                      0.0
Max Loss Streak                                     4.0
Best Trade [%]                                -1.529613
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                               NaN
Avg Losing Trade [%]                          -3.786739
Avg Winning Trade Duration                          NaT
Avg Losing Trade Duration               4 days 00:00:00
Profit Factor                                       0.0
Expectancy                                    -5.446748
SQN                                           -1.794214
Name: agg_stats, dtype: object
```

To also include open trades/positions when calculating metrics such as win rate, pass `incl_open=True`:

```pycon
>>> pf.trades['a'].stats(settings=dict(incl_open=True))
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                       49 days 00:00:00
Total Records                                      19.0
Total Long Trades                                   2.0
Total Short Trades                                 17.0
Total Closed Trades                                18.0
Total Open Trades                                   1.0
Open Trade PnL                                   16.063
Win Rate [%]                                  61.111111
Max Win Streak                                     12.0
Max Loss Streak                                     7.0
Best Trade [%]                                 3.526377
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                          2.238896
Avg Losing Trade [%]                          -3.601313
Avg Winning Trade Duration             33 days 18:00:00
Avg Losing Trade Duration               5 days 00:00:00
Profit Factor                                  1.733143
Expectancy                                     0.872096
SQN                                            0.804714
Name: agg_stats, dtype: object
```

`Trades.stats` also supports (re-)grouping:

```pycon
>>> pf.trades.stats(group_by=True)
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                61 days 00:00:00
Overlap Coverage                        61 days 00:00:00
Total Records                                         37
Total Long Trades                                      5
Total Short Trades                                    32
Total Closed Trades                                   35
Total Open Trades                                      2
Open Trade PnL                                  1.336259
Win Rate [%]                                   37.142857
Max Win Streak                                        11
Max Loss Streak                                       10
Best Trade [%]                                  3.526377
Worst Trade [%]                                -8.710238
Avg Winning Trade [%]                           1.907799
Avg Losing Trade [%]                           -3.259135
Avg Winning Trade Duration    28 days 14:46:09.230769231
Avg Losing Trade Duration               14 days 00:00:00
Profit Factor                                   0.340493
Expectancy                                     -1.292596
SQN                                            -2.509223
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Trades.subplots`.

`Trades` class has two subplots based on `Trades.plot` and `Trades.plot_pnl`:

```pycon
>>> pf.trades['a'].plots().show()
```

![](/assets/images/api/trades_plots.svg){: .iimg loading=lazy }
"""

from functools import partialmethod

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array, to_pd_array, broadcast_to
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.generic.enums import range_dt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import TradeDirection, TradeStatus, trade_dt
from vectorbtpro.portfolio.orders import Orders
from vectorbtpro.records.decorators import attach_fields, override_field_config, attach_shortcut_properties
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.array_ import min_rel_rescale, max_rel_rescale
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.template import Rep, RepEval, RepFunc

__all__ = [
    "Trades",
    "EntryTrades",
    "ExitTrades",
    "Positions",
]

__pdoc__ = {}

# ############# Trades ############# #

trades_field_config = ReadonlyConfig(
    dict(
        dtype=trade_dt,
        settings={
            "id": dict(title="Trade Id"),
            "idx": dict(name="exit_idx"),  # remap field of Records
            "start_idx": dict(name="entry_idx"),  # remap field of Ranges
            "end_idx": dict(name="exit_idx"),  # remap field of Ranges
            "size": dict(title="Size"),
            "entry_order_id": dict(title="Entry Order Id", mapping="ids"),
            "entry_idx": dict(title="Entry Index", mapping="index"),
            "entry_price": dict(title="Avg Entry Price"),
            "entry_fees": dict(title="Entry Fees"),
            "exit_order_id": dict(title="Exit Order Id", mapping="ids"),
            "exit_idx": dict(title="Exit Index", mapping="index"),
            "exit_price": dict(title="Avg Exit Price"),
            "exit_fees": dict(title="Exit Fees"),
            "pnl": dict(title="PnL"),
            "return": dict(title="Return", hovertemplate="$title: %{customdata[$index]:,%}"),
            "direction": dict(title="Direction", mapping=TradeDirection),
            "status": dict(title="Status", mapping=TradeStatus),
            "parent_id": dict(title="Position Id", mapping="ids"),
        },
    )
)
"""_"""

__pdoc__[
    "trades_field_config"
] = f"""Field config for `Trades`.

```python
{trades_field_config.prettify()}
```
"""

trades_attach_field_config = ReadonlyConfig(
    {
        "return": dict(attach="returns"),
        "direction": dict(attach_filters=True),
        "status": dict(attach_filters=True, on_conflict="ignore"),
    }
)
"""_"""

__pdoc__[
    "trades_attach_field_config"
] = f"""Config of fields to be attached to `Trades`.

```python
{trades_attach_field_config.prettify()}
```
"""

trades_shortcut_config = ReadonlyConfig(
    dict(
        ranges=dict(),
        winning=dict(),
        losing=dict(),
        winning_streak=dict(obj_type="mapped_array"),
        losing_streak=dict(obj_type="mapped_array"),
        win_rate=dict(obj_type="red_array"),
        profit_factor=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_profit_factor=dict(
            obj_type="red_array",
            method_name="get_profit_factor",
            method_kwargs=dict(use_returns=True, wrap_kwargs=dict(name_or_index="rel_profit_factor")),
        ),
        expectancy=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_expectancy=dict(
            obj_type="red_array",
            method_name="get_expectancy",
            method_kwargs=dict(use_returns=True, wrap_kwargs=dict(name_or_index="rel_expectancy")),
        ),
        sqn=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_sqn=dict(
            obj_type="red_array",
            method_name="get_sqn",
            method_kwargs=dict(use_returns=True, wrap_kwargs=dict(name_or_index="rel_sqn")),
        ),
        best_price=dict(obj_type="mapped_array"),
        worst_price=dict(obj_type="mapped_array"),
        best_price_idx=dict(obj_type="mapped_array"),
        worst_price_idx=dict(obj_type="mapped_array"),
        expanding_best_price=dict(obj_type="array"),
        expanding_worst_price=dict(obj_type="array"),
        mfe=dict(obj_type="mapped_array"),
        mfe_returns=dict(
            obj_type="mapped_array",
            method_name="get_mfe",
            method_kwargs=dict(use_returns=True),
        ),
        mae=dict(obj_type="mapped_array"),
        mae_returns=dict(
            obj_type="mapped_array",
            method_name="get_mae",
            method_kwargs=dict(use_returns=True),
        ),
        expanding_mfe=dict(obj_type="array"),
        expanding_mfe_returns=dict(
            obj_type="array",
            method_name="get_expanding_mfe",
            method_kwargs=dict(use_returns=True),
        ),
        expanding_mae=dict(obj_type="array"),
        expanding_mae_returns=dict(
            obj_type="array",
            method_name="get_expanding_mae",
            method_kwargs=dict(use_returns=True),
        ),
        edge_ratio=dict(obj_type="red_array"),
        running_edge_ratio=dict(obj_type="array"),
    )
)
"""_"""

__pdoc__[
    "trades_shortcut_config"
] = f"""Config of shortcut properties to be attached to `Trades`.

```python
{trades_shortcut_config.prettify()}
```
"""

TradesT = tp.TypeVar("TradesT", bound="Trades")


@attach_shortcut_properties(trades_shortcut_config)
@attach_fields(trades_attach_field_config)
@override_field_config(trades_field_config)
class Trades(Ranges):
    """Extends `vectorbtpro.generic.ranges.Ranges` for working with trade-like records, such as
    entry trades, exit trades, and positions."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    def get_ranges(self, **kwargs) -> Ranges:
        """Get records of type `vectorbtpro.generic.ranges.Ranges`."""
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("entry_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("exit_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    # ############# Stats ############# #

    def get_winning(self: TradesT, **kwargs) -> TradesT:
        """Get winning trades."""
        filter_mask = self.get_field_arr("pnl") > 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_losing(self: TradesT, **kwargs) -> TradesT:
        """Get losing trades."""
        filter_mask = self.get_field_arr("pnl") < 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_winning_streak(self, **kwargs) -> MappedArray:
        """Get winning streak at each trade in the current column.

        See `vectorbtpro.portfolio.nb.records.trade_winning_streak_nb`."""
        return self.apply(nb.trade_winning_streak_nb, dtype=np.int_, **kwargs)

    def get_losing_streak(self, **kwargs) -> MappedArray:
        """Get losing streak at each trade in the current column.

        See `vectorbtpro.portfolio.nb.records.trade_losing_streak_nb`."""
        return self.apply(nb.trade_losing_streak_nb, dtype=np.int_, **kwargs)

    def get_win_rate(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get rate of winning trades."""
        wrap_kwargs = merge_dicts(dict(name_or_index="win_rate"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.win_rate_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_profit_factor(
        self,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get profit factor."""
        wrap_kwargs = merge_dicts(dict(name_or_index="profit_factor"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.profit_factor_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_expectancy(
        self,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get average profitability."""
        wrap_kwargs = merge_dicts(dict(name_or_index="expectancy"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.expectancy_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_sqn(
        self,
        ddof: int = 1,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get System Quality Number (SQN)."""
        wrap_kwargs = merge_dicts(dict(name_or_index="sqn"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.sqn_reduce_nb,
            ddof,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_best_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        **kwargs,
    ) -> MappedArray:
        """Get best price.

        See `vectorbtpro.portfolio.nb.records.best_price_nb`."""
        return self.apply(
            nb.best_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            **kwargs,
        )

    def get_worst_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        **kwargs,
    ) -> MappedArray:
        """Get worst price.

        See `vectorbtpro.portfolio.nb.records.worst_price_nb`."""
        return self.apply(
            nb.worst_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            **kwargs,
        )

    def get_best_price_idx(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        relative: bool = True,
        **kwargs,
    ) -> MappedArray:
        """Get (relative) index of best price.

        See `vectorbtpro.portfolio.nb.records.best_price_idx_nb`."""
        return self.apply(
            nb.best_price_idx_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            relative,
            dtype=np.int_,
            **kwargs,
        )

    def get_worst_price_idx(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        relative: bool = True,
        **kwargs,
    ) -> MappedArray:
        """Get (relative) index of worst price.

        See `vectorbtpro.portfolio.nb.records.worst_price_idx_nb`."""
        return self.apply(
            nb.worst_price_idx_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            relative,
            dtype=np.int_,
            **kwargs,
        )

    def get_expanding_best_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        index_stack_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get expanding best price.

        See `vectorbtpro.portfolio.nb.records.expanding_best_price_nb`."""
        func = jit_reg.resolve_option(nb.expanding_best_price_nb, jitted)
        out = func(
            self.values,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        new_columns = stack_indexes((
            self.wrapper.columns[self.get_field_arr("col")],
            pd.Index(self.get_field_arr("id"), name="id"),
        ), **index_stack_kwargs)
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(
            out,
            group_by=False,
            index=pd.RangeIndex(stop=len(out)),
            columns=new_columns,
            **wrap_kwargs
        )

    def get_expanding_worst_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        index_stack_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get expanding worst price.

        See `vectorbtpro.portfolio.nb.records.expanding_worst_price_nb`."""
        func = jit_reg.resolve_option(nb.expanding_worst_price_nb, jitted)
        out = func(
            self.values,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        new_columns = stack_indexes((
            self.wrapper.columns[self.get_field_arr("col")],
            pd.Index(self.get_field_arr("id"), name="id"),
        ), **index_stack_kwargs)
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(
            out,
            group_by=False,
            index=pd.RangeIndex(stop=len(out)),
            columns=new_columns,
            **wrap_kwargs
        )

    def get_mfe(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get MFE.

        See `vectorbtpro.portfolio.nb.records.mfe_nb`."""
        best_price = self.resolve_shortcut_attr(
            "best_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mfe_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mfe = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            best_price.values,
            use_returns=use_returns,
        )
        return self.map_array(mfe, **kwargs)

    def get_mae(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get MAE.

        See `vectorbtpro.portfolio.nb.records.mae_nb`."""
        worst_price = self.resolve_shortcut_attr(
            "worst_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mae_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mae = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            worst_price.values,
            use_returns=use_returns,
        )
        return self.map_array(mae, **kwargs)

    def get_expanding_mfe(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Get expanding MFE.

        See `vectorbtpro.portfolio.nb.records.expanding_mfe_nb`."""
        expanding_best_price = self.resolve_shortcut_attr(
            "expanding_best_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            **kwargs,
        )
        func = jit_reg.resolve_option(nb.expanding_mfe_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            expanding_best_price.values,
            use_returns=use_returns,
        )
        return ArrayWrapper.from_obj(expanding_best_price).wrap(out)

    def get_expanding_mae(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Get expanding MAE.

        See `vectorbtpro.portfolio.nb.records.expanding_mae_nb`."""
        expanding_worst_price = self.resolve_shortcut_attr(
            "expanding_worst_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            **kwargs,
        )
        func = jit_reg.resolve_option(nb.expanding_mae_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            expanding_worst_price.values,
            use_returns=use_returns,
        )
        return ArrayWrapper.from_obj(expanding_worst_price).wrap(out)

    def get_edge_ratio(
        self,
        volatility: tp.Optional[tp.ArrayLike] = None,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get edge ratio.

        See `vectorbtpro.portfolio.nb.records.edge_ratio_nb`.

        If `volatility` is None, calculates the 14-period ATR if both high and low are provided,
        otherwise the 14-period rolling standard deviation."""
        if self._close is None:
            raise ValueError("Must provide close")

        if volatility is None:
            if self._high is not None and self._low is not None:
                from vectorbtpro.indicators.nb import atr_nb
                from vectorbtpro.generic.enums import WType

                if self._high is None or self._low is None:
                    raise ValueError("Must provide high and low for ATR calculation")

                volatility = atr_nb(
                    high=to_2d_array(self._high),
                    low=to_2d_array(self._low),
                    close=to_2d_array(self._close),
                    window=14,
                    wtype=WType.Wilder,
                )[1]
            else:
                from vectorbtpro.indicators.nb import msd_nb
                from vectorbtpro.generic.enums import WType

                volatility = msd_nb(
                    close=to_2d_array(self._close),
                    window=14,
                    wtype=WType.Wilder,
                )
        else:
            volatility = broadcast_to(volatility, self.wrapper, to_pd=False, keep_flex=True)
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.edge_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            col_map,
            self._open,
            self._high,
            self._low,
            self._close,
            volatility,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    def get_running_edge_ratio(
        self,
        volatility: tp.Optional[tp.ArrayLike] = None,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        incl_shorter: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get running edge ratio.

        See `vectorbtpro.portfolio.nb.records.running_edge_ratio_nb`.

        If `volatility` is None, calculates the 14-period ATR if both high and low are provided,
        otherwise the 14-period rolling standard deviation."""
        if self._close is None:
            raise ValueError("Must provide close")

        if volatility is None:
            if self._high is not None and self._low is not None:
                from vectorbtpro.indicators.nb import atr_nb
                from vectorbtpro.generic.enums import WType

                if self._high is None or self._low is None:
                    raise ValueError("Must provide high and low for ATR calculation")

                volatility = atr_nb(
                    high=to_2d_array(self._high),
                    low=to_2d_array(self._low),
                    close=to_2d_array(self._close),
                    window=14,
                    wtype=WType.Wilder,
                )[1]
            else:
                from vectorbtpro.indicators.nb import msd_nb
                from vectorbtpro.generic.enums import WType

                volatility = msd_nb(
                    close=to_2d_array(self._close),
                    window=14,
                    wtype=WType.Wilder,
                )
        else:
            volatility = broadcast_to(volatility, self.wrapper, to_pd=False, keep_flex=True)
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.running_edge_ratio_nb, jitted)
        out = func(
            self.values,
            col_map,
            self._open,
            self._high,
            self._low,
            self._close,
            volatility,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            incl_shorter=incl_shorter,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(out, group_by=group_by, index=pd.RangeIndex(stop=len(out)), **wrap_kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.stats`.

        Merges `vectorbtpro.generic.ranges.Ranges.stats_defaults` and
        `stats` from `vectorbtpro._settings.trades`."""
        from vectorbtpro._settings import settings

        trades_stats_cfg = settings["trades"]["stats"]

        return merge_dicts(Ranges.stats_defaults.__get__(self), trades_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(title="Start", calc_func=lambda self: self.wrapper.index[0], agg_func=None, tags="wrapper"),
            end=dict(title="End", calc_func=lambda self: self.wrapper.index[-1], agg_func=None, tags="wrapper"),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            first_trade_start=dict(
                title="First Trade Start",
                calc_func="entry_idx.nth",
                n=0,
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            last_trade_end=dict(
                title="Last Trade End",
                calc_func="exit_idx.nth",
                n=-1,
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            total_long_trades=dict(
                title="Total Long Trades", calc_func="direction_long.count", tags=["trades", "long"]
            ),
            total_short_trades=dict(
                title="Total Short Trades", calc_func="direction_short.count", tags=["trades", "short"]
            ),
            total_closed_trades=dict(
                title="Total Closed Trades", calc_func="status_closed.count", tags=["trades", "closed"]
            ),
            total_open_trades=dict(title="Total Open Trades", calc_func="status_open.count", tags=["trades", "open"]),
            open_trade_pnl=dict(title="Open Trade PnL", calc_func="status_open.pnl.sum", tags=["trades", "open"]),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="status_closed.get_win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            winning_streak=dict(
                title="Max Win Streak",
                calc_func=RepEval("'winning_streak.max' if incl_open else 'status_closed.winning_streak.max'"),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            losing_streak=dict(
                title="Max Loss Streak",
                calc_func=RepEval("'losing_streak.max' if incl_open else 'status_closed.losing_streak.max'"),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func=RepEval("'returns.max' if incl_open else 'status_closed.returns.max'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func=RepEval("'returns.min' if incl_open else 'status_closed.returns.min'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func=RepEval("'winning.returns.mean' if incl_open else 'status_closed.winning.returns.mean'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func=RepEval("'losing.returns.mean' if incl_open else 'status_closed.losing.returns.mean'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func=RepEval("'winning.avg_duration' if incl_open else 'status_closed.winning.get_avg_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func=RepEval("'losing.avg_duration' if incl_open else 'status_closed.losing.get_avg_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func=RepEval("'profit_factor' if incl_open else 'status_closed.get_profit_factor'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func=RepEval("'expectancy' if incl_open else 'status_closed.get_expectancy'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            sqn=dict(
                title="SQN",
                calc_func=RepEval("'sqn' if incl_open else 'status_closed.get_sqn'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            edge_ratio=dict(
                title="Edge Ratio",
                calc_func=RepEval("'edge_ratio' if incl_open else 'status_closed.get_edge_ratio'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_pnl(
        self,
        column: tp.Optional[tp.Label] = None,
        pct_scale: bool = False,
        marker_size_range: tp.Tuple[float, float] = (7, 14),
        opacity_range: tp.Tuple[float, float] = (0.75, 0.9),
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot trade PnL or returns.

        Args:
            column (str): Name of the column to plot.
            pct_scale (bool): Whether to set y-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            marker_size_range (tuple): Range of marker size.
            opacity_range (tuple): Range of marker opacity.
            closed_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed" markers.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], index=index)
            >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_pnl().show()
            ```

            ![](/assets/images/api/trades_plot_pnl.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        marker_size_range = tuple(marker_size_range)
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
            def_layout_kwargs[yaxis]["title"] = "Return"
        else:
            def_layout_kwargs[yaxis]["title"] = "PnL"
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask, name, color, kwargs):
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=exit_idx[mask],
                            y=returns[mask] if pct_scale else pnl[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=marker_size[mask],
                                opacity=opacity[mask],
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(neutral_mask, "Closed", plotting_cfg["contrast_color_schema"]["gray"], closed_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(open_mask, "Open", plotting_cfg["contrast_color_schema"]["orange"], open_trace_kwargs)

        # Plot zeroline
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    plot_returns = partialmethod(plot_pnl, pct_scale=True)
    """`Trades.plot_pnl` for `Trades.returns`."""

    def plot_against_pnl(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Label] = None,
        pct_scale: bool = False,
        field_pct_scale: bool = False,
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        vline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a field against PnL or returns.

        Args:
            field (str, MappedArray, or array_like): Field to be plotted.

                Can be also provided as a mapped array or 1-dim array.
            field_label (str): Label of the field.
            column (str): Name of the column to plot.
            pct_scale (bool): Whether to set x-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            field_pct_scale (bool): Whether to make y-axis a percentage scale.
            closed_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed" markers.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for horizontal zeroline.
            vline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for vertical zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=10)
            >>> price = pd.Series([1., 2., 3., 4., 5., 6., 5., 3., 2., 1.], index=index)
            >>> orders = pd.Series([1., -0.5, 0., -0.5, 2., 0., -0.5, -0.5, 0., -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> trades = pf.trades
            >>> trades.plot_against_pnl("MFE").show()
            ```

            ![](/assets/images/api/trades_plot_against_pnl.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[xaxis]["tickformat"] = ".2%"
            def_layout_kwargs[xaxis]["title"] = "Return"
        else:
            def_layout_kwargs[xaxis]["title"] = "PnL"
        if field_pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
        def_layout_kwargs[yaxis]["title"] = field_label
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask, name, color, kwargs):
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=returns[mask] if pct_scale else pnl[mask],
                            y=field[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=7,
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(neutral_mask, "Closed", plotting_cfg["contrast_color_schema"]["gray"], closed_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(open_mask, "Open", plotting_cfg["contrast_color_schema"]["orange"], open_trace_kwargs)

        # Plot zerolines
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref=xref,
                    yref="paper",
                    x0=0,
                    y0=y_domain[0],
                    x1=0,
                    y1=y_domain[1],
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                vline_shape_kwargs,
            )
        )
        return fig

    plot_mfe = partialmethod(
        plot_against_pnl,
        field="mfe",
        field_label="MFE",
    )
    """`Trades.plot_against_pnl` for `Trades.mfe`."""

    plot_mfe_returns = partialmethod(
        plot_against_pnl,
        field="mfe_returns",
        field_label="MFE Return",
        pct_scale=True,
        field_pct_scale=True,
    )
    """`Trades.plot_against_pnl` for `Trades.mfe_returns`."""

    plot_mae = partialmethod(
        plot_against_pnl,
        field="mae",
        field_label="MAE",
    )
    """`Trades.plot_against_pnl` for `Trades.mae`."""

    plot_mae_returns = partialmethod(
        plot_against_pnl,
        field="mae_returns",
        field_label="MAE Return",
        pct_scale=True,
        field_pct_scale=True,
    )
    """`Trades.plot_against_pnl` for `Trades.mae_returns`."""

    def plot_expanding(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Label] = None,
        plot_bands: bool = False,
        colorize: tp.Union[bool, str, tp.Callable] = "last",
        field_pct_scale: bool = False,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot projections of an expanding field.

        Args:
            field (str or array_like): Field to be plotted.

                 Can be also provided as a 2-dim array.
            field_label (str): Label of the field.
            column (str): Name of the column to plot. Optional.
            plot_bands (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            colorize (bool, str or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            field_pct_scale (bool): Whether to make y-axis a percentage scale.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **kwargs: Keyword arguments passed to `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=10)
            >>> price = pd.Series([1., 2., 3., 2., 4., 5., 6., 5., 6., 7.], index=index)
            >>> orders = pd.Series([1., 0., 0., -2., 0., 0., 2., 0., 0., -1.], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_expanding("MFE").show()
            ```

            ![](/assets/images/api/trades_plot_expanding.svg){: .iimg loading=lazy }
        """
        if column is not None:
            self_col = self.select_col(column=column, group_by=False)
        else:
            self_col = self

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            if not field.lower().startswith("expanding_"):
                field = "expanding_" + field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"
        field = to_pd_array(field)

        fig = field.vbt.plot_projections(
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            plot_bands=plot_bands,
            colorize=colorize,
            **kwargs,
        )
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "yaxis"
        if field_label is not None and "title" not in kwargs.get(yaxis, {}):
            fig.update_layout(**{yaxis: dict(title=field_label)})
        if field_pct_scale and "tickformat" not in kwargs.get(yaxis, {}):
            fig.update_layout(**{yaxis: dict(tickformat=".2%")})
        return fig

    plot_expanding_mfe = partialmethod(
        plot_expanding,
        field="expanding_mfe",
        field_label="MFE",
    )
    """`Trades.plot_expanding` for `Trades.expanding_mfe`."""

    plot_expanding_mfe_returns = partialmethod(
        plot_expanding,
        field="expanding_mfe_returns",
        field_label="MFE Return",
        field_pct_scale=True,
    )
    """`Trades.plot_expanding` for `Trades.expanding_mfe_returns`."""

    plot_expanding_mae = partialmethod(
        plot_expanding,
        field="expanding_mae",
        field_label="MAE",
    )
    """`Trades.plot_expanding` for `Trades.expanding_mae`."""

    plot_expanding_mae_returns = partialmethod(
        plot_expanding,
        field="expanding_mae_returns",
        field_label="MAE Return",
        field_pct_scale=True,
    )
    """`Trades.plot_expanding` for `Trades.expanding_mae_returns`."""

    def plot_against_pnl(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Label] = None,
        pct_scale: bool = False,
        field_pct_scale: bool = False,
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        vline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a field against PnL or returns.

        Args:
            field (str, MappedArray, or array_like): Field to be plotted.
            field_label (str): Label of the field.
            column (str): Name of the column to plot.
            pct_scale (bool): Whether to set x-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            field_pct_scale (bool): Whether to make y-axis a percentage scale.
            closed_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed" markers.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for horizontal zeroline.
            vline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for vertical zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=10)
            >>> price = pd.Series([1., 2., 3., 4., 5., 6., 5., 3., 2., 1.], index=index)
            >>> orders = pd.Series([1., -0.5, 0., -0.5, 2., 0., -0.5, -0.5, 0., -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> trades = pf.trades
            >>> trades.plot_against_pnl("MFE").show()
            ```

            ![](/assets/images/api/trades_plot_against_pnl.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[xaxis]["tickformat"] = ".2%"
            def_layout_kwargs[xaxis]["title"] = "Return"
        else:
            def_layout_kwargs[xaxis]["title"] = "PnL"
        if field_pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
        def_layout_kwargs[yaxis]["title"] = field_label
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask, name, color, kwargs):
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=returns[mask] if pct_scale else pnl[mask],
                            y=field[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=7,
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(neutral_mask, "Closed", plotting_cfg["contrast_color_schema"]["gray"], closed_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(open_mask, "Open", plotting_cfg["contrast_color_schema"]["orange"], open_trace_kwargs)

        # Plot zerolines
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref=xref,
                    yref="paper",
                    x0=0,
                    y0=y_domain[0],
                    x1=0,
                    y1=y_domain[1],
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                vline_shape_kwargs,
            )
        )
        return fig

    def plot_running_edge_ratio(
        self,
        column: tp.Optional[tp.Label] = None,
        volatility: tp.Optional[tp.ArrayLike] = None,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        incl_shorter: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of edge ratio.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain

        running_edge_ratio = self.resolve_shortcut_attr(
            "running_edge_ratio",
            volatility=volatility,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            incl_shorter=incl_shorter,
            group_by=group_by,
            jitted=jitted,
        )
        running_edge_ratio = self.select_col_from_obj(
            running_edge_ratio, column, wrapper=self.wrapper.regroup(group_by)
        )
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(name="Edge Ratio"),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = running_edge_ratio.vbt.plot_against(1, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=1.0,
                    x1=x_domain[1],
                    y1=1.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        plot_by_type: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        exit_profit_trace_kwargs: tp.KwargsLike = None,
        exit_loss_trace_kwargs: tp.KwargsLike = None,
        active_trace_kwargs: tp.KwargsLike = None,
        profit_shape_kwargs: tp.KwargsLike = None,
        loss_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot trades.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            plot_markers (bool): Whether to plot markers.
            plot_zones (bool): Whether to plot zones.
            plot_by_type (bool): Whether to plot exit trades by type.

                Otherwise, the appearance will be controlled using `exit_trace_kwargs`.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Trades.close`.
            entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Entry" markers.
            exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit" markers.
            exit_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Profit" markers.
            exit_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Loss" markers.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Active" markers.
            profit_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for profit zones.
            loss_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for loss zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import pandas as pd
            >>> import vectorbtpro as vbt

            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], index=index)
            >>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, size)
            >>> pf.trades.plot().show()
            ```

            ![](/assets/images/api/trades_plot.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if entry_trace_kwargs is None:
            entry_trace_kwargs = {}
        if exit_trace_kwargs is None:
            exit_trace_kwargs = {}
        if exit_profit_trace_kwargs is None:
            exit_profit_trace_kwargs = {}
        if exit_loss_trace_kwargs is None:
            exit_loss_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if profit_shape_kwargs is None:
            profit_shape_kwargs = {}
        if loss_shape_kwargs is None:
            loss_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            entry_idx = self_col.get_map_field_to_index("entry_idx", minus_one_to_zero=True)
            entry_price = self_col.get_field_arr("entry_price")
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            exit_price = self_col.get_field_arr("exit_price")
            pnl = self_col.get_field_arr("pnl")
            status = self_col.get_field_arr("status")

            duration = to_1d_array(
                self_col.wrapper.arr_to_timedelta(self_col.duration.values, to_pd=True, silence_warnings=True).astype(
                    str
                )
            )

            if plot_markers:
                # Plot Entry markers
                if self_col.get_field_setting("parent_id", "ignore", False):
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "direction",
                        ]
                    )
                else:
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "parent_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "direction",
                        ]
                    )
                _entry_trace_kwargs = merge_dicts(
                    dict(
                        x=entry_idx,
                        y=entry_price,
                        mode="markers",
                        marker=dict(
                            symbol="square",
                            color=plotting_cfg["contrast_color_schema"]["blue"],
                            size=7,
                            line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["blue"])),
                        ),
                        name="Entry",
                        customdata=entry_customdata,
                        hovertemplate=entry_hovertemplate,
                    ),
                    entry_trace_kwargs,
                )
                entry_scatter = go.Scatter(**_entry_trace_kwargs)
                fig.add_trace(entry_scatter, **add_trace_kwargs)

                # Plot end markers
                def _plot_end_markers(mask, name, color, kwargs, incl_status=False) -> None:
                    if np.any(mask):
                        if self_col.get_field_setting("parent_id", "ignore", False):
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "exit_order_id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                    *(("status",) if incl_status else ()),
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        else:
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "exit_order_id",
                                    "parent_id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                    *(("status",) if incl_status else ()),
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        _kwargs = merge_dicts(
                            dict(
                                x=exit_idx[mask],
                                y=exit_price[mask],
                                mode="markers",
                                marker=dict(
                                    symbol="square",
                                    color=color,
                                    size=7,
                                    line=dict(width=1, color=adjust_lightness(color)),
                                ),
                                name=name,
                                customdata=exit_customdata,
                                hovertemplate=exit_hovertemplate,
                            ),
                            kwargs,
                        )
                        scatter = go.Scatter(**_kwargs)
                        fig.add_trace(scatter, **add_trace_kwargs)

                if plot_by_type:
                    # Plot Exit markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl == 0.0),
                        "Exit",
                        plotting_cfg["contrast_color_schema"]["gray"],
                        exit_trace_kwargs,
                    )

                    # Plot Exit - Profit markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl > 0.0),
                        "Exit - Profit",
                        plotting_cfg["contrast_color_schema"]["green"],
                        exit_profit_trace_kwargs,
                    )

                    # Plot Exit - Loss markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl < 0.0),
                        "Exit - Loss",
                        plotting_cfg["contrast_color_schema"]["red"],
                        exit_loss_trace_kwargs,
                    )

                    # Plot Active markers
                    _plot_end_markers(
                        status == TradeStatus.Open,
                        "Active",
                        plotting_cfg["contrast_color_schema"]["orange"],
                        active_trace_kwargs,
                    )
                else:
                    # Plot Exit markers
                    _plot_end_markers(
                        np.full(len(status), True),
                        "Exit",
                        plotting_cfg["contrast_color_schema"]["pink"],
                        exit_trace_kwargs,
                        incl_status=True,
                    )

            if plot_zones:
                # Plot profit zones
                self_col.winning.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    shape_kwargs=merge_dicts(
                        dict(
                            yref=Rep("yref"),
                            y0=RepFunc(lambda record: record["entry_price"]),
                            y1=RepFunc(lambda record: record["exit_price"]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["green"],
                        ),
                        profit_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot loss zones
                self_col.losing.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    shape_kwargs=merge_dicts(
                        dict(
                            yref=Rep("yref"),
                            y0=RepFunc(lambda record: record["entry_price"]),
                            y1=RepFunc(lambda record: record["exit_price"]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["red"],
                        ),
                        loss_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.plots`.

        Merges `vectorbtpro.generic.ranges.Ranges.plots_defaults` and
        `plots` from `vectorbtpro._settings.trades`."""
        from vectorbtpro._settings import settings

        trades_plots_cfg = settings["trades"]["plots"]

        return merge_dicts(Ranges.plots_defaults.__get__(self), trades_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot",
                tags="trades",
            ),
            plot_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func="plot_pnl",
                tags="trades",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Trades.override_field_config_doc(__pdoc__)
Trades.override_metrics_doc(__pdoc__)
Trades.override_subplots_doc(__pdoc__)

# ############# EntryTrades ############# #

entry_trades_field_config = ReadonlyConfig(
    dict(settings={"id": dict(title="Entry Trade Id"), "idx": dict(name="entry_idx")})  # remap field of Records,
)
"""_"""

__pdoc__[
    "entry_trades_field_config"
] = f"""Field config for `EntryTrades`.

```python
{entry_trades_field_config.prettify()}
```
"""

EntryTradesT = tp.TypeVar("EntryTradesT", bound="EntryTrades")


@override_field_config(entry_trades_field_config)
class EntryTrades(Trades):
    """Extends `Trades` for working with entry trade records."""

    @classmethod
    def from_orders(
        cls: tp.Type[EntryTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> EntryTradesT:
        """Build `EntryTrades` from `vectorbtpro.portfolio.orders.Orders`."""
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_entry_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

    def plot_signals(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        long_entry_trace_kwargs: tp.KwargsLike = None,
        short_entry_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot entry trade signals.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `EntryTrades.close`.
            long_entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Long Entry" markers.
            short_entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Short Entry" markers.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1, 2, 3, 2, 3, 4, 3], index=index)
            >>> orders = pd.Series([1, 0, -1, 0, -1, 2, -2], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.entry_trades.plot_signals().show()
            ```

            ![](/assets/images/api/entry_trades_plot_signals.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if long_entry_trace_kwargs is None:
            long_entry_trace_kwargs = {}
        if short_entry_trace_kwargs is None:
            short_entry_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            entry_idx = self_col.get_map_field_to_index("entry_idx", minus_one_to_zero=True)
            entry_price = self_col.get_field_arr("entry_price")
            direction = self_col.get_field_arr("direction")

            def _plot_entry_markers(mask, name, color, kwargs):
                if np.any(mask):
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "parent_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "pnl",
                            "return",
                            "status",
                        ],
                        mask=mask,
                    )
                    _kwargs = merge_dicts(
                        dict(
                            x=entry_idx[mask],
                            y=entry_price[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color="rgba(0, 0, 0, 0)",
                                size=15,
                                line=dict(
                                    color=color,
                                    width=2,
                                ),
                            ),
                            name=name,
                            customdata=entry_customdata,
                            hovertemplate=entry_hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Long Entry markers
            _plot_entry_markers(
                direction == TradeDirection.Long,
                "Long Entry",
                plotting_cfg["contrast_color_schema"]["green"],
                long_entry_trace_kwargs,
            )

            # Plot Short Entry markers
            _plot_entry_markers(
                direction == TradeDirection.Short,
                "Short Entry",
                plotting_cfg["contrast_color_schema"]["red"],
                short_entry_trace_kwargs,
            )

        return fig


# ############# ExitTrades ############# #

exit_trades_field_config = ReadonlyConfig(dict(settings={"id": dict(title="Exit Trade Id")}))
"""_"""

__pdoc__[
    "exit_trades_field_config"
] = f"""Field config for `ExitTrades`.

```python
{exit_trades_field_config.prettify()}
```
"""

ExitTradesT = tp.TypeVar("ExitTradesT", bound="ExitTrades")


@override_field_config(exit_trades_field_config)
class ExitTrades(Trades):
    """Extends `Trades` for working with exit trade records."""

    @classmethod
    def from_orders(
        cls: tp.Type[ExitTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> ExitTradesT:
        """Build `ExitTrades` from `vectorbtpro.portfolio.orders.Orders`."""
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_exit_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

    def plot_signals(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        long_exit_trace_kwargs: tp.KwargsLike = None,
        short_exit_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot exit trade signals.

        Args:
            column (str): Name of the column to plot.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ExitTrades.close`.
            long_exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Long Exit" markers.
            short_exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Short Exit" markers.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1, 2, 3, 2, 3, 4, 3], index=index)
            >>> orders = pd.Series([1, 0, -1, 0, -1, 2, -2], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.exit_trades.plot_signals().show()
            ```

            ![](/assets/images/api/exit_trades_plot_signals.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if long_exit_trace_kwargs is None:
            long_exit_trace_kwargs = {}
        if short_exit_trace_kwargs is None:
            short_exit_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            exit_idx = self_col.get_map_field_to_index("exit_idx", minus_one_to_zero=True)
            exit_price = self_col.get_field_arr("exit_price")
            direction = self_col.get_field_arr("direction")

            def _plot_exit_markers(mask, name, color, kwargs):
                if np.any(mask):
                    exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "exit_order_id",
                            "parent_id",
                            "exit_idx",
                            "size",
                            "exit_price",
                            "exit_fees",
                            "pnl",
                            "return",
                            "status",
                        ],
                        mask=mask,
                    )
                    _kwargs = merge_dicts(
                        dict(
                            x=exit_idx[mask],
                            y=exit_price[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=8,
                            ),
                            name=name,
                            customdata=exit_customdata,
                            hovertemplate=exit_hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Long Exit markers
            _plot_exit_markers(
                direction == TradeDirection.Long,
                "Long Exit",
                plotting_cfg["contrast_color_schema"]["green"],
                long_exit_trace_kwargs,
            )

            # Plot Short Exit markers
            _plot_exit_markers(
                direction == TradeDirection.Short,
                "Short Exit",
                plotting_cfg["contrast_color_schema"]["red"],
                short_exit_trace_kwargs,
            )

        return fig


# ############# Positions ############# #

positions_field_config = ReadonlyConfig(
    dict(settings={"id": dict(title="Position Id"), "parent_id": dict(title="Parent Id", ignore=True)}),
)
"""_"""

__pdoc__[
    "positions_field_config"
] = f"""Field config for `Positions`.

```python
{positions_field_config.prettify()}
```
"""

PositionsT = tp.TypeVar("PositionsT", bound="Positions")


@override_field_config(positions_field_config)
class Positions(Trades):
    """Extends `Trades` for working with position records."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_trades(
        cls: tp.Type[PositionsT],
        trades: Trades,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PositionsT:
        """Build `Positions` from `Trades`."""
        if open is None:
            open = trades._open
        if high is None:
            high = trades._high
        if low is None:
            low = trades._low
        if close is None:
            close = trades._close
        func = jit_reg.resolve_option(nb.get_positions_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        position_records_arr = func(trades.values, trades.col_mapper.col_map)
        return cls.from_records(
            trades.wrapper,
            position_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )
