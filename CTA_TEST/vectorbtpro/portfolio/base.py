# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for simulating a portfolio and measuring its performance."""

import string
import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import (
    to_1d_array,
    to_2d_array,
    broadcast_array_to,
    broadcast_to,
    to_pd_array,
    to_2d_shape,
)
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.grouping.base import ExceptLevel
from vectorbtpro.data.base import Data
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.portfolio import nb, enums
from vectorbtpro.portfolio.decorators import attach_shortcut_properties, attach_returns_acc_methods
from vectorbtpro.portfolio.logs import Logs
from vectorbtpro.portfolio.orders import Orders
from vectorbtpro.portfolio.trades import Trades, EntryTrades, ExitTrades, Positions
from vectorbtpro.portfolio.pfopt.base import PortfolioOptimizer
from vectorbtpro.portfolio.preparing import (
    PFPrepResult,
    BasePFPreparer,
    FOPreparer,
    FSPreparer,
    FOFPreparer,
    FDOFPreparer,
)
from vectorbtpro.records.base import Records
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.signals.generators import RANDNX, RPROBNX
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig, atomic_dict
from vectorbtpro.utils.decorators import custom_property, cached_property, class_or_instancemethod
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.template import Rep, RepEval, RepFunc

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from vectorbtpro.returns.qs_adapter import QSAdapter as QSAdapterT
except ImportError:
    QSAdapterT = tp.Any

__all__ = [
    "Portfolio",
    "PF",
]

__pdoc__ = {}


def fix_wrapper_for_records(pf: "Portfolio") -> ArrayWrapper:
    """Allow flags for records that were restricted for portfolio."""
    if pf.cash_sharing:
        return pf.wrapper.replace(allow_enable=True, allow_modify=True)
    return pf.wrapper


def records_indexing_func(
    self: "Portfolio",
    obj: tp.RecordArray,
    wrapper_meta: dict,
    cls: tp.Union[type, str],
    groups_only: bool = False,
    **kwargs,
) -> tp.RecordArray:
    """Apply indexing function on records."""
    wrapper = fix_wrapper_for_records(self)
    if groups_only:
        wrapper = wrapper.resolve()
        wrapper_meta = dict(wrapper_meta)
        wrapper_meta["col_idxs"] = wrapper_meta["group_idxs"]
    if isinstance(cls, str):
        cls = getattr(self, cls)
    records = cls(wrapper, obj)
    records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
    return records.indexing_func(records_meta=records_meta).values


def records_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    cls: tp.Union[type, str],
    **kwargs,
) -> tp.RecordArray:
    """Apply resampling function on records."""
    if isinstance(cls, str):
        cls = getattr(self, cls)
    return cls(wrapper, obj).resample(resampler).values


def returns_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    fill_with_zero: bool = True,
    log_returns: bool = False,
    **kwargs,
):
    """Apply resampling function on returns."""
    return (
        pd.DataFrame(obj, index=wrapper.index)
        .vbt.returns(log_returns=log_returns)
        .resample(
            resampler,
            fill_with_zero=fill_with_zero,
        )
        .obj.values
    )


returns_acc_config = ReadonlyConfig(
    {
        "daily_returns": dict(source_name="daily"),
        "annual_returns": dict(source_name="annual"),
        "cumulative_returns": dict(source_name="cumulative"),
        "annualized_return": dict(source_name="annualized"),
        "annualized_volatility": dict(),
        "calmar_ratio": dict(),
        "omega_ratio": dict(),
        "sharpe_ratio": dict(),
        "sharpe_ratio_std": dict(),
        "prob_sharpe_ratio": dict(),
        "deflated_sharpe_ratio": dict(),
        "downside_risk": dict(),
        "sortino_ratio": dict(),
        "information_ratio": dict(),
        "beta": dict(),
        "alpha": dict(),
        "tail_ratio": dict(),
        "value_at_risk": dict(),
        "cond_value_at_risk": dict(),
        "capture_ratio": dict(),
        "up_capture_ratio": dict(),
        "down_capture_ratio": dict(),
        "drawdown": dict(),
        "max_drawdown": dict(),
    }
)
"""_"""

__pdoc__[
    "returns_acc_config"
] = f"""Config of returns accessor methods to be attached to `Portfolio`.

```python
{returns_acc_config.prettify()}
```
"""

shortcut_config = ReadonlyConfig(
    {
        "filled_close": dict(group_by_aware=False, decorator=cached_property),
        "filled_bm_close": dict(group_by_aware=False, decorator=cached_property),
        "orders": dict(
            obj_type="records",
            field_aliases=("order_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.orders_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="orders_cls"),
            resample_func=partial(records_resample_func, cls="orders_cls"),
        ),
        "logs": dict(
            obj_type="records",
            field_aliases=("log_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.logs_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="logs_cls"),
            resample_func=partial(records_resample_func, cls="logs_cls"),
        ),
        "entry_trades": dict(
            obj_type="records",
            field_aliases=("entry_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.entry_trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="entry_trades_cls"),
            resample_func=partial(records_resample_func, cls="entry_trades_cls"),
        ),
        "exit_trades": dict(
            obj_type="records",
            field_aliases=("exit_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.exit_trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="exit_trades_cls"),
            resample_func=partial(records_resample_func, cls="exit_trades_cls"),
        ),
        "trades": dict(
            obj_type="records",
            field_aliases=("trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.trades_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="trades_cls"),
            resample_func=partial(records_resample_func, cls="trades_cls"),
        ),
        "trade_history": dict(),
        "positions": dict(
            obj_type="records",
            field_aliases=("position_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.positions_cls.from_records(
                pf.orders.wrapper,
                obj,
                open=pf._open,
                high=pf._high,
                low=pf._low,
                close=pf._close,
            ),
            indexing_func=partial(records_indexing_func, cls="positions_cls"),
            resample_func=partial(records_resample_func, cls="positions_cls"),
        ),
        "drawdowns": dict(
            obj_type="records",
            field_aliases=("drawdown_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.drawdowns_cls.from_records(pf.orders.wrapper.resolve(), obj),
            indexing_func=partial(records_indexing_func, cls="drawdowns_cls", groups_only=True),
            resample_func=partial(records_resample_func, cls="drawdowns_cls"),
        ),
        "init_position": dict(obj_type="red_array", group_by_aware=False),
        "asset_flow": dict(
            group_by_aware=False,
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "long_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "short_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "assets": dict(group_by_aware=False),
        "long_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
        ),
        "short_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
        ),
        "position_mask": dict(),
        "long_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="longonly")),
        "short_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="shortonly")),
        "position_coverage": dict(obj_type="red_array"),
        "long_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="longonly"),
        ),
        "short_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="shortonly"),
        ),
        "init_cash": dict(obj_type="red_array"),
        "cash_deposits": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "cash_earnings": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "cash_flow": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "free_cash_flow": dict(
            method_name="get_cash_flow",
            method_kwargs=dict(free=True),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "cash": dict(),
        "position": dict(method_name="get_assets", group_by_aware=False),
        "debt": dict(method_name=None, group_by_aware=False),
        "locked_cash": dict(method_name=None, group_by_aware=False),
        "free_cash": dict(method_name="get_cash", method_kwargs=dict(free=True)),
        "init_price": dict(obj_type="red_array", group_by_aware=False),
        "init_position_value": dict(obj_type="red_array", group_by_aware=False),
        "init_value": dict(obj_type="red_array"),
        "input_value": dict(obj_type="red_array"),
        "asset_value": dict(),
        "long_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="longonly")),
        "short_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="shortonly")),
        "gross_exposure": dict(),
        "long_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="longonly")),
        "short_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="shortonly")),
        "net_exposure": dict(),
        "value": dict(),
        "allocations": dict(group_by_aware=False),
        "long_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="longonly"),
            group_by_aware=False,
        ),
        "short_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="shortonly"),
            group_by_aware=False,
        ),
        "total_profit": dict(obj_type="red_array"),
        "final_value": dict(obj_type="red_array"),
        "total_return": dict(obj_type="red_array"),
        "returns": dict(resample_func=returns_resample_func),
        "log_returns": dict(
            method_name="get_returns",
            method_kwargs=dict(log_returns=True),
            resample_func=partial(returns_resample_func, log_returns=True),
        ),
        "daily_log_returns": dict(
            method_name="get_returns",
            method_kwargs=dict(daily_returns=True, log_returns=True),
            resample_func=partial(returns_resample_func, log_returns=True),
        ),
        "asset_pnl": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "asset_returns": dict(resample_func=returns_resample_func),
        "market_value": dict(),
        "market_returns": dict(resample_func=returns_resample_func),
        "bm_value": dict(),
        "bm_returns": dict(resample_func=returns_resample_func),
        "total_market_return": dict(obj_type="red_array"),
        "daily_returns": dict(resample_func=returns_resample_func),
        "annual_returns": dict(resample_func=returns_resample_func),
        "cumulative_returns": dict(),
        "annualized_return": dict(obj_type="red_array"),
        "annualized_volatility": dict(obj_type="red_array"),
        "calmar_ratio": dict(obj_type="red_array"),
        "omega_ratio": dict(obj_type="red_array"),
        "sharpe_ratio": dict(obj_type="red_array"),
        "sharpe_ratio_std": dict(obj_type="red_array"),
        "prob_sharpe_ratio": dict(obj_type="red_array"),
        "deflated_sharpe_ratio": dict(obj_type="red_array"),
        "downside_risk": dict(obj_type="red_array"),
        "sortino_ratio": dict(obj_type="red_array"),
        "information_ratio": dict(obj_type="red_array"),
        "beta": dict(obj_type="red_array"),
        "alpha": dict(obj_type="red_array"),
        "tail_ratio": dict(obj_type="red_array"),
        "value_at_risk": dict(obj_type="red_array"),
        "cond_value_at_risk": dict(obj_type="red_array"),
        "capture_ratio": dict(obj_type="red_array"),
        "up_capture_ratio": dict(obj_type="red_array"),
        "down_capture_ratio": dict(obj_type="red_array"),
        "drawdown": dict(),
        "max_drawdown": dict(obj_type="red_array"),
    }
)
"""_"""

__pdoc__[
    "shortcut_config"
] = f"""Config of shortcut properties to be attached to `Portfolio`.

```python
{shortcut_config.prettify()}
```
"""

PortfolioT = tp.TypeVar("PortfolioT", bound="Portfolio")
PortfolioResultT = tp.Union[PortfolioT, BasePFPreparer, PFPrepResult, enums.SimulationOutput]


class MetaInOutputs(type):
    """Meta class that exposes a read-only class property `MetaFields.in_output_config`."""

    @property
    def in_output_config(cls) -> Config:
        """In-output config."""
        return cls._in_output_config


class PortfolioWithInOutputs(metaclass=MetaInOutputs):
    """Class exposes a read-only class property `RecordsWithFields.field_config`."""

    @property
    def in_output_config(self) -> Config:
        """In-output config of `${cls_name}`.

        ```python
        ${in_output_config}
        ```
        """
        return self._in_output_config


class MetaPortfolio(type(Analyzable), type(PortfolioWithInOutputs)):
    pass


@attach_shortcut_properties(shortcut_config)
@attach_returns_acc_methods(returns_acc_config)
class Portfolio(Analyzable, PortfolioWithInOutputs, metaclass=MetaPortfolio):
    """Class for simulating a portfolio and measuring its performance.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        close (array_like): Last asset price at each time step.
        order_records (array_like): A structured NumPy array of order records.
        open (array_like): Open price of each bar.
        high (array_like): High price of each bar.
        low (array_like): Low price of each bar.
        log_records (array_like): A structured NumPy array of log records.
        cash_sharing (bool): Whether to share cash within the same group.
        init_cash (InitCashMode or array_like of float): Initial capital.

            Can be provided in a format suitable for flexible indexing.
        init_position (array_like of float): Initial position.

            Can be provided in a format suitable for flexible indexing.
        init_price (array_like of float): Initial position price.

            Can be provided in a format suitable for flexible indexing.
        cash_deposits (array_like of float): Cash deposited/withdrawn at each timestamp.

            Can be provided in a format suitable for flexible indexing.
        cash_earnings (array_like of float): Earnings added at each timestamp.

            Can be provided in a format suitable for flexible indexing.
        call_seq (array_like of int): Sequence of calls per row and group. Defaults to None.
        in_outputs (namedtuple): Named tuple with in-output objects.

            To substitute `Portfolio` attributes, provide already broadcasted and grouped objects.
            Also see `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.
        use_in_outputs (bool): Whether to return in-output objects when calling properties.
        bm_close (array_like): Last benchmark asset price at each time step.
        fillna_close (bool): Whether to forward and backward fill NaN values in `close`.

            Applied after the simulation to avoid NaNs in asset value.

            See `Portfolio.get_filled_close`.
        trades_type (str or int): Default `vectorbtpro.portfolio.trades.Trades` to use across `Portfolio`.

            See `vectorbtpro.portfolio.enums.TradesType`.
        orders_cls (type): Class for wrapping order records.
        logs_cls (type): Class for wrapping log records.
        trades_cls (type): Class for wrapping trade records.
        entry_trades_cls (type): Class for wrapping entry trade records.
        exit_trades_cls (type): Class for wrapping exit trade records.
        positions_cls (type): Class for wrapping position records.
        drawdowns_cls (type): Class for wrapping drawdown records.

    For defaults, see `vectorbtpro._settings.portfolio`.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Portfolio.replace`."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_in_output_config"}

    @classmethod
    def row_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        row_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack (two-dimensional) objects along rows.

        `row_stack_func` must take the portfolio class, and all the arguments passed to this method.
        If you don't need any of the arguments, make `row_stack_func` accept them as `**kwargs`.

        If all the objects are None, boolean, or empty, returns the first one."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if row_stack_func is not None:
            return row_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(objs[0]):
            n_cols = wrapper.get_shape_2d(group_by=obj_group_by)[1]
            can_stack = (objs[0].ndim == 1 and n_cols == 1) or (objs[0].ndim == 2 and objs[0].shape[1] == n_cols)
        elif obj_type is not None and obj_type == "array":
            can_stack = True
        else:
            can_stack = False
        if can_stack:
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.row_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along rows")

    @classmethod
    def row_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` along rows.

        All in-output tuples must be either None or have the same fields.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs stacking on the in-output objects of the same field using `Portfolio.row_stack_objs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                row_stack_func = prop_options.get("row_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                row_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    row_stack_func=field_options.get("row_stack_func", row_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.row_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @classmethod
    def row_stack(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        combine_init_cash: bool = False,
        combine_init_position: bool = False,
        combine_init_price: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        Cash sharing must be the same among all objects.

        Close, benchmark close, cash deposits, cash earnings, call sequence, and other two-dimensional arrays
        are stacked using `vectorbtpro.base.wrapping.ArrayWrapper.row_stack_arrs`. In-outputs
        are stacked using `Portfolio.row_stack_in_outputs`. Records are stacked using
        `vectorbtpro.records.base.Records.row_stack_records_arrs`.

        If the initial cash of each object is one of the options in `vectorbtpro.portfolio.enums.InitCashMode`,
        it will be retained for the resulting object. Once any of the objects has the initial cash listed
        as an absolute amount or an array, the initial cash of the first object will be copied over to
        the final object, while the initial cash of all other objects will be resolved and used
        as cash deposits, unless they all are zero. Set `combine_init_cash` to True to simply sum all
        initial cash arrays.

        If only the first object has an initial position greater than zero, it will be copied over to
        the final object. Otherwise, an error will be thrown, unless `combine_init_position` is enabled
        to sum all initial position arrays. The same goes for the initial price, which becomes
        a candidate for stacking only if any of the arrays are not NaN.

        !!! note
            When possible, avoid using initial position and price in objects to be stacked:
            there is currently no way of injecting them in the correct order, while simply taking
            the sum or weighted average may distort the reality since they weren't available
            prior to the actual simulation."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        for i in range(1, len(objs)):
            if objs[i].cash_sharing != objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        kwargs["cash_sharing"] = objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False
        cs_n_cols = kwargs["wrapper"].get_shape_2d(group_by=cs_group_by)[1]
        n_cols = kwargs["wrapper"].shape_2d[1]

        if "close" not in kwargs:
            kwargs["close"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.close for obj in objs],
                group_by=False,
                wrap=False,
            )
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.open for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.high for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.low for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.row_stack_records_arrs(*[obj.orders for obj in objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.row_stack_records_arrs(*[obj.logs for obj in objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in enums.InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                stack_init_cash_objs = False
                init_cash_objs = []
                for i, obj in enumerate(objs):
                    init_cash_obj = obj.get_init_cash(group_by=cs_group_by)
                    init_cash_obj = to_1d_array(init_cash_obj)
                    init_cash_obj = broadcast_array_to(init_cash_obj, cs_n_cols)
                    if i > 0 and (init_cash_obj != 0).any():
                        stack_init_cash_objs = True
                    init_cash_objs.append(init_cash_obj)
                if stack_init_cash_objs:
                    if not combine_init_cash:
                        cash_deposits_objs = []
                        for i, obj in enumerate(objs):
                            cash_deposits_obj = obj.get_cash_deposits(group_by=cs_group_by)
                            cash_deposits_obj = to_2d_array(cash_deposits_obj)
                            cash_deposits_obj = broadcast_array_to(
                                cash_deposits_obj,
                                (cash_deposits_obj.shape[0], cs_n_cols),
                            )
                            cash_deposits_obj = cash_deposits_obj.copy()
                            if i > 0:
                                cash_deposits_obj[0] = init_cash_objs[i]
                            cash_deposits_objs.append(cash_deposits_obj)
                        kwargs["cash_deposits"] = np.row_stack(cash_deposits_objs)
                        kwargs["init_cash"] = init_cash_objs[0]
                    else:
                        kwargs["init_cash"] = np.asarray(init_cash_objs).sum(axis=0)
                else:
                    kwargs["init_cash"] = init_cash_objs[0]
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            init_position_objs = []
            for i, obj in enumerate(objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = broadcast_array_to(init_position_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any():
                    stack_init_position_objs = True
                init_position_objs.append(init_position_obj)
            if stack_init_position_objs:
                if not combine_init_position:
                    raise ValueError("Initial position cannot be stacked along rows")
                kwargs["init_position"] = np.asarray(init_position_objs).sum(axis=0)
            else:
                kwargs["init_position"] = init_position_objs[0]
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            init_position_objs = []
            init_price_objs = []
            for i, obj in enumerate(objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = broadcast_array_to(init_position_obj, n_cols)
                init_price_obj = obj.get_init_price()
                init_price_obj = to_1d_array(init_price_obj)
                init_price_obj = broadcast_array_to(init_price_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any() and not np.isnan(init_price_obj).all():
                    stack_init_price_objs = True
                init_position_objs.append(init_position_obj)
                init_price_objs.append(init_price_obj)
            if stack_init_price_objs:
                if not combine_init_price:
                    raise ValueError("Initial price cannot be stacked along rows")
                init_position_objs = np.asarray(init_position_objs)
                init_price_objs = np.asarray(init_price_objs)
                mask1 = (init_position_objs != 0).any(axis=1)
                mask2 = (~np.isnan(init_price_objs)).any(axis=1)
                mask = mask1 & mask2
                init_position_objs = init_position_objs[mask]
                init_price_objs = init_price_objs[mask]
                nom = (init_position_objs * init_price_objs).sum(axis=0)
                denum = init_position_objs.sum(axis=0)
                kwargs["init_price"] = nom / denum
            else:
                kwargs["init_price"] = init_price_objs[0]
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in objs],
                    group_by=cs_group_by,
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in objs],
                    group_by=False,
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.call_seq for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                kwargs["bm_close"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.bm_close for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.row_stack_in_outputs(*objs, **kwargs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        column_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack (one and two-dimensional) objects along column.

        `column_stack_func` must take the portfolio class, and all the arguments passed to this method.
        If you don't need any of the arguments, make `column_stack_func` accept them as `**kwargs`.

        If all the objects are None, boolean, or empty, returns the first one."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if column_stack_func is not None:
            return column_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(obj):
            n_cols = wrapper.get_shape_2d(group_by=obj_group_by)[1]
            if to_2d_shape(objs[0].shape) == wrappers[0].get_shape_2d(group_by=obj_group_by):
                can_stack = True
                reduced = False
            elif objs[0].shape == (wrappers[0].get_shape_2d(group_by=obj_group_by)[1],):
                can_stack = True
                reduced = True
            else:
                can_stack = False
        elif obj_type is not None and obj_type == "array":
            can_stack = True
            reduced = False
        elif obj_type is not None and obj_type == "red_array":
            can_stack = True
            reduced = True
        else:
            can_stack = False
        if can_stack:
            if reduced:
                wrapped_objs = []
                for i, obj in enumerate(objs):
                    wrapped_objs.append(wrappers[i].wrap_reduced(obj, group_by=obj_group_by))
                return wrapper.concat_arrs(*wrapped_objs, group_by=obj_group_by).values
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.column_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along columns")

    @classmethod
    def column_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` along columns.

        All in-output tuples must be either None or have the same fields.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs stacking on the in-output objects of the same field using `Portfolio.column_stack_objs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                column_stack_func = prop_options.get("column_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                column_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    column_stack_func=field_options.get("column_stack_func", column_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.column_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @classmethod
    def column_stack(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        Cash sharing must be the same among all objects.

        Two-dimensional arrays are stacked using
        `vectorbtpro.base.wrapping.ArrayWrapper.column_stack_arrs`
        while one-dimensional arrays are stacked using
        `vectorbtpro.base.wrapping.ArrayWrapper.concat_arrs`.
        In-outputs are stacked using `Portfolio.column_stack_in_outputs`. Records are stacked using
        `vectorbtpro.records.base.Records.column_stack_records_arrs`."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        for i in range(1, len(objs)):
            if objs[i].cash_sharing != objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        kwargs["cash_sharing"] = objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False

        if "close" not in kwargs:
            new_close = kwargs["wrapper"].column_stack_arrs(
                *[obj.close for obj in objs],
                group_by=False,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
            kwargs["close"] = new_close
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.open for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.high for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.low for obj in objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.column_stack_records_arrs(*[obj.orders for obj in objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.column_stack_records_arrs(*[obj.logs for obj in objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in enums.InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                kwargs["init_cash"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.get_init_cash(group_by=cs_group_by) for obj in objs],
                        group_by=cs_group_by,
                    )
                )
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            for obj in objs:
                if (to_1d_array(obj.init_position) != 0).any():
                    stack_init_position_objs = True
                    break
            if stack_init_position_objs:
                kwargs["init_position"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_position for obj in objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_position"] = np.array([0.0])
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            for obj in objs:
                if not np.isnan(to_1d_array(obj.init_price)).all():
                    stack_init_price_objs = True
                    break
            if stack_init_price_objs:
                kwargs["init_price"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_price for obj in objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_price"] = np.array([np.nan])
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in objs],
                    group_by=cs_group_by,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.call_seq for obj in objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                new_bm_close = kwargs["wrapper"].column_stack_arrs(
                    *[obj.bm_close for obj in objs],
                    group_by=False,
                    wrap=False,
                )
                if fbfill_close:
                    new_bm_close = new_bm_close.vbt.fbfill()
                elif ffill_close:
                    new_bm_close = new_bm_close.vbt.ffill()
                kwargs["bm_close"] = new_bm_close
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.column_stack_in_outputs(*objs, **kwargs)

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "order_records",
        "close",
        "open",
        "high",
        "low",
        "log_records",
        "cash_sharing",
        "init_cash",
        "init_position",
        "init_price",
        "cash_deposits",
        "cash_earnings",
        "call_seq",
        "in_outputs",
        "use_in_outputs",
        "bm_close",
        "fillna_close",
        "trades_type",
        "orders_cls",
        "logs_cls",
        "trades_cls",
        "entry_trades_cls",
        "exit_trades_cls",
        "positions_cls",
        "drawdowns_cls",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        order_records: tp.Union[tp.RecordArray, enums.SimulationOutput],
        *,
        close: tp.ArrayLike,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        log_records: tp.Optional[tp.RecordArray] = None,
        cash_sharing: bool = False,
        init_cash: tp.Union[str, tp.ArrayLike] = "auto",
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        cash_deposits: tp.ArrayLike = 0.0,
        cash_earnings: tp.ArrayLike = 0.0,
        call_seq: tp.Optional[tp.Array2d] = None,
        in_outputs: tp.Optional[tp.NamedTuple] = None,
        use_in_outputs: tp.Optional[bool] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        fillna_close: tp.Optional[bool] = None,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        orders_cls: tp.Optional[type] = None,
        logs_cls: tp.Optional[type] = None,
        trades_cls: tp.Optional[type] = None,
        entry_trades_cls: tp.Optional[type] = None,
        exit_trades_cls: tp.Optional[type] = None,
        positions_cls: tp.Optional[type] = None,
        drawdowns_cls: tp.Optional[type] = None,
        **kwargs,
    ) -> None:

        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if cash_sharing:
            if wrapper.grouper.allow_enable or wrapper.grouper.allow_modify:
                wrapper = wrapper.replace(allow_enable=False, allow_modify=False)
        if isinstance(order_records, enums.SimulationOutput):
            sim_out = order_records
            order_records = sim_out.order_records
            log_records = sim_out.log_records
            cash_deposits = sim_out.cash_deposits
            cash_earnings = sim_out.cash_earnings
            call_seq = sim_out.call_seq
            in_outputs = sim_out.in_outputs
        Analyzable.__init__(
            self,
            wrapper,
            order_records=order_records,
            open=open,
            high=high,
            low=low,
            close=close,
            log_records=log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            call_seq=call_seq,
            in_outputs=in_outputs,
            use_in_outputs=use_in_outputs,
            bm_close=bm_close,
            fillna_close=fillna_close,
            trades_type=trades_type,
            orders_cls=orders_cls,
            logs_cls=logs_cls,
            trades_cls=trades_cls,
            entry_trades_cls=entry_trades_cls,
            exit_trades_cls=exit_trades_cls,
            positions_cls=positions_cls,
            drawdowns_cls=drawdowns_cls,
            **kwargs,
        )

        close = to_2d_array(close)
        if open is not None:
            open = to_2d_array(open)
        if high is not None:
            high = to_2d_array(high)
        if low is not None:
            low = to_2d_array(low)
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, enums.InitCashMode)
        if not checks.is_int(init_cash) or init_cash not in enums.InitCashMode:
            init_cash = to_1d_array(init_cash)
        init_position = to_1d_array(init_position)
        init_price = to_1d_array(init_price)
        cash_deposits = to_2d_array(cash_deposits)
        cash_earnings = to_2d_array(cash_earnings)
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = to_2d_array(bm_close)
        if log_records is None:
            log_records = np.array([], dtype=enums.log_dt)
        if use_in_outputs is None:
            use_in_outputs = portfolio_cfg["use_in_outputs"]
        if fillna_close is None:
            fillna_close = portfolio_cfg["fillna_close"]
        if trades_type is None:
            trades_type = portfolio_cfg["trades_type"]
        if isinstance(trades_type, str):
            trades_type = map_enum_fields(trades_type, enums.TradesType)

        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._order_records = order_records
        self._log_records = log_records
        self._cash_sharing = cash_sharing
        self._init_cash = init_cash
        self._init_position = init_position
        self._init_price = init_price
        self._cash_deposits = cash_deposits
        self._cash_earnings = cash_earnings
        self._call_seq = call_seq
        self._in_outputs = in_outputs
        self._use_in_outputs = use_in_outputs
        self._bm_close = bm_close
        self._fillna_close = fillna_close
        self._trades_type = trades_type
        self._orders_cls = orders_cls
        self._logs_cls = logs_cls
        self._trades_cls = trades_cls
        self._entry_trades_cls = entry_trades_cls
        self._exit_trades_cls = exit_trades_cls
        self._positions_cls = positions_cls
        self._drawdowns_cls = drawdowns_cls

        # Only slices of rows can be selected
        self._range_only_select = True

        # Copy writeable attrs
        self._in_output_config = type(self)._in_output_config.copy()

    # ############# In-outputs ############# #

    _in_output_config: tp.ClassVar[Config] = HybridConfig(
        dict(
            cash=dict(grouping="cash_sharing"),
            position=dict(grouping="columns"),
            debt=dict(grouping="columns"),
            locked_cash=dict(grouping="columns"),
            free_cash=dict(grouping="cash_sharing"),
            returns=dict(grouping="cash_sharing"),
        )
    )

    @property
    def in_output_config(self) -> Config:
        """In-output config of `${cls_name}`.

        ```python
        ${in_output_config}
        ```

        Returns `${cls_name}._in_output_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change in_outputs, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._in_output_config`.
        """
        return self._in_output_config

    @classmethod
    def parse_field_options(cls, field: str) -> tp.Kwargs:
        """Parse options based on the name of a field.

        Returns a dictionary with the parsed grouping, object type, and cleaned field name.

        Grouping is parsed by looking for the following suffixes:

        * '_cs': per group if grouped with cash sharing, otherwise per column
        * '_pcg': per group if grouped, otherwise per column
        * '_pg': per group
        * '_pc': per column
        * '_records': records

        Object type is parsed by looking for the following suffixes:

        * '_2d': element per timestamp and column/group (time series)
        * '_1d': element per column/group (reduced time series)

        Those substrings are then removed to produce a clean field name."""
        options = dict()
        new_parts = []
        for part in field.split("_"):
            if part == "1d":
                options["obj_type"] = "red_array"
            elif part == "2d":
                options["obj_type"] = "array"
            elif part == "records":
                options["obj_type"] = "records"
            elif part == "pc":
                options["grouping"] = "columns"
            elif part == "pg":
                options["grouping"] = "groups"
            elif part == "pcg":
                options["grouping"] = "columns_or_groups"
            elif part == "cs":
                options["grouping"] = "cash_sharing"
            else:
                new_parts.append(part)
        field = "_".join(new_parts)
        options["field"] = field
        return options

    def matches_field_options(
        self,
        options: tp.Kwargs,
        obj_type: tp.Optional[str] = None,
        group_by_aware: bool = True,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> bool:
        """Return whether options of a field match the requirements.

        Requirements include the type of the object (array, reduced array, records),
        the grouping of the object (1/2 dimensions, group/column-wise layout). The current
        grouping and cash sharing of this portfolio object are also taken into account.

        When an option is not in `options`, it's automatically marked as matching."""
        field_obj_type = options.get("obj_type", None)
        field_grouping = options.get("grouping", None)
        if field_obj_type is not None and obj_type is not None:
            if field_obj_type != obj_type:
                return False
        if field_grouping is not None:
            if wrapper is None:
                wrapper = self.wrapper
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if is_grouped:
                if group_by_aware:
                    if field_grouping == "groups":
                        return True
                    if field_grouping == "columns_or_groups":
                        return True
                    if self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
                else:
                    if field_grouping == "columns":
                        return True
                    if not self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
            else:
                if field_grouping == "columns":
                    return True
                if field_grouping == "columns_or_groups":
                    return True
                if field_grouping == "cash_sharing":
                    return True
            return False
        return True

    def wrap_obj(
        self,
        obj: tp.Any,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_func: tp.Optional[tp.Callable] = None,
        wrap_kwargs: tp.KwargsLike = None,
        force_wrapping: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Wrap an object.

        `wrap_func` must take the portfolio, `obj`, all the arguments passed to this method, and `**kwargs`.
        If you don't need any of the arguments, make `indexing_func` accept them as `**kwargs`.

        If the object is None or boolean, returns as-is."""
        if obj is None or isinstance(obj, bool):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
        if wrap_func is not None:
            return wrap_func(
                self,
                obj,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
                force_wrapping=force_wrapping,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _wrap_reduced_grouped(obj):
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=group_by, **_wrap_kwargs)

        def _wrap_reduced(obj):
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=False, **_wrap_kwargs)

        def _wrap_grouped(obj):
            return wrapper.wrap(obj, group_by=group_by, **resolve_dict(wrap_kwargs))

        def _wrap(obj):
            return wrapper.wrap(obj, group_by=False, **resolve_dict(wrap_kwargs))

        if obj_type is not None and obj_type not in {"records"}:
            if grouping == "cash_sharing":
                if obj_type == "array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj_type == "red_array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    if is_grouped and self.cash_sharing:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj.ndim == 1:
                    if is_grouped and self.cash_sharing:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
            if grouping == "columns_or_groups":
                if obj_type == "array":
                    if is_grouped:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj_type == "red_array":
                    if is_grouped:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    if is_grouped:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj.ndim == 1:
                    if is_grouped:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
            if grouping == "groups":
                if obj_type == "array":
                    return _wrap_grouped(obj)
                if obj_type == "red_array":
                    return _wrap_reduced_grouped(obj)
                if obj.ndim == 2:
                    return _wrap_grouped(obj)
                if obj.ndim == 1:
                    return _wrap_reduced_grouped(obj)
            if grouping == "columns":
                if obj_type == "array":
                    return _wrap(obj)
                if obj_type == "red_array":
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    return _wrap(obj)
                if obj.ndim == 1:
                    return _wrap_reduced(obj)
        if obj_type not in {"records"}:
            if checks.is_np_array(obj) and not checks.is_record_array(obj):
                if is_grouped:
                    if obj_type is not None and obj_type == "array":
                        return _wrap_grouped(obj)
                    if obj_type is not None and obj_type == "red_array":
                        return _wrap_reduced_grouped(obj)
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _wrap_grouped(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return _wrap_reduced_grouped(obj)
                if obj_type is not None and obj_type == "array":
                    return _wrap(obj)
                if obj_type is not None and obj_type == "red_array":
                    return _wrap_reduced(obj)
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _wrap(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return _wrap_reduced(obj)
        if force_wrapping:
            raise NotImplementedError(f"Cannot wrap object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to wrap object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def get_in_output(
        self,
        field: str,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> tp.Union[None, bool, tp.AnyArray]:
        """Find and wrap an in-output object matching the field.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        If `field` is not in `Portfolio.in_outputs`, searches for the field in aliases and options.
        In such case, to narrow down the number of candidates, options are additionally matched against
        the requirements using `Portfolio.matches_field_options`. Finally, the matched in-output object is
        wrapped using `Portfolio.wrap_obj`."""
        if self.in_outputs is None:
            raise ValueError("No in-outputs attached")

        if field in self.cls_dir:
            prop = getattr(type(self), field)
            prop_options = getattr(prop, "options", {})
            obj_type = prop_options.get("obj_type", "array")
            group_by_aware = prop_options.get("group_by_aware", True)
            wrap_func = prop_options.get("wrap_func", None)
            wrap_kwargs = prop_options.get("wrap_kwargs", None)
            force_wrapping = prop_options.get("force_wrapping", False)
            silence_warnings = prop_options.get("silence_warnings", False)
            field_aliases = prop_options.get("field_aliases", None)
            if field_aliases is None:
                field_aliases = []
            field_aliases = {field, *field_aliases}
            found_attr = True
        else:
            obj_type = None
            group_by_aware = True
            wrap_func = None
            wrap_kwargs = None
            force_wrapping = False
            silence_warnings = False
            field_aliases = {field}
            found_attr = False

        found_field = None
        found_field_options = None
        for _field in set(self.in_outputs._fields):
            _field_options = merge_dicts(
                self.parse_field_options(_field),
                self.in_output_config.get(_field, None),
            )
            if (not found_attr and field == _field) or (
                (_field in field_aliases or _field_options.get("field", _field) in field_aliases)
                and self.matches_field_options(
                    _field_options,
                    obj_type=obj_type,
                    group_by_aware=group_by_aware,
                    group_by=group_by,
                    wrapper=wrapper,
                )
            ):
                if found_field is not None:
                    raise ValueError(f"Multiple fields for '{field}' found in in_outputs")
                found_field = _field
                found_field_options = _field_options
        if found_field is None:
            raise AttributeError(f"No compatible field for '{field}' found in in_outputs")
        obj = getattr(self.in_outputs, found_field)
        if found_attr and checks.is_np_array(obj) and obj.shape == (0, 0):  # for returns
            return None
        kwargs = merge_dicts(
            dict(
                grouping=found_field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                obj_type=found_field_options.get("obj_type", obj_type),
                wrap_func=found_field_options.get("wrap_func", wrap_func),
                wrap_kwargs=found_field_options.get("wrap_kwargs", wrap_kwargs),
                force_wrapping=found_field_options.get("force_wrapping", force_wrapping),
                silence_warnings=found_field_options.get("silence_warnings", silence_warnings),
            ),
            kwargs,
        )
        return self.wrap_obj(
            obj,
            found_field_options.get("field", found_field),
            group_by=group_by,
            wrapper=wrapper,
            **kwargs,
        )

    # ############# Indexing ############# #

    def index_obj(
        self,
        obj: tp.Any,
        wrapper_meta: dict,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        indexing_func: tp.Optional[tp.Callable] = None,
        force_indexing: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Perform indexing on an object.

        `indexing_func` must take the portfolio, all the arguments passed to this method, and `**kwargs`.
        If you don't need any of the arguments, make `indexing_func` accept them as `**kwargs`.

        If the object is None, boolean, or empty, returns as-is."""
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if indexing_func is not None:
            return indexing_func(
                self,
                obj,
                wrapper_meta,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                force_indexing=force_indexing,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _index_1d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["group_idxs"]]

        def _index_1d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["col_idxs"]]

        def _index_2d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["group_idxs"]]

        def _index_2d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["col_idxs"]]

        def _index_records(obj: tp.RecordArray) -> tp.RecordArray:
            records = Records(wrapper, obj)
            records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
            return records.indexing_func(records_meta=records_meta).values

        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
        if obj_type is not None and obj_type == "records":
            return _index_records(obj)
        if grouping == "cash_sharing":
            if obj_type is not None and obj_type == "array":
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "columns_or_groups":
            if obj_type is not None and obj_type == "array":
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "groups":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_group(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_group(obj)
            if obj.ndim == 2:
                return _index_2d_by_group(obj)
            if obj.ndim == 1:
                return _index_1d_by_group(obj)
        if grouping == "columns":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                return _index_1d_by_col(obj)
        if checks.is_np_array(obj):
            if is_grouped:
                if obj_type is not None and obj_type == "array":
                    return _index_2d_by_group(obj)
                if obj_type is not None and obj_type == "red_array":
                    return _index_1d_by_group(obj)
                if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                    return _index_2d_by_group(obj)
                if obj.shape == (wrapper.get_shape_2d()[1],):
                    return _index_1d_by_group(obj)
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if to_2d_shape(obj.shape) == wrapper.shape_2d:
                return _index_2d_by_col(obj)
            if obj.shape == (wrapper.shape_2d[1],):
                return _index_1d_by_col(obj)
        if force_indexing:
            raise NotImplementedError(f"Cannot index object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to index object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def in_outputs_indexing_func(self, wrapper_meta: dict, **kwargs) -> tp.Optional[tp.NamedTuple]:
        """Perform indexing on `Portfolio.in_outputs`.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs indexing on the in-output object using `Portfolio.index_obj`."""
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                indexing_func = prop_options.get("indexing_func", None)
                force_indexing = prop_options.get("force_indexing", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                group_by_aware = True
                indexing_func = None
                force_indexing = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    indexing_func=field_options.get("indexing_func", indexing_func),
                    force_indexing=field_options.get("force_indexing", force_indexing),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.index_obj(obj, wrapper_meta, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def indexing_func(
        self: PortfolioT,
        *args,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Perform indexing on `Portfolio`.

        In-outputs are indexed using `Portfolio.in_outputs_indexing_func`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                range_only_select=self.range_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        row_idxs = wrapper_meta["row_idxs"]
        rows_changed = wrapper_meta["rows_changed"]
        col_idxs = wrapper_meta["col_idxs"]
        columns_changed = wrapper_meta["columns_changed"]
        group_idxs = wrapper_meta["group_idxs"]
        groups_changed = wrapper_meta["groups_changed"]

        new_close = ArrayWrapper.select_from_flex_array(
            self._close,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if self._open is not None:
            new_open = ArrayWrapper.select_from_flex_array(
                self._open,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_open = self._open
        if self._high is not None:
            new_high = ArrayWrapper.select_from_flex_array(
                self._high,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_high = self._high
        if self._low is not None:
            new_low = ArrayWrapper.select_from_flex_array(
                self._low,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_low = self._low
        new_order_records = self.orders.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_log_records = self.logs.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_init_cash = self._init_cash
        if not checks.is_int(new_init_cash):
            new_init_cash = to_1d_array(new_init_cash)
            if rows_changed and row_idxs.start > 0:
                if self.wrapper.grouper.is_grouped() and not self.cash_sharing:
                    cash = self.get_cash(group_by=False)
                else:
                    cash = self.cash
                new_init_cash = to_1d_array(cash.iloc[row_idxs.start - 1])
            if columns_changed and new_init_cash.shape[0] > 1:
                if self.cash_sharing:
                    new_init_cash = new_init_cash[group_idxs]
                else:
                    new_init_cash = new_init_cash[col_idxs]
        new_init_position = to_1d_array(self._init_position)
        if rows_changed and row_idxs.start > 0:
            new_init_position = to_1d_array(self.assets.iloc[row_idxs.start - 1])
        if columns_changed and new_init_position.shape[0] > 1:
            new_init_position = new_init_position[col_idxs]
        new_init_price = to_1d_array(self._init_price)
        if rows_changed and row_idxs.start > 0:
            new_init_price = to_1d_array(self.close.iloc[: row_idxs.start].ffill().iloc[-1])
        if columns_changed and new_init_price.shape[0] > 1:
            new_init_price = new_init_price[col_idxs]
        new_cash_deposits = ArrayWrapper.select_from_flex_array(
            self._cash_deposits,
            row_idxs=row_idxs,
            col_idxs=group_idxs if self.cash_sharing else col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        new_cash_earnings = ArrayWrapper.select_from_flex_array(
            self._cash_earnings,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if self._call_seq is not None:
            new_call_seq = ArrayWrapper.select_from_flex_array(
                self._call_seq,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_call_seq = None
        if self._bm_close is not None and not isinstance(self._bm_close, bool):
            new_bm_close = ArrayWrapper.select_from_flex_array(
                self._bm_close,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_bm_close = self._bm_close
        new_in_outputs = self.in_outputs_indexing_func(wrapper_meta, **resolve_dict(in_output_kwargs))

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
            log_records=new_log_records,
            init_cash=new_init_cash,
            init_position=new_init_position,
            init_price=new_init_price,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            call_seq=new_call_seq,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
        )

    # ############# Resampling ############# #

    def resample_obj(
        self,
        obj: tp.Any,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        resample_func: tp.Union[None, str, tp.Callable] = None,
        resample_kwargs: tp.KwargsLike = None,
        force_resampling: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Resample an object.

        `resample_func` must take the portfolio, `obj`, `resampler`, all the arguments passed to this method,
        and `**kwargs`. If you don't need any of the arguments, make `resample_func` accept them as `**kwargs`.
        If `resample_func` is a string, will use it as `reduce_func_nb` in
        `vectorbtpro.generic.accessors.GenericAccessor.resample_apply`. Default is 'last'.

        If the object is None, boolean, or empty, returns as-is."""
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if resample_func is None:
            resample_func = "last"
        if not isinstance(resample_func, str):
            return resample_func(
                self,
                obj,
                resampler,
                obj_name=obj_name,
                obj_type=obj_type,
                group_by=group_by,
                wrapper=wrapper,
                resample_kwargs=resample_kwargs,
                force_resampling=force_resampling,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _resample(obj: tp.Array) -> tp.SeriesFrame:
            wrapped_obj = ArrayWrapper.from_obj(obj, index=wrapper.index).wrap(obj)
            return wrapped_obj.vbt.resample_apply(resampler, resample_func, **resolve_dict(resample_kwargs)).values

        if obj_type is not None and obj_type == "red_array":
            return obj
        if obj_type is None or obj_type == "array":
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if checks.is_np_array(obj):
                if is_grouped:
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _resample(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return obj
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _resample(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return obj
        if force_resampling:
            raise NotImplementedError(f"Cannot resample object '{obj_name}'")
        if not silence_warnings:
            warnings.warn(
                f"Cannot figure out how to resample object '{obj_name}'",
                stacklevel=2,
            )
        return obj

    def resample_in_outputs(
        self,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Resample `Portfolio.in_outputs`.

        If the field can be found in the attributes of this `Portfolio` instance, reads the
        attribute's options to get requirements for the type and layout of the in-output object.

        For each field in `Portfolio.in_outputs`, resolves the field's options by parsing its name with
        `Portfolio.parse_field_options` and also looks for options in `Portfolio.in_output_config`.
        Performs indexing on the in-output object using `Portfolio.resample_obj`."""
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                resample_func = prop_options.get("resample_func", None)
                resample_kwargs = prop_options.get("resample_kwargs", None)
                force_resampling = prop_options.get("force_resampling", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                resample_func = None
                resample_kwargs = None
                force_resampling = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    resample_func=field_options.get("resample_func", resample_func),
                    resample_kwargs=field_options.get("resample_kwargs", resample_kwargs),
                    force_resampling=field_options.get("force_resampling", force_resampling),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.resample_obj(obj, resampler, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def resample(
        self: PortfolioT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Resample the `Portfolio` instance.

        !!! warning
            Downsampling is associated with information loss:

            * Cash deposits and earnings are assumed to be added/removed at the beginning of each time step.
                Imagine depositing $100 and using them up in the same bar, and then depositing another $100
                and using them up. Downsampling both bars into a single bar will aggregate cash deposits
                and earnings, and put both of them at the beginning of the new bar, even though the second
                deposit was added later in time.
            * Market/benchmark returns are computed by applying the initial value on the close price
                of the first bar and by tracking the price change to simulate holding. Moving the close
                price of the first bar further into the future will affect this computation and almost
                certainly produce a different market value and returns. To mitigate this, make sure
                to downsample to an index with the first bar containing only the first bar from the
                origin timeframe."""
        if self._call_seq is not None:
            raise ValueError("Cannot resample call_seq")
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        resampler = wrapper_meta["resampler"]
        new_wrapper = wrapper_meta["new_wrapper"]

        new_close = self.close.vbt.resample_apply(resampler, "last")
        if fbfill_close:
            new_close = new_close.vbt.fbfill()
        elif ffill_close:
            new_close = new_close.vbt.ffill()
        new_close = new_close.values
        if self._open is not None:
            new_open = self.open.vbt.resample_apply(resampler, "first").values
        else:
            new_open = self._open
        if self._high is not None:
            new_high = self.high.vbt.resample_apply(resampler, "max").values
        else:
            new_high = self._high
        if self._low is not None:
            new_low = self.low.vbt.resample_apply(resampler, "min").values
        else:
            new_low = self._low
        new_order_records = self.orders.resample_records_arr(resampler)
        new_log_records = self.logs.resample_records_arr(resampler)
        if self._cash_deposits.size > 1 or self._cash_deposits.item() != 0:
            new_cash_deposits = self.get_cash_deposits(group_by=None if self.cash_sharing else False)
            new_cash_deposits = new_cash_deposits.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_deposits = new_cash_deposits.fillna(0.0)
            new_cash_deposits = new_cash_deposits.values
        else:
            new_cash_deposits = self._cash_deposits
        if self._cash_earnings.size > 1 or self._cash_earnings.item() != 0:
            new_cash_earnings = self.get_cash_earnings(group_by=False)
            new_cash_earnings = new_cash_earnings.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_earnings = new_cash_earnings.fillna(0.0)
            new_cash_earnings = new_cash_earnings.values
        else:
            new_cash_earnings = self._cash_earnings
        if self._bm_close is not None and not isinstance(self._bm_close, bool):
            new_bm_close = self.bm_close.vbt.resample_apply(resampler, "last")
            if fbfill_close:
                new_bm_close = new_bm_close.vbt.fbfill()
            elif ffill_close:
                new_bm_close = new_bm_close.vbt.ffill()
            new_bm_close = new_bm_close.values
        else:
            new_bm_close = self._bm_close
        if self._in_outputs is not None:
            new_in_outputs = self.resample_in_outputs(resampler, **resolve_dict(in_output_kwargs))
        else:
            new_in_outputs = None

        return self.replace(
            wrapper=new_wrapper,
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
            log_records=new_log_records,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
        )

    # ############# Class methods ############# #

    @classmethod
    def from_orders(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data, FOPreparer, PFPrepResult],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        save_state: tp.Optional[bool] = None,
        save_value: tp.Optional[bool] = None,
        save_returns: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from orders - size, price, fees, and other information.

        See `vectorbtpro.portfolio.nb.from_orders.from_orders_nb`.

        Prepared by `vectorbtpro.portfolio.preparing.FOPreparer`.

        Args:
            close (array_like, Data, FOPreparer, or PFPrepResult): Latest asset price at each time step.
                Will broadcast.

                Used for calculating unrealized PnL and portfolio value.

                If an instance of `vectorbtpro.data.base.Data`, will extract the open, high, low, and close price.
                If an instance of `vectorbtpro.portfolio.preparing.FOPreparer`, will use it as a preparer.
                If an instance of `vectorbtpro.portfolio.preparing.PFPrepResult`, will use it as a preparer result.
            size (float or array_like): Size to order.
                See `vectorbtpro.portfolio.enums.Order.size`. Will broadcast.
            size_type (SizeType or array_like): See `vectorbtpro.portfolio.enums.SizeType` and
                `vectorbtpro.portfolio.enums.Order.size_type`. Will broadcast.
            direction (Direction or array_like): See `vectorbtpro.portfolio.enums.Direction` and
                `vectorbtpro.portfolio.enums.Order.direction`. Will broadcast.
            price (array_like of float): Order price.
                Will broadcast.

                See `vectorbtpro.portfolio.enums.Order.price`. Can be also provided as
                `vectorbtpro.portfolio.enums.PriceType`. Options `PriceType.NextOpen` and `PriceType.NextClose`
                are only applicable per column, that is, they cannot be used inside full arrays.
                In addition, they require the argument `from_ago` to be None.
            fees (float or array_like): Fees in percentage of the order value.
                See `vectorbtpro.portfolio.enums.Order.fees`. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
                See `vectorbtpro.portfolio.enums.Order.fixed_fees`. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price.
                See `vectorbtpro.portfolio.enums.Order.slippage`. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.
                See `vectorbtpro.portfolio.enums.Order.min_size`. Will broadcast.
            max_size (float or array_like): Maximum size for an order.
                See `vectorbtpro.portfolio.enums.Order.max_size`. Will broadcast.

                Will be partially filled if exceeded.
            size_granularity (float or array_like): Granularity of the size.
                See `vectorbtpro.portfolio.enums.Order.size_granularity`. Will broadcast.
            leverage (float or array_like): Leverage.
                See `vectorbtpro.portfolio.enums.Order.leverage`. Will broadcast.
            leverage_mode (LeverageMode or array_like): Leverage mode.
                See `vectorbtpro.portfolio.enums.Order.leverage_mode`. Will broadcast.
            reject_prob (float or array_like): Order rejection probability.
                See `vectorbtpro.portfolio.enums.Order.reject_prob`. Will broadcast.
            price_area_vio_mode (PriceAreaVioMode or array_like): See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
                Will broadcast.
            allow_partial (bool or array_like): Whether to allow partial fills.
                See `vectorbtpro.portfolio.enums.Order.allow_partial`. Will broadcast.

                Does not apply when size is `np.inf`.
            raise_reject (bool or array_like): Whether to raise an exception if order gets rejected.
                See `vectorbtpro.portfolio.enums.Order.raise_reject`. Will broadcast.
            log (bool or array_like): Whether to log orders.
                See `vectorbtpro.portfolio.enums.Order.log`. Will broadcast.
            val_price (array_like of float): Asset valuation price.
                Will broadcast.

                Can be also provided as `vectorbtpro.portfolio.enums.ValPriceType`.

                * Any `-np.inf` element is replaced by the latest valuation price
                    (`open` or the latest known valuation price if `ffill_val_price`).
                * Any `np.inf` element is replaced by the current order price.

                Used at the time of decision making to calculate value of each asset in the group,
                for example, to convert target value into target amount.

                !!! note
                    In contrast to `Portfolio.from_order_func`, order price is known beforehand (kind of),
                    thus `val_price` is set to the current order price (using `np.inf`) by default.
                    To valuate using previous close, set it in the settings to `-np.inf`.

                !!! note
                    Make sure to use timestamp for `val_price` that comes before timestamps of
                    all orders in the group with cash sharing (previous `close` for example),
                    otherwise you're cheating yourself.
            open (array_like of float): First asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            high (array_like of float): Highest asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            low (array_like of float): Lowest asset price at each time step.
                Defaults to `np.nan`. Will broadcast.

                Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            init_cash (InitCashMode, float or array_like): Initial capital.

                By default, will broadcast to the final number of columns.
                But if cash sharing is enabled, will broadcast to the number of groups.
                See `vectorbtpro.portfolio.enums.InitCashMode` to find optimal initial cash.

                !!! note
                    Mode `InitCashMode.AutoAlign` is applied after the portfolio is initialized
                    to set the same initial cash for all columns/groups. Changing grouping
                    will change the initial cash, so be aware when indexing.
            init_position (float or array_like): Initial position.

                By default, will broadcast to the final number of columns.
            init_price (float or array_like): Initial position price.

                By default, will broadcast to the final number of columns.
            cash_deposits (float or array_like): Cash to be deposited/withdrawn at each timestamp.
                Will broadcast to the final shape. Must have the same number of columns as `init_cash`.

                Applied at the beginning of each timestamp.
            cash_earnings (float or array_like): Earnings in cash to be added at each timestamp.
                Will broadcast to the final shape.

                Applied at the end of each timestamp.
            cash_dividends (float or array_like): Dividends in cash to be added at each timestamp.
                Will broadcast to the final shape.

                Gets multiplied by the position and saved into `cash_earnings`.

                Applied at the end of each timestamp.
            cash_sharing (bool): Whether to share cash within the same group.

                If `group_by` is None and `cash_sharing` is True, `group_by` becomes True to form a single
                group with cash sharing.

                !!! warning
                    Introduces cross-asset dependencies.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel.
            from_ago (int or array_like): Take order information from a number of bars ago.
                Will broadcast.

                Negative numbers will be cast to positive to avoid the look-ahead bias. Defaults to 0.
                Remember to account of it if you're using a custom signal function!
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                Each value in this sequence must indicate the position of column in the group to
                call next. Processing of `call_seq` goes always from left to right.
                For example, `[2, 0, 1]` would first call column 'c', then 'a', and finally 'b'.

                Supported are multiple options:

                * Set to None to generate the default call sequence on the fly. Will create a
                    full array only if `attach_call_seq` is True.
                * Use `vectorbtpro.portfolio.enums.CallSeqType` to create a full array of a specific type.
                * Set to array to specify a custom call sequence.

                If `CallSeqType.Auto` selected, rearranges calls dynamically based on order value.
                Calculates value of all orders per row and group, and sorts them by this value.
                Sell orders will be executed first to release funds for buy orders.

                !!! warning
                    `CallSeqType.Auto` should be used with caution:

                    * It not only presumes that order prices are known beforehand, but also that
                        orders can be executed in arbitrary order and still retain their price.
                        In reality, this is hardly the case: after processing one asset, some time
                        has passed and the price for other assets might have already changed.
                    * Even if you're able to specify a slippage large enough to compensate for
                        this behavior, slippage itself should depend upon execution order.
                        This method doesn't let you do that.
                    * Orders in the same queue are executed regardless of whether previous orders
                        have been filled, which can leave them without required funds.

                    For more control, use `Portfolio.from_order_func`.
            attach_call_seq (bool): Whether to attach `call_seq` to the instance.

                Makes sense if you want to analyze the simulation order. Otherwise, just takes memory.
            ffill_val_price (bool): Whether to track valuation price only if it's known.

                Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp.
            update_value (bool): Whether to update group value after each filled order.
            save_state (bool): Whether to save the state.

                The arrays will be available as `cash`, `position`, `debt`, `locked_cash`, and `free_cash` in in-outputs.
            save_value (bool): Whether to save the value.

                The array will be available as `value` in in-outputs.
            save_returns (bool): Whether to save the returns.

                The array will be available as `returns` in in-outputs.
            max_orders (int): The max number of order records expected to be filled at each column.
                Defaults to the maximum number of non-NaN values across all columns of the size array.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            max_logs (int): The max number of log records expected to be filled at each column.
                Defaults to the maximum number of True values across all columns of the log array.

                Set to a lower number if you run out of memory, and to 0 to not fill.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            group_by (any): Group columns. See `vectorbtpro.base.grouping.base.Grouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (any): See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            freq (any): Index frequency in case it cannot be parsed from `close`.
            bm_close (array_like): Latest benchmark price at each time step.
                Will broadcast.

                If not provided, will use `close`. If False, will not use any benchmark.
            records (array_like): Records to construct arrays from.

                See `vectorbtpro.base.indexing.IdxRecords`.
            return_preparer (bool): Whether to return the preparer of the type
                `vectorbtpro.portfolio.preparing.FOPreparer`.

                !!! note
                    Seed won't be set in this case, you need to explicitly call `preparer.set_seed()`.
            return_prep_result (bool): Whether to return the preparer result of the type
                `vectorbtpro.portfolio.preparing.PFPrepResult`.
            return_sim_out (bool): Whether to return the simulation output of the type
                `vectorbtpro.portfolio.enums.SimulationOutput`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        !!! note
            When `call_seq` is not `CallSeqType.Auto`, at each timestamp, processing of the assets in
            a group goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why rebalancing
            cannot work properly in this setting: one has to specify percentages for all assets beforehand
            and then tweak the processing order to sell to-be-sold assets first in order to release funds
            for to-be-bought assets. This can be automatically done by using `CallSeqType.Auto`.

        !!! hint
            All broadcastable arguments can be set per frame, series, row, column, or element.

        Usage:
            * Buy 10 units each tick:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_orders(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Reverse each position by first closing it:

            ```pycon
            >>> size = [1, 0, -1, 0, 1]
            >>> pf = vbt.Portfolio.from_orders(close, size, size_type='targetpercent')

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * Regularly deposit cash at open and invest it within the same bar at close:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> cash_deposits = pd.Series([10., 0., 10., 0., 10.])
            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size=cash_deposits,  # invest the amount deposited
            ...     size_type='value',
            ...     cash_deposits=cash_deposits
            ... )

            >>> pf.cash
            0    100.0
            1    100.0
            2    100.0
            3    100.0
            4    100.0
            dtype: float64

            >>> pf.asset_flow
            0    10.000000
            1     0.000000
            2     3.333333
            3     0.000000
            4     2.000000
            dtype: float64
            ```

            * Equal-weighted portfolio as in `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb` example
            (it's more compact but has less control over execution):

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))
            >>> size = pd.Series(np.full(5, 1/3))  # each column 33.3%
            >>> size[1::2] = np.nan  # skip every second tick

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,  # acts both as reference and order price here
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',  # first sell then buy
            ...     group_by=True,  # one group
            ...     cash_sharing=True,  # assets share the same cash
            ...     fees=0.001, fixed_fees=1., slippage=0.001  # costs
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_nb_example.svg){: .iimg loading=lazy }

            * Test 10 random weight combinations:

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(
            ...     np.random.uniform(1, 10, size=(5, 3)),
            ...     columns=pd.Index(['a', 'b', 'c'], name='asset'))

            >>> # Generate random weight combinations
            >>> rand_weights = []
            >>> for i in range(10):
            ...     rand_weights.append(np.random.dirichlet(np.ones(close.shape[1]), size=1)[0])
            >>> rand_weights
            [array([0.15474873, 0.27706078, 0.5681905 ]),
             array([0.30468598, 0.18545189, 0.50986213]),
             array([0.15780486, 0.36292607, 0.47926907]),
             array([0.25697713, 0.64902589, 0.09399698]),
             array([0.43310548, 0.53836359, 0.02853093]),
             array([0.78628605, 0.15716865, 0.0565453 ]),
             array([0.37186671, 0.42150531, 0.20662798]),
             array([0.22441579, 0.06348919, 0.71209502]),
             array([0.41619664, 0.09338007, 0.49042329]),
             array([0.01279537, 0.87770864, 0.10949599])]

            >>> # Bring close and rand_weights to the same shape
            >>> rand_weights = np.concatenate(rand_weights)
            >>> close = close.vbt.tile(10, keys=pd.Index(np.arange(10), name='weights_vector'))
            >>> size = vbt.broadcast_to(weights, close).copy()
            >>> size[1::2] = np.nan
            >>> size
            weights_vector                            0  ...                               9
            asset                  a         b        c  ...           a         b         c
            0               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            1                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            2               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            3                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            4               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496

            [5 rows x 30 columns]

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',
            ...     group_by='weights_vector',  # group by column level
            ...     cash_sharing=True,
            ...     fees=0.001, fixed_fees=1., slippage=0.001
            ... )

            >>> pf.total_return
            weights_vector
            0   -0.294372
            1    0.139207
            2   -0.281739
            3    0.041242
            4    0.467566
            5    0.829925
            6    0.320672
            7   -0.087452
            8    0.376681
            9   -0.702773
            Name: total_return, dtype: float64
            ```
        """
        if isinstance(close, FOPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            if isinstance(close, Data):
                local_kwargs["data"] = close
                local_kwargs["close"] = None
            preparer = FOPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data, FSPreparer, PFPrepResult],
        entries: tp.Optional[tp.ArrayLike] = None,
        exits: tp.Optional[tp.ArrayLike] = None,
        *,
        direction: tp.Optional[tp.ArrayLike] = None,
        long_entries: tp.Optional[tp.ArrayLike] = None,
        long_exits: tp.Optional[tp.ArrayLike] = None,
        short_entries: tp.Optional[tp.ArrayLike] = None,
        short_exits: tp.Optional[tp.ArrayLike] = None,
        adjust_func_nb: tp.Union[None, tp.PathLike, nb.AdjustFuncT] = None,
        adjust_args: tp.Args = (),
        signal_func_nb: tp.Union[None, tp.PathLike, nb.SignalFuncT] = None,
        signal_args: tp.ArgsLike = (),
        post_segment_func_nb: tp.Union[None, tp.PathLike, nb.PostSegmentFuncT] = None,
        post_segment_args: tp.ArgsLike = (),
        order_mode: bool = False,
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        accumulate: tp.Optional[tp.ArrayLike] = None,
        upon_long_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_short_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_dir_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opposite_entry: tp.Optional[tp.ArrayLike] = None,
        order_type: tp.Optional[tp.ArrayLike] = None,
        limit_delta: tp.Optional[tp.ArrayLike] = None,
        limit_tif: tp.Optional[tp.ArrayLike] = None,
        limit_expiry: tp.Optional[tp.ArrayLike] = None,
        limit_reverse: tp.Optional[tp.ArrayLike] = None,
        upon_adj_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        use_stops: tp.Optional[bool] = None,
        stop_ladder: tp.Optional[bool] = None,
        sl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_th: tp.Optional[tp.ArrayLike] = None,
        tp_stop: tp.Optional[tp.ArrayLike] = None,
        td_stop: tp.Optional[tp.ArrayLike] = None,
        dt_stop: tp.Optional[tp.ArrayLike] = None,
        stop_entry_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_type: tp.Optional[tp.ArrayLike] = None,
        stop_order_type: tp.Optional[tp.ArrayLike] = None,
        stop_limit_delta: tp.Optional[tp.ArrayLike] = None,
        upon_stop_update: tp.Optional[tp.ArrayLike] = None,
        upon_adj_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        delta_format: tp.Optional[tp.ArrayLike] = None,
        time_delta_format: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_pos_info: tp.Optional[bool] = None,
        save_state: tp.Optional[bool] = None,
        save_value: tp.Optional[bool] = None,
        save_returns: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        staticized: tp.StaticizedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from entry and exit signals.

        Supports the following modes:

        1. `entries` and `exits`:
            Uses `vectorbtpro.portfolio.nb.from_signals.dir_signal_func_nb` as `signal_func_nb`
            if an adjustment function is provided (not cacheable), otherwise translates signals using
            `vectorbtpro.portfolio.nb.from_signals.dir_to_ls_signals_nb` then simulates statically (cacheable)
        2. `entries` (acting as long), `exits` (acting as long), `short_entries`, and `short_exits`:
            Uses `vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb` as `signal_func_nb`
            if an adjustment function is provided (not cacheable), otherwise simulates statically (cacheable)
        3. `order_mode=True` without signals:
            Uses `vectorbtpro.portfolio.nb.from_signals.order_signal_func_nb` as `signal_func_nb` (not cacheable)
        4. `signal_func_nb` and `signal_args`: Custom signal function (not cacheable)

        Prepared by `vectorbtpro.portfolio.preparing.FSPreparer`.

        Args:
            close (array_like, Data, FSPreparer, or PFPrepResult): See `Portfolio.from_orders`.
            entries (array_like of bool): Boolean array of entry signals.
                Defaults to True if all other signal arrays are not set, otherwise False. Will broadcast.

                * If `short_entries` and `short_exits` are not set: Acts as a long signal if `direction`
                    is 'all' or 'longonly', otherwise short.
                * If `short_entries` or `short_exits` are set: Acts as `long_entries`.
            exits (array_like of bool): Boolean array of exit signals.
                Defaults to False. Will broadcast.

                * If `short_entries` and `short_exits` are not set: Acts as a short signal if `direction`
                    is 'all' or 'longonly', otherwise long.
                * If `short_entries` or `short_exits` are set: Acts as `long_exits`.
            direction (Direction or array_like): See `Portfolio.from_orders`.

                Takes only effect if `short_entries` and `short_exits` are not set.
            long_entries (array_like of bool): Boolean array of long entry signals.
                Defaults to False. Will broadcast.
            long_exits (array_like of bool): Boolean array of long exit signals.
                Defaults to False. Will broadcast.
            short_entries (array_like of bool): Boolean array of short entry signals.
                Defaults to False. Will broadcast.
            short_exits (array_like of bool): Boolean array of short exit signals.
                Defaults to False. Will broadcast.
            adjust_func_nb (path_like or callable): User-defined function to adjust the current simulation state.
                Defaults to `vectorbtpro.portfolio.nb.from_signals.no_adjust_func_nb`.

                Passed as argument to `vectorbtpro.portfolio.nb.from_signals.dir_signal_func_nb`,
                `vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb`, and
                `vectorbtpro.portfolio.nb.from_signals.order_signal_func_nb`. Has no effect
                when using other signal functions.

                Can be a path to a module when using staticizing.
            adjust_args (tuple): Packed arguments passed to `adjust_func_nb`.
            signal_func_nb (path_like or callable): Function called to generate signals.

                See `vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb`.

                Can be a path to a module when using staticizing.
            signal_args (tuple): Packed arguments passed to `signal_func_nb`.
            post_segment_func_nb (path_like or callable): Post-segment function.

                See `vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb`.

                Can be a path to a module when using staticizing.
            post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
            order_mode (bool): Whether to simulate as orders without signals.
            size (float or array_like): See `Portfolio.from_orders`.

                !!! note
                    Negative size is not allowed. You must express direction using signals.
            size_type (SizeType or array_like): See `Portfolio.from_orders`.

                Only `SizeType.Amount`, `SizeType.Value`, `SizeType.Percent(100)`, and
                `SizeType.ValuePercent(100)` are supported. Other modes such as target percentage
                are not compatible with signals since their logic may contradict the direction of the signal.

                !!! note
                    `SizeType.Percent(100)` does not support position reversal. Switch to a single
                    direction or use `OppositeEntryMode.Close` to close the position first.

                See warning in `Portfolio.from_orders`.
            price (array_like of float): See `Portfolio.from_orders`.
            fees (float or array_like): See `Portfolio.from_orders`.
            fixed_fees (float or array_like): See `Portfolio.from_orders`.
            slippage (float or array_like): See `Portfolio.from_orders`.
            min_size (float or array_like): See `Portfolio.from_orders`.
            max_size (float or array_like): See `Portfolio.from_orders`.

                Will be partially filled if exceeded. You might not be able to properly close
                the position if accumulation is enabled and `max_size` is too low.
            size_granularity (float or array_like): See `Portfolio.from_orders`.
            leverage (float or array_like): See `Portfolio.from_orders`.
            leverage_mode (LeverageMode or array_like): See `Portfolio.from_orders`.
            reject_prob (float or array_like): See `Portfolio.from_orders`.
            price_area_vio_mode (PriceAreaVioMode or array_like): See `Portfolio.from_orders`.
            allow_partial (bool or array_like): See `Portfolio.from_orders`.
            raise_reject (bool or array_like): See `Portfolio.from_orders`.
            log (bool or array_like): See `Portfolio.from_orders`.
            val_price (array_like of float): See `Portfolio.from_orders`.
            accumulate (bool, AccumulationMode or array_like): See `vectorbtpro.portfolio.enums.AccumulationMode`.
                If True, becomes 'both'. If False, becomes 'disabled'. Will broadcast.

                When enabled, `Portfolio.from_signals` behaves similarly to `Portfolio.from_orders`.
            upon_long_conflict (ConflictMode or array_like): Conflict mode for long signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_short_conflict (ConflictMode or array_like): Conflict mode for short signals.
                See `vectorbtpro.portfolio.enums.ConflictMode`. Will broadcast.
            upon_dir_conflict (DirectionConflictMode or array_like): See `vectorbtpro.portfolio.enums.DirectionConflictMode`.
                Will broadcast.
            upon_opposite_entry (OppositeEntryMode or array_like): See `vectorbtpro.portfolio.enums.OppositeEntryMode`.
                Will broadcast.
            order_type (OrderType or array_like): See `vectorbtpro.portfolio.enums.OrderType`.

                Only one active limit order is allowed at a time.
            limit_delta (float or array_like): Delta from `price` to build the limit price.
                Will broadcast.

                If NaN, `price` becomes the limit price. Otherwise, applied on top of `price` depending
                on the current direction: if the direction-aware size is positive (= buying), a positive delta
                will decrease the limit price; if the direction-aware size is negative (= selling), a positive delta
                will increase the limit price. Delta can be negative.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            limit_tif (frequency_like or array_like): Time in force for limit signals.
                Will broadcast.

                Any frequency-like object is converted using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
                Any array must either contain timedeltas or integers, and will be cast into integer format
                after broadcasting. If the object provided is of data type `object`, will be converted
                to timedelta automatically.

                Measured in the distance after the open time of the signal bar. If the expiration time happens
                in the middle of the current bar, we pessimistically assume that the order has been expired.
                The check is performed at the beginning of the bar, and the first check is performed at the
                next bar after the signal. For example, if the format is `TimeDeltaFormat.Rows`, 0 or 1 means
                the order must execute at the same bar or not at all; 2 means the order must execute at the
                same or next bar or not at all.

                Set an element to `-1` to disable. Use `time_delta_format` to specify the format.
            limit_expiry (frequency_like, datetime_like, or array_like): Expiration time.
                Will broadcast.

                Any frequency-like object is used to build a period index, such that each timestamp in the original
                index is pointing to the timestamp where the period ends. For example, providing "d" will
                make any limit order expire on the next day. Any array must either contain timestamps or integers
                (not timedeltas!), and will be cast into integer format after broadcasting. If the object
                provided is of data type `object`, will be converted to datetime and its timezone will
                be removed automatically (as done on the index).

                Behaves in a similar way as `limit_tif`.

                Set an element to `-1` or `pd.Timestamp.max` to disable. Use `time_delta_format` to specify the format.
            limit_reverse (bool or array_like): Whether to reverse the price hit detection.
                Will broadcast.

                If True, a buy/sell limit price will be checked against high/low (not low/high).
                Also, the limit delta will be applied above/below (not below/above) the initial price.
            upon_adj_limit_conflict (PendingConflictMode or array_like): Conflict mode for limit and user-defined
                signals of adjacent sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            upon_opp_limit_conflict (PendingConflictMode or array_like): Conflict mode for limit and user-defined
                signals of opposite sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            use_stops (bool): Whether to use stops.
                Defaults to None, which becomes True if any of the stops are not NaN or
                the adjustment function is not the default one.

                Disable this to make simulation a bit faster for simple use cases.
            stop_ladder (bool or StopLadderMode): Whether and which kind of stop laddering to use.
                See `vectorbtpro.portfolio.enums.StopLadderMode`.

                If so, rows in the supplied arrays will become ladder steps. Make sure that
                they are increasing. If one column should have less steps, pad it with NaN
                for price-based stops and -1 for time-based stops.

                Rows in each array can be of an arbitrary length but columns must broadcast against
                the number of columns in the data. Applied on all stop types.
            sl_stop (array_like of float): Stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tsl_stop (array_like of float): Trailing stop loss for the trailing stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tsl_th (array_like of float): Take profit threshold for the trailing stop loss.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            tp_stop (array_like of float): Take profit.
                Will broadcast.

                Set an element to `np.nan` to disable. Use `delta_format` to specify the format.
            td_stop (frequency_like or array_like): Timedelta-stop.
                Will broadcast.

                Set an element to `-1` to disable. Use `time_delta_format` to specify the format.
            dt_stop (frequency_like, datetime_like, or array_like): Datetime-stop.
                Will broadcast.

                Set an element to `-1` to disable. Use `time_delta_format` to specify the format.
            stop_entry_price (StopEntryPrice or array_like): See `vectorbtpro.portfolio.enums.StopEntryPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry. If a positive value is provided,
                used directly as a price, otherwise used as an enumerated value.
            stop_exit_price (StopExitPrice or array_like): See `vectorbtpro.portfolio.enums.StopExitPrice`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry. If a positive value is provided,
                used directly as a price, otherwise used as an enumerated value.
            stop_exit_type (StopExitType or array_like): See `vectorbtpro.portfolio.enums.StopExitType`.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry.
            stop_order_type (OrderType or array_like): Similar to `order_type` but for stop orders.
                Will broadcast.

                If provided on per-element basis, gets applied upon entry.
            stop_limit_delta (float or array_like): Similar to `limit_delta` but for stop orders.
                Will broadcast.
            upon_stop_update (StopUpdateMode or array_like): See `vectorbtpro.portfolio.enums.StopUpdateMode`.
                Will broadcast.

                Only has effect if accumulation is enabled.

                If provided on per-element basis, gets applied upon repeated entry.
            upon_adj_stop_conflict (PendingConflictMode or array_like): Conflict mode for stop and user-defined
                signals of adjacent sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            upon_opp_stop_conflict (PendingConflictMode or array_like): Conflict mode for stop and user-defined
                signals of opposite sign. See `vectorbtpro.portfolio.enums.PendingConflictMode`. Will broadcast.
            delta_format (DeltaFormat or array_like): See `vectorbtpro.portfolio.enums.DeltaFormat`.
                Will broadcast.
            time_delta_format (TimeDeltaFormat or array_like): See `vectorbtpro.portfolio.enums.TimeDeltaFormat`.
                Will broadcast.
            open (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` gets replaced by `close`.
            high (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the maximum out of `open` and `close`.
            low (array_like of float): See `Portfolio.from_orders`.

                For stop signals, `np.nan` replaced by the minimum out of `open` and `close`.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            init_price (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_dividends (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): See `Portfolio.from_orders`.
            from_ago (int or array_like): See `Portfolio.from_orders`.

                Take effect only for user-defined signals, not for stop signals.
            call_seq (CallSeqType or array_like): See `Portfolio.from_orders`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            ffill_val_price (bool): See `Portfolio.from_orders`.
            update_value (bool): See `Portfolio.from_orders`.
            fill_pos_info (bool): fill_pos_info (bool): Whether to fill position record.

                Disable this to make simulation faster for simple use cases.
            save_state (bool): See `Portfolio.from_orders`.
            save_value (bool): See `Portfolio.from_orders`.
            save_returns (bool): See `Portfolio.from_orders`.
            max_orders (int): See `Portfolio.from_orders`.
            max_logs (int): See `Portfolio.from_orders`.
            in_outputs (mapping_like): Mapping with in-output objects. Only for flexible mode.

                Will be available via `Portfolio.in_outputs` as a named tuple.

                To substitute `Portfolio` attributes, provide already broadcasted and grouped objects,
                for example, by using `broadcast_named_args` and templates. Also see
                `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.

                When chunking, make sure to provide the chunk taking specification and the merging function.
                See `vectorbtpro.portfolio.chunking.merge_sim_outs`.

                !!! note
                    When using Numba below 0.54, `in_outputs` cannot be a mapping, but must be a named tuple
                    defined globally so Numba can introspect its attributes for pickling.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): Dictionary with named arguments to broadcast.

                You can then pass argument names wrapped with `vectorbtpro.utils.template.Rep`
                and this method will substitute them by their corresponding broadcasted objects.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_context (mapping): Mapping to replace templates in arguments.
            jitted (any): See `Portfolio.from_orders`.
            chunked (any): See `Portfolio.from_orders`.
            staticized (bool, dict, hashable, or callable): Keyword arguments or task id for staticizing.

                If True or dictionary, will be passed as keyword arguments to
                `vectorbtpro.utils.cutting.cut_and_save_func` to save a cacheable version of the
                simulator to a file. If a hashable or callable, will be used as a task id of an
                already registered jittable and chunkable simulator. Dictionary allows additional options
                `override` and `reload` to override and reload an already existing module respectively.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            records (array_like): See `Portfolio.from_orders`.
            return_preparer (bool): See `Portfolio.from_orders`.
            return_prep_result (bool): See `Portfolio.from_orders`.
            return_sim_out (bool): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        All broadcastable arguments will broadcast using `vectorbtpro.base.reshaping.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        Also see notes and hints for `Portfolio.from_orders`.

        Usage:
            * By default, if all signal arrays are None, `entries` becomes True,
            which opens a position at the very first tick and does nothing else:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_signals(close, size=1)
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * Entry opens long, exit closes long:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='longonly'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=pd.Series([False, False, True, True, True]),  # long_exits
            ...     short_entries=False,
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64
            ```

            Notice how both `short_entries` and `short_exits` are provided as constants - as any other
            broadcastable argument, they are treated as arrays where each element is False.

            * Entry opens short, exit closes short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='shortonly'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=False,  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([True, True, True, False, False]),
            ...     short_exits=pd.Series([False, False, True, True, True]),
            ...     size=1
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64
            ```

            * Entry opens long and closes short, exit closes long and opens short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([False, False, True, True, True]),
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64
            ```

            * More complex signal combinations are best expressed using direction-aware arrays.
            For example, ignore opposite signals as long as the current position is open:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries      =pd.Series([True, False, False, False, False]),  # long_entries
            ...     exits        =pd.Series([False, False, True, False, False]),  # long_exits
            ...     short_entries=pd.Series([False, True, False, True, False]),
            ...     short_exits  =pd.Series([False, False, False, False, True]),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            * First opposite signal closes the position, second one opens a new position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both',
            ...     upon_opposite_entry='close'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * If both long entry and exit signals are True (a signal conflict), choose exit:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='longonly',
            ...     upon_long_conflict='exit')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * If both long entry and short entry signal are True (a direction conflict), choose short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     upon_dir_conflict='short')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -2.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            !!! note
                Remember that when direction is set to 'both', entries become `long_entries` and exits become
                `short_entries`, so this becomes a conflict of directions rather than signals.

            * If there are both signal and direction conflicts:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=True,  # long_entries
            ...     exits=True,  # long_exits
            ...     short_entries=True,
            ...     short_exits=True,
            ...     size=1,
            ...     upon_long_conflict='entry',
            ...     upon_short_conflict='entry',
            ...     upon_dir_conflict='short'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * Turn on accumulation of signals. Entry means long order, exit means short order
            (acts similar to `from_orders`):

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate=True)
            >>> pf.asset_flow
            0    1.0
            1    1.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * Allow increasing a position (of any direction), deny decreasing a position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate='addonly')
            >>> pf.asset_flow
            0    1.0  << open a long position
            1    1.0  << add to the position
            2    0.0
            3   -3.0  << close and open a short position
            4   -1.0  << add to the position
            dtype: float64
            ```

            * Test multiple parameters via regular broadcasting:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=[list(Direction)],
            ...     broadcast_kwargs=dict(columns_from=pd.Index(vbt.pf_enums.Direction._fields, name='direction')))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            * Test multiple parameters via `vectorbtpro.base.reshaping.BCO`:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=vbt.Param(Direction))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            * Set risk/reward ratio by passing trailing stop loss and take profit thresholds:

            ```pycon
            >>> close = pd.Series([10, 11, 12, 11, 10, 9])
            >>> entries = pd.Series([True, False, False, False, False, False])
            >>> exits = pd.Series([False, False, False, False, False, True])
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.2)  # take profit hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.3)  # trailing stop loss hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4   -10.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=np.inf, tp_stop=np.inf)  # nothing hit, exit as usual
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4     0.0
            5   -10.0
            dtype: float64
            ```

            * Test different stop combinations:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=vbt.Param([0.1, 0.2]),
            ...     tp_stop=vbt.Param([0.2, 0.3])
            ... )
            >>> pf.asset_flow
            tsl_stop   0.1         0.2
            tp_stop    0.2   0.3   0.2   0.3
            0         10.0  10.0  10.0  10.0
            1          0.0   0.0   0.0   0.0
            2        -10.0   0.0 -10.0   0.0
            3          0.0   0.0   0.0   0.0
            4          0.0 -10.0   0.0   0.0
            5          0.0   0.0   0.0 -10.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            * We can implement our own stop loss or take profit, or adjust the existing one at each time step.
            Let's implement [stepped stop-loss](https://www.freqtrade.io/en/stable/strategy-advanced/#stepped-stoploss):

            ```pycon
            >>> @njit
            ... def adjust_func_nb(c):
            ...     val_price_now = c.last_val_price[c.col]
            ...     tsl_init_price = c.last_tsl_info["init_price"][c.col]
            ...     current_profit = (val_price_now - tsl_init_price) / tsl_init_price
            ...     if current_profit >= 0.40:
            ...         c.last_tsl_info["stop"][c.col] = 0.25
            ...     elif current_profit >= 0.25:
            ...         c.last_tsl_info["stop"][c.col] = 0.15
            ...     elif current_profit >= 0.20:
            ...         c.last_tsl_info["stop"][c.col] = 0.07

            >>> close = pd.Series([10, 11, 12, 11, 10])
            >>> pf = vbt.Portfolio.from_signals(close, adjust_func_nb=adjust_func_nb)
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0  # 7% from 12 hit
            4    11.16
            dtype: float64
            ```

            * Sometimes there is a need to provide or transform signals dynamically. For this, we can implement
            a custom signal function `signal_func_nb`. For example, let's implement a signal function that
            takes two numerical arrays - long and short one - and transforms them into 4 direction-aware boolean
            arrays that vectorbt understands:

            ```pycon
            >>> @njit
            ... def signal_func_nb(c, long_num_arr, short_num_arr):
            ...     long_num = vbt.pf_nb.select_nb(c, long_num_arr)
            ...     short_num = vbt.pf_nb.select_nb(c, short_num_arr)
            ...     is_long_entry = long_num > 0
            ...     is_long_exit = long_num < 0
            ...     is_short_entry = short_num > 0
            ...     is_short_exit = short_num < 0
            ...     return is_long_entry, is_long_exit, is_short_entry, is_short_exit

            >>> pf = vbt.Portfolio.from_signals(
            ...     pd.Series([1, 2, 3, 4, 5]),
            ...     signal_func_nb=signal_func_nb,
            ...     signal_args=(vbt.Rep('long_num_arr'), vbt.Rep('short_num_arr')),
            ...     broadcast_named_args=dict(
            ...         long_num_arr=pd.Series([1, 0, -1, 0, 0]),
            ...         short_num_arr=pd.Series([0, 1, 0, 1, -1])
            ...     ),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            Passing both arrays as `broadcast_named_args` broadcasts them internally as any other array,
            so we don't have to worry about their dimensions every time we change our data.
        """
        if isinstance(close, FSPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            if isinstance(close, Data):
                local_kwargs["data"] = close
                local_kwargs["close"] = None
            preparer = FSPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_holding(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        direction: tp.Optional[int] = None,
        at_first_valid_in: tp.Optional[str] = "close",
        close_at_end: tp.Optional[bool] = None,
        dynamic_mode: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from plain holding using signals.

        If `close_at_end` is True, will place an opposite signal at the very end.

        `**kwargs` are passed to the class method `Portfolio.from_signals`."""
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if direction is None:
            direction = portfolio_cfg["hold_direction"]
        direction = map_enum_fields(direction, enums.Direction)
        if not checks.is_int(direction):
            raise TypeError("Direction must be a scalar")
        if close_at_end is None:
            close_at_end = portfolio_cfg["close_at_end"]

        if dynamic_mode:
            return cls.from_signals(
                close,
                signal_func_nb=nb.holding_enex_signal_func_nb,
                signal_args=(direction, close_at_end),
                accumulate=False,
                **kwargs,
            )

        def _entries(wrapper, new_objs):
            if at_first_valid_in is None:
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            ts = new_objs[at_first_valid_in]
            valid_index = generic_nb.first_valid_index_nb(ts)
            if (valid_index == -1).all():
                return np.array([[False]])
            if (valid_index == 0).all():
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            entries = np.full(wrapper.shape_2d, False)
            entries[valid_index, np.arange(wrapper.shape_2d[1])] = True
            return entries

        def _exits(wrapper):
            if close_at_end:
                exits = np.full((wrapper.shape_2d[0], 1), False)
                exits[-1] = True
            else:
                exits = np.array([[False]])
            return exits

        return cls.from_signals(
            close,
            entries=RepFunc(_entries),
            exits=RepFunc(_exits),
            direction=direction,
            accumulate=False,
            **kwargs,
        )

    @classmethod
    def from_random_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        n: tp.Optional[tp.ArrayLike] = None,
        prob: tp.Optional[tp.ArrayLike] = None,
        entry_prob: tp.Optional[tp.ArrayLike] = None,
        exit_prob: tp.Optional[tp.ArrayLike] = None,
        param_product: bool = False,
        seed: tp.Optional[int] = None,
        run_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from random entry and exit signals.

        Generates signals based either on the number of signals `n` or the probability
        of encountering a signal `prob`.

        * If `n` is set, see `vectorbtpro.signals.generators.RANDNX`.
        * If `prob` is set, see `vectorbtpro.signals.generators.RPROBNX`.

        Based on `Portfolio.from_signals`.

        !!! note
            To generate random signals, the shape of `close` is used. Broadcasting with other
            arrays happens after the generation.

        Usage:
            * Test multiple combinations of random entries and exits:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_random_signals(close, n=[2, 1, 0], seed=42)
            >>> pf.orders.count()
            randnx_n
            2    4
            1    2
            0    0
            Name: count, dtype: int64
            ```

            * Test the Cartesian product of entry and exit encounter probabilities:

            ```pycon
            >>> pf = vbt.Portfolio.from_random_signals(
            ...     close,
            ...     entry_prob=[0, 0.5, 1],
            ...     exit_prob=[0, 0.5, 1],
            ...     param_product=True,
            ...     seed=42)
            >>> pf.orders.count()
            rprobnx_entry_prob  rprobnx_exit_prob
            0.0                 0.0                  0
                                0.5                  0
                                1.0                  0
            0.5                 0.0                  1
                                0.5                  4
                                1.0                  3
            1.0                 0.0                  1
                                0.5                  4
                                1.0                  5
            Name: count, dtype: int64
            ```
        """
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if isinstance(close, Data):
            data = close
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            close_wrapper = data.symbol_wrapper
        else:
            close = to_pd_array(close)
            close_wrapper = ArrayWrapper.from_obj(close)
            data = close
        if entry_prob is None:
            entry_prob = prob
        if exit_prob is None:
            exit_prob = prob
        if seed is None:
            seed = portfolio_cfg["seed"]
        if run_kwargs is None:
            run_kwargs = {}

        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or entry_prob and exit_prob must be provided")
        if n is not None:
            rand = RANDNX.run(
                n=n,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rand.entries
            exits = rand.exits
        elif entry_prob is not None and exit_prob is not None:
            rprobnx = RPROBNX.run(
                entry_prob=entry_prob,
                exit_prob=exit_prob,
                param_product=param_product,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rprobnx.entries
            exits = rprobnx.exits
        else:
            raise ValueError("At least n or entry_prob and exit_prob must be provided")

        return cls.from_signals(data, entries, exits, seed=seed, **kwargs)

    @classmethod
    def from_optimizer(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data],
        optimizer: PortfolioOptimizer,
        pf_method: str = "from_orders",
        squeeze_groups: bool = True,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        size_type: tp.ArrayLike = "targetpercent",
        direction: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = True,
        call_seq: tp.Optional[tp.ArrayLike] = "auto",
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> PortfolioResultT:
        """Build portfolio from an optimizer of type `vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer`.

        Uses `Portfolio.from_orders` as the base simulation method.

        The size type is 'targetpercent'. If there are positive and negative values, the direction
        is automatically set to 'both', otherwise to 'longonly' for positive-only and `shortonly`
        for negative-only values. Also, the cash sharing is set to True, the call sequence is set
        to 'auto', and the grouper is set to the grouper of the optimizer by default.

        Usage:
            ```pycon
            >>> close = pd.DataFrame({
            ...     "MSFT": [1, 2, 3, 4, 5],
            ...     "GOOG": [5, 4, 3, 2, 1],
            ...     "AAPL": [1, 2, 3, 2, 1]
            ... }, index=pd.date_range(start="2020-01-01", periods=5))

            >>> pf_opt = vbt.PortfolioOptimizer.from_random(
            ...     close.vbt.wrapper,
            ...     every="2D",
            ...     seed=42
            ... )
            >>> pf_opt.fill_allocations()
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812
            2020-01-02        NaN       NaN       NaN
            2020-01-03   0.657381  0.171323  0.171296
            2020-01-04        NaN       NaN       NaN
            2020-01-05   0.038078  0.567845  0.394077

            >>> pf = vbt.Portfolio.from_optimizer(close, pf_opt)
            >>> pf.get_asset_value(group_by=False).vbt / pf.value
            alloc_group                         group
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812  << rebalanced
            2020-01-02   0.251907  0.255771  0.492322
            2020-01-03   0.657381  0.171323  0.171296  << rebalanced
            2020-01-04   0.793277  0.103369  0.103353
            2020-01-05   0.038078  0.567845  0.394077  << rebalanced
            ```
        """
        size = optimizer.fill_allocations(squeeze_groups=squeeze_groups, dropna=dropna, fill_value=fill_value)
        if direction is None:
            pos_size_any = (size.values > 0).any()
            neg_size_any = (size.values < 0).any()
            if pos_size_any and neg_size_any:
                direction = "both"
            elif pos_size_any:
                direction = "longonly"
            else:
                direction = "shortonly"
                size = size.abs()

        if group_by is None:

            def _substitute_group_by(index):
                columns = optimizer.wrapper.columns
                if squeeze_groups and optimizer.wrapper.grouped_ndim == 1:
                    columns = columns.droplevel(level=0)
                if not index.equals(columns):
                    if "symbol" in index.names:
                        return ExceptLevel("symbol")
                    raise ValueError("Column hierarchy has changed. Disable squeeze_groups and provide group_by.")
                return optimizer.wrapper.grouper.group_by

            group_by = RepFunc(_substitute_group_by)

        if pf_method.lower() == "from_orders":
            return cls.from_orders(
                close,
                size=size,
                size_type=size_type,
                direction=direction,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                group_by=group_by,
                **kwargs,
            )
        elif pf_method.lower() == "from_signals":
            return cls.from_signals(
                close,
                order_mode=True,
                size=size,
                size_type=size_type,
                direction=direction,
                accumulate=True,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                group_by=group_by,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid option pf_method='{pf_method}'")

    @classmethod
    def from_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data, FOFPreparer, PFPrepResult],
        *,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        segment_mask: tp.Optional[tp.ArrayLike] = None,
        call_pre_segment: tp.Optional[bool] = None,
        call_post_segment: tp.Optional[bool] = None,
        pre_sim_func_nb: tp.Optional[nb.PreSimFuncT] = None,
        pre_sim_args: tp.Args = (),
        post_sim_func_nb: tp.Optional[nb.PostSimFuncT] = None,
        post_sim_args: tp.Args = (),
        pre_group_func_nb: tp.Optional[nb.PreGroupFuncT] = None,
        pre_group_args: tp.Args = (),
        post_group_func_nb: tp.Optional[nb.PostGroupFuncT] = None,
        post_group_args: tp.Args = (),
        pre_row_func_nb: tp.Optional[nb.PreRowFuncT] = None,
        pre_row_args: tp.Args = (),
        post_row_func_nb: tp.Optional[nb.PostRowFuncT] = None,
        post_row_args: tp.Args = (),
        pre_segment_func_nb: tp.Optional[nb.PreSegmentFuncT] = None,
        pre_segment_args: tp.Args = (),
        post_segment_func_nb: tp.Optional[nb.PostSegmentFuncT] = None,
        post_segment_args: tp.Args = (),
        order_func_nb: tp.Optional[nb.OrderFuncT] = None,
        order_args: tp.Args = (),
        flex_order_func_nb: tp.Optional[nb.FlexOrderFuncT] = None,
        flex_order_args: tp.Args = (),
        post_order_func_nb: tp.Optional[nb.PostOrderFuncT] = None,
        post_order_args: tp.Args = (),
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_pos_info: tp.Optional[bool] = None,
        track_value: tp.Optional[bool] = None,
        row_wise: tp.Optional[bool] = None,
        max_orders: tp.Optional[int] = None,
        max_logs: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        keep_inout_flex: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        staticized: tp.StaticizedOption = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Build portfolio from a custom order function.

        !!! hint
            See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb` for illustrations and argument definitions.

        For more details on individual simulation functions:

        * `order_func_nb`: See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb`
        * `order_func_nb` and `row_wise`: See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_rw_nb`
        * `flex_order_func_nb`: See `vectorbtpro.portfolio.nb.from_order_func.from_flex_order_func_nb`
        * `flex_order_func_nb` and `row_wise`: See `vectorbtpro.portfolio.nb.from_order_func.from_flex_order_func_rw_nb`

        Prepared by `vectorbtpro.portfolio.preparing.FOFPreparer`.

        Args:
            close (array_like, Data, FOFPreparer, or PFPrepResult): Latest asset price at each time step.
                Will broadcast.

                If an instance of `vectorbtpro.data.base.Data`, will extract the open, high,
                low, and close price.

                Used for calculating unrealized PnL and portfolio value.
            init_cash (InitCashMode, float or array_like): See `Portfolio.from_orders`.
            init_position (float or array_like): See `Portfolio.from_orders`.
            init_price (float or array_like): See `Portfolio.from_orders`.
            cash_deposits (float or array_like): See `Portfolio.from_orders`.
            cash_earnings (float or array_like): See `Portfolio.from_orders`.
            cash_sharing (bool): Whether to share cash within the same group.

                If `group_by` is None, `group_by` becomes True to form a single group with cash sharing.
            call_seq (CallSeqType or array_like): Default sequence of calls per row and group.

                * Use `vectorbtpro.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

                !!! note
                    CallSeqType.Auto must be implemented manually.
                    Use `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_1d_nb`
                    or `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_out_1d_nb` in `pre_segment_func_nb`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            segment_mask (int or array_like of bool): Mask of whether a particular segment should be executed.

                Supplying an integer will activate every n-th row.
                Supplying a boolean or an array of boolean will broadcast to the number of rows and groups.

                Does not broadcast together with `close` and `broadcast_named_args`, only against the final shape.
            call_pre_segment (bool): Whether to call `pre_segment_func_nb` regardless of `segment_mask`.
            call_post_segment (bool): Whether to call `post_segment_func_nb` regardless of `segment_mask`.
            pre_sim_func_nb (callable): Function called before simulation.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_sim_args (tuple): Packed arguments passed to `pre_sim_func_nb`.
            post_sim_func_nb (callable): Function called after simulation.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_sim_args (tuple): Packed arguments passed to `post_sim_func_nb`.
            pre_group_func_nb (callable): Function called before each group.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.

                Called only if `row_wise` is False.
            pre_group_args (tuple): Packed arguments passed to `pre_group_func_nb`.
            post_group_func_nb (callable): Function called after each group.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.

                Called only if `row_wise` is False.
            post_group_args (tuple): Packed arguments passed to `post_group_func_nb`.
            pre_row_func_nb (callable): Function called before each row.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.

                Called only if `row_wise` is True.
            pre_row_args (tuple): Packed arguments passed to `pre_row_func_nb`.
            post_row_func_nb (callable): Function called after each row.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.

                Called only if `row_wise` is True.
            post_row_args (tuple): Packed arguments passed to `post_row_func_nb`.
            pre_segment_func_nb (callable): Function called before each segment.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_segment_args (tuple): Packed arguments passed to `pre_segment_func_nb`.
            post_segment_func_nb (callable): Function called after each segment.
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
            order_func_nb (callable): Order generation function.
            order_args: Packed arguments passed to `order_func_nb`.
            flex_order_func_nb (callable): Flexible order generation function.
            flex_order_args: Packed arguments passed to `flex_order_func_nb`.
            post_order_func_nb (callable): Callback that is called after the order has been processed.
            post_order_args (tuple): Packed arguments passed to `post_order_func_nb`.
            open (array_like of float): See `Portfolio.from_orders`.
            high (array_like of float): See `Portfolio.from_orders`.
            low (array_like of float): See `Portfolio.from_orders`.
            ffill_val_price (bool): Whether to track valuation price only if it's known.

                Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp.
            update_value (bool): Whether to update group value after each filled order.
            fill_pos_info (bool): Whether to fill position record.

                Disable this to make simulation faster for simple use cases.
            track_value (bool): Whether to track value metrics such as
                the current valuation price, value, and return.

                Disable this to make simulation faster for simple use cases.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.
            max_orders (int): The max number of order records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape.

                Set to a lower number if you run out of memory, to 0 to not fill, and to a higher number
                if there are more than one order expected at each timestamp.
            max_logs (int): The max number of log records expected to be filled at each column.
                Defaults to the number of rows in the broadcasted shape.

                Set to a lower number if you run out of memory, to 0 to not fill, and to a higher number
                if there are more than one order expected at each timestamp.
            in_outputs (mapping_like): Mapping with in-output objects.

                Will be available via `Portfolio.in_outputs` as a named tuple.

                To substitute `Portfolio` attributes, provide already broadcasted and grouped objects,
                for example, by using `broadcast_named_args` and templates. Also see
                `Portfolio.in_outputs_indexing_func` on how in-output objects are indexed.

                When chunking, make sure to provide the chunk taking specification and the merging function.
                See `vectorbtpro.portfolio.chunking.merge_sim_outs`.

                !!! note
                    When using Numba below 0.54, `in_outputs` cannot be a mapping, but must be a named tuple
                    defined globally so Numba can introspect its attributes for pickling.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): See `Portfolio.from_signals`.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_context (mapping): See `Portfolio.from_signals`.
            keep_inout_flex (bool): Whether to keep arrays that can be edited in-place raw when broadcasting.

                Disable this to be able to edit `segment_mask`, `cash_deposits`, and
                `cash_earnings` during the simulation.
            jitted (any): See `Portfolio.from_orders`.

                !!! note
                    Disabling jitting will not disable jitter (such as Numba) on other functions,
                    only on the main (simulation) function. If neccessary, you should ensure that every other
                    function is not compiled as well. For example, when working with Numba, you can do this
                    by using the `py_func` attribute of that function. Or, you can disable Numba
                    entirely by running `os.environ['NUMBA_DISABLE_JIT'] = '1'` before importing vectorbtpro.

                !!! warning
                    Parallelization assumes that groups are independent and there is no data flowing between them.
            chunked (any): See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            staticized (bool, dict, hashable, or callable): Keyword arguments or task id for staticizing.

                If True or dictionary, will be passed as keyword arguments to
                `vectorbtpro.utils.cutting.cut_and_save_func` to save a cacheable version of the
                simulator to a file. If a hashable or callable, will be used as a task id of an
                already registered jittable and chunkable simulator. Dictionary allows additional options
                `override` and `reload` to override and reload an already existing module respectively.
            freq (any): See `Portfolio.from_orders`.
            bm_close (array_like): See `Portfolio.from_orders`.
            records (array_like): See `Portfolio.from_orders`.
            return_preparer (bool): See `Portfolio.from_orders`.
            return_prep_result (bool): See `Portfolio.from_orders`.
            return_sim_out (bool): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `Portfolio` constructor.

        For defaults, see `vectorbtpro._settings.portfolio`. Those defaults are not used to fill
        NaN values after reindexing: vectorbt uses its own sensible defaults, which are usually NaN
        for floating arrays and default flags for integer arrays. Use `vectorbtpro.base.reshaping.BCO`
        with `fill_value` to override.

        !!! note
            All passed functions must be Numba-compiled if Numba is enabled.

            Also see notes on `Portfolio.from_orders`.

        !!! note
            In contrast to other methods, the valuation price is previous `close` instead of the order price
            since the price of an order is unknown before the call (which is more realistic by the way).
            You can still override the valuation price in `pre_segment_func_nb`.

        Usage:
            * Buy 10 units each tick using closing price:

            ```pycon
            >>> @njit
            ... def order_func_nb(c, size):
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     order_args=(10,)
            ... )

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Reverse each position by first closing it. Keep state of last position to determine
            which position to open next (just as an example, there are easier ways to do this):

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     last_pos_state = np.array([-1])
            ...     return (last_pos_state,)

            >>> @njit
            ... def order_func_nb(c, last_pos_state):
            ...     if c.position_now != 0:
            ...         return vbt.pf_nb.close_position_nb()
            ...
            ...     if last_pos_state[0] == 1:
            ...         size = -np.inf  # open short
            ...         last_pos_state[0] = -1
            ...     else:
            ...         size = np.inf  # open long
            ...         last_pos_state[0] = 1
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     pre_group_func_nb=pre_group_func_nb
            ... )

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * Equal-weighted portfolio as in the example under `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb`:

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     order_value_out = np.empty(c.group_len, dtype=np.float_)
            ...     return (order_value_out,)

            >>> @njit
            ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.last_val_price[col] = vbt.pf_nb.select_from_col_nb(c, col, price)
            ...     vbt.pf_nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            ...     return ()

            >>> @njit
            ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
            ...     return vbt.pf_nb.order_nb(
            ...         size=vbt.pf_nb.select_nb(c, size),
            ...         price=vbt.pf_nb.select_nb(c, price),
            ...         size_type=vbt.pf_nb.select_nb(c, size_type),
            ...         direction=vbt.pf_nb.select_nb(c, direction),
            ...         fees=vbt.pf_nb.select_nb(c, fees),
            ...         fixed_fees=vbt.pf_nb.select_nb(c, fixed_fees),
            ...         slippage=vbt.pf_nb.select_nb(c, slippage)
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> size_template = vbt.RepEval('np.array([[1 / group_lens[0]]])')

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     order_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction'),
            ...         vbt.Rep('fees'),
            ...         vbt.Rep('fixed_fees'),
            ...         vbt.Rep('slippage'),
            ...     ),
            ...     segment_mask=2,  # rebalance every second tick
            ...     pre_group_func_nb=pre_group_func_nb,
            ...     pre_segment_func_nb=pre_segment_func_nb,
            ...     pre_segment_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction')
            ...     ),
            ...     broadcast_named_args=dict(  # broadcast against each other
            ...         price=close,
            ...         size_type=vbt.pf_enums.SizeType.TargetPercent,
            ...         direction=vbt.pf_enums.Direction.LongOnly,
            ...         fees=0.001,
            ...         fixed_fees=1.,
            ...         slippage=0.001
            ...     ),
            ...     template_context=dict(np=np),  # required by size_template
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_nb_example.svg){: .iimg loading=lazy }

            Templates are a very powerful tool to prepare any custom arguments after they are broadcast and
            before they are passed to the simulation function. In the example above, we use `broadcast_named_args`
            to broadcast some arguments against each other and templates to pass those objects to callbacks.
            Additionally, we used an evaluation template to compute the size based on the number of assets in each group.

            You may ask: why should we bother using broadcasting and templates if we could just pass `size=1/3`?
            Because of flexibility those features provide: we can now pass whatever parameter combinations we want
            and it will work flawlessly. For example, to create two groups of equally-allocated positions,
            we need to change only two parameters:

            ```pycon
            >>> close = np.random.uniform(1, 10, size=(5, 6))  # 6 columns instead of 3
            >>> group_by = ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']  # 2 groups instead of 1
            >>> # Replace close and group_by in the example above

            >>> pf['g1'].get_asset_value(group_by=False).vbt.plot()
            >>> pf['g2'].get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_g1.svg){: .iimg loading=lazy }

            ![](/assets/images/api/from_order_func_g2.svg){: .iimg loading=lazy }

            * Combine multiple exit conditions. Exit early if the price hits some threshold before an actual exit:

            ```pycon
            >>> @njit
            ... def pre_sim_func_nb(c):
            ...     # We need to define stop price per column once
            ...     stop_price = np.full(c.target_shape[1], np.nan, dtype=np.float_)
            ...     return (stop_price,)

            >>> @njit
            ... def order_func_nb(c, stop_price, entries, exits, size):
            ...     # Select info related to this order
            ...     entry_now = vbt.pf_nb.select_nb(c, entries)
            ...     exit_now = vbt.pf_nb.select_nb(c, exits)
            ...     size_now = vbt.pf_nb.select_nb(c, size)
            ...     price_now = vbt.pf_nb.select_nb(c, c.close)
            ...     stop_price_now = stop_price[c.col]
            ...
            ...     # Our logic
            ...     if entry_now:
            ...         if c.position_now == 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     elif exit_now or price_now >= stop_price_now:
            ...         if c.position_now > 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=-size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     return vbt.pf_enums.NoOrder

            >>> @njit
            ... def post_order_func_nb(c, stop_price, stop):
            ...     # Same broadcasting as for size
            ...     stop_now = vbt.pf_nb.select_nb(c, stop)
            ...
            ...     if c.order_result.status == vbt.pf_enums.OrderStatus.Filled:
            ...         if c.order_result.side == vbt.pf_enums.OrderSide.Buy:
            ...             # Position entered: Set stop condition
            ...             stop_price[c.col] = (1 + stop_now) * c.order_result.price
            ...         else:
            ...             # Position exited: Remove stop condition
            ...             stop_price[c.col] = np.nan

            >>> def simulate(close, entries, exits, size, stop):
            ...     return vbt.Portfolio.from_order_func(
            ...         close,
            ...         order_func_nb=order_func_nb,
            ...         order_args=(vbt.Rep('entries'), vbt.Rep('exits'), vbt.Rep('size')),
            ...         pre_sim_func_nb=pre_sim_func_nb,
            ...         post_order_func_nb=post_order_func_nb,
            ...         post_order_args=(vbt.Rep('stop'),),
            ...         broadcast_named_args=dict(  # broadcast against each other
            ...             entries=entries,
            ...             exits=exits,
            ...             size=size,
            ...             stop=stop
            ...         )
            ...     )

            >>> close = pd.Series([10, 11, 12, 13, 14])
            >>> entries = pd.Series([True, True, False, False, False])
            >>> exits = pd.Series([False, False, False, True, True])
            >>> simulate(close, entries, exits, np.inf, 0.1).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, 0.2).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, np.nan).asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0
            4     0.0
            dtype: float64
            ```

            The reason why stop of 10% does not result in an order at the second time step is because
            it comes at the same time as entry, so it must wait until no entry is present.
            This can be changed by replacing the statement "elif" with "if", which would execute
            an exit regardless if an entry is present (similar to using `ConflictMode.Opposite` in
            `Portfolio.from_signals`).

            We can also test the parameter combinations above all at once (thanks to broadcasting
            using `vectorbtpro.base.reshaping.broadcast`):

            ```pycon
            >>> stop = pd.DataFrame([[0.1, 0.2, np.nan]])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
                  0     1     2
            0  10.0  10.0  10.0
            1   0.0   0.0   0.0
            2 -10.0 -10.0   0.0
            3   0.0   0.0 -10.0
            4   0.0   0.0   0.0
            ```

            Or much simpler using Cartesian product:

            ```pycon
            >>> stop = vbt.Param([0.1, 0.2, np.nan])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
            threshold   0.1   0.2   NaN
            0          10.0  10.0  10.0
            1           0.0   0.0   0.0
            2         -10.0 -10.0   0.0
            3           0.0   0.0 -10.0
            4           0.0   0.0   0.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            * Let's illustrate how to generate multiple orders per symbol and bar.
            For each bar, buy at open and sell at close:

            ```pycon
            >>> @njit
            ... def flex_order_func_nb(c, size):
            ...     if c.call_idx == 0:
            ...         return c.from_col, vbt.pf_nb.order_nb(size=size, price=c.open[c.i, c.from_col])
            ...     if c.call_idx == 1:
            ...         return c.from_col, vbt.pf_nb.close_position_nb(price=c.close[c.i, c.from_col])
            ...     return -1, vbt.pf_enums.NoOrder

            >>> open = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> close = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 4, 5]})
            >>> size = 1
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     flex_order_func_nb=flex_order_func_nb,
            ...     flex_order_args=(size,),
            ...     open=open,
            ...     max_orders=close.shape[0] * 2
            ... )

            >>> pf.orders.records_readable
                Order Id Column  Timestamp  Size  Price  Fees  Side
            0          0      a          0   1.0    1.0   0.0   Buy
            1          1      a          0   1.0    2.0   0.0  Sell
            2          2      a          1   1.0    2.0   0.0   Buy
            3          3      a          1   1.0    3.0   0.0  Sell
            4          4      a          2   1.0    3.0   0.0   Buy
            5          5      a          2   1.0    4.0   0.0  Sell
            6          0      b          0   1.0    4.0   0.0   Buy
            7          1      b          0   1.0    3.0   0.0  Sell
            8          2      b          1   1.0    5.0   0.0   Buy
            9          3      b          1   1.0    4.0   0.0  Sell
            10         4      b          2   1.0    6.0   0.0   Buy
            11         5      b          2   1.0    5.0   0.0  Sell
            ```

            !!! warning
                Each bar is effectively a black box - we don't know how the price moves in-between.
                Since trades should come in an order that closely replicates that of the real world, the only
                pieces of information that always remain in the correct order are the opening and closing price.
        """
        if isinstance(close, FOFPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            if isinstance(close, Data):
                local_kwargs["data"] = close
                local_kwargs["close"] = None
            preparer = FOFPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_def_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, Data, FDOFPreparer, PFPrepResult],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        pre_segment_func_nb: tp.Optional[nb.PreSegmentFuncT] = None,
        order_func_nb: tp.Optional[nb.OrderFuncT] = None,
        flex_order_func_nb: tp.Optional[nb.FlexOrderFuncT] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        flexible: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        chunked: tp.ChunkedOption = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Build portfolio from the default order function.

        Default order function takes size, price, fees, and other available information, and issues
        an order at each column and time step. Additionally, it uses a segment preprocessing function
        that overrides the valuation price and sorts the call sequence. This way, it behaves similarly to
        `Portfolio.from_orders`, but allows injecting pre- and postprocessing functions to have more
        control over the execution. It also knows how to chunk each argument. The only disadvantage is
        that `Portfolio.from_orders` is more optimized towards performance (up to 5x).

        If `flexible` is True:

        * `pre_segment_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_flex_pre_segment_func_nb`
        * `flex_order_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_flex_order_func_nb`

        If `flexible` is False:

        * `pre_segment_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_pre_segment_func_nb`
        * `order_func_nb` is `vectorbtpro.portfolio.nb.from_order_func.def_order_func_nb`

        Prepared by `vectorbtpro.portfolio.preparing.FDOFPreparer`.

        For details on other arguments, see `Portfolio.from_orders` and `Portfolio.from_order_func`.

        Usage:
            * Working with `Portfolio.from_def_order_func` is a similar experience as working
            with `Portfolio.from_orders`:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_def_order_func(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Equal-weighted portfolio as in the example under `Portfolio.from_order_func`
            but much less verbose and with asset value pre-computed during the simulation (= faster):

            ```pycon
            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))

            >>> @njit
            ... def post_segment_func_nb(c):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.in_outputs.asset_value_pc[c.i, col] = c.last_position[col] * c.last_val_price[col]

            >>> pf = vbt.Portfolio.from_def_order_func(
            ...     close,
            ...     size=1/3,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     fees=0.001,
            ...     fixed_fees=1.,
            ...     slippage=0.001,
            ...     segment_mask=2,
            ...     cash_sharing=True,
            ...     group_by=True,
            ...     call_seq='auto',
            ...     post_segment_func_nb=post_segment_func_nb,
            ...     call_post_segment=True,
            ...     in_outputs=dict(asset_value_pc=vbt.RepEval('np.empty_like(close)'))
            ... )

            >>> asset_value = pf.wrapper.wrap(pf.in_outputs.asset_value_pc, group_by=False)
            >>> asset_value.vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_nb_example.svg){: .iimg loading=lazy }
        """
        if isinstance(close, FDOFPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            if isinstance(close, Data):
                local_kwargs["data"] = close
                local_kwargs["close"] = None
            preparer = FDOFPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    # ############# Grouping ############# #

    def regroup(self: PortfolioT, group_by: tp.GroupByLike, **kwargs) -> PortfolioT:
        """Regroup this object.

        See `vectorbtpro.base.wrapping.Wrapping.regroup`.

        !!! note
            All cached objects will be lost."""
        if self.cash_sharing:
            if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                raise ValueError("Cannot modify grouping globally when cash_sharing=True")
        return Wrapping.regroup(self, group_by, **kwargs)

    # ############# Properties ############# #

    @property
    def cash_sharing(self) -> bool:
        """Whether to share cash within the same group."""
        return self._cash_sharing

    @property
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Named tuple with in-output objects."""
        return self._in_outputs

    @property
    def use_in_outputs(self) -> bool:
        """Whether to return in-output objects when calling properties."""
        return self._use_in_outputs

    @property
    def fillna_close(self) -> bool:
        """Whether to forward-backward fill NaN values in `Portfolio.close`."""
        return self._fillna_close

    @property
    def trades_type(self) -> int:
        """Default `vectorbtpro.portfolio.trades.Trades` to use across `Portfolio`."""
        return self._trades_type

    @property
    def orders_cls(self) -> type:
        """Class for wrapping order records."""
        if self._orders_cls is None:
            return Orders
        return self._orders_cls

    @property
    def logs_cls(self) -> type:
        """Class for wrapping log records."""
        if self._logs_cls is None:
            return Logs
        return self._logs_cls

    @property
    def trades_cls(self) -> type:
        """Class for wrapping trade records."""
        if self._trades_cls is None:
            return Trades
        return self._trades_cls

    @property
    def entry_trades_cls(self) -> type:
        """Class for wrapping entry trade records."""
        if self._entry_trades_cls is None:
            return EntryTrades
        return self._entry_trades_cls

    @property
    def exit_trades_cls(self) -> type:
        """Class for wrapping exit trade records."""
        if self._exit_trades_cls is None:
            return ExitTrades
        return self._exit_trades_cls

    @property
    def positions_cls(self) -> type:
        """Class for wrapping position records."""
        if self._positions_cls is None:
            return Positions
        return self._positions_cls

    @property
    def drawdowns_cls(self) -> type:
        """Class for wrapping drawdown records."""
        if self._drawdowns_cls is None:
            return Drawdowns
        return self._drawdowns_cls

    @custom_property(group_by_aware=False)
    def call_seq(self) -> tp.Optional[tp.SeriesFrame]:
        """Sequence of calls per row and group."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "call_seq"):
            call_seq = self.in_outputs.call_seq
        else:
            call_seq = self._call_seq
        if call_seq is None:
            return None

        return self.wrapper.wrap(call_seq, group_by=False)

    # ############# Price ############# #

    @custom_property(group_by_aware=False, resample_func="first")
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "open"):
            open = self.in_outputs.open
        else:
            open = self._open

        if open is None:
            return None
        return self.wrapper.wrap(open, group_by=False)

    @custom_property(group_by_aware=False, resample_func="max")
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "high"):
            high = self.in_outputs.high
        else:
            high = self._high

        if high is None:
            return None
        return self.wrapper.wrap(high, group_by=False)

    @custom_property(group_by_aware=False, resample_func="min")
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low price of each bar."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "low"):
            low = self.in_outputs.low
        else:
            low = self._low

        if low is None:
            return None
        return self.wrapper.wrap(low, group_by=False)

    @custom_property(group_by_aware=False, resample_func="last")
    def close(self) -> tp.SeriesFrame:
        """Last asset price at each time step."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "close"):
            close = self.in_outputs.close
        else:
            close = self._close

        return self.wrapper.wrap(close, group_by=False)

    @class_or_instancemethod
    def get_filled_close(
        cls_or_self,
        close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get forward and backward filled closing price.

        See `vectorbtpro.generic.nb.base.fbfill_nb`."""
        if not isinstance(cls_or_self, type):
            if close is None:
                close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_close = func(to_2d_array(close))
        return wrapper.wrap(filled_close, group_by=False, **resolve_dict(wrap_kwargs))

    @custom_property(group_by_aware=False, resample_func="last")
    def bm_close(self) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Benchmark price per unit series."""
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "bm_close"):
            bm_close = self.in_outputs.bm_close
        else:
            bm_close = self._bm_close

        if bm_close is None or isinstance(bm_close, bool):
            return bm_close
        return self.wrapper.wrap(bm_close, group_by=False)

    @class_or_instancemethod
    def get_filled_bm_close(
        cls_or_self,
        bm_close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Get forward and backward filled benchmark closing price.

        See `vectorbtpro.generic.nb.base.fbfill_nb`."""
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if bm_close is None or isinstance(bm_close, bool):
                    return bm_close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(bm_close)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_bm_close = func(to_2d_array(bm_close))
        return wrapper.wrap(filled_bm_close, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Records ############# #

    @property
    def order_records(self) -> tp.RecordArray:
        """A structured NumPy array of order records."""
        return self._order_records

    @class_or_instancemethod
    def get_orders(
        cls_or_self,
        order_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> Orders:
        """Get order records.

        See `vectorbtpro.portfolio.orders.Orders`."""
        if not isinstance(cls_or_self, type):
            if order_records is None:
                order_records = cls_or_self.order_records
            if open is None:
                open = cls_or_self._open
            if high is None:
                high = cls_or_self._high
            if low is None:
                low = cls_or_self._low
            if close is None:
                close = cls_or_self._close
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
            if orders_cls is None:
                orders_cls = cls_or_self.orders_cls
        else:
            checks.assert_not_none(order_records)
            checks.assert_not_none(wrapper)
            if orders_cls is None:
                orders_cls = Orders

        return orders_cls(
            wrapper,
            order_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @property
    def log_records(self) -> tp.RecordArray:
        """A structured NumPy array of log records."""
        return self._log_records

    @class_or_instancemethod
    def get_logs(
        cls_or_self,
        log_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        logs_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> Logs:
        """Get log records.

        See `vectorbtpro.portfolio.logs.Logs`."""
        if not isinstance(cls_or_self, type):
            if log_records is None:
                log_records = cls_or_self.log_records
            if open is None:
                open = cls_or_self._open
            if high is None:
                high = cls_or_self._high
            if low is None:
                low = cls_or_self._low
            if close is None:
                close = cls_or_self._close
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
            if logs_cls is None:
                logs_cls = cls_or_self.logs_cls
        else:
            checks.assert_not_none(log_records)
            checks.assert_not_none(wrapper)
            if logs_cls is None:
                logs_cls = Logs

        return logs_cls(
            wrapper,
            log_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_entry_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        entry_trades_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> EntryTrades:
        """Get entry trade records.

        See `vectorbtpro.portfolio.trades.EntryTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if entry_trades_cls is None:
                entry_trades_cls = cls_or_self.entry_trades_cls
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if entry_trades_cls is None:
                entry_trades_cls = EntryTrades

        return entry_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_exit_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        exit_trades_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> ExitTrades:
        """Get exit trade records.

        See `vectorbtpro.portfolio.trades.ExitTrades`."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if exit_trades_cls is None:
                exit_trades_cls = cls_or_self.exit_trades_cls
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if exit_trades_cls is None:
                exit_trades_cls = ExitTrades

        return exit_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            **kwargs,
        ).regroup(group_by)

    @class_or_instancemethod
    def get_positions(
        cls_or_self,
        trades: tp.Optional[Trades] = None,
        positions_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Positions:
        """Get position records.

        See `vectorbtpro.portfolio.trades.Positions`."""
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.exit_trades
            if positions_cls is None:
                positions_cls = cls_or_self.positions_cls
        else:
            checks.assert_not_none(trades)
            if positions_cls is None:
                positions_cls = Positions

        return positions_cls.from_trades(trades, **kwargs).regroup(group_by)

    def get_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> Trades:
        """Get trade/position records depending upon `Portfolio.trades_type`."""
        if self.trades_type == enums.TradesType.Trades:
            raise NotImplementedError
        if self.trades_type == enums.TradesType.EntryTrades:
            return self.resolve_shortcut_attr("entry_trades", group_by=group_by, **kwargs)
        if self.trades_type == enums.TradesType.ExitTrades:
            return self.resolve_shortcut_attr("exit_trades", group_by=group_by, **kwargs)
        return self.resolve_shortcut_attr("positions", group_by=group_by, **kwargs)

    def get_trade_history(
        self,
        orders: tp.Optional[Orders] = None,
        entry_trades_cls: tp.Optional[type] = None,
        exit_trades_cls: tp.Optional[type] = None,
        **kwargs,
    ) -> tp.Frame:
        """Get (entry and exit) trade history as a DataFrame."""
        if orders is None:
            orders = self.orders
        entry_trades = self.resolve_shortcut_attr(
            "entry_trades", orders=orders, entry_trades_cls=entry_trades_cls, **kwargs
        )
        exit_trades = self.resolve_shortcut_attr(
            "exit_trades", orders=orders, exit_trades_cls=exit_trades_cls, **kwargs
        )

        order_history = orders.records_readable
        del order_history["Size"]
        del order_history["Price"]
        del order_history["Fees"]
        entry_trade_history = entry_trades.records_readable
        exit_trade_history = exit_trades.records_readable
        entry_trade_history.rename(columns={"Entry Order Id": "Order Id"}, inplace=True)
        exit_trade_history.rename(columns={"Exit Order Id": "Order Id"}, inplace=True)
        del entry_trade_history["Entry Index"]
        del exit_trade_history["Exit Index"]
        entry_trade_history.rename(columns={"Avg Entry Price": "Price"}, inplace=True)
        exit_trade_history.rename(columns={"Avg Exit Price": "Price"}, inplace=True)
        entry_trade_history.rename(columns={"Entry Fees": "Fees"}, inplace=True)
        exit_trade_history.rename(columns={"Exit Fees": "Fees"}, inplace=True)
        del entry_trade_history["Exit Order Id"]
        del exit_trade_history["Entry Order Id"]
        del entry_trade_history["Exit Index"]
        del exit_trade_history["Entry Index"]
        del entry_trade_history["Avg Exit Price"]
        del exit_trade_history["Avg Entry Price"]
        del entry_trade_history["Exit Fees"]
        del exit_trade_history["Entry Fees"]

        trade_history = pd.concat((entry_trade_history, exit_trade_history), axis=0)
        trade_history = pd.merge(order_history, trade_history, on=["Column", "Order Id"])
        trade_history = trade_history.sort_values(by=["Column", "Order Id", "Position Id"])
        trade_history["Entry Trade Id"] = trade_history["Entry Trade Id"].fillna(-1).astype(int)
        trade_history["Exit Trade Id"] = trade_history["Exit Trade Id"].fillna(-1).astype(int)
        trade_history["Entry Trade Id"] = trade_history.pop("Entry Trade Id")
        trade_history["Exit Trade Id"] = trade_history.pop("Exit Trade Id")
        trade_history["Position Id"] = trade_history.pop("Position Id")
        return trade_history

    @class_or_instancemethod
    def get_drawdowns(
        cls_or_self,
        value: tp.Optional[tp.SeriesFrame] = None,
        drawdowns_cls: tp.Optional[type] = None,
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Drawdowns:
        """Get drawdown records from `Portfolio.get_value`.

        See `vectorbtpro.generic.drawdowns.Drawdowns`."""
        if not isinstance(cls_or_self, type):
            if value is None:
                value = cls_or_self.resolve_shortcut_attr("value", group_by=group_by)
            wrapper_kwargs = merge_dicts(cls_or_self.orders.wrapper.config, wrapper_kwargs, dict(group_by=None))
            if drawdowns_cls is None:
                drawdowns_cls = cls_or_self.drawdowns_cls
        else:
            checks.assert_not_none(value)
            if drawdowns_cls is None:
                drawdowns_cls = Drawdowns

        return drawdowns_cls.from_price(value, wrapper_kwargs=wrapper_kwargs, **kwargs)

    # ############# Assets ############# #

    @class_or_instancemethod
    def get_init_position(
        cls_or_self,
        init_position_raw: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial position per column."""
        if not isinstance(cls_or_self, type):
            if init_position_raw is None:
                init_position_raw = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_raw)
            checks.assert_not_none(wrapper)

        init_position = broadcast_array_to(init_position_raw, wrapper.shape_2d[1])
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_flow(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset flow series per column.

        Returns the total transacted amount of assets at each time step."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if init_position is None:
                init_position = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        direction = map_enum_fields(direction, enums.Direction)
        func = jit_reg.resolve_option(nb.asset_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            direction=direction,
            init_position=to_1d_array(init_position),
        )
        return wrapper.wrap(asset_flow, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_assets(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        asset_flow: tp.Optional[tp.SeriesFrame] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset series per column.

        Returns the position at each time step."""
        if not isinstance(cls_or_self, type):
            if asset_flow is None:
                asset_flow = cls_or_self.resolve_shortcut_attr(
                    "asset_flow",
                    direction=enums.Direction.Both,
                    jitted=jitted,
                    chunked=chunked,
                )
            if init_position is None:
                init_position = cls_or_self._init_position
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_flow)
            if init_position is None:
                init_position = 0.0
            checks.assert_not_none(wrapper)

        direction = map_enum_fields(direction, enums.Direction)
        func = jit_reg.resolve_option(nb.assets_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        assets = func(to_2d_array(asset_flow), init_position=to_1d_array(init_position))
        if direction == enums.Direction.LongOnly:
            func = jit_reg.resolve_option(nb.long_assets_nb, jitted)
            assets = func(assets)
        elif direction == enums.Direction.ShortOnly:
            func = jit_reg.resolve_option(nb.short_assets_nb, jitted)
            assets = func(assets)
        return wrapper.wrap(assets, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_mask(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        assets: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get position mask per column/group.

        An element is True if there is a position at the given time step."""
        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        position_mask = to_2d_array(assets) != 0
        if wrapper.grouper.is_grouped(group_by=group_by):
            position_mask = (
                wrapper.wrap(position_mask, group_by=False)
                .vbt(wrapper=wrapper)
                .squeeze_grouped(
                    jit_reg.resolve_option(generic_nb.any_reduce_nb, jitted),
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            )
        return wrapper.wrap(position_mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_position_coverage(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        position_mask: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get position coverage per column/group.

        Position coverage is the number of time steps in the market divided by the total number of time steps."""
        if not isinstance(cls_or_self, type):
            if position_mask is None:
                position_mask = cls_or_self.resolve_shortcut_attr(
                    "position_mask",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                    group_by=False,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(position_mask)
            checks.assert_not_none(wrapper)

        position_coverage = position_mask.vbt(wrapper=wrapper).reduce(
            jit_reg.resolve_option(generic_nb.mean_reduce_nb, jitted),
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="position_coverage"), wrap_kwargs)
        return wrapper.wrap_reduced(position_coverage, group_by=group_by, **wrap_kwargs)

    # ############# Cash ############# #

    @class_or_instancemethod
    def get_cash_deposits(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        split_shared: bool = False,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get cash deposit series per column/group.

        Set `keep_flex` to True to keep format suitable for flexible indexing.
        This consumes less memory."""
        if not isinstance(cls_or_self, type):
            if cash_deposits_raw is None:
                cash_deposits_raw = cls_or_self._cash_deposits
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_deposits_raw is None:
                cash_deposits_raw = 0.0
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(wrapper)

        cash_deposits_raw = to_2d_array(cash_deposits_raw)
        if wrapper.grouper.is_grouped(group_by=group_by):
            if keep_flex and cash_sharing:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_deposits_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(wrapper.shape_2d, cash_deposits_raw, group_lens, cash_sharing)
        else:
            if keep_flex and not cash_sharing:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens()
            func = jit_reg.resolve_option(nb.cash_deposits_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(
                wrapper.shape_2d,
                cash_deposits_raw,
                group_lens,
                cash_sharing,
                split_shared=split_shared,
            )
        if keep_flex:
            return cash_deposits
        return wrapper.wrap(cash_deposits, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash_earnings(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_earnings_raw: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get earnings in cash series per column/group.

        Set `keep_flex` to True to keep format suitable for flexible indexing.
        This consumes less memory."""
        if not isinstance(cls_or_self, type):
            if cash_earnings_raw is None:
                cash_earnings_raw = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_earnings_raw is None:
                cash_earnings_raw = 0.0
            checks.assert_not_none(wrapper)

        cash_earnings_raw = to_2d_array(cash_earnings_raw)
        if wrapper.grouper.is_grouped(group_by=group_by):
            cash_earnings = broadcast_array_to(cash_earnings_raw, wrapper.shape_2d)
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_earnings = func(cash_earnings, group_lens)
        else:
            if keep_flex:
                return cash_earnings_raw
            cash_earnings = broadcast_array_to(cash_earnings_raw, wrapper.shape_2d)
        if keep_flex:
            return cash_earnings
        return wrapper.wrap(cash_earnings, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_cash_flow(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        free: bool = False,
        orders: tp.Optional[Orders] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash flow series per column/group.

        Use `free` to return the flow of the free cash, which never goes above the initial level,
        because an operation always costs money.

        !!! note
            Does not include cash deposits, but includes earnings.

            Using `free` yields the same result as during the simulation only when `leverage=1`.
            For anything else, prefill the state instead of reconstructing it."""
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.orders
            if cash_earnings is None:
                cash_earnings = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        func = jit_reg.resolve_option(nb.cash_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        cash_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            free=free,
            cash_earnings=to_2d_array(cash_earnings),
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_flow = func(cash_flow, group_lens)
        return wrapper.wrap(cash_flow, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_init_cash(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_cash_raw: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        free_cash_flow: tp.Optional[tp.SeriesFrame] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial amount of cash per column/group."""
        if not isinstance(cls_or_self, type):
            if init_cash_raw is None:
                init_cash_raw = cls_or_self._init_cash
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash_raw)
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(wrapper)

        if checks.is_int(init_cash_raw) and init_cash_raw in enums.InitCashMode:
            if not isinstance(cls_or_self, type):
                if free_cash_flow is None:
                    free_cash_flow = cls_or_self.resolve_shortcut_attr(
                        "cash_flow",
                        group_by=group_by,
                        free=True,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            else:
                checks.assert_not_none(free_cash_flow)
                if cash_deposits is None:
                    cash_deposits = 0.0
            func = jit_reg.resolve_option(nb.align_init_cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            init_cash = func(
                init_cash_raw,
                to_2d_array(free_cash_flow),
                cash_deposits=to_2d_array(cash_deposits),
            )
        else:
            init_cash_raw = to_1d_array(init_cash_raw)
            if wrapper.grouper.is_grouped(group_by=group_by):
                group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
                func = jit_reg.resolve_option(nb.init_cash_grouped_nb, jitted)
                init_cash = func(init_cash_raw, group_lens, cash_sharing)
            else:
                group_lens = wrapper.grouper.get_group_lens()
                func = jit_reg.resolve_option(nb.init_cash_nb, jitted)
                init_cash = func(init_cash_raw, group_lens, cash_sharing, split_shared=split_shared)
        wrap_kwargs = merge_dicts(dict(name_or_index="init_cash"), wrap_kwargs)
        return wrapper.wrap_reduced(init_cash, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_cash(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        free: bool = False,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash balance series per column/group.

        For `free`, see `Portfolio.get_cash_flow`."""
        if not isinstance(cls_or_self, type):
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    free=free,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        "init_cash",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash = func(
                wrapper.shape_2d,
                to_2d_array(cash_flow),
                group_lens,
                to_1d_array(init_cash),
                cash_deposits_grouped=to_2d_array(cash_deposits),
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_cash is None:
                    init_cash = cls_or_self.resolve_shortcut_attr(
                        "init_cash",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            func = jit_reg.resolve_option(nb.cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash = func(
                to_2d_array(cash_flow),
                to_1d_array(init_cash),
                cash_deposits=to_2d_array(cash_deposits),
            )
        return wrapper.wrap(cash, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Value ############# #

    @class_or_instancemethod
    def get_init_price(
        cls_or_self,
        init_price_raw: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial price per column."""
        if not isinstance(cls_or_self, type):
            if init_price_raw is None:
                init_price_raw = cls_or_self._init_price
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_price_raw)
            checks.assert_not_none(wrapper)

        init_price = broadcast_array_to(init_price_raw, wrapper.shape_2d[1])
        wrap_kwargs = merge_dicts(dict(name_or_index="init_price"), wrap_kwargs)
        return wrapper.wrap_reduced(init_price, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_init_position_value(
        cls_or_self,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial position value per column."""
        if not isinstance(cls_or_self, type):
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if init_position is None:
                init_position = 0.0
            checks.assert_not_none(init_price)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.init_position_value_nb, jitted)
        init_position_value = func(
            n_cols=wrapper.shape_2d[1],
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position_value, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def get_init_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        init_cash: tp.Optional[tp.MaybeSeries] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial value per column/group.

        Includes initial cash and the value of initial position."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    "init_cash",
                    group_by=group_by,
                    split_shared=split_shared,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(init_cash)
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.init_value_grouped_nb, jitted)
            init_value = func(group_lens, to_1d_array(init_position_value), to_1d_array(init_cash))
        else:
            func = jit_reg.resolve_option(nb.init_value_nb, jitted)
            init_value = func(to_1d_array(init_position_value), to_1d_array(init_cash))
        wrap_kwargs = merge_dicts(dict(name_or_index="init_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_input_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash_sharing: tp.Optional[bool] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        split_shared: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total input value per column/group.

        Includes initial value and any cash deposited at any point in time."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits_raw is None:
                cash_deposits_raw = cls_or_self._cash_deposits
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash_sharing)
            checks.assert_not_none(init_value)
            if cash_deposits_raw is None:
                cash_deposits_raw = 0.0
            checks.assert_not_none(wrapper)

        cash_deposits_raw = to_2d_array(cash_deposits_raw)
        cash_deposits_sum = cash_deposits_raw.sum(axis=0)
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.init_cash_grouped_nb, jitted)
            input_value = func(cash_deposits_sum, group_lens, cash_sharing)
        else:
            group_lens = wrapper.grouper.get_group_lens()
            func = jit_reg.resolve_option(nb.init_cash_nb, jitted)
            input_value = func(cash_deposits_sum, group_lens, cash_sharing, split_shared=split_shared)
        input_value += to_1d_array(init_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="input_value"), wrap_kwargs)
        return wrapper.wrap_reduced(input_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_asset_value(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        assets: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset value series per column/group."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(assets)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_value = func(to_2d_array(close), to_2d_array(assets))
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.sum_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            asset_value = func(asset_value, group_lens)
        return wrapper.wrap(asset_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_gross_exposure(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get gross exposure.

        !!! note
            When both directions, `asset_value` must include the addition of the absolute long-only
            and short-only asset values."""
        direction = map_enum_fields(direction, enums.Direction)

        if not isinstance(cls_or_self, type):
            if asset_value is None:
                if direction == enums.Direction.Both and cls_or_self.wrapper.grouper.is_grouped(group_by=group_by):
                    long_asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        direction="longonly",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                    short_asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        direction="shortonly",
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                    asset_value = long_asset_value + short_asset_value
                else:
                    asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        group_by=group_by,
                        direction=direction,
                        jitted=jitted,
                        chunked=chunked,
                    )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value)
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.gross_exposure_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        gross_exposure = func(to_2d_array(asset_value), to_2d_array(value))
        return wrapper.wrap(gross_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_net_exposure(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        long_exposure: tp.Optional[tp.SeriesFrame] = None,
        short_exposure: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get net exposure."""
        if not isinstance(cls_or_self, type):
            if long_exposure is None:
                long_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction=enums.Direction.LongOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if short_exposure is None:
                short_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction=enums.Direction.ShortOnly,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(long_exposure)
            checks.assert_not_none(short_exposure)
            checks.assert_not_none(wrapper)

        net_exposure = to_2d_array(long_exposure) - to_2d_array(short_exposure)
        return wrapper.wrap(net_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        cash: tp.Optional[tp.SeriesFrame] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get portfolio value series per column/group.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with the initial cash balance and position being that of the
        entire group. Useful for generating returns and comparing assets within the same group."""
        if not isinstance(cls_or_self, type):
            if cash is None:
                cash = cls_or_self.resolve_shortcut_attr("cash", group_by=group_by, jitted=jitted, chunked=chunked)
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        value = func(to_2d_array(cash), to_2d_array(asset_value))
        return wrapper.wrap(value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_allocations(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        group_by: tp.GroupByLike = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get portfolio allocation series per column."""
        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    direction=direction,
                    group_by=False,
                    jitted=jitted,
                    chunked=chunked,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value)
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        func = jit_reg.resolve_option(nb.allocations_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        allocations = func(to_2d_array(asset_value), to_2d_array(value), group_lens)
        return wrapper.wrap(allocations, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_total_profit(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total profit per column/group.

        Calculated directly from order records (fast)."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if orders is None:
                orders = cls_or_self.orders
            if init_position is None:
                init_position = cls_or_self._init_position
            if init_price is None:
                init_price = cls_or_self._init_price
            if cash_earnings is None:
                cash_earnings = cls_or_self._cash_earnings
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders)
            if close is None:
                close = orders.close
            checks.assert_not_none(close)
            checks.assert_not_none(init_price)
            if init_position is None:
                init_position = 0.0
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper

        func = jit_reg.resolve_option(nb.total_profit_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        total_profit = func(
            wrapper.shape_2d,
            to_2d_array(close),
            orders.values,
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            cash_earnings=to_2d_array(cash_earnings),
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.total_profit_grouped_nb, jitted)
            total_profit = func(total_profit, group_lens)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_profit"), wrap_kwargs)
        return wrapper.wrap_reduced(total_profit, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_final_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total profit per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        final_value = to_1d_array(input_value) + to_1d_array(total_profit)
        wrap_kwargs = merge_dicts(dict(name_or_index="final_value"), wrap_kwargs)
        return wrapper.wrap_reduced(final_value, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_total_return(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total return per column/group."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(total_profit)
            checks.assert_not_none(wrapper)

        total_return = to_1d_array(total_profit) / to_1d_array(input_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get return series per column/group based on portfolio value."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr("value", group_by=group_by, jitted=jitted, chunked=chunked)
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        returns = func(
            to_2d_array(value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            log_returns=log_returns,
        )
        returns = wrapper.wrap(returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            returns = returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return returns

    @class_or_instancemethod
    def get_asset_pnl(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset (realized and unrealized) PnL series per column/group."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_pnl_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_pnl = func(
            to_1d_array(init_position_value),
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
        )
        return wrapper.wrap(asset_pnl, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_asset_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset return series per column/group.

        This type of returns is based solely on cash flows and asset value rather than portfolio
        value. It ignores passive cash and thus it will return the same numbers irrespective of the amount of
        cash currently available, even `np.inf`. The scale of returns is comparable to that of going
        all in and keeping available cash at zero."""
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.init_position_value
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value)
            checks.assert_not_none(asset_value)
            checks.assert_not_none(cash_flow)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.asset_returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_returns = func(
            to_1d_array(init_position_value),
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
            log_returns=log_returns,
        )
        asset_returns = wrapper.wrap(asset_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            asset_returns = asset_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return asset_returns

    @class_or_instancemethod
    def get_market_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get market value series per column/group.

        If grouped, evenly distributes the initial cash among assets in the group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close)
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(wrapper)

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        split_shared=True,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.market_value_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                group_lens,
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        group_by=False,
                        jitted=jitted,
                        chunked=chunked,
                        keep_flex=True,
                    )
            func = jit_reg.resolve_option(nb.market_value_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
            )
        return wrapper.wrap(market_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def get_market_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get market return series per column/group."""
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                    keep_flex=True,
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value)
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(market_value)
            checks.assert_not_none(wrapper)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        market_returns = func(
            to_2d_array(market_value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            log_returns=log_returns,
        )
        market_returns = wrapper.wrap(market_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            market_returns = market_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return market_returns

    @class_or_instancemethod
    def get_bm_value(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark value series per column/group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_value`."""
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if isinstance(bm_close, bool):
                    if not bm_close:
                        return None
                    bm_close = None
                if bm_close is not None:
                    if cls_or_self.fillna_close:
                        bm_close = cls_or_self.filled_bm_close
        return cls_or_self.get_market_value(
            group_by=group_by,
            close=bm_close,
            init_value=init_value,
            cash_deposits=cash_deposits,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
        )

    @class_or_instancemethod
    def get_bm_returns(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        bm_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark return series per column/group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_returns`."""
        if not isinstance(cls_or_self, type):
            bm_value = cls_or_self.resolve_shortcut_attr("bm_value", group_by=group_by, jitted=jitted, chunked=chunked)
            if bm_value is None:
                return None
        return cls_or_self.get_market_returns(
            group_by=group_by,
            init_value=init_value,
            cash_deposits=cash_deposits,
            market_value=bm_value,
            log_returns=log_returns,
            daily_returns=daily_returns,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
        )

    @class_or_instancemethod
    def get_total_market_return(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get total market return."""
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value)
            checks.assert_not_none(market_value)
            checks.assert_not_none(wrapper)

        input_value = to_1d_array(input_value)
        final_value = to_2d_array(market_value)[-1]
        total_return = (final_value - input_value) / input_value
        wrap_kwargs = merge_dicts(dict(name_or_index="total_market_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @class_or_instancemethod
    def get_returns_acc(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        returns: tp.Optional[tp.SeriesFrame] = None,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        defaults: tp.KwargsLike = None,
        **kwargs,
    ) -> ReturnsAccessor:
        """Get returns accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`.

        !!! hint
            You can find most methods of this accessor as (cacheable) attributes of this portfolio."""
        if not isinstance(cls_or_self, type):
            if returns is None:
                if use_asset_returns:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "asset_returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
                else:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        group_by=group_by,
                        jitted=jitted,
                        chunked=chunked,
                    )
            if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
                bm_returns = cls_or_self.resolve_shortcut_attr(
                    "bm_returns",
                    log_returns=log_returns,
                    daily_returns=daily_returns,
                    group_by=group_by,
                    jitted=jitted,
                    chunked=chunked,
                )
            elif isinstance(bm_returns, bool) and not bm_returns:
                bm_returns = None
            if freq is None:
                freq = cls_or_self.wrapper.freq
        else:
            checks.assert_not_none(returns)

        if daily_returns:
            freq = "D"
        return returns.vbt.returns(
            bm_returns=bm_returns,
            log_returns=log_returns,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            **kwargs,
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """`Portfolio.get_returns_acc` with default arguments."""
        return self.get_returns_acc()

    @class_or_instancemethod
    def get_qs(
        cls_or_self,
        group_by: tp.GroupByLike = None,
        returns: tp.Optional[tp.SeriesFrame] = None,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        defaults: tp.KwargsLike = None,
        **kwargs,
    ) -> QSAdapterT:
        """Get quantstats adapter of type `vectorbtpro.returns.qs_adapter.QSAdapter`.

        `**kwargs` are passed to the adapter constructor."""
        from vectorbtpro.returns.qs_adapter import QSAdapter

        returns_acc = cls_or_self.get_returns_acc(
            group_by=group_by,
            returns=returns,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            jitted=jitted,
            chunked=chunked,
            defaults=defaults,
        )
        return QSAdapter(returns_acc, **kwargs)

    @property
    def qs(self) -> QSAdapterT:
        """`Portfolio.get_qs` with default arguments."""
        return self.get_qs()

    # ############# Resolution ############# #

    @property
    def self_aliases(self) -> tp.Set[str]:
        """Names to associate with this object."""
        return {"self", "portfolio", "pf"}

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Uses the following keys:

        * `use_asset_returns`: Whether to use `Portfolio.get_asset_returns` when resolving `returns` argument.
        * `trades_type`: Which trade type to use when resolving `trades` argument."""
        if "use_asset_returns" in final_kwargs:
            if attr == "returns" and final_kwargs["use_asset_returns"]:
                attr = "asset_returns"
        if "trades_type" in final_kwargs:
            trades_type = final_kwargs["trades_type"]
            if isinstance(final_kwargs["trades_type"], str):
                trades_type = map_enum_fields(trades_type, enums.TradesType)
            if attr == "trades" and trades_type != self.trades_type:
                if trades_type == enums.TradesType.EntryTrades:
                    attr = "entry_trades"
                elif trades_type == enums.TradesType.ExitTrades:
                    attr = "exit_trades"
                else:
                    attr = "positions"
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process an object after resolution.

        Uses the following keys:

        * `incl_open`: Whether to include open trades/positions when resolving an argument
            that is an instance of `vectorbtpro.portfolio.trades.Trades`."""
        if "incl_open" in final_kwargs:
            if isinstance(out, Trades) and not final_kwargs["incl_open"]:
                out = out.status_closed
        return out

    def resolve_shortcut_attr(self, attr_name: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute that may have shortcut properties.

        If `attr_name` has a prefix `get_`, checks whether the respective shortcut property can be called.
        This way, complex call hierarchies can utilize cacheable properties."""
        if not attr_name.startswith("get_"):
            if "get_" + attr_name not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr_name), property):
                    return getattr(self, attr_name)
                return getattr(self, attr_name)(*args, **kwargs)
            attr_name = "get_" + attr_name

        if len(args) == 0:
            naked_attr_name = attr_name[4:]
            prop_name = naked_attr_name
            _kwargs = dict(kwargs)

            if "free" in _kwargs:
                if _kwargs.pop("free"):
                    prop_name = "free_" + naked_attr_name
            if "direction" in _kwargs:
                direction = map_enum_fields(_kwargs.pop("direction"), enums.Direction)
                if direction == enums.Direction.LongOnly:
                    prop_name = "long_" + naked_attr_name
                elif direction == enums.Direction.ShortOnly:
                    prop_name = "short_" + naked_attr_name

            if prop_name in self.cls_dir:
                prop = getattr(type(self), prop_name)
                options = getattr(prop, "options", {})

                can_call_prop = True
                if "group_by" in _kwargs:
                    group_by = _kwargs.pop("group_by")
                    group_aware = options.get("group_aware", True)
                    if group_aware:
                        if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                            can_call_prop = False
                    else:
                        group_by = _kwargs.pop("group_by")
                        if self.wrapper.grouper.is_grouping_enabled(group_by=group_by):
                            can_call_prop = False
                if can_call_prop:
                    _kwargs.pop("jitted", None)
                    _kwargs.pop("chunked", None)
                    for k, v in get_func_kwargs(getattr(type(self), attr_name)).items():
                        if k in _kwargs and v is not _kwargs.pop(k):
                            can_call_prop = False
                            break
                    if can_call_prop:
                        if len(_kwargs) > 0:
                            can_call_prop = False
                        if can_call_prop:
                            return getattr(self, prop_name)

        return getattr(self, attr_name)(*args, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.portfolio`."""
        from vectorbtpro._settings import settings

        returns_cfg = settings["returns"]
        portfolio_stats_cfg = settings["portfolio"]["stats"]

        return merge_dicts(
            Analyzable.stats_defaults.__get__(self),
            dict(settings=dict(year_freq=returns_cfg["year_freq"], trades_type=self.trades_type)),
            portfolio_stats_cfg,
        )

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
            start_value=dict(title="Start Value", calc_func="init_value", tags="portfolio"),
            min_value=dict(title="Min Value", calc_func="value.vbt.min", tags="portfolio"),
            max_value=dict(title="Max Value", calc_func="value.vbt.max", tags="portfolio"),
            end_value=dict(title="End Value", calc_func="final_value", tags="portfolio"),
            cash_deposits=dict(
                title="Cash Deposits", calc_func="cash_deposits.vbt.sum", check_has_cash_deposits=True, tags="portfolio"
            ),
            cash_earnings=dict(
                title="Cash Earnings", calc_func="cash_earnings.vbt.sum", check_has_cash_earnings=True, tags="portfolio"
            ),
            total_return=dict(
                title="Total Return [%]",
                calc_func="total_return",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            bm_return=dict(
                title="Benchmark Return [%]",
                calc_func="bm_returns.vbt.returns.total",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_bm_returns=True,
                tags="portfolio",
            ),
            total_time_exposure=dict(
                title="Total Time Exposure [%]",
                calc_func="position_mask.vbt.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_gross_exposure=dict(
                title="Max Gross Exposure [%]",
                calc_func="gross_exposure.vbt.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_dd=dict(
                title="Max Drawdown [%]",
                calc_func="drawdowns.max_drawdown",
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=["portfolio", "drawdowns"],
            ),
            max_dd_duration=dict(
                title="Max Drawdown Duration",
                calc_func="drawdowns.max_duration",
                fill_wrap_kwargs=True,
                tags=["portfolio", "drawdowns", "duration"],
            ),
            total_orders=dict(
                title="Total Orders",
                calc_func="orders.count",
                tags=["portfolio", "orders"],
            ),
            total_fees_paid=dict(
                title="Total Fees Paid",
                calc_func="orders.fees.sum",
                tags=["portfolio", "orders"],
            ),
            total_trades=dict(
                title="Total Trades",
                calc_func="trades.count",
                incl_open=True,
                tags=["portfolio", "trades"],
            ),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="trades.win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func="trades.returns.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func="trades.returns.min",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func="trades.winning.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func="trades.losing.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func="trades.winning.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func="trades.losing.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func="trades.profit_factor",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func="trades.expectancy",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            sharpe_ratio=dict(
                title="Sharpe Ratio",
                calc_func="returns_acc.sharpe_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            calmar_ratio=dict(
                title="Calmar Ratio",
                calc_func="returns_acc.calmar_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            omega_ratio=dict(
                title="Omega Ratio",
                calc_func="returns_acc.omega_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            sortino_ratio=dict(
                title="Sortino Ratio",
                calc_func="returns_acc.sortino_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    def returns_stats(
        self,
        group_by: tp.GroupByLike = None,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        use_asset_returns: bool = False,
        defaults: tp.KwargsLike = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Compute various statistics on returns of this portfolio.

        See `Portfolio.returns_acc` and `vectorbtpro.returns.accessors.ReturnsAccessor.metrics`.

        `kwargs` will be passed to `vectorbtpro.returns.accessors.ReturnsAccessor.stats` method.
        If `bm_returns` is not set, uses `Portfolio.get_market_returns`."""
        returns_acc = self.get_returns_acc(
            group_by=group_by,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            defaults=defaults,
            chunked=chunked,
        )
        return getattr(returns_acc, "stats")(**kwargs)

    # ############# Plotting ############# #

    def plot_trade_signals(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_positions: tp.Union[bool, str] = "zones",
        long_entry_trace_kwargs: tp.KwargsLike = None,
        short_entry_trace_kwargs: tp.KwargsLike = None,
        long_exit_trace_kwargs: tp.KwargsLike = None,
        short_exit_trace_kwargs: tp.KwargsLike = None,
        long_shape_kwargs: tp.KwargsLike = None,
        short_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of trade signals.

        Markers and shapes are colored by trade direction (green = long, red = short)."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        entry_trades = self.resolve_shortcut_attr("entry_trades")
        exit_trades = self.resolve_shortcut_attr("exit_trades")
        positions = self.resolve_shortcut_attr("positions")

        fig = entry_trades.plot_signals(
            column=column,
            long_entry_trace_kwargs=long_entry_trace_kwargs,
            short_entry_trace_kwargs=short_entry_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **kwargs,
        )
        fig = exit_trades.plot_signals(
            column=column,
            plot_ohlc=False,
            plot_close=False,
            long_exit_trace_kwargs=long_exit_trace_kwargs,
            short_exit_trace_kwargs=short_exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if isinstance(plot_positions, bool):
            if plot_positions:
                plot_positions = "zones"
            else:
                plot_positions = None
        if plot_positions is not None:
            if plot_positions.lower() == "zones":
                long_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["green"]),
                    long_shape_kwargs,
                )
                short_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["red"]),
                    short_shape_kwargs,
                )
            elif plot_positions.lower() == "lines":
                base_shape_kwargs = dict(
                    type="line",
                    line=dict(dash="dot"),
                    xref=Rep("xref"),
                    yref=Rep("yref"),
                    x0=Rep("start_index"),
                    x1=Rep("end_index"),
                    y0=RepFunc(lambda record: record["entry_price"]),
                    y1=RepFunc(lambda record: record["exit_price"]),
                    opacity=0.75,
                )
                long_shape_kwargs = atomic_dict(
                    merge_dicts(
                        base_shape_kwargs,
                        dict(line=dict(color=plotting_cfg["contrast_color_schema"]["green"])),
                        long_shape_kwargs,
                    )
                )
                short_shape_kwargs = atomic_dict(
                    merge_dicts(
                        base_shape_kwargs,
                        dict(line=dict(color=plotting_cfg["contrast_color_schema"]["red"])),
                        short_shape_kwargs,
                    )
                )
            else:
                raise ValueError(f"Invalid option plot_positions='{plot_positions}'")
            fig = positions.direction_long.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=long_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x",
                yref=fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y",
                fig=fig,
            )
            fig = positions.direction_short.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=short_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x",
                yref=fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y",
                fig=fig,
            )
        return fig

    def plot_asset_flow(
        self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of asset flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        asset_flow = self.resolve_shortcut_attr("asset_flow", direction=direction, jitted=jitted, chunked=chunked)
        asset_flow = self.select_col_from_obj(asset_flow, column, wrapper=self.wrapper.regroup(False))
        kwargs = merge_dicts(
            dict(trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["brown"]), name="Assets")),
            kwargs,
        )
        fig = asset_flow.vbt.lineplot(**kwargs)
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
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_cash_flow(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        free: bool = False,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cash flow.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        cash_flow = self.resolve_shortcut_attr(
            "cash_flow",
            group_by=group_by,
            free=free,
            jitted=jitted,
            chunked=chunked,
        )
        cash_flow = self.select_col_from_obj(cash_flow, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["green"]), name="Cash")),
            kwargs,
        )
        fig = cash_flow.vbt.lineplot(**kwargs)
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
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_assets(
        self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of assets.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        assets = self.resolve_shortcut_attr("assets", direction=direction, jitted=jitted, chunked=chunked)
        assets = self.select_col_from_obj(assets, column, wrapper=self.wrapper.regroup(False))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["brown"]), name="Assets"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["brown"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = assets.vbt.plot_against(0, **kwargs)
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
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_cash(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        free: bool = False,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cash balance.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        init_cash = self.resolve_shortcut_attr("init_cash", group_by=group_by, jitted=jitted, chunked=chunked)
        init_cash = self.select_col_from_obj(init_cash, column, wrapper=self.wrapper.regroup(group_by))
        cash = self.resolve_shortcut_attr("cash", group_by=group_by, free=free, jitted=jitted, chunked=chunked)
        cash = self.select_col_from_obj(cash, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["green"]), name="Cash"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["green"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = cash.vbt.plot_against(init_cash, **kwargs)
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
                    y0=init_cash,
                    x1=x_domain[1],
                    y1=init_cash,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_asset_value(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of asset value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        asset_value = self.resolve_shortcut_attr(
            "asset_value",
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        asset_value = self.select_col_from_obj(asset_value, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["cyan"]), name="Asset Value"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = asset_value.vbt.plot_against(0, **kwargs)
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
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_value(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of value.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        init_cash = self.resolve_shortcut_attr("init_cash", group_by=group_by, jitted=jitted, chunked=chunked)
        init_cash = self.select_col_from_obj(init_cash, column, wrapper=self.wrapper.regroup(group_by))
        value = self.resolve_shortcut_attr("value", group_by=group_by, jitted=jitted, chunked=chunked)
        value = self.select_col_from_obj(value, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value"),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = value.vbt.plot_against(init_cash, **kwargs)
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
                    y0=init_cash,
                    x1=x_domain[1],
                    y1=init_cash,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_allocations(
        self,
        column: tp.Optional[tp.Label] = None,
        line_shape: str = "hv",
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one group of allocations."""
        filled_allocations = self.resolve_shortcut_attr(
            "allocations",
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        filled_allocations = self.select_col_from_obj(filled_allocations, column, obj_ungrouped=True)
        group_names = self.wrapper.grouper.get_index().names
        filled_allocations = filled_allocations.vbt.drop_levels(group_names, strict=False)
        fig = filled_allocations.vbt.areaplot(line_shape=line_shape, **kwargs)
        return fig

    def plot_cum_returns(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        use_asset_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of cumulative returns.

        If `bm_returns` is None, will use `Portfolio.get_market_returns`.

        `**kwargs` are passed to `vectorbtpro.returns.accessors.ReturnsSRAccessor.plot_cumulative`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
            bm_returns = self.resolve_shortcut_attr(
                "bm_returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        elif isinstance(bm_returns, bool) and not bm_returns:
            bm_returns = None
        else:
            bm_returns = broadcast_to(bm_returns, self.obj)
        bm_returns = self.select_col_from_obj(bm_returns, column, wrapper=self.wrapper.regroup(group_by))
        if use_asset_returns:
            returns = self.resolve_shortcut_attr(
                "asset_returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        else:
            returns = self.resolve_shortcut_attr(
                "returns",
                log_returns=log_returns,
                daily_returns=False,
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
        returns = self.select_col_from_obj(returns, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                bm_returns=bm_returns,
                main_kwargs=dict(
                    trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value"),
                ),
                hline_shape_kwargs=dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
            ),
            kwargs,
        )
        return returns.vbt.returns(log_returns=log_returns).plot_cumulative(**kwargs)

    def plot_drawdowns(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of drawdowns.

        `**kwargs` are passed to `vectorbtpro.generic.drawdowns.Drawdowns.plot`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        kwargs = merge_dicts(
            dict(close_trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["purple"]), name="Value")),
            kwargs,
        )
        return self.resolve_shortcut_attr("drawdowns", group_by=group_by).plot(column=column, **kwargs)

    def plot_underwater(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of underwater.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        drawdown = self.resolve_shortcut_attr("drawdown", group_by=group_by, jitted=jitted, chunked=chunked)
        drawdown = self.select_col_from_obj(drawdown, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(color=plotting_cfg["color_schema"]["red"]),
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3),
                    fill="tozeroy",
                    name="Drawdown",
                )
            ),
            kwargs,
        )
        fig = drawdown.vbt.lineplot(**kwargs)
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
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        yaxis = "yaxis" + yref[1:]
        fig.layout[yaxis]["tickformat"] = "%"
        return fig

    def plot_gross_exposure(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        direction: tp.Union[str, int] = "both",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of gross exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        gross_exposure = self.resolve_shortcut_attr(
            "gross_exposure",
            direction=direction,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
        )
        gross_exposure = self.select_col_from_obj(gross_exposure, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["pink"]), name="Exposure"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = gross_exposure.vbt.plot_against(1, **kwargs)
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
                    y0=1,
                    x1=x_domain[1],
                    y1=1,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_net_exposure(
        self,
        column: tp.Optional[tp.Label] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column/group of net exposure.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`."""
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        net_exposure = self.resolve_shortcut_attr("net_exposure", group_by=group_by, jitted=jitted, chunked=chunked)
        net_exposure = self.select_col_from_obj(net_exposure, column, wrapper=self.wrapper.regroup(group_by))
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(line=dict(color=plotting_cfg["color_schema"]["pink"]), name="Exposure"),
                pos_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3)),
                neg_trace_kwargs=dict(fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3)),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = net_exposure.vbt.plot_against(0, **kwargs)
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
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.plot`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.portfolio`."""
        from vectorbtpro._settings import settings

        returns_cfg = settings["returns"]
        portfolio_plots_cfg = settings["portfolio"]["plots"]

        return merge_dicts(
            Analyzable.plots_defaults.__get__(self),
            dict(settings=dict(year_freq=returns_cfg["year_freq"], trades_type=self.trades_type)),
            portfolio_plots_cfg,
        )

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            orders=dict(
                title="Orders",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="orders.plot",
                tags=["portfolio", "orders"],
            ),
            trades=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="trades.plot",
                tags=["portfolio", "trades"],
            ),
            trade_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func="trades.plot_pnl",
                pct_scale=True,
                tags=["portfolio", "trades"],
            ),
            trade_signals=dict(
                title="Trade Signals",
                yaxis_kwargs=dict(title="Trade Signals"),
                check_is_not_grouped=True,
                plot_func="plot_trade_signals",
                tags=["portfolio", "trades"],
            ),
            asset_flow=dict(
                title="Asset Flow",
                yaxis_kwargs=dict(title="Asset flow"),
                check_is_not_grouped=True,
                plot_func="plot_asset_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            cash_flow=dict(
                title="Cash Flow",
                yaxis_kwargs=dict(title="Cash flow"),
                plot_func="plot_cash_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            assets=dict(
                title="Assets",
                yaxis_kwargs=dict(title="Assets"),
                check_is_not_grouped=True,
                plot_func="plot_assets",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            cash=dict(
                title="Cash",
                yaxis_kwargs=dict(title="Cash"),
                plot_func="plot_cash",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            asset_value=dict(
                title="Asset Value",
                yaxis_kwargs=dict(title="Asset value"),
                plot_func="plot_asset_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets", "value"],
            ),
            value=dict(
                title="Value",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value"],
            ),
            cum_returns=dict(
                title="Cumulative Returns",
                yaxis_kwargs=dict(title="Cumulative returns"),
                plot_func="plot_cum_returns",
                pass_hline_shape_kwargs=True,
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "returns"],
            ),
            drawdowns=dict(
                title="Drawdowns",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_drawdowns",
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            underwater=dict(
                title="Underwater",
                yaxis_kwargs=dict(title="Drawdown"),
                plot_func="plot_underwater",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            gross_exposure=dict(
                title="Gross Exposure",
                yaxis_kwargs=dict(title="Gross exposure"),
                plot_func="plot_gross_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
            net_exposure=dict(
                title="Net Exposure",
                yaxis_kwargs=dict(title="Net exposure"),
                plot_func="plot_net_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
        )
    )

    plot = Analyzable.plots

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_in_output_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build in-output config documentation."""
        if source_cls is None:
            source_cls = Portfolio
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "in_output_config").__doc__)).substitute(
            {"in_output_config": cls.in_output_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_in_output_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Data.in_output_config`."""
        __pdoc__[cls.__name__ + ".in_output_config"] = cls.build_in_output_config_doc(source_cls=source_cls)


Portfolio.override_in_output_config_doc(__pdoc__)
Portfolio.override_metrics_doc(__pdoc__)
Portfolio.override_subplots_doc(__pdoc__)

__pdoc__["Portfolio.plot"] = "See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."

PF = Portfolio
"""Shortcut for `Portfolio`."""

__pdoc__["PF"] = False
