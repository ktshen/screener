# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base functions and classes for portfolio optimization."""

import inspect
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.parsing import get_func_arg_names, warn_stdout
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.template import substitute_templates, Rep, RepFunc, CustomTemplate
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.params import Param, combine_params, find_params_in_obj, param_product_to_objs
from vectorbtpro.utils.pickling import pdict
from vectorbtpro.base.indexes import combine_indexes, stack_indexes, select_levels
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.reshaping import to_pd_array, to_1d_array, to_2d_array, to_dict
from vectorbtpro.base.indexing import point_idxr_defaults, range_idxr_defaults
from vectorbtpro.data.base import Data
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.enums import RangeStatus
from vectorbtpro.portfolio.enums import alloc_range_dt, alloc_point_dt, Direction
from vectorbtpro.portfolio.pfopt import nb
from vectorbtpro.portfolio.pfopt.records import AllocRanges, AllocPoints
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg

if tp.TYPE_CHECKING:
    from vectorbtpro.portfolio.base import Portfolio as PortfolioT
else:
    PortfolioT = tp.Any

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from pypfopt.base_optimizer import BaseOptimizer as BaseOptimizerT
except ImportError as e:
    BaseOptimizerT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from riskfolio import Portfolio as RPortfolio, HCPortfolio as RHCPortfolio

    RPortfolioT = tp.TypeVar("RPortfolioT", bound=tp.Union[RPortfolio, RHCPortfolio])
except ImportError as e:
    RPortfolioT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from universal.algo import Algo
    from universal.result import AlgoResult

    AlgoT = tp.TypeVar("AlgoT", bound=Algo)
    AlgoResultT = tp.TypeVar("AlgoResultT", bound=AlgoResult)
except ImportError as e:
    AlgoT = tp.Any
    AlgoResultT = tp.Any

__all__ = [
    "pfopt_func_dict",
    "pypfopt_optimize",
    "riskfolio_optimize",
    "PortfolioOptimizer",
    "PFO",
]

__pdoc__ = {}


# ############# PyPortfolioOpt ############# #


class pfopt_func_dict(pdict):
    """Dict that contains optimization functions as keys.

    Keys can be functions themselves, their names, or `_def` for the default value."""

    pass


def select_pfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    kwargs: tp.Union[None, tp.Kwargs, pfopt_func_dict] = None,
) -> tp.Kwargs:
    """Select keyword arguments belonging to `pypfopt_func`."""
    if kwargs is None:
        return {}
    if isinstance(kwargs, pfopt_func_dict):
        if pypfopt_func in kwargs:
            _kwargs = kwargs[pypfopt_func]
        elif pypfopt_func.__name__ in kwargs:
            _kwargs = kwargs[pypfopt_func.__name__]
        elif "_def" in kwargs:
            _kwargs = kwargs["_def"]
        else:
            _kwargs = {}
    else:
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, pfopt_func_dict):
                if pypfopt_func in v:
                    _kwargs[k] = v[pypfopt_func]
                elif pypfopt_func.__name__ in v:
                    _kwargs[k] = v[pypfopt_func.__name__]
                elif "_def" in v:
                    _kwargs[k] = v["_def"]
            else:
                _kwargs[k] = v
    return _kwargs


def resolve_pypfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    cache: tp.KwargsLike = None,
    var_kwarg_names: tp.Optional[tp.Iterable[str]] = None,
    used_arg_names: tp.Optional[tp.Set[str]] = None,
    **kwargs,
) -> tp.Kwargs:
    """Resolve keyword arguments passed to any optimization function with the layout of PyPortfolioOpt.

    Parses the signature of `pypfopt_func`, and for each accepted argument, looks for an argument
    with the same name in `kwargs`. If not found, tries to resolve that argument using other arguments
    or by calling other optimization functions.

    Argument `frequency` gets resolved with (global) `freq` and `year_freq` using
    `vectorbtpro.returns.accessors.ReturnsAccessor.get_ann_factor`.

    Any argument in `kwargs` can be wrapped using `pfopt_func_dict` to define the argument
    per function rather than globally.

    !!! note
        When providing custom functions, make sure that the arguments they accept are visible
        in the signature (that is, no variable arguments) and have the same naming as in PyPortfolioOpt.

        Functions `market_implied_prior_returns` and `BlackLittermanModel.bl_weights` take `risk_aversion`,
        which is different from arguments with the same name in other functions. To set it, pass `delta`."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    signature = inspect.signature(pypfopt_func)
    kwargs = select_pfopt_func_kwargs(pypfopt_func, kwargs)
    if cache is None:
        cache = {}
    arg_names = get_func_arg_names(pypfopt_func)
    if len(arg_names) == 0:
        return {}
    if used_arg_names is None:
        used_arg_names = set()

    pass_kwargs = dict()

    def _process_arg(arg_name, arg_value):
        orig_arg_name = arg_name
        if pypfopt_func.__name__ in ("market_implied_prior_returns", "bl_weights"):
            if arg_name == "risk_aversion":
                # In some methods, risk_aversion is expected as array and means delta
                arg_name = "delta"

        def _get_kwarg(*args):
            used_arg_names.add(args[0])
            return kwargs.get(*args)

        def _get_prices():
            prices = None
            if "prices" in cache:
                prices = cache["prices"]
            elif "prices" in kwargs:
                if not _get_kwarg("returns_data", False):
                    prices = _get_kwarg("prices")
            return prices

        def _get_returns():
            returns = None
            if "returns" in cache:
                returns = cache["returns"]
            elif "returns" in kwargs:
                returns = _get_kwarg("returns")
            elif "prices" in kwargs and _get_kwarg("returns_data", False):
                returns = _get_kwarg("prices")
            return returns

        def _prices_from_returns():
            from pypfopt.expected_returns import prices_from_returns

            cache["prices"] = prices_from_returns(_get_returns(), _get_kwarg("log_returns", False))
            return cache["prices"]

        def _returns_from_prices():
            from pypfopt.expected_returns import returns_from_prices

            cache["returns"] = returns_from_prices(_get_prices(), _get_kwarg("log_returns", False))
            return cache["returns"]

        if arg_name == "expected_returns":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "expected_returns" not in cache:
                cache["expected_returns"] = resolve_pypfopt_expected_returns(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["expected_returns"]
        elif arg_name == "cov_matrix":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "cov_matrix" not in cache:
                cache["cov_matrix"] = resolve_pypfopt_cov_matrix(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["cov_matrix"]
        elif arg_name == "optimizer":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "optimizer" not in cache:
                cache["optimizer"] = resolve_pypfopt_optimizer(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["optimizer"]

        if orig_arg_name not in pass_kwargs:
            if arg_name in kwargs:
                if arg_name == "market_prices":
                    if pypfopt_func.__name__ != "market_implied_risk_aversion" and checks.is_series(
                        _get_kwarg(arg_name)
                    ):
                        pass_kwargs[orig_arg_name] = _get_kwarg(arg_name).to_frame().copy(deep=False)
                    else:
                        pass_kwargs[orig_arg_name] = _get_kwarg(arg_name).copy(deep=False)
                else:
                    pass_kwargs[orig_arg_name] = _get_kwarg(arg_name)
            else:
                if arg_name == "frequency":
                    ann_factor = ReturnsAccessor.get_ann_factor(_get_kwarg("year_freq", None), _get_kwarg("freq", None))
                    if ann_factor is not None:
                        pass_kwargs[orig_arg_name] = ann_factor
                elif arg_name == "prices":
                    if "returns_data" in arg_names:
                        if "returns_data" in kwargs:
                            if _get_kwarg("returns_data", False):
                                if _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _get_returns()
                                elif _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _returns_from_prices()
                            else:
                                if _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _get_prices()
                                elif _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _prices_from_returns()
                        else:
                            if _get_prices() is not None:
                                pass_kwargs[orig_arg_name] = _get_prices()
                                pass_kwargs["returns_data"] = False
                            elif _get_returns() is not None:
                                pass_kwargs[orig_arg_name] = _get_returns()
                                pass_kwargs["returns_data"] = True
                    else:
                        if _get_prices() is not None:
                            pass_kwargs[orig_arg_name] = _get_prices()
                        elif _get_returns() is not None:
                            pass_kwargs[orig_arg_name] = _prices_from_returns()
                elif arg_name == "returns":
                    if _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = _get_returns()
                    elif _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = _returns_from_prices()
                elif arg_name == "latest_prices":
                    from pypfopt.discrete_allocation import get_latest_prices

                    if _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(_get_prices())
                    elif _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(_prices_from_returns())
                elif arg_name == "delta":
                    if "delta" not in cache:
                        from pypfopt.black_litterman import market_implied_risk_aversion

                        cache["delta"] = resolve_pypfopt_func_call(
                            market_implied_risk_aversion,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["delta"]
                elif arg_name == "pi":
                    if "pi" not in cache:
                        from pypfopt.black_litterman import market_implied_prior_returns

                        cache["pi"] = resolve_pypfopt_func_call(
                            market_implied_prior_returns,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["pi"]

        if orig_arg_name not in pass_kwargs:
            if arg_value.default != inspect.Parameter.empty:
                pass_kwargs[orig_arg_name] = arg_value.default

    for arg_name, arg_value in signature.parameters.items():
        if arg_value.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"Variable positional arguments in {pypfopt_func} cannot be parsed")
        elif arg_value.kind == inspect.Parameter.VAR_KEYWORD:
            if var_kwarg_names is None:
                var_kwarg_names = []
            for var_arg_name in var_kwarg_names:
                _process_arg(var_arg_name, arg_value)
        else:
            _process_arg(arg_name, arg_value)

    return pass_kwargs


def resolve_pypfopt_func_call(pypfopt_func: tp.Callable, **kwargs) -> tp.Any:
    """Resolve arguments using `resolve_pypfopt_func_kwargs` and call the function with that arguments."""
    return pypfopt_func(**resolve_pypfopt_func_kwargs(pypfopt_func, **kwargs))


def resolve_pypfopt_expected_returns(
    expected_returns: tp.Union[tp.Callable, tp.AnyArray, str] = "mean_historical_return",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the expected returns.

    `expected_returns` can be an array, an attribute of `pypfopt.expected_returns`, a function,
    or one of the following options:

    * 'mean_historical_return': `pypfopt.expected_returns.mean_historical_return`
    * 'ema_historical_return': `pypfopt.expected_returns.ema_historical_return`
    * 'capm_return': `pypfopt.expected_returns.capm_return`
    * 'bl_returns': `pypfopt.black_litterman.BlackLittermanModel.bl_returns`

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(expected_returns, str):
        if expected_returns.lower() == "mean_historical_return":
            from pypfopt.expected_returns import mean_historical_return

            return resolve_pypfopt_func_call(mean_historical_return, **kwargs)
        if expected_returns.lower() == "ema_historical_return":
            from pypfopt.expected_returns import ema_historical_return

            return resolve_pypfopt_func_call(ema_historical_return, **kwargs)
        if expected_returns.lower() == "capm_return":
            from pypfopt.expected_returns import capm_return

            return resolve_pypfopt_func_call(capm_return, **kwargs)
        if expected_returns.lower() == "bl_returns":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            ).bl_returns()
        import pypfopt.expected_returns

        if hasattr(pypfopt.expected_returns, expected_returns):
            return resolve_pypfopt_func_call(getattr(pypfopt.expected_returns, expected_returns), **kwargs)
        raise NotImplementedError("Return model '{}' is not supported".format(expected_returns))
    if callable(expected_returns):
        return resolve_pypfopt_func_call(expected_returns, **kwargs)
    return expected_returns


def resolve_pypfopt_cov_matrix(
    cov_matrix: tp.Union[tp.Callable, tp.AnyArray, str] = "ledoit_wolf",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the covariance matrix.

    `cov_matrix` can be an array, an attribute of `pypfopt.risk_models`, a function,
    or one of the following options:

    * 'sample_cov': `pypfopt.risk_models.sample_cov`
    * 'semicovariance' or 'semivariance': `pypfopt.risk_models.semicovariance`
    * 'exp_cov': `pypfopt.risk_models.exp_cov`
    * 'ledoit_wolf' or 'ledoit_wolf_constant_variance': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'constant_variance' as shrinkage factor
    * 'ledoit_wolf_single_factor': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'single_factor' as shrinkage factor
    * 'ledoit_wolf_constant_correlation': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'constant_correlation' as shrinkage factor
    * 'oracle_approximating': `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf`
        with 'oracle_approximating' as shrinkage factor

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(cov_matrix, str):
        if cov_matrix.lower() == "sample_cov":
            from pypfopt.risk_models import sample_cov

            return resolve_pypfopt_func_call(sample_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "semicovariance" or cov_matrix.lower() == "semivariance":
            from pypfopt.risk_models import semicovariance

            return resolve_pypfopt_func_call(semicovariance, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "exp_cov":
            from pypfopt.risk_models import exp_cov

            return resolve_pypfopt_func_call(exp_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "ledoit_wolf" or cov_matrix.lower() == "ledoit_wolf_constant_variance":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf()
        if cov_matrix.lower() == "ledoit_wolf_single_factor":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="single_factor"
            )
        if cov_matrix.lower() == "ledoit_wolf_constant_correlation":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="constant_correlation"
            )
        if cov_matrix.lower() == "oracle_approximating":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).oracle_approximating()
        import pypfopt.risk_models

        if hasattr(pypfopt.risk_models, cov_matrix):
            return resolve_pypfopt_func_call(getattr(pypfopt.risk_models, cov_matrix), **kwargs)
        raise NotImplementedError("Risk model '{}' is not supported".format(cov_matrix))
    if callable(cov_matrix):
        return resolve_pypfopt_func_call(cov_matrix, **kwargs)
    return cov_matrix


def resolve_pypfopt_optimizer(
    optimizer: tp.Union[tp.Callable, BaseOptimizerT, str] = "efficient_frontier",
    **kwargs,
) -> BaseOptimizerT:
    """Resolve the optimizer.

    `optimizer` can be an instance of `pypfopt.base_optimizer.BaseOptimizer`, an attribute of `pypfopt`,
    a subclass of  `pypfopt.base_optimizer.BaseOptimizer`, or one of the following options:

    * 'efficient_frontier': `pypfopt.efficient_frontier.EfficientFrontier`
    * 'efficient_cdar': `pypfopt.efficient_frontier.EfficientCDaR`
    * 'efficient_cvar': `pypfopt.efficient_frontier.EfficientCVaR`
    * 'efficient_semivariance': `pypfopt.efficient_frontier.EfficientSemivariance`
    * 'black_litterman' or 'bl': `pypfopt.black_litterman.BlackLittermanModel`
    * 'hierarchical_portfolio', 'hrpopt', or 'hrp': `pypfopt.hierarchical_portfolio.HRPOpt`
    * 'cla': `pypfopt.cla.CLA`

    Any function is resolved using `resolve_pypfopt_func_call`."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")
    from pypfopt.base_optimizer import BaseOptimizer

    if isinstance(optimizer, str):
        if optimizer.lower() == "efficient_frontier":
            from pypfopt.efficient_frontier import EfficientFrontier

            return resolve_pypfopt_func_call(EfficientFrontier, **kwargs)
        if optimizer.lower() == "efficient_cdar":
            from pypfopt.efficient_frontier import EfficientCDaR

            return resolve_pypfopt_func_call(EfficientCDaR, **kwargs)
        if optimizer.lower() == "efficient_cvar":
            from pypfopt.efficient_frontier import EfficientCVaR

            return resolve_pypfopt_func_call(EfficientCVaR, **kwargs)
        if optimizer.lower() == "efficient_semivariance":
            from pypfopt.efficient_frontier import EfficientSemivariance

            return resolve_pypfopt_func_call(EfficientSemivariance, **kwargs)
        if optimizer.lower() == "black_litterman" or optimizer.lower() == "bl":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            )
        if optimizer.lower() == "hierarchical_portfolio" or optimizer.lower() == "hrpopt" or optimizer.lower() == "hrp":
            from pypfopt.hierarchical_portfolio import HRPOpt

            return resolve_pypfopt_func_call(HRPOpt, **kwargs)
        if optimizer.lower() == "cla":
            from pypfopt.cla import CLA

            return resolve_pypfopt_func_call(CLA, **kwargs)
        import pypfopt

        if hasattr(pypfopt, optimizer):
            return resolve_pypfopt_func_call(getattr(pypfopt, optimizer), **kwargs)
        raise NotImplementedError("Optimizer '{}' is not supported".format(optimizer))
    if isinstance(optimizer, type) and issubclass(optimizer, BaseOptimizer):
        return resolve_pypfopt_func_call(optimizer, **kwargs)
    if isinstance(optimizer, BaseOptimizer):
        return optimizer
    raise NotImplementedError("Optimizer {} is not supported".format(optimizer))


def pypfopt_optimize(
    target: tp.Optional[tp.Union[tp.Callable, str]] = None,
    target_is_convex: tp.Optional[bool] = None,
    weights_sum_to_one: tp.Optional[bool] = None,
    target_constraints: tp.Optional[tp.List[tp.Kwargs]] = None,
    target_solver: tp.Optional[str] = None,
    target_initial_guess: tp.Optional[tp.Array] = None,
    objectives: tp.Optional[tp.MaybeIterable[tp.Union[tp.Callable, str]]] = None,
    constraints: tp.Optional[tp.MaybeIterable[tp.Callable]] = None,
    sector_mapper: tp.Optional[dict] = None,
    sector_lower: tp.Optional[dict] = None,
    sector_upper: tp.Optional[dict] = None,
    discrete_allocation: tp.Optional[bool] = None,
    allocation_method: tp.Optional[str] = None,
    silence_warnings: tp.Optional[bool] = None,
    ignore_opt_errors: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Dict[str, float]:
    """Get allocation using PyPortfolioOpt.

    First, it resolves the optimizer using `resolve_pypfopt_optimizer`. Depending upon which arguments it takes,
    it may further resolve expected returns, covariance matrix, etc. Then, it adds objectives and constraints
    to the optimizer instance, calls the target metric, extracts the weights, and finally, converts
    the weights to an integer allocation (if requested).

    To specify the optimizer, use `optimizer` (see `resolve_pypfopt_optimizer`).
    To specify the expected returns, use `expected_returns` (see `resolve_pypfopt_expected_returns`).
    To specify the covariance matrix, use `cov_matrix` (see `resolve_pypfopt_cov_matrix`).
    All other keyword arguments in `**kwargs` are used by `resolve_pypfopt_func_call`.

    Each objective can be a function, an attribute of `pypfopt.objective_functions`, or an iterable of such.

    Each constraint can be a function or an interable of such.

    The target can be an attribute of the optimizer, or a stand-alone function.
    If `target_is_convex` is True, the function is added as a convex function.
    Otherwise, the function is added as a non-convex function. The keyword arguments
    `weights_sum_to_one` and those starting with `target` are passed
    `pypfopt.base_optimizer.BaseConvexOptimizer.convex_objective`
    and `pypfopt.base_optimizer.BaseConvexOptimizer.nonconvex_objective` respectively.
    Set `ignore_opt_errors` to True to ignore any target optimization errors.

    If `discrete_allocation` is True, resolves `pypfopt.discrete_allocation.DiscreteAllocation`
    and calls `allocation_method` as an attribute of the allocation object.

    Any function is resolved using `resolve_pypfopt_func_call`.

    For defaults, see `pypfopt` under `vectorbtpro._settings.pfopt`.

    Usage:
        * Using mean historical returns, Ledoit-Wolf covariance matrix with constant variance,
        and efficient frontier:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> data = vbt.YFData.fetch(["MSFT", "AMZN", "KO", "MA"])
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> vbt.pypfopt_optimize(prices=data.get("Close"))
        {'MSFT': 0.13324, 'AMZN': 0.10016, 'KO': 0.03229, 'MA': 0.73431}
        ```

        * EMA historical returns and sample covariance:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     cov_matrix="sample_cov"
        ... )
        {'MSFT': 0.08984, 'AMZN': 0.0, 'KO': 0.91016, 'MA': 0.0}
        ```

        * EMA historical returns, efficient Conditional Value at Risk, and other parameters automatically
        passed to their respective functions. Optimized towards lowest CVaR:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     optimizer="efficient_cvar",
        ...     beta=0.9,
        ...     weight_bounds=(-1, 1),
        ...     target="min_cvar"
        ... )
        {'MSFT': 0.14779, 'AMZN': 0.07224, 'KO': 0.77552, 'MA': 0.00445}
        ```

        * Adding custom objectives:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     objectives=["L2_reg"],
        ...     gamma=0.1,
        ...     target="min_volatility"
        ... )
        {'MSFT': 0.22228, 'AMZN': 0.15685, 'KO': 0.28712, 'MA': 0.33375}
        ```

        * Adding custom constraints:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     constraints=[lambda w: w[data.symbols.index("MSFT")] <= 0.1]
        ... )
        {'MSFT': 0.1, 'AMZN': 0.10676, 'KO': 0.04341, 'MA': 0.74982}
        ```

        * Optimizing towards a custom convex objective (to add a non-convex objective,
        set `target_is_convex` to False):

        ```pycon
        >>> import cvxpy as cp

        >>> def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
        ...     log_sum = cp.sum(cp.log(w))
        ...     var = cp.quad_form(w, cov_matrix)
        ...     return var - k * log_sum

        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     target=logarithmic_barrier_objective
        ... )
        {'MSFT': 0.24595, 'AMZN': 0.23047, 'KO': 0.25862, 'MA': 0.26496}
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")
    from pypfopt.exceptions import OptimizationError
    from cvxpy.error import SolverError

    from vectorbtpro._settings import settings

    pypfopt_cfg = dict(settings["pfopt"]["pypfopt"])

    def _resolve_setting(k, v):
        setting = pypfopt_cfg.pop(k)
        if v is None:
            return setting
        return v

    target = _resolve_setting("target", target)
    target_is_convex = _resolve_setting("target_is_convex", target_is_convex)
    weights_sum_to_one = _resolve_setting("weights_sum_to_one", weights_sum_to_one)
    target_constraints = _resolve_setting("target_constraints", target_constraints)
    target_solver = _resolve_setting("target_solver", target_solver)
    target_initial_guess = _resolve_setting("target_initial_guess", target_initial_guess)
    objectives = _resolve_setting("objectives", objectives)
    constraints = _resolve_setting("constraints", constraints)
    sector_mapper = _resolve_setting("sector_mapper", sector_mapper)
    sector_lower = _resolve_setting("sector_lower", sector_lower)
    sector_upper = _resolve_setting("sector_upper", sector_upper)
    discrete_allocation = _resolve_setting("discrete_allocation", discrete_allocation)
    allocation_method = _resolve_setting("allocation_method", allocation_method)
    silence_warnings = _resolve_setting("silence_warnings", silence_warnings)
    ignore_opt_errors = _resolve_setting("ignore_opt_errors", ignore_opt_errors)
    kwargs = merge_dicts(pypfopt_cfg, kwargs)

    if "cache" not in kwargs:
        kwargs["cache"] = {}
    if "used_arg_names" not in kwargs:
        kwargs["used_arg_names"] = set()

    with warnings.catch_warnings():
        if silence_warnings:
            warnings.simplefilter("ignore")

        optimizer = kwargs["optimizer"] = resolve_pypfopt_optimizer(**kwargs)

        if objectives is not None:
            if not checks.is_iterable(objectives) or isinstance(objectives, str):
                objectives = [objectives]
            for objective in objectives:
                if isinstance(objective, str):
                    import pypfopt.objective_functions

                    objective = getattr(pypfopt.objective_functions, objective)
                objective_kwargs = resolve_pypfopt_func_kwargs(objective, **kwargs)
                optimizer.add_objective(objective, **objective_kwargs)
        if constraints is not None:
            if not checks.is_iterable(constraints):
                constraints = [constraints]
            for constraint in constraints:
                optimizer.add_constraint(constraint)
        if sector_mapper is not None:
            if sector_lower is None:
                sector_lower = {}
            if sector_upper is None:
                sector_upper = {}
            optimizer.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

        try:
            if isinstance(target, str):
                resolve_pypfopt_func_call(getattr(optimizer, target), **kwargs)
            else:
                if target_is_convex:
                    optimizer.convex_objective(
                        target,
                        weights_sum_to_one=weights_sum_to_one,
                        **resolve_pypfopt_func_kwargs(target, **kwargs),
                    )
                else:
                    optimizer.nonconvex_objective(
                        target,
                        objective_args=tuple(resolve_pypfopt_func_kwargs(target, **kwargs).values()),
                        weights_sum_to_one=weights_sum_to_one,
                        constraints=target_constraints,
                        solver=target_solver,
                        initial_guess=target_initial_guess,
                    )
        except (OptimizationError, SolverError, ValueError) as e:
            if isinstance(e, ValueError) and "expected return exceeding the risk-free rate" not in str(e):
                raise e
            if ignore_opt_errors:
                warnings.warn(str(e), stacklevel=2)
                return {}
            raise e

        weights = kwargs["weights"] = resolve_pypfopt_func_call(optimizer.clean_weights, **kwargs)
        if discrete_allocation:
            from pypfopt.discrete_allocation import DiscreteAllocation

            allocator = resolve_pypfopt_func_call(DiscreteAllocation, **kwargs)
            return resolve_pypfopt_func_call(getattr(allocator, allocation_method), **kwargs)[0]

        passed_arg_names = set(kwargs.keys())
        passed_arg_names.remove("cache")
        passed_arg_names.remove("used_arg_names")
        passed_arg_names.remove("optimizer")
        passed_arg_names.remove("weights")
        unused_arg_names = passed_arg_names.difference(kwargs["used_arg_names"])
        if len(unused_arg_names) > 0:
            warnings.warn(f"Some arguments were not used: {unused_arg_names}", stacklevel=2)

        if not discrete_allocation:
            weights = {k: 1 if v >= 1 else v for k, v in weights.items()}

    return dict(weights)


# ############# Riskfolio-Lib ############# #


def prepare_returns(
    returns: tp.AnyArray2d,
    nan_to_zero: bool = True,
    dropna_rows: bool = True,
    dropna_cols: bool = True,
    dropna_any: bool = True,
) -> tp.Frame:
    """Prepare returns."""
    returns = to_pd_array(returns)
    if returns.size == 0:
        return returns
    if nan_to_zero or dropna_rows or dropna_cols or dropna_any:
        returns = returns.replace([np.inf, -np.inf], np.nan)
    if nan_to_zero:
        returns = returns.fillna(0.0)
    if dropna_rows or dropna_cols:
        if nan_to_zero:
            valid_mask = returns != 0
        else:
            valid_mask = ~returns.isnull()
        if dropna_rows:
            if nan_to_zero or not dropna_any:
                returns = returns.loc[valid_mask.any(axis=1)]
                if returns.size == 0:
                    return returns
        if dropna_cols:
            returns = returns.loc[:, valid_mask.any(axis=0)]
            if returns.size == 0:
                return returns
    if not nan_to_zero and dropna_any:
        returns = returns.dropna()
    return returns


def resolve_riskfolio_func_kwargs(
    riskfolio_func: tp.Callable,
    unused_arg_names: tp.Optional[tp.Set[str]] = None,
    func_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Kwargs:
    """Select keyword arguments belonging to `riskfolio_func`."""
    func_arg_names = get_func_arg_names(riskfolio_func)
    matched_kwargs = dict()
    for k, v in kwargs.items():
        if k in func_arg_names:
            matched_kwargs[k] = v
            if unused_arg_names is not None:
                if k in unused_arg_names:
                    unused_arg_names.remove(k)
    if func_kwargs is not None:
        return merge_dicts(
            select_pfopt_func_kwargs(riskfolio_func, matched_kwargs),
            select_pfopt_func_kwargs(riskfolio_func, pfopt_func_dict(func_kwargs)),
        )
    return select_pfopt_func_kwargs(riskfolio_func, matched_kwargs)


def resolve_asset_classes(
    asset_classes: tp.Union[None, tp.Frame, tp.Sequence],
    columns: tp.Index,
    col_indices: tp.Optional[tp.Sequence[int]] = None,
) -> tp.Frame:
    """Resolve asset classes for Riskfolio-Lib.

    Supports the following formats:

    * None: Takes columns where the bottom-most level is assumed to be assets
    * Index: Each level in the index must be a different asset class set
    * Nested dict: Each sub-dict must be a different asset class set
    * Sequence of strings or ints: Matches them against level names in the columns.
    If the columns have a single level, or some level names were not found, uses the sequence
    directly as one class asset set named 'Class'.
    * Sequence of dicts: Each dict becomes a row in the new DataFrame
    * DataFrame where the first column is the asset list and the next columns are the
    different asset’s classes sets (this is the target format accepted by Riskfolio-Lib).
    See an example [here](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_constraints).

    !!! note
        If `asset_classes` is neither None nor a DataFrame, the bottom-most level in `columns`
        gets renamed to 'Assets' and becomes the first column of the new DataFrame."""
    if asset_classes is None:
        asset_classes = columns.to_frame().reset_index(drop=True).iloc[:, ::-1]
        asset_classes = asset_classes.rename(columns={asset_classes.columns[0]: "Assets"})
    if not isinstance(asset_classes, pd.DataFrame):
        if isinstance(asset_classes, dict):
            asset_classes = pd.DataFrame(asset_classes)
        elif isinstance(asset_classes, pd.Index):
            asset_classes = asset_classes.to_frame().reset_index(drop=True)
        elif checks.is_sequence(asset_classes) and isinstance(asset_classes[0], int):
            asset_classes = select_levels(columns, asset_classes).to_frame().reset_index(drop=True)
        elif checks.is_sequence(asset_classes) and isinstance(asset_classes[0], str):
            if isinstance(columns, pd.MultiIndex) and set(asset_classes) <= set(columns.names):
                asset_classes = select_levels(columns, asset_classes).to_frame().reset_index(drop=True)
            else:
                asset_classes = pd.Index(asset_classes, name="Class").to_frame().reset_index(drop=True)
        else:
            asset_classes = pd.DataFrame.from_records(asset_classes)
        if isinstance(columns, pd.MultiIndex):
            assets = columns.get_level_values(-1)
        else:
            assets = columns
        if col_indices is not None and len(col_indices) > 0:
            asset_classes = asset_classes.iloc[col_indices]
        asset_classes.insert(loc=0, column="Assets", value=assets)
    return asset_classes


def resolve_assets_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve asset constraints for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_constraints),
    also accepts a sequence of dicts such that each dict becomes a row in a new DataFrame.
    Dicts don't have to specify all column names, the function will autofill any missing elements/columns."""
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Weight",
                "Type Relative",
                "Relative Set",
                "Relative",
                "Factor",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def resolve_factors_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve factors constraints for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_constraints),
    also accepts a sequence of dicts such that each dict becomes a row in a new DataFrame.
    Dicts don't have to specify all column names, the function will autofill any missing elements/columns."""
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Factor",
                "Sign",
                "Value",
                "Relative Factor",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def resolve_assets_views(views: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve asset views for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_views),
    also accepts a sequence of dicts such that each dict becomes a row in a new DataFrame.
    Dicts don't have to specify all column names, the function will autofill any missing elements/columns."""
    if not isinstance(views, pd.DataFrame):
        if isinstance(views, dict):
            views = pd.DataFrame(views)
        else:
            views = pd.DataFrame.from_records(views)
        views.columns = views.columns.str.title()
        new_views = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Return",
                "Type Relative",
                "Relative Set",
                "Relative",
            ],
            dtype=object,
        )
        for c in new_views.columns:
            if c in views.columns:
                new_views[c] = views[c]
        new_views.fillna("", inplace=True)
        new_views["Disabled"].replace("", False, inplace=True)
        views = new_views
    return views


def resolve_factors_views(views: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve factors views for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_views),
    also accepts a sequence of dicts such that each dict becomes a row in a new DataFrame.
    Dicts don't have to specify all column names, the function will autofill any missing elements/columns."""
    if not isinstance(views, pd.DataFrame):
        if isinstance(views, dict):
            views = pd.DataFrame(views)
        else:
            views = pd.DataFrame.from_records(views)
        views.columns = views.columns.str.title()
        new_views = pd.DataFrame(
            columns=[
                "Disabled",
                "Factor",
                "Sign",
                "Value",
                "Relative Factor",
            ],
            dtype=object,
        )
        for c in new_views.columns:
            if c in views.columns:
                new_views[c] = views[c]
        new_views.fillna("", inplace=True)
        new_views["Disabled"].replace("", False, inplace=True)
        views = new_views
    return views


def resolve_hrp_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve HRP constraints for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.hrp_constraints),
    also accepts a sequence of dicts such that each dict becomes a row in a new DataFrame.
    Dicts don't have to specify all column names, the function will autofill any missing elements/columns."""
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Weight",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def riskfolio_optimize(
    returns: tp.AnyArray2d,
    nan_to_zero: tp.Optional[bool] = None,
    dropna_rows: tp.Optional[bool] = None,
    dropna_cols: tp.Optional[bool] = None,
    dropna_any: tp.Optional[bool] = None,
    factors: tp.Optional[tp.AnyArray2d] = None,
    port: tp.Optional[RPortfolioT] = None,
    port_cls: tp.Union[None, str, tp.Type] = None,
    opt_method: tp.Union[None, str, tp.Callable] = None,
    stats_methods: tp.Optional[tp.Sequence[str]] = None,
    model: tp.Optional[str] = None,
    asset_classes: tp.Union[None, tp.Frame, tp.Sequence] = None,
    constraints_method: tp.Optional[str] = None,
    constraints: tp.Union[None, tp.Frame, tp.Sequence] = None,
    views_method: tp.Optional[str] = None,
    views: tp.Union[None, tp.Frame, tp.Sequence] = None,
    solvers: tp.Optional[tp.Sequence[str]] = None,
    sol_params: tp.KwargsLike = None,
    freq: tp.Optional[tp.FrequencyLike] = None,
    year_freq: tp.Optional[tp.FrequencyLike] = None,
    pre_opt: bool = False,
    pre_opt_kwargs: tp.KwargsLike = None,
    pre_opt_as_w: bool = False,
    func_kwargs: tp.KwargsLike = None,
    silence_warnings: bool = True,
    return_port: bool = False,
    **kwargs,
) -> tp.Union[tp.Dict[str, float], tp.Tuple[tp.Dict[str, float], RPortfolioT]]:
    """Get allocation using Riskfolio-Lib.

    Args:
        returns (array_like): A dataframe that contains the returns of the assets.
        nan_to_zero (bool): Whether to convert NaN values to zero.
        dropna_rows (bool): Whether to drop rows with all NaN/zero values.

            Gets applied only if `nan_to_zero` is True or `dropna_any` is False.
        dropna_cols (bool): Whether to drop columns with all NaN/zero values.
        dropna_any (bool): Whether to drop any NaN values.

            Gets applied only if `nan_to_zero` is False.
        factors (array_like): A dataframe that contains the factors.
        port (Portfolio or HCPortfolio): Already initialized portfolio.
        port_cls (str or type): Portfolio class.

            Supports the following values:

            * None: Uses `Portfolio`
            * 'hc' or 'hcportfolio' (case-insensitive): Uses `HCPortfolio`
            * Other string: Uses attribute of `riskfolio`
            * Class: Uses a custom class
        opt_method (str or callable): Optimization method.

            Supports the following values:

            * None or 'optimization': Uses `port.optimization` (where `port` is a portfolio instance)
            * 'wc' or 'wc_optimization': Uses `port.wc_optimization`
            * 'rp' or 'rp_optimization': Uses `port.rp_optimization`
            * 'rrp' or 'rrp_optimization': Uses `port.rrp_optimization`
            * 'owa' or 'owa_optimization': Uses `port.owa_optimization`
            * String: Uses attribute of `port`
            * Callable: Uses a custom optimization function
        stats_methods (str or sequence of str): Sequence of stats methods to call before optimization.

            If None, tries to automatically populate the sequence using `opt_method` and `model`.
            For example, calls `port.assets_stats` if `model="Classic"` is used.
            Also, if `func_kwargs` is not empty, adds all functions whose name ends with '_stats'.
        model (str): The model used to optimize the portfolio.
        asset_classes (any): Asset classes matrix.

            See `resolve_asset_classes` for possible formats.
        constraints_method (str): Constraints method.

            Supports the following values:

            * 'assets' or 'assets_constraints': [assets constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_constraints)
            * 'factors' or 'factors_constraints': [factors constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_constraints)
            * 'hrp' or 'hrp_constraints': [HRP constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.hrp_constraints)

            If None and the class `Portfolio` is used, will use factors constraints if `factors_stats` is used,
            otherwise assets constraints. If the class `HCPortfolio` is used, will use HRP constraints.
        constraints (any): Constraints matrix.

            See `resolve_assets_constraints` for possible formats of assets constraints,
            `resolve_factors_constraints` for possible formats of factors constraints, and
            `resolve_hrp_constraints` for possible formats of HRP constraints.
        views_method (str): Views method.

            Supports the following values:

            * 'assets' or 'assets_views': [assets views](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_views)
            * 'factors' or 'factors_views': [factors views](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_views)

            If None, will use factors views if `blfactors_stats` is used, otherwise assets views.
        views (any): Views matrix.

            See `resolve_assets_views` for possible formats of assets views and
            `resolve_factors_views` for possible formats of factors views.
        solvers (list of str): Solvers.
        sol_params (dict): Solver parameters.
        freq (frequency_like): Frequency to be used to compute the annualization factor.

            Make sure to provide it when using views.
        year_freq (frequency_like): Year frequency to be used to compute the annualization factor.

            Make sure to provide it when using views.
        pre_opt (bool): Whether to pre-optimize the portfolio with `pre_opt_kwargs`.
        pre_opt_kwargs (dict): Call `riskfolio_optimize` with these keyword arguments
            and use the returned portfolio for further optimization.
        pre_opt_as_w (bool): Whether to use the weights as `w` from the pre-optimization step.
        func_kwargs (dict): Further keyword arguments by function.

            Can be used to override any arguments from `kwargs` matched with the function,
            or to add more arguments. Will be wrapped with `pfopt_func_dict` and passed to
            `select_pfopt_func_kwargs` when calling each Riskfolio-Lib's function.
        silence_warnings (bool): Whether to silence all warnings.
        return_port (bool): Whether to also return the portfolio.
        **kwargs: Keyword arguments that will be passed to any Riskfolio-Lib's function
            that needs them (i.e., lists any of them in its signature).

    For defaults, see `riskfolio` under `vectorbtpro._settings.pfopt`.

    Usage:
        * Classic Mean Risk Optimization:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> data = vbt.YFData.fetch(["MSFT", "AMZN", "KO", "MA"])
        >>> returns = data.close.vbt.to_returns()
        ```

        [=100% "100%"]{: .candystripe}

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     method_mu='hist', method_cov='hist', d=0.94,  # assets_stats
        ...     model='Classic', rm='MV', obj='Sharpe', hist=True, rf=0, l=0  # optimization
        ... )
        {'MSFT': 0.26297126323056036,
         'AMZN': 0.13984467450137006,
         'KO': 0.35870315943426767,
         'MA': 0.238480902833802}
        ```

        * The same by splitting arguments:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     func_kwargs=dict(
        ...         assets_stats=dict(method_mu='hist', method_cov='hist', d=0.94),
        ...         optimization=dict(model='Classic', rm='MV', obj='Sharpe', hist=True, rf=0, l=0)
        ...     )
        ... )
        {'MSFT': 0.26297126323056036,
         'AMZN': 0.13984467450137006,
         'KO': 0.35870315943426767,
         'MA': 0.238480902833802}
        ```

        * Asset constraints:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     constraints=[
        ...         {
        ...             "Type": "Assets",
        ...             "Position": "MSFT",
        ...             "Sign": "<=",
        ...             "Weight": 0.01
        ...         }
        ...     ]
        ... )
        {'MSFT': 0.009999990814976588,
         'AMZN': 0.19788481506569947,
         'KO': 0.4553600308839969,
         'MA': 0.336755163235327}
        ```

        * Asset class constraints:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     asset_classes=["C1", "C1", "C2", "C2"],
        ...     constraints=[
        ...         {
        ...             "Type": "Classes",
        ...             "Set": "Class",
        ...             "Position": "C1",
        ...             "Sign": "<=",
        ...             "Weight": 0.1
        ...         }
        ...     ]
        ... )
        {'MSFT': 0.03501297245802569,
         'AMZN': 0.06498702655063979,
         'KO': 0.4756624658301967,
         'MA': 0.4243375351611379}
        ```

        * Hierarchical Risk Parity (HRP) Portfolio Optimization:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     port_cls="HCPortfolio",
        ...     model='HRP',
        ...     codependence='pearson',
        ...     rm='MV',
        ...     rf=0,
        ...     linkage='single',
        ...     max_k=10,
        ...     leaf_order=True
        ... )
        {'MSFT': 0.19091632057853536,
         'AMZN': 0.11069893826556164,
         'KO': 0.28589872132122485,
         'MA': 0.41248601983467814}
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("riskfolio")
    import riskfolio as rp
    from vectorbtpro._settings import settings

    riskfolio_cfg = dict(settings["pfopt"]["riskfolio"])
    wrapping_cfg = settings["wrapping"]
    returns_cfg = settings["returns"]

    def _resolve_setting(k, v):
        setting = riskfolio_cfg.pop(k)
        if v is None:
            return setting
        return v

    nan_to_zero = _resolve_setting("nan_to_zero", nan_to_zero)
    dropna_rows = _resolve_setting("dropna_rows", dropna_rows)
    dropna_cols = _resolve_setting("dropna_cols", dropna_cols)
    dropna_any = _resolve_setting("dropna_any", dropna_any)
    factors = _resolve_setting("factors", factors)
    port = _resolve_setting("port", port)
    port_cls = _resolve_setting("port_cls", port_cls)
    opt_method = _resolve_setting("opt_method", opt_method)
    stats_methods = _resolve_setting("stats_methods", stats_methods)
    model = _resolve_setting("model", model)
    asset_classes = _resolve_setting("asset_classes", asset_classes)
    constraints_method = _resolve_setting("constraints_method", constraints_method)
    constraints = _resolve_setting("constraints", constraints)
    views_method = _resolve_setting("views_method", views_method)
    views = _resolve_setting("views", views)
    solvers = _resolve_setting("solvers", solvers)
    sol_params = _resolve_setting("sol_params", sol_params)
    freq = _resolve_setting("freq", freq)
    if freq is None:
        freq = wrapping_cfg["freq"]
    year_freq = _resolve_setting("year_freq", year_freq)
    if year_freq is None:
        year_freq = returns_cfg["year_freq"]
    pre_opt = _resolve_setting("pre_opt", pre_opt)
    pre_opt_kwargs = merge_dicts(riskfolio_cfg.pop("pre_opt_kwargs"), pre_opt_kwargs)
    pre_opt_as_w = _resolve_setting("pre_opt_as_w", pre_opt_as_w)
    func_kwargs = merge_dicts(riskfolio_cfg.pop("func_kwargs"), func_kwargs)
    silence_warnings = _resolve_setting("silence_warnings", silence_warnings)
    return_port = _resolve_setting("return_port", return_port)
    kwargs = merge_dicts(riskfolio_cfg, kwargs)
    if pre_opt_kwargs is None:
        pre_opt_kwargs = {}
    if func_kwargs is None:
        func_kwargs = {}
    func_kwargs = pfopt_func_dict(func_kwargs)
    unused_arg_names = set(kwargs.keys())

    with warnings.catch_warnings():
        if silence_warnings:
            warnings.simplefilter("ignore")

        # Prepare returns
        new_returns = prepare_returns(
            returns,
            nan_to_zero=nan_to_zero,
            dropna_rows=dropna_rows,
            dropna_cols=dropna_cols,
            dropna_any=dropna_any,
        )
        col_indices = [i for i, c in enumerate(returns.columns) if c in new_returns.columns]
        returns = new_returns
        if returns.size == 0:
            return {}

        # Pre-optimize
        if pre_opt:
            w, port = riskfolio_optimize(returns, port=port, return_port=True, **pre_opt_kwargs)
            if pre_opt_as_w:
                w = pd.DataFrame.from_records([w]).T.rename(columns={0: "weights"})
                kwargs["w"] = w
                unused_arg_names.add("w")

        # Build portfolio
        if port_cls is None:
            port_cls = rp.Portfolio
        elif isinstance(port_cls, str) and port_cls.lower() in ("hc", "hcportfolio"):
            port_cls = rp.HCPortfolio
        elif isinstance(port_cls, str):
            port_cls = getattr(rp, port_cls)
        else:
            port_cls = port_cls
        matched_kwargs = resolve_riskfolio_func_kwargs(
            port_cls,
            unused_arg_names=unused_arg_names,
            func_kwargs=func_kwargs,
            **kwargs,
        )
        if port is None:
            port = port_cls(returns, **matched_kwargs)
        else:
            for k, v in matched_kwargs.items():
                setattr(port, k, v)
        if solvers is not None:
            port.solvers = list(solvers)
        if sol_params is not None:
            port.sol_params = dict(sol_params)
        if factors is not None:
            factors = to_pd_array(factors).dropna()
            port.factors = factors

        # Resolve optimization and stats methods
        if opt_method is None:
            if len(func_kwargs) > 0:
                for name_or_func in func_kwargs:
                    if isinstance(name_or_func, str):
                        if name_or_func.endswith("optimization"):
                            if opt_method is not None:
                                raise ValueError("Function keyword arguments list multiple optimization methods")
                            opt_method = name_or_func
        if opt_method is None:
            opt_method = "optimization"
        if stats_methods is None:
            if len(func_kwargs) > 0:
                for name_or_func in func_kwargs:
                    if isinstance(name_or_func, str):
                        if name_or_func.endswith("_stats"):
                            if stats_methods is None:
                                stats_methods = []
                            stats_methods.append(name_or_func)
        if isinstance(port, rp.Portfolio):
            if isinstance(opt_method, str) and opt_method.lower() == "optimization":
                opt_func = port.optimization
                if model is None:
                    opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                    model = opt_func_kwargs.get("model", "Classic")
                if model.lower() == "classic":
                    model = "Classic"
                    if stats_methods is None:
                        stats_methods = ["assets_stats"]
                elif model.lower() == "fm":
                    model = "FM"
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "factors_stats"]
                elif model.lower() == "bl":
                    model = "BL"
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "blacklitterman_stats"]
                elif model.lower() in ("bl_fm", "blfm"):
                    model = "BL_FM"
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "factors_stats", "blfactors_stats"]
            elif isinstance(opt_method, str) and opt_method.lower() in ("wc", "wc_optimization"):
                opt_func = port.wc_optimization
                if stats_methods is None:
                    stats_methods = ["assets_stats", "wc_stats"]
            elif isinstance(opt_method, str) and opt_method.lower() in ("rp", "rp_optimization"):
                opt_func = port.rp_optimization
                if model is None:
                    opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                    model = opt_func_kwargs.get("model", "Classic")
                if model.lower() == "classic":
                    model = "Classic"
                    if stats_methods is None:
                        stats_methods = ["assets_stats"]
                elif model.lower() == "fm":
                    model = "FM"
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "factors_stats"]
            elif isinstance(opt_method, str) and opt_method.lower() in ("rrp", "rrp_optimization"):
                opt_func = port.rrp_optimization
                if model is None:
                    opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                    model = opt_func_kwargs.get("model", "Classic")
                if model.lower() == "classic":
                    model = "Classic"
                    if stats_methods is None:
                        stats_methods = ["assets_stats"]
                elif model.lower() == "fm":
                    model = "FM"
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "factors_stats"]
            elif isinstance(opt_method, str) and opt_method.lower() in ("owa", "owa_optimization"):
                opt_func = port.owa_optimization
                if stats_methods is None:
                    stats_methods = ["assets_stats"]
            elif isinstance(opt_method, str):
                opt_func = getattr(port, opt_method)
            else:
                opt_func = opt_method
        else:
            if isinstance(opt_method, str):
                opt_func = getattr(port, opt_method)
            else:
                opt_func = opt_method
        if model is not None:
            kwargs["model"] = model
            unused_arg_names.add("model")
        if stats_methods is None:
            stats_methods = []

        # Apply constraints
        if constraints is not None:
            if constraints_method is None:
                if isinstance(port, rp.Portfolio):
                    if "factors_stats" in stats_methods:
                        constraints_method = "factors"
                    else:
                        constraints_method = "assets"
                elif isinstance(port, rp.HCPortfolio):
                    constraints_method = "hrp"
                else:
                    raise ValueError("Constraints method is required")
            if constraints_method.lower() in ("assets", "assets_constraints"):
                asset_classes = resolve_asset_classes(asset_classes, returns.columns, col_indices)
                kwargs["asset_classes"] = asset_classes
                unused_arg_names.add("asset_classes")
                constraints = resolve_assets_constraints(constraints)
                kwargs["constraints"] = constraints
                unused_arg_names.add("constraints")
                matched_kwargs = resolve_riskfolio_func_kwargs(
                    rp.assets_constraints,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                port.ainequality, port.binequality = warn_stdout(rp.assets_constraints)(**matched_kwargs)
            elif constraints_method.lower() in ("factors", "factors_constraints"):
                if "loadings" not in kwargs:
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.loadings_matrix,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    if "X" not in matched_kwargs:
                        matched_kwargs["X"] = port.factors
                    if "Y" not in matched_kwargs:
                        matched_kwargs["Y"] = port.returns
                    loadings = warn_stdout(rp.loadings_matrix)(**matched_kwargs)
                    kwargs["loadings"] = loadings
                    unused_arg_names.add("loadings")
                constraints = resolve_factors_constraints(constraints)
                kwargs["constraints"] = constraints
                unused_arg_names.add("constraints")

                matched_kwargs = resolve_riskfolio_func_kwargs(
                    rp.factors_constraints,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                port.ainequality, port.binequality = warn_stdout(rp.factors_constraints)(**matched_kwargs)
            elif constraints_method.lower() in ("hrp", "hrp_constraints"):
                asset_classes = resolve_asset_classes(asset_classes, returns.columns, col_indices)
                kwargs["asset_classes"] = asset_classes
                unused_arg_names.add("asset_classes")
                constraints = resolve_hrp_constraints(constraints)
                kwargs["constraints"] = constraints
                unused_arg_names.add("constraints")
                matched_kwargs = resolve_riskfolio_func_kwargs(
                    rp.hrp_constraints,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                port.w_max, port.w_min = warn_stdout(rp.hrp_constraints)(**matched_kwargs)
            else:
                raise ValueError(f"Constraints method '{constraints_method}' is not supported")

        # Resolve views
        if views is not None:
            if views_method is None:
                if "blfactors_stats" in stats_methods:
                    views_method = "factors"
                else:
                    views_method = "assets"
            if views_method.lower() in ("assets", "assets_views"):
                asset_classes = resolve_asset_classes(asset_classes, returns.columns, col_indices)
                kwargs["asset_classes"] = asset_classes
                unused_arg_names.add("asset_classes")
                views = resolve_assets_views(views)
                kwargs["views"] = views
                unused_arg_names.add("views")
                matched_kwargs = resolve_riskfolio_func_kwargs(
                    rp.assets_views,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                P, Q = warn_stdout(rp.assets_views)(**matched_kwargs)
                ann_factor = ReturnsAccessor.get_ann_factor(year_freq, freq)
                if ann_factor is not None:
                    Q /= ann_factor
                else:
                    warnings.warn(f"Set frequency and year frequency to adjust expected returns", stacklevel=2)
                kwargs["P"] = P
                unused_arg_names.add("P")
                kwargs["Q"] = Q
                unused_arg_names.add("Q")
            elif views_method.lower() in ("factors", "factors_views"):
                if "loadings" not in kwargs:
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.loadings_matrix,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    if "X" not in matched_kwargs:
                        matched_kwargs["X"] = port.factors
                    if "Y" not in matched_kwargs:
                        matched_kwargs["Y"] = port.returns
                    loadings = warn_stdout(rp.loadings_matrix)(**matched_kwargs)
                    kwargs["loadings"] = loadings
                    unused_arg_names.add("loadings")
                if "B" not in kwargs:
                    kwargs["B"] = kwargs["loadings"]
                    unused_arg_names.add("B")
                views = resolve_factors_views(views)
                kwargs["views"] = views
                unused_arg_names.add("views")
                matched_kwargs = resolve_riskfolio_func_kwargs(
                    rp.factors_views,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                P_f, Q_f = warn_stdout(rp.factors_views)(**matched_kwargs)
                ann_factor = ReturnsAccessor.get_ann_factor(year_freq, freq)
                if ann_factor is not None:
                    Q_f /= ann_factor
                else:
                    warnings.warn(f"Set frequency and year frequency to adjust expected returns", stacklevel=2)
                kwargs["P_f"] = P_f
                unused_arg_names.add("P_f")
                kwargs["Q_f"] = Q_f
                unused_arg_names.add("Q_f")
            else:
                raise ValueError(f"Views method '{constraints_method}' is not supported")

        # Run stats
        for stats_method in stats_methods:
            stats_func = getattr(port, stats_method)
            matched_kwargs = resolve_riskfolio_func_kwargs(
                stats_func,
                unused_arg_names=unused_arg_names,
                func_kwargs=func_kwargs,
                **kwargs,
            )
            warn_stdout(stats_func)(**matched_kwargs)

        # Run optimization
        matched_kwargs = resolve_riskfolio_func_kwargs(
            opt_func,
            unused_arg_names=unused_arg_names,
            func_kwargs=func_kwargs,
            **kwargs,
        )
        weights = warn_stdout(opt_func)(**matched_kwargs)

        # Post-process weights
        if len(unused_arg_names) > 0:
            warnings.warn(f"Some arguments were not used: {unused_arg_names}", stacklevel=2)
        if weights is None:
            weights = {}
        if isinstance(weights, pd.DataFrame):
            if "weights" not in weights.columns:
                raise ValueError("Weights column wasn't returned")
            weights = weights["weights"]
    if return_port:
        return dict(weights), port
    return dict(weights)


# ############# PortfolioOptimizer ############# #


PortfolioOptimizerT = tp.TypeVar("PortfolioOptimizerT", bound="PortfolioOptimizer")


class PortfolioOptimizer(Analyzable):
    """Class that exposes methods for generating allocations."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "alloc_records",
        "allocations",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        alloc_records: tp.Union[AllocRanges, AllocPoints],
        allocations: tp.Array2d,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            alloc_records=alloc_records,
            allocations=allocations,
            **kwargs,
        )

        self._alloc_records = alloc_records
        self._allocations = allocations

        # Only slices of rows can be selected
        self._range_only_select = True

    def indexing_func(
        self: PortfolioOptimizerT,
        *args,
        wrapper_meta: tp.DictLike = None,
        alloc_wrapper_meta: tp.DictLike = None,
        alloc_records_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Perform indexing on `PortfolioOptimizer`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        if alloc_records_meta is None:
            alloc_records_meta = self.alloc_records.indexing_func_meta(
                *args,
                wrapper_meta=alloc_wrapper_meta,
                **kwargs,
            )
        new_alloc_records = self.alloc_records.indexing_func(
            *args,
            records_meta=alloc_records_meta,
            **kwargs,
        )
        new_allocations = to_2d_array(self._allocations)[alloc_records_meta["new_indices"]]
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            alloc_records=new_alloc_records,
            allocations=new_allocations,
        )

    def resample(self: PortfolioOptimizerT, *args, **kwargs) -> PortfolioOptimizerT:
        """Perform resampling on `PortfolioOptimizer`."""
        new_wrapper = self.wrapper.resample(*args, **kwargs)
        new_alloc_records = self.alloc_records.resample(*args, **kwargs)
        return self.replace(
            wrapper=new_wrapper,
            alloc_records=new_alloc_records,
        )

    # ############# Class methods ############# #

    @classmethod
    def from_allocate_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocate_func: tp.Callable,
        *args,
        every: tp.Union[None, tp.FrequencyLike, Param] = point_idxr_defaults["every"],
        normalize_every: tp.Union[bool, Param] = point_idxr_defaults["normalize_every"],
        at_time: tp.Union[None, tp.TimeLike, Param] = point_idxr_defaults["at_time"],
        start: tp.Union[None, int, tp.DatetimeLike, Param] = point_idxr_defaults["start"],
        end: tp.Union[None, int, tp.DatetimeLike, Param] = point_idxr_defaults["end"],
        exact_start: tp.Union[bool, Param] = point_idxr_defaults["exact_start"],
        on: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = point_idxr_defaults["on"],
        add_delta: tp.Union[None, tp.FrequencyLike, Param] = point_idxr_defaults["add_delta"],
        kind: tp.Union[None, str, Param] = point_idxr_defaults["kind"],
        indexer_method: tp.Union[None, str, Param] = point_idxr_defaults["indexer_method"],
        indexer_tolerance: tp.Union[None, str, Param] = point_idxr_defaults["indexer_tolerance"],
        skip_minus_one: tp.Union[bool, Param] = point_idxr_defaults["skip_minus_one"],
        index_points: tp.Union[None, tp.MaybeSequence[int], Param] = None,
        search_max_len: tp.Optional[int] = None,
        search_max_depth: tp.Optional[int] = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        group_configs: tp.Union[None, tp.Dict[tp.Hashable, tp.Kwargs], tp.Sequence[tp.Kwargs]] = None,
        pre_group_func: tp.Optional[tp.Callable] = None,
        jitted_loop: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        template_context: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        random_subset: tp.Optional[int] = None,
        index_stack_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an allocation function.

        Generates date points and allocates at those points.

        Similar to `PortfolioOptimizer.from_optimize_func`, but generates points using
        `vectorbtpro.base.wrapping.ArrayWrapper.get_index_points` and makes each point available
        as `index_point` in the context.

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.allocate_meta_nb`.

        Also, in contrast to `PortfolioOptimizer.from_optimize_func`, creates records of type
        `vectorbtpro.portfolio.pfopt.records.AllocPoints`.

        Usage:
            * Allocate uniformly:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np

            >>> data = vbt.YFData.fetch(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def uniform_allocate_func(n_cols):
            ...     return np.full(n_cols, 1 / n_cols)

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     uniform_allocate_func,
            ...     close.shape[1]
            ... )
            >>> pf_opt.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00+00:00  0.333333  0.333333  0.333333
            ```

            * Allocate randomly every first date of the year:

            ```pycon
            >>> def random_allocate_func(n_cols):
            ...     weights = np.random.uniform(size=n_cols)
            ...     return weights / weights.sum()

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     every="AS-JAN"
            ... )
            >>> pf_opt.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2011-01-03 00:00:00+00:00  0.160335  0.122434  0.717231
            2012-01-03 00:00:00+00:00  0.071386  0.469564  0.459051
            2013-01-02 00:00:00+00:00  0.125853  0.168480  0.705668
            2014-01-02 00:00:00+00:00  0.391565  0.169205  0.439231
            2015-01-02 00:00:00+00:00  0.115075  0.602844  0.282081
            2016-01-04 00:00:00+00:00  0.244070  0.046547  0.709383
            2017-01-03 00:00:00+00:00  0.316065  0.335000  0.348935
            2018-01-02 00:00:00+00:00  0.422142  0.252154  0.325704
            2019-01-02 00:00:00+00:00  0.368748  0.195147  0.436106
            ```

            * Specify index points manually:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60]
            ... )
            >>> pf_opt.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00+00:00  0.257878  0.308287  0.433835
            2010-02-17 00:00:00+00:00  0.090927  0.471980  0.437094
            2010-03-31 00:00:00+00:00  0.395855  0.148516  0.455629
            ```

            * Specify allocations manually:

            ```pycon
            >>> def manual_allocate_func(weights):
            ...     return weights

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     manual_allocate_func,
            ...     vbt.RepEval("weights[i]", context=dict(weights=[
            ...         [1, 0, 0],
            ...         [0, 1, 0],
            ...         [0, 0, 1]
            ...     ])),
            ...     index_points=[0, 30, 60]
            ... )
            >>> pf_opt.allocations
            symbol                     MSFT  AMZN  AAPL
            Date
            2010-01-04 00:00:00+00:00     1     0     0
            2010-02-17 00:00:00+00:00     0     1     0
            2010-03-31 00:00:00+00:00     0     0     1
            ```

            * Use Numba-compiled loop:

            ```pycon
            >>> from numba import njit

            >>> @njit
            ... def random_allocate_func_nb(i, idx, n_cols):
            ...     weights = np.random.uniform(0, 1, n_cols)
            ...     return weights / weights.sum()

            >>> pf_opt = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func_nb,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60],
            ...     jitted_loop=True
            ... )
            >>> pf_opt.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00+00:00  0.231925  0.351085  0.416990
            2010-02-17 00:00:00+00:00  0.163050  0.070292  0.766658
            2010-03-31 00:00:00+00:00  0.497465  0.500215  0.002319
            ```

            !!! hint
                There is no big reason of using the Numba-compiled loop, apart from when having
                to rebalance many thousands of times. Usually, using a regular Python loop
                and a Numba-compiled allocation function should suffice.
        """
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        if pbar_kwargs is None:
            pbar_kwargs = {}

        # Prepare group config names
        gc_names = []
        gc_names_none = True
        n_configs = 0
        if group_configs is not None:
            if isinstance(group_configs, dict):
                new_group_configs = []
                for k, v in group_configs.items():
                    v = dict(v)
                    v["_name"] = k
                    new_group_configs.append(v)
                group_configs = new_group_configs
            else:
                group_configs = list(group_configs)
            for i, group_config in enumerate(group_configs):
                group_config = dict(group_config)
                if "args" in group_config:
                    for k, arg in enumerate(group_config.pop("args")):
                        group_config[f"args_{k}"] = arg
                if "kwargs" in group_config:
                    for k, v in enumerate(group_config.pop("kwargs")):
                        group_config[k] = v
                if "_name" in group_config and group_config["_name"] is not None:
                    gc_names.append(group_config.pop("_name"))
                    gc_names_none = False
                else:
                    gc_names.append(n_configs)
                group_configs[i] = group_config
                n_configs += 1
        else:
            group_configs = []

        # Combine parameters
        paramable_kwargs = {
            "every": every,
            "normalize_every": normalize_every,
            "at_time": at_time,
            "start": start,
            "end": end,
            "exact_start": exact_start,
            "on": on,
            "add_delta": add_delta,
            "kind": kind,
            "indexer_method": indexer_method,
            "indexer_tolerance": indexer_tolerance,
            "skip_minus_one": skip_minus_one,
            "index_points": index_points,
            **{f"args_{i}": args[i] for i in range(len(args))},
            **kwargs,
        }
        param_dct = find_params_in_obj(
            paramable_kwargs,
            search_max_len=search_max_len,
            search_max_depth=search_max_depth,
        )
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                index_stack_kwargs=index_stack_kwargs,
                name_tuple_to_str=name_tuple_to_str,
            )
            product_group_configs = param_product_to_objs(paramable_kwargs, param_product)
            if len(group_configs) == 0:
                group_configs = product_group_configs
            else:
                new_group_configs = []
                for i in range(len(product_group_configs)):
                    for group_config in group_configs:
                        new_group_config = merge_dicts(product_group_configs[i], group_config)
                        new_group_configs.append(new_group_config)
                group_configs = new_group_configs

        # Build group index
        n_config_params = len(gc_names)
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = param_columns
            else:
                group_index = combine_indexes(
                    (
                        param_columns,
                        pd.Index(gc_names, name="group_config"),
                    ),
                    **index_stack_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = pd.Index(["group"], name="group")
            else:
                group_index = pd.Index(gc_names, name="group_config")

        # Create group config from arguments if empty
        if len(group_configs) == 0:
            single_group = True
            group_configs.append(dict())
        else:
            single_group = False

        # Resolve each group
        groupable_kwargs = {
            "allocate_func": allocate_func,
            **paramable_kwargs,
            "jitted_loop": jitted_loop,
            "jitted": jitted,
            "chunked": chunked,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
        }
        new_group_configs = []
        for group_config in group_configs:
            new_group_config = merge_dicts(groupable_kwargs, group_config)
            _args = ()
            while True:
                if f"args_{len(_args)}" in new_group_config:
                    _args += (new_group_config.pop(f"args_{len(_args)}"),)
                else:
                    break
            new_group_config["args"] = _args
            new_group_configs.append(new_group_config)
        group_configs = new_group_configs

        # Generate allocations
        alloc_points = []
        allocations = []
        if show_progress is None:
            show_progress = len(group_configs) > 1
        with get_pbar(total=len(group_configs), show_progress=show_progress, **pbar_kwargs) as pbar:
            for g, group_config in enumerate(group_configs):
                pbar.set_description(str(group_index[g]))

                group_config = dict(group_config)
                if pre_group_func is not None:
                    pre_group_func(group_config)

                _allocate_func = group_config.pop("allocate_func")
                _every = group_config.pop("every")
                _normalize_every = group_config.pop("normalize_every")
                _at_time = group_config.pop("at_time")
                _start = group_config.pop("start")
                _end = group_config.pop("end")
                _exact_start = group_config.pop("exact_start")
                _on = group_config.pop("on")
                _add_delta = group_config.pop("add_delta")
                _kind = group_config.pop("kind")
                _indexer_method = group_config.pop("indexer_method")
                _indexer_tolerance = group_config.pop("indexer_tolerance")
                _skip_minus_one = group_config.pop("skip_minus_one")
                _index_points = group_config.pop("index_points")
                _jitted_loop = group_config.pop("jitted_loop")
                _jitted = group_config.pop("jitted")
                _chunked = group_config.pop("chunked")
                _template_context = group_config.pop("template_context")
                _execute_kwargs = group_config.pop("execute_kwargs")
                _args = group_config.pop("args")
                _kwargs = group_config

                _template_context = merge_dicts(
                    dict(
                        group_configs=group_configs,
                        group_index=group_index,
                        group_idx=g,
                        wrapper=wrapper,
                        allocate_func=_allocate_func,
                        every=_every,
                        normalize_every=_normalize_every,
                        at_time=_at_time,
                        start=_start,
                        end=_end,
                        exact_start=_exact_start,
                        on=_on,
                        add_delta=_add_delta,
                        kind=_kind,
                        indexer_method=_indexer_method,
                        indexer_tolerance=_indexer_tolerance,
                        skip_minus_one=_skip_minus_one,
                        index_points=_index_points,
                        jitted_loop=_jitted_loop,
                        jitted=_jitted,
                        chunked=_chunked,
                        execute_kwargs=_execute_kwargs,
                        args=_args,
                        kwargs=_kwargs,
                    ),
                    _template_context,
                )

                if _index_points is None:
                    get_index_points_kwargs = substitute_templates(
                        dict(
                            every=_every,
                            normalize_every=_normalize_every,
                            at_time=_at_time,
                            start=_start,
                            end=_end,
                            exact_start=_exact_start,
                            on=_on,
                            add_delta=_add_delta,
                            kind=_kind,
                            indexer_method=_indexer_method,
                            indexer_tolerance=_indexer_tolerance,
                            skip_minus_one=_skip_minus_one,
                        ),
                        _template_context,
                        sub_id="get_index_points_defaults",
                        strict=True,
                    )
                    _index_points = wrapper.get_index_points(**get_index_points_kwargs)
                    _template_context = merge_dicts(
                        _template_context,
                        get_index_points_kwargs,
                        dict(index_points=_index_points),
                    )
                else:
                    _index_points = substitute_templates(
                        _index_points,
                        _template_context,
                        sub_id="index_points",
                        strict=True,
                    )
                    _index_points = to_1d_array(_index_points)
                    _template_context = merge_dicts(_template_context, dict(index_points=_index_points))

                if jitted_loop:
                    _allocate_func = substitute_templates(
                        _allocate_func,
                        _template_context,
                        sub_id="allocate_func",
                        strict=True,
                    )
                    _args = substitute_templates(_args, _template_context, sub_id="args")
                    _kwargs = substitute_templates(_kwargs, _template_context, sub_id="kwargs")
                    func = jit_reg.resolve_option(nb.allocate_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    _allocations = func(len(wrapper.columns), _index_points, _allocate_func, *_args, **_kwargs)
                else:
                    funcs_args = []
                    for i in range(len(_index_points)):
                        __template_context = merge_dicts(dict(i=i, index_point=_index_points[i]), _template_context)
                        __allocate_func = substitute_templates(
                            _allocate_func,
                            __template_context,
                            sub_id="allocate_func",
                            strict=True,
                        )
                        __args = substitute_templates(_args, __template_context, sub_id="args")
                        __kwargs = substitute_templates(_kwargs, __template_context, sub_id="kwargs")
                        funcs_args.append((__allocate_func, __args, __kwargs))

                    _execute_kwargs = merge_dicts(
                        dict(
                            show_progress=False,
                            pbar_kwargs=pbar_kwargs,
                        ),
                        _execute_kwargs,
                    )
                    results = execute(funcs_args, **_execute_kwargs)
                    _allocations = pd.DataFrame(results, columns=wrapper.columns)
                    if isinstance(_allocations.columns, pd.RangeIndex):
                        _allocations = _allocations.values
                    else:
                        _allocations = _allocations[list(wrapper.columns)].values

                _alloc_points, _allocations = nb.prepare_alloc_points_nb(_index_points, _allocations, g)
                alloc_points.append(_alloc_points)
                allocations.append(_allocations)

                pbar.update(1)

        # Build column hierarchy
        new_columns = combine_indexes((group_index, wrapper.columns), **index_stack_kwargs)

        # Create instance
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.names if group_index.nlevels > 1 else group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_points = AllocPoints(
            ArrayWrapper(
                index=wrapper.index,
                columns=new_wrapper.get_columns(),
                ndim=new_wrapper.get_ndim(),
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
            ),
            np.concatenate(alloc_points),
        )
        allocations = np.row_stack(allocations)
        return cls(new_wrapper, alloc_points, allocations)

    @classmethod
    def from_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocations: tp.ArrayLike,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from an array.

        Uses `PortfolioOptimizer.from_allocate_func`.

        If `allocations` is a NumPy array, uses `vectorbtpro.portfolio.pfopt.nb.pick_idx_allocate_func_nb`
        and a Numba-compiled loop. Otherwise, uses a regular Python function to pick each allocation
        (which can be a dict, Series, etc.).

        If `allocations` is a DataFrame, additionally uses its index as labels."""
        if isinstance(allocations, pd.DataFrame):
            kwargs = merge_dicts(
                dict(on=allocations.index, kind="labels"),
                kwargs,
            )
            allocations = allocations.values
        if isinstance(allocations, np.ndarray):

            def _resolve_allocations(index_points):
                if len(index_points) != len(allocations):
                    raise ValueError(f"Allocation array must have {len(index_points)} rows")
                return to_2d_array(allocations, expand_axis=0)

            return cls.from_allocate_func(
                wrapper,
                nb.pick_idx_allocate_func_nb,
                RepFunc(_resolve_allocations),
                jitted_loop=True,
                **kwargs,
            )

        def _pick_allocate_func(index_points, i):
            if len(index_points) != len(allocations):
                raise ValueError(f"Allocation array must have {len(index_points)} rows")
            return allocations[i]

        return cls.from_allocate_func(wrapper, _pick_allocate_func, Rep("index_points"), Rep("i"), **kwargs)

    @classmethod
    def from_filled_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        allocations: tp.AnyArray2d,
        valid_only: bool = True,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from an already filled array.

        Uses `PortfolioOptimizer.from_allocate_func`.

        Uses `vectorbtpro.portfolio.pfopt.nb.pick_point_allocate_func_nb` and a Numba-compiled loop.

        Extracts allocation points using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`."""
        if wrapper is None:
            if checks.is_frame(allocations):
                wrapper = ArrayWrapper.from_obj(allocations)
            else:
                raise TypeError("Wrapper is required if allocations is not a DataFrame")
        allocations = to_2d_array(allocations, expand_axis=0)
        if allocations.shape != wrapper.shape_2d:
            raise ValueError("Allocation array must have the same shape as wrapper")
        on = nb.get_alloc_points_nb(
            allocations,
            valid_only=valid_only,
            nonzero_only=nonzero_only,
            unique_only=unique_only,
        )
        kwargs = merge_dicts(dict(on=on), kwargs)
        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            allocations,
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_uniform(cls: tp.Type[PortfolioOptimizerT], wrapper: ArrayWrapper, **kwargs) -> PortfolioOptimizerT:
        """Generate uniform allocations.

        Uses `PortfolioOptimizer.from_allocate_func`."""

        def _uniform_allocate_func():
            return np.full(wrapper.shape_2d[1], 1 / wrapper.shape_2d[1])

        return cls.from_allocate_func(wrapper, _uniform_allocate_func, **kwargs)

    @classmethod
    def from_random(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        direction: tp.Union[str, int] = "longonly",
        n: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate random allocations.

        Uses `PortfolioOptimizer.from_allocate_func`.

        Uses `vectorbtpro.portfolio.pfopt.nb.random_allocate_func_nb` and a Numba-compiled loop."""
        if isinstance(direction, str):
            direction = map_enum_fields(direction, Direction)
        if seed is not None:
            set_seed_nb(seed)
        return cls.from_allocate_func(
            wrapper,
            nb.random_allocate_func_nb,
            wrapper.shape_2d[1],
            direction,
            n,
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_universal_algo(
        cls: tp.Type[PortfolioOptimizerT],
        algo: tp.Union[str, tp.Type[AlgoT], AlgoT, AlgoResultT],
        S: tp.Optional[tp.AnyArray2d] = None,
        n_jobs: int = 1,
        log_progress: bool = False,
        valid_only: bool = True,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations using [Universal Portfolios](https://github.com/Marigold/universal-portfolios).

        `S` can be any price, while `algo` must be either an attribute of the package, subclass of
        `universal.algo.Algo`, instance of `universal.algo.Algo`, or instance of `universal.result.AlgoResult`.

        Extracts allocation points using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("universal")
        from universal.algo import Algo
        from universal.result import AlgoResult

        if wrapper is None:
            if S is None or not checks.is_frame(S):
                raise TypeError("Wrapper is required if allocations is not a DataFrame")
            else:
                wrapper = ArrayWrapper.from_obj(S)

        def _pre_group_func(group_config, _algo=algo):
            _ = group_config.pop("args", ())
            if isinstance(_algo, str):
                import universal.algos

                _algo = getattr(universal.algos, _algo)
            if isinstance(_algo, type) and issubclass(_algo, Algo):
                reserved_arg_names = get_func_arg_names(cls.from_allocate_func)
                algo_keys = set(group_config.keys()).difference(reserved_arg_names)
                algo_kwargs = {}
                for k in algo_keys:
                    algo_kwargs[k] = group_config.pop(k)
                _algo = _algo(**algo_kwargs)
            if isinstance(_algo, Algo):
                if S is None:
                    raise ValueError("S is required")
                _algo = _algo.run(S, n_jobs=n_jobs, log_progress=log_progress)
            if isinstance(_algo, AlgoResult):
                weights = _algo.weights[wrapper.columns].values
            else:
                raise TypeError(f"Algo {_algo} not supported")
            if "on" not in kwargs:
                group_config["on"] = nb.get_alloc_points_nb(
                    weights, valid_only=valid_only, nonzero_only=nonzero_only, unique_only=unique_only
                )
            group_config["args"] = (weights,)

        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            jitted_loop=True,
            pre_group_func=_pre_group_func,
            **kwargs,
        )

    @classmethod
    def from_optimize_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        optimize_func: tp.Callable,
        *args,
        every: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults["every"],
        normalize_every: tp.Union[bool, Param] = range_idxr_defaults["normalize_every"],
        split_every: tp.Union[bool, Param] = range_idxr_defaults["split_every"],
        start_time: tp.Union[None, tp.TimeLike, Param] = range_idxr_defaults["start_time"],
        end_time: tp.Union[None, tp.TimeLike, Param] = range_idxr_defaults["end_time"],
        lookback_period: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults["lookback_period"],
        start: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = range_idxr_defaults["start"],
        end: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = range_idxr_defaults["end"],
        exact_start: tp.Union[bool, Param] = range_idxr_defaults["exact_start"],
        fixed_start: tp.Union[bool, Param] = range_idxr_defaults["fixed_start"],
        closed_start: tp.Union[bool, Param] = range_idxr_defaults["closed_start"],
        closed_end: tp.Union[bool, Param] = range_idxr_defaults["closed_end"],
        add_start_delta: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults["add_start_delta"],
        add_end_delta: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults["add_end_delta"],
        kind: tp.Union[None, str, Param] = range_idxr_defaults["kind"],
        skip_minus_one: tp.Union[bool, Param] = range_idxr_defaults["skip_minus_one"],
        index_ranges: tp.Union[None, tp.MaybeSequence[tp.MaybeSequence[int]], Param] = None,
        index_loc: tp.Union[None, tp.MaybeSequence[int], Param] = None,
        alloc_wait: tp.Union[int, Param] = 1,
        search_max_len: tp.Optional[int] = None,
        search_max_depth: tp.Optional[int] = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        group_configs: tp.Union[None, tp.Dict[tp.Hashable, tp.Kwargs], tp.Sequence[tp.Kwargs]] = None,
        pre_group_func: tp.Optional[tp.Callable] = None,
        jitted_loop: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        template_context: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        random_subset: tp.Optional[int] = None,
        index_stack_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an optimization function.

        Generates date ranges, performs optimization on the subset of data that belongs to each date range,
        and allocates at the end of each range.

        This is a parameterized method that allows testing multiple combinations on most arguments.
        First, it checks whether any of the arguments is wrapped with `vectorbtpro.utils.params.Param`
        and combines their values. It then combines them over `group_configs`, if provided.
        Before execution, it additionally processes the group config using `pre_group_func`.

        It then resolves the date ranges, either using the ready-to-use `index_ranges` or
        by passing all the arguments ranging from `every` to `jitted` to
        `vectorbtpro.base.wrapping.ArrayWrapper.get_index_ranges`. The optimization
        function `optimize_func` is then called on each date range by first substituting
        any templates found in `*args` and `**kwargs`. To forward any reserved arguments
        such as `jitted` to the optimization function, specify their names in `forward_args`
        and `forward_kwargs`.

        !!! note
            Make sure to use vectorbt's own templates to select the current date range
            (available as `index_slice` in the context mapping) from each array.

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.optimize_meta_nb`.
        Otherwise, must take template-substituted `*args` and `**kwargs`, and return an array or
        dictionary with asset allocations (also empty).

        !!! note
            When `jitted_loop` is True and in case of multiple groups, use templates
            to substitute by the current group index (available as `group_idx` in the context mapping).

        All allocations of all groups are stacked into one big 2-dim array where columns are assets
        and rows are allocations. Furthermore, date ranges are used to fill a record array of type
        `vectorbtpro.portfolio.pfopt.records.AllocRanges` that acts as an indexer for allocations.
        For example, the field `col` stores the group index corresponding to each allocation. Since
        this record array does not hold any information on assets themselves, it has its own wrapper
        that holds groups instead of columns, while the wrapper of the `PortfolioOptimizer` instance
        contains regular columns grouped by groups.

        Usage:
            * Allocate once:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> data = vbt.YFData.fetch(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def optimize_func(df):
            ...     sharpe = df.mean() / df.std()
            ...     return sharpe / sharpe.sum()

            >>> df_arg = vbt.RepEval("close.iloc[index_slice]", context=dict(close=close))
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     end="2015-01-01"
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2015-01-02 00:00:00+00:00  0.402459  0.309351  0.288191
            ```

            * Allocate every first date of the year:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN"
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2011-01-03 00:00:00+00:00  0.480693  0.257317  0.261990
                        2012-01-03 00:00:00+00:00  0.489893  0.215381  0.294727
                        2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                        2014-01-02 00:00:00+00:00  0.339649  0.273996  0.386354
                        2015-01-02 00:00:00+00:00  0.350406  0.418638  0.230956
                        2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                        2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                        2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                        2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            ```

            * Specify index ranges manually:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ]
            ... )
            >>> pf_opt.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2010-02-16 00:00:00+00:00  0.340641  0.285897  0.373462
                        2010-03-30 00:00:00+00:00  0.596392  0.206317  0.197291
                        2010-05-12 00:00:00+00:00  0.437481  0.283160  0.279358
            ```

            * Test multiple combinations of one argument:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN",
            ...     start="2015-01-01",
            ...     lookback_period=vbt.Param(["3MS", "6MS"])
            ... )
            >>> pf_opt.allocations
            symbol                                         MSFT      AMZN      AAPL
            lookback_period Date
            3MS             2016-01-04 00:00:00+00:00  0.282725  0.234970  0.482305
                            2017-01-03 00:00:00+00:00  0.318100  0.269355  0.412545
                            2018-01-02 00:00:00+00:00  0.387499  0.236432  0.376068
                            2019-01-02 00:00:00+00:00  0.575464  0.254808  0.169728
            6MS             2016-01-04 00:00:00+00:00  0.265035  0.198619  0.536346
                            2017-01-03 00:00:00+00:00  0.314144  0.409020  0.276836
                            2018-01-02 00:00:00+00:00  0.322741  0.282639  0.394621
                            2019-01-02 00:00:00+00:00  0.565691  0.234760  0.199549
            ```

            * Test multiple cross-argument combinations:

            ```pycon
            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN",
            ...     group_configs=[
            ...         dict(start="2015-01-01"),
            ...         dict(start="2019-06-01", every="MS"),
            ...         dict(end="2014-01-01")
            ...     ]
            ... )
            >>> pf_opt.allocations
            symbol                                      MSFT      AMZN      AAPL
            group_config Date
            0            2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                         2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                         2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                         2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            1            2019-07-01 00:00:00+00:00  0.351461  0.327334  0.321205
                         2019-08-01 00:00:00+00:00  0.418411  0.249799  0.331790
                         2019-09-03 00:00:00+00:00  0.400439  0.374044  0.225517
                         2019-10-01 00:00:00+00:00  0.509387  0.250497  0.240117
                         2019-11-01 00:00:00+00:00  0.349983  0.469181  0.180835
                         2019-12-02 00:00:00+00:00  0.260437  0.380563  0.359000
            2            2012-01-03 00:00:00+00:00  0.489892  0.215381  0.294727
                         2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                         2014-01-02 00:00:00+00:00  0.339649  0.273997  0.386354
            ```

            * Use Numba-compiled loop:

            ```pycon
            >>> from numba import njit
            >>> import numpy as np

            >>> @njit
            ... def optimize_func_nb(i, from_idx, to_idx, close):
            ...     mean = vbt.nb.nanmean_nb(close[from_idx:to_idx])
            ...     std = vbt.nb.nanstd_nb(close[from_idx:to_idx])
            ...     sharpe = mean / std
            ...     return sharpe / np.sum(sharpe)

            >>> pf_opt = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func_nb,
            ...     np.asarray(close),
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ],
            ...     jitted_loop=True
            ... )
            >>> pf_opt.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-02-17 00:00:00+00:00  0.336384  0.289598  0.374017
            2010-03-31 00:00:00+00:00  0.599417  0.207158  0.193425
            2010-05-13 00:00:00+00:00  0.434084  0.281246  0.284670
            ```

            !!! hint
                There is no big reason of using the Numba-compiled loop, apart from when having
                to rebalance many thousands of times. Usually, using a regular Python loop
                and a Numba-compiled optimization function suffice.
        """
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        if pbar_kwargs is None:
            pbar_kwargs = {}

        # Prepare group config names
        gc_names = []
        gc_names_none = True
        n_configs = 0
        if group_configs is not None:
            group_configs = list(group_configs)
            for i, group_config in enumerate(group_configs):
                if isinstance(group_configs, dict):
                    new_group_configs = []
                    for k, v in group_configs.items():
                        v = dict(v)
                        v["_name"] = k
                        new_group_configs.append(v)
                    group_configs = new_group_configs
                else:
                    group_configs = list(group_configs)
                if "args" in group_config:
                    for k, arg in enumerate(group_config.pop("args")):
                        group_config[f"args_{k}"] = arg
                if "kwargs" in group_config:
                    for k, v in enumerate(group_config.pop("kwargs")):
                        group_config[k] = v
                if "_name" in group_config and group_config["_name"] is not None:
                    gc_names.append(group_config.pop("_name"))
                    gc_names_none = False
                else:
                    gc_names.append(n_configs)
                group_configs[i] = group_config
                n_configs += 1
        else:
            group_configs = []

        # Combine parameters
        paramable_kwargs = {
            "every": every,
            "normalize_every": normalize_every,
            "split_every": split_every,
            "start_time": start_time,
            "end_time": end_time,
            "lookback_period": lookback_period,
            "start": start,
            "end": end,
            "exact_start": exact_start,
            "fixed_start": fixed_start,
            "closed_start": closed_start,
            "closed_end": closed_end,
            "add_start_delta": add_start_delta,
            "add_end_delta": add_end_delta,
            "kind": kind,
            "skip_minus_one": skip_minus_one,
            "index_ranges": index_ranges,
            "index_loc": index_loc,
            "alloc_wait": alloc_wait,
            **{f"args_{i}": args[i] for i in range(len(args))},
            **kwargs,
        }
        param_dct = find_params_in_obj(
            paramable_kwargs,
            search_max_len=search_max_len,
            search_max_depth=search_max_depth,
        )
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                index_stack_kwargs=index_stack_kwargs,
                name_tuple_to_str=name_tuple_to_str,
            )
            product_group_configs = param_product_to_objs(paramable_kwargs, param_product)
            if len(group_configs) == 0:
                group_configs = product_group_configs
            else:
                new_group_configs = []
                for i in range(len(product_group_configs)):
                    for group_config in group_configs:
                        new_group_config = merge_dicts(product_group_configs[i], group_config)
                        new_group_configs.append(new_group_config)
                group_configs = new_group_configs

        # Build group index
        n_config_params = len(gc_names)
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = param_columns
            else:
                group_index = combine_indexes(
                    (
                        param_columns,
                        pd.Index(gc_names, name="group_config"),
                    ),
                    **index_stack_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = pd.Index(["group"], name="group")
            else:
                group_index = pd.Index(gc_names, name="group_config")

        # Create group config from arguments if empty
        if len(group_configs) == 0:
            single_group = True
            group_configs.append(dict())
        else:
            single_group = False

        # Resolve each group
        groupable_kwargs = {
            "optimize_func": optimize_func,
            **paramable_kwargs,
            "jitted_loop": jitted_loop,
            "jitted": jitted,
            "chunked": chunked,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
        }
        new_group_configs = []
        for group_config in group_configs:
            new_group_config = merge_dicts(groupable_kwargs, group_config)
            _args = ()
            while True:
                if f"args_{len(_args)}" in new_group_config:
                    _args += (new_group_config.pop(f"args_{len(_args)}"),)
                else:
                    break
            new_group_config["args"] = _args
            new_group_configs.append(new_group_config)
        group_configs = new_group_configs

        alloc_ranges = []
        allocations = []
        if show_progress is None:
            show_progress = len(group_configs) > 1
        with get_pbar(total=len(group_configs), show_progress=show_progress, **pbar_kwargs) as pbar:
            for g, group_config in enumerate(group_configs):
                pbar.set_description(str(group_index[g]))

                group_config = dict(group_config)
                if pre_group_func is not None:
                    pre_group_func(group_config)

                _optimize_func = group_config.pop("optimize_func")
                _every = group_config.pop("every")
                _normalize_every = group_config.pop("normalize_every")
                _split_every = group_config.pop("split_every")
                _start_time = group_config.pop("start_time")
                _end_time = group_config.pop("end_time")
                _lookback_period = group_config.pop("lookback_period")
                _start = group_config.pop("start")
                _end = group_config.pop("end")
                _exact_start = group_config.pop("exact_start")
                _fixed_start = group_config.pop("fixed_start")
                _closed_start = group_config.pop("closed_start")
                _closed_end = group_config.pop("closed_end")
                _add_start_delta = group_config.pop("add_start_delta")
                _add_end_delta = group_config.pop("add_end_delta")
                _kind = group_config.pop("kind")
                _skip_minus_one = group_config.pop("skip_minus_one")
                _index_ranges = group_config.pop("index_ranges")
                _index_loc = group_config.pop("index_loc")
                _alloc_wait = group_config.pop("alloc_wait")
                _jitted_loop = group_config.pop("jitted_loop")
                _jitted = group_config.pop("jitted")
                _chunked = group_config.pop("chunked")
                _template_context = group_config.pop("template_context")
                _execute_kwargs = group_config.pop("execute_kwargs")
                _args = group_config.pop("args")
                _kwargs = group_config

                _template_context = merge_dicts(
                    dict(
                        group_configs=group_configs,
                        group_index=group_index,
                        group_idx=g,
                        wrapper=wrapper,
                        optimize_func=_optimize_func,
                        every=_every,
                        normalize_every=_normalize_every,
                        split_every=_split_every,
                        start_time=_start_time,
                        end_time=_end_time,
                        lookback_period=_lookback_period,
                        start=_start,
                        end=_end,
                        exact_start=_exact_start,
                        fixed_start=_fixed_start,
                        closed_start=_closed_start,
                        closed_end=_closed_end,
                        add_start_delta=_add_start_delta,
                        add_end_delta=_add_end_delta,
                        kind=_kind,
                        skip_minus_one=_skip_minus_one,
                        index_ranges=_index_ranges,
                        index_loc=_index_loc,
                        alloc_wait=_alloc_wait,
                        jitted_loop=_jitted_loop,
                        jitted=_jitted,
                        chunked=_chunked,
                        args=_args,
                        kwargs=_kwargs,
                        execute_kwargs=_execute_kwargs,
                    ),
                    _template_context,
                )

                if _index_ranges is None:
                    get_index_ranges_defaults = substitute_templates(
                        dict(
                            every=_every,
                            normalize_every=_normalize_every,
                            split_every=_split_every,
                            start_time=_start_time,
                            end_time=_end_time,
                            lookback_period=_lookback_period,
                            start=_start,
                            end=_end,
                            exact_start=_exact_start,
                            fixed_start=_fixed_start,
                            closed_start=_closed_start,
                            closed_end=_closed_end,
                            add_start_delta=_add_start_delta,
                            add_end_delta=_add_end_delta,
                            kind=_kind,
                            skip_minus_one=_skip_minus_one,
                            jitted=_jitted,
                        ),
                        _template_context,
                        sub_id="get_index_ranges_defaults",
                        strict=True,
                    )
                    _index_ranges = wrapper.get_index_ranges(**get_index_ranges_defaults)
                    _template_context = merge_dicts(
                        _template_context,
                        get_index_ranges_defaults,
                        dict(index_ranges=_index_ranges),
                    )
                else:
                    _index_ranges = substitute_templates(
                        _index_ranges,
                        _template_context,
                        sub_id="index_ranges",
                        strict=True,
                    )
                    if isinstance(_index_ranges, np.ndarray):
                        _index_ranges = (_index_ranges[:, 0], _index_ranges[:, 1])
                    elif not isinstance(_index_ranges[0], np.ndarray) and not isinstance(_index_ranges[1], np.ndarray):
                        _index_ranges = to_2d_array(_index_ranges, expand_axis=0)
                        _index_ranges = (_index_ranges[:, 0], _index_ranges[:, 1])
                    _template_context = merge_dicts(_template_context, dict(index_ranges=_index_ranges))
                if _index_loc is not None:
                    _index_loc = substitute_templates(
                        _index_loc,
                        _template_context,
                        sub_id="index_loc",
                        strict=True,
                    )
                    _index_loc = to_1d_array(_index_loc)
                    _template_context = merge_dicts(_template_context, dict(index_loc=_index_loc))

                if jitted_loop:
                    _optimize_func = substitute_templates(
                        _optimize_func,
                        _template_context,
                        sub_id="optimize_func",
                        strict=True,
                    )
                    _args = substitute_templates(_args, _template_context, sub_id="args")
                    _kwargs = substitute_templates(_kwargs, _template_context, sub_id="kwargs")
                    func = jit_reg.resolve_option(nb.optimize_meta_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    _allocations = func(
                        len(wrapper.columns),
                        _index_ranges[0],
                        _index_ranges[1],
                        _optimize_func,
                        *_args,
                        **_kwargs,
                    )
                else:
                    funcs_args = []
                    for i in range(len(_index_ranges[0])):
                        index_slice = slice(max(0, _index_ranges[0][i]), _index_ranges[1][i])
                        __template_context = merge_dicts(dict(i=i, index_slice=index_slice), _template_context)
                        __optimize_func = substitute_templates(
                            _optimize_func,
                            __template_context,
                            sub_id="optimize_func",
                            strict=True,
                        )
                        __args = substitute_templates(_args, __template_context, sub_id="args")
                        __kwargs = substitute_templates(_kwargs, __template_context, sub_id="kwargs")
                        funcs_args.append((__optimize_func, __args, __kwargs))

                    _execute_kwargs = merge_dicts(
                        dict(
                            show_progress=False,
                            pbar_kwargs=pbar_kwargs,
                        ),
                        _execute_kwargs,
                    )
                    results = execute(funcs_args, **_execute_kwargs)
                    _allocations = pd.DataFrame(results, columns=wrapper.columns)
                    if isinstance(_allocations.columns, pd.RangeIndex):
                        _allocations = _allocations.values
                    else:
                        _allocations = _allocations[list(wrapper.columns)].values

                if _index_loc is None:
                    _alloc_wait = substitute_templates(
                        _alloc_wait,
                        _template_context,
                        sub_id="alloc_wait",
                        strict=True,
                    )
                    alloc_idx = _index_ranges[1] - 1 + _alloc_wait
                else:
                    alloc_idx = _index_loc
                status = np.where(
                    alloc_idx >= len(wrapper.index),
                    RangeStatus.Open,
                    RangeStatus.Closed,
                )
                _alloc_ranges, _allocations = nb.prepare_alloc_ranges_nb(
                    _index_ranges[0],
                    _index_ranges[1],
                    alloc_idx,
                    status,
                    _allocations,
                    g,
                )
                alloc_ranges.append(_alloc_ranges)
                allocations.append(_allocations)

                pbar.update(1)

        # Build column hierarchy
        new_columns = combine_indexes((group_index, wrapper.columns), **index_stack_kwargs)

        # Create instance
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.names if group_index.nlevels > 1 else group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_ranges = AllocRanges(
            ArrayWrapper(
                index=wrapper.index,
                columns=new_wrapper.get_columns(),
                ndim=new_wrapper.get_ndim(),
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
            ),
            np.concatenate(alloc_ranges),
        )
        allocations = np.row_stack(allocations)
        return cls(new_wrapper, alloc_ranges, allocations)

    @classmethod
    def from_pypfopt(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """`PortfolioOptimizer.from_optimize_func` applied on `pypfopt_optimize`.

        If a wrapper is not provided, parses the wrapper from `prices` or `returns`, if provided."""
        if wrapper is None:
            if "prices" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["prices"])
            elif "returns" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["returns"])
            else:
                raise TypeError("Must provide a wrapper if price and returns are not set")
        if "prices" in kwargs and not isinstance(kwargs["prices"], CustomTemplate):
            kwargs["prices"] = RepFunc(lambda index_slice, _prices=kwargs["prices"]: _prices.iloc[index_slice])
        if "returns" in kwargs and not isinstance(kwargs["returns"], CustomTemplate):
            kwargs["returns"] = RepFunc(lambda index_slice, _returns=kwargs["returns"]: _returns.iloc[index_slice])
        return cls.from_optimize_func(wrapper, pypfopt_optimize, **kwargs)

    @classmethod
    def from_riskfolio(
        cls: tp.Type[PortfolioOptimizerT],
        returns: tp.AnyArray2d,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """`PortfolioOptimizer.from_optimize_func` applied on Riskfolio-Lib."""
        if wrapper is None:
            if not isinstance(returns, CustomTemplate):
                wrapper = ArrayWrapper.from_obj(returns)
            else:
                raise TypeError("Must provide a wrapper if returns are a template")
        if not isinstance(returns, CustomTemplate):
            returns = RepFunc(lambda index_slice, _returns=returns: _returns.iloc[index_slice])
        return cls.from_optimize_func(wrapper, riskfolio_optimize, returns, **kwargs)

    # ############# Properties ############# #

    @property
    def alloc_records(self) -> tp.Union[AllocRanges, AllocPoints]:
        """Allocation ranges of type `vectorbtpro.portfolio.pfopt.records.AllocRanges`
        or points of type `vectorbtpro.portfolio.pfopt.records.AllocPoints`."""
        return self._alloc_records

    def get_allocations(self, squeeze_groups: bool = True) -> tp.Frame:
        """Get a DataFrame with allocation groups concatenated along the index axis."""
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            closed_mask = self.alloc_records.get_field_arr("status") == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            index = self.wrapper.index[idx_arr]
        else:
            index = stack_indexes((self.alloc_records.wrapper.columns[group_arr], self.wrapper.index[idx_arr]))
        n_group_levels = self.wrapper.grouper.get_index().nlevels
        columns = self.wrapper.columns.droplevel(tuple(range(n_group_levels))).unique()
        return pd.DataFrame(allocations, index=index, columns=columns)

    @property
    def allocations(self) -> tp.Frame:
        """Calls `PortfolioOptimizer.get_allocations` with default arguments."""
        return self.get_allocations()

    @property
    def mean_allocation(self) -> tp.Series:
        """Get the mean allocation per column."""
        group_level_names = self.wrapper.grouper.get_index().names
        return self.get_allocations(squeeze_groups=False).groupby(group_level_names).mean().transpose()

    def fill_allocations(
        self,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        wrap_kwargs: tp.KwargsLike = None,
        squeeze_groups: bool = True,
    ) -> tp.Frame:
        """Fill an empty DataFrame with allocations.

        Set `dropna` to 'all' to remove all NaN rows, or to 'head' to remove any rows coming before
        the first allocation."""
        if wrap_kwargs is None:
            wrap_kwargs = {}
        out = self.wrapper.fill(fill_value, group_by=False, **wrap_kwargs)
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            status_arr = self.alloc_records.get_field_arr("status")
            closed_mask = status_arr == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        for g in range(len(self.alloc_records.wrapper.columns)):
            group_mask = group_arr == g
            index_mask = np.full(len(self.wrapper.index), False)
            index_mask[idx_arr[group_mask]] = True
            column_mask = self.wrapper.grouper.get_groups() == g
            out.loc[index_mask, column_mask] = allocations[group_mask]
        if dropna is not None:
            if dropna.lower() == "all":
                out = out.dropna(how="all")
            elif dropna.lower() == "head":
                out = out.iloc[idx_arr.min() :]
            else:
                raise ValueError(f"Invalid option dropna='{dropna}'")
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            n_group_levels = self.wrapper.grouper.get_index().nlevels
            out = out.droplevel(tuple(range(n_group_levels)), axis=1)
        return out

    @property
    def filled_allocations(self) -> tp.Frame:
        """Calls `PortfolioOptimizer.fill_allocations` with default arguments."""
        return self.fill_allocations()

    # ############# Simulation ############# #

    def simulate(self, close: tp.Union[tp.ArrayLike, Data], **kwargs) -> PortfolioT:
        """Run `vectorbtpro.portfolio.base.Portfolio.from_optimizer` on this instance."""
        from vectorbtpro.portfolio.base import Portfolio

        return Portfolio.from_optimizer(close, self, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `PortfolioOptimizer.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.pfopt`."""
        from vectorbtpro._settings import settings

        pfopt_stats_cfg = settings["pfopt"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), pfopt_stats_cfg)

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
            total_records=dict(title="Total Records", calc_func="alloc_records.count", tags="alloc_records"),
            coverage=dict(
                title="Coverage",
                calc_func="alloc_records.get_coverage",
                overlapping=False,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="alloc_records.get_coverage",
                overlapping=True,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            mean_allocation=dict(
                title="Mean Allocation",
                calc_func="mean_allocation",
                post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
                tags="allocations",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        dropna: tp.Optional[str] = "head",
        line_shape: str = "hv",
        plot_rb_dates: tp.Optional[bool] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot allocations.

        Args:
            column (str): Name of the allocation group to plot.
            dropna (int): See `PortfolioOptimizer.fill_allocations`.
            line_shape (str): Line shape.
            plot_rb_dates (bool): Whether to plot rebalancing dates.

                Defaults to True if there are no more than 20 rebalancing dates.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` for rebalancing dates.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            * Continuing with the examples under `PortfolioOptimizer.from_optimize_func`:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> pf_opt = vbt.PortfolioOptimizer.from_random(
            ...     vbt.ArrayWrapper(
            ...         index=pd.date_range("2020-01-01", "2021-01-01"),
            ...         columns=["MSFT", "AMZN", "AAPL"],
            ...         ndim=2
            ...     ),
            ...     every="MS",
            ...     seed=40
            ... )
            >>> pf_opt.plot().show()
            ```

            ![](/assets/images/api/pfopt_plot.svg){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure

        self_group = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if self_group.alloc_records.count() > 0:
            filled_allocations = self_group.fill_allocations(dropna=dropna).ffill()

            fig = filled_allocations.vbt.areaplot(
                line_shape=line_shape,
                trace_kwargs=trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

            if plot_rb_dates is None or (isinstance(plot_rb_dates, bool) and plot_rb_dates):
                rb_dates = self_group.allocations.index
                if plot_rb_dates is None:
                    plot_rb_dates = len(rb_dates) <= 20
                if plot_rb_dates:
                    add_shape_kwargs = merge_dicts(
                        dict(
                            type="line",
                            line=dict(
                                color=fig.layout.template.layout.plot_bgcolor,
                                dash="dot",
                                width=1,
                            ),
                            xref="x",
                            yref="paper",
                            y0=0,
                            y1=1,
                        ),
                        add_shape_kwargs,
                    )
                    for rb_date in rb_dates:
                        fig.add_shape(x0=rb_date, x1=rb_date, **add_shape_kwargs)
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `PortfolioOptimizer.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.pfopt`."""
        from vectorbtpro._settings import settings

        pfopt_plots_cfg = settings["pfopt"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), pfopt_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            alloc_ranges=dict(
                title="Allocation Ranges",
                plot_func="alloc_records.plot",
                check_alloc_ranges=True,
                tags="alloc_ranges",
            ),
            plot=dict(
                title="Allocations",
                plot_func="plot",
                tags="allocations",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


PortfolioOptimizer.override_metrics_doc(__pdoc__)
PortfolioOptimizer.override_subplots_doc(__pdoc__)

PFO = PortfolioOptimizer
"""Shortcut for `PortfolioOptimizer`."""

__pdoc__["PFO"] = False
