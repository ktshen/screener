# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Adapter class for QuantStats.

!!! note
    Accessors do not utilize caching.

We can access the adapter from `ReturnsAccessor`:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbtpro as vbt
>>> import quantstats as qs

>>> np.random.seed(42)
>>> rets = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))
>>> bm_returns = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))

>>> rets.vbt.returns.qs.r_squared(benchmark=bm_returns)
0.0011582111228735541
```

Which is the same as:

```pycon
>>> qs.stats.r_squared(rets, bm_returns)
```

So why not just using `qs.stats`?

First, we can define all parameters such as benchmark returns once and avoid passing them repeatedly
to every function. Second, vectorbt automatically translates parameters passed to `ReturnsAccessor`
for the use in quantstats.

```pycon
>>> # Defaults that vectorbt understands
>>> ret_acc = rets.vbt.returns(
...     bm_returns=bm_returns,
...     freq='d',
...     year_freq='365d',
...     defaults=dict(risk_free=0.001)
... )

>>> ret_acc.qs.r_squared()
0.0011582111228735541

>>> ret_acc.qs.sharpe()
-1.9158923252075455

>>> # Defaults that only quantstats understands
>>> qs_defaults = dict(
...     benchmark=bm_returns,
...     periods=365,
...     rf=0.001
... )
>>> ret_acc_qs = rets.vbt.returns.qs(defaults=qs_defaults)

>>> ret_acc_qs.r_squared()
0.0011582111228735541

>>> ret_acc_qs.sharpe()
-1.9158923252075455
```

The adapter automatically passes the returns to the particular function.
It also merges the defaults defined in the settings, the defaults passed to `ReturnsAccessor`,
and the defaults passed to `QSAdapter` itself, and matches them with the argument names listed
in the function's signature.

For example, the `periods` argument defaults to the annualization factor
`ReturnsAccessor.ann_factor`, which itself is based on the `freq` argument. This makes the results
produced by quantstats and vectorbt at least somewhat similar.

```pycon
>>> vbt.settings.wrapping['freq'] = 'h'
>>> vbt.settings.returns['year_freq'] = '365d'

>>> rets.vbt.returns.sharpe_ratio()  # ReturnsAccessor
-9.38160953971508

>>> rets.vbt.returns.qs.sharpe()  # quantstats via QSAdapter
-9.38160953971508
```

We can still override any argument by overriding its default or by passing it directly to the function:

```pycon
>>> rets.vbt.returns.qs(defaults=dict(periods=252)).sharpe()
-1.5912029345745982

>>> rets.vbt.returns.qs.sharpe(periods=252)
-1.5912029345745982

>>> qs.stats.sharpe(rets)
-1.5912029345745982
```
"""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("quantstats")

from inspect import getmembers, isfunction, signature, Parameter

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "QSAdapter",
]


def attach_qs_methods(cls: tp.Type[tp.T], replace_signature: bool = True) -> tp.Type[tp.T]:
    """Class decorator to attach quantstats methods."""
    import quantstats as qs

    checks.assert_subclass_of(cls, "QSAdapter")

    for module_name in ["utils", "stats", "plots", "reports"]:
        for qs_func_name, qs_func in getmembers(getattr(qs, module_name), isfunction):
            if not qs_func_name.startswith("_") and checks.func_accepts_arg(qs_func, "returns"):
                if module_name == "plots":
                    new_method_name = "plot_" + qs_func_name
                elif module_name == "reports":
                    new_method_name = qs_func_name + "_report"
                else:
                    new_method_name = qs_func_name

                def new_method(
                    self,
                    *,
                    _func: tp.Callable = qs_func,
                    column: tp.Optional[tp.Label] = None,
                    **kwargs,
                ) -> tp.Any:
                    func_arg_names = get_func_arg_names(_func)
                    defaults = self.defaults

                    pass_kwargs = dict()
                    for arg_name in func_arg_names:
                        if arg_name not in kwargs:
                            if arg_name in defaults:
                                pass_kwargs[arg_name] = defaults[arg_name]
                            elif arg_name == "benchmark":
                                if self.returns_acc.bm_returns is not None:
                                    pass_kwargs["benchmark"] = self.returns_acc.bm_returns
                            elif arg_name == "periods":
                                pass_kwargs["periods"] = int(self.returns_acc.ann_factor)
                            elif arg_name == "periods_per_year":
                                pass_kwargs["periods_per_year"] = int(self.returns_acc.ann_factor)
                        else:
                            pass_kwargs[arg_name] = kwargs[arg_name]

                    returns = self.returns_acc.select_col_from_obj(
                        self.returns_acc.obj,
                        column=column,
                        wrapper=self.returns_acc.wrapper.regroup(False),
                    )
                    if returns.name is None:
                        returns = returns.rename("Strategy")
                    else:
                        returns = returns.rename(str(returns.name))
                    null_mask = returns.isnull()
                    if "benchmark" in pass_kwargs:
                        benchmark = pass_kwargs["benchmark"]
                        benchmark = self.returns_acc.select_col_from_obj(
                            benchmark,
                            column=column,
                            wrapper=self.returns_acc.wrapper.regroup(False),
                        )
                        if benchmark.name is None:
                            benchmark = benchmark.rename("Benchmark")
                        else:
                            benchmark = benchmark.rename(str(benchmark.name))
                        bm_null_mask = benchmark.isnull()
                        null_mask = null_mask | bm_null_mask
                        benchmark = benchmark.loc[~null_mask]
                        if isinstance(benchmark.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                            benchmark = benchmark.tz_localize(None)
                        pass_kwargs["benchmark"] = benchmark
                    returns = returns.loc[~null_mask]
                    if isinstance(returns.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                        returns = returns.tz_localize(None)

                    signature(_func).bind(returns=returns, **pass_kwargs)
                    return _func(returns=returns, **pass_kwargs)

                if replace_signature:
                    # Replace the function's signature with the original one
                    source_sig = signature(qs_func)
                    new_method_params = tuple(signature(new_method).parameters.values())
                    self_arg = new_method_params[0]
                    column_arg = new_method_params[2]
                    other_args = [
                        p.replace(kind=Parameter.KEYWORD_ONLY)
                        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                        else p
                        for p in list(source_sig.parameters.values())[1:]
                    ]
                    source_sig = source_sig.replace(parameters=(self_arg, column_arg) + tuple(other_args))
                    new_method.__signature__ = source_sig

                new_method.__doc__ = f"See `quantstats.{module_name}.{qs_func_name}`."
                new_method.__qualname__ = f"{cls.__name__}.{new_method_name}"
                new_method.__name__ = new_method_name
                setattr(cls, new_method_name, new_method)
    return cls


QSAdapterT = tp.TypeVar("QSAdapterT", bound="QSAdapter")


@attach_qs_methods
class QSAdapter(Configured):
    """Adapter class for quantstats."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "returns_acc",
        "defaults",
    }

    def __init__(self, returns_acc: ReturnsAccessor, defaults: tp.KwargsLike = None, **kwargs) -> None:
        checks.assert_instance_of(returns_acc, ReturnsAccessor)

        self._returns_acc = returns_acc
        self._defaults = defaults

        Configured.__init__(self, returns_acc=returns_acc, defaults=defaults, **kwargs)

    def __call__(self: QSAdapterT, **kwargs) -> QSAdapterT:
        """Allows passing arguments to the initializer."""

        return self.replace(**kwargs)

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """Returns accessor."""
        return self._returns_acc

    @property
    def defaults_mapping(self) -> tp.Dict:
        """Common argument names in quantstats mapped to `ReturnsAccessor.defaults`."""
        return dict(rf="risk_free", rolling_period="window")

    @property
    def defaults(self) -> tp.Kwargs:
        """Defaults for `QSAdapter`.

        Merges `defaults` from `vectorbtpro._settings.qs_adapter`, `returns_acc.defaults`
        (with adapted naming), and `defaults` from `QSAdapter.__init__`."""
        from vectorbtpro._settings import settings

        qs_adapter_defaults_cfg = settings["qs_adapter"]["defaults"]

        mapped_defaults = dict()
        for k, v in self.defaults_mapping.items():
            if v in self.returns_acc.defaults:
                mapped_defaults[k] = self.returns_acc.defaults[v]
        return merge_dicts(qs_adapter_defaults_cfg, mapped_defaults, self._defaults)
