# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom signal generators built with the signal factory."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_col_param_config, flex_elem_param_config
from vectorbtpro.signals.enums import StopType
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import (
    rand_enex_apply_nb,
    rand_by_prob_place_nb,
    stop_place_nb,
    ohlc_stop_place_nb,
    rand_place_nb,
)
from vectorbtpro.utils.config import ReadonlyConfig, merge_dicts

__all__ = [
    "RAND",
    "RANDX",
    "RANDNX",
    "RPROB",
    "RPROBX",
    "RPROBCX",
    "RPROBNX",
    "STX",
    "STCX",
    "OHLCSTX",
    "OHLCSTCX",
]

# ############# RAND ############# #

RAND = SignalFactory(
    class_name="RAND",
    module_name=__name__,
    short_name="rand",
    mode="entries",
    param_names=["n"],
).with_place_func(
    entry_place_func_nb=rand_place_nb,
    entry_settings=dict(pass_params=["n"]),
    param_settings=dict(n=flex_col_param_config),
    seed=None,
)


class _RAND(RAND):
    """Random entry signal generator based on the number of signals.

    Generates `entries` based on `vectorbtpro.signals.nb.rand_place_nb`.

    !!! hint
        Parameter `n` can be either a single value (per frame) or a NumPy array (per column).
        To generate multiple combinations, pass it as a list.

    Usage:
        Test three different entry counts values:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rand = vbt.RAND.run(input_shape=(6,), n=[1, 2, 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3
        0        True   True   True
        1       False  False   True
        2       False  False  False
        3       False   True  False
        4       False  False   True
        5       False  False  False
        ```

        Entry count can also be set per column:

        ```pycon
        >>> import numpy as np

        >>> rand = vbt.RAND.run(input_shape=(8, 2), n=[np.array([1, 2]), 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3      3
                    0      1      0      1
        0       False  False   True  False
        1        True  False  False  False
        2       False  False  False   True
        3       False   True   True  False
        4       False  False  False  False
        5       False  False  False   True
        6       False  False   True  False
        7       False   True  False   True
        ```
    """

    pass


setattr(RAND, "__doc__", _RAND.__doc__)

RANDX = SignalFactory(class_name="RANDX", module_name=__name__, short_name="randx", mode="exits").with_place_func(
    exit_place_func_nb=rand_place_nb,
    exit_settings=dict(pass_kwargs=dict(n=np.array([1]))),
    seed=None,
)


class _RANDX(RANDX):
    """Random exit signal generator based on the number of signals.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.rand_place_nb`.

    See `RAND` for notes on parameters.

    Usage:
        Generate an exit for each entry:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd

        >>> entries = pd.Series([True, False, False, True, False, False])
        >>> randx = vbt.RANDX.run(entries, seed=42)

        >>> randx.exits
        0    False
        1    False
        2     True
        3    False
        4     True
        5    False
        dtype: bool
        ```
    """

    pass


setattr(RANDX, "__doc__", _RANDX.__doc__)

RANDNX = SignalFactory(
    class_name="RANDNX",
    module_name=__name__,
    short_name="randnx",
    mode="both",
    param_names=["n"],
).with_apply_func(  # apply_func since function is (almost) vectorized
    rand_enex_apply_nb,
    require_input_shape=True,
    param_settings=dict(n=flex_col_param_config),
    kwargs_as_args=["entry_wait", "exit_wait"],
    entry_wait=1,
    exit_wait=1,
    seed=None,
)


class _RANDNX(RANDNX):
    """Random entry and exit signal generator based on the number of signals.

    Generates `entries` and `exits` based on `vectorbtpro.signals.nb.rand_enex_apply_nb`.

    See `RAND` for notes on parameters.

    Usage:
        Test three different entry and exit counts:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> randnx = vbt.RANDNX.run(
        ...     input_shape=(6,),
        ...     n=[1, 2, 3],
        ...     seed=42)

        >>> randnx.entries
        randnx_n      1      2      3
        0          True   True   True
        1         False  False  False
        2         False   True   True
        3         False  False  False
        4         False  False   True
        5         False  False  False

        >>> randnx.exits
        randnx_n      1      2      3
        0         False  False  False
        1          True   True   True
        2         False  False  False
        3         False   True   True
        4         False  False  False
        5         False  False   True
        ```
    """

    pass


setattr(RANDNX, "__doc__", _RANDNX.__doc__)

# ############# RPROB ############# #

RPROB = SignalFactory(
    class_name="RPROB",
    module_name=__name__,
    short_name="rprob",
    mode="entries",
    param_names=["prob"],
).with_place_func(
    entry_place_func_nb=rand_by_prob_place_nb,
    entry_settings=dict(pass_params=["prob"], pass_kwargs=["pick_first"]),
    param_settings=dict(prob=flex_elem_param_config),
    seed=None,
)


class _RPROB(RPROB):
    """Random entry signal generator based on probabilities.

    Generates `entries` based on `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    Usage:
        Generate three columns with different entry probabilities:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=[0., 0.5, 1.], seed=42)

        >>> rprob.entries
        rprob_prob    0.0    0.5   1.0
        0           False   True  True
        1           False   True  True
        2           False  False  True
        3           False  False  True
        4           False  False  True
        ```

        Probability can also be set per row, column, or element:

        ```pycon
        >>> import numpy as np

        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=np.array([0., 0., 1., 1., 1.]), seed=42)

        >>> rprob.entries
        0    False
        1    False
        2     True
        3     True
        4     True
        Name: array_0, dtype: bool
        ```
    """

    pass


setattr(RPROB, "__doc__", _RPROB.__doc__)

rprobx_config = ReadonlyConfig(
    dict(class_name="RPROBX", module_name=__name__, short_name="rprobx", mode="exits", param_names=["prob"]),
)
"""Factory config for `RPROBX`."""

rprobx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=rand_by_prob_place_nb,
        exit_settings=dict(pass_params=["prob"], pass_kwargs=["pick_first"]),
        param_settings=dict(prob=flex_elem_param_config),
        seed=None,
    )
)
"""Exit function config for `RPROBX`."""

RPROBX = SignalFactory(**rprobx_config).with_place_func(**rprobx_func_config)


class _RPROBX(RPROBX):
    """Random exit signal generator based on probabilities.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBX, "__doc__", _RPROBX.__doc__)

RPROBCX = SignalFactory(
    **rprobx_config.merge_with(dict(class_name="RPROBCX", short_name="rprobcx", mode="chain")),
).with_place_func(**rprobx_func_config)


class _RPROBCX(RPROBCX):
    """Random exit signal generator based on probabilities.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBCX, "__doc__", _RPROBCX.__doc__)

RPROBNX = SignalFactory(
    class_name="RPROBNX",
    module_name=__name__,
    short_name="rprobnx",
    mode="both",
    param_names=["entry_prob", "exit_prob"],
).with_place_func(
    entry_place_func_nb=rand_by_prob_place_nb,
    entry_settings=dict(pass_params=["entry_prob"], pass_kwargs=["pick_first"]),
    exit_place_func_nb=rand_by_prob_place_nb,
    exit_settings=dict(pass_params=["exit_prob"], pass_kwargs=["pick_first"]),
    param_settings=dict(entry_prob=flex_elem_param_config, exit_prob=flex_elem_param_config),
    seed=None,
)


class _RPROBNX(RPROBNX):
    """Random entry and exit signal generator based on probabilities.

    Generates `entries` and `exits` based on `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters.

    Usage:
        Test all probability combinations:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> rprobnx = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[0.5, 1.],
        ...     exit_prob=[0.5, 1.],
        ...     param_product=True,
        ...     seed=42)

        >>> rprobnx.entries
        rprobnx_entry_prob    0.5    0.5    1.0    0.5
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                    True   True   True   True
        1                   False  False  False  False
        2                   False  False  False   True
        3                   False  False  False  False
        4                   False  False   True   True

        >>> rprobnx.exits
        rprobnx_entry_prob    0.5    0.5    1.0    1.0
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                   False  False  False  False
        1                   False   True  False   True
        2                   False  False  False  False
        3                   False  False   True   True
        4                    True  False  False  False
        ```

        Probabilities can also be set per row, column, or element:

        ```pycon
        >>> import numpy as np

        >>> entry_prob1 = np.array([1., 0., 1., 0., 1.])
        >>> entry_prob2 = np.array([0., 1., 0., 1., 0.])
        >>> rprobnx = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[entry_prob1, entry_prob2],
        ...     exit_prob=1.,
        ...     seed=42)

        >>> rprobnx.entries
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                     True   False
        1                    False    True
        2                     True   False
        3                    False    True
        4                     True   False

        >>> rprobnx.exits
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                    False   False
        1                     True   False
        2                    False    True
        3                     True   False
        4                    False    True
        ```
    """

    pass


setattr(RPROBNX, "__doc__", _RPROBNX.__doc__)

# ############# ST ############# #

stx_config = ReadonlyConfig(
    dict(
        class_name="STX",
        module_name=__name__,
        short_name="stx",
        mode="exits",
        input_names=["entry_ts", "ts", "follow_ts"],
        in_output_names=["stop_ts"],
        param_names=["stop", "trailing"],
    )
)
"""Factory config for `STX`."""

stx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_ts", "ts", "follow_ts"],
            pass_in_outputs=["stop_ts"],
            pass_params=["stop", "trailing"],
        ),
        param_settings=dict(stop=flex_elem_param_config, trailing=flex_elem_param_config),
        trailing=False,
        ts=np.nan,
        follow_ts=np.nan,
        stop_ts=np.nan,
    )
)
"""Exit function config for `STX`."""

STX = SignalFactory(**stx_config).with_place_func(**stx_func_config)


class _STX(STX):
    """Exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.stop_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists."""

    pass


setattr(STX, "__doc__", _STX.__doc__)

STCX = SignalFactory(**stx_config.merge_with(dict(class_name="STCX", short_name="stcx", mode="chain"))).with_place_func(
    **stx_func_config
)


class _STCX(STCX):
    """Exit signal generator based on stop values.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.stop_place_nb`.

    See `STX` for notes on parameters."""

    pass


setattr(STCX, "__doc__", _STCX.__doc__)

# ############# OHLCST ############# #

ohlcstx_config = ReadonlyConfig(
    dict(
        class_name="OHLCSTX",
        module_name=__name__,
        short_name="ohlcstx",
        mode="exits",
        input_names=["entry_price", "open", "high", "low", "close"],
        in_output_names=["stop_price", "stop_type"],
        param_names=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
        attr_settings=dict(stop_type=dict(dtype=StopType)),  # creates rand_type_readable
    )
)
"""Factory config for `OHLCSTX`."""

ohlcstx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=ohlc_stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_price", "open", "high", "low", "close"],  # do not pass entries
            pass_in_outputs=["stop_price", "stop_type"],
            pass_params=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
            pass_kwargs=["is_entry_open"],
        ),
        in_output_settings=dict(stop_price=dict(dtype=np.float_), stop_type=dict(dtype=np.int_)),
        param_settings=dict(
            sl_stop=flex_elem_param_config,
            tsl_th=flex_elem_param_config,
            tsl_stop=flex_elem_param_config,
            tp_stop=flex_elem_param_config,
            reverse=flex_elem_param_config,
        ),
        open=np.nan,
        high=np.nan,
        low=np.nan,
        close=np.nan,
        stop_price=np.nan,
        stop_type=-1,
        sl_stop=np.nan,
        tsl_th=np.nan,
        tsl_stop=np.nan,
        tp_stop=np.nan,
        reverse=False,
        is_entry_open=False,
    )
)
"""Exit function config for `OHLCSTX`."""

OHLCSTX = SignalFactory(**ohlcstx_config).with_place_func(**ohlcstx_func_config)


def _bind_ohlcstx_plot(base_cls: type, entries_attr: str) -> tp.Callable:

    base_cls_plot = base_cls.plot

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        ohlc_kwargs: tp.KwargsLike = None,
        entry_price_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        _base_cls_plot: tp.Callable = base_cls_plot,
        **layout_kwargs
    ) -> tp.BaseFigure:
        self_col = self.select_col(column=column, group_by=False)

        if ohlc_kwargs is None:
            ohlc_kwargs = {}
        if entry_price_kwargs is None:
            entry_price_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        open_any = not self_col.open.isnull().all()
        high_any = not self_col.high.isnull().all()
        low_any = not self_col.low.isnull().all()
        close_any = not self_col.close.isnull().all()
        if open_any and high_any and low_any and close_any:
            ohlc_df = pd.concat((
                self_col.open,
                self_col.high,
                self_col.low,
                self_col.close
            ), axis=1)
            ohlc_df.columns = ["Open", "High", "Low", "Close"]
            ohlc_kwargs = merge_dicts(layout_kwargs, dict(ohlc_trace_kwargs=dict(opacity=0.5)), ohlc_kwargs)
            fig = ohlc_df.vbt.ohlcv.plot(fig=fig, **ohlc_kwargs)
        else:
            entry_price_kwargs = merge_dicts(layout_kwargs, entry_price_kwargs)
            fig = self_col.entry_price.rename("Entry price").vbt.lineplot(fig=fig, **entry_price_kwargs)

        # Plot entry and exit markers
        _base_cls_plot(
            self_col,
            entry_y=self_col.entry_price,
            exit_y=self_col.stop_price,
            exit_types=self_col.stop_type_readable,
            entry_trace_kwargs=entry_trace_kwargs,
            exit_trace_kwargs=exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )
        return fig

    plot.__doc__ = """Plot OHLC, `{0}.{1}` and `{0}.exits`.
    
    Args:
        ohlc_kwargs (dict): Keyword arguments passed to 
            `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot`.
        entry_trace_kwargs (dict): Keyword arguments passed to 
            `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entries` for `{0}.{1}`.
        exit_trace_kwargs (dict): Keyword arguments passed to 
            `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exits` for `{0}.exits`.
        fig (Figure or FigureWidget): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.""".format(
        base_cls.__name__,
        entries_attr,
    )

    if entries_attr == "entries":
        plot.__doc__ += """
    Usage:
        ```pycon
        >>> ohlcstx.iloc[:, 0].plot().show()
        ```
        
        ![](/assets/images/api/OHLCSTX.svg){: .iimg loading=lazy }
    """
    return plot


class _OHLCSTX(OHLCSTX):
    """Exit signal generator based on OHLC and stop values.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    !!! warning
        Searches for an exit after each entry. If two entries come one after another, no exit can be placed.
        Consider either cleaning up entry signals prior to passing, or using `OHLCSTCX`.

    Usage:
        Test each stop type:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd
        >>> import numpy as np

        >>> entries = pd.Series([True, False, False, False, False, False])
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10, 9],
        ...     'high': [11, 12, 13, 12, 11, 10],
        ...     'low': [9, 10, 11, 10, 9, 8],
        ...     'close': [10, 11, 12, 11, 10, 9]
        ... })
        >>> ohlcstx = vbt.OHLCSTX.run(
        ...     entries,
        ...     price['open'],
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=[0.1, np.nan, np.nan, np.nan],
        ...     tsl_th=[np.nan, np.nan, 0.2, np.nan],
        ...     tsl_stop=[np.nan, 0.1, 0.3, np.nan],
        ...     tp_stop=[np.nan, np.nan, np.nan, 0.1],
        ...     is_entry_open=True)

        >>> ohlcstx.entries
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                   True   True   True   True
        1                  False  False  False  False
        2                  False  False  False  False
        3                  False  False  False  False
        4                  False  False  False  False
        5                  False  False  False  False

        >>> ohlcstx.exits
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                  False  False  False  False
        1                  False  False  False   True
        2                  False  False  False  False
        3                  False   True  False  False
        4                   True  False   True  False
        5                  False  False  False  False

        >>> ohlcstx.stop_price
        ohlcstx_sl_stop    0.1   NaN  NaN   NaN
        ohlcstx_tsl_th     NaN   NaN  0.2   NaN
        ohlcstx_tsl_stop   NaN   0.1  0.3   NaN
        ohlcstx_tp_stop    NaN   NaN  NaN   0.1
        0                  NaN   NaN  NaN   NaN
        1                  NaN   NaN  NaN  11.0
        2                  NaN   NaN  NaN   NaN
        3                  NaN  11.7  NaN   NaN
        4                  9.0   NaN  9.1   NaN
        5                  NaN   NaN  NaN   NaN

        >>> ohlcstx.stop_type_readable
        ohlcstx_sl_stop     0.1   NaN   NaN   NaN
        ohlcstx_tsl_th      NaN   NaN   0.2   NaN
        ohlcstx_tsl_stop    NaN   0.1   0.3   NaN
        ohlcstx_tp_stop     NaN   NaN   NaN   0.1
        0                  None  None  None  None
        1                  None  None  None    TP
        2                  None  None  None  None
        3                  None   TSL  None  None
        4                    SL  None   TTP  None
        5                  None  None  None  None
        ```
    """

    plot = _bind_ohlcstx_plot(OHLCSTX, "entries")


setattr(OHLCSTX, "__doc__", _OHLCSTX.__doc__)
setattr(OHLCSTX, "plot", _OHLCSTX.plot)

OHLCSTCX = SignalFactory(
    **ohlcstx_config.merge_with(dict(class_name="OHLCSTCX", short_name="ohlcstcx", mode="chain")),
).with_place_func(**ohlcstx_func_config)


class _OHLCSTCX(OHLCSTCX):
    """Exit signal generator based on OHLC and stop values.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

    See `OHLCSTX` for notes on parameters."""

    plot = _bind_ohlcstx_plot(OHLCSTCX, "new_entries")


setattr(OHLCSTCX, "__doc__", _OHLCSTCX.__doc__)
setattr(OHLCSTCX, "plot", _OHLCSTCX.plot)
