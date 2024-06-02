# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Factory for building signal generators.

The signal factory class `SignalFactory` extends `vectorbtpro.indicators.factory.IndicatorFactory`
to offer a convenient way to create signal generators of any complexity. By providing it with information
such as entry and exit functions and the names of inputs, parameters, and outputs, it will create a
stand-alone class capable of generating signals for an arbitrary combination of inputs and parameters.
"""

import inspect

import numpy as np
from numba import njit

from vectorbtpro import _typing as tp
from vectorbtpro.base import combining
from vectorbtpro.indicators.factory import IndicatorFactory, IndicatorBase, CacheOutputT
from vectorbtpro.signals.enums import FactoryMode
from vectorbtpro.signals.nb import generate_nb, generate_ex_nb, generate_enex_nb, first_place_nb
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.params import to_typed_list

__all__ = [
    "SignalFactory",
]


class SignalFactory(IndicatorFactory):
    """A factory for building signal generators.

    Extends `vectorbtpro.indicators.factory.IndicatorFactory` with place functions.

    Generates a fixed number of outputs (depending upon `mode`).
    If you need to generate other outputs, use in-place outputs (via `in_output_names`).

    See `vectorbtpro.signals.enums.FactoryMode` for supported generation modes.

    Other arguments are passed to `vectorbtpro.indicators.factory.IndicatorFactory`.
    """

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (IndicatorFactory._expected_keys or set()) | {
        "mode",
    }

    def __init__(
        self,
        *args,
        mode: tp.Union[str, int] = FactoryMode.Both,
        input_names: tp.Optional[tp.Sequence[str]] = None,
        attr_settings: tp.KwargsLike = None,
        **kwargs
    ) -> None:
        mode = map_enum_fields(mode, FactoryMode)
        if input_names is None:
            input_names = []
        else:
            input_names = list(input_names)
        if attr_settings is None:
            attr_settings = {}

        if "entries" in input_names:
            raise ValueError("entries cannot be used in input_names")
        if "exits" in input_names:
            raise ValueError("exits cannot be used in input_names")
        if mode == FactoryMode.Entries:
            output_names = ["entries"]
        elif mode == FactoryMode.Exits:
            input_names = ["entries"] + input_names
            output_names = ["exits"]
        elif mode == FactoryMode.Both:
            output_names = ["entries", "exits"]
        else:
            input_names = ["entries"] + input_names
            output_names = ["new_entries", "exits"]
        if "entries" in input_names:
            attr_settings["entries"] = dict(dtype=np.bool_)
        for output_name in output_names:
            attr_settings[output_name] = dict(dtype=np.bool_)

        IndicatorFactory.__init__(
            self,
            *args,
            mode=mode,
            input_names=input_names,
            output_names=output_names,
            attr_settings=attr_settings,
            **kwargs,
        )
        self._mode = mode

        def plot(
            _self,
            column: tp.Optional[tp.Label] = None,
            entry_y: tp.Union[None, str, tp.ArrayLike] = None,
            exit_y: tp.Union[None, str, tp.ArrayLike] = None,
            entry_types: tp.Optional[tp.ArrayLike] = None,
            exit_types: tp.Optional[tp.ArrayLike] = None,
            entry_trace_kwargs: tp.KwargsLike = None,
            exit_trace_kwargs: tp.KwargsLike = None,
            fig: tp.Optional[tp.BaseFigure] = None,
            **kwargs
        ) -> tp.BaseFigure:
            self_col = _self.select_col(column=column, group_by=False)
            if entry_y is not None and isinstance(entry_y, str):
                entry_y = getattr(self_col, entry_y)
            if exit_y is not None and isinstance(exit_y, str):
                exit_y = getattr(self_col, exit_y)

            if entry_trace_kwargs is None:
                entry_trace_kwargs = {}
            if exit_trace_kwargs is None:
                exit_trace_kwargs = {}
            entry_trace_kwargs = merge_dicts(
                dict(name="New Entries" if mode == FactoryMode.Chain else "Entries"),
                entry_trace_kwargs,
            )
            exit_trace_kwargs = merge_dicts(dict(name="Exits"), exit_trace_kwargs)
            if entry_types is not None:
                entry_types = np.asarray(entry_types)
                entry_trace_kwargs = merge_dicts(
                    dict(customdata=entry_types, hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"),
                    entry_trace_kwargs,
                )
            if exit_types is not None:
                exit_types = np.asarray(exit_types)
                exit_trace_kwargs = merge_dicts(
                    dict(customdata=exit_types, hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"),
                    exit_trace_kwargs,
                )
            if mode == FactoryMode.Entries:
                fig = self_col.entries.vbt.signals.plot_as_entries(
                    y=entry_y,
                    trace_kwargs=entry_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
            elif mode == FactoryMode.Exits:
                fig = self_col.entries.vbt.signals.plot_as_entries(
                    y=entry_y,
                    trace_kwargs=entry_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
                fig = self_col.exits.vbt.signals.plot_as_exits(
                    y=exit_y,
                    trace_kwargs=exit_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
            elif mode == FactoryMode.Both:
                fig = self_col.entries.vbt.signals.plot_as_entries(
                    y=entry_y,
                    trace_kwargs=entry_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
                fig = self_col.exits.vbt.signals.plot_as_exits(
                    y=exit_y,
                    trace_kwargs=exit_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
            else:
                fig = self_col.new_entries.vbt.signals.plot_as_entries(
                    y=entry_y,
                    trace_kwargs=entry_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )
                fig = self_col.exits.vbt.signals.plot_as_exits(
                    y=exit_y,
                    trace_kwargs=exit_trace_kwargs,
                    fig=fig,
                    **kwargs,
                )

            return fig

        plot.__doc__ = """Plot `{0}.{1}` and `{0}.exits`.

        Args:
            entry_y (array_like): Y-axis values to plot entry markers on.
            exit_y (array_like): Y-axis values to plot exit markers on.
            entry_types (array_like): Entry types in string format.
            exit_types (array_like): Exit types in string format.
            entry_trace_kwargs (dict): Keyword arguments passed to
                `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entries` for `{0}.{1}`.
            exit_trace_kwargs (dict): Keyword arguments passed to 
                `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exits` for `{0}.exits`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **kwargs: Keyword arguments passed to `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_markers`.
        """.format(
            self.class_name,
            "new_entries" if mode == FactoryMode.Chain else "entries",
        )

        setattr(self.Indicator, "plot", plot)

    @property
    def mode(self):
        """Factory mode."""
        return self._mode

    def with_place_func(
        self,
        entry_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
        exit_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
        generate_func_nb: tp.Callable = generate_nb,
        generate_ex_func_nb: tp.Callable = generate_ex_nb,
        generate_enex_func_nb: tp.Callable = generate_enex_nb,
        cache_func: tp.Callable = None,
        entry_settings: tp.KwargsLike = None,
        exit_settings: tp.KwargsLike = None,
        cache_settings: tp.KwargsLike = None,
        jit_kwargs: tp.KwargsLike = None,
        **kwargs
    ) -> tp.Type[IndicatorBase]:
        """Build signal generator class around entry and exit placement functions.

        A placement function is simply a function that places signals.
        There are two types of it: entry placement function and exit placement function.
        Each placement function takes broadcast time series, broadcast in-place output time series,
        broadcast parameter arrays, and other arguments, and returns an array of indices
        corresponding to chosen signals. See `vectorbtpro.signals.nb.generate_nb`.

        Args:
            entry_place_func_nb (callable): `place_func_nb` that returns indices of entries.

                Defaults to `vectorbtpro.signals.nb.first_place_nb` for `FactoryMode.Chain`.
            exit_place_func_nb (callable): `place_func_nb` that returns indices of exits.
            generate_func_nb (callable): Entry generation function.

                Defaults to `vectorbtpro.signals.nb.generate_nb`.
            generate_ex_func_nb (callable): Exit generation function.

                Defaults to `vectorbtpro.signals.nb.generate_ex_nb`.
            generate_enex_func_nb (callable): Entry and exit generation function.

                Defaults to `vectorbtpro.signals.nb.generate_enex_nb`.
            cache_func (callable): A caching function to preprocess data beforehand.

                All returned objects will be passed as last arguments to placement functions.
            entry_settings (dict): Settings dict for `entry_place_func_nb`.
            exit_settings (dict): Settings dict for `exit_place_func_nb`.
            cache_settings (dict): Settings dict for `cache_func`.
            jit_kwargs (dict): Keyword arguments passed to `@njit` decorator of the parameter selection function.

                By default, has `nogil` set to True.
            **kwargs: Keyword arguments passed to `IndicatorFactory.with_custom_func`.

        !!! note
            Choice functions must be Numba-compiled.

            Which inputs, parameters and arguments to pass to each function must be
            explicitly indicated in the function's settings dict. By default, nothing is passed.

            Passing keyword arguments directly to the placement functions is not supported.
            Use `pass_kwargs` in a settings dict to pass keyword arguments as positional.

        Settings dict of each function can have the following keys:

        Attributes:
            pass_inputs (list of str): Input names to pass to the placement function.

                Defaults to []. Order matters. Each name must be in `input_names`.
            pass_in_outputs (list of str): In-place output names to pass to the placement function.

                Defaults to []. Order matters. Each name must be in `in_output_names`.
            pass_params (list of str): Parameter names to pass to the placement function.

                Defaults to []. Order matters. Each name must be in `param_names`.
            pass_kwargs (dict, list of str or list of tuple): Keyword arguments from `kwargs` dict to
                pass as positional arguments to the placement function.

                Defaults to []. Order matters.

                If any element is a tuple, must contain the name and the default value.
                If any element is a string, the default value is None.

                Built-in keys include:

                * `input_shape`: Input shape if no input time series passed.
                    Default is provided by the pipeline if `pass_input_shape` is True.
                * `wait`: Number of ticks to wait before placing signals.
                    Default is 1.
                * `until_next`: Whether to place signals up to the next entry signal.
                    Default is True. Applied in `generate_ex_func_nb` only.
                * `skip_until_exit`: Whether to skip processing entry signals until the next exit.
                    Default is False. Applied in `generate_ex_func_nb` only.
                * `pick_first`: Whether to stop as soon as the first exit signal is found.
                    Default is False with `FactoryMode.Entries`, otherwise is True.
                * `temp_idx_arr`: Empty integer array used to temporarily store indices.
                    Default is an automatically generated array of shape `input_shape[0]`.
                    You can also pass `temp_idx_arr1`, `temp_idx_arr2`, etc. to generate multiple.
            pass_cache (bool): Whether to pass cache from `cache_func` to the placement function.

                Defaults to False. Cache is passed unpacked.

        The following arguments can be passed to `run` and `run_combs` methods:

        Args:
            *args: Must be used instead of `entry_args` with `FactoryMode.Entries` and instead of
                `exit_args` with `FactoryMode.Exits` and `FactoryMode.Chain` with default `entry_place_func_nb`.
            entry_args (tuple): Arguments passed to the entry placement function.
            exit_args (tuple): Arguments passed to the exit placement function.
            cache_args (tuple): Arguments passed to the cache function.
            entry_kwargs (tuple): Settings for the entry placement function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            exit_kwargs (tuple): Settings for the exit placement function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            cache_kwargs (tuple): Settings for the cache function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            return_cache (bool): Whether to return only cache.
            use_cache (any): Cache to use.
            **kwargs: Must be used instead of `entry_kwargs` with `FactoryMode.Entries` and instead of
                `exit_kwargs` with `FactoryMode.Exits` and `FactoryMode.Chain` with default `entry_place_func_nb`.

        For more arguments, see `vectorbtpro.indicators.factory.IndicatorBase.run_pipeline`.

        Usage:
            * The simplest signal indicator that places True at the very first index:

            ```pycon
            >>> from numba import njit
            >>> import vectorbtpro as vbt
            >>> import numpy as np

            >>> @njit
            ... def entry_place_func_nb(c):
            ...     c.out[0] = True
            ...     return 0

            >>> @njit
            ... def exit_place_func_nb(c):
            ...     c.out[0] = True
            ...     return 0

            >>> MySignals = vbt.SignalFactory().with_place_func(
            ...     entry_place_func_nb=entry_place_func_nb,
            ...     exit_place_func_nb=exit_place_func_nb,
            ...     entry_kwargs=dict(wait=1),
            ...     exit_kwargs=dict(wait=1)
            ... )

            >>> my_sig = MySignals.run(input_shape=(3, 3))
            >>> my_sig.entries
                   0      1      2
            0   True   True   True
            1  False  False  False
            2   True   True   True
            >>> my_sig.exits
                   0      1      2
            0  False  False  False
            1   True   True   True
            2  False  False  False
            ```

            * Take the first entry and place an exit after waiting `n` ticks. Find the next entry and repeat.
            Test three different `n` values.

            ```pycon
            >>> from numba import njit
            >>> from vectorbtpro.signals.factory import SignalFactory

            >>> @njit
            ... def wait_place_nb(c, n):
            ...     if n < len(c.out):
            ...         c.out[n] = True
            ...         return n
            ...     return -1

            >>> # Build signal generator
            >>> MySignals = SignalFactory(
            ...     mode='chain',
            ...     param_names=['n']
            ... ).with_place_func(
            ...     exit_place_func_nb=wait_place_nb,
            ...     exit_settings=dict(
            ...         pass_params=['n']
            ...     )
            ... )

            >>> # Run signal generator
            >>> entries = [True, True, True, True, True]
            >>> my_sig = MySignals.run(entries, [0, 1, 2])

            >>> my_sig.entries  # input entries
            custom_n     0     1     2
            0         True  True  True
            1         True  True  True
            2         True  True  True
            3         True  True  True
            4         True  True  True

            >>> my_sig.new_entries  # output entries
            custom_n      0      1      2
            0          True   True   True
            1         False  False  False
            2          True  False  False
            3         False   True  False
            4          True  False   True

            >>> my_sig.exits  # output exits
            custom_n      0      1      2
            0         False  False  False
            1          True  False  False
            2         False   True  False
            3          True  False   True
            4         False  False  False
            ```

            * To combine multiple iterative signals, you would need to create a custom placement function.
            Here is an example of combining two random generators using "OR" rule (the first signal wins):

            ```pycon
            >>> import numpy as np
            >>> from numba import njit
            >>> from collections import namedtuple
            >>> from vectorbtpro.indicators.configs import flex_elem_param_config
            >>> from vectorbtpro.signals.factory import SignalFactory
            >>> from vectorbtpro.signals.nb import rand_by_prob_place_nb

            >>> # Enum to distinguish random generators
            >>> RandType = namedtuple('RandType', ['R1', 'R2'])(0, 1)

            >>> # Define exit placement function
            >>> @njit
            ... def rand_exit_place_nb(c, rand_type, prob1, prob2):
            ...     for out_i in range(len(c.out)):
            ...         if np.random.uniform(0, 1) < prob1:
            ...             c.out[out_i] = True
            ...             rand_type[c.from_i + out_i] = RandType.R1
            ...             return out_i
            ...         if np.random.uniform(0, 1) < prob2:
            ...             c.out[out_i] = True
            ...             rand_type[c.from_i + out_i] = RandType.R2
            ...             return out_i
            ...     return -1

            >>> # Build signal generator
            >>> MySignals = SignalFactory(
            ...     mode='chain',
            ...     in_output_names=['rand_type'],
            ...     param_names=['prob1', 'prob2'],
            ...     attr_settings=dict(
            ...         rand_type=dict(dtype=RandType)  # creates rand_type_readable
            ...     )
            ... ).with_place_func(
            ...     exit_place_func_nb=rand_exit_place_nb,
            ...     exit_settings=dict(
            ...         pass_in_outputs=['rand_type'],
            ...         pass_params=['prob1', 'prob2']
            ...     ),
            ...     param_settings=dict(
            ...         prob1=flex_elem_param_config,  # param per frame/row/col/element
            ...         prob2=flex_elem_param_config
            ...     ),
            ...     rand_type=-1  # fill with this value
            ... )

            >>> # Run signal generator
            >>> entries = [True, True, True, True, True]
            >>> my_sig = MySignals.run(entries, [0., 1.], [0., 1.], param_product=True)

            >>> my_sig.new_entries
            custom_prob1           0.0           1.0
            custom_prob2    0.0    1.0    0.0    1.0
            0              True   True   True   True
            1             False  False  False  False
            2             False   True   True   True
            3             False  False  False  False
            4             False   True   True   True

            >>> my_sig.exits
            custom_prob1           0.0           1.0
            custom_prob2    0.0    1.0    0.0    1.0
            0             False  False  False  False
            1             False   True   True   True
            2             False  False  False  False
            3             False   True   True   True
            4             False  False  False  False

            >>> my_sig.rand_type_readable
            custom_prob1     0.0     1.0
            custom_prob2 0.0 1.0 0.0 1.0
            0
            1                 R2  R1  R1
            2
            3                 R2  R1  R1
            4
            ```
        """
        Indicator = self.Indicator

        setattr(Indicator, "entry_place_func_nb", entry_place_func_nb)
        setattr(Indicator, "exit_place_func_nb", exit_place_func_nb)

        module_name = self.module_name
        mode = self.mode
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names

        if mode == FactoryMode.Entries:
            require_input_shape = len(input_names) == 0
            checks.assert_not_none(entry_place_func_nb)
            if exit_place_func_nb is not None:
                raise ValueError("exit_place_func_nb cannot be used with FactoryMode.Entries")
        elif mode == FactoryMode.Exits:
            require_input_shape = False
            if entry_place_func_nb is not None:
                raise ValueError("entry_place_func_nb cannot be used with FactoryMode.Exits")
            checks.assert_not_none(exit_place_func_nb)
        elif mode == FactoryMode.Both:
            require_input_shape = len(input_names) == 0
            checks.assert_not_none(entry_place_func_nb)
            checks.assert_not_none(exit_place_func_nb)
        else:
            require_input_shape = False
            if entry_place_func_nb is None:
                entry_place_func_nb = first_place_nb
            if entry_settings is None:
                entry_settings = {}
            entry_settings = merge_dicts(dict(pass_inputs=["entries"]), entry_settings)
            checks.assert_not_none(entry_place_func_nb)
            checks.assert_not_none(exit_place_func_nb)
        require_input_shape = kwargs.pop("require_input_shape", require_input_shape)

        if entry_settings is None:
            entry_settings = {}
        if exit_settings is None:
            exit_settings = {}
        if cache_settings is None:
            cache_settings = {}

        valid_keys = ["pass_inputs", "pass_in_outputs", "pass_params", "pass_kwargs", "pass_cache"]
        checks.assert_dict_valid(entry_settings, valid_keys)
        checks.assert_dict_valid(exit_settings, valid_keys)
        checks.assert_dict_valid(cache_settings, valid_keys)

        # Get input names for each function
        def _get_func_names(func_settings: tp.Kwargs, setting: str, all_names: tp.Sequence[str]) -> tp.List[str]:
            func_input_names = func_settings.get(setting, None)
            if func_input_names is None:
                return []
            else:
                for name in func_input_names:
                    checks.assert_in(name, all_names)
            return func_input_names

        entry_input_names = _get_func_names(entry_settings, "pass_inputs", input_names)
        exit_input_names = _get_func_names(exit_settings, "pass_inputs", input_names)
        cache_input_names = _get_func_names(cache_settings, "pass_inputs", input_names)

        entry_in_output_names = _get_func_names(entry_settings, "pass_in_outputs", in_output_names)
        exit_in_output_names = _get_func_names(exit_settings, "pass_in_outputs", in_output_names)
        cache_in_output_names = _get_func_names(cache_settings, "pass_in_outputs", in_output_names)

        entry_param_names = _get_func_names(entry_settings, "pass_params", param_names)
        exit_param_names = _get_func_names(exit_settings, "pass_params", param_names)
        cache_param_names = _get_func_names(cache_settings, "pass_params", param_names)

        # Build a function that selects a parameter tuple
        if mode == FactoryMode.Entries:
            _0 = "i"
            _0 += ", shape"
            _0 += ", only_once"
            _0 += ", entry_wait"
            if len(entry_input_names) > 0:
                _0 += ", " + ", ".join(entry_input_names)
            if len(entry_in_output_names) > 0:
                _0 += ", " + ", ".join(entry_in_output_names)
            if len(entry_param_names) > 0:
                _0 += ", " + ", ".join(entry_param_names)
            _0 += ", entry_args"
            _1 = "shape"
            _1 += ", only_once"
            _1 += ", entry_wait"
            _1 += ", entry_place_func_nb"
            if len(entry_input_names) > 0:
                _1 += ", " + ", ".join(entry_input_names)
            if len(entry_in_output_names) > 0:
                _1 += ", " + ", ".join(map(lambda x: x + "[i]", entry_in_output_names))
            if len(entry_param_names) > 0:
                _1 += ", " + ", ".join(map(lambda x: x + "[i]", entry_param_names))
            _1 += ", *entry_args"
            func_str = "def apply_func({0}):\n   return generate_func_nb({1})".format(_0, _1)
            scope = {"generate_func_nb": generate_func_nb, "entry_place_func_nb": entry_place_func_nb}

        elif mode == FactoryMode.Exits:
            _0 = "i"
            _0 += ", entries"
            _0 += ", exit_wait"
            _0 += ", until_next"
            _0 += ", skip_until_exit"
            if len(exit_input_names) > 0:
                _0 += ", " + ", ".join(exit_input_names)
            if len(exit_in_output_names) > 0:
                _0 += ", " + ", ".join(exit_in_output_names)
            if len(exit_param_names) > 0:
                _0 += ", " + ", ".join(exit_param_names)
            _0 += ", exit_args"
            _1 = "entries"
            _1 += ", exit_wait"
            _1 += ", until_next"
            _1 += ", skip_until_exit"
            _1 += ", exit_place_func_nb"
            if len(exit_input_names) > 0:
                _1 += ", " + ", ".join(exit_input_names)
            if len(exit_in_output_names) > 0:
                _1 += ", " + ", ".join(map(lambda x: x + "[i]", exit_in_output_names))
            if len(exit_param_names) > 0:
                _1 += ", " + ", ".join(map(lambda x: x + "[i]", exit_param_names))
            _1 += ", *exit_args"
            func_str = "def apply_func({0}):\n   return generate_ex_func_nb({1})".format(_0, _1)
            scope = {"generate_ex_func_nb": generate_ex_func_nb, "exit_place_func_nb": exit_place_func_nb}

        else:
            _0 = "i"
            _0 += ", shape"
            _0 += ", entry_wait"
            _0 += ", exit_wait"
            if len(entry_input_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_entry_" + x, entry_input_names))
            if len(entry_in_output_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_entry_" + x, entry_in_output_names))
            if len(entry_param_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_entry_" + x, entry_param_names))
            _0 += ", entry_args"
            if len(exit_input_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_exit_" + x, exit_input_names))
            if len(exit_in_output_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_exit_" + x, exit_in_output_names))
            if len(exit_param_names) > 0:
                _0 += ", " + ", ".join(map(lambda x: "_exit_" + x, exit_param_names))
            _0 += ", exit_args"
            _1 = "shape"
            _1 += ", entry_wait"
            _1 += ", exit_wait"
            _1 += ", entry_place_func_nb"
            _1 += ", ("
            if len(entry_input_names) > 0:
                _1 += ", ".join(map(lambda x: "_entry_" + x, entry_input_names)) + ", "
            if len(entry_in_output_names) > 0:
                _1 += ", ".join(map(lambda x: "_entry_" + x + "[i]", entry_in_output_names)) + ", "
            if len(entry_param_names) > 0:
                _1 += ", ".join(map(lambda x: "_entry_" + x + "[i]", entry_param_names)) + ", "
            _1 += "*entry_args,)"
            _1 += ", exit_place_func_nb"
            _1 += ", ("
            if len(exit_input_names) > 0:
                _1 += ", ".join(map(lambda x: "_exit_" + x, exit_input_names)) + ", "
            if len(exit_in_output_names) > 0:
                _1 += ", ".join(map(lambda x: "_exit_" + x + "[i]", exit_in_output_names)) + ", "
            if len(exit_param_names) > 0:
                _1 += ", ".join(map(lambda x: "_exit_" + x + "[i]", exit_param_names)) + ", "
            _1 += "*exit_args,)"
            func_str = "def apply_func({0}):\n   return generate_enex_func_nb({1})".format(_0, _1)
            scope = {
                "generate_enex_func_nb": generate_enex_func_nb,
                "entry_place_func_nb": entry_place_func_nb,
                "exit_place_func_nb": exit_place_func_nb,
            }

        filename = inspect.getfile(lambda: None)
        code = compile(func_str, filename, "single")
        exec(code, scope)
        apply_func = scope["apply_func"]
        if module_name is not None:
            apply_func.__module__ = module_name
        jit_kwargs = merge_dicts(dict(nogil=True), jit_kwargs)
        apply_func = njit(apply_func, **jit_kwargs)

        setattr(Indicator, "apply_func", apply_func)

        def custom_func(
            input_list: tp.List[tp.AnyArray],
            in_output_list: tp.List[tp.List[tp.AnyArray]],
            param_list: tp.List[tp.List[tp.Param]],
            *args,
            input_shape: tp.Optional[tp.Shape] = None,
            entry_args: tp.Optional[tp.Args] = None,
            exit_args: tp.Optional[tp.Args] = None,
            cache_args: tp.Optional[tp.Args] = None,
            entry_kwargs: tp.KwargsLike = None,
            exit_kwargs: tp.KwargsLike = None,
            cache_kwargs: tp.KwargsLike = None,
            return_cache: bool = False,
            use_cache: tp.Optional[CacheOutputT] = None,
            execute_kwargs: tp.KwargsLike = None,
            **_kwargs
        ) -> tp.Union[CacheOutputT, tp.Array2d, tp.List[tp.Array2d]]:
            # Get arguments
            if len(input_list) == 0:
                if input_shape is None:
                    raise ValueError("Pass input_shape if no input time series were passed")
            else:
                input_shape = input_list[0].shape

            if entry_args is None:
                entry_args = ()
            if exit_args is None:
                exit_args = ()
            if cache_args is None:
                cache_args = ()
            if mode == FactoryMode.Entries:
                if len(entry_args) > 0:
                    raise ValueError("Use *args instead of entry_args with FactoryMode.Entries")
                entry_args = args
            elif mode == FactoryMode.Exits or (mode == FactoryMode.Chain and entry_place_func_nb == first_place_nb):
                if len(exit_args) > 0:
                    raise ValueError("Use *args instead of exit_args with FactoryMode.Exits or FactoryMode.Chain")
                exit_args = args
            else:
                if len(args) > 0:
                    raise ValueError("*args cannot be used with FactoryMode.Both")

            if entry_kwargs is None:
                entry_kwargs = {}
            if exit_kwargs is None:
                exit_kwargs = {}
            if cache_kwargs is None:
                cache_kwargs = {}
            if mode == FactoryMode.Entries:
                if len(entry_kwargs) > 0:
                    raise ValueError("Use **kwargs instead of entry_kwargs with FactoryMode.Entries")
                entry_kwargs = _kwargs
            elif mode == FactoryMode.Exits or (mode == FactoryMode.Chain and entry_place_func_nb == first_place_nb):
                if len(exit_kwargs) > 0:
                    raise ValueError("Use **kwargs instead of exit_kwargs with FactoryMode.Exits or FactoryMode.Chain")
                exit_kwargs = _kwargs
            else:
                if len(_kwargs) > 0:
                    raise ValueError("*args cannot be used with FactoryMode.Both")

            kwargs_defaults = dict(
                input_shape=input_shape,
                only_once=mode == FactoryMode.Entries,
                wait=1,
                until_next=True,
                skip_until_exit=False,
                pick_first=mode != FactoryMode.Entries,
            )
            entry_kwargs = merge_dicts(kwargs_defaults, entry_kwargs)
            exit_kwargs = merge_dicts(kwargs_defaults, exit_kwargs)
            cache_kwargs = merge_dicts(kwargs_defaults, cache_kwargs)
            only_once = entry_kwargs["only_once"]
            entry_wait = entry_kwargs["wait"]
            exit_wait = exit_kwargs["wait"]
            until_next = exit_kwargs["until_next"]
            skip_until_exit = exit_kwargs["skip_until_exit"]

            # Distribute arguments across functions
            entry_input_list = []
            exit_input_list = []
            cache_input_list = []
            for input_name in entry_input_names:
                entry_input_list.append(input_list[input_names.index(input_name)])
            for input_name in exit_input_names:
                exit_input_list.append(input_list[input_names.index(input_name)])
            for input_name in cache_input_names:
                cache_input_list.append(input_list[input_names.index(input_name)])

            entry_in_output_list = []
            exit_in_output_list = []
            cache_in_output_list = []
            for in_output_name in entry_in_output_names:
                entry_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])
            for in_output_name in exit_in_output_names:
                exit_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])
            for in_output_name in cache_in_output_names:
                cache_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])

            entry_param_list = []
            exit_param_list = []
            cache_param_list = []
            for param_name in entry_param_names:
                entry_param_list.append(param_list[param_names.index(param_name)])
            for param_name in exit_param_names:
                exit_param_list.append(param_list[param_names.index(param_name)])
            for param_name in cache_param_names:
                cache_param_list.append(param_list[param_names.index(param_name)])

            n_params = len(param_list[0]) if len(param_list) > 0 else 1

            def _build_more_args(func_settings: tp.Kwargs, func_kwargs: tp.Kwargs) -> tp.Args:
                pass_kwargs = func_settings.get("pass_kwargs", [])
                if isinstance(pass_kwargs, dict):
                    pass_kwargs = list(pass_kwargs.items())
                more_args = ()
                for key in pass_kwargs:
                    value = None
                    if isinstance(key, tuple):
                        key, value = key
                    else:
                        if key.startswith("temp_idx_arr"):
                            value = np.empty((input_shape[0],), dtype=np.int_)
                    value = func_kwargs.get(key, value)
                    more_args += (value,)
                return more_args

            entry_more_args = _build_more_args(entry_settings, entry_kwargs)
            exit_more_args = _build_more_args(exit_settings, exit_kwargs)
            cache_more_args = _build_more_args(cache_settings, cache_kwargs)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                _cache_in_output_list = cache_in_output_list
                _cache_param_list = cache_param_list
                if checks.is_numba_func(cache_func):
                    _cache_in_output_list = list(map(to_typed_list, cache_in_output_list))
                    _cache_param_list = list(map(to_typed_list, cache_param_list))

                cache = cache_func(
                    *cache_input_list,
                    *_cache_in_output_list,
                    *_cache_param_list,
                    *cache_args,
                    *cache_more_args,
                )
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            entry_cache = ()
            exit_cache = ()
            if entry_settings.get("pass_cache", False):
                entry_cache = cache
            if exit_settings.get("pass_cache", False):
                exit_cache = cache

            # Apply and concatenate
            if mode == FactoryMode.Entries:
                _entry_in_output_list = list(map(to_typed_list, entry_in_output_list))
                _entry_param_list = list(map(to_typed_list, entry_param_list))

                return combining.apply_and_concat(
                    n_params,
                    apply_func,
                    input_shape,
                    only_once,
                    entry_wait,
                    *entry_input_list,
                    *_entry_in_output_list,
                    *_entry_param_list,
                    entry_args + entry_more_args + entry_cache,
                    n_outputs=1,
                    jitted_loop=True,
                    execute_kwargs=execute_kwargs,
                )

            elif mode == FactoryMode.Exits:
                _exit_in_output_list = list(map(to_typed_list, exit_in_output_list))
                _exit_param_list = list(map(to_typed_list, exit_param_list))

                return combining.apply_and_concat(
                    n_params,
                    apply_func,
                    input_list[0],
                    exit_wait,
                    until_next,
                    skip_until_exit,
                    *exit_input_list,
                    *_exit_in_output_list,
                    *_exit_param_list,
                    exit_args + exit_more_args + exit_cache,
                    n_outputs=1,
                    jitted_loop=True,
                    execute_kwargs=execute_kwargs,
                )

            else:
                _entry_in_output_list = list(map(to_typed_list, entry_in_output_list))
                _entry_param_list = list(map(to_typed_list, entry_param_list))
                _exit_in_output_list = list(map(to_typed_list, exit_in_output_list))
                _exit_param_list = list(map(to_typed_list, exit_param_list))

                return combining.apply_and_concat(
                    n_params,
                    apply_func,
                    input_shape,
                    entry_wait,
                    exit_wait,
                    *entry_input_list,
                    *_entry_in_output_list,
                    *_entry_param_list,
                    entry_args + entry_more_args + entry_cache,
                    *exit_input_list,
                    *_exit_in_output_list,
                    *_exit_param_list,
                    exit_args + exit_more_args + exit_cache,
                    n_outputs=2,
                    jitted_loop=True,
                    execute_kwargs=execute_kwargs,
                )

        return self.with_custom_func(custom_func, pass_packed=True, require_input_shape=require_input_shape, **kwargs)
