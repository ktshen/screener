# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom Pandas accessors for signals.

Methods can be accessed as follows:

* `SignalsSRAccessor` -> `pd.Series.vbt.signals.*`
* `SignalsDFAccessor` -> `pd.DataFrame.vbt.signals.*`

```pycon
>>> import vectorbtpro as vbt
>>> from vectorbtpro.signals.enums import StopType
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> # vectorbtpro.signals.accessors.SignalsAccessor.pos_rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.pos_rank()
0   -1
1    0
2    1
3    2
4   -1
dtype: int64
```

The accessors extend `vectorbtpro.generic.accessors`.

!!! note
    The underlying Series/DataFrame must already be a signal series and have boolean data type.

    Grouping is only supported by the methods that accept the `group_by` argument.

    Accessors do not utilize caching.

Run for the examples below:
    
```pycon
>>> mask = pd.DataFrame({
...     'a': [True, False, False, False, False],
...     'b': [True, False, True, False, True],
...     'c': [True, True, True, False, False]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> mask
                a      b      c
2020-01-01   True   True   True
2020-01-02  False  False   True
2020-01-03  False   True   True
2020-01-04  False  False  False
2020-01-05  False   True  False
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `SignalsAccessor.metrics`.

```pycon
>>> mask.vbt.signals.stats(column='a')
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           1
Rate [%]                                     20.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance: Min                                 NaT
Distance: Median                              NaT
Distance: Max                                 NaT
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min                       NaT
Partition Distance: Median                    NaT
Partition Distance: Max                       NaT
Name: a, dtype: object
```

We can pass another signal array to compare this array with:

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(other=mask['b']))

Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           1
Rate [%]                                     20.0
Total Overlapping                               1
Overlapping Rate [%]                    33.333333
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance -> Other: Min            0 days 00:00:00
Distance -> Other: Median         2 days 00:00:00
Distance -> Other: Max            4 days 00:00:00
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min                       NaT
Partition Distance: Median                    NaT
Partition Distance: Max                       NaT
Name: a, dtype: object
```

We can also return duration as a floating number rather than a timedelta:

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(to_timedelta=False))
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                                          5
Total                                           1
Rate [%]                                     20.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance: Min                                 NaN
Distance: Median                              NaN
Distance: Max                                 NaN
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min                         1.0
Partition Length: Median                      1.0
Partition Length: Max                         1.0
Partition Distance: Min                       NaN
Partition Distance: Median                    NaN
Partition Distance: Max                       NaN
Name: a, dtype: object
```

`SignalsAccessor.stats` also supports (re-)grouping:

```pycon
>>> mask.vbt.signals.stats(column=0, group_by=[0, 0, 1])
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           4
Rate [%]                                     40.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-05 00:00:00
Norm Avg Index [-1, 1]                      -0.25
Distance: Min                     2 days 00:00:00
Distance: Median                  2 days 00:00:00
Distance: Max                     2 days 00:00:00
Total Partitions                                4
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min           2 days 00:00:00
Partition Distance: Median        2 days 00:00:00
Partition Distance: Max           2 days 00:00:00
Name: 0, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `SignalsAccessor.subplots`.

This class inherits subplots from `vectorbtpro.generic.accessors.GenericAccessor`.
"""

import warnings
from functools import partialmethod

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.accessors import register_vbt_accessor, register_df_vbt_accessor, register_sr_vbt_accessor
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base import reshaping
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.signals import nb
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.decorators import class_or_instancemethod, class_or_instanceproperty
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.template import RepEval, substitute_templates

__all__ = [
    "SignalsAccessor",
    "SignalsSRAccessor",
    "SignalsDFAccessor",
]

__pdoc__ = {}


@register_vbt_accessor("signals")
class SignalsAccessor(GenericAccessor):
    """Accessor on top of signal series. For both, Series and DataFrames.

    Accessible via `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        GenericAccessor.__init__(self, wrapper, obj=obj, **kwargs)

        checks.assert_dtype(self._obj, np.bool_)

    @class_or_instanceproperty
    def sr_accessor_cls(cls_or_self) -> tp.Type["SignalsSRAccessor"]:
        """Accessor class for `pd.Series`."""
        return SignalsSRAccessor

    @class_or_instanceproperty
    def df_accessor_cls(cls_or_self) -> tp.Type["SignalsDFAccessor"]:
        """Accessor class for `pd.DataFrame`."""
        return SignalsDFAccessor

    # ############# Overriding ############# #

    @classmethod
    def empty(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbtpro.base.accessors.BaseAccessor.empty` with `fill_value=False`."""
        return GenericAccessor.empty(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbtpro.base.accessors.BaseAccessor.empty_like` with `fill_value=False`."""
        return GenericAccessor.empty_like(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    bshift = partialmethod(GenericAccessor.bshift, fill_value=False)
    fshift = partialmethod(GenericAccessor.fshift, fill_value=False)
    ago = partialmethod(GenericAccessor.ago, fill_value=False)
    latest_at_index = partialmethod(GenericAccessor.latest_at_index, nan_value=False)

    # ############# Generation ############# #

    @classmethod
    def generate(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        place_func_nb: tp.PlaceFunc,
        *args,
        only_once: bool = True,
        wait: int = 1,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.signals.nb.generate_nb`.

        `shape` can be a shape-like tuple or an instance of `vectorbtpro.base.wrapping.ArrayWrapper`
        (will be used as `wrapper`).

        Usage:
            * Generate random signals manually:

            ```pycon
            >>> @njit
            ... def place_func_nb(c):
            ...     i = np.random.choice(len(c.out))
            ...     c.out[i] = True
            ...     return i

            >>> vbt.pd_acc.signals.generate(
            ...     (5, 3),
            ...     place_func_nb,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01   True  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False   True
            2020-01-04  False  False  False
            2020-01-05  False  False  False
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(
                broadcast_named_args,
                to_shape=shape_2d,
                **broadcast_kwargs
            )
        template_context = merge_dicts(
            broadcast_named_args,
            dict(shape=shape, shape_2d=shape_2d, wait=wait),
            template_context,
        )
        args = substitute_templates(args, template_context, sub_id="args")
        func = jit_reg.resolve_option(nb.generate_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        result = func(shape_2d, only_once, wait, place_func_nb, *args)

        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result, **wrap_kwargs)

    @classmethod
    def generate_both(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        entry_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
        entry_args: tp.ArgsLike = None,
        exit_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
        exit_args: tp.ArgsLike = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """See `vectorbtpro.signals.nb.generate_enex_nb`.

        `shape` can be a shape-like tuple or an instance of `vectorbtpro.base.wrapping.ArrayWrapper`
        (will be used as `wrapper`).

        Usage:
            * Generate entry and exit signals one after another:

            ```pycon
            >>> @njit
            ... def place_func_nb(c):
            ...     c.out[0] = True
            ...     return 0

            >>> en, ex = vbt.pd_acc.signals.generate_both(
            ...     (5, 3),
            ...     entry_place_func_nb=place_func_nb,
            ...     exit_place_func_nb=place_func_nb,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False
            2020-01-03   True   True   True
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```

            * Generate three entries and one exit one after another:

            ```pycon
            >>> @njit
            ... def entry_place_func_nb(c, n):
            ...     c.out[:n] = True
            ...     return n - 1

            >>> @njit
            ... def exit_place_func_nb(c, n):
            ...     c.out[:n] = True
            ...     return n - 1

            >>> en, ex = vbt.pd_acc.signals.generate_both(
            ...     (5, 3),
            ...     entry_place_func_nb=entry_place_func_nb,
            ...     entry_args=(3,),
            ...     exit_place_func_nb=exit_place_func_nb,
            ...     exit_args=(1,),
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02   True   True   True
            2020-01-03   True   True   True
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        if entry_args is None:
            entry_args = ()
        if exit_args is None:
            exit_args = ()
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(
                broadcast_named_args,
                to_shape=shape_2d,
                **broadcast_kwargs,
            )
        template_context = merge_dicts(
            broadcast_named_args,
            dict(
                shape=shape,
                shape_2d=shape_2d,
                entry_wait=entry_wait,
                exit_wait=exit_wait,
            ),
            template_context,
        )
        entry_args = substitute_templates(entry_args, template_context, sub_id="entry_args")
        exit_args = substitute_templates(exit_args, template_context, sub_id="exit_args")
        func = jit_reg.resolve_option(nb.generate_enex_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        result1, result2 = func(
            shape_2d,
            entry_wait,
            exit_wait,
            entry_place_func_nb,
            entry_args,
            exit_place_func_nb,
            exit_args,
        )
        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result1, **wrap_kwargs), wrapper.wrap(result2, **wrap_kwargs)

    def generate_exits(
        self,
        exit_place_func_nb: tp.PlaceFunc,
        *args,
        wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.signals.nb.generate_ex_nb`.

        Usage:
            * Generate an exit just before the next entry:

            ```pycon
            >>> @njit
            ... def exit_place_func_nb(c):
            ...     c.out[-1] = True
            ...     return len(c.out) - 1

            >>> mask.vbt.signals.generate_exits(exit_place_func_nb)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04  False   True  False
            2020-01-05   True  False   True
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        obj = self.obj
        if len(broadcast_named_args) > 0:
            broadcast_named_args = {"obj": obj, **broadcast_named_args}
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
            obj = broadcast_named_args["obj"]
        else:
            wrapper = self.wrapper
            obj = reshaping.to_2d_array(obj)
        template_context = merge_dicts(
            broadcast_named_args,
            dict(wait=wait, until_next=until_next, skip_until_exit=skip_until_exit),
            template_context,
        )
        args = substitute_templates(args, template_context, sub_id="args")
        func = jit_reg.resolve_option(nb.generate_ex_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        exits = func(obj, wait, until_next, skip_until_exit, exit_place_func_nb, *args)
        return wrapper.wrap(exits, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Cleaning ############# #

    @class_or_instancemethod
    def clean(
        cls_or_self,
        *args,
        force_first: bool = True,
        keep_conflicts: bool = False,
        reverse_order: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Clean signals.

        If one array passed, see `SignalsAccessor.first`. If two arrays passed, entries and exits,
        see `vectorbtpro.signals.nb.clean_enex_nb`."""
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}
        if not isinstance(cls_or_self, type):
            args = (cls_or_self.obj, *args)
        if len(args) == 1:
            obj = args[0]
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                obj = ArrayWrapper.from_obj(obj).wrap(obj)
            return obj.vbt.signals.first(wrap_kwargs=wrap_kwargs, jitted=jitted, chunked=chunked)
        if len(args) == 2:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcasted_args, wrapper = reshaping.broadcast(
                dict(entries=args[0], exits=args[1]),
                return_wrapper=True,
                **broadcast_kwargs,
            )
            func = jit_reg.resolve_option(nb.clean_enex_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            entries_out, exits_out = func(
                broadcasted_args["entries"],
                broadcasted_args["exits"],
                force_first,
                keep_conflicts,
                reverse_order
            )
            return (
                wrapper.wrap(entries_out, group_by=False, **wrap_kwargs),
                wrapper.wrap(exits_out, group_by=False, **wrap_kwargs),
            )
        raise ValueError("Either one or two arrays must be passed")

    # ############# Random signals ############# #

    @classmethod
    def generate_random(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        n: tp.Optional[tp.ArrayLike] = None,
        prob: tp.Optional[tp.ArrayLike] = None,
        pick_first: bool = False,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate signals randomly.

        `shape` can be a shape-like tuple or an instance of `vectorbtpro.base.wrapping.ArrayWrapper`
        (will be used as `wrapper`).

        If `n` is set, uses `vectorbtpro.signals.nb.rand_place_nb`.
        If `prob` is set, uses `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

        For arguments, see `SignalsAccessor.generate`.

        `n` must be either a scalar or an array that will broadcast to the number of columns.
        `prob` must be either a single number or an array that will broadcast to match `shape`.

        Specify `seed` to make output deterministic.

        Usage:
            * For each column, generate a variable number of signals:

            ```pycon
            >>> vbt.pd_acc.signals.generate_random(
            ...     (5, 3),
            ...     n=[0, 1, 2],
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False   True
            2020-01-04  False   True  False
            2020-01-05  False  False   True
            ```

            * For each column and time step, pick a signal with 50% probability:

            ```pycon
            >>> vbt.pd_acc.signals.generate_random(
            ...     (5, 3),
            ...     prob=0.5,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04  False  False   True
            2020-01-05   True  False   True
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        shape_2d = cls.resolve_shape(shape)
        if n is not None and prob is not None:
            raise ValueError("Either n or prob must be provided, not both")

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = reshaping.broadcast_array_to(n, shape_2d[1])
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(base_ch.FlexArraySlicer())),
            )
            return cls.generate(
                shape,
                jit_reg.resolve_option(nb.rand_place_nb, jitted),
                n,
                jitted=jitted,
                chunked=chunked,
                **kwargs,
            )
        if prob is not None:
            prob = reshaping.to_2d_array(reshaping.broadcast_array_to(prob, shape))
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(base_ch.FlexArraySlicer(axis=1), None, None)),
            )
            return cls.generate(
                shape,
                jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob,
                pick_first,
                jitted=jitted,
                chunked=chunked,
                **kwargs,
            )
        raise ValueError("At least n or prob must be provided")

    @classmethod
    def generate_random_both(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        n: tp.Optional[tp.ArrayLike] = None,
        entry_prob: tp.Optional[tp.ArrayLike] = None,
        exit_prob: tp.Optional[tp.ArrayLike] = None,
        seed: tp.Optional[int] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        entry_pick_first: bool = True,
        exit_pick_first: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Generate chain of entry and exit signals randomly.

        `shape` can be a shape-like tuple or an instance of `vectorbtpro.base.wrapping.ArrayWrapper`
        (will be used as `wrapper`).

        If `n` is set, uses `vectorbtpro.signals.nb.generate_rand_enex_nb`.
        If `entry_prob` and `exit_prob` are set, uses `SignalsAccessor.generate_both` with
        `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

        Usage:
            * For each column, generate two entries and exits randomly:

            ```pycon
            >>> en, ex = vbt.pd_acc.signals.generate_random_both(
            ...     (5, 3),
            ...     n=2,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01  False  False   True
            2020-01-02   True   True  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False   True
            2020-01-03   True   True  False
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            ```

            * For each column and time step, pick entry with 50% probability and exit right after:

            ```pycon
            >>> en, ex = vbt.pd_acc.signals.generate_random_both(
            ...     (5, 3),
            ...     entry_prob=0.5,
            ...     exit_prob=1.,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False   True
            2020-01-05   True  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False   True
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        shape_2d = cls.resolve_shape(shape)
        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or any of the entry_prob and exit_prob must be provided, not both")

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = reshaping.broadcast_array_to(n, shape_2d[1])
            func = jit_reg.resolve_option(nb.generate_rand_enex_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            entries, exits = func(shape_2d, n, entry_wait, exit_wait)
            if wrapper is None:
                wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
            if wrap_kwargs is None:
                wrap_kwargs = resolve_dict(wrap_kwargs)
            return wrapper.wrap(entries, **wrap_kwargs), wrapper.wrap(exits, **wrap_kwargs)
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = reshaping.to_2d_array(reshaping.broadcast_array_to(entry_prob, shape))
            exit_prob = reshaping.to_2d_array(reshaping.broadcast_array_to(exit_prob, shape))
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(base_ch.FlexArraySlicer(axis=1), None, None),
                    exit_args=ch.ArgsTaker(base_ch.FlexArraySlicer(axis=1), None, None),
                ),
            )
            return cls.generate_both(
                shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                entry_args=(entry_prob, entry_pick_first),
                exit_place_func_nb=jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                exit_args=(exit_prob, exit_pick_first),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                jitted=jitted,
                chunked=chunked,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
            )
        raise ValueError("At least n, or entry_prob and exit_prob must be provided")

    def generate_random_exits(
        self,
        prob: tp.Optional[tp.ArrayLike] = None,
        seed: tp.Optional[int] = None,
        wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate exit signals randomly.

        If `prob` is None, uses `vectorbtpro.signals.nb.rand_place_nb`.
        Otherwise, uses `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

        Uses `SignalsAccessor.generate_exits`.

        Specify `seed` to make output deterministic.

        Usage:
            * After each entry in `mask`, generate exactly one exit:

            ```pycon
            >>> mask.vbt.signals.generate_random_exits(seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04   True   True  False
            2020-01-05  False  False   True
            ```

            * After each entry in `mask` and at each time step, generate exit with 50% probability:

            ```pycon
            >>> mask.vbt.signals.generate_random_exits(prob=0.5, seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False   True
            ```
        """
        if seed is not None:
            set_seed_nb(seed)
        if prob is not None:
            broadcast_kwargs = merge_dicts(
                dict(keep_flex=dict(obj=False, prob=True)),
                broadcast_kwargs,
            )
            broadcasted_args = reshaping.broadcast(
                dict(obj=self.obj, prob=prob),
                **broadcast_kwargs,
            )
            obj = broadcasted_args["obj"]
            prob = broadcasted_args["prob"]
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(base_ch.FlexArraySlicer(axis=1), None, None)),
            )
            return obj.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob,
                True,
                wait=wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
        n = reshaping.broadcast_array_to(1, self.wrapper.shape_2d[1])
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(args=ch.ArgsTaker(base_ch.FlexArraySlicer())),
        )
        return self.generate_exits(
            jit_reg.resolve_option(nb.rand_place_nb, jitted),
            n,
            wait=wait,
            until_next=until_next,
            skip_until_exit=skip_until_exit,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    # ############# Stop signals ############# #

    def generate_stop_exits(
        self,
        entry_ts: tp.ArrayLike,
        ts: tp.ArrayLike = np.nan,
        follow_ts: tp.ArrayLike = np.nan,
        stop: tp.ArrayLike = np.nan,
        trailing: tp.ArrayLike = False,
        out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        chain: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when `ts` hits the stop.

        For arguments, see `vectorbtpro.signals.nb.stop_place_nb`.
        If `chain` is True, uses `SignalsAccessor.generate_both`.
        Otherwise, uses `SignalsAccessor.generate_exits`.

        Use `out_dict` as a dict to pass `stop_ts` array. You can also set `out_dict` to {}
        to produce this array automatically and still have access to it.

        All array-like arguments including stops and `out_dict` will broadcast using
        `vectorbtpro.base.reshaping.broadcast` and `broadcast_kwargs`.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that come between an entry and its exit,
            set `until_next` to False and `skip_until_exit` to True.

            To remove all entries that come between an entry and its exit,
            set `chain` to True. This will return two arrays: new entries and exits.

        Usage:
            * Regular stop loss:

            ```pycon
            >>> ts = pd.Series([1, 2, 3, 2, 1])

            >>> mask.vbt.signals.generate_stop_exits(ts, stop=-0.1)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False
            ```

            * Trailing stop loss:

            ```pycon
            >>> mask.vbt.signals.generate_stop_exits(ts, stop=-0.1, trailing=True)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```

            * Testing multiple take profit stops:

            ```pycon
            >>> mask.vbt.signals.generate_stop_exits(ts, stop=vbt.Param([1.0, 1.5]))
            stop                        1.0                  1.5
                            a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False
            2020-01-02   True   True  False  False  False  False
            2020-01-03  False  False  False   True  False  False
            2020-01-04  False  False  False  False  False  False
            2020-01-05  False  False  False  False  False  False
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        entries = self.obj
        if out_dict is None:
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        stop_ts = out_dict.get("stop_ts", np.nan if out_dict_passed else None)

        broadcastable_args = dict(
            entries=entries,
            entry_ts=entry_ts,
            ts=ts,
            follow_ts=follow_ts,
            stop=stop,
            trailing=trailing,
            stop_ts=stop_ts,
        )
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=dict(entries=False, stop_ts=False, _def=True),
                require_kwargs=dict(requirements="W"),
            ),
            broadcast_kwargs,
        )
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args["entries"]
        stop_ts = broadcasted_args["stop_ts"]
        if stop_ts is None:
            stop_ts = np.empty_like(entries, dtype=np.float_)
        stop_ts = reshaping.to_2d_array(stop_ts)

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(ch.ArraySlicer(axis=1)),
                    exit_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    ),
                ),
            )
            out_dict["stop_ts"] = wrapper.wrap(stop_ts, group_by=False, **wrap_kwargs)
            return cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.first_place_nb, jitted),
                entry_args=(entries_arr,),
                exit_place_func_nb=jit_reg.resolve_option(nb.stop_place_nb, jitted),
                exit_args=(
                    broadcasted_args["entry_ts"],
                    broadcasted_args["ts"],
                    broadcasted_args["follow_ts"],
                    stop_ts,
                    broadcasted_args["stop"],
                    broadcasted_args["trailing"],
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
        else:
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    )
                ),
            )
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)
            out_dict["stop_ts"] = wrapper.wrap(stop_ts, group_by=False, **wrap_kwargs)
            return entries.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.stop_place_nb, jitted),
                broadcasted_args["entry_ts"],
                broadcasted_args["ts"],
                broadcasted_args["follow_ts"],
                stop_ts,
                broadcasted_args["stop"],
                broadcasted_args["trailing"],
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )

    def generate_ohlc_stop_exits(
        self,
        entry_price: tp.ArrayLike,
        open: tp.ArrayLike = np.nan,
        high: tp.ArrayLike = np.nan,
        low: tp.ArrayLike = np.nan,
        close: tp.ArrayLike = np.nan,
        sl_stop: tp.ArrayLike = np.nan,
        tsl_th: tp.ArrayLike = np.nan,
        tsl_stop: tp.ArrayLike = np.nan,
        tp_stop: tp.ArrayLike = np.nan,
        reverse: tp.ArrayLike = False,
        is_entry_open: bool = False,
        out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        chain: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when the price hits (trailing) stop loss or take profit.

        Use `out_dict` as a dict to pass `stop_price` and `stop_type` arrays. You can also
        set `out_dict` to {} to produce these arrays automatically and still have access to them.

        For arguments, see `vectorbtpro.signals.nb.ohlc_stop_place_nb`.
        If `chain` is True, uses `SignalsAccessor.generate_both`.
        Otherwise, uses `SignalsAccessor.generate_exits`.

        All array-like arguments including stops and `out_dict` will broadcast using
        `vectorbtpro.base.reshaping.broadcast` and `broadcast_kwargs`.

        For arguments, see `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that come between an entry and its exit,
            set `until_next` to False and `skip_until_exit` to True.

            To remove all entries that come between an entry and its exit,
            set `chain` to True. This will return two arrays: new entries and exits.

        Usage:
            * Generate exits for TSL and TP of 10%:

            ```pycon
            >>> price = pd.DataFrame({
            ...     'open': [10, 11, 12, 11, 10],
            ...     'high': [11, 12, 13, 12, 11],
            ...     'low': [9, 10, 11, 10, 9],
            ...     'close': [10, 11, 12, 11, 10]
            ... })
            >>> out_dict = {}
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price["open"],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ... )
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True  False
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False

            >>> out_dict['stop_price']
                           a     b     c
            2020-01-01   NaN   NaN   NaN
            2020-01-02  11.0  11.0   NaN
            2020-01-03   NaN   NaN   NaN
            2020-01-04   NaN  10.8  10.8
            2020-01-05   NaN   NaN   NaN

            >>> out_dict['stop_type'].vbt(mapping=StopType).apply_mapping()
                           a     b     c
            2020-01-01  None  None  None
            2020-01-02    TP    TP  None
            2020-01-03  None  None  None
            2020-01-04  None   TSL   TSL
            2020-01-05  None  None  None
            ```

            Notice how the first two entry signals in the third column have no exit signal -
            there is no room between them for an exit signal.

            * To find an exit for the first entry and ignore all entries that are in-between them,
            we can pass `until_next=False` and `skip_until_exit=True`:

            ```pycon
            >>> out_dict = {}
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ...     until_next=False,
            ...     skip_until_exit=True
            ... )
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False

            >>> out_dict['stop_price']
                           a     b     c
            2020-01-01   NaN   NaN   NaN
            2020-01-02  11.0  11.0  11.0
            2020-01-03   NaN   NaN   NaN
            2020-01-04   NaN  10.8  10.8
            2020-01-05   NaN   NaN   NaN

            >>> out_dict['stop_type'].vbt(mapping=StopType).apply_mapping()
                           a     b     c
            2020-01-01  None  None  None
            2020-01-02    TP    TP    TP
            2020-01-03  None  None  None
            2020-01-04  None   TSL   TSL
            2020-01-05  None  None  None
            ```

            Now, the first signal in the third column gets executed regardless of the entries that come next,
            which is very similar to the logic that is implemented in `vectorbtpro.portfolio.base.Portfolio.from_signals`.

            * To automatically remove all ignored entry signals, pass `chain=True`.
            This will return a new entries array:

            ```pycon
            >>> out_dict = {}
            >>> new_entries, exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ...     chain=True
            ... )
            >>> new_entries
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False  << removed entry in the third column
            2020-01-03  False   True   True
            2020-01-04  False  False  False
            2020-01-05  False   True  False
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False
            ```

            !!! warning
                The last two examples above make entries dependent upon exits - this makes only sense
                if you have no other exit arrays to combine this stop exit array with.

            * Test multiple parameter combinations:

            ```pycon
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     sl_stop=vbt.Param([False, 0.1]),
            ...     tsl_stop=vbt.Param([False, 0.1]),
            ...     is_entry_open=True
            ... )
            >>> exits
            sl_stop     False                                       0.1                \\
            tsl_stop    False                  0.1                False
                            a      b      c      a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False  False  False  False
            2020-01-02  False  False  False  False  False  False  False  False  False
            2020-01-03  False  False  False  False  False  False  False  False  False
            2020-01-04  False  False  False   True   True   True  False   True   True
            2020-01-05  False  False  False  False  False  False   True  False  False

            sl_stop
            tsl_stop      0.1
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        entries = self.obj
        if out_dict is None:
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        stop_price = out_dict.get("stop_price", np.nan if out_dict_passed else None)
        stop_type = out_dict.get("stop_type", -1 if out_dict_passed else None)

        broadcastable_args = dict(
            entries=entries,
            entry_price=entry_price,
            open=open,
            high=high,
            low=low,
            close=close,
            sl_stop=sl_stop,
            tsl_th=tsl_th,
            tsl_stop=tsl_stop,
            tp_stop=tp_stop,
            reverse=reverse,
            stop_price=stop_price,
            stop_type=stop_type,
        )
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=dict(entries=False, stop_price=False, stop_type=False, _def=True),
                require_kwargs=dict(requirements="W"),
            ),
            broadcast_kwargs,
        )
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args["entries"]
        stop_price = broadcasted_args["stop_price"]
        if stop_price is None:
            stop_price = np.empty_like(entries, dtype=np.float_)
        stop_price = reshaping.to_2d_array(stop_price)
        stop_type = broadcasted_args["stop_type"]
        if stop_type is None:
            stop_type = np.empty_like(entries, dtype=np.int_)
        stop_type = reshaping.to_2d_array(stop_type)

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(ch.ArraySlicer(axis=1)),
                    exit_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                        None,
                    ),
                ),
            )
            new_entries, exits = cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.first_place_nb, jitted),
                entry_args=(entries_arr,),
                exit_place_func_nb=jit_reg.resolve_option(nb.ohlc_stop_place_nb, jitted),
                exit_args=(
                    broadcasted_args["entry_price"],
                    broadcasted_args["open"],
                    broadcasted_args["high"],
                    broadcasted_args["low"],
                    broadcasted_args["close"],
                    stop_price,
                    stop_type,
                    broadcasted_args["sl_stop"],
                    broadcasted_args["tsl_th"],
                    broadcasted_args["tsl_stop"],
                    broadcasted_args["tp_stop"],
                    broadcasted_args["reverse"],
                    is_entry_open,
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
            out_dict["stop_price"] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict["stop_type"] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return new_entries, exits
        else:
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                        None,
                    )
                ),
            )
            exits = entries.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.ohlc_stop_place_nb, jitted),
                broadcasted_args["entry_price"],
                broadcasted_args["open"],
                broadcasted_args["high"],
                broadcasted_args["low"],
                broadcasted_args["close"],
                stop_price,
                stop_type,
                broadcasted_args["sl_stop"],
                broadcasted_args["tsl_th"],
                broadcasted_args["tsl_stop"],
                broadcasted_args["tp_stop"],
                broadcasted_args["reverse"],
                is_entry_open,
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
            out_dict["stop_price"] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict["stop_type"] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return exits

    # ############# Ranking ############# #

    def rank(
        self,
        rank_func_nb: tp.RankFunc,
        *args,
        reset_by: tp.Optional[tp.ArrayLike] = None,
        after_false: bool = False,
        after_reset: bool = False,
        reset_wait: int = 1,
        as_mapped: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """See `vectorbtpro.signals.nb.rank_nb`.

        Will broadcast with `reset_by` using `vectorbtpro.base.reshaping.broadcast` and `broadcast_kwargs`.

        Set `as_mapped` to True to return an instance of `vectorbtpro.records.mapped_array.MappedArray`."""
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        if reset_by is not None:
            broadcast_named_args = {"obj": self.obj, "reset_by": reset_by, **broadcast_named_args}
        else:
            broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        obj = reshaping.to_2d_array(broadcast_named_args["obj"])
        if reset_by is not None:
            reset_by = reshaping.to_2d_array(broadcast_named_args["reset_by"])
        template_context = merge_dicts(
            dict(
                obj=obj,
                reset_by=reset_by,
                after_false=after_false,
                after_reset=after_reset,
                reset_wait=reset_wait,
            ),
            template_context,
        )
        args = substitute_templates(args, template_context, sub_id="args")
        func = jit_reg.resolve_option(nb.rank_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        rank = func(obj, reset_by, after_false, after_reset, reset_wait, rank_func_nb, *args)
        rank_wrapped = wrapper.wrap(rank, group_by=False, **wrap_kwargs)
        if as_mapped:
            rank_wrapped = rank_wrapped.replace(-1, np.nan)
            return rank_wrapped.vbt.to_mapped(dropna=True, dtype=np.int_, **kwargs)
        return rank_wrapped

    def pos_rank(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        allow_gaps: bool = False,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get signal position ranks.

        Uses `SignalsAccessor.rank` with `vectorbtpro.signals.nb.sig_pos_rank_nb`.

        Usage:
            * Rank each True value in each partition in `mask`:

            ```pycon
            >>> mask.vbt.signals.pos_rank()
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  0  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1

            >>> mask.vbt.signals.pos_rank(after_false=True)
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02 -1 -1 -1
            2020-01-03 -1  0 -1
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1

            >>> mask.vbt.signals.pos_rank(allow_gaps=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  1  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  2 -1

            >>> mask.vbt.signals.pos_rank(reset_by=~mask, allow_gaps=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  0  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1
            ```
        """
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.rank(
            jit_reg.resolve_option(nb.sig_pos_rank_nb, jitted),
            allow_gaps,
            jitted=jitted,
            chunked=chunked,
            **kwargs,
        )

    def pos_rank_after(
        self,
        reset_by: tp.ArrayLike,
        after_reset: bool = True,
        allow_gaps: bool = True,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get signal position ranks after each signal in `reset_by`.

        !!! note
            `allow_gaps` is enabled by default."""
        return self.pos_rank(reset_by=reset_by, after_reset=after_reset, allow_gaps=allow_gaps, **kwargs)

    def partition_pos_rank(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get partition position ranks.

        Uses `SignalsAccessor.rank` with `vectorbtpro.signals.nb.part_pos_rank_nb`.

        Usage:
            * Rank each partition of True values in `mask`:

            ```pycon
            >>> mask.vbt.signals.partition_pos_rank()
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  0
            2020-01-03 -1  1  0
            2020-01-04 -1 -1 -1
            2020-01-05 -1  2 -1

            >>> mask.vbt.signals.partition_pos_rank(after_false=True)
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02 -1 -1 -1
            2020-01-03 -1  0 -1
            2020-01-04 -1 -1 -1
            2020-01-05 -1  1 -1

            >>> mask.vbt.signals.partition_pos_rank(reset_by=mask)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  0
            2020-01-03 -1  0  0
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1
            ```
        """
        return self.rank(
            jit_reg.resolve_option(nb.part_pos_rank_nb, jitted),
            jitted=jitted,
            chunked=chunked,
            **kwargs,
        )

    def partition_pos_rank_after(self, reset_by: tp.ArrayLike, **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get partition position ranks after each signal in `reset_by`."""
        return self.partition_pos_rank(reset_by=reset_by, after_reset=True, **kwargs)

    def first(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == 0`.

        Uses `SignalsAccessor.pos_rank`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **resolve_dict(wrap_kwargs))

    def first_after(
        self,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == 0`.

        Uses `SignalsAccessor.pos_rank_after`."""
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **resolve_dict(wrap_kwargs))

    def nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == n`.

        Uses `SignalsAccessor.pos_rank`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == n, group_by=False, **resolve_dict(wrap_kwargs))

    def nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == n`.

        Uses `SignalsAccessor.pos_rank_after`."""
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank == n, group_by=False, **resolve_dict(wrap_kwargs))

    def from_nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank >= n`.

        Uses `SignalsAccessor.pos_rank`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **resolve_dict(wrap_kwargs))

    def from_nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank >= n`.

        Uses `SignalsAccessor.pos_rank_after`."""
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **resolve_dict(wrap_kwargs))

    def to_nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank < n`.

        Uses `SignalsAccessor.pos_rank`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank < n, group_by=False, **resolve_dict(wrap_kwargs))

    def to_nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank < n`.

        Uses `SignalsAccessor.pos_rank_after`."""
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank < n, group_by=False, **resolve_dict(wrap_kwargs))

    def pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of signal position ranks.

        Uses `SignalsAccessor.pos_rank`."""
        return self.pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    def partition_pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of partition position ranks.

        Uses `SignalsAccessor.partition_pos_rank`."""
        return self.partition_pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    # ############# Conversion ############# #

    def to_mapped(
        self,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArray:
        """Convert this object into an instance of `vectorbtpro.records.mapped_array.MappedArray`."""
        mapped_arr = self.to_2d_array().flatten(order="F")
        col_arr = np.repeat(np.arange(self.wrapper.shape_2d[1]), self.wrapper.shape_2d[0])
        idx_arr = np.tile(np.arange(self.wrapper.shape_2d[0]), self.wrapper.shape_2d[1])
        new_mapped_arr = mapped_arr[mapped_arr]
        new_col_arr = col_arr[mapped_arr]
        new_idx_arr = idx_arr[mapped_arr]
        return MappedArray(
            wrapper=self.wrapper,
            mapped_arr=new_mapped_arr,
            col_arr=new_col_arr,
            idx_arr=new_idx_arr,
            **kwargs,
        ).regroup(group_by)

    # ############# Ranges ############# #

    def delta_ranges(
        self,
        delta: tp.Union[str, int, tp.FrequencyLike],
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Ranges:
        """Build a record array of the type `vectorbtpro.generic.ranges.Ranges`
        from a delta applied after each signal (or before if delta is negative)."""
        return Ranges.from_delta(self.to_mapped(), delta=delta, **kwargs).regroup(group_by)

    def between_ranges(
        self,
        other: tp.Optional[tp.ArrayLike] = None,
        from_other: bool = False,
        incl_open: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        attach_other: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Wrap the result of `vectorbtpro.signals.nb.between_ranges_nb`
        with `vectorbtpro.generic.ranges.Ranges`.

        If `other` specified, see `vectorbtpro.signals.nb.between_two_ranges_nb`.
        Both will broadcast using `vectorbtpro.base.reshaping.broadcast` and `broadcast_kwargs`.

        Usage:
            * One array:

            ```pycon
            >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
            >>> ranges = mask_sr.vbt.signals.between_ranges()
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29ea7c7b8>

            >>> ranges.records_readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              3  Closed
            1         1       0                3              5  Closed
            2         2       0                5              6  Closed

            >>> ranges.duration.values
            array([3, 2, 1])
            ```

            * Two arrays, traversing the signals of the first array:

            ```pycon
            >>> mask_sr = pd.Series([True, True, True, False, False])
            >>> mask_sr2 = pd.Series([False, False, True, False, True])
            >>> ranges = mask_sr.vbt.signals.between_ranges(other=mask_sr2)
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29e3b80f0>

            >>> ranges.records_readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              2  Closed
            1         1       0                1              2  Closed
            2         2       0                2              2  Closed

            >>> ranges.duration.values
            array([2, 1, 0])
            ```

            * Two arrays, traversing the signals of the second array:

            ```pycon
            >>> ranges = mask_sr.vbt.signals.between_ranges(other=mask_sr2, from_other=True)
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29eccbd68>

            >>> ranges.records_readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                2              2  Closed
            1         1       0                2              4  Closed

            >>> ranges.duration.values
            array([0, 2])
            ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if other is None:
            # One input array
            func = jit_reg.resolve_option(nb.between_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            range_records = func(self.to_2d_array(), incl_open=incl_open)
            wrapper = self.wrapper
            to_attach = self.obj
        else:
            # Two input arrays
            broadcasted_args = reshaping.broadcast(
                dict(obj=self.obj, other=other),
                **broadcast_kwargs
            )
            obj = broadcasted_args["obj"]
            other = broadcasted_args["other"]
            func = jit_reg.resolve_option(nb.between_two_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            range_records = func(
                reshaping.to_2d_array(obj),
                reshaping.to_2d_array(other),
                from_other=from_other,
                incl_open=incl_open,
            )
            wrapper = ArrayWrapper.from_obj(obj)
            to_attach = other if attach_other else obj
        kwargs = merge_dicts(dict(close=to_attach), kwargs)
        return Ranges.from_records(wrapper, range_records, **kwargs).regroup(group_by)

    def partition_ranges(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Wrap the result of `vectorbtpro.signals.nb.partition_ranges_nb`
        with `vectorbtpro.generic.ranges.Ranges`.

        If `use_end_idxs` is True, uses the index of the last signal in each partition as `idx_arr`.
        Otherwise, uses the index of the first signal.

        Usage:
            ```pycon
            >>> mask_sr = pd.Series([True, True, True, False, True, True])
            >>> mask_sr.vbt.signals.partition_ranges().records_readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              3  Closed
            1         1       0                4              5    Open
            ```
        """
        func = jit_reg.resolve_option(nb.partition_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        kwargs = merge_dicts(dict(close=self.obj), kwargs)
        return Ranges.from_records(self.wrapper, range_records, **kwargs).regroup(group_by)

    def between_partition_ranges(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Wrap the result of `vectorbtpro.signals.nb.between_partition_ranges_nb`
        with `vectorbtpro.generic.ranges.Ranges`.

        Usage:
            ```pycon
            >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
            >>> mask_sr.vbt.signals.between_partition_ranges().records_readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              3  Closed
            1         1       0                3              5  Closed
            ```
        """
        func = jit_reg.resolve_option(nb.between_partition_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        kwargs = merge_dicts(dict(close=self.obj), kwargs)
        return Ranges.from_records(self.wrapper, range_records, **kwargs).regroup(group_by)

    # ############# Index ############# #

    def nth_index(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """See `vectorbtpro.signals.nb.nth_index_nb`.

        Usage:
            ```pycon
            >>> mask.vbt.signals.nth_index(0)
            a   2020-01-01
            b   2020-01-01
            c   2020-01-01
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(2)
            a          NaT
            b   2020-01-05
            c   2020-01-03
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(-1)
            a   2020-01-01
            b   2020-01-05
            c   2020-01-03
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(-1, group_by=True)
            Timestamp('2020-01-05 00:00:00')
            ```
        """
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            squeezed = self.squeeze_grouped(
                jit_reg.resolve_option(generic_nb.any_reduce_nb, jitted),
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
            arr = reshaping.to_2d_array(squeezed)
        else:
            arr = self.to_2d_array()
        func = jit_reg.resolve_option(nb.nth_index_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        nth_index = func(arr, n)
        wrap_kwargs = merge_dicts(dict(name_or_index="nth_index", to_index=True), wrap_kwargs)
        return self.wrapper.wrap_reduced(nth_index, group_by=group_by, **wrap_kwargs)

    def norm_avg_index(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """See `vectorbtpro.signals.nb.norm_avg_index_nb`.

        Normalized average index measures the average signal location relative to the middle of the column.
        This way, we can quickly see where the majority of signals are located.

        Common values are:

        * -1.0: only the first signal is set
        * 1.0: only the last signal is set
        * 0.0: symmetric distribution around the middle
        * [-1.0, 0.0): average signal is on the left
        * (0.0, 1.0]: average signal is on the right

        Usage:
            ```pycon
            >>> pd.Series([True, False, False, False]).vbt.signals.norm_avg_index()
            -1.0

            >>> pd.Series([False, False, False, True]).vbt.signals.norm_avg_index()
            1.0

            >>> pd.Series([True, False, False, True]).vbt.signals.norm_avg_index()
            0.0
            ```
        """
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.norm_avg_index_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array(), group_lens)
        else:
            func = jit_reg.resolve_option(nb.norm_avg_index_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index="norm_avg_index"), wrap_kwargs)
        return self.wrapper.wrap_reduced(norm_avg_index, group_by=group_by, **wrap_kwargs)

    def index_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of indices.

        See `vectorbtpro.generic.accessors.GenericAccessor.to_mapped`.

        Only True values will be considered."""
        indices = np.arange(len(self.wrapper.index), dtype=np.float_)[:, None]
        indices = np.tile(indices, (1, len(self.wrapper.columns)))
        indices = reshaping.soft_to_ndim(indices, self.wrapper.ndim)
        indices[~self.obj.values] = np.nan
        return self.wrapper.wrap(indices).vbt.to_mapped(dropna=True, dtype=np.int_, group_by=group_by, **kwargs)

    def total(self, wrap_kwargs: tp.KwargsLike = None, group_by: tp.GroupByLike = None) -> tp.MaybeSeries:
        """Total number of True values in each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="total"), wrap_kwargs)
        return self.sum(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def rate(self, wrap_kwargs: tp.KwargsLike = None, group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """`SignalsAccessor.total` divided by the total index length in each column/group."""
        total = reshaping.to_1d_array(self.total(group_by=group_by, **kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index="rate"), wrap_kwargs)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total / total_steps, group_by=group_by, **wrap_kwargs)

    def total_partitions(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Total number of partitions of True values in each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index="total_partitions"), wrap_kwargs)
        return self.partition_ranges(**kwargs).count(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def partition_rate(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """`SignalsAccessor.total_partitions` divided by `SignalsAccessor.total` in each column/group."""
        total_partitions = reshaping.to_1d_array(self.total_partitions(group_by=group_by, *kwargs))
        total = reshaping.to_1d_array(self.total(group_by=group_by, *kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index="partition_rate"), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_partitions / total, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `SignalsAccessor.stats`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults` and
        `stats` from `vectorbtpro._settings.signals`."""
        from vectorbtpro._settings import settings

        signals_stats_cfg = settings["signals"]["stats"]

        return merge_dicts(GenericAccessor.stats_defaults.__get__(self), signals_stats_cfg)

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
            total=dict(title="Total", calc_func="total", tags="signals"),
            rate=dict(
                title="Rate [%]",
                calc_func="rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="signals",
            ),
            total_overlapping=dict(
                title="Total Overlapping",
                calc_func=lambda self, other, group_by: (self & other).vbt.signals.total(group_by=group_by),
                check_silent_has_other=True,
                tags=["signals", "other"],
            ),
            overlapping_rate=dict(
                title="Overlapping Rate [%]",
                calc_func=lambda self, other, group_by: (self & other).vbt.signals.total(group_by=group_by)
                / (self | other).vbt.signals.total(group_by=group_by),
                post_calc_func=lambda self, out, settings: out * 100,
                check_silent_has_other=True,
                tags=["signals", "other"],
            ),
            first_index=dict(
                title="First Index",
                calc_func="nth_index",
                n=0,
                wrap_kwargs=dict(to_index=True),
                tags=["signals", "index"],
            ),
            last_index=dict(
                title="Last Index",
                calc_func="nth_index",
                n=-1,
                wrap_kwargs=dict(to_index=True),
                tags=["signals", "index"],
            ),
            norm_avg_index=dict(title="Norm Avg Index [-1, 1]", calc_func="norm_avg_index", tags=["signals", "index"]),
            distance=dict(
                title=RepEval(
                    "f'Distance {\"<-\" if from_other else \"->\"} {other_name}' if other is not None else 'Distance'"
                ),
                calc_func="between_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=RepEval("['signals', 'distance', 'other'] if other is not None else ['signals', 'distance']"),
            ),
            total_partitions=dict(
                title="Total Partitions",
                calc_func="total_partitions",
                tags=["signals", "partitions"],
            ),
            partition_rate=dict(
                title="Partition Rate [%]",
                calc_func="partition_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=["signals", "partitions"],
            ),
            partition_len=dict(
                title="Partition Length",
                calc_func="partition_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=["signals", "partitions", "distance"],
            ),
            partition_distance=dict(
                title="Partition Distance",
                calc_func="between_partition_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=["signals", "partitions", "distance"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        yref: str = "y",
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals.

        Args:
            yref (str): Y coordinate axis.
            column (hashable): Column to plot.
            **kwargs: Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.lineplot`.

        Usage:
            ```pycon
            >>> mask[['a', 'c']].vbt.signals.plot().show()
            ```

            ![](/assets/images/api/signals_df_plot.svg){: .iimg loading=lazy }
        """
        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self
        default_kwargs = dict(trace_kwargs=dict(line=dict(shape="hv")))
        default_kwargs["yaxis" + yref[1:]] = dict(tickmode="array", tickvals=[0, 1], ticktext=["false", "true"])
        return _self.obj.vbt.lineplot(**merge_dicts(default_kwargs, kwargs))

    def plot_as_markers(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot Series as markers.

        Args:
            y (array_like): Y-axis values to plot markers on.
            column (hashable): Column to plot.
            **kwargs: Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.scatterplot`.

        Usage:
            ```pycon
            >>> ts = pd.Series([1, 2, 3, 2, 1], index=mask.index)
            >>> fig = ts.vbt.lineplot()
            >>> mask['b'].vbt.signals.plot_as_entries(y=ts, fig=fig)
            >>> (~mask['b']).vbt.signals.plot_as_exits(y=ts, fig=fig).show()
            ```

            ![](/assets/images/api/signals_plot_as_markers.svg){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        obj = self.obj
        if isinstance(obj, pd.DataFrame):
            obj = self.select_col_from_obj(obj, column=column)
        if y is None:
            y = pd.Series.vbt.empty_like(obj, 1)
        else:
            y = reshaping.to_pd_array(y)
            if isinstance(y, pd.DataFrame):
                y = self.select_col_from_obj(y, column=column)
            obj, y = reshaping.broadcast(obj, y, columns_from="keep")
            obj = obj.fillna(False).astype(np.bool_)
            if y.name is None:
                y = y.rename("Y")

        def_kwargs = dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol="circle",
                    color=plotting_cfg["contrast_color_schema"]["blue"],
                    size=7,
                ),
                name=obj.name,
            )
        )
        kwargs = merge_dicts(def_kwargs, kwargs)
        if "marker_color" in kwargs["trace_kwargs"]:
            marker_color = kwargs["trace_kwargs"]["marker_color"]
        else:
            marker_color = kwargs["trace_kwargs"]["marker"]["color"]
        if isinstance(marker_color, str) and "rgba" not in marker_color:
            line_color = adjust_lightness(marker_color)
        else:
            line_color = marker_color
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    marker=dict(
                        line=dict(width=1, color=line_color),
                    ),
                ),
            ),
            kwargs,
        )
        return y[obj].vbt.scatterplot(**kwargs)

    def plot_as_entries(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as entry markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="triangle-up",
                            color=plotting_cfg["contrast_color_schema"]["green"],
                            size=8,
                        ),
                        name="Entries",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_exits(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as exit markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="triangle-down",
                            color=plotting_cfg["contrast_color_schema"]["red"],
                            size=8,
                        ),
                        name="Exits",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_entry_marks(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as marked entry markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="circle",
                            color="rgba(0, 0, 0, 0)",
                            size=20,
                            line=dict(
                                color=plotting_cfg["contrast_color_schema"]["green"],
                                width=2,
                            ),
                        ),
                        name="Entry marks",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_exit_marks(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Label] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as marked exit markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="circle",
                            color="rgba(0, 0, 0, 0)",
                            size=20,
                            line=dict(
                                color=plotting_cfg["contrast_color_schema"]["red"],
                                width=2,
                            ),
                        ),
                        name="Exit marks",
                    )
                ),
                kwargs,
            ),
        )

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `SignalsAccessor.plots`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults` and
        `plots` from `vectorbtpro._settings.signals`."""
        from vectorbtpro._settings import settings

        signals_plots_cfg = settings["signals"]["plots"]

        return merge_dicts(GenericAccessor.plots_defaults.__get__(self), signals_plots_cfg)

    @property
    def subplots(self) -> Config:
        return self._subplots


SignalsAccessor.override_metrics_doc(__pdoc__)
SignalsAccessor.override_subplots_doc(__pdoc__)


@register_sr_vbt_accessor("signals")
class SignalsSRAccessor(SignalsAccessor, GenericSRAccessor):
    """Accessor on top of signal series. For Series only.

    Accessible via `pd.Series.vbt.signals`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        GenericSRAccessor.__init__(self, wrapper, obj=obj, **kwargs)
        SignalsAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_df_vbt_accessor("signals")
class SignalsDFAccessor(SignalsAccessor, GenericDFAccessor):
    """Accessor on top of signal series. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.signals`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, **kwargs)
        SignalsAccessor.__init__(self, wrapper, obj=obj, **kwargs)
