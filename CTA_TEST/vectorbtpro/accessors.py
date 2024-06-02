# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Root Pandas accessors of vectorbtpro.

An accessor adds additional "namespace" to pandas objects.

The `vectorbtpro.accessors` registers a custom `vbt` accessor on top of each `pd.Index`, `pd.Series`,
and `pd.DataFrame` object. It is the main entry point for all other accessors:

```plaintext
vbt.base.accessors.BaseIDX/SR/DFAccessor       -> pd.Index/Series/DataFrame.vbt.*
vbt.generic.accessors.GenericSR/DFAccessor     -> pd.Series/DataFrame.vbt.*
vbt.signals.accessors.SignalsSR/DFAccessor     -> pd.Series/DataFrame.vbt.signals.*
vbt.returns.accessors.ReturnsSR/DFAccessor     -> pd.Series/DataFrame.vbt.returns.*
vbt.ohlcv.accessors.OHLCVDFAccessor            -> pd.DataFrame.vbt.ohlcv.*
vbt.px.accessors.PXSR/DFAccessor               -> pd.Series/DataFrame.vbt.px.*
```

Additionally, some accessors subclass other accessors building the following inheritance hiearchy:

```plaintext
vbt.base.accessors.BaseIDXAccessor
vbt.base.accessors.BaseSR/DFAccessor
    -> vbt.generic.accessors.GenericSR/DFAccessor
        -> vbt.signals.accessors.SignalsSR/DFAccessor
        -> vbt.returns.accessors.ReturnsSR/DFAccessor
        -> vbt.ohlcv.accessors.OHLCVDFAccessor
    -> vbt.px.accessors.PXSR/DFAccessor
```

So, for example, the method `pd.Series.vbt.to_2d_array` is also available as
`pd.Series.vbt.returns.to_2d_array`.

Class methods of any accessor can be conveniently accessed using `pd_acc`, `sr_acc`, and `df_acc` shortcuts:

```pycon
>>> import vectorbtpro as vbt

>>> vbt.pd_acc.signals.generate
<bound method SignalsAccessor.generate of <class 'vectorbtpro.signals.accessors.SignalsAccessor'>>
```

!!! note
    Accessors in vectorbt are not cached, so querying `df.vbt` twice will also call `Vbt_DFAccessor` twice.
    You can change this in global settings."""

import warnings

import pandas as pd
from pandas.core.accessor import DirNamesMixin

from vectorbtpro import _typing as tp
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor

__all__ = [
    "Vbt_Accessor",
    "Vbt_SRAccessor",
    "Vbt_DFAccessor",
    "pd_acc",
    "sr_acc",
    "df_acc",
]

__pdoc__ = {}

ParentAccessorT = tp.TypeVar("ParentAccessorT", bound=object)
AccessorT = tp.TypeVar("AccessorT", bound=object)


class Accessor:
    """Accessor."""

    def __init__(self, name: str, accessor: tp.Type[AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: ParentAccessorT, cls: DirNamesMixin) -> AccessorT:
        if obj is None:
            return self._accessor
        if isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)):
            accessor_obj = self._accessor(obj)
        elif issubclass(self._accessor, type(obj)):
            accessor_obj = obj.replace(cls_=self._accessor)
        else:
            accessor_obj = self._accessor(obj.wrapper, obj=obj._obj)
        return accessor_obj


class CachedAccessor:
    """Cached accessor.

    !!! warning
        Does not prevent from using old index data if the object's index has been changed in-place."""

    def __init__(self, name: str, accessor: tp.Type[AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: ParentAccessorT, cls: DirNamesMixin) -> AccessorT:
        if obj is None:
            return self._accessor
        if isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)):
            accessor_obj = self._accessor(obj)
        elif issubclass(self._accessor, type(obj)):
            accessor_obj = obj.replace(cls_=self._accessor)
        else:
            accessor_obj = self._accessor(obj.wrapper, obj=obj._obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def register_accessor(name: str, cls: tp.Type[DirNamesMixin]) -> tp.Callable:
    """Register a custom accessor.

    `cls` must subclass `pandas.core.accessor.DirNamesMixin`."""

    def decorator(accessor: tp.Type[AccessorT]) -> tp.Type[AccessorT]:
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                "attribute with the same name.",
                UserWarning,
                stacklevel=2,
            )
        if caching_cfg["use_cached_accessors"]:
            setattr(cls, name, CachedAccessor(name, accessor))
        else:
            setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator


def register_index_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.Index` accessor."""
    return register_accessor(name, pd.Index)


def register_series_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.Series` accessor."""
    return register_accessor(name, pd.Series)


def register_dataframe_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.DataFrame` accessor."""
    return register_accessor(name, pd.DataFrame)


@register_index_accessor("vbt")
class Vbt_IDXAccessor(DirNamesMixin, BaseIDXAccessor):
    """The main vectorbt accessor for `pd.Index`."""

    def __init__(self, obj: tp.Index, **kwargs) -> None:
        self._obj = obj

        DirNamesMixin.__init__(self)
        BaseIDXAccessor.__init__(self, obj, **kwargs)


class Vbt_Accessor(DirNamesMixin, GenericAccessor):
    """The main vectorbt accessor for `pd.Series` and `pd.DataFrame`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericAccessor.__init__(self, wrapper, obj=obj, **kwargs)


pd_acc = Vbt_Accessor
"""Shortcut for `Vbt_Accessor`."""

__pdoc__["pd_acc"] = False


@register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, GenericSRAccessor):
    """The main vectorbt accessor for `pd.Series`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericSRAccessor.__init__(self, wrapper, obj=obj, **kwargs)


sr_acc = Vbt_SRAccessor
"""Shortcut for `Vbt_SRAccessor`."""

__pdoc__["sr_acc"] = False


@register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, GenericDFAccessor):
    """The main vectorbt accessor for `pd.DataFrame`."""

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericDFAccessor.__init__(self, wrapper, obj=obj, **kwargs)


df_acc = Vbt_DFAccessor
"""Shortcut for `Vbt_DFAccessor`."""

__pdoc__["df_acc"] = False


def register_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_Accessor) -> tp.Callable:
    """Decorator to register an accessor on top of a parent accessor."""
    return register_accessor(name, parent)


def register_idx_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_IDXAccessor) -> tp.Callable:
    """Decorator to register a `pd.Index` accessor on top of a parent accessor."""
    return register_accessor(name, parent)


def register_sr_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_SRAccessor) -> tp.Callable:
    """Decorator to register a `pd.Series` accessor on top of a parent accessor."""
    return register_accessor(name, parent)


def register_df_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_DFAccessor) -> tp.Callable:
    """Decorator to register a `pd.DataFrame` accessor on top of a parent accessor."""
    return register_accessor(name, parent)
