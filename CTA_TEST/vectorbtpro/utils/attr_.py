# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with class/instance attributes."""

import re
import inspect
from collections.abc import Iterable
from functools import cached_property as cachedproperty

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "deep_getattr",
]


def get_dict_attr(obj: tp.Union[object, type], attr: str) -> tp.Any:
    """Get attribute without invoking the attribute lookup machinery."""
    if inspect.isclass(obj):
        cls = obj
    else:
        cls = type(obj)
    for obj in [obj] + cls.mro():
        if attr in obj.__dict__:
            return obj.__dict__[attr]
    raise AttributeError


def default_getattr_func(
    obj: tp.Any,
    attr: str,
    args: tp.Optional[tp.Args] = None,
    kwargs: tp.Optional[tp.Kwargs] = None,
    call_attr: bool = True,
) -> tp.Any:
    """Default `getattr_func`."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    out = getattr(obj, attr)
    if callable(out) and call_attr:
        return out(*args, **kwargs)
    return out


def deep_getattr(
    obj: tp.Any,
    attr_chain: tp.Union[str, tuple, Iterable],
    getattr_func: tp.Callable = default_getattr_func,
    call_last_attr: bool = True,
) -> tp.Any:
    """Retrieve attribute consecutively.

    The attribute chain `attr_chain` can be:

    * string -> get variable/property or method without arguments
    * tuple of string -> call method without arguments
    * tuple of string and tuple -> call method and pass positional arguments (unpacked)
    * tuple of string, tuple, and dict -> call method and pass positional and keyword arguments (unpacked)
    * iterable of any of the above

    Use `getattr_func` to overwrite the default behavior of accessing an attribute (see `default_getattr_func`).

    !!! hint
        If your chain includes only attributes and functions without arguments,
        you can represent this chain as a single (but probably long) string.
    """
    checks.assert_instance_of(attr_chain, (str, tuple, Iterable))

    if isinstance(attr_chain, str):
        if "." in attr_chain:
            return deep_getattr(obj, attr_chain.split("."), getattr_func=getattr_func, call_last_attr=call_last_attr)
        outer = re.compile(r"(\w+)\((.*)\)")
        match = outer.match(attr_chain)
        if isinstance(attr_chain, str) and match:
            args = ()
            kwargs = dict()
            for arg in match.group(2).split(","):
                arg = arg.strip()
                if len(arg) == 0:
                    continue
                if "=" in arg:
                    kwargs[arg.split("=")[0]] = eval(arg.split("=")[1])
                else:
                    args += (eval(arg),)
            return deep_getattr(
                obj,
                (match.group(1), args, kwargs),
                getattr_func=getattr_func,
                call_last_attr=call_last_attr,
            )
        return getattr_func(obj, attr_chain, call_attr=call_last_attr)
    if isinstance(attr_chain, tuple):
        if len(attr_chain) == 1 and isinstance(attr_chain[0], str):
            return getattr_func(obj, attr_chain[0])
        if len(attr_chain) == 2 and isinstance(attr_chain[0], str) and isinstance(attr_chain[1], tuple):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1])
        if (
            len(attr_chain) == 3
            and isinstance(attr_chain[0], str)
            and isinstance(attr_chain[1], tuple)
            and isinstance(attr_chain[2], dict)
        ):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1], kwargs=attr_chain[2])
    result = obj
    for i, attr in enumerate(attr_chain):
        if i < len(attr_chain) - 1:
            result = deep_getattr(result, attr, getattr_func=getattr_func, call_last_attr=True)
        else:
            result = deep_getattr(result, attr, getattr_func=getattr_func, call_last_attr=call_last_attr)
    return result


AttrResolverMixinT = tp.TypeVar("AttrResolverMixinT", bound="AttrResolverMixin")


class AttrResolverMixin:
    """Class that implements resolution of self and its attributes.

    Resolution is `getattr` that works for self, properties, and methods. It also utilizes built-in caching."""

    @property
    def self_aliases(self) -> tp.Set[str]:
        """Names to associate with this object."""
        return {"self"}

    def resolve_self(
        self: AttrResolverMixinT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.ClassVar[tp.Optional[tp.Set[str]]] = None,
        impacts_caching: bool = True,
        silence_warnings: bool = False,
    ) -> AttrResolverMixinT:
        """Resolve self.

        !!! note
            `cond_kwargs` can be modified in-place."""
        return self

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Must return an attribute."""
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process an object after resolution.

        Must return an object."""
        return out

    @cachedproperty
    def cls_dir(self) -> tp.Set[str]:
        """Get set of attribute names."""
        return set(dir(type(self)))

    def resolve_shortcut_attr(self, attr: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute that may have shortcut properties."""
        if not attr.startswith("get_"):
            if "get_" + attr not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr), property):
                    return getattr(self, attr)
                return getattr(self, attr)(*args, **kwargs)
            attr = "get_" + attr

        return getattr(self, attr)(*args, **kwargs)

    def resolve_attr(
        self,
        attr: str,
        args: tp.ArgsLike = None,
        cond_kwargs: tp.KwargsLike = None,
        kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Container[str]] = None,
        cache_dct: tp.KwargsLike = None,
        use_caching: bool = True,
        passed_kwargs_out: tp.KwargsLike = None,
        use_shortcuts: bool = True,
    ) -> tp.Any:
        """Resolve an attribute using keyword arguments and built-in caching.

        * If there is a `get_{arg}` method, uses `get_{arg}` as `attr`.
        * If `attr` is a property, returns its value.
        * If `attr` is a method, passes `*args`, `**kwargs`, and `**cond_kwargs` with keys found in the signature.
        * If `use_shortcuts` is True, resolves the potential shortcuts using `AttrResolverMixin.resolve_shortcut_attr`.

        Won't cache if `use_caching` is False or any passed argument is in `custom_arg_names`.

        Use `passed_kwargs_out` to get keyword arguments that were passed."""
        # Resolve defaults
        if custom_arg_names is None:
            custom_arg_names = list()
        if cache_dct is None:
            cache_dct = {}
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if passed_kwargs_out is None:
            passed_kwargs_out = {}
        final_kwargs = merge_dicts(cond_kwargs, kwargs)

        # Resolve attribute
        cls = type(self)
        _attr = self.pre_resolve_attr(attr, final_kwargs=final_kwargs)
        if "get_" + attr in dir(cls):
            _attr = "get_" + attr
        if inspect.ismethod(getattr(cls, _attr)) or inspect.isfunction(getattr(cls, _attr)):
            attr_func = getattr(self, _attr)
            attr_func_kwargs = dict()
            attr_func_arg_names = get_func_arg_names(attr_func)
            custom_k = False
            for k, v in final_kwargs.items():
                if k in attr_func_arg_names or k in kwargs:
                    if k in custom_arg_names:
                        custom_k = True
                    attr_func_kwargs[k] = v
                    passed_kwargs_out[k] = v
            if use_caching and not custom_k and attr in cache_dct:
                out = cache_dct[attr]
            else:
                if use_shortcuts:
                    out = self.resolve_shortcut_attr(_attr, *args, **attr_func_kwargs)
                else:
                    out = attr_func(*args, **attr_func_kwargs)
                if use_caching and not custom_k:
                    cache_dct[attr] = out
        else:
            if use_caching and attr in cache_dct:
                out = cache_dct[attr]
            else:
                if use_shortcuts:
                    out = self.resolve_shortcut_attr(_attr)
                else:
                    out = getattr(self, _attr)
                if use_caching:
                    cache_dct[attr] = out
        out = self.post_resolve_attr(attr, out, final_kwargs=final_kwargs)
        return out

    def deep_getattr(self, *args, **kwargs) -> tp.Any:
        """See `deep_getattr`."""
        return deep_getattr(self, *args, **kwargs)


def parse_attrs(obj: tp.Any, own_only: bool = False, sort_by: tp.Optional[str] = None) -> tp.Frame:
    """Parse attributes of a class, object, or a module, and return a DataFrame with types and paths."""
    if inspect.isclass(obj) or inspect.ismodule(obj):
        cls = obj
    else:
        cls = type(obj)
    attr_info = {"type": {}, "path": {}}
    mro_dirs = []
    if not inspect.ismodule(cls):
        for super_cls in inspect.getmro(cls)[1:]:
            mro_dirs.append(set(dir(super_cls)))
    for k in dir(cls):
        if not k.startswith("_"):
            v = inspect.getattr_static(cls, k)
            if inspect.ismodule(v):
                if hasattr(v, "__path__"):
                    attr_info["type"][k] = "package"
                elif v.__spec__.origin in (None, "namespace"):
                    attr_info["type"][k] = "namespace"
                else:
                    attr_info["type"][k] = "module"
            else:
                attr_info["type"][k] = type(v).__name__

            if inspect.ismodule(cls):
                k_module = inspect.getmodule(v)
                if k_module is not None:
                    attr_info["path"][k] = inspect.getmodule(v).__name__
                else:
                    attr_info["path"][k] = "?"
            else:
                found_super_cls = False
                for i, super_cls in enumerate(inspect.getmro(cls)[1:]):
                    if k in mro_dirs[i]:
                        v2 = inspect.getattr_static(super_cls, k, None)
                        if v2 is not None and v == v2:
                            attr_info["path"][k] = super_cls.__module__ + "." + super_cls.__name__
                            found_super_cls = True
                if not found_super_cls:
                    attr_info["path"][k] = cls.__module__ + "." + cls.__name__

    if own_only:
        for k, v in list(attr_info["path"].items()):
            if inspect.ismodule(cls):
                if not v.startswith(cls.__name__):
                    del attr_info["type"][k]
                    del attr_info["path"][k]
            else:
                if not v.startswith(cls.__module__ + "." + cls.__name__):
                    del attr_info["type"][k]
                    del attr_info["path"][k]

    attr_info = pd.DataFrame(attr_info, dtype=object)
    if len(attr_info) > 0:
        attr_info.index.name = "attr"
        attr_info = attr_info.reset_index()
        if sort_by is not None:
            if isinstance(sort_by, str):
                if sort_by != "attr":
                    sort_by = [sort_by, "attr"]
            else:
                if "attr" not in sort_by:
                    sort_by = [*sort_by, "attr"]
            attr_info = attr_info.sort_values(sort_by)
        attr_info = attr_info.set_index("attr", drop=True)
    return attr_info
