# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for preparing arguments."""

import inspect
import string
from collections import defaultdict
from datetime import timedelta, time
from functools import cached_property as cachedproperty
from pathlib import Path

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import index_dict, IdxSetter, IdxSetterFactory, IdxRecords
from vectorbtpro.base.reshaping import BCO, Default, Ref
from vectorbtpro.base.reshaping import broadcast
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.decorators import override_arg_config, attach_arg_properties
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.config import merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.cutting import suggest_module_path, cut_and_save_func
from vectorbtpro.utils.datetime_ import (
    freq_to_timedelta64,
    parse_timedelta,
    time_to_timedelta,
    try_align_to_dt_index,
    to_ns,
)
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.module_ import import_module_from_path
from vectorbtpro.utils.params import Param
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "BasePreparer",
]

__pdoc__ = {}


base_arg_config = ReadonlyConfig(
    dict(
        broadcast_named_args=dict(is_dict=True),
        broadcast_kwargs=dict(is_dict=True),
        template_context=dict(is_dict=True),
        seed=dict(),
        jitted=dict(),
        chunked=dict(),
        staticized=dict(),
        records=dict(),
    )
)
"""_"""

__pdoc__[
    "base_arg_config"
] = f"""Argument config for `BasePreparer`.

```python
{base_arg_config.prettify()}
```
"""


class MetaArgs(type):
    """Meta class that exposes a read-only class property `MetaArgs.arg_config`."""

    @property
    def arg_config(cls) -> Config:
        """Argument config."""
        return cls._arg_config


@attach_arg_properties
@override_arg_config(base_arg_config)
class BasePreparer(Configured, metaclass=MetaArgs):
    """Base class for preparing target functions and arguments.

    !!! warning
        Most properties are force-cached - create a new instance to override any attribute."""

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_arg_config"}

    _setting_keys: tp.SettingsKeys = None

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

        # Copy writeable attrs
        self._arg_config = type(self)._arg_config.copy()

    _arg_config: tp.ClassVar[Config] = HybridConfig()

    @property
    def arg_config(self) -> Config:
        """Argument config of `${cls_name}`.

        ```python
        ${arg_config}
        ```
        """
        return self._arg_config

    @classmethod
    def map_enum_value(cls, value: tp.ArrayLike, look_for_type: tp.Optional[type] = None, **kwargs) -> tp.ArrayLike:
        """Map enumerated value(s)."""
        if look_for_type is not None:
            if isinstance(value, look_for_type):
                return map_enum_fields(value, **kwargs)
            return value
        if isinstance(value, (CustomTemplate, Ref)):
            return value
        if isinstance(value, (Param, BCO, Default)):
            attr_dct = attr.asdict(value)
            if isinstance(value, Param) and attr_dct["map_template"] is None:
                attr_dct["map_template"] = RepFunc(lambda values: cls.map_enum_value(values, **kwargs))
            elif not isinstance(value, Param):
                attr_dct["value"] = cls.map_enum_value(attr_dct["value"], **kwargs)
            return type(value)(**attr_dct)
        if isinstance(value, index_dict):
            return index_dict({k: cls.map_enum_value(v, **kwargs) for k, v in value.items()})
        if isinstance(value, IdxSetterFactory):
            value = value.get()
            if not isinstance(value, IdxSetter):
                raise ValueError("Index setter factory must return exactly one index setter")
        if isinstance(value, IdxSetter):
            return IdxSetter([(k, cls.map_enum_value(v, **kwargs)) for k, v in value.idx_items])
        return map_enum_fields(value, **kwargs)

    @classmethod
    def prepare_td_obj(cls, td_obj: object) -> object:
        """Prepare a timedelta object for broadcasting."""
        if isinstance(td_obj, (str, timedelta, pd.DateOffset, pd.Timedelta)):
            td_obj = freq_to_timedelta64(td_obj)
        elif isinstance(td_obj, pd.Index):
            td_obj = td_obj.values
        return td_obj

    @classmethod
    def prepare_dt_obj(cls, dt_obj: object, ns_ago: int = 0) -> object:
        """Prepare a datetime object for broadcasting."""
        if isinstance(dt_obj, (str, time, timedelta, pd.DateOffset, pd.Timedelta)):
            dt_obj_dt_template = RepEval(
                "try_align_to_dt_index([dt_obj], wrapper.index).vbt.to_ns() - ns_ago",
                context=dict(try_align_to_dt_index=try_align_to_dt_index, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            dt_obj_td_template = RepEval(
                "wrapper.index.vbt.to_period_ns(parse_timedelta(dt_obj)) - ns_ago",
                context=dict(parse_timedelta=parse_timedelta, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            dt_obj_time_template = RepEval(
                '(wrapper.index.floor("1d") + time_to_timedelta(dt_obj)).vbt.to_ns() - ns_ago',
                context=dict(time_to_timedelta=time_to_timedelta, dt_obj=dt_obj, ns_ago=ns_ago),
            )
            if isinstance(dt_obj, str):
                try:
                    time.fromisoformat(dt_obj)
                    dt_obj = dt_obj_time_template
                except Exception as e:
                    try:
                        parse_timedelta(dt_obj)
                        dt_obj = dt_obj_td_template
                    except Exception as e:
                        dt_obj = dt_obj_dt_template
            elif isinstance(dt_obj, time):
                dt_obj = dt_obj_time_template
            else:
                dt_obj = dt_obj_td_template
        elif isinstance(dt_obj, pd.Index):
            dt_obj = dt_obj.values
        return dt_obj

    def get_raw_arg_default(self, arg_name: str, is_dict: bool = False) -> tp.Any:
        """Get raw argument default."""
        if self._setting_keys is None:
            if is_dict:
                return {}
            return None
        value = self.get_setting(arg_name)
        if is_dict and value is None:
            return {}
        return value

    def get_raw_arg(self, arg_name: str, is_dict: bool = False, has_default: bool = True) -> tp.Any:
        """Get raw argument."""
        value = self.config.get(arg_name, None)
        if is_dict:
            if has_default:
                return merge_dicts(self.get_raw_arg_default(arg_name), value)
            if value is None:
                return {}
            return value
        if value is None and has_default:
            return self.get_raw_arg_default(arg_name)
        return value

    @cachedproperty
    def idx_setters(self) -> tp.Optional[tp.Dict[tp.Label, IdxSetter]]:
        """Index setters from resolving the argument `records`."""
        arg_config = self.arg_config["records"]
        records = self.get_raw_arg(
            "records",
            is_dict=arg_config.get("is_dict", False),
            has_default=arg_config.get("has_default", True),
        )
        if records is None:
            return None
        if not isinstance(records, IdxRecords):
            records = IdxRecords(records)
        idx_setters = records.get()
        for k in idx_setters:
            if k in self.arg_config and not self.arg_config[k].get("broadcast", False):
                raise ValueError(f"Field {k} is not broadcastable and cannot be included in records")
        rename_fields = arg_config.get("rename_fields", {})
        new_idx_setters = {}
        for k, v in idx_setters.items():
            if k in rename_fields:
                k = rename_fields[k]
            new_idx_setters[k] = v
        return new_idx_setters

    def get_arg_default(self, arg_name: str) -> tp.Any:
        """Get argument default according to the argument config."""
        arg_config = self.arg_config[arg_name]
        arg = self.get_raw_arg_default(
            arg_name,
            is_dict=arg_config.get("is_dict", False),
        )
        if arg is not None:
            if len(arg_config.get("map_enum_kwargs", {})) > 0:
                arg = self.map_enum_value(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.prepare_td_obj(arg)
            if arg_config.get("is_dt", False):
                arg = self.prepare_dt_obj(arg, ns_ago=arg_config.get("ns_ago", 0))
        return arg

    def get_arg(self, arg_name: str, use_idx_setter: bool = True, use_default: bool = True) -> tp.Any:
        """Get mapped argument according to the argument config."""
        arg_config = self.arg_config[arg_name]
        if use_idx_setter and self.idx_setters is not None and arg_name in self.idx_setters:
            arg = self.idx_setters[arg_name]
        else:
            arg = self.get_raw_arg(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                has_default=arg_config.get("has_default", True) if use_default else False,
            )
        if arg is not None:
            if len(arg_config.get("map_enum_kwargs", {})) > 0:
                arg = self.map_enum_value(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.prepare_td_obj(arg)
            if arg_config.get("is_dt", False):
                arg = self.prepare_dt_obj(arg, ns_ago=arg_config.get("ns_ago", 0))
        return arg

    def __getitem__(self, arg_name) -> tp.Any:
        return self.get_arg(arg_name)

    @classmethod
    def td_arr_to_ns(cls, td_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Prepare a timedelta array."""
        if td_arr.dtype == object:
            if td_arr.ndim in (0, 1):
                td_arr = pd.to_timedelta(td_arr)
                if isinstance(td_arr, pd.Timedelta):
                    td_arr = td_arr.to_timedelta64()
                else:
                    td_arr = td_arr.values
            else:
                td_arr_cols = []
                for col in range(td_arr.shape[1]):
                    td_arr_col = pd.to_timedelta(td_arr[:, col])
                    td_arr_cols.append(td_arr_col.values)
                td_arr = np.column_stack(td_arr_cols)
        return to_ns(td_arr)

    @classmethod
    def dt_arr_to_ns(cls, dt_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Prepare a datetime array."""
        if dt_arr.dtype == object:
            if dt_arr.ndim in (0, 1):
                dt_arr = pd.to_datetime(dt_arr).tz_localize(None)
                if isinstance(dt_arr, pd.Timestamp):
                    dt_arr = dt_arr.to_datetime64()
                else:
                    dt_arr = dt_arr.values
            else:
                dt_arr_cols = []
                for col in range(dt_arr.shape[1]):
                    dt_arr_col = pd.to_datetime(dt_arr[:, col]).tz_localize(None)
                    dt_arr_cols.append(dt_arr_col.values)
                dt_arr = np.column_stack(dt_arr_cols)
        return to_ns(dt_arr)

    def prepare_post_arg(self, arg_name: str, value: tp.Optional[tp.ArrayLike] = None) -> object:
        """Prepare an argument after broadcasting and/or template substitution."""
        if value is None:
            if arg_name in self.post_args:
                arg = self.post_args[arg_name]
            else:
                arg = getattr(self, "_pre_" + arg_name)
        else:
            arg = value
        if arg is not None:
            arg_config = self.arg_config[arg_name]
            if arg_config.get("substitute_templates", False):
                arg = substitute_templates(arg, self.template_context, sub_id=arg_name)
            if "map_enum_kwargs" in arg_config:
                arg = map_enum_fields(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.td_arr_to_ns(arg)
            if arg_config.get("is_dt", False):
                arg = self.dt_arr_to_ns(arg)
            if "subdtype" in arg_config:
                checks.assert_subdtype(arg, arg_config["subdtype"], arg_name=arg_name)
        return arg

    @classmethod
    def adapt_staticized_to_udf(cls, staticized: tp.Kwargs, func: tp.Union[str, tp.Callable], func_name: str) -> None:
        """Adapt `staticized` dictionary to a UDF."""
        target_func_module = inspect.getmodule(staticized["func"])
        if isinstance(func, (str, Path)):
            if isinstance(func, str) and not func.endswith(".py") and hasattr(target_func_module, func):
                staticized[f"{func_name}_block"] = func
                return None
            func = Path(func)
            module_path = func.resolve()
        else:
            if inspect.getmodule(func) == target_func_module:
                staticized[f"{func_name}_block"] = func.__name__
                return None
            module = inspect.getmodule(func)
            if not hasattr(module, "__file__"):
                raise TypeError(f"{func_name} must be defined in a Python file")
            module_path = Path(module.__file__).resolve()
        if "import_lines" not in staticized:
            staticized["import_lines"] = []
        reload = staticized.get("reload", False)
        staticized["import_lines"].extend(
            [
                f'{func_name}_path = r"{module_path}"',
                f"globals().update(vbt.import_module_from_path({func_name}_path).__dict__, reload={reload})",
            ]
        )

    @classmethod
    def find_target_func(cls, target_func_name: str) -> tp.Callable:
        """Find target function by its name."""
        raise NotImplementedError

    @classmethod
    def resolve_dynamic_target_func(cls, target_func_name: str, staticized: tp.KwargsLike) -> tp.Callable:
        """Resolve a dynamic target function."""
        if staticized is None:
            func = cls.find_target_func(target_func_name)
        else:
            if isinstance(staticized, dict):
                staticized = dict(staticized)
                module_path = suggest_module_path(
                    staticized.get("suggest_fname", target_func_name),
                    path=staticized.pop("path", None),
                    mkdir_kwargs=staticized.get("mkdir_kwargs", None),
                )
                if "new_func_name" not in staticized:
                    staticized["new_func_name"] = target_func_name

                if staticized.pop("override", False) or not module_path.exists():
                    if "skip_func" not in staticized:

                        def _skip_func(out_lines, func_name):
                            to_skip = lambda x: f"def {func_name}" in x or x.startswith(f"{func_name}_path =")
                            return any(map(to_skip, out_lines))

                        staticized["skip_func"] = _skip_func
                    module_path = cut_and_save_func(path=module_path, **staticized)
                reload = staticized.pop("reload", False)
                module = import_module_from_path(module_path, reload=reload)
                func = getattr(module, staticized["new_func_name"])
            else:
                func = staticized
        return func

    def set_seed(self) -> None:
        """Set seed."""
        seed = self.seed
        if seed is not None:
            set_seed(seed)

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_template_context(self) -> tp.Kwargs:
        """Argument `template_context` before broadcasting."""
        return merge_dicts(dict(preparer=self), self["template_context"])

    # ############# Broadcasting ############# #

    @cachedproperty
    def pre_args(self) -> tp.Kwargs:
        """Arguments before broadcasting."""
        pre_args = dict()
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                pre_args[k] = getattr(self, "_pre_" + k)
        return pre_args

    @cachedproperty
    def args_to_broadcast(self) -> dict:
        """Arguments to broadcast."""
        return merge_dicts(self.idx_setters, self.pre_args, self.broadcast_named_args)

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        """Default keyword arguments for broadcasting."""
        return dict(
            to_pd=False,
            keep_flex=dict(cash_earnings=self.keep_inout_flex, _def=True),
            wrapper_kwargs=dict(
                freq=self._pre_freq,
                group_by=self.group_by,
            ),
            return_wrapper=True,
        )

    @cachedproperty
    def broadcast_kwargs(self) -> tp.Kwargs:
        """Argument `broadcast_kwargs`."""
        arg_broadcast_kwargs = defaultdict(dict)
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                broadcast_kwargs = v.get("broadcast_kwargs", None)
                if broadcast_kwargs is None:
                    broadcast_kwargs = {}
                for k2, v2 in broadcast_kwargs.items():
                    arg_broadcast_kwargs[k2][k] = v2
        for k in self.args_to_broadcast:
            new_fill_value = None
            if k in self.pre_args:
                fill_default = self.arg_config[k].get("fill_default", True)
                if self.idx_setters is not None and k in self.idx_setters:
                    new_fill_value = self.get_arg(k, use_idx_setter=False, use_default=fill_default)
                elif fill_default and self.arg_config[k].get("has_default", True):
                    new_fill_value = self.get_arg_default(k)
            elif k in self.broadcast_named_args:
                if self.idx_setters is not None and k in self.idx_setters:
                    new_fill_value = self.broadcast_named_args[k]
            if new_fill_value is not None:
                if not np.isscalar(new_fill_value):
                    raise TypeError(f"Argument '{k}' (and its default) must be a scalar when also provided via records")
                if "reindex_kwargs" not in arg_broadcast_kwargs:
                    arg_broadcast_kwargs["reindex_kwargs"] = {}
                if k not in arg_broadcast_kwargs["reindex_kwargs"]:
                    arg_broadcast_kwargs["reindex_kwargs"][k] = {}
                arg_broadcast_kwargs["reindex_kwargs"][k]["fill_value"] = new_fill_value

        return merge_dicts(
            self.def_broadcast_kwargs,
            dict(arg_broadcast_kwargs),
            self["broadcast_kwargs"],
        )

    @cachedproperty
    def broadcast_result(self) -> tp.Any:
        """Result of broadcasting."""
        return broadcast(self.args_to_broadcast, **self.broadcast_kwargs)

    @cachedproperty
    def post_args(self) -> tp.Kwargs:
        """Arguments after broadcasting."""
        return self.broadcast_result[0]

    @cachedproperty
    def post_broadcast_named_args(self) -> tp.Kwargs:
        """Custom arguments after broadcasting."""
        if self.broadcast_named_args is None:
            return dict()
        post_broadcast_named_args = dict()
        for k, v in self.post_args.items():
            if k in self.broadcast_named_args:
                post_broadcast_named_args[k] = v
            elif self.idx_setters is not None and k in self.idx_setters and k not in self.pre_args:
                post_broadcast_named_args[k] = v
        return post_broadcast_named_args

    @cachedproperty
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self.broadcast_result[1]

    @cachedproperty
    def target_shape(self) -> tp.Shape:
        """Target shape."""
        return self.wrapper.shape_2d

    @cachedproperty
    def index(self) -> tp.Array1d:
        """Index in nanosecond format."""
        return self.wrapper.ns_index

    @cachedproperty
    def freq(self) -> int:
        """Frequency in nanosecond format."""
        return self.wrapper.ns_freq

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        """Argument `template_context`."""
        builtin_args = {}
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                builtin_args[k] = getattr(self, k)
        return merge_dicts(
            dict(
                wrapper=self.wrapper,
                target_shape=self.target_shape,
                index=self.index,
                freq=self.freq,
            ),
            builtin_args,
            self.post_broadcast_named_args,
            self._pre_template_context,
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        """Target function."""
        return None

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        """Map of the target arguments to the preparer attributes."""
        return dict()

    @cachedproperty
    def target_args(self) -> tp.Optional[tp.Kwargs]:
        """Arguments to be passed to the target function."""
        if self.target_func is not None:
            target_arg_map = self.target_arg_map
            func_arg_names = get_func_arg_names(self.target_func)
            target_args = {}
            for k in func_arg_names:
                arg_attr = target_arg_map.get(k, k)
                if arg_attr is not None:
                    target_args[k] = getattr(self, arg_attr)
            return target_args
        return None

    # ############# Docs ############# #

    @classmethod
    def build_arg_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build argument config documentation."""
        if source_cls is None:
            source_cls = BasePreparer
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "arg_config").__doc__)).substitute(
            {"arg_config": cls.arg_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_arg_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `BasePreparer.arg_config`."""
        __pdoc__[cls.__name__ + ".arg_config"] = cls.build_arg_config_doc(source_cls=source_cls)


BasePreparer.override_arg_config_doc(__pdoc__)
