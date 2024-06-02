# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class decorators for records."""

import keyword
from functools import partial

from vectorbtpro import _typing as tp
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.decorators import cacheable_property, cached_property
from vectorbtpro.utils.mapping import to_value_mapping

__all__ = []


def override_field_config(config: Config, merge_configs: bool = True) -> tp.ClassWrapper:
    """Class decorator to override the field config of a class subclassing
    `vectorbtpro.records.base.Records`.

    Instead of overriding `_field_config` class attribute, you can pass `config` directly to this decorator.

    Disable `merge_configs` to not merge, which will effectively disable field inheritance."""

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")
        if merge_configs:
            new_config = merge_dicts(cls.field_config, config)
        else:
            new_config = config
        if not isinstance(new_config, Config):
            new_config = HybridConfig(new_config)
        setattr(cls, "_field_config", new_config)
        return cls

    return wrapper


def attach_fields(*args, on_conflict: str = "raise") -> tp.FlexClassWrapper:
    """Class decorator to attach field properties in a `vectorbtpro.records.base.Records` class.

    Will extract `dtype` and other relevant information from `vectorbtpro.records.base.Records.field_config`
    and map its fields as properties. This behavior can be changed by using `config`.

    !!! note
        Make sure to run `attach_fields` after `override_field_config`.

    `config` must contain fields (keys) and dictionaries (values) with the following keys:

    * `attach`: Whether to attach the field property. Can be provided as a string to be used
        as a target attribute name. Defaults to True.
    * `defaults`: Dictionary with default keyword arguments for `vectorbtpro.records.base.Records.map_field`.
        Defaults to an empty dict.
    * `attach_filters`: Whether to attach filters based on the field's values. Can be provided as a dict
        to be used instead of the mapping (filter value -> target filter name). Defaults to False.
        If True, defaults to `mapping` in `vectorbtpro.records.base.Records.field_config`.
    * `filter_defaults`: Dictionary with default keyword arguments for `vectorbtpro.records.base.Records.apply_mask`.
        Can be provided by target filter name. Defaults to an empty dict.
    * `on_conflict`: Overrides global `on_conflict` for both field and filter properties.

    Any potential attribute name is prepared by placing underscores between capital letters and
    converting to the lower case.

    If an attribute with the same name already exists in the class but the name is not listed in the field config:

    * it will be overridden if `on_conflict` is 'override'
    * it will be ignored if `on_conflict` is 'ignore'
    * an error will be raised if `on_conflict` is 'raise'
    """

    def wrapper(cls: tp.Type[tp.T], config: tp.DictLike = None) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")

        dtype = cls.field_config.get("dtype", None)
        checks.assert_not_none(dtype.fields)

        if config is None:
            config = {}

        def _prepare_attr_name(attr_name: str) -> str:
            checks.assert_instance_of(attr_name, str)
            attr_name = attr_name.replace("NaN", "Nan")
            startswith_ = attr_name.startswith("_")
            new_attr_name = ""
            for i in range(len(attr_name)):
                if attr_name[i].isupper():
                    if i > 0 and attr_name[i - 1].islower():
                        new_attr_name += "_"
                new_attr_name += attr_name[i]
            attr_name = new_attr_name
            if not startswith_ and attr_name.startswith("_"):
                attr_name = attr_name[1:]
            attr_name = attr_name.lower()
            if keyword.iskeyword(attr_name):
                attr_name += "_"
            return attr_name.replace("__", "_")

        def _check_attr_name(attr_name, _on_conflict: str = on_conflict) -> None:
            if attr_name not in cls.field_config.get("settings", {}):
                # Consider only attributes that are not listed in the field config
                if hasattr(cls, attr_name):
                    if _on_conflict.lower() == "raise":
                        raise ValueError(f"An attribute with the name '{attr_name}' already exists in {cls}")
                    if _on_conflict.lower() == "ignore":
                        return
                    if _on_conflict.lower() == "override":
                        return
                    raise ValueError(f"Value '{_on_conflict}' is invalid for on_conflict")
                if keyword.iskeyword(attr_name):
                    raise ValueError(f"Name '{attr_name}' is a keyword and cannot be used as an attribute name")

        if dtype is not None:
            for field_name in dtype.names:
                settings = config.get(field_name, {})
                attach = settings.get("attach", True)
                if not isinstance(attach, bool):
                    target_name = attach
                    attach = True
                else:
                    target_name = field_name
                defaults = settings.get("defaults", None)
                if defaults is None:
                    defaults = {}
                attach_filters = settings.get("attach_filters", False)
                filter_defaults = settings.get("filter_defaults", None)
                if filter_defaults is None:
                    filter_defaults = {}
                _on_conflict = settings.get("on_conflict", on_conflict)

                if attach:
                    target_name = _prepare_attr_name(target_name)
                    _check_attr_name(target_name, _on_conflict)

                    def new_prop(
                        self,
                        _field_name: str = field_name,
                        _defaults: tp.KwargsLike = defaults,
                    ) -> MappedArray:
                        return self.get_map_field(_field_name, **_defaults)

                    new_prop.__doc__ = f"Mapped array of the field `{field_name}`."
                    new_prop.__name__ = target_name
                    setattr(cls, target_name, cached_property(new_prop))

                if attach_filters:
                    if isinstance(attach_filters, bool):
                        if not attach_filters:
                            continue
                        mapping = cls.field_config.get("settings", {}).get(field_name, {}).get("mapping", None)
                    else:
                        mapping = attach_filters
                    if mapping is None:
                        raise ValueError(f"Field '{field_name}': Mapping is required to attach filters")
                    mapping = to_value_mapping(mapping)

                    for filter_value, target_filter_name in mapping.items():
                        if target_filter_name is None:
                            continue
                        if isinstance(attach_filters, bool):
                            target_filter_name = field_name + "_" + target_filter_name
                        target_filter_name = _prepare_attr_name(target_filter_name)
                        _check_attr_name(target_filter_name, _on_conflict)
                        if target_filter_name in filter_defaults:
                            __filter_defaults = filter_defaults[target_filter_name]
                        else:
                            __filter_defaults = filter_defaults

                        def new_filter_prop(
                            self,
                            _field_name: str = field_name,
                            _filter_value: tp.Any = filter_value,
                            _filter_defaults: tp.KwargsLike = __filter_defaults,
                        ) -> MappedArray:
                            filter_mask = self.get_field_arr(_field_name) == _filter_value
                            return self.apply_mask(filter_mask, **_filter_defaults)

                        new_filter_prop.__doc__ = f"Records filtered by `{field_name} == {filter_value}`."
                        new_filter_prop.__name__ = target_filter_name
                        setattr(cls, target_filter_name, cached_property(new_filter_prop))

        return cls

    if len(args) == 0:
        return wrapper
    elif len(args) == 1:
        if isinstance(args[0], type):
            return wrapper(args[0])
        return partial(wrapper, config=args[0])
    elif len(args) == 2:
        return wrapper(args[0], config=args[1])
    raise ValueError("Either class, config, class and config, or keyword arguments must be passed")


def attach_shortcut_properties(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach shortcut properties.

    `config` must contain target property names (keys) and settings (values) with the following keys:

    * `method_name`: Name of the source method. Defaults to the target name prepended with the prefix `get_`.
    * `obj_type`: Type of the returned object. Can be 'array' for 2-dim arrays, 'red_array' for 1-dim arrays,
        'records' for record arrays, and 'mapped_array' for mapped arrays. Defaults to 'records'.
    * `group_aware`: Whether the returned object is aligned based on the current grouping.
        Defaults to True.
    * `method_kwargs`: Keyword arguments passed to the source method. Defaults to None.
    * `decorator`: Defaults to `vectorbtpro.utils.decorators.cached_property` for object types
        'records' and 'red_array'. Otherwise, to `vectorbtpro.utils.decorators.cacheable_property`.
    * `decorator_kwargs`: Keyword arguments passed to the decorator. By default,
        includes options `obj_type` and `group_aware`.
    * `docstring`: Method docstring.

    The class must be a subclass of `vectorbtpro.records.base.Records`."""

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")

        for target_name, settings in config.items():
            if target_name.startswith("get_"):
                raise ValueError(f"Property names cannot have prefix 'get_' ('{target_name}')")
            method_name = settings.get("method_name", "get_" + target_name)
            obj_type = settings.get("obj_type", "records")
            group_by_aware = settings.get("group_by_aware", True)
            method_kwargs = settings.get("method_kwargs", None)
            method_kwargs = resolve_dict(method_kwargs)
            decorator = settings.get("decorator", None)
            if decorator is None:
                if obj_type in ("red_array", "records"):
                    decorator = cached_property
                else:
                    decorator = cacheable_property
            decorator_kwargs = merge_dicts(
                dict(obj_type=obj_type, group_by_aware=group_by_aware),
                settings.get("decorator_kwargs", None),
            )
            docstring = settings.get("docstring", None)
            if docstring is None:
                if len(method_kwargs) == 0:
                    docstring = f"`{cls.__name__}.{method_name}` with default arguments."
                else:
                    docstring = f"`{cls.__name__}.{method_name}` with arguments `{method_kwargs}`."

            def new_prop(self, _method_name: str = method_name, _method_kwargs: tp.Kwargs = method_kwargs) -> tp.Any:
                return getattr(self, _method_name)(**_method_kwargs)

            new_prop.__name__ = target_name
            new_prop.__qualname__ = f"{cls.__name__}.{target_name}"
            new_prop.__doc__ = docstring
            setattr(cls, new_prop.__name__, decorator(new_prop, **decorator_kwargs))
        return cls

    return wrapper
