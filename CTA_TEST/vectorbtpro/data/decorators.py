# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class decorators for data."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import copy_dict

__all__ = []


def attach_symbol_dict_methods(target_names: tp.Iterable[str]) -> tp.ClassWrapper:
    """Class decorator to attach methods for updating symbol dictionaries."""

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Data")

        for target_name in target_names:
            def new_method(self, _target_name=target_name, **kwargs):
                new_kwargs = copy_dict(getattr(self, _target_name))
                for s in self.symbols:
                    if s not in new_kwargs:
                        new_kwargs[s] = dict()
                for k, v in kwargs.items():
                    from vectorbtpro.data.base import symbol_dict

                    if isinstance(v, symbol_dict):
                        for s, _v in v.items():
                            new_kwargs[s][k] = _v
                    else:
                        for s in new_kwargs:
                            new_kwargs[s][k] = v
                return self.replace(**{_target_name: new_kwargs})

            new_method.__name__ = "update_" + target_name
            new_method.__qualname__ = f"{cls.__name__}.get_{target_name}"
            new_method.__doc__ = f"""Update `Data.{target_name}`. Returns a new instance."""
            setattr(cls, new_method.__name__, new_method)
        return cls

    return wrapper
