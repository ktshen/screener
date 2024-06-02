# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class decorators for attaching magic methods."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Config, ReadonlyConfig

__all__ = []

__pdoc__ = {}

binary_magic_config = ReadonlyConfig(
    {
        "__eq__": dict(func=np.equal),
        "__ne__": dict(func=np.not_equal),
        "__lt__": dict(func=np.less),
        "__gt__": dict(func=np.greater),
        "__le__": dict(func=np.less_equal),
        "__ge__": dict(func=np.greater_equal),
        # arithmetic ops
        "__add__": dict(func=np.add),
        "__sub__": dict(func=np.subtract),
        "__mul__": dict(func=np.multiply),
        "__pow__": dict(func=np.power),
        "__mod__": dict(func=np.mod),
        "__floordiv__": dict(func=np.floor_divide),
        "__truediv__": dict(func=np.true_divide),
        "__radd__": dict(func=lambda x, y: np.add(y, x)),
        "__rsub__": dict(func=lambda x, y: np.subtract(y, x)),
        "__rmul__": dict(func=lambda x, y: np.multiply(y, x)),
        "__rpow__": dict(func=lambda x, y: np.power(y, x)),
        "__rmod__": dict(func=lambda x, y: np.mod(y, x)),
        "__rfloordiv__": dict(func=lambda x, y: np.floor_divide(y, x)),
        "__rtruediv__": dict(func=lambda x, y: np.true_divide(y, x)),
        # mask ops
        "__and__": dict(func=np.bitwise_and),
        "__or__": dict(func=np.bitwise_or),
        "__xor__": dict(func=np.bitwise_xor),
        "__rand__": dict(func=lambda x, y: np.bitwise_and(y, x)),
        "__ror__": dict(func=lambda x, y: np.bitwise_or(y, x)),
        "__rxor__": dict(func=lambda x, y: np.bitwise_xor(y, x)),
    }
)
"""_"""

__pdoc__[
    "binary_magic_config"
] = f"""Config of binary magic methods to be attached to a class.

```python
{binary_magic_config.prettify()}
```
"""

BinaryTranslateFuncT = tp.Callable[[tp.Any, tp.Any, tp.Callable], tp.Any]


def attach_binary_magic_methods(
    translate_func: BinaryTranslateFuncT,
    config: tp.Optional[Config] = None,
) -> tp.ClassWrapper:
    """Class decorator to attach binary magic methods to a class.

    `translate_func` must

    * take `self`, `other`, and unary function,
    * perform computation, and
    * return the result.

    `config` defaults to `binary_magic_config` and must contain target method names (keys)
    and dictionaries (values) with the following keys:

    * `func`: Function that combines two array-like objects.
    """
    if config is None:
        config = binary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings["func"]

            def new_method(
                self,
                other: tp.Any,
                _translate_func: BinaryTranslateFuncT = translate_func,
                _func: tp.Callable = func,
            ) -> tp.SeriesFrame:
                return _translate_func(self, other, _func)

            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper


unary_magic_config = ReadonlyConfig(
    {
        "__neg__": dict(func=np.negative),
        "__pos__": dict(func=np.positive),
        "__abs__": dict(func=np.absolute),
        "__invert__": dict(func=np.invert),
    }
)
"""_"""

__pdoc__[
    "unary_magic_config"
] = f"""Config of unary magic methods to be attached to a class.

```python
{unary_magic_config.prettify()}
```
"""

UnaryTranslateFuncT = tp.Callable[[tp.Any, tp.Callable], tp.Any]


def attach_unary_magic_methods(
    translate_func: UnaryTranslateFuncT,
    config: tp.Optional[Config] = None,
) -> tp.ClassWrapper:
    """Class decorator to attach unary magic methods to a class.

    `translate_func` must

    * take `self` and unary function,
    * perform computation, and
    * return the result.

    `config` defaults to `unary_magic_config` and must contain target method names (keys)
    and dictionaries (values) with the following keys:

    * `func`: Function that transforms one array-like object.
    """
    if config is None:
        config = unary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings["func"]

            def new_method(
                self,
                _translate_func: UnaryTranslateFuncT = translate_func,
                _func: tp.Callable = func,
            ) -> tp.SeriesFrame:
                return _translate_func(self, _func)

            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper
