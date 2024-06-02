# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class decorators for generic accessors."""

import inspect

from vectorbtpro import _typing as tp
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Config
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = []


def attach_nb_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach Numba methods.

    `config` must contain target method names (keys) and dictionaries (values) with the following keys:

    * `func`: Function that must be wrapped. The first argument must expect a 2-dim array.
    * `is_reducing`: Whether the function is reducing. Defaults to False.
    * `disable_jitted`: Whether to disable the `jitted` option.
    * `disable_chunked`: Whether to disable the `chunked` option.
    * `replace_signature`: Whether to replace the target signature with the source signature. Defaults to True.
    * `wrap_kwargs`: Default keyword arguments for wrapping. Will be merged with the dict supplied by the user.
        Defaults to `dict(name_or_index=target_name)` for reducing functions.

    The class must be a subclass of `vectorbtpro.base.wrapping.Wrapping`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        from vectorbtpro.base.wrapping import Wrapping

        checks.assert_subclass_of(cls, Wrapping)

        for target_name, settings in config.items():
            func = settings["func"]
            is_reducing = settings.get("is_reducing", False)
            disable_jitted = settings.get("disable_jitted", False)
            disable_chunked = settings.get("disable_chunked", False)
            replace_signature = settings.get("replace_signature", True)
            default_wrap_kwargs = settings.get("wrap_kwargs", dict(name_or_index=target_name) if is_reducing else None)

            def new_method(
                self,
                *args,
                _target_name: str = target_name,
                _func: tp.Callable = func,
                _is_reducing: bool = is_reducing,
                _disable_jitted: bool = disable_jitted,
                _disable_chunked: bool = disable_chunked,
                _default_wrap_kwargs: tp.KwargsLike = default_wrap_kwargs,
                jitted: tp.JittedOption = None,
                chunked: tp.ChunkedOption = None,
                wrap_kwargs: tp.KwargsLike = None,
                **kwargs,
            ) -> tp.SeriesFrame:
                args = (self.to_2d_array(),) + args
                inspect.signature(_func).bind(*args, **kwargs)

                if not _disable_jitted:
                    _func = jit_reg.resolve_option(_func, jitted)
                elif jitted is not None:
                    raise ValueError("This method doesn't support jitting")
                if not _disable_chunked:
                    _func = ch_reg.resolve_option(_func, chunked)
                elif chunked is not None:
                    raise ValueError("This method doesn't support chunking")
                a = _func(*args, **kwargs)
                wrap_kwargs = merge_dicts(_default_wrap_kwargs, wrap_kwargs)
                if _is_reducing:
                    return self.wrapper.wrap_reduced(a, **wrap_kwargs)
                return self.wrapper.wrap(a, **wrap_kwargs)

            if replace_signature:
                # Replace the function's signature with the original one
                source_sig = inspect.signature(func)
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                self_arg = new_method_params[0]
                jitted_arg = new_method_params[-4]
                chunked_arg = new_method_params[-3]
                wrap_kwargs_arg = new_method_params[-2]
                new_parameters = (self_arg,) + tuple(source_sig.parameters.values())[1:]
                if not disable_jitted:
                    new_parameters += (jitted_arg,)
                if not disable_chunked:
                    new_parameters += (chunked_arg,)
                new_parameters += (wrap_kwargs_arg,)
                new_method.__signature__ = source_sig.replace(parameters=new_parameters)

            new_method.__doc__ = f"See `{func.__module__ + '.' + func.__name__}`."
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper


def attach_transform_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to add transformation methods.

    `config` must contain target method names (keys) and dictionaries (values) with the following keys:

    * `transformer`: Transformer class/object.
    * `docstring`: Method docstring.
    * `replace_signature`: Whether to replace the target signature. Defaults to True.

    The class must be a subclass of `vectorbtpro.generic.accessors.GenericAccessor`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        from vectorbtpro.generic.accessors import TransformerT

        checks.assert_subclass_of(cls, "GenericAccessor")

        for target_name, settings in config.items():
            transformer = settings["transformer"]
            docstring = settings.get("docstring", f"See `{transformer.__name__}`.")
            replace_signature = settings.get("replace_signature", True)

            def new_method(
                self,
                _target_name: str = target_name,
                _transformer: tp.Union[tp.Type[TransformerT], TransformerT] = transformer,
                **kwargs,
            ) -> tp.SeriesFrame:
                if inspect.isclass(_transformer):
                    arg_names = get_func_arg_names(_transformer.__init__)
                    transformer_kwargs = dict()
                    for arg_name in arg_names:
                        if arg_name in kwargs:
                            transformer_kwargs[arg_name] = kwargs.pop(arg_name)
                    return self.transform(_transformer(**transformer_kwargs), **kwargs)
                return self.transform(_transformer, **kwargs)

            if replace_signature:
                source_sig = inspect.signature(transformer.__init__)
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                if inspect.isclass(transformer):
                    transformer_params = tuple(source_sig.parameters.values())
                    source_sig = inspect.Signature(
                        (new_method_params[0],) + transformer_params[1:] + (new_method_params[-1],),
                    )
                    new_method.__signature__ = source_sig
                else:
                    source_sig = inspect.Signature((new_method_params[0],) + (new_method_params[-1],))
                    new_method.__signature__ = source_sig

            new_method.__doc__ = docstring
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper
