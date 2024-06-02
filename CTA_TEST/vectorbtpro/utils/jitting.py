# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for jitting."""

from numba import jit as nb_jit, generated_jit as nb_generated_jit

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts, Configured

__all__ = [
    "jitted",
]


class Jitter(Configured):
    """Abstract class for decorating jitable functions.

    Represents a single configuration for jitting.

    When overriding `Jitter.decorate`, make sure to check whether wrapping is disabled
    globally using `Jitter.wrapping_disabled`."""

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    @property
    def wrapping_disabled(self) -> bool:
        """Whether wrapping is disabled globally."""
        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        return jitting_cfg["disable_wrapping"]

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        """Decorate a jitable function."""
        if self.wrapping_disabled:
            return py_func
        raise NotImplementedError


class NumPyJitter(Jitter):
    """Class for decorating functions that use NumPy.

    Returns the function without decorating."""

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        return py_func


class NumbaJitter(Jitter):
    """Class for decorating functions using Numba.

    !!! note
        If `fix_cannot_parallel` is True, `parallel=True` will be ignored if there is no `can_parallel` tag."""

    def __init__(
        self,
        is_generated_jit: bool = False,
        fix_cannot_parallel: bool = True,
        nopython: bool = True,
        nogil: bool = True,
        parallel: bool = False,
        cache: bool = False,
        boundscheck: bool = False,
        **options,
    ) -> None:
        Jitter.__init__(
            self,
            is_generated_jit=is_generated_jit,
            fix_cannot_parallel=fix_cannot_parallel,
            nopython=nopython,
            nogil=nogil,
            parallel=parallel,
            cache=cache,
            boundscheck=boundscheck,
            **options,
        )

        self._is_generated_jit = is_generated_jit
        self._fix_cannot_parallel = fix_cannot_parallel
        self._nopython = nopython
        self._nogil = nogil
        self._parallel = parallel
        self._cache = cache
        self._boundscheck = boundscheck
        self._options = options

    @property
    def is_generated_jit(self) -> bool:
        """Whether to use `numba.generated_jit`, otherwise `numba.jit`."""
        return self._is_generated_jit

    @property
    def fix_cannot_parallel(self) -> bool:
        """Whether to set `parallel` to False if there is no 'can_parallel' tag."""
        return self._fix_cannot_parallel

    @property
    def options(self) -> tp.Kwargs:
        """Options passed to the Numba decorator."""
        return self._options

    @property
    def nopython(self) -> bool:
        """Whether to run in nopython mode."""
        return self._nopython

    @property
    def nogil(self) -> bool:
        """Whether to release the GIL."""
        return self._nogil

    @property
    def parallel(self) -> bool:
        """Whether to enable automatic parallelization."""
        return self._parallel

    @property
    def boundscheck(self) -> bool:
        """Whether to enable bounds checking for array indices."""
        return self._boundscheck

    @property
    def cache(self) -> bool:
        """Whether to write the result of function compilation into a file-based cache."""
        return self._cache

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        if self.wrapping_disabled:
            return py_func

        if tags is None:
            tags = set()
        if self.is_generated_jit:
            decorator = nb_generated_jit
        else:
            decorator = nb_jit
        options = dict(self.options)
        parallel = self.parallel
        if self.fix_cannot_parallel and parallel and "can_parallel" not in tags:
            parallel = False
        cache = self.cache
        if parallel and cache:
            cache = False
        return decorator(
            nopython=self.nopython,
            nogil=self.nogil,
            parallel=parallel,
            cache=cache,
            boundscheck=self.boundscheck,
            **options,
        )(py_func)


def get_func_suffix(py_func: tp.Callable) -> tp.Optional[str]:
    """Get the suffix of the function."""
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    splitted_name = py_func.__name__.split("_")
    if len(splitted_name) == 1:
        return None
    suffix = splitted_name[-1].lower()
    if suffix not in jitting_cfg["jitters"]:
        return None
    return suffix


def resolve_jitter_type(
    jitter: tp.Optional[tp.JitterLike] = None,
    py_func: tp.Optional[tp.Callable] = None,
) -> tp.Type[Jitter]:
    """Resolve `jitter`.

    * If `jitter` is None and `py_func` is not None, uses `get_func_suffix`
    * If `jitter` is a string, looks in `jitters` in `vectorbtpro._settings.jitting`
    * If `jitter` is a subclass of `Jitter`, returns it
    * If `jitter` is an instance of `Jitter`, returns its class
    * Otherwise, throws an error"""
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    if jitter is None:
        if py_func is None:
            raise ValueError("Could not parse jitter without a function")
        jitter = get_func_suffix(py_func)
        if jitter is None:
            raise ValueError(f"Could not parse jitter from suffix of function {py_func}")

    if isinstance(jitter, str):
        if jitter in jitting_cfg["jitters"]:
            jitter = jitting_cfg["jitters"][jitter]["cls"]
        else:
            found = False
            for k, v in jitting_cfg["jitters"].items():
                if jitter in v.get("aliases", set()):
                    jitter = v["cls"]
                    found = True
                    break
            if not found:
                raise ValueError(f"Jitter with name '{jitter}' not registered")
    if isinstance(jitter, type) and issubclass(jitter, Jitter):
        return jitter
    if isinstance(jitter, Jitter):
        return type(jitter)
    raise TypeError(f"Jitter type {jitter} is not supported")


def get_id_of_jitter_type(jitter_type: tp.Type[Jitter]) -> tp.Optional[tp.Hashable]:
    """Get id of a jitter type using `jitters` in `vectorbtpro._settings.jitting`."""
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    for jitter_id, jitter_cfg in jitting_cfg["jitters"].items():
        if jitter_type is jitter_cfg["cls"]:
            return jitter_id
    return None


def resolve_jitted_option(option: tp.JittedOption = None) -> tp.KwargsLike:
    """Return keyword arguments for `jitted`.

    `option` can be:

    * True: Decorate using default settings
    * False: Do not decorate (returns None)
    * string: Use `option` as the name of the jitter
    * dict: Use `option` as keyword arguments for jitting

    For defaults, see `option` in `vectorbtpro._settings.jitting`."""
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    if option is None:
        option = jitting_cfg["option"]

    if isinstance(option, bool):
        if not option:
            return None
        return dict()
    if isinstance(option, dict):
        return option
    elif isinstance(option, str):
        return dict(jitter=option)
    raise TypeError(f"Type {type(option)} is invalid for a jitting option")


def specialize_jitted_option(option: tp.JittedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve `option` and merge it with `kwargs` if it's not None so the dict can be passed
    as an option to other functions."""
    jitted_kwargs = resolve_jitted_option(option)
    if jitted_kwargs is None:
        return None
    return merge_dicts(kwargs, jitted_kwargs)


def resolve_jitted_kwargs(option: tp.JittedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve keyword arguments for `jitted`.

    Resolves `option` using `resolve_jitted_option`.

    !!! note
        Keys in `option` have more priority than in `kwargs`."""
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    jitted_kwargs = resolve_jitted_option(option=option)
    if jitted_kwargs is None:
        return None
    if isinstance(jitting_cfg["option"], dict):
        jitted_kwargs = merge_dicts(jitting_cfg["option"], kwargs, jitted_kwargs)
    else:
        jitted_kwargs = merge_dicts(kwargs, jitted_kwargs)
    return jitted_kwargs


def resolve_jitter(
    jitter: tp.Optional[tp.JitterLike] = None,
    py_func: tp.Optional[tp.Callable] = None,
    **jitter_kwargs,
) -> Jitter:
    """Resolve jitter.

    !!! note
        If `jitter` is already an instance of `Jitter` and there are other keyword arguments, discards them."""
    if not isinstance(jitter, Jitter):
        jitter_type = resolve_jitter_type(jitter=jitter, py_func=py_func)
        jitter = jitter_type(**jitter_kwargs)
    return jitter


def jitted(*args, tags: tp.Optional[set] = None, **jitted_kwargs) -> tp.Callable:
    """Decorate a jitable function.

    Resolves `jitter` using `resolve_jitter`.

    The wrapping mechanism can be disabled by using the global setting `disable_wrapping`
    (=> returns the wrapped function).

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt

        >>> @vbt.jitted
        ... def my_func_nb():
        ...     total = 0
        ...     for i in range(1000000):
        ...         total += 1
        ...     return total

        >>> %timeit my_func_nb()
        68.1 ns ± 0.32 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        ```

        Jitter is automatically detected using the suffix of the wrapped function.
    """

    def decorator(py_func: tp.Callable) -> tp.Callable:
        jitter = resolve_jitter(py_func=py_func, **jitted_kwargs)
        return jitter.decorate(py_func, tags=tags)

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
