# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Global registry for chunkables."""

import attr

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.chunking import chunked, resolve_chunked, resolve_chunked_option
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.hashing import Hashable
from vectorbtpro.utils.template import RepEval

__all__ = [
    "ch_reg",
    "register_chunkable",
]


@attr.s(frozen=True, eq=False)
class ChunkedSetup(Hashable):
    """Class that represents a chunkable setup.

    !!! note
        Hashed solely by `setup_id`."""

    setup_id: tp.Hashable = attr.ib()
    """Setup id."""

    func: tp.Callable = attr.ib()
    """Chunkable function."""

    options: tp.DictLike = attr.ib(default=None)
    """Options dictionary."""

    tags: tp.SetLike = attr.ib(default=None)
    """Set of tags."""

    @staticmethod
    def get_hash(setup_id: tp.Hashable) -> int:
        return hash((setup_id,))

    @property
    def hash_key(self) -> tuple:
        return (self.setup_id,)


class ChunkableRegistry:
    """Class that registers chunkable functions."""

    def __init__(self) -> None:
        self._setups = {}

    @property
    def setups(self) -> tp.Dict[tp.Hashable, ChunkedSetup]:
        """Dict of registered `ChunkedSetup` instances by `ChunkedSetup.setup_id`."""
        return self._setups

    def register(
        self,
        func: tp.Callable,
        setup_id: tp.Optional[tp.Hashable] = None,
        options: tp.DictLike = None,
        tags: tp.SetLike = None,
    ) -> None:
        """Register a new setup."""
        if setup_id is None:
            setup_id = func.__module__ + "." + func.__name__
        setup = ChunkedSetup(setup_id=setup_id, func=func, options=options, tags=tags)
        self.setups[setup_id] = setup

    def match_setups(self, expression: tp.Optional[str] = None, context: tp.KwargsLike = None) -> tp.Set[ChunkedSetup]:
        """Match setups against an expression with each setup being a context."""
        matched_setups = set()
        for setup in self.setups.values():
            if expression is None:
                result = True
            else:
                result = RepEval(expression).substitute(context=merge_dicts(attr.asdict(setup), context))
                checks.assert_instance_of(result, bool)

            if result:
                matched_setups.add(setup)
        return matched_setups

    def get_setup(self, setup_id_or_func: tp.Union[tp.Hashable, tp.Callable]) -> tp.Optional[ChunkedSetup]:
        """Get setup by its id or function.

        `setup_id_or_func` can be an identifier or a function.
        If it's a function, will build the identifier using its module and name."""
        if hasattr(setup_id_or_func, "py_func"):
            nb_setup_id = setup_id_or_func.__module__ + "." + setup_id_or_func.__name__
            if nb_setup_id in self.setups:
                setup_id = nb_setup_id
            else:
                setup_id = setup_id_or_func.py_func.__module__ + "." + setup_id_or_func.py_func.__name__
        elif callable(setup_id_or_func):
            setup_id = setup_id_or_func.__module__ + "." + setup_id_or_func.__name__
        else:
            setup_id = setup_id_or_func
        if setup_id not in self.setups:
            return None
        return self.setups[setup_id]

    def decorate(
        self,
        setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        target_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Callable:
        """Decorate the setup's function using the `vectorbtpro.utils.chunking.chunked` decorator.

        Finds setup using `ChunkableRegistry.get_setup`.

        Merges setup's options with `options`.

        Specify `target_func` to apply the found setup on another function."""
        setup = self.get_setup(setup_id_or_func)
        if setup is None:
            raise KeyError(f"Setup for {setup_id_or_func} not registered")

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return chunked(func, **merge_dicts(setup.options, kwargs))

    def resolve_option(
        self,
        setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        option: tp.ChunkedOption,
        target_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Callable:
        """Same as `ChunkableRegistry.decorate` but using `vectorbtpro.utils.chunking.resolve_chunked`."""
        setup = self.get_setup(setup_id_or_func)
        if setup is None:
            if callable(setup_id_or_func):
                option = resolve_chunked_option(option=option)
                if option is None:
                    return setup_id_or_func
            raise KeyError(f"Setup for {setup_id_or_func} not registered")

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return resolve_chunked(func, option=option, **merge_dicts(setup.options, kwargs))


ch_reg = ChunkableRegistry()
"""Default registry of type `ChunkableRegistry`."""


def register_chunkable(
    func: tp.Optional[tp.Callable] = None,
    setup_id: tp.Optional[tp.Hashable] = None,
    registry: ChunkableRegistry = ch_reg,
    tags: tp.SetLike = None,
    return_wrapped: bool = False,
    **options,
) -> tp.Callable:
    """Register a new chunkable function.

    If `return_wrapped` is True, wraps with the `vectorbtpro.utils.chunking.chunked` decorator.
    Otherwise, leaves the function as-is (preferred).

    Options are merged in the following order:

    * `options` in `vectorbtpro._settings.chunking`
    * `setup_options.{setup_id}` in `vectorbtpro._settings.chunking`
    * `options`
    * `override_options` in `vectorbtpro._settings.chunking`
    * `override_setup_options.{setup_id}` in `vectorbtpro._settings.chunking`

    !!! note
        Calling the `register_chunkable` decorator before (or below) the `vectorbtpro.registries.jit_registry.register_jitted`
        decorator with `return_wrapped` set to True won't work. Doing the same after (or above)
        `vectorbtpro.registries.jit_registry.register_jitted` will work for calling the function from Python but not from Numba.
        Generally, avoid wrapping right away and use `ChunkableRegistry.decorate` to perform decoration."""

    def decorator(_func: tp.Callable) -> tp.Callable:
        nonlocal setup_id, options

        from vectorbtpro._settings import settings

        chunking_cfg = settings["chunking"]

        if setup_id is None:
            setup_id = _func.__module__ + "." + _func.__name__
        options = merge_dicts(
            chunking_cfg.get("options", None),
            chunking_cfg.get("setup_options", {}).get(setup_id, None),
            options,
            chunking_cfg.get("override_options", None),
            chunking_cfg.get("override_setup_options", {}).get(setup_id, None),
        )

        registry.register(func=_func, setup_id=setup_id, options=options, tags=tags)
        if return_wrapped:
            return chunked(_func, **options)
        return _func

    if func is None:
        return decorator
    return decorator(func)
