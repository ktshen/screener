# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for caching."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.decorators import class_or_instancemethod

__all__ = [
    "clear_pycache",
    "Cacheable",
]


def delete_dir(p):
    """Delete directory with files."""
    for sub in p.iterdir():
        if sub.is_dir():
            delete_dir(sub)
        else:
            sub.unlink()
    p.rmdir()


def clear_pycache():
    """Clear __pycache__ folders and .pyc files."""
    import pathlib

    for p in pathlib.Path(__file__).parent.parent.rglob("__pycache__"):
        delete_dir(p)
    for p in pathlib.Path(__file__).parent.parent.rglob("*.py[co]"):
        p.unlink()


class Cacheable:
    """Class that contains cacheable properties and methods.

    Required to register `vectorbtpro.utils.decorators.cacheable_property` and
    `vectorbtpro.utils.decorators.cacheable_method`.

    See `vectorbtpro.registries.ca_registry` for details on the caching procedure."""

    def __init__(self) -> None:
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if not caching_cfg["register_lazily"]:
            instance_setup = self.get_ca_setup()
            if instance_setup is not None:
                for unbound_setup in instance_setup.unbound_setups:
                    unbound_setup.cacheable.get_ca_setup(self)

    @class_or_instancemethod
    def get_ca_setup(cls_or_self) -> tp.Union["CAClassSetup", "CAInstanceSetup"]:
        """Get instance setup of type `vectorbtpro.registries.ca_registry.CAInstanceSetup` if the instance method
        was called and class setup of type `vectorbtpro.registries.ca_registry.CAClassSetup` otherwise."""
        from vectorbtpro.registries.ca_registry import CAClassSetup, CAInstanceSetup

        if isinstance(cls_or_self, type):
            return CAClassSetup.get(cls_or_self)
        return CAInstanceSetup.get(cls_or_self)
