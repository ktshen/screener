# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for hashing."""

from functools import cached_property as cachedproperty

from vectorbtpro import _typing as tp

__all__ = []


class Hashable:
    """Hashable class."""

    @staticmethod
    def get_hash(*args, **kwargs) -> int:
        """Static method to get the hash of the instance based on its arguments."""
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        """Key that can be used for hashing the instance."""
        raise NotImplementedError

    @cachedproperty
    def hash(self) -> int:
        """Hash of the instance."""
        return hash(self.hash_key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, type(self)):
            return self.hash_key == other.hash_key
        raise NotImplementedError
