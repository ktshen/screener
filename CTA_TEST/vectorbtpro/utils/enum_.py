# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Enum utilities.

In vectorbt, enums are represented by instances of named tuples to be easily used in Numba.
Their values start with 0, while -1 means there is no value."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.mapping import to_value_mapping, apply_mapping

__all__ = [
    "map_enum_fields",
]


def map_enum_fields(
    field: tp.Any,
    enum: tp.Union[tp.NamedTuple, tp.EnumMeta],
    enum_unkval: tp.Any = -1,
    ignore_type=int,
    **kwargs
) -> tp.Any:
    """Map fields to values.

    See `vectorbtpro.utils.mapping.apply_mapping`."""
    mapping = to_value_mapping(enum, reverse=True, enum_unkval=enum_unkval)

    return apply_mapping(field, mapping, ignore_type=ignore_type, **kwargs)


def map_enum_values(
    value: tp.Any,
    enum: tp.Union[tp.NamedTuple, tp.EnumMeta],
    enum_unkval: tp.Any = -1,
    ignore_type=str,
    **kwargs
) -> tp.Any:
    """Map values to fields.

    See `vectorbtpro.utils.mapping.apply_mapping`."""
    mapping = to_value_mapping(enum, reverse=False, enum_unkval=enum_unkval)

    return apply_mapping(value, mapping, ignore_type=ignore_type, **kwargs)
