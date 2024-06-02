# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Extensions for chunking of base operations."""

import uuid

import attr
import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.chunking import (
    ArgGetter,
    ArgSizer,
    ChunkMeta,
    ChunkMapper,
    ChunkSlicer,
    ShapeSlicer,
    ArraySelector,
    ArraySlicer,
)
from vectorbtpro.utils.parsing import Regex

__all__ = [
    "GroupLensSizer",
    "GroupLensSlicer",
    "GroupLensMapper",
    "GroupMapSlicer",
    "GroupIdxsMapper",
    "FlexArraySelector",
    "FlexArraySlicer",
    "shape_gl_slicer",
    "flex_1d_array_gl_slicer",
    "flex_array_gl_slicer",
    "array_gl_slicer",
]


class GroupLensSizer(ArgSizer):
    """Class for getting the size from group lengths.

    Argument can be either a group map tuple or a group lengths array."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if isinstance(arg, tuple):
            return len(arg[1])
        return len(arg)


class GroupLensSlicer(ChunkSlicer):
    """Class for slicing multiple elements from group lengths based on the chunk range."""

    def take(self, obj: tp.Union[tp.GroupLens, tp.GroupMap], chunk_meta: ChunkMeta, **kwargs) -> tp.GroupMap:
        if isinstance(obj, tuple):
            return obj[1][chunk_meta.start : chunk_meta.end]
        return obj[chunk_meta.start : chunk_meta.end]


def get_group_lens_slice(group_lens: tp.Array1d, chunk_meta: ChunkMeta) -> slice:
    """Get slice of each chunk in group lengths."""
    group_lens_cumsum = np.cumsum(group_lens[: chunk_meta.end])
    start = group_lens_cumsum[chunk_meta.start] - group_lens[chunk_meta.start]
    end = group_lens_cumsum[-1]
    return slice(start, end)


@attr.s(frozen=True)
class GroupLensMapper(ChunkMapper, ArgGetter):
    """Class for mapping chunk metadata to per-group column lengths.

    Argument can be either a group map tuple or a group lengths array."""

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        group_lens = self.get_arg(ann_args)
        if isinstance(group_lens, tuple):
            group_lens = group_lens[1]
        group_lens_slice = get_group_lens_slice(group_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=group_lens_slice.start,
            end=group_lens_slice.stop,
            indices=None,
        )


group_lens_mapper = GroupLensMapper(arg_query=Regex(r"(group_lens|group_map)"))
"""Default instance of `GroupLensMapper`."""


class GroupMapSlicer(ChunkSlicer):
    """Class for slicing multiple elements from a group map based on the chunk range."""

    def take(self, obj: tp.GroupMap, chunk_meta: ChunkMeta, **kwargs) -> tp.GroupMap:
        group_idxs, group_lens = obj
        group_lens = group_lens[chunk_meta.start : chunk_meta.end]
        return np.arange(np.sum(group_lens)), group_lens


@attr.s(frozen=True)
class GroupIdxsMapper(ChunkMapper, ArgGetter):
    """Class for mapping chunk metadata to per-group column indices.

    Argument must be a group map tuple."""

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        group_map = self.get_arg(ann_args)
        group_idxs, group_lens = group_map
        group_lens_slice = get_group_lens_slice(group_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=None,
            end=None,
            indices=group_idxs[group_lens_slice],
        )


group_idxs_mapper = GroupIdxsMapper(arg_query="group_map")
"""Default instance of `GroupIdxsMapper`."""


@attr.s(frozen=True)
class FlexArraySelector(ArraySelector):
    """Class for selecting one element from a NumPy array's axis flexibly based on the chunk index.

    The result is intended to be used together with `vectorbtpro.base.flex_indexing.flex_select_1d_nb`
    and `vectorbtpro.base.flex_indexing.flex_select_nb`."""

    def take(
        self,
        obj: tp.ArrayLike,
        chunk_meta: ChunkMeta,
        ann_args: tp.Optional[tp.AnnArgs] = None,
        **kwargs,
    ) -> tp.ArrayLike:
        if np.isscalar(obj):
            return obj
        obj = np.asarray(obj)
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if obj.ndim == 1:
            if obj.shape[0] == 1:
                return obj
            if self.keep_dims:
                return obj[chunk_meta.idx : chunk_meta.idx + 1]
            return obj[chunk_meta.idx]
        if obj.ndim == 2:
            if axis == 1:
                if obj.shape[1] == 1:
                    return obj
                if self.keep_dims:
                    return obj[: chunk_meta.idx : chunk_meta.idx + 1]
                return obj[: chunk_meta.idx]
            if obj.shape[0] == 1:
                return obj
            if self.keep_dims:
                return obj[chunk_meta.idx : chunk_meta.idx + 1, :]
            return obj[chunk_meta.idx, :]
        raise ValueError(f"FlexArraySelector supports max 2 dimensions, not {obj.ndim}")


@attr.s(frozen=True)
class FlexArraySlicer(ArraySlicer):
    """Class for selecting one element from a NumPy array's axis flexibly based on the chunk index.

    The result is intended to be used together with `vectorbtpro.base.flex_indexing.flex_select_1d_nb`
    and `vectorbtpro.base.flex_indexing.flex_select_nb`."""

    def take(
        self,
        obj: tp.ArrayLike,
        chunk_meta: ChunkMeta,
        ann_args: tp.Optional[tp.AnnArgs] = None,
        **kwargs,
    ) -> tp.ArrayLike:
        if np.isscalar(obj):
            return obj
        obj = np.asarray(obj)
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if obj.ndim == 1:
            if obj.shape[0] == 1:
                return obj
            return obj[chunk_meta.start : chunk_meta.end]
        if obj.ndim == 2:
            if axis == 1:
                if obj.shape[1] == 1:
                    return obj
                return obj[:, chunk_meta.start : chunk_meta.end]
            if obj.shape[0] == 1:
                return obj
            return obj[chunk_meta.start : chunk_meta.end, :]
        raise ValueError(f"FlexArraySlicer supports max 2 dimensions, not {obj.ndim}")


shape_gl_slicer = ShapeSlicer(axis=1, mapper=group_lens_mapper)
"""Flexible 2-dim shape slicer along the column axis based on group lengths."""

flex_1d_array_gl_slicer = FlexArraySlicer(mapper=group_lens_mapper)
"""Flexible 1-dim array slicer along the column axis based on group lengths."""

flex_array_gl_slicer = FlexArraySlicer(axis=1, mapper=group_lens_mapper)
"""Flexible 2-dim array slicer along the column axis based on group lengths."""

array_gl_slicer = ArraySlicer(axis=1, mapper=group_lens_mapper)
"""2-dim array slicer along the column axis based on group lengths."""
