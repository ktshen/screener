# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Extensions for chunking records and mapped arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.chunking import GroupLensMapper, GroupIdxsMapper
from vectorbtpro.utils.chunking import ChunkMeta, ChunkMapper
from vectorbtpro.utils.parsing import Regex

__all__ = []

col_lens_mapper = GroupLensMapper(arg_query=Regex(r"(col_lens|col_map)"))
"""Default instance of `vectorbtpro.base.chunking.GroupLensMapper` for per-column lengths."""

col_idxs_mapper = GroupIdxsMapper(arg_query="col_map")
"""Default instance of `vectorbtpro.base.chunking.GroupIdxsMapper` for per-column indices."""


def fix_field_in_records(
    record_arrays: tp.List[tp.RecordArray],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.Optional[tp.AnnArgs] = None,
    mapper: tp.Optional[ChunkMapper] = None,
    field: str = "col",
) -> None:
    """Fix a field of the record array in each chunk."""
    for _chunk_meta in chunk_meta:
        if mapper is None:
            record_arrays[_chunk_meta.idx][field] += _chunk_meta.start
        else:
            _chunk_meta_mapped = mapper.map(_chunk_meta, ann_args=ann_args)
            record_arrays[_chunk_meta.idx][field] += _chunk_meta_mapped.start


def merge_records(
    results: tp.List[tp.RecordArray],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.Optional[tp.AnnArgs] = None,
    mapper: tp.Optional[ChunkMapper] = None,
) -> tp.RecordArray:
    """Merge chunks of record arrays.

    Mapper is only applied on the column field."""
    if "col" in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, ann_args=ann_args, mapper=mapper, field="col")
    if "group" in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, field="group")
    return np.concatenate(results)
