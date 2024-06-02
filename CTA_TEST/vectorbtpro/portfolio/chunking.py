# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Extensions for chunking of portfolio."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio.enums import SimulationOutput
from vectorbtpro.records.chunking import merge_records
from vectorbtpro.utils.chunking import ChunkMeta, ArraySlicer
from vectorbtpro.utils.config import ReadonlyConfig
from vectorbtpro.utils.template import Rep

__all__ = []


def get_init_cash_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Get slicer for `init_cash` based on cash sharing."""
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer()
    return base_ch.flex_1d_array_gl_slicer


def get_cash_deposits_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Get slicer for `cash_deposits` based on cash sharing."""
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer(axis=1)
    return base_ch.flex_array_gl_slicer


def in_outputs_merge_func(
    results: tp.List[SimulationOutput],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
):
    """Merge chunks of in-output objects.

    Concatenates 1-dim arrays, stacks columns of 2-dim arrays, and fixes and concatenates record arrays
    using `vectorbtpro.records.chunking.merge_records`. Other objects will throw an error."""
    in_outputs = dict()
    for k, v in results[0].in_outputs._asdict().items():
        if v is None:
            in_outputs[k] = None
            continue
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Cannot merge in-output object '{k}' of type {type(v)}")
        if v.ndim == 2:
            in_outputs[k] = np.column_stack([getattr(r.in_outputs, k) for r in results])
        elif v.ndim == 1:
            if v.dtype.fields is None:
                in_outputs[k] = np.concatenate([getattr(r.in_outputs, k) for r in results])
            else:
                records = [getattr(r.in_outputs, k) for r in results]
                in_outputs[k] = merge_records(records, chunk_meta, ann_args=ann_args, mapper=mapper)
        else:
            raise ValueError(f"Cannot merge in-output object '{k}' with number of dimensions {v.ndim}")
    return type(results[0].in_outputs)(**in_outputs)


def merge_sim_outs(
    results: tp.List[SimulationOutput],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
    in_outputs_merge_func: tp.Callable = in_outputs_merge_func,
    **kwargs,
) -> SimulationOutput:
    """Merge chunks of `vectorbtpro.portfolio.enums.SimulationOutput` instances.

    If `SimulationOutput.in_outputs` is not None, must provide `in_outputs_merge_func` or similar."""
    order_records = [r.order_records for r in results]
    order_records = merge_records(order_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    log_records = [r.log_records for r in results]
    log_records = merge_records(log_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    target_shape = ann_args["target_shape"]["value"]
    if results[0].cash_deposits.shape == target_shape:
        cash_deposits = np.column_stack([r.cash_deposits for r in results])
    else:
        cash_deposits = results[0].cash_deposits
    if results[0].cash_earnings.shape == target_shape:
        cash_earnings = np.column_stack([r.cash_earnings for r in results])
    else:
        cash_earnings = results[0].cash_earnings
    if results[0].call_seq is not None:
        call_seq = np.column_stack([r.call_seq for r in results])
    else:
        call_seq = None
    if results[0].in_outputs is not None:
        in_outputs = in_outputs_merge_func(results, chunk_meta, ann_args, mapper, **kwargs)
    else:
        in_outputs = None
    return SimulationOutput(
        order_records=order_records,
        log_records=log_records,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


merge_sim_outs_config = ReadonlyConfig(
    dict(
        merge_func=merge_sim_outs,
        merge_kwargs=dict(chunk_meta=Rep("chunk_meta"), ann_args=Rep("ann_args"), mapper=base_ch.group_lens_mapper),
    )
)
"""Config for merging using `merge_sim_outs`."""
