# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Functions for working with call sequence arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.portfolio.enums import CallSeqType
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


@register_jitted(cache=True)
def shuffle_call_seq_nb(call_seq: tp.Array2d, group_lens: tp.Array1d) -> None:
    """Shuffle the call sequence array."""
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col


@register_jitted(cache=True)
def build_call_seq_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    call_seq_type: int = CallSeqType.Default,
) -> tp.Array2d:
    """Build a new call sequence array."""
    if call_seq_type == CallSeqType.Reversed:
        out = np.full(target_shape[1], 1, dtype=np.int_)
        out[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        out = np.cumsum(out[::-1])[::-1] - 1
        out = out * np.ones((target_shape[0], 1), dtype=np.int_)
        return out
    out = np.full(target_shape[1], 1, dtype=np.int_)
    out[np.cumsum(group_lens)[:-1]] -= group_lens[:-1]
    out = np.cumsum(out) - 1
    out = out * np.ones((target_shape[0], 1), dtype=np.int_)
    if call_seq_type == CallSeqType.Random:
        shuffle_call_seq_nb(out, group_lens)
    return out


def require_call_seq(call_seq: tp.Array2d) -> tp.Array2d:
    """Force the call sequence array to pass our requirements."""
    return np.require(call_seq, dtype=np.int_, requirements=["A", "O", "W", "F"])


def build_call_seq(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    call_seq_type: int = CallSeqType.Default,
) -> tp.Array2d:
    """Not compiled but faster version of `build_call_seq_nb`."""
    call_seq = np.full(target_shape[1], 1, dtype=np.int_)
    if call_seq_type == CallSeqType.Reversed:
        call_seq[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        call_seq = np.cumsum(call_seq[::-1])[::-1] - 1
    else:
        call_seq[np.cumsum(group_lens[:-1])] -= group_lens[:-1]
        call_seq = np.cumsum(call_seq) - 1
    call_seq = np.broadcast_to(call_seq, target_shape)
    if call_seq_type == CallSeqType.Random:
        call_seq = require_call_seq(call_seq)
        shuffle_call_seq_nb(call_seq, group_lens)
    return require_call_seq(call_seq)
