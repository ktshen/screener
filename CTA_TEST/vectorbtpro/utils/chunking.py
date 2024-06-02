# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for chunking."""

import inspect
import multiprocessing
import uuid
import warnings
from functools import wraps

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Config
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.parsing import annotate_args, match_ann_arg, get_func_arg_names, Regex
from vectorbtpro.utils.template import substitute_templates, Rep

__all__ = [
    "ChunkMeta",
    "ArgChunkMeta",
    "LenChunkMeta",
    "ArgSizer",
    "LenSizer",
    "ShapeSizer",
    "ArraySizer",
    "ChunkMapper",
    "ChunkSelector",
    "ChunkSlicer",
    "CountAdapter",
    "ShapeSelector",
    "ShapeSlicer",
    "ArraySelector",
    "ArraySlicer",
    "SequenceTaker",
    "MappingTaker",
    "ArgsTaker",
    "KwargsTaker",
    "chunked",
]

__pdoc__ = {}


# ############# Universal ############# #


def _assert_value_not_none(instance: object, attribute: attr.Attribute, value: tp.Any) -> None:
    """Assert that value is not None."""
    if value is None:
        raise ValueError("Please provide {}".format(attribute.name))


@attr.s(frozen=True)
class ArgGetter:
    """Class for getting an argument from annotated arguments."""

    arg_query: tp.Optional[tp.AnnArgQuery] = attr.ib(default=None, validator=_assert_value_not_none)
    """Query for annotated argument to derive the size from."""

    def get_arg(self, ann_args: tp.AnnArgs) -> tp.Any:
        """Get argument using `vectorbtpro.utils.parsing.match_ann_arg`."""
        return match_ann_arg(ann_args, self.arg_query)


@attr.s(frozen=True)
class AxisSpecifier:
    """Class with an attribute for specifying an axis."""

    axis: tp.Optional[int] = attr.ib(default=None)
    """Axis of the argument to take from."""


@attr.s(frozen=True)
class DimRetainer:
    """Class with an attribute for retaining dimensions."""

    keep_dims: bool = attr.ib(default=False)
    """Whether to retain dimensions."""


# ############# Chunk sizing ############# #


class Sizer:
    """Abstract class for getting the size from annotated arguments.

    !!! note
        Use `Sizer.apply` instead of `Sizer.get_size`."""

    def apply(self, ann_args: tp.AnnArgs) -> int:
        """Apply the sizer."""
        return self.get_size(ann_args)

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        """Get the size given the annotated arguments."""
        raise NotImplementedError


@attr.s(frozen=True)
class ArgSizer(Sizer, ArgGetter):
    """Class for getting the size from an argument."""

    single_type: tp.Optional[tp.TypeLike] = attr.ib(default=None)
    """One or multiple types to consider as a single value."""

    def apply(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if self.single_type is not None:
            if checks.is_instance_of(arg, self.single_type):
                return 1
        return self.get_size(ann_args)

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        return self.get_arg(ann_args)


class LenSizer(ArgSizer):
    """Class for getting the size from the length of an argument."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        return len(self.get_arg(ann_args))


@attr.s(frozen=True)
class ShapeSizer(ArgSizer, AxisSpecifier):
    """Class for getting the size from the length of an axis in a shape."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if len(arg) == 0:
            return 0
        axis = self.axis
        if axis is None:
            if len(arg) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis <= len(arg) - 1:
            return arg[axis]
        return 0


class ArraySizer(ShapeSizer):
    """Class for getting the size from the length of an axis in an array."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if len(arg.shape) == 0:
            return 0
        axis = self.axis
        if axis is None:
            if len(arg.shape) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis <= len(arg.shape) - 1:
            return arg.shape[axis]
        return 0


# ############# Chunk generation ############# #


@attr.s(frozen=True)
class ChunkMeta:
    """Class that represents a chunk metadata."""

    uuid: str = attr.ib()
    """Unique identifier of the chunk.

    Used for caching."""

    idx: int = attr.ib()
    """Chunk index."""

    start: tp.Optional[int] = attr.ib()
    """Start of the chunk range (including). Can be None."""

    end: tp.Optional[int] = attr.ib()
    """End of the chunk range (excluding). Can be None."""

    indices: tp.Optional[tp.Sequence[int]] = attr.ib()
    """Indices included in the chunk range. Can be None.

    Has priority over `ChunkMeta.start` and `ChunkMeta.end`."""


def yield_chunk_meta(
    n_chunks: tp.Optional[int] = None,
    size: tp.Optional[int] = None,
    min_size: tp.Optional[int] = None,
    chunk_len: tp.Optional[int] = None,
) -> tp.Generator[ChunkMeta, None, None]:
    """Yield meta of each successive chunk from a sequence with a number of elements.

    If both `n_chunks` and `chunk_len` are None (after resolving them from settings),
    sets `n_chunks` to the number of cores.

    For defaults, see `vectorbtpro._settings.chunking`."""
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    if n_chunks is None:
        n_chunks = chunking_cfg["n_chunks"]
    if min_size is None:
        min_size = chunking_cfg["min_size"]
    if chunk_len is None:
        chunk_len = chunking_cfg["chunk_len"]

    if size is not None and min_size is not None and size < min_size:
        yield ChunkMeta(uuid=str(uuid.uuid4()), idx=0, start=0, end=size, indices=None)
    else:
        if n_chunks is None and chunk_len is None:
            n_chunks = "auto"
        if n_chunks is not None and chunk_len is not None:
            raise ValueError("Either n_chunks or chunk_len must be set, not both")
        if n_chunks is not None:
            if isinstance(n_chunks, str):
                if n_chunks.lower() == "auto":
                    n_chunks = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid option n_chunks='{n_chunks}'")
            if n_chunks == 0:
                raise ValueError("Chunk count cannot be zero")
            if size is not None:
                if n_chunks > size:
                    n_chunks = size
                d, r = divmod(size, n_chunks)
                for i in range(n_chunks):
                    si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    yield ChunkMeta(
                        uuid=str(uuid.uuid4()),
                        idx=i,
                        start=si,
                        end=si + (d + 1 if i < r else d),
                        indices=None,
                    )
            else:
                for i in range(n_chunks):
                    yield ChunkMeta(uuid=str(uuid.uuid4()), idx=i, start=None, end=None, indices=None)
        if chunk_len is not None:
            checks.assert_not_none(size)
            if isinstance(chunk_len, str):
                if chunk_len.lower() == "auto":
                    chunk_len = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid option chunk_len='{chunk_len}'")
            if chunk_len == 0:
                raise ValueError("Chunk length cannot be zero")
            for chunk_i, i in enumerate(range(0, size, chunk_len)):
                yield ChunkMeta(
                    uuid=str(uuid.uuid4()),
                    idx=chunk_i,
                    start=i,
                    end=min(i + chunk_len, size),
                    indices=None,
                )


class ChunkMetaGenerator:
    """Abstract class for generating chunk metadata from annotated arguments."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs) -> tp.Iterable[ChunkMeta]:
        """Get chunk metadata."""
        raise NotImplementedError


class ArgChunkMeta(ChunkMetaGenerator, ArgGetter):
    """Class for generating chunk metadata from an argument."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs) -> tp.Iterable[ChunkMeta]:
        return self.get_arg(ann_args)


class LenChunkMeta(ArgChunkMeta):
    """Class for generating chunk metadata from a sequence of chunk lengths."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs) -> tp.Iterable[ChunkMeta]:
        arg = self.get_arg(ann_args)
        start = 0
        end = 0
        for i, chunk_len in enumerate(arg):
            end += chunk_len
            yield ChunkMeta(uuid=str(uuid.uuid4()), idx=i, start=start, end=end, indices=None)
            start = end


def get_chunk_meta_from_args(
    ann_args: tp.AnnArgs,
    n_chunks: tp.Optional[tp.SizeLike] = None,
    size: tp.Optional[tp.SizeLike] = None,
    min_size: tp.Optional[int] = None,
    chunk_len: tp.Optional[tp.SizeLike] = None,
    chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
) -> tp.Iterable[ChunkMeta]:
    """Get chunk metadata from annotated arguments.

    Args:
        ann_args (dict): Arguments annotated with `vectorbtpro.utils.parsing.annotate_args`.
        n_chunks (int, Sizer, or callable): Number of chunks.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        size (int, Sizer, or callable): Size of the space to split.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        min_size (int): If `size` is lower than this number, returns a single chunk.
        chunk_len (int, Sizer, or callable): Length of each chunk.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        chunk_meta (iterable of ChunkMeta, ChunkMetaGenerator, or callable): Chunk meta.

            Can be an iterable of `ChunkMeta`, an instance of `ChunkMetaGenerator`, or
            a callable taking the annotated arguments and returning an iterable."""
    if chunk_meta is None:
        if n_chunks is not None:
            if isinstance(n_chunks, Sizer):
                n_chunks = n_chunks.apply(ann_args)
            elif callable(n_chunks):
                n_chunks = n_chunks(ann_args)
            elif not isinstance(n_chunks, int):
                raise TypeError(f"Type {type(n_chunks)} for n_chunks is not supported")
        if size is not None:
            if isinstance(size, Sizer):
                size = size.apply(ann_args)
            elif callable(size):
                size = size(ann_args)
            elif not isinstance(size, int):
                raise TypeError(f"Type {type(size)} for size is not supported")
        if chunk_len is not None:
            if isinstance(chunk_len, Sizer):
                chunk_len = chunk_len.apply(ann_args)
            elif callable(chunk_len):
                chunk_len = chunk_len(ann_args)
            elif not isinstance(chunk_len, int):
                raise TypeError(f"Type {type(chunk_len)} for chunk_len is not supported")
        return yield_chunk_meta(n_chunks=n_chunks, size=size, min_size=min_size, chunk_len=chunk_len)
    if isinstance(chunk_meta, ChunkMetaGenerator):
        return chunk_meta.get_chunk_meta(ann_args)
    if callable(chunk_meta):
        return chunk_meta(ann_args)
    return chunk_meta


# ############# Chunk mapping ############# #


@attr.s(frozen=True)
class ChunkMapper:
    """Abstract class for mapping chunk metadata.

    Implements the abstract `ChunkMapper.map` method.

    Supports caching of each pair of incoming and outgoing `ChunkMeta` instances.

    !!! note
        Use `ChunkMapper.apply` instead of `ChunkMapper.map`."""

    should_cache: bool = attr.ib(default=True)
    """Whether should cache."""

    chunk_meta_cache: tp.Dict[str, ChunkMeta] = attr.ib(factory=dict)
    """Cache for outgoing `ChunkMeta` instances keyed by UUID of the incoming ones."""

    def apply(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Apply the mapper."""
        if not self.should_cache:
            return self.map(chunk_meta, **kwargs)
        if chunk_meta.uuid not in self.chunk_meta_cache:
            new_chunk_meta = self.map(chunk_meta, **kwargs)
            self.chunk_meta_cache[chunk_meta.uuid] = new_chunk_meta
            return new_chunk_meta
        return self.chunk_meta_cache[chunk_meta.uuid]

    def map(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Abstract method for mapping chunk metadata.

        Takes the chunk metadata of type `ChunkMeta` and returns a new chunk metadata of the same type."""
        raise NotImplementedError


# ############# Chunk taking ############# #


@attr.s(frozen=True)
class ChunkTaker:
    """Abstract class for taking one or more elements based on the chunk index or range.

    !!! note
        Use `ChunkTaker.apply` instead of `ChunkTaker.take`."""

    single_type: tp.Optional[tp.TypeLike] = attr.ib(default=None)
    """One or multiple types to consider as a single value."""

    ignore_none: bool = attr.ib(default=True)
    """Whether to ignore None."""

    mapper: tp.Optional[ChunkMapper] = attr.ib(default=None)
    """Chunk mapper of type `ChunkMapper`."""

    def should_take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> bool:
        """Check whether should take a chunk or leave the argument as it is."""
        if self.ignore_none and obj is None:
            return False
        if self.single_type is not None:
            if checks.is_instance_of(obj, self.single_type):
                return False
        return True

    def apply(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Apply the taker."""
        if self.mapper is not None:
            chunk_meta = self.mapper.apply(chunk_meta, **kwargs)
        if not self.should_take(obj, chunk_meta, **kwargs):
            return obj
        return self.take(obj, chunk_meta, **kwargs)

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Abstract method for taking subset of data.

        Takes the argument object, the chunk meta (tuple out of the index, start index,
        and end index of the chunk), and other keyword arguments passed down the stack,
        at least the entire argument specification `arg_take_spec`."""
        raise NotImplementedError


@attr.s(frozen=True)
class ChunkSelector(ChunkTaker, DimRetainer):
    """Class for selecting one element based on the chunk index."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        if self.keep_dims:
            return obj[chunk_meta.idx : chunk_meta.idx + 1]
        return obj[chunk_meta.idx]


class ChunkSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the chunk range."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        if chunk_meta.indices is not None:
            return obj[chunk_meta.indices]
        return obj[chunk_meta.start : chunk_meta.end]


class CountAdapter(ChunkSlicer):
    """Class for adapting a count based on the chunk range."""

    def take(self, obj: int, chunk_meta: ChunkMeta, **kwargs) -> int:
        checks.assert_instance_of(obj, int)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj):
                raise IndexError(f"Positional indexers are out-of-bounds")
            return len(indices)
        if chunk_meta.start >= obj:
            return 0
        return min(obj, chunk_meta.end) - chunk_meta.start


@attr.s(frozen=True)
class ShapeSelector(ChunkSelector, AxisSpecifier):
    """Class for selecting one element from a shape's axis based on the chunk index."""

    def take(self, obj: tp.Shape, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        if chunk_meta.idx >= obj[axis]:
            raise IndexError(f"Index {chunk_meta.idx} is out of bounds for axis {axis} with size {obj[axis]}")
        obj = list(obj)
        if self.keep_dims:
            obj[axis] = 1
        else:
            del obj[axis]
        return tuple(obj)


@attr.s(frozen=True)
class ShapeSlicer(ChunkSlicer, AxisSpecifier):
    """Class for slicing multiple elements from a shape's axis based on the chunk range."""

    def take(self, obj: tp.Shape, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        obj = list(obj)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj[axis]):
                raise IndexError(f"Positional indexers are out-of-bounds")
            obj[axis] = len(indices)
        else:
            if chunk_meta.start >= obj[axis]:
                del obj[axis]
            else:
                obj[axis] = min(obj[axis], chunk_meta.end) - chunk_meta.start
        return tuple(obj)


class ArraySelector(ShapeSelector):
    """Class for selecting one element from an array's axis based on the chunk index."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if self.keep_dims:
            slc[axis] = slice(chunk_meta.idx, chunk_meta.idx + 1)
        else:
            slc[axis] = chunk_meta.idx
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


class ArraySlicer(ShapeSlicer):
    """Class for slicing multiple elements from an array's axis based on the chunk range."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
            else:
                raise ValueError("Axis is required")
        if axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if chunk_meta.indices is not None:
            slc[axis] = np.asarray(chunk_meta.indices)
        else:
            slc[axis] = slice(chunk_meta.start, chunk_meta.end)
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


@attr.s(frozen=True, init=False)
class ContainerTaker(ChunkTaker):
    """Class for taking from a container with other chunk takers.

    Accepts the specification of the container."""

    cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = attr.ib(default=None, validator=_assert_value_not_none)
    """Specification of the container."""

    def __init__(
        self,
        cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = None,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
    ):
        self.__attrs_init__(
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=cont_take_spec,
        )

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        raise NotImplementedError


class SequenceTaker(ContainerTaker):
    """Class for taking from a sequence container.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, silence_warnings: bool = False, **kwargs) -> tp.Sequence:
        new_obj = []
        for i, v in enumerate(obj):
            if i < len(self.cont_take_spec):
                take_spec = self.cont_take_spec[i]
            else:
                if not silence_warnings:
                    warnings.warn(
                        f"Argument at index {i} not found in SequenceTaker.cont_take_spec. Setting to None.",
                        stacklevel=2,
                    )
                take_spec = None
            new_obj.append(take_from_arg(v, take_spec, chunk_meta, **kwargs))
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)


class MappingTaker(ContainerTaker):
    """Class for taking from a mapping container.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Mapping, chunk_meta: ChunkMeta, silence_warnings: bool = False, **kwargs) -> tp.Mapping:
        new_obj = {}
        for k, v in obj.items():
            if k in self.cont_take_spec:
                take_spec = self.cont_take_spec[k]
            else:
                if not silence_warnings:
                    warnings.warn(
                        f"Argument with key '{k}' not found in MappingTaker.cont_take_spec. Setting to None.",
                        stacklevel=2,
                    )
                take_spec = None
            new_obj[k] = take_from_arg(v, take_spec, chunk_meta, **kwargs)
        return type(obj)(new_obj)


class ArgsTaker(SequenceTaker):
    """Class for taking from a variable arguments container."""

    def __init__(
        self,
        *args,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
    ):
        self.__attrs_init__(
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=args,
        )


class KwargsTaker(MappingTaker):
    """Class for taking from a variable keyword arguments container."""

    def __init__(
        self,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
        **kwargs,
    ):
        self.__attrs_init__(
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=kwargs,
        )


def take_from_arg(arg: tp.Any, take_spec: tp.TakeSpec, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
    """Take from the argument given the specification `take_spec`.

    If `take_spec` is None, returns the original object. Otherwise, must be an instance of `ChunkTaker`.

    `**kwargs` are passed to `ChunkTaker.apply`."""
    if take_spec is None:
        return arg
    if isinstance(take_spec, ChunkTaker):
        return take_spec.apply(arg, chunk_meta, **kwargs)
    raise TypeError(f"Specification of type {type(take_spec)} is not supported")


def take_from_args(
    ann_args: tp.AnnArgs,
    arg_take_spec: tp.ArgTakeSpec,
    chunk_meta: ChunkMeta,
    silence_warnings: bool = False,
) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Take from each in the annotated arguments given the specification using `take_from_arg`.

    Additionally, passes to `take_from_arg` as keyword arguments `ann_args` and `arg_take_spec`.

    `arg_take_spec` must be a dictionary, with keys being argument positions or names as generated by
    `vectorbtpro.utils.parsing.annotate_args`. For values, see `take_from_arg`.

    Returns arguments and keyword arguments that can be directly passed to the function
    using `func(*args, **kwargs)`."""
    new_args = ()
    new_kwargs = dict()
    for i, (arg_name, ann_arg) in enumerate(ann_args.items()):
        take_spec_found = False
        found_take_spec = None
        for take_spec_name, take_spec in arg_take_spec.items():
            if isinstance(take_spec_name, int):
                if take_spec_name == i:
                    take_spec_found = True
                    found_take_spec = take_spec
                    break
            elif isinstance(take_spec_name, Regex):
                if take_spec_name.matches(arg_name):
                    take_spec_found = True
                    found_take_spec = take_spec
                    break
            elif isinstance(take_spec, Regex):
                if take_spec.matches(take_spec_name):
                    take_spec_found = True
                    found_take_spec = take_spec
                    break
            else:
                if take_spec_name == arg_name:
                    take_spec_found = True
                    found_take_spec = take_spec
                    break
        if not take_spec_found and not silence_warnings:
            warnings.warn(f"Argument '{arg_name}' not found in arg_take_spec. Setting to None.", stacklevel=2)
        result = take_from_arg(
            ann_arg["value"],
            found_take_spec,
            chunk_meta,
            ann_args=ann_args,
            arg_take_spec=arg_take_spec,
            silence_warnings=silence_warnings,
        )
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            for new_arg in result:
                new_args += (new_arg,)
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            for new_kwarg_name, new_kwarg in result.items():
                new_kwargs[new_kwarg_name] = new_kwarg
        elif ann_arg["kind"] == inspect.Parameter.KEYWORD_ONLY:
            new_kwargs[arg_name] = result
        else:
            new_args += (result,)
    return new_args, new_kwargs


def yield_arg_chunks(
    func: tp.Callable,
    ann_args: tp.AnnArgs,
    chunk_meta: tp.Iterable[ChunkMeta],
    arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
    template_context: tp.Optional[tp.Mapping] = None,
    **kwargs,
) -> tp.Generator[tp.FuncArgs, None, None]:
    """Split annotated arguments into chunks using `take_from_args` and yield each chunk.

    Args:
        func (callable): Callable.
        ann_args (dict): Arguments annotated with `vectorbtpro.utils.parsing.annotate_args`.
        chunk_meta (iterable of ChunkMeta): Chunk metadata.
        arg_take_spec (sequence, mapping or callable): Chunk taking specification.

            Can be a dictionary (see `take_from_args`), a sequence that will be converted into a
            dictionary, or a callable taking the annotated arguments and chunk metadata of type
            `ChunkMeta`, and returning new arguments and keyword arguments.
        template_context (mapping): Mapping to replace templates in arguments and specification.
        **kwargs: Keyword arguments passed to `take_from_args` or to `arg_take_spec` if it's a callable.

    For defaults, see `vectorbtpro._settings.chunking`."""

    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    template_context = merge_dicts(chunking_cfg["template_context"], template_context)
    if arg_take_spec is None:
        arg_take_spec = {}

    for _chunk_meta in chunk_meta:
        context = merge_dicts(dict(ann_args=ann_args, chunk_meta=_chunk_meta), template_context)
        chunk_ann_args = substitute_templates(ann_args, context=context, sub_id="chunk_ann_args")
        if callable(arg_take_spec):
            chunk_args, chunk_kwargs = arg_take_spec(chunk_ann_args, _chunk_meta, **kwargs)
        else:
            chunk_arg_take_spec = arg_take_spec
            if not checks.is_mapping(chunk_arg_take_spec):
                chunk_arg_take_spec = dict(zip(range(len(chunk_arg_take_spec)), chunk_arg_take_spec))
            chunk_arg_take_spec = substitute_templates(chunk_arg_take_spec, context=context, sub_id="chunk_arg_take_spec")
            chunk_args, chunk_kwargs = take_from_args(chunk_ann_args, chunk_arg_take_spec, _chunk_meta, **kwargs)
        yield func, chunk_args, chunk_kwargs


def chunked(
    *args,
    n_chunks: tp.Optional[tp.SizeLike] = None,
    size: tp.Optional[tp.SizeLike] = None,
    min_size: tp.Optional[int] = None,
    chunk_len: tp.Optional[tp.SizeLike] = None,
    chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
    skip_one_chunk: tp.Optional[bool] = None,
    arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
    template_context: tp.Optional[tp.Mapping] = None,
    prepend_chunk_meta: tp.Optional[bool] = None,
    merge_func: tp.Union[None, str, tuple, tp.Callable] = None,
    merge_kwargs: tp.KwargsLike = None,
    return_raw_chunks: bool = False,
    silence_warnings: tp.Optional[bool] = None,
    disable: tp.Optional[bool] = None,
    forward_kwargs_as: tp.KwargsLike = None,
    execute_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that chunks the inputs of a function. Engine-agnostic.
    Returns a new function with the same signature as the passed one.

    Does the following:

    1. Generates chunk metadata by passing `n_chunks`, `size`, `min_size`, `chunk_len`, and `chunk_meta`
        to `get_chunk_meta_from_args`.
    2. Splits arguments and keyword arguments by passing chunk metadata, `arg_take_spec`,
        and `template_context` to `yield_arg_chunks`, which yields one chunk at a time.
    3. Executes all chunks by passing `**execute_kwargs` to `vectorbtpro.utils.execution.execute`.
    4. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`.

    Argument `merge_func` is resolved using `vectorbtpro.base.merging.resolve_merge_func`.

    Any template in both `execute_kwargs` and `merge_kwargs` will be substituted. You can use
    the keys `ann_args`, `chunk_meta`, `arg_take_spec`, and `funcs_args` to be replaced by the actual objects.

    Use `prepend_chunk_meta` to prepend an instance of `ChunkMeta` to the arguments.
    If None, prepends automatically if the first argument is named 'chunk_meta'.

    Each parameter can be modified in the `options` attribute of the wrapper function or
    directly passed as a keyword argument with a leading underscore.

    Chunking can be disabled using `disable` argument. Additionally, the entire wrapping mechanism
    can be disabled by using the global setting `disable_wrapping` (=> returns the wrapped function).

    For defaults, see `vectorbtpro._settings.chunking`.
    For example, to change the engine globally:

    ```pycon
    >>> import vectorbtpro as vbt

    >>> vbt.settings.chunking["execute_kwargs"]["engine"] = 'dask'
    ```

    !!! note
        If less than two chunks were generated and `skip_one_chunk` is True,
        executes the function without chunking.

    Usage:
        For testing purposes, let's divide the input array into 2 chunks and compute the mean in a sequential manner:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import numpy as np

        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer())
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        The `chunked` function is a decorator that takes `f` and creates a function that splits
        passed arguments, runs each chunk using an engine, and optionally, merges the results.
        It has the same signature as the original function:

        ```pycon
        >>> f
        <function __main__.f(a)>
        ```

        We can change any option at any time:

        ```pycon
        >>> # Change the option directly on the function
        >>> f.options.n_chunks = 3

        >>> f(np.arange(10))
        [1.5, 5.0, 8.0]

        >>> # Pass a new option with a leading underscore
        >>> f(np.arange(10), _n_chunks=4)
        [1.0, 4.0, 6.5, 8.5]
        ```

        When we run the wrapped function, it first generates a list of chunk metadata of type `ChunkMeta`.
        Chunk metadata contains the chunk index that can be used to split any input:

        ```pycon
        >>> from vectorbtpro.utils.chunking import yield_chunk_meta

        >>> list(yield_chunk_meta(n_chunks=2))
        [ChunkMeta(uuid='84d64eed-fbac-41e7-ad61-c917e809b3b8', idx=0, start=None, end=None, indices=None),
         ChunkMeta(uuid='577817c4-fdee-4ceb-ab38-dcd663d9ab11', idx=1, start=None, end=None, indices=None)]
        ```

        Additionally, it may contain the start and end index of the space we want to split.
        The space can be defined by the length of an input array, for example. In our case:

        ```pycon
        >>> list(yield_chunk_meta(n_chunks=2, size=10))
        [ChunkMeta(uuid='c1593842-dc31-474c-a089-e47200baa2be', idx=0, start=0, end=5, indices=None),
         ChunkMeta(uuid='6d0265e7-1204-497f-bc2c-c7b7800ec57d', idx=1, start=5, end=10, indices=None)]
        ```

        If we know the size of the space in advance, we can pass it as an integer constant.
        Otherwise, we need to tell `chunked` to derive the size from the inputs dynamically
        by passing any subclass of `Sizer`. In the example above, we instruct the wrapped function
        to derive the size from the length of the input array `a`.

        Once all chunks are generated, the wrapped function attempts to split inputs into chunks.
        The specification for this operation can be provided by the `arg_take_spec` argument, which
        in most cases is a dictionary of `ChunkTaker` instances keyed by the input name.
        Here's an example of a complex specification:

        ```pycon
        >>> arg_take_spec = dict(
        ...     a=vbt.ChunkSelector(),
        ...     args=vbt.ArgsTaker(
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ),
        ...     b=vbt.SequenceTaker([
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ]),
        ...     kwargs=vbt.KwargsTaker(
        ...         c=vbt.MappingTaker(dict(
        ...             d=vbt.ChunkSelector(),
        ...             e=None
        ...         ))
        ...     )
        ... )

        >>> @vbt.chunked(
        ...     n_chunks=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=arg_take_spec
        ... )
        ... def f(a, *args, b=None, **kwargs):
        ...     return a + sum(args) + sum(b) + sum(kwargs['c'].values())

        >>> f([1, 2, 3], 10, [1, 2, 3], b=(100, [1, 2, 3]), c=dict(d=[1, 2, 3], e=1000))
        [1114, 1118, 1122]
        ```

        After splitting all inputs into chunks, the wrapped function forwards them to the engine function.
        The engine argument can be either the name of a supported engine, or a callable. Once the engine
        has finished all tasks and returned a list of results, we can merge them back using `merge_func`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     merge_func="concat"
        ... )
        ... def f(a):
        ...     return a

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Instead of (or in addition to) specifying `arg_take_spec`, we can define our function with the
        first argument being `chunk_meta` to be able to split the arguments during the execution.
        The `chunked` decorator will automatically recognize and replace it with the actual `ChunkMeta` object:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat"
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        This may be a good idea in multi-threading, but a bad idea in multi-processing.

        The same can be accomplished by using templates (here we tell `chunked` to not replace
        the first argument by setting `prepend_chunk_meta` to False):

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat",
        ...     prepend_chunk_meta=False
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(vbt.Rep('chunk_meta'), np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Templates in arguments are substituted right before taking a chunk from them.

        Keyword arguments to the engine can be provided using `execute_kwargs`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     show_progress=True
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        100% |█████████████████████████████████| 2/2 [00:00<00:00, 81.11it/s]
        [2.0, 7.0]
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        nonlocal prepend_chunk_meta

        from vectorbtpro._settings import settings

        chunking_cfg = settings["chunking"]

        if chunking_cfg["disable_wrapping"]:
            return func

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == "chunk_meta":
                    prepend_chunk_meta = True

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            disable = kwargs.pop("_disable", wrapper.options["disable"])
            if disable is None:
                disable = chunking_cfg["disable"]
            if disable:
                return func(*args, **kwargs)

            n_chunks = kwargs.pop("_n_chunks", wrapper.options["n_chunks"])
            size = kwargs.pop("_size", wrapper.options["size"])
            min_size = kwargs.pop("_min_size", wrapper.options["min_size"])
            chunk_len = kwargs.pop("_chunk_len", wrapper.options["chunk_len"])
            skip_one_chunk = kwargs.pop("_skip_one_chunk", wrapper.options["skip_one_chunk"])
            if skip_one_chunk is None:
                skip_one_chunk = chunking_cfg["skip_one_chunk"]
            chunk_meta = kwargs.pop("_chunk_meta", wrapper.options["chunk_meta"])
            arg_take_spec = kwargs.pop("_arg_take_spec", wrapper.options["arg_take_spec"])
            if arg_take_spec is None:
                arg_take_spec = {}
            if isinstance(arg_take_spec, dict) and "chunk_meta" not in arg_take_spec:
                arg_take_spec["chunk_meta"] = None
            template_context = merge_dicts(wrapper.options["template_context"], kwargs.pop("_template_context", {}))
            execute_kwargs = merge_dicts(wrapper.options["execute_kwargs"], kwargs.pop("_execute_kwargs", {}))
            merge_func = kwargs.pop("_merge_func", wrapper.options["merge_func"])
            merge_kwargs = merge_dicts(wrapper.options["merge_kwargs"], kwargs.pop("_merge_kwargs", {}))
            return_raw_chunks = kwargs.pop("_return_raw_chunks", wrapper.options["return_raw_chunks"])
            silence_warnings = kwargs.pop("_silence_warnings", wrapper.options["silence_warnings"])
            if silence_warnings is None:
                silence_warnings = chunking_cfg["silence_warnings"]
            forward_kwargs_as = merge_dicts(wrapper.options["forward_kwargs_as"], kwargs.pop("_forward_kwargs_as", {}))
            if len(forward_kwargs_as) > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in forward_kwargs_as:
                        new_kwargs[forward_kwargs_as.pop(k)] = v
                    else:
                        new_kwargs[k] = v
                kwargs = new_kwargs
            if len(forward_kwargs_as) > 0:
                for k, v in forward_kwargs_as.items():
                    kwargs[v] = locals()[k]

            if prepend_chunk_meta:
                args = (Rep("chunk_meta"), *args)

            ann_args = annotate_args(func, args, kwargs)
            chunk_meta = list(
                get_chunk_meta_from_args(
                    ann_args,
                    n_chunks=n_chunks,
                    size=size,
                    min_size=min_size,
                    chunk_len=chunk_len,
                    chunk_meta=chunk_meta,
                )
            )
            if len(chunk_meta) < 2 and skip_one_chunk:
                return func(*args, **kwargs)
            funcs_args = yield_arg_chunks(
                func,
                ann_args,
                chunk_meta=chunk_meta,
                arg_take_spec=arg_take_spec,
                template_context=template_context,
                silence_warnings=silence_warnings,
            )
            if return_raw_chunks:
                return chunk_meta, funcs_args
            context = merge_dicts(
                dict(
                    ann_args=ann_args,
                    chunk_meta=chunk_meta,
                    arg_take_spec=arg_take_spec,
                ),
                template_context,
            )
            execute_kwargs = substitute_templates(execute_kwargs, context, sub_id="execute_kwargs")
            results = execute(funcs_args, n_calls=len(chunk_meta), **execute_kwargs)
            if merge_func is not None:
                context["funcs_args"] = funcs_args
                if isinstance(merge_func, (str, tuple)):
                    from vectorbtpro.base.merging import resolve_merge_func

                    merge_func = resolve_merge_func(merge_func)
                merge_kwargs = substitute_templates(merge_kwargs, context, sub_id="merge_kwargs")
                return merge_func(results, **merge_kwargs)
            return results

        wrapper.is_chunked = True
        wrapper.options = Config(
            dict(
                n_chunks=n_chunks,
                size=size,
                min_size=min_size,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta,
                skip_one_chunk=skip_one_chunk,
                arg_take_spec=arg_take_spec,
                template_context=template_context,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                return_raw_chunks=return_raw_chunks,
                silence_warnings=silence_warnings,
                disable=disable,
                forward_kwargs_as=forward_kwargs_as,
                execute_kwargs=merge_dicts(kwargs, execute_kwargs),
            ),
            options_=dict(
                frozen_keys=True,
                as_attrs=True,
            )
        )

        if prepend_chunk_meta:
            signature = inspect.signature(wrapper)
            wrapper.__signature__ = signature.replace(parameters=tuple(signature.parameters.values())[1:])

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def resolve_chunked_option(option: tp.ChunkedOption = None) -> tp.KwargsLike:
    """Return keyword arguments for `chunked`.

    `option` can be:

    * True: Chunk using default settings
    * None or False: Do not chunk
    * string: Use `option` as the name of an execution engine (see `vectorbtpro.utils.execution.execute`)
    * dict: Use `option` as keyword arguments passed to `chunked`

    For defaults, see `option` in `vectorbtpro._settings.chunking`."""
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    if option is None:
        option = chunking_cfg["option"]

    if isinstance(option, bool):
        if not option:
            return None
        return dict()
    if isinstance(option, dict):
        return option
    elif isinstance(option, str):
        return dict(engine=option)
    raise TypeError(f"Type {type(option)} is invalid for a chunking option")


def specialize_chunked_option(option: tp.ChunkedOption = None, **kwargs):
    """Resolve `option` and merge it with `kwargs` if it's not None so the dict can be passed
    as an option to other functions."""
    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        return merge_dicts(kwargs, chunked_kwargs)
    return chunked_kwargs


def resolve_chunked(func: tp.Callable, option: tp.ChunkedOption = None, **kwargs) -> tp.Callable:
    """Decorate with `chunked` based on an option."""
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        if isinstance(chunking_cfg["option"], dict):
            chunked_kwargs = merge_dicts(chunking_cfg["option"], kwargs, chunked_kwargs)
        else:
            chunked_kwargs = merge_dicts(kwargs, chunked_kwargs)
        return chunked(func, **chunked_kwargs)
    return func
