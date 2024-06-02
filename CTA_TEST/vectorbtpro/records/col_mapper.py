# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class for mapping column arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.grouping import nb as grouping_nb
from vectorbtpro.records import nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.decorators import cached_property, cached_method

__all__ = [
    "ColumnMapper",
]


ColumnMapperT = tp.TypeVar("ColumnMapperT", bound="ColumnMapper")


class ColumnMapper(Wrapping):
    """Used by `vectorbtpro.records.base.Records` and `vectorbtpro.records.mapped_array.MappedArray`
    classes to make use of column and group metadata."""

    @classmethod
    def row_stack(
        cls: tp.Type[ColumnMapperT],
        *objs: tp.MaybeTuple[ColumnMapperT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ColumnMapperT:
        """Stack multiple `ColumnMapper` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, ColumnMapper):
                raise TypeError("Each object to be merged must be an instance of ColumnMapper")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        if "col_arr" not in kwargs:
            col_arrs = []
            for col in range(kwargs["wrapper"].shape_2d[1]):
                for obj in objs:
                    col_idxs, col_lens = obj.col_map
                    if len(col_idxs) > 0:
                        if col > 0 and obj.wrapper.shape_2d[1] == 1:
                            col_arrs.append(np.full(col_lens[0], col))
                        elif col_lens[col] > 0:
                            col_arrs.append(np.full(col_lens[col], col))
            kwargs["col_arr"] = np.concatenate(col_arrs)
        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[ColumnMapperT],
        *objs: tp.MaybeTuple[ColumnMapperT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ColumnMapperT:
        """Stack multiple `ColumnMapper` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        !!! note
            Will produce a column-sorted array."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, ColumnMapper):
                raise TypeError("Each object to be merged must be an instance of ColumnMapper")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "col_arr" not in kwargs:
            col_arrs = []
            col_sum = 0
            for obj in objs:
                col_idxs, col_lens = obj.col_map
                if len(col_idxs) > 0:
                    col_arrs.append(obj.col_arr[col_idxs] + col_sum)
                col_sum += obj.wrapper.shape_2d[1]
            kwargs["col_arr"] = np.concatenate(col_arrs)
        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Wrapping._expected_keys or set()) | {
        "col_arr",
    }

    def __init__(self, wrapper: ArrayWrapper, col_arr: tp.Array1d, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, col_arr=col_arr, **kwargs)

        self._col_arr = col_arr

        # Cannot select rows
        self._column_only_select = True

    def select_cols(
        self,
        col_idxs: tp.MaybeIndexArray,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Select columns.

        Returns indices and new column array. Automatically decides whether to use column lengths or column map."""
        if len(self.col_arr) == 0:
            return np.arange(len(self.col_arr)), self.col_arr
        if isinstance(col_idxs, slice):
            if col_idxs.start is None and col_idxs.stop is None:
                return np.arange(len(self.col_arr)), self.col_arr
            col_idxs = np.arange(col_idxs.start, col_idxs.stop)
        if self.is_sorted():
            func = jit_reg.resolve_option(grouping_nb.group_lens_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_lens, to_1d_array(col_idxs))  # faster
        else:
            func = jit_reg.resolve_option(grouping_nb.group_map_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_map, to_1d_array(col_idxs))  # more flexible
        return new_indices, new_col_arr

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `ColumnMapper` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        new_indices, new_col_arr = self.select_cols(wrapper_meta["col_idxs"])
        return dict(
            wrapper_meta=wrapper_meta,
            new_indices=new_indices,
            new_col_arr=new_col_arr,
        )

    def indexing_func(self: ColumnMapperT, *args, col_mapper_meta: tp.DictLike = None, **kwargs) -> ColumnMapperT:
        """Perform indexing on `ColumnMapper`."""
        if col_mapper_meta is None:
            col_mapper_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=col_mapper_meta["wrapper_meta"]["new_wrapper"],
            col_arr=col_mapper_meta["new_col_arr"],
        )

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array."""
        return self._col_arr

    @cached_method(whitelist=True)
    def get_col_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Get group-aware column array."""
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        return col_arr

    @cached_property(whitelist=True)
    def col_lens(self) -> tp.GroupLens:
        """Column lengths.

        Faster than `ColumnMapper.col_map` but only compatible with sorted columns."""
        func = jit_reg.resolve_option(nb.col_lens_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_lens(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None) -> tp.GroupLens:
        """Get group-aware column lengths."""
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_lens
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.col_lens_nb, jitted)
        return func(col_arr, len(columns))

    @cached_property(whitelist=True)
    def col_map(self) -> tp.GroupMap:
        """Column map.

        More flexible than `ColumnMapper.col_lens`.
        More suited for mapped arrays."""
        func = jit_reg.resolve_option(nb.col_map_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_map(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None) -> tp.GroupMap:
        """Get group-aware column map."""
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_map
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.col_map_nb, jitted)
        return func(col_arr, len(columns))

    @cached_method(whitelist=True)
    def is_sorted(self, jitted: tp.JittedOption = None) -> bool:
        """Check whether column array is sorted."""
        func = jit_reg.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)

    @cached_property(whitelist=True)
    def new_id_arr(self) -> tp.Array1d:
        """Generate a new id array."""
        func = jit_reg.resolve_option(nb.generate_ids_nb, None)
        return func(self.col_arr, self.wrapper.shape_2d[1])

    @cached_method(whitelist=True)
    def get_new_id_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Generate a new group-aware id array."""
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.generate_ids_nb, None)
        return func(col_arr, len(columns))
