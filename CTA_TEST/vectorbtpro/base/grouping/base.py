# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base classes and functions for grouping.

Class `Grouper` stores metadata related to grouping index. It can return, for example,
the number of groups, the start indices of groups, and other information useful for reducing
operations that utilize grouping. It also allows to dynamically enable/disable/modify groups
and checks whether a certain operation is permitted."""

import attr

import numpy as np
import pandas as pd
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.array_ import is_sorted
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.decorators import cached_method
from vectorbtpro.utils.template import CustomTemplate
from vectorbtpro.base.grouping import nb

__all__ = [
    "Grouper",
    "ExceptLevel",
]

GroupByT = tp.Union[None, bool, tp.Index]


@attr.s(frozen=True)
class ExceptLevel:
    """Class for grouping except one or more levels."""

    level: tp.MaybeLevelSequence = attr.ib()
    """Level position or name."""


def group_by_to_index(index: tp.Index, group_by: tp.GroupByLike, def_lvl_name: tp.Hashable = "group") -> GroupByT:
    """Convert mapper `group_by` to `pd.Index`.

    !!! note
        Index and mapper must have the same length."""
    if group_by is None or group_by is False:
        return group_by
    if isinstance(group_by, CustomTemplate):
        group_by = group_by.substitute(context=dict(index=index), strict=True, sub_id="group_by")
    if group_by is True:
        group_by = pd.Index(["group"] * len(index), name=def_lvl_name)  # one group
    elif isinstance(group_by, ExceptLevel):
        except_levels = group_by.level
        if isinstance(except_levels, (int, str)):
            except_levels = [except_levels]
        new_group_by = []
        for i, name in enumerate(index.names):
            if i not in except_levels and name not in except_levels:
                new_group_by.append(name)
        if len(new_group_by) == 0:
            group_by = pd.Index(["group"] * len(index), name=def_lvl_name)
        else:
            if len(new_group_by) == 1:
                new_group_by = new_group_by[0]
            group_by = indexes.select_levels(index, new_group_by)
    elif isinstance(group_by, (int, str)):
        group_by = indexes.select_levels(index, group_by)
    elif isinstance(group_by, (tuple, list)) and len(group_by) <= len(index.names):
        try:
            group_by = indexes.select_levels(index, group_by)
        except (IndexError, KeyError):
            pass
    if not isinstance(group_by, pd.Index):
        group_by = pd.Index(group_by, name=def_lvl_name)
    if len(group_by) != len(index):
        raise ValueError("group_by and index must have the same length")
    return group_by


def get_groups_and_index(index: tp.Index, group_by: tp.GroupByLike, def_lvl_name: tp.Hashable = "group",) -> tp.Tuple[tp.Array1d, tp.Index]:
    """Return array of group indices pointing to the original index, and grouped index."""
    if group_by is None or group_by is False:
        return np.arange(len(index)), index

    group_by = group_by_to_index(index, group_by, def_lvl_name)
    codes, uniques = pd.factorize(group_by)
    if not isinstance(uniques, pd.Index):
        new_index = pd.Index(uniques)
    else:
        new_index = uniques
    if isinstance(group_by, pd.MultiIndex):
        new_index.names = group_by.names
    elif isinstance(group_by, (pd.Index, pd.Series)):
        new_index.name = group_by.name
    return codes, new_index


GrouperT = tp.TypeVar("GrouperT", bound="Grouper")


class Grouper(Configured):
    """Class that exposes methods to group index.

    `group_by` can be:

    * boolean (False for no grouping, True for one group),
    * integer (level by position),
    * string (level by name),
    * sequence of integers or strings that is shorter than `index` (multiple levels),
    * any other sequence that has the same length as `index` (group per index).

    Set `allow_enable` to False to prohibit grouping if `Grouper.group_by` is None.
    Set `allow_disable` to False to prohibit disabling of grouping if `Grouper.group_by` is not None.
    Set `allow_modify` to False to prohibit modifying groups (you can still change their labels).

    All properties are read-only to enable caching.

    !!! note
        Columns must form monolithic groups for using `get_group_lens_nb`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Grouper.replace`."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "index",
        "group_by",
        "def_lvl_name",
        "allow_enable",
        "allow_disable",
        "allow_modify",
    }

    def __init__(
        self,
        index: tp.Index,
        group_by: tp.GroupByLike = None,
        def_lvl_name: tp.Hashable = "group",
        allow_enable: bool = True,
        allow_disable: bool = True,
        allow_modify: bool = True,
        **kwargs,
    ) -> None:

        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if group_by is None or group_by is False:
            group_by = None
        else:
            group_by = group_by_to_index(index, group_by, def_lvl_name=def_lvl_name)

        self._index = index
        self._group_by = group_by
        self._def_lvl_name = def_lvl_name
        self._allow_enable = allow_enable
        self._allow_disable = allow_disable
        self._allow_modify = allow_modify

        Configured.__init__(
            self,
            index=index,
            group_by=group_by,
            def_lvl_name=def_lvl_name,
            allow_enable=allow_enable,
            allow_disable=allow_disable,
            allow_modify=allow_modify,
            **kwargs,
        )

    @classmethod
    def from_pd_group_by(
        cls: tp.Type[GrouperT],
        pd_group_by: tp.PandasGroupByLike,
        **kwargs,
    ) -> GrouperT:
        """Build a `Grouper` instance from a pandas `GroupBy` object.

        Indices are stored under `index` and group labels under `group_by`."""
        if not isinstance(pd_group_by, (PandasGroupBy, PandasResampler)):
            raise TypeError("pd_group_by must be an instance of GroupBy or Resampler")
        indices = list(pd_group_by.indices.values())
        group_lens = np.asarray(list(map(len, indices)))
        groups = np.full(int(np.sum(group_lens)), 0, dtype=np.int_)
        group_start_idxs = np.cumsum(group_lens)[1:] - group_lens[1:]
        groups[group_start_idxs] = 1
        groups = np.cumsum(groups)
        index = pd.Index(np.concatenate(indices))
        group_by = pd.Index(list(pd_group_by.indices.keys()), name="group")[groups]
        return cls(
            index=index,
            group_by=group_by,
            **kwargs,
        )

    @property
    def index(self) -> tp.Index:
        """Original index."""
        return self._index

    @property
    def group_by(self) -> GroupByT:
        """Mapper for grouping."""
        return self._group_by

    @property
    def def_lvl_name(self) -> tp.Hashable:
        """Default level name."""
        return self._def_lvl_name

    @property
    def allow_enable(self) -> bool:
        """Whether to allow enabling grouping."""
        return self._allow_enable

    @property
    def allow_disable(self) -> bool:
        """Whether to allow disabling grouping."""
        return self._allow_disable

    @property
    def allow_modify(self) -> bool:
        """Whether to allow changing groups."""
        return self._allow_modify

    def is_grouped(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether index are grouped."""
        if group_by is False:
            return False
        if group_by is None:
            group_by = self.group_by
        return group_by is not None

    def is_grouping_enabled(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping has been enabled."""
        return self.group_by is None and self.is_grouped(group_by=group_by)

    def is_grouping_disabled(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping has been disabled."""
        return self.group_by is not None and not self.is_grouped(group_by=group_by)

    @cached_method(whitelist=True)
    def is_grouping_modified(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping has been modified.

        Doesn't care if grouping labels have been changed."""
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        group_by = group_by_to_index(self.index, group_by, def_lvl_name=self.def_lvl_name)
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            if not pd.Index.equals(group_by, self.group_by):
                groups1 = get_groups_and_index(self.index, group_by, def_lvl_name=self.def_lvl_name)[0]
                groups2 = get_groups_and_index(self.index, self.group_by, def_lvl_name=self.def_lvl_name)[0]
                if not np.array_equal(groups1, groups2):
                    return True
            return False
        return True

    @cached_method(whitelist=True)
    def is_grouping_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping has changed in any way."""
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            if pd.Index.equals(group_by, self.group_by):
                return False
        return True

    def is_group_count_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether the number of groups has changed."""
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            return len(group_by) != len(self.group_by)
        return True

    def check_group_by(
        self,
        group_by: tp.GroupByLike = None,
        allow_enable: tp.Optional[bool] = None,
        allow_disable: tp.Optional[bool] = None,
        allow_modify: tp.Optional[bool] = None,
    ) -> None:
        """Check passed `group_by` object against restrictions."""
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if allow_modify is None:
            allow_modify = self.allow_modify

        if self.is_grouping_enabled(group_by=group_by):
            if not allow_enable:
                raise ValueError("Enabling grouping is not allowed")
        elif self.is_grouping_disabled(group_by=group_by):
            if not allow_disable:
                raise ValueError("Disabling grouping is not allowed")
        elif self.is_grouping_modified(group_by=group_by):
            if not allow_modify:
                raise ValueError("Modifying groups is not allowed")

    def resolve_group_by(self, group_by: tp.GroupByLike = None, **kwargs) -> GroupByT:
        """Resolve `group_by` from either object variable or keyword argument."""
        if group_by is None:
            group_by = self.group_by
        if group_by is False and self.group_by is None:
            group_by = None
        self.check_group_by(group_by=group_by, **kwargs)
        return group_by_to_index(self.index, group_by, def_lvl_name=self.def_lvl_name)

    @cached_method(whitelist=True)
    def get_groups_and_index(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.Tuple[tp.Array1d, tp.Index]:
        """See `get_groups_and_index`."""
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        return get_groups_and_index(self.index, group_by, def_lvl_name=self.def_lvl_name)

    def get_groups(self, **kwargs) -> tp.Array1d:
        """Return groups array."""
        return self.get_groups_and_index(**kwargs)[0]

    def get_index(self, **kwargs) -> tp.Index:
        """Return grouped index."""
        return self.get_groups_and_index(**kwargs)[1]

    def get_stretched_index(self, **kwargs) -> tp.Index:
        """Return stretched index."""
        groups, index = self.get_groups_and_index(**kwargs)
        return index[groups]

    def get_group_count(self, **kwargs) -> int:
        """Get number of groups."""
        return len(self.get_index(**kwargs))

    @cached_method(whitelist=True)
    def is_sorted(self, group_by: tp.GroupByLike = None, **kwargs) -> bool:
        """Return whether groups are monolithic, sorted."""
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        groups = self.get_groups(group_by=group_by)
        return is_sorted(groups)

    @cached_method(whitelist=True)
    def get_group_lens(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None, **kwargs) -> tp.GroupLens:
        """See get_group_lens_nb."""
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if group_by is None or group_by is False:  # no grouping
            return np.full(len(self.index), 1)
        if not self.is_sorted(group_by=group_by):
            raise ValueError("group_by must form monolithic, sorted groups")
        groups = self.get_groups(group_by=group_by)
        func = jit_reg.resolve_option(nb.get_group_lens_nb, jitted)
        return func(groups)

    def get_group_start_idxs(self, **kwargs) -> tp.Array1d:
        """Get first index of each group as an array."""
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens) - group_lens

    def get_group_end_idxs(self, **kwargs) -> tp.Array1d:
        """Get end index of each group as an array."""
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens)

    @cached_method(whitelist=True)
    def get_group_map(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None, **kwargs) -> tp.GroupMap:
        """See get_group_map_nb."""
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if group_by is None or group_by is False:  # no grouping
            return np.arange(len(self.index)), np.full(len(self.index), 1)
        groups, new_index = self.get_groups_and_index(group_by=group_by)
        func = jit_reg.resolve_option(nb.get_group_map_nb, jitted)
        return func(groups, len(new_index))

    def yield_group_idxs(self, **kwargs) -> tp.Generator[tp.GroupIdxs, None, None]:
        """Yield indices of each group."""
        group_idxs, group_lens = self.get_group_map(**kwargs)
        group_start = 0
        group_end = 0
        for g in range(len(group_lens)):
            group_len = group_lens[g]
            group_end += group_len
            yield group_idxs[group_start:group_end]
            group_start += group_len

    def __iter__(self) -> tp.Generator[tp.Tuple[tp.Label, tp.GroupIdxs], None, None]:
        index = self.get_index()
        for g, group_idxs in enumerate(self.yield_group_idxs()):
            yield index[g], group_idxs

    def select_groups(self, group_idxs: tp.Array1d, jitted: tp.JittedOption = None) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Select groups.

        Returns indices and new group array. Automatically decides whether to use group lengths or group map."""
        from vectorbtpro.base.reshaping import to_1d_array

        if self.is_sorted():
            func = jit_reg.resolve_option(nb.group_lens_select_nb, jitted)
            new_group_idxs, new_groups = func(self.get_group_lens(), to_1d_array(group_idxs))  # faster
        else:
            func = jit_reg.resolve_option(nb.group_map_select_nb, jitted)
            new_group_idxs, new_groups = func(self.get_group_map(), to_1d_array(group_idxs))  # more flexible
        return new_group_idxs, new_groups
