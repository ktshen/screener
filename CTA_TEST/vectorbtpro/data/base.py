# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with data sources.

Class `Data` allows storing, downloading, updating, and managing data. It stores data
as a dictionary of Series/DataFrames keyed by symbol, and makes sure that
all pandas objects have the same index and columns by aligning them.
"""

import warnings
from pathlib import Path
import traceback
import inspect
import string
from collections import defaultdict

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_any_array, to_pd_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.data.decorators import attach_symbol_dict_methods
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig, copy_dict
from vectorbtpro.utils.datetime_ import is_tz_aware, to_timezone, try_to_datetime_index
from vectorbtpro.utils.parsing import get_func_arg_names, extend_args
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.template import RepEval
from vectorbtpro.utils.pickling import pdict
from vectorbtpro.utils.execution import execute

__all__ = ["symbol_dict", "run_func_dict", "Data"]

__pdoc__ = {}


class symbol_dict(pdict):
    """Dict that contains symbols as keys."""

    pass


class run_func_dict(pdict):
    """Dict that contains function names as keys for `Data.run`."""

    pass


DataT = tp.TypeVar("DataT", bound="Data")


class MetaColumns(type):
    """Meta class that exposes a read-only class property `MetaColumns.column_config`."""

    @property
    def column_config(cls) -> Config:
        """Column config."""
        return cls._column_config


class DataWithColumns(metaclass=MetaColumns):
    """Class exposes a read-only class property `DataWithColumns.field_config`."""

    @property
    def column_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${column_config}
        ```
        """
        return self._column_config


class MetaData(type(Analyzable), type(DataWithColumns)):
    pass


@attach_symbol_dict_methods(
    [
        "symbol_classes",
        "fetch_kwargs",
        "returned_kwargs",
    ]
)
class Data(Analyzable, DataWithColumns, metaclass=MetaData):
    """Class that downloads, updates, and manages data coming from a data source."""

    _setting_keys: tp.SettingsKeys = dict(base="data")

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_column_config"}

    _column_config: tp.ClassVar[Config] = HybridConfig()

    @property
    def column_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${column_config}
        ```

        Returns `${cls_name}._column_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change fields, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._column_config`.
        """
        return self._column_config

    def use_column_config_of(self, cls: tp.Type[DataT]) -> None:
        """Copy column config from another `Data` class."""
        self._column_config = cls.column_config.copy()

    @classmethod
    def row_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        All objects to be merged must have the same symbols. Metadata such as the last index,
        fetching and returned keyword arguments, are taken from the last object."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        symbols = set()
        for obj in objs:
            symbols = symbols.union(set(obj.data.keys()))
        for obj in objs:
            if len(symbols.difference(set(obj.data.keys()))) > 0:
                raise ValueError("Objects to be merged must have the same symbols")
        if "data" not in kwargs:
            new_data = symbol_dict()
            for s in symbols:
                new_data[s] = kwargs["wrapper"].row_stack_arrs(*[obj.data[s] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        if "symbol_classes" not in kwargs:
            kwargs["symbol_classes"] = objs[-1].symbol_classes
        if "fetch_kwargs" not in kwargs:
            kwargs["fetch_kwargs"] = objs[-1].fetch_kwargs
        if "returned_kwargs" not in kwargs:
            kwargs["returned_kwargs"] = objs[-1].returned_kwargs
        if "last_index" not in kwargs:
            kwargs["last_index"] = objs[-1].last_index
        if "delisted" not in kwargs:
            kwargs["delisted"] = objs[-1].delisted

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        All objects to be merged must have the same symbols and index. Metadata such as the last index,
        fetching and returned keyword arguments, must be the same in all objects."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        symbols = set()
        for obj in objs:
            symbols = symbols.union(set(obj.data.keys()))
        for obj in objs:
            if len(symbols.difference(set(obj.data.keys()))) > 0:
                raise ValueError("Objects to be merged must have the same symbols")
        if "data" not in kwargs:
            new_data = symbol_dict()
            for s in symbols:
                new_data[s] = kwargs["wrapper"].column_stack_arrs(*[obj.data[s] for obj in objs], group_by=False)
            kwargs["data"] = new_data

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "data",
        "single_symbol",
        "symbol_classes",
        "fetch_kwargs",
        "returned_kwargs",
        "last_index",
        "delisted",
        "tz_localize",
        "tz_convert",
        "missing_index",
        "missing_columns",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        data: symbol_dict,
        single_symbol: bool = True,
        symbol_classes: tp.Optional[symbol_dict] = None,
        fetch_kwargs: tp.Optional[symbol_dict] = None,
        returned_kwargs: tp.Optional[symbol_dict] = None,
        last_index: tp.Optional[symbol_dict] = None,
        delisted: tp.Optional[symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            data=data,
            single_symbol=single_symbol,
            symbol_classes=symbol_classes,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

        if symbol_classes is None:
            symbol_classes = {}
        if fetch_kwargs is None:
            fetch_kwargs = {}
        if returned_kwargs is None:
            returned_kwargs = {}
        if last_index is None:
            last_index = {}
        if delisted is None:
            delisted = {}
        checks.assert_instance_of(data, dict)
        checks.assert_instance_of(symbol_classes, dict)
        checks.assert_instance_of(fetch_kwargs, dict)
        checks.assert_instance_of(returned_kwargs, dict)
        checks.assert_instance_of(last_index, dict)
        checks.assert_instance_of(delisted, dict)
        for symbol, obj in data.items():
            checks.assert_meta_equal(obj, data[list(data.keys())[0]])
        if len(data) > 1:
            single_symbol = False

        self._data = symbol_dict(data)
        self._single_symbol = single_symbol
        self._symbol_classes = symbol_dict(symbol_classes)
        self._fetch_kwargs = symbol_dict(fetch_kwargs)
        self._returned_kwargs = symbol_dict(returned_kwargs)
        self._last_index = symbol_dict(last_index)
        self._delisted = symbol_dict(delisted)
        self._tz_localize = tz_localize
        self._tz_convert = tz_convert
        self._missing_index = missing_index
        self._missing_columns = missing_columns

        # Copy writeable attrs
        self._column_config = type(self)._column_config.copy()

    def replace(self: DataT, **kwargs) -> DataT:
        """See `vectorbtpro.utils.config.Configured.replace`.

        Replaces the data's index and/or columns if they were changed in the wrapper."""
        if "wrapper" in kwargs and "data" not in kwargs:
            wrapper = kwargs["wrapper"]
            if isinstance(wrapper, dict):
                new_index = wrapper.get("index", self.wrapper.index)
                new_columns = wrapper.get("columns", self.wrapper.columns)
            else:
                new_index = wrapper.index
                new_columns = wrapper.columns
            data = self.config["data"]
            new_data = symbol_dict()
            data_changed = False
            for k, v in data.items():
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    if not v.index.equals(new_index):
                        v = v.copy(deep=False)
                        v.index = new_index
                        data_changed = True
                    if isinstance(v, pd.DataFrame):
                        if not v.columns.equals(new_columns):
                            v = v.copy(deep=False)
                            v.columns = new_columns
                            data_changed = True
                new_data[k] = v
            if data_changed:
                kwargs["data"] = new_data
        return Analyzable.replace(self, **kwargs)

    def indexing_func(self: DataT, *args, **kwargs) -> DataT:
        """Perform indexing on `Data`."""
        wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_wrapper = wrapper_meta["new_wrapper"]
        new_data = {}
        for k, v in self.data.items():
            if wrapper_meta["rows_changed"]:
                v = v.iloc[wrapper_meta["row_idxs"]]
            if wrapper_meta["columns_changed"]:
                v = v.iloc[:, wrapper_meta["col_idxs"]]
            new_data[k] = v
        new_last_index = dict()
        for s, obj in self.data.items():
            if s in self.last_index:
                new_last_index[s] = min([self.last_index[s], new_wrapper.index[-1]])
        return self.replace(wrapper=new_wrapper, data=new_data, last_index=new_last_index)

    @property
    def data(self) -> symbol_dict:
        """Data dictionary keyed by symbol of type `symbol_dict`."""
        return self._data

    @property
    def single_symbol(self) -> bool:
        """Whether there is only one symbol in `Data.data`."""
        return self._single_symbol

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        """List of symbols."""
        return list(self.data.keys())

    @property
    def symbol_classes(self) -> symbol_dict:
        """Symbol classes of type `symbol_dict`."""
        return self._symbol_classes

    @property
    def fetch_kwargs(self) -> symbol_dict:
        """Keyword arguments of type `symbol_dict` initially passed to `Data.fetch_symbol`."""
        return self._fetch_kwargs

    @property
    def returned_kwargs(self) -> symbol_dict:
        """Keyword arguments of type `symbol_dict` returned by `Data.fetch_symbol`."""
        return self._returned_kwargs

    @property
    def last_index(self) -> symbol_dict:
        """Last fetched index per symbol of type `symbol_dict`."""
        return self._last_index

    @property
    def delisted(self) -> symbol_dict:
        """Delisted flag per symbol of type `symbol_dict`."""
        return self._delisted

    @property
    def tz_localize(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """`tz_localize` initially passed to `Data.fetch_symbol`."""
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """`tz_convert` initially passed to `Data.fetch_symbol`."""
        return self._tz_convert

    @property
    def missing_index(self) -> tp.Optional[str]:
        """`missing_index` initially passed to `Data.fetch_symbol`."""
        return self._missing_index

    @property
    def missing_columns(self) -> tp.Optional[str]:
        """`missing_columns` initially passed to `Data.fetch_symbol`."""
        return self._missing_columns

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object was two-dimensional.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.shape_2d

    @property
    def columns(self) -> tp.Index:
        """Columns.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.columns

    @property
    def index(self) -> tp.Index:
        """Index.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.index

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """Frequency.

        Based on the default symbol wrapper."""
        return self.symbol_wrapper.freq

    # ############# Pre- and post-processing ############# #

    @classmethod
    def prepare_tzaware_index(
        cls,
        obj: tp.SeriesFrame,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
    ) -> tp.SeriesFrame:
        """Prepare a timezone-aware index of a pandas object.

        If the index is tz-naive, convert to a timezone using `tz_localize`.
        Convert the index from one timezone to another using `tz_convert`.
        See `vectorbtpro.utils.datetime_.to_timezone`.

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if tz_localize is None:
            tz_localize = data_cfg["tz_localize"]
        if isinstance(tz_localize, bool):
            if tz_localize:
                raise ValueError("tz_localize cannot be True")
            else:
                tz_localize = None
        if tz_convert is None:
            tz_convert = data_cfg["tz_convert"]
        if isinstance(tz_convert, bool):
            if tz_convert:
                raise ValueError("tz_convert cannot be True")
            else:
                tz_convert = None

        if isinstance(obj.index, pd.DatetimeIndex):
            if tz_localize is not None:
                if not is_tz_aware(obj.index):
                    obj = obj.tz_localize(to_timezone(tz_localize))
            if tz_convert is not None:
                obj = obj.tz_convert(to_timezone(tz_convert))
        obj.index = try_to_datetime_index(obj.index)
        return obj

    @classmethod
    def align_index(
        cls,
        data: symbol_dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> symbol_dict:
        """Align data to have the same index.

        The argument `missing` accepts the following values:

        * 'nan': set missing data points to NaN
        * 'drop': remove missing data points
        * 'raise': raise an error

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if missing is None:
            missing = data_cfg["missing_index"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        index = None
        for symbol, obj in data.items():
            obj_index = obj.index.sort_values()
            if index is None:
                index = obj_index
            else:
                if not index.equals(obj_index):
                    if missing == "nan":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching index. Setting missing data points to NaN.",
                                stacklevel=2,
                            )
                        index = index.union(obj_index)
                    elif missing == "drop":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching index. Dropping missing data points.",
                                stacklevel=2,
                            )
                        index = index.intersection(obj_index)
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching index")
                    else:
                        raise ValueError(f"Invalid option missing='{missing}'")

        # reindex
        new_data = symbol_dict({symbol: obj.reindex(index=index) for symbol, obj in data.items()})
        return new_data

    @classmethod
    def align_columns(
        cls,
        data: symbol_dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> symbol_dict:
        """Align data to have the same columns.

        See `Data.align_index` for `missing`."""
        if len(data) == 1:
            return data

        data_cfg = cls.get_settings(key_id="base")

        if missing is None:
            missing = data_cfg["missing_columns"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        columns = None
        multiple_columns = False
        name_is_none = False
        for symbol, obj in data.items():
            if checks.is_series(obj):
                if obj.name is None:
                    name_is_none = True
                obj = obj.to_frame()
            else:
                multiple_columns = True
            obj_columns = obj.columns
            if columns is None:
                columns = obj_columns
            else:
                if not columns.equals(obj_columns):
                    if missing == "nan":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching columns. Setting missing data points to NaN.",
                                stacklevel=2,
                            )
                        columns = columns.union(obj_columns)
                    elif missing == "drop":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching columns. Dropping missing data points.",
                                stacklevel=2,
                            )
                        columns = columns.intersection(obj_columns)
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching columns")
                    else:
                        raise ValueError(f"Invalid option missing='{missing}'")

        # reindex
        new_data = symbol_dict()
        for symbol, obj in data.items():
            if checks.is_series(obj):
                obj = obj.to_frame()
            obj = obj.reindex(columns=columns)
            if not multiple_columns:
                obj = obj[columns[0]]
                if name_is_none:
                    obj = obj.rename(None)
            new_data[symbol] = obj
        return new_data

    @staticmethod
    def select_symbol_kwargs(symbol: tp.Symbol, kwargs: tp.DictLike) -> dict:
        """Select keyword arguments belonging to `symbol`."""
        if kwargs is None:
            return {}
        if isinstance(kwargs, symbol_dict):
            if symbol not in kwargs:
                return {}
            kwargs = kwargs[symbol]
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, symbol_dict):
                if symbol in v:
                    _kwargs[k] = v[symbol]
            else:
                _kwargs[k] = v
        return _kwargs

    def switch_class(
        self,
        new_cls: tp.Type[DataT],
        clear_fetch_kwargs: bool = False,
        clear_returned_kwargs: bool = False,
        **kwargs,
    ) -> DataT:
        """Switch the class of the data instance."""
        if clear_fetch_kwargs:
            new_fetch_kwargs = symbol_dict({k: dict() for k in self.symbols})
        else:
            new_fetch_kwargs = copy_dict(self.fetch_kwargs)
        if clear_returned_kwargs:
            new_returned_kwargs = symbol_dict({k: dict() for k in self.symbols})
        else:
            new_returned_kwargs = copy_dict(self.returned_kwargs)
        return self.replace(cls_=new_cls, fetch_kwargs=new_fetch_kwargs, returned_kwargs=new_returned_kwargs, **kwargs)

    @classmethod
    def invert_data(cls, pd_dict: tp.Dict[tp.Hashable, tp.SeriesFrame]) -> tp.Dict[tp.Hashable, tp.SeriesFrame]:
        """Invert data by swapping keys and columns."""
        if len(pd_dict) == 0:
            return pd_dict
        new_pd_dict = defaultdict(list)
        for k, v in pd_dict.items():
            if isinstance(v, pd.Series):
                new_pd_dict[v.name].append(v.rename(k))
            else:
                for c in v.columns:
                    new_pd_dict[c].append(v[c].rename(k))
        new_pd_dict2 = {}
        for k, v in new_pd_dict.items():
            if len(v) == 1:
                new_pd_dict2[k] = v[0]
            else:
                new_pd_dict2[k] = pd.concat(v, axis=1)
        return new_pd_dict2

    @classmethod
    def from_data(
        cls: tp.Type[DataT],
        data: tp.Union[symbol_dict, tp.SeriesFrame],
        symbol_oriented: bool = False,
        single_symbol: bool = True,
        symbol_classes: tp.Optional[symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        fetch_kwargs: tp.Optional[symbol_dict] = None,
        returned_kwargs: tp.Optional[symbol_dict] = None,
        last_index: tp.Optional[symbol_dict] = None,
        delisted: tp.Optional[symbol_dict] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance from data.

        Args:
            data (dict): Dictionary of array-like objects keyed by symbol.
            symbol_oriented (bool): Whether columns are symbols and should be swapped.

                Uses `Data.invert_data`.
            single_symbol (bool): Whether there is only one symbol in `data`.
            symbol_classes (symbol_dict): See `Data.symbol_classes`.
            tz_localize (timezone_like): See `Data.prepare_tzaware_index`.
            tz_convert (timezone_like): See `Data.prepare_tzaware_index`.
            missing_index (str): See `Data.align_index`.
            missing_columns (str): See `Data.align_columns`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            fetch_kwargs (symbol_dict): Keyword arguments initially passed to `Data.fetch_symbol`.
            returned_kwargs (symbol_dict): Keyword arguments returned by `Data.fetch_symbol`.
            last_index (symbol_dict): Last fetched index per symbol.
            delisted (symbol_dict): Whether symbol has been delisted.
            silence_warnings (bool): Whether to silence all warnings.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if tz_localize is None:
            tz_localize = data_cfg["tz_localize"]
        if tz_convert is None:
            tz_convert = data_cfg["tz_convert"]
        if missing_index is None:
            missing_index = data_cfg["missing_index"]
        if missing_columns is None:
            missing_columns = data_cfg["missing_columns"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if symbol_classes is None:
            symbol_classes = symbol_dict()
        if fetch_kwargs is None:
            fetch_kwargs = symbol_dict()
        if returned_kwargs is None:
            returned_kwargs = symbol_dict()
        if last_index is None:
            last_index = symbol_dict()
        if delisted is None:
            delisted = symbol_dict()

        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = dict(symbol=data)
        if symbol_oriented:
            data = cls.invert_data(data)

        data = symbol_dict(data)
        for symbol, obj in data.items():
            obj = to_pd_array(obj)
            obj = obj[~obj.index.duplicated(keep="last")]
            obj = cls.prepare_tzaware_index(obj, tz_localize=tz_localize, tz_convert=tz_convert)
            data[symbol] = obj
            if symbol not in last_index:
                last_index[symbol] = obj.index[-1]
            if symbol not in delisted:
                delisted[symbol] = False

        data = cls.align_index(data, missing=missing_index, silence_warnings=silence_warnings)
        data = cls.align_columns(data, missing=missing_columns, silence_warnings=silence_warnings)

        for symbol, obj in data.items():
            if isinstance(obj.index, pd.DatetimeIndex):
                obj.index.freq = obj.index.inferred_freq

        symbols = list(data.keys())
        wrapper = ArrayWrapper.from_obj(data[symbols[0]], **wrapper_kwargs)
        return cls(
            wrapper,
            data,
            single_symbol=single_symbol,
            symbol_classes=symbol_classes,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

    # ############# Fetching ############# #

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Fetch a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.
        If there are keyword arguments `tz_localize`, `tz_convert`, or `freq` in this dict,
        will pop them and use them to override global settings.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    @classmethod
    def try_fetch_symbol(
        cls,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        fetch_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Try to fetch a symbol using `Data.fetch_symbol`."""
        if fetch_kwargs is None:
            fetch_kwargs = {}
        try:
            out = cls.fetch_symbol(symbol, **fetch_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Symbol '{str(symbol)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    @classmethod
    def fetch(
        cls: tp.Type[DataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        symbol_classes: tp.Optional[tp.MaybeSequence[tp.Union[tp.Hashable, dict]]] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Fetch data of each symbol using `Data.fetch_symbol` and pass to `Data.from_data`.

        Iteration over symbols is done using `vectorbtpro.utils.execution.execute`.
        That is, it can be distributed and parallelized when needed.

        Args:
            symbols (hashable, sequence of hashable, or dict): One or multiple symbols.

                If provided as a dictionary, will use keys as symbols and values as keyword arguments.

                !!! note
                    Tuple is considered as a single symbol (tuple is a hashable).
            symbol_classes (symbol_dict): See `Data.symbol_classes`.

                Can be a hashable (single value), a dictionary (class names as keys and
                class values as values), or a sequence of such.

                !!! note
                    Tuple is considered as a single class (tuple is a hashable).
            tz_localize (any): See `Data.from_data`.
            tz_convert (any): See `Data.from_data`.
            missing_index (str): See `Data.from_data`.
            missing_columns (str): See `Data.from_data`.
            wrapper_kwargs (dict): See `Data.from_data`.
            skip_on_error (bool): Whether to skip the symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            execute_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.execution.execute`.
            **kwargs: Passed to `Data.fetch_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.

        For defaults, see `vectorbtpro._settings.data`.
        """
        data_cfg = cls.get_settings(key_id="base")

        fetch_kwargs = symbol_dict()
        if checks.is_hashable(symbols):
            single_symbol = True
            symbols = [symbols]
        elif isinstance(symbols, dict):
            new_symbols = []
            for symbol, symbol_fetch_kwargs in symbols.items():
                new_symbols.append(symbol)
                fetch_kwargs[symbol] = symbol_fetch_kwargs
            symbols = new_symbols
            single_symbol = False
        elif checks.is_iterable(symbols):
            symbols = list(symbols)
            single_symbol = False
        else:
            raise TypeError("Symbols must be either a hashable or a sequence of hashable")
        if symbol_classes is not None:
            if not isinstance(symbol_classes, symbol_dict):
                new_symbol_classes = symbol_dict()
                single_class = checks.is_hashable(symbol_classes) or isinstance(symbol_classes, dict)
                if single_class:
                    for symbol in symbols:
                        if isinstance(symbol_classes, dict):
                            new_symbol_classes[symbol] = symbol_classes
                        else:
                            new_symbol_classes[symbol] = {"symbol_class": symbol_classes}
                else:
                    for i, symbol in enumerate(symbols):
                        _symbol_classes = symbol_classes[i]
                        if not isinstance(_symbol_classes, dict):
                            _symbol_classes = {"symbol_class": _symbol_classes}
                        new_symbol_classes[symbol] = _symbol_classes
                symbol_classes = new_symbol_classes
        wrapper_kwargs = merge_dicts(data_cfg["wrapper_kwargs"], wrapper_kwargs)
        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]
        execute_kwargs = merge_dicts(data_cfg["execute_kwargs"], execute_kwargs)
        if not single_symbol and "show_progress" not in execute_kwargs:
            execute_kwargs["show_progress"] = True

        funcs_args = []
        func_arg_names = get_func_arg_names(cls.fetch_symbol)
        for symbol in symbols:
            symbol_fetch_kwargs = cls.select_symbol_kwargs(symbol, kwargs)
            if "silence_warnings" in func_arg_names:
                symbol_fetch_kwargs["silence_warnings"] = silence_warnings
            if symbol in fetch_kwargs:
                symbol_fetch_kwargs = merge_dicts(symbol_fetch_kwargs, fetch_kwargs[symbol])
            funcs_args.append(
                (
                    cls.try_fetch_symbol,
                    (symbol,),
                    dict(
                        skip_on_error=skip_on_error,
                        silence_warnings=silence_warnings,
                        fetch_kwargs=symbol_fetch_kwargs,
                    ),
                )
            )
            fetch_kwargs[symbol] = symbol_fetch_kwargs

        outputs = execute(funcs_args, n_calls=len(symbols), progress_desc=symbols, **execute_kwargs)

        data = symbol_dict()
        returned_kwargs = symbol_dict()
        for i, out in enumerate(outputs):
            symbol = symbols[i]
            if out is not None:
                if isinstance(out, tuple):
                    _data = out[0]
                    _returned_kwargs = out[1]
                else:
                    _data = out
                    _returned_kwargs = {}
                _data = to_any_array(_data)
                if tz_localize is None:
                    tz_localize = _returned_kwargs.pop("tz_localize", None)
                if tz_convert is None:
                    tz_convert = _returned_kwargs.pop("tz_convert", None)
                if wrapper_kwargs.get("freq", None) is None:
                    wrapper_kwargs["freq"] = _returned_kwargs.pop("freq", None)
                if _data.size == 0:
                    if not silence_warnings:
                        warnings.warn(
                            f"Symbol '{str(symbol)}' returned an empty array. Skipping.",
                            stacklevel=2,
                        )
                else:
                    data[symbol] = _data
                    returned_kwargs[symbol] = _returned_kwargs

        if len(data) == 0:
            raise ValueError("No symbols could be fetched")

        # Create new instance from data
        return cls.from_data(
            data,
            single_symbol=single_symbol,
            symbol_classes=symbol_classes,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            wrapper_kwargs=wrapper_kwargs,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            silence_warnings=silence_warnings,
        )

    # ############# Updating ############# #

    def update_symbol(
        self,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Update a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def try_update_symbol(
        self,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        update_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Try to update a symbol using `Data.update_symbol`."""
        if update_kwargs is None:
            update_kwargs = {}
        try:
            out = self.update_symbol(symbol, **update_kwargs)
            if out is None:
                if not silence_warnings:
                    warnings.warn(
                        f"Symbol '{str(symbol)}' returned None. Skipping.",
                        stacklevel=2,
                    )
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                    stacklevel=2,
                )
        return None

    def update(
        self: DataT,
        *,
        concat: bool = True,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Fetch additional data of each symbol using `Data.update_symbol`.

        Args:
            concat (bool): Whether to concatenate existing and updated/new data.
            skip_on_error (bool): Whether to skip the symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.update_symbol` if accepted by `Data.fetch_symbol`.
            execute_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.execution.execute`.
            **kwargs: Passed to `Data.update_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.

        !!! note
            Returns a new `Data` instance instead of changing the data in place.
        """
        data_cfg = self.get_settings(key_id="base")

        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]
        execute_kwargs = merge_dicts(data_cfg["execute_kwargs"], execute_kwargs)
        if "show_progress" not in execute_kwargs:
            execute_kwargs["show_progress"] = False
        func_arg_names = get_func_arg_names(self.fetch_symbol)
        if "show_progress" in func_arg_names and "show_progress" not in kwargs:
            kwargs["show_progress"] = False

        funcs_args = []
        symbol_indices = []
        for i, symbol in enumerate(self.symbols):
            if not self.delisted.get(symbol, False):
                symbol_update_kwargs = self.select_symbol_kwargs(symbol, kwargs)
                if "silence_warnings" in func_arg_names:
                    symbol_update_kwargs["silence_warnings"] = silence_warnings
                funcs_args.append(
                    (
                        self.try_update_symbol,
                        (symbol,),
                        dict(
                            skip_on_error=skip_on_error,
                            silence_warnings=silence_warnings,
                            update_kwargs=symbol_update_kwargs,
                        ),
                    )
                )
                symbol_indices.append(i)

        outputs = execute(funcs_args, n_calls=len(self.symbols), progress_desc=self.symbols, **execute_kwargs)

        new_data = symbol_dict()
        new_last_index = symbol_dict()
        new_returned_kwargs = symbol_dict()
        i = 0
        for symbol, obj in self.data.items():
            if self.delisted.get(symbol, False):
                out = None
            else:
                out = outputs[i]
                i += 1
            skip_symbol = False
            if out is not None:
                if isinstance(out, tuple):
                    new_obj = out[0]
                    new_returned_kwargs[symbol] = out[1]
                else:
                    new_obj = out
                    new_returned_kwargs[symbol] = {}
                new_obj = to_any_array(new_obj)
                if new_obj.size == 0:
                    if not silence_warnings:
                        warnings.warn(
                            f"Symbol '{str(symbol)}' returned an empty array. Skipping.",
                            stacklevel=2,
                        )
                    skip_symbol = True
                else:
                    if not checks.is_pandas(new_obj):
                        new_obj = to_pd_array(new_obj)
                        new_obj.index = pd.RangeIndex(
                            start=obj.index[-1],
                            stop=obj.index[-1] + new_obj.shape[0],
                            step=1,
                        )
                    new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
                    new_obj = self.prepare_tzaware_index(
                        new_obj,
                        tz_localize=self.tz_localize,
                        tz_convert=self.tz_convert,
                    )
                    new_data[symbol] = new_obj
                    if len(new_obj.index) > 0:
                        new_last_index[symbol] = new_obj.index[-1]
                    else:
                        new_last_index[symbol] = self.last_index[symbol]
            else:
                skip_symbol = True
            if skip_symbol:
                new_data[symbol] = obj.iloc[0:0]
                new_last_index[symbol] = self.last_index[symbol]

        # Get the last index in the old data from where the new data should begin
        from_index = None
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                index = new_obj.index[0]
            else:
                continue
            if from_index is None or index < from_index:
                from_index = index
        if from_index is None:
            if not silence_warnings:
                warnings.warn(
                    f"None of the symbols have been updated",
                    stacklevel=2,
                )
            return self.copy()

        # Concatenate the updated old data and the new data
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                to_index = new_obj.index[0]
            else:
                to_index = None
            obj = self.data[symbol]
            if checks.is_frame(obj) and checks.is_frame(new_obj):
                shared_columns = obj.columns.intersection(new_obj.columns)
                obj = obj[shared_columns]
                new_obj = new_obj[shared_columns]
            elif checks.is_frame(new_obj):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            elif checks.is_frame(obj):
                if new_obj.name is not None:
                    obj = obj[new_obj.name]
                else:
                    obj = obj[0]
            obj = obj.loc[from_index:to_index]
            new_obj = pd.concat((obj, new_obj), axis=0)
            new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
            new_data[symbol] = new_obj

        # Align the index and columns in the new data
        new_data = self.align_index(new_data, missing=self.missing_index, silence_warnings=silence_warnings)
        new_data = self.align_columns(new_data, missing=self.missing_columns, silence_warnings=silence_warnings)

        # Align the columns and data type in the old and new data
        for symbol, new_obj in new_data.items():
            obj = self.data[symbol]
            if checks.is_frame(obj) and checks.is_frame(new_obj):
                new_obj = new_obj[obj.columns]
            elif checks.is_frame(new_obj):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            if checks.is_frame(obj):
                new_obj = new_obj.astype(obj.dtypes)
            else:
                new_obj = new_obj.astype(obj.dtype)
            new_data[symbol] = new_obj

        if not concat:
            # Do not concatenate with the old data
            for symbol, new_obj in new_data.items():
                if isinstance(new_obj.index, pd.DatetimeIndex):
                    new_obj.index.freq = new_obj.index.inferred_freq
            new_index = new_data[self.symbols[0]].index
            return self.replace(
                wrapper=self.wrapper.replace(index=new_index),
                data=new_data,
                returned_kwargs=new_returned_kwargs,
                last_index=new_last_index,
            )

        # Append the new data to the old data
        for symbol, new_obj in new_data.items():
            obj = self.data[symbol]
            obj = obj.loc[:from_index]
            if obj.index[-1] == from_index:
                obj = obj.iloc[:-1]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if isinstance(new_obj.index, pd.DatetimeIndex):
                new_obj.index.freq = new_obj.index.inferred_freq
            new_data[symbol] = new_obj

        new_index = new_data[self.symbols[0]].index

        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),
            data=new_data,
            returned_kwargs=new_returned_kwargs,
            last_index=new_last_index,
        )

    # ############# Getting ############# #

    def get_symbol_wrapper(
        self,
        symbols: tp.Optional[tp.Symbols] = None,
        level_name: str = "symbol",
        index_stack_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ArrayWrapper:
        """Get wrapper where columns are symbols."""
        if index_stack_kwargs is None:
            index_stack_kwargs = {}
        if symbols is None:
            symbols = self.symbols
            ndim = 1 if self.single_symbol else 2
        else:
            if isinstance(symbols, list):
                ndim = 2
            else:
                symbols = [symbols]
                ndim = 1
        symbol_columns = pd.Index(symbols, name=level_name)
        symbol_classes = []
        all_have_symbol_classes = True
        for symbol in symbols:
            if symbol in self.symbol_classes:
                classes = self.symbol_classes[symbol]
                if len(classes) > 0:
                    symbol_classes.append(classes)
                else:
                    all_have_symbol_classes = False
            else:
                all_have_symbol_classes = False
        if len(symbol_classes) > 0 and not all_have_symbol_classes:
            raise ValueError("Some symbols have symbol classes while others not")
        if len(symbol_classes) > 0:
            symbol_classes_frame = pd.DataFrame(symbol_classes)
            if len(symbol_classes_frame.columns) == 1:
                symbol_classes_columns = pd.Index(symbol_classes_frame.iloc[:, 0])
            else:
                symbol_classes_columns = pd.MultiIndex.from_frame(symbol_classes_frame)
            symbol_columns = stack_indexes((symbol_classes_columns, symbol_columns), **index_stack_kwargs)
        return self.wrapper.replace(
            columns=symbol_columns,
            ndim=ndim,
            grouper=None,
            **kwargs,
        )

    @property
    def symbol_wrapper(self):
        """`Data.get_symbol_wrapper` with default arguments."""
        return self.get_symbol_wrapper()

    def concat(
        self,
        symbols: tp.Optional[tp.Symbols] = None,
        level_name: str = "symbol",
        index_stack_kwargs: tp.KwargsLike = None,
    ) -> dict:
        """Return a dict of Series/DataFrames with symbols as columns, keyed by column name."""
        symbol_wrapper = self.get_symbol_wrapper(
            symbols=symbols,
            level_name=level_name,
            index_stack_kwargs=index_stack_kwargs,
        )
        if symbols is None:
            symbols = self.symbols

        new_data = {}
        first_data = self.data[symbols[0]]
        if self.single_symbol:
            if checks.is_series(first_data):
                new_data[first_data.name] = symbol_wrapper.wrap(first_data.values, zero_to_none=False)
            else:
                for c in first_data.columns:
                    new_data[c] = symbol_wrapper.wrap(first_data[c].values, zero_to_none=False)
        else:
            if checks.is_series(first_data):
                columns = pd.Index([first_data.name])
            else:
                columns = first_data.columns
            for c in columns:
                col_data = []
                for s in symbols:
                    if checks.is_series(self.data[s]):
                        col_data.append(self.data[s].values)
                    else:
                        col_data.append(self.data[s][c].values)
                new_data[c] = symbol_wrapper.wrap(np.column_stack(col_data), zero_to_none=False)

        return new_data

    def get(
        self,
        columns: tp.Optional[tp.Union[tp.Label, tp.Labels]] = None,
        symbols: tp.Union[None, tp.Symbol, tp.Symbols] = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Get one or more columns of one or more symbols of data.

        If one symbol, returns data for that symbol. If multiple symbols, performs concatenation
        first and returns a DataFrame if one column and a tuple of DataFrames if a list of columns passed."""
        if symbols is None:
            single_symbol = self.single_symbol
            symbols = self.symbols
        else:
            if isinstance(symbols, list):
                single_symbol = False
            else:
                single_symbol = True
                symbols = [symbols]

        if single_symbol:
            if columns is None:
                return self.data[symbols[0]]
            return self.data[symbols[0]][columns]

        concat_data = self.concat(symbols=symbols, **kwargs)
        if len(concat_data) == 1:
            return tuple(concat_data.values())[0]
        if columns is not None:
            if isinstance(columns, list):
                return tuple([concat_data[c] for c in columns])
            return concat_data[columns]
        return tuple(concat_data.values())

    def get_column_index(self, label: tp.Hashable) -> tp.MaybeList[int]:
        """Return one or more indexes of the columns that match the label."""

        def _prepare_label(x):
            if isinstance(x, tuple):
                return tuple([_prepare_label(_x) for _x in x])
            if isinstance(x, str):
                return x.lower().strip().replace(" ", "_")
            return x

        label = _prepare_label(label)

        found_indices = []
        for i, c in enumerate(self.wrapper.columns):
            c = _prepare_label(c)
            if label == c:
                found_indices.append(i)
        if len(found_indices) == 0:
            return -1
        if len(found_indices) == 1:
            return found_indices[0]
        return found_indices

    def get_column(self, idx_or_label: tp.Hashable, raise_error: bool = False) -> tp.Optional[tp.SeriesFrame]:
        """Get column(s) that match a column index or label."""
        if checks.is_int(idx_or_label):
            return self.get(columns=self.wrapper.columns[idx_or_label])
        column_idx = self.get_column_index(idx_or_label)
        if column_idx == -1:
            if raise_error:
                raise ValueError(f"Column(s) {idx_or_label} not found")
            return None
        return self.get(columns=self.wrapper.columns[column_idx])

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open."""
        return self.get_column("Open")

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High."""
        return self.get_column("High")

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low."""
        return self.get_column("Low")

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close."""
        return self.get_column("Close")

    @property
    def volume(self) -> tp.Optional[tp.SeriesFrame]:
        """Volume."""
        return self.get_column("Volume")

    @property
    def trade_count(self) -> tp.Optional[tp.SeriesFrame]:
        """Trade count."""
        return self.get_column("Trade count")

    @property
    def vwap(self) -> tp.Optional[tp.SeriesFrame]:
        """VWAP."""
        return self.get_column("VWAP")

    @property
    def hlc3(self) -> tp.Optional[tp.SeriesFrame]:
        """HLC/3."""
        high = self.get_column("High", raise_error=True)
        low = self.get_column("Low", raise_error=True)
        close = self.get_column("Close", raise_error=True)
        return (high + low + close) / 3

    @property
    def ohlc4(self) -> tp.Optional[tp.SeriesFrame]:
        """OHLC/4."""
        open = self.get_column("Open", raise_error=True)
        high = self.get_column("High", raise_error=True)
        low = self.get_column("Low", raise_error=True)
        close = self.get_column("Close", raise_error=True)
        return (open + high + low + close) / 4

    @property
    def returns_acc(self) -> tp.Optional[tp.SeriesFrame]:
        """Return accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`."""
        return ReturnsAccessor.from_value(
            self.get_column("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
        )

    @property
    def returns(self) -> tp.Optional[tp.SeriesFrame]:
        """Returns."""
        return ReturnsAccessor.from_value(
            self.get_column("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
        )

    @property
    def log_returns(self) -> tp.Optional[tp.SeriesFrame]:
        """Log returns."""
        return ReturnsAccessor.from_value(
            self.get_column("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
            log_returns=True,
        )

    @property
    def daily_returns(self) -> tp.Optional[tp.SeriesFrame]:
        """Daily returns."""
        return ReturnsAccessor.from_value(
            self.get_column("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
        ).daily()

    @property
    def daily_log_returns(self) -> tp.Optional[tp.SeriesFrame]:
        """Daily log returns."""
        return ReturnsAccessor.from_value(
            self.get_column("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            log_returns=True,
        ).daily()

    # ############# Selecting ############# #

    def select(self: DataT, symbols: tp.Union[tp.Symbol, tp.Symbols], **kwargs) -> DataT:
        """Create a new `Data` instance with one or more symbols from this instance."""
        if isinstance(symbols, list):
            single_symbol = False
        else:
            single_symbol = True
            symbols = [symbols]
        return self.replace(
            data=symbol_dict({k: v for k, v in self.data.items() if k in symbols}),
            single_symbol=single_symbol,
            symbol_classes=symbol_dict({k: v for k, v in self.symbol_classes.items() if k in symbols}),
            fetch_kwargs=symbol_dict({k: v for k, v in self.fetch_kwargs.items() if k in symbols}),
            returned_kwargs=symbol_dict(
                {k: v for k, v in self.returned_kwargs.items() if k in symbols},
            ),
            last_index=symbol_dict({k: v for k, v in self.last_index.items() if k in symbols}),
            delisted=symbol_dict({k: v for k, v in self.delisted.items() if k in symbols}),
            **kwargs,
        )

    # ############# Renaming ############# #

    def rename(self: DataT, rename: tp.Dict[tp.Hashable, tp.Hashable]) -> DataT:
        """Rename symbols using `rename` dict that maps old symbols and new symbols."""
        return self.replace(
            data={rename.get(k, k): v for k, v in self.data.items()},
            symbol_classes={rename.get(k, k): v for k, v in self.symbol_classes.items()},
            fetch_kwargs={rename.get(k, k): v for k, v in self.fetch_kwargs.items()},
            returned_kwargs={rename.get(k, k): v for k, v in self.returned_kwargs.items()},
            last_index={rename.get(k, k): v for k, v in self.last_index.items()},
            delisted={rename.get(k, k): v for k, v in self.delisted.items()},
        )

    # ############# Merging ############# #

    @classmethod
    def merge(
        cls: tp.Type[DataT],
        *datas: DataT,
        rename: tp.Optional[tp.Dict[tp.Hashable, tp.Hashable]] = None,
        **kwargs,
    ) -> DataT:
        """Merge multiple `Data` instances.

        Can merge both different symbols and different data across overlapping symbols.
        Data is overridden in the order as provided in `datas`."""
        if len(datas) == 1:
            datas = datas[0]
        datas = list(datas)

        data = symbol_dict()
        single_symbol = True
        symbol_classes = symbol_dict()
        fetch_kwargs = symbol_dict()
        returned_kwargs = symbol_dict()
        last_index = symbol_dict()
        delisted = symbol_dict()
        for instance in datas:
            if not instance.single_symbol:
                single_symbol = False
            for s in instance.symbols:
                if rename is None:
                    new_s = s
                else:
                    new_s = rename[s]
                if new_s in data:
                    data[new_s] = instance.data[s].combine_first(data[new_s])
                else:
                    data[new_s] = instance.data[s]
                if s in instance.symbol_classes:
                    symbol_classes[new_s] = instance.symbol_classes[s]
                if s in instance.fetch_kwargs:
                    fetch_kwargs[new_s] = instance.fetch_kwargs[s]
                if s in instance.returned_kwargs:
                    returned_kwargs[new_s] = instance.returned_kwargs[s]
                if s in instance.last_index:
                    last_index[new_s] = instance.last_index[s]
                if s in instance.delisted:
                    delisted[new_s] = instance.delisted[s]

        return cls.from_data(
            data=data,
            single_symbol=single_symbol,
            symbol_classes=symbol_classes,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
            **kwargs,
        )

    # ############# Saving ############# #

    def to_csv(
        self,
        dir_path: tp.Union[tp.PathLike, symbol_dict] = ".",
        ext: tp.Union[str, symbol_dict] = "csv",
        path_or_buf: tp.Optional[tp.Union[str, symbol_dict]] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, symbol_dict] = None,
        **kwargs,
    ) -> None:
        """Save data into CSV file(s).

        Any argument can be provided per symbol using `symbol_dict`.

        Each symbol gets saved to a separate file, that's why the first argument is the path
        to the directory, not file! If there's only one file, you can specify the file path via
        `path_or_buf`. If there are multiple files, use the same argument but wrap the multiple paths
        with `symbol_dict`."""
        for k, v in self.data.items():
            if path_or_buf is None:
                if isinstance(dir_path, symbol_dict):
                    _dir_path = dir_path[k]
                else:
                    _dir_path = dir_path
                _dir_path = Path(_dir_path)
                if isinstance(ext, symbol_dict):
                    _ext = ext[k]
                else:
                    _ext = ext
                _path_or_buf = str(Path(_dir_path) / f"{k}.{_ext}")
            elif isinstance(path_or_buf, symbol_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf

            _kwargs = self.select_symbol_kwargs(k, kwargs)
            sep = _kwargs.pop("sep", None)
            if isinstance(_path_or_buf, (str, Path)):
                _path_or_buf = Path(_path_or_buf)
                if isinstance(mkdir_kwargs, symbol_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_symbol_kwargs(k, mkdir_kwargs)
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                if _path_or_buf.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path_or_buf.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
            if sep is None:
                sep = ","
            v.to_csv(path_or_buf=_path_or_buf, sep=sep, **_kwargs)

    def to_hdf(
        self,
        file_path: tp.Union[tp.PathLike, symbol_dict] = ".",
        key: tp.Optional[tp.Union[str, symbol_dict]] = None,
        path_or_buf: tp.Optional[tp.Union[str, symbol_dict]] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, symbol_dict] = None,
        format: str = "table",
        **kwargs,
    ) -> None:
        """Save data into an HDF file.

        Any argument can be provided per symbol using `symbol_dict`.

        If `file_path` exists, and it's a directory, will create inside it a file named
        after this class. This won't work with directories that do not exist, otherwise
        they could be confused with file names."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        for k, v in self.data.items():
            if path_or_buf is None:
                if isinstance(file_path, symbol_dict):
                    _file_path = file_path[k]
                else:
                    _file_path = file_path
                _file_path = Path(_file_path)
                if _file_path.exists() and _file_path.is_dir():
                    _file_path /= type(self).__name__ + ".h5"
                _dir_path = _file_path.parent
                if isinstance(mkdir_kwargs, symbol_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_symbol_kwargs(k, mkdir_kwargs)
                check_mkdir(_dir_path, **_mkdir_kwargs)
                _path_or_buf = str(_file_path)
            elif isinstance(path_or_buf, symbol_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf
            if key is None:
                _key = str(k)
            elif isinstance(key, symbol_dict):
                _key = key[k]
            else:
                _key = key
            _kwargs = self.select_symbol_kwargs(k, kwargs)
            v.to_hdf(path_or_buf=_path_or_buf, key=_key, format=format, **_kwargs)

    # ############# Loading ############# #

    @classmethod
    def from_csv(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Use `CSVData` to load data from CSV and switch the class back to this class.

        Use `fetch_kwargs` to provide keyword arguments that were originally used in fetching."""
        from vectorbtpro.data.custom import CSVData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = CSVData.fetch(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    @classmethod
    def from_hdf(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Use `HDFData` to load data from HDF and switch the class back to this class.

        Use `fetch_kwargs` to provide keyword arguments that were originally used in fetching."""
        from vectorbtpro.data.custom import HDFData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        if len(args) == 0 and "symbols" not in kwargs:
            args = (cls.__name__ + ".h5",)
        data = HDFData.fetch(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    # ############# Transforming ############# #

    def transform(self: DataT, transform_func: tp.Callable, *args, **kwargs) -> DataT:
        """Transform data.

        Concatenates all the data into a single DataFrame and calls `transform_func` on it.
        Then, splits the data by symbol and builds a new `Data` instance."""
        if self.single_symbol:
            concat_data = self.data[self.symbols[0]]
        else:
            concat_data = pd.concat(self.data.values(), axis=1, keys=pd.Index(self.symbols, name="symbol"))
        new_concat_data = transform_func(concat_data, *args, **kwargs)
        if self.single_symbol:
            new_wrapper = ArrayWrapper.from_obj(new_concat_data)
            new_data = symbol_dict({self.symbols[0]: new_concat_data})
        else:
            new_wrapper = None
            new_data = symbol_dict()
            for k in self.symbols:
                if isinstance(new_concat_data.columns, pd.MultiIndex):
                    new_v = new_concat_data.xs(k, axis=1, level="symbol")
                else:
                    if len(self.wrapper.columns) == 1 and self.wrapper.columns[0] != 0:
                        new_v = new_concat_data[k].rename(self.wrapper.columns[0])
                    else:
                        new_v = new_concat_data[k].rename(None)
                _new_wrapper = ArrayWrapper.from_obj(new_v)
                if new_wrapper is None:
                    new_wrapper = _new_wrapper
                else:
                    if not checks.is_index_equal(new_wrapper.columns, _new_wrapper.columns):
                        raise ValueError("Transformed symbols must have the same columns")
                new_data[k] = new_v
        return self.replace(
            wrapper=new_wrapper,
            data=new_data,
        )

    def resample(self: DataT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> DataT:
        """Perform resampling on `Data` based on `Data.column_config`.

        Columns "open", "high", "low", "close", "volume", "trade count", and "vwap" (case-insensitive)
        are recognized and resampled automatically.

        Looks for `resample_func` of each column in `Data.column_config`. The function must
        accept the `Data` instance, object, and resampler."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        new_data = symbol_dict()
        for k, v in self.data.items():
            if checks.is_series(v):
                columns = [v.name]
            else:
                columns = v.columns
            new_v = []
            for c in columns:
                if checks.is_series(v):
                    obj = v
                else:
                    obj = v[c]
                resample_func = self.column_config.get(c, {}).get("resample_func", None)
                if resample_func is not None:
                    if isinstance(resample_func, str):
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], resample_func))
                    else:
                        new_v.append(resample_func(self, obj, wrapper_meta["resampler"]))
                else:
                    if isinstance(c, str) and c.lower() == "open":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.first_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "high":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.max_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "low":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.min_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "close":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.last_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "volume":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.sum_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "trade count":
                        new_v.append(
                            obj.vbt.resample_apply(
                                wrapper_meta["resampler"],
                                generic_nb.sum_reduce_nb,
                                wrap_kwargs=dict(dtype=int),
                            )
                        )
                    elif isinstance(c, str) and c.lower() == "vwap":
                        volume_obj = None
                        for c2 in columns:
                            if isinstance(c2, str) and c2.lower() == "volume":
                                volume_obj = v[c2]
                        if volume_obj is None:
                            raise ValueError("Volume is required to resample VWAP")
                        new_v.append(
                            pd.DataFrame.vbt.resample_apply(
                                wrapper_meta["resampler"],
                                generic_nb.wmean_range_reduce_meta_nb,
                                to_2d_array(obj),
                                to_2d_array(volume_obj),
                                wrapper=self.wrapper[c],
                            )
                        )
                    else:
                        raise ValueError(f"Cannot resample column '{c}'. Specify resample_func in column_config.")
            if checks.is_series(v):
                new_v = new_v[0]
            else:
                new_v = pd.concat(new_v, axis=1)
            new_data[k] = new_v
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    # ############# Running ############# #

    def run(
        self,
        func: tp.MaybeIterable[tp.Union[str, tp.Callable]],
        *args,
        on_indices: tp.Union[None, slice, tp.Sequence] = None,
        on_dates: tp.Union[None, slice, tp.Sequence] = None,
        on_symbols: tp.Union[None, tp.Symbol, tp.Symbols] = None,
        pass_as_first: bool = False,
        rename_args: tp.DictLike = None,
        unpack: tp.Union[bool, str] = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Run a function on data.

        Looks into the signature of the function and searches for arguments with the name `data` or
        those found among columns or attributes.

        For example, the argument `open` will be substituted by `Data.open`.

        `func` can be one of the following:

        * Location to compute all indicators from. See `vectorbtpro.indicators.factory.IndicatorFactory.list_locations`.
        * Indicator name. See `vectorbtpro.indicators.factory.IndicatorFactory.get_indicator`.
        * Simulation method. See `vectorbtpro.portfolio.base.Portfolio`.
        * Any callable object
        * Iterable with any of the above. Will be stacked as columns into a DataFrame.

        Use `on_indices` (using `iloc`), `on_dates` (using `loc`), and `on_symbols` (using `Data.select`)
        to filter data. For example, to filter a date range, pass `slice(start_date, end_date)`.

        Use `rename_args` to rename arguments. For example, in `vectorbtpro.portfolio.base.Portfolio`,
        data can be passed instead of `close`.

        Set `unpack` to True, "dict", or "frame" to use
        `vectorbtpro.indicators.factory.IndicatorBase.unpack`,
        `vectorbtpro.indicators.factory.IndicatorBase.to_dict`, and
        `vectorbtpro.indicators.factory.IndicatorBase.to_frame` respectively.

        Any argument in `*args` and `**kwargs` can be wrapped with `run_func_dict` to specify
        the value per function name or index when `func` is iterable."""
        from vectorbtpro.indicators.factory import IndicatorBase, IndicatorFactory
        from vectorbtpro.portfolio.base import Portfolio

        _self = self
        if on_indices is not None:
            _self = _self.iloc[on_indices]
        if on_dates is not None:
            _self = _self.loc[on_dates]
        if on_symbols is not None:
            _self = _self.select(on_symbols)

        if pass_as_first:
            return func(_self, *args, **kwargs)

        def _select_func_args(i, func_name, args) -> tuple:
            _args = ()
            for v in args:
                if isinstance(v, run_func_dict):
                    if func_name in v:
                        _args += (v[func_name],)
                    elif i in v:
                        _args += (v[i],)
                    elif "_def" in v:
                        _args += (v["_def"],)
                else:
                    _args += (v,)
            return _args

        def _select_func_kwargs(i, func_name, kwargs) -> dict:
            _kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, run_func_dict):
                    if func_name in v:
                        _kwargs[k] = v[func_name]
                    elif i in v:
                        _kwargs[k] = v[i]
                    elif "_def" in v:
                        _kwargs[k] = v["_def"]
                else:
                    _kwargs[k] = v
            return _kwargs

        if checks.is_iterable(func) and not isinstance(func, str):
            outputs = []
            keys = []
            for i, f in enumerate(func):
                if callable(f):
                    f_name = f.__name__
                else:
                    if isinstance(f, str):
                        f_name = f.lower().strip().replace(":", "_")
                    else:
                        f_name = f
                try:
                    out = _self.run(
                        f,
                        *_select_func_args(i, f_name, args),
                        rename_args=rename_args,
                        silence_warnings=silence_warnings,
                        unpack=unpack,
                        **_select_func_kwargs(i, f_name, kwargs),
                    )
                    if isinstance(out, pd.Series):
                        out = out.to_frame()
                    if isinstance(out, IndicatorBase):
                        out = out.to_frame()
                    outputs.append(out)
                    keys.append(str(f_name))
                except Exception as e:
                    if not silence_warnings:
                        warnings.warn(f_name + ": " + str(e), stacklevel=2)
            return pd.concat(outputs, keys=pd.Index(keys, name="run_func"), axis=1)
        if isinstance(func, str):
            func = func.lower().strip()
            if func.startswith("from_") and getattr(Portfolio, func):
                func = getattr(Portfolio, func)
                pf = func(_self, *args, **kwargs)
                if isinstance(pf, Portfolio) and unpack:
                    raise ValueError("Portfolio cannot be unpacked")
                return pf
            locations = IndicatorFactory.list_locations()
            if func in locations or (func.endswith("_all") and func[:-4] in locations):
                if func.endswith("_all"):
                    func = func[:-4]
                    prepend_location = True
                else:
                    prepend_location = False
                return _self.run(
                    IndicatorFactory.list_indicators(func, prepend_location=prepend_location),
                    *args,
                    rename_args=rename_args,
                    silence_warnings=silence_warnings,
                    **kwargs,
                )
            func = IndicatorFactory.get_indicator(func)
        if isinstance(func, type) and issubclass(func, IndicatorBase):
            func = func.run

        with_kwargs = dict()
        for arg_name in get_func_arg_names(func):
            real_arg_name = arg_name
            if rename_args is not None:
                if arg_name in rename_args:
                    arg_name = rename_args[arg_name]
            if real_arg_name not in kwargs:
                if arg_name == "data":
                    with_kwargs[real_arg_name] = _self
                elif arg_name == "wrapper":
                    with_kwargs[real_arg_name] = _self.symbol_wrapper
                elif arg_name in ("input_shape", "shape"):
                    with_kwargs[real_arg_name] = _self.shape
                elif arg_name in ("target_shape", "shape_2d"):
                    with_kwargs[real_arg_name] = _self.shape_2d
                elif arg_name in ("input_index", "index"):
                    with_kwargs[real_arg_name] = _self.index
                elif arg_name in ("input_columns", "columns"):
                    with_kwargs[real_arg_name] = _self.columns
                elif arg_name == "freq":
                    with_kwargs[real_arg_name] = _self.freq
                elif arg_name == "hlc3":
                    with_kwargs[real_arg_name] = _self.hlc3
                elif arg_name == "ohlc4":
                    with_kwargs[real_arg_name] = _self.ohlc4
                elif arg_name == "returns":
                    with_kwargs[real_arg_name] = _self.returns
                else:
                    column_idx = _self.get_column_index(arg_name)
                    if column_idx != -1:
                        with_kwargs[real_arg_name] = _self.get_column(column_idx)
        new_args, new_kwargs = extend_args(func, args, kwargs, **with_kwargs)
        out = func(*new_args, **new_kwargs)
        if isinstance(out, IndicatorBase):
            if isinstance(unpack, bool):
                if unpack:
                    out = out.unpack()
            elif isinstance(unpack, str) and unpack.lower() == "dict":
                out = out.to_dict()
            elif isinstance(unpack, str) and unpack.lower() == "frame":
                out = out.to_frame()
            else:
                raise ValueError(f"Invalid option unpack='{unpack}'")
        return out

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.data`."""
        data_stats_cfg = self.get_settings(key_id="base")["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), data_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Start",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end=dict(
                title="End",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_symbols=dict(
                title="Total Symbols",
                calc_func=lambda self: len(self.symbols),
                agg_func=None,
                tags="data",
            ),
            last_index=dict(
                title="Last Index",
                calc_func="last_index",
                agg_func=None,
                tags="data",
            ),
            delisted=dict(
                title="Delisted",
                calc_func="delisted",
                agg_func=None,
                tags="data",
            ),
            null_counts=dict(
                title="Null Counts",
                calc_func=lambda self, group_by: {
                    symbol: obj.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by)
                    for symbol, obj in self.data.items()
                },
                agg_func=lambda x: x.sum(),
                tags="data",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        column_names: tp.KwargsLike = None,
        plot_volume: tp.Optional[bool] = None,
        base: tp.Optional[float] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot either one column of multiple symbols, or OHLC(V) of one symbol.

        Args:
            column (str): Name of the column to plot.
            symbol (str): Symbol to plot.
            column_names (sequence of str): Dictionary mapping the column names to OHLCV.

                Applied only if OHLC(V) is plotted.
            plot_volume (bool): Whether to plot volume beneath.

                Applied only if OHLC(V) is plotted.
            base (float): Rebase all series of a column to a given initial base.

                !!! note
                    The column must contain prices.

                Applied only if lines are plotted.
            kwargs (dict): Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`
                for lines and to `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot` for OHLC(V).

        Usage:
            * Plot the lines of one column across all symbols:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> start = '2021-01-01 UTC'  # crypto is in UTC
            >>> end = '2021-06-01 UTC'
            >>> data = vbt.YFData.fetch(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
            ```

            [=100% "100%"]{: .candystripe}

            ```pycon
            >>> data.plot(column='Close', base=1).show()
            ```

            * Plot OHLC(V) of one symbol (only if data contains the respective columns):

            ![](/assets/images/api/data_plot.svg){: .iimg loading=lazy }

            ```pycon
            >>> data.plot(symbol='BTC-USD').show()
            ```

            ![](/assets/images/api/data_plot_ohlcv.svg){: .iimg loading=lazy }
        """
        if column is None:
            first_data = self.data[self.symbols[0]]
            if checks.is_frame(first_data):
                ohlc = first_data.vbt.ohlcv(column_names=column_names).ohlc
                if ohlc is not None and len(ohlc.columns) == 4:
                    if symbol is None:
                        if self.single_symbol or len(self.symbols) == 1:
                            symbol = self.symbols[0]
                        else:
                            raise ValueError("Only one symbol is allowed. Use indexing or symbol argument.")
                    return (
                        self.get(
                            symbols=symbol,
                        )
                        .vbt.ohlcv(
                            column_names=column_names,
                        )
                        .plot(
                            plot_volume=plot_volume,
                            **kwargs,
                        )
                    )
        self_col = self.select_col(column=column, group_by=False)
        if symbol is not None:
            self_col = self_col.select(symbol)
        data = self_col.get()
        if base is not None:
            data = data.vbt.rebase(base)
        return data.vbt.lineplot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.data`."""
        data_plots_cfg = self.get_settings(key_id="base")["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), data_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=RepEval(
                """
                if symbols is None:
                    symbols = self.symbols
                if not isinstance(symbols, list):
                    symbols = [symbols]
                [
                    dict(
                        check_is_not_grouped=True,
                        plot_func="plot",
                        plot_volume=False,
                        symbol=s,
                        title=s,
                        pass_add_trace_kwargs=True,
                        xaxis_kwargs=dict(rangeslider_visible=False, showgrid=True),
                        yaxis_kwargs=dict(showgrid=True),
                        tags="data",
                    )
                    for s in symbols
                ]""",
                context=dict(symbols=None),
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_column_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build column config documentation."""
        if source_cls is None:
            source_cls = Data
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "column_config").__doc__)).substitute(
            {"column_config": cls.column_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_column_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Data.column_config`."""
        __pdoc__[cls.__name__ + ".column_config"] = cls.build_column_config_doc(source_cls=source_cls)


Data.override_column_config_doc(__pdoc__)
Data.override_metrics_doc(__pdoc__)
Data.override_subplots_doc(__pdoc__)
