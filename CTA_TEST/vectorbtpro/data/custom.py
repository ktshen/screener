# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Custom data source classes.

!!! note
    Use absolute start and end dates instead of relative ones when fetching multiple
    symbols of data: some symbols may take a considerable amount of time to fetch
    such that they may shift the time period for the symbols coming next.

    This happens when relative dates are parsed in `vectorbtpro.data.base.Data.fetch_symbol`
    instead of parsing them once and for all symbols in `vectorbtpro.data.base.Data.fetch`."""

import time
import traceback
import warnings
from functools import wraps, lru_cache, partial
from pathlib import Path, PurePath
from glob import glob
import re
import requests
import urllib.parse

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, broadcast_array_to
from vectorbtpro.data import nb
from vectorbtpro.data.base import Data, symbol_dict
from vectorbtpro.data.tv import TVClient
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.ohlcv import nb as ohlcv_nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.datetime_ import (
    to_timestamp,
    to_tzaware_timestamp,
    to_naive_timestamp,
    to_tzaware_datetime,
    datetime_to_ms,
    split_freq_str,
    prepare_freq,
)
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.parsing import glob2re, get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.template import substitute_templates

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from binance.client import Client as BinanceClientT
except ImportError:
    BinanceClientT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from ccxt.base.exchange import Exchange as CCXTExchangeT
except ImportError:
    CCXTExchangeT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from alpaca.common.rest import RESTClient as AlpacaClientT
except ImportError:
    AlpacaClientT = tp.Any
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from polygon import RESTClient as PolygonClientT
except ImportError:
    PolygonClientT = tp.Any

__all__ = [
    "CustomData",
    "SyntheticData",
    "RandomData",
    "RandomOHLCData",
    "GBMData",
    "GBMOHLCData",
    "LocalData",
    "CSVData",
    "HDFData",
    "RemoteData",
    "YFData",
    "BinanceData",
    "CCXTData",
    "AlpacaData",
    "PolygonData",
    "AVData",
    "NDLData",
    "TVData",
]

__pdoc__ = {}


class CustomData(Data):
    """Data class for fetching custom data."""

    _setting_keys: tp.SettingsKeys = dict(custom=None)

    @classmethod
    def get_custom_settings(cls) -> dict:
        """`CustomData.get_settings` with `key_id="custom"`."""
        return cls.get_settings(key_id="custom")

    @classmethod
    def set_custom_settings(cls, **kwargs) -> None:
        """`CustomData.set_settings` with `key_id="custom"`."""
        cls.set_settings(key_id="custom", **kwargs)

    @classmethod
    def reset_custom_settings(cls) -> None:
        """`CustomData.reset_settings` with `key_id="custom"`."""
        cls.reset_settings(key_id="custom")

    @staticmethod
    def symbol_match(symbol: str, pattern: str, use_regex: bool = False):
        """Return whether symbol matches pattern.

        If `use_regex` is True, checks against a regular expression.
        Otherwise, checks against a glob-style pattern."""
        if use_regex:
            return re.match(pattern, symbol)
        return re.match(glob2re(pattern), symbol)


# ############# Synthetic ############# #


class SyntheticData(CustomData):
    """Data class for fetching synthetic data.

    Exposes an abstract class method `SyntheticData.generate_symbol`.
    Everything else is taken care of."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.synthetic")

    @classmethod
    def generate_symbol(cls, symbol: tp.Symbol, index: tp.Index, **kwargs) -> tp.SeriesFrame:
        """Abstract method to generate data of a symbol."""
        raise NotImplementedError

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        periods: tp.Optional[int] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        normalize: tp.Optional[bool] = None,
        inclusive: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to generate a symbol.

        Generates datetime index using `pd.date_range` and passes it to `SyntheticData.generate_symbol`
        to fill the Series/DataFrame with generated data.

        If `start` and `periods` are None, will set `start` to the beginning of the Unix epoch.

        If `end` is `periods` are None, will set `end` to the current time.

        For defaults, see `custom.synthetic` in `vectorbtpro._settings.data`."""
        synthetic_cfg = cls.get_settings(key_id="custom")

        if start is None:
            start = synthetic_cfg["start"]
        if end is None:
            end = synthetic_cfg["end"]
        if freq is None:
            freq = synthetic_cfg["freq"]
        if freq is not None:
            freq = prepare_freq(freq)
        if tz is None:
            tz = synthetic_cfg["tz"]
        if normalize is None:
            normalize = synthetic_cfg["normalize"]
        if inclusive is None:
            inclusive = synthetic_cfg["inclusive"]

        if start is not None:
            start = to_timestamp(start, tz=tz)
        if end is not None:
            end = to_timestamp(end, tz=tz)
        if start is None and periods is None:
            if tz is not None:
                start = to_timestamp(0, tz=tz)
            elif end is not None and end.tzinfo is not None:
                start = to_timestamp(0, tz=end.tzinfo)
            else:
                start = to_naive_timestamp(0)
        if end is None and periods is None:
            if tz is not None:
                end = to_timestamp("now", tz=tz)
            elif start is not None and start.tzinfo is not None:
                end = to_timestamp("now", tz=start.tzinfo)
            else:
                end = to_naive_timestamp("now")

        index = pd.date_range(
            start=start,
            end=end,
            periods=periods,
            freq=freq,
            normalize=normalize,
            inclusive=inclusive,
        )
        if tz is None:
            tz = index.tzinfo
        if len(index) == 0:
            raise ValueError("Date range is empty")
        return cls.generate_symbol(symbol, index, **kwargs), dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class RandomData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_random_data_nb`."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.random")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        columns: tp.Union[tp.Hashable, tp.IndexLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            columns (hashable or index_like): Column labels.

                Provide a single value (hashable) to make a Series.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            symmetric (bool): Whether to diminish negative returns and make them symmetric to positive ones.
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        For defaults, see `custom.random` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        random_cfg = cls.get_settings(key_id="custom")

        if checks.is_hashable(columns):
            columns = [columns]
            make_series = True
        else:
            make_series = False
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if start_value is None:
            start_value = random_cfg["start_value"]
        if mean is None:
            mean = random_cfg["mean"]
        if std is None:
            std = random_cfg["std"]
        if symmetric is None:
            symmetric = random_cfg["symmetric"]
        if seed is None:
            seed = random_cfg["seed"]
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_nb, jitted)
        out = func(
            (len(index), len(columns)),
            start_value=to_1d_array(start_value),
            mean=to_1d_array(mean),
            std=to_1d_array(std),
            symmetric=to_1d_array(symmetric),
        )
        if make_series:
            return pd.Series(out[:, 0], index=index, name=columns[0])
        return pd.DataFrame(out, index=index, columns=columns)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol].iloc[-2]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)


class RandomOHLCData(RandomData):
    """`RandomData` for data generated using `vectorbtpro.data.nb.generate_random_data_1d_nb`
    and then resampled using `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb`."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.random_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            n_ticks (int or array_like): Number of ticks per bar.

                Flexible argument. Can be a template with a context containing `symbol` and `index`.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            symmetric (bool): Whether to diminish negative returns and make them symmetric to positive ones.
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (dict): Template context.

        For defaults, see `custom.random_ohlc` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        random_ohlc_cfg = cls.get_settings(key_id="custom")

        if n_ticks is None:
            n_ticks = random_ohlc_cfg["n_ticks"]
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, sub_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        if start_value is None:
            start_value = random_ohlc_cfg["start_value"]
        if mean is None:
            mean = random_ohlc_cfg["mean"]
        if std is None:
            std = random_ohlc_cfg["std"]
        if symmetric is None:
            symmetric = random_ohlc_cfg["symmetric"]
        if seed is None:
            seed = random_ohlc_cfg["seed"]
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_1d_nb, jitted)
        ticks = func(np.sum(n_ticks), start_value=start_value, mean=mean, std=std, symmetric=symmetric)
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)


class GBMData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_gbm_data_nb`."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.gbm")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        columns: tp.Union[tp.Hashable, tp.IndexLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            columns (hashable or index_like): Column labels.

                Provide a single value (hashable) to make a Series.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            dt (float): Time change (one period of time).
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        For defaults, see `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        gbm_cfg = cls.get_settings(key_id="custom")

        if checks.is_hashable(columns):
            columns = [columns]
            make_series = True
        else:
            make_series = False
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if start_value is None:
            start_value = gbm_cfg["start_value"]
        if mean is None:
            mean = gbm_cfg["mean"]
        if std is None:
            std = gbm_cfg["std"]
        if dt is None:
            dt = gbm_cfg["dt"]
        if seed is None:
            seed = gbm_cfg["seed"]
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_gbm_data_nb, jitted)
        out = func(
            (len(index), len(columns)),
            start_value=to_1d_array(start_value),
            mean=to_1d_array(mean),
            std=to_1d_array(std),
            dt=to_1d_array(dt),
        )
        if make_series:
            return pd.Series(out[:, 0], index=index, name=columns[0])
        return pd.DataFrame(out, index=index, columns=columns)


class GBMOHLCData(GBMData):
    """`GBMData` for data generated using `vectorbtpro.data.nb.generate_gbm_data_1d_nb`
    and then resampled using `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb`."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.gbm_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate a symbol.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            n_ticks (int or array_like): Number of ticks per bar.

                Flexible argument. Can be a template with a context containing `symbol` and `index`.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            dt (float): Time change (one period of time).
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (dict): Template context.

        For defaults, see `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        gbm_cfg = cls.get_settings(key_id="custom")

        if n_ticks is None:
            n_ticks = gbm_cfg["n_ticks"]
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, sub_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        if start_value is None:
            start_value = gbm_cfg["start_value"]
        if mean is None:
            mean = gbm_cfg["mean"]
        if std is None:
            std = gbm_cfg["std"]
        if dt is None:
            dt = gbm_cfg["dt"]
        if seed is None:
            seed = gbm_cfg["seed"]
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_gbm_data_1d_nb, jitted)
        ticks = func(
            np.sum(n_ticks),
            start_value=start_value,
            mean=mean,
            std=std,
            dt=dt,
        )
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)


# ############# Local ############# #

LocalDataT = tp.TypeVar("LocalDataT", bound="LocalData")


class LocalData(CustomData):
    """Data class for fetching local data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.local")


# ############# File ############# #

FileDataT = tp.TypeVar("FileDataT", bound="FileData")


class FileData(LocalData):
    """Data class for fetching file data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.file")

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        recursive: bool = True,
        **kwargs,
    ) -> tp.List[Path]:
        """Get the list of all paths matching a path."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists():
            if path.is_dir():
                sub_paths = [p for p in path.iterdir() if p.is_file()]
            else:
                sub_paths = [path]
        else:
            sub_paths = list([Path(p) for p in glob(str(path), recursive=recursive)])
        if match_regex is not None:
            sub_paths = [p for p in sub_paths if re.match(match_regex, str(p))]
        if sort_paths:
            sub_paths = sorted(sub_paths)
        return sub_paths

    @classmethod
    def path_to_symbol(cls, path: tp.PathLike, **kwargs) -> str:
        """Convert a path into a symbol."""
        return Path(path).stem

    @classmethod
    def fetch(
        cls: tp.Type[FileDataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        paths: tp.Any = None,
        match_paths: tp.Optional[bool] = None,
        match_regex: tp.Optional[str] = None,
        sort_paths: tp.Optional[bool] = None,
        match_path_kwargs: tp.KwargsLike = None,
        path_to_symbol_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> FileDataT:
        """Override `vectorbtpro.data.base.Data.fetch` to take care of paths.

        Use either `symbols` or `paths` to specify the path to one or multiple files.
        Allowed are paths in a string or `pathlib.Path` format, or string expressions accepted by `glob.glob`.

        Set `match_paths` to False to not parse paths and behave like a regular
        `vectorbtpro.data.base.Data` instance.

        For defaults, see `custom.local` in `vectorbtpro._settings.data`.
        """
        local_cfg = cls.get_settings(key_id="custom")

        if match_paths is None:
            match_paths = local_cfg["match_paths"]
        if match_regex is None:
            match_regex = local_cfg["match_regex"]
        if sort_paths is None:
            sort_paths = local_cfg["sort_paths"]

        if match_paths:
            sync = False
            if paths is None:
                paths = symbols
                sync = True
            elif symbols is None:
                sync = True
            if paths is None:
                raise ValueError("At least symbols or paths must be set")
            if match_path_kwargs is None:
                match_path_kwargs = {}
            if path_to_symbol_kwargs is None:
                path_to_symbol_kwargs = {}

            single_symbol = False
            if isinstance(symbols, (str, Path)):
                # Single symbol
                symbols = [symbols]
                single_symbol = True

            single_path = False
            if isinstance(paths, (str, Path)):
                # Single path
                paths = [paths]
                single_path = True
                if sync:
                    single_symbol = True

            if isinstance(paths, symbol_dict):
                # Dict of path per symbol
                if sync:
                    symbols = list(paths.keys())
                elif len(symbols) != len(paths):
                    raise ValueError("The number of symbols must be equal to the number of matched paths")
            elif checks.is_iterable(paths) or checks.is_sequence(paths):
                # Multiple paths
                matched_paths = [
                    p
                    for sub_path in paths
                    for p in cls.match_path(
                        sub_path,
                        match_regex=match_regex,
                        sort_paths=sort_paths,
                        **match_path_kwargs,
                    )
                ]
                if len(matched_paths) == 0:
                    raise FileNotFoundError(f"No paths could be matched with {paths}")
                if sync:
                    symbols = []
                    paths = symbol_dict()
                    for p in matched_paths:
                        s = cls.path_to_symbol(p, **path_to_symbol_kwargs)
                        symbols.append(s)
                        paths[s] = p
                elif len(symbols) != len(matched_paths):
                    raise ValueError("The number of symbols must be equal to the number of matched paths")
                else:
                    paths = symbol_dict({s: matched_paths[i] for i, s in enumerate(symbols)})
                if len(matched_paths) == 1 and single_path:
                    paths = matched_paths[0]
            else:
                raise TypeError(f"Path '{paths}' is not supported")
            if len(symbols) == 1 and single_symbol:
                symbols = symbols[0]

        return super(FileData, cls).fetch(
            symbols,
            path=paths,
            **kwargs,
        )


CSVDataT = tp.TypeVar("CSVDataT", bound="CSVData")


class CSVData(FileData):
    """Data class for fetching CSV data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.csv")

    @classmethod
    def list_symbols(
        cls,
        path: tp.PathLike = ".",
        **match_path_kwargs,
    ) -> tp.List[str]:
        """List all symbols under a path."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "**" / "*.csv"
        return list(map(str, cls.match_path(path, **match_path_kwargs)))

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        path: tp.Any = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        header: tp.Optional[tp.MaybeSequence[int]] = None,
        index_col: tp.Optional[int] = None,
        parse_dates: tp.Optional[bool] = None,
        squeeze: tp.Optional[bool] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_csv_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load a CSV file.

        Args:
            symbol (str): Symbol.
            path (str): Path.

                If `path` is None, uses `symbol` as the path to the CSV file.
            start (any): Start datetime.

                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (any): End datetime.

                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (int): Start row (inclusive).

                Must exclude header rows.
            end_row (int): End row (exclusive).

                Must exclude header rows.
            header (int or sequence of int): See `pd.read_csv`.
            index_col (int): See `pd.read_csv`.
            parse_dates (bool): See `pd.read_csv`.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            chunk_func (callable): Function to select and concatenate chunks from `TextFileReader`.

                Gets called only if `iterator` or `chunksize` are set.
            **read_csv_kwargs: Keyword arguments passed to `pd.read_csv`.

        `skiprows` and `nrows` will be automatically calculated based on `start_row` and `end_row`.

        When either `start` or `end` is provided, will fetch the entire data first and filter it thereafter.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for other arguments.

        For defaults, see `custom.csv` in `vectorbtpro._settings.data`."""
        from pandas.io.parsers import TextFileReader
        from pandas.api.types import is_object_dtype

        csv_cfg = cls.get_settings(key_id="custom")

        if start is None:
            start = csv_cfg["start"]
        if end is None:
            end = csv_cfg["end"]
        if tz is None:
            tz = csv_cfg["tz"]
        if start_row is None:
            start_row = csv_cfg["start_row"]
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = csv_cfg["end_row"]
        if header is None:
            header = csv_cfg["header"]
        if index_col is None:
            index_col = csv_cfg["index_col"]
        if checks.is_int(index_col) and index_col is False:
            index_col = None
        if parse_dates is None:
            parse_dates = csv_cfg["parse_dates"]
        if squeeze is None:
            squeeze = csv_cfg["squeeze"]
        read_csv_kwargs = merge_dicts(csv_cfg["read_csv_kwargs"], read_csv_kwargs)

        if path is None:
            path = symbol
        if isinstance(header, int):
            header = [header]
        header_rows = header[-1] + 1
        start_row += header_rows
        if end_row is not None:
            end_row += header_rows
        skiprows = range(header_rows, start_row)
        if end_row is not None:
            nrows = end_row - start_row
        else:
            nrows = None

        sep = read_csv_kwargs.pop("sep", None)
        if isinstance(path, (str, Path)):
            try:
                _path = Path(path)
                if _path.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
            except Exception as e:
                pass
        if sep is None:
            sep = ","

        obj = pd.read_csv(
            path,
            sep=sep,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            skiprows=skiprows,
            nrows=nrows,
            **read_csv_kwargs,
        )

        if isinstance(obj, TextFileReader):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        if index_col is not None and parse_dates and is_object_dtype(obj.index.dtype):
            obj.index = pd.to_datetime(obj.index, utc=True)
            if tz is not None:
                obj.index = obj.index.tz_convert(tz)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tzinfo
        if start is not None or end is not None:
            if not isinstance(obj.index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if obj.index.tzinfo is not None:
                if start is not None:
                    start = to_tzaware_timestamp(start, naive_tz=tz, tz=obj.index.tzinfo)
                if end is not None:
                    end = to_tzaware_timestamp(end, naive_tz=tz, tz=obj.index.tzinfo)
            else:
                if start is not None:
                    start = to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= obj.index >= start
            if end is not None:
                mask &= obj.index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            obj = obj.iloc[mask_indices[0] : mask_indices[-1] + 1]
            start_row += mask_indices[0]
        return obj, dict(last_row=start_row - header_rows + len(obj.index) - 1, tz_convert=tz)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start_row"] = self.returned_kwargs[symbol]["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class HDFPathNotFoundError(Exception):
    """Gets raised if the path to an HDF file could not be found."""

    pass


class HDFKeyNotFoundError(Exception):
    """Gets raised if the key to an HDF object could not be found."""

    pass


HDFDataT = tp.TypeVar("HDFDataT", bound="HDFData")


class HDFData(FileData):
    """Data class for fetching HDF data."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.hdf")

    @classmethod
    def list_symbols(
        cls,
        path: tp.PathLike = ".",
        **match_path_kwargs,
    ) -> tp.List[str]:
        """List all symbols under a path."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "**" / "*.h5"
        return list(map(str, cls.match_path(path, **match_path_kwargs)))

    @classmethod
    def split_hdf_path(
        cls,
        path: tp.PathLike,
        key: tp.Optional[str] = None,
        _full_path: tp.Optional[Path] = None,
    ) -> tp.Tuple[Path, tp.Optional[str]]:
        """Split the path to an HDF object into the path to the file and the key."""
        path = Path(path)
        if _full_path is None:
            _full_path = path
        if path.exists():
            if path.is_dir():
                raise HDFPathNotFoundError(f"No HDF files could be matched with {_full_path}")
            return path, key
        new_path = path.parent
        if key is None:
            new_key = path.name
        else:
            new_key = str(Path(path.name) / key)
        return cls.split_hdf_path(new_path, new_key, _full_path=_full_path)

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        recursive: bool = True,
        **kwargs,
    ) -> tp.List[Path]:
        """Override `FileData.match_path` to return a list of HDF paths
        (path to file + key) matching a path."""
        path = Path(path)
        if path.exists():
            if path.is_dir():
                sub_paths = [p for p in path.iterdir() if p.is_file()]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
            else:
                with pd.HDFStore(str(path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                key_paths = [path / k for k in keys]
        else:
            try:
                file_path, key = cls.split_hdf_path(path)
                with pd.HDFStore(str(file_path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                if key is None:
                    key_paths = [file_path / k for k in keys]
                elif key in keys:
                    key_paths = [file_path / key]
                else:
                    matching_keys = []
                    for k in keys:
                        if k.startswith(key) or PurePath("/" + str(k)).match("/" + str(key)):
                            matching_keys.append(k)
                    if len(matching_keys) == 0:
                        raise HDFKeyNotFoundError(f"No HDF keys could be matched with {key}")
                    key_paths = [file_path / k for k in matching_keys]
            except HDFPathNotFoundError:
                sub_paths = list([Path(p) for p in glob(str(path), recursive=recursive)])
                if len(sub_paths) == 0 and re.match(r".+\..+", str(path)):
                    base_path = None
                    base_ended = False
                    key_path = None
                    for part in path.parts:
                        part = Path(part)
                        if base_ended:
                            if key_path is None:
                                key_path = part
                            else:
                                key_path /= part
                        else:
                            if re.match(r".+\..+", str(part)):
                                base_ended = True
                            if base_path is None:
                                base_path = part
                            else:
                                base_path /= part
                    sub_paths = list([Path(p) for p in glob(str(base_path), recursive=recursive)])
                    if key_path is not None:
                        sub_paths = [p / key_path for p in sub_paths]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
        if match_regex is not None:
            key_paths = [p for p in key_paths if re.match(match_regex, str(p))]
        if sort_paths:
            key_paths = sorted(key_paths)
        return key_paths

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        path: tp.Any = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_hdf_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to load an HDF object.

        Args:
            symbol (str): Symbol.
            path (str): Path.

                Will be resolved with `HDFData.split_hdf_path`.

                If `path` is None, uses `symbol` as the path to the HDF file.
            start (any): Start datetime.

                Will extract the object's index and compare the index to the date.
                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Can only be used if the object was saved in the table format!
            end (any): End datetime.

                Will extract the object's index and compare the index to the date.
                Will use the timezone of the object. See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Can only be used if the object was saved in the table format!
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (int): Start row (inclusive).

                Will use it when querying index as well.
            end_row (int): End row (exclusive).

                Will use it when querying index as well.
            chunk_func (callable): Function to select and concatenate chunks from `TableIterator`.

                Gets called only if `iterator` or `chunksize` are set.
            **read_hdf_kwargs: Keyword arguments passed to `pd.read_hdf`.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments.

        For defaults, see `custom.hdf` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        from pandas.io.pytables import TableIterator

        hdf_cfg = cls.get_settings(key_id="custom")

        if start is None:
            start = hdf_cfg["start"]
        if end is None:
            end = hdf_cfg["end"]
        if tz is None:
            tz = hdf_cfg["tz"]
        if start_row is None:
            start_row = hdf_cfg["start_row"]
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = hdf_cfg["end_row"]
        read_hdf_kwargs = merge_dicts(hdf_cfg["read_hdf_kwargs"], read_hdf_kwargs)

        if path is None:
            path = symbol
        path = Path(path)
        file_path, key = cls.split_hdf_path(path)

        if start is not None or end is not None:
            hdf_store_arg_names = get_func_arg_names(pd.HDFStore.__init__)
            hdf_store_kwargs = dict()
            for k, v in read_hdf_kwargs.items():
                if k in hdf_store_arg_names:
                    hdf_store_kwargs[k] = v
            with pd.HDFStore(str(file_path), mode="r", **hdf_store_kwargs) as store:
                index = store.select_column(key, "index", start=start_row, stop=end_row)
            if not isinstance(index, pd.Index):
                index = pd.Index(index)
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if tz is None:
                tz = index.tzinfo
            if index.tzinfo is not None:
                if start is not None:
                    start = to_tzaware_timestamp(start, naive_tz=tz, tz=index.tzinfo)
                if end is not None:
                    end = to_tzaware_timestamp(end, naive_tz=tz, tz=index.tzinfo)
            else:
                if start is not None:
                    start = to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= index >= start
            if end is not None:
                mask &= index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            start_row += mask_indices[0]
            end_row = start_row + mask_indices[-1] - mask_indices[0] + 1

        obj = pd.read_hdf(file_path, key=key, start=start_row, stop=end_row, **read_hdf_kwargs)
        if isinstance(obj, TableIterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tzinfo
        return obj, dict(last_row=start_row + len(obj.index) - 1, tz_convert=tz)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.Tuple[tp.SeriesFrame, dict]:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start_row"] = self.returned_kwargs[symbol]["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


# ############# Remote ############# #


RemoteDataT = tp.TypeVar("RemoteDataT", bound="RemoteData")


class RemoteData(CustomData):
    """Data class for fetching remote data.

    Remote data usually has arguments such as `start`, `end`, and `timeframe`.

    Overrides `vectorbtpro.data.base.Data.update_symbol` to update data based on the `start` argument."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.remote")

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


class YFData(RemoteData):
    """Data class for fetching from Yahoo Finance.

    See https://github.com/ranaroussi/yfinance for API.

    See `YFData.fetch_symbol` for arguments.

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt

        >>> data = vbt.YFData.fetch(
        ...     "BTC-USD",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.yf")

    _column_config: tp.ClassVar[Config] = HybridConfig(
        {
            "Dividends": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Stock Splits": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.nonzero_prod_reduce_nb,
                )
            ),
        }
    )

    @property
    def column_config(self) -> Config:
        return self._column_config

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        period: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        **history_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Yahoo Finance.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            **history_kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.

        For defaults, see `custom.yf` in `vectorbtpro._settings.data`.

        !!! warning
            Data coming from Yahoo is not the most stable data out there. Yahoo may manipulate data
            how they want, add noise, return missing data points (see volume in the example below), etc.
            It's only used in vectorbt for demonstration purposes.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("yfinance")
        import yfinance as yf

        yf_cfg = cls.get_settings(key_id="custom")

        if period is None:
            period = yf_cfg["period"]
        if start is None:
            start = yf_cfg["start"]
        if end is None:
            end = yf_cfg["end"]
        if timeframe is None:
            timeframe = yf_cfg["timeframe"]
        if tz is None:
            tz = yf_cfg["tz"]
        history_kwargs = merge_dicts(yf_cfg["history_kwargs"], history_kwargs)

        ticker = yf.Ticker(symbol)
        def_history_kwargs = get_func_kwargs(ticker.history)
        ticker_tz = ticker._get_ticker_tz(
            history_kwargs.get("debug", def_history_kwargs["debug"]),
            history_kwargs.get("proxy", def_history_kwargs["proxy"]),
            history_kwargs.get("timeout", def_history_kwargs["timeout"]),
        )
        if tz is None:
            tz = ticker_tz
        if start is not None:
            start = to_tzaware_datetime(start, naive_tz=tz, tz=ticker_tz)
        if end is not None:
            end = to_tzaware_datetime(end, naive_tz=tz, tz=ticker_tz)
        freq = prepare_freq(timeframe)
        split = split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "t":
                unit = "m"
            elif unit == "W":
                unit = "wk"
            elif unit == "M":
                unit = "mo"
            timeframe = str(multiplier) + unit

        df = ticker.history(period=period, start=start, end=end, interval=timeframe, **history_kwargs)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize(ticker_tz)

        if not df.empty:
            if start is not None:
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz_convert=tz, freq=freq)


YFData.override_column_config_doc(__pdoc__)

BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


class BinanceData(RemoteData):
    """Data class for fetching from Binance.

    See https://github.com/sammchardy/python-binance for API.

    See `BinanceData.fetch_symbol` for arguments.

    !!! note
        If you are using an exchange from the US, Japan or other TLD then make sure pass `tld="us"`
        in `client_config` when creating the client.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.BinanceData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY",
        ...         api_secret="YOUR_SECRET"
        ...     )
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.BinanceData.fetch(
        ...     "BTCUSDT",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.binance")

    _column_config: tp.ClassVar[Config] = HybridConfig(
        {
            "Quote volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Taker base volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Taker quote volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
        }
    )

    @property
    def column_config(self) -> Config:
        return self._column_config

    @classmethod
    def resolve_client(cls, client: tp.Optional[BinanceClientT] = None, **client_config) -> BinanceClientT:
        """Resolve the client.

        If provided, must be of the type `binance.client.Client`.
        Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("binance")
        from binance.client import Client

        binance_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = binance_cfg["client"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(binance_cfg["client_config"], client_config)
        if client is None:
            client = Client(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        client: tp.Optional[BinanceClientT] = None,
        client_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `CustomData.symbol_match` to check each symbol against `pattern`."""
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        all_symbols = []
        for dct in client.get_exchange_info()["symbols"]:
            symbol = dct["symbol"]
            if pattern is not None:
                if not cls.symbol_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)
        return sorted(all_symbols)

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[BinanceClientT] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        klines_type: tp.Union[None, int, str] = None,
        limit: tp.Optional[int] = None,
        delay: tp.Optional[float] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        **get_klines_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Binance.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Client.

                See `BinanceData.resolve_client`.
            client_config (dict): Client config.

                See `BinanceData.resolve_client`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            klines_type (int or str): Kline type.

                See `binance.enums.HistoricalKlinesType`. Supports strings.
            limit (int): The maximum number of returned items.
            delay (float): Time to sleep after each request (in milliseconds).
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
            silence_warnings (bool): Whether to silence all warnings.
            **get_klines_kwargs: Keyword arguments passed to `binance.client.Client.get_klines`.

        For defaults, see `custom.binance` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("binance")
        from binance.enums import HistoricalKlinesType

        binance_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if start is None:
            start = binance_cfg["start"]
        if end is None:
            end = binance_cfg["end"]
        if timeframe is None:
            timeframe = binance_cfg["timeframe"]
        if tz is None:
            tz = binance_cfg["tz"]
        if klines_type is None:
            klines_type = binance_cfg["klines_type"]
        if isinstance(klines_type, str):
            klines_type = map_enum_fields(klines_type, HistoricalKlinesType)
        if isinstance(klines_type, int):
            klines_type = {i.value: i for i in HistoricalKlinesType}[klines_type]
        if limit is None:
            limit = binance_cfg["limit"]
        if delay is None:
            delay = binance_cfg["delay"]
        if show_progress is None:
            show_progress = binance_cfg["show_progress"]
        pbar_kwargs = merge_dicts(binance_cfg["pbar_kwargs"], pbar_kwargs)
        if silence_warnings is None:
            silence_warnings = binance_cfg["silence_warnings"]
        get_klines_kwargs = merge_dicts(binance_cfg["get_klines_kwargs"], get_klines_kwargs)

        # Prepare parameters
        freq = prepare_freq(timeframe)
        split = split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "t":
                unit = "m"
            elif unit == "W":
                unit = "w"
            timeframe = str(multiplier) + unit
        if start is not None:
            start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
            first_valid_ts = client._get_earliest_valid_timestamp(symbol, timeframe, klines_type)
            start_ts = max(start_ts, first_valid_ts)
        else:
            start_ts = None
        prev_end_ts = None
        if end is not None:
            end_ts = datetime_to_ms(to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
        else:
            end_ts = None

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "/"
            return str(pd.Timestamp(ts, unit="ms", tz="utc"))

        def _filter_func(d: tp.Sequence, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d[0] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d[0] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d[0] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = client._klines(
                        symbol=symbol,
                        interval=timeframe,
                        limit=limit,
                        startTime=start_ts if prev_end_ts is None else prev_end_ts,
                        endTime=end_ts,
                        klines_type=klines_type,
                        **get_klines_kwargs,
                    )
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0][0]
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1][0]),
                        )
                    )
                    pbar.update(1)
                    prev_end_ts = next_data[-1][0]
                    if end_ts is not None and prev_end_ts >= end_ts:
                        break
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                    "Use update() method to fetch missing data.",
                    stacklevel=2,
                )

        # Convert data to a DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote volume",
                "Trade count",
                "Taker base volume",
                "Taker quote volume",
                "Ignore",
            ],
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df["Quote volume"] = df["Quote volume"].astype(float)
        df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        df["Taker base volume"] = df["Taker base volume"].astype(float)
        df["Taker quote volume"] = df["Taker quote volume"].astype(float)
        del df["Open time"]
        del df["Close time"]
        del df["Ignore"]

        return df, dict(tz_convert=tz, freq=freq)


BinanceData.override_column_config_doc(__pdoc__)


class CCXTData(RemoteData):
    """Data class for fetching using CCXT.

    See https://github.com/ccxt/ccxt for API.

    See `CCXTData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.CCXTData.set_custom_settings(
        ...     exchanges=dict(
        ...         binance=dict(
        ...             exchange_config=dict(
        ...                 apiKey="YOUR_KEY",
        ...                 secret="YOUR_SECRET"
        ...             )
        ...         )
        ...     )
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.CCXTData.fetch(
        ...     "BTCUSDT",
        ...     exchange="binance",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.ccxt")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        exchange: tp.Optional[tp.Union[str, CCXTExchangeT]] = None,
        exchange_config: tp.Optional[tp.KwargsLike] = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `CustomData.symbol_match` to check each symbol against `pattern`."""
        if exchange_config is None:
            exchange_config = {}
        exchange = cls.resolve_exchange(exchange=exchange, **exchange_config)
        all_symbols = []
        for symbol in exchange.load_markets():
            if pattern is not None:
                if not cls.symbol_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)
        return sorted(all_symbols)

    @classmethod
    def resolve_exchange(
        cls,
        exchange: tp.Optional[tp.Union[str, CCXTExchangeT]] = None,
        **exchange_config,
    ) -> CCXTExchangeT:
        """Resolve the exchange.

        If provided, must be of the type `ccxt.base.exchange.Exchange`.
        Otherwise, will be created using `exchange_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ccxt")
        import ccxt

        ccxt_cfg = cls.get_settings(key_id="custom")

        if exchange is None:
            exchange = ccxt_cfg["exchange"]
        if isinstance(exchange, str):
            exchange = exchange.lower()
            exchange_name = exchange
        elif isinstance(exchange, ccxt.Exchange):
            exchange_name = type(exchange).__name__
        else:
            raise ValueError(f"Unknown exchange of type {type(exchange)}")
        if exchange_config is None:
            exchange_config = {}
        has_exchange_config = len(exchange_config) > 0
        exchange_config = merge_dicts(
            ccxt_cfg["exchange_config"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("exchange_config", {}),
            exchange_config,
        )
        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange '{exchange}' not found in CCXT")
            exchange = getattr(ccxt, exchange)(exchange_config)
        else:
            if has_exchange_config:
                raise ValueError("Cannot apply config after instantiation of the exchange")
        return exchange

    @staticmethod
    def _find_earliest_date(
        fetch_func: tp.Callable,
        start: tp.DatetimeLike = 0,
        end: tp.DatetimeLike = "now",
        tz: tp.Optional[tp.TimezoneLike] = None,
        for_internal_use: bool = False,
    ) -> tp.Optional[pd.Timestamp]:
        """Find the earliest date using binary search."""
        if start is not None:
            start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
            fetched_data = fetch_func(start_ts, 1)
            if for_internal_use and len(fetched_data) > 0:
                return pd.Timestamp(start_ts, unit="ms", tz="utc")
        else:
            fetched_data = []
        if len(fetched_data) == 0 and start != 0:
            fetched_data = fetch_func(0, 1)
            if for_internal_use and len(fetched_data) > 0:
                return pd.Timestamp(0, unit="ms", tz="utc")
        if len(fetched_data) == 0:
            if start is not None:
                start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
            else:
                start_ts = datetime_to_ms(to_tzaware_datetime(0, naive_tz=tz, tz="UTC"))
            start_ts = start_ts - start_ts % 86400000
            if end is not None:
                end_ts = datetime_to_ms(to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
            else:
                end_ts = datetime_to_ms(to_tzaware_datetime("now", naive_tz=tz, tz="UTC"))
            end_ts = end_ts - end_ts % 86400000 + 86400000
            start_time = start_ts
            end_time = end_ts
            while True:
                mid_time = (start_time + end_time) // 2
                mid_time = mid_time - mid_time % 86400000
                if mid_time == start_time:
                    break
                _fetched_data = fetch_func(mid_time, 1)
                if len(_fetched_data) == 0:
                    start_time = mid_time
                else:
                    end_time = mid_time
                    fetched_data = _fetched_data
        if len(fetched_data) > 0:
            return pd.Timestamp(fetched_data[0][0], unit="ms", tz="utc")
        return None

    @classmethod
    def find_earliest_date(cls, symbol: str, for_internal_use: bool = False, **kwargs) -> tp.Optional[pd.Timestamp]:
        """Find the earliest date using binary search.

        See `CCXTData.fetch_symbol` for arguments."""
        return cls._find_earliest_date(
            **cls.fetch_symbol(symbol, return_fetch_method=True, **kwargs),
            for_internal_use=for_internal_use,
        )

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        exchange: tp.Optional[tp.Union[str, CCXTExchangeT]] = None,
        exchange_config: tp.Optional[tp.KwargsLike] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        find_earliest_date: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        delay: tp.Optional[float] = None,
        retries: tp.Optional[int] = None,
        fetch_params: tp.Optional[tp.KwargsLike] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        return_fetch_method: bool = False,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from CCXT.

        Args:
            symbol (str): Symbol.

                Symbol can be in the `EXCHANGE:SYMBOL` format, in this case `exchange` argument will be ignored.
            exchange (str or object): Exchange identifier or an exchange object.

                See `CCXTData.resolve_exchange`.
            exchange_config (dict): Exchange config.

                See `CCXTData.resolve_exchange`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            find_earliest_date (bool): Whether to find the earliest date using `CCXTData.find_earliest_date`.
            limit (int): The maximum number of returned items.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            retries (int): The number of retries on failure to fetch data.
            fetch_params (dict): Exchange-specific keyword arguments passed to `fetch_ohlcv`.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
            silence_warnings (bool): Whether to silence all warnings.
            return_fetch_method (bool): Required by `CCXTData.find_earliest_date`.

        For defaults, see `custom.ccxt` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ccxt")
        import ccxt

        ccxt_cfg = cls.get_settings(key_id="custom")

        if ":" in symbol:
            exchange, symbol = symbol.split(":")
        if exchange_config is None:
            exchange_config = {}
        exchange = cls.resolve_exchange(exchange=exchange, **exchange_config)
        exchange_name = type(exchange).__name__

        if start is None:
            start = ccxt_cfg["exchanges"].get(exchange_name, {}).get("start", ccxt_cfg["start"])
        if end is None:
            end = ccxt_cfg["exchanges"].get(exchange_name, {}).get("end", ccxt_cfg["end"])
        if timeframe is None:
            timeframe = ccxt_cfg["exchanges"].get(exchange_name, {}).get("timeframe", ccxt_cfg["timeframe"])
        if tz is None:
            tz = ccxt_cfg["exchanges"].get(exchange_name, {}).get("tz", ccxt_cfg["tz"])
        if find_earliest_date is None:
            find_earliest_date = (
                ccxt_cfg["exchanges"].get(exchange_name, {}).get("find_earliest_date", ccxt_cfg["find_earliest_date"])
            )
        if limit is None:
            limit = ccxt_cfg["exchanges"].get(exchange_name, {}).get("limit", ccxt_cfg["limit"])
        if delay is None:
            delay = ccxt_cfg["exchanges"].get(exchange_name, {}).get("delay", ccxt_cfg["delay"])
        if retries is None:
            retries = ccxt_cfg["exchanges"].get(exchange_name, {}).get("retries", ccxt_cfg["retries"])
        fetch_params = merge_dicts(
            ccxt_cfg["fetch_params"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("fetch_params", {}),
            fetch_params,
        )
        if show_progress is None:
            show_progress = ccxt_cfg["exchanges"].get(exchange_name, {}).get("show_progress", ccxt_cfg["show_progress"])
        pbar_kwargs = merge_dicts(
            ccxt_cfg["pbar_kwargs"],
            ccxt_cfg["exchanges"].get(exchange_name, {}).get("pbar_kwargs", {}),
            pbar_kwargs,
        )
        if silence_warnings is None:
            silence_warnings = (
                ccxt_cfg["exchanges"].get(exchange_name, {}).get("silence_warnings", ccxt_cfg["silence_warnings"])
            )
        if not exchange.has["fetchOHLCV"]:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
        if exchange.has["fetchOHLCV"] == "emulated":
            if not silence_warnings:
                warnings.warn("Using emulated OHLCV candles", stacklevel=2)

        freq = prepare_freq(timeframe)
        split = split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "t":
                unit = "m"
            elif unit == "W":
                unit = "w"
            elif unit == "Y":
                unit = "y"
            timeframe = str(multiplier) + unit
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except ccxt.NetworkError as e:
                        if i == retries - 1:
                            raise e
                        if not silence_warnings:
                            warnings.warn(traceback.format_exc(), stacklevel=2)
                        if delay is not None:
                            time.sleep(delay / 1000)

            return retry_method

        @_retry
        def _fetch(_since, _limit):
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=_since,
                limit=_limit,
                params=fetch_params,
            )

        if return_fetch_method:
            return dict(fetch_func=_fetch, start=start, end=end, tz=tz)

        # Establish the timestamps
        if find_earliest_date and start is not None:
            start = cls._find_earliest_date(_fetch, start=start, end=end, tz=tz, for_internal_use=True)
        if start is not None:
            start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
        else:
            start_ts = None
        if end is not None:
            end_ts = datetime_to_ms(to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
        else:
            end_ts = None
        prev_end_ts = None

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "/"
            return str(pd.Timestamp(ts, unit="ms", tz="utc"))

        def _filter_func(d: tp.Sequence, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d[0] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d[0] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d[0] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = _fetch(start_ts if prev_end_ts is None else prev_end_ts, limit)
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0][0]
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1][0]),
                        )
                    )
                    pbar.update(1)
                    prev_end_ts = next_data[-1][0]
                    if end_ts is not None and prev_end_ts >= end_ts:
                        break
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                    "Use update() method to fetch missing data.",
                    stacklevel=2,
                )

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume"])
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

        return df, dict(tz_convert=tz, freq=freq)


AlpacaDataT = tp.TypeVar("AlpacaDataT", bound="AlpacaData")


class AlpacaData(RemoteData):
    """Data class for fetching from Alpaca.

    See https://github.com/alpacahq/alpaca-py for API.

    See `AlpacaData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional for crypto):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.AlpacaData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY",
        ...         secret_key="YOUR_SECRET"
        ...     )
        ... )
        ```

        * Fetch stock data:

        ```pycon
        >>> data = vbt.AlpacaData.fetch(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        * Fetch crypto data:

        ```pycon
        >>> data = vbt.AlpacaData.fetch(
        ...     "BTCUSD",
        ...     client_type="crypto",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.alpaca")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        status: tp.Optional[str] = None,
        asset_class: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        trading_client: tp.Optional[AlpacaClientT] = None,
        client_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `CustomData.symbol_match` to check each symbol against `pattern`.

        Arguments `status`, `asset_class`, and `exchange` can be strings, such as `asset_class="crypto"`.
        For possible values, take a look into `alpaca.trading.enums`.

        !!! note
            If you get an authorization error, make sure that you either enable or disable
            the `paper` flag in `client_config` depending upon the account whose credentials you used.
            By default, the credentials are assumed to be of a live trading account (`paper=False`)."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange

        alpaca_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(alpaca_cfg["client_config"], client_config)
        if trading_client is None:
            arg_names = get_func_arg_names(TradingClient.__init__)
            client_config = {k: v for k, v in client_config.items() if k in arg_names}
            trading_client = TradingClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")

        if status is not None:
            if isinstance(status, str):
                status = getattr(AssetStatus, status.upper())
        if asset_class is not None:
            if isinstance(asset_class, str):
                asset_class = getattr(AssetClass, asset_class.upper())
        if exchange is not None:
            if isinstance(exchange, str):
                exchange = getattr(AssetExchange, exchange.upper())
        search_params = GetAssetsRequest(status=status, asset_class=asset_class, exchange=exchange)
        assets = trading_client.get_all_assets(search_params)
        all_symbols = []
        for asset in assets:
            symbol = asset.symbol
            if pattern is not None:
                if not cls.symbol_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)
        return sorted(all_symbols)

    @classmethod
    def resolve_client(
        cls,
        client: tp.Optional[AlpacaClientT] = None,
        client_type: tp.Optional[str] = None,
        **client_config,
    ) -> AlpacaClientT:
        """Resolve the client.

        If provided, must be of the type `alpaca.data.historical.CryptoHistoricalDataClient`
        for `client_type="crypto"` and `alpaca.data.historical.StockHistoricalDataClient` for
        `client_type="stocks"`. Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        alpaca_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = alpaca_cfg["client"]
        if client_type is None:
            client_type = alpaca_cfg["client_type"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(alpaca_cfg["client_config"], client_config)
        if client is None:
            if client_type == "crypto":
                arg_names = get_func_arg_names(CryptoHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = CryptoHistoricalDataClient(**client_config)
            elif client_type == "stocks":
                arg_names = get_func_arg_names(StockHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = StockHistoricalDataClient(**client_config)
            else:
                raise ValueError(f"Invalid client type '{client_type}'")
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[AlpacaClientT] = None,
        client_type: tp.Optional[str] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        adjustment: tp.Optional[str] = None,
        feed: tp.Optional[str] = None,
        limit: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpaca.

        Args:
            symbol (str): Symbol.
            client (alpaca.common.rest.RESTClient): Client.

                See `AlpacaData.resolve_client`.
            client_type (str): Client type.

                See `AlpacaData.resolve_client`.
            client_config (dict): Client config.

                See `AlpacaData.resolve_client`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjustment (str): Specifies the corporate action adjustment for the returned bars.

                Options are: "raw", "split", "dividend" or "all". Default is "raw".
            feed (str): The feed to pull market data from.

                This is either "iex", "otc", or "sip". Feeds "sip" and "otc" are only available to
                those with a subscription. Default is "iex" for free plans and "sip" for paid.
            limit (int): The maximum number of returned items.

        For defaults, see `custom.alpaca` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        alpaca_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, client_type=client_type, **client_config)
        if start is None:
            start = alpaca_cfg["start"]
        if end is None:
            end = alpaca_cfg["end"]
        if timeframe is None:
            timeframe = alpaca_cfg["timeframe"]
        if tz is None:
            tz = alpaca_cfg["tz"]
        if adjustment is None:
            adjustment = alpaca_cfg["adjustment"]
        if feed is None:
            feed = alpaca_cfg["feed"]
        if limit is None:
            limit = alpaca_cfg["limit"]

        freq = prepare_freq(timeframe)
        split = split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "t":
                unit = TimeFrameUnit.Minute
            elif unit == "h":
                unit = TimeFrameUnit.Hour
            elif unit == "d":
                unit = TimeFrameUnit.Day
            elif unit == "W":
                unit = TimeFrameUnit.Week
            elif unit == "M":
                unit = TimeFrameUnit.Month
            else:
                raise ValueError(f"Invalid timeframe '{timeframe}'")
        else:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        timeframe = TimeFrame(multiplier, unit)

        if start is not None:
            start = to_tzaware_datetime(start, naive_tz=tz, tz="UTC")
            start_str = start.replace(tzinfo=None).isoformat("T")
        else:
            start_str = None
        if end is not None:
            end = to_tzaware_datetime(end, naive_tz=tz, tz="UTC")
            end_str = end.replace(tzinfo=None).isoformat("T")
        else:
            end_str = None

        if isinstance(client, CryptoHistoricalDataClient):
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
            )
            df = client.get_crypto_bars(request).df
        elif isinstance(client, StockHistoricalDataClient):
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
            )
            df = client.get_stock_bars(request).df
        else:
            raise TypeError(f"Invalid client of type {type(client)}")

        df = df.droplevel("symbol", axis=0)
        df.index = df.index.rename("Open time")
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "Trade count",
                "vwap": "VWAP",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        if not df.empty:
            if start is not None:
                start = to_timestamp(start, tz=df.index.tzinfo)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = to_timestamp(end, tz=df.index.tzinfo)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz_convert=tz, freq=freq)


AlpacaData.override_column_config_doc(__pdoc__)

PolygonDataT = tp.TypeVar("PolygonDataT", bound="PolygonData")


class PolygonData(RemoteData):
    """Data class for fetching from Polygon.

    See https://github.com/polygon-io/client-python for API.

    See `PolygonData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.PolygonData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY"
        ...     )
        ... )
        ```

        * Fetch stock data:

        ```pycon
        >>> data = vbt.PolygonData.fetch(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        * Fetch crypto data:

        ```pycon
        >>> data = vbt.PolygonData.fetch(
        ...     "X:BTCUSD",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.polygon")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        **list_tickers_kwargs,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `CustomData.symbol_match` to check each symbol against `pattern`.

        For supported keyword arguments, see `polygon.RESTClient.list_tickers`."""
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        all_symbols = []
        for ticker in client.list_tickers(**list_tickers_kwargs):
            symbol = ticker.ticker
            if pattern is not None:
                if not cls.symbol_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)
        return sorted(all_symbols)

    @classmethod
    def resolve_client(cls, client: tp.Optional[PolygonClientT] = None, **client_config) -> PolygonClientT:
        """Resolve the client.

        If provided, must be of the type `polygon.rest.RESTClient`.
        Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("polygon")
        from polygon import RESTClient

        polygon_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = polygon_cfg["client"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(polygon_cfg["client_config"], client_config)
        if client is None:
            client = RESTClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        adjusted: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        params: tp.KwargsLike = None,
        delay: tp.Optional[float] = None,
        retries: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Polygon.

        Args:
            symbol (str): Symbol.

                Supports the following APIs:

                * Stocks and equities
                * Currencies - symbol must have the prefix `C:`
                * Crypto - symbol must have the prefix `X:`
            client (polygon.rest.RESTClient): Client.

                See `PolygonData.resolve_client`.
            client_config (dict): Client config.

                See `PolygonData.resolve_client`.
            start (any): The start of the aggregate time window.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): The end of the aggregate time window.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (str): Whether the results are adjusted for splits.

                By default, results are adjusted.
                Set this to False to get results that are NOT adjusted for splits.
            limit (int): Limits the number of base aggregates queried to create the aggregate results.

                Max 50000 and Default 5000.
            params (dict): Any additional query params.
            delay (float): Time to sleep after each request (in milliseconds).
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
            silence_warnings (bool): Whether to silence all warnings.

        For defaults, see `custom.polygon` in `vectorbtpro._settings.data`.

        !!! note
            If you're using a free plan that has an API rate limit of several requests per minute,
            make sure to set `delay` to a higher number, such as 12000 (which makes 5 requests per minute).
        """
        polygon_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if start is None:
            start = polygon_cfg["start"]
        if end is None:
            end = polygon_cfg["end"]
        if timeframe is None:
            timeframe = polygon_cfg["timeframe"]
        if tz is None:
            tz = polygon_cfg["tz"]
        if adjusted is None:
            adjusted = polygon_cfg["adjusted"]
        if limit is None:
            limit = polygon_cfg["limit"]
        params = merge_dicts(polygon_cfg["params"], params)
        if delay is None:
            delay = polygon_cfg["delay"]
        if retries is None:
            retries = polygon_cfg["retries"]
        if show_progress is None:
            show_progress = polygon_cfg["show_progress"]
        pbar_kwargs = merge_dicts(polygon_cfg["pbar_kwargs"], pbar_kwargs)
        if silence_warnings is None:
            silence_warnings = polygon_cfg["silence_warnings"]

        # Resolve the timeframe
        freq = prepare_freq(timeframe)
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        split = split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "t":
            unit = "minute"
        elif unit == "h":
            unit = "hour"
        elif unit == "d":
            unit = "day"
        elif unit == "W":
            unit = "week"
        elif unit == "M":
            unit = "month"
        elif unit == "Q":
            unit = "quarter"
        elif unit == "Y":
            unit = "year"

        # Establish the timestamps
        if start is not None:
            start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
        else:
            start_ts = None
        if end is not None:
            end_ts = datetime_to_ms(to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
        else:
            end_ts = None
        prev_end_ts = None

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except requests.exceptions.HTTPError as e:
                        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                            if not silence_warnings:
                                warnings.warn(traceback.format_exc(), stacklevel=2)
                                # Polygon.io API rate limit is per minute
                                warnings.warn("Waiting 1 minute...", stacklevel=2)
                            time.sleep(60)
                        else:
                            raise e
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if i == retries - 1:
                            raise e
                        if not silence_warnings:
                            warnings.warn(traceback.format_exc(), stacklevel=2)
                        if delay is not None:
                            time.sleep(delay / 1000)

            return retry_method

        def _postprocess(agg):
            return dict(
                o=agg.open,
                h=agg.high,
                l=agg.low,
                c=agg.close,
                v=agg.volume,
                vw=agg.vwap,
                t=agg.timestamp,
                n=agg.transactions,
            )

        @_retry
        def _fetch(_start_ts, _limit):
            return list(
                map(
                    _postprocess,
                    client.get_aggs(
                        ticker=symbol,
                        multiplier=multiplier,
                        timespan=unit,
                        from_=_start_ts,
                        to=end_ts,
                        adjusted=adjusted,
                        sort="asc",
                        limit=_limit,
                        params=params,
                        raw=False,
                    ),
                )
            )

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "/"
            return str(pd.Timestamp(ts, unit="ms", tz="utc"))

        def _filter_func(d: tp.Dict, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d["t"] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d["t"] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d["t"] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = _fetch(start_ts if prev_end_ts is None else prev_end_ts, limit)
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0]["t"]
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1]["t"]),
                        )
                    )
                    pbar.update(1)
                    prev_end_ts = next_data[-1]["t"]
                    if end_ts is not None and prev_end_ts >= end_ts:
                        break
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                    "Use update() method to fetch missing data.",
                    stacklevel=2,
                )

        df = pd.DataFrame(data)
        df = df[["t", "o", "h", "l", "c", "v", "n", "vw"]]
        df = df.rename(
            columns={
                "t": "Open time",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "n": "Trade count",
                "vw": "VWAP",
            }
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        return df, dict(tz_convert=tz, freq=freq)


PolygonData.override_column_config_doc(__pdoc__)

AVDataT = tp.TypeVar("AVDataT", bound="AVData")


class AVData(RemoteData):
    """Data class for fetching from Alpha Vantage.

    See https://www.alphavantage.co/documentation/ for API.

    Instead of using https://github.com/RomelTorres/alpha_vantage package, which is stale and has
    many issues, this class parses the API documentation with `AVData.parse_api_meta` using
    `BeautifulSoup4` and builds the API query based on this metadata. It then uses
    [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) to collect
    and format the CSV data.

    This approach is the most flexible we can get since we can instantly react to Alpha Vantage's changes
    in the API. If the data provider changes its API documentation, you can always adapt the parsing
    procedure by overriding `AVData.parse_api_meta`.

    If parser still fails, you can disable parsing entirely and specify all information manually
    by setting `function` and disabling `match_params`

    See `AVData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.AVData.set_custom_settings(
        ...     apikey="YOUR_KEY"
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.AVData.fetch(
        ...     "GOOGL",
        ...     timeframe="1 day",  # premium?
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "BTC_USD",
        ...     timeframe="30 minutes",  # premium?
        ...     category="digital-currency",
        ...     outputsize="full"
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "REAL_GDP",
        ...     category="economic-indicators"
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "IBM",
        ...     category="technical-indicators",
        ...     function="STOCHRSI",
        ...     params=dict(fastkperiod=14)
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.av")

    @classmethod
    def list_symbols(cls, keywords: str, apikey: tp.Optional[str] = None) -> tp.List[str]:
        """List all symbols."""
        av_cfg = cls.get_settings(key_id="custom")

        if apikey is None:
            apikey = av_cfg["apikey"]
        query = dict()
        query["function"] = "SYMBOL_SEARCH"
        query["keywords"] = keywords
        query["datatype"] = "csv"
        query["apikey"] = apikey
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(query)
        df = pd.read_csv(url)
        return sorted(df["symbol"].tolist())

    @classmethod
    @lru_cache()
    def parse_api_meta(cls) -> dict:
        """Parse API metadata from the documentation at https://www.alphavantage.co/documentation

        Cached class method. To avoid re-parsing the same metadata in different runtimes, save it manually."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup

        page = requests.get("https://www.alphavantage.co/documentation")
        soup = BeautifulSoup(page.content, "html.parser")
        api_meta = {}
        for section in soup.select("article section"):
            category = {}
            function = None
            function_args = dict(req_args=set(), opt_args=set())
            for tag in section.find_all(True):
                if tag.name == "h6":
                    if function is not None and tag.select("b")[0].getText().strip() == "API Parameters":
                        category[function] = function_args
                        function = None
                        function_args = dict(req_args=set(), opt_args=set())
                if tag.name == "b":
                    b_text = tag.getText().strip()
                    if b_text.startswith("❚ Required"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["req_args"].add(arg)
                if tag.name == "p":
                    p_text = tag.getText().strip()
                    if p_text.startswith("❚ Optional"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["opt_args"].add(arg)
                if tag.name == "code":
                    code_text = tag.getText().strip()
                    if code_text.startswith("function="):
                        function = code_text.replace("function=", "")
            if function is not None:
                category[function] = function_args
            api_meta[section.select("h2")[0]["id"]] = category

        return api_meta

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        apikey: tp.Optional[str] = None,
        api_meta: tp.Optional[dict] = None,
        category: tp.Optional[str] = None,
        function: tp.Optional[str] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        adjusted: tp.Optional[bool] = None,
        extended: tp.Optional[bool] = None,
        slice: tp.Optional[str] = None,
        series_type: tp.Optional[str] = None,
        time_period: tp.Optional[int] = None,
        outputsize: tp.Optional[str] = None,
        match_params: tp.Optional[bool] = None,
        params: tp.KwargsLike = None,
        read_csv_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpha Vantage.

        See https://www.alphavantage.co/documentation/ for API endpoints and their parameters.

        !!! note
            Supports the CSV format only.

        Args:
            symbol (str): Symbol.

                May combine symbol/from_currency and market/to_currency using an underscore.
            apikey (str): API key.
            api_meta (dict): API meta.

                If None, will use `AVData.parse_api_meta` if `function` is not provided
                or `match_params` is True.
            category (str): API category of your choice.

                Used if `function` is not provided or `match_params` is True.

                Supported are:

                * "time-series-data"
                * "fundamentals"
                * "fx"
                * "digital-currency"
                * "economic-indicators"
                * "technical-indicators"
            function (str): API function of your choice.

                If None, will try to resolve it based on other arguments, such as `timeframe`,
                `adjusted`, and `extended`. Required for technical indicators, economic indicators,
                and fundamental data.

                See the keys in sub-dictionaries returned by `AVData.parse_api_meta`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".

                For time series, forex, and crypto, looks for interval type in the function's name.
                Defaults to "60min" if extended, otherwise to "daily".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (bool): Whether to return time series adjusted by historical split and dividend events.
            extended (bool): Whether to return historical intraday time series for the trailing 2 years.
            slice (str): Slice of the trailing 2 years.
            series_type (str): The desired price type in the time series.
            time_period (int): Number of data points used to calculate each window value.
            outputsize (str): Output size.

                Supported are

                * "compact" that returns only the latest 100 data points
                * "full" that returns the full-length time series
            match_params (bool): Whether to match parameters with the ones required by the endpoint.

                Otherwise, uses only (resolved) `function`, `apikey`, `datatype="csv"`, and `params`.
            params: Additional keyword arguments passed as key/value pairs in the URL.
            read_csv_kwargs (dict): Keyword arguments passed to `pd.read_csv`.
            silence_warnings (bool): Whether to silence all warnings.

        For defaults, see `custom.av` in `vectorbtpro._settings.data`.
        """
        av_cfg = cls.get_settings(key_id="custom")

        if apikey is None:
            apikey = av_cfg["apikey"]
        if api_meta is None:
            api_meta = av_cfg["api_meta"]
        if category is None:
            category = av_cfg["category"]
        if function is None:
            function = av_cfg["function"]
        if timeframe is None:
            timeframe = av_cfg["timeframe"]
        if tz is None:
            tz = av_cfg["tz"]
        if adjusted is None:
            adjusted = av_cfg["adjusted"]
        if extended is None:
            extended = av_cfg["extended"]
        if slice is None:
            slice = av_cfg["slice"]
        if series_type is None:
            series_type = av_cfg["series_type"]
        if time_period is None:
            time_period = av_cfg["time_period"]
        if outputsize is None:
            outputsize = av_cfg["outputsize"]
        read_csv_kwargs = merge_dicts(av_cfg["read_csv_kwargs"], read_csv_kwargs)
        if match_params is None:
            match_params = av_cfg["match_params"]
        params = merge_dicts(av_cfg["params"], params)
        if silence_warnings is None:
            silence_warnings = av_cfg["silence_warnings"]

        if api_meta is None and (function is None or match_params):
            if not silence_warnings and cls.parse_api_meta.cache_info().misses == 0:
                warnings.warn("Parsing API documentation...", stacklevel=2)
            try:
                api_meta = cls.parse_api_meta()
            except Exception as e:
                raise ValueError("Can't fetch/parse the API documentation. Specify function and disable match_params.")

        # Resolve the timeframe
        freq = prepare_freq(timeframe)
        interval = None
        interval_type = None
        if timeframe is not None:
            if not isinstance(timeframe, str):
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            split = split_freq_str(timeframe)
            if split is None:
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            multiplier, unit = split
            if unit == "t":
                interval = str(multiplier) + "min"
                interval_type = "INTRADAY"
            elif unit == "h":
                interval = str(60 * multiplier) + "min"
                interval_type = "INTRADAY"
            elif unit == "d":
                interval = "daily"
                interval_type = "DAILY"
            elif unit == "W":
                interval = "weekly"
                interval_type = "WEEKLY"
            elif unit == "M":
                interval = "monthly"
                interval_type = "MONTHLY"
            elif unit == "Q":
                interval = "quarterly"
                interval_type = "QUARTERLY"
            elif unit == "Y":
                interval = "annual"
                interval_type = "ANNUAL"
            if interval is None and multiplier > 1:
                raise ValueError("Multipliers are supported only for intraday timeframes")
        else:
            if extended:
                interval_type = "INTRADAY"
                interval = "60min"
            else:
                interval_type = "DAILY"
                interval = "daily"

        # Resolve the function
        if function is None and category is not None and category == "economic-indicators":
            function = symbol
        if function is None:
            if category is None:
                category = "time-series-data"
            if category in ("technical-indicators", "fundamentals"):
                raise ValueError("Function is required")
            adjusted_in_functions = False
            extended_in_functions = False
            matched_functions = []
            for k, v in api_meta[category].items():
                if interval_type is None or interval_type in k:
                    if "ADJUSTED" in k:
                        adjusted_in_functions = True
                    if "EXTENDED" in k:
                        extended_in_functions = True
                    matched_functions.append(k)

            if adjusted_in_functions:
                matched_functions = [
                    k
                    for k in matched_functions
                    if (adjusted and "ADJUSTED" in k) or (not adjusted and "ADJUSTED" not in k)
                ]
            if extended_in_functions:
                matched_functions = [
                    k
                    for k in matched_functions
                    if (extended and "EXTENDED" in k) or (not extended and "EXTENDED" not in k)
                ]
            if len(matched_functions) == 0:
                raise ValueError("No functions satisfy the requirements")
            if len(matched_functions) > 1:
                raise ValueError("More than one function satisfies the requirements")
            function = matched_functions[0]

        # Resolve the parameters
        if match_params:
            if function is not None and category is None:
                category = None
                for k, v in api_meta.items():
                    if function in v:
                        category = k
                        break
            if category is None:
                raise ValueError("Category is required")
            req_args = api_meta[category][function]["req_args"]
            opt_args = api_meta[category][function]["opt_args"]
            args = set(req_args) | set(opt_args)

            matched_params = dict()
            matched_params["function"] = function
            matched_params["datatype"] = "csv"
            matched_params["apikey"] = apikey
            if "symbol" in args and "market" in args:
                matched_params["symbol"] = symbol.split("_")[0]
                matched_params["market"] = symbol.split("_")[1]
            elif "from_" in args and "to_currency" in args:
                matched_params["from_currency"] = symbol.split("_")[0]
                matched_params["to_currency"] = symbol.split("_")[1]
            elif "from_currency" in args and "to_currency" in args:
                matched_params["from_currency"] = symbol.split("_")[0]
                matched_params["to_currency"] = symbol.split("_")[1]
            elif "symbol" in args:
                matched_params["symbol"] = symbol
            if "interval" in args:
                matched_params["interval"] = interval
            if "adjusted" in args:
                matched_params["adjusted"] = adjusted
            if "extended" in args:
                matched_params["extended"] = extended
            if "slice" in args:
                matched_params["slice"] = slice
            if "series_type" in args:
                matched_params["series_type"] = series_type
            if "time_period" in args:
                matched_params["time_period"] = time_period
            if "outputsize" in args:
                matched_params["outputsize"] = outputsize
            for k, v in params.items():
                if k in args:
                    matched_params[k] = v
                else:
                    raise ValueError(f"Function '{function}' does not expect parameter '{k}'")
            for arg in req_args:
                if arg not in matched_params:
                    raise ValueError(f"Function '{function}' requires parameter '{arg}'")
        else:
            matched_params = dict(params)
            matched_params["function"] = function
            matched_params["apikey"] = apikey
            matched_params["datatype"] = "csv"

        # Collect and format the data
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(matched_params)
        df = pd.read_csv(url, **read_csv_kwargs)
        df.index.name = None
        new_columns = []
        for c in df.columns:
            new_c = re.sub(r"^\d+\w*\.\s*", "", c)
            new_c = new_c[0].title() + new_c[1:]
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        if not df.empty and df.index[0] > df.index[1]:
            df = df.iloc[::-1]
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

        return df, dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        raise NotImplementedError


NDLDataT = tp.TypeVar("NDLDataT", bound="NDLData")


class NDLData(RemoteData):
    """Data class for fetching from Nasdaq Data Link.

    See https://github.com/Nasdaq/data-link-python for API.

    See `NDLData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.NDLData.set_custom_settings(
        ...     api_key="YOUR_KEY"
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.NDLData.fetch(
        ...     "EIA/PET_RWTC_D",
        ...     start="2020-01-01",
        ...     end="2021-01-01"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.ndl")

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        api_key: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        column_indices: tp.Optional[tp.MaybeIterable[int]] = None,
        collapse: tp.Optional[str] = None,
        transform: tp.Optional[str] = None,
        **params,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Nasdaq Data Link.

        Args:
            symbol (str): Symbol.
            api_key (str): API key.
            start (any): Retrieve data rows on and after the specified start date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): Retrieve data rows up to and including the specified end date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            column_indices (int or iterable): Request one or more specific columns.

                Column 0 is the date column and is always returned. Data begins at column 1.
            collapse (str): Change the sampling frequency of the returned data.

                Options are "daily", "weekly", "monthly", "quarterly", and "annual".
            transform (str): Perform elementary calculations on the data prior to downloading.

                Options are "diff", "rdiff", "cumul", and "normalize".
            **params: Keyword arguments sent as field/value params to Nasdaq Data Link with no interference.

        For defaults, see `custom.ndl` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nasdaqdatalink")

        import nasdaqdatalink

        ndl_cfg = cls.get_settings(key_id="custom")

        if api_key is None:
            api_key = ndl_cfg["api_key"]
        if start is None:
            start = ndl_cfg["start"]
        if end is None:
            end = ndl_cfg["end"]
        if tz is None:
            tz = ndl_cfg["tz"]
        if column_indices is None:
            column_indices = ndl_cfg["column_indices"]
        if column_indices is not None:
            if isinstance(column_indices, int):
                dataset = symbol + "." + str(column_indices)
            else:
                dataset = [symbol + "." + str(index) for index in column_indices]
        else:
            dataset = symbol
        if collapse is None:
            collapse = ndl_cfg["collapse"]
        if transform is None:
            transform = ndl_cfg["transform"]
        params = merge_dicts(ndl_cfg["params"], params)

        # Establish the timestamps
        if start is not None:
            start = to_tzaware_datetime(start, naive_tz=tz, tz="UTC")
            start_date = pd.Timestamp(start).isoformat()
        else:
            start_date = None
        if end is not None:
            end = to_tzaware_datetime(end, naive_tz=tz, tz="UTC")
            end_date = pd.Timestamp(end).isoformat()
        else:
            end_date = None

        # Collect and format the data
        df = nasdaqdatalink.get(
            dataset,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            collapse=collapse,
            transform=transform,
            **params,
        )
        new_columns = []
        for c in df.columns:
            new_c = c
            if isinstance(symbol, str):
                new_c = new_c.replace(symbol + " - ", "")
            if new_c == "Last":
                new_c = "Close"
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")
        if not df.empty:
            if start is not None:
                start = to_timestamp(start, tz=df.index.tzinfo)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = to_timestamp(end, tz=df.index.tzinfo)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz_convert=tz)


TVDataT = tp.TypeVar("TVDataT", bound="TVData")


class TVData(RemoteData):
    """Data class for fetching from TradingView.

    See `TVData.fetch_symbol` for arguments.

    !!! note
        If you're getting the error "Please confirm that you are not a robot by clicking the captcha box."
        when attempting to authenticate, use `token` instead of `username` and `password`. To get the
        token, see [this issue](https://github.com/StreamAlpha/tvdatafeed/issues/96).

    Usage:
        * Set up the credentials globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.TVData.set_custom_settings(
        ...     client_config=dict(
        ...         username="YOUR_USERNAME",
        ...         password="YOUR_PASSWORD",
        ...         user_agent="YOUR_USER_AGENT"  # optional, see https://useragentstring.com/
        ...     )
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.TVData.fetch(
        ...     "NASDAQ:AAPL",
        ...     timeframe="1 hour"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.tv")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        market: tp.Optional[str] = None,
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        delay: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses market scanner when `market` is provided (returns all symbols, big payload)
        Uses symbol search when either `text` or `exchange` is provided (returns a subset of symbols)."""
        tv_cfg = cls.get_settings(key_id="custom")

        if market is None and text is None and exchange is None:
            raise ValueError("Please provide either market, or text and/or exchange")
        if market is not None and (text is not None or exchange is not None):
            raise ValueError("Please provide either market, or text and/or exchange")
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if delay is None:
            delay = tv_cfg["delay"]
        if show_progress is None:
            show_progress = tv_cfg["show_progress"]
        pbar_kwargs = merge_dicts(tv_cfg["pbar_kwargs"], pbar_kwargs)

        if market is None:
            data = client.search_symbol(
                text=text,
                exchange=exchange,
                delay=delay,
                show_progress=show_progress,
                pbar_kwargs=pbar_kwargs,
            )
            all_symbols = map(lambda x: x["exchange"] + ":" + x["symbol"], data)
        else:
            data = client.scan_symbols(market.lower())
            all_symbols = map(lambda x: x["s"], data)
        found_symbols = []
        for symbol in all_symbols:
            if pattern is not None:
                if not cls.symbol_match(symbol.split(":")[1], pattern, use_regex=use_regex):
                    continue
            found_symbols.append(symbol)
        return sorted(found_symbols)

    @classmethod
    def resolve_client(cls, client: tp.Optional[TVClient] = None, **client_config) -> TVClient:
        """Resolve the client.

        If provided, must be of the type `vectorbtpro.data.tv.TVClient`.
        Otherwise, will be created using `client_config`."""
        tv_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = tv_cfg["client"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(tv_cfg["client_config"], client_config)
        if client is None:
            client = TVClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[TVClient] = None,
        client_config: tp.KwargsLike = None,
        exchange: tp.Optional[str] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        fut_contract: tp.Optional[int] = None,
        adjustment: tp.Optional[str] = None,
        extended_session: tp.Optional[bool] = None,
        pro_data: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from TradingView.

        Args:
            symbol (str): Symbol.

                Symbol must be in the `EXCHANGE:SYMBOL` format if `exchange` is None.
            client (vectorbtpro.data.tv.TVClient): Client.

                See `TVData.resolve_client`.
            client_config (dict): Client config.

                See `TVData.resolve_client`.
            exchange (str): Exchange.

                Can be omitted if already provided via `symbol`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            fut_contract (int): None for cash, 1 for continuous current contract in front,
                2 for continuous next contract in front.
            adjustment (str): Adjustment.

                Either "splits" (default) or "dividends".
            extended_session (bool): Regular session if False, extended session if True.
            pro_data (bool): Whether to use pro data.
            limit (int): The maximum number of returned items.

        For defaults, see `custom.tv` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.data.tv import Interval

        tv_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if exchange is None:
            exchange = tv_cfg["exchange"]
        if timeframe is None:
            timeframe = tv_cfg["timeframe"]
        if tz is None:
            tz = tv_cfg["tz"]
        if fut_contract is None:
            fut_contract = tv_cfg["fut_contract"]
        if adjustment is None:
            adjustment = tv_cfg["adjustment"]
        if extended_session is None:
            extended_session = tv_cfg["extended_session"]
        if pro_data is None:
            pro_data = tv_cfg["pro_data"]
        if limit is None:
            limit = tv_cfg["limit"]

        freq = prepare_freq(timeframe)
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        split = split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "t":
            interval = getattr(Interval, f"in_{str(multiplier)}_minute")
        elif unit == "h":
            interval = getattr(Interval, f"in_{str(multiplier)}_hour")
        elif unit == "d":
            if multiplier > 1:
                raise ValueError("Multiplier cannot be greater than 1 for daily")
            interval = getattr(Interval, "in_daily")
        elif unit == "W":
            if multiplier > 1:
                raise ValueError("Multiplier cannot be greater than 1 for weekly")
            interval = getattr(Interval, "in_weekly")
        elif unit == "M":
            if multiplier > 1:
                raise ValueError("Multiplier cannot be greater than 1 for monthly")
            interval = getattr(Interval, "in_monthly")
        else:
            raise ValueError(f"Invalid timeframe '{timeframe}'")

        df = client.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            fut_contract=fut_contract,
            adjustment=adjustment,
            extended_session=extended_session,
            pro_data=pro_data,
            limit=limit,
        )
        df.rename(
            columns={
                "symbol": "Symbol",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

        if "Symbol" in df:
            del df["Symbol"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

        return df, dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


TVData.override_column_config_doc(__pdoc__)
