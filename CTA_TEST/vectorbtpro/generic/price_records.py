# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Base class for working with records that can make use of OHLC data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.generic import nb
from vectorbtpro.records.base import Records
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.records.decorators import attach_shortcut_properties
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "PriceRecords",
]

__pdoc__ = {}

price_records_shortcut_config = ReadonlyConfig(
    dict(
        bar_open_time=dict(obj_type="mapped"),
        bar_close_time=dict(obj_type="mapped"),
        bar_open=dict(obj_type="mapped"),
        bar_high=dict(obj_type="mapped"),
        bar_low=dict(obj_type="mapped"),
        bar_close=dict(obj_type="mapped"),
    )
)
"""_"""

__pdoc__[
    "price_records_shortcut_config"
] = f"""Config of shortcut properties to be attached to `PriceRecords`.

```python
{price_records_shortcut_config.prettify()}
```
"""

PriceRecordsT = tp.TypeVar("PriceRecordsT", bound="PriceRecords")


@attach_shortcut_properties(price_records_shortcut_config)
class PriceRecords(Records):
    """Extends `vectorbtpro.records.base.Records` for records that can make use of OHLC data."""

    @classmethod
    def from_records(
        cls: tp.Type[PriceRecordsT],
        wrapper: ArrayWrapper,
        records: tp.RecordArray,
        data: tp.Optional["Data"] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        attach_data: bool = True,
        **kwargs,
    ) -> PriceRecordsT:
        """Build `PriceRecords` from records."""
        if open is None and data is not None:
            open = data.open
        if high is None and data is not None:
            high = data.high
        if low is None and data is not None:
            low = data.low
        if close is None and data is not None:
            close = data.close
        return cls(
            wrapper,
            records,
            open=open if attach_data else None,
            high=high if attach_data else None,
            low=low if attach_data else None,
            close=close if attach_data else None,
            **kwargs,
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeTuple[PriceRecordsT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PriceRecords` after stacking along columns."""
        kwargs = Records.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    kwargs[price_name] = kwargs["wrapper"].row_stack_arrs(
                        *price_objs,
                        group_by=False,
                        wrap=False,
                    )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeTuple[PriceRecordsT],
        reindex_kwargs: tp.KwargsLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PriceRecords` after stacking along columns."""
        kwargs = Records.resolve_column_stack_kwargs(*objs, reindex_kwargs=reindex_kwargs, **kwargs)
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, "_" + price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    new_price = kwargs["wrapper"].column_stack_arrs(
                        *price_objs,
                        reindex_kwargs=reindex_kwargs,
                        group_by=False,
                        wrap=True,
                    )
                    if price_name == "close":
                        if fbfill_close:
                            new_price = new_price.vbt.fbfill()
                        elif ffill_close:
                            new_price = new_price.vbt.ffill()
                    kwargs[price_name] = new_price.values
        return kwargs

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Records._expected_keys or set()) | {
        "open",
        "high",
        "low",
        "close",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        Records.__init__(
            self,
            wrapper,
            records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

        if open is not None:
            open = to_2d_array(open)
        if high is not None:
            high = to_2d_array(high)
        if low is not None:
            low = to_2d_array(low)
        if close is not None:
            close = to_2d_array(close)

        self._open = open
        self._high = high
        self._low = low
        self._close = close

    def indexing_func_meta(self, *args, records_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `PriceRecords` and return metadata."""
        if records_meta is None:
            records_meta = Records.indexing_func_meta(self, *args, **kwargs)
        prices = {}
        for price_name in ("open", "high", "low", "close"):
            if getattr(self, "_" + price_name) is not None:
                new_price = ArrayWrapper.select_from_flex_array(
                    getattr(self, "_" + price_name),
                    row_idxs=records_meta["wrapper_meta"]["row_idxs"],
                    col_idxs=records_meta["wrapper_meta"]["col_idxs"],
                    rows_changed=records_meta["wrapper_meta"]["rows_changed"],
                    columns_changed=records_meta["wrapper_meta"]["columns_changed"],
                )
            else:
                new_price = None
            prices[price_name] = new_price
        return {**records_meta, **prices}

    def indexing_func(self: PriceRecordsT, *args, price_records_meta: tp.DictLike = None, **kwargs) -> PriceRecordsT:
        """Perform indexing on `PriceRecords`."""
        if price_records_meta is None:
            price_records_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=price_records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=price_records_meta["new_records_arr"],
            open=price_records_meta["open"],
            high=price_records_meta["high"],
            low=price_records_meta["low"],
            close=price_records_meta["close"],
        )

    def resample(
        self: PriceRecordsT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        records_meta: tp.DictLike = None,
        **kwargs,
    ) -> PriceRecordsT:
        """Perform resampling on `PriceRecords`."""
        if records_meta is None:
            records_meta = self.resample_meta(*args, **kwargs)
        if self._open is None:
            new_open = None
        else:
            new_open = self.open.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.first_reduce_nb,
            )
        if self._high is None:
            new_high = None
        else:
            new_high = self.high.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.max_reduce_nb,
            )
        if self._low is None:
            new_low = None
        else:
            new_low = self.low.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.min_reduce_nb,
            )
        if self._close is None:
            new_close = None
        else:
            new_close = self.close.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.last_reduce_nb,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
        )

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open price."""
        if self._open is None:
            return None
        return self.wrapper.wrap(self._open, group_by=False)

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High price."""
        if self._high is None:
            return None
        return self.wrapper.wrap(self._high, group_by=False)

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low price."""
        if self._low is None:
            return None
        return self.wrapper.wrap(self._low, group_by=False)

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close price."""
        if self._close is None:
            return None
        return self.wrapper.wrap(self._close, group_by=False)

    def get_bar_open_time(self, **kwargs) -> MappedArray:
        """Get a mapped array with the opening time of the bar."""
        return self.map_array(self.wrapper.index[self.idx_arr], **kwargs)

    def get_bar_close_time(self, **kwargs) -> MappedArray:
        """Get a mapped array with the closing time of the bar."""
        if self.wrapper.freq is None:
            raise ValueError("Frequency must be provided")
        return self.map_array(Resampler.get_rbound_index(
            index=self.wrapper.index[self.idx_arr],
            freq=self.wrapper.freq
        ), **kwargs)

    def get_bar_open(self, **kwargs) -> MappedArray:
        """Get a mapped array with the opening price of the bar."""
        return self.apply(nb.bar_price_nb, self._open, **kwargs)

    def get_bar_high(self, **kwargs) -> MappedArray:
        """Get a mapped array with the high price of the bar."""
        return self.apply(nb.bar_price_nb, self._high, **kwargs)

    def get_bar_low(self, **kwargs) -> MappedArray:
        """Get a mapped array with the low price of the bar."""
        return self.apply(nb.bar_price_nb, self._low, **kwargs)

    def get_bar_close(self, **kwargs) -> MappedArray:
        """Get a mapped array with the closing price of the bar."""
        return self.apply(nb.bar_price_nb, self._close, **kwargs)
