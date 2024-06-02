# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for scheduling data saves."""

import logging

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data
from vectorbtpro.data.updater import DataUpdater
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "DataSaver",
    "CSVDataSaver",
    "HDFDataSaver",
]

logger = logging.getLogger(__name__)


class DataSaver(DataUpdater):
    """Base class for scheduling data saves.

    Subclasses `vectorbtpro.data.updater.DataUpdater`.

    Args:
        data (Data): Data instance.
        save_kwargs (dict): Default keyword arguments for `DataSaver.init_save_data` and `DataSaver.save_data`.
        init_save_kwargs (dict): Default keyword arguments overriding `save_kwargs` for `DataSaver.init_save_data`.
        **kwargs: Keyword arguments passed to the constructor of `DataUpdater`.
    """

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (DataUpdater._expected_keys or set()) | {
        "save_kwargs",
        "init_save_kwargs",
    }

    def __init__(
        self,
        data: Data,
        save_kwargs: tp.KwargsLike = None,
        init_save_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        DataUpdater.__init__(
            self,
            data=data,
            save_kwargs=save_kwargs,
            init_save_kwargs=init_save_kwargs,
            **kwargs,
        )
        self._save_kwargs = save_kwargs
        self._init_save_kwargs = init_save_kwargs

    @property
    def save_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments passed to `DataSaver.save_data`."""
        return self._save_kwargs

    @property
    def init_save_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments passed to `DataSaver.init_save_data`."""
        return self._init_save_kwargs

    def init_save_data(self, **kwargs) -> None:
        """Save initial data.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def save_data(self, **kwargs) -> None:
        """Save data.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def update(self, save_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """Update and save data using `DataSaver.save_data`.

        Override to do pre- and postprocessing.

        To stop this method from running again, raise `vectorbtpro.utils.schedule_.CancelledError`."""
        # In case the method was called by the user
        kwargs = merge_dicts(
            dict(save_kwargs=self.save_kwargs), self.update_kwargs, {"save_kwargs": save_kwargs, **kwargs}
        )
        save_kwargs = kwargs.pop("save_kwargs")

        self._data = self.data.update(concat=False, **kwargs)
        self.update_config(data=self.data)
        if save_kwargs is None:
            save_kwargs = {}
        self.save_data(**save_kwargs)

    def update_every(
        self,
        *args,
        save_kwargs: tp.KwargsLike = None,
        init_save: bool = False,
        init_save_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        """Overrides `vectorbtpro.data.updater.DataUpdater` to save initial data prior to updating."""
        if init_save:
            init_save_kwargs = merge_dicts(
                self.save_kwargs,
                save_kwargs,
                self.init_save_kwargs,
                init_save_kwargs,
            )
            self.init_save_data(**init_save_kwargs)
        DataUpdater.update_every(self, *args, save_kwargs=save_kwargs, **kwargs)


class CSVDataSaver(DataSaver):
    """Subclass of `DataSaver` for saving data to CSV files using `vectorbtpro.data.base.Data.to_csv`."""

    def init_save_data(self, **to_csv_kwargs) -> None:
        """Save initial data."""
        # In case the method was called by the user
        to_csv_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_csv_kwargs,
        )

        self._data.to_csv(**to_csv_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_csv_kwargs) -> None:
        """Save data.

        By default, appends new data without header."""
        # In case the method was called by the user
        to_csv_kwargs = merge_dicts(
            dict(mode="a", header=False),
            self.save_kwargs,
            to_csv_kwargs,
        )

        self._data.to_csv(**to_csv_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")


class HDFDataSaver(DataSaver):
    """Subclass of `DataSaver` for saving data to HDF files using `vectorbtpro.data.base.Data.to_hdf`."""

    def init_save_data(self, **to_hdf_kwargs) -> None:
        """Save initial data."""
        # In case the method was called by the user
        to_hdf_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_hdf_kwargs,
        )

        self._data.to_hdf(**to_hdf_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_hdf_kwargs) -> None:
        """Save data.

        By default, appends new data in a table format."""
        # In case the method was called by the user
        to_hdf_kwargs = merge_dicts(
            dict(mode="a", append=True),
            self.save_kwargs,
            to_hdf_kwargs,
        )

        self._data.to_hdf(**to_hdf_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")
