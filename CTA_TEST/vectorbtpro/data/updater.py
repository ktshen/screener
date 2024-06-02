# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for scheduling data updates."""

import logging

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data
from vectorbtpro.utils.config import Configured, merge_dicts
from vectorbtpro.utils.schedule_ import ScheduleManager

__all__ = [
    "DataUpdater",
]

logger = logging.getLogger(__name__)


class DataUpdater(Configured):
    """Base class for scheduling data updates.

    Args:
        data (Data): Data instance.
        update_kwargs (dict): Default keyword arguments for `DataSaver.update`.
        **kwargs: Keyword arguments passed to the constructor of `Configured`.
    """

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "data",
        "schedule_manager",
        "update_kwargs",
    }

    def __init__(
        self,
        data: Data,
        schedule_manager: tp.Optional[ScheduleManager] = None,
        update_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        if schedule_manager is None:
            schedule_manager = ScheduleManager()
        Configured.__init__(
            self,
            data=data,
            schedule_manager=schedule_manager,
            update_kwargs=update_kwargs,
            **kwargs,
        )

        self._data = data
        self._schedule_manager = schedule_manager
        self._update_kwargs = update_kwargs

    @property
    def data(self) -> Data:
        """Data instance.

        See `vectorbtpro.data.base.Data`."""
        return self._data

    @property
    def schedule_manager(self) -> ScheduleManager:
        """Schedule manager instance.

        See `vectorbtpro.utils.schedule_.ScheduleManager`."""
        return self._schedule_manager

    @property
    def update_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments passed to `DataSaver.update`."""
        return self._update_kwargs

    def update(self, **kwargs) -> None:
        """Method that updates data.

        Override to do pre- and postprocessing.

        To stop this method from running again, raise `vectorbtpro.utils.schedule_.CancelledError`."""
        # In case the method was called by the user
        kwargs = merge_dicts(self.update_kwargs, kwargs)

        self._data = self.data.update(**kwargs)
        self.update_config(data=self.data)
        new_index = self.data.wrapper.index
        logger.info(f"New data has {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def update_every(
        self,
        *args,
        to: int = None,
        tags: tp.Optional[tp.Iterable[tp.Hashable]] = None,
        in_background: bool = False,
        replace: bool = True,
        start: bool = True,
        start_kwargs: tp.KwargsLike = None,
        **update_kwargs,
    ) -> None:
        """Schedule `DataUpdater.update` as a job.

        For `*args`, `to` and `tags`, see `vectorbtpro.utils.schedule_.ScheduleManager.every`.

        If `in_background` is set to True, starts in the background as an `asyncio` task.
        The task can be stopped with `vectorbtpro.utils.schedule_.ScheduleManager.stop`.

        If `replace` is True, will delete scheduled jobs with the same tags, or all jobs if tags are omitted.

        If `start` is False, will add the job to the scheduler without starting.

        `**update_kwargs` are merged over `DataUpdater.update_kwargs` and passed to `DataUpdater.update`."""
        if replace:
            self.schedule_manager.clear_jobs(tags)
        update_kwargs = merge_dicts(self.update_kwargs, update_kwargs)
        self.schedule_manager.every(*args, to=to, tags=tags).do(self.update, **update_kwargs)
        if start:
            if start_kwargs is None:
                start_kwargs = {}
            if in_background:
                self.schedule_manager.start_in_background(**start_kwargs)
            else:
                self.schedule_manager.start(**start_kwargs)
