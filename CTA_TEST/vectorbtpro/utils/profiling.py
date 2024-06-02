# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for profiling time and memory."""

import tracemalloc
from datetime import timedelta
from timeit import default_timer

import humanize

from vectorbtpro import _typing as tp

__all__ = [
    "Timer",
    "MemTracer",
]

TimerT = tp.TypeVar("TimerT", bound="Timer")


class Timer:
    """Context manager to measure execution time using `timeit`.

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt
        >>> import time

        >>> with vbt.Timer() as timer:
        >>>     time.sleep(1)

        >>> timer.elapsed()
        '1.01 seconds'

        >>> timer.elapsed(readable=False)
        datetime.timedelta(seconds=1, microseconds=5110)
        ```
    """

    def __init__(self) -> None:
        self._start_time = default_timer()
        self._end_time = None

    @property
    def start_time(self) -> float:
        """Start time."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time."""
        if self._end_time is None:
            return default_timer()
        return self._end_time

    def elapsed(self, readable: bool = True, **kwargs) -> tp.Union[str, timedelta]:
        """Get elapsed time.

        `**kwargs` are passed to `humanize.precisedelta`."""
        elapsed = self.end_time - self.start_time
        elapsed_delta = timedelta(seconds=elapsed)
        if readable:
            if "minimum_unit" not in kwargs:
                kwargs["minimum_unit"] = "seconds" if elapsed >= 1 else "milliseconds"
            return humanize.precisedelta(elapsed_delta, **kwargs)
        return elapsed_delta

    def __enter__(self: TimerT) -> TimerT:
        self._start_time = default_timer()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = default_timer()


MemTracerT = tp.TypeVar("MemTracerT", bound="MemTracer")


class MemTracer:
    """Context manager to trace peak and final memory usage using `tracemalloc`.

    Usage:
        ```pycon
        >>> import vectorbtpro as vbt
        >>> import numpy as np

        >>> with vbt.MemTracer() as tracer:
        >>>     np.random.uniform(size=1000000)

        >>> tracer.peak_usage()
        '8.0 MB'

        >>> tracer.peak_usage(readable=False)
        8005360
        ```
    """

    def __init__(self) -> None:
        self._final_usage = None
        self._peak_usage = None

    def final_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get final memory usage.

        `**kwargs` are passed to `humanize.naturalsize`."""
        if self._final_usage is None:
            final_usage = tracemalloc.get_traced_memory()[0]
        else:
            final_usage = self._final_usage
        if readable:
            return humanize.naturalsize(final_usage, **kwargs)
        return final_usage

    def peak_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get peak memory usage.

        `**kwargs` are passed to `humanize.naturalsize`."""
        if self._peak_usage is None:
            peak_usage = tracemalloc.get_traced_memory()[1]
        else:
            peak_usage = self._peak_usage
        if readable:
            return humanize.naturalsize(peak_usage, **kwargs)
        return peak_usage

    def __enter__(self: MemTracerT) -> MemTracerT:
        tracemalloc.start()
        tracemalloc.clear_traces()
        return self

    def __exit__(self, *args) -> None:
        self._final_usage, self._peak_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()
