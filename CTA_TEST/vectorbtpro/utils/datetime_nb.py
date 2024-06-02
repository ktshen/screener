# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled utilities for working with dates and time."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []

us_ns = 1000
"""Microsecond (nanoseconds)."""

ms_ns = us_ns * 1000
"""Millisecond (nanoseconds)."""

s_ns = ms_ns * 1000
"""Second (nanoseconds)."""

m_ns = s_ns * 60
"""Minute (nanoseconds)."""

h_ns = m_ns * 60
"""Hour (nanoseconds)."""

d_ns = h_ns * 24
"""Day (nanoseconds)."""

w_ns = d_ns * 7
"""Week (nanoseconds)."""

ns_td = np.timedelta64(1, "ns")
"""Nanosecond (timedelta)."""

us_td = us_ns * ns_td
"""Microsecond (timedelta)."""

ms_td = ms_ns * ns_td
"""Millisecond (timedelta)."""

s_td = s_ns * ns_td
"""Second (timedelta)."""

m_td = m_ns * ns_td
"""Minute (timedelta)."""

h_td = h_ns * ns_td
"""Hour (timedelta)."""

d_td = d_ns * ns_td
"""Day (timedelta)."""

w_td = w_ns * ns_td
"""Week (timedelta)."""

unix_epoch_dt = np.datetime64(0, "ns")
"""Unix epoch (datetime)."""


@register_jitted(cache=True)
def nanosecond_nb(ts: int) -> int:
    """Get the nanosecond."""
    return ts % 1000


@register_jitted(cache=True)
def microseconds_nb(ts: int) -> int:
    """Get the number of microseconds."""
    return ts // us_ns


@register_jitted(cache=True)
def microsecond_nb(ts: int) -> int:
    """Get the microsecond."""
    return microseconds_nb(ts) % (ms_ns // us_ns)


@register_jitted(cache=True)
def milliseconds_nb(ts: int) -> int:
    """Get the number of milliseconds."""
    return ts // ms_ns


@register_jitted(cache=True)
def millisecond_nb(ts: int) -> int:
    """Get the millisecond."""
    return milliseconds_nb(ts) % (s_ns // ms_ns)


@register_jitted(cache=True)
def seconds_nb(ts: int) -> int:
    """Get the number of seconds."""
    return ts // s_ns


@register_jitted(cache=True)
def second_nb(ts: int) -> int:
    """Get the seconds."""
    return seconds_nb(ts) % (m_ns // s_ns)


@register_jitted(cache=True)
def minutes_nb(ts: int) -> int:
    """Get the number of minutes."""
    return ts // m_ns


@register_jitted(cache=True)
def minute_nb(ts: int) -> int:
    """Get the minute."""
    return minutes_nb(ts) % (h_ns // m_ns)


@register_jitted(cache=True)
def hours_nb(ts: int) -> int:
    """Get the number of hours."""
    return ts // h_ns


@register_jitted(cache=True)
def hour_nb(ts: int) -> int:
    """Get the hour."""
    return hours_nb(ts) % (d_ns // h_ns)


@register_jitted(cache=True)
def days_nb(ts: int) -> int:
    """Get the number of hours."""
    return ts // d_ns


@register_jitted(cache=True)
def to_civil_nb(ts: int) -> tp.Tuple[int, int, int]:
    """Convert a timestamp into a tuple of the year, month, and day."""
    z = days_nb(ts)
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return y + (m <= 2), m, d


@register_jitted(cache=True)
def from_civil_nb(y: int, m: int, d: int) -> int:
    """Convert a year, month, and day into the timestamp."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return d_ns * days


@register_jitted(cache=True)
def matches_date_nb(ts: int, y: int, m: int, d: int) -> int:
    """Check whether the timestamp match the date provided in the civil format."""
    midnight_ts1 = midnight_nb(ts)
    midnight_ts2 = from_civil_nb(y, m, d)
    return midnight_ts1 == midnight_ts2


@register_jitted(cache=True)
def day_nb(ts: int) -> int:
    """Get the day of the month."""
    y, m, d = to_civil_nb(ts)
    return d


@register_jitted(cache=True)
def midnight_nb(ts: int) -> int:
    """Get the midnight of this day."""
    return ts - ts % d_ns


@register_jitted(cache=True)
def day_changed_nb(ts1: int, ts2: int) -> bool:
    """Whether the day changed."""
    return midnight_nb(ts1) != midnight_nb(ts2)


@register_jitted(cache=True)
def weekday_from_days_nb(days: int, zero_start: bool = True) -> int:
    """Get the weekday from the total number of days.

    Weekdays are ranging from 0 (Monday) to 6 (Sunday)."""
    c_weekday = (days + 4) % 7 if days >= -4 else (days + 5) % 7 + 6
    if c_weekday == 0:
        c_weekday = 7
    if zero_start:
        c_weekday = c_weekday - 1
    return c_weekday


@register_jitted(cache=True)
def weekday_nb(ts: int, zero_start: bool = True) -> int:
    """Get the weekday.

    Weekdays are ranging from 0 (Monday) to 6 (Sunday)."""
    return weekday_from_days_nb(days_nb(ts), zero_start=zero_start)


@register_jitted(cache=True)
def weekday_diff_nb(weekday1: int, weekday2: int, zero_start: bool = True) -> int:
    """Get the difference in days between two weekdays."""
    if zero_start:
        if weekday1 > 6 or weekday1 < 0:
            raise ValueError("Weekday must be in [0, 6]")
        if weekday2 > 6 or weekday2 < 0:
            raise ValueError("Weekday must be in [0, 6]")
    else:
        if weekday1 > 7 or weekday1 < 1:
            raise ValueError("Weekday must be in [1, 7]")
        if weekday2 > 7 or weekday2 < 1:
            raise ValueError("Weekday must be in [1, 7]")
    weekday_diff = weekday1 - weekday2
    if weekday_diff <= 0:
        weekday_diff += 7
    return weekday_diff


@register_jitted(cache=True)
def past_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Get the timestamp of a weekday in the past."""
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(this_weekday, weekday, zero_start=zero_start)
    return midnight_nb(ts) - weekday_diff * d_ns


@register_jitted(cache=True)
def future_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Get the timestamp of a weekday in the future."""
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(weekday, this_weekday, zero_start=zero_start)
    return midnight_nb(ts) + weekday_diff * d_ns


@register_jitted(cache=True)
def day_of_year_nb(ts: int) -> int:
    """Get the day of the year."""
    y, m, d = to_civil_nb(ts)
    y_ts = from_civil_nb(y, 1, 1)
    return (ts - y_ts) // d_ns + 1


@register_jitted(cache=True)
def week_nb(ts: int) -> int:
    """Get the week of the year."""
    return day_of_year_nb(ts) // 7


@register_jitted(cache=True)
def month_nb(ts: int) -> int:
    """Get the month of the year."""
    y, m, d = to_civil_nb(ts)
    return m


@register_jitted(cache=True)
def year_nb(ts: int) -> int:
    """Get the year."""
    y, m, d = to_civil_nb(ts)
    return y


@register_jitted(cache=True)
def is_leap_year_nb(y: int) -> int:
    """Get whether the year is a leap year."""
    return (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)


@register_jitted(cache=True)
def last_day_of_month_nb(y: int, m: int) -> int:
    """Get the last day of the month."""
    if m == 1:
        return 31
    if m == 2:
        if is_leap_year_nb(y):
            return 29
        return 28
    if m == 3:
        return 31
    if m == 4:
        return 30
    if m == 5:
        return 31
    if m == 6:
        return 30
    if m == 7:
        return 31
    if m == 8:
        return 31
    if m == 9:
        return 30
    if m == 10:
        return 31
    if m == 11:
        return 30
    return 31
