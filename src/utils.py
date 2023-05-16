import json
import csv
from pytz import timezone
from datetime import datetime, timedelta
import pandas_market_calendars as mcal


def read_tokens(filename="api_keys.json"):
    with open(filename) as f:
        return json.load(f)


def read_tradingview_csv(filename="tradingview.csv", column_name="Ticker"):
    with open(filename) as f:
        csv_reader = csv.DictReader(f)
        column_values = [row[column_name] for row in csv_reader]
        column_values.sort()
        return column_values


def get_closest_market_datetime(dt=datetime.now(), backward=True):
    dt = check_timezone_to_ny(dt)
    nyse = mcal.get_calendar('NYSE')
    if backward:
        start_date = dt - timedelta(days=7)
        end_date = dt
    else:
        start_date = dt
        end_date = dt + timedelta(days=7)
    market_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    market_close_times = market_schedule['market_close'].tolist()
    closest_market_close = None
    if backward:
        for idx in range(1, len(market_close_times)+1):
            if market_close_times[idx * -1] <= dt:
                closest_market_close = market_close_times[idx * -1]
                break
    else:
        for market_close in market_close_times:
            if market_close >= dt:
                closest_market_close = market_close
                break

    return closest_market_close


def check_timezone_to_ny(dt):
    if dt.tzinfo is None:
        dt = timezone('America/New_York').localize(dt)
    elif dt.tzinfo != timezone('America/New_York'):
        dt = dt.astimezone(timezone('America/New_York'))
    return dt


