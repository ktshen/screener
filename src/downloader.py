from concurrent.futures import ThreadPoolExecutor, as_completed
from pytz import timezone
import pandas as pd
import pytz
import json
import re
from stocksymbol import StockSymbol
from polygon import RESTClient
from urllib3.util.retry import Retry
from binance import Client
from pathlib import Path
import os
import time
from datetime import datetime, timedelta

# Configurable SMA periods
STOCK_SMA = [20, 30, 45, 50, 60, 150, 200]
CRYPTO_SMA = [30, 45, 60]


def parse_time_string(time_string):
    pattern_with_number = r"(\d+)([mhdMHD])$"
    pattern_without_number = r"([dD])$"
    match_with_number = re.match(pattern_with_number, time_string)
    match_without_number = re.match(pattern_without_number, time_string)

    if match_with_number:
        number = int(match_with_number.group(1))
        unit = match_with_number.group(2)
    elif match_without_number:
        number = 1
        unit = match_without_number.group(1)
    else:
        raise ValueError("Invalid time format. Only formats like '15m', '4h', 'd' are allowed.")

    unit = unit.lower()
    unit_match = {
        "m": "minute",
        "h": "hour",
        "d": "day"
    }
    return number, unit_match[unit]


class StockDownloader:
    def __init__(self, save_dir: str = ".", api_file: str = "api_keys.json"):
        with open(api_file) as f:
            self.api_keys = json.load(f)
        self.save_dir = Path(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[413, 429, 499, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
            raise_on_status=False,
            respect_retry_after_header=True
        )

        self.client = RESTClient(
            api_key=self.api_keys["polygon"],
            num_pools=5,
            connect_timeout=10.0,
            read_timeout=10.0,
            retries=retry_strategy
        )

    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        - Check if latest data is within a week
        - Check for stale prices (same closing price for 10+ consecutive periods)
        """
        if df.empty:
            return False

        # Check data freshness
        latest_ts = df['Datetime'].astype(int).max() / 1e9
        week_ago = time.time() - (7 * 24 * 3600)
        if latest_ts < week_ago:
            return False

        # Check for stale prices
        consecutive_same_price = df['Close'].rolling(window=10).apply(
            lambda x: len(set(x)) == 1
        )
        if consecutive_same_price.any():
            return False

        return True

    def get_data(self, ticker: str, start_ts: int, timeframe: str = "1d") -> tuple[bool, pd.DataFrame]:
        """
        Get stock data with SMA calculation and data quality validation
        Args:
            ticker: Stock symbol
            start_ts: Start timestamp
            timeframe: Time interval ("1d" or "1h")
        Returns:
            (success, DataFrame)
        """
        # Calculate extended start for SMA calculation
        max_sma = max(STOCK_SMA)
        extension = max_sma * 24 * 3600 if timeframe == "1d" else max_sma * 7 * 3600
        extended_start = start_ts - extension

        # Get current time
        end_ts = int(time.time())

        # Parse timeframe
        multiplier, timespan = parse_time_string(timeframe)

        # Request data from Polygon
        aggs = self.client.list_aggs(
            ticker,
            multiplier,
            timespan,
            from_=extended_start * 1000,
            to=end_ts * 1000,
            limit=50000
        )

        if not aggs:
            return False, pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([{
            'Datetime': pd.Timestamp.fromtimestamp(agg.timestamp // 1000, tz=pytz.UTC),
            'Open': float(agg.open),
            'Close': float(agg.close),
            'High': float(agg.high),
            'Low': float(agg.low),
            'Volume': float(agg.volume)
        } for agg in aggs])

        if df.empty:
            return False, df

        # Convert timezone and sort
        df['Datetime'] = df['Datetime'].dt.tz_convert('America/New_York')
        df = df.sort_values('Datetime')

        # Filter market hours (9:00 AM - 4:00 PM NY time)
        if timespan == "hour":
            df = df[
                df['Datetime'].dt.time.between(
                    pd.to_datetime('09:00').time(),
                    pd.to_datetime('16:00').time(),
                    inclusive='left'
                )
            ]
        elif timespan == "minute":
            df = df[
                df['Datetime'].dt.time.between(
                    pd.to_datetime('09:30').time(),
                    pd.to_datetime('16:00').time(),
                    inclusive='left'
                )
            ]

        # Validate data quality
        if not self._validate_data_quality(df):
            return False, pd.DataFrame()

        # Calculate SMAs
        for period in STOCK_SMA:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

        # Drop rows with NaN values
        df = df.dropna()

        # Filter to requested time range and reset index
        df = df[df['Datetime'].astype(int) / 1e9 >= start_ts]
        df = df.reset_index(drop=True)

        return True, df

    def get_all_tickers(self):
        """Get all stock symbols from both StockSymbol and Polygon"""
        # Get symbols from StockSymbol
        ss = StockSymbol(self.api_keys["stocksymbol"])
        stock_symbol_list = [x for x in ss.get_symbol_list(market="US", symbols_only=True)
                           if "." not in x]

        # Get symbols from Polygon
        polygon_stocks = self.client.list_tickers(
            market="stocks",
            type="CS",
            active=True,
            limit=1000
        )
        polygon_common_stocks = [ticker.ticker for ticker in polygon_stocks]

        # Merge and return unique symbols
        all_symbols = sorted(set(stock_symbol_list).union(set(polygon_common_stocks)))
        print(f"Found {len(all_symbols)} unique stock symbols")
        return all_symbols


class CryptoDownloader:
    def __init__(self):
        self.binance_client = Client(requests_params={"timeout": 300})

    def get_all_symbols(self):
        """
        Get all USDT pairs in binance
        """
        binance_response = self.binance_client.futures_exchange_info()
        binance_symbols = set()
        for item in binance_response["symbols"]:
            symbol_name = item["pair"]
            if symbol_name[-4:] == "USDT":
                binance_symbols.add(symbol_name)
        return list(binance_symbols)

    def request_binance(self, crypto, time_interval="15m", current_tz="America/Los_Angeles"):
        """
        1500 data points for all timeframe using binance futures instead of binance spots
        """
        response = self.binance_client.futures_klines(symbol=crypto, interval=time_interval, limit=1500)
        data = pd.DataFrame(response, columns=["Datetime", "Open Price", "High Price", "Low Price", "Close Price",
                                               "Volume", "Close Time", "Quote Volume", "Number of Trades",
                                               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
        data["Open Price"] = data["Open Price"].astype(float)
        data["High Price"] = data["High Price"].astype(float)
        data["Low Price"] = data["Low Price"].astype(float)
        data["Close Price"] = data["Close Price"].astype(float)
        data["Volume"] = data["Volume"].astype(float)
        local_timezone = pytz.timezone(current_tz)
        data["Datetime"] = pd.to_datetime(data['Datetime'], unit='ms', utc=True).dt.tz_convert(local_timezone).dt.strftime('%Y-%m-%d %H:%M:%S')
        data.drop(["Close Time", "Quote Volume", "Number of Trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                   "Ignore"], axis=1, inplace=True)
        return data

    def get_crypto(self, crypto, start_date=None, time_interval="4h", timezone="America/Los_Angeles"):
        success = False
        try:
            if start_date is None:
                # Fetch only the latest 1500 datapoints
                response = self.binance_client.futures_klines(
                    symbol=crypto,
                    interval=time_interval,
                    limit=1500
                )
                df = pd.DataFrame(response,
                                  columns=["Datetime", "Open Price", "High Price", "Low Price", "Close Price", "Volume",
                                           "Close Time", "Quote Volume", "Number of Trades",
                                           "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
            else:
                # Fetch historical data from the start_date
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                end_timestamp = int(datetime.now().timestamp() * 1000)

                all_data = []
                current_timestamp = start_timestamp

                while current_timestamp < end_timestamp:
                    response = self.binance_client.futures_klines(
                        symbol=crypto,
                        interval=time_interval,
                        startTime=current_timestamp,
                        limit=1500
                    )

                    if not response:
                        break

                    df = pd.DataFrame(response,
                                      columns=["Datetime", "Open Price", "High Price", "Low Price", "Close Price",
                                               "Volume", "Close Time", "Quote Volume", "Number of Trades",
                                               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
                    all_data.append(df)

                    current_timestamp = int(df.iloc[-1]['Close Time']) + 1

                if not all_data:
                    raise Exception("No data retrieved")

                df = pd.concat(all_data, ignore_index=True)

            df = df.drop_duplicates(subset=['Datetime'], keep='first')

            for col in ["Open Price", "High Price", "Low Price", "Close Price", "Volume"]:
                df[col] = df[col].astype(float)

            local_timezone = pytz.timezone(timezone)
            df["Datetime"] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert(
                local_timezone).dt.strftime('%Y-%m-%d %H:%M:%S')

            df.drop(["Close Time", "Quote Volume", "Number of Trades", "Taker buy base asset volume",
                     "Taker buy quote asset volume", "Ignore"], axis=1, inplace=True)

            df.reset_index(drop=True, inplace=True)

            for duration in CRYPTO_SMA:
                df[f"SMA_{duration}"] = round(df.loc[:, "Close Price"].rolling(window=duration).mean(), 20)

            success = True
            print(f"{crypto} -> Get data from binance successfully ({df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]})")

            return crypto, success, df

        except Exception as e:
            print(f"{crypto} -> Error: {e}")
            return crypto, success, str(e)
