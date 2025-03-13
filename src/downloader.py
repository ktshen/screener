import pandas as pd
import json
import re
from pytz import timezone
from stocksymbol import StockSymbol
from polygon import RESTClient
from urllib3.util.retry import Retry
from binance import Client
from pathlib import Path
import os
import time
from datetime import datetime

# Configurable parameters
STOCK_SMA = [20, 30, 45, 50, 60, 150, 200]
CRYPTO_SMA = [30, 45, 60]
ATR_PERIOD = 60  


def calculate_atr(df, period=ATR_PERIOD):
    """
    Calculate Average True Range (ATR) for the given dataframe
    
    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        period: Period for ATR calculation (default: ATR_PERIOD)
        
    Returns:
        Series containing ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # Get the maximum of the three price ranges
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR as the simple moving average of True Range
    atr = tr.rolling(window=period).mean()
    
    return atr


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
    def __init__(self, api_file: str = "api_keys.json"):
        with open(api_file) as f:
            self.api_keys = json.load(f)

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
            num_pools=100,
            connect_timeout=1.0,
            read_timeout=1.0,
            retries=10
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
        latest_ts = df['timestamp'].max()
        week_ago = time.time() - (7 * 24 * 3600)
        if latest_ts < week_ago:
            return False

        # Check for stale prices
        consecutive_same_price = df['close'].rolling(window=10).apply(
            lambda x: len(set(x)) == 1
        )
        if consecutive_same_price.any():
            return False

        return True

    def get_data(self, ticker: str, start_ts: int, end_ts: int = None, timeframe: str = "1d", dropna=True, atr=True) -> tuple[bool, pd.DataFrame]:
        """
        Get stock data with SMA calculation and data quality validation
        Args:
            ticker: Stock symbol
            start_ts: Start timestamp
            end_ts: End timestamp (default: current time)
            timeframe: Time interval ("1d" or "1h")
            dropna: Whether to drop NA values
            atr: Whether to calculate ATR (default: True)
        Returns:
            (success, DataFrame)
        """
        # Calculate extended start for SMA calculation
        max_sma = max(STOCK_SMA)
        fc = 1.3 if timeframe == "1d" else 0.6
        extension = int(max_sma * 24 * 3600 * fc) 
        extended_start = start_ts - extension

        # Get current time if end_ts not provided
        if end_ts is None:
            end_ts = int(time.time())

        # Parse timeframe
        multiplier, timespan = parse_time_string(timeframe)

        # Request data from Polygon
        aggs = self.client.list_aggs(
            ticker,
            multiplier,
            timespan,
            int(extended_start * 1000),
            int(end_ts * 1000),
            limit=10000
        )

        if not aggs:
            return False, pd.DataFrame()

        # Convert to DataFrame with timestamp
        df = pd.DataFrame([{
            'timestamp': agg.timestamp // 1000,
            'open': float(agg.open),
            'close': float(agg.close),
            'high': float(agg.high),
            'low': float(agg.low),
            'volume': float(agg.volume)
        } for agg in aggs])

        if df.empty:
            return False, df

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Filter market hours (9:00 AM - 4:00 PM NY time)
        if timespan == "hour" or timespan == "minute":
            # Create temporary datetime column in NY timezone for filtering
            ny_tz = timezone('America/New_York')
            temp_dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(ny_tz)
            
            # Create filter based on NY market hours
            if timespan == "hour":
                market_hours_filter = temp_dt.dt.time.between(
                    pd.to_datetime('09:00').time(),
                    pd.to_datetime('16:00').time(),
                    inclusive='left'
                )
            else:  # minute timeframe
                market_hours_filter = temp_dt.dt.time.between(
                    pd.to_datetime('09:30').time(),
                    pd.to_datetime('16:00').time(),
                    inclusive='left'
                )
            
            # Apply filter and drop temporary column
            df = df[market_hours_filter]

        # Validate data quality
        if not self._validate_data_quality(df):
            return False, pd.DataFrame()

        # Calculate SMAs
        for period in STOCK_SMA:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # Calculate ATR if requested
        if atr:
            df['atr'] = calculate_atr(df, period=ATR_PERIOD)

        # Drop rows with NaN values
        if dropna:
            df = df.dropna()

        # Filter to requested time range and reset index
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
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
            # type="CS",
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
        return sorted(list(binance_symbols))

    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate crypto data quality
        - Check if latest data is within a week
        - Check for stale prices (same closing price for 10+ consecutive periods)
        """
        if df.empty:
            return False

        # Check data freshness
        latest_ts = df['timestamp'].max()
        week_ago = time.time() - (7 * 24 * 3600)
        if latest_ts < week_ago:
            return False

        # Check for stale prices
        consecutive_same_price = df['close'].rolling(window=10).apply(
            lambda x: len(set(x)) == 1
        )
        if consecutive_same_price.any():
            return False

        return True

    def get_data(self, crypto, start_ts=None, end_ts=None, timeframe="4h", dropna=True, atr=True) -> tuple[bool, pd.DataFrame]:
        """
        Get cryptocurrency data with SMA calculation and data quality validation
        Args:
            crypto: Cryptocurrency symbol
            start_ts: Start timestamp (default: None, fetches latest 1500 datapoints)
            end_ts: End timestamp (default: current time)
            timeframe: Time interval (e.g., "5m", "15m", "1h", "4h")
            dropna: Whether to drop NA values (default: True)
            atr: Whether to calculate ATR (default: True)
        Returns:
            (success, DataFrame)
        """
        try:
            # Default end_ts to current time if not provided
            if end_ts is None:
                end_ts = int(time.time())
            
            # Convert to milliseconds for Binance API
            end_ts_ms = end_ts * 1000
            
            if start_ts is None:
                # Fetch only the latest 1500 datapoints
                response = self.binance_client.futures_klines(
                    symbol=crypto,
                    interval=timeframe,
                    limit=1500
                )
            else:
                # Calculate extended start for SMA calculation
                max_sma = max(CRYPTO_SMA) 
                
                # Calculate number of time intervals in max_sma
                num_intervals, unit = parse_time_string(timeframe)
                if unit == "minute":
                    interval_seconds = num_intervals * 60
                elif unit == "hour":
                    interval_seconds = num_intervals * 3600
                else:  # day
                    interval_seconds = num_intervals * 86400
                
                # Calculate extension in milliseconds (number of bars needed for max SMA)
                extension_ms = max_sma * interval_seconds * 1000 * 1.2 # 20% buffer
                
                # Extended start timestamp with buffer for SMA calculation
                extended_start_ts_ms = start_ts * 1000 - extension_ms
                
                # Fetch historical data from the extended start date
                all_data = []
                current_timestamp = extended_start_ts_ms

                while current_timestamp < end_ts_ms:
                    response = self.binance_client.futures_klines(
                        symbol=crypto,
                        interval=timeframe,
                        startTime=int(current_timestamp),
                        endTime=int(end_ts_ms),
                        limit=1500
                    )

                    if not response:
                        break

                    all_data.extend(response)
                    
                    # Update current timestamp to the last received data point + 1
                    if response:
                        current_timestamp = int(response[-1][6]) + 1
                    else:
                        break

                if not all_data:
                    print(f"{crypto} -> No data retrieved")
                    return False, pd.DataFrame()
                
                response = all_data

            # Convert to DataFrame with timestamp as primary field
            df = pd.DataFrame(response, 
                            columns=["Datetime", "Open Price", "High Price", "Low Price", "Close Price",
                                    "Volume", "Close Time", "Quote Volume", "Number of Trades",
                                    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
            
            # Check if DataFrame is empty
            if df.empty:
                print(f"{crypto} -> Empty DataFrame after initial conversion")
                return False, pd.DataFrame()
            
            # Convert datetime to timestamp (in seconds)
            df['timestamp'] = df['Datetime'].astype(int) // 1000
            
            # Rename columns to match stock df format (lowercase)
            df['open'] = df['Open Price'].astype(float)
            df['high'] = df['High Price'].astype(float)
            df['low'] = df['Low Price'].astype(float)
            df['close'] = df['Close Price'].astype(float)
            df['volume'] = df['Volume'].astype(float)
            
            # Drop duplicate timestamps
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Validate data quality
            if not self._validate_data_quality(df):
                print(f"{crypto} -> Failed data quality validation")
                return False, pd.DataFrame()
            
            # Calculate SMAs
            for duration in CRYPTO_SMA:
                df[f"sma_{duration}"] = df['close'].rolling(window=duration).mean()

            # Calculate ATR if requested
            if atr:
                df['atr'] = calculate_atr(df, period=ATR_PERIOD)

            # Drop NaN values if requested
            if dropna:
                df = df.dropna()
            
            # Filter to requested time range (only after calculating SMAs)
            if start_ts is not None:
                df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
            
            # Final check if we have any data left
            if df.empty:
                print(f"{crypto} -> No data left after filtering")
                return False, pd.DataFrame()
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Keep only necessary columns
            columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Add SMA columns
            columns_to_keep += [f'sma_{period}' for period in CRYPTO_SMA]
            
            # Add ATR column if calculated
            if atr:
                columns_to_keep.append('atr')
                
            df = df[columns_to_keep]
            
            print(f"{crypto} -> Get data from binance successfully ({len(df)} rows from {datetime.fromtimestamp(df['timestamp'].iloc[0])} to {datetime.fromtimestamp(df['timestamp'].iloc[-1])})")
            return True, df

        except Exception as e:
            print(f"{crypto} -> Error: {e}")
            return False, pd.DataFrame()
        