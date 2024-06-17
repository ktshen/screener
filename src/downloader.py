import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytz import timezone
import pandas as pd
import pytz
import json
import polygon
import re
from stocksymbol import StockSymbol
from binance import Client

STRFTIME_FORMAT = "%Y-%m-%d %H:%M:%S"
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

    def check_symbol_legibility(self, symbol):
        """
        Check if symbol is legit, need to be common stock and the latest data should be recent
        """
        print(f"Checking {symbol}...")
        ref_client = polygon.ReferenceClient(self.api_keys["polygon"])
        response = ref_client.get_tickers(symbol)
        try:
            if response["count"] != 1:
                print("Error", response)
                return None
            data = response["results"][0]
            if "type" in data and not data["type"] in ["CS", "ADRC", "PFD"]:
                return None
        except Exception as e:
            return None

        success, response = self.request_ticker_by_traceback_days(symbol, 10, timeframe="1d")
        if not success:
            return None
        latest_datetime = response['Datetime'].max()
        current_datetime = datetime.now(pytz.utc)
        one_week_ago = current_datetime - timedelta(days=5)
        if latest_datetime < one_week_ago:
            return None
        return symbol

    def get_all_tickers(self):
        """
        - Download all stock symbols from StockSymbol and Polygon.io API to csv file and update the file every month
        - Avoid using this method in thread
        """
        saved_symbols_path = self.save_dir / "company-list.csv"
        symbols_expired = False
        if saved_symbols_path.exists():
            # Obtain the file modification timestamp of a file
            m_time = os.path.getmtime(saved_symbols_path)
            # Obtain now timestamp
            now = datetime.now()
            # Find the timedelta between now and the file modification timestamp
            delta = now - datetime.fromtimestamp(m_time)
            if delta > timedelta(days=30):
                symbols_expired = True

        if not saved_symbols_path.exists() or symbols_expired:
            print("Fetching all symbols...")
            ss = StockSymbol(self.api_keys["stocksymbol"])
            # First we download all ticker in US market
            stock_symbol_list = ss.get_symbol_list(market="US", symbols_only=True)
            # The first 2 rows are for discarding any tickers that is not stock
            stock_symbol_list = [x for x in stock_symbol_list if "." not in x]

            ref_client = polygon.ReferenceClient(self.api_keys["polygon"])
            response = ref_client.get_tickers(market='stocks', symbol_type="CS", limit=1000, all_pages=True)
            polygon_common_stocks = [x["ticker"] for x in response]

            stock_symbol_set = set(stock_symbol_list)
            polygon_common_stocks_set = set(polygon_common_stocks)

            merged_set = stock_symbol_set.union(polygon_common_stocks_set)
            merged_list = list(merged_set)
            merged_list.sort()

            clean_stock_symbol_list = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = {executor.submit(self.check_symbol_legibility, symbol): symbol for symbol in merged_list}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result:
                            clean_stock_symbol_list.append(result)
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
            clean_stock_symbol_list.sort()
            df = pd.DataFrame({'Symbol': clean_stock_symbol_list}, columns=['Symbol'])
            # saving the dataframe as csv file
            print(f"Generating {saved_symbols_path}...")
            print(f"There are total {len(merged_list)}, and only {len(clean_stock_symbol_list)} tickers.")
            df.to_csv(saved_symbols_path, sep=' ', index=False)
            return clean_stock_symbol_list
        else:
            data = pd.read_csv(saved_symbols_path, header=0)
            return list(data.iloc[:, 0])

    @staticmethod
    def parse_polygon_response(response):
        success = False
        if response:
            success = True
            df = pd.DataFrame(response)
            df = df.drop(columns=['vw', 'n'])
            df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            new_york_tz = pytz.timezone('America/New_York')
            df = df[["t", "o", "c", "h", "l", "v"]]
            df = df.rename(columns={
                't':  "Datetime",
                'o': 'Open',
                'c': 'Close',
                'h': 'High',
                'l': 'Low',
                'v': 'Volume',
            })
            df['Datetime'] = df['Datetime'].dt.tz_convert(new_york_tz)
            return success, df
        else:
            return success, response

    def request_ticker_all_range(self, ticker, timeframe="4h", current_tz="America/Los_Angeles"):
        stock_client = polygon.StocksClient(self.api_keys["polygon"])
        start_date = datetime.now() - timedelta(days=365*2)
        end_date = timezone(current_tz).localize(datetime.now())
        multiplier, timespan = parse_time_string(timeframe)
        response = stock_client.get_aggregate_bars(ticker, start_date, end_date, multiplier=multiplier,
                                                   timespan=timespan, full_range=True, run_parallel=False,
                                                   warnings=False, info=False)
        return self.parse_polygon_response(response)

    def request_ticker_by_date(self, ticker, start_datetime, end_datetime, timeframe="4h"):
        stock_client = polygon.StocksClient(self.api_keys["polygon"])
        multiplier, timespan = parse_time_string(timeframe)
        response = stock_client.get_aggregate_bars(ticker, start_datetime, end_datetime, multiplier=multiplier,
                                                   timespan=timespan, full_range=True, run_parallel=False,
                                                   warnings=False, info=False)
        return self.parse_polygon_response(response)


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

    def get_crypto(self, crypto, time_interval="15m", timezone="America/Los_Angeles"):
        success = False  # 0 - fail, 1 - success
        try:
            binance_df = self.request_binance(crypto, time_interval, timezone)
            binance_df.set_index('Datetime', inplace=True)
            binance_df = binance_df[~binance_df.index.duplicated(keep='first')]
            response = binance_df
            response.reset_index(inplace=True)
            for duration in CRYPTO_SMA:
                response["SMA_" + str(duration)] = round(response.loc[:, "Close Price"].rolling(window=duration).mean(), 20)
            success = True
            print(f"{crypto} -> Get data from binance successfully ({response.iloc[0]['Datetime']} to {response.iloc[-1]['Datetime']})")
        except Exception as e:
            print(f"{crypto} -> Error: {e}")
            response = str(e)
        return crypto, success, response
