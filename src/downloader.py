"""
20230504
This script includes functions to download stock market data and crypto from multiple sources
"""

import os
import pandas as pd
from pathlib import Path
from stocksymbol import StockSymbol
from datetime import datetime, timedelta
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter, Retry
from .utils import read_tokens

SMA = [20, 30, 45, 50, 60, 150, 200]
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date_str}&endDate={end_date_str}&format=json"


class BaseDownloader:
    def __init__(self, api_keys: dict = None, save_dir: str = "."):
        if not api_keys:
            self.api_keys = read_tokens()
        else:
            self.api_keys = api_keys
        self.save_dir = Path(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


class StockDownloader(BaseDownloader):
    def get_all_symbols(self):
        """
        download all stock symbols from StockSymbol API to csv file and update the file every month
        return:
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
            api_key = self.api_keys["stocksymbol"]
            ss = StockSymbol(api_key)
            # First we download all ticker in US market
            symbol_only_list = ss.get_symbol_list(market="US", symbols_only=True)
            # The first 2 rows are for discarding any tickers that is not stock
            symbol_only_list = [x for x in symbol_only_list if "." not in x]
            symbol_only_list = [i for i in symbol_only_list if len(i) <= 4]
            symbol_only_list.sort()
            # Convert the list to pandas dataframe
            df = pd.DataFrame({'Symbol': symbol_only_list}, columns=['Symbol'])
            # saving the dataframe as csv file
            df.to_csv(saved_symbols_path, sep=' ', index=False)
            return symbol_only_list
        else:
            data = pd.read_csv(saved_symbols_path, header=0)
            return data

    def get_ticker(self, symbol, start_date, end_date):
        data = self.request_tiingo(symbol, start_date, end_date)
        if not data:
            data = self.request_yfinance(symbol, start_date, end_date)
        return data

    def request_yfinance(self, symbol, start_date, end_date):
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_date_str = end_date.strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start_date_str, end=end_date_str)
        df = df.reset_index()
        df.rename(columns={"Date": "Datetime"}, inplace=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        for duration in SMA:
            df["SMA_" + str(duration)] = round(df.loc[:, "Adj Close"].rolling(window=duration).mean(), 2)
        return df

    def request_tiingo(self, symbol, start_date, end_date):
        session = requests.Session()
        retry = Retry(
            total=1,
            read=1,
            connect=1,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_keys["tiingo"]
        }
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        url = TIINGO_URL.format(symbol=symbol, start_date_str=start_date_str, end_date_str=end_date_str)
        response = session.get(url, headers=headers, timeout=5)

        df = None
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df.rename(columns={"date": "Datetime", "close": "Close", "high": "High", "low": "Low", "open": "Open",
                               'adjClose': 'Adj Close', 'volume': 'Abs Volume', 'adjVolume': 'Volume', }, inplace=True)
            df['Datetime'] = pd.to_datetime(df.Datetime).dt.strftime('%Y-%m-%d %H:%M:%S')   # change to datetime object

            for duration in SMA:
                df["SMA_" + str(duration)] = round(df.loc[:, "Adj Close"].rolling(window=duration).mean(), 2)
        return df


class CryptoDownloader(BaseDownloader):
    pass
