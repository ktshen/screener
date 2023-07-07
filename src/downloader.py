"""
20230504
This script includes functions to download stock market data and crypto from multiple sources
"""
import os
import time
import pandas as pd
import yfinance as yf
import requests
import pytz
from pathlib import Path
from stocksymbol import StockSymbol
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance import Client

from src.utils import read_tokens, read_tradingview_csv, get_closest_market_datetime, check_timezone_to_ny, connect_db


###########################################################################
STOCK_SMA = [20, 30, 45, 50, 60, 150, 200]
CRYPTO_SMA = [30, 45, 60]
###########################################################################
REQUEST_STOCK_TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date_str}&endDate={end_date_str}&format=json"
REQUEST_CRYPTO_TIINGO_URL = "https://api.tiingo.com/tiingo/crypto/prices?tickers={crypto}&startDate={start_date_str}&endDate={end_date_str}&resampleFreq={interval}"
DB_STRFTIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CURRENT_TIMEZONE = "America/Los_Angeles"
INCLUDE_STOCK_SYMBOLS = ["SPY", "QQQ", "DIA"]
###########################################################################


class BaseDownloader:
    def __init__(self, api_keys: dict = None, save_dir: str = ".", db_name="screen.db"):
        if not api_keys:
            self.api_keys = read_tokens()
        else:
            self.api_keys = api_keys
        self.save_dir = Path(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.db_path = self.save_dir / db_name


class StockDownloader(BaseDownloader):
    def __init__(self, api_keys: dict = None, save_dir: str = ".", db_name="screen.db"):
        super().__init__(api_keys, save_dir, db_name)

    def check_stock_table(self):
        conn, cursor = connect_db(self.db_path)
        try:
            # Check if the information in database is available
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock'")
            existing_table = cursor.fetchone()
            # Table doesn't exist, so create it
            if not existing_table:
                cursor.execute('''CREATE TABLE stock (
                                   Stock TEXT,
                                   Datetime DATETIME,
                                   "Close Price" FLOAT,
                                   "High Price" FLOAT,
                                   "Low Price" FLOAT,
                                   "Open Price" FLOAT,
                                   "Adj Close" FLOAT,
                                   Volume INTEGER,
                                   SMA_20 FLOAT,
                                   SMA_30 FLOAT,
                                   SMA_45 FLOAT,
                                   SMA_50 FLOAT, 
                                   SMA_60 FLOAT,
                                   SMA_150 FLOAT,
                                   SMA_200 FLOAT,
                                   PRIMARY KEY (Stock, Datetime)
                                   )''')
                print("Table \"stock\" doesn't exist. A new table has been created successfully.")

        except Exception as e:
            raise Exception(f"Failed to create the table \"stock\": {e}")

        finally:
            cursor.close()
            conn.close()

    def get_all_symbols(self, tradingview_csv="tradingview.csv", csv_column_name="Ticker"):
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
            tradingview_symbol_only_list = None
            if os.path.exists(self.save_dir / tradingview_csv):
                tradingview_symbol_only_list = read_tradingview_csv(self.save_dir / tradingview_csv, csv_column_name)
            api_key = self.api_keys["stocksymbol"]
            ss = StockSymbol(api_key)
            # First we download all ticker in US market
            symbol_only_list = ss.get_symbol_list(market="US", symbols_only=True)
            # The first 2 rows are for discarding any tickers that is not stock
            symbol_only_list = [x for x in symbol_only_list if "." not in x]
            symbol_only_list = [i for i in symbol_only_list if len(i) <= 4]
            if tradingview_symbol_only_list:
                symbol_only_list = list(set(symbol_only_list) | set(tradingview_symbol_only_list))
            symbol_only_list.sort()
            # Convert the list to pandas dataframe
            df = pd.DataFrame({'Symbol': symbol_only_list}, columns=['Symbol'])
            # saving the dataframe as csv file
            df.to_csv(saved_symbols_path, sep=' ', index=False)
            return symbol_only_list
        else:
            data = pd.read_csv(saved_symbols_path, header=0)
            return list(data.iloc[:, 0])

    def update_database(self, start_date=(datetime.now() - timedelta(days=400)), end_date=datetime.now(),
                        exclude_symbols=[], include_symbols=INCLUDE_STOCK_SYMBOLS):

        def get_ticker_loop(ticker, start_d, end_d):
            return self.get_ticker(ticker, start_d, end_d)

        start_date = check_timezone_to_ny(start_date)
        end_date = check_timezone_to_ny(end_date)
        nearest_market_close_start_date = get_closest_market_datetime(start_date)
        nearest_market_close_end_date = get_closest_market_datetime(end_date)
        # Get all the symbols and submit jobs to executor pool
        all_symbols = self.get_all_symbols()
        filtered_symbols = sorted(list((set(all_symbols) - set(exclude_symbols)) | set(include_symbols)))
        executor = ThreadPoolExecutor(max_workers=os.cpu_count()*2)
        futures = []
        for symbol in filtered_symbols:
            print(f"Updating {symbol}...")
            future = executor.submit(get_ticker_loop, symbol, nearest_market_close_start_date,
                                     nearest_market_close_end_date)
            futures.append(future)

        success = []
        fail = []
        for future in as_completed(futures):
            symbol, status, resp = future.result()
            if status == 0:
                fail.append((symbol, resp))
            else:
                success.append((symbol, resp))
        executor.shutdown()

        print(f"{len(success)} out of {len(filtered_symbols)} symbols have updated successfully -> ",
              ", ".join([x[0] for x in success]))

        if fail:
            failed_symbols_dump_file = "failed_to_download_symbols_{}.txt".format(end_date.strftime("%Y-%m-%d"))
            with open(str(self.save_dir / failed_symbols_dump_file), "w") as f:
                for item in fail:
                    f.write(f"{item[0]} failed -> {item[1]}\n")

    def get_ticker(self, symbol, start_date, end_date):
        start_date = check_timezone_to_ny(start_date)
        end_date = check_timezone_to_ny(end_date)
        status = 0  # 0 - fail, 1 - data from database, 2 - data from API
        response = None
        start_date = get_closest_market_datetime(start_date)
        end_date = get_closest_market_datetime(end_date)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        try:
            conn, cursor = connect_db(self.db_path)
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            response = str(e)
            return symbol, status, response

        try:
            # Check if start date and end date are in the table already
            cursor.execute("SELECT * FROM stock WHERE Datetime LIKE ? AND Stock = ?", (start_date_str + '%', symbol))
            start_date_result = cursor.fetchall()
            cursor.execute("SELECT * FROM stock WHERE Datetime LIKE ? AND Stock = ? ", (end_date_str + '%', symbol))
            end_date_result = cursor.fetchall()

            # If start date and end_date exist in the table, fetch data from database, otherwise fetch data from web
            if start_date_result and end_date_result:
                cursor.execute("SELECT * FROM stock WHERE date(Datetime) BETWEEN date(?) AND date(?) AND Stock = ?",
                               (start_date_str, end_date_str, symbol))
                rows = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                response = pd.DataFrame(rows, columns=column_names)
                status = 1
                print(f"{symbol} -> Get data from database")
            else:
                try:
                    response = self.request_tiingo(symbol, start_date, end_date)
                    if response is None:
                        response = self.request_yfinance(symbol, start_date, end_date)
                except Exception as e:
                    raise Exception(f"{symbol} -> Error in requesting data from web: {e}")
                if response.empty:
                    raise ValueError(f"{symbol} -> No stock data")
                # store the data to sqlite3
                response.insert(0, "Stock", symbol)
                stored_columns = ['Stock', 'Datetime', 'Close Price', 'High Price', 'Low Price', 'Open Price',
                                  'Adj Close', 'Volume', "SMA_20", "SMA_30", "SMA_45", "SMA_50", "SMA_60", "SMA_150",
                                  "SMA_200"]
                for col in response.columns:
                    if not col in stored_columns:
                        response = response.drop(columns=[col, ])
                for _, row in response.iterrows():
                    # Create a tuple with the values of the row
                    columns = ', '.join(f'"{column}"' for column in stored_columns)
                    values = ', '.join('?' for _ in row.index)
                    # Generate the SQLite query to insert or replace the row
                    query = f'INSERT OR REPLACE INTO stock ({columns}) VALUES ({values})'
                    # Execute the query with the row values
                    cursor.execute(query, tuple(row))
                # Commit the changes to the database
                conn.commit()
                status = 2
                print(f"{symbol} -> Successfully fetch data from web and store data to database")

        except Exception as e:
            print(f"{symbol} -> Error: {e}")
            response = str(e)

        finally:
            cursor.close()
            conn.close()
        return symbol, status, response

    def request_yfinance(self, symbol, start_date, end_date):
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_date_str = end_date.strftime("%Y-%m-%d")
        try:
            df = yf.download(symbol, start=start_date_str, end=end_date_str)
            time.sleep(0.2)
        except Exception as e:
            raise Exception(f"{symbol} -> Fetching data from yfinance. Error: {e}")
        df = df.reset_index()
        df.rename(columns={"Date": "Datetime", "Open": "Open Price", "Close": "Close Price", "High": "High Price",
                           "Low": "Low Price"}, inplace=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime(DB_STRFTIME_FORMAT)
        for duration in STOCK_SMA:
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
        url = REQUEST_STOCK_TIINGO_URL.format(symbol=symbol, start_date_str=start_date_str, end_date_str=end_date_str)
        response = session.get(url, headers=headers, timeout=5)
        df = None
        if response.status_code == 200:
            data = response.json()
            if not data:
                return None
            df = pd.DataFrame(data)
            df.rename(columns={"date": "Datetime", "close": "Close Price", "high": "High Price", "low": "Low Price",
                               "open": "Open Price", 'adjClose': 'Adj Close', 'volume': 'Abs Volume',
                               'adjVolume': 'Volume', }, inplace=True)
            df['Datetime'] = pd.to_datetime(df.Datetime).dt.strftime(DB_STRFTIME_FORMAT)  # change to datetime object

            for duration in STOCK_SMA:
                df["SMA_" + str(duration)] = round(df.loc[:, "Adj Close"].rolling(window=duration).mean(), 2)
        return df


class CryptoDownloader(BaseDownloader):
    def __init__(self, api_keys: dict = None, save_dir: str = ".", db_name="screen.db"):
        super().__init__(api_keys, save_dir, db_name)
        self.binance_client = Client(requests_params={"timeout": 30})

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

    def check_crypto_table(self):
        conn, cursor = connect_db(self.db_path)
        try:
            # Check if the information in database is available
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='crypto'")
            existing_table = cursor.fetchone()
            # Table doesn't exist, so create it
            if not existing_table:
                cursor.execute('''CREATE TABLE crypto (
                                   CRYPTO TEXT,
                                   Datetime DATETIME,
                                   "Close Price" FLOAT,
                                   "High Price" FLOAT,
                                   "Low Price" FLOAT,
                                   "Open Price" FLOAT,
                                   Volume FLOAT,
                                   PRIMARY KEY (CRYPTO, Datetime)
                                   )''')
                print("Table \"crypto\" doesn't exist. A new table has been created successfully.")

        except Exception as e:
            raise Exception(f"Failed to create the table \"crypto\": {e}")

        finally:
            cursor.close()
            conn.close()

    def update_database(self, start_date=(datetime.now() - timedelta(days=60))):
        def get_crypto_loop(s, start_d):
            return self.get_crypto(s, start_d)

        # Get all the symbols and submit jobs to executor pool
        all_symbols = self.get_all_symbols()
        executor = ThreadPoolExecutor(max_workers=os.cpu_count()*2)
        futures = []
        for symbol in all_symbols:
            print(f"Updating {symbol}...")
            future = executor.submit(get_crypto_loop, symbol, start_date)
            futures.append(future)

        success = []
        fail = []
        for future in as_completed(futures):
            symbol, status, resp = future.result()
            if status == 0:
                fail.append((symbol, resp))
            else:
                success.append((symbol, resp))
        executor.shutdown()

        print(f"{len(success)} out of {len(all_symbols)} symbols have updated successfully -> ",
              ", ".join([x[0] for x in success]))

        if fail:
            failed_symbols_dump_file = "failed_to_download_cryptos_{}.txt".format(datetime.now().strftime("%Y-%m-%d"))
            with open(str(self.save_dir / failed_symbols_dump_file), "w") as f:
                for item in fail:
                    f.write(f"{item[0]} failed -> {item[1]}\n")

    def get_crypto(self, crypto, time_interval="1h", start_date=datetime.now() - timedelta(days=60)):
        """
        1. Request binance data anyone, since it's free to download. For older data, use request_tiingo
        2. Store data to database
        3. Get data from database again, calculate SMA and return the data
        """
        status = 0  # 0 - fail, 1 - data from database
        response = None

        if time_interval == "15m":
            try:
                binance_df = self.request_binance(crypto, time_interval)
                binance_df.set_index('Datetime', inplace=True)
                binance_df = binance_df[~binance_df.index.duplicated(keep='first')]
                response = binance_df
                response.reset_index(inplace=True)
                for duration in CRYPTO_SMA:
                    response["SMA_" + str(duration)] = round(response.loc[:, "Close Price"].rolling(window=duration).mean(), 20)
                status = 1
                print(f"{crypto} -> Get data from binance successfully ({response.iloc[0]['Datetime']} to {response.iloc[-1]['Datetime']})")
            except Exception as e:
                print(f"{crypto} -> Error: {e}")
                response = str(e)
            return crypto, status, response

        try:
            conn, cursor = connect_db(self.db_path)
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            response = str(e)
            return crypto, status, response
        try:
            start_date_str = start_date.strftime("%Y-%m-%d %H%")
            end_date_str = datetime.now().strftime("%Y-%m-%d %H:00")
            binance_df = self.request_binance(crypto)
            binance_df.set_index('Datetime', inplace=True)
            binance_df = binance_df[~binance_df.index.duplicated(keep='first')]
            cursor.execute("SELECT * FROM crypto WHERE Datetime LIKE ? AND Crypto = ?", (start_date_str, crypto))
            start_date_result = cursor.fetchall()
            if not start_date_result:
                tiingo_df = self.request_tiingo(crypto, start_date)
                if tiingo_df is not None and not tiingo_df.empty and len(binance_df) < len(tiingo_df):
                    tiingo_df.set_index('Datetime', inplace=True)
                    tiingo_df = tiingo_df[~tiingo_df.index.duplicated(keep='first')]
                    tiingo_df.update(binance_df)
                    response = tiingo_df
            if response is None:
                response = binance_df
            response.reset_index(inplace=True)
            if response.empty:
                raise ValueError(f"{crypto} -> No stock data")
            response.insert(0, "Crypto", crypto)
            stored_columns = ['Crypto', 'Datetime', 'Close Price', 'High Price', 'Low Price', 'Open Price', 'Volume']
            for _, row in response.iterrows():
                # Create a tuple with the values of the row
                columns = ', '.join(f'"{column}"' for column in stored_columns)
                values = ', '.join('?' for _ in row.index)
                query = f'INSERT OR REPLACE INTO crypto ({columns}) VALUES ({values})'
                cursor.execute(query, tuple(row))
            conn.commit()
            cursor.execute("SELECT * FROM crypto WHERE strftime('%Y-%m-%d %H:00', Datetime) BETWEEN ? AND ? AND Crypto = ?",
                           (start_date_str, end_date_str, crypto))
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            response = pd.DataFrame(rows, columns=column_names)
            for duration in CRYPTO_SMA:
                response["SMA_" + str(duration)] = round(response.loc[:, "Close Price"].rolling(window=duration).mean(), 20)
            status = 1
            print(f"{crypto} -> Get data from database successfully ({response.iloc[0]['Datetime']} to {response.iloc[-1]['Datetime']})")

        except Exception as e:
            print(f"{crypto} -> Error: {e}")
            response = str(e)

        finally:
            cursor.close()
            conn.close()

        return crypto, status, response

    def request_binance(self, crypto, time_interval="1h", timezone="America/Los_Angeles"):
        """
        Fetch data from binance.com.
        15min - 1000 data points
        1hr - 382 data points only
        4hr - 96 data points only
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
        local_timezone = pytz.timezone(timezone)
        data["Datetime"] = pd.to_datetime(data['Datetime'], unit='ms', utc=True).dt.tz_convert(local_timezone).dt.strftime('%Y-%m-%d %H:%M:%S')
        data.drop(["Close Time", "Quote Volume", "Number of Trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                   "Ignore"], axis=1, inplace=True)
        return data

    def request_tiingo(self, crypto, start_date, timezone="America/Los_Angeles"):
        interval = "1hour"
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
        end_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")  # plus one day to get the latest
        url = REQUEST_CRYPTO_TIINGO_URL.format(crypto=crypto, start_date_str=start_date_str, end_date_str=end_date_str,
                                               interval=interval)
        response = session.get(url, headers=headers, timeout=5)
        df = None
        if response.status_code == 200:
            data = response.json()
            if not data:
                return None
            df = pd.DataFrame(data[0]["priceData"])
            df.rename(columns={"open": "Open Price", "high": "High Price", "low": "Low Price", "close": "Close Price",
                               'date': 'Datetime', 'tradesDone': 'Trades Done', 'volume': 'Volume',
                               "volumeNotional": "Volume Notional"}, inplace=True)
            df.drop(["Trades Done", "Volume Notional"], axis=1, inplace=True)
            local_timezone = pytz.timezone(timezone)
            df['Datetime'] = pd.to_datetime(df.Datetime, utc=True).dt.tz_convert(local_timezone).dt.strftime(DB_STRFTIME_FORMAT)
        return df
