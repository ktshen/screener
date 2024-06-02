# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes for communicating with TradingView."""

import datetime
import enum
import random
import re
import string
import pandas as pd
import requests
import json
import time
from websocket import WebSocket

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.pbar import get_pbar

__all__ = [
    "TVClient",
]


class Interval(enum.Enum):
    in_1_minute = "1"
    in_3_minute = "3"
    in_5_minute = "5"
    in_15_minute = "15"
    in_30_minute = "30"
    in_45_minute = "45"
    in_1_hour = "1H"
    in_2_hour = "2H"
    in_3_hour = "3H"
    in_4_hour = "4H"
    in_daily = "1D"
    in_weekly = "1W"
    in_monthly = "1M"


SIGNIN_URL = "https://www.tradingview.com/accounts/signin/"
SEARCH_URL = "https://symbol-search.tradingview.com/symbol_search/v3/?text={}&exchange={}&start={}&hl=2&lang=en&domain=production"
SCAN_URL = "https://scanner.tradingview.com/{}/scan"
ORIGIN_URL = "https://data.tradingview.com"
REFERER_URL = "https://www.tradingview.com"
WS_URL = "wss://data.tradingview.com/socket.io/websocket"
PRO_WS_URL = "wss://prodata.tradingview.com/socket.io/websocket"
WS_TIMEOUT = 5


class TVClient(Configured):
    """Client for TradingView."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "username",
        "password",
        "user_agent",
        "token",
    }

    def __init__(
        self,
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
        user_agent: tp.Optional[str] = None,
        token: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Client for TradingView."""
        Configured.__init__(
            self,
            username=username,
            password=password,
            user_agent=user_agent,
            token=token,
            **kwargs,
        )

        if token is None:
            token = self.auth(username, password, user_agent=user_agent)
        elif username is not None or password is not None:
            raise ValueError("Either username and password, or token must be provided")

        self._token = token
        self._ws = None
        self._session = self.generate_session()
        self._chart_session = self.generate_chart_session()

    @property
    def token(self) -> str:
        """Token."""
        return self._token

    @property
    def ws(self) -> WebSocket:
        """Instance of `websocket.Websocket`."""
        return self._ws

    @property
    def session(self) -> str:
        """Session."""
        return self._session

    @property
    def chart_session(self) -> str:
        """Chart session."""
        return self._chart_session

    def auth(
        self,
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
        user_agent: tp.Optional[str] = None,
    ) -> str:
        """Authenticate."""
        if username is not None and password is not None:
            data = {"username": username, "password": password, "remember": "on"}
            headers = {"Referer": REFERER_URL}
            if user_agent is not None:
                headers["User-Agent"] = user_agent
            response = requests.post(url=SIGNIN_URL, data=data, headers=headers)
            return response.json()["user"]["auth_token"]
        if username is not None or password is not None:
            raise ValueError("Both username and password must be provided")
        return "unauthorized_user_token"

    @staticmethod
    def generate_session() -> str:
        """Generate session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "qs_" + random_string

    @staticmethod
    def generate_chart_session() -> str:
        """Generate chart session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "cs_" + random_string

    def create_connection(self, pro_data: bool = True) -> None:
        """Create a websocket connection."""
        from websocket import create_connection

        if pro_data:
            self._ws = create_connection(PRO_WS_URL, headers=json.dumps({"Origin": ORIGIN_URL}), timeout=WS_TIMEOUT)
        else:
            self._ws = create_connection(WS_URL, headers=json.dumps({"Origin": ORIGIN_URL}), timeout=WS_TIMEOUT)

    @staticmethod
    def filter_raw_message(text) -> tp.Tuple[str, str]:
        """Filter raw message."""
        found = re.search('"m":"(.+?)",', text).group(1)
        found2 = re.search('"p":(.+?"}"])}', text).group(1)
        return found, found2

    @staticmethod
    def prepend_header(st: str) -> str:
        """Prepend a header."""
        return "~m~" + str(len(st)) + "~m~" + st

    @staticmethod
    def construct_message(func: str, param_list: tp.List[str]) -> str:
        """Construct a message."""
        return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

    def create_message(self, func: str, param_list: tp.List[str]) -> str:
        """Create a message."""
        return self.prepend_header(self.construct_message(func, param_list))

    def send_message(self, func: str, param_list: tp.List[str]) -> None:
        """Send a message."""
        m = self.create_message(func, param_list)
        self.ws.send(m)

    @staticmethod
    def convert_raw_data(raw_data: str, symbol: str) -> pd.DataFrame:
        """Process raw data into a DataFrame."""
        out = re.search('"s":\[(.+?)\}\]', raw_data).group(1)
        x = out.split(',{"')
        data = list()
        volume_data = True
        for xi in x:
            xi = re.split("\[|:|,|\]", xi)
            ts = datetime.datetime.utcfromtimestamp(float(xi[4]))
            row = [ts]
            for i in range(5, 10):
                # skip converting volume data if does not exists
                if not volume_data and i == 9:
                    row.append(0.0)
                    continue
                try:
                    row.append(float(xi[i]))
                except ValueError:
                    volume_data = False
                    row.append(0.0)
            data.append(row)
        data = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])
        data = data.set_index("datetime")
        data.insert(0, "symbol", value=symbol)
        return data

    @staticmethod
    def format_symbol(symbol: str, exchange: str, fut_contract: tp.Optional[int] = None) -> str:
        """Format a symbol."""
        if ":" in symbol:
            pass
        elif fut_contract is None:
            symbol = f"{exchange}:{symbol}"
        elif isinstance(fut_contract, int):
            symbol = f"{exchange}:{symbol}{fut_contract}!"
        else:
            raise ValueError(f"Invalid option fut_contract='{fut_contract}'")
        return symbol

    def get_hist(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: Interval = Interval.in_daily,
        fut_contract: tp.Optional[int] = None,
        adjustment: str = "splits",
        extended_session: bool = False,
        pro_data: bool = True,
        limit: int = 20000,
        return_raw: bool = False,
    ) -> tp.Union[str, tp.Frame]:
        """Get historical data."""
        symbol = self.format_symbol(symbol=symbol, exchange=exchange, fut_contract=fut_contract)
        interval = interval.value

        self.create_connection(pro_data=pro_data)
        self.send_message("set_auth_token", [self.token])
        self.send_message("chart_create_session", [self.chart_session, ""])
        self.send_message("quote_create_session", [self.session])
        self.send_message(
            "quote_set_fields",
            [
                self.session,
                "ch",
                "chp",
                "current_session",
                "description",
                "local_description",
                "language",
                "exchange",
                "fractional",
                "is_tradable",
                "lp",
                "lp_time",
                "minmov",
                "minmove2",
                "original_name",
                "pricescale",
                "pro_name",
                "short_name",
                "type",
                "update_mode",
                "volume",
                "currency_code",
                "rchp",
                "rtc",
            ],
        )
        self.send_message("quote_add_symbols", [self.session, symbol, {"flags": ["force_permission"]}])
        self.send_message("quote_fast_symbols", [self.session, symbol])
        self.send_message(
            "resolve_symbol",
            [
                self.chart_session,
                "symbol_1",
                '={"symbol":"'
                + symbol
                + '","adjustment":"'
                + adjustment
                + '","session":'
                + ('"regular"' if not extended_session else '"extended"')
                + "}",
            ],
        )
        self.send_message("create_series", [self.chart_session, "s1", "s1", "symbol_1", interval, limit])
        self.send_message("switch_timezone", [self.chart_session, "exchange"])

        raw_data = ""
        while True:
            try:
                result = self.ws.recv()
                raw_data = raw_data + result + "\n"
            except Exception as e:
                break
            if "series_completed" in result:
                break
        if return_raw:
            return raw_data
        return self.convert_raw_data(raw_data, symbol)

    @staticmethod
    def search_symbol(
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        delay: tp.Optional[int] = None,
        show_progress: bool = True,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.List[dict]:
        """Search for a symbol."""
        if text is None:
            text = ""
        if exchange is None:
            exchange = ""
        if pbar_kwargs is None:
            pbar_kwargs = {}
        symbols_remaining = None
        symbols_list = []
        pbar = None

        while symbols_remaining is None or symbols_remaining > 0:
            url = SEARCH_URL.format(text, exchange.upper(), len(symbols_list))
            resp = requests.get(url)
            symbols_data = json.loads(resp.text)
            symbols_remaining = symbols_data.get("symbols_remaining", 0)
            new_symbols = symbols_data.get("symbols", [])
            symbols_list.extend(new_symbols)
            if pbar is None and symbols_remaining > 0:
                pbar = get_pbar(
                    total=len(new_symbols) + symbols_remaining,
                    show_progress=show_progress,
                    **pbar_kwargs,
                )
            if pbar is not None:
                pbar.update(len(new_symbols))
            if delay is not None:
                time.sleep(delay / 1000)
        if pbar is not None:
            pbar.close()
        return symbols_list

    @staticmethod
    def scan_symbols(market: str) -> tp.List[dict]:
        """Scan symbols in a region/market."""
        url = SCAN_URL.format(market.lower())
        resp = requests.get(url)
        symbols_list = json.loads(resp.text)["data"]
        return symbols_list
