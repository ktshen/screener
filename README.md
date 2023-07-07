# Screener

下載美股和加密貨幣的歷史數據並透過自己的策略去找到強勢標的

The purpose of this project is to download historical data for US stocks and cryptocurrencies, and use different strategies to identify strong performing assets.

## Installation

```bash
pip3 install -r requirements.txt
```

API keys are needed for [Tiingo](https://tiingo.com/) and [Stocksymbol](https://stock-symbol.herokuapp.com). (Not a requirement for Crypto usage)

## Strategy Usage

Crypto relative strength: Within a 15-minute timeframe, identify strong performing assets by comparing them with SMA-30, SMA-45 and SMA-60.

```bash
python3 crypto_relative_strength.py
```

US stock trend template: Utilize Mark Minervini's trend template to filter out strong performing stocks.

```bash
python3 stock_trend_template.py
```

Both scripts will generate a TXT file that can be imported into [TradingView](https://www.tradingview.com/)'s watchlist.

## Crypto Relative Strength Formula 
$$ bars = 4 \times 25 \times days  \text{  (15m time frame)} $$

$$ W = \frac{(bars-i)\times days}{bars} + 1 $$

$$ \begin{align*}
N_i & = \left [ (P_i - MA30_i) + (P_i - MA45_i) + (P_i - MA60_i) \right ]\times W\\  
                       & +(MA30_i - MA60_i) +(MA30_i - MA45_i) + (MA45_i - MA60_i)\\  
\end{align*} $$

$$ Score = \sum_{i=1}^{bars} \frac{N_i} {MA60_i} \times (bars - i)  \\ \text{where i=1 means the closest bar} $$


## Download historical data only
To import crypto or stock downloader for your own usage, simply include the following line in your Python code:

```python3
from src.downloader import StockDownloader
from src.downloader import CryptoDownloader
```

When devising your own strategy, feel free to refer to the existing strategies for guidance and inspiration. The stock data is downloaded from Tiingo and Yahoo Finance, and the cryptocurrency data is obtained from Binance.

- Change CURRENT_TIMEZONE in src/downloader.py if timezone is essential to you.


## License

[MIT](https://choosealicense.com/licenses/mit/)
