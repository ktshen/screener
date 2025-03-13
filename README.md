# Screener

下載美股和加密貨幣的歷史數據並透過自己的策略去找到強勢標的

The purpose of this project is to download historical data for US stocks and cryptocurrencies, and use different strategies to identify strong performing assets.

## Installation

```bash
pip3 install -r requirements.txt
```

API keys are needed for [Polygon](https://polygon.io) and [Stocksymbol](https://stock-symbol.herokuapp.com) (Not a requirement for Crypto usage)

Rename `api_keys.json.example` to `api_keys.json`

## Usage

### 1. Crypto Screener

Identify strong performing crypto currencies by comparing them with SMA-30, SMA-45 and SMA-60.

```bash
python3 crypto_screener.py
python3 crypto_screener.py -t "15m" -d 3
```

Options
* `-t` Time frame (5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d) (default: 15m)
* `-d` Calculation duration in days (default: 3 days)

The script will generate a TXT file in `output/<date>/` directory that can be imported into [TradingView](https://www.tradingview.com/)'s watchlist.

### 2. Stock Screener 

Analyze US stocks using relative strength calculation and trend template conditions.

```bash
python3 stock_screener.py
python3 stock_screener.py -a -g 
```

Options
* `-a, --all` Include all strong targets in output instead of just top 980 (TradingView has import limit)
* `-g` Ignore Minervini trend template conditions and calculate RS score only

The script will generate a TXT file in `output/<date>/` directory that can be imported into [TradingView](https://www.tradingview.com/)'s watchlist.


## Relative Strength Formula 
The RS score is calculated using a weighted sum of relative strength indicators:


$$ bars = \text{total bars (depend on time frame, e.g. } 4 \times 24 \times days \text{ for 15m})$$

$$ W_i = e^{2 \times \ln(2) \times i / bars}  $$

$$ \begin{align*}
N_i & = \frac{(P_i - MA30_i) + (P_i - MA45_i) + (P_i - MA60_i) + (MA30_i - MA45_i) + (MA30_i - MA60_i) + (MA45_i - MA60_i)}{ATR_i}\\  
\end{align*}  $$

$$ Score = \frac{\sum_{i=1}^{bars} N_i \times W_i}{\sum_{i=1}^{bars} W_i}   $$

Where:
- Weight is calculated as an exponential function to make the weight at the midpoint (L/2) is exactly half of the weight at the endpoint (L)
- ATR (Average True Range) is used for normalization


## Output Files

Both scripts save results in the `output/<YYYY-MM-DD>/` directory with the following files:
- For stocks: `<timestamp>_stock_strong_targets_<top_980|all>.txt`
- For crypto: `<timestamp>_crypto_relative_strength_<timeframe>.txt`
- Failed symbols: `<timestamp>_failed_<tickers|cryptos>.txt`

## License

[MIT](https://choosealicense.com/licenses/mit/)