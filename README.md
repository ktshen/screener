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

Identify strong performing cryptocurrencies by comparing them with SMA-30, SMA-45 and SMA-60.

```bash
python3 crypto_screener.py
python3 crypto_screener.py -t "15m" -d 3
```

Options:
* `-t` Time frame (5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d) (default: 15m)
* `-d` Calculation duration in days (default: 3 days)

The script will generate a TXT file in `output/<date>/` directory that can be imported into [TradingView](https://www.tradingview.com/)'s watchlist.

### 2. Stock Screener 

Analyze US stocks using relative strength calculation and Minervini trend template conditions.

```bash
python3 stock_screener.py
python3 stock_screener.py -a -g 
```

Options:
* `-a, --all` Include all strong targets in output instead of just top 980 (TradingView has import limit)
* `-g` Ignore Minervini trend template conditions and calculate RS score only

The script will generate a TXT file in `output/<date>/` directory that can be imported into [TradingView](https://www.tradingview.com/)'s watchlist.

### 3. Crypto Trend Screener

Find cryptocurrencies with price patterns similar to predefined reference trends using Dynamic Time Warping (DTW) algorithms.

```bash
python3 crypto_trend_screener.py
python3 crypto_trend_screener.py --asset=crypto -f strong_targets.txt -k 10 -s 1.0
```

Options:
* `-f, --file` Path to strong target file (default: use all available symbols)
* `--asset` Asset type: 'crypto' or 'stock' (default: crypto)
* `-nv, --no_visualize` Disable DTW alignment visualizations
* `-k, --topk` Number of top symbols to record per reference trend (default: 6)
* `-s, --sleep` Sleep time between API requests in seconds (default: 0.5)

**Key Parameters:**
- **REFERENCE_TRENDS**: Define reference patterns in format `[start_datetime, end_datetime, timeframe, label]`
- **TIMEFRAMES_TO_ANALYZE**: Analysis timeframes, default ["15m", "30m", "1h", "2h", "4h"]
- **DTW_WINDOW_RATIO**: Controls horizontal matching flexibility (default: 0.2)
- **PRICE_WEIGHT/DIFF_WEIGHT**: Balance price vs moving average matching (default: 0.4/0.6)

The script generates TXT files and visualization charts in `similarity_output/<timestamp>/` directory.

### 4. Crypto Historical Trend Finder

Search historical cryptocurrency data for patterns similar to reference trends and analyze their future price movements with statistical insights.

```bash
python3 crypto_historical_trend_finder.py
python3 crypto_historical_trend_finder.py -k 500 -s 10
```

Options:
* `-k, --topk` Number of top matches to keep per reference trend (default: 300)
* `-s, --sleep` Sleep time between API requests in seconds (default: 15)

**Key Features:**
- Multi-timeframe pattern matching (15m, 30m, 1h, 2h, 4h)
- Future trend prediction with rise/fall statistics
- Three-panel visualizations (past + pattern + future)
- Statistical analysis with multiple extension factors (0.25x to 2.5x)
- Overlap filtering for clean analysis samples

**Key Parameters:**
- **REFERENCE_TRENDS**: Define reference patterns in format `[start_datetime, end_datetime, timeframe, label]`. Choose patterns with similar high/low bar counts and end at ideal entry points
- **TIMEZONE**: Timezone setting for datetime display and reference trend parsing (default: "America/Los_Angeles")
- **HISTORICAL_START_DATE**: Starting date for historical data collection (default: 2021-01-01)
- **TIMEFRAMES_TO_ANALYZE**: Analysis timeframes for pattern searching (default: ["15m", "30m", "1h", "2h", "4h"])
- **VIS_EXTENSION_FUTURE_LENGTH_FACTOR**: Determines how far into the future to observe after pattern completion for rise/fall classification. For example, 2.0 means observing 2x the pattern length into the future (default: 2.0)
- **EXTENSION_FACTORS_FOR_STATS**: Multiple factors for comprehensive statistical analysis [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
- **GLOBAL_OVERLAP_FILTERING**: 
  - `True`: No overlapping patterns across all symbols (stricter, higher quality)
  - `False`: Allow overlaps between different symbols (more samples)


**Statistical Output Example:**
```
Extension Factor Analysis:
  0.25x: Rise 198(43.5%) | Fall 254(55.8%) | Insufficient 3(0.7%)
  0.5x: Rise 191(42.0%) | Fall 259(56.9%) | Insufficient 5(1.1%)
  0.75x: Rise 186(40.9%) | Fall 260(57.1%) | Insufficient 9(2.0%)
  1.0x: Rise 169(37.1%) | Fall 275(60.4%) | Insufficient 11(2.4%)
  1.5x: Rise 166(36.5%) | Fall 268(58.9%) | Insufficient 21(4.6%)
  2.0x: Rise 159(34.9%) | Fall 267(58.7%) | Insufficient 29(6.4%)
  2.5x: Rise 167(36.7%) | Fall 250(54.9%) | Insufficient 38(8.4%)
```

Results are saved in `past_similar_trends_report/<timestamp>/` directory with comprehensive visualizations and statistical reports.

## Dynamic Time Warping (DTW)

DTW is an algorithm for measuring similarity between two temporal sequences that may vary in speed. Unlike Euclidean distance, DTW can handle sequences of unequal lengths and is invariant to time shifts, making it ideal for financial pattern matching.

**Applications in this project:**
- Price pattern similarity detection
- Moving average relationship matching  
- Historical trend analysis with future outcome prediction

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

## Configuration

### Reference Trends (for DTW-based tools)

Define reference patterns in the `REFERENCE_TRENDS` dictionary:

```python
REFERENCE_TRENDS = {
    "AVAX": [
        [datetime(2023, 11, 9, 12, 0), datetime(2023, 11, 14, 15, 0), "1h", "standard"],
    ],
    "SOL": [
        [datetime(2023, 9, 23, 0, 0), datetime(2023, 10, 15, 21, 0), "4h", "standard"]
    ],
}
```

**Tips:**
- Choose patterns with similar high/low bar counts for better normalization
- End patterns at ideal entry points
- Consider different timeframes for various market conditions

## Output Files

Scripts save results with the following patterns:
- **Crypto Screener**: `output/<YYYY-MM-DD>/<timestamp>_crypto_<timeframe>_strong_targets.txt`
- **Stock Screener**: `output/<YYYY-MM-DD>/<timestamp>_stock_strong_targets_<top_980|all>.txt`
- **Crypto Trend Screener**: `similarity_output/<timestamp>/<timestamp>_similar_trend_tradingview.txt` + visualization PNG files
- **Historical Trend Finder**: `past_similar_trends_report/<timestamp>/overall_summary.txt` + detailed analysis + visualization PNG files

**Visualization Features:**
- Candlestick charts with volume bars
- Moving average overlays (SMA-30, SMA-45, SMA-60)
- DTW alignment connection lines
- Normalized price scales for fair comparison
- Three-panel analysis (past/pattern/future) for historical finder

All TXT output files are compatible with [TradingView](https://www.tradingview.com/) watchlist import format.

## License

[MIT](https://choosealicense.com/licenses/mit/)
