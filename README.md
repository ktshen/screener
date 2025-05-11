# Screener

## Introduction

該專案提供市場分析工具，能透過相對強度計算和趨勢模板條件篩選出表現優異的金融標的，同時運用動態時間規整算法尋找與歷史成功模式相似的當前價格形態。使用者可將所有分析結果直接匯入TradingView作為觀察清單，便於追蹤並進行深入分析。

This project offers powerful market analysis tools that identify outperforming assets through relative strength calculations and trend template conditions, while also utilizing Dynamic Time Warping to detect current price patterns similar to historically successful reference trends. Users can export all analysis results directly to TradingView watchlists for streamlined tracking and in-depth analysis.

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


Here's the revised section with all Important Notes integrated into a more cohesive format:


### 3. Similar Trend Screener

Analyze current market trends against reference patterns to find similar trend 

```bash
python3 similar_trend_screener.py
python3 similar_trend_screener.py --asset crypto -f output/2024-05-01/crypto_strong_targets.txt
```

Options
* `-f, --file` Path to file containing target symbols
* `--asset` Asset type (crypto or stock) [default: crypto]
* `-nv, --no_visualize` Disable visualization of results
* `-k, --topk` Number of top matches to record per reference [default: 15]
* `-s, --sleep` Sleep time between API requests in seconds [default: 1]

Important Notes:
- **Stock asset type is NOT YET SUPPORTED** in this version
- Before running, configure these settings in the script:
  * `UTC_ZONE`: Set your timezone (e.g., "Asia/Taipei")
  * `REFERENCE_TRENDS`: Define reference patterns with format `[start_datetime, end_datetime, timeframe, label]`
  * `TIMEFRAMES_TO_ANALYZE`: Choose which timeframes to analyze
- For effective reference trends:
  * End time should be the entry point instead of the whole reference trend you want to search for
  * Balance the number of high and low points for optimal normalization
- TradingView import limitation: 
  * When importing results, each symbol can only appear once in a TradingView watchlist. If a symbol appears in multiple sections, only its first occurrence will be imported

Results include:
1. Detailed similarity scores saved to `similarity_output/<timestamp>/dtw_similarity_detail.txt`
2. Visualizations of the best matches showing DTW alignment in `similarity_output/<timestamp>/vis_*` directories
3. TradingView-compatible watchlist files in `similarity_output/<timestamp>/<timestamp>_tradingview.txt`

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

## Dynamic Time Warping Similarity Calculation

Dynamic Time Warping (DTW) is a technique used to compare time series data when the time indices don't perfectly align. Unlike traditional Euclidean distance, DTW allows flexibility along the time axis, enabling optimal alignment between curves despite timing delays or stretches.

Our DTW implementation features:

1. **Multi-feature comparison** - Simultaneously considers Close price, SMA-30, SMA-45, SMA-60, and their relationships
2. **Normalization** - Normalizes price and indicator differences to enable cross-asset comparisons
3. **Weighted similarity calculation** - Considers both price levels and indicator differences with adjustable weights
4. **Time warping constraints** - Sets warping path limitations to ensure optimal pattern alignment

This approach identifies assets that are currently forming patterns similar to known successful reference trends, potentially indicating trading opportunities. Through visualization of DTW alignment paths, traders can intuitively see how current trends map to reference patterns.

## Output Files

The scripts save results in the following directories:
- For stocks & crypto relative strength: `output/<YYYY-MM-DD>/`
  - `<timestamp>_stock_strong_targets_<top_980|all>.txt`
  - `<timestamp>_crypto_relative_strength_<timeframe>.txt`
  - Failed symbols: `<timestamp>_failed_<tickers|cryptos>.txt`
- For similar trend analysis: `similarity_output/<timestamp>/`
  - Detailed results: `dtw_similarity_detail.txt`
  - TradingView format: `<timestamp>_tradingview.txt`
  - Visualizations: `vis_<timeframe>_<reference>_<label>/`

## License

[MIT](https://choosealicense.com/licenses/mit/)
