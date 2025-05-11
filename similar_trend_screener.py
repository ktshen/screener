"""
Trend Similarity Analyzer
-----------------------------------------------
This script identifies assets currently forming price patterns similar to historical successful
setups using Dynamic Time Warping (DTW). By analyzing price and moving average relationships
across multiple timeframes, it discovers high-probability trading opportunities.

Key Features:
- Advanced pattern recognition using price levels and indicator relationships
- Customizable reference trends with visual alignment reports
- TradingView-compatible output for efficient tracking

Usage:
 python similar_trend_screener.py [options]

Configuration:
- Set your timezone (UTC_ZONE), reference patterns (REFERENCE_TRENDS), and analysis timeframes
- Define reference trends with format [start_datetime, end_datetime, timeframe, label]
- End reference patterns at ideal entry points and balance high/low point distribution

Notes:
- Stock asset type is not yet supported
- Results saved to similarity_output/[timestamp] directory
- TradingView watchlists only display the first occurrence of duplicate symbols
"""
import os
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from dtaidistance import dtw, dtw_ndim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytz import timezone
from src.downloader import StockDownloader, CryptoDownloader


# =========== Reference Trend Configuration ===========
# Define reference trends in datetime format (will be converted to timestamps)
# Format: [start_datetime, end_datetime, timeframe, label]
REFERENCE_TRENDS = {
    "AVAX": [
        [datetime(2023, 11, 9, 12, 0), datetime(2023, 11, 14, 15, 0), "1h", "standard"],
    ],
    "MKR" :[
        [datetime(2023, 6, 26, 13, 0), datetime(2023, 7, 17, 12, 0), "4h", "standard"],
    ],
    "CRV": [
        [datetime(2024, 11, 4, 0, 0), datetime(2024, 11, 21, 0, 0), "4h", "uptrend"],
        [datetime(2024, 11, 4, 0, 0), datetime(2024, 11, 28, 0, 0), "4h", "uptrend_2"],

    ],
    "GMT": [
        [datetime(2022, 3, 26, 9, 0), datetime(2022, 4, 14, 21, 0), "4h", "uptrend"]
    ],
    "SOL": [
        [datetime(2023, 9, 23, 0, 0), datetime(2023, 10, 15, 21, 0), "4h", "standard"]
    ],
}

# ================ Configuration ================
# Timezone for datetime conversion
UTC_ZONE = "America/Los_Angeles"

# Output directory path
OUTPUT_DIR = "similarity_output"

# Reference trend temporary storage directory
REF_TEMP_DIR = os.path.join(OUTPUT_DIR, "ref_temp")

# Timeframes to analyze
TIMEFRAMES_TO_ANALYZE = ["15m", "1h", "2h", "4h"]

# Top K symbols to record per reference trend
TOP_K = 15

# Constants for similarity calculation
SMA_PERIODS = [30, 45, 60]
DATA_MULTIPLIER = 3  # How many times more historical data to fetch for SMA calculation
DTW_WINDOW_FACTORS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # Window size factors for DTW
MIN_QUERY_LENGTH = 60  # Minimum number of data points required for query trend

# DTW window constraint - limit the warping path
DTW_WINDOW_RATIO = 0.4  # Window size as a fraction of sequence length

# Weights for price and difference features in DTW calculation
PRICE_WEIGHT = 0.6
DIFF_WEIGHT = 0.4

# BINANCE API request interval parameter
API_SLEEP_SECONDS = 1  # Sleep time between requests (seconds)

# Maximum distance limit for DTW matching points - points beyond this distance won't be matched
DTW_MAX_POINT_DISTANCE = 1.75

# Request buffer ratio
REQUEST_TIME_BUFFER_RATIO = 1.8
# ================ End of Configuration ================


# ================ Utility Functions ================

def convert_datetime_to_timestamp(dt_obj, tz_name):
    """
    Convert datetime object to timestamp with timezone consideration.

    Args:
        dt_obj: Datetime object to convert
        tz_name: Timezone name string

    Returns:
        int: Unix timestamp
    """
    tz = timezone(tz_name)
    dt_with_tz = tz.localize(dt_obj)
    return int(dt_with_tz.timestamp())


def calculate_timeframe_seconds(timeframe: str) -> int:
    """
    Calculate seconds corresponding to the specified timeframe.

    Args:
        timeframe: Timeframe string (e.g., "15m", "1h")

    Returns:
        int: Number of seconds in the timeframe
    """
    # Parse timeframe string
    if 'm' in timeframe:
        # Minute timeframe
        minutes = int(timeframe.replace('m', ''))
        return minutes * 60
    elif 'h' in timeframe:
        # Hour timeframe
        hours = int(timeframe.replace('h', ''))
        return hours * 3600
    else:
        # Default case, return 1 hour in seconds
        print(f"Unknown timeframe format: {timeframe}, defaulting to 1 hour")
        return 3600


def parse_strong_targets(filepath: str) -> list[str]:
    """
    Parse strong target symbols from file.

    Args:
        filepath: Path to the file containing target symbols

    Returns:
        list: List of target symbols
    """
    if not os.path.exists(filepath):
        print(f"Target file not found: {filepath}")
        return []

    targets = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    target_section = False
    for line in lines:
        if "###TARGETS" in line:
            target_section = True
            continue
        if target_section and line.strip():
            symbols = line.strip().split(',')
            targets.extend([s.split(':')[-1].replace('USDT.P', '').strip()
                         for s in symbols if s.strip()])

    if not targets:
        print(f"No valid targets found in {filepath}")
        return []

    return targets


def process_symbol_dtw(args: tuple) -> dict:
    """
    Process a single symbol DTW comparison (designed for multiprocessing).

    Args:
        args: Tuple containing (target_symbol, target_df, timeframe, ref_symbol, ref_idx, ref_df,
                               ref_timeframe, ref_label)

    Returns:
        dict: Dictionary containing similarity results
    """
    target_symbol, target_df, timeframe, ref_symbol, ref_idx, ref_df, ref_timeframe, ref_label = args

    print(f"Processing DTW for {target_symbol} [{timeframe}] against {ref_symbol} reference #{ref_idx} ({ref_label}) [{ref_timeframe}]...")

    # Calculate similarity
    dtw_calculator = DTWSimilarityCalculator()
    similarity, distance, warping_path, window_data, window_info, norm_params = dtw_calculator.find_best_similarity_window(
        ref_df, target_df
    )

    return {
        "symbol": target_symbol,
        "timeframe": timeframe,
        "ref_symbol": ref_symbol,
        "ref_idx": ref_idx,
        "ref_timeframe": ref_timeframe,
        "ref_label": ref_label,
        "score": similarity,
        "distance": distance,
        "warping_path": warping_path,
        "window_data": window_data,
        "window_info": window_info,
        "norm_params": norm_params
    }


def load_or_fetch_reference_data(ref_symbol, ref_trend_info, asset_type, ref_idx):
    """
    Load or fetch reference trend data, caching to file.

    Args:
        ref_symbol: Symbol of reference trend
        ref_trend_info: List containing [start_datetime, end_datetime, timeframe, label]
        asset_type: Asset type ("crypto" or "stock")
        ref_idx: Index of reference trend

    Returns:
        dict: Dictionary containing reference data or None if failed
    """
    # Create reference data cache directory
    os.makedirs(REF_TEMP_DIR, exist_ok=True)

    # Unpack reference trend information
    start_datetime, end_datetime, ref_timeframe, label = ref_trend_info

    # Convert datetime to timestamp
    start_ts = convert_datetime_to_timestamp(start_datetime, UTC_ZONE)
    end_ts = convert_datetime_to_timestamp(end_datetime, UTC_ZONE)

    # Modified cache file path, using new naming format <symbol>_<timeframe>_<label>_<start timestamp>
    cache_file = os.path.join(REF_TEMP_DIR, f"{ref_symbol}_{ref_timeframe}_{label}_{start_ts}.pkl")

    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached reference data for {ref_symbol} #{ref_idx}...")
        with open(cache_file, 'rb') as f:
            ref_data = pickle.load(f)
            # Create original price chart for already cached data
            chart_file = os.path.join(REF_TEMP_DIR, f"{ref_symbol}_{ref_timeframe}_{label}_{start_ts}.png")
            if not os.path.exists(chart_file):
                save_reference_chart(ref_symbol, ref_data['df'], ref_timeframe, label, start_ts)
            return ref_data

    # If cache doesn't exist, fetch data
    print(f"Fetching reference data for {ref_symbol} #{ref_idx} ({ref_trend_info[3]}) at {ref_trend_info[2]} timeframe...")

    # Initialize data processor
    data_processor = DataProcessor(asset_type)

    # Get reference trend data
    ref_df = data_processor.get_data(
        ref_symbol,
        ref_timeframe,
        start_ts,
        end_ts,
        is_crypto=(asset_type == "crypto"),
        include_buffer=False,
        is_reference=True  # Mark as reference trend to disable validation
    )

    if ref_df.empty:
        print(f"Could not get data for reference symbol {ref_symbol} at {ref_timeframe}")
        return None

    if len(ref_df) < MIN_QUERY_LENGTH:
        print(f"Insufficient data points for reference trend: {len(ref_df)} < {MIN_QUERY_LENGTH}")
        return None

    print(f"Reference data length: {len(ref_df)} points")
    print(f"Reference period: {ref_df.index[0]} to {ref_df.index[-1]}")

    # Create data object to cache
    ref_data = {
        'df': ref_df,
        'timeframe': ref_timeframe,
        'label': label,
        'start_ts': start_ts,
        'end_ts': end_ts
    }

    # Save to cache file
    with open(cache_file, 'wb') as f:
        pickle.dump(ref_data, f)

    # Create and save original price chart for reference trend
    save_reference_chart(ref_symbol, ref_df, ref_timeframe, label, start_ts)

    return ref_data


def save_reference_chart(ref_symbol, ref_df, ref_timeframe, label, start_ts):
    """
    Create and save original price chart for reference trend (non-normalized).

    Args:
        ref_symbol: Symbol of reference trend
        ref_df: DataFrame containing reference data
        ref_timeframe: Timeframe of reference data
        label: Label of reference trend
        start_ts: Start timestamp
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot closing price and moving averages
    ax.plot(ref_df.index, ref_df['Close'], 'r-', linewidth=2, label='Close')
    ax.plot(ref_df.index, ref_df['SMA_30'], 'g-', linewidth=1, alpha=0.7, label='SMA30')
    ax.plot(ref_df.index, ref_df['SMA_45'], 'b-', linewidth=1, alpha=0.7, label='SMA45')
    ax.plot(ref_df.index, ref_df['SMA_60'], 'y-', linewidth=1, alpha=0.7, label='SMA60')

    # Add basic information
    plt.title(f'Reference Trend: {ref_symbol} {ref_timeframe} ({label})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # Add annotation information
    info_text = (f"Symbol: {ref_symbol}\n"
                f"Timeframe: {ref_timeframe}\n"
                f"Label: {label}\n"
                f"Start: {datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M')}\n"
                f"End: {ref_df.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
                f"Data Points: {len(ref_df)}")

    # Add text box, positioned at bottom left of chart
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Format date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add grid lines
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save image to same directory as cache file, using same naming convention
    chart_filename = f"{ref_symbol}_{ref_timeframe}_{label}_{start_ts}.png"
    chart_filepath = os.path.join(REF_TEMP_DIR, chart_filename)
    plt.savefig(chart_filepath, dpi=150)
    plt.close(fig)

    print(f"Saved reference chart to {chart_filepath}")


def visualize_dtw_alignment(query_df, window_df, warping_path, ref_symbol, target_symbol,
                          timeframe, similarity, save_dir, ref_label):
    """
    Visualize DTW alignment, using independent normalization to better display trends.

    Args:
        query_df: DataFrame containing reference data
        window_df: DataFrame containing target window data
        warping_path: Warping path from DTW calculation
        ref_symbol: Symbol of reference trend
        target_symbol: Target symbol
        timeframe: Timeframe being analyzed
        similarity: Similarity score
        save_dir: Directory to save visualization
        ref_label: Label of reference trend
    """
    if not warping_path:
        print(f"No warping path available for visualization of {target_symbol}")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False, gridspec_kw={'height_ratios': [1, 1]})

    # Define columns to display
    price_cols = ['Close', 'SMA_30', 'SMA_45', 'SMA_60']

    # Collect all price values for normalization, but normalize reference and target sequences separately
    # Calculate normalization parameters for reference sequence
    ref_prices = []
    for col in price_cols:
        ref_prices.extend(query_df[col].values)

    ref_mean = np.mean(ref_prices)
    ref_std = np.std(ref_prices)

    # Calculate normalization parameters for target sequence
    target_prices = []
    for col in price_cols:
        target_prices.extend(window_df[col].values)

    target_mean = np.mean(target_prices)
    target_std = np.std(target_prices)

    # Normalization functions
    def normalize_ref(series):
        return (series - ref_mean) / ref_std

    def normalize_target(series):
        return (series - target_mean) / target_std

    # Plot reference sequence
    ax1.plot(query_df.index, normalize_ref(query_df['Close']), 'r-', linewidth=2, label='Close')
    ax1.plot(query_df.index, normalize_ref(query_df['SMA_30']), 'g-', linewidth=1, alpha=0.7, label='SMA30')
    ax1.plot(query_df.index, normalize_ref(query_df['SMA_45']), 'b-', linewidth=1, alpha=0.7, label='SMA45')
    ax1.plot(query_df.index, normalize_ref(query_df['SMA_60']), 'y-', linewidth=1, alpha=0.7, label='SMA60')
    ax1.set_ylabel('Normalized Price (z-score)')
    ax1.set_title(f'{ref_symbol} Reference Trend - {ref_label}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Plot target sequence
    ax2.plot(window_df.index, normalize_target(window_df['Close']), 'r-', linewidth=2, label='Close')
    ax2.plot(window_df.index, normalize_target(window_df['SMA_30']), 'g-', linewidth=1, alpha=0.7, label='SMA30')
    ax2.plot(window_df.index, normalize_target(window_df['SMA_45']), 'b-', linewidth=1, alpha=0.7, label='SMA45')
    ax2.plot(window_df.index, normalize_target(window_df['SMA_60']), 'y-', linewidth=1, alpha=0.7, label='SMA60')
    ax2.set_ylabel('Normalized Price (z-score)')
    ax2.set_title(f'{target_symbol} Current Trend (Latest data: {window_df.index[-1].strftime("%Y-%m-%d")})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set y-axis range for each chart separately, adding padding for aesthetics
    ref_norm_values = [normalize_ref(query_df[col]) for col in price_cols]
    all_ref_norm = np.concatenate(ref_norm_values)
    ref_ymin = np.nanmin(all_ref_norm)
    ref_ymax = np.nanmax(all_ref_norm)
    ref_padding = (ref_ymax - ref_ymin) * 0.1
    ax1.set_ylim(ref_ymin - ref_padding, ref_ymax + ref_padding)

    target_norm_values = [normalize_target(window_df[col]) for col in price_cols]
    all_target_norm = np.concatenate(target_norm_values)
    target_ymin = np.nanmin(all_target_norm)
    target_ymax = np.nanmax(all_target_norm)
    target_padding = (target_ymax - target_ymin) * 0.1
    ax2.set_ylim(target_ymin - target_padding, target_ymax + target_padding)

    # Draw connecting lines - need to convert normalized values from two charts to relative positions
    ref_close_norm = normalize_ref(query_df['Close'])
    target_close_norm = normalize_target(window_df['Close'])

    # Determine number of connection points
    step_size = max(1, len(warping_path) // 100)

    # Add markers at connection points
    connected_ref_indices = [i for i, _ in warping_path[::step_size]]
    connected_target_indices = [j for _, j in warping_path[::step_size]]

    ax1.scatter(query_df.index[connected_ref_indices],
               ref_close_norm[connected_ref_indices],
               color='darkred', s=15, alpha=0.6, zorder=5)
    ax2.scatter(window_df.index[connected_target_indices],
               target_close_norm[connected_target_indices],
               color='darkred', s=15, alpha=0.6, zorder=5)

    # Draw connecting lines - using relative positions in figure space
    for idx, (i, j) in enumerate(warping_path):
        if idx % step_size == 0:
            ref_x, ref_y = query_df.index[i], ref_close_norm[i]
            target_x, target_y = window_df.index[j], target_close_norm[j]

            # Convert to data space coordinates
            ref_disp = ax1.transData.transform((mdates.date2num(ref_x), ref_y))
            target_disp = ax2.transData.transform((mdates.date2num(target_x), target_y))

            # Convert to figure space coordinates
            ref_fig = fig.transFigure.inverted().transform(ref_disp)
            target_fig = fig.transFigure.inverted().transform(target_disp)

            # Draw connecting line
            fig.lines.append(plt.Line2D([ref_fig[0], target_fig[0]],
                                       [ref_fig[1], target_fig[1]],
                                       transform=fig.transFigure,
                                       color='gray',
                                       alpha=0.4,
                                       linewidth=0.7,
                                       linestyle='-'))

    # Add title and normalization information
    plt.suptitle(f'DTW Alignment: {ref_symbol}({ref_label}) vs {target_symbol} ({timeframe})\n'
                f'Similarity: {similarity:.4f}, Max Step: {DTW_MAX_POINT_DISTANCE}',
                fontsize=16)

    # Ensure layout is compact
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)

    # Save image
    os.makedirs(save_dir, exist_ok=True)
    filename = f"score_{similarity:.4f}_{target_symbol}_{timeframe}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {filepath}")


# ================ Core Classes ================

class DataProcessor:
    """
    Base class for data processing. Handles data retrieval and preprocessing
    for both stocks and cryptocurrencies.
    """
    def __init__(self, asset_type: str):
        """
        Initialize appropriate downloader based on asset type.

        Args:
            asset_type: Type of asset ("crypto" or "stock")
        """
        if asset_type == "crypto":
            self.downloader = CryptoDownloader()
        else:
            self.downloader = StockDownloader(save_dir=".", api_file="api_keys.json")
        self.asset_type = asset_type

    def get_data(self, symbol: str, timeframe: str, start_ts: int, end_ts: int,
                 is_crypto: bool = False, include_buffer: bool = True,
                 is_reference: bool = False) -> pd.DataFrame:
        """
        Get data with buffer period for SMA calculation.

        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe of data
            start_ts: Start timestamp
            end_ts: End timestamp
            is_crypto: Whether the symbol is a cryptocurrency
            include_buffer: Whether to include buffer period for SMA calculation
            is_reference: Whether this is a reference trend (disables validation)

        Returns:
            DataFrame: Processed data with calculated features
        """
        if include_buffer:
            # Calculate buffer period for SMA calculation
            interval = end_ts - start_ts
            buffer_start_ts = start_ts - (interval * DATA_MULTIPLIER)
        else:
            buffer_start_ts = start_ts

        # Get data using the appropriate downloader
        if is_crypto:
            # For crypto, add USDT if not already there
            if not symbol.endswith("USDT"):
                symbol_full = f"{symbol}USDT"
            else:
                symbol_full = symbol

            # Set validate=False for reference trends, otherwise use default (True)
            success, df = self.downloader.get_data(
                symbol_full,
                buffer_start_ts,
                end_ts,
                validate=not is_reference,  # Disable validation for reference trends
                timeframe=timeframe
            )
        else:  # stock
            success, df = self.downloader.get_data(
                symbol,
                buffer_start_ts,
                end_ts,
                validate=not is_reference,  # Disable validation for reference trends
                timeframe=timeframe
            )

        if not success or df is None or df.empty:
            print(f"Failed to get data for {symbol} ({timeframe})")
            return pd.DataFrame()

        # Convert timestamp to datetime if not already done
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # Set datetime as index if not already done
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('datetime')

        # Filter to requested time range
        start_time = pd.Timestamp.fromtimestamp(start_ts)
        end_time = pd.Timestamp.fromtimestamp(end_ts)
        df = df[(df.index >= start_time) & (df.index <= end_time)]

        # Calculate SMAs if not already present
        close_col = 'close'
        for period in SMA_PERIODS:
            sma_col = f'sma_{period}'
            if sma_col not in df.columns:
                df[sma_col] = df[close_col].rolling(window=period).mean()

        # Standardize column names for the DTW algorithm
        df = df.rename(columns={
            'close': 'Close',
            'sma_30': 'SMA_30',
            'sma_45': 'SMA_45',
            'sma_60': 'SMA_60'
        })

        # Pre-calculate difference features
        df['Close_SMA30'] = df['Close'] - df['SMA_30']
        df['Close_SMA45'] = df['Close'] - df['SMA_45']
        df['Close_SMA60'] = df['Close'] - df['SMA_60']
        df['SMA30_SMA45'] = df['SMA_30'] - df['SMA_45']
        df['SMA45_SMA60'] = df['SMA_45'] - df['SMA_60']
        df['SMA30_SMA60'] = df['SMA_30'] - df['SMA_60']

        return df


class DTWSimilarityCalculator:
    """
    Calculate similarity using Dynamic Time Warping (DTW) algorithm.
    Handles feature normalization and similarity calculations.
    """
    def check_c_availability(self) -> bool:
        """
        Check if C implementation of dtaidistance is available.

        Returns:
            bool: True if C implementation is available, False otherwise
        """
        try:
            # This function will attempt to import the C library and return a status
            c_available = dtw.try_import_c(verbose=False)
            return c_available
        except Exception as e:
            print(f"Error checking C availability: {e}")
            return False

    def normalize_features(self, df: pd.DataFrame) -> tuple:
        """
        Normalize price and difference features based on current window data.
        For each feature category, normalization uses global mean and standard deviation
        of all values in that category to maintain relative relationships between features.

        Args:
            df: DataFrame containing features to normalize

        Returns:
            tuple: (price_features_normalized, diff_features_normalized, norm_params)
        """
        # Define price and difference feature columns
        price_cols = ['Close', 'SMA_30', 'SMA_45', 'SMA_60']
        diff_cols = ['Close_SMA30', 'Close_SMA45', 'Close_SMA60',
                     'SMA30_SMA45', 'SMA45_SMA60', 'SMA30_SMA60']

        # Normalize price features - using global mean and std of price features
        price_values = df[price_cols].values.flatten()
        price_mean = np.mean(price_values)
        price_std = np.std(price_values)

        price_series = []
        for col in price_cols:
            if price_std > 0:
                normalized = (df[col] - price_mean) / price_std
            else:
                min_val = np.min(price_values)
                max_val = np.max(price_values)
                if max_val > min_val:
                    normalized = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros(len(df))

            price_series.append(normalized.values)

        # Normalize difference features - using global mean and std of difference features
        diff_values = df[diff_cols].values.flatten()
        diff_mean = np.mean(diff_values)
        diff_std = np.std(diff_values)

        diff_series = []
        for col in diff_cols:
            if diff_std > 0:
                normalized = (df[col] - diff_mean) / diff_std
            else:
                min_val = np.min(diff_values)
                max_val = np.max(diff_values)
                if max_val > min_val:
                    normalized = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros(len(df))

            diff_series.append(normalized.values)

        # Stack features into 2D arrays
        price_features = np.column_stack(price_series)
        diff_features = np.column_stack(diff_series)

        # Return normalized features and normalization parameters (for visualization)
        norm_params = {
            'price': {'mean': price_mean, 'std': price_std},
            'diff': {'mean': diff_mean, 'std': diff_std},
            'method': 'z-score'
        }

        return price_features, diff_features, norm_params

    def calculate_dtw_similarity(self, query_series: np.ndarray, target_series: np.ndarray) -> tuple:
        """
        Calculate multivariate DTW similarity and return warping path.
        Applies point distance limit, points exceeding DTW_MAX_POINT_DISTANCE won't be matched.

        Args:
            query_series: Query series (normalized features)
            target_series: Target series (normalized features)

        Returns:
            tuple: (similarity, distance, path)
        """
        # Check if C implementation is available
        use_c = self.check_c_availability()

        # Calculate window size as a proportion of the shorter sequence length
        window_size = int(min(len(query_series), len(target_series)) * DTW_WINDOW_RATIO)

        # Use dtw_ndim.warping_paths to get distance and path matrix, with window constraint
        distance, paths = dtw_ndim.warping_paths(
            query_series,
            target_series,
            window=window_size,  # Apply window constraint
            use_c=use_c,  # Use C implementation if available
            max_step=DTW_MAX_POINT_DISTANCE  # Add point distance
        )

        # If distance is inf (no valid path found), return low similarity
        if np.isinf(distance):
            return 0.0, float('inf'), []

        # Use dtw.best_path to find optimal path in path matrix
        path = dtw.best_path(paths)

        # Convert distance to similarity score (inverse and normalize)
        similarity = 1 / (1 + distance)

        return similarity, distance, path

    def find_best_similarity_window(self, query_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple:
        """
        Find best similarity window based on price and difference features.
        Compares the most recent data window, with right boundary fixed at the
        last time point of the target sequence.

        Args:
            query_df: DataFrame containing reference data
            target_df: DataFrame containing target data

        Returns:
            tuple: (similarity, distance, path, window_data, window_info, norm_params)
        """
        query_len = len(query_df)

        # Check if target sequence is long enough
        if len(target_df) < query_len * min(DTW_WINDOW_FACTORS):
            return 0.0, float('inf'), None, None, None, None

        best_similarity = -1
        best_distance = float('inf')
        best_path = None
        best_window_data = None
        best_window_info = None
        best_norm_params = None

        # Pre-process reference sequence features
        query_price_norm, query_diff_norm, _ = self.normalize_features(query_df)

        # Try different window sizes, but fix right boundary at the last time point of target sequence
        for factor in DTW_WINDOW_FACTORS:
            window_size = int(query_len * factor)

            # Skip if window size exceeds target sequence length
            if window_size > len(target_df):
                continue

            # Calculate window start index, fixing right boundary at the latest data point
            start_idx = len(target_df) - window_size

            # Extract window data
            window = target_df.iloc[start_idx:len(target_df)]

            # Confirm window length is correct
            if len(window) != window_size:
                print(f"Warning: Window size mismatch. Expected {window_size}, got {len(window)}")
                continue

            # Normalize target window features - based on current window data
            window_price_norm, window_diff_norm, norm_params = self.normalize_features(window)

            # Calculate DTW for price features
            price_similarity, price_distance, price_path = self.calculate_dtw_similarity(
                query_price_norm, window_price_norm
            )

            # If no valid path found (distance is inf), continue to next factor
            if np.isinf(price_distance):
                print(f"  Factor {factor}: No valid path found due to max_step constraint")
                continue

            # Calculate DTW for difference features
            diff_similarity, diff_distance, diff_path = self.calculate_dtw_similarity(
                query_diff_norm, window_diff_norm
            )

            # If no valid path found for difference features, continue to next factor
            if np.isinf(diff_distance):
                print(f"  Factor {factor}: No valid path found for diff features")
                continue

            # Consider feature count difference (4 price features vs 6 difference features)
            feature_ratio = 6/4  # diff:price feature ratio

            # Weighted combination score
            similarity = (PRICE_WEIGHT * price_similarity * feature_ratio +
                         DIFF_WEIGHT * diff_similarity) / (PRICE_WEIGHT * feature_ratio + DIFF_WEIGHT)

            # Weighted combination distance
            distance = (PRICE_WEIGHT * price_distance * feature_ratio +
                       DIFF_WEIGHT * diff_distance) / (PRICE_WEIGHT * feature_ratio + DIFF_WEIGHT)

            # Use price path as final path
            path = price_path

            # If similarity is higher, update best results
            if similarity > best_similarity:
                best_similarity = similarity
                best_distance = distance
                best_path = path
                best_window_data = window
                best_window_info = (factor, start_idx, len(target_df))
                best_norm_params = norm_params

            print(f"  Factor {factor}: similarity={similarity:.4f}, window={start_idx}:{len(target_df)}, "
                  f"period={window.index[0]} to {window.index[-1]}")

        if best_window_data is not None:
            print(f"  Best window: {best_window_data.index[0]} to {best_window_data.index[-1]}")

        # Return all relevant information
        return best_similarity, best_distance, best_path, best_window_data, best_window_info, best_norm_params


# ================ Main Function ================

def main():
    """
    Main function to run the DTW similarity analysis.
    Handles command-line arguments, processes data, and generates visualizations and reports.
    """
    # Record start time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Analyze trend similarity for crypto or stock symbols')
    parser.add_argument('-f', '--file', required=False, default="", help='Path to strong target file')
    parser.add_argument('--asset', choices=['crypto', 'stock'], default='crypto', help='Asset type of input file (default: crypto)')
    parser.add_argument('-nv', '--no_visualize', action='store_true', default=False, help='Enable visualization of DTW alignments')
    parser.add_argument('-k', '--topk', type=int, default=TOP_K, help=f'Number of top symbols to record per reference trend (default: {TOP_K})')
    parser.add_argument('-s', '--sleep', type=float, default=API_SLEEP_SECONDS, help=f'Sleep time between API requests in seconds (default: {API_SLEEP_SECONDS})')
    args = parser.parse_args()

    # Use parameters
    enable_visualization = not args.no_visualize
    top_k = args.topk
    api_sleep_seconds = args.sleep

    print(f"\nAnalysis Configuration:")
    print(f"Input Asset Type: {args.asset}")
    print(f"Reference Trends: {len(REFERENCE_TRENDS)} symbols with {sum(len(trends) for trends in REFERENCE_TRENDS.values())} total trends")
    print(f"Timeframes to Analyze: {TIMEFRAMES_TO_ANALYZE}")
    print(f"DTW Window Ratio: {DTW_WINDOW_RATIO}")
    print(f"DTW Max Point Distance: {DTW_MAX_POINT_DISTANCE}")
    print(f"Price Weight: {PRICE_WEIGHT}, Diff Weight: {DIFF_WEIGHT}")
    print(f"Request Time Buffer Ratio: {REQUEST_TIME_BUFFER_RATIO}")
    print(f"API Sleep Time: {api_sleep_seconds} seconds")
    print(f"Visualization Enabled: {enable_visualization}")
    print(f"Top K Symbols: {top_k}\n")

    # Initialize data processor
    data_processor = DataProcessor(args.asset)

    # Get target symbols - either from file or all available
    if args.file:
        target_symbols = parse_strong_targets(args.file)
        if not target_symbols:
            # If file provided but parsing failed, use all symbols
            if args.asset == "crypto":
                target_symbols = data_processor.downloader.get_all_symbols()
                # Remove USDT suffix for clean symbol names
                target_symbols = [s.replace('USDT', '') for s in target_symbols]
            else:
                target_symbols = data_processor.downloader.get_all_tickers()
    else:
        # No file provided, use all symbols
        if args.asset == "crypto":
            target_symbols = data_processor.downloader.get_all_symbols()
            # Remove USDT suffix for clean symbol names
            target_symbols = [s.replace('USDT', '') for s in target_symbols]
        else:
            target_symbols = data_processor.downloader.get_all_tickers()

    print(f"Found {len(target_symbols)} targets to analyze")

    # Create output directory
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = f"{OUTPUT_DIR}/{date_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Store all results
    all_results = {}

    # Load or retrieve all reference trend data
    reference_data = {}
    for ref_symbol, ref_trends in REFERENCE_TRENDS.items():
        for ref_idx, ref_trend_info in enumerate(ref_trends):
            ref_data = load_or_fetch_reference_data(ref_symbol, ref_trend_info, args.asset, ref_idx)

            if ref_data is not None:
                reference_data[(ref_symbol, ref_idx)] = ref_data

    # Find longest sequence length (data point count) in reference data
    max_ref_length = 0
    for ref_info in reference_data.values():
        ref_df = ref_info['df']
        max_ref_length = max(max_ref_length, len(ref_df))

    print(f"Maximum reference trend length: {max_ref_length} data points")

    # Process each timeframe for target symbols
    for timeframe in TIMEFRAMES_TO_ANALYZE:
        print(f"\nProcessing timeframe: {timeframe}")

        # Initialize results for this timeframe
        all_results[timeframe] = {}

        # Calculate seconds corresponding to current timeframe
        timeframe_seconds = calculate_timeframe_seconds(timeframe)

        # Calculate history duration to request (seconds)
        # Use longest reference sequence length * timeframe seconds * buffer ratio
        history_seconds = int(max_ref_length * timeframe_seconds * REQUEST_TIME_BUFFER_RATIO)

        # Get current time as end_ts
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - history_seconds

        print(f"Current timestamp: {end_ts}, date: {datetime.fromtimestamp(end_ts)}")
        print(f"Calculated history duration: {history_seconds} seconds ({history_seconds/86400:.1f} days)")
        print(f"Start timestamp: {start_ts}, date: {datetime.fromtimestamp(start_ts)}")

        # For crypto, pre-fetch all target data (has API request delay)
        if args.asset == "crypto":
            print(f"Getting data for all crypto symbols in timeframe {timeframe}...")
            target_data = {}

            for symbol in target_symbols:
                print(f"Getting data for {symbol} [{timeframe}]...")

                # Get data for this symbol
                df = data_processor.get_data(
                    symbol,
                    timeframe,
                    start_ts,
                    end_ts,
                    is_crypto=True
                )

                if not df.empty and len(df) > 0:
                    print(f"  Got data from {df.index[0]} to {df.index[-1]}, {len(df)} points")
                    target_data[symbol] = df

                # Sleep to avoid triggering API rate limits
                time.sleep(api_sleep_seconds)

            print(f"Successfully retrieved data for {len(target_data)} out of {len(target_symbols)} symbols")
        else:
            # For stocks, we'll fetch data on-demand in parallel processing
            target_data = None

        # Process each reference trend, comparing with current time range
        for (ref_symbol, ref_idx), ref_info in reference_data.items():
            ref_df = ref_info['df']
            ref_timeframe = ref_info['timeframe']
            label = ref_info['label']

            print(f"\nAnalyzing {ref_symbol} reference #{ref_idx} ({label}, {ref_timeframe}) against target timeframe {timeframe}:")

            # For crypto, use pre-fetched data and process in parallel
            if args.asset == "crypto":
                # Filter symbols with sufficient data points
                valid_symbols = []
                valid_dfs = []

                for symbol, df in target_data.items():
                    if len(df) >= len(ref_df) * min(DTW_WINDOW_FACTORS):
                        valid_symbols.append(symbol)
                        valid_dfs.append(df)
                        print(f"  {symbol}: data period {df.index[0]} to {df.index[-1]}, {len(df)} points")

                # Prepare multiprocessing arguments
                process_args = [
                    (symbol, df, timeframe, ref_symbol, ref_idx, ref_df, ref_timeframe, label)
                    for symbol, df in zip(valid_symbols, valid_dfs)
                ]

                # Process DTW calculations in parallel
                with Pool() as pool:
                    results = pool.map(process_symbol_dtw, process_args)

                # Process results
                target_scores = {}
                for result in results:
                    target_scores[result["symbol"]] = result
                    if result["window_data"] is not None:
                        window_start = result["window_data"].index[0]
                        window_end = result["window_data"].index[-1]
                        print(f"  {result['symbol']}: score={result['score']:.4f}, window={window_start} to {window_end}")

            else:  # For stocks, fetch data and process in one step (not implemented here)
                # This part needs additional code to handle stocks
                continue

            # Sort targets by score
            sorted_targets = sorted(
                target_scores.keys(),
                key=lambda x: target_scores[x]["score"],
                reverse=True
            )

            # Store results
            key = f"{ref_symbol}_{ref_idx}_{ref_timeframe}"
            all_results[timeframe][key] = {
                'ref_symbol': ref_symbol,
                'ref_idx': ref_idx,
                'ref_timeframe': ref_timeframe,
                'label': label,
                'targets': sorted_targets[:top_k],
                'results': target_scores
            }

            # If enabled, generate visualizations for top K symbols
            if enable_visualization:
                print(f"Generating visualizations for top {top_k} symbols...")
                # Use new folder naming format
                vis_dir = f"{output_dir}/vis_{timeframe}_{ref_symbol}_{label}"
                for symbol in sorted_targets[:top_k]:
                    result = target_scores[symbol]
                    if result["warping_path"] and result["window_data"] is not None:
                        visualize_dtw_alignment(
                            ref_df,
                            result["window_data"],
                            result["warping_path"],
                            ref_symbol,
                            symbol,
                            timeframe,
                            result["score"],
                            vis_dir,
                            label,
                        )

    # Print detailed results to console
    print("\n============= DETAILED RESULTS =============")
    print(f"DTW Similarity Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Asset Type: {args.asset}")
    print(f"DTW Window Ratio: {DTW_WINDOW_RATIO}")
    print(f"DTW Max Point Distance: {DTW_MAX_POINT_DISTANCE}")
    print(f"Price Weight: {PRICE_WEIGHT}, Diff Weight: {DIFF_WEIGHT}")
    print(f"Request Time Buffer Ratio: {REQUEST_TIME_BUFFER_RATIO}\n")

    for timeframe in TIMEFRAMES_TO_ANALYZE:
        if timeframe not in all_results:
            continue

        print(f"\n{'='*40}")
        print(f"TIMEFRAME: {timeframe}")
        print(f"{'='*40}")

        for key, results in all_results[timeframe].items():
            ref_symbol = results['ref_symbol']
            ref_idx = results['ref_idx']
            ref_timeframe = results['ref_timeframe']
            label = results['label']

            print(f"\n--- {ref_symbol} Reference #{ref_idx} ({label}, {ref_timeframe}) ---")
            print("Top Similarity Scores:")

            # Filter out results with infinite distance or score <= 0
            valid_targets = [symbol for symbol in results['targets']
                        if not np.isinf(results['results'][symbol]["distance"])
                        and results['results'][symbol]["score"] > 0]

            for symbol in valid_targets:
                score = results['results'][symbol]["score"]
                distance = results['results'][symbol]["distance"]
                window_data = results['results'][symbol]["window_data"]
                if window_data is not None:
                    window_period = f"{window_data.index[0]} to {window_data.index[-1]}"
                else:
                    window_period = "N/A"
                print(f"{symbol}: Score={score:.4f}, Distance={distance:.4f}, Window={window_period}")

    # Save detailed results to file
    detail_file = f"{output_dir}/similarity_detail.txt"
    with open(detail_file, "w") as f:
        f.write(f"DTW Similarity Analysis\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Asset Type: {args.asset}\n")
        f.write(f"DTW Window Ratio: {DTW_WINDOW_RATIO}\n")
        f.write(f"DTW Max Point Distance: {DTW_MAX_POINT_DISTANCE}\n")
        f.write(f"Price Weight: {PRICE_WEIGHT}, Diff Weight: {DIFF_WEIGHT}\n")
        f.write(f"Request Time Buffer Ratio: {REQUEST_TIME_BUFFER_RATIO}\n\n")

        for timeframe in TIMEFRAMES_TO_ANALYZE:
            if timeframe not in all_results:
                continue

            f.write(f"\n{'='*40}\n")
            f.write(f"TIMEFRAME: {timeframe}\n")
            f.write(f"{'='*40}\n\n")

            for key, results in all_results[timeframe].items():
                ref_symbol = results['ref_symbol']
                ref_idx = results['ref_idx']
                ref_timeframe = results['ref_timeframe']
                label = results['label']

                f.write(f"\n--- {ref_symbol} Reference #{ref_idx} ({label}, {ref_timeframe}) ---\n")
                f.write("Top Similarity Scores:\n")

                for symbol in results['targets']:
                    score = results['results'][symbol]["score"]
                    distance = results['results'][symbol]["distance"]
                    window_data = results['results'][symbol]["window_data"]
                    if window_data is not None:
                        window_period = f"{window_data.index[0]} to {window_data.index[-1]}"
                    else:
                        window_period = "N/A"
                    f.write(f"{symbol}: Score={score:.4f}, Distance={distance:.4f}, Window={window_period}\n")

    print(f"\nDetailed results saved to: {detail_file}")

    # Save TradingView format (excluding results with score -1)
    tv_file = f"{output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}_similar_trend_tradingview.txt"
    with open(tv_file, "w") as f:
        for timeframe in TIMEFRAMES_TO_ANALYZE:
            if timeframe not in all_results:
                continue

            for key, results in all_results[timeframe].items():
                ref_symbol = results['ref_symbol']
                ref_idx = results['ref_idx']
                ref_timeframe = results['ref_timeframe']
                label = results['label']

                f.write(f"\n###{timeframe}_{ref_symbol}_{ref_idx}_{label}\n")

                # Filter symbols with score greater than 0 (exclude -1 score results)
                valid_targets = [s for s in results['targets'] if results['results'][s]["score"] > 0]

                if args.asset == "crypto":
                    symbols_str = ','.join([f"BINANCE:{s}USDT.P" for s in valid_targets])
                else:
                    symbols_str = ','.join(valid_targets)

                f.write(symbols_str)

    print(f"TradingView format saved to: {tv_file}")

    # Calculate and output total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
