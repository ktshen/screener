"""
Trend Similarity Analyzer
-----------------------------------------------
This script identifies assets currently forming price patterns similar to historical successful 
setups using Dynamic Time Warping (DTW) and Shape Dynamic Time Warping (ShapeDTW). By analyzing price 
and moving average relationships across multiple timeframes, it discovers trading opportunities 
where the user may consider entering a position.

Key Features:
- Customizable reference trends with visual alignment reports
- TradingView-compatible output for efficient tracking
- Score system for ranking similar candidate trends (higher is better)
- Enhanced shape detection using ShapeDTW for SMA differences

Scoring Methodology:
- Price score = 1 / (1 + price_distance)
- SMA difference score = 1 / (1 + diff_distance * BALANCE_PD_RATIO)
- Overall score = (price_score * PRICE_WEIGHT + diff_score * DIFF_WEIGHT)

Usage:
python crypto_trend_screener.py [options]

Configuration:
- Set your timezone (TIMEZONE), reference patterns (REFERENCE_TRENDS), and analysis timeframes
- Define reference trends with format [start_datetime, end_datetime, timeframe, label]
- End reference patterns at ideal entry points and balance high/low point distribution

Notes:
- Stock asset type is not yet supported
- Results saved to similarity_output/[timestamp] directory
- TradingView watchlists only display the first occurrence of duplicate symbols
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from src.downloader import StockDownloader, CryptoDownloader
from src.common import (
   TrendAnalysisConfig,
   DataNormalizer,
   DTWCalculator,
   FileManager,
   ReferenceDataManager,
   BaseDataProcessor,
   calculate_timeframe_seconds,
   format_dt_with_tz,
   parse_target_symbols,
   create_output_directory,
   plot_candlesticks_with_volume
)


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
    "LQTY": [
        [datetime(2025, 5, 7, 5, 0), datetime(2025, 5, 9, 21, 0), "30m", "standard"]
    ],
    "MOODENG":[
        [datetime(2025, 5, 8, 0, 0), datetime(2025, 5, 11, 1, 0), "1h", "standard"]
    ],
}

# ================ Configuration ================
# Timezone for datetime conversion
TIMEZONE = "America/Los_Angeles"

# Output directory path

OUTPUT_DIR = "similarity_output"

# Timeframes to analyze
TIMEFRAMES_TO_ANALYZE = ["15m", "30m", "1h", "2h", "4h"]

# Top K symbols to record per reference trend
TOP_K = 10

# DTW window constraint
DTW_WINDOW_RATIO = 0.2  # Range: 0.0 to 1.0
DTW_WINDOW_RATIO_FOR_DIFF = 0.1  # Range: 0.0 to 1.0

# Maximum distance limit for DTW matching points
DTW_MAX_POINT_DISTANCE = 0.66   # Range: 0.0 to 2.0
DTW_MAX_POINT_DISTANCE_FOR_DIFF = 0.5  # Range: 0.0 to 2.0

# Weights for price and difference features in Shape DTW calculation
SHAPEDTW_BALANCE_PD_RATIO = 4  # Ratio of price distance to SMA difference distance
PRICE_WEIGHT = 0.4  
DIFF_WEIGHT = 0.6   

# Shape descriptor parameters for SMA differences
SLOPE_WINDOW_SIZE = 5  # Window size for slope descriptor
PAA_WINDOW_SIZE = 5    # Window size for PAA descriptor

# Constants for similarity calculation
SMA_PERIODS = [30, 45, 60]
DTW_WINDOW_FACTORS = [0.9, 0.95, 1.0, 1.05, 1.1]  # Window size factors for DTW
MIN_QUERY_LENGTH = 60  # Minimum number of data points required for query trend

# BINANCE API request interval parameter
API_SLEEP_SECONDS = 0.5 # Sleep time between requests (seconds)

# Request buffer ratio
REQUEST_TIME_BUFFER_RATIO = 1.2


# ================ Data Processing Classes ================

class DataProcessor(BaseDataProcessor):
    """Data processor for both stocks and cryptocurrencies"""
    
    def __init__(self, asset_type: str, config: TrendAnalysisConfig = None):
        """Initialize appropriate downloader based on asset type"""
        super().__init__(asset_type, config.sma_periods if config else None)
        
        if asset_type == "crypto":
            self.downloader = CryptoDownloader()
        else:
            self.downloader = StockDownloader(save_dir=".", api_file="api_keys.json")
        
        self.config = config or TrendAnalysisConfig()

    def get_data(self, symbol: str, timeframe: str, start_ts: int, end_ts: int,
                is_crypto: bool = True, include_buffer: bool = True, 
                is_reference: bool = False) -> pd.DataFrame:
        """Get data with buffer period for SMA calculation"""
        if include_buffer:
            # Calculate buffer period for SMA calculation
            interval = end_ts - start_ts
            buffer_start_ts = start_ts - interval
        else:
            buffer_start_ts = start_ts

        if is_crypto is None:
            is_crypto = (self.asset_type == "crypto")

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

        # Filter to requested time range
        start_time = pd.Timestamp.fromtimestamp(start_ts)
        end_time = pd.Timestamp.fromtimestamp(end_ts)
        
        # Use the processor from common to prepare the dataframe
        df = self.processor.prepare_dataframe(df)
        
        # Filter to requested time range after preparation
        df = df[(df.index >= start_time) & (df.index <= end_time)]

        return df


# ================ DTW Similarity Calculator ================

class DTWSimilarityCalculator:
    """Calculate similarity using DTW and ShapeDTW algorithms"""
    
    def __init__(self, config: TrendAnalysisConfig):
        """Initialize DTW calculator with configuration"""
        self.config = config
        self.dtw_calc = DTWCalculator(config)

    def find_best_similarity_window(self, query_df: pd.DataFrame, target_df: pd.DataFrame) -> dict:
        """Find best similarity window based on price and difference features"""
        query_len = len(query_df)
    
        # Check if target sequence is long enough
        if len(target_df) < query_len * min(self.config.window_scale_factors):
            return {
                "similarity": 0.0,
                "price_distance": float('inf'),
                "diff_distance": float('inf'),
                "price_path": None,
                "diff_path": None,
                "window_data": None,
                "window_info": None
            }
        
        best_similarity = -1
        best_price_distance = float('inf')
        best_diff_distance = float('inf')
        best_price_path = None
        best_diff_path = None
        best_window_data = None
        best_window_info = None
        
        # Pre-process reference sequence features
        query_price_norm, query_diff_norm = self.dtw_calc.normalize_features(query_df)
        
        # Define shape descriptors
        price_descriptor, diff_descriptor = self.dtw_calc.create_shape_descriptors()
        
        # Try different window sizes, but fix right boundary at the last time point of target sequence
        for factor in self.config.window_scale_factors:
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
            window_price_norm, window_diff_norm = self.dtw_calc.normalize_features(window)
            
            # Calculate DTW for price features (using dtaidistance) - for initial screening
            _, dtw_price_distance, _ = self.dtw_calc.calculate_dtw_similarity(
                query_price_norm, window_price_norm, self.config.dtw_window_ratio, self.config.dtw_max_point_distance
            )
            
            # If no valid path found for price features (distance is inf), continue to next factor
            if np.isinf(dtw_price_distance):
                print(f"  Factor {factor}: No valid price path found due to max_step constraint")
                continue
            
            # Calculate DTW for SMA difference features (using dtaidistance) - for initial screening
            _, dtw_diff_distance, _ = self.dtw_calc.calculate_dtw_similarity(
                query_diff_norm, window_diff_norm, self.config.dtw_window_ratio_diff, self.config.dtw_max_point_distance_diff
            )
            
            # If no valid path found for SMA difference features (distance is inf), continue to next factor
            if np.isinf(dtw_diff_distance):
                print(f"  Factor {factor}: No valid SMA diff path found due to max_step constraint")
                continue
            
            # Use dynamic subsequence width based on window factor
            subsequence_width = max(2, min(5, int(factor * 3)))
            
            # Calculate ShapeDTW for price features
            price_shape_dist, price_shape_path = self.dtw_calc.calculate_shapedtw(
                query_price_norm, window_price_norm, price_descriptor, self.config.dtw_window_ratio, subsequence_width
            )
            
            # If no valid path found for price features, continue to next factor
            if np.isinf(price_shape_dist):
                print(f"  Factor {factor}: No valid shape path found for price features")
                continue
            
            # Calculate ShapeDTW for difference features
            diff_shape_dist, diff_shape_path = self.dtw_calc.calculate_shapedtw(
                query_diff_norm, window_diff_norm, diff_descriptor, self.config.dtw_window_ratio_diff, subsequence_width
            )
            
            # If no valid path found for difference features, continue to next factor
            if np.isinf(diff_shape_dist):
                print(f"  Factor {factor}: No valid shape path found for diff features")
                continue
            
            # Calculate overall similarity using arithmetic mean
            price_score = 1 / (1 + price_shape_dist)
            sma_diff_score = 1 / (1 + diff_shape_dist * self.config.shapedtw_balance_pd_ratio)
            similarity = (price_score * self.config.price_weight) + (sma_diff_score * self.config.diff_weight)
                
            # If similarity is higher, update best results
            if similarity > best_similarity:
                best_similarity = similarity
                best_price_distance = price_shape_dist
                best_diff_distance = diff_shape_dist
                best_price_path = price_shape_path
                best_diff_path = diff_shape_path
                best_window_data = window
                best_window_info = (factor, start_idx, len(target_df))
                
            print(f"  Factor {factor}: similarity={similarity:.4f}, price_shape_dist={price_shape_dist:.4f}, "
                f"diff_shape_dist={diff_shape_dist:.4f}, window={start_idx}:{len(target_df)}, "
                f"period={window.index[0]} to {window.index[-1]}")
        
        if best_window_data is not None:
            print(f"  Best window: {best_window_data.index[0]} to {best_window_data.index[-1]}")
        
        # Return all relevant information as a dictionary
        return {
            "similarity": best_similarity,
            "price_distance": best_price_distance,
            "diff_distance": best_diff_distance,
            "price_path": best_price_path,
            "diff_path": best_diff_path,
            "window_data": best_window_data,
            "window_info": best_window_info
        }


# ================ Utility Functions ================

def process_symbol_dtw(args: tuple) -> dict:
    """Process a single symbol DTW comparison (designed for multiprocessing)"""
    target_symbol, target_df, timeframe, ref_symbol, ref_idx, ref_df, ref_timeframe, ref_label, config = args
    
    print(f"Processing DTW for {target_symbol} [{timeframe}] against {ref_symbol} reference #{ref_idx} ({ref_label}) [{ref_timeframe}]...")
    
    # Calculate similarity
    dtw_calculator = DTWSimilarityCalculator(config)
    similarity_result = dtw_calculator.find_best_similarity_window(
        ref_df, target_df
    )
    
    return {
        "symbol": target_symbol,
        "timeframe": timeframe,
        "ref_symbol": ref_symbol,
        "ref_idx": ref_idx,
        "ref_timeframe": ref_timeframe,
        "ref_label": ref_label,
        "score": similarity_result["similarity"],
        "price_distance": similarity_result["price_distance"],
        "diff_distance": similarity_result["diff_distance"],
        "price_path": similarity_result["price_path"],
        "diff_path": similarity_result["diff_path"],
        "window_data": similarity_result["window_data"],
        "window_info": similarity_result["window_info"]
    }


# ================ Visualization Functions ================

def visualize_dtw_alignment(query_df, window_df, price_path, ref_symbol, target_symbol, 
                         timeframe, similarity, save_dir, ref_label, price_distance, diff_distance):
    """Visualize DTW alignment using candlestick charts with volume and connection lines"""
    if not price_path:
        print(f"No warping path available for visualization of {target_symbol}")
        return
    
    # Create figure with normalized data for plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), sharex=False, gridspec_kw={'height_ratios': [1, 1]})
    
    ref_normalized_df, _ = DataNormalizer.normalize_ohlc_dataframe(query_df, include_volume=True)
    target_normalized_df, _ = DataNormalizer.normalize_ohlc_dataframe(window_df, include_volume=True)
    
    # Plot reference sequence with candlesticks and volume
    plot_candlesticks_with_volume(ax1, ref_normalized_df, volume_ratio=0.12)
    ax1.plot(ref_normalized_df.index, ref_normalized_df['SMA_30'], 'blue', linewidth=1, alpha=0.7, label='SMA30')
    ax1.plot(ref_normalized_df.index, ref_normalized_df['SMA_45'], 'orange', linewidth=1, alpha=0.7, label='SMA45')
    ax1.plot(ref_normalized_df.index, ref_normalized_df['SMA_60'], 'purple', linewidth=1, alpha=0.7, label='SMA60')
    ax1.set_ylabel('Normalized Price')
    ax1.set_title(f'{ref_symbol} Reference Trend - {ref_label} (Length: {len(query_df)} points)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot target sequence with candlesticks and volume
    plot_candlesticks_with_volume(ax2, target_normalized_df)
    ax2.plot(target_normalized_df.index, target_normalized_df['SMA_30'], 'blue', linewidth=1, alpha=0.7, label='SMA30')
    ax2.plot(target_normalized_df.index, target_normalized_df['SMA_45'], 'orange', linewidth=1, alpha=0.7, label='SMA45')
    ax2.plot(target_normalized_df.index, target_normalized_df['SMA_60'], 'purple', linewidth=1, alpha=0.7, label='SMA60')
    ax2.set_ylabel('Normalized Price')
    ax2.set_title(f'{target_symbol} {timeframe} Trend (Length: {len(window_df)} points, Latest: {format_dt_with_tz(window_df.index[-1], TIMEZONE)})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    price_cols = ['Open', 'High', 'Low', 'Close']
    ref_price_min = ref_normalized_df[price_cols].values.min()
    ref_price_max = ref_normalized_df[price_cols].values.max()
    target_price_min = target_normalized_df[price_cols].values.min()
    target_price_max = target_normalized_df[price_cols].values.max()
    
    step_size = max(1, len(price_path) // 100)
    
    for idx, (i, j) in enumerate(price_path):
        if idx % step_size == 0:
            ref_x = query_df.index[i]
            ref_y = ref_normalized_df['Close'].iloc[i]
            target_x = window_df.index[j]
            target_y = target_normalized_df['Close'].iloc[j]
            
            if (ref_price_min <= ref_y <= ref_price_max and 
                target_price_min <= target_y <= target_price_max):
                
                con = ConnectionPatch(
                    xyA=(mdates.date2num(ref_x), ref_y), coordsA=ax1.transData,
                    xyB=(mdates.date2num(target_x), target_y), coordsB=ax2.transData,
                    color='gray', alpha=0.4, linewidth=0.7, linestyle='-',
                    zorder=1
                )
                
                fig.add_artist(con)
    
    # Add title with parameters and metrics information
    plt.suptitle(f'Price/SMA Alignment with Volume\n {ref_symbol}({ref_label}) vs {target_symbol} ({timeframe})\n'
                f'Score: {similarity:.4f}, Price Distance: {price_distance:.4f}, SMA Diff Distance: {diff_distance:.4f}',
                fontsize=12)

    # Add info text
    info_text = (f"Reference Period: {format_dt_with_tz(query_df.index[0], TIMEZONE)} to {format_dt_with_tz(query_df.index[-1], TIMEZONE)}\n"
            f"Target Period: {format_dt_with_tz(window_df.index[0], TIMEZONE)} to {format_dt_with_tz(window_df.index[-1], TIMEZONE)}")
                
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Ensure layout is compact
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    
    # Save image
    FileManager.ensure_directories(save_dir)
    filename = f"score_{similarity:.4f}_{target_symbol}_{timeframe}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization to {filepath}")


def visualize_sma_differences(query_df, window_df, diff_path, ref_symbol, target_symbol, 
                             timeframe, similarity, save_dir, ref_label, price_distance, diff_distance):
    """Visualize SMA differences between reference and target sequences (no volume)"""
    if not diff_path:
        print(f"No SMA diff warping path available for visualization of {target_symbol}")
        return
        
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), sharex=False, gridspec_kw={'height_ratios': [1, 1]})
    
    # Define SMA difference columns
    diff_cols = ['SMA30_SMA45', 'SMA30_SMA60', 'SMA45_SMA60']
    
    # Extract and normalize reference SMA differences for plotting
    ref_diffs = query_df[diff_cols].values
    ref_normalized = DataNormalizer.normalize_to_range(ref_diffs)
    
    # Reshape back to format needed for plotting
    ref_normalized_reshaped = ref_normalized.reshape(query_df[diff_cols].shape)
    ref_normalized_df = pd.DataFrame(
        ref_normalized_reshaped, 
        index=query_df.index, 
        columns=diff_cols
    )
    
    # Extract and normalize target SMA differences for plotting  
    target_diffs = window_df[diff_cols].values
    target_normalized = DataNormalizer.normalize_to_range(target_diffs)
    
    # Reshape back to format needed for plotting
    target_normalized_reshaped = target_normalized.reshape(window_df[diff_cols].shape)
    target_normalized_df = pd.DataFrame(
        target_normalized_reshaped, 
        index=window_df.index, 
        columns=diff_cols
    )
    
    # Define line styles and colors for better visual distinction
    line_styles = [
        {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'label': 'SMA30-SMA45'},
        {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'label': 'SMA30-SMA60'},
        {'color': 'orange', 'linestyle': '-', 'linewidth': 2, 'label': 'SMA45-SMA60'}
    ]
    
    # Plot reference sequence differences
    for i, col in enumerate(diff_cols):
        ax1.plot(ref_normalized_df.index, ref_normalized_df[col], **line_styles[i])
    ax1.set_ylabel('Normalized SMA Differences')
    ax1.set_title(f'{ref_symbol} Reference SMA Diff - {ref_label} (Length: {len(query_df)} points)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot target sequence differences
    for i, col in enumerate(diff_cols):
        ax2.plot(target_normalized_df.index, target_normalized_df[col], **line_styles[i])
    ax2.set_ylabel('Normalized SMA Differences')
    ax2.set_title(f'{target_symbol} {timeframe} SMA Diff (Length: {len(window_df)} points, Latest: {window_df.index[-1].strftime("%Y-%m-%d")})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set y-axis range for each chart with padding for aesthetics
    ref_min = ref_normalized_df.values.min()
    ref_max = ref_normalized_df.values.max()
    ref_padding = (ref_max - ref_min) * 0.1
    ax1.set_ylim(ref_min - ref_padding, ref_max + ref_padding)
    
    target_min = target_normalized_df.values.min()
    target_max = target_normalized_df.values.max()
    target_padding = (target_max - target_min) * 0.1
    ax2.set_ylim(target_min - target_padding, target_max + ref_padding)
    
    # Add warping path visualization
    # Determine number of connection points (reduce density of connections)
    step_size = max(1, len(diff_path) // 100)
    
    # Add markers at connection points - focus on SMA30-SMA45 as the primary line
    connected_ref_indices = [i for i, _ in diff_path[::step_size]]
    connected_target_indices = [j for _, j in diff_path[::step_size]]
    
    # Place small markers at connection points on SMA30-SMA45 line
    ax1.scatter(query_df.index[connected_ref_indices], 
            ref_normalized_df['SMA30_SMA45'].iloc[connected_ref_indices], 
            color='darkgreen', s=15, alpha=0.6, zorder=5)
    ax2.scatter(window_df.index[connected_target_indices], 
            target_normalized_df['SMA30_SMA45'].iloc[connected_target_indices], 
            color='darkgreen', s=15, alpha=0.6, zorder=5)
    
    # Draw connecting lines using ConnectionPatch
    for idx, (i, j) in enumerate(diff_path):
        if idx % step_size == 0:
            ref_x = query_df.index[i]
            ref_y = ref_normalized_df['SMA30_SMA45'].iloc[i]
            target_x = window_df.index[j]
            target_y = target_normalized_df['SMA30_SMA45'].iloc[j]
            
            # Create a connection patch between the two points
            con = ConnectionPatch(
                xyA=(mdates.date2num(ref_x), ref_y), coordsA=ax1.transData,
                xyB=(mdates.date2num(target_x), target_y), coordsB=ax2.transData,
                color='gray', alpha=0.4, linewidth=0.7, linestyle='-',
                zorder=1
            )
            
            fig.add_artist(con)
    
    # Add title with parameters and metrics information
    plt.suptitle(f'SMA Differences\n {ref_symbol}({ref_label}) vs {target_symbol} ({timeframe})\n'
                f'Score: {similarity:.4f}, Price Distance: {price_distance:.4f}, SMA Diff Distance: {diff_distance:.4f}',
                fontsize=12)
    
    # Add info text
    info_text = (f"Reference Period: {format_dt_with_tz(query_df.index[0], TIMEZONE)} to {format_dt_with_tz(query_df.index[-1], TIMEZONE)}\n"
            f"Target Period: {format_dt_with_tz(window_df.index[0], TIMEZONE)} to {format_dt_with_tz(window_df.index[-1], TIMEZONE)}")
                
    # Add text box
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Ensure layout is compact
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    
    # Save image with modified filename including the diff_distance
    FileManager.ensure_directories(save_dir)
    filename = f"diff_distance_{diff_distance:.4f}_{target_symbol}_{timeframe}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved SMA differences visualization to {filepath}")


# ================ Main Function ================

def main():
    """Main function to run the DTW similarity analysis"""
    # Record start time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Analyze trend similarity for crypto or stock symbols')
    parser.add_argument('-f', '--file', required=False, default="", help='Path to strong target file')
    parser.add_argument('--asset', choices=['crypto', 'stock'], default='crypto', help='Asset type of input file (default: crypto)')
    parser.add_argument('-nv', '--no_visualize', action='store_true', default=False, help='Enable visualization of DTW alignments')
    parser.add_argument('-k', '--topk', type=int, default=TOP_K, help=f'Number of top symbols to record per reference trend (default: {TOP_K})')
    parser.add_argument('-s', '--sleep', type=float, default=API_SLEEP_SECONDS, help=f'Sleep time between API requests in seconds (default: {API_SLEEP_SECONDS})')
    args = parser.parse_args()
    
    # Create configuration from script constants
    config = TrendAnalysisConfig()
    config.sma_periods = SMA_PERIODS
    config.dtw_window_ratio = DTW_WINDOW_RATIO
    config.dtw_window_ratio_diff = DTW_WINDOW_RATIO_FOR_DIFF
    config.dtw_max_point_distance = DTW_MAX_POINT_DISTANCE
    config.dtw_max_point_distance_diff = DTW_MAX_POINT_DISTANCE_FOR_DIFF
    config.shapedtw_balance_pd_ratio = SHAPEDTW_BALANCE_PD_RATIO
    config.price_weight = PRICE_WEIGHT
    config.diff_weight = DIFF_WEIGHT
    config.slope_window_size = SLOPE_WINDOW_SIZE
    config.paa_window_size = PAA_WINDOW_SIZE
    config.window_scale_factors = DTW_WINDOW_FACTORS
    config.min_query_length = MIN_QUERY_LENGTH
    config.api_sleep_seconds = args.sleep
    config.request_time_buffer_ratio = REQUEST_TIME_BUFFER_RATIO
    
    # Use parameters
    enable_visualization = not args.no_visualize
    top_k = args.topk
    api_sleep_seconds = args.sleep
    
    print(f"\nAnalysis Configuration:")
    print(f"Input Asset Type: {args.asset}")
    print(f"Reference Trends: {len(REFERENCE_TRENDS)} symbols with {sum(len(trends) for trends in REFERENCE_TRENDS.values())} total trends")
    print(f"Timeframes to Analyze: {TIMEFRAMES_TO_ANALYZE}")
    print(f"DTW Window Ratio: {config.dtw_window_ratio}")
    print(f"DTW Window Ratio for DIFF: {config.dtw_window_ratio_diff}")
    print(f"DTW Max Point Distance: {config.dtw_max_point_distance}")
    print(f"DTW Max Point Distance for DIFF: {config.dtw_max_point_distance_diff}")
    print(f"Slope Window Size: {config.slope_window_size}")
    print(f"PAA Window Size: {config.paa_window_size}")
    print(f"Price Weight: {config.price_weight}, Diff Weight: {config.diff_weight}")
    print(f"Using ShapeDTW for both price and difference features")
    print(f"Request Time Buffer Ratio: {config.request_time_buffer_ratio}")
    print(f"API Sleep Time: {api_sleep_seconds} seconds")
    print(f"Visualization Enabled: {enable_visualization}")
    print(f"Top K Symbols: {top_k}\n")
    
    # Initialize data processor
    data_processor = DataProcessor(args.asset, config)
    
    # Get target symbols - either from file or all available
    if args.file:
        target_symbols = parse_target_symbols(args.file)
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
    output_dir = create_output_directory(OUTPUT_DIR)
    
    # Store all results
    all_results = {}
    
    # Load or retrieve all reference trend data using unified manager
    reference_data = {}
    for ref_symbol, ref_trends in REFERENCE_TRENDS.items():
        for ref_idx, ref_trend_info in enumerate(ref_trends):
            start_datetime, end_datetime, ref_timeframe, ref_label = ref_trend_info
            
            ref_df = ReferenceDataManager.load_or_fetch_reference_data(
                ref_symbol, start_datetime, end_datetime, ref_timeframe, ref_label,
                OUTPUT_DIR, TIMEZONE, data_processor, config
            )
            
            if ref_df is not None:
                reference_data[(ref_symbol, ref_idx)] = {
                    'df': ref_df,
                    'timeframe': ref_timeframe,
                    'label': ref_label
                }
    
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
        # Use longest reference sequence length * timeframe seconds * max_window_factor * buffer ratio
        history_seconds = int(max_ref_length * timeframe_seconds * max(config.window_scale_factors) * config.request_time_buffer_ratio)
        
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
                    if len(df) >= len(ref_df) * min(config.window_scale_factors):
                        valid_symbols.append(symbol)
                        valid_dfs.append(df)
                        print(f"  {symbol}: data period {df.index[0]} to {df.index[-1]}, {len(df)} points")
                
                # Prepare multiprocessing arguments
                process_args = [
                    (symbol, df, timeframe, ref_symbol, ref_idx, ref_df, ref_timeframe, label, config)
                    for symbol, df in zip(valid_symbols, valid_dfs)
                ]
                
                # Process DTW calculations in parallel
                with Pool(processes=min(cpu_count()-1, len(valid_symbols))) if len(valid_symbols) > 1 else Pool(processes=1) as pool:
                    results = pool.map(process_symbol_dtw, process_args)
                
                # Process results
                target_scores = {}
                for result in results:
                    target_scores[result["symbol"]] = result
                    if result["window_data"] is not None:
                        window_start = result["window_data"].index[0]
                        window_end = result["window_data"].index[-1]
                        print(f"  {result['symbol']}: score={result['score']:.4f}, " 
                            f"price_distance={result['price_distance']:.4f}, "
                            f"diff_distance={result['diff_distance']:.4f}, "
                            f"window={window_start} to {window_end}")
                        
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
                # Use updated folder naming format with reference timeframe
                vis_dir = f"{output_dir}/vis_{timeframe}_{ref_symbol}_{ref_timeframe}_{label}"
                for symbol in sorted_targets[:top_k]:
                    result = target_scores[symbol]
                    if result["price_path"] and result["window_data"] is not None:
                        visualize_dtw_alignment(
                            ref_df,
                            result["window_data"],
                            result["price_path"],
                            ref_symbol,
                            symbol,
                            timeframe,
                            result["score"],
                            vis_dir,
                            label,
                            result["price_distance"],
                            result["diff_distance"]
                        )

                        visualize_sma_differences(
                            ref_df,
                            result["window_data"],
                            result["diff_path"],
                            ref_symbol,
                            symbol,
                            timeframe,
                            result["score"],
                            vis_dir,
                            label,
                            result["price_distance"],
                            result["diff_distance"]
                        )
    
        # Create a string list for the detailed results (optimization)
        summary = []
        summary.append("\n============= DETAILED RESULTS =============")
        summary.append(f"DTW Similarity Analysis")
        summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        summary.append(f"Asset Type: {args.asset}")
        summary.append(f"DTW Window Ratio: {config.dtw_window_ratio}")
        summary.append(f"DTW Window Ratio for DIFF: {config.dtw_window_ratio_diff}")
        summary.append(f"DTW Max Point Distance: {config.dtw_max_point_distance}")
        summary.append(f"DTW Max Point Distance for DIFF: {config.dtw_max_point_distance_diff}")
        summary.append(f"Slope Window Size: {config.slope_window_size}")
        summary.append(f"PAA Window Size: {config.paa_window_size}")
        summary.append(f"Price Weight: {config.price_weight}, Diff Weight: {config.diff_weight}")
        summary.append(f"Scoring Method: Arithmetic mean of price and SMA difference similarities")
        summary.append(f"ShapeDTW enabled for both price and SMA difference features")
        summary.append(f"Request Time Buffer Ratio: {config.request_time_buffer_ratio}\n")

        for timeframe in TIMEFRAMES_TO_ANALYZE:
            if timeframe not in all_results:
                continue
                
            summary.append(f"\n{'='*40}")
            summary.append(f"TIMEFRAME: {timeframe}")
            summary.append(f"{'='*40}")
            
            for key, results in all_results[timeframe].items():
                ref_symbol = results['ref_symbol']
                ref_idx = results['ref_idx']
                ref_timeframe = results['ref_timeframe']
                label = results['label']
                
                summary.append(f"\n--- {ref_symbol} Reference #{ref_idx} ({label}, {ref_timeframe}) ---")
                
                # Filter out results with infinite distance or score <= 0
                valid_targets = [symbol for symbol in results['targets'] 
                            if not np.isinf(results['results'][symbol]["price_distance"]) 
                            and not np.isinf(results['results'][symbol]["diff_distance"])
                            and results['results'][symbol]["score"] > 0]
                
                if len(valid_targets) > 0:
                    summary.append("Top Similarity Scores:")
                
                for symbol in valid_targets:
                    score = results['results'][symbol]["score"]
                    price_distance = results['results'][symbol]["price_distance"]
                    diff_distance = results['results'][symbol]["diff_distance"]
                    window_data = results['results'][symbol]["window_data"]
                    if window_data is not None:
                        window_period = f"{format_dt_with_tz(window_data.index[0], TIMEZONE)} to {format_dt_with_tz(window_data.index[-1], TIMEZONE)}"
                    else:
                        window_period = "N/A"
                    summary.append(f"{symbol}: Score={score:.4f}, Price Dist={price_distance:.4f}, SMA Diff Dist={diff_distance:.4f}, Window={window_period}")

        # Join all summary lines into a single string
        summary_text = '\n'.join(summary)
        
        # Print to console
        print(summary_text)
        
        # Save detailed results to file
        detail_file = f"{output_dir}/similarity_search_report.txt"
        with open(detail_file, "w") as f:
            f.write(summary_text)
        
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