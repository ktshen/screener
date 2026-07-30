"""
Stock Similar Pattern Finder using Dynamic Time Warping (DTW)
==============================================================
Memory-efficient version that processes symbols one by one with multiple matches per symbol.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from src.downloader import StockDownloader
from src.common import (
    TrendAnalysisConfig,
    DataNormalizer,
    DTWCalculator,
    FileManager,
    ReferenceDataManager,
    BaseDataProcessor,
    create_output_directory,
    filter_non_overlapping_results,
    format_dt_with_tz,
    convert_datetime_to_timestamp,
)

# ================ Configuration ================
REFERENCE_TRENDS = {
    "PAHC": [
        [datetime(2025, 4, 21, 9, 0), datetime(2025, 11, 6, 13, 0), "2h", "to_horizontal"],
    ],
    "AS": [
        [datetime(2024, 7, 3, 9, 0), datetime(2024, 9, 12, 13, 0), "2h", "correction_A"],
        [datetime(2024, 9, 12, 9, 0), datetime(2024, 11, 6, 13, 0), "2h", "correction_B"],
    ],

}

DEFAULT_START_DATE = datetime(2020, 1, 1)
TIMEZONE = "America/New_York"
TIMEFRAMES_TO_ANALYZE = ["2h"]
OUTPUT_DIR = "stock_trend_finder_reports"
TOP_K = 300

# Global filtering: True = no overlaps across all symbols, False = allow overlaps between symbols
GLOBAL_OVERLAP_FILTERING = False

# ================ DTW Parameters ================
# Lower values = stricter matching, higher values = more flexible matching

# Controls how much warping is allowed in time alignment
DTW_WINDOW_RATIO = 0.2        
# Controls how different normalized price values can be to still match            
DTW_MAX_POINT_DISTANCE = 0.5       
# DTW window ratio for SMA difference features   
# Usually smaller than price window ratio since differences are more sensitive     
DTW_WINDOW_RATIO_FOR_DIFF = 0.2       
# Maximum point distance for SMA difference DTW
# Controls similarity requirement for SMA relationships
DTW_MAX_POINT_DISTANCE_FOR_DIFF = 0.3

# ================ ShapeDTW Parameters ================
# Balance factor between price and difference features in ShapeDTW
SHAPEDTW_BALANCE_PD_RATIO = 4               
# Weight given to price features in final similarity calculation (0.0 to 1.0)
PRICE_WEIGHT = 0.4                         
# Weight given to SMA difference features (should sum with PRICE_WEIGHT to 1.0)
DIFF_WEIGHT = 0.6                          
# Window size for slope descriptor calculation
# Controls granularity of trend direction analysis
# Smaller values (3-5) = capture short-term slope changes
# Larger values (7-10) = focus on longer-term trend directions
SLOPE_WINDOW_SIZE = 5 
# Window size for Piecewise Aggregate Approximation
# Controls data compression level for pattern comparison
# Smaller values (3-5) = preserve more detail in pattern matching
# Larger values (7-10) = more generalized pattern matching                ÃŸ     
PAA_WINDOW_SIZE = 5                         

# ================ Search Parameters ================
WINDOW_SCALE_FACTORS = [0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1]   # Pattern size variations
SMA_PERIODS = [30, 45, 60]                           # Moving average periods
SLIDING_WINDOW_STEP_RATIO = 0.25                     # Search step size 
MIN_SIMILARITY_SCORE = 0.25                          # Minimum match threshold 

# ================ Analysis Parameters ================
VIS_EXTENSION_PAST_LENGTH_FACTOR = 1.0      # Past context length 
VIS_EXTENSION_FUTURE_LENGTH_FACTOR = 2.0    # Future analysis length 
EXTENSION_FACTORS_FOR_STATS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]

# Maximum non-overlapping matches per symbol 
MAX_MATCHES_PER_SYMBOL = 5

VALID_SYMBOLS_CACHE_FILE = "valid_stock_symbols.json"

# ================ Symbol Validation ================

def validate_single_symbol(args: tuple) -> tuple:
    """Validate a single stock symbol"""
    symbol, start_ts, end_ts, timeframe = args
    
    try:
        downloader = StockDownloader()
        success, df = downloader.get_data(
            symbol, start_ts, end_ts, timeframe=timeframe, 
            dropna=True, atr=False, validate=True
        )
        
        if success and not df.empty and len(df) >= 50:
            print(f"âœ“ {symbol}: Valid ({len(df)} data points)")
            return symbol, True
        else:
            return symbol, False
            
    except Exception as e:
        print(f"âœ— {symbol}: Error - {e}")
        return symbol, False


def get_valid_symbols(start_date: datetime, timeframe: str = "1d", force_refresh: bool = False) -> list:
    """Get valid stock symbols with parallel validation and caching"""
    cache_file = os.path.join(OUTPUT_DIR, VALID_SYMBOLS_CACHE_FILE)
    
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                valid_symbols = cached_data.get('valid_symbols', [])
                
            if valid_symbols:
                print(f"Loaded {len(valid_symbols)} valid symbols from cache")
                return valid_symbols
        except Exception as e:
            print(f"Error loading cached symbols: {e}")
    
    print("Validating stock symbols in parallel...")
    
    downloader = StockDownloader()
    all_symbols = downloader.get_all_tickers()
    print(f"Found {len(all_symbols)} total symbols to validate")
    
    start_ts = convert_datetime_to_timestamp(start_date, TIMEZONE)
    end_ts = int(time.time())
    
    validation_args = [(symbol, start_ts, end_ts, timeframe) for symbol in all_symbols]
    max_workers = cpu_count() - 1
    
    valid_symbols = []
    with Pool(processes=max_workers) as pool:
        results = pool.map(validate_single_symbol, validation_args)
    
    for symbol, is_valid in results:
        if is_valid:
            valid_symbols.append(symbol)
    
    print(f"Validation complete: {len(valid_symbols)} valid symbols")
    
    FileManager.ensure_directories(OUTPUT_DIR)
    cache_data = {
        'valid_symbols': sorted(valid_symbols),
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    return valid_symbols


# ================ Helper Functions ================

def check_time_overlap(period1: tuple, period2: tuple) -> bool:
    """Check if two time periods overlap"""
    start1, end1 = period1
    start2, end2 = period2
    return (start1 <= end2) and (start2 <= end1)


def filter_overlapping_matches(matches: list, max_matches: int) -> list:
    """Filter matches to remove overlaps within the same symbol"""
    if not matches:
        return []
    
    # Sort by similarity descending
    sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    selected_matches = []
    
    for match in sorted_matches:
        if len(selected_matches) >= max_matches:
            break
            
        window_data = match.get('window_data')
        if window_data is None or window_data.empty:
            continue
        
        current_period = (window_data.index[0], window_data.index[-1])
        
        # Check for overlap with already selected matches
        has_overlap = False
        for selected_match in selected_matches:
            selected_window = selected_match.get('window_data')
            if selected_window is None or selected_window.empty:
                continue
            
            selected_period = (selected_window.index[0], selected_window.index[-1])
            
            if check_time_overlap(current_period, selected_period):
                has_overlap = True
                break
        
        # If no overlap, add to selected
        if not has_overlap:
            selected_matches.append(match)
    
    return selected_matches


# ================ Core Processing Function ================

def process_single_symbol_against_reference(args: tuple) -> dict:
    """Process a single symbol against a reference trend - returns multiple matches"""
    symbol, timeframe, start_ts, end_ts, reference_df_data, reference_symbol, reference_timeframe, reference_label, config_dict = args
    
    try:
        # Create new instances in worker process
        downloader = StockDownloader()
        
        # Reconstruct config
        config = TrendAnalysisConfig()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        dtw_calc = DTWCalculator(config)
        
        # Reconstruct reference DataFrame
        reference_df = pd.DataFrame(reference_df_data['data'], index=pd.to_datetime(reference_df_data['index']))
        
        # Format target symbol date range for printing with time
        start_datetime = pd.Timestamp.fromtimestamp(start_ts, tz='UTC').tz_convert(TIMEZONE)
        end_datetime = pd.Timestamp.fromtimestamp(end_ts, tz='UTC').tz_convert(TIMEZONE)
        target_date_range = f"{start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')}"
        
        print(f"Processing {symbol} ({timeframe}, {target_date_range}) against {reference_symbol} ({reference_timeframe}, {reference_label})...")
        
        # Download data for this symbol
        interval = end_ts - start_ts
        buffer_start_ts = start_ts - interval
        
        success, df = downloader.get_data(
            symbol, buffer_start_ts, end_ts, timeframe=timeframe,
            dropna=True, atr=False, validate=True
        )
        
        if not success or df is None or df.empty:
            print(f"No data for {symbol}")
            return {"symbol": symbol, "results": []}
        
        # Process the data - ensure timezone consistency
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(TIMEZONE)
        df = df.set_index('datetime')
        
        # Standardize column names (SMA already calculated by downloader)
        column_mapping = {
            'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low',
            'sma_30': 'SMA_30', 'sma_45': 'SMA_45', 'sma_60': 'SMA_60', 'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Calculate difference features using the SMAs that are already available
        if 'SMA_30' in df.columns and 'SMA_45' in df.columns:
            df['SMA30_SMA45'] = df['SMA_30'] - df['SMA_45']
        if 'SMA_30' in df.columns and 'SMA_60' in df.columns:
            df['SMA30_SMA60'] = df['SMA_30'] - df['SMA_60']
        if 'SMA_45' in df.columns and 'SMA_60' in df.columns:
            df['SMA45_SMA60'] = df['SMA_45'] - df['SMA_60']
            
        # Calculate price-SMA differences
        if 'SMA_30' in df.columns:
            df['Close_SMA30'] = df['Close'] - df['SMA_30']
        if 'SMA_45' in df.columns:
            df['Close_SMA45'] = df['Close'] - df['SMA_45']
        if 'SMA_60' in df.columns:
            df['Close_SMA60'] = df['Close'] - df['SMA_60']
        
        # Drop NaN and filter to time range
        df = df.dropna()
        start_time = pd.Timestamp.fromtimestamp(start_ts, tz='UTC').tz_convert(TIMEZONE)
        end_time = pd.Timestamp.fromtimestamp(end_ts, tz='UTC').tz_convert(TIMEZONE)
        df = df[(df.index >= start_time) & (df.index <= end_time)]
        
        if len(df) < len(reference_df):
            print(f"Insufficient data for {symbol}: {len(df)} < {len(reference_df)}")
            return {"symbol": symbol, "results": []}
        
        # Run sliding window DTW - collect ALL valid matches
        reference_length = len(reference_df)
        step_size = max(1, int(reference_length * SLIDING_WINDOW_STEP_RATIO))
        
        all_matches = []
        max_start_index = len(df) - reference_length
        
        for start_index in range(max_start_index, 0, -step_size):
            for factor in config.window_scale_factors:
                window_size = int(reference_length * factor)
                
                if start_index + window_size > len(df):
                    continue
                
                # Extract window
                window = df.iloc[start_index:start_index + window_size]
                
                # Normalize features
                reference_price_normalized, reference_diff_normalized = normalize_features(reference_df)
                window_price_normalized, window_diff_normalized = normalize_features(window)
                
                # DTW screening
                _, price_dtw_distance, _ = dtw_calc.calculate_dtw_similarity(
                    reference_price_normalized, window_price_normalized, 
                    config.dtw_window_ratio, config.dtw_max_point_distance
                )
                
                if np.isinf(price_dtw_distance):
                    continue
                
                _, diff_dtw_distance, _ = dtw_calc.calculate_dtw_similarity(
                    reference_diff_normalized, window_diff_normalized, 
                    config.dtw_window_ratio_diff, config.dtw_max_point_distance_diff
                )
                
                if np.isinf(diff_dtw_distance):
                    continue
                
                # ShapeDTW
                price_descriptor, diff_descriptor = dtw_calc.create_shape_descriptors()
                
                price_shape_distance, _ = dtw_calc.calculate_shapedtw(
                    reference_price_normalized, window_price_normalized, price_descriptor, config.dtw_window_ratio
                )
                
                if np.isinf(price_shape_distance):
                    continue
                
                diff_shape_distance, _ = dtw_calc.calculate_shapedtw(
                    reference_diff_normalized, window_diff_normalized, diff_descriptor, config.dtw_window_ratio_diff
                )
                
                if np.isinf(diff_shape_distance):
                    continue
                
                # Calculate similarity
                price_score = 1 / (1 + price_shape_distance)
                diff_score = 1 / (1 + diff_shape_distance * config.shapedtw_balance_pd_ratio)
                similarity = (price_score * config.price_weight) + (diff_score * config.diff_weight)
                
                if similarity >= MIN_SIMILARITY_SCORE:
                    match_result = {
                        "similarity": similarity,
                        "price_distance": price_shape_distance,
                        "diff_distance": diff_shape_distance,
                        "window_data": window,
                        "full_data": df,
                        "window_info": (start_index, window_size, factor)
                    }
                    all_matches.append(match_result)
        
        # Filter to get non-overlapping matches for this symbol
        filtered_matches = filter_overlapping_matches(all_matches, MAX_MATCHES_PER_SYMBOL)
        
        # Serialize the filtered matches for return
        serialized_matches = []
        for match in filtered_matches:
            serialized_match = {
                "similarity": match["similarity"],
                "price_distance": match["price_distance"],
                "diff_distance": match["diff_distance"],
                "window_data_serialized": {
                    'data': match["window_data"].to_dict('records'),
                    'index': match["window_data"].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                },
                "full_data_serialized": {
                    'data': match["full_data"].to_dict('records'),
                    'index': match["full_data"].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
            }
            serialized_matches.append(serialized_match)
        
        if serialized_matches:
            print(f"Found {len(serialized_matches)} non-overlapping matches for {symbol} (best score: {serialized_matches[0]['similarity']:.4f})")
        
        return {"symbol": symbol, "results": serialized_matches}
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return {"symbol": symbol, "results": []}
    

def normalize_features(df: pd.DataFrame) -> tuple:
    """Helper function to normalize features"""
    price_columns = ['Close', 'SMA_30', 'SMA_45', 'SMA_60']
    diff_columns = ['SMA30_SMA45', 'SMA30_SMA60', 'SMA45_SMA60']
    
    available_price_columns = [col for col in price_columns if col in df.columns]
    available_diff_columns = [col for col in diff_columns if col in df.columns]
    
    if available_price_columns:
        price_features = DataNormalizer.normalize_to_range(df[available_price_columns].values)
    else:
        price_features = np.array([])
    
    if available_diff_columns:
        diff_features = DataNormalizer.normalize_to_range(df[available_diff_columns].values)
    else:
        diff_features = np.array([])
    
    return price_features, diff_features


# ================ Analysis Functions ================

def analyze_future_trend(pattern_df: pd.DataFrame, target_df: pd.DataFrame, 
                       extension_factors: list = None) -> dict:
    """Analyze future trend for different extension factors"""
    if extension_factors is None:
        extension_factors = EXTENSION_FACTORS_FOR_STATS
    
    pattern_end_date = pattern_df.index[-1]
    future_data = target_df[target_df.index > pattern_end_date]
    
    if len(future_data) == 0:
        return {factor: {'trend': 'no_future_data', 'insufficient_data': False} for factor in extension_factors}
    
    pattern_length = len(pattern_df)
    pattern_last_close = pattern_df['Close'].iloc[-1]
    
    results = {}
    
    for factor in extension_factors:
        future_length = int(pattern_length * factor)
        
        if future_length < 1:
            results[factor] = {'trend': 'invalid_factor', 'insufficient_data': False}
            continue
        
        if future_length > len(future_data):
            if len(future_data) > 0:
                future_sample = future_data
                future_last_close = future_sample['Close'].iloc[-1]
                trend = 'rise' if future_last_close > pattern_last_close else 'fall'
                results[factor] = {'trend': trend, 'insufficient_data': True}
            else:
                results[factor] = {'trend': 'no_future_data', 'insufficient_data': False}
            continue
        
        future_sample = future_data.iloc[:future_length]
        future_last_close = future_sample['Close'].iloc[-1]
        
        trend = 'rise' if future_last_close > pattern_last_close else 'fall'
        results[factor] = {'trend': trend, 'insufficient_data': False}
    
    return results


def calculate_trend_statistics(results: list, extension_factors: list = None) -> dict:
    """Calculate trend statistics"""
    if extension_factors is None:
        extension_factors = EXTENSION_FACTORS_FOR_STATS
    
    if not results:
        return {
            'total_results': 0,
            'default_factor_stats': {'rise': 0, 'fall': 0, 'insufficient_data': 0, 'no_future_data': 0},
            'extension_factor_stats': {factor: {'rise': 0, 'fall': 0, 'insufficient_data': 0, 'no_future_data': 0} for factor in extension_factors}
        }
    
    default_stats = {'rise': 0, 'fall': 0, 'insufficient_data': 0, 'no_future_data': 0}
    extension_stats = {factor: {'rise': 0, 'fall': 0, 'insufficient_data': 0, 'no_future_data': 0} for factor in extension_factors}
    
    for result in results:
        if result.get('window_data') is None:
            continue
            
        pattern_df = result['window_data']
        target_df = result.get('full_data')
        
        if target_df is None:
            continue
        
        # Default factor analysis
        default_trend = analyze_future_trend(pattern_df, target_df, [VIS_EXTENSION_FUTURE_LENGTH_FACTOR])
        
        if VIS_EXTENSION_FUTURE_LENGTH_FACTOR in default_trend:
            trend_info = default_trend[VIS_EXTENSION_FUTURE_LENGTH_FACTOR]
            trend_result = trend_info['trend']
            
            if trend_result in ['rise', 'fall']:
                if trend_info.get('insufficient_data', False):
                    default_stats['insufficient_data'] += 1
                else:
                    default_stats[trend_result] += 1
            else:
                default_stats['no_future_data'] += 1
        
        # All extension factors
        all_trends = analyze_future_trend(pattern_df, target_df, extension_factors)
        
        for factor in extension_factors:
            if factor in all_trends:
                trend_info = all_trends[factor]
                trend_result = trend_info['trend']
                
                if trend_result in ['rise', 'fall']:
                    if trend_info.get('insufficient_data', False):
                        extension_stats[factor]['insufficient_data'] += 1
                    else:
                        extension_stats[factor][trend_result] += 1
                else:
                    extension_stats[factor]['no_future_data'] += 1
    
    return {
        'total_results': len(results),
        'default_factor_stats': default_stats,
        'extension_factor_stats': extension_stats
    }


def format_trend_statistics(stats: dict, factor_name: str = "Default") -> list:
    """Format trend statistics into readable text"""
    lines = []
    total = stats['total_results']
    
    if total == 0:
        lines.append(f"{factor_name}: No results available")
        return lines
    
    default_stats = stats['default_factor_stats']
    rise_count = default_stats['rise']
    fall_count = default_stats['fall']
    insufficient_data_count = default_stats.get('insufficient_data', 0)
    no_future_data_count = default_stats.get('no_future_data', 0)
    
    rise_percentage = (rise_count / total) * 100 if total > 0 else 0
    fall_percentage = (fall_count / total) * 100 if total > 0 else 0
    
    lines.append(f"{factor_name} Extension Factor ({VIS_EXTENSION_FUTURE_LENGTH_FACTOR}x):")
    lines.append(f"  Rise: {rise_count}/{total} ({rise_percentage:.1f}%)")
    lines.append(f"  Fall: {fall_count}/{total} ({fall_percentage:.1f}%)")
    if insufficient_data_count > 0:
        lines.append(f"  Insufficient Future Data: {insufficient_data_count}/{total}")
    if no_future_data_count > 0:
        lines.append(f"  No Future Data: {no_future_data_count}/{total}")
    
    extension_stats = stats['extension_factor_stats']
    lines.append(f"\nExtension Factor Analysis:")
    
    for factor in sorted(extension_stats.keys()):
        factor_stats = extension_stats[factor]
        rise_count = factor_stats['rise']
        fall_count = factor_stats['fall']
        
        rise_percentage = (rise_count / total) * 100 if total > 0 else 0
        fall_percentage = (fall_count / total) * 100 if total > 0 else 0
        
        lines.append(f"  {factor}x: Rise {rise_count}({rise_percentage:.1f}%) | Fall {fall_count}({fall_percentage:.1f}%)")
    
    return lines


# ================ Improved Visualization ================

def plot_candlesticks_with_volume_stock(ax: plt.Axes, df: pd.DataFrame, width_factor: float = 0.6, volume_ratio: float = 0.15):
    """Plot candlestick chart with volume bars using sequential positioning for stocks"""
    if len(df) <= 1:
        print("Not enough data points to plot candlesticks")
        return
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        print("Missing required OHLC columns")
        return
    
    # Use sequential integer positions instead of timestamps to avoid gaps
    positions = np.arange(len(df))
    width = width_factor
    
    # Define colors (green up, red down)
    up_color = 'green'
    down_color = 'red'
    
    # Get price range for proper scaling
    price_min = df[['Low']].min().iloc[0]
    price_max = df[['High']].max().iloc[0]
    price_range = price_max - price_min
    
    # Calculate volume range and normalization if volume exists
    has_volume = 'Volume' in df.columns
    if has_volume:
        volume_min = 0  # Volume is already normalized to [0,1]
        volume_max = df['Volume'].max()
        if volume_max > volume_min:
            volume_height = price_range * volume_ratio
            volume_base = price_min - price_range * 0.1  # Gap from price data
            scaled_volume = df['Volume'] * volume_height
        else:
            has_volume = False
    
    # Plot candlesticks using sequential positions
    for i, (timestamp, row) in enumerate(df.iterrows()):
        pos = positions[i]
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # Determine if it's an up or down candle
        is_upward_candle = close_price >= open_price
        color = up_color if is_upward_candle else down_color
        
        # Draw the high-low line (wick)
        ax.plot([pos, pos], [low_price, high_price], 
                color=color, linewidth=1, alpha=0.8)
        
        # Draw the open-close rectangle (body)
        if is_upward_candle:
            rect_bottom = open_price
            rect_height = close_price - open_price
        else:
            rect_bottom = close_price
            rect_height = open_price - close_price
        
        # Create rectangle for the body
        rect = Rectangle((pos - width/2, rect_bottom), width, rect_height,
                        facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Plot volume bars if available
        if has_volume:
            volume_value = scaled_volume.iloc[i]
            volume_rect = Rectangle((pos - width/2, volume_base), width, volume_value,
                                  facecolor=color, edgecolor=color, alpha=0.5)
            ax.add_patch(volume_rect)
    
    # Set x-axis limits and custom labels
    ax.set_xlim(-0.5, len(df) - 0.5)
    
    # Create custom date labels (show every nth date to avoid crowding)
    num_labels = min(8, len(df))  # Show max 8 labels
    if num_labels > 1:
        label_indices = np.linspace(0, len(df)-1, num_labels, dtype=int)
        ax.set_xticks(label_indices)
        ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in label_indices], rotation=45)
    
    # Set y-axis limits to accommodate both price and volume
    if has_volume:
        y_bottom = volume_base - volume_height * 0.05
    else:
        y_bottom = price_min - price_range * 0.05
    
    y_top = price_max + price_range * 0.05
    ax.set_ylim(y_bottom, y_top)


def create_comprehensive_stock_analysis_chart(reference_df: pd.DataFrame, window_df: pd.DataFrame, target_df: pd.DataFrame, 
                                            symbol: str, reference_symbol: str, timeframe: str, reference_timeframe: str, 
                                            reference_label: str, similarity: float, price_distance: float, diff_distance: float, 
                                            output_path: str) -> str:
    """Create comprehensive visualization with three subplots like crypto version"""
    try:
        # Calculate extension periods
        pattern_length = len(window_df)
        past_length = int(pattern_length * VIS_EXTENSION_PAST_LENGTH_FACTOR)
        future_length = int(pattern_length * VIS_EXTENSION_FUTURE_LENGTH_FACTOR)
        
        # Get pattern period information
        pattern_start_date = window_df.index[0]
        pattern_end_date = window_df.index[-1]
        
        # Get past data (before pattern)      
        past_df = target_df[target_df.index < pattern_start_date]
        if len(past_df) >= past_length:
            past_data = past_df.iloc[-past_length:]
        else:
            past_data = past_df
        
        # Get future data (after pattern)
        future_df = target_df[target_df.index > pattern_end_date]
        
        # Analyze future trend to determine file name suffix
        trend_analysis = analyze_future_trend(window_df, target_df, [VIS_EXTENSION_FUTURE_LENGTH_FACTOR])
        
        if VIS_EXTENSION_FUTURE_LENGTH_FACTOR in trend_analysis:
            trend_info = trend_analysis[VIS_EXTENSION_FUTURE_LENGTH_FACTOR]
            trend_result = trend_info['trend']
            
            if trend_result in ['rise', 'fall']:
                if trend_info.get('insufficient_data', False):
                    trend_suffix = f"_{trend_result}_insufficient"
                else:
                    trend_suffix = f"_{trend_result}"
            elif trend_result == 'no_future_data':
                trend_suffix = "_no_future"
            else:
                trend_suffix = "_unknown"
        else:
            trend_suffix = "_unknown"
        
        if len(future_df) >= future_length:
            future_data = future_df.iloc[:future_length]
        else:
            future_data = future_df
        
        # Combine all data for extended view
        extended_parts = []
        if not past_data.empty:
            extended_parts.append(past_data)
        extended_parts.append(window_df)
        if not future_data.empty:
            extended_parts.append(future_data)
        
        extended_df = pd.concat(extended_parts) if extended_parts else window_df
        
        # Normalize data independently for each subplot
        reference_normalized_df, _ = DataNormalizer.normalize_ohlc_dataframe(reference_df, include_volume=True)
        window_normalized_df, _ = DataNormalizer.normalize_ohlc_dataframe(window_df, include_volume=True)
        
        # For extended view, use pattern normalization parameters for OHLC
        pattern_ohlc = window_df[['Open', 'High', 'Low', 'Close']].values
        extended_norm_params = DataNormalizer.calculate_normalization_params(pattern_ohlc, (-1, 1))
        
        # Apply pattern normalization to extended OHLC data
        extended_ohlc = extended_df[['Open', 'High', 'Low', 'Close']].values
        extended_normalized = DataNormalizer.apply_normalization_params(extended_ohlc, extended_norm_params)
        
        extended_normalized_df = extended_df.copy()
        for i, column in enumerate(['Open', 'High', 'Low', 'Close']):
            extended_normalized_df[column] = extended_normalized[:, i]
        
        if 'Volume' in extended_df.columns:
            volume_values = extended_df['Volume'].values.reshape(-1, 1)
            volume_norm_params = DataNormalizer.calculate_normalization_params(volume_values, (0, 1))
            normalized_volume = DataNormalizer.apply_normalization_params(volume_values, volume_norm_params)
            extended_normalized_df['Volume'] = normalized_volume.flatten()
        
        # Also normalize SMA columns for extended view
        sma_columns = ['SMA_30', 'SMA_45', 'SMA_60']
        for column in sma_columns:
            if column in extended_normalized_df.columns:
                extended_normalized_df[column] = DataNormalizer.apply_normalization_params(
                    extended_normalized_df[column].values.reshape(-1, 1), 
                    extended_norm_params
                ).flatten()
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 24))
        
        # Plot 1: Reference trend with volume
        plot_candlesticks_with_volume_stock(ax1, reference_normalized_df, volume_ratio=0.12)
        positions1 = np.arange(len(reference_normalized_df))
        if 'SMA_30' in reference_normalized_df.columns:
            ax1.plot(positions1, reference_normalized_df['SMA_30'], 'blue', linewidth=1.5, alpha=0.9, label='SMA30')
        if 'SMA_45' in reference_normalized_df.columns:
            ax1.plot(positions1, reference_normalized_df['SMA_45'], 'orange', linewidth=1.5, alpha=0.9, label='SMA45')
        if 'SMA_60' in reference_normalized_df.columns:
            ax1.plot(positions1, reference_normalized_df['SMA_60'], 'purple', linewidth=1.5, alpha=0.9, label='SMA60')
        ax1.set_title(f'Reference Trend: {reference_symbol} ({reference_timeframe}, {reference_label})', fontsize=14)
        ax1.set_ylabel('Normalized Price [-1, 1]')
        ax1.set_ylim(-1.2, 1.2)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Target pattern with volume
        plot_candlesticks_with_volume_stock(ax2, window_normalized_df, volume_ratio=0.12)
        positions2 = np.arange(len(window_normalized_df))
        if 'SMA_30' in window_normalized_df.columns:
            ax2.plot(positions2, window_normalized_df['SMA_30'], 'blue', linewidth=1.5, alpha=0.9, label='SMA30')
        if 'SMA_45' in window_normalized_df.columns:
            ax2.plot(positions2, window_normalized_df['SMA_45'], 'orange', linewidth=1.5, alpha=0.9, label='SMA45')
        if 'SMA_60' in window_normalized_df.columns:
            ax2.plot(positions2, window_normalized_df['SMA_60'], 'purple', linewidth=1.5, alpha=0.9, label='SMA60')
        ax2.set_title(f'Target Pattern: {symbol} ({timeframe})', fontsize=14)
        ax2.set_ylabel('Normalized Price [-1, 1]')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Extended view (past + pattern + future) with volume
        plot_candlesticks_with_volume_stock(ax3, extended_normalized_df, volume_ratio=0.12)
        positions3 = np.arange(len(extended_normalized_df))
        if 'SMA_30' in extended_normalized_df.columns:
            ax3.plot(positions3, extended_normalized_df['SMA_30'], 'blue', linewidth=1.5, alpha=0.9, label='SMA30')
        if 'SMA_45' in extended_normalized_df.columns:
            ax3.plot(positions3, extended_normalized_df['SMA_45'], 'orange', linewidth=1.5, alpha=0.9, label='SMA45')
        if 'SMA_60' in extended_normalized_df.columns:
            ax3.plot(positions3, extended_normalized_df['SMA_60'], 'purple', linewidth=1.5, alpha=0.9, label='SMA60')
        
        # Add vertical lines to mark pattern boundaries in extended view
        # Calculate positions in the combined data
        past_length_actual = len(past_data) if not past_data.empty else 0
        ref_start_pos = past_length_actual
        ref_end_pos = past_length_actual + len(window_df) - 1
        
        ax3.axvline(x=ref_start_pos, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='Pattern Start')
        ax3.axvline(x=ref_end_pos, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Pattern End')
        
        ax3.set_title(f'Extended Analysis: {symbol} ({timeframe}) - Past + Pattern + Future', fontsize=14)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Normalized Price (pattern range: [-1, 1])')
        ax3.legend(loc='upper left', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Set y-axis range dynamically for extended view
        extended_values = extended_normalized_df[['Open', 'High', 'Low', 'Close']].values.flatten()
        y_min, y_max = np.min(extended_values), np.max(extended_values)
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        ax3.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add info textbox
        actual_future_length = len(future_data)
        expected_future_length = future_length
        
        future_status = ""
        if actual_future_length == 0:
            future_status = " (No future data available)"
        elif actual_future_length < expected_future_length:
            future_status = f" (Only {actual_future_length}/{expected_future_length} available)"
        
        info_text = (
            f"Similarity Score: {similarity:.4f}\n"
            f"Price Distance: {price_distance:.4f}\n"
            f"SMA Diff Distance: {diff_distance:.4f}\n"
            f"Pattern Period: {format_dt_with_tz(pattern_start_date, TIMEZONE)} to {format_dt_with_tz(pattern_end_date, TIMEZONE)}\n"
            f"Extended View:\n"
            f"  Past Factor: {VIS_EXTENSION_PAST_LENGTH_FACTOR}x ({len(past_data)} bars)\n"
            f"  Pattern: 1.0x ({len(window_df)} bars)\n"
            f"  Future Factor: {VIS_EXTENSION_FUTURE_LENGTH_FACTOR}x ({actual_future_length} bars{future_status})"
        )
        
        plt.figtext(0.02, 0.02, info_text, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Add main title
        fig.suptitle(f"Stock Trend Analysis: {reference_symbol} vs {symbol}", fontsize=18, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        plt.subplots_adjust(hspace=0.15)
        
        # Update output filename to include trend suffix
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        dir_path = os.path.dirname(output_path)
        new_filename = f"{base_filename}{trend_suffix}.png"
        final_output_path = os.path.join(dir_path, new_filename)
        
        # Save figure
        plt.savefig(final_output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved comprehensive stock analysis to {final_output_path}")
        return final_output_path
        
    except Exception as e:
        print(f"Error in comprehensive stock visualization: {e}")
        return output_path


def create_stock_visualization(args: tuple):
    """Create visualization for a single result - for multiprocessing"""
    result, symbol, reference_symbol, reference_timeframe, reference_label, reference_df_data, viz_dir = args
    
    if result is None or result.get("window_data") is None:
        return None
    
    window_df = result["window_data"]
    full_df = result["full_data"]
    similarity = result["similarity"]
    price_distance = result["price_distance"]
    diff_distance = result["diff_distance"]
    
    # Reconstruct reference DataFrame
    reference_df = pd.DataFrame(reference_df_data['data'], index=pd.to_datetime(reference_df_data['index']))
    
    # Create output path
    timestamp = window_df.index[0].strftime("%Y%m%d")
    filename = f"score_{similarity:.4f}_{symbol}_{timestamp}.png"
    output_path = os.path.join(viz_dir, filename)
    
    # Create comprehensive analysis chart
    final_path = create_comprehensive_stock_analysis_chart(
        reference_df, window_df, full_df,
        symbol, reference_symbol, "target_timeframe", reference_timeframe, reference_label,
        similarity, price_distance, diff_distance, output_path
    )
    
    return final_path


# ================ Data Processor (for references only) ================

class StockDataProcessor(BaseDataProcessor):
    """Data processor for reference trends only"""
    
    def __init__(self, config: TrendAnalysisConfig = None):
        super().__init__("stock", config.sma_periods if config else None)
        self.downloader = StockDownloader()
        self.config = config or TrendAnalysisConfig()

    def get_data(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, 
            include_buffer: bool = True, is_reference: bool = False) -> pd.DataFrame:
        """Get stock data - only used for reference trends"""
        if include_buffer:
            interval = end_ts - start_ts
            buffer_start_ts = start_ts - interval
        else:
            buffer_start_ts = start_ts
            
        success, df = self.downloader.get_data(
            symbol, buffer_start_ts, end_ts, timeframe=timeframe,
            dropna=True, atr=False, validate=not is_reference
        )

        if not success or df is None or df.empty:
            return pd.DataFrame()

        # Ensure timezone consistency
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(TIMEZONE)
        df = df.set_index('datetime')
        df = self.processor.prepare_dataframe(df, include_volume=True)
        
        start_time = pd.Timestamp.fromtimestamp(start_ts, tz='UTC').tz_convert(TIMEZONE)
        end_time = pd.Timestamp.fromtimestamp(end_ts, tz='UTC').tz_convert(TIMEZONE)
        df = df[(df.index >= start_time) & (df.index <= end_time)]

        return df


# ================ Main Function ================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Find similar trends in stock data')
    parser.add_argument('-k', '--topk', type=int, default=TOP_K, help=f'Number of top matches (default: {TOP_K})')
    parser.add_argument('-r', '--refresh', action='store_true', help='Force refresh valid symbols cache')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create configuration
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
    config.window_scale_factors = WINDOW_SCALE_FACTORS
    
    # Convert config to dict for serialization
    config_dict = {
        'sma_periods': config.sma_periods,
        'dtw_window_ratio': config.dtw_window_ratio,
        'dtw_window_ratio_diff': config.dtw_window_ratio_diff,
        'dtw_max_point_distance': config.dtw_max_point_distance,
        'dtw_max_point_distance_diff': config.dtw_max_point_distance_diff,
        'shapedtw_balance_pd_ratio': config.shapedtw_balance_pd_ratio,
        'price_weight': config.price_weight,
        'diff_weight': config.diff_weight,
        'slope_window_size': config.slope_window_size,
        'paa_window_size': config.paa_window_size,
        'window_scale_factors': config.window_scale_factors
    }
    
    run_directory = create_output_directory(OUTPUT_DIR, "stock")
    
    print(f"Stock Trend Similarity Analysis")
    print(f"SMA Periods: {SMA_PERIODS}")
    print(f"Timeframes: {TIMEFRAMES_TO_ANALYZE}")
    print(f"Top K Results: {args.topk}")
    print(f"Max matches per symbol: {MAX_MATCHES_PER_SYMBOL}")
    
    # Step 1: Get valid symbols
    print(f"\n{'='*60}")
    print("STEP 1: Getting Valid Stock Symbols")
    print(f"{'='*60}")
    
    valid_symbols = get_valid_symbols(DEFAULT_START_DATE, timeframe="1d", force_refresh=args.refresh)
    
    if not valid_symbols:
        print("No valid symbols found. Exiting.")
        return
    
    print(f"Using {len(valid_symbols)} valid symbols")
    
    # Step 2: Load reference trends
    print(f"\n{'='*60}")
    print("STEP 2: Loading Reference Trends")
    print(f"{'='*60}")
    
    data_processor = StockDataProcessor(config)
    reference_trends = []
    reference_data = {}
    
    for reference_symbol, trends in REFERENCE_TRENDS.items():
        for trend_info in trends:
            start_datetime, end_datetime, reference_timeframe, reference_label = trend_info
            
            reference_df = ReferenceDataManager.load_or_fetch_reference_data(
                reference_symbol, start_datetime, end_datetime, reference_timeframe, reference_label,
                OUTPUT_DIR, TIMEZONE, data_processor, config
            )
            
            if reference_df is not None and not reference_df.empty:
                reference_key = (reference_symbol, reference_timeframe, reference_label)
                
                # Serialize reference data for multiprocessing
                reference_data[reference_key] = {
                    'data': reference_df.to_dict('records'),
                    'index': reference_df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
                reference_trends.append(reference_key)
                print(f"Loaded {reference_symbol} ({reference_timeframe}, {reference_label}): {len(reference_df)} points")
    
    if not reference_trends:
        print("No valid reference trends found. Exiting.")
        return
    
    # Step 3: Process each timeframe
    print(f"\n{'='*60}")
    print("STEP 3: Processing Timeframes")
    print(f"{'='*60}")
    
    all_results = {}
    start_ts = convert_datetime_to_timestamp(DEFAULT_START_DATE, TIMEZONE)
    end_ts = int(time.time())
    
    for timeframe in TIMEFRAMES_TO_ANALYZE:
        print(f"\nProcessing timeframe: {timeframe}")
        
        timeframe_dir = os.path.join(run_directory, f"{timeframe}_results")
        FileManager.ensure_directories(timeframe_dir)
        
        timeframe_results = {}
        
        # Process each reference trend
        for reference_key in reference_trends:
            reference_symbol, reference_timeframe, reference_label = reference_key
            reference_df_data = reference_data[reference_key]
            
            print(f"\nProcessing reference: {reference_symbol} ({reference_timeframe}, {reference_label})")
            
            ref_dir = os.path.join(timeframe_dir, f"{reference_symbol}_{reference_timeframe}_{reference_label}")
            FileManager.ensure_directories(ref_dir)
            
            # Prepare arguments for parallel processing of all symbols
            process_args = []
            for symbol in valid_symbols:
                if symbol == reference_symbol:
                    continue
                    
                process_args.append((
                    symbol, timeframe, start_ts, end_ts, 
                    reference_df_data, reference_symbol, reference_timeframe, reference_label,
                    config_dict
                ))
            
            print(f"Processing {len(process_args)} symbols in parallel...")
            
            # Process all symbols in parallel
            max_workers = cpu_count() - 1
            with Pool(processes=max_workers) as pool:
                raw_results = pool.map(process_single_symbol_against_reference, process_args)
            
            # Collect ALL valid results from all symbols (multiple matches per symbol)
            all_symbol_results = []
            total_matches = 0
            for result in raw_results:
                symbol = result["symbol"]
                for match in result["results"]:
                    # Deserialize data for further processing
                    window_data_dict = match["window_data_serialized"]
                    window_df = pd.DataFrame(window_data_dict['data'])
                    window_df.index = pd.to_datetime(window_data_dict['index'])
                    
                    full_data_dict = match["full_data_serialized"]
                    full_df = pd.DataFrame(full_data_dict['data'])
                    full_df.index = pd.to_datetime(full_data_dict['index'])
                    
                    final_result = {
                        "symbol": symbol,
                        "similarity": match["similarity"],
                        "price_distance": match["price_distance"],
                        "diff_distance": match["diff_distance"],
                        "window_data": window_df,
                        "full_data": full_df
                    }
                    all_symbol_results.append(final_result)
                    total_matches += 1
            
            print(f"Collected {total_matches} total matches from all symbols")
            
            # Sort all matches by similarity score
            all_symbol_results.sort(key=lambda x: x["similarity"], reverse=True)
            print(f"Sorted all matches by similarity score")
            
            # Apply global overlap filtering if enabled
            if GLOBAL_OVERLAP_FILTERING:
                filtered_results = filter_non_overlapping_results(all_symbol_results, True)
                print(f"After global overlap filtering: {len(filtered_results)} results")
            else:
                filtered_results = all_symbol_results
                print(f"No global overlap filtering applied: {len(filtered_results)} results")
            
            # Get top K results
            top_results = filtered_results[:args.topk]
            
            # Calculate statistics
            timeframe_statistics = calculate_trend_statistics(top_results, EXTENSION_FACTORS_FOR_STATS)
            
            # Create summary
            reference_df = pd.DataFrame(reference_df_data['data'])
            reference_df.index = pd.to_datetime(reference_df_data['index'])
            
            summary_lines = []
            summary_lines.append(f"Reference: {reference_symbol} ({reference_timeframe}, {reference_label})")
            summary_lines.append(f"Reference Period: {format_dt_with_tz(reference_df.index[0], TIMEZONE)} to {format_dt_with_tz(reference_df.index[-1], TIMEZONE)}")
            summary_lines.append(f"Data Points: {len(reference_df)}")
            summary_lines.append(f"Max matches per symbol: {MAX_MATCHES_PER_SYMBOL}")
            summary_lines.append(f"Total matches collected: {total_matches}")
            summary_lines.append(f"Global overlap filtering: {GLOBAL_OVERLAP_FILTERING}")
            summary_lines.append("-" * 50)
            
            # Add statistics
            stat_lines = format_trend_statistics(timeframe_statistics, f"Timeframe {timeframe}")
            summary_lines.extend(stat_lines)
            summary_lines.append("-" * 50)
            
            if top_results:
                # Generate visualizations in parallel
                print(f"Generating {len(top_results)} visualizations...")
                viz_dir = os.path.join(ref_dir, "visualizations")
                FileManager.ensure_directories(viz_dir)
                
                viz_args = []
                for result in top_results:
                    viz_args.append((
                        result,
                        result["symbol"],
                        reference_symbol,
                        reference_timeframe,
                        reference_label,
                        reference_df_data,
                        viz_dir
                    ))
                
                # Create visualizations in parallel
                with Pool(processes=max_workers) as pool:
                    pool.map(create_stock_visualization, viz_args)
                
                # Add top results to summary
                summary_lines.append("Top Results:")
                for i, result in enumerate(top_results):
                    symbol = result["symbol"]
                    score = result["similarity"]
                    window_data = result["window_data"]
                    
                    window_period = f"{format_dt_with_tz(window_data.index[0], TIMEZONE)} to {format_dt_with_tz(window_data.index[-1], TIMEZONE)}"
                    
                    summary_lines.append(f"{i+1}. {symbol}")
                    summary_lines.append(f"   Score: {score:.4f}")
                    summary_lines.append(f"   Period: {window_period}")
                    summary_lines.append("")
            else:
                summary_lines.append("No matching trends found")
            
            # Save summary
            summary_text = '\n'.join(summary_lines)
            summary_file = os.path.join(ref_dir, "results_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            timeframe_results[reference_key] = {
                "top_results": top_results,
                "statistics": timeframe_statistics,
                "total_matches": total_matches
            }
            
            print(f"\n{summary_text}")
        
        all_results[timeframe] = timeframe_results
    
    # Step 4: Create overall summary
    print(f"\n{'='*60}")
    print("STEP 4: Creating Overall Summary")
    print(f"{'='*60}")
    
    overall_summary = []
    overall_summary.append(f"Stock Trend Similarity Analysis Report")
    overall_summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    overall_summary.append(f"Valid Symbols: {len(valid_symbols)}")
    overall_summary.append(f"SMA Periods: {SMA_PERIODS}")
    overall_summary.append(f"Timeframes: {TIMEFRAMES_TO_ANALYZE}")
    overall_summary.append(f"Top K Results: {args.topk}")
    overall_summary.append(f"Max matches per symbol: {MAX_MATCHES_PER_SYMBOL}")
    overall_summary.append(f"Global overlap filtering: {GLOBAL_OVERLAP_FILTERING}")
    overall_summary.append(f"{'='*50}\n")
    
    # Add results for each timeframe
    for timeframe, timeframe_results in all_results.items():
        if not timeframe_results:
            continue
            
        overall_summary.append(f"TIMEFRAME: {timeframe}")
        overall_summary.append(f"{'='*30}")
        
        for reference_key, results in timeframe_results.items():
            reference_symbol, reference_timeframe, reference_label = reference_key
            top_results = results["top_results"]
            statistics = results["statistics"]
            total_matches = results["total_matches"]
            
            overall_summary.append(f"\nReference: {reference_symbol} ({reference_timeframe}, {reference_label})")
            overall_summary.append(f"Total matches collected: {total_matches}")
            overall_summary.append(f"{'-'*40}")
            
            # Add statistics
            stat_lines = format_trend_statistics(statistics, f"Timeframe {timeframe}")
            overall_summary.extend(stat_lines)
            
            if top_results:
                overall_summary.append(f"\nFound {len(top_results)} top matching patterns")
                
                # Count trends by direction
                rise_count = 0
                fall_count = 0
                for result in top_results:
                    pattern_df = result["window_data"]
                    target_df = result["full_data"]
                    
                    trend_analysis = analyze_future_trend(pattern_df, target_df, [VIS_EXTENSION_FUTURE_LENGTH_FACTOR])
                    if VIS_EXTENSION_FUTURE_LENGTH_FACTOR in trend_analysis:
                        trend_result = trend_analysis[VIS_EXTENSION_FUTURE_LENGTH_FACTOR]['trend']
                        if trend_result == 'rise':
                            rise_count += 1
                        elif trend_result == 'fall':
                            fall_count += 1
                
                overall_summary.append(f"Trend Distribution: {rise_count} RISE, {fall_count} FALL")
            else:
                overall_summary.append(f"\nNo matching trends found")
            
            overall_summary.append("")
    
    # Save overall summary
    overall_summary_text = '\n'.join(overall_summary)
    overall_summary_file = os.path.join(run_directory, "overall_summary.txt")
    with open(overall_summary_file, 'w') as f:
        f.write(overall_summary_text)
    
    # Print final summary
    print("\n" + overall_summary_text)
    print(f"\nOverall summary saved to: {overall_summary_file}")
    print(f"Results saved to: {run_directory}")
    
    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"Analyzed {len(valid_symbols)} stock symbols across {len(TIMEFRAMES_TO_ANALYZE)} timeframes")


if __name__ == "__main__":
    main()