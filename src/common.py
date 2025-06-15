"""
Common utilities for trend similarity analysis
Shared functions and classes for both crypto and stock analysis
Supports multiple asset types and provides foundation for volume analysis
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dtaidistance import dtw, dtw_ndim
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, CompoundDescriptor, RawSubsequenceDescriptor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# ================ Configuration Management ================

class TrendAnalysisConfig:
    """Configuration manager for trend analysis parameters"""
    
    def __init__(self):
        # Default SMA periods
        self.sma_periods = [30, 45, 60]
        
        # Default DTW parameters
        self.dtw_window_ratio = 0.2
        self.dtw_window_ratio_diff = 0.1
        self.dtw_max_point_distance = 0.66
        self.dtw_max_point_distance_diff = 0.5
        
        # Default ShapeDTW parameters
        self.shapedtw_balance_pd_ratio = 4
        self.price_weight = 0.4
        self.diff_weight = 0.6
        self.slope_window_size = 5
        self.paa_window_size = 5
        
        # Default window scaling factors
        self.window_scale_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        # Minimum query length
        self.min_query_length = 60
        
        # API settings
        self.api_sleep_seconds = 0.5
        self.request_time_buffer_ratio = 1.2


# ================ Time and Date Utilities ================

def calculate_timeframe_seconds(timeframe: str) -> int:
    """Calculate seconds for timeframe string (e.g., "15m", "1h", "4h")"""
    if 'm' in timeframe:
        minutes = int(timeframe.replace('m', ''))
        return minutes * 60
    elif 'h' in timeframe:
        hours = int(timeframe.replace('h', ''))
        return hours * 3600
    elif 'd' in timeframe:
        days = int(timeframe.replace('d', ''))
        return days * 86400
    else:
        print(f"Unknown timeframe format: {timeframe}, defaulting to 1 hour")
        return 3600


def convert_datetime_to_timestamp(dt_obj: datetime, tz_name: str) -> int:
    """Convert datetime object to timestamp with timezone consideration"""
    tz = timezone(tz_name)
    dt_with_tz = tz.localize(dt_obj)
    return int(dt_with_tz.timestamp())


def format_dt_with_tz(dt: pd.Timestamp, tz_name: str = "UTC") -> str:
    """Format datetime with timezone consideration"""   
    if dt.tz is None:
        # If no timezone info, assume UTC and convert
        dt = dt.tz_localize('UTC')
    
    # Convert to target timezone
    tz = timezone(tz_name)
    dt_local = dt.tz_convert(tz)
    
    return dt_local.strftime('%Y-%m-%d %H:%M')


# ================ Data Normalization ================

class DataNormalizer:
    """Handles data normalization with support for different asset types"""
   
    @staticmethod
    def normalize_to_range(data_array: np.ndarray, target_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """Normalize data using Z-score followed by min-max scaling to target range"""
        # Convert to numpy array if needed
        if isinstance(data_array, (pd.Series, pd.DataFrame)):
            data_array = data_array.values
        
        # Flatten array to compute global statistics
        flat_data = data_array.flatten()
        
        # Step 1: Z-score normalization
        global_mean = np.mean(flat_data)
        global_std = np.std(flat_data)
        
        # Handle zero standard deviation
        if global_std > 0:
            z_scored = (data_array - global_mean) / global_std
        else:
            # Fallback to min-max if std is zero
            global_min = np.min(flat_data)
            global_max = np.max(flat_data)
            if global_max > global_min:
                z_scored = (data_array - global_min) / (global_max - global_min)
            else:
                z_scored = np.zeros_like(data_array)
        
        # Step 2: Min-max scaling to target range
        z_min = np.min(z_scored)
        z_max = np.max(z_scored)
        
        # Handle case where min equals max
        if z_max > z_min:
            target_min, target_max = target_range
            normalized = target_min + (z_scored - z_min) * (target_max - target_min) / (z_max - z_min)
        else:
            # If all values are the same, set to the middle of the target range
            target_min, target_max = target_range
            normalized = np.full_like(z_scored, (target_min + target_max) / 2)
        
        return normalized
    
    @staticmethod
    def calculate_normalization_params(data_array: np.ndarray, target_range: Tuple[float, float] = (-1, 1)) -> Dict:
        """Calculate normalization parameters from data without applying normalization"""
        # Convert to numpy array if needed
        if isinstance(data_array, (pd.Series, pd.DataFrame)):
            data_array = data_array.values
        
        # Flatten array to compute global statistics
        flat_data = data_array.flatten()
        
        # Step 1: Z-score normalization parameters
        global_mean = np.mean(flat_data)
        global_std = np.std(flat_data)
        
        # Handle zero standard deviation case
        if global_std > 0:
            z_scored = (data_array - global_mean) / global_std
        else:
            # Fallback to min-max if std is zero
            global_min = np.min(flat_data)
            global_max = np.max(flat_data)
            if global_max > global_min:
                z_scored = (data_array - global_min) / (global_max - global_min)
            else:
                z_scored = np.zeros_like(data_array)
        
        # Step 2: Min-max scaling parameters
        z_min = np.min(z_scored)
        z_max = np.max(z_scored)
        
        target_min, target_max = target_range
        
        # Return normalization parameters
        norm_params = {
            'global_mean': global_mean,
            'global_std': global_std,
            'z_min': z_min,
            'z_max': z_max,
            'target_min': target_min,
            'target_max': target_max
        }
        
        return norm_params
   
    @staticmethod
    def apply_normalization_params(data_array: np.ndarray, norm_params: Dict) -> np.ndarray:
        """Apply normalization parameters to data"""
        # Convert to numpy array if needed
        if isinstance(data_array, (pd.Series, pd.DataFrame)):
            data_array = data_array.values
        
        # Extract parameters
        global_mean = norm_params['global_mean']
        global_std = norm_params['global_std']
        z_min = norm_params['z_min']
        z_max = norm_params['z_max']
        target_min = norm_params['target_min']
        target_max = norm_params['target_max']
        
        # Apply Z-score normalization using stored parameters
        if global_std > 0:
            z_scored = (data_array - global_mean) / global_std
        else:
            # This is a fallback case that should rarely happen
            z_scored = np.zeros_like(data_array)
        
        # Apply min-max scaling using stored parameters
        if z_max > z_min:
            normalized = target_min + (z_scored - z_min) * (target_max - target_min) / (z_max - z_min)
        else:
            # This is a fallback case that should rarely happen
            normalized = np.full_like(z_scored, (target_min + target_max) / 2)
        
        return normalized
    
    @staticmethod
    def normalize_ohlc_dataframe(df: pd.DataFrame, include_volume: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Normalize OHLC data in a DataFrame to range [-1, 1]"""
        # Extract OHLC columns
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        
        # Get values for normalization 
        ohlc_values = df[ohlc_columns].values
        
        # Calculate normalization parameters
        norm_params = DataNormalizer.calculate_normalization_params(ohlc_values, (-1, 1))
        
        # Apply normalization to OHLC
        normalized_values = DataNormalizer.apply_normalization_params(ohlc_values, norm_params)
        
        # Create a copy of the original dataframe
        normalized_df = df.copy()
        
        # Replace OHLC values with normalized ones
        for i, column in enumerate(ohlc_columns):
            normalized_df[column] = normalized_values[:, i]
        
        if include_volume and 'Volume' in df.columns:
            volume_values = df['Volume'].values.reshape(-1, 1)
            volume_norm_params = DataNormalizer.calculate_normalization_params(volume_values, (0, 1))
            normalized_volume = DataNormalizer.apply_normalization_params(volume_values, volume_norm_params)
            normalized_df['Volume'] = normalized_volume.flatten()
            norm_params['volume_norm_params'] = volume_norm_params
        
        # Also normalize SMA columns using OHLC parameters
        sma_columns = ['SMA_30', 'SMA_45', 'SMA_60']
        for column in sma_columns:
            if column in normalized_df.columns:
                normalized_df[column] = DataNormalizer.apply_normalization_params(
                    normalized_df[column].values.reshape(-1, 1), norm_params
                ).flatten()
        
        return normalized_df, norm_params


# ================ Time Series Processing ================

class TimeSeriesProcessor:
    """Handles time series data preprocessing and feature calculation"""
    
    def __init__(self, sma_periods: List[int] = None):
        """Initialize processor with SMA periods"""
        self.sma_periods = sma_periods or [30, 45, 60]
    
    def prepare_dataframe(self, df: pd.DataFrame, include_volume: bool = True) -> pd.DataFrame:
        """Prepare and standardize dataframe for analysis"""
        # Convert timestamp to datetime if needed
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Set datetime as index if not already done
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('datetime')
        
        # Calculate SMAs if not already present
        for period in self.sma_periods:
            sma_column = f'sma_{period}'
            if sma_column not in df.columns:
                df[sma_column] = df['close'].rolling(window=period).mean()
        
        # Standardize column names
        column_mapping = {
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'sma_30': 'SMA_30',
            'sma_45': 'SMA_45',
            'sma_60': 'SMA_60'
        }
        
        # Add volume mapping if needed and available
        if include_volume and 'volume' in df.columns:
            column_mapping['volume'] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        # Calculate SMA difference features
        df['SMA30_SMA45'] = df['SMA_30'] - df['SMA_45']
        df['SMA30_SMA60'] = df['SMA_30'] - df['SMA_60']
        df['SMA45_SMA60'] = df['SMA_45'] - df['SMA_60']
        
        # Calculate price-SMA differences
        df['Close_SMA30'] = df['Close'] - df['SMA_30']
        df['Close_SMA45'] = df['Close'] - df['SMA_45']
        df['Close_SMA60'] = df['Close'] - df['SMA_60']
        
        return df
    
    def calculate_sma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA and related features for the dataframe"""
        for period in self.sma_periods:
            sma_col = f'SMA_{period}'
            if sma_col not in df.columns:
                df[sma_col] = df['Close'].rolling(window=period).mean()
        
        return df


# ================ DTW Calculation Engine ================

class DTWCalculator:
    """Core DTW calculation engine supporting multiple asset types"""
    
    def __init__(self, config: TrendAnalysisConfig = None):
        """Initialize DTW calculator with configuration"""
        self.config = config or TrendAnalysisConfig()
        self.c_available = self._check_c_availability()
    
    def _check_c_availability(self) -> bool:
        """Check if C implementation of dtaidistance is available"""
        try:
            return dtw.try_import_c(verbose=False)
        except Exception as e:
            print(f"Error checking C availability: {e}")
            return False
    
    def normalize_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize price and difference features"""
        # Define feature columns
        price_columns = ['Close', 'SMA_30', 'SMA_45', 'SMA_60']
        diff_columns = ['SMA30_SMA45', 'SMA30_SMA60', 'SMA45_SMA60']
        
        # Normalize price features
        price_features = DataNormalizer.normalize_to_range(df[price_columns].values)
        
        # Normalize difference features
        diff_features = DataNormalizer.normalize_to_range(df[diff_columns].values)
        
        return price_features, diff_features
    
    def calculate_dtw_similarity(self, query_series: np.ndarray, target_series: np.ndarray, 
                                window_ratio: float, max_point_distance: float) -> Tuple[float, float, List]:
        """Calculate DTW similarity between two series"""
        # Calculate window size
        window_size = int(max(len(query_series), len(target_series)) * window_ratio)
        
        # Calculate DTW
        try:
            distance, paths = dtw_ndim.warping_paths(
                query_series, 
                target_series,
                window=window_size,
                use_c=self.c_available,
                max_step=max_point_distance
            )
            
            # If distance is inf (no valid path), return low similarity
            if np.isinf(distance):
                return 0.0, float('inf'), []
            
            # Find best path
            path = dtw.best_path(paths)
            
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            
            return similarity, distance, path
        
        except Exception as e:
            print(f"Error in DTW calculation: {e}")
            return 0.0, float('inf'), []
    
    def calculate_shapedtw(self, query_series: np.ndarray, target_series: np.ndarray, 
                        shape_descriptor, window_ratio: float, subsequence_width: int = 5) -> Tuple[float, List]:
        """Calculate Shape DTW between two series"""
        try:
            # Calculate window size
            window_size = int(max(len(query_series), len(target_series)) * window_ratio)
            
            # Calculate ShapeDTW
            shape_dtw_results = shape_dtw(
                x=query_series,
                y=target_series,
                step_pattern="asymmetric",
                open_begin=False,
                subsequence_width=subsequence_width,
                shape_descriptor=shape_descriptor,
                multivariate_version="dependent",
                window_type="sakoechiba",
                window_args={"window_size": window_size},
            )
            
            # Extract results
            distance = shape_dtw_results.shape_normalized_distance
            path = list(zip(shape_dtw_results.index1, shape_dtw_results.index2))
            
            return distance, path
            
        except Exception as e:
            print(f"Error in ShapeDTW calculation: {e}")
            return float('inf'), []
    
    def create_shape_descriptors(self) -> Tuple:
        """Create shape descriptors for price and difference features"""
        # For price features - emphasis on raw shape
        price_descriptor = CompoundDescriptor(
            [RawSubsequenceDescriptor(), SlopeDescriptor(slope_window=self.config.slope_window_size)],
            descriptors_weights=[4.0, 1.0]
        )
        
        # For difference features - emphasis on slope and patterns
        diff_descriptor = CompoundDescriptor(
            [SlopeDescriptor(slope_window=self.config.slope_window_size), 
                PAADescriptor(piecewise_aggregation_window=self.config.paa_window_size)],
            descriptors_weights=[3.0, 1.0]
        )
        
        return price_descriptor, diff_descriptor


# ================ Visualization Functions ================

def plot_candlesticks_with_volume(ax: plt.Axes, df: pd.DataFrame, width_factor: float = 0.6, volume_ratio: float = 0.15):
    """Plot candlestick chart with volume bars in same subplot"""
    if len(df) <= 1:
        print("Not enough data points to plot candlesticks")
        return
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        print("Missing required OHLC columns")
        return
    
    # Calculate appropriate width for candlesticks in days
    time_diff = (df.index[1] - df.index[0]).total_seconds() / 86400  # Convert to days
    width = time_diff * width_factor
    
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
        volume_min = 0  # 因為Volume已經標準化到[0,1]，所以最小值是0
        volume_max = df['Volume'].max()
        if volume_max > volume_min:
            volume_height = price_range * volume_ratio
            volume_base = price_min - price_range * 0.1  # Gap from price data
            scaled_volume = df['Volume'] * volume_height
        else:
            has_volume = False
    
    # Plot candlesticks
    for timestamp, row in df.iterrows():
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # Determine if it's an up or down candle
        is_upward_candle = close_price >= open_price
        color = up_color if is_upward_candle else down_color
        
        # Draw the high-low line (wick)
        ax.plot([timestamp, timestamp], [low_price, high_price], 
                color=color, linewidth=1, alpha=0.8)
        
        # Calculate rectangle coordinates in timestamp space
        half_width_timedelta = pd.Timedelta(days=width/2)
        
        # Draw the open-close rectangle (body)
        if is_upward_candle:
            rect_bottom = open_price
            rect_height = close_price - open_price
        else:
            rect_bottom = close_price
            rect_height = open_price - close_price
        
        # Create rectangle for the body
        rect = Rectangle((timestamp - half_width_timedelta, rect_bottom),
                            pd.Timedelta(days=width), rect_height,
                            facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Plot volume bars if available
        if has_volume:
            volume_value = scaled_volume.loc[timestamp]
            volume_rect = Rectangle((timestamp - half_width_timedelta, volume_base),
                                    pd.Timedelta(days=width), volume_value,
                                    facecolor=color, edgecolor=color, alpha=0.5)
            ax.add_patch(volume_rect)
    
    # Set y-axis limits to accommodate both price and volume
    if has_volume:
        y_bottom = volume_base - volume_height * 0.05
    else:
        y_bottom = price_min - price_range * 0.05
    
    y_top = price_max + price_range * 0.05
    ax.set_ylim(y_bottom, y_top)

# ================ Data Management Classes ================

class DataCacheManager:
    """Manages data caching for timeframe data"""
    
    @staticmethod
    def get_timeframe_cache_path(output_dir: str, timeframe: str) -> str:
        """Generate cache filename for timeframe data"""
        return os.path.join(output_dir, f"all_symbols_{timeframe}.pkl")
    
    @staticmethod
    def download_timeframe_data(timeframe: str, output_dir: str, config: TrendAnalysisConfig, 
                                historical_start_date: datetime, data_processor) -> dict:
        """Download and cache data for all symbols of a specific timeframe"""
        # Check for cached data
        cache_file = DataCacheManager.get_timeframe_cache_path(output_dir, timeframe)
        
        if os.path.exists(cache_file):
            print(f"Loading cached data for timeframe {timeframe}...")
            data_dict = FileManager.load_from_cache(cache_file)
            if data_dict is not None:
                print(f"Loaded cached data for {len(data_dict)} symbols in timeframe {timeframe}")
                return data_dict
        
        print(f"No cached data found for timeframe {timeframe}, downloading...")
        
        # Get all available symbols
        all_symbols = data_processor.downloader.get_all_symbols()
        symbols = [s.replace('USDT', '') for s in all_symbols]
        print(f"Found {len(symbols)} available symbols")
        
        # Convert start date to timestamp
        start_timestamp = int(historical_start_date.timestamp())
        
        # Get current time as end timestamp
        end_timestamp = int(time.time())
        
        # Download data for all symbols
        data_dict = {}
        
        print(f"Downloading data for timeframe {timeframe} from {historical_start_date} to now...")
        
        for symbol in symbols:
            print(f"Downloading data for {symbol} ({timeframe})...")
            
            # Get data
            df = data_processor.get_data(
                symbol,
                timeframe,
                start_timestamp,
                end_timestamp
            )
            
            if not df.empty:
                data_dict[symbol] = df
                print(f"Downloaded {len(df)} data points for {symbol} ({timeframe})")
            else:
                print(f"Failed to download data for {symbol} ({timeframe})")
                data_dict[symbol] = None
            
            # Sleep to avoid API rate limits
            time.sleep(config.api_sleep_seconds)
        
        # Save to cache
        FileManager.save_to_cache(data_dict, cache_file)
        
        print(f"Saved data for timeframe {timeframe} to cache")
        
        return data_dict


class ReferenceDataManager:
    """Manages reference trend data loading and visualization"""
    
    @staticmethod
    def get_reference_cache_path(output_dir: str, symbol: str, timeframe: str, label: str, start_ts: int, end_ts: int) -> str:
        """Generate cache path for reference data"""
        reference_dir = os.path.join(output_dir, "reference")
        return os.path.join(reference_dir, f"ref_{symbol}_{timeframe}_{label}_{start_ts}_{end_ts}.pkl")
    
    @staticmethod
    def load_or_fetch_reference_data(symbol: str, start_datetime: datetime, end_datetime: datetime,
                                    timeframe: str, label: str, output_dir: str, timezone_name: str,
                                    data_processor, config: TrendAnalysisConfig) -> pd.DataFrame:
        """Load or fetch reference trend data with unified caching"""
        print(f"Loading reference trend for {symbol} ({timeframe}) from {start_datetime} to {end_datetime}...")
        
        # Convert datetime to timestamp
        start_ts = convert_datetime_to_timestamp(start_datetime, timezone_name)
        end_ts = convert_datetime_to_timestamp(end_datetime, timezone_name)
        
        # Create reference directory
        reference_dir = os.path.join(output_dir, "reference")
        FileManager.ensure_directories(reference_dir)
        
        # Cache file path
        cache_file = ReferenceDataManager.get_reference_cache_path(output_dir, symbol, timeframe, label, start_ts, end_ts)
        
        # Check if cache exists
        reference_data = FileManager.load_from_cache(cache_file)
        if reference_data is not None:
            print(f"Loading cached reference data for {symbol}...")
            
            # Create reference visualization if not already done
            viz_file = os.path.join(reference_dir, f"ref_{symbol}_{timeframe}_{label}_{start_ts}_{end_ts}.png")
            if not os.path.exists(viz_file):
                ReferenceDataManager.create_reference_visualization(
                    reference_data['df'], 
                    reference_data.get('past_df'), 
                    reference_data.get('future_df'), 
                    symbol, timeframe, label, viz_file,
                    timezone_name
                )
            
            return reference_data['df']  # Return only reference data for comparison
        
        # Calculate extended period for past and future
        time_difference = end_ts - start_ts
        extended_start_ts = start_ts - time_difference * 1.0  # 1x past
        extended_end_ts = end_ts + time_difference * 2.0      # 2x future
        
        # Get extended data (past + reference + future)
        extended_df = data_processor.get_data(
            symbol,
            timeframe,
            extended_start_ts,
            extended_end_ts,
            include_buffer=False,
            is_reference=True
        )
        
        if extended_df.empty:
            print(f"Failed to get extended data for {symbol} at {timeframe}")
            return None
        
        # Split data into past, reference, and future
        reference_start_time = pd.Timestamp.fromtimestamp(start_ts, tz='UTC')
        reference_end_time = pd.Timestamp.fromtimestamp(end_ts, tz='UTC')

        if extended_df.index.tz is None:
            extended_df.index = pd.to_datetime(extended_df.index, utc=True)
        elif extended_df.index.tz != pd.Timestamp.now(tz='UTC').tz:
            extended_df.index = extended_df.index.tz_convert('UTC')
        
        past_df = extended_df[extended_df.index < reference_start_time]
        reference_df = extended_df[(extended_df.index >= reference_start_time) & 
                                (extended_df.index <= reference_end_time)]
        future_df = extended_df[extended_df.index > reference_end_time]
        
        # Create reference data object to cache
        reference_data = {
            'df': reference_df,           # Only reference data for comparison
            'past_df': past_df if not past_df.empty else None,
            'future_df': future_df if not future_df.empty else None
        }
        
        # Save to cache
        FileManager.save_to_cache(reference_data, cache_file)
        
        # Create reference visualization with past + reference + future
        viz_file = os.path.join(reference_dir, f"ref_{symbol}_{timeframe}_{label}_{start_ts}_{end_ts}.png")
        ReferenceDataManager.create_reference_visualization(
            reference_df, past_df, future_df, symbol, timeframe, label, viz_file
        )
        
        print(f"Saved reference data for {symbol} with {len(reference_df)} data points")
        return reference_df  # Return only reference data for comparison


    @staticmethod
    def create_reference_visualization(reference_df: pd.DataFrame, past_df: pd.DataFrame, 
                                    future_df: pd.DataFrame, symbol: str, timeframe: str, 
                                    label: str, output_path: str, timezone_name: str = "UTC"):
        """Create reference visualization with two subplots: reference only and past + reference + future"""
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [1, 1]})
            
            # Normalize reference data (include volume)
            reference_normalized_df, _ = DataNormalizer.normalize_ohlc_dataframe(reference_df, include_volume=True)
            
            # Plot 1: Reference trend only with volume
            plot_candlesticks_with_volume(ax1, reference_normalized_df, volume_ratio=0.12)
            ax1.plot(reference_normalized_df.index, reference_normalized_df['SMA_30'], 'blue', linewidth=2, alpha=0.8, label='SMA30')
            ax1.plot(reference_normalized_df.index, reference_normalized_df['SMA_45'], 'orange', linewidth=2, alpha=0.8, label='SMA45')
            ax1.plot(reference_normalized_df.index, reference_normalized_df['SMA_60'], 'purple', linewidth=2, alpha=0.8, label='SMA60')
            ax1.set_title(f'Reference Trend: {symbol} ({timeframe}, {label})', fontsize=14)
            ax1.set_ylabel('Normalized Price [-1, 1]', fontsize=12)
            ax1.set_ylim(-1.2, 1.2)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Past + Reference + Future
            # Combine all available data
            combined_parts = []
            if past_df is not None and not past_df.empty:
                # Take last 1x reference length of past data
                past_length = len(reference_df)
                if len(past_df) >= past_length:
                    past_data = past_df.iloc[-past_length:]
                else:
                    past_data = past_df
                combined_parts.append(past_data)
            
            combined_parts.append(reference_df)
            
            if future_df is not None and not future_df.empty:
                # Take first 2x reference length of future data
                future_length = len(reference_df) * 2
                if len(future_df) >= future_length:
                    future_data = future_df.iloc[:future_length]
                else:
                    future_data = future_df
                combined_parts.append(future_data)
            
            if combined_parts:
                combined_df = pd.concat(combined_parts)
                
                # Use reference normalization parameters for entire combined data
                ref_ohlc = reference_df[['Open', 'High', 'Low', 'Close']].values
                ref_norm_params = DataNormalizer.calculate_normalization_params(ref_ohlc, (-1, 1))
                
                # Apply normalization to combined OHLC data
                combined_ohlc = combined_df[['Open', 'High', 'Low', 'Close']].values
                combined_normalized = DataNormalizer.apply_normalization_params(combined_ohlc, ref_norm_params)
                
                combined_normalized_df = combined_df.copy()
                for i, column in enumerate(['Open', 'High', 'Low', 'Close']):
                    combined_normalized_df[column] = combined_normalized[:, i]
                
                # Normalize SMA columns using reference parameters
                sma_columns = ['SMA_30', 'SMA_45', 'SMA_60']
                for column in sma_columns:
                    if column in combined_normalized_df.columns:
                        combined_normalized_df[column] = DataNormalizer.apply_normalization_params(
                            combined_normalized_df[column].values.reshape(-1, 1), ref_norm_params
                        ).flatten()
                
                # Separately normalize Volume
                if 'Volume' in combined_df.columns:
                    volume_values = combined_df['Volume'].values.reshape(-1, 1)
                    volume_norm_params = DataNormalizer.calculate_normalization_params(volume_values, (0, 1))
                    normalized_volume = DataNormalizer.apply_normalization_params(volume_values, volume_norm_params)
                    combined_normalized_df['Volume'] = normalized_volume.flatten()
                
                plot_candlesticks_with_volume(ax2, combined_normalized_df, volume_ratio=0.12)
                ax2.plot(combined_normalized_df.index, combined_normalized_df['SMA_30'], 'blue', linewidth=2, alpha=0.8, label='SMA30')
                ax2.plot(combined_normalized_df.index, combined_normalized_df['SMA_45'], 'orange', linewidth=2, alpha=0.8, label='SMA45')
                ax2.plot(combined_normalized_df.index, combined_normalized_df['SMA_60'], 'purple', linewidth=2, alpha=0.8, label='SMA60')
                
                # Add vertical lines to mark reference boundaries
                reference_start = reference_df.index[0]
                reference_end = reference_df.index[-1]
                ax2.axvline(x=reference_start, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='Reference Start')
                ax2.axvline(x=reference_end, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Reference End')
                
                ax2.set_title(f'Extended View: {symbol} - Past + Reference + Future', fontsize=14)
                ax2.set_ylabel('Normalized Price (ref range: [-1, 1])', fontsize=12)
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)
                
                # Set y-axis range dynamically for combined data
                combined_values = combined_normalized_df[['Open', 'High', 'Low', 'Close']].values.flatten()
                y_min, y_max = np.min(combined_values), np.max(combined_values)
                y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
                ax2.set_ylim(y_min - y_padding, y_max + y_padding)
            else:
                ax2.text(0.5, 0.5, 'No extended data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Extended View: No Data', fontsize=14)
            
            # Format date ticks
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add info textbox
            past_length = len(past_data) if past_df is not None and not past_df.empty else 0
            future_length = len(future_data) if future_df is not None and not future_df.empty else 0
            
            info_text = (
                f"Symbol: {symbol}\n"
                f"Timeframe: {timeframe}\n"
                f"Label: {label}\n"
                f"Reference Period: {format_dt_with_tz(reference_df.index[0], timezone_name)} to {format_dt_with_tz(reference_df.index[-1], timezone_name)}\n"
                f"Data Points: {len(reference_df)}\n"
                f"Extended View:\n"
                f"  Past: {past_length} bars\n"
                f"  Reference: {len(reference_df)} bars\n"
                f"  Future: {future_length} bars"
            )
            
            plt.figtext(0.02, 0.02, info_text, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            plt.subplots_adjust(hspace=0.2)
            
            # Save and close
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved reference visualization to {output_path}")
        except Exception as e:
            print(f"Error in reference visualization: {e}")


# ================ File and Directory Management ================

class FileManager:
    """Handles file operations and directory management"""
    
    @staticmethod
    def ensure_directories(*dirs: str) -> None:
        """Create directories if they don't exist"""
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_cache_filename(base_dir: str, prefix: str, **kwargs) -> str:
        """Generate cache filename with parameters"""
        parts = [prefix]
        for key, value in kwargs.items():
            parts.append(f"{key}_{value}")
        
        filename = "_".join(parts) + ".pkl"
        return os.path.join(base_dir, filename)
    
    @staticmethod
    def save_to_cache(data: any, filepath: str) -> None:
        """Save data to cache file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_from_cache(filepath: str) -> any:
        """Load data from cache file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None


# ================ Abstract Base Classes ================

class BaseDataDownloader(ABC):
    """Abstract base class for data downloaders"""
    
    @abstractmethod
    def get_data(self, symbol: str, start_timestamp: int, end_timestamp: int, 
                timeframe: str = "1h", **kwargs) -> Tuple[bool, pd.DataFrame]:
        """Get data for a symbol"""
        pass
    
    @abstractmethod
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        pass


class BaseDataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, asset_type: str, sma_periods: List[int] = None):
        """Initialize data processor"""
        self.asset_type = asset_type
        self.processor = TimeSeriesProcessor(sma_periods)
    
    @abstractmethod
    def get_data(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, **kwargs) -> pd.DataFrame:
        """Get and process data for a symbol"""
        pass


# ================ Utility Functions ================

def parse_target_symbols(filepath: str, target_section: str = "###TARGETS") -> List[str]:
    """Parse target symbols from file"""
    if not os.path.exists(filepath):
        print(f"Target file not found: {filepath}")
        return []

    targets = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    target_section_found = False
    for line in lines:
        if target_section in line:
            target_section_found = True
            continue
        if target_section_found and line.strip():
            symbols = line.strip().split(',')
            targets.extend([s.split(':')[-1].replace('USDT.P', '').strip()
                        for s in symbols if s.strip()])

    if not targets:
        print(f"No valid targets found in {filepath}")
        return []

    return targets


def create_output_directory(base_dir: str, prefix: str = "") -> str:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    output_path = os.path.join(base_dir, dir_name)
    FileManager.ensure_directories(output_path)
    return output_path


def get_period_overlap(period1: Tuple[datetime, datetime], period2: Tuple[datetime, datetime]) -> bool:
    """Check if two time periods overlap"""
    start1, end1 = period1
    start2, end2 = period2
    
    return (start1 <= end2) and (start2 <= end1)


def filter_non_overlapping_results(results: List[Dict], global_filtering: bool = True) -> List[Dict]:
    """Filter results to keep only non-overlapping periods with best scores"""
    if not results:
        return []
    
    if global_filtering:
        # Global filtering: no overlaps across all symbols
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        selected_results = []
        
        for result in sorted_results:
            window_data = result.get('window_data')
            if window_data is None:
                continue
            
            current_period = (window_data.index[0], window_data.index[-1])
            
            # Check overlap with already selected periods
            has_overlap = False
            for selected_result in selected_results:
                selected_data = selected_result.get('window_data')
                if selected_data is None:
                    continue
                selected_period = (selected_data.index[0], selected_data.index[-1])
                
                if get_period_overlap(current_period, selected_period):
                    has_overlap = True
                    break
            
            # If no overlap, add to selected
            if not has_overlap:
                selected_results.append(result)
        
        return selected_results
    
    else:
        # Per-symbol filtering: filter within each symbol independently
        symbol_groups = {}
        for result in results:
            symbol = result.get('symbol', 'unknown')
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(result)
        
        # Filter each symbol group independently
        all_selected_results = []
        for symbol, symbol_results in symbol_groups.items():
            sorted_symbol_results = sorted(symbol_results, key=lambda x: x['similarity'], reverse=True)
            symbol_selected_results = []
            
            for result in sorted_symbol_results:
                window_data = result.get('window_data')
                if window_data is None:
                    continue
                
                current_period = (window_data.index[0], window_data.index[-1])
                
                # Check overlap with already selected periods for this symbol
                has_overlap = False
                for selected_result in symbol_selected_results:
                    selected_data = selected_result.get('window_data')
                    if selected_data is None:
                        continue
                    selected_period = (selected_data.index[0], selected_data.index[-1])
                    
                    if get_period_overlap(current_period, selected_period):
                        has_overlap = True
                        break
                
                # If no overlap, add to selected for this symbol
                if not has_overlap:
                    symbol_selected_results.append(result)
            
            # Add all selected results from this symbol to the overall list
            all_selected_results.extend(symbol_selected_results)
        
        # Sort the final list by similarity score
        all_selected_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_selected_results


# ================ Export Functions ================

__all__ = [
    'TrendAnalysisConfig',
    'DataNormalizer',
    'TimeSeriesProcessor',
    'DTWCalculator',
    'FileManager',
    'DataCacheManager',
    'ReferenceDataManager',
    'BaseDataDownloader',
    'BaseDataProcessor',
    'plot_candlesticks_with_volume',
    'calculate_timeframe_seconds',
    'convert_datetime_to_timestamp',
    'format_dt_with_tz',
    'parse_target_symbols',
    'create_output_directory',
    'get_period_overlap',
    'filter_non_overlapping_results',
]