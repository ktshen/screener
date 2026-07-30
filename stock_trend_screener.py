"""
Stock Volatility and Volume Screener
====================================
Screens stocks based on 1-minute bar volatility and volume metrics using 
previous trading day's data.

Criteria:
- Average 1-minute bar volatility >= 0.3%
- Average 1-minute bar volume >= 100,000
- Results sorted by average volume (descending)
"""

import os
import time
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.common import FileManager
from src.downloader import StockDownloader


# =============== Global Constants ================
MIN_AVG_VOLATILITY = 0.003  # 0.3% minimum average volatility per 1-minute bar
MIN_AVG_VOLUME = 100000     # Minimum average volume per 1-minute bar


# ================ Configuration Classes ================

@dataclass
class VolatilityScreenerConfig:
    """Configuration for volatility and volume screener"""
    min_avg_volatility: float = MIN_AVG_VOLATILITY
    min_avg_volume: float = MIN_AVG_VOLUME
    num_processes: Optional[int] = None
    timeout_seconds: int = 15
    lookback_days: int = 5  # Look back a few days to ensure we get previous trading day
    
    def __post_init__(self):
        """Set default number of processes"""
        if self.num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)


@dataclass
class ScreeningResult:
    """Data class for screening results"""
    stock: str
    status: str
    avg_volatility: Optional[float] = None
    avg_volume: Optional[float] = None
    total_bars: Optional[int] = None
    reason: Optional[str] = None
    
    def is_success(self) -> bool:
        """Check if screening was successful"""
        return self.status == "success"
    
    def meets_criteria(self, config: VolatilityScreenerConfig) -> bool:
        """Check if stock meets screening criteria"""
        if not self.is_success():
            return False
        return (self.avg_volatility >= config.min_avg_volatility and 
                self.avg_volume >= config.min_avg_volume)


# ================ Core Analysis Functions ================

def calculate_bar_volatility(open_price: float, high: float, low: float, close_price: float) -> float:
    """
    Calculate volatility for a single bar as percentage range.
    
    Volatility = (High - Low) / Open * 100
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price  
        close_price: Closing price
        
    Returns:
        Volatility as a percentage (e.g., 0.005 = 0.5%)
    """
    if open_price <= 0:
        return 0.0
    return (high - low) / open_price


def analyze_previous_day_metrics(minute_data: pd.DataFrame, ticker: str = "unknown") -> Tuple[bool, float, float, int, str]:
    """
    Analyze previous trading day's 1-minute bar data for volatility and volume.
    
    Args:
        minute_data: DataFrame with 1-minute OHLCV data
        ticker: Stock ticker for logging
        
    Returns:
        Tuple of (success, avg_volatility, avg_volume, total_bars, error_message)
    """
    if minute_data.empty:
        return False, 0.0, 0.0, 0, "No data available"
    
    # Convert timestamp to datetime for easier date filtering
    minute_data['datetime'] = pd.to_datetime(minute_data['timestamp'], unit='s', utc=True)
    minute_data['datetime'] = minute_data['datetime'].dt.tz_convert('America/New_York')
    minute_data['date'] = minute_data['datetime'].dt.date
    
    # Get unique trading dates
    unique_dates = sorted(minute_data['date'].unique(), reverse=True)
    
    if len(unique_dates) < 1:
        return False, 0.0, 0.0, 0, "No trading days found"
    
    # Get the most recent complete trading day (not today)
    today = datetime.now().date()
    previous_trading_days = [d for d in unique_dates if d < today]
    
    if len(previous_trading_days) < 1:
        return False, 0.0, 0.0, 0, "No previous trading day data available"
    
    previous_day = previous_trading_days[0]
    
    # Filter data for previous trading day
    previous_day_data = minute_data[minute_data['date'] == previous_day].copy()
    
    if len(previous_day_data) < 100:  # Require at least 100 bars (less than 2 hours suggests incomplete data)
        return False, 0.0, 0.0, len(previous_day_data), f"Insufficient bars for previous day: {len(previous_day_data)}"
    
    # Calculate volatility for each bar
    previous_day_data['volatility'] = previous_day_data.apply(
        lambda row: calculate_bar_volatility(row['open'], row['high'], row['low'], row['close']),
        axis=1
    )
    
    # Calculate average metrics
    avg_volatility = previous_day_data['volatility'].mean()
    avg_volume = previous_day_data['volume'].mean()
    total_bars = len(previous_day_data)
    
    return True, avg_volatility, avg_volume, total_bars, ""


def screen_single_stock(ticker: str, config: VolatilityScreenerConfig = None) -> ScreeningResult:
    """
    Screen a single stock for volatility and volume criteria.
    
    Args:
        ticker: Stock ticker symbol
        config: Screener configuration
        
    Returns:
        ScreeningResult object
    """
    if config is None:
        config = VolatilityScreenerConfig()
    
    try:
        sd = StockDownloader()
        
        # Calculate time range for data request
        now = int(time.time())
        start_ts = now - (config.lookback_days * 24 * 3600)
        
        # Get 1-minute data
        success, minute_data = sd.get_data(
            ticker, 
            start_ts, 
            end_ts=now, 
            timeframe="1m",
            dropna=False,
            atr=False,
            vwap=False,
            validate=True
        )
        
        if not success or minute_data is None or minute_data.empty:
            return ScreeningResult(
                stock=ticker,
                status="failed",
                reason="Failed to retrieve 1-minute data"
            )
        
        # Analyze previous day's metrics
        success, avg_volatility, avg_volume, total_bars, error = analyze_previous_day_metrics(
            minute_data, ticker
        )
        
        if not success:
            return ScreeningResult(
                stock=ticker,
                status="failed",
                reason=error
            )
        
        # Check if meets criteria
        meets_criteria = (avg_volatility >= config.min_avg_volatility and 
                         avg_volume >= config.min_avg_volume)
        
        status = "success" if meets_criteria else "below_threshold"
        
        print(f"{ticker} -> Avg Vol: {avg_volatility*100:.3f}%, Avg Volume: {avg_volume:,.0f}, Bars: {total_bars}, Status: {status}")
        
        return ScreeningResult(
            stock=ticker,
            status=status,
            avg_volatility=avg_volatility,
            avg_volume=avg_volume,
            total_bars=total_bars,
            reason=None if meets_criteria else "Below threshold"
        )
        
    except Exception as e:
        print(f"{ticker} -> Error: {e}")
        return ScreeningResult(
            stock=ticker,
            status="failed",
            reason=str(e)
        )


# ================ Batch Screening Class ================

class VolatilityVolumeScreener:
    """Main screener class for volatility and volume analysis"""
    
    def __init__(self, config: VolatilityScreenerConfig = None):
        """Initialize screener with configuration"""
        self.config = config if config is not None else VolatilityScreenerConfig()
        self.downloader = StockDownloader()
    
    def get_all_tickers(self) -> List[str]:
        """Get all available stock tickers"""
        return self.downloader.get_all_tickers()
    
    def screen_stocks(self, tickers: List[str] = None) -> Dict:
        """
        Screen stocks for volatility and volume criteria.
        
        Args:
            tickers: List of tickers to screen (if None, screens all available)
            
        Returns:
            Dictionary containing screening results
        """
        # Get tickers to process
        if tickers is None:
            tickers = self.get_all_tickers()
        
        print(f"Total tickers to process: {len(tickers)}")
        print(f"Criteria: Avg Volatility >= {self.config.min_avg_volatility*100:.2f}%, Avg Volume >= {self.config.min_avg_volume:,.0f}")
        print(f"Using {self.config.num_processes} processes\n")
        
        # Process results
        qualified_stocks = []
        all_results = []
        failed_tickers = []
        
        with ProcessPoolExecutor(max_workers=self.config.num_processes) as executor:
            futures = {
                executor.submit(screen_single_stock, ticker, self.config): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    
                    if result.is_success() and result.meets_criteria(self.config):
                        qualified_stocks.append(result)
                    
                    if result.is_success():
                        all_results.append(result)
                    else:
                        failed_tickers.append((ticker, result.reason))
                        
                except TimeoutError:
                    print(f"{ticker} -> Timeout")
                    failed_tickers.append((ticker, "Timeout"))
                except Exception as e:
                    print(f"{ticker} -> Exception: {e}")
                    failed_tickers.append((ticker, str(e)))
        
        # Sort qualified stocks by average volume (descending)
        qualified_stocks.sort(key=lambda x: x.avg_volume, reverse=True)
        
        return {
            'qualified_stocks': qualified_stocks,
            'all_results': all_results,
            'failed_tickers': failed_tickers,
            'total_processed': len(tickers),
            'config': self.config
        }
    
    def save_results(self, results: Dict) -> Tuple[str, str]:
        """
        Save screening results to files.
        
        Args:
            results: Results dictionary from screen_stocks
            
        Returns:
            Tuple of (summary_file_path, detailed_file_path)
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        base_folder = "stock_volatility_output"
        date_folder = os.path.join(base_folder, date_str)
        FileManager.ensure_directories(date_folder)
        
        # Prepare summary file
        summary_file = f"{timestamp}_qualified_stocks.txt"
        summary_path = os.path.join(date_folder, summary_file)
        
        with open(summary_path, 'w') as f:
            f.write("###TARGETS\n")
            tickers = [result.stock for result in results['qualified_stocks']]
            f.write(",".join(tickers))
        
        # Prepare detailed CSV file
        detailed_file = f"{timestamp}_detailed_results.csv"
        detailed_path = os.path.join(date_folder, detailed_file)
        
        detailed_data = []
        for result in results['qualified_stocks']:
            detailed_data.append({
                'Ticker': result.stock,
                'Avg_Volatility_Pct': f"{result.avg_volatility * 100:.3f}",
                'Avg_Volume': f"{result.avg_volume:,.0f}",
                'Total_Bars': result.total_bars
            })
        
        if detailed_data:
            df = pd.DataFrame(detailed_data)
            df.to_csv(detailed_path, index=False)
        
        return summary_path, detailed_path


# ================ Analysis Functions ================

def analyze_results(results: Dict) -> str:
    """
    Analyze and format screening results.
    
    Args:
        results: Results dictionary from screen_stocks
        
    Returns:
        Formatted analysis string
    """
    config = results['config']
    qualified = results['qualified_stocks']
    
    analysis = [
        f"\n{'='*60}",
        f"Volatility & Volume Screening Results",
        f"{'='*60}",
        f"Criteria:",
        f"  - Min Average Volatility: {config.min_avg_volatility*100:.2f}%",
        f"  - Min Average Volume: {config.min_avg_volume:,.0f}",
        f"\nResults:",
        f"  - Total Processed: {results['total_processed']}",
        f"  - Qualified Stocks: {len(qualified)}",
        f"  - Failed/Insufficient Data: {len(results['failed_tickers'])}",
        f"\n{'='*60}",
        f"Top 50 Stocks (Sorted by Average Volume):",
        f"{'='*60}",
    ]
    
    # Display top 50 qualified stocks
    for i, result in enumerate(qualified[:50], 1):
        analysis.append(
            f"{i:2d}. {result.stock:6s} | "
            f"Vol: {result.avg_volatility*100:5.2f}% | "
            f"Avg Volume: {result.avg_volume:>12,.0f} | "
            f"Bars: {result.total_bars:>3d}"
        )
    
    analysis.append(f"{'='*60}\n")
    
    return '\n'.join(analysis)


# ================ Module Interface Functions ================

def screen_stocks_simple(tickers: List[str] = None, 
                        config: VolatilityScreenerConfig = None) -> Dict:
    """
    Simple interface for screening stocks.
    
    Args:
        tickers: List of tickers to screen
        config: Screener configuration
        
    Returns:
        Dictionary containing screening results
    """
    screener = VolatilityVolumeScreener(config)
    return screener.screen_stocks(tickers=tickers)


def get_qualified_stocks(min_volatility: float = MIN_AVG_VOLATILITY,
                        min_volume: float = MIN_AVG_VOLUME) -> List[str]:
    """
    Get list of stocks meeting volatility and volume criteria.
    
    Args:
        min_volatility: Minimum average volatility (as decimal, e.g., 0.003 for 0.3%)
        min_volume: Minimum average volume
        
    Returns:
        List of qualified stock tickers sorted by volume
    """
    config = VolatilityScreenerConfig(
        min_avg_volatility=min_volatility,
        min_avg_volume=min_volume
    )
    screener = VolatilityVolumeScreener(config)
    results = screener.screen_stocks()
    return [result.stock for result in results['qualified_stocks']]


# ================ Command Line Interface ================

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='Screen stocks based on 1-minute volatility and volume'
    )
    parser.add_argument(
        '-v', '--min-volatility',
        type=float,
        default=MIN_AVG_VOLATILITY,
        help=f'Minimum average volatility as decimal (default: {MIN_AVG_VOLATILITY} = 0.3%%)'
    )
    parser.add_argument(
        '-vol', '--min-volume',
        type=float,
        default=MIN_AVG_VOLUME,
        help=f'Minimum average volume (default: {MIN_AVG_VOLUME:,.0f})'
    )
    parser.add_argument(
        '-t', '--tickers',
        nargs='+',
        help='Specific tickers to screen (if not provided, screens all)'
    )
    
    args = parser.parse_args()
    
    # Create config with custom parameters
    config = VolatilityScreenerConfig(
        min_avg_volatility=args.min_volatility,
        min_avg_volume=args.min_volume
    )
    
    # Initialize and run screener
    screener = VolatilityVolumeScreener(config)
    results = screener.screen_stocks(tickers=args.tickers)
    
    # Analyze and print results
    analysis = analyze_results(results)
    print(analysis)
    
    # Save results
    summary_path, detailed_path = screener.save_results(results)
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {detailed_path}")


# ================ Export ================

__all__ = [
    'VolatilityVolumeScreener',
    'VolatilityScreenerConfig',
    'ScreeningResult',
    'calculate_bar_volatility',
    'analyze_previous_day_metrics',
    'screen_single_stock',
    'screen_stocks_simple',
    'get_qualified_stocks',
    'analyze_results'
]


if __name__ == '__main__':
    main()