"""
Stock Screener Module
=====================
Modularized stock screening functionality for relative strength analysis.
Can be used both as a standalone script and as an importable module.

Features:
- Calculate relative strength (RS) scores using hourly data
- Apply Minervini trend template conditions
- Process stocks in parallel for efficiency
- Compare stocks against SPY benchmark
- Export results in multiple formats
"""

import os
import time
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from src.downloader import StockDownloader


# =============== Global Constants ================
MINERVINI_LOOKBACK_DAYS = 252  # Lookback period for Minervini conditions   
ONE_HOUR_LOOKBACK_DAYS = 7         # Lookback period for 1-hour data 
MIN_TURNOVER = 10000000         # Minimum average daily turnover


# ================ Configuration Classes ================

@dataclass
class ScreenerConfig:
    """Configuration class for stock screener parameters"""
    days_traceback_1d: int = MINERVINI_LOOKBACK_DAYS
    days_traceback_1h: int = ONE_HOUR_LOOKBACK_DAYS
    min_turnover: float = MIN_TURNOVER
    num_processes: Optional[int] = None
    timeout_seconds: int = 10
    
    def __post_init__(self):
        """Set default number of processes"""
        if self.num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)


@dataclass
class ScreeningResult:
    """Data class for screening results"""
    stock: str
    status: str
    rs_score: Optional[float] = None
    reason: Optional[str] = None
    
    def is_success(self) -> bool:
        """Check if screening was successful"""
        return self.status == "success"


# ================ Core Screening Functions ================

def calculate_rs_score(hourly_data: pd.DataFrame, ticker: str = "unknown") -> Tuple[bool, float, str]:
    """
    Calculate RS score from hourly data without using Z-score normalization.
    
    The RS score is calculated as a weighted sum of relative strength indicators,
    with newer data given higher weight. ATR is used for normalization to allow
    comparison across different stocks.
    
    Args:
        hourly_data: DataFrame with hourly price and indicator data
        ticker: Stock ticker symbol for logging
        
    Returns:
        Tuple of (success, rs_score, error_message)
    """
    config = ScreenerConfig()
    required_bars = config.days_traceback_1h * 8
    
    # Check if we have enough data
    if len(hourly_data) < required_bars:
        return False, 0, f"Insufficient hourly data: {len(hourly_data)} < {required_bars}"
        
    # Take the most recent required_bars data points
    data = hourly_data.tail(required_bars).reset_index(drop=True)
    
    # Calculate RS Score
    rs_score = 0.0
    total_weight = 0.0
    
    # Calculate for each data point
    for i in range(required_bars):
        # Current data point values
        current_close = data['close'].iloc[i]
        moving_average_30 = data['sma_30'].iloc[i]
        moving_average_45 = data['sma_45'].iloc[i]
        moving_average_60 = data['sma_60'].iloc[i]
        current_atr = data['atr'].iloc[i]
        
        # Calculate relative strength numerator
        numerator = ((current_close - moving_average_30) +
                     (current_close - moving_average_45) +
                     (current_close - moving_average_60) +
                     (moving_average_30 - moving_average_45) +
                     (moving_average_30 - moving_average_60) +
                     (moving_average_45 - moving_average_60))
        
        # Use ATR as denominator with small epsilon to avoid division by zero
        denominator = current_atr + 0.001
        
        # Calculate relative strength for this point
        relative_strength = numerator / denominator
        
        # Gives higher importance to newer data
        # k = 2 * np.log(2) / required_bars   
        # weight = np.exp(k * i)              # Exponential weight where w(L/2) * 2 = w(L)
        weight = 1
        
        # Add to weighted sum
        rs_score += relative_strength * weight
        total_weight += weight
    
    # Normalize the final score by total weight
    if total_weight > 0:
        rs_score = rs_score / total_weight
    else:
        return False, 0, "Weight calculation error"

    return True, rs_score, ""


def calculate_spy_rs_score(config: ScreenerConfig = None) -> float:
    """
    Calculate SPY's RS score for benchmark comparison.
    
    Args:
        config: Screener configuration object
        
    Returns:
        SPY's RS score
        
    Raises:
        ValueError: If SPY data cannot be retrieved or processed
    """
    if config is None:
        config = ScreenerConfig()
        
    sd = StockDownloader()
    print("Processing SPY RS score calculation...")

    # Request more data than needed to ensure we have enough after filtering
    now = int(time.time())
    buffer_days = int(config.days_traceback_1h * 2)  # 200% buffer for safety
    hourly_start_ts = now - (buffer_days * 24 * 3600)

    success, buffer_hourly_data = sd.get_data("SPY", hourly_start_ts, end_ts=now, timeframe="1h", atr=True)
    if not success or buffer_hourly_data is None:
        raise ValueError("Failed to get hourly data for SPY")
    if len(buffer_hourly_data) < config.days_traceback_1h * 8:
        raise ValueError(f"Insufficient hourly data for SPY: {len(buffer_hourly_data)} < {config.days_traceback_1h * 8}")
    
    hourly_data = buffer_hourly_data.tail(config.days_traceback_1h * 8).reset_index(drop=True)

    success, rs_score, error = calculate_rs_score(hourly_data, "SPY")
    if not success:
        raise ValueError(f"Failed to calculate SPY RS score: {error}")

    print(f"Finished SPY -> RS Score {rs_score}")
    return rs_score


def check_minervini_conditions(daily_data: pd.DataFrame) -> Tuple[bool, List[int]]:
    """
    Check Minervini trend template conditions.
    
    Args:
        daily_data: DataFrame with daily price and indicator data
        
    Returns:
        Tuple of (all_conditions_passed, failed_conditions)
    """
    # Get required values for trend template
    current_close = daily_data['close'].values[-1]
    moving_average_50 = daily_data['sma_50'].values[-1]
    moving_average_60 = daily_data['sma_60'].values[-1]
    moving_average_150 = daily_data['sma_150'].values[-1]
    moving_average_200 = daily_data['sma_200'].values[-1]

    # Calculate high/low using configured lookback period
    low_of_period = daily_data["close"].min()
    high_of_period = daily_data["close"].max()

    # Check Minervini trend template conditions
    conditions = [
        (current_close > moving_average_150 and current_close > moving_average_200),  # Condition 1
        moving_average_150 > moving_average_200,  # Condition 2
        True,  # Condition 3 (assumed true as per original)
        moving_average_50 > moving_average_150 > moving_average_200,  # Condition 4
        True,  # Condition 5 (assumed true as per original)
        current_close > low_of_period * 1.3,  # Condition 6
        current_close > high_of_period * 0.75,  # Condition 7
        True,  # Condition 8 (assumed true as per original)
        current_close >= 10  # Condition 9
    ]

    all_passed = all(conditions)
    failed_conditions = [i + 1 for i, cond in enumerate(conditions) if not cond]
    
    return all_passed, failed_conditions


def calc_relative_strength(ticker: str, use_template: bool, config: ScreenerConfig = None) -> ScreeningResult:
    """
    Calculate relative strength and check trend template conditions for a given stock ticker.

    Args:
        ticker: The stock ticker symbol
        use_template: Flag to determine whether to apply Minervini trend template conditions
        config: Screener configuration object

    Returns:
        ScreeningResult object containing the analysis results
    """
    if config is None:
        config = ScreenerConfig()

    print(f"Processing {ticker}...")
    sd = StockDownloader()
    now = int(time.time())

    # Request more data than needed for daily timeframe
    buffer_days = int(config.days_traceback_1d * 2)  
    daily_start_ts = now - (buffer_days * 24 * 3600)
    success, buffer_daily_data = sd.get_data(ticker, daily_start_ts, end_ts=now, timeframe="1d", dropna=False, atr=False)

    if not success or buffer_daily_data is None:
        msg = "No daily data"
        print(f"Finished {ticker} -> Failed: {msg}")
        return ScreeningResult(stock=ticker, status="failed", reason=msg)

    # Take the most recent required days
    if len(buffer_daily_data) < config.days_traceback_1d:
        msg = f"Insufficient daily data: {len(buffer_daily_data)} < {config.days_traceback_1d}"
        print(f"Finished {ticker} -> Failed: {msg}")
        return ScreeningResult(stock=ticker, status="failed", reason=msg)

    daily_data = buffer_daily_data.tail(config.days_traceback_1d).reset_index(drop=True)

    # Check turnover
    last_10_days = daily_data.tail(10)
    average_turnover = (last_10_days['volume'] * last_10_days['close']).mean()
    if average_turnover < config.min_turnover:
        msg = "Insufficient turnover"
        print(f"Finished {ticker} -> Failed: {msg}")
        return ScreeningResult(stock=ticker, status="failed", reason=msg)

    # Check Minervini trend template conditions if requested
    if use_template:
        conditions_passed, failed_conditions = check_minervini_conditions(daily_data)
        if not conditions_passed:
            msg = f"Failed conditions: {failed_conditions}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return ScreeningResult(stock=ticker, status="failed", reason=msg)

    # Get hourly data with buffer for RS score calculation
    buffer_days = int(config.days_traceback_1h * 2)
    hourly_start_ts = now - (buffer_days * 24 * 3600)
    success, buffer_hourly_data = sd.get_data(ticker, hourly_start_ts, end_ts=now, timeframe="1h", atr=True)

    if not success or buffer_hourly_data is None:
        msg = "No hourly data"
        print(f"Finished {ticker} -> Failed: {msg}")
        return ScreeningResult(stock=ticker, status="failed", reason=msg)
    
    if len(buffer_hourly_data) < config.days_traceback_1h * 8:
        msg = f"Insufficient hourly data: {len(buffer_hourly_data)} < {config.days_traceback_1h * 8}"
        print(f"Finished {ticker} -> Failed: {msg}")
        return ScreeningResult(stock=ticker, status="failed", reason=msg)
    
    hourly_data = buffer_hourly_data.tail(config.days_traceback_1h * 8).reset_index(drop=True)

    success, rs_score, error = calculate_rs_score(hourly_data, ticker)
    if not success:
        print(f"Finished {ticker} -> Failed: {error}")
        return ScreeningResult(stock=ticker, status="failed", reason=error)

    print(f"Finished {ticker} -> RS Score {rs_score}")
    return ScreeningResult(stock=ticker, status="success", rs_score=rs_score)


# ================ Main Screening Class ================

class StockScreener:
    """
    Main stock screening class that orchestrates the screening process.
    """
    
    def __init__(self, config: ScreenerConfig = None):
        """
        Initialize the stock screener.
        
        Args:
            config: Screener configuration object
        """
        self.config = config or ScreenerConfig()
        self.downloader = StockDownloader()
        
    def get_all_tickers(self) -> List[str]:
        """Get all available stock tickers"""
        return self.downloader.get_all_tickers()
    
    def screen_stocks(self, tickers: List[str] = None, use_template: bool = True, 
                     spy_rs_score: float = None) -> Dict:
        """
        Screen stocks for relative strength.
        
        Args:
            tickers: List of stock tickers to screen (if None, uses all available tickers)
            use_template: Whether to apply Minervini trend template conditions
            spy_rs_score: Pre-calculated SPY RS score (if None, will calculate)
            
        Returns:
            Dictionary containing screening results
        """
        # Get tickers if not provided
        if tickers is None:
            tickers = self.get_all_tickers()
        
        print(f"Total tickers to process: {len(tickers)}")
        
        # Calculate SPY's RS score if not provided
        if spy_rs_score is None:
            try:
                spy_rs_score = calculate_spy_rs_score(self.config)
                print(f"SPY RS Score: {spy_rs_score}")
            except Exception as e:
                print(f"Failed to calculate SPY RS score: {e}")
                raise
        
        # Process all tickers using ProcessPoolExecutor
        print(f"Using {self.config.num_processes} processes")
        
        # Process results
        strong_targets = []
        target_rs_score = {}
        failed_tickers = []
        
        with ProcessPoolExecutor(max_workers=self.config.num_processes) as executor:
            futures = {
                executor.submit(calc_relative_strength, ticker, use_template, self.config): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result.is_success() and result.rs_score >= spy_rs_score:
                        strong_targets.append(ticker)
                        target_rs_score[ticker] = result.rs_score
                    elif not result.is_success():
                        failed_tickers.append((ticker, result.reason))
                except TimeoutError:
                    print(f"{ticker} took too long to process")
                    failed_tickers.append((ticker, "Timeout"))
                except Exception as e:
                    failed_tickers.append((ticker, str(e)))
        
        # Sort by RS score
        strong_targets.sort(key=lambda x: target_rs_score[x], reverse=True)
        
        return {
            'spy_rs_score': spy_rs_score,
            'strong_targets': strong_targets,
            'target_rs_scores': target_rs_score,
            'failed_tickers': failed_tickers,
            'total_processed': len(tickers),
            'use_template': use_template
        }
    
    def save_results(self, results: Dict, output_all: bool = False, 
                    no_conditions: bool = False) -> Tuple[str, str]:
        """
        Save screening results to files.
        
        Args:
            results: Results dictionary from screen_stocks
            output_all: Whether to include all strong targets or limit to top 980
            no_conditions: Whether Minervini conditions were ignored
            
        Returns:
            Tuple of (txt_file_path, failed_file_path)
        """
        # Prepare file content
        full_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        date_str = datetime.now().strftime("%Y-%m-%d")
        txt_content = "###INDEX\nSPY,QQQ,DJI\n###TARGETS\n"
        
        # Use all strong targets or just top 980 based on flag
        strong_targets = results['strong_targets']
        topk = 980 if len(strong_targets) > 980 else len(strong_targets)
        output_targets = strong_targets if output_all else strong_targets[:topk]
        txt_content += ",".join(output_targets)
        
        # Create output directory structure
        base_folder = "stock_rs_output"
        date_folder = os.path.join(base_folder, date_str)
        os.makedirs(date_folder, exist_ok=True)
        
        # Create output files with full timestamp in filename
        without_conditions = "_no_conditions" if no_conditions else ""
        for_tv = "all" if output_all else f"top{topk}"
        output_file = f"{full_date_str}_stock_{for_tv}{without_conditions}_strong_targets.txt"
        file_path = os.path.join(date_folder, output_file)
        
        with open(file_path, "w") as f:
            f.write(txt_content)
        
        return file_path, None  # No longer saving failed tickers file


# ================ Results Analysis Functions ================

def analyze_results(results: Dict) -> str:
    """
    Analyze and format screening results.
    
    Args:
        results: Results dictionary from screen_stocks
        
    Returns:
        Formatted analysis string
    """
    total_analyzed = results['total_processed'] - len(results['failed_tickers'])
    success_rate = len(results['strong_targets']) / total_analyzed * 100 if total_analyzed > 0 else 0
    
    analysis = [
        f"\nAnalysis Results:",
        f"Total tickers processed: {results['total_processed']}",
        f"Failed tickers: {len(results['failed_tickers'])}",
        f"Found {len(results['strong_targets'])} stocks that meet requirements and are stronger than SPY",
        f"Success rate: {success_rate:.2f}%",
        f"\nSPY RS Score: {results['spy_rs_score']:.4f}",
        f"Strong targets: {', '.join(results['strong_targets'][:50])}",  # Show top 50 only
        f"\n====== Top 50 Targets by RS Score ======"
    ]
    
    # Add top 50 targets with scores
    for ticker in results['strong_targets'][:50]:
        score = results['target_rs_scores'][ticker]
        analysis.append(f"{ticker}: {score:.4f}")
    
    analysis.append("=======================================")
    
    return '\n'.join(analysis)

# ================ Module Interface Functions ================

def screen_stocks_simple(tickers: List[str] = None, use_template: bool = True, 
                        config: ScreenerConfig = None) -> Dict:
    """
    Simple interface function for screening stocks.
    
    Args:
        tickers: List of stock tickers to screen (if None, uses all available)
        use_template: Whether to apply Minervini trend template conditions
        config: Screener configuration object
        
    Returns:
        Dictionary containing screening results
    """
    screener = StockScreener(config)
    return screener.screen_stocks(tickers=tickers, use_template=use_template)


def get_strong_stocks(spy_rs_threshold: float = None, use_template: bool = True,
                     config: ScreenerConfig = None) -> List[str]:
    """
    Get list of stocks stronger than SPY (or custom threshold).
    
    Args:
        spy_rs_threshold: Custom RS threshold (if None, uses SPY's current RS score)
        use_template: Whether to apply Minervini trend template conditions
        config: Screener configuration object
        
    Returns:
        List of strong stock tickers
    """
    screener = StockScreener(config)
    
    # Calculate SPY RS score if no threshold provided
    if spy_rs_threshold is None:
        spy_rs_threshold = calculate_spy_rs_score(config)
    
    results = screener.screen_stocks(use_template=use_template, spy_rs_score=spy_rs_threshold)
    return results['strong_targets']

# ================ Command Line Interface ================

def main():
    """Main function for command line usage"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Trend Analysis')
    parser.add_argument('-a', '--all', action='store_true', help='Include all strong targets in output')
    parser.add_argument('-g', action='store_true', help='Ignore Minervini conditions and calculate RS score only')
    args = parser.parse_args()
    
    # Initialize screener
    screener = StockScreener()
    
    # Run screening
    use_template = not args.g
    results = screener.screen_stocks(use_template=use_template)
    
    # Analyze and print results
    analysis = analyze_results(results)
    print(analysis)
    
    # Save results to files
    file_path, _ = screener.save_results(results, output_all=args.all, no_conditions=args.g)
    print(f"\nResults saved to {file_path}")
    count = 'all' if args.all else f"top {min(980, len(results['strong_targets']))}"
    print(f"Included {count} strong targets in output file")

# ================ Export ================

__all__ = [
    'StockScreener',
    'ScreenerConfig', 
    'ScreeningResult',
    'calculate_rs_score',
    'calculate_spy_rs_score',
    'check_minervini_conditions',
    'calc_relative_strength',
    'screen_stocks_simple',
    'get_strong_stocks',
    'analyze_results'
]


if __name__ == '__main__':
    main()