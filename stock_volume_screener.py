"""
Volume-based Stock Screener
Identifies stocks with significantly increased trading volume compared to previous trading day
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
from pathlib import Path

# Assuming these imports work from your project structure
from src.downloader import StockDownloader


# ================= CONFIGURATIONS =================#
INTRADAY_VOLUME_THRESHOLD_RATIO = 3.0  
DAILY_VOLUME_THRESHOLD_RATIO = 3.0     
INTRADAY_MIN_TURNOVER_USD = 1000000  
DAILY_MIN_TURNOVER_USD = 5000000 
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
NY_TIMEZONE = 'America/New_York'
# ==================================================#


def is_market_open(current_time: datetime = None) -> bool:
    """
    Check if the US stock market is currently open
    
    Args:
        current_time: datetime object in NY timezone (default: current time)
    
    Returns:
        bool: True if market is open, False otherwise
    """
    if current_time is None:
        ny_tz = timezone(NY_TIMEZONE)
        current_time = datetime.now(ny_tz)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if current_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's within market hours (9:30 AM - 4:00 PM)
    market_open = current_time.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = current_time.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    
    return market_open <= current_time < market_close


def get_last_trading_day(current_time: datetime) -> datetime:
    """
    Get the last trading day before the current time
    
    Args:
        current_time: datetime object in NY timezone
    
    Returns:
        datetime: Last trading day
    """
    # Start from yesterday
    last_day = current_time - timedelta(days=1)
    
    # Keep going back until we find a weekday
    while last_day.weekday() >= 5:  # Skip weekends
        last_day = last_day - timedelta(days=1)
    
    return last_day


def get_previous_trading_day(reference_day: datetime) -> datetime:
    """
    Get the trading day before the reference day
    
    Args:
        reference_day: datetime object in NY timezone
    
    Returns:
        datetime: Previous trading day
    """
    prev_day = reference_day - timedelta(days=1)
    
    # Keep going back until we find a weekday
    while prev_day.weekday() >= 5:  # Skip weekends
        prev_day = prev_day - timedelta(days=1)
    
    return prev_day


def calculate_accurate_turnover(data: pd.DataFrame) -> float:
    """
    Calculate accurate turnover (dollar volume) from OHLCV data
    Prefers VWAP if available, otherwise uses typical price (H+L+C)/3 * Volume
    
    Args:
        data: DataFrame with OHLC, volume, and optionally vwap columns
    
    Returns:
        float: Total turnover in USD
    """
    # Use VWAP if available (most accurate)
    if 'vwap' in data.columns and data['vwap'].notna().any():
        turnover = (data['vwap'] * data['volume']).sum()
    else:
        # Fall back to typical price for better accuracy than simple average
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        turnover = (typical_price * data['volume']).sum()
    return turnover


def check_intraday_volume_surge(ticker: str, volume_ratio: float = INTRADAY_VOLUME_THRESHOLD_RATIO) -> dict:
    """
    Check if a stock has volume surge during intraday trading
    Compares today's volume (9:30 - current time) with previous day's volume at same time
    
    Additional filters:
    - Today's opening price must be higher than previous day's price at the same time
    - Today's total turnover (volume * price) must exceed $1 million
    
    Args:
        ticker: Stock symbol
        volume_ratio: Minimum ratio for volume surge (default: 5.0)
    
    Returns:
        dict: Result containing status and volume information
    """
    print(f"Processing {ticker}...")
    
    try:
        sd = StockDownloader()
        ny_tz = timezone(NY_TIMEZONE)
        current_time = datetime.now(ny_tz)
        
        # Get current time in minutes since market open
        market_open_today = current_time.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, 
                                                  second=0, microsecond=0)
        minutes_since_open = int((current_time - market_open_today).total_seconds() / 60)
        
        # Round down to nearest 5-minute interval
        complete_intervals = minutes_since_open // 5
        
        if complete_intervals < 1:
            msg = "Market just opened, not enough data"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Calculate the end time for comparison (last complete 5-min bar)
        comparison_end_time = market_open_today + timedelta(minutes=complete_intervals * 5)
        
        # Get last trading day
        last_trading_day = get_last_trading_day(current_time)
        
        # Set time ranges for today
        today_start = market_open_today
        today_end = comparison_end_time
        today_start_ts = int(today_start.timestamp())
        today_end_ts = int(today_end.timestamp())
        
        # Set time ranges for previous day (same time window)
        prev_day_start = last_trading_day.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE,
                                                   second=0, microsecond=0)
        prev_day_end = prev_day_start + timedelta(minutes=complete_intervals * 5)
        prev_day_start_ts = int(prev_day_start.timestamp())
        prev_day_end_ts = int(prev_day_end.timestamp())
        
        # Get today's data (5-minute bars with VWAP)
        success_today, today_data = sd.get_data(
            ticker, 
            today_start_ts, 
            end_ts=today_end_ts,
            timeframe="5m",
            dropna=False,
            atr=False,
            vwap=True,  # Request VWAP for accurate turnover calculation
            validate=False
        )
        
        if not success_today or today_data.empty:
            msg = "No data for today"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Get previous day's data (5-minute bars with VWAP)
        success_prev, prev_data = sd.get_data(
            ticker,
            prev_day_start_ts,
            end_ts=prev_day_end_ts,
            timeframe="5m",
            dropna=False,
            atr=False,
            vwap=True,  # Request VWAP for accurate turnover calculation
            validate=False
        )
        
        if not success_prev or prev_data.empty:
            msg = "No data for previous day"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Verify data integrity - ensure data starts at 9:30 and doesn't go beyond 4:00 PM
        market_close_today = current_time.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE,
                                                   second=0, microsecond=0)
        market_close_today_ts = int(market_close_today.timestamp())
        
        # Filter today's data to ensure it's within valid range (9:30 - 4:00 PM)
        today_data = today_data[
            (today_data['timestamp'] >= today_start_ts) & 
            (today_data['timestamp'] <= market_close_today_ts)
        ]
        
        # Filter previous day's data to ensure it's within valid range
        market_close_prev = last_trading_day.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE,
                                                      second=0, microsecond=0)
        market_close_prev_ts = int(market_close_prev.timestamp())
        
        prev_data = prev_data[
            (prev_data['timestamp'] >= prev_day_start_ts) & 
            (prev_data['timestamp'] <= market_close_prev_ts)
        ]
        
        # Verify we still have data after filtering
        if today_data.empty or prev_data.empty:
            msg = "No valid data within market hours after filtering"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # New Filter 1: Check if today's opening price is higher than previous day's price at same time
        today_open_price = today_data['open'].iloc[0]  # First bar's open (9:30)
        today_close_price = today_data['close'].iloc[-1]  # Last bar's close at comparison time
        prev_comparison_price = prev_data['close'].iloc[-1]  # Last bar's close at comparison time
        
        if today_open_price <= prev_comparison_price:
            msg = f"Opening price ${today_open_price:.2f} not higher than prev day's price ${prev_comparison_price:.2f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # New Filter 1b: Check if today's closing price is also higher than previous day's price
        if today_close_price <= prev_comparison_price:
            msg = f"Closing price ${today_close_price:.2f} not higher than prev day's price ${prev_comparison_price:.2f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Calculate total volume for each period
        today_volume = today_data['volume'].sum()
        prev_volume = prev_data['volume'].sum()
        
        # New Filter 2: Check if today's turnover exceeds minimum threshold
        # Use typical price (H+L+C)/3 for more accurate turnover calculation
        today_turnover_usd = calculate_accurate_turnover(today_data)
        
        if today_turnover_usd < INTRADAY_MIN_TURNOVER_USD:
            msg = f"Turnover ${today_turnover_usd:,.0f} below minimum ${INTRADAY_MIN_TURNOVER_USD:,.0f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Check if today's volume meets the threshold
        if prev_volume == 0:
            msg = "Previous day volume is zero"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        actual_ratio = today_volume / prev_volume
        
        if actual_ratio >= volume_ratio:
            print(f"Finished {ticker} -> SUCCESS: Volume ratio {actual_ratio:.2f}x, "
                  f"Turnover ${today_turnover_usd:,.0f}, "
                  f"Open ${today_open_price:.2f} > Prev ${prev_comparison_price:.2f}, "
                  f"Close ${today_close_price:.2f} > Prev ${prev_comparison_price:.2f}")
            return {
                "stock": ticker,
                "status": "success",
                "today_volume": today_volume,
                "prev_volume": prev_volume,
                "volume_ratio": actual_ratio,
                "today_turnover_usd": today_turnover_usd,
                "today_open_price": today_open_price,
                "today_close_price": today_close_price,
                "prev_comparison_price": prev_comparison_price
            }
        else:
            msg = f"Volume ratio {actual_ratio:.2f}x below threshold {volume_ratio}x"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
            
    except Exception as e:
        print(f"Finished {ticker} -> Error: {str(e)}")
        return {"stock": ticker, "status": "failed", "reason": str(e)}


def check_daily_volume_surge(ticker: str, volume_ratio: float = DAILY_VOLUME_THRESHOLD_RATIO) -> dict:
    """
    Check if a stock has volume surge based on daily data (after market close)
    Compares last trading day's volume with previous trading day's volume
    
    Additional filters:
    - Last trading day's opening price must be higher than previous day's closing price
    - Last trading day's total turnover (volume * average price) must exceed $1 million
    
    Args:
        ticker: Stock symbol
        volume_ratio: Minimum ratio for volume surge (default: 3.0)
    
    Returns:
        dict: Result containing status and volume information
    """
    print(f"Processing {ticker}...")
    
    try:
        sd = StockDownloader()
        ny_tz = timezone(NY_TIMEZONE)
        current_time = datetime.now(ny_tz)
        
        # Get last trading day
        last_trading_day = get_last_trading_day(current_time)
        
        # Get previous trading day
        prev_trading_day = get_previous_trading_day(last_trading_day)
        
        # Calculate timestamps - we need enough data to get 2 complete daily bars
        # Add buffer to ensure we get the data
        end_ts = int(current_time.timestamp())
        start_ts = int((prev_trading_day - timedelta(days=3)).timestamp())
        
        # Get daily data with VWAP
        success, daily_data = sd.get_data(
            ticker,
            start_ts,
            end_ts=end_ts,
            timeframe="1d",
            dropna=False,
            atr=False,
            vwap=True,  # Request VWAP for accurate turnover calculation
            validate=False
        )
        
        if not success or daily_data.empty:
            msg = "No daily data available"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        if len(daily_data) < 2:
            msg = "Insufficient daily data (need at least 2 days)"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # Get the last two trading days data
        recent_open = daily_data['open'].iloc[-1]
        recent_close = daily_data['close'].iloc[-1]
        recent_volume = daily_data['volume'].iloc[-1]
        prev_close = daily_data['close'].iloc[-2]
        prev_volume = daily_data['volume'].iloc[-2]
        
        # New Filter 1: Check if recent day's opening price is higher than previous day's closing price
        if recent_open <= prev_close:
            msg = f"Opening price ${recent_open:.2f} not higher than prev close ${prev_close:.2f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # New Filter 1b: Check if recent day's closing price is also higher than previous day's closing price
        if recent_close <= prev_close:
            msg = f"Closing price ${recent_close:.2f} not higher than prev close ${prev_close:.2f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        # New Filter 2: Check if recent day's turnover exceeds minimum threshold
        # Use typical price for more accurate turnover calculation
        recent_day_data = daily_data.tail(1)
        recent_turnover_usd = calculate_accurate_turnover(recent_day_data)
        
        if recent_turnover_usd < DAILY_MIN_TURNOVER_USD:
            msg = f"Turnover ${recent_turnover_usd:,.0f} below minimum ${DAILY_MIN_TURNOVER_USD:,.0f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        if prev_volume == 0:
            msg = "Previous day volume is zero"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
        
        actual_ratio = recent_volume / prev_volume
        
        if actual_ratio >= volume_ratio:
            recent_date = datetime.fromtimestamp(daily_data['timestamp'].iloc[-1])
            prev_date = datetime.fromtimestamp(daily_data['timestamp'].iloc[-2])
            
            print(f"Finished {ticker} -> SUCCESS: Volume ratio {actual_ratio:.2f}x "
                  f"({recent_date.strftime('%Y-%m-%d')} vs {prev_date.strftime('%Y-%m-%d')}), "
                  f"Turnover ${recent_turnover_usd:,.0f}, "
                  f"Open ${recent_open:.2f} > Prev Close ${prev_close:.2f}, "
                  f"Close ${recent_close:.2f} > Prev Close ${prev_close:.2f}")
            return {
                "stock": ticker,
                "status": "success",
                "recent_volume": recent_volume,
                "prev_volume": prev_volume,
                "volume_ratio": actual_ratio,
                "recent_date": recent_date.strftime('%Y-%m-%d'),
                "prev_date": prev_date.strftime('%Y-%m-%d'),
                "recent_turnover_usd": recent_turnover_usd,
                "recent_open_price": recent_open,
                "recent_close_price": recent_close,
                "prev_close_price": prev_close
            }
        else:
            msg = f"Volume ratio {actual_ratio:.2f}x below threshold {volume_ratio}x"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}
            
    except Exception as e:
        print(f"Finished {ticker} -> Error: {str(e)}")
        return {"stock": ticker, "status": "failed", "reason": str(e)}


def parse_target_file(filepath: str) -> list:
    """
    Parse target symbols from stock_screener output file
    
    Args:
        filepath: Path to the target file
    
    Returns:
        list: List of stock symbols
    """
    if not os.path.exists(filepath):
        print(f"Target file not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the TARGETS section
    if "###TARGETS" in content:
        targets_section = content.split("###TARGETS")[1].strip()
        symbols = [s.strip() for s in targets_section.split(',') if s.strip()]
        return symbols
    
    return []


def get_latest_target_file(base_dir: str = "output") -> str:
    """
    Get the most recent target file from the output directory
    
    Args:
        base_dir: Base directory to search
    
    Returns:
        str: Path to the most recent target file, or None if not found
    """
    if not os.path.exists(base_dir):
        return None
    
    # Find all target files
    target_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "strong_targets.txt" in file:
                filepath = os.path.join(root, file)
                target_files.append(filepath)
    
    if not target_files:
        return None
    
    # Sort by modification time and return the most recent
    target_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return target_files[0]


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Volume-based Stock Screener')
    parser.add_argument('-f', '--file', type=str, help='Path to target file from stock_screener.py')
    parser.add_argument('-r', '--ratio', type=float, default=None,
                       help=f'Volume threshold ratio (default: {INTRADAY_VOLUME_THRESHOLD_RATIO}x for intraday, {DAILY_VOLUME_THRESHOLD_RATIO}x for daily)')
    args = parser.parse_args()
    
    # Determine which symbols to process
    if args.file:
        print(f"Reading symbols from: {args.file}")
        all_symbols = parse_target_file(args.file)
        if not all_symbols:
            print("No symbols found in target file. Exiting.")
            exit(1)
        print(f"Found {len(all_symbols)} symbols in target file")
    else:
        print("Using all stock symbols (default behavior)...")
        sd = StockDownloader()
        all_symbols = sd.get_all_tickers()
        print(f"Found {len(all_symbols)} stock symbols")
    
    # Check if market is open
    ny_tz = timezone(NY_TIMEZONE)
    current_time = datetime.now(ny_tz)
    market_is_open = is_market_open(current_time)
    
    # Determine the appropriate volume ratio based on market status
    if args.ratio is not None:
        volume_ratio = args.ratio
    else:
        volume_ratio = INTRADAY_VOLUME_THRESHOLD_RATIO if market_is_open else DAILY_VOLUME_THRESHOLD_RATIO
    
    print(f"\nCurrent time (NY): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market status: {'OPEN' if market_is_open else 'CLOSED'}")
    print(f"Volume threshold ratio: {volume_ratio}x")
    print(f"Total symbols to process: {len(all_symbols)}\n")
    
    # Choose the appropriate function based on market status
    if market_is_open:
        print("Using intraday volume comparison (5-minute timeframe)...")
        print("Additional filters: Opening price > Prev day's price at same time\n")
        check_function = check_intraday_volume_surge
    else:
        print("Using daily volume comparison (1-day timeframe)...")
        print("Additional filters: Opening price > Prev day's closing price\n")
        check_function = check_daily_volume_surge
    
    # Process all symbols using multiprocessing
    num_cores = max(1, mp.cpu_count()-2)
    print(f"Using {num_cores} processes\n")
    
    volume_surge_stocks = []
    failed_stocks = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(check_function, ticker, volume_ratio): ticker 
                  for ticker in all_symbols}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result(timeout=15)
                if result["status"] == "success":
                    volume_surge_stocks.append(result)
                else:
                    failed_stocks.append((ticker, result["reason"]))
            except TimeoutError:
                print(f"{ticker} -> Timeout")
                failed_stocks.append((ticker, "Timeout"))
            except Exception as e:
                print(f"{ticker} -> Error: {str(e)}")
                failed_stocks.append((ticker, str(e)))
    
    # Sort by turnover (dollar volume) instead of volume ratio
    if market_is_open:
        volume_surge_stocks.sort(key=lambda x: x.get('today_turnover_usd', 0), reverse=True)
    else:
        volume_surge_stocks.sort(key=lambda x: x.get('recent_turnover_usd', 0), reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total symbols processed: {len(all_symbols)}")
    print(f"Volume surge detected: {len(volume_surge_stocks)}")
    print(f"Failed/Filtered: {len(failed_stocks)}")
    print(f"Success rate: {len(volume_surge_stocks)/len(all_symbols)*100:.2f}%")
    
    # Print top stocks
    print(f"\n{'='*60}")
    print(f"TOP {min(50, len(volume_surge_stocks))} STOCKS BY TURNOVER (DOLLAR VOLUME)")
    print(f"{'='*60}")
    for result in volume_surge_stocks[:50]:
        ticker = result['stock']
        ratio = result['volume_ratio']
        if market_is_open:
            turnover = result.get('today_turnover_usd', 0)
            open_price = result.get('today_open_price', 0)
            close_price = result.get('today_close_price', 0)
            prev_price = result.get('prev_comparison_price', 0)
            print(f"{ticker}: ${turnover:,.0f} | Ratio: {ratio:.2f}x | "
                  f"Open: ${open_price:.2f} Close: ${close_price:.2f} > Prev: ${prev_price:.2f}")
        else:
            turnover = result.get('recent_turnover_usd', 0)
            open_price = result.get('recent_open_price', 0)
            close_price = result.get('recent_close_price', 0)
            prev_close = result.get('prev_close_price', 0)
            print(f"{ticker}: ${turnover:,.0f} ({result['recent_date']} vs {result['prev_date']}) | "
                  f"Ratio: {ratio:.2f}x | Open: ${open_price:.2f} Close: ${close_price:.2f} > Prev Close: ${prev_close:.2f}")
    
    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d")
    full_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Create output directory
    base_folder = "stock_volume_screener_output"
    date_folder = os.path.join(base_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)
    
    # Prepare content for TradingView
    txt_content = "###INDEX\nSPY,QQQ,DJI\n###TARGETS\n"
    txt_content += ",".join([result['stock'] for result in volume_surge_stocks])
    
    # Save to file
    market_status = "intraday" if market_is_open else "daily"
    output_filename = f"{full_timestamp}_volume_{market_status}_ratio{volume_ratio}x.txt"
    output_path = os.path.join(date_folder, output_filename)
    
    with open(output_path, 'w') as f:
        f.write(txt_content)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Optionally save detailed results
    detailed_filename = f"{full_timestamp}_volume_{market_status}_ratio{volume_ratio}x_detailed.txt"
    detailed_path = os.path.join(date_folder, detailed_filename)
    
    with open(detailed_path, 'w') as f:
        f.write(f"Volume Surge Analysis - {current_time.strftime('%Y-%m-%d %H:%M:%S')} NY Time\n")
        f.write(f"Market Status: {'OPEN' if market_is_open else 'CLOSED'}\n")
        f.write(f"Volume Threshold: {volume_ratio}x\n")
        f.write(f"Additional Filters:\n")
        f.write(f"  - Opening price must be higher than previous comparison price\n")
        f.write(f"  - Closing price must be higher than previous comparison price\n")
        f.write(f"Sorted by: Turnover (Dollar Volume) - Highest to Lowest\n")
        f.write(f"{'='*60}\n\n")
        
        for result in volume_surge_stocks:
            ticker = result['stock']
            ratio = result['volume_ratio']
            if market_is_open:
                turnover = result.get('today_turnover_usd', 0)
                open_price = result.get('today_open_price', 0)
                close_price = result.get('today_close_price', 0)
                prev_price = result.get('prev_comparison_price', 0)
                f.write(f"{ticker}: Volume Ratio {ratio:.2f}x\n")
                f.write(f"  Today's Volume: {result['today_volume']:,.0f}\n")
                f.write(f"  Previous Day Volume: {result['prev_volume']:,.0f}\n")
                f.write(f"  Today's Turnover: ${turnover:,.0f}\n")
                f.write(f"  Opening Price: ${open_price:.2f}\n")
                f.write(f"  Closing Price: ${close_price:.2f}\n")
                f.write(f"  Prev Day Comparison Price: ${prev_price:.2f}\n\n")
            else:
                turnover = result.get('recent_turnover_usd', 0)
                open_price = result.get('recent_open_price', 0)
                close_price = result.get('recent_close_price', 0)
                prev_close = result.get('prev_close_price', 0)
                f.write(f"{ticker}: Volume Ratio {ratio:.2f}x\n")
                f.write(f"  Date Comparison: {result['recent_date']} vs {result['prev_date']}\n")
                f.write(f"  Recent Volume: {result['recent_volume']:,.0f}\n")
                f.write(f"  Previous Volume: {result['prev_volume']:,.0f}\n")
                f.write(f"  Recent Day Turnover: ${turnover:,.0f}\n")
                f.write(f"  Recent Day Opening Price: ${open_price:.2f}\n")
                f.write(f"  Recent Day Closing Price: ${close_price:.2f}\n")
                f.write(f"  Previous Day Closing Price: ${prev_close:.2f}\n\n")
    
    print(f"Detailed results saved to: {detailed_path}\n")