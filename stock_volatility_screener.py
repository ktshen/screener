"""
High Volatility & High Turnover Stock Screener (Multi-Day Average)
====================================================================
Identifies stocks with high intraday volatility and high trading turnover
based on multi-day averages.

Filters:
- Average 1-minute bar volatility >= 0.25% (averaged over lookback period)
- Average daily turnover >= $100M (averaged over lookback period)
- Sorted by volatility (highest to lowest)
"""

import os
import time
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.downloader import StockDownloader


# ================= CONFIGURATIONS =================
LOOKBACK_DAYS = 7  # Number of trading days to look back
MIN_AVG_VOLATILITY_PCT = 0.2  # Minimum average volatility in percentage
MIN_DAILY_TURNOVER = 100_000_000  # Minimum average daily turnover in USD (100M)
NY_TIMEZONE = 'America/New_York'
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
# ==================================================


def get_last_n_trading_days(n: int, current_time: datetime = None) -> List[datetime]:
    """
    Get the last N complete trading days before the current time

    Args:
        n: Number of trading days to retrieve
        current_time: datetime object in NY timezone (default: now)

    Returns:
        List[datetime]: List of last N trading days (most recent first)
    """
    if current_time is None:
        ny_tz = timezone(NY_TIMEZONE)
        current_time = datetime.now(ny_tz)

    trading_days = []
    current_day = current_time - timedelta(days=1)

    while len(trading_days) < n:
        # Skip weekends
        if current_day.weekday() < 5:
            trading_days.append(current_day)
        current_day = current_day - timedelta(days=1)

    return trading_days


def calculate_bar_volatility(row: pd.Series) -> float:
    """
    Calculate the volatility of a single bar as percentage
    Uses (high - low) / close * 100

    Args:
        row: DataFrame row with 'high', 'low', 'close' columns

    Returns:
        float: Volatility percentage
    """
    if row['close'] == 0:
        return 0.0
    return ((row['high'] - row['low']) / row['close']) * 100


def check_volatility_and_volume(ticker: str,
                                 lookback_days: int,
                                 min_volatility_pct: float,
                                 min_avg_turnover: float) -> Dict:
    """
    Check if a stock meets volatility and turnover criteria based on multi-day averages

    Args:
        ticker: Stock symbol
        lookback_days: Number of trading days to analyze
        min_volatility_pct: Minimum average volatility percentage
        min_avg_turnover: Minimum average daily turnover in USD

    Returns:
        dict: Result containing status and metrics
    """
    print(f"Processing {ticker}...")

    sd = None
    try:
        sd = StockDownloader()
        ny_tz = timezone(NY_TIMEZONE)
        current_time = datetime.now(ny_tz)

        # Get last N trading days
        trading_days = get_last_n_trading_days(lookback_days, current_time)

        all_daily_data = []
        daily_turnovers = []
        daily_volatilities = []

        # Process each trading day
        for trading_day in trading_days:
            # Set time range for the full trading day (9:30 AM - 4:00 PM)
            day_start = trading_day.replace(
                hour=MARKET_OPEN_HOUR,
                minute=MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0
            )
            day_end = trading_day.replace(
                hour=MARKET_CLOSE_HOUR,
                minute=MARKET_CLOSE_MINUTE,
                second=0,
                microsecond=0
            )

            day_start_ts = int(day_start.timestamp())
            day_end_ts = int(day_end.timestamp())

            # Get 1-minute bars for this trading day
            success, data = sd.get_data(
                ticker,
                day_start_ts,
                end_ts=day_end_ts,
                timeframe="1m",
                dropna=False,
                atr=False,
                vwap=False,
                validate=False
            )

            if not success or data.empty:
                # Skip this day if no data
                continue

            # Check if we have enough data (at least 50 bars for a partial day)
            if len(data) < 50:
                continue

            # Calculate volatility for each 1-minute bar
            data['volatility_pct'] = data.apply(calculate_bar_volatility, axis=1)

            # Calculate daily metrics
            daily_avg_volatility = data['volatility_pct'].mean()
            total_volume = data['volume'].sum()
            avg_close = data['close'].mean()
            daily_turnover = total_volume * avg_close

            # Store daily data
            all_daily_data.append(data)
            daily_turnovers.append(daily_turnover)
            daily_volatilities.append(daily_avg_volatility)

        # Check if we have enough valid trading days
        if len(daily_turnovers) < max(1, lookback_days // 2):
            msg = f"Insufficient data: only {len(daily_turnovers)} valid days out of {lookback_days}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {
                "stock": ticker,
                "status": "failed",
                "reason": msg
            }

        # Calculate multi-day averages
        avg_daily_turnover = np.mean(daily_turnovers)
        avg_volatility_pct = np.mean(daily_volatilities)

        # Check if criteria are met
        if avg_volatility_pct < min_volatility_pct:
            msg = f"Low avg volatility: {avg_volatility_pct:.3f}% < {min_volatility_pct}%"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {
                "stock": ticker,
                "status": "failed",
                "reason": msg
            }

        if avg_daily_turnover < min_avg_turnover:
            msg = f"Low avg turnover: ${avg_daily_turnover:,.0f} < ${min_avg_turnover:,.0f}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {
                "stock": ticker,
                "status": "failed",
                "reason": msg
            }

        # Calculate additional statistics
        max_daily_volatility = max(daily_volatilities)
        min_daily_volatility = min(daily_volatilities)
        max_daily_turnover = max(daily_turnovers)
        min_daily_turnover = min(daily_turnovers)

        # Combine all data for overall statistics
        combined_data = pd.concat(all_daily_data, ignore_index=True)
        total_bars = len(combined_data)
        overall_max_volatility = combined_data['volatility_pct'].max()
        overall_min_volatility = combined_data['volatility_pct'].min()

        result = {
            "stock": ticker,
            "status": "success",
            "avg_volatility_pct": avg_volatility_pct,
            "max_daily_volatility_pct": max_daily_volatility,
            "min_daily_volatility_pct": min_daily_volatility,
            "overall_max_volatility_pct": overall_max_volatility,
            "overall_min_volatility_pct": overall_min_volatility,
            "avg_daily_turnover_usd": avg_daily_turnover,
            "max_daily_turnover_usd": max_daily_turnover,
            "min_daily_turnover_usd": min_daily_turnover,
            "lookback_days": lookback_days,
            "valid_days": len(daily_turnovers),
            "total_bars": total_bars,
            "period_start": trading_days[-1].strftime("%Y-%m-%d"),
            "period_end": trading_days[0].strftime("%Y-%m-%d")
        }

        print(f"Finished {ticker} -> Success: Avg Vol={avg_volatility_pct:.3f}%, "
              f"Avg Daily Turnover=${avg_daily_turnover:,.0f}")
        return result

    except Exception as e:
        msg = f"Error: {str(e)}"
        print(f"Finished {ticker} -> Failed: {msg}")
        return {
            "stock": ticker,
            "status": "failed",
            "reason": msg
        }
    finally:
        # Explicitly close the connection pool
        if sd is not None and hasattr(sd, 'client'):
            try:
                # Close the underlying session in RESTClient
                if hasattr(sd.client, '_session'):
                    sd.client._session.close()
            except Exception as e:
                # Ignore errors during cleanup
                pass


def parse_target_file(filepath: str) -> List[str]:
    """
    Parse a target file and extract stock symbols

    Expected format:
        ###INDEX
        SPY,QQQ,DJI
        ###TARGETS
        AAPL,MSFT,GOOGL
        ...

    Args:
        filepath: Path to the target file

    Returns:
        List of stock symbols
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    symbols = []
    targets_section = False

    for line in lines:
        line = line.strip()
        if '###TARGETS' in line:
            targets_section = True
            continue
        if targets_section and line.strip():
            # Parse comma-separated symbols
            line_symbols = [s.strip() for s in line.split(',') if s.strip()]
            symbols.extend(line_symbols)

    # Remove duplicates and sort
    symbols = sorted(set(symbols))
    return symbols


def save_results(results: List[Dict],
                 lookback_days: int,
                 min_volatility_pct: float,
                 min_avg_turnover: float,
                 output_dir: str = "stock_volatility_screener_output") -> Tuple[str, str]:
    """
    Save screening results to files

    Args:
        results: List of result dictionaries
        lookback_days: Number of lookback days used
        min_volatility_pct: Minimum volatility percentage used
        min_avg_turnover: Minimum average turnover in USD used
        output_dir: Base output directory

    Returns:
        Tuple of (summary_file_path, detailed_file_path)
    """
    # Create output directory structure
    date_str = datetime.now().strftime("%Y-%m-%d")
    full_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    date_folder = os.path.join(output_dir, date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Prepare content for TradingView format
    txt_content = "###INDEX\nSPY,QQQ,DJI\n###TARGETS\n"
    txt_content += ",".join([result['stock'] for result in results])

    # Save summary file (TradingView format)
    summary_filename = f"{full_timestamp}_volatility_{lookback_days}d_vol{min_volatility_pct}pct_turnover{min_avg_turnover/1e6:.0f}M.txt"
    summary_path = os.path.join(date_folder, summary_filename)

    with open(summary_path, 'w') as f:
        f.write(txt_content)

    # Save detailed results
    detailed_filename = f"{full_timestamp}_volatility_{lookback_days}d_vol{min_volatility_pct}pct_turnover{min_avg_turnover/1e6:.0f}M_detailed.txt"
    detailed_path = os.path.join(date_folder, detailed_filename)

    with open(detailed_path, 'w') as f:
        f.write(f"High Volatility & High Turnover Stock Screener ({lookback_days}-Day Average)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Criteria:\n")
        f.write(f"  - Lookback Period: {lookback_days} trading days\n")
        f.write(f"  - Minimum Average 1-min Volatility: {min_volatility_pct}%\n")
        f.write(f"  - Minimum Average Daily Turnover: ${min_avg_turnover:,.0f}\n")
        f.write(f"  - Sorted by: Average Volatility (Highest to Lowest)\n\n")
        f.write(f"Total stocks found: {len(results)}\n")
        f.write(f"{'='*80}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['stock']}\n")
            f.write(f"   Analysis Period: {result['period_start']} to {result['period_end']}\n")
            f.write(f"   Valid Trading Days: {result['valid_days']} / {result['lookback_days']}\n")
            f.write(f"   Total Bars Analyzed: {result['total_bars']}\n")
            f.write(f"\n")
            f.write(f"   Average Volatility (across days): {result['avg_volatility_pct']:.3f}%\n")
            f.write(f"   Max Daily Average Volatility: {result['max_daily_volatility_pct']:.3f}%\n")
            f.write(f"   Min Daily Average Volatility: {result['min_daily_volatility_pct']:.3f}%\n")
            f.write(f"   Overall Max Bar Volatility: {result['overall_max_volatility_pct']:.3f}%\n")
            f.write(f"   Overall Min Bar Volatility: {result['overall_min_volatility_pct']:.3f}%\n")
            f.write(f"\n")
            f.write(f"   Average Daily Turnover: ${result['avg_daily_turnover_usd']:,.0f}\n")
            f.write(f"   Max Daily Turnover: ${result['max_daily_turnover_usd']:,.0f}\n")
            f.write(f"   Min Daily Turnover: ${result['min_daily_turnover_usd']:,.0f}\n")
            f.write(f"\n")

    return summary_path, detailed_path


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='High Volatility & High Turnover Stock Screener (Multi-Day Average)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen all stocks with default 7-day lookback
  python stock_volatility_screener.py

  # Screen stocks from a target file with 5-day lookback
  python stock_volatility_screener.py -f stock_rs_output/2024-01-01/targets.txt --lookback-days 5

  # Use custom thresholds with 10-day lookback
  python stock_volatility_screener.py --lookback-days 10 --min-volatility 0.3 --min-turnover 200000000
        """
    )
    parser.add_argument('-f', '--file', type=str,
                       help='Path to target file (optional, otherwise screens all stocks)')
    parser.add_argument('--lookback-days', type=int, default=LOOKBACK_DAYS,
                       help=f'Number of trading days to look back (default: {LOOKBACK_DAYS})')
    parser.add_argument('--min-volatility', type=float, default=MIN_AVG_VOLATILITY_PCT,
                       help=f'Minimum average volatility percentage (default: {MIN_AVG_VOLATILITY_PCT})')
    parser.add_argument('--min-turnover', type=float, default=MIN_DAILY_TURNOVER,
                       help=f'Minimum average daily turnover in USD (default: ${MIN_DAILY_TURNOVER:,.0f})')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 1)')

    args = parser.parse_args()

    # Use argument values or defaults
    lookback_days = args.lookback_days
    min_volatility_pct = args.min_volatility
    min_avg_turnover = args.min_turnover

    # Determine which symbols to process
    if args.file:
        print(f"Reading symbols from: {args.file}")
        all_symbols = parse_target_file(args.file)
        if not all_symbols:
            print("No symbols found in target file. Exiting.")
            return
        print(f"Found {len(all_symbols)} symbols in target file")
    else:
        print("Fetching all stock symbols...")
        sd = StockDownloader()
        all_symbols = sd.get_all_tickers()
        print(f"Found {len(all_symbols)} stock symbols")

    # Determine number of workers
    num_workers = args.workers if args.workers else max(1, mp.cpu_count() - 1)

    # Get trading days for reference
    ny_tz = timezone(NY_TIMEZONE)
    current_time = datetime.now(ny_tz)
    trading_days = get_last_n_trading_days(lookback_days, current_time)

    print(f"\n{'='*80}")
    print(f"HIGH VOLATILITY & HIGH TURNOVER STOCK SCREENER ({lookback_days}-DAY AVERAGE)")
    print(f"{'='*80}")
    print(f"Current time (NY): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis period: {trading_days[-1].strftime('%Y-%m-%d')} to {trading_days[0].strftime('%Y-%m-%d')}")
    print(f"Lookback period: {lookback_days} trading days")
    print(f"Minimum average volatility: {min_volatility_pct}%")
    print(f"Minimum average daily turnover: ${min_avg_turnover:,.0f}")
    print(f"Total symbols to process: {len(all_symbols)}")
    print(f"Using {num_workers} worker processes")
    print(f"{'='*80}\n")

    # Process all symbols using multiprocessing
    qualified_stocks = []
    failed_stocks = []

    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(check_volatility_and_volume, ticker, lookback_days,
                          min_volatility_pct, min_avg_turnover): ticker
            for ticker in all_symbols
        }

        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1

            # Show progress every 100 stocks
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = len(all_symbols) - completed
                eta = remaining / rate if rate > 0 else 0
                print(f"\n[Progress] {completed}/{len(all_symbols)} completed | "
                      f"Rate: {rate:.1f} stocks/sec | ETA: {eta/60:.1f} min\n")

            try:
                result = future.result(timeout=60)  # Increased timeout for multi-day data
                if result["status"] == "success":
                    qualified_stocks.append(result)
                else:
                    failed_stocks.append((ticker, result.get("reason", "Unknown")))
            except TimeoutError:
                print(f"{ticker} -> Timeout")
                failed_stocks.append((ticker, "Timeout"))
            except Exception as e:
                print(f"{ticker} -> Error: {str(e)}")
                failed_stocks.append((ticker, str(e)))

    elapsed_time = time.time() - start_time

    # Sort by average volatility (highest to lowest)
    qualified_stocks.sort(key=lambda x: x['avg_volatility_pct'], reverse=True)

    # Print summary
    print(f"\n{'='*80}")
    print("SCREENING RESULTS")
    print(f"{'='*80}")
    print(f"Total symbols processed: {len(all_symbols)}")
    print(f"Qualified stocks: {len(qualified_stocks)}")
    print(f"Failed/Filtered: {len(failed_stocks)}")
    print(f"Success rate: {len(qualified_stocks)/len(all_symbols)*100:.2f}%")
    print(f"Processing time: {elapsed_time:.1f} seconds")

    if qualified_stocks:
        # Print top stocks
        print(f"\n{'='*80}")
        print(f"TOP {min(50, len(qualified_stocks))} STOCKS BY AVERAGE VOLATILITY")
        print(f"{'='*80}")

        for i, result in enumerate(qualified_stocks[:50], 1):
            print(f"{i:2d}. {result['stock']:6s} | "
                  f"Avg Vol: {result['avg_volatility_pct']:5.3f}% | "
                  f"Avg Daily Turnover: ${result['avg_daily_turnover_usd']:>15,.0f} | "
                  f"Days: {result['valid_days']}/{result['lookback_days']}")

        # Save results
        summary_path, detailed_path = save_results(qualified_stocks, lookback_days,
                                                   min_volatility_pct, min_avg_turnover)

        print(f"\n{'='*80}")
        print("OUTPUT FILES")
        print(f"{'='*80}")
        print(f"Summary (TradingView format): {summary_path}")
        print(f"Detailed results: {detailed_path}")
        print(f"{'='*80}\n")
    else:
        print("\nNo stocks met the criteria.")


if __name__ == '__main__':
    main()

