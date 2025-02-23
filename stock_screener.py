from datetime import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import argparse
from src.downloader import StockDownloader


#================= CONFIGURATIONS =================#
_1D_OF_DAYS_TRACEBACK = 252
_1H_OF_DAYS_TRACEBACK = 63
CURRENT_TIMEZONE = "America/Los_Angeles"
MIN_TURNOVER = 10000000
#==================================================#


def calculate_rs_score(hourly_data: pd.DataFrame) -> tuple[bool, float, str]:
    """
    Calculate RS score from hourly data
    """
    # Get the most recent hourly_lookback * 8 bars
    required_bars = _1H_OF_DAYS_TRACEBACK * 8
    if len(hourly_data) < required_bars:
        return False, 0, f"Insufficient hourly data: {len(hourly_data)} < {required_bars}"

    hourly_data = hourly_data.tail(required_bars)

    # Calculate RS Score
    rs_score = 0.0
    try:
        calc_bars = len(hourly_data) - 60
        days = _1H_OF_DAYS_TRACEBACK

        for i in range(1, calc_bars + 1):
            current_close = hourly_data['Close'].values[-i]
            moving_average_30 = hourly_data['SMA_30'].values[-i]
            moving_average_45 = hourly_data['SMA_45'].values[-i]
            moving_average_60 = hourly_data['SMA_60'].values[-i]

            weight = (((current_close - moving_average_30) +
                      (current_close - moving_average_45) +
                      (current_close - moving_average_60)) *
                     (((calc_bars - i) * days / calc_bars) + 1) +
                     (moving_average_30 - moving_average_45) +
                     (moving_average_30 - moving_average_60) +
                     (moving_average_45 - moving_average_60)) / moving_average_60

            rs_score += weight * (calc_bars - i)

        return True, rs_score, ""

    except Exception as e:
        return False, 0, f"Error calculating RS score: {str(e)}"


def calculate_spy_rs_score() -> float:
    """Calculate SPY's RS score regardless of trend template conditions"""
    sd = StockDownloader()
    print("Processing SPY RS score calculation...")

    # Request more data than needed to ensure we have enough after filtering
    now = int(time.time())
    buffer_days = int(_1H_OF_DAYS_TRACEBACK * 2.5)  # 100% buffer
    hourly_start_ts = now - (buffer_days * 24 * 3600)

    success, hourly_data = sd.get_data("SPY", hourly_start_ts, timeframe="1h")
    if not success or hourly_data is None:
        raise ValueError("Failed to get hourly data for SPY")

    success, rs_score, error = calculate_rs_score(hourly_data)
    if not success:
        raise ValueError(f"Failed to calculate SPY RS score: {error}")

    print(f"Finished SPY -> RS Score {rs_score}")
    return rs_score
    

def calc_relative_strength(ticker: str):
    """Calculate relative strength and check trend template conditions"""
    print(f"Processing {ticker}...")
    try:
        sd = StockDownloader()
        now = int(time.time())

        # Request more data than needed
        buffer_days = int(_1D_OF_DAYS_TRACEBACK * 2.5)  # 100% buffer
        daily_start_ts = now - (buffer_days * 24 * 3600)
        success, daily_data = sd.get_data(ticker, daily_start_ts, timeframe="1d")

        if not success or daily_data is None:
            msg = "No daily data"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}

        # Take the most recent required days
        if len(daily_data) < _1D_OF_DAYS_TRACEBACK:
            msg = f"Insufficient daily data: {len(daily_data)} < {_1D_OF_DAYS_TRACEBACK}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}

        daily_data = daily_data.tail(_1D_OF_DAYS_TRACEBACK)

        # Check turnover
        last_10_days = daily_data.tail(10)
        average_turnover = (last_10_days['Volume'] * last_10_days['Close']).mean()
        if average_turnover < MIN_TURNOVER:
            msg = "Insufficient turnover"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}

        # Get required values
        current_close = daily_data['Close'].values[-1]
        moving_average_50 = daily_data['SMA_50'].values[-1]
        moving_average_60 = daily_data['SMA_60'].values[-1]
        moving_average_150 = daily_data['SMA_150'].values[-1]
        moving_average_200 = daily_data['SMA_200'].values[-1]

        # Calculate high/low using configured lookback period
        low_of_period = daily_data["Close"].min()
        high_of_period = daily_data["Close"].max()

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

        if not all(conditions):
            failed_conditions = [i + 1 for i, cond in enumerate(conditions) if not cond]
            msg = f"Failed conditions: {failed_conditions}"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}

        # Get hourly data and calculate RS score
        buffer_days = int(_1H_OF_DAYS_TRACEBACK * 2.5)
        hourly_start_ts = now - (buffer_days * 24 * 3600)
        success, hourly_data = sd.get_data(ticker, hourly_start_ts, timeframe="1h")

        if not success or hourly_data is None:
            msg = "No hourly data"
            print(f"Finished {ticker} -> Failed: {msg}")
            return {"stock": ticker, "status": "failed", "reason": msg}

        success, rs_score, error = calculate_rs_score(hourly_data)
        if not success:
            print(f"Finished {ticker} -> Failed: {error}")
            return {"stock": ticker, "status": "failed", "reason": error}

        print(f"Finished {ticker} -> RS Score {rs_score}")
        return {
            "stock": ticker,
            "status": "success",
            "rs_score": rs_score
        }

    except Exception as e:
        msg = str(e)
        print(f"Finished {ticker} -> Failed: {msg}")
        return {"stock": ticker, "status": "failed", "reason": msg}


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Trend Analysis')
    parser.add_argument('--all', action='store_true', help='Include all strong targets in output')
    args = parser.parse_args()

    # Initialize stock downloader
    sd = StockDownloader()

    # Get list of all tickers
    all_tickers = sd.get_all_tickers()
    print(f"Total tickers to process: {len(all_tickers)}")

    # Calculate SPY's RS score first
    try:
        spy_rs_score = calculate_spy_rs_score()
        print(f"SPY RS Score: {spy_rs_score}")
    except Exception as e:
        print(f"Failed to calculate SPY RS score: {e}")
        exit(1)

    # Process all tickers using multiprocessing
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} processes")

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(calc_relative_strength, all_tickers)

    # Process results
    strong_targets = []
    target_rs_score = {}
    failed_tickers = []

    for result in results:
        ticker = result["stock"]
        if result["status"] == "success":
            if result["rs_score"] >= spy_rs_score:
                strong_targets.append(ticker)
                target_rs_score[ticker] = result["rs_score"]
        else:
            failed_tickers.append((ticker, result["reason"]))

    # Sort by RS score
    strong_targets.sort(key=lambda x: target_rs_score[x], reverse=True)

    # Print results
    total_analyzed = len(all_tickers) - len(failed_tickers)
    success_rate = len(strong_targets) / total_analyzed * 100 if total_analyzed > 0 else 0

    print(f"\nAnalysis Results:")
    print(f"Total tickers processed: {len(all_tickers)}")
    print(f"Failed tickers: {len(failed_tickers)}")
    print(f"Found {len(strong_targets)} stocks that meet requirements and are stronger than SPY")
    print(f"Success rate: {success_rate:.2f}%")

    print(f"\nStrong targets: {', '.join(strong_targets[:50])}")  # Show top 50 only in console

    print("\n====== Top 50 Targets by RS Score ======")
    for ticker in strong_targets[:50]:
        score = target_rs_score[ticker]
        print(f"{ticker}: {score}")
    print("=======================================")

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d %H-%M")
    txt_content = "###INDEX\nSPY,QQQ,DJI\n###TARGETS\n"

    # Use all strong targets or just top 980 based on --all flag
    output_targets = strong_targets if args.all else strong_targets[:980]
    txt_content += ",".join(output_targets)

    output_file = f"{date_str}_stock_strong_targets.txt"
    with open(output_file, "w") as f:
        f.write(txt_content)

    # Save failed tickers for analysis
    failed_file = f"{date_str}_failed_tickers.txt"
    with open(failed_file, "w") as f:
        for ticker, reason in failed_tickers:
            f.write(f"{ticker}: {reason}\n")

    print(f"\nResults saved to {output_file}")
    print(f"Failed tickers saved to {failed_file}")
    print(f"Included {'all' if args.all else 'top 980'} strong targets in output file")
