from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
import pandas as pd
import time
import os
import numpy as np
import argparse
from src.downloader import StockDownloader


#================= CONFIGURATIONS =================#
_1D_OF_DAYS_TRACEBACK = 252
_1H_OF_DAYS_TRACEBACK = 126
MIN_TURNOVER = 10000000 # Minimum turnover for stock to be considered (volume * close)
#==================================================#


def calculate_rs_score(hourly_data: pd.DataFrame, ticker: str = "unknown") -> tuple[bool, float, str]:
    """
    Calculate RS score from hourly data without using Z-score normalization.
    
    The RS score is calculated as a weighted sum of relative strength indicators,
    with newer data given higher weight. ATR is used for normalization to allow
    comparison across different stocks.
    """
    # Define fixed parameters
    required_bars = _1H_OF_DAYS_TRACEBACK * 8
    
    # Check if we have enough data
    if len(hourly_data) < required_bars:
        return False, 0, f"Insufficient hourly data: {len(hourly_data)} < {required_bars}"
    
    # Create a copy to avoid modifying the original data
    data = hourly_data.copy()
    
    # Take the most recent required_bars data points
    data = data.tail(required_bars).reset_index(drop=True)
    
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
        # weight = i 
        k = 2 * np.log(2) / required_bars   
        weight = np.exp(k * i)              # Exponential weight where w(L/2) * 2 = w(L)
        
        # Add to weighted sum
        rs_score += relative_strength * weight
        total_weight += weight
    
    # Normalize the final score by total weight
    if total_weight > 0:
        rs_score = rs_score / total_weight
    else:
        return False, 0, "Weight calculation error"

    return True, rs_score, ""


def calculate_spy_rs_score() -> float:

    sd = StockDownloader()
    print("Processing SPY RS score calculation...")

    # Request more data than needed to ensure we have enough after filtering
    now = int(time.time())
    buffer_days = int(_1H_OF_DAYS_TRACEBACK * 3)  # 200% buffer for safety
    hourly_start_ts = now - (buffer_days * 24 * 3600)

    success, hourly_data = sd.get_data("SPY", hourly_start_ts, end_ts=now, timeframe="1h", atr=True)
    if not success or hourly_data is None:
        raise ValueError("Failed to get hourly data for SPY")

    success, rs_score, error = calculate_rs_score(hourly_data, "SPY")
    if not success:
        raise ValueError(f"Failed to calculate SPY RS score: {error}")

    print(f"Finished SPY -> RS Score {rs_score}")
    return rs_score
    

def calc_relative_strength(ticker: str, use_template: bool) -> dict:
    """
    Calculate relative strength and check trend template conditions for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol.
        use_template (bool): Flag to determine whether to apply Minervini trend template conditions.

    Returns:
        dict: A dictionary containing the stock ticker, status of the calculation ('success' or 'failed'), 
              and additional information such as the reason for failure or the calculated RS score.

    The function performs the following steps:
        1. Downloads daily stock data for the given ticker.
        2. Checks if there is sufficient daily data.
        3. Verifies if the stock meets the minimum turnover requirement.
        4. If `use_template` is True, checks the stock against Minervini trend template conditions.
        5. Downloads hourly stock data for the given ticker.
        6. Calculates the relative strength (RS) score using the hourly data.
        7. Returns the result with the RS score if successful, or the reason for failure.
    """

    print(f"Processing {ticker}...")
    sd = StockDownloader()
    now = int(time.time())

    # Request more data than needed for daily timeframe
    buffer_days = int(_1D_OF_DAYS_TRACEBACK * 2)  
    daily_start_ts = now - (buffer_days * 24 * 3600)
    success, daily_data = sd.get_data(ticker, daily_start_ts, end_ts=now, timeframe="1d", dropna=False, atr=False)

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
    average_turnover = (last_10_days['volume'] * last_10_days['close']).mean()
    if average_turnover < MIN_TURNOVER:
        msg = "Insufficient turnover"
        print(f"Finished {ticker} -> Failed: {msg}")
        return {"stock": ticker, "status": "failed", "reason": msg}

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
    if use_template:
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

    # Get hourly data with buffer for RS score calculation
    buffer_days = int(_1H_OF_DAYS_TRACEBACK * 3)
    hourly_start_ts = now - (buffer_days * 24 * 3600)
    success, hourly_data = sd.get_data(ticker, hourly_start_ts, end_ts=now, timeframe="1h", atr=True)

    if not success or hourly_data is None:
        msg = "No hourly data"
        print(f"Finished {ticker} -> Failed: {msg}")
        return {"stock": ticker, "status": "failed", "reason": msg}

    success, rs_score, error = calculate_rs_score(hourly_data, ticker)
    if not success:
        print(f"Finished {ticker} -> Failed: {error}")
        return {"stock": ticker, "status": "failed", "reason": error}

    print(f"Finished {ticker} -> RS Score {rs_score}")
    return {
        "stock": ticker,
        "status": "success",
        "rs_score": rs_score
    }


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Trend Analysis')
    parser.add_argument('-a', '--all', action='store_true', help='Include all strong targets in output')
    parser.add_argument('-g', action='store_true', help='Ignore Minerivini conditions and calculate RS score only')
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

    # Process all tickers using ProcessPoolExecutor
    num_cores = min(36, mp.cpu_count())
    print(f"Using {num_cores} processes")

    # Process results
    strong_targets = []
    target_rs_score = {}
    failed_tickers = []
    use_template = not args.g

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(calc_relative_strength, ticker, use_template): ticker for ticker in all_tickers}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result(timeout=10)
                if result["status"] == "success":
                    if result["rs_score"] >= spy_rs_score:
                        strong_targets.append(ticker)
                        target_rs_score[ticker] = result["rs_score"]
                else:
                    failed_tickers.append((ticker, result["reason"]))
            except TimeoutError:
                print(f"{ticker} took too long to process")
            except Exception as e:
                failed_tickers.append((ticker, str(e)))

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
    full_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    date_str = datetime.now().strftime("%Y-%m-%d")
    txt_content = "###INDEX\nSPY,QQQ,DJI\n###TARGETS\n"

    # Use all strong targets or just top 980 based on --all flag
    output_targets = strong_targets if args.all else strong_targets[:980]
    txt_content += ",".join(output_targets)

    # Create output/<date> directory structure
    base_folder = "output"
    date_folder = os.path.join(base_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)
    
    # Create output files with full timestamp in filename
    without_conditions = "_no_conditions" if args.g else ""
    for_tv = "all" if args.all else "top980"
    output_file = f"{full_date_str}_stock_{for_tv}{without_conditions}_strong_targets.txt"
    file_path = os.path.join(date_folder, output_file)
    with open(file_path, "w") as f:
        f.write(txt_content)

    # Save failed tickers for analysis
    # failed_file = f"{full_date_str}_failed_tickers.txt"
    # failed_path = os.path.join(date_folder, failed_file)
    # with open(failed_path, "w") as f:
    #     for ticker, reason in failed_tickers:
    #         f.write(f"{ticker}: {reason}\n")

    print(f"\nResults saved to {file_path}")
    # print(f"Failed tickers saved to {failed_path}")
    print(f"Included {'all' if args.all else 'top 980'} strong targets in output file")