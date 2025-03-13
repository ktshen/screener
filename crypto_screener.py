import argparse
import time
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.downloader import CryptoDownloader


def calc_total_bars(time_interval, days):
    bars_dict = {
        "5m": 12 * 24 * days,
        "15m": 4 * 24 * days,
        "30m": 2 * 24 * days,
        "1h":  24 * days,
        "2h": 12 * days,
        "4h": 6 * days,
        "8h": 3 * days,
    }
    return bars_dict.get(time_interval)


def calculate_rs_score(crypto_data, required_bars):
    """
    Calculate RS score for cryptocurrency
    
    Args:
        crypto_data: DataFrame with cryptocurrency data
        required_bars: Number of bars required for calculation
        
    Returns:
        tuple[bool, float, str]: Success flag, RS score, error message
    """
    # Check if we have enough data
    if len(crypto_data) < required_bars:
        return False, 0, f"Insufficient data: {len(crypto_data)} < {required_bars}"
    
    # Create a copy to avoid modifying the original data
    data = crypto_data.copy()
    
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
        denominator = current_atr + 0.0000000000000000001
        # denominator = (moving_average_30 + moving_average_45 + moving_average_60) / 3
        
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


def process_crypto(symbol, timeframe, days):
    """Process a single cryptocurrency and calculate its RS score"""
    try:
        cd = CryptoDownloader()
        
        # Calculate required bars
        required_bars = calc_total_bars(timeframe, days)
        
        # Calculate start timestamp with some buffer (20% more time to ensure we get enough data)
        buffer_factor = 1.2
        now = int(time.time())
        
        # Estimate interval seconds based on timeframe
        if "m" in timeframe:
            minutes = int(timeframe.replace("m", ""))
            interval_seconds = minutes * 60
        elif "h" in timeframe:
            hours = int(timeframe.replace("h", ""))
            interval_seconds = hours * 3600
        elif "d" in timeframe:
            days = int(timeframe.replace("d", ""))
            interval_seconds = days * 24 * 3600
        else:
            # Default to 1h if unknown format
            interval_seconds = 3600
        
        start_ts = now - int(required_bars * interval_seconds * buffer_factor)
        
        # Get crypto data
        success, data = cd.get_data(symbol, start_ts=start_ts, end_ts=now, timeframe=timeframe, atr=True)
        
        if not success or data.empty:
            error_msg = "Failed to get data or empty dataset"
            print(f"{symbol} -> Error: {error_msg}")
            return {"crypto": symbol, "status": "failed", "reason": error_msg}
        
        # Calculate RS score
        success, rs_score, error = calculate_rs_score(data, required_bars)
        if not success:
            print(f"{symbol} -> Error: {error}")
            return {"crypto": symbol, "status": "failed", "reason": error}
        
        print(f"{symbol} -> Successfully calculated RS Score: {rs_score}")
        return {
            "crypto": symbol,
            "status": "success",
            "rs_score": rs_score
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"{symbol} -> Error: {error_msg}")
        return {"crypto": symbol, "status": "failed", "reason": error_msg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeframe', type=str, help='Time frame (5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d)', default="15m")
    parser.add_argument('-d', '--days', type=int, help='Calculation duration in days (default 3 days)', default=3)
    args = parser.parse_args()
    timeframe = args.timeframe
    days = args.days
    
    # Initialize crypto downloader
    crypto_downloader = CryptoDownloader()
    
    # Get list of all symbols
    all_cryptos = crypto_downloader.get_all_symbols()
    print(f"Total cryptos to process: {len(all_cryptos)}")
    
    # Process all cryptos using ProcessPoolExecutor
    num_cores = min(4, mp.cpu_count())  # Use maximum 4 cores, binance rest api has rate limit
    print(f"Using {num_cores} processes")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_crypto, crypto, timeframe, days): crypto for crypto in all_cryptos}
        results = []
        
        for future in as_completed(futures):
            crypto = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"{crypto} -> Error: {str(e)}")
                results.append({"crypto": crypto, "status": "failed", "reason": str(e)})
    
    # Process results
    failed_targets = []     # Failed to download data or error happened
    target_score = {}
    
    for result in results:
        if result["status"] == "success":
            target_score[result["crypto"]] = result["rs_score"]
        else:
            failed_targets.append((result["crypto"], result["reason"]))
    
    # Sort by RS score
    targets = [x for x in target_score.keys()]
    targets.sort(key=lambda x: target_score[x], reverse=True)
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"Total cryptos processed: {len(all_cryptos)}")
    print(f"Failed cryptos: {len(failed_targets)}")
    print(f"Successfully calculated: {len(targets)}")
    
    print("\n=========================== Target : Score (TOP 20) ===========================")
    for idx, crypto in enumerate(targets[:20], 1):
        score = target_score[crypto]
        print(f"{idx}. {crypto}: {score:.6f}")
    print("===============================================================================")
    
    # Save results
    full_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    date_str = datetime.now().strftime("%Y-%m-%d")
    txt_content = "###BTCETH\nBINANCE:BTCUSDT.P,BINANCE:ETHUSDT\n###Targets (Sort by score)\n"
    
    # Add all targets
    if targets:
        txt_content += ",".join([f"BINANCE:{crypto}.P" for crypto in targets])
    
    # Create output/<date> directory structure
    base_folder = "output"
    date_folder = os.path.join(base_folder, date_str)
    os.makedirs(date_folder, exist_ok=True)
    
    # Save the file with full timestamp in filename
    output_file = f"{full_date_str}_crypto_{timeframe}_strong_targets.txt"
    file_path = os.path.join(date_folder, output_file)
    with open(file_path, "w") as f:
        f.write(txt_content)
    
    # Save failed cryptos for analysis
    # failed_file = f"{full_date_str}_failed_cryptos_{timeframe}.txt"
    # failed_path = os.path.join(date_folder, failed_file)
    # with open(failed_path, "w") as f:
    #     for crypto, reason in failed_targets:
    #         f.write(f"{crypto}: {reason}\n")
    
    print(f"\nResults saved to {file_path}")
    # print(f"Failed cryptos saved to {failed_path}")
