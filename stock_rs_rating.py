"""
20230515
Calculate the relative strength rating for one stock or a list of stocks.
Parameters:
    -s : the stock symbol
    -t : text filename

Reference[https://github.com/skyte/relative-strength]
"""
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from src.downloader import StockDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_relative_strength_rating(closes: pd.Series, closes_ref: pd.Series):
    if len(closes) != len(closes_ref):
        raise ValueError("Closes and closes_ref must have the same length")
    if len(closes) < 252 or len(closes_ref) < 252:
        raise ValueError("Closes and closes_ref must have at least 252 entries")
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)
    rs = (1 + rs_stock) / (1 + rs_ref) * 100
    rs = int(rs * 100) / 100  # round to 2 decimals
    return rs


def strength(closes: pd.Series):
    """Calculates the performance of the last year (most recent quarter is weighted double)"""
    try:
        quarters1 = quarters_perf(closes, 1)
        quarters2 = quarters_perf(closes, 2)
        quarters3 = quarters_perf(closes, 3)
        quarters4 = quarters_perf(closes, 4)
        return 0.4*quarters1 + 0.2*quarters2 + 0.2*quarters3 + 0.2*quarters4
    except Exception as e:
        print(f"Failed to calculate strength performance: {e}")
        return 0


def quarters_perf(closes: pd.Series, n):
    length = min(len(closes), n*int(252/4))
    prices = closes.tail(length)
    pct_chg = prices.pct_change().dropna()
    perf_cum = (pct_chg + 1).cumprod() - 1
    return perf_cum.tail(1).item()


def run_job(symbol, start_date, end_date, ref_adj_close):
    try:
        symbol_c, symbol_status, symbol_data = sd.get_ticker(symbol, start_date, end_date)
        if symbol_status == 0:
            print(f"{symbol} has no correct data")
        score = calculate_relative_strength_rating(symbol_data["Adj Close"], ref_adj_close)
        return symbol, score
    except Exception as e:
        print(f"{symbol} -> Failed to calculate relative strength rating: {e}")
        return symbol, None


if __name__ == "__main__":
    sd = StockDownloader()
    sd.check_stock_table()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", help="Symbol", type=str, default="")
    parser.add_argument("-t", help="Text file including all the symbols",
                        type=str, default="{date}_strong_targets.txt".format(date=datetime.now().strftime("%Y-%m-%d")))
    args = parser.parse_args()
    reference_symbol = "SPY"
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    if args.s:
        symbols = [args.s]
    elif args.t:
        with open(args.t, "r") as f:
            symbols = f.read().split(",")

    ref_c, ref_status, ref_data = sd.get_ticker(reference_symbol, start_date, end_date)
    if ref_status == 0:
        print(f"{reference_symbol} has no correct data")
        exit()

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(run_job, symbol, start_date, end_date, ref_data["Adj Close"]) for symbol in symbols]
        for future in as_completed(future_tasks):
            symbol, data = future.result()
            if data is not None:
                results.append((symbol, data))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    print(f"==========RS Rating Results===========")
    for symbol, rs_rating in results:
        print(f"{symbol} vs {reference_symbol} -> {rs_rating}")

    if args.t:
        filename = os.path.splitext(args.t)[0]
        with open(f"{filename}_rs_rating.txt", "w") as f:
            f.write(",".join([x[0] for x in results]))
