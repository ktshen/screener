import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from src.downloader import StockDownloader

#================= CONFIGURATIONS =================#
_1D_OF_DAYS_TRACEBACK = 252
_1H_OF_DAYS_TRACEBACK = 126
CURRENT_TIMEZONE = "America/Los_Angeles"
STOCK_SMA = [20, 30, 45, 50, 60, 150, 200]
#==================================================#


def find_relative_strength(ticker: str, sd: StockDownloader):
    try:
        success, daily_stock_data = sd.request_ticker_all_range(ticker, timeframe="1d", current_tz=CURRENT_TIMEZONE)
    except Exception as e:
        print("ERROR in request_ticker_all_range")
        print(e)
        return None
    if not success:
        print(f"{ticker} fails to get data -> {daily_stock_data}")
        return None

    try:
        if len(daily_stock_data) < _1D_OF_DAYS_TRACEBACK:
            print(f"{ticker} fails to meet the requirements. Less than {_1D_OF_DAYS_TRACEBACK} days.")
            return {"stock": ticker, "meet_requirements": False}

        daily_stock_data = daily_stock_data.tail(_1D_OF_DAYS_TRACEBACK)
        daily_stock_data.reset_index(drop=True, inplace=True)

        for duration in STOCK_SMA:
            daily_stock_data["SMA_" + str(duration)] = round(daily_stock_data.loc[:, "Close"].rolling(window=duration).mean(), 2)

        current_close = daily_stock_data['Close'].values[-1]
        moving_average_50 = daily_stock_data['SMA_50'].values[-1]
        moving_average_150 = daily_stock_data['SMA_150'].values[-1]
        moving_average_200 = daily_stock_data['SMA_200'].values[-1]
        low_of_52_week = daily_stock_data["Close"].values[-260:].min()
        high_of_52_week = daily_stock_data["Close"].values[-260:].max()
        daily_stock_data["std_of_6_days"] = daily_stock_data["Close"].rolling(window=6).std()
        last_10_days_turnover = daily_stock_data.tail(10).assign(Turnover=daily_stock_data['Volume'] * daily_stock_data['Close'])
        average_turnover = last_10_days_turnover['Turnover'].mean()
        conditions_checklist = [False] * 9

        if average_turnover < 5000000:
            print(f"{ticker} fails to meet the requirements. Not enough turnover.")
            return {"stock": ticker, "meet_requirements": False}

        # Mark Minervini's Trend Template's Conditions Check
        # Condition 1 :  Current Price > 150 SMA and Current Price > 200 SMA
        if (current_close > moving_average_150) and (current_close > moving_average_200):
            conditions_checklist[0] = True

        # Condition 2 : 150 SMA and > 200 SMA
        if moving_average_150 > moving_average_200:
            conditions_checklist[1] = True

        # Condition 3 : 200 SMA trending up for at least 1 month (ideally 4-5 months)
        conditions_checklist[2] = True  # PASS NOW

        # Condition 4 : 50 SMA > 150 SMA and 50 SMA > 200 SMA
        if moving_average_50 > moving_average_150 > moving_average_200:
            conditions_checklist[3] = True
        # conditions_checklist[3] = True  # Pass now

        # Condition 5 : Current Price > 50 SMA
        if current_close > moving_average_50:
            conditions_checklist[4] = True
        conditions_checklist[4] = True  # Pass now

        # Condition 6 : Current Price must at least outperform 52week low about 30%
        if current_close > low_of_52_week * 1.3:
            conditions_checklist[5] = True

        # Condition 7 : Current price must not below 52week high over 25%
        if current_close > high_of_52_week * 0.75:
            conditions_checklist[6] = True

        # Condition 8 : Relative strength rating needs to be over 70
        # PASS NOW
        conditions_checklist[7] = True

        # Condition 9 : Avoid Penny Stock (Price < 10)
        if current_close >= 10:
            conditions_checklist[8] = True

        meet_requirements = False
        if all(conditions_checklist):
            print(f"{ticker} made the requirements")
            meet_requirements = True
        else:
            print(f"{ticker} fails to meet the requirements")

        success, one_hour_stock_data = sd.request_ticker_all_range(ticker, timeframe="1h", current_tz=CURRENT_TIMEZONE)
        if not success:
            print(f"{ticker} fails to get data -> {one_hour_stock_data}")
            return None

        start_time = '09:00:00'
        end_time = '16:00:00'
        one_hour_stock_data = one_hour_stock_data[one_hour_stock_data['Datetime'].dt.time.between(pd.to_datetime(start_time).time(), pd.to_datetime(end_time).time())]
        one_hour_stock_data = one_hour_stock_data.tail(_1H_OF_DAYS_TRACEBACK * 8)
        one_hour_stock_data.reset_index(drop=True, inplace=True)
        for duration in STOCK_SMA:
            one_hour_stock_data["SMA_" + str(duration)] = round(one_hour_stock_data.loc[:, "Close"].rolling(window=duration).mean(), 2)

        rs_score = 0.0
        bars = len(one_hour_stock_data) - 60
        if bars <= 0:
            raise ValueError("Not enough days to calculate MA60")
        weights = np.exp(-0.0015 * np.arange(bars))
        weights /= np.sum(weights)
        for i in range(1, bars + 1):
            # close = one_hour_stock_data["Close"].values[-i]
            one_hour_ma30 = one_hour_stock_data['SMA_30'].values[-i]
            one_hour_ma45 = one_hour_stock_data['SMA_45'].values[-i]
            one_hour_ma60 = one_hour_stock_data['SMA_60'].values[-i]
            # weight = ((close - one_hour_ma30) + (close - four_hour_ma45) + (close - four_hour_ma60) + (one_hour_ma30 - four_hour_ma45) + (one_hour_ma30 - four_hour_ma60) + (four_hour_ma45 - four_hour_ma60)) / four_hour_ma60
            M = ((one_hour_ma30 - one_hour_ma45) + (one_hour_ma30 - one_hour_ma60) + (one_hour_ma45 - one_hour_ma60)) / one_hour_ma60
            rs_score += M * weights[i-1]
            # rs_score += M * (bars - i)
        return {"stock": ticker, "meet_requirements": meet_requirements, "rs_score": rs_score}

    except Exception as e:
        print(f"{ticker} failed: {e}")
        return None


if __name__ == '__main__':
    stock_downloader = StockDownloader()
    all_tickers = stock_downloader.get_all_tickers()

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_tasks = [executor.submit(find_relative_strength, ticker, stock_downloader) for ticker in all_tickers]
        results = [future.result() for future in as_completed(future_tasks)]

    strong_targets = []
    target_rs_score = {}
    for result in results:
        if not result:
            continue
        if result["meet_requirements"]:
            strong_targets.append(result["stock"])
            target_rs_score[result["stock"]] = result["rs_score"]
    strong_targets.sort(key=lambda x: target_rs_score[x], reverse=True)

    print(f"Found {len(strong_targets)} stocks that meet the requirements. Percentage: {len(strong_targets) / len(all_tickers) * 100:.2f}%")
    print(f"Strong targets: {', '.join(strong_targets)}")
    print("============================== Target : Score (TOP 50) ==============================")
    for ticker in strong_targets[:50]:
        score = target_rs_score[ticker]
        print(f"{ticker}: {score}")
    print("========================================================================================")
    date_str = datetime.now().strftime("%Y-%m-%d")
    txt_content = "###INDEX\nSPY,IXIC,DJI\n###TARGETS\n"
    txt_content += ",".join(strong_targets[:980])
    with open(f"{date_str}_stock_strong_targets_using_weight.txt", "w") as f:
        f.write(txt_content)
