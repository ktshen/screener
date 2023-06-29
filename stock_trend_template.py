from datetime import datetime, timedelta
from src.downloader import StockDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_strategy(ticker: str, strict=False):
    """
    :param strict: If true, SMA30 > SMA45 > SMA60
    """
    print(f"Analyzing {ticker} in thread...")
    sd = StockDownloader()
    start_date = datetime.now() - timedelta(days=400)
    end_date = datetime.now()
    try:
        stock, status, stock_info = sd.get_ticker(ticker, start_date, end_date)
        if status == 0:
            if stock_info:
                print(f"{ticker} fails to get data -> {stock_info}")
            return None
        if stock_info.empty:
            return None
    except Exception as e:
        print(f"Error in getting {ticker} info: {e}")
        return None

    try:
        if len(stock_info) < 260:
            print(f"{ticker} fails to meet the requirements. Less than 260 days.")
            return {"stock": ticker, "meet_requirements": False}

        current_close = stock_info['Adj Close'].values[-1]
        previous_day_close = stock_info['Adj Close'].values[-2]
        moving_average_20 = stock_info['SMA_20'].values[-1]
        moving_average_50 = stock_info['SMA_50'].values[-1]
        moving_average_150 = stock_info['SMA_150'].values[-1]
        moving_average_200 = stock_info['SMA_200'].values[-1]
        turnover = stock_info["Volume"].values[-1] * stock_info["Adj Close"].values[-1]
        low_of_52_week = stock_info["Adj Close"].values[-260:].min()
        high_of_52_week = stock_info["Adj Close"].values[-260:].max()
        stock_info["std_of_6_days"] = stock_info["Adj Close"].rolling(window=6).std()
        #degree_of_contraction = round(stock_info["Adj Close"].std() / stock_info["std_of_6_days"].values[-1], 3)

        conditions_checklist = [False] * 9

        if turnover < 1000000:
            print(f"{ticker} fails to meet the requirements. Not enough turnover.")
            return {"stock": ticker, "meet_requirements": False}

        # Condition 1 :  Current Price > 150 SMA and Current Price > 200 SMA
        if (current_close > moving_average_150) and (current_close > moving_average_200):
            conditions_checklist[0] = True

        # Condition 2 : 150 SMA and > 200 SMA
        if moving_average_150 > moving_average_200:
            conditions_checklist[1] = True

        # Condition 3 : 200 SMA trending up for at least 1 month (ideally 4-5 months)
        # PASS NOW
        conditions_checklist[2] = True

        # Condition 4 : 50 SMA > 150 SMA and 50 SMA > 200 SMA
        if moving_average_50 > moving_average_150 > moving_average_200:
            conditions_checklist[3] = True

        # Condition 5 : Current Price > 50 SMA
        if current_close > moving_average_50:
            conditions_checklist[4] = True

        # Condition 6 : Current Price must at least outperform 52week low about 30%
        if current_close > low_of_52_week * 1.3:
            conditions_checklist[5] = True

        # Condition 7 : Current price must not below 52week high over 25%
        if current_close > high_of_52_week * 0.75:
            conditions_checklist[6] = True

        # Condition 8 : Relative strength rating needs to be over 70
        # PASS NOW
        conditions_checklist[7] = True

        # Strict mode
        # Condition 9 : SMA30 > SMA45 > SMA60
        if strict:
            moving_average_30 = stock_info['SMA_30'].values[-1]
            moving_average_45 = stock_info['SMA_45'].values[-1]
            moving_average_60 = stock_info['SMA_60'].values[-1]
            if moving_average_30 > moving_average_45 > moving_average_60:
                conditions_checklist[8] = True
        else:
            conditions_checklist[8] = True

        meet_requirements = False
        if all(conditions_checklist):
            print(f"{ticker} made the requirements")
            meet_requirements = True
        else:
            print(f"{ticker} fails to meet the requirements")

        rs_score = 0
        if meet_requirements:
            rs_score_3m = 0.4 * ((current_close - stock_info['Adj Close'].values[-63]) / stock_info['Adj Close'].values[-63])
            rs_score_6m = 0.2 * ((current_close - stock_info['Adj Close'].values[-126]) / stock_info['Adj Close'].values[-126])
            rs_score_9m = 0.2 * ((current_close - stock_info['Adj Close'].values[-189]) / stock_info['Adj Close'].values[-189])
            rs_score_12m = 0.2 * ((current_close - stock_info['Adj Close'].values[-250]) / stock_info['Adj Close'].values[-250])
            rs_score = (rs_score_3m + rs_score_6m + rs_score_9m + rs_score_12m) * 100

        return {"stock": ticker, "meet_requirements": meet_requirements, "rs_score": rs_score}

    except Exception as e:
        print(f"{ticker} failed: {e}")
        return None


if __name__ == '__main__':
    stock_downloader = StockDownloader()
    stock_downloader.check_stock_table()
    stock_downloader.update_database()
    all_symbols = stock_downloader.get_all_symbols()

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(test_strategy, symbol, False) for symbol in all_symbols]
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

    print(f"Found {len(strong_targets)} stocks that meet the requirements. Percentage: {len(strong_targets) / len(all_symbols) * 100:.2f}%")
    print(f"Strong targets: {', '.join(strong_targets)}")
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(f"{date_str}_stock_strong_targets.txt", "w") as f:
        f.write(",".join(strong_targets))
