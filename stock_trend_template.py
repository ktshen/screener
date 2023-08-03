from datetime import datetime, timedelta
from pytz import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from discord_webhook import DiscordWebhook
from src.downloader import StockDownloader
import schedule
import time

##################### CONFIGURATIONS #####################
CURRENT_TIMEZONE = "America/Los_Angeles"
##########################################################

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--webhook', type=str, help='discord webhook url', default='')
args = parser.parse_args()
webhook_url = args.webhook

def test_strategy(ticker: str, strict=False):
    """
    :param strict: If true, SMA30 > SMA45 > SMA60
    """
    print(f"Analyzing {ticker} in thread...")
    sd = StockDownloader()
    start_date = timezone(CURRENT_TIMEZONE).localize(datetime.now() - timedelta(days=480))
    end_date = timezone(CURRENT_TIMEZONE).localize(datetime.now())
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
        if len(stock_info) < 320:
            print(f"{ticker} fails to meet the requirements. Less than 260 days.")
            return {"stock": ticker, "meet_requirements": False}

        current_close = stock_info['Adj Close'].values[-1]
        moving_average_50 = stock_info['SMA_50'].values[-1]
        moving_average_150 = stock_info['SMA_150'].values[-1]
        moving_average_200 = stock_info['SMA_200'].values[-1]
        low_of_52_week = stock_info["Adj Close"].values[-260:].min()
        high_of_52_week = stock_info["Adj Close"].values[-260:].max()
        stock_info["std_of_6_days"] = stock_info["Adj Close"].rolling(window=6).std()
        last_30_days_turnover = stock_info.tail(30).assign(Turnover=stock_info['Volume'] * stock_info['Adj Close'])
        average_turnover = last_30_days_turnover['Turnover'].mean()
        conditions_checklist = [False] * 9

        if average_turnover < 1000000:
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

        # IBD RS ranking style
        # rs_score = 0
        # if meet_requirements:
        #     rs_score_3m = 0.4 * ((current_close - stock_info['Adj Close'].values[-63]) / stock_info['Adj Close'].values[-63])
        #     rs_score_6m = 0.2 * ((current_close - stock_info['Adj Close'].values[-126]) / stock_info['Adj Close'].values[-126])
        #     rs_score_9m = 0.2 * ((current_close - stock_info['Adj Close'].values[-189]) / stock_info['Adj Close'].values[-189])
        #     rs_score_12m = 0.2 * ((current_close - stock_info['Adj Close'].values[-252]) / stock_info['Adj Close'].values[-252])
        #     rs_score = (rs_score_3m + rs_score_6m + rs_score_9m + rs_score_12m) * 100
        rs_score = 0.0
        bars = 252
        for i in range(1, bars + 1):
            close = stock_info["Adj Close"].values[-i]
            moving_average_30 = stock_info['SMA_30'].values[-i]
            moving_average_45 = stock_info['SMA_45'].values[-i]
            moving_average_60 = stock_info['SMA_60'].values[-i]
            weight = (((close - moving_average_30) + (close - moving_average_45) + (close - moving_average_60)) * (((bars - i) * 4 / bars) + 1) + (moving_average_30 - moving_average_45) + (moving_average_30 - moving_average_60) + (moving_average_45 - moving_average_60)) / moving_average_60
            rs_score += weight * (bars - i)

        return {"stock": ticker, "meet_requirements": meet_requirements, "rs_score": rs_score}

    except Exception as e:
        print(f"{ticker} failed: {e}")
        return None

def main(webhook_url):
    stock_downloader = StockDownloader()
    stock_downloader.check_stock_table()
    stock_downloader.update_database()
    all_symbols = stock_downloader.get_all_symbols()

    print(webhook_url)

    with ThreadPoolExecutor(max_workers=32) as executor:
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
    print("============================== Target : Score (TOP 50) ==============================")
    for crypto in strong_targets[:50]:
        score = target_rs_score[crypto]
        print(f"{crypto}: {score}")
    print("========================================================================================")
    date_str = datetime.now().strftime("%Y-%m-%d")
    webhook_str = f'{date_str} 美股標的篩選\n1~10: {", ".join(strong_targets[0:10])}\n11~20: {", ".join(strong_targets[10:20])}\n21~30: {", ".join(strong_targets[20:30])}\n31~40: {", ".join(strong_targets[30:40])}\n41~50: {", ".join(strong_targets[40:50])}'
    print(webhook_str)
    if webhook_url:
        webhook = DiscordWebhook(url=webhook_url, content=webhook_str)
        response = webhook.execute()
        print(response)
    # txt_content = "###INDEX\nSPY,IXIC,DJI\n###TARGETS\n"
    # txt_content += ",".join(strong_targets)
    # with open(f"{date_str}_stock_strong_targets.txt", "w") as f:
    #     f.write(txt_content)

schedule.every().day.at("12:00").do(main, webhook_url=webhook_url)

while True:
    # print("Start schedule")
    schedule.run_pending()
    time.sleep(60)