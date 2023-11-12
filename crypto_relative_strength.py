import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.downloader import CryptoDownloader
from discord_webhook import DiscordWebhook
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone
from dotenv import load_dotenv
##################### CONFIGURATIONS #####################
CURRENT_TIMEZONE = "America/Los_Angeles"
##########################################################

def calc_total_bars(time_interval, days):
    bars_dict = {
        "5m": 12 * 24 * days,
        "15m": 4 * 24 * days,
        "30m": 2 * 24 * days,
        "1h":  24 * days,
        "2h": 12 * days,
        "4h": 6 * days,
    }
    return bars_dict.get(time_interval)


def test_strategy(symbol: str, time_interval: str, days: int):
    try:
        cd = CryptoDownloader()
        crypto, status, crypto_data = cd.get_crypto(symbol, time_interval=time_interval, timezone=CURRENT_TIMEZONE)
        if status == 0:
            if crypto_data:
                print(f"{symbol} fails to get data -> {crypto_data}")
            return {"crypto": symbol, "rs_score": 0}
        if crypto_data.empty:
            return {"crypto": symbol, "rs_score": 0}
    except Exception as e:
        print(f"Error in getting {symbol} info: {e}")
        return {"crypto": symbol, "rs_score": 0}

    bars = calc_total_bars(time_interval, days)
    if bars > 1500 - 60:
        raise ValueError(f"Requesting too many bars. Limitation: 1440 bars. Your are requesting {bars} bars. Please decrease total days.")
    if len(crypto_data) < bars + 60:
        return {"crypto": symbol, "rs_score": 0}

    rs_score = 0.0
    for i in range(1, bars+1):
        current_close = crypto_data['Close Price'].values[-i]
        moving_average_30 = crypto_data['SMA_30'].values[-i]
        moving_average_45 = crypto_data['SMA_45'].values[-i]
        moving_average_60 = crypto_data['SMA_60'].values[-i]
        weight = (((current_close - moving_average_30) + (current_close - moving_average_45) + (current_close - moving_average_60)) * (((bars - i) * days / bars) + 1) + (moving_average_30 - moving_average_45) + (moving_average_30 - moving_average_60) + (moving_average_45 - moving_average_60)) / moving_average_60
        rs_score += weight * (bars - i)

    return {"crypto": symbol, "rs_score": rs_score}

# if __name__ == '__main__':
def crpto(timeframe, total_days, webhook_url):
    print("crypto")
    crypto_downloader = CryptoDownloader()
    crypto_downloader.check_crypto_table()
    all_cryptos = crypto_downloader.get_all_symbols()

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(test_strategy, crypto, timeframe, total_days) for crypto in all_cryptos]
        results = [future.result() for future in as_completed(future_tasks)]

    failed_targets = []     # Failed to download data or error happened
    target_score = {}
    for result in results:
        if result["rs_score"] != 0:
            target_score[result["crypto"]] = result["rs_score"]
        else:
            failed_targets.append(result["crypto"])
    targets = [x for x in target_score.keys()]
    targets.sort(key=lambda x: target_score[x], reverse=True)
    # Show results
    print("Failed targets: %s" % ", ".join(failed_targets))
    print("\n=========================== Target : Score (TOP 20) ===========================")
    strong_targets = []
    for crypto in targets[:20]:
        score = target_score[crypto]
        strong_targets.append(crypto.split('USDT')[0])
        print(f"{crypto}: {score}")
    print("===============================================================================")
    print(strong_targets[0:10])
    print(strong_targets[11:])
    local_date = datetime.now().date()
    gmt_offset = timedelta(hours=8)
    gmt_date = local_date + gmt_offset
    d1 = gmt_date.strftime("%Y/%m/%d")
    date_str = d1.replace('/', '_')
    high_potential_target = ', '.join(strong_targets[0:10])
    moderate_potential_target = ', '.join(strong_targets[10:])
    top_10_volume = crypto_downloader.get_futures_top(30)
    important_targets = crypto_downloader.find_important_crypto_targets(top_10_volume, strong_targets)
    important_targets_str = ', '.join(important_targets)
    crypto_downloader.write_targets_file(date_str, {"重點標的": important_targets, "強勢標的": strong_targets, "前三十交易量":top_10_volume})
    webhook_str = f'{d1} 標的篩選\n強勢標的: {high_potential_target}\n次強勢標的: {moderate_potential_target}\n重點標的: {important_targets_str}'
    print(webhook_str)
    if webhook_url:
        webhook = DiscordWebhook(url=webhook_url, content=webhook_str)
        with open(f"{date_str}_標的.txt", "rb") as f:
            webhook.add_file(file=f.read(), filename=f"{date_str}_strong_targets.txt")
        response = webhook.execute()
        print(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeframe', type=str, help='Time frame (3m, 5m, 15m, 30m, 1h, 2h, 4h)', default="4h")
    parser.add_argument('-d', '--total_days', type=int, help='Calculation duration in days (default 7 days)', default=5)
    parser.add_argument('-u', '--webhook', type=str, help='discord webhook url', default='')
    args = parser.parse_args()
    timeframe = args.timeframe
    total_days = args.total_days
    webhook_url = args.webhook

    scheduler = BlockingScheduler()
    scheduler.timezone = timezone('Asia/Taipei')

    scheduler.add_job(crpto, 'cron', hour=8, args=[timeframe, total_days, webhook_url])

    scheduler.add_job(crpto, 'cron', hour=20, args=[timeframe, total_days, webhook_url])

    # scheduler.add_job(crpto, 'cron', hour=14, minute=30, args=[timeframe, total_days, webhook_url])

    # scheduler.add_job(crpto, 'cron', hour=13, minute=15, args=[timeframe, total_days, webhook_url])

    scheduler.start()

    # Write to txt file
    # txt_content = "###BTCETH\nBINANCE:BTCUSDT.P,BINANCE:ETHUSDT\n###Targets (Sort by score)\n"
    # for crypto in targets:
    #     txt_content += f"BINANCE:{crypto}.P,"
    # date_str = datetime.now().strftime("%Y-%m-%d %H%M")
    # with open(f"{date_str}_crypto_relative_strength_{timeframe}_{total_days}.txt", "w") as f:
    #     f.write(txt_content)
    
