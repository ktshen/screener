import ccxt
import talib
import argparse
import pandas as pd
import numpy as np
import concurrent.futures
import os
from discord_webhook import DiscordWebhook
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone

# 创建币安现货交易所
exchange = ccxt.binance() 

def slope(point0, point1, point2):
    gap = point2 - point0
    normalization_point = (point1 - point0) / gap
    normalization_slope = (1 - normalization_point) / 0.5
    return normalization_slope

# 获取所有交易对 
keywords = ["ABC", "UP", "DOWN", "BULL", "BEAR", "GXS"]
symbols = exchange.load_markets().keys()  
usdt_symbols = [s for s in symbols if s[-5:] == "/USDT"]
cleaned_list = [x for x in usdt_symbols if all(key not in x for key in keywords)]
timeframes = ['1d', '4h', '2h', '1h', '15m']
ema_cols = ['ema20', 'ema50', 'ema200']

def run(symbol):
    conditions_met = {
        '1d': False, 
        '4h': False,  
        '2h': False,
        '1h': False,
        '15m': False
    }
    
    for timeframe in timeframes:
        # 获取历史K线数据
        bars = None
        if timeframe == "1d":
            bars = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
        else:
            bars = exchange.fetch_ohlcv(symbol, timeframe, limit=280)

        df_frames = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        closes = df_frames['close']
        
        # 计算EMA
        ema20 = talib.EMA(closes, 20).replace({np.nan: None}).tolist()
        ema50 = talib.EMA(closes, 50).replace({np.nan: None}).tolist()
        ema200 = talib.EMA(closes, 200).replace({np.nan: None}).tolist()

        # 判断条件
        condition1 = False
        if ema20[-1] is not None and ema50[-1] is not None and ema200[-1] is not None:
            condition1 = ema20[-1] > ema50[-1] and ema50[-1] > ema200[-1]

        if timeframe == "1d":
            if ema20[-1] is not None and ema50[-1] is not None:
                condition5 = ema20[-1] > ema50[-1]
                conditions_met[timeframe] = condition5
        elif timeframe == "1h":
            last_close = closes.tail(60)
            test_ema200_60 = talib.EMA(closes, 200).tail(60)
            mask = last_close < test_ema200_60
            condition5 = mask.sum() > 0
            conditions_met[timeframe] = all([condition1, not condition5])
        elif timeframe == "4h":
            condition5 = False
            if (None in ema20[-3:]):
                break
            four_hour_slope = slope(ema20[-1],ema20[-2],ema20[-3])
            if four_hour_slope >= 1:
                condition5 = True
                conditions_met[timeframe] = all([condition1, condition5])
        else:
        # 设置条件结果
            conditions_met[timeframe] = all([condition1])

    return (symbol, conditions_met)

def execute(webhook_url):
    results = []
    selected_symbols = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        
        futures = []
        for symbol in usdt_symbols:
            futures.append(executor.submit(
                run, symbol))
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    selected_symbols = [r[0] for r in results if all(r[1].values())]
    print("Selected symbols: ", selected_symbols) 

    formatted_symbols = [f"BINANCE:{symbol.replace('/', '')}" for symbol in selected_symbols]
    doc_content = "###教主嚴選,"

    # 输出路径 
    output_path = os.path.join(os.getcwd(), "output.txt")  

    # 写入文档
    with open(output_path, "w") as f:
        f.write(doc_content + ",".join(formatted_symbols))
    # 过滤并打印结果  
    webhook = DiscordWebhook(url=webhook_url, content=", ".join(selected_symbols))
    with open("output.txt", "rb") as f:
        webhook.add_file(file=f.read(), filename="test.txt")
    response = webhook.execute()
    print(response)

if __name__ == '__main__':
    print("aaa")
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--webhook', type=str, help='discord webhook url', default='')
    args = parser.parse_args()

    webhook_url = args.webhook

    scheduler = BlockingScheduler()
    scheduler.timezone = timezone('Asia/Taipei')

    scheduler.add_job(execute, 'cron', hour=8, minute=5, args=[webhook_url])

    scheduler.add_job(execute, 'cron', hour=20, minute=5, args=[webhook_url])

    scheduler.start()
