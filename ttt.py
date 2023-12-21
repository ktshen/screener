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
import time
# 创建币安现货交易所
# exchange = ccxt.binance() 
binance = ccxt.binance()
bybit = ccxt.bybit()
okx = ccxt.okx()

def slope(point0, point1, point2):
    gap = point2 - point0
    normalization_point = (point1 - point0) / gap
    normalization_slope = (1 - normalization_point) / 0.5
    return normalization_slope

def return_usdt_symbols(symbols):
    temp = [s for s in symbols if s[-5:] == "/USDT"]
    return [x for x in temp if all(key not in x for key in keywords)]


def select_symbols(results):
    return [r[0] for r in results if all(r[1].values())]
# 获取所有交易对 
keywords = ["ABC", "UP", "DOWN", "BULL", "BEAR", "GXS", "2L", "2S", "3L", "3S"]
# symbols = exchange.load_markets().keys()  
binance_symbols = binance.load_markets().keys()
bybit_symbols = bybit.load_markets().keys()
okx_symbols = okx.load_markets().keys()

binance_usdt_symbols = return_usdt_symbols(binance_symbols)
bybit_usdt_symbols = return_usdt_symbols(bybit_symbols)
okx_usdt_symbols =  return_usdt_symbols(okx_symbols)

bybit_only_symbols = list(set(bybit_usdt_symbols) - set(binance_usdt_symbols))
okx_only_symbols = list(set(okx_usdt_symbols) - set(binance_usdt_symbols) - set(bybit_usdt_symbols))

timeframes = ['1d', '4h', '2h', '1h', '15m']
ema_cols = ['ema20', 'ema50', 'ema200']

def run(symbol, exchange):
    conditions_met = {
        '1d': False,  
        '4h': False,  
        '2h': False,
        '1h': False,
        '15m': False
    }
    
    for timeframe in timeframes:
        time.sleep(0.2)
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
            result = (last_close > test_ema200_60).all()
            conditions_met[timeframe] = all([condition1, result])
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

def execute(webhook_url = None):
    binance_results = []
    bybit_results = []
    okx_results = []

    # selected_symbols = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        
        binance_futures = []
        bybit_futures = []
        okx_futures = []

        # 幣安
        for symbol in binance_usdt_symbols:
            binance_futures.append(executor.submit(
                run, symbol, binance))
        
        for future in concurrent.futures.as_completed(binance_futures):
            binance_results.append(future.result())

        # bybit
        for symbol in bybit_only_symbols:
            bybit_futures.append(executor.submit(
                run, symbol, bybit))
        
        for future in concurrent.futures.as_completed(bybit_futures):
            bybit_results.append(future.result())

        #okx
        for symbol in okx_only_symbols:
            okx_futures.append(executor.submit(
                run, symbol, okx))
        
        for future in concurrent.futures.as_completed(okx_futures):
            okx_results.append(future.result())

    binance_selected_symbols = select_symbols(binance_results)
    bybit_selected_symbols = select_symbols(bybit_results)
    okx_selected_symbols = select_symbols(okx_results)
    print("BINANCE Selected symbols: ", binance_selected_symbols) 
    print("BYBIT Selected symbols: ", bybit_selected_symbols) 
    print("OKX Selected symbols: ", okx_selected_symbols) 

    binance_formatted_symbols = [f"BINANCE:{symbol.replace('/', '')}" for symbol in binance_selected_symbols]
    bybit_formatted_symbols = [f"BYBIT:{symbol.replace('/', '')}" for symbol in bybit_selected_symbols]
    okx_formatted_symbols = [f"OKX:{symbol.replace('/', '')}" for symbol in okx_selected_symbols]
    
    binance_doc_content = "###幣安,"
    bybit_doc_content = "###BYBIT,"
    okx_doc_content = "###OKX,"

    # 输出路径 
    output_path = os.path.join(os.getcwd(), "output.txt")  

    # 写入文档
    with open(output_path, "w") as f:
        f.write(binance_doc_content + ",".join(binance_formatted_symbols) + "\n")
        f.write(bybit_doc_content + ",".join(bybit_formatted_symbols) + "\n")
        f.write(okx_doc_content + ",".join(okx_formatted_symbols))

    # 过滤并打印结果  
    if webhook_url is not None:
        webhook = DiscordWebhook(url=webhook_url, content=", ".join(binance_selected_symbols))
        with open("output.txt", "rb") as f:
            webhook.add_file(file=f.read(), filename="test.txt")
        response = webhook.execute()
        print(response)

if __name__ == '__main__':
    print("aaa")
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--webhook', type=str, help='discord webhook url', default='')
    args = parser.parse_args()

    execute()
    # webhook_url = args.webhook

    # scheduler = BlockingScheduler()
    # scheduler.timezone = timezone('Asia/Taipei')

    # scheduler.add_job(execute, 'cron', hour=8, minute=5, args=[webhook_url])

    # scheduler.add_job(execute, 'cron', hour=20, minute=5, args=[webhook_url])

    # scheduler.start()
