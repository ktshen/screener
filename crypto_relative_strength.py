from datetime import datetime
from src.downloader import CryptoDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_strategy(symbol: str):
    try:
        cd = CryptoDownloader()
        crypto, status, crypto_data = cd.get_crypto(symbol, time_interval="15m")
        if status == 0:
            if crypto_data:
                print(f"{symbol} fails to get data -> {crypto_data}")
            return {"crypto": symbol, "rs_score": 0}
        if crypto_data.empty:
            return {"crypto": symbol, "rs_score": 0}
    except Exception as e:
        print(f"Error in getting {symbol} info: {e}")
        return {"crypto": symbol, "rs_score": 0}

    bars = 672   # 7 days
    if len(crypto_data) < bars + 60:
        return {"crypto": symbol, "rs_score": 0}

    score = 0
    for i in range(bars):
        current_close = crypto_data['Close Price'].values[-i]
        moving_average_30 = crypto_data['SMA_30'].values[-i]
        moving_average_45 = crypto_data['SMA_45'].values[-i]
        moving_average_60 = crypto_data['SMA_60'].values[-i]
        weight = i // 4
        if current_close > moving_average_30 > moving_average_45 > moving_average_60:
            score += 8 * weight
        elif moving_average_30 > moving_average_45 > moving_average_60:
            score += 6 * weight
        elif moving_average_45 > moving_average_30 > moving_average_60:
            score += 5 * weight
        elif moving_average_45 > moving_average_60 > moving_average_30:
            score += 4 * weight
        elif moving_average_30 > moving_average_60 > moving_average_45:
            score += 3 * weight
        elif moving_average_60 > moving_average_30 > moving_average_45:
            score += 2 * weight
        elif moving_average_60 > moving_average_45 > moving_average_30:
            score += 1 * weight
    return {"crypto": symbol, "rs_score": score}


if __name__ == '__main__':
    crypto_downloader = CryptoDownloader()
    crypto_downloader.check_crypto_table()
    all_cryptos = crypto_downloader.get_all_symbols()

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(test_strategy, crypto) for crypto in all_cryptos]
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
    print("\n============================== Target : Score (TOP 20) ==============================")
    for crypto in targets[:20]:
        score = target_score[crypto]
        print(f"{crypto}: {score}")
    print("========================================================================================")
    # Write to txt file
    txt_content = "###BTCETH\nBINANCE:BTCUSDT.P,BINANCE:ETHUSDT\n###Targets (Sort by score)\n"
    for crypto in targets:
        txt_content += f",BINANCE:{crypto}.P"
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(f"{date_str}_crypto_relative_strength.txt", "w") as f:
        f.write(txt_content)
