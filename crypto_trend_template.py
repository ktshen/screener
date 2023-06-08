from datetime import datetime, timedelta
from src.downloader import CryptoDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_strategy(symbol: str):
    cd = CryptoDownloader()
    start_date = datetime.now() - timedelta(days=100)
    try:
        crypto, status, crypto_data = cd.get_crypto(symbol, start_date)
        if status == 0:
            if crypto_data:
                print(f"{symbol} fails to get data -> {crypto_data}")
            return None
        if crypto_data.empty:
            return None
    except Exception as e:
        print(f"Error in getting {symbol} info: {e}")
        return None

    conditions_checklist = [False]

    try:
        # Condition 1: Check if SMA30 > SMA45 > SMA60
        moving_average_30 = crypto_data['SMA_30'].values[-1]
        moving_average_45 = crypto_data['SMA_45'].values[-1]
        moving_average_60 = crypto_data['SMA_60'].values[-1]
        if moving_average_30 > moving_average_45 > moving_average_60:
            conditions_checklist[0] = True

        meet_requirements = False
        if all(conditions_checklist):
            print(f"{symbol} made the requirements")
            meet_requirements = True
        else:
            print(f"{symbol} fails to meet the requirements")

        return {"crypto": symbol, "meet_requirements": meet_requirements}

    except Exception as e:
        print(f"{symbol} failed: {e}")
        return None


if __name__ == '__main__':
    crypto_downloader = CryptoDownloader()
    crypto_downloader.check_crypto_table()
    crypto_downloader.update_database()
    all_cryptos = crypto_downloader.get_all_symbols()

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(test_strategy, crypto) for crypto in all_cryptos]
        results = [future.result() for future in as_completed(future_tasks)]

    strong_targets = []
    for result in results:
        if not result:
            continue
        if result["meet_requirements"]:
            strong_targets.append(result["crypto"])

    print(f"Found {len(strong_targets)} cryptos that meet the requirements. Percentage: {len(strong_targets) / len(all_cryptos) * 100:.2f}%")
    txt_content = "BINANCE:BTCUSDT, BINANCE:ETHUSDT"
    for crypto in strong_targets:
        txt_content += f",BINANCE:{crypto}"
    print(f"Strong targets: {', '.join(strong_targets)}")
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(f"{date_str}_crypto_strong_targets.txt", "w") as f:
        f.write(txt_content)
