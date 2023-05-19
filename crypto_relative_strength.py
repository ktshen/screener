from datetime import datetime, timedelta
from src.downloader import CryptoDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed







if __name__ == '__main__':
    crypto_downloader = CryptoDownloader()
    crypto_downloader.check_crypto_table()
    crypto_downloader.update_database()
    all_symbols = stock_downloader.get_all_symbols()

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_tasks = [executor.submit(test_strategy, symbol) for symbol in all_symbols]
        results = [future.result() for future in as_completed(future_tasks)]

    strong_targets = []
    for result in results:
        if not result:
            continue
        if result["meet_requirements"]:
            strong_targets.append(result["stock"])

    print(f"Found {len(strong_targets)} stocks that meet the requirements. Percentage: {len(strong_targets) / len(all_symbols) * 100:.2f}%")
    print(f"Strong targets: {', '.join(strong_targets)}")
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(f"{date_str}_strong_targets.txt", "w") as f:
        f.write(",".join(strong_targets))


