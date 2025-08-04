from data_manager import StockDataManager, CryptoDataManager
from datetime import datetime, timedelta
from tabulate import tabulate


sdm = StockDataManager()

symbol = "SPY"
interval='1d'
end_date = datetime(2025, 7, 28)
start_date = end_date - timedelta(days=7)

data = sdm.get_market_days(start_date, end_date)



mm = sdm.get_stock_data(symbol, interval, start_date, end_date)

print(tabulate(mm, headers='keys', tablefmt='github'))
print("Crypto Data...")
cm = CryptoDataManager()

symbol = "XRPUSDT"
interval = '1d'
end_date = datetime(2025, 7, 28)
start_date = end_date - timedelta(days=30)
cm_data = cm.get_crypto_data(symbol, interval, start_date, end_date)

print(tabulate(cm_data, headers='keys', tablefmt='github'))

print("Top Pairs:")
cm.print_top_pairs(top_n=20, quote="USDT")