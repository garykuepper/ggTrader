import pandas as pd
from datetime import datetime, timedelta, timezone
from trading_strategy import MovingAverageStrategy, EMA_Strategy
from data_manager import CryptoDataManager
# Example: create synthetic price data for 30 periods
# dates = pd.date_range(start="2025-08-01", periods=30, freq="D")
# prices = [100 + i*0.5 for i in range(30)]  # simple upward trend
#
# # Initialize strategy with $1000 capital
# strategy = MovingAverageStrategy(capital=1000, short_window=3, long_window=5)
#
# # Simulate feeding the data one point at a time
# for date, price in zip(dates, prices):
#     strategy.on_new_data(price, date)
#
# print(f"Final capital: {strategy.capital:.2f}")


# Load CSV with at least 'Date' and 'Close' columns
# df = pd.read_csv('your_historical_data.csv', parse_dates=['Date'])

cm = CryptoDataManager()
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=30)
df = cm.get_crypto_data("LTCUSDT", "4h", start_date, end_date)

strategy = EMA_Strategy(capital=1000, short_window=5, long_window=21, trailing_stop_pct=0.03)



strategy.on_new_data(df)
current_value = strategy.get_position_value()

print(f"Final capital: {strategy.capital:.2f}")
print(f"Open position value: {current_value:.2f}")
