import pandas as pd
from datetime import datetime, timedelta
from swing_trader.data.data_loader import DataLoader
from swing_trader.strategy.swing_strategy import SwingStrategy
from swing_trader.backtest.backtester import Backtester
import pandas_market_calendars as mcal



# 1. Load SPY data for the last 5 years and cache in DB if needed
data_loader = DataLoader()

# Get the last valid business day in the US stock market calendar
start_date = datetime.today() - timedelta(days=5*365)
nyse = mcal.get_calendar('NYSE')
valid_days = nyse.valid_days(start_date=start_date, end_date=datetime.today())
if len(valid_days) == 0:
    raise Exception("No valid trading days in range!")
end_date = valid_days[-1].to_pydatetime()

df_records = data_loader.get_stock_data(
    'SPY',
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

if not df_records:
    print("No data found for requested end date. Trying with last date in DB...")
    all_docs = data_loader.mongo_client.find_stock_data('SPY')
    if not all_docs:
        raise Exception("No stock data loaded at all! Check your loader.")
    all_dates = [d['date'] for d in all_docs if 'date' in d]

    last_db_date = max(all_dates)
    print("Retrying with last available date in DB:", last_db_date)
    df_records = data_loader.get_stock_data(
        'SPY',
        start_date.strftime('%Y-%m-%d'),
        last_db_date
    )

if not df_records:
    raise Exception("Still no stock data loaded. Check your data loader or DB.")

# 2. Convert to DataFrame for inspection (optional)
df = pd.DataFrame(df_records)
print(f"Loaded SPY data from {df['date'].min()} to {df['date'].max()}")
print(df.head())

# 3. Generate signals using SwingStrategy (fetches from DB)
strategy = SwingStrategy('SPY', short_window=20, long_window=50)
signals_df = strategy.generate_signals()  # DataFrame with 'signal', 'date', 'close', etc.

print("Signals sample:")
print(signals_df[['date', 'close', 'sma_short', 'sma_long', 'signal']].tail())

# 4. Backtest
backtester = Backtester(signals_df)
trades = backtester.run()
pnl = backtester.calculate_pnl()

# 5. Calculate final portfolio value
starting_value = 1000
final_value = starting_value + pnl

print(f"Total PnL: {pnl:.2f}")
print(f"Final Portfolio Value: {final_value:.2f}")
print("Trades:")
for t in trades:
    print(t)
