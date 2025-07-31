import backtrader as bt
import pandas as pd

from ggTrader.strats.ema_macd_rsi import EmaMacdRsiStrategy  # if saved as a separate file
from ggTrader.data_manager.universal_data_manager import UniversalDataManager

import os
from dotenv import load_dotenv
from tabulate import tabulate
load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
dm = UniversalDataManager(mongo_uri=mongo_uri)
your_dataframe = dm.load_or_fetch("SPY", "1d", "2020-01-01", "2023-01-01", market="stock")
# Sample data (replace with your real data)
data = bt.feeds.PandasData(dataname=your_dataframe)

# Initialize Cerebro
cerebro = bt.Cerebro()  # Set to 1 to avoid multiprocessing issues with some analyzers
cerebro.adddata(data)

# Optimization: vary fast/slow EMAs and RSI period
cerebro.optstrategy(
    EmaMacdRsiStrategy,
    ema_fast=range(5, 15, 2),
    ema_slow=range(20, 40, 5),
    rsi_period=range(10, 20, 2),
)

# Add analyzers (e.g., Sharpe Ratio)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

# Run optimization
results = cerebro.run()

# Flatten the nested results
flat_results = [res[0] for res in results]

# Sort and display results by Sharpe Ratio
sorted_by_sharpe = sorted(
    flat_results,
    key=lambda strat: strat.analyzers.sharpe.get_analysis().get('sharperatio', float('-inf')),
    reverse=True
)

# Print top 5 parameter sets
for strat in sorted_by_sharpe[:5]:
    p = strat.params
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    print(f"Fast: {p.ema_fast}, Slow: {p.ema_slow}, RSI: {p.rsi_period}, Sharpe: {sharpe}")
