import pandas as pd
from datetime import datetime, timedelta
from swing_trader.data.mr_data import MrData
from swing_trader.strategy.swing_strategy import SwingStrategy
from swing_trader.backtest.backtester import Backtester
import pandas_market_calendars as mcal

mr_data = MrData()

df = mr_data.get_stock_data('SPY', '2015-01-01', '2017-12-31')
print(df.head())