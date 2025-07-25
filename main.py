from swing_trader.data.mr_data import MrData
from swing_trader.trading.portfolio import Portfolio
from swing_trader.strategy.swing_strategy import SwingStrategy
from datetime import datetime, timedelta
from swing_trader.backtest.backtester import Backtester

start_dt = datetime(2018, 1, 1)
end_dt = datetime.today() - timedelta(days=1)
start_date = start_dt.strftime("%Y-%m-%d")
end_date = end_dt.strftime("%Y-%m-%d")
signal_ticker = "SPY"
long_ticker = "UPRO"
# long_ticker = "SSO"
short_ticker = "SH"

# TODO: Add Parameters for ema, macd, rsi in strategy?
#  Have the strategy inputs be the indicators to use?
#  Strategy takes in the database and tickers, so can create the indicators there since the strategy can adjust them?

portfolio = Portfolio(cash=10000.0, name="Strategy Portfolio")
strategy = SwingStrategy(
    db=MrData().db,
    signal_ticker=signal_ticker,
    long_ticker=long_ticker,
    short_ticker=short_ticker,
    rsi_high=75,
    rsi_low=25
)

# TODO: Are these the right parameters for Backtesting?
#  Store backtest results in database.  Each run unique, so can store the parameters used.
#  Store the strategy name, parameters, and results in a collection.
backtester = Backtester(
    db=MrData().db,
    tickers=[long_ticker, signal_ticker],
    strategy=strategy,
    portfolio=portfolio,
    start_date=start_date,
    end_date=end_date
)
backtester.prepare_data()
backtester.run_strategy()
backtester.simulate_portfolios()
backtester.report()
