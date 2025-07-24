from swing_trader.data.mr_data import MrData
from swing_trader.trading.portfolio import Portfolio
from swing_trader.strategy.swing_strategy import SwingStrategy
from datetime import datetime, timedelta
from swing_trader.backtest.backtester import Backtester

start_dt = datetime(2015, 1, 5)
end_dt = datetime.today() - timedelta(days=1)
start_date = start_dt.strftime("%Y-%m-%d")
end_date = end_dt.strftime("%Y-%m-%d")

portfolio = Portfolio(cash=10000.0, name="Strategy Portfolio")
strategy = SwingStrategy(
    db=MrData().db,
    signal_ticker="SPY",
    long_ticker="SSO",
    short_ticker="SH",
    rsi_high=75,
    rsi_low=25
)

backtester = Backtester(
    db=MrData().db,
    tickers=["SSO", "SPY"],
    strategy=strategy,
    portfolio=portfolio,
    start_date=start_date,
    end_date=end_date
)
backtester.prepare_data()
backtester.run_strategy()
backtester.simulate_portfolios()
backtester.report()
