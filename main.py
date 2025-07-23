
from swing_trader.backtest.backtester import Backtester
from swing_trader.data.mr_data import MrData
from datetime import datetime, timedelta

mr_data = MrData()

# TODO: Use datetime objects for start and end dates

start_date = datetime(2015, 1, 1)
end_date = datetime.today() - timedelta(days=1)

for ticker in ["SSO", "SH","SPY"]:
    mr_data.get_stock_data(ticker, start_date, end_date)


# TODO: backtester should download data based on strat_swing_target
bt = Backtester(
    db=mr_data.db,
    signal_ticker="SPY",
    signal_field="strat_swing_signal",
    target_field="strat_swing_target",
    initial_cash=10000
)

results = bt.run(start_date, end_date, benchmark_ticker="SPY")
bt.plot_equity_curve(results)

