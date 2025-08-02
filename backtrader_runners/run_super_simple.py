import backtrader as bt
from old.ggTrader_old.strats.ema_macd_rsi import EmaMacdRsiStrategy
from old.ggTrader_old.data_manager import UniversalDataManager
import os
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
dm = UniversalDataManager(mongo_uri=mongo_uri)
df = dm.load_or_fetch("SPY", "1d", "2020-01-01", "2023-01-01", market="stock")

cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df)

cerebro.adddata(data)
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.001)

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addstrategy(EmaMacdRsiStrategy)

results = cerebro.run()
strat = results[0]

# Extract metrics
trades = strat.analyzers.trades.get_analysis()
sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
final_value = cerebro.broker.getvalue()
pnl = final_value - 10000

# Build table
table = [
    ["Start Cash", 10000],
    ["End Value", round(final_value, 2)],
    ["Net PnL", round(pnl, 2)],
    ["Total Trades", trades.total.closed if 'closed' in trades.total else 0],
    ["Winning Trades", trades.won.total if 'won' in trades else 0],
    ["Losing Trades", trades.lost.total if 'lost' in trades else 0],
    ["Win Rate (%)", round(100 * trades.won.total / trades.total.closed, 2) if trades.total.closed else 0],
    ["Sharpe Ratio", round(sharpe, 4) if sharpe else "N/A"],
]

print("\n📊 Backtest Performance Summary:")
print(tabulate(table, headers=["Metric", "Value"], tablefmt="github"))

# Plot
cerebro.plot()
