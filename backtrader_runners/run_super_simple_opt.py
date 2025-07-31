import backtrader as bt
import pandas as pd
from ggTrader.strats.ema_macd import EMAMACDStrategy  # Your updated strategy class
from ggTrader.data_manager.universal_data_manager import UniversalDataManager
import os
from dotenv import load_dotenv
from tabulate import tabulate

# === Setup ===
load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
dm = UniversalDataManager(mongo_uri=mongo_uri)
df = dm.load_or_fetch("SPY", "1d", "2023-01-01", "2025-01-01", market="stock")

data = bt.feeds.PandasData(dataname=df)

# === Optimization Setup ===
cerebro = bt.Cerebro(maxcpus=1)
cerebro.adddata(data)
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.001)

# Run parameter optimization
cerebro.optstrategy(
    EMAMACDStrategy,
    ema_fast=range(5, 15, 5),
    ema_slow=range(20, 40, 10),
    macd_fast=[12],
    macd_slow=[26],
    macd_signal=range(8, 13, 2),
    stop_loss_pct=[0.03, 0.05, 0.07],
    position_pct=[0.95],
    log_enabled=[False]  # off during optimization
)

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

# === Run Optimization ===
results = cerebro.run(optreturn=False)

# === Collect and Display Results ===
best_result = None
best_score = float('-inf')
rows = []

for strat_list in results:
    strat = strat_list[0]
    p = strat.params
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    final_val = strat.broker.getvalue()
    pnl = final_val - 10000

    closed = trades.get('total', {}).get('closed', 0)
    won = trades.get('won', {}).get('total', 0)
    lost = trades.get('lost', {}).get('total', 0)

    if closed:
        win_rate = round(100 * won / closed, 2)
        score = sharpe if sharpe is not None else -999

        row = [
            p.ema_fast,
            p.ema_slow,
            p.macd_signal,
            p.stop_loss_pct,
            round(final_val, 2),
            round(pnl, 2),
            closed,
            won,
            lost,
            win_rate,
            round(sharpe, 4) if sharpe else "N/A"
        ]
        rows.append(row)

        if score > best_score:
            best_score = score
            best_result = row

# === Print Results ===
headers = [
    "EMA Fast", "EMA Slow", "MACD Signal", "Stop Loss %",
    "End Value", "Net PnL", "Trades", "Wins", "Losses",
    "Win Rate (%)", "Sharpe"
]

print("\n📊 Optimization Results:")
print(tabulate(rows, headers=headers, tablefmt="github"))

if best_result:
    print("\n🏆 Best Parameters:")
    print(tabulate([best_result], headers=headers, tablefmt="fancy_grid"))
