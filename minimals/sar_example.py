# psar_adx_4h_ta_backtest_mpf.py
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from ta.trend import ADXIndicator, PSARIndicator
from datetime import datetime, timedelta, timezone

# ----------------------------
# Config
# ----------------------------
TICKER = "BTC-USD"
INTERVAL = "4h"        # yfinance intervals: 1h, 2h, 4h, 1d, etc.
LOOKBACK_DAYS = 365   # ~1 year
ADX_PERIOD = 14
ADX_THRESHOLD = 45.0
PSAR_STEP = 0.02
PSAR_MAX = 0.20
FEE_BP = 10            # fees in basis points per round trip

# ----------------------------
# Download data
# ----------------------------
# Option A (reproducible):
end = datetime(year=2025, month=6, day=30, tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
# Option B (dynamic, aligned to hour):
# end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=LOOKBACK_DAYS)

df = yf.download(
    TICKER,
    interval=INTERVAL,
    start=start,
    end=end,
    auto_adjust=False,
    progress=False,
    multi_level_index=False
)

# Keep OHLCV only and ensure proper columns/index
df = df[['Open','High','Low','Close','Volume']].dropna().copy()

# ----------------------------
# Indicators (ta)
# ----------------------------
adx_ind = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=ADX_PERIOD)
df['ADX'] = adx_ind.adx()

psar = PSARIndicator(
    high=df['High'], low=df['Low'], close=df['Close'],
    step=PSAR_STEP, max_step=PSAR_MAX
)
df['PSAR'] = psar.psar()

# ----------------------------
# Trading rules
#   Enter long when Close > PSAR and ADX >= threshold
#   Exit when Close < PSAR
# ----------------------------
in_pos = False
trades = []

for i in range(1, len(df)):
    c, ps, adxv = df['Close'].iloc[i], df['PSAR'].iloc[i], df['ADX'].iloc[i]

    if not in_pos:
        if pd.notna(ps) and pd.notna(adxv) and (c > ps) and (adxv >= ADX_THRESHOLD):
            in_pos = True
            trades.append({'time': df.index[i], 'type': 'BUY', 'price': c})
    else:
        if pd.notna(ps) and (c < ps):
            in_pos = False
            trades.append({'time': df.index[i], 'type': 'SELL', 'price': c})

# Close any open position at the last bar
if in_pos:
    trades.append({'time': df.index[-1], 'type': 'SELL', 'price': df['Close'].iloc[-1]})

# ----------------------------
# PnL + metrics
# ----------------------------
pnl = []
for j in range(0, len(trades)-1, 2):
    buy, sell = trades[j], trades[j+1]
    if buy['type'] == 'BUY' and sell['type'] == 'SELL':
        gross_ret = (sell['price'] / buy['price']) - 1.0
        fee_ret = - (FEE_BP / 10000.0)
        pnl.append(gross_ret + fee_ret)

pnl = pd.Series(pnl, dtype=float)
wins, losses = int((pnl > 0).sum()), int((pnl <= 0).sum())
n_trades = wins + losses
win_rate = wins / n_trades if n_trades else 0.0
gross_profit, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

equity = (1.0 + pnl).cumprod()
total_return = equity.iloc[-1] - 1.0 if len(equity) else 0.0
roll_max = equity.cummax() if len(equity) else pd.Series(dtype=float)
max_dd = ((equity / roll_max) - 1.0).min() if len(equity) else 0.0

print(f"=== {TICKER} {INTERVAL} | SAR (step={PSAR_STEP}, max={PSAR_MAX}) + ADX({ADX_PERIOD})>={ADX_THRESHOLD} ===")
print(f"Trades: {n_trades} | Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1%}")
print(f"Total return (trade-to-trade): {total_return:.2%}")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Max drawdown (trade-to-trade): {max_dd:.2%}")
print(f"Fees assumed: {FEE_BP} bps per round trip\n")

# ----------------------------
# mplfinance plots
#  - Main panel: Candles + PSAR dots + BUY/SELL arrows + Volume
#  - Panel 1: ADX line + threshold
# ----------------------------

# PSAR dots split by side (below/above price)
psar_bull = df['PSAR'].where(df['Close'] > df['PSAR'])
psar_bear = df['PSAR'].where(df['Close'] < df['PSAR'])

# Trade markers (place buys slightly below the low, sells slightly above the high so they don't cover candles)
buy_series = pd.Series(np.nan, index=df.index)
sell_series = pd.Series(np.nan, index=df.index)
for t in trades:
    if t['type'] == 'BUY':
        ts = t['time']
        if ts in df.index:
            buy_series.loc[ts] = df.loc[ts, 'Low'] * 0.995
    elif t['type'] == 'SELL':
        ts = t['time']
        if ts in df.index:
            sell_series.loc[ts] = df.loc[ts, 'High'] * 1.005

# ADX and threshold (panel 1)
adx_series = df['ADX']
adx_thresh_series = pd.Series(ADX_THRESHOLD, index=df.index)

add_plots = [
    # PSAR
    mpf.make_addplot(psar_bull, type='scatter', markersize=20, marker='.', panel=0),
    mpf.make_addplot(psar_bear, type='scatter', markersize=20, marker='.', panel=0),
    # Trades
    mpf.make_addplot(buy_series, type='scatter', markersize=60, marker='^', panel=0),
    mpf.make_addplot(sell_series, type='scatter', markersize=60, marker='v', panel=0),
    # ADX panel
    mpf.make_addplot(adx_series, panel=1, ylabel='ADX'),
    mpf.make_addplot(adx_thresh_series, panel=1, linestyle='--'),
]

mpf.plot(
    df,
    type='line',
    addplot=add_plots,
    volume=True,
    style='yahoo',
    title=f"{TICKER} — {INTERVAL} Candles | PSAR + Trades (panel 0), ADX (panel 1)",
    figratio=(16,9),
    figscale=1.2,
    tight_layout=True,
    mav=(20,50, 200)
)
