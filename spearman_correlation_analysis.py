# Python 3.x
# Full script: fetch BTC-USD 4h data from yfinance, compute TA-based signals directly
# using the ta library (EMA, MACD, SMA, ADX, PSAR, ATR), and evaluate their
# monotonic relationship to the next-interval move using Spearman correlation.
# Also includes a seaborn visualization of the results.

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# TA indicators
from ta.trend import EMAIndicator, MACD, SMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import AverageTrueRange

# For Spearman correlation
from scipy.stats import spearmanr

# Import indicator computation from your project (adjust import path as needed)
# Assuming Signals.py is accessible in PYTHONPATH or the same project dir
from ggTrader.Signals import Signals

# ----------------------------
# Config
# ----------------------------
TICKER = "BTC-USD"
INTERVAL = "4h"          # yfinance intervals: 1h, 2h, 4h, 1d, ...
LOOKBACK_DAYS = 365
EMA_FAST = 9
EMA_SLOW = 21
SMA_FAST = 10
ADA = 14  # ADX window
ATR_WINDOW = 14

# ----------------------------
# Data download
# ----------------------------
end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=LOOKBACK_DAYS)

df = yf.download(
    TICKER,
    interval=INTERVAL,
    start=start,
    end=end,
    auto_adjust=True,
    progress=False,
    multi_level_index=False
)

# Validate and trim
df = df[['Open','High','Low','Close','Volume']].dropna().copy()
df.columns = df.columns.str.lower()

# ----------------------------
# Indicators (ta) - compute signals directly
# ----------------------------
signals = pd.DataFrame(index=df.index)

# EMA signals
signals['ema_fast'] = EMAIndicator(close=df['close'], window=EMA_FAST, fillna=False).ema_indicator()
signals['ema_slow'] = EMAIndicator(close=df['close'], window=EMA_SLOW, fillna=False).ema_indicator()

# Simple SMA as an additional indicator
signals['sma_fast'] = SMAIndicator(close=df['close'], window=SMA_FAST, fillna=False).sma_indicator()

# MACD components
macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
signals['macd'] = macd.macd()
signals['macd_signal'] = macd.macd_signal()
signals['macd_diff'] = macd.macd_diff()

# Crossover-based signal (simple directional)
signals['ema_cross'] = (signals['ema_fast'] - signals['ema_slow']).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
signals['ema_cross_change'] = signals['ema_cross'].diff().fillna(0)

# ATR and ATR-based levels
atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_WINDOW, fillna=False)
signals['atr'] = atr.average_true_range()
signals['atr_sell'] = df['close'] - signals['atr']
signals['atr_sell'] = signals['atr_sell'].shift(1)
signals['atr_sell_signal'] = df['close'] < signals['atr_sell']

# PSAR
signals['psar'] = PSARIndicator(high=df['high'], low=df['low'], close=df['close'],
                               step=0.02, max_step=0.20).psar()

# ADX
signals['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'],
                             window=ADA, fillna=False).adx()

# Align with main df index
signals = signals.reindex(df.index)

# Target: next-interval move (1 for Up, 0 for Down)
df['NextUp'] = (df['close'].shift(-1) > df['close']).astype(int)

# Combine into a working frame (drop rows with NaNs in any predictor or target)
indicator_cols = [
    'ema_fast', 'ema_slow', 'sma_fast',
    'macd', 'macd_signal', 'macd_diff',
    'atr', 'atr_sell', 'atr_sell_signal',
    'psar', 'adx', 'ema_cross', 'ema_cross_change'
]
data = df.join(signals[indicator_cols])
data = data.dropna(subset=indicator_cols + ['NextUp'])

# ----------------------------
# Spearman correlation: Indicator -> NextUp
# ----------------------------
rho_results = []
for col in indicator_cols:
    if col not in data.columns:
        continue
    x = data[col]
    y = data['NextUp']
    if x.isna().any() or y.isna().any():
        continue
    rho, pval = spearmanr(x, y)
    rho_results.append({'Indicator': col, 'Spearman_rho': float(rho), 'p_value': float(pval)})

rho_df = pd.DataFrame(rho_results).sort_values('Spearman_rho', ascending=False)

# ----------------------------
# Visualization with seaborn
# ----------------------------
sns.set(style="whitegrid", context="talk", font_scale=1.0)

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=rho_df, x='Indicator', y='Spearman_rho', palette='viridis')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Spearman rho: TA Indicators vs NextInterval Up (BTC-USD, 4h)')
plt.xlabel('TA Indicator')
plt.ylabel('Spearman rho')

# annotate p-values
for i, row in rho_df.iterrows():
    rho = row['Spearman_rho']
    pval = row['p_value']
    ax.text(i, rho, f" p={pval:.3g}", ha='center', va='bottom' if rho >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.show()

# Optional: quick cross-tab view for intuition
def cross_tab_for(col_name: str):
    if col_name not in data.columns:
        return
    pred_up = data[col_name] > 0
    tbl = pd.crosstab(pred_up, data['NextUp'], rownames=[col_name], colnames=['NextUp'])
    print(f"\nCross-tab for {col_name}:\n{tbl}\n")

for c in indicator_cols:
    cross_tab_for(c)

# End of script
