import talib as ta
import numpy as np
import yfinance as yf
import pandas as pd
import mplfinance as mpf
from tabulate import tabulate
from utils.KrakenHistoricalData import KrakenHistoricalData

c = np.random.randn(100)
symbols = ["BTC"]
interval = "1d"
start = pd.to_datetime("2025-01-01").tz_localize('UTC')
end = start + pd.Timedelta(days=120)  # one quarter
sar_acceleration = 0.02
sar_maximum = 0.2
atr_multiplier = 1.5
k = KrakenHistoricalData()

df_multi = k.get_ohlcv_df(symbols,interval=interval)
df_multi.columns = df_multi.columns.droplevel(0)
df = df_multi.loc[start:end]

# df = yf.download("BTC-USD", start="2025-08-01", end="2025-10-30", interval="1d", multi_level_index=False)

print(df.head())

df['rsi'] = ta.RSI(df['close'])
df['adx'] = ta.ADX(df['high'], df['low'], df['close'])

df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
df['atr_stop_loss'] = df['close'] - (atr_multiplier * df['atr'].shift(1))
df['atr_stop_loss_updated'] = df['atr_stop_loss']  # initialize with original values
prev_stop_loss = np.nan

for index, row in df.iterrows():
    new_stop_loss = df.loc[index, 'atr_stop_loss']

    if pd.isna(new_stop_loss):
        continue

    if pd.isna(prev_stop_loss):
        prev_stop_loss = new_stop_loss

    if new_stop_loss > prev_stop_loss:
        # update if new stop loss is higher
        df.loc[index, 'atr_stop_loss_updated'] = new_stop_loss

    else:
        # keep the previous stop loss
        df.loc[index, 'atr_stop_loss_updated'] = prev_stop_loss

    prev_stop_loss = df.loc[index, 'atr_stop_loss_updated']

    # check if triggered
    if df.loc[index, 'atr_stop_loss_updated'] > df.loc[index, 'close']:
        df.loc[index, 'atr_stop_loss_updated'] = new_stop_loss
        prev_stop_loss = new_stop_loss

# SAR Signal
df['sar'] = ta.SAR(df['high'],
                   df['low'],
                   acceleration=sar_acceleration,
                   maximum=sar_maximum)
cross_up = (df['close'] > df['sar']) & (df['close'].shift(1) <= df['sar'].shift(1))
cross_down = (df['close'] < df['sar']) & (df['close'].shift(1) >= df['sar'].shift(1))

signals = pd.Series(0, index=df.index, dtype=int)
signals[cross_up] = 1
signals[cross_down] = -1
df['signals'] = signals
print(tabulate(df.head(40), headers="keys", tablefmt="github"))
# Create signal marker series that only contain markers at the signal points
up_markers = df['close'].where(df['signals'] == 1)
down_markers = df['close'].where(df['signals'] == -1)

# Build addplots for SAR and ATR as before
addplots = [
    mpf.make_addplot(df['sar'], type='scatter', color='orange', width=1.0, label='SAR'),
    mpf.make_addplot(df['atr_stop_loss_updated'], color='blue', width=1.0, label='ATR_updated'),
    # Scatter addplots for signal markers (only plot where not NaN)
    mpf.make_addplot(up_markers, type='scatter', markersize=100, marker='^', color='green', secondary_y=False),
    mpf.make_addplot(down_markers, type='scatter', markersize=100, marker='v', color='red', secondary_y=False),
]

all_addplots = addplots

mpf.plot(
    df,
    type="candle",
    volume=False,
    # mav=(20, 50),
    addplot=all_addplots,
    figsize=(19, 14),
    title="BTC-USD with SAR and ATR",
    style="yahoo"
)
