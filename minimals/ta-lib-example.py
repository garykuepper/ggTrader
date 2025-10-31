import pandas_ta as pta
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
end = start + pd.Timedelta(days=30 * 6)  # one quarter
sar_acceleration = 0.02
sar_maximum = 0.2
atr_multiplier = 1.5
k = KrakenHistoricalData()

df_multi = k.get_ohlcv_df(symbols, interval=interval)

print(df_multi.head(10))
df_multi.columns = df_multi.columns.droplevel(0)
df = df_multi.loc[start:end]

# moving average
ma = pta.ohlc4(df['open'],df['high'],df['low'],df['close'])

# Calculate technical indicators
df['rsi'] = pta.rsi(df['close'])

# adx > 25 strong trend
adx = pta.adx(df['high'], df['low'], df['close'])

print(tabulate(adx.tail(20), headers="keys", tablefmt="github"))
df['adx'] = adx.iloc[:, 1]
df['atr'] = pta.atr(df['high'], df['low'], df['close'])
ce = pta.chandelier_exit(df['high'],
                                            df['low'],
                                            df['close'],
                                            multiplier=atr_multiplier,)

ce['chandelier_long'] = np.where(ce.iloc[:, 2]>0,ce.iloc[:, 0],np.nan)
ce['chandelier_short'] = np.where(ce.iloc[:, 2]<0,ce.iloc[:, 1],np.nan)
print(tabulate(ce.tail(20), headers="keys", tablefmt="github"))


st = pta.supertrend(df['high'],
                    df['low'],
                    df['close'],
                    period=14,
                    multiplier=atr_multiplier)
df['supertrend'] = st.iloc[:, 0]

# SAR Signal
psar = pta.psar(df['high'],
                df['low'],
                close=df['close'], )

print(tabulate(psar.tail(20), headers="keys", tablefmt="github"))
df['sar'] = psar.iloc[:, 0]

# Build addplots for SAR and ATR as before

adx_line = pd.Series(25, index=df.index)

addplots = [
    mpf.make_addplot(psar.iloc[:, 0], type='scatter', color='blue', width=1.0, label='SAR_buy'),
    mpf.make_addplot(psar.iloc[:, 1], type='scatter', color='red', width=1.0, label='SAR_sell'),
    mpf.make_addplot(ce.iloc[:, 3], type='scatter', marker='x', color='teal', width=1.0, label='ATR_buy',
                     secondary_y=False),
    mpf.make_addplot(ce.iloc[:, 4], type='scatter', marker='x', color='orangered', width=1.0, label='ATR_sell',
                     secondary_y=False),
    mpf.make_addplot(df['adx'], type='line', width=1.0, label='ADX',panel=1),
    mpf.make_addplot(adx_line, type='line', color="black",linestyle='--', width=1.0, label='ADX_line',panel=1),
    mpf.make_addplot(ma, type='line', color="black",linestyle='-', width=1.0, label='ma',panel=0),
]

all_addplots = addplots

mpf.plot(
    df,
    type="ohlc",
    volume=False,
    style="yahoo",
    # mav=(20, 50),
    addplot=all_addplots,
    figratio=(18, 9),
    figscale=1.7,
    tight_layout=True,
    title="BTC-USD with SAR and ATR",
    xlim=[end - pd.Timedelta(days=150), end],
)
