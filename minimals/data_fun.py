import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


def get_sample_data(ticker: str):
    df = yf.download(ticker,
                     period='14d',
                     interval='4h',
                     auto_adjust=True,
                     progress=False,
                     multi_level_index=False)
    df.columns = [col.lower() for col in df.columns]
    return df

def calc_signals(df: pd.DataFrame, ema_fast: int = 5, ema_slow: int = 20):
    signals = pd.DataFrame()
    signals['close'] = df['close'].copy()
    # ... existing code above ...

    # Compute EMA signals (as you currently do)
    signals['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    signals['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
    signals['signal'] = signals['crossover'].diff().fillna(0) / 2
    signals['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False).average_true_range()
    signals.loc[signals['atr'] == 0, 'atr'] = np.nan
    return signals


# end_date = datetime.now(timezone.utc)
# start_date = end_date - timedelta(days=30)

symbol = 'LTC-USD'

ema_fast = 5
ema_slow = 15
df = get_sample_data(symbol)
signals = calc_signals(df, ema_fast, ema_slow)

signals_short = signals.loc[(signals['signal'] == 1) | (signals['signal'] == -1),]

print("\n Data")
print(tabulate(df, headers='keys', tablefmt='github'))
print("\n Signals")
print(tabulate(signals, headers='keys', tablefmt='github'))
print("\n Signal short Table")
print(tabulate(signals_short, headers='keys', tablefmt='github'))

buy_marker_y = df['close'].where(signals['signal'] == 1)
sell_marker_y = df['close'].where(signals['signal'] == -1)

apds = [
    mpf.make_addplot(signals['ema_slow'], color='blue', width=1.0, linestyle='-', label=f'EMA {ema_slow}'),
    mpf.make_addplot(signals['ema_fast'], color='orange', width=1.0, linestyle='-', label=f'EMA {ema_fast}'),
    # Use aligned 1D marker series instead of DataFrames
    mpf.make_addplot(buy_marker_y, type='scatter',
                     marker='^', edgecolors='black', color='green', label='Buy Signal', secondary_y=False),
    mpf.make_addplot(sell_marker_y, type='scatter',
                     marker='v', edgecolors='black', color='red', label='Sell Signal', secondary_y=False)
]

mpf.plot(df,
         type='candle',
         volume=True,
         style='yahoo',
         addplot=apds,
         figsize=(13, 7),
         title=f"Trading Chart for {symbol} ",
         )
