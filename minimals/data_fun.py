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
                     period='45d',
                     interval='4h',
                     auto_adjust=True,
                     progress=False,
                     multi_level_index=False)
    df.columns = df.columns.str.lower()
    return df

def calc_signals(df: pd.DataFrame, ema_fast: int = 5, ema_slow: int = 20, atr_multiplier: int = 1):
    signals = pd.DataFrame()
    signals['close'] = df['close'].copy()
    # ... existing code above ...

    # Compute EMA signals (as you currently do)
    signals['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    signals['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    signals['ema_superslow'] = EMAIndicator(close=df['close'], window=ema_slow * 2).ema_indicator()
    signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
    signals['signal'] = signals['crossover'].diff().fillna(0) / 2
    signals['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False).average_true_range()
    signals.loc[signals['atr'] == 0, 'atr'] = np.nan
    signals['atr_sell'] = df['close'] - signals['atr'] * atr_multiplier
    signals['atr_sell'] = signals['atr_sell'].shift(1)
    signals['atr_sell_signal'] = df['close'] < signals['atr_sell']
    return signals


# end_date = datetime.now(timezone.utc)
# start_date = end_date - timedelta(days=30)

symbol = 'BTC-USD'

ema_fast = 9
ema_slow = 21
atr_multiplier = 1.5
df = get_sample_data(symbol)
signals = calc_signals(df, ema_fast, ema_slow, atr_multiplier)

signals_short = signals.loc[(signals['signal'] == 1) | (signals['signal'] == -1),]

print("\n Data")
print(tabulate(df, headers='keys', tablefmt='github'))
print("\n Signals")
print(tabulate(signals, headers='keys', tablefmt='github'))
print("\n Signal short Table")
print(tabulate(signals_short, headers='keys', tablefmt='github'))

buy_marker_y = df['close'].where(signals['signal'] == 1)
sell_marker_y = df['close'].where(signals['signal'] == -1)
atr_marker = df['close'].where(signals['atr_sell_signal'] == True)
apds = [
    mpf.make_addplot(signals['ema_slow'], color='blue', width=1.0, linestyle='-', label=f'EMA {ema_slow}'),
    mpf.make_addplot(signals['ema_fast'], color='orange', width=1.0, linestyle='-', label=f'EMA {ema_fast}'),
    mpf.make_addplot(signals['ema_superslow'],  width=1.0, linestyle='-', label=f'EMA {ema_slow * 2}'),
    # Use aligned 1D marker series instead of DataFrames
    mpf.make_addplot(buy_marker_y, type='scatter',
                     marker='^', edgecolors='black',markersize=60, color='green', label='Buy Signal', secondary_y=False),
    mpf.make_addplot(sell_marker_y, type='scatter',
                     marker='v', edgecolors='black', markersize=60,color='red', label='Sell Signal', secondary_y=False),
    mpf.make_addplot(signals['atr_sell'], width=1.0, color='black',  linestyle='--', label='ATR'),
    mpf.make_addplot(atr_marker, type='scatter', edgecolors='black',color='red', markersize=80, marker='*', label='ATR Sell Signal', secondary_y=False),]

mpf.plot(df,
         type='candle',
         volume=True,
         style='yahoo',
         addplot=apds,
         figsize=(15, 8),
         title=f"Trading Chart for {symbol} ",
         )
