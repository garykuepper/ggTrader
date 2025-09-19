import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs
from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd
from utils.top_crypto import get_top_cmc
from tabulate import tabulate
from utils.kraken_ohlcv import fetch_ohlcv_df
import ccxt
import pandas as pd
from ggTrader.Portfolio import Portfolio
from ggTrader.Position import Position


def get_top_crypto_ohlcv(top_n=20, limit=30, interval="4h"):
    top_crypto = get_top_cmc(limit=top_n + 5, print_table=False)
    # filter out non-Kraken pairs
    kraken_usd_pairs = pd.DataFrame(get_kraken_asset_pairs_usd())
    top_crypto = top_crypto[top_crypto["Symbol"].isin(kraken_usd_pairs["base_common"])]
    top_crypto = top_crypto.reset_index(drop=True)
    # Slice to top N
    top_crypto = top_crypto.head(top_n)
    print(tabulate(top_crypto, headers="keys", tablefmt="github"))
    kraken = ccxt.kraken()
    kraken.load_markets()
    df = {}
    # time index
    # If you want to keep UTC with tz-aware:
    latest_4h = pd.Timestamp.utcnow().floor(interval)
    datetime_index = pd.date_range(end=latest_4h, periods=limit, freq=interval)

    for _, row in top_crypto.iterrows():

        symbol = row.get("Symbol")
        print(f"Fetching {symbol} OHLC data...")
        try:
            df[symbol] = fetch_ohlcv_df(kraken, symbol + '/USD', timeframe=interval, limit=limit)
            df[symbol] = df[symbol].reindex(datetime_index)
        except Exception as e:
            print(f"Error fetching {symbol} OHLC data: {e}")
    return df


def calc_signals(df: pd.DataFrame, ema_fast: int = 5, ema_slow: int = 20, atr_multiplier: float = 1.0):
    signals = pd.DataFrame()
    signals['close'] = df['close'].copy()

    # Compute EMA signals (as you currently do)
    signals['ema_fast'] = EMAIndicator(close=df['close'],
                                       window=ema_fast).ema_indicator()
    signals['ema_slow'] = EMAIndicator(close=df['close'],
                                       window=ema_slow).ema_indicator()
    signals['ema_superslow'] = EMAIndicator(close=df['close'],
                                            window=ema_slow * 2).ema_indicator()
    signals['crossover'] = np.sign(signals['ema_fast'] - signals['ema_slow'])
    signals['signal'] = signals['crossover'].diff().fillna(0) / 2
    signals['atr'] = AverageTrueRange(high=df['high'],
                                      low=df['low'],
                                      close=df['close'],
                                      window=14,
                                      fillna=False).average_true_range()
    signals.loc[signals['atr'] == 0, 'atr'] = np.nan
    signals['atr_sell'] = df['close'] - signals['atr'] * atr_multiplier
    signals['atr_sell'] = signals['atr_sell'].shift(1)
    signals['atr_sell_signal'] = df['close'] < signals['atr_sell']
    return signals


def get_signals(ohlcv: dict, ema_fast: int = 5, ema_slow: int = 20, atr_multiplier: float = 1.0):
    signals_dict = {}
    for key in ohlcv.keys():
        signals = calc_signals(ohlcv[key], ema_fast, ema_slow, atr_multiplier)
        signals_dict[key] = signals

    return signals_dict


ohlcv = get_top_crypto_ohlcv(top_n=20, limit=700)

# first_ohlcv = next(iter(ohlcv.values()), None)
date_index = next(iter(ohlcv.values()), None).index
# first_ohlcv_symbol = next(iter(ohlcv.keys()), None)
# print(f"\n First OHLCV data: {first_ohlcv_symbol}")
# print(tabulate(first_ohlcv, headers="keys", tablefmt="github"))

signals_dict = get_signals(ohlcv, ema_fast=5, ema_slow=20, atr_multiplier=1.0)

first_signals = next(iter(signals_dict.values()), None)
first_signals_symbol = next(iter(signals_dict.keys()), None)
print(f"\n First Signals: {first_signals_symbol}")
print(tabulate(first_signals, headers="keys", tablefmt="github"))

print("\n Signals")

portfolio = Portfolio(cash=10000)
for date in date_index:
    for symbol in signals_dict.keys():
        signal = signals_dict[symbol].loc[date, 'signal']
        close_price = signals_dict[symbol].loc[date, 'close']
        if portfolio.in_position(symbol):
            portfolio.update_position_price(symbol, close_price, date)
            if signal == -1:
                pos = portfolio.get_position(symbol)
                portfolio.close_position(pos, date)
                print(f"{date}: SELL {symbol} at {close_price}")
        elif signal == 1:
            # position sizing
            total_value = portfolio.total_value
            qty = total_value / close_price * 0.05
            cost = qty * close_price
            if cost > portfolio.cash:
                continue
            portfolio.add_position(Position(symbol, qty, close_price, date))
            print(f"{date}: BUY {symbol} at {close_price}, qty: {qty}")

portfolio.print_positions()

portfolio.print_profit_per_symbol()

portfolio.print_stats()
