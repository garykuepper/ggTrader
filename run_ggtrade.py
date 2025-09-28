import os
import time
import json
import numpy as np
import optuna
import ccxt
import pandas as pd
import pickle
from tabulate import tabulate
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd, get_top_kraken_by_volume
from utils.top_crypto import get_top_cmc, get_kraken_top_crypto
from utils.kraken_ohlcv import fetch_ohlcv_df
from ggTrader.Portfolio import Portfolio
from ggTrader.Position import Position
from ggTrader.Signals import EMASignals


def get_top_crypto_ohlcv(top_n=20, limit=30, interval="4h"):
    top_crypto = get_top_cmc(limit=top_n + 5, print_table=False)
    # top_crypto = get_top_kraken_by_volume(top_n=top_n)
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
    # datetime_index = pd.date_range(start="2023-01-01", end="2025-06-30", freq="4h")

    for label, row in top_crypto.iterrows():

        symbol = row.get("Symbol")
        pos = top_crypto.index.get_loc(label)

        print(f"Fetching {symbol} OHLC data...{pos + 1}/{top_n}")
        try:
            df[symbol] = fetch_ohlcv_df(kraken, symbol + '/USD', timeframe=interval, limit=limit)
            df[symbol] = df[symbol].reindex(datetime_index)
        except Exception as e:
            print(f"Error fetching {symbol} OHLC data: {e}")

    return df


def get_ohlcv_csv(path: str, tickers: list = None):
    with os.scandir(path) as it:
        files = [entry.name for entry in it if entry.is_file()]
    num_files = len(files)

    ohlcv_dict = {}
    datetime_index = pd.date_range(start="2024-01-01",
                                   end="2025-06-30",
                                   freq="4h",
                                   tz="UTC")
    for f in files:
        ticker = f.split("_")[0]
        if tickers is not None and ticker not in tickers:
            continue
        print(f"({files.index(f) + 1}/{num_files}) Processing {f} ")
        file_path = os.path.join(path, f)
        df = pd.read_csv(file_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
        else:
            # If there is no 'date' column, try to infer from index or adjust accordingly
            df = df.set_index(pd.to_datetime(df.index, utc=True))
        # print(df.head())
        ohlcv_dict[ticker] = df.reindex(datetime_index)
        # print(ohlcv_dict[ticker].head())

    return ohlcv_dict


def get_signals(ohlcv: dict, ema_fast: int = 5, ema_slow: int = 20, atr_multiplier: float = 1.0):
    signals_dict = {}
    signals = EMASignals(ema_fast, ema_slow, atr_multiplier)
    for key in ohlcv.keys():
        signals_dict[key] = signals.compute(ohlcv[key])

    return signals_dict


def save_ohlcv_dict(ohlcv, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(ohlcv, f)


def load_ohlcv_dict(file_path):
    with open(file_path, "rb") as f:
        ohlcv = pickle.load(f)
    return ohlcv


def position_sizing(portfolio: Portfolio, symbol: str, close_price: float, date, position_size: float = 0.05):
    qty = portfolio.total_value / close_price * position_size
    return Position(symbol, qty, close_price, date)


def backtest(signals_dict: dict, plot=False, print_stats=False, cooldown_min=4, position_size=0.05, ):
    # print("\n Backtest")
    date_index = next(iter(signals_dict.values()), None).index
    portfolio = Portfolio(cash=10000)
    reentry = dict.fromkeys(signals_dict.keys(), 0)

    for date in date_index:
        for symbol in signals_dict.keys():
            signal = signals_dict[symbol].loc[date, 'signal']
            crossover = signals_dict[symbol].loc[date, 'crossover']
            atr_sell_signal = signals_dict[symbol].loc[date, 'atr_sell_signal']
            atr_sell = signals_dict[symbol].loc[date, 'atr_sell']
            close_price = signals_dict[symbol].loc[date, 'close']
            if portfolio.in_position(symbol):
                portfolio.update_position_price(symbol, close_price, date)
                # Update and check stop loss
                stop_loss_exit = portfolio.check_stop_loss(symbol, atr_sell)
                if signal == -1 or atr_sell_signal or stop_loss_exit:
                    pos = portfolio.get_position(symbol)
                    if atr_sell_signal:
                        pos.stop_loss_triggered = True
                    # Close position
                    reentry[symbol] = 0  # reset reentry counter
                    portfolio.close_position(pos, date)
                    # print(f"{date}: SELL {symbol} at {close_price}")
            elif signal == 1:
                # position sizing
                pos = position_sizing(portfolio, symbol, close_price, date, position_size)
                pos.stop_loss = atr_sell
                if pos.cost > portfolio.cash:
                    continue
                portfolio.add_position(pos)
                # print(f"{date}: BUY {symbol} at {close_price}, qty: {qty}")
            # reentry
            # elif crossover == 1:
            #     reentry[symbol] += 1
            #     if reentry[symbol] > cooldown_min:
            #         pos = position_sizing(portfolio, symbol, close_price, date, position_size)
            #         if pos.cost > portfolio.cash:
            #             # skip if cost is greater than cash available
            #             continue
            #         portfolio.add_position(pos)
        portfolio.record_equity(date)

    # portfolio.print_trades()
    # portfolio.print_positions()
    # portfolio.print_profit_per_symbol()
    # portfolio.print_stats()
    # portfolio.reconcile()
    # portfolio.print_stats_df()
    if print_stats:
        portfolio.print_stats_df()
    if plot:
        portfolio.plot_equity_curve()

    return portfolio.get_stats_df()


def objective(trial):
    fast_w = trial.suggest_int("fast_window", 8, 50, step=2)
    slow_w = trial.suggest_int("slow_window", fast_w + 10, 80, step=2)
    atr_multi = trial.suggest_float("atr_multiplier", .5, 2.5, step=0.0625)

    signals_dict = get_signals(ohlcv,
                               ema_fast=fast_w,
                               ema_slow=slow_w,
                               atr_multiplier=atr_multi)

    stats = backtest(signals_dict)

    # extract metrics
    sharpe = float(stats['sharpe'].values[0])
    max_dd = float(stats.get('max_drawdown', [0.0])[0]) if 'max_drawdown' in stats else 0.0
    fees = float(stats.get('transaction_fee_total', [0.0])[0]) if 'transaction_fee_total' in stats else 0.0
    trades = int(stats.get('total_trades', [0])[0])

    # basic constraints: require some minimum trades to avoid tiny-sample winners
    MIN_TRADES = 20
    if trades < MIN_TRADES:
        return -1e6  # very poor

    # score = sharpe minus penalties:
    # alpha: penalty per unit drawdown (fraction), beta: penalty per $100 fees, gamma: penalty for high turnover
    alpha = 5.0
    beta = 0.002  # per dollar of fees
    gamma = 0.002  # per trade
    max_dd_wt = alpha * max_dd
    fees_wt = beta * fees
    trades_wt = gamma * trades
    # print(f'sharpe: {sharpe:.2f} max_dd: {max_dd_wt:.2f} fees: ${fees_wt:.2f} trades: {trades_wt}')
    score = sharpe - max_dd_wt - fees_wt - trades_wt

    return float(score)


def save_to_json(study_name: str, out: dict):
    with open(study_name + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


top_crypto_list = get_kraken_top_crypto(top_n=20)

interval = "4h"
top_n = 20
# filename = f"ohlcv_dict_{top_n}_{date_str}.pkl"
file_path = "data/kraken_dict/kraken_historial_4h.pkl"
path = "data/kraken_hist_4h_latest"

latest_4h = pd.Timestamp.utcnow().floor(interval)
date_str = latest_4h.strftime("%Y-%m-%d_%H")

# ohlcv = get_top_crypto_ohlcv(top_n=top_n, limit=720, interval=interval)
# ohlcv = get_ohlcv_csv(path, top_crypto_list["Symbol"].tolist())

# save_ohlcv_dict(ohlcv,file_path)
ohlcv = load_ohlcv_dict(file_path)

study = True

study_name = f"renetry"
if study:

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///ema_optuna.db")

    study.optimize(objective, n_trials=100, n_jobs=1)

    time.sleep(0.3)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print("\nRunning backtest with best params:")
    fast_w = study.best_params['fast_window']
    slow_w = study.best_params['slow_window']
    atr_multi = study.best_params['atr_multiplier']

else:
    fast_w = 48
    slow_w = 98
    atr_multi = 1.5

signals_dict = get_signals(ohlcv, ema_fast=fast_w, ema_slow=slow_w, atr_multiplier=atr_multi)
date_index = next(iter(signals_dict.values()), None).index
delta = (date_index[-1] - date_index[0])
print(f"Date range: {date_index[0]} to {date_index[-1]}. Days: {delta.days}")
stats = backtest(signals_dict, plot=True, print_stats=True)


# print(tabulate(signals_dict['BTC'].tail(), headers="keys", tablefmt="github"))
# print(tabulate(ohlcv['BTC'].tail(), headers="keys", tablefmt="github"))
if study:
    out = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "Best Params Stats": stats.T.to_dict()[0]
    }

    save_to_json(study_name, out)
