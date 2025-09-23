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

from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd
from utils.top_crypto import get_top_cmc
from utils.kraken_ohlcv import fetch_ohlcv_df
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


    for label, row in top_crypto.iterrows():

        symbol = row.get("Symbol")
        pos = top_crypto.index.get_loc(label)

        print(f"Fetching {symbol} OHLC data...{pos+1}/{top_n}")
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


def backtest(signals_dict: dict, plot=False, print_stats=False, cooldown_min=4, position_size=0.05,):
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
                portfolio.update_position_stop_loss(symbol, atr_sell)
                # Check stop loss
                if signal == -1 or atr_sell_signal:
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
                if pos.cost > portfolio.cash:
                    continue
                portfolio.add_position(pos)
                # print(f"{date}: BUY {symbol} at {close_price}, qty: {qty}")
            # # reentry
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
    slow_w = trial.suggest_int("slow_window", fast_w + 10, 100, step=2)
    atr_multi = trial.suggest_float("atr_multiplier", .5, 2.0, step=0.0625)
    cooldown_period = 4

    signals_dict = get_signals(ohlcv,
                               ema_fast=fast_w,
                               ema_slow=slow_w,
                               atr_multiplier=atr_multi)

    stats = backtest(signals_dict, cooldown_min=cooldown_period)

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
    alpha = 2.0
    beta = 0.01  # per dollar of fees
    gamma = 0.001  # per trade

    score = sharpe - alpha * max_dd - beta * fees - gamma * trades

    return float(score)


def save_to_json(study_name: str, out: dict):
    with open(study_name + ".json", "w", encoding="utf-8") as f:

        json.dump(out, f, indent=2, ensure_ascii=False)


# ohlcv = get_top_crypto_ohlcv(top_n=30, limit=720, interval="4h")
# save_ohlcv_dict(ohlcv, "ohlcv_dict_4h_top30.pkl")
ohlcv = load_ohlcv_dict("ohlcv_dict.pkl")
study_name = "top_crypto_sharpe_noreentry_v1"
study = optuna.create_study(direction="maximize",
                            storage="sqlite:///ema_optuna.db",
                            study_name=study_name,
                            load_if_exists=True)

study.optimize(objective, n_trials=10, n_jobs=1)

time.sleep(0.3)
print("Best value:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



print("\nRunning backtest with best params:")
fast_w = study.best_params['fast_window']
slow_w = study.best_params['slow_window']
atr_multi = study.best_params['atr_multiplier']
signals_dict = get_signals(ohlcv, ema_fast=fast_w, ema_slow=slow_w, atr_multiplier=atr_multi)
date_index = next(iter(signals_dict.values()), None).index
delta = (date_index[-1] - date_index[0])
print(f"Date range: {date_index[0]} to {date_index[-1]}. Days: {delta.days}")
stats = backtest(signals_dict, plot=True, print_stats=True)

out = {
    "best_params": study.best_params,
    "best_value": study.best_value,
    "Best Params Stats": stats.T.to_dict()[0]
}

save_to_json(study_name, out)