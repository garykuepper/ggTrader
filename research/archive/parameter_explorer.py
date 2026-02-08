import numpy as np
import pandas as pd
from utils.KrakenHistoricalData import KrakenHistoricalData
from ggTrader.Backtest import Backtest
from tabulate import tabulate
from ggTrader.Signals import Signals
import optuna
from optuna.samplers import RandomSampler

k = KrakenHistoricalData()

symbols = ["BTC"]
interval = "4h"
end = pd.to_datetime("2025-09-30").tz_localize('UTC')
start = end - pd.Timedelta(days=30 * 6)
# start = pd.to_datetime("2023-01-01").tz_localize('UTC')
# Indicator Params
sar_acceleration = 0.02
sar_maximum = 0.2
atr_multiplier = 3  # atr multiplier for chandelier exit
adx_threshold = 25  # trend strength
ce_high_length = 22  # look back length

# Fees
maker_fee = 0.0025
taker_fee = 0.004
transaction_fee = (maker_fee + taker_fee) / 2

# k.use_remote("https://garygigabytes.com/kraken/parquet")  # no trailing slash
# df_multi = k.get_ohlcv_df_remote(symbols, interval=interval)
df_multi = k.get_ohlcv_df(symbols, interval=interval)

#
df_multi.columns = df_multi.columns.droplevel(0)
df = df_multi.loc[start:end]


def objective(trial: optuna.Trial):
    atr_multiplier = trial.suggest_int("atr_multiplier", 1, 5)
    adx_threshold = trial.suggest_int("adx_threshold", 20, 30)
    ce_high_length = trial.suggest_int("ce_high_length", 10, 30)

    s = Signals(adx_threshold=adx_threshold,
                atr_multiplier=atr_multiplier,
                ce_high_length=ce_high_length)

    signals = s.calc_signals(df.copy())
    bt = Backtest(signals, interval=interval, transaction_fee=transaction_fee)
    stats, profit_df = bt.run()

    sortino = stats['sortino']
    return sortino

sampler = RandomSampler(seed=42)  # optional seed for reproducibility

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)
