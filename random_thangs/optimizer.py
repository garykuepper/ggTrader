import optuna
from data_manager import CryptoDataManager
from datetime import datetime, timedelta
from ema_trailing_strategy import EmaTrailingStrategy
from optuna.samplers import RandomSampler
# Load data once
symbol = 'XRPUSDT'
interval = '4h'
end_date = datetime(2025, 8, 5)
start_date = end_date - timedelta(days=30)
df = CryptoDataManager().get_crypto_data(symbol, interval, start_date, end_date)

starting_cash = 10000

def objective(trial):
    fast_window = trial.suggest_int('fast_window', 5, 20)
    slow_window = trial.suggest_int('slow_window', fast_window + 5, 35)
    trailing_pct = trial.suggest_float('trailing_pct', 0.005, 0.10)
    min_hold_bars = trial.suggest_int('min_hold_bars', 3, 12)

    strat = EmaTrailingStrategy(df, fast_window, slow_window, trailing_pct, min_hold_bars, starting_cash)
    trades = strat.run()

    if trades.empty:
        return -1e9
    return trades['profit'].sum()

study = optuna.create_study(direction='maximize', sampler=RandomSampler())
study.optimize(objective, n_trials=1000, n_jobs=-1)

print("Best parameters:")
for key, value in study.best_params.items():
    if isinstance(value, int):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.4f}")
print(f"Best total profit: {study.best_value:.2f}")
