import pandas as pd
from optuna.samplers import RandomSampler
from ta.trend import EMAIndicator
import optuna
import time

# Load and preprocess data once
df = pd.read_csv("yf_ltc_5y.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

starting_cash = 10000

def objective(trial):
    # Suggest EMA window parameters
    fast_window = trial.suggest_int('fast_window', 6, 20)
    slow_window = trial.suggest_int('slow_window', fast_window + 1, 35)  # slow > fast

    # Calculate EMAs with trial parameters
    df['ema_fast'] = EMAIndicator(df['close'], window=fast_window).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=slow_window).ema_indicator()

    # Generate signals
    ema_fast_above = df['ema_fast'] > df['ema_slow']

    # Fix for FutureWarning: split chained calls
    shifted = ema_fast_above.shift(1)
    filled = shifted.fillna(False)
    bool_series = filled.astype(bool)

    cross_up = (ema_fast_above) & (~bool_series)
    cross_down = (~ema_fast_above) & (bool_series)

    # Prepare buy and sell signals
    buy_signals = df.loc[cross_up, ['close']].reset_index().rename(columns={'date':'buy_date', 'close':'buy_price'})
    sell_signals = df.loc[cross_down, ['close']].reset_index().rename(columns={'date':'sell_date', 'close':'sell_price'})

    # Merge buy with next sell after buy_date
    trades = pd.merge_asof(
        buy_signals.sort_values('buy_date'),
        sell_signals.sort_values('sell_date'),
        left_on='buy_date', right_on='sell_date',
        direction='forward'
    )
    trades = trades.dropna()

    # If no trades, return very low profit
    if trades.empty:
        return -1e9

    # Simulate trading
    cash = starting_cash
    profit = []
    for _, row in trades.iterrows():
        shares_bought = cash / row['buy_price']
        sell_value = shares_bought * row['sell_price']
        profit.append(sell_value - cash)
        cash = sell_value

    total_profit = sum(profit)
    return total_profit

# Create and run Optuna study with RandomSampler
study = optuna.create_study(direction='maximize', sampler=RandomSampler())
study.optimize(objective, n_trials=1000, n_jobs=-1)

time.sleep(1)

print("Best parameters:")
print(study.best_params)
print(f"Best total profit: {study.best_value:.2f}")
