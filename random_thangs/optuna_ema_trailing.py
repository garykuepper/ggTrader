import pandas as pd
from optuna.samplers import RandomSampler
from ta.trend import EMAIndicator
import optuna
import time
from data_manager import CryptoDataManager
from datetime import datetime, timedelta

# Load data once
symbol = 'BNBUSDT'
interval = '4h'
end_date = datetime(2025, 8, 5)
start_date = end_date - timedelta(days=60)

df = CryptoDataManager().get_crypto_data(symbol, interval, start_date, end_date)
starting_cash = 1000

def objective(trial):
    # Suggest hyperparameters
    fast_window = trial.suggest_int('fast_window', 5, 15)
    slow_window = trial.suggest_int('slow_window', 20, 30)  # slow > fast
    trailing_pct = trial.suggest_float('trailing_pct', 0.005, 0.10)
    min_hold_bars = trial.suggest_int('min_hold_bars', 3, 12)

    # Calculate EMAs
    df['ema_fast'] = EMAIndicator(df['close'], window=fast_window).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=slow_window).ema_indicator()

    ema_fast_above = df['ema_fast'] > df['ema_slow']
    shifted = ema_fast_above.shift(1).fillna(False).astype(bool)

    cross_up = ema_fast_above & (~shifted)
    cross_down = (~ema_fast_above) & shifted

    buy_signals = df.loc[cross_up, ['close']].reset_index().rename(columns={'date':'buy_date', 'close':'buy_price'})
    sell_signals = df.loc[cross_down, ['close']].reset_index().rename(columns={'date':'sell_date', 'close':'sell_price'})

    trades = pd.merge_asof(
        buy_signals.sort_values('buy_date'),
        sell_signals.sort_values('sell_date'),
        left_on='buy_date', right_on='sell_date',
        direction='forward'
    ).dropna()

    if trades.empty:
        return -1e9

    cash = starting_cash
    profit = []

    for _, trade in trades.iterrows():
        buy_date = trade['buy_date']
        buy_price = trade['buy_price']
        planned_sell_date = trade['sell_date']
        planned_sell_price = trade['sell_price']

        df_slice = df.loc[buy_date:planned_sell_date]

        highest_price = buy_price
        trailing_stop_price = highest_price * (1 - trailing_pct)
        triggered = False
        actual_sell_price = planned_sell_price
        actual_sell_date = planned_sell_date

        for i, (dt, row) in enumerate(df_slice.iterrows()):
            if i < min_hold_bars:
                continue  # enforce minimum hold period

            price_high = row['high']
            price_low = row['low']

            if price_high > highest_price:
                highest_price = price_high
                trailing_stop_price = highest_price * (1 - trailing_pct)

            if price_low <= trailing_stop_price:
                actual_sell_price = row['open']  # realistic exit price
                actual_sell_date = dt
                triggered = True
                break

        shares_bought = cash / buy_price
        sell_value = shares_bought * actual_sell_price
        profit.append(sell_value - cash)
        cash = sell_value

    return sum(profit)

# Create and run Optuna study with RandomSampler
study = optuna.create_study(direction='maximize', sampler=RandomSampler())
study.optimize(objective, n_trials=1000, n_jobs=-1)

time.sleep(1)

print("Best parameters:")
for key, value in study.best_params.items():
    if isinstance(value, int):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.4f}")
print(f"Best total profit: {study.best_value:.2f}")

print(f"Best total profit: {study.best_value:.2f}")
