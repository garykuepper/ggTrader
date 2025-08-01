import pandas as pd
from optuna.samplers import RandomSampler
from ta.trend import EMAIndicator
import optuna
import time as t
from ggTrader.data_manager.universal_data_manager import UniversalDataManager
from datetime import datetime, timedelta, date, time

pd.set_option('future.no_silent_downcasting', True)
# Load and preprocess data once
# df = pd.read_csv("yf_BTC_1m_5d.csv")
# df['date'] = pd.to_datetime(df['date'])
# df = df.set_index('date')



symbol = "AVAXUSDT"
interval = "1h"

end_date = datetime.combine(date.today() - timedelta(days=1), time.min)
start_date = end_date - timedelta(days=30)
time_diff = end_date - start_date
num_days = time_diff.days
marketType = "crypto"
# Initialize UniversalDataManager to handle data loading and fetching
dm = UniversalDataManager()

df = dm.load_or_fetch(symbol, interval, start_date, end_date, market=marketType)

starting_cash = 10000


def objective(trial):
    # Suggest EMA window parameters
    fast_window = trial.suggest_int('fast_window', 6, 25)
    slow_window = trial.suggest_int('slow_window', fast_window + 2, 40)  # slow > fast

    # Calculate EMAs with trial parameters
    df['ema_fast'] = EMAIndicator(df['close'], window=fast_window).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=slow_window).ema_indicator()

    # Generate signals
    ema_fast_above = df['ema_fast'] > df['ema_slow']

    # Detect bullish crossovers with proper handling of FutureWarning
    cross_up = (ema_fast_above) & (~ema_fast_above.shift(1).fillna(False).astype(bool))
    cross_down = (~ema_fast_above) & (ema_fast_above.shift(1).fillna(False).astype(bool))
    # Prepare buy and sell signals
    buy_signals = df.loc[cross_up, ['close']].reset_index().rename(
        columns={'datetime': 'buy_date', 'close': 'buy_price'})
    sell_signals = df.loc[cross_down, ['close']].reset_index().rename(
        columns={'datetime': 'sell_date', 'close': 'sell_price'})

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

t.sleep(1)

print("Best parameters:")
print(study.best_params)
print(f"Best total profit: ${study.best_value:.2f}")
print(f"Best daily profit: ${study.best_value/num_days:.2f}")
# Save the best parameters
best_params = study.best_params
dm.save_optimization_parameters(
    symbol=symbol,
    strategy_name="ema_crossover",
    interval=interval,
    start_date=start_date,
    end_date=end_date,
    parameters={
        'ema_fast': best_params['fast_window'],
        'ema_slow': best_params['slow_window']
    }
)