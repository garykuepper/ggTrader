# Python
from trading_strategy import EMAStrategy
from data_manager import CryptoDataManager
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import optuna
import numpy as np
import pandas as pd


def align_to_binance_4h(dt):
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=(dt.hour // 4) * 4)


# Load data once
cm = CryptoDataManager()
symbol, interval = "BTCUSDT", "4h"
end_date = align_to_binance_4h(datetime.now(timezone.utc))
start_date = end_date - timedelta(days=30*8)  # use more history for windows
df = cm.get_crypto_data(symbol, interval, start_date, end_date)

# Ensure usable dataframe
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()


def generate_walk_forward_windows(
    data: pd.DataFrame,
    window_days: int = 90,   # slightly longer windows to increase trade chances
    step_days: int = 14,
    val_ratio: float = 0.2
):
    """
    Yields (train_df, val_df) tuples for walk-forward evaluation.
    Each window covers `window_days`, split into train:(1 - val_ratio) and val:val_ratio.
    Windows move forward by `step_days`.
    """
    if data.empty:
        return
    start_time = data.index[0]
    end_time = data.index[-1]
    win_delta = timedelta(days=window_days)
    step_delta = timedelta(days=step_days)

    left = start_time
    while True:
        right = left + win_delta
        if right > end_time:
            break
        window_df = data.loc[left:right]
        if window_df.empty:
            left += step_delta
            continue

        split_idx = int(len(window_df) * (1 - val_ratio))
        if split_idx <= 1 or split_idx >= len(window_df) - 1:
            left += step_delta
            continue

        train_df = window_df.iloc[:split_idx].copy()
        val_df = window_df.iloc[split_idx:].copy()

        # Require enough bars for EMA warm-up and meaningful validation
        if len(train_df) < 20 or len(val_df) < 10:
            left += step_delta
            continue

        yield train_df, val_df
        left += step_delta


def objective_moving_window(trial: optuna.Trial) -> float:
    # Hyperparameters to optimize
    ema_fast = trial.suggest_int("ema_fast", 5, 20)
    # enforce a minimum separation to reduce degenerate signals
    ema_slow = trial.suggest_int("ema_slow", ema_fast + 10, 80)
    trailing_pct = trial.suggest_float("trailing_pct", 0.01, 0.15, step=0.01)
    min_hold_bars = trial.suggest_int("min_hold_bars", 0, 10)

    # Window config
    window_days = 90
    step_days = 14
    val_ratio = 0.20

    strategy = EMAStrategy(
        name="ema_crossover",
        params={"ema_fast": ema_fast, "ema_slow": ema_slow},
        trailing_pct=trailing_pct

    )

    val_returns = []
    min_valid_windows_for_pruning = 2  # only prune once we have meaningful signal
    last_reported_valid_count = 0      # track to avoid duplicate step reporting

    for train_df, val_df in generate_walk_forward_windows(
        df, window_days=window_days, step_days=step_days, val_ratio=val_ratio
    ):
        # Fit on train slice (indicators/signals internal to backtest)
        _ = strategy.backtest(train_df,
                              starting_cash=1000,
                              min_hold_bars=min_hold_bars,
                              use_trailing=True,
                              trailing_price_mode="ohlc4")

        val_result = strategy.backtest(val_df,
                                       starting_cash=1000,
                                       min_hold_bars=min_hold_bars,
                                       use_trailing=True,
                                       trailing_price_mode="ohlc4")
        # Use total_return_pct; treat windows with zero trades as NaN to avoid flattening the median
        ret = val_result.get("total_return_pct", 0.0)
        trades = val_result.get("trades", [])
        if trades and len(trades) > 0:
            val_returns.append(float(ret))
        else:
            # ignore no-trade windows in scoring
            val_returns.append(np.nan)

        # Compute current valid values (exclude NaNs)
        valid_vals = [x for x in val_returns if not (isinstance(x, float) and np.isnan(x))]

        # Only report if we have strictly more valid windows than previously reported
        if len(valid_vals) >= min_valid_windows_for_pruning and len(valid_vals) > last_reported_valid_count:
            # Use the count of valid windows as the step; strictly increases now
            step = len(valid_vals)
            trial.report(float(np.median(valid_vals)), step=step)
            last_reported_valid_count = step
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final aggregation: require at least 2 valid windows
    valid_vals = [x for x in val_returns if not (isinstance(x, float) and np.isnan(x))]
    if len(valid_vals) < 2:
        # Penalize parameter sets that never trade (or almost never) during validation
        return -1e-6

    return float(np.median(valid_vals))


def run_optimization_with_walk_forward(n_trials: int = 200):
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,   # don't prune first 20 trials
        n_warmup_steps=2,      # don't prune until 2 intermediate reports
        interval_steps=1
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective_moving_window, n_trials=n_trials, n_jobs=-1)

    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"- {k}: {v}")
    print(f"Best median validation return: {study.best_value:.2f}%")

    # Re-test on the last validation window for a quick sanity check
    windows = list(generate_walk_forward_windows(df, window_days=90, step_days=14, val_ratio=0.20))
    if windows:
        _, last_val = windows[-1]
        best_strategy = EMAStrategy(
            name="ema_crossover",
            params={
                "ema_fast": study.best_params["ema_fast"],
                "ema_slow": study.best_params["ema_slow"],
            },
            trailing_pct=study.best_params["trailing_pct"],
        )
        result = best_strategy.backtest(
            last_val,
            starting_cash=1000,
            min_hold_bars=study.best_params["min_hold_bars"],
            use_trailing=True,
        )
        print(f"Last window re-test: final cash {result['final_cash']:.2f} (Return: {result['total_return_pct']:.2f}%), trades={len(result['trades'])}")


if __name__ == "__main__":
    # Run walk-forward Optuna search
    run_optimization_with_walk_forward(n_trials=150)
