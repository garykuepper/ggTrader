# python
"""
optuna_ema_backtest.py

Run Optuna optimization with rolling (walk-forward) evaluation over multiple symbols.
Requires: pandas, optuna, your mini_trader_trader.Backtest and EMAStrategy available in PYTHONPATH.
"""

import math
import time
import optuna
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List

# Import your Backtest and EMAStrategy from the file you improved.
from mini_trader_trader import Backtest, EMAStrategy, get_yf_data

# ---------- CONFIG ----------
SYMBOLS = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD']
INTERVAL = '4h'
END_DATE = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
HISTORY_DAYS = 365  # how much history to download (days)
START_DATE = END_DATE - timedelta(days=HISTORY_DAYS)

STORAGE = "sqlite:///optuna_ema_backtest.db"
N_TRIALS = 100
N_JOBS = -1  # change if you understand parallel workers and caching
WINDOW_DAYS = 90      # length of each rolling window (days)
WINDOW_STEP_DAYS = 30 # step between windows (days)

# ---------- HELPERS ----------
def periods_per_year_for_interval(interval: str) -> int:
    if interval.endswith("h"):
        h = int(interval[:-1])
        per_day = max(1, 24 // h)
        return per_day * 365
    if interval.endswith("d"):
        d = int(interval[:-1])
        per_day = 1 // max(1, d)
        return per_day * 365
    return {"4h": 6 * 365, "1h": 24 * 365, "1d": 365}.get(interval, 6 * 365)

def sharpe_from_series(eq: pd.Series, periods_per_year: int) -> float:
    if eq is None or len(eq) < 3:
        return float("-inf")
    rets = eq.pct_change().dropna()
    if rets.empty:
        return float("-inf")
    mu = rets.mean()
    sigma = rets.std(ddof=1)
    if sigma <= 0 or pd.isna(sigma):
        return float("-inf")
    return float((mu / sigma) * math.sqrt(periods_per_year))

# Convert days -> number of bars roughly by using index spacing of a single df later.
# We'll compute windows using timestamps (start/end dates) and slice dataframes by date index.

# ---------- PRELOAD OHLC (one-time) ----------
print("Downloading OHLC for symbols (one-time)...")
ohlc_cache = {}
for sym in SYMBOLS:
    print(f"  fetching {sym}")
    df = get_yf_data(sym, INTERVAL, START_DATE, END_DATE)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {sym}")
    # ensure index is DatetimeIndex and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    ohlc_cache[sym] = df

# determine bar delta from first symbol
_sample_df = next(iter(ohlc_cache.values()))
if len(_sample_df.index) < 2:
    raise RuntimeError("Not enough bars to determine bar delta.")
BAR_DELTA = _sample_df.index[1] - _sample_df.index[0]
PERIODS_PER_YEAR = periods_per_year_for_interval(INTERVAL)
print("Downloaded. Bar delta:", BAR_DELTA, "Periods/year approx:", PERIODS_PER_YEAR)

# --------- ROLLING WINDOW SLICER ----------
def rolling_windows(start: datetime, end: datetime, window_days: int, step_days: int):
    cur_start = start
    while True:
        cur_end = cur_start + timedelta(days=window_days)
        if cur_end > end:
            break
        yield cur_start, cur_end
        cur_start = cur_start + timedelta(days=step_days)

# ---------- OBJECTIVE FACTORY ----------
def make_objective(symbols: List[str], interval: str, ohlc_cache: dict):
    def objective(trial):
        # Hyperparameters
        max_window = 100
        min_fast = 6
        max_fast = int(math.floor(max_window * 0.5))
        fast_w = trial.suggest_int("fast_window", min_fast, max_fast, step=2)
        min_slow = max(fast_w + 2, int(math.floor((fast_w * 1.4) / 2.0) * 2))
        slow_w = trial.suggest_int("slow_window", min_slow, max_window, step=2)

        cooldown_period = trial.suggest_int("cooldown_period", 0, 10)     # bars
        hold_min_periods = trial.suggest_int("hold_min_periods", 1, 8)    # trailing stop consecutive
        trail_pct = trial.suggest_int("trail_pct", 1, 10)                # trailing stop percent

        # For each rolling window, run a backtest and collect metric (Sharpe)
        window_metrics = []

        global_start = min(df.index[0] for df in ohlc_cache.values())
        global_end = max(df.index[-1] for df in ohlc_cache.values())

        # generate windows based on timestamps
        for ws, we in rolling_windows(global_start, global_end, WINDOW_DAYS, WINDOW_STEP_DAYS):
            # build a per-window Backtest using cached OHLC sliced to [ws, we)
            bt = Backtest(symbols=symbols, interval=interval, start_date=ws, end_date=we,
                          cooldown_period=cooldown_period, hold_min_periods=hold_min_periods, trail_pct=trail_pct)

            # inject window-sliced ohlc
            bt.ohlc_data_dict = {}
            skip_window = False
            for s in symbols:
                df = ohlc_cache[s]
                df_win = df.loc[(df.index >= ws) & (df.index < we)]
                if df_win.empty or len(df_win.index) < 10:
                    # not enough data in this window: skip it (or penalize)
                    skip_window = True
                    break
                bt.ohlc_data_dict[s] = df_win
            if skip_window:
                continue

            # set bar delta from one of the windowed dfs
            first_df = next(iter(bt.ohlc_data_dict.values()))
            bt.bar_delta = first_df.index[1] - first_df.index[0]

            # compute signals for this trial's EMA windows (per symbol)
            bt.signal_data_dict = {}
            for s in symbols:
                bt.signal_data_dict[s] = EMAStrategy(bt.ohlc_data_dict[s], fast_w, slow_w).calc_signals()

            # run backtest for this window
            try:
                bt.run()
            except Exception as e:
                # trial fails for this window; penalize and stop early
                print(f"Window {ws}..{we} failed: {e}")
                return float("-1e9")

            # prefer sharpe from equity_curve if available
            eq = getattr(bt.portfolio, "equity_curve", None)
            if isinstance(eq, pd.Series) and not eq.empty:
                sh = sharpe_from_series(eq.sort_index(), PERIODS_PER_YEAR)
                if sh == float("-inf"):
                    # fallback to profit if sharpe invalid
                    profit = getattr(bt.portfolio, "profit", None)
                    window_metrics.append(float(profit or -1e9))
                else:
                    window_metrics.append(sh)
            else:
                profit = getattr(bt.portfolio, "profit", None)
                window_metrics.append(float(profit or -1e9))

        # If we have no usable windows, penalize
        if not window_metrics:
            return float("-1e9")

        # Aggregate metric across windows - prefer mean Sharpe (robustness)
        # You can change this to median, min, or sum of profits.
        metric = float(pd.Series(window_metrics).mean())
        return metric

    return objective

# ---------- RUN OPTUNA ----------
def main():
    study = optuna.create_study(direction="maximize", storage=STORAGE, study_name="ema_rolling_multi", load_if_exists=True)
    obj = make_objective(SYMBOLS, INTERVAL, ohlc_cache)
    study.optimize(obj, n_trials=N_TRIALS, n_jobs=N_JOBS)

    time.sleep(0.3)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
