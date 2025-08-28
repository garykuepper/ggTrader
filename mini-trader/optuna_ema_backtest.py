# python
import time
import math
import optuna
import pandas as pd
from datetime import datetime, timedelta, timezone

# Import your Backtest and EMAStrategy (and get_yf_data) from the module you improved
from mini_trader_trader import Backtest, EMAStrategy, get_yf_data


def nearest_4hr(date: datetime):
    hour = date.hour
    floored_hour = (hour // 4) * 4
    return date.replace(hour=floored_hour)


# ---------- Configuration ----------
SYMBOLS = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD','DOGE-USD', 'LTC-USD','SHIB-USD','XLM-USD','LINK-USD']
INTERVAL = '4h'
END_DATE = nearest_4hr(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
DAYS_OF_HISTORY = 180  # lower -> faster
START_DATE = END_DATE - timedelta(days=DAYS_OF_HISTORY)

STORAGE = "sqlite:///ema_multi_symbol_cached.db"
N_TRIALS = 100
N_JOBS = -1  # increase to parallelize trials (be careful with memory & CPU)

# ---------- Preload OHLC (once) ----------
print("Downloading data for all symbols (one-time)...")
ohlc_cache = {}
for sym in SYMBOLS:
    print(f"  fetching {sym} ...")
    df = get_yf_data(sym, INTERVAL, START_DATE, END_DATE)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {sym} - check symbol/interval/dates")
    ohlc_cache[sym] = df

# compute bar_delta from first downloaded DataFrame
_any_df = next(iter(ohlc_cache.values()))
date_index = _any_df.index
if len(date_index) < 2:
    raise RuntimeError("Downloaded data has fewer than 2 rows; cannot compute bar spacing")
BAR_DELTA = date_index[1] - date_index[0]
print("Data downloaded. Bar delta:", BAR_DELTA)


# ----- thangs


# ---------- Utilities ----------
def sharpe_from_equity_series(series: pd.Series, periods_per_year: int) -> float:
    if series is None or len(series) < 3:
        return -1e9
    rets = series.pct_change().dropna()
    if rets.empty:
        return -1e9
    mu = rets.mean()
    sigma = rets.std(ddof=1)
    if sigma <= 0 or pd.isna(sigma):
        return -1e9
    return float(mu / sigma * math.sqrt(periods_per_year))


def periods_per_year_for_interval(interval: str) -> int:
    if interval.endswith("h"):
        h = int(interval[:-1])
        per_day = max(1, 24 // h)
        return per_day * 365
    if interval.endswith("d"):
        d = int(interval[:-1])
        per_day = 1 // max(1, d)
        return per_day * 365
    # fallback mapping
    return {"4h": 6 * 365, "1h": 24 * 365, "1d": 365}.get(interval, 6 * 365)


# ---------- Objective factory (uses cached OHLC) ----------
def make_objective(symbols, interval, ohlc_cache, bar_delta):
    periods_per_year = periods_per_year_for_interval(interval)

    def objective(trial):
        # EMA windows
        max_window = 200
        min_fast = 20
        max_fast = int(math.floor(max_window * 0.5))
        fast_w = trial.suggest_int("fast_window", min_fast, max_fast, step=2)
        min_slow = max(fast_w + 2, int(math.floor((fast_w * 1.4) / 2.0) * 2))
        slow_w = trial.suggest_int("slow_window", min_slow, max_window, step=2)

        # other params to optimize
        cooldown_period = trial.suggest_int("cooldown_period", 1, 10)  # in bars
        hold_min_periods = trial.suggest_int("hold_min_periods", 1, 10)  # trailing stop hold min
        trail_pct = trial.suggest_int("trail_pct", 1, 10)  # trailing stop percent

        # Build Backtest and inject cached OHLC to avoid downloads
        bt = Backtest(symbols=symbols, interval=interval, start_date=START_DATE, end_date=END_DATE,
                      cooldown_period=cooldown_period, hold_min_periods=hold_min_periods, trail_pct=trail_pct)

        # Inject the cached data and bar_delta directly
        bt.ohlc_data_dict = {sym: ohlc_cache[sym] for sym in symbols}
        bt.bar_delta = bar_delta

        # Compute signals on cached OHLC for this trial's EMA windows
        bt.signal_data_dict = {}
        for sym in symbols:
            df = bt.ohlc_data_dict[sym]
            sig = EMAStrategy(df, fast_w, slow_w).calc_signals()
            bt.signal_data_dict[sym] = sig

        # Run backtest
        try:
            bt.run()
        except Exception as e:
            # If a trial fails, penalize it and continue
            print(f"Trial failed: {e}")
            return -1e9

        # Prefer Sharpe from equity curve if available
        eq_series = getattr(bt.portfolio, "equity_curve", None)
        if isinstance(eq_series, pd.Series) and not eq_series.empty:
            sh = sharpe_from_equity_series(eq_series.sort_index(), periods_per_year)
            return float(sh)

    return objective


# ---------- Run Optuna ----------
def main():
    study_name = f"{END_DATE.strftime('%Y-%m-%d-%H')}_bro_dude"
    study = optuna.create_study(direction="maximize", storage=STORAGE, study_name=study_name, load_if_exists=True)
    obj = make_objective(SYMBOLS, INTERVAL, ohlc_cache, BAR_DELTA)
    study.optimize(obj, n_trials=N_TRIALS, n_jobs=N_JOBS)

    time.sleep(0.3)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
