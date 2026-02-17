# src/ggTrader/core/Optimization.py
import optuna
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from ggTrader.core.fast_backtest import FastBacktest


class WalkForwardOptimizer:
    def __init__(self, ohlcv_df, train_days, test_days, n_trials, n_jobs=-1):
        self.ohlcv_df = ohlcv_df
        self.train_days = train_days
        self.test_days = test_days
        self.n_trials = n_trials
        self.n_jobs = n_jobs

    def _objective(self, trial, train_data, param_ranges):
        # 1. Build Params from Ranges
        params = {}
        for key, config in param_ranges.items():
            if config["type"] == "int":
                params[key] = trial.suggest_int(
                    key, config["min"], config["max"], step=config.get("step", 1)
                )
            elif config["type"] == "float":
                params[key] = trial.suggest_float(
                    key, config["min"], config["max"], step=config.get("step", 0.1)
                )
            elif config["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, config["choices"])

        # 2. Run FastBacktest
        bt = FastBacktest(train_data, params)
        pf = bt.run()

        # 3. Return Metric (Sortino)
        metric = pf.sortino_ratio()
        if isinstance(metric, pd.Series):
            metric = metric.mean()
        return metric if np.isfinite(metric) else -10.0

    def optimize_window(self, train_start, train_end, param_ranges):
        train_data = self.ohlcv_df[
            (self.ohlcv_df.index >= train_start) & (self.ohlcv_df.index < train_end)
        ]
        if train_data.empty:
            return {}

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: self._objective(t, train_data, param_ranges),
            n_trials=self.n_trials,
        )
        return study.best_params

    def run(self, param_ranges):
        # 1. Generate Windows
        windows = []
        curr = self.ohlcv_df.index.min()
        end_time = self.ohlcv_df.index.max()

        while True:
            train_end = curr + pd.Timedelta(days=self.train_days)
            test_end = train_end + pd.Timedelta(days=self.test_days)
            if test_end > end_time:
                break
            windows.append(
                {
                    "train_start": curr,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                }
            )
            curr += pd.Timedelta(days=self.test_days)

        # 2. Parallel Optimization
        print(f"Optimizing {len(windows)} windows (Jobs: {self.n_jobs})...")
        best_params_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.optimize_window)(
                w["train_start"], w["train_end"], param_ranges
            )
            for w in windows
        )

        # 3. Stitch Results
        results = []
        current_capital = 10000.0

        for i, window in enumerate(windows):
            params = best_params_list[i]
            if not params:
                continue

            test_data = self.ohlcv_df[
                (self.ohlcv_df.index >= window["test_start"])
                & (self.ohlcv_df.index < window["test_end"])
            ]
            start_cap = current_capital
            bt = FastBacktest(test_data, params, start_cash=start_cap)
            bt.run()
            stats = bt.get_stats()

            current_capital = stats["total_value"]
            results.append(
                {
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "params": params,
                    "start_capital": start_cap,
                    "end_capital": stats["total_value"],
                    "profit": stats["total_profit"],
                    "return_pct": stats["profit_pct"],
                    "sharpe": stats["sharpe"],
                    "sortino": stats["sortino"],
                }
            )

        return pd.DataFrame(results)
