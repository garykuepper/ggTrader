import optuna
import pandas as pd
import sys
import os
import argparse
import traceback
import json
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.config import load_symbols_from_json
from ggTrader.indicators.signals import Signals
import vectorbt as vbt
import numpy as np
from tqdm import tqdm
import logging
from tabulate import tabulate
from joblib import Parallel, delayed
from optuna.samplers import RandomSampler

# --- Configuration ---
SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "DOT", "LINK", "LTC", "BCH"]
INTERVAL = "4h"
TRAIN_WINDOW_DAYS = 365
TEST_WINDOW_DAYS = 120
N_TRIALS = 20  # Trials per window
MIN_TRADES = 5  # Minimum trades to avoid skewed results
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
N_JOBS = -1  # Use all available cores for parallel windows
DEFAULT_SYMBOLS = [
    "BTC",
    "ETH",
    "XRP",
    "SOL",
    "DOGE",
    "ADA",
    "DOT",
    "LINK",
    "LTC",
    "BCH",
]
SYMBOLS_FILE_DEFAULT = "data/top_50_consistent_movers.json"


# Global data for optimization (loaded once)
global_ohlcv_df = pd.DataFrame()
global_date_range = None


# Helper to ensure DataFrames are converted to writable NumPy arrays (handles CuPy to NumPy)
def force_numpy_writable(df):
    if df is None:
        return None
    try:
        # to_numpy() usually handles the conversion, but we need an explicit copy
        vals = df.to_numpy()
        if hasattr(vals, "get"):  # CuPy array detection
            vals = vals.get()
        return pd.DataFrame(
            np.array(vals, copy=True), index=df.index, columns=df.columns
        )
    except Exception:
        # Fallback for older pandas or edge cases
        vals = df.values
        if hasattr(vals, "get"):
            vals = vals.get()
        return pd.DataFrame(
            np.array(vals, copy=True), index=df.index, columns=df.columns
        )


def run_optimization(train_start, train_end, n_trials=N_TRIALS):
    window_ohlcv = global_ohlcv_df[
        (global_ohlcv_df.index >= train_start) & (global_ohlcv_df.index < train_end)
    ]

    if window_ohlcv.empty:
        return None

    # Pre-calculate data for optimization to avoid repetitive overhead
    close = window_ohlcv.xs("close", axis=1, level=1, drop_level=True).copy()
    high = window_ohlcv.xs("high", axis=1, level=1, drop_level=True).copy()
    low = window_ohlcv.xs("low", axis=1, level=1, drop_level=True).copy()
    open_ = window_ohlcv.xs("open", axis=1, level=1, drop_level=True).copy()

    def objective(trial):
        params = {
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
            "adx_length": trial.suggest_int("adx_length", 10, 20),
            "sar_acceleration": trial.suggest_float(
                "sar_acceleration", 0.01, 0.05, step=0.005
            ),
            "sar_maximum": trial.suggest_float("sar_maximum", 0.1, 0.3, step=0.05),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 4.0, step=0.5),
            "atr_length": trial.suggest_int("atr_length", 10, 20),
            "use_dmp_cross": trial.suggest_categorical("use_dmp_cross", [True, False]),
        }

        try:
            # Use Signals class directly for lightning fast VectorBT-based backtesting
            entries, exits, _, price_for_orders = Signals.calc_signals(
                close=close, high=high, low=low, open_=open_, **params
            )

            # Force writable NumPy copies for VectorBT stability
            entries_vbt = force_numpy_writable(entries)
            exits_vbt = force_numpy_writable(exits)
            price_vbt = force_numpy_writable(price_for_orders)

            pf = vbt.Portfolio.from_signals(
                close=price_vbt,
                entries=entries_vbt,
                exits=exits_vbt,
                init_cash=10000,
                fees=0.001,  # Standard Kraken fee
                slippage=0.0005,
            )

            # Robustness check: Min number of trades
            total_trades = pf.trades.count().sum()
            if total_trades < MIN_TRADES:
                return -10.0  # Heavy penalty for low statistical significance

            return pf.sortino_ratio() if not np.isnan(pf.sortino_ratio()) else -5.0

        except Exception:
            return -20.0  # Extreme penalty for failure

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=RandomSampler())
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def run_backtest(test_start, test_end, params, start_cash):
    window_ohlcv = global_ohlcv_df[
        (global_ohlcv_df.index >= test_start) & (global_ohlcv_df.index < test_end)
    ]

    if window_ohlcv.empty:
        return start_cash, {}

    # Use Trading engine for the actual walk-forward "out of sample" execution
    # to maintain compatibility with the real trading loop logic
    engine = Trading(
        ohlcv_df=global_ohlcv_df,
        date_range=window_ohlcv.index,
        start_cash=start_cash,
        top_n_movers=5,
        max_position=0.2,
        strategy_params=params,
    )

    try:
        engine.run()
        stats = engine.portfolio.stats_dict()
        # Add ratios for tabulate
        stats["sharpe"] = engine.portfolio.sharpe_ratio()
        stats["sortino"] = engine.portfolio.sortino_ratio()
        return engine.portfolio.total_value, stats
    except Exception as e:
        print(f"  Test run failed: {e}")
        return start_cash, {}


def main():
    global global_ohlcv_df, global_date_range

    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=SYMBOLS_FILE_DEFAULT,
        help="Path to JSON file with symbols",
    )
    parser.add_argument(
        "--n-trials", type=int, default=N_TRIALS, help="Number of trials per window"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=N_JOBS, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    rm = ResultsManager("run_wfo")

    # 1. Load Symbols
    symbols = load_symbols_from_json(args.symbols_file)
    if symbols:
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    else:
        symbols = DEFAULT_SYMBOLS
        print(f"Using default symbols: {symbols}")

    # 2. Load Data
    print("Loading data for WFO...")
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(START_DATE).tz_localize("UTC")
    end_dt = pd.to_datetime(END_DATE).tz_localize("UTC")

    global_ohlcv_df = k_h.get_ohlcv_df(
        symbols, interval=INTERVAL, start=start_dt, end=end_dt
    )

    if isinstance(global_ohlcv_df.columns, pd.MultiIndex):
        global_ohlcv_df.columns.names = [None] * global_ohlcv_df.columns.nlevels
    else:
        global_ohlcv_df.columns.name = None
    global_ohlcv_df.index.name = None
    global_date_range = global_ohlcv_df.index

    print(f"Data loaded: {len(global_ohlcv_df)} rows.")

    # Check for GPU (Enhanced Diagnostics)
    try:
        import cupy
        import os

        # Only print if we actually have a device
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count > 0:
            print(
                f"GPU (CuPy) detected! {device_count} device(s) available. VectorBT will leverage GPU acceleration."
            )
        else:
            print("CuPy is installed, but no CUDA devices were found. Running on CPU.")
            print(
                "Note: If you have an NVIDIA GPU, ensure the 'CUDA Toolkit' is installed and CUDA_PATH is set."
            )
    except Exception:
        # If it's the common CUDA_PATH warning, we can be more specific
        print("No functional GPU (CuPy) detected. Running on CPU.")

    # 2. Window Generation
    windows = []
    curr_start = start_dt
    while True:
        train_end = curr_start + timedelta(days=TRAIN_WINDOW_DAYS)
        test_start = train_end
        test_end = test_start + timedelta(days=TEST_WINDOW_DAYS)

        if test_end > end_dt:
            break

        windows.append(
            {
                "train_start": curr_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        curr_start = curr_start + timedelta(days=TEST_WINDOW_DAYS)

    print(f"Starting WFO with {len(windows)} windows...")

    # 3. Parallel Optimization
    print(f"Running optimizations in parallel (n_jobs={args.n_jobs})...")
    # This is the heavy lifting
    all_best_params = Parallel(n_jobs=args.n_jobs)(
        delayed(run_optimization)(
            w["train_start"], w["train_end"], n_trials=args.n_trials
        )
        for w in windows
    )

    # 4. Sequential Backtesting (to handle capital flow)
    results = []
    current_capital = 10000
    all_test_stats = []

    print("Executing walk-forward backtests...")
    for i, window in enumerate(windows):
        best_params = all_best_params[i]

        if best_params:
            final_value, stats = run_backtest(
                window["test_start"], window["test_end"], best_params, current_capital
            )

            profit = final_value - current_capital
            pct_return = (profit / current_capital) * 100

            results.append(
                {
                    "test_start": window["test_start"].date(),
                    "test_end": window["test_end"].date(),
                    "start_capital": current_capital,
                    "end_capital": final_value,
                    "profit": profit,
                    "return_pct": pct_return,
                    "sharpe": stats.get("sharpe", 0),
                    "sortino": stats.get("sortino", 0),
                    "params": best_params,
                }
            )
            current_capital = final_value
            all_test_stats.append(stats)
        else:
            print(f"  Warning: No params for window {i}. Skipping.")

    # 5. Final Report & Metrics
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("      WALK FORWARD OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Overall Period: {START_DATE} to {END_DATE}")
    print(f"Total Return:   {((current_capital - 10000)/10000)*100:.2f}%")
    print(f"Final Capital:  ${current_capital:,.2f}")

    if not df_res.empty:
        # Save results
        rm.save_metrics(df_res, "wfo_results.csv")
        rm.save_params(results[-1]["params"])

        # Better formatting with tabulate
        print("\nWindow Details:")
        view_cols = [
            "test_start",
            "test_end",
            "start_capital",
            "end_capital",
            "profit",
            "return_pct",
            "sharpe",
            "sortino",
        ]
        print(
            tabulate(
                df_res[view_cols], headers="keys", tablefmt="github", floatfmt=".2f"
            )
        )

        # --- 6. Aggregate Visualization ---
        print("\nGenerating Aggregate Equity Curve...")
        try:
            # We have the end_capital from each window.
            # Let's create a series starting from 10000
            capital_values = [10000] + df_res["end_capital"].tolist()
            # Create a index for the plot (test_end dates)
            plot_dates = [start_dt.date()] + df_res["test_end"].tolist()

            equity_series = pd.Series(capital_values, index=plot_dates)

            # Use plotly directly for the simplest, most robust plot
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=equity_series.index,
                    y=equity_series.values,
                    mode="lines+markers",
                    name="Walk-Forward Equity",
                )
            )
            fig.update_layout(
                title="Walk-Forward Stitched Equity Curve",
                xaxis_title="Date",
                yaxis_title="Account Balance ($)",
                template="plotly_dark",
            )
            rm.save_plotly_figure(fig, "wfo_stitched_equity")

            # Save equity series to database
            rm.save_equity_curve(equity_series, "wfo_stitched_equity")
        except Exception as e:
            print(f"Stitched plot failed: {e}")

        # Export to Excel
        metadata = {
            "total_return_pct": ((current_capital - 10000) / 10000) * 100,
            "final_capital": current_capital,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "test_window_days": TEST_WINDOW_DAYS,
            "avg_sharpe": df_res["sharpe"].mean(),
            "avg_sortino": df_res["sortino"].mean(),
        }
        params_df = pd.DataFrame([r["params"] for r in results])
        params_df.index = df_res["test_start"]

        excel_data = {
            "Summary": pd.DataFrame.from_dict(
                metadata, orient="index", columns=["Value"]
            ),
            "Window Results": df_res,
            "Parameters": params_df,
        }

        # Save metadata to DB (this triggers add_run)
        metadata["params"] = results[-1]["params"]  # Representative params for the run
        rm.save_metadata(metadata)

        excel_path = rm.save_excel(excel_data, "wfo_results.xlsx")
        print(f"\nExcel report saved to: {excel_path}")

        # --- 7. Final Full Period Backtest with Best Parameters ---
        print("\n" + "=" * 50)
        print("      FINAL FULL PERIOD BACKTEST (STATIONARY)")
        print("=" * 50)
        print(f"Using best params from final window: {results[-1]['params']}")

        try:
            # 1. Get full data
            close = global_ohlcv_df.xs("close", axis=1, level=1, drop_level=True).copy()
            high = global_ohlcv_df.xs("high", axis=1, level=1, drop_level=True).copy()
            low = global_ohlcv_df.xs("low", axis=1, level=1, drop_level=True).copy()
            open_ = global_ohlcv_df.xs("open", axis=1, level=1, drop_level=True).copy()

            # 2. Strategy Calculation
            best_params = results[-1]["params"]
            entries, exits, _, price_for_orders = Signals.calc_signals(
                close=close, high=high, low=low, open_=open_, **best_params
            )

            # Force writable NumPy copies for VectorBT stability
            entries_vbt = force_numpy_writable(entries)
            exits_vbt = force_numpy_writable(exits)
            price_vbt = force_numpy_writable(price_for_orders)

            # 3. VectorBT Backtest - Use group_by=True for portfolio-level stats
            pf = vbt.Portfolio.from_signals(
                close=price_vbt,
                entries=entries_vbt,
                exits=exits_vbt,
                init_cash=10000,
                fees=0.001,
                slippage=0.0005,
                group_by=True,  # Aggregates multiple symbols into one portfolio for stats
            )

            # 4. Output Stats
            print("\nVectorBT Full Period Statistics:")
            try:
                # Try specific metrics first to avoid the problematic ones if necessary
                # We prioritize the core metrics the user needs
                metrics_to_try = [
                    "total_return",
                    "benchmark_return",
                    "max_drawdown",
                    "max_drawdown_duration",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "total_trades",
                    "win_rate",
                    "expectancy",
                ]
                # Note: 'profit_factor' is excluded here as it's the known source of the read-only error
                stats = pf.stats(metrics=metrics_to_try)
                print(tabulate(stats.to_frame(), headers="keys", tablefmt="github"))

                # Manually calculate Profit Factor (portfolio total)
                try:
                    p = pf.trades.total_profit()
                    l = abs(pf.trades.total_loss())
                    # Because of group_by=True, these should already be aggregated, but we sum just in case
                    total_p = p.sum() if hasattr(p, "sum") else p
                    total_l = l.sum() if hasattr(l, "sum") else l
                    # Force to scalar for truth checks
                    if isinstance(total_l, (pd.Series, pd.DataFrame)):
                        total_l = total_l.iloc[0] if not total_l.empty else 0
                    if isinstance(total_p, (pd.Series, pd.DataFrame)):
                        total_p = total_p.iloc[0] if not total_p.empty else 0

                    pf_val = total_p / total_l if total_l != 0 else float("inf")
                    print(
                        f"| Profit Factor | {pf_val:.4f} | (Manually Calculated Portfolio Total)"
                    )
                except Exception as pfe:
                    print(f"Manual Profit Factor calculation failed: {pfe}")
            except Exception as stats_err:
                print(
                    f"Standard stats() failed: {stats_err}. Falling back to manual calculation."
                )
                # Manual Fallback for the most important metrics
                try:
                    total_ret = pf.total_return()
                    mdd = pf.max_drawdown()
                    sharpe = pf.sharpe_ratio()
                    # Flatten to scalars for dictionary
                    manual_stats = {
                        "Total Return [%]": (
                            total_ret.sum() if hasattr(total_ret, "sum") else total_ret
                        )
                        * 100,
                        "Max Drawdown [%]": (mdd.max() if hasattr(mdd, "max") else mdd)
                        * 100,
                        "Sharpe Ratio": (
                            sharpe.mean() if hasattr(sharpe, "mean") else sharpe
                        ),
                        "Total Trades": (
                            pf.trades.count().sum()
                            if hasattr(pf.trades.count(), "sum")
                            else pf.trades.count()
                        ),
                    }
                    print(
                        tabulate(
                            pd.Series(manual_stats).to_frame(),
                            headers="keys",
                            tablefmt="github",
                        )
                    )
                except Exception as manual_err:
                    print(f"Manual stats calculation also failed: {manual_err}")

            # 5. Full Period Equity Plot
            import plotly.graph_objects as go

            fig_full = go.Figure()
            # value() is safe when inputs are fresh copies
            equity_values = pf.value()
            fig_full.add_trace(
                go.Scatter(
                    x=equity_values.index,
                    y=equity_values.values,
                    mode="lines",
                    name="Full Period Equity",
                )
            )
            fig_full.update_layout(
                title=f"Full Period Backtest (2023-2025) - Best Params",
                xaxis_title="Date",
                yaxis_title="Account Balance ($)",
                template="plotly_dark",
            )
            rm.save_plotly_figure(fig_full, "full_period_backtest_equity")
            print("Full period backtest plot saved.")

        except Exception as e:
            print(f"Final full backtest failed: {e}")

        print(f"\nAll results saved in: {rm.run_dir}")


if __name__ == "__main__":
    main()
