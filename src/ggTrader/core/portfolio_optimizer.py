"""
Portfolio Competition & Optimization
Generates allocation weights from WFO signals and exports `portfolio_weights.json`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.utils.setup import load_data_with_movers


def load_wfo_results(results_dir: str) -> Tuple[Dict, Dict]:
    """Load results from a specific WFO run directory."""
    res_path = Path(results_dir) / "run_results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"Could not find run_results.json in {results_dir}")

    with open(res_path, 'r') as f:
        data = json.load(f)

    config = data.get("configuration", {})
    raw_config = config.get("_raw_config", {})

    # 1. Look in strategy_parameters -> per_coin (new pipeline structure)
    sp = data.get("strategy_parameters", {})
    if "per_coin" in sp:
        per_coin_results = sp["per_coin"]
    # 2. Look in root (legacy pipeline structure)
    elif "per_coin_results" in data:
        per_coin_results = data["per_coin_results"]
    # 3. Look in configuration -> _raw_config -> per_coin_results (old recalibration structure)
    elif "per_coin_results" in raw_config:
        per_coin_results = raw_config["per_coin_results"]
    else:
        raise ValueError("Could not find per_coin_results in JSON")

    return per_coin_results, raw_config

def generate_combined_signals(ohlcv: pd.DataFrame, per_coin_results: Dict, global_config: Dict):
    """Generate entries and exits for all coins using their best WFO parameters."""
    combined_entries = []
    combined_exits = []
    combined_close = []
    symbol_robustness = {}

    symbols = list(per_coin_results.keys())

    for symbol in symbols:
        if symbol not in ohlcv.columns.get_level_values(0):
            print(f"Warning: Symbol {symbol} not in OHLCV, skipping.")
            continue

        res = per_coin_results[symbol]
        best_params = res["best_params"]
        best_strategy = res["best_strategy"]
        best_exit = res["best_exit"]
        rob_score = res.get("robustness_score", 0.0)

        # Hard Gate: Skip assets that failed or have negative robustness
        if rob_score <= 0.0:
            print(f"Skipping {symbol} (Gate Failed: Robustness = {rob_score:.2f})")
            continue

        symbol_robustness[symbol] = rob_score

        # Setup engine for this symbol natively
        config = {
            **global_config,
            "ENTRY_STRATEGY": best_strategy,
            "EXIT_STRATEGY": best_exit,
            "USE_VECTORIZED": False,
            "USE_CASH_SHARING": False
        }

        sym_ohlcv = ohlcv[[symbol]]
        engine = FastBacktest(sym_ohlcv, best_params, config=config)
        engine.run(show_progress=False)

        def _get_single_col(df, sym):
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    return df.xs(sym, level=-1, axis=1)
                except Exception:
                    return df.iloc[:, [0]]
            else:
                return df[[sym]]

        entries = _get_single_col(engine.entries, symbol)
        exits = _get_single_col(engine.exits, symbol)
        close = sym_ohlcv.xs("close", level=1, axis=1)

        if isinstance(entries, pd.DataFrame) and entries.shape[1] > 1:
            entries = entries.iloc[:, [0]]
        if isinstance(exits, pd.DataFrame) and exits.shape[1] > 1:
            exits = exits.iloc[:, [0]]
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            close = close.iloc[:, [0]]

        entries = entries.copy()
        exits = exits.copy()
        close = close.copy()

        entries.columns = [symbol]
        exits.columns = [symbol]
        close.columns = [symbol]

        combined_entries.append(entries)
        combined_exits.append(exits)
        combined_close.append(close)

    e_df = pd.concat(combined_entries, axis=1)
    x_df = pd.concat(combined_exits, axis=1)
    c_df = pd.concat(combined_close, axis=1)

    e_df.columns = [str(c) if not isinstance(c, tuple) else str(c[-1]) for c in e_df.columns]
    x_df.columns = [str(c) if not isinstance(c, tuple) else str(c[-1]) for c in x_df.columns]
    c_df.columns = [str(c) if not isinstance(c, tuple) else str(c[-1]) for c in c_df.columns]

    return e_df, x_df, c_df, symbol_robustness

def run_strategy_equal_share(close, entries, exits, config, share=0.1):
    c = close.astype(float).copy()
    e = entries.astype(bool).copy()
    x = exits.astype(bool).copy()

    pf = vbt.Portfolio.from_signals(
        close=c, entries=e, exits=x,
        init_cash=float(config.get("START_CASH", 1000.0)),
        fees=float(config.get("FEES", 0.004)),
        slippage=float(config.get("SLIPPAGE", 0.0)),
        freq=config.get("FREQ", "4h"),
        size=float(share),
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0)
    )
    weights = pd.DataFrame(share, index=close.index, columns=close.columns)
    return pf, weights

def run_strategy_max_diversified(close, entries, exits, config):
    n = close.shape[1]
    pf = vbt.Portfolio.from_signals(
        close=close.astype(float),
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        init_cash=float(config.get("START_CASH", 1000.0)),
        fees=float(config.get("FEES", 0.004)),
        slippage=float(config.get("SLIPPAGE", 0.0)),
        freq=config.get("FREQ", "4h"),
        size=1.0 / n,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0)
    )
    weights = pd.DataFrame(1.0 / n, index=close.index, columns=close.columns)
    return pf, weights

def run_strategy_robustness_weighted(close, entries, exits, config, robustness_scores):
    scores = pd.Series(robustness_scores)
    scores[scores < 0] = 0.0

    score_sum = scores.sum()
    if score_sum > 0:
        weights = scores / score_sum
    else:
        weights = pd.Series(0.0, index=scores.index)

    weights_df = pd.DataFrame(
        [weights.values] * len(close), index=close.index, columns=close.columns
    )

    pf = vbt.Portfolio.from_signals(
        close=close.astype(float),
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        init_cash=float(config.get("START_CASH", 1000.0)),
        fees=float(config.get("FEES", 0.004)),
        slippage=float(config.get("SLIPPAGE", 0.0)),
        freq=config.get("FREQ", "4h"),
        size=weights_df,
        size_type="percent",
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0)
    )
    return pf, weights_df

def run_portfolio_competition(results_dir: str):
    """Orchestrates the portfolio competition and generates final weights."""
    print(f"Loading Master WFO results from {results_dir}...")
    per_coin_results, raw_config = load_wfo_results(results_dir)

    # CRITICAL: Overwrite raw_config symbols with the full set of optimized results
    # so that the data loader fetches OHLCV for ALL symbols, not just the worker's subset.
    raw_config["SYMBOLS"] = list(per_coin_results.keys())
    # Remove SYMBOLS_FILE so the loader doesn't try to reload from original
    # file which might also be truncated.
    raw_config["SYMBOLS_FILE"] = None

    print("Loading OHLCV Universe Data...")
    ohlcv, _ = load_data_with_movers(raw_config)

    print("Generating combined signals block natively...")
    entries, exits, close, robustness = generate_combined_signals(
        ohlcv, per_coin_results, raw_config
    )

    strategies = {}
    weights = {}

    print("Running Baseline (Current: Flat 10%)...")
    strategies["Flat 10%"], weights["Flat 10%"] = run_strategy_equal_share(
        close, entries, exits, raw_config, share=0.1
    )

    print("Running Max Diversified (1/N)...")
    strategies["Max Diversified (1/N)"], weights["Max Diversified (1/N)"] = (
        run_strategy_max_diversified(close, entries, exits, raw_config)
    )

    print("Running Robustness Weighted...")
    strategies["Robustness Weighted"], weights["Robustness Weighted"] = (
        run_strategy_robustness_weighted(close, entries, exits, raw_config, robustness)
    )

    stats = []
    for name, pf in strategies.items():
        s = pf.stats()
        stats.append({
            "Strategy": name,
            "Total Return [%]": s['Total Return [%]'],
            "Sharpe Ratio": s['Sharpe Ratio'],
            "Sortino Ratio": s['Sortino Ratio'],
            "Max Drawdown [%]": s['Max Drawdown [%]'],
            "Win Rate [%]": s['Win Rate [%]'],
            "Profit Factor": s['Profit Factor'],
            "Total Trades": s['Total Trades']
        })

    df_stats = pd.DataFrame(stats).sort_values("Sharpe Ratio", ascending=False)

    print("\n=========================================")
    print("  PORTFOLIO STRATEGY COMPETITION RESULTS")
    print("=========================================")
    print(df_stats.to_markdown(index=False))

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path("results") / "production" / f"production_{timestamp_str}" / "portfolio_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(output_dir / "comparison_stats.csv", index=False)

    # Automatically select the strategy with the best Sharpe Ratio
    best_strat_name = df_stats.iloc[0]["Strategy"]

    best_weights = weights[best_strat_name]
    latest_weights = best_weights.iloc[-1].to_dict()

    # Normalize weights to sum to 1.0 and cap per MAX_COIN_ALLOCATION so that
    # the live execution engine (which has no cash-sharing logic) allocates the
    # same fractions that vectorbt used in the backtest.
    max_alloc = float(raw_config.get("MAX_COIN_ALLOCATION", 0.25))
    w_total = sum(latest_weights.values())
    if w_total > 0:
        latest_weights = {s: w / w_total for s, w in latest_weights.items()}
        # Iterative cap (mirrors _compute_allocation_weights in orchestrator)
        for _ in range(len(latest_weights) + 1):
            over = {s: w for s, w in latest_weights.items() if w > max_alloc + 1e-12}
            if not over:
                break
            excess = sum(w - max_alloc for w in over.values())
            for s in over:
                latest_weights[s] = max_alloc
            under = {s: w for s, w in latest_weights.items() if s not in over}
            if not under:
                extra = excess / len(latest_weights)
                latest_weights = {s: w + extra for s, w in latest_weights.items()}
                break
            per_under = excess / len(under)
            for s in under:
                latest_weights[s] += per_under

    weight_file = output_dir / "portfolio_weights.json"
    with open(weight_file, 'w') as f:
        json.dump({
            "strategy": best_strat_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weights": latest_weights
        }, f, indent=4)

    print(f"\n[WINNER] Selected strategy: '{best_strat_name}'")
    print(f"Exported final Live Trading weights to {weight_file}")
    print("\nPhase 4: PRODUCTION RECALIBRATION COMPLETE!")
