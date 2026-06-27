#!/usr/bin/env python
"""Portfolio blend analysis: compare S&P 500, MidCap 400, and blended strategies.

Calculates OOS WFO returns for:
  - S&P 500 (Large Cap)
  - MidCap 400 (Mid Cap)
  - Unified blended universe (Large + Mid Cap combined)
  - Blended portfolios (50/50, 70/30, and Risk Parity)

Measures asset-class correlation and details diversification benefits.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Type

import pandas as pd

from ggTrader.lab.data import (
    STOCK_BASE_CONFIG,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import combo_name, split_params
from ggTrader.lab.wfo import generate_folds

# ── Configuration ──────────────────────────────────────────────────────
EVAL_START = "2021-01-31"
EVAL_END = None  # now

# Recommended stable 5-voter parameters from midcap WFO and ablation WFO
BEST_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.5,
    "divergence_window": 10,
    "ema_fast": 10,
    "ema_slow": 50,
    "macd_fast": 12,
    "macd_signal": 9,
    "macd_slow": 26,
    "min_agree": 3,
    "min_agree_exit": 2,
    "rsi_oversold": 25,
    "rsi_period": 14,
    "vol_mult": 1.5,
    "vol_period": 20,
    "weekly_rsi_exit": 50,
    "weekly_rsi_oversold": 30,
    "weekly_rsi_period": 14,
}


def _row(label: str, s: Dict[str, float]) -> str:
    """Format one Markdown performance-table row from a curve_stats dict."""
    return (
        f"| {label} | {s['cagr_pct']:.2f}% | {s['sharpe']:.2f} | {s['sortino']:.2f} "
        f"| {s['ann_vol_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% |"
    )


def get_wfo_equity_curve(
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_start: str,
    eval_end: str,
    base_config: Dict[str, Any],
    winner_params: Dict[str, Any],
) -> pd.Series:
    """Run WFO test folds with fixed parameters and return the cumulative OOS equity curve."""
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)

    strat_instance = strategy_cls(cfg)
    strategy_name = strategy_cls.name if hasattr(strategy_cls, "name") else "ensemble"

    oos_curves = []
    start_cash = float(base_config["START_CASH"])
    running_cash = start_cash

    # Split stop params from entry params
    signal_combos = [split_params(winner_params)[0]]
    _, stop_p = split_params(winner_params)
    sim_config = {**base_config, **stop_p}

    for _f_idx, (_train_start, _train_end, test_start, test_end) in enumerate(folds):
        test_start_str = str(test_start.date())
        test_end_str = str(test_end.date())

        # Slice test window (including padding for indicator calculations)
        test_ohlcv = ohlcv.loc[:test_end_str]
        if test_ohlcv.empty:
            continue

        symbols = sorted(test_ohlcv.columns.get_level_values(0).unique())
        prices = pd.concat({s: test_ohlcv[s]["close"] for s in symbols}, axis=1)

        # Calculate targets (sweep_signals handles indicator warmups)
        targets = strat_instance.sweep_signals(signal_combos, symbols, test_ohlcv)

        key = combo_name(strategy_name, signal_combos[0])
        full_key = combo_name(strategy_name, winner_params)

        ohlcv_arg = test_ohlcv if "atr_mult" in stop_p else None

        _r, eq, _d = simulate_signals({full_key: targets[key]}, prices, sim_config, ohlcv=ohlcv_arg)

        # Extract only the test portion
        eq_test = eq[full_key].loc[test_start_str:test_end_str].dropna()

        if len(eq_test) > 0:
            normalized = running_cash * (eq_test / eq_test.iloc[0])
            running_cash = float(normalized.iloc[-1])
            oos_curves.append(normalized)

    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        return oos_equity
    return pd.Series(dtype=float)


def main() -> None:
    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC").normalize()

    eval_start_str = str(eval_start.date())
    eval_end_str = str(eval_end.date())

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    data_start_str = str(data_start.date())
    data_end_str = eval_end_str

    print("Loading universes...", flush=True)
    sp500_members = equity_universe_between(eval_start, eval_end, universe="sp500")
    midcap_members = equity_universe_between(eval_start, eval_end, universe="midcap400")

    # Deduplicate combined universe
    combined_members = sorted(list(set(sp500_members + midcap_members)))

    print(f"S&P 500 PIT: {len(sp500_members)} symbols")
    print(f"MidCap 400:  {len(midcap_members)} symbols")
    print(f"Combined:    {len(combined_members)} unique symbols")

    all_symbols = sorted(list(set(combined_members + ["SPY", "MDY"])))

    print("\nLoading market data from TimescaleDB...", flush=True)
    t_load = time.time()
    ohlcv = load_ohlcv(
        all_symbols,
        data_start_str,
        data_end_str,
        use_negative_cache=True,
    )
    print(f"Loaded market data in {time.time() - t_load:.1f}s")

    # Slice benchmark closes
    spy_close = ohlcv["SPY"]["close"].dropna()
    mdy_close = ohlcv["MDY"]["close"].dropna()

    # Separate dataframes
    available = ohlcv.columns.get_level_values(0)
    sp500_symbols = [s for s in sp500_members if s in available]
    midcap_symbols = [s for s in midcap_members if s in available]

    sp500_ohlcv = ohlcv[sp500_symbols]
    midcap_ohlcv = ohlcv[midcap_symbols]
    combined_ohlcv = ohlcv[[s for s in combined_members if s in available]]

    base_config = dict(STOCK_BASE_CONFIG)

    print("\nRunning WFO for S&P 500 (Large Cap)...", flush=True)
    t0 = time.time()
    sp500_eq = get_wfo_equity_curve(
        EnsembleSignal, cfg, sp500_ohlcv, eval_start_str, eval_end_str, base_config, BEST_PARAMS
    )
    print(f"Completed S&P 500 in {time.time() - t0:.1f}s")

    print("Running WFO for MidCap 400 (Mid Cap)...", flush=True)
    t0 = time.time()
    midcap_eq = get_wfo_equity_curve(
        EnsembleSignal, cfg, midcap_ohlcv, eval_start_str, eval_end_str, base_config, BEST_PARAMS
    )
    print(f"Completed MidCap 400 in {time.time() - t0:.1f}s")

    print("Running WFO for Unified Blended Universe (Combined 900+ tickers)...", flush=True)
    t0 = time.time()
    combined_eq = get_wfo_equity_curve(
        EnsembleSignal, cfg, combined_ohlcv, eval_start_str, eval_end_str, base_config, BEST_PARAMS
    )
    print(f"Completed Combined Universe in {time.time() - t0:.1f}s")

    # Ensure indices align
    common_idx = sp500_eq.index.intersection(midcap_eq.index)
    sp500_eq = sp500_eq.loc[common_idx]
    midcap_eq = midcap_eq.loc[common_idx]
    combined_eq = combined_eq.loc[common_idx]

    # Calculate daily returns
    sp500_ret = sp500_eq.pct_change().dropna()
    midcap_ret = midcap_eq.pct_change().dropna()
    combined_ret = combined_eq.pct_change().dropna()

    # Compute correlations
    corr = sp500_ret.corr(midcap_ret)
    corr_combined_sp = combined_ret.corr(sp500_ret)
    corr_combined_mc = combined_ret.corr(midcap_ret)
    print(f"\nOOS Daily Returns Correlation (Large Cap vs Mid Cap): {corr:.4f}")
    print(f"OOS Daily Returns Correlation (Combined vs Large Cap):  {corr_combined_sp:.4f}")
    print(f"OOS Daily Returns Correlation (Combined vs Mid Cap):    {corr_combined_mc:.4f}")

    # ── Reconstruct benchmarks ──
    spy_oos = spy_close.reindex(common_idx).ffill().dropna()
    mdy_oos = mdy_close.reindex(common_idx).ffill().dropna()

    spy_ret = spy_oos.pct_change().dropna()
    mdy_ret = mdy_oos.pct_change().dropna()

    spy_curve = spy_oos / spy_oos.iloc[0] * 10000.0
    mdy_curve = mdy_oos / mdy_oos.iloc[0] * 10000.0

    # Blended Benchmark (50/50 return rebalanced daily)
    blend_bench_ret = 0.5 * spy_ret + 0.5 * mdy_ret
    blend_bench_eq = (1.0 + blend_bench_ret).cumprod() * 10000.0

    # ── Mathematical Blends ──
    # 1. 50/50 capital allocation
    blend_50_50_ret = 0.5 * sp500_ret + 0.5 * midcap_ret
    blend_50_50_eq = (1.0 + blend_50_50_ret).cumprod() * 10000.0

    # 2. 70/30 capital allocation (tilted large cap)
    blend_70_30_ret = 0.7 * sp500_ret + 0.3 * midcap_ret
    blend_70_30_eq = (1.0 + blend_70_30_ret).cumprod() * 10000.0

    # 3. Risk Parity (inverse volatility allocation).
    #    NOTE: weights are derived from full-period realized volatility, so this
    #    row is IN-SAMPLE / illustrative only — not a tradeable out-of-sample result.
    vol_sp = sp500_ret.std()
    vol_mc = midcap_ret.std()
    w_sp = (1.0 / vol_sp) / ((1.0 / vol_sp) + (1.0 / vol_mc))
    w_mc = 1.0 - w_sp

    blend_rp_ret = w_sp * sp500_ret + w_mc * midcap_ret
    blend_rp_eq = (1.0 + blend_rp_ret).cumprod() * 10000.0

    # ── Compute statistics ──
    stats_sp500 = curve_stats(sp500_eq)
    stats_midcap = curve_stats(midcap_eq)
    stats_combined = curve_stats(combined_eq)

    stats_50_50 = curve_stats(blend_50_50_eq)
    stats_70_30 = curve_stats(blend_70_30_eq)
    stats_rp = curve_stats(blend_rp_eq)

    stats_spy = curve_stats(spy_curve)
    stats_mdy = curve_stats(mdy_curve)
    stats_blend_bench = curve_stats(blend_bench_eq)

    # Print Markdown Report
    print("\n" + "=" * 80)
    print("PORTFOLIO BLEND & DIVERSIFICATION REPORT")
    print("=" * 80)
    print(f"Evaluation Period: {eval_start_str} to {eval_end_str}")
    print(f"Voters:            {EnsembleSignal.name} (5-voters: BB, RSI, EMA, MACD, VolBB)")
    print(f"OOS Return Correlation: **{corr:.4f}**")
    print("-" * 80)

    print("\n### Performance Comparison Table")
    print("| Strategy / Portfolio | CAGR (%) | Sharpe | Sortino | Annual Vol (%) | Max DD (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Standalone Strategies
    print(_row("S&P 500 (Large Cap)", stats_sp500))
    print(_row("MidCap 400 (Mid Cap)", stats_midcap))

    # Combined Universe (WFO on 900+ tickers)
    print(_row("Unified Blended Universe", stats_combined))

    # Mathematical Blends
    print(_row("Blended Portfolio (50/50)", stats_50_50))
    print(_row("Blended Portfolio (70/30)", stats_70_30))
    print(_row(f"Blended Portfolio (Risk Parity, {w_sp:.1%}/{w_mc:.1%}, in-sample)", stats_rp))

    # Benchmarks
    print(_row("SPY (S&P 500 Buy & Hold)", stats_spy))
    print(_row("MDY (MidCap 400 Buy & Hold)", stats_mdy))
    print(_row("Blended Benchmark (50/50)", stats_blend_bench))

    print("\n### Key Observations:")
    print(
        f"1. **Correlation benefit**: The Large/Mid returns correlation is **{corr:.4f}** "
        f"(Combined vs Large {corr_combined_sp:.4f}, Combined vs Mid {corr_combined_mc:.4f}), "
        "indicating strong but imperfect correlation, allowing for noticeable diversification."
    )
    print(
        "2. **Unified vs. Return Blending**: The Unified Blended Universe runs a single "
        "portfolio across all 900+ tickers. Look at whether it outperforms mathematical "
        "returns-blending (often it does due to dynamic asset selection)."
    )
    print(
        f"3. **Risk Parity**: Allocates {w_sp:.1%} S&P 500 and {w_mc:.1%} MidCap 400 based on "
        "inverse volatility. Weights are fit on full-period realized vol, so treat this row as "
        "in-sample/illustrative — a tradeable version needs rolling-window volatility estimates."
    )


if __name__ == "__main__":
    main()
