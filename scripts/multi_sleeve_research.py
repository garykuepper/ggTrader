#!/usr/bin/env python
"""Multi-sleeve target-vol research harness.

Runs sp500 / midcap400 / nasdaq100 through their own gated WFO, blends the
gate-honest OOS curves with rolling inverse-vol weighting scaled to a target
volatility (leverage capped), and reports vs the gated SP500 core.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import STOCK_BASE_CONFIG, equity_universe_between, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo

EVAL_START = "2021-01-31"
SLEEVES = ("sp500", "midcap400", "nasdaq100")


def _row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['cagr_pct']:.2f}% | {s['sharpe']:.2f} | {s['sortino']:.2f} "
        f"| {s['ann_vol_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-sleeve target-vol research harness")
    ap.add_argument("--target-vol", type=float, default=0.068)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC").normalize()
    eval_start_str, eval_end_str = str(eval_start.date()), str(eval_end.date())

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start_str = str((eval_start - pd.Timedelta(days=warmup_days)).date())

    members: dict[str, list[str]] = {
        s: equity_universe_between(eval_start, eval_end, universe=s) for s in SLEEVES
    }
    all_symbols = sorted({sym for ms in members.values() for sym in ms} | {"SPY"})
    print(f"[load] {len(all_symbols)} symbols across {len(SLEEVES)} sleeves", flush=True)
    ohlcv = load_ohlcv(all_symbols, data_start_str, eval_end_str, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    grid = build_grid(EnsembleSignal)

    curves: dict[str, pd.Series] = {}
    for s in SLEEVES:
        syms = [x for x in members[s] if x in available]
        print(f"[wfo] {s}: {len(syms)} symbols", flush=True)
        t0 = time.time()
        result = run_wfo(
            "ensemble",
            EnsembleSignal,
            cfg,
            ohlcv[syms],
            ohlcv["SPY"]["close"].dropna(),
            eval_start=eval_start_str,
            eval_end=eval_end_str,
            market="equity",
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )
        curves[s] = result.oos_equity
        print(f"[wfo] {s} done in {time.time() - t0:.0f}s", flush=True)

    # Align gated OOS curves on common dates -> daily returns.
    common = curves["sp500"].index
    for s in SLEEVES:
        common = common.intersection(curves[s].index)
    returns_df = pd.DataFrame({s: curves[s].reindex(common).pct_change() for s in SLEEVES}).dropna()

    blended, diag = combine_sleeves(
        returns_df,
        target_vol=args.target_vol,
        window=args.window,
        max_leverage=args.max_leverage,
    )
    blend_eq = (1.0 + blended).cumprod() * float(STOCK_BASE_CONFIG["START_CASH"])

    # -- Report --
    print("\n" + "=" * 80)
    print("MULTI-SLEEVE TARGET-VOL RESEARCH REPORT (gate-honest)")
    print("=" * 80)
    print(f"Eval: {eval_start_str} to {eval_end_str}")
    print(f"target_vol={args.target_vol}  max_leverage={args.max_leverage}x  window={args.window}d")
    print("\nGated sleeve OOS correlation matrix:")
    print(returns_df.corr().round(4).to_string())
    print(f"\nRealized leverage: avg {diag['scale'].mean():.2f}x  max {diag['scale'].max():.2f}x")

    print("\n| Strategy | CAGR | Sharpe | Sortino | Vol | Max DD |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(_row("S&P 500 (gated core)", curve_stats(curves["sp500"].reindex(common))))
    print(_row("MidCap 400 (gated)", curve_stats(curves["midcap400"].reindex(common))))
    print(_row("Nasdaq-100 (gated)", curve_stats(curves["nasdaq100"].reindex(common))))
    print(_row("Inverse-vol + target-vol blend", curve_stats(blend_eq)))

    print(
        "\nCaveats: (1) sleeve curves are gate-honest (anchor/halt applied) but the "
        "leverage carries NO borrow or transaction cost in this model. (2) target_vol "
        "is a fixed a-priori constant, not re-fit per period."
    )


if __name__ == "__main__":
    main()
