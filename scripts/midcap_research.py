#!/usr/bin/env python
"""MidCap 400 research run: 5-voter EnsembleSignal WFO vs MDY benchmark.

Runs the standard 5-voter ensemble on the midcap400 universe, applies the
survivorship-bias haircut from the SP500 calibration (Task 3), and reports
a PASS/FAIL verdict on whether the strategy beats MDY after adjustment.

The haircut (delta) is read at run-time from bias_calibration.log (the output
of scripts/midcap_bias_calibration.py), or can be overridden via CLI flags.

Run:
    source .venv/bin/activate
    python scripts/midcap_research.py 2>&1 | tee midcap_research.log
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Dict, Tuple

import pandas as pd

from ggTrader.lab.data import (
    STOCK_BASE_CONFIG,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo

# parse_table lives in a sibling script — add scripts/ to path if needed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ablation_voters import parse_table  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────
EVAL_START = "2021-01-31"
EVAL_END = None  # -> now
CALIBRATION_LOG = os.path.join(os.path.dirname(__file__), "..", "bias_calibration.log")


def _parse_calibration_log(log_path: str) -> Tuple[float, float]:
    """Extract delta_cagr and delta_sharpe from bias_calibration.log.

    Looks for a line like:
        Survivorship haircut  Δ: CAGR +2.3pp  Sharpe +0.15
    Returns (delta_cagr, delta_sharpe).
    """
    pattern = re.compile(
        r"Survivorship haircut\s+[ΔD]:\s*CAGR\s+([\-+\d.]+)pp\s+Sharpe\s+([\-+\d.]+)"
    )
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise ValueError(f"Could not find haircut delta line in {log_path}")


def _resolve_deltas(args: argparse.Namespace) -> Tuple[float, float]:
    """Resolve delta_cagr and delta_sharpe from CLI args or calibration log."""
    if args.delta_cagr is not None and args.delta_sharpe is not None:
        print(
            f"[delta] CLI overrides: CAGR {args.delta_cagr:+.1f}pp  "
            f"Sharpe {args.delta_sharpe:+.2f}",
            flush=True,
        )
        return args.delta_cagr, args.delta_sharpe

    log_path = os.path.normpath(CALIBRATION_LOG)
    if not os.path.isfile(log_path):
        print(
            f"ERROR: Calibration log not found at {log_path}\n"
            f"Either run scripts/midcap_bias_calibration.py first, or provide\n"
            f"--delta-cagr and --delta-sharpe explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        delta_cagr, delta_sharpe = _parse_calibration_log(log_path)
    except ValueError:
        print(
            f"ERROR: Calibration log exists at {log_path} but does not contain\n"
            f"the haircut delta line. The calibration run may still be in progress.\n"
            f"Wait for it to finish, or provide --delta-cagr and --delta-sharpe.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"[delta] From {os.path.basename(log_path)}: "
        f"CAGR {delta_cagr:+.1f}pp  Sharpe {delta_sharpe:+.2f}",
        flush=True,
    )

    # CLI can override one while the other comes from log
    if args.delta_cagr is not None:
        delta_cagr = args.delta_cagr
        print(f"[delta] CLI override delta_cagr: {delta_cagr:+.1f}pp", flush=True)
    if args.delta_sharpe is not None:
        delta_sharpe = args.delta_sharpe
        print(f"[delta] CLI override delta_sharpe: {delta_sharpe:+.2f}", flush=True)

    return delta_cagr, delta_sharpe


def _compute_buy_hold(close: pd.Series, start: str, end: str) -> Dict[str, float]:
    """Compute simple buy-hold CAGR and Sharpe for a benchmark series."""
    ts = close.loc[start:end].dropna()
    if len(ts) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan")}
    total_ret = ts.iloc[-1] / ts.iloc[0]
    days = (ts.index[-1] - ts.index[0]).days
    years = days / 365.25
    cagr = (total_ret ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    daily_rets = ts.pct_change().dropna()
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * (252**0.5)
    else:
        sharpe = 0.0
    return {"cagr": cagr, "sharpe": sharpe}


def main() -> None:
    parser = argparse.ArgumentParser(description="MidCap 400 research run vs MDY")
    parser.add_argument(
        "--delta-cagr",
        type=float,
        default=None,
        help="Override survivorship haircut for CAGR (pp)",
    )
    parser.add_argument(
        "--delta-sharpe",
        type=float,
        default=None,
        help="Override survivorship haircut for Sharpe",
    )
    args = parser.parse_args()

    delta_cagr, delta_sharpe = _resolve_deltas(args)

    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = (
        pd.Timestamp(EVAL_END, tz="UTC") if EVAL_END else pd.Timestamp.now(tz="UTC").normalize()
    )

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    data_start_str = str(data_start.date())
    data_end_str = str(eval_end.date())
    eval_start_str = str(eval_start.date())
    eval_end_str = data_end_str

    grid = build_grid(EnsembleSignal)
    print(f"[config] eval {eval_start_str} -> {eval_end_str}, grid {len(grid)} combos", flush=True)

    # ── Load midcap400 universe + MDY + SPY ──────────────────────────
    members = equity_universe_between(eval_start, eval_end, universe="midcap400")
    requested_count = len(members)
    print(f"[universe] midcap400: {requested_count} symbols (span-union)", flush=True)

    all_symbols = sorted(set(members + ["MDY", "SPY"]))
    ohlcv = load_ohlcv(
        all_symbols,
        data_start_str,
        data_end_str,
        use_negative_cache=True,
    )

    # Coverage reporting
    loaded_symbols = ohlcv.columns.get_level_values(0).unique().tolist()
    member_with_data = [s for s in members if s in loaded_symbols]
    coverage_pct = len(member_with_data) / requested_count * 100 if requested_count else 0
    print(
        f"[coverage] {len(member_with_data)}/{requested_count} members with data "
        f"({coverage_pct:.0f}%)",
        flush=True,
    )
    if coverage_pct < 85:
        print(
            "WARNING: Coverage below 85% — results may be unreliable.",
            flush=True,
        )

    # Split out MDY and SPY
    mdy_close = ohlcv["MDY"]["close"].dropna()
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in loaded_symbols if s not in ("MDY", "SPY")]
    ohlcv = ohlcv[sym_cols]

    # ── Run WFO with MDY as benchmark ────────────────────────────────
    t0 = time.time()
    print(
        f"\n{'=' * 70}\n[midcap400] Starting 5-voter WFO (benchmark=MDY)...\n{'=' * 70}", flush=True
    )
    table = run_wfo(
        "ensemble",
        EnsembleSignal,
        cfg,
        ohlcv,
        mdy_close,
        eval_start=eval_start_str,
        eval_end=eval_end_str,
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )
    elapsed = time.time() - t0
    print(f"[midcap400] Finished in {elapsed:.0f}s", flush=True)

    mid = parse_table(table)

    # ── SPY cross-reference (buy-hold) ───────────────────────────────
    spy_bh = _compute_buy_hold(spy_close, eval_start_str, eval_end_str)

    # ── Apply haircut and report ─────────────────────────────────────
    adj_cagr = mid["oos_cagr"] - delta_cagr
    adj_sharpe = mid["oos_sharpe"] - delta_sharpe

    beats = adj_cagr > mid["spy_cagr"] and adj_sharpe > mid["spy_sharpe"]

    print(f"\n{'=' * 70}")
    print("MIDCAP 400 RESEARCH RESULT")
    print(f"{'=' * 70}")
    print(f"Coverage:            {len(member_with_data)}/{requested_count} ({coverage_pct:.0f}%)")
    print(f"midcap400 raw:       CAGR {mid['oos_cagr']:.1f}%  Sharpe {mid['oos_sharpe']:.2f}")
    print(
        f"midcap400 haircut:   CAGR {adj_cagr:.1f}%  Sharpe {adj_sharpe:.2f}  "
        f"(Δ from SP500 calibration: CAGR {delta_cagr:+.1f}pp  Sharpe {delta_sharpe:+.2f})"
    )
    print(f"MDY benchmark:       CAGR {mid['spy_cagr']:.1f}%  Sharpe {mid['spy_sharpe']:.2f}")
    print(f"SPY cross-ref:       CAGR {spy_bh['cagr']:.1f}%  Sharpe {spy_bh['sharpe']:.2f}")
    print(
        f"Gate pass:           {mid.get('gate_pass', '?')}/{mid.get('n_folds', '?')} folds  "
        f"(anchor: {mid.get('anchor_used', '?')})"
    )
    print(f"VERDICT: beats MDY after haircut -> {'PASS' if beats else 'FAIL'}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
