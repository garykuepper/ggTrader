#!/usr/bin/env python
"""Survivorship-bias calibration: SP500 PIT (span-union) vs current snapshot.

Runs the SAME 5-voter EnsembleSignal WFO two ways on SP500, differing only in
which tickers load:

  1. PIT (span-union): equity_universe_between(eval_start, eval_end, "sp500")
     — the survivorship-bias-free set used in all lab research.
  2. Snapshot: data/universe/sp500_tickers_snapshot_2026-06-09.txt
     — current constituents only, introducing survivorship bias.

The delta (snapshot - PIT) quantifies the survivorship haircut, which Task 4
applies to the midcap-400 result (where we only have a snapshot).

Run:
    source .venv/bin/activate
    python scripts/midcap_bias_calibration.py 2>&1 | tee bias_calibration.log
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker
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
SNAPSHOT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "universe"
    / "sp500_tickers_snapshot_2026-06-09.txt"
)


def _load_universe_pit(eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> List[str]:
    """PIT span-union universe (survivorship-bias-free)."""
    members = equity_universe_between(eval_start, eval_end, universe="sp500")
    print(f"[PIT]      {len(members)} symbols (span-union)", flush=True)
    return members


def _load_universe_snapshot() -> List[str]:
    """Current-snapshot universe (introduces survivorship bias)."""
    raw = SNAPSHOT_FILE.read_text().split()
    members = sorted({normalize_yf_ticker(t) for t in raw if t.strip()})
    print(f"[snapshot] {len(members)} symbols from {SNAPSHOT_FILE.name}", flush=True)
    return members


def _load_and_split(
    members: List[str],
    data_start: str,
    data_end: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load OHLCV for members + SPY, split SPY into spy_close, drop SPY."""
    ohlcv = load_ohlcv(
        members + ["SPY"],
        data_start,
        data_end,
        use_negative_cache=True,
    )
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]
    return ohlcv, spy_close


def _run_one_wfo(
    label: str,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    cfg: LabConfig,
    eval_start: str,
    eval_end: str,
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a single WFO and parse the result table."""
    t0 = time.time()
    print(f"\n{'=' * 70}\n[{label}] Starting WFO...\n{'=' * 70}", flush=True)
    table = run_wfo(
        "ensemble",
        EnsembleSignal,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )
    elapsed = time.time() - t0
    print(f"[{label}] Finished in {elapsed:.0f}s", flush=True)
    parsed = parse_table(table)
    parsed["label"] = label
    parsed["elapsed_s"] = round(elapsed, 1)
    return parsed


def main() -> None:
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

    # ── Run 1: PIT (span-union) ───────────────────────────────────────
    pit_members = _load_universe_pit(eval_start, eval_end)
    pit_ohlcv, pit_spy = _load_and_split(pit_members, data_start_str, data_end_str)
    pit = _run_one_wfo("SP500-PIT", pit_ohlcv, pit_spy, cfg, eval_start_str, eval_end_str, grid)

    # ── Run 2: Snapshot ───────────────────────────────────────────────
    snap_members = _load_universe_snapshot()
    snap_ohlcv, snap_spy = _load_and_split(snap_members, data_start_str, data_end_str)
    snap = _run_one_wfo(
        "SP500-snapshot", snap_ohlcv, snap_spy, cfg, eval_start_str, eval_end_str, grid
    )

    # ── Report ────────────────────────────────────────────────────────
    delta_cagr = snap["oos_cagr"] - pit["oos_cagr"]
    delta_sharpe = snap["oos_sharpe"] - pit["oos_sharpe"]
    print(f"\n{'=' * 70}")
    print("SURVIVORSHIP-BIAS CALIBRATION RESULT")
    print(f"{'=' * 70}")
    print(f"SP500 PIT:      CAGR {pit['oos_cagr']:.1f}% Sharpe {pit['oos_sharpe']:.2f}")
    print(f"SP500 snapshot: CAGR {snap['oos_cagr']:.1f}% Sharpe {snap['oos_sharpe']:.2f}")
    print(f"Survivorship haircut  Δ: CAGR {delta_cagr:+.1f}pp  Sharpe {delta_sharpe:+.2f}")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
