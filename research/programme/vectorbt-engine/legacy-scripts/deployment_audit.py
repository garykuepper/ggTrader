#!/usr/bin/env python
"""One-off deployment audit: measure the 5-voter ensemble's actual capital
deployment across the WFO eval window.

Answers the under-deployment hypothesis: is the majority-vote ensemble sitting
in cash most of the time?  Prints mean % invested, idle-cash %, and concurrent
position counts.

Run:
    source .venv/bin/activate
    python scripts/deployment_audit.py
"""

from __future__ import annotations

import sys

import pandas as pd

from ggTrader.lab.data import (
    DEFAULT_UNIVERSE,
    STOCK_BASE_CONFIG,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig

# ── Configuration ──────────────────────────────────────────────────────
EVAL_START = "2021-01-31"
EVAL_END = None  # -> now
UNIVERSE = DEFAULT_UNIVERSE

# Live-recommended combo (single point, no sweep).
MODAL_COMBO = {
    "min_agree": 3,
    "min_agree_exit": 2,
    "bb_std": 2.5,
    "ema_fast": 20,
    "rsi_oversold": 30,
}


def main() -> None:
    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = (
        pd.Timestamp(EVAL_END, tz="UTC") if EVAL_END else pd.Timestamp.now(tz="UTC").normalize()
    )

    # ── Universe + OHLCV (same pattern as ablation_voters.py) ──────────
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end, universe=UNIVERSE)
    print(f"[universe] {UNIVERSE}: {len(universe)} symbols", flush=True)

    print("[data] loading OHLCV...", flush=True)
    ohlcv = load_ohlcv(
        universe + ["SPY"],
        str(data_start.date()),
        str(eval_end.date()),
        use_negative_cache=True,
    )
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    # ── Build 5-voter targets (single modal combo) ─────────────────────
    print("[signals] generating 5-voter ensemble targets...", flush=True)
    ensemble = EnsembleSignal(cfg)  # default = 5-voter (bb,rsi,ema,macd,vbb)
    targets_map = ensemble.sweep_signals([MODAL_COMBO], sym_cols, ohlcv)
    assert len(targets_map) == 1, f"expected 1 combo, got {len(targets_map)}"
    combo_name = next(iter(targets_map))
    targets = targets_map[combo_name]

    # ── Extract close prices aligned to signal index ───────────────────
    prices = pd.DataFrame(
        {
            s: ohlcv[s]["close"]
            for s in targets.entries.columns
            if s in ohlcv.columns.get_level_values(0)
        },
    )
    prices = prices.reindex(targets.entries.index).ffill()

    # ── Simulate and extract portfolio ─────────────────────────────────
    print("[simulate] running...", flush=True)
    base_config = dict(STOCK_BASE_CONFIG)
    _returns, _equity, _diags, pf = simulate_signals(
        {combo_name: targets},
        prices,
        base_config,
        return_pf=True,
    )

    # ── Deployment measurement ─────────────────────────────────────────
    invested = pf.asset_value(group_by=False).sum(axis=1)
    total = pf.value()
    deploy_pct = (invested / total).clip(0, None)
    n_positions = (pf.asset_value(group_by=False) > 0).sum(axis=1)

    print()
    print("=" * 60)
    print("5-VOTER DEPLOYMENT AUDIT")
    print("=" * 60)
    print(f"eval window:          {eval_start.date()} -> {eval_end.date()}")
    print(f"universe:             {UNIVERSE} ({len(sym_cols)} symbols)")
    print(f"combo:                {MODAL_COMBO}")
    print(f"bars:                 {len(deploy_pct)}")
    print()
    print(f"mean deployment:      {deploy_pct.mean():.1%}")
    print(f"idle cash:            {1 - deploy_pct.mean():.1%}")
    print(
        f"concurrent positions: mean {n_positions.mean():.1f}"
        f" / median {n_positions.median():.0f}"
        f" / max {n_positions.max()}"
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    sys.exit(main())
