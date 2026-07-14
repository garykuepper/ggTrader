"""Generate today's ensemble signals from a given equity universe."""

from __future__ import annotations

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import fetch_stock_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.paper.overlay import (
    SLEEVE_UNIVERSES,
    compute_sleeve_curve,
    compute_weights_and_scale,
    should_rebalance,
)
from ggTrader.paper.persist import get_rebalance_state, save_rebalance_state


def generate_signals(universe: str = "sp500", lookback_days: int = 120) -> dict:
    """Fetch recent data for a PIT universe and return today's ensemble signals.

    Returns dict with keys: buys (list[str]), sells (list[str]),
    as_of (str date), universe_size (int).
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=lookback_days)

    members = universe_members_asof(universe, today)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(today.date()))
    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)

    if close.empty:
        return {"buys": [], "sells": [], "as_of": str(today.date()), "universe_size": 0}

    last_bar = close.index[-1]
    cfg = LabConfig(min_history_bars=60)
    ensemble = EnsembleSignal(cfg)
    plan = ensemble.select(last_bar, ohlcv, sym_cols)
    if not plan:
        return {
            "buys": [],
            "sells": [],
            "as_of": str(last_bar.date()),
            "universe_size": len(sym_cols),
        }
    targets = ensemble.to_targets({last_bar: plan}, ohlcv)

    last_entries = targets.entries.loc[last_bar]
    last_exits = targets.exits.loc[last_bar]

    buys = sorted(last_entries[last_entries].index.tolist())
    sells = sorted(last_exits[last_exits].index.tolist())

    # ML feature gate — filter low-confidence buy signals
    gate_info: dict = {}
    from ggTrader.paper.feature_gate import FeatureGate

    gate = FeatureGate()
    if gate.enabled and buys:
        raw_count = len(buys)
        buys, scores = gate.filter_buys(buys, ohlcv)
        gate_info = {
            "gate_enabled": True,
            "raw_buys": raw_count,
            "kept_buys": len(buys),
            "scores": scores,
        }
    else:
        gate_info = {"gate_enabled": gate.enabled}

    return {
        "buys": buys,
        "sells": sells,
        "as_of": str(last_bar.date()),
        "universe_size": len(sym_cols),
        "gate": gate_info,
    }


def generate_blended_signals() -> dict:
    """Generate today's signals for all three sleeves, recomputing the
    inverse-vol/target-vol overlay monthly. On any failure to recompute
    (e.g. an OHLCV fetch error on a rebalance date), falls back to the last
    stored weights/scale rather than raising."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    sleeves = {universe: generate_signals(universe=universe) for universe in SLEEVE_UNIVERSES}

    state = get_rebalance_state()
    rebalanced_today = False
    fallback_used = False

    if should_rebalance(state["rebalance_date"] if state else None, today):
        try:
            curves = {u: compute_sleeve_curve(u, today) for u in SLEEVE_UNIVERSES}
            weights, scale = compute_weights_and_scale(curves)
            save_rebalance_state(str(today.date()), weights, scale)
            rebalanced_today = True
        except Exception:
            if state is None:
                raise  # no fallback available on the very first run
            weights, scale = state["weights"], state["scale"]
            fallback_used = True
    else:
        weights, scale = state["weights"], state["scale"]

    return {
        "sleeves": sleeves,
        "weights": weights,
        "scale": scale,
        "rebalanced_today": rebalanced_today,
        "fallback_used": fallback_used,
    }
