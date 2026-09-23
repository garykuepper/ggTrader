"""Generate today's ensemble signals from a given equity universe."""

from __future__ import annotations

import logging

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

_log = logging.getLogger(__name__)

#: Symbols no sleeve trades but the lab benchmarks against or builds
#: sleeves from. Nothing else on the box fetches them daily, so the
#: paper run keeps their tape alive (SPY's went dead 2026-08-21 when the
#: old benchmark-only writer was retired; TLT/GLD/DBC stalled 2026-07-20).
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "TLT", "GLD", "DBC", "IEF")


def refresh_benchmark_tape(lookback_days: int = 30) -> list[str]:
    """Re-fetch the trailing window of the benchmark/ETF tape and upsert it.

    Deliberately bypasses `CachedYFinanceLoader.fetch_ohlcv`'s freshness
    check: this runs at 12:45 PT while the session is open, so the newest
    bar written is always partial, and the freshness check would judge it
    current the next day and never refetch it. Overwriting the trailing
    window every run (the writer upserts on timestamp/symbol) means only
    the current day's bar is ever partial. Never raises: this is a side job
    of the live run and must not block trading. Returns the symbols that
    came back with data (empty on any failure).
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=lookback_days)
    try:
        from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
        from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

        df = YFinanceDataLoader().fetch_ohlcv(
            list(BENCHMARK_SYMBOLS), "1d", start_date=start, end_date=today
        )
        if df.empty:
            return []
        CachedYFinanceLoader()._cache_to_db(df, "1d")
    except Exception as exc:
        _log.warning("benchmark tape refresh failed (non-fatal): %s", exc)
        return []
    return sorted(df.columns.get_level_values(0).unique().tolist())


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
        except Exception:
            if state is None:
                raise  # no fallback available on the very first run
            weights, scale = state["weights"], state["scale"]
            fallback_used = True
        else:
            # Only a recompute failure (fetch/compute) falls back to stale
            # state. A persistence failure here is a genuine error and
            # should propagate rather than be swallowed into fallback_used.
            save_rebalance_state(str(today.date()), weights, scale)
            rebalanced_today = True
    else:
        weights, scale = state["weights"], state["scale"]

    return {
        "sleeves": sleeves,
        "weights": weights,
        "scale": scale,
        "rebalanced_today": rebalanced_today,
        "fallback_used": fallback_used,
    }


def generate_core_signals() -> dict:
    """Generate today's signals for the standalone SP500 core strategy,
    wrapped in the same shape `generate_blended_signals()` returns so
    `trader.py`'s sleeve-iteration/buy-sizing logic needs no changes.

    Deployed 2026-09 in place of the 3-sleeve blend: the corrected-tape
    pinned-window re-baseline
    (`docs/research/_rebaseline_corrected_tape_20260822.json`) shows the
    blend underperforming this standalone core (Sharpe 0.69 vs 0.99, third
    independent confirmation) -- see `docs/next_steps.md`. Kept alongside
    `generate_blended_signals` (not deleted) since the blend's WFO/research
    infrastructure (`ggt lab --blend`) is still valid tooling for any future
    diversification-sleeve candidate that actually clears the bar.
    """
    core_signals = generate_signals(universe="sp500")
    return {
        "sleeves": {"sp500": core_signals},
        "weights": {"sp500": 1.0},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }
