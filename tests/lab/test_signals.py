# tests/lab/test_signals.py
import json

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.signals import (
    EmaCrossSignal,
    WfoTournamentSignal,
    build_signal_strategy,
)
from ggTrader.lab.strategy import LabConfig, SignalTargets


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(symbols, n=600):
    idx = _idx(n)
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_ema_cross_select_returns_eligible_symbols():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    asof = ohlcv.index[-1]
    sels = strat.select(asof, ohlcv, ["A", "B", "C"])
    assert [s["symbol"] for s in sels] == ["A", "B", "C"]
    assert all("ema_fast" in s and "ema_slow" in s for s in sels)
    assert all(s["weight"] == 0.0 for s in sels)


def test_ema_cross_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)  # fewer bars than min_history_bars=400
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_ema_cross_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof1 = ohlcv.index[300]
    asof2 = ohlcv.index[450]
    plans = {
        asof1: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50}],
        asof2: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
            {"symbol": "B", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.shape == result.exits.shape
    assert set(result.entries.columns) == {"A", "B"}
    assert result.entries.dtypes.eq(bool).all()
    assert result.exits.dtypes.eq(bool).all()


def test_ema_cross_no_lookahead():
    ohlcv = _ohlcv(["A"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof = ohlcv.index[-30]
    full = strat.select(asof, ohlcv.loc[:asof], ["A"])
    truncated = strat.select(asof, ohlcv.loc[:asof].copy(), ["A"])
    unmasked = strat.select(asof, ohlcv, ["A"])
    assert (
        json.dumps(full, sort_keys=True)
        == json.dumps(truncated, sort_keys=True)
        == json.dumps(unmasked, sort_keys=True)
    )


def test_build_signal_strategy_dispatch():
    cfg = LabConfig()
    assert build_signal_strategy("ema_cross", cfg).name == "ema_cross"
    try:
        build_signal_strategy("bogus", cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_wfo_tournament_select_returns_plan_with_params():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = WfoTournamentSignal(LabConfig(top_n=3, min_history_bars=100))
    asof = ohlcv.index[-1]
    sels = strat.select(asof, ohlcv, ["A", "B", "C"])
    assert len(sels) <= 3
    if sels:
        assert "ema_fast" in sels[0] and "ema_slow" in sels[0]
        assert "is_sharpe" in sels[0]
        # params must come from the known combo list
        assert sels[0]["ema_fast"] in (5, 10, 20, 50)


def test_wfo_tournament_select_no_lookahead():
    ohlcv = _ohlcv(["A"])
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    asof = ohlcv.index[-30]
    full = strat.select(asof, ohlcv.loc[:asof], ["A"])
    unmasked = strat.select(asof, ohlcv, ["A"])
    assert json.dumps(full, sort_keys=True) == json.dumps(unmasked, sort_keys=True)


def test_wfo_tournament_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    asof1 = ohlcv.index[300]
    asof2 = ohlcv.index[450]
    plans = {
        asof1: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50, "is_sharpe": 0.5}],
        asof2: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 10, "ema_slow": 30, "is_sharpe": 0.7},
            {"symbol": "B", "weight": 0.0, "ema_fast": 10, "ema_slow": 30, "is_sharpe": 0.7},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert "A" in result.entries.columns
    assert result.entries.dtypes.eq(bool).all()
    assert result.entries.shape[0] == len(ohlcv)


def test_build_signal_strategy_dispatch_wfo():
    strat = build_signal_strategy("wfo_tournament", LabConfig())
    assert strat.name == "wfo_tournament"


def _reference_wfo_to_targets(plans, data):
    """Mirrors the pre-vectorization per-(period, symbol) single-column
    ema_signals call with per-symbol dropna -- the exact code Phase C item
    16 replaced with a (ema_fast, ema_slow)-grouped, memoized batch call."""
    from ggTrader.lab.strategies.indicators import ema_signals
    from ggTrader.lab.strategy import SignalTargets

    all_symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
    entries = pd.DataFrame(False, index=data.index, columns=all_symbols)
    exits = pd.DataFrame(False, index=data.index, columns=all_symbols)

    sorted_dates = sorted(plans.keys())
    have = set(data.columns.get_level_values(0).unique())

    for i, asof in enumerate(sorted_dates):
        next_asof = sorted_dates[i + 1] if i + 1 < len(sorted_dates) else data.index[-1]
        period_mask = (data.index > asof) & (data.index <= next_asof)
        period_index = data.index[period_mask]
        if len(period_index) == 0:
            continue

        active = {s["symbol"] for s in plans[asof]}
        if i > 0:
            prev_active = {s["symbol"] for s in plans[sorted_dates[i - 1]]}
            for sym in prev_active - active:
                if sym in exits.columns and len(period_index) > 0:
                    exits.loc[period_index[0], sym] = True

        for sel in plans[asof]:
            sym = sel["symbol"]
            if sym not in have:
                continue
            close_sym = data[sym]["close"].dropna().to_frame(sym)
            sym_ent, sym_ext = ema_signals(
                close_sym, int(sel.get("ema_fast", 20)), int(sel.get("ema_slow", 50))
            )
            entries.loc[period_index, sym] = (
                sym_ent[sym].reindex(period_index).fillna(False).to_numpy()
            )
            exits.loc[period_index, sym] = (
                sym_ext[sym].reindex(period_index).fillna(False).to_numpy()
            )

    return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))


def test_wfo_tournament_to_targets_matches_reference_per_symbol_loop():
    """Phase C item 16 equivalence check: staggered inception dates (NaN
    prefixes of different lengths per symbol), multiple rebalance periods,
    and a repeated (ema_fast, ema_slow) pair across periods -- exercising
    both the grouped-by-pair batching and the cross-period memoization."""
    ohlcv = _ohlcv(["A", "B", "C", "D"], n=500)
    # Stagger each symbol's start by NaN-ing out a prefix, so the grouped
    # batch call sees differing per-symbol NaN spans within one call.
    for sym, cutoff in zip(["A", "B", "C", "D"], [0, 15, 40, 90]):
        ohlcv.loc[ohlcv.index[:cutoff], (sym, slice(None))] = float("nan")

    dates = ohlcv.index
    plans = {
        dates[150]: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
            {"symbol": "B", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
        ],
        dates[250]: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 10, "ema_slow": 30},
            {"symbol": "B", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
            {"symbol": "C", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
        ],
        dates[350]: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
            {"symbol": "D", "weight": 0.0, "ema_fast": 10, "ema_slow": 30},
        ],
    }

    strat = WfoTournamentSignal(LabConfig())
    result = strat.to_targets(plans, ohlcv)
    reference = _reference_wfo_to_targets(plans, ohlcv)

    pd.testing.assert_frame_equal(
        result.entries.sort_index(axis=1), reference.entries.sort_index(axis=1)
    )
    pd.testing.assert_frame_equal(
        result.exits.sort_index(axis=1), reference.exits.sort_index(axis=1)
    )
