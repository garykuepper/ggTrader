"""Tests for the IC-weighted EnsembleICSignal strategy."""

import numpy as np
import pandas as pd

import ggTrader.lab.strategies.ensemble_ic as eic_mod
from ggTrader.lab.strategies import STRATEGY_REGISTRY
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.ensemble_ic import EnsembleICSignal
from ggTrader.lab.strategies.signals import SIGNAL_STRATEGY_NAMES, build_signal_strategy
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=400, n_syms=12, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[f"S{i}"] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=idx,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def _plans(ohlcv):
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    return {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}


def test_reduces_to_baseline_under_equal_weights(monkeypatch):
    """Equal weights + threshold 0.4 == baseline 2-of-5 entries."""
    ohlcv = _ohlcv()

    def _equal_weights(raw_by_voter, close, **kwargs):
        eq = 1.0 / len(raw_by_voter)
        return pd.DataFrame(eq, index=close.index, columns=list(raw_by_voter))

    monkeypatch.setattr(eic_mod, "ic_weight_schedule", _equal_weights)

    cfg = LabConfig(min_history_bars=50)
    ic = EnsembleICSignal(cfg, consensus_threshold=0.4)
    base = EnsembleSignal(cfg, min_agree=2)
    ic_t = ic.to_targets(_plans(ohlcv), ohlcv)
    base_t = base.to_targets(_plans(ohlcv), ohlcv)
    pd.testing.assert_frame_equal(ic_t.entries, base_t.entries)


def test_higher_threshold_is_subset_of_lower(monkeypatch):
    ohlcv = _ohlcv()

    def _equal_weights(raw_by_voter, close, **kwargs):
        eq = 1.0 / len(raw_by_voter)
        return pd.DataFrame(eq, index=close.index, columns=list(raw_by_voter))

    monkeypatch.setattr(eic_mod, "ic_weight_schedule", _equal_weights)
    cfg = LabConfig(min_history_bars=50)
    low = EnsembleICSignal(cfg, consensus_threshold=0.4).to_targets(_plans(ohlcv), ohlcv)
    high = EnsembleICSignal(cfg, consensus_threshold=0.8).to_targets(_plans(ohlcv), ohlcv)
    # every high-threshold entry is also a low-threshold entry
    assert (high.entries & ~low.entries).sum().sum() == 0


def test_exits_match_baseline():
    ohlcv = _ohlcv()
    cfg = LabConfig(min_history_bars=50)
    ic = EnsembleICSignal(cfg).to_targets(_plans(ohlcv), ohlcv)
    base = EnsembleSignal(cfg, min_agree=2).to_targets(_plans(ohlcv), ohlcv)
    pd.testing.assert_frame_equal(ic.exits, base.exits)


def test_to_targets_truncation_invariance():
    """Entries up to date d unchanged when post-d rows are removed (leak test)."""
    ohlcv = _ohlcv()
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleICSignal(cfg, ic_lookback_months=6)
    full = strat.to_targets(_plans(ohlcv), ohlcv)
    d = ohlcv.index[300]
    trunc = strat.to_targets(_plans(ohlcv.loc[:d]), ohlcv.loc[:d])
    pd.testing.assert_frame_equal(full.entries.loc[:d], trunc.entries.loc[:d])


def test_ensemble_ic_registered():
    assert "ensemble_ic" in SIGNAL_STRATEGY_NAMES
    assert "ensemble_ic" in STRATEGY_REGISTRY
    cfg = LabConfig(min_history_bars=50)
    strat = build_signal_strategy("ensemble_ic", cfg)
    assert strat.name == "ensemble_ic"
    assert strat.target_kind == "signals"
