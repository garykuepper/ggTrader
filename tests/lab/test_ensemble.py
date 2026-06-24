"""Tests for the EnsembleSignal majority-vote strategy."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=300, n_syms=3, seed=42):
    """Synthetic OHLCV with (symbol, field) MultiIndex columns."""
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[sym] = pd.DataFrame(
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


def test_ensemble_no_entry_when_fewer_than_min_agree():
    """If only 1 sub-signal fires, ensemble should NOT enter."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=3)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # With min_agree=3 (all must agree), entries should be very rare or zero
    # on random synthetic data where signals are uncorrelated
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.01


def test_ensemble_enters_when_all_agree():
    """With min_agree=1, any single signal triggers entry."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=1)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # min_agree=1 means any sub-signal fires -> should have more entries
    assert targets.entries.sum().sum() > 0


def test_ensemble_sweep_params_returns_dict():
    assert "min_agree" in EnsembleSignal.sweep_params()
    assert "bb_period" in EnsembleSignal.sweep_params()


def test_ensemble_sweep_signals_keys_match_combos():
    from ggTrader.lab.sweep import combo_name

    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    ohlcv = _ohlcv(n=200)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    combos = [
        {
            "min_agree": 2,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "ema_fast": 20,
            "ema_slow": 50,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "divergence_window": 20,
            "vol_period": 20,
            "vol_mult": 2.0,
            "weekly_rsi_period": 14,
            "weekly_rsi_oversold": 30,
            "weekly_rsi_exit": 50,
        }
    ]
    result = strat.sweep_signals(combos, symbols, ohlcv)
    expected_key = combo_name("ensemble", combos[0])
    assert expected_key in result
    assert hasattr(result[expected_key], "entries")
    assert hasattr(result[expected_key], "exits")


def test_ensemble_select_respects_asof():
    """select() must not look at data past asof."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    mid = ohlcv.index[150]
    plan_full = strat.select(mid, ohlcv.loc[:mid], symbols)
    plan_trunc = strat.select(mid, ohlcv, symbols)
    assert len(plan_full) == len(plan_trunc)


def test_ensemble_min_agree_2_fewer_entries_than_1():
    """Higher min_agree -> fewer entries (more filtering)."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=123)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)

    strat1 = EnsembleSignal(cfg, min_agree=1)
    strat2 = EnsembleSignal(cfg, min_agree=2)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    t1 = strat1.to_targets(plans, ohlcv)
    t2 = strat2.to_targets(plans, ohlcv)
    assert t1.entries.sum().sum() >= t2.entries.sum().sum()


def test_ensemble_target_kind():
    """Ensemble must declare target_kind='signals'."""
    assert EnsembleSignal.target_kind == "signals"
    assert EnsembleSignal.name == "ensemble"


def test_ensemble_sweep_params_includes_new_signals():
    params = EnsembleSignal.sweep_params()
    assert "macd_fast" in params
    assert "vol_mult" in params
    assert "weekly_rsi_period" in params
    assert 4 in params["min_agree"]


def test_ensemble_6_voter_min_agree_4():
    """With min_agree=4 (majority of 6), entries should be very rare."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=4)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.005


def test_asymmetric_exit_more_exits_than_symmetric():
    """min_agree_exit=1 should produce >= exits than min_agree_exit=min_agree."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=99)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    sym = EnsembleSignal(cfg, min_agree=2, min_agree_exit=2)
    asym = EnsembleSignal(cfg, min_agree=2, min_agree_exit=1)

    t_sym = sym.to_targets(plans, ohlcv)
    t_asym = asym.to_targets(plans, ohlcv)

    assert t_asym.exits.sum().sum() >= t_sym.exits.sum().sum()


def test_min_agree_exit_defaults_to_min_agree():
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=3)
    assert strat.min_agree_exit == 3


def test_sweep_params_includes_min_agree_exit():
    params = EnsembleSignal.sweep_params()
    assert "min_agree_exit" in params
    assert 1 in params["min_agree_exit"]
