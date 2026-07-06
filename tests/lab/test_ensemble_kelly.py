"""Tests for EnsembleKellySignal — Kelly-criterion-sized ensemble."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.ensemble_kelly import EnsembleKellySignal
from ggTrader.lab.strategy import LabConfig, SignalTargets


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=300, n_syms=3, seed=42):
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


def _plans(ohlcv):
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    return {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}


class TestEnsembleKellySignal:
    def test_returns_signal_targets_with_sizes(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        assert isinstance(targets, SignalTargets)
        assert targets.sizes is not None
        assert targets.sizes.shape == targets.entries.shape

    def test_entries_exits_match_plain_ensemble(self):
        """Entry/exit logic must be identical to EnsembleSignal — only sizing differs."""
        cfg = LabConfig(min_history_bars=50)
        ohlcv = _ohlcv(n=300, seed=77)

        plain = EnsembleSignal(cfg, min_agree=2)
        kelly = EnsembleKellySignal(cfg, min_agree=2)

        t_plain = plain.to_targets(_plans(ohlcv), ohlcv)
        t_kelly = kelly.to_targets(_plans(ohlcv), ohlcv)

        pd.testing.assert_frame_equal(t_plain.entries, t_kelly.entries)
        pd.testing.assert_frame_equal(t_plain.exits, t_kelly.exits)

    def test_sizes_bounded_by_max_size(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(
            cfg, min_agree=1, kelly_multiplier=1.0, base_size=0.03, max_size=0.05
        )
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        valid = targets.sizes[targets.entries].dropna()
        if len(valid) > 0:
            assert valid.max() <= 0.05 + 1e-10

    def test_sizes_nan_where_no_entry(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        no_entry = ~targets.entries
        assert targets.sizes[no_entry].isna().all().all()

    def test_sweep_params_is_kelly_multiplier_only(self):
        params = EnsembleKellySignal.sweep_params()
        assert params == {"kelly_multiplier": [0.25, 0.5, 1.0]}

    def test_name_and_target_kind(self):
        assert EnsembleKellySignal.name == "ensemble_kelly"
        assert EnsembleKellySignal.target_kind == "signals"

    def test_select_delegates_to_eligible(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plan = strat.select(ohlcv.index[200], ohlcv, symbols)
        assert len(plan) == len(symbols)

    def test_sweep_signals_returns_sizes_for_each_multiplier(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg, min_agree=1)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        combos = [{"kelly_multiplier": 0.25}, {"kelly_multiplier": 0.5}, {"kelly_multiplier": 1.0}]
        result = strat.sweep_signals(combos, symbols, ohlcv)
        assert len(result) == 3
        for targets in result.values():
            assert targets.sizes is not None


def test_registered_in_strategy_registry():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "ensemble_kelly" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["ensemble_kelly"] is EnsembleKellySignal


def test_registered_in_signal_strategy_names():
    from ggTrader.lab.strategies.registry import signal_strategy_names

    assert "ensemble_kelly" in signal_strategy_names()


def test_build_signal_strategy():
    from ggTrader.lab.strategies.signals import build_signal_strategy

    strat = build_signal_strategy("ensemble_kelly", LabConfig())
    assert strat.name == "ensemble_kelly"


def test_cli_accepts_ensemble_kelly():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "ensemble_kelly"])
    assert args.strategy == "ensemble_kelly"
