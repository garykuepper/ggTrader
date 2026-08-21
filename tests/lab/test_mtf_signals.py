# tests/lab/test_mtf_signals.py
"""Tests for multi-timeframe reversion indicator functions."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import mtf_signals, mtf_strength


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _close(n=300, n_syms=3, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        frames[sym] = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))),
            index=idx,
        )
    return pd.DataFrame(frames)


class TestMTFSignals:
    def test_output_shape_matches_input(self):
        close = _close(300)
        entries, exits = mtf_signals(
            close,
            weekly_rsi_period=14,
            weekly_rsi_oversold=30,
            weekly_rsi_exit=50,
            daily_bb_period=20,
            daily_bb_std=2.0,
        )
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close = _close(300)
        entries, exits = mtf_signals(close, 14, 30, 50, 20, 2.0)
        assert entries.dtypes.apply(lambda d: d == "bool").all()
        assert exits.dtypes.apply(lambda d: d == "bool").all()

    def test_no_entries_during_warmup(self):
        close = _close(300)
        entries, _ = mtf_signals(close, 14, 30, 50, 20, 2.0)
        warmup = 14 * 5 + 20
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_stricter_oversold_fewer_entries(self):
        close = _close(500, seed=123)
        ent_loose, _ = mtf_signals(close, 14, 40, 50, 20, 2.0)
        ent_strict, _ = mtf_signals(close, 14, 20, 50, 20, 2.0)
        assert ent_loose.sum().sum() >= ent_strict.sum().sum()

    def test_weekly_resampling_doesnt_crash_on_short_data(self):
        close = _close(30)
        entries, exits = mtf_signals(close, 7, 30, 50, 15, 2.0)
        assert entries.shape == close.shape


class TestMTFStrength:
    def test_output_shape_matches_input(self):
        close = _close(300)
        strength = mtf_strength(
            close,
            weekly_rsi_period=14,
            weekly_rsi_oversold=30,
            daily_bb_period=20,
            daily_bb_std=2.0,
        )
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close(300)
        strength = mtf_strength(close, 14, 30, 20, 2.0)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()


def test_mtf_registered():
    from ggTrader.lab.strategies.signals import _get_registry

    assert "mtf_reversion" in _get_registry()


def test_build_mtf():
    from ggTrader.lab.strategies.signals import build_signal_strategy
    from ggTrader.lab.strategy import LabConfig

    strat = build_signal_strategy("mtf_reversion", LabConfig())
    assert strat.name == "mtf_reversion"


def test_cli_accepts_mtf():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "mtf_reversion"])
    assert args.strategy == "mtf_reversion"


def test_mtf_to_targets():
    from ggTrader.lab.strategies.signals import MultiTimeframeReversionSignal
    from ggTrader.lab.strategy import LabConfig, SignalTargets

    cfg = LabConfig(min_history_bars=50)
    strat = MultiTimeframeReversionSignal(cfg)
    close_data = _close(300)
    idx = close_data.index
    frames = {}
    for sym in close_data.columns:
        frames[sym] = pd.DataFrame(
            {
                "open": close_data[sym] * 0.999,
                "high": close_data[sym] * 1.005,
                "low": close_data[sym] * 0.995,
                "close": close_data[sym],
                "volume": np.random.randint(1000, 10000, len(idx)).astype(float),
            },
            index=idx,
        )
    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns.names = ["symbol", "field"]
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert isinstance(targets, SignalTargets)
