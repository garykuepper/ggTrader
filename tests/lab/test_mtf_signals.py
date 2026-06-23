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
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

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
