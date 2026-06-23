"""Tests for volume-confirmed BB reversion indicator functions and signal class."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import (
    extract_volume,
    volume_bb_signals,
    volume_bb_strength,
)


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


def _close_and_volume(n=300, n_syms=3, seed=42):
    ohlcv = _ohlcv(n, n_syms, seed)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    vol = extract_volume(ohlcv, symbols)
    return close, vol


class TestExtractVolume:
    def test_shape_matches_close(self):
        ohlcv = _ohlcv(100)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        vol = extract_volume(ohlcv, symbols)
        assert vol.shape == (100, 3)
        assert list(vol.columns) == symbols

    def test_missing_symbol_skipped(self):
        ohlcv = _ohlcv(100)
        vol = extract_volume(ohlcv, ["S0", "NOSUCH"])
        assert "S0" in vol.columns
        assert "NOSUCH" not in vol.columns


class TestVolumeBBSignals:
    def test_output_shape_matches_input(self):
        close, vol = _close_and_volume(200)
        entries, exits = volume_bb_signals(
            close, vol, bb_period=20, bb_std=2.0, vol_period=20, vol_mult=1.5
        )
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close, vol = _close_and_volume(200)
        entries, exits = volume_bb_signals(close, vol, 20, 2.0, 20, 1.5)
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

    def test_higher_vol_mult_fewer_entries(self):
        """Stricter volume filter -> fewer entries."""
        close, vol = _close_and_volume(300, seed=123)
        ent_low, _ = volume_bb_signals(close, vol, 20, 2.0, 20, 1.0)
        ent_high, _ = volume_bb_signals(close, vol, 20, 2.0, 20, 3.0)
        assert ent_low.sum().sum() >= ent_high.sum().sum()

    def test_no_entries_during_warmup(self):
        close, vol = _close_and_volume(200)
        entries, _ = volume_bb_signals(
            close, vol, bb_period=20, bb_std=2.0, vol_period=20, vol_mult=2.0
        )
        warmup = max(20, 20)
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_exits_match_plain_bb_exits(self):
        """Exits should be identical to plain bb_reversion exits."""
        from ggTrader.lab.strategies.indicators import bb_signals

        close, vol = _close_and_volume(300)
        _, vol_exits = volume_bb_signals(close, vol, 20, 2.0, 20, 1.5)
        _, bb_exits = bb_signals(close, 20, 2.0)
        pd.testing.assert_frame_equal(vol_exits, bb_exits)


class TestVolumeBBStrength:
    def test_output_shape_matches_input(self):
        close, vol = _close_and_volume(200)
        strength = volume_bb_strength(close, vol, bb_period=20, bb_std=2.0, vol_period=20)
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close, vol = _close_and_volume(200)
        strength = volume_bb_strength(close, vol, 20, 2.0, 20)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()
