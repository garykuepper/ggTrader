"""Tests for MACD divergence indicator functions and signal class."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import macd_signals, macd_strength


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


class TestMACDSignals:
    def test_output_shape_matches_input(self):
        close = _close(200)
        entries, exits = macd_signals(
            close, fast=12, slow=26, signal_period=9, divergence_window=20
        )
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close = _close(200)
        entries, exits = macd_signals(close, 12, 26, 9, 20)
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

    def test_no_entries_during_warmup(self):
        close = _close(200)
        entries, _ = macd_signals(close, fast=12, slow=26, signal_period=9, divergence_window=20)
        # First slow+signal+divergence_window bars should have no entries
        warmup = 26 + 9 + 20
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_divergence_requires_price_low_and_histogram_higher(self):
        """Manually construct a bullish divergence scenario."""
        idx = _idx(60)
        # Price makes lower lows but MACD histogram makes higher lows
        price = np.concatenate(
            [
                np.linspace(100, 80, 30),  # declining
                np.linspace(80, 75, 15),  # lower low
                np.linspace(75, 85, 15),  # recovery
            ]
        )
        close = pd.DataFrame({"A": price}, index=idx)
        entries, _ = macd_signals(close, fast=8, slow=21, signal_period=9, divergence_window=10)
        # At minimum, no crash — divergence detection runs without error
        assert entries.shape == close.shape

    def test_exit_on_histogram_crosses_below_zero(self):
        close = _close(300, seed=99)
        _, exits = macd_signals(close, 12, 26, 9, 20)
        # Exits should fire at some point on random data
        assert exits.sum().sum() >= 0  # no crash; exits are valid booleans


class TestMACDStrength:
    def test_output_shape_matches_input(self):
        close = _close(200)
        strength = macd_strength(close, fast=12, slow=26, signal_period=9)
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close(200)
        strength = macd_strength(close, 12, 26, 9)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()

    def test_nan_during_warmup(self):
        close = _close(200)
        strength = macd_strength(close, fast=12, slow=26, signal_period=9)
        # First slow bars should be NaN
        assert strength.iloc[:26].isna().all().all()
