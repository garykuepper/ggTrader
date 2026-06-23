"""Tests for conviction strength indicator functions."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import bb_strength, ema_strength, rsi_strength


def _close(n=200, n_syms=2, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    data = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, (n, n_syms)), axis=0))
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(n_syms)])


class TestBBStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = bb_strength(close, period=20, std=2.0)
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)

    def test_values_in_zero_one_range(self):
        close = _close()
        result = bb_strength(close, period=20, std=2.0)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_nan_during_warmup(self):
        close = _close(n=50)
        result = bb_strength(close, period=20, std=2.0)
        assert result.iloc[:19].isna().all().all()

    def test_zero_when_price_at_lower_band(self):
        """Price exactly at lower band -> strength ~ 0."""
        idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
        close = pd.DataFrame({"A": np.full(100, 100.0)}, index=idx)
        # Flat price => std=0 => band_width=0 => should return 0 (not inf)
        result = bb_strength(close, period=20, std=2.0)
        valid = result.dropna()
        assert (valid == 0.0).all().all()


class TestRSIStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = rsi_strength(close, period=14, oversold=30)
        assert result.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close()
        result = rsi_strength(close, period=14, oversold=30)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_zero_when_rsi_above_oversold(self):
        """RSI above oversold threshold -> strength = 0."""
        close = _close(seed=99)
        result = rsi_strength(close, period=14, oversold=30)
        # Compute RSI to verify
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        above_mask = rsi >= 30
        strength_above = result[above_mask].dropna()
        if len(strength_above) > 0:
            assert (strength_above == 0.0).all().all()


class TestEMAStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        assert result.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_zero_when_fast_below_slow(self):
        """Fast EMA below slow -> strength = 0."""
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        ema_f = close.ewm(span=20, adjust=False).mean()
        ema_s = close.ewm(span=50, adjust=False).mean()
        below_mask = ema_f < ema_s
        strength_below = result[below_mask].dropna()
        if len(strength_below) > 0:
            assert (strength_below == 0.0).all().all()
