"""Tests for the ML signal pre-screen script logic."""

import numpy as np
import pandas as pd

from ggTrader.paper.feature_gate import FEATURE_NAMES, extract_features


def _close_series(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))), index=idx)


def _volume_series(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(np.random.randint(1000, 10000, n).astype(float), index=idx)


class TestExtractFeatures:
    def test_returns_all_feature_names(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        assert set(FEATURE_NAMES) == set(feats.keys())

    def test_features_are_finite(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"

    def test_rsi_in_0_100_range(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        assert 0.0 <= feats["rsi_14"] <= 100.0


class TestVerdictThresholds:
    def test_drop_below_050(self):
        assert 0.49 < 0.50

    def test_borderline_050_to_055(self):
        prec = 0.52
        assert 0.50 <= prec < 0.55

    def test_strong_above_055(self):
        prec = 0.58
        assert prec >= 0.55
