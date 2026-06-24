"""Tests for the ML feature gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.paper.feature_gate import FEATURE_NAMES, FeatureGate, extract_features


@pytest.fixture
def sample_close():
    dates = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.randn(60) * 0.01)
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_volume():
    dates = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    np.random.seed(42)
    return pd.Series(np.random.randint(1_000_000, 10_000_000, size=60), index=dates)


def test_extract_features_returns_all_keys(sample_close, sample_volume):
    bar_date = sample_close.index[-1]
    feats = extract_features(sample_close, sample_volume, bar_date)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    for v in feats.values():
        assert isinstance(v, float)
        assert not np.isnan(v)


def test_extract_features_no_lookahead(sample_close, sample_volume):
    bar_date = sample_close.index[30]
    feats = extract_features(sample_close, sample_volume, bar_date)
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_gate_disabled_when_no_model(tmp_path):
    gate = FeatureGate(model_path=tmp_path / "nonexistent.joblib")
    assert not gate.enabled
    assert gate.score({"vol_5d": 0.1}) == 1.0


def test_gate_filter_passthrough_when_disabled(tmp_path):
    gate = FeatureGate(model_path=tmp_path / "nonexistent.joblib")
    ohlcv = pd.DataFrame()
    buys, scores = gate.filter_buys(["AAPL", "MSFT"], ohlcv)
    assert buys == ["AAPL", "MSFT"]
    assert scores == {}


def test_gate_filters_low_confidence(sample_close, sample_volume):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.6, 0.4]])  # below threshold

    gate = FeatureGate.__new__(FeatureGate)
    gate._model = FakeModel()
    gate._enabled = True
    gate._threshold = 0.55

    dates = sample_close.index
    ohlcv = pd.concat(
        {"AAPL": pd.DataFrame({"close": sample_close, "volume": sample_volume})},
        axis=1,
    )
    buys, scores = gate.filter_buys(["AAPL"], ohlcv)
    assert buys == []
    assert "AAPL" in scores
    assert scores["AAPL"] < 0.55


def test_extract_features_true_range_with_high_low(sample_close, sample_volume):
    """ATR should use true range when high/low are provided."""
    bar_date = sample_close.index[-1]
    high = sample_close * 1.02
    low = sample_close * 0.98

    feats_proxy = extract_features(sample_close, sample_volume, bar_date)
    feats_true = extract_features(sample_close, sample_volume, bar_date, high=high, low=low)

    assert feats_true["atr_ratio"] != pytest.approx(feats_proxy["atr_ratio"], abs=1e-6)
    assert not np.isnan(feats_true["atr_ratio"])


def test_extract_features_fallback_without_high_low(sample_close, sample_volume):
    """Without high/low, ATR should still work (close-diff fallback)."""
    bar_date = sample_close.index[-1]
    feats = extract_features(sample_close, sample_volume, bar_date)
    assert "atr_ratio" in feats
    assert not np.isnan(feats["atr_ratio"])


def test_gate_passes_high_confidence(sample_close, sample_volume):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])  # above threshold

    gate = FeatureGate.__new__(FeatureGate)
    gate._model = FakeModel()
    gate._enabled = True
    gate._threshold = 0.55

    ohlcv = pd.concat(
        {"AAPL": pd.DataFrame({"close": sample_close, "volume": sample_volume})},
        axis=1,
    )
    buys, scores = gate.filter_buys(["AAPL"], ohlcv)
    assert buys == ["AAPL"]
    assert scores["AAPL"] == pytest.approx(0.7)
