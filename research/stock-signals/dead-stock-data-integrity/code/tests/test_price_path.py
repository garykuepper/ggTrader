"""Tests for descriptive last-month price-path labels of dead stocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import price_path
import pytest
from price_path import classify_path

pytestmark = pytest.mark.study


def test_flat_prices_are_calm_like_deal() -> None:
    close = pd.Series([50.0, 50.01] * 15)
    out = classify_path(close)
    assert out["label"] == "calm_like_deal"
    assert out["ratio"] == pytest.approx(50.01 / 50.0)
    assert out["daily_std"] <= price_path.CALM_STD


def test_halving_is_collapse_like() -> None:
    close = pd.Series([100.0] * 10 + list(np.linspace(100, 40, 22)))
    out = classify_path(close)
    assert out["label"] == "collapse_like"
    assert out["ratio"] == pytest.approx(0.4)


def test_volatile_flat_ratio_is_unclear() -> None:
    # WINDOW=21 is odd, so a 2-step alternating series needs an extra 100.0 at the end
    # for the last close and the close 21 valid closes earlier to both be 100.0 (ratio=1.0).
    close = pd.Series([100.0, 112.0] * 15 + [100.0, 100.0])
    out = classify_path(close)
    assert out["ratio"] == pytest.approx(1.0)
    assert out["daily_std"] > price_path.CALM_STD
    assert out["label"] == "unclear"


def test_brief_alternating_100_112_series_is_drifting_down() -> None:
    # The brief's 31-bar fixture pairs close[-1]=100.0 against close[-22]=112.0
    # (ratio=100/112=0.8929 < DRIFT_BELOW=0.9), so the rules classify it as drifting_down.
    close = pd.Series([100.0, 112.0] * 15 + [100.0])
    out = classify_path(close)
    assert out["ratio"] == pytest.approx(100.0 / 112.0)
    assert out["daily_std"] > price_path.CALM_STD
    assert out["label"] == "drifting_down"


def test_short_series_is_too_short() -> None:
    out = classify_path(pd.Series([10.0] * 21))
    assert out == {"ratio": None, "daily_std": None, "label": "too_short"}


def test_nan_closes_are_dropped_before_counting() -> None:
    close = pd.Series([50.0] * 5 + [np.nan] * 3 + [50.0] * 20)
    out = classify_path(close)
    assert out["label"] == "calm_like_deal"
    assert out["ratio"] == pytest.approx(1.0)
    assert out["daily_std"] == pytest.approx(0.0)


def test_drifting_down_and_above_calm_band_labels() -> None:
    drifting = pd.Series(np.linspace(100.0, 75.0, 22))
    assert classify_path(drifting)["label"] == "drifting_down"

    # Steady 20% climb has low daily_std but sits above CALM_BAND=(0.9, 1.1) -> unclear.
    rising = pd.Series(np.linspace(100.0, 120.0, 22))
    out = classify_path(rising)
    assert out["ratio"] == pytest.approx(1.2)
    assert out["daily_std"] <= price_path.CALM_STD
    assert out["label"] == "unclear"


def test_constants_and_zero_baseline_guard() -> None:
    assert price_path.WINDOW == 21
    assert price_path.CALM_STD == 0.01
    assert price_path.COLLAPSE_BELOW == 0.5
    assert price_path.DRIFT_BELOW == 0.9
    assert price_path.CALM_BAND == (0.9, 1.1)

    zero_start = pd.Series([0.0] + [10.0] * 21)
    assert classify_path(zero_start) == {"ratio": None, "daily_std": None, "label": "unclear"}
