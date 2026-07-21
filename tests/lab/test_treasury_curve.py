"""Tests for the treasury_curve strategy (candidate A5: Treasury term-
structure factors, ETF-approximation version -- an explicitly-labeled
3-ETF approximation of Filipović/Pelger/Ye's 4-factor investable term-
structure model, not a replication)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.strategies.treasury_curve import (
    TREASURY_CURVE_UNIVERSE,
    TreasuryCurveStrategy,
    curve_regime_weights,
    curve_slope_zscore,
)

from ggTrader.lab.strategy import LabConfig


class TestCurveSlopeZscore:
    def test_zero_at_the_series_mean(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        slope = pd.Series(np.concatenate([np.full(60, 1.0), np.full(40, 1.0)]), index=idx)
        z = curve_slope_zscore(slope, zscore_window=100)
        assert z == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_slope_is_currently_steep_vs_history(self):
        idx = pd.date_range("2020-01-01", periods=260, freq="D")
        slope = pd.Series(np.concatenate([np.full(250, 0.5), np.full(10, 3.0)]), index=idx)
        z = curve_slope_zscore(slope, zscore_window=260)
        assert z > 1.0

    def test_negative_when_slope_is_currently_flat_vs_history(self):
        idx = pd.date_range("2020-01-01", periods=260, freq="D")
        slope = pd.Series(np.concatenate([np.full(250, 2.0), np.full(10, -0.5)]), index=idx)
        z = curve_slope_zscore(slope, zscore_window=260)
        assert z < -1.0

    def test_insufficient_history_returns_zero(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        slope = pd.Series(np.full(10, 1.0), index=idx)
        z = curve_slope_zscore(slope, zscore_window=260)
        assert z == 0.0


class TestCurveRegimeWeights:
    def test_steep_curve_goes_all_in_long_duration(self):
        w = curve_regime_weights(z=2.0, steep_threshold=1.0, flat_threshold=-1.0)
        assert w == {"TLT": 1.0, "IEF": 0.0, "SHY": 0.0}

    def test_flat_curve_goes_all_in_short_duration(self):
        w = curve_regime_weights(z=-2.0, steep_threshold=1.0, flat_threshold=-1.0)
        assert w == {"TLT": 0.0, "IEF": 0.0, "SHY": 1.0}

    def test_neutral_curve_goes_all_in_belly(self):
        w = curve_regime_weights(z=0.0, steep_threshold=1.0, flat_threshold=-1.0)
        assert w == {"TLT": 0.0, "IEF": 1.0, "SHY": 0.0}


def _make_ohlcv(tickers: list[str], n: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    frames = {}
    for t in tickers:
        px = pd.Series(100.0, index=idx)
        frames[(t, "close")] = px
        frames[(t, "open")] = px
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestTreasuryCurveSelect:
    def test_returns_single_full_notional_pick_from_fred_loader(self):
        data = _make_ohlcv(TREASURY_CURVE_UNIVERSE, n=400)
        asof = data.index[-1]

        def fake_fred_loader(series_id: str, asof: pd.Timestamp) -> pd.Series:
            idx = pd.date_range("2015-01-01", periods=2500, freq="D")
            if series_id == "DGS10":
                return pd.Series(np.concatenate([np.full(2490, 2.5), np.full(10, 4.0)]), index=idx)
            return pd.Series(np.full(2500, 1.0), index=idx)  # DGS2 flat -> slope spikes steep

        strat = TreasuryCurveStrategy(LabConfig(min_history_bars=20), _fred_loader=fake_fred_loader)
        plan = strat.select(asof, data, eligible=TREASURY_CURVE_UNIVERSE)
        assert plan == [{"symbol": "TLT", "weight": 1.0}]

    def test_empty_plan_when_insufficient_price_history(self):
        data = _make_ohlcv(TREASURY_CURVE_UNIVERSE, n=10)
        strat = TreasuryCurveStrategy(LabConfig(min_history_bars=400))
        plan = strat.select(data.index[-1], data, eligible=TREASURY_CURVE_UNIVERSE)
        assert plan == []

    def test_sweep_params_present(self):
        params = TreasuryCurveStrategy.sweep_params()
        assert "steep_threshold" in params and "flat_threshold" in params
