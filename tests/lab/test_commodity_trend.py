"""Tests for the commodity_trend strategy (candidate A3: cross-sectional
12-1 momentum across liquid single-commodity ETFs, with a volatility-
regime filter to avoid crash periods)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.strategies.commodity_trend import (
    COMMODITY_TREND_UNIVERSE,
    CommodityTrendStrategy,
    cross_sectional_momentum_scores,
    regime_vol_zscore,
)

from ggTrader.lab.strategy import LabConfig


class TestCrossSectionalMomentumScores:
    def test_ranks_by_trailing_return_excluding_skip_window(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        close = pd.DataFrame(
            {
                "A": np.linspace(100, 200, 300),  # strong uptrend
                "B": np.linspace(100, 90, 300),  # downtrend
            },
            index=idx,
        )
        scores = cross_sectional_momentum_scores(close, lookback=252, skip=21)
        assert scores["A"] > scores["B"]

    def test_returns_empty_when_insufficient_history(self):
        idx = pd.bdate_range("2020-01-01", periods=50)
        close = pd.DataFrame({"A": np.linspace(100, 110, 50)}, index=idx)
        scores = cross_sectional_momentum_scores(close, lookback=252, skip=21)
        assert scores.empty


class TestRegimeVolZscore:
    def test_elevated_recent_vol_gives_positive_zscore(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=300)
        calm = rng.normal(0, 0.005, 250)
        stressed = rng.normal(0, 0.05, 50)  # 10x vol in the recent window
        returns = pd.DataFrame(
            {"A": np.concatenate([calm, stressed]), "B": np.concatenate([calm, stressed])},
            index=idx,
        )
        z = regime_vol_zscore(returns, vol_lookback=20, zscore_window=200)
        assert z > 1.0

    def test_flat_vol_gives_zscore_near_zero(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=300)
        returns = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.01, 300)}, index=idx
        )
        z = regime_vol_zscore(returns, vol_lookback=20, zscore_window=200)
        assert abs(z) < 1.5


def _make_ohlcv(tickers: list[str], n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    frames = {}
    for i, t in enumerate(tickers):
        drift = 0.0002 * (i + 1)
        px = 100.0 * np.cumprod(1 + rng.normal(drift, 0.01, n))
        frames[(t, "close")] = pd.Series(px, index=idx)
        frames[(t, "open")] = frames[(t, "close")]
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestCommodityTrendSelect:
    def test_picks_top_n_equal_weighted_when_regime_is_calm(self):
        data = _make_ohlcv(COMMODITY_TREND_UNIVERSE, n=400)
        strat = CommodityTrendStrategy(
            LabConfig(min_history_bars=280), top_n=5, vol_z_threshold=10.0
        )
        plan = strat.select(data.index[-1], data, eligible=COMMODITY_TREND_UNIVERSE)
        assert len(plan) == 5
        weights = [s["weight"] for s in plan]
        assert all(w == pytest.approx(0.2) for w in weights)

    def test_goes_to_cash_when_regime_filter_trips(self):
        data = _make_ohlcv(COMMODITY_TREND_UNIVERSE, n=400)
        strat = CommodityTrendStrategy(
            LabConfig(min_history_bars=280), top_n=5, vol_z_threshold=-10.0
        )
        plan = strat.select(data.index[-1], data, eligible=COMMODITY_TREND_UNIVERSE)
        assert plan == []

    def test_sweep_params_present(self):
        params = CommodityTrendStrategy.sweep_params()
        assert "top_n" in params and "vol_z_threshold" in params
