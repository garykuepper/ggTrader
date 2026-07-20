"""Tests for the fx_hedge_overlay strategy (candidate A1: dynamic FX hedge
overlay via carry + PPP-value + trend, retail proxy using hedged/unhedged
ETF pairs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.strategies.fx_hedge_overlay import (
    FX_HEDGE_PAIRS,
    FxHedgeOverlayStrategy,
    real_fx_index,
    rolling_zscore,
    trend_signal,
    unhedged_weight,
)

from ggTrader.lab.strategy import LabConfig


class TestRollingZscore:
    def test_zero_at_the_series_mean(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        z = rolling_zscore(s, window=10, min_periods=5)
        assert z.iloc[-1] == pytest.approx((s.iloc[-1] - s.iloc[-10:].mean()) / s.iloc[-10:].std())

    def test_nan_before_min_periods(self):
        s = pd.Series([1.0, 2.0, 3.0])
        z = rolling_zscore(s, window=10, min_periods=5)
        assert z.isna().all()


class TestRealFxIndex:
    def test_flat_cpi_leaves_spot_unchanged(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="MS")
        spot = pd.Series([1.10, 1.11, 1.09, 1.12, 1.10], index=idx)
        us_cpi = pd.Series([100.0] * 5, index=idx)
        foreign_cpi = pd.Series([100.0] * 5, index=idx)
        out = real_fx_index(spot, us_cpi, foreign_cpi)
        pd.testing.assert_series_equal(out, spot, check_names=False)

    def test_higher_relative_foreign_inflation_lowers_real_fx(self):
        """Foreign inflation running hotter than the US, at constant nominal
        spot, means the foreign currency is getting cheaper in real terms --
        real_fx should fall below the nominal spot rate."""
        idx = pd.date_range("2020-01-01", periods=2, freq="MS")
        spot = pd.Series([1.10, 1.10], index=idx)
        us_cpi = pd.Series([100.0, 100.0], index=idx)
        foreign_cpi = pd.Series([100.0, 110.0], index=idx)  # foreign inflation +10%
        out = real_fx_index(spot, us_cpi, foreign_cpi)
        assert out.iloc[-1] < out.iloc[0]


class TestTrendSignal:
    def test_positive_for_a_steadily_rising_ratio(self):
        ratio = pd.Series(np.linspace(1.0, 1.2, 300))
        assert trend_signal(ratio, lookback=252, skip=21) > 0

    def test_negative_for_a_steadily_falling_ratio(self):
        ratio = pd.Series(np.linspace(1.2, 1.0, 300))
        assert trend_signal(ratio, lookback=252, skip=21) < 0

    def test_nan_when_insufficient_history(self):
        ratio = pd.Series(np.linspace(1.0, 1.2, 50))
        assert pd.isna(trend_signal(ratio, lookback=252, skip=21))


class TestUnhedgedWeight:
    def test_neutral_score_is_half_weight(self):
        assert unhedged_weight(0.0) == pytest.approx(0.5)

    def test_monotonically_increasing_in_score(self):
        assert unhedged_weight(-2.0) < unhedged_weight(-1.0) < unhedged_weight(0.0)
        assert unhedged_weight(0.0) < unhedged_weight(1.0) < unhedged_weight(2.0)

    def test_bounded_between_zero_and_one(self):
        assert 0.0 < unhedged_weight(-10.0) < 0.01
        assert 0.99 < unhedged_weight(10.0) < 1.0


def _make_ohlcv(tickers: list[str], n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n, tz="UTC")
    frames = {}
    for t in tickers:
        base = 100.0 + rng.normal(0, 0.3, n).cumsum()
        frames[(t, "close")] = pd.Series(base, index=idx)
        frames[(t, "open")] = frames[(t, "close")]
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestFxHedgeOverlaySelect:
    def test_returns_two_symbols_per_pair_with_weights_summing_to_equal_share(self):
        tickers = [t for pair in FX_HEDGE_PAIRS for t in (pair.unhedged, pair.hedged)]
        data = _make_ohlcv(tickers, n=500)
        asof = data.index[-1]

        def fake_fred_loader(series_id: str, asof: pd.Timestamp) -> pd.Series:
            idx = pd.date_range("2018-01-01", periods=60, freq="MS")
            return pd.Series(1.0 + 0.001 * np.arange(60), index=idx)

        strat = FxHedgeOverlayStrategy(
            LabConfig(min_history_bars=20), _fred_loader=fake_fred_loader
        )
        plan = strat.select(asof, data, eligible=[])

        assert len(plan) == 2 * len(FX_HEDGE_PAIRS)
        per_pair_share = 1.0 / len(FX_HEDGE_PAIRS)
        for pair in FX_HEDGE_PAIRS:
            entries = {
                s["symbol"]: s["weight"]
                for s in plan
                if s["symbol"] in (pair.unhedged, pair.hedged)
            }
            assert set(entries) == {pair.unhedged, pair.hedged}
            assert sum(entries.values()) == pytest.approx(per_pair_share, abs=1e-6)

    def test_empty_plan_when_insufficient_price_history(self):
        tickers = [t for pair in FX_HEDGE_PAIRS for t in (pair.unhedged, pair.hedged)]
        data = _make_ohlcv(tickers, n=10)
        asof = data.index[-1]
        strat = FxHedgeOverlayStrategy(LabConfig(min_history_bars=400))
        plan = strat.select(asof, data, eligible=[])
        assert plan == []

    def test_sweep_params_present(self):
        params = FxHedgeOverlayStrategy.sweep_params()
        assert "k" in params
        assert isinstance(params["k"], list) and len(params["k"]) >= 2
