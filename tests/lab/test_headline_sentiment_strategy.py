"""Tests for the headline_sentiment strategy (candidate A8: LLM-scored
news-headline sentiment, long the top cross-sectional quintile by
trailing average sentiment, rebalanced monthly)."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.strategies.headline_sentiment import HeadlineSentimentStrategy
from ggTrader.lab.strategy import LabConfig


def _make_ohlcv(symbols: list[str], n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    frames = {}
    for s in symbols:
        px = pd.Series(100.0, index=idx)
        frames[(s, "close")] = px
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestHeadlineSentimentSelect:
    def test_picks_top_quintile_by_trailing_mean_sentiment(self):
        symbols = [f"SYM{i}" for i in range(10)]
        data = _make_ohlcv(symbols, n=100)
        asof = data.index[-1]

        scores = pd.DataFrame(
            {
                "symbol": symbols,
                "score": [float(i) for i in range(10)],  # SYM9 highest, SYM0 lowest
                "created_at": [asof - pd.Timedelta(days=1)] * 10,
            }
        )

        def fake_loader(symbols_arg, start, end):
            return scores[scores["symbol"].isin(symbols_arg)]

        strat = HeadlineSentimentStrategy(
            LabConfig(min_history_bars=20), quintile=5, _sentiment_loader=fake_loader
        )
        plan = strat.select(asof, data, eligible=symbols)
        picked = {s["symbol"] for s in plan}
        assert picked == {"SYM9", "SYM8"}  # top quintile of 10 = top 2
        weights = [s["weight"] for s in plan]
        assert all(w == pytest.approx(0.5) for w in weights)

    def test_no_sentiment_data_returns_empty_plan(self):
        symbols = [f"SYM{i}" for i in range(10)]
        data = _make_ohlcv(symbols, n=100)
        strat = HeadlineSentimentStrategy(
            LabConfig(min_history_bars=20),
            _sentiment_loader=lambda *a: pd.DataFrame(columns=["symbol", "score", "created_at"]),
        )
        plan = strat.select(data.index[-1], data, eligible=symbols)
        assert plan == []

    def test_symbols_with_no_recent_headlines_are_excluded_not_zero_filled(self):
        symbols = [f"SYM{i}" for i in range(10)]
        data = _make_ohlcv(symbols, n=100)
        asof = data.index[-1]
        # Only 4 symbols have any sentiment data at all.
        scores = pd.DataFrame(
            {
                "symbol": ["SYM0", "SYM1", "SYM2", "SYM3"],
                "score": [1.0, 1.0, 1.0, 1.0],
                "created_at": [asof - pd.Timedelta(days=1)] * 4,
            }
        )
        strat = HeadlineSentimentStrategy(
            LabConfig(min_history_bars=20),
            quintile=5,
            _sentiment_loader=lambda *a: scores,
        )
        plan = strat.select(asof, data, eligible=symbols)
        picked = {s["symbol"] for s in plan}
        assert picked.issubset({"SYM0", "SYM1", "SYM2", "SYM3"})

    def test_sweep_params_present(self):
        params = HeadlineSentimentStrategy.sweep_params()
        assert "lookback_days" in params and "quintile" in params
