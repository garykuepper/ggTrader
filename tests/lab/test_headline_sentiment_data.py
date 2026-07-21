"""Tests for the headline-sentiment data module: real, point-in-time
historical news headlines (Alpaca's /news API, already integrated in this
project for paper trading) scored via a cheap/free LLM through the local
LiteLLM proxy."""

from __future__ import annotations

import pandas as pd
import pytest

from ggTrader.lab.headline_sentiment_data import (
    PUBLISH_LAG_DAYS,
    available_as_of,
    fetch_news,
    parse_sentiment_response,
    score_headline,
)


class TestFetchNews:
    def test_expands_multi_symbol_articles_into_one_row_per_symbol(self):
        page = [
            {
                "id": 1,
                "headline": "Apple and Microsoft partner on cloud deal",
                "created_at": "2020-01-01T10:00:00Z",
                "symbols": ["AAPL", "MSFT"],
            }
        ]

        def fake_fetch(symbols, start, end, page_token):
            return (page, None) if page_token is None else ([], None)

        df = fetch_news(["AAPL", "MSFT"], "2020-01-01", "2020-01-02", news_fetch=fake_fetch)
        assert set(df.columns) == {"news_id", "symbol", "headline", "created_at"}
        assert sorted(df["symbol"]) == ["AAPL", "MSFT"]
        assert (df["news_id"] == 1).all()

    def test_paginates_until_no_next_page_token(self):
        page1 = [
            {"id": 1, "headline": "H1", "created_at": "2020-01-01T10:00:00Z", "symbols": ["AAPL"]}
        ]
        page2 = [
            {"id": 2, "headline": "H2", "created_at": "2020-01-02T10:00:00Z", "symbols": ["AAPL"]}
        ]
        calls = []

        def fake_fetch(symbols, start, end, page_token):
            calls.append(page_token)
            if page_token is None:
                return page1, "tok2"
            if page_token == "tok2":
                return page2, None
            return [], None

        df = fetch_news(["AAPL"], "2020-01-01", "2020-01-03", news_fetch=fake_fetch)
        assert calls == [None, "tok2"]
        assert len(df) == 2

    def test_empty_result_returns_empty_frame_with_expected_columns(self):
        df = fetch_news(["AAPL"], "2020-01-01", "2020-01-02", news_fetch=lambda *a: ([], None))
        assert df.empty
        assert set(df.columns) == {"news_id", "symbol", "headline", "created_at"}


class TestParseSentimentResponse:
    def test_parses_bullish(self):
        assert parse_sentiment_response("1") == 1.0
        assert parse_sentiment_response("The sentiment is 1 (bullish).") == 1.0

    def test_parses_bearish(self):
        assert parse_sentiment_response("-1") == -1.0
        assert parse_sentiment_response("I'd say -1, this is bad news.") == -1.0

    def test_parses_neutral(self):
        assert parse_sentiment_response("0") == 0.0

    def test_unparseable_response_defaults_to_neutral(self):
        assert parse_sentiment_response("I cannot determine this.") == 0.0

    def test_first_number_wins_when_multiple_present(self):
        # Defensive: don't let a later stray digit (e.g. in a footnote) win.
        assert parse_sentiment_response("1 -- confidence about 80%") == 1.0


class TestScoreHeadline:
    def test_calls_llm_and_parses_result(self):
        calls = []

        def fake_llm_call(prompt: str) -> str:
            calls.append(prompt)
            return "1"

        score = score_headline("Apple beats earnings estimates", llm_call=fake_llm_call)
        assert score == 1.0
        assert "Apple beats earnings estimates" in calls[0]

    def test_llm_failure_defaults_to_neutral_not_a_crash(self):
        def failing_llm_call(prompt: str) -> str:
            raise RuntimeError("rate limited")

        score = score_headline("Some headline", llm_call=failing_llm_call)
        assert score == 0.0


class TestAvailableAsOf:
    def test_excludes_headlines_within_the_publish_lag(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "created_at": pd.to_datetime(["2026-06-01T10:00:00Z", "2026-06-02T10:00:00Z"]),
            }
        )
        asof = pd.Timestamp("2026-06-01T12:00:00Z")
        out = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
        assert list(out["symbol"]) == ["AAPL"]

    def test_headlines_are_available_same_day_zero_lag(self):
        df = pd.DataFrame(
            {"symbol": ["AAPL"], "created_at": pd.to_datetime(["2026-06-01T10:00:00Z"])}
        )
        asof = pd.Timestamp("2026-06-01T15:00:00Z")
        out = available_as_of(df, asof, lag_days=0)
        assert len(out) == 1

    def test_handles_tz_naive_and_aware_mix(self):
        df = pd.DataFrame(
            {"symbol": ["AAPL"], "created_at": [pd.Timestamp("2026-06-01T10:00:00", tz="UTC")]}
        )
        asof = pd.Timestamp("2026-06-01T15:00:00")  # naive
        out = available_as_of(df, asof, lag_days=0)
        assert len(out) == 1
        assert out["created_at"].dt.tz is None


@pytest.mark.integration
def test_cache_news_and_scores_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.headline_sentiment_data import (
        cache_news,
        cache_sentiment_scores,
        ensure_schema,
        load_news,
        load_sentiment_scores,
    )
    from ggTrader.lab.persist import get_engine

    ensure_schema()
    marker = "ZZTEST_NEWS"
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM news_headlines WHERE symbol = :s"), {"s": marker})
        conn.execute(text("DELETE FROM headline_sentiment_scores WHERE symbol = :s"), {"s": marker})

    news_df = pd.DataFrame(
        {
            "news_id": [999001],
            "symbol": [marker],
            "headline": ["Test headline for roundtrip"],
            "created_at": pd.to_datetime(["2026-06-01T10:00:00Z"]),
        }
    )
    n = cache_news(news_df)
    assert n == 1

    loaded = load_news([marker], "2026-01-01", "2026-12-31")
    assert len(loaded) == 1
    assert loaded.iloc[0]["headline"] == "Test headline for roundtrip"

    n2 = cache_sentiment_scores(
        pd.DataFrame({"news_id": [999001], "symbol": [marker], "score": [1.0]})
    )
    assert n2 == 1
    scores = load_sentiment_scores([marker], "2026-01-01", "2026-12-31")
    assert len(scores) == 1
    assert scores.iloc[0]["score"] == pytest.approx(1.0)

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM news_headlines WHERE symbol = :s"), {"s": marker})
        conn.execute(text("DELETE FROM headline_sentiment_scores WHERE symbol = :s"), {"s": marker})
