"""Headline/LLM sentiment: candidate A8's data layer. Real, point-in-time
historical news headlines via Alpaca's /news API (already integrated in
this project for paper trading, no new subscription), scored via a
cheap/free LLM through the local LiteLLM proxy -- the same mechanism
Lopez-Lira & Tang's "Can ChatGPT Forecast Stock Price Movements?" tests,
not a lexicon/keyword proxy.

Point-in-time note: unlike SEC/FINRA filings (which have a real reporting
lag between the event and its public disclosure), a news headline's
`created_at` timestamp already IS the moment it became publicly knowable
-- no additional publish lag is structurally required. PUBLISH_LAG_DAYS
defaults to 0 for that reason, kept as a parameter for symmetry with this
project's other data modules and in case Alpaca's own feed has any
backfill latency worth guarding against.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.persist import get_engine

PUBLISH_LAG_DAYS = 0

LiteLlmCall = Callable[[str], str]
#: (symbols, start, end, page_token) -> (page of raw news dicts, next_page_token)
NewsFetch = Callable[[List[str], str, str, Optional[str]], Tuple[List[dict], Optional[str]]]


def _default_news_fetch(
    symbols: List[str], start: str, end: str, page_token: Optional[str]
) -> Tuple[List[dict], Optional[str]]:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    from ggTrader.utils.config import _load_env

    _load_env()
    client = NewsClient(
        api_key=os.environ["APCA_API_KEY_ID"], secret_key=os.environ["APCA_API_SECRET_KEY"]
    )
    req = NewsRequest(
        symbols=",".join(symbols), start=start, end=end, limit=50, page_token=page_token
    )
    resp = client.get_news(req)
    items = [
        {
            "id": item.id,
            "headline": item.headline,
            "created_at": item.created_at,
            "symbols": item.symbols,
        }
        for item in resp.data["news"]
    ]
    return items, resp.data.get("next_page_token")


def fetch_news(
    symbols: List[str], start: str, end: str, news_fetch: NewsFetch | None = None
) -> pd.DataFrame:
    """Fetch every news article touching ``symbols`` in [start, end],
    expanded to one row per (article, symbol) -- a single article can tag
    several tickers (e.g. "Apple and Microsoft partner..."), and each
    tagged symbol independently gets that headline as a same-day signal
    input."""
    fetch = news_fetch or _default_news_fetch
    rows: list[dict] = []
    page_token: Optional[str] = None
    while True:
        page, page_token = fetch(symbols, start, end, page_token)
        for item in page:
            for sym in item["symbols"]:
                if sym in symbols:
                    rows.append(
                        {
                            "news_id": item["id"],
                            "symbol": sym,
                            "headline": item["headline"],
                            "created_at": item["created_at"],
                        }
                    )
        if not page_token:
            break
    if not rows:
        return pd.DataFrame(columns=["news_id", "symbol", "headline", "created_at"])
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


_SENTIMENT_PROMPT = (
    "Classify the likely near-term stock-price sentiment of this financial "
    "news headline. Respond with exactly one number and nothing else: "
    "-1 (bearish), 0 (neutral), or 1 (bullish).\n\nHeadline: {headline}"
)

_NUMBER_PATTERN = re.compile(r"-?1(?:\.0)?|0(?:\.0)?")


def parse_sentiment_response(response: str) -> float:
    """Extract the first -1/0/1 token from an LLM's raw text response.
    Defaults to neutral (0.0) if nothing parseable is found -- a
    fail-safe, not a crash, since sentiment scoring runs over thousands
    of headlines and a handful of unparseable replies shouldn't halt a
    backfill."""
    match = _NUMBER_PATTERN.search(response)
    if match is None:
        return 0.0
    try:
        return float(match.group())
    except ValueError:
        return 0.0


def _default_llm_call(prompt: str) -> str:
    import requests

    resp = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
        json={
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def score_headline(headline: str, llm_call: LiteLlmCall | None = None) -> float:
    """Score one headline's sentiment. A failed LLM call (rate limit,
    timeout, malformed response) defaults to neutral rather than raising
    -- one bad call must not kill a multi-thousand-headline backfill."""
    call = llm_call or _default_llm_call
    try:
        response = call(_SENTIMENT_PROMPT.format(headline=headline))
    except Exception:  # noqa: BLE001 -- fail-safe scoring, see docstring
        return 0.0
    return parse_sentiment_response(response)


def available_as_of(
    df: pd.DataFrame, asof: pd.Timestamp, lag_days: int = PUBLISH_LAG_DAYS
) -> pd.DataFrame:
    """Filter to headlines knowable by ``asof`` given the publish lag
    (0 by default -- see module docstring). Normalizes both sides to
    tz-naive and returns a tz-naive "created_at" column, matching this
    project's other data modules' available_as_of convention."""
    asof_naive = (
        pd.Timestamp(asof).tz_localize(None)
        if pd.Timestamp(asof).tz is not None
        else pd.Timestamp(asof)
    )
    out = df.copy()
    created_naive = (
        out["created_at"].dt.tz_localize(None)
        if out["created_at"].dt.tz is not None
        else out["created_at"]
    )
    out["created_at"] = created_naive
    mask = created_naive + pd.Timedelta(days=lag_days) <= asof_naive
    return out[mask].reset_index(drop=True)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_headlines (
    news_id bigint NOT NULL,
    symbol text NOT NULL,
    headline text NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (news_id, symbol)
)
"""

_SENTIMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS headline_sentiment_scores (
    news_id bigint NOT NULL,
    symbol text NOT NULL,
    score double precision NOT NULL,
    PRIMARY KEY (news_id, symbol)
)
"""


def ensure_schema() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(_SCHEMA))
        conn.execute(text(_SENTIMENT_SCHEMA))


def cache_news(news_df: pd.DataFrame) -> int:
    """Upsert a (news_id, symbol, headline, created_at) frame into the DB."""
    if news_df.empty:
        return 0
    ensure_schema()
    rows = [
        {
            "news_id": int(rec["news_id"]),
            "symbol": rec["symbol"],
            "headline": rec["headline"],
            "created_at": rec["created_at"],
        }
        for rec in news_df.to_dict("records")
    ]
    upsert_sql = text(
        """
        INSERT INTO news_headlines (news_id, symbol, headline, created_at)
        VALUES (:news_id, :symbol, :headline, :created_at)
        ON CONFLICT (news_id, symbol) DO UPDATE SET
            headline = EXCLUDED.headline, created_at = EXCLUDED.created_at
        """
    )
    with get_engine().begin() as conn:
        conn.execute(upsert_sql, rows)
    return len(rows)


def load_news(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Load cached headlines for ``symbols`` in [start, end]."""
    query = text(
        """
        SELECT news_id, symbol, headline, created_at FROM news_headlines
        WHERE symbol = ANY(:symbols) AND created_at >= :start AND created_at <= :end
        ORDER BY created_at
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query, {"symbols": symbols, "start": start, "end": end}).fetchall()
    df = pd.DataFrame(rows, columns=["news_id", "symbol", "headline", "created_at"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def cache_sentiment_scores(scores_df: pd.DataFrame) -> int:
    """Upsert a (news_id, symbol, score) frame into the DB."""
    if scores_df.empty:
        return 0
    ensure_schema()
    rows = [
        {"news_id": int(rec["news_id"]), "symbol": rec["symbol"], "score": float(rec["score"])}
        for rec in scores_df.to_dict("records")
    ]
    upsert_sql = text(
        """
        INSERT INTO headline_sentiment_scores (news_id, symbol, score)
        VALUES (:news_id, :symbol, :score)
        ON CONFLICT (news_id, symbol) DO UPDATE SET score = EXCLUDED.score
        """
    )
    with get_engine().begin() as conn:
        conn.execute(upsert_sql, rows)
    return len(rows)


def load_sentiment_scores(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Load cached sentiment scores, joined to their headline's created_at,
    for ``symbols`` in [start, end]."""
    query = text(
        """
        SELECT s.news_id, s.symbol, s.score, h.created_at FROM headline_sentiment_scores s
        JOIN news_headlines h ON h.news_id = s.news_id AND h.symbol = s.symbol
        WHERE s.symbol = ANY(:symbols) AND h.created_at >= :start AND h.created_at <= :end
        ORDER BY h.created_at
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query, {"symbols": symbols, "start": start, "end": end}).fetchall()
    df = pd.DataFrame(rows, columns=["news_id", "symbol", "score", "created_at"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df
