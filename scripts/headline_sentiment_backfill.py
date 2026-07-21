"""Pilot backfill for candidate A8 (headline/LLM sentiment): a deliberately
bounded first pass -- 50 midcap400 names, 2 years -- before committing to
the full-universe, multi-year scope this project's other candidates
usually get. Real headline volume (confirmed via a live sample after
fixing Alpaca's broken pagination) is far higher than initially assumed
(~115 headlines/symbol/year for a representative midcap400 sample), and
each unique headline needs a real LLM call to score -- a materially
bigger time/cost commitment than any prior candidate's data-only fetch,
so this pilot exists to get a first real read before scaling up.

Resumable: skips headlines already cached, and skips scoring any news_id
that already has a cached score, so an interrupted run picks up cleanly.
Paced conservatively between LLM calls (this project's Google Trends
backfill hit a real rate-limit lockout from being too aggressive on a
first attempt -- same caution applies here even though this routes
through a local proxy, since the proxy's free-tier model still has its
own upstream rate limits).
"""

from __future__ import annotations

import time

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.headline_sentiment_data import (
    cache_news,
    cache_sentiment_scores,
    ensure_schema,
    fetch_news,
    score_headline,
)
from ggTrader.lab.persist import get_engine

PILOT_SYMBOL_COUNT = 50
PILOT_YEARS = 2
LLM_CALL_DELAY_SECONDS = 0.5


def _already_scored_ids() -> set[int]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT news_id FROM headline_sentiment_scores")).fetchall()
    return {r[0] for r in rows}


def main() -> None:
    ensure_schema()

    eval_end = pd.Timestamp.now(tz="UTC")
    eval_start = eval_end - pd.DateOffset(years=PILOT_YEARS)
    universe = equity_universe_between(eval_start, eval_end, universe="midcap400")
    symbols = sorted(universe)[:PILOT_SYMBOL_COUNT]
    print(
        f"Pilot: {len(symbols)} symbols, {eval_start.date()} .. {eval_end.date()}",
        flush=True,
    )

    print("Fetching news (paginated)...", flush=True)
    news_df = fetch_news(symbols, str(eval_start.date()), str(eval_end.date()))
    print(f"  {len(news_df)} headline-symbol rows fetched", flush=True)
    if news_df.empty:
        print("No news found for this window/universe. Nothing to do.")
        return

    n_cached = cache_news(news_df)
    print(f"  cached {n_cached} rows", flush=True)

    unique_news = news_df.drop_duplicates(subset="news_id")
    print(f"  {len(unique_news)} unique headlines to consider scoring", flush=True)

    already_scored = _already_scored_ids()
    to_score = unique_news[~unique_news["news_id"].isin(already_scored)]
    print(
        f"  {len(already_scored)} already scored (resuming), {len(to_score)} remaining",
        flush=True,
    )

    scored = 0
    for i, (_, row) in enumerate(to_score.iterrows()):
        score = score_headline(row["headline"])
        cache_sentiment_scores(pd.DataFrame({"news_id": [row["news_id"]], "score": [score]}))
        scored += 1
        if scored % 100 == 0:
            print(f"  [{scored}/{len(to_score)}] scored so far", flush=True)
        time.sleep(LLM_CALL_DELAY_SECONDS)

    print(f"Done. {scored} headlines scored this run.", flush=True)


if __name__ == "__main__":
    main()
