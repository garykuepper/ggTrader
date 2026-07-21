"""Headline/LLM sentiment -- candidate A8 from WEB_RESEARCH_CANDIDATES.md's
2026-07-19 cross-asset register. Long the top cross-sectional quintile of
stocks by trailing average LLM-scored news-headline sentiment, rebalanced
monthly.

Source: Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?
Return Predictability and Large Language Models," arXiv:2304.07619 --
GPT sentiment scores predict returns, stronger in smaller stocks and
after negative news. Authors themselves warn returns decay as LLM use
becomes more widespread (a crowding effect, not a data artifact).

Data: real, point-in-time historical headlines via Alpaca's /news API
(headline_sentiment_data.py), scored via a cheap/free LLM through this
project's local LiteLLM proxy -- not a lexicon/keyword proxy, matching
the paper's actual mechanism.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

SentimentLoader = Callable[[List[str], str, str], pd.DataFrame]


def _default_sentiment_loader(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    from ggTrader.lab.headline_sentiment_data import load_sentiment_scores

    return load_sentiment_scores(symbols, start, end)


class HeadlineSentimentStrategy:
    """Long-only weights sleeve: equal-weight the top quintile of the
    eligible universe by trailing average headline-sentiment score,
    rebalanced monthly. Symbols with no recent headline coverage are
    excluded from ranking entirely, not treated as neutral (0)."""

    name = "headline_sentiment"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        lookback_days: int = 20,
        quintile: int = 5,
        _sentiment_loader: SentimentLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.lookback_days = lookback_days
        self.quintile = quintile
        self._sentiment_loader: SentimentLoader = _sentiment_loader or _default_sentiment_loader

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "lookback_days": [10, 20, 40],
            "quintile": [4, 5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if not elig:
            return []

        start = (asof - pd.Timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        scores = self._sentiment_loader(elig, start, end)
        if scores.empty:
            return []

        mean_score = scores.groupby("symbol")["score"].mean()
        if len(mean_score) < self.quintile:
            return []

        bucket_size = max(1, len(mean_score) // self.quintile)
        ranked = mean_score.sort_values(ascending=False)
        top = ranked.index[:bucket_size].tolist()
        if not top:
            return []

        weight = 1.0 / len(top)
        return [{"symbol": s, "weight": weight} for s in top]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
