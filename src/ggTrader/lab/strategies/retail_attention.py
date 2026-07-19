"""Retail-attention (Google Trends search-spike) strategy.

Tests Da, Engelberg & Gao's "In Search of Attention" (JF 2011) validated
core finding directly -- abnormal search-volume spikes precede a
short-term price increase (retail attention-driven buying pressure) --
rather than the WEB_RESEARCH_CANDIDATES.md #8 write-up's derivative
"condition an unspecified factor on attention" framing, whose specific
cited figures were flagged unconfirmed. Longs the top quintile by recent
search-volume spike (opposite ranking direction from idio_vol/max_effect/
short_interest's "avoid the risky quintile", same direction as pead),
same quintile-bucket pattern as the other cross-sectional strategies.

Known v1 limitation: the paper's strongest finding is at a ~2-week
horizon; this lab's harness only supports monthly rebalancing (the same
constraint noted in pairs_stat_arb.py's docstring), so this necessarily
tests a coarser cadence than the original result.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.google_trends_data import PUBLISH_LAG_DAYS, available_as_of, load_search_interest
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

#: How far back to pull search-interest history per select() call.
_LOOKBACK_DAYS = 400

InterestLoader = Callable[[List[str], str, str], pd.DataFrame]


class RetailAttentionStrategy:
    """Long-only sleeve: equal-weight the quintile of the eligible universe
    with the LARGEST recent search-interest spike (latest reading vs.
    trailing lookback_months average), rebalanced monthly.
    """

    name = "retail_attention"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        lookback_months: int = 3,
        quintile: int = 5,
        publish_lag_days: int = PUBLISH_LAG_DAYS,
        _interest_loader: InterestLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.lookback_months = lookback_months
        self.quintile = quintile
        self.publish_lag_days = publish_lag_days
        self._interest_loader: InterestLoader = _interest_loader or load_search_interest
        self._interest_cache: Dict[tuple, pd.DataFrame] = {}

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "lookback_months": [2, 3, 6],
            "quintile": [4, 5],
        }

    def _load_interest(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._interest_cache:
            self._interest_cache[key] = self._interest_loader(symbols, start, end)
        return self._interest_cache[key]

    def _spikes(self, interest: pd.DataFrame) -> Dict[str, float]:
        """Per symbol: (latest reading / trailing lookback_months average) - 1.
        Requires at least lookback_months+1 readings for that symbol."""
        out: Dict[str, float] = {}
        for sym, g in interest.sort_values("date").groupby("symbol"):
            if len(g) <= self.lookback_months:
                continue
            values = g["search_interest"].to_numpy()
            latest = values[-1]
            baseline = values[-1 - self.lookback_months : -1].mean()
            if pd.isna(latest) or pd.isna(baseline) or baseline == 0:
                continue
            out[sym] = (latest / baseline) - 1.0
        return out

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        start = (asof - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        interest = self._load_interest(elig, start, end)
        if interest.empty:
            return []

        interest = available_as_of(interest, asof, self.publish_lag_days)
        if interest.empty:
            return []

        spikes = self._spikes(interest)
        if len(spikes) < self.quintile:
            return []

        # Descending: biggest search-interest spike first (attention-driven
        # buying pressure), same ranking direction as pead.
        ranked = pd.Series(spikes).sort_values(ascending=False)
        bucket_size = max(1, len(ranked) // self.quintile)
        top = ranked.index[:bucket_size].tolist()
        if not top:
            return []

        weight = 1.0 / len(top)
        return [{"symbol": s, "weight": weight} for s in top]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
