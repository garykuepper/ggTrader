"""Hard-to-borrow / short-interest signal -- free-data-only cut.

Avoids the paid real-time cost-to-borrow/utilization feed (Ortex, IHS
Markit) entirely; uses only FINRA's free consolidated short-interest
(bi-monthly settlement-date positions, days-to-cover) as a proxy for
"transitioning from easy- to hard-to-borrow." See
short_interest_data.py and RESEARCH_SNAPSHOT.md / WEB_RESEARCH_CANDIDATES.md
candidate #3 for the full mechanism writeup and its data-feasibility caveats.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.short_interest_data import (
    PUBLISH_LAG_DAYS,
    available_as_of,
    load_short_interest,
)
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

#: How far back to pull short-interest history per select() call -- enough
#: trailing settlement cycles (bi-monthly, so ~24/year) to cover the widest
#: swept lookback_cycles with room to spare.
_LOOKBACK_DAYS = 400

SiLoader = Callable[[List[str], str, str], pd.DataFrame]


class ShortInterestStrategy:
    """Long-only defensive sleeve: equal-weight the quintile of the eligible
    universe with the SMALLEST recent increase in days-to-cover (avoiding
    names transitioning toward hard-to-borrow), rebalanced monthly. Same
    quintile-bucket, weight-based pattern as idio_vol.py/max_effect.py.

    Point-in-time gated against FINRA's real settlement->publish reporting
    lag (available_as_of) -- a settlement's days-to-cover figure is never
    used before it would actually have been published.
    """

    name = "short_interest"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        lookback_cycles: int = 1,
        quintile: int = 5,
        publish_lag_days: int = PUBLISH_LAG_DAYS,
        _si_loader: SiLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.lookback_cycles = lookback_cycles
        self.quintile = quintile
        self.publish_lag_days = publish_lag_days
        self._si_loader: SiLoader = _si_loader or load_short_interest
        self._si_cache: Dict[tuple, pd.DataFrame] = {}

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "lookback_cycles": [1, 2, 3],
            "quintile": [4, 5],
        }

    def _load_si(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._si_cache:
            self._si_cache[key] = self._si_loader(symbols, start, end)
        return self._si_cache[key]

    def _days_to_cover_trend(self, si: pd.DataFrame) -> Dict[str, float]:
        """Per symbol: % change in days-to-cover between the latest available
        settlement and lookback_cycles settlements before it. Requires at
        least lookback_cycles+1 available settlements for that symbol."""
        trend: Dict[str, Any] = {}
        for sym, g in si.sort_values("settlement_date").groupby("symbol"):
            if len(g) <= self.lookback_cycles:
                continue
            latest = g["days_to_cover"].iloc[-1]
            prior = g["days_to_cover"].iloc[-1 - self.lookback_cycles]
            if pd.isna(latest) or pd.isna(prior) or prior == 0:
                continue
            trend[sym] = (latest - prior) / prior
        return trend

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        start = (asof - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        si = self._load_si(elig, start, end)
        if si.empty:
            return []

        si = available_as_of(si, asof, self.publish_lag_days)
        if si.empty:
            return []

        trend = self._days_to_cover_trend(si)
        if len(trend) < self.quintile:
            return []

        # Ascending: smallest (or most-negative) days-to-cover change first --
        # avoids the "increasingly hard to borrow" names, not a replacement
        # for them; the safest quintile goes long.
        ranked = pd.Series(trend).sort_values()
        bucket_size = max(1, len(ranked) // self.quintile)
        bottom = ranked.index[:bucket_size].tolist()
        if not bottom:
            return []

        weight = 1.0 / len(bottom)
        return [{"symbol": s, "weight": weight} for s in bottom]

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
