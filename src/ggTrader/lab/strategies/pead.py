"""Post-earnings-announcement drift (PEAD).

Unlike idio_vol.py/max_effect.py/short_interest.py (all "avoid the risky
quintile" constructions), classic PEAD is the opposite: seek OUT the
biggest recent positive surprises and ride the underreaction drift. See
earnings_surprise_data.py and RESEARCH_SNAPSHOT.md / WEB_RESEARCH_CANDIDATES.md
candidate #12 for the full mechanism writeup, including the literature's
own caveat that the effect is contested/weak in large, liquid U.S. stocks
specifically (Martineau 2022) -- this project tests it primarily on
lower-coverage universes (nasdaq100/russell2000 already available in this
lab) rather than SP500, per that caveat.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.earnings_surprise_data import (
    REPORT_LAG_DAYS,
    available_as_of,
    load_earnings_surprises,
)
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

#: How far back to pull earnings-surprise history per select() call -- one
#: year comfortably covers every symbol's most recent quarterly report
#: regardless of swept max_age_days.
_LOOKBACK_DAYS = 400

SurpriseLoader = Callable[[List[str], str, str], pd.DataFrame]


class PeadStrategy:
    """Long-only momentum-on-surprise sleeve: equal-weight the quintile of
    the eligible universe with the LARGEST recent positive earnings
    surprise (underreaction drift), rebalanced monthly. Same
    quintile-bucket, weight-based pattern as idio_vol.py/max_effect.py/
    short_interest.py, but ranked descending (biggest surprise wins) rather
    than ascending (avoid the riskiest).

    Point-in-time gated against a conservative report->tradeable lag
    (available_as_of) and a drift-window cutoff (max_age_days) -- a
    surprise older than the window has already drifted its course and must
    not still be driving a position.
    """

    name = "pead"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        max_age_days: int = 90,
        quintile: int = 5,
        report_lag_days: int = REPORT_LAG_DAYS,
        _surprise_loader: SurpriseLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.max_age_days = max_age_days
        self.quintile = quintile
        self.report_lag_days = report_lag_days
        self._surprise_loader: SurpriseLoader = _surprise_loader or load_earnings_surprises
        self._surprise_cache: Dict[tuple, pd.DataFrame] = {}

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "max_age_days": [30, 60, 90],
            "quintile": [4, 5],
        }

    def _load_surprises(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._surprise_cache:
            self._surprise_cache[key] = self._surprise_loader(symbols, start, end)
        return self._surprise_cache[key]

    def _most_recent_surprise(
        self, surprises: pd.DataFrame, asof: pd.Timestamp
    ) -> Dict[str, float]:
        """Per symbol: the most recent surprise_pct still inside the drift
        window (asof - earnings_date <= max_age_days)."""
        out: Dict[str, float] = {}
        for sym, g in surprises.sort_values("earnings_date").groupby("symbol"):
            latest_date = g["earnings_date"].iloc[-1]
            age_days = (pd.Timestamp(asof).tz_localize(None) - pd.Timestamp(latest_date)).days
            if age_days > self.max_age_days:
                continue
            out[sym] = float(g["surprise_pct"].iloc[-1])
        return out

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        start = (asof - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        surprises = self._load_surprises(elig, start, end)
        if surprises.empty:
            return []

        surprises = available_as_of(surprises, asof, self.report_lag_days)
        if surprises.empty:
            return []

        recent = self._most_recent_surprise(surprises, asof)
        if len(recent) < self.quintile:
            return []

        # Descending: biggest positive surprise first -- PEAD longs the
        # underreaction drift, the opposite ranking direction of the
        # "avoid the risky quintile" sleeves elsewhere in this lab.
        ranked = pd.Series(recent).sort_values(ascending=False)
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
