"""S&P 500 index-reconstitution deletion overshoot (fade the deletion).

Event-driven, not the cross-sectional quintile-bucket pattern used
elsewhere in this lab (idio_vol/max_effect/short_interest/pead): go long a
stock shortly after it's forced out of the index by mechanical
index/benchmark-tracking selling, betting on mean-reversion of that
non-fundamental price pressure. Needs no new data source -- deletion
events are derived from the point-in-time SP500 membership history this
lab already maintains (ggTrader.data.core.index_constituents), and the
standard SP500 OHLCV pull already includes every historically-deleted
symbol (it's a union of all-ever-members, see equity_universe_between).

Structural note: a deleted symbol is, by definition, NOT part of the
"eligible" (current-membership) list select() receives -- the strategy
must look past that argument at the deletion-events side channel instead,
the same pattern pairs_stat_arb.py uses for its sector map.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.data.core.index_constituents import load_sp500_history, normalize_yf_ticker
from ggTrader.lab.strategy import LabConfig, Plan

EventsLoader = Callable[[], pd.DataFrame]


def _default_events_loader() -> pd.DataFrame:
    return deletion_events(load_sp500_history())


def deletion_events(history: pd.DataFrame) -> pd.DataFrame:
    """Diff consecutive membership snapshots: any ticker present in row i-1
    but absent from row i was deleted as of row i's date. A ticker deleted
    and later re-added produces one event per deletion, not just the first.
    """
    rows: List[dict] = []
    prev_tickers: set[str] | None = None
    for date, tickers in zip(history.index, history["tickers"]):
        current = set(tickers)
        if prev_tickers is not None:
            for removed in sorted(prev_tickers - current):
                rows.append({"symbol": normalize_yf_ticker(removed), "deletion_date": date})
        prev_tickers = current
    if not rows:
        return pd.DataFrame(columns=["symbol", "deletion_date"])
    return pd.DataFrame(rows)


class IndexDeletionFadeStrategy:
    """Long-only event-driven sleeve: equal-weight every symbol still
    within hold_days of its S&P 500 deletion date, rebalanced monthly.
    Positions accumulate across overlapping deletion events and roll off
    once a given event's hold window elapses.
    """

    name = "index_deletion_fade"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        hold_days: int = 63,
        _events_loader: EventsLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.hold_days = hold_days
        self._events_loader: EventsLoader = _events_loader or _default_events_loader
        self._events: pd.DataFrame | None = None

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {"hold_days": [21, 42, 63]}

    def _load_events(self) -> pd.DataFrame:
        if self._events is None:
            self._events = self._events_loader()
        return self._events

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        events = self._load_events()
        if events.empty:
            return []

        asof_naive = (
            pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
        )
        deletion_date = pd.to_datetime(events["deletion_date"])
        if deletion_date.dt.tz is not None:
            deletion_date = deletion_date.dt.tz_localize(None)
        age_days = (asof_naive - deletion_date).dt.days
        # No look-ahead (age_days >= 0) and still inside the drift window.
        active = events[(age_days >= 0) & (age_days <= self.hold_days)]
        if active.empty:
            return []

        symbols = sorted(active["symbol"].unique())
        available_symbols = set(data.columns.get_level_values(0).unique())
        selected: List[str] = []
        for sym in symbols:
            if sym not in available_symbols:
                continue
            close = data[sym]["close"]
            if close.empty or pd.isna(close.iloc[-1]):
                continue
            selected.append(sym)
        if not selected:
            return []

        weight = 1.0 / len(selected)
        return [{"symbol": s, "weight": weight} for s in selected]

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
