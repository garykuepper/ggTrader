"""Pre-FOMC long-Treasury drift -- candidate A7 from WEB_RESEARCH_CANDIDATES.md's
2026-07-19 cross-asset register. Long TLT/IEF/EDV the trading day before a
scheduled FOMC announcement, exit at/after the announcement.

Source: Jun Pan & Qing Peng, "The Pre-FOMC Drift in Long-Term Treasury
Bonds" (current draft June 2026) -- documents positive, statistically
significant long-Treasury returns specifically on the day BEFORE scheduled
FOMC announcements, concentrated at longer maturities, driven primarily by
the term-premium component and uncertainty resolution.

Event-driven, daily-granular (a 0-3 trading-day hold), so this uses the
`target_kind="signals"` protocol (entries/exits at exact bars) rather than
this lab's monthly-rebalance `"weights"` protocol -- the latter can't
express a hold window shorter than a month. select() is a fixed-universe
pass-through (like fx_hedge_overlay, this strategy doesn't filter the
`eligible` SP500-style argument at all); all the real logic lives in
to_targets(), which places entries/exits directly from the FOMC calendar
regardless of which asof select() was called at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

FOMC_TREASURY_UNIVERSE = ["TLT", "IEF", "EDV"]

EventsLoader = Callable[[], List[pd.Timestamp]]

#: scripts/fomc_calendar_backfill.py writes this; avoids re-scraping the
#: Fed's site (~11 requests) on every process start / WFO run.
_CACHE_CSV = Path(__file__).resolve().parents[4] / "data" / "universe" / "fomc_meeting_dates.csv"


def _default_events_loader() -> List[pd.Timestamp]:
    if _CACHE_CSV.exists():
        df = pd.read_csv(_CACHE_CSV)
        return sorted(pd.to_datetime(df["announcement_date"]).tolist())

    from ggTrader.lab.fomc_calendar import historical_fomc_announcement_dates

    return historical_fomc_announcement_dates(start_year=2011)


class FomcDriftStrategy:
    name = "fomc_drift"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        entry_offset_days: int = 1,
        hold_days: int = 0,
        _events_loader: EventsLoader | None = None,
    ) -> None:
        self.cfg = cfg
        #: Trading days BEFORE the announcement to enter (1 = day before).
        self.entry_offset_days = entry_offset_days
        #: Trading days to hold AFTER the announcement bar (0 = exit same bar).
        self.hold_days = hold_days
        self._events_loader: EventsLoader = _events_loader or _default_events_loader
        self._events: List[pd.Timestamp] | None = None

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "entry_offset_days": [1, 2],
            "hold_days": [0, 1, 2],
        }

    def _load_events(self) -> List[pd.Timestamp]:
        if self._events is None:
            self._events = self._events_loader()
        return self._events

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        have = set(data.columns.get_level_values(0).unique())
        universe = [s for s in FOMC_TREASURY_UNIVERSE if s in have]
        return [{"symbol": s, "weight": 0.0} for s in universe]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        return _event_signals(
            close.index, symbols, self._load_events(), self.entry_offset_days, self.hold_days
        )

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        events = self._load_events()
        cache: dict[tuple[int, int], SignalTargets] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            entry_offset_days = int(combo["entry_offset_days"])
            hold_days = int(combo["hold_days"])
            key = (entry_offset_days, hold_days)
            if key not in cache:
                cache[key] = _event_signals(
                    close.index, symbols, events, entry_offset_days, hold_days
                )
            result[combo_name(self.name, combo)] = cache[key]
        return result


def _event_signals(
    index: pd.DatetimeIndex,
    symbols: List[str],
    events: List[pd.Timestamp],
    entry_offset_days: int,
    hold_days: int,
) -> SignalTargets:
    """Entries entry_offset_days trading bars before each event, exits
    hold_days bars after -- events with no room on either side (window
    edges) are silently skipped, not an error."""
    entries = pd.DataFrame(False, index=index, columns=symbols)
    exits = pd.DataFrame(False, index=index, columns=symbols)

    # Compare on calendar date only -- real OHLCV bars carry a non-midnight
    # time-of-day (e.g. "16:00:00+00:00", this lab's close-time convention)
    # while FOMC event dates are plain midnight-normalized Timestamps; an
    # exact-timestamp match between the two silently matches nothing.
    idx_dates = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
    date_to_pos = {d: i for i, d in enumerate(idx_dates)}

    for event in events:
        event_date = pd.Timestamp(event).tz_localize(None).normalize()
        pos = date_to_pos.get(event_date)
        if pos is None:
            continue  # event date not a trading day in this window (unexpected but safe)
        entry_pos = pos - entry_offset_days
        exit_pos = pos + hold_days
        if 0 <= entry_pos < len(index):
            entries.iloc[entry_pos] = True
        if 0 <= exit_pos < len(index):
            exits.iloc[exit_pos] = True

    return SignalTargets(entries=entries, exits=exits)
