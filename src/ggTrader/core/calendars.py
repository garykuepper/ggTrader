"""Concrete TradingCalendar implementations.

Phase 3.5 ships ``Crypto24x7Calendar`` (always open). ``USEquityRTHCalendar``
lands in Phase 4 alongside the Alpaca equity broker.
"""

from __future__ import annotations

from datetime import datetime, timezone


class Crypto24x7Calendar:
    """Crypto markets trade 24/7."""

    def __init__(self, calendar_id: str = "crypto_24_7") -> None:
        self.calendar_id = calendar_id

    def is_open(self, ts: datetime) -> bool:
        return True

    def next_open(self, ts: datetime) -> datetime:
        return ts

    def next_close(self, ts: datetime) -> datetime:
        # No session boundary for 24/7 markets; return far future.
        return datetime.max.replace(tzinfo=timezone.utc)

    def sessions_in_range(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        # One continuous "session" from start to end.
        return [(start, end)]
