"""TradingCalendar: market-hours abstraction. Crypto is 24/7; equities are RTH.

Strategies query the calendar instead of hardcoding market hours, so the same
strategy code works across asset classes. Implementations land in Phase 4.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class TradingCalendar(Protocol):
    calendar_id: str

    def is_open(self, ts: datetime) -> bool: ...

    def next_open(self, ts: datetime) -> datetime: ...

    def next_close(self, ts: datetime) -> datetime: ...

    def sessions_in_range(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]: ...
