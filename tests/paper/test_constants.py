"""Lock down shared paper constants and clock behavior."""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from ggTrader.paper import gates, signal_runner, trader


def test_terminal_statuses_equal_each_legacy_set() -> None:
    from ggTrader.paper.constants import TERMINAL_STATUSES

    legacy = (
        {"filled", "canceled", "rejected", "expired"},  # trader.py:251, set
        ("filled", "canceled", "rejected", "expired"),  # trader.py:389, tuple
        ("filled", "canceled", "rejected", "expired"),  # trader.py:447, tuple
        {"filled", "canceled", "expired", "rejected"},  # persist.py:108, set
    )
    assert type(TERMINAL_STATUSES) is tuple
    assert TERMINAL_STATUSES == ("filled", "canceled", "rejected", "expired")
    assert all(set(TERMINAL_STATUSES) == set(values) for values in legacy)


def test_time_zones_match_legacy() -> None:
    from ggTrader.paper.constants import ET, PT

    assert ET == ZoneInfo("America/New_York")  # trader.py:78; signal_runner.py:51
    assert PT == ZoneInfo("America/Los_Angeles")  # trader.py:79; gates.py:39
    assert trader._ET == signal_runner._ET == ET
    assert trader._PT == gates.PT == PT


@pytest.mark.parametrize(
    ("instant", "et_date", "pt_time"),
    [
        (datetime(2026, 1, 2, 3, 30, tzinfo=timezone.utc), date(2026, 1, 1), (19, 30)),
        (datetime(2026, 7, 1, 3, 30, tzinfo=timezone.utc), date(2026, 6, 30), (20, 30)),
    ],
)
def test_trader_date_helpers_use_new_york_and_pacific(monkeypatch, instant, et_date, pt_time):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr(trader, "datetime", FixedDateTime)
    assert trader._today_et() == et_date
    now_pt = trader._now_pt()
    assert now_pt.date() == et_date
    assert (now_pt.hour, now_pt.minute) == pt_time
