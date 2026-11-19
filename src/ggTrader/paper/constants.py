"""Shared paper trading constants."""

from zoneinfo import ZoneInfo

TERMINAL_STATUSES: tuple[str, ...] = ("filled", "canceled", "rejected", "expired")
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
