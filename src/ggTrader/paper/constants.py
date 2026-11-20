"""Shared paper trading constants."""

from zoneinfo import ZoneInfo

TERMINAL_STATUSES: tuple[str, ...] = ("filled", "canceled", "rejected", "expired")
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")

TRADER_LOGGER_NAME = "ggTrader.paper.trader"
MAX_SIGNAL_AGE_DAYS = 4
STALE_PENDING_DAYS = 5
SPLIT_LOOKBACK_DAYS = 14
