"""Shared helpers for paper-trading runs."""

import logging
from datetime import date, datetime

from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.ports import Notifier

_log = logging.getLogger(TRADER_LOGGER_NAME)


def as_date(value: date | datetime) -> date:
    """Return a date when a database driver returns a datetime."""
    return value.date() if isinstance(value, datetime) else value


def account_read_halt(notifier: Notifier, exc: Exception) -> None:
    """Report an unavailable account before any orders are submitted."""
    _log.error("Account read failed (%s); stopping before orders", type(exc).__name__)
    reason = "Account data unavailable; no orders submitted"
    try:
        notifier.send(f"<b>🛑 HALTED:</b> {reason}")
    except Exception:
        _log.exception("Could not send account read failure notice")
