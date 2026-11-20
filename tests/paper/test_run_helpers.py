"""Shared helpers retain the trader's behavior."""

import logging
from datetime import date, datetime
from unittest.mock import MagicMock

from ggTrader.paper import constants, run_helpers, trader


def test_account_read_halt_logs_and_survives_notifier_failure(caplog):
    notifier = MagicMock()
    notifier.send.side_effect = RuntimeError("offline")
    with caplog.at_level(logging.ERROR, logger=constants.TRADER_LOGGER_NAME):
        run_helpers.account_read_halt(notifier, ValueError())
    assert [record.name for record in caplog.records] == [trader._log.name] * 2
    assert "Account read failed (ValueError); stopping before orders" in caplog.text
    assert "Could not send account read failure notice" in caplog.text
    notifier.send.assert_called_once_with(
        "<b>🛑 HALTED:</b> Account data unavailable; no orders submitted"
    )


def test_as_date_accepts_datetime_and_date():
    value = date(2026, 1, 2)
    assert run_helpers.as_date(datetime(2026, 1, 2)) == value
    assert run_helpers.as_date(value) is value


def test_trader_aliases_are_the_new_homes():
    assert trader._as_date is run_helpers.as_date
    assert trader._account_read_halt is run_helpers.account_read_halt
    assert constants.TRADER_LOGGER_NAME == trader._log.name
    assert trader._MAX_SIGNAL_AGE_DAYS == constants.MAX_SIGNAL_AGE_DAYS
    assert trader._STALE_PENDING_DAYS == constants.STALE_PENDING_DAYS
    assert trader._SPLIT_LOOKBACK_DAYS == constants.SPLIT_LOOKBACK_DAYS
