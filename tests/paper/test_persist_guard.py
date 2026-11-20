"""Tests for the trader persistence isolation guard."""

from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper import trader
from tests.paper.support import SpyLedger, UnstubbedPersistCall, persist_guard


def test_unstubbed_call_raises():
    raised = False
    with pytest.raises(UnstubbedPersistCall, match="^Unstubbed persist calls"):
        with persist_guard():
            try:
                trader.PgPaperLedger().get_peak_value()
            except UnstubbedPersistCall:
                raised = True
    assert raised


def test_swallowed_call_fails_at_teardown():
    with pytest.raises(UnstubbedPersistCall, match="Unstubbed persist calls"):
        with persist_guard():
            try:
                trader.PgPaperLedger().get_peak_value()
            except UnstubbedPersistCall:
                pass


def test_test_level_override_wins():
    with persist_guard():
        with patch("ggTrader.paper.trader.PgPaperLedger") as factory:
            factory.return_value.get_peak_value.return_value = 42
            assert trader.PgPaperLedger().get_peak_value() == 42


def test_injected_ledger_bypasses_the_default_factory():
    ledger = SpyLedger()
    with persist_guard():
        instance = trader.PaperTrader(
            MagicMock(), MagicMock(), dry_run=False, membership="db", ledger=ledger
        )
        assert instance._ledger is ledger
        assert instance._ledger.get_peak_value() is None
