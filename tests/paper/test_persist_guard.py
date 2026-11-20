"""Tests for the trader persistence isolation guard."""

from unittest.mock import patch

import pytest

from ggTrader.paper import trader
from tests.paper.support import UnstubbedPersistCall, persist_guard


def test_unstubbed_call_raises():
    raised = False
    with pytest.raises(UnstubbedPersistCall, match="^Unstubbed persist calls"):
        with persist_guard():
            try:
                trader.get_peak_value()
            except UnstubbedPersistCall:
                raised = True
    assert raised


def test_swallowed_call_fails_at_teardown():
    with pytest.raises(UnstubbedPersistCall, match="Unstubbed persist calls"):
        with persist_guard():
            try:
                trader.get_peak_value()
            except UnstubbedPersistCall:
                pass


def test_test_level_override_wins():
    with persist_guard():
        with patch("ggTrader.paper.trader.get_peak_value", return_value=42):
            assert trader.get_peak_value() == 42
