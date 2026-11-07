"""Tests for the untradable-symbol rules (OD17): configured set, broker status, 422 classifier."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ggTrader.paper.untradable import (
    configured_untradable,
    is_untradable_error,
    resolve_untradable,
)


class _ApiError(Exception):
    """Stand-in for alpaca-py's APIError: carries `status_code` and embeds the JSON body."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code


AVB_BODY = '{"code":42210000,"message":"asset AVB is not active"}'


class TestConfigured:
    def test_parses_a_comma_list_upper_cased(self):
        assert configured_untradable({"UNTRADABLE_SYMBOLS": " avb, Eqr ,,"}) == {"AVB", "EQR"}

    def test_empty_or_missing_is_empty(self):
        assert configured_untradable({}) == frozenset()
        assert configured_untradable({"UNTRADABLE_SYMBOLS": "  "}) == frozenset()


class TestErrorClassifier:
    def test_the_avb_422_is_untradable(self):
        assert is_untradable_error(_ApiError(422, AVB_BODY))

    def test_plain_text_without_status_is_matched(self):
        assert is_untradable_error(RuntimeError("asset EQR is not tradable"))

    @pytest.mark.parametrize(
        "exc",
        [
            _ApiError(422, '{"code":42210000,"message":"qty must be > 0"}'),
            _ApiError(403, '{"message":"insufficient buying power"}'),
            _ApiError(500, "internal error"),
            ConnectionError("timeout"),
        ],
    )
    def test_other_failures_are_real_errors(self, exc):
        assert not is_untradable_error(exc)

    def test_a_server_error_that_mentions_the_words_is_not_matched(self):
        assert not is_untradable_error(_ApiError(500, "asset is not active (retry)"))


class TestResolve:
    def test_union_of_configured_and_broker_status(self):
        broker = MagicMock()
        broker.get_untradable.return_value = {"EQR"}
        got = resolve_untradable(broker, ["AAPL", "AVB", "EQR"], {"AVB", "ZZZ"})
        assert got == {"AVB", "EQR"}
        # Configured names are not asked about again.
        assert broker.get_untradable.call_args.args[0] == ["AAPL", "EQR"]

    def test_lookup_failure_degrades_to_the_configured_set(self):
        broker = MagicMock()
        broker.get_untradable.side_effect = RuntimeError("api down")
        assert resolve_untradable(broker, ["AAPL", "AVB"], {"AVB"}) == {"AVB"}

    def test_nothing_to_ask_makes_no_call(self):
        broker = MagicMock()
        assert resolve_untradable(broker, [], {"AVB"}) == set()
        broker.get_untradable.assert_not_called()
