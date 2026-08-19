"""Tests for detecting stock-split events the paper broker failed to apply.

Context: Alpaca's paper-trading environment can mark a position to the
post-split market price without adjusting the position's own qty/avg_entry
(observed on MNST's 2026-08-11 2-for-1 split). That silently corrupts any
downstream unrealized P&L math built from the broker's position fields.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from ggTrader.paper.split_check import (
    find_unadjusted_split_symbols,
    get_recent_splits,
    is_split_unadjusted,
)


class TestIsSplitUnadjusted:
    def test_flags_unchanged_qty_after_split(self):
        # Broker never multiplied qty by the 2-for-1 ratio -> unadjusted.
        assert is_split_unadjusted(qty_before=20.8, qty_after=20.8, split_ratio=2.0) is True

    def test_does_not_flag_correctly_adjusted_qty(self):
        # Broker doubled qty for the 2-for-1 split -> correctly adjusted.
        assert is_split_unadjusted(qty_before=20.8, qty_after=41.6, split_ratio=2.0) is False

    def test_tolerates_small_float_drift(self):
        assert is_split_unadjusted(qty_before=20.804, qty_after=41.60799, split_ratio=2.0) is False

    def test_reverse_split_flags_unchanged_qty(self):
        # 1-for-10 reverse split (ratio 0.1); broker left qty unchanged.
        assert is_split_unadjusted(qty_before=100.0, qty_after=100.0, split_ratio=0.1) is True

    def test_reverse_split_does_not_flag_correctly_adjusted_qty(self):
        assert is_split_unadjusted(qty_before=100.0, qty_after=10.0, split_ratio=0.1) is False


class TestFindUnadjustedSplitSymbols:
    def test_flags_symbol_with_unadjusted_split_in_window(self):
        prev_positions = {"MNST": {"qty": 20.804}, "VZ": {"qty": 77.05}}
        curr_positions = {"MNST": {"qty": 20.804}, "VZ": {"qty": 77.05}}
        splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 12),
        )

        assert result == ["MNST"]

    def test_ignores_symbols_with_no_split(self):
        prev_positions = {"VZ": {"qty": 77.05}}
        curr_positions = {"VZ": {"qty": 77.05}}
        splits: dict[str, list] = {}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 12),
        )

        assert result == []

    def test_ignores_symbol_no_longer_held(self):
        # Position was closed since the split -> nothing to flag, no stale
        # unrealized P&L can leak into the report.
        prev_positions = {"MNST": {"qty": 20.804}}
        curr_positions: dict[str, dict] = {}
        splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 12),
        )

        assert result == []

    def test_ignores_symbol_not_held_before_split(self):
        # No prior qty on record -> nothing to compare against, skip rather
        # than risk a false positive.
        prev_positions: dict[str, dict] = {}
        curr_positions = {"MNST": {"qty": 20.804}}
        splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 12),
        )

        assert result == []

    def test_first_day_after_split_flags_unadjusted_broker(self):
        # The real MNST case: prev snapshot predates the ex-date, today is
        # on/after it, and the broker never touched qty.
        prev_positions = {"MNST": {"qty": 20.804}}
        curr_positions = {"MNST": {"qty": 20.804}}
        splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 11),
        )

        assert result == ["MNST"]

    def test_day_two_after_correctly_adjusted_split_not_reflagged(self):
        # Regression test: on day 1 (run_date == ex_date) the broker already
        # doubled qty correctly. On day 2, prev_snapshot_date has rolled
        # past the ex-date, so the split no longer applies to this window's
        # comparison and the (now-stable) qty must NOT be re-flagged.
        prev_positions = {"MNST": {"qty": 41.608}}  # already-adjusted qty from day 1
        curr_positions = {"MNST": {"qty": 41.608}}  # unchanged on day 2 -- correct
        splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 11),
            today=date(2026, 8, 12),
        )

        assert result == []

    def test_reverse_split_flags_unadjusted_symbol(self):
        prev_positions = {"XYZ": {"qty": 100.0}}
        curr_positions = {"XYZ": {"qty": 100.0}}  # never reduced for 1-for-10 reverse split
        splits = {"XYZ": [(date(2026, 8, 11), 0.1)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 11),
        )

        assert result == ["XYZ"]

    def test_reverse_split_does_not_flag_correctly_adjusted_symbol(self):
        prev_positions = {"XYZ": {"qty": 100.0}}
        curr_positions = {"XYZ": {"qty": 10.0}}  # correctly reduced
        splits = {"XYZ": [(date(2026, 8, 11), 0.1)]}

        result = find_unadjusted_split_symbols(
            prev_positions,
            curr_positions,
            splits,
            prev_snapshot_date=date(2026, 8, 10),
            today=date(2026, 8, 11),
        )

        assert result == []


def _hist_with_split(ex_date: str, factor: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(ex_date)])
    return pd.DataFrame({"Stock Splits": [factor]}, index=idx)


def _hist_no_splits() -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-05")])
    return pd.DataFrame({"Stock Splits": [0.0]}, index=idx)


class TestGetRecentSplits:
    @patch("ggTrader.paper.split_check.yf.Ticker")
    def test_returns_ex_dates_for_symbol_with_split(self, mock_ticker):
        mock_ticker.return_value.history.return_value = _hist_with_split("2026-08-11", 2.0)

        splits, all_failed = get_recent_splits(["MNST"], "2026-08-01")

        assert splits == {"MNST": [(date(2026, 8, 11), 2.0)]}
        assert all_failed is False

    @patch("ggTrader.paper.split_check.yf.Ticker")
    def test_omits_symbol_with_no_splits(self, mock_ticker):
        mock_ticker.return_value.history.return_value = _hist_no_splits()

        splits, all_failed = get_recent_splits(["VZ"], "2026-08-01")

        assert splits == {}
        assert all_failed is False

    def test_empty_symbol_list_returns_empty_not_all_failed(self):
        splits, all_failed = get_recent_splits([], "2026-08-01")

        assert splits == {}
        assert all_failed is False

    @patch("ggTrader.paper.split_check.yf.Ticker")
    def test_single_symbol_failure_is_not_all_failed_when_others_succeed(self, mock_ticker):
        good = MagicMock()
        good.history.return_value = _hist_no_splits()
        bad = MagicMock()
        bad.history.side_effect = Exception("network error")

        def _ticker(symbol):
            return bad if symbol == "BAD" else good

        mock_ticker.side_effect = _ticker

        splits, all_failed = get_recent_splits(["GOOD", "BAD"], "2026-08-01")

        assert splits == {}
        assert all_failed is False

    @patch("ggTrader.paper.split_check.yf.Ticker")
    def test_all_symbols_failing_reports_all_failed(self, mock_ticker):
        mock_ticker.return_value.history.side_effect = Exception("network error")

        splits, all_failed = get_recent_splits(["A", "B", "C"], "2026-08-01")

        assert splits == {}
        assert all_failed is True
