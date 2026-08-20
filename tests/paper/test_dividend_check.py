"""Tests for computing dividend accruals for cash dividends the broker
never credited.

Context: verified 2026-08-18, the Alpaca PAPER account's non-trade activity
log contains only FEE entries since 2026-05-01 -- `/v2/account/activities/DIV`
returns `[]` -- while the corporate-actions feed lists 21 cash dividends on
held symbols since 2026-05-15 that were never credited.
"""

from __future__ import annotations

from datetime import date

import pytest

from ggTrader.paper.dividend_check import (
    compute_dividend_accruals,
    normalize_since,
    qty_held_at_ex_date,
)


class TestNormalizeSince:
    def test_passes_through_a_date(self):
        d = date(2026, 5, 15)
        assert normalize_since(d) is d

    def test_parses_an_iso_date_string(self):
        assert normalize_since("2026-05-15") == date(2026, 5, 15)

    def test_wrong_type_raises_loudly(self):
        # The split-check landmine: a caller passing the wrong type for
        # `since` must fail loudly here, not be swallowed downstream into a
        # `{}` that looks identical to "no corrections known".
        with pytest.raises(TypeError, match="since must be a date or ISO date string"):
            normalize_since(12345)

    def test_malformed_string_raises(self):
        with pytest.raises(ValueError):
            normalize_since("not-a-date")


class TestQtyHeldAtExDate:
    def test_returns_qty_from_most_recent_snapshot_at_or_before_ex_date(self):
        history = [
            (date(2026, 5, 10), {"VZ": {"qty": 50.0}}),
            (date(2026, 6, 1), {"VZ": {"qty": 77.05}}),
        ]

        assert qty_held_at_ex_date("VZ", date(2026, 6, 15), history) == 77.05

    def test_uses_the_snapshot_exactly_on_the_ex_date(self):
        history = [(date(2026, 8, 1), {"VZ": {"qty": 77.05}})]

        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), history) == 77.05

    def test_bought_after_ex_date_is_not_held(self):
        # No snapshot at/before the ex-date shows the position -- the first
        # one to hold it comes after.
        history = [(date(2026, 8, 15), {"VZ": {"qty": 77.05}})]

        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), history) is None

    def test_no_snapshot_history_at_all(self):
        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), []) is None

    def test_symbol_absent_from_snapshot_is_not_held(self):
        history = [(date(2026, 8, 1), {"AAPL": {"qty": 10.0}})]

        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), history) is None

    def test_position_closed_before_ex_date_is_not_held(self):
        # Held earlier, then sold (dropped from the snapshot) before the
        # ex-date -- the most recent snapshot at/before ex_date no longer
        # carries the symbol.
        history = [
            (date(2026, 7, 1), {"VZ": {"qty": 50.0}}),
            (date(2026, 7, 20), {}),
        ]

        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), history) is None

    def test_zero_qty_treated_as_not_held(self):
        history = [(date(2026, 8, 1), {"VZ": {"qty": 0.0}})]

        assert qty_held_at_ex_date("VZ", date(2026, 8, 1), history) is None


class TestComputeDividendAccruals:
    def test_accrues_a_held_dividend(self):
        corp_dividends = {"VZ": [(date(2026, 8, 1), 0.6725)]}
        history = [(date(2026, 7, 1), {"VZ": {"qty": 77.05}})]

        accruals, skipped = compute_dividend_accruals(corp_dividends, set(), set(), history)

        assert accruals == [
            {
                "symbol": "VZ",
                "ex_date": date(2026, 8, 1),
                "rate": 0.6725,
                "qty": 77.05,
                "amount": pytest.approx(77.05 * 0.6725),
            }
        ]
        assert skipped == []

    def test_already_credited_by_broker_is_not_accrued(self):
        # The case that would double-count if Alpaca's paper environment
        # ever starts crediting dividends correctly.
        corp_dividends = {"VZ": [(date(2026, 8, 1), 0.6725)]}
        credited_keys = {("VZ", date(2026, 8, 1))}
        history = [(date(2026, 7, 1), {"VZ": {"qty": 77.05}})]

        accruals, skipped = compute_dividend_accruals(corp_dividends, credited_keys, set(), history)

        assert accruals == []
        assert skipped == [
            {
                "symbol": "VZ",
                "ex_date": date(2026, 8, 1),
                "rate": 0.6725,
                "reason": "already_credited_by_broker",
            }
        ]

    def test_already_accrued_is_idempotent(self):
        corp_dividends = {"VZ": [(date(2026, 8, 1), 0.6725)]}
        already_accrued = {("VZ", date(2026, 8, 1))}
        history = [(date(2026, 7, 1), {"VZ": {"qty": 77.05}})]

        accruals, skipped = compute_dividend_accruals(
            corp_dividends, set(), already_accrued, history
        )

        assert accruals == []
        assert skipped[0]["reason"] == "already_accrued"

    def test_symbol_bought_after_ex_date_is_skipped_not_guessed(self):
        corp_dividends = {"VZ": [(date(2026, 8, 1), 0.6725)]}
        history = [(date(2026, 8, 15), {"VZ": {"qty": 77.05}})]  # bought after ex-date

        accruals, skipped = compute_dividend_accruals(corp_dividends, set(), set(), history)

        assert accruals == []
        assert skipped == [
            {
                "symbol": "VZ",
                "ex_date": date(2026, 8, 1),
                "rate": 0.6725,
                "reason": "not_held_on_ex_date",
            }
        ]

    def test_never_held_symbol_is_skipped(self):
        corp_dividends = {"VZ": [(date(2026, 8, 1), 0.6725)]}

        accruals, skipped = compute_dividend_accruals(corp_dividends, set(), set(), [])

        assert accruals == []
        assert skipped[0]["reason"] == "not_held_on_ex_date"

    def test_multiple_symbols_and_events_mixed_outcomes(self):
        corp_dividends = {
            "VZ": [(date(2026, 6, 1), 0.6725), (date(2026, 8, 1), 0.6725)],
            "AEE": [(date(2026, 7, 1), 0.6)],
        }
        credited_keys = {("VZ", date(2026, 6, 1))}
        history = [
            (date(2026, 5, 20), {"VZ": {"qty": 77.05}}),
            (date(2026, 8, 15), {"AEE": {"qty": 10.0}}),  # AEE bought after its ex-date
        ]

        accruals, skipped = compute_dividend_accruals(corp_dividends, credited_keys, set(), history)

        assert len(accruals) == 1
        assert accruals[0]["symbol"] == "VZ"
        assert accruals[0]["ex_date"] == date(2026, 8, 1)
        reasons = {(s["symbol"], s["ex_date"]): s["reason"] for s in skipped}
        assert reasons[("VZ", date(2026, 6, 1))] == "already_credited_by_broker"
        assert reasons[("AEE", date(2026, 7, 1))] == "not_held_on_ex_date"

    def test_amount_is_rate_times_qty_at_ex_date_not_todays_qty(self):
        # The core correctness requirement: a position trimmed after the
        # ex-date must still be credited at the qty held ON the ex-date.
        corp_dividends = {"VZ": [(date(2026, 6, 1), 0.6725)]}
        history = [
            (date(2026, 5, 20), {"VZ": {"qty": 100.0}}),  # held 100 on ex-date
            (date(2026, 7, 1), {"VZ": {"qty": 10.0}}),  # trimmed down after
        ]

        accruals, _skipped = compute_dividend_accruals(corp_dividends, set(), set(), history)

        assert accruals[0]["qty"] == 100.0
        assert accruals[0]["amount"] == pytest.approx(100.0 * 0.6725)
