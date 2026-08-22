"""Tests for computing corrections for stock splits the broker never applied.

Context: Alpaca's paper-trading environment can mark a position to the
post-split market price while leaving its qty/avg_entry at the pre-split
values (observed on MNST's 2026-08-11 2-for-1 split: broker qty stayed at
20.80410098, cost_basis $1,887.11, avg_entry $90.708558, but current_price
was already the post-split $47.77). Alpaca's own corporate-actions feed
proves the split happened; its account-activities feed proves it was never
booked against the account. Cross-referencing the two is the detector.
"""

from __future__ import annotations

from datetime import date

import pytest

from ggTrader.paper.split_check import (
    apply_corrections_to_positions,
    compute_split_corrections,
    corrected_market_value,
    corrected_unrealized_pl,
    find_split_applied_symbols,
)

# Real MNST numbers from the 2026-08-19 incident report.
_MNST_QTY = 20.80410098
_MNST_COST_BASIS = 1887.11
_MNST_POST_SPLIT_PRICE = 47.77
_MNST_BROKER_MARKET_VALUE = _MNST_QTY * _MNST_POST_SPLIT_PRICE  # 993.81..., broker's bogus value
_MNST_FACTOR = 2.0
_MNST_TRUE_SHARES = _MNST_QTY * _MNST_FACTOR  # 41.60820196
_MNST_TRUE_MARKET_VALUE = _MNST_TRUE_SHARES * _MNST_POST_SPLIT_PRICE  # 1987.62...
_MNST_TRUE_UNREALIZED_PL = _MNST_TRUE_MARKET_VALUE - _MNST_COST_BASIS  # +100.51...


class TestComputeSplitCorrections:
    def test_unapplied_forward_split_is_flagged(self):
        corp_splits = {"MNST": [(date(2026, 8, 11), 2.0)]}
        applied_symbols: set[str] = set()  # no matching SPLIT activity booked

        result = compute_split_corrections(corp_splits, applied_symbols)

        assert result == {"MNST": 2.0}

    def test_split_with_matching_activity_not_flagged(self):
        # The common case: the broker DID book the split correctly. This is
        # exactly what the old qty-delta heuristic falsely re-flagged for up
        # to 14 days after a correct adjustment -- must not happen here.
        corp_splits = {"MNST": [(date(2026, 8, 11), 2.0)]}
        applied_symbols = {"MNST"}

        result = compute_split_corrections(corp_splits, applied_symbols)

        assert result == {}

    def test_reverse_split_factor_below_one(self):
        corp_splits = {"XYZ": [(date(2026, 8, 11), 0.1)]}  # 1-for-10 reverse

        result = compute_split_corrections(corp_splits, set())

        assert result == {"XYZ": 0.1}

    def test_multiple_splits_in_window_multiply(self):
        corp_splits = {"ABC": [(date(2026, 8, 1), 2.0), (date(2026, 8, 15), 3.0)]}

        result = compute_split_corrections(corp_splits, set())

        assert result == {"ABC": 6.0}

    def test_symbol_with_no_split_events_is_not_flagged(self):
        result = compute_split_corrections({"VZ": []}, set())

        assert result == {}

    def test_unrelated_held_symbol_without_a_split_is_absent(self):
        corp_splits = {"MNST": [(date(2026, 8, 11), 2.0)]}

        result = compute_split_corrections(corp_splits, set())

        assert "VZ" not in result

    def test_mix_of_applied_and_unapplied_symbols(self):
        corp_splits = {
            "MNST": [(date(2026, 8, 11), 2.0)],
            "XYZ": [(date(2026, 8, 11), 0.1)],
        }
        applied_symbols = {"XYZ"}  # only XYZ's split was booked correctly

        result = compute_split_corrections(corp_splits, applied_symbols)

        assert result == {"MNST": 2.0}


class TestCorrectedMarketValue:
    def test_scales_market_value_by_factor(self):
        assert corrected_market_value(_MNST_BROKER_MARKET_VALUE, _MNST_FACTOR) == (
            _MNST_TRUE_MARKET_VALUE
        )

    def test_matches_real_mnst_true_market_value(self):
        result = corrected_market_value(_MNST_BROKER_MARKET_VALUE, _MNST_FACTOR)
        assert result == pytest.approx(1987.62, abs=0.01)


class TestCorrectedUnrealizedPl:
    def test_matches_real_mnst_true_unrealized_pl(self):
        # Broker reported roughly -$893.30 (-47.3%); the true position is
        # actually up about +$100.51 (+5.3%) once the split is corrected.
        result = corrected_unrealized_pl(_MNST_BROKER_MARKET_VALUE, _MNST_FACTOR, _MNST_COST_BASIS)
        assert result == pytest.approx(100.51, abs=0.01)
        assert result > 0  # true position is a gain, not the phantom loss

    def test_broker_reported_loss_was_phantom(self):
        broker_reported_pl = _MNST_BROKER_MARKET_VALUE - _MNST_COST_BASIS
        assert broker_reported_pl == pytest.approx(-893.30, abs=0.01)

        corrected = corrected_unrealized_pl(
            _MNST_BROKER_MARKET_VALUE, _MNST_FACTOR, _MNST_COST_BASIS
        )
        assert corrected != pytest.approx(broker_reported_pl, abs=1.0)


class TestApplyCorrectionsToPositions:
    def test_corrects_flagged_symbol_using_real_mnst_numbers(self):
        positions = {
            "MNST": {
                "qty": _MNST_QTY,  # broker's uncorrected qty -- must NOT change
                "market_value": _MNST_BROKER_MARKET_VALUE,
                "unrealized_pl": _MNST_BROKER_MARKET_VALUE - _MNST_COST_BASIS,
                "unrealized_plpc": (_MNST_BROKER_MARKET_VALUE - _MNST_COST_BASIS)
                / _MNST_COST_BASIS,
                "cost_basis": _MNST_COST_BASIS,
                "current_price": _MNST_POST_SPLIT_PRICE,
            },
        }

        result = apply_corrections_to_positions(positions, {"MNST": _MNST_FACTOR})

        mnst = result["MNST"]
        assert mnst["market_value"] == pytest.approx(_MNST_TRUE_MARKET_VALUE, abs=0.01)
        assert mnst["unrealized_pl"] == pytest.approx(_MNST_TRUE_UNREALIZED_PL, abs=0.01)
        assert mnst["unrealized_plpc"] == pytest.approx(
            _MNST_TRUE_UNREALIZED_PL / _MNST_COST_BASIS, abs=1e-6
        )
        # Broker qty is left untouched -- it's what can actually be sold.
        assert mnst["qty"] == _MNST_QTY

    def test_leaves_unflagged_position_unchanged(self):
        positions = {"VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22}}

        result = apply_corrections_to_positions(positions, {})

        assert result == positions
        assert result is not positions  # defensive copy, not the same dict

    def test_only_touches_flagged_symbols_in_mixed_book(self):
        positions = {
            "MNST": {
                "qty": _MNST_QTY,
                "market_value": _MNST_BROKER_MARKET_VALUE,
                "unrealized_pl": -893.30,
                "unrealized_plpc": -0.4733,
                "cost_basis": _MNST_COST_BASIS,
            },
            "VZ": {"qty": 77.05, "market_value": 3743.85, "unrealized_pl": 365.22},
        }

        result = apply_corrections_to_positions(positions, {"MNST": _MNST_FACTOR})

        assert result["VZ"] == positions["VZ"]
        assert result["MNST"]["unrealized_pl"] == pytest.approx(_MNST_TRUE_UNREALIZED_PL, abs=0.01)


class TestFindSplitAppliedSymbols:
    """Snapshot-history-based evidence for whether a broker-known split was
    actually applied to this account -- the hardening this module's
    docstring describes, replacing the never-exercised activities-feed
    guard (see docs/next_steps.md, "The one real defect this audit did
    surface")."""

    _EX_DATE = date(2026, 8, 11)
    _TODAY = date(2026, 8, 22)
    _CORP_SPLITS = {"MNST": [(_EX_DATE, 2.0)]}

    def test_qty_doubling_across_ex_date_is_applied(self):
        # Broker correctly booked the split: qty on the books doubles
        # between the last pre-ex-date snapshot and the first post-ex-date
        # one, with no trade in between to explain it another way.
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == {"MNST"}
        assert unresolved == set()

    def test_flat_qty_mnst_shaped_fixture_is_unapplied(self):
        # The real, observed MNST incident shape: qty never moved across
        # the ex-date, so the split was never applied and must still be
        # corrected.
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 8, 12), {"MNST": {"qty": 20.80410098}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == set()

    def test_trade_across_ex_date_is_not_mistaken_for_a_split(self):
        # qty happens to double, but a trade in the same window fully
        # explains it (e.g. a same-size buy) -- must not be inferred as a
        # confirmed split application from the qty ratio alone.
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}}),
        ]
        trade_dates_by_symbol = {"MNST": [date(2026, 8, 10)]}

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, trade_dates_by_symbol, self._TODAY
        )

        assert applied == set()
        assert unresolved == set()

    def test_trade_outside_window_does_not_suppress_a_real_split(self):
        # A trade exists for the symbol, but well before the pre-ex-date
        # snapshot -- it must not disqualify a genuine split confirmation.
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}}),
        ]
        trade_dates_by_symbol = {"MNST": [date(2026, 7, 1)]}

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, trade_dates_by_symbol, self._TODAY
        )

        assert applied == {"MNST"}

    def test_missing_snapshot_before_ex_date_is_unresolved(self):
        # No snapshot exists before the ex-date (e.g. history starts after
        # it, or the position was only bought after) -- there is no
        # before/after evidence to compare, so this must fall back to the
        # caller's default behavior (correct + log a warning), not guess.
        snapshot_history = [
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == {"MNST"}

    def test_missing_snapshot_after_ex_date_is_unresolved(self):
        # No snapshot exists on/after the ex-date yet (e.g. the split just
        # happened and tomorrow's run hasn't logged a snapshot past it).
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == {"MNST"}

    def test_no_snapshot_history_at_all_is_unresolved(self):
        applied, unresolved = find_split_applied_symbols(self._CORP_SPLITS, [], {}, self._TODAY)

        assert applied == set()
        assert unresolved == {"MNST"}

    def test_symbol_not_held_before_ex_date_is_unresolved(self):
        # A snapshot exists before the ex-date, but the symbol wasn't held
        # then (e.g. bought later) -- no qty to compare, still unresolved
        # rather than a false "applied".
        snapshot_history = [
            (date(2026, 8, 8), {"VZ": {"qty": 77.05}}),
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == {"MNST"}

    def test_future_ex_date_split_is_ignored(self):
        # The split hasn't happened yet as of `today` -- nothing to
        # confirm either way.
        future_splits = {"MNST": [(date(2026, 9, 1), 2.0)]}
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}}),
            (date(2026, 8, 12), {"MNST": {"qty": 20.80410098}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            future_splits, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == set()

    def test_qty_within_tolerance_of_factor_is_applied(self):
        # Fractional-share rounding: qty jumps to 4.9% shy of an exact
        # double, still within the 5% tolerance.
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.0}}),
            (date(2026, 8, 12), {"MNST": {"qty": 38.02}}),  # ratio 1.901, factor 2.0
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == {"MNST"}

    def test_qty_outside_tolerance_of_factor_is_not_applied(self):
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.0}}),
            (date(2026, 8, 12), {"MNST": {"qty": 30.0}}),  # ratio 1.5, nowhere near 2.0
        ]

        applied, unresolved = find_split_applied_symbols(
            self._CORP_SPLITS, snapshot_history, {}, self._TODAY
        )

        assert applied == set()
        assert unresolved == set()

    def test_multiple_symbols_resolved_independently(self):
        corp_splits = {
            "MNST": [(self._EX_DATE, 2.0)],
            "XYZ": [(self._EX_DATE, 0.1)],
        }
        snapshot_history = [
            (date(2026, 8, 8), {"MNST": {"qty": 20.80410098}, "XYZ": {"qty": 100.0}}),
            (date(2026, 8, 12), {"MNST": {"qty": 41.60820196}, "XYZ": {"qty": 100.0}}),
        ]

        applied, unresolved = find_split_applied_symbols(
            corp_splits, snapshot_history, {}, self._TODAY
        )

        assert applied == {"MNST"}
        assert unresolved == set()
