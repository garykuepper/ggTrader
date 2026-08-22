"""Tests for the catastrophe-stop sizing/decision logic (pure functions)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ggTrader.paper import catastrophe_stop


class TestFlagDefaults:
    @patch.dict(os.environ, {}, clear=False)
    def test_disabled_by_default(self):
        os.environ.pop("CATASTROPHE_STOP_ENABLED", None)
        assert catastrophe_stop.catastrophe_stop_enabled() is False

    @patch.dict(os.environ, {"CATASTROPHE_STOP_ENABLED": "true"})
    def test_enabled_when_flag_set(self):
        assert catastrophe_stop.catastrophe_stop_enabled() is True

    @patch.dict(os.environ, {}, clear=False)
    def test_default_threshold_is_negative_25_pct(self):
        os.environ.pop("CATASTROPHE_STOP_PCT", None)
        assert catastrophe_stop.catastrophe_stop_pct() == pytest.approx(-0.25)

    @patch.dict(os.environ, {"CATASTROPHE_STOP_PCT": "-0.30"})
    def test_threshold_overridable_via_env(self):
        assert catastrophe_stop.catastrophe_stop_pct() == pytest.approx(-0.30)


class TestUnrealizedPct:
    def test_computed_from_cost_basis_not_broker_plpc(self):
        # Broker's own unrealized_plpc disagrees with cost_basis math here;
        # the function must use cost_basis, not trust unrealized_plpc.
        position = {"unrealized_pl": -2500.0, "cost_basis": 10000.0, "unrealized_plpc": -0.10}
        assert catastrophe_stop.unrealized_pct(position) == pytest.approx(-0.25)

    def test_zero_cost_basis_returns_none(self):
        assert catastrophe_stop.unrealized_pct({"unrealized_pl": -100.0, "cost_basis": 0.0}) is None

    def test_missing_cost_basis_returns_none(self):
        assert catastrophe_stop.unrealized_pct({"unrealized_pl": -100.0}) is None


class TestFindCatastropheStops:
    def test_fires_at_exactly_threshold(self):
        positions = {"NXPI": {"unrealized_pl": -2500.0, "cost_basis": 10000.0}}
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == ["NXPI"]

    def test_fires_below_threshold(self):
        positions = {"NXPI": {"unrealized_pl": -3000.0, "cost_basis": 10000.0}}  # -30%
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == ["NXPI"]

    def test_does_not_fire_one_point_above_threshold(self):
        # -24% is above (less severe than) the -25% floor -- must not fire.
        positions = {"NXPI": {"unrealized_pl": -2400.0, "cost_basis": 10000.0}}
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == []

    def test_gain_never_fires(self):
        positions = {"MSFT": {"unrealized_pl": 500.0, "cost_basis": 10000.0}}
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == []

    def test_multiple_symbols_only_breaching_ones_returned(self):
        positions = {
            "NXPI": {"unrealized_pl": -3000.0, "cost_basis": 10000.0},  # -30%, fires
            "MSFT": {"unrealized_pl": -1000.0, "cost_basis": 10000.0},  # -10%, no
            "AAPL": {"unrealized_pl": -2600.0, "cost_basis": 10000.0},  # -26%, fires
        }
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == ["AAPL", "NXPI"]

    def test_split_adjusted_cost_basis_no_phantom_trigger(self):
        # MNST-shaped fixture: the broker never doubled qty for a 2-for-1
        # split, but current_price/market_value ARE post-split. Naively
        # dividing the broker's raw (pre-correction) unrealized_pl by
        # cost_basis would read as a huge phantom loss. This function must
        # be called on the split-CORRECTED view (as
        # split_check.apply_corrections_to_positions produces), where
        # unrealized_pl has already been fixed relative to the
        # split-invariant cost_basis -- yielding the true (small gain), not
        # a phantom catastrophe.
        mnst_qty = 20.80410098  # pre-split qty, broker never doubled it
        cost_basis = 1887.11
        current_price = 47.77  # post-split price
        broker_market_value = mnst_qty * current_price  # 993.81..., bogus (uncorrected)
        corrected_market_value = broker_market_value * 2.0  # true value post-correction
        corrected_unrealized_pl = corrected_market_value - cost_basis  # true (small) gain

        positions = {
            "MNST": {
                "unrealized_pl": corrected_unrealized_pl,
                "cost_basis": cost_basis,
            }
        }
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == []
        # Sanity: the UNCORRECTED figure would have looked like a
        # catastrophic loss and (wrongly) fired.
        uncorrected_positions = {
            "MNST": {
                "unrealized_pl": broker_market_value - cost_basis,  # -893.30..., phantom
                "cost_basis": cost_basis,
            }
        }
        assert catastrophe_stop.find_catastrophe_stops(uncorrected_positions, -0.25) == ["MNST"]

    def test_missing_cost_basis_symbol_excluded_not_fired(self):
        positions = {"XYZ": {"unrealized_pl": -9999.0, "cost_basis": 0.0}}
        assert catastrophe_stop.find_catastrophe_stops(positions, -0.25) == []
