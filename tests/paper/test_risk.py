"""Tests for risk guardrails."""

from __future__ import annotations

import pytest

from ggTrader.paper.risk import RiskConfig, RiskGuard


@pytest.fixture
def guard():
    cfg = RiskConfig(
        max_positions=30,
        position_pct=0.033,
        max_concentration_pct=0.05,
        daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
    )
    return RiskGuard(cfg)


def test_position_notional(guard):
    assert guard.position_notional(1000.0) == 33.0
    assert guard.position_notional(10000.0) == 330.0


def test_max_new_positions(guard):
    assert guard.max_new_positions(0) == 30
    assert guard.max_new_positions(25) == 5
    assert guard.max_new_positions(30) == 0
    assert guard.max_new_positions(35) == 0


def test_drawdown_halt_not_triggered(guard):
    guard.update_peak(1000.0)
    halted, _ = guard.check_drawdown_halt(900.0)
    assert not halted


def test_drawdown_halt_triggered(guard):
    guard.update_peak(1000.0)
    halted, reason = guard.check_drawdown_halt(840.0)
    assert halted
    assert "Max drawdown" in reason


def test_drawdown_peak_updates(guard):
    guard.update_peak(1000.0)
    guard.update_peak(1100.0)
    halted, _ = guard.check_drawdown_halt(950.0)
    assert not halted
    halted, _ = guard.check_drawdown_halt(930.0)
    assert halted


def test_daily_loss_not_triggered(guard):
    halted, _ = guard.check_daily_loss(980.0, 1000.0)
    assert not halted


def test_daily_loss_triggered(guard):
    halted, reason = guard.check_daily_loss(960.0, 1000.0)
    assert halted
    assert "Daily loss" in reason


def test_concentration_no_existing_position(guard):
    assert not guard.check_concentration("AAPL", {}, 1000.0)


def test_concentration_below_limit(guard):
    positions = {"AAPL": {"market_value": 40.0}}
    assert not guard.check_concentration("AAPL", positions, 1000.0)


def test_concentration_above_limit(guard):
    positions = {"AAPL": {"market_value": 60.0}}
    assert guard.check_concentration("AAPL", positions, 1000.0)


def test_sleeve_slot_caps_proportional(guard):
    caps = guard.sleeve_slot_caps({"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2})
    assert caps["sp500"] == 15
    assert caps["midcap400"] == 9
    assert caps["nasdaq100"] == 6
    assert sum(caps.values()) <= guard.cfg.max_positions


def test_sleeve_slot_caps_minimum_one_slot_for_small_weight():
    cfg = RiskConfig(max_positions=10)
    guard = RiskGuard(cfg)
    caps = guard.sleeve_slot_caps({"sp500": 0.95, "midcap400": 0.03, "nasdaq100": 0.02})
    assert caps["midcap400"] >= 1
    assert caps["nasdaq100"] >= 1
    assert sum(caps.values()) <= 10


def test_sleeve_slot_caps_overflow_correction_holds_invariant_six_sleeves():
    # Reviewer-reported failing case: 6 sleeves at equal weight ~0.1667 with
    # max_positions=5. Every sleeve's floor is max(1, int(0.1667*5)) = 1, so
    # the naive total is 6 -- one over budget. The old one-shot subtraction
    # from the single largest-weight sleeve, followed by a clamp back up to
    # a minimum of 1, silently undid the correction and left total == 6.
    cfg = RiskConfig(max_positions=5)
    guard = RiskGuard(cfg)
    weights = {f"sleeve_{i}": 1 / 6 for i in range(6)}
    caps = guard.sleeve_slot_caps(weights)
    assert sum(caps.values()) <= 5


def test_sleeve_position_notional_fixed_fraction_of_sleeve_capital(guard):
    # portfolio_value * sleeve_weight * scale * position_pct(0.033) --
    # independent of how many signals fire that day.
    notional = guard.sleeve_position_notional(portfolio_value=10000.0, sleeve_weight=0.4, scale=0.9)
    assert notional == round(10000.0 * 0.4 * 0.9 * 0.033, 2)


def test_sleeve_position_notional_zero_weight_is_zero(guard):
    notional = guard.sleeve_position_notional(portfolio_value=10000.0, sleeve_weight=0.0, scale=0.9)
    assert notional == 0.0


def test_sleeve_position_notional_matches_flat_when_full_weight_full_scale(guard):
    # Sanity check: a single sleeve at weight=1.0, scale=1.0 must reproduce
    # today's flat position_notional exactly -- this is the degenerate case
    # a 1-sleeve "blend" should collapse back to current live behavior.
    sleeve_notional = guard.sleeve_position_notional(
        portfolio_value=10000.0, sleeve_weight=1.0, scale=1.0
    )
    assert sleeve_notional == guard.position_notional(10000.0)
