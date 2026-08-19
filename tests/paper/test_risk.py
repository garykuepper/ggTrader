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
    weights = {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2}
    # 3 positive-weight sleeves * 30 slots each = 90 global cap.
    assert guard.max_new_positions(0, weights) == 90
    assert guard.max_new_positions(85, weights) == 5
    assert guard.max_new_positions(90, weights) == 0
    assert guard.max_new_positions(95, weights) == 0


def test_max_new_positions_ignores_zero_weight_sleeves(guard):
    weights = {"sp500": 0.5, "midcap400": 0.5, "dead_sleeve": 0.0}
    assert guard.max_new_positions(0, weights) == 60


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


def test_sleeve_slot_caps_equal_regardless_of_weight(guard):
    # Every positive-weight sleeve gets the SAME slot budget -- unequal
    # weights must NOT produce unequal caps. This is the regression test for
    # the bug: weight already scales dollar size per position
    # (sleeve_position_notional), so applying it again here squared its
    # effect on max deployable exposure.
    caps = guard.sleeve_slot_caps({"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2})
    assert caps["sp500"] == caps["midcap400"] == caps["nasdaq100"] == 30


def test_sleeve_slot_caps_zero_weight_sleeve_gets_zero_slots(guard):
    caps = guard.sleeve_slot_caps({"sp500": 0.7, "midcap400": 0.3, "dead_sleeve": 0.0})
    assert caps["dead_sleeve"] == 0
    assert caps["sp500"] == caps["midcap400"] == 30


def test_sleeve_slot_caps_total_invariant():
    # sum(caps) == max_positions * n_positive_sleeves, exactly (no trimming).
    cfg = RiskConfig(max_positions=10)
    guard = RiskGuard(cfg)
    weights = {f"sleeve_{i}": 1 / 6 for i in range(6)}
    caps = guard.sleeve_slot_caps(weights)
    assert sum(caps.values()) == 60
    assert all(cap == 10 for cap in caps.values())


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


# --- Regression fixture: the live 2026-08-03 rebalance state -----------------
# sp500 0.2855, midcap400 0.2808, nasdaq100 0.4337 (sums to 1.0), scale 0.7574.
# Before the fix this state reached only ~26% deployment at full slots
# (weight applied twice: once in sleeve_slot_caps, once in
# sleeve_position_notional). After the fix it must reach ~75% -- scale *
# sum(weights) -- since weight is now applied exactly once.
LIVE_WEIGHTS = {"sp500": 0.2855, "midcap400": 0.2808, "nasdaq100": 0.4337}
LIVE_SCALE = 0.7574
LIVE_PORTFOLIO_VALUE = 103_800.0


def test_sleeve_position_notional_matches_live_state_per_symbol_dollars(guard):
    # Per-position dollar amounts stay exactly what they are today -- a
    # future refactor must not silently resize positions while fixing slot
    # counts. Values from the bug report: $741 sp500 / $729 midcap /
    # $1,126 nasdaq100 at the live 2026-08-03 state.
    sp500 = guard.sleeve_position_notional(LIVE_PORTFOLIO_VALUE, LIVE_WEIGHTS["sp500"], LIVE_SCALE)
    midcap = guard.sleeve_position_notional(
        LIVE_PORTFOLIO_VALUE, LIVE_WEIGHTS["midcap400"], LIVE_SCALE
    )
    nasdaq = guard.sleeve_position_notional(
        LIVE_PORTFOLIO_VALUE, LIVE_WEIGHTS["nasdaq100"], LIVE_SCALE
    )
    assert sp500 == pytest.approx(741, abs=1)
    assert midcap == pytest.approx(729, abs=1)
    assert nasdaq == pytest.approx(1126, abs=1)


def test_full_deployment_matches_scale_times_sum_weights_not_sum_weights_squared(guard):
    # The 26%-vs-75% regression test. Full deployment (every slot filled in
    # every sleeve) must land at ~scale * sum(weights) of portfolio value,
    # not the old scale * position_pct * sum(weight_i**2) (~26%).
    caps = guard.sleeve_slot_caps(LIVE_WEIGHTS)
    total_deployed = sum(
        caps[sleeve] * guard.sleeve_position_notional(LIVE_PORTFOLIO_VALUE, weight, LIVE_SCALE)
        for sleeve, weight in LIVE_WEIGHTS.items()
    )
    deployed_fraction = total_deployed / LIVE_PORTFOLIO_VALUE

    # position_pct * max_positions = 0.033 * 30 = 0.99, so full deployment is
    # ~99% of scale * sum(weights), not exactly 100% (floor(1/position_pct)).
    expected_fraction = LIVE_SCALE * sum(LIVE_WEIGHTS.values())
    assert deployed_fraction == pytest.approx(expected_fraction, rel=0.02)
    assert 0.70 < deployed_fraction < 0.80  # sanity band from the bug report

    # The old (buggy) formula for comparison -- must NOT match the new one.
    # Old cap_i ~= weight_i * max_positions, so
    # old deployment = scale * (position_pct * max_positions) * sum(weight_i**2).
    old_buggy_fraction = (
        LIVE_SCALE
        * (guard.cfg.position_pct * guard.cfg.max_positions)
        * sum(w**2 for w in LIVE_WEIGHTS.values())
    )
    assert 0.20 < old_buggy_fraction < 0.32
    assert deployed_fraction > 2 * old_buggy_fraction


def test_global_cap_matches_live_weights_sleeve_count(guard):
    assert guard.max_new_positions(0, LIVE_WEIGHTS) == 90
