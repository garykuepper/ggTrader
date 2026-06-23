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
