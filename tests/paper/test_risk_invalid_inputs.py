"""Reject account values that make loss-limit arithmetic meaningless."""

import pytest

from ggTrader.paper.risk import RiskGuard


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_current_equity_halts_even_without_a_peak(value):
    halted, reason = RiskGuard().check_drawdown_halt(value)
    assert halted
    assert "invalid" in reason.lower()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_peak_is_never_recorded(value):
    guard = RiskGuard()
    with pytest.raises(ValueError, match="peak"):
        guard.update_peak(value)
    assert guard.peak_value is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_daily_baseline_halts(bad):
    halted, reason = RiskGuard().check_daily_loss(100.0, bad)
    assert halted
    assert "invalid" in reason.lower()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_daily_current_equity_halts(bad):
    halted, reason = RiskGuard().check_daily_loss(bad, 100.0)
    assert halted
    assert "invalid" in reason.lower()


def test_missing_values_fail_closed():
    guard = RiskGuard()
    assert guard.check_drawdown_halt(None)[0]
    assert guard.check_daily_loss(None, 100.0)[0]
    assert guard.check_daily_loss(100.0, None)[0]
    with pytest.raises(ValueError):
        guard.update_peak(None)
