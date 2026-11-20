"""Risk assessment preserves halts and notice boundaries."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ggTrader.paper.models import RunContext
from ggTrader.paper.risk import RiskGuard
from ggTrader.paper.risk_assessor import RiskAssessor


def setup_assessor(*, peak=None, value=100.0, last=100.0, pending=(), splits=None):
    risk = RiskGuard()
    ledger, notifier = MagicMock(), MagicMock()
    ledger.get_peak_value.return_value = peak
    assessor = RiskAssessor(risk=risk, ledger=ledger, notifier=notifier)
    ctx = RunContext(date(2026, 1, 1), MagicMock(), frozenset(pending))
    view = SimpleNamespace(
        portfolio_value=value, account={"last_equity": last}, split_corrections=splits or {}
    )
    return assessor, ctx, view, ledger, notifier


def test_assess_seeds_peak_and_blocks_buys_on_drawdown():
    assessor, ctx, view, ledger, notifier = setup_assessor(peak=200)
    verdict = assessor.assess(ctx, view)
    assert verdict.hard_stop is None
    assert verdict.buy_halts
    ledger.save_peak_value.assert_not_called()
    assert "HALTED buys" in notifier.send.call_args.args[0]


def configure_failure(failure, view, ledger):
    if failure == "read":
        ledger.get_peak_value.side_effect = RuntimeError("read failed")
    elif failure == "stored_peak":
        ledger.get_peak_value.return_value = -1
    elif failure == "value":
        view.portfolio_value = float("nan")
    elif failure == "write":
        ledger.save_peak_value.side_effect = RuntimeError("write failed")
    elif failure == "missing_last":
        view.account["last_equity"] = None
    else:
        view.account["last_equity"] = -1


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("read", "Risk history unavailable: could not read persisted peak"),
        ("stored_peak", "Invalid stored risk peak: Invalid portfolio value for risk peak"),
        ("value", "Invalid portfolio value; drawdown cannot be assessed"),
        ("write", "Risk history unavailable: could not persist drawdown peak"),
        ("missing_last", "Invalid broker previous-close equity; daily loss cannot be assessed"),
        ("invalid_last", "Invalid portfolio value or daily baseline; loss cannot be assessed"),
    ],
)
def test_hard_stops_send_original_notice(failure, reason, caplog):
    assessor, ctx, view, ledger, notifier = setup_assessor()
    configure_failure(failure, view, ledger)
    result = assessor.assess(ctx, view)
    assert result.hard_stop == reason
    notifier.send.assert_called_once_with(f"<b>🛑 HALTED:</b> {reason}")
    if failure == "read":
        assert "Could not load persisted peak: read failed" in caplog.text
    if failure == "write":
        assert "Could not persist peak value: write failed" in caplog.text


def test_split_note_names_symbols_sorted_even_with_non_string_keys():
    assessor, ctx, view, _, _ = setup_assessor(last=200, splits={"Z": 1, 3: 2, "A": 1})
    verdict = assessor.assess(ctx, view)
    assert "3, A, Z" in verdict.buy_halts[0]


def test_loss_halt_notices_fail_soft_and_the_pending_notice_still_goes():
    assessor, ctx, view, _, notifier = setup_assessor(last=200, pending=["ABC"])
    notifier.send.side_effect = [RuntimeError("failed"), None]
    assert assessor.assess(ctx, view).buy_halts
    assert notifier.send.call_count == 2
    assert "Pending orders" in notifier.send.call_args.args[0]


@pytest.mark.parametrize(
    "failure", ["read", "stored_peak", "value", "write", "missing_last", "invalid_last"]
)
def test_hard_stop_notice_failure_raises_through(failure):
    assessor, ctx, view, ledger, notifier = setup_assessor()
    configure_failure(failure, view, ledger)
    notifier.send.side_effect = RuntimeError("notice failed")
    with pytest.raises(RuntimeError, match="notice failed"):
        assessor.assess(ctx, view)


def test_assess_uses_the_guard_assigned_after_construction():
    assessor, ctx, view, _, _ = setup_assessor()
    assessor.risk = RiskGuard()
    assessor.risk.update_peak(200)
    assert assessor.assess(ctx, view).buy_halts


def test_peak_is_saved_only_when_no_loss_halt():
    assessor, ctx, view, ledger, _ = setup_assessor()
    assert not assessor.assess(ctx, view).buy_halts
    ledger.save_peak_value.assert_called_once_with(100)


def test_daily_loss_alone_still_saves_peak():
    assessor, ctx, view, ledger, _ = setup_assessor(last=200)
    assert assessor.assess(ctx, view).buy_halts == ("Daily loss limit: 50.0% ($200.00 → $100.00)",)
    ledger.save_peak_value.assert_called_once_with(100)


def test_both_loss_halts_keep_order_and_joined_notice():
    assessor, ctx, view, ledger, notifier = setup_assessor(peak=200, last=200)
    expected = (
        "Max drawdown breached: 50.0% (peak $200.00 → $100.00)",
        "Daily loss limit: 50.0% ($200.00 → $100.00)",
    )
    assert assessor.assess(ctx, view).buy_halts == expected
    notifier.send.assert_called_once_with(
        f"<b>🛑 HALTED buys; exits only:</b> {'; '.join(expected)}"
    )
    ledger.save_peak_value.assert_not_called()


def test_pending_notice_failure_is_swallowed(caplog):
    assessor, ctx, view, _, notifier = setup_assessor(last=200, pending=["ABC"])
    notifier.send.side_effect = [None, RuntimeError("pending notice")]
    assert assessor.assess(ctx, view).buy_halts
    assert notifier.send.call_count == 2
    assert "Could not send pending-order notice" in caplog.text


def test_pending_notice_requires_a_loss_halt_and_pending_orders():
    assessor, ctx, view, _, notifier = setup_assessor(last=200)
    assert assessor.assess(ctx, view).buy_halts
    notifier.send.assert_called_once()
    assessor, ctx, view, _, notifier = setup_assessor(pending=["ABC"])
    assert not assessor.assess(ctx, view).buy_halts
    notifier.send.assert_not_called()


def test_trader_assesses_with_guard_assigned_after_construction(monkeypatch):
    from ggTrader.paper.trader import PaperTrader
    from tests.paper.transcript.harness import run_scenario
    from tests.paper.transcript.scenarios import SCENARIOS

    original_run = PaperTrader._run

    def run_with_replacement(self, *, stop_threshold):
        self._risk = RiskGuard()
        self._risk.update_peak(200_000)
        return original_run(self, stop_threshold=stop_threshold)

    monkeypatch.setattr(PaperTrader, "_run", run_with_replacement)
    result = run_scenario(SCENARIOS["drawdown_halt"])
    assert result["raised"] is None
    assert "50.0%" in result["result"]["errors"][0]
