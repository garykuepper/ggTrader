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


@pytest.mark.parametrize(
    "failure", ["read", "stored_peak", "value", "write", "missing_last", "invalid_last"]
)
def test_hard_stops_send_original_notice(failure):
    assessor, ctx, view, ledger, notifier = setup_assessor()
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
    result = assessor.assess(ctx, view)
    assert result.hard_stop
    assert notifier.send.call_args.args[0] == f"<b>🛑 HALTED:</b> {result.hard_stop}"


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


def test_hard_stop_notice_failure_raises_through():
    assessor, ctx, view, ledger, notifier = setup_assessor()
    ledger.get_peak_value.side_effect = ValueError("failed")
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
