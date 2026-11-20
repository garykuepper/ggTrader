"""Corporate action handling keeps its failure and reconciliation behavior."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper.corporate_actions import CorporateActions, missed_exits
from ggTrader.paper.trader import PaperTrader


@pytest.fixture
def collaborators():
    ledger = MagicMock()
    ledger.get_open_split_states.return_value = {}
    ledger.get_snapshot_history.return_value = []
    ledger.get_trade_history_dates.return_value = []
    ledger.get_total_dividend_accrual.return_value = 7.0
    ledger.get_accrued_dividend_keys.return_value = set()
    broker = MagicMock()
    broker.get_split_evidence.return_value = {"corp_splits": {"A": [(date(2026, 1, 1), 2.0)]}}
    hooks = SimpleNamespace(find_split_applied=MagicMock(return_value=(set(), set())))
    return ledger, broker, hooks


@pytest.mark.parametrize(
    "method,warning",
    [
        ("get_open_split_states", "Could not load persisted split state"),
        ("get_snapshot_history", "Could not load snapshot history for split check"),
        ("get_trade_history_dates", "Could not load trade history"),
        ("save_split_correction", "Could not persist split state"),
        ("clear_split_correction", "Could not clear stale split state"),
    ],
)
def test_split_corrections_fail_soft_at_every_ledger_read(collaborators, caplog, method, warning):
    ledger, broker, hooks = collaborators
    if method == "clear_split_correction":
        ledger.get_open_split_states.return_value = {
            "OLD": {"factor": 2, "ex_date": date(2026, 1, 1)}
        }
    getattr(ledger, method).side_effect = RuntimeError("oops")
    result = CorporateActions(broker=broker, ledger=ledger, hooks=hooks).split_corrections(
        {"A": {}}, date(2025, 1, 1), date(2026, 2, 1)
    )
    assert result == {"A": 2.0}
    assert warning in caplog.text


def test_confirmed_split_clear_failure_remains_nonfatal(collaborators, caplog):
    ledger, broker, hooks = collaborators
    ledger.get_open_split_states.return_value = {"A": {"factor": 2, "ex_date": date(2026, 1, 1)}}
    ledger.clear_split_correction.side_effect = RuntimeError("oops")
    hooks.find_split_applied.return_value = ({"A"}, set())
    result = CorporateActions(broker=broker, ledger=ledger, hooks=hooks).split_corrections(
        {"A": {}}, date(2025, 1, 1), date(2026, 2, 1)
    )
    assert result == {}
    assert "Could not clear confirmed-applied split state" in caplog.text


def test_persisted_state_for_an_unheld_symbol_is_cleared(collaborators):
    ledger, broker, hooks = collaborators
    ledger.get_open_split_states.return_value = {"OLD": {"factor": 2, "ex_date": date(2026, 1, 1)}}
    CorporateActions(broker=broker, ledger=ledger, hooks=hooks).split_corrections(
        {}, date(2025, 1, 1), date(2026, 2, 1)
    )
    ledger.clear_split_correction.assert_called_once_with("OLD")


def test_applied_split_clears_the_persisted_row(collaborators):
    ledger, broker, hooks = collaborators
    ledger.get_open_split_states.return_value = {"A": {"factor": 2, "ex_date": date(2026, 1, 1)}}
    hooks.find_split_applied.return_value = ({"A"}, set())
    assert (
        CorporateActions(broker=broker, ledger=ledger, hooks=hooks).split_corrections(
            {"A": {}}, date(2025, 1, 1), date(2026, 2, 1)
        )
        == {}
    )
    ledger.clear_split_correction.assert_called_once_with("A")


@pytest.mark.parametrize(
    "method,total",
    [
        ("get_total_dividend_accrual", 0.0),
        ("get_snapshot_history", 7.0),
        ("get_accrued_dividend_keys", 7.0),
    ],
)
def test_dividend_accrual_returns_the_partial_result_on_each_failure_point(
    collaborators, method, total
):
    ledger, broker, hooks = collaborators
    broker.get_dividend_corrections.return_value = {
        "corp_dividends": {"A": [(date(2026, 1, 1), 1)]}
    }
    getattr(ledger, method).side_effect = RuntimeError("oops")
    assert CorporateActions(broker=broker, ledger=ledger, hooks=hooks).accrue_dividends(
        {"A": {}}
    ) == {"total": total, "new_accruals": [], "skipped": []}


def test_missed_exits_orders_candidates_and_survives_a_lookup_failure(collaborators):
    ledger, _, _ = collaborators
    ledger.get_last_buy_dates.return_value = {"A": date(2026, 1, 1), "B": date(2026, 1, 1)}
    assert missed_exits(
        ledger, {"B": {}, "A": {}}, {"A": "2026-01-02", "B": "2026-01-02"}, set()
    ) == ["A", "B"]
    ledger.get_last_buy_dates.side_effect = RuntimeError("oops")
    assert missed_exits(ledger, {"A": {}}, {"A": "2026-01-02"}, set()) == []


def test_split_check_reaches_the_patched_trader_name(collaborators):
    ledger, broker, _ = collaborators
    trader = PaperTrader(broker, MagicMock(), ledger=ledger, read_only=False)
    with patch(
        "ggTrader.paper.trader.find_split_applied_symbols", return_value=({"A"}, set())
    ) as find:
        assert (
            trader._compute_split_corrections({"A": {}}, date(2025, 1, 1), date(2026, 2, 1)) == {}
        )
    find.assert_called_once()


@pytest.mark.parametrize("failure", ["persist", "reload"])
def test_dividend_late_failure_preserves_accruals_and_skipped(collaborators, failure, caplog):
    ledger, broker, hooks = collaborators
    broker.get_dividend_corrections.return_value = {"corp_dividends": {"A": [object()]}}
    accrual = {"symbol": "A", "ex_date": date(2026, 1, 1), "rate": 1, "qty": 2, "amount": 2}
    skipped = {"symbol": "B", "ex_date": date(2026, 1, 1), "reason": "no snapshot"}
    if failure == "persist":
        ledger.log_dividend_accrual.side_effect = RuntimeError("oops")
    else:
        ledger.get_total_dividend_accrual.side_effect = [7.0, RuntimeError("oops")]
    with patch(
        "ggTrader.paper.corporate_actions.compute_dividend_accruals",
        return_value=([accrual], [skipped]),
    ):
        result = CorporateActions(broker=broker, ledger=ledger, hooks=hooks).accrue_dividends(
            {"A": {}}
        )
    assert result == {"total": 7.0, "new_accruals": [accrual], "skipped": [skipped]}
    assert ("Could not persist" if failure == "persist" else "Could not reload") in caplog.text
