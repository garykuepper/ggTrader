"""Tests for order execution boundaries."""

from unittest.mock import MagicMock

from ggTrader.paper.execution import DryRunExecutor, OrderExecutor
from ggTrader.paper.models import IntentKind, OrderIntent


def test_executor_submit_routes_through_the_instance_seam():
    seams = MagicMock()
    executor = OrderExecutor(
        broker=MagicMock(),
        notifier=MagicMock(),
        ledger=MagicMock(),
        hooks=MagicMock(),
        lock=lambda: None,
        read_only=False,
        seams=seams,
    )
    intent = OrderIntent(IntentKind.BUY, "BUY", "ABC", 42.0, None, 42.0)
    assert executor.submit(intent) is seams.submit_order.return_value
    seams.submit_order.assert_called_once_with("BUY", "ABC", 42.0, None)


def test_dry_run_executor_never_reaches_submit():
    seams = MagicMock()
    seams.submit_order.side_effect = AssertionError
    executor = DryRunExecutor(
        broker=MagicMock(),
        notifier=MagicMock(),
        ledger=MagicMock(),
        hooks=MagicMock(),
        lock=lambda: None,
        read_only=True,
        seams=seams,
    )
    assert executor.submit(OrderIntent(IntentKind.BUY, "BUY", "ABC", 42.0, None, 42.0)) is None


def test_submit_order_saves_the_intent_then_verifies_the_lock_then_submits_then_attaches():
    from datetime import date

    from ggTrader.paper.trader import PaperTrader

    calls = MagicMock()
    trader = PaperTrader(
        calls.broker, calls.notifier, dry_run=False, membership="db", ledger=calls.ledger
    )
    trader._paper_lock = calls.lock
    trader._hooks = MagicMock(
        return_value=MagicMock(today_et=lambda: date(2026, 1, 1), new_uuid=lambda: "uuid")
    )
    calls.broker.submit_buy.return_value = "oid"
    assert trader._submit_order("BUY", "ABC", 10, None) == "oid"
    assert [call[0] for call in calls.mock_calls] == [
        "ledger.create_order_intent",
        "lock.verify",
        "broker.submit_buy",
        "ledger.attach_broker_order",
    ]


def test_after_orders_reconciles_for_a_list_holding_only_sells():
    from ggTrader.paper.models import SubmittedOrder

    seams = MagicMock()
    executor = OrderExecutor(
        broker=MagicMock(),
        notifier=MagicMock(),
        ledger=MagicMock(),
        hooks=MagicMock(),
        lock=lambda: None,
        read_only=False,
        seams=seams,
    )
    executor.after_orders([])
    seams.reconcile_intents.assert_not_called()
    executor.after_orders([SubmittedOrder("oid", "SELL", "ABC", 1, None)])
    seams.reconcile_intents.assert_called_once_with()


def test_dry_run_executor_still_polls_through_the_seam_with_an_empty_list():
    seams = MagicMock()
    executor = DryRunExecutor(
        broker=MagicMock(),
        notifier=MagicMock(),
        ledger=MagicMock(),
        hooks=MagicMock(),
        lock=lambda: None,
        read_only=True,
        seams=seams,
    )
    executor.poll(())
    seams.poll_orders.assert_called_once_with([])


def test_executors_declare_submits_orders():
    assert OrderExecutor.submits_orders is True
    assert DryRunExecutor.submits_orders is False
