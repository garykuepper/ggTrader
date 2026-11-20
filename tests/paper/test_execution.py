"""Tests for order execution boundaries."""

from unittest.mock import MagicMock

import pytest

from ggTrader.paper.execution import DryRunExecutor, OrderExecutor
from ggTrader.paper.models import IntentKind, OrderIntent


def test_executor_submit_routes_through_the_instance_seam():
    from ggTrader.paper.trader import PaperTrader

    trader = PaperTrader(
        MagicMock(), MagicMock(), dry_run=False, membership="db", ledger=MagicMock()
    )
    executor = trader._executor_for_call()
    trader._submit_order = MagicMock(return_value="oid")
    intent = OrderIntent(IntentKind.BUY, "BUY", "ABC", 42.0, None, 42.0)
    assert executor.submit(intent) == "oid"
    trader._submit_order.assert_called_once_with("BUY", "ABC", 42.0, None)


def test_dry_run_executor_never_reaches_submit():
    seams = MagicMock()
    seams.submit_order.side_effect = AssertionError
    broker = MagicMock()
    broker.submit_buy.side_effect = AssertionError
    broker.submit_sell.side_effect = AssertionError
    executor = DryRunExecutor(
        broker=broker,
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


def make_executor(cls=OrderExecutor, *, read_only=False):
    from datetime import date, datetime, timezone

    broker, notifier, ledger, hooks, seams, lock = (MagicMock() for _ in range(6))
    hooks.today_et.return_value = date(2026, 1, 2)
    hooks.utcnow.return_value = datetime(2026, 1, 10, tzinfo=timezone.utc)
    hooks.new_uuid.return_value = "client-1"
    executor = cls(
        broker=broker,
        notifier=notifier,
        ledger=ledger,
        hooks=hooks,
        lock=lambda: lock,
        read_only=read_only,
        seams=seams,
    )
    return executor, broker, notifier, ledger, hooks, seams, lock


@pytest.mark.parametrize("stage", ["intent", "lock", "submit", "attach"])
def test_submit_order_names_the_failing_stage_in_the_notice_and_reraises(stage):
    executor, broker, notifier, ledger, _, _, lock = make_executor()
    target = {
        "intent": ledger.create_order_intent,
        "lock": lock.verify,
        "submit": broker.submit_buy,
        "attach": ledger.attach_broker_order,
    }[stage]
    error = RuntimeError(stage)
    target.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        executor.submit_order("BUY", "ABC", 12, None)
    assert caught.value is error
    notifier.send.assert_called_once_with(
        f"Order client-1 failed at {stage}; investigation required"
    )


def test_submit_order_rejects_a_side_other_than_buy_or_sell():
    executor, broker, _, ledger, _, _, _ = make_executor()
    with pytest.raises(ValueError, match="Invalid order side"):
        executor.submit_order("HOLD", "ABC", 12, None)
    ledger.create_order_intent.assert_not_called()
    broker.submit_buy.assert_not_called()


def test_submit_order_reads_the_run_date_per_call_from_the_hook():
    from datetime import date

    executor, _, _, ledger, hooks, _, _ = make_executor()
    hooks.today_et.side_effect = [date(2026, 1, 2), date(2026, 1, 3)]
    executor.submit_order("BUY", "ABC", 12, None)
    executor.submit_order("SELL", "XYZ", 3, None)
    assert [c.args[1] for c in ledger.create_order_intent.call_args_list] == [
        "2026-01-02",
        "2026-01-03",
    ]


def test_poll_orders_reads_the_clock_through_the_hooks_and_sleeps_one_second():
    from ggTrader.paper.models import SubmittedOrder

    executor, broker, _, _, hooks, _, _ = make_executor()
    hooks.clock_time.side_effect = [0, 0, 1]
    broker.get_clock.return_value = {"is_open": True}
    broker.get_order.side_effect = [{"status": "new"}, {"status": "filled"}]
    assert executor.poll_orders([SubmittedOrder("oid", "BUY", "ABC", 12, None)]) == {
        "oid": {"status": "filled"}
    }
    hooks.sleep.assert_called_once_with(1.0)


def test_poll_orders_of_no_orders_does_not_touch_the_broker():
    executor, broker, _, _, hooks, _, _ = make_executor()
    assert executor.poll_orders([]) == {}
    broker.assert_not_called()
    assert broker.mock_calls == []
    hooks.clock_time.assert_not_called()


@pytest.mark.parametrize("side,notional", [("BUY", 12), ("SELL", None)])
def test_poll_fallback_record_matches_the_old_dictionary(side, notional):
    from ggTrader.paper.models import SubmittedOrder

    executor, broker, _, _, hooks, _, _ = make_executor()
    hooks.clock_time.side_effect = [0, 3]
    broker.get_clock.return_value = {"is_open": False}
    broker.get_order.side_effect = RuntimeError("offline")
    assert executor.poll_orders([SubmittedOrder("oid", side, "ABC", 12, None)]) == {
        "oid": {
            "id": "oid",
            "symbol": "ABC",
            "side": side,
            "qty": None,
            "notional": notional,
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "status": "pending",
        }
    }


@pytest.mark.parametrize(
    "changes,exception,message",
    [
        ({"id": ""}, ValueError, "identity"),
        ({"notional": 13}, ValueError, "amount"),
        ({"filled_qty": -1}, ValueError, "fill"),
        ({"status": "filled", "filled_qty": 0}, ValueError, "no fill"),
        ({"status": "mystery"}, ValueError, "status"),
        ({"status": "new"}, RuntimeError, "working"),
    ],
)
def test_reconcile_order_intents_has_each_validation_branch(changes, exception, message):
    executor, broker, _, ledger, _, _, _ = make_executor()
    ledger.get_pending_orders.return_value = []
    ledger.get_unresolved_order_intents.return_value = [
        {
            "client_order_id": "c",
            "symbol": "ABC",
            "side": "BUY",
            "amount_kind": "notional",
            "amount": 12,
        }
    ]
    info = {
        "id": "oid",
        "client_order_id": "c",
        "symbol": "ABC",
        "side": "BUY",
        "notional": 12,
        "filled_qty": 1,
        "filled_avg_price": 12,
        "status": "filled",
    }
    info.update(changes)
    broker.get_order_by_client_id.return_value = info
    with pytest.raises(exception, match=message):
        executor.reconcile_order_intents()


def test_reconcile_pending_orders_read_only_returns_symbols_without_a_broker_call():
    executor, broker, _, ledger, _, _, _ = make_executor(read_only=True)
    ledger.get_pending_orders.return_value = [{"symbol": "ABC"}]
    assert executor.reconcile_pending_orders() == {"ABC"}
    assert broker.mock_calls == []


def test_reconcile_pending_orders_load_failure_live_notifies_and_raises():
    executor, _, notifier, ledger, _, _, _ = make_executor()
    error = RuntimeError("offline")
    ledger.get_pending_orders.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        executor.reconcile_pending_orders()
    assert caught.value is error
    assert "offline" in notifier.send.call_args.args[0]


def test_reconcile_pending_orders_load_failure_dry_warns_and_returns_empty(caplog):
    executor, _, notifier, ledger, _, _, _ = make_executor(DryRunExecutor)
    ledger.get_pending_orders.side_effect = RuntimeError("offline")
    assert executor.reconcile_pending_orders() == set()
    assert "dry-run intentions may conflict" in caplog.text
    notifier.send.assert_not_called()


def test_flag_if_stale_is_idempotent():
    from datetime import datetime, timezone

    executor, _, notifier, ledger, _, _, _ = make_executor()
    pending = {
        "created_at": datetime(2025, 1, 1),
        "side": "BUY",
        "symbol": "ABC",
        "order_id": "oid",
    }
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    executor.flag_if_stale(pending, now)
    pending["flagged_stale"] = True
    executor.flag_if_stale(pending, now)
    notifier.send.assert_called_once()
    ledger.mark_pending_order_stale.assert_called_once_with("oid")


def test_executor_uses_the_hooks_not_the_modules(monkeypatch):
    from ggTrader.paper.trader import PaperTrader

    broker, ledger = MagicMock(), MagicMock()
    trader = PaperTrader(broker, MagicMock(), dry_run=False, membership="db", ledger=ledger)
    trader._paper_lock = MagicMock()
    executor = trader._executor_for_call()
    monkeypatch.setattr("ggTrader.paper.trader.uuid4", lambda: "patched-uuid")
    executor.submit_order("BUY", "ABC", 10, None)
    assert ledger.create_order_intent.call_args.args[0] == "patched-uuid"


def test_executor_poll_routes_through_the_class_patched_seam(monkeypatch):
    from ggTrader.paper.trader import PaperTrader

    trader = PaperTrader(MagicMock(), MagicMock(), dry_run=True, ledger=MagicMock())
    patched = MagicMock(return_value={"oid": {"status": "filled"}})
    monkeypatch.setattr(PaperTrader, "_poll_orders", patched)
    assert trader._executor_for_call().poll([]) == {"oid": {"status": "filled"}}
    patched.assert_called_once_with([])


def test_executor_reconcile_routes_through_the_instance_seam():
    from ggTrader.paper.trader import PaperTrader

    trader = PaperTrader(
        MagicMock(), MagicMock(), dry_run=False, membership="db", ledger=MagicMock()
    )
    executor = trader._executor_for_call()
    trader._reconcile_order_intents = MagicMock(return_value=set())
    executor.recover_intents()
    trader._reconcile_order_intents.assert_called_once_with()


def test_the_default_seam_chain_reaches_the_moved_body_without_recursion():
    from ggTrader.paper.trader import PaperTrader

    broker, ledger = MagicMock(), MagicMock()
    broker.submit_buy.return_value = "oid"
    trader = PaperTrader(broker, MagicMock(), dry_run=False, membership="db", ledger=ledger)
    trader._paper_lock = MagicMock()
    assert (
        trader._executor_for_call().submit(OrderIntent(IntentKind.BUY, "BUY", "ABC", 12, None, 12))
        == "oid"
    )
    ledger.create_order_intent.assert_called_once()
    broker.submit_buy.assert_called_once()


@pytest.mark.parametrize("value", ["oid-1", None, MagicMock(name="order-id")])
def test_executor_submit_returns_the_seam_value_untouched(value):
    executor, broker, _, ledger, _, seams, _ = make_executor()
    seams.submit_order.return_value = value
    assert executor.submit(OrderIntent(IntentKind.BUY, "BUY", "ABC", 12, None, 12)) is value
    assert broker.mock_calls == []
    assert ledger.mock_calls == []


def test_after_sells_reconciles_only_for_a_non_empty_filled_mapping():
    executor, _, _, _, _, seams, _ = make_executor()
    executor.after_sells({})
    seams.reconcile_intents.assert_not_called()
    executor.after_sells({"oid": {"status": "filled"}})
    seams.reconcile_intents.assert_called_once_with()


def test_dry_run_after_hooks_do_nothing():
    executor, _, _, _, _, seams, _ = make_executor(DryRunExecutor)
    executor.recover_intents()
    executor.after_sells({"oid": {"status": "filled"}})
    executor.after_orders([MagicMock()])
    seams.reconcile_intents.assert_not_called()


def test_live_announce_sends_only_the_catastrophe_stop_notice():
    executor, _, notifier, _, _, _, _ = make_executor()
    for kind in IntentKind:
        executor.announce(
            OrderIntent(
                kind,
                "SELL",
                "ABC",
                3,
                None,
                20,
                sleeve="core",
                unrealized_pct=-0.3,
                stop_threshold=-0.25,
            )
        )
    notifier.send.assert_called_once_with(
        "<b>🛑 Catastrophe stop:</b> ABC unrealized -30.0% breached -25.0% floor — selling qty 3."
    )


def test_dry_run_announce_sends_the_dry_run_notice_of_each_kind():
    executor, _, notifier, _, _, _, _ = make_executor(DryRunExecutor)
    for kind in IntentKind:
        executor.announce(
            OrderIntent(
                kind,
                "SELL",
                "ABC",
                3,
                None,
                20,
                sleeve="core",
                unrealized_pct=-0.3,
                stop_threshold=-0.25,
            )
        )
    assert [c.args[0] for c in notifier.send.call_args_list] == [
        "<b>🛑 DRY RUN catastrophe stop:</b> ABC (qty 3, unrealized -30.0% breached -25.0% floor)",
        "<b>🔍 DRY RUN sell:</b> ABC (qty 3)",
        "<b>🔍 DRY RUN sweep sell:</b> ABC ($20, funding strategy buys)",
        "<b>🔍 DRY RUN buy:</b> ABC ($3, sleeve=core)",
        "<b>🔍 DRY RUN sweep buy:</b> ABC ($3)",
    ]


def test_unfilled_exit_errors_use_the_raw_broker_positions():
    from ggTrader.paper.cash_sweep import SWEEP_TRADE_REASON
    from ggTrader.paper.models import SubmittedOrder

    executor, *_ = make_executor()
    orders = [
        SubmittedOrder("one", "SELL", "ABC", 10, None),
        SubmittedOrder("two", "SELL", "XYZ", 10, None),
        SubmittedOrder("sweep", "SELL", "SGOV", 10, SWEEP_TRADE_REASON),
    ]
    filled = {
        "one": {"status": "filled", "filled_qty": 2},
        "sweep": {"status": "new", "filled_qty": 0},
    }
    assert executor.unfilled_exit_errors(
        orders, filled, {"ABC": {"qty": 3}, "XYZ": {"qty": 1}}
    ) == [
        "SELL ABC not fully filled (filled); new strategy buys blocked",
        "SELL XYZ not fully filled (unknown); new strategy buys blocked",
    ]


def test_unfilled_exit_errors_is_empty_in_a_dry_run():
    from ggTrader.paper.models import SubmittedOrder

    executor, *_ = make_executor(DryRunExecutor)
    assert (
        executor.unfilled_exit_errors([SubmittedOrder("oid", "SELL", "ABC", 1, None)], {}, {}) == []
    )


def test_full_dry_run_after_assignment_never_calls_submit_seam():
    from tests.paper.test_seam_contract import drive
    from tests.paper.transcript.scenarios import SCENARIOS

    submit = MagicMock(side_effect=AssertionError("dry run submitted"))

    def configure(trader, _run, _stack):
        trader._dry_run = True
        trader._submit_order = submit

    driven = drive("new", SCENARIOS["plain_buys_and_sells"], configure)
    assert driven.error is None
    assert driven.result["buys"] or driven.result["sells"]
    submit.assert_not_called()
