"""Order flow behavior across execution stages."""

from unittest.mock import MagicMock

import pytest

from ggTrader.paper.models import RiskVerdict
from ggTrader.paper.order_flow import OrderFlow


def test_exit_validation_without_qty_raises_after_the_stop_orders():
    executor = MagicMock(submits_orders=True)
    executor.submit.return_value = "order"
    view = MagicMock()
    view.strategy_positions = {"A": {"qty": 1, "market_value": 10}, "B": {}}
    view.corrected_positions = {
        "A": {"qty": 1, "market_value": 10, "cost_basis": 20, "unrealized_pl": -10}
    }
    view.buy_candidates = ()
    view.sells = ("B",)
    view.portfolio_value = 100
    signals = MagicMock(weights={"s": 1}, scale=1)
    ctx = MagicMock(pending_symbols=frozenset())
    flow = OrderFlow(
        executor=executor,
        broker=MagicMock(),
        notifier=MagicMock(),
        risk=MagicMock(),
        ctx=ctx,
        stop_threshold=-0.1,
    )
    with pytest.raises(KeyError, match="qty"):
        flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    executor.submit.assert_called_once()


def make_flow(positions, *, dry=False, sells=(), buys=(("BUY", "s"),)):
    executor = MagicMock(submits_orders=not dry)
    executor.submit.return_value = None
    risk = MagicMock()
    risk.sleeve_slot_caps.return_value = {"s": 1}
    risk.sleeve_position_notional.return_value = 100
    view = MagicMock()
    view.strategy_positions = positions
    view.corrected_positions = positions
    view.buy_candidates = buys
    view.sells = sells
    view.portfolio_value = 1000
    signals = MagicMock(weights={"s": 1}, scale=1)
    ctx = MagicMock(pending_symbols=frozenset())
    flow = OrderFlow(
        executor=executor,
        broker=MagicMock(),
        notifier=MagicMock(),
        risk=risk,
        ctx=ctx,
        stop_threshold=-0.1,
    )
    return flow, view, signals


def position(qty=1, **extra):
    return {"qty": qty, "market_value": 10, "cost_basis": 20, "unrealized_pl": -10, **extra}


@pytest.mark.parametrize("dry", [False, True])
def test_a_stop_sell_is_booked_then_announced_in_that_order(dry):
    flow, view, signals = make_flow({"A": position()}, dry=dry)
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert state.book.sells == ["A"]
    assert state.stopped == {"A"}
    assert state.strategy_positions == {}
    assert len(state.book.orders) == (0 if dry else 1)
    if not dry:
        assert state.book.orders[0].order_id is None
    flow.executor.announce.assert_called_once()


def test_unassessable_blocks_buys_without_blocking_assessable_stops():
    flow, view, signals = make_flow({"A": position(), "B": position(cost_basis=0)})
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert state.stopped == {"A"}
    assert not state.all_buys
    assert state.loss_halts == ["Cannot assess catastrophe stop for: B"]
    flow.notifier.send.assert_called_once()


def test_invalid_stop_quantity_blocks_buys_not_other_stops():
    flow, view, signals = make_flow({"A": position(qty=0), "B": position()})
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert state.stopped == {"B"}
    assert state.invalid_exit_symbols == {"A"}
    assert state.all_buys == []
    assert len(state.book.errors) == 1


def test_stop_stage_submits_before_it_plans_the_next_symbol():
    flow, view, signals = make_flow({"A": position(), "B": {**position()}})
    del view.strategy_positions["B"]["qty"]
    with pytest.raises(KeyError, match="qty"):
        flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.executor.submit.assert_called_once()


def test_exit_validation_runs_after_the_stops_and_before_the_limits():
    flow, view, signals = make_flow({"A": position(), "B": {**position(qty=0)}}, sells=("B",))
    view.corrected_positions = {"A": position()}
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert state.invalid_exit_symbols == {"B"}
    assert state.buys_by_sleeve == {}
    flow.executor.submit.assert_called_once()
    flow.risk.sleeve_slot_caps.assert_called_once()


def test_flow_state_after_stage_one_holds_every_field_the_inline_remainder_reads():
    flow, view, signals = make_flow({})
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert state.sleeve_open_initial == {"s": 0}
    assert state.buys_by_sleeve == {"s": ["BUY"]}
    assert state.sleeve_notional_by_universe == {"s": 100}
    assert not hasattr(state, "sleeve_open_count")
    assert not hasattr(state, "buys_attempted")


@pytest.mark.parametrize("dry", [False, True])
def test_stop_market_value_is_read_after_submit_and_never_in_dry_run(dry):
    events = []

    class Position(dict):
        def __getitem__(self, key):
            if key == "market_value":
                events.append("market_value")
            return super().__getitem__(key)

    flow, view, signals = make_flow({"A": Position(position())}, dry=dry)
    flow.executor.submit.side_effect = lambda intent: events.append("submit")
    flow.executor.announce.side_effect = lambda intent: events.append("announce")
    flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert events == (["submit", "announce"] if dry else ["submit", "market_value", "announce"])


def test_limits_are_computed_after_validation_and_in_order():
    flow, view, signals = make_flow({})
    events = []
    flow.risk.sleeve_slot_caps.side_effect = lambda weights: events.append("caps") or {"s": 1}
    flow.risk.sleeve_position_notional.side_effect = lambda *args: events.append("notional") or 10
    flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert events == ["caps", "notional"]


def test_the_flow_builds_the_submitted_order_from_the_returned_id_after_the_counters(monkeypatch):
    from ggTrader.paper import order_flow
    from ggTrader.paper.catastrophe_stop import CATASTROPHE_STOP_REASON
    from ggTrader.paper.models import FlowState, SubmittedOrder

    made = []
    events = []

    def new_state(**kwargs):
        state = FlowState(**kwargs)
        made.append(state)
        return state

    class Position(dict):
        def __getitem__(self, key):
            if key == "market_value":
                assert made[0].stopped == {"A"}
                assert made[0].book.sells == ["A"]
                events.append("market_value")
            return super().__getitem__(key)

    monkeypatch.setattr(order_flow, "FlowState", new_state)
    flow, view, signals = make_flow({"A": Position(position())})
    flow.executor.submit.side_effect = lambda intent: events.append("submit") or "oid-1"

    def announce(intent):
        assert len(made[0].book.orders) == 1
        assert made[0].book.orders[0] == SubmittedOrder(
            "oid-1", "SELL", "A", 10, CATASTROPHE_STOP_REASON
        )
        events.append("announce")

    flow.executor.announce.side_effect = announce
    flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert events == ["submit", "market_value", "announce"]


@pytest.mark.parametrize("dry", [False, True])
def test_stop_is_recorded_before_announcement(monkeypatch, dry):
    from ggTrader.paper import order_flow
    from ggTrader.paper.models import FlowState

    states = []

    def new_state(**kwargs):
        state = FlowState(**kwargs)
        states.append(state)
        return state

    monkeypatch.setattr(order_flow, "FlowState", new_state)
    flow, view, signals = make_flow({"A": position()}, dry=dry)

    def announce(_intent):
        assert states[0].stopped == {"A"}
        assert states[0].book.sells == ["A"]
        assert len(states[0].book.orders) == (0 if dry else 1)

    flow.executor.announce.side_effect = announce
    flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.executor.announce.assert_called_once()


@pytest.mark.parametrize("failing", ["sleeve_slot_caps", "sleeve_position_notional"])
def test_limit_failure_occurs_after_the_stop_was_submitted(failing):
    flow, view, signals = make_flow({"A": position()})
    getattr(flow.risk, failing).side_effect = ValueError("limits")
    with pytest.raises(ValueError, match="limits"):
        flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.executor.submit.assert_called_once()
    flow.executor.announce.assert_called_once()


def test_unassessable_and_pending_notices_precede_stop_even_if_first_fails():
    flow, view, signals = make_flow({"A": position(), "B": position(cost_basis=0)})
    flow.ctx.pending_symbols = frozenset({"P"})
    events = []

    def notify(message):
        events.append(message)
        if len(events) == 1:
            raise RuntimeError("notice")

    flow.notifier.send.side_effect = notify
    flow.executor.submit.side_effect = lambda _intent: events.append("submit") or "oid-1"
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert "HALTED buys" in events[0]
    assert "Pending orders may still fill" in events[1]
    assert events[2] == "submit"
    assert state.stopped == {"A"} and state.all_buys == []


def test_stage_two_polls_sells_and_blocks_buys_on_unfilled_exit():
    flow, view, signals = make_flow({"A": position()}, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    view.positions = view.strategy_positions
    flow.executor.poll.return_value = {"oid": {"status": "canceled"}}
    flow.executor.submit.return_value = "oid"
    flow.executor.unfilled_exit_errors.return_value = ["SELL A not fully filled (canceled)"]
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    assert state.regular_sell_count == 1
    assert state.book.sells == ["A"]
    assert state.buys_by_sleeve == {}
    assert state.book.errors == ["SELL A not fully filled (canceled)"]
    flow.executor.poll.assert_called_once_with(state.book.orders)
    flow.executor.unfilled_exit_errors.assert_called_once_with(
        state.book.orders, state.book.filled, view.positions
    )


def _check_stop_sweep_and_regular_sells(monkeypatch):
    from ggTrader.paper import order_flow
    from ggTrader.paper.models import IntentKind, OrderIntent

    flow, view, signals = make_flow({"A": position(), "B": position()}, sells=("B",))
    flow.executor.submit.side_effect = ["stop-oid", "sweep-oid", "sell-oid"]
    view.corrected_positions = {"A": position()}
    view.sweep_on = True
    view.positions = view.strategy_positions
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))

    def sweep(**kwargs):
        assert set(kwargs["strategy_positions"]) == {"B"}
        assert state.book.sells == ["A"]
        return OrderIntent(IntentKind.SWEEP_SELL, "SELL", "SGOV", 2, "sweep", 20)

    monkeypatch.setattr(order_flow, "plan_sweep_sell", sweep)
    flow.executor.poll.return_value = {}
    flow.executor.unfilled_exit_errors.return_value = []
    flow.run_stage_two(view, state)
    assert state.book.sells == ["A", "B"]
    assert state.regular_sell_count == 1
    assert [o.symbol for o in state.book.orders] == ["A", "SGOV", "B"]
    assert state.book.orders[1].order_id == "sweep-oid"


def test_sweep_sell_is_planned_after_the_stops_were_executed(monkeypatch):
    _check_stop_sweep_and_regular_sells(monkeypatch)


def test_regular_sell_count_excludes_stop_and_sweep_sells(monkeypatch):
    _check_stop_sweep_and_regular_sells(monkeypatch)


def test_a_working_exit_raises_in_reconciliation_before_any_error_is_built():
    flow, view, signals = make_flow({"A": position()}, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    flow.executor.poll.return_value = {"oid": {"status": "new"}}
    flow.executor.after_sells.side_effect = RuntimeError("working")
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    with pytest.raises(RuntimeError, match="working"):
        flow.run_stage_two(view, state)
    flow.executor.unfilled_exit_errors.assert_not_called()


def test_dry_run_polls_an_empty_list_and_reconciles_nothing():
    flow, view, signals = make_flow({"A": position()}, dry=True, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    flow.executor.poll.return_value = {}
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    flow.executor.poll.assert_called_once_with([])
    flow.executor.after_sells.assert_called_once_with({})
    flow.executor.unfilled_exit_errors.assert_not_called()
    assert state.regular_sell_count == 1


def sweep_view(flow, view):
    """Set up a funded sweep position with cash below the reserve."""
    view.sweep_on = True
    view.positions = {**view.strategy_positions, "SGOV": {"market_value": 500, "current_price": 10}}
    view.account = {"cash": 0}
    flow.ctx.config.sweep_symbol = "SGOV"
    flow.ctx.config.reserve_pct.return_value = 0.05
    flow.risk.max_new_positions.return_value = 1


def test_a_bad_reserve_raises_after_the_stop_orders_went_out():
    flow, view, signals = make_flow({"A": position()})
    sweep_view(flow, view)
    flow.ctx.config.reserve_pct.side_effect = ValueError("bad reserve")
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    with pytest.raises(ValueError, match="bad reserve"):
        flow.run_stage_two(view, state)
    assert state.book.sells == ["A"]
    flow.executor.submit.assert_called_once()
    flow.executor.poll.assert_not_called()


def test_an_invalid_exit_still_allows_a_reserve_restoring_sweep_sell():
    from ggTrader.paper.cash_sweep import SWEEP_TRADE_REASON

    flow, view, signals = make_flow({"A": position(qty=0)}, sells=("A",))
    flow.stop_threshold = None
    sweep_view(flow, view)
    flow.executor.poll.return_value = {}
    flow.executor.unfilled_exit_errors.return_value = []
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    assert not state.buys_by_sleeve
    assert state.invalid_exit_symbols == {"A"}
    assert len(state.book.orders) == 1
    assert state.book.orders[0].symbol == "SGOV"
    assert state.book.orders[0].amount == 50
    assert state.book.orders[0].reason == SWEEP_TRADE_REASON
    assert state.regular_sell_count == 0


@pytest.mark.parametrize("status", ["canceled", "rejected", "expired"])
def test_a_terminal_unfilled_exit_blocks_all_buys_and_names_the_status(status):
    from ggTrader.paper.execution import OrderExecutor

    flow, view, signals = make_flow({"A": position()}, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    view.positions = view.strategy_positions
    flow.executor.submit.return_value = "oid"
    flow.executor.poll.return_value = {"oid": {"status": status, "filled_qty": 0}}
    flow.executor.unfilled_exit_errors.side_effect = lambda *args: (
        OrderExecutor.unfilled_exit_errors(flow.executor, *args)
    )
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    assert state.book.errors == [f"SELL A not fully filled ({status}); new strategy buys blocked"]
    assert state.buys_by_sleeve == {}
    assert state.loss_halts == []
    assert not state.invalid_exit_symbols


def test_unfilled_exit_errors_are_given_the_raw_broker_positions():
    from ggTrader.paper.execution import OrderExecutor

    flow, view, signals = make_flow({"A": position(qty=3)}, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    view.positions = {"A": position(qty=2)}
    flow.executor.submit.return_value = "oid"
    flow.executor.poll.return_value = {"oid": {"status": "filled", "filled_qty": 2}}
    flow.executor.unfilled_exit_errors.side_effect = lambda *args: (
        OrderExecutor.unfilled_exit_errors(flow.executor, *args)
    )
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    assert state.book.errors == []
    assert state.buys_by_sleeve == {"s": ["BUY"]}


@pytest.mark.parametrize("dry", [False, True])
def test_the_flow_builds_each_sell_order_after_its_counters_and_its_market_value_read(dry):
    from ggTrader.paper.models import SubmittedOrder

    events = []
    flow, view, signals = make_flow({}, dry=dry, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    flow.executor.poll.return_value = {}
    flow.executor.unfilled_exit_errors.return_value = []
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))

    class Position(dict):
        def __getitem__(self, key):
            if key == "market_value":
                assert state.book.sells == ["A"]
                assert state.regular_sell_count == 1
                assert state.book.orders == []
                events.append("market_value")
            return super().__getitem__(key)

    state.strategy_positions["A"] = Position(position())
    flow.executor.submit.side_effect = lambda _: events.append("submit") or "oid"

    def announce(_):
        assert state.book.sells == ["A"] and state.regular_sell_count == 1
        assert state.book.orders == ([] if dry else [SubmittedOrder("oid", "SELL", "A", 10, None)])
        events.append("announce")

    flow.executor.announce.side_effect = announce
    flow.run_stage_two(view, state)
    assert events == (["submit", "announce"] if dry else ["submit", "market_value", "announce"])


def test_only_sells_are_polled_before_the_buys():
    from ggTrader.paper.models import SubmittedOrder

    flow, view, signals = make_flow({"A": position()}, sells=("A",))
    flow.stop_threshold = None
    view.sweep_on = False
    flow.executor.poll.return_value = {}
    flow.executor.unfilled_exit_errors.return_value = []
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    state.book.orders.append(SubmittedOrder("buy", "BUY", "BUY", 10, None))
    flow.run_stage_two(view, state)
    assert [o.symbol for o in flow.executor.poll.call_args.args[0]] == ["A"]


def test_an_unfilled_exit_leaves_the_sweep_buy_enabled():
    from tests.paper import rules_support as rs

    scenario = rs.from_scenario(
        "sell_partial_fill_canceled",
        env={**rs.LIVE_ENV, **rs.SWEEP_ENV},
        fills={"AAPL": [{"status": "rejected", "filled_qty": 0, "filled_avg_price": None}]},
    )
    transcript = rs.run("new", scenario)
    assert [symbol for _, symbol, _ in rs.submissions(transcript, "BUY")] == ["SPY"]
    assert transcript["result"]["buys"] == []
    assert transcript["result"]["errors"] == [
        "SELL AAPL not fully filled (rejected); new strategy buys blocked"
    ]


def test_stage_three_buys_then_settles_all_orders_and_preserves_open_counts():
    from ggTrader.paper.models import SubmittedOrder

    flow, view, signals = make_flow({})
    view.sweep_on = False
    flow.risk.max_new_positions.return_value = 2
    flow.risk.check_concentration.return_value = False
    flow.executor.poll.return_value = {}
    flow.executor.submit.return_value = "buy-id"
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    sell = SubmittedOrder("sell-id", "SELL", "OLD", 10, None)
    state.book.orders.append(sell)
    state.book.filled = {"sell-id": {"status": "filled"}}
    flow.run_stage_three(view, state)
    assert state.book.buys == ["BUY"]
    assert state.sleeve_open_initial == {"s": 0}
    assert not hasattr(state, "buys_attempted")
    assert flow.executor.poll.call_args.args[0] == [state.book.orders[1]]
    flow.executor.after_orders.assert_called_once_with(state.book.orders)
    assert state.book.orders[0] == sell


def test_live_sweep_is_polled_before_appending_and_reconciled(monkeypatch):
    from ggTrader.paper import order_flow
    from ggTrader.paper.models import IntentKind, OrderIntent, SubmittedOrder

    flow, view, signals = make_flow({}, buys=())
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    flow.executor.submit.return_value = "sweep-id"
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    monkeypatch.setattr(
        order_flow,
        "plan_sweep_buy",
        lambda **_: OrderIntent(IntentKind.SWEEP_BUY, "BUY", "SGOV", 100, "sweep", 100),
    )
    events = []
    polled = []
    fill = {"sweep-id": {"status": "filled"}}

    def poll(orders):
        events.append(("poll", len(state.book.orders), len(orders)))
        polled.extend(orders)
        return fill if orders else {}

    flow.executor.poll.side_effect = poll
    flow.executor.recover_intents.side_effect = lambda: events.append(
        ("recover", len(state.book.orders))
    )
    flow.run_stage_three(view, state)
    assert events == [("poll", 0, 0), ("poll", 0, 1), ("recover", 1)]
    assert polled == [SubmittedOrder("sweep-id", "BUY", "SGOV", 100, "cash_sweep")]
    assert state.book.orders[0] is polled[0]
    assert state.book.filled == fill


def test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy():
    flow, view, signals = make_flow({}, buys=(("A", "s"), ("B", "missing")))
    view.sweep_on = False
    flow.risk.sleeve_slot_caps.return_value = {"s": 2, "missing": 2}
    flow.risk.max_new_positions.return_value = 2
    flow.risk.check_concentration.return_value = False
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    with pytest.raises(KeyError, match="missing"):
        flow.run_stage_three(view, state)
    assert state.book.buys == ["A"]
    flow.executor.submit.assert_called_once()


def test_buy_count_and_sleeve_open_count_follow_yielded_intents(monkeypatch):
    from ggTrader.paper import order_flow

    flow, view, signals = make_flow({}, buys=(("A", "s"), ("B", "s")))
    view.sweep_on = False
    flow.risk.sleeve_slot_caps.return_value = {"s": 2}
    flow.risk.max_new_positions.return_value = 2
    flow.risk.check_concentration.return_value = False
    flow.executor.poll.return_value = {}
    original = order_flow.plan_buys
    planners = []

    def record_planner(**kwargs):
        planner = original(**kwargs)
        planners.append(planner)
        return planner

    monkeypatch.setattr(order_flow, "plan_buys", record_planner)
    counts = []
    flow.executor.submit.side_effect = lambda _: counts.append(
        (planners[0].buys_attempted, planners[0].open_by_sleeve["s"])
    )
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_three(view, state)
    assert counts == [(1, 1), (2, 2)]
    assert state.sleeve_open_initial == {"s": 0}


def test_a_halted_run_submits_no_buys_and_no_sweep_buy():
    flow, view, signals = make_flow({}, buys=(("A", "s"),))
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    flow.risk.check_concentration.return_value = False
    flow.executor.poll.return_value = {}
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ("loss halt",)))
    flow.run_stage_three(view, state)
    assert state.book.buys == []
    flow.executor.submit.assert_not_called()
    flow.broker.get_account.assert_not_called()


@pytest.mark.parametrize("blocked", ["halt", "invalid"])
def test_halt_or_invalid_exit_blocks_sweep_buy(blocked):
    flow, view, signals = make_flow({}, buys=())
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    flow.executor.poll.return_value = {}
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    if blocked == "halt":
        state.loss_halts.append("halt")
    else:
        state.invalid_exit_symbols.add("A")
    flow.run_stage_three(view, state)
    flow.broker.get_account.assert_not_called()
    flow.executor.submit.assert_not_called()


@pytest.mark.parametrize("source", ["broker", "reserve_pct", "min_clip_usd", "buy_trigger_pct"])
def test_sweep_buy_sizing_failure_logs_and_buys_nothing(source, caplog):
    flow, view, signals = make_flow({}, buys=())
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    flow.executor.poll.return_value = {}
    flow.broker.get_account.return_value = {"cash": 1000, "portfolio_value": 1000}
    for name in ("reserve_pct", "min_clip_usd", "buy_trigger_pct"):
        getattr(flow.ctx.config, name).return_value = 0.1
    if source == "broker":
        flow.broker.get_account.side_effect = ValueError("bad value")
    else:
        getattr(flow.ctx.config, source).side_effect = ValueError("bad value")
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_three(view, state)
    assert "Cash sweep buy sizing failed (non-fatal): bad value" in caplog.text
    flow.executor.submit.assert_not_called()


def test_three_held_one_stopped_one_sold_permits_two_buys():
    flow, view, signals = make_flow(
        {"A": position(), "B": position(), "C": position()},
        sells=("B",),
        buys=(("X", "s"), ("Y", "s")),
    )
    view.corrected_positions = {"A": position()}
    view.sweep_on = False
    view.positions = view.strategy_positions
    flow.risk.sleeve_slot_caps.return_value = {"s": 2}
    flow.risk.max_new_positions.return_value = 2
    flow.risk.check_concentration.return_value = False
    flow.executor.poll.return_value = {}
    flow.executor.unfilled_exit_errors.return_value = []
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_two(view, state)
    flow.run_stage_three(view, state)
    flow.risk.max_new_positions.assert_called_with(1)
    assert state.book.buys == ["X", "Y"]


def test_dry_sweep_buy_only_announces_and_does_not_enter_book(monkeypatch):
    from ggTrader.paper import order_flow
    from ggTrader.paper.execution import DryRunExecutor
    from ggTrader.paper.models import IntentKind, OrderIntent
    from tests.paper.test_execution import make_executor

    flow, view, signals = make_flow({}, dry=True, buys=())
    executor, _, notifier, _, _, seams, _ = make_executor(
        DryRunExecutor, stub=("reconcile_order_intents", "poll_orders")
    )
    seams.poll_orders.return_value = {}
    flow.executor = executor
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    monkeypatch.setattr(
        order_flow,
        "plan_sweep_buy",
        lambda **_: OrderIntent(IntentKind.SWEEP_BUY, "BUY", "SGOV", 100, "sweep", 100),
    )
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_three(view, state)
    notifier.send.assert_called_once_with("<b>🔍 DRY RUN sweep buy:</b> SGOV ($100)")
    assert state.book.orders == [] and state.book.buys == []
    seams.submit_order.assert_not_called()
    seams.reconcile_order_intents.assert_not_called()
    seams.poll_orders.assert_called_once_with([])


@pytest.mark.parametrize("dry", [False, True])
def test_buy_and_sweep_announcements_use_executor_mode(monkeypatch, dry):
    from ggTrader.paper import order_flow
    from ggTrader.paper.execution import DryRunExecutor, OrderExecutor
    from ggTrader.paper.models import IntentKind, OrderIntent
    from tests.paper.test_execution import make_executor

    flow, view, signals = make_flow({}, dry=dry)
    executor, _, notifier, _, _, seams, _ = make_executor(
        DryRunExecutor if dry else OrderExecutor, stub=("poll_orders",)
    )
    seams.submit_order.side_effect = ["buy-id", "sweep-id"]
    seams.poll_orders.return_value = {}
    flow.executor = executor
    view.sweep_on = True
    flow.risk.max_new_positions.return_value = 1
    flow.risk.check_concentration.return_value = False
    monkeypatch.setattr(
        order_flow,
        "plan_sweep_buy",
        lambda **_: OrderIntent(IntentKind.SWEEP_BUY, "BUY", "SGOV", 200, "sweep", 200),
    )
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_three(view, state)
    assert [call.args[0] for call in notifier.send.call_args_list] == (
        ["<b>🔍 DRY RUN buy:</b> BUY ($100, sleeve=s)", "<b>🔍 DRY RUN sweep buy:</b> SGOV ($200)"]
        if dry
        else []
    )


def test_submit_failure_after_saving_intent_stops_later_buys():
    from tests.paper.test_execution import make_executor

    flow, view, signals = make_flow({}, buys=(("A", "s"), ("B", "s")))
    executor, broker, _, ledger, _, seams, _ = make_executor(stub=("poll_orders",))
    seams.submit_order.side_effect = executor.submit_order

    def fail_submit(*_args):
        ledger.create_order_intent.assert_called_once()
        raise RuntimeError("submit failed")

    broker.submit_buy.side_effect = fail_submit
    flow.executor = executor
    view.sweep_on = False
    flow.risk.max_new_positions.return_value = 2
    flow.risk.sleeve_slot_caps.return_value = {"s": 2}
    flow.risk.check_concentration.return_value = False
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    with pytest.raises(RuntimeError, match="submit failed"):
        flow.run_stage_three(view, state)
    ledger.create_order_intent.assert_called_once()
    assert ledger.create_order_intent.call_args.args[3] == "A"
    broker.submit_buy.assert_called_once()
    ledger.attach_broker_order.assert_not_called()
    assert state.book.buys == []
    seams.poll_orders.assert_not_called()


def test_rejected_buy_reaches_normal_settlement():
    from tests.paper.test_execution import make_executor

    flow, view, signals = make_flow({})
    executor, _, _, _, _, seams, _ = make_executor(stub=("reconcile_order_intents", "poll_orders"))
    seams.submit_order.return_value = "buy-id"
    seams.poll_orders.return_value = {"buy-id": {"status": "rejected"}}
    flow.executor = executor
    view.sweep_on = False
    flow.risk.max_new_positions.return_value = 1
    flow.risk.check_concentration.return_value = False
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    flow.run_stage_three(view, state)
    assert state.book.filled["buy-id"]["status"] == "rejected"
    seams.reconcile_order_intents.assert_called_once_with()


@pytest.mark.parametrize("has_sell", [False, True])
def test_after_orders_settles_a_sell_only_batch_or_skips_an_empty_batch(has_sell):
    from ggTrader.paper.models import SubmittedOrder
    from tests.paper.test_execution import make_executor

    flow, view, signals = make_flow({}, buys=())
    executor, _, _, _, _, seams, _ = make_executor(stub=("reconcile_order_intents", "poll_orders"))
    seams.poll_orders.return_value = {}
    flow.executor = executor
    view.sweep_on = False
    flow.risk.max_new_positions.return_value = 1
    state = flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    if has_sell:
        state.book.orders.append(SubmittedOrder("sell-id", "SELL", "OLD", 10, None))
    flow.run_stage_three(view, state)
    assert seams.reconcile_order_intents.call_count == int(has_sell)
