"""Stage-one order flow behavior."""

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


def test_sweep_sell_is_planned_after_the_stops_were_executed(monkeypatch):
    from ggTrader.paper import order_flow
    from ggTrader.paper.models import IntentKind, OrderIntent

    flow, view, signals = make_flow({"A": position(), "B": position()}, sells=("B",))
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
def test_the_flow_builds_each_sell_order_after_its_counters_and_market_value_read(dry):
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
        assert [o.order_id for o in state.book.orders] == ([] if dry else ["oid"])
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
