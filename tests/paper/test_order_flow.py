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
    from ggTrader.paper.models import FlowState

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
    flow.executor.submit.side_effect = lambda intent: events.append("submit")

    def announce(intent):
        assert len(made[0].book.orders) == 1
        assert made[0].book.orders[0].order_id is None
        events.append("announce")

    flow.executor.announce.side_effect = announce
    flow.run_stage_one(view, signals, RiskVerdict(None, ()))
    assert events == ["submit", "market_value", "announce"]
