"""Behavior of the pure signal and exit planners."""

import inspect
import logging
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from ggTrader.paper import catastrophe_stop, planning
from ggTrader.paper.models import IntentKind, InvalidExit, Skip


def drain(planner):
    result = []
    while (decision := planner.advance()) is not None:
        result.append(decision)
    return result


def test_strategy_signals_merges_sleeves_in_order_and_sorts_sells():
    bundle = {
        "weights": {"a": 0.5},
        "scale": 1.0,
        "sleeves": {
            "a": {
                "buys": ["Z", "A"],
                "sells": ["Z", "B"],
                "as_of": "first",
                "last_exit": {"B": "old"},
            },
            "b": {"buys": ["Z"], "sells": ["B", "A"], "as_of": "second", "last_exit": {"B": "x"}},
        },
    }
    signals = planning.strategy_signals(bundle)
    assert signals.buys == (("Z", "a"), ("A", "a"), ("Z", "b"))
    assert signals.sells == ("A", "B", "Z")
    assert signals.as_of == "first"
    assert signals.last_exit == {"B": "x"}
    assert signals.weights == {"a": 0.5} and signals.scale == 1.0


def test_empty_sleeves_raises_stop_iteration():
    with pytest.raises(StopIteration):
        planning.strategy_signals({"weights": {}, "scale": 1.0, "sleeves": {}})


def test_assess_stops_flags_unassessable_in_order(monkeypatch):
    calls = []

    def pct(position):
        calls.append(position)
        if position == 2:
            raise ValueError("bad")
        return None if position == 1 else 0.0

    monkeypatch.setattr(catastrophe_stop, "unrealized_pct", pct)
    assert planning.assess_stops({"a": 1, "b": 2, "c": 3}) == frozenset({"a", "b"})
    assert calls == [1, 2, 3]


def test_stops_validate_and_skip_before_percentage(monkeypatch):
    positions = {"a": {"qty": 2}, "b": {"qty": 0}, "c": {"qty": 3}}
    corrected = {s: {"cost_basis": 100, "unrealized_pl": -30} for s in positions}
    calls = []
    original = catastrophe_stop.unrealized_pct

    def pct(position):
        calls.append(position)
        return original(position)

    monkeypatch.setattr(catastrophe_stop, "unrealized_pct", pct)
    planner = planning.plan_stops(
        assessable=corrected,
        strategy_positions=positions,
        corrected_positions=corrected,
        threshold=-0.25,
        pending_symbols=frozenset({"c"}),
    )
    intent, invalid, skip = drain(planner)
    assert (intent.kind, intent.symbol, intent.size, intent.book_amount, intent.reason) == (
        IntentKind.STOP,
        "a",
        2,
        None,
        catastrophe_stop.CATASTROPHE_STOP_REASON,
    )
    assert (intent.unrealized_pct, intent.stop_threshold) == (-0.3, -0.25)
    assert invalid == InvalidExit(
        "b", "Cannot submit risk-reducing SELL b: invalid long quantity; new strategy buys blocked"
    )
    assert skip == Skip(logging.INFO, "Skipping catastrophe stop c: order already pending")
    assert calls == [corrected["a"], corrected["b"], corrected["c"], corrected["a"]]
    assert planner.advance() is None and planner.advance() is None


def test_stop_lookup_is_late_and_exceptions_pass_through(monkeypatch):
    error = StopIteration("percentage")
    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", lambda *_: ["a"])
    monkeypatch.setattr(catastrophe_stop, "unrealized_pct", Mock(side_effect=error))
    planner = planning.plan_stops(
        assessable={},
        strategy_positions={"a": {"qty": 1}},
        corrected_positions={"a": {}},
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    with pytest.raises(StopIteration) as caught:
        planner.advance()
    assert caught.value is error


def test_collaborator_runtime_error_passes_through(monkeypatch):
    def broken(*_):
        yield next(iter(()))

    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", broken)
    planner = planning.plan_stops(
        assessable={},
        strategy_positions={},
        corrected_positions={},
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    with pytest.raises(RuntimeError, match="generator raised StopIteration") as caught:
        planner.advance()
    assert isinstance(caught.value.__cause__, StopIteration)


def test_exit_validation_and_planning_are_lazy_and_preserve_quantities():
    class Position(dict):
        def __getitem__(self, key):
            assert key != "market_value"
            return super().__getitem__(key)

    positions = MappingProxyType({"a": Position(qty=0), "b": Position(qty=2), "c": Position()})
    invalid = planning.validate_exits(
        sells=["missing", "a", "a", "b"], strategy_positions=positions, invalid_symbols=frozenset()
    )
    assert [x.symbol for x in drain(invalid)] == ["a"]
    exits = planning.plan_exits(
        sells=["a", "b", "c"],
        strategy_positions=positions,
        invalid_symbols=frozenset({"a"}),
        pending_symbols=frozenset({"b"}),
    )
    assert exits.advance() == Skip(logging.INFO, "Skipping SELL b: order already pending")
    with pytest.raises(KeyError, match="qty"):
        exits.advance()
    exits = planning.plan_exits(
        sells=["b"],
        strategy_positions=positions,
        invalid_symbols=frozenset(),
        pending_symbols=frozenset(),
    )
    intent = exits.advance()
    assert (intent.kind, intent.size, intent.book_amount, intent.reason) == (
        IntentKind.EXIT,
        2,
        None,
        None,
    )
    assert exits.advance() is None and exits.advance() is None
    assert not inspect.isgenerator(exits)


def test_real_stop_finder_propagates_generator_runtime_error(monkeypatch):
    error = StopIteration("exhausted")
    monkeypatch.setattr(catastrophe_stop, "unrealized_pct", Mock(side_effect=error))
    planner = planning.plan_stops(
        assessable={"a": {}},
        strategy_positions={},
        corrected_positions={},
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    with pytest.raises(RuntimeError, match="generator raised StopIteration") as caught:
        planner.advance()
    assert caught.value.__cause__ is error


def test_stop_planner_defers_finder_and_next_position(monkeypatch):
    calls = []

    class Positions(dict):
        def __getitem__(self, symbol):
            calls.append(symbol)
            return super().__getitem__(symbol)

    def find(*_):
        calls.append("find")
        return ["a", "b"]

    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", find)
    positions = Positions(a={"qty": 1}, b={"qty": 2})
    corrected = {"a": {"cost_basis": 100, "unrealized_pl": -30}, "b": {"cost_basis": "bad"}}
    planner = planning.plan_stops(
        assessable=MappingProxyType(corrected),
        strategy_positions=MappingProxyType(positions),
        corrected_positions=MappingProxyType(corrected),
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    assert calls == []
    assert planner.advance().symbol == "a"
    assert calls == ["find", "a"]
    with pytest.raises(ValueError):
        planner.advance()
    assert calls == ["find", "a", "b"]


def test_no_planner_reads_market_value(monkeypatch):
    class Position(dict):
        def __getitem__(self, key):
            assert key != "market_value"
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key != "market_value"
            return super().get(key, default)

    position = Position(qty=3, cost_basis=100, unrealized_pl=-30, market_value=999)
    positions = {"a": position}
    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", lambda *_: ["a"])
    stop = planning.plan_stops(
        assessable=positions,
        strategy_positions=positions,
        corrected_positions=positions,
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    validation = planning.validate_exits(
        sells=["a"],
        strategy_positions=positions,
        invalid_symbols=frozenset(),
    )
    exits = planning.plan_exits(
        sells=["a"],
        strategy_positions=positions,
        invalid_symbols=frozenset(),
        pending_symbols=frozenset(),
    )
    assert not any(inspect.isgenerator(p) for p in (stop, validation, exits))
    assert stop.advance().book_amount is None
    assert validation.advance() is None
    assert exits.advance().book_amount is None
    assert position == {"qty": 3, "cost_basis": 100, "unrealized_pl": -30, "market_value": 999}


def test_assess_stops_propagates_stop_iteration(monkeypatch):
    error = StopIteration("scan")
    monkeypatch.setattr(catastrophe_stop, "unrealized_pct", Mock(side_effect=error))
    with pytest.raises(StopIteration) as caught:
        planning.assess_stops({"a": {}})
    assert caught.value is error


def test_later_stop_missing_qty_raises_after_first_decision(monkeypatch):
    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", lambda *_: ["a", "b"])
    corrected = {"a": {"cost_basis": 100, "unrealized_pl": -30}, "b": {}}
    positions = {"a": {"qty": 2}, "b": {}}
    planner = planning.plan_stops(
        assessable=corrected,
        strategy_positions=positions,
        corrected_positions=corrected,
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    assert planner.advance().symbol == "a"
    with pytest.raises(KeyError, match="qty"):
        planner.advance()
    assert positions == {"a": {"qty": 2}, "b": {}}


def test_exit_validation_is_lazy_and_skips_already_invalid():
    positions = {"invalid": {}, "bad": {"qty": 0}, "later": {}}
    invalid = frozenset({"invalid"})
    planner = planning.validate_exits(
        sells=["missing", "invalid", "bad", "later"],
        strategy_positions=positions,
        invalid_symbols=invalid,
    )
    assert planner.advance().symbol == "bad"
    with pytest.raises(KeyError, match="qty"):
        planner.advance()
    assert invalid == frozenset({"invalid"})
    assert positions == {"invalid": {}, "bad": {"qty": 0}, "later": {}}
    finished = planning.validate_exits(
        sells=["missing", "invalid"],
        strategy_positions=positions,
        invalid_symbols=invalid,
    )
    assert finished.advance() is None and finished.advance() is None


def test_invalid_stop_takes_precedence_over_pending(monkeypatch):
    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", lambda *_: ["a"])
    planner = planning.plan_stops(
        assessable={},
        strategy_positions={"a": {"qty": 0}},
        corrected_positions={},
        threshold=-0.25,
        pending_symbols=frozenset({"a"}),
    )
    assert isinstance(planner.advance(), InvalidExit)


@pytest.mark.parametrize("qty", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_quantities_are_rejected_by_stop_and_exit_validation(monkeypatch, qty):
    monkeypatch.setattr(catastrophe_stop, "find_catastrophe_stops", lambda *_: ["a"])
    positions = {"a": {"qty": qty}}
    stop = planning.plan_stops(
        assessable={},
        strategy_positions=positions,
        corrected_positions={},
        threshold=-0.25,
        pending_symbols=frozenset(),
    )
    validation = planning.validate_exits(
        sells=["a"],
        strategy_positions=positions,
        invalid_symbols=frozenset(),
    )
    assert isinstance(stop.advance(), InvalidExit)
    assert isinstance(validation.advance(), InvalidExit)


def test_pending_exit_is_skipped_before_quantity_is_read():
    planner = planning.plan_exits(
        sells=["a"],
        strategy_positions={"a": {}},
        invalid_symbols=frozenset(),
        pending_symbols=frozenset({"a"}),
    )
    assert planner.advance() == Skip(logging.INFO, "Skipping SELL a: order already pending")


def buy_args(**changes):
    args = dict(
        buys_by_sleeve={"a": ["X"]},
        notional_by_sleeve={"a": 100},
        slot_caps={"a": 5},
        strategy_positions={},
        corrected_positions={},
        stopped=frozenset(),
        pending_symbols=frozenset(),
        slots_available=5,
        max_positions=10,
        risk=Mock(check_concentration=Mock(return_value=False)),
        portfolio_value=10000,
        open_by_sleeve={"a": 0},
    )
    args.update(changes)
    return args


def sweep_args(**changes):
    cfg = Mock(
        sweep_symbol="SPY",
        reserve_pct=Mock(return_value=0.05),
        min_clip_usd=Mock(return_value=500),
        buy_trigger_pct=Mock(return_value=0.08),
    )
    args = dict(
        cfg=cfg,
        account={"cash": 0},
        positions={"SPY": {"market_value": 1000, "current_price": 30}},
        strategy_positions={},
        sells=[],
        pending_symbols=frozenset(),
        risk=Mock(max_new_positions=Mock(return_value=5)),
        buys_by_sleeve={"a": ["X"]},
        notional_by_sleeve={"a": 100},
        slot_caps={"a": 5},
        portfolio_value=10000,
    )
    args.update(changes)
    return args


def test_plan_sweep_sell_returns_none_when_cash_covers_the_buys():
    assert planning.plan_sweep_sell(**sweep_args(account={"cash": 1000})) is None


def test_plan_sweep_sell_sizes_qty_from_price_and_rounds_to_four_places():
    intent = planning.plan_sweep_sell(
        **sweep_args(positions={"SPY": {"market_value": 1000, "current_price": 31}})
    )
    assert (intent.kind, intent.size, intent.book_amount) == (IntentKind.SWEEP_SELL, 19.3548, 600)


@pytest.mark.parametrize("price", [0, -1, float("inf")])
def test_plan_sweep_sell_returns_none_for_a_zero_price_or_zero_quantity(price):
    args = sweep_args(positions={"SPY": {"market_value": 1000, "current_price": price}})
    assert planning.plan_sweep_sell(**args) is None


def test_plan_sweep_sell_counts_only_sells_not_pending_for_the_slot_estimate():
    args = sweep_args(
        strategy_positions={"A": {}, "B": {}},
        sells=["A", "B", "C"],
        pending_symbols=frozenset({"B"}),
    )
    planning.plan_sweep_sell(**args)
    args["risk"].max_new_positions.assert_called_once_with(1)


def test_plan_sweep_sell_still_restores_the_reserve_when_all_buys_were_cleared():
    assert planning.plan_sweep_sell(**sweep_args(buys_by_sleeve={})).book_amount == 500


def test_plan_sweep_sell_reraises_the_stored_reserve_error_after_reading_the_account_cash():
    args = sweep_args(account={})
    args["cfg"].reserve_pct.side_effect = ValueError("reserve")
    with pytest.raises(KeyError):
        planning.plan_sweep_sell(**args)
    args["account"] = {"cash": 0}
    with pytest.raises(ValueError, match="reserve"):
        planning.plan_sweep_sell(**args)


def test_plan_buys_follows_held_stopped_pending_slots_sleeve_cap_concentration_in_that_order():
    args = buy_args(
        buys_by_sleeve={"a": ["H", "S", "P", "C", "B"]},
        strategy_positions={"H": {}},
        stopped=frozenset({"H", "S"}),
        pending_symbols=frozenset({"S", "P"}),
    )
    args["risk"].check_concentration.side_effect = lambda symbol, *a, **kw: symbol == "C"
    decisions = drain(planning.plan_buys(**args))
    assert [d.message for d in decisions[:-1]] == [
        "Skipping BUY S: catastrophe-stopped this run",
        "Skipping BUY P: order already pending",
        "Concentration limit for C, skipping",
    ]
    assert decisions[-1].symbol == "B"


def test_plan_buys_breaks_at_the_slot_limit_per_sleeve_and_goes_on_to_the_next_sleeve():
    args = buy_args(
        buys_by_sleeve={"a": ["A", "B"], "b": ["C"]},
        notional_by_sleeve={"a": 1, "b": 1},
        slot_caps={"a": 0, "b": 2},
        open_by_sleeve={"a": 0, "b": 0},
    )
    decisions = drain(planning.plan_buys(**args))
    assert decisions[0].message == "Sleeve a slot cap reached, skipping A"
    assert decisions[1].symbol == "C"


def test_plan_buys_counts_each_yielded_intent_toward_slots_and_sleeve_caps():
    planner = planning.plan_buys(**buy_args(buys_by_sleeve={"a": ["X", "Y"]}, slots_available=1))
    assert planner.advance().symbol == "X"
    assert planner.advance().message == "Max positions reached (10), skipping Y"
    assert planner.buys_attempted == 1 and planner.open_by_sleeve == {"a": 1}


def test_plan_buys_does_not_write_the_open_counts_it_was_given():
    counts = {"a": 0}
    planning.plan_buys(**buy_args(open_by_sleeve=counts)).advance()
    assert counts == {"a": 0}


def test_plan_buys_never_rebuys_a_stopped_symbol():
    assert drain(planning.plan_buys(**buy_args(stopped=frozenset({"X"}))))[0].message.startswith(
        "Skipping BUY X"
    )


def test_plan_buys_slot_arithmetic_is_the_callers_slots_available():
    assert (
        planning.plan_buys(**buy_args(slots_available=0)).advance().message
        == "Max positions reached (10), skipping X"
    )


def test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed():
    args = buy_args(buys_by_sleeve={"a": ["X", "Y"]})
    args["risk"].check_concentration.side_effect = [False, ValueError("later")]
    planner = planning.plan_buys(**args)
    assert planner.advance().symbol == "X"
    with pytest.raises(ValueError, match="later"):
        planner.advance()


def test_a_stop_iteration_from_check_concentration_surfaces_as_stop_iteration():
    args = buy_args()
    error = StopIteration("same")
    args["risk"].check_concentration.side_effect = error
    with pytest.raises(StopIteration) as exc:
        planning.plan_buys(**args).advance()
    assert exc.value is error


def test_a_symbol_bought_by_two_sleeves_is_planned_twice_as_today():
    args = buy_args(
        buys_by_sleeve={"a": ["X"], "b": ["X"]},
        notional_by_sleeve={"a": 1, "b": 2},
        slot_caps={"a": 1, "b": 1},
        open_by_sleeve={"a": 0, "b": 0},
    )
    assert [d.symbol for d in drain(planning.plan_buys(**args))] == ["X", "X"]


def test_a_buy_signal_for_the_sweep_symbol_is_an_ordinary_candidate():
    assert planning.plan_buys(**buy_args(buys_by_sleeve={"a": ["SPY"]})).advance().symbol == "SPY"


def test_plan_buys_raises_key_error_for_a_sleeve_missing_from_the_open_counts_at_the_old_point():
    args = buy_args(
        buys_by_sleeve={"a": ["H", "X"]}, strategy_positions={"H": {}}, open_by_sleeve={}
    )
    with pytest.raises(KeyError, match="a"):
        planning.plan_buys(**args).advance()
    assert not args["risk"].check_concentration.called


def test_plan_sweep_buy_matches_compute_sweep_buy_for_valid_settings():
    cfg = sweep_args()["cfg"]
    intent = planning.plan_sweep_buy(cfg=cfg, mid_account={"cash": 2000, "portfolio_value": 10000})
    assert (intent.kind, intent.symbol, intent.size, intent.book_amount) == (
        IntentKind.SWEEP_BUY,
        "SPY",
        1500,
        1500,
    )


@pytest.mark.parametrize("method", ["reserve_pct", "min_clip_usd", "buy_trigger_pct"])
def test_plan_sweep_buy_reraises_each_stored_parse_error_in_its_old_order(method):
    cfg = sweep_args()["cfg"]
    getattr(cfg, method).side_effect = ValueError(method)
    with pytest.raises(ValueError, match=method):
        planning.plan_sweep_buy(cfg=cfg, mid_account={"cash": 2000, "portfolio_value": 10000})
    methods = ["reserve_pct", "min_clip_usd", "buy_trigger_pct"]
    for later in methods[methods.index(method) + 1 :]:
        assert not getattr(cfg, later).called


def test_plan_sweep_buy_returns_none_when_the_action_is_not_a_buy():
    assert (
        planning.plan_sweep_buy(
            cfg=sweep_args()["cfg"], mid_account={"cash": 0, "portfolio_value": 10000}
        )
        is None
    )
