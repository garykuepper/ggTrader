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
            "a": {"buys": ["Z", "A"], "sells": ["Z", "B"], "as_of": "first"},
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

    position = Position(qty=3, cost_basis=100, unrealized_pl=-30)
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
    assert position == {"qty": 3, "cost_basis": 100, "unrealized_pl": -30}


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
