"""Tests of the typed records of the paper trader (`ggTrader.paper.models`)."""

from __future__ import annotations

import dataclasses
from typing import get_type_hints
from unittest.mock import patch

import pytest

from ggTrader.paper.models import (
    Account,
    MarketClock,
    OrderStatus,
    Position,
    RunResult,
    SignalBundle,
    SleeveSignals,
    SubmittedOrder,
)


def _order() -> SubmittedOrder:
    return SubmittedOrder("ord-1", "SELL", "SPY", 7300.0, "cash_sweep")


def test_submitted_order_is_frozen() -> None:
    order = _order()
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.symbol = "QQQ"  # type: ignore[misc]


def test_submitted_order_fields_in_order() -> None:
    names = [f.name for f in dataclasses.fields(SubmittedOrder)]
    assert names == ["order_id", "side", "symbol", "amount", "reason"]
    order = _order()
    assert (order.order_id, order.side, order.symbol, order.amount, order.reason) == (
        "ord-1",
        "SELL",
        "SPY",
        7300.0,
        "cash_sweep",
    )


def test_submitted_order_equality_and_hash() -> None:
    assert _order() == _order()
    assert hash(_order()) == hash(_order())
    assert _order() != SubmittedOrder("ord-1", "SELL", "SPY", 7300.0, None)
    assert len({_order(), _order()}) == 1


def test_submitted_order_has_no_tuple_behavior() -> None:
    """Deviation 7: the five-item tuple is gone, so no length, indexing or unpacking remains."""
    order = _order()
    with pytest.raises(TypeError):
        len(order)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        order[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        _a, _b, _c, _d, _e = order  # type: ignore[misc]


# ---------------------------------------------------------------------------- RunResult

FULL_KEYS = ["buys", "sells", "errors", "untradable", "skipped_writes"]


def _shapes() -> dict[str, RunResult]:
    """The four result shapes `PaperTrader._run` returns, built by hand."""
    return {
        "early": RunResult([], [], []),
        "late_run": RunResult([], [], [], gate="late_run"),
        "early_error": RunResult([], [], ["reason"]),
        "final": RunResult(["AAPL"], ["MSFT"], ["e"], untradable=["AVB"], skipped_writes=["x"]),
    }


def test_run_result_as_dict_key_order() -> None:
    shapes = _shapes()
    assert list(shapes["early"].as_dict()) == ["buys", "sells", "errors"]
    assert list(shapes["early_error"].as_dict()) == ["buys", "sells", "errors"]
    assert list(shapes["late_run"].as_dict()) == ["buys", "sells", "errors", "gate"]
    assert list(shapes["final"].as_dict()) == FULL_KEYS
    both = RunResult([], [], [], gate="g", untradable=[], skipped_writes=[]).as_dict()
    assert list(both) == ["buys", "sells", "errors", "gate", "untradable", "skipped_writes"]
    assert shapes["late_run"].as_dict() == {
        "buys": [],
        "sells": [],
        "errors": [],
        "gate": "late_run",
    }


def test_run_result_omits_none_fields() -> None:
    result = RunResult(["A"], [], []).as_dict()
    assert result == {"buys": ["A"], "sells": [], "errors": []}
    assert "gate" not in result and "untradable" not in result and "skipped_writes" not in result
    assert "skipped_writes" not in RunResult([], [], [], untradable=[]).as_dict()


def test_empty_untradable_is_present_not_omitted() -> None:
    result = RunResult([], [], [], untradable=[], skipped_writes=[]).as_dict()
    assert result["untradable"] == []
    assert result["skipped_writes"] == []
    assert list(result) == FULL_KEYS


def test_as_dict_returns_the_same_objects() -> None:
    result = RunResult(["A"], ["B"], ["C"], untradable=["D"], skipped_writes=["E"])
    out = result.as_dict()
    assert out["buys"] is result.buys
    assert out["sells"] is result.sells
    assert out["errors"] is result.errors
    assert out["untradable"] is result.untradable
    assert out["skipped_writes"] is result.skipped_writes


def test_run_result_is_frozen() -> None:
    result = RunResult([], [], [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.buys = ["X"]  # type: ignore[misc]


@pytest.mark.parametrize("shape", ["early", "late_run", "early_error", "final"])
def test_cli_consumes_every_shape(shape: str) -> None:
    from typer.testing import CliRunner

    from ggTrader.cli.main import app

    with patch("ggTrader.paper.trader.run_paper_trading") as run:
        run.return_value = _shapes()[shape].as_dict()
        result = CliRunner().invoke(app, ["paper"])
    assert result.exit_code == 0, result.output
    assert "Planned buys:" in result.output


@pytest.mark.parametrize(
    ("record", "keys"),
    [
        (Account, {"cash", "portfolio_value", "last_equity", "buying_power", "multiplier"}),
        (
            Position,
            {
                "qty",
                "market_value",
                "current_price",
                "avg_entry",
                "cost_basis",
                "unrealized_pl",
                "unrealized_plpc",
                "change_today",
            },
        ),
        (
            OrderStatus,
            {
                "id",
                "client_order_id",
                "symbol",
                "side",
                "status",
                "qty",
                "notional",
                "filled_qty",
                "filled_avg_price",
            },
        ),
        (MarketClock, {"is_open", "next_open", "next_close"}),
        (
            SleeveSignals,
            {
                "buys",
                "sells",
                "as_of",
                "universe_size",
                "membership_gate",
                "last_exit",
                "engine",
                "bars",
            },
        ),
        (
            SignalBundle,
            {"sleeves", "weights", "scale", "rebalanced_today", "fallback_used"},
        ),
    ],
)
def test_typed_dict_keys_are_the_documented_ones(record: type, keys: set[str]) -> None:
    assert set(record.__annotations__) == keys
    assert set(get_type_hints(record)) == keys
    assert record.__total__ is False


def test_new_run_records():
    from dataclasses import FrozenInstanceError

    from ggTrader.paper.models import (
        ExecutionBook,
        FlowState,
        IntentKind,
        InvalidExit,
        OrderIntent,
        RiskVerdict,
        RunContext,
        Skip,
        StrategySignals,
    )

    intent = OrderIntent(IntentKind.BUY, "BUY", "ABC", 5.0, None, None)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        intent.symbol = "DEF"
    assert not hasattr(intent, "__dict__")
    for value, name in (
        (Skip(20, "skip"), "message"),
        (InvalidExit("ABC", "error"), "error"),
        (RunContext(None, None, frozenset()), "today_et"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, "changed")
    first = ExecutionBook()
    second = ExecutionBook()
    first.buys.append("ABC")
    assert second.buys == []
    a, b = FlowState(), FlowState()
    a.all_buys.append(("ABC", "one"))
    assert b.all_buys == []
    verdict = RiskVerdict(None, ("first",))
    assert verdict.with_halt("second").buy_halts == ("first", "second")
    assert verdict.buy_halts == ("first",)
    signals = StrategySignals((("ABC", "one"), ("ABC", "two")), (), {}, "", {}, 1.0)
    assert signals.buys == (("ABC", "one"), ("ABC", "two"))
    with pytest.raises(TypeError):
        signals.weights["one"] = 1.0
