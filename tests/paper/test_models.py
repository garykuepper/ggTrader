"""Tests of the typed records of the paper trader (`ggTrader.paper.models`)."""

from __future__ import annotations

import dataclasses

import pytest

from ggTrader.paper.models import SubmittedOrder


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
