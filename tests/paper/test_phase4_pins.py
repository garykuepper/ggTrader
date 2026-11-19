"""Phase 4 shape pins (net K): result key order, harness name discovery, poll fallback fields.

The transcript goldens sort keys (``to_json``), so they cannot see the order of the result dict;
and they never read the fallback record `_poll_orders` builds from the order tuple. The tests here
pin both, plus the way the harness finds the persist functions it wraps.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper import persist
from ggTrader.paper import trader as trader_module
from ggTrader.paper.models import SubmittedOrder
from ggTrader.paper.trader import PaperTrader
from tests.paper.transcript import harness
from tests.paper.transcript.harness import persist_function_names, run_scenario
from tests.paper.transcript.scenarios import FIXTURE_SPEC, SCENARIOS

#: The 26 names `persist_function_names()` returns on unchanged code (sorted): the 25 the harness
#: wraps one by one, plus `paper_run_lock`, which it replaces with a context manager.
PERSIST_NAMES_26 = [
    "attach_broker_order",
    "clear_pending_order",
    "clear_split_correction",
    "create_order_intent",
    "get_accrued_dividend_keys",
    "get_earliest_snapshot",
    "get_last_buy_dates",
    "get_latest_snapshot",
    "get_open_split_states",
    "get_peak_value",
    "get_pending_orders",
    "get_snapshot_history",
    "get_total_dividend_accrual",
    "get_trade_history_dates",
    "get_unresolved_order_intents",
    "init_paper_schema",
    "log_dividend_accrual",
    "log_pending_order",
    "log_snapshot",
    "log_trade",
    "mark_pending_order_stale",
    "paper_run_lock",
    "save_peak_value",
    "save_split_correction",
    "settle_legacy_pending_order",
    "settle_order_intent",
]

#: The nine runs that return before the final result (`trader.py`, ten return sites in all).
EARLY_RETURNS = [
    "market_closed",
    "late_run_live",
    "stale_as_of",
    "peak_unreadable",
    "peak_invalid",
    "invalid_equity_halt",
    "peak_save_failure",
    "last_equity_missing_halt",
    "last_equity_invalid_halt",
]

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- result key order


def _result(name: str) -> dict:
    transcript = run_scenario(SCENARIOS[name])
    assert transcript["raised"] is None, f"{name} raised {transcript['raised']}"
    assert transcript["result"] is not None
    return transcript["result"]


def test_result_key_order_per_shape() -> None:
    assert list(_result("market_closed")) == ["buys", "sells", "errors"]
    assert list(_result("late_run_live")) == ["buys", "sells", "errors", "gate"]
    full = ["buys", "sells", "errors", "untradable", "skipped_writes"]
    # A drawdown halt blocks buys but runs on to the final result: the full shape, not the short
    # one (the task brief listed the short shape for it; the code returns the full one).
    assert list(_result("drawdown_halt")) == full
    assert list(_result("plain_buys_and_sells")) == full


@pytest.mark.parametrize("name", EARLY_RETURNS)
def test_early_returns_carry_neither_untradable_nor_skipped_writes(name: str) -> None:
    keys = list(_result(name))
    assert keys[:3] == ["buys", "sells", "errors"]
    assert "untradable" not in keys and "skipped_writes" not in keys


# --------------------------------------------------------------------------- harness discovery


def test_harness_finds_the_26_persist_names() -> None:
    assert persist_function_names() == PERSIST_NAMES_26
    assert trader_module.paper_run_lock.__module__ == "ggTrader.paper.persist"
    assert persist.paper_run_lock is trader_module.paper_run_lock


def test_harness_wraps_25_names_and_replaces_the_lock_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_scenario` patches the trader namespace: 25 persist wrappers and one lock manager."""
    recorded: list[tuple[object, str, object]] = []
    real_patch = harness.patch

    def recording_object(
        target: object, attribute: str, new: object, *a: object, **k: object
    ) -> Any:
        recorded.append((target, attribute, new))
        return real_patch.object(target, attribute, new, *a, **k)

    spy = SimpleNamespace(object=recording_object, dict=real_patch.dict)
    monkeypatch.setattr(harness, "patch", spy)
    run_scenario(SCENARIOS["market_closed"])

    in_trader = {attr: new for target, attr, new in recorded if target is trader_module}
    wrapped = sorted(name for name in PERSIST_NAMES_26 if name in in_trader)
    assert wrapped == PERSIST_NAMES_26  # every persist name is replaced, one way or another
    ordinary = [n for n in PERSIST_NAMES_26 if n != "paper_run_lock"]
    assert len(ordinary) == 25
    for name in ordinary:
        assert in_trader[name].__name__ == name  # the wrapper `_write` records by name
        assert in_trader[name].__module__ != "ggTrader.paper.persist"
    lock = in_trader["paper_run_lock"]
    assert lock.__name__ == "paper_run_lock"
    assert lock.__qualname__.startswith("_Run.")  # the context manager of the run
    assert not any(in_trader[n] is lock for n in ordinary)


@pytest.mark.parametrize(
    "name", ["sweep_min_clip_invalid_fail_soft", "sweep_buy_trigger_invalid_fail_soft"]
)
def test_invalid_sweep_buy_setting_logs_warning(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=trader_module._log.name):
        run_scenario(SCENARIOS[name])
    assert any("Cash sweep buy sizing failed (non-fatal)" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("name", "setting"),
    [
        ("sweep_min_clip_invalid_fail_soft", "SWEEP_MIN_CLIP_USD"),
        ("sweep_buy_trigger_invalid_fail_soft", "SWEEP_BUY_TRIGGER_PCT"),
    ],
)
def test_invalid_sweep_setting_changes_outcome(name: str, setting: str) -> None:
    invalid = SCENARIOS[name]
    assert invalid.env[setting] == "abc"
    valid = copy.deepcopy(invalid)
    del valid.env[setting]
    bad_result = run_scenario(invalid)
    good_result = run_scenario(valid)
    bad_buys = [c for c in bad_result["calls"] if c["target"] == "broker.submit_buy"]
    good_buys = [c for c in good_result["calls"] if c["target"] == "broker.submit_buy"]
    assert not bad_buys
    assert good_buys and good_buys[0]["args"][0] == "SPY"


# --------------------------------------------------------------------------- the poll fallback


class _Clock:
    """A clock whose sleeps advance it, so the poll's wait budget ends without real waiting."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def _trader(broker: MagicMock) -> Iterator[PaperTrader]:
    with patch.object(trader_module, "time", _Clock()):
        yield PaperTrader(broker, MagicMock(), dry_run=True, membership=FIXTURE_SPEC)


def _broker(get_order: Callable[[str], dict]) -> MagicMock:
    broker = MagicMock()
    broker.get_clock.return_value = {"is_open": True}
    broker.get_order.side_effect = get_order
    return broker


def test_poll_fallback_record_is_built_from_the_order_tuple() -> None:
    """Every field of the fallback, for a SELL (no notional) and a BUY (notional = the amount)."""

    def get_order(oid: str) -> dict:
        raise RuntimeError("down")

    orders = [
        SubmittedOrder("ord-1", "SELL", "SPY", 7300.0, "cash_sweep"),
        SubmittedOrder("ord-2", "BUY", "MSFT", 3300.0, None),
    ]
    with _trader(_broker(get_order)) as trader:
        filled = trader._poll_orders(orders)

    base = {"qty": None, "filled_qty": 0.0, "filled_avg_price": 0.0, "status": "pending"}
    assert filled == {
        "ord-1": {"id": "ord-1", "symbol": "SPY", "side": "SELL", "notional": None, **base},
        "ord-2": {"id": "ord-2", "symbol": "MSFT", "side": "BUY", "notional": 3300.0, **base},
    }
    assert list(filled["ord-2"]) == [
        "id",
        "symbol",
        "side",
        "qty",
        "notional",
        "filled_qty",
        "filled_avg_price",
        "status",
    ]


def test_poll_looks_orders_up_by_the_first_tuple_field() -> None:
    asked: list[str] = []

    def get_order(oid: str) -> dict:
        asked.append(oid)
        return {"id": oid, "status": "filled", "filled_qty": 1.0, "filled_avg_price": 2.0}

    orders = [
        SubmittedOrder("ord-1", "SELL", "AAA", 1.0, None),
        SubmittedOrder("ord-2", "BUY", "BBB", 2.0, None),
    ]
    with _trader(_broker(get_order)) as trader:
        filled = trader._poll_orders(orders)

    assert asked == ["ord-1", "ord-2"] and sorted(filled) == ["ord-1", "ord-2"]


def test_poll_of_no_orders_returns_an_empty_dict_without_a_broker_call() -> None:
    broker = _broker(RuntimeError("must not be called"))
    with _trader(broker) as trader:
        assert trader._poll_orders([]) == {}
    broker.get_clock.assert_not_called()


# --------------------------------------------------------------------------- the order records


def _submitted_orders(name: str) -> list[SubmittedOrder]:
    """Every distinct order record `run` hands to `_poll_orders`, in first-seen order."""
    seen: dict[str, SubmittedOrder] = {}
    real = PaperTrader._poll_orders

    def spy(self: PaperTrader, orders: list[SubmittedOrder]) -> dict[str, dict]:
        for order in orders:
            seen.setdefault(order.order_id, order)
        return real(self, orders)

    with patch.object(PaperTrader, "_poll_orders", spy):
        run_scenario(SCENARIOS[name])
    return list(seen.values())


def test_pending_order_records_carry_every_field_including_the_reason() -> None:
    """No transcript shows the ledger tag of a pending order (the alert and the slot check read
    only the sweep tag), so this pins the record each submit site builds."""
    orders = _submitted_orders("pending_order_all_five_fields")
    assert [(o.side, o.symbol, o.reason) for o in orders] == [
        ("SELL", "NXPI", "catastrophe_stop"),
        ("SELL", "SPY", "cash_sweep"),
        ("SELL", "AAPL", None),
        ("BUY", "MSFT", None),
    ]
    assert [o.amount for o in orders] == [6000.0, 7300.0, 1500.0, 3300.0]
    assert all(o.order_id for o in orders) and len({o.order_id for o in orders}) == 4


def test_sweep_buy_record_carries_the_sweep_reason() -> None:
    orders = _submitted_orders("sweep_buy_after_strategy_orders")
    sweep = [o for o in orders if o.reason is not None]
    assert [(o.side, o.symbol, o.amount, o.reason) for o in sweep] == [
        ("BUY", "SPY", 41700.0, "cash_sweep")
    ]
    assert all(o.reason is None for o in orders if o not in sweep)
