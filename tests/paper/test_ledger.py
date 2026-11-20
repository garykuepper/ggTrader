"""Tests of the `PaperLedger` protocol and its in-memory and read-only forms."""

from __future__ import annotations

import inspect
import json
import logging
from datetime import date
from pathlib import Path

import pytest

from ggTrader.paper import ledger, persist
from ggTrader.paper.ledger import (
    WRITE_METHODS,
    InMemoryLedger,
    PaperLedger,
    ReadOnlyLedger,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "persist_sql.json"


def _public_persist_functions() -> dict[str, object]:
    return {
        name: fn
        for name, fn in inspect.getmembers(persist, inspect.isfunction)
        if not name.startswith("_") and fn.__module__ == persist.__name__
    }


def _protocol_methods() -> dict[str, object]:
    return {
        name: fn
        for name, fn in inspect.getmembers(PaperLedger, inspect.isfunction)
        if not name.startswith("_")
    }


def _params(fn: object) -> list[tuple[str, object, object]]:
    """Name, kind and default of each parameter, `self` dropped."""
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


# --------------------------------------------------------------------------- the protocol


def test_protocol_names_equal_persist_function_names() -> None:
    assert sorted(_protocol_methods()) == sorted(_public_persist_functions())
    assert len(_protocol_methods()) == 31


def test_protocol_parameters_and_returns_follow_the_persist_functions() -> None:
    """Parameters match by name, kind and default; the two deliberate differences are named."""
    persist_fns = _public_persist_functions()
    for name, method in _protocol_methods().items():
        assert _params(method) == _params(persist_fns[name]), name
        for key, parameter in inspect.signature(method).parameters.items():
            if key == "self" or (name, key) == ("log_snapshot", "positions"):
                continue
            assert (
                parameter.annotation
                == inspect.signature(persist_fns[name]).parameters[key].annotation
            )
        if name != "paper_run_lock":
            expected = inspect.signature(persist_fns[name]).return_annotation
            assert inspect.signature(method).return_annotation == expected, name
    assert (
        inspect.signature(_protocol_methods()["paper_run_lock"]).return_annotation
        == "AbstractContextManager[RunLock]"
    )
    log_snapshot = inspect.signature(_protocol_methods()["log_snapshot"])
    assert log_snapshot.parameters["positions"].annotation == "dict[str, Position]"


# --------------------------------------------------------------------------- ReadOnlyLedger


class _Recorder:
    """An inner ledger that records every call and returns a marker."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def call(*args: object, **kwargs: object) -> str:
            self.calls.append(name)
            return f"inner:{name}"

        return call


def test_write_methods_are_the_fifteen_named_writes() -> None:
    assert isinstance(WRITE_METHODS, frozenset)
    assert len(WRITE_METHODS) == 15
    assert WRITE_METHODS <= set(_public_persist_functions())
    assert not WRITE_METHODS & {"paper_run_lock"}
    assert not {name for name in _public_persist_functions() if name.startswith("get_")} & (
        WRITE_METHODS
    )


def test_read_only_ledger_skips_every_write_and_records_names_in_order() -> None:
    inner = _Recorder()
    skipped: list[str] = []
    ledger_ = ReadOnlyLedger(inner, skipped, logging.getLogger("test"))
    names = sorted(WRITE_METHODS)
    for name in names:
        assert getattr(ledger_, name)("a", b=1) is None
    assert skipped == names
    assert inner.calls == []
    assert ledger_._skipped is skipped  # the list passed in, not a copy


def test_read_only_ledger_passes_reads_through_and_never_swallows() -> None:
    class Inner:
        def get_peak_value(self) -> float:
            return 7.0

        def get_pending_orders(self) -> list[dict]:
            raise RuntimeError("database down")

    skipped: list[str] = []
    ledger_ = ReadOnlyLedger(Inner(), skipped, logging.getLogger("test"))
    assert ledger_.get_peak_value() == 7.0
    with pytest.raises(RuntimeError, match="database down"):
        ledger_.get_pending_orders()
    assert skipped == []


def test_read_only_ledger_passes_the_run_lock_through() -> None:
    inner = _Recorder()
    ledger_ = ReadOnlyLedger(inner, [], logging.getLogger("test"))
    assert ledger_.paper_run_lock() == "inner:paper_run_lock"
    assert inner.calls == ["paper_run_lock"]


def test_read_only_ledger_logs_through_the_logger_it_was_given(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("ggTrader.paper.trader")
    ledger_ = ReadOnlyLedger(_Recorder(), [], log)
    with caplog.at_level(logging.INFO, logger="ggTrader.paper.trader"):
        ledger_.log_snapshot("2026-06-22", 1.0, 1.0, {})
    records = [r for r in caplog.records if r.name == "ggTrader.paper.trader"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.INFO, "read-only: skipped write log_snapshot")
    ]


def test_read_only_ledger_uses_the_given_logger_not_its_own(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        ReadOnlyLedger(_Recorder(), [], logging.getLogger("elsewhere")).save_peak_value(1.0)
    assert [r.name for r in caplog.records] == ["elsewhere"]


# --------------------------------------------------------------------------- InMemoryLedger


def _intent(store: InMemoryLedger, client_id: str = "c1", side: str = "BUY") -> None:
    kind = "notional" if side == "BUY" else "qty"
    store.create_order_intent(client_id, "2026-06-22", side, "AAPL", 100.0, kind, None)


def test_in_memory_ledger_intent_round_trip_and_error_messages() -> None:
    store = InMemoryLedger()
    _intent(store)
    assert [r["state"] for r in store.get_unresolved_order_intents()] == ["intent"]

    with pytest.raises(ValueError, match="Missing intent or conflicting broker order"):
        store.attach_broker_order("nope", "o1")
    store.attach_broker_order("c1", "o1")
    store.attach_broker_order("c1", "o1")  # same id again is a no-op
    with pytest.raises(ValueError, match="Missing intent or conflicting broker order"):
        store.attach_broker_order("c1", "o2")
    assert store.get_unresolved_order_intents()[0]["state"] == "submitted"

    _intent(store, "c2")
    store.attach_broker_order("c2", "o2")
    store.settle_order_intent("c2", "o2", 0.0, None, "canceled")
    store.settle_order_intent("c2", "o2", 0.0, None, "canceled")  # identical replay: no-op
    with pytest.raises(ValueError, match="Conflicting settlement"):
        store.settle_order_intent("c2", "o2", 0.0, None, "expired")

    for bad in (
        (None, None, "filled"),
        (float("nan"), None, "filled"),
        (-1.0, None, "canceled"),
        (1.0, 10.0, "open"),
        (0.0, None, "filled"),
        (1.0, 0.0, "filled"),
        (1.0, float("inf"), "filled"),
    ):
        with pytest.raises(ValueError, match="Invalid terminal fill"):
            store.settle_order_intent("c1", "o1", *bad)

    store.settle_order_intent("c1", "o1", 2.0, 50.0, "filled")
    assert store.trades == [
        {
            "run_date": "2026-06-22",
            "side": "BUY",
            "symbol": "AAPL",
            "amount": 100.0,
            "order_id": "o1",
            "reason": None,
        }
    ]
    assert store.get_unresolved_order_intents() == []


def test_in_memory_ledger_settle_checks_trades_and_pending_representation() -> None:
    store = InMemoryLedger()
    _intent(store)
    store.attach_broker_order("c1", "o1")
    store.log_trade("2026-06-22", "BUY", "AAPL", 999.0, "o1", None)
    with pytest.raises(ValueError, match="Conflicting historical trade"):
        store.settle_order_intent("c1", "o1", 2.0, 50.0, "filled")

    store = InMemoryLedger()
    _intent(store)
    store.log_pending_order("2026-06-22", "BUY", "AAPL", 100.0, "o1")
    with pytest.raises(ValueError, match="Order has both pending and intent representations"):
        store.attach_broker_order("c1", "o1")
    with pytest.raises(ValueError, match="Order has both pending and intent representations"):
        store.settle_order_intent("c1", "o1", 2.0, 50.0, "filled")


def test_in_memory_ledger_legacy_pending_orders() -> None:
    store = InMemoryLedger()
    with pytest.raises(ValueError, match="Missing legacy pending order"):
        store.settle_legacy_pending_order("o1", 1.0, 10.0, "filled")
    store.log_pending_order("2026-06-22", "BUY", "AAPL", 10.0, "o1", "cash_sweep")
    store.log_pending_order("2026-06-22", "BUY", "AAPL", 10.0, "o1", "cash_sweep")  # idempotent
    store.mark_pending_order_stale("o1")
    assert [(r["order_id"], r["flagged_stale"]) for r in store.get_pending_orders()] == [
        ("o1", True)
    ]
    with pytest.raises(ValueError, match="Invalid terminal fill"):
        store.settle_legacy_pending_order("o1", 0.0, None, "filled")
    store.settle_legacy_pending_order("o1", 1.0, 10.0, "filled")
    assert store.get_pending_orders() == []
    assert [t["reason"] for t in store.trades] == ["cash_sweep"]
    store.log_pending_order("2026-06-22", "BUY", "AAPL", 10.0, "o2")
    store.clear_pending_order("o2")
    assert store.get_pending_orders() == []


def test_in_memory_ledger_reads_return_what_was_seeded_or_written() -> None:
    store = InMemoryLedger()
    store.init_paper_schema()
    assert store.get_latest_snapshot() is None
    assert store.get_earliest_snapshot() is None
    assert store.get_latest_snapshot_positions() is None
    assert store.get_latest_snapshot_run_date() is None
    assert store.get_peak_value() is None
    assert store.get_rebalance_state() is None
    assert store.get_total_dividend_accrual() == 0.0

    store.log_snapshot("2026-06-22", 100.0, 10.0, {"AAPL": {"qty": 1.0}})
    store.log_snapshot("2026-06-23", 110.0, 5.0, {})
    store.log_snapshot("2026-06-23", 120.0, 5.0, {"MSFT": {"qty": 2.0}})  # upsert by date
    assert store.get_latest_snapshot() == 120.0
    assert store.get_latest_snapshot(before=date(2026, 6, 23)) == 100.0
    assert store.get_earliest_snapshot() == 100.0
    assert store.get_latest_snapshot_run_date() == "2026-06-23"
    assert store.get_latest_snapshot_positions() == {"MSFT": {"qty": 2.0}}
    assert store.get_snapshot_history() == [
        (date(2026, 6, 22), {"AAPL": {"qty": 1.0}}),
        (date(2026, 6, 23), {"MSFT": {"qty": 2.0}}),
    ]

    store.save_peak_value(130.0)
    assert store.get_peak_value() == 130.0
    store.save_rebalance_state("2026-06-22", {"AAPL": 0.5}, 0.9)
    assert store.get_rebalance_state() == {
        "rebalance_date": "2026-06-22",
        "weights": {"AAPL": 0.5},
        "scale": 0.9,
    }

    store.log_trade("2026-06-22", "BUY", "AAPL", 10.0, "o1")
    store.log_trade("2026-06-22", "BUY", "AAPL", 10.0, "o1")  # same trade again: no-op
    store.log_trade("2026-06-24", "BUY", "AAPL", 10.0, "o2")
    store.log_trade("2026-06-25", "SELL", "AAPL", 10.0, "o3")
    with pytest.raises(ValueError, match="Conflicting historical trade"):
        store.log_trade("2026-06-22", "BUY", "AAPL", 11.0, "o1")
    assert store.get_trade_history_dates("AAPL") == [
        date(2026, 6, 22),
        date(2026, 6, 24),
        date(2026, 6, 25),
    ]
    assert store.get_last_buy_dates() == {"AAPL": date(2026, 6, 24)}

    store.log_dividend_accrual("AAPL", date(2026, 6, 1), 0.2, 10.0, 2.0)
    store.log_dividend_accrual("AAPL", date(2026, 6, 1), 0.3, 10.0, 3.0)  # conflict: no-op
    assert store.get_accrued_dividend_keys() == {("AAPL", date(2026, 6, 1))}
    assert store.get_total_dividend_accrual() == 2.0

    store.save_split_correction("MNST", "2026-08-11", 3.0)
    store.save_split_correction("MNST", "2026-08-11", 2.0)  # refreshes the factor
    assert store.get_open_split_corrections() == {"MNST": 2.0}
    assert store.get_open_split_states() == {"MNST": {"factor": 2.0, "ex_date": date(2026, 8, 11)}}
    store.clear_split_correction("MNST")
    assert store.get_open_split_corrections() == {}


def test_in_memory_ledger_returns_copies_not_internal_state() -> None:
    store = InMemoryLedger()
    store.log_snapshot("2026-06-22", 1.0, 1.0, {"AAPL": {"qty": 1.0}})
    store.get_latest_snapshot_positions()["AAPL"]["qty"] = 99.0
    assert store.get_latest_snapshot_positions() == {"AAPL": {"qty": 1.0}}


def test_in_memory_ledger_lock_records_verify() -> None:
    store = InMemoryLedger()
    with store.paper_run_lock() as lock:
        lock.verify()
        lock.verify()
    assert store.lock_verifications == 2
    assert store.lock_entries == 1
    with store.paper_run_lock() as lock:
        pass
    assert store.lock_entries == 2


def test_in_memory_ledger_has_every_protocol_method_with_the_same_parameters() -> None:
    for name, method in _protocol_methods().items():
        assert _params(getattr(InMemoryLedger, name)) == _params(method), name


def test_in_memory_ledger_states_that_it_does_not_claim_sql_parity() -> None:
    assert "not claim" in (InMemoryLedger.__doc__ or "").lower()


# --------------------------------------------------------------------------- the SQL golden


def _golden_base_names_with_a_write_event() -> set[str]:
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
    writers: set[str] = set()
    for case_name, record in cases.items():
        for event in record["events"]:
            is_commit = event["op"] == "commit"
            is_insert = event["op"] == "execute" and event["sql"].lstrip().upper().startswith(
                "INSERT"
            )
            if is_commit or is_insert:
                writers.add(case_name.split("__")[0])
    return writers


def test_write_methods_set_matches_the_sql_golden() -> None:
    writers = _golden_base_names_with_a_write_event()
    assert WRITE_METHODS == writers
    others = set(_public_persist_functions()) - WRITE_METHODS
    assert others.isdisjoint(writers)


def test_module_exposes_its_public_names() -> None:
    for name in (
        "RunLock",
        "PaperLedger",
        "WRITE_METHODS",
        "ReadOnlyLedger",
        "InMemoryLedger",
    ):
        assert hasattr(ledger, name), name
