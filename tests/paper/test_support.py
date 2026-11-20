"""Contract checks for the in-memory trader test rig."""

from datetime import datetime, timezone

import pytest

from ggTrader.paper import persist
from ggTrader.paper.ledger import InMemoryLedger
from tests.paper.support import (
    SpyLedger,
    UnstubbedPersistCall,
    make_trader,
    persist_guard,
    seed_pending_orders,
)


def test_spy_snapshots_arguments_and_rejects_unknown_names():
    ledger = SpyLedger()
    positions = {"A": {"qty": 2}}
    ledger.log_snapshot("2026-01-02", 100, 50, positions)
    positions["A"]["qty"] = 3
    assert ledger.calls("log_snapshot")[0][1][3]["A"]["qty"] == 2
    with pytest.raises(ValueError, match="unknown"):
        ledger.calls("log_snapshott")
    assert InMemoryLedger.get_latest_snapshot_positions(ledger)["A"]["qty"] == 2
    assert len(ledger.calls()) == 1


def test_seed_pending_preserves_stale_fields_without_recording_setup():
    ledger = SpyLedger()
    created = datetime(2025, 12, 1, tzinfo=timezone.utc)
    row = {
        "run_date": "2025-12-01",
        "side": "SELL",
        "symbol": "A",
        "notional": 20,
        "order_id": "o",
        "created_at": created,
        "flagged_stale": True,
    }
    seed_pending_orders(ledger, [row])
    assert ledger.calls() == []
    assert InMemoryLedger.get_pending_orders(ledger)[0] == {**row, "reason": None}


def test_guard_reports_swallowed_calls_from_both_namespaces():
    from ggTrader.paper import trader

    with pytest.raises(UnstubbedPersistCall, match="get_rebalance_state"):
        with persist_guard():
            try:
                persist.get_rebalance_state()
            except UnstubbedPersistCall:
                pass
    with pytest.raises(UnstubbedPersistCall, match="log_snapshot"):
        with persist_guard():
            try:
                trader.log_snapshot()
            except UnstubbedPersistCall:
                pass


def test_guard_chains_a_preexisting_exception():
    from ggTrader.paper import trader

    with pytest.raises(UnstubbedPersistCall) as exc:
        with persist_guard():
            try:
                trader.log_snapshot()
            except UnstubbedPersistCall:
                raise RuntimeError("original failure") from None
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_make_trader_exposes_injected_ledger_in_read_only_mode():
    ledger = SpyLedger()
    trader, _, _, returned = make_trader(ledger=ledger, dry_run=True, read_only=True)
    assert returned is ledger
    assert trader._read_only
    assert trader._ledger._inner is ledger
