"""Characterization of the paper persist writers against a real, throwaway PostgreSQL database.

Every test creates its own `ggt_scratch_<8 hex>` database (see `persist_scratch_db.py`), applies
the paper tables' DDL from `persist._SCHEMA`, points `persist._get_engine` at it, and drops it.
The tests are marked `integration` and skip when `GGT_SCRATCH_DB_ADMIN_URL` is not set; run them
only through `with_scratch_db.py`:

    with_scratch_db.py scripts/run_tests.sh tests/paper/test_persist_real_db.py -m integration -q

What they pin, for each of the 15 writers and the 15 readers: the rows a write leaves behind (read
back with the reader functions and with raw SQL on a second connection), and, for every
multi-statement writer, exactly what is and is not committed when a later statement fails. The
same file ran on the `connect()` + `commit()` writers and again after they moved to
`engine.begin()`; set `GGT_PERSIST_CHAR_OUT` to a path to also write every observation as JSON,
so the two runs can be compared byte for byte.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from ggTrader.paper import persist
from tests.paper.persist_scratch_db import scratch_db

pytestmark = pytest.mark.integration

D1 = "2026-06-22"
D2 = "2026-06-23"
EX = date(2026, 6, 12)

PAPER_TABLES = {
    "paper_order_intents",
    "paper_trades",
    "paper_snapshots",
    "paper_pending_orders",
    "paper_rebalance_state",
    "paper_risk_state",
    "paper_dividend_accruals",
    "paper_split_state",
}

OBSERVATIONS: dict[str, Any] = {}


@pytest.fixture(scope="module", autouse=True)
def _write_observations() -> Iterator[None]:
    """Write the observations as JSON when `GGT_PERSIST_CHAR_OUT` names a file."""
    yield
    target = os.environ.get("GGT_PERSIST_CHAR_OUT")
    if target:
        Path(target).write_text(
            json.dumps(OBSERVATIONS, indent=1, sort_keys=True, default=str) + "\n"
        )


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    with scratch_db() as engine:
        monkeypatch.setattr(persist, "_get_engine", lambda: engine)
        yield engine


@pytest.fixture
def bare_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """A scratch database with no paper tables, for `init_paper_schema`."""
    with scratch_db(schema=False) as engine:
        monkeypatch.setattr(persist, "_get_engine", lambda: engine)
        yield engine


def observe(request: pytest.FixtureRequest, **facts: Any) -> None:
    OBSERVATIONS[request.node.name] = facts


def rows(engine: Engine, sql: str, **params: Any) -> list[tuple]:
    """Raw rows on a connection of their own, so only committed data is seen."""
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(sql), params)]


def tables(engine: Engine) -> set[str]:
    return {
        name
        for (name,) in rows(
            engine,
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'",
        )
    }


def counts(engine: Engine) -> dict[str, int]:
    return {t: rows(engine, f"SELECT count(*) FROM {t}")[0][0] for t in sorted(PAPER_TABLES)}


def violation(exc: BaseException) -> str:
    """The SQLSTATE of a database error, or the class name of anything else."""
    if isinstance(exc, DBAPIError) and getattr(exc.orig, "pgcode", None):
        return f"{type(exc).__name__}/{exc.orig.pgcode}"
    return type(exc).__name__


def xact_lock_free(engine: Engine, order_id: str) -> bool:
    """True when the per-order transaction advisory lock can be taken, so no txn still holds it."""
    with engine.begin() as conn:
        return conn.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:id, 0))"), {"id": order_id}
        ).scalar_one()


# ---------------------------------------------------------------------- the 11 hand-commit writers


def test_init_paper_schema_creates_the_tables_and_is_idempotent(bare_db, request):
    assert tables(bare_db) == set()
    persist.init_paper_schema()
    assert tables(bare_db) == PAPER_TABLES
    persist.init_paper_schema()
    assert tables(bare_db) == PAPER_TABLES
    cols = rows(
        bare_db,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'paper_pending_orders' ORDER BY ordinal_position",
    )
    assert [c[0] for c in cols] == [
        "order_id",
        "run_date",
        "side",
        "symbol",
        "notional",
        "created_at",
        "flagged_stale",
        "reason",
    ]
    observe(request, tables=sorted(tables(bare_db)), pending_columns=[c[0] for c in cols])


def test_init_paper_schema_failing_late_statement_commits_nothing(bare_db, request):
    """Statement 6 (ALTER on a view) fails after five CREATEs: today none of them survives."""
    with bare_db.begin() as conn:
        conn.execute(text("CREATE VIEW paper_pending_orders AS SELECT 1 AS order_id"))
    with pytest.raises(DBAPIError) as err:
        persist.init_paper_schema()
    assert violation(err.value) == "ProgrammingError/42809"  # wrong object type
    assert tables(bare_db) == set()  # the view is not a BASE TABLE, and no paper table exists
    views = rows(
        bare_db, "SELECT table_name FROM information_schema.views WHERE table_schema='public'"
    )
    assert views == [("paper_pending_orders",)]
    observe(request, raised=violation(err.value), base_tables=sorted(tables(bare_db)), views=views)


def test_log_trade_roundtrip_idempotence_and_conflict(db, request):
    persist.log_trade(D1, "BUY", "AAPL", 1000.0, "ord-1")
    persist.log_trade(D2, "SELL", "MSFT", 50.0, "ord-2", reason="cash_sweep")
    assert persist.get_trade_history_dates("AAPL") == [date(2026, 6, 22)]
    assert persist.get_last_buy_dates() == {"AAPL": date(2026, 6, 22)}
    stored = rows(
        db, "SELECT run_date, side, symbol, amount, order_id, reason FROM paper_trades ORDER BY id"
    )
    assert stored == [
        (date(2026, 6, 22), "BUY", "AAPL", 1000.0, "ord-1", None),
        (date(2026, 6, 23), "SELL", "MSFT", 50.0, "ord-2", "cash_sweep"),
    ]
    # An identical replay returns without writing (the early `return`).
    assert persist.log_trade(D1, "BUY", "AAPL", 1000.0, "ord-1") is None
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(2,)]
    assert xact_lock_free(db, "ord-1")
    # A conflicting replay raises and changes nothing.
    with pytest.raises(ValueError, match="Conflicting historical trade"):
        persist.log_trade(D1, "BUY", "AAPL", 999.0, "ord-1")
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(2,)]
    assert xact_lock_free(db, "ord-1")
    observe(request, stored=stored, last_buys=persist.get_last_buy_dates())


def test_log_trade_failing_insert_commits_nothing_and_frees_the_lock(db, request):
    """Statement 3 (the INSERT) violates NOT NULL after the lock and the SELECT ran."""
    with pytest.raises(DBAPIError) as err:
        persist.log_trade(D1, "BUY", "AAPL", None, "ord-1")
    assert violation(err.value) == "IntegrityError/23502"  # not_null_violation
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(0,)]
    assert xact_lock_free(db, "ord-1")
    persist.log_trade(D1, "BUY", "AAPL", 1000.0, "ord-1")  # the ledger is usable afterwards
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(1,)]
    observe(request, raised=violation(err.value), rows_after_failure=0, rows_after_retry=1)


def test_log_trade_failing_second_statement_commits_nothing(db, request):
    """Statement 2 (the SELECT) fails once the lock statement succeeded."""
    with db.begin() as conn:
        conn.execute(text("ALTER TABLE paper_trades RENAME COLUMN reason TO reason_renamed"))
    with pytest.raises(DBAPIError) as err:
        persist.log_trade(D1, "BUY", "AAPL", 1000.0, "ord-1")
    assert violation(err.value) == "ProgrammingError/42703"  # undefined_column
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(0,)]
    assert xact_lock_free(db, "ord-1")
    observe(request, raised=violation(err.value), rows_after_failure=0)


def test_pending_order_roundtrip_stale_flag_and_clear(db, request):
    persist.log_pending_order(D1, "BUY", "AAPL", 1000.0, "ord-1")
    persist.log_pending_order(D1, "SELL", "MSFT", 50.0, "ord-2", reason="cash_sweep")
    persist.log_pending_order(D2, "BUY", "ZZZZ", 1.0, "ord-1")  # ON CONFLICT DO NOTHING
    pending = persist.get_pending_orders()
    assert [
        (p["order_id"], p["side"], p["symbol"], p["notional"], p["reason"]) for p in pending
    ] == [
        ("ord-1", "BUY", "AAPL", 1000.0, None),
        ("ord-2", "SELL", "MSFT", 50.0, "cash_sweep"),
    ]
    assert [p["flagged_stale"] for p in pending] == [False, False]
    persist.mark_pending_order_stale("ord-1")
    persist.mark_pending_order_stale("nope")  # no row: no error, no change
    assert [p["flagged_stale"] for p in persist.get_pending_orders()] == [True, False]
    persist.clear_pending_order("ord-1")
    persist.clear_pending_order("nope")
    left = persist.get_pending_orders()
    assert [p["order_id"] for p in left] == ["ord-2"]
    observe(
        request,
        before_clear=[(p["order_id"], p["symbol"], p["flagged_stale"]) for p in pending],
        after_clear=[(p["order_id"], p["flagged_stale"]) for p in left],
    )


def test_snapshot_roundtrip_upsert_and_readers(db, request):
    assert persist.get_latest_snapshot() is None
    assert persist.get_earliest_snapshot() is None
    assert persist.get_latest_snapshot_positions() is None
    assert persist.get_latest_snapshot_run_date() is None
    persist.log_snapshot(D1, 100000.0, 40000.0, {"AAPL": {"qty": 10.0}})
    persist.log_snapshot(
        D2, 101000.0, 39000.0, {"AAPL": {"qty": 10.0}, "MSFT": {"qty": float("nan")}}
    )
    persist.log_snapshot(D2, 102000.0, 38000.0, {"AAPL": {"qty": 12.0}})  # upsert
    assert persist.get_latest_snapshot() == 102000.0
    assert persist.get_latest_snapshot(before=date(2026, 6, 23)) == 100000.0
    assert persist.get_earliest_snapshot() == 100000.0
    assert persist.get_latest_snapshot_positions() == {"AAPL": {"qty": 12.0}}
    assert persist.get_latest_snapshot_run_date() == D2
    history = persist.get_snapshot_history()
    assert history == [
        (date(2026, 6, 22), {"AAPL": {"qty": 10.0}}),
        (date(2026, 6, 23), {"AAPL": {"qty": 12.0}}),
    ]
    assert rows(db, "SELECT count(*) FROM paper_snapshots") == [(2,)]
    observe(request, history=history, latest=102000.0, before=100000.0)


def test_snapshot_nan_positions_are_stored_as_null(db, request):
    persist.log_snapshot(D1, 1.0, 1.0, {"X": {"qty": float("nan")}})
    assert persist.get_latest_snapshot_positions() == {"X": {"qty": None}}
    observe(request, positions=persist.get_latest_snapshot_positions())


def test_peak_value_upsert_keeps_one_row(db, request):
    assert persist.get_peak_value() is None
    persist.save_peak_value(105000.0)
    persist.save_peak_value(110000.0)
    assert persist.get_peak_value() == 110000.0
    assert rows(db, "SELECT id, peak_value FROM paper_risk_state") == [(1, 110000.0)]
    observe(request, rows=rows(db, "SELECT id, peak_value FROM paper_risk_state"))


def test_rebalance_state_upsert_and_reader(db, request):
    assert persist.get_rebalance_state() is None
    persist.save_rebalance_state(D1, {"AAPL": 0.5}, 0.9)
    persist.save_rebalance_state(D2, {"AAPL": 0.25, "MSFT": 0.25}, 0.8)
    state = persist.get_rebalance_state()
    assert state == {
        "rebalance_date": "2026-06-23",
        "weights": {"AAPL": 0.25, "MSFT": 0.25},
        "scale": 0.8,
    }
    assert rows(db, "SELECT count(*) FROM paper_rebalance_state") == [(1,)]
    observe(request, state=state)


def test_dividend_accrual_roundtrip_and_idempotence(db, request):
    assert persist.get_accrued_dividend_keys() == set()
    assert persist.get_total_dividend_accrual() == 0.0
    persist.log_dividend_accrual("AAPL", EX, 0.25, 10.0, 2.5)
    persist.log_dividend_accrual("AAPL", "2026-06-12", 9.0, 9.0, 81.0)  # same key: no-op
    persist.log_dividend_accrual("MSFT", date(2026, 6, 13), 0.75, 4.0, 3.0)
    assert persist.get_accrued_dividend_keys() == {("AAPL", EX), ("MSFT", date(2026, 6, 13))}
    assert persist.get_total_dividend_accrual() == 5.5
    observe(request, keys=sorted(persist.get_accrued_dividend_keys()), total=5.5)


def test_split_correction_roundtrip_upsert_and_clear(db, request):
    assert persist.get_open_split_corrections() == {}
    persist.save_split_correction("MNST", "2026-08-11", 2.0)
    persist.save_split_correction("MNST", "2026-08-11", 3.0)  # refreshes the factor
    persist.save_split_correction("NVDA", "2026-06-10", 10.0)
    assert persist.get_open_split_corrections() == {"MNST": 3.0, "NVDA": 10.0}
    states = persist.get_open_split_states()
    assert states == {
        "MNST": {"factor": 3.0, "ex_date": date(2026, 8, 11)},
        "NVDA": {"factor": 10.0, "ex_date": date(2026, 6, 10)},
    }
    persist.clear_split_correction("MNST")
    persist.clear_split_correction("nope")
    assert persist.get_open_split_corrections() == {"NVDA": 10.0}
    observe(request, states=states, after_clear=persist.get_open_split_corrections())


@pytest.mark.parametrize(
    ("name", "call", "table"),
    [
        (
            "log_pending_order",
            lambda: persist.log_pending_order(D1, "BUY", "A", None, "o"),
            "paper_pending_orders",
        ),
        ("log_snapshot", lambda: persist.log_snapshot(D1, None, 1.0, {}), "paper_snapshots"),
        ("save_peak_value", lambda: persist.save_peak_value(None), "paper_risk_state"),
        (
            "log_dividend_accrual",
            lambda: persist.log_dividend_accrual("A", EX, None, 1.0, 1.0),
            "paper_dividend_accruals",
        ),
        (
            "save_rebalance_state",
            lambda: persist.save_rebalance_state(D1, {"A": 1.0}, None),
            "paper_rebalance_state",
        ),
        (
            "save_split_correction",
            lambda: persist.save_split_correction("A", D1, None),
            "paper_split_state",
        ),
    ],
)
def test_single_statement_writers_leave_nothing_when_the_statement_fails(
    db, request, name, call, table
):
    with pytest.raises(DBAPIError) as err:
        call()
    assert violation(err.value) == "IntegrityError/23502", name  # not_null_violation
    assert rows(db, f"SELECT count(*) FROM {table}") == [(0,)]
    observe(request, raised=violation(err.value), rows=0)


# -------------------------------------------------------------------------- the 4 begin() writers


def test_order_intent_lifecycle_roundtrip(db, request):
    persist.create_order_intent("cid-1", D1, "BUY", "AAPL", 1000.0, "notional", None)
    persist.create_order_intent("cid-2", D1, "SELL", "MSFT", 5.0, "qty", "exit_signal")
    assert [r["client_order_id"] for r in persist.get_unresolved_order_intents()] == [
        "cid-1",
        "cid-2",
    ]
    persist.attach_broker_order("cid-1", "ord-1")
    persist.attach_broker_order("cid-1", "ord-1")  # already attached: no-op
    persist.settle_order_intent("cid-1", "ord-1", 10.0, 100.0, "filled")
    persist.settle_order_intent("cid-1", "ord-1", 10.0, 100.0, "filled")  # identical replay
    persist.settle_order_intent("cid-2", "ord-2", 0.0, None, "canceled")
    assert persist.get_unresolved_order_intents() == []
    intents = rows(
        db,
        "SELECT client_order_id, broker_order_id, state, terminal_status, filled_qty, "
        "filled_avg_price FROM paper_order_intents ORDER BY client_order_id",
    )
    trades = rows(db, "SELECT run_date, side, symbol, amount, order_id, reason FROM paper_trades")
    assert intents == [
        ("cid-1", "ord-1", "resolved", "filled", 10.0, 100.0),
        ("cid-2", "ord-2", "resolved", "canceled", 0.0, None),
    ]
    assert trades == [(date(2026, 6, 22), "BUY", "AAPL", 1000.0, "ord-1", None)]
    observe(request, intents=intents, trades=trades)


def test_order_intent_rejections_change_nothing(db, request):
    persist.create_order_intent("cid-1", D1, "BUY", "AAPL", 1000.0, "notional", None)
    before = counts(db)
    with pytest.raises(ValueError, match="Missing intent"):
        persist.attach_broker_order("cid-missing", "ord-9")
    with pytest.raises(ValueError, match="Invalid terminal fill"):
        persist.settle_order_intent("cid-1", "ord-1", 10.0, None, "filled")
    with pytest.raises(DBAPIError) as err:
        persist.create_order_intent(
            "cid-1", D1, "BUY", "AAPL", 1.0, "notional", None
        )  # duplicate PK
    assert violation(err.value) == "IntegrityError/23505"  # unique_violation
    assert counts(db) == before
    observe(request, counts=before, duplicate=violation(err.value))


def test_attach_broker_order_leaves_the_intent_untouched_when_its_update_fails(db, request):
    """A refused UPDATE (a trigger raises) leaves the intent with no broker id and its state."""
    persist.create_order_intent("cid-1", D1, "BUY", "AAPL", 1000.0, "notional", None)
    with db.begin() as conn:
        conn.execute(
            text(
                "CREATE FUNCTION refuse() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'refused'; END $$"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER refuse_update BEFORE UPDATE ON paper_order_intents "
                "FOR EACH ROW EXECUTE FUNCTION refuse()"
            )
        )
    with pytest.raises(DBAPIError) as err:
        persist.attach_broker_order("cid-1", "ord-1")
    assert violation(err.value) == "InternalError/P0001"  # the trigger's RAISE EXCEPTION
    assert rows(db, "SELECT state, broker_order_id FROM paper_order_intents") == [("intent", None)]
    assert xact_lock_free(db, "ord-1")
    observe(request, raised=violation(err.value), state="intent", broker_order_id=None)


def test_settle_order_intent_is_atomic_when_the_last_update_fails(db, request):
    """The trade INSERT and the intent UPDATE share a transaction; a failing UPDATE undoes both."""
    persist.create_order_intent("cid-1", D1, "BUY", "AAPL", 1000.0, "notional", None)
    persist.attach_broker_order("cid-1", "ord-1")
    with db.begin() as conn:
        conn.execute(
            text(
                "CREATE FUNCTION refuse() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'refused'; END $$"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER refuse_update BEFORE UPDATE ON paper_order_intents "
                "FOR EACH ROW EXECUTE FUNCTION refuse()"
            )
        )
    with pytest.raises(DBAPIError) as err:
        persist.settle_order_intent("cid-1", "ord-1", 10.0, 100.0, "filled")
    assert violation(err.value) == "InternalError/P0001"  # the trigger's RAISE EXCEPTION
    assert rows(db, "SELECT count(*) FROM paper_trades") == [(0,)]
    assert rows(db, "SELECT state FROM paper_order_intents") == [("submitted",)]
    assert xact_lock_free(db, "ord-1")
    observe(request, raised=violation(err.value), trades=0, state="submitted")


def test_legacy_pending_settlement_roundtrip_and_atomicity(db, request):
    persist.log_pending_order(D1, "BUY", "AAPL", 1000.0, "ord-1")
    persist.log_pending_order(D1, "BUY", "MSFT", 500.0, "ord-2")
    persist.settle_legacy_pending_order("ord-1", 10.0, 100.0, "filled")
    assert [p["order_id"] for p in persist.get_pending_orders()] == ["ord-2"]
    trades = rows(db, "SELECT symbol, amount, order_id FROM paper_trades")
    assert trades == [("AAPL", 1000.0, "ord-1")]
    with db.begin() as conn:
        conn.execute(
            text(
                "CREATE FUNCTION refuse() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'refused'; END $$"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER refuse_delete BEFORE DELETE ON paper_pending_orders "
                "FOR EACH ROW EXECUTE FUNCTION refuse()"
            )
        )
    with pytest.raises(DBAPIError) as err:
        persist.settle_legacy_pending_order("ord-2", 5.0, 100.0, "filled")
    assert violation(err.value) == "InternalError/P0001"  # the trigger's RAISE EXCEPTION
    assert rows(db, "SELECT symbol, order_id FROM paper_trades ORDER BY id") == [("AAPL", "ord-1")]
    assert [p["order_id"] for p in persist.get_pending_orders()] == ["ord-2"]
    observe(request, trades=trades, raised=violation(err.value), pending_left=["ord-2"])


# ------------------------------------------------------------------------------- the session lock


def test_paper_run_lock_is_exclusive_and_released(db, request):
    with persist.paper_run_lock() as lock:
        lock.verify()
        with pytest.raises(RuntimeError, match="already locked"):
            with persist.paper_run_lock():
                pytest.fail("a second holder must not get the lock")
    with persist.paper_run_lock() as lock:  # free again after the first block
        lock.verify()
    with pytest.raises(ValueError, match="boom"):
        with persist.paper_run_lock():
            raise ValueError("boom")
    with persist.paper_run_lock():
        pass
    observe(request, exclusive=True, released_on_exit=True, released_on_error=True)
