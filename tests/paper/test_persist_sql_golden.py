"""Golden record of the SQL, bind dictionaries and transaction events of `paper/persist.py`.

Each case runs one persist function against a recording engine (`sql_recorder.py`) and compares
the events, the returned value and the raised exception with `golden/persist_sql.json`. To record
a deliberate change, run once with ``GGT_REGEN_PERSIST_SQL=1`` (it writes the file and fails),
read the diff against `persist.py`, and rerun. The events are modeled context events, not
SQLAlchemy's real transaction semantics (see the recorder's docstring). Bind dictionary insertion
order is not part of the contract.
"""

from __future__ import annotations

import copy
import difflib
import functools
import inspect
import json
import os
import re
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from ggTrader.paper import persist
from tests.paper.sql_recorder import FakeResult, run_case

GOLDEN_PATH = Path(__file__).parent / "golden" / "persist_sql.json"
REGEN_ENV = "GGT_REGEN_PERSIST_SQL"

D_RUN = date(2026, 6, 22)
D_EX = date(2026, 6, 12)
CREATED = datetime(2026, 6, 22, 19, 45, 0)
DB_DOWN = RuntimeError("db down")


def R(rows: list | None = None, scalar: object = None) -> FakeResult:
    """A scripted result."""
    return FakeResult(rows or [], scalar)


def intent_row(**over: object) -> dict:
    """A `paper_order_intents` mapping row (synthetic values)."""
    row: dict = {
        "client_order_id": "cid-1",
        "run_date": D_RUN,
        "side": "BUY",
        "symbol": "AAPL",
        "amount": 1000.0,
        "amount_kind": "notional",
        "reason": None,
        "broker_order_id": None,
        "state": "submitted",
        "terminal_status": None,
        "filled_qty": None,
        "filled_avg_price": None,
        "created_at": CREATED,
        "updated_at": CREATED,
    }
    row.update(over)
    return row


def pending_row(**over: object) -> dict:
    """A `paper_pending_orders` mapping row (synthetic values)."""
    row: dict = {
        "order_id": "ord-1",
        "run_date": D_RUN,
        "side": "BUY",
        "symbol": "AAPL",
        "notional": 1000.0,
        "created_at": CREATED,
        "flagged_stale": False,
        "reason": None,
    }
    row.update(over)
    return row


MATCHING_TRADE = (D_RUN, "BUY", "AAPL", 1000.0, None)
SETTLE = ("cid-1", "ord-1", 10.0, 100.0, "filled")
LEGACY = ("ord-1", 10.0, 100.0, "filled")
TRADE = ("2026-06-22", "BUY", "AAPL", 1000.0, "ord-1")


def _case(fn: Callable, args: tuple, script: list, **kwargs: object) -> Callable[[], dict]:
    return lambda: run_case(fn, args, kwargs, script)


def _hold_lock(body_raises: bool = False) -> str:
    with persist.paper_run_lock() as lock:
        if body_raises:
            raise ValueError("boom")
        return type(lock).__name__


def _verify_on_connection() -> str:
    with persist._get_engine().connect() as conn:
        persist.PaperRunLock(conn).verify()
    return "held"


def _first_fails(n: int = 1) -> list:
    """A script whose `n`-th statement raises and whose earlier ones succeed."""
    return [None] * (n - 1) + [DB_DOWN]


def _settle_intent_script(trades: list | None = None, row: dict | None = None) -> list:
    """lock, pending check, intent select, trades select (+ the write results that follow)."""
    return [None, R(), R([row or intent_row()]), R(trades or [])]


def _settle_legacy_script(
    trades: list | None = None, row: dict | None = None, intent: list | None = None
) -> list:
    """lock, pending select, intent probe, trades select (+ the write results that follow)."""
    return [None, R([row or pending_row()]), R(intent or []), R(trades or [])]


# --------------------------------------------------------------------------------------- cases

CASES: dict[str, Callable[[], dict]] = {}


def _add(name: str, case: Callable[[], dict]) -> None:
    assert name not in CASES, f"duplicate case {name}"
    CASES[name] = case


# --- init_paper_schema
_add("init_paper_schema", _case(persist.init_paper_schema, (), [None] * 11))
_add("init_paper_schema__execute_fails", _case(persist.init_paper_schema, (), _first_fails(1)))
_add("init_paper_schema__execute_fails_at_6", _case(persist.init_paper_schema, (), _first_fails(6)))

# --- create_order_intent (the reference records the stray `conn` key of locals(), Deviation 5)
_INTENT_ARGS = ("cid-1", "2026-06-22", "BUY", "AAPL", 1000.0, "notional", None)
_add("create_order_intent", _case(persist.create_order_intent, _INTENT_ARGS, [None]))
_add(
    "create_order_intent__sell_with_reason",
    _case(
        persist.create_order_intent,
        ("cid-2", "2026-06-22", "SELL", "AAPL", 5.0, "qty", "exit_signal"),
        [None],
    ),
)
_add(
    "create_order_intent__execute_fails",
    _case(persist.create_order_intent, _INTENT_ARGS, _first_fails(1)),
)

# --- get_unresolved_order_intents
_add(
    "get_unresolved_order_intents",
    _case(
        persist.get_unresolved_order_intents,
        (),
        [R([intent_row(), intent_row(client_order_id="cid-2", state="intent", symbol="MSFT")])],
    ),
)
_add("get_unresolved_order_intents__empty", _case(persist.get_unresolved_order_intents, (), [R()]))

# --- attach_broker_order
_ATTACH = ("cid-1", "ord-1")
_add(
    "attach_broker_order__legacy_pending_present",
    _case(persist.attach_broker_order, _ATTACH, [None, R([(1,)])]),
)
_add(
    "attach_broker_order__missing_intent",
    _case(persist.attach_broker_order, _ATTACH, [None, R(), R()]),
)
_add(
    "attach_broker_order__conflicting_broker_id",
    _case(
        persist.attach_broker_order,
        _ATTACH,
        [None, R(), R([{"broker_order_id": "ord-other", "state": "submitted"}])],
    ),
)
_add(
    "attach_broker_order__state_not_intent",
    _case(
        persist.attach_broker_order,
        _ATTACH,
        [None, R(), R([{"broker_order_id": None, "state": "submitted"}])],
    ),
)
_add(
    "attach_broker_order__already_attached",
    _case(
        persist.attach_broker_order,
        _ATTACH,
        [None, R(), R([{"broker_order_id": "ord-1", "state": "submitted"}])],
    ),
)
_ATTACH_OK = [None, R(), R([{"broker_order_id": None, "state": "intent"}]), None]
_add("attach_broker_order", _case(persist.attach_broker_order, _ATTACH, list(_ATTACH_OK)))
_add(
    "attach_broker_order__execute_fails",
    _case(persist.attach_broker_order, _ATTACH, _first_fails(1)),
)
_add(
    "attach_broker_order__update_fails",
    _case(persist.attach_broker_order, _ATTACH, _ATTACH_OK[:3] + [DB_DOWN]),
)

# --- settle_order_intent: invalid fills raise before any engine use
_INVALID_FILLS: dict[str, tuple] = {
    "none_qty": ("cid-1", "ord-1", None, 100.0, "filled"),
    "nan_qty": ("cid-1", "ord-1", float("nan"), 100.0, "filled"),
    "negative_qty": ("cid-1", "ord-1", -1.0, 100.0, "filled"),
    "unknown_status": ("cid-1", "ord-1", 10.0, 100.0, "partially_filled"),
    "filled_zero_qty": ("cid-1", "ord-1", 0.0, None, "filled"),
    "zero_price": ("cid-1", "ord-1", 10.0, 0.0, "filled"),
    "missing_price": ("cid-1", "ord-1", 10.0, None, "filled"),
    "nonfinite_price": ("cid-1", "ord-1", 10.0, float("inf"), "filled"),
    "nonfinite_price_zero_qty": ("cid-1", "ord-1", 0.0, float("nan"), "canceled"),
    "overflowing_value": ("cid-1", "ord-1", 1e308, 1e308, "filled"),
}
for _label, _args in _INVALID_FILLS.items():
    _add(
        f"settle_order_intent__invalid_fill_{_label}",
        _case(persist.settle_order_intent, _args, []),
    )
    _add(
        f"settle_legacy_pending_order__invalid_fill_{_label}",
        _case(persist.settle_legacy_pending_order, _args[1:], []),
    )

# --- settle_order_intent: engine paths
_add(
    "settle_order_intent__missing_intent",
    _case(persist.settle_order_intent, SETTLE, [None, R(), R()]),
)
_add(
    "settle_order_intent__legacy_pending_present",
    _case(persist.settle_order_intent, SETTLE, [None, R([(1,)])]),
)
_add(
    "settle_order_intent__conflicting_broker_order",
    _case(
        persist.settle_order_intent,
        SETTLE,
        _settle_intent_script(row=intent_row(broker_order_id="ord-other"))[:3],
    ),
)
_RESOLVED = dict(
    state="resolved",
    broker_order_id="ord-1",
    filled_qty=10.0,
    filled_avg_price=100.0,
    terminal_status="filled",
)
_add(
    "settle_order_intent__already_resolved_identical",
    _case(
        persist.settle_order_intent,
        SETTLE,
        _settle_intent_script(row=intent_row(**_RESOLVED))[:3],
    ),
)
_add(
    "settle_order_intent__resolved_conflicting",
    _case(
        persist.settle_order_intent,
        SETTLE,
        _settle_intent_script(row=intent_row(**{**_RESOLVED, "filled_avg_price": 101.0}))[:3],
    ),
)
_add(
    "settle_order_intent__conflicting_trade_two_rows",
    _case(
        persist.settle_order_intent,
        SETTLE,
        _settle_intent_script(trades=[MATCHING_TRADE, MATCHING_TRADE]),
    ),
)
_add(
    "settle_order_intent__conflicting_trade_differs",
    _case(
        persist.settle_order_intent,
        SETTLE,
        _settle_intent_script(trades=[(D_RUN, "BUY", "AAPL", 999.0, None)]),
    ),
)
_add(
    "settle_order_intent__zero_fill_with_historical_trade",
    _case(
        persist.settle_order_intent,
        ("cid-1", "ord-1", 0.0, None, "canceled"),
        _settle_intent_script(trades=[MATCHING_TRADE]),
    ),
)
_add(
    "settle_order_intent__fill_inserts_trade_then_updates",
    _case(persist.settle_order_intent, SETTLE, _settle_intent_script() + [None, None]),
)
_add(
    "settle_order_intent__fill_with_matching_trade_updates_only",
    _case(persist.settle_order_intent, SETTLE, _settle_intent_script([MATCHING_TRADE]) + [None]),
)
_add(
    "settle_order_intent__zero_fill_updates_only",
    _case(
        persist.settle_order_intent,
        ("cid-1", "ord-1", 0.0, None, "canceled"),
        _settle_intent_script() + [None],
    ),
)
_add(
    "settle_order_intent__sell_with_reason",
    _case(
        persist.settle_order_intent,
        ("cid-2", "ord-2", 5.0, 200.0, "filled"),
        _settle_intent_script(
            row=intent_row(
                client_order_id="cid-2",
                side="SELL",
                amount_kind="qty",
                reason="exit_signal",
                amount=5.0,
            )
        )
        + [None, None],
    ),
)
_add(
    "settle_order_intent__execute_fails",
    _case(persist.settle_order_intent, SETTLE, _first_fails(1)),
)
_add(
    "settle_order_intent__insert_fails",
    _case(persist.settle_order_intent, SETTLE, _settle_intent_script() + [DB_DOWN]),
)
_add(
    "settle_order_intent__update_fails",
    _case(persist.settle_order_intent, SETTLE, _settle_intent_script() + [None, DB_DOWN]),
)

# --- settle_legacy_pending_order: engine paths
_add(
    "settle_legacy_pending_order__missing_row",
    _case(persist.settle_legacy_pending_order, LEGACY, [None, R()]),
)
_add(
    "settle_legacy_pending_order__intent_exists",
    _case(
        persist.settle_legacy_pending_order,
        LEGACY,
        _settle_legacy_script(intent=[(1,)])[:3],
    ),
)
_add(
    "settle_legacy_pending_order__conflicting_trade_two_rows",
    _case(
        persist.settle_legacy_pending_order,
        LEGACY,
        _settle_legacy_script(trades=[MATCHING_TRADE, MATCHING_TRADE]),
    ),
)
_add(
    "settle_legacy_pending_order__conflicting_trade_differs",
    _case(
        persist.settle_legacy_pending_order,
        LEGACY,
        _settle_legacy_script(trades=[(D_RUN, "SELL", "AAPL", 1000.0, None)]),
    ),
)
_add(
    "settle_legacy_pending_order__zero_fill_with_historical_trade",
    _case(
        persist.settle_legacy_pending_order,
        ("ord-1", 0.0, None, "canceled"),
        _settle_legacy_script(trades=[MATCHING_TRADE]),
    ),
)
_add(
    "settle_legacy_pending_order__fill_inserts_trade_then_deletes",
    _case(persist.settle_legacy_pending_order, LEGACY, _settle_legacy_script() + [None, None]),
)
_add(
    "settle_legacy_pending_order__fill_with_matching_trade_deletes_only",
    _case(
        persist.settle_legacy_pending_order,
        LEGACY,
        _settle_legacy_script([MATCHING_TRADE]) + [None],
    ),
)
_add(
    "settle_legacy_pending_order__delete_only",
    _case(
        persist.settle_legacy_pending_order,
        ("ord-1", 0.0, None, "canceled"),
        _settle_legacy_script() + [None],
    ),
)
_add(
    "settle_legacy_pending_order__execute_fails",
    _case(persist.settle_legacy_pending_order, LEGACY, _first_fails(1)),
)
_add(
    "settle_legacy_pending_order__insert_fails",
    _case(persist.settle_legacy_pending_order, LEGACY, _settle_legacy_script() + [DB_DOWN]),
)
_add(
    "settle_legacy_pending_order__delete_fails",
    _case(persist.settle_legacy_pending_order, LEGACY, _settle_legacy_script() + [None, DB_DOWN]),
)

# --- paper_run_lock and PaperRunLock.verify
_add(
    "paper_run_lock__acquired_then_released",
    _case(_hold_lock, (), [R(scalar=True), R(scalar=True), None]),
)
_add(
    "paper_run_lock__not_acquired",
    _case(_hold_lock, (), [R(scalar=False)]),
)
_add(
    "paper_run_lock__verification_false_still_unlocks",
    _case(_hold_lock, (), [R(scalar=True), R(scalar=False), None]),
)
_add(
    "paper_run_lock__body_raises_still_unlocks",
    _case(_hold_lock, (True,), [R(scalar=True), R(scalar=True), None]),
)
_add("paper_run_lock__execute_fails", _case(_hold_lock, (), _first_fails(1)))
_add(
    "paper_run_lock__PaperRunLock_verify_held",
    _case(_verify_on_connection, (), [R(scalar=True)]),
)
_add(
    "paper_run_lock__PaperRunLock_verify_lost",
    _case(_verify_on_connection, (), [R(scalar=False)]),
)

# --- log_trade
_add("log_trade__new_trade", _case(persist.log_trade, TRADE, [None, R(), None]))
_add(
    "log_trade__new_trade_with_reason",
    _case(persist.log_trade, TRADE, [None, R(), None], reason="cash_sweep"),
)
_add(
    "log_trade__identical_existing",
    _case(persist.log_trade, TRADE, [None, R([MATCHING_TRADE])]),
)
_add(
    "log_trade__identical_existing_with_reason",
    _case(
        persist.log_trade,
        TRADE,
        [None, R([(D_RUN, "BUY", "AAPL", 1000.0, "cash_sweep")])],
        reason="cash_sweep",
    ),
)
_add(
    "log_trade__conflicting_trade",
    _case(persist.log_trade, TRADE, [None, R([(D_RUN, "BUY", "AAPL", 999.0, None)])]),
)
_add(
    "log_trade__conflicting_reason",
    _case(
        persist.log_trade,
        TRADE,
        [None, R([(D_RUN, "BUY", "AAPL", 1000.0, "cash_sweep")])],
    ),
)
_add(
    "log_trade__two_existing_rows",
    _case(persist.log_trade, TRADE, [None, R([MATCHING_TRADE, MATCHING_TRADE])]),
)
_add("log_trade__execute_fails", _case(persist.log_trade, TRADE, _first_fails(1)))
_add("log_trade__insert_fails", _case(persist.log_trade, TRADE, [None, R(), DB_DOWN]))

# --- pending orders
_PENDING = ("2026-06-22", "BUY", "AAPL", 1000.0, "ord-1")
_add("log_pending_order", _case(persist.log_pending_order, _PENDING, [None]))
_add(
    "log_pending_order__with_reason",
    _case(persist.log_pending_order, _PENDING, [None], reason="cash_sweep"),
)
_add("log_pending_order__execute_fails", _case(persist.log_pending_order, _PENDING, _first_fails()))
_add(
    "get_pending_orders",
    _case(
        persist.get_pending_orders,
        (),
        [
            R(
                [
                    pending_row(),
                    pending_row(order_id="ord-2", flagged_stale=True, reason="cash_sweep"),
                ]
            )
        ],
    ),
)
_add("get_pending_orders__empty", _case(persist.get_pending_orders, (), [R()]))
_add("clear_pending_order", _case(persist.clear_pending_order, ("ord-1",), [None]))
_add(
    "clear_pending_order__execute_fails",
    _case(persist.clear_pending_order, ("ord-1",), _first_fails()),
)
_add("mark_pending_order_stale", _case(persist.mark_pending_order_stale, ("ord-1",), [None]))
_add(
    "mark_pending_order_stale__execute_fails",
    _case(persist.mark_pending_order_stale, ("ord-1",), _first_fails()),
)

# --- snapshots
_add("get_latest_snapshot__latest_row", _case(persist.get_latest_snapshot, (), [R([(101234.5,)])]))
_add("get_latest_snapshot__latest_no_row", _case(persist.get_latest_snapshot, (), [R()]))
_add(
    "get_latest_snapshot__before_row",
    _case(persist.get_latest_snapshot, (), [R([(99000.0,)])], before=D_RUN),
)
_add(
    "get_latest_snapshot__before_no_row",
    _case(persist.get_latest_snapshot, (), [R()], before=D_RUN),
)
_add("get_earliest_snapshot", _case(persist.get_earliest_snapshot, (), [R([(100000.0,)])]))
_add("get_earliest_snapshot__no_row", _case(persist.get_earliest_snapshot, (), [R()]))
_POSITIONS = {"AAPL": {"qty": 10.0, "value": 1000.0}, "MSFT": {"qty": 5.0, "value": 1500.5}}
_add(
    "log_snapshot",
    _case(persist.log_snapshot, ("2026-06-22", 101234.5, 5000.0, _POSITIONS), [None]),
)
_add(
    "log_snapshot__integer_value",
    _case(persist.log_snapshot, ("2026-06-22", 101234, 5000.0, _POSITIONS), [None]),
)
_add(
    "log_snapshot__nonfinite_position_value",
    _case(
        persist.log_snapshot,
        (
            "2026-06-22",
            101234.5,
            5000.0,
            {"AAPL": {"qty": 10.0, "value": float("nan")}, "MSFT": {"value": float("inf")}},
        ),
        [None],
    ),
)
_add(
    "log_snapshot__execute_fails",
    _case(persist.log_snapshot, ("2026-06-22", 101234.5, 5000.0, _POSITIONS), _first_fails()),
)
_add(
    "get_latest_snapshot_positions__dict_value",
    _case(persist.get_latest_snapshot_positions, (), [R([(_POSITIONS,)])]),
)
_add(
    "get_latest_snapshot_positions__json_string_value",
    _case(
        persist.get_latest_snapshot_positions,
        (),
        [R([(json.dumps(_POSITIONS),)])],
    ),
)
_add(
    "get_latest_snapshot_positions__no_row", _case(persist.get_latest_snapshot_positions, (), [R()])
)
_add(
    "get_latest_snapshot_run_date", _case(persist.get_latest_snapshot_run_date, (), [R([(D_RUN,)])])
)
_add("get_latest_snapshot_run_date__no_row", _case(persist.get_latest_snapshot_run_date, (), [R()]))
_add(
    "get_snapshot_history",
    _case(
        persist.get_snapshot_history,
        (),
        [R([(date(2026, 6, 20), _POSITIONS), (D_RUN, json.dumps({"AAPL": {"qty": 12.0}}))])],
    ),
)
_add("get_snapshot_history__empty", _case(persist.get_snapshot_history, (), [R()]))

# --- risk state, rebalance state
_add("get_peak_value", _case(persist.get_peak_value, (), [R([(105000.0,)])]))
_add("get_peak_value__no_row", _case(persist.get_peak_value, (), [R()]))
_add("save_peak_value", _case(persist.save_peak_value, (105000.0,), [None]))
_add("save_peak_value__execute_fails", _case(persist.save_peak_value, (105000.0,), _first_fails()))
_WEIGHTS = {"AAPL": 0.5, "MSFT": 0.25}
_add(
    "get_rebalance_state__dict_weights",
    _case(persist.get_rebalance_state, (), [R([(D_RUN, _WEIGHTS, 0.8)])]),
)
_add(
    "get_rebalance_state__string_weights",
    _case(persist.get_rebalance_state, (), [R([(D_RUN, json.dumps(_WEIGHTS), 0.8)])]),
)
_add("get_rebalance_state__no_row", _case(persist.get_rebalance_state, (), [R()]))
_add(
    "save_rebalance_state",
    _case(persist.save_rebalance_state, ("2026-06-22", _WEIGHTS, 0.8), [None]),
)
_add(
    "save_rebalance_state__execute_fails",
    _case(persist.save_rebalance_state, ("2026-06-22", _WEIGHTS, 0.8), _first_fails()),
)

# --- trade history reads
_add(
    "get_trade_history_dates",
    _case(persist.get_trade_history_dates, ("AAPL",), [R([(date(2026, 6, 1),), (D_RUN,)])]),
)
_add("get_trade_history_dates__empty", _case(persist.get_trade_history_dates, ("AAPL",), [R()]))
_add(
    "get_last_buy_dates",
    _case(persist.get_last_buy_dates, (), [R([("AAPL", D_RUN), ("MSFT", date(2026, 6, 1))])]),
)

# --- dividends
_add(
    "get_accrued_dividend_keys",
    _case(
        persist.get_accrued_dividend_keys,
        (),
        [R([("MSFT", date(2026, 5, 15)), ("AAPL", D_EX), ("AAPL", date(2026, 2, 9))])],
    ),
)
_add(
    "log_dividend_accrual",
    _case(persist.log_dividend_accrual, ("AAPL", D_EX, 0.25, 10.0, 2.5), [None]),
)
_add(
    "log_dividend_accrual__string_ex_date",
    _case(persist.log_dividend_accrual, ("AAPL", "2026-06-12", 0.25, 10.0, 2.5), [None]),
)
_add(
    "log_dividend_accrual__execute_fails",
    _case(persist.log_dividend_accrual, ("AAPL", D_EX, 0.25, 10.0, 2.5), _first_fails()),
)
_add(
    "get_total_dividend_accrual",
    _case(persist.get_total_dividend_accrual, (), [R([(Decimal("12.50"),)])]),
)
_add("get_total_dividend_accrual__no_row", _case(persist.get_total_dividend_accrual, (), [R()]))
_add(
    "get_total_dividend_accrual__empty_table",
    _case(persist.get_total_dividend_accrual, (), [R([(0.0,)])]),
)

# --- split state
_add(
    "get_open_split_corrections",
    _case(persist.get_open_split_corrections, (), [R([("MNST", 3.0), ("NVDA", 10.0)])]),
)
_add("get_open_split_corrections__empty", _case(persist.get_open_split_corrections, (), [R()]))
_add(
    "get_open_split_states",
    _case(persist.get_open_split_states, (), [R([("MNST", 3.0, D_EX), ("NVDA", 10.0, D_RUN)])]),
)
_add("get_open_split_states__empty", _case(persist.get_open_split_states, (), [R()]))
_add(
    "save_split_correction",
    _case(persist.save_split_correction, ("MNST", "2026-06-12", 3.0), [None]),
)
_add(
    "save_split_correction__execute_fails",
    _case(persist.save_split_correction, ("MNST", "2026-06-12", 3.0), _first_fails()),
)
_add("clear_split_correction", _case(persist.clear_split_correction, ("MNST",), [None]))
_add(
    "clear_split_correction__execute_fails",
    _case(persist.clear_split_correction, ("MNST",), _first_fails()),
)

# ------------------------------------------------------------------------------------ the tests

MUTATORS = [
    "init_paper_schema",
    "create_order_intent",
    "attach_broker_order",
    "settle_order_intent",
    "settle_legacy_pending_order",
    "log_trade",
    "log_pending_order",
    "clear_pending_order",
    "mark_pending_order_stale",
    "log_snapshot",
    "save_peak_value",
    "save_rebalance_state",
    "log_dividend_accrual",
    "save_split_correction",
    "clear_split_correction",
]

#: Shape of each public function: `name:kind[=default]` per parameter (p positional-or-keyword,
#: k keyword-only), then the return annotation. Used again after the persist split (Task 38).
SIGNATURES: dict[str, tuple[tuple[str, ...], str]] = {
    "attach_broker_order": (("client_order_id:p", "order_id:p"), "None"),
    "clear_pending_order": (("order_id:p",), "None"),
    "clear_split_correction": (("symbol:p",), "None"),
    "create_order_intent": (
        (
            "client_order_id:p",
            "run_date:p",
            "side:p",
            "symbol:p",
            "amount:p",
            "amount_kind:p",
            "reason:p",
        ),
        "None",
    ),
    "get_accrued_dividend_keys": ((), "set[tuple]"),
    "get_earliest_snapshot": ((), "float | None"),
    "get_last_buy_dates": ((), "dict[str, date]"),
    "get_latest_snapshot": (("before:k=None",), "float | None"),
    "get_latest_snapshot_positions": ((), "dict | None"),
    "get_latest_snapshot_run_date": ((), "str | None"),
    "get_open_split_corrections": ((), "dict[str, float]"),
    "get_open_split_states": ((), "dict[str, dict]"),
    "get_peak_value": ((), "float | None"),
    "get_pending_orders": ((), "list[dict]"),
    "get_rebalance_state": ((), "dict | None"),
    "get_snapshot_history": ((), "list[tuple]"),
    "get_total_dividend_accrual": ((), "float"),
    "get_trade_history_dates": (("symbol:p",), "list"),
    "get_unresolved_order_intents": ((), "list[dict]"),
    "init_paper_schema": ((), "None"),
    "log_dividend_accrual": (("symbol:p", "ex_date:p", "rate:p", "qty:p", "amount:p"), "None"),
    "log_pending_order": (
        ("run_date:p", "side:p", "symbol:p", "notional:p", "order_id:p", "reason:p=None"),
        "None",
    ),
    "log_snapshot": (("run_date:p", "portfolio_value:p", "cash:p", "positions:p"), "None"),
    "log_trade": (
        ("run_date:p", "side:p", "symbol:p", "amount:p", "order_id:p", "reason:p=None"),
        "None",
    ),
    "mark_pending_order_stale": (("order_id:p",), "None"),
    "paper_run_lock": ((), "Iterator[PaperRunLock]"),
    "save_peak_value": (("peak_value:p",), "None"),
    "save_rebalance_state": (("rebalance_date:p", "weights:p", "scale:p"), "None"),
    "save_split_correction": (("symbol:p", "ex_date:p", "factor:p"), "None"),
    "settle_legacy_pending_order": (
        ("order_id:p", "filled_qty:p", "filled_avg_price:p", "terminal_status:p"),
        "None",
    ),
    "settle_order_intent": (
        (
            "client_order_id:p",
            "order_id:p",
            "filled_qty:p",
            "filled_avg_price:p",
            "terminal_status:p",
        ),
        "None",
    ),
}


@functools.cache
def record(name: str) -> dict:
    return CASES[name]()


def _public_persist_functions() -> dict[str, Callable]:
    return {
        name: fn
        for name, fn in inspect.getmembers(persist, inspect.isfunction)
        if not name.startswith("_") and fn.__module__ == persist.__name__
    }


def _shape(fn: Callable) -> tuple[tuple[str, ...], str]:
    kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "p",
        inspect.Parameter.KEYWORD_ONLY: "k",
        inspect.Parameter.POSITIONAL_ONLY: "o",
        inspect.Parameter.VAR_POSITIONAL: "*",
        inspect.Parameter.VAR_KEYWORD: "**",
    }
    params = tuple(
        f"{p.name}:{kinds[p.kind]}"
        + ("" if p.default is inspect.Parameter.empty else f"={p.default!r}")
        for p in inspect.signature(fn).parameters.values()
    )
    return params, str(inspect.signature(fn).return_annotation)


def _dumps(cases: dict) -> str:
    return json.dumps({"cases": cases}, indent=1, sort_keys=False)


def _diff(expected: object, actual: object, name: str) -> str:
    lines = list(
        difflib.unified_diff(
            json.dumps(expected, indent=1).splitlines(),
            json.dumps(actual, indent=1).splitlines(),
            fromfile=f"golden/persist_sql.json[{name}]",
            tofile=f"{name} (this run)",
            lineterm="",
        )
    )
    shown = lines[:120]
    more = f"\n... {len(lines) - 120} more diff lines" if len(lines) > 120 else ""
    return "\n".join(shown) + more


def _golden() -> dict:
    assert GOLDEN_PATH.exists(), f"no golden: run once with {REGEN_ENV}=1 and review it"
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def normalize_reference(name: str, entry: dict) -> dict:
    """The reference entry with the stray `conn` bind key removed from `create_order_intent`.

    Deviation 5: the old `create_order_intent` passed `locals()` as its bind dictionary, which
    also carried the connection under the key `conn`; the SQL never binds it. Since Task 38 the
    function passes exactly the seven named parameters. The reference file is never edited and
    still records `conn` (`test_persist_ledger.py` checks that); this is the one explicit
    normalization, applied before comparing: it touches the `params` of every `execute` event of
    the cases of `create_order_intent` and nothing else.
    """
    result = copy.deepcopy(entry)
    if name.split("__")[0] == "create_order_intent":
        for event in result["events"]:
            if event["op"] == "execute" and event["params"] is not None:
                event["params"].pop("conn", None)
    return result


@pytest.mark.parametrize("name", sorted(CASES))
def test_persist_sql_case_matches_golden(name: str) -> None:
    actual = record(name)
    if os.environ.get(REGEN_ENV):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(_dumps({n: record(n) for n in CASES}) + "\n", encoding="utf-8")
        pytest.fail(
            f"regenerated {GOLDEN_PATH.name}: review the diff and rerun without {REGEN_ENV}"
        )
    golden = _golden()
    assert name in golden, f"no golden entry for {name}"
    expected = normalize_reference(name, golden[name])
    assert actual == expected, f"{name} differs from its golden:\n" + _diff(expected, actual, name)


def test_every_case_has_a_golden_entry() -> None:
    assert set(CASES) <= set(_golden()), sorted(set(CASES) - set(_golden()))


def test_no_orphan_entries() -> None:
    assert set(_golden()) <= set(CASES), sorted(set(_golden()) - set(CASES))


def test_every_public_persist_function_has_a_case() -> None:
    functions = _public_persist_functions()
    assert len(functions) == 31
    covered = {name.split("__")[0] for name in CASES}
    assert set(functions) <= covered, sorted(set(functions) - covered)
    assert covered <= set(functions), sorted(covered - set(functions))


def test_persist_signatures_are_frozen() -> None:
    actual = {name: _shape(fn) for name, fn in _public_persist_functions().items()}
    assert actual == SIGNATURES


def test_every_mutator_has_an_execution_failure_case() -> None:
    missing = [name for name in MUTATORS if f"{name}__execute_fails" not in CASES]
    assert missing == []
    assert set(MUTATORS) <= set(_public_persist_functions())


# ------------------------------------------ what each case's name claims about its record

#: Cases that raise, with the exact message; every other non-failure case must not raise.
RAISES: dict[str, str] = {
    "attach_broker_order__legacy_pending_present": (
        "ValueError: Order has both pending and intent representations"
    ),
    "attach_broker_order__missing_intent": "ValueError: Missing intent or conflicting broker order",
    "attach_broker_order__conflicting_broker_id": (
        "ValueError: Missing intent or conflicting broker order"
    ),
    "attach_broker_order__state_not_intent": "ValueError: Invalid intent state",
    "settle_order_intent__missing_intent": "ValueError: Missing intent or conflicting broker order",
    "settle_order_intent__legacy_pending_present": (
        "ValueError: Order has both pending and intent representations"
    ),
    "settle_order_intent__conflicting_broker_order": (
        "ValueError: Missing intent or conflicting broker order"
    ),
    "settle_order_intent__resolved_conflicting": "ValueError: Conflicting settlement",
    "settle_order_intent__conflicting_trade_two_rows": "ValueError: Conflicting historical trade",
    "settle_order_intent__conflicting_trade_differs": "ValueError: Conflicting historical trade",
    "settle_order_intent__zero_fill_with_historical_trade": (
        "ValueError: Conflicting historical trade"
    ),
    "settle_legacy_pending_order__missing_row": "ValueError: Missing legacy pending order",
    "settle_legacy_pending_order__intent_exists": (
        "ValueError: Order has both pending and intent representations"
    ),
    "settle_legacy_pending_order__conflicting_trade_two_rows": (
        "ValueError: Conflicting historical trade"
    ),
    "settle_legacy_pending_order__conflicting_trade_differs": (
        "ValueError: Conflicting historical trade"
    ),
    "settle_legacy_pending_order__zero_fill_with_historical_trade": (
        "ValueError: Conflicting historical trade"
    ),
    "paper_run_lock__not_acquired": "RuntimeError: Paper run already locked",
    "paper_run_lock__verification_false_still_unlocks": "RuntimeError: Paper run lock lost",
    "paper_run_lock__body_raises_still_unlocks": "ValueError: boom",
    "paper_run_lock__PaperRunLock_verify_lost": "RuntimeError: Paper run lock lost",
    "log_trade__conflicting_trade": "ValueError: Conflicting historical trade",
    "log_trade__conflicting_reason": "ValueError: Conflicting historical trade",
    "log_trade__two_existing_rows": "ValueError: Conflicting historical trade",
}
_FAILURE_NAME = re.compile(r"__\w*fails\w*$")


def _ops(record_: dict) -> list[str]:
    return [event["op"] for event in record_["events"]]


def _writes(record_: dict, verb: str) -> int:
    return sum(
        1
        for event in record_["events"]
        if event["op"] == "execute" and event["sql"].lstrip().upper().startswith(verb)
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_shows_what_its_name_claims(name: str) -> None:
    rec = record(name)
    ops = _ops(rec)
    if "__invalid_fill_" in name:
        assert rec["events"] == []
        assert rec["raised"].startswith("ValueError: Invalid ")
        return
    if _FAILURE_NAME.search(name):
        assert rec["raised"] == "RuntimeError: db down"
        assert "commit" not in ops
        assert rec["events"][-1] in ({"op": "close", "committed": False}, {"op": "rollback"})
        assert ops.count("execute") >= 1
        return
    if name in RAISES:
        assert rec["raised"] == RAISES[name]
        assert rec["events"][-1] in (
            {"op": "close", "committed": False},
            {"op": "rollback"},
        ), "a raising case must end without a commit"
        return
    assert rec["raised"] is None, f"{name} raised {rec['raised']}"
    if name.startswith("log_trade__new_trade"):
        assert ops.count("commit") == 1 and _writes(rec, "INSERT") == 1
        assert rec["events"][-1] == {"op": "close", "committed": True}
    if name.startswith("log_trade__identical_existing"):
        assert "commit" not in ops and _writes(rec, "INSERT") == 0
        assert rec["events"][-1] == {"op": "close", "committed": False}
    if name == "init_paper_schema":
        assert ops.count("execute") == 11 and ops.count("commit") == 1
    if name.endswith("__fill_inserts_trade_then_updates") or name.endswith(
        "__fill_inserts_trade_then_deletes"
    ):
        assert _writes(rec, "INSERT") == 1
    if name.endswith("_updates_only") or name.endswith("__delete_only"):
        assert _writes(rec, "INSERT") == 0
    if (
        name.endswith("__already_resolved_identical")
        or name == "attach_broker_order__already_attached"
    ):
        assert _writes(rec, "UPDATE") == 0 and _writes(rec, "INSERT") == 0
    if name == "create_order_intent":
        params = rec["events"][1]["params"]
        assert "conn" not in params and len(params) == 7  # the stray key of Deviation 5 is gone


@pytest.mark.parametrize("name", sorted(CASES))
def test_events_are_well_formed(name: str) -> None:
    rec = record(name)
    events = rec["events"]
    if not events:
        return
    first, last = events[0]["op"], events[-1]["op"]
    assert sum(1 for e in events if e["op"] in ("connect", "begin")) == 1
    if first == "connect":
        assert last == "close"
    else:
        assert first == "begin"
        assert last == ("commit" if rec["raised"] is None else "rollback")


def test_golden_has_no_bare_nonfinite_numbers() -> None:
    """Floats are tagged strings, so the file never holds a bare NaN or Infinity number."""

    def refuse(token: str) -> None:
        raise AssertionError(f"bare {token} in golden")

    json.loads(GOLDEN_PATH.read_text(encoding="utf-8"), parse_constant=refuse)
