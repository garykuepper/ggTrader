"""TimescaleDB persistence for paper trading trades and snapshots."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

from sqlalchemy import text

from ggTrader.lab.persist import get_engine as _get_engine
from ggTrader.utils.json_safe import json_safe

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_order_intents (
    client_order_id TEXT PRIMARY KEY,
    run_date DATE NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    symbol TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL CHECK (amount > 0 AND amount < 'Infinity'::float8),
    amount_kind TEXT NOT NULL CHECK ((side = 'BUY' AND amount_kind = 'notional') OR
                                     (side = 'SELL' AND amount_kind = 'qty')),
    reason TEXT,
    broker_order_id TEXT UNIQUE,
    state TEXT NOT NULL DEFAULT 'intent' CHECK (state IN ('intent', 'submitted', 'resolved')),
    terminal_status TEXT,
    filled_qty DOUBLE PRECISION,
    filled_avg_price DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    side TEXT NOT NULL,
    symbol TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    order_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason TEXT
);
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS reason TEXT;
CREATE TABLE IF NOT EXISTS paper_snapshots (
    run_date DATE PRIMARY KEY,
    portfolio_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_pending_orders (
    order_id TEXT PRIMARY KEY,
    run_date DATE NOT NULL,
    side TEXT NOT NULL,
    symbol TEXT NOT NULL,
    notional DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    flagged_stale BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT
);
ALTER TABLE paper_pending_orders
    ADD COLUMN IF NOT EXISTS flagged_stale BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE paper_pending_orders ADD COLUMN IF NOT EXISTS reason TEXT;
CREATE TABLE IF NOT EXISTS paper_rebalance_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    rebalance_date DATE NOT NULL,
    weights JSONB NOT NULL,
    scale DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);
CREATE TABLE IF NOT EXISTS paper_risk_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    peak_value DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);
CREATE TABLE IF NOT EXISTS paper_dividend_accruals (
    symbol TEXT NOT NULL,
    ex_date DATE NOT NULL,
    rate DOUBLE PRECISION NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, ex_date)
);
CREATE TABLE IF NOT EXISTS paper_split_state (
    symbol TEXT PRIMARY KEY,
    ex_date DATE NOT NULL,
    factor DOUBLE PRECISION NOT NULL,
    first_detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_paper_schema() -> None:
    with _get_engine().connect() as conn:
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()


_LOCK_KEY = 1734837291
_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected"}


def create_order_intent(
    client_order_id: str,
    run_date: str,
    side: str,
    symbol: str,
    amount: float,
    amount_kind: str,
    reason: str | None,
) -> None:
    """Commit an order intent before contacting the broker."""
    with _get_engine().begin() as conn:
        conn.execute(
            text("""INSERT INTO paper_order_intents
            (client_order_id, run_date, side, symbol, amount, amount_kind, reason)
            VALUES (:client_order_id, :run_date, :side, :symbol, :amount, :amount_kind, :reason)
        """),
            locals(),
        )


def get_unresolved_order_intents() -> list[dict]:
    """Return durable intents that still require reconciliation."""
    with _get_engine().connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                text(
                    "SELECT * FROM paper_order_intents WHERE state != 'resolved' "
                    "ORDER BY created_at"
                )
            ).mappings()
        ]


def _lock_order_and_reject_pending(conn, order_id: str) -> None:
    """Serialize representations and reject an existing legacy pending row."""
    conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:id, 0))"), {"id": order_id})
    if (
        conn.execute(
            text("SELECT 1 FROM paper_pending_orders WHERE order_id = :id"), {"id": order_id}
        ).first()
        is not None
    ):
        raise ValueError("Order has both pending and intent representations")


def attach_broker_order(client_order_id: str, order_id: str) -> None:
    """Attach a broker ID, rejecting missing or conflicting intents."""
    with _get_engine().begin() as conn:
        _lock_order_and_reject_pending(conn, order_id)
        row = (
            conn.execute(
                text(
                    "SELECT broker_order_id, state FROM paper_order_intents "
                    "WHERE client_order_id = :id FOR UPDATE"
                ),
                {"id": client_order_id},
            )
            .mappings()
            .first()
        )
        if row is None or (row["broker_order_id"] not in (None, order_id)):
            raise ValueError("Missing intent or conflicting broker order")
        if row["broker_order_id"] is None:
            if row["state"] != "intent":
                raise ValueError("Invalid intent state")
            conn.execute(
                text(
                    "UPDATE paper_order_intents SET broker_order_id = :order_id, "
                    "state = 'submitted', updated_at = now() WHERE client_order_id = :id"
                ),
                {"id": client_order_id, "order_id": order_id},
            )


def settle_order_intent(
    client_order_id: str,
    order_id: str,
    filled_qty: float,
    filled_avg_price: float | None,
    terminal_status: str,
) -> None:
    """Record validated terminal execution and resolve its intent atomically."""
    if (
        filled_qty is None
        or not math.isfinite(filled_qty)
        or filled_qty < 0
        or terminal_status not in _TERMINAL_STATUSES
        or (terminal_status == "filled" and filled_qty == 0)
        or (
            filled_qty > 0
            and (
                filled_avg_price is None
                or not math.isfinite(filled_avg_price)
                or filled_avg_price <= 0
            )
        )
        or (filled_avg_price is not None and not math.isfinite(filled_avg_price))
    ):
        raise ValueError("Invalid terminal fill")
    amount = filled_qty * filled_avg_price if filled_qty > 0 else 0
    if not math.isfinite(amount):
        raise ValueError("Invalid fill value")
    with _get_engine().begin() as conn:
        _lock_order_and_reject_pending(conn, order_id)
        row = (
            conn.execute(
                text("SELECT * FROM paper_order_intents WHERE client_order_id = :id FOR UPDATE"),
                {"id": client_order_id},
            )
            .mappings()
            .first()
        )
        if row is None or row["broker_order_id"] not in (None, order_id):
            raise ValueError("Missing intent or conflicting broker order")
        if row["state"] == "resolved":
            if (
                row["broker_order_id"],
                row["filled_qty"],
                row["filled_avg_price"],
                row["terminal_status"],
            ) != (order_id, filled_qty, filled_avg_price, terminal_status):
                raise ValueError("Conflicting settlement")
            return
        trades = conn.execute(
            text(
                "SELECT run_date, side, symbol, amount, reason FROM paper_trades "
                "WHERE order_id = :id FOR UPDATE"
            ),
            {"id": order_id},
        ).all()
        expected = (row["run_date"], row["side"], row["symbol"], amount, row["reason"])
        if len(trades) > 1 or (trades and (filled_qty == 0 or tuple(trades[0]) != expected)):
            raise ValueError("Conflicting historical trade")
        if filled_qty > 0 and not trades:
            conn.execute(
                text(
                    "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id, reason) "
                    "VALUES (:run_date, :side, :symbol, :amount, :order_id, :reason)"
                ),
                {
                    "run_date": row["run_date"],
                    "side": row["side"],
                    "symbol": row["symbol"],
                    "amount": amount,
                    "order_id": order_id,
                    "reason": row["reason"],
                },
            )
        conn.execute(
            text(
                "UPDATE paper_order_intents SET broker_order_id = :order_id, state = 'resolved', "
                "terminal_status = :status, filled_qty = :qty, filled_avg_price = :price, "
                "updated_at = now() WHERE client_order_id = :id"
            ),
            {
                "id": client_order_id,
                "order_id": order_id,
                "status": terminal_status,
                "qty": filled_qty,
                "price": filled_avg_price,
            },
        )


class PaperRunLock:
    """A session lock whose ownership can be checked before submission."""

    def __init__(self, conn):
        self._conn = conn

    def verify(self) -> None:
        """Fail if the dedicated session no longer owns its advisory lock."""
        held = self._conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE "
                "locktype = 'advisory' AND pid = pg_backend_pid() AND classid = 0 "
                "AND objid = :key AND objsubid = 1 AND granted)"
            ),
            {"key": _LOCK_KEY},
        ).scalar_one()
        if not held:
            raise RuntimeError("Paper run lock lost")


@contextmanager
def paper_run_lock() -> Iterator[PaperRunLock]:
    """Hold an exclusive advisory lock on a dedicated database session."""
    with _get_engine().connect() as conn:
        acquired = conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": _LOCK_KEY}
        ).scalar_one()
        if not acquired:
            raise RuntimeError("Paper run already locked")
        try:
            lock = PaperRunLock(conn)
            lock.verify()
            yield lock
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _LOCK_KEY})


def log_trade(
    run_date: str,
    side: str,
    symbol: str,
    amount: float,
    order_id: str,
    reason: str | None = None,
) -> None:
    """`reason` tags why the trade happened (e.g. "cash_sweep" for a sweep
    order -- see `cash_sweep.SWEEP_TRADE_REASON`); NULL/omitted means an
    ordinary strategy trade, preserving the historical meaning of this
    ledger for every caller that doesn't pass it."""
    with _get_engine().connect() as conn:
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:id, 0))"), {"id": order_id}
        )
        trades = conn.execute(
            text(
                "SELECT run_date, side, symbol, amount, reason FROM paper_trades "
                "WHERE order_id = :id FOR UPDATE"
            ),
            {"id": order_id},
        ).all()
        expected = (date.fromisoformat(run_date), side, symbol, amount, reason)
        if len(trades) > 1 or (trades and tuple(trades[0]) != expected):
            raise ValueError("Conflicting historical trade")
        if trades:
            return
        conn.execute(
            text(
                "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id, reason) "
                "VALUES (:run_date, :side, :symbol, :amount, :order_id, :reason)"
            ),
            {
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "amount": amount,
                "order_id": order_id,
                "reason": reason,
            },
        )
        conn.commit()


def settle_legacy_pending_order(
    order_id: str, filled_qty: float, filled_avg_price: float | None, terminal_status: str
) -> None:
    """Atomically book a legacy execution and remove its pending row."""
    if (
        filled_qty is None
        or not math.isfinite(filled_qty)
        or filled_qty < 0
        or terminal_status not in _TERMINAL_STATUSES
        or (terminal_status == "filled" and filled_qty == 0)
        or (filled_avg_price is not None and not math.isfinite(filled_avg_price))
        or (filled_qty > 0 and (filled_avg_price is None or filled_avg_price <= 0))
    ):
        raise ValueError("Invalid terminal fill")
    amount = filled_qty * filled_avg_price if filled_qty > 0 else 0
    if not math.isfinite(amount):
        raise ValueError("Invalid fill value")
    with _get_engine().begin() as conn:
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:id, 0))"), {"id": order_id}
        )
        row = (
            conn.execute(
                text("SELECT * FROM paper_pending_orders WHERE order_id = :id FOR UPDATE"),
                {"id": order_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            raise ValueError("Missing legacy pending order")
        intent = conn.execute(
            text("SELECT 1 FROM paper_order_intents WHERE broker_order_id = :id LIMIT 1"),
            {"id": order_id},
        ).first()
        if intent is not None:
            raise ValueError("Order has both pending and intent representations")
        trades = conn.execute(
            text(
                "SELECT run_date, side, symbol, amount, reason FROM paper_trades "
                "WHERE order_id = :id FOR UPDATE"
            ),
            {"id": order_id},
        ).all()
        expected = (row["run_date"], row["side"], row["symbol"], amount, row["reason"])
        if len(trades) > 1 or (trades and (filled_qty == 0 or tuple(trades[0]) != expected)):
            raise ValueError("Conflicting historical trade")
        if filled_qty > 0 and not trades:
            conn.execute(
                text(
                    "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id, reason) "
                    "VALUES (:run_date, :side, :symbol, :amount, :order_id, :reason)"
                ),
                {
                    "run_date": row["run_date"],
                    "side": row["side"],
                    "symbol": row["symbol"],
                    "amount": amount,
                    "order_id": order_id,
                    "reason": row["reason"],
                },
            )
        conn.execute(
            text("DELETE FROM paper_pending_orders WHERE order_id = :id"), {"id": order_id}
        )


def log_pending_order(
    run_date: str,
    side: str,
    symbol: str,
    notional: float,
    order_id: str,
    reason: str | None = None,
) -> None:
    """Record an order that was submitted but had not filled by run end, so the
    next run can reconcile its final status. Idempotent on order_id.

    `reason` mirrors `log_trade`'s tag (e.g. "cash_sweep"), NULL for an
    ordinary strategy order.
    """
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_pending_orders "
                "(order_id, run_date, side, symbol, notional, reason) "
                "VALUES (:order_id, :run_date, :side, :symbol, :notional, :reason) "
                "ON CONFLICT (order_id) DO NOTHING"
            ),
            {
                "order_id": order_id,
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "notional": notional,
                "reason": reason,
            },
        )
        conn.commit()


def get_pending_orders() -> list[dict]:
    """Return all unresolved pending orders awaiting reconciliation."""
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT order_id, run_date, side, symbol, notional, created_at, "
                "flagged_stale, reason "
                "FROM paper_pending_orders ORDER BY created_at"
            )
        ).mappings()
        return [dict(r) for r in rows]


def clear_pending_order(order_id: str) -> None:
    """Remove a pending order once it has reached a terminal status."""
    with _get_engine().connect() as conn:
        conn.execute(
            text("DELETE FROM paper_pending_orders WHERE order_id = :order_id"),
            {"order_id": order_id},
        )
        conn.commit()


def mark_pending_order_stale(order_id: str) -> None:
    """Flag a pending order as having already triggered a staleness alert.

    The row is left in place (reconciliation keeps polling it) -- this only
    stops the notifier from re-alerting on it every run."""
    with _get_engine().connect() as conn:
        conn.execute(
            text("UPDATE paper_pending_orders SET flagged_stale = TRUE WHERE order_id = :order_id"),
            {"order_id": order_id},
        )
        conn.commit()


def get_latest_snapshot(*, before: date | None = None) -> float | None:
    """Return the latest portfolio value, optionally strictly before a date."""
    with _get_engine().connect() as conn:
        if before is None:
            row = conn.execute(
                text("SELECT portfolio_value FROM paper_snapshots ORDER BY run_date DESC LIMIT 1")
            ).first()
        else:
            row = conn.execute(
                text(
                    "SELECT portfolio_value FROM paper_snapshots "
                    "WHERE run_date < :before ORDER BY run_date DESC LIMIT 1"
                ),
                {"before": before},
            ).first()
    return float(row[0]) if row else None


def get_earliest_snapshot() -> float | None:
    """Return the oldest snapshot's portfolio_value (starting capital), or None."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT portfolio_value FROM paper_snapshots ORDER BY run_date ASC LIMIT 1")
        ).first()
    return float(row[0]) if row else None


def log_snapshot(run_date: str, portfolio_value: float, cash: float, positions: dict) -> None:
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_snapshots (run_date, portfolio_value, cash, positions) "
                "VALUES (:run_date, :pv, :cash, :pos) "
                "ON CONFLICT (run_date) DO UPDATE SET "
                "portfolio_value = EXCLUDED.portfolio_value, "
                "cash = EXCLUDED.cash, "
                "positions = EXCLUDED.positions"
            ),
            {
                "run_date": run_date,
                "pv": portfolio_value,
                "cash": cash,
                "pos": json.dumps(json_safe(positions), allow_nan=False),
            },
        )
        conn.commit()


def get_latest_snapshot_positions() -> dict | None:
    """Return the most recent snapshot's positions dict, or None if none exist.

    Used to detect broker-side data gaps (e.g. a stock split whose qty
    adjustment the paper broker never applied) by diffing against the
    positions returned by the live broker on the next run.
    """
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT positions FROM paper_snapshots ORDER BY run_date DESC LIMIT 1")
        ).first()
    if row is None:
        return None
    positions = row[0]
    return positions if isinstance(positions, dict) else json.loads(positions)


def get_latest_snapshot_run_date() -> str | None:
    """Return the most recent snapshot's run_date (ISO string), or None if none exist.

    Paired with `get_latest_snapshot_positions()` so the split-check gate can
    tell whether a known split's ex-date falls between the prior snapshot and
    today (see `split_check.find_unadjusted_split_symbols`).
    """
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT run_date FROM paper_snapshots ORDER BY run_date DESC LIMIT 1")
        ).first()
    return str(row[0]) if row else None


def get_peak_value() -> float | None:
    """Return the persisted portfolio peak (for drawdown tracking), or None."""
    with _get_engine().connect() as conn:
        row = conn.execute(text("SELECT peak_value FROM paper_risk_state WHERE id = 1")).first()
    return float(row[0]) if row else None


def save_peak_value(peak_value: float) -> None:
    """Upsert the single current peak-value row (id=1)."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_risk_state (id, peak_value) VALUES (1, :peak) "
                "ON CONFLICT (id) DO UPDATE SET "
                "peak_value = EXCLUDED.peak_value, updated_at = now()"
            ),
            {"peak": peak_value},
        )
        conn.commit()


def get_rebalance_state() -> dict | None:
    """Return the current sleeve weights/scale, or None if never set."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT rebalance_date, weights, scale FROM paper_rebalance_state WHERE id = 1")
        ).first()
    if row is None:
        return None
    rebalance_date, weights, scale = row
    return {
        "rebalance_date": str(rebalance_date),
        "weights": weights if isinstance(weights, dict) else json.loads(weights),
        "scale": float(scale),
    }


def get_snapshot_history() -> list[tuple]:
    """Return all persisted snapshots as `[(run_date, positions_dict), ...]`
    ascending by `run_date`.

    Used by dividend accrual (see `dividend_check.qty_held_at_ex_date`) to
    establish, from what's already persisted, whether -- and how much of --
    a symbol was held as of a given ex-dividend date. `run_date` comes back
    as a real `datetime.date` (the column type), not a string.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT run_date, positions FROM paper_snapshots ORDER BY run_date ASC")
        ).all()
    result = []
    for run_date, positions in rows:
        pos = positions if isinstance(positions, dict) else json.loads(positions)
        result.append((run_date, pos))
    return result


def get_trade_history_dates(symbol: str) -> list:
    """Return every persisted `paper_trades.run_date` for `symbol`, ascending.

    Used by `split_check.find_split_applied_symbols` to rule out a plain
    buy/sell explaining a qty change across a split's ex-date, so a trim or
    exit is never mistaken for the broker silently applying a split. Dates
    come back as real `datetime.date` objects (the column type), matching
    `get_snapshot_history`'s convention.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT run_date FROM paper_trades WHERE symbol = :symbol ORDER BY run_date ASC"),
            {"symbol": symbol},
        ).all()
    return [run_date for (run_date,) in rows]


def get_last_buy_dates() -> dict[str, date]:
    """Return `{symbol: most recent BUY run_date}` across `paper_trades`.

    The trader treats this as each held position's entry date when catching
    up exit signals that fired on a bar after entry but not on today's bar.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, max(run_date) FROM paper_trades WHERE side = 'BUY' GROUP BY symbol"
            )
        ).all()
    return {symbol: run_date for symbol, run_date in rows}


def get_accrued_dividend_keys() -> set[tuple]:
    """Return `{(symbol, ex_date), ...}` already recorded in
    `paper_dividend_accruals` -- the idempotency guard so a re-run never
    double-credits the same dividend event."""
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT symbol, ex_date FROM paper_dividend_accruals")).all()
    return {(symbol, ex_date) for symbol, ex_date in rows}


def log_dividend_accrual(symbol: str, ex_date, rate: float, qty: float, amount: float) -> None:
    """Persist one dividend accrual. Idempotent on `(symbol, ex_date)` --
    a conflicting insert is a no-op, never a double-credit."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_dividend_accruals (symbol, ex_date, rate, qty, amount) "
                "VALUES (:symbol, :ex_date, :rate, :qty, :amount) "
                "ON CONFLICT (symbol, ex_date) DO NOTHING"
            ),
            {
                "symbol": symbol,
                "ex_date": ex_date,
                "rate": rate,
                "qty": qty,
                "amount": amount,
            },
        )
        conn.commit()


def get_total_dividend_accrual() -> float:
    """Cumulative dollar total of all persisted dividend accruals -- a
    reporting-only correction, not real cash (see `dividend_check.py`)."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT COALESCE(SUM(amount), 0) FROM paper_dividend_accruals")
        ).first()
    return float(row[0]) if row else 0.0


def save_rebalance_state(rebalance_date: str, weights: dict[str, float], scale: float) -> None:
    """Upsert the single current rebalance-state row (id=1)."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_rebalance_state (id, rebalance_date, weights, scale) "
                "VALUES (1, :rd, :w, :s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "rebalance_date = EXCLUDED.rebalance_date, "
                "weights = EXCLUDED.weights, "
                "scale = EXCLUDED.scale"
            ),
            {"rd": rebalance_date, "w": json.dumps(weights), "s": scale},
        )
        conn.commit()


def get_open_split_corrections() -> dict[str, float]:
    """Return `{symbol: factor}` for every split this account has detected
    as broker-unapplied and not yet confirmed applied.

    Persists past `trader._SPLIT_LOOKBACK_DAYS` (the rolling window used to
    detect *new* splits from the broker's corporate-actions feed) so a
    long-held unapplied split keeps being corrected instead of the
    correction silently expiring while the position is still held -- the
    MNST incident, unapplied since 2026-08-11, correction expired
    2026-08-25 while still held. See `split_check.py`'s module docstring.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT symbol, factor FROM paper_split_state")).all()
    return {symbol: float(factor) for symbol, factor in rows}


def get_open_split_states() -> dict[str, dict]:
    """Return `{symbol: {"factor": float, "ex_date": date}}` for every
    persisted split-correction row -- a richer sibling of
    `get_open_split_corrections()` that also surfaces the stored `ex_date`.

    `get_open_split_corrections()` is kept unchanged (its existing
    factor-only contract is already relied on); this getter exists so a
    persisted correction can be re-verified against snapshot evidence using
    its *own* ex_date even once the broker's live corporate-actions feed
    stops mentioning it (the feed only reports events inside its own
    rolling lookback window -- see `trader._compute_split_corrections`).
    `ex_date` comes back as a real `datetime.date` (the column type), same
    convention as `get_snapshot_history`.
    """
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT symbol, factor, ex_date FROM paper_split_state")).all()
    return {
        symbol: {"factor": float(factor), "ex_date": ex_date} for symbol, factor, ex_date in rows
    }


def save_split_correction(symbol: str, ex_date: str, factor: float) -> None:
    """Upsert a detected-unapplied split for `symbol`. Idempotent -- calling
    again for the same symbol just refreshes `factor`/`updated_at`."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_split_state (symbol, ex_date, factor) "
                "VALUES (:symbol, :ex_date, :factor) "
                "ON CONFLICT (symbol) DO UPDATE SET "
                "factor = EXCLUDED.factor, updated_at = now()"
            ),
            {"symbol": symbol, "ex_date": ex_date, "factor": factor},
        )
        conn.commit()


def clear_split_correction(symbol: str) -> None:
    """Remove a symbol's persisted split-correction state. Call once the
    broker's own snapshot history confirms the split was actually applied
    (qty jumped by ~factor) or the position is fully closed -- see
    `trader._compute_split_corrections`."""
    with _get_engine().connect() as conn:
        conn.execute(
            text("DELETE FROM paper_split_state WHERE symbol = :symbol"),
            {"symbol": symbol},
        )
        conn.commit()
