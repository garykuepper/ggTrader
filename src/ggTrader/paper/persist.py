"""TimescaleDB persistence for paper trading trades and snapshots."""

from __future__ import annotations

import json

from sqlalchemy import text

from ggTrader.lab.persist import get_engine as _get_engine

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    side TEXT NOT NULL,
    symbol TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    order_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
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
    flagged_stale BOOLEAN NOT NULL DEFAULT FALSE
);
ALTER TABLE paper_pending_orders
    ADD COLUMN IF NOT EXISTS flagged_stale BOOLEAN NOT NULL DEFAULT FALSE;
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
"""


def init_paper_schema() -> None:
    with _get_engine().connect() as conn:
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()


def log_trade(run_date: str, side: str, symbol: str, amount: float, order_id: str) -> None:
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id) "
                "VALUES (:run_date, :side, :symbol, :amount, :order_id)"
            ),
            {
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "amount": amount,
                "order_id": order_id,
            },
        )
        conn.commit()


def log_pending_order(
    run_date: str, side: str, symbol: str, notional: float, order_id: str
) -> None:
    """Record an order that was submitted but had not filled by run end, so the
    next run can reconcile its final status. Idempotent on order_id."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_pending_orders "
                "(order_id, run_date, side, symbol, notional) "
                "VALUES (:order_id, :run_date, :side, :symbol, :notional) "
                "ON CONFLICT (order_id) DO NOTHING"
            ),
            {
                "order_id": order_id,
                "run_date": run_date,
                "side": side,
                "symbol": symbol,
                "notional": notional,
            },
        )
        conn.commit()


def get_pending_orders() -> list[dict]:
    """Return all unresolved pending orders awaiting reconciliation."""
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT order_id, run_date, side, symbol, notional, created_at, flagged_stale "
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


def get_latest_snapshot() -> float | None:
    """Return the most recent snapshot's portfolio_value, or None if no snapshots exist."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT portfolio_value FROM paper_snapshots ORDER BY run_date DESC LIMIT 1")
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
                "pos": json.dumps(positions),
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
