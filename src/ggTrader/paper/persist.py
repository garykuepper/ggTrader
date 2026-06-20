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


def get_latest_snapshot() -> float | None:
    """Return the most recent snapshot's portfolio_value, or None if no snapshots exist."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT portfolio_value FROM paper_snapshots ORDER BY run_date DESC LIMIT 1")
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
