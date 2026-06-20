"""TimescaleDB persistence for paper trading trades and snapshots."""

from __future__ import annotations

import json

from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_db_connection_string())
    return _ENGINE


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
