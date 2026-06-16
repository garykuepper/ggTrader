"""TimescaleDB persistence for lab runs — the only store for research state."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string

_ENGINE = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lab_runs (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    market TEXT NOT NULL,
    freq TEXT NOT NULL,
    eval_start TEXT NOT NULL,
    eval_end TEXT NOT NULL,
    params JSONB,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lab_plans (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    asof TIMESTAMPTZ NOT NULL,
    plan JSONB NOT NULL,
    eligible_count INT,
    coverage JSONB,
    PRIMARY KEY (run_id, strategy, asof)
);
CREATE TABLE IF NOT EXISTS lab_returns (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    ret DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, strategy, date)
);
CREATE TABLE IF NOT EXISTS lab_equity (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    strategy_equity DOUBLE PRECISION NOT NULL,
    benchmark_equity DOUBLE PRECISION,
    PRIMARY KEY (run_id, strategy, date)
);
CREATE TABLE IF NOT EXISTS lab_summary (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    metrics JSONB,
    benchmark_metrics JSONB,
    diagnostics JSONB,
    PRIMARY KEY (run_id, strategy)
);
"""


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_db_connection_string())
    return _ENGINE


def init_schema() -> None:
    eng = get_engine()
    with eng.begin() as conn:
        for stmt in [s for s in _SCHEMA.split(";") if s.strip()]:
            conn.execute(text(stmt))
        for tbl in ("lab_returns", "lab_equity"):
            try:
                conn.execute(
                    text(
                        f"SELECT create_hypertable('{tbl}', 'date', "
                        "if_not_exists => TRUE, migrate_data => TRUE)"
                    )
                )
            except Exception:
                pass  # plain Postgres (no TimescaleDB extension) — table still works


def start_run(
    strategy: str, market: str, freq: str, eval_start: str, eval_end: str, params: Dict[str, Any]
) -> str:
    run_id = f"{strategy}_{uuid.uuid4().hex[:8]}"
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO lab_runs"
                " (run_id, strategy, market, freq, eval_start, eval_end, params) "
                "VALUES (:r, :s, :m, :f, :es, :ee, :p)"
            ),
            {
                "r": run_id,
                "s": strategy,
                "m": market,
                "f": freq,
                "es": eval_start,
                "ee": eval_end,
                "p": json.dumps(params),
            },
        )
    return run_id


def finish_run(run_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(text("UPDATE lab_runs SET status='done' WHERE run_id=:r"), {"r": run_id})


def plan_done(run_id: str, strategy: str, asof: pd.Timestamp) -> bool:
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM lab_plans WHERE run_id=:r AND strategy=:s AND asof=:a"),
            {"r": run_id, "s": strategy, "a": asof.to_pydatetime()},
        ).first()
    return row is not None


def write_plan(
    run_id: str,
    strategy: str,
    asof: pd.Timestamp,
    plan: List[Dict[str, Any]],
    eligible_count: int,
    coverage: Dict[str, Any],
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO lab_plans (run_id, strategy, asof, plan, eligible_count, coverage) "
                "VALUES (:r, :s, :a, :p, :ec, :c) "
                "ON CONFLICT (run_id, strategy, asof) DO UPDATE SET plan=EXCLUDED.plan"
            ),
            {
                "r": run_id,
                "s": strategy,
                "a": asof.to_pydatetime(),
                "p": json.dumps(plan),
                "ec": eligible_count,
                "c": json.dumps(coverage),
            },
        )


def read_plan(run_id: str, strategy: str, asof: pd.Timestamp) -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT plan FROM lab_plans WHERE run_id=:r AND strategy=:s AND asof=:a"),
            {"r": run_id, "s": strategy, "a": asof.to_pydatetime()},
        ).first()
    if row is None:
        return []
    return row[0] if isinstance(row[0], list) else json.loads(row[0])


def read_all_plans(run_id: str, strategy: str) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT asof, plan FROM lab_plans WHERE run_id=:r AND strategy=:s ORDER BY asof"),
            {"r": run_id, "s": strategy},
        ).fetchall()
    out: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for asof, plan in rows:
        out[pd.Timestamp(asof)] = plan if isinstance(plan, list) else json.loads(plan)
    return out


def write_returns_equity(
    run_id: str,
    strategy: str,
    rets: pd.Series,
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
) -> None:
    bench = benchmark_equity.reindex(equity.index) if benchmark_equity is not None else None
    with get_engine().begin() as conn:
        for dt, r in rets.items():
            conn.execute(
                text(
                    "INSERT INTO lab_returns (run_id, strategy, date, ret) VALUES (:r,:s,:d,:v) "
                    "ON CONFLICT (run_id, strategy, date) DO UPDATE SET ret=EXCLUDED.ret"
                ),
                {"r": run_id, "s": strategy, "d": pd.Timestamp(dt).to_pydatetime(), "v": float(r)},
            )
        for dt, e in equity.items():
            be = (
                None
                if bench is None
                else (None if pd.isna(bench.loc[dt]) else float(bench.loc[dt]))
            )
            conn.execute(
                text(
                    "INSERT INTO lab_equity"
                    " (run_id, strategy, date, strategy_equity, benchmark_equity) "
                    "VALUES (:r,:s,:d,:e,:b) ON CONFLICT (run_id, strategy, date) DO UPDATE "
                    "SET strategy_equity=EXCLUDED.strategy_equity,"
                    " benchmark_equity=EXCLUDED.benchmark_equity"
                ),
                {
                    "r": run_id,
                    "s": strategy,
                    "d": pd.Timestamp(dt).to_pydatetime(),
                    "e": float(e),
                    "b": be,
                },
            )


def write_summary(
    run_id: str,
    strategy: str,
    metrics: Dict[str, Any],
    benchmark_metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO lab_summary"
                " (run_id, strategy, metrics, benchmark_metrics, diagnostics) "
                "VALUES (:r,:s,:m,:b,:d) ON CONFLICT (run_id, strategy) DO UPDATE "
                "SET metrics=EXCLUDED.metrics, benchmark_metrics=EXCLUDED.benchmark_metrics, "
                "diagnostics=EXCLUDED.diagnostics"
            ),
            {
                "r": run_id,
                "s": strategy,
                "m": json.dumps(metrics),
                "b": json.dumps(benchmark_metrics),
                "d": json.dumps(diagnostics),
            },
        )
