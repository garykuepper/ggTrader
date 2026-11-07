"""Unit tests (no database): the provenance column is additive and migration-safe."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from ggTrader.lab import persist


class FakeConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params))


class FakeEngine:
    def __init__(self) -> None:
        self.conn = FakeConn()

    @contextmanager
    def begin(self):
        yield self.conn


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    engine = FakeEngine()
    monkeypatch.setattr(persist, "get_engine", lambda: engine)
    return engine


def _sql(engine: FakeEngine) -> list[str]:
    return [" ".join(s.split()) for s, _ in engine.conn.calls]


def test_init_schema_adds_the_columns_with_if_not_exists(fake: FakeEngine) -> None:
    persist.init_schema()
    sql = _sql(fake)
    assert "ALTER TABLE lab_runs ADD COLUMN IF NOT EXISTS provenance JSONB" in sql
    assert "ALTER TABLE lab_sweeps ADD COLUMN IF NOT EXISTS provenance JSONB" in sql


def test_migrations_are_idempotent_and_never_destructive() -> None:
    for statement in persist.PROVENANCE_MIGRATIONS:
        upper = statement.upper()
        assert "IF NOT EXISTS" in upper
        assert not any(word in upper for word in ("DROP", "DELETE", "TRUNCATE", "NOT NULL"))
    assert "provenance" not in persist._SCHEMA  # CREATE TABLE text is unchanged


def test_start_run_without_provenance_uses_the_old_insert(fake: FakeEngine) -> None:
    """A database that never ran the migration must keep working."""
    persist.start_run("s", "equity", "1D", "2021-01-31", "2022-01-31", {"a": 1})
    (sql,) = _sql(fake)
    assert "provenance" not in sql


def test_start_run_with_provenance_stores_json(fake: FakeEngine) -> None:
    block = {"vectorbt": "1.2.0", "baseline_epoch": "E1"}
    persist.start_run("s", "equity", "1D", "2021-01-31", "2022-01-31", {}, provenance=block)
    (sql,) = _sql(fake)
    params = fake.conn.calls[0][1]
    assert "provenance" in sql
    assert json.loads(params["pv"]) == block


def test_start_sweep_provenance_is_optional_and_stored(fake: FakeEngine) -> None:
    persist.start_sweep("s", "equity", {"x": [1]}, 1)
    assert "provenance" not in _sql(fake)[0]
    persist.start_sweep("s", "equity", {"x": [1]}, 1, provenance={"baseline_epoch": "E1"})
    assert "provenance" in _sql(fake)[1]
    assert json.loads(fake.conn.calls[1][1]["pv"]) == {"baseline_epoch": "E1"}
