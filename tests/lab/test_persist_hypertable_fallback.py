"""Check the plain PostgreSQL fallback is recorded."""

import logging
from contextlib import contextmanager

from ggTrader.lab import persist


def test_plain_postgres_fallback_is_logged_at_debug(monkeypatch, caplog) -> None:
    statements = []

    class Connection:
        def execute(self, statement):
            statements.append(str(statement))
            if "create_hypertable" in str(statement):
                raise RuntimeError("no extension")

    class Engine:
        @contextmanager
        def begin(self):
            yield Connection()

    monkeypatch.setattr(persist, "get_engine", lambda: Engine())
    with caplog.at_level(logging.DEBUG, logger="ggTrader.lab.persist"):
        persist.init_schema()
    assert len(statements) == 5
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)
