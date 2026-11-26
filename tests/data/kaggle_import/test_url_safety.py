"""Connection targets cannot be redirected by PostgreSQL URL options."""

import argparse

import pytest

from ggTrader.data.kaggle_import.guards import assert_connected_database, checked_pg_url


@pytest.mark.parametrize(
    "key",
    [
        "dbname",
        "database",
        "host",
        "hostaddr",
        "port",
        "user",
        "service",
        "options",
        "target_session_attrs",
    ],
)
def test_redirecting_options_are_refused_without_echoing_url(key):
    secret = f"postgresql://user:secret@host/scratch?{key}=ggtrader"
    with pytest.raises(ValueError) as exc:
        checked_pg_url(secret)
    assert "secret" not in str(exc.value) and "postgresql://" not in str(exc.value)


def test_rejects_fragments_and_other_schemes():
    for url in ("postgresql://host/db#fragment", "mysql://host/db", "postgresql://host/"):
        with pytest.raises(ValueError):
            checked_pg_url(url)
    assert checked_pg_url("postgresql://host/db?sslmode=require", expect_db="db")


def test_connected_database_mismatch_is_non_disclosing():
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            assert sql.lower() == "select current_database()"

        def fetchone(self):
            return ("ggtrader",)

    class Connection:
        def cursor(self):
            return Cursor()

    with pytest.raises(ValueError, match="database"):
        assert_connected_database(Connection(), "scratch")


def test_pilot_refuses_override_before_connect(monkeypatch):
    from scripts import kaggle_pilot

    calls = []
    monkeypatch.setattr(kaggle_pilot.psycopg2, "connect", lambda *a, **k: calls.append(a))
    args = argparse.Namespace(scratch_url="postgresql://localhost/postgres?dbname=ggtrader")
    with pytest.raises(ValueError):
        kaggle_pilot.run(args)
    assert not calls


def test_import_mismatch_prevents_write(monkeypatch):
    import ggTrader.data.kaggle_import.ddl as ddl
    import ggTrader.data.universe_db as universe
    from ggTrader.cli import cmd_db

    writes = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            assert sql.lower() == "select current_database()"

        def fetchone(self):
            return ("ggtrader",)

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    monkeypatch.setattr(universe, "connect", lambda *a, **k: Connection())
    monkeypatch.setattr(ddl, "apply_ddl", lambda conn: writes.append(True))
    args = argparse.Namespace(
        db_command="import",
        import_command="apply-ddl",
        no_backup_check=True,
        backup_dir=None,
        yes=True,
        use_configured_db=False,
        db_url_env="FAKE_TEST_URL",
    )
    monkeypatch.setenv("FAKE_TEST_URL", "postgresql://localhost/scratch")
    with pytest.raises(ValueError, match="database"):
        cmd_db.run_db(args)
    assert not writes


def test_pilot_connected_mismatch_precedes_target_writes(monkeypatch):
    from scripts import kaggle_pilot

    target_statements = []

    class Cursor:
        def __init__(self, target):
            self.target = target

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, statement):
            if self.target:
                target_statements.append(statement)

        def fetchone(self):
            return ("ggtrader",)

    class Connection:
        def __init__(self, target):
            self.target = target

        def cursor(self):
            return Cursor(self.target)

        def close(self):
            pass

    calls = []

    def connect(*args):
        calls.append(args)
        return Connection(len(calls) == 2)

    monkeypatch.setattr(kaggle_pilot.psycopg2, "connect", connect)
    monkeypatch.setattr(kaggle_pilot, "rows_by_year", lambda path: {2000: 1})
    monkeypatch.setattr(kaggle_pilot, "apply_ddl", lambda conn: pytest.fail("must not write"))
    args = argparse.Namespace(
        scratch_url="postgresql://localhost/postgres", file="unused", year=2000
    )
    with pytest.raises(ValueError, match="database"):
        kaggle_pilot.run(args)
    assert target_statements == ["select current_database()"]


def test_import_refuses_url_override_before_connect(monkeypatch):
    import ggTrader.data.universe_db as universe
    from ggTrader.cli import cmd_db

    calls = []
    monkeypatch.setattr(universe, "connect", lambda *a, **k: calls.append(a))
    monkeypatch.setenv(
        "FAKE_TEST_URL", "postgresql://alice:secret@localhost/scratch?dbname=ggtrader"
    )
    args = argparse.Namespace(
        db_command="import",
        import_command="apply-ddl",
        no_backup_check=True,
        backup_dir=None,
        yes=True,
        use_configured_db=False,
        db_url_env="FAKE_TEST_URL",
    )
    with pytest.raises(ValueError) as exc:
        cmd_db.run_db(args)
    assert not calls and "secret" not in str(exc.value)
