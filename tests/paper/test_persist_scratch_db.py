"""The scratch-database fixture's safety rules, tested without any database."""

from __future__ import annotations

import pytest

from tests.paper import persist_scratch_db as fx

BAD_NAMES = [
    "ggtrader",
    "ohlcv",
    "postgres",
    "template1",
    "",
    "ggt_scratch_",
    "ggt_scratch_1234567",  # 7 hex
    "ggt_scratch_123456789",  # 9 hex
    "ggt_scratch_1234567G",  # not hex
    "GGT_SCRATCH_12345678",  # upper case prefix
    "ggt_scratch_ABCDEF12",  # upper case hex
    "ggt_test_12345678",
    'ggt_scratch_12345678"; DROP DATABASE ggtrader; --',
    "ggt_scratch_12345678\n",  # a trailing newline must not slip past `$`
    "x ggt_scratch_12345678",
]


class FakeCursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: str) -> None:
        self._log.append(statement)


class FakeConn:
    def __init__(self, log: list[str], dsns: list[str]) -> None:
        self._log = log
        self.autocommit = False
        self.closed = False
        dsns.append("opened")

    def cursor(self) -> FakeCursor:
        assert self.autocommit is True, "the administrator connection must be AUTOCOMMIT"
        return FakeCursor(self._log)

    def close(self) -> None:
        self.closed = True


def _fake_connect() -> tuple[list[str], list[str], object]:
    log: list[str] = []
    opened: list[str] = []

    def connect(dsn: str) -> FakeConn:
        assert dsn.startswith("postgresql://")
        return FakeConn(log, opened)

    return log, opened, connect


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_is_refused(name: str) -> None:
    with pytest.raises(fx.ScratchNameError):
        fx.assert_scratch_name(name)
    with pytest.raises(fx.ScratchNameError):
        fx.create_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.drop_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.scratch_engine_url("postgresql://u:p@h/db", name)


def test_non_string_name_is_refused() -> None:
    for name in (None, 12345678, b"ggt_scratch_12345678"):
        with pytest.raises(fx.ScratchNameError):
            fx.assert_scratch_name(name)  # type: ignore[arg-type]


def test_generated_names_match_and_differ() -> None:
    names = {fx.new_scratch_name() for _ in range(50)}
    assert len(names) == 50
    assert all(fx.SCRATCH_NAME.fullmatch(n) for n in names)


def test_statements_are_exactly_create_and_drop_force() -> None:
    assert fx.create_sql("ggt_scratch_0123abcd") == 'CREATE DATABASE "ggt_scratch_0123abcd"'
    assert fx.drop_sql("ggt_scratch_0123abcd") == (
        'DROP DATABASE "ggt_scratch_0123abcd" WITH (FORCE)'
    )


def test_bad_name_raises_before_any_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    log, opened, connect = _fake_connect()
    monkeypatch.setattr(fx, "new_scratch_name", lambda: "ohlcv")
    with pytest.raises(fx.ScratchNameError):
        with fx.scratch_database_name("postgresql://u:p@h/db", connect):
            pytest.fail("the body must not run")
    assert log == [] and opened == []


def test_create_then_drop_with_force_and_nothing_else() -> None:
    log, opened, connect = _fake_connect()
    with fx.scratch_database_name("postgresql+psycopg2://u:p@h/db", connect) as name:
        assert fx.SCRATCH_NAME.fullmatch(name)
        assert log == [f'CREATE DATABASE "{name}"']
    assert log == [f'CREATE DATABASE "{name}"', f'DROP DATABASE "{name}" WITH (FORCE)']
    assert len(opened) == 2


def test_drop_runs_when_the_body_raises() -> None:
    log, _, connect = _fake_connect()
    with pytest.raises(RuntimeError, match="boom"):
        with fx.scratch_database_name("postgresql://u:p@h/db", connect) as name:
            raise RuntimeError("boom")
    assert log[-1] == f'DROP DATABASE "{name}" WITH (FORCE)'


def test_scratch_url_replaces_only_the_database() -> None:
    url = fx.scratch_engine_url(
        "postgresql+psycopg2://user:pw@dbhost:5433/adminDb", "ggt_scratch_0123abcd"
    )
    assert (url.username, url.host, url.port) == ("user", "dbhost", 5433)
    assert url.database == "ggt_scratch_0123abcd"
    assert url.password == "pw"
    assert "pw" not in repr(url)  # SQLAlchemy hides the password in a repr


def test_scratch_db_skips_without_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fx.ADMIN_URL_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        with fx.scratch_db():
            pytest.fail("the body must not run")
