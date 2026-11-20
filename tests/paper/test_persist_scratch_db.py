"""The scratch-database fixture's safety rules, tested without any database."""

from __future__ import annotations

import psycopg2
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from tests.paper import persist_scratch_db as fx

MARK = "PWMARK9f3a"  # a password-like marker injected through fake URLs; it must never surface

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

# Administrator URLs that must be refused before any connection is opened.
OVERRIDE_URLS = [
    "postgresql://u:p@h/db?dbname=ggtrader",
    "postgresql+psycopg2://u:p@h/db?dbname=ggtrader",
    "postgresql://u:p@h/db?database=ggtrader",
    "postgresql://u:p@h/db?host=other",
    "postgresql://u:p@h/db?DBNAME=ggtrader",
    "postgresql://u:p@h/db?sslmode=require&dbname=ggtrader",
    "postgresql://u:p@h/db?hostaddr=10.0.0.9",
    "postgresql://u:p@h/db?service=prod",
    "postgresql://u:p@h/db?port=5555",
    "postgresql://u:p@h/db?user=root",
    f"postgresql://u:p@h/db?password={MARK}",
]


def secret(url: str = "postgresql://u:p@h/db") -> fx._Secret:
    return fx._Secret(url)


class FakeCursor:
    def __init__(self, log: list[str], raise_on_exit: bool = False) -> None:
        self._log = log
        self._raise_on_exit = raise_on_exit

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._raise_on_exit:
            raise RuntimeError("cursor cleanup failed")

    def execute(self, statement: str) -> None:
        self._log.append(statement)


class FakeConn:
    def __init__(
        self,
        log: list[str],
        dsns: list[str],
        *,
        cursor_exit_raises: bool = False,
        close_raises: bool = False,
    ) -> None:
        self._log = log
        self._cursor_exit_raises = cursor_exit_raises
        self._close_raises = close_raises
        self.autocommit = False
        self.closed = False
        dsns.append("opened")

    def cursor(self) -> FakeCursor:
        assert self.autocommit is True, "the administrator connection must be AUTOCOMMIT"
        return FakeCursor(self._log, self._cursor_exit_raises)

    def close(self) -> None:
        self.closed = True
        if self._close_raises:
            raise RuntimeError("close failed")


def _fake_connect(
    *, first_cursor_exit_raises: bool = False, first_close_raises: bool = False
) -> tuple[list[str], list[str], object]:
    """A fake driver; the FIRST connection's cleanup can be made to fail, later ones are clean."""
    log: list[str] = []
    opened: list[str] = []

    def connect(dsn: str) -> FakeConn:
        assert dsn.startswith("postgresql://")
        first = not opened
        return FakeConn(
            log,
            opened,
            cursor_exit_raises=first and first_cursor_exit_raises,
            close_raises=first and first_close_raises,
        )

    return log, opened, connect


def _failing_connect(error: Exception) -> tuple[list[str], object]:
    attempts: list[str] = []

    def connect(dsn: str) -> FakeConn:
        attempts.append("attempt")
        raise error

    return attempts, connect


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_is_refused(name: str) -> None:
    with pytest.raises(fx.ScratchNameError):
        fx.assert_scratch_name(name)
    with pytest.raises(fx.ScratchNameError):
        fx.create_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.drop_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.scratch_engine_url(secret(), name)


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


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_raises_before_any_connection(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    log, opened, connect = _fake_connect()
    monkeypatch.setattr(fx, "new_scratch_name", lambda: name)
    with pytest.raises(fx.ScratchNameError):
        with fx.scratch_database_name(secret(), connect):
            pytest.fail("the body must not run")
    assert log == [] and opened == []


def test_create_then_drop_with_force_and_nothing_else() -> None:
    log, opened, connect = _fake_connect()
    with fx.scratch_database_name(secret("postgresql+psycopg2://u:p@h/db"), connect) as name:
        assert fx.SCRATCH_NAME.fullmatch(name)
        assert log == [f'CREATE DATABASE "{name}"']
    assert log == [f'CREATE DATABASE "{name}"', f'DROP DATABASE "{name}" WITH (FORCE)']
    assert len(opened) == 2


def test_drop_runs_when_the_body_raises() -> None:
    log, _, connect = _fake_connect()
    with pytest.raises(RuntimeError, match="boom"):
        with fx.scratch_database_name(secret(), connect) as name:
            raise RuntimeError("boom")
    assert log[-1] == f'DROP DATABASE "{name}" WITH (FORCE)'


def test_scratch_url_replaces_only_the_database() -> None:
    url = fx.scratch_engine_url(
        secret("postgresql+psycopg2://user:pw@dbhost:5433/adminDb"), "ggt_scratch_0123abcd"
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


# ------------------------------------------------------------ a URL query cannot re-point the run


@pytest.mark.parametrize("url", OVERRIDE_URLS)
def test_query_override_is_refused_before_any_connection(url: str) -> None:
    log, opened, connect = _fake_connect()
    with pytest.raises(fx.ScratchConfigError) as err:
        with fx.scratch_database_name(secret(url), connect):
            pytest.fail("the body must not run")
    assert log == [] and opened == []
    assert MARK not in str(err.value) and "ggtrader" not in str(err.value)
    with pytest.raises(fx.ScratchConfigError):
        fx.scratch_engine_url(secret(url), "ggt_scratch_0123abcd")


def test_query_override_cannot_reach_the_scratch_engine_url() -> None:
    """`?dbname=` would win over the path in the psycopg2 dialect; the fixture refuses it."""
    url = "postgresql+psycopg2://u:p@h/postgres?dbname=ggtrader"
    with pytest.raises(fx.ScratchConfigError):
        fx.scratch_engine_url(secret(url), "ggt_scratch_0123abcd")


def test_the_whole_query_is_dropped_from_the_scratch_url() -> None:
    admin = secret(f"postgresql+psycopg2://u:p@h:5433/db?sslmode=require&sslpassword={MARK}")
    url = fx.scratch_engine_url(admin, "ggt_scratch_0123abcd")
    assert dict(url.query) == {}
    assert url.database == "ggt_scratch_0123abcd"
    assert url.drivername == "postgresql+psycopg2"
    assert (url.username, url.password, url.host, url.port) == ("u", "p", "h", 5433)
    assert "?" not in url.render_as_string(hide_password=False)


def test_scratch_url_ignores_a_password_that_contains_the_scratch_name() -> None:
    name = "ggt_scratch_0123abcd"
    url = fx.scratch_engine_url(secret(f"postgresql://u:{name}@h/db"), name)
    assert url.database == name and url.password == name
    assert url.render_as_string(hide_password=True).count(name) == 1  # only the database


def test_non_secret_admin_value_is_refused() -> None:
    with pytest.raises(fx.ScratchConfigError):
        fx.validate_admin_url("postgresql://u:p@h/db")  # type: ignore[arg-type]


def test_unparseable_admin_url_is_refused_without_quoting_it() -> None:
    with pytest.raises(fx.ScratchConfigError) as err:
        fx.validate_admin_url(secret(f"not a url {MARK}"))
    assert MARK not in str(err.value) and MARK not in repr(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


# ---------------------------------------------------------------- the URL never leaves the holder


def test_secret_renders_as_redacted_everywhere() -> None:
    holder = secret(f"postgresql://admin:{MARK}@h/db")
    for rendered in (
        repr(holder),
        str(holder),
        f"{holder}",
        f"{holder!r}",
        "%s %r" % (holder, holder),
    ):
        assert MARK not in rendered
        assert "admin" not in rendered
    assert repr(holder) == str(holder) == "<redacted>"
    assert MARK not in repr([holder]) and MARK not in repr({"admin": holder})
    with pytest.raises(AttributeError):
        holder._value = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        vars(holder)  # no __dict__ to read the value from
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(holder)


def test_secret_is_read_from_the_environment_and_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@h/db")
    holder = fx.read_admin_secret()
    assert isinstance(holder, fx._Secret)
    assert MARK not in repr(holder) and MARK not in str(holder)
    assert holder.reveal() == f"postgresql://admin:{MARK}@h/db"
    monkeypatch.setenv(fx.ADMIN_URL_ENV, "")
    assert fx.read_admin_secret() is None
    monkeypatch.delenv(fx.ADMIN_URL_ENV)
    assert fx.read_admin_secret() is None


def test_scratch_engine_and_url_reprs_hide_the_password() -> None:
    url = fx.scratch_engine_url(
        secret(f"postgresql+psycopg2://admin:{MARK}@h:5433/db"), "ggt_scratch_0123abcd"
    )
    assert MARK not in repr(url)
    assert MARK not in url.render_as_string(hide_password=True)
    engine = create_engine(url, poolclass=NullPool)  # builds no connection
    try:
        assert MARK not in repr(engine)
    finally:
        engine.dispose()


def test_errors_and_skips_never_quote_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = secret(f"postgresql://admin:{MARK}@secrethost/db")
    attempts, connect = _failing_connect(psycopg2.OperationalError(f"could not connect {MARK}"))
    with pytest.raises(pytest.skip.Exception) as skipped:
        with fx.scratch_database_name(admin, connect):
            pytest.fail("the body must not run")
    assert MARK not in str(skipped.value) and "secrethost" not in str(skipped.value)
    assert skipped.value.__context__ is None  # the driver's message does not ride along
    for bad in ("ohlcv", f'x{MARK}"'):
        monkeypatch.setattr(fx, "new_scratch_name", lambda bad=bad: bad)
        with pytest.raises(fx.ScratchNameError) as err:
            with fx.scratch_database_name(admin, _fake_connect()[2]):
                pytest.fail("the body must not run")
        assert MARK not in str(err.value) and "secrethost" not in str(err.value)


def test_scratch_db_skip_message_never_quotes_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@secrethost/db")
    _, connect = _failing_connect(psycopg2.OperationalError("x"))
    with pytest.raises(pytest.skip.Exception) as skipped:
        with fx.scratch_db(connect=connect):
            pytest.fail("the body must not run")
    assert MARK not in str(skipped.value) and "secrethost" not in str(skipped.value)


# ------------------------------------------------- unreachable server skips; later errors fail


def test_unreachable_server_skips_and_runs_no_statement() -> None:
    attempts, connect = _failing_connect(psycopg2.OperationalError("refused"))
    with pytest.raises(pytest.skip.Exception):
        with fx.scratch_database_name(secret(), connect):
            pytest.fail("the body must not run")
    assert attempts == ["attempt"]  # no DROP attempt either: nothing was created


def test_non_connection_error_on_first_connect_is_not_a_skip() -> None:
    _, connect = _failing_connect(RuntimeError("driver bug"))
    with pytest.raises(RuntimeError, match="driver bug"):
        with fx.scratch_database_name(secret(), connect):
            pytest.fail("the body must not run")


def test_error_after_create_fails_loudly_and_still_drops() -> None:
    log, _, connect = _fake_connect()
    with pytest.raises(RuntimeError, match="after create"):
        with fx.scratch_database_name(secret(), connect) as name:
            raise RuntimeError("after create")
    assert log[-1] == f'DROP DATABASE "{name}" WITH (FORCE)'


def test_failing_drop_connection_fails_loudly() -> None:
    log, opened, real = _fake_connect()

    def connect(dsn: str):
        if opened:  # the second connection is the DROP
            raise psycopg2.OperationalError("server went away")
        return real(dsn)

    with pytest.raises(psycopg2.OperationalError, match="went away"):
        with fx.scratch_database_name(secret(), connect):
            pass
    assert len(log) == 1  # CREATE only; the failed DROP is an error, never a skip


# ---------------------------------------- a failing administrator cleanup never prevents the DROP


@pytest.mark.parametrize("which", ["first_cursor_exit_raises", "first_close_raises"])
def test_failing_create_cleanup_still_drops(which: str) -> None:
    log, _, connect = _fake_connect(**{which: True})
    with pytest.raises(RuntimeError):
        with fx.scratch_database_name(secret(), connect):
            pytest.fail("the body must not run after a failed CREATE cleanup")
    assert len(log) == 2
    assert log[0].startswith("CREATE DATABASE ")
    name = log[0].split('"')[1]
    assert log[1] == f'DROP DATABASE "{name}" WITH (FORCE)'


def test_failing_create_cleanup_and_body_error_still_drops() -> None:
    log, _, connect = _fake_connect(first_close_raises=True)
    with pytest.raises(RuntimeError):
        with fx.scratch_database_name(secret(), connect):
            pytest.fail("unreachable")
    assert log[-1].startswith("DROP DATABASE ")
