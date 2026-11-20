"""The scratch-database fixture's safety rules, tested without any database."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import psycopg2
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from tests.paper import persist_scratch_db as fx

MARK = "PWMARK9f3a"  # a password-like marker injected through fake URLs; it must never surface
NAME = "ggt_scratch_0123abcd"  # a fixed generated name for the tests that pin it
REPO = Path(__file__).resolve().parents[2]

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

# Administrator URLs that must be refused before any connection is opened: any query string at
# all, a password in the query, and more than one host.
REFUSED_URLS = [
    "postgresql://u:p@h/db?dbname=ggtrader",
    "postgresql+psycopg2://u:p@h/db?dbname=ggtrader",
    "postgresql://u:p@h/db?database=ggtrader",
    "postgresql://u:p@h/db?host=other",
    "postgresql://u:p@h/db?DBNAME=ggtrader",
    "postgresql://u:p@h/db?sslmode=require",
    "postgresql://u:p@h/db?target_session_attrs=read-write",
    "postgresql://u:p@h/db?hostaddr=10.0.0.9",
    "postgresql://u:p@h/db?service=prod",
    "postgresql://u:p@h/db?port=5555",
    "postgresql://u:p@h/db?user=root",
    f"postgresql://u:p@h/db?password={MARK}",
    "postgresql://u:p@h/db?",  # an empty query is still a query string
    "postgresql://u:p@h1,h2/db",
    "postgresql://u:p@h1:5432,h2:5433/db",
    f"postgresql://u:{MARK}@h1,h2:5433/db",
]


def secret(url: str = "postgresql://u:p@h/db") -> fx._Secret:
    return fx._Secret(url)


class PgError(psycopg2.OperationalError):
    """A driver error that carries a SQLSTATE."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self._code = code

    @property
    def pgcode(self) -> str | None:  # type: ignore[override]
        return self._code


class FakeCursor:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self._row: tuple[int] | None = None
        self._kind = ""

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server.cursor_exit_raises and self._kind == "CREATE":
            raise RuntimeError("cursor cleanup failed")

    def execute(self, statement: str, params: tuple[str, ...] | None = None) -> None:
        kind = self._kind = statement.split(" ", 1)[0]
        self._server.attempts.append(statement)
        queue = self._server.errors.get(kind, [])
        if queue:
            raise queue.pop(0)
        self._server.log.append(statement)
        self._server.params.append(params)
        if kind == "SELECT":
            assert params is not None
            self._row = (1,) if params[0] in self._server.existing else None
        elif kind == "CREATE":
            self._server.existing.add(statement.split('"')[1])
        elif kind == "DROP":
            self._server.existing.discard(statement.split('"')[1])

    def fetchone(self) -> tuple[int] | None:
        return self._row


class FakeConn:
    def __init__(self, server: FakeServer) -> None:
        self._server = server
        self.autocommit = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        assert self.autocommit is True, "the administrator connection must be AUTOCOMMIT"
        return FakeCursor(self._server)

    def close(self) -> None:
        self.closed = True
        if self._server.close_raises:
            raise RuntimeError("close failed")


class FakeServer:
    """A fake driver that records the statements an administrator session runs.

    `errors` maps a statement kind (SELECT, CREATE, DROP) to exceptions raised in order;
    `connect_errors` are raised by the next connection attempts.
    """

    def __init__(
        self,
        *,
        existing: tuple[str, ...] = (),
        errors: dict[str, list[Exception]] | None = None,
        connect_errors: list[Exception] | None = None,
        cursor_exit_raises: bool = False,
        close_raises: bool = False,
    ) -> None:
        self.existing = set(existing)
        self.errors = errors or {}
        self.connect_errors = connect_errors or []
        self.cursor_exit_raises = cursor_exit_raises
        self.close_raises = close_raises
        self.log: list[str] = []  # statements that ran
        self.params: list[tuple[str, ...] | None] = []
        self.attempts: list[str] = []  # statements sent, including ones that raised
        self.opened = 0
        self.connect_attempts = 0

    def connect(self, dsn: str) -> FakeConn:
        assert dsn.startswith("postgresql://")
        self.connect_attempts += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        self.opened += 1
        return FakeConn(self)

    def drops(self) -> list[str]:
        return [s for s in self.log if s.startswith("DROP ")]

    def creates(self) -> list[str]:
        return [s for s in self.log if s.startswith("CREATE ")]


def fixed_name(monkeypatch: pytest.MonkeyPatch, name: str = NAME) -> str:
    monkeypatch.setattr(fx, "new_scratch_name", lambda: name)
    return name


def run_fixture(server: FakeServer, admin: fx._Secret | None = None) -> str:
    """Enter and leave the fixture with an empty body; returns the name."""
    with fx.scratch_database_name(admin or secret(), server.connect) as name:
        return name


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_is_refused(name: str) -> None:
    with pytest.raises(fx.ScratchNameError):
        fx.assert_scratch_name(name)
    with pytest.raises(fx.ScratchNameError):
        fx.create_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.drop_sql(name)
    with pytest.raises(fx.ScratchNameError):
        fx.exists_query(name)
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


def test_statements_are_exactly_exists_create_and_drop_force() -> None:
    assert fx.create_sql(NAME) == f'CREATE DATABASE "{NAME}"'
    assert fx.drop_sql(NAME) == f'DROP DATABASE "{NAME}" WITH (FORCE)'
    # The one SELECT: the name travels as a bound parameter, never inside the text.
    assert fx.exists_query(NAME) == ("SELECT 1 FROM pg_database WHERE datname = %s", (NAME,))


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_raises_before_any_connection(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    server = FakeServer()
    fixed_name(monkeypatch, name)
    with pytest.raises(fx.ScratchNameError):
        with fx.scratch_database_name(secret(), server.connect):
            pytest.fail("the body must not run")
    assert server.attempts == [] and server.connect_attempts == 0


# ------------------------------------------- ownership: DROP only for what this run created


def test_success_issues_exists_create_and_one_drop_on_one_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeServer()
    fixed_name(monkeypatch)
    with fx.scratch_database_name(secret("postgresql+psycopg2://u:p@h/db"), server.connect) as name:
        assert name == NAME
        assert server.log == [fx.exists_query(NAME)[0], f'CREATE DATABASE "{NAME}"']
        assert server.params[0] == (NAME,) and server.params[1:] == [None]
    assert server.log == [
        fx.exists_query(NAME)[0],
        f'CREATE DATABASE "{NAME}"',
        f'DROP DATABASE "{NAME}" WITH (FORCE)',
    ]
    assert len(server.drops()) == 1
    assert server.opened == 1  # CREATE and DROP share ONE administrator session
    assert server.connect_attempts == 1


def test_name_already_taken_issues_neither_create_nor_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(existing=(NAME,))
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError, match="already exists"):
        with fx.scratch_database_name(secret(), server.connect):
            pytest.fail("the body must not run")
    assert server.log == [fx.exists_query(NAME)[0]]
    assert server.creates() == [] and server.drops() == []
    assert NAME in server.existing  # someone else's database was left alone


def test_create_collision_42p04_does_not_drop_a_foreign_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SELECT said free; another run created the name first; CREATE raises 42P04."""
    server = FakeServer(errors={"CREATE": [PgError(f'database "{NAME}" already exists', "42P04")]})
    fixed_name(monkeypatch)
    server.existing.add("some_other_db")
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(secret(), server.connect):
            pytest.fail("the body must not run")
    assert server.drops() == [] and not any(a.startswith("DROP ") for a in server.attempts)
    assert "42P04" in str(err.value) and "ggt_scratch_%" in str(err.value)
    assert "already exists" not in str(err.value)  # driver text is withheld


@pytest.mark.parametrize(
    "error",
    [
        psycopg2.OperationalError("server closed the connection unexpectedly"),
        psycopg2.OperationalError(f"connection lost {MARK}"),
        TimeoutError(f"timed out {MARK}"),
        RuntimeError("driver bug"),
    ],
)
def test_create_with_an_unknown_outcome_does_not_drop_and_says_to_check(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    server = FakeServer(errors={"CREATE": [error]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(secret(), server.connect):
            pytest.fail("the body must not run")
    assert server.drops() == [] and not any(a.startswith("DROP ") for a in server.attempts)
    message = str(err.value)
    assert "ggt_scratch_%" in message and "manually" in message
    assert MARK not in message and "driver bug" not in message
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_failing_existence_check_creates_and_drops_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(errors={"SELECT": [PgError(f"boom {MARK}", "42501")]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(secret(), server.connect):
            pytest.fail("the body must not run")
    assert server.creates() == [] and server.drops() == []
    assert MARK not in str(err.value) and "42501" in str(err.value)


def test_drop_only_ever_names_the_database_this_run_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = iter(["ggt_scratch_aaaaaaaa", "ggt_scratch_bbbbbbbb"])
    monkeypatch.setattr(fx, "new_scratch_name", lambda: next(names))
    server = FakeServer(existing=("ggt_scratch_ffffffff", "ggtrader"))
    first = run_fixture(server)
    second = run_fixture(server)
    assert server.drops() == [f'DROP DATABASE "{first}" WITH (FORCE)'] + [
        f'DROP DATABASE "{second}" WITH (FORCE)'
    ]
    assert server.existing == {"ggt_scratch_ffffffff", "ggtrader"}  # the others were never touched


def test_drop_runs_when_the_body_raises() -> None:
    server = FakeServer()
    with pytest.raises(RuntimeError, match="boom"):
        with fx.scratch_database_name(secret(), server.connect) as name:
            raise RuntimeError("boom")
    assert server.drops() == [f'DROP DATABASE "{name}" WITH (FORCE)']
    assert server.opened == 1


@pytest.mark.parametrize("which", ["cursor_exit_raises", "close_raises"])
def test_failing_cleanup_after_create_still_drops(which: str) -> None:
    """CREATE ran (execute returned), then cursor cleanup failed: the database is ours, drop it."""
    server = FakeServer(**{which: True})
    ran: list[str] = []
    try:
        with fx.scratch_database_name(secret(), server.connect) as name:
            ran.append(name)
    except Exception as exc:  # cursor cleanup failing is loud; closing at the very end is not
        assert which == "cursor_exit_raises" and isinstance(exc, fx.ScratchDbError)
    created = server.creates()
    assert len(created) == 1
    assert server.drops() == [f'DROP DATABASE "{created[0].split(chr(34))[1]}" WITH (FORCE)']


# -------------------------------- one administrator session; a lost session retries once


def test_a_lost_session_retries_the_drop_once_on_a_fresh_connection() -> None:
    lost = psycopg2.OperationalError("server closed the connection unexpectedly")
    server = FakeServer(errors={"DROP": [lost]})
    with fx.scratch_database_name(secret(), server.connect) as name:
        pass
    assert server.opened == 2  # the original session, then one fresh connection for the retry
    assert server.drops() == [f'DROP DATABASE "{name}" WITH (FORCE)']
    assert len([a for a in server.attempts if a.startswith("DROP ")]) == 2


def test_two_failed_drops_fail_loudly_naming_the_leftover() -> None:
    lost = [psycopg2.OperationalError(f"gone {MARK}") for _ in range(3)]
    server = FakeServer(errors={"DROP": lost})
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(secret(), server.connect):
            pass
    assert server.drops() == []
    assert len([a for a in server.attempts if a.startswith("DROP ")]) == 2  # once, then one retry
    assert "ggt_scratch_%" in str(err.value) and MARK not in str(err.value)


def test_a_failed_fresh_connection_for_the_retry_fails_loudly() -> None:
    server = FakeServer(errors={"DROP": [psycopg2.OperationalError("gone")]})
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(secret(), server.connect):
            server.connect_errors.append(psycopg2.OperationalError(f"refused {MARK}"))
    assert server.drops() == []
    assert "ggt_scratch_%" in str(err.value) and MARK not in str(err.value)


def test_a_body_error_and_a_failed_drop_both_surface() -> None:
    server = FakeServer(errors={"DROP": [RuntimeError("x"), RuntimeError("y")]})
    with pytest.raises(fx.ScratchDbError):
        with fx.scratch_database_name(secret(), server.connect):
            raise ValueError("test body failed")


def test_scratch_url_replaces_only_the_database() -> None:
    url = fx.scratch_engine_url(secret("postgresql+psycopg2://user:pw@dbhost:5433/adminDb"), NAME)
    assert (url.username, url.host, url.port) == ("user", "dbhost", 5433)
    assert url.database == NAME
    assert url.password == "pw"
    assert "pw" not in repr(url)  # SQLAlchemy hides the password in a repr


def test_scratch_db_skips_without_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fx.ADMIN_URL_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        with fx.scratch_db():
            pytest.fail("the body must not run")


# ---------------------------- a query string or a second host cannot re-point the run


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_query_or_multi_host_url_is_refused_before_any_connection(url: str) -> None:
    server = FakeServer()
    with pytest.raises(fx.ScratchConfigError) as err:
        with fx.scratch_database_name(secret(url), server.connect):
            pytest.fail("the body must not run")
    assert server.attempts == [] and server.connect_attempts == 0
    assert MARK not in str(err.value) and "ggtrader" not in str(err.value)
    assert MARK not in repr(err.value)
    with pytest.raises(fx.ScratchConfigError):
        fx.scratch_engine_url(secret(url), NAME)


def test_a_refused_url_is_not_held_in_a_local_of_the_raising_frame() -> None:
    """`?password=` must not sit in a parsed URL (its repr shows the query) when it is refused."""
    with pytest.raises(fx.ScratchConfigError) as err:
        fx.validate_admin_url(secret(f"postgresql://u:p@h/db?password={MARK}"))
    tb = err.tb
    while tb is not None:
        for value in tb.tb_frame.f_locals.values():
            assert MARK not in repr(value)
        tb = tb.tb_next


def test_the_scratch_url_has_no_query_and_keeps_the_single_host() -> None:
    url = fx.scratch_engine_url(secret(f"postgresql+psycopg2://u:{MARK}@h:5433/db"), NAME)
    assert dict(url.query) == {}
    assert url.database == NAME
    assert url.drivername == "postgresql+psycopg2"
    assert (url.username, url.password, url.host, url.port) == ("u", MARK, "h", 5433)
    assert "?" not in url.render_as_string(hide_password=False)


def test_scratch_url_ignores_a_password_that_contains_the_scratch_name() -> None:
    url = fx.scratch_engine_url(secret(f"postgresql://u:{NAME}@h/db"), NAME)
    assert url.database == NAME and url.password == NAME
    assert url.render_as_string(hide_password=True).count(NAME) == 1  # only the database


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
    url = fx.scratch_engine_url(secret(f"postgresql+psycopg2://admin:{MARK}@h:5433/db"), NAME)
    assert MARK not in repr(url)
    assert MARK not in url.render_as_string(hide_password=True)
    engine = create_engine(url, poolclass=NullPool)  # builds no connection
    try:
        assert MARK not in repr(engine)
    finally:
        engine.dispose()


def test_the_scratch_engine_carries_no_password_to_inspect() -> None:
    engine = fx.build_scratch_engine(secret(f"postgresql://admin:{MARK}@127.0.0.1:1/db"), NAME)
    try:
        assert MARK not in repr(engine) and MARK not in repr(engine.url)
        assert engine.url.password is None  # the DSN lives only in the connection factory
    finally:
        engine.dispose()


# --------------------------- skip ONLY for a positively identified connection failure

SKIP_ERRORS = [
    psycopg2.OperationalError(
        'connection to server at "127.0.0.1", port 5432 failed: Connection refused'
    ),
    psycopg2.OperationalError('connection to server at "h" failed: timeout expired'),
    psycopg2.OperationalError('connection to server at "h" failed: Connection timed out'),
    psycopg2.OperationalError('could not translate host name "nohost" to address: no such name'),
    psycopg2.OperationalError('connection to server at "h" failed: No route to host'),
    psycopg2.OperationalError('connection to server at "h" failed: Network is unreachable'),
    PgError("some wording we do not know", "08006"),
    PgError("some wording we do not know", "08001"),
]

FAIL_ERRORS = [
    psycopg2.OperationalError('FATAL:  password authentication failed for user "admin"'),
    PgError("FATAL: no matching rule", "28000"),
    PgError("wording", "28P01"),
    # An authentication failure whose text also mentions a skip phrase still fails.
    PgError("password authentication failed after Connection refused by proxy", "28P01"),
    psycopg2.OperationalError('FATAL:  database "x" does not exist'),
    psycopg2.OperationalError("something nobody has classified"),
    PgError("too many connections", "53300"),
    RuntimeError("Connection refused"),  # not a driver connection error at all
    TimeoutError("timed out"),
]


@pytest.mark.parametrize("error", SKIP_ERRORS, ids=lambda e: str(e)[:40])
def test_positively_identified_connection_failures_skip(error: Exception) -> None:
    server = FakeServer(connect_errors=[error])
    with pytest.raises(pytest.skip.Exception) as skipped:
        with fx.scratch_database_name(
            secret(f"postgresql://admin:{MARK}@secrethost/db"), server.connect
        ):
            pytest.fail("the body must not run")
    assert server.attempts == []  # no statement, so no DROP
    assert MARK not in str(skipped.value) and "secrethost" not in str(skipped.value)
    assert skipped.value.__context__ is None  # the driver's message does not ride along


@pytest.mark.parametrize("error", FAIL_ERRORS, ids=lambda e: str(e)[:40])
def test_authentication_and_unclassified_errors_fail_with_a_redacted_message(
    error: Exception,
) -> None:
    server = FakeServer(connect_errors=[error])
    with pytest.raises(fx.ScratchDbError) as err:
        with fx.scratch_database_name(
            secret(f"postgresql://svcuser:{MARK}@secrethost/db"), server.connect
        ):
            pytest.fail("the body must not run")
    assert server.attempts == []
    text = str(err.value)
    assert MARK not in text and "secrethost" not in text and "svcuser" not in text
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_scratch_db_skip_message_never_quotes_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@secrethost/db")
    server = FakeServer(connect_errors=[psycopg2.OperationalError("Connection refused")])
    with pytest.raises(pytest.skip.Exception) as skipped:
        with fx.scratch_db(connect=server.connect):
            pytest.fail("the body must not run")
    assert MARK not in str(skipped.value) and "secrethost" not in str(skipped.value)


def test_errors_and_skips_never_quote_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = secret(f"postgresql://admin:{MARK}@secrethost/db")
    for bad in ("ohlcv", f'x{MARK}"'):
        monkeypatch.setattr(fx, "new_scratch_name", lambda bad=bad: bad)
        with pytest.raises(fx.ScratchNameError) as err:
            with fx.scratch_database_name(admin, FakeServer().connect):
                pytest.fail("the body must not run")
        assert MARK not in str(err.value) and "secrethost" not in str(err.value)


def test_a_failing_drop_connection_is_never_a_skip() -> None:
    server = FakeServer(errors={"DROP": [psycopg2.OperationalError("gone")]})
    with pytest.raises(fx.ScratchDbError):
        with fx.scratch_database_name(secret(), server.connect):
            server.connect_errors.append(psycopg2.OperationalError("Connection refused"))


# ------------------------------------- a traceback with locals shown never reveals the password

TINY_TESTS = textwrap.dedent(
    """
    import psycopg2

    from tests.paper import persist_scratch_db as fx
    from tests.paper.test_persist_scratch_db import FakeServer, PgError

    MARK = @@MARK@@
    CONTROL = @@CONTROL@@


    def admin_url(host="h", query=""):
        # built at run time: pytest prints the source of a failing test, never this helper's frame
        return "postgresql://admin:" + MARK + "@" + host + "/db" + query


    def _leaky_connect(dsn):
        # a frame whose argument and local both hold the marker, as the driver's own frame does
        leaked_dsn = dsn + MARK
        raise PgError("FATAL: password authentication failed " + leaked_dsn, "28P01")


    def test_first_connection_fails():
        with fx.scratch_database_name(fx._Secret(admin_url()), _leaky_connect):
            pass


    def test_existence_check_fails():
        server = FakeServer(errors={"SELECT": [PgError("x " + MARK, "42501")]})
        with fx.scratch_database_name(fx._Secret(admin_url()), server.connect):
            pass


    def test_create_fails():
        server = FakeServer(errors={"CREATE": [psycopg2.OperationalError("lost " + MARK)]})
        with fx.scratch_database_name(fx._Secret(admin_url()), server.connect):
            pass


    def test_drop_fails_twice():
        server = FakeServer(errors={"DROP": [RuntimeError(MARK), RuntimeError(MARK)]})
        with fx.scratch_database_name(fx._Secret(admin_url()), server.connect):
            pass


    def test_schema_connection_fails(monkeypatch):
        monkeypatch.setenv(fx.ADMIN_URL_ENV, admin_url("127.0.0.1:1"))
        with fx.scratch_db(connect=FakeServer().connect):
            pass


    def test_test_body_connection_fails(monkeypatch):
        monkeypatch.setenv(fx.ADMIN_URL_ENV, admin_url("127.0.0.1:1"))
        with fx.scratch_db(schema=False, connect=FakeServer().connect) as engine:
            with engine.connect():
                pass


    def test_query_password_is_refused():
        fx.validate_admin_url(fx._Secret(admin_url(query="?password=" + MARK)))


    def test_control_shows_locals():
        control_local = CONTROL
        assert control_local == "never"
    """
)


def test_a_failing_test_with_locals_shown_never_prints_the_password(tmp_path: Path) -> None:
    control = "CONTROLMARK7e21"
    (tmp_path / "test_tiny.py").write_text(
        TINY_TESTS.replace("@@MARK@@", repr(MARK)).replace("@@CONTROL@@", repr(control))
    )
    env = {k: v for k, v in os.environ.items() if k != fx.ADMIN_URL_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src")])
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "test_tiny.py", "--showlocals", "-p", "no:cacheprovider",
         "-q", "-rA", "--tb=long", "-W", "ignore"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    output = done.stdout + done.stderr
    assert "8 failed" in output, output[-3000:]  # every scenario failed (none skipped or passed)
    assert control in output  # control: --showlocals does print locals, so the check has teeth
    assert MARK not in output
