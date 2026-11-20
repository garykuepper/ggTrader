"""The scratch-database fixture's safety rules, tested without any database."""

from __future__ import annotations

import os
import pickle
import socket
import subprocess
import sys
import textwrap
from contextlib import AbstractContextManager
from pathlib import Path

import psycopg2
import pytest

from tests.paper import persist_scratch_db as fx

MARK = "PWMARK9f3a"  # a password-like marker injected through fake URLs; it must never surface
NAME = "ggt_scratch_0123abcd"  # a fixed generated name for the tests that pin it
URL = "postgresql://u:p@localhost:5433/db"
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

# Administrator URLs refused before any TCP probe or connection: anything but exactly one of
# localhost, 127.0.0.1 or [::1], any query or fragment, `%` in the host, a comma in the authority.
REFUSED_URLS = {
    "query-dbname": "postgresql://u:p@localhost:5433/db?dbname=ggtrader",
    "query-host": "postgresql://u:p@localhost:5433/db?host=other",
    "query-sslmode": "postgresql://u:p@localhost:5433/db?sslmode=require",
    "query-password": f"postgresql://u:p@localhost:5433/db?password={MARK}",
    "query-empty": "postgresql://u:p@localhost:5433/db?",
    "fragment": "postgresql://u:p@localhost:5433/db#x",
    "hostless": "postgresql:///db",
    "hostless-with-user": f"postgresql://u:{MARK}@/db",
    "hostless-port-only": f"postgresql://u:{MARK}@:5433/db",
    "encoded-comma-host": f"postgresql://u:{MARK}@h1%2Ch2:5433/db",
    "encoded-comma-after-localhost": f"postgresql://u:{MARK}@localhost%2C10.0.0.9:5433/db",
    "percent-socket-host": f"postgresql://u:{MARK}@%2Fvar%2Frun%2Fpostgresql/db",
    "percent-localhost": f"postgresql://u:{MARK}@local%68ost:5433/db",
    "comma-hosts": f"postgresql://u:{MARK}@localhost,127.0.0.1/db",
    "comma-host-ports": f"postgresql://u:{MARK}@localhost:5432,localhost:5433/db",
    "at-then-comma-in-password": f"postgresql://u:p@{MARK},x@localhost:5433/db",
    "at-in-password": f"postgresql://u:p@{MARK}@localhost:5433/db",
    "other-host": f"postgresql://u:{MARK}@db.example.com:5433/db",
    "docker-host": f"postgresql://u:{MARK}@host.docker.internal:5433/db",
    "other-loopback": f"postgresql://u:{MARK}@127.0.0.2:5433/db",
    "upper-case-localhost": f"postgresql://u:{MARK}@LOCALHOST:5433/db",
    "bare-ipv6": f"postgresql://u:{MARK}@::1/db",
    "no-user": "postgresql://localhost:5433/db",
    "port-zero": "postgresql://u:p@localhost:0/db",
    "port-too-big": "postgresql://u:p@localhost:65536/db",
    "port-not-digits": "postgresql://u:p@localhost:54x3/db",
    "not-postgres": f"mysql://u:{MARK}@localhost:5433/db",
    "not-a-url": f"not a url {MARK}",
}


def secret(url: str = URL) -> fx._Secret:
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
        late = self._server.after_errors.get(kind, [])
        if late:  # the server did the work, then the acknowledgment was lost
            raise late.pop(0)

    def fetchone(self) -> tuple[int] | None:
        return self._row

    def close(self) -> None:
        if self._server.cursor_close_raises and self._kind == "CREATE":
            raise RuntimeError("cursor cleanup failed")


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
    """A fake driver that records what an administrator session does.

    `errors` maps a statement kind (SELECT, CREATE, DROP) to exceptions raised before the work is
    done; `after_errors` to exceptions raised after it (a lost acknowledgment); `connect_errors`
    are raised by the next connection attempts; `reachable` is what the TCP probe answers.
    """

    def __init__(
        self,
        *,
        existing: tuple[str, ...] = (),
        errors: dict[str, list[BaseException]] | None = None,
        after_errors: dict[str, list[BaseException]] | None = None,
        connect_errors: list[BaseException] | None = None,
        cursor_close_raises: bool = False,
        close_raises: bool = False,
        reachable: bool = True,
    ) -> None:
        self.existing = set(existing)
        self.errors = errors or {}
        self.after_errors = after_errors or {}
        self.connect_errors = connect_errors or []
        self.cursor_close_raises = cursor_close_raises
        self.close_raises = close_raises
        self.reachable = reachable
        self.log: list[str] = []  # statements that ran
        self.params: list[tuple[str, ...] | None] = []
        self.attempts: list[str] = []  # statements sent, including ones that raised
        self.connect_kwargs: list[dict[str, object]] = []
        self.probes: list[tuple[str, int]] = []
        self.conns: list[FakeConn] = []

    @property
    def connect_attempts(self) -> int:
        return len(self.connect_kwargs)

    def probe(self, host: str, port: int) -> bool:
        self.probes.append((host, port))
        return self.reachable

    def connect(self, **kwargs: object) -> FakeConn:
        assert set(kwargs) <= {"host", "port", "user", "password", "dbname", "connect_timeout"}
        self.connect_kwargs.append(kwargs)
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        conn = FakeConn(self)
        self.conns.append(conn)
        return conn

    def drops(self) -> list[str]:
        return [s for s in self.log if s.startswith("DROP ")]

    def drop_attempts(self) -> list[str]:
        return [s for s in self.attempts if s.startswith("DROP ")]

    def creates(self) -> list[str]:
        return [s for s in self.log if s.startswith("CREATE ")]


def fixed_name(monkeypatch: pytest.MonkeyPatch, name: str = NAME) -> str:
    monkeypatch.setattr(fx, "new_scratch_name", lambda: name)
    return name


def enter(server: FakeServer, admin: fx._Secret | None = None) -> AbstractContextManager[str]:
    return fx.scratch_database_name(admin or secret(), server.connect, server.probe)


def chain(exc: BaseException) -> list[BaseException]:
    """`exc` and every exception linked to it as cause or context."""
    seen: list[BaseException] = []
    todo: list[BaseException | None] = [exc]
    while todo:
        item = todo.pop()
        if item is None or any(item is s for s in seen):
            continue
        seen.append(item)
        todo += [item.__cause__, item.__context__]
    return seen


def assert_redacted(exc: BaseException) -> None:
    """No exception in the chain quotes the marker or is a driver exception."""
    for item in chain(exc):
        assert MARK not in str(item) and MARK not in repr(item)
        assert not isinstance(item, (psycopg2.Error, PgError))


def visible_locals(tb: object) -> list[str]:
    """Reprs of the locals pytest would print: frames without `__tracebackhide__`."""
    shown: list[str] = []
    while tb is not None:
        local_vars = tb.tb_frame.f_locals  # type: ignore[attr-defined]
        if not local_vars.get("__tracebackhide__"):
            shown += [repr(v) for v in local_vars.values()]
        tb = tb.tb_next  # type: ignore[attr-defined]
    return shown


# ------------------------------------------------------------------------ names and statements


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_is_refused(name: str) -> None:
    for build in (fx.assert_scratch_name, fx.create_sql, fx.drop_sql, fx.exists_query):
        with pytest.raises(fx.ScratchNameError):
            build(name)


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
        with enter(server):
            pytest.fail("the body must not run")
    assert server.attempts == [] and server.connect_attempts == 0 and server.probes == []


# ------------------------------------------------------- one host, explicit port, no routing


@pytest.mark.parametrize("url_id", sorted(REFUSED_URLS))
def test_unsafe_admin_urls_are_refused_before_any_probe_or_connection(url_id: str) -> None:
    server = FakeServer()
    with pytest.raises(fx.ScratchConfigError) as err:
        with enter(server, secret(REFUSED_URLS[url_id])):
            pytest.fail("the body must not run")
    assert server.probes == [] and server.connect_attempts == 0 and server.attempts == []
    assert_redacted(err.value)
    assert "ggtrader" not in str(err.value) and "example" not in str(err.value)
    assert all(MARK not in shown for shown in visible_locals(err.tb))


@pytest.mark.parametrize(
    ("url", "host", "port", "dbname"),
    [
        ("postgresql://u:p@localhost:5433/db", "localhost", 5433, "db"),
        ("postgresql+psycopg2://u:p@127.0.0.1:6000/db", "127.0.0.1", 6000, "db"),
        ("postgresql://u:p@[::1]:5433/db", "::1", 5433, "db"),
        ("postgresql://u:p@localhost/db", "localhost", 5433, "db"),  # port set explicitly
        ("postgresql://u:p@localhost:5433", "localhost", 5433, "postgres"),  # database explicit
        ("postgresql://u:p%40%2Cq@localhost:5433/db", "localhost", 5433, "db"),
    ],
)
def test_accepted_urls_connect_with_explicit_host_port_and_database(
    url: str, host: str, port: int, dbname: str
) -> None:
    server = FakeServer()
    with enter(server, secret(url)):
        pass
    assert server.probes == [(host, port)]
    kwargs = server.connect_kwargs[0]
    assert (kwargs["host"], kwargs["port"], kwargs["dbname"], kwargs["user"]) == (
        host,
        port,
        dbname,
        "u",
    )
    assert isinstance(kwargs["port"], int)
    if "%40" in url:
        assert kwargs["password"] == "p@,q"  # decoded, handed over as one keyword


@pytest.mark.parametrize("variable", ["PGHOSTADDR", "PGSERVICE"])
def test_libpq_routing_variables_are_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "10.0.0.9")
    server = FakeServer()
    with pytest.raises(fx.ScratchConfigError):
        with enter(server):
            pytest.fail("the body must not run")
    assert server.probes == [] and server.connect_attempts == 0


def test_a_raw_string_is_refused_and_not_kept_in_the_raising_frame() -> None:
    with pytest.raises(fx.ScratchConfigError) as err:
        fx.admin_endpoint(f"postgresql://u:{MARK}@localhost:5433/db")  # type: ignore[arg-type]
    tb = err.tb
    while tb is not None:
        if tb.tb_frame.f_code is fx.admin_endpoint.__code__:
            assert "admin" not in tb.tb_frame.f_locals
        tb = tb.tb_next
    assert all(MARK not in shown for shown in visible_locals(err.tb))


# ------------------------------------------------ ownership: DROP only for what this run created


def test_success_runs_exists_create_and_one_drop_on_one_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeServer()
    fixed_name(monkeypatch)
    with enter(server) as name:
        assert name == NAME
        assert server.log == [fx.exists_query(NAME)[0], f'CREATE DATABASE "{NAME}"']
        assert server.params == [(NAME,), None]
    assert server.log == [
        fx.exists_query(NAME)[0],
        f'CREATE DATABASE "{NAME}"',
        f'DROP DATABASE "{NAME}" WITH (FORCE)',
    ]
    assert server.connect_attempts == 1  # CREATE and DROP share ONE administrator session
    assert server.conns[0].closed


def test_name_already_taken_issues_neither_create_nor_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(existing=(NAME,))
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError, match="already exists"):
        with enter(server):
            pytest.fail("the body must not run")
    assert server.log == [fx.exists_query(NAME)[0]]
    assert server.creates() == [] and server.drops() == []
    assert NAME in server.existing  # someone else's database was left alone


def test_failing_existence_check_creates_and_drops_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(errors={"SELECT": [PgError(f"boom {MARK}", "42501")]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert server.creates() == [] and server.drop_attempts() == []
    assert "42501" in str(err.value)
    assert_redacted(err.value)


@pytest.mark.parametrize(
    "error",
    [
        PgError(f'database "{NAME}" already exists {MARK}', "42P04"),
        psycopg2.OperationalError(f"server closed the connection unexpectedly {MARK}"),
        TimeoutError(f"timed out {MARK}"),
        RuntimeError(f"driver bug {MARK}"),
    ],
)
def test_a_failed_create_does_not_drop_and_names_the_database(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    server = FakeServer(errors={"CREATE": [error]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert server.drop_attempts() == []
    assert NAME in str(err.value) and "manual" in str(err.value)
    assert_redacted(err.value)


def test_create_done_but_acknowledgment_lost_is_ambiguous_no_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server created the database, then the reply was lost: we cannot prove it is ours."""
    server = FakeServer(after_errors={"CREATE": [psycopg2.OperationalError(f"lost {MARK}")]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert server.drop_attempts() == [] and NAME in server.existing
    assert NAME in str(err.value)
    assert_redacted(err.value)


def test_keyboard_interrupt_during_create_is_loud_names_the_database_and_drops_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeServer(after_errors={"CREATE": [KeyboardInterrupt()]})
    fixed_name(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert server.drop_attempts() == []
    assert NAME in str(err.value)
    assert server.conns[0].closed


def test_drop_only_ever_names_the_database_this_run_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = iter(["ggt_scratch_aaaaaaaa", "ggt_scratch_bbbbbbbb"])
    monkeypatch.setattr(fx, "new_scratch_name", lambda: next(names))
    server = FakeServer(existing=("ggt_scratch_ffffffff", "ggtrader"))
    for _ in range(2):
        with enter(server):
            pass
    assert server.drops() == [
        'DROP DATABASE "ggt_scratch_aaaaaaaa" WITH (FORCE)',
        'DROP DATABASE "ggt_scratch_bbbbbbbb" WITH (FORCE)',
    ]
    assert server.existing == {"ggt_scratch_ffffffff", "ggtrader"}  # the others were never touched


def test_drop_runs_when_the_body_raises_and_the_body_error_propagates() -> None:
    server = FakeServer()
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError) as err:
        with enter(server) as name:
            raise boom
    assert err.value is boom
    assert server.drops() == [f'DROP DATABASE "{name}" WITH (FORCE)']


def test_a_keyboard_interrupt_in_the_body_drops_and_propagates_itself() -> None:
    server = FakeServer()
    stop = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as err:
        with enter(server) as name:
            raise stop
    assert err.value is stop
    assert server.drops() == [f'DROP DATABASE "{name}" WITH (FORCE)']


def test_failing_cursor_cleanup_after_create_still_drops() -> None:
    """CREATE's execute returned, then cursor cleanup failed: the database is ours, drop it."""
    server = FakeServer(cursor_close_raises=True)
    with pytest.raises(fx.ScratchDbError):
        with enter(server):
            pytest.fail("the body must not run")
    created = server.creates()
    assert len(created) == 1
    assert server.drops() == [created[0].replace("CREATE", "DROP") + " WITH (FORCE)"]


def test_a_failing_close_at_the_end_is_ignored_after_the_drop() -> None:
    server = FakeServer(close_raises=True)
    with enter(server):
        pass
    assert len(server.drops()) == 1


# ------------------------------------------- one session: no retry, no second connection


@pytest.mark.parametrize("where", ["before", "after"])
def test_a_failed_drop_is_never_retried_and_names_the_leftover(
    monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Lost before or after the server dropped: either way, ONE connection and one DROP."""
    lost = psycopg2.OperationalError(f"server closed the connection {MARK}")
    server = FakeServer(**{"errors" if where == "before" else "after_errors": {"DROP": [lost]}})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            pass
    assert server.connect_attempts == 1
    assert len(server.drop_attempts()) == 1
    assert NAME in str(err.value)
    assert f'DROP DATABASE "{NAME}" WITH (FORCE);' in str(err.value)
    assert_redacted(err.value)


def test_a_keyboard_interrupt_during_drop_is_loud_and_stays_an_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeServer(errors={"DROP": [KeyboardInterrupt()]})
    fixed_name(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as err:
        with enter(server):
            pass
    assert NAME in str(err.value) and server.connect_attempts == 1


def test_a_body_error_and_a_failed_drop_both_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(errors={"DROP": [RuntimeError(MARK)]})
    fixed_name(monkeypatch)
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            raise ValueError("test body failed")
    assert NAME in str(err.value)
    assert isinstance(err.value.__context__, ValueError)
    assert_redacted(err.value)


# --------------------------------------- a forked child never touches the parent's database


def test_a_forked_child_unwinding_the_fixture_does_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeServer()
    with enter(server):
        monkeypatch.setattr(fx, "_pid", lambda: -1)  # this process is not the one that created it
    assert server.drop_attempts() == []
    assert not server.conns[0].closed  # the parent's socket is not touched either
    assert capsys.readouterr() == ("", "")


def test_a_forked_child_unwinding_after_a_body_error_only_propagates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeServer(errors={"DROP": [RuntimeError("must not run")]})
    with pytest.raises(ValueError, match="child"):
        with enter(server):
            monkeypatch.setattr(fx, "_pid", lambda: -1)
            raise ValueError("child")
    assert server.drop_attempts() == []


# ---------------------------- skip ONLY when the TCP probe fails; after it, every error fails


def test_an_unreachable_server_skips_without_any_connection() -> None:
    server = FakeServer(reachable=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        with enter(server, secret(f"postgresql://admin:{MARK}@localhost:5433/db")):
            pytest.fail("the body must not run")
    assert server.connect_attempts == 0 and server.attempts == []
    assert MARK not in str(skipped.value) and "admin" not in str(skipped.value)
    assert skipped.value.__context__ is None


FAIL_ERRORS = [
    psycopg2.OperationalError(f'FATAL:  password authentication failed for user "x" {MARK}'),
    PgError("FATAL: no matching rule", "28000"),
    PgError("wording", "28P01"),
    psycopg2.OperationalError('FATAL:  database "timed out" does not exist'),
    psycopg2.OperationalError("connection to server failed: Connection refused"),
    PgError("some wording we do not know", "08006"),
    PgError("too many connections", "53300"),
    RuntimeError(f"Connection refused {MARK}"),
    TimeoutError("timed out"),
]


@pytest.mark.parametrize("error", FAIL_ERRORS, ids=lambda e: str(e)[:40])
def test_after_the_probe_every_connection_error_fails_redacted(error: Exception) -> None:
    server = FakeServer(connect_errors=[error])
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server, secret(f"postgresql://svcuser:{MARK}@localhost:5433/db")):
            pytest.fail("the body must not run")
    assert server.attempts == [] and server.connect_attempts == 1
    assert "svcuser" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert_redacted(err.value)


def test_system_exit_during_connect_propagates_redacted() -> None:
    server = FakeServer(connect_errors=[SystemExit(f"postgresql://u:{MARK}@localhost/db")])
    with pytest.raises(SystemExit) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert err.value.code == 1
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert_redacted(err.value)


def test_keyboard_interrupt_during_connect_propagates_itself_unchained() -> None:
    server = FakeServer(connect_errors=[KeyboardInterrupt(MARK)])
    with pytest.raises(KeyboardInterrupt) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert_redacted(err.value)


def test_another_base_exception_during_connect_becomes_a_redacted_failure() -> None:
    class Odd(BaseException):
        pass

    server = FakeServer(connect_errors=[Odd(MARK)])
    with pytest.raises(fx.ScratchDbError) as err:
        with enter(server):
            pytest.fail("the body must not run")
    assert_redacted(err.value)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (None, True),
        (ConnectionRefusedError(), False),
        (socket.timeout(), False),
        (socket.gaierror(), False),
        (OSError("No route to host"), False),
    ],
)
def test_the_tcp_probe_is_a_plain_socket_connect_with_a_timeout(
    monkeypatch: pytest.MonkeyPatch, outcome: OSError | None, expected: bool
) -> None:
    calls: list[tuple[object, ...]] = []

    class Sock:
        def __enter__(self) -> Sock:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def create_connection(address: tuple[str, int], timeout: float) -> Sock:
        calls.append((address, timeout))
        if outcome is not None:
            raise outcome
        return Sock()

    monkeypatch.setattr(fx.socket, "create_connection", create_connection)
    assert fx.tcp_reachable("localhost", 5433) is expected
    assert calls == [(("localhost", 5433), 3.0)]


# ---------------------------------------------------------------- the engine and the URL holder


def test_scratch_db_skips_without_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fx.ADMIN_URL_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        with fx.scratch_db():
            pytest.fail("the body must not run")


def test_the_engine_connects_to_the_scratch_database_on_the_same_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@127.0.0.1/db")
    fixed_name(monkeypatch)
    admin, engine_side = FakeServer(), FakeServer()
    with fx.scratch_db(
        schema=False, connect=admin.connect, engine_connect=engine_side.connect, probe=admin.probe
    ) as engine:
        assert MARK not in repr(engine) and MARK not in repr(engine.url)
        assert engine.url.password is None and engine.url.host is None
        with pytest.raises(Exception):  # the fake is not a DBAPI connection; only kwargs matter
            engine.connect()
    assert engine_side.connect_kwargs[0] == {
        "host": "127.0.0.1",
        "port": 5433,
        "user": "admin",
        "password": MARK,
        "dbname": NAME,
        "connect_timeout": fx.CONNECT_TIMEOUT_S,
    }
    assert len(admin.drops()) == 1


def test_a_failed_engine_connection_in_a_test_body_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@localhost:5433/db")
    admin = FakeServer()
    engine_side = FakeServer(connect_errors=[PgError(f"FATAL {MARK}", "28P01")])
    with fx.scratch_db(
        schema=False, connect=admin.connect, engine_connect=engine_side.connect, probe=admin.probe
    ) as engine:
        with pytest.raises(fx.ScratchDbError) as err:
            engine.connect()
    assert_redacted(err.value)
    assert len(admin.drops()) == 1


def test_secret_renders_as_redacted_everywhere() -> None:
    holder = secret(f"postgresql://admin:{MARK}@localhost/db")
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


def test_the_endpoint_renders_without_credentials() -> None:
    endpoint = fx.admin_endpoint(secret(f"postgresql://admin:{MARK}@localhost:5433/db"))
    assert MARK not in repr(endpoint) and "admin" not in repr(endpoint)
    assert (endpoint.host, endpoint.port) == ("localhost", 5433)
    with pytest.raises(TypeError):
        pickle.dumps(endpoint)


def test_secret_is_read_from_the_environment_and_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fx.ADMIN_URL_ENV, f"postgresql://admin:{MARK}@localhost/db")
    holder = fx.read_admin_secret()
    assert isinstance(holder, fx._Secret)
    assert MARK not in repr(holder) and MARK not in str(holder)
    assert holder.reveal() == f"postgresql://admin:{MARK}@localhost/db"
    monkeypatch.setenv(fx.ADMIN_URL_ENV, "")
    assert fx.read_admin_secret() is None
    monkeypatch.delenv(fx.ADMIN_URL_ENV)
    assert fx.read_admin_secret() is None


# ------------------------------------- a traceback with locals shown never reveals the password

TINY_TESTS = textwrap.dedent(
    """
    import psycopg2
    import pytest

    from tests.paper import persist_scratch_db as fx
    from tests.paper.test_persist_scratch_db import FakeServer, PgError

    MARK = "PWMA" + "RK9f3a"  # built at run time, so no printed source line holds it


    def admin_url(host="localhost:5433", query=""):
        return "postgresql://admin:" + MARK + "@" + host + "/db" + query


    def enter(server, url=None):
        return fx.scratch_database_name(fx._Secret(url or admin_url()), server.connect,
                                        server.probe)


    def _leaky_connect(**kwargs):
        # a frame whose argument and local both hold the password, as the driver's own frame does
        leaked_dsn = str(kwargs) + MARK
        raise PgError("FATAL: password authentication failed " + leaked_dsn, "28P01")


    def _exiting_connect(**kwargs):
        raise SystemExit("exit " + str(kwargs))


    def test_first_connection_fails():
        with fx.scratch_database_name(fx._Secret(admin_url()), _leaky_connect, lambda h, p: True):
            pass


    def test_connect_raises_system_exit():
        with fx.scratch_database_name(fx._Secret(admin_url()), _exiting_connect,
                                      lambda h, p: True):
            pass


    def test_existence_check_fails():
        with enter(FakeServer(errors={"SELECT": [PgError("x " + MARK, "42501")]})):
            pass


    def test_create_fails():
        with enter(FakeServer(errors={"CREATE": [psycopg2.OperationalError("lost " + MARK)]})):
            pass


    def test_drop_fails():
        with enter(FakeServer(errors={"DROP": [RuntimeError(MARK)]})):
            pass


    def test_schema_connection_fails(monkeypatch):
        monkeypatch.setenv(fx.ADMIN_URL_ENV, admin_url())
        server = FakeServer()
        with fx.scratch_db(connect=server.connect, engine_connect=_leaky_connect,
                           probe=server.probe):
            pass


    def test_test_body_connection_fails(monkeypatch):
        monkeypatch.setenv(fx.ADMIN_URL_ENV, admin_url())
        server = FakeServer()
        with fx.scratch_db(schema=False, connect=server.connect, engine_connect=_leaky_connect,
                           probe=server.probe) as engine:
            with engine.connect():
                pass


    def test_query_password_is_refused():
        fx.admin_endpoint(fx._Secret(admin_url(query="?password=" + MARK)))


    def test_encoded_comma_host_is_refused():
        fx.admin_endpoint(fx._Secret(admin_url(host="h1%2Ch2:5433")))


    def test_raw_string_is_refused():
        fx.admin_endpoint(admin_url())


    @pytest.fixture
    def drop_fails_in_teardown():
        with enter(FakeServer(errors={"DROP": [PgError("gone " + MARK, "57P01")]})) as name:
            yield name


    def test_teardown_drop_fails(drop_fails_in_teardown):
        assert drop_fails_in_teardown.startswith("ggt_scratch_")


    @pytest.fixture
    def engine_whose_drop_fails(monkeypatch):
        monkeypatch.setenv(fx.ADMIN_URL_ENV, admin_url())
        server = FakeServer(after_errors={"DROP": [psycopg2.OperationalError("ack " + MARK)]})
        with fx.scratch_db(schema=False, connect=server.connect, probe=server.probe) as engine:
            yield engine


    def test_scratch_db_teardown_drop_fails(engine_whose_drop_fails):
        assert engine_whose_drop_fails is not None


    def test_control_shows_locals():
        shown_only_by_showlocals = "".join(["VISIBLE", "LOCAL", "5c7d"])
        raise RuntimeError("control failure")
    """
)
CONTROL_VALUE = "VISIBLELOCAL5c7d"


def run_tiny(tmp_path: Path, *extra: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != fx.ADMIN_URL_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src")])
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "test_tiny.py", "-p", "no:cacheprovider", "-q", "-rA",
         "--tb=long", "-W", "ignore", *extra],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    return done.stdout + done.stderr


def test_a_failing_test_or_teardown_with_locals_shown_never_prints_the_password(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_tiny.py").write_text(TINY_TESTS)
    shown = run_tiny(tmp_path, "--showlocals")
    # Every scenario failed: 11 tests, plus 2 teardown errors after 2 passing tests.
    assert "11 failed, 2 passed, 2 errors" in shown, shown[-3000:]
    assert "ERROR at teardown of test_teardown_drop_fails" in shown
    assert "ERROR at teardown of test_scratch_db_teardown_drop_fails" in shown
    assert CONTROL_VALUE in shown  # --showlocals is active: a plain local's value is printed
    assert MARK not in shown
    hidden = run_tiny(tmp_path)
    assert CONTROL_VALUE not in hidden  # the control value appears only through --showlocals
    assert MARK not in hidden
