"""Verify the paper test connection guard."""

import socket
from contextlib import closing

import psycopg2
import pytest
from curl_cffi import requests
from sqlalchemy import create_engine

from tests.netguard import (
    EXEMPT_NODEIDS,
    RealConnectionAttempted,
    blocked_connections,
    guard_applies,
)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_inet_socket_connect_raises(method: str) -> None:
    with pytest.raises(RealConnectionAttempted, match="127.0.0.1"):
        with blocked_connections(), socket.socket() as sock:
            getattr(sock, method)(("127.0.0.1", 1))


def test_psycopg2_connect_raises() -> None:
    with pytest.raises(RealConnectionAttempted, match="psycopg2.connect"):
        with blocked_connections():
            psycopg2.connect("postgresql://u:p@127.0.0.1:1/x")


def test_sqlalchemy_postgres_engine_raises() -> None:
    with pytest.raises(RealConnectionAttempted):
        with blocked_connections():
            create_engine("postgresql+psycopg2://u:p@127.0.0.1:1/x").connect()


def test_sqlite_engine_allowed() -> None:
    with blocked_connections(), create_engine("sqlite://").connect() as conn:
        assert not conn.closed


def test_unix_socketpair_allowed() -> None:
    with blocked_connections():
        left, right = socket.socketpair()
        left.close()
        right.close()


@pytest.mark.parametrize("session", [False, True])
def test_curl_cffi_request_raises(session: bool) -> None:
    with pytest.raises(RealConnectionAttempted, match="Curl.perform.*example.invalid"):
        with blocked_connections():
            if session:
                with requests.Session() as client:
                    client.get("http://example.invalid")
            else:
                requests.get("http://example.invalid")


@pytest.mark.parametrize("lookup", ["gethostbyname", "getaddrinfo"])
def test_hostname_lookup_raises(lookup: str) -> None:
    with pytest.raises(RealConnectionAttempted, match="example.invalid"):
        with blocked_connections():
            getattr(socket, lookup)("example.invalid", 1) if lookup == "getaddrinfo" else getattr(
                socket, lookup
            )("example.invalid")


def test_swallowed_attempt_still_fails() -> None:
    with pytest.raises(RealConnectionAttempted, match="gethostbyname.*example.invalid"):
        with blocked_connections():
            try:
                socket.gethostbyname("example.invalid")
            except RealConnectionAttempted:
                pass


def test_guard_applies_decision() -> None:
    assert not EXEMPT_NODEIDS
    assert guard_applies("tests/paper/test_paper_persist.py::test_real_db", set())
    assert guard_applies("node", {"db"})
    assert not guard_applies("node", {"integration"})


@pytest.mark.parametrize(
    "operation", ["raw_connect", "raw_lookup", "host_ex", "nameinfo", "fqdn", "sendto"]
)
def test_audit_socket_paths_blocked(operation: str) -> None:
    import _socket

    with pytest.raises(RealConnectionAttempted):
        with blocked_connections():
            if operation == "raw_connect":
                with closing(_socket.socket()) as sock:
                    sock.connect(("127.0.0.1", 1))
            elif operation == "raw_lookup":
                _socket.getaddrinfo("example.invalid", 1)
            elif operation == "host_ex":
                socket.gethostbyname_ex("example.invalid")
            elif operation == "nameinfo":
                socket.getnameinfo(("127.0.0.1", 1), 0)
            elif operation == "fqdn":
                socket.getfqdn("example.invalid")
            else:
                with socket.socket(type=socket.SOCK_DGRAM) as sock:
                    sock.sendto(b"x", ("127.0.0.1", 1))


def test_unix_path_allowed(tmp_path) -> None:
    with blocked_connections(), socket.socket(socket.AF_UNIX) as server:
        server.bind(str(tmp_path / "sock"))
        server.listen()
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(tmp_path / "sock"))


def test_hook_inert_outside_guard() -> None:
    import _socket

    with closing(_socket.socket()) as sock, pytest.raises(OSError):
        sock.settimeout(1)
        sock.connect(("127.0.0.1", 1))


def test_nested_guards_record_hook_attempt() -> None:
    import _socket

    with pytest.raises(RealConnectionAttempted, match="127.0.0.1"):
        with blocked_connections():
            with pytest.raises(RealConnectionAttempted):
                with blocked_connections():
                    try:
                        with closing(_socket.socket()) as sock:
                            sock.connect(("127.0.0.1", 1))
                    except RealConnectionAttempted:
                        pass


def test_subprocess_allowed() -> None:
    import subprocess
    import sys

    with blocked_connections():
        subprocess.run([sys.executable, "-c", "pass"], check=True)
