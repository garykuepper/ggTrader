"""Verify the paper test connection guard."""

import socket

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
