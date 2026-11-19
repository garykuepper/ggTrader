"""Fail paper tests on attempted external connections.

Does not cover child processes, threads connecting after exit, pre-opened pools,
fixtures scoped above a paper test, or tests outside paper and the chain test.
An already-bound psycopg2.connect can bypass its patch (libpq may bypass audit).
Patched paths stop before audit, so each attempt is recorded once per guard.
"""

import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import curl_cffi
import psycopg2

EXEMPT_NODEIDS: frozenset[str] = frozenset()


class RealConnectionAttempted(RuntimeError):
    """An external connection was attempted during a test."""


_lock = threading.RLock()
_active: list[list[str]] = []
_events = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "socket.sendto",
    }
)


def _record(attempt: str) -> None:
    with _lock:
        for attempts in _active:
            attempts.append(attempt)


def _audit(event: str, args: tuple) -> None:
    if event not in _events:
        return
    with _lock:
        if not _active:
            return
        try:
            destination = args[-1] if event in {"socket.connect", "socket.sendto"} else args[0]
            if event in {"socket.connect", "socket.sendto"} and not isinstance(destination, tuple):
                return
            attempt = f"{event}: {destination!r}"
            _record(attempt)
        except BaseException:
            attempt = f"{event}: unprintable destination"
            _record(attempt)
        raise RealConnectionAttempted(attempt)


sys.addaudithook(_audit)


def guard_applies(nodeid: str, markers: set[str]) -> bool:
    """Select tests that must run without external connections."""
    return "integration" not in markers and nodeid not in EXEMPT_NODEIDS


@contextmanager
def blocked_connections() -> Iterator[None]:
    """Record and block synchronous network and database connection attempts."""
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_setopt = curl_cffi.Curl.setopt
    curl_urls: dict[curl_cffi.Curl, object] = {}

    def block(name: str, destination: object) -> None:
        attempt = f"{name}: {destination!r}"
        _record(attempt)
        raise RealConnectionAttempted(attempt)

    def connect(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            block("socket.connect", address)
        return original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: object) -> int:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            block("socket.connect_ex", address)
        return original_connect_ex(sock, address)

    def setopt(curl: curl_cffi.Curl, option: object, value: object) -> int:
        if option == curl_cffi.CurlOpt.URL:
            curl_urls[curl] = value
        return original_setopt(curl, option, value)

    def perform(curl: curl_cffi.Curl, *args: object, **kwargs: object) -> None:
        block("Curl.perform", curl_urls.get(curl, "URL set before guard"))

    def deny(name: str):
        def blocked(*args, **kwargs):
            block(name, (args, kwargs))

        return blocked

    with _lock:
        _active.append(attempts)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", connect))
            stack.enter_context(patch.object(socket.socket, "connect_ex", connect_ex))
            stack.enter_context(patch.object(socket, "getaddrinfo", deny("socket.getaddrinfo")))
            stack.enter_context(patch.object(socket, "gethostbyname", deny("socket.gethostbyname")))
            stack.enter_context(patch.object(psycopg2, "connect", deny("psycopg2.connect")))
            stack.enter_context(patch.object(curl_cffi.Curl, "setopt", setopt))
            stack.enter_context(patch.object(curl_cffi.Curl, "perform", perform))
            yield
    finally:
        with _lock:
            _active.pop(next(i for i, item in enumerate(_active) if item is attempts))
        if attempts:
            raise RealConnectionAttempted("; ".join(attempts))
