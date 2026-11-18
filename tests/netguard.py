"""Fail paper tests on attempted external connections."""

import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import curl_cffi
import psycopg2

EXEMPT_NODEIDS: frozenset[str] = frozenset()


class RealConnectionAttempted(RuntimeError):
    """An external connection was attempted during a test."""


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
        attempts.append(attempt)
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
        if attempts:
            raise RealConnectionAttempted("; ".join(attempts))
