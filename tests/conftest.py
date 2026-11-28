"""Shared disposable database fixture for forward-collection tests."""

import os
import secrets
import subprocess
import time

import psycopg2
import pytest


@pytest.fixture(scope="module")
def database() -> object:
    """Create a separate ephemeral container, bound to a random loopback port."""
    name = "ggt_forward_test_" + secrets.token_hex(12)
    password = secrets.token_urlsafe(24)
    environment = os.environ.copy()
    environment["POSTGRES_PASSWORD"] = password
    started = False
    conn = None
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-p",
                "127.0.0.1::5432",
                "--tmpfs",
                "/var/lib/postgresql/data:rw,size=128m",
                "-e",
                "POSTGRES_PASSWORD",
                "-e",
                "POSTGRES_USER=ggtrader",
                "-e",
                "POSTGRES_DB=postgres",
                "postgres:16",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        started = True
        address = subprocess.run(
            ["docker", "port", name, "5432/tcp"], check=True, capture_output=True, text=True
        ).stdout.strip()
        host, port = address.rsplit(":", 1)
        assert host == "127.0.0.1" and port.isdecimal() and int(port) != 5433
        for attempt in range(60):
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=int(port),
                    user="ggtrader",
                    password=password,
                    dbname="postgres",
                    connect_timeout=2,
                )
                break
            except psycopg2.OperationalError:
                if attempt == 59:
                    raise
                time.sleep(1)
        assert conn is not None
        conn.autocommit = True
        yield conn
    finally:
        if conn is not None:
            conn.close()
        if started:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)
