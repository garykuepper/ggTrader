"""Exercise the forward collector with fake sources and disposable Postgres."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name: str, path: Path) -> object:
    """Load a script or fixture by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture_module = load_file("forward_schema_fixture", ROOT / "tests/db/test_forward_schema.py")
database = fixture_module.database
collector = load_file("forward_collect", ROOT / "scripts/forward_collect.py")


@pytest.fixture
def conn(database: object) -> object:
    """Use a fresh scratch schema for each test."""
    with database.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS forward CASCADE")
        cur.execute((ROOT / "sql/forward_schema.sql").read_text())
    return database


class Source:
    """A network-free source with controllable listing and prices."""

    def __init__(self) -> None:
        self.listed = ["AAA", "BBB"]
        self.returned = ["AAA"]
        self.calls = 0

    def listing(self) -> list[dict]:
        self.calls += 1
        return [
            {
                "symbol": x,
                "exchange": "Q",
                "security_name": x + " Common Stock",
                "is_etf": False,
                "is_test": False,
            }
            for x in self.listed
        ]

    def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
        return [
            {"symbol": x, "open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "volume": 10}
            for x in self.returned
            if x in symbols
        ]

    def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
        return [{"symbol": x, "shares": 100} for x in symbols]


def test_rerun_same_day_adds_no_rows(conn: object) -> None:
    source = Source()
    day = dt.date(2026, 11, 23)
    collector.collect(day, False, source, conn)
    collector.collect(day, False, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 1
    assert source.calls == 1


def test_missing_symbol_is_recorded_as_gap(conn: object) -> None:
    collector.collect(dt.date(2026, 11, 23), False, Source(), conn)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM forward.gaps")
        assert cur.fetchall() == [("BBB",)]


def test_symbol_ended_only_after_five_missing_trading_days(conn: object) -> None:
    source = Source()
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.returned = ["AAA"]
    for day in [17, 18, 19, 20]:
        assert collector.collect(dt.date(2026, 11, day), False, source, conn)["newly_ended"] == 0
    assert collector.collect(dt.date(2026, 11, 23), False, source, conn)["newly_ended"] == 1


def test_holiday_makes_no_gaps(conn: object) -> None:
    source = Source()
    assert collector.collect(dt.date(2026, 11, 26), False, source, conn)["gaps"] == 0
    assert source.calls == 0


def test_low_coverage_returns_alert_flag(conn: object) -> None:
    assert collector.collect(dt.date(2026, 11, 23), True, Source(), conn)["alert"] is True
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 0


def test_first_trading_day_uses_holiday_calendar() -> None:
    assert collector.is_shares_day(dt.date(2026, 5, 26))
    assert not collector.is_shares_day(dt.date(2026, 5, 29))


def test_excludes_special_securities() -> None:
    base = {"is_etf": False, "is_test": False}
    for name in ["ABC Warrants", "ABC Units", "ABC Rights", "ABC Preferred Stock"]:
        assert not collector.common_candidate({**base, "security_name": name})
    assert collector.common_candidate({**base, "security_name": "ABC Common Stock"})


def test_disappearing_symbol_is_recorded_as_gap(conn: object) -> None:
    source = Source()
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.listed = ["AAA"]
    source.returned = ["AAA"]
    collector.collect(dt.date(2026, 11, 17), False, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM forward.gaps WHERE day = '2026-11-17'")
        assert cur.fetchall() == [("BBB",)]


def test_new_symbol_missing_for_five_sessions_can_end(conn: object) -> None:
    source = Source()
    for day in [16, 17, 18, 19]:
        assert collector.collect(dt.date(2026, 11, day), False, source, conn)["newly_ended"] == 0
    assert collector.collect(dt.date(2026, 11, 20), False, source, conn)["newly_ended"] == 1


def test_rate_limit_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = []
    calls = []

    class Response:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers = {"Retry-After": "3"}

        def raise_for_status(self) -> None:
            assert self.status_code == 200

    def request(*args: object, **kwargs: object) -> Response:
        calls.append(1)
        return Response(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(collector.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(collector.requests, "get", request)
    source = collector.Sources(None)
    source.get("https://example.invalid", kind="alpaca")
    assert sleeps == [0.5, 3.0, 0.5]
    assert source.requests["alpaca"] == 2


def test_share_count_has_observation_date_and_dry_source_no_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    class Ticker:
        def get_shares_full(self, **kwargs: object) -> object:
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", lambda symbol: Ticker())
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    result = collector.Sources(None).shares(["AAA"], dt.date(2026, 11, 23))
    assert result[0]["shares"] == 100
    assert result[0]["observed_day"] == collector.observation_day()
