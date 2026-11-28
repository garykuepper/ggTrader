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


def test_alpaca_failure_includes_status_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    response = requests.Response()
    response.status_code = 403
    monkeypatch.setenv("APCA_API_KEY_ID", "example_key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "example_secret")
    source = collector.Sources(None)

    def fail(*args: object, **kwargs: object) -> object:
        raise requests.HTTPError("sensitive response", response=response)

    monkeypatch.setattr(source, "get", fail)
    with pytest.raises(RuntimeError, match="HTTP 403") as exc:
        source.bars(["AAA"], dt.date(2026, 11, 27))
    assert "example_secret" not in str(exc.value)


def test_dry_run_shares_sample_on_nonweekly_day() -> None:
    class SampleSource(Source):
        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.sampled = symbols
            return super().shares(symbols, day)

    source = SampleSource()
    result = collector.collect(dt.date(2026, 11, 27), True, source, shares_sample=1)
    assert result["shares"] == 1
    assert source.sampled == ["AAA"]


def test_sample_cannot_be_used_for_writes(conn: object) -> None:
    with pytest.raises(ValueError, match="dry.run"):
        collector.collect(dt.date(2026, 11, 27), False, Source(), conn, shares_sample=1)


def test_half_day_window_ends_after_close_and_before_next_session() -> None:
    start, end = collector.bar_window(dt.date(2026, 11, 27))
    assert start == "2026-11-27T00:00:00Z"
    assert end == "2026-11-27T21:00:00Z"


def test_same_day_window_waits_for_sip_delay() -> None:
    day = dt.date(2026, 11, 25)
    with pytest.raises(ValueError, match="15 minutes"):
        collector.bar_window(day, dt.datetime(2026, 11, 25, 21, 10, tzinfo=dt.timezone.utc))
    _, end = collector.bar_window(day, dt.datetime(2026, 11, 25, 21, 20, tzinfo=dt.timezone.utc))
    assert end == "2026-11-25T21:00:00Z"


def test_retry_after_http_date() -> None:
    now = dt.datetime(2026, 11, 28, 17, 0, tzinfo=dt.timezone.utc)
    assert collector.retry_delay("Sat, 28 Nov 2026 17:00:20 GMT", now, 0) == 20


def test_daily_collection_never_calls_yahoo(conn: object) -> None:
    class NoYahoo(Source):
        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            raise AssertionError("daily collection must not fetch shares")

    collector.collect(dt.date(2026, 11, 23), False, NoYahoo(), conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.shares_outstanding")
        assert cur.fetchone()[0] == 0


def test_shares_only_is_idempotent_and_does_not_duplicate_daily(conn: object) -> None:
    source = Source()
    day = dt.date(2026, 11, 23)
    collector.collect(day, False, source, conn)
    collector.collect_shares(day, source, conn)
    collector.collect_shares(day, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM forward.shares_outstanding")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM forward.daily_bars")
        assert cur.fetchone()[0] == 1


def test_catch_up_fills_missed_day_from_earlier_listing(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 18), lambda day: source, conn)
    assert result["collected"] == [dt.date(2026, 11, 17), dt.date(2026, 11, 18)]
    collector.catch_up(dt.date(2026, 11, 18), lambda day: source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day='2026-11-17'")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT source FROM forward.loads WHERE day='2026-11-17'")
        assert "catchup" in cur.fetchone()[0]
        cur.execute(
            "SELECT reason FROM forward.gaps WHERE day='2026-11-17' AND symbol=%s",
            (collector.LISTING_GAP_SYMBOL,),
        )
        assert cur.fetchone()[0] == "listing_missing"


def test_catch_up_stops_without_an_earlier_listing(conn: object) -> None:
    with pytest.raises(ValueError, match="earlier.*listing"):
        collector.catch_up(
            dt.date(2026, 11, 18), lambda day: Source(), conn, start_day=dt.date(2026, 11, 17)
        )


def test_catch_up_stops_at_ten_days_with_clear_message(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 2), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 23), lambda day: source, conn)
    assert len(result["collected"]) == 10
    assert "10" in result["message"]
    assert result["remaining"] > 0


def test_catch_up_uses_saved_same_day_listing_before_earlier_one(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)

    class SavedSource(Source):
        def saved_listing(self, day: dt.date) -> list[dict] | None:
            if day == dt.date(2026, 11, 17):
                self.listed = ["CCC"]
                return self.listing()
            return None

    saved = SavedSource()
    collector.catch_up(dt.date(2026, 11, 18), lambda day: saved, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM forward.listing_snapshots WHERE day='2026-11-17'")
        assert cur.fetchall() == [("CCC",)]
        cur.execute(
            "SELECT count(*) FROM forward.gaps WHERE reason='listing_missing' AND day='2026-11-17'"
        )
        assert cur.fetchone()[0] == 0


def test_catch_up_never_uses_a_later_listing(conn: object) -> None:
    collector.collect(dt.date(2026, 11, 18), False, Source(), conn)
    with pytest.raises(ValueError, match="earlier.*listing"):
        collector.catch_up(
            dt.date(2026, 11, 18), lambda day: Source(), conn, start_day=dt.date(2026, 11, 17)
        )


def test_shares_load_does_not_block_daily_load(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    day = dt.date(2026, 11, 23)
    collector.collect_shares(day, source, conn)
    assert not collector.collect(day, False, source, conn).get("already_collected", False)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day=%s", (day,))
        assert cur.fetchone()[0] == 2


def test_shares_failure_writes_no_partial_load_and_can_retry(conn: object) -> None:
    class RetrySource(Source):
        fail = True

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            if self.fail:
                raise RuntimeError("temporary failure")
            return super().shares(symbols, day)

    source = RetrySource()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    day = dt.date(2026, 11, 23)
    with pytest.raises(RuntimeError, match="temporary"):
        collector.collect_shares(day, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day=%s", (day,))
        assert cur.fetchone()[0] == 0
    source.fail = False
    assert collector.collect_shares(day, source, conn)["shares"] == 2
    assert collector.collect_shares(day, source, conn)["already_collected"]


def test_share_checkpoint_resumes_without_refetching_completed_symbols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    calls = []
    fail = [True]

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            calls.append(self.symbol)
            if self.symbol == "BBB" and fail[0]:
                raise OSError("temporary")
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    path = tmp_path / "shares.json"
    with pytest.raises(RuntimeError, match="Yahoo shares failed"):
        collector.Sources(path).shares(["AAA", "BBB"], dt.date(2026, 11, 23))
    fail[0] = False
    result = collector.Sources(path).shares(["AAA", "BBB"], dt.date(2026, 11, 23))
    assert calls == ["AAA", "BBB", "BBB"]
    assert [row["symbol"] for row in result] == ["AAA", "BBB"]


def test_catch_up_fills_hole_before_a_later_success(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    collector.collect(dt.date(2026, 11, 18), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 19), lambda day: source, conn)
    assert result["collected"] == [dt.date(2026, 11, 17), dt.date(2026, 11, 19)]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day='2026-11-18'")
        assert cur.fetchone()[0] == 1
