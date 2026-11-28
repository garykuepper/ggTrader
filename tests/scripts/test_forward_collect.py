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
        self.raw = []
        self.progress = {}
        self.fillers = [f"FILL{i}" for i in range(8)]

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
            for x in self.listed + self.fillers
        ]

    def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
        return [
            {"symbol": x, "open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "volume": 10}
            for x in self.returned + self.fillers
            if x in symbols
        ]

    def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
        return [{"symbol": x, "shares": 100} for x in symbols]


@pytest.mark.db
@pytest.mark.integration
def test_rerun_same_day_adds_no_rows(conn: object) -> None:
    source = Source()
    day = dt.date(2026, 11, 23)
    collector.collect(day, False, source, conn)
    collector.collect(day, False, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 1
    assert source.calls == 1


@pytest.mark.db
@pytest.mark.integration
def test_missing_symbol_is_recorded_as_gap(conn: object) -> None:
    collector.collect(dt.date(2026, 11, 23), False, Source(), conn)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM forward.gaps")
        assert cur.fetchall() == [("BBB",)]


@pytest.mark.db
@pytest.mark.integration
def test_symbol_ended_only_after_five_missing_trading_days(conn: object) -> None:
    source = Source()
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.returned = ["AAA"]
    for day in [17, 18, 19, 20]:
        assert collector.collect(dt.date(2026, 11, day), False, source, conn)["newly_ended"] == 0
    assert collector.collect(dt.date(2026, 11, 23), False, source, conn)["newly_ended"] == 1


@pytest.mark.db
@pytest.mark.integration
def test_holiday_makes_no_gaps(conn: object) -> None:
    source = Source()
    assert collector.collect(dt.date(2026, 11, 26), False, source, conn)["gaps"] == 0
    assert source.calls == 0


@pytest.mark.db
@pytest.mark.integration
def test_low_coverage_returns_alert_flag(conn: object) -> None:
    source = Source()
    source.returned = []
    source.fillers = []
    assert collector.collect(dt.date(2026, 11, 23), True, source, conn)["alert"] is True
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 0


def test_holiday_week_uses_monday_week_key() -> None:
    assert collector.sessions(dt.date(2026, 5, 25), dt.date(2026, 5, 26)) == [dt.date(2026, 5, 26)]
    assert collector.shares_week(dt.date(2026, 5, 26), dt.date(2026, 5, 26)) == dt.date(2026, 5, 25)


def test_excludes_special_securities() -> None:
    base = {"is_etf": False, "is_test": False}
    for name in ["ABC Warrants", "ABC Units", "ABC Rights", "ABC Preferred Stock"]:
        assert not collector.common_candidate({**base, "security_name": name})
    assert collector.common_candidate({**base, "security_name": "ABC Common Stock"})


@pytest.mark.db
@pytest.mark.integration
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


@pytest.mark.db
@pytest.mark.integration
def test_symbol_never_seen_with_bar_does_not_gain_status(conn: object) -> None:
    source = Source()
    for day in [16, 17, 18, 19, 20]:
        assert collector.collect(dt.date(2026, 11, day), False, source, conn)["newly_ended"] == 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.symbol_status WHERE symbol = 'BBB'")
        assert cur.fetchone()[0] == 0


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


@pytest.mark.db
@pytest.mark.integration
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


@pytest.mark.db
@pytest.mark.integration
def test_daily_collection_never_calls_yahoo(conn: object) -> None:
    class NoYahoo(Source):
        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            raise AssertionError("daily collection must not fetch shares")

    collector.collect(dt.date(2026, 11, 23), False, NoYahoo(), conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.shares_outstanding")
        assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
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
        assert cur.fetchone()[0] == 10
        cur.execute("SELECT count(*) FROM forward.daily_bars")
        assert cur.fetchone()[0] == 9


@pytest.mark.db
@pytest.mark.integration
def test_bootstrap_weekend_waits_for_next_session(conn: object) -> None:
    result = collector.catch_up(
        dt.date(2026, 11, 27), lambda day: Source(), conn, current_day=dt.date(2026, 11, 28)
    )
    assert result["collected"] == []
    assert result["message"] == "collection starts at the next session"
    assert not result.get("failures")


@pytest.mark.db
@pytest.mark.integration
def test_bootstrap_monday_does_not_collect_friday_and_rerun_is_noop(conn: object) -> None:
    monday = dt.date(2026, 11, 30)
    calls = []

    def factory(day: dt.date) -> Source:
        calls.append(day)
        return Source()

    assert collector.catch_up(monday, factory, conn, current_day=monday)["collected"] == [monday]
    assert collector.catch_up(monday, factory, conn, current_day=monday)["collected"] == []
    assert calls == [monday]


@pytest.mark.db
@pytest.mark.integration
def test_catch_up_fills_missed_day_from_earlier_listing(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 18), lambda day: source, conn)
    assert result["collected"] == [dt.date(2026, 11, 18), dt.date(2026, 11, 17)]
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


@pytest.mark.db
@pytest.mark.integration
def test_bootstrap_ignores_explicit_earlier_start(conn: object) -> None:
    result = collector.catch_up(
        dt.date(2026, 11, 18), lambda day: Source(), conn, start_day=dt.date(2026, 11, 17)
    )
    assert result["collected"] == [dt.date(2026, 11, 18)]
    assert not result["failures"]


@pytest.mark.db
@pytest.mark.integration
def test_catch_up_stops_at_ten_days_with_clear_message(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 2), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 23), lambda day: source, conn)
    assert len(result["collected"]) == 11
    assert "10" in result["message"]
    assert result["remaining"] > 0


@pytest.mark.db
@pytest.mark.integration
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
    saved.returned = ["AAA", "BBB", "CCC"]
    collector.catch_up(dt.date(2026, 11, 18), lambda day: saved, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM forward.listing_snapshots WHERE day='2026-11-17'")
        assert set(cur.fetchall()) == {("CCC",)} | {(x,) for x in saved.fillers}
        cur.execute(
            "SELECT count(*) FROM forward.gaps WHERE reason='listing_missing' AND day='2026-11-17'"
        )
        assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
def test_catch_up_never_uses_a_later_listing(conn: object) -> None:
    collector.collect(dt.date(2026, 11, 18), False, Source(), conn)
    with pytest.raises(ValueError, match="earlier saved listing"):
        collector.saved_listing(conn, dt.date(2026, 11, 17), strictly_before=True)


@pytest.mark.db
@pytest.mark.integration
def test_shares_load_does_not_block_daily_load(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    day = dt.date(2026, 11, 23)
    collector.collect_shares(day, source, conn)
    assert not collector.collect(day, False, source, conn).get("already_collected", False)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day=%s", (day,))
        assert cur.fetchone()[0] == 2


@pytest.mark.db
@pytest.mark.integration
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
    assert collector.collect_shares(day, source, conn)["shares"] == 10
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
    with pytest.raises(RuntimeError, match="10%"):
        collector.Sources(path).shares(["AAA", "BBB"], dt.date(2026, 11, 23))
    fail[0] = False
    result = collector.Sources(path).shares(["AAA", "BBB"], dt.date(2026, 11, 23))
    assert calls == ["AAA", "BBB", "BBB"]
    assert [row["symbol"] for row in result] == ["AAA", "BBB"]


@pytest.mark.db
@pytest.mark.integration
def test_catch_up_fills_hole_before_a_later_success(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    collector.collect(dt.date(2026, 11, 18), False, source, conn)
    result = collector.catch_up(dt.date(2026, 11, 19), lambda day: source, conn)
    assert result["collected"] == [dt.date(2026, 11, 19), dt.date(2026, 11, 17)]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE day='2026-11-18'")
        assert cur.fetchone()[0] == 1


@pytest.mark.db
@pytest.mark.integration
def test_low_coverage_rejected_without_writing(conn: object) -> None:
    source = Source()
    source.returned = []
    source.fillers = []
    with pytest.raises(ValueError, match="90%"):
        collector.collect(dt.date(2026, 11, 23), False, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
def test_transaction_rolls_back_on_midway_invalid_bar(conn: object) -> None:
    class BadBar(Source):
        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            rows = super().bars(symbols, day)
            rows[-1]["volume"] = -1
            return rows

    import psycopg2

    with pytest.raises(psycopg2.errors.CheckViolation):
        collector.collect(dt.date(2026, 11, 23), False, BadBar(), conn)
    with conn.cursor() as cur:
        for table in (
            "loads",
            "listing_snapshots",
            "daily_bars",
            "gaps",
            "symbol_status",
            "shares_outstanding",
        ):
            from psycopg2 import sql

            cur.execute(sql.SQL("SELECT count(*) FROM forward.{}").format(sql.Identifier(table)))
            assert cur.fetchone()[0] == 0


def test_retry_delay_is_capped() -> None:
    assert collector.retry_delay("500", dt.datetime.now(dt.timezone.utc), 0) == 120


def test_nasdaq_keeps_na_etfs_and_test_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        text = (
            "Symbol|Security Name|ETF|Test Issue\nNA|NA Common Stock|N|N\n"
            "ETF1|Example ETF|Y|N\nTEST1|Test Common Stock|N|Y\n"
        )

    class Other:
        text = "ACT Symbol|Security Name|ETF|Test Issue|Exchange\n"

    source = collector.Sources(None)
    monkeypatch.setattr(
        source, "get", lambda url, **kwargs: Response() if "nasdaqlisted" in url else Other()
    )
    rows = source.listing()
    assert {row["symbol"] for row in rows} == {"NA", "ETF1", "TEST1"}
    assert next(row for row in rows if row["symbol"] == "ETF1")["is_etf"]
    assert next(row for row in rows if row["symbol"] == "TEST1")["is_test"]


@pytest.mark.db
@pytest.mark.integration
def test_today_is_collected_before_old_backlog_and_failure_does_not_block(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 2), False, source, conn)
    calls = []

    class Flaky(Source):
        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            calls.append(day)
            if day == dt.date(2026, 11, 3):
                raise RuntimeError("old-day failure")
            return super().bars(symbols, day)

    result = collector.catch_up(dt.date(2026, 11, 23), lambda day: Flaky(), conn)
    assert calls[0] == dt.date(2026, 11, 23)
    assert dt.date(2026, 11, 4) in result["collected"]
    assert result["failures"][0]["day"] == dt.date(2026, 11, 3)
    assert len(calls) == 11


def test_yahoo_continues_after_one_symbol_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            if self.symbol == "BAD":
                raise OSError("failure")
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    source = collector.Sources(None)
    result = source.shares(["BAD"] + [f"X{i}" for i in range(9)], dt.date(2026, 11, 23))
    assert len(result) == 9
    assert source.errors == [{"source": "yahoo", "symbol": "BAD", "error": "OSError"}]


def test_yahoo_more_than_ten_percent_failures_refuses_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class Ticker:
        def get_shares_full(self, **kwargs: object) -> object:
            raise OSError("failure")

    monkeypatch.setattr(collector.yf, "Ticker", lambda symbol: Ticker())
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="10%"):
        collector.Sources(None).shares(["AAA", "BBB"], dt.date(2026, 11, 23))


@pytest.mark.db
@pytest.mark.integration
def test_main_commits_using_a_real_unentered_connection(
    conn: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg2

    import ggTrader.utils.config

    real_connect = psycopg2.connect
    day = dt.date(2026, 11, 23)
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda url: real_connect(
            host=conn.info.host,
            port=conn.info.port,
            user=conn.info.user,
            password=conn.info.password,
            dbname=conn.info.dbname,
        ),
    )
    monkeypatch.setattr(ggTrader.utils.config, "get_db_connection_string", lambda: "unused")
    monkeypatch.setattr(collector, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(collector, "observation_day", lambda: day)
    monkeypatch.setattr(collector, "last_closed_day", lambda now: day)

    class FakeMainSource(Source):
        def __init__(self, path: object) -> None:
            super().__init__()
            self.requests = {"listing": 0, "alpaca": 0, "yahoo": 0}
            self.errors = []

    monkeypatch.setattr(collector, "Sources", FakeMainSource)
    monkeypatch.setattr("sys.argv", ["forward_collect.py"])
    collector.main()
    assert "2026-11-23" in capsys.readouterr().out
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 1


@pytest.mark.db
@pytest.mark.integration
def test_coverage_gate_counts_candidates_not_previously_tracked_symbols(conn: object) -> None:
    source = Source()
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.listed = ["AAA"]
    source.returned = ["AAA"]
    source.fillers = []
    result = collector.collect(dt.date(2026, 11, 17), False, source, conn)
    assert result["alert"] is False
    assert result["gaps"] == 9


def test_http_server_and_connection_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class Response:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers = {}

        def raise_for_status(self) -> None:
            assert self.status_code == 200

    calls = []
    delays = []

    def request(*args: object, **kwargs: object) -> Response:
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("test")
        return Response(501 if len(calls) == 2 else 200)

    monkeypatch.setattr(collector.requests, "get", request)
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: delays.append(seconds))
    source = collector.Sources(None)
    source.get("https://example.invalid", kind="alpaca")
    assert source.requests["alpaca"] == 3
    assert delays == [0.5, 10.0, 0.5, 20.0, 0.5]


def test_raw_listing_files_are_kept_outside_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = dt.date(2026, 11, 23)
    nasdaq = "Symbol|Security Name|ETF|Test Issue\nNA|NA Common Stock|N|N\n"
    other = "ACT Symbol|Security Name|ETF|Test Issue|Exchange\n"

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(collector, "observation_day", lambda: day)
    source = collector.Sources(tmp_path / "results" / f"{day}.json")
    monkeypatch.setattr(
        source, "get", lambda url, **kwargs: Response(nasdaq if "nasdaqlisted" in url else other)
    )
    assert source.listing()[0]["symbol"] == "NA"
    assert (tmp_path / "data" / "forward_raw" / str(day) / "nasdaqlisted.txt").read_text() == nasdaq


@pytest.mark.db
@pytest.mark.integration
def test_main_shares_mode_commits_without_nested_connection(
    conn: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg2

    import ggTrader.utils.config

    day = dt.date(2026, 11, 23)
    collector.collect(day, False, Source(), conn)
    connect = psycopg2.connect
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda url: connect(
            host=conn.info.host,
            port=conn.info.port,
            user=conn.info.user,
            password=conn.info.password,
            dbname=conn.info.dbname,
        ),
    )
    monkeypatch.setattr(ggTrader.utils.config, "get_db_connection_string", lambda: "unused")
    monkeypatch.setattr(collector, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(collector, "observation_day", lambda: day)

    class FakeSource(Source):
        def __init__(self, path: object) -> None:
            super().__init__()
            self.requests = {"listing": 0, "alpaca": 0, "yahoo": 0}
            self.errors = []

    monkeypatch.setattr(collector, "Sources", FakeSource)
    monkeypatch.setattr("sys.argv", ["forward_collect.py", "--shares-only"])
    collector.main()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.shares_outstanding")
        assert cur.fetchone()[0] == 10


@pytest.mark.db
@pytest.mark.integration
def test_low_coverage_discards_cached_partial_bars_for_retry(conn: object, tmp_path: Path) -> None:
    class Partial(Source):
        def __init__(self) -> None:
            super().__init__()
            self.fillers = []
            self.returned = []
            self.checkpoint = tmp_path / "2026-11-23.json"
            self.cached = self.checkpoint.with_name(self.checkpoint.stem + "_bars.json")
            self.cached.write_text('{"incomplete": true}')

        def invalidate_bars(self) -> None:
            self.cached.write_text("{}")

    source = Partial()
    with pytest.raises(ValueError, match="90%"):
        collector.collect(dt.date(2026, 11, 23), False, source, conn)
    assert source.cached.read_text() == "{}"


def test_shares_drop_nan_and_reject_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            values = [12.0, float("nan")] if self.symbol == "AAA" else [-1.0, float("nan")]
            return pd.Series(values, index=pd.to_datetime(["2026-11-20", "2026-11-21"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    result = collector.Sources(None).shares(["AAA", "BBB"], dt.date(2026, 11, 23))
    assert [(row["symbol"], row["shares"]) for row in result] == [("AAA", 12)]


@pytest.mark.db
@pytest.mark.integration
def test_conflicting_duplicate_bars_refuse_load(conn: object) -> None:
    class Duplicates(Source):
        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            rows = super().bars(symbols, day)
            rows.append({**rows[0], "close": 1.5})
            return rows

    with pytest.raises(ValueError, match="conflicting"):
        collector.collect(dt.date(2026, 11, 23), False, Duplicates(), conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads")
        assert cur.fetchone()[0] == 0


def test_last_closed_day_waits_for_sip_delay() -> None:
    now = dt.datetime(2026, 11, 25, 21, 5, tzinfo=dt.timezone.utc)
    assert collector.last_closed_day(now) == dt.date(2026, 11, 24)


@pytest.mark.db
@pytest.mark.integration
def test_transaction_rolls_back_after_first_bar_batch(conn: object) -> None:
    import psycopg2
    from psycopg2 import sql

    class LargeSource(Source):
        def __init__(self) -> None:
            super().__init__()
            self.listed = [f"SYM{i}" for i in range(1001)]
            self.returned = self.listed[:]
            self.fillers = []

        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            rows = super().bars(symbols, day)
            rows[-1]["volume"] = -1
            return rows

    with pytest.raises(psycopg2.errors.CheckViolation):
        collector.collect(dt.date(2026, 11, 23), False, LargeSource(), conn)
    with conn.cursor() as cur:
        for table in (
            "loads",
            "listing_snapshots",
            "daily_bars",
            "gaps",
            "symbol_status",
            "shares_outstanding",
        ):
            cur.execute(sql.SQL("SELECT count(*) FROM forward.{}").format(sql.Identifier(table)))
            assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
def test_never_traded_symbol_stops_after_five_missing_sessions(conn: object) -> None:
    source = Source()
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.listed = ["AAA"]
    for day in [17, 18, 19, 20, 23]:
        collector.collect(dt.date(2026, 11, day), False, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT day, reason FROM forward.gaps WHERE symbol='BBB' ORDER BY day")
        rows = cur.fetchall()
        assert len(rows) == 5
        assert rows[-1] == (dt.date(2026, 11, 20), "never_traded")
        cur.execute("SELECT count(*) FROM forward.symbol_status WHERE symbol='BBB'")
        assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
def test_main_exits_nonzero_after_catch_up_failures(
    conn: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg2

    import ggTrader.utils.config

    real_connect = psycopg2.connect
    day = dt.date(2026, 11, 23)
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda url: real_connect(
            host=conn.info.host,
            port=conn.info.port,
            user=conn.info.user,
            password=conn.info.password,
            dbname=conn.info.dbname,
        ),
    )
    monkeypatch.setattr(ggTrader.utils.config, "get_db_connection_string", lambda: "unused")
    monkeypatch.setattr(collector, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(collector, "observation_day", lambda: day)
    monkeypatch.setattr(collector, "last_closed_day", lambda now: day)
    monkeypatch.setattr(
        collector,
        "catch_up",
        lambda *args, **kwargs: {
            "collected": [day],
            "remaining": 0,
            "failures": [{"day": day, "error": "failed"}],
        },
    )
    monkeypatch.setattr("sys.argv", ["forward_collect.py"])
    with pytest.raises(SystemExit) as exc:
        collector.main()
    assert exc.value.code == 1


@pytest.mark.db
@pytest.mark.integration
def test_git_failure_does_not_discard_shares(conn: object, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Source()
    day = dt.date(2026, 11, 23)
    collector.collect(day, False, source, conn)

    def fail(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(collector.subprocess, "run", fail)
    result = collector.collect_shares(day, source, conn)
    assert result["shares"] == 10
    with conn.cursor() as cur:
        cur.execute("SELECT code_commit FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchone()[0] == "unknown"


@pytest.mark.db
@pytest.mark.integration
def test_today_first_keeps_earliest_first_seen_after_catch_up(conn: object) -> None:
    baseline = Source()
    collector.collect(dt.date(2026, 11, 16), False, baseline, conn)

    class NewStock(Source):
        def __init__(self) -> None:
            super().__init__()
            self.listed = ["AAA", "BBB", "CCC"]
            self.returned = ["AAA", "BBB", "CCC"]

        def saved_listing(self, day: dt.date) -> list[dict] | None:
            return self.listing() if day == dt.date(2026, 11, 17) else None

    source = NewStock()
    collector.catch_up(dt.date(2026, 11, 18), lambda day: source, conn)
    collector.collect(dt.date(2026, 11, 19), False, source, conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT first_seen FROM forward.symbol_status WHERE symbol='CCC' "
            "ORDER BY status_id DESC LIMIT 1"
        )
        assert cur.fetchone()[0] == dt.date(2026, 11, 17)


def test_exhausted_rate_limit_stops_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    class YFRateLimitError(Exception):
        pass

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            if self.symbol == "BAD":
                raise YFRateLimitError()
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    source = collector.Sources(None)
    with pytest.raises(RuntimeError, match="rate limit persisted"):
        source.shares(["BAD"] + [f"X{i}" for i in range(9)], dt.date(2026, 11, 23))
    assert source.errors[0]["error"] == "YFRateLimitError"


def test_legacy_listing_checkpoint_is_rebuilt_from_raw_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    day = dt.date(2026, 11, 23)
    source = collector.Sources(tmp_path / f"{day}.json")
    nasdaq = "Symbol|Security Name|ETF|Test Issue\nNA|NA Common Stock|N|N\nETF1|Fund|Y|N\n"
    other = "ACT Symbol|Security Name|ETF|Test Issue|Exchange\n"
    source.checkpoint.with_name(f"{day}_listing.json").write_text(
        json.dumps({"day": str(day), "raw": [nasdaq, other], "listing": [{"symbol": "NA"}]})
    )
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    rows = source.saved_listing(day)
    assert {row["symbol"] for row in rows} == {"NA", "ETF1"}
    assert (tmp_path / "data" / "forward_raw" / str(day) / "nasdaqlisted.txt").read_text() == nasdaq


@pytest.mark.db
@pytest.mark.integration
def test_shares_retry_failed_symbols_later_in_same_week(conn: object) -> None:
    class Partial(Source):
        def __init__(self) -> None:
            super().__init__()
            self.errors = []
            self.calls_by_symbol = []
            self.first = True

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.calls_by_symbol.append(symbols[:])
            if self.first:
                self.first = False
                self.errors.append({"source": "yahoo", "symbol": "BBB", "error": "OSError"})
                return super().shares([x for x in symbols if x != "BBB"], day)
            return super().shares(symbols, day)

    source = Partial()
    monday, tuesday = dt.date(2026, 11, 23), dt.date(2026, 11, 24)
    collector.collect(monday, False, source, conn)
    first = collector.collect_shares(monday, source, conn)
    assert first["shares"] == 9 and first["remaining"] == 1
    second = collector.collect_shares(tuesday, source, conn)
    assert second["shares"] == 1 and second["remaining"] == 0
    assert source.calls_by_symbol[-1] == ["BBB"]
    complete = collector.collect_shares(tuesday, source, conn)
    assert complete["already_collected"] and "complete" in complete["message"].lower()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*),count(DISTINCT symbol) FROM forward.shares_outstanding")
        assert cur.fetchone() == (10, 10)
        cur.execute("SELECT count(*) FROM forward.gaps WHERE reason='shares_failed'")
        assert cur.fetchone()[0] == 1


@pytest.mark.db
@pytest.mark.integration
def test_conflicting_duplicate_invalidates_bar_checkpoint(conn: object) -> None:
    class Duplicate(Source):
        def __init__(self) -> None:
            super().__init__()
            self.invalidated = False

        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            rows = super().bars(symbols, day)
            return rows + [{**rows[0], "close": 9}]

        def invalidate_bars(self) -> None:
            self.invalidated = True

    source = Duplicate()
    with pytest.raises(ValueError, match="conflicting"):
        collector.collect(dt.date(2026, 11, 23), False, source, conn)
    assert source.invalidated


def test_advisory_lock_is_bounded_and_has_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Cursor:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def execute(self, query: str, params: object) -> None:
            assert "pg_try_advisory_lock" in query

        def fetchone(self) -> tuple[bool]:
            return (False,)

    class Conn:
        def cursor(self) -> Cursor:
            return Cursor()

    sleeps = []
    monkeypatch.setattr(collector.time, "sleep", lambda n: sleeps.append(n))
    with pytest.raises(TimeoutError, match="advisory lock"):
        collector.acquire_lock(Conn())
    assert len(sleeps) == 4


def test_unlock_failure_is_logged_without_masking_original(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Conn:
        def cursor(self) -> object:
            raise OSError("broken connection")

    try:
        raise ValueError("original")
    except ValueError:
        collector.release_lock(Conn())
        assert "unlock" in caplog.text.lower()


def test_duplicate_symbol_uses_otherlisted_row(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

    nasdaq = "Symbol|Security Name|ETF|Test Issue\nNA|Nasdaq Common Stock|N|N\n"
    other = "ACT Symbol|Security Name|ETF|Test Issue|Exchange\nNA|Other Common Stock|N|N|N\n"
    source = collector.Sources(None)
    monkeypatch.setattr(
        source, "get", lambda url, **kwargs: Response(nasdaq if "nasdaqlisted" in url else other)
    )
    rows = source.listing()
    assert len(rows) == 1 and rows[0]["security_name"] == "Other Common Stock"


@pytest.mark.db
@pytest.mark.integration
def test_weekly_checkpoint_resumes_same_universe_after_failed_run(
    conn: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    monday, tuesday = dt.date(2026, 11, 23), dt.date(2026, 11, 24)
    listing = Source()
    collector.collect(monday, False, listing, conn)
    failing = [True]
    calls = []

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            calls.append(self.symbol)
            if failing[0] and self.symbol in {"AAA", "BBB"}:
                raise OSError("offline")
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    monkeypatch.setattr(collector, "observation_day", lambda: tuesday)
    checkpoint = tmp_path / "week.json"
    with pytest.raises(RuntimeError, match="10%"):
        collector.collect_shares(monday, collector.Sources(checkpoint), conn)
    listing.listed.append("CCC")
    listing.returned.append("CCC")
    collector.collect(tuesday, False, listing, conn)
    failing[0] = False
    calls.clear()
    result = collector.collect_shares(tuesday, collector.Sources(checkpoint), conn)
    assert result["shares"] == 10
    assert set(calls) == set(listing.listed + listing.fillers) - {"CCC"}


def test_week_key_forbids_backdating_shares() -> None:
    assert collector.shares_week(dt.date(2026, 11, 24), dt.date(2026, 11, 26)) == dt.date(
        2026, 11, 23
    )
    with pytest.raises(ValueError, match="current week"):
        collector.shares_week(dt.date(2026, 11, 16), dt.date(2026, 11, 26))


@pytest.mark.db
@pytest.mark.integration
def test_shares_failure_is_not_counted_as_missing_price_session(conn: object) -> None:
    source = Source()
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 16), False, source, conn)
    source.returned = ["BBB"]
    for day in [17, 18, 19]:
        collector.collect(dt.date(2026, 11, day), False, source, conn)
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 20), False, source, conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO forward.loads (day,source,raw_sha256,normalized_sha256,code_commit,actor) "
            "VALUES ('2026-11-16','yahoo-shares',%s,%s,'x','test') RETURNING load_id",
            ("a" * 64, "b" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.gaps VALUES (%s,'2026-11-20','AAA','shares_failed')", (load,)
        )
    source.returned = ["BBB"]
    assert collector.collect(dt.date(2026, 11, 23), False, source, conn)["newly_ended"] == 0


@pytest.mark.db
@pytest.mark.integration
def test_never_traded_symbol_remains_requested_if_still_listed(conn: object) -> None:
    class Recording(Source):
        def bars(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.requested = symbols[:]
            return super().bars(symbols, day)

    source = Recording()
    for day in [16, 17, 18, 19, 20, 23]:
        collector.collect(dt.date(2026, 11, day), False, source, conn)
    assert "BBB" in source.requested
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM forward.gaps WHERE symbol='BBB' AND reason='never_traded'"
        )
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM forward.gaps WHERE symbol='BBB' AND day='2026-11-23'")
        assert cur.fetchone()[0] == 0


@pytest.mark.db
@pytest.mark.integration
def test_clean_unavailable_shares_are_not_retried(conn: object) -> None:
    class NoData(Source):
        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.calls += 1
            self.errors = []
            return super().shares([x for x in symbols if x != "BBB"], day)

    source = NoData()
    monday = dt.date(2026, 11, 23)
    collector.collect(monday, False, source, conn)
    assert collector.collect_shares(monday, source, conn)["shares"] == 9
    assert collector.collect_shares(dt.date(2026, 11, 24), source, conn)["already_collected"]
    with conn.cursor() as cur:
        cur.execute("SELECT reason FROM forward.gaps WHERE symbol='BBB' AND reason LIKE 'shares_%'")
        assert cur.fetchall() == [("shares_unavailable",)]


@pytest.mark.db
@pytest.mark.integration
def test_empty_first_shares_writes_no_load(conn: object) -> None:
    class Empty(Source):
        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            return []

    source = Empty()
    day = dt.date(2026, 11, 23)
    collector.collect(day, False, source, conn)
    with pytest.raises(ValueError, match="no shares"):
        collector.collect_shares(day, source, conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchone()[0] == 0


def test_retry_subset_does_not_fail_ten_percent_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            if self.symbol == "BAD":
                raise OSError("unavailable")
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    source = collector.Sources(None)
    source.share_is_retry = True
    rows = source.shares(["BAD", "AAA"], dt.date(2026, 11, 23))
    assert [row["symbol"] for row in rows] == ["AAA"]


@pytest.mark.db
@pytest.mark.integration
def test_never_traded_still_listed_can_start_trading(conn: object) -> None:
    source = Source()
    for day in [16, 17, 18, 19, 20]:
        collector.collect(dt.date(2026, 11, day), False, source, conn)
    source.returned = ["AAA", "BBB"]
    collector.collect(dt.date(2026, 11, 23), False, source, conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM forward.daily_bars WHERE symbol='BBB' AND day='2026-11-23'"
        )
        assert cur.fetchone()[0] == 1


@pytest.mark.db
@pytest.mark.integration
def test_retry_subset_commits_recoveries_and_zero_retry_writes_nothing(
    conn: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    class Many(Source):
        def __init__(self) -> None:
            super().__init__()
            self.listed = [f"S{i:02}" for i in range(50)]
            self.returned = self.listed[:]
            self.fillers = []

    day = dt.date(2026, 11, 23)
    collector.collect(day, False, Many(), conn)
    first = [True]

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_shares_full(self, **kwargs: object) -> object:
            if self.symbol == "S00" or (first[0] and self.symbol in {"S01", "S02", "S03", "S04"}):
                raise OSError("offline")
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"]))

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    checkpoint = tmp_path / "shares.json"
    initial = collector.collect_shares(day, collector.Sources(checkpoint), conn)
    assert initial["shares"] == 45 and initial["remaining"] == 5
    first[0] = False
    retry = collector.collect_shares(dt.date(2026, 11, 24), collector.Sources(checkpoint), conn)
    assert retry["shares"] == 4 and retry["remaining"] == 1
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE source='yahoo-shares'")
        before = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM forward.gaps WHERE reason='shares_failed'")
        before_gaps = cur.fetchone()[0]
        assert before_gaps == 5
    result = collector.collect_shares(dt.date(2026, 11, 25), collector.Sources(checkpoint), conn)
    assert result["shares"] == 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchone()[0] == before
        cur.execute("SELECT count(*) FROM forward.gaps WHERE reason='shares_failed'")
        assert cur.fetchone()[0] == before_gaps


def test_listing_failure_does_not_save_partial_raw_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = dt.date(2026, 11, 23)
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(collector, "observation_day", lambda: day)
    source = collector.Sources(tmp_path / "results" / f"{day}.json")
    calls = []

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

    def fetch(url: str, **kwargs: object) -> Response:
        calls.append(url)
        if len(calls) == 2:
            raise OSError("second file failed")
        if "nasdaqlisted" in url:
            return Response("Symbol|Security Name|ETF|Test Issue\nAAA|A Common Stock|N|N\n")
        return Response("ACT Symbol|Security Name|ETF|Test Issue|Exchange\n")

    monkeypatch.setattr(source, "get", fetch)
    with pytest.raises(OSError):
        source.listing()
    assert not (tmp_path / "data" / "forward_raw" / str(day)).exists()
    assert source.listing()[0]["symbol"] == "AAA"


def test_saved_listing_falls_back_to_durable_raw_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    day = dt.date(2026, 11, 23)
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    folder = tmp_path / "data" / "forward_raw" / str(day)
    folder.mkdir(parents=True)
    (folder / "nasdaqlisted.txt").write_text(
        "Symbol|Security Name|ETF|Test Issue\nNA|A Common Stock|N|N\n"
    )
    (folder / "otherlisted.txt").write_text("ACT Symbol|Security Name|ETF|Test Issue|Exchange\n")
    source = collector.Sources(tmp_path / "results" / f"{day}.json")
    assert source.saved_listing(day)[0]["symbol"] == "NA"


@pytest.mark.db
@pytest.mark.integration
def test_daily_and_shares_advisory_locks_are_independent(
    conn: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg2

    other = psycopg2.connect(
        host=conn.info.host,
        port=conn.info.port,
        user=conn.info.user,
        password=conn.info.password,
        dbname=conn.info.dbname,
    )
    other.autocommit = True
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    try:
        collector.acquire_lock(conn, "daily")
        collector.acquire_lock(other, "shares")
        with pytest.raises(TimeoutError):
            collector.acquire_lock(other, "daily")
        collector.release_lock(other, "shares")
        collector.release_lock(conn, "daily")
    finally:
        other.close()


def test_closed_connection_does_not_mask_write_failure() -> None:
    class Conn:
        _autocommit = True
        closed = False

        @property
        def autocommit(self) -> bool:
            return self._autocommit

        @autocommit.setter
        def autocommit(self, value: bool) -> None:
            if self.closed:
                raise AssertionError("must not restore autocommit on a closed connection")
            self._autocommit = value

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            raise AssertionError("must not roll back a closed connection")

    conn = Conn()
    with pytest.raises(ValueError, match="original"):
        with collector.write_transaction(conn):
            conn.closed = True
            raise ValueError("original")


@pytest.mark.db
@pytest.mark.integration
def test_unavailable_only_retry_resolves_failed_without_retrying_again(conn: object) -> None:
    class SourceWithFailure(Source):
        def __init__(self) -> None:
            super().__init__()
            self.errors = []
            self.stage = 0

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            if self.stage == 0:
                self.stage = 1
                self.errors.append({"source": "yahoo", "symbol": "BBB", "error": "OSError"})
                return super().shares([x for x in symbols if x != "BBB"], day)
            return []

    day = dt.date(2026, 11, 23)
    source = SourceWithFailure()
    collector.collect(day, False, source, conn)
    collector.collect_shares(day, source, conn)
    result = collector.collect_shares(dt.date(2026, 11, 24), source, conn)
    assert result["shares"] == 0 and result["unavailable"] == 1
    assert collector.collect_shares(dt.date(2026, 11, 25), source, conn)["already_collected"]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT reason FROM forward.gaps WHERE symbol='BBB' "
            "AND reason LIKE 'shares_%' ORDER BY load_id"
        )
        assert cur.fetchall() == [("shares_failed",), ("shares_unavailable",)]
        cur.execute("SELECT count(*) FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchone()[0] == 2


@pytest.mark.db
@pytest.mark.integration
def test_retry_resolving_nothing_adds_no_load(conn: object) -> None:
    class AlwaysFailed(Source):
        def __init__(self) -> None:
            super().__init__()
            self.errors = []

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.errors.append({"source": "yahoo", "symbol": "BBB", "error": "OSError"})
            return super().shares([x for x in symbols if x != "BBB"], day)

    day = dt.date(2026, 11, 23)
    source = AlwaysFailed()
    collector.collect(day, False, source, conn)
    collector.collect_shares(day, source, conn)
    result = collector.collect_shares(dt.date(2026, 11, 24), source, conn)
    assert result["shares"] == 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchone()[0] == 1


@pytest.mark.db
@pytest.mark.integration
def test_retry_with_unknown_listing_date_uses_null(conn: object) -> None:
    class Partial(Source):
        def __init__(self) -> None:
            super().__init__()
            self.errors = []

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            self.errors.append({"source": "yahoo", "symbol": "BBB", "error": "OSError"})
            return super().shares([x for x in symbols if x != "BBB"], day)

    class Retry(Source):
        checkpoint = None

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            assert self.share_listing_day is None
            return super().shares(symbols, day)

    day = dt.date(2026, 11, 23)
    source = Partial()
    collector.collect(day, False, source, conn)
    collector.collect_shares(day, source, conn)
    assert collector.collect_shares(dt.date(2026, 11, 24), Retry(), conn)["shares"] == 1


def test_listing_is_not_dated_before_second_file_was_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = iter([dt.date(2026, 11, 23), dt.date(2026, 11, 24)])
    monkeypatch.setattr(collector, "observation_day", lambda: next(days))

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

    source = collector.Sources(None)
    monkeypatch.setattr(
        source,
        "get",
        lambda url, **kwargs: Response(
            "Symbol|Security Name|ETF|Test Issue\nAAA|A Common Stock|N|N\n"
            if "nasdaqlisted" in url
            else "ACT Symbol|Security Name|ETF|Test Issue|Exchange\n"
        ),
    )
    with pytest.raises(ValueError, match="date changed"):
        source.listing()


def test_real_yfinance_connection_exception_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    import yfinance.base
    from yfinance.data import YfData

    collector.yf.config.debug.hide_exceptions = True
    monkeypatch.setattr(yfinance.base.TickerBase, "_get_ticker_tz", lambda *args, **kwargs: "UTC")

    def fail(*args: object, **kwargs: object) -> None:
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(YfData, "cache_get", fail)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    source = collector.Sources(None)
    with pytest.raises(RuntimeError, match="exceed 10%"):
        source.shares(["AAA"], dt.date(2026, 11, 23))
    assert source.errors[0]["error"] == "ConnectionError"


@pytest.mark.db
@pytest.mark.integration
def test_empty_first_shares_retry_refetches(
    conn: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    day = dt.date(2026, 11, 23)
    collector.collect(day, False, Source(), conn)
    available = [False]

    class Ticker:
        def __init__(self, symbol: str) -> None:
            pass

        def get_shares_full(self, **kwargs: object) -> object:
            return pd.Series([100], index=pd.to_datetime(["2026-11-20"])) if available[0] else None

    monkeypatch.setattr(collector.yf, "Ticker", Ticker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    checkpoint = tmp_path / "week.json"
    with pytest.raises(ValueError, match="no shares"):
        collector.collect_shares(day, collector.Sources(checkpoint), conn)
    assert not checkpoint.exists()
    available[0] = True
    assert (
        collector.collect_shares(day + dt.timedelta(days=1), collector.Sources(checkpoint), conn)[
            "shares"
        ]
        == 10
    )


@pytest.mark.db
@pytest.mark.integration
def test_holiday_monday_week_shares_retry_on_wednesday(conn: object) -> None:
    class Partial(Source):
        def __init__(self) -> None:
            super().__init__()
            self.errors = []
            self.first = True

        def shares(self, symbols: list[str], day: dt.date) -> list[dict]:
            if self.first:
                self.first = False
                self.errors.append({"source": "yahoo", "symbol": "BBB", "error": "OSError"})
                return super().shares([x for x in symbols if x != "BBB"], day)
            return super().shares(symbols, day)

    source = Partial()
    tuesday = dt.date(2026, 9, 8)
    collector.collect(tuesday, False, source, conn)
    assert collector.collect_shares(tuesday, source, conn)["remaining"] == 1
    assert collector.collect_shares(dt.date(2026, 9, 9), source, conn)["shares"] == 1
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT day FROM forward.loads WHERE source='yahoo-shares'")
        assert cur.fetchall() == [(dt.date(2026, 9, 7),)]
