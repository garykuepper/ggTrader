"""Fast tests of ``ggt db membership check-upstream`` with the network and database replaced.

Every HTTP call goes through a fake ``get``; the store is an in-memory fake. Nothing here opens a
socket or a database connection (the one test of the real store's SQL uses a recording stub).
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest
import requests

from ggTrader.cli import cmd_membership
from ggTrader.data import membership_upstream as up
from ggTrader.data.membership_upstream import (
    CheckResult,
    PgCheckStore,
    UpstreamFetchError,
    UpstreamFile,
    check_upstream,
    compare_with_latest,
    fetch_upstream,
    raw_url,
)
from ggTrader.data.universe_algebra import History, collapse, content_sha256, raw_sha256
from ggTrader.data.universe_db import VintageData
from tests.data.membership_fakes import synthetic_history

COMMIT = "a" * 40
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)


def csv_bytes(history: History) -> bytes:
    """A ``date,tickers`` file in upstream's quoting."""
    lines = ["date,tickers"] + [f'{d.isoformat()},"{",".join(t)}"' for d, t in history.items()]
    return ("\n".join(lines) + "\n").encode()


def vintage_data(history: History, *, raw: bytes | None = None, load_id: int = 1) -> VintageData:
    """The latest vintage as the store returns it, for ``history``."""
    days = sorted(history)
    load = {
        "load_id": load_id,
        "coverage_start": days[0],
        "coverage_end": days[-1],
        "raw_sha256": raw_sha256(raw if raw is not None else csv_bytes(history)),
        "content_sha256": content_sha256(history),
    }
    rows = list(enumerate(collapse(history), start=1))
    return VintageData(load, {d: len(history[d]) for d in days}, rows)


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        payload: Any = None,
    ) -> None:
        self.status_code = status
        self.content = body
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._payload = payload
        self.served = 0  # bytes handed out by iter_content
        self.closed = False

    def iter_content(self, chunk_size: int = 1) -> Any:
        for start in range(0, len(self.content), chunk_size):
            chunk = self.content[start : start + chunk_size]
            self.served += len(chunk)
            yield chunk

    def close(self) -> None:
        self.closed = True

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeGet:
    """Stands in for ``requests.get``: one commits reply, one raw-file reply, a call log."""

    def __init__(
        self, raw: FakeResponse | Exception, commits: FakeResponse | Exception | None = None
    ):
        self.raw = raw
        self.commits = commits or FakeResponse(payload=[{"sha": COMMIT}])
        self.calls: list[str] = []

    def __call__(self, url: str, **_: Any) -> FakeResponse:
        self.calls.append(url)
        reply = self.commits if url == up.COMMITS_URL else self.raw
        if isinstance(reply, Exception):
            raise reply
        return reply


def good(body: bytes, **extra: str) -> FakeResponse:
    headers = {"content-type": "text/plain; charset=utf-8", "content-length": str(len(body))}
    headers.update({k.lower(): v for k, v in extra.items()})
    return FakeResponse(200, body, headers)


class FakeStore:
    def __init__(self, latest: VintageData | None, fail_record: bool = False) -> None:
        self._latest = latest
        self.fail_record = fail_record
        self.recorded: list[CheckResult] = []

    def latest(self, index_name: str) -> VintageData | None:
        return self._latest

    def record(self, index_name: str, result: CheckResult) -> int:
        if self.fail_record:
            raise RuntimeError("postgres://user:secret@host/db is down")
        self.recorded.append(result)
        return len(self.recorded)


def upstream_file(raw: bytes) -> UpstreamFile:
    return UpstreamFile(raw, raw_url(COMMIT), COMMIT, '"etag"', NOW)


# --------------------------------------------------------------------------- fetch


def test_a_good_download_is_returned_with_its_commit_and_etag():
    raw = csv_bytes(synthetic_history())
    get = FakeGet(good(raw, ETag='"abc"'))
    fetched = fetch_upstream(get=get, now=lambda: NOW)
    assert fetched.raw == raw and fetched.commit == COMMIT and fetched.etag == '"abc"'
    assert fetched.url == raw_url(COMMIT) and fetched.retrieved_at == NOW
    assert get.calls == [up.COMMITS_URL, raw_url(COMMIT)]  # the bytes are read at that commit
    assert "S%26P%20500%20Historical" in fetched.url


def test_the_default_getter_is_looked_up_at_call_time(monkeypatch):
    raw = csv_bytes(synthetic_history())
    monkeypatch.setattr(up.requests, "get", FakeGet(good(raw)))
    assert fetch_upstream(now=lambda: NOW).raw == raw


@pytest.mark.parametrize(
    ("raw_reply", "commits_reply", "reason"),
    [
        (requests.ConnectionError("no route"), None, "unreachable"),
        (good(b""), requests.Timeout("slow"), "unreachable"),
        (FakeResponse(404, b"Not Found"), None, "http_status"),
        (good(b""), FakeResponse(403, b"rate limited"), "http_status"),
        (good(b""), FakeResponse(200, payload=[]), "invalid"),
        (good(b""), FakeResponse(200, payload=[{"sha": "not-a-sha"}]), "invalid"),
        (
            FakeResponse(200, b"<html>login</html>", {"content-type": "text/html"}),
            None,
            "content_type",
        ),
        (FakeResponse(200, b"x", {}), None, "content_type"),
        (
            FakeResponse(200, b"date,tickers\n", {"content-type": "text/plain"}),
            None,
            "invalid",
        ),  # header only: no snapshot rows
    ],
)
def test_an_unusable_download_is_a_typed_failure(raw_reply, commits_reply, reason):
    with pytest.raises(UpstreamFetchError) as caught:
        fetch_upstream(get=FakeGet(raw_reply, commits_reply), now=lambda: NOW)
    assert caught.value.reason == reason


def test_a_truncated_download_is_refused():
    raw = csv_bytes(synthetic_history())
    short = raw[: len(raw) // 2]
    # the server promised the full length but the body stopped early
    cut = FakeResponse(200, short, {"content-type": "text/plain", "content-length": str(len(raw))})
    with pytest.raises(UpstreamFetchError) as caught:
        fetch_upstream(get=FakeGet(cut), now=lambda: NOW)
    assert caught.value.reason == "truncated"
    # no length header at all, and the cut lands mid-line: no trailing newline
    no_newline = FakeResponse(200, raw[:-20], {"content-type": "text/plain"})
    with pytest.raises(UpstreamFetchError) as caught:
        fetch_upstream(get=FakeGet(no_newline), now=lambda: NOW)
    assert caught.value.reason == "truncated"


def test_a_compressed_body_is_not_judged_by_content_length():
    raw = csv_bytes(synthetic_history())
    reply = good(raw, **{"content-length": "12345", "content-encoding": "gzip"})
    assert fetch_upstream(get=FakeGet(reply), now=lambda: NOW).raw == raw


def test_an_oversized_upstream_body_is_refused_while_streaming():
    big = FakeResponse(
        200, b"x" * (up.MAX_DOWNLOAD_BYTES + 5 * 65536), {"content-type": "text/plain"}
    )
    with pytest.raises(UpstreamFetchError) as caught:
        fetch_upstream(get=FakeGet(big), now=lambda: NOW)
    assert caught.value.reason == "too_large"
    assert big.served <= up.MAX_DOWNLOAD_BYTES + 65536 and big.closed


# --------------------------------------------------------------------------- compare


def test_equal_bytes_are_unchanged_and_match_the_latest_load():
    history = synthetic_history()
    raw = csv_bytes(history)
    result = compare_with_latest(upstream_file(raw), vintage_data(history))
    assert (result.outcome, result.result, result.matched_load_id) == ("unchanged", "equal", 1)
    assert result.raw_sha256 == hashlib.sha256(raw).hexdigest() and result.exit_code == 0
    assert result.detail["upstream_coverage_end"] == max(history).isoformat()
    assert result.detail["compared"] == "raw_sha256"


def test_different_bytes_with_identical_content_still_count_as_unchanged():
    history = synthetic_history()
    latest = vintage_data(history, raw=b"the loaded bytes were formatted differently")
    result = compare_with_latest(upstream_file(csv_bytes(history)), latest)
    assert (result.outcome, result.result) == ("unchanged", "content_equal")
    assert result.matched_load_id == 1 and result.detail["compared"] == "content_sha256"


def test_a_differing_upstream_is_reported_and_never_matched():
    history = synthetic_history(n_dates=10)
    newer = dict(history)
    last = max(history)
    day = last + timedelta(days=30)
    members = list(history[last])
    newer[day] = tuple(sorted(members[3:] + ["NEW1", "NEW2", "NEW3"]))
    result = compare_with_latest(upstream_file(csv_bytes(newer)), vintage_data(history))
    assert (result.outcome, result.result) == ("refused", "differs")
    assert result.matched_load_id is None and result.exit_code == 2
    detail = result.detail
    assert detail["upstream_coverage_end"] == day.isoformat()
    assert detail["latest_coverage_end"] == last.isoformat()
    assert detail["dates_only_in_upstream"] == [day.isoformat()]
    assert detail["dates_only_in_latest"] == [] and detail["historical_rewrites"] == []
    assert len(detail["tail_changes"]) == 6 and "nothing was changed" in detail["action"]


def test_an_upstream_that_rewrites_old_history_lists_the_historical_rewrites():
    history = synthetic_history(n_dates=10)
    days = sorted(history)
    changed = dict(history)
    changed[days[2]] = tuple(sorted(list(history[days[2]])[1:] + ["ZZZZ"]))
    result = compare_with_latest(upstream_file(csv_bytes(changed)), vintage_data(history))
    assert result.result == "differs"
    assert result.detail["dates_changed"] and result.detail["historical_rewrites"]


def test_an_empty_database_is_a_difference_not_a_match():
    result = compare_with_latest(upstream_file(csv_bytes(synthetic_history())), None)
    assert (result.outcome, result.matched_load_id) == ("refused", None)


# --------------------------------------------------------------------------- the runner


def test_check_upstream_records_exactly_one_row_per_outcome():
    history = synthetic_history()
    equal = FakeStore(vintage_data(history))
    run = check_upstream(equal, fetch=lambda: upstream_file(csv_bytes(history)))
    assert [r.outcome for r in equal.recorded] == ["unchanged"] and run.check_id == 1

    failing = FakeStore(vintage_data(history))

    def boom() -> UpstreamFile:
        raise UpstreamFetchError("unreachable", "cannot reach fja05680/sp500: no route")

    run = check_upstream(failing, fetch=boom)
    assert [r.outcome for r in failing.recorded] == ["fetch_failed"]
    assert run.result.raw_sha256 is None and run.result.exit_code == 1
    assert run.result.detail["reason"] == "unreachable"


def test_a_row_that_cannot_be_written_is_reported_without_leaking_the_error_text():
    history = synthetic_history()
    store = FakeStore(vintage_data(history), fail_record=True)
    run = check_upstream(store, fetch=lambda: upstream_file(csv_bytes(history)))
    assert run.check_id is None and run.record_error
    assert "secret" not in run.record_error and "postgres://" not in run.record_error


class _Cursor:
    def __init__(self, log: list[tuple[str, tuple]]) -> None:
        self.log = log

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.log.append((sql, params))

    def fetchone(self) -> tuple[int]:
        return (7,)


class _Conn:
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []
        self.committed = False

    def cursor(self) -> _Cursor:
        return _Cursor(self.log)

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        return None


def test_the_real_store_only_ever_inserts_one_row_into_checks(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(up, "connect", lambda url=None, read_only=False: conn)
    result = CheckResult("unchanged", "equal", 1, "f" * 64, COMMIT, {"upstream_bytes": 5})
    assert PgCheckStore().record("sp500", result) == 7
    assert conn.committed and len(conn.log) == 1
    sql, params = conn.log[0]
    assert sql.startswith("INSERT INTO universe.checks")
    assert "universe.loads" not in sql and "universe.membership" not in sql
    assert params[0] == "sp500" and params[3] == "unchanged" and params[4] == 1
    payload = params[5].adapted
    assert payload["result"] == "equal" and payload["upstream_bytes"] == 5


# --------------------------------------------------------------------------- the command


def _args() -> argparse.Namespace:
    return argparse.Namespace(membership_command="check-upstream")


@pytest.mark.parametrize(
    ("scenario", "exit_code", "headline"),
    [("equal", 0, "EQUAL"), ("differs", 2, "DIFFERS"), ("unreachable", 1, "FETCH_FAILED")],
)
def test_the_command_prints_and_exits_by_outcome(
    monkeypatch, capsys, scenario, exit_code, headline
):
    history = synthetic_history(n_dates=10)
    store = FakeStore(vintage_data(history))
    if scenario == "equal":
        reply: FakeResponse | Exception = good(csv_bytes(history))
    elif scenario == "differs":
        newer = dict(history)
        newer[max(history) + timedelta(days=9)] = history[max(history)]
        reply = good(csv_bytes(newer))
    else:
        reply = requests.ConnectionError("no route")
    monkeypatch.setattr(up.requests, "get", FakeGet(reply))
    monkeypatch.setattr(cmd_membership, "PgCheckStore", lambda: store)
    assert cmd_membership.run_membership(_args()) == exit_code
    out = capsys.readouterr().out
    assert headline in out and len(store.recorded) == 1
    if scenario == "differs":
        assert "nothing was loaded or changed" in out and "NOT reset" in out


def test_the_command_exits_one_when_the_row_cannot_be_recorded(monkeypatch, capsys):
    history = synthetic_history()
    store = FakeStore(vintage_data(history), fail_record=True)
    monkeypatch.setattr(up.requests, "get", FakeGet(good(csv_bytes(history))))
    monkeypatch.setattr(cmd_membership, "PgCheckStore", lambda: store)
    assert cmd_membership.run_membership(_args()) == 1
    captured = capsys.readouterr()
    assert "NOT RECORDED" in captured.err and "secret" not in captured.out + captured.err


def test_the_command_is_registered_with_status_and_diff():
    from ggTrader.cli.cmd_db import register_db_parser

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="cmd"))
    for argv in (
        ["db", "membership", "check-upstream"],
        ["db", "membership", "status", "--asof", "2026-11-29"],
        ["db", "membership", "diff", "--from", "1", "--to", "2"],
    ):
        assert parser.parse_args(argv).membership_command == argv[2]
    assert parser.parse_args(["db", "membership", "status", "--asof", "2026-11-29"]).asof == date(
        2026, 11, 29
    )
