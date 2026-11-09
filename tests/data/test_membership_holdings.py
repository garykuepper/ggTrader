"""Fast, offline tests of ``ggt db membership check-holdings`` and its standard-library xlsx reader.

Every workbook is built in memory with ``zipfile`` (``tests/data/xlsx_builder.py``); every HTTP
call goes through a fake ``get``; the store is an in-memory fake. Nothing opens a socket or a
database connection (the SQL of the real store is checked with a recording stub).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest
import requests

from ggTrader.cli import cmd_membership
from ggTrader.data import membership_holdings as mh
from ggTrader.data.membership_holdings import (
    HOLDINGS_URL,
    USER_AGENT,
    HoldingsFetchError,
    HoldingsFile,
    HoldingsResult,
    PgHoldingsStore,
    check_holdings,
    compare_holdings,
    fetch_holdings,
    parse_holdings,
)
from tests.data.test_membership_upstream import FakeResponse, vintage_data
from tests.data.xlsx_builder import build_xlsx, holdings_xlsx

TODAY = date(2026, 11, 8)
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
AS_OF = date(2026, 11, 5)
CLASSES = ["BRK.B", "BF.B", "GOOG", "GOOGL"]  # the project's raw spelling, as in the list


def base_tickers(n: int = 495) -> list[str]:
    """``n`` plain tickers plus the four share-class names (a list inside the loader's range)."""
    return [f"X{i:04d}" for i in range(n)] + CLASSES


def history(latest: list[str], *, first: date = date(2026, 5, 1)):
    """Three snapshots; the last one is ``latest`` (raw tickers)."""
    older = [t for t in latest if t != latest[-1]] + ["OLD1"]
    return {
        first: tuple(sorted(older)),
        date(2026, 6, 2): tuple(sorted(older)),
        date(2026, 9, 1): tuple(sorted(latest)),
    }


def holdings_file(raw: bytes, **kwargs: Any) -> HoldingsFile:
    return HoldingsFile(
        raw=raw,
        url=HOLDINGS_URL,
        final_url=HOLDINGS_URL,
        content_type=kwargs.get("content_type", "application/vnd.ms-excel"),
        etag='"abc"',
        last_modified=None,
        retrieved_at=NOW,
    )


def compare(tickers: list[str], latest_tickers: list[str], **kwargs: Any) -> HoldingsResult:
    raw = holdings_xlsx(tickers, **kwargs)
    return compare_holdings(holdings_file(raw), vintage_data(history(latest_tickers)), today=TODAY)


# --------------------------------------------------------------------------- parsing


def test_the_parser_reads_the_date_and_keeps_only_equity_lines():
    holdings = parse_holdings(holdings_xlsx(base_tickers()))
    assert holdings.as_of == AS_OF and holdings.as_of_text == "As of 05-Nov-2026"
    assert len(holdings.tickers) == 499 and holdings.n_table_rows == 499 + 3
    # cash, a future with a dash ticker and a blank-ticker collateral line are dropped, and listed
    assert {e["name"] for e in holdings.excluded} == {
        "US DOLLAR",
        "E-MINI S&P 500 FUTURE",
        "CASH COLLATERAL",
    }
    assert "USD" not in holdings.tickers and "" not in holdings.tickers
    assert holdings.tickers == tuple(sorted(holdings.tickers))


def test_share_classes_are_normalised_to_the_project_convention():
    holdings = parse_holdings(holdings_xlsx(base_tickers()))
    assert {"BRK-B", "BF-B", "GOOG", "GOOGL"} <= set(holdings.tickers)
    assert "BRK.B" not in holdings.tickers
    spaced = parse_holdings(holdings_xlsx([t for t in base_tickers() if t != "BRK.B"] + ["BRK B"]))
    assert "BRK-B" in spaced.tickers  # the fund's other spelling


@pytest.mark.parametrize(
    ("text", "day"),
    [
        ("As of 05-Nov-2026", date(2026, 11, 5)),
        ("Holdings: As of 5-Nov-2026", date(2026, 11, 5)),
        ("AS OF 31-DEC-2026", date(2026, 12, 31)),
        ("As of 01-Jan-2027 ", date(2027, 1, 1)),
    ],
)
def test_the_as_of_text_variants(text, day):
    assert parse_holdings(holdings_xlsx(base_tickers(), text)).as_of == day


def test_the_as_of_date_may_sit_inside_a_longer_cell_and_rich_or_inline_text_is_fine():
    for kwargs in ({"rich_text": True}, {"inline": True}):
        holdings = parse_holdings(
            holdings_xlsx(base_tickers(), "Fund holdings (As of 05-Nov-2026) end of day", **kwargs)
        )
        assert holdings.as_of == AS_OF


@pytest.mark.parametrize(
    ("raw", "reason", "message"),
    [
        (b"<html>Request blocked</html>", "not_xlsx", "not a usable workbook"),
        (holdings_xlsx(base_tickers())[:-80], "not_xlsx", "not a usable workbook"),
        (build_xlsx([["Name", "Symbol"], ["a", "AAPL"]]), "invalid", "no header row"),
        (holdings_xlsx(base_tickers(), None), "invalid", "As of"),
        (holdings_xlsx(base_tickers(100)), "invalid", "outside 480..520"),
        (holdings_xlsx(base_tickers(600)), "invalid", "outside 480..520"),
        (holdings_xlsx(base_tickers() + ["X0001"]), "invalid", "duplicate tickers"),
        (holdings_xlsx(base_tickers(), doctype=True), "invalid", "DOCTYPE"),
    ],
)
def test_an_unusable_download_is_a_typed_failure_that_keeps_the_bytes(raw, reason, message):
    with pytest.raises(HoldingsFetchError, match=message) as caught:
        parse_holdings(raw)
    assert caught.value.reason == reason and caught.value.raw == raw


# --------------------------------------------------------------------------- comparing


def test_equal_sets_match_and_the_attested_date_is_the_holdings_date():
    result = compare(base_tickers(), base_tickers())
    assert (result.outcome, result.result, result.exit_code) == ("matched", "matched", 0)
    assert result.matched_load_id == 1 and result.holdings_as_of == AS_OF
    detail = result.detail
    assert detail["only_in_list"] == [] and detail["only_in_holdings"] == []
    assert (
        detail["compared_snapshot_date"] == "2026-09-01" and detail["known_through"] == "2026-11-05"
    )
    assert detail["n_holdings"] == detail["n_list"] == 499
    assert detail["raw_sha256"] == result.raw_sha256 == hashlib.sha256(result.raw).hexdigest()
    assert detail["source_url"] == HOLDINGS_URL and detail["vintage_coverage_end"] == "2026-09-01"


def test_share_class_spellings_on_either_side_still_match():
    """The list says BRK.B, the fund BRK-B (or the other way round): the same ticker."""
    hyphen = [t.replace(".", "-") for t in base_tickers()]
    assert compare(base_tickers(), hyphen).outcome == "matched"
    assert compare(hyphen, base_tickers()).outcome == "matched"


def test_one_addition_and_one_removal_is_a_mismatch_with_both_lists():
    ours = base_tickers()
    theirs = [t for t in ours if t != "X0007"] + ["NEWCO"]
    result = compare(theirs, ours)
    assert (result.outcome, result.result, result.exit_code) == ("mismatch", "mismatch", 2)
    assert result.matched_load_id is None
    assert result.detail["only_in_list"] == ["X0007"]
    assert result.detail["only_in_holdings"] == ["NEWCO"]
    assert result.detail["compared_load_id"] == 1


def test_a_cash_line_is_not_a_difference():
    assert compare(base_tickers(), base_tickers(), cash_lines=True).outcome == "matched"
    assert compare(base_tickers(), base_tickers(), cash_lines=False).outcome == "matched"


def test_the_comparison_uses_the_snapshot_at_the_holdings_date_not_the_latest():
    """Holdings dated between two snapshots are compared with the earlier one."""
    older = [t for t in base_tickers() if t != "GOOGL"] + ["OLD1"]
    data = vintage_data(history(base_tickers()))
    file = holdings_file(holdings_xlsx(older, "As of 15-Aug-2026"))
    result = compare_holdings(file, data, today=date(2026, 8, 18))  # inside the age limit
    # snapshot of 2026-06-02 is `older` apart from the last-sorted ticker swap: see history()
    assert result.detail["compared_snapshot_date"] == "2026-06-02"
    assert result.holdings_as_of == date(2026, 8, 15)
    assert result.detail["holdings_as_of_before_coverage_end"] is True


def test_holdings_older_than_the_last_change_that_match_never_move_known_through_back():
    snapshot_2 = history(base_tickers())[date(2026, 6, 2)]
    raw_list = [t for t in snapshot_2]
    result = compare_holdings(
        holdings_file(holdings_xlsx(raw_list, "As of 15-Aug-2026")),
        vintage_data(history(base_tickers())),
        today=date(2026, 8, 18),  # inside the age limit
    )
    assert result.outcome == "matched"
    assert result.detail["known_through"] == "2026-09-01"  # the vintage's own coverage end


def test_holdings_dated_before_the_list_begins_cannot_be_compared():
    result = compare_holdings(
        holdings_file(holdings_xlsx(base_tickers(), "As of 01-Jan-2026")),
        vintage_data(history(base_tickers())),
        today=date(2026, 1, 3),  # fresh enough, so the coverage rule is the one that fires
    )
    assert (result.outcome, result.exit_code) == ("fetch_failed", 1)
    assert result.detail["reason"] == "as_of_before_coverage"
    assert result.holdings_as_of == date(2026, 1, 1) and result.raw is not None


def test_holdings_dated_in_the_future_are_refused():
    result = compare(base_tickers(), base_tickers(), as_of_text="As of 09-Nov-2026")
    assert (result.outcome, result.exit_code) == ("fetch_failed", 1)
    assert result.detail["reason"] == "as_of_in_future" and result.matched_load_id is None
    assert result.raw is not None and "holdings_age_days" not in result.detail
    today = compare_holdings(
        holdings_file(holdings_xlsx(base_tickers(), "As of 08-Nov-2026")),
        vintage_data(history(base_tickers())),
        today=TODAY,
    )
    assert today.outcome == "matched"  # dated today is fine


@pytest.mark.parametrize(
    ("as_of_text", "age", "outcome"),
    [
        ("As of 02-Nov-2026", 6, "matched"),
        ("As of 01-Nov-2026", 7, "matched"),  # exactly at the limit is still accepted
        ("As of 31-Oct-2026", 8, "fetch_failed"),
    ],
)
def test_a_holdings_file_older_than_the_limit_is_a_stale_source(as_of_text, age, outcome):
    result = compare(base_tickers(), base_tickers(), as_of_text=as_of_text)
    assert result.outcome == outcome
    assert result.detail["holdings_age_days"] == age and result.detail["max_holdings_age_days"] == 7
    if outcome == "fetch_failed":
        assert result.detail["reason"] == "stale_source" and result.exit_code == 1
        assert result.matched_load_id is None and "known_through" not in result.detail
        assert result.holdings_as_of == TODAY - timedelta(days=age)
        assert result.raw is not None
        assert result.raw_sha256 == hashlib.sha256(result.raw).hexdigest()
    else:
        assert result.matched_load_id is not None and result.exit_code == 0


def test_a_stale_file_with_a_different_list_is_still_only_a_stale_source():
    result = compare(base_tickers()[:-1], base_tickers(), as_of_text="As of 20-Oct-2026")
    assert result.outcome == "fetch_failed" and result.detail["reason"] == "stale_source"
    assert "only_in_list" not in result.detail  # no comparison was made


def test_the_age_limit_can_be_overridden_by_the_environment(monkeypatch):
    assert mh.max_holdings_age_days({}) == mh.MAX_HOLDINGS_AGE_DAYS == 7
    assert mh.max_holdings_age_days({mh.HOLDINGS_AGE_ENV: "14"}) == 14
    assert mh.max_holdings_age_days({mh.HOLDINGS_AGE_ENV: "1"}) == 1
    assert mh.HOLDINGS_AGE_ENV == "GGT_HOLDINGS_MAX_AGE_DAYS"
    monkeypatch.setenv(mh.HOLDINGS_AGE_ENV, "14")
    result = compare(base_tickers(), base_tickers(), as_of_text="As of 27-Oct-2026")  # 12 days
    assert result.outcome == "matched" and result.detail["max_holdings_age_days"] == 14


@pytest.mark.parametrize("bad", ["0", "-3", "seven", "1.5", "15", "99999"])
def test_a_malformed_age_override_is_refused_before_any_download(monkeypatch, bad):
    monkeypatch.setenv(mh.HOLDINGS_AGE_ENV, bad)

    def fetch():
        raise AssertionError("must not download with a malformed override")

    with pytest.raises(mh.HoldingsConfigError, match=mh.HOLDINGS_AGE_ENV):
        check_holdings(FakeStore(), fetch=fetch, today=lambda: TODAY)


def test_a_stale_file_is_recorded_as_fetch_failed_and_its_bytes_are_archived():
    raw = holdings_xlsx(base_tickers(), "As of 25-Oct-2026")
    store = FakeStore()
    run_ = check_holdings(store, fetch=lambda: holdings_file(raw), today=lambda: TODAY)
    assert run_.result.outcome == "fetch_failed" and run_.check_id is not None
    (recorded,) = store.recorded
    assert recorded.raw == raw and recorded.detail["reason"] == "stale_source"


def test_no_vintage_loaded_is_a_mismatch_that_says_so():
    result = compare_holdings(holdings_file(holdings_xlsx(base_tickers())), None, today=TODAY)
    assert result.outcome == "mismatch" and result.matched_load_id is None
    assert "nothing to compare" in result.detail["reason"]


def test_an_unparsable_workbook_is_fetch_failed_and_keeps_the_bytes():
    result = compare_holdings(holdings_file(b"<html>nope</html>"), None, today=TODAY)
    assert result.outcome == "fetch_failed" and result.detail["reason"] == "not_xlsx"
    assert result.raw == b"<html>nope</html>"
    assert result.detail["raw_sha256"] == hashlib.sha256(b"<html>nope</html>").hexdigest()


# --------------------------------------------------------------------------- the download


class FakeGet:
    """Stands in for ``requests.get``: one reply (or error) and a log of the calls."""

    def __init__(self, reply: FakeResponse | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def ok(raw: bytes, **headers: str) -> FakeResponse:
    return FakeResponse(200, raw, {"content-length": str(len(raw)), **headers})


def test_the_download_sends_a_browser_user_agent_and_follows_redirects():
    raw = holdings_xlsx(base_tickers())
    get = FakeGet(ok(raw, etag='"e1"', **{"content-type": "application/octet-stream"}))
    file = fetch_holdings(get=get, now=lambda: NOW)
    url, kwargs = get.calls[0]
    assert url == HOLDINGS_URL and kwargs["allow_redirects"] is True
    assert kwargs["headers"]["User-Agent"] == USER_AGENT
    assert USER_AGENT.startswith("Mozilla/5.0") and "python-requests" not in USER_AGENT
    assert file.raw == raw and file.etag == '"e1"' and file.retrieved_at == NOW
    assert file.content_type == "application/octet-stream"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (requests.ConnectionError("boom"), "unreachable"),
        (requests.Timeout("slow"), "unreachable"),
        (FakeResponse(403, b"denied"), "http_status"),
        (FakeResponse(500, b""), "http_status"),
    ],
)
def test_a_fetch_failure_is_a_typed_error_without_bytes(reply, reason):
    with pytest.raises(HoldingsFetchError) as caught:
        fetch_holdings(get=FakeGet(reply))
    assert caught.value.reason == reason and caught.value.raw is None


def test_a_short_body_is_a_truncated_download_that_keeps_what_arrived():
    raw = holdings_xlsx(base_tickers())
    reply = FakeResponse(200, raw[:-500], {"content-length": str(len(raw))})
    with pytest.raises(HoldingsFetchError, match="content-length") as caught:
        fetch_holdings(get=FakeGet(reply))
    assert caught.value.reason == "truncated" and caught.value.raw == raw[:-500]


def test_an_oversized_download_is_refused_unread():
    big = b"x" * (mh.MAX_DOWNLOAD_BYTES + 1)
    with pytest.raises(HoldingsFetchError) as caught:
        fetch_holdings(get=FakeGet(FakeResponse(200, big, {})))
    assert caught.value.reason == "too_large" and caught.value.raw is None


# --------------------------------------------------------------------------- the runner


class FakeStore:
    """An in-memory ``HoldingsStore``: the latest vintage, and every recorded result."""

    def __init__(self, latest=None, *, fail_record: bool = False) -> None:
        self._latest = latest
        self.fail_record = fail_record
        self.recorded: list[HoldingsResult] = []

    def latest(self, index_name: str):
        return self._latest

    def record(self, index_name: str, result: HoldingsResult) -> int:
        if self.fail_record:
            raise RuntimeError("database down")
        self.recorded.append(result)
        return len(self.recorded)


def run(store: FakeStore, raw: bytes | Exception):
    def fetch() -> HoldingsFile:
        if isinstance(raw, Exception):
            raise raw
        return holdings_file(raw)

    return check_holdings(store, fetch=fetch, today=lambda: TODAY)


def test_every_run_records_exactly_one_row_whatever_happens():
    ours = base_tickers()
    store = FakeStore(vintage_data(history(ours)))
    outcomes = [
        run(store, holdings_xlsx(ours)),
        run(store, holdings_xlsx([t for t in ours if t != "X0003"] + ["NEWCO"])),
        run(store, HoldingsFetchError("unreachable", "no route")),
        run(store, b"<html>blocked</html>"),
        run(store, holdings_xlsx(ours, "As of 20-Dec-2026")),
    ]
    assert [r.result.outcome for r in outcomes] == [
        "matched", "mismatch", "fetch_failed", "fetch_failed", "fetch_failed",
    ]  # fmt: skip
    assert [r.result.exit_code for r in outcomes] == [0, 2, 1, 1, 1]
    assert [r.check_id for r in outcomes] == [1, 2, 3, 4, 5] and len(store.recorded) == 5
    assert store.recorded[2].raw is None  # nothing arrived, nothing to archive
    assert store.recorded[3].raw == b"<html>blocked</html>"  # HTML under a 200 is kept as evidence


def test_a_record_failure_is_reported_not_hidden():
    store = FakeStore(vintage_data(history(base_tickers())), fail_record=True)
    outcome = run(store, holdings_xlsx(base_tickers()))
    assert outcome.check_id is None and "could not record" in (outcome.record_error or "")
    assert outcome.result.outcome == "matched"


# --------------------------------------------------------------------------- the real store's SQL


class _Cursor:
    def __init__(self, log: list[tuple[str, tuple]]) -> None:
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params: tuple = ()):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return (42,)


class _Connection:
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []
        self.committed = self.closed = False

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_the_real_store_only_inserts_into_checks_and_check_blobs(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(mh, "connect", lambda url=None, **kw: conn)
    raw = holdings_xlsx(base_tickers())
    result = compare(base_tickers(), base_tickers())
    assert PgHoldingsStore("ignored").record("sp500", result) == 42
    statements = [sql for sql, _ in conn.log]
    assert len(statements) == 2
    assert statements[0].startswith("INSERT INTO universe.checks ")
    assert statements[1].startswith("INSERT INTO universe.check_blobs ")
    assert conn.committed and conn.closed
    check_params, blob_params = conn.log[0][1], conn.log[1][1]
    assert check_params[3] == "matched" and check_params[4] == 1  # outcome, matched_load_id
    check_id, sha, blob = blob_params
    assert (check_id, sha) == (42, hashlib.sha256(raw).hexdigest())
    assert gzip.decompress(blob) == raw  # the archive is the exact bytes, gzipped


def test_nothing_arrived_means_no_blob_row(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(mh, "connect", lambda url=None, **kw: conn)
    result = mh.failed_holdings(HoldingsFetchError("unreachable", "no route"))
    PgHoldingsStore("ignored").record("sp500", result)
    assert [sql.split(" (")[0] for sql, _ in conn.log] == ["INSERT INTO universe.checks"]


# --------------------------------------------------------------------------- the command


def _patch_cli(monkeypatch, store: FakeStore, raw: bytes | Exception) -> None:
    def fetch() -> HoldingsFile:
        if isinstance(raw, Exception):
            raise raw
        return holdings_file(raw)

    monkeypatch.setattr(cmd_membership, "PgHoldingsStore", lambda: store)
    original = check_holdings  # the real one, not an earlier patch
    monkeypatch.setattr(
        cmd_membership, "check_holdings", lambda s: original(s, fetch=fetch, today=lambda: TODAY)
    )


def _args() -> argparse.Namespace:
    return argparse.Namespace(membership_command="check-holdings")


def test_the_command_exit_codes_and_report(monkeypatch, capsys):
    ours = base_tickers()
    store = FakeStore(vintage_data(history(ours)))
    _patch_cli(monkeypatch, store, holdings_xlsx(ours))
    assert cmd_membership.run_membership(_args()) == 0
    out = capsys.readouterr().out
    assert "MATCHED" in out and "attested through 2026-11-05" in out and "row 1" in out

    _patch_cli(monkeypatch, store, holdings_xlsx([t for t in ours if t != "X0009"] + ["ZZNEW"]))
    assert cmd_membership.run_membership(_args()) == 2
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "['X0009']" in out and "['ZZNEW']" in out
    assert "NOT advanced" in out

    _patch_cli(monkeypatch, store, HoldingsFetchError("http_status", "the host answered HTTP 403"))
    assert cmd_membership.run_membership(_args()) == 1
    assert "FETCH_FAILED" in capsys.readouterr().out


def test_the_command_fails_when_the_row_cannot_be_recorded(monkeypatch, capsys):
    store = FakeStore(vintage_data(history(base_tickers())), fail_record=True)
    _patch_cli(monkeypatch, store, holdings_xlsx(base_tickers()))
    assert cmd_membership.run_membership(_args()) == 1
    assert "NOT RECORDED" in capsys.readouterr().err


def test_the_command_fails_cleanly_when_the_database_cannot_be_read(monkeypatch, capsys):
    class Down(FakeStore):
        def latest(self, index_name):
            raise RuntimeError("connection refused")

    _patch_cli(monkeypatch, Down(), holdings_xlsx(base_tickers()))
    assert cmd_membership.run_membership(_args()) == 1
    assert "cannot compare" in capsys.readouterr().err


def test_check_holdings_is_registered_on_the_parser():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    cmd_membership.register_membership_parsers(subs)
    args = parser.parse_args(["membership", "check-holdings"])
    assert args.membership_command == "check-holdings"
