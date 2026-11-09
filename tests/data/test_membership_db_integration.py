"""Integration tests of ``DbMembershipSource`` against a real PostgreSQL server.

Never part of the fast, full or CI tiers (marked ``integration`` and ``db``). Two groups:

* Scratch-database tests (need ``GGT_TEST_DB_URL``, like ``test_universe_db_integration.py``):
  load vintage 1 (and a tail-only vintage 2) into a throwaway database, read it back through
  ``PgMembershipStore`` and compare with the frozen fixture and with the old file functions.
* One production read-only check (needs ``GGT_TEST_PROD_READONLY=1`` and the project's own
  connection settings): ``db:1`` on the real database equals the fixture. It only reads.

Run::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 GGT_TEST_PROD_READONLY=1 \\
        scripts/run_tests.sh -m integration tests/data/test_membership_db_integration.py
"""

from __future__ import annotations

import gzip
import os
import random
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import psycopg2
import pytest

from ggTrader.data import membership
from ggTrader.data.core.index_constituents import SP500_HISTORY_CSV, sp500_members_asof
from ggTrader.data.membership import (
    DbMembershipSource,
    FixtureMembershipSource,
    MembershipSource,
    MembershipUnavailable,
    PgMembershipStore,
    load_manifest,
    membership_source,
)
from ggTrader.data.universe_algebra import History, parse_history_csv
from ggTrader.data.universe_db import LoadRequest, load_vintage

pytestmark = [pytest.mark.db, pytest.mark.integration]

V1 = "sp500_v1_2026-06-02"
V1_AT = datetime(2026, 6, 9, tzinfo=timezone.utc)
V2_AT = datetime(2026, 11, 7, 18, 26, 50, tzinfo=timezone.utc)
FIRST = date(1996, 1, 2)
CALENDAR_END = date(2026, 7, 2)
TAIL_DAY = date(2026, 8, 18)


def _v1_raw() -> bytes:
    if not SP500_HISTORY_CSV.exists():
        pytest.skip("vintage-1 file not present")
    return gzip.decompress(SP500_HISTORY_CSV.read_bytes())


def _csv_bytes(history: History) -> bytes:
    lines = ["date,tickers"] + [
        f'{d.isoformat()},"{",".join(t)}"' for d, t in sorted(history.items())
    ]
    return ("\n".join(lines) + "\n").encode()


def _v2_raw() -> bytes:
    """Vintage 1 plus one new tail snapshot (two names swapped): a tail-only change."""
    history = dict(parse_history_csv(_v1_raw()))
    last = history[max(history)]
    history[TAIL_DAY] = tuple(sorted((set(last) - {last[0], last[1]}) | {"NEWAAA", "NEWBBB"}))
    return _csv_bytes(history)


def _load(conn: psycopg2.extensions.connection, raw: bytes, at: datetime, **kwargs: object) -> None:
    request = LoadRequest(
        raw=raw,
        source="fixture-import",
        source_url="file:test",
        retrieved_at=at,
        commit=True,
        loader_git_sha="test",
        **kwargs,  # type: ignore[arg-type]
    )
    load_vintage(conn, request)


def _window_pairs(n: int) -> list[tuple[date, date]]:
    rng = random.Random(7)
    pairs = []
    for _ in range(n):
        start = FIRST + timedelta(days=rng.randint(0, 11_000))
        pairs.append((start, start + timedelta(days=rng.randint(0, 3_000))))
    return pairs


def _assert_same_as_fixture_and_old_reader(source: MembershipSource) -> None:
    """The database source equals the frozen fixture and the old file reader, on every date."""
    fixture = FixtureMembershipSource(V1)
    a, b = source.vintage(), fixture.vintage()
    assert (a.index_name, a.content_sha256, a.raw_sha256) == (
        b.index_name,
        b.content_sha256,
        b.raw_sha256,
    )
    assert (a.coverage_start, a.coverage_end, a.retrieved_at) == (
        b.coverage_start,
        b.coverage_end,
        b.retrieved_at,
    )
    pd.testing.assert_frame_equal(source.history(), fixture.history(), check_exact=True)
    snapshot_dates = fixture.snapshot_dates()
    assert source.snapshot_dates() == snapshot_dates and len(snapshot_dates) == 2712
    for day in snapshot_dates:  # all snapshot dates, against the old reader
        assert source.members_asof(day) == sp500_members_asof(pd.Timestamp(day))
    day = FIRST  # every calendar day, against the fixture (itself proven equal to the old reader)
    while day <= CALENDAR_END:
        assert source.members_asof(day) == fixture.members_asof(day)
        day += timedelta(days=1)
    for start, end in _window_pairs(300):
        assert source.all_members_between(start, end) == fixture.all_members_between(start, end)


# ---- scratch database ------------------------------------------------------------------------


def test_vintage_one_from_the_database_equals_the_fixture_and_the_old_reader(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _load(conn, _v1_raw(), V1_AT, parent="none")
    source = DbMembershipSource(1, store=PgMembershipStore(db_url))
    _assert_same_as_fixture_and_old_reader(source)
    assert source.vintage().kind == "db" and source.vintage().load_id == 1
    assert source.vintage().verified_at == V1_AT  # a 'loaded' check does not count as verification


def test_latest_follows_new_loads_but_a_pinned_vintage_does_not(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    store = PgMembershipStore(db_url)
    _load(conn, _v1_raw(), V1_AT, parent="none")
    assert membership_source("db", store=store).vintage().load_id == 1
    _load(conn, _v2_raw(), V2_AT)
    latest = membership_source("db", store=store)
    pinned = membership_source("db:1", store=store)
    assert (latest.vintage().load_id, pinned.vintage().load_id) == (2, 1)
    assert latest.vintage().content_sha256 != pinned.vintage().content_sha256
    assert latest.coverage_end == TAIL_DAY and pinned.coverage_end == date(2026, 6, 2)
    _assert_same_as_fixture_and_old_reader(pinned)  # v1 is still exactly v1
    assert {"NEWAAA", "NEWBBB"} <= set(latest.members_asof(TAIL_DAY))
    assert not {"NEWAAA", "NEWBBB"} & set(pinned.members_asof(TAIL_DAY))
    assert latest.members_asof(date(2026, 6, 2)) == pinned.members_asof(date(2026, 6, 2))


def test_an_unchanged_check_moves_neither_verified_at_nor_the_match_clock(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    """Appendix C3: only a ``matched`` holdings check counts; the loader's no-op row does not."""
    _load(conn, _v1_raw(), V1_AT, parent="none")
    _load(conn, _v1_raw(), V1_AT)  # same bytes: no new vintage, an 'unchanged' check row
    vintage = membership_source("db", store=PgMembershipStore(db_url)).vintage()
    assert vintage.load_id == 1 and vintage.retrieved_at == V1_AT
    assert vintage.verified_at == V1_AT
    assert vintage.matched_at is None and vintage.known_through is None


def test_an_empty_database_and_an_unknown_load_are_unavailable(db_url: str) -> None:
    store = PgMembershipStore(db_url)
    with pytest.raises(MembershipUnavailable, match="no vintage"):
        membership_source("db", store=store)
    with pytest.raises(MembershipUnavailable, match="no load 7"):
        membership_source("db:7", store=store)


def test_a_load_that_commits_during_the_read_does_not_disturb_it(
    conn: psycopg2.extensions.connection, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader's snapshot is fixed by its first statement: load 2 lands mid-read, unseen."""
    _load(conn, _v1_raw(), V1_AT, parent="none")
    real = membership.read_vintage
    landed: list[bool] = []

    def read_after_a_competing_load(cur: object, load_id: int, index_name: str) -> object:
        if not landed:
            _load(conn, _v2_raw(), V2_AT)
            landed.append(True)
        return real(cur, load_id, index_name)  # type: ignore[arg-type]

    monkeypatch.setattr(membership, "read_vintage", read_after_a_competing_load)
    source = membership_source("db", store=PgMembershipStore(db_url))
    assert landed == [True]
    assert source.vintage().load_id == 1  # latest was fixed by the first statement
    _assert_same_as_fixture_and_old_reader(source)


# ---- the production database, read only ------------------------------------------------------


def test_production_load_one_equals_the_fixture_and_the_manifest() -> None:
    """Reads the real database through the project's own connection; writes nothing."""
    if os.environ.get("GGT_TEST_PROD_READONLY") != "1":
        pytest.skip("set GGT_TEST_PROD_READONLY=1 to read the production database (read only)")
    entry = load_manifest()[V1]
    source = membership_source(f"db:{entry.db_load_id_on_encom}")
    _assert_same_as_fixture_and_old_reader(source)
    vintage = source.vintage()
    assert vintage.content_sha256 == entry.content_sha256
    assert vintage.raw_sha256 == entry.raw_sha256
    assert vintage.retrieved_at == entry.retrieved_at
