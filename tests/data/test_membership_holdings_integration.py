"""Integration tests of ``check-holdings`` against a throwaway database on a real server.

Never part of the fast, full or CI tiers (marked ``integration`` and ``db``); they skip unless
``GGT_TEST_DB_URL`` names a server, and work only inside their own ``ggt_test_<uuid>`` database
(``tests/data/scratch_db.py``). Run::

    GGT_TEST_DB_URL=... GGT_TEST_DB_ALLOW_SHARED=1 \\
        scripts/run_tests.sh -m integration tests/data/test_membership_holdings_integration.py
"""

from __future__ import annotations

import gzip
import hashlib
from datetime import date, datetime, timezone

import psycopg2
import psycopg2.extras
import pytest
from psycopg2 import errors

from ggTrader.data.membership import DbMembershipSource, PgMembershipStore
from ggTrader.data.membership_gate import decide_gate
from ggTrader.data.membership_holdings import HoldingsFetchError, PgHoldingsStore, check_holdings
from ggTrader.data.universe_db import LoadRequest, apply_ddl, load_vintage
from tests.data.membership_fakes import synthetic_history
from tests.data.test_membership_holdings import holdings_file
from tests.data.test_membership_upstream import csv_bytes
from tests.data.xlsx_builder import holdings_xlsx

pytestmark = [pytest.mark.db, pytest.mark.integration]

V1_AT = datetime(2026, 6, 9, tzinfo=timezone.utc)
TODAY = date(2026, 11, 8)
PROTECTED = ("loads", "membership", "snapshot_dates", "load_blobs")


def _counts(conn: psycopg2.extensions.connection, tables: tuple[str, ...] = PROTECTED) -> dict:
    out = {}
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT count(*) FROM universe.{table}")
            out[table] = cur.fetchone()[0]
    conn.rollback()
    return out


def _load_synthetic(conn: psycopg2.extensions.connection):
    history = synthetic_history(3, n_dates=6)
    load_vintage(
        conn,
        LoadRequest(
            raw=csv_bytes(history),
            source="fixture-import",
            source_url="file:test",
            retrieved_at=V1_AT,
            commit=True,
            loader_git_sha="test",
            parent="none",
        ),
    )
    return history, history[max(history)]


def _run(db_url: str, raw: bytes | Exception):
    def fetch():
        if isinstance(raw, Exception):
            raise raw
        return holdings_file(raw)

    return check_holdings(PgHoldingsStore(db_url), fetch=fetch, today=lambda: TODAY)


def test_the_ddl_is_idempotent_and_widens_an_old_outcome_list(
    conn: psycopg2.extensions.connection,
) -> None:
    apply_ddl(conn)  # a second run on top of the fixture's first
    with conn.cursor() as cur:  # put the table back to the pre-M4b shape
        cur.execute("ALTER TABLE universe.checks DROP CONSTRAINT checks_outcome_check")
        cur.execute(
            "ALTER TABLE universe.checks ADD CONSTRAINT checks_outcome_check CHECK "
            "(outcome IN ('unchanged', 'loaded', 'refused', 'fetch_failed'))"
        )
        cur.execute("INSERT INTO universe.checks (index_name, outcome) VALUES ('sp500', 'refused')")
    conn.commit()
    apply_ddl(conn)
    apply_ddl(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'universe.checks'::regclass AND conname = 'checks_outcome_check'"
        )
        definition = cur.fetchone()[0]
        assert "matched" in definition and "mismatch" in definition
        cur.execute("SELECT outcome FROM universe.checks")
        assert cur.fetchall() == [("refused",)]  # the old row survived untouched
        cur.execute(
            "INSERT INTO universe.checks (index_name, outcome) VALUES ('sp500', 'mismatch')"
        )
        with pytest.raises(errors.CheckViolation):
            cur.execute(
                "INSERT INTO universe.checks (index_name, outcome) VALUES ('sp500', 'bogus')"
            )
    conn.rollback()


def test_a_matched_row_must_name_its_vintage_and_the_holdings_date(
    conn: psycopg2.extensions.connection,
) -> None:
    _load_synthetic(conn)
    insert = (
        "INSERT INTO universe.checks (index_name, outcome, matched_load_id, detail) "
        "VALUES ('sp500', 'matched', %s, %s)"
    )
    for load_id, detail in (
        (None, '{"holdings_as_of": "2026-11-05"}'),
        (1, "{}"),
        (1, None),
        (1, '{"holdings_as_of": "soon"}'),
    ):
        with conn.cursor() as cur, pytest.raises(errors.CheckViolation):
            cur.execute(insert, (load_id, detail))
        conn.rollback()
    with conn.cursor() as cur:
        cur.execute(insert, (1, '{"holdings_as_of": "2026-11-05"}'))  # complete: accepted
        cur.execute("INSERT INTO universe.checks (index_name, outcome) VALUES ('sp500', 'refused')")
    conn.commit()
    apply_ddl(conn)  # idempotent with the constraint (and rows) already there


def test_a_matched_row_without_a_holdings_date_never_counts_even_without_the_constraint(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _load_synthetic(conn)
    with conn.cursor() as cur:  # an older database that lacks the new constraint
        cur.execute("ALTER TABLE universe.checks DROP CONSTRAINT checks_matched_complete_check")
        cur.execute(
            "INSERT INTO universe.checks (index_name, outcome, matched_load_id, detail) "
            "VALUES ('sp500', 'matched', 1, '{}')"
        )
    conn.commit()
    vintage = DbMembershipSource("latest", store=PgMembershipStore(db_url)).vintage()
    assert vintage.matched_at is None and vintage.known_through is None
    assert decide_gate(vintage, TODAY).level == "refuse"


def test_a_match_is_recorded_archived_and_read_by_the_gate_and_nothing_else_changes(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    history, latest = _load_synthetic(conn)
    before = _counts(conn)
    store = PgMembershipStore(db_url)
    vintage0 = DbMembershipSource("latest", store=store).vintage()
    assert vintage0.matched_at is None and vintage0.known_through is None
    assert decide_gate(vintage0, TODAY).level == "refuse"  # never matched

    raw = holdings_xlsx(list(latest))
    outcome = _run(db_url, raw)
    assert outcome.result.outcome == "matched" and outcome.check_id is not None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT outcome, matched_load_id, detail->>'holdings_as_of', raw_sha256 "
            "FROM universe.checks WHERE check_id = %s",
            (outcome.check_id,),
        )
        assert cur.fetchone() == ("matched", 1, "2026-11-05", hashlib.sha256(raw).hexdigest())
        cur.execute(
            "SELECT raw_sha256, raw_gzip FROM universe.check_blobs WHERE check_id = %s",
            (outcome.check_id,),
        )
        sha, blob = cur.fetchone()
        assert sha == hashlib.sha256(raw).hexdigest() and gzip.decompress(bytes(blob)) == raw
    assert _counts(conn) == before

    vintage = DbMembershipSource("latest", store=store).vintage()
    assert vintage.known_through == date(2026, 11, 5) and vintage.matched_at is not None
    assert decide_gate(vintage, TODAY).level == "pass"


def test_every_outcome_writes_one_row_and_never_touches_the_list(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _, latest = _load_synthetic(conn)
    before = _counts(conn)
    runs = [
        _run(db_url, holdings_xlsx(list(latest))),  # matched
        _run(db_url, holdings_xlsx(list(latest)[:-1] + ["NEWCO"])),  # mismatch
        _run(db_url, holdings_xlsx(list(latest), "As of 20-Dec-2026")),  # future
        _run(db_url, b"<html>blocked</html>"),  # HTML under a 200
        _run(db_url, HoldingsFetchError("unreachable", "no route")),  # nothing arrived
    ]
    assert [r.result.outcome for r in runs] == [
        "matched", "mismatch", "fetch_failed", "fetch_failed", "fetch_failed",
    ]  # fmt: skip
    assert [r.check_id for r in runs] == [2, 3, 4, 5, 6]  # row 1 is the load's own 'loaded' row
    assert _counts(conn) == before
    assert _counts(conn, ("checks", "check_blobs")) == {"checks": 6, "check_blobs": 4}
    with conn.cursor() as cur:
        cur.execute("SELECT check_id FROM universe.check_blobs ORDER BY 1")
        assert [r[0] for r in cur.fetchall()] == [2, 3, 4, 5]  # no blob when nothing arrived
        cur.execute(
            "SELECT detail->'only_in_list', detail->'only_in_holdings' FROM universe.checks "
            "WHERE check_id = 3"
        )
        assert cur.fetchone() == ([latest[-1]], ["NEWCO"])
    conn.rollback()


def test_only_a_matched_check_extends_the_clock_a_mismatch_and_a_failure_do_not(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _, latest = _load_synthetic(conn)
    _run(db_url, holdings_xlsx(list(latest)[:-1] + ["NEWCO"]))
    _run(db_url, HoldingsFetchError("http_status", "HTTP 403"))
    vintage = DbMembershipSource("latest", store=PgMembershipStore(db_url)).vintage()
    assert vintage.matched_at is None and vintage.known_through is None


def test_a_match_belongs_to_one_vintage_only(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    """After a newer vintage is loaded the old match says nothing about it (gate refuses again)."""
    history, latest = _load_synthetic(conn)
    _run(db_url, holdings_xlsx(list(latest)))
    newer = dict(history)
    newer[date(2026, 6, 3)] = tuple(sorted(set(latest) - {sorted(latest)[0]} | {"ZZZNEW"}))
    load_vintage(
        conn,
        LoadRequest(
            raw=csv_bytes(newer), source="fixture-import", source_url="file:test",
            retrieved_at=datetime(2026, 6, 10, tzinfo=timezone.utc), commit=True,
            loader_git_sha="test",
        ),
    )  # fmt: skip
    store = PgMembershipStore(db_url)
    assert DbMembershipSource("latest", store=store).vintage().matched_at is None
    assert DbMembershipSource(1, store=store).vintage().known_through == date(2026, 11, 5)


def test_check_blobs_are_append_only(conn: psycopg2.extensions.connection, db_url: str) -> None:
    _, latest = _load_synthetic(conn)
    _run(db_url, holdings_xlsx(list(latest)))
    with conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.RaiseException, match="append-only"):
            cur.execute("UPDATE universe.check_blobs SET raw_sha256 = repeat('0', 64)")
    conn.rollback()
    with conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.RaiseException, match="append-only"):
            cur.execute("DELETE FROM universe.check_blobs")
    conn.rollback()
    with conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.RaiseException, match="append-only"):
            cur.execute("TRUNCATE universe.check_blobs")
    conn.rollback()


def _record(conn, outcome: str, as_of: str | None, load: int | None, compared: int | None = None):
    """Insert one holdings-check row the way ``PgHoldingsStore`` writes it."""
    detail = {}
    if as_of:
        detail["holdings_as_of"] = as_of
    if compared is not None:
        detail["compared_load_id"] = compared
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO universe.checks (index_name, outcome, matched_load_id, detail) "
            "VALUES ('sp500', %s, %s, %s) RETURNING check_id",
            (outcome, load, psycopg2.extras.Json(detail)),
        )
        check_id = cur.fetchone()[0]
    conn.commit()
    return check_id


def _newest(db_url: str, load: int | str = "latest"):
    v = DbMembershipSource(load, store=PgMembershipStore(db_url)).vintage()
    return v.latest_check, v.latest_check_as_of


def test_the_newest_holdings_check_by_as_of_date_decides_the_gate(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _load_synthetic(conn)
    assert _newest(db_url) == (None, None)
    _record(conn, "matched", "2026-11-03", 1)
    assert _newest(db_url) == ("matched", date(2026, 11, 3))
    # a match, then a newer mismatch: the list is known to differ, the gate closes
    bad = _record(conn, "mismatch", "2026-11-05", 1, compared=1)
    assert _newest(db_url) == ("mismatch", date(2026, 11, 5))
    v = DbMembershipSource("latest", store=PgMembershipStore(db_url)).vintage()
    assert v.latest_check_id == bad
    today = datetime.now(timezone.utc).date()  # the rows carry the real check time
    assert decide_gate(v, today).code == "membership_mismatch"
    # a mismatch with an OLDER as-of date than the match does not close the gate
    _record(conn, "mismatch", "2026-11-01", 1, compared=1)
    # a fetch_failed row never counts, however new
    _record(conn, "fetch_failed", "2026-11-09", None)
    assert _newest(db_url) == ("mismatch", date(2026, 11, 5))
    # a later match reopens it
    _record(conn, "matched", "2026-11-06", 1)
    assert _newest(db_url) == ("matched", date(2026, 11, 6))
    reopened = DbMembershipSource("latest", store=PgMembershipStore(db_url)).vintage()
    assert decide_gate(reopened, today).level in ("pass", "warn")


def test_an_older_mismatch_after_a_match_is_ignored_and_a_tie_goes_to_the_later_check(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    _load_synthetic(conn)
    _record(conn, "matched", "2026-11-05", 1)
    _record(conn, "mismatch", "2026-11-04", 1, compared=1)  # recorded later, dated earlier
    assert _newest(db_url) == ("matched", date(2026, 11, 5))
    _record(conn, "mismatch", "2026-11-05", 1, compared=1)  # same day, recorded later
    assert _newest(db_url) == ("mismatch", date(2026, 11, 5))


def test_a_mismatch_against_an_older_vintage_is_ignored(
    conn: psycopg2.extensions.connection, db_url: str
) -> None:
    history, latest = _load_synthetic(conn)
    _record(conn, "mismatch", "2026-11-05", 1, compared=1)
    newer = dict(history)
    newer[date(2026, 6, 3)] = tuple(sorted(set(latest) - {sorted(latest)[0]} | {"ZZZNEW"}))
    load_vintage(
        conn,
        LoadRequest(
            raw=csv_bytes(newer), source="fixture-import", source_url="file:test",
            retrieved_at=datetime(2026, 6, 10, tzinfo=timezone.utc), commit=True,
            loader_git_sha="test",
        ),
    )  # fmt: skip
    assert _newest(db_url) == (None, None)  # the new latest vintage has no comparison yet
    assert _newest(db_url, 1) == ("mismatch", date(2026, 11, 5))  # the old one keeps its own
