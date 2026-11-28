"""Check weekly forward reporting against an isolated database."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.db, pytest.mark.integration]


def load_file(name: str, path: Path) -> object:
    """Load a script or test fixture by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = load_file("forward_health", ROOT / "scripts/forward_health.py")


@pytest.fixture
def populated(database: object) -> object:
    """Populate two collection days in the temporary database."""
    with database.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS forward CASCADE")
        cur.execute((ROOT / "sql/forward_schema.sql").read_text())
        for date, gap_count in [("2026-11-23", 0), ("2026-11-24", 1)]:
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES (%s, %s, %s, %s, %s, %s) RETURNING load_id",
                (date, "test", "a" * 64, "b" * 64, "abc", "test"),
            )
            load = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO forward.listing_snapshots "
                "(load_id,day,symbol,exchange,security_name,is_etf,is_test) "
                "VALUES (%s,%s,'AAA','Q','AAA Common Stock',false,false)",
                (load, date),
            )
            if gap_count:
                cur.execute(
                    "INSERT INTO forward.listing_snapshots "
                    "(load_id,day,symbol,exchange,security_name,is_etf,is_test) "
                    "VALUES (%s,%s,'BBB','Q','BBB Common Stock',false,false)",
                    (load, date),
                )
            cur.execute(
                "INSERT INTO forward.daily_bars VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (load, date, "AAA", "sip", 1, 2, 1, 2, 10),
            )
            if gap_count:
                cur.execute(
                    "INSERT INTO forward.gaps VALUES (%s,%s,%s,%s)",
                    (load, date, "BBB", "missing_bar"),
                )
        cur.execute(
            "INSERT INTO forward.symbol_status (symbol, first_seen, last_seen, ended_on) "
            "VALUES ('CCC', '2026-11-01', '2026-11-16', '2026-11-24')"
        )
    return database


def test_weekly_report_computes_mean_coverage(populated: object) -> None:
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["mean_coverage"] == 0.75
    assert result["worst_day"] == dt.date(2026, 11, 24)
    assert result["days"] == 2
    assert result["gaps"] == 1


def test_report_lists_ended_symbols(populated: object) -> None:
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["ended"] == ["CCC"]
    message = health.format_message(result)
    assert "CCC" in message and "75.0%" in message


def test_empty_week_is_reported_without_inventing_coverage(populated: object) -> None:
    result = health.weekly_report(populated, dt.date(2027, 1, 1))
    assert result["days"] == 0
    assert result["mean_coverage"] is None
    assert result["worst_day"] is None
    assert "no collections" in health.format_message(result).lower()


def test_shares_only_load_and_listing_gap_do_not_distort_coverage(populated: object) -> None:
    with populated.cursor() as cur:
        cur.execute("SELECT load_id FROM forward.loads WHERE day='2026-11-24'")
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.gaps VALUES (%s,%s,%s,%s)",
            (load, dt.date(2026, 11, 24), "::listing-missing::", "listing_missing"),
        )
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-25','yahoo-shares',%s,%s,'a','test')",
            ("a" * 64, "b" * 64),
        )
    report = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert report["days"] == 2
    assert report["mean_coverage"] == 0.75
    assert report["missing_listings"] == [dt.date(2026, 11, 24)]
    assert "listing" in health.format_message(report).lower()


def test_weekly_shares_load_is_reported(populated: object) -> None:
    missing = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert missing["shares_loaded"] is False
    assert "shares: missing" in health.format_message(missing).lower()
    with populated.cursor() as cur:
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-23','yahoo-shares',%s,%s,'a','test') "
            "RETURNING load_id",
            ("a" * 64, "b" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.shares_outstanding (load_id,day,symbol,shares,source) "
            "VALUES (%s,'2026-11-23','AAA',100,'yahoo')",
            (load,),
        )
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["shares_loaded"] is True
    assert result["shares_count"] == 1
    assert "shares: loaded" in health.format_message(result).lower()


def test_coverage_excludes_tracked_symbols_not_in_current_listing(populated: object) -> None:
    with populated.cursor() as cur:
        cur.execute("SELECT load_id FROM forward.loads WHERE day='2026-11-23'")
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.gaps VALUES (%s,'2026-11-23','OLD','missing_bar')", (load,)
        )
        cur.execute(
            "INSERT INTO forward.daily_bars VALUES (%s,'2026-11-23','EXTRA','sip',1,2,1,2,10)",
            (load,),
        )
    report = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert report["mean_coverage"] == 0.75


def test_late_never_traded_marker_does_not_change_original_coverage(populated: object) -> None:
    with populated.cursor() as cur:
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-22','catchup',%s,%s,'x','x') RETURNING load_id",
            ("a" * 64, "b" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.gaps VALUES (%s,'2026-11-22','BBB','never_traded')", (load,)
        )
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["mean_coverage"] == 0.75


def test_weekly_shares_report_shows_unresolved_failures(populated: object) -> None:
    with populated.cursor() as cur:
        cur.execute(
            "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
            "code_commit, actor) VALUES ('2026-11-23','yahoo-shares',%s,%s,'a','test') "
            "RETURNING load_id",
            ("a" * 64, "b" * 64),
        )
        load = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO forward.gaps VALUES (%s,'2026-11-23','BBB','shares_failed')", (load,)
        )
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["shares_pending"] == 1
    assert "1 unresolved" in health.format_message(result).lower()


def test_resolved_shares_failure_is_not_pending(populated: object) -> None:
    with populated.cursor() as cur:
        for index in range(2):
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES ('2026-11-23','yahoo-shares',%s,%s,'a','test') "
                "RETURNING load_id",
                ("a" * 64, "b" * 64),
            )
            load = cur.fetchone()[0]
            if index == 0:
                cur.execute(
                    "INSERT INTO forward.gaps VALUES (%s,'2026-11-23','BBB','shares_failed')",
                    (load,),
                )
            else:
                cur.execute(
                    "INSERT INTO forward.shares_outstanding "
                    "VALUES (%s,'2026-11-24','BBB',100,'yahoo')",
                    (load,),
                )
    assert health.weekly_report(populated, dt.date(2026, 11, 28))["shares_pending"] == 0


def test_unavailable_only_load_resolves_previous_share_failure(populated: object) -> None:
    with populated.cursor() as cur:
        for reason in ["shares_failed", "shares_unavailable"]:
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES ('2026-11-23','yahoo-shares',%s,%s,'a','test') "
                "RETURNING load_id",
                ("a" * 64, "b" * 64),
            )
            load = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO forward.gaps VALUES (%s,'2026-11-24','BBB',%s)", (load, reason)
            )
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["shares_pending"] == 0
    assert result["shares_loaded"] is False


def test_earlier_unavailable_does_not_resolve_later_failure(populated: object) -> None:
    with populated.cursor() as cur:
        for reason in ["shares_unavailable", "shares_failed"]:
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES ('2026-11-23','yahoo-shares',%s,%s,'a','test') "
                "RETURNING load_id",
                ("a" * 64, "b" * 64),
            )
            load = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO forward.gaps VALUES (%s,'2026-11-24','BBB',%s)", (load, reason)
            )
    result = health.weekly_report(populated, dt.date(2026, 11, 28))
    assert result["shares_pending"] == 1
