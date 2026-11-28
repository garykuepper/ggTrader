"""Check weekly forward reporting against an isolated database."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name: str, path: Path) -> object:
    """Load a script or test fixture by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture_module = load_file("health_db_fixture", ROOT / "tests/db/test_forward_schema.py")
database = fixture_module.database
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
