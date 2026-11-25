"""Tests for early_names: McGill company-name attachment and SQL guardrails."""

from __future__ import annotations

import datetime as dt
from typing import Any

import early_names
import pandas as pd
import pytest

pytestmark = pytest.mark.study


class FakeCursor:
    """In-memory cursor answering only ``kaggle_chars.mcgill_of`` queries."""

    def __init__(self, mcgill_rows: list[tuple[Any, ...]]) -> None:
        self._mcgill = mcgill_rows
        self._rows: list[tuple[Any, ...]] = []
        self.executed: list[tuple[str, Any]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if "kaggle_chars.mcgill_of" not in sql:
            raise AssertionError(f"unexpected query: {sql}")
        _load_id, permnos, max_date = params
        wanted = set(permnos)
        self._rows = [r for r in self._mcgill if r[0] in wanted and r[1] <= max_date]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class FakeConn:
    """Read-only connection double holding one ``FakeCursor``."""

    def __init__(self, mcgill_rows: list[tuple[Any, ...]]) -> None:
        self.cur = FakeCursor(mcgill_rows)

    def cursor(self) -> FakeCursor:
        return self.cur


def _sample_roster(*, tz: str | None = None) -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2001-06-15", "2002-03-29", "2000-02-14", "2003-08-10", "2004-01-12"],
        utc=(tz == "UTC"),
    )
    frame = pd.DataFrame(
        {
            "column": [
                "PERMNO-10001",
                "PERMNO-20002~1",
                "PERMNO-30003",
                "PERMNO-40004",
                "NO_PERMNO",
            ],
            "permno": pd.Series([10001, 20002, 30003, 40004, pd.NA], dtype="Int64"),
            "symbol": ["PERMNO-10001", "PERMNO-20002", "PERMNO-30003", "PERMNO-40004", "NO_PERMNO"],
            "last_bar": dates,
            "decision_date": dates - pd.Timedelta(days=10),
            "rank_band": ["1-50", "51-150", "151-500", "501+", "501+"],
            "held_by_rule": [True, False, False, False, False],
        }
    )
    return frame


@pytest.fixture
def fake_conn() -> FakeConn:
    # (permno, date, comp_name, exchcd)
    rows = [
        (10001, dt.date(2001, 3, 30), "ALPHA OLD", 1),
        (10001, dt.date(2001, 5, 31), "  ALPHA CORP  ", 1),
        (10001, dt.date(2001, 6, 29), "ALPHA AFTER", 1),
        # Exact date match on last_bar (2002-03-29):
        (20002, dt.date(2002, 2, 28), "BETA EARLIER", 3),
        (20002, dt.date(2002, 3, 29), "BETA EXACT", 3),
        # Only a strictly later row exists (2000-02-29 > 2000-02-14):
        (30003, dt.date(2000, 2, 29), "GAMMA LATER", 1),
        # 40004 is absent from McGill completely.
    ]
    return FakeConn(rows)


def test_attach_names_picks_latest_mcgill_row_on_or_before_last_bar(
    fake_conn: FakeConn,
) -> None:
    roster = _sample_roster()
    out = early_names.attach_names(fake_conn, roster)

    assert len(out) == len(roster)
    assert set(early_names.NAME_COLUMNS) <= set(out.columns)
    assert str(out["exchcd"].dtype) == "Int64"

    by_col = out.set_index("column")
    # 10001 died 2001-06-15 -> picks 2001-05-31 ("ALPHA CORP", stripped), ignores 2001-06-29.
    assert by_col.loc["PERMNO-10001", "company_name"] == "ALPHA CORP"
    assert by_col.loc["PERMNO-10001", "name_source"] == "mcgill"
    assert by_col.loc["PERMNO-10001", "exchcd"] == 1

    # 20002 died 2002-03-29 -> exact-day match picks 2002-03-29 ("BETA EXACT").
    assert by_col.loc["PERMNO-20002~1", "company_name"] == "BETA EXACT"
    assert by_col.loc["PERMNO-20002~1", "name_source"] == "mcgill"
    assert by_col.loc["PERMNO-20002~1", "exchcd"] == 3


def test_attach_names_leaves_missing_later_only_and_na_permno_as_none(
    fake_conn: FakeConn,
) -> None:
    roster = _sample_roster()
    before_cols = list(roster.columns)
    out = early_names.attach_names(fake_conn, roster)

    # Caller's frame is never mutated.
    assert list(roster.columns) == before_cols

    by_col = out.set_index("column")
    for col in ("PERMNO-30003", "PERMNO-40004", "NO_PERMNO"):
        assert by_col.loc[col, "company_name"] is None
        assert by_col.loc[col, "name_source"] == ""
        assert pd.isna(by_col.loc[col, "exchcd"])

    # Only "mcgill" or "" are ever emitted (no CRSP fallback, no mcgill_later).
    assert set(out["name_source"]).issubset({"mcgill", ""})


def test_attach_names_skips_blank_or_null_comp_name_rows() -> None:
    conn = FakeConn(
        [
            (10001, dt.date(2001, 4, 30), "VALID PRIOR NAME", 1),
            (10001, dt.date(2001, 5, 31), "   ", 1),
            (20002, dt.date(2001, 5, 31), None, 3),
        ]
    )
    roster = pd.DataFrame(
        {
            "column": ["PERMNO-10001", "PERMNO-20002"],
            "permno": pd.Series([10001, 20002], dtype="Int64"),
            "symbol": ["PERMNO-10001", "PERMNO-20002"],
            "last_bar": pd.to_datetime(["2001-06-15", "2001-06-15"]),
            "decision_date": pd.to_datetime(["2001-05-31", "2001-05-31"]),
            "rank_band": ["1-50", "51-150"],
            "held_by_rule": [True, False],
        }
    )
    out = early_names.attach_names(conn, roster).set_index("column")
    assert out.loc["PERMNO-10001", "company_name"] == "VALID PRIOR NAME"
    assert out.loc["PERMNO-10001", "name_source"] == "mcgill"
    assert out.loc["PERMNO-10001", "exchcd"] == 1
    assert out.loc["PERMNO-20002", "company_name"] is None
    assert out.loc["PERMNO-20002", "name_source"] == ""
    assert pd.isna(out.loc["PERMNO-20002", "exchcd"])


def test_attach_names_handles_tz_aware_last_bar_and_empty_roster(fake_conn: FakeConn) -> None:
    tz_roster = _sample_roster(tz="UTC")
    out = early_names.attach_names(fake_conn, tz_roster).set_index("column")
    assert out.loc["PERMNO-10001", "company_name"] == "ALPHA CORP"
    assert out.loc["PERMNO-20002~1", "company_name"] == "BETA EXACT"
    assert out.loc["PERMNO-30003", "company_name"] is None

    empty_conn = FakeConn([])
    empty = tz_roster.iloc[0:0].copy()
    empty_out = early_names.attach_names(empty_conn, empty)
    assert empty_out.empty
    assert set(early_names.NAME_COLUMNS) <= set(empty_out.columns)
    assert str(empty_out["exchcd"].dtype) == "Int64"
    assert empty_conn.cur.executed == []

    na_only = tz_roster[tz_roster["permno"].isna()].copy()
    na_out = early_names.attach_names(empty_conn, na_only)
    assert len(na_out) == 1
    assert na_out["company_name"].iloc[0] is None
    assert na_out["name_source"].iloc[0] == ""
    assert pd.isna(na_out["exchcd"].iloc[0])
    assert empty_conn.cur.executed == []


def test_attach_names_sql_is_parameterised_and_reads_only_mcgill_name_columns(
    fake_conn: FakeConn,
) -> None:
    roster = _sample_roster()
    early_names.attach_names(fake_conn, roster)

    assert len(fake_conn.cur.executed) == 1
    sql, params = fake_conn.cur.executed[0]
    lower = sql.lower()

    # Uses kaggle_chars.mcgill_of(2) with parameterised permno list and max last_bar date.
    assert "kaggle_chars.mcgill_of(%s::bigint)" in lower
    assert "permno = any(%s::bigint[])" in lower
    assert "date <= %s::date" in lower
    assert params[0] == 2
    assert params[1] == [10001, 20002, 30003, 40004]
    assert params[2] == dt.date(2003, 8, 10)

    # No literal permno or date interpolated into SQL.
    for literal in ("10001", "20002", "2003-08-10", "2004-01-12"):
        assert literal not in sql

    # Never touches any other table or non-name/non-exchange columns.
    forbidden = (
        "kaggle_crsp",
        "daily_of",
        "market.",
        "ohlcv",
        "symbol_map",
        "stock_spans",
        "prc",
        "shrout",
        "market_equity",
        "ret_",
    )
    for token in forbidden:
        assert token not in lower, f"forbidden token {token!r} found in SQL: {sql}"


def test_attach_names_rejects_missing_required_columns(fake_conn: FakeConn) -> None:
    bad = pd.DataFrame({"column": ["A"], "permno": [10001]})
    with pytest.raises(ValueError, match="last_bar"):
        early_names.attach_names(fake_conn, bad)
