"""Tests for the roster of held dead stocks: exits parsing, name attachment, file reading."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import roster

pytestmark = pytest.mark.study

MCGILL_COLUMNS = ["permno", "date", "comp_name", "stock_ticker", "cusip", "exchcd"]
MAP_COLUMNS = ["permno", "symbol", "valid_from", "valid_to", "source", "release_id"]


class FakeCursor:
    """A cursor that answers the two read-only queries the roster makes, from lists."""

    def __init__(self, mcgill: list[tuple], symbol_map: list[tuple]) -> None:
        self._mcgill = mcgill
        self._map = symbol_map
        self._rows: list[tuple] = []
        self.executed: list[tuple[str, Any]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if "mcgill_of" in sql:
            wanted = set(params[1])
            self._rows = [r for r in self._mcgill if r[0] in wanted]
        elif "symbol_map" in sql:
            wanted = set(params[1])
            self._rows = [r for r in self._map if r[1] in wanted and r[5] == params[0]]
        else:
            raise AssertionError(f"unexpected query: {sql}")

    def fetchall(self) -> list[tuple]:
        return self._rows


class FakeConn:
    """A connection holding one fake cursor."""

    def __init__(self, mcgill: list[tuple], symbol_map: list[tuple]) -> None:
        self.cur = FakeCursor(mcgill, symbol_map)

    def cursor(self) -> FakeCursor:
        return self.cur


@pytest.fixture
def fake_conn() -> FakeConn:
    # McGill has 89260 -> ALPHA CORP; the symbol map sends ARG to 11111, which McGill lacks.
    mcgill = [
        (89260, dt.date(2005, 3, 31), "ALPHA OLD", "ALP", "00000001", 1),
        (89260, dt.date(2005, 5, 31), "ALPHA CORP", "ALP", "00000001", 1),
        (89260, dt.date(2005, 8, 31), "ALPHA LATER", "ALP", "00000001", 1),
        (50000, dt.date(2010, 1, 29), "BETA INC", "BET", "00000002", 3),
    ]
    symbol_map = [
        (11111, "ARG", dt.date(2002, 1, 31), dt.date(2016, 4, 29), "mcgill", 1),
        (22222, "ARG", dt.date(1990, 1, 31), dt.date(1995, 1, 31), "mcgill", 1),
        (33333, "ARG", dt.date(2002, 1, 31), dt.date(2016, 4, 29), "mcgill", 2),
    ]
    return FakeConn(mcgill, symbol_map)


def roster_frame() -> pd.DataFrame:
    result = {
        "result": {
            "death_price_paths": {
                "exits": [
                    {"symbol": "PERMNO-89260", "exit_date": "2005-06-24", "last_close": 74.98},
                    {"symbol": "ARG", "exit_date": "2010-05-14", "last_close": 40.0},
                    {"symbol": "PERMNO-50000", "exit_date": "2009-06-01", "last_close": 5.0},
                    {"symbol": "PERMNO-777", "exit_date": "2012-02-01", "last_close": 1.0},
                ]
            }
        }
    }
    return roster.parse_exits(result, window="dev", universe="A")


def test_parse_exits_reads_permno_and_ticker_rows() -> None:
    result = {
        "result": {
            "death_price_paths": {
                "exits": [
                    {"symbol": "PERMNO-89260", "exit_date": "2005-06-24", "last_close": 74.98},
                    {"symbol": "ARG", "exit_date": "2010-05-14", "last_close": 40.0},
                ]
            }
        }
    }
    out = roster.parse_exits(result, window="dev", universe="A")
    assert out["permno"].iloc[0] == 89260
    assert out["permno"].isna().tolist() == [False, True]
    assert out["symbol"].tolist() == ["PERMNO-89260", "ARG"]
    assert out["window"].tolist() == ["dev", "dev"]
    assert out["universe"].tolist() == ["A", "A"]
    assert str(out["permno"].dtype) == "Int64"
    assert out["exit_date"].iloc[0] == pd.Timestamp("2005-06-24")


def test_parse_exits_rejects_missing_exits_key() -> None:
    with pytest.raises(KeyError):
        roster.parse_exits({"result": {}}, "dev", "A")


def test_attach_names_marks_unnamed_and_fills_ticker_permno(fake_conn: FakeConn) -> None:
    out = roster.attach_names(fake_conn, roster_frame(), mcgill_load_id=2, release_id=1)
    by = out.set_index("symbol")
    assert by.loc["PERMNO-89260", "name_source"] == "mcgill"
    assert by.loc["ARG", "permno"] == 11111
    assert by.loc["ARG", "name_source"] == "none"
    assert len(out) == 4


def test_attach_names_takes_latest_row_on_or_before_exit(fake_conn: FakeConn) -> None:
    by = roster.attach_names(fake_conn, roster_frame(), 2, 1).set_index("symbol")
    row = by.loc["PERMNO-89260"]
    # exit 2005-06-24: the 2005-05-31 row is the latest on or before; 2005-08-31 is later.
    assert row["stock_name"] == "ALPHA CORP"
    assert row["ticker"] == "ALP"
    assert row["cusip"] == "00000001"
    assert row["exchcd"] == 1


def test_attach_names_uses_earliest_row_when_only_later_rows_exist(fake_conn: FakeConn) -> None:
    by = roster.attach_names(fake_conn, roster_frame(), 2, 1).set_index("symbol")
    row = by.loc["PERMNO-50000"]  # exit 2009-06-01, McGill row is 2010-01-29
    assert row["name_source"] == "mcgill_later"
    assert row["stock_name"] == "BETA INC"
    assert row["exchcd"] == 3


def test_attach_names_keeps_stock_absent_from_mcgill(fake_conn: FakeConn) -> None:
    by = roster.attach_names(fake_conn, roster_frame(), 2, 1).set_index("symbol")
    row = by.loc["PERMNO-777"]
    assert row["name_source"] == "none"
    assert pd.isna(row["stock_name"])


def test_attach_names_uses_parameterised_sql(fake_conn: FakeConn) -> None:
    roster.attach_names(fake_conn, roster_frame(), 2, 1)
    for sql, params in fake_conn.cur.executed:
        assert "%s" in sql
        assert params is not None
        assert "89260" not in sql and "ARG" not in sql


def test_build_roster_reads_both_files(tmp_path: Path, fake_conn: FakeConn) -> None:
    first = tmp_path / "dead-stock-first-test"
    first.mkdir()
    study_root = tmp_path / "dead-stock-data-integrity"
    study_root.mkdir()

    def payload(exits: list[dict]) -> str:
        return json.dumps({"result": {"death_price_paths": {"exits": exits}}})

    (first / "_dev-A.json").write_text(
        payload([{"symbol": "PERMNO-89260", "exit_date": "2005-06-24", "last_close": 74.98}])
    )
    (first / "_holdout-A.json").write_text(
        payload([{"symbol": "PERMNO-50000", "exit_date": "2021-06-01", "last_close": 5.0}])
    )
    out = roster.build_roster(fake_conn, study_root, mcgill_load_id=2, release_id=1)
    assert out["window"].tolist() == ["dev", "holdout"]
    assert out["symbol"].tolist() == ["PERMNO-89260", "PERMNO-50000"]
    expected = {
        "symbol",
        "permno",
        "window",
        "universe",
        "exit_date",
        "last_close",
        "stock_name",
        "ticker",
        "cusip",
        "exchcd",
        "name_source",
    }
    assert expected <= set(out.columns)
