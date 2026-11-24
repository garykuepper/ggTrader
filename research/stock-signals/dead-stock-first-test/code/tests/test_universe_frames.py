"""Tests for the point-in-time eligibility table built from raw CRSP rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import universe_frames
from universe_frames import UniverseRule, eligibility, month_end_fields

pytestmark = pytest.mark.study

MONTH_ENDS = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-28", "2020-03-31"])
DAYS = pd.bdate_range("2020-01-02", "2020-04-30")


def make_fields(rows: list[dict]) -> pd.DataFrame:
    """Month-end rows; absent keys default to a plain NYSE common stock priced at 20."""
    base = {"shrcd": 11, "exchcd": 1, "abs_prc": 20.0, "cap": 1.0e9}
    return pd.DataFrame([{**base, **row} for row in rows])[
        ["permno", "month_end", "shrcd", "exchcd", "abs_prc", "cap"]
    ]


def make_spans(rows: list[tuple]) -> pd.DataFrame:
    """Spans rows as (symbol, permno, kind, first_date, last_date, possible_double)."""
    frame = pd.DataFrame(
        rows,
        columns=["symbol", "permno", "kind", "first_date", "last_date", "possible_double"],
    )
    frame["spell"] = 1
    frame["first_date"] = pd.to_datetime(frame["first_date"])
    frame["last_date"] = pd.to_datetime(frame["last_date"])
    return frame


def simple_spans(*permnos: int) -> pd.DataFrame:
    """One always-alive placeholder column per permno, named P<permno>."""
    return make_spans(
        [(f"P{p}", p, "placeholder", "1990-01-01", "2030-01-01", False) for p in permnos]
    )


def run(fields: pd.DataFrame, spans: pd.DataFrame, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("universe", "A")
    return eligibility(fields, spans, daily_index=DAYS, **kwargs)


def test_negative_price_uses_absolute_value() -> None:
    fields = make_fields([{"permno": 1, "month_end": m, "abs_prc": -7.0} for m in MONTH_ENDS])
    out = run(fields, simple_spans(1))
    assert out.loc["2020-02-03", "P1"]
    assert out["P1"].all()


def test_price_below_floor_is_out_and_universe_b_floor_is_one() -> None:
    fields = make_fields([{"permno": 1, "month_end": m, "abs_prc": 3.0} for m in MONTH_ENDS])
    assert not run(fields, simple_spans(1))["P1"].any()
    assert run(fields, simple_spans(1), universe="B")["P1"].all()
    cheap = make_fields([{"permno": 1, "month_end": m, "abs_prc": 0.5} for m in MONTH_ENDS])
    assert not run(cheap, simple_spans(1), universe="B")["P1"].any()


def test_rank_limit_keeps_top_n_by_cap_only() -> None:
    caps = {1: 4.0e9, 2: 3.0e9, 3: 2.0e9, 4: 1.0e9}
    fields = make_fields(
        [{"permno": p, "month_end": m, "cap": c} for p, c in caps.items() for m in MONTH_ENDS]
    )
    out = run(fields, simple_spans(1, 2, 3, 4), rule=UniverseRule(price_floor=5.0, rank_limit=2))
    assert out[["P1", "P2"]].all().all()
    assert not out[["P3", "P4"]].any().any()


def test_universe_b_has_no_rank_limit() -> None:
    caps = {p: float(10 - p) * 1e8 for p in range(1, 6)}
    fields = make_fields(
        [{"permno": p, "month_end": m, "cap": c} for p, c in caps.items() for m in MONTH_ENDS]
    )
    out = run(fields, simple_spans(*caps), universe="B")
    assert out.all().all()


def test_default_rank_limit_for_universe_a_is_one_thousand() -> None:
    rule = universe_frames.UNIVERSE_RULES["A"]
    assert (rule.price_floor, rule.rank_limit) == (5.0, 1000)
    rule_b = universe_frames.UNIVERSE_RULES["B"]
    assert (rule_b.price_floor, rule_b.rank_limit) == (1.0, None)


def test_rank_is_among_stocks_that_pass_the_other_tests() -> None:
    # A huge preferred share must not use up a place in the top-N.
    fields = make_fields(
        [{"permno": 1, "month_end": m, "cap": 9.0e9, "shrcd": 73} for m in MONTH_ENDS]
        + [{"permno": 2, "month_end": m, "cap": 2.0e9} for m in MONTH_ENDS]
        + [{"permno": 3, "month_end": m, "cap": 1.0e9} for m in MONTH_ENDS]
    )
    out = run(fields, simple_spans(1, 2, 3), rule=UniverseRule(price_floor=5.0, rank_limit=1))
    assert not out["P1"].any()
    assert out["P2"].all()
    assert not out["P3"].any()


def test_uses_previous_month_end_not_current() -> None:
    rows = [
        {"permno": 1, "month_end": "2019-12-31", "abs_prc": 4.0},
        {"permno": 1, "month_end": "2020-01-31", "abs_prc": 6.0},
        {"permno": 1, "month_end": "2020-02-28", "abs_prc": 6.0},
    ]
    out = run(make_fields(rows), simple_spans(1))
    january = out.loc["2020-01-02":"2020-01-31", "P1"]
    assert not january.any()  # the Dec 31 print (4.0) governs all of January, the 31st included
    assert out.loc["2020-02-03", "P1"]  # first trading day of February
    assert out.loc["2020-02-03":"2020-03-02", "P1"].all()


def test_exchange_change_is_dated() -> None:
    rows = [
        {"permno": 1, "month_end": "2019-12-31", "exchcd": 2},
        {"permno": 1, "month_end": "2020-01-31", "exchcd": 2},
        {"permno": 1, "month_end": "2020-02-28", "exchcd": 1},
        {"permno": 1, "month_end": "2020-03-31", "exchcd": 1},
    ]
    out = run(make_fields(rows), simple_spans(1))
    assert not out.loc[:"2020-02-28", "P1"].any()
    assert out.loc["2020-03-02":, "P1"].all()
    # exchcd 3 (Nasdaq) is accepted; 4 (other) is not
    nasdaq = make_fields([{"permno": 1, "month_end": m, "exchcd": 3} for m in MONTH_ENDS])
    assert run(nasdaq, simple_spans(1))["P1"].all()
    other = make_fields([{"permno": 1, "month_end": m, "exchcd": 4} for m in MONTH_ENDS])
    assert not run(other, simple_spans(1))["P1"].any()


def test_share_class_change_is_dated() -> None:
    rows = [
        {"permno": 1, "month_end": "2019-12-31", "shrcd": 73},
        {"permno": 1, "month_end": "2020-01-31", "shrcd": 10},
        {"permno": 1, "month_end": "2020-02-28", "shrcd": 10},
    ]
    out = run(make_fields(rows), simple_spans(1))
    assert not out.loc[:"2020-01-31", "P1"].any()
    assert out.loc["2020-02-03":"2020-03-31", "P1"].all()


def test_preferred_and_adr_codes_excluded() -> None:
    fields = make_fields(
        [{"permno": 1, "month_end": m, "shrcd": 11} for m in MONTH_ENDS]
        + [{"permno": 2, "month_end": m, "shrcd": 10} for m in MONTH_ENDS]
        + [{"permno": 3, "month_end": m, "shrcd": 73} for m in MONTH_ENDS]
        + [{"permno": 4, "month_end": m, "shrcd": 31} for m in MONTH_ENDS]
    )
    out = run(fields, simple_spans(1, 2, 3, 4))
    assert out[["P1", "P2"]].all().all()
    assert not out[["P3", "P4"]].any().any()


def test_at_most_one_column_per_permno_per_day() -> None:
    # Permno 7 is both a demoted Yahoo column and a CRSP placeholder; permno 8 is plain.
    spans = make_spans(
        [
            ("ZZZ", 7, "yahoo", "2000-01-03", "2030-01-01", True),
            ("PERMNO-7", 7, "placeholder", "1990-01-01", "2030-01-01", True),
            ("OK", 8, "linked_yahoo", "1990-01-01", "2030-01-01", False),
        ]
    )
    fields = make_fields([{"permno": p, "month_end": m} for p in (7, 8) for m in MONTH_ENDS])
    out = run(fields, spans)
    permno_of = spans.set_index("symbol")["permno"]
    per_permno = out.T.groupby(permno_of.reindex(out.columns).to_numpy()).sum()
    assert per_permno.max().max() <= 1
    assert out["PERMNO-7"].all()
    assert not out["ZZZ"].any()
    assert out["OK"].all()


def test_placeholder_wins_even_when_listed_second_or_first() -> None:
    rows = [
        ("PERMNO-7", 7, "placeholder", "1990-01-01", "2030-01-01", True),
        ("ZZZ", 7, "yahoo", "2000-01-03", "2030-01-01", True),
    ]
    fields = make_fields([{"permno": 7, "month_end": m} for m in MONTH_ENDS])
    for ordered in (rows, rows[::-1]):
        out = run(fields, make_spans(ordered))
        assert out["PERMNO-7"].all()
        assert not out["ZZZ"].any()


def test_duplicate_yields_when_placeholder_is_not_covering_the_day() -> None:
    # The placeholder ends mid-February; the Yahoo column alone covers later days.
    spans = make_spans(
        [
            ("PERMNO-7", 7, "placeholder", "1990-01-01", "2020-02-14", True),
            ("ZZZ", 7, "yahoo", "2000-01-03", "2030-01-01", True),
        ]
    )
    fields = make_fields([{"permno": 7, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert out.loc["2020-02-14", "PERMNO-7"] and not out.loc["2020-02-14", "ZZZ"]
    assert not out.loc["2020-02-17", "PERMNO-7"]
    assert (out.sum(axis=1) <= 1).all()


def test_two_spells_of_one_permno_are_each_eligible_in_their_own_dates() -> None:
    spans = make_spans(
        [
            ("PERMNO-5~1", 5, "spell", "1990-01-01", "2020-01-31", False),
            ("PERMNO-5", 5, "placeholder", "2020-02-10", "2030-01-01", False),
        ]
    )
    fields = make_fields([{"permno": 5, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert out.loc["2020-01-31", "PERMNO-5~1"]
    assert not out.loc["2020-02-03", ["PERMNO-5~1", "PERMNO-5"]].any()
    assert out.loc["2020-02-10", "PERMNO-5"]


def test_dead_after_last_date_not_eligible() -> None:
    spans = make_spans([("P1", 1, "placeholder", "1990-01-01", "2020-02-14", False)])
    fields = make_fields([{"permno": 1, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert out.loc["2020-02-14", "P1"]
    assert not out.loc["2020-02-17":, "P1"].any()


def test_stock_not_yet_listed_not_eligible() -> None:
    spans = make_spans([("P1", 1, "placeholder", "2020-02-10", "2030-01-01", False)])
    fields = make_fields([{"permno": 1, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert not out.loc[:"2020-02-07", "P1"].any()
    assert out.loc["2020-02-10", "P1"]


def test_no_stale_month_end_row_carries_forward_after_death() -> None:
    # Last row in January; nothing for February. February is covered by the January row,
    # March is not (no row for February).
    fields = make_fields(
        [
            {"permno": 1, "month_end": "2019-12-31"},
            {"permno": 1, "month_end": "2020-01-31"},
        ]
    )
    out = run(fields, simple_spans(1))
    assert out.loc["2020-02-03":"2020-02-28", "P1"].all()
    assert not out.loc["2020-03-02":, "P1"].any()


def test_symbol_without_permno_or_fields_is_never_eligible() -> None:
    spans = make_spans(
        [
            ("NOPERMNO", float("nan"), "yahoo", "1990-01-01", "2030-01-01", False),
            ("P9", 9, "placeholder", "1990-01-01", "2030-01-01", False),
        ]
    )
    fields = make_fields([{"permno": 1, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert list(out.columns) == ["NOPERMNO", "P9"]
    assert not out.any().any()


def test_output_is_boolean_on_the_requested_index() -> None:
    fields = make_fields([{"permno": 1, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, simple_spans(1))
    assert out.index.equals(DAYS)
    assert all(dtype == np.dtype(bool) for dtype in out.dtypes)


def test_unknown_universe_and_missing_columns_raise() -> None:
    fields = make_fields([{"permno": 1, "month_end": m} for m in MONTH_ENDS])
    with pytest.raises(ValueError, match="universe"):
        run(fields, simple_spans(1), universe="C")
    with pytest.raises(ValueError, match="cap"):
        run(fields.drop(columns=["cap"]), simple_spans(1))
    with pytest.raises(ValueError, match="possible_double|first_date|columns"):
        run(fields, simple_spans(1).drop(columns=["first_date"]))


def test_same_month_duplicate_rows_keep_the_latest() -> None:
    rows = [
        {"permno": 1, "month_end": "2019-12-31", "abs_prc": 2.0},
        {"permno": 1, "month_end": "2019-12-30", "abs_prc": 50.0},
        {"permno": 1, "month_end": "2020-01-31"},
    ]
    out = run(make_fields(rows), simple_spans(1))
    assert not out.loc["2020-01-02", "P1"]


class StubCursor:
    """Records the SQL and parameters and returns canned rows."""

    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple]] = []

    def __enter__(self) -> StubCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.calls.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self.rows


class StubConn:
    def __init__(self, rows: list[tuple]) -> None:
        self.cur = StubCursor(rows)

    def cursor(self) -> StubCursor:
        return self.cur


def test_month_end_fields_reads_daily_of_with_the_load_id() -> None:
    import datetime as dt

    rows = [
        (10, dt.date(2020, 1, 31), 11, 1, 7.0, 7.0e6),
        (10, dt.date(2020, 2, 28), 11, 1, 8.0, 8.0e6),
    ]
    conn = StubConn(rows)
    out = month_end_fields(conn, 3, "2019-12-01", "2020-03-31")
    sql, params = conn.cur.calls[0]
    assert "daily_of(%s)" in sql
    assert "abs(prc)" in sql.lower()
    assert params[0] == 3
    assert params[1:] == ("2019-12-01", "2020-03-31")
    assert list(out.columns) == ["permno", "month_end", "shrcd", "exchcd", "abs_prc", "cap"]
    assert pd.api.types.is_datetime64_any_dtype(out["month_end"])
    assert out["abs_prc"].tolist() == [7.0, 8.0]
    assert len(conn.cur.calls) == 1


def test_month_end_fields_empty_result_has_the_columns() -> None:
    out = month_end_fields(StubConn([]), 3, "2019-12-01", "2020-03-31")
    assert out.empty
    assert list(out.columns) == ["permno", "month_end", "shrcd", "exchcd", "abs_prc", "cap"]


def test_month_end_fields_never_writes() -> None:
    conn = StubConn([])
    month_end_fields(conn, 3, "2019-12-01", "2020-03-31")
    sql = conn.cur.calls[0][0].lower()
    for word in ("insert", "update", "delete", "drop", "create", "alter"):
        assert word not in sql.split()


def test_missing_identifier_before_duplicate_columns_keeps_one_column() -> None:
    spans = make_spans(
        [
            ("NO-ID", float("nan"), "yahoo", "1990-01-01", "2030-01-01", False),
            ("ZZZ", 7, "yahoo", "2000-01-03", "2030-01-01", True),
            ("PERMNO-7", 7, "placeholder", "1990-01-01", "2030-01-01", True),
        ]
    )
    fields = make_fields([{"permno": 7, "month_end": m} for m in MONTH_ENDS])
    out = run(fields, spans)
    assert not out["NO-ID"].any()
    assert not out["ZZZ"].any()
    assert out["PERMNO-7"].all()
    assert (out.sum(axis=1) <= 1).all()


def test_missing_month_end_price_fails_instead_of_reusing_an_older_row() -> None:
    import datetime as dt

    # The latest January row has no price and an excluded share code; an older January row
    # (price 6, common stock) must not stand in for it.
    rows = [
        (10, dt.date(2019, 12, 31), 11, 1, 20.0, 2.0e7),
        (10, dt.date(2020, 1, 31), 73, 1, None, None),
    ]
    conn = StubConn(rows)
    fields = month_end_fields(conn, 3, "2019-12-01", "2020-03-31")
    sql = conn.cur.calls[0][0].lower()
    assert "prc is not null" not in sql
    assert fields["abs_prc"].dtype == float
    assert fields["abs_prc"].isna().tolist() == [False, True]
    for universe in ("A", "B"):
        out = run(fields, simple_spans(10), universe=universe)
        assert out.loc["2020-01-02":"2020-01-31", "P10"].all()  # December row governs January
        assert not out.loc["2020-02-03":"2020-02-28", "P10"].any()  # NaN price row fails


def test_nan_price_row_fails_both_rules_even_without_rank_limit() -> None:
    nan = float("nan")
    fields = make_fields(
        [{"permno": 1, "month_end": m, "abs_prc": nan, "cap": nan} for m in MONTH_ENDS]
    )
    for universe in ("A", "B"):
        assert not run(fields, simple_spans(1), universe=universe)["P1"].any()
