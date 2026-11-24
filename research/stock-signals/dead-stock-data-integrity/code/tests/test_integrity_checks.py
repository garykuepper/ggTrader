"""Tests for CRSP row integrity checks and holding-interval overlap."""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import integrity_checks
import numpy as np
import pandas as pd
import pytest
from integrity_checks import (
    FLAG_COLUMNS,
    INTERVAL_COLUMNS,
    REFERENCE_CELL,
    check_bad_price,
    check_duplicates,
    check_gaps,
    check_shares,
    check_spikes,
    check_stale,
    flags_inside_holdings,
    holding_intervals,
    read_trades,
)

pytestmark = pytest.mark.study

ROW_COLUMNS = ["permno", "date", "prc", "ret", "shrout"]


def frame(records: list[tuple[int, str, float, float, float]]) -> pd.DataFrame:
    """Build a CRSP-like rows DataFrame from (permno, date, prc, ret, shrout) tuples."""
    df = pd.DataFrame(records, columns=ROW_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def frame_returns(permno: int, returns: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a daily rows DataFrame for one permno with given simple returns."""
    dates = pd.bdate_range(start, periods=len(returns))
    prices = [10.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1.0 + ret))
    return pd.DataFrame(
        {
            "permno": [permno] * len(returns),
            "date": dates,
            "prc": prices,
            "ret": returns,
            "shrout": [100.0] * len(returns),
        }
    )


def rows_missing(cal: pd.DatetimeIndex, permno: int, missing: slice) -> pd.DataFrame:
    """Build a stock's rows across `cal` with the slice `missing` dropped."""
    drop_dates = set(cal[missing])
    kept = [day for day in cal if day not in drop_dates]
    return pd.DataFrame(
        {
            "permno": [permno] * len(kept),
            "date": kept,
            "prc": [10.0] * len(kept),
            "ret": [0.0] * len(kept),
            "shrout": [100.0] * len(kept),
        }
    )


def constant_prices(permno: int, count: int, price: float = 10.0) -> pd.DataFrame:
    """Build `count` consecutive business-day rows at a constant price."""
    dates = pd.bdate_range("2020-01-02", periods=count)
    return pd.DataFrame(
        {
            "permno": [permno] * count,
            "date": dates,
            "prc": [price] * count,
            "ret": [0.0] * count,
            "shrout": [100.0] * count,
        }
    )


def write_gz(tmp_path: Path, text: str, name: str = "trades.csv.gz") -> Path:
    """Write UTF-8 text into a gzipped CSV file in `tmp_path`."""
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def trades_frame(records: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Build a trades DataFrame from (cell, date, symbol, qty) tuples."""
    df = pd.DataFrame(records, columns=["cell", "date", "symbol", "qty"])
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = 10.0
    df["reason"] = "rebalance"
    return df


def flag(permno: int, date: str, check: str = "spike", detail: str = "test") -> pd.DataFrame:
    """Build a single-row flag DataFrame matching the check output schema."""
    return pd.DataFrame(
        [
            {
                "check": check,
                "permno": permno,
                "date": pd.Timestamp(date),
                "detail": detail,
            }
        ],
        columns=FLAG_COLUMNS,
    )


# --- Brief-specified tests ---


def test_duplicates_flag_second_copy() -> None:
    rows = frame([(1, "2020-01-02", 10.0, 0.0, 5.0), (1, "2020-01-02", 10.0, 0.0, 5.0)])
    out = check_duplicates(rows)
    assert len(out) == 1
    assert list(out.columns) == FLAG_COLUMNS
    assert out.iloc[0]["permno"] == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2020-01-02")


def test_spike_that_reverses_is_flagged_and_real_jump_is_not() -> None:
    up_down = frame_returns(1, [0.0, 0.8, -0.44])  # 1.8 * 0.56 = 1.008
    up_stay = frame_returns(2, [0.0, 0.8, 0.0])  # stays up
    flagged = check_spikes(pd.concat([up_down, up_stay], ignore_index=True))
    assert flagged["permno"].tolist() == [1]
    assert flagged.iloc[0]["date"] == up_down.iloc[1]["date"]


def test_gap_of_five_flagged_gap_of_four_not() -> None:
    cal = pd.bdate_range("2020-01-01", periods=30)
    five = rows_missing(cal, permno=1, missing=slice(10, 15))
    four = rows_missing(cal, permno=2, missing=slice(10, 14))
    out = check_gaps(pd.concat([five, four], ignore_index=True), cal)
    assert out["permno"].tolist() == [1]
    assert out.iloc[0]["detail"] == "5 days missing"
    assert out.iloc[0]["date"] == cal[10]


def test_stale_run_of_ten_flagged_nine_not() -> None:
    ten = check_stale(constant_prices(1, 10))
    nine = check_stale(constant_prices(2, 9))
    assert len(ten) == 1
    assert ten.iloc[0]["date"] == pd.Timestamp("2020-01-02")
    assert len(nine) == 0
    assert list(nine.columns) == FLAG_COLUMNS


def test_negative_price_is_not_a_bad_price() -> None:
    rows = frame([(1, "2020-01-02", -10.0, 0.01, 5.0)])
    out = check_bad_price(rows)
    assert out.empty
    assert list(out.columns) == FLAG_COLUMNS


def test_zero_price_with_a_return_is_flagged() -> None:
    rows = frame([(1, "2020-01-02", 0.0, 0.01, 5.0)])
    assert len(check_bad_price(rows)) == 1


def test_missing_shares_on_month_end_flagged() -> None:
    rows = frame([(1, "2020-01-30", 10.0, 0.0, np.nan), (1, "2020-01-31", 10.0, 0.0, 0.0)])
    assert check_shares(rows)["date"].dt.day.tolist() == [31]


def test_read_trades_parses_numpy_text(tmp_path: Path) -> None:
    path = write_gz(
        tmp_path,
        "cell,date,symbol,qty,price,reason\nc,2020-01-02,PERMNO-7,np.float64(10.5),3.0,rebalance\n",
    )
    df = read_trades(path)
    assert df["qty"].iloc[0] == 10.5
    assert pd.api.types.is_float_dtype(df["qty"])


def test_holding_interval_and_flag_inside() -> None:
    trades = trades_frame(
        [("c", "2020-01-02", "PERMNO-7", 10.0), ("c", "2020-02-03", "PERMNO-7", -10.0)]
    )
    iv = holding_intervals(trades, "c")
    assert iv.iloc[0]["start"] == pd.Timestamp("2020-01-02")
    assert iv.iloc[0]["end"] == pd.Timestamp("2020-02-03")
    inside = flags_inside_holdings(flag(7, "2020-01-15"), iv)
    outside = flags_inside_holdings(flag(7, "2020-03-01"), iv)
    assert len(inside) == 1 and outside.empty


# --- Additional edge-case and contract tests ---


def test_bad_price_null_with_return_flagged_and_null_return_ignored() -> None:
    rows = frame(
        [
            (1, "2020-01-02", np.nan, 0.02, 5.0),
            (2, "2020-01-02", np.nan, np.nan, 5.0),
            (3, "2020-01-02", 0.0, np.nan, 5.0),
        ]
    )
    out = check_bad_price(rows)
    assert out["permno"].tolist() == [1]


def test_down_spike_reversing_flagged_and_no_cross_stock_bleed() -> None:
    # Stock 1 drops 60% then rebounds 150%: 0.4 * 2.5 = 1.0 -> reversal spike.
    down_up = frame_returns(1, [0.0, -0.6, 1.5])
    # Stock 2 ends on +80% and stock 3 starts on -44%: must not pair across stocks.
    s2 = frame_returns(2, [0.0, 0.8])
    s3 = frame_returns(3, [-0.44, 0.0])
    out = check_spikes(pd.concat([down_up, s2, s3], ignore_index=True))
    assert out["permno"].tolist() == [1]
    assert out.iloc[0]["date"] == down_up.iloc[1]["date"]


def test_gaps_ignore_dates_outside_stock_span_and_catch_multiple_gaps() -> None:
    cal = pd.bdate_range("2020-01-01", periods=40)
    # Stock starts at index 10 and ends at index 30, with a 6-day gap inside (15..20).
    stock_dates = list(cal[10:15]) + list(cal[21:31])
    rows = pd.DataFrame(
        {
            "permno": [1] * len(stock_dates),
            "date": stock_dates,
            "prc": [10.0] * len(stock_dates),
            "ret": [0.0] * len(stock_dates),
            "shrout": [100.0] * len(stock_dates),
        }
    )
    out = check_gaps(rows, cal)
    assert len(out) == 1
    assert out.iloc[0]["date"] == cal[15]
    assert out.iloc[0]["detail"] == "6 days missing"


def test_stale_uses_abs_price_and_ignores_nans_and_stock_boundaries() -> None:
    # Alternating sign around 10.0 is still the same absolute price for 10 rows.
    signed = constant_prices(1, 10, price=10.0)
    signed.loc[signed.index % 2 == 1, "prc"] = -10.0
    # 12 NaN prices must never flag as stale.
    nans = constant_prices(2, 12, price=np.nan)
    # Two different stocks each with 5 identical rows must not combine into 10.
    half_a = constant_prices(3, 5, price=20.0)
    half_b = constant_prices(4, 5, price=20.0)
    out = check_stale(pd.concat([signed, nans, half_a, half_b], ignore_index=True))
    assert out["permno"].tolist() == [1]


def test_shares_allows_negative_valid_price_and_ignores_missing_price() -> None:
    rows = frame(
        [
            # Negative price is a valid CRSP bid-ask midpoint -> missing shares must flag.
            (1, "2020-01-31", -12.5, 0.0, np.nan),
            # Missing price on month-end -> not a valid price, so shares check skips it.
            (2, "2020-01-31", np.nan, 0.0, np.nan),
            # Positive price and positive shares -> clean.
            (3, "2020-01-31", 12.5, 0.0, 50.0),
        ]
    )
    out = check_shares(rows)
    assert out["permno"].tolist() == [1]


def test_read_trades_handles_plain_and_negative_numpy_qty(tmp_path: Path) -> None:
    path = write_gz(
        tmp_path,
        "cell,date,symbol,qty,price,reason\n"
        "c,2020-01-02,PERMNO-7,np.float64(-10.5),3.0,rebalance\n"
        "c,2020-01-03,PERMNO-7,15.25,3.5,rebalance\n",
    )
    df = read_trades(path)
    assert df["qty"].tolist() == pytest.approx([-10.5, 15.25])


def test_holding_intervals_reentry_partial_trim_non_permno_and_boundaries() -> None:
    trades = trades_frame(
        [
            ("other", "2020-01-02", "PERMNO-7", 10.0),
            ("c", "2020-01-02", "SPY", 100.0),
            ("c", "2020-01-02", "PERMNO-7", 10.0),
            ("c", "2020-01-10", "PERMNO-7", -4.0),  # partial trim -> still open
            ("c", "2020-01-20", "PERMNO-7", -6.0),  # closed within 1e-9
            ("c", "2020-03-02", "PERMNO-7", 5.0),  # re-entry
            ("c", "2020-03-10", "PERMNO-7", -5.0 + 1e-12),  # closed with float residue
        ]
    )
    iv = holding_intervals(trades, "c")
    assert list(iv.columns) == INTERVAL_COLUMNS
    assert len(iv) == 2
    assert iv["start"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-03-02")]
    assert iv["end"].tolist() == [pd.Timestamp("2020-01-20"), pd.Timestamp("2020-03-10")]

    # Boundary dates (start and end) count as inside; date between intervals does not.
    candidate_flags = pd.concat(
        [
            flag(7, "2020-01-02"),
            flag(7, "2020-01-20"),
            flag(7, "2020-02-15"),
            flag(7, "2020-03-05"),
            flag(8, "2020-01-10"),
        ],
        ignore_index=True,
    )
    matched = flags_inside_holdings(candidate_flags, iv)
    assert matched["date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-20"),
        pd.Timestamp("2020-03-05"),
    ]


def test_checks_do_not_mutate_input_and_return_empty_frame_on_empty_input() -> None:
    cal = pd.bdate_range("2020-01-01", periods=5)
    empty_rows = pd.DataFrame(columns=ROW_COLUMNS)
    empty_rows["date"] = pd.to_datetime(empty_rows["date"])

    for result in (
        check_duplicates(empty_rows),
        check_bad_price(empty_rows),
        check_spikes(empty_rows),
        check_gaps(empty_rows, cal),
        check_stale(empty_rows),
        check_shares(empty_rows),
    ):
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == FLAG_COLUMNS

    # Unsorted input must not be reordered or mutated in place.
    unsorted = frame(
        [
            (2, "2020-01-03", 10.0, 0.0, 5.0),
            (1, "2020-01-03", 10.0, 0.0, 5.0),
            (1, "2020-01-02", 10.0, 0.0, 5.0),
        ]
    )
    snapshot = unsorted.copy(deep=True)
    check_duplicates(unsorted)
    check_bad_price(unsorted)
    check_spikes(unsorted)
    check_gaps(unsorted, cal)
    check_stale(unsorted)
    check_shares(unsorted)
    pd.testing.assert_frame_equal(unsorted, snapshot)

    assert REFERENCE_CELL == "loss0_cost18.3bps_taxlow"
    assert integrity_checks.REFERENCE_CELL == REFERENCE_CELL


# --- Review-finding regression tests ---


def test_flags_inside_holdings_bounded_many_intervals_and_flags() -> None:
    # 500 disjoint 2-day intervals every 4 days across 2,000 days for one permno,
    # plus overlapping/nested intervals to verify prefix-max handling and deduplication.
    days = pd.date_range("2000-01-01", periods=2000, freq="D")
    starts = days[0::4]
    ends = days[1::4]
    assert len(starts) == 500

    intervals = pd.DataFrame(
        {
            "permno": np.full(len(starts), 42, dtype="int64"),
            "start": starts,
            "end": ends,
        },
        columns=INTERVAL_COLUMNS,
    )
    # Shuffle interval order and add a nested duplicate inside the first interval.
    nested = pd.DataFrame(
        [{"permno": 42, "start": days[0], "end": days[0]}],
        columns=INTERVAL_COLUMNS,
    )
    intervals = pd.concat([intervals.iloc[::-1], nested], ignore_index=True)

    # 2,000 flags on permno 42 (one per day, reversed order) plus 5 on an unheld permno.
    permno_42_flags = pd.DataFrame(
        {
            "check": ["spike"] * len(days),
            "permno": np.full(len(days), 42, dtype="int64"),
            "date": days[::-1],
            "detail": [f"day-{i}" for i in range(len(days) - 1, -1, -1)],
            "extra_col": np.arange(len(days)),
        }
    )
    other_flags = pd.DataFrame(
        {
            "check": ["spike"] * 5,
            "permno": np.full(5, 99, dtype="int64"),
            "date": days[:5],
            "detail": ["other"] * 5,
            "extra_col": np.arange(5),
        }
    )
    all_flags = pd.concat([permno_42_flags, other_flags], ignore_index=True)

    t0 = time.perf_counter()
    matched = flags_inside_holdings(all_flags, intervals)
    elapsed = time.perf_counter() - t0

    # Days with index % 4 in {0, 1} lie inside [4k, 4k+1]; reversed input order must hold.
    expected_indices = [i for i in range(len(days) - 1, -1, -1) if i % 4 in (0, 1)]
    assert len(matched) == 1000
    assert elapsed < 1.0
    assert list(matched.columns) == list(all_flags.columns)
    assert matched["permno"].eq(42).all()
    assert matched["date"].tolist() == [days[i] for i in expected_indices]
    assert matched["detail"].tolist() == [f"day-{i}" for i in expected_indices]


def test_identical_duplicates_do_not_create_false_stale_or_spike() -> None:
    # Nine distinct unchanged dates plus one identical repeat (10 rows total) must not flag stale.
    nine_days = constant_prices(1, 9, price=10.0)
    stale_with_dup = pd.concat([nine_days, nine_days.iloc[[4]]], ignore_index=True)
    assert check_stale(stale_with_dup).empty
    assert len(check_duplicates(stale_with_dup)) == 1

    # Two identical copies of ret=-2.0 would give (1 - 2) * (1 - 2) = 1.0 without dedupe.
    spike_dup = frame(
        [
            (2, "2020-01-02", 10.0, 0.0, 50.0),
            (2, "2020-01-03", 10.0, -2.0, 50.0),
            (2, "2020-01-03", 10.0, -2.0, 50.0),
            (2, "2020-01-06", 10.0, 0.0, 50.0),
        ]
    )
    assert check_spikes(spike_dup).empty
    assert len(check_duplicates(spike_dup)) == 1

    # A genuine spike whose spike row is duplicated identically still flags exactly once.
    genuine = frame_returns(3, [0.0, 0.8, -0.44])
    genuine_with_dup = pd.concat([genuine, genuine.iloc[[1]]], ignore_index=True)
    flagged = check_spikes(genuine_with_dup)
    assert len(flagged) == 1
    assert flagged.iloc[0]["permno"] == 3
    assert flagged.iloc[0]["date"] == genuine.iloc[1]["date"]
    assert len(check_duplicates(genuine_with_dup)) == 1


def test_conflicting_duplicates_produce_no_false_spike_or_stale() -> None:
    # Two copies of the same date with ret=0.8 and ret=-0.44 must not form a two-day reversal.
    spike_conflict = frame(
        [
            (1, "2020-01-02", 10.0, 0.0, 100.0),
            (1, "2020-01-03", 18.0, 0.8, 100.0),
            (1, "2020-01-03", 10.08, -0.44, 100.0),
            (1, "2020-01-06", 10.0, 0.0, 100.0),
        ]
    )
    assert check_spikes(spike_conflict).empty
    assert len(check_duplicates(spike_conflict)) == 1

    # Nine unchanged dates plus a 10th date with conflicting shrout drops the 10th date.
    ten_days = constant_prices(2, 10, price=10.0)
    conflict_copy = ten_days.iloc[[9]].copy()
    conflict_copy["shrout"] = 200.0
    stale_conflict = pd.concat([ten_days, conflict_copy], ignore_index=True)
    assert check_stale(stale_conflict).empty
    assert len(check_duplicates(stale_conflict)) == 1


def test_conflicting_duplicate_date_does_not_create_false_gap() -> None:
    cal = pd.bdate_range("2020-01-01", periods=20)
    # Missing 4 days (cal[5:9]) is below min_days=5; cal[9] has two conflicting copies.
    base = rows_missing(cal, permno=1, missing=slice(5, 9))
    boundary_row = base[base["date"] == cal[9]].copy()
    boundary_row["prc"] = 25.0
    boundary_row["ret"] = 0.5
    rows = pd.concat([base, boundary_row], ignore_index=True)

    # Dropping cal[9] would widen the missing stretch to 5 days; keeping it avoids a false gap.
    assert check_gaps(rows, cal).empty
    dups = check_duplicates(rows)
    assert len(dups) == 1
    assert dups.iloc[0]["date"] == cal[9]


def test_stale_run_longer_than_ten_flags_once_at_first_date() -> None:
    rows = constant_prices(1, 15, price=12.5)
    out = check_stale(rows)
    assert len(out) == 1
    assert out.iloc[0]["permno"] == 1
    assert out.iloc[0]["date"] == rows.iloc[0]["date"]
    assert out.iloc[0]["detail"] == "15 rows at 12.5"


def test_holding_intervals_open_ended_position_and_flags() -> None:
    trades = trades_frame(
        [
            ("c", "2020-01-02", "PERMNO-7", 10.0),
            ("c", "2020-01-20", "PERMNO-7", -10.0),  # closed first interval
            ("c", "2020-02-03", "PERMNO-7", 5.0),  # re-entry left open
            ("c", "2020-02-10", "PERMNO-7", -2.0),  # partial trim, still open (+3.0)
        ]
    )
    iv = holding_intervals(trades, "c")
    assert len(iv) == 2
    assert iv.iloc[1]["start"] == pd.Timestamp("2020-02-03")
    assert pd.isna(iv.iloc[1]["end"])

    candidate_flags = pd.concat(
        [
            flag(7, "2020-01-01"),  # before first interval
            flag(7, "2020-01-25"),  # gap between closed and open intervals
            flag(7, "2020-02-03"),  # exact start of open interval
            flag(7, "2025-12-31"),  # far future, still inside open-ended interval
        ],
        ignore_index=True,
    )
    matched = flags_inside_holdings(candidate_flags, iv)
    assert matched["date"].tolist() == [
        pd.Timestamp("2020-02-03"),
        pd.Timestamp("2025-12-31"),
    ]


def test_spikes_nan_return_single_row_and_unsorted_dates() -> None:
    # A NaN return between +0.8 and -0.44 breaks the adjacent pair; leading NaN does not.
    nan_between = frame_returns(1, [0.0, 0.8, np.nan, -0.44])
    nan_before = frame_returns(2, [np.nan, 0.8, -0.44])
    # Single-row stock with a large return has no next day to reverse.
    single_row = frame([(3, "2020-01-02", 18.0, 0.8, 100.0)])
    # Unsorted chronological dates for permno 4 must still flag the true spike day (day 1).
    unsorted_spike = frame(
        [
            (4, "2020-01-06", 10.08, -0.44, 100.0),
            (4, "2020-01-02", 10.0, 0.0, 100.0),
            (4, "2020-01-03", 18.0, 0.8, 100.0),
        ]
    )

    combined = pd.concat([nan_between, nan_before, single_row, unsorted_spike], ignore_index=True)
    snapshot = combined.copy(deep=True)
    flagged = check_spikes(combined)

    pd.testing.assert_frame_equal(combined, snapshot)
    assert flagged["permno"].tolist() == [2, 4]
    assert flagged["date"].tolist() == [
        nan_before.iloc[1]["date"],
        pd.Timestamp("2020-01-03"),
    ]
