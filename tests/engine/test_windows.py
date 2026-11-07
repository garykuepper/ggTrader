"""L1: calendar folds map to bar positions exactly like label slicing; range_split wrapper."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import windows as win


class Fold(NamedTuple):
    """Same field names as lab.wfo.Fold; the engine reads them by attribute."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@pytest.fixture(scope="module")
def index() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", "2023-12-29")


def _folds(eval_start: str, eval_end: str) -> list[Fold]:
    """Same recurrence as lab.wfo.generate_folds (12m train, 3m test, 3m step)."""
    folds, cursor, end = [], pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    while True:
        train_end = cursor + pd.DateOffset(months=12)
        test_end = train_end + pd.DateOffset(months=3)
        if test_end > end:
            return folds
        folds.append(Fold(cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(months=3)


def test_slice_positions_equal_inclusive_label_slicing(index) -> None:
    rng = np.random.default_rng(0)
    days = pd.date_range("2019-12-01", "2024-02-01")
    for _ in range(200):
        a, b = sorted(rng.choice(days, 2, replace=False))
        i0, i1 = win.slice_positions(index, pd.Timestamp(a), pd.Timestamp(b))
        pd.testing.assert_index_equal(index[i0:i1], index[(index >= a) & (index <= b)])


def test_non_trading_dates_are_handled_sunday_start(index) -> None:
    sunday = pd.Timestamp("2021-01-31")
    assert sunday.day_of_week == 6
    i0, i1 = win.slice_positions(index, sunday, pd.Timestamp("2021-04-30"))
    assert index[i0] == pd.Timestamp("2021-02-01")
    assert index[i1 - 1] == pd.Timestamp("2021-04-30")


@pytest.mark.parametrize(
    ("closed", "first", "last"),
    [
        ("both", "2021-03-01", "2021-03-31"),
        ("left", "2021-03-01", "2021-03-30"),
        ("right", "2021-03-02", "2021-03-31"),
        ("neither", "2021-03-02", "2021-03-30"),
    ],
)
def test_closed_matches_pandas_semantics(index, closed, first, last) -> None:
    i0, i1 = win.slice_positions(
        index, pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-31"), closed=closed
    )
    assert (index[i0], index[i1 - 1]) == (pd.Timestamp(first), pd.Timestamp(last))


def test_slice_positions_outside_the_index_is_empty(index) -> None:
    i0, i1 = win.slice_positions(index, pd.Timestamp("2030-01-01"), pd.Timestamp("2030-06-01"))
    assert i0 == i1 == len(index)


def test_slice_positions_rejects_unsorted_index_and_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="sorted"):
        win.slice_positions(
            pd.DatetimeIndex(["2020-01-02", "2020-01-01"]),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
        )
    with pytest.raises(ValueError, match="before"):
        win.slice_positions(
            pd.bdate_range("2020-01-01", periods=5),
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-01"),
        )


def test_fold_positions_for_the_17_fold_reference_window(index) -> None:
    long_index = pd.bdate_range("2020-01-01", "2026-06-30")
    folds = _folds("2021-01-31", "2026-04-30")
    assert len(folds) == 17
    table = win.fold_positions(long_index, folds)
    assert list(table.columns) == ["train_start", "train_end", "test_start", "test_end"]
    for fold, (_, row) in zip(folds, table.iterrows(), strict=True):
        train = long_index[row.train_start : row.train_end]
        test = long_index[row.test_start : row.test_end]
        assert train.equals(
            long_index[(long_index >= fold.train_start) & (long_index <= fold.train_end)]
        )
        assert test.equals(
            long_index[(long_index >= fold.test_start) & (long_index <= fold.test_end)]
        )


# --- range_split wrapper ---------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        np.arange(60.0).reshape(30, 2),
        index=pd.bdate_range("2020-01-01", periods=30),
        columns=["a", "b"],
    )


def test_stack_windows_equals_iloc_slices() -> None:
    f = _frame()
    stacked = win.stack_windows(f, [(0, 9), (5, 11), (20, 30)])
    assert stacked.frame.columns.nlevels == 2
    assert list(stacked.frame.columns.get_level_values(0).unique()) == [0, 1, 2]
    for k, (a, b) in enumerate([(0, 9), (5, 11), (20, 30)]):
        got = stacked.frame[k].dropna(how="all")
        np.testing.assert_array_equal(got.to_numpy(), f.iloc[a:b].to_numpy())
        pd.testing.assert_index_equal(
            stacked.indexes[k], f.index[a:b], check_names=False, check_freq=False
        )


def test_stack_windows_pads_ragged_windows_with_nan() -> None:
    stacked = win.stack_windows(_frame(), [(0, 9), (5, 11)])
    assert stacked.frame.shape[0] == 9
    assert stacked.frame[1].iloc[6:].isna().all().all()


def test_stack_windows_rejects_bad_windows() -> None:
    f = _frame()
    for bad in ([(5, 5)], [(-1, 4)], [(0, 31)], [(7, 3)], []):
        with pytest.raises(ValueError):
            win.stack_windows(f, bad)


def test_stack_windows_restores_dates_for_a_column() -> None:
    f = _frame()
    stacked = win.stack_windows(f, [(2, 6)])
    series = stacked.restore(stacked.frame[0]["a"], 0)
    assert series.index.equals(f.index[2:6])
    np.testing.assert_array_equal(series.to_numpy(), f["a"].to_numpy()[2:6])
