"""Spell assignment and streaming gap counts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.kaggle_import.spells import assign_spells, gap_row_counts

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cases():
    rows = pd.read_csv(FIXTURES / "crsp_cases.csv", parse_dates=["date"])
    calendar = pd.DatetimeIndex(pd.read_csv(FIXTURES / "crsp_calendar.csv")["date"])
    return rows, calendar


def test_consecutive_splits_never_cut(cases):
    rows, calendar = cases
    for permno in (12490, 14593, 10107):
        result = assign_spells(rows[rows.permno == permno], calendar)
        consecutive = result.gap_sessions.eq(0) & result.chained_return.notna()
        assert result.loc[consecutive, "chained_return"].equals(result.loc[consecutive, "ret"])
        assert result.spell.diff().fillna(0).loc[consecutive].eq(0).all()


def test_10066_has_four_spells(cases):
    rows, calendar = cases
    stock = rows[rows.permno == 10066]
    # The fixture omits intervening years; test each observed real break separately.
    for before, after in (
        ("1990-01-29", "1999-03-19"),
        ("2001-11-06", "2002-02-20"),
        ("2002-08-23", "2021-03-30"),
    ):
        pair = stock[stock.date.isin(pd.to_datetime([before, after]))]
        assert len(pair) == 2
        assert assign_spells(pair, calendar).spell.tolist() == [0, 1]


def test_10128_cut_at_1986_04_23(cases):
    rows, calendar = cases
    result = assign_spells(rows[rows.permno == 10128], calendar).set_index("date")
    assert result.loc["1986-04-23", "spell"] > 0
    assert pd.isna(result.loc["1986-04-23", "chained_return"])


def test_10669_bridged_with_price_return(cases):
    rows, calendar = cases
    row = assign_spells(rows[rows.permno == 10669], calendar).set_index("date").loc["1987-05-08"]
    assert row.chained_return == pytest.approx(18 / 18.25 - 1, abs=1e-9)
    assert bool(row.gap_return_approx)


def synthetic():
    calendar = pd.date_range("2020-01-01", periods=4)
    rows = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": calendar[[0, 2]],
            "prc": [10.0, 10.0],
            "ret": [0.0, 0.0],
            "shrout": [1.0, 1.0],
        }
    )
    return rows, calendar


def test_zero_shares_after_gap_starts_new_spell():
    rows, calendar = synthetic()
    rows.loc[1, "shrout"] = 0
    assert assign_spells(rows, calendar).spell.tolist() == [0, 1]


def test_nan_price_starts_new_spell():
    rows, calendar = synthetic()
    rows.loc[1, "prc"] = np.nan
    assert assign_spells(rows, calendar).spell.tolist() == [0, 1]


def test_missing_calendar_date_raises():
    rows, calendar = synthetic()
    with pytest.raises(ValueError):
        assign_spells(rows, calendar[:2])


def test_multi_stock_independent():
    rows, calendar = synthetic()
    other = rows.copy().assign(permno=2)
    result = assign_spells(pd.concat([rows, other], ignore_index=True), calendar)
    assert result.spell.tolist() == [0, 0, 0, 0]
    assert pd.isna(result.chained_return.iloc[2])


def test_gap_row_counts_on_fixture(cases):
    rows, calendar = cases
    result = assign_spells(rows, calendar)
    counts = gap_row_counts(result)
    assert (
        counts["short_rows"]
        == counts["short_cut_disagree"]
        + counts["short_cut_no_shares"]
        + result.gap_return_approx.sum()
    )


def test_streamed_counts_match_whole_frame(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.kaggle_gap_counts import count_file

    rows, calendar = synthetic()
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pandas(rows), path, row_group_size=1)
    assert count_file(path, calendar) == gap_row_counts(assign_spells(rows, calendar))
