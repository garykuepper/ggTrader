"""Tests for readable release column planning."""

import pandas as pd
import pytest

from ggTrader.data.kaggle_import.columns import plan_columns


def test_column_cases():
    spells = pd.DataFrame(
        [
            (1, 0, "1980-01-01", "2024-12-31"),
            (2, 0, "1997-01-01", "2002-01-22"),
            (2, 1, "2002-02-19", "2024-12-31"),
            (3, 0, "2007-03-30", "2018-08-22"),
            (3, 1, "2020-01-15", "2024-12-31"),
            (4, 0, "1990-01-01", "2009-03-05"),
            (4, 1, "2011-01-11", "2024-12-31"),
            (5, 0, "1990-01-01", "1995-01-01"),
            (5, 1, "1997-01-01", "2024-12-31"),
            (6, 0, "1980-01-01", "1990-01-01"),
            (6, 1, "1991-01-01", "2000-01-01"),
            (6, 2, "2001-01-01", "2010-01-01"),
            (7, 0, "1990-01-01", "2009-01-01"),
            (8, 0, "2010-01-01", "2024-12-31"),
        ],
        columns=["permno", "spell", "first_date", "last_date"],
    )
    yahoo = pd.DataFrame(
        [
            ("AAPL", "linked", 1, "2000-01-03", "2024-12-31", "2000-01-03", 0),
            ("TTWO", "linked", 2, "2000-01-03", "2024-12-31", "2000-01-03", 0),
            ("SMCI", "linked", 3, "2007-03-29", "2024-12-31", "2007-03-30", 0),
            ("VC", "linked", 4, "2010-10-05", "2024-12-31", None, None),
            ("X", "linked", 5, "2000-01-03", "2024-12-31", "2000-01-03", 1),
            ("GM", "linked", 8, "2010-01-01", "2024-12-31", "2010-01-01", 0),
            ("PX", "duplicate", 8, "2010-01-01", "2024-12-31", None, None),
        ],
        columns=[
            "symbol",
            "status",
            "permno",
            "first_yf_date",
            "last_yf_date",
            "common_day",
            "common_spell",
        ],
    )
    out = plan_columns(spells, yahoo).set_index("symbol")
    assert set(out.index) == {
        "AAPL",
        "TTWO",
        "SMCI",
        "VC",
        "X",
        "GM",
        "PX",
        "PERMNO-4~1",
        "PERMNO-5~1",
        "PERMNO-6~1",
        "PERMNO-6~2",
        "PERMNO-6",
        "PERMNO-7",
    }
    assert out.loc["SMCI", "first_date"] == pd.Timestamp("2007-03-29")
    assert out.loc["SMCI", "crsp_until"] == pd.Timestamp("2007-03-29")
    assert out.loc["VC", "kind"] == "yahoo"
    assert out.loc["PX", "possible_double"]
    yahoo.loc[yahoo.symbol.eq("VC"), "status"] = "unlinked"
    out = plan_columns(spells, yahoo).set_index("symbol")
    assert out.loc[["VC", "PERMNO-4~1", "PERMNO-4"], "possible_double"].all()
    duplicate = pd.concat([yahoo, yahoo.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        plan_columns(spells, duplicate)
    duplicate.loc[duplicate.index[-1], "symbol"] = "OTHER"
    with pytest.raises(ValueError):
        plan_columns(spells, duplicate)


def test_no_candidate_and_symbol_clash():
    spells = pd.DataFrame(
        [(10942, 0, "1986-12-01", "2007-01-31"), (10942, 1, "2011-09-26", "2013-02-04")],
        columns=["permno", "spell", "first_date", "last_date"],
    )
    yahoo = pd.DataFrame(
        [
            ("CMVT", "linked", 10942, "2000-01-03", "2024-12-31", "2000-01-03", 0),
            ("NONE", "unlinked", None, "2000-01-03", "2024-12-31", None, None),
        ],
        columns=[
            "symbol",
            "status",
            "permno",
            "first_yf_date",
            "last_yf_date",
            "common_day",
            "common_spell",
        ],
    )
    out = plan_columns(spells, yahoo).set_index("symbol")
    assert set(out.index) == {"CMVT", "NONE"}
    assert not out.loc["NONE", "possible_double"]
    yahoo.loc[0, "status"] = "unlinked"
    yahoo.loc[1, "symbol"] = "PERMNO-10942"
    with pytest.raises(ValueError, match="duplicate symbol"):
        plan_columns(spells, yahoo)
