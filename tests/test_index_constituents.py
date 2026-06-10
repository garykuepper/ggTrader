"""Offline tests for point-in-time S&P 500 membership (committed history file)."""

import pandas as pd
import pytest

from ggTrader.data.core.index_constituents import (
    SP500_HISTORY_CSV,
    all_members_between,
    coverage_stats,
    load_sp500_history,
    normalize_yf_ticker,
    sp500_members_asof,
)

pytestmark = pytest.mark.skipif(
    not SP500_HISTORY_CSV.exists(), reason="constituent history file not present"
)


def test_history_loads_and_is_sorted():
    hist = load_sp500_history()
    assert hist.index.is_monotonic_increasing
    assert hist.index[0].year == 1996
    # every snapshot holds roughly an index-sized list
    sizes = hist["tickers"].map(len)
    assert sizes.min() > 400
    assert sizes.max() < 600


def test_tsla_membership_changeover():
    """TSLA joined the S&P 500 on 2020-12-21."""
    before = sp500_members_asof(pd.Timestamp("2020-11-30"))
    after = sp500_members_asof(pd.Timestamp("2021-01-15"))
    assert "TSLA" not in before
    assert "TSLA" in after


def test_asof_predating_history_raises():
    with pytest.raises(ValueError, match="predates"):
        sp500_members_asof(pd.Timestamp("1990-01-01"))


def test_union_between_superset_of_endpoints():
    start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")
    union = all_members_between(start, end)
    assert set(sp500_members_asof(start)).issubset(union)
    assert set(sp500_members_asof(end)).issubset(union)
    assert len(union) > len(sp500_members_asof(end))  # churn adds names


def test_coverage_stats_and_ticker_normalization():
    assert normalize_yf_ticker("BRK.B") == "BRK-B"
    stats = coverage_stats(["AAPL", "BRK.B", "GONE"], ["AAPL", "BRK-B"])
    assert stats["n_members"] == 3
    assert stats["n_with_data"] == 2
    assert stats["missing"] == ["GONE"]
    assert stats["coverage_pct"] == pytest.approx(66.6667, abs=0.01)
