"""Offline tests for point-in-time S&P 500 membership (committed history file)."""

import pandas as pd
import pytest

from ggTrader.data import membership
from ggTrader.data.core.index_constituents import (
    SP500_HISTORY_CSV,
    all_members_between,
    coverage_stats,
    load_sp500_history,
    normalize_yf_ticker,
    sp500_members_asof,
    universe_all_between,
    universe_members_asof,
)
from ggTrader.data.membership import FixtureMembershipSource, MembershipSourceNotSelected

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


def test_the_default_reader_uses_the_selected_source_and_has_no_file_default(monkeypatch):
    """Passing no membership reaches GGT_MEMBERSHIP; with none selected it is an error."""
    membership.clear_source_cache()
    monkeypatch.delenv("GGT_MEMBERSHIP")
    for call in (
        lambda: load_sp500_history(),
        lambda: sp500_members_asof(pd.Timestamp("2021-01-15")),
        lambda: all_members_between(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        lambda: universe_members_asof("sp500", pd.Timestamp("2021-01-15")),
        lambda: universe_all_between(
            "sp500", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")
        ),
    ):
        with pytest.raises(MembershipSourceNotSelected):
            call()
    # static universes never read the S&P 500 list
    assert universe_members_asof("nasdaq100", pd.Timestamp("2021-01-15"))


def test_an_explicit_source_or_table_wins_over_the_environment(monkeypatch):
    membership.clear_source_cache()
    monkeypatch.delenv("GGT_MEMBERSHIP")
    source = FixtureMembershipSource("sp500_v1_2026-06-02")
    when = pd.Timestamp("2021-01-15")
    expected = sp500_members_asof(when, load_sp500_history(SP500_HISTORY_CSV))
    assert sp500_members_asof(when, membership=source) == expected
    assert sp500_members_asof(when, source.history()) == expected
    assert universe_members_asof("sp500", when, membership=source) == expected
    span = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"))
    assert universe_all_between("sp500", *span, membership=source) == all_members_between(
        *span, load_sp500_history(SP500_HISTORY_CSV)
    )


def test_the_path_form_of_load_sp500_history_is_a_pure_file_reader(monkeypatch):
    """Frozen studies name their file; no source is selected for them."""
    membership.clear_source_cache()
    monkeypatch.delenv("GGT_MEMBERSHIP")
    table = load_sp500_history(SP500_HISTORY_CSV)
    assert table.equals(FixtureMembershipSource("sp500_v1_2026-06-02").history())
    with pytest.raises(FileNotFoundError):
        load_sp500_history(SP500_HISTORY_CSV.with_name("nope.csv.gz"))


def test_nothing_in_src_writes_a_membership_file():
    from ggTrader.data.core import index_constituents

    assert not hasattr(index_constituents, "download_sp500_history")
