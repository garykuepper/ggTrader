"""Checks for return matching."""

from datetime import date

import numpy as np
import pandas as pd

from ggTrader.data.kaggle_import.links import (
    LinkSettings,
    find_links,
    join_returns_hash,
    pairwise_correlation,
)


def test_pairwise_correlation_matches_brute_force():
    rng = np.random.default_rng(42)
    y, c = rng.normal(size=(50, 4)), rng.normal(size=(50, 6))
    y[rng.random(y.shape) < 0.2] = np.nan
    c[rng.random(c.shape) < 0.2] = np.nan
    counts, corr = pairwise_correlation(y, c, 10)
    for i in range(y.shape[1]):
        for j in range(c.shape[1]):
            mask = np.isfinite(y[:, i]) & np.isfinite(c[:, j])
            assert counts[i, j] == mask.sum()
            np.testing.assert_allclose(corr[i, j], np.corrcoef(y[mask, i], c[mask, j])[0, 1])


def test_identical_returns_link():
    dates = pd.bdate_range("2020-01-01", periods=100)
    values = np.random.default_rng(1).normal(size=100)
    y = pd.DataFrame({"A": values}, index=dates)
    c = pd.DataFrame({1: values}, index=dates)
    settings = LinkSettings(80, 20, 0.99, 0.01, date(2020, 1, 1), date(2021, 1, 1), 60)
    result = find_links(
        y,
        c,
        pd.Series({"A": dates[0]}),
        pd.Series({1: dates[0]}),
        pd.Series({"A": dates[-1]}),
        pd.Series({1: dates[-1]}),
        settings,
        dates,
    )
    assert result.iloc[0].status == "linked"


def test_join_returns_hash_stable_under_rescale():
    dates = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.Series(np.arange(10) + 100.0, index=dates)
    original = join_returns_hash(prices, dates[0].date(), 5)
    assert original == join_returns_hash(prices * 0.9, dates[0].date(), 5)
    prices.iloc[3] *= 1.1
    assert original != join_returns_hash(prices, dates[0].date(), 5)


def match(
    y,
    c,
    calendar,
    *,
    minimum=50,
    short=10,
    bar=0.99,
    margin=0.01,
    yf=None,
    cf=None,
    yl=None,
    cl=None,
):
    """Run synthetic inputs with independently supplied lifetime dates."""
    yf = yf if yf is not None else pd.Series({key: y.index[0] for key in y})
    cf = cf if cf is not None else pd.Series({key: c.index[0] for key in c})
    yl = yl if yl is not None else pd.Series({key: y.index[-1] for key in y})
    cl = cl if cl is not None else pd.Series({key: c.index[-1] for key in c})
    settings = LinkSettings(minimum, short, bar, margin, y.index[0].date(), y.index[-1].date(), 60)
    return find_links(y, c, yf, cf, yl, cl, settings, calendar).set_index("symbol")


def test_noisy_first_year_still_links_on_late_window():
    dates = pd.bdate_range("2000-01-03", periods=600)
    rng = np.random.default_rng(7)
    values = rng.normal(size=len(dates))
    noisy = values.copy()
    noisy[:250] = rng.normal(size=250)
    y, c = pd.DataFrame({"A": noisy}, index=dates), pd.DataFrame({1: values}, index=dates)
    assert match(y, c, dates).loc["A", "status"] == "unlinked"
    assert match(y.iloc[250:], c.iloc[250:], dates).loc["A", "status"] == "linked"


def test_two_candidates_within_margin_is_ambiguous():
    dates = pd.bdate_range("2020-01-01", periods=100)
    values = np.random.default_rng(2).normal(size=100)
    result = match(
        pd.DataFrame({"A": values}, index=dates),
        pd.DataFrame({1: values, 2: values}, index=dates),
        dates,
    )
    assert result.loc["A", "status"] == "ambiguous"


def test_second_yahoo_symbol_for_same_permno_is_duplicate():
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(3)
    values = rng.normal(size=100)
    result = match(
        pd.DataFrame({"A": values, "B": values + rng.normal(scale=0.001, size=100)}, index=dates),
        pd.DataFrame({1: values}, index=dates),
        dates,
    )
    assert result.loc["A", "status"] == "linked"
    assert result.loc["B", "status"] == "duplicate"


def test_crsp_outliving_yahoo_is_refused():
    dates = pd.bdate_range("2020-01-01", periods=130)
    values = np.random.default_rng(4).normal(size=100)
    y, c = (
        pd.DataFrame({"A": values}, index=dates[:100]),
        pd.DataFrame({1: values}, index=dates[:100]),
    )
    assert match(y, c, dates, cl=pd.Series({1: dates[119]})).loc["A", "status"] == "linked"
    row = match(y, c, dates, cl=pd.Series({1: dates[120]})).loc["A"]
    assert (row.status, row.reason) == ("unlinked", "crsp_outlives_yahoo")


def test_start_day_match_allows_one_session_later():
    dates = pd.bdate_range("2020-01-01", periods=30)
    values = np.random.default_rng(5).normal(size=30)
    y, c = pd.DataFrame({"A": values}, index=dates), pd.DataFrame({1: values}, index=dates)
    assert match(y, c, dates, cf=pd.Series({1: dates[1]})).loc["A", "status"] == "linked"
    assert match(y, c, dates, cf=pd.Series({1: dates[2]})).loc["A", "status"] == "unlinked"


def test_short_listing_uses_min_sessions_short():
    dates = pd.bdate_range("2020-01-01", periods=30)
    values = np.random.default_rng(6).normal(size=30)
    y, c = pd.DataFrame({"A": values}, index=dates), pd.DataFrame({1: values}, index=dates)
    assert match(y, c, dates, minimum=50, short=30).loc["A", "status"] == "linked"
    assert match(y, c, dates, minimum=50, short=31).loc["A", "status"] == "unlinked"


def test_join_check_uses_first_k_returns_not_the_join_day():
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(10)
    a, b = rng.normal(0, 0.01, 100), rng.normal(0, 0.01, 100)
    b[0] = a[0]
    assert a[0] == b[0]
    closes_a = pd.Series(100 * np.cumprod(1 + a), index=dates)
    closes_b = pd.Series(100 * np.cumprod(1 + b), index=dates)
    assert join_returns_hash(closes_a, dates[0].date(), 60) != join_returns_hash(
        closes_b, dates[0].date(), 60
    )
    assert (
        match(pd.DataFrame({"A": a}, index=dates), pd.DataFrame({1: b}, index=dates), dates).loc[
            "A", "status"
        ]
        == "unlinked"
    )
