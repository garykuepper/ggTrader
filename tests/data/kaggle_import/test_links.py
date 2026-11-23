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
    )
    assert result.iloc[0].status == "linked"


def test_join_returns_hash_stable_under_rescale():
    dates = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.Series(np.arange(10) + 100.0, index=dates)
    original = join_returns_hash(prices, dates[0].date(), 5)
    assert original == join_returns_hash(prices * 0.9, dates[0].date(), 5)
    prices.iloc[3] *= 1.1
    assert original != join_returns_hash(prices, dates[0].date(), 5)
