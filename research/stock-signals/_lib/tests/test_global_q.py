"""Check expected-growth portfolio selection using invented observations."""

import pandas as pd
import pytest
from global_q import extract_expected_growth

pytestmark = pytest.mark.study


def inputs():
    """Return two months with both the desired portfolio and decoys."""
    portfolios = pd.DataFrame(
        [
            [2020, 1, 2, 3, 50, 2.0, 20.0],
            [2020, 1, 1, 3, 30, 9.0, 90.0],
            [2020, 2, 2, 3, 51, -1.0, -10.0],
            [2020, 2, 2, 1, 25, 8.0, 80.0],
        ],
        columns=["year", "month", "rank_ME", "rank_EG", "nstocks", "ret_vw", "retx_vw"],
    )
    factors = pd.DataFrame(
        [[2020, 1, 0.1, 0.5], [2020, 2, 0.2, -0.7]],
        columns=["year", "month", "R_F", "R_MKT"],
    )
    return portfolios, factors


def test_selects_total_returns_and_market_in_decimal_units():
    portfolios, factors = inputs()
    candidate, market = extract_expected_growth(portfolios, factors, "2020-01", "2020-02")
    assert candidate.tolist() == pytest.approx([0.02, -0.01])
    assert market.tolist() == pytest.approx([0.006, -0.005])
    assert candidate.index.equals(pd.to_datetime(["2020-01-31", "2020-02-29"]))
    assert candidate.index.equals(market.index)


def test_text_ranks_select_the_same_portfolio():
    portfolios, factors = inputs()
    portfolios["rank_ME"] = portfolios["rank_ME"].astype(str)
    portfolios["rank_EG"] = portfolios["rank_EG"].astype(str)
    candidate, _ = extract_expected_growth(portfolios, factors, "2020-01", "2020-02")
    assert candidate.tolist() == pytest.approx([0.02, -0.01])


@pytest.mark.parametrize("column", ["rank_ME", "rank_EG"])
def test_rejects_unrecognized_portfolio_rank(column):
    portfolios, factors = inputs()
    portfolios.loc[1, column] = 9
    with pytest.raises(ValueError, match="rank"):
        extract_expected_growth(portfolios, factors, "2020-01", "2020-02")


@pytest.mark.parametrize(
    "error",
    ["duplicate", "missing", "no_stocks", "fractional_count", "nan", "invalid_date", "total_loss"],
)
def test_rejects_unusable_candidate(error):
    portfolios, factors = inputs()
    if error == "duplicate":
        portfolios = pd.concat([portfolios, portfolios.iloc[[0]]], ignore_index=True)
    elif error == "missing":
        portfolios = portfolios.drop(index=2)
    elif error == "no_stocks":
        portfolios.loc[0, "nstocks"] = 0
    elif error == "fractional_count":
        portfolios["nstocks"] = portfolios["nstocks"].astype(float)
        portfolios.loc[0, "nstocks"] = 0.5
    elif error == "nan":
        portfolios.loc[0, "ret_vw"] = float("nan")
    elif error == "total_loss":
        portfolios.loc[0, "ret_vw"] = -100.0
    else:
        portfolios.loc[0, "month"] = 13
    with pytest.raises(ValueError):
        extract_expected_growth(portfolios, factors, "2020-01", "2020-02")


def test_rejects_backwards_window():
    portfolios, factors = inputs()
    with pytest.raises(ValueError, match="precedes"):
        extract_expected_growth(portfolios, factors, "2020-02", "2020-01")


@pytest.mark.parametrize("error", ["duplicate", "missing", "nan"])
def test_rejects_unusable_market(error):
    portfolios, factors = inputs()
    if error == "duplicate":
        factors = pd.concat([factors, factors.iloc[[0]]], ignore_index=True)
    elif error == "missing":
        factors = factors.iloc[:1]
    else:
        factors.loc[0, "R_F"] = float("nan")
    with pytest.raises(ValueError):
        extract_expected_growth(portfolios, factors, "2020-01", "2020-02")
