"""Synthetic checks for expected-growth accounting predictors."""

import math

import expected_growth_features as features
import pytest

pytestmark = pytest.mark.study


def test_investment_change_is_difference_between_successive_asset_growth_rates():
    assert features.investment_change(
        prior_assets=100, assets=120, next_assets=132
    ) == pytest.approx(-0.1)


@pytest.mark.parametrize("bad", [0, -1, None, True, float("nan"), float("inf"), 10**400])
def test_investment_change_rejects_invalid_assets(bad):
    for assets in ((bad, 100, 120), (100, bad, 120), (100, 120, bad)):
        with pytest.raises(ValueError):
            features.investment_change(
                **dict(zip(("prior_assets", "assets", "next_assets"), assets))
            )


def test_investment_change_rejects_nonfinite_result():
    with pytest.raises(ValueError):
        features.investment_change(prior_assets=1e-300, assets=1e300, next_assets=1e300)


def test_investment_change_requires_named_years():
    with pytest.raises(TypeError):
        features.investment_change(100, 120, 132)


def inputs():
    return dict(
        market_equity=200,
        long_debt=30,
        short_debt=10,
        assets=100,
        revenue=50,
        cost_of_goods=20,
        selling_admin=10,
        research_development=2,
        receivables_change=1,
        inventory_change=2,
        prepaid_change=3,
        deferred_revenue_change=9,
        payables_change=6,
        accrued_change=7,
        income=12,
        prior_equity=100,
        year_ago_income=8,
        year_ago_prior_equity=80,
    )


def test_known_predictors():
    assert features.predictors(**inputs()) == pytest.approx((math.log(2.4), 0.38, 0.02))


def test_optional_accounting_items_can_be_missing():
    data = inputs()
    for key in (
        "research_development",
        "receivables_change",
        "inventory_change",
        "prepaid_change",
        "deferred_revenue_change",
        "payables_change",
        "accrued_change",
    ):
        data[key] = None
    data["year_ago_income"] = None
    assert features.predictors(**data) == (pytest.approx(math.log(2.4)), 0.2, None)


@pytest.mark.parametrize(
    "key,value",
    [
        ("assets", 0),
        ("prior_equity", 0),
        ("revenue", None),
        ("income", float("nan")),
        ("market_equity", True),
        ("market_equity", -1),
        ("long_debt", -1),
        ("revenue", 10**400),
    ],
)
def test_invalid_values_are_rejected(key, value):
    data = inputs()
    data[key] = value
    with pytest.raises(ValueError):
        features.predictors(**data)


@pytest.mark.parametrize(
    "field,sign",
    [
        ("revenue", 1),
        ("cost_of_goods", -1),
        ("selling_admin", -1),
        ("research_development", 1),
        ("receivables_change", -1),
        ("inventory_change", -1),
        ("prepaid_change", -1),
        ("deferred_revenue_change", 1),
        ("payables_change", 1),
        ("accrued_change", 1),
    ],
)
def test_each_cash_flow_term_has_the_expected_sign(field, sign):
    data = inputs()
    base = features.predictors(**data)[1]
    data[field] += 1
    assert features.predictors(**data)[1] - base == pytest.approx(sign / 100)
