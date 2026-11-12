"""Synthetic end-to-end check for filing revisions and stock ranking."""

import expected_growth_features
import expected_growth_model
import expected_growth_training
import numpy as np
import pandas as pd
import pytest
import sec_annual_flows
import sec_facts

pytestmark = pytest.mark.study


def test_training_target_uses_only_asset_versions_available_at_fit_time():
    document = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"accn": "a", "end": "2021-12-31", "val": 100},
                            {"accn": "b", "end": "2022-12-31", "val": 120},
                            {"accn": "c", "end": "2023-12-31", "val": 132},
                            {"accn": "d", "end": "2023-12-31", "val": 144},
                        ]
                    }
                }
            }
        },
    }
    facts = sec_facts.instant_facts(
        document,
        "Assets",
        "USD",
        {
            "a": "2022-02-01T12:00:00Z",
            "b": "2023-02-01T12:00:00Z",
            "c": "2024-02-01T12:00:00Z",
            "d": "2025-02-01T12:00:00Z",
        },
        expected_cik=123,
    )

    ends = ("2021-12-31", "2022-12-31", "2023-12-31")

    def target_at(fit_at, known_year_ends=None):
        return expected_growth_training.investment_target(
            facts,
            issuer="0000000123",
            ends=ends,
            known_year_ends=known_year_ends if known_year_ends is not None else set(ends),
            known_continuity_breaks={},
            fit_at=fit_at,
        )

    assert target_at("2024-04-30T03:59:59Z") is None
    assert target_at("2024-04-30T04:00:00Z") == pytest.approx(-0.1)
    assert target_at("2025-02-01T12:00:00Z") == pytest.approx(0.0)
    assert target_at("2024-04-30T04:00:00Z") == pytest.approx(-0.1)
    known = set(ends) | {"2024-12-31"}
    assert target_at("2025-04-30T03:59:59Z", known) == pytest.approx(0.0)
    assert target_at("2025-04-30T04:00:00Z", known) is None


def test_future_asset_revision_cannot_change_earlier_cross_section_fit():
    """Connect point-in-time targets to a fit without using market returns."""
    features = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=float,
    )
    ends = ("2021-12-31", "2022-12-31", "2023-12-31")
    issuers = [f"{i:010d}" for i in range(1, len(features) + 1)]
    expected = np.array([0.10, 0.02, 0.03, 0.04])
    targets = expected[0] + features @ expected[1:]
    rows = []
    for issuer, target in zip(issuers, targets, strict=True):
        for end, value in zip(ends, (100, 100, 100 * (1 + target)), strict=True):
            rows.append(
                {
                    "issuer_id": issuer,
                    "fact_key": f'["us-gaap", "Assets", "USD", "{end}"]',
                    "accession": "initial",
                    "available_at": "2024-02-01T12:00:00Z",
                    "value": value,
                }
            )
    rows.append(
        {
            "issuer_id": issuers[0],
            "fact_key": f'["us-gaap", "Assets", "USD", "{ends[-1]}"]',
            "accession": "revision",
            "available_at": "2024-06-01T12:00:00Z",
            "value": 150,
        }
    )
    facts = pd.DataFrame(rows)

    def fit_at(cutoff):
        observations = [
            expected_growth_training.investment_target(
                facts,
                issuer=issuer,
                ends=ends,
                known_year_ends=set(ends),
                known_continuity_breaks={},
                fit_at=cutoff,
            )
            for issuer in issuers
        ]
        return expected_growth_model.fit_cross_section(
            features, observations, np.ones(len(issuers))
        )

    earlier = fit_at("2024-05-01T12:00:00Z")
    later = fit_at("2024-06-01T12:00:00Z")
    assert earlier == pytest.approx(expected)
    assert not np.allclose(later, earlier)
    assert fit_at("2024-05-01T12:00:00Z") == pytest.approx(earlier)

    months = pd.period_range("2021-10", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [month.end_time.tz_localize("UTC") for month in months],
            **dict(zip(("intercept", "log_q", "cash_flow", "roe_change"), earlier, strict=True)),
        }
    )
    revision_fit = pd.DataFrame(
        [
            {
                "month": "2024-06",
                "available_at": pd.Timestamp("2024-06-01T12:00:00Z"),
                **dict(zip(("intercept", "log_q", "cash_flow", "roe_change"), later, strict=True)),
            }
        ]
    )
    with_future = pd.concat([fits, revision_fit], ignore_index=True)

    def selected(decision, fitted):
        candidate = expected_growth_model.forecast(fitted, decision, [0, 0, 0])
        panel = pd.DataFrame(
            {
                "security": [f"reference-{i}" for i in range(1, 11)] + ["candidate"],
                "market_equity": list(range(1, 11)) + [20],
                "forecast": [0.1 + i / 1000 for i in range(1, 11)] + [candidate],
                "nyse": [True] * 10 + [False],
            }
        )
        return set(expected_growth_model.select_large_high(panel).security)

    decision = "2024-05-01T12:00:00Z"
    assert selected(decision, fits) == selected(decision, with_future)
    assert "candidate" not in selected(decision, with_future)
    assert "candidate" in selected("2024-07-01T12:00:00Z", with_future)


@pytest.mark.parametrize(
    ("start", "end"), [("2021-09-26", "2022-09-24"), ("2022-09-25", "2023-09-30")]
)
def test_annual_flow_accepts_52_and_53_week_periods(start, end):
    document = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"accn": "annual", "start": start, "end": end, "val": 10},
                        ]
                    }
                }
            }
        },
    }
    facts = sec_annual_flows.annual_flows(
        document,
        "Revenues",
        "USD",
        {"annual": "2023-11-01T12:00:00Z"},
        {(start, end)},
        expected_cik=123,
    )
    assert facts.value.tolist() == [10]


def test_missing_latest_revenue_cannot_mix_fiscal_years():
    document = {
        "cik": 123,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"accn": "old", "end": "2022-12-31", "val": 80},
                            {"accn": "asset", "end": "2023-12-31", "val": 100},
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {"accn": "old", "start": "2022-01-01", "end": "2022-12-31", "val": 40},
                            {"accn": "late", "start": "2023-01-01", "end": "2023-12-31", "val": 60},
                        ]
                    }
                },
            }
        },
    }
    times = {
        "old": "2023-02-01T12:00:00Z",
        "asset": "2024-02-01T12:00:00Z",
        "late": "2024-06-01T12:00:00Z",
    }
    assets = sec_facts.instant_facts(document, "Assets", "USD", times, expected_cik=123)
    revenue = sec_annual_flows.annual_flows(
        document,
        "Revenues",
        "USD",
        times,
        {("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")},
        expected_cik=123,
    )
    ends = {("0000000123", "2022-12-31"), ("0000000123", "2023-12-31")}
    select = sec_facts.latest_annual_instant
    decision = "2024-05-01T12:00:00Z"
    assert select(assets, decision, ends).value.tolist() == [100]
    assert select(revenue, decision, ends).empty
    assert select(revenue, "2024-06-01T12:00:00Z", ends).value.tolist() == [60]


def test_later_filing_revision_cannot_change_an_earlier_rank():
    def asset_facts(cik, revised_value):
        document = {
            "cik": cik,
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"accn": "initial", "end": "2023-09-30", "val": 100},
                                {"accn": "revision", "end": "2023-09-30", "val": revised_value},
                            ]
                        }
                    }
                }
            },
        }
        return sec_facts.instant_facts(
            document,
            "Assets",
            "USD",
            {"initial": "2023-11-01T12:00:00Z", "revision": "2024-02-01T12:00:00Z"},
            expected_cik=cik,
        )

    def predict(assets, decision):
        x = expected_growth_features.predictors(
            market_equity=100,
            long_debt=0,
            short_debt=0,
            assets=assets,
            revenue=60,
            cost_of_goods=10,
            selling_admin=10,
            research_development=0,
            receivables_change=0,
            inventory_change=0,
            prepaid_change=0,
            deferred_revenue_change=0,
            payables_change=0,
            accrued_change=0,
            income=10,
            prior_equity=100,
            year_ago_income=10,
            year_ago_prior_equity=100,
        )
        return expected_growth_model.forecast(fits, decision, x)

    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    beta = expected_growth_model.fit_cross_section(x, x[:, 1], np.ones(5))
    months = pd.period_range("2020-01", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": beta[0],
            "log_q": beta[1],
            "cash_flow": beta[2],
            "roe_change": beta[3],
        }
    )
    first = "2024-01-31T21:00:00Z"
    later = "2024-02-02T21:00:00Z"
    low_revision = asset_facts(123, 100)
    high_revision = asset_facts(123, 1000)
    annual_ends = {("0000000123", "2023-09-30")}

    def annual_assets(facts, decision):
        return sec_facts.latest_annual_instant(facts, decision, annual_ends)

    # A public filing alone does not satisfy the prototype's four-month lag.
    assert annual_assets(high_revision, "2024-01-30T04:59:59Z").empty
    assert annual_assets(high_revision, "2024-01-30T05:00:00Z").value.tolist() == [100]
    earlier_assets = annual_assets(high_revision, first).value.iloc[0]
    assert earlier_assets == annual_assets(low_revision, first).value.iloc[0] == 100
    assert predict(earlier_assets, first) > predict(200, first)
    assert predict(annual_assets(high_revision, later).value.iloc[0], later) < predict(200, later)

    def selected(facts, decision):
        assets = annual_assets(facts, decision).value.iloc[0]
        panel = pd.DataFrame(
            {
                "security": [f"reference-{i}" for i in range(1, 11)] + ["candidate"],
                "market_equity": list(range(101, 111)) + [200],
                "forecast": [i / 20 for i in range(1, 11)] + [predict(assets, decision)],
                "nyse": [True] * 10 + [False],
            }
        )
        return set(expected_growth_model.select_large_high(panel).security)

    assert selected(low_revision, first) == selected(high_revision, first)
    assert "candidate" in selected(high_revision, first)
    assert "candidate" not in selected(high_revision, later)
