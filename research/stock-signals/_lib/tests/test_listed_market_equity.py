"""Verify listed-class market values using invented prices and filing times."""

import listed_market_equity
import pytest

pytestmark = pytest.mark.study


def observations():
    return [
        {
            "security": "A",
            "shares": 10,
            "price": 100,
            "price_basis": "as_traded",
            "shares_asof": "2024-01-25",
            "shares_available_at": "2024-02-01T12:00:00Z",
            "price_at": "2024-02-02T21:00:00Z",
        },
        {
            "security": "C",
            "shares": 20,
            "price": 105,
            "price_basis": "as_traded",
            "shares_asof": "2024-01-25",
            "shares_available_at": "2024-02-01T12:00:00Z",
            "price_at": "2024-02-02T21:00:00Z",
        },
    ]


def test_sums_expected_classes_at_verified_cutoff():
    assert (
        listed_market_equity.calculate(observations(), {"A", "C"}, "2024-02-02T21:00:00Z") == 3100
    )


def test_rejects_split_adjusted_price_and_missing_class():
    rows = observations()
    rows[0]["price_basis"] = "split_adjusted"
    with pytest.raises(ValueError, match="basis"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")
    with pytest.raises(ValueError, match="classes"):
        listed_market_equity.calculate(observations()[:1], {"A", "C"}, "2024-02-02T21:00:00Z")


def test_rejects_information_not_yet_available():
    rows = observations()
    rows[0]["shares_available_at"] = "2024-02-03T12:00:00Z"
    with pytest.raises(ValueError, match="available"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


def test_rejects_count_published_before_count_date():
    rows = observations()
    rows[0]["shares_available_at"] = "2024-01-24T12:00:00Z"
    with pytest.raises(ValueError, match="before"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


@pytest.mark.parametrize(
    "changes",
    [
        {"price": float("nan")},
        {"shares": True},
        {"price_at": "2024-02-02"},
        {"price_at": "2024-02-02T22:00:00Z"},
        {"shares_asof": "2024-02-03"},
    ],
)
def test_rejects_invalid_or_future_inputs(changes):
    rows = observations()
    rows[0].update(changes)
    with pytest.raises(ValueError):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


def test_rejects_duplicate_class_and_different_price_timestamps():
    rows = observations()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="classes"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")
    rows = observations()
    rows[1]["price_at"] = "2024-02-01T21:00:00Z"
    with pytest.raises(ValueError, match="timestamp"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


def test_rejects_all_prices_after_decision_even_if_they_match():
    rows = observations()
    for row in rows:
        row["price_at"] = "2024-02-02T22:00:00Z"
    with pytest.raises(ValueError, match="available"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


def test_rejects_share_date_after_price_even_if_available_by_decision():
    rows = observations()
    for row in rows:
        row["price_at"] = "2024-02-01T21:00:00Z"
        row["shares_asof"] = "2024-02-02"
        row["shares_available_at"] = "2024-02-02T19:00:00Z"
    with pytest.raises(ValueError, match="Share date follows"):
        listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z")


def test_accepts_information_exactly_at_cutoff():
    rows = observations()
    for row in rows:
        row["shares_available_at"] = "2024-02-02T21:00:00Z"
    assert listed_market_equity.calculate(rows, {"A", "C"}, "2024-02-02T21:00:00Z") == 3100
