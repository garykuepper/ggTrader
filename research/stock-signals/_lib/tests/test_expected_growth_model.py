"""Synthetic checks for the expected-growth calculation primitives."""

import warnings

import expected_growth_model as model
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.study


def test_large_high_selection_uses_nyse_breakpoints_and_value_weights():
    panel = pd.DataFrame(
        {
            "security": [f"N{i}" for i in range(1, 11)] + ["other"],
            "market_equity": list(range(1, 11)) + [20],
            "forecast": list(range(1, 11)) + [11],
            "nyse": [True] * 10 + [False],
        }
    )
    result = model.select_large_high(panel).set_index("security")
    assert set(result.index) == {"N8", "N9", "N10", "other"}
    assert result.loc["other", "weight"] == pytest.approx(20 / 47)
    assert result.weight.sum() == pytest.approx(1)


def test_large_high_excludes_equal_breakpoints_and_rejects_duplicate_security():
    panel = pd.DataFrame(
        {
            "security": ["a", "b", "c", "d", "e", "f"],
            "market_equity": [1, 2, 3, 3, 2, 3],
            "forecast": [1, 2, 3, 3, 3, 2.4],
            "nyse": [True, True, True, False, False, False],
        }
    )
    assert set(model.select_large_high(panel).security) == {"c", "d"}
    panel.loc[3, "security"] = "c"
    with pytest.raises(ValueError, match="security"):
        model.select_large_high(panel)


def test_large_high_rejects_no_nyse_reference_and_bad_forecast():
    panel = pd.DataFrame(
        {
            "security": ["a", "b"],
            "market_equity": [1, 2],
            "forecast": [1.0, float("nan")],
            "nyse": [True, True],
        }
    )
    with pytest.raises(ValueError, match="inputs"):
        model.select_large_high(panel)
    panel["forecast"] = [1, 2]
    panel["nyse"] = [False, False]
    with pytest.raises(ValueError, match="NYSE"):
        model.select_large_high(panel)


@pytest.mark.parametrize("bad", [None, 1, "NYSE"])
def test_large_high_rejects_missing_or_nonboolean_exchange(bad):
    panel = pd.DataFrame(
        {"security": ["a", "b"], "market_equity": [1, 2], "forecast": [1, 2], "nyse": [True, bad]}
    )
    with pytest.raises(ValueError, match="exchange"):
        model.select_large_high(panel)


@pytest.mark.parametrize("bad", [True, "12"])
def test_large_high_rejects_non_numeric_market_equity(bad):
    panel = pd.DataFrame(
        {
            "security": ["a", "b"],
            "market_equity": [1, bad],
            "forecast": [1, 2],
            "nyse": [True, True],
        }
    )
    with pytest.raises(ValueError, match="inputs"):
        model.select_large_high(panel)


def test_forecast_rejects_empty_input():
    with pytest.raises(ValueError, match="30"):
        model.forecast(pd.DataFrame(), "2025-01-01T00:00:00Z", [0, 0, 0])


def test_forecast_rejects_overflow():
    months = pd.period_range("2020-01", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": "2022-06-30T23:59:59Z",
            "intercept": 1e308,
            "log_q": 1e308,
            "cash_flow": 0,
            "roe_change": 0,
        }
    )
    with pytest.raises(ValueError, match="Non-finite"):
        model.forecast(fits, "2022-07-01T00:00:00Z", [1e308, 0, 0])


def test_weighted_cross_section_recovers_known_coefficients():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    y = 2 + x @ np.array([1, 2, -1])
    result = model.fit_cross_section(x, y, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(result, [2, 1, 2, -1], atol=1e-12)


@pytest.mark.parametrize("field", ["features", "target", "weights"])
@pytest.mark.parametrize("bad", [True, 1 + 2j, "1"])
def test_weighted_cross_section_rejects_nonreal_input(field, bad):
    inputs = {
        "features": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        "target": [1, 2, 3, 4, 5],
        "weights": [1, 1, 1, 1, 1],
    }
    if field == "features":
        inputs[field][0][0] = bad
    else:
        inputs[field][0] = bad
    with pytest.raises(ValueError, match="regression inputs"):
        model.fit_cross_section(inputs["features"], inputs["target"], inputs["weights"])


@pytest.mark.parametrize("field", ["coefficient", "feature"])
@pytest.mark.parametrize("bad", [True, 1 + 2j, "1"])
def test_forecast_rejects_nonreal_input(field, bad):
    months = pd.period_range("2021-01", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": [1] * 30,
            "log_q": [0] * 30,
            "cash_flow": [0] * 30,
            "roe_change": [0] * 30,
        }
    )
    features = [0, 0, 0]
    if field == "coefficient":
        fits["intercept"] = fits["intercept"].astype(object)
        fits.loc[0, "intercept"] = bad
    else:
        features[0] = bad
    with pytest.raises(ValueError, match="forecast inputs"):
        model.forecast(fits, "2023-07-01T00:00:00Z", features)


def test_weighted_cross_section_gives_large_company_more_influence():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    y = np.array([0, 0, 0, 0, 10], dtype=float)
    result = model.fit_cross_section(x, y, np.array([1, 1, 1, 1, 100]))
    np.testing.assert_allclose(result, np.array([-2000, 3000, 3000, 3000]) / 701)


def test_weighted_cross_section_rejects_nearly_collinear_inputs():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 3))
    x[:, 2] = x[:, 1] + 1e-10 * rng.normal(size=100)
    with pytest.raises(ValueError, match="condition"):
        model.fit_cross_section(x, rng.normal(size=100), np.ones(100))


def test_weighted_cross_section_is_invariant_to_feature_units():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    scales = np.array([1e-9, 1e7, 1.0])
    y = 2 + x @ np.array([1, 2, -1])
    result = model.fit_cross_section(x * scales, y, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(result[0], 2, atol=1e-12)
    np.testing.assert_allclose(result[1:] * scales, [1, 2, -1], atol=1e-12)


def test_unrepresentable_coefficient_is_a_value_error_even_with_strict_warnings():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    y = x[:, 0].copy()
    x[:, 0] *= 1e-310
    with warnings.catch_warnings(), pytest.raises(ValueError, match="Non-finite"):
        warnings.simplefilter("error", RuntimeWarning)
        model.fit_cross_section(x, y, np.ones(5))


@pytest.mark.parametrize("problem", ["zero_weight", "nan", "singular"])
def test_weighted_cross_section_rejects_invalid_inputs(problem):
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    y, weights = np.ones(5), np.ones(5)
    if problem == "zero_weight":
        weights[0] = 0
    elif problem == "nan":
        y[0] = float("nan")
    else:
        x[:, 2] = x[:, 1]
    with pytest.raises(ValueError):
        model.fit_cross_section(x, y, weights)


def test_forecast_uses_only_prior_available_months():
    months = pd.period_range("2021-01", periods=31, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": [1.0] * 30 + [1000.0],
            "log_q": [2.0] * 31,
            "cash_flow": [3.0] * 31,
            "roe_change": [4.0] * 31,
        }
    )
    assert model.forecast(fits, "2023-07-01T00:00:00Z", [1, 2, 3]) == pytest.approx(21.0)
    fits.loc[30, "intercept"] = -999.0
    assert model.forecast(fits, "2023-07-01T00:00:00Z", [1, 2, 3]) == pytest.approx(21.0)


def test_forecast_requires_thirty_available_months_and_unique_months():
    months = pd.period_range("2021-01", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": 1.0,
            "log_q": 0.0,
            "cash_flow": 0.0,
            "roe_change": 0.0,
        }
    )
    with pytest.raises(ValueError, match="30"):
        model.forecast(fits.iloc[:-1], "2023-07-01T00:00:00Z", [0, 0, 0])
    with pytest.raises(ValueError, match="duplicate"):
        model.forecast(pd.concat([fits, fits.iloc[[0]]]), "2023-07-01T00:00:00Z", [0, 0, 0])


def test_forecast_includes_a_fit_only_at_its_exact_availability():
    months = pd.period_range("2021-10", periods=30, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": 1.0,
            "log_q": 0.0,
            "cash_flow": 0.0,
            "roe_change": 0.0,
        }
    )
    cutoff = pd.Timestamp("2024-04-01T12:00:00Z")
    fits.loc[29, "available_at"] = cutoff
    with pytest.raises(ValueError, match="30"):
        model.forecast(fits, cutoff - pd.Timedelta(1, unit="ns"), [0, 0, 0])
    assert model.forecast(fits, cutoff, [0, 0, 0]) == pytest.approx(1.0)


def test_forecast_excludes_stale_and_not_yet_available_fits():
    months = pd.period_range("2013-01", periods=151, freq="M")
    fits = pd.DataFrame(
        {
            "month": months.astype(str),
            "available_at": [m.end_time.tz_localize("UTC") for m in months],
            "intercept": [500.0] * 30 + [2.0] * 120 + [500.0],
            "log_q": 0.0,
            "cash_flow": 0.0,
            "roe_change": 0.0,
        }
    )
    fits.loc[149, "available_at"] = pd.Timestamp("2025-08-01T00:00:00Z")
    assert model.forecast(fits, "2025-07-01T00:00:00Z", [0, 0, 0]) == pytest.approx(2.0)
