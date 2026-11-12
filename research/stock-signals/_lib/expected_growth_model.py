"""Numerical primitives for an experimental expected-growth forecast."""

import re
from numbers import Real

import numpy as np
import pandas as pd
from filing_vintages import _utc

_COEFFICIENTS = ("intercept", "log_q", "cash_flow", "roe_change")


def _real_array(value: object) -> np.ndarray:
    """Reject values that would be silently coerced before numerical checks."""
    data = np.asarray(value, dtype=object)
    if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in data.flat):
        raise ValueError("Expected real numeric inputs")
    return np.asarray(data, dtype=float)


def select_large_high(panel: pd.DataFrame) -> pd.DataFrame:
    """Select large high-forecast stocks using an experimental NYSE reference sort.

    The supplied panel must include the full eligible NYSE reference universe, not
    merely the desired trading list. Two NYSE rows are a computational minimum,
    not evidence of adequate coverage. The caller verifies identity, share-class
    treatment, eligibility, primary exchange, point-in-time market equity and
    forecasts. The ``nyse`` flag means NYSE is the primary exchange, not merely
    one exchange on which the security is registered; this helper cannot verify it.
    The same supplied market equity is used for both sorting and security weighting;
    its treatment of multi-class companies must be validated before use. Linear
    interpolation and strict greater-than ties are unverified provider assumptions.
    This does not validate input dates or produce trading orders.
    """
    required = ["security", "market_equity", "forecast", "nyse"]
    if (
        not isinstance(panel, pd.DataFrame)
        or panel.columns.has_duplicates
        or not set(required).issubset(panel.columns)
        or panel.empty
    ):
        raise ValueError("Invalid selection inputs")
    data = panel.loc[:, required].copy()
    if (
        not data.security.map(lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all()
        or data.security.duplicated().any()
        or not data.nyse.map(lambda x: isinstance(x, (bool, np.bool_))).all()
    ):
        raise ValueError("Invalid security or exchange")
    if not all(
        data[column]
        .map(lambda x: isinstance(x, Real) and not isinstance(x, (bool, np.bool_)))
        .all()
        for column in ("market_equity", "forecast")
    ):
        raise ValueError("Invalid selection inputs")
    try:
        values = data[["market_equity", "forecast"]].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid selection inputs") from exc
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any():
        raise ValueError("Invalid selection inputs")
    reference = values[data.nyse.to_numpy(dtype=bool)]
    if len(reference) < 2:
        raise ValueError("Insufficient NYSE reference stocks")
    size_cutoff = np.quantile(reference[:, 0], 0.5)
    growth_cutoff = np.quantile(reference[:, 1], 0.7)
    selected = (values[:, 0] > size_cutoff) & (values[:, 1] > growth_cutoff)
    result = data.loc[selected, ["security"]].copy()
    market = values[selected, 0]
    if len(market):
        scaled = market / market.max()
        result["weight"] = scaled / scaled.sum()
    else:
        result["weight"] = pd.Series(index=result.index, dtype=float)
    return result


def fit_cross_section(features: object, target: object, weights: object) -> np.ndarray:
    """Fit an intercept and three slopes using positive market-value weights.

    The published method uses market equity as weights (Hou et al., 2021,
    https://theinvestmentcapm.com/uploads/1/2/2/6/122679606/houmoxuezhang2020rf.pdf).
    Inputs must already be aligned, point-in-time eligible and complete. It winsorizes variables
    monthly at the 1st/99th percentiles and lags predictors by 12 months; callers
    must apply these rules and the prescribed missing-ROE handling.
    Columns are rescaled for numerical stability; a scaled singular-value ratio
    at or below sqrt(machine epsilon) is rejected as an engineering safeguard,
    not a criterion taken from the published study.
    """
    try:
        x = _real_array(features)
        y = _real_array(target)
        w = _real_array(weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid regression inputs") from exc
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or x.shape[0] <= 4
        or y.shape != (x.shape[0],)
        or w.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(w).all()
        or (w <= 0).any()
    ):
        raise ValueError("Invalid regression inputs")
    design = np.column_stack((np.ones(len(x)), x))
    normalized_weight = w / np.max(w)
    if (normalized_weight == 0).any():
        raise ValueError("Invalid regression weight scale")
    root_weight = np.sqrt(normalized_weight)
    weighted = design * root_weight[:, None]
    scales = np.max(np.abs(weighted), axis=0)
    if (scales == 0).any():
        raise ValueError("Ill-conditioned regression")
    scaled = weighted / scales
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] / singular[0] <= np.sqrt(np.finfo(float).eps):
        raise ValueError("Ill-conditioned regression")
    coefficients, _, rank, _ = np.linalg.lstsq(scaled, y * root_weight, rcond=None)
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = coefficients / scales
    if rank != 4:
        raise ValueError("Singular regression")
    if not np.isfinite(coefficients).all():
        raise ValueError("Non-finite regression coefficients")
    return coefficients


def forecast(fits: pd.DataFrame, decision_at: object, features: object) -> float:
    """Average available prior monthly coefficients in the previous 120 calendar months.

    ``month`` is the UTC calendar month in which the fit was estimated. The decision
    month's fit is excluded, even if available. Each fit's availability must be
    independently documented. The caller must also verify that the fit was actually
    estimated in its named month and used only training observations and outcomes
    available then; the availability check does not establish those facts or whether
    current features were known in time.
    """
    if not isinstance(fits, pd.DataFrame) or fits.empty:
        raise ValueError("At least 30 available monthly fits required")
    required = ["month", "available_at", *_COEFFICIENTS]
    if fits.columns.has_duplicates or not set(required).issubset(fits.columns):
        raise ValueError("Missing or duplicate fit columns")
    decision = _utc(decision_at)
    decision_month = decision.tz_localize(None).to_period("M")
    rows = fits.loc[:, required].copy()
    if (
        not rows["month"]
        .map(
            lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9]{4}-[0-9]{2}", value))
        )
        .all()
    ):
        raise ValueError("Invalid regression month")
    try:
        months = pd.PeriodIndex(rows["month"], freq="M")
        times = pd.DatetimeIndex([_utc(value) for value in rows["available_at"]])
        coefficients = _real_array(rows.loc[:, _COEFFICIENTS])
        values = _real_array(features)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid forecast inputs") from exc
    if months.has_duplicates:
        raise ValueError("duplicate regression month")
    if coefficients.shape != (len(rows), 4) or values.shape != (3,):
        raise ValueError("Invalid forecast inputs")
    if not np.isfinite(coefficients).all() or not np.isfinite(values).all():
        raise ValueError("Invalid forecast inputs")
    starts = months.to_timestamp().tz_localize("UTC")
    if (times < starts).any():
        raise ValueError("Fit available before its month")
    use = (months >= decision_month - 120) & (months < decision_month) & (times <= decision)
    if use.sum() < 30:
        raise ValueError("At least 30 available monthly fits required")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(coefficients[use], axis=0) @ np.r_[1.0, values])
    if not np.isfinite(result):
        raise ValueError("Non-finite forecast")
    return result
