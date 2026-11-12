"""Select matched calendar-year downside observations for a proposed study."""

from numbers import Integral

import numpy as np
import pandas as pd


def calendar_year_returns(
    curves: pd.DataFrame, sessions: pd.DatetimeIndex, years: list[int]
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate annual returns using a caller-verified complete session calendar.

    The supplied calendar must contain every exchange session in each requested
    year, a previous-year endpoint and a session after the final requested year.
    Curves must share every expected session and already include modeled costs.
    Wealth may fall to zero but cannot recover without an external flow; a year
    starting at zero has no defined percentage return and is rejected.
    Calendar accuracy, distributions, initial purchases and input provenance are
    the caller's responsibility. Intervening sessions between requested years
    must also be present and valid so a total loss cannot silently recover.
    """
    if (
        not isinstance(sessions, pd.DatetimeIndex)
        or sessions.tz is not None
        or sessions.has_duplicates
        or not sessions.is_monotonic_increasing
        or not sessions.equals(sessions.normalize())
        or not years
        or any(isinstance(y, bool) or not isinstance(y, Integral) for y in years)
        or sorted(set(years)) != years
        or sessions.empty
        or sessions[-1].year <= years[-1]
    ):
        raise ValueError("Invalid or incomplete calendar")
    if (
        not isinstance(curves, pd.DataFrame)
        or list(curves.columns) != ["spy_gross", "candidate", "fixed", "spy"]
        or not isinstance(curves.index, pd.DatetimeIndex)
        or curves.index.tz is not None
        or curves.index.has_duplicates
        or not curves.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid session curves")
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in curves.dtypes
    ):
        raise ValueError("Invalid curve values")
    annual = []
    exhausted = False
    last_end = None
    for year in years:
        if exhausted:
            raise ValueError("Invalid curve values: wealth was previously exhausted")
        before = sessions[sessions.year == year - 1]
        current = sessions[sessions.year == year]
        if before.empty or current.empty:
            raise ValueError(f"Incomplete calendar for {year}")
        start, end = before[-1], current[-1]
        if last_end is not None and start > last_end:
            expected_gap = sessions[(sessions >= last_end) & (sessions <= start)]
            gap = curves.loc[last_end:start]
            if not gap.index.equals(expected_gap):
                raise ValueError("Missing or extra intervening session")
            try:
                gap_values = gap.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid curve values") from exc
            if not np.isfinite(gap_values).all() or (gap_values <= 0).any():
                raise ValueError("Invalid curve values between requested years")
        expected = sessions[(sessions >= start) & (sessions <= end)]
        actual = curves.loc[start:end]
        if not actual.index.equals(expected):
            raise ValueError(f"Missing or extra session for {year}")
        try:
            values = actual.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid curve values") from exc
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values[0] == 0).any()
            or ((values[:-1] == 0) & (values[1:] > 0)).any()
        ):
            raise ValueError("Invalid curve values")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = values[-1] / values[0] - 1
        if not np.isfinite(result).all():
            raise ValueError("Invalid annual return values")
        annual.append(result)
        exhausted = bool((values[-1] == 0).any())
        last_end = end
    returns = pd.DataFrame(annual, index=years, columns=curves.columns)
    return returns["spy_gross"], returns[["candidate", "fixed", "spy"]]


def matched_down_years(gross_spy: pd.Series, net: pd.DataFrame) -> pd.DataFrame:
    """Select strictly negative gross SPY years and compare after-cost returns.

    The caller must verify each row covers the same complete calendar year and
    that gross SPY is measured before modeled trading costs. This function does
    not check the trading calendar, calculate returns, or establish evidence.
    """
    if not isinstance(gross_spy, pd.Series) or not isinstance(net, pd.DataFrame):
        raise ValueError("Invalid return inputs")
    years = gross_spy.index
    if (
        not years.equals(net.index)
        or years.has_duplicates
        or not years.is_monotonic_increasing
        or any(isinstance(y, bool) or not isinstance(y, Integral) for y in years)
    ):
        raise ValueError("Invalid or mismatched calendar years")
    if list(net.columns) != ["candidate", "fixed", "spy"]:
        raise ValueError("Invalid return columns")
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in (gross_spy.dtype, *net.dtypes)
    ):
        raise ValueError("Invalid return values")
    try:
        values = np.column_stack((gross_spy.to_numpy(dtype=float), net.to_numpy(dtype=float)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid return values") from exc
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("Invalid return values")
    result = net.loc[gross_spy < 0].copy()
    result["candidate_minus_fixed"] = result["candidate"] - result["fixed"]
    return result
