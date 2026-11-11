"""Select aligned monthly returns from Global-q expected-growth portfolios."""

import numpy as np
import pandas as pd


def _month_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Reject missing or invalid year/month labels and construct month-end dates."""
    try:
        year = pd.to_numeric(frame["year"], errors="raise")
        month = pd.to_numeric(frame["month"], errors="raise")
        if (year.isna() | month.isna() | (year % 1 != 0) | (month % 1 != 0)).any():
            raise ValueError("Year and month must be whole numbers")
        dates = pd.to_datetime({"year": year, "month": month, "day": 1}, errors="raise")
        return pd.DatetimeIndex(dates + pd.offsets.MonthEnd(0))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid year or month") from exc


def extract_expected_growth(
    portfolios: pd.DataFrame, factors: pd.DataFrame, start: str, end: str
) -> tuple[pd.Series, pd.Series]:
    """Return big/high-growth and market total returns on the requested months.

    The provider expresses monthly returns in percent; the market is R_MKT + R_F.
    This only transforms supplied data and does not establish provenance or approval.
    """
    if portfolios.columns.has_duplicates or factors.columns.has_duplicates:
        raise ValueError("Duplicate columns")
    try:
        size_rank = pd.to_numeric(portfolios["rank_ME"], errors="raise")
        growth_rank = pd.to_numeric(portfolios["rank_EG"], errors="raise")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid portfolio ranks") from exc
    if not size_rank.isin([1, 2]).all() or not growth_rank.isin([1, 2, 3]).all():
        raise ValueError("Unrecognized portfolio ranks")
    try:
        chosen = portfolios.loc[(size_rank == 2) & (growth_rank == 3)].copy()
        counts = pd.to_numeric(chosen["nstocks"], errors="raise")
        candidate = pd.to_numeric(chosen["ret_vw"], errors="raise") / 100
        market = (
            pd.to_numeric(factors["R_MKT"], errors="raise")
            + pd.to_numeric(factors["R_F"], errors="raise")
        ) / 100
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid portfolio or factor column") from exc
    if not np.isfinite(counts).all() or (counts <= 0).any() or (counts % 1 != 0).any():
        raise ValueError("Selected portfolios need positive whole stock counts")
    candidate.index = _month_index(chosen)
    market.index = _month_index(factors)
    if candidate.index.has_duplicates or market.index.has_duplicates:
        raise ValueError("Duplicate portfolio or market months")
    start_month, end_month = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    if start_month > end_month:
        raise ValueError("End month precedes start month")
    dates = pd.period_range(start_month, end_month, freq="M").to_timestamp("M")
    candidate, market = candidate.reindex(dates), market.reindex(dates)
    for series in (candidate, market):
        if not np.isfinite(series).all() or (series <= -1).any():
            raise ValueError("Missing or invalid monthly return")
    return candidate, market
