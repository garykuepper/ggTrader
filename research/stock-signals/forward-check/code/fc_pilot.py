"""Compare listing coverage and daily closing prices."""

import pandas as pd


def filter_listing(listed: pd.DataFrame) -> pd.DataFrame:
    """Keep ordinary, non-test listings."""
    return listed.loc[~listed["is_etf"] & ~listed["is_test"]].copy()


def coverage(listed: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Count listed symbols with at least one returned bar."""
    symbols = set(listed["symbol"])
    returned = len(symbols & set(bars["symbol"]))
    return {
        "listed": len(symbols),
        "returned": returned,
        "missing": len(symbols) - returned,
        "share": returned / len(symbols) if symbols else 0.0,
    }


def price_agreement(bars: pd.DataFrame, reference: pd.DataFrame, tol: float) -> float:
    """Return the fraction of matched closes within relative tolerance."""
    matched = bars.merge(reference, on="symbol", suffixes=("_bar", "_reference"))
    if matched.empty:
        return 0.0
    return float(
        (
            (matched["close_bar"] - matched["close_reference"]).abs()
            <= tol * matched["close_reference"].abs()
        ).mean()
    )
