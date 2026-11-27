"""Read the registered market comparison without silently joining the missing January gap."""

import pandas as pd

from ggTrader.data.bars import connect_readonly

SPY_START = pd.Timestamp("1993-01-29")
CRSP_END = pd.Timestamp("1992-12-31")


def comparison_label(start: str, end: str) -> str:
    """Name the single source for the requested period; reject the uncovered gap."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("start must not follow end")
    if last <= CRSP_END:
        return "CRSP market return (stand-in)"
    if first >= SPY_START:
        return "SPY"
    raise ValueError("comparison unavailable for 1993-01-01 through 1993-01-28")


def comparison_returns(start: str, end: str) -> pd.Series:
    """Return daily returns for one covered period from a read-only connection."""
    label = comparison_label(start, end)
    conn = connect_readonly()
    try:
        with conn.cursor() as cursor:
            if label == "SPY":
                cursor.execute(
                    "SELECT date, close FROM market.spy_daily_yahoo_1993_1999 "
                    "WHERE date BETWEEN %s AND %s ORDER BY date",
                    (start, end),
                )
            else:
                cursor.execute(
                    "SELECT date, min(vwretd), max(vwretd) FROM kaggle_crsp.daily_of(1) "
                    "WHERE date BETWEEN %s AND %s GROUP BY date ORDER BY date",
                    (start, end),
                )
            rows = cursor.fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError("comparison has no observations")
    dates = pd.DatetimeIndex([row[0] for row in rows])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("comparison dates must be unique and ordered")
    if dates[0] != pd.Timestamp(start) or dates[-1] != pd.Timestamp(end):
        raise ValueError("comparison does not cover both requested endpoints")
    if label == "SPY":
        prices = pd.Series([row[1] for row in rows], index=dates, dtype=float)
        if prices.isna().any() or (prices <= 0).any():
            raise ValueError("invalid SPY prices")
        returns = prices.pct_change(fill_method=None).fillna(0.0)
    else:
        if any(row[1] is None or row[1] != row[2] for row in rows):
            raise ValueError("CRSP market returns missing or inconsistent")
        returns = pd.Series([row[1] for row in rows], index=dates, dtype=float)
    return returns.rename(label)


def after_tax_growth(returns: pd.Series, tax_rate: float) -> float:
    """Compute buy-and-hold growth after tax on a positive final gain."""
    if returns.empty or returns.isna().any() or (returns < -1).any():
        raise ValueError("invalid comparison returns")
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax rate must be between zero and one")
    growth = float((1 + returns).prod())
    return growth - max(growth - 1, 0) * tax_rate
