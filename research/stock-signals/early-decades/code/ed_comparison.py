"""Read the market stand-in and SPY comparison returns on their registered dates."""

import pandas as pd

from ggTrader.data.bars import connect_readonly

CRSP_END = pd.Timestamp("1993-01-29")
SPY_START = pd.Timestamp("1993-02-01")


def comparison_label(start: str, end: str) -> str:
    """Name the source for a period, or the combined comparison across the join."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("start must not follow end")
    if last <= CRSP_END:
        return "CRSP market return (stand-in)"
    if first >= SPY_START:
        return "SPY"
    return "CRSP market return (stand-in) + SPY"


def comparison_returns(start: str, end: str) -> pd.Series:
    """Read dated returns, including SPY's return from its first close to the next."""
    label = comparison_label(start, end)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    conn = connect_readonly()
    try:
        with conn.cursor() as cursor:
            parts = []
            if first <= CRSP_END:
                cursor.execute(
                    "SELECT date, min(vwretd), max(vwretd) FROM kaggle_crsp.daily_of(1) "
                    "WHERE date BETWEEN %s AND %s GROUP BY date ORDER BY date",
                    (start, min(last, CRSP_END).date()),
                )
                rows = cursor.fetchall()
                if not rows or any(row[1] is None or row[1] != row[2] for row in rows):
                    raise ValueError("CRSP market returns missing or inconsistent")
                parts.append(
                    pd.Series(
                        [r[1] for r in rows],
                        index=pd.to_datetime([r[0] for r in rows]),
                        dtype=float,
                    )
                )
            if last >= SPY_START:
                spy_first = max(first, SPY_START)
                cursor.execute(
                    "SELECT date, close FROM market.spy_daily_yahoo_1993_1999 "
                    "WHERE date BETWEEN (SELECT max(date) FROM market.spy_daily_yahoo_1993_1999 "
                    "WHERE date < %s) AND %s ORDER BY date",
                    (spy_first.date(), end),
                )
                rows = cursor.fetchall()
                if len(rows) < 2:
                    raise ValueError("SPY needs a preceding close")
                prices = pd.Series(
                    [r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows]), dtype=float
                )
                if prices.isna().any() or (prices <= 0).any():
                    raise ValueError("invalid SPY prices")
                parts.append(prices.pct_change(fill_method=None).iloc[1:])
    finally:
        conn.close()
    returns = pd.concat(parts)
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("comparison dates must be unique and ordered")
    if returns.empty or returns.index[0] != first or returns.index[-1] != last:
        raise ValueError("comparison does not cover both requested endpoints")
    return returns.rename(label)


def after_tax_growth(returns: pd.Series, tax_rate: float) -> float:
    """Compute buy-and-hold growth after tax on a positive final gain."""
    if returns.empty or returns.isna().any() or (returns < -1).any():
        raise ValueError("invalid comparison returns")
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax rate must be between zero and one")
    growth = float((1 + returns).prod())
    return growth - max(growth - 1, 0) * tax_rate
