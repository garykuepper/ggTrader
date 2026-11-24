"""Point-in-time eligibility (which stocks could be held each day) from raw CRSP rows.

Two steps. ``month_end_fields`` reads, in one read-only query, the last priced row of every
calendar month for every stock from the raw CRSP table (share code, exchange code, absolute
price, market value). ``eligibility`` is pure pandas: a stock is eligible on a day when the
**previous** calendar month's row passes the universe rule and the stock's column in the
published release covers that day.

Decisions recorded here (the brief fixes the rule, not these mechanics):

* The share code and exchange code come from the same month-end row as the price, so a stock
  that moves exchange or share class is eligible only from the month after the change.
* There is no carry-forward: a stock whose last row is in month M is eligible through month
  M+1 (the row is the previous month-end) and not after. Its release column also ends on its
  last date, so a dead stock is never eligible past that date.
* The market-value rank is taken among the stocks that pass the share-code, exchange and
  price tests that month. Ties break on the lower permno.
* One company, one column: when several release columns share a permno on a day (a demoted
  Yahoo link next to its ``PERMNO-<n>`` placeholder), only the highest-priority eligible one
  stays true. Priority: placeholder, then spell, then linked_yahoo, then yahoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

COMMON_SHARE_CODES = (10, 11)
NYSE_NASDAQ_EXCHANGE_CODES = (1, 3)
FIELD_COLUMNS = ["permno", "month_end", "shrcd", "exchcd", "abs_prc", "cap"]
SPAN_COLUMNS = ["symbol", "permno", "kind", "first_date", "last_date"]
# Lower number wins when two columns of one company are eligible on the same day.
KIND_PRIORITY = {"placeholder": 0, "spell": 1, "linked_yahoo": 2, "yahoo": 3}

# One read: the last row of each calendar month that has a price, per stock, from one load.
MONTH_END_SQL = """
SELECT DISTINCT ON (permno, date_trunc('month', date))
       permno, date AS month_end, shrcd, exchcd, abs(prc) AS abs_prc, abs(prc) * shrout AS cap
FROM kaggle_crsp.daily_of(%s)
WHERE date >= %s::date AND date <= %s::date AND prc IS NOT NULL
ORDER BY permno, date_trunc('month', date), date DESC
"""


@dataclass(frozen=True)
class UniverseRule:
    """A universe: lowest allowed absolute price and the market-value rank limit (None = none)."""

    price_floor: float
    rank_limit: int | None


UNIVERSE_RULES: dict[str, UniverseRule] = {
    "A": UniverseRule(price_floor=5.0, rank_limit=1000),
    "B": UniverseRule(price_floor=1.0, rank_limit=None),
}


def month_end_fields(conn: Any, load_id: int, start: str, end: str) -> pd.DataFrame:
    """Read each stock's last priced row of every month from the raw CRSP rows.

    Args:
        conn: An open read-only connection (from ``ggTrader.data.bars.connect_readonly``).
        load_id: The complete CRSP daily load to read through ``kaggle_crsp.daily_of``.
        start: First date to read, ``YYYY-MM-DD``.
        end: Last date to read, ``YYYY-MM-DD``.

    Returns:
        Columns ``permno, month_end, shrcd, exchcd, abs_prc, cap``: one row per permno per
        calendar month, ``month_end`` being the date of that last priced row. ``cap`` is
        absolute price times shares outstanding; it is used for ranking only, so the file's
        share units do not matter.
    """
    with conn.cursor() as cur:
        cur.execute(MONTH_END_SQL, (int(load_id), start, end))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=FIELD_COLUMNS)
    frame["month_end"] = pd.to_datetime(frame["month_end"])
    for name in ("permno", "shrcd", "exchcd"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in ("abs_prc", "cap"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
    return frame


def _require(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}")


def _passing_by_month(fields: pd.DataFrame, rule: UniverseRule) -> dict[pd.Period, np.ndarray]:
    """For each calendar month, the sorted permnos whose month-end row passes the rule."""
    rows = fields.copy()
    rows["month_end"] = pd.to_datetime(rows["month_end"])
    rows["period"] = rows["month_end"].dt.to_period("M")
    # Two rows in one month: the later date is the month-end row.
    rows = rows.sort_values(["permno", "period", "month_end"]).drop_duplicates(
        ["permno", "period"], keep="last"
    )
    ok = (
        rows["shrcd"].isin(COMMON_SHARE_CODES)
        & rows["exchcd"].isin(NYSE_NASDAQ_EXCHANGE_CODES)
        & (rows["abs_prc"].abs() >= rule.price_floor)
    )
    rows = rows[ok]
    if rule.rank_limit is not None:
        rows = rows.sort_values(["period", "cap", "permno"], ascending=[True, False, True])
        rows = rows[rows["cap"].notna()]
        rows = rows[rows.groupby("period").cumcount() < rule.rank_limit]
    return {
        period: np.sort(group["permno"].to_numpy())
        for period, group in rows.groupby("period", sort=False)
    }


def _one_column_per_company(mask: np.ndarray, spans: pd.DataFrame, permno: pd.Series) -> np.ndarray:
    """Keep, on each day, at most one true column per permno (highest priority first)."""
    priority = spans["kind"].map(KIND_PRIORITY).fillna(len(KIND_PRIORITY)).to_numpy()
    for _, positions in permno.dropna().groupby(permno.dropna()).indices.items():
        if len(positions) < 2:
            continue
        cols = np.array(positions)
        cols = cols[np.lexsort((spans["symbol"].to_numpy()[cols], priority[cols]))]
        block = mask[:, cols]
        mask[:, cols] = block & (block.cumsum(axis=1) == 1)
    return mask


def eligibility(
    fields: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    universe: Literal["A", "B"],
    daily_index: pd.DatetimeIndex,
    rule: UniverseRule | None = None,
) -> pd.DataFrame:
    """Which release columns are eligible on each day, using only the previous month-end.

    Args:
        fields: Output of ``month_end_fields`` (``permno, month_end, shrcd, exchcd, abs_prc,
            cap``); the sign of ``abs_prc`` is ignored.
        spans: The release spans frame (``Bars.spans`` columns ``symbol, permno, kind,
            first_date, last_date``; ``possible_double`` is not needed because the one-column
            rule is applied on every shared permno).
        universe: ``"A"`` (price at least 5, top 1,000 by market value) or ``"B"`` (price at
            least 1, no rank limit).
        daily_index: The trading days of the result.
        rule: Replaces the universe's rule (tests and sensitivity runs).

    Returns:
        A boolean frame, ``daily_index`` by symbol (one column per ``spans`` row, in order).

    Raises:
        ValueError: Unknown universe or missing columns.
    """
    if universe not in UNIVERSE_RULES:
        raise ValueError(f"unknown universe {universe!r}; expected one of {sorted(UNIVERSE_RULES)}")
    _require(fields, FIELD_COLUMNS, "fields")
    _require(spans, SPAN_COLUMNS, "spans")
    rule = rule or UNIVERSE_RULES[universe]
    days = pd.DatetimeIndex(daily_index)
    spans = spans.reset_index(drop=True)
    permno = pd.to_numeric(spans["permno"], errors="coerce")
    first = pd.to_datetime(spans["first_date"]).to_numpy("datetime64[ns]")
    last = pd.to_datetime(spans["last_date"]).to_numpy("datetime64[ns]")
    passing = _passing_by_month(fields, rule)
    day_values = days.to_numpy("datetime64[ns]")
    day_periods = days.to_period("M")
    mask = np.zeros((len(days), len(spans)), dtype=bool)
    permno_values = permno.to_numpy()
    for period in day_periods.unique():
        rows = np.flatnonzero(day_periods == period)
        permnos = passing.get(period - 1)
        if permnos is None or len(permnos) == 0:
            continue
        passes = np.isin(permno_values, permnos) & permno.notna().to_numpy()
        covered = (day_values[rows, None] >= first[None, :]) & (
            day_values[rows, None] <= last[None, :]
        )
        mask[rows] = covered & passes[None, :]
    mask = _one_column_per_company(mask, spans, permno)
    return pd.DataFrame(mask, index=days, columns=spans["symbol"].tolist())
