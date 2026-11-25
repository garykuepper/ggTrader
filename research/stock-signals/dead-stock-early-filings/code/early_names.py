"""Attach company names to the early-window death roster from the McGill table.

The raw CRSP table has no company-name column, so names come only from
``kaggle_chars.mcgill_of(2)`` (coverage 2000-01-31 onward) by ``permno`` on or before
each stock's ``last_bar``. A stock with no McGill row on or before its last trading day
receives ``company_name = None``, ``name_source = ""``, and ``exchcd = pd.NA``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

DEFAULT_MCGILL_LOAD_ID = 2
NAME_COLUMNS = ["company_name", "name_source", "exchcd"]
MCGILL_COLUMNS = ["permno", "date", "comp_name", "exchcd"]
# Parameterised read-only query against McGill only; never touches CRSP or market tables.
MCGILL_SQL = (
    "SELECT permno, date, comp_name, exchcd "
    "FROM kaggle_chars.mcgill_of(%s::bigint) "
    "WHERE permno = ANY(%s::bigint[]) AND date <= %s::date"
)


def _as_date(value: Any) -> dt.date:
    """Normalise a Timestamp, datetime, date, or ISO string to a plain ``datetime.date``."""
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return pd.Timestamp(value).date()


def _clean_name(value: Any) -> str | None:
    """Return a stripped non-empty company name, or ``None`` for null/blank values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _fetch_mcgill(
    conn: Any, mcgill_load_id: int, permnos: list[int], max_date: dt.date
) -> pd.DataFrame:
    """Run the single parameterised read-only McGill query and return its rows as a frame."""
    with conn.cursor() as cur:
        cur.execute(MCGILL_SQL, (int(mcgill_load_id), permnos, max_date))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=MCGILL_COLUMNS)
    if not frame.empty:
        frame["date"] = [_as_date(d) for d in frame["date"]]
    return frame


def _pick_name_row(rows: pd.DataFrame, last_day: dt.date) -> tuple[str | None, Any, str]:
    """Pick the latest named McGill row on or before ``last_day``; else no name."""
    before = rows[rows["date"] <= last_day]
    if before.empty:
        return None, pd.NA, ""
    ordered = before.sort_values("date")
    for idx in range(len(ordered) - 1, -1, -1):
        row = ordered.iloc[idx]
        name = _clean_name(row["comp_name"])
        if name is not None:
            exchcd = int(row["exchcd"]) if pd.notna(row["exchcd"]) else pd.NA
            return name, exchcd, "mcgill"
    return None, pd.NA, ""


def attach_names(
    conn: Any,
    roster: pd.DataFrame,
    *,
    mcgill_load_id: int = DEFAULT_MCGILL_LOAD_ID,
) -> pd.DataFrame:
    """Add ``company_name``, ``name_source``, and ``exchcd`` from McGill to ``roster``.

    Args:
        conn: An open read-only database connection (``ggTrader.data.bars.connect_readonly``).
        roster: Output of ``early_roster.build_death_roster`` (requires ``permno`` and
            ``last_bar`` columns).
        mcgill_load_id: McGill load to query through ``kaggle_chars.mcgill_of`` (default 2).

    Returns:
        A copy of ``roster`` with ``company_name`` (``str`` or ``None``), ``name_source``
        (``"mcgill"`` or ``""``), and ``exchcd`` (nullable ``Int64``) added. Nothing is
        dropped and ``roster`` is not mutated.

    Raises:
        ValueError: If ``roster`` is missing ``permno`` or ``last_bar``.
    """
    missing = [col for col in ("permno", "last_bar") if col not in roster.columns]
    if missing:
        raise ValueError(f"roster is missing required columns {missing}")

    out = roster.copy().reset_index(drop=True)
    valid_mask = out["permno"].notna() & out["last_bar"].notna()
    permnos = sorted({int(p) for p in out.loc[valid_mask, "permno"]})

    by_permno: dict[int, pd.DataFrame] = {}
    if permnos:
        max_date = max(_as_date(d) for d in out.loc[valid_mask, "last_bar"])
        mcgill = _fetch_mcgill(conn, mcgill_load_id, permnos, max_date)
        if not mcgill.empty:
            by_permno = {int(k): g for k, g in mcgill.groupby("permno")}

    company_names: list[str | None] = []
    name_sources: list[str] = []
    exchcds: list[Any] = []
    for _, row in out.iterrows():
        if pd.isna(row["permno"]) or pd.isna(row["last_bar"]):
            company_names.append(None)
            name_sources.append("")
            exchcds.append(pd.NA)
            continue
        rows = by_permno.get(int(row["permno"]))
        if rows is None or rows.empty:
            company_names.append(None)
            name_sources.append("")
            exchcds.append(pd.NA)
            continue
        name, exchcd, source = _pick_name_row(rows, _as_date(row["last_bar"]))
        company_names.append(name)
        name_sources.append(source)
        exchcds.append(exchcd)

    out["company_name"] = pd.Series(company_names, index=out.index, dtype=object)
    out["name_source"] = pd.Series(name_sources, index=out.index, dtype=object)
    out["exchcd"] = pd.Series(exchcds, index=out.index, dtype="Int64")
    return out


__all__ = [
    "DEFAULT_MCGILL_LOAD_ID",
    "MCGILL_COLUMNS",
    "MCGILL_SQL",
    "NAME_COLUMNS",
    "attach_names",
]
