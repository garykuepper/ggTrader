"""List the dead stocks the first test held and attach company names from the McGill table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# The closed first test keeps its results next to this study; this study only reads them.
FIRST_TEST_FOLDER = "dead-stock-first-test"
# Which result files to read, as (file name, window label, universe label).
RESULT_FILES = (("_dev-A.json", "dev", "A"), ("_holdout-A.json", "holdout", "A"))
PERMNO_PREFIX = "PERMNO-"
ROSTER_COLUMNS = ["symbol", "permno", "window", "universe", "exit_date", "last_close"]
NAME_COLUMNS = ["stock_name", "ticker", "cusip", "exchcd", "name_source"]

# Parametrised SQL only: the load id, release id and the lists travel as parameters.
MCGILL_SQL = (
    "SELECT permno, date, comp_name, stock_ticker, cusip, exchcd "
    "FROM kaggle_chars.mcgill_of(%s::bigint) WHERE permno = ANY(%s::bigint[])"
)
SYMBOL_MAP_SQL = (
    "SELECT permno, symbol, valid_from, valid_to, source, release_id "
    "FROM ref.symbol_map WHERE release_id = %s::bigint AND symbol = ANY(%s::text[])"
)
MCGILL_COLUMNS = ["permno", "date", "comp_name", "stock_ticker", "cusip", "exchcd"]
MAP_COLUMNS = ["permno", "symbol", "valid_from", "valid_to", "source", "release_id"]


def parse_exits(result: dict, window: str, universe: str) -> pd.DataFrame:
    """Read the held dead-stock exits out of one first-test result file.

    Args:
        result: The parsed result JSON (key path ``result.death_price_paths.exits``).
        window: Window label to record, ``"dev"`` or ``"holdout"``.
        universe: Universe label to record, for example ``"A"``.

    Returns:
        One row per exit with columns ``symbol, permno, window, universe, exit_date,
        last_close``. ``permno`` is a nullable integer, empty for Yahoo tickers.

    Raises:
        KeyError: If the exits key path is missing, so a changed file layout fails loudly.
    """
    exits = result["result"]["death_price_paths"]["exits"]
    rows = []
    for exit_row in exits:
        symbol = str(exit_row["symbol"])
        permno = int(symbol[len(PERMNO_PREFIX) :]) if symbol.startswith(PERMNO_PREFIX) else None
        rows.append(
            {
                "symbol": symbol,
                "permno": permno,
                "window": window,
                "universe": universe,
                "exit_date": exit_row["exit_date"],
                "last_close": float(exit_row["last_close"]),
            }
        )
    frame = pd.DataFrame(rows, columns=ROSTER_COLUMNS)
    frame["permno"] = frame["permno"].astype("Int64")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    return frame


def _fetch(conn: Any, sql: str, params: tuple, columns: list[str]) -> pd.DataFrame:
    """Run one read-only query and return its rows as a frame."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _resolve_ticker_permno(row: pd.Series, candidates: pd.DataFrame) -> Any:
    """Pick the PERMNO a Yahoo ticker pointed to on its exit date.

    A ticker can be reused by different companies over time, so the span that covers the exit
    date wins; with none covering it, the span ending closest to the exit date is used.
    """
    mine = candidates[candidates["symbol"] == row["symbol"]]
    if mine.empty:
        return pd.NA
    exit_day = row["exit_date"].date()
    covering = mine[(mine["valid_from"] <= exit_day) & (mine["valid_to"] >= exit_day)]
    if not covering.empty:
        return int(covering.sort_values("valid_from").iloc[-1]["permno"])
    gap = mine["valid_to"].map(lambda day: abs((day - exit_day).days))
    return int(mine.loc[gap.idxmin(), "permno"])


def _pick_name_row(rows: pd.DataFrame, exit_day: Any) -> tuple[pd.Series, str]:
    """The McGill row for one stock: latest on or before the exit, else its earliest row."""
    ordered = rows.sort_values("date")
    before = ordered[ordered["date"] <= exit_day]
    if not before.empty:
        return before.iloc[-1], "mcgill"
    return ordered.iloc[0], "mcgill_later"


def attach_names(
    conn: Any, roster: pd.DataFrame, mcgill_load_id: int, release_id: int
) -> pd.DataFrame:
    """Add company names to the roster and fill the PERMNO of Yahoo-ticker exits.

    Args:
        conn: An open read-only database connection.
        roster: Output of ``parse_exits`` (one or more windows stacked).
        mcgill_load_id: McGill load to read through ``kaggle_chars.mcgill_of``.
        release_id: Release whose ``ref.symbol_map`` turns tickers into PERMNOs.

    Returns:
        The roster with ``stock_name, ticker, cusip, exchcd, name_source`` added.
        ``name_source`` is ``"mcgill"`` (a row on or before the exit date), ``"mcgill_later"``
        (only later rows exist) or ``"none"`` (the stock is not in McGill). Nothing is dropped.
    """
    out = roster.copy().reset_index(drop=True)
    tickers = sorted(out.loc[out["permno"].isna(), "symbol"].unique().tolist())
    if tickers:
        mapping = _fetch(conn, SYMBOL_MAP_SQL, (int(release_id), tickers), MAP_COLUMNS)
        for idx in out.index[out["permno"].isna()]:
            out.at[idx, "permno"] = _resolve_ticker_permno(out.loc[idx], mapping)
    out["permno"] = out["permno"].astype("Int64")

    permnos = sorted({int(p) for p in out["permno"].dropna()})
    mcgill = _fetch(conn, MCGILL_SQL, (int(mcgill_load_id), permnos), MCGILL_COLUMNS)
    by_permno = {int(k): g for k, g in mcgill.groupby("permno")}

    names: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        rows = by_permno.get(int(row["permno"])) if pd.notna(row["permno"]) else None
        if rows is None or rows.empty:
            names.append(
                {
                    "stock_name": None,
                    "ticker": None,
                    "cusip": None,
                    "exchcd": pd.NA,
                    "name_source": "none",
                }
            )
            continue
        picked, source = _pick_name_row(rows, row["exit_date"].date())
        names.append(
            {
                "stock_name": str(picked["comp_name"]).strip(),
                "ticker": picked["stock_ticker"],
                "cusip": picked["cusip"],
                "exchcd": picked["exchcd"],
                "name_source": source,
            }
        )
    extra = pd.DataFrame(names, columns=NAME_COLUMNS)
    extra["exchcd"] = extra["exchcd"].astype("Int64")
    return pd.concat([out, extra], axis=1)


def build_roster(
    conn: Any, study_root: Path, mcgill_load_id: int = 2, release_id: int = 1
) -> pd.DataFrame:
    """Build the full roster of held dead stocks with names.

    Args:
        conn: An open read-only database connection.
        study_root: This study's folder; the first test's results sit beside it.
        mcgill_load_id: McGill load to read names from.
        release_id: Release used to map Yahoo tickers to PERMNOs.

    Returns:
        The development exits followed by the holdout exits, with names attached.
    """
    first_test = Path(study_root).parent / FIRST_TEST_FOLDER
    parts = []
    for file_name, window, universe in RESULT_FILES:
        result = json.loads((first_test / file_name).read_text(encoding="utf-8"))
        parts.append(parse_exits(result, window=window, universe=universe))
    stacked = pd.concat(parts, ignore_index=True)
    return attach_names(conn, stacked, mcgill_load_id, release_id)
