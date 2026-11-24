"""Phase 1 of the dead-stock data-integrity study: scan CRSP rows, count flags, match holdings."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import integrity_checks as ic
import pandas as pd
import price_path
import roster as roster_mod

from ggTrader.utils.io import atomic_write_text, write_json

# CRSP ends here; the brief reads nothing later.
CRSP_END = "2024-12-31"
# The scan starts a year before the first test's span so early flags have context.
SCAN_START = "2004-01-01"
BATCH_SIZE = 500
# A date counts as a trading day when at least this many stocks have a row on it.
CALENDAR_MIN_STOCKS = 1000
EXAMPLES_PER_CHECK = 20
# CRSP share codes 10/11 are ordinary common shares; exchange codes 1/3 are NYSE and Nasdaq.
SHARE_CODES = [10, 11]
EXCHANGE_CODES = [1, 3]
# This study's own folder (code/ sits inside it); inputs are located from here, never from --out.
STUDY_DIR = Path(__file__).resolve().parent.parent
FETCH_CHUNK = 200_000
SPELL_SYMBOL_RE = r"PERMNO-\d+~\d+"
TRADE_FILES = ("dev-A", "dev-B", "holdout-A", "holdout-B")
FIRST_TEST_RESULTS = Path("results") / "dead_stock_first_test"
OUTPUT_JSON = "_integrity.json"
OUTPUT_ROSTER = "_roster.csv"
# Report key, the check function's flag name, and how to read a size out of the flag detail.
CHECKS = (
    ("duplicates", "duplicate", None),
    ("bad_price", "bad_price", None),
    ("spikes", "spike", r"ret=(-?\d+\.?\d*)"),
    ("gaps", "gap", r"^(\d+) days"),
    ("stale", "stale", r"^(\d+) rows"),
    ("shares", "shares", None),
)
SCAN_COLUMNS = ["permno", "date", "prc", "ret", "shrout"]

# Parametrised SQL only: ids, dates and lists all travel as parameters. The share and exchange
# codes choose WHICH stocks are scanned (PERMNOS_SQL) and the trading calendar (CALENDAR_SQL);
# they never thin a stock's own rows (BATCH_SQL), or a move between exchanges would look like a gap.
PERMNOS_SQL = (
    "SELECT DISTINCT d.permno FROM kaggle_crsp.daily d "
    "WHERE d.load_id = %s::bigint AND d.shrcd = ANY(%s::int[]) AND d.exchcd = ANY(%s::int[]) "
    "AND d.date >= %s::date AND d.date <= %s::date "
    "AND d.permno IN (SELECT permno FROM market.stock_spans "
    "WHERE release_id = %s::bigint AND permno IS NOT NULL) ORDER BY d.permno"
)
CALENDAR_SQL = (
    "SELECT d.date FROM kaggle_crsp.daily d "
    "WHERE d.load_id = %s::bigint AND d.shrcd = ANY(%s::int[]) AND d.exchcd = ANY(%s::int[]) "
    "AND d.date >= %s::date AND d.date <= %s::date "
    "GROUP BY d.date HAVING count(*) >= %s ORDER BY d.date"
)
BATCH_SQL = (
    "SELECT d.permno, d.date, d.prc, d.ret, d.shrout FROM kaggle_crsp.daily d "
    "WHERE d.load_id = %s::bigint AND d.permno = ANY(%s::bigint[]) "
    "AND d.date >= %s::date AND d.date <= %s::date"
)
# A first probe through daily_of() with the stock-spans filter ran over 12 minutes and was
# stopped, while the same question on the table by load id takes under a second (see
# _hand_checks.md; the cause was not determined). The scan therefore reads the table by load id
# after the same completeness test.
LOAD_COMPLETE_SQL = "SELECT 1 FROM ref.loads WHERE load_id = %s::bigint AND status = 'complete'"
ROSTER_RET_SQL = (
    "SELECT d.permno, d.date, d.ret FROM kaggle_crsp.daily d "
    "WHERE d.load_id = %s::bigint AND d.permno = ANY(%s::bigint[]) AND d.date <= %s::date "
    "ORDER BY d.permno, d.date"
)


def permno_batches(permnos: Sequence[int], size: int = BATCH_SIZE) -> list[list[int]]:
    """Split the sorted, de-duplicated permnos into consecutive lists of at most ``size``."""
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    ordered = sorted({int(p) for p in permnos})
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def _fetch_chunks(conn: Any, sql: str, params: tuple, columns: list[str]) -> pd.DataFrame:
    """Run one read-only query and build its frame chunk by chunk to keep peak memory low."""
    parts: list[pd.DataFrame] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        while True:
            rows = cur.fetchmany(FETCH_CHUNK)
            if not rows:
                break
            parts.append(pd.DataFrame(rows, columns=columns))
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def require_complete_load(conn: Any, load_id: int) -> None:
    """Refuse a load that is not marked complete (the test ``daily_of`` makes itself)."""
    with conn.cursor() as cur:
        cur.execute(LOAD_COMPLETE_SQL, (int(load_id),))
        if cur.fetchone() is None:
            raise RuntimeError(f"load {load_id} is not complete")


def read_scan_permnos(conn: Any, release_id: int, crsp_load: int, eval_end: str) -> list[int]:
    """Permnos in the release's stock spans with at least one qualifying row since the start.

    A row qualifies when its share code is in ``SHARE_CODES`` and its exchange code in
    ``EXCHANGE_CODES``. This is the only place those codes pick stocks.
    """
    params = (
        int(crsp_load),
        SHARE_CODES,
        EXCHANGE_CODES,
        SCAN_START,
        eval_end,
        int(release_id),
    )
    with conn.cursor() as cur:
        cur.execute(PERMNOS_SQL, params)
        return [int(row[0]) for row in cur.fetchall()]


def read_calendar(conn: Any, crsp_load: int, eval_end: str) -> pd.DatetimeIndex:
    """The dates on which at least ``CALENDAR_MIN_STOCKS`` filtered stocks have a row."""
    params = (
        int(crsp_load),
        SHARE_CODES,
        EXCHANGE_CODES,
        SCAN_START,
        eval_end,
        CALENDAR_MIN_STOCKS,
    )
    with conn.cursor() as cur:
        cur.execute(CALENDAR_SQL, params)
        dates = [row[0] for row in cur.fetchall()]
    return pd.DatetimeIndex(pd.to_datetime(dates))


def read_batch(conn: Any, crsp_load: int, permnos: list[int], eval_end: str) -> pd.DataFrame:
    """All CRSP rows (``SCAN_COLUMNS``) of the given permnos since ``SCAN_START``.

    No share or exchange code filter is applied to the rows: a stock that moved between
    exchanges keeps every day, so the gap check sees only real holes.
    """
    params = (int(crsp_load), permnos, SCAN_START, eval_end)
    frame = _fetch_chunks(conn, BATCH_SQL, params, SCAN_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def read_roster_returns(
    conn: Any, crsp_load: int, permnos: list[int], eval_end: str
) -> pd.DataFrame:
    """Daily returns of the roster stocks (no share or exchange filter), oldest first."""
    frame = _fetch_chunks(
        conn, ROSTER_RET_SQL, (int(crsp_load), permnos, eval_end), ["permno", "date", "ret"]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def label_roster(roster: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Add ``ratio, daily_std, price_label`` to the roster from split-aware return chains.

    Each close is the cumulative product of ``1 + ret`` from the stock's first row up to its
    exit date (ret already includes splits and dividends, raw prices do not). Missing returns
    leave a missing close, which ``classify_path`` drops (the "own valid closes" rule).

    Args:
        roster: Roster with ``permno`` (nullable) and ``exit_date``.
        returns: Rows ``permno, date, ret`` for the roster permnos.

    Returns:
        A copy of the roster with the three new columns; a stock without rows is ``too_short``.
    """
    by_permno = {int(k): g.sort_values("date") for k, g in returns.groupby("permno")}
    ratios: list[float | None] = []
    stds: list[float | None] = []
    labels: list[str] = []
    for _, row in roster.iterrows():
        group = by_permno.get(int(row["permno"])) if pd.notna(row["permno"]) else None
        if group is None:
            out = price_path.classify_path(pd.Series(dtype="float64"))
        else:
            upto = group[group["date"] <= row["exit_date"]]
            close = (1.0 + upto["ret"].astype("float64")).cumprod()
            out = price_path.classify_path(close)
        ratios.append(out["ratio"])
        stds.append(out["daily_std"])
        labels.append(out["label"])
    result = roster.copy()
    result["ratio"] = ratios
    result["daily_std"] = stds
    result["price_label"] = labels
    return result


def map_trade_symbols(trades: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rewrite Yahoo-ticker symbols to ``PERMNO-<n>`` using the roster's date-aware rule.

    Args:
        trades: Trades of one cell with ``date`` and ``symbol``.
        mapping: ``ref.symbol_map`` rows for the tickers (``roster.MAP_COLUMNS``).

    Returns:
        ``(trades, info)``: the trades with every mappable ticker rewritten, and ``info`` with
        ``ticker_rows`` (non-PERMNO rows), ``mapped_rows``, ``unmapped_rows``,
        ``unmapped_symbols`` (sorted, with row counts) and ``fallback_rows`` (rows where no span
        covered the trade date and the nearest span was used).
    """
    out = trades.copy()
    symbols = out["symbol"].astype(str)
    # A later spell of the same stock is named ``PERMNO-<n>~<k>``; it is still permno n.
    spell = symbols.str.fullmatch(SPELL_SYMBOL_RE)
    symbols = symbols.where(~spell, symbols.str.replace(r"~\d+$", "", regex=True))
    is_ticker = ~symbols.str.startswith(roster_mod.PERMNO_PREFIX)
    info: dict[str, Any] = {
        "spell_suffix_rows": int(spell.sum()),
        "ticker_rows": int(is_ticker.sum()),
        "mapped_rows": 0,
        "unmapped_rows": 0,
        "unmapped_symbols": {},
        "fallback_rows": 0,
    }
    out["symbol"] = symbols
    if not is_ticker.any():
        return out, info
    cache: dict[tuple[str, pd.Timestamp], Any] = {}
    new_symbols = symbols.copy()
    unmapped: dict[str, int] = {}
    for idx in out.index[is_ticker]:
        symbol, day = symbols[idx], out.at[idx, "date"]
        key = (symbol, day)
        if key not in cache:
            cache[key] = roster_mod._resolve_ticker_permno(
                pd.Series({"symbol": symbol, "exit_date": day}), mapping
            )
        permno = cache[key]
        if pd.isna(permno):
            unmapped[symbol] = unmapped.get(symbol, 0) + 1
            continue
        new_symbols[idx] = f"{roster_mod.PERMNO_PREFIX}{int(permno)}"
        info["mapped_rows"] += 1
        mine = mapping[mapping["symbol"] == symbol]
        covered = (mine["valid_from"] <= day.date()) & (mine["valid_to"] >= day.date())
        if not covered.any():
            info["fallback_rows"] += 1
    out["symbol"] = new_symbols
    info["unmapped_rows"] = int(sum(unmapped.values()))
    info["unmapped_symbols"] = dict(sorted(unmapped.items()))
    return out, info


def load_intervals(
    conn: Any, release_id: int, trades_dir: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Holding intervals of the reference cell for each trade file, tickers mapped to permnos.

    Returns:
        ``(intervals_by_file, info_by_file)``; ``info`` carries the file's SHA-256, its row
        counts and the symbol-mapping report from ``map_trade_symbols``.
    """
    cell_trades: dict[str, pd.DataFrame] = {}
    info: dict[str, dict] = {}
    tickers: set[str] = set()
    for name in TRADE_FILES:
        path = Path(trades_dir) / f"{name}-trades.csv.gz"
        trades = ic.read_trades(path)
        cell = trades[trades["cell"] == ic.REFERENCE_CELL].reset_index(drop=True)
        symbols = cell["symbol"].astype(str)
        tickers.update(symbols[~symbols.str.startswith(roster_mod.PERMNO_PREFIX)].unique())
        cell_trades[name] = cell
        info[name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trade_rows_all_cells": int(len(trades)),
            "trade_rows_reference_cell": int(len(cell)),
        }
    mapping = (
        roster_mod._fetch(
            conn,
            roster_mod.SYMBOL_MAP_SQL,
            (int(release_id), sorted(tickers)),
            roster_mod.MAP_COLUMNS,
        )
        if tickers
        else pd.DataFrame(columns=roster_mod.MAP_COLUMNS)
    )
    intervals: dict[str, pd.DataFrame] = {}
    for name, cell in cell_trades.items():
        mapped, report = map_trade_symbols(cell, mapping)
        iv = ic.holding_intervals(mapped, ic.REFERENCE_CELL)
        intervals[name] = iv
        info[name].update(report)
        info[name]["positions"] = int(len(iv))
        info[name]["positions_open_at_end"] = int(iv["end"].isna().sum())
        info[name]["stocks_held"] = int(iv["permno"].nunique())
    return intervals, info


def _with_magnitude(flags: pd.DataFrame, pattern: str | None) -> pd.DataFrame:
    """Add a numeric ``magnitude`` (return size, days missing or run length; 0 when none)."""
    out = flags.copy()
    if pattern is None or out.empty:
        out["magnitude"] = 0.0
    else:
        out["magnitude"] = (
            out["detail"].str.extract(pattern, expand=False).astype("float64").abs().fillna(0.0)
        )
    return out


def _top(frame: pd.DataFrame, n: int = EXAMPLES_PER_CHECK) -> pd.DataFrame:
    """The ``n`` flags with the largest magnitude (ties broken by permno then date)."""
    ordered = frame.sort_values(
        ["magnitude", "permno", "date"], ascending=[False, True, True], kind="mergesort"
    )
    return ordered.head(n).reset_index(drop=True)


def _examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Plain-dict example rows with ISO dates, safe to write as JSON."""
    return [
        {
            "permno": int(row.permno),
            "date": pd.Timestamp(row.date).date().isoformat(),
            "detail": str(row.detail),
            "magnitude": float(row.magnitude),
        }
        for row in frame.itertuples(index=False)
    ]


class _Tally:
    """Running counts for one check across all batches (every flag counted, 20 stored)."""

    def __init__(self, files: Iterable[str]) -> None:
        self.rows = 0
        self.stocks: set[int] = set()
        self.examples = pd.DataFrame(columns=[*ic.FLAG_COLUMNS, "magnitude"])
        self.inside_rows = {name: 0 for name in files}
        self.inside_stocks: dict[str, set[int]] = {name: set() for name in files}
        self.inside_examples = {
            name: pd.DataFrame(columns=[*ic.FLAG_COLUMNS, "magnitude"]) for name in files
        }

    def add(self, flags: pd.DataFrame, intervals: dict[str, pd.DataFrame]) -> None:
        """Count one batch's flags and the ones that sit inside each file's holdings."""
        if flags.empty:
            return
        self.rows += len(flags)
        self.stocks.update(int(p) for p in flags["permno"].unique())
        self.examples = _top(pd.concat([self.examples, flags], ignore_index=True))
        for name, iv in intervals.items():
            inside = ic.flags_inside_holdings(flags, iv)
            if inside.empty:
                continue
            self.inside_rows[name] += len(inside)
            self.inside_stocks[name].update(int(p) for p in inside["permno"].unique())
            self.inside_examples[name] = _top(
                pd.concat([self.inside_examples[name], inside], ignore_index=True)
            )


def compute_integrity(
    batches: Iterable[pd.DataFrame],
    calendar: pd.DatetimeIndex,
    intervals: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Run the six checks batch by batch and tally flags, examples and holdings overlap.

    Args:
        batches: CRSP row frames (``permno, date, prc, ret, shrout``); each stock must sit in
            exactly one batch because gaps, spikes and stale runs need its whole history.
        calendar: Dates on which enough stocks traded (the gap check's reference calendar).
        intervals: Holding intervals per trade file, from ``load_intervals``.

    Returns:
        A dict with the six check names as keys (``rows_flagged``, ``stocks_flagged``,
        ``examples``, ``flags_inside_holdings``), ``holdings_overlap`` (per file, per check),
        and ``scan`` (batches, permnos and rows seen).
    """
    files = list(intervals)
    functions: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "duplicates": ic.check_duplicates,
        "bad_price": ic.check_bad_price,
        "spikes": ic.check_spikes,
        "gaps": lambda rows: ic.check_gaps(rows, calendar),
        "stale": ic.check_stale,
        "shares": ic.check_shares,
    }
    tallies = {key: _Tally(files) for key, _, _ in CHECKS}
    patterns = {key: pattern for key, _, pattern in CHECKS}
    n_batches = n_rows = 0
    permnos_seen: set[int] = set()
    for batch in batches:
        n_batches += 1
        n_rows += len(batch)
        permnos_seen.update(int(p) for p in batch["permno"].unique())
        for key, func in functions.items():
            flags = _with_magnitude(func(batch), patterns[key])
            tallies[key].add(flags, intervals)

    result: dict[str, Any] = {}
    for key, tally in tallies.items():
        result[key] = {
            "rows_flagged": tally.rows,
            "stocks_flagged": len(tally.stocks),
            "examples": _examples(tally.examples),
            "flags_inside_holdings": dict(tally.inside_rows),
        }
    result["holdings_overlap"] = {
        name: {
            key: {
                "flags_inside": tallies[key].inside_rows[name],
                "stocks_inside": len(tallies[key].inside_stocks[name]),
                "examples": _examples(tallies[key].inside_examples[name]),
            }
            for key, _, _ in CHECKS
        }
        for name in files
    }
    result["scan"] = {"batches": n_batches, "permnos": len(permnos_seen), "rows": n_rows}
    return result


def _git(root: Path, *args: str) -> str | None:
    """Run git in ``root``; return its stripped output, or ``None`` if git itself fails.

    An empty string is a real answer (for example a clean ``status``), so a failure must not
    look like one.
    """
    try:
        done = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=15, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def git_provenance(root: Path) -> dict[str, Any]:
    """The commit the run started from and whether files differed from it.

    Returns:
        ``git_head`` (the commit hash, or ``"unknown"`` if git failed or gave nothing) and
        ``git_dirty`` (True or False, or ``"unknown"`` if git could not be asked).
    """
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "git_head": head if head else "unknown",
        "git_dirty": "unknown" if status is None else bool(status),
    }


def code_hashes(code_dir: Path) -> dict[str, str]:
    """SHA-256 of each study module that shaped the numbers (the commit comes after the run)."""
    names = [
        "run.py",
        "dead_stock_data_integrity.py",
        "integrity_checks.py",
        "price_path.py",
        "roster.py",
    ]
    return {n: hashlib.sha256((code_dir / n).read_bytes()).hexdigest() for n in names}


def run_integrity(
    conn: Any,
    *,
    release_id: int,
    crsp_load: int,
    mcgill_load: int,
    eval_start: str,
    eval_end: str,
    out_dir: Path,
    repo_root: Path,
    trades_dir: Path | None = None,
    run_date: str,
    batches_fn: Callable[[Any, list[int]], Iterator[pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Phase 1: build the labelled roster, scan CRSP in batches, and write both outputs.

    Args:
        conn: Open read-only connection.
        release_id: Published release whose stock spans and symbol map are read.
        crsp_load: CRSP load id.
        mcgill_load: McGill load id (company names).
        eval_start: Recorded in the provenance only.
        eval_end: Last scanned date (never after ``CRSP_END``).
        out_dir: Folder that receives ``_roster.csv`` and ``_integrity.json`` (never read from).
        repo_root: Repository root (for git provenance and the default trade-file folder).
        trades_dir: Folder of the first test's trade files.
        run_date: Date of the run, recorded in the provenance.
        batches_fn: Test hook replacing the database batch reader.

    Returns:
        The payload written to ``_integrity.json``.
    """
    out_dir = Path(out_dir)
    trades_dir = Path(trades_dir) if trades_dir else repo_root / FIRST_TEST_RESULTS

    require_complete_load(conn, crsp_load)
    # The first test's result files sit beside this study's real folder, wherever --out points.
    roster = roster_mod.build_roster(conn, STUDY_DIR, mcgill_load, release_id)
    permnos = sorted({int(p) for p in roster["permno"].dropna()})
    returns = read_roster_returns(conn, crsp_load, permnos, eval_end)
    roster = label_roster(roster, returns)
    del returns

    intervals, trade_info = load_intervals(conn, release_id, trades_dir)
    scan_permnos = read_scan_permnos(conn, release_id, crsp_load, eval_end)
    calendar = read_calendar(conn, crsp_load, eval_end)

    def default_batches(connection: Any, ids: list[int]) -> Iterator[pd.DataFrame]:
        for batch in permno_batches(ids, BATCH_SIZE):
            yield read_batch(connection, crsp_load, batch, eval_end)

    reader = batches_fn or default_batches
    checks = compute_integrity(reader(conn, scan_permnos), calendar, intervals)

    scan = checks.pop("scan")
    overlap = checks.pop("holdings_overlap")
    for name, report in trade_info.items():
        overlap[name]["positions"] = report["positions"]
        overlap[name]["trade_file"] = report
    payload: dict[str, Any] = {
        "study": "dead-stock-data-integrity",
        "phase": "integrity",
        "provenance": {
            "release_id": int(release_id),
            "crsp_load": int(crsp_load),
            "mcgill_load": int(mcgill_load),
            "eval_start": eval_start,
            "eval_end": eval_end,
            "scan_start": SCAN_START,
            "batch_size": BATCH_SIZE,
            "reference_cell": ic.REFERENCE_CELL,
            **git_provenance(repo_root),
            "code_sha256": code_hashes(Path(__file__).resolve().parent),
            "run_date": run_date,
            "permnos_in_scan_list": len(scan_permnos),
            "permnos_scanned": scan["permnos"],
            "rows_scanned": scan["rows"],
            "batches": scan["batches"],
        },
        "calendar": {
            "dates": int(len(calendar)),
            "min_stocks_per_date": CALENDAR_MIN_STOCKS,
            "first": str(calendar.min().date()) if len(calendar) else None,
            "last": str(calendar.max().date()) if len(calendar) else None,
        },
        "roster": {
            "stocks": int(len(roster)),
            "with_name": int((roster["name_source"] != "none").sum()),
            "name_source": roster["name_source"].value_counts().sort_index().to_dict(),
            "price_label": roster["price_label"].value_counts().sort_index().to_dict(),
        },
        **checks,
        "holdings_overlap": overlap,
    }
    # The roster is written first: a failed JSON write leaves the cheaper file, not half a pair.
    atomic_write_text(out_dir / OUTPUT_ROSTER, roster.to_csv(index=False))
    write_json(out_dir / OUTPUT_JSON, payload)
    return payload
