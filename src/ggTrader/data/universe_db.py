"""Versioned index membership in PostgreSQL: apply the DDL, load a vintage, verify one.

The write path of ``ggt db load-sp500`` and the checks of ``ggt db membership verify`` (spec
docs/superpowers/specs/2026-11-07-membership-database-design.md, sections 1, 2 and 8, Appendix B).
Everything here touches schema ``universe`` only. Pure logic (parsing, intervals, hashes, the
refusal rules) lives in :mod:`ggTrader.data.universe_algebra`.

Connections come from ``GGT_MEMBERSHIP_LOADER_URL`` when set, else from the project's own
``get_db_connection_string``. The connection string is never printed or logged.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor
from psycopg2.extras import Json, execute_values

from ggTrader.data.core.index_constituents import (
    load_sp500_history,
    normalize_yf_ticker,
    sp500_members_asof,
)
from ggTrader.data.universe_algebra import (
    NORMALIZER_VERSION,
    DiffClassification,
    History,
    HistoryValidationError,
    Interval,
    as_date,
    classify_changes,
    collapse,
    content_sha256,
    diff_histories,
    history_members_asof,
    history_stats,
    parse_history_csv,
    plan_supersession,
    raw_sha256,
    reconstruct,
    refusal_reasons,
)
from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.io import atomic_write_text
from ggTrader.utils.paths import REPO_ROOT

#: The only index this slice loads (the tables are shaped for more; spec B10).
INDEX_NAME = "sp500"
#: Versioned, idempotent DDL (spec B11); shipped in git, applied by :func:`apply_ddl`.
SCHEMA_SQL = REPO_ROOT / "sql" / "universe_schema.sql"
#: Environment variable that points the loader at its own login role (default: existing URL).
LOADER_URL_ENV = "GGT_MEMBERSHIP_LOADER_URL"
#: Statement timeout (ms) for the read-only sessions used by verify and dry runs.
READ_STATEMENT_TIMEOUT_MS = 120_000
#: The sources the ``loads.source`` CHECK accepts for a loader-written vintage.
LOAD_SOURCES = ("fja05680/sp500", "fixture-import")


class LoadRefused(RuntimeError):
    """A load was refused (suspicious diff, stale parent, bad hash); nothing was written."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class LoadRequest:
    """Everything one ``load-sp500`` run needs besides the connection."""

    raw: bytes
    source: str
    source_url: str
    retrieved_at: datetime
    upstream_commit: str | None = None
    upstream_etag: str | None = None
    note: str | None = None
    expect_sha256: str | None = None
    #: ``None`` = no pin; ``"none"`` = expect an empty index; an int = expect that latest load.
    parent: int | Literal["none"] | None = None
    force: bool = False
    force_reason: str | None = None
    commit: bool = False
    loader_git_sha: str = "unknown"
    index_name: str = INDEX_NAME


@dataclass
class LoadReport:
    """What a load did (commit) or would do (dry run); ``--report-json`` saves it."""

    committed: bool
    no_op: bool
    index_name: str
    load_id: int | None
    parent_load_id: int | None
    raw_sha256: str
    content_sha256: str
    coverage_start: date
    coverage_end: date
    parent_coverage: tuple[date, date] | None
    n_snapshot_dates: int
    n_intervals: int
    n_symbols: int
    min_members: int
    max_members: int
    rows_added: int
    rows_superseded: int
    dates_only_in_new: list[date] = field(default_factory=list)
    dates_only_in_parent: list[date] = field(default_factory=list)
    dates_changed: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    historical_rewrites: list[str] = field(default_factory=list)
    tail_changes: list[str] = field(default_factory=list)
    renames: list[tuple[str, str]] = field(default_factory=list)
    refusal_reasons: list[str] = field(default_factory=list)
    forced: bool = False

    def to_json(self) -> str:
        """Stable JSON text of the report (dates as ISO strings)."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def save(self, path: Path) -> None:
        """Write the JSON report atomically."""
        atomic_write_text(path, self.to_json())


@dataclass
class VintageData:
    """One vintage as stored: its ``loads`` row, snapshot dates and visible interval rows."""

    load: dict[str, Any]
    dates: dict[date, int]
    rows: list[tuple[int, Interval]]

    @property
    def intervals(self) -> list[Interval]:
        """The visible intervals without their row ids."""
        return [interval for _, interval in self.rows]

    def history(self) -> History:
        """Rebuild every snapshot of this vintage from its intervals."""
        return reconstruct(self.intervals, self.dates)


@dataclass
class VerifyCheck:
    """One named verification result."""

    name: str
    ok: bool
    detail: str


@dataclass
class VerifyReport:
    """The outcome of :func:`verify_vintage`: ``ok`` only when every check passed."""

    load_id: int
    checks: list[VerifyCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every check passed."""
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        """Append one check result."""
        self.checks.append(VerifyCheck(name, ok, detail))


def to_psycopg_url(url: str) -> str:
    """SQLAlchemy URL -> libpq URL (drops the ``+psycopg2`` driver suffix)."""
    return url.replace("postgresql+psycopg2://", "postgresql://", 1)


def loader_url() -> str:
    """The URL the loader uses: ``GGT_MEMBERSHIP_LOADER_URL`` or the project's connection."""
    return to_psycopg_url(os.environ.get(LOADER_URL_ENV) or get_db_connection_string())


def connect(url: str | None = None, *, read_only: bool = False) -> PgConnection:
    """Open a psycopg2 connection; a read-only one is server-enforced (the ``bars.py`` pattern)."""
    options = f"-c statement_timeout={READ_STATEMENT_TIMEOUT_MS}" if read_only else ""
    if read_only:
        options += " -c default_transaction_read_only=on"
    return psycopg2.connect(
        url or loader_url(), application_name="ggt-universe", options=options or None
    )


def git_sha() -> str:
    """Short HEAD sha of this checkout, with ``-dirty`` for local edits; ``unknown`` without git."""
    try:
        cwd = REPO_ROOT
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def apply_ddl(conn: PgConnection, sql_file: Path = SCHEMA_SQL) -> None:
    """Run the idempotent DDL file in one transaction (spec step M0); safe to repeat."""
    sql = sql_file.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _dict_row(cur: PgCursor) -> dict[str, Any] | None:
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d.name for d in cur.description], row, strict=True))


def latest_load(cur: PgCursor, index_name: str) -> dict[str, Any] | None:
    """The newest ``loads`` row of an index, or None when nothing was loaded."""
    cur.execute(
        "SELECT * FROM universe.loads WHERE index_name = %s ORDER BY load_id DESC LIMIT 1",
        (index_name,),
    )
    return _dict_row(cur)


def read_vintage(cur: PgCursor, load_id: int, index_name: str = INDEX_NAME) -> VintageData:
    """Read a vintage with the Q2 visibility rule: ``load_id <= v`` and not superseded by v."""
    cur.execute(
        "SELECT * FROM universe.loads WHERE load_id = %s AND index_name = %s",
        (load_id, index_name),
    )
    load = _dict_row(cur)
    if load is None:
        raise LookupError(f"no load {load_id} for index {index_name!r}")
    cur.execute(
        "SELECT snapshot_date, n_members FROM universe.snapshot_dates WHERE load_id = %s",
        (load_id,),
    )
    dates = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute(
        "SELECT interval_id, symbol_raw, valid_from, valid_to FROM universe.membership "
        "WHERE index_name = %s AND load_id <= %s "
        "AND (superseded_by_load_id IS NULL OR superseded_by_load_id > %s) "
        "ORDER BY symbol_raw, valid_from",
        (index_name, load_id, load_id),
    )
    rows = [(r[0], Interval(r[1], r[2], r[3])) for r in cur.fetchall()]
    return VintageData(load=load, dates=dates, rows=rows)


def _yf_map(history: History) -> dict[str, str]:
    """Raw symbol -> yfinance form, refusing a vintage where two raw symbols collide on a date."""
    mapping: dict[str, str] = {}
    for day, tickers in history.items():
        seen: dict[str, str] = {}
        for raw in tickers:
            yf = mapping.setdefault(raw, normalize_yf_ticker(raw))
            if yf in seen and seen[yf] != raw:
                raise HistoryValidationError(
                    f"{seen[yf]!r} and {raw!r} both map to {yf!r} on {day}"
                )
            seen[yf] = raw
    return mapping


def _prepare(req: LoadRequest) -> tuple[History, list[Interval], dict[str, str], str, str]:
    """Validate the input and derive its intervals, yfinance map and both hashes."""
    if req.source not in LOAD_SOURCES:
        raise LoadRefused([f"source must be one of {LOAD_SOURCES}, got {req.source!r}"])
    if req.retrieved_at.tzinfo is None:
        raise LoadRefused(["retrieved_at must be timezone-aware (an instant, not a naive clock)"])
    raw_hash = raw_sha256(req.raw)
    if req.expect_sha256 is not None and req.expect_sha256 != raw_hash:
        raise LoadRefused([f"raw sha256 {raw_hash} != --expect-sha256 {req.expect_sha256}"])
    if req.force and not req.force_reason:
        raise LoadRefused(["--force needs --reason"])
    history = parse_history_csv(req.raw)
    yf = _yf_map(history)
    intervals = collapse(history)
    if reconstruct(intervals, history) != history:
        raise HistoryValidationError("snapshot -> interval -> snapshot round trip is not identical")
    return history, intervals, yf, raw_hash, content_sha256(history)


def _new_report(
    req: LoadRequest,
    history: History,
    intervals: list[Interval],
    raw_hash: str,
    content_hash: str,
) -> LoadReport:
    stats = history_stats(history)
    return LoadReport(
        committed=False,
        no_op=False,
        index_name=req.index_name,
        load_id=None,
        parent_load_id=None,
        raw_sha256=raw_hash,
        content_sha256=content_hash,
        coverage_start=stats["coverage_start"],  # type: ignore[arg-type]
        coverage_end=stats["coverage_end"],  # type: ignore[arg-type]
        parent_coverage=None,
        n_snapshot_dates=stats["n_snapshot_dates"],  # type: ignore[arg-type]
        n_intervals=len(intervals),
        n_symbols=stats["n_symbols"],  # type: ignore[arg-type]
        min_members=stats["min_members"],  # type: ignore[arg-type]
        max_members=stats["max_members"],  # type: ignore[arg-type]
        rows_added=len(intervals),
        rows_superseded=0,
    )


def load_vintage(conn: PgConnection, req: LoadRequest) -> LoadReport:
    """Validate a constituent file and write it as the next vintage (dry run unless ``commit``).

    One transaction. The advisory lock is its first statement; the latest vintage is re-read
    after the lock and compared with ``req.parent`` (compare and set). Raises
    :class:`LoadRefused` or :class:`HistoryValidationError` with nothing written.
    """
    history, intervals, yf, raw_hash, content_hash = _prepare(req)
    report = _new_report(req, history, intervals, raw_hash, content_hash)
    conn.rollback()
    previous_readonly = conn.readonly
    conn.readonly = not req.commit
    try:
        with conn.cursor() as cur:
            if req.commit:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('universe:' || %s))", (req.index_name,)
                )
            latest = latest_load(cur, req.index_name)
            _check_parent_pin(req, latest)
            to_supersede: list[int] = []
            to_insert = intervals
            if latest is not None:
                parent = read_vintage(cur, latest["load_id"], req.index_name)
                report.parent_load_id = latest["load_id"]
                report.parent_coverage = (latest["coverage_start"], latest["coverage_end"])
                if latest["raw_sha256"] == raw_hash:
                    report.no_op = True
                    report.rows_added = 0
                    if req.commit:
                        _insert_check(cur, req, raw_hash, "unchanged", latest["load_id"])
                        conn.commit()
                        report.committed = True
                    return report
                to_supersede, to_insert = plan_supersession(parent.rows, intervals)
                _fill_parent_diff(report, parent, history, intervals)
            report.rows_added, report.rows_superseded = len(to_insert), len(to_supersede)
            report.forced = bool(report.refusal_reasons) and req.force
            if report.refusal_reasons and not req.force:
                raise LoadRefused(report.refusal_reasons)
            if not req.commit:
                return report
            load_id = _write_vintage(cur, req, report, history, yf, to_supersede, to_insert)
            _verify_readback(cur, load_id, req.index_name, history, content_hash)
            _insert_check(cur, req, raw_hash, "loaded", load_id, report)
            conn.commit()
            report.load_id, report.committed = load_id, True
            return report
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.rollback()
        conn.readonly = previous_readonly


def _check_parent_pin(req: LoadRequest, latest: dict[str, Any] | None) -> None:
    if req.parent is None:
        return
    expected = None if req.parent == "none" else int(req.parent)
    actual = None if latest is None else latest["load_id"]
    if expected != actual:
        raise LoadRefused([f"the latest vintage is {actual}, but the run was pinned to {expected}"])


def _fill_parent_diff(
    report: LoadReport, parent: VintageData, history: History, intervals: list[Interval]
) -> None:
    """Add the diff against the parent vintage to the report and collect refusal reasons."""
    parent_history = parent.history()
    diff = diff_histories(parent_history, history)
    report.dates_only_in_parent = diff.only_in_a
    report.dates_only_in_new = diff.only_in_b
    report.dates_changed = {
        day.isoformat(): {
            "removed": sorted(set(old) - set(new)),
            "added": sorted(set(new) - set(old)),
        }
        for day, (old, new) in diff.changed.items()
    }
    coverage_end = parent.load["coverage_end"]
    classification: DiffClassification = classify_changes(parent.intervals, intervals, coverage_end)
    report.historical_rewrites = list(classification.historical)
    report.tail_changes = list(classification.tail)
    report.renames = [tuple(pair) for pair in classification.renames]  # type: ignore[misc]
    report.refusal_reasons = refusal_reasons(
        parent_coverage=(parent.load["coverage_start"], coverage_end),
        new_coverage=(report.coverage_start, report.coverage_end),
        classification=classification,
    )


def _write_vintage(
    cur: PgCursor,
    req: LoadRequest,
    report: LoadReport,
    history: History,
    yf: dict[str, str],
    to_supersede: list[int],
    to_insert: list[Interval],
) -> int:
    """Insert the load, its bytes, snapshot dates and interval changes; return the new load id."""
    cur.execute(
        "INSERT INTO universe.loads (index_name, source, source_url, upstream_commit, "
        "upstream_etag, retrieved_at, raw_sha256, content_sha256, normalizer_version, "
        "coverage_start, coverage_end, n_snapshot_dates, n_intervals, n_symbols, min_members, "
        "max_members, rows_added, rows_superseded, parent_load_id, forced, force_reason, "
        "loader_git_sha, note) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::date, %s::date, "
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING load_id",
        (
            req.index_name,
            req.source,
            req.source_url,
            req.upstream_commit,
            req.upstream_etag,
            req.retrieved_at,
            report.raw_sha256,
            report.content_sha256,
            NORMALIZER_VERSION,
            as_date(report.coverage_start),
            as_date(report.coverage_end),
            report.n_snapshot_dates,
            report.n_intervals,
            report.n_symbols,
            report.min_members,
            report.max_members,
            len(to_insert),
            len(to_supersede),
            report.parent_load_id,
            report.forced,
            req.force_reason if report.forced else None,
            req.loader_git_sha,
            req.note,
        ),
    )
    load_id: int = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO universe.load_blobs (load_id, raw_gzip) VALUES (%s, %s)",
        (load_id, psycopg2.Binary(gzip.compress(req.raw, mtime=0))),
    )
    execute_values(
        cur,
        "INSERT INTO universe.snapshot_dates (load_id, snapshot_date, n_members) VALUES %s",
        [(load_id, as_date(day), len(tickers)) for day, tickers in sorted(history.items())],
        template="(%s, %s::date, %s)",
    )
    if to_supersede:
        # Supersede first: the exclusion constraint only lets one live interval per symbol stand.
        cur.execute(
            "UPDATE universe.membership SET superseded_by_load_id = %s "
            "WHERE interval_id = ANY(%s) AND superseded_by_load_id IS NULL",
            (load_id, to_supersede),
        )
        if cur.rowcount != len(to_supersede):
            raise LoadRefused(["a row to supersede was already superseded (concurrent change)"])
    if to_insert:
        execute_values(
            cur,
            "INSERT INTO universe.membership (index_name, symbol_raw, symbol_yf, valid_from, "
            "valid_to, load_id) VALUES %s",
            [
                (
                    req.index_name,
                    i.symbol_raw,
                    yf[i.symbol_raw],
                    as_date(i.valid_from),
                    None if i.valid_to is None else as_date(i.valid_to),
                    load_id,
                )
                for i in to_insert
            ],
            template="(%s, %s, %s, %s::date, %s::date, %s)",
        )
    return load_id


def _verify_readback(
    cur: PgCursor, load_id: int, index_name: str, history: History, content_hash: str
) -> None:
    """Read the new vintage back with the Q2 rule; any difference from the input aborts the load."""
    back = read_vintage(cur, load_id, index_name)
    rebuilt = back.history()
    if rebuilt != history or content_sha256(rebuilt) != content_hash:
        raise LoadRefused(["read-back of the new vintage differs from the input; rolled back"])
    if len(back.rows) != back.load["n_intervals"]:
        raise LoadRefused(["visible interval count differs from n_intervals; rolled back"])


def _insert_check(
    cur: PgCursor,
    req: LoadRequest,
    raw_hash: str,
    outcome: str,
    matched_load_id: int,
    report: LoadReport | None = None,
) -> None:
    detail = None
    if report is not None:
        detail = {
            "rows_added": report.rows_added,
            "rows_superseded": report.rows_superseded,
            "forced": report.forced,
        }
    cur.execute(
        "INSERT INTO universe.checks (index_name, raw_sha256, upstream_commit, outcome, "
        "matched_load_id, detail) VALUES (%s, %s, %s, %s, %s, %s)",
        (
            req.index_name,
            raw_hash,
            req.upstream_commit,
            outcome,
            matched_load_id,
            Json(detail) if detail is not None else None,
        ),
    )


def verify_vintage(
    conn: PgConnection, load_id: int, csv_path: Path, index_name: str = INDEX_NAME
) -> VerifyReport:
    """Compare a stored vintage with a constituent file, snapshot date by snapshot date.

    Read-only. Checks the stored bytes and both hashes, that the stored snapshot dates equal the
    file's, that the intervals reconstruct every snapshot (set-equal per date), that membership
    on every calendar day equals the project's own ``sp500_members_asof`` on the file, and that
    the SQL function ``universe.snapshot`` agrees with the Python reconstruction.
    """
    report = VerifyReport(load_id=load_id)
    raw = (
        csv_path.read_bytes()
        if csv_path.suffix != ".gz"
        else gzip.decompress(csv_path.read_bytes())
    )
    file_history = parse_history_csv(raw)
    conn.rollback()
    previous_readonly = conn.readonly
    conn.readonly = True
    try:
        with conn.cursor() as cur:
            vintage = read_vintage(cur, load_id, index_name)
            load = vintage.load
            report.add(
                "loads row",
                True,
                f"index={load['index_name']} source={load['source']} "
                f"coverage={load['coverage_start']}..{load['coverage_end']}",
            )
            _verify_into(report, cur, vintage, raw, file_history, csv_path)
    finally:
        conn.rollback()
        conn.readonly = previous_readonly
    return report


def _verify_into(
    report: VerifyReport,
    cur: PgCursor,
    vintage: VintageData,
    raw: bytes,
    file_history: History,
    csv_path: Path,
) -> None:
    load = vintage.load
    load_id = load["load_id"]
    file_raw_hash = raw_sha256(raw)
    report.add(
        "raw bytes hash",
        load["raw_sha256"] == file_raw_hash,
        f"stored {load['raw_sha256'][:12]} file {file_raw_hash[:12]}",
    )
    cur.execute("SELECT raw_gzip FROM universe.load_blobs WHERE load_id = %s", (load_id,))
    blob = cur.fetchone()
    blob_ok = blob is not None and raw_sha256(gzip.decompress(bytes(blob[0]))) == load["raw_sha256"]
    report.add("stored gzip decompresses to raw_sha256", blob_ok, "load_blobs row")

    file_dates = set(file_history)
    stored_dates = set(vintage.dates)
    sizes_ok = all(vintage.dates.get(d) == len(file_history[d]) for d in file_dates & stored_dates)
    report.add(
        "snapshot dates",
        file_dates == stored_dates and sizes_ok and load["n_snapshot_dates"] == len(stored_dates),
        f"file {len(file_dates)} stored {len(stored_dates)} "
        f"(only file {len(file_dates - stored_dates)}, "
        f"only stored {len(stored_dates - file_dates)})",
    )

    rebuilt = vintage.history()
    mismatched = [d for d in sorted(file_dates) if rebuilt.get(d) != file_history[d]]
    report.add(
        "per-date set equality (intervals -> snapshots)",
        not mismatched and len(rebuilt) == len(file_history),
        f"{len(file_history) - len(mismatched)}/{len(file_history)} snapshot dates equal"
        + (f"; first mismatch {mismatched[0]}" if mismatched else ""),
    )

    file_content = content_sha256(file_history)
    report.add(
        "content hash",
        content_sha256(rebuilt) == load["content_sha256"] == file_content,
        f"stored {load['content_sha256'][:12]} rebuilt {content_sha256(rebuilt)[:12]} "
        f"file {file_content[:12]}",
    )

    stats = history_stats(file_history)
    counts_ok = (
        load["n_intervals"] == len(vintage.rows)
        and load["n_symbols"] == stats["n_symbols"]
        and load["min_members"] == stats["min_members"]
        and load["max_members"] == stats["max_members"]
        and load["coverage_start"] == stats["coverage_start"]
        and load["coverage_end"] == stats["coverage_end"]
    )
    report.add(
        "stored counts",
        counts_ok,
        f"intervals {len(vintage.rows)} symbols {stats['n_symbols']} "
        f"members {stats['min_members']}-{stats['max_members']}",
    )

    # Every calendar day, against the project's own file reader (independent of the algebra).
    oracle = load_sp500_history(csv_path) if csv_path.suffix == ".gz" else None
    first, last = min(file_dates), max(file_dates)
    days = [first + timedelta(days=n) for n in range((last - first).days + 1)]
    calendar = reconstruct(vintage.intervals, days)
    bad_days = [d for d in days if calendar[d] != history_members_asof(file_history, d)]
    detail = f"{len(days) - len(bad_days)}/{len(days)} calendar days equal the file's carry-forward"
    if oracle is not None:
        import pandas as pd

        bad_oracle = [
            d for d in days if tuple(sp500_members_asof(pd.Timestamp(d), oracle)) != calendar[d]
        ]
        detail += f"; {len(days) - len(bad_oracle)}/{len(days)} equal sp500_members_asof"
        bad_days += bad_oracle
    report.add("calendar days", not bad_days, detail)

    cur.execute(
        "SELECT snapshot_date, array_agg(symbol_raw ORDER BY symbol_raw) "
        "FROM universe.snapshot(%s, %s) GROUP BY snapshot_date",
        (load["index_name"], load_id),
    )
    sql_history = {r[0]: tuple(r[1]) for r in cur.fetchall()}
    report.add(
        "universe.snapshot() SQL function",
        sql_history == rebuilt,
        f"{len(sql_history)} dates from SQL, {len(rebuilt)} from Python",
    )
