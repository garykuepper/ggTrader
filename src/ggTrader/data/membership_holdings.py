"""``ggt db membership check-holdings``: match the S&P 500 list against the SPDR SPY holdings.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, Appendix C (C1 and C3).
The upstream feed is dead, so the list is attested by an *independent* source: the daily holdings
workbook of the SPDR S&P 500 ETF (SPY), a public file that carries its own "As of" date. One run:

1. downloads the workbook (browser-style User-Agent, redirects followed) and parses it with the
   standard library only (:mod:`ggTrader.utils.xlsx_min`): the "As of DD-Mon-YYYY" text and the
   ``Ticker`` column, dropping cash and other non-equity lines;
2. normalises share classes with ``normalize_yf_ticker`` (``BRK.B`` -> ``BRK-B``, the project
   convention) on both sides and compares the set with the latest vintage's snapshot **on the
   holdings' own as-of date**;
3. appends **one row** to ``universe.checks`` and archives the raw bytes (gzip) in
   ``universe.check_blobs``:

   * ``matched``: the two sets are equal. ``matched_load_id`` is the vintage. This is the only
     outcome that advances the live gate (freshness clock and ``known_through``).
   * ``mismatch``: they differ; the lists of tickers only in our list and only in the holdings are
     in ``detail``. Nothing is loaded or changed.
   * ``fetch_failed``: no usable answer (unreachable, HTTP error, truncated, HTML instead of a
     workbook, wrong structure, an as-of date in the future, more than
     :data:`MAX_HOLDINGS_AGE_DAYS` days old (``stale_source``) or before our coverage).

It never touches ``universe.loads``, ``universe.membership`` or ``universe.snapshot_dates``.
The network call (:func:`fetch_holdings`, ``get`` injectable) and the database (a
:class:`HoldingsStore`) are isolated, so tests run offline and without a server.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

import requests
from psycopg2.extras import Json

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.membership_upstream import CheckResult, PgCheckStore
from ggTrader.data.universe_algebra import (
    MAX_MEMBERS,
    MIN_MEMBERS,
    history_members_asof,
    raw_sha256,
)
from ggTrader.data.universe_db import INDEX_NAME, VintageData, connect
from ggTrader.utils.http_cap import DownloadTooLarge, read_capped
from ggTrader.utils.xlsx_min import XlsxError, read_first_sheet

#: The public SPY daily holdings workbook (State Street / SPDR).
HOLDINGS_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
#: A normal browser identity: the host refuses clients that look like scripts.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
#: Seconds to wait for the download.
TIMEOUT_SECONDS = 60
#: A real workbook is about 100 kB; anything over this is refused rather than stored.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
#: Oldest holdings file a check accepts, in calendar days before the check time. The 45-day
#: freshness clock only bounds the time since the last match, so without this bound an old or
#: cached file would still count as a match and reset that clock.
MAX_HOLDINGS_AGE_DAYS = 7
#: Largest value the override may take: a bigger one would let an old file count as a match.
MAX_HOLDINGS_AGE_CEILING_DAYS = 14
#: Environment variable that overrides :data:`MAX_HOLDINGS_AGE_DAYS` (a whole number of days from
#: 1 to :data:`MAX_HOLDINGS_AGE_CEILING_DAYS`).
HOLDINGS_AGE_ENV = "GGT_HOLDINGS_MAX_AGE_DAYS"
#: Outcomes this module writes to ``universe.checks.outcome`` (the CHECK list allows them).
OUTCOME_MATCHED = "matched"
OUTCOME_MISMATCH = "mismatch"
OUTCOME_FAILED = "fetch_failed"

#: Header cell that marks the table's first row (``Ticker``), compared case-insensitively.
TICKER_HEADER = "ticker"
#: Optional columns used to recognise non-equity lines.
NAME_HEADER = "name"
SECTOR_HEADER = "sector"
#: Tickers that are cash or placeholders, not equities.
CASH_TICKERS = frozenset({"", "-", "--", "USD", "CASH", "N/A", "NA"})
#: Sector labels the fund uses for non-equity lines (equities carry a GICS sector).
NON_EQUITY_SECTORS = frozenset({"cash", "cash and/or derivatives", "other", "derivatives"})
#: What an equity ticker looks like once upper-cased: letters/digits and an optional share class.
_EQUITY_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-/ ][A-Z0-9]{1,2})?$")
_AS_OF = re.compile(r"as\s+of\s*:?\s*(\d{1,2})[-\s/]([A-Za-z]{3})[A-Za-z]*[-\s/,.]*(\d{4})", re.I)
_MONTHS = {
    m: i for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}  # fmt: skip

GetFn = Callable[..., Any]


class HoldingsConfigError(ValueError):
    """The holdings-age override is not a whole number of days in range; the check refuses."""


def max_holdings_age_days(environ: Mapping[str, str] | None = None) -> int:
    """The holdings-age limit in force: the environment override, else the constant."""
    text = (os.environ if environ is None else environ).get(HOLDINGS_AGE_ENV)
    if text is None or text == "":
        return MAX_HOLDINGS_AGE_DAYS
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_HOLDINGS_AGE_CEILING_DAYS:
        raise HoldingsConfigError(
            f"{HOLDINGS_AGE_ENV}={text!r} is not a whole number of days from 1 to "
            f"{MAX_HOLDINGS_AGE_CEILING_DAYS}; unset it to use the default of "
            f"{MAX_HOLDINGS_AGE_DAYS}"
        )
    return value


class HoldingsFetchError(RuntimeError):
    """No usable holdings: the download or its content is unusable.

    ``reason`` is a short machine code; ``raw`` is the bytes that were downloaded, when there
    were any, so the failed run can still archive what the host sent.
    """

    def __init__(self, reason: str, message: str, raw: bytes | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.raw = raw


@dataclass(frozen=True)
class HoldingsFile:
    """The downloaded bytes with where they came from."""

    raw: bytes
    url: str
    final_url: str
    content_type: str | None
    etag: str | None
    last_modified: str | None
    retrieved_at: datetime


@dataclass(frozen=True)
class Holdings:
    """The parsed workbook: the as-of date and the equity tickers (normalised, sorted)."""

    as_of: date
    as_of_text: str
    tickers: tuple[str, ...]
    n_table_rows: int
    excluded: tuple[dict[str, str], ...]


# --------------------------------------------------------------------------- fetch


def fetch_holdings(
    *, get: GetFn | None = None, now: Callable[[], datetime] | None = None
) -> HoldingsFile:
    """Download the SPY holdings workbook; raise :class:`HoldingsFetchError` if it is unusable.

    Redirects are followed. The reply must be HTTP 200 and, when ``Content-Length`` is stated,
    exactly that long (a short body is a truncated download). Whether the bytes are a workbook is
    decided by :func:`parse_holdings`, so a 200 reply with an HTML page still reaches the archive.
    ``get`` is ``requests.get``-compatible (looked up at call time so tests can replace it).
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    get = get or requests.get
    try:
        response = get(
            HOLDINGS_URL,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
            stream=True,  # read under the byte cap instead of buffering the whole reply
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
                "Accept-Encoding": "identity",
            },
        )
    except requests.RequestException as exc:
        raise HoldingsFetchError("unreachable", f"cannot reach ssga.com: {exc}") from None
    retrieved_at = clock()
    if response.status_code != 200:
        raise HoldingsFetchError("http_status", f"the host answered HTTP {response.status_code}")
    try:
        raw = read_capped(response, MAX_DOWNLOAD_BYTES)
    except DownloadTooLarge as exc:
        raise HoldingsFetchError("too_large", f"the download is refused: {exc}") from None
    except requests.RequestException as exc:
        raise HoldingsFetchError("unreachable", f"the download broke off: {exc}") from None
    declared = response.headers.get("content-length")
    if declared is not None and not response.headers.get("content-encoding"):
        if declared.isdigit() and int(declared) != len(raw):
            raise HoldingsFetchError(
                "truncated",
                f"download is {len(raw)} bytes but content-length says {declared}",
                raw,
            )
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    return HoldingsFile(
        raw=raw,
        url=HOLDINGS_URL,
        final_url=str(getattr(response, "url", None) or HOLDINGS_URL),
        content_type=content_type or None,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        retrieved_at=retrieved_at,
    )


# --------------------------------------------------------------------------- parse


def _parse_as_of(text: str) -> date | None:
    match = _AS_OF.search(text)
    if not match:
        return None
    month = _MONTHS.get(match.group(2).lower())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(1)))
    except ValueError:
        return None


def _cell(row: list[str], col: int | None) -> str:
    return row[col].strip() if col is not None and col < len(row) else ""


def normalize_holding_ticker(ticker: str) -> str:
    """The project's ticker convention (``BRK.B`` and ``BRK B`` and ``BRK/B`` -> ``BRK-B``)."""
    return normalize_yf_ticker(ticker.replace("/", ".").replace(" ", "."))


def parse_holdings(raw: bytes) -> Holdings:
    """Parse the workbook bytes; raise :class:`HoldingsFetchError` for anything unusable.

    The header row is the first row with a cell equal to ``Ticker``; the as-of date is the first
    "As of DD-Mon-YYYY" text anywhere above it (else anywhere in the sheet). Cash, blank and
    other non-equity lines are dropped and listed in ``excluded`` (a lone footnote cell is
    skipped). A duplicate ticker, or a count outside the loader's 480..520 member range, means
    the structure is not what we expect.
    """
    try:
        rows = read_first_sheet(raw)
    except XlsxError as exc:
        reason = "not_xlsx" if "not a zip" in str(exc) else "invalid"
        raise HoldingsFetchError(reason, f"download is not a usable workbook: {exc}", raw) from None
    header_at = next(
        (
            i
            for i, row in enumerate(rows)
            if any(cell.strip().lower() == TICKER_HEADER for cell in row)
        ),
        None,
    )
    if header_at is None:
        raise HoldingsFetchError("invalid", "no header row with a 'Ticker' cell", raw)
    header = [cell.strip().lower() for cell in rows[header_at]]
    ticker_col = header.index(TICKER_HEADER)
    name_col = header.index(NAME_HEADER) if NAME_HEADER in header else None
    sector_col = header.index(SECTOR_HEADER) if SECTOR_HEADER in header else None
    as_of, as_of_text = None, ""
    for scope in (rows[:header_at], rows[header_at:]):
        for row in scope:
            for cell in row:
                found = _parse_as_of(cell)
                if found is not None:
                    as_of, as_of_text = found, cell.strip()
                    break
            if as_of is not None:
                break
        if as_of is not None:
            break
    if as_of is None:
        raise HoldingsFetchError("invalid", "no 'As of DD-Mon-YYYY' date in the workbook", raw)
    tickers: list[str] = []
    excluded: list[dict[str, str]] = []
    table_rows = 0
    for row in rows[header_at + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        if sum(1 for cell in row if cell.strip()) == 1 and not _cell(row, ticker_col):
            continue  # a lone text cell with no ticker is a footnote, not a holdings line
        table_rows += 1
        ticker, name, sector = (_cell(row, c) for c in (ticker_col, name_col, sector_col))
        upper = ticker.upper()
        if (
            upper in CASH_TICKERS
            or sector.lower() in NON_EQUITY_SECTORS
            or not _EQUITY_TICKER.match(upper)
        ):
            excluded.append({"ticker": ticker, "name": name, "sector": sector})
            continue
        tickers.append(normalize_holding_ticker(ticker))
    if len(tickers) != len(set(tickers)):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise HoldingsFetchError("invalid", f"duplicate tickers in the holdings: {dupes}", raw)
    if not MIN_MEMBERS <= len(tickers) <= MAX_MEMBERS:
        raise HoldingsFetchError(
            "invalid",
            f"{len(tickers)} equity tickers is outside {MIN_MEMBERS}..{MAX_MEMBERS}; "
            "the workbook is not the SPY holdings table",
            raw,
        )
    return Holdings(as_of, as_of_text, tuple(sorted(tickers)), table_rows, tuple(excluded))


# --------------------------------------------------------------------------- compare


@dataclass
class HoldingsResult(CheckResult):
    """One run's outcome: a ``CheckResult`` plus the raw bytes to archive (None = none arrived).

    ``outcome`` is ``matched``, ``mismatch`` or ``fetch_failed``; ``holdings_as_of`` is the
    workbook's own date when it was readable.
    """

    raw: bytes | None = field(default=None, repr=False)
    holdings_as_of: date | None = None

    @property
    def exit_code(self) -> int:
        """0 matched, 2 mismatch (needs a human), 1 no usable answer."""
        return {OUTCOME_MATCHED: 0, OUTCOME_MISMATCH: 2}.get(self.outcome, 1)


def _file_detail(file: HoldingsFile) -> dict[str, Any]:
    return {
        "source_url": file.url,
        "final_url": file.final_url,
        "content_type": file.content_type,
        "etag": file.etag,
        "last_modified": file.last_modified,
        "retrieved_at": file.retrieved_at.astimezone(timezone.utc).isoformat(),
        "raw_bytes": len(file.raw),
        "raw_sha256": raw_sha256(file.raw),
    }


def failed_holdings(
    error: HoldingsFetchError, *, holdings_as_of: date | None = None, extra: dict | None = None
) -> HoldingsResult:
    """The ``fetch_failed`` record for an unusable download (bytes kept when there were any)."""
    detail: dict[str, Any] = {"reason": error.reason, "error": str(error), **(extra or {})}
    raw = error.raw
    if raw is not None:
        detail["raw_sha256"] = raw_sha256(raw)
        detail["raw_bytes"] = len(raw)
    if holdings_as_of is not None:
        detail["holdings_as_of"] = holdings_as_of.isoformat()
    return HoldingsResult(
        OUTCOME_FAILED,
        "fetch_failed",
        None,
        raw_sha256(raw) if raw is not None else None,
        None,
        detail,
        raw=raw,
        holdings_as_of=holdings_as_of,
    )


def compare_holdings(
    file: HoldingsFile,
    latest: VintageData | None,
    *,
    today: date,
    max_age_days: int | None = None,
) -> HoldingsResult:
    """Compare the workbook with the latest vintage on the holdings' as-of date; pure.

    Order of judgement: unreadable workbook -> ``fetch_failed``; as-of date after ``today``,
    more than ``max_age_days`` (default :func:`max_holdings_age_days`) before it
    (``stale_source``) or before the vintage's first snapshot -> ``fetch_failed`` (no honest
    comparison exists, and none of these counts as a match); no vintage loaded or a different
    set -> ``mismatch``; equal sets -> ``matched``. The raw bytes are kept in every case.
    """
    limit = max_holdings_age_days() if max_age_days is None else max_age_days
    raw_hash = raw_sha256(file.raw)
    try:
        holdings = parse_holdings(file.raw)
    except HoldingsFetchError as exc:
        return failed_holdings(exc, extra=_file_detail(file))
    detail: dict[str, Any] = {
        **_file_detail(file),
        "holdings_as_of": holdings.as_of.isoformat(),
        "holdings_as_of_text": holdings.as_of_text,
        "n_table_rows": holdings.n_table_rows,
        "n_holdings": len(holdings.tickers),
        "n_excluded": len(holdings.excluded),
        "excluded_lines": list(holdings.excluded),
        "ticker_convention": "normalize_yf_ticker (BRK.B -> BRK-B), both sides",
    }
    if holdings.as_of > today:
        error = HoldingsFetchError(
            "as_of_in_future",
            f"holdings are dated {holdings.as_of}, after today ({today})",
            file.raw,
        )
        return failed_holdings(error, holdings_as_of=holdings.as_of, extra=detail)
    age_days = (today - holdings.as_of).days
    detail.update(holdings_age_days=age_days, max_holdings_age_days=limit)
    if holdings.as_of < today - timedelta(days=limit):
        error = HoldingsFetchError(
            "stale_source",
            f"holdings are dated {holdings.as_of}, {age_days} days before {today} "
            f"(limit {limit}); an old or cached file cannot count as a match",
            file.raw,
        )
        return failed_holdings(error, holdings_as_of=holdings.as_of, extra=detail)
    if latest is None:
        detail["reason"] = "no vintage is loaded, so there is nothing to compare with"
        detail["only_in_holdings"] = list(holdings.tickers)
        detail["only_in_list"] = []
        return HoldingsResult(
            OUTCOME_MISMATCH, "mismatch", None, raw_hash, None, detail,
            raw=file.raw, holdings_as_of=holdings.as_of,
        )  # fmt: skip
    load = latest.load
    detail.update(
        compared_load_id=load["load_id"],
        vintage_coverage_start=load["coverage_start"].isoformat(),
        vintage_coverage_end=load["coverage_end"].isoformat(),
    )
    if holdings.as_of < load["coverage_start"]:
        error = HoldingsFetchError(
            "as_of_before_coverage",
            f"holdings are dated {holdings.as_of}, before the list's first snapshot "
            f"{load['coverage_start']}; nothing to compare",
            file.raw,
        )
        return failed_holdings(error, holdings_as_of=holdings.as_of, extra=detail)
    history = latest.history()
    snapshot_day = max(d for d in history if d <= holdings.as_of)
    ours = {normalize_yf_ticker(t) for t in history_members_asof(history, holdings.as_of)}
    theirs = set(holdings.tickers)
    only_in_list, only_in_holdings = sorted(ours - theirs), sorted(theirs - ours)
    detail.update(
        compared_snapshot_date=snapshot_day.isoformat(),
        n_list=len(ours),
        only_in_list=only_in_list,
        only_in_holdings=only_in_holdings,
        holdings_as_of_before_coverage_end=holdings.as_of < load["coverage_end"],
    )
    if not only_in_list and not only_in_holdings:
        detail["known_through"] = max(holdings.as_of, load["coverage_end"]).isoformat()
        return HoldingsResult(
            OUTCOME_MATCHED, "matched", load["load_id"], raw_hash, None, detail,
            raw=file.raw, holdings_as_of=holdings.as_of,
        )  # fmt: skip
    return HoldingsResult(
        OUTCOME_MISMATCH, "mismatch", None, raw_hash, None, detail,
        raw=file.raw, holdings_as_of=holdings.as_of,
    )  # fmt: skip


# --------------------------------------------------------------------------- store and runner


class HoldingsStore(Protocol):
    """The two database operations a holdings check needs."""

    def latest(self, index_name: str) -> VintageData | None:
        """The newest vintage of the index, or None when nothing is loaded."""
        ...

    def record(self, index_name: str, result: HoldingsResult) -> int:
        """Append one ``universe.checks`` row (and its blob) and return its ``check_id``."""
        ...


class PgHoldingsStore(PgCheckStore):
    """``HoldingsStore`` on the real database: a read-only read, then one transaction of INSERTs.

    The ``checks`` row and its ``check_blobs`` row go in together or not at all. These are the
    only writes this module makes.
    """

    def insert(self, cur: Any, index_name: str, result: HoldingsResult) -> int:
        """INSERT the check row and, when bytes arrived, their gzip; return the ``check_id``.

        Runs on the caller's cursor and commits nothing, so a caller (the weekly job) can add
        more rows in the same transaction.
        """
        detail = {**result.detail, "result": result.result}
        cur.execute(
            "INSERT INTO universe.checks (index_name, raw_sha256, upstream_commit, "
            "outcome, matched_load_id, detail) VALUES (%s, %s, %s, %s, %s, %s) "
            "RETURNING check_id",
            (
                index_name,
                result.raw_sha256,
                None,
                result.outcome,
                result.matched_load_id,
                Json(json.loads(json.dumps(detail, default=str))),
            ),
        )
        check_id = int(cur.fetchone()[0])
        if result.raw is not None:
            cur.execute(
                "INSERT INTO universe.check_blobs (check_id, raw_sha256, raw_gzip) "
                "VALUES (%s, %s, %s)",
                (
                    check_id,
                    hashlib.sha256(result.raw).hexdigest(),
                    gzip.compress(result.raw, mtime=0),
                ),
            )
        return check_id

    def record(self, index_name: str, result: HoldingsResult) -> int:  # type: ignore[override]
        """Append the check row and, when bytes arrived, their gzip (deterministic, mtime 0)."""
        conn = connect(self._url)
        try:
            with conn.cursor() as cur:
                check_id = self.insert(cur, index_name, result)
            conn.commit()
            return check_id
        finally:
            conn.close()


@dataclass
class HoldingsRun:
    """What ``check_holdings`` did: the result, the recorded row id, and any record error."""

    result: HoldingsResult
    check_id: int | None
    record_error: str | None = None


def check_holdings(
    store: HoldingsStore,
    *,
    index_name: str = INDEX_NAME,
    fetch: Callable[[], HoldingsFile] = fetch_holdings,
    today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
) -> HoldingsRun:
    """Fetch the holdings, compare with the latest vintage and append one ``checks`` row.

    Never raises for a bad download: that is recorded as ``fetch_failed``. If the row itself
    cannot be written the result is still returned with ``record_error`` set (the live gate then
    simply does not see a fresh match). Reading the latest vintage may raise (database down), and
    so does a malformed ``GGT_HOLDINGS_MAX_AGE_DAYS`` (before anything is downloaded).
    """
    limit = max_holdings_age_days()
    try:
        file = fetch()
    except HoldingsFetchError as exc:
        result = failed_holdings(exc)
    else:
        result = compare_holdings(file, store.latest(index_name), today=today(), max_age_days=limit)
    try:
        check_id = store.record(index_name, result)
    except Exception as exc:  # noqa: BLE001 - report, do not hide the comparison
        return HoldingsRun(result, None, f"{type(exc).__name__}: could not record the check row")
    return HoldingsRun(result, check_id)
