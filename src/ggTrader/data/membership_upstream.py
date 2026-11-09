"""``ggt db membership check-upstream``: compare the public upstream list with the latest vintage.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, sections 1.2 (``checks``),
2 (no-op rule) and 5. The check fetches the fja05680/sp500 history file at its newest commit,
validates it like the loader does, compares it with the latest vintage and appends **one row**
to ``universe.checks``:

* ``unchanged``: the upstream bytes (or, failing that, the parsed content) equal the latest
  vintage. Informational only since slice M4b: the upstream feed is dead and the live gate
  counts from ``check-holdings`` (the SPY match), so this outcome extends no clock.
* ``refused``: upstream differs. It is reported with the diff summary and left alone; in this
  slice nothing is ever loaded automatically (``detail.result = "differs"``).
* ``fetch_failed``: the download was unreachable, not a CSV, truncated or invalid.

The check never touches ``universe.loads`` or ``universe.membership``. The network call is
isolated in :func:`fetch_upstream` (its ``get`` argument is injectable), and the database in a
:class:`CheckStore`, so tests run offline and without a server.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import quote

import requests
from psycopg2.extras import Json

from ggTrader.data.universe_algebra import (
    DiffClassification,
    History,
    HistoryValidationError,
    classify_changes,
    collapse,
    content_sha256,
    diff_histories,
    parse_history_csv,
    raw_sha256,
    refusal_reasons,
)
from ggTrader.data.universe_db import INDEX_NAME, VintageData, connect, latest_load, read_vintage
from ggTrader.utils.http_cap import DownloadTooLarge, read_capped

#: The upstream repository and file the sp500 history comes from (MIT licence).
UPSTREAM_REPO = "fja05680/sp500"
UPSTREAM_PATH = "S&P 500 Historical Components & Changes (Updated).csv"
COMMITS_URL = f"https://api.github.com/repos/{UPSTREAM_REPO}/commits"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}"
#: Seconds to wait for each request.
TIMEOUT_SECONDS = 60
#: A real history file is a few MB; anything over this is refused rather than stored.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
#: Content types a CSV may legitimately arrive as (GitHub raw serves ``text/plain``).
ACCEPTED_CONTENT_TYPES = ("text/plain", "text/csv")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

GetFn = Callable[..., Any]


class UpstreamFetchError(RuntimeError):
    """The upstream file could not be fetched or is not a usable history file."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class UpstreamFile:
    """The upstream bytes with where they came from."""

    raw: bytes
    url: str
    commit: str
    etag: str | None
    retrieved_at: datetime


def raw_url(commit: str) -> str:
    """The raw-file URL of the history file at one commit (bytes and commit always match)."""
    return f"{RAW_BASE}/{commit}/{quote(UPSTREAM_PATH)}"


def fetch_upstream(
    *, get: GetFn | None = None, now: Callable[[], datetime] | None = None
) -> UpstreamFile:
    """Fetch the newest upstream history file; raise :class:`UpstreamFetchError` if unusable.

    Resolves the newest commit that touched the file through the GitHub commits API, then reads
    the file at exactly that commit. The body must be a plain-text 200 response whose length
    equals ``Content-Length``, ends in a newline and parses as a valid ``date,tickers`` history;
    anything else is a failure, never a partial result. ``get`` is ``requests.get``-compatible.
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    get = get or requests.get  # looked up at call time so tests can replace it
    commit = _latest_commit(get)
    url = raw_url(commit)
    try:
        # identity encoding keeps Content-Length equal to the body length we count
        response = get(
            url, timeout=TIMEOUT_SECONDS, stream=True, headers={"Accept-Encoding": "identity"}
        )
    except requests.RequestException as exc:
        raise UpstreamFetchError("unreachable", f"cannot reach {UPSTREAM_REPO}: {exc}") from None
    retrieved_at = clock()
    if response.status_code != 200:
        raise UpstreamFetchError("http_status", f"upstream answered HTTP {response.status_code}")
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type not in ACCEPTED_CONTENT_TYPES:
        raise UpstreamFetchError(
            "content_type", f"upstream sent content-type {content_type!r}, not a CSV"
        )
    try:
        raw = read_capped(response, MAX_DOWNLOAD_BYTES)
    except DownloadTooLarge as exc:
        raise UpstreamFetchError("too_large", f"the download is refused: {exc}") from None
    except requests.RequestException as exc:
        raise UpstreamFetchError("unreachable", f"the download broke off: {exc}") from None
    declared = response.headers.get("content-length")
    if declared is not None and not response.headers.get("content-encoding"):
        if declared.isdigit() and int(declared) != len(raw):
            raise UpstreamFetchError(
                "truncated", f"download is {len(raw)} bytes but content-length says {declared}"
            )
    if not raw.endswith(b"\n"):
        raise UpstreamFetchError("truncated", "download does not end with a newline")
    try:
        parse_history_csv(raw)
    except HistoryValidationError as exc:
        raise UpstreamFetchError(
            "invalid", f"download is not a valid history file: {exc}"
        ) from None
    return UpstreamFile(raw, url, commit, response.headers.get("etag"), retrieved_at)


def _latest_commit(get: GetFn) -> str:
    try:
        response = get(
            COMMITS_URL,
            params={"path": UPSTREAM_PATH, "per_page": 1},
            timeout=TIMEOUT_SECONDS,
            headers={"Accept": "application/vnd.github+json"},
        )
    except requests.RequestException as exc:
        raise UpstreamFetchError("unreachable", f"cannot reach api.github.com: {exc}") from None
    if response.status_code != 200:
        raise UpstreamFetchError(
            "http_status", f"the commits API answered HTTP {response.status_code}"
        )
    try:
        commit = response.json()[0]["sha"]
    except (ValueError, LookupError, TypeError):
        raise UpstreamFetchError("invalid", "the commits API reply has no commit sha") from None
    if not isinstance(commit, str) or not _COMMIT.match(commit):
        raise UpstreamFetchError("invalid", f"the commits API sent a bad sha {commit!r}")
    return commit


# --------------------------------------------------------------------------- comparison


@dataclass
class CheckResult:
    """The outcome of one check, ready to record and to print."""

    outcome: str  # 'unchanged' | 'refused' | 'fetch_failed'
    result: str  # 'equal' | 'content_equal' | 'differs' | 'fetch_failed'
    matched_load_id: int | None
    raw_sha256: str | None
    upstream_commit: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """0 equal, 2 differs (needs a human), 1 the fetch failed."""
        return {"unchanged": 0, "refused": 2}.get(self.outcome, 1)


def _classification_summary(cls: DiffClassification) -> dict[str, Any]:
    return {
        "historical_rewrites": list(cls.historical),
        "tail_changes": list(cls.tail),
        "renames": [list(pair) for pair in cls.renames],
    }


def compare_with_latest(upstream: UpstreamFile, latest: VintageData | None) -> CheckResult:
    """Compare the upstream file with the latest vintage; pure, no network and no database.

    Equal raw bytes match (``equal``). Different bytes with identical parsed content, which can
    be a formatting change only, also match (``content_equal``). Otherwise the result is
    ``differs`` with coverage, date-level and interval-level counts in ``detail``.
    """
    history: History = parse_history_csv(upstream.raw)
    raw_hash = raw_sha256(upstream.raw)
    content_hash = content_sha256(history)
    days = sorted(history)
    detail: dict[str, Any] = {
        "upstream_url": upstream.url,
        "upstream_etag": upstream.etag,
        "upstream_bytes": len(upstream.raw),
        "upstream_content_sha256": content_hash,
        "upstream_coverage_start": days[0].isoformat(),
        "upstream_coverage_end": days[-1].isoformat(),
        "upstream_n_snapshot_dates": len(history),
    }
    if latest is None:
        detail["reason"] = "no vintage is loaded, so there is nothing to compare with"
        return CheckResult("refused", "differs", None, raw_hash, upstream.commit, detail)
    load = latest.load
    detail["compared_load_id"] = load["load_id"]
    detail["latest_coverage_end"] = load["coverage_end"].isoformat()
    if load["raw_sha256"] == raw_hash:
        detail["compared"] = "raw_sha256"
        return CheckResult("unchanged", "equal", load["load_id"], raw_hash, upstream.commit, detail)
    if load["content_sha256"] == content_hash:
        detail["compared"] = "content_sha256"
        detail["note"] = "raw bytes differ but the parsed content is identical"
        return CheckResult(
            "unchanged", "content_equal", load["load_id"], raw_hash, upstream.commit, detail
        )
    own = latest.history()
    diff = diff_histories(own, history)
    classification = classify_changes(latest.intervals, collapse(history), load["coverage_end"])
    reasons = refusal_reasons(
        parent_coverage=(load["coverage_start"], load["coverage_end"]),
        new_coverage=(days[0], days[-1]),
        classification=classification,
    )
    detail.update(
        {
            "compared": "differs",
            "dates_only_in_upstream": [d.isoformat() for d in diff.only_in_b],
            "dates_only_in_latest": [d.isoformat() for d in diff.only_in_a],
            "dates_changed": [d.isoformat() for d in diff.changed],
            **_classification_summary(classification),
            "would_need_force": reasons,
            "action": "none: loading is manual (`ggt db load-sp500`); nothing was changed",
        }
    )
    return CheckResult("refused", "differs", None, raw_hash, upstream.commit, detail)


def failed_result(error: UpstreamFetchError) -> CheckResult:
    """The ``fetch_failed`` record for an unusable download."""
    return CheckResult(
        "fetch_failed",
        "fetch_failed",
        None,
        None,
        None,
        {"reason": error.reason, "error": str(error)},
    )


# --------------------------------------------------------------------------- store and runner


class CheckStore(Protocol):
    """The two database operations a check needs: read the latest vintage, append one row."""

    def latest(self, index_name: str) -> VintageData | None:
        """The newest vintage of the index, or None when nothing is loaded."""
        ...

    def record(self, index_name: str, result: CheckResult) -> int:
        """Append one ``universe.checks`` row and return its ``check_id``."""
        ...


class PgCheckStore:
    """``CheckStore`` on the real database: reads read-only, then one INSERT in its own session."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url

    def latest(self, index_name: str) -> VintageData | None:
        """Read the latest vintage in a server-enforced read-only session."""
        conn = connect(self._url, read_only=True)
        try:
            with conn.cursor() as cur:
                row = latest_load(cur, index_name)
                return None if row is None else read_vintage(cur, row["load_id"], index_name)
        finally:
            conn.close()

    def record(self, index_name: str, result: CheckResult) -> int:
        """Append one check row (the only write this module ever makes)."""
        detail = {**result.detail, "result": result.result}
        conn = connect(self._url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO universe.checks (index_name, raw_sha256, upstream_commit, "
                    "outcome, matched_load_id, detail) VALUES (%s, %s, %s, %s, %s, %s) "
                    "RETURNING check_id",
                    (
                        index_name,
                        result.raw_sha256,
                        result.upstream_commit,
                        result.outcome,
                        result.matched_load_id,
                        Json(json.loads(json.dumps(detail, default=str))),
                    ),
                )
                check_id = int(cur.fetchone()[0])
            conn.commit()
            return check_id
        finally:
            conn.close()


def recent_checks(
    index_name: str = INDEX_NAME, limit: int = 3, url: str | None = None
) -> list[dict[str, Any]]:
    """The newest ``universe.checks`` rows (newest first), read in a read-only session."""
    conn = connect(url, read_only=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT check_id, checked_at, outcome, matched_load_id, raw_sha256, "
                "upstream_commit, detail FROM universe.checks WHERE index_name = %s "
                "ORDER BY check_id DESC LIMIT %s",
                (index_name, limit),
            )
            names = [d.name for d in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
    finally:
        conn.close()


@dataclass
class CheckRun:
    """What ``check_upstream`` did: the result, the recorded row id, and any record error."""

    result: CheckResult
    check_id: int | None
    record_error: str | None = None


def check_upstream(
    store: CheckStore,
    *,
    index_name: str = INDEX_NAME,
    fetch: Callable[[], UpstreamFile] = fetch_upstream,
) -> CheckRun:
    """Fetch upstream, compare it with the latest vintage and append one ``checks`` row.

    Never raises for a bad download: that is recorded as ``fetch_failed``. If the row itself
    cannot be written, the result is still returned with ``record_error`` set, so the caller
    can say so (the live gate then simply does not see a fresh verification).
    """
    try:
        upstream = fetch()
    except UpstreamFetchError as exc:
        result = failed_result(exc)
    else:
        result = compare_with_latest(upstream, store.latest(index_name))
    try:
        check_id = store.record(index_name, result)
    except Exception as exc:  # noqa: BLE001 - report, do not hide the comparison
        return CheckRun(result, None, f"{type(exc).__name__}: could not record the check row")
    return CheckRun(result, check_id)
