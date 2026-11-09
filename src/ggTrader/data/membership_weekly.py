"""``ggt db membership weekly``: fetch, archive, cross-check, alert and PROPOSE; a human loads.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, Appendix C1 and section 5.
One run, in this order:

1. Read the latest vintage (read-only). If the database cannot be read the run stops and alerts.
2. Fetch the SPDR SPY holdings workbook (``membership_holdings``, with its 7-day staleness guard)
   and the four public sources (``membership_sources``). Every fetch is independent: a failure is
   recorded and reported, never fatal to the others.
3. Compare the latest vintage with SPY on the holdings' own as-of date. A match is recorded as a
   ``matched`` row in ``universe.checks`` (it extends the live gate's freshness clock and
   ``known_through``); a difference is a ``mismatch`` row; an unusable file is ``fetch_failed``.
4. Compare our list with the other sources. Disagreements there are *warnings only*.
5. Apply the refuse-and-alert conditions: our list differs from SPY (or SPY is unusable); a ledger
   row after what is already attested is dated in the future or on its announcement date; a
   removal with no matching addition would leave 480..520 members; a source is missing, has not
   changed for 14 days, or has changed structure.
6. On any refusal with candidate changes, write a *proposed events file* (``membership_proposal``)
   under ``data/exports/membership_proposals/`` (git-ignored) and name the commands that confirm it.
7. Record one ``checks`` row plus one ``source_snapshots`` row per source (raw bytes gzipped) in a
   single transaction, then send one message through the notifier.

It never loads a vintage: it does not write ``loads``, ``membership``, ``snapshot_dates`` or
``load_blobs``. A refusal about a side source does not stop the SPY ``matched`` row from being
recorded, because that row only says "our list equals SPY on that date", which stays true.

The network (``get``), the clock, the database (:class:`WeeklyStore`) and the notifier are all
injected, so every path is tested offline with the dry-run notifier.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from psycopg2.extras import Json

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.membership import redact
from ggTrader.data.membership_holdings import (
    HoldingsFetchError,
    HoldingsResult,
    PgHoldingsStore,
    compare_holdings,
    failed_holdings,
    fetch_holdings,
    max_holdings_age_days,
)
from ggTrader.data.membership_press import (
    PRESS_LISTING,
    PRESS_PREFIX,
    PressChange,
    PressFetch,
    fetch_press,
    latest_changes,
    parse_press_listing,
    parse_press_release,
)
from ggTrader.data.membership_proposal import (
    PROPOSAL_DIRNAME,
    Proposal,
    build_proposal,
    proposal_filename,
    render_proposal_csv,
)
from ggTrader.data.membership_sources import (
    CHANGES,
    COMPONENTS,
    CURRENT,
    FETCHED_SOURCES,
    LEDGER,
    SPY,
    Ledger,
    SourceFetch,
    SourceStructureError,
    fetch_sources,
    parse_chino_changes,
    parse_chino_components,
    parse_current_list,
    parse_ledger,
)
from ggTrader.data.universe_algebra import history_members_asof, raw_sha256
from ggTrader.data.universe_db import INDEX_NAME, VintageData, connect
from ggTrader.utils.paths import REPO_ROOT

#: A source whose own last-change date is older than this many days raises an alert (C1).
STALE_SOURCE_DAYS = 14
#: Where proposed events files go: under ``data/exports/`` (git-ignored, bind-mounted in compose).
DEFAULT_PROPOSAL_DIR = REPO_ROOT / "data" / "exports" / PROPOSAL_DIRNAME
#: Telegram refuses messages over 4096 characters; stay clear of it.
MAX_MESSAGE_CHARS = 3500
#: Rows of the proposal quoted in the message; the rest are in the file.
MAX_MESSAGE_ROWS = 12
#: File name the message suggests for ``ggt db membership proposal --out``.
PROPOSAL_OUT = "proposed_events.csv"
#: Statuses of a run, worst last.
STATUS_OK, STATUS_WARN, STATUS_REFUSE, STATUS_ERROR = "ok", "warn", "refuse", "error"

GetFn = Callable[..., Any]


class Notifier(Protocol):
    """Anything with ``send(message) -> bool``: ``TelegramNotifier`` or ``DryRunNotifier``."""

    def send(self, message: str) -> bool:
        """Deliver one HTML message; return whether it went out."""
        ...


@dataclass(frozen=True)
class Finding:
    """One thing the run found: a ``refuse`` stops the quiet heartbeat, a ``warn`` only informs."""

    severity: str  # 'refuse' | 'warn'
    code: str
    text: str


@dataclass
class SnapshotRow:
    """One ``universe.source_snapshots`` row: a source of this run, with its bytes if kept here."""

    source: str
    url: str
    status: str  # 'ok' | 'failed'
    fetched_at: datetime
    last_change: date | None
    raw: bytes | None
    detail: dict[str, Any]
    raw_sha256: str | None = None
    size_bytes: int | None = None


class WeeklyStore(Protocol):
    """The three database operations the job needs; nothing else is ever done to the database."""

    def latest(self, index_name: str) -> VintageData | None:
        """The newest vintage of the index, or None when nothing is loaded."""
        ...

    def known_through(self, load_id: int, index_name: str) -> date | None:
        """The newest holdings as-of date any ``matched`` check of this load attested, else None."""
        ...

    def record_run(self, index_name: str, result: HoldingsResult, rows: list[SnapshotRow]) -> int:
        """Append the ``checks`` row, its SPY blob and the snapshot rows in one transaction."""
        ...


class PgWeeklyStore(PgHoldingsStore):
    """``WeeklyStore`` on the real database: read-only reads, then one transaction of INSERTs.

    The writes are INSERTs into ``universe.checks``, ``universe.check_blobs`` and
    ``universe.source_snapshots`` only (a test with a recording connection proves it).
    """

    def known_through(self, load_id: int, index_name: str) -> date | None:
        """Newest holdings as-of date among this load's ``matched`` checks (read-only)."""
        conn = connect(self._url, read_only=True)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT max((detail->>'holdings_as_of')::date) FROM universe.checks "
                    "WHERE index_name = %s AND matched_load_id = %s AND outcome = 'matched'",
                    (index_name, load_id),
                )
                return cur.fetchone()[0]
        finally:
            conn.close()

    def record_run(self, index_name: str, result: HoldingsResult, rows: list[SnapshotRow]) -> int:
        """Insert the check row (+ SPY blob) and every snapshot row, or nothing at all."""
        conn = connect(self._url)
        try:
            with conn.cursor() as cur:
                check_id = self.insert(cur, index_name, result)
                for row in rows:
                    cur.execute(
                        "INSERT INTO universe.source_snapshots (check_id, source, url, status, "
                        "fetched_at, last_change, raw_sha256, size_bytes, raw_gzip, detail) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            check_id,
                            row.source,
                            row.url,
                            row.status,
                            row.fetched_at,
                            row.last_change,
                            row.raw_sha256,
                            row.size_bytes,
                            None if row.raw is None else gzip.compress(row.raw, mtime=0),
                            Json(json.loads(json.dumps(row.detail, default=str))),
                        ),
                    )
            conn.commit()
            return check_id
        finally:
            conn.close()


class ProposalNotFound(LookupError):
    """The check has no stored proposal (it proposed nothing, or the id is wrong)."""


def decode_stored_proposal(
    check_id: int, raw_gzip: bytes | None, expected_sha256: str | None
) -> bytes:
    """The proposal bytes of a stored ``source_snapshots`` row, checked against its hash."""
    if raw_gzip is None:
        raise ProposalNotFound(f"check {check_id} has a proposal row but it holds no bytes")
    data = gzip.decompress(bytes(raw_gzip))
    if expected_sha256 is not None and raw_sha256(data) != expected_sha256:
        raise ValueError(f"the stored proposal of check {check_id} does not match its sha256")
    return data


def read_stored_proposal(check_id: int, url: str | None = None) -> bytes:
    """The proposed events file the weekly job stored for ``check_id`` (read-only session).

    The file written to ``data/exports/membership_proposals/`` can vanish with a container; the
    bytes in ``universe.source_snapshots`` (source ``proposal``) cannot.
    """
    conn = connect(url, read_only=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT raw_gzip, raw_sha256 FROM universe.source_snapshots "
                "WHERE check_id = %s AND source = 'proposal'",
                (check_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise ProposalNotFound(f"check {check_id} stored no proposal")
    return decode_stored_proposal(check_id, row[0], row[1])


@dataclass
class WeeklyRun:
    """Everything one run decided, for the report, the message and the tests."""

    status: str
    findings: list[Finding] = field(default_factory=list)
    spy: HoldingsResult | None = None
    load_id: int | None = None
    check_id: int | None = None
    record_error: str | None = None
    sources: dict[str, SnapshotRow] = field(default_factory=dict)
    proposal: Proposal | None = None
    press: list[PressChange] = field(default_factory=list)
    proposal_path: Path | None = None
    proposal_sha256: str | None = None
    message: str = ""
    delivered: bool | None = None
    floor: date | None = None

    @property
    def exit_code(self) -> int:
        """0 clean or warnings only, 2 a refusal or a proposal (a human acts), 1 the job broke."""
        return {STATUS_OK: 0, STATUS_WARN: 0, STATUS_REFUSE: 2}.get(self.status, 1)


# --------------------------------------------------------------------------- helpers


def _norm_set(tickers: Any) -> set[str]:
    return {normalize_yf_ticker(t) for t in tickers}


def _diff_text(ours: set[str], theirs: set[str], label: str) -> str:
    only_ours, only_theirs = sorted(ours - theirs), sorted(theirs - ours)
    return f"{label} differs from our list: only in ours {only_ours}, only in {label} {only_theirs}"


def _ledger_pending(ledger: Ledger, floor: date) -> list:
    return [row for row in ledger.rows if row.effective > floor]


def _source_row(
    source: SourceFetch, last_change: date | None, detail: dict[str, Any]
) -> SnapshotRow:
    return SnapshotRow(
        source=source.name,
        url=source.url,
        status="ok" if source.ok else "failed",
        fetched_at=source.retrieved_at,
        last_change=last_change,
        raw=source.raw,
        raw_sha256=None if source.raw is None else raw_sha256(source.raw),
        size_bytes=None if source.raw is None else len(source.raw),
        detail={
            **source.detail,
            **detail,
            "http_status": source.http_status,
            "etag": source.etag,
            "reason": source.reason,
            "error": source.error,
        },
    )


# --------------------------------------------------------------------------- the run


def run_weekly(
    store: WeeklyStore,
    notifier: Notifier,
    *,
    get: GetFn | None = None,
    now: Callable[[], datetime] | None = None,
    proposal_dir: Path | None = None,
    index_name: str = INDEX_NAME,
) -> WeeklyRun:
    """Run the job once and return what it decided. Never loads a vintage; sends one message.

    ``get`` is ``requests.get``-compatible (all five sources go through it), ``now`` the clock.
    Raises nothing for a bad source or a failed record; a database that cannot be read ends the
    run with status ``error`` and an alert.
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    today = clock().astimezone(timezone.utc).date()
    run = WeeklyRun(STATUS_OK)
    try:
        limit = max_holdings_age_days()
        latest = store.latest(index_name)
    except Exception as exc:  # noqa: BLE001 - alert on anything that stops the comparison
        first = (str(exc).strip().splitlines() or [""])[0]
        run.status = STATUS_ERROR
        run.findings.append(
            Finding(
                "refuse",
                "job_error",
                f"cannot start: {type(exc).__name__}: {redact(first, None)[:200]}",
            )
        )
        run.message = compose_message(run, today)
        _deliver(run, notifier)
        return run
    if latest is None:
        run.findings.append(Finding("refuse", "no_vintage", "no vintage of the index is loaded"))

    # --- fetch everything; each source on its own
    try:
        spy_file = fetch_holdings(get=get, now=clock)
        spy = compare_holdings(spy_file, latest, today=today, max_age_days=limit)
    except HoldingsFetchError as exc:
        spy = failed_holdings(exc)
    run.spy = spy
    fetched = fetch_sources(get=get, now=clock)
    press_fetch = fetch_press(get=get, now=clock)

    # --- parse
    parsed: dict[str, Any] = {}
    structure: dict[str, SourceStructureError] = {}
    parsers = {
        LEDGER: parse_ledger,
        CURRENT: parse_current_list,
        COMPONENTS: parse_chino_components,
        CHANGES: parse_chino_changes,
    }
    for name in FETCHED_SOURCES:
        source = fetched[name]
        if not source.ok or source.raw is None:
            continue
        try:
            parsed[name] = parsers[name](source.raw)
        except SourceStructureError as exc:
            structure[name] = exc
        except Exception as exc:  # noqa: BLE001 - a parser bug must read as "cannot parse"
            structure[name] = SourceStructureError("internal", f"{type(exc).__name__}: {exc}")
    components = parsed.get(COMPONENTS)

    # --- the sources' own last-change dates
    last_change = {name: fetched[name].last_change for name in FETCHED_SOURCES}
    if components is not None:
        last_change[COMPONENTS] = components.day  # its own newest daily row
    ledger: Ledger | None = parsed.get(LEDGER)
    for name in FETCHED_SOURCES:
        source = fetched[name]
        extra: dict[str, Any] = {}
        if name in structure:
            extra = {"structure_error": structure[name].code, "problems": structure[name].problems}
        elif name == LEDGER and ledger is not None:
            extra = {
                "n_rows": len(ledger.rows),
                "newest_effective": ledger.newest_effective.isoformat(),
                "revid": ledger.revid,
                "warnings": ledger.warnings,
            }
        elif name == CURRENT and CURRENT in parsed:
            extra = {"n_tickers": len(parsed[CURRENT].tickers), "revid": parsed[CURRENT].revid}
        elif name == COMPONENTS and components is not None:
            extra = {"newest_snapshot": components.day.isoformat(), "n_rows": components.n_rows}
        elif name == CHANGES and CHANGES in parsed:
            extra = {"n_rows": len(parsed[CHANGES]), "newest": parsed[CHANGES][-1].day.isoformat()}
        run.sources[name] = _source_row(source, last_change[name], extra)
    run.sources[SPY] = SnapshotRow(
        source=SPY,
        url=spy.detail.get("source_url", ""),
        status="failed" if spy.outcome == "fetch_failed" else "ok",
        fetched_at=clock(),
        last_change=spy.holdings_as_of,
        raw=None,  # the workbook is in check_blobs
        raw_sha256=spy.raw_sha256,
        size_bytes=None if spy.raw is None else len(spy.raw),
        detail={"stored_in": "check_blobs", "outcome": spy.outcome, **_spy_summary(spy)},
    )

    _press_sources(run, press_fetch)

    # --- findings
    _spy_findings(run, spy)
    for name in FETCHED_SOURCES:
        _source_findings(run, fetched[name], structure.get(name), last_change[name], today)
    if ledger is not None and ledger.warnings:
        run.findings.append(
            Finding(
                "warn",
                "ledger_old_rows",
                f"{len(ledger.warnings)} older ledger rows did not parse "
                f"(e.g. {ledger.warnings[0]})",
            )
        )
    if latest is not None:
        _compare_and_propose(
            run,
            latest,
            store,
            spy,
            parsed,
            ledger,
            today,
            clock,
            proposal_dir or DEFAULT_PROPOSAL_DIR,
        )

    run.status = (
        STATUS_REFUSE
        if any(f.severity == "refuse" for f in run.findings)
        else STATUS_WARN
        if run.findings
        else STATUS_OK
    )
    spy.detail["weekly"] = _weekly_detail(run, today)
    try:
        run.check_id = store.record_run(index_name, spy, list(run.sources.values()))
    except Exception as exc:  # noqa: BLE001 - report, do not hide the comparison
        run.record_error = f"{type(exc).__name__}: could not record the check row"
        run.status = STATUS_ERROR
        run.findings.append(Finding("refuse", "record_failed", run.record_error))
    run.message = compose_message(run, today)
    _deliver(run, notifier)
    return run


def _spy_summary(spy: HoldingsResult) -> dict[str, Any]:
    keys = ("holdings_as_of", "n_holdings", "only_in_list", "only_in_holdings", "reason", "error")
    return {k: spy.detail[k] for k in keys if k in spy.detail}


def _spy_findings(run: WeeklyRun, spy: HoldingsResult) -> None:
    if spy.outcome == "fetch_failed":
        run.findings.append(
            Finding(
                "refuse",
                "spy_failed",
                f"SPY holdings unusable ({spy.detail.get('reason')}): {spy.detail.get('error')}",
            )
        )
    elif spy.outcome == "mismatch":
        run.findings.append(
            Finding(
                "refuse",
                "spy_mismatch",
                f"our list differs from the SPY holdings as of {spy.holdings_as_of}: "
                f"only in our list {spy.detail.get('only_in_list')}, "
                f"only in the holdings {spy.detail.get('only_in_holdings')}",
            )
        )


def _source_findings(
    run: WeeklyRun,
    source: SourceFetch,
    error: SourceStructureError | None,
    changed: date | None,
    today: date,
) -> None:
    name = source.name
    if not source.ok:
        run.findings.append(
            Finding(
                "refuse", "source_missing", f"{name}: not fetched ({source.reason}: {source.error})"
            )
        )
        return
    if error is not None:
        run.findings.append(
            Finding(
                "refuse",
                "source_structure",
                f"{name}: the structure changed ({error.code}): {str(error)[:300]}",
            )
        )
        return
    if changed is None:
        run.findings.append(
            Finding(
                "refuse",
                "source_age_unknown",
                f"{name}: its own last-change date could not be read "
                f"({source.detail.get('last_change_error', 'no date')})",
            )
        )
    elif (today - changed).days > STALE_SOURCE_DAYS:
        run.findings.append(
            Finding(
                "refuse",
                "source_stale",
                f"{name}: last changed {changed}, {(today - changed).days} days ago "
                f"(limit {STALE_SOURCE_DAYS})",
            )
        )


def _press_sources(run: WeeklyRun, fetched: PressFetch) -> None:
    """Record the S&P DJI press releases and parse their S&P 500 tables; warnings only.

    This source is official but irregular (no release for weeks is normal), so it never refuses,
    never counts as stale and never stops the SPY ``matched`` row from being recorded.
    """
    listing = fetched.listing
    newest: date | None = None
    extra: dict[str, Any] = {}
    if listing.ok and listing.raw is not None:
        try:
            releases = parse_press_listing(listing.raw)
            newest = releases[0].published if releases else None
            extra = {"n_sp500_releases": len(releases), "n_fetched": len(fetched.releases)}
        except SourceStructureError as exc:
            extra = {"structure_error": exc.code, "problems": exc.problems}
            run.findings.append(
                Finding("warn", "press_structure", f"{listing.name}: {exc.code}: {str(exc)[:200]}")
            )
    else:
        run.findings.append(
            Finding(
                "warn",
                "press_unavailable",
                f"{listing.name}: not fetched ({listing.reason}: {listing.error})",
            )
        )
    run.sources[listing.name] = _source_row(listing, newest, extra)
    for release, source in fetched.releases:
        detail: dict[str, Any] = {"title": release.title, "published": release.published}
        if source.ok and source.raw is not None:
            try:
                changes = parse_press_release(source.raw, release)
                run.press.extend(changes)
                detail["n_sp500_rows"] = len(changes)
            except SourceStructureError as exc:
                detail.update({"structure_error": exc.code, "problems": exc.problems})
                run.findings.append(
                    Finding(
                        "warn",
                        "press_structure",
                        f"{source.name}: {exc.code}: {str(exc)[:200]}",
                    )
                )
        else:
            run.findings.append(
                Finding(
                    "warn",
                    "press_unavailable",
                    f"{source.name}: not fetched ({source.reason}: {source.error})",
                )
            )
        run.sources[source.name] = _source_row(source, release.published, detail)


def _press_vs_list(run: WeeklyRun, ours: set[str], floor: date, today: date) -> None:
    """Warn when an announced S&P 500 change that is already effective is not in our list."""
    changes = latest_changes(run.press)
    boundary = min(floor, today)
    by_ticker: dict[str, list[PressChange]] = {}
    for c in changes:
        by_ticker.setdefault(normalize_yf_ticker(c.ticker), []).append(c)
    for change in changes:
        if change.effective > boundary:
            continue  # not yet attested by our list or SPY, or not yet effective
        norm = normalize_yf_ticker(change.ticker)
        later = [
            c
            for c in by_ticker[norm]
            if c.action != change.action and change.effective <= c.effective <= boundary
        ]
        if later:
            continue  # superseded by a later announced change of the same ticker
        if change.action == "add" and norm not in ours:
            what = f"S&P DJI announced {change.ticker} added effective {change.effective}"
        elif change.action == "remove" and norm in ours:
            what = f"S&P DJI announced {change.ticker} removed effective {change.effective}"
        else:
            continue
        run.findings.append(
            Finding("warn", "press_change_not_in_list", f"{what}, but our list disagrees")
        )


def _compare_and_propose(
    run: WeeklyRun,
    latest: VintageData,
    store: WeeklyStore,
    spy: HoldingsResult,
    parsed: Mapping[str, Any],
    ledger: Ledger | None,
    today: date,
    clock: Callable[[], datetime],
    proposal_dir: Path,
) -> None:
    load = latest.load
    run.load_id = int(load["load_id"])
    history = latest.history()
    last_day = max(history)
    ours = _norm_set(history[last_day])
    coverage_end: date = load["coverage_end"]
    try:
        prior_through = store.known_through(run.load_id, load.get("index_name", INDEX_NAME))
    except Exception:  # noqa: BLE001 - fall back to the vintage's own coverage end
        prior_through = None
    prior = max(coverage_end, prior_through) if prior_through else coverage_end
    floor = max(prior, spy.holdings_as_of) if spy.outcome == "matched" else prior
    run.floor = floor

    # warnings only: the other sources against our list
    if CURRENT in parsed and set(parsed[CURRENT].tickers) != ours:
        run.findings.append(
            Finding(
                "warn",
                "current_list_differs",
                _diff_text(ours, set(parsed[CURRENT].tickers), "the Wikipedia list"),
            )
        )
    components = parsed.get(COMPONENTS)
    if components is not None and set(components.tickers) != ours:
        run.findings.append(
            Finding(
                "warn",
                "chinobing_differs",
                _diff_text(ours, set(components.tickers), f"chinobing ({components.day})"),
            )
        )

    _press_vs_list(run, ours, floor, today)

    # refusals about the ledger's newest rows (those after what is already attested)
    pending = _ledger_pending(ledger, floor) if ledger is not None else []
    for row in pending:
        what = f"{row.effective} +{row.added or '-'} -{row.removed or '-'}"
        if row.effective > today:
            run.findings.append(
                Finding("refuse", "ledger_future", f"ledger row {what} is dated in the future")
            )
        elif row.announcement == row.effective:
            run.findings.append(
                Finding(
                    "refuse",
                    "ledger_announcement_date",
                    f"ledger row {what} is dated on its announcement date, not necessarily its "
                    "effective date",
                )
            )

    # candidate changes: the set difference with SPY, else the ledger's pending rows
    if spy.outcome == "mismatch":
        adds = list(spy.detail.get("only_in_holdings", []))
        removes = list(spy.detail.get("only_in_list", []))
        snapshot_day = date.fromisoformat(
            spy.detail.get("compared_snapshot_date", last_day.isoformat())
        )
        spy_as_of = spy.holdings_as_of
    else:
        adds = sorted({normalize_yf_ticker(r.added) for r in pending if r.added} - ours)
        removes = sorted({normalize_yf_ticker(r.removed) for r in pending if r.removed} & ours)
        snapshot_day, spy_as_of = last_day, None
    refusing = any(f.severity == "refuse" for f in run.findings)
    if not (adds or removes) or not refusing:
        return
    our_raw = {normalize_yf_ticker(t): t for t in history_members_asof(history, snapshot_day)}
    chino = parsed.get(CHANGES, [])
    run.proposal = build_proposal(
        adds=adds,
        removes=removes,
        our_raw=our_raw,
        ledger=ledger.rows if ledger is not None else [],
        chino=chino if isinstance(chino, list) else [],
        floor=floor,
        today=today,
        our_count=len(history_members_asof(history, snapshot_day)),
        spy_as_of=spy_as_of,
        press=run.press,
    )
    for problem in run.proposal.count_problems:
        run.findings.append(Finding("refuse", "count_out_of_range", problem))
    _write_proposal(run, proposal_dir, clock)


def _write_proposal(run: WeeklyRun, directory: Path, clock: Callable[[], datetime]) -> None:
    """Write the proposed file (never overwriting) and keep its bytes for the database."""
    assert run.proposal is not None
    data = render_proposal_csv(run.proposal)
    path = directory / proposal_filename(clock())
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    except OSError as exc:
        run.findings.append(
            Finding("refuse", "proposal_not_written", f"could not write {path.name}: {exc}")
        )
        run.proposal_path = None
    else:
        run.proposal_path = path
    run.proposal_sha256 = hashlib.sha256(data).hexdigest()
    run.sources["proposal"] = SnapshotRow(
        source="proposal",
        url=f"file:{path.name}",
        status="ok",
        fetched_at=clock(),
        last_change=None,
        raw=data,
        raw_sha256=run.proposal_sha256,
        size_bytes=len(data),
        detail={
            "rows": len(run.proposal.rows),
            "needs_confirmation": run.proposal.n_needing_confirmation,
            "written_to": str(path),
        },
    )


def _weekly_detail(run: WeeklyRun, today: date) -> dict[str, Any]:
    return {
        "job": "weekly",
        "status": run.status,
        "run_date": today.isoformat(),
        "attested_floor": run.floor.isoformat() if run.floor else None,
        "findings": [
            {"severity": f.severity, "code": f.code, "text": f.text} for f in run.findings
        ],
        "sources": {
            name: {
                "status": row.status,
                "last_change": row.last_change.isoformat() if row.last_change else None,
                "size_bytes": row.size_bytes,
                "raw_sha256": row.raw_sha256,
            }
            for name, row in run.sources.items()
        },
        "proposal": None
        if run.proposal is None
        else {
            "file": None if run.proposal_path is None else run.proposal_path.name,
            "sha256": run.proposal_sha256,
            "rows": len(run.proposal.rows),
            "needs_confirmation": run.proposal.n_needing_confirmation,
        },
        "never_loads": "this job writes checks and source_snapshots rows only",
    }


# --------------------------------------------------------------------------- messages


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _source_line(row: SnapshotRow) -> str:
    changed = row.last_change.isoformat() if row.last_change else "unknown"
    return f"{row.source} {'ok' if row.status == 'ok' else 'FAILED'} (changed {changed})"


def compose_message(run: WeeklyRun, today: date) -> str:
    """The one Telegram message of the run (HTML): a heartbeat, a note or an action request."""
    esc = html.escape
    spy = run.spy
    names = [
        r
        for n, r in run.sources.items()
        if n != "proposal" and (n == PRESS_LISTING or not n.startswith(PRESS_PREFIX))
    ]
    if run.status == STATUS_ERROR and spy is None:
        return "<b>S&amp;P 500 list check: JOB ERROR</b>\n" + "\n".join(
            esc(f.text) for f in run.findings
        )
    if run.status == STATUS_OK and spy is not None:
        return (
            f"S&amp;P 500 list check {today}: OK. Load {run.load_id} equals the SPY holdings as of "
            f"{spy.holdings_as_of} ({spy.detail.get('n_list')} names). "
            + "; ".join(esc(_source_line(r)) for r in names)
            + f". Check #{run.check_id}."
        )
    head = {
        STATUS_WARN: "S&amp;P 500 list check: note",
        STATUS_REFUSE: "S&amp;P 500 list check: ACTION NEEDED",
        STATUS_ERROR: "S&amp;P 500 list check: JOB ERROR",
    }[run.status]
    lines = [f"<b>{head}</b> ({today})"]
    if spy is not None:
        lines.append(
            f"Load {run.load_id} vs SPY holdings as of {spy.holdings_as_of}: {spy.outcome.upper()}"
            + (f". Check #{run.check_id} recorded." if run.check_id else ". NOT recorded.")
        )
    refusals = [f for f in run.findings if f.severity == "refuse"]
    warnings = [f for f in run.findings if f.severity == "warn"]
    if refusals:
        lines.append("<b>Refused / alerts</b>")
        lines += [f"- {esc(f.text)}" for f in refusals]
    if warnings:
        lines.append("<b>Warnings</b>")
        lines += [f"- {esc(f.text)}" for f in warnings]
    if run.proposal is not None and run.proposal.rows:
        lines.append(_proposal_text(run))
    lines.append("Sources: " + "; ".join(esc(_source_line(r)) for r in names))
    text = "\n".join(lines)
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[: MAX_MESSAGE_CHARS - 40] + "\n... (cut; see the log and the file)"
    return text


def _proposal_text(run: WeeklyRun) -> str:
    esc = html.escape
    assert run.proposal is not None
    path = run.proposal_path
    if run.check_id is not None:
        # The file on disk may vanish with a container; the stored bytes do not.
        where = PROPOSAL_OUT
        fetch = f"<code>ggt db membership proposal --id {run.check_id} --out {where}</code>"
    else:
        where = _relative(path) if path else "(file not written)"
        fetch = f"<code>{esc(where)}</code>"
    lines = [
        f"<b>Proposed events file</b> ({len(run.proposal.rows)} rows, "
        f"{run.proposal.n_needing_confirmation} need confirmation; nothing was loaded): {fetch}"
    ]
    for row in run.proposal.rows[:MAX_MESSAGE_ROWS]:
        when = row.effective.isoformat() if row.effective else "date unknown"
        flag = " CONFIRM" if row.needs_confirmation else ""
        name = f"{row.ticker}->{row.new_ticker}" if row.action == "rename" else row.ticker
        lines.append(f"  {row.action} {esc(name)} {when} [{row.date_source}]{flag}")
    if len(run.proposal.rows) > MAX_MESSAGE_ROWS:
        lines.append(f"  ... {len(run.proposal.rows) - MAX_MESSAGE_ROWS} more in the file")
    target = esc(where)
    lines += [
        "To check: confirm each effective date in the S&amp;P DJI press release; fill "
        "evidence_url (and evidence_sha256 with the saved page in --evidence-dir) and "
        "announcement_date; set needs_confirmation to N; delete rows that are wrong.",
        "Then:",
        f"<code>ggt db load-sp500 --from-events {target} [--evidence-dir DIR]</code> (dry run)",
        f"<code>ggt db load-sp500 --from-events {target} [--evidence-dir DIR] --commit</code>",
    ]
    return "\n".join(lines)


def _deliver(run: WeeklyRun, notifier: Notifier) -> None:
    try:
        run.delivered = bool(notifier.send(run.message))
    except Exception:  # noqa: BLE001 - a dead channel must not hide the recorded result
        run.delivered = False


_TAG = re.compile(r"<[^>]+>")


def plain_text(message: str) -> str:
    """The message without HTML tags and entities, for the terminal."""
    return html.unescape(_TAG.sub("", message))
