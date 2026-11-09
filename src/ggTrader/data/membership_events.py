"""Curated S&P 500 vintages: apply an events file to the parent vintage and prepare the load.

The upstream feed is dead, so membership since 2026-08-19 is written from evidence (spec
docs/superpowers/specs/2026-11-07-membership-database-design.md, Appendix C, C1 and C4). A human
confirms every effective date against the S&P DJI press release, records it in an *events file*
and runs ``ggt db load-sp500 --from-events FILE``. This module turns that file into the next
vintage: it parses and checks the events, applies them to the parent vintage's last snapshot at
their effective dates, renders the full ``date,tickers`` CSV (the raw bytes of the new vintage,
same format as the other vintages) and collects the evidence files to store with it.

Pure logic first (:func:`parse_events_csv`, :func:`plan_events`, :func:`render_history_csv`,
:func:`collect_evidence`), no database. :func:`prepare_events_load` reads the parent vintage and
builds the :class:`~ggTrader.data.universe_db.LoadRequest`; the write is the loader's.

Events file columns, in order: ``effective_date, action, ticker, announcement_date,
evidence_url, evidence_sha256, note, new_ticker``. ``action`` is ``add``, ``remove`` or
``rename``. A rename row names the old symbol in ``ticker`` and the new one in ``new_ticker``;
membership still moves by an ordinary ``remove`` plus ``add`` pair (as in every earlier
vintage), and the rename row only records company continuity in ``universe.symbol_changes``.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from psycopg2.extensions import connection as PgConnection

from ggTrader.data.universe_algebra import (
    MAX_MEMBERS,
    MIN_MEMBERS,
    History,
    content_sha256,
    history_members_asof,
    raw_sha256,
)
from ggTrader.data.universe_db import (
    CURATED_SOURCE,
    INDEX_NAME,
    EvidenceFile,
    LoadRefused,
    LoadRequest,
    SymbolChange,
    latest_load,
    read_vintage,
)

#: Header an events file must carry, in this order.
EVENT_COLUMNS = (
    "effective_date",
    "action",
    "ticker",
    "announcement_date",
    "evidence_url",
    "evidence_sha256",
    "note",
    "new_ticker",
)
#: Two trailing columns a *proposed* events file (``ggt db membership weekly``) adds. A file may
#: carry them; any row still marked ``needs_confirmation`` = ``Y`` is refused, so a proposal
#: cannot be loaded until a human has confirmed each date and set the flag to ``N`` (or blanked it).
PROPOSAL_COLUMNS = ("date_source", "needs_confirmation")
ACTIONS = ("add", "remove", "rename")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class EventsFileError(ValueError):
    """The events file is malformed; ``problems`` lists every defect found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Event:
    """One line of an events file."""

    effective_date: date
    action: str
    ticker: str
    announcement_date: date
    evidence_url: str
    evidence_sha256: str
    note: str
    new_ticker: str
    line: int


@dataclass(frozen=True)
class DayChange:
    """What one effective date does: the names added and removed, and the resulting count."""

    day: date
    adds: tuple[str, ...]
    removes: tuple[str, ...]
    members: int

    @property
    def net(self) -> int:
        """Members gained minus members lost on this date."""
        return len(self.adds) - len(self.removes)


@dataclass
class EventsPlan:
    """The generated history, the per-date changes and everything that refuses the load."""

    history: History
    parent_last_day: date | None = None
    parent_members: int = 0
    days: list[DayChange] = field(default_factory=list)
    symbol_changes: list[SymbolChange] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


@dataclass
class PreparedEventsLoad:
    """Everything ``--from-events`` needs: the plan, the loader request and what stops it."""

    plan: EventsPlan
    request: LoadRequest
    parent_load_id: int
    problems: list[str]
    warnings: list[str]


def _parse_date(text: str, what: str, line: int, problems: list[str]) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        problems.append(f"line {line}: {what} {text!r} is not an ISO date")
        return None


def parse_events_csv(raw: bytes) -> list[Event]:
    """Parse and check an events file; raise :class:`EventsFileError` listing every defect."""
    problems: list[str] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EventsFileError([f"not valid UTF-8: {exc}"]) from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    header = tuple(rows[0]) if rows else ()
    if header not in (EVENT_COLUMNS, EVENT_COLUMNS + PROPOSAL_COLUMNS):
        raise EventsFileError(
            [
                f"header must be {','.join(EVENT_COLUMNS)} "
                f"(optionally followed by {','.join(PROPOSAL_COLUMNS)}), got {rows[:1]}"
            ]
        )
    events: list[Event] = []
    seen: set[tuple[date, str, str]] = set()
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            problems.append(f"line {line}: expected {len(header)} columns, got {len(row)}")
            continue
        eff_text, action, ticker, ann_text, url, sha, note, new_ticker = row[: len(EVENT_COLUMNS)]
        if len(header) > len(EVENT_COLUMNS) and row[-1].strip().upper() not in ("", "N", "NO"):
            problems.append(
                f"line {line}: still marked needs_confirmation={row[-1]!r}; confirm the date "
                "against the S&P DJI press release, then set it to N"
            )
        effective = _parse_date(eff_text, "effective_date", line, problems)
        announced = _parse_date(ann_text, "announcement_date", line, problems)
        if action not in ACTIONS:
            problems.append(f"line {line}: action {action!r} is not one of {ACTIONS}")
        for name, symbol in (("ticker", ticker), ("new_ticker", new_ticker)):
            if symbol != symbol.strip() or "," in symbol or '"' in symbol:
                problems.append(f"line {line}: {name} {symbol!r} is padded or has a comma/quote")
        if not ticker:
            problems.append(f"line {line}: ticker is empty")
        if action == "rename":
            if not new_ticker or new_ticker == ticker:
                problems.append(f"line {line}: a rename needs a new_ticker different from ticker")
        elif new_ticker:
            problems.append(f"line {line}: new_ticker is only for rename rows")
        if not url.strip():
            problems.append(f"line {line}: evidence_url is empty (every event needs its evidence)")
        if sha and not _SHA_RE.match(sha):
            problems.append(f"line {line}: evidence_sha256 is not 64 lowercase hex characters")
        if effective is not None and announced is not None and announced > effective:
            problems.append(f"line {line}: announced {announced} after it took effect {effective}")
        if effective is None or announced is None:
            continue
        key = (effective, action, ticker)
        if key in seen:
            problems.append(f"line {line}: duplicate event {action} {ticker} on {effective}")
        seen.add(key)
        events.append(Event(effective, action, ticker, announced, url, sha, note, new_ticker, line))
    if problems:
        raise EventsFileError(problems)
    if not events:
        raise EventsFileError(["the events file has no events"])
    return events


def plan_events(parent: History, events: list[Event], *, today: date) -> EventsPlan:
    """Apply the events to the parent's last snapshot and return the generated history.

    Every effective date of an add/remove event becomes a snapshot row (carry-forward, as in the
    file format), so a date where only one name changes still has its own row. Problems found:
    an effective date not after the parent's last snapshot (only tail changes are curated) or in
    the future; a removal that is not a present member; an addition that is already one; a name
    added and removed on one date; a member count outside ``MIN_MEMBERS``..``MAX_MEMBERS``; no
    add/remove at all, or a result equal to the parent; a rename whose old name was not a member
    before and whose new name is not one on its date.
    """
    plan = EventsPlan(history=dict(parent))
    if not parent:
        plan.problems.append("the parent vintage has no snapshots")
        return plan
    last = max(parent)
    state = set(parent[last])
    plan.parent_last_day, plan.parent_members = last, len(state)
    moves = [e for e in events if e.action in ("add", "remove")]
    if not moves:
        plan.problems.append("the events add or remove nothing (renames alone change no member)")
    by_day: dict[date, list[Event]] = {}
    for event in moves:
        by_day.setdefault(event.effective_date, []).append(event)
    for day in sorted(by_day):
        if day <= last:
            plan.problems.append(
                f"{day}: not after the parent's last snapshot {last}; only tail changes are curated"
            )
            continue
        if day > today:
            plan.problems.append(f"{day}: is after today ({today}); it has not taken effect")
            continue
        adds = sorted(e.ticker for e in by_day[day] if e.action == "add")
        removes = sorted(e.ticker for e in by_day[day] if e.action == "remove")
        for ticker in sorted(set(adds) & set(removes)):
            plan.problems.append(f"{day}: {ticker} is both added and removed on one date")
        for ticker in removes:
            if ticker not in state:
                plan.problems.append(f"{day}: removal of {ticker} has no matching present member")
        for ticker in adds:
            if ticker in state:
                plan.problems.append(f"{day}: addition of {ticker}, already a member")
        state = (state - set(removes)) | set(adds)
        if not MIN_MEMBERS <= len(state) <= MAX_MEMBERS:
            plan.problems.append(
                f"{day}: {len(state)} members is outside {MIN_MEMBERS}..{MAX_MEMBERS}"
            )
        plan.history[day] = tuple(sorted(state))
        plan.days.append(DayChange(day, tuple(adds), tuple(removes), len(state)))
    if moves and not plan.problems and content_sha256(plan.history) == content_sha256(parent):
        plan.problems.append("the generated history equals the parent's content")
    for event in (e for e in events if e.action == "rename"):
        old, new = event.ticker, event.new_ticker
        day, before_day = event.effective_date, event.effective_date - timedelta(days=1)
        now_members = history_members_asof(plan.history, day)
        before = history_members_asof(plan.history, before_day)
        if old not in before or new in before:
            plan.problems.append(
                f"{day}: rename {old}->{new}: {old} was not a member (or {new} already was) "
                f"before {day}"
            )
        elif old in now_members or new not in now_members:
            plan.problems.append(
                f"{day}: rename {old}->{new}: the list does not hold {new} (and not {old}) on "
                f"{day}; add the remove/add rows"
            )
        sha = f" sha256:{event.evidence_sha256}" if event.evidence_sha256 else ""
        plan.symbol_changes.append(
            SymbolChange(old, new, day, f"{event.evidence_url}{sha}".strip())
        )
    return plan


def render_history_csv(history: History) -> bytes:
    """The ``date,tickers`` file for a history: header, one row per date, sorted raw tickers.

    Same layout as the upstream file the earlier vintages came from: LF line ends and the ticker
    string quoted (it contains commas).
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["date", "tickers"])
    for day in sorted(history):
        writer.writerow([day.isoformat(), ",".join(sorted(history[day]))])
    return buffer.getvalue().encode("utf-8")


def collect_evidence(
    events_raw: bytes, events_name: str, events: list[Event], directory: Path | None
) -> tuple[list[EvidenceFile], list[str], list[str]]:
    """The files to store with the vintage, plus ``(files, problems, warnings)``.

    The events file itself, and for every distinct ``evidence_sha256`` the saved page with that
    hash found directly in ``directory``. A hash with no such file is a problem (the evidence
    would not be reproducible); a row with no hash at all is only a warning (secondary
    evidence).
    """
    files = [EvidenceFile(events_name, "events_file", events_raw)]
    problems: list[str] = []
    warnings: list[str] = []
    wanted: dict[str, str] = {}
    for event in events:
        if not event.evidence_sha256:
            warnings.append(
                f"line {event.line}: {event.action} {event.ticker} has no evidence_sha256 "
                "(secondary evidence, nothing saved)"
            )
        elif wanted.setdefault(event.evidence_sha256, event.evidence_url) != event.evidence_url:
            problems.append(f"line {event.line}: sha256 {event.evidence_sha256[:12]} has two urls")
    found: dict[str, Path] = {}
    if directory is not None:
        if not directory.is_dir():
            problems.append(f"--evidence-dir {directory} is not a directory")
        else:
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    found.setdefault(raw_sha256(path.read_bytes()), path)
    for sha, url in sorted(wanted.items(), key=lambda item: item[1]):
        path = found.get(sha)
        if path is None:
            where = "no --evidence-dir given" if directory is None else f"not in {directory}"
            problems.append(f"saved page with sha256 {sha[:12]}... for {url} not found ({where})")
            continue
        files.append(EvidenceFile(path.name, "evidence", path.read_bytes(), url))
    unused = sorted(p.name for s, p in found.items() if s not in wanted)
    if unused:
        warnings.append(f"files in --evidence-dir that no event cites are not stored: {unused}")
    return files, problems, warnings


def prepare_events_load(
    conn: PgConnection,
    *,
    events_path: Path,
    evidence_dir: Path | None,
    today: date | None = None,
    parent: int | None = None,
    retrieved_at: datetime | None = None,
    commit: bool = False,
    loader_git_sha: str = "unknown",
    note: str | None = None,
    source_url: str | None = None,
    expect_sha256: str | None = None,
    index_name: str = INDEX_NAME,
) -> PreparedEventsLoad:
    """Read the parent vintage and build the curated load; nothing is written here.

    The parent is the latest vintage; ``parent`` pins which one that must be (compare and set, as
    in every load: the loader checks it again under its lock). Raises :class:`EventsFileError`
    for a malformed file and :class:`LoadRefused` for a missing or stale parent; defects of the
    content are returned in ``problems`` so the caller can print them with the dry-run table.
    """
    events_raw = events_path.read_bytes()
    events = parse_events_csv(events_raw)
    conn.rollback()
    try:
        with conn.cursor() as cur:
            latest = latest_load(cur, index_name)
            if latest is None:
                raise LoadRefused(["nothing is loaded, so there is no parent to build on"])
            if parent is not None and parent != latest["load_id"]:
                raise LoadRefused(
                    [
                        f"the latest vintage is {latest['load_id']}, "
                        f"but the run was pinned to {parent}"
                    ]
                )
            parent_history = read_vintage(cur, latest["load_id"], index_name).history()
    finally:
        conn.rollback()
    plan = plan_events(parent_history, events, today=today or datetime.now(timezone.utc).date())
    files, evidence_problems, warnings = collect_evidence(
        events_raw, events_path.name, events, evidence_dir
    )
    problems = [*plan.problems, *evidence_problems]
    raw = render_history_csv(plan.history)
    if expect_sha256 is not None and expect_sha256 != raw_sha256(raw):
        problems.append(
            f"generated csv sha256 {raw_sha256(raw)} != --expect-sha256 {expect_sha256}"
        )
    urls = sorted({e.evidence_url for e in events})
    request = LoadRequest(
        raw=raw,
        source=CURATED_SOURCE,
        source_url=source_url or f"events:{events_path.name} " + " ".join(urls),
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        note=note
        or (
            f"curated from events file {events_path.name} (sha256 {raw_sha256(events_raw)}); "
            f"{sum(len(d.adds) for d in plan.days)} additions, "
            f"{sum(len(d.removes) for d in plan.days)} removals on {len(plan.days)} dates; "
            f"parent load {latest['load_id']}"
        ),
        parent=latest["load_id"],
        commit=commit,
        loader_git_sha=loader_git_sha,
        index_name=index_name,
        evidence=tuple(files),
        symbol_changes=tuple(plan.symbol_changes),
    )
    return PreparedEventsLoad(plan, request, latest["load_id"], problems, warnings)
