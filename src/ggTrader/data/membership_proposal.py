"""A proposed events file for a human to confirm and load: derive, render, never load.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, Appendix C1. When the SPDR
SPY holdings differ from our list, or a source shows a change we do not have, the weekly job writes
the set difference as candidate ``add`` / ``remove`` (and possible ``rename``) rows in the
events-file format of ``ggt db load-sp500 --from-events`` (``membership_events``), with two extra
columns:

* ``date_source``: where the effective date came from (``ledger``, ``ledger+chinobing``,
  ``chinobing`` or ``none``);
* ``needs_confirmation``: ``Y`` unless two independent sources give the same date, the date is not
  in the future and the ledger row does not look like it carries the announcement date.

``evidence_url`` and ``evidence_sha256`` are always blank (the human takes them from the S&P DJI
press release) and the loader refuses a row with a blank URL or a ``Y`` flag, so a proposal cannot
be loaded by accident. Nothing here touches the database.

Dates: the Wikipedia ledger carries effective dates (except that a row can carry its announcement
date instead); chinobing's dates are the days its daily snapshot *observed* the change, one day
before to two days after the effective date. Neither is trusted alone.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.membership_events import EVENT_COLUMNS, PROPOSAL_COLUMNS
from ggTrader.data.membership_sources import ChinoChange, LedgerRow
from ggTrader.data.universe_algebra import MAX_MEMBERS, MIN_MEMBERS

#: Columns of a proposed events file: the loader's eight, then the two annotation columns.
FILE_COLUMNS = EVENT_COLUMNS + PROPOSAL_COLUMNS
#: A ledger row up to this many days before the floor still counts as a candidate for a ticker
#: that is in the set difference (the row may carry its announcement date).
LEDGER_SLACK_DAYS = 14
#: A chinobing row up to this many days before the floor still counts (it observes a change up
#: to one day before the effective date).
CHINO_SLACK_DAYS = 3
#: Date sources, as written in the file.
SOURCE_LEDGER, SOURCE_BOTH, SOURCE_CHINO, SOURCE_NONE = (
    "ledger",
    "ledger+chinobing",
    "chinobing",
    "none",
)
_ACTION_ORDER = {"remove": 0, "add": 1, "rename": 2}
#: Directory name under ``data/exports`` (git-ignored) where proposals are written.
PROPOSAL_DIRNAME = "membership_proposals"


@dataclass
class ProposalRow:
    """One candidate line of the proposed file."""

    action: str
    ticker: str
    new_ticker: str = ""
    effective: date | None = None
    announcement: date | None = None
    date_source: str = SOURCE_NONE
    needs_confirmation: bool = True
    note: str = ""
    ledger_date: date | None = None
    chino_date: date | None = None

    def as_cells(self) -> list[str]:
        """The CSV cells in :data:`FILE_COLUMNS` order; evidence stays blank for the human."""
        return [
            self.effective.isoformat() if self.effective else "",
            self.action,
            self.ticker,
            self.announcement.isoformat() if self.announcement else "",
            "",  # evidence_url
            "",  # evidence_sha256
            self.note,
            self.new_ticker,
            self.date_source,
            "Y" if self.needs_confirmation else "N",
        ]


@dataclass
class Proposal:
    """The candidate rows and anything about them that should make the job refuse."""

    rows: list[ProposalRow] = field(default_factory=list)
    count_problems: list[str] = field(default_factory=list)

    @property
    def n_needing_confirmation(self) -> int:
        """Rows whose flag is ``Y``."""
        return sum(1 for r in self.rows if r.needs_confirmation)


def _norm(ticker: str) -> str:
    return normalize_yf_ticker(ticker)


def raw_spelling(normalised: str, *seen: str) -> str:
    """The project's raw spelling of a ticker: a spelling a source used, else ``-`` -> ``.``."""
    for candidate in seen:
        if candidate and _norm(candidate) == normalised:
            return candidate.strip()
    return normalised.replace("-", ".")


def _choose_date(
    ledger: LedgerRow | None, chino_day: date | None, today: date
) -> tuple[date | None, str, bool, list[str]]:
    """``(effective, date_source, needs_confirmation, notes)`` from the two sources' dates."""
    notes: list[str] = []
    led = ledger.effective if ledger else None
    if led:
        notes.append(f"ledger {led}")
    if chino_day:
        notes.append(f"chinobing {chino_day} (observation date)")
    suspect = False
    if ledger is not None and led is not None:
        if led > today:
            suspect = True
            notes.append("ledger date is in the future (not yet effective)")
        if ledger.announcement == led:
            suspect = True
            notes.append(
                "ledger date equals its announcement date (it may not be the effective date)"
            )
    if led and not suspect:
        if chino_day is None:
            return led, SOURCE_LEDGER, True, [*notes, "no second source dates it"]
        if chino_day != led:
            notes.append("sources disagree on the date")
        return led, SOURCE_BOTH, chino_day != led, notes
    if led and suspect:
        if chino_day and led <= today:
            return chino_day, SOURCE_CHINO, True, notes
        return led, SOURCE_LEDGER, True, notes
    if chino_day:
        return chino_day, SOURCE_CHINO, True, [*notes, "no ledger row dates it"]
    return None, SOURCE_NONE, True, ["no source dates this change; fill in the effective date"]


def build_proposal(
    *,
    adds: Sequence[str],
    removes: Sequence[str],
    our_raw: Mapping[str, str],
    ledger: Sequence[LedgerRow],
    chino: Sequence[ChinoChange],
    floor: date,
    today: date,
    our_count: int,
    spy_as_of: date | None = None,
) -> Proposal:
    """Candidate rows for ``adds`` / ``removes`` (normalised tickers); pure.

    ``our_raw`` maps a normalised ticker to our spelling of it (used for removals);
    ``floor`` is the date through which our list is already attested (candidate dates come from
    ledger and chinobing rows after it, with a little slack because a row may carry an announcement
    date or an observation date). ``our_count`` is the size of our list before the changes: dated
    rows are applied in order, and a running count outside 480..520 (typically a removal with no
    matching addition) is reported in ``count_problems``.
    """
    ledger_cut = floor - timedelta(days=LEDGER_SLACK_DAYS)
    chino_cut = floor - timedelta(days=CHINO_SLACK_DAYS)
    led_add: dict[str, LedgerRow] = {}
    led_rem: dict[str, LedgerRow] = {}
    for row in sorted(ledger, key=lambda r: r.effective):  # later rows overwrite earlier ones
        if row.effective <= ledger_cut:
            continue
        if row.added:
            led_add[_norm(row.added)] = row
        if row.removed:
            led_rem[_norm(row.removed)] = row
    ch_add: dict[str, date] = {}
    ch_rem: dict[str, date] = {}
    spelling: dict[str, str] = {}
    for change in sorted(chino, key=lambda c: c.day):
        if change.day <= chino_cut:
            continue
        for t in change.added:
            ch_add[_norm(t)] = change.day
            spelling[_norm(t)] = t
        for t in change.removed:
            ch_rem[_norm(t)] = change.day
            spelling[_norm(t)] = t
    proposal = Proposal()
    by_key: dict[tuple[str, str], ProposalRow] = {}
    for action, tickers, led, ch in (
        ("remove", removes, led_rem, ch_rem),
        ("add", adds, led_add, ch_add),
    ):
        for norm in sorted(tickers):
            lrow = led.get(norm)
            effective, source, needs, notes = _choose_date(lrow, ch.get(norm), today)
            ticker = (
                our_raw.get(norm, raw_spelling(norm, spelling.get(norm, "")))
                if action == "remove"
                else raw_spelling(norm, spelling.get(norm, ""), lrow.added if lrow else "")
            )
            if spy_as_of is not None:
                notes.append(
                    f"SPY holdings as of {spy_as_of} "
                    + ("show it" if action == "add" else "do not show it")
                )
            row = ProposalRow(
                action=action,
                ticker=ticker,
                effective=effective,
                announcement=lrow.announcement if lrow else None,
                date_source=source,
                needs_confirmation=needs,
                note="; ".join(notes),
                ledger_date=lrow.effective if lrow else None,
                chino_date=ch.get(norm),
            )
            by_key[(action, norm)] = row
            proposal.rows.append(row)
    _pair_renames(proposal, by_key, led_add, led_rem, chino, chino_cut)
    _check_counts(proposal, our_count)
    proposal.rows.sort(key=lambda r: (r.effective or date.max, _ACTION_ORDER[r.action], r.ticker))
    return proposal


def _pair_renames(
    proposal: Proposal,
    by_key: dict[tuple[str, str], ProposalRow],
    led_add: Mapping[str, LedgerRow],
    led_rem: Mapping[str, LedgerRow],
    chino: Sequence[ChinoChange],
    chino_cut: date,
) -> None:
    """Add a ``rename`` row where a removal and an addition are unexplained by the ledger.

    The ledger leaves out ticker changes ("a name change is not a change to the index"), while
    chinobing shows them as an add plus a remove on one day. If exactly one removed and one added
    name in a chinobing row are in our set difference and neither is in the ledger, they are
    proposed as a rename; the human decides whether it is one company.
    """
    for change in sorted(chino, key=lambda c: c.day):
        if change.day <= chino_cut:
            continue
        rem = [
            t for t in change.removed if ("remove", _norm(t)) in by_key and _norm(t) not in led_rem
        ]
        add = [t for t in change.added if ("add", _norm(t)) in by_key and _norm(t) not in led_add]
        if len(rem) != 1 or len(add) != 1:
            continue
        old_row, new_row = by_key[("remove", _norm(rem[0]))], by_key[("add", _norm(add[0]))]
        text = (
            f"possible ticker change {old_row.ticker} -> {new_row.ticker}: only chinobing shows "
            "both names on one day and the ledger omits ticker changes; confirm it is the same "
            "company in the press release, else delete this row"
        )
        proposal.rows.append(
            ProposalRow(
                action="rename",
                ticker=old_row.ticker,
                new_ticker=new_row.ticker,
                effective=new_row.effective or old_row.effective or change.day,
                date_source=SOURCE_CHINO,
                needs_confirmation=True,
                note=text,
                chino_date=change.day,
            )
        )


def _check_counts(proposal: Proposal, our_count: int) -> None:
    """Apply the dated add/remove rows in date order; flag any running count outside 480..520."""
    steps: dict[date, list[ProposalRow]] = {}
    for row in proposal.rows:
        if row.action != "rename":
            steps.setdefault(row.effective or date.max, []).append(row)
    count = our_count
    for day in sorted(steps):
        group = steps[day]
        n_add = sum(1 for r in group if r.action == "add")
        n_rem = sum(1 for r in group if r.action == "remove")
        count += n_add - n_rem
        if n_rem > n_add:
            names = ", ".join(r.ticker for r in group if r.action == "remove")
            when = "on an unknown date" if day == date.max else f"on {day}"
            for r in group:
                if r.action == "remove":
                    r.note = (r.note + "; " if r.note else "") + "no matching addition on this date"
            if not MIN_MEMBERS <= count <= MAX_MEMBERS:
                proposal.count_problems.append(
                    f"removal of {names} {when} has no matching addition and leaves {count} "
                    f"members, outside {MIN_MEMBERS}..{MAX_MEMBERS}"
                )
        elif not MIN_MEMBERS <= count <= MAX_MEMBERS:
            proposal.count_problems.append(
                f"the proposed changes on {day} leave {count} members, outside "
                f"{MIN_MEMBERS}..{MAX_MEMBERS}"
            )


def render_proposal_csv(proposal: Proposal) -> bytes:
    """The proposed events file as UTF-8 CSV bytes (LF line ends; header = :data:`FILE_COLUMNS`)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(FILE_COLUMNS)
    for row in proposal.rows:
        writer.writerow(row.as_cells())
    return buffer.getvalue().encode("utf-8")


def proposal_filename(now: datetime) -> str:
    """``proposed_events_YYYYMMDDTHHMMSSZ.csv`` for an aware instant (the name sorts by time)."""
    if now.tzinfo is None:
        raise ValueError("the proposal timestamp must be timezone-aware")
    return f"proposed_events_{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
