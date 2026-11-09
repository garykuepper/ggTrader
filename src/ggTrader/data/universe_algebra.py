"""Pure interval algebra for versioned index membership: parse, collapse, reconstruct, diff, hash.

No database and no pandas here. The loader (``universe_db``) and the tests share these functions,
so what is written to the database and what is read back go through one definition
(spec docs/superpowers/specs/2026-11-07-membership-database-design.md, sections 1, 2 and 7).

Vocabulary. A *history* maps each snapshot date to the sorted tickers of that day's snapshot. An
*interval* ``[valid_from, valid_to)`` says a symbol was in every snapshot from ``valid_from``
(inclusive) up to the first snapshot where it is absent (``valid_to``, exclusive; ``None`` means
still a member at the end of coverage, the file's carry-forward rule). Membership on any calendar
day d is the latest snapshot dated on or before d, which equals ``valid_from <= d < valid_to``.
"""

from __future__ import annotations

import csv
import hashlib
import io
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

#: Header every constituent file must carry.
CSV_HEADER = ["date", "tickers"]
#: Members per snapshot the loader accepts (today's file ranges 487-507).
MIN_MEMBERS = 480
MAX_MEMBERS = 520
#: Version of ``normalize_yf_ticker`` (index_constituents.py); stored with every load.
NORMALIZER_VERSION = 1
#: ``--force`` is needed above these counts (spec section 2); symbols, not rows.
MAX_HISTORICAL_REWRITES = 5
MAX_TAIL_CHANGES = 25
MAX_RENAMES = 3

#: One snapshot date -> its sorted raw tickers.
History = dict[date, tuple[str, ...]]


class HistoryValidationError(ValueError):
    """The constituent file broke a rule the loader always enforces."""


@dataclass(frozen=True, order=True)
class Interval:
    """One membership run of a raw symbol: in every snapshot of ``[valid_from, valid_to)``."""

    symbol_raw: str
    valid_from: date
    valid_to: date | None

    def contains(self, day: date) -> bool:
        """True when ``day`` falls inside the half-open interval."""
        return self.valid_from <= day and (self.valid_to is None or day < self.valid_to)


@dataclass(frozen=True)
class DiffClassification:
    """Interval changes between a parent vintage and a candidate, split the way section 2 does."""

    historical: tuple[str, ...]
    tail: tuple[str, ...]
    renames: tuple[tuple[str, str], ...]


@dataclass
class HistoryDiff:
    """Snapshot-level differences between two histories (the Q3b and per-date report)."""

    only_in_a: list[date] = field(default_factory=list)
    only_in_b: list[date] = field(default_factory=list)
    changed: dict[date, tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=dict)


def as_date(value: object) -> date:
    """Return ``value`` if it is a plain calendar date, else raise.

    The database session zone is America/Los_Angeles, so a timezone-aware timestamp compared with
    a DATE column can shift a day (the project's day-shift bug). A ``datetime`` (and therefore a
    ``pandas.Timestamp``) is refused outright: callers pass ``datetime.date`` only.
    """
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(
            f"expected a datetime.date, got {type(value).__name__}: pass .date() explicitly "
            "(timestamps are refused so a timezone cannot shift the day)"
        )
    return value


def parse_history_csv(raw: bytes) -> History:
    """Parse and validate a ``date,tickers`` file; raise :class:`HistoryValidationError`.

    Enforced: the header, ISO dates that strictly increase, no empty, padded or duplicate ticker
    in a row, and between ``MIN_MEMBERS`` and ``MAX_MEMBERS`` tickers on every date.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoryValidationError(f"not valid UTF-8: {exc}") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != CSV_HEADER:
        raise HistoryValidationError(f"header must be {','.join(CSV_HEADER)}, got {rows[:1]}")
    history: History = {}
    previous: date | None = None
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise HistoryValidationError(f"line {number}: expected 2 columns, got {len(row)}")
        try:
            day = date.fromisoformat(row[0])
        except ValueError as exc:
            raise HistoryValidationError(f"line {number}: bad ISO date {row[0]!r}") from exc
        if previous is not None and day <= previous:
            raise HistoryValidationError(f"line {number}: date {day} does not increase")
        tickers = row[1].split(",")
        for ticker in tickers:
            if ticker == "" or ticker != ticker.strip():
                raise HistoryValidationError(f"line {number}: empty or padded ticker {ticker!r}")
        if len(set(tickers)) != len(tickers):
            raise HistoryValidationError(f"line {number}: duplicate ticker on {day}")
        if not MIN_MEMBERS <= len(tickers) <= MAX_MEMBERS:
            raise HistoryValidationError(
                f"line {number}: {len(tickers)} members on {day}, outside "
                f"[{MIN_MEMBERS}, {MAX_MEMBERS}]"
            )
        history[day] = tuple(sorted(tickers))
        previous = day
    if not history:
        raise HistoryValidationError("no snapshot rows")
    return history


def raw_sha256(raw: bytes) -> str:
    """Hex sha256 of the uncompressed input bytes."""
    return hashlib.sha256(raw).hexdigest()


def content_sha256(history: Mapping[date, Iterable[str]]) -> str:
    """Canonical content hash (spec 1.4): the identity of a vintage in provenance.

    sha256 of UTF-8 lines ``YYYY-MM-DD,<raw tickers sorted, joined by ','>\\n``, one per snapshot
    date ascending, no header. Independent of the input's quoting, line endings and ticker order.
    """
    digest = hashlib.sha256()
    for day in sorted(history):
        digest.update(f"{day.isoformat()},{','.join(sorted(history[day]))}\n".encode())
    return digest.hexdigest()


def collapse(history: Mapping[date, Iterable[str]]) -> list[Interval]:
    """Turn daily snapshots into the minimal sorted list of membership intervals."""
    opened: dict[str, date] = {}
    out: list[Interval] = []
    previous: frozenset[str] = frozenset()
    for day in sorted(history):
        current = frozenset(history[day])
        for symbol in previous - current:
            out.append(Interval(symbol, opened.pop(symbol), day))
        for symbol in current - previous:
            opened[symbol] = day
        previous = current
    out.extend(Interval(symbol, start, None) for symbol, start in opened.items())
    return sorted(out)


def reconstruct(intervals: Iterable[Interval], dates: Iterable[date]) -> History:
    """Rebuild the snapshot for every date in ``dates`` from the intervals alone."""
    ordered = sorted(set(dates))
    starts: dict[date, list[str]] = {}
    ends: dict[date, list[str]] = {}
    for interval in intervals:
        starts.setdefault(interval.valid_from, []).append(interval.symbol_raw)
        if interval.valid_to is not None:
            ends.setdefault(interval.valid_to, []).append(interval.symbol_raw)
    boundaries = sorted(set(starts) | set(ends))
    active: set[str] = set()
    out: History = {}
    index = 0
    for day in ordered:
        while index < len(boundaries) and boundaries[index] <= day:
            boundary = boundaries[index]
            active.difference_update(ends.get(boundary, ()))
            active.update(starts.get(boundary, ()))
            index += 1
        out[day] = tuple(sorted(active))
    return out


def members_on(intervals: Sequence[Interval], day: date) -> tuple[str, ...]:
    """Sorted raw symbols that are members on calendar day ``day`` (brute force, for tests)."""
    return tuple(sorted(i.symbol_raw for i in intervals if i.contains(day)))


def history_members_asof(history: Mapping[date, Iterable[str]], day: date) -> tuple[str, ...]:
    """The latest snapshot on or before ``day`` (the file semantics); empty before the first."""
    dates = sorted(history)
    position = bisect_right(dates, day)
    return tuple(sorted(history[dates[position - 1]])) if position else ()


def diff_intervals(
    a: Iterable[Interval], b: Iterable[Interval]
) -> tuple[list[Interval], list[Interval]]:
    """Intervals only in ``a`` and only in ``b`` (the Q3a set difference), each sorted."""
    set_a, set_b = set(a), set(b)
    return sorted(set_a - set_b), sorted(set_b - set_a)


def diff_histories(a: History, b: History) -> HistoryDiff:
    """Snapshot dates only on one side, and for shared dates the ones whose members differ."""
    out = HistoryDiff(
        only_in_a=sorted(set(a) - set(b)),
        only_in_b=sorted(set(b) - set(a)),
    )
    for day in sorted(set(a) & set(b)):
        if a[day] != b[day]:
            out.changed[day] = (a[day], b[day])
    return out


def _day_ranges(intervals: Sequence[Interval]) -> list[tuple[date, date | None]]:
    return sorted((i.valid_from, i.valid_to) for i in intervals)


def _first_differing_day(
    a: Sequence[tuple[date, date | None]], b: Sequence[tuple[date, date | None]]
) -> date | None:
    """Earliest calendar day covered by exactly one of two range lists, else None.

    Coverage only changes at range boundaries, so testing each boundary day decides every piece
    (the last piece is open-ended and is decided by the last boundary).
    """

    def covered(ranges: Sequence[tuple[date, date | None]], day: date) -> bool:
        return any(lo <= day and (hi is None or day < hi) for lo, hi in ranges)

    for day in sorted({p for r in (*a, *b) for p in r if p is not None}):
        if covered(a, day) != covered(b, day):
            return day
    return None


def classify_changes(
    parent: Iterable[Interval], new: Iterable[Interval], parent_coverage_end: date
) -> DiffClassification:
    """Split changed symbols into historical rewrites, tail changes and renames.

    A symbol is a *tail* change only if every calendar day on which its membership differs lies
    after ``parent_coverage_end`` (Appendix A item 7); otherwise it is a *historical* rewrite. A
    *rename* is a symbol whose intervals vanish while another symbol appears with identical dates:
    both names are reported under renames and neither counts as historical or tail.
    """
    only_parent, only_new = diff_intervals(parent, new)
    by_symbol_parent: dict[str, list[Interval]] = {}
    by_symbol_new: dict[str, list[Interval]] = {}
    for interval in only_parent:
        by_symbol_parent.setdefault(interval.symbol_raw, []).append(interval)
    for interval in only_new:
        by_symbol_new.setdefault(interval.symbol_raw, []).append(interval)

    parent_all: dict[str, list[Interval]] = {}
    for interval in parent:
        parent_all.setdefault(interval.symbol_raw, []).append(interval)
    new_all: dict[str, list[Interval]] = {}
    for interval in new:
        new_all.setdefault(interval.symbol_raw, []).append(interval)

    gone = {s for s in by_symbol_parent if s not in new_all}
    appeared = {s for s in by_symbol_new if s not in parent_all}
    renames: list[tuple[str, str]] = []
    claimed: set[str] = set()
    for old in sorted(gone):
        old_dates = [(i.valid_from, i.valid_to) for i in by_symbol_parent[old]]
        for fresh in sorted(appeared - claimed):
            if [(i.valid_from, i.valid_to) for i in by_symbol_new[fresh]] == old_dates:
                renames.append((old, fresh))
                claimed.add(fresh)
                break
    renamed = {s for pair in renames for s in pair}

    historical: list[str] = []
    tail: list[str] = []
    for symbol in sorted(set(by_symbol_parent) | set(by_symbol_new)):
        if symbol in renamed:
            continue
        first = _first_differing_day(
            _day_ranges(parent_all.get(symbol, [])), _day_ranges(new_all.get(symbol, []))
        )
        if first is None:
            continue
        (tail if first > parent_coverage_end else historical).append(symbol)
    return DiffClassification(tuple(historical), tuple(tail), tuple(renames))


def refusal_reasons(
    *,
    parent_coverage: tuple[date, date] | None,
    new_coverage: tuple[date, date],
    classification: DiffClassification | None,
) -> list[str]:
    """Why a load needs ``--force`` (spec section 2); an empty list means it may proceed."""
    if parent_coverage is None or classification is None:
        return []
    reasons: list[str] = []
    if new_coverage[0] > parent_coverage[0]:
        reasons.append(f"coverage_start moved later ({parent_coverage[0]} -> {new_coverage[0]})")
    if new_coverage[1] < parent_coverage[1]:
        reasons.append(f"coverage_end moved earlier ({parent_coverage[1]} -> {new_coverage[1]})")
    if len(classification.historical) > MAX_HISTORICAL_REWRITES:
        reasons.append(
            f"{len(classification.historical)} historical rewrites > {MAX_HISTORICAL_REWRITES}"
        )
    if len(classification.tail) > MAX_TAIL_CHANGES:
        reasons.append(f"{len(classification.tail)} tail changes > {MAX_TAIL_CHANGES}")
    if len(classification.renames) > MAX_RENAMES:
        reasons.append(f"{len(classification.renames)} renames > {MAX_RENAMES}")
    return reasons


def plan_supersession(
    visible: Iterable[tuple[int, Interval]], new: Iterable[Interval]
) -> tuple[list[int], list[Interval]]:
    """Split a new vintage against the parent's visible rows.

    Returns ``(interval_ids to supersede, intervals to insert)``: visible rows missing from the
    new set are superseded, new intervals not already visible are inserted, and an interval that
    is identical in the new vintage is left alone (lineage is per row, spec 1.1).
    """
    new_set = set(new)
    visible_list = list(visible)
    present = {interval for _, interval in visible_list}
    to_supersede = sorted(i for i, interval in visible_list if interval not in new_set)
    to_insert = sorted(new_set - present)
    return to_supersede, to_insert


def history_stats(history: Mapping[date, Iterable[str]]) -> dict[str, int | date]:
    """Counts the ``loads`` row stores about a history."""
    days = sorted(history)
    sizes = [len(tuple(history[d])) for d in days]
    symbols = {t for d in days for t in history[d]}
    return {
        "coverage_start": days[0],
        "coverage_end": days[-1],
        "n_snapshot_dates": len(days),
        "n_symbols": len(symbols),
        "min_members": min(sizes),
        "max_members": max(sizes),
    }
