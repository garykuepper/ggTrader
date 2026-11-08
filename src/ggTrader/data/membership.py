"""One door to S&P 500 membership: the database or a frozen fixture file, always chosen by name.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, sections 3.1, 3.4, 3.5 and 4.
A :class:`MembershipSource` answers "who was in the index on day d" and states which vintage it
read (:class:`MembershipVintage`: vintage number or fixture name, content checksum, coverage,
retrieval and verification instants). Two implementations:

* :class:`DbMembershipSource` reads schema ``universe`` in a server-enforced read-only session.
  "latest" is resolved to one concrete load id when the source is built, in the same snapshot
  that reads the rows, so a run cannot straddle a load.
* :class:`FixtureMembershipSource` reads a frozen file listed in
  ``data/universe/membership/MANIFEST.json`` and checks its file hash, raw hash and content hash.

Selection is explicit (:func:`membership_source`): the spec string ``db``, ``db:<load_id>`` or
``fixture:<name>``, else the environment variable ``GGT_MEMBERSHIP``, else
:class:`MembershipSourceNotSelected`. There is no default, and **no fallback in any direction**:
if the database is chosen and cannot be read, :class:`MembershipUnavailable` is raised; a fixture
is never read instead. Errors never carry a connection URL, host or password.

The pure lookups mirror ``ggTrader.data.core.index_constituents`` (``sp500_members_asof``,
``all_members_between`` and the ``load_sp500_history`` table) result for result; the equivalence
tests in ``tests/data/test_membership_equivalence.py`` prove it for vintage 1. Nothing here writes
to the database. Library callers take ``membership=`` explicitly; only when they pass ``None`` do
they reach :func:`selected_source`, which honours ``GGT_MEMBERSHIP`` and otherwise raises
:class:`MembershipSourceNotSelected` (slice M3, spec 3.2). Nothing ever guesses a source.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlparse

import pandas as pd
import psycopg2

from ggTrader.data.universe_algebra import (
    History,
    HistoryValidationError,
    Interval,
    as_date,
    content_sha256,
    parse_history_csv,
    raw_sha256,
    reconstruct,
)
from ggTrader.data.universe_db import INDEX_NAME, connect, latest_load, read_vintage, to_psycopg_url
from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.paths import REPO_ROOT

#: Environment variable that selects the source when no spec string is passed.
ENV_VAR = "GGT_MEMBERSHIP"
#: The frozen vintage-1 fixture: the default of the oracle drivers, which must reproduce the
#: reference numbers recorded against it.
ORACLE_SPEC = "fixture:sp500_v1_2026-06-02"
#: Optional reader login (spec Appendix B5); defaults to the project's own connection.
READER_URL_ENV = "GGT_MEMBERSHIP_DB_URL"
#: Where the fixtures registry lives, relative to the repository root.
MANIFEST_RELPATH = Path("data") / "universe" / "membership" / "MANIFEST.json"
#: Fixture names: letters, digits, ``_``, ``.`` and ``-``.
_FIXTURE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SPEC_HELP = "use 'db', 'db:<load_id>' or 'fixture:<name>'"


# --------------------------------------------------------------------------- errors


class MembershipError(RuntimeError):
    """Base class of every error raised by this module."""


class MembershipSourceNotSelected(MembershipError):
    """No spec string and no ``GGT_MEMBERSHIP``: the caller must choose a source."""


class MembershipSpecError(MembershipError, ValueError):
    """The spec string, or a fixture name in it, is not valid."""


class MembershipUnavailable(MembershipError):
    """The chosen source cannot be read. Never falls back to another source."""


class MembershipIntegrityError(MembershipError):
    """The data read does not match its recorded hashes or counts."""


class MembershipStaleError(MembershipError):
    """The vintage is older than the allowed age (``check_membership_age(enforce=True)``)."""


class MembershipStaleWarning(UserWarning):
    """The vintage is past a warn threshold (``check_membership_age(enforce=True)``)."""


# --------------------------------------------------------------------------- vintage record


def _utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        raise ValueError("instants must be timezone-aware")
    return instant.astimezone(timezone.utc)


@dataclass(frozen=True)
class MembershipVintage:
    """Which list a source read: the identity that goes into every result's provenance."""

    index_name: str
    kind: Literal["db", "fixture"]
    load_id: int | None
    fixture: str | None
    content_sha256: str
    raw_sha256: str
    coverage_start: date
    coverage_end: date
    retrieved_at: datetime
    verified_at: datetime

    def provenance(self, asof: date | None = None) -> dict[str, Any]:
        """The provenance fields of spec 3.5: ``membership_sha256`` and the ``membership`` block.

        ``asof`` (a calendar date, never "now" by default) fills ``age_since_verified_days``;
        without it that field is ``None``. All instants are ISO strings in UTC.
        """
        age = None if asof is None else age_since_verified(self, asof)
        return {
            "membership_sha256": self.content_sha256,
            "membership": {
                "kind": self.kind,
                "load_id": self.load_id,
                "fixture": self.fixture,
                "raw_sha256": self.raw_sha256,
                "coverage_start": self.coverage_start.isoformat(),
                "coverage_end": self.coverage_end.isoformat(),
                "retrieved_at": _utc(self.retrieved_at).isoformat(),
                "verified_at": _utc(self.verified_at).isoformat(),
                "age_since_verified_days": age,
            },
        }


def age_since_verified(vintage: MembershipVintage, asof: date) -> int:
    """Days from ``verified_at`` (UTC date) to ``asof``; negative if verified after ``asof``."""
    return (as_date(asof) - _utc(vintage.verified_at).date()).days


# --------------------------------------------------------------------------- age check (3.4)


@dataclass(frozen=True)
class MembershipAge:
    """Result of :func:`check_membership_age`.

    ``age_since_change_days`` is ``asof - coverage_end`` (information, plus the optional lag
    bound); ``age_since_verified_days`` is ``asof - verified_at``. ``verdict`` is ``ok``,
    ``warn`` or ``refuse``; ``reasons`` says which threshold fired.
    """

    asof: date
    age_since_verified_days: int
    age_since_change_days: int
    verdict: Literal["ok", "warn", "refuse"]
    reasons: tuple[str, ...]


def check_membership_age(
    vintage: MembershipVintage,
    asof: date,
    *,
    verified_warn_days: int | None = None,
    verified_refuse_days: int | None = None,
    lag_warn_days: int | None = None,
    lag_refuse_days: int | None = None,
    enforce: bool = False,
) -> MembershipAge:
    """Age of a vintage at ``asof`` against thresholds you choose (spec 3.4 and decision D8).

    *Stale* means ``age_since_verified > verified_*_days``: our copy was last proved equal to
    upstream too long ago. A vintage verified after ``asof`` has a negative age and is never
    stale, so a replay of an old backtest is not penalised. The lag bound is the separate
    ``asof - coverage_end`` limit of D8 (upstream's last change date). A threshold left ``None``
    is not checked; a warn threshold must not exceed its refuse threshold.

    With ``enforce=True`` a ``warn`` emits :class:`MembershipStaleWarning` and a ``refuse`` raises
    :class:`MembershipStaleError`; by default the result is returned and the caller decides.
    Who applies this and when (the live gate only, only for the latest ``db`` vintage on a
    wall-clock run) is slice M4's business, not this helper's.
    """
    for warn, refuse, name in (
        (verified_warn_days, verified_refuse_days, "verified"),
        (lag_warn_days, lag_refuse_days, "lag"),
    ):
        if warn is not None and refuse is not None and warn > refuse:
            raise ValueError(f"{name}_warn_days ({warn}) must not exceed {name}_refuse_days")
    day = as_date(asof)
    verified = age_since_verified(vintage, day)
    change = (day - vintage.coverage_end).days
    refuse_reasons: list[str] = []
    warn_reasons: list[str] = []
    for value, warn, refuse, what in (
        (verified, verified_warn_days, verified_refuse_days, "last verified"),
        (change, lag_warn_days, lag_refuse_days, "last change date"),
    ):
        if refuse is not None and value > refuse:
            refuse_reasons.append(f"{what} {value} days before {day} (refuse above {refuse})")
        elif warn is not None and value > warn:
            warn_reasons.append(f"{what} {value} days before {day} (warn above {warn})")
    verdict: Literal["ok", "warn", "refuse"] = (
        "refuse" if refuse_reasons else "warn" if warn_reasons else "ok"
    )
    result = MembershipAge(day, verified, change, verdict, tuple(refuse_reasons + warn_reasons))
    if enforce and verdict != "ok":
        text = f"membership {_describe(vintage)}: " + "; ".join(result.reasons)
        if verdict == "refuse":
            raise MembershipStaleError(text)
        warnings.warn(text, MembershipStaleWarning, stacklevel=2)
    return result


def _describe(vintage: MembershipVintage) -> str:
    name = f"load {vintage.load_id}" if vintage.kind == "db" else f"fixture {vintage.fixture}"
    return f"{vintage.index_name} {name} ({vintage.content_sha256[:12]})"


# --------------------------------------------------------------------------- the interface


class MembershipSource(Protocol):
    """What every membership reader offers; the day arguments follow ``sp500_members_asof``."""

    def vintage(self) -> MembershipVintage:
        """Which list this source read."""
        ...

    def history(self) -> pd.DataFrame:
        """``load_sp500_history()``'s shape: UTC ``DatetimeIndex`` named ``date``, column
        ``tickers`` holding the sorted raw tickers of each snapshot date. Do not mutate it."""
        ...

    def snapshot_dates(self) -> list[date]:
        """Every snapshot date, ascending."""
        ...

    def snapshot_asof(self, asof: Any) -> tuple[date, list[str]]:
        """The snapshot in force on ``asof`` (latest on or before it): its date and tickers."""
        ...

    def members_asof(self, asof: Any) -> list[str]:
        """Sorted raw tickers that are members on ``asof``; ValueError before the first snapshot."""
        ...

    def all_members_between(self, start: Any, end: Any) -> list[str]:
        """Sorted union of members over ``[start, end]``: the snapshot in force at ``start`` plus
        every snapshot dated in ``(start, end]``."""
        ...

    @property
    def coverage_start(self) -> date:
        """First snapshot date."""
        ...

    @property
    def coverage_end(self) -> date:
        """Last snapshot date (the file's last change date)."""
        ...

    def provenance(self, asof: date | None = None) -> dict[str, Any]:
        """The spec 3.5 provenance fields for this source (see ``MembershipVintage.provenance``)."""
        ...


def _utc_day(value: Any) -> date:
    """Calendar day of a timestamp the way ``sp500_members_asof`` reads it.

    Naive stamps are UTC, aware ones are converted to UTC. Snapshots sit at midnight UTC, so
    ``snapshot <= timestamp`` is exactly ``snapshot_date <= timestamp's UTC date``.
    """
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("a date is required, got NaT")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.date()


#: Per-process cache of history frames by content hash (spec 3.2). Frames are shared, read-only.
_HISTORY_CACHE: dict[str, pd.DataFrame] = {}


class _SnapshotSource:
    """Shared lookups over one vintage held in memory (the history is small: ~2,700 rows)."""

    def __init__(self, snapshots: History, vintage: MembershipVintage) -> None:
        self._snapshots = snapshots
        self._dates = sorted(snapshots)
        self._vintage = vintage
        self._frame: pd.DataFrame | None = None

    def vintage(self) -> MembershipVintage:
        """Which list this source read."""
        return self._vintage

    @property
    def coverage_start(self) -> date:
        """First snapshot date."""
        return self._vintage.coverage_start

    @property
    def coverage_end(self) -> date:
        """Last snapshot date."""
        return self._vintage.coverage_end

    def provenance(self, asof: date | None = None) -> dict[str, Any]:
        """The spec 3.5 provenance fields."""
        return self._vintage.provenance(asof)

    def snapshot_dates(self) -> list[date]:
        """Every snapshot date, ascending."""
        return list(self._dates)

    def history(self) -> pd.DataFrame:
        """The ``load_sp500_history()`` table, built once per process per content (do not mutate).

        Frames are cached by ``content_sha256`` (spec 3.2): equal content, equal frame, whichever
        source (fixture or database) produced it.
        """
        if self._frame is None:
            key = self._vintage.content_sha256
            frame = _HISTORY_CACHE.get(key)
            if frame is None:
                # Same construction as load_sp500_history so dtype and resolution match exactly.
                dates = pd.Series([d.isoformat() for d in self._dates])
                index = pd.DatetimeIndex(pd.to_datetime(dates), tz="UTC", name="date")
                frame = pd.DataFrame(
                    {"tickers": [list(self._snapshots[d]) for d in self._dates]}, index=index
                )
                _HISTORY_CACHE[key] = frame
            self._frame = frame
        return self._frame

    def snapshot_asof(self, asof: Any) -> tuple[date, list[str]]:
        """The snapshot in force on ``asof`` (latest dated on or before it)."""
        day = _utc_day(asof)
        position = bisect_right(self._dates, day)
        if position == 0:
            raise ValueError(f"asof={asof} predates constituent history (starts {self._dates[0]})")
        found = self._dates[position - 1]
        return found, list(self._snapshots[found])

    def members_asof(self, asof: Any) -> list[str]:
        """Sorted raw tickers that are members on ``asof``."""
        return self.snapshot_asof(asof)[1]

    def all_members_between(self, start: Any, end: Any) -> list[str]:
        """Sorted union of members over ``[start, end]``."""
        first, last = _utc_day(start), _utc_day(end)
        union = set(self.members_asof(first))
        for day in self._dates[bisect_right(self._dates, first) : bisect_right(self._dates, last)]:
            union.update(self._snapshots[day])
        return sorted(union)


# --------------------------------------------------------------------------- fixtures


@dataclass(frozen=True)
class FixtureEntry:
    """One manifest entry: a frozen vintage file and the hashes it must keep."""

    name: str
    path: str
    index_name: str
    gz_sha256: str
    raw_sha256: str
    content_sha256: str
    coverage_start: date
    coverage_end: date
    n_snapshot_dates: int
    upstream_commit: str | None
    retrieved_at: datetime
    db_load_id_on_encom: int | None
    note: str | None = None


def _entry_from_json(name: str, body: Mapping[str, Any]) -> FixtureEntry:
    try:
        coverage = body["coverage"]
        entry = FixtureEntry(
            name=name,
            path=str(body["path"]),
            index_name=str(body["index_name"]),
            gz_sha256=str(body["gz_sha256"]),
            raw_sha256=str(body["raw_sha256"]),
            content_sha256=str(body["content_sha256"]),
            coverage_start=date.fromisoformat(coverage["start"]),
            coverage_end=date.fromisoformat(coverage["end"]),
            n_snapshot_dates=int(body["n_snapshot_dates"]),
            upstream_commit=body.get("upstream_commit"),
            retrieved_at=_utc(datetime.fromisoformat(body["retrieved_at"])),
            db_load_id_on_encom=body.get("db_load_id_on_encom"),
            note=body.get("note"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MembershipIntegrityError(
            f"fixture manifest entry {name!r} is malformed ({type(exc).__name__}: {exc})"
        ) from None
    if not _FIXTURE_NAME.match(name):
        raise MembershipIntegrityError(
            f"fixture name {name!r} has characters outside [A-Za-z0-9_.-]"
        )
    for field_name in ("gz_sha256", "raw_sha256", "content_sha256"):
        if not _HEX64.match(getattr(entry, field_name)):
            raise MembershipIntegrityError(f"fixture {name!r}: {field_name} is not a sha256")
    return entry


def load_manifest(root: Path = REPO_ROOT) -> dict[str, FixtureEntry]:
    """Read ``data/universe/membership/MANIFEST.json`` under ``root`` into entries by name."""
    path = root / MANIFEST_RELPATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        fixtures = document["fixtures"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MembershipUnavailable(
            f"fixtures manifest {MANIFEST_RELPATH} cannot be read ({type(exc).__name__})"
        ) from None
    return {name: _entry_from_json(name, body) for name, body in fixtures.items()}


class FixtureMembershipSource(_SnapshotSource):
    """A frozen vintage file from the manifest, checked against all of its recorded hashes."""

    def __init__(self, name: str, *, root: Path = REPO_ROOT) -> None:
        entries = load_manifest(root)
        if name not in entries:
            raise MembershipSpecError(
                f"unknown membership fixture {name!r}; the manifest lists {sorted(entries)}"
            )
        entry = entries[name]
        self.entry = entry
        self.path = root / entry.path
        try:
            gz_bytes = self.path.read_bytes()
        except OSError as exc:
            raise MembershipUnavailable(
                f"fixture {name!r}: cannot read {entry.path} ({type(exc).__name__})"
            ) from None
        if raw_sha256(gz_bytes) != entry.gz_sha256:
            raise MembershipIntegrityError(
                f"fixture {name!r}: {entry.path} no longer matches the manifest's gz_sha256 "
                "(a frozen fixture must never change; a new vintage needs a new name)"
            )
        try:
            raw = gzip.decompress(gz_bytes)
        except (OSError, EOFError) as exc:
            raise MembershipIntegrityError(
                f"fixture {name!r}: {entry.path} is not valid gzip ({type(exc).__name__})"
            ) from None
        if raw_sha256(raw) != entry.raw_sha256:
            raise MembershipIntegrityError(f"fixture {name!r}: raw sha256 differs from manifest")
        try:
            snapshots = parse_history_csv(raw)
        except HistoryValidationError as exc:
            raise MembershipIntegrityError(f"fixture {name!r}: {exc}") from None
        if content_sha256(snapshots) != entry.content_sha256:
            raise MembershipIntegrityError(
                f"fixture {name!r}: content sha256 differs from manifest"
            )
        days = sorted(snapshots)
        if (
            len(days) != entry.n_snapshot_dates
            or days[0] != entry.coverage_start
            or days[-1] != entry.coverage_end
        ):
            raise MembershipIntegrityError(f"fixture {name!r}: coverage or date count differs")
        super().__init__(
            snapshots,
            MembershipVintage(
                index_name=entry.index_name,
                kind="fixture",
                load_id=None,
                fixture=name,
                content_sha256=entry.content_sha256,
                raw_sha256=entry.raw_sha256,
                coverage_start=days[0],
                coverage_end=days[-1],
                retrieved_at=entry.retrieved_at,
                verified_at=entry.retrieved_at,
            ),
        )


# --------------------------------------------------------------------------- database


@dataclass(frozen=True)
class StoredVintage:
    """One vintage as the store returns it: the ``loads`` row, snapshot dates, visible intervals.

    ``verified_at`` is Q4's ``GREATEST(retrieved_at, last 'unchanged' check)`` for this load.
    """

    load: Mapping[str, Any]
    dates: Mapping[date, int]
    intervals: Sequence[Interval]
    verified_at: datetime


class MembershipStore(Protocol):
    """Reads a vintage in one consistent snapshot; ``None`` means the latest load."""

    def read(self, load_id: int | None, index_name: str) -> StoredVintage:
        """Return the vintage or raise :class:`MembershipUnavailable`."""
        ...


_VERIFIED_SQL = (
    "SELECT GREATEST(l.retrieved_at, COALESCE(max(c.checked_at), l.retrieved_at)) "
    "FROM universe.loads l LEFT JOIN universe.checks c "
    "ON c.index_name = l.index_name AND c.matched_load_id = l.load_id "
    "AND c.outcome = 'unchanged' WHERE l.load_id = %s GROUP BY l.load_id, l.retrieved_at"
)


def reader_url() -> str:
    """The read login: ``GGT_MEMBERSHIP_DB_URL`` when set, else the project's own connection."""
    return to_psycopg_url(os.environ.get(READER_URL_ENV) or get_db_connection_string())


def redact(text: str, url: str | None) -> str:
    """Remove a connection URL, its user, password and host, and any ``scheme://...@`` from text."""
    secrets: set[str] = set()
    if url:
        secrets.update({url, to_psycopg_url(url)})
        try:
            parts = urlparse(url)
            for piece in (parts.password, parts.username, parts.hostname):
                if piece:
                    secrets.update({piece, unquote(piece)})
        except ValueError:
            pass
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "***")
    return re.sub(r"://[^/\s@]*@", "://***@", text)


class PgMembershipStore:
    """Reads schema ``universe`` over a read-only, repeatable-read session (spec B7).

    Every error is re-raised as :class:`MembershipUnavailable` with the connection details
    redacted and the original exception suppressed from the traceback chain.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url

    def read(self, load_id: int | None, index_name: str = INDEX_NAME) -> StoredVintage:
        """Read one vintage (latest when ``load_id`` is None) in a single snapshot."""
        url: str | None = self._url
        try:
            url = url or reader_url()
            conn = connect(url, read_only=True)
            try:
                conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
                with conn.cursor() as cur:
                    if load_id is None:
                        newest = latest_load(cur, index_name)
                        if newest is None:
                            raise LookupError(f"no vintage of index {index_name!r} is loaded")
                        load_id = newest["load_id"]
                    data = read_vintage(cur, load_id, index_name)
                    cur.execute(_VERIFIED_SQL, (load_id,))
                    verified = cur.fetchone()[0]
            finally:
                conn.close()
        except LookupError as exc:
            raise MembershipUnavailable(str(exc)) from None
        except (psycopg2.Error, OSError, ValueError) as exc:
            first_line = (str(exc).strip().splitlines() or [""])[0]
            raise MembershipUnavailable(
                "cannot read membership from the database "
                f"({type(exc).__name__}: {redact(first_line, url)})"
            ) from None
        return StoredVintage(data.load, data.dates, data.intervals, verified)


class DbMembershipSource(_SnapshotSource):
    """A vintage read from the database once, at construction, then served from memory.

    ``vintage`` is ``"latest"`` (resolved to one concrete load id here and now) or a load id.
    The rebuilt history is checked against ``loads.content_sha256`` and the stored counts;
    any difference raises :class:`MembershipIntegrityError`.
    """

    def __init__(
        self,
        vintage: int | Literal["latest"] = "latest",
        *,
        store: MembershipStore | None = None,
        index_name: str = INDEX_NAME,
    ) -> None:
        if vintage != "latest" and (
            isinstance(vintage, bool) or not isinstance(vintage, int) or vintage < 1
        ):
            raise MembershipSpecError(
                f"vintage must be 'latest' or a load id >= 1, got {vintage!r}"
            )
        stored = (store or PgMembershipStore()).read(
            None if vintage == "latest" else vintage, index_name
        )
        load = stored.load
        snapshots = reconstruct(stored.intervals, stored.dates)
        problems = []
        if load["index_name"] != index_name:
            problems.append("index_name differs")
        if content_sha256(snapshots) != load["content_sha256"]:
            problems.append("rebuilt content sha256 differs from loads.content_sha256")
        if len(stored.dates) != load["n_snapshot_dates"]:
            problems.append("snapshot date count differs from loads.n_snapshot_dates")
        if len(stored.intervals) != load["n_intervals"]:
            problems.append("interval count differs from loads.n_intervals")
        if any(len(snapshots[d]) != n for d, n in stored.dates.items()):
            problems.append("a snapshot's member count differs from snapshot_dates.n_members")
        days = sorted(stored.dates)
        if not days or (days[0], days[-1]) != (load["coverage_start"], load["coverage_end"]):
            problems.append("coverage differs from loads")
        if problems:
            raise MembershipIntegrityError(
                f"load {load['load_id']} of {index_name!r} is inconsistent: " + "; ".join(problems)
            )
        #: True when a load id was named; the live gate (``membership_gate``) skips pinned reads.
        self.pinned = vintage != "latest"
        super().__init__(
            snapshots,
            MembershipVintage(
                index_name=index_name,
                kind="db",
                load_id=int(load["load_id"]),
                fixture=None,
                content_sha256=load["content_sha256"],
                raw_sha256=load["raw_sha256"],
                coverage_start=days[0],
                coverage_end=days[-1],
                retrieved_at=_utc(load["retrieved_at"]),
                verified_at=_utc(stored.verified_at),
            ),
        )


# --------------------------------------------------------------------------- selection


def parse_spec(spec: str) -> tuple[Literal["db", "fixture"], int | str | None]:
    """``db`` -> (db, None = latest); ``db:N`` -> (db, N); ``fixture:NAME`` -> (fixture, NAME)."""
    if spec == "db":
        return "db", None
    match = re.fullmatch(r"db:(\d+)", spec)
    if match and int(match.group(1)) >= 1:
        return "db", int(match.group(1))
    match = re.fullmatch(r"fixture:([A-Za-z0-9_.-]+)", spec)
    if match:
        return "fixture", match.group(1)
    raise MembershipSpecError(f"bad membership spec {spec!r}: {_SPEC_HELP}")


def membership_source(
    spec: str | None = None,
    *,
    store: MembershipStore | None = None,
    root: Path = REPO_ROOT,
) -> MembershipSource:
    """Build the source named by ``spec``, else by ``GGT_MEMBERSHIP``; never guess one.

    ``store`` and ``root`` exist for tests (an in-memory store, a scratch fixtures tree).
    """
    chosen = spec or os.environ.get(ENV_VAR) or None
    if chosen is None:
        raise MembershipSourceNotSelected(
            f"no membership source selected: pass --membership or set {ENV_VAR}; {_SPEC_HELP}. "
            "Cited results should pin a vintage (db:<load_id> or fixture:<name>); there is "
            "deliberately no default."
        )
    kind, which = parse_spec(chosen)
    if kind == "fixture":
        return FixtureMembershipSource(str(which), root=root)
    return DbMembershipSource("latest" if which is None else int(which), store=store)


# --------------------------------------------------------------------------- implicit selection

#: Sources already built in this process, by (spec, root); see :func:`selected_source`.
_SELECTED: dict[tuple[str, str], MembershipSource] = {}


def selected_source(spec: str | None = None) -> MembershipSource:
    """The source named by ``spec`` or ``GGT_MEMBERSHIP``, built once per process.

    This is what a library function reaches when its caller passed ``membership=None``: the
    selection is still explicit (a spec string or the environment variable) and there is still no
    default, so :class:`MembershipSourceNotSelected` is raised when neither is set. Caching keeps
    a per-bar caller (``eligible_at``) from re-reading and re-hashing the file on every call, and
    pins ``db`` (latest) to one load id for the life of the process, as spec 3.1 requires.
    """
    chosen = spec or os.environ.get(ENV_VAR) or None
    if chosen is None:
        return membership_source(None)  # raises MembershipSourceNotSelected with the help text
    key = (chosen, str(REPO_ROOT))
    source = _SELECTED.get(key)
    if source is None:
        source = _SELECTED[key] = membership_source(chosen)
    return source


def resolve_source(membership: MembershipSource | None) -> MembershipSource:
    """``membership`` itself, or :func:`selected_source` when the caller passed ``None``."""
    return membership if membership is not None else selected_source()


def clear_source_cache() -> None:
    """Forget every cached source and history frame (tests, long-lived processes)."""
    _SELECTED.clear()
    _HISTORY_CACHE.clear()


# --------------------------------------------------------------------------- command lines


def add_membership_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    """Add ``--membership SPEC`` and ``--expect-membership-sha`` to a driver's parser.

    ``default=None`` makes the source come from ``GGT_MEMBERSHIP`` or fail (``ggt lab``);
    the oracle drivers pass :data:`ORACLE_SPEC` so they reproduce their recorded numbers.
    """
    shown = f" (default: {default})" if default else " (default: $GGT_MEMBERSHIP, else an error)"
    parser.add_argument(
        "--membership",
        default=default,
        metavar="SPEC",
        help="S&P 500 membership source: 'db' (latest vintage), 'db:<load_id>' or "
        f"'fixture:<name>'{shown}. Cited results should pin a vintage.",
    )
    parser.add_argument(
        "--expect-membership-sha",
        default=None,
        metavar="SHA256",
        help="refuse to run unless the chosen vintage's content sha256 equals this (spec 3.6).",
    )


def source_from_args(args: argparse.Namespace) -> MembershipSource:
    """The source chosen by ``--membership`` (else ``GGT_MEMBERSHIP``), after the pin check.

    Raises :class:`MembershipSourceNotSelected` when nothing is chosen and
    :class:`MembershipIntegrityError` when ``--expect-membership-sha`` does not match.
    """
    source = selected_source(args.membership)
    expected = getattr(args, "expect_membership_sha", None)
    if expected and source.vintage().content_sha256 != expected:
        raise MembershipIntegrityError(
            f"membership content sha256 {source.vintage().content_sha256} is not the expected "
            f"{expected} ({_describe(source.vintage())})"
        )
    return source
