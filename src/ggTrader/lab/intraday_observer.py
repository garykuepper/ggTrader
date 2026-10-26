"""Isolated file-based recorder for the prospective SPY intraday observation (no orders)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests

from ggTrader.lab.intraday_probe import (
    ENDPOINTS,
    MINUTE_NS,
    NEW_YORK,
    RequestDeadline,
    SchemaError,
    atomic_checkpoint,
    canonical_hash,
    metadata,
    request_deadline,
    timestamp_ns,
    valid_date,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_FILES = (
    "src/ggTrader/lab/intraday_observer.py",
    "scripts/intraday_observe.py",
    "src/ggTrader/lab/intraday_probe.py",
)
SCHEMA = "intraday_observer"
VERSION = 1
SYMBOL = "SPY"
BARS_URL = ENDPOINTS["bars"]  # the only URL this module can ever GET
BAR_PARAMS = {
    "symbols": SYMBOL,
    "timeframe": "1Min",
    "adjustment": "raw",
    "asof": "-",
    "feed": "sip",
    "sort": "asc",
    "limit": 10000,
}
SLOTS = 252
CHECKPOINT_SLOT = 20
LOOKAHEAD_DAYS = 470
START_EQUITY = 100_000.0
BASE_COST = 1e-4  # per side
STRESS_COST = 2e-4
TAX_RATE = 0.24
NW_LAGS = 5
BUDGET_PER_DATE = 12
MIN_GAP_SECONDS = 0.65
REQUEST_DEADLINE_SECONDS = 30
PAGE_ATTEMPTS = 3
MAX_PAGES = 5
MAX_RETRY_WAIT = 30
STAGE_CLOCKS = {  # America/New_York wall clock
    "recover": "15:15",
    "inputs": "15:20",
    "cutoff": "15:25",
    "decision_open": "15:25",
    "decision_close": "15:26",
    "archive": "16:20",
    "archive_due": "16:25",
}
STAGE_TABLE = [
    (
        "recover",
        "15:15",
        "conditional: capture the prior session archive if it has no canonical value",
    ),
    (
        "inputs",
        "15:20",
        "prior anchor from canonical archive + 09:59 close; cutoff before 15:25:00",
    ),
    ("decide", "15:25", "one immutable side, window 15:25:00 inclusive to 15:26:00 exclusive"),
    ("archive", "16:20", "regular-session snapshot, due 16:25; grace until next session 15:20"),
]
RECORD_DIRS = ("records", "claims", "raw", "budget")
SOURCE_HOURS = ("09:30", "16:00")
CLOCK_STATUS_OK = "synchronized"
LOCAL_TIMESTAMP_NOTE = (
    "Timestamps are local host evidence only; decisions are not independently witnessed."
)


class ObserverError(Exception):
    """Refusal that must leave the state untouched."""


class CalendarMismatch(ObserverError):
    """Official and offline NYSE calendars disagree."""


class BindingMismatch(ObserverError):
    """Protocol, code, runtime, path or manifest identity differs from the frozen binding."""


class ClockMismatch(ObserverError):
    """The clock is naive, moved backwards or the date does not match the registration."""


class TooEarly(ObserverError):
    """The stage cannot start yet; nothing is recorded."""


class DuplicateStage(ObserverError):
    """The session stage already has a completed record."""


class StageBusy(ObserverError):
    """Another process holds the exclusive stage claim."""


class BudgetExhausted(ObserverError):
    """The shared per-date GET attempt budget is spent."""


class NotActivated(ObserverError):
    """No activation record exists in this state directory."""


# --- time -----------------------------------------------------------------------------------


def utc_at(day: str, clock: str) -> datetime:
    """New York wall-clock time on a session date as an aware UTC instant (DST-safe)."""
    return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=NEW_YORK).astimezone(UTC)


def iso(moment: datetime) -> str:
    """UTC ISO text with microseconds retained."""
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    """Inverse of iso()."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ny_date(moment: datetime) -> str:
    """New York calendar date of an instant."""
    return moment.astimezone(NEW_YORK).date().isoformat()


def minute_before(clock: str) -> str:
    """The closing minute label for a close clock (16:00 -> 15:59, 13:00 -> 12:59)."""
    return (datetime.strptime(clock, "%H:%M") - timedelta(minutes=1)).strftime("%H:%M")


def system_clock() -> datetime:
    """Real wall clock; the only clock the CLI uses."""
    return datetime.now(UTC)


def host_clock_status() -> str:
    """Host NTP status from timedatectl: synchronized, unsynchronized or unknown."""
    try:
        done = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return {"yes": CLOCK_STATUS_OK, "no": "unsynchronized"}.get(done.stdout.strip(), "unknown")


# --- calendar -------------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _offline_rows(start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    if schedule.empty:
        return ()
    opens = schedule["market_open"].dt.tz_convert(NEW_YORK)
    closes = schedule["market_close"].dt.tz_convert(NEW_YORK)
    return tuple(
        (day.date().isoformat(), o.strftime("%H:%M"), c.strftime("%H:%M"))
        for day, o, c in zip(schedule.index, opens, closes, strict=True)
    )


def offline_sessions(start: str, end: str) -> list[dict]:
    """Offline NYSE sessions (date, open, close in New York time); imported lazily."""
    return [{"date": d, "open": o, "close": c} for d, o, c in _offline_rows(start, end)]


def offline_calendar_version() -> str:
    """Version of the offline calendar library."""
    import pandas_market_calendars as mcal

    return mcal.__version__


def session_kind(row: dict) -> str:
    """full or early_close; any other hours are not supported by this protocol."""
    if row["open"] != SOURCE_HOURS[0] or row["close"] > SOURCE_HOURS[1]:
        raise ValueError(f"unsupported_session_hours:{row['date']}")
    return "full" if row["close"] == SOURCE_HOURS[1] else "early_close"


def build_calendar(official: list[dict] | None, activation_date: str) -> dict:
    """Derive the 252 registered slots from the offline calendar and compare the official one.

    Slot 1 is the third full session strictly after the activation date. ``official=None`` skips the
    comparison (plan only) and says so; a disagreement raises and prepares nothing.
    """
    valid_date(activation_date)
    after = date.fromisoformat(activation_date)
    offline = offline_sessions(
        after.isoformat(), (after + timedelta(days=LOOKAHEAD_DAYS)).isoformat()
    )
    full = [r["date"] for r in offline if r["date"] > activation_date and session_kind(r) == "full"]
    if len(full) < 2 + SLOTS:
        raise ObserverError("calendar_too_short")
    slots = full[2 : 2 + SLOTS]
    first, last = slots[0], slots[-1]
    anchor = max(r["date"] for r in offline if r["date"] < first)
    following = min(r["date"] for r in offline if r["date"] > last)
    span = [r for r in offline if anchor <= r["date"] <= following]
    if official is None:
        agreement = "not_checked"
    else:
        wanted = {r["date"]: (r["open"], r["close"]) for r in span}
        rows = [r for r in official if anchor <= r["date"] <= following]
        seen = {r["date"]: (r["open"], r["close"]) for r in rows}
        problems = [
            f"duplicate_official_date:{d}"
            for d in sorted({r["date"] for r in rows})
            if sum(r["date"] == d for r in rows) > 1
        ]
        problems += [f"official_missing:{d}" for d in sorted(set(wanted) - set(seen))]
        problems += [f"official_extra:{d}" for d in sorted(set(seen) - set(wanted))]
        problems += [
            f"hours_differ:{d}" for d in sorted(set(seen) & set(wanted)) if seen[d] != wanted[d]
        ]
        if problems:
            raise CalendarMismatch(problems)
        agreement = "agrees_with_offline_nyse_" + offline_calendar_version().replace(".", "_")
    slot_of = {day: n for n, day in enumerate(slots, start=1)}
    sessions = [
        {
            "date": r["date"],
            "kind": session_kind(r),
            "open": r["open"],
            "close": r["close"],
            "close_minute": minute_before(r["close"]),
            "slot": slot_of.get(r["date"]),
        }
        for r in span
        if r["date"] <= last
    ]
    year_ends = [
        offline_sessions(f"{y}-12-15", f"{y}-12-31")[-1]["date"]
        for y in range(int(first[:4]), int(last[:4]) + 1)
    ]
    return {
        "source": "offline_nyse_calendar",
        "library_version": offline_calendar_version(),
        "official_agreement": agreement,
        "activation_date": activation_date,
        "first_date": first,
        "checkpoint_date": slots[CHECKPOINT_SLOT - 1],
        "last_date": last,
        "anchor_date": anchor,
        "post_window_next_session": following,
        "early_closes": [s["date"] for s in sessions if s["kind"] == "early_close"],
        "year_end_sessions": [d for d in year_ends if d <= last],
        "slot_dates_sha256": canonical_hash(slots),
        "schedule_sha256": canonical_hash(sessions),
        "sessions": sessions,
    }


# --- identity bindings ----------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Hash stored bytes exactly as read."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_binding() -> dict:
    """Hash of every source file this recorder executes."""
    files = {name: file_sha256(PROJECT_ROOT / name) for name in CODE_FILES}
    return {"files": files, "sha256": canonical_hash(files)}


def runtime_binding() -> dict:
    """Interpreter and library versions that the evidence depends on."""
    versions = {
        "python": platform.python_version(),
        "requests": requests.__version__,
        "pandas_market_calendars": offline_calendar_version(),
        **{name: package_version(name) for name in ("numpy", "pandas", "statsmodels")},
    }
    return {**versions, "sha256": canonical_hash(versions)}


def protocol_binding(path: Path, expected: str | None = None) -> dict:
    """Hash the protocol file; a different hash than expected is refused."""
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        raise ValueError("protocol_hash_mismatch")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def load_official_calendar(path: Path) -> tuple[list[dict], str]:
    """Parse an Alpaca calendar snapshot file and return validated rows plus its byte hash."""
    raw = Path(path).read_bytes()
    try:
        rows, _ = metadata("calendar", json.loads(raw))
    except (ValueError, TypeError) as error:
        raise ObserverError("official_calendar_snapshot_invalid") from error
    return rows, hashlib.sha256(raw).hexdigest()


def init_state(
    state_dir: Path,
    *,
    mode: str,
    activation_date: str,
    official_calendar_file: Path,
    protocol_file: Path,
    protocol_sha256: str,
    source_delay_seconds: int,
    now: datetime,
) -> dict:
    """Explicit offline preparation: manifest and empty directories only; nothing is active."""
    state_dir = Path(state_dir)
    if mode not in ("live", "rehearsal"):
        raise ValueError("mode_invalid")
    if state_dir.exists() and any(state_dir.iterdir()):
        raise FileExistsError("state_dir_not_empty")
    if source_delay_seconds < 0:
        raise ValueError("source_delay_invalid")
    protocol = protocol_binding(protocol_file, protocol_sha256)
    rows, snapshot_sha = load_official_calendar(official_calendar_file)
    calendar = build_calendar(rows, activation_date)
    calendar["official_snapshot_sha256"] = snapshot_sha
    calendar["official_agreement"] = "operator_supplied_snapshot_matches_offline_calendar"
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": mode,
        "status": "prepared_offline",
        "activated": False,
        "source_preflight": "not_performed",
        "source_delay_seconds": source_delay_seconds,
        "source_delay_verified": False,
        "created_utc": iso(now),
        "state_dir": str(state_dir.resolve()),
        "bindings": {"protocol": protocol, "code": code_binding(), "runtime": runtime_binding()},
        "source": {"url": BARS_URL, "parameters": BAR_PARAMS},
        "economics": {
            "start_equity": START_EQUITY,
            "base_cost_per_side": BASE_COST,
            "stress_cost_per_side": STRESS_COST,
            "tax_rate": TAX_RATE,
        },
        "evidence_note": LOCAL_TIMESTAMP_NOTE,
        "calendar": calendar,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in RECORD_DIRS:
        (state_dir / name).mkdir(exist_ok=True)
    atomic_checkpoint(state_dir / "manifest.json", manifest, initial=True)
    return manifest


def plan_calendar(activation_date: str) -> dict:
    """Offline slot summary without the full session list."""
    calendar = build_calendar(None, activation_date)
    return {k: v for k, v in calendar.items() if k != "sessions"}


def plan(activation_date: str | None = None) -> dict:
    """Describe the stages and bounds; reads only the offline calendar, never the network."""
    return {
        "network": False,
        "writes": False,
        "activated": False,
        "source_preflight": "not_performed",
        "official_calendar_agreement": "not_checked",
        "stages": [{"stage": s, "new_york_time": t, "purpose": p} for s, t, p in STAGE_TABLE],
        "request_bounds": {
            "attempts_per_new_york_date": BUDGET_PER_DATE,
            "min_seconds_between_starts": MIN_GAP_SECONDS,
            "request_deadline_seconds": REQUEST_DEADLINE_SECONDS,
            "attempts_per_page": PAGE_ATTEMPTS,
        },
        "source": {"url": BARS_URL, "parameters": BAR_PARAMS},
        "calendar": plan_calendar(activation_date) if activation_date else None,
        "notes": [LOCAL_TIMESTAMP_NOTE, "No orders are possible: the only request is a bars GET."],
    }


# --- transport ------------------------------------------------------------------------------


@dataclass
class Response:
    """One raw HTTP outcome; error is set when no status was received."""

    status: int | None
    headers: dict
    body: bytes
    error: str | None = None


class HttpTransport:
    """Fixed read-only GET of the bars endpoint with credentials from the inherited environment."""

    def __init__(self, environ: dict | None = None, session: Any = None) -> None:
        env = os.environ if environ is None else environ
        key, secret = env.get("APCA_API_KEY_ID"), env.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("credentials_missing")
        self.session = session or requests.Session()
        self.session.trust_env = False  # no proxy or netrc credentials from the environment
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def get(self, url: str, params: dict) -> Response:
        """One GET with a total wall-clock deadline; never follows redirects."""
        if url != BARS_URL:
            raise ValueError("url_not_allowed")
        reply = None
        try:
            with request_deadline(REQUEST_DEADLINE_SECONDS):
                reply = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_DEADLINE_SECONDS,
                    stream=True,
                    allow_redirects=False,
                )
                body = reply.content
            headers = {str(k).lower(): v for k, v in reply.headers.items()}
            return Response(reply.status_code, headers, body)
        except (requests.RequestException, RequestDeadline) as error:
            timed_out = isinstance(error, (requests.Timeout, RequestDeadline))
            return Response(None, {}, b"", "timeout" if timed_out else "transport_error")
        finally:
            if reply is not None:
                reply.close()


def default_price(day: str, hhmm: str) -> tuple[float, float, float, float]:
    """Deterministic synthetic OHLC that drifts upward through the session."""
    minutes = (int(hhmm[:2]) - 9) * 60 + int(hhmm[3:5]) - 30
    base = 100 + 0.01 * minutes
    return base, base + 0.05, base - 0.05, base + 0.02


class SyntheticTransport:
    """Offline stand-in for the bars endpoint: honours start/end and pagination, never networks."""

    def __init__(
        self,
        price_fn: Callable = default_price,
        closes: dict | None = None,
        script: list | None = None,
        per_page: int | None = None,
    ) -> None:
        self.price_fn = price_fn
        self.closes = closes or {}
        self.script = list(script or [])
        self.per_page = per_page
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict) -> Response:
        """Return the next scripted outcome, or synthetic bars for the requested range."""
        self.calls.append((url, dict(params)))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            if item is not None:
                return item
        start, end = (parse_iso(params[k]) for k in ("start", "end"))
        rows, moment = [], start
        while moment <= end:
            local = moment.astimezone(NEW_YORK)
            day, hhmm = local.date().isoformat(), local.strftime("%H:%M")
            if "09:30" <= hhmm <= minute_before(self.closes.get(day, SOURCE_HOURS[1])):
                o, h, low, c = self.price_fn(day, hhmm)
                rows.append(
                    {
                        "t": iso(moment).replace(".000000", ""),
                        "o": o,
                        "h": h,
                        "l": low,
                        "c": c,
                        "v": 10,
                    }
                )
            moment += timedelta(minutes=1)
        token, offset = None, int(params.get("page_token", 0))
        if self.per_page:
            token = str(offset + self.per_page) if offset + self.per_page < len(rows) else None
            rows = rows[offset : offset + self.per_page]
        body = json.dumps({"bars": {SYMBOL: rows}, "next_page_token": token}).encode()
        return Response(200, {}, body)


def parse_bar_page(payload: Any, start_ns: int, end_ns: int) -> tuple[list[dict], str | None, list]:
    """Validate one bars page strictly; any defect marks the whole page untrustworthy."""
    if not isinstance(payload, dict) or "next_page_token" not in payload:
        return [], None, ["payload_shape"]
    token = payload["next_page_token"]
    groups = payload.get("bars")
    if token is not None and (not isinstance(token, str) or not token):
        return [], None, ["page_token"]
    if not isinstance(groups, dict) or not isinstance(groups.get(SYMBOL, []), list):
        return [], None, ["payload_shape"]
    bars, defects, previous = [], [], -1
    for row in groups.get(SYMBOL, []):
        try:
            stamp = timestamp_ns(row["t"])
            o, h, low, c, v = (row[k] for k in ("o", "h", "l", "c", "v"))
        except (SchemaError, KeyError, TypeError):
            defects.append("bad_row")
            continue
        if stamp % MINUTE_NS:
            defects.append(f"misaligned:{row['t']}")
        if not start_ns <= stamp <= end_ns:
            defects.append(f"out_of_range:{row['t']}")
        if stamp <= previous:
            defects.append(f"duplicate_or_unsorted:{row['t']}")
        previous = stamp
        prices = (o, h, low, c)
        sane = all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
            for x in prices
        )
        sane = sane and low <= min(o, c) and max(o, c) <= h
        sane = sane and isinstance(v, (int, float)) and not isinstance(v, bool)
        sane = sane and math.isfinite(v) and v >= 0
        if not sane:
            defects.append(f"bad_ohlcv:{row['t']}")
        bars.append({"t": row["t"], "ns": stamp, "o": o, "h": h, "l": low, "c": c, "v": v})
    return bars, token, defects


# --- observer state -------------------------------------------------------------------------


class Observer:
    """One state directory: evidence ledger, exclusive stage claims, request budget and fetch."""

    def __init__(self, state_dir: Path, deps: Deps | None = None) -> None:
        self.dir = Path(state_dir).resolve()
        self.deps = deps or Deps()
        try:
            raw = (self.dir / "manifest.json").read_bytes()
        except OSError:
            raise ObserverError("state_not_initialized") from None
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest.get("state_dir") != str(self.dir):
            raise BindingMismatch("state_path_mismatch")
        frozen = self.manifest["bindings"]
        if (
            protocol_binding(Path(frozen["protocol"]["path"]))["sha256"]
            != frozen["protocol"]["sha256"]
        ):
            raise BindingMismatch("protocol_hash_mismatch")
        if code_binding()["sha256"] != frozen["code"]["sha256"]:
            raise BindingMismatch("code_hash_mismatch")
        if runtime_binding()["sha256"] != frozen["runtime"]["sha256"]:
            raise BindingMismatch("runtime_mismatch")
        self.bindings = {
            "protocol": frozen["protocol"]["sha256"],
            "code": frozen["code"]["sha256"],
            "runtime": frozen["runtime"]["sha256"],
            "manifest": self.manifest_sha256,
        }
        for name in self._names():
            try:
                record = json.loads((self.dir / "records" / name).read_text())
            except (ValueError, OSError):
                continue  # verify reports unreadable evidence without decoding its fields.
            if isinstance(record, dict) and record.get("bindings") != self.bindings:
                raise BindingMismatch("record_binding_mismatch")

    # clock -------------------------------------------------------------------------------

    def clock(self) -> datetime:
        """Current time from the injected clock; naive instants are refused."""
        moment = self.deps.clock()
        if moment.tzinfo is None:
            raise ClockMismatch("naive_clock")
        return moment

    # evidence ledger ---------------------------------------------------------------------

    def _names(self) -> list[str]:
        return sorted(
            n for n in os.listdir(self.dir / "records") if re.fullmatch(r"\d{6}__.*\.json", n)
        )

    def has_record(self, session: str | None, stage: str) -> bool:
        """True when a completed record exists for the session stage."""
        suffix = f"__{session or 'none'}__{stage}.json"
        return any(name.endswith(suffix) for name in self._names())

    def records(self) -> list[dict]:
        """All records in sequence order."""
        try:
            records = [json.loads((self.dir / "records" / n).read_text()) for n in self._names()]
            if any(
                not isinstance(r, dict)
                or not {
                    "seq",
                    "stage",
                    "session",
                    "kind",
                    "body",
                    "bindings",
                    "sha256",
                    "recorded_utc",
                }
                <= r.keys()
                or not isinstance(r["body"], dict)
                for r in records
            ):
                raise ValueError("record_fields_missing")
            return records
        except (ValueError, TypeError, OSError):
            raise ObserverError("ledger_record_invalid") from None

    @contextmanager
    def _ledger_lock(self) -> Iterator[None]:
        with open(self.dir / "records" / ".lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def append(
        self,
        kind: str,
        session: str | None,
        stage: str,
        body: dict,
        at: datetime,
        *,
        clock_status: str | None = None,
    ) -> dict:
        """Append one hash-chained record; duplicates and clock regressions are refused."""
        if at.tzinfo is None:
            raise ClockMismatch("naive_clock")
        with self._ledger_lock():
            names = self._names()
            if self.has_record(session, stage):
                raise DuplicateStage(f"{session}:{stage}")
            previous = json.loads((self.dir / "records" / names[-1]).read_text()) if names else None
            if previous and at < parse_iso(previous["recorded_utc"]):
                raise ClockMismatch("clock_regression")
            record = {
                "schema": "intraday_observer_record",
                "version": VERSION,
                "seq": len(names) + 1,
                "kind": kind,
                "session": session,
                "stage": stage,
                "recorded_utc": iso(at),
                "prev_sha256": previous["sha256"] if previous else None,
                "bindings": self.bindings,
                "clock_status": clock_status,
                "body": body,
            }
            record["sha256"] = canonical_hash(record)
            name = f"{record['seq']:06d}__{session or 'none'}__{stage}.json"
            atomic_checkpoint(self.dir / "records" / name, record, initial=True)
        return record

    def verify_chain(self) -> list[str]:
        """Recompute every hash link and raw-payload hash; return the problems found."""
        problems, previous = [], None
        for number, name in enumerate(self._names(), start=1):
            try:
                record = json.loads((self.dir / "records" / name).read_text())
            except (ValueError, OSError):
                problems.append(f"record_unreadable:{name}")
                continue
            if not isinstance(record, dict):
                problems.append(f"record_invalid:{name}")
                continue
            claimed = record.pop("sha256", None)
            if canonical_hash(record) != claimed:
                problems.append(f"record_hash_mismatch:{name}")
            if record.get("seq") != number or not name.startswith(f"{number:06d}__"):
                problems.append(f"sequence_gap:{name}")
            if record.get("prev_sha256") != previous:
                problems.append(f"chain_broken:{name}")
            if record.get("bindings") != self.bindings:
                problems.append(f"record_binding_mismatch:{name}")
            if not isinstance(record.get("body"), dict):
                problems.append(f"record_invalid:{name}")
            for raw_path in self._raw_references(record):
                if not re.fullmatch(r"raw/[a-f0-9]{64}\.bin", raw_path):
                    problems.append(f"raw_path_invalid:{name}")
                elif not (self.dir / raw_path).is_file():
                    problems.append(f"raw_missing:{raw_path}")
            previous = claimed
        for path in (self.dir / "raw").iterdir():
            if not re.fullmatch(r"[a-f0-9]{64}\.bin", path.name):
                continue  # Unreferenced crash temporaries are reported separately, never evidence.
            if path.stem != hashlib.sha256(path.read_bytes()).hexdigest():
                problems.append(f"raw_hash_mismatch:{path.name}")
        return problems

    @staticmethod
    def _raw_references(value: Any) -> Iterator[str]:
        """Find referenced payloads even if the corresponding file has been deleted."""
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "raw_path" and isinstance(item, str):
                    yield item
                else:
                    yield from Observer._raw_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from Observer._raw_references(item)

    def stray_raw_files(self) -> list[str]:
        """Non-evidence files left by interrupted writes; no automatic deletion."""
        return sorted(
            p.name
            for p in (self.dir / "raw").iterdir()
            if not re.fullmatch(r"[a-f0-9]{64}\.bin", p.name)
        )

    def store_raw(self, body: bytes) -> tuple[str, str]:
        """Write the exact response bytes once, keyed by their hash."""
        digest = hashlib.sha256(body).hexdigest()
        path = self.dir / "raw" / f"{digest}.bin"
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(handle.name, path)
            except FileExistsError:
                pass
            finally:
                os.unlink(handle.name)
        return digest, f"raw/{digest}.bin"

    # claims and budget -------------------------------------------------------------------

    @contextmanager
    def claim(self, session: str | None, stage: str) -> Iterator[dict | None]:
        """Exclusive claim on a session stage; yields the crashed predecessor's claim, if any."""
        path = self.dir / "claims" / f"{session or 'none'}__{stage}.claim"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StageBusy(f"{session}:{stage}") from None
            if self.has_record(session, stage):
                raise DuplicateStage(f"{session}:{stage}")
            text = os.pread(fd, 65536, 0)
            try:
                prior = json.loads(text) if text else None
            except ValueError:
                prior = {"unreadable": True}
            os.ftruncate(fd, 0)
            mine = {"pid": os.getpid(), "claimed_utc": iso(self.clock())}
            os.pwrite(fd, json.dumps(mine).encode(), 0)
            os.fsync(fd)
            yield prior
        finally:
            os.close(fd)

    def reserve_request(self, session: str | None, stage: str) -> None:
        """Count one GET attempt against the shared per-New-York-date budget and pace starts."""
        now = self.clock()
        day = ny_date(now)
        with open(self.dir / "budget" / f"{day}.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.dir / "budget" / f"{day}.json"
            data = json.loads(path.read_text()) if path.exists() else {"date": day, "attempts": []}
            if len(data["attempts"]) >= BUDGET_PER_DATE:
                raise BudgetExhausted(day)
            if data["attempts"]:
                gap = (now - parse_iso(data["attempts"][-1]["utc"])).total_seconds()
                if gap < MIN_GAP_SECONDS:
                    self.deps.sleep(MIN_GAP_SECONDS - gap)
                    now = self.clock()
            data["attempts"].append({"utc": iso(now), "session": session, "stage": stage})
            atomic_checkpoint(path, data)

    # source access -----------------------------------------------------------------------

    def fetch_bars(
        self, session: str, stage: str, start: datetime, end: datetime, cutoff: datetime
    ) -> dict:
        """GET minute bars over [start, end] inclusive, keeping every attempt's raw bytes.

        Never starts or retries an attempt at or after ``cutoff``, nor with an ``end`` newer than
        now minus the registered source delay.
        """
        delay = timedelta(seconds=self.manifest["source_delay_seconds"])
        params = {
            **BAR_PARAMS,
            "start": iso(start).replace(".000000", ""),
            "end": iso(end).replace(".000000", ""),
        }
        start_ns, end_ns = (
            int(start.timestamp()) * 1_000_000_000,
            int(end.timestamp()) * 1_000_000_000,
        )
        result = {
            "outcome": None,
            "parameters": params,
            "attempts": [],
            "bars": [],
            "pages": 0,
            "defects": [],
        }
        token, seen_tokens, seen_stamps = None, set(), set()
        for page in range(1, MAX_PAGES + 1):
            body = None
            for attempt in range(1, PAGE_ATTEMPTS + 1):
                now = self.clock()
                if now >= cutoff:
                    result["outcome"] = "cutoff_reached"
                    return result
                if end > now - delay:
                    result["outcome"] = "source_not_available"
                    return result
                try:
                    self.reserve_request(session, stage)
                except BudgetExhausted:
                    result["outcome"] = "budget_exhausted"
                    return result
                request = dict(params, **({"page_token": token} if token else {}))
                requested = self.clock()
                if requested >= cutoff:
                    result["outcome"] = "cutoff_reached"
                    return result
                try:
                    reply = self.deps.transport.get(BARS_URL, request)
                except (RequestDeadline, requests.RequestException):
                    reply = Response(None, {}, b"", "timeout")
                received = self.clock()
                digest, raw_path = self.store_raw(reply.body)
                result["attempts"].append(
                    {
                        "page": page,
                        "attempt": attempt,
                        "requested_utc": iso(requested),
                        "received_utc": iso(received),
                        "status_code": reply.status,
                        "transport_error": reply.error,
                        "body_sha256": digest,
                        "raw_path": raw_path,
                    }
                )
                if reply.status == 200:
                    body = reply.body
                    break
                retryable = reply.status is None or reply.status == 429 or reply.status >= 500
                if not retryable:
                    result["outcome"] = {401: "authentication_error", 403: "entitlement_error"}.get(
                        reply.status, "invalid_request"
                    )
                    return result
                wait = float(attempt)
                for name in ("retry-after", "x-ratelimit-reset"):
                    try:
                        value = float(reply.headers.get(name, "nan"))
                        wait = max(wait, value - received.timestamp() if "reset" in name else value)
                    except (TypeError, ValueError):
                        pass
                if wait > MAX_RETRY_WAIT:
                    result["outcome"] = "deferred"
                    return result
                if attempt == PAGE_ATTEMPTS:
                    result["outcome"] = (
                        "throttled_exhausted" if reply.status == 429 else "transient_exhausted"
                    )
                    return result
                if self.clock() + timedelta(seconds=wait) >= cutoff:
                    result["outcome"] = "cutoff_reached"
                    return result
                self.deps.sleep(wait)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                result.update(outcome="schema_error", defects=["not_json"])
                return result
            rows, next_token, defects = parse_bar_page(payload, start_ns, end_ns)
            defects += [f"duplicate_across_pages:{r['t']}" for r in rows if r["ns"] in seen_stamps]
            if defects:
                result.update(outcome="schema_error", bars=[], defects=defects)
                return result
            seen_stamps.update(r["ns"] for r in rows)
            result["bars"].extend(rows)
            result["pages"] = page
            if next_token is None:
                result["outcome"] = "success"
                return result
            if next_token in seen_tokens:
                break
            seen_tokens.add(next_token)
            token = next_token
        result["outcome"] = "incomplete_pagination"
        result["bars"] = []
        return result

    # session lookups ---------------------------------------------------------------------

    def amendments(self) -> list[dict]:
        """Owner-approved append-only amendments in recorded order."""
        return [r for r in self.records() if r["kind"] == "amendment"]

    def effective_rows(self) -> list[dict]:
        """Registered sessions with calendar amendments applied; originals stay in the manifest."""
        rows = {r["date"]: dict(r) for r in self.manifest["calendar"]["sessions"]}
        for amendment in self.amendments():
            if amendment["body"]["type"] == "session_status":
                rows[amendment["session"]] = dict(amendment["body"]["revised"])
        return [rows[r["date"]] for r in self.manifest["calendar"]["sessions"]]

    def session_row(self, day: str) -> dict:
        """The effective session row; unknown dates are refused."""
        for row in self.effective_rows():
            if row["date"] == day:
                return row
        raise ObserverError(f"unknown_session:{day}")

    def neighbours(self, day: str) -> tuple[str | None, str]:
        """Previous and next trading session; cancelled (closed) sessions are skipped."""
        rows = self.effective_rows()
        index = [r["date"] for r in rows].index(self.session_row(day)["date"])
        previous = next((r["date"] for r in reversed(rows[:index]) if r["kind"] != "closed"), None)
        following = next(
            (r["date"] for r in rows[index + 1 :] if r["kind"] != "closed"),
            self.manifest["calendar"]["post_window_next_session"],
        )
        return previous, following

    def stage_record(self, day: str, stage: str) -> dict | None:
        """One completed stage record, if it exists."""
        return self.completed_stage([r for r in self.records() if r["session"] == day], stage)

    @staticmethod
    def completed_stage(records: list[dict], stage: str) -> dict | None:
        """An interrupted durable write cannot become a completed timely decision."""
        stages = {r["stage"]: r for r in records}
        base = stages.get(stage)
        if base and base["body"].get("requires_completion"):
            return stages.get(f"{stage}.completion")
        return base

    def append_timed(
        self,
        day: str,
        stage: str,
        body: dict,
        at: datetime,
        cutoff: datetime,
        sync: str,
    ) -> dict:
        """Preserve the durable stage, then attest when its publication finished."""
        prepared = self.append(
            "stage", day, stage, {**body, "requires_completion": True}, at, clock_status=sync
        )
        finished = self.clock()
        complete = {
            **body,
            "source_record_sha256": prepared["sha256"],
            "publication_completed_utc": iso(finished),
        }
        if finished >= cutoff and (body.get("usable") or body.get("timely_valid")):
            complete.update(
                status="fault",
                usable=False,
                timely=False,
                timely_valid=False,
                observer_side=0,
                signal_side=None,
            )
            complete["faults"] = [*body["faults"], f"{stage}_publication_after_cutoff"]
        return self.append(
            "stage_completion", day, f"{stage}.completion", complete, finished, clock_status=sync
        )

    def canonical(self, day: str) -> dict | None:
        """The first valid archive capture accepted as the session's valuation."""
        return next(
            (
                r
                for r in self.records()
                if r["session"] == day
                and r["stage"].startswith("archive.")
                and not r["body"].get("requires_completion")
                and r["body"].get("canonical")
            ),
            None,
        )

    def next_capture_no(self, day: str) -> int:
        """Sequence number for the next archive capture of the session."""
        return 1 + sum(
            r["session"] == day and re.fullmatch(r"archive\.\d+", r["stage"]) is not None
            for r in self.records()
        )

    def require_active(self) -> None:
        """Capture exists only after an explicit activation record."""
        if not self.has_record(None, "activation"):
            raise NotActivated("activation_record_missing")

    def clock_status(self) -> str:
        """Host clock-sync evidence; a failing probe counts as unknown."""
        try:
            return self.deps.clock_sync()
        except Exception:
            return "unknown"

    def activate(
        self,
        owner_approval: str,
        readiness_sha256: str | None = None,
        readiness_file: Path | None = None,
    ) -> dict:
        """Write the activation record; only on the registered date and with a named approval."""
        if not owner_approval.strip():
            raise ObserverError("owner_approval_required")
        now = self.clock()
        if ny_date(now) != self.manifest["calendar"]["activation_date"]:
            raise ClockMismatch("activation_date_mismatch")
        readiness_path = None
        if self.manifest["mode"] == "live":
            if readiness_file is None:
                raise ObserverError("readiness_file_required")
            raw = Path(readiness_file).read_bytes()
            evidence = json.loads(raw)
            required = {
                "schema": "intraday_observer_readiness",
                "status": "verified",
                "manifest_sha256": self.manifest_sha256,
                "source_delay_seconds": self.manifest["source_delay_seconds"],
                "calendar_snapshot_sha256": self.manifest["calendar"]["official_snapshot_sha256"],
            }
            checks = ("sip_access", "source_delay", "calendar_source", "synthetic_rehearsal")
            if (
                not isinstance(evidence, dict)
                or any(evidence.get(k) != v for k, v in required.items())
                or not isinstance(evidence.get("checks"), dict)
                or any(evidence.get("checks", {}).get(k) is not True for k in checks)
            ):
                raise ObserverError("readiness_evidence_mismatch")
            digest = hashlib.sha256(raw).hexdigest()
            if readiness_sha256 is not None and readiness_sha256 != digest:
                raise ObserverError("readiness_hash_mismatch")
            readiness_sha256, readiness_path = self.store_raw(raw)
        body = {
            "owner_approval": owner_approval,
            "readiness_sha256": readiness_sha256,
            "raw_path": readiness_path,
            "source_preflight": "verified" if readiness_path else "synthetic_only",
            "source_delay_verified": readiness_path is not None,
        }
        return self.append("activation", None, "activation", body, now)

    # stages ------------------------------------------------------------------------------

    def capture(
        self,
        day: str,
        number: int,
        *,
        cutoff: datetime,
        grace_end: datetime,
        bootstrap: bool = False,
    ) -> dict:
        """Fetch a session's regular-session snapshot once; the first valid one is canonical."""
        row = self.session_row(day)
        stage = f"archive.{number}"
        with self.claim(day, stage) as prior:
            end = utc_at(day, row["close_minute"] + ":00")
            fetch = self.fetch_bars(day, stage, utc_at(day, "09:30:00"), end, cutoff)
            sync = self.clock_status()
            completed = self.clock()
            values, missing = extract_values(fetch["bars"], day, row)
            success = fetch["outcome"] == "success"
            faults = [] if success else [f"fetch_{fetch['outcome']}"]
            faults += [f"missing_bar:{label}" for label in missing] if success else []
            valid = success and not missing
            on_time = (
                completed < cutoff
                if bootstrap
                else completed <= utc_at(day, STAGE_CLOCKS["archive_due"] + ":00")
            )
            if valid and completed >= grace_end:
                faults.append("completed_after_grace")
            if not on_time:
                faults.append("archive_late")
            if sync != CLOCK_STATUS_OK:
                faults.append(f"clock_{sync}")
            body = {
                "capture_no": number,
                "status": "ok" if valid else "fault",
                "faults": faults,
                "valuation_valid": valid,
                "canonical": valid and completed < grace_end and self.canonical(day) is None,
                "on_time": on_time,
                "late_capture": valid and not on_time,
                "bootstrap": bootstrap,
                "close_minute": row["close_minute"],
                "grace_end_utc": iso(grace_end),
                "completed_utc": iso(completed),
                "values": values,
                "bars_returned": len(fetch["bars"]),
                "fetch": {k: v for k, v in fetch.items() if k != "bars"},
                "reconciled_claim": prior,
            }
            inputs = self.stage_record(day, "inputs")
            seen = inputs["body"].get("bar_0959") if inputs else None
            if valid and seen and values.get("close_0959") != seen["c"]:
                body["input_revision"] = {
                    "input_close_0959": seen["c"],
                    "archive_close_0959": values["close_0959"],
                }
            prepared = self.append(
                "stage",
                day,
                stage,
                {**body, "requires_completion": True},
                completed,
                clock_status=sync,
            )
            finished = self.clock()
            body["publication_completed_utc"] = iso(finished)
            body["source_record_sha256"] = prepared["sha256"]
            body["on_time"] = (
                finished < cutoff
                if bootstrap
                else finished <= utc_at(day, STAGE_CLOCKS["archive_due"] + ":00")
            )
            body["late_capture"] = valid and not body["on_time"]
            body["canonical"] = valid and finished < grace_end and self.canonical(day) is None
            if not body["on_time"] and "archive_late" not in faults:
                faults.append("archive_late")
            if valid and finished >= grace_end and "completed_after_grace" not in faults:
                faults.append("completed_after_grace")
            return self.append(
                "stage_completion", day, f"{stage}.completion", body, finished, clock_status=sync
            )

    def run_inputs(self, day: str | None = None) -> dict:
        """15:20 stage: freeze the prior anchor and today's 09:59 close, nothing later."""
        self.require_active()
        day = day or ny_date(self.clock())
        row = self.session_row(day)
        now, cutoff = self.clock(), utc_at(day, STAGE_CLOCKS["cutoff"] + ":00")
        if now < utc_at(day, STAGE_CLOCKS["inputs"] + ":00"):
            raise TooEarly(f"inputs:{day}")
        with self.claim(day, "inputs") as prior:
            body: dict = {
                "started_utc": iso(now),
                "cutoff_utc": iso(cutoff),
                "reconciled_claim": prior,
            }
            faults: list[str] = []
            if row["kind"] != "full":
                body.update(
                    status="not_applicable", reason=na_reason(row), usable=False, faults=faults
                )
                return self.append("stage", day, "inputs", body, now)
            if now >= cutoff:
                faults.append("inputs_late_not_attempted")
                body["timely"] = False
            else:
                previous, _ = self.neighbours(day)
                first_full = next(
                    r["date"] for r in self.effective_rows() if r["slot"] and r["kind"] == "full"
                )
                if day == first_full and self.canonical(previous) is None:
                    self.capture(
                        previous,
                        self.next_capture_no(previous),
                        cutoff=cutoff,
                        grace_end=cutoff,
                        bootstrap=True,
                    )
                anchor = self.canonical(previous)
                if anchor is None:
                    faults.append("prior_anchor_missing")
                else:
                    body["prior"] = {
                        "session": previous,
                        "seq": anchor["seq"],
                        "record_sha256": anchor["sha256"],
                        "close": anchor["body"]["values"]["close_final"],
                        "close_minute": anchor["body"]["close_minute"],
                        "late_capture": anchor["body"]["late_capture"],
                        "bootstrap": anchor["body"]["bootstrap"],
                    }
                    minute = utc_at(day, "09:59:00")
                    fetch = self.fetch_bars(day, "inputs", minute, minute, cutoff)
                    body["fetch"] = {k: v for k, v in fetch.items() if k != "bars"}
                    if fetch["outcome"] != "success":
                        faults.append(f"fetch_{fetch['outcome']}")
                    elif len(fetch["bars"]) != 1:
                        faults.append("missing_0959_bar")
                    else:
                        body["bar_0959"] = fetch["bars"][0]
                sync = self.clock_status()
                completed = self.clock()
                body["completed_utc"] = iso(completed)
                body["timely"] = completed < cutoff
                if not body["timely"]:
                    faults.append("inputs_completed_after_cutoff")
            if now >= cutoff:
                sync = self.clock_status()
            if sync != CLOCK_STATUS_OK:
                faults.append(f"clock_{sync}")
            body.update(faults=faults, usable=not faults, status="fault" if faults else "ok")
            return self.append_timed(day, "inputs", body, self.clock(), cutoff, sync)

    def halt_reasons(self, day: str) -> list[dict]:
        """Open halting faults before the session, minus those cleared by an approved resume."""
        reasons: list[dict] = []
        if self.verify_chain():
            reasons.append({"reason": "ledger_integrity", "origin": None})
        try:
            records = self.records()
        except ObserverError:
            return reasons
        decisions: dict[str, list[dict]] = {}
        for record in records:
            if record["stage"] == "decision":
                completed = self.completed_stage(
                    [r for r in records if r["session"] == record["session"]], "decision"
                )
                if completed:
                    decisions.setdefault(record["session"], []).append(completed)
        now = self.clock()
        canonical_days = {
            r["session"]
            for r in records
            if r["body"].get("canonical")
            and r["stage"].startswith("archive.")
            and not r["body"].get("requires_completion")
        }
        for row in self.effective_rows():
            found = decisions.get(row["date"], [])
            if len(found) > 1:
                reasons.append({"reason": "conflicting_decisions", "origin": row["date"]})
            traded = any(r["body"].get("observer_side") == 1 for r in found)
            if row["slot"] and row["date"] < day and traded and row["date"] not in canonical_days:
                grace_end = utc_at(self.neighbours(row["date"])[1], STAGE_CLOCKS["inputs"] + ":00")
                state = "unresolved_valuation" if now >= grace_end else "pending_valuation"
                reasons.append({"reason": state, "origin": row["date"]})
        checkpoint = self.checkpoint_status(records)
        if checkpoint["reached"] and checkpoint["failures"]:
            reasons.append({"reason": "checkpoint_failed", "origin": checkpoint["date"]})
        resumes = [
            a["session"]
            for a in records
            if a["kind"] == "amendment" and a["body"]["type"] == "resume"
        ]
        return [
            r
            for r in reasons
            if r["origin"] is None or not any(r["origin"] < eff <= day for eff in resumes)
        ]

    def checkpoint_status(self, records: list[dict] | None = None) -> dict:
        """Check all eligible observation slots, including stages never recorded."""
        day = self.manifest["calendar"]["checkpoint_date"]
        reached = self.clock() >= utc_at(self.neighbours(day)[1], "15:20:00")
        failures = []
        if reached:
            by_day: dict[str, list[dict]] = {}
            for record in records if records is not None else self.records():
                by_day.setdefault(record["session"], []).append(record)
            for row in self.effective_rows():
                if not row["slot"] or row["slot"] > CHECKPOINT_SLOT:
                    continue
                entries = by_day.get(row["date"], [])
                required = ("inputs", "decision") if row["kind"] == "full" else ()
                for stage in required:
                    if self.completed_stage(entries, stage) is None:
                        failures.append(
                            {
                                "session": row["date"],
                                "stage": stage,
                                "fault": f"missing_stage:{stage}",
                            }
                        )
                archives = [
                    r
                    for r in entries
                    if r["stage"].startswith("archive.")
                    and not r["body"].get("requires_completion")
                ]
                if row["kind"] != "closed" and not any(
                    r["body"].get("canonical") and r["body"].get("on_time") for r in archives
                ):
                    failures.append(
                        {
                            "session": row["date"],
                            "stage": "archive",
                            "fault": "missing_timely_valuation",
                        }
                    )
                failures.extend(
                    {"session": row["date"], "stage": r["stage"], "fault": fault}
                    for r in entries
                    if not r["stage"].startswith("revision.")
                    and not r["body"].get("requires_completion")
                    for fault in r["body"].get("faults", [])
                )
        return {
            "slot": CHECKPOINT_SLOT,
            "date": day,
            "reached": reached,
            "conditions_met": not failures if reached else None,
            "failures": failures,
        }

    def amend_resume(self, effective_session: str, owner_approval: str) -> dict:
        """Owner-approved resume of new positions, effective from a future session's 15:20."""
        if not owner_approval.strip():
            raise ObserverError("owner_approval_required")
        self.session_row(effective_session)
        now = self.clock()
        if now >= utc_at(effective_session, STAGE_CLOCKS["inputs"] + ":00"):
            raise ClockMismatch("resume_after_effective_stage_start")
        number = len(self.amendments()) + 1
        body = {
            "type": "resume",
            "effective_session": effective_session,
            "owner_approval": owner_approval,
            "open_halts": self.halt_reasons(effective_session),
        }
        return self.append("amendment", effective_session, f"amendment.{number}", body, now)

    def amend_session(self, day: str, status: str, close: str | None, owner_approval: str) -> dict:
        """Owner-approved calendar amendment; the original schedule stays in the record."""
        if not owner_approval.strip():
            raise ObserverError("owner_approval_required")
        original = self.session_row(day)
        now = self.clock()
        if now >= utc_at(day, "15:20:00") or any(
            r["session"] == day
            and (r["stage"] in ("inputs", "decision") or r["stage"].startswith("archive."))
            for r in self.records()
        ):
            raise ObserverError("calendar_amendment_after_observation")
        if status == "exchange_closed":
            revised = {**original, "kind": "closed", "close": None, "close_minute": None}
        elif status == "early_close":
            if (
                not close
                or not re.fullmatch(r"\d{2}:\d{2}", close)
                or not "09:31" <= close < "16:00"
            ):
                raise ObserverError("early_close_time_invalid")
            revised = {
                **original,
                "kind": "early_close",
                "close": close,
                "close_minute": minute_before(close),
            }
        else:
            raise ObserverError("amendment_status_invalid")
        number = len(self.amendments()) + 1
        body = {
            "type": "session_status",
            "owner_approval": owner_approval,
            "original": original,
            "revised": revised,
        }
        return self.append("amendment", day, f"amendment.{number}", body, now)

    def run_decision(self, day: str | None = None) -> dict:
        """15:25 stage: one immutable side from the frozen inputs only; never reads source data."""
        self.require_active()
        day = day or ny_date(self.clock())
        row = self.session_row(day)
        now = self.clock()
        window_end = utc_at(day, STAGE_CLOCKS["decision_close"] + ":00")
        if now < utc_at(day, STAGE_CLOCKS["decision_open"] + ":00"):
            raise TooEarly(f"decision:{day}")
        with self.claim(day, "decision") as prior:
            body: dict = {
                "started_utc": iso(now),
                "reconciled_claim": prior,
                "signal_side": None,
                "observer_side": 0,
                "halted": False,
                "halt_reasons": [],
                "r_first": None,
                "inputs_ref": None,
                "timely_valid": False,
            }
            faults: list[str] = []
            if row["kind"] != "full":
                body.update(status="not_applicable", reason=na_reason(row), faults=faults)
                return self.append("stage", day, "decision", body, now)
            inputs = self.stage_record(day, "inputs")  # the only evidence a decision may read
            sync = self.clock_status()
            reasons = self.halt_reasons(day)
            completed = self.clock()
            if now >= window_end:
                faults.append("decision_window_missed")
            else:
                if inputs is None:
                    faults.append("inputs_missing")
                elif not inputs["body"].get("usable"):
                    faults.append("inputs_unusable")
                    body["inputs_ref"] = inputs["sha256"]
                if completed >= window_end:
                    faults.append("decision_completed_after_window")
                if sync != CLOCK_STATUS_OK:
                    faults.append(f"clock_{sync}")
            if not faults:
                prior_close, p0959 = (
                    inputs["body"]["prior"]["close"],
                    inputs["body"]["bar_0959"]["c"],
                )
                body.update(
                    inputs_ref=inputs["sha256"],
                    r_first=first_return(prior_close, p0959),
                    signal_side=signal_side(prior_close, p0959),
                    timely_valid=True,
                    halted=bool(reasons),
                    halt_reasons=reasons,
                )
                body["observer_side"] = 0 if reasons else body["signal_side"]
            body["faults"] = faults
            body["completed_utc"] = iso(completed)
            body["status"] = "fault" if faults else "halted_no_position" if body["halted"] else "ok"
            return self.append_timed(day, "decision", body, completed, window_end, sync)

    def run_archive(self, day: str | None = None) -> dict:
        """16:20 stage (also recovery): capture the session's closing snapshot inside its grace."""
        self.require_active()
        day = day or ny_date(self.clock())
        row = self.session_row(day)
        delay = timedelta(seconds=self.manifest["source_delay_seconds"])
        now = self.clock()
        if row["kind"] == "closed":
            with self.claim(day, "archive.1"):
                body = {"status": "not_applicable", "reason": "exchange_closed", "faults": []}
                return self.append("stage", day, "archive.1", body, now)
        if now < utc_at(day, row["close_minute"] + ":00") + delay:
            raise TooEarly(f"archive:{day}")
        grace_end = utc_at(self.neighbours(day)[1], STAGE_CLOCKS["inputs"] + ":00")
        if now >= grace_end:
            raise ObserverError("grace_expired_use_revise")
        if self.canonical(day) is not None:
            raise DuplicateStage(f"{day}:canonical_valuation_exists_use_revise")
        return self.capture(day, self.next_capture_no(day), cutoff=grace_end, grace_end=grace_end)

    def run_recover(self, day: str | None = None) -> dict | None:
        """15:15 job: capture the prior session's missing archive; nothing to do returns None."""
        self.require_active()
        day = day or ny_date(self.clock())
        previous, _ = self.neighbours(day)
        if (
            previous is None
            or self.canonical(previous)
            or self.session_row(previous)["kind"] == "closed"
        ):
            return None
        if self.clock() < utc_at(day, STAGE_CLOCKS["recover"] + ":00"):
            raise TooEarly(f"recover:{day}")
        return self.run_archive(previous)

    def run_revise(self, day: str | None = None) -> dict:
        """Capture the snapshot again as an appendix record; it can never become canonical."""
        self.require_active()
        day = day or ny_date(self.clock())
        row = self.session_row(day)
        if row["kind"] == "closed":
            raise ObserverError("session_not_applicable")
        now = self.clock()
        end = utc_at(day, row["close_minute"] + ":00")
        if now < end + timedelta(seconds=self.manifest["source_delay_seconds"]):
            raise TooEarly(f"revision:{day}")
        number = 1 + sum(
            r["session"] == day and r["stage"].startswith("revision.") for r in self.records()
        )
        stage = f"revision.{number}"
        with self.claim(day, stage) as prior:
            fetch = self.fetch_bars(
                day, stage, utc_at(day, "09:30:00"), end, now + timedelta(minutes=5)
            )
            values, _ = extract_values(fetch["bars"], day, row)
            canonical = self.canonical(day)
            differences = {}
            for label, revised in values.items():
                original = canonical["body"]["values"].get(label) if canonical else None
                if canonical and original != revised:
                    differences[label] = {"canonical": original, "revised": revised}
            sync = self.clock_status()
            body = {
                "appendix_only": True,
                "status": "ok" if fetch["outcome"] == "success" else "fault",
                "faults": [] if fetch["outcome"] == "success" else [f"fetch_{fetch['outcome']}"],
                "canonical_ref": canonical["sha256"] if canonical else None,
                "values": values,
                "differences": differences,
                "fetch": {k: v for k, v in fetch.items() if k != "bars"},
                "reconciled_claim": prior,
            }
            return self.append("stage", day, stage, body, self.clock(), clock_status=sync)

    # issuer distributions ----------------------------------------------------------------

    def ingest_distributions(
        self,
        path: Path,
        *,
        sheet: str,
        source_url: str,
        cusip: str,
        fund: str,
        currency: str | None,
        decimals: int,
        header_row: int,
        columns: dict,
        date_format: str,
        covers_through: str,
        semantics_verified: bool,
        vendor_rows: list[dict] | None = None,
    ) -> dict:
        """Freeze one local issuer publication (bytes, hash, columns); revisions are new records."""
        if cusip != "78462F103":
            raise ObserverError("spy_cusip_required")
        raw = Path(path).read_bytes()
        now = self.clock()
        rows, defects = parse_distributions(
            read_xlsx(raw, sheet),
            columns,
            header_row=header_row,
            date_format=date_format,
            cusip=cusip,
        )
        digest, raw_path = self.store_raw(raw)
        number = 1 + sum(r["stage"].startswith("distributions.") for r in self.records())
        body = {
            "source_url": source_url,
            "source_sha256": digest,
            "raw_path": raw_path,
            "obtained_utc": iso(now),
            "sheet": sheet,
            "header_row": header_row,
            "columns": columns,
            "date_format": date_format,
            "cusip": cusip,
            "fund": fund,
            "currency": currency,
            "decimals": decimals,
            "covers_through": covers_through,
            "semantics_verified": semantics_verified,
            "rows": rows,
            "defects": defects,
            "vendor_rows": vendor_rows or [],
            "vendor_differences": compare_vendor(rows, vendor_rows or [], decimals),
        }
        return self.append("distribution_source", None, f"distributions.{number}", body, now)

    # report ------------------------------------------------------------------------------

    def report(self) -> dict:
        """Descriptive report of everything recorded; counts, faults and unknowns, no verdicts."""
        if self.verify_chain():
            raise BindingMismatch("ledger_integrity")
        as_of = self.clock()
        cal = self.manifest["calendar"]
        records = self.records()
        by_session: dict[str | None, list[dict]] = {}
        for record in records:
            by_session.setdefault(record["session"], []).append(record)
        faults = [
            {"session": r["session"], "stage": r["stage"], "fault": fault}
            for r in records
            if not r["body"].get("requires_completion")
            for fault in r["body"].get("faults", [])
        ]
        slots = dict.fromkeys(
            (
                "registered",
                "future",
                "started_full_sessions",
                "exchange_closed",
                "early_close",
                "long",
                "flat",
                "no_timely_decision",
                "halted_no_position",
            ),
            0,
        )
        valuation = dict.fromkeys(("ok", "late_capture", "unresolved", "pending", "future"), 0)
        operational = dict.fromkeys(
            ("inputs_usable", "decisions_timely_valid", "archives_on_time", "late_captures"), 0
        )
        fixed_entries, gaps, fixed_stopped = [], [], False
        observer_entries, observer_stopped = [], False
        closes: dict[str, float] = {}
        for row in self.effective_rows():
            day = row["date"]
            recs = by_session.get(day, [])
            inputs, decision = (self.completed_stage(recs, s) for s in ("inputs", "decision"))
            captures = [
                r
                for r in recs
                if r["stage"].startswith("archive.") and not r["body"].get("requires_completion")
            ]
            canonical = next((r for r in captures if r["body"].get("canonical")), None)
            started = as_of >= utc_at(day, STAGE_CLOCKS["inputs"] + ":00")
            grace_end = utc_at(self.neighbours(day)[1], STAGE_CLOCKS["inputs"] + ":00")
            full = row["kind"] == "full"
            if row["slot"] is not None and not started:
                slots["future"] += 1
            if row["kind"] != "closed" and not captures and as_of >= utc_at(day, "16:25:00"):
                faults.append(
                    {"session": day, "stage": "archive", "fault": "missing_stage:archive"}
                )
            if full and started and row["slot"] is not None:
                if inputs is None and as_of >= utc_at(day, STAGE_CLOCKS["cutoff"] + ":00"):
                    faults.append(
                        {"session": day, "stage": "inputs", "fault": "missing_stage:inputs"}
                    )
                if decision is None and as_of >= utc_at(
                    day, STAGE_CLOCKS["decision_close"] + ":00"
                ):
                    faults.append(
                        {"session": day, "stage": "decision", "fault": "missing_stage:decision"}
                    )
            if row["kind"] == "closed":
                state = "not_applicable"
            elif canonical:
                state = "late_capture" if canonical["body"]["late_capture"] else "ok"
                closes[day] = canonical["body"]["values"]["close_final"]
            elif as_of >= grace_end:
                state = "unresolved"
                faults.append({"session": day, "stage": "archive", "fault": "unresolved_valuation"})
            else:
                state = "pending" if started else "future"
            if row["slot"] is None:
                continue
            if row["kind"] == "closed":
                slots["exchange_closed"] += 1
            elif row["kind"] == "early_close":
                slots["early_close"] += 1
            if state in valuation:
                valuation[state] += 1
            if not (full and started):
                continue
            slots["started_full_sessions"] += 1
            body = decision["body"] if decision else None
            decision_due = decision is not None or as_of >= utc_at(
                day, STAGE_CLOCKS["decision_close"] + ":00"
            )
            timely = bool(body and body.get("timely_valid"))
            signal = body["signal_side"] if timely else None
            operational["inputs_usable"] += bool(inputs and inputs["body"].get("usable"))
            operational["decisions_timely_valid"] += timely
            operational["archives_on_time"] += bool(canonical and canonical["body"]["on_time"])
            operational["late_captures"] += state == "late_capture"
            if timely:
                slots["long" if signal == 1 else "flat"] += 1
            elif decision_due:
                slots["no_timely_decision"] += 1
            slots["halted_no_position"] += bool(body and body["status"] == "halted_no_position")
            r_last = None
            if canonical:
                v = canonical["body"]["values"]
                r_last = v["close_final"] / v["open_1530"] - 1
            if not fixed_stopped:
                if not decision_due:
                    fixed_stopped = True
                elif signal is None:
                    gaps.append({"date": day, "reason": "no_timely_signal"})
                    fixed_stopped = True
                elif r_last is None:
                    if state != "pending":
                        gaps.append({"date": day, "reason": "unresolved_valuation"})
                    fixed_stopped = True
                else:
                    fixed_entries.append((day, signal, r_last))
            if not observer_stopped:
                side = body["observer_side"] if body else 0
                if not decision_due or (side == 1 and canonical is None and state == "pending"):
                    observer_stopped = True
                else:
                    observer_entries.append((day, side, r_last))
                    observer_stopped = side == 1 and canonical is None
        slots["registered"] = sum(r["slot"] is not None for r in self.effective_rows())
        years, anchor = cal["year_end_sessions"], cal["anchor_date"]
        sources = [r for r in records if r["kind"] == "distribution_source"]
        latest = sources[-1] if sources else None
        reasons = distribution_reasons(latest, cal["last_date"])
        through = fixed_entries[-1][0] if fixed_entries else None
        cash = {}
        if latest and not latest["body"]["defects"]:
            for item in latest["body"]["rows"]:
                known = [
                    item["dividend"],
                    *[v for v in item["capital_gains"].values() if v is not None],
                ]
                cash[item["ex_date"]] = float(sum(Decimal(x) for x in known))
        benchmark = {
            "status": "incomplete",
            "reasons": list(reasons),
            "through_date": through,
            "total_return_wealth_end": None,
            "cagr": None,
            "distribution_dates_reinvested": None,
            "price_only": None,
        }
        if through is None:
            benchmark["reasons"].append("no_resolved_sessions")
        else:
            in_window = {d: a for d, a in cash.items() if anchor < d <= through}
            wealth, missing = benchmark_wealth(closes, anchor, through, in_window if latest else {})
            if missing:
                benchmark["reasons"] += missing
            elif latest and not latest["body"]["defects"]:
                benchmark.update(
                    status="provisional" if reasons else "complete",
                    total_return_wealth_end=wealth,
                    cagr=cagr(wealth, through, anchor),
                    distribution_dates_reinvested=len(in_window),
                )
            if anchor in closes and through in closes:
                benchmark["price_only"] = {
                    "wealth_end": START_EQUITY * closes[through] / closes[anchor],
                    "note": "price return only; excludes distributions; not a comparison basis",
                }
        sensitivity = {"status": "unavailable", "reason": reasons[0] if reasons else "unknown"}
        if cash and latest and not latest["body"]["defects"]:
            flat = [(d, 0 if d in cash else side, r) for d, side, r in fixed_entries]
            sensitivity = {
                "status": "provisional" if reasons else "complete",
                "base_pretax_wealth_end": compound(flat, BASE_COST, years, tax=False)["wealth_end"],
                "base_after_tax_wealth_end": compound(flat, BASE_COST, years)[
                    "wealth_end_after_terminal"
                ],
                "ex_dates_zeroed": sorted(d for d in cash if any(d == e[0] for e in fixed_entries)),
            }
        checkpoint = self.checkpoint_status(records)
        halts = self.halt_reasons("9999-12-31")
        activation = next((r["body"] for r in records if r["stage"] == "activation"), {})
        return {
            "schema": "intraday_observer_report",
            "version": VERSION,
            "as_of_utc": iso(as_of),
            "evidence_note": LOCAL_TIMESTAMP_NOTE,
            "identity": {
                "mode": self.manifest["mode"],
                "activated": self.has_record(None, "activation"),
                "readiness_sha256": next(
                    (r["body"]["readiness_sha256"] for r in records if r["stage"] == "activation"),
                    None,
                ),
                "manifest_sha256": self.manifest_sha256,
                "bindings": self.bindings,
                "state_dir": self.manifest["state_dir"],
                "source_preflight": activation.get(
                    "source_preflight", self.manifest["source_preflight"]
                ),
                "source_delay_seconds": self.manifest["source_delay_seconds"],
                "source_delay_verified": activation.get("source_delay_verified", False),
            },
            "integrity": {
                "records": len(records),
                "problems": self.verify_chain(),
                "stray_raw_files": self.stray_raw_files(),
            },
            "calendar": {
                "first_date": cal["first_date"],
                "checkpoint_date": cal["checkpoint_date"],
                "last_date": cal["last_date"],
                "anchor_date": anchor,
                "early_closes_registered": cal["early_closes"],
                "amendments": [
                    {
                        "stage": a["stage"],
                        "session": a["session"],
                        **{
                            k: a["body"].get(k)
                            for k in ("type", "original", "revised", "effective_session")
                        },
                    }
                    for a in self.amendments()
                ],
            },
            "slots": {**slots, "valuation": valuation},
            "faults": faults,
            "operational": {
                **operational,
                "checkpoint": checkpoint,
            },
            "fixed_rule": {
                "complete": not fixed_stopped and not gaps and slots["future"] == 0,
                "partial_through": through,
                "gaps": gaps,
                "statistics": describe(fixed_entries, years, anchor),
                "ex_distribution_flat_sensitivity": sensitivity,
                "note": "Needs a timely signal and a valuation for every full session.",
            },
            "observer": {
                "through_date": observer_entries[-1][0] if observer_entries else None,
                "statistics": describe(observer_entries, years, anchor),
                "halting_faults": halts,
                "note": "Hypothetical positions only; faults and halts leave the observer flat.",
            },
            "benchmark": benchmark,
            "distributions": [
                {
                    "stage": s["stage"],
                    "source_sha256": s["body"]["source_sha256"],
                    "obtained_utc": s["body"]["obtained_utc"],
                    "rows": len(s["body"]["rows"]),
                    "defects": s["body"]["defects"],
                    "vendor_differences": s["body"]["vendor_differences"],
                }
                for s in sources
            ],
            "tax_policy": "24% of positive yearly net P&L after costs, no loss benefit; a proxy.",
        }


def na_reason(row: dict) -> str:
    """Why a non-full session has no trading stages."""
    return "exchange_closed" if row["kind"] == "closed" else row["kind"]


def first_return(prior_close: float, p0959: float) -> float:
    """r_first = P_0959 / P_prior_close - 1."""
    return p0959 / prior_close - 1


def signal_side(prior_close: float, p0959: float) -> int:
    """Fixed rule: long only when the first-half-hour return is strictly positive."""
    return 1 if first_return(prior_close, p0959) > 0 else 0


def extract_values(bars: list[dict], day: str, row: dict) -> tuple[dict, list[str]]:
    """Pick the minute values the protocol uses; absent ones stay None and are listed."""
    wanted = {"close_final": (row["close_minute"], "c")}
    if row["kind"] == "full":
        wanted = {"close_0959": ("09:59", "c"), "open_1530": ("15:30", "o"), **wanted}
    by_stamp = {bar["ns"]: bar for bar in bars}
    values, missing = {}, []
    for label, (hhmm, field_name) in wanted.items():
        bar = by_stamp.get(int(utc_at(day, hhmm + ":00").timestamp()) * 1_000_000_000)
        values[label] = bar[field_name] if bar else None
        if bar is None:
            missing.append(label)
    return values, missing


# --- accounting and statistics --------------------------------------------------------------


def nw_t(values: Any) -> float | None:
    """Intercept-only OLS t with Bartlett HAC (maxlags=5, small-sample correction).

    Zero or non-finite variance is unresolved (None), never a huge t from floating residual noise.
    """
    import numpy as np
    import statsmodels.api as sm

    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or not np.all(np.isfinite(arr)) or np.all(arr == arr[0]):
        return None
    fit = sm.OLS(arr, np.ones(len(arr))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS, "use_correction": True}
    )
    se = float(fit.bse[0])
    return float(fit.params[0]) / se if math.isfinite(se) and se > 0 else None


def compound(
    entries: list[tuple],
    cost: float,
    year_ends: list[str],
    *,
    tax: bool = True,
    start: float = START_EQUITY,
    tax_rate: float = TAX_RATE,
) -> dict:
    """Compound whole equity through (date, side, r_last) entries; unknown stays unknown.

    A traded day returns r_last - 2*cost; a flat day returns zero and needs no price. Tax: at each
    registered year-end session debit ``tax_rate`` of the year's positive net P&L (after costs),
    no loss benefit or carryforward; the unfinished final year is accrued, not debited.
    """
    ends = set(year_ends)
    lookup = {day: (side, r) for day, side, r in entries}
    last = entries[-1][0] if entries else None
    days = sorted(set(lookup) | {d for d in ends if entries and entries[0][0] <= d <= last})
    wealth, pnl, pnl_year = start, 0.0, None
    annual, path, returns, unknown = [], [], [], None

    def debit(year: int) -> None:
        nonlocal wealth, pnl
        owed = tax_rate * max(pnl, 0.0)
        wealth -= owed
        annual.append({"year": year, "net_pnl": pnl, "tax_debit": owed, "wealth_after_tax": wealth})
        pnl = 0.0

    for day in days:
        if day in lookup:
            side, r = lookup[day]
            if side is None or (side == 1 and r is None):
                unknown = day
                break
            if tax and pnl_year is not None and int(day[:4]) != pnl_year:
                debit(pnl_year)  # safety net if a year-end session was not registered
            pnl_year = int(day[:4])
            s = r - 2 * cost if side == 1 else 0.0
            gain = wealth * s
            wealth, pnl = wealth + gain, pnl + gain
            returns.append(s)
            path.append((day, wealth))
        if tax and day in ends:
            debit(int(day[:4]))
            pnl_year = None
            if path and path[-1][0] == day:
                path[-1] = (day, wealth)
    if unknown:
        return {
            "wealth_end": None,
            "terminal_accrual": None,
            "wealth_end_after_terminal": None,
            "unknown_from": unknown,
            "annual": annual,
            "path": path,
            "returns": returns,
        }
    accrual = tax_rate * max(pnl, 0.0) if tax else 0.0
    return {
        "wealth_end": wealth,
        "terminal_accrual": accrual,
        "wealth_end_after_terminal": wealth - accrual,
        "unknown_from": None,
        "annual": annual,
        "path": path,
        "returns": returns,
    }


def cagr(wealth_end: float | None, through_date: str | None, anchor_date: str) -> float | None:
    """Growth rate over actual elapsed days from the anchor close divided by 365.25."""
    if wealth_end is None or through_date is None or wealth_end <= 0:
        return None
    years = (date.fromisoformat(through_date) - date.fromisoformat(anchor_date)).days / 365.25
    return (wealth_end / START_EQUITY) ** (1 / years) - 1 if years > 0 else None


def describe(entries: list[tuple], year_ends: list[str], anchor_date: str) -> dict:
    """Plain descriptive numbers for gross, base-cost and stress-cost paths (no thresholds)."""
    through = entries[-1][0] if entries else None
    traded = [(side, r) for _, side, r in entries if side == 1 and r is not None]
    summary = {
        "counts": {
            "sessions": len(entries),
            "long": sum(side == 1 for _, side, _ in entries),
            "flat": sum(side == 0 for _, side, _ in entries),
        },
        "through_date": through,
        "break_even_cost_per_side": (
            sum(r for _, r in traded) / (2 * len(traded)) if traded else None
        ),
    }
    for label, cost in (("gross", 0.0), ("base", BASE_COST), ("stress", STRESS_COST)):
        pre = compound(entries, cost, year_ends, tax=False)
        post = compound(entries, cost, year_ends, tax=True)
        known = pre["wealth_end"] is not None
        returns = pre["returns"]
        peak, drawdown, months, streak, run = START_EQUITY, 0.0, {}, 0, 0
        for (day, wealth), (_, side, _), s in zip(pre["path"], entries, returns, strict=False):
            peak = max(peak, wealth)
            drawdown = min(drawdown, wealth / peak - 1)
            months[day[:7]] = months.get(day[:7], 1.0) * (1 + s)
            if side == 1:
                run = run + 1 if s < 0 else 0
                streak = max(streak, run)
        summary[label] = {
            "pretax": {"wealth_end": pre["wealth_end"], "unknown_from": pre["unknown_from"]},
            "after_tax": {
                k: post[k]
                for k in (
                    "wealth_end",
                    "terminal_accrual",
                    "wealth_end_after_terminal",
                    "unknown_from",
                    "annual",
                )
            },
            "mean_bp": sum(returns) / len(returns) * 1e4 if known and returns else None,
            "nw5_t": nw_t(returns) if known else None,
            "cagr_pretax": cagr(pre["wealth_end"], through, anchor_date),
            "cagr_after_tax": cagr(post["wealth_end_after_terminal"], through, anchor_date),
            "tax_drag_wealth": (
                pre["wealth_end"] - post["wealth_end_after_terminal"] if known else None
            ),
            "max_drawdown": drawdown if known else None,
            "worst_month": min(v - 1 for v in months.values()) if known and months else None,
            "losing_streak": streak if known else None,
        }
    return summary


# --- issuer distribution schedule -----------------------------------------------------------

XLSX_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
EXCEL_EPOCH = date(1899, 12, 30)


def read_xlsx(raw: bytes, sheet: str) -> list[list]:
    """Cell values of one worksheet using only the standard library (blank cells are None)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        book = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        match = [s for s in book.iter(f"{XLSX_MAIN}sheet") if s.get("name") == sheet]
        if not match:
            raise ValueError(f"sheet_not_found:{sheet}")
        target = targets[match[0].get(XLSX_REL)].lstrip("/")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{XLSX_MAIN}t")) for si in root]
        sheet_xml = ElementTree.fromstring(
            archive.read(target if target.startswith("xl/") else f"xl/{target}")
        )
    rows = []
    for row in sheet_xml.iter(f"{XLSX_MAIN}row"):
        cells: dict[int, Any] = {}
        for cell in row.iter(f"{XLSX_MAIN}c"):
            letters = re.match(r"[A-Z]+", cell.get("r")).group()
            column = sum((ord(ch) - 64) * 26**i for i, ch in enumerate(reversed(letters))) - 1
            kind, value = cell.get("t"), cell.find(f"{XLSX_MAIN}v")
            if kind == "s" and value is not None:
                cells[column] = shared[int(value.text)]
            elif kind == "inlineStr":
                cells[column] = "".join(t.text or "" for t in cell.iter(f"{XLSX_MAIN}t"))
            elif kind == "str" and value is not None:
                cells[column] = value.text
            elif kind in (None, "n") and value is not None and value.text is not None:
                number = float(value.text)
                cells[column] = int(number) if number.is_integer() else number
        rows.append([cells.get(i) for i in range(max(cells, default=-1) + 1)])
    return rows


def _cell_date(value: Any, date_format: str) -> str | None:
    if value is None or value == "":
        return None
    if date_format == "excel_serial":
        return (EXCEL_EPOCH + timedelta(days=int(float(value)))).isoformat()
    if date_format == "mdy":
        return datetime.strptime(str(value), "%m/%d/%Y").date().isoformat()
    return valid_date(str(value))


def parse_distributions(
    rows: list[list], columns: dict, *, header_row: int, date_format: str, cusip: str
) -> tuple[list[dict], list[str]]:
    """Map issuer sheet rows to distribution records, listing every defect found."""
    header = rows[header_row - 1] if 0 < header_row <= len(rows) else []
    index = {str(name): i for i, name in enumerate(header)}
    wanted = [columns["ex_date"], columns["dividend"], *columns.get("capital_gains", [])]
    wanted += [columns["cusip"]] if columns.get("cusip") else []
    wanted += [columns[k] for k in ("record_date", "pay_date") if columns.get(k)]
    defects = [f"header_missing:{name}" for name in wanted if name not in index]
    if defects:
        return [], defects

    def at(row: list, name: str | None) -> Any:
        i = index.get(name) if name else None
        return row[i] if i is not None and i < len(row) else None

    parsed, seen = [], set()
    for row in rows[header_row:]:
        if all(v in (None, "") for v in row):
            continue
        if columns.get("cusip") and str(at(row, columns["cusip"])).strip() != cusip:
            continue
        try:
            ex = _cell_date(at(row, columns["ex_date"]), date_format)
            record = _cell_date(at(row, columns.get("record_date")), date_format)
            pay = _cell_date(at(row, columns.get("pay_date")), date_format)
        except (ValueError, SchemaError, OverflowError):
            defects.append("date_unreadable")
            continue
        if ex is None:
            defects.append("ex_date_blank")
            continue
        try:
            dividend = Decimal(str(at(row, columns["dividend"])))
        except InvalidOperation:
            dividend = Decimal(0)
        if dividend <= 0:
            defects.append(f"non_positive_dividend:{ex}")
        if ex in seen:
            defects.append(f"duplicate_ex_date:{ex}")
        seen.add(ex)
        if not offline_sessions(ex, ex):
            defects.append(f"ex_date_not_session:{ex}")
        if pay and pay < ex:
            defects.append(f"pay_before_ex:{ex}")
        gains = {}
        for name in columns.get("capital_gains", []):
            cell = at(row, name)
            gains[name] = None if cell in (None, "") else str(Decimal(str(cell)))
        parsed.append(
            {
                "ex_date": ex,
                "record_date": record,
                "pay_date": pay,
                "dividend": str(dividend),
                "capital_gains": gains,
            }
        )
    if not parsed:
        defects.append("no_matching_distribution_rows")
    return parsed, defects


def compare_vendor(issuer: list[dict], vendor: list[dict], decimals: int) -> list[dict]:
    """Differences after rounding both to the issuer's decimals half-up; never prefers a side."""
    quantum = Decimal(1).scaleb(-decimals)
    by_date = {row["ex_date"]: row for row in issuer}
    mismatches = []
    for row in vendor:
        if row["ex_date"] in by_date:
            a = Decimal(by_date[row["ex_date"]]["dividend"]).quantize(quantum, ROUND_HALF_UP)
            b = Decimal(str(row["amount"])).quantize(quantum, ROUND_HALF_UP)
            if a != b:
                mismatches.append({"ex_date": row["ex_date"], "issuer": str(a), "vendor": str(b)})
    return mismatches


def distribution_reasons(source: dict | None, last_date: str) -> list[str]:
    """Why the issuer comparison is not a complete terminal one (empty means complete)."""
    if source is None:
        return ["no_distribution_schedule"]
    body, reasons = source["body"], []
    if body["defects"]:
        reasons.append("source_defects")
    if body["currency"] != "USD":
        reasons.append("currency_unresolved")
    if not body["semantics_verified"]:
        reasons.append("field_semantics_unverified")
    if not body["columns"].get("cusip"):
        reasons.append("fund_identity_unverified")
    if not body["columns"].get("capital_gains"):
        reasons.append("capital_gain_columns_unmapped")
    if any(v is None for row in body["rows"] for v in row["capital_gains"].values()):
        reasons.append("capital_gain_blank_unknown")
    if body["vendor_differences"]:
        reasons.append("vendor_difference_unresolved")
    if body["covers_through"] < last_date:
        reasons.append("does_not_cover_window")
    if parse_iso(body["obtained_utc"]) <= utc_at(last_date, "16:00:00"):
        reasons.append("obtained_before_final_session")
    return reasons


def benchmark_wealth(
    closes: dict, anchor: str, through: str, cash: dict
) -> tuple[float | None, list]:
    """SPY units bought at the anchor close, distributions reinvested at the ex-date close proxy."""
    missing = [f"missing_close:{d}" for d in [anchor, through, *cash] if d not in closes]
    if missing:
        return None, sorted(set(missing))
    units = START_EQUITY / closes[anchor]
    for day in sorted(cash):
        if anchor < day <= through:
            units *= 1 + cash[day] / closes[day]
    return units * closes[through], []


# --- rehearsal --------------------------------------------------------------------------------


class SimClock:
    """Settable clock for offline rehearsals; sleeping just advances it."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def set(self, day: str, text: str) -> None:
        self.now = utc_at(day, text)


class RehearsalPrices:
    """Synthetic closes 100, 101, 102, ... per session: odd sessions open up, even ones down."""

    def __init__(self, sessions: list[dict]) -> None:
        self.index = {row["date"]: i for i, row in enumerate(sessions)}

    def __call__(self, day: str, hhmm: str) -> tuple[float, float, float, float]:
        close = 100.0 + self.index.get(day, 0)
        if hhmm == "09:59":
            price = close - 1 + (0.5 if self.index.get(day, 0) % 2 else -0.5)
        elif hhmm == "15:30":
            price = close - 0.4
        else:
            price = close
        return price, price + 0.01, price - 0.01, price


def rehearse(work_dir: Path, *, sessions: int = 3, activation_date: str = "2026-11-23") -> dict:
    """Run synthetic sessions through every stage with a simulated clock; no network, no secrets."""
    work = Path(work_dir)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError("work_dir_not_empty")
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    protocol = work / "inputs" / "protocol_stub.txt"
    protocol.write_text("synthetic rehearsal protocol stub; not the frozen protocol\n")
    snapshot = work / "inputs" / "official_calendar.json"
    after = date.fromisoformat(activation_date)
    snapshot.write_text(
        json.dumps(
            offline_sessions(
                activation_date, (after + timedelta(days=LOOKAHEAD_DAYS + 40)).isoformat()
            )
        )
    )
    clock = SimClock(utc_at(activation_date, "09:00:00"))
    manifest = init_state(
        work / "state",
        mode="rehearsal",
        activation_date=activation_date,
        official_calendar_file=snapshot,
        protocol_file=protocol,
        protocol_sha256=file_sha256(protocol),
        source_delay_seconds=900,
        now=clock(),
    )
    calendar = manifest["calendar"]
    early = {r["date"]: r["close"] for r in calendar["sessions"] if r["kind"] == "early_close"}
    deps = Deps(
        clock=clock,
        sleep=clock.sleep,
        transport=SyntheticTransport(RehearsalPrices(calendar["sessions"]), early),
        clock_sync=lambda: CLOCK_STATUS_OK,
    )
    ob = Observer(work / "state", deps)
    clock.set(activation_date, "10:00:00")
    ob.activate("rehearsal with synthetic data; not an owner approval of collection")
    clock.set(calendar["anchor_date"], "16:20:05")
    ob.run_archive(calendar["anchor_date"])
    days = [r["date"] for r in calendar["sessions"] if r["slot"]][:sessions]
    for day in days:
        for text, run in (
            ("15:20:05", ob.run_inputs),
            ("15:25:05", ob.run_decision),
            ("16:20:05", ob.run_archive),
        ):
            clock.set(day, text)
            run(day)
    clock.set(days[-1], "17:00:00")
    target = work / "rehearsal_report.json"
    atomic_checkpoint(target, ob.report(), initial=True)
    return {
        "mode": "rehearsal",
        "state_dir": str(work / "state"),
        "report_path": str(target),
        "anchor": calendar["anchor_date"],
        "sessions": days,
    }


# --- command line ---------------------------------------------------------------------------


@dataclass
class Deps:
    """Injectable collaborators; the command line never exposes these."""

    clock: Callable[[], datetime] = system_clock
    sleep: Callable[[float], None] = time.sleep
    transport: Any = None
    clock_sync: Callable[[], str] = host_clock_status


FETCHING = ("inputs", "archive", "recover", "revise")
SESSION_COMMANDS = ("inputs", "decide", "archive", "recover", "revise")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser; building it touches no files, credentials or network."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str, state: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        if state:
            p.add_argument("--state-dir", type=Path, required=True)
        return p

    command("plan", "print stages and bounds (offline, no writes)", False).add_argument(
        "--activation-date"
    )
    p = command("init", "explicit offline preparation of a state directory")
    p.add_argument("--mode", choices=("live", "rehearsal"), default="live")
    p.add_argument("--activation-date", required=True)
    p.add_argument("--official-calendar", type=Path, required=True)
    p.add_argument("--protocol-file", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--source-delay-seconds", type=int, required=True)
    p = command("activate", "write the activation record (registered date, owner approval)")
    p.add_argument("--owner-approval", required=True)
    p.add_argument("--readiness-sha256")
    p.add_argument(
        "--readiness-file", type=Path, help="verified, manifest-bound preflight JSON; required live"
    )
    for name, text in (
        ("inputs", "15:20 ET: freeze prior anchor and 09:59 close"),
        ("decide", "15:25 ET: write the one immutable side from frozen inputs"),
        ("archive", "16:20 ET: capture the session snapshot (also used for late recovery)"),
        ("recover", "15:15 ET: capture the prior session's missing archive, if any"),
        ("revise", "re-capture a session as an appendix-only revision"),
    ):
        command(name, text).add_argument("--session", help="New York session date (default today)")
    p = command("amend-resume", "owner-approved resume of new positions after a halt")
    p.add_argument("--effective-session", required=True)
    p.add_argument("--owner-approval", required=True)
    p = command("amend-session", "owner-approved calendar change (closure or early close)")
    p.add_argument("--session", required=True)
    p.add_argument("--status", choices=("exchange_closed", "early_close"), required=True)
    p.add_argument("--close", help="early close time HH:MM New York")
    p.add_argument("--owner-approval", required=True)
    p = command("distributions", "freeze a local issuer distribution workbook (no network)")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--sheet", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--cusip", required=True)
    p.add_argument("--fund", required=True)
    p.add_argument("--currency", help="documented currency (USD); omitted means unresolved")
    p.add_argument("--decimals", type=int, required=True)
    p.add_argument("--header-row", type=int, default=1)
    p.add_argument("--date-format", choices=("iso", "excel_serial", "mdy"), required=True)
    p.add_argument("--col-cusip", help="CUSIP column used to select the requested fund")
    p.add_argument("--col-ex-date", required=True)
    p.add_argument("--col-dividend", required=True)
    p.add_argument("--col-record-date")
    p.add_argument("--col-pay-date")
    p.add_argument("--col-capital-gain", action="append", default=[])
    p.add_argument("--covers-through", required=True)
    p.add_argument("--semantics-verified", action="store_true")
    p.add_argument("--vendor-json", type=Path)
    command("verify", "check the hash chain and raw payload hashes")
    command("report", "descriptive report as JSON").add_argument("--out", type=Path)
    p = command("rehearse", "run synthetic sessions offline in a new work directory", False)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--activation-date", default="2026-11-23")
    return parser


def emit(value: Any) -> None:
    """Print JSON for the operator."""
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def stage_summary(record: dict | None) -> dict:
    """Short operator summary of a stage record."""
    if record is None:
        return {"recorded": False}
    body = record["body"]
    return {
        "recorded": True,
        "session": record["session"],
        "stage": record["stage"],
        "seq": record["seq"],
        "status": body.get("status"),
        "faults": body.get("faults", []),
        "record_sha256": record["sha256"],
    }


def run_observer_command(args: argparse.Namespace, deps: Deps) -> int:
    """Commands that act on an existing state directory."""
    ob = Observer(args.state_dir, deps)
    if ob.manifest["mode"] == "rehearsal" and deps.transport is None:
        raise ObserverError("rehearsal state cannot use the live clock or source; use rehearse")
    if args.command in FETCHING and deps.transport is None:
        deps.transport = HttpTransport()  # fails on missing credentials before any claim or write
    day = getattr(args, "session", None)
    command = args.command
    if command == "activate":
        emit(
            stage_summary(
                ob.activate(args.owner_approval, args.readiness_sha256, args.readiness_file)
            )
        )
    elif command in SESSION_COMMANDS:
        run = {
            "inputs": ob.run_inputs,
            "decide": ob.run_decision,
            "archive": ob.run_archive,
            "recover": ob.run_recover,
            "revise": ob.run_revise,
        }[command]
        emit(stage_summary(run(day)))
    elif command == "amend-resume":
        emit(stage_summary(ob.amend_resume(args.effective_session, args.owner_approval)))
    elif command == "amend-session":
        emit(stage_summary(ob.amend_session(day, args.status, args.close, args.owner_approval)))
    elif command == "distributions":
        columns = {
            "cusip": args.col_cusip,
            "ex_date": args.col_ex_date,
            "dividend": args.col_dividend,
            "record_date": args.col_record_date,
            "pay_date": args.col_pay_date,
            "capital_gains": args.col_capital_gain,
        }
        vendor = json.loads(args.vendor_json.read_text()) if args.vendor_json else None
        record = ob.ingest_distributions(
            args.source,
            sheet=args.sheet,
            source_url=args.source_url,
            cusip=args.cusip,
            fund=args.fund,
            currency=args.currency,
            decimals=args.decimals,
            header_row=args.header_row,
            columns=columns,
            date_format=args.date_format,
            covers_through=args.covers_through,
            semantics_verified=args.semantics_verified,
            vendor_rows=vendor,
        )
        emit(
            {
                "stage": record["stage"],
                "rows": len(record["body"]["rows"]),
                "defects": record["body"]["defects"],
                "vendor_differences": record["body"]["vendor_differences"],
            }
        )
    elif command == "verify":
        problems = ob.verify_chain()
        emit(
            {
                "problems": problems,
                "stray_raw_files": ob.stray_raw_files(),
                "records": len(ob._names()),
            }
        )
        return 1 if problems else 0
    elif command == "report":
        report = ob.report()
        if args.out:
            atomic_checkpoint(args.out, report, initial=True)  # immutable local file, write once
        emit(report)
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    """Run one command. Only argument parsing happens before a subcommand is chosen."""
    args = build_parser().parse_args(argv)
    deps = deps or Deps()
    try:
        if args.command == "plan":
            emit(plan(args.activation_date))
        elif args.command == "init":
            manifest = init_state(
                args.state_dir,
                mode=args.mode,
                activation_date=args.activation_date,
                official_calendar_file=args.official_calendar,
                protocol_file=args.protocol_file,
                protocol_sha256=args.protocol_sha256,
                source_delay_seconds=args.source_delay_seconds,
                now=deps.clock(),
            )
            emit(
                {"prepared": True, "activated": False, "manifest_sha256": canonical_hash(manifest)}
            )
        elif args.command == "rehearse":
            emit(
                rehearse(
                    args.work_dir, sessions=args.sessions, activation_date=args.activation_date
                )
            )
        else:
            return run_observer_command(args, deps)
    except (ObserverError, ValueError, OSError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0
