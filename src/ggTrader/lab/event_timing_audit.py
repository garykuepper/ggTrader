"""Read-only, evidence-backed audit of event timing; no return computations."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from itertools import combinations as route_combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pandas_market_calendars as mcal
import psycopg2

from ggTrader.data.core.index_constituents import all_members_between, load_sp500_history
from ggTrader.lab.event_timing_sources import (
    ReferencedSources,
    SecEdgarClient,
    atomic_bytes,
    atomic_json,
    collect_filing,
    collect_identity_candidates,
    collect_release_candidates,
    collect_yahoo,
    digest,
    enumerate_sec,
    filing_urls,
    parse_filing_header,
    reviewed_fields,
)

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
MEMBERSHIP_FILE = ROOT / "data/universe/sp500_constituents_history.csv.gz"
RUSSELL_SNAPSHOT_FILE = ROOT / "data/universe/russell2000_tickers_snapshot_2026-06-09.txt"
CACHE_DIR = ROOT / "data/external/event_timing_audit"
SOURCE_MANIFEST = ROOT / "docs/research/briefs/2026-10-21-event-timing-audit-sources.md"
BRIEF = ROOT / "docs/research/briefs/2026-10-21-event-timing-audit.md"
APPROVED_AMENDMENT = (
    ROOT / "docs/research/briefs/2026-10-22-event-timing-source-conflict-proposal.md"
)
FILING_DATE_ONLY = "insider_filing_date_only"
CUTOFF = datetime.date(2026, 4, 30)
SMOKE_STRATUM_QUOTA = 2
PRED_PASS = "PASS"
PRED_FAIL = "FAIL"
PRED_UNVERIFIED = "UNVERIFIED"
STATUS_FREE_VERIFIED = "FREE—VERIFIED"
STATUS_PAID_DOCUMENTED = "PAID—DOCUMENTED PATH"
STATUS_NO_VERIFIED_PATH = "NO VERIFIED PATH"
STATUS_UNRESOLVED = "UNRESOLVED"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file without loading all into RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 2. Exchange Calendar & Audit Clock (§8 A1–A3)
# ============================================================================


class AuditClock:
    """NYSE exchange clock enforcing the timing rules in §8 A1–A3."""

    def __init__(self, calendar_name: str = "NYSE") -> None:
        self.calendar_name = calendar_name
        self.cal = mcal.get_calendar(calendar_name)
        self.tz = "America/New_York"
        # Cache schedules lazily
        self._schedules: Dict[Tuple[str, str], pd.DataFrame] = {}

    def get_schedule(self, start_date: str, end_date: str) -> pd.DataFrame:
        key = (start_date, end_date)
        if key not in self._schedules:
            sched = self.cal.schedule(start_date=start_date, end_date=end_date)
            # Normalize to America/New_York
            sched["market_open"] = sched["market_open"].dt.tz_convert(self.tz)
            sched["market_close"] = sched["market_close"].dt.tz_convert(self.tz)
            self._schedules[key] = sched
        return self._schedules[key]

    def normalize_to_ny(self, dt: Any) -> pd.Timestamp:
        """Convert any date/timestamp representation to America/New_York Timestamp."""
        ts = pd.Timestamp(dt)
        if pd.isna(ts):
            raise ValueError("Missing observed timestamp")
        normalized = (
            ts.tz_localize(self.tz, ambiguous="NaT", nonexistent="NaT")
            if ts.tz is None
            else ts.tz_convert(self.tz)
        )
        if pd.isna(normalized):
            raise ValueError("Ambiguous/nonexistent local timestamp")
        return normalized

    def is_nyse_session(self, dt: datetime.date | str | pd.Timestamp) -> bool:
        """Check if a date is a regular NYSE trading session."""
        date_str = str(pd.Timestamp(dt).date())
        sched = self.get_schedule(date_str, date_str)
        return not sched.empty

    def next_trading_session(self, dt: datetime.date | str | pd.Timestamp) -> datetime.date:
        """Return the date of the next regular NYSE trading session strictly after dt."""
        ts = pd.Timestamp(dt)
        next_day = ts + pd.Timedelta(days=1)
        horizon = ts + pd.Timedelta(days=14)
        sched = self.get_schedule(str(next_day.date()), str(horizon.date()))
        if sched.empty:
            raise RuntimeError(f"No NYSE trading session found within 14 days after {dt}")
        return sched.index[0].date()

    def tradeable_close_for_timestamp(
        self, availability_dt: Any
    ) -> Tuple[datetime.date, pd.Timestamp, str]:
        """A1: First regular session CLOSE at least 30 minutes after signal availability.

        Returns:
            (tradeable_date, tradeable_close_timestamp, precision_label='timestamp')
        """
        avail_ny = self.normalize_to_ny(availability_dt)
        cutoff_dt = avail_ny + pd.Timedelta(minutes=30)

        # Look up schedule starting on availability date through next 14 days
        start_date = str(avail_ny.date())
        end_date = str((avail_ny + pd.Timedelta(days=14)).date())
        sched = self.get_schedule(start_date, end_date)

        for session_idx, row in sched.iterrows():
            session_close = row["market_close"]
            if session_close >= cutoff_dt:
                session_date = (
                    session_idx.date()
                    if hasattr(session_idx, "date")
                    else pd.Timestamp(session_idx).date()
                )
                return session_date, session_close, "timestamp"

        raise RuntimeError(
            f"Could not determine tradeable close for availability {availability_dt}"
        )

    def tradeable_open_for_timestamp(
        self, availability_dt: Any
    ) -> Tuple[datetime.date, pd.Timestamp, str]:
        """A1 Diagnostic: First regular session OPEN at least 30 minutes after availability.

        Returns:
            (tradeable_date, tradeable_open_timestamp, precision_label='timestamp')
        """
        avail_ny = self.normalize_to_ny(availability_dt)
        cutoff_dt = avail_ny + pd.Timedelta(minutes=30)

        start_date = str(avail_ny.date())
        end_date = str((avail_ny + pd.Timedelta(days=14)).date())
        sched = self.get_schedule(start_date, end_date)

        for session_idx, row in sched.iterrows():
            session_open = row["market_open"]
            if session_open >= cutoff_dt:
                session_date = (
                    session_idx.date()
                    if hasattr(session_idx, "date")
                    else pd.Timestamp(session_idx).date()
                )
                return session_date, session_open, "timestamp"

        raise RuntimeError(f"Could not determine tradeable open for availability {availability_dt}")

    def tradeable_close_for_date_only(
        self, event_date: Any
    ) -> Tuple[datetime.date, pd.Timestamp, str]:
        """A2/A3 Date-only fallback: close of first trading session AFTER event_date.

        A1's 30-minute buffer is NOT applied again per §8 A1.
        Returns:
            (tradeable_date, tradeable_close_timestamp, precision_label='date_only')
        """
        event_d = (
            event_date if isinstance(event_date, datetime.date) else pd.Timestamp(event_date).date()
        )
        next_session_date = self.next_trading_session(event_d)
        date_str = str(next_session_date)
        sched = self.get_schedule(date_str, date_str)
        if sched.empty:
            raise RuntimeError(f"No NYSE session for date {date_str}")
        session_close = sched.iloc[0]["market_close"]
        return next_session_date, session_close, "date_only"


# ============================================================================
# 3. Anomaly Quarantine Reason Codes (§8 A8)
# ============================================================================


class QuarantineReason:
    """Frozen quarantine reason codes per §8 A8."""

    TX_DATE_PRE_1990 = "TX_DATE_PRE_1990"
    TX_DATE_POST_FILING = "TX_DATE_POST_FILING"
    FUTURE_DATED = "FUTURE_DATED"
    UNPARSEABLE_OR_MISSING_DATE = "UNPARSEABLE_OR_MISSING_DATE"
    UNRESOLVED_TIMESTAMP_CONFLICT = "UNRESOLVED_TIMESTAMP_CONFLICT"


def check_transaction_quarantine(
    transaction_date: Any,
    filing_date: Any,
    query_run_date: datetime.date,
    timestamp_conflict: bool = False,
) -> Optional[str]:
    """Apply the frozen transaction anomaly rules before grouping or reconstruction."""
    if timestamp_conflict:
        return QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT
    tx, filed = parse_date(transaction_date), parse_date(filing_date)
    if tx is None or filed is None:
        return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    if tx > query_run_date or filed > query_run_date:
        return QuarantineReason.FUTURE_DATED
    if tx < datetime.date(1990, 1, 1):
        return QuarantineReason.TX_DATE_PRE_1990
    if tx > filed:
        return QuarantineReason.TX_DATE_POST_FILING
    return None


def check_earnings_quarantine(
    earnings_date: Any, query_run_date: datetime.date, timestamp_conflict: bool = False
) -> Optional[str]:
    """Quarantine missing/future earnings dates; the pre-1990 rule is transaction-only."""
    if timestamp_conflict:
        return QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT
    date = parse_date(earnings_date)
    if date is None:
        return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    return QuarantineReason.FUTURE_DATED if date > query_run_date else None


def compute_deterministic_sample_hash(canonical_key: str) -> str:
    """Ascending SHA-256 of '20261021|canonical_key' per §8 A9."""
    text = f"20261021|{canonical_key}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_stratum(
    items: List[Any],
    key_fn: Callable[[Any], str],
    quota: int,
    smoke: bool = False,
) -> Tuple[List[Any], int]:
    """Select items without replacement in ascending SHA-256(20261021|key).

    Returns:
        (sampled_items, shortfall)
    """
    effective_quota = min(SMOKE_STRATUM_QUOTA, quota) if smoke else quota

    if not items:
        return [], effective_quota

    # Pair each item with its deterministic hash
    hashed = []
    seen_keys = set()
    for item in items:
        k = key_fn(item)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        h = compute_deterministic_sample_hash(k)
        hashed.append((h, item))

    # Sort ascending by hash
    hashed.sort(key=lambda t: t[0])

    if len(hashed) <= effective_quota:
        # Census the stratum if smaller than or equal to quota
        selected = [item for _, item in hashed]
        shortfall = max(0, effective_quota - len(selected))
        return selected, shortfall

    selected = [item for _, item in hashed[:effective_quota]]
    return selected, 0


@dataclass
class AnnualA6Coverage:
    year: int
    denominator_units: int
    resolved_identity_units: int
    retrieved_units: int
    in_scope_events: int
    quarantined_events: int
    priced_events: int
    missing_units: int
    missing_events: int

    @property
    def identity_ratio(self) -> float:
        if self.denominator_units <= 0:
            return 0.0
        return self.resolved_identity_units / self.denominator_units

    @property
    def retrieval_ratio(self) -> float:
        if self.resolved_identity_units <= 0:
            return 0.0
        return self.retrieved_units / self.resolved_identity_units

    @property
    def price_ratio(self) -> float:
        # In-scope events includes quarantined and right-censored per §8 A6(c)
        return self.priced_events / self.in_scope_events if self.in_scope_events > 0 else 1.0

    @property
    def unit_missingness_ratio(self) -> float:
        return self.missing_units / self.denominator_units if self.denominator_units > 0 else 0.0

    @property
    def event_missingness_ratio(self) -> float:
        return self.missing_events / self.in_scope_events if self.in_scope_events > 0 else 0.0

    def passes_a6(self) -> bool:
        if self.denominator_units == 0:
            return False
        # If in_scope_events is 0, passes (c) only if (a), (b), and (d) pass
        c_pass = (self.price_ratio >= 0.95) if self.in_scope_events > 0 else True
        return (
            self.identity_ratio >= 0.95
            and self.retrieval_ratio >= 0.95
            and c_pass
            and self.unit_missingness_ratio <= 0.02
            and self.event_missingness_ratio <= 0.02
        )


# ============================================================================
# 13. A7 Requirement Matrix Evaluator (§8 A7)
# ============================================================================


def numeric_value(value: Any) -> float | None:
    """Recognize missing/malformed numeric metadata without calculating a signal."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def parse_date(value: Any) -> datetime.date | None:
    """Parse required dates without repairing malformed values."""
    try:
        return pd.Timestamp(value).date() if value is not None and not pd.isna(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def evaluate_eligible_window(
    annual_results: dict[int, bool], min_years: int = 5
) -> tuple[tuple[int, int] | None, int]:
    """Longest contiguous run of complete years; ties favour the most recent."""
    runs: list[tuple[int, int, int]] = []
    start = None
    previous = None
    for year in sorted(annual_results):
        passed = annual_results[year] and year < CUTOFF.year
        if previous is not None and (year != previous + 1 or not passed):
            if start is not None:
                runs.append((previous - start + 1, previous, start))
            start = None
        if passed and start is None:
            start = year
        previous = year
    if start is not None and previous is not None:
        runs.append((previous - start + 1, previous, start))
    valid = [r for r in runs if r[0] >= min_years]
    if not valid:
        return None, 0
    length, end, start = max(valid)
    return (start, end), length


def reconcile_times(readings: list[Any], clock: AuditClock) -> dict:
    """Compare readings of one field only; use latest supported time within one minute."""
    try:
        times = [clock.normalize_to_ny(t) for t in readings if t is not None]
        if any(pd.isna(t) for t in times):
            raise ValueError("Missing timestamp")
    except (ValueError, TypeError):
        return {"time": None, "predicate": PRED_UNVERIFIED, "reason": "unparseable timestamp"}
    if not times:
        return {"time": None, "predicate": PRED_UNVERIFIED, "reason": "no timestamp evidence"}
    if (max(times) - min(times)).total_seconds() > 60:
        return {
            "time": None,
            "predicate": PRED_FAIL,
            "reason": QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT,
        }
    return {"time": max(times), "predicate": PRED_PASS, "reason": "latest supported reading"}


@dataclass
class Denominators:
    """Dated PIT units, independent of whether any event was stored."""

    membership_sha256: str
    pead_member_quarters: dict[int, int]
    pead_member_quarters_detail: dict[str, int]
    insider_member_months: dict[int, int]
    insider_member_months_detail: dict[str, int]
    russell_snapshot_count: int
    ever_members_count: int
    latest_members_count: int
    removed_members_count: int
    units: dict[str, list[dict]] = field(default_factory=dict)


def membership_units(history: pd.DataFrame, start: str, frequency: str) -> list[dict]:
    """Union incumbents and every change; retain each unit's actual membership intervals."""
    units = []
    for period in pd.period_range(start, CUTOFF, freq=frequency):
        first = period.start_time.tz_localize("UTC")
        last = min(period.end_time.tz_localize("UTC"), pd.Timestamp(CUTOFF, tz="UTC"))
        if first < history.index[0]:
            continue
        members = all_members_between(first, last, history)
        changes = list(history.index[(history.index > first) & (history.index <= last)])
        boundaries = [first, *changes, last + pd.Timedelta(days=1)]
        segments = [
            (left, right, set(all_members_between(left, left, history)))
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ]
        for symbol in members:
            intervals = []
            for left, right, active in segments:
                if symbol in active:
                    intervals.append([str(left.date()), str((right - pd.Timedelta(days=1)).date())])
            units.append(
                {
                    "key": f"{symbol}|{period}",
                    "symbol": symbol,
                    "period": str(period),
                    "year": period.year,
                    "start": str(first.date()),
                    "end": str(last.date()),
                    "membership_intervals": intervals,
                    "partial_year": period.year == 2026,
                }
            )
    return units


def build_universe_denominators(
    membership_file: Path = MEMBERSHIP_FILE, russell_file: Path = RUSSELL_SNAPSHOT_FILE
) -> Denominators:
    """Build all A5 periods, including periods with no constituent change."""
    history = load_sp500_history(membership_file)
    units = {
        "pead": membership_units(history, "2002-01", "Q"),
        "insider_cluster": membership_units(history, "2015-01", "M"),
    }
    counts: dict[str, tuple[dict, dict]] = {}
    for signal, rows in units.items():
        annual: dict[int, int] = {}
        detail: dict[str, int] = {}
        for row in rows:
            annual[row["year"]] = annual.get(row["year"], 0) + 1
            detail[row["period"]] = detail.get(row["period"], 0) + 1
        counts[signal] = annual, detail
    ever = set().union(*map(set, history["tickers"]))
    latest = set(history.iloc[-1]["tickers"])
    return Denominators(
        compute_file_sha256(membership_file),
        *counts["pead"],
        *counts["insider_cluster"],
        len(russell_file.read_text().splitlines()) if russell_file.exists() else 0,
        len(ever),
        len(latest),
        len(ever - latest),
        units,
    )


class ReadOnlyDb:
    """Explicit SELECTs in read-only transactions, using exported settings only."""

    def __init__(self) -> None:
        self.query_log: list[dict] = []

    def execute_select(
        self, sql: str, params: tuple | None = None
    ) -> tuple[list[str], list[tuple]]:
        """Reject write syntax before connecting; never expose a DSN or connection exception."""
        statement = sql.strip().rstrip(";")
        if (
            not re.match(r"^SELECT\s", statement, re.I)
            or ";" in statement
            or re.search(
                r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|CALL|INTO|DO|SET)\b",
                statement,
                re.I,
            )
        ):
            raise ValueError("Audit accepts a single explicit read-only SELECT")
        dsn = os.environ.get("POSTGRES_CONNECTION_STRING", "").replace(
            "postgresql+psycopg2://", "postgresql://"
        )
        settings = {
            key: os.environ[env]
            for key, env in (
                ("host", "DB_HOST"),
                ("port", "DB_PORT"),
                ("dbname", "DB_NAME"),
                ("user", "DB_USER"),
                ("password", "DB_PASS"),
            )
            if env in os.environ
        }
        if not dsn and not settings:
            raise ValueError("Export database settings; the audit does not read .env")
        conn = None
        entry = {
            "sql": statement,
            "params": list(params or ()),
            "query_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "observation_cutoff": str(CUTOFF),
        }
        try:
            conn = psycopg2.connect(
                dsn,
                options="-c statement_timeout=30000 -c default_transaction_read_only=on",
                **settings,
            )
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor() as cur:
                cur.execute(statement, params or ())
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            entry.update(columns=columns, rows_returned=len(rows), outcome="success")
            return columns, rows
        except psycopg2.Error:
            entry["outcome"] = "database unavailable"
            raise RuntimeError("Read-only audit query unavailable") from None
        finally:
            self.query_log.append(entry)
            if conn is not None:
                conn.rollback()
                conn.close()


def load_existing_artifact_flags(path: Path) -> dict | None:
    """Load a counts/flags-only export; unknown schemas never imply clean evidence."""
    try:
        data = json.loads(path.read_text())
        # The legacy replica artifact is not a documented exhaustive security-date flag map.
        if data.get("schema") != "event-timing-detector-flags-v1":
            return None
        if not isinstance(data.get("screened"), list) or not isinstance(data.get("flags"), list):
            return None
        for interval in data["screened"]:
            if not all(interval.get(k) for k in ("symbol", "start", "end", "source_ref")):
                return None
            if parse_date(interval["start"]) is None or parse_date(interval["end"]) is None:
                return None
        for flag in data["flags"]:
            if not flag.get("symbol") or parse_date(flag.get("date")) is None:
                return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def check_tape_integrity_batch(
    db: ReadOnlyDb | None, clock: AuditClock, events: list[dict], artifact_flags: dict | None
) -> dict[str, dict]:
    """Count bars for every usable event; never SELECT price values or infer clean screening."""
    results = {}
    for event in events:
        key = event["event_id"]
        date = parse_date(event.get("tradeable_date"))
        result = {
            "predicate": PRED_UNVERIFIED,
            "bar_count": None,
            "reason": "missing tradeable date",
        }
        if event.get("quarantine"):
            result["reason"] = "quarantined input; not used"
        elif date is None:
            pass
        elif not clock.is_nyse_session(date):
            result.update(predicate=PRED_FAIL, reason="not an NYSE session")
        elif not event.get("identity_resolved"):
            result["reason"] = "unresolved historical price security"
        elif db is None:
            result["reason"] = "bar-existence evidence unavailable"
        else:
            _, rows = db.execute_select(
                "SELECT count(*) FROM ohlcv WHERE venue = 'yfinance' "
                "AND interval = '1d' AND symbol = %s "
                "AND timestamp::date = %s",
                (event["price_symbol"], date),
            )
            count = int(rows[0][0])
            result["bar_count"] = count
            screened = artifact_flags is not None and any(
                r["symbol"] == event["price_symbol"]
                and parse_date(r["start"]) <= date <= parse_date(r["end"])
                for r in artifact_flags["screened"]
            )
            flagged = artifact_flags is not None and any(
                r["symbol"] == event["price_symbol"] and parse_date(r["date"]) == date
                for r in artifact_flags["flags"]
            )
            if count != 1 or flagged:
                result.update(
                    predicate=PRED_FAIL, reason="missing/duplicate bar or existing detector flag"
                )
            elif not screened:
                result["reason"] = "detector evidence unavailable for security-date"
            else:
                result.update(
                    predicate=PRED_PASS,
                    reason="unique session bar; detector-screened, not certified",
                )
        results[key] = result
    return results


def evidence_supported(record: dict, client: SecEdgarClient) -> bool:
    """Check byte integrity only; this does not validate semantic annotations."""
    if not isinstance(record, dict):
        return False
    refs = record.get("source_refs", [])
    if not isinstance(refs, list) or not refs:
        return False
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("url") or not ref.get("sha256"):
            return False
        try:
            actual = client.reference(ref["url"])
        except ValueError:
            return False
        if actual is None or actual.get("status_code") != 200 or actual["sha256"] != ref["sha256"]:
            return False
    return True


def semantic_linkage(kind: str, record: dict) -> dict:
    """Bind interpreted facts to their specific security/issuer, period or product."""
    keys = {
        "identities": ("symbol", "cik", "security_id", "valid_from", "valid_to"),
        "pead_events": ("symbol", "issuer_cik", "fiscal_period", "earnings_date"),
        "eps": ("issuer_cik", "fiscal_period"),
        "classifications": ("symbol", "security_id", "effective_date"),
        "amendments": ("accession", "issuer_cik", "symbol"),
        "coverage_causes": ("symbol", "start", "end"),
        "routes": ("route", "product", "representation"),
        "dissemination": ("cik", "form", "filing_date"),
    }[kind]
    return {k: record[k] for k in keys} if all(record.get(k) for k in keys) else {}


def semantic_fields_supported(kind: str, record: dict, client: SecEdgarClient) -> bool:
    """Require independently located, field-specific reviewed interpretations."""
    fields = {
        "identities": ["symbol", "cik", "security_id", "price_symbol", "valid_from", "valid_to"],
        "pead_events": ["symbol", "issuer_cik", "fiscal_period", "earnings_date"],
        "eps": ["issuer_cik", "fiscal_period", "value", "basis", "currency", "split_basis"],
        "classifications": ["status", "effective_date", "security_id", "evidence_form"],
        "amendments": ["replaces_lot_ids"]
        if not record.get("no_signal_input_change")
        else ["no_signal_input_change"],
        "coverage_causes": ["symbol", "start", "end", "reason"],
        "dissemination": ["dissemination_bound_seconds"],
    }[kind]
    scalar_fields = [
        f
        for f in fields
        if f
        not in (
            "value",
            "replaces_lot_ids",
            "no_signal_input_change",
            "dissemination_bound_seconds",
        )
    ]
    if not all(isinstance(record.get(f), str) and record[f] for f in scalar_fields):
        return False
    if kind in ("identities", "pead_events", "eps"):
        if not str(
            record.get("cik") if kind == "identities" else record.get("issuer_cik")
        ).isdigit():
            return False
    if kind == "identities" and not (
        parse_date(record["valid_from"])
        and parse_date(record["valid_to"])
        and parse_date(record["valid_from"]) <= parse_date(record["valid_to"])
    ):
        return False
    if kind == "amendments" and not record.get("no_signal_input_change"):
        if not isinstance(record.get("replaces_lot_ids"), list) or not all(
            isinstance(key, str) and key for key in record["replaces_lot_ids"]
        ):
            return False
    if kind == "eps" and "snapshot_time" in record:
        fields = [*fields, "snapshot_time"]
    return evidence_supported(record, client) and reviewed_fields(
        record, fields, client, semantic_linkage(kind, record)
    )


def validate_evidence_record(
    kind: str, record: dict, client: SecEdgarClient, inventories: list[dict]
) -> dict | None:
    """Reparse machine facts from declared bytes; never trust imported completeness flags."""
    if not evidence_supported(record, client):
        return None
    try:
        source = ReferencedSources(client, record["source_refs"])
        if kind == "inventories":
            rebuilt = enumerate_sec(source, record["cik"])
            dates = [
                parse_date(client.reference(r["url"]).get("timestamp"))
                for r in record["source_refs"]
            ]
            if not rebuilt["complete"] or not dates or any(d is None for d in dates):
                return None
            rebuilt["end"] = str(min(dates))
            if (
                record.get("filings") != rebuilt["filings"]
                or parse_date(record.get("start")) != parse_date(rebuilt["start"])
                or parse_date(record.get("end")) != parse_date(rebuilt["end"])
            ):
                return None
            return rebuilt
        if kind == "filings":
            entries = [
                (i, f)
                for i in inventories
                for f in i.get("filings", [])
                if f["accession"] == record.get("accession")
                and str(i["cik"]).lstrip("0") == str(record.get("cik")).lstrip("0")
            ]
            if len(entries) != 1:
                return None
            inventory, filing = entries[0]
            rebuilt = collect_filing(source, inventory["cik"], filing)
            if str(rebuilt.get("issuer_cik", "")).lstrip("0") != str(inventory["cik"]).lstrip("0"):
                return None
            # API/header/lot fields come only from their canonical accession URLs.
            for field in ("dissemination_bound_seconds", "field_reviews"):
                if field in record:
                    rebuilt[field] = record[field]
            rebuilt["source_refs"] = record["source_refs"]
            return rebuilt
        if kind == "pead_fetches":
            return collect_yahoo(source, record["symbol"])
        if kind == "release_candidates":
            entries = [
                f
                for i in inventories
                for f in i.get("filings", [])
                if f["accession"] == record.get("accession")
                and str(i["cik"]).lstrip("0") == str(record.get("cik")).lstrip("0")
            ]
            if len(entries) != 1:
                return None
            rebuilt = collect_release_candidates(source, record["cik"], entries[0])
            # Original-release semantics need reviewed issuer/period facts on the event.
            return rebuilt
        if kind == "routes":
            return record  # Every used dimension is reviewed separately by the route evaluator.
        return record if semantic_fields_supported(kind, record, client) else None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError):
        return None


def identity_for(symbol: str, date: datetime.date, identities: list[dict]) -> dict | None:
    """Resolve a unique historical security, issuer and tape mapping on a date."""
    matches = [
        r
        for r in identities
        if r["symbol"] == symbol
        and parse_date(r.get("valid_from")) is not None
        and parse_date(r.get("valid_to")) is not None
        and parse_date(r["valid_from"]) <= date <= parse_date(r["valid_to"])
        and all(r.get(k) for k in ("security_id", "cik", "price_symbol"))
    ]
    keys = {(str(r["cik"]), r["security_id"], r["price_symbol"]) for r in matches}
    return matches[0] if len(keys) == 1 else None


def resolve_unit(unit: dict, identities: list[dict]) -> list[dict] | None:
    """Require historical identity coverage of every day the security was a member."""
    mappings = []
    for start, end in unit["membership_intervals"]:
        cursor = parse_date(start)
        final = parse_date(end)
        while cursor <= final:
            mapping = identity_for(unit["symbol"], cursor, identities)
            if mapping is None:
                return None
            last = min(parse_date(mapping["valid_to"]), final)
            # Reject a competing mapping that begins before this interval ends.
            boundaries = [
                parse_date(r["valid_from"])
                for r in identities
                if r["symbol"] == unit["symbol"]
                and parse_date(r.get("valid_from")) is not None
                and cursor < parse_date(r["valid_from"]) <= last
            ]
            if boundaries:
                last = min(boundaries) - datetime.timedelta(days=1)
            mappings.append(mapping)
            cursor = last + datetime.timedelta(days=1)
    return mappings or None


def unit_inventory(unit: dict, mappings: list[dict] | None, signal: str, evidence: dict) -> dict:
    """Measure attempted/resolved/retrieved/stored independently for one PIT unit."""
    result = dict(
        unit,
        attempted=False,
        resolved=bool(mappings),
        retrieved=False,
        stored=False,
        missing_reason=None,
        evidence_refs=[],
    )
    result["issuer_ciks"] = sorted({str(m["cik"]).lstrip("0") for m in mappings or []})
    if not mappings:
        result["missing_reason"] = "historical_identity_unverified"
        return result
    if signal == "insider_cluster":
        inventories = [
            i
            for i in evidence.get("inventories", [])
            if any(str(i["cik"]).lstrip("0") == str(m["cik"]).lstrip("0") for m in mappings)
        ]
        result["attempted"] = bool(inventories)
        covered = []
        for mapping in mappings:
            matching = [
                i
                for i in inventories
                if str(i["cik"]).lstrip("0") == str(mapping["cik"]).lstrip("0")
                and i.get("request_succeeded")
                and i.get("complete")
                and parse_date(i.get("start")) is not None
                and parse_date(i.get("end")) is not None
                and parse_date(i["start"]) <= parse_date(unit["start"])
                and parse_date(i["end"]) >= parse_date(unit["end"])
            ]
            covered.append(bool(matching))
        result["retrieved"] = all(covered)
        result["evidence_refs"] = [r for i in inventories for r in i["source_refs"]]
    else:
        fetches = [r for r in evidence.get("pead_fetches", []) if r["symbol"] == unit["symbol"]]
        result["attempted"] = bool(fetches)
        records = [
            r
            for f in fetches
            if f.get("request_succeeded")
            for r in f.get("records", [])
            if parse_date(r.get("earnings_date")) is not None
            and parse_date(unit["start"])
            <= parse_date(r["earnings_date"])
            <= parse_date(unit["end"])
        ]
        result["retrieved"] = bool(records)
        result["evidence_refs"] = [r for f in fetches for r in f["source_refs"]]
    if not result["retrieved"]:
        result["missing_reason"] = "source_inventory_unavailable"
    return result


def filing_date_corroboration(
    record: dict, client: SecEdgarClient | None, inventories: list[dict]
) -> dict:
    """Reparse exact issuer/accession filing dates from checksummed declared sources."""
    result = {"predicate": PRED_UNVERIFIED, "reason": "required raw date pair unavailable"}
    if client is None:
        return result
    try:
        cik = str(int(record["cik"]))
        candidates = []
        for inventory in inventories:
            if str(int(inventory["cik"])) != cik:
                continue
            source = ReferencedSources(client, inventory["source_refs"])
            rebuilt = enumerate_sec(source, cik)
            candidates.extend(
                (rebuilt, filing)
                for filing in rebuilt["filings"]
                if filing["accession"] == record["accession"]
            )
        if len(candidates) != 1:
            return result
        inventory, filing = candidates[0]
        source = ReferencedSources(client, record["source_refs"])
        raw = collect_filing(source, cik, filing)
        if str(int(raw.get("issuer_cik", ""))) != cik:
            return dict(result, reason="raw ownership issuer absent or mismatched")
        _, header_url = filing_urls(cik, filing)
        header = source.fetch_sec_url(header_url)
        if header is None:
            return result
        checked = parse_filing_header(header, filing["accession"], issuer_cik=cik)
        api_date, header_date = filing.get("filing_date"), checked.get("header_filing_date")
        result.update(
            api_filing_date=api_date,
            header_filing_date=header_date,
            cik=cik,
            accession=filing["accession"],
            source_refs=inventory["source_refs"] + raw["source_refs"],
        )
        if parse_date(api_date) is None or parse_date(header_date) is None:
            return dict(result, reason=checked.get("reason", result["reason"]))
        if (
            api_date != header_date
            or record.get("filing_date") != api_date
            or record.get("header_filing_date") != header_date
            or str(int(record.get("issuer_cik", ""))) != cik
        ):
            return dict(result, predicate=PRED_FAIL, reason="filing-date or identity conflict")
        return dict(
            result, predicate=PRED_PASS, reason="exact raw filing-date pair; convention only"
        )
    except (ValueError, KeyError, TypeError, OSError, UnicodeError):
        return result


def event_clock(
    record: dict,
    signal: str,
    clock: AuditClock,
    query_date: datetime.date,
    client: SecEdgarClient | None = None,
    *,
    representation: str | None = None,
    inventories: list[dict] | None = None,
) -> dict:
    """Keep announcement, acceptance, availability and inferred execution separate."""
    if (
        representation not in (None, FILING_DATE_ONLY)
        or representation
        and signal != "insider_cluster"
    ):
        raise ValueError("Unsupported event timing representation")
    event = dict(record)
    date = record.get("earnings_date") if signal == "pead" else record.get("filing_date")
    quarantine = check_earnings_quarantine(date, query_date)
    for extra in (
        record.get("filing_date"),
        record.get("header_filing_date"),
        record.get("consensus", {}).get("snapshot_time"),
    ):
        if extra is not None:
            parsed = parse_date(extra)
            if parsed is None:
                quarantine = QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
            elif parsed > query_date:
                quarantine = QuarantineReason.FUTURE_DATED
    announcement = reconcile_times(record.get("announcement_readings", []), clock)
    acceptance = reconcile_times([record.get("acceptance"), record.get("header_acceptance")], clock)
    relevant = [announcement, acceptance] if signal == "pead" else [acceptance]
    if representation is None and any(r["predicate"] == PRED_FAIL for r in relevant):
        quarantine = QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT
    for reading in record.get("announcement_readings", []) + [
        record.get("acceptance"),
        record.get("header_acceptance"),
    ]:
        if reading is not None:
            try:
                if clock.normalize_to_ny(reading).date() > query_date:
                    quarantine = QuarantineReason.FUTURE_DATED
            except (TypeError, ValueError):
                quarantine = QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    event.update(
        quarantine=quarantine,
        announcement_time=announcement["time"],
        acceptance_time=acceptance["time"],
        availability=None,
        tradeable_date=None,
        tradeable_close=None,
        diagnostic_open=None,
        precision=None,
    )
    if representation == FILING_DATE_ONLY:
        corroboration = filing_date_corroboration(record, client, inventories or [])
        event.update(
            representation=representation,
            original_clock=event_clock(record, signal, clock, query_date, client),
            acceptance_reconciliation=acceptance["predicate"]
            if record.get("acceptance") is not None and record.get("header_acceptance") is not None
            else PRED_UNVERIFIED,
            filing_date_corroboration=corroboration,
            dissemination_predicate=PRED_UNVERIFIED,
            timestamp_predicate=PRED_UNVERIFIED,
            precision="date_only",
        )
        if corroboration["predicate"] == PRED_FAIL and quarantine is None:
            event["quarantine"] = QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT
        if event["quarantine"] or corroboration["predicate"] != PRED_PASS:
            return event
        date, close, _ = clock.tradeable_close_for_date_only(corroboration["api_filing_date"])
        event.update(
            tradeable_date=str(date),
            tradeable_close=close.isoformat(),
            timestamp_predicate=PRED_PASS,
        )
        return event
    if quarantine:
        return event
    supported = [r["time"] for r in relevant if r["time"] is not None]
    if signal == "insider_cluster":
        bound = numeric_value(record.get("dissemination_bound_seconds"))
        bound_supported = (
            client is not None
            and bound is not None
            and bound >= 0
            and semantic_fields_supported("dissemination", record, client)
        )
        event["dissemination_predicate"] = PRED_PASS if bound_supported else PRED_UNVERIFIED
        event["acceptance_reconciliation"] = (
            PRED_PASS
            if acceptance["predicate"] == PRED_PASS
            and record.get("acceptance") is not None
            and record.get("header_acceptance") is not None
            else PRED_UNVERIFIED
        )
        supported = (
            [acceptance["time"] + pd.Timedelta(seconds=bound)]
            if (bound_supported and acceptance["time"] is not None)
            else []
        )
    if supported:
        availability = max(supported)
        date, close, precision = clock.tradeable_close_for_timestamp(availability)
        _, diagnostic_open, _ = clock.tradeable_open_for_timestamp(availability)
        event.update(
            availability=availability.isoformat(), diagnostic_open=diagnostic_open.isoformat()
        )
    else:
        date, close, precision = clock.tradeable_close_for_date_only(date)
    event.update(tradeable_date=str(date), tradeable_close=close.isoformat(), precision=precision)
    if signal == "insider_cluster":
        event["timestamp_predicate"] = (
            PRED_PASS
            if (
                precision == "timestamp"
                and event["dissemination_predicate"] == PRED_PASS
                and event["acceptance_reconciliation"] == PRED_PASS
                or precision == "date_only"
                and parse_date(record.get("header_filing_date"))
                == parse_date(record.get("filing_date"))
            )
            else PRED_UNVERIFIED
        )
    if signal == "pead":
        event["filing_regime"] = (
            "no_filing_regime"
            if parse_date(record["earnings_date"]) <= datetime.date(2003, 3, 28)
            else "item_12_or_9"
            if parse_date(record["earnings_date"]) < datetime.date(2004, 8, 23)
            else "item_2.02"
        )
    return event


def monthly_close_comparison(event: dict, clock: AuditClock) -> dict:
    """Compare timing-only monthly-close conventions, without simulating stock selection."""
    stated = parse_date(event.get("earnings_date") or event.get("filing_date"))
    if stated is None or event.get("quarantine"):
        return {"predicate": PRED_UNVERIFIED, "reason": "required date unavailable/quarantined"}
    result = {"audit_close": event.get("tradeable_close"), "timing_only": True}
    for name, lag in (
        ("pead_calendar_lag_monthly_close", 1),
        ("insider_filing_date_monthly_close", 0),
    ):
        known = pd.Timestamp(stated) + pd.Timedelta(days=lag)
        for period in pd.period_range(known, known + pd.Timedelta(days=62), freq="M"):
            schedule = clock.get_schedule(
                str(period.start_time.date()), str(period.end_time.date())
            )
            decision = schedule.iloc[-1]["market_close"]
            if decision.tz_localize(None) >= known:
                next_day = clock.next_trading_session(decision.date())
                fill = clock.get_schedule(str(next_day), str(next_day)).iloc[0]["market_close"]
                result[name] = {
                    "decision_close": decision.isoformat(),
                    "next_session_fill_close": fill.isoformat(),
                }
                break
    return result


def holding_diagnostics(
    event: dict, signal: str, clock: AuditClock, bar_dates: set[datetime.date] | None
) -> list[dict]:
    """Only end past cutoff censors; retain both unresolved PEAD session conventions."""
    start = parse_date(event.get("tradeable_date"))
    if start is None or event.get("quarantine"):
        return []
    windows = (
        [(63, "entry_included"), (63, "entry_excluded")]
        if signal == "pead"
        else [(126, "calendar_days"), (189, "calendar_days"), (252, "calendar_days")]
    )
    diagnostics = []
    for length, convention in windows:
        if signal == "pead":
            schedule = clock.get_schedule(str(start), str(start + datetime.timedelta(days=160)))
            offset = int(convention == "entry_excluded")
            dates = [d.date() for d in schedule.index[offset : length + offset]]
            end = dates[-1]
        else:
            end = start + datetime.timedelta(days=length)
            dates = [d.date() for d in clock.get_schedule(str(start), str(end)).index]
        observed = [d for d in dates if d <= CUTOFF]
        missing_dates = None if bar_dates is None else set(observed) - bar_dates
        missing = None if missing_dates is None else len(missing_dates)
        cause_counts = {}
        for date in missing_dates or []:
            reasons = {
                r["reason"]
                for r in event.get("coverage_cause_intervals", [])
                if parse_date(r["start"]) <= date <= parse_date(r["end"])
            }
            reason = "|".join(sorted(reasons)) if reasons else "unexplained"
            cause_counts[reason] = cause_counts.get(reason, 0) + 1
        diagnostics.append(
            {
                "window": length,
                "unit": "trading_days" if signal == "pead" else "calendar_days",
                "convention": convention,
                "window_end": str(end),
                "administrative_right_censored": end > CUTOFF,
                "observed_sessions": len(observed),
                "missing_observed_bars": missing,
                "missing_by_cause": cause_counts if missing_dates is not None else None,
                "causes": event.get("missing_coverage_causes", [])
                or (["unexplained"] if missing else []),
            }
        )
    return diagnostics


def annual_accounting(
    year: int, units: list[dict], events: list[dict], population_complete: bool, smoke: bool
) -> tuple[dict, str]:
    """A6 from enumerated units and full events, never from reconciliation samples."""
    selected_units = [u for u in units if u["year"] == year]
    selected_events = [e for e in events if e.get("year") == year and e.get("in_scope")]
    coverage = AnnualA6Coverage(
        year,
        len(selected_units),
        sum(u["resolved"] for u in selected_units),
        sum(u["retrieved"] for u in selected_units),
        len(selected_events),
        sum(bool(e.get("quarantine")) for e in selected_events),
        sum(
            e.get("tape", {}).get("bar_count") == 1
            and bool(e.get("identity_resolved"))
            and e.get("tradeable_date") is not None
            and not e.get("quarantine")
            for e in selected_events
        ),
        sum(not u["retrieved"] and not u.get("missing_reason") for u in selected_units),
        sum(
            not e.get("missing_reason") and e.get("tape", {}).get("bar_count") == 0
            for e in selected_events
        ),
    )
    bar_evidence_available = all(
        e.get("quarantine") or e.get("tape", {}).get("bar_count") is not None
        for e in selected_events
    )
    measured = bool(selected_units) and population_complete and bar_evidence_available and not smoke
    predicate = (PRED_PASS if coverage.passes_a6() else PRED_FAIL) if measured else PRED_UNVERIFIED
    result = {
        **asdict(coverage),
        "identity_ratio": coverage.identity_ratio,
        "retrieval_ratio": coverage.retrieval_ratio,
        "price_ratio": coverage.price_ratio,
        "unit_missingness_ratio": coverage.unit_missingness_ratio,
        "event_missingness_ratio": coverage.event_missingness_ratio,
        "attempted_units": sum(u["attempted"] for u in selected_units),
        "stored_units": sum(u["stored"] for u in selected_units),
        "population_complete": population_complete,
        "predicate": predicate,
        "diagnostic_only": year == 2026 or smoke,
        "observed_event_lower_bound": coverage.in_scope_events,
    }
    if not bar_evidence_available:
        result["observed_priced_event_lower_bound"] = coverage.priced_events
        result["priced_events"] = None
        result["price_ratio"] = None
    if not population_complete:
        for key in (
            "in_scope_events",
            "quarantined_events",
            "missing_events",
            "price_ratio",
            "event_missingness_ratio",
        ):
            result[key] = None
    return result, predicate


ROUTE_REQUIREMENTS = {
    "R1": {"pead": {"consensus"}},
    "R2": {"pead": {"consensus"}},
    "R3": {"pead": {"consensus"}},
    "R4": {"pead": {"timestamp"}},
    "R5": {"pead": {"tape"}, "insider_cluster": {"tape"}},
    "R6": {"pead": {"tape"}, "insider_cluster": {"tape"}},
    "R7": {"pead": {"tape"}, "insider_cluster": {"tape"}},
    "R8": {"insider_cluster": {"timestamp", "amendments", "collisions", "load_boundary", "prefix"}},
    "R9": {"insider_cluster": {"timestamp", "amendments", "collisions", "load_boundary", "prefix"}},
}


def assess_source_routes(
    signal: str, matrix: list[dict], route_evidence: list[dict], client: SecEdgarClient
) -> dict:
    """Assess only registered routes and their combinations over common complete years."""
    relevant = {
        r: mapping[signal] for r, mapping in ROUTE_REQUIREMENTS.items() if signal in mapping
    }
    assessments = []
    for route, requirements in relevant.items():
        records = [r for r in route_evidence if r.get("route") == route]
        assessment = {
            "route": route,
            "requirements": sorted(requirements),
            "predicate": PRED_UNVERIFIED,
            "years": [],
            "reason": "documentation missing or incomplete",
        }
        if len(records) == 1:
            record = records[0]
            docs = record.get("documentation", {})
            if not isinstance(docs, dict):
                docs = {}
            supported = evidence_supported(record, client)
            cited_urls = {ref["url"] for ref in record.get("source_refs", [])}
            dimensions = ("fields", "history", "point_in_time", "access")
            if route in ("R5", "R6", "R7"):
                docs = dict(
                    docs,
                    point_in_time={"assessment": "not_required_execution_prices", "basis": "A10"},
                )
            documented = {
                d: bool(
                    d == "point_in_time"
                    and route in ("R5", "R6", "R7")
                    or (
                        isinstance(docs.get(d), dict)
                        and docs[d].get("citation_url") in cited_urls
                        and parse_date(docs[d].get("retrieved_date")) is not None
                        and isinstance(docs[d].get("statement"), str)
                        and docs[d]["statement"]
                        and reviewed_fields(
                            record,
                            [f"documentation.{d}.assessment", f"documentation.{d}.statement"],
                            client,
                            semantic_linkage("routes", record),
                        )
                        and any(
                            review.get("field") == f"documentation.{d}.statement"
                            and docs[d]["statement"] in review.get("extract", "")
                            and review.get("source_ref", {}).get("url") == docs[d]["citation_url"]
                            for review in record.get("field_reviews", [])
                        )
                    )
                )
                for d in dimensions
            }
            if supported:
                values = [
                    "supports"
                    if d == "point_in_time" and route in ("R5", "R6", "R7")
                    else docs.get(d, {}).get("assessment")
                    if documented[d]
                    else None
                    for d in dimensions
                ]
                # A contradiction must be explicit; silence is never an exclusion.
                if (
                    "contradicts" in values
                    and record.get("all_permitted_representations_ruled_out") is True
                    and reviewed_fields(
                        record,
                        ["all_permitted_representations_ruled_out"],
                        client,
                        semantic_linkage("routes", record),
                    )
                ):
                    assessment.update(
                        predicate=PRED_FAIL, reason="documented exclusion", documentation=docs
                    )
                elif (
                    all(v == "supports" for v in values)
                    and isinstance(record.get("requirements"), list)
                    and all(isinstance(r, str) for r in record["requirements"])
                    and reviewed_fields(
                        record,
                        [
                            "history_start",
                            "history_end",
                            "requirements",
                            "includes_delisted_acquired",
                        ],
                        client,
                        semantic_linkage("routes", record),
                    )
                ):
                    start, end = (
                        parse_date(record.get("history_start")),
                        parse_date(record.get("history_end")),
                    )
                    covered = set(record.get("requirements", [])) & requirements
                    years = [
                        year
                        for year in range(2015 if signal == "insider_cluster" else 2002, 2026)
                        if start is not None
                        and end is not None
                        and start <= datetime.date(year, 1, 1)
                        and end >= datetime.date(year, 12, 31)
                    ]
                    if years and covered and record.get("includes_delisted_acquired") is True:
                        assessment["requirements"] = sorted(covered)
                        assessment.update(
                            predicate=PRED_PASS,
                            years=sorted(set(years)),
                            reason="four documented evidence items",
                            documentation=docs,
                        )
        assessments.append(assessment)
    annual: dict[int, bool] = {}
    combinations: dict[int, dict] = {}
    unresolved = any(r["predicate"] == PRED_UNVERIFIED for r in assessments)
    for year in range(2002 if signal == "pead" else 2015, 2026):
        blockers = {
            r["requirement"] for r in matrix if r["year"] == year and r["predicate"] != PRED_PASS
        }
        covering = {
            requirement: [
                r["route"]
                for r in assessments
                if r["predicate"] == PRED_PASS
                and year in r["years"]
                and requirement in r["requirements"]
            ]
            for requirement in blockers
        }
        paid = any(route != "R9" for routes in covering.values() for route in routes)
        annual[year] = bool(blockers) and all(covering.values()) and paid
        combinations[year] = covering
    window, _ = evaluate_eligible_window(annual)
    all_route_requirements = set().union(*relevant.values())
    unaddressed = sorted(
        {
            r["requirement"]
            for r in matrix
            if r["predicate"] == PRED_UNVERIFIED and r["requirement"] not in all_route_requirements
        }
    )
    combination_assessments = []
    years = range(2015 if signal == "insider_cluster" else 2002, 2026)
    for size in range(1, len(assessments) + 1):
        for group in route_combinations(assessments, size):
            potential = set().union(*(relevant[r["route"]] for r in group))
            possible, supported = {}, {}
            for year in years:
                blockers = {
                    r["requirement"]
                    for r in matrix
                    if r["year"] == year and r["predicate"] != PRED_PASS
                }
                possible[year] = bool(blockers) and blockers <= potential
                supported[year] = possible[year] and all(
                    any(
                        r["predicate"] == PRED_PASS
                        and year in r["years"]
                        and requirement in r["requirements"]
                        for r in group
                    )
                    for requirement in blockers
                )
            if evaluate_eligible_window(possible)[0] is None:
                continue
            group_window, _ = evaluate_eligible_window(supported)
            predicate = (
                PRED_PASS
                if group_window
                else PRED_UNVERIFIED
                if any(r["predicate"] == PRED_UNVERIFIED for r in group)
                else PRED_FAIL
            )
            combination_assessments.append(
                {
                    "routes": [r["route"] for r in group],
                    "predicate": predicate,
                    "common_window": group_window,
                    "test": "all blocking requirements over a common five-complete-year window",
                }
            )
    all_excluded = (
        bool(combination_assessments)
        and not unaddressed
        and not unresolved
        and all(r["predicate"] == PRED_FAIL for r in combination_assessments)
    )
    return {
        "routes": assessments,
        "unresolved_requirements_outside_registered_routes": unaddressed,
        "common_window": window,
        "combinations": combinations,
        "paid_path_documented": window is not None and not unresolved,
        "requirement_complete_combinations": combination_assessments,
        "assessment_complete": not unresolved,
        "all_combinations_ruled_out": all_excluded,
        "manifest_sha256": compute_file_sha256(SOURCE_MANIFEST),
    }


def resolve_signal_status(
    signal: str,
    matrix: list[dict],
    route_evidence: list[dict],
    client: SecEdgarClient,
    smoke: bool = False,
) -> str:
    """Derive statuses from complete mandatory rows and checksummed registered route evidence."""
    required = {"coverage", "timestamp", "tape"} | (
        {"amendments", "collisions", "load_boundary", "prefix"}
        if signal == "insider_cluster"
        else {"actual_eps", "consensus"}
    )
    annual = {
        year: {r["requirement"] for r in matrix if r["year"] == year} == required
        and all(r["predicate"] == PRED_PASS for r in matrix if r["year"] == year)
        for year in {r["year"] for r in matrix}
    }
    if not smoke and evaluate_eligible_window(annual)[0] is not None:
        return STATUS_FREE_VERIFIED
    assessment = assess_source_routes(signal, matrix, route_evidence, client)
    if not smoke and assessment["common_window"] and assessment["assessment_complete"]:
        return STATUS_PAID_DOCUMENTED
    if not smoke and assessment["assessment_complete"] and assessment["all_combinations_ruled_out"]:
        return STATUS_NO_VERIFIED_PATH
    return STATUS_UNRESOLVED


AMBIGUITIES = [
    "A10 says any unresolved relevant evidence leaves UNRESOLVED, while section 5 "
    "describes one complete paid route; candidate combinations are reported, but no paid "
    "status is emitted while relevant route evidence is unresolved.",
    "A6 insider cluster denominator when quarantined/missing disclosures prevent "
    "determining whether a cluster exists; no transaction-to-cluster imputation is "
    "authorized.",
    "A7/A9 do not define ambiguous amended-lot linkage or re-signalling after a correction; "
    "unresolved links block reconstruction. Distinct owner CIKs and ABSENT checkbox states "
    "are inventoried without adding independence or verified-non-plan eligibility gates.",
    "A4 does not specify whether the entry session counts toward the 63-session "
    "diagnostic; both endpoint conventions are reported and neither controls "
    "eligibility.",
    "A6 does not explicitly fix the event-year/membership anchor where stated, public and "
    "execution dates cross a year or membership boundary. Disagreeing anchors and unknown "
    "execution dates stay unresolved and cannot qualify affected years.",
    "A10 source-manifest routes do not expressly supply historical security identity, "
    "actual-EPS basis or measured A6 coverage; vendor field documentation alone cannot "
    "repair these blockers.",
]


class EventTimingAuditRunner:
    """Collect insider evidence first, then PEAD; checkpoint evidence and persist "
    "every predicate."""

    def __init__(
        self,
        smoke: bool = False,
        resume: bool = False,
        output_file: Path | None = None,
        evidence_file: Path | None = None,
        offline: bool = True,
        cache_dir: Path = CACHE_DIR,
        db: ReadOnlyDb | None = None,
        membership_file: Path = MEMBERSHIP_FILE,
        detector_file: Path | None = None,
    ):
        self.smoke, self.resume, self.offline = smoke, resume, offline
        self.output_file = Path(
            output_file
            or (
                "/tmp/event_timing_audit_smoke.json"
                if smoke
                else "/tmp/event_timing_audit_partial.json"
            )
        )
        if smoke and not self.output_file.resolve().is_relative_to(Path("/tmp")):
            raise ValueError("Incomplete smoke output must stay in /tmp")
        canonical = ROOT / "docs/research/_event_timing_audit.json"
        if self.output_file.suffix != ".json" or (
            not self.output_file.resolve().is_relative_to(Path("/tmp"))
            and self.output_file.resolve() != canonical
        ):
            raise ValueError("Audit output must be /tmp/*.json or the registered final artifact")
        self.cache_dir = Path(cache_dir)
        if smoke:
            # Smoke progress stays separate from full-run evidence.
            self.progress_file = self.cache_dir / "smoke_progress.json"
        else:
            self.progress_file = self.cache_dir / "audit_progress.json"
        self.clock = AuditClock()
        self.client = SecEdgarClient(self.cache_dir, smoke=smoke, offline=offline)
        self.db = db
        self.membership_file = Path(membership_file)
        self.query_run_date = datetime.datetime.now(datetime.timezone.utc).date()
        self.evidence: dict = {}
        if evidence_file is not None:
            self.evidence = json.loads(Path(evidence_file).read_text())
            if self.evidence.get("schema") != "event-timing-evidence-v1":
                raise ValueError("Unsupported evidence schema")
        self.input_sha256 = digest(json.dumps(self.evidence, sort_keys=True).encode())
        self.detector_sha256 = compute_file_sha256(detector_file) if detector_file else None
        self.code_sha256 = {
            name: compute_file_sha256(Path(__file__).with_name(name))
            for name in (
                "event_timing_audit.py",
                "event_timing_sources.py",
                "event_timing_reconstruction.py",
            )
        }
        self.code_sha256["index_constituents.py"] = compute_file_sha256(
            Path(load_sp500_history.__code__.co_filename)
        )
        self.approved_amendment = {
            "identity": "2026-10-22-event-timing-source-conflict-proposal",
            "path": str(APPROVED_AMENDMENT.relative_to(ROOT))
            if APPROVED_AMENDMENT.is_relative_to(ROOT)
            else str(APPROVED_AMENDMENT),
            "sha256": compute_file_sha256(APPROVED_AMENDMENT),
        }
        self.code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.last_progress_save = 0.0
        self.input_errors: list[dict] = []
        self.pending: list[str] = []
        self.cpu_progress: dict = {}
        self.stored: dict[str, list[dict]] = {"insider_cluster": [], "pead": []}
        self.stored_frozen = False
        self.stored_query_log: list[dict] = []
        self.observation_queries: list[dict] = []
        self.samples: dict[str, dict] = {}
        self.filing_frame: list[dict] = []
        self.frames_frozen = False
        if self.progress_file.exists() and not resume:
            raise ValueError("Existing audit progress requires --resume or a new scratch run")
        if resume and not self.progress_file.exists():
            raise ValueError("Resume progress missing; preserve cache and recover original inputs")
        if resume and self.progress_file.exists():
            progress = json.loads(self.progress_file.read_text())
            if progress.get("schema") != "event-timing-progress-v2" or not all(
                k in progress
                for k in (
                    "stored",
                    "stored_frozen",
                    "samples",
                    "filing_frame",
                    "frames_frozen",
                    "input_errors",
                    "cpu_progress",
                    "stored_query_log",
                    "observation_queries",
                    "code_revision",
                )
            ):
                raise ValueError("Incomplete resume observations")
            if progress.get("input_sha256") != self.input_sha256 or progress.get("smoke") != smoke:
                raise ValueError(
                    "Resume input mismatch; preserve progress and use the original evidence"
                )
            if (
                progress.get("membership_sha256") != compute_file_sha256(self.membership_file)
                or progress.get("brief_sha256") != compute_file_sha256(BRIEF)
                or progress.get("source_manifest_sha256") != compute_file_sha256(SOURCE_MANIFEST)
            ):
                raise ValueError("Resume protocol or membership changed")
            if progress.get("approved_amendment") != self.approved_amendment:
                raise ValueError("Resume approved amendment changed or missing; start a new run")
            if progress.get("detector_sha256") != self.detector_sha256:
                raise ValueError("Resume detector changed")
            if progress["code_revision"] != self.code_revision:
                raise ValueError("Resume Git revision changed; start a new run")
            if progress.get("code_sha256") != self.code_sha256:
                raise ValueError("Resume audit implementation changed; start a new run")
            self.query_run_date = datetime.date.fromisoformat(progress["query_run_date"])
            self.evidence = progress["evidence"]
            self.stored = progress["stored"]
            self.stored_frozen = progress["stored_frozen"]
            self.samples = progress["samples"]
            self.filing_frame = progress["filing_frame"]
            self.frames_frozen = progress["frames_frozen"]
            self.input_errors = progress["input_errors"]
            self.cpu_progress = progress["cpu_progress"]
            self.stored_query_log = progress["stored_query_log"]
            self.observation_queries = progress["observation_queries"]
        for key in (
            "identities",
            "inventories",
            "filings",
            "pead_fetches",
            "pead_events",
            "classifications",
            "amendments",
            "routes",
            "release_candidates",
            "coverage_causes",
        ):
            valid = []
            records = self.evidence.get(key, [])
            if not isinstance(records, list):
                records = [records]
            for index, record in enumerate(records):
                if not self.work_available(f"validation:{key}"):
                    self.input_errors.append(
                        {
                            "kind": key,
                            "record": {"remaining": records[index:]},
                            "reason": "runtime budget exhausted",
                        }
                    )
                    break
                checked = validate_evidence_record(
                    key, record, self.client, self.evidence.get("inventories", [])
                )
                if checked is not None:
                    valid.append(checked)
                else:
                    error = {
                        "kind": key,
                        "record": record,
                        "reason": "unsupported raw facts or field-specific semantic review",
                    }
                    if error not in self.input_errors:
                        self.input_errors.append(error)
            self.evidence[key] = valid
        self.artifact_flags = load_existing_artifact_flags(detector_file) if detector_file else None
        if self.artifact_flags is not None:
            try:
                for interval in self.artifact_flags["screened"]:
                    raw = json.loads(self.client.referenced_bytes(interval["source_ref"]))
                    scope = {k: interval[k] for k in ("symbol", "start", "end")}
                    flags = [
                        r
                        for r in self.artifact_flags["flags"]
                        if r["symbol"] == interval["symbol"]
                        and interval["start"] <= r["date"] <= interval["end"]
                    ]
                    raw_flags = [
                        r
                        for r in raw["flags"]
                        if r["symbol"] == interval["symbol"]
                        and interval["start"] <= r["date"] <= interval["end"]
                    ]
                    if raw.get("schema") != "event-timing-detector-flags-v1" or (
                        scope not in raw["screened"] or flags != raw_flags
                    ):
                        raise ValueError("Detector interval/flags differ from raw export")
            except (ValueError, KeyError, TypeError, OSError):
                self.artifact_flags = None

    def checkpoint(self, stage: str, force: bool = False) -> None:
        """Persist accepted evidence and completed stages before moving to the next unit."""
        if stage not in self.client.checkpoint.completed_strata:
            self.client.checkpoint.completed_strata.append(stage)
        if (
            not force
            and time.monotonic() - self.last_progress_save < 30
            and stage
            not in (
                "collection",
                "classification_frozen",
                "insider_evaluation",
                "pead_evaluation",
            )
        ):
            return
        self.last_progress_save = time.monotonic()
        atomic_json(
            self.progress_file,
            {
                "schema": "event-timing-progress-v2",
                "input_sha256": self.input_sha256,
                "detector_sha256": self.detector_sha256,
                "code_sha256": self.code_sha256,
                "code_revision": self.code_revision,
                "approved_amendment": self.approved_amendment,
                "smoke": self.smoke,
                "membership_sha256": compute_file_sha256(self.membership_file),
                "brief_sha256": compute_file_sha256(BRIEF),
                "source_manifest_sha256": compute_file_sha256(SOURCE_MANIFEST),
                "query_run_date": str(self.query_run_date),
                "evidence": self.evidence,
                "pending": self.pending,
                "input_errors": self.input_errors,
                "stored": self.stored,
                "stored_frozen": self.stored_frozen,
                "stored_query_log": self.stored_query_log,
                "observation_queries": self.observation_queries,
                "samples": self.samples,
                "filing_frame": self.filing_frame,
                "frames_frozen": self.frames_frozen,
                "cpu_progress": self.cpu_progress,
            },
        )
        self.client.save_checkpoint()

    def work_available(self, stage: str) -> bool:
        """Checkpoint unfinished CPU work at its unit boundary on deadline exhaustion."""
        if self.client.within_deadline():
            return True
        reason = f"{stage}: runtime budget exhausted"
        if reason not in self.pending:
            self.pending.append(reason)
        self.checkpoint(stage, force=True)
        return False

    def read_stored(self) -> None:
        """Read metadata and shares; do not read equity price or outcome fields."""
        if self.stored_frozen:
            return
        if self.db is None:
            self.stored_frozen = True
            self.checkpoint("stored_snapshot", force=True)
            return
        if not self.client.within_deadline():
            self.pending.append("stored metadata: runtime budget exhausted")
            return
        statements = {
            "insider_cluster": "SELECT symbol, issuer_cik, insider_cik, accession_number, "
            "transaction_date, transaction_code, shares, acquired_disposed_code, "
            "is_10b5_1_plan, filing_date FROM form4_transactions "
            "ORDER BY accession_number, insider_cik, transaction_date",
            "pead": "SELECT symbol, earnings_date, eps_estimate, eps_reported, surprise_pct "
            "FROM earnings_surprise ORDER BY symbol, earnings_date",
        }
        for signal, sql in statements.items():
            if signal in self.cpu_progress.get("stored_signals", []):
                continue
            if not self.work_available(f"stored:{signal}"):
                return
            try:
                columns, rows = self.db.execute_select(sql)
                records = [dict(zip(columns, row, strict=True)) for row in rows]
                for record in records:
                    record["stored_quarantine"] = (
                        check_transaction_quarantine(
                            record.get("transaction_date"),
                            record.get("filing_date"),
                            self.query_run_date,
                        )
                        if signal == "insider_cluster"
                        else check_earnings_quarantine(
                            record.get("earnings_date"), self.query_run_date
                        )
                    )
                self.stored[signal] = records
            except (RuntimeError, ValueError):
                self.pending.append(f"{signal}: stored metadata unavailable")
            self.cpu_progress.setdefault("stored_signals", []).append(signal)
            self.stored_query_log = list(self.db.query_log)
            self.checkpoint(f"stored:{signal}", force=True)
        self.stored_frozen = True
        self.checkpoint("stored_snapshot", force=True)

    def collect(self, denominators: Denominators) -> None:
        """Enumerate SEC inputs, freeze classification, then execute A9 samples "
        "without replacement."""
        if not self.work_available("collection"):
            self.classification_manifest = {"entries": {}, "reason": "runtime budget exhausted"}
            return
        ever = sorted(set().union(*map(set, load_sp500_history(self.membership_file)["tickers"])))
        discovery = collect_identity_candidates(self.client, ever)
        self.evidence["identity_discovery"] = discovery
        candidates = {str(r["cik"]) for r in self.evidence["identities"]}
        candidates.update(str(r["cik"]) for r in discovery["candidates"])
        candidates.update(
            str(r["issuer_cik"]) for r in self.stored["insider_cluster"] if r.get("issuer_cik")
        )
        known = {str(r["cik"]) for r in self.evidence["inventories"] if r.get("complete")}
        ordered = sorted(candidates, key=compute_deterministic_sample_hash)
        if self.smoke:
            ordered = ordered[:2]
        for cik in ordered:
            if not self.client.within_deadline():
                self.pending.append(f"inventory:{cik}: runtime budget exhausted")
                break
            if cik not in known:
                inventory = enumerate_sec(self.client, cik)
                # Failures are retained as attempted evidence, not counted as successful coverage.
                self.evidence["inventories"] = [
                    r for r in self.evidence["inventories"] if str(r["cik"]) != cik
                ] + [inventory]
                self.checkpoint(f"inventory:{cik}")
        self.freeze_classification(denominators)
        # The required filing frame is the union, including amendments and external-only records.
        frame: dict[str, dict] = {}
        for inventory in self.evidence["inventories"]:
            for filing in inventory.get("filings", []):
                if filing["form"] in ("4", "4/A"):
                    frame[filing["accession"]] = dict(filing, cik=inventory["cik"])
        for filing in self.evidence["filings"]:
            frame.setdefault(filing["accession"], filing)
        for row in self.stored["insider_cluster"]:
            frame.setdefault(
                row["accession_number"],
                {
                    "accession": row["accession_number"],
                    "filing_date": str(row.get("filing_date")),
                    "form": "unknown_stored_form",
                    "cik": str(row["issuer_cik"]),
                    "primary_document": None,
                },
            )
        if self.frames_frozen:
            frame = {r["accession"]: r for r in self.filing_frame}
        else:
            self.filing_frame = list(frame.values())
        existing = {r["accession"] for r in self.evidence["filings"] if r.get("raw_complete")}
        for year in range(2015, 2027):
            items = [
                r
                for r in frame.values()
                if parse_date(r.get("filing_date")) is not None
                and parse_date(r["filing_date"]).year == year
            ]
            sample, shortfall = sample_stratum(items, lambda r: r["accession"], 25, self.smoke)
            selection = self.samples.setdefault(
                f"insider:{year}",
                {
                    "selected_keys": [r["accession"] for r in sample],
                    "shortfall": shortfall,
                    "frame_count": len(items),
                    "frame": items,
                },
            )
            sample = [frame[key] for key in selection["selected_keys"]]
            self.checkpoint(f"selection:insider:{year}")
            for filing in sample:
                if filing["accession"] not in existing and filing.get("primary_document"):
                    collected = collect_filing(self.client, filing["cik"], filing)
                    self.evidence["filings"] = [
                        r for r in self.evidence["filings"] if r["accession"] != filing["accession"]
                    ] + [collected]
                    if collected.get("raw_complete"):
                        existing.add(filing["accession"])
                self.checkpoint(f"filing:{filing['accession']}")
        attempted = {r["accession"] for r in self.evidence["filings"]}
        # Reconstruction and A6 require all raw filings, independently of the cross-check sample.
        if not self.smoke:
            for filing in sorted(
                frame.values(), key=lambda r: compute_deterministic_sample_hash(r["accession"])
            ):
                if filing["accession"] in attempted or not filing.get("primary_document"):
                    continue
                if not self.client.within_deadline():
                    self.pending.append("remaining raw filing population: runtime budget exhausted")
                    break
                self.evidence["filings"].append(collect_filing(self.client, filing["cik"], filing))
                self.checkpoint(f"raw:{filing['accession']}")
        for year in range(2002, 2027):
            items = [
                r
                for r in self.stored["pead"]
                if parse_date(r.get("earnings_date")) is not None
                and parse_date(r["earnings_date"]).year == year
            ]
            sample, shortfall = sample_stratum(
                items, lambda r: f"{r['symbol']}|{r['earnings_date']}", 12, self.smoke
            )
            self.samples.setdefault(
                f"pead:{year}",
                {
                    "selected_keys": [f"{r['symbol']}|{r['earnings_date']}" for r in sample],
                    "shortfall": shortfall,
                    "frame_count": len(items),
                    "frame": items,
                },
            )
        self.frames_frozen = True
        self.checkpoint("selection_frames", force=True)
        self.collect_pead(denominators)
        self.reconcile_selected_pead()
        self.collect_releases()
        self.checkpoint("collection")

    def freeze_classification(self, denominators: Denominators) -> None:
        """Search registered filing types before freezing any re-fetch selection."""
        history = load_sp500_history(self.membership_file)
        ever = sorted(set().union(*map(set, history["tickers"])))
        supplied = {r["symbol"]: r for r in self.evidence["classifications"]}
        candidates: dict[str, list[dict]] = {}
        for inventory in self.evidence["inventories"]:
            if not self.client.within_deadline():
                self.pending.append("classification: runtime budget exhausted")
                break
            identities = [
                r for r in self.evidence["identities"] if str(r["cik"]) == str(inventory["cik"])
            ]
            for filing in inventory.get("filings", []):
                if not self.work_available("classification"):
                    break
                form = filing["form"]
                is_notice = form.startswith("25") or form.startswith("15")
                is_merger_candidate = form == "8-K" and "2.01" in str(filing.get("items"))
                if not (is_notice or is_merger_candidate):
                    continue
                try:
                    url, _ = filing_urls(inventory["cik"], filing)
                    raw = self.client.fetch_sec_url(url)
                    candidate = {
                        "form": form,
                        "filing_date": filing["filing_date"],
                        "accession": filing["accession"],
                        "evidence_link": url,
                        "source_ref": self.client.reference(url) if raw is not None else None,
                        "reason": "notice/merger candidate needs effective-date and "
                        "security confirmation",
                    }
                    for identity in identities:
                        candidates.setdefault(identity["symbol"], []).append(candidate)
                except ValueError:
                    self.pending.append(
                        f"classification:{filing['accession']}: invalid archive identity"
                    )
        entries = {}
        for symbol in ever:
            record = supplied.get(symbol)
            valid = (
                record is not None
                and record.get("status") in ("DELISTED", "ACQUIRED", "OTHER")
                and parse_date(record.get("effective_date")) is not None
                and record.get("security_id")
                and record.get("evidence_link")
                and record.get("evidence_form") in ("25", "25-NSE", "15", "15-12B", "15-12G", "8-K")
            )
            entries[symbol] = (
                record
                if valid
                else {
                    "symbol": symbol,
                    "status": "UNKNOWN",
                    "reason": "SEC classification evidence not verified",
                    "candidates": candidates.get(symbol, []),
                }
            )
        payload = {
            "membership_sha256": denominators.membership_sha256,
            "entries": entries,
            "input_sha256": self.input_sha256,
            "smoke": self.smoke,
        }
        path = self.cache_dir / (
            "smoke_classification_manifest.json"
            if self.smoke
            else "classification_manifest_v2.json"
        )
        if self.resume and path.exists():
            previous = json.loads(path.read_text())
            if previous != payload:
                raise ValueError(
                    "Frozen classification changed on resume; preserve and review new evidence"
                )
        else:
            atomic_json(path, payload)
        self.classification_manifest = payload
        self.classification_sha256 = compute_file_sha256(path)
        self.checkpoint("classification_frozen")

    def collect_pead(self, denominators: Denominators) -> None:
        """Execute 30/70 re-fetch strata and per-year event reconciliation without substitution."""
        entries = self.classification_manifest["entries"]
        ordered_symbols = []
        for name, statuses, quota in (
            ("delisted_acquired", {"DELISTED", "ACQUIRED"}, 30),
            ("other", {"OTHER"}, 70),
        ):
            symbols = [s for s, row in entries.items() if row["status"] in statuses]
            sample, shortfall = sample_stratum(symbols, str, quota, self.smoke)
            selection = self.samples.setdefault(
                f"pead_refetch:{name}",
                {
                    "selected_keys": sample,
                    "shortfall": shortfall,
                    "frame_count": len(symbols),
                    "classification_sha256": self.classification_sha256,
                    "frame": sorted(symbols),
                },
            )
            ordered_symbols.extend(selection["selected_keys"])
        # Yearly samples retain their own order; duplicate source requests may reuse valid cache.
        for key, sample in self.samples.items():
            if key.startswith("pead:"):
                ordered_symbols.extend(item.split("|", 1)[0] for item in sample["selected_keys"])
        if not self.smoke:
            ordered_symbols.extend(
                sorted(
                    {u["symbol"] for u in denominators.units["pead"]},
                    key=compute_deterministic_sample_hash,
                )
            )
        known = {r["symbol"] for r in self.evidence["pead_fetches"] if r.get("request_succeeded")}
        for symbol in dict.fromkeys(ordered_symbols):
            if symbol in known:
                continue
            if not self.client.within_deadline():
                self.pending.append("remaining PEAD population: runtime budget exhausted")
                break
            self.evidence["pead_fetches"].append(collect_yahoo(self.client, symbol))
            self.checkpoint(f"pead_fetch:{symbol}")

    def reconcile_selected_pead(self) -> None:
        """Compare every frozen stored key with its fresh response; stability never proves PIT."""
        fetches = {r["symbol"]: r for r in self.evidence["pead_fetches"]}
        requests_by_symbol: dict[str, list[dict]] = {}
        for request in self.client.checkpoint.request_log:
            if request["provider"] == "yf":
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(request["url"]).query)
                for symbol in query.get("symbol", []):
                    requests_by_symbol.setdefault(symbol, []).append(request)
        for name, selection in self.samples.items():
            if not name.startswith("pead:") or "reconciliation" in selection:
                continue
            observations = []
            frame = {f"{r['symbol']}|{r['earnings_date']}": r for r in selection["frame"]}
            for key in selection["selected_keys"]:
                if not self.work_available(f"pead_reconciliation:{key}"):
                    return
                original = frame[key]
                fetched = fetches.get(original["symbol"], {})
                matches = [
                    r
                    for r in fetched.get("records", [])
                    if str(r.get("earnings_date")) == str(original["earnings_date"])
                ]
                fields = {}
                outcome = (
                    "failed"
                    if not fetched.get("request_succeeded")
                    else (
                        "unmatched"
                        if not matches
                        else "ambiguous"
                        if len(matches) != 1
                        else "agreement"
                    )
                )
                if len(matches) == 1:
                    for field in ("eps_estimate", "eps_reported", "surprise_pct"):
                        old, new = (
                            numeric_value(original.get(field)),
                            numeric_value(matches[0].get(field)),
                        )
                        fields[field] = {
                            "stored": old,
                            "fresh": new,
                            "comparison": "missing"
                            if old is None or new is None
                            else "equal"
                            if old == new
                            else "changed",
                        }
                    if any(v["comparison"] == "changed" for v in fields.values()):
                        outcome = "changed"
                    elif any(v["comparison"] == "missing" for v in fields.values()):
                        outcome = "missing_fields"
                observations.append(
                    {
                        "key": key,
                        "outcome": outcome,
                        "fields": fields,
                        "source_refs": fetched.get("source_refs", []),
                        "reason": fetched.get("reason"),
                        "requests": requests_by_symbol.get(original["symbol"], []),
                        "response_complete": fetched.get("complete", False),
                        "match_basis": "symbol|earnings_date; "
                        "fiscal/issuer lineage remains UNVERIFIED",
                        "proves_pit": False,
                    }
                )
            selection["reconciliation"] = observations
            self.checkpoint(f"reconciliation:{name}")

    def collect_releases(self) -> None:
        """Acquire 8-K release candidates with evidenced issuer/fiscal matching."""
        existing = {
            r["accession"] for r in self.evidence["release_candidates"] if r.get("complete")
        }
        candidates = []
        for inventory in self.evidence["inventories"]:
            for filing in inventory.get("filings", []):
                date = parse_date(filing.get("filing_date"))
                items = str(filing.get("items", ""))
                if date is None or filing["form"] != "8-K" or date > self.query_run_date:
                    continue
                if (
                    date >= datetime.date(2004, 8, 23)
                    and "2.02" in items
                    or datetime.date(2003, 3, 28) < date < datetime.date(2004, 8, 23)
                    and re.search(r"\b(9|12)\b", items)
                ):
                    candidates.append(dict(filing, cik=inventory["cik"]))
        selected = []
        if self.smoke:
            for year in range(2002, 2027):
                selected.extend(
                    sample_stratum(
                        [r for r in candidates if parse_date(r["filing_date"]).year == year],
                        lambda r: r["accession"],
                        2,
                        True,
                    )[0]
                )
        else:
            selected = sorted(
                candidates, key=lambda r: compute_deterministic_sample_hash(r["accession"])
            )
        for filing in selected:
            if filing["accession"] in existing:
                continue
            if not self.client.within_deadline():
                self.pending.append("release candidates: runtime budget exhausted")
                break
            collected = collect_release_candidates(self.client, filing["cik"], filing)
            self.evidence["release_candidates"] = [
                r
                for r in self.evidence["release_candidates"]
                if r["accession"] != filing["accession"]
            ] + [collected]
            self.checkpoint(f"release:{filing['accession']}")

    def prepare_insider(self, *, representation: str | None = None) -> tuple[list[dict], dict]:
        """Normalize raw lots, retain unassignable anomalies and reconcile stored losses."""
        from ggTrader.lab.event_timing_reconstruction import reconcile_lots

        lots, filing_checks = [], []
        amendments = []
        if representation is None:
            self.reconstruction_amendments = amendments
        preparation_key = "insider_preparation" + (f":{representation}" if representation else "")
        if not self.work_available(preparation_key):
            return lots, {"filings": filing_checks, "lots": lots, "amendment_actions": []}
        stored = [dict(r, accession=r["accession_number"]) for r in self.stored["insider_cluster"]]
        stored_by_accession: dict[str, list[dict]] = {}
        for row in stored:
            stored_by_accession.setdefault(row["accession"], []).append(row)
        progress = self.cpu_progress.setdefault(preparation_key, {})
        for filing in self.evidence["filings"]:
            if not self.work_available("insider_preparation"):
                break
            accession = filing["accession"]
            if accession in progress:
                saved = progress[accession]
                lots.extend(saved["lots"])
                filing_checks.append(saved["check"])
                amendments.extend(saved["amendments"])
                continue
            public = event_clock(
                filing,
                "insider_cluster",
                self.clock,
                self.query_run_date,
                self.client,
                representation=representation,
                inventories=self.evidence["inventories"],
            )
            amendment_start = len(amendments)
            if filing.get("form") == "4/A":
                links = [a for a in self.evidence["amendments"] if a["accession"] == accession]
                action = (
                    dict(links[0])
                    if len(links) == 1
                    else {"accession": accession, "source_refs": [], "linkage_unverified": True}
                )
                action.update(
                    availability=public.get("availability"),
                    decision_eligibility_time=public.get("tradeable_close")
                    if public.get("precision") == "date_only"
                    else None,
                    tradeable_close=public.get("tradeable_close"),
                    filing_date=filing.get("filing_date"),
                    issuer_cik=str(filing.get("issuer_cik", filing.get("cik", ""))),
                    symbol=filing.get("symbol")
                    or next(
                        (r.get("symbol") for r in filing.get("lots", []) if r.get("symbol")), None
                    ),
                )
                if not action["symbol"] or not (
                    action["availability"] or action["decision_eligibility_time"]
                ):
                    action["linkage_unverified"] = True
                amendments.append(action)
            raw_lots = []
            for lot in filing.get("lots", []):
                if not self.work_available(f"insider_preparation:{accession}"):
                    return lots, {
                        "filings": filing_checks,
                        "lots": lots,
                        "amendment_actions": amendments,
                    }
                row = dict(lot)
                if (
                    not row.get("lot_id")
                    or not row.get("raw_lot_sha256")
                    or not row.get("owners")
                    or any(not str(owner).isdigit() for owner in row.get("owners", []))
                ):
                    row["reconstruction_issue"] = "invalid raw lot identity/fingerprint/owner"
                row["quarantine"] = public["quarantine"] or check_transaction_quarantine(
                    lot.get("transaction_date"), lot.get("filing_date"), self.query_run_date
                )
                row["availability"] = public.get("availability")
                row["tradeable_close"] = public.get("tradeable_close")
                row["decision_eligibility_time"] = (
                    public.get("tradeable_close") if public["precision"] == "date_only" else None
                )
                row["year"] = (
                    parse_date(lot.get("filing_date")).year
                    if parse_date(lot.get("filing_date"))
                    else None
                )
                if row["quarantine"]:
                    row["availability"] = None
                mapping = (
                    identity_for(
                        lot.get("symbol", ""),
                        parse_date(lot.get("filing_date")),
                        self.evidence["identities"],
                    )
                    if parse_date(lot.get("filing_date"))
                    else None
                )
                if mapping and str(mapping["cik"]).lstrip("0") != str(lot.get("issuer_cik")).lstrip(
                    "0"
                ):
                    mapping = None
                    row["identity_issue"] = "raw issuer differs from historical ticker identity"
                row["identity_resolved"] = bool(mapping)
                if mapping:
                    row["symbol"] = mapping["symbol"]
                    row["security_id"] = mapping["security_id"]
                raw_lots.append(row)
                lots.append(row)
            raw_reconciliation = reconcile_lots(raw_lots, stored_by_accession.get(accession, []))
            filing_checks.append(
                {
                    "accession": accession,
                    "year": parse_date(filing.get("filing_date")).year
                    if parse_date(filing.get("filing_date"))
                    else None,
                    "raw_complete": filing.get("raw_complete", False),
                    "clock": public,
                    "lot_reconciliation": raw_reconciliation,
                    "source_refs": filing.get("source_refs", []),
                }
            )
            progress[accession] = {
                "lots": raw_lots,
                "check": filing_checks[-1],
                "amendments": amendments[amendment_start:],
            }
            self.checkpoint(f"prepared:{accession}")
        return lots, {
            "filings": filing_checks,
            "lots": lots,
            "amendment_actions": amendments,
        }

    def prepare_pead(self) -> tuple[list[dict], list[dict]]:
        """Build the full earnings population; sample identity is kept separately from "
        "fiscal identity."""
        events, duplicates = [], []
        raw = list(self.evidence["pead_events"])
        supplied_keys = {(r["symbol"], str(r.get("earnings_date"))) for r in raw}
        for fetched in self.evidence["pead_fetches"]:
            for record in fetched.get("records", []):
                key = fetched["symbol"], str(record.get("earnings_date"))
                if key not in supplied_keys:
                    raw.append(
                        dict(
                            record,
                            symbol=fetched["symbol"],
                            fiscal_period=record.get("fiscal_period"),
                        )
                    )
                    supplied_keys.add(key)
        by_date = {(r["symbol"], str(r.get("earnings_date"))): r for r in raw}
        # Stored records stay visible even when no fresh/raw match exists.
        for row in self.stored["pead"]:
            key = row["symbol"], str(row.get("earnings_date"))
            if key not in by_date:
                raw.append(dict(row, source_refs=[], fiscal_period=None))
        receipts = {
            (str(i["cik"]).lstrip("0"), f["accession"]): f
            for i in self.evidence["inventories"]
            for f in i.get("filings", [])
        }
        headers = {r["accession"]: r for r in self.evidence["release_candidates"]}
        fingerprint = digest(json.dumps(raw, sort_keys=True, default=str).encode())
        progress = self.cpu_progress.get("pead_preparation", {})
        if progress.get("fingerprint") != fingerprint:
            progress = {"fingerprint": fingerprint, "events": [], "duplicates": [], "next": 0}
            self.cpu_progress["pead_preparation"] = progress
        events, duplicates = progress["events"], progress["duplicates"]
        seen = {(r["symbol"], r.get("fiscal_period")) for r in events}
        for index, original in enumerate(raw):
            if index < progress["next"]:
                continue
            if not self.work_available("pead_preparation"):
                break
            row = dict(original)
            scope = semantic_linkage("pead_events", row)
            # Receipt fields are machine facts; a reviewed release is a separate observation.
            for attribute in ("acceptance", "header_acceptance"):
                row.pop(attribute, None)
            receipt = receipts.get((str(row.get("issuer_cik")).lstrip("0"), row.get("accession")))
            if receipt is not None and receipt["form"] == "8-K":
                row["acceptance"] = receipt.get("acceptance")
                row["header_acceptance"] = headers.get(receipt["accession"], {}).get(
                    "header_acceptance"
                )
            if "announcement_readings" in row and not reviewed_fields(
                row, ["announcement_readings"], self.client, scope
            ):
                row.pop("announcement_readings")
            event = event_clock(row, "pead", self.clock, self.query_run_date)
            fiscal = row.get("fiscal_period")
            key = f"{row['symbol']}|{fiscal or 'UNKNOWN'}|{row.get('earnings_date')}"
            event["event_id"] = key
            if fiscal and (row["symbol"], fiscal) in seen:
                duplicates.append(
                    {"event_id": key, "reason": "duplicate fiscal event requires reconciliation"}
                )
                event["population_issue"] = "duplicate fiscal-period record"
            seen.add((row["symbol"], fiscal))
            announcement = event.get("announcement_time")
            consensus = row.get("consensus", {})
            actual = row.get("actual_eps", {})
            snapshot = None
            try:
                snapshot = self.clock.normalize_to_ny(consensus["snapshot_time"])
                before = announcement is not None and snapshot < announcement
            except (ValueError, KeyError, TypeError):
                before = False
            event["consensus_predicate"] = (
                PRED_PASS
                if (
                    before
                    and consensus.get("fiscal_period") == fiscal
                    and fiscal
                    and consensus.get("issuer_cik")
                    and str(consensus["issuer_cik"]).lstrip("0")
                    == str(row.get("issuer_cik")).lstrip("0")
                    and numeric_value(consensus.get("value")) is not None
                    and consensus.get("basis")
                    and consensus.get("currency")
                    and consensus.get("split_basis")
                    and semantic_fields_supported("eps", consensus, self.client)
                )
                else PRED_UNVERIFIED
            )
            event["actual_eps_predicate"] = (
                PRED_PASS
                if (
                    actual.get("fiscal_period") == fiscal
                    and fiscal
                    and actual.get("issuer_cik")
                    and str(actual["issuer_cik"]).lstrip("0")
                    == str(row.get("issuer_cik")).lstrip("0")
                    and actual.get("basis")
                    and actual.get("currency")
                    and actual.get("split_basis")
                    and numeric_value(actual.get("value")) is not None
                    and semantic_fields_supported("eps", actual, self.client)
                )
                else PRED_UNVERIFIED
            )
            if (
                semantic_fields_supported("eps", consensus, self.client)
                and announcement is not None
                and snapshot is not None
                and not before
            ):
                event["consensus_predicate"] = PRED_FAIL
            if row.get("eps_reported") is not None and actual.get("value") is not None:
                if numeric_value(row["eps_reported"]) != numeric_value(actual["value"]):
                    event["stored_reported_eps_mismatch"] = True
            event["fiscal_identity_verified"] = semantic_fields_supported(
                "pead_events", row, self.client
            )
            event["timestamp_predicate"] = (
                PRED_PASS
                if (
                    not event["quarantine"]
                    and (
                        event.get("filing_regime") == "no_filing_regime"
                        or announcement is not None
                        and row.get("release_matched_by_issuer_period")
                        and row.get("release_issuer_cik") == row.get("issuer_cik")
                        and row.get("release_fiscal_period") == fiscal
                        and reviewed_fields(
                            row,
                            [
                                "release_matched_by_issuer_period",
                                "release_issuer_cik",
                                "release_fiscal_period",
                                "announcement_readings",
                            ],
                            self.client,
                            scope,
                        )
                    )
                )
                else PRED_UNVERIFIED
            )
            event["input_semantics"] = {
                "estimate_missing": numeric_value(row.get("eps_estimate")) is None,
                "reported_missing": numeric_value(row.get("eps_reported")) is None,
                "surprise_field_missing": row.get("surprise_pct") is None,
                "estimate_nonpositive": numeric_value(row.get("eps_estimate")) is not None
                and numeric_value(row["eps_estimate"]) <= 0,
                "upstream_percentage_scaling": row.get("surprise_scaling", "UNVERIFIED"),
                "currency_agreement": actual.get("currency") == consensus.get("currency")
                if actual.get("currency") and consensus.get("currency")
                else None,
                "basis_agreement": actual.get("basis") == consensus.get("basis")
                if actual.get("basis") and consensus.get("basis")
                else None,
                "split_basis_agreement": actual.get("split_basis") == consensus.get("split_basis")
                if actual.get("split_basis") and consensus.get("split_basis")
                else None,
            }
            if any(
                event["input_semantics"][key] is False
                for key in ("currency_agreement", "basis_agreement", "split_basis_agreement")
            ):
                for key in ("actual_eps_predicate", "consensus_predicate"):
                    if event[key] == PRED_PASS:
                        event[key] = PRED_UNVERIFIED
                event["lineage_issue"] = "raw EPS/consensus bases are not reconciled"
            events.append(event)
            progress["next"] = index + 1
            self.checkpoint("pead_preparation")
        return events, duplicates

    def attach_population(
        self,
        signal: str,
        events: list[dict],
        units: list[dict],
        *,
        representation: str | None = None,
    ) -> list[dict]:
        """Restrict known executions at the cutoff; unknowns remain explicit "
        "unresolved candidates."""
        progress_key = f"population:{signal}" + (f":{representation}" if representation else "")
        progress = self.cpu_progress.setdefault(progress_key, {})
        all_events = events
        events = [e for e in events if e["event_id"] not in progress]
        query_start = len(self.db.query_log) if self.db is not None else 0
        by_symbol: dict[str, list[dict]] = {}
        identities_by_symbol: dict[str, list[dict]] = {}
        for identity in self.evidence["identities"]:
            identities_by_symbol.setdefault(identity["symbol"], []).append(identity)
        for unit in units:
            by_symbol.setdefault(unit["symbol"], []).append(unit)
        for event in events:
            if not self.work_available(f"population:{signal}"):
                incomplete = {
                    e["event_id"]: dict(
                        e,
                        year=e.get("year"),
                        in_scope=True,
                        scope_assignment="runtime budget exhausted",
                        affected_years=[],
                        tape={"predicate": PRED_UNVERIFIED},
                    )
                    for e in events
                }
                return [
                    progress.get(e["event_id"], incomplete.get(e["event_id"], e))
                    for e in all_events
                ]
            date = parse_date(event.get("tradeable_date"))
            source_date = parse_date(event.get("earnings_date") or event.get("filing_date"))
            event["year"] = (date or source_date).year if date or source_date else None
            possible = by_symbol.get(event["symbol"], [])
            member = (
                any(
                    parse_date(left) <= date <= parse_date(right)
                    for u in possible
                    for left, right in u["membership_intervals"]
                )
                if date
                else bool(possible)
            )
            scope_uncertain = date is None
            anchors = [
                d
                for d in (
                    date,
                    source_date,
                    parse_date(event.get("announcement_time")),
                    parse_date(event.get("acceptance_time")),
                    parse_date(event.get("availability")),
                )
                if d is not None
            ]
            source_membership = [
                any(
                    parse_date(left) <= anchor <= parse_date(right)
                    for u in possible
                    for left, right in u["membership_intervals"]
                )
                for anchor in anchors
            ]
            period_member = (
                any(parse_date(u["start"]) <= date <= parse_date(u["end"]) for u in possible)
                if date
                else bool(possible)
            )
            if (
                date is not None
                and date <= CUTOFF
                and (
                    len(set([member, period_member, *source_membership])) > 1
                    or source_date is not None
                    and source_date.year != date.year
                    and any(source_membership)
                )
            ):
                member = any([member, period_member, *source_membership])
                scope_uncertain = True
            event["in_scope"] = member and (date is None or date <= CUTOFF)
            event["scope_assignment"] = (
                "unresolved membership/event-year anchor" if scope_uncertain else "known"
            )
            event["affected_years"] = sorted({d.year for d in anchors})
            identity = (
                identity_for(event["symbol"], date, identities_by_symbol.get(event["symbol"], []))
                if date
                else None
            )
            if (
                identity
                and event.get("issuer_cik")
                and str(identity["cik"]).lstrip("0") != str(event["issuer_cik"]).lstrip("0")
            ):
                identity = None
                event["identity_issue"] = "event issuer does not match dated security identity"
            event["identity_resolved"] = bool(identity)
            event["price_symbol"] = identity["price_symbol"] if identity else None
            event["coverage_cause_intervals"] = [
                r
                for r in self.evidence.get("coverage_causes", [])
                if r.get("symbol") in (event["symbol"], event["price_symbol"])
                and r.get("reason") in ("listing_inception", "delisting_merger", "halt")
                and parse_date(r.get("start")) is not None
                and parse_date(r.get("end")) is not None
                and parse_date(r["start"]) <= parse_date(r["end"]) <= self.query_run_date
                and evidence_supported(r, self.client)
            ]
            known_causes = {
                r["reason"]
                for r in event["coverage_cause_intervals"]
                if date is not None and parse_date(r["start"]) <= date <= parse_date(r["end"])
            }
            if event.get("quarantine"):
                event["missing_reason"] = event["quarantine"]
            elif not identity:
                event["missing_reason"] = "historical_price_identity_unverified"
            else:
                event["missing_reason"] = "|".join(sorted(known_causes)) or None
            event["tape"] = {
                "predicate": PRED_UNVERIFIED,
                "reason": "bar evidence unavailable or outside execution-date scope",
            }
        for event in events:
            if not self.work_available("tape checks"):
                break
            if event["in_scope"]:
                try:
                    event["tape"] = check_tape_integrity_batch(
                        self.db, self.clock, [event], self.artifact_flags
                    )[event["event_id"]]
                except (RuntimeError, ValueError):
                    self.pending.append(f"tape:{event['event_id']}: bar evidence unavailable")
            dates = None
            if (
                self.db is not None
                and self.client.within_deadline()
                and event.get("identity_resolved")
                and event["in_scope"]
                and not event.get("quarantine")
            ):
                start = parse_date(event["tradeable_date"])
                last = min(CUTOFF, start + datetime.timedelta(days=252))
                try:
                    _, rows = self.db.execute_select(
                        "SELECT timestamp::date FROM ohlcv WHERE interval = '1d' AND symbol = %s "
                        "AND timestamp::date BETWEEN %s AND %s ORDER BY timestamp",
                        (event["price_symbol"], start, last),
                    )
                    dates = {r[0] for r in rows}
                except (RuntimeError, ValueError):
                    self.pending.append(
                        f"holding_window:{event['event_id']}: session evidence unavailable"
                    )
            if self.work_available(f"diagnostics:{signal}"):
                event["holding_diagnostics"] = holding_diagnostics(event, signal, self.clock, dates)
                event["monthly_close_timing_comparison"] = monthly_close_comparison(
                    event, self.clock
                )
                progress[event["event_id"]] = event
                if self.db is not None:
                    self.observation_queries.extend(self.db.query_log[query_start:])
                    query_start = len(self.db.query_log)
                self.checkpoint(f"population:{signal}", force=True)
        return [progress.get(e["event_id"], e) for e in all_events]

    def rejected_population_evidence(self, signal: str, year: int) -> list[dict]:
        """Rejected candidates stay in population uncertainty instead of disappearing."""
        kinds = (
            {"identities", "inventories", "filings", "amendments"}
            if signal == "insider_cluster"
            else {"identities", "pead_events", "pead_fetches", "release_candidates"}
        )
        result = []
        for error in self.input_errors:
            if error["kind"] not in kinds:
                continue
            record = error["record"]
            if not isinstance(record, dict):
                result.append(error)
                continue
            date = parse_date(record.get("filing_date") or record.get("earnings_date"))
            if date is None or (
                date.year <= year
                if signal == "insider_cluster"
                else year in (date.year, date.year + 1)
            ):
                result.append(error)
        return result

    def evaluate_signal(
        self,
        signal: str,
        denominators: Denominators,
        lots: list[dict] | None = None,
        raw_checks: dict | None = None,
        *,
        representation: str | None = None,
    ) -> dict:
        """Evaluate annual raw-source requirements from complete units and event records."""
        from ggTrader.lab.event_timing_reconstruction import (
            evaluate_prefix_invariance,
            filing_lags,
            load_boundaries,
            replay_clusters,
        )

        if (
            representation not in (None, FILING_DATE_ONLY)
            or representation
            and signal != "insider_cluster"
        ):
            raise ValueError("Unsupported audit representation")
        suffix = f":{representation}" if representation else ""
        amendments = (
            (raw_checks or {}).get("amendment_actions", [])
            if representation
            else getattr(self, "reconstruction_amendments", [])
        )
        identities_by_symbol: dict[str, list[dict]] = {}
        inventories_by_cik: dict[str, list[dict]] = {}
        fetches_by_symbol: dict[str, list[dict]] = {}
        for identity in self.evidence["identities"]:
            identities_by_symbol.setdefault(identity["symbol"], []).append(identity)
        for inventory in self.evidence["inventories"]:
            inventories_by_cik.setdefault(str(inventory["cik"]).lstrip("0"), []).append(inventory)
        for fetched in self.evidence["pead_fetches"]:
            fetches_by_symbol.setdefault(fetched["symbol"], []).append(fetched)
        unit_progress = self.cpu_progress.setdefault(f"units:{signal}", {})
        units = []
        for unit in denominators.units[signal]:
            if not self.work_available(f"units:{signal}"):
                break
            if unit["key"] not in unit_progress:
                mappings = resolve_unit(unit, identities_by_symbol.get(unit["symbol"], []))
                scoped_evidence = {
                    "inventories": [
                        i
                        for cik in {str(m["cik"]).lstrip("0") for m in mappings or []}
                        for i in inventories_by_cik.get(cik, [])
                    ],
                    "pead_fetches": fetches_by_symbol.get(unit["symbol"], []),
                }
                unit_progress[unit["key"]] = unit_inventory(unit, mappings, signal, scoped_evidence)
            units.append(dict(unit_progress[unit["key"]]))
            self.checkpoint(f"units:{signal}")
        year_range = range(2015 if signal == "insider_cluster" else 2002, 2027)
        primary_ciks = {
            year: {c for u in units if u["year"] == year for c in u["issuer_ciks"]}
            for year in year_range
        }
        all_primary_ciks = set().union(*primary_ciks.values())
        complete_by_year = {}
        extra: dict = {}
        if signal == "insider_cluster":
            lots = lots or []
            filings = {r["accession"]: r for r in self.evidence["filings"]}
            required = {
                f["accession"]: f
                for f in self.filing_frame
                if parse_date(f.get("filing_date")) is not None
                and parse_date(f["filing_date"]) <= CUTOFF
                and str(f.get("cik", "")).lstrip("0") in all_primary_ciks
            }
            raw_status = {r["accession"]: bool(r.get("raw_complete")) for r in filings.values()}
            if representation:
                clocks = {r["accession"]: r["clock"] for r in (raw_checks or {}).get("filings", [])}
                raw_status = {
                    accession: complete
                    and clocks.get(accession, {}).get("timestamp_predicate") == PRED_PASS
                    and not clocks[accession].get("quarantine")
                    for accession, complete in raw_status.items()
                }
            lot_status: dict[str, bool] = {}
            for row in lots:
                accession = row.get("accession")
                lot_status[accession] = lot_status.get(accession, True) and bool(
                    not row.get("quarantine")
                    and not row.get("reconstruction_issue")
                    and (row.get("availability") or row.get("decision_eligibility_time"))
                    and row.get("identity_resolved")
                    and row.get("plan_state") in ("PRESENT-TRUE", "PRESENT-FALSE", "ABSENT")
                )
            prefix_ready = {}
            for period in pd.period_range("2015-01", "2026-04", freq="M"):
                if not self.work_available("historical_completeness"):
                    break
                # Full replay is cumulative: carry-in and earlier amendments remain dependencies.
                needed = [
                    f
                    for f in required.values()
                    if parse_date(f["filing_date"]) <= period.end_time.date()
                ]
                prefix_ready[str(period)] = (
                    bool(units)
                    and (
                        all(
                            raw_status.get(f["accession"], False)
                            and lot_status.get(f["accession"], True)
                            for f in needed
                        )
                        and all(
                            not a.get("linkage_unverified") and a.get("source_refs")
                            for a in amendments
                            if parse_date(a.get("filing_date")) is None
                            or parse_date(a["filing_date"]) <= period.end_time.date()
                        )
                        and all(
                            u["retrieved"]
                            for u in units
                            if parse_date(u["start"]) <= period.end_time.date()
                        )
                        and not any(
                            parse_date(f.get("filing_date")) is None for f in self.filing_frame
                        )
                        and not any(r.get("year") is None for r in lots)
                        and not self.rejected_population_evidence(signal, period.year)
                    )
                    and not self.smoke
                )
            for year in year_range:
                months = [
                    str(p)
                    for p in pd.period_range(
                        f"{year}-01", f"{year}-04" if year == 2026 else f"{year}-12", freq="M"
                    )
                ]
                complete_by_year[year] = all(prefix_ready.get(month, False) for month in months)
            horizon = self.clock.normalize_to_ny("2026-05-01")
            reconstruction_lots = [
                r
                for r in lots
                if r.get("identity_resolved")
                and not r.get("reconstruction_issue")
                and str(r.get("issuer_cik", "")).lstrip("0") in all_primary_ciks
            ]
            relevant_amendments = [
                a
                for a in amendments
                if str(a.get("issuer_cik", "")).lstrip("0") in all_primary_ciks
            ]
            events, issues = replay_clusters(
                reconstruction_lots,
                relevant_amendments,
                horizon,
                self.client.within_deadline,
                self.cpu_progress.setdefault("cluster_replay" + suffix, {}),
                lambda: self.checkpoint("cluster_replay"),
            )
            issue_years = {}
            for issue in issues:
                identifier = issue.rsplit(": ", 1)[-1]
                years = {r.get("year") for r in lots if identifier in (r["lot_id"], r["accession"])}
                years.update(
                    parse_date(f["filing_date"]).year
                    for f in self.evidence["filings"]
                    if f["accession"] == identifier and parse_date(f.get("filing_date")) is not None
                )
                issue_years[issue] = sorted(y for y in years if y is not None)
                if not years:
                    for year in complete_by_year:
                        complete_by_year[year] = False
                for year in years:
                    if year in complete_by_year:
                        complete_by_year[year] = False
            lot_lookup = {r["lot_id"]: r for r in lots}
            for event in events:
                event["filing_date"] = max(
                    str(lot_lookup[k]["filing_date"]) for k in event["lot_ids"]
                )
                if event.get("tradeable_close"):
                    event["tradeable_date"] = str(pd.Timestamp(event["tradeable_close"]).date())
                elif event.get("availability"):
                    date, close, _ = self.clock.tradeable_close_for_timestamp(event["availability"])
                    event.update(tradeable_date=str(date), tradeable_close=close.isoformat())
            for month in prefix_ready:
                year = int(month[:4])
                if any(not years or min(years) <= year for years in issue_years.values()):
                    prefix_ready[month] = False
                    complete_by_year[year] = False
            prefix = evaluate_prefix_invariance(
                reconstruction_lots,
                relevant_amendments,
                self.clock,
                prefix_ready,
                self.client.within_deadline,
                self.cpu_progress.setdefault("prefix" + suffix, {}),
                lambda: self.checkpoint("prefix"),
            )
            extra.update(
                raw_reconstruction=raw_checks,
                filing_lag_diagnostics=filing_lags(
                    lots,
                    self.clock,
                    self.client.within_deadline,
                    self.cpu_progress.setdefault("filing_lags" + suffix, {}),
                    lambda: self.checkpoint("filing_lags"),
                ),
                load_boundary_diagnostics=load_boundaries(
                    lots,
                    self.clock,
                    self.client.within_deadline,
                    self.cpu_progress.setdefault("boundaries" + suffix, {}),
                    lambda: self.checkpoint("boundaries"),
                ),
                plan_field_inventory={
                    year: {
                        state: sum(
                            r.get("year") == year and r.get("plan_state") == state for r in lots
                        )
                        for state in ("PRESENT-TRUE", "PRESENT-FALSE", "ABSENT", "UNPARSEABLE")
                    }
                    for year in year_range
                },
                reconstruction_issues=issues,
                reconstruction_issue_years=issue_years,
                prefix=prefix,
                quarantined_lots=[r for r in lots if r.get("quarantine")],
                unassignable_lots=[r for r in lots if r.get("year") is None],
            )
        else:
            events, duplicate_events = self.prepare_pead()
            events = self.attach_population(signal, events, units, representation=representation)
            for year in year_range:
                year_units = [u for u in units if u["year"] == year]
                year_events = [
                    e
                    for e in events
                    if parse_date(e.get("earnings_date")) is not None
                    and parse_date(e["earnings_date"]).year == year
                    and e["in_scope"]
                ]
                # Current snapshots do not prove historical event identity or a complete census.
                complete_by_year[year] = (
                    bool(year_units)
                    and not any(parse_date(e.get("earnings_date")) is None for e in events)
                    and all(u["retrieved"] for u in year_units)
                    and all(
                        any(
                            f["symbol"] == u["symbol"] and f.get("complete")
                            for f in self.evidence["pead_fetches"]
                        )
                        for u in year_units
                    )
                    and all(
                        e.get("fiscal_identity_verified") and not e.get("population_issue")
                        for e in year_events
                    )
                )
            extra.update(duplicate_events=duplicate_events)
        if signal == "insider_cluster":
            events = self.attach_population(signal, events, units, representation=representation)
        stored_periods = set()
        date_field = "filing_date" if signal == "insider_cluster" else "earnings_date"
        for row in self.stored[signal]:
            if not self.work_available(f"stored_periods:{signal}"):
                break
            date = parse_date(row.get(date_field))
            if date is not None:
                period = str(pd.Period(date, freq="M" if signal == "insider_cluster" else "Q"))
                stored_periods.add((row["symbol"], period))
        for unit in units:
            unit["stored"] = (unit["symbol"], unit["period"]) in stored_periods
        matrix, annual = [], {}
        requirements = ["coverage", "timestamp", "tape"] + (
            ["amendments", "collisions", "load_boundary", "prefix"]
            if signal == "insider_cluster"
            else ["actual_eps", "consensus"]
        )
        for year in year_range:
            year_events = [e for e in events if e.get("year") == year and e.get("in_scope")]
            expected_periods = (
                {f"{year}Q{q}" for q in range(1, 3 if year == 2026 else 5)}
                if signal == "pead"
                else {f"{year}-{m:02}" for m in range(1, 5 if year == 2026 else 13)}
            )
            present_periods = {u["period"] for u in units if u["year"] == year}
            missing_periods = sorted(expected_periods - present_periods)
            complete = (
                complete_by_year[year]
                and not missing_periods
                and not any(
                    e.get("scope_assignment") != "known"
                    and (year in e.get("affected_years", []) or e.get("year") is None)
                    for e in events
                    if e.get("in_scope")
                )
                and not self.rejected_population_evidence(signal, year)
                and self.client.within_deadline()
            )
            counts, coverage_pred = annual_accounting(year, units, events, complete, self.smoke)
            counts["missing_denominator_periods"] = missing_periods
            counts["rejected_population_evidence"] = self.rejected_population_evidence(signal, year)
            if missing_periods:
                counts["observed_denominator_lower_bound"] = counts["denominator_units"]
                counts["denominator_units"] = None
            annual[year] = counts
            for requirement in requirements:
                details: dict = {}
                if requirement == "coverage":
                    predicate = coverage_pred
                    details = counts
                elif requirement == "tape":
                    values = [e["tape"]["predicate"] for e in year_events]
                    predicate = aggregate_predicates(values, complete, proven_empty=not year_events)
                    details = {"event_ids": [e["event_id"] for e in year_events]}
                elif signal == "pead":
                    values = [
                        e.get(f"{requirement}_predicate", PRED_UNVERIFIED) for e in year_events
                    ]
                    predicate = aggregate_predicates(values, complete, proven_empty=not year_events)
                    details = {"event_ids": [e["event_id"] for e in year_events]}
                elif requirement == "prefix":
                    predicate = prefix["annual"].get(year, PRED_UNVERIFIED)
                else:
                    year_filings = [
                        r for r in (raw_checks or {}).get("filings", []) if r["year"] == year
                    ]
                    if requirement == "timestamp":
                        sampled = self.samples.get(f"insider:{year}", {}).get("selected_keys", [])
                        verified = all(
                            any(
                                r["accession"] == key
                                and r["clock"].get("timestamp_predicate") == PRED_PASS
                                and not r["clock"].get("quarantine")
                                for r in year_filings
                            )
                            for key in sampled
                        ) and all(
                            r["clock"].get("timestamp_predicate") == PRED_PASS for r in year_filings
                        )
                        predicate = PRED_PASS if complete and verified else PRED_UNVERIFIED
                        details["representations"] = [
                            r["clock"].get("precision") for r in year_filings
                        ]
                        details["dissemination"] = [
                            r["clock"].get("dissemination_predicate", PRED_UNVERIFIED)
                            for r in year_filings
                        ]
                    elif requirement == "amendments":
                        unresolved = [
                            i for i in issues if "amendment" in i and year in issue_years.get(i, [])
                        ]
                        predicate = PRED_PASS if complete and not unresolved else PRED_UNVERIFIED
                        details["unresolved"] = unresolved
                    elif requirement == "collisions":
                        predicate = aggregate_predicates(
                            [r["lot_reconciliation"]["predicate"] for r in year_filings],
                            complete,
                            proven_empty=not year_filings,
                        )
                        details["stored_diagnostics"] = [
                            r["lot_reconciliation"] for r in year_filings
                        ]
                    else:
                        predicate = PRED_PASS if complete else PRED_UNVERIFIED
                        details = {
                            "raw_public_time_inclusion": True,
                            "monthly_boundaries": [
                                r
                                for r in extra["load_boundary_diagnostics"]
                                if r["decision"].startswith(str(year))
                            ],
                        }
                if self.smoke and predicate == PRED_PASS:
                    predicate = PRED_UNVERIFIED
                matrix.append(
                    {
                        "signal": signal,
                        "year": year,
                        "requirement": requirement,
                        "representation_assessed": representation or "raw-source reconstruction",
                        "inputs": {"unit_keys": [u["key"] for u in units if u["year"] == year]},
                        "test": requirement,
                        "predicate": predicate,
                        "details": details,
                    }
                )
        sources = assess_source_routes(signal, matrix, self.evidence["routes"], self.client)
        annual_pass = {
            y: all(r["predicate"] == PRED_PASS for r in matrix if r["year"] == y)
            for y in year_range
        }
        window, length = evaluate_eligible_window(annual_pass)
        return {
            **(
                {
                    "representation": representation,
                    "timing_pass_basis": "declared date-only convention compliance",
                    "dissemination_predicate": PRED_UNVERIFIED,
                }
                if representation
                else {}
            ),
            "status": resolve_signal_status(
                signal, matrix, self.evidence["routes"], self.client, self.smoke
            ),
            "annual_coverage": annual,
            "requirement_matrix": matrix,
            "eligible_window": window,
            "eligible_years": length,
            "units": units,
            "events": events,
            "source_search": sources,
            **extra,
        }

    def run(self) -> dict:
        """Write partial JSON/Markdown off the canonical path; no missing input "
        "becomes measured PASS."""
        denominators = build_universe_denominators(self.membership_file)
        self.read_stored()
        self.collect(denominators)
        lots, checks = self.prepare_insider()
        insider = self.evaluate_signal("insider_cluster", denominators, lots, checks)
        self.checkpoint("insider_evaluation")
        pead = self.evaluate_signal("pead", denominators)
        self.checkpoint("pead_evaluation")
        date_lots, date_checks = self.prepare_insider(representation=FILING_DATE_ONLY)
        date_only = self.evaluate_signal(
            "insider_cluster", denominators, date_lots, date_checks, representation=FILING_DATE_ONLY
        )
        self.checkpoint(FILING_DATE_ONLY + "_evaluation", force=True)
        results = {
            "audit_meta": {
                "mode": "smoke" if self.smoke else "offline" if self.offline else "full",
                "complete": False,
                "query_run_date": str(self.query_run_date),
                "membership_sha256": denominators.membership_sha256,
                "brief_sha256": compute_file_sha256(BRIEF),
                "source_manifest_sha256": compute_file_sha256(SOURCE_MANIFEST),
                "input_sha256": self.input_sha256,
                "detector_sha256": self.detector_sha256,
                "code_sha256": self.code_sha256,
                "code_revision": self.code_revision,
                "approved_amendment": self.approved_amendment,
                "calendar": "pandas_market_calendars/NYSE",
                "sec_requests_used": self.client.checkpoint.sec_requests_total,
                "yf_requests_used": self.client.checkpoint.yf_requests_total,
                "elapsed_seconds": self.client.checkpoint.elapsed_seconds,
                "external_stop": self.client.checkpoint.stopped_reason,
            },
            "insider_cluster": insider,
            FILING_DATE_ONLY: date_only,
            "pead": pead,
            "samples": self.samples,
            "classification": self.classification_manifest,
            "evidence": self.evidence,
            "input_errors": self.input_errors,
            "pending": self.pending,
            "ambiguities": AMBIGUITIES,
            "db_queries": self.stored_query_log + self.observation_queries,
            "stored_records": self.stored,
            "filing_frame": self.filing_frame,
            "unassignable_stored_records": {
                signal: [
                    r
                    for r in self.stored[signal]
                    if parse_date(
                        r.get("filing_date" if signal == "insider_cluster" else "earnings_date")
                    )
                    is None
                ]
                for signal in self.stored
            },
            "network_accounting": asdict(self.client.checkpoint),
            "phase2_build_requirements": [
                "Stored parser omits 4/A and all but first owner; preserve raw lots "
                "and public-time amendment links.",
                "Stored PK includes transaction code but can collide on different "
                "security classes/lots; no database corrections made.",
                "ABSENT plan fields were stored as False; do not treat False as verified non-plan.",
                "Stored load boundary uses transaction_date; include pre-boundary "
                "transactions disclosed later.",
                "Stored monthly clustering filters future disclosures after cluster "
                "formation; compare with public-time replay before any Phase 2 "
                "implementation.",
                "Stored earnings DATE loses intraday precision; calendar-day "
                "lag/monthly close differs from A1 close timing.",
            ],
            "historical_exposure": {
                "sources": [
                    "docs/research/2026-07-25-strategy-implementation-audit.md",
                    "docs/research/briefs/2026-10-20-momentum-retest.md",
                ],
                "transaction_date_lookahead": "documented stored implementation exposure",
                "signal_sweep_membership_fix": "weight strategies: demonstrated "
                "applicability not established",
                "spy_duplicate_exposure_to_original_runs": "UNVERIFIED",
                "bar_shift_exposure_to_original_runs": "run-vintage exposure "
                "unresolved; metadata recheck only",
                "documented_contaminated_tickers_2010_2020": 72,
                "tape_2021_2026_04": "detector-screened, not certified",
                "prior_outcomes": "Existing NO-GO reports remain historical; no "
                "outcomes recalculated or inspected.",
            },
        }
        self.work_available("final_evaluation")
        results["audit_meta"]["complete"] = (
            not self.smoke
            and not self.offline
            and self.client.within_deadline()
            and not any("runtime budget exhausted" in reason for reason in self.pending)
        )
        results["audit_meta"]["work_completion_basis"] = (
            "bounded protocol attempted and both signals evaluated; "
            "truth and source search are separate"
        )
        results["audit_meta"]["source_search_complete"] = all(
            signal["source_search"]["assessment_complete"] for signal in (insider, pead)
        )
        results["audit_meta"]["stopping_reasons"] = list(
            dict.fromkeys(
                self.pending
                + (
                    [self.client.checkpoint.stopped_reason]
                    if self.client.checkpoint.stopped_reason
                    else []
                )
            )
        )
        canonical = ROOT / "docs/research/_event_timing_audit.json"
        if self.output_file.resolve() == canonical and not results["audit_meta"]["complete"]:
            raise ValueError(
                "Smoke/offline/interrupted work cannot publish the canonical audit artifact"
            )
        atomic_json(self.output_file, results)
        report_file = (
            ROOT / "docs/research/2026-10-21-event-timing-audit-report.md"
            if self.output_file.resolve() == canonical
            else self.output_file.with_suffix(".md")
        )
        atomic_bytes(report_file, render_report(results).encode())
        return results


def aggregate_predicates(values: list[str], complete: bool, proven_empty: bool = False) -> str:
    """A demonstrated failure survives incomplete evidence; absence of a test cannot PASS."""
    if PRED_FAIL in values:
        return PRED_FAIL
    return (
        PRED_PASS
        if complete and (set(values) == {PRED_PASS} or not values and proven_empty)
        else PRED_UNVERIFIED
    )


def render_report(results: dict) -> str:
    """Concise evidence report with immutable event identities in the companion JSON."""
    lines = [
        "# Event timing audit — " + results["audit_meta"]["mode"],
        "",
        "Completed bounded audit; unknown requirements and unfinished source searches "
        "remain explicit."
        if results["audit_meta"]["complete"]
        else "Partial evidence; no completed audit claim.",
        "",
    ]
    for name in ("insider_cluster", FILING_DATE_ONLY, "pead"):
        if name not in results:
            continue
        signal = results[name]
        lines.extend(
            [
                f"## {name}: {signal['status']}",
                "",
                f"Eligible window: {signal['eligible_window']}; "
                f"{signal['eligible_years']} complete years.",
                "",
                "| Year | Requirement | Predicate |",
                "|---|---|---|",
            ]
        )
        lines.extend(
            f"| {r['year']} | {r['requirement']} | {r['predicate']} |"
            for r in signal["requirement_matrix"]
        )
        lines.append("")
        if name == FILING_DATE_ONLY:
            lines.extend(
                [
                    "Timing PASS means declared date-only convention compliance; "
                    "historical dissemination and its bound remain UNVERIFIED. "
                    "Original receipt conflicts and disposition are retained above.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Ambiguities",
            "",
            *[f"- {a}" for a in results["ambiguities"]],
            "",
            "## Phase 2 stored implementation requirements",
            "",
            *[f"- {a}" for a in results["phase2_build_requirements"]],
            "",
            "No returns, price changes, event price paths, CAR or P&L computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Expose one CLI for collection and evidence replay."""
    parser = argparse.ArgumentParser(
        description="Read-only event timing evidence audit (offline by default)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="At most two items per stratum; output in /tmp"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume frozen inputs and cumulative budgets"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="/tmp partial JSON, or docs/research/_event_timing_audit.json for completed "
        "bounded --network work (UNRESOLVED allowed); Markdown is written alongside",
    )
    parser.add_argument(
        "--evidence", type=Path, help="Checksummed event-timing-evidence-v1 input JSON"
    )
    parser.add_argument(
        "--detector-flags", type=Path, help="Existing counts/flags-only detector evidence export"
    )
    parser.add_argument(
        "--network", action="store_true", help="Enable registered sources after code review"
    )
    parser.add_argument(
        "--read-db",
        action="store_true",
        help="Explicit read-only DB metadata queries; exported settings only",
    )
    return parser


def main() -> None:
    """Run the audit CLI without reading .env or silently enabling a database/network connection."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    runner = EventTimingAuditRunner(
        smoke=args.smoke,
        resume=args.resume,
        output_file=args.output,
        evidence_file=args.evidence,
        offline=not args.network,
        db=ReadOnlyDb() if args.read_db else None,
        detector_file=args.detector_flags,
    )
    results = runner.run()
    print(f"Evidence artifact: {runner.output_file}")
    print(f"Insider: {results['insider_cluster']['status']}; PEAD: {results['pead']['status']}")


if __name__ == "__main__":
    main()
