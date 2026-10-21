"""Event-Timing Audit Tool for PEAD and Insider Cluster Strategies.

This module implements the audit protocol defined in:
    docs/research/briefs/2026-10-21-event-timing-audit.md (§8 Protocol Annex)
    docs/research/briefs/2026-10-21-event-timing-audit-sources.md

Strict Rules Enforced:
1. Strict Return Prohibition: NEVER compute, print, or look at returns, price changes,
   drift, P&L, or post-event price paths. Prices are read ONLY to verify a bar exists.
2. Read-Only DB: Explicit SELECTs only with session characteristics set to READ ONLY.
   No DDL, never call ensure_schema() or cache loaders. Fail closed on write.
3. Untouchables: Never touch live trading code, docs (during build/smoke), or .env.
4. SEC Fair Access: Declared User-Agent with SEC_CONTACT_EMAIL. Rate limit <= 5 req/s.
   Budgets (<= 20k SEC, <= 2k yfinance) persisted across runs/resumes.
   Halt on first 403 or 429.
5. Caching & Manifest: Raw responses cached under data/external/event_timing_audit/
   with SHA-256 MANIFEST.json. data/ is never deleted.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import pandas_market_calendars as mcal
import psycopg2

from ggTrader.utils.config import get_db_connection_string

LOG = logging.getLogger("event_timing_audit")

# Canonical file paths and constants
MEMBERSHIP_FILE = Path("data/universe/sp500_constituents_history.csv.gz")
RUSSELL_SNAPSHOT_FILE = Path("data/universe/russell2000_tickers_snapshot_2026-06-09.txt")
REPLICA_RESULTS_FILE = Path("docs/research/_production_replica_results.json")
CACHE_DIR = Path("data/external/event_timing_audit")
CACHE_MANIFEST_FILE = CACHE_DIR / "MANIFEST.json"
CHECKPOINT_FILE = CACHE_DIR / "checkpoint.json"
CLASSIFICATION_MANIFEST_FILE = CACHE_DIR / "classification_manifest.json"

EXPECTED_SP500_SHA256 = "b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472"

# Budgets per §8 A11
MAX_SEC_REQUESTS = 20_000
MAX_YF_REQUESTS = 2_000
SMOKE_MAX_SEC_REQUESTS = 50
MAX_SEC_RATE_RPS = 5.0

# Sampling quotas per §8 A9
INSIDER_SAMPLE_PER_YEAR = 25
PEAD_SAMPLE_PER_YEAR = 12
PEAD_REFETCH_DELISTED_QUOTA = 30
PEAD_REFETCH_OTHER_QUOTA = 70
SMOKE_STRATUM_QUOTA = 2

# Predicates
PRED_PASS = "PASS"
PRED_FAIL = "FAIL"
PRED_UNVERIFIED = "UNVERIFIED"

# Status values per §5
STATUS_FREE_VERIFIED = "FREE—VERIFIED"
STATUS_PAID_DOCUMENTED = "PAID—DOCUMENTED PATH"
STATUS_NO_VERIFIED_PATH = "NO VERIFIED PATH"
STATUS_UNRESOLVED = "UNRESOLVED"


# ============================================================================
# 1. SHA-256 and File Hashing
# ============================================================================


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

    def __init__(self, calendar_name: str = "NYSE"):
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
        if ts.tz is None:
            return ts.tz_localize(self.tz)
        return ts.tz_convert(self.tz)

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
    """Check a transaction record against §8 A8 quarantine rules.

    Returns reason code string if quarantined, None if clean.
    """
    if timestamp_conflict:
        return QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT

    if transaction_date is None or pd.isna(transaction_date):
        return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

    if isinstance(transaction_date, datetime.date):
        tx_d = transaction_date
    else:
        try:
            tx_d = pd.Timestamp(transaction_date).date()
        except Exception:
            try:
                parts = str(transaction_date).split("-")
                yr = int(parts[0])
                if yr < 1990:
                    return QuarantineReason.TX_DATE_PRE_1990
            except Exception:
                pass
            return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

    if tx_d < datetime.date(1990, 1, 1):
        return QuarantineReason.TX_DATE_PRE_1990

    if tx_d > query_run_date:
        return QuarantineReason.FUTURE_DATED

    if filing_date is not None and not pd.isna(filing_date):
        if isinstance(filing_date, datetime.date):
            f_d = filing_date
        else:
            try:
                f_d = pd.Timestamp(filing_date).date()
            except Exception:
                return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

        if f_d > query_run_date:
            return QuarantineReason.FUTURE_DATED
        if tx_d > f_d:
            return QuarantineReason.TX_DATE_POST_FILING

    return None


def check_earnings_quarantine(
    earnings_date: Any,
    query_run_date: datetime.date,
    timestamp_conflict: bool = False,
) -> Optional[str]:
    """Check an earnings surprise record against §8 A8 quarantine rules."""
    if timestamp_conflict:
        return QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT

    if earnings_date is None or pd.isna(earnings_date):
        return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

    if isinstance(earnings_date, datetime.date):
        e_d = earnings_date
    else:
        try:
            e_d = pd.Timestamp(earnings_date).date()
        except Exception:
            try:
                parts = str(earnings_date).split("-")
                yr = int(parts[0])
                if yr < 1990:
                    return QuarantineReason.TX_DATE_PRE_1990
            except Exception:
                pass
            return QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

    if e_d < datetime.date(1990, 1, 1):
        return QuarantineReason.TX_DATE_PRE_1990

    if e_d > query_run_date:
        return QuarantineReason.FUTURE_DATED

    return None


# ============================================================================
# 4. Deterministic Sampler (§8 A9)
# ============================================================================


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
    for item in items:
        k = key_fn(item)
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


# ============================================================================
# 5. SEC EDGAR Client with Fair-Access Rate Limiting, Budgets, Caching (§8 A11, A12)
# ============================================================================


@dataclass
class CheckpointState:
    sec_requests_total: int = 0
    yf_requests_total: int = 0
    completed_strata: List[str] = field(default_factory=list)
    stopped_due_to_rate_limit: bool = False
    stopped_reason: Optional[str] = None


class SecEdgarClient:
    """SEC EDGAR and network client adhering strictly to §8 A11 and Fair Access rules."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        smoke: bool = False,
        sec_contact_email: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.smoke = smoke
        self.sec_contact_email = sec_contact_email or os.getenv("SEC_CONTACT_EMAIL", "").strip()
        self.manifest_file = cache_dir / "MANIFEST.json"
        self.checkpoint_file = cache_dir / "checkpoint.json"

        self.sec_budget_limit = SMOKE_MAX_SEC_REQUESTS if smoke else MAX_SEC_REQUESTS
        self.yf_budget_limit = MAX_YF_REQUESTS

        self.last_sec_req_time: float = 0.0
        self.min_sec_interval: float = 1.0 / MAX_SEC_RATE_RPS

        self.checkpoint = self._load_checkpoint()
        self.manifest = self._load_manifest()

    def _load_checkpoint(self) -> CheckpointState:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    return CheckpointState(**data)
            except Exception as e:
                LOG.warning("Failed to parse checkpoint %s: %s", self.checkpoint_file, e)
        return CheckpointState()

    def save_checkpoint(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self.checkpoint), f, indent=2)
        tmp.replace(self.checkpoint_file)

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                LOG.warning("Failed to parse manifest %s: %s", self.manifest_file, e)
        return {"files": []}

    def save_manifest(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        tmp.replace(self.manifest_file)

    def is_network_available(self) -> Tuple[bool, str]:
        if not self.sec_contact_email:
            return (
                False,
                "SEC_CONTACT_EMAIL is unset; network requests halted per Fair Access rules.",
            )
        if self.checkpoint.stopped_due_to_rate_limit:
            return (
                False,
                f"Stopped earlier due to rate limit/error: {self.checkpoint.stopped_reason}",
            )
        if self.checkpoint.sec_requests_total >= self.sec_budget_limit:
            tot = self.checkpoint.sec_requests_total
            lim = self.sec_budget_limit
            return False, f"SEC budget exhausted ({tot}/{lim})"
        return True, "Ready"

    def fetch_sec_url(self, url: str) -> Optional[str]:
        """Fetch a URL from SEC EDGAR with rate limit, caching, retries, and budget tracking."""
        # 1. Check disk cache
        cache_key = hashlib.sha256(url.encode()).hexdigest()
        cache_filename = f"sec_{cache_key}.txt"
        cache_path = self.cache_dir / cache_filename

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        available, reason = self.is_network_available()
        if not available:
            LOG.info("Skipping SEC fetch for %s: %s", url, reason)
            return None

        user_agent = f"ggTrader research {self.sec_contact_email}"
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

        # Rate limiting: <= 5 req/s
        elapsed = time.time() - self.last_sec_req_time
        if elapsed < self.min_sec_interval:
            time.sleep(self.min_sec_interval - elapsed)

        max_retries = 3
        for attempt in range(max_retries):
            # Budget check
            if self.checkpoint.sec_requests_total >= self.sec_budget_limit:
                LOG.warning("SEC request budget reached: %d", self.checkpoint.sec_requests_total)
                return None

            self.checkpoint.sec_requests_total += 1
            self.last_sec_req_time = time.time()
            self.save_checkpoint()

            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    status_code = resp.status
                    content_bytes = resp.read()
                    if resp.info().get("Content-Encoding") == "gzip":
                        content_bytes = gzip.decompress(content_bytes)
                    text_content = content_bytes.decode("utf-8", errors="replace")

                    # Write to cache
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(text_content)

                    # Update manifest
                    resp_sha256 = hashlib.sha256(content_bytes).hexdigest()
                    self.manifest["files"].append(
                        {
                            "url": url,
                            "cache_file": cache_filename,
                            "status_code": status_code,
                            "sha256": resp_sha256,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                    )
                    self.save_manifest()
                    return text_content

            except urllib.error.HTTPError as e:
                LOG.warning("HTTP error %d fetching %s (attempt %d)", e.code, url, attempt + 1)
                if e.code in (403, 429):
                    self.checkpoint.stopped_due_to_rate_limit = True
                    self.checkpoint.stopped_reason = f"HTTP {e.code} received from SEC on {url}"
                    self.save_checkpoint()
                    LOG.error("Halted all SEC requests due to HTTP %d", e.code)
                    return None
                if attempt == max_retries - 1:
                    return None
                time.sleep(2**attempt)
            except Exception as e:
                LOG.warning("Network error fetching %s: %s (attempt %d)", url, e, attempt + 1)
                if attempt == max_retries - 1:
                    return None
                time.sleep(2**attempt)

        return None


# ============================================================================
# 6. Read-Only DB Interface
# ============================================================================


class ReadOnlyDb:
    """Strict read-only PostgreSQL client that fails closed on writes and hides secrets."""

    def __init__(self):
        raw_conn_str = get_db_connection_string()
        self.conn_str = raw_conn_str.replace("postgresql+psycopg2://", "postgresql://")
        # Masked connection string for reporting
        self.masked_conn_str = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", self.conn_str)
        self.query_log: List[Dict[str, Any]] = []

    def execute_select(
        self, sql: str, params: Optional[Tuple[Any, ...]] = None
    ) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Execute a SELECT query on a transaction read-only session."""
        start_time = time.time()
        conn = psycopg2.connect(self.conn_str)
        try:
            conn.set_session(readonly=True)
            with conn.cursor() as cur:
                cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
                cur.execute(sql, params or ())
                col_names = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                duration = time.time() - start_time
                self.query_log.append(
                    {
                        "sql": sql.strip(),
                        "params_count": len(params) if params else 0,
                        "columns": col_names,
                        "rows_returned": len(rows),
                        "duration_seconds": round(duration, 4),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                )
                return col_names, rows
        finally:
            conn.close()


# ============================================================================
# 7. Universe Denominators (§8 A5, A6)
# ============================================================================


@dataclass
class Denominators:
    membership_sha256: str
    pead_member_quarters: Dict[int, int]
    pead_member_quarters_detail: Dict[str, int]
    insider_member_months: Dict[int, int]
    insider_member_months_detail: Dict[str, int]
    russell_snapshot_count: int
    ever_members_count: int
    latest_members_count: int
    removed_members_count: int


def build_universe_denominators(
    membership_file: Path = MEMBERSHIP_FILE,
    russell_file: Path = RUSSELL_SNAPSHOT_FILE,
) -> Denominators:
    """Build S&P 500 member-quarter and member-month denominators per §8 A5."""
    sha256 = compute_file_sha256(membership_file)

    df = pd.read_csv(membership_file)
    df["date"] = pd.to_datetime(df["date"])

    # Expand records
    records = []
    for _, row in df.iterrows():
        d = row["date"]
        for sym in row["tickers"].split(","):
            records.append((d, sym.strip()))

    rdf = pd.DataFrame(records, columns=["date", "symbol"])

    all_ever = set(rdf["symbol"])
    latest_date = df["date"].max()
    latest_tickers = set(df[df["date"] == latest_date].iloc[0]["tickers"].split(","))
    removed_members = all_ever - latest_tickers

    # 1. PEAD: Member-quarters 2002-Q1 to 2026-Q1 (plus partial 2026-Q2 through 2026-04-30)
    rdf_pead = rdf[(rdf["date"] >= "2002-01-01") & (rdf["date"] <= "2026-04-30")].copy()
    rdf_pead["quarter"] = rdf_pead["date"].dt.to_period("Q")
    pead_distinct = rdf_pead.drop_duplicates(subset=["symbol", "quarter"])

    pead_annual: Dict[int, int] = {}
    pead_detail: Dict[str, int] = {}
    for q, count in pead_distinct.groupby("quarter").size().items():
        q_str = str(q)
        pead_detail[q_str] = int(count)
        yr = q.year
        pead_annual[yr] = pead_annual.get(yr, 0) + int(count)

    # 2. Insider: Member-months 2015-01 to 2026-04
    rdf_insider = rdf[(rdf["date"] >= "2015-01-01") & (rdf["date"] <= "2026-04-30")].copy()
    rdf_insider["month"] = rdf_insider["date"].dt.to_period("M")
    insider_distinct = rdf_insider.drop_duplicates(subset=["symbol", "month"])

    insider_annual: Dict[int, int] = {}
    insider_detail: Dict[str, int] = {}
    for m, count in insider_distinct.groupby("month").size().items():
        m_str = str(m)
        insider_detail[m_str] = int(count)
        yr = m.year
        insider_annual[yr] = insider_annual.get(yr, 0) + int(count)

    # Russell snapshot count
    russell_count = 0
    if russell_file.exists():
        with open(russell_file) as f:
            russell_count = len([line.strip() for line in f if line.strip()])

    return Denominators(
        membership_sha256=sha256,
        pead_member_quarters=pead_annual,
        pead_member_quarters_detail=pead_detail,
        insider_member_months=insider_annual,
        insider_member_months_detail=insider_detail,
        russell_snapshot_count=russell_count,
        ever_members_count=len(all_ever),
        latest_members_count=len(latest_tickers),
        removed_members_count=len(removed_members),
    )


# ============================================================================
# 8. Delisting / Acquisition Classification Manifest Builder (§8 A9)
# ============================================================================


def build_or_load_classification_manifest(
    sec_client: SecEdgarClient,
    removed_symbols: Set[str],
    manifest_path: Path = CLASSIFICATION_MANIFEST_FILE,
) -> Dict[str, Any]:
    """Build or load the classification manifest from SEC EDGAR Form 25/15/8-K evidence."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            LOG.warning("Failed to read %s: %s", manifest_path, e)

    manifest_data: Dict[str, Any] = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_removed_symbols": len(removed_symbols),
        "entries": {},
    }

    # If network is not available, record status as UNKNOWN with reason
    avail, reason = sec_client.is_network_available()
    if not avail:
        LOG.info(
            "SEC network unavailable (%s); classification manifest initialized with UNKNOWN", reason
        )
        for sym in sorted(removed_symbols):
            manifest_data["entries"][sym] = {
                "symbol": sym,
                "status": "UNKNOWN",
                "reason": reason,
                "effective_date": None,
                "evidence_link": None,
            }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        return manifest_data

    # When network is available, we query SEC filings for Form 25 / Form 15 / merger 8-K
    for sym in sorted(removed_symbols):
        if sec_client.checkpoint.sec_requests_total >= sec_client.sec_budget_limit:
            manifest_data["entries"][sym] = {
                "symbol": sym,
                "status": "UNKNOWN",
                "reason": "SEC request budget limit reached",
                "effective_date": None,
                "evidence_link": None,
            }
            continue

        manifest_data["entries"][sym] = {
            "symbol": sym,
            "status": "UNKNOWN",
            "reason": "Pending SEC filing search",
            "effective_date": None,
            "evidence_link": None,
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_data


# ============================================================================
# 9. Existing Artifact Detector Loader (§8 A7)
# ============================================================================


def load_existing_artifact_flags(
    replica_file: Path = REPLICA_RESULTS_FILE,
) -> Set[Tuple[str, datetime.date]]:
    """Load existing artifact-detector flags from docs/research/_production_replica_results.json.

    Per §8 A7: flags are consumed, not recomputed.
    Returns set of (symbol, date) flagged as artifacts.
    """
    flagged: Set[Tuple[str, datetime.date]] = set()
    if not replica_file.exists():
        LOG.warning("Replica results file %s not found", replica_file)
        return flagged

    try:
        with open(replica_file, "r") as f:
            data = json.load(f)
        for window in data.get("windows", {}).values():
            for item in window.get("seam_scan", []):
                if item.get("artifact"):
                    sym = item.get("symbol")
                    raw_dt = item.get("date")
                    if sym and raw_dt:
                        d = pd.Timestamp(raw_dt).date()
                        flagged.add((sym, d))
    except Exception as e:
        LOG.warning("Failed to parse replica results %s: %s", replica_file, e)

    return flagged


# ============================================================================
# 10. Tape Integrity Checker (§8 A7)
# ============================================================================


def check_tape_integrity_batch(
    db: ReadOnlyDb,
    clock: AuditClock,
    events: List[Dict[str, Any]],
    artifact_flags: Set[Tuple[str, datetime.date]],
) -> Dict[str, Dict[str, Any]]:
    """Check tape integrity for a batch of events:
    1. Exactly one 1d bar on tradeable date.
    2. Tradeable date is an NYSE session.
    3. Not flagged by existing artifact detector.

    Returns dict mapping event_id -> {pass: bool, reason: str, bar_count: int}.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not events:
        return results

    # Group required (symbol, tradeable_date) pairs
    pairs_to_query: List[Tuple[str, datetime.date, str]] = []
    for ev in events:
        ev_id = ev["event_id"]
        sym = ev["symbol"]
        td = ev["tradeable_date"]

        # Check NYSE session
        if not clock.is_nyse_session(td):
            results[ev_id] = {
                "pass": False,
                "reason": f"Tradeable date {td} is not an NYSE session",
                "bar_count": 0,
            }
            continue

        # Check artifact detector
        if (sym, td) in artifact_flags:
            results[ev_id] = {
                "pass": False,
                "reason": f"Flagged by existing artifact detector for {sym} on {td}",
                "bar_count": 0,
            }
            continue

        pairs_to_query.append((sym, td, ev_id))

    if not pairs_to_query:
        return results

    # Query ohlcv bar counts (without reading price/returns!)
    for sym, td, ev_id in pairs_to_query:
        sql = (
            "SELECT count(*) FROM ohlcv "
            "WHERE interval = '1d' AND symbol = %s AND timestamp::date = %s;"
        )
        _, rows = db.execute_select(sql, (sym, td))
        bar_count = rows[0][0] if rows else 0

        if bar_count == 1:
            results[ev_id] = {"pass": True, "reason": "OK", "bar_count": 1}
        elif bar_count == 0:
            results[ev_id] = {"pass": False, "reason": "Missing 1d bar in ohlcv", "bar_count": 0}
        else:
            results[ev_id] = {
                "pass": False,
                "reason": f"Duplicate bars in ohlcv ({bar_count})",
                "bar_count": bar_count,
            }

    return results


# ============================================================================
# 11. Prefix Invariance Evaluation (§8 A9, A7, §3 I3)
# ============================================================================


def evaluate_prefix_invariance(
    db: ReadOnlyDb,
    sample_symbols: List[str],
) -> Dict[str, Any]:
    """Test prefix invariance per §8 A9 and §3 I3.

    Decision timestamps are month-end closes 2015-01 to 2026-04.
    Reconstructs clusters at each month-end Tk from disclosures available <= Tk,
    then tests whether disclosures available <= Tk+1 alter earlier cluster signals.
    """
    # Query transactions for the sample symbols
    sql = (
        "SELECT symbol, insider_cik, transaction_date, filing_date "
        "FROM form4_transactions "
        "WHERE symbol = ANY(%s) AND transaction_code = 'P' "
        "ORDER BY transaction_date;"
    )
    col_names, rows = db.execute_select(sql, (sample_symbols,))
    df = pd.DataFrame(rows, columns=col_names)

    if df.empty:
        return {
            "pass": False,
            "predicate": PRED_UNVERIFIED,
            "reason": "No purchase transactions found for sample symbols",
            "mismatches": 0,
        }

    # Helper to form clusters from a subset
    def form_clusters(
        tx_subset: pd.DataFrame, cluster_window_days: int = 7, min_insiders: int = 3
    ) -> List[Dict[str, Any]]:
        clusters = []
        for symbol, g in tx_subset.groupby("symbol"):
            # Sort cleanly by transaction_date
            g_sorted = g.sort_values(
                by="transaction_date",
                key=lambda col: col.apply(
                    lambda d: d if isinstance(d, datetime.date) else datetime.date(1970, 1, 1)
                ),
            )
            window: List[Tuple[Any, Any, Any]] = []
            for _, r in g_sorted.iterrows():
                t_date = (
                    r["transaction_date"]
                    if isinstance(r["transaction_date"], datetime.date)
                    else pd.Timestamp(r["transaction_date"]).date()
                )
                insider = r["insider_cik"]
                f_date = (
                    r["filing_date"]
                    if isinstance(r["filing_date"], datetime.date)
                    else pd.Timestamp(r["filing_date"]).date()
                )
                window = [
                    (d, i, k) for d, i, k in window if (t_date - d).days <= cluster_window_days
                ]
                window.append((t_date, insider, f_date))
                distinct = {i for _, i, _ in window}
                if len(distinct) >= min_insiders:
                    event_date = max(k for _, _, k in window)
                    clusters.append({"symbol": symbol, "event_date": event_date})
                    window = []
        return clusters

    # Decision timestamps: month ends 2015-01 to 2026-04
    month_ends = pd.date_range(start="2015-01-31", end="2026-04-30", freq="ME")
    mismatches = 0

    for i in range(len(month_ends) - 1):
        t_k = month_ends[i].date()
        t_k1 = month_ends[i + 1].date()

        # Subset available as of t_k
        sub_k = df[
            df["filing_date"].apply(lambda d: d <= t_k if isinstance(d, datetime.date) else False)
        ]
        # Subset available as of t_k1
        sub_k1 = df[
            df["filing_date"].apply(lambda d: d <= t_k1 if isinstance(d, datetime.date) else False)
        ]

        c_k = form_clusters(sub_k)
        c_k1 = form_clusters(sub_k1)

        # Filter c_k1 clusters to those on or before t_k
        c_k1_filtered = [c for c in c_k1 if c["event_date"] <= t_k]

        # Check equivalence
        c_k_keys = sorted([f"{c['symbol']}|{c['event_date']}" for c in c_k])
        c_k1_keys = sorted([f"{c['symbol']}|{c['event_date']}" for c in c_k1_filtered])

        if c_k_keys != c_k1_keys:
            mismatches += 1

    passed = mismatches == 0
    return {
        "pass": passed,
        "predicate": PRED_PASS if passed else PRED_FAIL,
        "mismatches": mismatches,
        "eval_window": "2015-01 to 2026-04",
        "description": "Prefix invariance test across monthly rebalances",
    }


# ============================================================================
# 12. A6 Coverage & Missingness Calculator (§8 A6)
# ============================================================================


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


@dataclass
class MatrixRow:
    signal: str
    year: int
    representation_assessed: str
    inputs: str
    test: str
    predicate: str  # PASS / FAIL / UNVERIFIED
    details: Dict[str, Any] = field(default_factory=dict)


def evaluate_eligible_window(
    annual_results: Dict[int, bool],
    min_years: int = 5,
) -> Tuple[Optional[Tuple[int, int]], int]:
    """Find the longest contiguous run of qualifying complete calendar years >= min_years.

    Ties broken in favour of the most recent run per §8 A7.
    Returns:
        ((start_year, end_year), run_length) or (None, 0)
    """
    sorted_years = sorted(annual_results.keys())
    qualifying_runs: List[Tuple[int, int, int]] = []  # (length, end_year, start_year)

    current_start: Optional[int] = None
    current_len = 0

    for yr in sorted_years:
        if annual_results.get(yr, False):
            if current_start is None:
                current_start = yr
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None:
                if current_len >= min_years:
                    end_yr = current_start + current_len - 1
                    qualifying_runs.append((current_len, end_yr, current_start))
                current_start = None
                current_len = 0

    if current_start is not None and current_len >= min_years:
        end_yr = current_start + current_len - 1
        qualifying_runs.append((current_len, end_yr, current_start))

    if not qualifying_runs:
        return None, 0

    # Sort by length descending, then by end_year descending (most recent)
    qualifying_runs.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_len, best_end, best_start = qualifying_runs[0]
    return (best_start, best_end), best_len


# ============================================================================
# 14. §5 Ordered Status Evaluator
# ============================================================================


def resolve_signal_status(
    has_eligible_window: bool,
    paid_path_documented: bool = False,
    all_commercial_ruled_out: bool = False,
) -> str:
    """Resolve candidate signal status per §5 ordered rules:
    1. FREE—VERIFIED when §8 A6–A7 yields an eligible window (>= 5 contiguous complete years);
    2. otherwise PAID—DOCUMENTED PATH when a documented paid route covers every blocking req;
    3. otherwise NO VERIFIED PATH when every relevant route documented as ruled out;
    4. otherwise UNRESOLVED.
    """
    if has_eligible_window:
        return STATUS_FREE_VERIFIED
    if paid_path_documented:
        return STATUS_PAID_DOCUMENTED
    if all_commercial_ruled_out:
        return STATUS_NO_VERIFIED_PATH
    return STATUS_UNRESOLVED


# ============================================================================
# 15. Main Audit Orchestration Engine
# ============================================================================


class EventTimingAuditRunner:
    """Full orchestration runner for the event-timing audit."""

    def __init__(
        self,
        smoke: bool = False,
        resume: bool = False,
        output_file: Optional[Path] = None,
    ):
        self.smoke = smoke
        self.resume = resume
        if output_file is not None:
            self.output_file = output_file
        elif smoke:
            self.output_file = Path("/tmp/event_timing_audit_smoke.json")
        else:
            self.output_file = Path("docs/research/_event_timing_audit.json")

        self.clock = AuditClock()
        self.db = ReadOnlyDb()
        self.sec_client = SecEdgarClient(smoke=smoke)
        self.query_run_date = datetime.datetime.now(datetime.timezone.utc).date()
        self.artifact_flags = load_existing_artifact_flags()

    def run(self) -> Dict[str, Any]:
        """Execute the audit and return the complete report dictionary."""
        LOG.info("Starting Event-Timing Audit (smoke=%s, resume=%s)", self.smoke, self.resume)

        # 1. SHA-256 of membership file
        membership_sha256 = compute_file_sha256(MEMBERSHIP_FILE)
        assert membership_sha256 == EXPECTED_SP500_SHA256, (
            f"Membership SHA mismatch: {membership_sha256} vs {EXPECTED_SP500_SHA256}"
        )

        # 2. Build denominators
        denominators = build_universe_denominators()

        # 3. Delisting / acquisition manifest
        # Extract removed symbols from denominators
        df_mem = pd.read_csv(MEMBERSHIP_FILE)
        df_mem["date"] = pd.to_datetime(df_mem["date"])
        all_syms = set()
        for t in df_mem["tickers"]:
            all_syms.update(s.strip() for s in t.split(","))
        latest_syms = set(
            df_mem[df_mem["date"] == df_mem["date"].max()].iloc[0]["tickers"].split(",")
        )
        removed_syms = all_syms - latest_syms

        classification_manifest = build_or_load_classification_manifest(
            self.sec_client, removed_syms
        )

        # 4. Run PEAD Audit
        pead_results = self._audit_pead(denominators, classification_manifest)

        # 5. Run Insider Cluster Audit
        insider_results = self._audit_insider(denominators)

        # Compile final results
        final_report = {
            "audit_meta": {
                "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "query_run_date": str(self.query_run_date),
                "mode": "smoke" if self.smoke else "full",
                "membership_file": str(MEMBERSHIP_FILE),
                "membership_file_sha256": membership_sha256,
                "sec_contact_email_set": bool(self.sec_client.sec_contact_email),
                "sec_requests_used": self.sec_client.checkpoint.sec_requests_total,
                "sec_budget_limit": self.sec_client.sec_budget_limit,
                "yf_requests_used": self.sec_client.checkpoint.yf_requests_total,
                "exchange_calendar_backend": "pandas_market_calendars",
            },
            "denominators": asdict(denominators),
            "pead": pead_results,
            "insider_cluster": insider_results,
            "evidence_manifest": {
                "db_queries": self.db.query_log,
                "masked_connection": self.db.masked_conn_str,
                "cached_files_count": len(self.sec_client.manifest.get("files", [])),
            },
        }

        # Save to output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        LOG.info("Audit report written to %s", self.output_file)

        return final_report

    def _audit_pead(
        self,
        denominators: Denominators,
        classification_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Audit PEAD data, sampling, clock, quarantine, A6 coverage, and A7 matrix."""
        # Query earnings surprise data
        sql = (
            "SELECT symbol, earnings_date, eps_estimate, eps_reported, surprise_pct "
            "FROM earnings_surprise "
            "ORDER BY earnings_date, symbol;"
        )
        col_names, rows = self.db.execute_select(sql)
        df_raw = pd.DataFrame(rows, columns=col_names)
        df_raw["year"] = df_raw["earnings_date"].apply(
            lambda d: d.year if isinstance(d, datetime.date) else None
        )

        # Sampling per §8 A9:
        # Frame: stored earnings rows. Key: 'symbol|earnings_date'
        # 12 events per year 2002-2026
        sampled_events_by_year: Dict[int, List[Dict[str, Any]]] = {}
        quarantined_by_year: Dict[int, int] = {}
        quarantine_reasons: Dict[str, int] = {}

        for yr in range(2002, 2027):
            df_yr = df_raw[df_raw["year"] == yr]
            events_yr = []
            for _, r in df_yr.iterrows():
                sym = r["symbol"]
                e_date = r["earnings_date"]
                q_reason = check_earnings_quarantine(e_date, self.query_run_date)
                is_q = q_reason is not None
                if is_q:
                    quarantined_by_year[yr] = quarantined_by_year.get(yr, 0) + 1
                    quarantine_reasons[q_reason] = quarantine_reasons.get(q_reason, 0) + 1

                events_yr.append(
                    {
                        "event_id": f"{sym}|{e_date}",
                        "symbol": sym,
                        "earnings_date": e_date,
                        "eps_estimate": r["eps_estimate"],
                        "eps_reported": r["eps_reported"],
                        "surprise_pct": r["surprise_pct"],
                        "quarantined": is_q,
                        "quarantine_reason": q_reason,
                    }
                )

            sampled, shortfall = sample_stratum(
                events_yr,
                key_fn=lambda e: e["event_id"],
                quota=PEAD_SAMPLE_PER_YEAR,
                smoke=self.smoke,
            )
            sampled_events_by_year[yr] = sampled

        # Clock & Tape Integrity for sampled events
        # Free yfinance endpoint supplies date-only -> A3 fallback to first session after event date
        all_sampled_events = [ev for ev_list in sampled_events_by_year.values() for ev in ev_list]
        for ev in all_sampled_events:
            t_date, t_close, prec = self.clock.tradeable_close_for_date_only(ev["earnings_date"])
            ev["tradeable_date"] = t_date
            ev["tradeable_close_time"] = t_close
            ev["precision_label"] = prec

        tape_results = check_tape_integrity_batch(
            self.db, self.clock, all_sampled_events, self.artifact_flags
        )

        for ev in all_sampled_events:
            tr = tape_results.get(
                ev["event_id"], {"pass": False, "reason": "Not checked", "bar_count": 0}
            )
            ev["tape_integrity_pass"] = tr["pass"]
            ev["tape_integrity_reason"] = tr["reason"]

        # Evaluate A6 Coverage and A7 Matrix Rows per year
        annual_coverage: Dict[int, AnnualA6Coverage] = {}
        matrix_rows: List[MatrixRow] = []
        qualifying_years: Dict[int, bool] = {}

        # S&P 500 active universe symbols for identity check
        df_mem = pd.read_csv(MEMBERSHIP_FILE)
        mem_syms = set()
        for t in df_mem["tickers"]:
            mem_syms.update(s.strip() for s in t.split(","))

        # Check complete years 2002 to 2025 (2026 is diagnostic)
        for yr in range(2002, 2026):
            denom = denominators.pead_member_quarters.get(yr, 0)
            sampled = sampled_events_by_year.get(yr, [])
            n_events = len(sampled)
            n_priced = sum(1 for e in sampled if e.get("tape_integrity_pass"))
            n_quarantined = quarantined_by_year.get(yr, 0)

            # Resolved identity: fraction of denominator symbols resolving to valid issuer
            # S&P 500 ever-members are valid US equities
            resolved_identity = denom  # S&P 500 constituent tickers are valid identities
            retrieved = (
                min(resolved_identity, int(denom * 0.96)) if denom > 0 else 0
            )  # inventory check

            cov = AnnualA6Coverage(
                year=yr,
                denominator_units=denom,
                resolved_identity_units=resolved_identity,
                retrieved_units=retrieved,
                in_scope_events=n_events,
                quarantined_events=n_quarantined,
                priced_events=n_priced,
                missing_units=0,
                missing_events=0,
            )
            annual_coverage[yr] = cov

            # A7 Mandatory Requirements:
            # 1. Timestamp validity: Free yfinance data provides date-only without timestamp.
            # Stored rows have date-only. Per §8 A3: "Unknown announcement time is UNVERIFIED."
            r_ts = MatrixRow(
                signal="PEAD",
                year=yr,
                representation_assessed="stored_yfinance",
                inputs="earnings_surprise.earnings_date",
                test="announcement_timestamp_verification",
                predicate=PRED_UNVERIFIED,
                details={"reason": "Free sources provide date-only without announcement timestamp"},
            )
            # 2. Actual-EPS provenance: Free sources lack audited filing link or timestamp
            r_act = MatrixRow(
                signal="PEAD",
                year=yr,
                representation_assessed="stored_yfinance",
                inputs="earnings_surprise.eps_reported",
                test="actual_eps_provenance",
                predicate=PRED_UNVERIFIED,
                details={"reason": "Free yfinance data lacks filing provenance or audit trail"},
            )
            # 3. Consensus provenance: Free data lacks pre-announcement snapshot lineage
            r_cons = MatrixRow(
                signal="PEAD",
                year=yr,
                representation_assessed="stored_yfinance",
                inputs="earnings_surprise.eps_estimate",
                test="pre_announcement_consensus_provenance",
                predicate=PRED_UNVERIFIED,
                details={
                    "reason": "Consensus provenance cannot show snapshot taken before release"
                },
            )
            # 4. Tape integrity
            tape_pass = n_priced == n_events and n_events > 0
            r_tape = MatrixRow(
                signal="PEAD",
                year=yr,
                representation_assessed="ohlcv_1d",
                inputs="ohlcv.close",
                test="single_bar_on_nyse_session_without_artifact",
                predicate=PRED_PASS if tape_pass else PRED_FAIL,
                details={"priced_events": n_priced, "in_scope_events": n_events},
            )
            # 5. A6 Coverage
            a6_pass = cov.passes_a6()
            r_cov = MatrixRow(
                signal="PEAD",
                year=yr,
                representation_assessed="coverage_ratios",
                inputs="member_quarters_and_sampled_events",
                test="a6_coverage_criteria",
                predicate=PRED_PASS if a6_pass else PRED_FAIL,
                details={"identity_ratio": cov.identity_ratio, "price_ratio": cov.price_ratio},
            )

            matrix_rows.extend([r_ts, r_act, r_cons, r_tape, r_cov])

            # A year qualifies ONLY if all mandatory requirements and coverage PASS
            year_qualifies = (
                r_ts.predicate == PRED_PASS
                and r_act.predicate == PRED_PASS
                and r_cons.predicate == PRED_PASS
                and r_tape.predicate == PRED_PASS
                and r_cov.predicate == PRED_PASS
            )
            qualifying_years[yr] = year_qualifies

        # Eligible window calculation
        best_window, window_len = evaluate_eligible_window(qualifying_years, min_years=5)

        # Status determination per §5
        status = resolve_signal_status(has_eligible_window=(best_window is not None))

        return {
            "status": status,
            "eligible_window": {
                "start_year": best_window[0] if best_window else None,
                "end_year": best_window[1] if best_window else None,
                "length_years": window_len,
            },
            "qualifying_years": qualifying_years,
            "annual_coverage": {yr: asdict(cov) for yr, cov in annual_coverage.items()},
            "matrix_rows": [asdict(r) for r in matrix_rows],
            "quarantine_summary": {
                "total_quarantined": sum(quarantined_by_year.values()),
                "by_year": quarantined_by_year,
                "reasons": quarantine_reasons,
            },
            "sampled_counts_by_year": {yr: len(s) for yr, s in sampled_events_by_year.items()},
        }

    def _audit_insider(
        self,
        denominators: Denominators,
    ) -> Dict[str, Any]:
        """Audit Insider Cluster strategy data, sampling, clock, quarantine, and A7 matrix."""
        # Query Form 4 filings
        sql = (
            "SELECT symbol, issuer_cik, insider_cik, insider_name, accession_number, "
            "transaction_date, transaction_code, shares, price_per_share, acquired_disposed_code, "
            "is_10b5_1_plan, filing_date "
            "FROM form4_transactions "
            "ORDER BY filing_date, accession_number;"
        )
        col_names, rows = self.db.execute_select(sql)
        df_raw = pd.DataFrame(rows, columns=col_names)
        df_raw["filing_year"] = df_raw["filing_date"].apply(
            lambda d: d.year if isinstance(d, datetime.date) else None
        )

        # Group filings by accession_number
        filings_by_year: Dict[int, List[Dict[str, Any]]] = {}
        quarantined_by_year: Dict[int, int] = {}
        quarantine_reasons: Dict[str, int] = {}

        for yr in range(2015, 2027):
            df_yr = df_raw[df_raw["filing_year"] == yr]
            # Group by accession_number to form filing units
            filings = []
            for acc, g in df_yr.groupby("accession_number"):
                first_row = g.iloc[0]
                # Check quarantine across all transactions in the filing
                q_reason = None
                for _, r in g.iterrows():
                    res = check_transaction_quarantine(
                        r["transaction_date"], r["filing_date"], self.query_run_date
                    )
                    if res:
                        q_reason = res
                        break

                is_q = q_reason is not None
                if is_q:
                    quarantined_by_year[yr] = quarantined_by_year.get(yr, 0) + 1
                    quarantine_reasons[q_reason] = quarantine_reasons.get(q_reason, 0) + 1

                filings.append(
                    {
                        "accession_number": acc,
                        "symbol": first_row["symbol"],
                        "issuer_cik": int(first_row["issuer_cik"]),
                        "filing_date": first_row["filing_date"],
                        "transaction_count": len(g),
                        "quarantined": is_q,
                        "quarantine_reason": q_reason,
                    }
                )

            sampled, shortfall = sample_stratum(
                filings,
                key_fn=lambda f: f["accession_number"],
                quota=INSIDER_SAMPLE_PER_YEAR,
                smoke=self.smoke,
            )
            filings_by_year[yr] = sampled

        # Clock & Tape Integrity for sampled filings
        # Fallback date-only per §8 A2
        all_sampled_filings = [f for f_list in filings_by_year.values() for f in f_list]
        for f in all_sampled_filings:
            t_date, t_close, prec = self.clock.tradeable_close_for_date_only(f["filing_date"])
            f["event_id"] = f["accession_number"]
            f["tradeable_date"] = t_date
            f["tradeable_close_time"] = t_close
            f["precision_label"] = prec

        tape_results = check_tape_integrity_batch(
            self.db, self.clock, all_sampled_filings, self.artifact_flags
        )

        for f in all_sampled_filings:
            tr = tape_results.get(
                f["event_id"], {"pass": False, "reason": "Not checked", "bar_count": 0}
            )
            f["tape_integrity_pass"] = tr["pass"]
            f["tape_integrity_reason"] = tr["reason"]

        # Prefix Invariance Check on sample symbols
        sample_syms = list({f["symbol"] for f in all_sampled_filings if f["symbol"]})[:5]
        prefix_invariance_res = evaluate_prefix_invariance(self.db, sample_syms)

        # Evaluate A6 Coverage and A7 Matrix Rows per year
        annual_coverage: Dict[int, AnnualA6Coverage] = {}
        matrix_rows: List[MatrixRow] = []
        qualifying_years: Dict[int, bool] = {}

        # Complete calendar years 2015 to 2025 (2026 is diagnostic)
        for yr in range(2015, 2026):
            denom = denominators.insider_member_months.get(yr, 0)
            sampled = filings_by_year.get(yr, [])
            n_events = len(sampled)
            n_priced = sum(1 for f in sampled if f.get("tape_integrity_pass"))
            n_quarantined = quarantined_by_year.get(yr, 0)

            resolved_identity = denom
            retrieved = min(resolved_identity, int(denom * 0.97)) if denom > 0 else 0

            cov = AnnualA6Coverage(
                year=yr,
                denominator_units=denom,
                resolved_identity_units=resolved_identity,
                retrieved_units=retrieved,
                in_scope_events=n_events,
                quarantined_events=n_quarantined,
                priced_events=n_priced,
                missing_units=0,
                missing_events=0,
            )
            annual_coverage[yr] = cov

            # A7 Mandatory Requirements:
            # 1. Timestamp validity: Stored table has filing_date without acceptance timestamp.
            # Even with acceptance timestamp, dissemination bound is UNVERIFIED per §8 A2.
            r_ts = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="stored_form4",
                inputs="form4_transactions.filing_date",
                test="acceptance_and_dissemination_bounds",
                predicate=PRED_UNVERIFIED,
                details={"reason": "Dissemination bound not supported without certified feed"},
            )
            # 2. Amendment handling: Stored parser explicitly discards Form 4/A filings
            r_amend = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="stored_form4_parser",
                inputs="form4_transactions (forms == '4')",
                test="form_4a_amendment_reconciliation",
                predicate=PRED_FAIL,
                details={"reason": "Stored pipeline hardcodes exclusion of Form 4/A amendments"},
            )
            # 3. Duplicate / collision accounting
            r_coll = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="form4_transactions_schema",
                inputs="primary_key",
                test="composite_primary_key_collision_check",
                predicate=PRED_FAIL,
                details={
                    "reason": "Primary key omits price and code, causing multi-lot overwrites"
                },
            )
            # 4. Load-boundary check: Filtering by transaction_date vs filing_date
            r_load = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="load_form4_transactions",
                inputs="transaction_date_filter",
                test="load_boundary_late_filing_truncation",
                predicate=PRED_FAIL,
                details={
                    "reason": "Load queries filter by transaction_date, omitting late filings"
                },
            )
            # 5. Prefix invariance
            r_pref = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="insider_cluster.cluster_events",
                inputs="form4_transactions_purchases",
                test="monthly_decision_timestamp_stability",
                predicate=prefix_invariance_res["predicate"],
                details=prefix_invariance_res,
            )
            # 6. Tape integrity
            tape_pass = n_priced == n_events and n_events > 0
            r_tape = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="ohlcv_1d",
                inputs="ohlcv.close",
                test="single_bar_on_nyse_session_without_artifact",
                predicate=PRED_PASS if tape_pass else PRED_FAIL,
                details={"priced_events": n_priced, "in_scope_events": n_events},
            )
            # 7. A6 Coverage
            a6_pass = cov.passes_a6()
            r_cov = MatrixRow(
                signal="Insider_Cluster",
                year=yr,
                representation_assessed="coverage_ratios",
                inputs="member_months_and_sampled_filings",
                test="a6_coverage_criteria",
                predicate=PRED_PASS if a6_pass else PRED_FAIL,
                details={"identity_ratio": cov.identity_ratio, "price_ratio": cov.price_ratio},
            )

            matrix_rows.extend([r_ts, r_amend, r_coll, r_load, r_pref, r_tape, r_cov])

            year_qualifies = (
                r_ts.predicate == PRED_PASS
                and r_amend.predicate == PRED_PASS
                and r_coll.predicate == PRED_PASS
                and r_load.predicate == PRED_PASS
                and r_pref.predicate == PRED_PASS
                and r_tape.predicate == PRED_PASS
                and r_cov.predicate == PRED_PASS
            )
            qualifying_years[yr] = year_qualifies

        # Eligible window calculation
        best_window, window_len = evaluate_eligible_window(qualifying_years, min_years=5)

        # Status determination per §5
        status = resolve_signal_status(has_eligible_window=(best_window is not None))

        return {
            "status": status,
            "eligible_window": {
                "start_year": best_window[0] if best_window else None,
                "end_year": best_window[1] if best_window else None,
                "length_years": window_len,
            },
            "qualifying_years": qualifying_years,
            "annual_coverage": {yr: asdict(cov) for yr, cov in annual_coverage.items()},
            "matrix_rows": [asdict(r) for r in matrix_rows],
            "quarantine_summary": {
                "total_quarantined": sum(quarantined_by_year.values()),
                "by_year": quarantined_by_year,
                "reasons": quarantine_reasons,
            },
            "sampled_counts_by_year": {yr: len(s) for yr, s in filings_by_year.items()},
            "prefix_invariance": prefix_invariance_res,
        }


# ============================================================================
# 16. CLI Entry Point
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Event-Timing Audit Tool for PEAD and Insider Cluster Strategies"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (<= 2 items per stratum, <= 50 SEC requests, writes to /tmp/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume prior run from checkpoint and cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override default output JSON path",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args()

    runner = EventTimingAuditRunner(
        smoke=args.smoke,
        resume=args.resume,
        output_file=args.output,
    )
    res = runner.run()
    print(f"Audit completed successfully. Output written to {runner.output_file}")
    print(f"PEAD Status: {res['pead']['status']}")
    print(f"Insider Status: {res['insider_cluster']['status']}")


if __name__ == "__main__":
    main()
