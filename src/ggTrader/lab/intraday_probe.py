"""Bounded metadata-only intraday Stage 0A coverage diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

ENDPOINTS = {
    "calendar": "https://paper-api.alpaca.markets/v2/calendar",
    "bars": "https://data.alpaca.markets/v2/stocks/bars",
    "auctions": "https://data.alpaca.markets/v2/stocks/auctions",
}
NEW_YORK = ZoneInfo("America/New_York")
TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})\Z"
)
VENUE = re.compile(r"[A-Z0-9]{1,4}\Z")
MINUTE_NS = 60_000_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "research/intraday/spy-momentum-feasibility"
DEFAULT_OUTPUT = STUDY_DIR / "_intraday_probe.json"
DOCUMENTS = {
    "strategy_brief": STUDY_DIR / "2026-10-24-intraday-feasibility.md",
    "contract": STUDY_DIR / "2026-10-25-intraday-stage0a-contract.md",
}
SUCCESS = {"nonempty_success", "empty_success"}


class SchemaError(ValueError):
    """The response lacks documented metadata."""


class RequestDeadline(Exception):
    """A request exceeded its total wall-clock budget."""


@contextmanager
def request_deadline(seconds: float = 20) -> Iterator[None]:
    """Bound the whole request/body read, including a slowly streaming response."""

    def expired(signum: int, frame: Any) -> None:
        raise RequestDeadline()

    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def sample_months() -> list[str]:
    """Select the twenty fixed diagnostic months from the 129-month range."""
    return [f"{2016 + (j * 128 // 19) // 12:04d}-{(j * 128 // 19) % 12 + 1:02d}" for j in range(20)]


def utc_bounds(day: str, start: str, end: str) -> tuple[str, str]:
    """Convert New York session bounds explicitly, including DST."""
    return tuple(
        datetime.fromisoformat(f"{day}T{clock}")
        .replace(tzinfo=NEW_YORK)
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z")
        for clock in (start, end)
    )


def timestamp_ns(value: Any) -> int:
    """Validate a zoned timestamp without losing sub-microsecond defects."""
    if not isinstance(value, str) or not (match := TIMESTAMP.fullmatch(value)):
        raise SchemaError()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(microsecond=0)
        epoch_seconds = int(parsed.timestamp())
    except (ValueError, OverflowError, OSError):
        raise SchemaError() from None
    return epoch_seconds * 1_000_000_000 + int((match.group(1) or "").ljust(9, "0"))


def valid_date(value: Any) -> str:
    """Accept only a canonical ISO date."""
    if not isinstance(value, str):
        raise SchemaError()
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    raise SchemaError()


def metadata(kind: str, payload: Any) -> tuple[list[dict], str | None]:
    """Copy only documented metadata; never retain prices or unknown fields."""
    records = []
    if kind == "calendar":
        if not isinstance(payload, list):
            raise SchemaError()
        for row in payload:
            if not isinstance(row, dict):
                raise SchemaError()
            day = valid_date(row.get("date"))
            for field in ("open", "close"):
                value = row.get(field)
                if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
                    raise SchemaError()
                try:
                    datetime.strptime(value, "%H:%M")
                except ValueError:
                    raise SchemaError() from None
            records.append({"date": day, "open": row["open"], "close": row["close"]})
        return records, None
    if not isinstance(payload, dict) or "next_page_token" not in payload:
        raise SchemaError()
    token = payload["next_page_token"]
    if token is not None and (not isinstance(token, str) or not token):
        raise SchemaError()
    groups = payload.get(kind)
    if not isinstance(groups, dict) or not isinstance(groups.get("SPY", []), list):
        raise SchemaError()
    for row in groups.get("SPY", []):
        if not isinstance(row, dict):
            raise SchemaError()
        if kind == "bars":
            timestamp_ns(row.get("t"))
            volume = row.get("v")
            if (
                isinstance(volume, bool)
                or not isinstance(volume, (int, float))
                or not math.isfinite(volume)
                or volume < 0
            ):
                raise SchemaError()
            records.append({"t": row["t"], "v": volume})
        else:
            day = valid_date(row.get("d"))
            for field, event_type in (("o", "opening"), ("c", "closing")):
                if not isinstance(row.get(field), list):
                    raise SchemaError()
                for event in row[field]:
                    if not isinstance(event, dict):
                        raise SchemaError()
                    timestamp_ns(event.get("t"))
                    venue = event.get("x")
                    if not isinstance(venue, str) or not VENUE.fullmatch(venue):
                        raise SchemaError()
                    records.append({"d": day, "t": event["t"], "x": venue, "type": event_type})
    return records, token


def select_session(month: str, records: list[dict]) -> str | None:
    """Use the first official full session within the first ten calendar days."""
    start = date.fromisoformat(f"{month}-01")
    end = start + timedelta(days=9)
    eligible = [
        row["date"]
        for row in records
        if start.isoformat() <= row["date"] <= end.isoformat()
        and row["open"] == "09:30"
        and row["close"] == "16:00"
    ]
    return min(eligible, default=None)


def grid_summary(records: list[dict], start: str, end: str) -> dict:
    """Retain every timestamp defect against a fixed inclusive/exclusive grid."""
    first, last = timestamp_ns(start), timestamp_ns(end)
    stamps = [timestamp_ns(row["t"]) for row in records]
    expected = set(range(first, last, MINUTE_NS))
    regular = [stamp for stamp in stamps if first <= stamp < last]
    missing = sorted(expected - set(stamps))
    return {
        "expected_count": len(expected),
        "row_count": len(records),
        "unique_timestamps": len(set(stamps)),
        "duplicate_count": len(stamps) - len(set(stamps)),
        "regular_duplicate_count": len(regular) - len(set(regular)),
        "regular_misaligned_count": sum(bool(stamp % MINUTE_NS) for stamp in regular),
        "missing": [
            datetime.fromtimestamp(value // 1_000_000_000, UTC).isoformat().replace("+00:00", "Z")
            for value in missing
        ],
        "out_of_grid": [
            row["t"]
            for row, stamp in zip(records, stamps, strict=True)
            if not first <= stamp < last
        ],
        "misaligned": [
            row["t"] for row, stamp in zip(records, stamps, strict=True) if stamp % MINUTE_NS
        ],
        "volume_sum": sum(row["v"] for row in records),
    }


class ProbeClient:
    """Read only the three fixed endpoints with bounded retries and pagination."""

    def __init__(
        self,
        session: requests.Session,
        *,
        clock: Callable = time.monotonic,
        sleep: Callable = time.sleep,
    ) -> None:
        self.session = session
        self.clock = clock
        self.sleep = sleep
        self.last_start: float | None = None

    def request(self, kind: str, params: dict) -> dict:
        """Return sanitized evidence, never raw response bodies or page tokens."""
        endpoint = ENDPOINTS[kind]
        result = {
            "endpoint": endpoint,
            "parameters": dict(params),
            "attempts": [],
            "records": [],
            "pagination": {"pages": 0, "complete": False},
        }
        token = None
        seen = set()
        for page in range(1, 21):
            payload = None
            for attempt in range(1, 4):
                if self.last_start is not None:
                    delay = 0.65 - (self.clock() - self.last_start)
                    if delay > 0:
                        self.sleep(delay)
                self.last_start = self.clock()
                evidence = {"page": page, "attempt": attempt, "utc": datetime.now(UTC).isoformat()}
                result["attempts"].append(evidence)
                response = None
                headers = {}
                try:
                    request_params = dict(params)
                    if token is not None:
                        request_params["page_token"] = token
                    with request_deadline():
                        response = self.session.get(
                            endpoint,
                            params=request_params,
                            timeout=20,
                            stream=True,
                            allow_redirects=False,
                        )
                        body = response.content
                    status = response.status_code
                    headers = {str(k).lower(): v for k, v in response.headers.items()}
                    evidence.update(
                        status_code=status, body_sha256=hashlib.sha256(body).hexdigest()
                    )
                except (requests.RequestException, RequestDeadline) as error:
                    status = None
                    evidence["transport_error"] = (
                        "timeout"
                        if isinstance(error, (requests.Timeout, RequestDeadline))
                        else "transport_error"
                    )
                finally:
                    if response is not None:
                        response.close()
                if status == 200:
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeError):
                        result["outcome"] = "schema_error"
                        return self._finish(result)
                    break
                retryable = status is None or status == 429 or 500 <= status <= 599
                if not retryable:
                    result["outcome"] = {401: "authentication_error", 403: "entitlement_error"}.get(
                        status, "invalid_request"
                    )
                    return self._finish(result)
                delay = float(attempt)
                for name in ("retry-after", "x-ratelimit-reset"):
                    try:
                        required = float(headers.get(name, "nan"))
                        if name == "x-ratelimit-reset":
                            required -= time.time()
                        if math.isfinite(required):
                            delay = max(delay, required)
                    except (ValueError, TypeError):
                        pass
                if delay > 30:
                    result["outcome"] = "deferred"
                    return self._finish(result)
                if attempt == 3:
                    result["outcome"] = (
                        "throttled_exhausted" if status == 429 else "transient_exhausted"
                    )
                    return self._finish(result)
                self.sleep(delay)
            try:
                records, next_token = metadata(kind, payload)
            except (SchemaError, TypeError, ValueError, OverflowError):
                result["outcome"] = "schema_error"
                return self._finish(result)
            result["records"].extend(records)
            result["pagination"]["pages"] = page
            if next_token is None:
                result["pagination"]["complete"] = True
                result["outcome"] = "nonempty_success" if result["records"] else "empty_success"
                return self._finish(result)
            if next_token in seen:
                break
            seen.add(next_token)
            token = next_token
        result["outcome"] = "incomplete_pagination"
        return self._finish(result)

    @staticmethod
    def _finish(result: dict) -> dict:
        """Identify observed timestamp bounds without claiming archive inception."""
        records = [row for row in result["records"] if "t" in row]
        records.sort(key=lambda row: timestamp_ns(row["t"]))
        result["first_observed_timestamp"] = records[0]["t"] if records else None
        result["last_observed_timestamp"] = records[-1]["t"] if records else None
        return result


def canonical_hash(value: Any) -> str:
    """Hash the canonical JSON used to freeze the calendar/sample manifest."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def atomic_checkpoint(output: Path, artifact: dict, *, initial: bool = False) -> None:
    """Publish a complete JSON checkpoint; initial publication never overwrites."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(artifact, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if initial:
            os.link(temporary, output)
        else:
            os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def request_plan(dates: list[str]) -> list[dict]:
    """Freeze every bar/auction request, including the separate coverage sentinels."""
    plans = []
    sessions = [("sample", day, "09:30", "16:00") for day in dates]
    sessions += [
        ("sentinel", day, "09:30", "09:35") for day in ("2014-01-02", "2015-01-02", "2016-01-04")
    ]
    sessions.append(("previous_close", "2015-12-31", "15:59", "16:01"))
    for role, day, opening, closing in sessions:
        start, end = utc_bounds(day, opening, closing)
        for feed in ("sip", "iex"):
            params = {
                "symbols": "SPY",
                "timeframe": "1Min",
                "adjustment": "raw",
                "asof": "-",
                "sort": "asc",
                "limit": 10000,
                "feed": feed,
                "start": start,
                "end": end,
            }
            plans.append(
                {
                    "id": f"{role}:{day}:{feed}",
                    "kind": "bars",
                    "role": role,
                    "date": day,
                    "feed": feed,
                    "parameters": params,
                    "required": role != "sentinel" or day == "2016-01-04",
                }
            )
        if role != "previous_close":
            auction_start, auction_end = utc_bounds(
                day, "09:30" if role == "sample" else "15:59", "16:01"
            )
            plans.append(
                {
                    "id": f"{role}:{day}:auctions",
                    "kind": "auctions",
                    "role": role,
                    "date": day,
                    "feed": "sip",
                    "parameters": {
                        "symbols": "SPY",
                        "feed": "sip",
                        "asof": "-",
                        "sort": "asc",
                        "limit": 10000,
                        "start": auction_start,
                        "end": auction_end,
                    },
                }
            )
    return plans


def conclude(artifact: dict) -> None:
    """Keep primary bar evidence separate from auction and IEX limitations."""
    completeness = {}
    for feed in ("sip", "iex"):
        bars = [
            row for row in artifact["requests"] if row.get("kind") == "bars" and row["feed"] == feed
        ]
        required = [row for row in bars if row["required"]]
        failed = [
            row["id"]
            for row in required
            if row["outcome"] not in SUCCESS
            or row["grid"]["missing"]
            or row["grid"]["regular_duplicate_count"]
            or row["grid"]["regular_misaligned_count"]
        ]
        completeness[feed] = {
            "required_checks": len(required),
            "failed_required_checks": failed,
            "limitations": [
                f"{row['id']}:{row['outcome']}" for row in bars if row["outcome"] not in SUCCESS
            ],
        }
    artifact["feed_completeness"] = completeness
    failed = completeness["sip"]["failed_required_checks"]
    artifact["reasons"] = [f"primary_bar_check_failed:{value}" for value in failed]
    if completeness["sip"]["required_checks"] != 22:
        artifact["reasons"].append("primary_bar_checks_incomplete")
    artifact["conclusion"] = "unresolved" if artifact["reasons"] else "sample_checks_passed"
    auctions = [row for row in artifact["requests"] if row.get("kind") == "auctions"]
    artifact["auction_limitations"] = [
        f"{row['id']}:{row['outcome']}" for row in auctions if row["outcome"] != "nonempty_success"
    ]
    for row in auctions:
        observed = {event["type"] for event in row["records"]}
        expected = {"opening", "closing"} if row["role"] == "sample" else {"closing"}
        artifact["auction_limitations"].extend(
            f"{row['id']}:missing_{event_type}" for event_type in sorted(expected - observed)
        )
    artifact["auction_conclusion"] = (
        "auction_branch_unresolved"
        if len(auctions) != 23 or artifact["auction_limitations"]
        else "auction_metadata_observed"
    )


def run_probe(output: Path = DEFAULT_OUTPUT, *, client: ProbeClient | None = None) -> dict:
    """Run the authorized fixed diagnostic, checkpointing only metadata evidence."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("result_already_exists")
    owned_session = None
    if client is None:
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("credentials_missing")
        owned_session = requests.Session()
        owned_session.trust_env = False
        owned_session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        client = ProbeClient(owned_session)
    revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("revision_unavailable")
    artifact = {
        "version": 1,
        "stage": "stage0A",
        "authorization": {
            "instruction": "execute",
            "date": "2026-10-25",
            "scope": "Stage 0A metadata coverage only",
        },
        "run_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "document_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in DOCUMENTS.items()
        },
        "diagnostic_range": {
            "first_month": "2016-01",
            "last_month": "2026-09",
            "month_count": 129,
            "selection": "j*128//19; j=0..19",
        },
        "manifest": {"calendar": [], "selected_dates": [], "requests": []},
        "manifest_sha256": None,
        "requests": [],
        "feed_completeness": {},
        "status": "in_progress",
        "conclusion": "unresolved",
        "reasons": ["probe_incomplete"],
        "auction_conclusion": "auction_branch_unresolved",
        "auction_limitations": [],
        "full_window_verified": False,
        "strategy_evaluated": False,
    }
    try:
        atomic_checkpoint(output, artifact, initial=True)
        for month in sample_months():
            start = f"{month}-01"
            end = f"{month}-10"
            result = client.request("calendar", {"start": start, "end": end})
            result.update(id=f"calendar:{month}", kind="calendar")
            artifact["requests"].append(result)
            records = result["records"]
            selected = select_session(month, records) if result["outcome"] in SUCCESS else None
            if any(not start <= row["date"] <= end for row in records) or len(
                {row["date"] for row in records}
            ) != len(records):
                selected = None
            artifact["manifest"]["calendar"].append(
                {
                    "month": month,
                    "start": start,
                    "end": end,
                    "records": records,
                    "selected_date": selected,
                }
            )
            if selected is None:
                artifact["status"] = "completed"
                artifact["reasons"] = [f"calendar_session_unresolved:{month}"]
                atomic_checkpoint(output, artifact)
                return artifact
            artifact["manifest"]["selected_dates"].append(selected)
            atomic_checkpoint(output, artifact)
        plans = request_plan(artifact["manifest"]["selected_dates"])
        artifact["manifest"]["requests"] = plans
        artifact["manifest_sha256"] = canonical_hash(artifact["manifest"])
        atomic_checkpoint(output, artifact)
        for plan in plans:
            result = client.request(plan["kind"], plan["parameters"])
            result.update({key: value for key, value in plan.items() if key != "parameters"})
            if plan["kind"] == "bars":
                result["grid"] = grid_summary(
                    result["records"], plan["parameters"]["start"], plan["parameters"]["end"]
                )
            artifact["requests"].append(result)
            atomic_checkpoint(output, artifact)
        conclude(artifact)
        artifact["status"] = "completed"
        atomic_checkpoint(output, artifact)
        return artifact
    except FileExistsError:
        raise
    except Exception:
        # Keep the last published checkpoint, even if serialization or storage failed.
        artifact["status"] = "in_progress"
        artifact["conclusion"] = "unresolved"
        artifact["local_failure"] = "unexpected_local_failure"
        artifact["reasons"] = ["unexpected_local_failure"]
        return artifact
    finally:
        if owned_session is not None:
            owned_session.close()
