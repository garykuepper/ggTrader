"""Bounded, resumable, GET-only collection of the frozen r8 input manifest.

Public interface:

- ``Collector(session, clock=, sleep=).fetch(spec) -> dict``: one logical request, paginated to
  completion or failure, with per-page body hashes and sanitized attempt evidence.
- ``run_collection(output_dir, session=None) -> dict``: freeze/compare ``manifest.json``, write one
  immutable ``chunks/<id>.json`` per completely paginated request and checkpoint
  ``progress.json`` atomically. Credentials come only from the inherited environment.

Running this against the real vendor needs the owner's separate data-and-scoring approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from ggTrader.lab import intraday_momentum as core
from ggTrader.lab.intraday_probe import (
    RequestDeadline,
    SchemaError,
    atomic_checkpoint,
    canonical_hash,
    metadata,
    request_deadline,
    timestamp_ns,
)

ATTEMPTS = 3
REQUEST_SPACING_SECONDS = 0.65  # ~92 requests/minute, under Basic's documented 200/minute
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRY_WAIT_SECONDS = 30
# Safety bound only: reaching it marks the chunk incomplete, never complete.
MAX_PAGES_PER_REQUEST = 1000
MAX_RUN_SECONDS = 2 * 60 * 60
STOP_RUN = {
    "authentication_error",
    "entitlement_error",
    "deferred",
    "throttled_exhausted",
    "transient_exhausted",
    "run_time_cap",
}
BAR_FIELDS = ("t", "o", "c", "v")
ACTION_FIELDS = (
    "symbol",
    "ex_date",
    "process_date",
    "rate",
    "special",
    "foreign",
    "sub_type",
    "old_rate",
    "new_rate",
)


def _reject_constant(value: str) -> None:
    raise SchemaError()


def parse_page(kind: str, payload: Any) -> tuple[list[dict], str | None]:
    """Whitelist one page into the stored schema; malformed pages are schema errors."""
    if kind == "calendar":
        return metadata("calendar", payload)
    if not isinstance(payload, dict) or "next_page_token" not in payload:
        raise SchemaError()
    token = payload["next_page_token"]
    if token is not None and (not isinstance(token, str) or not token):
        raise SchemaError()
    records = []
    if kind == "corporate_actions":
        groups = payload.get("corporate_actions")
        if not isinstance(groups, dict):
            raise SchemaError()
        for group, rows in sorted(groups.items()):
            if not isinstance(rows, list):
                raise SchemaError()
            for row in rows:
                if not isinstance(row, dict):
                    raise SchemaError()
                records.append({"group": group, **{k: row[k] for k in ACTION_FIELDS if k in row}})
        return records, token
    groups = payload.get("bars")
    if not isinstance(groups, dict) or not isinstance(groups.get("SPY", []), list):
        raise SchemaError()
    if set(groups) - {"SPY"}:
        raise SchemaError()
    for row in groups.get("SPY", []):
        if not isinstance(row, dict) or any(field not in row for field in BAR_FIELDS):
            raise SchemaError()
        timestamp_ns(row["t"])
        records.append({field: row[field] for field in BAR_FIELDS})
    return records, token


class Collector:
    """GET the fixed endpoints with spacing, finite retries and complete pagination."""

    def __init__(
        self,
        session: requests.Session,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session = session
        self.clock = clock
        self.sleep = sleep
        self.last_start: float | None = None
        self.run_start = clock()

    def fetch(self, spec: dict) -> dict:
        """Return a sanitized result; records only when every page arrived intact."""
        result = {"id": spec["id"], "attempts": [], "pages": [], "records": []}
        token, seen = None, set()
        for page in range(1, MAX_PAGES_PER_REQUEST + 1):
            body, outcome = self._page(spec, token, page, result["attempts"])
            if outcome:
                return self._fail(result, outcome)
            try:
                payload = json.loads(body, parse_constant=_reject_constant)
                records, token = parse_page(spec["kind"], payload)
            except (SchemaError, ValueError, TypeError, UnicodeError):
                return self._fail(result, "schema_error")
            result["pages"].append(
                {
                    "page": page,
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "record_count": len(records),
                }
            )
            result["records"].extend(records)
            if token is None:
                result["outcome"] = "complete"
                return result
            if token in seen:
                return self._fail(result, "repeated_page_token")
            seen.add(token)
        return self._fail(result, "page_cap_reached")

    @staticmethod
    def _fail(result: dict, outcome: str) -> dict:
        # Partial pages are discarded so a resumed run refetches the whole logical chunk.
        result.update(outcome=outcome, records=[], pages=[])
        return result

    def _page(
        self, spec: dict, token: str | None, page: int, attempts: list
    ) -> tuple[bytes, str | None]:
        for attempt in range(1, ATTEMPTS + 1):
            if self.clock() - self.run_start > MAX_RUN_SECONDS:
                return b"", "run_time_cap"
            if self.last_start is not None:
                delay = REQUEST_SPACING_SECONDS - (self.clock() - self.last_start)
                if delay > 0:
                    self.sleep(delay)
            self.last_start = self.clock()
            evidence = {"page": page, "attempt": attempt, "utc": datetime.now(UTC).isoformat()}
            attempts.append(evidence)
            params = dict(spec["parameters"])
            if token is not None:
                params["page_token"] = token
            response, status, headers, body = None, None, {}, b""
            try:
                with request_deadline(REQUEST_TIMEOUT_SECONDS):
                    response = self.session.get(
                        spec["endpoint"],
                        params=params,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                        stream=True,
                        allow_redirects=False,
                    )
                    body = response.content
                status = response.status_code
                headers = {str(k).lower(): v for k, v in response.headers.items()}
                evidence.update(status_code=status, body_sha256=hashlib.sha256(body).hexdigest())
            except (requests.RequestException, RequestDeadline) as error:
                evidence["transport_error"] = (
                    "timeout"
                    if isinstance(error, (requests.Timeout, RequestDeadline))
                    else "transport_error"
                )
            finally:
                if response is not None:
                    response.close()
            if status == 200:
                return body, None
            if status is not None and status != 429 and not 500 <= status <= 599:
                return b"", {401: "authentication_error", 403: "entitlement_error"}.get(
                    status, "invalid_request"
                )
            delay = float(attempt)
            for name in ("retry-after", "x-ratelimit-reset"):
                try:
                    required = float(headers.get(name, "nan"))
                except (TypeError, ValueError):
                    continue
                if name == "x-ratelimit-reset":
                    required -= time.time()
                if math.isfinite(required):
                    delay = max(delay, required)
            if delay > MAX_RETRY_WAIT_SECONDS:
                return b"", "deferred"
            if attempt == ATTEMPTS:
                return b"", "throttled_exhausted" if status == 429 else "transient_exhausted"
            self.sleep(delay)
        raise AssertionError("unreachable")


def make_session() -> requests.Session:
    """Session from inherited credentials only: no dotenv, no proxies, no redirects."""
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("credentials_missing")
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    return session


def _chunk(spec: dict, fetched: dict) -> dict:
    return {
        "schema": core.SCHEMAS["chunk"],
        "version": 1,
        "id": spec["id"],
        "kind": spec["kind"],
        "endpoint": spec["endpoint"],
        "parameters": spec["parameters"],
        "retrieved_utc": fetched["attempts"][-1]["utc"],
        "complete": True,
        "pages": fetched["pages"],
        "records": fetched["records"],
    }


def run_collection(
    output_dir: Path,
    *,
    session: requests.Session | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Collect or resume the frozen manifest; completed chunks are never refetched or replaced."""
    owned = session is None
    session = make_session() if owned else session  # credentials checked before any write
    try:
        return _collect(Path(output_dir), Collector(session, clock=clock, sleep=sleep))
    finally:
        if owned:
            session.close()


def _collect(output_dir: Path, collector: Collector) -> dict:
    manifest = core.build_manifest()
    manifest_path = output_dir / core.MANIFEST_FILE
    progress_path = output_dir / core.PROGRESS_FILE
    (output_dir / core.CHUNK_DIR).mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if canonical_hash(json.loads(manifest_path.read_text())) != canonical_hash(manifest):
            raise ValueError("resume_manifest_mismatch")
    else:
        atomic_checkpoint(manifest_path, manifest, initial=True)
    manifest_sha = canonical_hash(manifest)
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"manifest_sha256": manifest_sha, "chunks": {}, "runs": []}
    )
    if progress.get("manifest_sha256") != manifest_sha:
        raise ValueError("resume_manifest_mismatch")
    for spec in manifest["requests"]:
        path = core.chunk_path(output_dir, spec["id"])
        recorded = progress["chunks"].get(spec["id"])
        if not path.exists():
            if recorded:
                raise ValueError(f"recorded_chunk_missing:{spec['id']}")
            continue
        actual = core.file_sha256(path)
        if recorded is None:
            # A crash between the chunk link and the checkpoint: adopt it only if it is ours.
            stored = json.loads(path.read_text())
            if stored.get("parameters") != spec["parameters"] or stored.get("complete") is not True:
                raise ValueError(f"unrecorded_chunk_invalid:{spec['id']}")
            progress["chunks"][spec["id"]] = actual
        elif recorded != actual:
            raise ValueError(f"chunk_hash_mismatch:{spec['id']}")

    run = {"started_utc": datetime.now(UTC).isoformat(), "runtime": core.runtime_versions()}
    run["requests"] = []
    progress["runs"].append(run)
    for spec in manifest["requests"]:
        if spec["id"] in progress["chunks"]:
            continue
        fetched = collector.fetch(spec)
        run["requests"].append(
            {
                "id": spec["id"],
                "outcome": fetched["outcome"],
                "pages": len(fetched["pages"]),
                "attempts": fetched["attempts"],
            }
        )
        if fetched["outcome"] == "complete":
            path = core.chunk_path(output_dir, spec["id"])
            atomic_checkpoint(path, _chunk(spec, fetched), initial=True)
            progress["chunks"][spec["id"]] = core.file_sha256(path)
        atomic_checkpoint(progress_path, progress)
        if fetched["outcome"] in STOP_RUN:
            break
    missing = [s["id"] for s in manifest["requests"] if s["id"] not in progress["chunks"]]
    progress["status"] = "complete" if not missing else "incomplete"
    run["finished_utc"] = datetime.now(UTC).isoformat()
    atomic_checkpoint(progress_path, progress)
    return {"status": progress["status"], "missing": missing, "manifest_sha256": manifest_sha}
