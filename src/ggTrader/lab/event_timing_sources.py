"""Budgeted, immutable evidence collection for the registered timing audit."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from ggTrader.data.core.index_constituents import normalize_yf_ticker

SEC_HOSTS = {"www.sec.gov", "data.sec.gov", "archives.sec.gov"}
YF_HOSTS = {"finance.yahoo.com"}
MAX_RUNTIME = 6 * 60 * 60
MAX_SEC_REQUESTS = 20_000
MAX_YF_REQUESTS = 2_000


def digest(data: bytes) -> str:
    """Hash exact response bytes."""
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes, preserve_on_failure: bool = False) -> None:
    """Replace a state file only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists() and not preserve_on_failure:
            Path(name).unlink()  # Only this function's temporary state file, never raw evidence.


def atomic_json(path: Path, data: Any) -> None:
    """Checkpoint JSON atomically."""
    atomic_bytes(path, json.dumps(data, indent=2, default=str, allow_nan=False).encode())


@dataclass
class CheckpointState:
    """Cumulative accounting; reservations are persisted before requests."""

    schema: str = "event-timing-accounting-v2"
    sec_requests_total: int = 0
    yf_requests_total: int = 0
    completed_strata: list[str] = field(default_factory=list)
    stopped_due_to_rate_limit: bool = False
    stopped_reason: str | None = None
    elapsed_seconds: float = 0.0
    request_log: list[dict] = field(default_factory=list)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: each HTTP request must have its own reservation."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        return None


class SecEdgarClient:
    """SEC/Yahoo transport, offline by default, with shared stop and durable budgets."""

    def __init__(self, cache_dir: Path, smoke: bool = False, offline: bool = True):
        self.cache_dir = Path(cache_dir)
        self.smoke = smoke
        self.offline = offline
        self.sec_contact_email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
        self.user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not self.user_agent and self.sec_contact_email:
            self.user_agent = f"ggTrader research {self.sec_contact_email}"
        self.manifest_file = self.cache_dir / "MANIFEST.json"
        self.checkpoint_file = self.cache_dir / "checkpoint.json"
        self.sec_budget_limit = 50 if smoke else MAX_SEC_REQUESTS
        self.yf_budget_limit = MAX_YF_REQUESTS
        self.started = time.monotonic()
        self.last_sec_req_time = 0.0
        self.opener = urllib.request.build_opener(SafeRedirect())
        self.checkpoint = CheckpointState()
        self.manifest: dict = {"files": []}
        try:
            if self.checkpoint_file.exists():
                raw = json.loads(self.checkpoint_file.read_text())
                if set(raw) != set(asdict(CheckpointState())):
                    raise ValueError("Incomplete accounting schema")
                self.checkpoint = CheckpointState(**raw)
                if (
                    raw["schema"] != "event-timing-accounting-v2"
                    or type(raw["stopped_due_to_rate_limit"]) is not bool
                    or not isinstance(raw["request_log"], list)
                    or not isinstance(raw["completed_strata"], list)
                    or not all(isinstance(s, str) for s in raw["completed_strata"])
                    or raw["stopped_reason"] is not None
                    and not isinstance(raw["stopped_reason"], str)
                    or raw["stopped_due_to_rate_limit"] != bool(raw["stopped_reason"])
                ):
                    raise ValueError("Invalid accounting fields")
                for count in (
                    self.checkpoint.sec_requests_total,
                    self.checkpoint.yf_requests_total,
                ):
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid request count")
                if (
                    type(self.checkpoint.elapsed_seconds) not in (int, float)
                    or not math.isfinite(self.checkpoint.elapsed_seconds)
                    or self.checkpoint.elapsed_seconds < 0
                ):
                    raise ValueError("Invalid elapsed time")
                for provider, count in (
                    ("sec", raw["sec_requests_total"]),
                    ("yf", raw["yf_requests_total"]),
                ):
                    if sum(r.get("provider") == provider for r in raw["request_log"]) != count:
                        raise ValueError("Reservations do not reconcile with counters")
                for reservation in raw["request_log"]:
                    if (
                        reservation.get("provider") not in ("sec", "yf")
                        or not isinstance(reservation.get("url"), str)
                        or type(reservation.get("attempt")) is not int
                        or not 1 <= reservation["attempt"] <= 4
                        or not isinstance(reservation.get("outcome"), str)
                    ):
                        raise ValueError("Invalid reservation")
                if any(r["outcome"] in ("HTTP 403", "HTTP 429") for r in raw["request_log"]):
                    if not raw["stopped_due_to_rate_limit"]:
                        raise ValueError("Lost external stop")
            elif self.cache_dir.exists() and any(self.cache_dir.iterdir()):
                raise ValueError("Missing checkpoint for pre-existing responses")
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            raise ValueError(
                "Invalid/missing checkpoint; preserve cache and recover accounting"
            ) from exc
        try:
            if self.manifest_file.exists():
                self.manifest = json.loads(self.manifest_file.read_text())
                if not isinstance(self.manifest["files"], list):
                    raise ValueError("Invalid files list")
                reserved_urls = {r["url"] for r in self.checkpoint.request_log}
                for entry in self.manifest["files"]:
                    if not all(entry.get(k) for k in ("url", "cache_file", "sha256")):
                        raise ValueError("Incomplete manifest entry")
                    if Path(entry["cache_file"]).name != entry["cache_file"]:
                        raise ValueError("Unsafe cache path")
                    if entry["url"] not in reserved_urls:
                        raise ValueError("Manifest response has no durable request reservation")
                    if entry.get("status_code") in (403, 429) and not (
                        self.checkpoint.stopped_due_to_rate_limit
                    ):
                        raise ValueError("Manifest stop is absent from accounting")
            elif self.checkpoint.sec_requests_total or self.checkpoint.yf_requests_total:
                raise ValueError("Missing manifest for used budgets")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError("Invalid/missing MANIFEST; preserve raw evidence") from exc
        self.base_elapsed = self.checkpoint.elapsed_seconds

    def save_checkpoint(self) -> None:
        """Keep accounting across all invocations, regardless of --resume."""
        self.checkpoint.elapsed_seconds = self.base_elapsed + time.monotonic() - self.started
        atomic_json(self.checkpoint_file, asdict(self.checkpoint))

    def within_deadline(self) -> bool:
        """Six hours accumulates across resumes."""
        return self.base_elapsed + time.monotonic() - self.started < MAX_RUNTIME

    def is_network_available(self) -> tuple[bool, str]:
        """Report why external collection is blocked, without contact details."""
        if self.offline:
            return False, "offline"
        if self.checkpoint.stopped_due_to_rate_limit:
            return False, "external requests stopped after HTTP 403/429"
        if not self.within_deadline():
            return False, "six-hour runtime budget exhausted"
        if not self.user_agent or not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", self.user_agent):
            return False, "declared SEC User-Agent with contact unavailable in environment"
        if self.checkpoint.sec_requests_total >= self.sec_budget_limit:
            return False, "SEC request budget exhausted"
        return True, "ready"

    def reference(self, url: str) -> dict | None:
        """Return a manifest reference only for bytes that still match their checksum."""
        if getattr(self, "_manifest_size", -1) != len(self.manifest["files"]):
            self._latest_entries = {e["url"]: e for e in self.manifest["files"]}
            self._manifest_size = len(self.manifest["files"])
        entry = self._latest_entries.get(url)
        if entry is None:
            return None
        path = self.cache_dir / entry["cache_file"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(self.cache_dir.resolve())
            or not path.is_file()
            or digest(path.read_bytes()) != entry["sha256"]
        ):
            raise ValueError("Cached response missing or checksum mismatch; preserve evidence")
        return dict(entry)

    def referenced_bytes(self, ref: dict) -> bytes:
        """Read only the exact successful referenced response; never issue a request."""
        actual = self.reference(ref["url"])
        if not actual or actual["sha256"] != ref["sha256"] or actual["status_code"] != 200:
            raise ValueError("Unsupported source reference")
        raw = (self.cache_dir / actual["cache_file"]).read_bytes()
        encoding = actual.get("content_encoding", "identity")
        if encoding not in ("identity", "gzip"):
            raise ValueError("Unsupported source encoding")
        return gzip.decompress(raw) if encoding == "gzip" else raw

    def store_response(
        self, url: str, provider: str, raw: bytes, status: int, encoding: str = "identity"
    ) -> None:
        """Keep successful and failed HTTP response bodies as immutable evidence."""
        filename = f"{provider}_{digest(url.encode())}_{digest(raw)}.raw"
        path = self.cache_dir / filename
        if path.exists() and (path.is_symlink() or path.read_bytes() != raw):
            raise ValueError("Immutable cache collision")
        if not path.exists():
            atomic_bytes(path, raw, preserve_on_failure=True)
        self.manifest["files"].append(
            {
                "url": url,
                "cache_file": filename,
                "sha256": digest(raw),
                "content_encoding": encoding,
                "status_code": status,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        atomic_json(self.manifest_file, self.manifest)

    def fetch(self, url: str, provider: str = "sec", refresh: bool = False) -> bytes | None:
        """Fetch raw bytes with one initial attempt plus three counted retries."""
        parts = urllib.parse.urlsplit(url)
        allowed = SEC_HOSTS if provider == "sec" else YF_HOSTS if provider == "yf" else set()
        if (
            parts.scheme != "https"
            or parts.hostname not in allowed
            or parts.username
            or parts.password
        ):
            raise ValueError("Unregistered evidence URL")
        if parts.port not in (None, 443):
            raise ValueError("Unregistered evidence port")
        entry = self.reference(url)
        if entry and not refresh:
            if entry.get("status_code") != 200:
                return None
            raw = (self.cache_dir / entry["cache_file"]).read_bytes()
            encoding = entry.get("content_encoding", "identity")
            if encoding not in ("identity", "gzip"):
                return None
            try:
                return gzip.decompress(raw) if encoding == "gzip" else raw
            except OSError:
                return None
        available, _ = self.is_network_available()
        if not available:
            return None
        counter = "sec_requests_total" if provider == "sec" else "yf_requests_total"
        limit = self.sec_budget_limit if provider == "sec" else self.yf_budget_limit
        prior_attempts = 0 if refresh else sum(r["url"] == url for r in self.checkpoint.request_log)
        for attempt in range(prior_attempts, 4):
            if not self.within_deadline() or getattr(self.checkpoint, counter) >= limit:
                return None
            if provider == "sec":
                time.sleep(max(0.0, 0.2 - (time.monotonic() - self.last_sec_req_time)))
            if not self.within_deadline():
                return None
            setattr(self.checkpoint, counter, getattr(self.checkpoint, counter) + 1)
            log = {"url": url, "provider": provider, "attempt": attempt + 1, "outcome": "reserved"}
            self.checkpoint.request_log.append(log)
            self.save_checkpoint()
            # An empty manifest distinguishes a failed first request from lost accounting.
            atomic_json(self.manifest_file, self.manifest)
            self.last_sec_req_time = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with self.opener.open(
                    request,
                    timeout=min(
                        30,
                        max(1, MAX_RUNTIME - (self.base_elapsed + time.monotonic() - self.started)),
                    ),
                ) as response:
                    status = response.status
                    if status in (403, 429):
                        raise urllib.error.HTTPError(url, status, "stopped", {}, None)
                    if status != 200:
                        raise OSError("Non-success evidence response")
                    raw = response.read()
                    encoding = response.headers.get("Content-Encoding", "identity")
                self.store_response(url, provider, raw, status, encoding)
                if encoding not in ("identity", "gzip"):
                    raise ValueError("Unsupported response encoding")
                decoded = gzip.decompress(raw) if encoding == "gzip" else raw
                log["outcome"] = "success"
                log["sha256"] = digest(raw)
                self.save_checkpoint()
                return decoded
            except urllib.error.HTTPError as exc:
                log["outcome"] = f"HTTP {exc.code}"
                if exc.code in (403, 429):
                    self.checkpoint.stopped_due_to_rate_limit = True
                    self.checkpoint.stopped_reason = f"HTTP {exc.code}"
                    self.save_checkpoint()
                try:
                    self.store_response(url, provider, exc.read() if exc.fp else b"", exc.code)
                except OSError:
                    log["response_body"] = "unavailable; accounting preserved"
                if exc.code in (403, 429):
                    return None
            except (OSError, TimeoutError, ValueError):
                log["outcome"] = "response unavailable/invalid"
            self.save_checkpoint()
            if attempt < 3:
                time.sleep(
                    min(
                        2**attempt,
                        max(0, MAX_RUNTIME - (self.base_elapsed + time.monotonic() - self.started)),
                    )
                )
        return None

    def fetch_sec_url(self, url: str) -> str | None:
        """Decode SEC evidence without substituting invalid UTF-8."""
        raw = self.fetch(url)
        try:
            return raw.decode("utf-8") if raw is not None else None
        except UnicodeDecodeError:
            return None


def filing_rows(block: dict) -> list[dict]:
    """Preserve 4/A, dates, acceptance times and filing identities from SEC columns."""
    required = ("accessionNumber", "form", "filingDate", "primaryDocument")
    if not all(isinstance(block.get(k), list) for k in required):
        raise ValueError("Incomplete SEC filing columns")
    n = len(block["form"])
    if any(len(block[k]) != n for k in required):
        raise ValueError("Mismatched SEC filing columns")
    result = []
    for i in range(n):
        result.append(
            {
                "accession": block["accessionNumber"][i],
                "form": block["form"][i],
                "filing_date": block["filingDate"][i],
                "primary_document": block["primaryDocument"][i],
                "acceptance": block.get("acceptanceDateTime", [None] * n)[i],
                "report_date": block.get("reportDate", [None] * n)[i],
                "items": block.get("items", [None] * n)[i],
            }
        )
    return result


def enumerate_sec(client: SecEdgarClient, cik: str) -> dict:
    """Enumerate recent and every historical submissions page; partial is never complete."""
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    result: dict = {
        "cik": str(int(cik)),
        "filings": [],
        "complete": False,
        "source_refs": [],
        "request_succeeded": False,
        "reason": None,
    }
    raw = client.fetch_sec_url(url)
    if raw is None:
        result["reason"] = "submissions unavailable"
        return result
    try:
        data = json.loads(raw)
        if str(int(data["cik"])) != str(int(cik)):
            raise ValueError("Submissions issuer differs from requested issuer")
        if not isinstance(data["filings"]["files"], list):
            raise ValueError("Missing pagination inventory")
        result["filings"] = filing_rows(data["filings"]["recent"])
        result["source_refs"].append(client.reference(url))
        complete = True
        page_names = set()
        for page in data["filings"].get("files", []):
            name = page["name"]
            if not re.fullmatch(r"[A-Za-z0-9.-]+\.json", name) or name in page_names:
                raise ValueError("Unsafe SEC pagination name")
            page_names.add(name)
            first, last = pd.Timestamp(page["filingFrom"]), pd.Timestamp(page["filingTo"])
            if pd.isna(first) or pd.isna(last) or first > last:
                raise ValueError("Invalid historical page interval")
            page_url = f"https://data.sec.gov/submissions/{name}"
            older = client.fetch_sec_url(page_url)
            if older is None:
                complete = False
                continue
            rows = filing_rows(json.loads(older))
            if any(not page["filingFrom"] <= r["filing_date"] <= page["filingTo"] for r in rows):
                raise ValueError("Historical page interval mismatch")
            result["filings"].extend(rows)
            result["source_refs"].append(client.reference(page_url))
        by_key: dict = {}
        for row in result["filings"]:
            key = row["accession"]
            if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", key):
                raise ValueError("Invalid inventory accession")
            if pd.isna(pd.Timestamp(row["filing_date"])):
                raise ValueError("Missing inventory filing date")
            if key in by_key and by_key[key] != row:
                complete = False
                result["reason"] = "conflicting inventory records"
            by_key[key] = row
        result["filings"] = sorted(
            by_key.values(), key=lambda r: (r["filing_date"], r["accession"])
        )
        result["complete"] = complete
        result["request_succeeded"] = True
        dates = [row["filing_date"] for row in result["filings"] if row.get("filing_date")]
        result["start"] = min(dates) if dates else None
        observed = [pd.Timestamp(ref["timestamp"]).date() for ref in result["source_refs"]]
        result["end"] = min(observed).isoformat() if observed else None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        result["reason"] = "unparseable SEC inventory"
    return result


def reviewed_fields(record: dict, fields: list[str], client: SecEdgarClient, linkage: dict) -> bool:
    """Validate field-specific review receipts against exact extracts and identity/period scope.

    A receipt records a human interpretation, not a machine certificate of its truth.
    Unreviewed annotations and reviews of a different value or scope cannot authorize facts.
    """
    try:
        refs = {(r["url"], r["sha256"]) for r in record["source_refs"]}
        for path in fields:
            value: Any = record
            for part in path.split("."):
                value = value[part]
            matches = [r for r in record.get("field_reviews", []) if r.get("field") == path]
            if len(matches) != 1:
                return False
            review = matches[0]
            ref = review["source_ref"]
            if (
                (ref["url"], ref["sha256"]) not in refs
                or review["value"] != value
                or type(review["value"]) is not type(value)
                or review["linkage"] != linkage
                or not linkage
                or not isinstance(review["reviewer"], str)
                or not review["reviewer"].strip()
                or not isinstance(review["interpretation"], str)
                or not review["interpretation"].strip()
                or pd.isna(pd.Timestamp(review["reviewed_at"]))
                or type(review["byte_start"]) is not int
                or type(review["byte_end"]) is not int
            ):
                return False
            raw = client.referenced_bytes(ref)
            start, end = review["byte_start"], review["byte_end"]
            if not 0 <= start < end <= len(raw) or not review["extract"].strip():
                return False
            if raw[start:end].decode("utf-8") != review["extract"]:
                return False
            # Interpretation cannot attach an unrelated quote to a different identity/period.
            context = {k: v for k, v in linkage.items() if k not in ("route", "representation")}
            if any(str(value) not in review["extract"] for value in context.values()):
                return False
            if type(value) in (str, int, float) and not path.endswith(".assessment"):
                if str(value) not in review["extract"]:
                    return False
        return True
    except (KeyError, ValueError, TypeError, AttributeError, OSError, UnicodeError):
        return False


class ReferencedSources:
    """Replay parsers exclusively from a record's declared response set, without transport."""

    def __init__(self, client: SecEdgarClient, refs: list[dict]) -> None:
        self.client = client
        self.refs = {r["url"]: r for r in refs}

    def reference(self, url: str) -> dict | None:
        return self.refs.get(url)

    def fetch(self, url: str, provider: str = "sec") -> bytes | None:
        ref = self.refs.get(url)
        return self.client.referenced_bytes(ref) if ref else None

    def fetch_sec_url(self, url: str) -> str | None:
        raw = self.fetch(url)
        return raw.decode("utf-8") if raw is not None else None


def filing_urls(cik: str, filing: dict) -> tuple[str, str]:
    """Build validated archive document and full-header URLs."""
    accession = filing["accession"]
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise ValueError("Invalid accession")
    document = Path(filing["primary_document"]).name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", document):
        raise ValueError("Invalid primary document")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    return f"{base}/{document}", f"{base}/{accession}.txt"


def xml_value(root: ET.Element, path: str) -> str | None:
    """Read scalar/value XML encodings without fabricating absent values."""
    element = root.find(path)
    if element is None:
        return None
    return (element.findtext("value") or element.text or "").strip() or None


def parse_form4(xml: str, filing: dict) -> dict:
    """Inventory all owners and raw lots; never collapse legitimate key collisions."""
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("DTD/entity declarations are not accepted")
    root = ET.fromstring(xml)
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    if root.tag != "ownershipDocument":
        raise ValueError("Not an ownership document")
    owners = []
    for node in root.findall("reportingOwner"):
        owners.append(
            {
                "cik": xml_value(node, "reportingOwnerId/rptOwnerCik"),
                "name": xml_value(node, "reportingOwnerId/rptOwnerName"),
                "roles": {
                    k: xml_value(node, f"reportingOwnerRelationship/{k}")
                    for k in (
                        "isDirector",
                        "isOfficer",
                        "isTenPercentOwner",
                        "isOther",
                        "officerTitle",
                    )
                },
            }
        )
    plan = xml_value(root, "aff10b5One")
    if plan is None:
        plan_state = "ABSENT"
    elif plan.lower() in ("1", "true"):
        plan_state = "PRESENT-TRUE"
    elif plan.lower() in ("0", "false"):
        plan_state = "PRESENT-FALSE"
    else:
        plan_state = "UNPARSEABLE"
    lots = []
    for i, node in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction")):
        lots.append(
            {
                "lot_id": f"{filing['accession']}:{i}",
                "ordinal": i,
                "symbol": xml_value(root, "issuer/issuerTradingSymbol"),
                "issuer_cik": xml_value(root, "issuer/issuerCik"),
                "owners": [o["cik"] for o in owners],
                "owner_inventory": owners,
                "transaction_date": xml_value(node, "transactionDate"),
                "transaction_code": xml_value(node, "transactionCoding/transactionCode"),
                "shares": xml_value(node, "transactionAmounts/transactionShares"),
                "acquired_disposed_code": xml_value(
                    node, "transactionAmounts/transactionAcquiredDisposedCode"
                ),
                "security_class": xml_value(node, "securityTitle"),
                "ownership": xml_value(node, "ownershipNature/directOrIndirectOwnership"),
                "plan_state": plan_state,
                "filing_date": filing["filing_date"],
                "accession": filing["accession"],
                "form": filing["form"],
                "acceptance": filing.get("acceptance"),
                "raw_lot_sha256": digest(ET.tostring(node)),
            }
        )
    issuer_cik = xml_value(root, "issuer/issuerCik")
    schema_issues = []
    if not issuer_cik or not issuer_cik.isdigit():
        schema_issues.append("required issuer CIK missing/invalid")
    if not owners or any(not o["cik"] or not o["cik"].isdigit() for o in owners):
        schema_issues.append("required reporting owner CIK missing/invalid")
    if any(not lot["transaction_code"] for lot in lots):
        schema_issues.append("required transaction code missing")
    for owner in owners:
        if owner["cik"] and owner["cik"].isdigit():
            owner["cik"] = str(int(owner["cik"]))
    for lot in lots:
        lot["owners"] = [o["cik"] for o in owners]
    return {
        "schema_issues": schema_issues,
        "issuer_cik": issuer_cik,
        "symbol": xml_value(root, "issuer/issuerTradingSymbol"),
        "owners": owners,
        "lots": lots,
        "plan_state": plan_state,
        "original_submission_date": xml_value(root, "dateOfOriginalSubmission"),
        "footnotes": ["".join(n.itertext()) for n in root.findall("footnotes/footnote")],
        "derivative_lots": len(root.findall("derivativeTable/derivativeTransaction")),
    }


def parse_filing_header(header: str, accession: str) -> dict:
    """Read header dates only when its accession matches the requested filing."""
    identity = re.search(r"(?:<ACCESSION-NUMBER>|ACCESSION NUMBER:\s*)(\d{10}-\d{2}-\d{6})", header)
    if not identity or identity[1] != accession:
        return {"reason": "header accession absent or mismatched"}
    accepted = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", header)
    filed = re.search(r"(?:<FILED-AS-OF-DATE>|FILED AS OF DATE:\s*)(\d{8})", header)
    return {
        "header_acceptance": dt.datetime.strptime(accepted[1], "%Y%m%d%H%M%S").isoformat()
        if accepted
        else None,
        "header_filing_date": dt.datetime.strptime(filed[1], "%Y%m%d").date().isoformat()
        if filed
        else None,
    }


def collect_filing(client: SecEdgarClient, cik: str, filing: dict) -> dict:
    """Collect a Form 4 plus independent acceptance header for reconciliation."""
    result = dict(filing, cik=str(cik), raw_complete=False, source_refs=[])
    try:
        document_url, header_url = filing_urls(cik, filing)
        xml = client.fetch_sec_url(document_url)
        if xml is not None:
            result.update(parse_form4(xml, filing))
            result["source_refs"].append(client.reference(document_url))
            if str(result.get("issuer_cik", "")).lstrip("0") != str(cik).lstrip("0"):
                result["schema_issues"].append("XML issuer differs from requested issuer")
            result["raw_complete"] = not result["schema_issues"]
        header = client.fetch_sec_url(header_url)
        if header is not None:
            result.update(parse_filing_header(header, filing["accession"]))
            result["source_refs"].append(client.reference(header_url))
    except (ValueError, ET.ParseError, KeyError):
        result["reason"] = "invalid ownership/header evidence"
    return result


def collect_yahoo(client: SecEdgarClient, symbol: str) -> dict:
    """Use the installed yfinance earnings-calendar scrape protocol under counted transport."""
    result: dict = {
        "symbol": symbol,
        "records": [],
        "complete": False,
        "request_succeeded": False,
        "source_refs": [],
    }
    seen = set()
    for offset in range(0, 10000, 100):
        query = urllib.parse.urlencode({"symbol": symbol, "offset": offset, "size": 100})
        url = "https://finance.yahoo.com/calendar/earnings?" + query
        raw = client.fetch(url, "yf")
        if raw is None:
            result["reason"] = "Yahoo endpoint unavailable"
            break
        try:
            soup = BeautifulSoup(raw, "html.parser")
            table = soup.find("table")
            if table is None:
                result["reason"] = "no verified earnings inventory"
                break
            frame = pd.read_html(StringIO(str(table)), na_values=["-"])[0]
            result["source_refs"].append(client.reference(url))
            result["request_succeeded"] = True
            if "Earnings Date" not in frame:
                raise ValueError("Unknown Yahoo schema")
            for row in frame.to_dict("records"):
                stated = str(row["Earnings Date"])
                text = re.sub(r" (EDT|EST|America/New_York)$", "", stated)
                timestamp = pd.Timestamp(
                    dt.datetime.strptime(text, "%B %d, %Y at %I %p")
                ).tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
                key = timestamp.isoformat()
                if key in seen:
                    raise ValueError("Repeated Yahoo pagination")
                seen.add(key)
                if pd.isna(row.get("Reported EPS")):
                    continue
                result["records"].append(
                    {
                        "earnings_date": str(timestamp.date()),
                        "announcement_readings": [timestamp.isoformat()],
                        "eps_estimate": None
                        if pd.isna(row.get("EPS Estimate"))
                        else row.get("EPS Estimate"),
                        "eps_reported": row.get("Reported EPS"),
                        "surprise_pct": None
                        if pd.isna(row.get("Surprise (%)", row.get("Surprise(%)")))
                        else row.get("Surprise (%)", row.get("Surprise(%)")),
                        "source_refs": [client.reference(url)],
                    }
                )
            if len(frame) < 100:
                result["complete"] = True
                break
        except (ValueError, KeyError, TypeError, IndexError):
            result["reason"] = "Yahoo schema unverified"
            break
    return result


def collect_release_candidates(client: SecEdgarClient, cik: str, filing: dict) -> dict:
    """Cache 8-K index/exhibits and issuer/fiscal metadata; unmatched releases stay explicit."""
    result = {
        "cik": str(cik),
        "accession": filing["accession"],
        "source_refs": [],
        "acceptance": filing.get("acceptance"),
        "filing_date": filing["filing_date"],
        "releases": [],
        "complete": False,
    }
    try:
        document_url, header_url = filing_urls(cik, filing)
        base = document_url.rsplit("/", 1)[0]
        index_url = base + "/index.json"
        raw_index = client.fetch_sec_url(index_url)
        if raw_index is None:
            return result
        index = json.loads(raw_index)
        result["source_refs"].append(client.reference(index_url))
        header = client.fetch_sec_url(header_url)
        if header is not None:
            result.update(parse_filing_header(header, filing["accession"]))
            result["source_refs"].append(client.reference(header_url))
        primary = client.fetch_sec_url(document_url)
        if primary is None:
            return result
        result["source_refs"].append(client.reference(document_url))
        soup = BeautifulSoup(primary, "html.parser")
        known_files = {r["name"] for r in index["directory"]["item"]}
        names = []
        for link in soup.find_all("a", href=True):
            name = Path(urllib.parse.urlsplit(link["href"]).path).name
            if (
                name in known_files
                and name.lower().endswith((".htm", ".html"))
                and (re.search(r"(?:ex|exhibit)[-_]?99", name, re.I) or "99" in link.get_text())
            ):
                names.append(name)
        complete = True
        for name in dict.fromkeys(names):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                continue
            url = base + "/" + name
            html = client.fetch_sec_url(url)
            if html is None:
                complete = False
                continue
            document = BeautifulSoup(html, "html.parser")
            timestamp = document.find("meta", attrs={"property": "article:published_time"})
            published = timestamp.get("content") if timestamp else None
            # An explicitly labelled fiscal period avoids inventing quarter matches.
            fiscal = document.find("meta", attrs={"name": "fiscal-period-end"})
            fiscal_period = fiscal.get("content") if fiscal else None
            result["releases"].append(
                {
                    "document_url": url,
                    "issuer_cik": str(cik),
                    "fiscal_period": fiscal_period,
                    "announcement_readings": [published] if published else [],
                    "source_refs": [client.reference(url)],
                    "time_semantics": "Publication metadata; original release unconfirmed",
                }
            )
            result["source_refs"].append(client.reference(url))
        result["complete"] = complete
    except (ValueError, KeyError, TypeError):
        result["reason"] = "release index/exhibit evidence unavailable or unmatched"
    return result


def collect_identity_candidates(client: SecEdgarClient, symbols: list[str]) -> dict:
    """Discover current issuer candidates; never promote today's map to historical identity."""
    url = "https://www.sec.gov/files/company_tickers.json"
    result: dict = {
        "candidates": [],
        "unresolved_symbols": sorted(symbols),
        "source_refs": [],
        "historical_identity_verified": False,
    }
    raw = client.fetch_sec_url(url)
    if raw is None:
        result["reason"] = "current candidate map unavailable; no historical identity inference"
        return result
    try:
        rows = json.loads(raw)
        candidates: dict[str, list[dict]] = {}
        for row in rows.values():
            ticker = normalize_yf_ticker(row["ticker"])
            candidates.setdefault(ticker, []).append(row)
        for symbol in sorted(symbols):
            for row in candidates.get(normalize_yf_ticker(symbol), []):
                result["candidates"].append(
                    {
                        "symbol": symbol,
                        "cik": str(int(row["cik_str"])),
                        "current_name": row["title"],
                    }
                )
        found = {r["symbol"] for r in result["candidates"]}
        result["unresolved_symbols"] = sorted(set(symbols) - found)
        result["source_refs"] = [client.reference(url)]
    except (ValueError, KeyError, TypeError, AttributeError):
        result["reason"] = "candidate identity map unparseable"
    return result
