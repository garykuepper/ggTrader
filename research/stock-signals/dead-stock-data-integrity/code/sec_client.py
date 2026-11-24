"""Polite, caching SEC EDGAR downloader with fair-access rate limiting and integrity checks."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_HOSTS: set[str] = {"www.sec.gov", "data.sec.gov"}
ENV_USER_AGENT = "SEC_USER_AGENT"
DEFAULT_MIN_INTERVAL = 0.25
DEFAULT_TIMEOUT = 30.0
MAX_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 2.0
RETRYABLE_STATUSES: frozenset[int] = frozenset({403, 429})
HTML_ALLOWED_SUFFIXES = (".htm", ".html", "/")
HTML_PREFIXES = (b"<html", b"<!doctype html")
SHA256SUMS_FILENAME = "SHA256SUMS.txt"
_HASH_CHUNK_BYTES = 65536


class SecUserAgentMissing(RuntimeError):
    """Raised when the SEC fair-access User-Agent is absent or blank."""


class SecFetchError(RuntimeError):
    """Raised when an SEC EDGAR fetch fails after retries or returns an invalid payload."""


class SecNotFound(SecFetchError):
    """Raised immediately on HTTP 404 so callers can distinguish missing filings from failures."""


def load_user_agent(environ: Mapping[str, str]) -> str:
    """Read and validate ``SEC_USER_AGENT`` from ``environ`` without any default fallback.

    Args:
        environ: Environment mapping (for example ``os.environ``).

    Returns:
        Stripped non-empty User-Agent string suitable for SEC EDGAR fair-access headers.

    Raises:
        SecUserAgentMissing: If ``SEC_USER_AGENT`` is absent or contains only whitespace.
    """
    raw = environ.get(ENV_USER_AGENT, "")
    cleaned = raw.strip() if isinstance(raw, str) else ""
    if not cleaned:
        raise SecUserAgentMissing(
            f"{ENV_USER_AGENT} must be set to a non-empty SEC fair-access User-Agent string."
        )
    return cleaned


def _url_key(url: str) -> str:
    """Return the hex SHA-256 digest used to name cache files for ``url``."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Compute the hex SHA-256 digest of ``path`` by streaming fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to a temp file in the same directory and atomically rename into ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _validate_url(url: str) -> None:
    """Ensure ``url`` uses HTTPS and targets one of ``ALLOWED_HOSTS``."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"SEC URL must use https scheme: {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"SEC URL must not embed credentials: {url!r}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(
            f"Host {host!r} is not in allowed SEC hosts {sorted(ALLOWED_HOSTS)}: {url!r}"
        )


def _extract_final_url(response: Any) -> str | None:
    """Read the final post-redirect URL from a urllib response or fake opener response."""
    geturl = getattr(response, "geturl", None)
    if callable(geturl):
        try:
            url = geturl()
        except AttributeError:
            url = None
        if isinstance(url, str) and url:
            return url
    url_attr = getattr(response, "url", None)
    if isinstance(url_attr, str) and url_attr:
        return url_attr
    return None


def _extract_status(response: Any) -> int:
    """Read the HTTP status code from a urllib response or fake opener response."""
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        code = getcode()
        if isinstance(code, int):
            return code
    raise SecFetchError("Response object did not expose an integer HTTP status code.")


def _is_retryable_status(status: int) -> bool:
    """Return True for HTTP statuses that represent rate limits or transient server errors."""
    return status in RETRYABLE_STATUSES or 500 <= status < 600


def _is_unexpected_html(url: str, body: bytes) -> bool:
    """Return True when a non-HTML URL returns an HTML page such as an SEC soft rate-limit block."""
    path = urllib.parse.urlsplit(url).path.lower()
    if path == "" or path.endswith(HTML_ALLOWED_SUFFIXES):
        return False
    # Strip optional UTF-8 BOM and leading whitespace before checking the HTML tag prefix.
    prefix = body[:256].lstrip(b" \t\r\n\xef\xbb\xbf").lower()
    return any(prefix.startswith(tag) for tag in HTML_PREFIXES)


def write_sha256sums(cache_dir: Path) -> Path:
    """Write ``SHA256SUMS.txt`` (``<sha256>  <filename>``) for every cached body in ``cache_dir``.

    Args:
        cache_dir: Directory containing ``<sha256(url)>.body`` files.

    Returns:
        Path to the written ``SHA256SUMS.txt`` file.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    body_files = sorted(directory.glob("*.body"), key=lambda p: p.name)
    lines = [f"{_sha256_file(path)}  {path.name}\n" for path in body_files]
    payload = "".join(lines).encode("utf-8")
    return _atomic_write_bytes(directory / SHA256SUMS_FILENAME, payload)


class SecClient:
    """Rate-limited, integrity-checked caching client for SEC EDGAR endpoints.

    Attributes:
        corrupt_cache_refetches: Count of cached entries that failed verification and were
            refetched from the opener.
        cache_hits: Count of requests served from a verified cache file.
        network_fetches: Count of requests fulfilled by the opener and written to cache.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        cleaned_ua = user_agent.strip() if isinstance(user_agent, str) else ""
        if not cleaned_ua:
            raise SecUserAgentMissing("SecClient requires a non-empty SEC fair-access User-Agent.")
        if min_interval < 0:
            raise ValueError(f"min_interval must be non-negative, got {min_interval!r}")

        self._user_agent = cleaned_ua
        self._cache_dir = Path(cache_dir)
        self._min_interval = float(min_interval)
        self._opener = opener
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

        self.corrupt_cache_refetches = 0
        self.cache_hits = 0
        self.network_fetches = 0

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = _url_key(url)
        return self._cache_dir / f"{key}.body", self._cache_dir / f"{key}.meta.json"

    def _read_verified_cache(self, url: str) -> bytes | None:
        body_path, meta_path = self._cache_paths(url)
        has_body = body_path.exists()
        has_meta = meta_path.exists()
        if not has_body and not has_meta:
            return None
        if not has_body or not has_meta:
            # An orphan .body or .meta.json signals an interrupted write; refetch cleanly.
            self.corrupt_cache_refetches += 1
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_sha = meta.get("sha256")
            recorded_url = meta.get("url")
        except (OSError, ValueError, AttributeError):
            self.corrupt_cache_refetches += 1
            return None

        if not isinstance(expected_sha, str) or recorded_url != url:
            self.corrupt_cache_refetches += 1
            return None

        try:
            body = body_path.read_bytes()
        except OSError:
            self.corrupt_cache_refetches += 1
            return None

        actual_sha = hashlib.sha256(body).hexdigest()
        if actual_sha != expected_sha:
            self.corrupt_cache_refetches += 1
            return None
        return body

    def _write_cache(self, url: str, status: int, body: bytes) -> None:
        body_path, meta_path = self._cache_paths(url)
        body_sha = hashlib.sha256(body).hexdigest()
        # Record provenance fields only; never write User-Agent or request headers to disk.
        meta = {
            "url": url,
            "sha256": body_sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
        # Write the body atomically before metadata so a crash never leaves a half-written cache.
        _atomic_write_bytes(body_path, body)
        _atomic_write_bytes(meta_path, meta_bytes)

    def _throttle(self) -> None:
        if self._last_request_at is not None and self._min_interval > 0:
            elapsed = self._clock() - self._last_request_at
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)

    def _call_opener(self, request: urllib.request.Request) -> Any:
        # Real urlopen needs a timeout so hung sockets cannot stall a run; fake openers take 1 arg.
        if self._opener is urllib.request.urlopen:
            return self._opener(request, timeout=DEFAULT_TIMEOUT)
        return self._opener(request)

    def _attempt_once(self, url: str, request: urllib.request.Request) -> tuple[int, bytes]:
        try:
            response = self._call_opener(request)
        except urllib.error.HTTPError as exc:
            try:
                final_url = _extract_final_url(exc)
                if final_url is not None:
                    _validate_url(final_url)
                if exc.code == 404:
                    raise SecNotFound(f"SEC URL not found (HTTP 404): {url}") from exc
                if _is_retryable_status(exc.code):
                    raise _RetryableFetchError(f"HTTP {exc.code} fetching {url}") from exc
                raise SecFetchError(f"HTTP {exc.code} fetching {url}") from exc
            finally:
                close = getattr(exc, "close", None)
                if callable(close):
                    close()
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise _RetryableFetchError(f"Network error fetching {url}: {exc}") from exc

        try:
            final_url = _extract_final_url(response)
            if final_url is not None:
                _validate_url(final_url)
            status = _extract_status(response)
            if status == 404:
                raise SecNotFound(f"SEC URL not found (HTTP 404): {url}")
            if _is_retryable_status(status):
                raise _RetryableFetchError(f"HTTP {status} fetching {url}")
            if status != 200:
                raise SecFetchError(f"Unexpected HTTP {status} fetching {url}")
            try:
                body = response.read()
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                raise _RetryableFetchError(f"Stream read error fetching {url}: {exc}") from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if _is_unexpected_html(url, body):
            raise SecFetchError(f"Refused HTML response for non-HTML SEC URL: {url}")
        return status, body

    def get(self, url: str) -> bytes:
        """Fetch ``url`` from cache if valid, otherwise download with fair-access guards.

        Args:
            url: HTTPS URL on an allowed SEC host (``www.sec.gov`` or ``data.sec.gov``).

        Returns:
            Raw response bytes as received from SEC EDGAR.

        Raises:
            ValueError: If ``url`` (or a post-redirect final URL) uses a non-HTTPS scheme or a
                host outside ``ALLOWED_HOSTS``.
            SecNotFound: Immediately on HTTP 404 (subclass of ``SecFetchError``).
            SecFetchError: On non-200 failure, unexpected HTML payload, or after retry exhaustion.
        """
        _validate_url(url)
        cached = self._read_verified_cache(url)
        if cached is not None:
            self.cache_hits += 1
            return cached

        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt == 1:
                self._throttle()
            try:
                status, body = self._attempt_once(url, request)
            except _RetryableFetchError as exc:
                last_error = exc
                self._last_request_at = self._clock()
                if attempt < MAX_ATTEMPTS:
                    delay = max(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), self._min_interval)
                    self._sleep(delay)
                continue
            except BaseException:
                self._last_request_at = self._clock()
                raise
            else:
                self._last_request_at = self._clock()
                self._write_cache(url, status, body)
                self.network_fetches += 1
                return body

        raise SecFetchError(
            f"Failed to fetch {url} after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def manifest(self) -> dict[str, str]:
        """Return ``{url: sha256}`` for every cached body, sorted by URL.

        Verifies every cached body on disk against its recorded ``.meta.json`` digest so the
        returned manifest reflects intact files only.

        Returns:
            Dictionary mapping each cached URL to its hex SHA-256 digest, sorted by URL.

        Raises:
            SecFetchError: If a cached ``.body`` file is missing valid metadata or fails its
                checksum check.
        """
        if not self._cache_dir.exists():
            return {}

        entries: dict[str, str] = {}
        for body_path in self._cache_dir.glob("*.body"):
            meta_path = body_path.with_suffix(".meta.json")
            if not meta_path.exists():
                raise SecFetchError(f"Cached body missing metadata file: {body_path.name}")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                url = meta["url"]
                recorded_sha = meta["sha256"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SecFetchError(f"Invalid metadata file {meta_path.name}: {exc}") from exc

            actual_sha = _sha256_file(body_path)
            if actual_sha != recorded_sha:
                raise SecFetchError(
                    f"Checksum mismatch for cached body {body_path.name} ({url}): "
                    f"expected {recorded_sha}, got {actual_sha}"
                )
            entries[url] = actual_sha

        return dict(sorted(entries.items()))


class _RetryableFetchError(Exception):
    """Internal sentinel for transient HTTP or transport errors eligible for backoff."""
