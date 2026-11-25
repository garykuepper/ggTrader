"""Tests for the byte-identical SEC client copy and early-window fetch wrapper."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
import urllib.response
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import early_sec
import ef_sec_client
import pytest
from early_sec import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MIN_INTERVAL,
    ENV_CONTACT_PATH,
    EarlySecClient,
    load_contact_line,
)
from ef_sec_client import (
    ALLOWED_HOSTS,
    DEFAULT_MAX_BYTES,
    SecClient,
    SecFetchError,
    SecNotFound,
    SecUserAgentMissing,
    load_user_agent,
    write_sha256sums,
)

pytestmark = pytest.mark.study

URL = "https://data.sec.gov/submissions/CIK0000320193.json"
DEFAULT_UA = "Test Agent test@example.invalid"


class FakeResponse:
    """Minimal stand-in for urllib's HTTPResponse with headers and chunked read support."""

    def __init__(
        self,
        body: bytes = b"",
        status: int = 200,
        final_url: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        expose_geturl: bool = True,
    ) -> None:
        self._stream = io.BytesIO(body)
        self._chunks: deque[bytes] | None = deque(chunks) if chunks is not None else None
        self.status = status
        self.headers: dict[str, str] = dict(headers) if headers is not None else {}
        self._final_url = final_url
        self._expose_geturl = expose_geturl
        if final_url is not None and not expose_geturl:
            self.url = final_url
        self.closed = False
        self.read_calls = 0

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if self._chunks is not None:
            if not self._chunks:
                return b""
            chunk = self._chunks.popleft()
            if size is not None and size >= 0 and len(chunk) > size:
                self._chunks.appendleft(chunk[size:])
                return chunk[:size]
            return chunk
        return self._stream.read(size)

    def geturl(self) -> str | None:
        if not self._expose_geturl:
            raise AttributeError("geturl not exposed")
        return self._final_url

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeOpener:
    """Scripted fake URL opener that records calls, per-call headers, and injected sleeps."""

    def __init__(
        self,
        responses: list[FakeResponse | BaseException],
        sleeps: list[float],
        advance_clock: Callable[[float], None],
    ) -> None:
        self._responses = list(responses)
        self.sleeps = sleeps
        self.advance_clock = advance_clock
        self.calls = 0
        self.last_headers: dict[str, str] = {}
        self.headers_per_call: list[dict[str, str]] = []
        self.requested_urls: list[str] = []

    def __call__(self, request: urllib.request.Request) -> FakeResponse:
        self.calls += 1
        self.requested_urls.append(request.full_url)
        headers = dict(request.header_items())
        ua = request.get_header("User-agent")
        if ua is not None:
            headers["User-Agent"] = ua
        self.last_headers = headers
        self.headers_per_call.append(headers)

        idx = min(self.calls - 1, len(self._responses) - 1)
        outcome = self._responses[idx]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(
    cache_dir: Path,
    *,
    body: bytes = b"",
    status: int = 200,
    headers: dict[str, str] | None = None,
    min_interval: float = 0.2,
    user_agent: str = DEFAULT_UA,
    final_url: str | None = None,
    expose_geturl: bool = True,
    responses: list[FakeResponse | BaseException] | None = None,
    start_time: float = 100.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[SecClient, FakeOpener]:
    """Create a SecClient wired to an instant fake clock, sleep recorder, and fake opener."""
    now = [start_time]
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now[0]

    def advance_clock(seconds: float) -> None:
        now[0] += seconds

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    script = (
        responses
        if responses is not None
        else [
            FakeResponse(
                body=body,
                status=status,
                final_url=final_url,
                headers=headers,
                expose_geturl=expose_geturl,
            )
        ]
    )
    opener = FakeOpener(script, sleeps, advance_clock)
    client = SecClient(
        user_agent=user_agent,
        cache_dir=cache_dir,
        min_interval=min_interval,
        opener=opener,
        sleep=fake_sleep,
        clock=fake_clock,
        max_bytes=max_bytes,
    )
    return client, opener


def make_early_client(
    cache_dir: Path,
    *,
    body: bytes = b"",
    status: int = 200,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    user_agent: str = DEFAULT_UA,
    responses: list[FakeResponse | BaseException] | None = None,
    start_time: float = 100.0,
) -> tuple[EarlySecClient, FakeOpener]:
    """Create an EarlySecClient wired to a fake clock, sleep recorder, and fake opener."""
    now = [start_time]
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now[0]

    def advance_clock(seconds: float) -> None:
        now[0] += seconds

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    script = responses if responses is not None else [FakeResponse(body=body, status=status)]
    opener = FakeOpener(script, sleeps, advance_clock)
    client = EarlySecClient(
        user_agent=user_agent,
        cache_dir=cache_dir,
        min_interval=min_interval,
        opener=opener,
        sleep=fake_sleep,
        clock=fake_clock,
    )
    return client, opener


def make_http_error(
    code: int,
    url: str = URL,
    msg: str = "error",
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    """Build a urllib HTTPError with optional headers and an empty response stream."""
    hdrs = email.message.Message()
    for key, value in (headers or {}).items():
        hdrs[key] = value
    return urllib.error.HTTPError(url=url, code=code, msg=msg, hdrs=hdrs, fp=io.BytesIO(b""))


# --- Early-study wrapper tests (404 recording, 5 req/s throttle, contact file) ---


def test_early_client_defaults_match_frozen_brief() -> None:
    assert DEFAULT_CACHE_DIR == Path("/media/thesix/ggTrader-data/sec-early-window")
    assert DEFAULT_MIN_INTERVAL == pytest.approx(0.2)


def test_early_client_cache_hit_makes_no_request(tmp_path: Path) -> None:
    client, opener = make_early_client(tmp_path, body=b"cached-payload")
    assert client.get(URL) == b"cached-payload"
    assert client.get(URL) == b"cached-payload"
    assert opener.calls == 1
    assert client.network_fetches == 1
    assert client.cache_hits == 1


def test_early_client_records_404_and_does_not_repeat_request(tmp_path: Path) -> None:
    client, opener = make_early_client(tmp_path, status=404, body=b"not found")
    with pytest.raises(SecNotFound):
        client.get(URL)
    assert opener.calls == 1
    assert client.not_found_recorded == 1
    assert client.not_found_hits == 0

    # Second request on the same client returns SecNotFound without hitting opener or sleeping.
    with pytest.raises(SecNotFound):
        client.get(URL)
    assert opener.calls == 1
    assert opener.sleeps == []
    assert client.not_found_hits == 1

    # A fresh EarlySecClient pointing at the same cache directory also honours the recorded 404.
    fresh_client, fresh_opener = make_early_client(tmp_path, status=200, body=b"should-not-fetch")
    with pytest.raises(SecNotFound):
        fresh_client.get(URL)
    assert fresh_opener.calls == 0
    assert fresh_client.not_found_hits == 1

    # Recorded 404 file never contains the contact line.
    notfound_files = list(tmp_path.glob("*.notfound.json"))
    assert len(notfound_files) == 1
    raw_text = notfound_files[0].read_text(encoding="utf-8")
    assert "test@example.invalid" not in raw_text
    meta = json.loads(raw_text)
    assert meta["url"] == URL
    assert meta["status"] == 404


def test_early_client_enforces_five_requests_per_second_throttle(tmp_path: Path) -> None:
    client, opener = make_early_client(tmp_path, body=b"ok")
    client.get(URL + "?a=1")
    client.get(URL + "?a=2")
    assert opener.calls == 2
    assert opener.sleeps == [pytest.approx(0.2, abs=1e-9)]


def test_contact_line_read_from_environment_path_and_never_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contact_file = tmp_path / "sec_contact"
    contact_file.write_text(f"  {DEFAULT_UA}  \n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        loaded = load_contact_line({ENV_CONTACT_PATH: str(contact_file)})

    assert loaded == DEFAULT_UA
    assert DEFAULT_UA not in caplog.text
    assert "test@example.invalid" not in caplog.text

    # Client construction and fetch with that contact line also never log or persist it.
    with caplog.at_level(logging.DEBUG):
        client, opener = make_early_client(tmp_path / "cache", body=b"ok", user_agent=loaded)
        assert client.get(URL) == b"ok"
    assert opener.last_headers["User-Agent"] == DEFAULT_UA
    assert DEFAULT_UA not in caplog.text
    assert "test@example.invalid" not in caplog.text
    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            assert b"test@example.invalid" not in path.read_bytes()


def test_missing_or_blank_contact_file_raises_without_leaking_contact(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SecUserAgentMissing):
        load_contact_line({ENV_CONTACT_PATH: str(missing)})

    blank = tmp_path / "blank_contact"
    blank.write_text("   \n\t\n", encoding="utf-8")
    with pytest.raises(SecUserAgentMissing):
        load_contact_line({ENV_CONTACT_PATH: str(blank)})


def test_early_client_ignores_corrupt_notfound_marker_and_refetches(tmp_path: Path) -> None:
    client, opener = make_early_client(tmp_path, body=b"recovered")
    key = hashlib.sha256(URL.encode("utf-8")).hexdigest()
    (tmp_path / f"{key}.notfound.json").write_text("{corrupt", encoding="utf-8")

    assert client.get(URL) == b"recovered"
    assert opener.calls == 1
    assert not (tmp_path / f"{key}.notfound.json").exists()


# --- Adapted ef_sec_client core tests ---


def test_missing_user_agent_raises() -> None:
    with pytest.raises(SecUserAgentMissing):
        load_user_agent({})
    with pytest.raises(SecUserAgentMissing):
        load_user_agent({"SEC_USER_AGENT": "  "})


def test_second_get_comes_from_cache(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, body=b"abc")
    assert client.get(URL) == b"abc" and client.get(URL) == b"abc"
    assert opener.calls == 1


def test_rate_limit_status_is_not_cached(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, status=429, body=b"slow down")
    with pytest.raises(SecFetchError):
        client.get(URL)
    assert list(tmp_path.glob("*.body")) == []


def test_html_error_page_with_200_is_not_cached(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        status=200,
        body=b"<html>Request Rate Threshold Exceeded</html>",
    )
    with pytest.raises(SecFetchError):
        client.get(URL)
    assert list(tmp_path.glob("*.body")) == []


def test_interval_between_requests_is_enforced(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, body=b"x", min_interval=0.2)
    client.get(URL + "1")
    client.get(URL + "2")
    assert opener.sleeps and opener.sleeps[0] == pytest.approx(0.2, abs=0.01)


def test_other_host_is_refused(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, body=b"x")
    with pytest.raises(ValueError):
        client.get("https://example.com/a")


def test_user_agent_header_is_sent(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, body=b"x", user_agent=DEFAULT_UA)
    client.get(URL)
    assert opener.last_headers["User-Agent"] == DEFAULT_UA


def test_atomic_write_failure_leaves_no_body_or_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(tmp_path, body=b"partial-data")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("simulated disk failure during rename")

    monkeypatch.setattr(ef_sec_client.os, "replace", boom)

    with pytest.raises(OSError, match="simulated disk failure"):
        client.get(URL)

    assert list(tmp_path.glob("*.body")) == []
    assert list(tmp_path.glob("*.meta.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_meta_json_contains_only_expected_fields_and_no_headers(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, body=b'{"ok": true}', user_agent=DEFAULT_UA)
    client.get(URL)

    meta_files = list(tmp_path.glob("*.meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert set(meta.keys()) == {"url", "sha256", "fetched_at", "status"}
    assert meta["url"] == URL
    assert meta["sha256"] == hashlib.sha256(b'{"ok": true}').hexdigest()
    assert meta["status"] == 200
    assert "test@example.invalid" not in meta_files[0].read_text(encoding="utf-8")


def test_corrupt_cached_body_triggers_refetch_and_increments_counter(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[FakeResponse(b"good-v1"), FakeResponse(b"good-v2")],
    )
    assert client.get(URL) == b"good-v1"
    assert client.corrupt_cache_refetches == 0

    body_file = next(tmp_path.glob("*.body"))
    body_file.write_bytes(b"corrupted-bytes")

    assert client.get(URL) == b"good-v2"
    assert opener.calls == 2
    assert client.corrupt_cache_refetches == 1
    assert client.get(URL) == b"good-v2"
    assert opener.calls == 2


def test_manifest_sorted_by_url_and_write_sha256sums_format(tmp_path: Path) -> None:
    url_b = "https://www.sec.gov/Archives/edgar/full-index/2001/QTR2/master.zip"
    url_a = "https://data.sec.gov/submissions/CIK0000000001.json"
    client, _ = make_client(
        tmp_path,
        responses=[FakeResponse(b"zip-bytes"), FakeResponse(b"json-bytes")],
    )
    client.get(url_b)
    client.get(url_a)

    manifest = client.manifest()
    assert list(manifest.keys()) == [url_a, url_b]
    assert manifest[url_a] == hashlib.sha256(b"json-bytes").hexdigest()
    assert manifest[url_b] == hashlib.sha256(b"zip-bytes").hexdigest()

    sums_path = write_sha256sums(tmp_path)
    assert sums_path == tmp_path / "SHA256SUMS.txt"
    lines = sums_path.read_text(encoding="utf-8").splitlines()
    expected = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
        for p in sorted(tmp_path.glob("*.body"), key=lambda path: path.name)
    ]
    assert lines == expected


def test_redirect_between_allowed_sec_hosts_and_refusal_of_disallowed_hosts(
    tmp_path: Path,
) -> None:
    assert ALLOWED_HOSTS == {"www.sec.gov", "data.sec.gov"}
    target = "https://www.sec.gov/Archives/edgar/data/320193/index.json"
    client, opener = make_client(
        tmp_path / "ok",
        responses=[
            FakeResponse(status=302, headers={"Location": target}),
            FakeResponse(status=200, body=b'{"ok": 1}'),
        ],
    )
    assert client.get(URL) == b'{"ok": 1}'
    assert opener.requested_urls == [URL, target]

    bad_client, bad_opener = make_client(
        tmp_path / "bad",
        responses=[
            FakeResponse(status=302, headers={"Location": "https://evil.example/steal"}),
            FakeResponse(status=200, body=b"never"),
        ],
    )
    with pytest.raises(ValueError, match="allowed"):
        bad_client.get(URL)
    assert bad_opener.calls == 1


@pytest.mark.parametrize("status", [429, 403, 500, 503])
def test_retryable_status_retries_three_attempts_with_doubling_wait(
    tmp_path: Path,
    status: int,
) -> None:
    client, opener = make_client(tmp_path, status=status, body=b"try later")
    with pytest.raises(SecFetchError):
        client.get(URL)
    assert opener.calls == 3
    assert opener.sleeps == [pytest.approx(2.0), pytest.approx(4.0)]
    assert list(tmp_path.glob("*.body")) == []


def test_404_raises_sec_not_found_immediately_without_retry(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, status=404, body=b"not found")
    with pytest.raises(SecNotFound) as exc_info:
        client.get(URL)
    assert isinstance(exc_info.value, SecFetchError)
    assert opener.calls == 1
    assert opener.sleeps == []
    assert list(tmp_path.glob("*.body")) == []


def test_404_http_error_raises_sec_not_found_immediately(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, responses=[make_http_error(404)])
    with pytest.raises(SecNotFound):
        client.get(URL)
    assert opener.calls == 1
    assert opener.sleeps == []


def test_load_user_agent_strips_and_client_rejects_blank(tmp_path: Path) -> None:
    assert load_user_agent({"SEC_USER_AGENT": f"  {DEFAULT_UA}  "}) == DEFAULT_UA
    with pytest.raises(SecUserAgentMissing):
        SecClient("   ", tmp_path)


def test_default_opener_disables_automatic_redirects(tmp_path: Path) -> None:
    opener = ef_sec_client._build_default_opener()
    redirect_handlers = [
        h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)
    ]
    assert len(redirect_handlers) == 1
    assert isinstance(redirect_handlers[0], ef_sec_client._NoRedirectHandler)

    calls: list[str] = []

    class _FakeHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req: urllib.request.Request) -> Any:
            calls.append(req.full_url)
            hdrs = email.message.Message()
            hdrs["Location"] = "https://evil.example/steal"
            resp = urllib.response.addinfourl(io.BytesIO(b""), hdrs, req.full_url, code=302)
            resp.msg = "Found"
            return resp

    test_opener = urllib.request.build_opener(
        _FakeHTTPSHandler(),
        ef_sec_client._NoRedirectHandler(),
    )
    client = SecClient(user_agent=DEFAULT_UA, cache_dir=tmp_path, opener=test_opener.open)
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert calls == [URL]


def test_oversized_chunked_body_rejected_and_not_cached(tmp_path: Path) -> None:
    resp = FakeResponse(status=200, chunks=[b"aaaa", b"bbbb", b"cccc", b"dddd"])
    client, opener = make_client(tmp_path, responses=[resp], max_bytes=10)
    with pytest.raises(SecFetchError, match="response too large"):
        client.get(URL)
    assert opener.calls == 1
    assert resp.closed is True
    assert resp.read_calls == 3
    assert list(tmp_path.glob("*.body")) == []


def test_early_sec_module_does_not_import_other_study() -> None:
    source = Path(early_sec.__file__).read_text(encoding="utf-8")
    assert "dead-stock-data-integrity" not in source
    assert "sec_filings" not in source
    assert "sec_name_index" not in source
