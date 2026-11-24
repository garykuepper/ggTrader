"""Tests for the polite, caching SEC EDGAR downloader."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import urllib.error
import urllib.request
import urllib.response
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import sec_client
from sec_client import (
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
DEFAULT_UA = "ggTrader Study contact@example.test"


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
    min_interval: float = 0.25,
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


# --- Brief-specified tests (7 core cases) ---


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
    client, opener = make_client(tmp_path, body=b"x", min_interval=0.25)
    client.get(URL + "1")
    client.get(URL + "2")
    assert opener.sleeps and opener.sleeps[0] == pytest.approx(0.25, abs=0.01)


def test_other_host_is_refused(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, body=b"x")
    with pytest.raises(ValueError):
        client.get("https://example.com/a")


def test_user_agent_header_is_sent(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        body=b"x",
        user_agent="Project Name contact@example.test",
    )
    client.get(URL)
    assert opener.last_headers["User-Agent"] == "Project Name contact@example.test"


# --- Design point (a): atomic cache writes & metadata schema ---


def test_atomic_write_failure_leaves_no_body_or_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(tmp_path, body=b"partial-data")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("simulated disk failure during rename")

    monkeypatch.setattr(sec_client.os, "replace", boom)

    with pytest.raises(OSError, match="simulated disk failure"):
        client.get(URL)

    assert list(tmp_path.glob("*.body")) == []
    assert list(tmp_path.glob("*.meta.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_meta_json_contains_only_expected_fields_and_no_headers(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        body=b'{"ok": true}',
        user_agent="Private Header Check contact@example.test",
    )
    client.get(URL)

    meta_files = list(tmp_path.glob("*.meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert set(meta.keys()) == {"url", "sha256", "fetched_at", "status"}
    assert meta["url"] == URL
    assert meta["sha256"] == hashlib.sha256(b'{"ok": true}').hexdigest()
    assert meta["status"] == 200
    assert "contact@example.test" not in meta_files[0].read_text(encoding="utf-8")


# --- Design point (b): cache checksum verification and refetch counting ---


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
    # Repaired cache now serves from disk without hitting the opener again.
    assert client.get(URL) == b"good-v2"
    assert opener.calls == 2


def test_orphan_body_without_meta_triggers_refetch(tmp_path: Path) -> None:
    client, opener = make_client(tmp_path, body=b"fresh-body")
    key = hashlib.sha256(URL.encode("utf-8")).hexdigest()
    (tmp_path / f"{key}.body").write_bytes(b"orphan")

    assert client.get(URL) == b"fresh-body"
    assert opener.calls == 1
    assert client.corrupt_cache_refetches == 1


# --- Design point (c): manifest() and write_sha256sums() ---


def test_manifest_sorted_by_url_and_write_sha256sums_format(tmp_path: Path) -> None:
    url_b = "https://www.sec.gov/Archives/edgar/full-index/2020/QTR2/master.zip"
    url_a = "https://data.sec.gov/submissions/CIK0000000001.json"
    client, _ = make_client(
        tmp_path,
        responses=[FakeResponse(b"zip-bytes"), FakeResponse(b"json-bytes")],
    )
    # Fetch in reverse alphabetical order so sorting is verified explicitly.
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


def test_manifest_raises_on_corrupted_body(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, body=b"intact")
    client.get(URL)
    next(tmp_path.glob("*.body")).write_bytes(b"tampered")
    with pytest.raises(SecFetchError, match="Checksum mismatch"):
        client.manifest()


def test_manifest_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path / "not-created-yet", body=b"x")
    assert client.manifest() == {}


# --- Design point (d): redirect host guard & URL validation ---


def test_redirect_to_disallowed_host_via_geturl_is_refused(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        body=b"hijacked",
        status=200,
        final_url="https://evil.example.test/landing",
    )
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert opener.calls == 1
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_to_disallowed_host_via_url_attr_is_refused(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        body=b"hijacked",
        status=200,
        final_url="https://evil.example.test/landing",
        expose_geturl=False,
    )
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert opener.calls == 1
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_between_allowed_sec_hosts_is_permitted(tmp_path: Path) -> None:
    assert ALLOWED_HOSTS == {"www.sec.gov", "data.sec.gov"}
    client, _ = make_client(
        tmp_path,
        body=b'{"redirected": true}',
        status=200,
        final_url="https://www.sec.gov/submissions/CIK0000320193.json",
    )
    assert client.get(URL) == b'{"redirected": true}'


def test_non_https_scheme_is_refused(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, body=b"x")
    with pytest.raises(ValueError, match="https"):
        client.get("http://www.sec.gov/Archives/edgar/full-index/2020/QTR1/master.zip")


# --- Design point (e): retries, exponential backoff, network errors, 404 ---


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


def test_transient_http_errors_recover_on_third_attempt(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            make_http_error(503),
            make_http_error(429),
            FakeResponse(b"recovered"),
        ],
    )
    assert client.get(URL) == b"recovered"
    assert opener.calls == 3
    assert opener.sleeps == [pytest.approx(2.0), pytest.approx(4.0)]
    assert len(list(tmp_path.glob("*.body"))) == 1


def test_network_error_retries_three_attempts_then_raises(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            urllib.error.URLError("temporary dns failure"),
            TimeoutError("socket timed out"),
            ConnectionResetError("peer reset"),
        ],
    )
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


# --- HTML detection and User-Agent edge cases ---


def test_doctype_html_refused_for_xml_but_allowed_for_html_urls(tmp_path: Path) -> None:
    html_body = b"  \n<!DOCTYPE html><html><body>hello</body></html>"

    client_xml, _ = make_client(tmp_path / "xml", status=200, body=html_body)
    with pytest.raises(SecFetchError):
        client_xml.get("https://www.sec.gov/Archives/edgar/data/1/primary_doc.xml")

    for allowed_url in (
        "https://www.sec.gov/Archives/edgar/data/1/index.htm",
        "https://www.sec.gov/Archives/edgar/data/1/index.html",
        "https://www.sec.gov/Archives/edgar/data/1/",
    ):
        client_ok, _ = make_client(tmp_path / "html", status=200, body=html_body)
        assert client_ok.get(allowed_url) == html_body


def test_load_user_agent_strips_and_client_rejects_blank(tmp_path: Path) -> None:
    assert load_user_agent({"SEC_USER_AGENT": "  MyTool contact@example.test  "}) == (
        "MyTool contact@example.test"
    )
    with pytest.raises(SecUserAgentMissing):
        SecClient("   ", tmp_path)


# --- Manual redirect validation, pacing, and default opener ---


def test_default_opener_disables_automatic_redirects(tmp_path: Path) -> None:
    """Verify the default OpenerDirector surfaces 302 as HTTPError instead of following it."""
    opener = sec_client._build_default_opener()
    redirect_handlers = [
        h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)
    ]
    assert len(redirect_handlers) == 1
    assert isinstance(redirect_handlers[0], sec_client._NoRedirectHandler)

    # Confirm OpenerDirector with a fake HTTPS transport stops at 302 without issuing a second call.
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
        sec_client._NoRedirectHandler(),
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        test_opener.open(urllib.request.Request(URL))
    assert exc_info.value.code == 302
    assert exc_info.value.headers.get("Location") == "https://evil.example/steal"
    assert calls == [URL]
    exc_info.value.close()

    # Wire the same no-redirect opener into SecClient and confirm it refuses before hop 2.
    calls.clear()
    client = SecClient(user_agent=DEFAULT_UA, cache_dir=tmp_path, opener=test_opener.open)
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert calls == [URL]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_to_www_sec_gov_followed_and_cached_under_original_url(
    tmp_path: Path,
    status: int,
) -> None:
    target = "https://www.sec.gov/Archives/edgar/data/320193/index.json"
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=status, headers={"Location": target}),
            FakeResponse(status=200, body=b'{"ok": 1}'),
        ],
    )
    assert client.get(URL) == b'{"ok": 1}'
    assert opener.requested_urls == [URL, target]
    assert opener.calls == 2

    # Cached under the original URL only; second get hits cache with no extra network call.
    orig_key = hashlib.sha256(URL.encode("utf-8")).hexdigest()
    target_key = hashlib.sha256(target.encode("utf-8")).hexdigest()
    assert (tmp_path / f"{orig_key}.body").exists()
    assert (tmp_path / f"{orig_key}.meta.json").exists()
    assert not (tmp_path / f"{target_key}.body").exists()
    assert client.manifest() == {URL: hashlib.sha256(b'{"ok": 1}').hexdigest()}

    assert client.get(URL) == b'{"ok": 1}'
    assert opener.calls == 2


def test_redirect_via_http_error_and_relative_location_followed(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            make_http_error(
                302, url=URL, headers={"Location": "/submissions/CIK0000320193_v2.json"}
            ),
            FakeResponse(status=200, body=b'{"v": 2}'),
        ],
    )
    assert client.get(URL) == b'{"v": 2}'
    assert opener.requested_urls == [
        URL,
        "https://data.sec.gov/submissions/CIK0000320193_v2.json",
    ]


def test_redirect_to_http_www_sec_gov_refused_without_second_request(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=302, headers={"Location": "http://www.sec.gov/plain.json"}),
            FakeResponse(status=200, body=b"should-never-be-fetched"),
        ],
    )
    with pytest.raises(ValueError, match="https"):
        client.get(URL)
    assert opener.calls == 1
    assert opener.requested_urls == [URL]
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_to_evil_host_refused_without_second_request(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=302, headers={"Location": "https://evil.example/steal"}),
            FakeResponse(status=200, body=b"should-never-be-fetched"),
        ],
    )
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert opener.calls == 1
    assert opener.requested_urls == [URL]
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_to_lookalike_host_refused_without_second_request(tmp_path: Path) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(
                status=301,
                headers={"Location": "https://www.sec.gov.evil.example/submissions/1.json"},
            ),
            FakeResponse(status=200, body=b"should-never-be-fetched"),
        ],
    )
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert opener.calls == 1
    assert list(tmp_path.glob("*.body")) == []


@pytest.mark.parametrize(
    "bad_location",
    [
        "https://user:pass@www.sec.gov/submissions/1.json",
        "https://user@www.sec.gov/submissions/1.json",
        "https://@www.sec.gov/submissions/1.json",
    ],
)
def test_redirect_with_userinfo_refused_without_second_request(
    tmp_path: Path,
    bad_location: str,
) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=302, headers={"Location": bad_location}),
            FakeResponse(status=200, body=b"should-never-be-fetched"),
        ],
    )
    with pytest.raises(ValueError, match="credentials"):
        client.get(URL)
    assert opener.calls == 1
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_chain_allowed_then_forbidden_refused_at_forbidden_hop(tmp_path: Path) -> None:
    hop1 = "https://www.sec.gov/Archives/edgar/hop1.json"
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=302, headers={"Location": hop1}),
            FakeResponse(status=302, headers={"Location": "https://evil.example/hop2"}),
            FakeResponse(status=200, body=b"should-never-be-fetched"),
        ],
    )
    with pytest.raises(ValueError, match="allowed"):
        client.get(URL)
    assert opener.calls == 2
    assert opener.requested_urls == [URL, hop1]
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_four_hops_raises_sec_fetch_error(tmp_path: Path) -> None:
    hops = [f"https://www.sec.gov/Archives/edgar/hop{i}.json" for i in range(1, 5)]
    client, opener = make_client(
        tmp_path,
        responses=[
            FakeResponse(status=302, headers={"Location": hops[0]}),
            FakeResponse(status=302, headers={"Location": hops[1]}),
            FakeResponse(status=302, headers={"Location": hops[2]}),
            FakeResponse(status=302, headers={"Location": hops[3]}),
            FakeResponse(status=200, body=b"too-far"),
        ],
    )
    with pytest.raises(SecFetchError, match="Too many redirects"):
        client.get(URL)
    # Four requests were sent (initial + 3 allowed hops); the 4th redirect is refused before hop4.
    assert opener.calls == 4
    assert opener.requested_urls == [URL, hops[0], hops[1], hops[2]]
    assert list(tmp_path.glob("*.body")) == []


def test_redirect_three_hops_succeeds_and_paces_each_hop_with_user_agent(tmp_path: Path) -> None:
    ua = "Redirect Pacing Study contact@example.test"
    hops = [f"https://www.sec.gov/Archives/edgar/hop{i}.json" for i in range(1, 4)]
    client, opener = make_client(
        tmp_path,
        min_interval=0.25,
        user_agent=ua,
        responses=[
            FakeResponse(status=302, headers={"Location": hops[0]}),
            FakeResponse(status=307, headers={"Location": hops[1]}),
            FakeResponse(status=308, headers={"Location": hops[2]}),
            FakeResponse(status=200, body=b"final-body"),
        ],
    )
    assert client.get(URL) == b"final-body"
    assert opener.calls == 4
    assert opener.requested_urls == [URL, hops[0], hops[1], hops[2]]
    assert opener.sleeps == [pytest.approx(0.25), pytest.approx(0.25), pytest.approx(0.25)]
    assert len(opener.headers_per_call) == 4
    assert all(hdrs["User-Agent"] == ua for hdrs in opener.headers_per_call)


# --- Response size ceiling (max_bytes) ---


def test_oversized_chunked_body_without_content_length_rejected_and_not_cached(
    tmp_path: Path,
) -> None:
    resp = FakeResponse(status=200, chunks=[b"aaaa", b"bbbb", b"cccc", b"dddd"])
    client, opener = make_client(tmp_path, responses=[resp], max_bytes=10)
    with pytest.raises(SecFetchError, match="response too large"):
        client.get(URL)
    assert opener.calls == 1
    assert resp.closed is True
    # Stopped reading as soon as 11 bytes were seen, without draining the fourth chunk.
    assert resp.read_calls == 3
    assert list(tmp_path.glob("*.body")) == []


def test_oversized_content_length_refused_early_without_reading_or_caching(tmp_path: Path) -> None:
    resp = FakeResponse(
        status=200,
        body=b"x" * 20,
        headers={"Content-Length": "20"},
    )
    client, opener = make_client(tmp_path, responses=[resp], max_bytes=10)
    with pytest.raises(SecFetchError, match="response too large"):
        client.get(URL)
    assert opener.calls == 1
    assert resp.read_calls == 0
    assert resp.closed is True
    assert list(tmp_path.glob("*.body")) == []


def test_body_exactly_at_max_bytes_is_accepted_and_cached(tmp_path: Path) -> None:
    payload = b"0123456789"
    resp = FakeResponse(
        status=200,
        chunks=[b"0123", b"4567", b"89"],
        headers={"Content-Length": str(len(payload))},
    )
    client, opener = make_client(tmp_path, responses=[resp], max_bytes=10)
    assert client.get(URL) == payload
    assert client.get(URL) == payload
    assert opener.calls == 1
    assert len(list(tmp_path.glob("*.body"))) == 1


# --- Additional coverage: XML/ZIP payloads, cache-hit pacing, first-URL validation, meta write ---


def test_primary_doc_xml_body_accepted_and_cached(tmp_path: Path) -> None:
    xml_url = "https://www.sec.gov/Archives/edgar/data/320193/0001/primary_doc.xml"
    xml_body = (
        b'<?xml version="1.0" encoding="UTF-8"?><edgarSubmission><form>25</form></edgarSubmission>'
    )
    client, opener = make_client(tmp_path, status=200, body=xml_body)
    assert client.get(xml_url) == xml_body
    assert client.get(xml_url) == xml_body
    assert opener.calls == 1
    assert len(list(tmp_path.glob("*.body"))) == 1


def test_html_error_page_for_zip_url_is_refused(tmp_path: Path) -> None:
    zip_url = "https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/master.zip"
    client, _ = make_client(
        tmp_path,
        status=200,
        body=b"<html><body>Request Rate Threshold Exceeded</body></html>",
    )
    with pytest.raises(SecFetchError, match="HTML"):
        client.get(zip_url)
    assert list(tmp_path.glob("*.body")) == []


def test_cache_hit_does_not_sleep_or_reset_pacing_clock(tmp_path: Path) -> None:
    url_1 = URL + "?part=1"
    url_2 = URL + "?part=2"
    client, opener = make_client(
        tmp_path,
        min_interval=0.25,
        responses=[FakeResponse(b"one"), FakeResponse(b"two")],
    )
    # First real request at t=100.0; no sleep.
    assert client.get(url_1) == b"one"
    assert opener.sleeps == []

    # Advance 0.10s and serve a cache hit: must not sleep and must not move _last_request_at.
    opener.advance_clock(0.10)
    assert client.get(url_1) == b"one"
    assert opener.calls == 1
    assert opener.sleeps == []

    # Advance 0.05s (0.15s total since first real request); second real request waits only 0.10s.
    opener.advance_clock(0.05)
    assert client.get(url_2) == b"two"
    assert opener.calls == 2
    assert opener.sleeps == [pytest.approx(0.10, abs=1e-9)]


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://www.sec.gov.evil.example/submissions/CIK0000320193.json",
        "https://data.sec.gov.evil.example/submissions/CIK0000320193.json",
        "https://user:pass@www.sec.gov/submissions/CIK0000320193.json",
        "https://user@data.sec.gov/submissions/CIK0000320193.json",
        "https://www.sec.gov@evil.example/submissions/CIK0000320193.json",
    ],
)
def test_first_url_lookalike_host_and_userinfo_refused(tmp_path: Path, bad_url: str) -> None:
    client, opener = make_client(tmp_path, body=b"never")
    with pytest.raises(ValueError):
        client.get(bad_url)
    assert opener.calls == 0


def test_metadata_write_failure_after_body_write_leaves_no_usable_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, opener = make_client(
        tmp_path,
        responses=[FakeResponse(b"first-body"), FakeResponse(b"second-body")],
    )
    real_replace = sec_client.os.replace
    replace_calls = 0

    def fail_on_meta(src: Any, dst: Any) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated metadata write failure")
        real_replace(src, dst)

    monkeypatch.setattr(sec_client.os, "replace", fail_on_meta)

    with pytest.raises(OSError, match="simulated metadata write failure"):
        client.get(URL)

    # First replace wrote .body; second replace failed on .meta.json and cleaned up cleanly.
    assert replace_calls == 2
    assert list(tmp_path.glob("*.meta.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.body")) == []

    # Next get cannot use any partial cache entry and refetches from the opener.
    assert client.get(URL) == b"second-body"
    assert opener.calls == 2
    assert client.get(URL) == b"second-body"
    assert opener.calls == 2
