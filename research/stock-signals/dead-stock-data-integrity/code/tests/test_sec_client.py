"""Tests for the polite, caching SEC EDGAR downloader."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import sec_client
from sec_client import (
    ALLOWED_HOSTS,
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
    """Minimal stand-in for urllib's HTTPResponse."""

    def __init__(
        self,
        body: bytes = b"",
        status: int = 200,
        final_url: str | None = None,
        *,
        expose_geturl: bool = True,
    ) -> None:
        self._body = body
        self.status = status
        self._final_url = final_url
        self._expose_geturl = expose_geturl
        if final_url is not None and not expose_geturl:
            self.url = final_url
        self.closed = False

    def read(self) -> bytes:
        return self._body

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
    """Scripted fake URL opener that records calls, headers, and injected sleeps."""

    def __init__(
        self,
        responses: list[FakeResponse | BaseException],
        sleeps: list[float],
    ) -> None:
        self._responses = list(responses)
        self.sleeps = sleeps
        self.calls = 0
        self.last_headers: dict[str, str] = {}
        self.requested_urls: list[str] = []

    def __call__(self, request: urllib.request.Request) -> FakeResponse:
        self.calls += 1
        self.requested_urls.append(request.full_url)
        headers = dict(request.header_items())
        ua = request.get_header("User-agent")
        if ua is not None:
            headers["User-Agent"] = ua
        self.last_headers = headers

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
    min_interval: float = 0.25,
    user_agent: str = DEFAULT_UA,
    final_url: str | None = None,
    expose_geturl: bool = True,
    responses: list[FakeResponse | BaseException] | None = None,
    start_time: float = 100.0,
) -> tuple[SecClient, FakeOpener]:
    """Create a SecClient wired to an instant fake clock, sleep recorder, and fake opener."""
    now = [start_time]
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    script = (
        responses
        if responses is not None
        else [
            FakeResponse(body=body, status=status, final_url=final_url, expose_geturl=expose_geturl)
        ]
    )
    opener = FakeOpener(script, sleeps)
    client = SecClient(
        user_agent=user_agent,
        cache_dir=cache_dir,
        min_interval=min_interval,
        opener=opener,
        sleep=fake_sleep,
        clock=fake_clock,
    )
    return client, opener


def make_http_error(code: int, url: str = URL, msg: str = "error") -> urllib.error.HTTPError:
    """Build a urllib HTTPError with an empty response stream."""
    return urllib.error.HTTPError(url=url, code=code, msg=msg, hdrs=None, fp=io.BytesIO(b""))


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
