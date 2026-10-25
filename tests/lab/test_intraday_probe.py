"""Metadata-only Stage 0A probe contract tests, with no external access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ggTrader.lab import intraday_probe as probe


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, payload: object, status: int = 200, headers: dict | None = None) -> None:
        self.content = json.dumps(payload).encode()
        self.status_code = status
        self.headers = headers or {}

    def close(self) -> None:
        pass


class Session:
    def __init__(self, responses: list) -> None:
        self.responses = iter(responses)
        self.calls = []

    def get(self, url: str, **kwargs: object) -> Response:
        self.calls.append((url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def client(responses: list) -> tuple[probe.ProbeClient, Session, Clock]:
    session = Session(responses)
    clock = Clock()
    return probe.ProbeClient(session, clock=clock.monotonic, sleep=clock.sleep), session, clock


def bars(rows: list | None = None, token: str | None = None) -> Response:
    return Response({"bars": {"SPY": rows or []}, "next_page_token": token})


def test_month_selection_and_dst_bounds() -> None:
    assert probe.sample_months() == [
        "2016-01",
        "2016-07",
        "2017-02",
        "2017-09",
        "2018-03",
        "2018-10",
        "2019-05",
        "2019-12",
        "2020-06",
        "2021-01",
        "2021-08",
        "2022-03",
        "2022-09",
        "2023-04",
        "2023-11",
        "2024-06",
        "2024-12",
        "2025-07",
        "2026-02",
        "2026-09",
    ]
    assert probe.utc_bounds("2016-01-04", "09:30", "16:00") == (
        "2016-01-04T14:30:00Z",
        "2016-01-04T21:00:00Z",
    )
    assert probe.utc_bounds("2026-09-01", "09:30", "16:00") == (
        "2026-09-01T13:30:00Z",
        "2026-09-01T20:00:00Z",
    )


def test_pagination_does_not_treat_short_page_as_completion() -> None:
    api, session, clock = client(
        [
            bars([{"t": "2016-01-04T14:30:00Z", "v": 7, "c": 123}], "secret-token"),
            bars([{"t": "2016-01-04T14:31:00Z", "v": 8, "o": 456}]),
        ]
    )
    result = api.request("bars", {"feed": "sip"})
    assert result["outcome"] == "nonempty_success"
    assert result["records"] == [
        {"t": "2016-01-04T14:30:00Z", "v": 7},
        {"t": "2016-01-04T14:31:00Z", "v": 8},
    ]
    assert result["pagination"] == {"pages": 2, "complete": True}
    assert "secret-token" not in json.dumps(result)
    assert session.calls[1][1]["params"]["page_token"] == "secret-token"
    assert clock.waits == [0.65]


def test_repeated_token_and_page_limit_are_incomplete() -> None:
    api, _, _ = client([bars(token="repeat"), bars(token="repeat")])
    assert api.request("bars", {})["outcome"] == "incomplete_pagination"
    api, session, _ = client([bars(token=str(i)) for i in range(21)])
    result = api.request("bars", {})
    assert result["outcome"] == "incomplete_pagination"
    assert len(session.calls) == 20


@pytest.mark.parametrize(
    "status,outcome",
    [
        (401, "authentication_error"),
        (403, "entitlement_error"),
        (422, "invalid_request"),
        (302, "invalid_request"),
    ],
)
def test_errors_are_not_empty_archives_or_retried(status: int, outcome: str) -> None:
    api, session, _ = client([Response({"message": "secret-password price=999"}, status)])
    result = api.request("bars", {})
    assert result["outcome"] == outcome
    assert len(session.calls) == 1
    assert "secret-password" not in json.dumps(result)
    assert result["attempts"][0]["body_sha256"]
    api, _, _ = client([bars()])
    assert api.request("bars", {})["outcome"] == "empty_success"


def test_bounded_retries_waits_and_deferred() -> None:
    api, session, clock = client(
        [Response({}, 429, {"Retry-After": "3"}), Response({}, 503), bars()]
    )
    assert api.request("bars", {})["outcome"] == "empty_success"
    assert clock.waits == [3, 2]
    assert len(session.calls) == 3
    api, session, clock = client([Response({}, 429, {"Retry-After": "31"})])
    assert api.request("bars", {})["outcome"] == "deferred"
    assert len(session.calls) == 1
    assert clock.waits == []


@pytest.mark.parametrize(
    "failure,outcome",
    [
        (Response({}, 429), "throttled_exhausted"),
        (Response({}, 503), "transient_exhausted"),
        (requests.ConnectionError("SECRET REMOTE URL"), "transient_exhausted"),
        (requests.Timeout("SECRET REMOTE URL"), "transient_exhausted"),
    ],
)
def test_exhaustion_and_transport_errors_are_sanitized(failure: object, outcome: str) -> None:
    api, session, _ = client([failure, failure, failure])
    result = api.request("bars", {})
    assert result["outcome"] == outcome
    assert len(session.calls) == 3
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"bars": {"SPY": [{"t": "not-a-time", "v": 1}]}, "next_page_token": None},
        {"bars": {"SPY": [{"t": "2016-01-04T14:30:00Z"}]}, "next_page_token": None},
        {"bars": {"SPY": []}},
        {"bars": {"SPY": []}, "next_page_token": 42},
    ],
)
def test_missing_required_metadata_is_schema_error(payload: object) -> None:
    api, _, _ = client([Response(payload)])
    assert api.request("bars", {})["outcome"] == "schema_error"


def test_auction_lists_venue_and_calendar_are_sanitized() -> None:
    api, _, _ = client(
        [
            Response(
                {
                    "auctions": {
                        "SPY": [
                            {
                                "d": "2016-01-04",
                                "o": [
                                    {
                                        "t": "2016-01-04T14:30:00Z",
                                        "x": "P",
                                        "p": 100,
                                        "c": "A",
                                        "s": 5,
                                    }
                                ],
                                "c": [{"t": "2016-01-04T21:00:00Z", "x": "N", "p": 200, "c": "C"}],
                            }
                        ]
                    },
                    "next_page_token": None,
                    "currency": "USD",
                    "secret": "SECRET",
                }
            )
        ]
    )
    result = api.request("auctions", {})
    assert result["records"] == [
        {"d": "2016-01-04", "t": "2016-01-04T14:30:00Z", "x": "P", "type": "opening"},
        {"d": "2016-01-04", "t": "2016-01-04T21:00:00Z", "x": "N", "type": "closing"},
    ]
    assert "SECRET" not in json.dumps(result)
    api, _, _ = client(
        [
            Response(
                [
                    {"date": "2026-07-03", "open": "09:30", "close": "13:00", "price": 50},
                    {"date": "2026-07-06", "open": "09:30", "close": "16:00", "secret": "SECRET"},
                ]
            )
        ]
    )
    result = api.request("calendar", {"start": "2026-07-01", "end": "2026-07-10"})
    assert result["records"][0] == {"date": "2026-07-03", "open": "09:30", "close": "13:00"}
    assert probe.select_session("2026-07", result["records"]) == "2026-07-06"


def test_grid_preserves_duplicates_missing_misaligned_and_outside() -> None:
    records = [
        {"t": "2016-01-04T14:30:00Z", "v": 2},
        {"t": "2016-01-04T14:30:00Z", "v": 3},
        {"t": "2016-01-04T14:31:00.000000001Z", "v": 4},
        {"t": "2016-01-04T21:00:00Z", "v": 5},
    ]
    result = probe.grid_summary(records, "2016-01-04T14:30:00Z", "2016-01-04T14:33:00Z")
    assert result["row_count"] == 4
    assert result["unique_timestamps"] == 3
    assert result["duplicate_count"] == 1
    assert result["missing"] == ["2016-01-04T14:31:00Z", "2016-01-04T14:32:00Z"]
    assert result["misaligned"] == ["2016-01-04T14:31:00.000000001Z"]
    assert result["out_of_grid"] == ["2016-01-04T21:00:00Z"]
    assert result["volume_sum"] == 14


class CompleteProbe:
    """A provider boundary with complete minute grids and official calendars."""

    def __init__(
        self,
        output: Path,
        *,
        missing_calendar: bool = False,
        sip_gap: bool = False,
        fail_iex: bool = False,
        fail_auctions: bool = False,
    ) -> None:
        self.output = output
        self.missing_calendar = missing_calendar
        self.sip_gap = sip_gap
        self.fail_iex = fail_iex
        self.fail_auctions = fail_auctions
        self.calls = []

    def request(self, kind: str, params: dict) -> dict:
        from datetime import datetime, timedelta

        self.calls.append((kind, params))
        if kind != "calendar":
            checkpoint = json.loads(self.output.read_text())
            assert checkpoint["manifest_sha256"]
            assert len(checkpoint["manifest"]["selected_dates"]) == 20
            assert checkpoint["status"] == "in_progress"
        records = []
        outcome = "empty_success"
        if kind == "calendar" and not self.missing_calendar:
            records = [{"date": params["start"], "open": "09:30", "close": "16:00"}]
        elif kind == "bars":
            start = datetime.fromisoformat(params["start"])
            end = datetime.fromisoformat(params["end"])
            while start < end:
                records.append({"t": start.isoformat().replace("+00:00", "Z"), "v": 1})
                start += timedelta(minutes=1)
            if self.sip_gap and params["feed"] == "sip":
                records = records[1:]
            if self.fail_iex and params["feed"] == "iex":
                records = []
                outcome = "entitlement_error"
        if self.fail_auctions and kind == "auctions":
            outcome = "entitlement_error"
        if records:
            outcome = "nonempty_success"
        return {
            "endpoint": probe.ENDPOINTS[kind],
            "parameters": params,
            "outcome": outcome,
            "records": records,
            "attempts": [],
            "pagination": {"pages": 1, "complete": True},
        }


def test_run_freezes_manifest_and_uses_exact_requests_before_conclusion(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    api = CompleteProbe(output)
    result = probe.run_probe(output, client=api)
    assert result["status"] == "completed"
    assert result["conclusion"] == "sample_checks_passed"
    assert result["auction_conclusion"] == "auction_branch_unresolved"
    assert result["full_window_verified"] is False
    assert result["strategy_evaluated"] is False
    assert len(api.calls) == 91
    assert all(kind == "calendar" for kind, _ in api.calls[:20])
    assert result["feed_completeness"]["sip"]["required_checks"] == 22
    assert result["feed_completeness"]["sip"]["failed_required_checks"] == []
    bar_calls = [params for kind, params in api.calls if kind == "bars"]
    assert len(bar_calls) == 48
    assert all(
        params["symbols"] == "SPY"
        and params["timeframe"] == "1Min"
        and params["adjustment"] == "raw"
        and params["asof"] == "-"
        and params["limit"] == 10000
        and params["sort"] == "asc"
        for params in bar_calls
    )
    assert json.loads(output.read_text()) == result
    assert result["manifest_sha256"] == probe.canonical_hash(result["manifest"])
    assert set(result["document_hashes"]) == {"strategy_brief", "contract"}


def test_calendar_failure_stops_market_requests_and_is_not_empty_archive(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    api = CompleteProbe(output, missing_calendar=True)
    result = probe.run_probe(output, client=api)
    assert result["conclusion"] == "unresolved"
    assert result["reasons"] == ["calendar_session_unresolved:2016-01"]
    assert len(api.calls) == 1
    assert result["manifest_sha256"] is None


def test_primary_gap_fails_but_iex_and_auctions_are_limitations(tmp_path: Path) -> None:
    output = tmp_path / "gap.json"
    result = probe.run_probe(output, client=CompleteProbe(output, sip_gap=True))
    assert result["conclusion"] == "unresolved"
    assert len(result["feed_completeness"]["sip"]["failed_required_checks"]) == 22
    output = tmp_path / "sensitivity.json"
    result = probe.run_probe(
        output, client=CompleteProbe(output, fail_iex=True, fail_auctions=True)
    )
    assert result["conclusion"] == "sample_checks_passed"
    assert result["auction_conclusion"] == "auction_branch_unresolved"
    assert result["feed_completeness"]["iex"]["limitations"]


def test_existing_output_is_untouched(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    output.write_text("original")
    api = CompleteProbe(output)
    with pytest.raises(FileExistsError):
        probe.run_probe(output, client=api)
    assert output.read_text() == "original"
    assert api.calls == []


def test_local_failure_preserves_valid_checkpoint_without_exception_text(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"

    class BrokenProbe(CompleteProbe):
        def request(self, kind: str, params: dict) -> dict:
            if len(self.calls) == 21:
                raise RuntimeError("secret FROM EXTERNAL LIBRARY")
            return super().request(kind, params)

    result = probe.run_probe(output, client=BrokenProbe(output))
    assert result["status"] == "in_progress"
    assert result["conclusion"] == "unresolved"
    assert result["local_failure"] == "unexpected_local_failure"
    checkpoint = json.loads(output.read_text())
    assert checkpoint["status"] == "in_progress"
    assert len(checkpoint["requests"]) == 21
    assert "secret" not in json.dumps(checkpoint)


def test_atomic_checkpoint_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "probe.json"
    output.write_text('{"status":"in_progress"}')

    def fail_replace(*args: object) -> None:
        raise OSError("local-secret-path")

    monkeypatch.setattr(probe.os, "replace", fail_replace)
    with pytest.raises(OSError):
        probe.atomic_checkpoint(output, {"status": "completed"})
    assert json.loads(output.read_text()) == {"status": "in_progress"}
    assert list(tmp_path.iterdir()) == [output]


def test_total_request_deadline_includes_body_read(monkeypatch: pytest.MonkeyPatch) -> None:
    class StalledResponse(Response):
        @property
        def content(self) -> bytes:
            raise probe.RequestDeadline()

        @content.setter
        def content(self, value: bytes) -> None:
            pass

    api, session, _ = client([StalledResponse({}) for _ in range(3)])
    result = api.request("bars", {})
    assert result["outcome"] == "transient_exhausted"
    assert all(attempt["transport_error"] == "timeout" for attempt in result["attempts"])
    assert all(
        call[1]["timeout"] == 20 and call[1]["allow_redirects"] is False for call in session.calls
    )


def test_help_has_no_credentials_network_or_artifacts(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if not key.startswith("APCA_")}
    result = subprocess.run(
        [sys.executable, str(root / "scripts/intraday_coverage_probe.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["bars", "auctions"])
def test_documented_empty_symbol_map_is_empty_success(kind: str) -> None:
    api, _, _ = client([Response({kind: {}, "next_page_token": None})])
    assert api.request(kind, {})["outcome"] == "empty_success"


def test_outside_grid_duplicates_and_misalignment_are_separate_limitations(tmp_path: Path) -> None:
    class ExtraEndpointBars(CompleteProbe):
        def request(self, kind: str, params: dict) -> dict:
            result = super().request(kind, params)
            if kind == "bars":
                result["records"].extend(
                    [
                        {"t": params["end"], "v": 3},
                        {"t": params["end"], "v": 4},
                        {"t": params["end"].replace("Z", ".000000001Z"), "v": 5},
                    ]
                )
            return result

    output = tmp_path / "extra.json"
    result = probe.run_probe(output, client=ExtraEndpointBars(output))
    assert result["conclusion"] == "sample_checks_passed"
    bars = next(row for row in result["requests"] if row["kind"] == "bars")
    assert bars["grid"]["duplicate_count"] == 1
    assert len(bars["grid"]["out_of_grid"]) == 3
    assert len(bars["grid"]["misaligned"]) == 1


def test_auction_missing_close_stays_unresolved() -> None:
    artifact = {
        "requests": [
            {
                "kind": "auctions",
                "id": str(i),
                "role": "sample",
                "outcome": "nonempty_success",
                "records": [{"type": "opening"}],
            }
            for i in range(23)
        ]
    }
    probe.conclude(artifact)
    assert artifact["auction_conclusion"] == "auction_branch_unresolved"
    assert "missing_closing" in artifact["auction_limitations"][0]


def test_rate_reset_wait_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.time, "time", lambda: 1000)
    api, _, clock = client([Response({}, 429, {"X-RateLimit-Reset": "1004"}), bars()])
    assert api.request("bars", {})["outcome"] == "empty_success"
    assert clock.waits == [4]
    api, _, _ = client([Response({}, 503, {"X-RateLimit-Reset": "1031"})])
    assert api.request("bars", {})["outcome"] == "deferred"


def test_alarm_bounds_total_body_time_and_restores_handler() -> None:
    import signal
    import time

    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(probe.RequestDeadline):
        with probe.request_deadline(0.01):
            time.sleep(1)
    assert signal.getsignal(signal.SIGALRM) is previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


@pytest.mark.parametrize(
    "daily",
    [
        {"d": "2016-01-04", "o": []},
        {"d": "2016-01-04", "o": [], "c": None},
        {"d": "2016-01-04", "o": [], "c": [{"t": "2016-01-04T21:00:00Z"}]},
        {"d": "secret", "o": [], "c": []},
    ],
)
def test_malformed_auction_metadata_is_not_empty_success(daily: dict) -> None:
    api, _, _ = client([Response({"auctions": {"SPY": [daily]}, "next_page_token": None})])
    result = api.request("auctions", {})
    assert result["outcome"] == "schema_error"
    assert "secret" not in json.dumps(result)


def test_environment_credentials_never_enter_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AuthSession(Session):
        def __init__(self) -> None:
            super().__init__([Response({"message": "SECRET CREDENTIALS", "p": 123}, 403)])
            self.headers = {}
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = AuthSession()
    monkeypatch.setenv("APCA_API_KEY_ID", "SECRET KEY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "SECRET VALUE")
    monkeypatch.setattr(probe.requests, "Session", lambda: session)
    output = tmp_path / "credential-check.json"
    result = probe.run_probe(output)
    assert result["conclusion"] == "unresolved"
    assert result["requests"][0]["outcome"] == "entitlement_error"
    assert session.trust_env is False
    assert session.closed
    assert "SECRET" not in output.read_text()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not keys(result) & {
        "p",
        "o",
        "h",
        "l",
        "c",
        "vw",
        "price",
        "headers",
        "secret",
        "APCA-API-KEY-ID",
        "APCA-API-SECRET-KEY",
        "page_token",
    }
