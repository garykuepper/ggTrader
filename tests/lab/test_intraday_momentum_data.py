"""Mocked collector, resume and CLI safety tests for the r8 pipeline; no network or credentials."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path

import pytest
import requests

from ggTrader.lab import intraday_momentum as core
from ggTrader.lab import intraday_momentum_data as data
from ggTrader.lab.intraday_probe import utc_bounds

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/intraday_momentum_research.py"
SECRET = "SECRET-456-do-not-log"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, payload: object, status: int = 200, headers: dict | None = None) -> None:
        self.content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status_code = status
        self.headers = headers or {}

    def close(self) -> None:
        pass


class Session:
    def __init__(self, responses: list) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict]] = []
        self.headers = {"APCA-API-KEY-ID": "KEY-123", "APCA-API-SECRET-KEY": SECRET}

    def get(self, url: str, **kwargs: object) -> Response:
        self.calls.append((url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def utc(day: str, clock: str) -> str:
    return utc_bounds(day, clock, clock)[0]


def bars(rows: list, token: str | None = None) -> Response:
    return Response({"bars": {"SPY": rows}, "next_page_token": token})


def bar(day: str, clock: str, price: float) -> dict:
    return {"t": utc(day, clock), "o": price, "h": price, "l": price, "c": price, "v": 9, "n": 1}


SPEC = {
    "id": "minute_bars:2024-12",
    "kind": "minute_bars",
    "endpoint": core.ENDPOINTS["minute_bars"],
    "parameters": {"symbols": "SPY", "feed": "sip", "start": "a", "end": "b"},
}


def collector(responses: list) -> tuple[data.Collector, Session, Clock]:
    session, clock = Session(responses), Clock()
    return data.Collector(session, clock=clock.monotonic, sleep=clock.sleep), session, clock


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "ANCHOR", "2024-12-02")
    monkeypatch.setattr(core, "LAST", "2024-12-03")


def test_complete_pagination_hashes_every_page_and_keeps_parameters() -> None:
    rows = [bar("2024-12-02", "09:59", 1.0), bar("2024-12-02", "10:00", 2.0)]
    client, session, _ = collector([bars(rows[:1], "p2"), bars(rows[1:], "p3"), bars([], None)])
    result = client.fetch(SPEC)
    assert result["outcome"] == "complete"
    assert [page["record_count"] for page in result["pages"]] == [1, 1, 0]
    assert all(len(page["body_sha256"]) == 64 for page in result["pages"])
    assert result["records"] == [{k: row[k] for k in ("t", "o", "c", "v")} for row in rows]
    params = [kwargs["params"] for _, kwargs in session.calls]
    assert "page_token" not in params[0] and params[1]["page_token"] == "p2"
    assert all(
        {k: v for k, v in p.items() if k != "page_token"} == SPEC["parameters"] for p in params
    )
    assert all(kwargs["allow_redirects"] is False for _, kwargs in session.calls)


def test_repeated_token_and_page_cap_are_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = collector([bars([bar("2024-12-02", "09:59", 1.0)], "same")] * 2)
    result = client.fetch(SPEC)
    assert result["outcome"] == "repeated_page_token" and result["records"] == []
    monkeypatch.setattr(data, "MAX_PAGES_PER_REQUEST", 3)
    client, session, _ = collector([bars([], f"t{i}") for i in range(5)])
    result = client.fetch(SPEC)
    assert result["outcome"] == "page_cap_reached" and result["pages"] == []
    assert len(session.calls) == 3


def test_finite_retries_spacing_and_deferral() -> None:
    client, session, clock = collector(
        [Response({}, 500), requests.ConnectionError("boom"), bars([], None)]
    )
    assert client.fetch(SPEC)["outcome"] == "complete"
    assert clock.waits == [1.0, 2.0]  # backoff dominates the 0.65 s spacing
    client, session, _ = collector([Response({}, 429, {"Retry-After": "120"})])
    assert client.fetch(SPEC)["outcome"] == "deferred" and len(session.calls) == 1
    client, session, _ = collector([Response({}, 503)] * 3)
    assert client.fetch(SPEC)["outcome"] == "transient_exhausted" and len(session.calls) == 3
    client, session, clock = collector([bars([], None), bars([], None)])
    client.fetch(SPEC)
    client.fetch(SPEC)
    assert clock.waits == [pytest.approx(data.REQUEST_SPACING_SECONDS)]


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(401, "authentication_error"), (403, "entitlement_error"), (302, "invalid_request")],
)
def test_non_retryable_statuses_stop_without_retry(status: int, outcome: str) -> None:
    client, session, _ = collector([Response({"message": SECRET}, status)])
    result = client.fetch(SPEC)
    assert result["outcome"] == outcome and len(session.calls) == 1
    assert SECRET not in json.dumps(result)


@pytest.mark.parametrize(
    "body",
    [
        b'{"bars": {"SPY": [{"t": "2024-12-02T15:00:00Z", "o": NaN, "c": 1, "v": 1}]},'
        b' "next_page_token": null}',
        json.dumps(
            {
                "bars": {"SPY": [{"t": "2024-12-02T15:00:00Z", "o": 1, "v": 1}]},
                "next_page_token": None,
            }
        ).encode(),
        json.dumps({"bars": {"QQQ": []}, "next_page_token": None}).encode(),
        json.dumps({"bars": {"SPY": []}}).encode(),
        json.dumps(
            {"bars": {"SPY": [{"t": "bad", "o": 1, "c": 1, "v": 1}]}, "next_page_token": None}
        ).encode(),
    ],
)
def test_malformed_pages_are_schema_errors(body: bytes) -> None:
    client, _, _ = collector([Response(body)])
    assert client.fetch(SPEC)["outcome"] == "schema_error"


# Shapes follow the official v1/corporate-actions OpenAPI examples; every value is invented.
CASH_DIVIDEND = {
    "id": "11cfd108-292e-4cc6-bfbf-5999cdbc4029",
    "symbol": "SPY",
    "cusip": "000000000",
    "rate": 1.7,  # REST returns a JSON number
    "special": False,
    "foreign": False,
    "currency": "USD",
    "sub_type": "provider free text that must not be persisted",
    "ex_date": "2024-12-20",
    "record_date": "2024-12-20",
    "payable_date": "2025-01-31",
    "process_date": "2025-01-31",
}
UNIT_SPLIT = {
    "id": "3e68e87e-ae95-4d68-91d1-715d52ef143a",
    "old_symbol": "SPY",
    "old_cusip": "000000001",
    "old_rate": 1,
    "new_symbol": "SPY",
    "new_cusip": "000000002",
    "new_rate": 1,
    "alternate_symbol": "SPYW",
    "alternate_cusip": "000000003",
    "alternate_rate": 0.25,
    "effective_date": "2024-12-03",
    "process_date": "2024-12-20",
}
NAME_CHANGE = {
    "id": "5a774c35-edec-4532-a812-a56d0bbb623a",
    "old_symbol": "SPY",
    "old_cusip": "000000001",
    "new_symbol": "SPYX",
    "new_cusip": "000000002",
    "process_date": "2024-12-02",
}
REVERSE_SPLIT = {
    "symbol": "SPY",
    "new_symbol": "",  # documented: empty when the ticker does not change
    "old_rate": 2,
    "new_rate": 1,
    "ex_date": "2024-12-03",
    "process_date": "2024-12-03",
}
REORGANIZATION = {
    "symbol": "SPY",
    "cash_rate": 0.1,
    "effective_date": "2024-12-03",
    "process_date": "2024-12-03",
    "stock_movements": [{"symbol": "NEW", "cusip": "0", "new_rate": 1, "source_rate": 1}],
}


def actions_page(groups: dict) -> dict:
    return {"corporate_actions": groups, "next_page_token": None}


def test_corporate_actions_keep_validated_aliases_and_effective_dates_only() -> None:
    records, token = data.parse_page(
        "corporate_actions",
        actions_page(
            {
                "cash_dividends": [CASH_DIVIDEND],
                "forward_splits": [],
                "name_changes": [NAME_CHANGE],
                "reorganizations": [REORGANIZATION],
                "reverse_splits": [REVERSE_SPLIT],
                "unit_splits": [UNIT_SPLIT],
            }
        ),
    )
    assert token is None
    assert records == [
        {
            "group": "cash_dividends",
            "symbol": "SPY",
            "rate": 1.7,
            "currency": "USD",
            "ex_date": "2024-12-20",
            "process_date": "2025-01-31",
        },
        {
            "group": "name_changes",
            "old_symbol": "SPY",
            "new_symbol": "SPYX",
            "process_date": "2024-12-02",
        },
        {
            "group": "reorganizations",
            "symbol": "SPY",
            "stock_movements": [{"symbol": "NEW"}],
            "effective_date": "2024-12-03",
            "process_date": "2024-12-03",
        },
        {
            "group": "reverse_splits",
            "symbol": "SPY",
            "ex_date": "2024-12-03",
            "process_date": "2024-12-03",
        },
        {
            "group": "unit_splits",
            "old_symbol": "SPY",
            "new_symbol": "SPY",
            "alternate_symbol": "SPYW",
            "effective_date": "2024-12-03",
            "process_date": "2024-12-20",
        },
    ]
    for record in records:  # the persisted form revalidates to itself
        group = record["group"]
        assert core.action_record(group, {k: v for k, v in record.items() if k != "group"}) == {
            k: v for k, v in record.items() if k != "group"
        }


UNSAFE = "UNSAFE <provider> string"


@pytest.mark.parametrize(
    "groups",
    [
        {"mystery_actions": [CASH_DIVIDEND]},
        {"mystery_actions": []},  # unknown group names fail even when empty
        {"cash_dividends": [{**CASH_DIVIDEND, "rate": "1.7"}]},  # SSE-style string, not REST
        {"cash_dividends": [{**CASH_DIVIDEND, "rate": True}]},
        {"cash_dividends": [{k: v for k, v in CASH_DIVIDEND.items() if k != "rate"}]},
        {"cash_dividends": [{**CASH_DIVIDEND, "currency": "usd"}]},
        {"cash_dividends": [{**CASH_DIVIDEND, "currency": UNSAFE}]},
        {"cash_dividends": [{**CASH_DIVIDEND, "ex_date": "2024-12-2"}]},
        {"cash_dividends": [{**CASH_DIVIDEND, "symbol": UNSAFE}]},
        {"name_changes": [{**NAME_CHANGE, "new_symbol": 5}]},
        {"reorganizations": [{**REORGANIZATION, "stock_movements": UNSAFE}]},
        {"cash_dividends": CASH_DIVIDEND},
    ],
)
def test_invalid_corporate_action_metadata_is_a_fixed_schema_error(groups: dict) -> None:
    with pytest.raises(data.SchemaError) as error:
        data.parse_page("corporate_actions", actions_page(groups))
    assert str(error.value) == ""
    client, _, _ = collector([Response(actions_page(groups))])
    result = client.fetch({**SPEC, "kind": "corporate_actions"})
    assert result["outcome"] == "schema_error" and result["records"] == []
    assert UNSAFE not in json.dumps(result) and "mystery" not in json.dumps(result)


def vendor_responses() -> list:
    """Vendor-shaped responses for the four manifest requests of the two-session window."""
    calendar = [
        {"date": "2024-12-02", "open": "09:30", "close": "16:00", "session_open": "0400"},
        {"date": "2024-12-03", "open": "09:30", "close": "16:00", "session_open": "0400"},
    ]
    daily = [
        {"t": utc("2024-12-02", "00:00"), "o": 1, "h": 1, "l": 1, "c": 60.4, "v": 1},
        {"t": utc("2024-12-03", "00:00"), "o": 1, "h": 1, "l": 1, "c": 61.3, "v": 1},
    ]
    actions = {
        "corporate_actions": {
            "cash_dividends": [{"symbol": "SPY", "rate": 0.5, "ex_date": "2024-12-03"}]
        },
        "next_page_token": None,
    }
    minutes = [bar("2024-12-02", "15:59", 60.4)] + [
        bar("2024-12-03", clock, price)
        for clock, price in (("09:59", 60.9), ("15:30", 61.0), ("15:59", 61.3))
    ]
    return [
        Response(calendar),
        bars(daily),
        Response(actions),
        bars(minutes[:2], "next"),
        bars(minutes[2:], None),
    ]


def test_collect_validate_and_score_end_to_end(window: None, tmp_path: Path) -> None:
    clock = Clock()
    session = Session(vendor_responses())
    summary = data.run_collection(
        tmp_path / "c", session=session, clock=clock.monotonic, sleep=clock.sleep
    )
    assert summary == {
        "status": "complete",
        "missing": [],
        "manifest_sha256": summary["manifest_sha256"],
    }
    report = core.run_validation(tmp_path / "c", tmp_path / "v.json")
    assert report["status"] == "PASS", report["defects"]
    result = core.run_score(tmp_path / "c", tmp_path / "v.json", tmp_path / "r.json")
    assert result["a1"]["sessions"] == 1
    for path in (tmp_path / "c").rglob("*.json"):
        text = path.read_text()
        assert SECRET not in text and "KEY-123" not in text and '"next"' not in text


def test_resume_after_deferral_refetches_only_missing_chunks(window: None, tmp_path: Path) -> None:
    clock = Clock()
    first = vendor_responses()
    deferred = Session([first[0], Response({"message": SECRET}, 429, {"Retry-After": "600"})])
    summary = data.run_collection(
        tmp_path, session=deferred, clock=clock.monotonic, sleep=clock.sleep
    )
    assert summary["status"] == "incomplete"
    assert summary["missing"] == ["daily_bars", "corporate_actions", "minute_bars:2024-12"]
    calendar = core.chunk_path(tmp_path, "calendar")
    before = calendar.read_bytes()
    resumed = Session(first[1:])
    summary = data.run_collection(
        tmp_path, session=resumed, clock=clock.monotonic, sleep=clock.sleep
    )
    assert summary["status"] == "complete"
    assert calendar.read_bytes() == before
    assert core.ENDPOINTS["calendar"] not in [url for url, _ in resumed.calls]
    progress = json.loads((tmp_path / core.PROGRESS_FILE).read_text())
    assert [r["outcome"] for r in progress["runs"][0]["requests"]] == ["complete", "deferred"]
    assert SECRET not in json.dumps(progress)


def test_failed_chunk_keeps_no_partial_pages_and_run_continues(
    window: None, tmp_path: Path
) -> None:
    responses = vendor_responses()
    responses[2] = Response(b"not json")
    clock = Clock()
    summary = data.run_collection(
        tmp_path, session=Session(responses), clock=clock.monotonic, sleep=clock.sleep
    )
    assert summary["missing"] == ["corporate_actions"]
    assert not core.chunk_path(tmp_path, "corporate_actions").exists()
    report = core.run_validation(tmp_path, tmp_path / "v.json")
    assert "provenance_incomplete:corporate_actions" in report["defects"]


def test_resume_rejects_changed_code_or_tampered_chunk(
    window: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    data.run_collection(
        tmp_path,
        session=Session(vendor_responses()[:1] + [Response({}, 401)]),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    path = core.chunk_path(tmp_path, "calendar")
    original = path.read_text()
    path.write_text(original.replace("2024-12-03", "2024-12-04"))
    with pytest.raises(ValueError, match="chunk_hash_mismatch:calendar"):
        data.run_collection(tmp_path, session=Session([]))
    path.write_text(original)
    monkeypatch.setattr(core, "code_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="resume_manifest_mismatch"):
        data.run_collection(tmp_path, session=Session([]))


def test_credentials_come_only_from_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="credentials_missing"):
        data.make_session()
    monkeypatch.setenv("APCA_API_KEY_ID", "KEY-123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    session = data.make_session()
    assert session.trust_env is False
    assert session.headers["APCA-API-SECRET-KEY"] == SECRET
    session.close()


def load_script() -> object:
    spec = importlib.util.spec_from_file_location("intraday_momentum_research", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_help_default_and_plan_need_no_credentials_network_or_writes(
    offline: Path, capsys: pytest.CaptureFixture
) -> None:
    script = load_script()
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--help"])
    assert exit_info.value.code == 0
    assert script.main([]) == 0
    assert "collect" in capsys.readouterr().out
    assert script.main(["plan"]) == 0
    assert json.loads(capsys.readouterr().out)["protocol_sha256"] == core.PROTOCOL_SHA256
    assert list(offline.iterdir()) == []


def test_collect_without_credentials_refuses_before_any_request(
    offline: Path, capsys: pytest.CaptureFixture
) -> None:
    script = load_script()
    assert script.main(["collect", "--output-dir", str(offline / "c")]) == 2
    assert "credentials_missing" in capsys.readouterr().err
