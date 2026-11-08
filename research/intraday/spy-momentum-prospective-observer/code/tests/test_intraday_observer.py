"""Offline contract tests for the isolated intraday prospective recorder (no network, no .env)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import intraday_observer as obs
import pytest

from ggTrader.utils.study import repo_root, study_dir

pytestmark = pytest.mark.study


def test_review_live_activation_requires_readiness_evidence(tmp_path):
    ob, _, _ = make_observer(tmp_path, start=(ACTIVATION, "10:00:00"))
    with pytest.raises(obs.ObserverError, match="readiness"):
        ob.activate("owner approved")
    assert ob.records() == []


def test_review_calendar_amendments_cannot_reclassify_an_observed_day(tmp_path):
    ob, clock, _ = active(tmp_path)
    full_day(ob, clock, DAY1)
    with pytest.raises(obs.ObserverError, match="amendment"):
        ob.amend_session(DAY1, "early_close", "13:00", "owner approved")
    assert ob.session_row(DAY1)["kind"] == "full"


def test_review_bootstrap_is_timely_against_its_input_cutoff(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    anchor = ob.canonical(ANCHOR_DAY)
    assert anchor["body"]["bootstrap"]
    assert anchor["body"]["on_time"]
    assert "archive_late" not in anchor["body"]["faults"]


def test_review_failed_checkpoint_halts_slot21(tmp_path):
    ob, clock, _ = active(tmp_path)
    slots = [r for r in ob.effective_rows() if r["slot"]]
    day20, day21 = slots[19]["date"], slots[20]["date"]
    # Missing earlier decisions are observable operational failures, not erased days.
    archive_day(ob, clock, day20)
    clock.set(day21, "15:25:05")
    assert any(r["reason"] == "checkpoint_failed" for r in ob.halt_reasons(day21))


def test_review_twenty_clean_sessions_including_bootstrap_do_not_halt(tmp_path):
    ob, clock, _ = active(tmp_path)
    rows = ob.effective_rows()
    checkpoint = ob.manifest["calendar"]["checkpoint_date"]
    for row in rows:
        if row["date"] > checkpoint:
            break
        if row["date"] < DAY1:
            continue  # Exercise the permitted first-input bootstrap.
        if row["slot"]:
            full_day(ob, clock, row["date"])
        elif row["kind"] == "early_close":
            archive_day(ob, clock, row["date"])
    following = ob.neighbours(checkpoint)[1]
    clock.set(following, "15:25:00")
    assert ob.checkpoint_status()["conditions_met"] is True
    assert not ob.halt_reasons(following)


def test_review_live_readiness_must_match_the_prepared_manifest(tmp_path):
    ob, _, _ = make_observer(tmp_path, start=(ACTIVATION, "10:00:00"))
    path = write_readiness(ob)
    evidence = json.loads(path.read_text())
    evidence["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(evidence))
    with pytest.raises(obs.ObserverError, match="readiness_evidence_mismatch"):
        ob.activate("operator approval", readiness_file=path)
    assert not ob.has_record(None, "activation")


def test_review_slow_halt_check_cannot_backdate_a_decision(tmp_path, monkeypatch):
    book = Book().day(DAY1, p0959=101)
    ob, clock, _ = active(tmp_path, book)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:59")

    def delayed(day):
        clock.sleep(2)
        return []

    monkeypatch.setattr(ob, "halt_reasons", delayed)
    record = ob.run_decision(DAY1)
    assert record["body"]["observer_side"] == 0
    assert "decision_completed_after_window" in record["body"]["faults"]


def test_review_slow_durable_write_invalidates_the_decision(tmp_path, monkeypatch):
    ob, clock, _ = active(tmp_path, Book().day(DAY1, p0959=101))
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:59")
    original = obs.atomic_checkpoint

    def slow_write(path, value, **kwargs):
        original(path, value, **kwargs)
        if path.name.endswith("__decision.json"):
            clock.sleep(2)

    monkeypatch.setattr(obs, "atomic_checkpoint", slow_write)
    result = ob.run_decision(DAY1)
    assert result["body"]["observer_side"] == 0
    assert not ob.stage_record(DAY1, "decision")["body"]["timely_valid"]
    assert ob.verify_chain() == []


def test_review_slow_archive_write_cannot_cross_the_valuation_grace(tmp_path, monkeypatch):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:19:59")
    original = obs.atomic_checkpoint

    def slow_write(path, value, **kwargs):
        original(path, value, **kwargs)
        if path.name.endswith("__archive.1.json"):
            clock.sleep(2)

    monkeypatch.setattr(obs, "atomic_checkpoint", slow_write)
    record = ob.run_archive(ANCHOR_DAY)
    assert not record["body"]["canonical"]
    assert ob.canonical(ANCHOR_DAY) is None


def test_review_pacing_sleep_never_starts_a_get_after_cutoff(tmp_path):
    ob, clock, deps = make_observer(tmp_path, start=(DAY1, "15:24:59"))
    ob.reserve_request(DAY1, "previous")
    clock.sleep(0.5)
    result = ob.fetch_bars(
        DAY1, "inputs", at(DAY1, "09:59:00"), at(DAY1, "09:59:00"), at(DAY1, "15:24:59.600000")
    )
    assert result["outcome"] == "cutoff_reached"
    assert not deps.transport.calls


def test_review_slow_clock_probe_cannot_make_late_inputs_usable(tmp_path):
    ob, clock, deps = active(tmp_path)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:24:58")

    def slow_sync():
        clock.sleep(5)
        return "synchronized"

    deps.clock_sync = slow_sync
    record = ob.run_inputs(DAY1)
    assert not record["body"]["usable"]
    assert "inputs_completed_after_cutoff" in record["body"]["faults"]


def test_review_verify_detects_deleted_referenced_raw(tmp_path):
    ob, clock, _ = active(tmp_path)
    record = archive_day(ob, clock, ANCHOR_DAY)
    path = ob.dir / record["body"]["fetch"]["attempts"][0]["raw_path"]
    path.unlink()
    assert any("raw_missing" in p for p in ob.verify_chain())


def test_review_crash_temporary_is_not_ledger_corruption(tmp_path):
    ob, _, _ = make_observer(tmp_path)
    (ob.dir / "raw" / "tmp_crash_file").write_bytes(b"incomplete unreferenced write")
    assert ob.verify_chain() == []


def test_review_verify_binds_old_records_to_current_manifest(tmp_path):
    ob, clock, _ = active(tmp_path)
    archive_day(ob, clock, ANCHOR_DAY)
    path = ob.dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_delay_seconds"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(obs.BindingMismatch, match="record_binding"):
        obs.Observer(ob.dir, ob.deps)


def test_review_corrupt_evidence_cannot_produce_performance_statistics(tmp_path):
    ob, clock, _ = active(tmp_path)
    record = archive_day(ob, clock, ANCHOR_DAY)
    (ob.dir / record["body"]["fetch"]["attempts"][0]["raw_path"]).unlink()
    report = ob.report()
    assert report["status"] == "unresolved"
    assert report["integrity"]["problems"]
    assert "observer_policy" not in report and "fixed_rule" not in report


def test_review_runtime_binds_statistical_libraries():
    assert {"numpy", "pandas", "statsmodels"} <= obs.runtime_binding().keys()


def test_review_real_issuer_workbook_selects_only_spy(tmp_path):
    ob, _, _ = make_observer(tmp_path)
    source = (
        ROOT
        / "research/intraday/artifacts-2026-10-25-intraday-gap"
        / "spdr-etf-historical-distributions.xlsx"
    )
    record = ob.ingest_distributions(
        source,
        sheet="dividend",
        source_url="https://www.ssga.com/issuer.xlsx",
        cusip="78462F103",
        fund="SPY",
        currency="USD",
        decimals=6,
        header_row=1,
        columns={
            "cusip": "CUSIP",
            "ex_date": "EX-DATE",
            "dividend": "DIVIDEND ($)",
            "capital_gains": ["SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)"],
        },
        date_format="mdy",
        covers_through="2026-09-30",
        semantics_verified=True,
    )
    assert not record["body"]["defects"]
    assert len(record["body"]["rows"]) == 136
    assert any(r["ex_date"] == "2016-03-18" for r in record["body"]["rows"])


def test_review_unmapped_capital_gains_cannot_be_complete(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    clock.set(ob.manifest["calendar"]["last_date"], "17:00:00")
    record = ingest(
        ob,
        tmp_path,
        [HEADER, [serial(DAY1), None, None, "1", "0"]],
        columns={**COLUMNS, "capital_gains": []},
    )
    assert "capital_gain_columns_unmapped" in obs.distribution_reasons(
        record, ob.manifest["calendar"]["last_date"]
    )


ROOT = repo_root(__file__)
CODE = study_dir(__file__) / "code"
LIB = ROOT / "research/intraday/_lib"
ACTIVATION = "2026-10-28"  # Wed; third full session after it is Mon 2026-11-02 (DST ended 11-01)


def write_protocol(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "protocol.md"
    path.write_text("frozen protocol text\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def write_calendar(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    path = tmp_path / "official_calendar.json"
    path.write_text(json.dumps(rows or obs.offline_sessions("2026-10-01", "2028-01-31")))
    return path


def init_args(tmp_path: Path, **overrides) -> dict:
    protocol, digest = write_protocol(tmp_path)
    args = {
        "mode": "live",
        "activation_date": ACTIVATION,
        "official_calendar_file": overrides.get("official_calendar_file")
        or write_calendar(tmp_path),
        "protocol_file": protocol,
        "protocol_sha256": digest,
        "source_delay_seconds": 900,
        "now": datetime(2026, 10, 26, 18, 0, tzinfo=UTC),
    }
    args.update(overrides)
    return args


# --- inert import, help, plan -------------------------------------------------------------------


def test_import_is_inert(tmp_path):
    code = (
        f"import sys; sys.path[:0] = [{str(CODE)!r}, {str(LIB)!r}];"  # root-ok: child process
        "import intraday_observer;"
        "assert 'pandas_market_calendars' not in sys.modules;"
        "assert 'ggTrader' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network is forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_help_and_plan_need_no_credentials_network_or_writes(
    tmp_path, no_network, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        obs.main(["--help"])
    assert exit_info.value.code == 0
    assert "plan" in capsys.readouterr().out
    assert obs.main(["plan", "--activation-date", ACTIVATION]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["network"] is False and plan["writes"] is False
    assert plan["activated"] is False and plan["source_preflight"] == "not_performed"
    assert plan["official_calendar_agreement"] == "not_checked"
    assert plan["calendar"]["first_date"] == "2026-11-02"
    assert plan["request_bounds"]["attempts_per_new_york_date"] == 12
    assert list(tmp_path.iterdir()) == []


# --- calendar, DST, early closes ----------------------------------------------------------------


def test_utc_conversion_follows_dst():
    assert obs.utc_at("2026-10-30", "15:20") == datetime(2026, 10, 30, 19, 20, tzinfo=UTC)
    assert obs.utc_at("2026-11-02", "15:20") == datetime(2026, 11, 2, 20, 20, tzinfo=UTC)
    assert obs.utc_at("2027-03-15", "15:20") == datetime(2027, 3, 15, 19, 20, tzinfo=UTC)


def test_calendar_slots_anchor_and_early_closes():
    calendar = obs.build_calendar(obs.offline_sessions("2026-10-01", "2028-01-31"), ACTIVATION)
    slots = [row["date"] for row in calendar["sessions"] if row["slot"]]
    assert len(slots) == 252 and slots == sorted(slots)
    assert calendar["first_date"] == slots[0] == "2026-11-02"
    assert calendar["anchor_date"] == "2026-10-30"  # official session before slot 1
    assert calendar["checkpoint_date"] == slots[19] and calendar["last_date"] == slots[-1]
    by_date = {row["date"]: row for row in calendar["sessions"]}
    half = by_date["2026-11-27"]  # day after Thanksgiving
    assert (half["kind"], half["close"], half["close_minute"], half["slot"]) == (
        "early_close",
        "13:00",
        "12:59",
        None,
    )
    assert by_date["2026-11-30"]["close_minute"] == "15:59"
    assert "2026-11-27" in calendar["early_closes"] and "2026-12-24" in calendar["early_closes"]
    assert "2026-11-26" not in by_date  # Thanksgiving is closed
    assert calendar["year_end_sessions"] == ["2026-12-31"]
    assert calendar["post_window_next_session"] > calendar["last_date"]
    assert len(calendar["slot_dates_sha256"]) == 64


@pytest.mark.parametrize("tamper", ["drop_session", "change_hours", "extra_session"])
def test_calendar_disagreement_blocks_preparation(tamper):
    rows = obs.offline_sessions("2026-10-01", "2028-01-31")
    if tamper == "drop_session":
        rows = [row for row in rows if row["date"] != "2026-12-01"]
    elif tamper == "change_hours":
        rows = [dict(row, close="16:00") if row["date"] == "2026-11-27" else row for row in rows]
    else:
        rows = rows + [{"date": "2026-11-26", "open": "09:30", "close": "16:00"}]
    with pytest.raises(obs.CalendarMismatch) as error:
        obs.build_calendar(rows, ACTIVATION)
    assert error.value.args[0]


# --- explicit offline preparation ---------------------------------------------------------------


def test_init_writes_unverified_unactivated_manifest(tmp_path, no_network):
    state = tmp_path / "state"
    manifest = obs.init_state(state, **init_args(tmp_path))
    assert manifest["status"] == "prepared_offline"
    assert manifest["activated"] is False and manifest["source_preflight"] == "not_performed"
    assert manifest["source_delay_verified"] is False and manifest["source_delay_seconds"] == 900
    assert manifest["state_dir"] == str(state.resolve())
    assert (
        manifest["calendar"]["official_agreement"]
        == "operator_supplied_snapshot_matches_offline_calendar"
    )
    for key in ("protocol", "code", "runtime"):
        assert manifest["bindings"][key]
    assert json.loads((state / "manifest.json").read_text()) == manifest
    assert not list((state / "records").iterdir())  # preparation records no observation
    with pytest.raises(FileExistsError):
        obs.init_state(state, **init_args(tmp_path))


def test_init_rejects_protocol_hash_and_calendar_mismatch_without_writing(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="protocol_hash_mismatch"):
        obs.init_state(state, **init_args(tmp_path, protocol_sha256="0" * 64))
    rows = [
        r for r in obs.offline_sessions("2026-10-01", "2028-01-31") if r["date"] != "2026-12-01"
    ]
    with pytest.raises(obs.CalendarMismatch):
        obs.init_state(
            state, **init_args(tmp_path, official_calendar_file=write_calendar(tmp_path, rows))
        )
    assert not state.exists()


# --- shared fakes -------------------------------------------------------------------------------


def at(day: str, clock: str = "00:00:00") -> datetime:
    """New York wall clock as UTC; clock may carry seconds."""
    return obs.utc_at(day, clock)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)

    def set(self, day: str, clock: str) -> None:
        self.now = at(day, clock)


def make_observer(
    tmp_path, transport=None, *, start=("2026-11-02", "12:00:00"), sync="synchronized"
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = tmp_path / "state"
    if not state.exists():
        obs.init_state(state, **init_args(tmp_path))
    clock = FakeClock(at(*start))
    deps = obs.Deps(
        clock=clock,
        sleep=clock.sleep,
        transport=transport or obs.SyntheticTransport(),
        clock_sync=lambda: sync,
    )
    return obs.Observer(state, deps), clock, deps


def bar_bytes(minutes: list[str], day: str = "2026-11-02", **extra) -> bytes:
    rows = []
    for hhmm in minutes:
        stamp = at(day, hhmm + ":00").strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({"t": stamp, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10, **extra})
    return json.dumps({"bars": {"SPY": rows}, "next_page_token": None}).encode()


def window(day: str, start: str, end: str) -> tuple[datetime, datetime]:
    return at(day, start + ":00"), at(day, end + ":00")


# --- evidence ledger: bindings, claims, chain ---------------------------------------------------


def test_observer_rejects_path_protocol_code_and_runtime_mismatch(tmp_path, monkeypatch):
    make_observer(tmp_path)
    state = tmp_path / "state"
    moved = tmp_path / "moved"
    moved.mkdir()
    for item in state.iterdir():
        (moved / item.name).symlink_to(item)  # same content, different path
    with pytest.raises(obs.BindingMismatch, match="state_path_mismatch"):
        obs.Observer(moved)
    monkeypatch.setattr(obs, "runtime_binding", lambda: {"python": "0", "sha256": "x"})
    with pytest.raises(obs.BindingMismatch, match="runtime"):
        obs.Observer(state)
    monkeypatch.undo()
    monkeypatch.setattr(obs, "code_binding", lambda: {"files": {}, "sha256": "x"})
    with pytest.raises(obs.BindingMismatch, match="code"):
        obs.Observer(state)
    monkeypatch.undo()
    (tmp_path / "protocol.md").write_text("edited\n")
    with pytest.raises(obs.BindingMismatch, match="protocol"):
        obs.Observer(state)


def test_naive_clock_is_rejected(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    clock.now = datetime(2026, 11, 2, 12, 0)
    with pytest.raises(obs.ClockMismatch):
        ob.append("note", "2026-11-02", "x", {}, clock())


def test_append_chains_records_and_detects_tampering(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    first = ob.append("note", "2026-11-02", "one", {"a": 1}, clock())
    clock.now += timedelta(seconds=5)
    second = ob.append("note", "2026-11-02", "two", {"b": 2}, clock())
    assert (first["seq"], second["seq"]) == (1, 2)
    assert first["prev_sha256"] is None and second["prev_sha256"] == first["sha256"]
    assert first["bindings"]["manifest"] and first["bindings"]["code"]
    assert ob.verify_chain() == []
    path = next((tmp_path / "state" / "records").glob("000001__*"))
    path.write_text(path.read_text().replace('"a": 1', '"a": 2'))
    assert any("hash" in problem for problem in ob.verify_chain())


def test_append_rejects_duplicate_stage_and_clock_regression(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    ob.append("stage", "2026-11-02", "inputs", {}, clock())
    with pytest.raises(obs.DuplicateStage):
        ob.append("stage", "2026-11-02", "inputs", {"changed": True}, clock())
    with pytest.raises(obs.ClockMismatch):
        ob.append("stage", "2026-11-02", "decision", {}, clock() - timedelta(seconds=1))
    assert len(ob.records()) == 1


def test_claim_is_exclusive_and_reconciles_a_crashed_predecessor(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    other = obs.Observer(tmp_path / "state", ob.deps)
    with ob.claim("2026-11-02", "decision") as prior:
        assert prior is None
        with pytest.raises(obs.StageBusy):
            with other.claim("2026-11-02", "decision"):
                pass
    # the claim file persists after release; with no record it is a crashed predecessor
    with other.claim("2026-11-02", "decision") as prior:
        assert prior["pid"] == os.getpid() and prior["claimed_utc"]
        other.append("stage", "2026-11-02", "decision", {}, clock())
    with pytest.raises(obs.DuplicateStage):
        with ob.claim("2026-11-02", "decision"):
            pass


# --- request budget and fetch -------------------------------------------------------------------


def test_budget_is_shared_across_processes_stages_and_the_new_york_date(tmp_path):
    ob, clock, _ = make_observer(tmp_path, start=("2026-11-02", "13:00:00"))
    other = obs.Observer(tmp_path / "state", ob.deps)
    for n in range(12):
        (ob if n % 2 else other).reserve_request("2026-11-02", "inputs")
    with pytest.raises(obs.BudgetExhausted):
        ob.reserve_request("2026-11-02", "archive")
    clock.now = datetime(2026, 11, 3, 3, 30, tzinfo=UTC)  # next UTC day, still 22:30 EST on Nov 2
    with pytest.raises(obs.BudgetExhausted):
        other.reserve_request("2026-11-02", "archive")
    clock.set("2026-11-03", "09:00:00")
    other.reserve_request("2026-11-03", "inputs")  # a new New York date has a fresh allowance


def test_request_starts_are_spaced_by_the_minimum_gap(tmp_path):
    ob, clock, _ = make_observer(tmp_path)
    ob.reserve_request("2026-11-02", "inputs")
    ob.reserve_request("2026-11-02", "inputs")
    assert clock.sleeps == [pytest.approx(0.65)]


def run_fetch(ob, clock, minutes=("09:59", "09:59"), cutoff=("2026-11-02", "15:25:00")):
    start, end = window("2026-11-02", *minutes)
    return ob.fetch_bars("2026-11-02", "inputs", start, end, at(*cutoff))


def test_fetch_preserves_raw_bytes_hash_times_and_parameters(tmp_path):
    ob, clock, deps = make_observer(tmp_path, start=("2026-11-02", "15:20:00"))
    result = run_fetch(ob, clock)
    assert result["outcome"] == "success" and len(result["bars"]) == 1
    attempt = result["attempts"][0]
    body = (tmp_path / "state" / attempt["raw_path"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == attempt["body_sha256"]
    assert attempt["requested_utc"] and attempt["received_utc"] and attempt["status_code"] == 200
    params = deps.transport.calls[0][1]
    assert (params["feed"], params["adjustment"], params["asof"], params["sort"]) == (
        "sip",
        "raw",
        "-",
        "asc",
    )
    assert params["start"] == params["end"] == "2026-11-02T14:59:00Z"  # inclusive 09:59 ET bar
    assert deps.transport.calls[0][0] == obs.BARS_URL


def test_fetch_end_must_not_be_newer_than_now_minus_source_delay(tmp_path):
    ob, clock, deps = make_observer(tmp_path, start=("2026-11-02", "10:10:00"))
    result = run_fetch(ob, clock)  # 10:10 - 15 min = 09:55 < 09:59
    assert result["outcome"] == "source_not_available" and deps.transport.calls == []
    assert result["attempts"] == []


def test_transient_failures_retry_with_backoff_then_succeed(tmp_path):
    script = [obs.Response(503, {}, b"busy"), obs.Response(None, {}, b"", "timeout"), None]
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:20:00")
    )
    result = run_fetch(ob, clock)
    assert result["outcome"] == "success"
    assert [a["status_code"] for a in result["attempts"]] == [503, None, 200]
    assert result["attempts"][1]["transport_error"] == "timeout"
    assert 1.0 in clock.sleeps and 2.0 in clock.sleeps


def test_three_failed_attempts_exhaust_without_a_fourth(tmp_path):
    script = [obs.Response(500, {}, b"x")] * 5
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:20:00")
    )
    result = run_fetch(ob, clock)
    assert result["outcome"] == "transient_exhausted" and len(deps.transport.calls) == 3


def test_retry_never_runs_past_the_stage_cutoff(tmp_path):
    script = [obs.Response(500, {}, b"x")] * 5
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:24:59.500000")
    )
    result = run_fetch(ob, clock)
    assert result["outcome"] == "cutoff_reached" and len(deps.transport.calls) == 1
    assert clock() < at("2026-11-02", "15:25:00")  # it did not sleep across the cutoff


def test_long_retry_after_defers_and_auth_errors_are_not_retried(tmp_path):
    script = [obs.Response(429, {"retry-after": "120"}, b"slow")]
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:20:00")
    )
    assert run_fetch(ob, clock)["outcome"] == "deferred"
    for status, outcome in ((401, "authentication_error"), (403, "entitlement_error")):
        script = [obs.Response(status, {}, b"no")]
        ob, clock, deps = make_observer(
            tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-03", "15:20:00")
        )
        start, end = window("2026-11-03", "09:59", "09:59")
        result = ob.fetch_bars("2026-11-03", "inputs", start, end, at("2026-11-03", "15:25:00"))
        assert result["outcome"] == outcome and len(deps.transport.calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"bars": {"SPY": []}}).encode(),  # token key missing
        bar_bytes(["09:59", "09:59"]),  # duplicate timestamp
        bar_bytes(["10:00"]),  # outside the requested inclusive range
        json.dumps(
            {
                "bars": {
                    "SPY": [{"t": "2026-11-02T14:59:30Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]
                },
                "next_page_token": None,
            }
        ).encode(),  # misaligned
        bar_bytes(["09:59"], c=0.0),
        bar_bytes(["09:59"], h=50.0),  # high below the close
    ],
)
def test_corrupt_payload_is_preserved_and_yields_no_bars(tmp_path, body):
    script = [obs.Response(200, {}, body)]
    ob, clock, _ = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:20:00")
    )
    result = run_fetch(ob, clock)
    assert result["outcome"] == "schema_error" and result["bars"] == [] and result["defects"]
    assert (tmp_path / "state" / result["attempts"][0]["raw_path"]).read_bytes() == body


def test_pagination_collects_pages_and_counts_each_request(tmp_path):
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(per_page=2), start=("2026-11-02", "16:30:00")
    )
    start, end = window("2026-11-02", "09:30", "09:34")
    result = ob.fetch_bars("2026-11-02", "archive", start, end, at("2026-11-02", "17:00:00"))
    assert result["outcome"] == "success" and len(result["bars"]) == 5 and result["pages"] == 3
    assert len(deps.transport.calls) == 3
    assert (
        len(json.loads((tmp_path / "state" / "budget" / "2026-11-02.json").read_text())["attempts"])
        == 3
    )


def test_http_transport_is_fixed_readonly_and_never_persists_secrets(tmp_path):
    with pytest.raises(ValueError, match="credentials_missing"):
        obs.HttpTransport(environ={})
    calls = []

    class Reply:
        status_code = 200
        content = b"{}"
        headers = {"X-Test": "1"}

        def close(self):
            pass

    class Session:
        headers: dict = {}
        trust_env = True

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Reply()

    session = Session()
    transport = obs.HttpTransport(
        environ={"APCA_API_KEY_ID": "KEYSECRET", "APCA_API_SECRET_KEY": "TOPSECRET"},
        session=session,
    )
    assert session.trust_env is False
    response = transport.get(obs.BARS_URL, {"symbols": "SPY"})
    assert response.status == 200 and response.headers == {"x-test": "1"}
    assert calls[0][1]["allow_redirects"] is False and calls[0][1]["timeout"] == 30
    with pytest.raises(ValueError, match="url_not_allowed"):
        transport.get("https://paper-api.alpaca.markets/v2/orders", {})
    ob, clock, _ = make_observer(
        tmp_path, obs.SyntheticTransport(), start=("2026-11-02", "15:20:00")
    )
    run_fetch(ob, clock)
    blob = b"".join(p.read_bytes() for p in (tmp_path / "state").rglob("*") if p.is_file())
    assert b"KEYSECRET" not in blob and b"TOPSECRET" not in blob


# --- stages: inputs, decision, archive ----------------------------------------------------------

HALF_DAYS = {"2026-11-27": "13:00"}
ANCHOR_DAY, DAY1, DAY2, HALF_DAY, DAY_AFTER_HALF = (
    "2026-10-30",
    "2026-11-02",
    "2026-11-03",
    "2026-11-27",
    "2026-11-30",
)


class Book:
    """Synthetic prices: p0959 close, p1530 open and closing-minute close per session date."""

    def __init__(self) -> None:
        self.days: dict[str, tuple[float, float, float]] = {}

    def day(self, day: str, p0959: float = 100.0, p1530: float = 100.0, close: float = 100.0):
        self.days[day] = (p0959, p1530, close)
        return self

    def __call__(self, day: str, hhmm: str) -> tuple[float, float, float, float]:
        p0959, p1530, close = self.days.get(day, (100.0, 100.0, 100.0))
        last = "12:59" if day in HALF_DAYS else "15:59"
        c = {"09:59": p0959, last: close}.get(hhmm, 100.0)
        o = p1530 if hhmm == "15:30" else c
        return o, max(o, c) + 0.01, min(o, c) - 0.01, c


def active(tmp_path, book=None, transport=None, **kwargs):
    transport = transport or obs.SyntheticTransport(price_fn=book or Book(), closes=HALF_DAYS)
    ob, clock, deps = make_observer(tmp_path, transport, start=(ACTIVATION, "10:00:00"), **kwargs)
    ob.activate("synthetic test activation", readiness_file=write_readiness(ob))
    return ob, clock, deps


def write_readiness(ob):
    """Synthetic readiness evidence for tests; this never verifies a live source."""
    path = ob.dir / "synthetic-readiness.json"
    path.write_text(
        json.dumps(
            {
                "schema": "intraday_observer_readiness",
                "status": "verified",
                "manifest_sha256": ob.manifest_sha256,
                "source_delay_seconds": ob.manifest["source_delay_seconds"],
                "calendar_snapshot_sha256": ob.manifest["calendar"]["official_snapshot_sha256"],
                "checks": dict.fromkeys(
                    ("sip_access", "source_delay", "calendar_source", "synthetic_rehearsal"), True
                ),
            }
        )
    )
    return path


def archive_day(ob, clock, day, clock_text="16:20:05"):
    clock.set(day, clock_text)
    return ob.run_archive(day)


def full_day(ob, clock, day):
    clock.set(day, "15:20:05")
    inputs = ob.run_inputs(day)
    clock.set(day, "15:25:05")
    decision = ob.run_decision(day)
    clock.set(day, "16:20:05")
    return inputs, decision, ob.run_archive(day)


def test_stage_commands_need_explicit_activation(tmp_path):
    ob, clock, _ = make_observer(tmp_path, start=(DAY1, "15:20:05"))
    with pytest.raises(obs.NotActivated):
        ob.run_inputs(DAY1)
    assert ob.records() == []  # startup is never implicit


def test_activation_must_be_on_the_registered_date_and_names_an_approval(tmp_path):
    ob, clock, _ = make_observer(tmp_path, start=("2026-10-27", "10:00:00"))
    with pytest.raises(obs.ClockMismatch):
        ob.activate("approved")
    clock.set(ACTIVATION, "10:00:00")
    with pytest.raises(obs.ObserverError, match="owner_approval_required"):
        ob.activate("  ")
    ready = write_readiness(ob)
    record = ob.activate("approved by owner", readiness_file=ready)
    assert record["body"]["source_preflight"] == "verified"
    assert record["body"]["source_delay_verified"] is True
    with pytest.raises(obs.DuplicateStage):
        ob.activate("again", readiness_file=ready)


def test_a_normal_session_records_inputs_one_decision_and_canonical_valuation(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
    ob, clock, deps = active(tmp_path, book)
    anchor = archive_day(ob, clock, ANCHOR_DAY)
    assert anchor["body"]["canonical"] and anchor["body"]["values"]["close_final"] == 100.0
    inputs, decision, archive = full_day(ob, clock, DAY1)
    assert inputs["body"]["status"] == "ok" and inputs["body"]["timely"] is True
    assert inputs["body"]["prior"]["record_sha256"] == anchor["sha256"]
    assert inputs["body"]["bar_0959"]["c"] == 101.0
    body = decision["body"]
    assert (body["signal_side"], body["observer_side"], body["status"]) == (1, 1, "ok")
    assert body["r_first"] == pytest.approx(0.01) and body["inputs_ref"] == inputs["sha256"]
    assert body["timely_valid"] is True and body["halted"] is False
    values = archive["body"]["values"]
    assert (values["open_1530"], values["close_final"]) == (100.0, 101.0)
    assert archive["body"]["canonical"] and archive["body"]["on_time"]
    assert archive["body"]["late_capture"] is False
    # DST: 09:59 ET is 13:59Z before the clocks change and 14:59Z after
    starts = [call[1]["start"] for call in deps.transport.calls]
    assert starts[0] == "2026-10-30T13:30:00Z" and starts[1] == "2026-11-02T14:59:00Z"
    assert ob.verify_chain() == []


def test_a_nonpositive_first_move_is_a_genuine_flat_decision(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=100.0, p1530=100.0, close=105.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    _, decision, archive = full_day(ob, clock, DAY1)
    assert decision["body"]["signal_side"] == 0 and decision["body"]["observer_side"] == 0
    assert decision["body"]["status"] == "ok" and decision["body"]["faults"] == []
    assert archive["body"]["valuation_valid"]  # flat sessions are valued too


def test_decision_reads_only_the_frozen_inputs_and_never_the_source(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, close=90.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    inputs = ob.run_inputs(DAY1)
    # nothing after 09:59 was ever requested, so no entry or exit value exists at decision time
    assert deps.transport.calls[-1][1]["end"] == "2026-11-02T14:59:00Z"
    assert set(inputs["body"]) >= {"bar_0959", "prior"} and "open_1530" not in json.dumps(inputs)

    class Exploding:
        def get(self, *args):
            raise AssertionError("decision must not touch the source")

    deps.transport = Exploding()
    clock.set(DAY1, "15:25:05")
    assert ob.run_decision(DAY1)["body"]["signal_side"] == 1


def test_stages_before_their_window_are_refused_without_a_record(tmp_path):
    ob, clock, deps = active(tmp_path)
    clock.set(DAY1, "15:19:59")
    with pytest.raises(obs.TooEarly):
        ob.run_inputs(DAY1)
    clock.set(DAY1, "15:24:59")
    with pytest.raises(obs.TooEarly):
        ob.run_decision(DAY1)
    clock.set(DAY1, "16:10:00")  # 15:59 + 15 minute source delay is 16:14
    with pytest.raises(obs.TooEarly):
        ob.run_archive(DAY1)
    assert ob.records()[1:] == [] and deps.transport.calls == []


def test_inputs_started_at_the_cutoff_are_a_fault_and_nothing_is_fetched(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    calls = len(deps.transport.calls)
    clock.set(DAY1, "15:25:00")
    inputs = ob.run_inputs(DAY1)
    assert (
        inputs["body"]["faults"] == ["inputs_late_not_attempted"]
        and inputs["body"]["timely"] is False
    )
    assert len(deps.transport.calls) == calls
    clock.set(DAY1, "15:25:30")
    decision = ob.run_decision(DAY1)
    assert decision["body"]["signal_side"] is None and decision["body"]["observer_side"] == 0
    assert decision["body"]["status"] == "fault" and decision["body"]["timely_valid"] is False


def test_inputs_that_complete_after_the_cutoff_cannot_drive_a_timely_decision(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)

    class Slow(obs.SyntheticTransport):
        def get(self, url, params):
            clock.now += timedelta(seconds=3)
            return super().get(url, params)

    ob, clock, deps = active(tmp_path, transport=Slow(price_fn=book, closes=HALF_DAYS))
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:24:58")
    inputs = ob.run_inputs(DAY1)
    assert (
        "inputs_completed_after_cutoff" in inputs["body"]["faults"] and not inputs["body"]["timely"]
    )
    clock.set(DAY1, "15:25:10")
    assert ob.run_decision(DAY1)["body"]["signal_side"] is None


def test_retrospective_decision_is_a_fault_even_with_good_inputs(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:26:00")  # window is [15:25:00, 15:26:00)
    decision = ob.run_decision(DAY1)
    assert decision["body"]["faults"] == ["decision_window_missed"]
    assert decision["body"]["signal_side"] is None and decision["body"]["observer_side"] == 0
    with pytest.raises(obs.DuplicateStage):
        ob.run_decision(DAY1)


def test_decision_that_finishes_after_the_window_is_a_fault_with_no_side(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    deps.clock_sync = lambda: (clock.sleep(2.0), "synchronized")[1]
    clock.set(DAY1, "15:25:59")
    body = ob.run_decision(DAY1)["body"]
    assert "decision_completed_after_window" in body["faults"] and body["signal_side"] is None


@pytest.mark.parametrize("status", ["unsynchronized", "unknown"])
def test_unsynchronized_clock_is_an_evidence_fault_and_blocks_the_position(tmp_path, status):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    deps.clock_sync = lambda: status
    clock.set(DAY1, "15:25:05")
    record = ob.run_decision(DAY1)
    assert f"clock_{status}" in record["body"]["faults"] and record["body"]["observer_side"] == 0
    assert record["clock_status"] == status


def test_crashed_claim_is_reconciled_inside_the_window_and_faults_after_it(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0).day(DAY2, p0959=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:10")
    with ob.claim(DAY1, "decision"):
        pass  # the process dies here: a claim exists, no record
    restarted = obs.Observer(tmp_path / "state", ob.deps)
    record = restarted.run_decision(DAY1)
    assert record["body"]["reconciled_claim"]["pid"] == os.getpid()
    assert record["body"]["signal_side"] == 1
    assert len([r for r in restarted.records() if r["stage"] == "decision"]) == 1
    clock.set(DAY1, "16:20:05")
    ob.run_archive(DAY1)
    clock.set(DAY2, "15:20:05")
    ob.run_inputs(DAY2)
    clock.set(DAY2, "15:25:10")
    with ob.claim(DAY2, "decision"):
        pass
    clock.set(DAY2, "15:40:00")  # restart after the window: only a fault may be written
    record = obs.Observer(tmp_path / "state", ob.deps).run_decision(DAY2)
    assert record["body"]["faults"] == ["decision_window_missed"]
    assert record["body"]["reconciled_claim"] and record["body"]["observer_side"] == 0


def test_a_second_inputs_run_is_refused_and_a_busy_claim_blocks_a_concurrent_run(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    with ob.claim(DAY1, "inputs"):
        with pytest.raises(obs.StageBusy):
            obs.Observer(tmp_path / "state", ob.deps).run_inputs(DAY1)
    ob.run_inputs(DAY1)
    calls = len(deps.transport.calls)
    with pytest.raises(obs.DuplicateStage):
        ob.run_inputs(DAY1)
    assert len(deps.transport.calls) == calls


def test_missing_prior_anchor_makes_inputs_unavailable_except_for_slot_one_bootstrap(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, close=102.0)
    ob, clock, _ = active(tmp_path, book)
    # no anchor archive at all: only slot 1 may fetch it, inside its own input window
    clock.set(DAY1, "15:20:05")
    inputs = ob.run_inputs(DAY1)
    boot = [r for r in ob.records() if r["stage"] == "archive.1" and r["session"] == ANCHOR_DAY][0]
    assert boot["body"]["bootstrap"] is True and boot["body"]["canonical"] is True
    assert inputs["body"]["status"] == "ok" and inputs["body"]["prior"]["bootstrap"] is True
    clock.set(DAY1, "15:25:05")
    ob.run_decision(DAY1)
    clock.set(DAY1, "16:20:05")
    ob.run_archive(DAY1)
    clock.set(DAY2, "15:20:05")
    assert ob.run_inputs(DAY2)["body"]["status"] == "ok"


def test_failed_prior_archive_leaves_inputs_unavailable_after_slot_one(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:05")
    ob.run_decision(DAY1)
    # DAY1's own archive never happens
    clock.set(DAY2, "15:20:05")
    inputs = ob.run_inputs(DAY2)
    assert (
        inputs["body"]["faults"] == ["prior_anchor_missing"] and inputs["body"]["usable"] is False
    )
    clock.set(DAY2, "15:25:05")
    assert ob.run_decision(DAY2)["body"]["signal_side"] is None


def test_late_archive_inside_grace_is_canonical_but_marked_late_and_keeps_the_fault(tmp_path):
    script = [obs.Response(503, {}, b"down")] * 3
    transport = obs.SyntheticTransport(price_fn=Book(), closes=HALF_DAYS, script=script)
    ob, clock, deps = active(tmp_path, transport=transport)
    first = archive_day(ob, clock, ANCHOR_DAY)
    assert (
        first["body"]["canonical"] is False
        and "fetch_transient_exhausted" in first["body"]["faults"]
    )
    assert len(first["body"]["fetch"]["attempts"]) == 3  # every earlier payload retained
    clock.set(ANCHOR_DAY, "17:30:00")
    second = ob.run_archive(ANCHOR_DAY)
    body = second["body"]
    assert body["canonical"] and body["late_capture"] and not body["on_time"]
    assert "archive_late" in body["faults"] and body["capture_no"] == 2
    clock.set(ANCHOR_DAY, "18:00:00")
    with pytest.raises(obs.DuplicateStage):
        ob.run_archive(ANCHOR_DAY)


def test_archive_after_grace_is_refused_and_a_finish_after_grace_is_not_canonical(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:20:00")  # grace for the anchor ended strictly before this instant
    with pytest.raises(obs.ObserverError, match="grace_expired"):
        ob.run_archive(ANCHOR_DAY)

    class Slow(obs.SyntheticTransport):
        def get(self, url, params):
            self.clock.now += timedelta(seconds=5)
            return super().get(url, params)

    slow = Slow(price_fn=Book(), closes=HALF_DAYS)
    ob2, clock2, _ = active(tmp_path / "second", transport=slow)
    slow.clock = clock2
    clock2.set(DAY1, "15:19:58")
    body = ob2.run_archive(ANCHOR_DAY)["body"]
    assert body["valuation_valid"] and not body["canonical"]
    assert "completed_after_grace" in body["faults"]


def test_early_close_skips_trading_stages_and_archives_the_1259_anchor(tmp_path):
    book = Book().day(HALF_DAY, close=104.0).day(DAY_AFTER_HALF, p0959=105.0)
    ob, clock, deps = active(tmp_path, book)
    clock.set(HALF_DAY, "15:20:05")
    assert ob.run_inputs(HALF_DAY)["body"]["reason"] == "early_close"
    clock.set(HALF_DAY, "15:25:05")
    assert ob.run_decision(HALF_DAY)["body"]["status"] == "not_applicable"
    calls = len(deps.transport.calls)
    assert calls == 0  # no cash-market observation for a non-trading session
    clock.set(HALF_DAY, "16:20:05")
    archive = ob.run_archive(HALF_DAY)
    assert (
        archive["body"]["close_minute"] == "12:59"
        and archive["body"]["values"]["close_final"] == 104.0
    )
    assert deps.transport.calls[-1][1]["end"] == "2026-11-27T17:59:00Z"
    clock.set(DAY_AFTER_HALF, "15:20:05")
    assert ob.run_inputs(DAY_AFTER_HALF)["body"]["prior"]["close_minute"] == "12:59"


def test_missing_entry_bar_and_corrupt_archive_leave_valuation_unresolved(tmp_path):
    class Gap(obs.SyntheticTransport):
        def get(self, url, params):
            reply = super().get(url, params)
            body = json.loads(reply.body)
            body["bars"]["SPY"] = [
                b for b in body["bars"]["SPY"] if not b["t"].endswith("T20:30:00Z")
            ]
            return obs.Response(200, {}, json.dumps(body).encode())

    ob, clock, _ = active(tmp_path, transport=Gap(price_fn=Book(), closes=HALF_DAYS))
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:05")
    ob.run_decision(DAY1)
    body = archive_day(ob, clock, DAY1)["body"]
    assert body["valuation_valid"] is False and "missing_bar:open_1530" in body["faults"]
    assert body["values"]["open_1530"] is None  # unknown stays unknown, never zero
    corrupt = [obs.Response(200, {}, bar_bytes(["09:30", "09:30"], day=DAY2))]
    ob2, clock2, _ = active(tmp_path / "x", transport=obs.SyntheticTransport(script=corrupt))
    body = archive_day(ob2, clock2, ANCHOR_DAY)["body"]
    assert not body["valuation_valid"] and body["fetch"]["defects"]
    raw = body["fetch"]["attempts"][0]["raw_path"]
    assert (tmp_path / "x" / "state" / raw).exists()


def test_budget_exhaustion_is_recorded_as_a_fault_not_a_crash(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:00:00")
    for _ in range(12):
        ob.reserve_request(DAY1, "other")
    clock.set(DAY1, "15:20:05")
    inputs = ob.run_inputs(DAY1)  # slot 1 needs the bootstrap anchor and the 09:59 bar
    boot = [r for r in ob.records() if r["stage"] == "archive.1"][0]
    assert "fetch_budget_exhausted" in boot["body"]["faults"] and not boot["body"]["canonical"]
    assert (
        inputs["body"]["faults"] == ["prior_anchor_missing"] and inputs["body"]["usable"] is False
    )


# --- halting, resume, amendments, revisions -----------------------------------------------------

DAY3, DAY4 = "2026-11-04", "2026-11-05"


def run_inputs_and_decision(ob, clock, day):
    clock.set(day, "15:20:05")
    ob.run_inputs(day)
    clock.set(day, "15:25:05")
    return ob.run_decision(day)


def halted_scenario(tmp_path):
    """Traded day 1 is never valued, so its grace expires; day 3 has a valid signal while halted."""
    book = (
        Book()
        .day(ANCHOR_DAY, close=100.0)
        .day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
        .day(DAY2, p0959=102.0, p1530=100.0, close=101.0)
        .day(DAY3, p0959=105.0, p1530=100.0, close=101.0)
    )
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    run_inputs_and_decision(ob, clock, DAY1)  # long, then the archive is never captured
    run_inputs_and_decision(ob, clock, DAY2)  # prior anchor missing: no timely signal
    archive_day(ob, clock, DAY2)
    return ob, clock, deps


def test_unvalued_traded_day_halts_new_positions_but_signals_continue(tmp_path):
    ob, clock, _ = halted_scenario(tmp_path)
    decision = run_inputs_and_decision(ob, clock, DAY3)["body"]
    assert decision["status"] == "halted_no_position" and decision["halted"] is True
    assert decision["signal_side"] == 1 and decision["observer_side"] == 0  # separate fields
    assert decision["halt_reasons"] == [{"reason": "unresolved_valuation", "origin": DAY1}]


def test_unvalued_flat_day_does_not_halt(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=99.0).day(DAY2, p0959=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    run_inputs_and_decision(ob, clock, DAY1)  # flat, never archived
    run_inputs_and_decision(ob, clock, DAY2)
    archive_day(ob, clock, DAY2)
    decision = run_inputs_and_decision(ob, clock, DAY3)["body"]
    assert decision["halted"] is False and decision["observer_side"] == decision["signal_side"]


def test_conflicting_decisions_and_ledger_tampering_halt(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=99.0).day(DAY2, p0959=99.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    run_inputs_and_decision(ob, clock, DAY1)
    archive_day(ob, clock, DAY1)
    ob.has_record = lambda *args: False  # force a second decision record for the same session
    ob.append("stage", DAY1, "decision", {"signal_side": 1, "observer_side": 1}, clock())
    del ob.has_record
    run_inputs_and_decision(ob, clock, DAY2)
    archive_day(ob, clock, DAY2)
    decision = run_inputs_and_decision(ob, clock, DAY3)["body"]
    reasons = {r["reason"] for r in decision["halt_reasons"]}
    assert "conflicting_decisions" in reasons and decision["observer_side"] == 0
    path = next((tmp_path / "state" / "records").glob("000002__*"))
    path.write_text(path.read_text().replace('"canonical": true', '"canonical": false'))
    assert any(r["reason"] == "ledger_integrity" for r in ob.halt_reasons(DAY4))


def test_resume_needs_owner_approval_and_cannot_be_backdated(tmp_path):
    ob, clock, _ = halted_scenario(tmp_path)
    run_inputs_and_decision(ob, clock, DAY3)
    clock.set(DAY3, "17:00:00")
    with pytest.raises(obs.ObserverError, match="owner_approval_required"):
        ob.amend_resume(DAY4, "")
    clock.set(DAY4, "15:20:00")
    with pytest.raises(obs.ClockMismatch):  # the effective session's 15:20 stage has started
        ob.amend_resume(DAY4, "owner approved")
    clock.set(DAY3, "17:00:00")
    ob.amend_resume(DAY4, "owner approved resume after review")
    assert ob.halt_reasons(DAY4) == []  # resume clears origins before DAY4 only
    assert [r["origin"] for r in ob.halt_reasons(DAY3)] == [DAY1]  # history is not rewritten


def test_calendar_amendments_preserve_the_original_schedule(tmp_path):
    ob, clock, deps = active(tmp_path)
    clock.set(DAY2, "10:00:00")
    ob.amend_session(DAY2, "exchange_closed", None, "owner approved closure")
    clock.set(DAY2, "15:20:05")
    inputs = ob.run_inputs(DAY2)
    assert inputs["body"]["reason"] == "exchange_closed" and deps.transport.calls == []
    amendment = [r for r in ob.records() if r["kind"] == "amendment"][0]
    assert (
        amendment["body"]["original"]["kind"] == "full"
        and amendment["body"]["revised"]["kind"] == "closed"
    )
    assert ob.session_row(DAY2)["slot"] == 2  # the slot is retained, the endpoint is not extended
    clock.set(DAY3, "10:00:00")
    ob.amend_session(DAY3, "early_close", "13:00", "owner approved early close")
    clock.set(DAY3, "15:20:05")
    assert ob.run_inputs(DAY3)["body"]["reason"] == "early_close"
    clock.set(DAY3, "16:20:05")
    assert ob.run_archive(DAY3)["body"]["close_minute"] == "12:59"
    assert deps.transport.calls[-1][1]["end"] == "2026-11-04T17:59:00Z"


def test_revision_is_a_separate_appendix_record_and_never_replaces_the_canonical_value(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0)
    ob, clock, deps = active(tmp_path, book)
    original = archive_day(ob, clock, ANCHOR_DAY)
    book.day(ANCHOR_DAY, close=100.5)  # the vendor later publishes a different value
    clock.set(DAY1, "09:00:00")
    revision = ob.run_revise(ANCHOR_DAY)
    assert revision["stage"] == "revision.1" and revision["body"]["appendix_only"] is True
    assert revision["body"]["differences"] == {
        "close_final": {"canonical": 100.0, "revised": 100.5}
    }
    assert ob.canonical(ANCHOR_DAY)["sha256"] == original["sha256"]
    assert ob.run_revise(ANCHOR_DAY)["stage"] == "revision.2"


# --- arithmetic: signal, costs, tax, statistics -------------------------------------------------


def test_signal_rule_is_strictly_positive_first_return():
    assert obs.signal_side(100.0, 100.01) == 1
    assert obs.signal_side(100.0, 100.0) == 0 and obs.signal_side(100.0, 99.0) == 0
    assert obs.first_return(100.0, 101.0) == pytest.approx(0.01)


def test_compound_applies_round_trip_costs_to_traded_days_only():
    entries = [("2026-11-02", 1, 0.01), ("2026-11-03", 0, 0.5), ("2026-11-04", 1, -0.005)]
    base = obs.compound(entries, 1e-4, [], tax=False)
    assert base["returns"] == pytest.approx([0.0098, 0.0, -0.0052])
    assert base["wealth_end"] == pytest.approx(100000 * 1.0098 * 0.9948)
    stress = obs.compound(entries, 2e-4, [], tax=False)
    assert stress["wealth_end"] == pytest.approx(100000 * 1.0096 * 0.9946)
    gross = obs.compound(entries, 0.0, [], tax=False)
    assert gross["wealth_end"] == pytest.approx(100000 * 1.01 * 0.995)


def test_tax_debits_24pct_of_positive_after_cost_year_pnl_and_resizes_next_year():
    entries = [("2026-12-30", 1, 0.0202), ("2027-01-04", 1, 0.0202), ("2027-01-05", 1, -0.1)]
    result = obs.compound(entries, 1e-4, ["2026-12-31"], tax=True)
    pnl_2026 = 100000 * 0.02  # 0.0202 less 2 x 1 bp
    tax_2026 = 0.24 * pnl_2026
    wealth = 100000 + pnl_2026 - tax_2026
    annual = result["annual"][0]
    assert annual["year"] == 2026 and annual["net_pnl"] == pytest.approx(pnl_2026)
    assert annual["tax_debit"] == pytest.approx(tax_2026)
    assert annual["wealth_after_tax"] == pytest.approx(wealth)
    wealth_2027 = wealth * 1.02 * 0.8998  # debit lowered the sizing base
    assert result["wealth_end"] == pytest.approx(wealth_2027)
    # 2027 net P&L is negative: no benefit, no carryforward, nothing accrued at the end
    assert result["terminal_accrual"] == 0.0
    assert result["wealth_end_after_terminal"] == pytest.approx(wealth_2027)


def test_terminal_partial_year_tax_accrues_without_changing_the_path():
    result = obs.compound([("2026-11-02", 1, 0.0102)], 1e-4, ["2026-12-31"], tax=True)
    assert result["annual"] == [] and result["wealth_end"] == pytest.approx(101000.0)
    assert result["terminal_accrual"] == pytest.approx(0.24 * 1000.0)
    assert result["wealth_end_after_terminal"] == pytest.approx(101000.0 - 240.0)


def test_unknown_valuation_leaves_wealth_unknown_and_never_zero():
    entries = [("2026-11-02", 1, 0.01), ("2026-11-03", 1, None), ("2026-11-04", 0, 0.0)]
    result = obs.compound(entries, 1e-4, [], tax=False)
    assert result["wealth_end"] is None and result["unknown_from"] == "2026-11-03"
    assert result["returns"] == pytest.approx([0.0098])
    flat_unknown_price = obs.compound([("2026-11-02", 0, None)], 1e-4, [], tax=False)
    assert flat_unknown_price["wealth_end"] == pytest.approx(100000.0)  # a flat day needs no price
    assert obs.compound([("2026-11-02", None, 0.01)], 1e-4, [], tax=False)["wealth_end"] is None


def test_newey_west_matches_an_independent_bartlett_hac_computation():
    import numpy as np

    values = np.array([0.004, -0.002, 0.006, 0.001, -0.003, 0.005, 0.002, -0.001, 0.003, 0.0])
    n, e = len(values), values - values.mean()
    gamma = [float(e[j:] @ e[: n - j]) for j in range(6)]
    long_run = gamma[0] + 2 * sum((1 - j / 6) * gamma[j] for j in range(1, 6))
    expected = values.mean() / np.sqrt(long_run / n**2 * n / (n - 1))
    assert obs.nw_t(values) == pytest.approx(expected)
    assert obs.nw_t([0.001] * 8) is None and obs.nw_t([0.1]) is None
    assert obs.nw_t([0.1, float("nan"), 0.2]) is None


def test_descriptive_statistics_for_a_known_series():
    entries = [("2026-11-02", 1, 0.01), ("2026-11-03", 0, 0.5), ("2026-11-04", 1, -0.02)]
    summary = obs.describe(entries, [], "2026-10-30")
    assert summary["counts"] == {"sessions": 3, "long": 2, "flat": 1}
    # break-even per-side cost: (0.01 - 0.02) / (2 * 2 traded days) is negative -> reported as is
    assert summary["break_even_cost_per_side"] == pytest.approx(-0.0025)
    base = summary["base"]
    assert base["pretax"]["wealth_end"] == pytest.approx(100000 * 1.0098 * 0.9798)
    assert base["max_drawdown"] == pytest.approx(0.9798 - 1)
    assert base["losing_streak"] == 1 and base["worst_month"] == pytest.approx(1.0098 * 0.9798 - 1)
    years = (date(2026, 11, 4) - date(2026, 10, 30)).days / 365.25
    assert base["cagr_pretax"] == pytest.approx((1.0098 * 0.9798) ** (1 / years) - 1)
    assert base["tax_drag_wealth"] == 0.0 and summary["through_date"] == "2026-11-04"


# --- three synthetic sessions, report, distributions --------------------------------------------


def three_sessions(tmp_path):
    """Long +1%, flat, long -2%: known cost, wealth and statistic arithmetic."""
    book = (
        Book()
        .day(ANCHOR_DAY, close=100.0)
        .day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
        .day(DAY2, p0959=100.0, p1530=100.0, close=101.0)
        .day(DAY3, p0959=102.0, p1530=104.0, close=101.92)
    )
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    for day in (DAY1, DAY2, DAY3):
        full_day(ob, clock, day)
    clock.set(DAY3, "17:00:00")
    return ob, clock, deps


def keys_of(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            keys_of(item, found)
    elif isinstance(value, list):
        for item in value:
            keys_of(item, found)
    return found


def test_report_for_three_sessions_has_known_counts_and_arithmetic(tmp_path):
    ob, clock, _ = three_sessions(tmp_path)
    report = ob.report()
    assert report["identity"]["activated"] is True and report["identity"]["mode"] == "live"
    assert report["identity"]["source_preflight"] == "verified"
    assert (
        "locally" in report["evidence_note"].lower() or "local" in report["evidence_note"].lower()
    )
    slots = report["slots"]
    assert (slots["registered"], slots["future"], slots["started_full_sessions"]) == (252, 249, 3)
    assert (slots["long"], slots["flat"], slots["no_timely_decision"]) == (2, 1, 0)
    assert slots["valuation"]["ok"] == 3 and report["faults"] == []
    fixed, observer = report["fixed_rule"], report["observer"]
    assert fixed["complete"] is False and fixed["partial_through"] == DAY3
    assert fixed["gaps"] == []
    base = fixed["statistics"]["base"]
    assert base["pretax"]["wealth_end"] == pytest.approx(100000 * 1.0098 * 0.9798)
    assert fixed["statistics"]["gross"]["pretax"]["wealth_end"] == pytest.approx(
        100000 * 1.01 * 0.98
    )
    assert fixed["statistics"]["stress"]["pretax"]["wealth_end"] == pytest.approx(
        100000 * 1.0096 * 0.9796
    )
    assert observer["statistics"]["base"]["pretax"]["wealth_end"] == pytest.approx(
        base["pretax"]["wealth_end"]
    )
    assert report["operational"]["checkpoint"]["reached"] is False
    assert report["benchmark"]["status"] == "incomplete"
    assert "no_distribution_schedule" in report["benchmark"]["reasons"]


def test_report_has_no_performance_verdict_fields(tmp_path):
    ob, _, _ = three_sessions(tmp_path)
    words = {w for key in keys_of(ob.report()) for w in str(key).lower().split("_")}
    forbidden = {"pass", "passed", "verdict", "threshold", "go", "nogo", "significant", "a1", "a2"}
    assert not words & forbidden


def test_report_keeps_every_fault_and_reports_gaps_instead_of_inventing_zeros(tmp_path):
    book = (
        Book()
        .day(ANCHOR_DAY, close=100.0)
        .day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
        .day(DAY3, p0959=102.0)
    )
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    full_day(ob, clock, DAY1)
    # DAY2: nobody ran any stage. DAY3: the prior anchor is therefore missing.
    run_inputs_and_decision(ob, clock, DAY3)
    archive_day(ob, clock, DAY3, "16:30:00")
    clock.set(DAY3, "17:00:00")
    report = ob.report()
    faults = {(f["session"], f["fault"]) for f in report["faults"]}
    assert (DAY2, "missing_stage:inputs") in faults and (DAY2, "missing_stage:decision") in faults
    assert (DAY2, "missing_stage:archive") in faults and (DAY2, "unresolved_valuation") in faults
    assert (DAY3, "prior_anchor_missing") in faults
    fixed = report["fixed_rule"]
    assert fixed["partial_through"] == DAY1 and fixed["complete"] is False
    assert fixed["gaps"][0] == {"date": DAY2, "reason": "no_timely_signal"}
    observer = report["observer"]["statistics"]["base"]["pretax"]
    assert observer["wealth_end"] == pytest.approx(100000 * 1.0098)  # known: flat after day 1
    assert report["slots"]["no_timely_decision"] == 2


def test_unvalued_trade_keeps_observer_wealth_unknown(tmp_path):
    ob, clock, _ = halted_scenario(tmp_path)
    run_inputs_and_decision(ob, clock, DAY3)
    clock.set(DAY3, "17:00:00")
    report = ob.report()
    base = report["observer"]["statistics"]["base"]
    assert base["pretax"]["wealth_end"] is None and base["pretax"]["unknown_from"] == DAY1
    assert report["observer"]["halting_faults"][0]["reason"] == "unresolved_valuation"
    assert report["slots"]["halted_no_position"] == 1
    assert report["fixed_rule"]["gaps"][0] == {"date": DAY1, "reason": "unresolved_valuation"}


def test_amended_closed_and_early_close_slots_are_counted_not_returned(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set(DAY2, "10:00:00")
    ob.amend_session(DAY2, "exchange_closed", None, "owner approved")
    clock.set(DAY2, "17:00:00")
    report = ob.report()
    assert report["slots"]["exchange_closed"] == 1 and report["slots"]["registered"] == 252
    assert report["calendar"]["amendments"][0]["revised"]["kind"] == "closed"
    assert report["calendar"]["early_closes_registered"] == ob.manifest["calendar"]["early_closes"]


def make_xlsx(rows: list[list], sheet: str = "Distributions") -> bytes:
    strings: list[str] = []

    def cell(ref: str, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            strings.append(value)
            return f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'

    body = ""
    for r, row in enumerate(rows, start=1):
        cells = "".join(cell(f"{chr(65 + c)}{r}", v) for c, v in enumerate(row))
        body += f'<row r="{r}">{cells}</row>'
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    sheets = (
        f'<sheet name="Other" sheetId="1" r:id="rId1"/>'
        f'<sheet name="{sheet}" sheetId="2" r:id="rId2"/>'
    )
    links = (
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Target="worksheets/sheet2.xml"/>'
    )
    files = {
        "xl/workbook.xml": (
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>{sheets}</sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{pkg}">{links}</Relationships>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{ns}"><sheetData/></worksheet>',
        "xl/worksheets/sheet2.xml": (
            f'<worksheet xmlns="{ns}"><sheetData>{body}</sheetData></worksheet>'
        ),
    }
    files["xl/sharedStrings.xml"] = (
        f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def serial(day: str) -> int:
    return (date.fromisoformat(day) - date(1899, 12, 30)).days


COLUMNS = {
    "ex_date": "Ex-Date",
    "record_date": "Record Date",
    "pay_date": "Pay Date",
    "dividend": "Dividend",
    "capital_gains": ["Capital Gain"],
}


def ingest(ob, tmp_path, rows, **overrides):
    path = tmp_path / "issuer.xlsx"
    path.write_bytes(make_xlsx([["CUSIP", *rows[0]], *[["78462F103", *r] for r in rows[1:]]]))
    args = {
        "sheet": "Distributions",
        "source_url": "https://issuer.example/spy-distributions.xlsx",
        "cusip": "78462F103",
        "fund": "SPDR S&P 500 ETF Trust",
        "currency": "USD",
        "decimals": 4,
        "header_row": 1,
        "columns": {**COLUMNS, "cusip": "CUSIP"},
        "date_format": "excel_serial",
        "covers_through": ob.manifest["calendar"]["last_date"],
        "semantics_verified": True,
    }
    args.update(overrides)
    return ob.ingest_distributions(path, **args)


HEADER = ["Ex-Date", "Record Date", "Pay Date", "Dividend", "Capital Gain"]


def test_xlsx_reader_and_distribution_freeze_keep_source_bytes_and_unknown_blanks(tmp_path):
    ob, clock, _ = three_sessions(tmp_path)
    rows = [HEADER, [serial(DAY3), serial(DAY3), serial("2026-11-20"), "1.01", None]]
    record = ingest(ob, tmp_path, rows)
    body = record["body"]
    assert (
        body["source_sha256"] == hashlib.sha256((tmp_path / "issuer.xlsx").read_bytes()).hexdigest()
    )
    assert (tmp_path / "state" / body["raw_path"]).exists() and body["defects"] == []
    row = body["rows"][0]
    assert row["ex_date"] == DAY3 and row["dividend"] == "1.01"
    assert row["capital_gains"] == {"Capital Gain": None}  # blank is unknown, never zero
    assert body["fund"] == "SPDR S&P 500 ETF Trust" and body["cusip"] == "78462F103"


@pytest.mark.parametrize(
    "rows, defect",
    [
        (
            [
                HEADER,
                [serial(DAY3), None, None, "1.0", "0"],
                [serial(DAY3), None, None, "1.0", "0"],
            ],
            "duplicate_ex_date",
        ),
        ([HEADER, [serial(DAY3), None, None, "0", "0"]], "non_positive_dividend"),
        (
            [HEADER, [serial("2026-11-07"), None, None, "1.0", "0"]],
            "ex_date_not_session",
        ),  # Saturday
        ([HEADER, [serial(DAY3), None, serial("2026-11-01"), "1.0", "0"]], "pay_before_ex"),
    ],
)
def test_distribution_rows_require_unique_session_ex_dates_and_positive_amounts(
    tmp_path, rows, defect
):
    ob, _, _ = three_sessions(tmp_path)
    defects = ingest(ob, tmp_path, rows)["body"]["defects"]
    assert any(d.startswith(defect) for d in defects)


def test_vendor_comparison_rounds_half_up_to_issuer_decimals():
    issuer = [{"ex_date": DAY3, "dividend": "1.0250"}]
    assert obs.compare_vendor(issuer, [{"ex_date": DAY3, "amount": 1.03}], 2) == []
    assert obs.compare_vendor(issuer, [{"ex_date": DAY3, "amount": 1.02}], 2) == [
        {"ex_date": DAY3, "issuer": "1.03", "vendor": "1.02"}
    ]
    assert obs.compare_vendor(issuer, [], 2) == []  # no overlapping vendor row: nothing to compare


def test_benchmark_reinvests_issuer_distributions_and_stays_provisional_until_complete(tmp_path):
    ob, clock, _ = three_sessions(tmp_path)
    ingest(ob, tmp_path, [HEADER, [serial(DAY3), serial(DAY3), serial("2026-11-20"), "1.01", "0"]])
    report = ob.report()
    bench = report["benchmark"]
    assert bench["status"] == "provisional"
    assert bench["reasons"] == ["obtained_before_final_session"]
    # 1000 units bought at the 100 anchor, reinvesting 1.01 at the 101.92 close of the ex-date
    assert bench["total_return_wealth_end"] == pytest.approx(1000 * (101.92 + 1.01))
    assert bench["through_date"] == DAY3
    sensitivity = report["fixed_rule"]["ex_distribution_flat_sensitivity"]
    assert sensitivity["status"] == "provisional"
    assert sensitivity["base_pretax_wealth_end"] == pytest.approx(100000 * 1.0098)  # DAY3 zeroed
    primary = report["fixed_rule"]["statistics"]["base"]["pretax"]["wealth_end"]
    assert primary == pytest.approx(100000 * 1.0098 * 0.9798)  # primary path untouched


def test_unknown_currency_or_blank_capital_gain_keeps_the_benchmark_unresolved(tmp_path):
    ob, clock, _ = three_sessions(tmp_path)
    ingest(
        ob,
        tmp_path,
        [HEADER, [serial(DAY3), None, None, "1.01", None]],
        currency=None,
        semantics_verified=False,
    )
    bench = ob.report()["benchmark"]
    for reason in (
        "currency_unresolved",
        "field_semantics_unverified",
        "capital_gain_blank_unknown",
    ):
        assert reason in bench["reasons"]
    assert bench["status"] == "provisional"


# --- recovery, concurrency, atomicity, faults through stages ------------------------------------


def test_recover_fills_a_missed_prior_archive_before_the_next_inputs_stage(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, close=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    run_inputs_and_decision(ob, clock, DAY1)  # DAY1's 16:20 archive job never ran
    clock.set(DAY2, "15:14:59")
    with pytest.raises(obs.TooEarly):
        ob.run_recover(DAY2)
    clock.set(DAY2, "15:15:00")
    record = ob.run_recover(DAY2)
    assert (
        record["session"] == DAY1 and record["body"]["late_capture"] and record["body"]["canonical"]
    )
    assert "archive_late" in record["body"]["faults"]
    assert ob.run_recover(DAY2) is None  # nothing left to recover
    clock.set(DAY2, "15:20:05")
    assert ob.run_inputs(DAY2)["body"]["prior"]["late_capture"] is True


def test_transport_deadline_is_a_retryable_timeout(tmp_path):
    script = [obs.RequestDeadline()] * 3
    ob, clock, deps = make_observer(
        tmp_path, obs.SyntheticTransport(script=script), start=("2026-11-02", "15:20:00")
    )
    result = run_fetch(ob, clock)
    assert result["outcome"] == "transient_exhausted" and len(deps.transport.calls) == 3
    assert {a["transport_error"] for a in result["attempts"]} == {"timeout"}


def test_stale_credentials_become_a_recorded_fault_and_are_not_retried(tmp_path):
    transport = obs.SyntheticTransport(
        price_fn=Book().day(ANCHOR_DAY), script=[obs.Response(401, {}, b"unauthorized")] * 2
    )
    ob, clock, deps = active(tmp_path, transport=transport)
    body = archive_day(ob, clock, ANCHOR_DAY)["body"]
    assert body["faults"] == ["fetch_authentication_error"] and body["canonical"] is False
    clock.set(DAY1, "15:20:05")
    inputs = ob.run_inputs(DAY1)  # bootstrap fetch fails the same way
    assert "prior_anchor_missing" in inputs["body"]["faults"]
    assert len(deps.transport.calls) == 2


def test_unknown_session_is_refused(tmp_path):
    ob, clock, _ = active(tmp_path)
    clock.set("2026-12-25", "15:20:05")
    with pytest.raises(obs.ObserverError, match="unknown_session"):
        ob.run_inputs("2026-12-25")


def test_manifest_is_unchanged_by_activation_and_stays_unverified(tmp_path):
    ob, clock, _ = make_observer(tmp_path, start=(ACTIVATION, "10:00:00"))
    before = (tmp_path / "state" / "manifest.json").read_bytes()
    ob.activate("owner approved", readiness_file=write_readiness(ob))
    assert (tmp_path / "state" / "manifest.json").read_bytes() == before
    manifest = json.loads(before)
    assert manifest["activated"] is False and manifest["source_preflight"] == "not_performed"


def test_concurrent_appends_get_unique_sequence_numbers_and_exactly_one_duplicate_wins(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ob, clock, _ = make_observer(tmp_path)
    when = clock()

    def same_key(n):
        try:
            return obs.Observer(tmp_path / "state", ob.deps).append(
                "stage", DAY1, "decision", {"writer": n}, when
            )
        except obs.DuplicateStage:
            return None

    with ThreadPoolExecutor(8) as pool:
        winners = [r for r in pool.map(same_key, range(8)) if r]
    assert len(winners) == 1
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda n: ob.append("stage", DAY2, f"s{n}", {}, when), range(8)))
    assert [r["seq"] for r in ob.records()] == list(range(1, 10))
    assert ob.verify_chain() == []


def test_failed_write_leaves_no_partial_record_and_does_not_consume_a_sequence(
    tmp_path, monkeypatch
):
    ob, clock, _ = make_observer(tmp_path)
    monkeypatch.setattr(
        obs, "atomic_checkpoint", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        ob.append("stage", DAY1, "inputs", {}, clock())
    monkeypatch.undo()
    assert ob.records() == [] and not ob.has_record(DAY1, "inputs")
    assert ob.append("stage", DAY1, "inputs", {}, clock())["seq"] == 1


# --- command line, end to end with injected collaborators ---------------------------------------


def run_cli(argv, deps, capsys):
    code = obs.main(argv, deps)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_prepares_activates_runs_a_session_and_reports(tmp_path, capsys, monkeypatch):
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.setenv(name, "unused-in-this-test")
    protocol, digest = write_protocol(tmp_path)
    snapshot = write_calendar(tmp_path)
    state = tmp_path / "state"
    clock = FakeClock(at(ACTIVATION, "09:00:00"))
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
    deps = obs.Deps(
        clock=clock,
        sleep=clock.sleep,
        transport=obs.SyntheticTransport(price_fn=book, closes=HALF_DAYS),
        clock_sync=lambda: "synchronized",
    )
    base = ["--state-dir", str(state)]
    code, out, err = run_cli(
        [
            "init",
            *base,
            "--activation-date",
            ACTIVATION,
            "--official-calendar",
            str(snapshot),
            "--protocol-file",
            str(protocol),
            "--protocol-sha256",
            digest,
            "--source-delay-seconds",
            "900",
        ],
        deps,
        capsys,
    )
    assert code == 0 and json.loads(out)["activated"] is False
    code, _, err = run_cli(["inputs", *base, "--session", DAY1], deps, capsys)
    assert code == 2 and "NotActivated" in err  # prepared is not active
    ready = write_readiness(obs.Observer(state, deps))
    assert (
        run_cli(
            [
                "activate",
                *base,
                "--owner-approval",
                "synthetic test approval",
                "--readiness-file",
                str(ready),
            ],
            deps,
            capsys,
        )[0]
        == 0
    )
    steps = [
        (ANCHOR_DAY, "16:20:05", "archive"),
        (DAY1, "15:20:05", "inputs"),
        (DAY1, "15:25:05", "decide"),
        (DAY1, "16:20:05", "archive"),
    ]
    for day, text, command in steps:
        clock.set(day, text)
        code, out, err = run_cli([command, *base, "--session", day], deps, capsys)
        assert code == 0, err
        assert json.loads(out)["session"] == day
    code, out, _ = run_cli(["decide", *base, "--session", DAY1], deps, capsys)
    assert code == 2  # duplicates are refused
    code, out, _ = run_cli(["verify", *base], deps, capsys)
    assert code == 0 and json.loads(out)["problems"] == []
    target = tmp_path / "report.json"
    code, out, _ = run_cli(["report", *base, "--out", str(target)], deps, capsys)
    assert code == 0 and json.loads(target.read_text())["slots"]["long"] == 1
    assert run_cli(["report", *base, "--out", str(target)], deps, capsys)[0] == 2  # write-once


def test_cli_stage_commands_need_credentials_before_touching_state(tmp_path, capsys, monkeypatch):
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    ob, clock, _ = active(tmp_path)
    clock.set(DAY1, "15:20:05")
    before = len(ob.records())
    code, _, err = run_cli(
        ["inputs", "--state-dir", str(tmp_path / "state")], obs.Deps(clock=clock), capsys
    )
    assert code == 2 and "credentials_missing" in err
    assert len(ob.records()) == before
    assert not (tmp_path / "state" / "claims" / f"{DAY1}__inputs.claim").exists()


def test_cli_refuses_to_run_live_commands_on_a_rehearsal_state(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    make_observer(tmp_path)
    manifest_path = tmp_path / "state" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mode"] = "rehearsal"
    manifest_path.write_text(json.dumps(manifest))
    code, _, err = run_cli(["inputs", "--state-dir", str(tmp_path / "state")], obs.Deps(), capsys)
    assert code == 2 and "rehearsal" in err


# --- rehearsal ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation, first_slot, anchor_minute",
    [
        ("2026-11-23", "2026-11-30", "12:59"),  # anchor is the early close after Thanksgiving
        (ACTIVATION, "2026-11-02", "15:59"),  # slot 1 is the first session after the DST change
    ],
)
def test_rehearsal_runs_three_synthetic_sessions_offline(
    tmp_path, capsys, no_network, activation, first_slot, anchor_minute
):
    work = tmp_path / "work"
    code, out, err = run_cli(
        ["rehearse", "--work-dir", str(work), "--sessions", "3", "--activation-date", activation],
        obs.Deps(),
        capsys,
    )
    assert code == 0, err
    summary = json.loads(out)
    report = json.loads(Path(summary["report_path"]).read_text())
    assert (
        report["identity"]["mode"] == "rehearsal" and report["calendar"]["first_date"] == first_slot
    )
    assert report["slots"]["started_full_sessions"] == 3 and report["faults"] == []
    assert (report["slots"]["long"], report["slots"]["flat"]) == (2, 1)
    anchor = report["calendar"]["anchor_date"]
    ob = obs.Observer(work / "state")
    assert ob.canonical(anchor)["body"]["close_minute"] == anchor_minute
    assert ob.verify_chain() == []
    base = report["fixed_rule"]["statistics"]["base"]["pretax"]["wealth_end"]
    assert base == pytest.approx(report["observer"]["statistics"]["base"]["pretax"]["wealth_end"])
    assert base > 0
    code, _, _ = run_cli(["rehearse", "--work-dir", str(work)], obs.Deps(), capsys)
    assert code == 2  # never into an existing directory


# --- script entry point, host evidence, deadline, isolation ------------------------------------


def run_script(*argv, cwd):
    env = {"PATH": os.environ["PATH"], "HOME": str(cwd)}  # no credentials, no proxies
    return subprocess.run(
        [sys.executable, str(CODE / "intraday_observe.py"), *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_script_help_and_plan_run_without_credentials_or_writes(tmp_path):
    helped = run_script("--help", cwd=tmp_path)
    assert helped.returncode == 0 and "rehearse" in helped.stdout and "activate" in helped.stdout
    planned = run_script("plan", "--activation-date", ACTIVATION, cwd=tmp_path)
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)["source_preflight"] == "not_performed"
    assert list(tmp_path.iterdir()) == []
    bad = run_script("inputs", "--state-dir", str(tmp_path / "missing"), cwd=tmp_path)
    assert bad.returncode == 2 and "state_not_initialized" in bad.stderr
    assert list(tmp_path.iterdir()) == []  # a refused command creates nothing


def test_module_can_only_issue_a_bars_get():
    source = (CODE / "intraday_observer.py").read_text()
    for forbidden in ("/v2/orders", ".post(", ".delete(", ".put(", ".patch(", "paper-api"):
        assert forbidden not in source
    assert obs.BARS_URL == "https://data.alpaca.markets/v2/stocks/bars"


@pytest.mark.parametrize(
    "output, expected",
    [("yes\n", "synchronized"), ("no\n", "unsynchronized"), ("weird", "unknown")],
)
def test_host_clock_status_maps_timedatectl(monkeypatch, output, expected):
    done = subprocess.CompletedProcess([], 0, stdout=output)
    monkeypatch.setattr(obs.subprocess, "run", lambda *a, **k: done)
    assert obs.host_clock_status() == expected
    monkeypatch.setattr(obs.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert obs.host_clock_status() == "unknown"


def test_http_transport_enforces_a_total_deadline_on_a_slow_response(monkeypatch):
    import time as real_time

    class Slow:
        headers: dict = {}
        trust_env = True

        def get(self, *args, **kwargs):
            real_time.sleep(2)

    monkeypatch.setattr(obs, "REQUEST_DEADLINE_SECONDS", 0.3)
    transport = obs.HttpTransport(
        environ={"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}, session=Slow()
    )
    started = real_time.monotonic()
    reply = transport.get(obs.BARS_URL, {})
    assert reply.status is None and reply.error == "timeout"
    assert real_time.monotonic() - started < 1.5


def test_a_planted_future_record_cannot_change_the_decision(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    inputs = ob.run_inputs(DAY1)
    poison = {"canonical": True, "values": {"open_1530": 1.0, "close_final": 1e9}, "faults": []}
    ob.append("stage", DAY1, "archive.1", poison, clock())
    clock.set(DAY1, "15:25:05")
    body = ob.run_decision(DAY1)["body"]
    assert body["signal_side"] == 1 and body["inputs_ref"] == inputs["sha256"]
    assert set(body).isdisjoint({"open_1530", "close_final", "values"})


def test_archive_that_disagrees_with_the_frozen_input_is_flagged_not_applied(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    ob.run_inputs(DAY1)
    clock.set(DAY1, "15:25:05")
    decision = ob.run_decision(DAY1)
    book.day(DAY1, p0959=99.0, p1530=100.0, close=101.0)  # the vendor revised the 09:59 bar
    clock.set(DAY1, "16:20:05")
    archive = ob.run_archive(DAY1)
    assert archive["body"]["input_revision"] == {
        "input_close_0959": 101.0,
        "archive_close_0959": 99.0,
    }
    assert ob.stage_record(DAY1, "decision")["sha256"] == decision["sha256"]  # signal untouched


def test_losing_year_pays_no_tax_and_gets_no_refund_at_its_year_end():
    result = obs.compound(
        [("2026-12-30", 1, -0.0098), ("2027-01-04", 0, 0.0)], 1e-4, ["2026-12-31"], tax=True
    )
    assert result["annual"] == [
        {
            "year": 2026,
            "net_pnl": pytest.approx(-1000.0),
            "tax_debit": 0.0,
            "wealth_after_tax": pytest.approx(99000.0),
        }
    ]
    assert result["wealth_end"] == pytest.approx(99000.0) and result["terminal_accrual"] == 0.0


def test_inputs_finishing_exactly_at_the_cutoff_are_not_timely(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)

    class Exact(obs.SyntheticTransport):
        def get(self, url, params):
            self.clock.set(DAY1, "15:25:00")
            return super().get(url, params)

    transport = Exact(price_fn=book, closes=HALF_DAYS)
    ob, clock, _ = active(tmp_path, transport=transport)
    transport.clock = clock
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:24:59")
    body = ob.run_inputs(DAY1)["body"]
    assert body["timely"] is False and "inputs_completed_after_cutoff" in body["faults"]


# --- scheduler dispatch (the timer-launched wrapper) ---------------------------------------------

JOB_STAGES = ("recover", "inputs", "decide", "archive")
NON_SESSIONS = (
    "2026-10-29",  # before the anchor: a session, but outside the registered window
    "2026-10-31",  # Saturday
    "2026-11-26",  # Thanksgiving
)


def last_slot(ob):
    return [r for r in ob.effective_rows() if r["slot"]][-1]["date"]


@pytest.mark.parametrize("day", NON_SESSIONS)
@pytest.mark.parametrize("stage", JOB_STAGES)
def test_job_is_a_silent_noop_outside_registered_sessions(tmp_path, day, stage):
    ob, clock, deps = active(tmp_path)
    before = len(ob.records())
    clock.set(
        day,
        {"recover": "15:15:05", "inputs": "15:20:05", "decide": "15:25:05"}.get(stage, "16:20:05"),
    )
    result = obs.run_job(ob, stage)
    assert result["noop"] is True and result["reason"]
    assert len(ob.records()) == before and deps.transport.calls == []
    assert not list((ob.dir / "claims").iterdir()) and not list((ob.dir / "budget").iterdir())


def test_job_noop_uses_the_new_york_date_not_the_utc_date(tmp_path):
    ob, clock, deps = active(tmp_path)
    clock.now = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)  # Sunday noon ET on the DST change day
    assert obs.run_job(ob, "archive")["noop"] is True
    clock.now = datetime(2026, 11, 3, 3, 0, tzinfo=UTC)  # still Monday 22:00 ET, a session day
    result = obs.run_job(ob, "decide")  # a session day, so the stage is attempted (and faulted)
    assert result["session"] == DAY1 and "decision_window_missed" in result["faults"]


def test_job_runs_a_full_session_through_the_wrapper_with_the_same_records(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, p1530=100.0, close=101.0)
    ob, clock, _ = active(tmp_path, book)
    clock.set(ANCHOR_DAY, "16:20:05")
    assert obs.run_job(ob, "archive")["recorded"] is True  # the anchor session archives
    for stage in ("inputs", "decide"):  # but the anchor is not an observation slot
        clock.set(ANCHOR_DAY, "15:25:05")
        assert obs.run_job(ob, stage)["noop"] is True
    clock.set(ANCHOR_DAY, "15:15:05")
    assert obs.run_job(ob, "recover")["noop"] is True  # nothing earlier exists to recover
    out = []
    for stage, text in (
        ("recover", "15:15:05"),
        ("inputs", "15:20:05"),
        ("decide", "15:25:05"),
        ("archive", "16:20:05"),
    ):
        clock.set(DAY1, text)
        out.append(obs.run_job(ob, stage))
    assert [r.get("stage") for r in out[1:]] == [
        "inputs.completion",
        "decision.completion",
        "archive.1.completion",
    ]
    assert out[0]["noop"] is True  # the anchor already has its canonical value
    decision = ob.stage_record(DAY1, "decision")
    assert decision["body"]["observer_side"] == 1 and ob.verify_chain() == []


def test_job_early_close_runs_archive_and_recovery_only(tmp_path):
    book = Book().day(HALF_DAY, close=104.0).day(DAY_AFTER_HALF, p0959=105.0)
    ob, clock, deps = active(tmp_path, book)
    before = len(ob.records())
    for stage, text in (("inputs", "15:20:05"), ("decide", "15:25:05")):
        clock.set(HALF_DAY, text)
        result = obs.run_job(ob, stage)
        assert result["noop"] is True and result["reason"] == "early_close"
    assert len(ob.records()) == before and deps.transport.calls == []
    clock.set(HALF_DAY, "16:20:05")
    archive = obs.run_job(ob, "archive")
    assert archive["stage"] == "archive.1.completion"
    assert deps.transport.calls[-1][1]["end"].endswith("17:59:00Z")
    clock.set(DAY_AFTER_HALF, "15:15:05")  # the early close anchor is recovered like any other
    assert obs.run_job(ob, "recover")["noop"] is True


def test_job_recover_runs_on_an_early_close_day_for_the_prior_session(tmp_path):
    ob, clock, _ = active(tmp_path)
    prior = "2026-11-25"  # Wednesday before Thanksgiving; the next session is the Friday half day
    assert ob.neighbours(HALF_DAY)[0] == prior
    clock.set(HALF_DAY, "15:15:05")
    result = obs.run_job(ob, "recover")
    assert result["session"] == prior and result["stage"] == "archive.1.completion"
    assert ob.canonical(prior)["body"]["late_capture"] is True


def test_job_never_writes_a_retroactive_decision(tmp_path):
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0, close=101.0)
    ob, clock, _ = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    obs.run_job(ob, "inputs")
    clock.set(DAY1, "15:40:00")  # the timer fired after the window closed
    body = obs.run_job(ob, "decide")
    record = ob.stage_record(DAY1, "decision")["body"]
    assert body["faults"] == ["decision_window_missed"]
    assert record["signal_side"] is None and record["observer_side"] == 0
    clock.set(DAY2, "15:25:05")  # the next day's job cannot reach back to yesterday's session
    obs.run_job(ob, "decide")
    assert [r["session"] for r in ob.records() if r["stage"] == "decision"] == [DAY1, DAY2]


def test_job_recovers_the_final_session_on_the_next_session_without_a_slot_253(tmp_path):
    ob, clock, deps = active(tmp_path)
    final, after = last_slot(ob), ob.manifest["calendar"]["post_window_next_session"]
    assert ob.neighbours(final)[1] == after and ob.manifest["calendar"]["last_date"] == final
    clock.set(after, "15:14:59")
    with pytest.raises(obs.TooEarly):
        obs.run_job(ob, "recover")
    clock.set(after, "15:15:05")
    recovered = obs.run_job(ob, "recover")
    assert recovered["session"] == final and recovered["stage"] == "archive.1.completion"
    body = ob.canonical(final)["body"]
    assert body["late_capture"] and body["canonical"] and "archive_late" in body["faults"]
    for stage in ("inputs", "decide", "archive"):  # the day after the endpoint is never observed
        clock.set(after, "15:25:05" if stage != "archive" else "16:20:05")
        assert obs.run_job(ob, stage)["noop"] is True
    assert obs.run_job(ob, "recover")["noop"] is True  # nothing left to recover
    assert {r["session"] for r in ob.records()} == {None, final}
    assert ob.verify_chain() == [] and ob.report()["slots"]["registered"] == obs.SLOTS
    with pytest.raises(obs.ObserverError, match="unknown_session"):
        ob.run_inputs(after)


def test_job_final_recovery_expires_at_the_next_session_input_stage(tmp_path):
    ob, clock, _ = active(tmp_path)
    after = ob.manifest["calendar"]["post_window_next_session"]
    clock.set(after, "15:20:00")
    with pytest.raises(obs.ObserverError, match="grace_expired_use_revise"):
        obs.run_job(ob, "recover")
    assert ob.canonical(last_slot(ob)) is None


def test_job_stage_names_are_closed():
    with pytest.raises(SystemExit):
        obs.job_main(["--stage", "revise", "--state-dir", "x"])


def test_job_main_noop_needs_no_credentials_and_reports_json(tmp_path, capsys, monkeypatch):
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    ob, clock, _ = active(tmp_path)
    clock.set("2026-11-26", "16:20:05")
    deps = obs.Deps(clock=clock, clock_sync=lambda: "synchronized")
    code = obs.job_main(["--stage", "archive", "--state-dir", str(ob.dir)], deps)
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["noop"] is True and out["stage"] == "archive"
    clock.set(DAY1, "16:20:05")  # a real stage on a session day needs credentials before any write
    before = len(ob.records())
    assert obs.job_main(["--stage", "archive", "--state-dir", str(ob.dir)], deps) == 2
    assert "credentials_missing" in capsys.readouterr().err and len(ob.records()) == before


def test_job_main_reports_refusals_with_exit_code_2(tmp_path, capsys):
    ob, clock, deps = active(tmp_path)
    clock.set(DAY1, "15:00:00")
    assert obs.job_main(["--stage", "decide", "--state-dir", str(ob.dir)], deps) == 2
    assert "TooEarly" in capsys.readouterr().err and ob.records()[-1]["stage"] == "activation"
    assert obs.job_main(["--stage", "decide", "--state-dir", str(tmp_path / "nope")], deps) == 2


def drive_calendar(tmp_path, through_slot, faults, setup=None):
    """Run the timer wrapper on every calendar date with a fresh Observer per stage."""
    book = Book()
    transport = obs.SyntheticTransport(price_fn=book, closes=dict(HALF_DAYS))
    ob, clock, deps = active(tmp_path, transport=transport)
    slots = [r["date"] for r in ob.effective_rows() if r["slot"]]
    for day in slots:
        book.day(day, p0959=102.0, p1530=100.0, close=101.0)
    book.day(ANCHOR_DAY, close=101.0)
    last = slots[through_slot - 1]
    day, results = date.fromisoformat(ANCHOR_DAY), []
    while day.isoformat() <= last:
        text = day.isoformat()
        for stage, hhmm in (
            ("recover", "15:15:05"),
            ("inputs", "15:20:05"),
            ("decide", "15:25:05"),
            ("archive", "16:20:05"),
        ):
            clock.set(text, hhmm)
            if setup:
                setup(text, stage, ob, deps)
            if (text, stage) in faults:
                continue  # the timer never ran (host down, job killed)
            results.append((text, stage, obs.run_job(obs.Observer(ob.dir, deps), stage)))
        day += timedelta(days=1)
    return obs.Observer(ob.dir, deps), clock, slots, results


def test_job_fault_rehearsal_keeps_every_fault_and_never_decides_retroactively(tmp_path):
    slots_needed = 22

    def inject(day, stage, ob, deps):
        slots = [r["date"] for r in ob.effective_rows() if r["slot"]]
        index = slots.index(day) + 1 if day in slots else 0
        if index == 2 and stage == "inputs":  # outage during input capture
            deps.transport.script = [obs.Response(503, {}, b"down")] * 3
        if index == 4 and stage == "decide":  # the process dies after claiming the decision
            with obs.Observer(ob.dir, deps).claim(day, "decision"):
                pass
            deps.sleep(10)
        if index == 5 and stage == "recover":  # unexpected closure announced before the open
            deps.clock.now = obs.utc_at(day, "10:00:00")
            obs.Observer(ob.dir, deps).amend_session(day, "exchange_closed", None, "owner approved")
            deps.clock.set(day, "15:15:05")
        if index == 6 and stage == "recover":  # a listed full day becomes an early close
            deps.transport.closes[day] = "13:00"
            deps.clock.now = obs.utc_at(day, "10:00:00")
            obs.Observer(ob.dir, deps).amend_session(day, "early_close", "13:00", "owner approved")
            deps.clock.set(day, "15:15:05")

    probe, _, _ = make_observer(tmp_path / "probe")
    slots = [r["date"] for r in probe.effective_rows() if r["slot"]]
    skipped = {(slots[2], "archive")}  # day 3's 16:20 job never ran
    ob, clock, slots, results = drive_calendar(tmp_path / "run", slots_needed, skipped, inject)
    records = ob.records()
    decisions = [r for r in records if r["stage"] == "decision"]
    assert ob.verify_chain() == []
    # one immutable decision per observed full session; never a second, never a past one
    decided = [d["session"] for d in decisions]
    assert decided == sorted(set(decided)) and slots[4] not in decided and slots[5] not in decided
    assert len(decided) == slots_needed - 2
    for record in decisions:  # every decision record was written inside its own session's window
        day = record["session"]
        assert obs.utc_at(day, "15:25:00") <= obs.parse_iso(record["recorded_utc"])
        assert obs.parse_iso(record["recorded_utc"]) < obs.utc_at(day, "15:26:00")
    known = {None, ANCHOR_DAY, HALF_DAY, *slots[:slots_needed]}
    assert {r["session"] for r in records} <= known  # no weekend/holiday/post-window evidence
    faults = {(f["session"], f["stage"], f["fault"]) for f in ob.report()["faults"]}
    assert (slots[1], "inputs.completion", "fetch_transient_exhausted") in faults
    assert any(f[0] == slots[1] and f[2] == "inputs_unusable" for f in faults)
    assert (slots[2], "archive.1", "archive_late") in {(f[0], f[1][:9], f[2]) for f in faults}
    assert ob.stage_record(slots[3], "decision")["body"]["reconciled_claim"] is not None
    assert ob.canonical(slots[2])["body"]["late_capture"] is True
    # the clean-looking later sessions did not erase the month-one failure
    checkpoint = ob.report()["operational"]["checkpoint"]
    assert checkpoint["reached"] and not checkpoint["conditions_met"]
    day21 = ob.stage_record(slots[20], "decision")["body"]
    assert day21["status"] == "halted_no_position" and day21["signal_side"] == 1
    assert day21["observer_side"] == 0 and day21["halt_reasons"][0]["reason"] == "checkpoint_failed"
    slot_counts = ob.report()["slots"]
    assert (slot_counts["exchange_closed"], slot_counts["early_close"]) == (1, 1)
    assert ob.session_row(slots[4])["slot"] == 5  # closed slot retained, endpoint not extended
    assert not any(r["session"] == slots[5] and r["stage"] == "decision" for r in records)


def test_job_stage_windows_follow_the_spring_dst_change(tmp_path):
    ob, clock, deps = active(tmp_path)
    prior, day = "2027-03-12", "2027-03-15"  # clocks change on Sunday 2027-03-14
    assert ob.neighbours(day)[0] == prior
    clock.set(prior, "16:20:05")
    obs.run_job(ob, "archive")
    assert deps.transport.calls[-1][1]["end"] == "2027-03-12T20:59:00Z"  # EST: 15:59 ET
    clock.set(day, "15:20:05")
    assert obs.run_job(ob, "inputs")["status"] == "ok"
    assert deps.transport.calls[-1][1]["end"] == "2027-03-15T13:59:00Z"  # EDT: 09:59 ET
    assert obs.utc_at(day, "15:25:00") == datetime(2027, 3, 15, 19, 25, tzinfo=UTC)
    clock.now = datetime(2027, 3, 15, 19, 24, 59, tzinfo=UTC)
    with pytest.raises(obs.TooEarly):
        obs.run_job(ob, "decide")
    clock.now = datetime(2027, 3, 15, 19, 25, 5, tzinfo=UTC)
    assert obs.run_job(ob, "decide")["faults"] == []  # timely; halted only by the unobserved months
    assert ob.stage_record(day, "decision")["body"]["timely_valid"] is True


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("timedatectl"),
        subprocess.TimeoutExpired("timedatectl", 5),
        subprocess.CalledProcessError(1, "timedatectl"),
    ],
)
def test_clock_probe_failure_is_unknown_and_never_timely_evidence(tmp_path, monkeypatch, failure):
    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(obs.subprocess, "run", broken)
    assert obs.host_clock_status() == "unknown"
    book = Book().day(ANCHOR_DAY, close=100.0).day(DAY1, p0959=101.0)
    ob, clock, deps = active(tmp_path, book)
    archive_day(ob, clock, ANCHOR_DAY)
    clock.set(DAY1, "15:20:05")
    obs.run_job(ob, "inputs")
    deps.clock_sync = obs.host_clock_status  # the production probe, now failing
    clock.set(DAY1, "15:25:05")
    result = obs.run_job(ob, "decide")
    body = ob.stage_record(DAY1, "decision")["body"]
    assert "clock_unknown" in result["faults"] and body["observer_side"] == 0
    assert body["timely_valid"] is False and body["signal_side"] is None


def test_job_script_help_and_refusals_run_without_credentials_or_writes(tmp_path):
    def run_job_script(*argv):
        env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}  # no credentials, no proxies
        return subprocess.run(
            [sys.executable, str(CODE / "intraday_observer_job.py"), *argv],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    helped = run_job_script("--help")
    assert helped.returncode == 0 and "--stage" in helped.stdout and "--state-dir" in helped.stdout
    assert run_job_script("--stage", "revise", "--state-dir", "x").returncode == 2
    bad = run_job_script("--stage", "inputs", "--state-dir", str(tmp_path / "missing"))
    assert bad.returncode == 2 and "state_not_initialized" in bad.stderr
    assert list(tmp_path.iterdir()) == []  # a refused job creates nothing


def test_job_wrapper_is_part_of_the_code_binding():
    assert (
        "research/intraday/spy-momentum-prospective-observer/code/intraday_observer_job.py"
        in obs.code_binding()["files"]
    )


def test_recover_after_the_endpoint_captures_the_final_archive_directly(tmp_path):
    ob, clock, _ = active(tmp_path)
    final, after = last_slot(ob), ob.manifest["calendar"]["post_window_next_session"]
    assert after not in {r["date"] for r in ob.effective_rows()}  # no slot, no session row
    clock.set(after, "15:15:00")
    record = ob.run_recover(after)
    assert record["session"] == final and record["body"]["canonical"] is True
    assert ob.run_recover(after) is None


def test_readiness_review_init_prints_the_hash_activation_consumes(tmp_path, capsys):
    args = init_args(tmp_path)
    state = tmp_path / "cli-state"
    argv = [
        "init",
        "--state-dir",
        str(state),
        "--activation-date",
        ACTIVATION,
        "--official-calendar",
        str(args["official_calendar_file"]),
        "--protocol-file",
        str(args["protocol_file"]),
        "--protocol-sha256",
        args["protocol_sha256"],
        "--source-delay-seconds",
        "900",
    ]
    assert obs.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (
        printed["manifest_sha256"]
        == hashlib.sha256((state / "manifest.json").read_bytes()).hexdigest()
    )


def test_readiness_review_resume_cannot_waive_current_corruption(tmp_path):
    ob, clock, _ = active(tmp_path)
    record = archive_day(ob, clock, ANCHOR_DAY)
    raw = ob.dir / record["body"]["fetch"]["attempts"][0]["raw_path"]
    original = raw.read_bytes()
    raw.write_bytes(b"corrupted")
    clock.set(DAY1, "10:00:00")
    before = len(ob.records())
    with pytest.raises(obs.BindingMismatch, match="ledger_integrity"):
        ob.amend_resume(DAY1, "owner approved")
    assert len(ob.records()) == before
    raw.write_bytes(original)  # exact original bytes, not a re-fetch or a rewritten record
    ob.amend_resume(DAY1, "owner approved after exact evidence restoration")
    assert ob.verify_chain() == []


def test_readiness_runbook_attestation_is_accepted_by_activation(tmp_path):
    import re

    ob, clock, _ = make_observer(tmp_path, start=(ACTIVATION, "10:00:00"))
    commit = "a" * 40
    evidence = {
        "status": "preparation_verified_not_activated",
        "verification": {
            "code_commit": commit,
            "full_suite": {"exit_code": 0},
            "review_disposition": "important_findings_addressed_by_parent",
        },
        "source_delay_seconds": 900,
        "calendar_snapshot_sha256": ob.manifest["calendar"]["official_snapshot_sha256"],
        "final_code_sha256": ob.manifest["bindings"]["code"]["sha256"],
        "checks": dict.fromkeys(
            ("sip_access", "source_delay", "calendar_source", "synthetic_rehearsal"), True
        ),
    }
    path = tmp_path / "synthetic-preflight.json"
    path.write_text(json.dumps(evidence))
    runbook = (ROOT / "deploy/intraday_observer/OPERATOR_RUNBOOK.md").read_text()
    generators = re.findall(r"<<'PY'\n(.*?)\nPY", runbook, re.S)
    code = next(block for block in generators if "manifest_sha256" in block)
    result = subprocess.run(
        [sys.executable, "-c", code, str(ob.dir), str(path), commit],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    readiness = ob.dir / "readiness.json"
    bound = json.loads(readiness.read_text())
    assert bound["preflight_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bound["release_commit"] == commit
    assert (ob.dir / "preflight.json").read_bytes() == path.read_bytes()
    ob.activate("synthetic test approval", readiness_file=readiness)
    assert ob.has_record(None, "activation")
