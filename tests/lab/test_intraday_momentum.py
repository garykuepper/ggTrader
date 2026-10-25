"""Synthetic tests for r8 signal-blind validation and single-shot scoring; no market data."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from ggTrader.lab import intraday_momentum as core
from ggTrader.lab.intraday_probe import atomic_checkpoint, utc_bounds

# Distinctive synthetic prices so leakage into metadata artifacts is detectable.
ANCHOR_CLOSE = 100.11
HALF_CLOSE = 101.13  # 12:59 bar on the 13:00 early close
HALF_EXTENDED = 999.99  # a post-close 15:59 bar on the half-day that must never be used
DEC2 = {"09:59": 102.17, "15:30": 103.19, "15:59": 104.23}
DEC3 = {"09:59": 104.23, "15:30": 105.29, "15:59": 106.31}  # r_first exactly zero -> flat


def utc(day: str, clock: str) -> str:
    return utc_bounds(day, clock, clock)[0]


def bar(day: str, clock: str, price: object, **extra: object) -> dict:
    return {"t": utc(day, clock), "o": price, "c": price, "v": 100, **extra}


def thanksgiving_bars() -> list[dict]:
    return [
        bar("2024-11-27", "15:59", ANCHOR_CLOSE),
        bar("2024-11-29", "12:59", HALF_CLOSE),
        bar("2024-11-29", "15:59", HALF_EXTENDED),
        *[bar("2024-12-02", clock, price) for clock, price in DEC2.items()],
        *[bar("2024-12-03", clock, price) for clock, price in DEC3.items()],
    ]


CALENDAR = [
    {"date": "2024-11-27", "open": "09:30", "close": "16:00"},
    {"date": "2024-11-29", "open": "09:30", "close": "13:00"},
    {"date": "2024-12-02", "open": "09:30", "close": "16:00"},
    {"date": "2024-12-03", "open": "09:30", "close": "16:00"},
]
CLOSES = {"2024-11-27": ANCHOR_CLOSE, "2024-11-29": HALF_CLOSE, "2024-12-02": 104.23}
DAILY = {**CLOSES, "2024-12-03": 106.35}
DIVIDEND = {"group": "cash_dividends", "symbol": "SPY", "ex_date": "2024-12-02", "rate": 1.5}


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "ANCHOR", "2024-11-27")
    monkeypatch.setattr(core, "LAST", "2024-12-03")


def daily_rows(closes: dict) -> list[dict]:
    return [{"t": utc(day, "00:00"), "o": c, "c": c, "v": 1} for day, c in closes.items()]


def write_collection(
    directory: Path,
    *,
    calendar: list | None = None,
    bars: list | None = None,
    daily: dict | None = None,
    actions: list | None = None,
    skip: tuple = (),
) -> Path:
    """Write a complete synthetic collection in the exact stored schema."""
    manifest = core.build_manifest()
    (directory / core.CHUNK_DIR).mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(directory / core.MANIFEST_FILE, manifest, initial=True)
    bars = thanksgiving_bars() if bars is None else bars
    payloads = {
        "calendar": CALENDAR if calendar is None else calendar,
        "daily_bars": daily_rows(DAILY if daily is None else daily),
        "corporate_actions": [DIVIDEND] if actions is None else actions,
    }
    hashes = {}
    for spec in manifest["requests"]:
        if spec["id"] in skip:
            continue
        if spec["kind"] == "minute_bars":
            month = spec["id"].split(":")[1]
            records = [row for row in bars if utc_month(row["t"]) == month]
        else:
            records = payloads[spec["kind"]]
        chunk = {
            "schema": core.SCHEMAS["chunk"],
            "version": 1,
            "id": spec["id"],
            "kind": spec["kind"],
            "endpoint": spec["endpoint"],
            "parameters": spec["parameters"],
            "retrieved_utc": "2026-11-01T00:00:00+00:00",
            "complete": True,
            "pages": [{"page": 1, "body_sha256": "ab" * 32, "record_count": len(records)}],
            "records": records,
        }
        path = core.chunk_path(directory, spec["id"])
        atomic_checkpoint(path, chunk, initial=True)
        hashes[spec["id"]] = core.file_sha256(path)
    progress = {"manifest_sha256": "", "chunks": hashes, "runs": []}
    atomic_checkpoint(directory / core.PROGRESS_FILE, progress, initial=True)
    return directory


def utc_month(stamp: str) -> str:
    # Synthetic bars are all mid-month regular-hours minutes, so the UTC month is the NY month.
    return stamp[:7]


def validate(directory: Path) -> tuple[dict | None, dict]:
    return core.extract(core.load_dataset(directory))


# --- pure calculations -----------------------------------------------------------------------


def test_strategy_returns_hand_computed_with_equal_and_negative_signals() -> None:
    result = core.strategy_returns(
        [100, 100, 100], [101, 100, 99], [200, 200, 200], [202, 210, 210], core.BASE_COST
    )
    assert result["side"].tolist() == [1.0, 0.0, 0.0]  # r_first == 0 stays flat
    assert result["s"][0] == pytest.approx(0.01 - 0.0002)
    assert result["s"][1:].tolist() == [0.0, 0.0]  # no cost and no return on flat days
    stress = core.strategy_returns([100], [101], [200], [202], core.STRESS_COST)
    assert stress["s"][0] == pytest.approx(0.01 - 0.0004)


def test_annual_tax_compounds_after_debit_and_taxes_terminal_partial_year() -> None:
    ledger = core.tax_ledger(["2016-03-01", "2016-12-30", "2017-01-03"], [0.10, -0.05, 0.02])
    # 2016: W 1 -> 1.1 -> 1.045, P&L 0.045, tax 0.0108 -> 1.0342 (sizing base for 2017).
    # 2017 (terminal partial): P&L 1.0342*0.02 = 0.020684, W 1.054884, tax 0.00496416.
    assert [row["tax_debit"] for row in ledger["annual"]] == pytest.approx([0.0108, 0.00496416])
    assert ledger["wealth_end"] == pytest.approx(1.04991984)
    daily_tax = np.prod([1 + v * (0.76 if v > 0 else 1) for v in (0.10, -0.05, 0.02)])
    assert ledger["wealth_end"] != pytest.approx(daily_tax)


def test_loss_year_gets_no_benefit_or_carryforward() -> None:
    ledger = core.tax_ledger(["2016-06-01", "2017-06-01"], [-0.10, 0.20])
    assert ledger["annual"][0]["tax_debit"] == 0.0
    assert ledger["annual"][1]["tax_debit"] == pytest.approx(0.24 * 0.9 * 0.20)
    assert ledger["wealth_end"] == pytest.approx(0.9 * 1.2 - 0.24 * 0.18)


def test_cagr_uses_registered_calendar_span() -> None:
    assert core.registered_years() == pytest.approx(3922 / 365.25)
    assert core.cagr(2.0) == pytest.approx(2.0 ** (365.25 / 3922) - 1)
    assert core.cagr(0.0) is None


def test_nw_t_matches_bartlett_reference_with_small_sample_correction() -> None:
    x = np.array([0.01, -0.02, 0.0, 0.03, 0.0, 0.015, -0.005, 0.02, 0.0, 0.01, -0.01, 0.025])
    n, e, lags = len(x), x - x.mean(), 5
    s = e @ e + 2 * sum((1 - k / (lags + 1)) * (e[k:] @ e[:-k]) for k in range(1, lags + 1))
    reference = x.mean() / np.sqrt(s / n**2 * n / (n - 1))
    assert core.nw_t(x) == pytest.approx(reference, rel=1e-12)
    assert core.nw_t(x, 0) == pytest.approx(x.mean() / np.sqrt(e @ e / n**2 * n / (n - 1)))


@pytest.mark.parametrize("values", [np.zeros(50), np.full(50, 0.001), [0.1, np.nan, 0.2]])
def test_zero_or_nonfinite_variance_is_unresolved(values: object) -> None:
    assert core.nw_t(values) is None


def business_days(n: int) -> list[str]:
    days, day = [], date(2016, 1, 5)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days


def synthetic_inputs(gross_bp: float, n: int = 2000) -> dict:
    dates = business_days(n)
    noise = np.where(np.arange(n) % 2, 1e-4, -1e-4)
    p1559 = 100 * (1 + gross_bp * 1e-4 + noise)
    closes = {d: float(c) for d, c in zip(dates, p1559, strict=True)}
    return {
        "dates": dates,
        "prior_close": [100.0] * n,
        "p0959": [101.0] * n,
        "p1530": [100.0] * n,
        "p1559": p1559.tolist(),
        "closes": closes,
        "half_days": [],
        "daily_closes": closes,
        "dividends": {dates[10]: 1.0},
    }


def test_verdict_uses_base_cost_and_stress_cannot_change_it() -> None:
    result = core.score_inputs(synthetic_inputs(4.6))
    assert result["a1"]["status"] == "PASS" and result["a2"]["status"] == "PASS"
    assert result["diagnostics"]["stress"]["a2_pass"] is False  # stress fails A2 ...
    assert result["verdict"] == "GO"  # ... but cannot fail a base pass
    json.dumps(core._json_safe(result), allow_nan=False)
    weak = core.score_inputs(synthetic_inputs(3.5))
    assert weak["a1"]["status"] == "PASS" and weak["a2"]["status"] == "FAIL"
    assert weak["verdict"] == "NO-GO"


def test_all_flat_series_is_statistically_unresolved_not_infinite() -> None:
    inputs = synthetic_inputs(4.6, n=50)
    inputs["p0959"] = [99.0] * 50
    result = core.score_inputs(inputs)
    assert result["a1"]["status"] == "UNRESOLVED"
    assert result["verdict"] == "UNRESOLVED_STATISTICAL"
    assert result["diagnostics"]["signal_counts"]["negative_flat"] == 50


def test_failed_mandatory_diagnostic_cannot_be_reported_as_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "benchmark", lambda *a: 1 / 0)
    result = core.score_inputs(synthetic_inputs(4.6))
    assert result["diagnostic_errors"] == ["spy_displacement:ZeroDivisionError"]
    assert result["verdict"] == "UNRESOLVED_DIAGNOSTICS"


def test_benchmark_reinvests_distributions_on_aligned_closes() -> None:
    result = core.benchmark({"a": 100.0, "b": 110.0, "c": 121.0}, {"b": 1.0})
    assert result["wealth_end"] == pytest.approx(121 / 100 * (1 + 1 / 110))


# --- signal-blind validation -------------------------------------------------------------------


def test_complete_synthetic_dataset_passes_and_uses_half_day_1259_anchor(
    window: None, tmp_path: Path
) -> None:
    inputs, report = validate(write_collection(tmp_path))
    assert report["status"] == "PASS", report["defects"]
    assert inputs["dates"] == ["2024-12-02", "2024-12-03"]
    assert inputs["prior_close"] == [HALF_CLOSE, DEC2["15:59"]]  # 12:59 bar, never 15:59/999.99
    assert inputs["closes"]["2024-11-29"] == HALF_CLOSE
    assert inputs["dividends"] == {"2024-12-02": 1.5}
    assert report["counts"]["half_day_sessions"] == 1


def test_validator_is_signal_blind_and_emits_no_prices(
    window: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("validator computed a signal")

    for name in ("strategy_returns", "score_inputs", "nw_t", "tax_ledger"):
        monkeypatch.setattr(core, name, forbidden)
    report = core.validate_dataset(core.load_dataset(write_collection(tmp_path)))
    assert report["status"] == "PASS"
    text = json.dumps(report)
    for price in (ANCHOR_CLOSE, HALF_CLOSE, HALF_EXTENDED, *DEC2.values(), *DEC3.values()):
        assert str(price) not in text


@pytest.mark.parametrize(
    ("day", "clock", "defect"),
    [
        ("2024-11-27", "15:59", "endpoint_missing:close:2024-11-27"),  # fixed anchor
        ("2024-11-29", "12:59", "endpoint_missing:close:2024-11-29"),  # half-day anchor
        ("2024-12-02", "09:59", "endpoint_missing:signal_0959:2024-12-02"),
        ("2024-12-03", "15:30", "endpoint_missing:entry_1530:2024-12-03"),  # a flat day
        ("2024-12-03", "15:59", "endpoint_missing:close:2024-12-03"),
    ],
)
def test_any_missing_endpoint_is_unresolved_regardless_of_signal(
    window: None, tmp_path: Path, day: str, clock: str, defect: str
) -> None:
    rows = [row for row in thanksgiving_bars() if row["t"] != utc(day, clock)]
    inputs, report = validate(write_collection(tmp_path, bars=rows))
    assert inputs is None
    assert report["label"] == "UNRESOLVED (data)"
    assert defect in report["defects"]


@pytest.mark.parametrize("price", [0, -1.0, None, True, "104.23"])
def test_invalid_endpoint_price_is_unresolved(window: None, tmp_path: Path, price: object) -> None:
    rows = [
        {**row, "o": price} if row["t"] == utc("2024-12-02", "15:30") else row
        for row in thanksgiving_bars()
    ]
    _, report = validate(write_collection(tmp_path, bars=rows))
    assert report["defects"] == ["endpoint_invalid_price:entry_1530:2024-12-02"]


def test_duplicate_and_misaligned_endpoints_are_unresolved(window: None, tmp_path: Path) -> None:
    rows = thanksgiving_bars()
    rows.append(bar("2024-12-02", "09:59", 102.17))
    rows.append({**bar("2024-12-03", "15:30", 1.0), "t": utc("2024-12-03", "15:30:30")})
    _, report = validate(write_collection(tmp_path, bars=rows))
    assert "endpoint_duplicate:signal_0959:2024-12-02" in report["defects"]
    assert "endpoint_misaligned:entry_1530:2024-12-03" in report["defects"]


def test_dst_transition_requires_new_york_minutes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(core, "ANCHOR", "2024-03-07")
    monkeypatch.setattr(core, "LAST", "2024-03-11")
    calendar = [
        {"date": day, "open": "09:30", "close": "16:00"}
        for day in ("2024-03-07", "2024-03-08", "2024-03-11")
    ]
    prices = {"09:59": 50.5, "15:30": 51.0, "15:59": 51.5}
    rows = [bar("2024-03-07", "15:59", 50.0)]
    rows += [bar(day, c, p) for day in ("2024-03-08", "2024-03-11") for c, p in prices.items()]
    assert utc("2024-03-08", "09:59") == "2024-03-08T14:59:00Z"  # EST
    assert utc("2024-03-11", "09:59") == "2024-03-11T13:59:00Z"  # EDT
    daily = {"2024-03-07": 50.0, "2024-03-08": 51.5, "2024-03-11": 51.5}
    actions = [{**DIVIDEND, "ex_date": "2024-03-08"}]
    good = write_collection(
        tmp_path / "good", calendar=calendar, bars=rows, daily=daily, actions=actions
    )
    assert validate(good)[1]["status"] == "PASS"
    # An EST-assumed stamp on the first EDT session is the wrong minute, not the signal bar.
    wrong = [
        {**row, "t": "2024-03-11T14:59:00Z"} if row["t"] == "2024-03-11T13:59:00Z" else row
        for row in rows
    ]
    bad = write_collection(
        tmp_path / "bad", calendar=calendar, bars=wrong, daily=daily, actions=actions
    )
    assert "endpoint_missing:signal_0959:2024-03-11" in validate(bad)[1]["defects"]


@pytest.mark.parametrize(
    ("calendar", "defect"),
    [
        (CALENDAR + [CALENDAR[-1]], "calendar_malformed:not_strictly_increasing"),
        (CALENDAR[1:], "calendar_bounds:anchor_missing"),
        (
            [CALENDAR[0], {"date": "2024-11-30", "open": "09:30", "close": "16:00"}, *CALENDAR[1:]],
            "calendar_weekend:2024-11-30",
        ),
        (
            [*CALENDAR[:3], {**CALENDAR[3], "close": "14:00"}],
            "calendar_unexpected_hours:2024-12-03",
        ),
        ([CALENDAR[0], {"date": "2024-13-01"}, *CALENDAR[1:]], "calendar_malformed:row_1"),
        ([], "calendar_malformed:empty_or_not_list"),
    ],
)
def test_malformed_calendar_is_unresolved(
    window: None, tmp_path: Path, calendar: list, defect: str
) -> None:
    _, report = validate(write_collection(tmp_path, calendar=calendar))
    assert report["status"] == "UNRESOLVED"
    assert defect in report["defects"]


def test_unflagged_half_day_and_non_session_trading_are_calendar_stops(
    window: None, tmp_path: Path
) -> None:
    rows = [
        row
        for row in thanksgiving_bars()
        if row["t"] not in (utc("2024-12-03", "15:30"), utc("2024-12-03", "15:59"))
    ]
    rows.append(bar("2024-11-28", "10:00", 1.0))  # Thanksgiving is not a session
    _, report = validate(write_collection(tmp_path, bars=rows))
    assert "unflagged_half_day:2024-12-03" in report["defects"]
    assert "calendar_mismatch:regular_bars_on_non_session:2024-11-28" in report["defects"]


def test_interior_gaps_are_diagnostic_only(window: None, tmp_path: Path) -> None:
    _, report = validate(write_collection(tmp_path))
    assert report["status"] == "PASS"
    assert report["diagnostics"]["interior_missing_minutes_total"] == 3 * 390 + 210 - 8


@pytest.mark.parametrize(
    ("actions", "defect"),
    [
        (
            [DIVIDEND, {"group": "forward_splits", "symbol": "SPY", "ex_date": "2024-12-03"}],
            "corporate_action_stop:forward_splits:2024-12-03",
        ),
        ([], "dividend_coverage_missing:2024Q4"),  # empty is not verified absence
        ([{**DIVIDEND, "ex_date": "2024-11-28"}], "dividend_ex_date_not_session:2024-11-28"),
        ([{**DIVIDEND, "rate": 0}], "dividend_invalid_rate:2024-12-02"),
    ],
)
def test_corporate_action_coverage_and_split_stops(
    window: None, tmp_path: Path, actions: list, defect: str
) -> None:
    _, report = validate(write_collection(tmp_path, actions=actions))
    assert defect in report["defects"]


def test_missing_raw_daily_close_is_unresolved(window: None, tmp_path: Path) -> None:
    daily = {k: v for k, v in DAILY.items() if k != "2024-11-29"}
    _, report = validate(write_collection(tmp_path, daily=daily))
    assert report["defects"] == ["daily_bar_missing:2024-11-29"]


def test_missing_or_tampered_chunk_is_unresolved(window: None, tmp_path: Path) -> None:
    _, report = validate(write_collection(tmp_path / "a", skip=("daily_bars",)))
    assert "provenance_incomplete:daily_bars" in report["defects"]
    directory = write_collection(tmp_path / "b")
    path = core.chunk_path(directory, "minute_bars:2024-12")
    path.write_text(path.read_text().replace("103.19", "103.2"))
    _, report = validate(directory)
    assert report["defects"] == ["chunk_hash_mismatch:minute_bars:2024-12"]


def test_non_frozen_manifest_is_unresolved(window: None, tmp_path: Path) -> None:
    directory = write_collection(tmp_path)
    manifest = json.loads((directory / core.MANIFEST_FILE).read_text())
    manifest["requests"] = manifest["requests"][:-1]
    (directory / core.MANIFEST_FILE).write_text(json.dumps(manifest))
    _, report = validate(directory)
    assert "provenance:manifest_not_frozen_protocol_manifest" in report["defects"]


# --- single-shot scoring -----------------------------------------------------------------------


def validated(tmp_path: Path, **kwargs: object) -> tuple[Path, Path]:
    directory = write_collection(tmp_path / "data", **kwargs)
    validation = tmp_path / "validation.json"
    core.run_validation(directory, validation)
    return directory, validation


def test_score_once_on_matching_pass_with_hand_checked_diagnostics(
    window: None, tmp_path: Path
) -> None:
    directory, validation = validated(tmp_path)
    output = tmp_path / "result.json"
    result = core.run_score(directory, validation, output)
    assert json.loads(output.read_text())["verdict"] == result["verdict"]
    assert output.with_suffix(".md").read_text().startswith("# Intraday r8")
    diag = result["diagnostics"]
    r_last = DEC2["15:59"] / DEC2["15:30"] - 1
    assert diag["signal_counts"] == {
        "positive_traded": 1,
        "zero_flat": 1,
        "negative_flat": 0,
        "flat_sessions": 1,
    }
    assert diag["break_even_one_way_cost_bp"] == pytest.approx(r_last / 2 * 1e4)
    assert result["a2"]["annual_tax_ledger"][0]["net_pnl"] == pytest.approx(r_last - 2e-4)
    assert diag["exdiv_flat"]["mean_bp"] == 0.0  # the only trade was on the ex-date
    spy = (106.31 / ANCHOR_CLOSE) * (1 + 1.5 / 104.23)
    assert diag["spy_displacement"]["wealth_end"] == pytest.approx(spy)
    assert diag["close_convention_comparison"]["max_abs_bp"] == pytest.approx(
        abs(106.31 - 106.35) / 106.35 * 1e4
    )
    assert result["identities"] == json.loads(validation.read_text())["identities"]
    with pytest.raises(FileExistsError):
        core.run_score(directory, validation, output)


def test_claim_left_by_a_crashed_score_still_refuses(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    output = tmp_path / "result.json"
    output.with_suffix(".claim").touch()
    with pytest.raises(FileExistsError):
        core.run_score(directory, validation, output)


def test_tampering_after_pass_refuses_to_score(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    path = core.chunk_path(directory, "minute_bars:2024-12")
    path.write_text(path.read_text().replace("103.19", "103.2"))
    progress = json.loads((directory / core.PROGRESS_FILE).read_text())
    progress["chunks"]["minute_bars:2024-12"] = core.file_sha256(path)  # consistent re-record
    (directory / core.PROGRESS_FILE).write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="validation_identity_mismatch"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (tmp_path / "result.claim").exists()


def test_forged_pass_or_changed_code_or_protocol_refuses(
    window: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, validation = validated(tmp_path, skip=("daily_bars",))
    with pytest.raises(ValueError, match="validation_not_pass"):
        core.run_score(directory, validation, tmp_path / "r.json")
    forged = json.loads(validation.read_text()) | {"status": "PASS", "defects": []}
    validation.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="validation_identity_mismatch"):
        core.run_score(directory, validation, tmp_path / "r.json")

    good_dir, good = validated(tmp_path / "good")
    monkeypatch.setattr(core, "code_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="validation_identity_mismatch"):
        core.run_score(good_dir, good, tmp_path / "r.json")
    monkeypatch.setattr(core, "PROTOCOL_SHA256", "1" * 64)
    with pytest.raises(ValueError, match="protocol_hash_mismatch"):
        core.run_score(good_dir, good, tmp_path / "r.json")


def test_validation_artifact_is_write_once(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    with pytest.raises(FileExistsError):
        core.run_validation(directory, validation)


def test_manifest_is_frozen_raw_sip_and_dst_explicit() -> None:
    manifest = core.build_manifest()
    assert manifest["protocol_sha256"] == core.PROTOCOL_SHA256
    by_id = {spec["id"]: spec for spec in manifest["requests"]}
    assert len(by_id) == 3 + 129
    january, july = by_id["minute_bars:2016-01"], by_id["minute_bars:2016-07"]
    assert january["parameters"]["start"] == "2016-01-01T05:00:00Z"
    assert january["parameters"]["end"] == "2016-02-01T04:59:59Z"
    assert july["parameters"]["start"] == "2016-07-01T04:00:00Z"
    assert by_id["minute_bars:2026-09"]["parameters"]["end"] == "2026-10-01T03:59:59Z"
    for spec in manifest["requests"]:
        if "bars" in spec["kind"]:
            assert spec["parameters"]["adjustment"] == "raw"
            assert spec["parameters"]["feed"] == "sip"
    assert by_id["calendar"]["parameters"] == {"start": "2016-01-04", "end": "2026-09-30"}
    assert by_id["corporate_actions"]["parameters"]["start"] == "2015-12-04"
    assert "types" not in by_id["corporate_actions"]["parameters"]  # every action type
