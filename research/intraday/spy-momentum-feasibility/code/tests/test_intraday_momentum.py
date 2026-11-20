"""Synthetic tests for r9 signal-blind validation and single-shot scoring; no market data."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import intraday_momentum as core
import numpy as np
import pytest
from intraday_probe import atomic_checkpoint, utc_bounds

from ggTrader.utils.study import load_driver

pytestmark = pytest.mark.study

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


DEFAULT_REVIEW_PATH = core.VENDOR_REVIEW_PATH  # the registered r9 path, before any patching
R8_REVIEW = core.STUDY_DIR / "_intraday_r8_vendor_adjustment_review_2026-10-25.json"
REAL_ISSUER = core.ISSUER_WORKBOOK_PATH
REAL_ISSUER_SHA256 = core.ISSUER_WORKBOOK_SHA256
ISSUER_REFERENCE = core.PROJECT_ROOT / (
    "research/intraday/artifacts-2026-10-25-intraday-gap/issuer_spy_distributions.json"
)


def write_review(path: Path, **overrides: object) -> Path:
    """A synthetic reviewed vendor-adjustment record bound to the current (patched) window."""
    record = {**core.vendor_review_binding(), "status": "reviewed", "known_unresolved_changes": []}
    path.write_text(json.dumps({**record, **overrides}))
    return path


@pytest.fixture(autouse=True)
def review(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("review") / "review.json"
    monkeypatch.setattr(core, "VENDOR_REVIEW_PATH", path)
    return write_review(path)


@pytest.fixture(autouse=True)
def issuer(issuer_tools: SimpleNamespace) -> SimpleNamespace:
    """Default synthetic issuer schedule: one quarterly ex-date, matching DIVIDEND, on 12-02."""
    issuer_tools.install([issuer_tools.row("12/02/2024", "1.500000")])
    return issuer_tools


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch, review: Path) -> None:
    monkeypatch.setattr(core, "ANCHOR", "2024-11-27")
    monkeypatch.setattr(core, "LAST", "2024-12-03")
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", ())  # R1 sessions are not in this window
    write_review(review)


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


# OLS on an exact constant leaves float residual noise, which gave t ~ 1e15 for these.
# One property, six cases that between them use every length, level and lag of the former
# 6 x 5 x 4 = 120-case grid (each case was under 5 ms and re-proved the same guard).
@pytest.mark.parametrize(
    ("n", "level", "lags"),
    [
        (2, 0.0, 0),
        (3, 0.001, 5),
        (8, -0.0002, 10),
        (20, 1e-9, 20),
        (50, 0.1, 0),
        (2679, 0.001, 20),
    ],
)
def test_exact_constant_series_is_unresolved_at_any_length_and_level(
    n: int, level: float, lags: int
) -> None:
    assert core.nw_t(np.full(n, level), lags) is None


def test_valid_low_variance_series_is_not_erased_by_a_tolerance() -> None:
    values = np.full(8, 0.001)
    values[-1] += 1e-12
    t = core.nw_t(values)
    assert t is not None and np.isfinite(t)


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
        "distributions": {dates[10]: 1.0},
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


def test_constant_positive_series_is_unresolved_not_a1_pass() -> None:
    inputs = synthetic_inputs(4.6, n=200)
    inputs["p1559"] = [100.046] * 200  # every session trades the identical net return
    result = core.score_inputs(inputs)
    assert result["a1"]["status"] == "UNRESOLVED" and result["a1"]["nw5_t"] is None
    assert result["verdict"] == "UNRESOLVED_STATISTICAL"
    assert all(t is None for t in result["diagnostics"]["nw_lags"].values())


def test_beta1_t_is_none_when_regression_residuals_are_exactly_zero() -> None:
    inputs = synthetic_inputs(4.6, n=200)
    traded = np.arange(200) % 2 == 0
    inputs["p0959"] = np.where(traded, 101.0, 99.0).tolist()
    inputs["p1559"] = np.where(traded, 100.05, 100.01).tolist()  # r_last exact in side
    attribution = core.score_inputs(inputs)["diagnostics"]["attribution"]
    assert attribution["beta1"] == pytest.approx(0.0004)
    assert attribution["beta1_nw5_t"] is None


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


def test_benchmark_underflow_cannot_report_nonpositive_wealth() -> None:
    with pytest.raises(ValueError, match="invalid_benchmark_wealth_or_cagr"):
        core.benchmark({"a": 1e308, "b": 1e-308}, {})


def full_window_collection(
    tmp_path: Path, issuer: SimpleNamespace, *, rate: str = "10000000000.000000"
) -> tuple[Path, int]:
    """A complete r9 collection over the registered window: the real calendar, both R1 sessions
    missing exactly their registered bars, 43 issuer ex-dates. Returns (directory, scored)."""
    calendar = core.offline_schedule(core.ANCHOR, core.LAST)
    rows, closes, quarters = [], {}, {}
    scored = 0
    for session in calendar:
        day = session["date"]
        close = 100.0
        clock = "12:59" if session["close"] == "13:00" else "15:59"
        if day != core.ANCHOR and session["close"] == "16:00":
            close *= 1 + (3.6 if scored % 2 else 5.6) * 1e-4
            rows.append(bar(day, "09:59", 101.0))
            if day == "2019-08-12":  # the registered absences, with interior bars as in the data
                rows.append(bar(day, "14:00", 100.0))
            else:
                rows.append(bar(day, "15:30", 100.0))
            scored += 1
        if day != "2019-08-12":
            rows.append(bar(day, clock, close))
        closes[day] = close
        if day != core.ANCHOR:
            quarters.setdefault((day[:4], (int(day[5:7]) - 1) // 3), day)
    ex_dates = list(quarters.values())
    issuer.install(
        [issuer.row(f"{d[5:7]}/{d[8:]}/{d[:4]}", rate) for d in ex_dates],
    )
    actions = [
        {"group": "cash_dividends", "symbol": "SPY", "ex_date": d, "rate": float(rate)}
        for d in ex_dates
    ]
    directory = write_collection(
        tmp_path / "collection", calendar=calendar, bars=rows, daily=closes, actions=actions
    )
    return directory, scored


def test_validated_benchmark_overflow_is_a_named_failure_in_the_written_score(
    tmp_path: Path, issuer: SimpleNamespace
) -> None:
    directory, scored = full_window_collection(tmp_path, issuer)
    validation = tmp_path / "validation.json"
    report = core.run_validation(directory, validation)
    assert report["status"] == "PASS", report["defects"]
    # r8 scored 2,679 sessions; r9 removes the two registered ones entirely -> 2,677.
    assert scored == 2679 and report["counts"]["scored_sessions"] == 2677
    assert report["counts"]["excluded_sessions"] == 2
    assert report["counts"]["issuer_ex_dates_in_window"] == 43
    output = tmp_path / "score.json"
    core.run_score(directory, validation, output)
    result = json.loads(output.read_text())
    assert result["a1"]["sessions"] == 2677
    assert result["a1"]["status"] == result["a2"]["status"] == "PASS"
    assert result["verdict"] == "UNRESOLVED_DIAGNOSTICS"
    assert result["diagnostic_errors"] == ["spy_displacement:ValueError"]
    assert "spy_displacement" not in result["diagnostics"]
    assert result["exclusions"]["sessions"] == ["2019-08-12", "2019-08-13"]
    text = output.with_suffix(".md").read_text()
    assert "2019-08-12" in text and "2019-08-13" in text and "before any price" in text


def test_scorer_drops_both_excluded_sessions_instead_of_scoring_zeros(
    tmp_path: Path, issuer: SimpleNamespace
) -> None:
    directory, _ = full_window_collection(tmp_path, issuer, rate="1.500000")
    inputs, report = validate(directory)
    assert report["status"] == "PASS", report["defects"]
    assert len(inputs["dates"]) == 2677
    assert "2019-08-12" not in inputs["dates"] and "2019-08-13" not in inputs["dates"]
    assert "2019-08-12" not in inputs["closes"]  # that close does not exist
    assert "2019-08-13" in inputs["closes"]  # 2019-08-14's prior close, validated normally
    after = inputs["dates"].index("2019-08-14")
    assert inputs["dates"][after - 1] == "2019-08-09"  # removed, not left as flat days
    assert inputs["prior_close"][after] == inputs["closes"]["2019-08-13"]
    result = core.score_inputs(inputs)
    assert result["a1"]["sessions"] == 2677
    signal = result["diagnostics"]["signal_counts"]
    assert sum(signal[k] for k in ("positive_traded", "zero_flat", "negative_flat")) == 2677
    assert result["exclusions"]["scored_sessions_after_exclusion"] == 2677
    assert result["exclusions"]["tolerated_defects"] == [
        {"session": "2019-08-12", "endpoint": "close", "status": "absent"},
        {"session": "2019-08-12", "endpoint": "entry_1530", "status": "absent"},
        {"session": "2019-08-13", "endpoint": "prior_close", "status": "absent"},
    ]


def test_nonfinite_computed_diagnostic_is_not_silently_serialized_as_null() -> None:
    inputs = synthetic_inputs(4.6)
    inputs["daily_closes"] = {day: 1e-308 for day in inputs["closes"]}
    with np.errstate(over="ignore", invalid="ignore"):
        result = core.score_inputs(inputs)
    serialized = json.loads(json.dumps(core._json_safe(result), allow_nan=False))
    assert serialized["verdict"] == "UNRESOLVED_DIAGNOSTICS"
    assert serialized["diagnostic_errors"] == ["close_convention_comparison:ValueError"]
    assert "close_convention_comparison" not in serialized["diagnostics"]


def test_legitimately_undefined_diagnostics_preserve_otherwise_valid_results() -> None:
    inputs = synthetic_inputs(4.6)
    # A constant trade indicator makes beta undefined without invalidating the benchmark.
    result = core.score_inputs(inputs)
    assert result["verdict"] == "GO" and result["diagnostic_errors"] == []
    assert result["diagnostics"]["attribution"]["beta1"] is None
    assert result["diagnostics"]["attribution"]["beta1_nw5_t"] is None
    for period in result["diagnostics"]["subperiods"].values():
        if not period["sessions"]:
            assert period["mean_bp"] is period["nw5_t"] is period["hit_rate"] is None

    inputs["p0959"] = [99.0] * len(inputs["dates"])
    flat = core.score_inputs(inputs)
    assert flat["verdict"] == "UNRESOLVED_STATISTICAL" and flat["diagnostic_errors"] == []
    assert all(t is None for t in flat["diagnostics"]["nw_lags"].values())
    assert flat["diagnostics"]["conditional_on_trading_mean_bp"] is None


def test_finite_numpy_diagnostic_scalars_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    original = core._close_comparison

    def numpy_scalars(*args: object) -> dict:
        result = original(*args)
        result["sessions"] = np.int64(result["sessions"])
        result["mean_signed_bp"] = np.float32(result["mean_signed_bp"])
        return result

    monkeypatch.setattr(core, "_close_comparison", numpy_scalars)
    result = core.score_inputs(synthetic_inputs(4.6))
    assert result["verdict"] == "GO" and result["diagnostic_errors"] == []
    serialized = json.loads(json.dumps(core._json_safe(result), allow_nan=False))
    # The two R1 sessions fall inside the synthetic span and are left out of this diagnostic.
    assert serialized["diagnostics"]["close_convention_comparison"]["sessions"] == 1998


# --- signal-blind validation -------------------------------------------------------------------


def test_complete_synthetic_dataset_passes_and_uses_half_day_1259_anchor(
    window: None, tmp_path: Path
) -> None:
    inputs, report = validate(write_collection(tmp_path))
    assert report["status"] == "PASS", report["defects"]
    assert inputs["dates"] == ["2024-12-02", "2024-12-03"]
    assert inputs["prior_close"] == [HALF_CLOSE, DEC2["15:59"]]  # 12:59 bar, never 15:59/999.99
    assert inputs["closes"]["2024-11-29"] == HALF_CLOSE
    assert inputs["distributions"] == {"2024-12-02": 1.5}
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, review: Path, issuer: SimpleNamespace
) -> None:
    monkeypatch.setattr(core, "ANCHOR", "2024-03-07")
    monkeypatch.setattr(core, "LAST", "2024-03-11")
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", ())
    issuer.install([issuer.row("03/08/2024", "1.500000")])
    write_review(review)
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
    assert not any("2024-13-01" in d for d in report["defects"])  # never echoed


def test_dropping_an_internal_session_with_all_its_data_is_a_calendar_mismatch(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    # Remove 2024-12-02 from the official calendar *and* its bars and daily close; without the
    # offline schedule this is a shorter, self-consistent series with the wrong prior anchor.
    calendar = [row for row in CALENDAR if row["date"] != "2024-12-02"]
    bars = [row for row in thanksgiving_bars() if not row["t"].startswith("2024-12-02")]
    daily = {k: v for k, v in DAILY.items() if k != "2024-12-02"}
    actions = [{**DIVIDEND, "ex_date": "2024-12-03"}]
    issuer.install([issuer.row("12/03/2024", "1.500000")])
    inputs, report = validate(
        write_collection(tmp_path, calendar=calendar, bars=bars, daily=daily, actions=actions)
    )
    assert inputs is None
    assert report["defects"] == ["calendar_mismatch:offline_session_missing:2024-12-02"]


def test_extra_session_or_changed_hours_versus_offline_schedule_are_mismatches(
    window: None, tmp_path: Path
) -> None:
    thanksgiving = {"date": "2024-11-28", "open": "09:30", "close": "16:00"}
    calendar = [CALENDAR[0], thanksgiving, *CALENDAR[1:]]
    _, report = validate(write_collection(tmp_path / "a", calendar=calendar))
    assert "calendar_mismatch:session_not_in_offline_schedule:2024-11-28" in report["defects"]
    calendar = [CALENDAR[0], {**CALENDAR[1], "close": "16:00"}, *CALENDAR[2:]]
    _, report = validate(write_collection(tmp_path / "b", calendar=calendar))
    assert "calendar_mismatch:offline_hours:2024-11-29" in report["defects"]


def test_offline_schedule_is_compared_and_recorded(
    window: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas_market_calendars as mcal

    report = core.validate_dataset(core.load_dataset(write_collection(tmp_path / "a")))
    assert report["status"] == "PASS"
    assert report["offline_calendar"] == {
        "library": "pandas_market_calendars",
        "version": mcal.__version__,
        "calendar": "NYSE",
        "sessions": 4,
        "schedule_sha256": core.canonical_hash(CALENDAR),
    }
    assert core.runtime_versions()["pandas_market_calendars"] == mcal.__version__
    # A bounded mocked schedule that lacks the last session makes the official one unexpected.
    monkeypatch.setattr(core, "offline_schedule", lambda start, end: CALENDAR[:3])
    _, report = validate(write_collection(tmp_path / "b"))
    assert report["defects"] == ["calendar_mismatch:session_not_in_offline_schedule:2024-12-03"]

    def unavailable(start: str, end: str) -> list:
        raise RuntimeError("calendar library broken")

    monkeypatch.setattr(core, "offline_schedule", unavailable)
    _, report = validate(write_collection(tmp_path / "c"))
    assert report["defects"] == ["calendar_offline_schedule_unavailable"]


def test_offline_schedule_for_the_registered_window() -> None:
    schedule = core.offline_schedule(core.ANCHOR, core.LAST)
    assert (schedule[0]["date"], schedule[-1]["date"]) == (core.ANCHOR, core.LAST)
    assert len(schedule) == 2701
    assert sum(row["close"] == "13:00" for row in schedule) == 21
    assert {(row["open"], row["close"]) for row in schedule} == {
        core.FULL_HOURS,
        core.HALF_HOURS,
    }
    days = {row["date"] for row in schedule}
    assert "2018-12-05" not in days and "2025-01-09" not in days  # unscheduled closures


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


def test_in_window_spy_split_still_stops_the_study(window: None, tmp_path: Path) -> None:
    # R3: the fresh pull keeps Alpaca corporate actions for exactly this automatic stop.
    split = {"group": "forward_splits", "symbol": "SPY", "ex_date": "2024-12-03"}
    inputs, report = validate(write_collection(tmp_path, actions=[DIVIDEND, split]))
    assert inputs is None
    assert report["defects"] == ["corporate_action_stop:split:forward_splits:2024-12-03"]
    assert core.build_manifest()["requests"][2]["kind"] == "corporate_actions"


ALPACA_VALUE_PROBLEMS = {
    "bad_rate": ([{**DIVIDEND, "rate": 0}], ["dividend_invalid_rate:2024-12-02"]),
    "not_session": (
        [{**DIVIDEND, "ex_date": "2024-11-28"}],
        ["dividend_ex_date_not_session:2024-11-28"],
    ),
    "currency": ([{**DIVIDEND, "currency": "CAD"}], ["dividend_currency_not_usd:2024-12-02"]),
    "duplicate": ([DIVIDEND, DIVIDEND], ["dividend_duplicate_ex_date:2024-12-02"]),
}


@pytest.mark.parametrize("name", [*ALPACA_VALUE_PROBLEMS, "empty"])
def test_alpaca_dividend_gaps_and_value_problems_never_block(
    window: None, tmp_path: Path, name: str
) -> None:
    # r8 stopped on exactly these; r9 takes every distribution from the issuer schedule.
    actions, issues = ALPACA_VALUE_PROBLEMS.get(name, ([], []))
    inputs, report = validate(write_collection(tmp_path, actions=actions))
    assert report["status"] == "PASS", report["defects"]
    assert inputs["distributions"] == {"2024-12-02": 1.5}
    assert report["alpaca_dividend_cross_check"]["alpaca_record_issues"] == issues
    assert report["alpaca_dividend_cross_check"]["gating"] is False


# Stored records keep the official field names (see the Alpaca corporate-actions reference);
# values are invented. Real types carry aliases/effective_date, not a universal symbol/ex_date.
UNIT_SPLIT = {
    "group": "unit_splits",
    "old_symbol": "SPY",
    "new_symbol": "SPY",
    "alternate_symbol": "SPYW",
    "effective_date": "2024-12-03",
    "process_date": "2024-12-20",  # processed after the scored window
}
NAME_CHANGE = {
    "group": "name_changes",
    "old_symbol": "SPY",
    "new_symbol": "SPYX",
    "process_date": "2024-12-02",
}
CASH_MERGER = {
    "group": "cash_mergers",
    "acquiree_symbol": "SPY",
    "acquirer_symbol": "ACQ",
    "effective_date": "2024-12-03",
    "process_date": "2024-12-03",
}
SPLIT = {"group": "forward_splits", "symbol": "SPY", "ex_date": "2024-12-03"}


@pytest.mark.parametrize(
    ("actions", "defects"),
    [
        ([DIVIDEND, UNIT_SPLIT], ["corporate_action_stop:split:unit_splits:2024-12-03"]),
        # A harmless-looking name change must not hide a real split; neither passes silently.
        (
            [DIVIDEND, NAME_CHANGE, SPLIT],
            [
                "corporate_action_stop:split:forward_splits:2024-12-03",
                "corporate_action_unsupported_undated:name_changes:1",
            ],
        ),
        ([DIVIDEND, CASH_MERGER], ["corporate_action_unsupported:cash_mergers:2024-12-03"]),
        (
            [DIVIDEND, {k: v for k, v in SPLIT.items() if k != "ex_date"}],
            ["corporate_action_unsupported_undated:forward_splits:1"],
        ),
        ([{**DIVIDEND, "symbol": "QQQ"}], ["corporate_action_unrelated_symbol:0"]),
    ],
)
def test_real_action_types_use_aliases_and_effective_dates(
    window: None, tmp_path: Path, actions: list, defects: list
) -> None:
    inputs, report = validate(write_collection(tmp_path, actions=actions))
    assert inputs is None
    assert [d for d in report["defects"] if not d.startswith("dividend_coverage")] == defects


def test_split_effective_before_window_but_processed_inside_is_irrelevant(
    window: None, tmp_path: Path
) -> None:
    early = {**SPLIT, "ex_date": "2024-11-01", "process_date": "2024-12-02"}
    _, report = validate(write_collection(tmp_path, actions=[DIVIDEND, early]))
    assert report["status"] == "PASS", report["defects"]


def test_alpaca_dividend_without_ex_date_is_never_placed_by_process_date(
    window: None, tmp_path: Path
) -> None:
    undated = {
        "group": "cash_dividends",
        "symbol": "SPY",
        "rate": 1.5,
        "process_date": "2024-12-02",
    }
    _, report = validate(write_collection(tmp_path, actions=[undated]))
    assert report["status"] == "PASS", report["defects"]
    check = report["alpaca_dividend_cross_check"]
    assert check["alpaca_record_issues"] == ["dividend_missing_ex_date:0"]
    assert check["alpaca_ex_dates"] == 0 and check["missing_from_alpaca"] == ["2024-12-02"]


@pytest.mark.parametrize("currency", ["USD", "", "CAD"])
def test_alpaca_dividend_currency_is_cross_check_only(
    window: None, tmp_path: Path, currency: str
) -> None:
    inputs, report = validate(
        write_collection(tmp_path, actions=[{**DIVIDEND, "currency": currency}])
    )
    assert inputs is not None and report["status"] == "PASS"
    issues = report["alpaca_dividend_cross_check"]["alpaca_record_issues"]
    assert issues == (["dividend_currency_not_usd:2024-12-02"] if currency == "CAD" else [])


@pytest.mark.parametrize(
    "stored",
    [
        {**DIVIDEND, "group": "mystery_actions"},
        {**DIVIDEND, "rate": "1.5"},
        {**DIVIDEND, "cusip": "78462F103"},  # not in the persisted whitelist
        {**DIVIDEND, "ex_date": "2024-12-2"},
    ],
)
def test_stored_corporate_action_is_revalidated(window: None, tmp_path: Path, stored: dict) -> None:
    _, report = validate(write_collection(tmp_path, actions=[DIVIDEND, stored]))
    assert report["defects"] == ["corporate_action_malformed:1"]


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


# --- R1: the closed two-session exclusion list ---------------------------------------------------

# A five-session stand-in for 2019-08-12/13: 12-04 lacks its 15:30 and 15:59 bars, so 12-05
# lacks its prior close. Registered exactly like the real list, only on different dates.
R1_DAYS = ("2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05", "2024-12-06")
R1_CALENDAR = [{"date": day, "open": "09:30", "close": "16:00"} for day in R1_DAYS]
R1_DAILY = {day: 100.0 + i for i, day in enumerate(R1_DAYS)}
R1_EXCLUDED = (("2024-12-04", ("entry_1530", "close")), ("2024-12-05", ("prior_close",)))


def r1_bars() -> list[dict]:
    """Complete bars except exactly the registered absences (12-04 keeps an interior bar)."""
    rows = [bar("2024-12-02", "15:59", 100.0)]
    for day, o, c, close in (
        ("2024-12-03", 101.0, 102.0, 103.0),
        ("2024-12-05", 104.0, 105.0, 106.0),
        ("2024-12-06", 107.0, 108.0, 109.0),
    ):
        rows += [bar(day, "09:59", o), bar(day, "15:30", c), bar(day, "15:59", close)]
    rows += [bar("2024-12-04", "09:59", 103.5), bar("2024-12-04", "14:00", 103.6)]
    return rows


@pytest.fixture
def r1(window: None, monkeypatch: pytest.MonkeyPatch, review: Path, issuer: SimpleNamespace):
    monkeypatch.setattr(core, "ANCHOR", "2024-12-02")
    monkeypatch.setattr(core, "LAST", "2024-12-06")
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", R1_EXCLUDED)
    issuer.install([issuer.row("12/03/2024", "1.500000")])
    write_review(review)


def r1_collection(tmp_path: Path, bars: list[dict] | None = None, **kwargs: object) -> Path:
    return write_collection(
        tmp_path,
        calendar=R1_CALENDAR,
        bars=r1_bars() if bars is None else bars,
        daily=R1_DAILY,
        actions=[{**DIVIDEND, "ex_date": "2024-12-03"}],
        **kwargs,
    )


def test_registered_exclusion_list_is_exactly_the_two_sessions() -> None:
    assert core.EXCLUDED_SESSIONS == (
        ("2019-08-12", ("entry_1530", "close")),
        ("2019-08-13", ("prior_close",)),
    )
    assert "before any price" in core.EXCLUSION_DISCLOSURE
    assert "bias" in core.EXCLUSION_DISCLOSURE


def test_tolerated_exclusion_passes_and_is_recorded(r1: None, tmp_path: Path) -> None:
    inputs, report = validate(r1_collection(tmp_path))
    assert report["status"] == "PASS", report["defects"]
    # The two excluded sessions are removed from the series, not scored as flat days.
    assert inputs["dates"] == ["2024-12-03", "2024-12-06"]
    assert inputs["prior_close"] == [100.0, 106.0]  # 12-06 anchors on the validated 12-05 close
    assert "2024-12-04" not in inputs["closes"] and "2024-12-05" in inputs["closes"]
    assert report["counts"]["scored_sessions"] == 2 and report["counts"]["excluded_sessions"] == 2
    exclusions = report["exclusions"]
    assert exclusions["sessions"] == ["2024-12-04", "2024-12-05"]
    assert exclusions["tolerated_defects"] == [
        {"session": "2024-12-04", "endpoint": "close", "status": "absent"},
        {"session": "2024-12-04", "endpoint": "entry_1530", "status": "absent"},
        {"session": "2024-12-05", "endpoint": "prior_close", "status": "absent"},
    ]
    assert exclusions["scored_sessions_before_exclusion"] == 4
    assert exclusions["scored_sessions_after_exclusion"] == 2
    assert exclusions == inputs["exclusions"]


@pytest.mark.parametrize(
    ("day", "clock", "defect"),
    [
        ("2024-12-06", "09:59", "endpoint_missing:signal_0959:2024-12-06"),
        ("2024-12-06", "15:30", "endpoint_missing:entry_1530:2024-12-06"),
        ("2024-12-06", "15:59", "endpoint_missing:close:2024-12-06"),
        ("2024-12-03", "09:59", "endpoint_missing:signal_0959:2024-12-03"),
        ("2024-12-02", "15:59", "endpoint_missing:close:2024-12-02"),  # the anchor
    ],
)
def test_any_other_missing_endpoint_is_still_unresolved(
    r1: None, tmp_path: Path, day: str, clock: str, defect: str
) -> None:
    rows = [row for row in r1_bars() if row["t"] != utc(day, clock)]
    inputs, report = validate(r1_collection(tmp_path, rows))
    assert inputs is None and report["label"] == "UNRESOLVED (data)"
    assert defect in report["defects"]


@pytest.mark.parametrize(
    ("day", "clock", "defect"),
    [
        # Excluded sessions keep every other required endpoint (r9 R1).
        ("2024-12-04", "09:59", "endpoint_missing:signal_0959:2024-12-04"),
        ("2024-12-05", "09:59", "endpoint_missing:signal_0959:2024-12-05"),
        ("2024-12-05", "15:30", "endpoint_missing:entry_1530:2024-12-05"),
        ("2024-12-05", "15:59", "endpoint_missing:close:2024-12-05"),  # 12-06's prior close
    ],
)
def test_other_endpoints_of_an_excluded_session_must_still_validate(
    r1: None, tmp_path: Path, day: str, clock: str, defect: str
) -> None:
    rows = [row for row in r1_bars() if row["t"] != utc(day, clock)]
    inputs, report = validate(r1_collection(tmp_path, rows))
    assert inputs is None
    assert defect in report["defects"]


@pytest.mark.parametrize(
    ("extra", "defect"),
    [
        (bar("2024-12-04", "15:30", 1.0), "exclusion_basis_violated:entry_1530:2024-12-04"),
        (bar("2024-12-04", "15:59", 1.0), "exclusion_basis_violated:close:2024-12-04"),
        (
            {**bar("2024-12-04", "15:30", 1.0), "t": utc("2024-12-04", "15:30:30")},
            "exclusion_basis_violated:entry_1530:2024-12-04",  # misaligned still counts as present
        ),
    ],
)
def test_expected_missing_bars_present_ends_the_study(
    r1: None, tmp_path: Path, extra: dict, defect: str
) -> None:
    inputs, report = validate(r1_collection(tmp_path, [*r1_bars(), extra]))
    assert inputs is None and report["label"] == "UNRESOLVED (data)"
    assert defect in report["defects"]


def test_prior_close_listed_absent_but_present_means_the_list_basis_is_false(
    r1: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", (R1_EXCLUDED[1],))
    rows = [*r1_bars(), bar("2024-12-04", "15:30", 1.0), bar("2024-12-04", "15:59", 1.0)]
    _, report = validate(r1_collection(tmp_path, rows))
    assert report["defects"] == ["exclusion_basis_violated:prior_close:2024-12-05"]


def test_a_session_after_an_excluded_one_cannot_borrow_its_missing_close(
    window: None, monkeypatch: pytest.MonkeyPatch, r1: None, tmp_path: Path
) -> None:
    # Only 12-05 is registered to lack its prior close; if 12-06 did too, it is a plain defect.
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", (R1_EXCLUDED[0],))
    _, report = validate(r1_collection(tmp_path))
    assert "endpoint_missing:prior_close:2024-12-05" in report["defects"]


def test_exclusions_are_never_added_at_runtime(r1: None, tmp_path: Path) -> None:
    rows = [row for row in r1_bars() if row["t"] != utc("2024-12-06", "15:30")]
    inputs, report = validate(r1_collection(tmp_path, rows))
    assert inputs is None
    assert report["exclusions"]["sessions"] == ["2024-12-04", "2024-12-05"]  # still the registered


@pytest.mark.parametrize(
    "excluded",
    [
        (("2024-12-09", ("entry_1530", "close")),),  # not in the calendar
        (("2024-12-02", ("close",)),),  # the anchor is a session but is never scored
    ],
)
def test_excluded_session_must_be_a_scored_calendar_session(
    r1: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, excluded: tuple
) -> None:
    monkeypatch.setattr(core, "EXCLUDED_SESSIONS", excluded)
    inputs, report = validate(r1_collection(tmp_path))
    assert inputs is None
    assert report["status"] == "UNRESOLVED"


def test_full_pipeline_scores_without_the_excluded_sessions(r1: None, tmp_path: Path) -> None:
    directory, validation = validated_r1(tmp_path)
    result = core.run_score(directory, validation, tmp_path / "result.json")
    assert result["a1"]["sessions"] == 2
    assert result["exclusions"]["sessions"] == ["2024-12-04", "2024-12-05"]
    text = (tmp_path / "result.json").with_suffix(".md").read_text()
    assert "2024-12-04, 2024-12-05" in text and core.EXCLUSION_DISCLOSURE in text


def validated_r1(tmp_path: Path) -> tuple[Path, Path]:
    directory = r1_collection(tmp_path / "data")
    validation = tmp_path / "validation.json"
    core.run_validation(directory, validation)
    return directory, validation


def test_validation_artifact_states_the_exclusion(r1: None, tmp_path: Path) -> None:
    _, validation = validated_r1(tmp_path)
    saved = json.loads(validation.read_text())
    assert saved["exclusions"]["tolerated_defects"][0]["session"] == "2024-12-04"
    assert saved["ancillary_coverage"]["exclusions"] == core.EXCLUSION_DISCLOSURE


# --- r9 binds nothing from r8 --------------------------------------------------------------------


def test_every_schema_default_directory_and_output_name_is_r9() -> None:
    assert set(core.SCHEMAS.values()) == {
        f"intraday_momentum_r9_{name}"
        for name in ("manifest", "chunk", "dataset", "validation", "score", "score_claim")
    }
    run = load_driver(core.STUDY_DIR / "code/run.py")
    assert run.DEFAULT_COLLECTION.name == "intraday_momentum_r9"
    assert "r9" in run.DEFAULT_VALIDATION.name and "r9" in run.DEFAULT_RESULT.name
    assert "r8" not in run.DEFAULT_VALIDATION.name + run.DEFAULT_RESULT.name


def rewrite_json(path: Path, change: object) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def re_record(directory: Path, request_id: str) -> None:
    path = core.chunk_path(directory, request_id)
    progress = json.loads((directory / core.PROGRESS_FILE).read_text())
    progress["chunks"][request_id] = core.file_sha256(path)
    (directory / core.PROGRESS_FILE).write_text(json.dumps(progress))


def test_r8_chunks_and_manifest_are_rejected(window: None, tmp_path: Path) -> None:
    directory = write_collection(tmp_path)
    request = "minute_bars:2024-12"
    rewrite_json(
        core.chunk_path(directory, request),
        lambda chunk: chunk.update(schema="intraday_momentum_r8_chunk"),
    )
    re_record(directory, request)
    inputs, report = validate(directory)
    assert inputs is None and f"provenance_invalid:{request}" in report["defects"]

    other = write_collection(tmp_path / "m")
    rewrite_json(
        other / core.MANIFEST_FILE,
        lambda m: m.update(
            schema="intraday_momentum_r8_manifest",
            protocol_sha256="be395d950c1da7abf010acf90ffbe461db597672a21dd72b73c0294ba7b45b85",
        ),
    )
    _, report = validate(other)
    assert "provenance:manifest_not_frozen_protocol_manifest" in report["defects"]
    assert "identity:manifest_protocol_mismatch" in report["defects"]


def test_r8_validation_artifact_and_claim_cannot_score_r9(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    rewrite_json(validation, lambda v: v.update(schema="intraday_momentum_r8_validation"))
    with pytest.raises(ValueError, match="^validation_not_pass$"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


def test_r8_protocol_hash_in_the_validation_identities_is_refused(
    window: None, tmp_path: Path
) -> None:
    directory, validation = validated(tmp_path)
    r8 = "be395d950c1da7abf010acf90ffbe461db597672a21dd72b73c0294ba7b45b85"
    rewrite_json(validation, lambda v: v["identities"].update(protocol_sha256=r8))
    with pytest.raises(ValueError, match="^validation_identity_mismatch$"):
        core.run_score(directory, validation, tmp_path / "result.json")


def test_the_closed_r8_collection_directory_name_is_not_the_default() -> None:
    run = load_driver(core.STUDY_DIR / "code/run.py")
    assert not str(run.DEFAULT_COLLECTION).endswith("intraday_momentum_r8")


def test_the_issuer_workbook_is_bound_into_manifest_and_identities(
    window: None, tmp_path: Path
) -> None:
    digest = core.ISSUER_WORKBOOK_SHA256
    assert core.build_manifest()["issuer_workbook_sha256"] == digest
    _, validation = validated(tmp_path)
    assert json.loads(validation.read_text())["identities"]["issuer_workbook_sha256"] == digest


def test_issuer_workbook_swapped_after_pass_refuses_to_score(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    directory, validation = validated(tmp_path)
    issuer.install([issuer.row("12/02/2024", "1.500000"), issuer.row("12/03/2024", "2.0")])
    with pytest.raises(ValueError, match="^validation_identity_mismatch$"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


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
    assert output.with_suffix(".md").read_text().startswith("# Intraday r9")
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


def test_claim_is_bound_to_the_collection_not_the_output_name(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    result = core.run_score(directory, validation, tmp_path / "result.json")
    claim = json.loads((directory / core.SCORE_CLAIM_FILE).read_text())
    assert claim["identities"] == result["identities"]
    assert claim["validation_sha256"] == result["validation_sha256"]
    other = tmp_path / "elsewhere" / "other.json"
    with pytest.raises(FileExistsError, match="score_already_claimed"):
        core.run_score(directory, validation, other)
    assert not other.exists() and not other.with_suffix(".md").exists()


def test_claim_left_by_a_crashed_score_still_refuses(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    (directory / core.SCORE_CLAIM_FILE).touch()
    for name in ("result.json", "retry.json"):
        with pytest.raises(FileExistsError, match="score_already_claimed"):
            core.run_score(directory, validation, tmp_path / name)
        assert not (tmp_path / name).exists()


def test_existing_output_is_never_overwritten(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("kept")
    with pytest.raises(FileExistsError):
        core.run_score(directory, validation, output)
    assert output.read_text() == "kept"
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


def test_tampering_after_pass_refuses_to_score(window: None, tmp_path: Path) -> None:
    directory, validation = validated(tmp_path)
    path = core.chunk_path(directory, "minute_bars:2024-12")
    path.write_text(path.read_text().replace("103.19", "103.2"))
    progress = json.loads((directory / core.PROGRESS_FILE).read_text())
    progress["chunks"]["minute_bars:2024-12"] = core.file_sha256(path)  # consistent re-record
    (directory / core.PROGRESS_FILE).write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="validation_identity_mismatch"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


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


# --- second fix round: name changes, capital gains, currency, vendor review, runtime, chunks ----

SAME_NAME = {"group": "name_changes", "old_symbol": "SPY", "new_symbol": "SPY"}
# Invented legs in the official REST shape (long_term_rate/short_term_rate, at least one).
CAPITAL_GAIN = {
    "group": "capital_gains_distributions",
    "symbol": "SPY",
    "ex_date": "2024-12-02",
    "long_term_rate": 0.4,
    "short_term_rate": 0.1,
    "currency": "USD",
}


@pytest.mark.parametrize("process_date", ["2024-11-10", "2024-12-02"])  # margin, in window
def test_same_ticker_spy_name_change_is_counted_metadata_not_a_stop(
    window: None, tmp_path: Path, process_date: str
) -> None:
    name = {**SAME_NAME, "process_date": process_date}
    inputs, report = validate(write_collection(tmp_path, actions=[DIVIDEND, name]))
    assert report["status"] == "PASS", report["defects"]
    assert report["counts"]["same_ticker_name_changes_non_gating"] == 1
    assert inputs["distributions"] == {"2024-12-02": 1.5}


def test_same_ticker_name_change_cannot_hide_a_split(window: None, tmp_path: Path) -> None:
    name = {**SAME_NAME, "process_date": "2024-12-02"}
    _, report = validate(write_collection(tmp_path, actions=[DIVIDEND, name, SPLIT]))
    assert report["defects"] == ["corporate_action_stop:split:forward_splits:2024-12-03"]


@pytest.mark.parametrize(
    "name",
    [
        NAME_CHANGE,  # SPY -> SPYX: a real identity change
        {"group": "name_changes", "old_symbol": "SPY", "process_date": "2024-12-02"},
        {"group": "name_changes", "old_symbol": "SPYX", "new_symbol": "SPY"},
    ],
)
def test_symbol_identity_change_is_unsupported_valuation_not_a_split(
    window: None, tmp_path: Path, name: dict
) -> None:
    _, report = validate(write_collection(tmp_path, actions=[DIVIDEND, name]))
    assert report["defects"] == ["corporate_action_unsupported_undated:name_changes:1"]
    assert report["counts"]["same_ticker_name_changes_non_gating"] == 0


def test_issuer_capital_gain_legs_are_summed_and_reach_the_benchmark(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    issuer.install([issuer.row("12/02/2024", "1.500000", "0.100000", "0.400000")])
    directory, validation = validated(tmp_path, actions=[DIVIDEND])
    inputs, report = validate(directory)
    assert report["status"] == "PASS", report["defects"]
    assert inputs["distributions"] == {"2024-12-02": pytest.approx(1.5 + 0.4 + 0.1)}
    assert report["counts"]["issuer_capital_gain_legs_present"] == 2
    result = core.run_score(directory, validation, tmp_path / "result.json")
    spy = (106.31 / ANCHOR_CLOSE) * (1 + 2.0 / 104.23)
    assert result["diagnostics"]["spy_displacement"]["wealth_end"] == pytest.approx(spy)
    assert result["diagnostics"]["exdiv_flat"]["mean_bp"] == 0.0
    # The Alpaca dividend (1.5) still matches the issuer's dividend column, not the total.
    assert report["alpaca_dividend_cross_check"]["matches"] == 1


def test_blank_capital_gain_is_no_data_never_zero_and_zero_is_a_value(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    issuer.install([issuer.row("12/02/2024", "1.500000", "", "")])
    inputs, report = validate(write_collection(tmp_path / "blank"))
    assert inputs["distributions"] == {"2024-12-02": 1.5}
    assert report["counts"]["issuer_capital_gain_legs_blank"] == 2
    assert report["counts"]["issuer_capital_gain_legs_present"] == 0
    issuer.install([issuer.row("12/02/2024", "1.500000", "0.000000", "")])
    inputs, report = validate(write_collection(tmp_path / "zero"))
    assert inputs["distributions"] == {"2024-12-02": 1.5}
    assert report["counts"]["issuer_capital_gain_legs_blank"] == 1
    assert report["counts"]["issuer_capital_gain_legs_present"] == 1


ISSUER_ROW_DEFECTS = [
    ("wrong CUSIP", ("12/02/2024", "1.5", "", "", "SPY", "000000000"), "identity_mismatch:row_3"),
    ("wrong ticker", ("12/02/2024", "1.5", "", "", "SPYG", "78462F103"), "identity_mismatch:row_3"),
    ("malformed date", ("2024-12-02", "1.5"), "ex_date_malformed:row_3"),
    ("impossible date", ("02/30/2024", "1.5"), "ex_date_malformed:row_3"),
    ("non-session date", ("11/28/2024", "1.5"), "ex_date_not_session:2024-11-28"),
    ("zero amount", ("12/03/2024", "0.000000"), "dividend_invalid:2024-12-03"),
    ("negative amount", ("12/03/2024", "-1.5"), "dividend_invalid:2024-12-03"),
    ("blank amount", ("12/03/2024", ""), "dividend_invalid:2024-12-03"),
    ("text amount", ("12/03/2024", "n/a"), "dividend_invalid:2024-12-03"),
    ("negative capital gain", ("12/03/2024", "1.5", "-0.1"), "capital_gain_invalid:2024-12-03"),
    ("text capital gain", ("12/03/2024", "1.5", "", "n/a"), "capital_gain_invalid:2024-12-03"),
    ("duplicate ex-date", ("12/02/2024", "1.6"), "duplicate_ex_date:2024-12-02"),
]


@pytest.mark.parametrize(("case", "fields", "defect"), ISSUER_ROW_DEFECTS)
def test_issuer_row_checks_fixed_in_advance(
    window: None,
    tmp_path: Path,
    issuer: SimpleNamespace,
    case: str,
    fields: tuple,
    defect: str,
) -> None:
    issuer.install([issuer.row("12/02/2024", "1.500000"), issuer.row(*fields)])
    inputs, report = validate(write_collection(tmp_path))
    assert inputs is None and report["label"] == "UNRESOLVED (data)"
    assert f"issuer_{defect}" in report["defects"]
    assert "n/a" not in json.dumps(report["defects"])  # file text is never echoed


def test_other_funds_in_the_workbook_are_ignored(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    other = issuer.row("12/02/2024", "9.999999", ticker="QQQ", cusip="46090E103")
    issuer.install([issuer.row("12/02/2024", "1.500000"), other])
    inputs, report = validate(write_collection(tmp_path))
    assert inputs["distributions"] == {"2024-12-02": 1.5}
    assert report["counts"]["issuer_spy_rows"] == 1


def test_issuer_schedule_must_show_every_window_quarter(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    # Only Alpaca's matching dividend would have satisfied r8; the check now reads the issuer.
    issuer.install([issuer.row("09/20/2024", "1.500000")])
    inputs, report = validate(write_collection(tmp_path, actions=[DIVIDEND]))
    assert inputs is None
    assert report["defects"] == ["dividend_coverage_missing:2024Q4"]


def test_issuer_rows_outside_the_window_are_checked_for_shape_not_sessions(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    old = issuer.row("03/19/1993", "0.213191")  # a valid 1993 ex-date: outside, never a session
    issuer.install([issuer.row("12/02/2024", "1.500000"), old])
    inputs, _ = validate(write_collection(tmp_path / "ok"))
    assert inputs is not None
    issuer.install([issuer.row("12/02/2024", "1.500000"), issuer.row("03/19/1993", "0")])
    _, report = validate(write_collection(tmp_path / "bad"))
    assert report["defects"] == ["issuer_dividend_invalid:1993-03-19"]


def test_ex_date_on_the_anchor_is_not_a_window_distribution(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    issuer.install([issuer.row("11/27/2024", "1.4"), issuer.row("12/02/2024", "1.500000")])
    inputs, report = validate(write_collection(tmp_path))
    assert inputs["distributions"] == {"2024-12-02": 1.5}  # bought at the anchor close
    assert report["counts"]["issuer_ex_dates_in_window"] == 1


def test_issuer_workbook_hash_mismatch_and_missing_file_are_refused(
    window: None, tmp_path: Path, issuer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_collection(tmp_path)
    issuer.install(
        [issuer.row("12/02/2024", "1.500000"), issuer.row("12/03/2024", "1.6")], pin=False
    )
    with pytest.raises(ValueError, match="^issuer_workbook_hash_mismatch$"):
        core.build_manifest()
    inputs, report = validate(directory)
    assert inputs is None
    assert {
        "identity:issuer_workbook_hash_mismatch",
        "issuer_schedule:issuer_workbook_hash_mismatch",
    } <= set(report["defects"])
    full = core.validate_dataset(core.load_dataset(directory))
    assert full["identities"]["issuer_workbook_sha256"] is None
    monkeypatch.setattr(core, "ISSUER_WORKBOOK_PATH", tmp_path / "absent.xlsx")
    with pytest.raises(ValueError, match="^issuer_workbook_missing$"):
        core.issuer_workbook_sha256()


def test_registered_issuer_hash_is_the_approved_one() -> None:
    assert REAL_ISSUER_SHA256 == (
        "6b65cbf956b5d25705d08470006a0e97e6a7d2a0d5cfd08b63f6f79783b06320"
    )
    assert REAL_ISSUER_SHA256 == hashlib.sha256(REAL_ISSUER.read_bytes()).hexdigest()
    assert REAL_ISSUER.name == "spdr-etf-historical-distributions.xlsx"


@pytest.mark.parametrize("damage", ["header", "garbage"])
def test_unreadable_or_mislabelled_workbook_is_refused(
    window: None,
    tmp_path: Path,
    issuer: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    if damage == "header":
        header = ["FUND NAME", "TICKER", "CUSIP", "EX-DATE", "", "", "AMOUNT", "", "", ""]
        issuer.install([issuer.row("12/02/2024", "1.5")], header=header)
        reason = "issuer_workbook_header_mismatch"
    else:
        path = issuer.install([issuer.row("12/02/2024", "1.5")])
        path.write_bytes(b"not a zip")
        monkeypatch.setattr(
            core, "ISSUER_WORKBOOK_SHA256", hashlib.sha256(b"not a zip").hexdigest()
        )
        reason = "issuer_workbook_unreadable"
    _, report = validate(write_collection(tmp_path))
    assert f"issuer_schedule:{reason}" in report["defects"]


def test_parser_matches_the_verified_extraction_of_the_real_workbook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "ISSUER_WORKBOOK_PATH", REAL_ISSUER)
    monkeypatch.setattr(core, "ISSUER_WORKBOOK_SHA256", REAL_ISSUER_SHA256)
    reference = json.loads(ISSUER_REFERENCE.read_text())
    rows = core.parse_issuer_workbook(REAL_ISSUER.read_bytes())
    spy = {row["row"]: row for row in rows if row.get("B") == "SPY"}
    assert len(spy) == reference["all_SPY_rows"] == 136
    for record in reference["records"]:
        row = spy[record["worksheet_row"]]
        assert {col: row[col] for col in "BCDGHI"} == {col: record[col] for col in "BCDGHI"}
        assert core._issuer_date(row["D"]) == record["ex_date"]
    sessions = {s["date"]: s for s in core.offline_schedule(core.ANCHOR, core.LAST)}
    defects: list[str] = []
    payouts, dividends, info = core.issuer_distributions(sessions, defects)
    assert defects == []
    assert sorted(payouts) == sorted(r["ex_date"] for r in reference["records"])
    assert info["issuer_ex_dates_in_window"] == 43 == len(dividends)
    assert info["issuer_window_count_matches_expected"] is True
    assert "2016-03-18" in dividends and "2018-06-15" in dividends  # Alpaca lacks both
    assert dividends["2016-03-18"] == pytest.approx(1.049604)
    assert dividends["2018-06-15"] == pytest.approx(1.245568)
    assert info["issuer_capital_gain_legs_blank"] > 0  # blanks are counted, not zeroed


@pytest.mark.parametrize(
    "actions",
    [
        [{**DIVIDEND, "rate": 1.366242}],  # equal
        [{**DIVIDEND, "rate": 1.3662419}],  # the real 2020-06-19 case: differs only at 7 dp
    ],
)
def test_cross_check_compares_at_six_decimals_and_records_gaps(
    window: None, tmp_path: Path, issuer: SimpleNamespace, actions: list
) -> None:
    issuer.install(
        [
            issuer.row("12/02/2024", "1.366242"),
            issuer.row("12/03/2024", "1.000000"),
        ]
    )
    extra = {**DIVIDEND, "ex_date": "2024-11-29", "rate": 0.7}  # Alpaca-only ex-date
    differ = {**DIVIDEND, "ex_date": "2024-12-03", "rate": 1.000002}  # differs at 6 dp
    inputs, report = validate(write_collection(tmp_path, actions=[*actions, extra, differ]))
    assert inputs is not None and report["status"] == "PASS"
    check = report["alpaca_dividend_cross_check"]
    assert check["decimals"] == 6 and check["gating"] is False
    assert check["shared_ex_dates"] == 2 and check["matches"] == 1
    assert check["mismatches"] == [{"ex_date": "2024-12-03", "issuer": 1.0, "alpaca": 1.000002}]
    assert check["missing_from_issuer"] == ["2024-11-29"]
    assert check["missing_from_alpaca"] == []


def test_cross_check_records_an_unrepresentable_alpaca_rate_without_crashing(
    window: None, tmp_path: Path, issuer: SimpleNamespace
) -> None:
    issuer.install([issuer.row("12/02/2024", "1.366242")])
    huge = {**DIVIDEND, "ex_date": "2024-12-02", "rate": 1e30}
    inputs, report = validate(write_collection(tmp_path, actions=[huge]))
    assert inputs is not None and report["status"] == "PASS"
    check = report["alpaca_dividend_cross_check"]
    assert check["matches"] == 0 and check["mismatches"][0]["alpaca"] == 1e30


def test_close_comparison_leaves_out_both_excluded_sessions() -> None:
    closes = {"2019-08-09": 100.0, "2019-08-13": 150.0, "2019-08-14": 100.0}
    daily = {"2019-08-09": 100.0, "2019-08-13": 100.0, "2019-08-14": 100.0}
    result = core._close_comparison(closes, daily, [])
    assert result["sessions"] == 2 and result["max_abs_bp"] == 0.0


def test_cross_check_records_dates_missing_from_alpaca_without_blocking(
    window: None, tmp_path: Path
) -> None:
    inputs, report = validate(write_collection(tmp_path, actions=[]))
    assert inputs is not None
    check = report["alpaca_dividend_cross_check"]
    assert check["missing_from_alpaca"] == ["2024-12-02"] and check["matches"] == 0


# --- vendor adjustment review evidence ---------------------------------------------------------


def test_registered_review_path_schema_and_binding_are_r9(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert DEFAULT_REVIEW_PATH == core.STUDY_DIR / "_intraday_r9_vendor_adjustment_review.json"
    assert core.VENDOR_REVIEW_SCHEMA == "intraday_r9_vendor_adjustment_review"
    assert core.PROTOCOL_PATH.name == "2026-11-15-intraday-feasibility-r9.md"
    assert core.PROTOCOL_SHA256 == (
        "222521cebe23aed66f9a44672dd1645a210f0c545a7cfbd30482496922593ed3"
    )
    assert core.protocol_sha256() == core.PROTOCOL_SHA256  # the protocol file is untouched
    binding = core.vendor_review_binding()
    assert binding["protocol_sha256"] == core.PROTOCOL_SHA256
    assert binding["schema"] == "intraday_r9_vendor_adjustment_review"
    # A synthetic record written exactly to the r9 binding is accepted at the registered path.
    record = {**binding, "status": "reviewed", "known_unresolved_changes": []}
    path = tmp_path / "r9_review.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(core, "VENDOR_REVIEW_PATH", path)
    assert core.vendor_review_sha256() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_r8_review_record_cannot_satisfy_r9(monkeypatch: pytest.MonkeyPatch) -> None:
    assert R8_REVIEW.exists()  # kept untouched as history
    r8 = json.loads(R8_REVIEW.read_text())
    assert r8["schema"] == "intraday_r8_vendor_adjustment_review"
    monkeypatch.setattr(core, "VENDOR_REVIEW_PATH", R8_REVIEW)
    with pytest.raises(ValueError, match="^vendor_review_binding_mismatch$"):
        core.vendor_review_sha256()
    with pytest.raises(ValueError, match="^vendor_review_binding_mismatch$"):
        core.build_manifest()
    # Even re-pointed at the r8 file's own fields, schema and protocol hash differ from r9's.
    differing = {key for key, value in core.vendor_review_binding().items() if r8.get(key) != value}
    assert {"schema", "protocol_sha256"} <= differing


def test_missing_r9_review_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(core, "VENDOR_REVIEW_PATH", tmp_path / "absent.json")
    with pytest.raises(ValueError, match="^vendor_review_missing$"):
        core.build_manifest()


def test_manifest_and_validation_identities_bind_the_review_file(
    window: None, tmp_path: Path, review: Path
) -> None:
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    assert core.build_manifest()["vendor_review_sha256"] == digest
    _, validation = validated(tmp_path)
    assert json.loads(validation.read_text())["identities"]["vendor_review_sha256"] == digest


BAD_REVIEWS = [
    ("missing", None, "vendor_review_missing"),
    ("not json", "{", "vendor_review_malformed"),
    ("list", "[]", "vendor_review_malformed"),
    ("pending", {"status": "pending"}, "vendor_review_not_reviewed"),
    (
        "documented change",
        {"known_unresolved_changes": [{"url": "https://example.invalid/change"}]},
        "vendor_review_stop:documented_adjustment_change",
    ),
    ("changes not a list", {"known_unresolved_changes": None}, "vendor_review_malformed"),
    ("other symbol", {"symbol": "QQQ"}, "vendor_review_binding_mismatch"),
    (
        "other window",
        {"window": {"anchor": "2016-01-04", "end": "2026-09-30"}},
        "vendor_review_binding_mismatch",
    ),
    ("other protocol", {"protocol_sha256": "0" * 64}, "vendor_review_binding_mismatch"),
    (
        "r8 protocol",
        {"protocol_sha256": "be395d950c1da7abf010acf90ffbe461db597672a21dd72b73c0294ba7b45b85"},
        "vendor_review_binding_mismatch",
    ),
    (
        "r8 schema",
        {"schema": "intraday_r8_vendor_adjustment_review"},
        "vendor_review_binding_mismatch",
    ),
    ("price data", {"market_data_requested": True}, "vendor_review_binding_mismatch"),
]


def apply_review(path: Path, content: object) -> None:
    if content is None:
        path.unlink()
    elif isinstance(content, str):
        path.write_text(content)
    else:
        write_review(path, **content)


@pytest.mark.parametrize(("case", "content", "reason"), BAD_REVIEWS)
def test_bad_review_refuses_manifest_and_validation_pass(
    window: None, tmp_path: Path, review: Path, case: str, content: object, reason: str
) -> None:
    directory = write_collection(tmp_path)
    apply_review(review, content)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        core.build_manifest()
    inputs, report = validate(directory)
    assert inputs is None and f"identity:{reason}" in report["defects"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ({"reviewed_on": "2026-10-26"}, "validation_identity_mismatch"),  # edited, still valid
        (
            {"known_unresolved_changes": [{"url": "https://example.invalid/change"}]},
            "vendor_review_stop:documented_adjustment_change",
        ),
        (None, "vendor_review_missing"),
    ],
)
def test_review_edited_after_pass_refuses_to_score(
    window: None, tmp_path: Path, review: Path, content: object, reason: str
) -> None:
    directory, validation = validated(tmp_path)
    apply_review(review, content)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


# --- runtime identity ------------------------------------------------------------------------


def test_manifest_binds_runtime_versions() -> None:
    runtime = core.build_manifest()["runtime"]
    assert set(runtime) == {
        "python",
        "numpy",
        "pandas",
        "statsmodels",
        "requests",
        "pandas_market_calendars",
    }
    assert runtime == core.runtime_versions()


@pytest.mark.parametrize("library", ["statsmodels", "numpy"])
def test_changed_runtime_after_pass_refuses_to_score(
    window: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: str
) -> None:
    directory, validation = validated(tmp_path)
    changed = {**core.runtime_versions(), library: "0.0.0-changed"}
    monkeypatch.setattr(core, "runtime_versions", lambda: changed)
    _, report = validate(directory)
    assert "identity:manifest_runtime_mismatch" in report["defects"]
    with pytest.raises(ValueError, match="validation_identity_mismatch"):
        core.run_score(directory, validation, tmp_path / "result.json")
    assert not (directory / core.SCORE_CLAIM_FILE).exists()


# --- corrupt chunk structure ---------------------------------------------------------------


@pytest.mark.parametrize("payload", ["[1, 2]", "null", "5", '"CORRUPT-PAYLOAD"'])
def test_non_object_chunk_json_is_a_structured_defect(
    window: None, tmp_path: Path, payload: str
) -> None:
    directory = write_collection(tmp_path)
    path = core.chunk_path(directory, "calendar")
    path.write_text(payload)
    progress = json.loads((directory / core.PROGRESS_FILE).read_text())
    progress["chunks"]["calendar"] = core.file_sha256(path)
    (directory / core.PROGRESS_FILE).write_text(json.dumps(progress))
    inputs, report = validate(directory)
    assert inputs is None
    assert {"chunk_unreadable:calendar", "provenance_incomplete:calendar"} <= set(report["defects"])
    assert "CORRUPT" not in json.dumps(report)


def test_non_object_page_entry_is_a_structured_defect(window: None, tmp_path: Path) -> None:
    directory = write_collection(tmp_path)
    path = core.chunk_path(directory, "calendar")
    chunk = json.loads(path.read_text()) | {"pages": [5]}
    path.write_text(json.dumps(chunk))
    progress = json.loads((directory / core.PROGRESS_FILE).read_text())
    progress["chunks"]["calendar"] = core.file_sha256(path)
    (directory / core.PROGRESS_FILE).write_text(json.dumps(progress))
    _, report = validate(directory)
    assert "provenance_invalid:calendar" in report["defects"]
