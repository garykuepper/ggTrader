"""Signal-blind validation and single-shot scoring for the approved r8 intraday SPY screen.

Schema and public interface (all versions are 1):

- ``build_manifest() -> dict``: the frozen input request manifest, derived from the protocol
  constants alone (official calendar, raw SIP minute bars, raw SIP daily bars, corporate actions).
- Storage (written by ``intraday_momentum_data``): ``manifest.json``, ``progress.json`` and one
  write-once ``chunks/<id>.json`` per *completely paginated* logical request.
- ``load_dataset(directory) -> dict``: rehash every stored file into the dataset schema
  ``{"schema", "version", "manifest", "chunks", "identities", "integrity_defects"}``.
- ``validate_dataset(dataset) -> dict``: PASS or UNRESOLVED (data); counts and defects only, never
  a signal, side or return.
- ``run_validation(directory, output) -> dict``: write-once validation artifact.
- ``score_inputs(inputs) -> dict``: pure scoring of extracted endpoint prices.
- ``run_score(directory, validation_path, output) -> dict``: one score, only on a PASS artifact
  whose protocol, code and data identities match a fresh rehash and revalidation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import statsmodels
import statsmodels.api as sm

from ggTrader.lab.intraday_probe import (
    NEW_YORK,
    SchemaError,
    atomic_checkpoint,
    canonical_hash,
    timestamp_ns,
    utc_bounds,
    valid_date,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = PROJECT_ROOT / "docs/research/briefs/2026-10-25-intraday-feasibility-r8.md"
# SHA256 of the whole approved r8 file; any edit must be a deliberate, re-approved code change.
PROTOCOL_SHA256 = "be395d950c1da7abf010acf90ffbe461db597672a21dd72b73c0294ba7b45b85"
CODE_FILES = (
    "src/ggTrader/lab/intraday_momentum.py",
    "src/ggTrader/lab/intraday_momentum_data.py",
    "src/ggTrader/lab/intraday_probe.py",
    "scripts/intraday_momentum_research.py",
)

ANCHOR = "2016-01-04"  # signal input only, never scored
LAST = "2026-09-30"
FULL_HOURS = ("09:30", "16:00")
HALF_HOURS = ("09:30", "13:00")
BASE_COST = 1e-4  # per side
STRESS_COST = 2e-4
TAX_RATE = 0.24
A1_THRESHOLD = 3.0
A2_THRESHOLD = 0.03
NW_LAGS = 5
DIAGNOSTIC_LAGS = (0, 10, 20)
SUBPERIODS = (
    ("2016-2019", "2016-01-01", "2019-12-31"),
    ("2020-2023", "2020-01-01", "2023-12-31"),
    ("2024-2026-09", "2024-01-01", "2026-09-30"),
)
MINUTE_NS = 60_000_000_000
# Any SPY action other than a cash dividend inside the window stops the study (r8 D4.3).
STOP_ACTION_GROUPS = frozenset(
    {
        "capital_gains_distributions",
        "cash_mergers",
        "forward_splits",
        "name_changes",
        "partial_calls",
        "redemptions",
        "reorganizations",
        "reverse_splits",
        "rights_distributions",
        "spin_offs",
        "stock_and_cash_mergers",
        "stock_dividends",
        "stock_mergers",
        "unit_splits",
        "worthless_removals",
    }
)

ENDPOINTS = {
    "calendar": "https://paper-api.alpaca.markets/v2/calendar",
    "minute_bars": "https://data.alpaca.markets/v2/stocks/bars",
    "daily_bars": "https://data.alpaca.markets/v2/stocks/bars",
    "corporate_actions": "https://data.alpaca.markets/v1/corporate-actions",
}
BAR_PARAMETERS = {
    "symbols": "SPY",
    "adjustment": "raw",
    "feed": "sip",
    "asof": "-",
    "sort": "asc",
    "limit": 10000,
}
CORPORATE_ACTION_MARGIN_DAYS = 31  # the endpoint filters on process_date, not ex_date

MANIFEST_FILE = "manifest.json"
PROGRESS_FILE = "progress.json"
CHUNK_DIR = "chunks"
SCHEMAS = {
    "manifest": "intraday_momentum_r8_manifest",
    "chunk": "intraday_momentum_r8_chunk",
    "dataset": "intraday_momentum_r8_dataset",
    "validation": "intraday_momentum_r8_validation",
    "score": "intraday_momentum_r8_score",
}
LABEL = "Post-original-sample and single-shot proxy feasibility screen (not an untouched holdout)"
MISMATCH_NOTE = (
    "A standalone A1/A2 pass says nothing about beating SPY going forward; SPY displacement "
    "is a mandatory non-gating diagnostic."
)


# --- identities ----------------------------------------------------------------------------


def protocol_sha256() -> str:
    """Hash the approved r8 file and refuse if it differs from the frozen identity."""
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("protocol_hash_mismatch")
    return digest


def code_sha256() -> str:
    """Hash every source file the pipeline executes, independent of git state."""
    return canonical_hash(
        {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in CODE_FILES
        }
    )


def runtime_versions() -> dict:
    """Record the execution runtime that the r8 evidence appendix requires."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
        "requests": requests.__version__,
    }


def chunk_path(directory: Path, request_id: str) -> Path:
    """Map a manifest request id to its write-once chunk file."""
    return Path(directory) / CHUNK_DIR / f"{request_id.replace(':', '_')}.json"


def file_sha256(path: Path) -> str:
    """Hash stored bytes exactly as they will be read."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- frozen manifest -------------------------------------------------------------------------


def _months() -> list[str]:
    first, last = date.fromisoformat(ANCHOR), date.fromisoformat(LAST)
    months, year, month = [], first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def build_manifest() -> dict:
    """Freeze every request before any response exists; nothing here depends on data."""
    requests_ = [
        {
            "id": "calendar",
            "kind": "calendar",
            "endpoint": ENDPOINTS["calendar"],
            "parameters": {"start": ANCHOR, "end": LAST},
        },
        {
            "id": "daily_bars",
            "kind": "daily_bars",
            "endpoint": ENDPOINTS["daily_bars"],
            "parameters": {**BAR_PARAMETERS, "timeframe": "1Day", "start": ANCHOR, "end": LAST},
        },
        {
            "id": "corporate_actions",
            "kind": "corporate_actions",
            "endpoint": ENDPOINTS["corporate_actions"],
            "parameters": {
                "symbols": "SPY",
                "start": (
                    date.fromisoformat(ANCHOR) - timedelta(days=CORPORATE_ACTION_MARGIN_DAYS)
                ).isoformat(),
                "end": (
                    date.fromisoformat(LAST) + timedelta(days=CORPORATE_ACTION_MARGIN_DAYS)
                ).isoformat(),
                "limit": 1000,
                "sort": "asc",
            },
        },
    ]
    for month in _months():
        first = date.fromisoformat(f"{month}-01")
        last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        start = utc_bounds(first.isoformat(), "00:00", "00:00")[0]
        end = utc_bounds(last.isoformat(), "23:59:59", "23:59:59")[0]
        requests_.append(
            {
                "id": f"minute_bars:{month}",
                "kind": "minute_bars",
                "endpoint": ENDPOINTS["minute_bars"],
                "parameters": {**BAR_PARAMETERS, "timeframe": "1Min", "start": start, "end": end},
            }
        )
    return {
        "schema": SCHEMAS["manifest"],
        "version": 1,
        "protocol_sha256": protocol_sha256(),
        "code_sha256": code_sha256(),
        "window": {"anchor": ANCHOR, "last": LAST},
        "requests": requests_,
    }


# --- dataset loading -------------------------------------------------------------------------


def load_dataset(directory: Path) -> dict:
    """Rehash a stored collection; integrity faults are reported, never repaired."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    progress_path = directory / PROGRESS_FILE
    recorded = (
        json.loads(progress_path.read_text()).get("chunks", {}) if progress_path.exists() else {}
    )
    chunks, hashes, defects = {}, {}, []
    for spec in manifest.get("requests", []):
        path = chunk_path(directory, spec["id"])
        if not path.exists():
            chunks[spec["id"]], hashes[spec["id"]] = None, None
            continue
        hashes[spec["id"]] = file_sha256(path)
        if recorded.get(spec["id"]) != hashes[spec["id"]]:
            defects.append(f"chunk_hash_mismatch:{spec['id']}")
        try:
            chunks[spec["id"]] = json.loads(path.read_text())
        except ValueError:
            chunks[spec["id"]] = None
            defects.append(f"chunk_unreadable:{spec['id']}")
    manifest_hash = canonical_hash(manifest)
    return {
        "schema": SCHEMAS["dataset"],
        "version": 1,
        "manifest": manifest,
        "chunks": chunks,
        "integrity_defects": defects,
        "identities": {
            "manifest_sha256": manifest_hash,
            "data_manifest_sha256": canonical_hash(
                {"manifest_sha256": manifest_hash, "chunks": hashes}
            ),
        },
    }


# --- signal-blind validation -----------------------------------------------------------------


def _price(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _minute_ns(day: str, clock: str) -> int:
    local = datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=NEW_YORK)
    return int(local.timestamp()) * 1_000_000_000


def _shift(clock: str, minutes: int) -> str:
    value = datetime.strptime(clock, "%H:%M") + timedelta(minutes=minutes)
    return value.strftime("%H:%M")


def _calendar(rows: Any, defects: list[str]) -> list[dict]:
    """Return calendar sessions, recording every malformed or unexpected row."""
    if not isinstance(rows, list) or not rows:
        defects.append("calendar_malformed:empty_or_not_list")
        return []
    sessions = []
    for index, row in enumerate(rows):
        try:
            day = valid_date(row.get("date") if isinstance(row, dict) else None)
        except SchemaError:
            defects.append(f"calendar_malformed:row_{index}")
            continue
        hours = (row.get("open"), row.get("close"))
        if date.fromisoformat(day).weekday() >= 5:
            defects.append(f"calendar_weekend:{day}")
        if hours not in (FULL_HOURS, HALF_HOURS):
            defects.append(f"calendar_unexpected_hours:{day}")
            continue
        sessions.append({"date": day, "half": hours == HALF_HOURS, "close": hours[1]})
    days = [row["date"] for row in sessions]
    if days != sorted(set(days)):
        defects.append("calendar_malformed:not_strictly_increasing")
    if not days or days[0] != ANCHOR:
        defects.append("calendar_bounds:anchor_missing")
    if not days or days[-1] != LAST:
        defects.append("calendar_bounds:last_session_missing")
    if any(not ANCHOR <= day <= LAST for day in days):
        defects.append("calendar_bounds:outside_window")
    return sessions


def _provenance(dataset: dict, defects: list[str]) -> dict:
    """Require the exact frozen manifest and a complete chunk for every request."""
    manifest = dataset["manifest"]
    expected = build_manifest()
    if canonical_hash(manifest) != canonical_hash(expected):
        defects.append("provenance:manifest_not_frozen_protocol_manifest")
    defects.extend(dataset.get("integrity_defects", []))
    good = {}
    for spec in manifest.get("requests", []):
        chunk = dataset["chunks"].get(spec["id"])
        if chunk is None:
            defects.append(f"provenance_incomplete:{spec['id']}")
            continue
        if (
            chunk.get("schema") != SCHEMAS["chunk"]
            or chunk.get("complete") is not True
            or chunk.get("id") != spec["id"]
            or chunk.get("kind") != spec["kind"]
            or chunk.get("endpoint") != spec["endpoint"]
            or chunk.get("parameters") != spec["parameters"]
            or not chunk.get("pages")
            or not all(page.get("body_sha256") for page in chunk["pages"])
            or not isinstance(chunk.get("records"), list)
        ):
            defects.append(f"provenance_invalid:{spec['id']}")
            continue
        good[spec["id"]] = chunk
    return good


def _minute_index(rows: list, defects: list[str]) -> tuple[dict, list[int], list[int]]:
    by_ns: dict[int, list[dict]] = defaultdict(list)
    misaligned, stamps = [], []
    for index, row in enumerate(rows):
        try:
            stamp = timestamp_ns(row.get("t") if isinstance(row, dict) else None)
        except SchemaError:
            defects.append(f"minute_bar_bad_timestamp:{index}")
            continue
        stamps.append(stamp)
        if stamp % MINUTE_NS:
            misaligned.append(stamp)
        else:
            by_ns[stamp].append(row)
    return by_ns, misaligned, stamps


def extract(dataset: dict) -> tuple[dict | None, dict]:
    """Return (score inputs or None, validation report). Never computes a signal or return."""
    defects: list[str] = []
    try:
        if dataset["manifest"].get("protocol_sha256") != protocol_sha256():
            defects.append("identity:manifest_protocol_mismatch")
        if dataset["manifest"].get("code_sha256") != code_sha256():
            defects.append("identity:manifest_code_mismatch")
    except ValueError as error:
        defects.append(f"identity:{error}")
    chunks = _provenance(dataset, defects)
    calendar_chunk = chunks.get("calendar")
    sessions = _calendar(calendar_chunk["records"] if calendar_chunk else None, defects)
    session_days = {row["date"]: row for row in sessions}

    minute_rows = [
        row
        for spec in dataset["manifest"].get("requests", [])
        if spec["kind"] == "minute_bars" and spec["id"] in chunks
        for row in chunks[spec["id"]]["records"]
    ]
    by_ns, misaligned, stamps = _minute_index(minute_rows, defects)
    misaligned_floor = {stamp - stamp % MINUTE_NS for stamp in misaligned}

    def endpoint(day: str, clock: str, field: str, role: str) -> float | None:
        stamp = _minute_ns(day, clock)
        rows = by_ns.get(stamp, [])
        if stamp in misaligned_floor:
            defects.append(f"endpoint_misaligned:{role}:{day}")
        if not rows:
            defects.append(f"endpoint_missing:{role}:{day}")
            return None
        if len(rows) > 1:
            defects.append(f"endpoint_duplicate:{role}:{day}")
            return None
        if not _price(rows[0].get(field)):
            defects.append(f"endpoint_invalid_price:{role}:{day}")
            return None
        return float(rows[0][field])

    closes, scored = {}, []
    for index, row in enumerate(sessions):
        day = row["date"]
        if not ANCHOR <= day <= LAST:
            continue
        # The last minute bar before the official close, 12:59 on a 13:00 early close.
        close = endpoint(day, _shift(row["close"], -1), "c", "close")
        closes[day] = close
        if day == ANCHOR or row["half"]:
            continue
        previous = sessions[index - 1]["date"] if index else None
        scored.append(
            {
                "date": day,
                "previous": previous,
                "p0959": endpoint(day, "09:59", "c", "signal_0959"),
                "p1530": endpoint(day, "15:30", "o", "entry_1530"),
                "p1559": close,
            }
        )
    for row in scored:
        row["prior_close"] = closes.get(row["previous"])
        if row["previous"] is None:
            defects.append(f"endpoint_missing:prior_close:{row['date']}")

    # Calendar/bar consistency: regular-hours trades on a non-session date, or a full session
    # whose bars stop before 13:00 (an unflagged half-day).
    local = pd.to_datetime(np.array(stamps, dtype="int64"), utc=True).tz_convert(NEW_YORK)
    local_days = np.asarray(local.strftime("%Y-%m-%d"))
    minutes = np.asarray(local.hour * 60 + local.minute)
    regular = (minutes >= 570) & (minutes < 960)
    for day in sorted(set(local_days[regular]) - set(session_days)):
        defects.append(f"calendar_mismatch:regular_bars_on_non_session:{day}")
    afternoon = set(local_days[regular & (minutes >= 780)])
    morning = set(local_days[regular & (minutes < 780)])
    for day in sorted(morning - afternoon):
        if day in session_days and not session_days[day]["half"]:
            defects.append(f"unflagged_half_day:{day}")

    # Interior minutes are a non-gating diagnostic (r8 D4.4).
    present = set(by_ns)
    gaps = {}
    for row in sessions:
        last = 960 if not row["half"] else 780
        grid = [_minute_ns(row["date"], "09:30") + i * MINUTE_NS for i in range(last - 570)]
        missing = sum(stamp not in present for stamp in grid)
        if missing:
            gaps[row["date"]] = missing

    daily = _daily_closes(chunks.get("daily_bars"), session_days, defects)
    dividends = _corporate_actions(chunks.get("corporate_actions"), session_days, defects)

    counts = {
        "calendar_sessions": len(sessions),
        "half_day_sessions": sum(row["half"] for row in sessions),
        "scored_sessions": len(scored),
        "minute_bars": len(minute_rows),
        "daily_bars": len(daily),
        "cash_dividends_in_window": len(dividends),
        "chunks_complete": len(chunks),
        "chunks_expected": len(dataset["manifest"].get("requests", [])),
    }
    diagnostics = {
        "interior_missing_minutes_total": sum(gaps.values()),
        "sessions_with_interior_gaps": len(gaps),
        "interior_gaps_by_session": gaps,
        "misaligned_bars": len(misaligned),
    }
    defects = sorted(set(defects))
    report = {
        "status": "UNRESOLVED" if defects else "PASS",
        "label": "UNRESOLVED (data)" if defects else "PASS",
        "defects": defects,
        "counts": counts,
        "diagnostics": diagnostics,
    }
    if defects:
        return None, report
    inputs = {
        "dates": [row["date"] for row in scored],
        "prior_close": [row["prior_close"] for row in scored],
        "p0959": [row["p0959"] for row in scored],
        "p1530": [row["p1530"] for row in scored],
        "p1559": [row["p1559"] for row in scored],
        "closes": closes,
        "half_days": [row["date"] for row in sessions if row["half"]],
        "daily_closes": daily,
        "dividends": dividends,
    }
    return inputs, report


def _daily_closes(chunk: dict | None, session_days: dict, defects: list[str]) -> dict:
    """Raw daily closes must cover exactly the calendar sessions."""
    if chunk is None:
        return {}
    closes = {}
    for index, row in enumerate(chunk["records"]):
        try:
            stamp = timestamp_ns(row.get("t") if isinstance(row, dict) else None)
        except SchemaError:
            defects.append(f"daily_bar_bad_timestamp:{index}")
            continue
        local = datetime.fromtimestamp(stamp // 1_000_000_000, UTC).astimezone(NEW_YORK)
        day = local.date().isoformat()
        if (local.hour, local.minute, local.second) != (0, 0, 0) or stamp % 1_000_000_000:
            defects.append(f"daily_bar_misaligned:{day}")
        elif day not in session_days:
            defects.append(f"daily_bar_non_session:{day}")
        elif day in closes:
            defects.append(f"daily_bar_duplicate:{day}")
        elif not _price(row.get("c")):
            defects.append(f"daily_bar_invalid_close:{day}")
        else:
            closes[day] = float(row["c"])
    for day in sorted(set(session_days) - set(closes)):
        defects.append(f"daily_bar_missing:{day}")
    return closes


def _corporate_actions(chunk: dict | None, session_days: dict, defects: list[str]) -> dict:
    """Dividends for the benchmark/ex-div diagnostics; any other SPY action stops the study.

    An empty response is never accepted as verified absence: every window quarter must show a
    cash-dividend ex-date (SPY distributes quarterly), so split absence is only accepted inside a
    complete all-types response that demonstrably covered the window.
    """
    if chunk is None:
        return {}
    dividends = {}
    for index, row in enumerate(chunk["records"]):
        if not isinstance(row, dict) or row.get("symbol") != "SPY":
            defects.append(f"corporate_action_malformed:{index}")
            continue
        try:
            day = valid_date(row.get("ex_date") or row.get("process_date"))
        except SchemaError:
            defects.append(f"corporate_action_malformed:{index}")
            continue
        if not ANCHOR <= day <= LAST:
            continue
        group = row.get("group")
        if group in STOP_ACTION_GROUPS:
            defects.append(f"corporate_action_stop:{group}:{day}")
        elif group != "cash_dividends":
            defects.append(f"corporate_action_unknown_group:{group}:{day}")
        elif not _price(row.get("rate")):
            defects.append(f"dividend_invalid_rate:{day}")
        elif day not in session_days:
            defects.append(f"dividend_ex_date_not_session:{day}")
        elif day in dividends:
            defects.append(f"dividend_duplicate_ex_date:{day}")
        elif day != ANCHOR:
            dividends[day] = float(row["rate"])
    first, last = date.fromisoformat(ANCHOR), date.fromisoformat(LAST)
    covered = {
        (date.fromisoformat(day).year, (date.fromisoformat(day).month - 1) // 3)
        for day in dividends
    }
    year, quarter = first.year, (first.month - 1) // 3
    while (year, quarter) <= (last.year, (last.month - 1) // 3):
        if (year, quarter) not in covered:
            defects.append(f"dividend_coverage_missing:{year}Q{quarter + 1}")
        year, quarter = (year + 1, 0) if quarter == 3 else (year, quarter + 1)
    return dividends


def validate_dataset(dataset: dict) -> dict:
    """Signal-blind validation report with the data identities it was computed on."""
    report = extract(dataset)[1]
    return {
        "schema": SCHEMAS["validation"],
        "version": 1,
        **report,
        "identities": {
            "protocol_sha256": PROTOCOL_SHA256,
            "code_sha256": code_sha256(),
            **dataset["identities"],
        },
    }


# --- pure scoring --------------------------------------------------------------------------


def strategy_returns(
    prior: Any, p0959: Any, p1530: Any, p1559: Any, cost_per_side: float
) -> dict[str, np.ndarray]:
    """Registered rule: long 15:30 open -> 15:59 close only if r_first > 0; flat days are zero."""
    prior, p0959, p1530, p1559 = (np.asarray(x, dtype=float) for x in (prior, p0959, p1530, p1559))
    r_first = p0959 / prior - 1
    r_last = p1559 / p1530 - 1
    side = (r_first > 0).astype(float)
    return {
        "r_first": r_first,
        "r_last": r_last,
        "side": side,
        "s": side * r_last - side * 2 * cost_per_side,
    }


def nw_t(values: Any, lags: int = NW_LAGS) -> float | None:
    """Intercept-only OLS t with Bartlett HAC; None when the variance is zero or non-finite."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return None
    fit = sm.OLS(values, np.ones(len(values))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    se = float(fit.bse[0])
    if not math.isfinite(se) or se <= 0:
        return None
    return float(fit.params[0]) / se


def tax_ledger(dates: list[str], s: Any, tax_rate: float = TAX_RATE) -> dict:
    """Compound 100% of after-debit equity; debit 24% of each year's positive net P&L.

    The debit lands after the year's last scored session (half-days are flat), so it reduces the
    next trade's sizing base. The final partial year accrues its liability at the end.
    """
    wealth, rows, year, pnl = 1.0, [], None, 0.0

    def close_year() -> None:
        nonlocal wealth
        tax = tax_rate * max(pnl, 0.0)
        wealth -= tax
        rows.append({"year": year, "net_pnl": pnl, "tax_debit": tax, "wealth_after_tax": wealth})

    for day, value in zip(dates, np.asarray(s, dtype=float), strict=True):
        if year is not None and int(day[:4]) != year:
            close_year()
            pnl = 0.0
        year = int(day[:4])
        gain = wealth * value
        pnl += gain
        wealth += gain
    if year is not None:
        close_year()
    return {"wealth_end": wealth, "annual": rows}


def registered_years() -> float:
    """y = days(anchor -> last) / 365.25."""
    return (date.fromisoformat(LAST) - date.fromisoformat(ANCHOR)).days / 365.25


def cagr(wealth_end: float) -> float | None:
    """CAGR over the registered calendar span; None if wealth is not positive."""
    if not math.isfinite(wealth_end) or wealth_end <= 0:
        return None
    return wealth_end ** (1 / registered_years()) - 1


def benchmark(closes: dict, dividends: dict) -> dict:
    """SPY with distributions reinvested at the ex-date session close, on the sleeve's prices.

    Starts at the anchor close-bar and ends at the last session's close-bar; no tax.
    """
    days = sorted(closes)
    units = 1.0 / closes[days[0]]
    for day in days[1:]:
        if day in dividends:
            units *= 1 + dividends[day] / closes[day]
    wealth = units * closes[days[-1]]
    return {"wealth_end": wealth, "cagr": cagr(wealth), "dividends_reinvested": len(dividends)}


def _summary(dates: list[str], s: np.ndarray) -> dict:
    ledger = tax_ledger(dates, s)
    t = nw_t(s)
    growth = cagr(ledger["wealth_end"])
    return {
        "mean_bp": float(np.mean(s) * 1e4) if len(s) else None,
        "nw5_t": t,
        "a1_pass": None if t is None else bool(t >= A1_THRESHOLD),
        "after_tax_cagr": growth,
        "a2_pass": bool(growth is not None and growth >= A2_THRESHOLD),
        "annual_tax_ledger": ledger["annual"],
    }


def _drawdown(s: np.ndarray) -> float:
    path = np.concatenate([[1.0], np.cumprod(1 + s)])
    return float(np.min(path / np.maximum.accumulate(path) - 1))


def _losing_streak(s: np.ndarray, side: np.ndarray) -> int:
    """Longest run of losing traded sessions; flat sessions are skipped, not counted."""
    best = run = 0
    for value in s[side > 0]:
        run = run + 1 if value < 0 else 0
        best = max(best, run)
    return best


def _attribution(side: np.ndarray, r_last: np.ndarray) -> dict:
    n = len(side)
    cov = float(np.cov(side, r_last, ddof=1)[0, 1]) if n > 1 else math.nan
    result = {
        "mean_d_times_r_bp": float(np.mean(side * r_last) * 1e4),
        "cov_term_bp": cov * (n - 1) / n * 1e4,
        "mean_product_term_bp": float(np.mean(side) * np.mean(r_last) * 1e4),
        "beta1": None,
        "beta1_nw5_t": None,
    }
    if np.ptp(side) > 0:
        fit = sm.OLS(r_last, sm.add_constant(side)).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAGS, "use_correction": True}
        )
        se = float(fit.bse[1])
        result["beta1"] = float(fit.params[1])
        result["beta1_nw5_t"] = float(fit.params[1]) / se if math.isfinite(se) and se > 0 else None
    return result


def _close_comparison(closes: dict, daily: dict, half_days: list[str]) -> dict:
    diff = np.array([(closes[d] - daily[d]) / daily[d] * 1e4 for d in sorted(closes)])
    magnitude = np.abs(diff)
    return {
        "sessions": len(diff),
        "half_day_sessions": len(half_days),
        "mean_signed_bp": float(np.mean(diff)),
        "mean_abs_bp": float(np.mean(magnitude)),
        "median_abs_bp": float(np.median(magnitude)),
        "p95_abs_bp": float(np.percentile(magnitude, 95)),
        "max_abs_bp": float(np.max(magnitude)),
    }


def _verdict(a1_t: float | None, a2_pass: bool) -> str:
    if a1_t is None:
        return "UNRESOLVED_STATISTICAL"
    return "GO" if a1_t >= A1_THRESHOLD and a2_pass else "NO-GO"


def score_inputs(inputs: dict) -> dict:
    """Score the registered rule at base cost; stress and diagnostics never change the verdict."""
    dates = inputs["dates"]
    legs = [inputs[key] for key in ("prior_close", "p0959", "p1530", "p1559")]
    base = strategy_returns(*legs, BASE_COST)
    stress = strategy_returns(*legs, STRESS_COST)
    s, side, r_first, r_last = base["s"], base["side"], base["r_first"], base["r_last"]
    primary = _summary(dates, s)
    verdict = _verdict(primary["nw5_t"], primary["a2_pass"])

    diagnostics, errors = {}, []

    def diagnostic(name: str, compute: Any) -> None:
        try:
            diagnostics[name] = compute()
        except Exception as error:  # a failed mandatory diagnostic is surfaced, never hidden
            errors.append(f"{name}:{type(error).__name__}")

    years = np.array([int(d[:4]) for d in dates])
    months = np.array([d[:7] for d in dates])
    traded = side > 0

    def subperiods() -> dict:
        result = {}
        for name, start, end in SUBPERIODS:
            mask = np.array([start <= d <= end for d in dates])
            part, part_side = s[mask], side[mask]
            result[name] = {
                "sessions": int(mask.sum()),
                "traded": int(part_side.sum()),
                "mean_bp": float(np.mean(part) * 1e4) if mask.any() else None,
                "nw5_t": nw_t(part),
                "hit_rate": float(np.mean(part[part_side > 0] > 0)) if part_side.any() else None,
            }
        return result

    def worst_month() -> dict:
        monthly = {m: float(np.prod(1 + s[months == m]) - 1) for m in sorted(set(months))}
        month = min(monthly, key=monthly.get)
        return {"month": month, "return": monthly[month]}

    def exdiv_flat() -> dict:
        flat = s.copy()
        flat[np.array([d in inputs["dividends"] for d in dates])] = 0.0
        summary = _summary(dates, flat)
        summary.pop("annual_tax_ledger")
        return summary

    def always_long() -> dict:
        net = r_last - 2 * BASE_COST
        summary = _summary(dates, net)
        summary.pop("annual_tax_ledger")
        return summary

    def displacement() -> dict:
        spy = benchmark(inputs["closes"], inputs["dividends"])
        return {
            **spy,
            "sleeve_after_tax_cagr": primary["after_tax_cagr"],
            "sleeve_minus_spy_cagr": (
                None
                if spy["cagr"] is None or primary["after_tax_cagr"] is None
                else primary["after_tax_cagr"] - spy["cagr"]
            ),
            "note": MISMATCH_NOTE,
        }

    gross = float(np.sum(side * r_last))
    trades = int(side.sum())
    diagnostic("stress", lambda: {**_summary(dates, stress["s"]), "cannot_change_verdict": True})
    diagnostic("nw_lags", lambda: {str(lag): nw_t(s, lag) for lag in DIAGNOSTIC_LAGS})
    diagnostic("subperiods", subperiods)
    diagnostic("break_even_one_way_cost_bp", lambda: gross / (2 * trades) * 1e4 if trades else None)
    diagnostic("worst_month", worst_month)
    diagnostic("longest_losing_streak_traded_sessions", lambda: _losing_streak(s, side))
    diagnostic("max_drawdown_pre_tax", lambda: _drawdown(s))
    diagnostic(
        "signal_counts",
        lambda: {
            "positive_traded": int(np.sum(r_first > 0)),
            "zero_flat": int(np.sum(r_first == 0)),
            "negative_flat": int(np.sum(r_first < 0)),
            "flat_sessions": int(np.sum(~traded)),
        },
    )
    diagnostic(
        "conditional_on_trading_mean_bp",
        lambda: float(np.mean(s[traded]) * 1e4) if trades else None,
    )
    diagnostic("exdiv_flat", exdiv_flat)
    diagnostic("attribution", lambda: _attribution(side, r_last))
    diagnostic("always_long_net", always_long)
    diagnostic("spy_displacement", displacement)
    diagnostic(
        "close_convention_comparison",
        lambda: _close_comparison(inputs["closes"], inputs["daily_closes"], inputs["half_days"]),
    )
    diagnostic(
        "yearly_mean_bp",
        lambda: {str(y): float(np.mean(s[years == y]) * 1e4) for y in sorted(set(years))},
    )
    if errors and verdict == "GO":
        verdict = "UNRESOLVED_DIAGNOSTICS"
    return {
        "verdict": verdict,
        "a1": {
            "nw5_t": primary["nw5_t"],
            "threshold": A1_THRESHOLD,
            "status": (
                "UNRESOLVED"
                if primary["nw5_t"] is None
                else "PASS"
                if primary["a1_pass"]
                else "FAIL"
            ),
            "mean_bp": primary["mean_bp"],
            "sessions": len(dates),
        },
        "a2": {
            "after_tax_cagr": primary["after_tax_cagr"],
            "threshold": A2_THRESHOLD,
            "status": "PASS" if primary["a2_pass"] else "FAIL",
            "years": registered_years(),
            "annual_tax_ledger": primary["annual_tax_ledger"],
        },
        "diagnostics": diagnostics,
        "diagnostic_errors": errors,
        "assumptions": ASSUMPTIONS,
    }


ASSUMPTIONS = [
    "Idealized bar proxy: 09:59 close signal, 15:30 open entry, 15:59 close exit, prior close = "
    "last minute bar before the preceding session's official close (12:59 on early closes).",
    "Costs 1 bp/side base (verdict) and 2 bp/side stress (reported only).",
    "100% of after-tax equity per trade, zero return on idle cash, fractional sizing ceiling.",
    "24% of each calendar year's positive net realized P&L debited after its last session; "
    "no loss benefit or carryforward; terminal partial year taxed at the end.",
    "CAGR = (W_end/W_0)^(1/y) - 1, y = days(anchor -> last)/365.25.",
    "SPY benchmark: same close-bar prices from the anchor, distributions reinvested at the "
    "ex-date session close, no tax.",
    "Losing streak counts consecutive losing traded sessions; flat sessions are skipped.",
    "Worst month and drawdown use the pre-tax base-cost sleeve.",
    "Hit rate is the share of traded sessions with a positive net return.",
]


# --- write-once artifacts ----------------------------------------------------------------------


def run_validation(directory: Path, output: Path) -> dict:
    """Validate once and publish the artifact; an existing artifact is never overwritten."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("validation_already_exists")
    result = {
        **validate_dataset(load_dataset(directory)),
        "runtime": runtime_versions(),
        "run_utc": datetime.now(UTC).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(output, result, initial=True)
    return result


VALIDATION_KEYS = (
    "schema",
    "version",
    "status",
    "label",
    "defects",
    "counts",
    "diagnostics",
    "identities",
)


def run_score(directory: Path, validation_path: Path, output: Path) -> dict:
    """Score exactly once against a PASS artifact that matches a fresh rehash and revalidation."""
    output = Path(output)
    report_path = output.with_suffix(".md")
    claim = output.with_suffix(".claim")
    if output.exists() or report_path.exists() or claim.exists():
        raise FileExistsError("score_already_claimed")
    validation_bytes = Path(validation_path).read_bytes()
    validation = json.loads(validation_bytes)
    if validation.get("status") != "PASS" or validation.get("schema") != SCHEMAS["validation"]:
        raise ValueError("validation_not_pass")
    protocol_sha256()
    dataset = load_dataset(directory)
    inputs, report = extract(dataset)
    fresh = {
        "schema": SCHEMAS["validation"],
        "version": 1,
        **report,
        "identities": {
            "protocol_sha256": PROTOCOL_SHA256,
            "code_sha256": code_sha256(),
            **dataset["identities"],
        },
    }
    if any(validation.get(key) != fresh[key] for key in VALIDATION_KEYS) or inputs is None:
        raise ValueError("validation_identity_mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Claim before computing: a crash after this point still forbids a second score.
    os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    result = {
        "schema": SCHEMAS["score"],
        "version": 1,
        "label": LABEL,
        "identities": fresh["identities"],
        "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "runtime": runtime_versions(),
        "run_utc": datetime.now(UTC).isoformat(),
        **score_inputs(inputs),
    }
    result = _json_safe(result)
    atomic_checkpoint(output, result, initial=True)
    _write_once(report_path, render_report(result))
    return result


def _json_safe(value: Any) -> Any:
    """Numpy scalars to Python, non-finite floats to null, so the claimed write cannot fail."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_once(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


def render_report(result: dict) -> str:
    """Render the GO/NO-GO summary from the raw score only; no hand-entered numbers."""
    a1, a2, diag = result["a1"], result["a2"], result["diagnostics"]
    stress = diag.get("stress", {})
    spy = diag.get("spy_displacement", {})
    lines = [
        "# Intraday r8 SPY proxy screen: automatic score",
        "",
        f"**Verdict: {result['verdict']}** — {result['label']}.",
        "",
        f"- A1 NW-5 t (unrounded, threshold {a1['threshold']}): {a1['nw5_t']!r} -> {a1['status']}",
        f"- A2 after-tax-policy CAGR (threshold {a2['threshold']}): {a2['after_tax_cagr']!r} "
        f"-> {a2['status']}",
        f"- Stress (2 bp/side, cannot change verdict): t {stress.get('nw5_t')!r}, "
        f"CAGR {stress.get('after_tax_cagr')!r}",
        f"- SPY distribution-reinvested CAGR (no tax): {spy.get('cagr')!r}; sleeve minus SPY: "
        f"{spy.get('sleeve_minus_spy_cagr')!r}",
        f"- {MISMATCH_NOTE}",
        f"- Diagnostic errors: {result['diagnostic_errors'] or 'none'}",
        "",
        f"Identities: {json.dumps(result['identities'], sort_keys=True)}",
        f"Validation SHA256: {result['validation_sha256']}",
        "",
        "All diagnostics are in the raw JSON next to this file.",
        "",
    ]
    return "\n".join(lines)
