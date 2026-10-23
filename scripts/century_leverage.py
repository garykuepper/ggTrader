"""Run the frozen century synthetic leverage study with resumable rolling windows."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.lab.century_leverage import (  # noqa: E402
    BRACKETS,
    END,
    FRENCH_FILE,
    MIXES,
    SHILLER_FILE,
    WINDOWS,
    build_legs,
    checked_file,
    load_french_daily,
    load_shiller_yield,
    recommendation_rule,
    simulate,
)

OUTPUT = ROOT / "docs" / "research" / "_century_leverage_results.json"
CALIBRATION = ROOT / "docs" / "research" / "_century_leverage_calibration.json"
FRENCH_DIR = ROOT / "data" / "external" / "french"
SHILLER_DIR = ROOT / "data" / "external" / "shiller"
A23_DIR = ROOT / "data" / "external" / "a23_inputs"
ROLL_CLOSE: pd.DataFrame | None = None
ROLL_DIST: pd.DataFrame | None = None


def save(path: Path, data: dict) -> None:
    """Atomically checkpoint the complete result so interruption is resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(calibration: dict) -> dict:
    """Record all frozen input identities and the r2 endpoint."""
    frozen = {}
    for directory, names in (
        (FRENCH_DIR, [FRENCH_FILE]),
        (SHILLER_DIR, [SHILLER_FILE]),
        (A23_DIR, ["SPY.csv", "SSO.csv", "BIL.csv", "DTB3.csv"]),
    ):
        for name in names:
            _, record = checked_file(directory, name)
            frozen[name] = {
                "sha256": record["sha256"],
                "url": record["url"],
                "downloaded_utc": record["downloaded_utc"],
            }
    return {
        "synthetic": True,
        "full_window": ["1926-07-01", END],
        "registered_windows": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "excluded_months": {
            "months": ["2026-07", "2026-08"],
            "reason": "Shiller S&P dividend field is blank; ruling r2 forbids filling",
        },
        "inputs": frozen,
        "calibration_sha256": sha256(CALIBRATION.read_bytes()).hexdigest(),
        "calibration_window": calibration["registered_window"],
        "spreads": {name: row["fitted_spread_annual"] for name, row in calibration["fees"].items()},
        "market_proxy": "Ken French value-weighted all-US market, not S&P 500 or SPY",
        "dividend_model": (
            "Shiller monthly S&P Composite trailing annual D/P divided by 12; "
            "monthly distribution at last trading close"
        ),
        "tax_model": (
            "A23 federal lot book, current rates applied hypothetically throughout; "
            "no $3,000 ordinary-loss offset; SSO zero distributions; "
            "approximate wash handling"
        ),
        "distribution_basis": "S&P yield on all-US French market is an approximation",
        "known_market_closures": {
            "1933-03-03_to_1933-03-15": (
                "https://www.federalreservehistory.org/essays/emergency-banking-act-of-1933"
            ),
            "2001-09-10_to_2001-09-17": "https://www.sec.gov/investor/mktrecovery.shtml",
        },
    }


def sso_distribution_check() -> dict:
    """Summarize real frozen SSO cash distributions without using them in the synthetic leg."""
    path, _ = checked_file(A23_DIR, "SSO.csv")
    frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
    frame = frame.loc[:END]
    events = frame[frame["Dividends"] > 0]
    yield_on_prior_close = frame["Dividends"] / frame["Close"].shift(1)
    return {
        "window": [str(frame.index[0].date()), str(frame.index[-1].date())],
        "distribution_days": len(events),
        "first_distribution": str(events.index[0].date()) if len(events) else None,
        "last_distribution": str(events.index[-1].date()) if len(events) else None,
        "sum_per_share": float(events["Dividends"].sum()),
        "sum_event_yields_on_prior_close": float(yield_on_prior_close.loc[events.index].sum()),
    }


def rolling_pairs(index: pd.DatetimeIndex, years: int) -> list[tuple[str, str]]:
    """Pair each month-end session with the same calendar month years later."""
    months = index.to_period("M")
    last = index.to_series().groupby(months).tail(1)
    by_month = {date.to_period("M"): date for date in last.index}
    pairs = []
    for month, start in by_month.items():
        end = by_month.get(month + years * 12)
        if end is not None:
            pairs.append((str(start.date()), str(end.date())))
    return pairs


def _init_rolling(close: pd.DataFrame, dist: pd.DataFrame) -> None:
    """Make the fixed primary legs available to each rolling-window worker."""
    global ROLL_CLOSE, ROLL_DIST
    ROLL_CLOSE, ROLL_DIST = close, dist


def _rolling_one(pair: tuple[str, str]) -> dict:
    """Evaluate one matched pair of dates with fresh lots in every arm."""
    assert ROLL_CLOSE is not None and ROLL_DIST is not None
    start, end = pair
    by_mix = {}
    for mix in MIXES:
        result = simulate(
            ROLL_CLOSE,
            ROLL_DIST,
            start,
            end,
            mix,
            *BRACKETS["24/15"],
            False,
            track_nav=False,
        )
        by_mix[str(int(mix * 100))] = result["after_tax_cagr"]
    base = by_mix["0"]
    return {
        "window": [start, end],
        "cagr": by_mix,
        "difference_vs_0": {mix: value - base for mix, value in by_mix.items()},
    }


def complete_rolling(data: dict, close: pd.DataFrame, dist: pd.DataFrame, out: Path) -> None:
    """Checkpoint every ten fresh-lot month-end windows for both registered tenors."""
    with ProcessPoolExecutor(
        max_workers=3,
        mp_context=get_context("fork"),
        initializer=_init_rolling,
        initargs=(close, dist),
    ) as pool:
        for years in (10, 20):
            key = str(years)
            pairs = rolling_pairs(close.index, years)
            rows = data["rolling_windows"].setdefault(key, [])
            for i in range(len(rows), len(pairs), 10):
                rows.extend(pool.map(_rolling_one, pairs[i : i + 10]))
                save(out, data)
            data["rolling_summary"][key] = {}
            for mix in MIXES:
                label = str(int(mix * 100))
                worst = min(rows, key=lambda row: row["difference_vs_0"][label])
                data["rolling_summary"][key][label] = {
                    "windows": len(rows),
                    "worst_difference_vs_0": worst["difference_vs_0"][label],
                    "worst_window": worst["window"],
                    "share_trailing_0": sum(row["difference_vs_0"][label] < 0 for row in rows)
                    / len(rows),
                }
            save(out, data)


def main() -> None:
    """Run all fixed scenarios and restart safely from an existing checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT, help="result and checkpoint JSON path")
    args = parser.parse_args()
    calibration = json.loads(CALIBRATION.read_text())
    identity = provenance(calibration)
    daily = load_french_daily(FRENCH_DIR)
    yields = load_shiller_yield(SHILLER_DIR)
    data = (
        json.loads(args.out.read_text())
        if args.out.exists()
        else {
            "provenance": identity,
            "calibration": calibration,
            "sso_real_distribution_check": sso_distribution_check(),
            "scenarios": {},
            "rolling_windows": {},
            "rolling_summary": {},
        }
    )
    if data["provenance"] != identity or data["calibration"] != calibration:
        raise ValueError("Checkpoint inputs or calibration changed")
    for fee_name, fee_row in calibration["fees"].items():
        for distribution in ("shiller", "zero"):
            close, dist = build_legs(
                daily,
                yields,
                fee_row["expense_ratio_annual"],
                fee_row["fitted_spread_annual"],
                distribution == "zero",
            )
            for bracket, (st, lt) in BRACKETS.items():
                for wash in (False, True):
                    scenario = f"{fee_name}|{bracket}|wash={wash}|{distribution}"
                    rows = data["scenarios"].setdefault(scenario, {})
                    for mix in MIXES:
                        label = str(int(mix * 100))
                        arm = rows.setdefault(label, {})
                        for name, (start, end) in WINDOWS.items():
                            if name not in arm:
                                arm[name] = simulate(close, dist, start, end, mix, st, lt, wash)
                                save(args.out, data)
    primary = data["scenarios"]["0.84%|24/15|wash=False|shiller"]
    data["recommendation_rule"] = recommendation_rule(primary)
    save(args.out, data)
    fee = calibration["fees"]["0.84%"]
    close, dist = build_legs(
        daily, yields, fee["expense_ratio_annual"], fee["fitted_spread_annual"]
    )
    complete_rolling(data, close, dist, args.out)
    data["complete"] = True
    save(args.out, data)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
