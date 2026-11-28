"""Investigate whether daily return disagreements reflect a one-day shift."""

import argparse
import datetime as dt
import json
import logging

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.study import bootstrap, study_dir

bootstrap(__file__)

from fc_pilot import fetch_bars  # noqa: E402


def compare(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Compare paired daily returns and show the largest absolute differences."""
    pairs = left.merge(right, on="symbol", suffixes=("_left", "_right"))
    pairs = pairs.dropna(subset=["previous_left", "close_left", "previous_right", "close_right"])
    pairs = pairs[(pairs.previous_left > 0) & (pairs.previous_right > 0)].copy()
    if pairs.empty:
        return {"matched": 0, "within_half_point": None, "largest": []}
    pairs["return_left"] = pairs.close_left / pairs.previous_left - 1
    pairs["return_right"] = pairs.close_right / pairs.previous_right - 1
    pairs["difference"] = (pairs.return_left - pairs.return_right).abs()
    return {
        "matched": len(pairs),
        "within_half_point": float((pairs.difference <= 0.005).mean()),
        "largest": pairs.nlargest(10, "difference").to_dict("records"),
    }


def pair(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Join two close-price frames by symbol."""
    return (
        old[["symbol", "close"]]
        .merge(new[["symbol", "close"]], on="symbol", suffixes=("_previous", ""))
        .rename(columns={"close_previous": "previous"})
    )


def main() -> None:
    """Fetch the needed prices and save the read-only comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    load_dotenv("/home/flynn/ggTrader/.env")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    root = study_dir(__file__)
    path = root / "_source_pilot.json"
    report = json.loads(path.read_text())
    with create_engine(get_db_connection_string()).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        ref = pd.read_sql(
            text(
                "SELECT timestamp, symbol, close FROM "
                "market.yahoo_daily_frozen_20261126 WHERE venue='yfinance' AND interval='1d' "
                "AND timestamp >= '2026-11-02' AND timestamp < '2026-11-06'"
            ),
            conn,
        )
    ref["day"] = pd.to_datetime(ref.timestamp).dt.strftime("%Y-%m-%d")
    days = {
        day: ref[ref.day == day][["symbol", "close"]]
        for day in ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
    }
    requested = sorted(set(days["2026-11-03"].symbol) & set(days["2026-11-04"].symbol))
    frames = {}
    raw = {
        "reference": ref.assign(timestamp=ref.timestamp.astype(str)).to_dict("records"),
        "source": {},
        "requested": requested,
        "provenance": {
            "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reference": "market.yahoo_daily_frozen_20261126",
            "source_days": ["2026-11-03", "2026-11-04"],
            "reference_days": ["2026-11-02", "2026-11-05"],
        },
    }
    raw_path = root / "_alignment_raw.json"
    for source in ["alpaca", "yahoo"]:
        raw["source"][source] = {}
        for day in ["2026-11-03", "2026-11-04"]:
            frame = fetch_bars(requested, dt.date.fromisoformat(day), source)
            raw["source"][source][day] = json.loads(frame.to_json(orient="records"))
            raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        frames[source] = pair(
            pd.DataFrame(raw["source"][source]["2026-11-03"]),
            pd.DataFrame(raw["source"][source]["2026-11-04"]),
        )
    result = {
        "dates": ["2026-11-03", "2026-11-04"],
        "tolerance": 0.005,
        "raw_file": "_alignment_raw.json",
        "alpaca_vs_yahoo": compare(frames["alpaca"], frames["yahoo"]),
        "frozen_counts_by_day": {day: len(frame) for day, frame in days.items()},
        "shifted": {},
    }
    for source in frames:
        result["shifted"][source] = {}
        for label, start, end in [
            ("earlier", "2026-11-02", "2026-11-03"),
            ("same", "2026-11-03", "2026-11-04"),
            ("later", "2026-11-04", "2026-11-05"),
        ]:
            result["shifted"][source][label] = compare(frames[source], pair(days[start], days[end]))
    report["alignment"] = result
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
