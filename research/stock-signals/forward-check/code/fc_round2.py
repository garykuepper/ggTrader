"""Retry the forward source pilot with paced, resumable requests."""

import argparse
import datetime as dt
import json
import logging
import time

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.study import bootstrap, study_dir

bootstrap(__file__)

from fc_pilot import (  # noqa: E402
    classify_security,
    coverage,
    fetch_bars,
    fetch_listing,
    price_agreement,
    return_agreement,
)

# Each date precedes the removal of the named ticker; ticker reuse is not resolved here.
DELISTED = {
    "2017-06-01": ["XLNX", "AET", "MON", "CELG", "CTL", "ETFC", "TIF", "RTN", "VIAB", "DPS"],
    "2022-06-01": ["SIVB", "FRC", "BBBY", "TWTR", "ATVI", "SAVE", "SBNY", "PXD", "NLSN", "CERN"],
}


def main() -> None:
    """Run missing parts and checkpoint results after each sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv("/home/flynn/ggTrader/.env")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    path = study_dir(__file__) / "_source_pilot.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if not args.resume:
        report["round2"] = {}
    result = report.setdefault("round2", {})
    result.setdefault(
        "provenance",
        {
            "reference_days": ["2026-11-03", "2026-11-04"],
            "live_day": "2026-11-25",
            "return_tolerance": 0.005,
            "listing_fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "delisted_samples": DELISTED,
        },
    )

    def save() -> None:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    listing = fetch_listing()
    symbols = listing.symbol.tolist()
    result["listing_count"] = len(symbols)
    engine = create_engine(get_db_connection_string())
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        refs = pd.read_sql(
            text(
                "SELECT timestamp, symbol, close FROM "
                "market.yahoo_daily_frozen_20261126 WHERE venue='yfinance' "
                "AND interval='1d' AND timestamp >= '2026-11-03' "
                "AND timestamp < '2026-11-05'"
            ),
            conn,
        )
    old = refs[pd.to_datetime(refs.timestamp).dt.date == dt.date(2026, 11, 3)]
    new = refs[pd.to_datetime(refs.timestamp).dt.date == dt.date(2026, 11, 4)]
    for source in ["alpaca", "yahoo"]:
        src = result.setdefault(source, {})
        for label in ["comparison", "live"]:
            if label in src:
                continue
            try:
                if label == "live":
                    progress = src.setdefault("live_batches", {})
                    for offset in range(0, len(symbols), 200):
                        batch = symbols[offset : offset + 200]
                        key = str(offset)
                        if key in progress:
                            if progress[key]["requested"] != batch:
                                raise ValueError("Listing changed since the checkpoint")
                            continue
                        part = fetch_bars(batch, dt.date(2026, 11, 25), source)
                        progress[key] = {"requested": batch, "returned": part.symbol.tolist()}
                        save()
                    returned = [s for batch in progress.values() for s in batch["returned"]]
                    bars = pd.DataFrame({"symbol": returned})
                    entry = coverage(listing, bars)
                    if source == "yahoo":
                        missing = listing[~listing.symbol.isin(bars.symbol)].copy()
                        counts = missing.apply(
                            lambda r: classify_security(r["symbol"], r["name"]), axis=1
                        )
                        entry["missing_by_description"] = counts.value_counts().to_dict()
                        entry["classification_note"] = (
                            "Explicit descriptions only; other is ambiguous."
                        )
                else:
                    common = sorted(set(old.symbol) & set(new.symbol))
                    first = fetch_bars(common, dt.date(2026, 11, 3), source)
                    second = fetch_bars(common, dt.date(2026, 11, 4), source)
                    n, agreement = return_agreement(first, second, old, new, 0.005)
                    entry = {
                        "requested": len(common),
                        "matched_pairs": n,
                        "return_within_0_005": agreement,
                        "level_matched": len(set(second.symbol) & set(new.symbol)),
                        "level_within_1_percent": price_agreement(second, new, 0.01)
                        if not second.empty
                        else None,
                    }
                src[label] = entry
            except Exception as exc:
                src[label + "_failure"] = {
                    "error": type(exc).__name__,
                    "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                }
            save()
        samples = src.setdefault("delisted", {})
        for day, tickers in DELISTED.items():
            if day in samples:
                continue
            try:
                bars = fetch_bars(tickers, dt.date.fromisoformat(day), source)
                samples[day] = {"requested": tickers, "returned": bars.symbol.tolist()}
            except Exception as exc:
                samples[day + "_failure"] = {
                    "error": type(exc).__name__,
                    "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                }
            save()
    shares = result.setdefault("yahoo_shares", {})
    common = listing[
        listing.apply(lambda r: classify_security(r["symbol"], r["name"]) == "common", axis=1)
    ]
    for symbol in common.symbol.head(20):
        if symbol in shares:
            continue
        for attempt in range(5):
            time.sleep(1)
            try:
                values = yf.Ticker(symbol).get_shares_full(start="2026-01-01")
                shares[symbol] = {"returned": values is not None and not values.empty}
                break
            except Exception as exc:
                if type(exc).__name__ != "YFRateLimitError" or attempt == 4:
                    shares[symbol] = {"error": type(exc).__name__}
                    break
                time.sleep(10 * 2**attempt)
        save()


if __name__ == "__main__":
    main()
