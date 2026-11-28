"""Compare historical SIP bars with the saved source pilot prices."""

import argparse
import datetime as dt
import json

import pandas as pd
from dotenv import load_dotenv

from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

from fc_alignment import compare, pair  # noqa: E402
from fc_pilot import coverage, fetch_bars  # noqa: E402


def main() -> None:
    """Resume the read-only SIP comparison and checkpoint each listing batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    root = study_dir(__file__)
    path = root / "_source_pilot.json"
    report = json.loads(path.read_text())
    raw = json.loads((root / "_alignment_raw.json").read_text())
    checkpoint_path = root / "_source_pilot_checkpoints.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Local pilot checkpoint is missing; restore it before this run")
    checkpoint = json.loads(checkpoint_path.read_text())
    previous = checkpoint.get("round2", {}).get("alpaca", {}).get("live_batches")
    if previous is None:
        raise ValueError("The saved Alpaca listing checkpoint is unavailable")
    if not args.resume:
        report["sip_followup"] = {}
        checkpoint["sip_followup"] = {}
    result = report.setdefault("sip_followup", {})
    result.setdefault(
        "provenance",
        {
            "feed": "sip",
            "adjustment": "raw",
            "prior_alpaca_feed": "iex",
            "prices_source": "_alignment_raw.json",
            "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    def save() -> None:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, allow_nan=False) + "\n")

    ref = pd.DataFrame(raw["reference"])
    ref["day"] = pd.to_datetime(ref.timestamp).dt.strftime("%Y-%m-%d")
    reference = pair(ref[ref.day == "2026-11-03"], ref[ref.day == "2026-11-04"])
    yahoo = pair(
        pd.DataFrame(raw["source"]["yahoo"]["2026-11-03"]),
        pd.DataFrame(raw["source"]["yahoo"]["2026-11-04"]),
    )
    if "yahoo_vs_frozen" not in result:
        result["yahoo_vs_frozen"] = compare(yahoo, reference)
        result["actions"] = {
            "status": "unavailable",
            "reason": "Saved Yahoo bars contain no "
            "dividend or split information; no new Yahoo requests were made.",
        }
        save()
    prices = checkpoint.setdefault("sip_followup", {}).setdefault("comparison_prices", {})
    for day in ["2026-11-03", "2026-11-04"]:
        if day not in prices:
            bars = fetch_bars(raw["requested"], dt.date.fromisoformat(day), "alpaca", feed="sip")
            prices[day] = json.loads(bars.to_json(orient="records"))
            save()
    sip = pair(pd.DataFrame(prices["2026-11-03"]), pd.DataFrame(prices["2026-11-04"]))
    result["sip_vs_yahoo"] = compare(sip, yahoo)
    result["sip_vs_frozen"] = compare(sip, reference)
    save()
    samples = result.setdefault("delisted", {})
    for day, entry in report["round2"]["alpaca"]["delisted"].items():
        if day not in samples:
            bars = fetch_bars(entry["requested"], dt.date.fromisoformat(day), "alpaca", feed="sip")
            samples[day] = {"requested": entry["requested"], "returned": bars.symbol.tolist()}
            save()
    progress = checkpoint.setdefault("sip_followup", {}).setdefault("listing_batches", {})
    for key, batch in previous.items():
        if key in progress:
            continue
        bars = fetch_bars(batch["requested"], dt.date(2026, 11, 25), "alpaca", feed="sip")
        progress[key] = {"requested": batch["requested"], "returned": bars.symbol.tolist()}
        save()
    symbols = [x for batch in previous.values() for x in batch["requested"]]
    returned = [x for batch in progress.values() for x in batch["returned"]]
    result["listing_coverage"] = coverage(
        pd.DataFrame({"symbol": symbols}), pd.DataFrame({"symbol": returned})
    )
    save()


if __name__ == "__main__":
    main()
