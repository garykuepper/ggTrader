"""P7 evidence: the rebuilt paper signal runner against P0's live golden, 120 sessions.

Runs the real ``ggTrader.paper.signal_runner.generate_signals`` (engine self-check, production
sanity bounds, the v2 ensemble with the live defaults) once per golden session on the frozen
tape, with the live loader replaced by a slice of the hashed Parquet snapshot, and compares
``buys``, ``sells``, ``last_exit``, ``as_of`` and ``universe_size`` with the legacy golden.
It also records the margins of every sanity gate (so the bounds can be tuned from data).

Reads the frozen snapshot only: no network, no write to ``ohlcv``. The S&P 500 list comes from
``--membership`` (default ``fixture:sp500_v1_2026-06-02``; ``db:1`` reads the same list from the
database) and its vintage is recorded in the JSON's ``provenance`` block.

Usage:
    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \\
        .venv/bin/python scripts/p7_live_golden.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ggTrader.data.bars import load_bars
from ggTrader.data.membership import ORACLE_SPEC, add_membership_argument, source_from_args
from ggTrader.engine.provenance import provenance as build_provenance
from ggTrader.paper import signal_runner
from ggTrader.paper.gates import SanityConfig
from ggTrader.utils.results import write_result_json

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "research/programme/vectorbt-engine/legacy-references/live_golden_legacy.json"
# Scratch by default; the committed copy is research/programme/vectorbt-engine/p7-evidence/
# p7_live_golden.json and is only rewritten by passing it as --out.
OUT = Path(tempfile.gettempdir()) / "ggtrader_evidence" / "p7_live_golden.json"
EVAL_SPAN = ("2021-01-31", "2026-04-30")
BAR_WINDOW = ("2019-03-01", "2026-10-30")
EXTRAS = ("SPY", "TLT", "GLD", "PDBC", "BIL")
COMPARED = ("buys", "sells", "last_exit", "as_of", "universe_size")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the 120-session comparison and write the evidence JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--out", type=Path, default=OUT)
    add_membership_argument(parser, default=ORACLE_SPEC)
    args = parser.parse_args(argv)
    membership = source_from_args(args)

    golden = json.loads(args.golden.read_text())
    bars = load_bars(
        universe="sp500",
        universe_span=EVAL_SPAN,
        extra_symbols=EXTRAS,
        start=BAR_WINDOW[0],
        end=BAR_WINDOW[1],
        membership=membership,
    )
    frame = bars.legacy_frame()
    symbols_of = frame.columns.get_level_values(0)

    def fake_fetch(symbols, start, end):
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        rows = (frame.index >= lo) & (frame.index <= hi)
        return frame.loc[rows, symbols_of.isin(symbols)]

    rows, mismatches = [], []
    t0 = time.perf_counter()
    with patch.object(signal_runner, "fetch_stock_ohlcv", side_effect=fake_fetch):
        for rec in golden:
            now = datetime.fromisoformat(rec["session"]).replace(
                hour=20, minute=0, tzinfo=timezone.utc
            )
            got = signal_runner.generate_signals(now=now, membership=membership)
            diff = {k: {"new": got[k], "legacy": rec[k]} for k in COMPARED if got[k] != rec[k]}
            rows.append(
                {
                    "session": rec["session"],
                    "as_of": got["as_of"],
                    "universe_size": got["universe_size"],
                    "last_bar_coverage": got["bars"]["last_bar_coverage"],
                    "eligible": got["bars"]["eligible"],
                    "buys": len(got["buys"]),
                    "sells": len(got["sells"]),
                    "trimmed_partial_bar": got["bars"]["trimmed_partial_bar"],
                    "equal": not diff,
                }
            )
            if diff:
                mismatches.append({"session": rec["session"], **diff})
    seconds = round(time.perf_counter() - t0, 1)

    cfg = SanityConfig()
    result = {
        "phase": "P7",
        "what": "rebuilt paper signal runner vs the P0 live golden",
        "sessions": len(golden),
        "first": golden[0]["session"],
        "last": golden[-1]["session"],
        "sessions_equal": sum(r["equal"] for r in rows),
        "sessions_different": len(mismatches),
        "fields_compared": list(COMPARED),
        "total_buys_new": sum(r["buys"] for r in rows),
        "total_sells_new": sum(r["sells"] for r in rows),
        "total_buys_golden": sum(len(r["buys"]) for r in golden),
        "total_sells_golden": sum(len(r["sells"]) for r in golden),
        "gate_margins": {
            "config": cfg.__dict__,
            "min_universe_seen": min(r["universe_size"] for r in rows),
            "min_last_bar_coverage_seen": min(r["last_bar_coverage"] for r in rows),
            "max_buys_seen": max(r["buys"] for r in rows),
            "max_sells_seen": max(r["sells"] for r in rows),
            "sessions_with_a_trimmed_bar": sum(r["trimmed_partial_bar"] for r in rows),
        },
        "seconds": seconds,
        "mismatches": mismatches[:25],
        "per_session": rows,
    }
    write_result_json(
        args.out,
        result,
        provenance=build_provenance(
            membership=membership.provenance(pd.Timestamp(EVAL_SPAN[1]).date())
        ),
    )
    log(
        f"live golden: {result['sessions_equal']}/{len(golden)} sessions equal in {seconds}s; "
        f"margins {result['gate_margins']}"
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
