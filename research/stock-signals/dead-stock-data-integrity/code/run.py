"""Driver for the dead-stock-data-integrity study: Phase 1 scans CRSP rows and counts flags."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ggTrader.data.bars import connect_readonly
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import dead_stock_data_integrity as study  # noqa: E402  (found on sys.path through bootstrap)
from dead_stock_data_integrity import permno_batches  # noqa: E402,F401  (public helper)


def build_parser() -> argparse.ArgumentParser:
    """The command line: the evaluation window is pinned, never "today"."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["integrity"], required=True, help="which phase to run")
    parser.add_argument("--release-id", type=int, required=True, help="published release id")
    parser.add_argument("--crsp-load", type=int, required=True, help="CRSP load id")
    parser.add_argument("--mcgill-load", type=int, required=True, help="McGill load id")
    parser.add_argument("--eval-start", default="2005-01-03", help="first evaluation date")
    parser.add_argument("--eval-end", default=study.CRSP_END, help="last scanned date")
    parser.add_argument("--out", type=Path, default=study_dir(__file__), help="output folder")
    parser.add_argument(
        "--trades-dir",
        type=Path,
        default=None,
        help="first test's trade files (default: results/dead_stock_first_test)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check the arguments, then run the integrity phase on a read-only connection."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if dt.date.fromisoformat(args.eval_end) > dt.date.fromisoformat(study.CRSP_END):
        parser.error(f"--eval-end {args.eval_end} is after the CRSP end {study.CRSP_END}")
    existing = [n for n in (study.OUTPUT_JSON, study.OUTPUT_ROSTER) if (args.out / n).exists()]
    if existing and not args.force:
        parser.error(f"{', '.join(existing)} already exist in {args.out}; use --force to replace")
    conn = connect_readonly()
    try:
        payload = study.run_integrity(
            conn,
            release_id=args.release_id,
            crsp_load=args.crsp_load,
            mcgill_load=args.mcgill_load,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            out_dir=args.out,
            repo_root=ROOT,
            trades_dir=args.trades_dir,
            run_date=dt.date.today().isoformat(),
        )
    finally:
        conn.close()
    scan = payload["provenance"]
    print(f"scanned {scan['permnos_scanned']} permnos, {scan['rows_scanned']} rows")
    for key in ("duplicates", "bad_price", "spikes", "gaps", "stale", "shares"):
        print(
            f"{key}: {payload[key]['rows_flagged']} rows, {payload[key]['stocks_flagged']} stocks"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
