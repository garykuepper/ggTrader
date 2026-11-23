#!/usr/bin/env python3
"""Compare three dead-stock recovery assumptions for one combined-data strategy."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Parse the strategy, data release and evaluation dates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expect-membership-sha", default=None)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--release-id", type=int, default=None)
    parser.add_argument("--result-dir", type=Path, default=Path("results/kaggle_dead_stock"))
    return parser


def main() -> None:
    """Run the three assumptions, then save a single comparison."""
    args = build_parser().parse_args()
    import json
    import tempfile

    import pandas as pd

    from ggTrader.data.membership import source_from_args
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo.cli_support import run_vbt_wfo
    from ggTrader.utils.io import write_json

    membership = source_from_args(args)
    release_id = args.release_id
    scenarios = []
    with tempfile.TemporaryDirectory() as scratch:
        for loss in (0.0, 0.5, 1.0):
            path = Path(scratch) / f"loss-{loss}.json"
            result = run_vbt_wfo(
                args.strategy,
                args.universe,
                LabConfig(),
                pd.Timestamp(args.eval_start, tz="UTC"),
                pd.Timestamp(args.eval_end, tz="UTC"),
                market="equity",
                base_config=dict(STOCK_BASE_CONFIG),
                source="combined",
                release_id=release_id,
                dead_stock_loss=loss,
                membership=membership,
                result_json=str(path),
            )
            if isinstance(result, str):
                raise SystemExit(result)
            saved = json.loads(path.read_text())
            release = saved["data_release"]
            release_id = release["release_id"]
            diagnostics = result.diagnostics
            scenarios.append(
                {
                    "loss": loss,
                    "oos_equity_end": float(result.oos_equity.iloc[-1])
                    if not result.oos_equity.empty
                    else None,
                    "oos_metrics": diagnostics["oos_metrics"],
                    "spy_metrics": diagnostics["spy_metrics"],
                    "dead_stock_rule": diagnostics["dead_stock_rule"],
                    "data_release": release,
                    "bars_sha256": saved["provenance"]["tape_sha256"],
                }
            )
    if len({item["bars_sha256"] for item in scenarios}) != 1:
        raise RuntimeError("bars changed between runs; no comparison was saved")
    out = args.result_dir / "dead_stock_sensitivity.json"
    write_json(
        out,
        {
            "strategy": args.strategy,
            "universe": args.universe,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "scenarios": scenarios,
        },
    )
    print("Dead-stock results are optimistic about what holders recover.")
    print("Loss  | Final equity | CAGR % | Sharpe | Max drawdown %")
    for item in scenarios:
        metrics = item["oos_metrics"]
        print(
            f"{item['loss']:4.1f}  | {item['oos_equity_end']} | {metrics['cagr_pct']:.2f} | "
            f"{metrics['sharpe']:.2f} | {metrics['max_drawdown_pct']:.2f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
