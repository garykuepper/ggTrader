"""Reproduce the first test's already-used 2020–2024 arm A cells."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ggTrader.utils.binding import git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import ed_dead_stock_first_test as study  # noqa: E402
import ed_inputs  # noqa: E402
from ed_comparison import after_tax_growth, comparison_returns  # noqa: E402
from ed_ledger_run import TaxParams, run_spy  # noqa: E402
from ed_simulation import (
    comparison_metrics,
    loss_monotone,
    period_metrics,
    simulate_arm,
    validate_calendar,
)  # noqa: E402

START = "2020-01-02"
END = "2024-12-31"
EARLY_START = "1969-02-03"
EARLY_END = "1998-07-31"


def git_dirty(root: Path) -> bool:
    """Check tracked modifications while allowing untracked result artifacts."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def rule_hash() -> str:
    """Hash the study implementation and frozen-copy record."""
    digest = hashlib.sha256()
    code = study_dir(__file__) / "code"
    for path in sorted([*code.glob("ed_*.py"), code / "frozen_copies.json"]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_linked_drift() -> list[str]:
    """Check pinned Yahoo links without reading the study window."""
    from ggTrader.data.bars import connect_readonly

    conn = connect_readonly()
    try:
        return ed_inputs.bars_reader.read_linked_drift(conn, 1)
    finally:
        conn.close()


def preflight() -> None:
    """Refuse a dirty checkout or a drifted release before claiming the window."""
    if git_dirty(ROOT):
        raise SystemExit("dirty git tree; commit or remove changes before scoring")
    drift = read_linked_drift()
    if drift:
        raise SystemExit(f"linked Yahoo drift detected in {len(drift)} symbols")


def run_early(args: argparse.Namespace) -> None:
    """Read the pinned early window and write its six cells with provenance."""
    if (args.eval_start, args.eval_end) != (EARLY_START, EARLY_END):
        raise SystemExit("early window dates must match the registration")
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing result")
    current_commit = git_sha(ROOT)
    if not current_commit:
        raise SystemExit("early run requires a recorded git commit")
    if args.regression_proof is None:
        raise SystemExit("the matching 2020–2024 regression output is required")
    proof = json.loads(args.regression_proof.read_text())
    saved = json.loads(
        (study_dir(__file__).parent / "dead-stock-first-test" / "_holdout-A.json").read_text()
    )
    keys = [
        study.cell_key(loss, cost, tax)
        for loss in study.LOSSES
        for cost in study.COSTS
        for tax in study.TAXES
    ]
    expected = {key: saved["result"]["strategy"][key] for key in keys}
    if (proof.get("start"), proof.get("end"), proof.get("matches"), proof.get("cell")) != (
        START,
        END,
        True,
        expected,
    ):
        raise SystemExit("regression output does not exactly match the saved cells")
    if (proof.get("code_commit"), proof.get("rule_hash"), proof.get("clean")) != (
        current_commit,
        rule_hash(),
        True,
    ):
        raise SystemExit("regression proof does not match the current clean code")
    preflight()
    claim = study_dir(__file__) / f"_early-{args.arm}.claim"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "git_head": current_commit,
                        "command": shlex.join(sys.argv),
                        "output": str(args.out),
                    }
                )
                + "\n"
            )
    except FileExistsError as exc:
        raise SystemExit(f"early arm {args.arm} already claimed: {claim}") from exc
    checkpoint = {
        "status": "loading",
        "eval_start": EARLY_START,
        "eval_end": EARLY_END,
        "arm": args.arm,
        "release_id": 1,
        "code_commit": current_commit,
        "rule_hash": rule_hash(),
    }
    write_json(args.out, checkpoint)
    inputs = ed_inputs.read_early_inputs(EARLY_START, EARLY_END, 1, 1)
    if inputs["provenance"].get("drift"):
        raise SystemExit("linked Yahoo drift found during input read; claim retained")
    comparison = comparison_returns(EARLY_START, EARLY_END)
    validate_calendar(inputs["close"].index, comparison, EARLY_START, EARLY_END)
    cells = simulate_arm(args.arm, inputs, EARLY_START, EARLY_END)
    checkpoint.update(
        {
            "status": "metrics_pending",
            "cells": cells,
            "data_provenance": inputs["provenance"],
            "comparison_daily_curve": [
                [str(day.date()), float(value)]
                for day, value in (1.0 + comparison).cumprod().items()
            ],
        }
    )
    write_json(args.out, checkpoint)
    for cell in cells:
        account = study.pd.Series(
            [row[1] for row in cell["account_value_daily"]],
            index=study.pd.to_datetime([row[0] for row in cell["account_value_daily"]]),
        )
        cell["periods"] = {
            "early": period_metrics(account, EARLY_START, "1993-01-29"),
            "later": period_metrics(account, "1993-02-01", EARLY_END),
            "combined": period_metrics(account, EARLY_START, EARLY_END),
        }
    first = comparison.loc[:"1993-01-29"]
    second = comparison.loc["1993-02-01":]
    result = {
        "status": "complete",
        "comparison_daily_curve": checkpoint["comparison_daily_curve"],
        "eval_start": EARLY_START,
        "eval_end": EARLY_END,
        "release_id": 1,
        "crsp_load": 1,
        "rule_hash": rule_hash(),
        "code_commit": current_commit,
        "arm": args.arm,
        "cells": cells,
        "loss_monotone_violations": loss_monotone(cells),
        "data_provenance": inputs["provenance"],
        "comparison": {
            "early": {
                "start": EARLY_START,
                "end": "1993-01-29",
                "label": "CRSP market return (stand-in)",
                "growth_after_tax": after_tax_growth(first, 0.15),
                "metrics": comparison_metrics(comparison, EARLY_START, "1993-01-29"),
            },
            "later": {
                "start": "1993-02-01",
                "end": EARLY_END,
                "label": "SPY",
                "growth_after_tax": after_tax_growth(second, 0.15),
                "metrics": comparison_metrics(comparison, "1993-02-01", EARLY_END),
            },
            "combined_growth_after_tax": after_tax_growth(comparison, 0.15),
            "combined_metrics": comparison_metrics(comparison, EARLY_START, EARLY_END),
        },
    }
    write_json(args.out, result)


def main() -> None:
    """Run the authorized regression and compare its saved cells exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["holdout", "early"], required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regression-proof", type=Path, default=None)
    args = parser.parse_args()
    if args.window == "early":
        run_early(args)
        return
    if (args.arm, args.eval_start, args.eval_end) != ("A", START, END):
        parser.error("holdout regression requires arm A and the registered 2020–2024 dates")
    if git_dirty(ROOT):
        raise SystemExit("dirty git tree; commit or remove changes before regression")
    regression_commit = git_sha(ROOT)
    regression_hash = rule_hash()
    if not regression_commit:
        raise SystemExit("regression requires a recorded git commit")
    inputs = ed_inputs.read_early_inputs(START, END, 1, 1)
    read_start = (
        study.pd.Timestamp(START) - study.pd.Timedelta(days=study.WARMUP_CALENDAR_DAYS)
    ).strftime("%Y-%m-%d")
    spy, _ = ed_inputs.bars_reader.read_close_batched(["SPY"], read_start, END, release_id=1)
    cells = simulate_arm(
        "A",
        inputs,
        START,
        END,
        costs=study.COSTS,
        taxes=tuple((name, *rates) for name, rates in study.TAXES.items()),
    )
    saved = json.loads(
        (study_dir(__file__).parent / "dead-stock-first-test" / "_holdout-A.json").read_text()
    )["result"]["strategy"]
    spy_cagr = {}
    for cost in study.COSTS:
        for tax, rates in study.TAXES.items():
            book = run_spy(
                spy["SPY"],
                params=TaxParams(*rates),
                one_way_cost=cost,
                start=START,
                end=END,
                start_cash=study.START_CASH,
            )
            spy_cagr[(cost, tax)] = book.cagr_after_tax
    actual = {}
    for cell in cells:
        key = study.cell_key(cell["dead_stock_loss"], cell["cost"], cell["tax"])
        result = {field: cell[field] for field in saved[key] if field in cell}
        result["loss"] = cell["dead_stock_loss"]
        result["one_way_cost"] = cell["cost"]
        result["gap_cagr_after_tax_pp"] = (
            cell["cagr_after_tax"] - spy_cagr[(cell["cost"], cell["tax"])]
        ) * 100.0
        actual[key] = result
    expected = {key: saved[key] for key in saved}
    if git_dirty(ROOT) or git_sha(ROOT) != regression_commit or rule_hash() != regression_hash:
        raise SystemExit("code or checkout changed during regression; no proof written")
    write_json(
        args.out,
        {
            "start": START,
            "end": END,
            "cell": actual,
            "matches": actual == expected,
            "code_commit": regression_commit,
            "rule_hash": regression_hash,
            "clean": True,
        },
    )
    if actual != expected:
        raise SystemExit("Regression did not match the saved cells exactly")
    print("Regression matched exactly")


if __name__ == "__main__":
    main()
