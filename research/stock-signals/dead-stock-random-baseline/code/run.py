"""Driver for the dead-stock random baseline: one window per run, random picks against the rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ggTrader.utils.binding import binding, git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import baseline_inputs  # noqa: E402  (found on sys.path through bootstrap)
import dead_stock_random_baseline as logic  # noqa: E402
import numpy as np  # noqa: E402
import random_plans  # noqa: E402
import rb_dead_stock_first_test as fp  # noqa: E402
import rb_replay  # noqa: E402
from rb_ledger_run import TaxParams, run_book, run_spy  # noqa: E402

STUDY_DIR = study_dir(__file__)
STUDY_ID = "dead-stock-random-baseline"
DEFAULT_RESULTS_DIR = ROOT / "results" / "dead_stock_random_baseline"
SAVED_FILES: dict[str, Path] = {
    "early": STUDY_DIR.parent / "dead-stock-fresh-window" / "_fresh-A.json",
    "dev": STUDY_DIR.parent / "dead-stock-first-test" / "_dev-A.json",
    "holdout": STUDY_DIR.parent / "dead-stock-first-test" / "_holdout-A.json",
}
ARMS = ("R", "M")
PIN_RELEASE_ID = 1
PIN_CRSP_LOAD = 1
DEFAULT_DRAWS = 100
DEFAULT_SANITY_DRAWS = 20
EXIT_CHECK_FAILED = 3
EXIT_REFUSED = 2


def build_parser() -> argparse.ArgumentParser:
    """Command line: a registered window, the pinned data, and how many draws."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(baseline_inputs.WINDOWS), required=True)
    parser.add_argument("--release-id", type=int, required=True, help="pinned release (1)")
    parser.add_argument("--crsp-load", type=int, required=True, help="pinned CRSP load (1)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="draws per arm")
    parser.add_argument(
        "--first-draw", type=int, default=0, help="index of the first draw (timing probes only)"
    )
    parser.add_argument(
        "--sanity-draws", type=int, default=DEFAULT_SANITY_DRAWS, help="arm R draws re-run free"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="results JSON (default: the study folder)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None, help="folder for the resumable partial file"
    )
    return parser


def _tax_params() -> TaxParams:
    """The low tax set of the headline cell (built on use, not at import)."""
    return TaxParams(*fp.TAXES[logic.TAX])


def _no_tax() -> TaxParams:
    return TaxParams(0.0, 0.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_cells(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))["result"]
    return result["strategy"], result["spy"]


class Prepared:
    """Everything the books need for one window, built once."""

    def __init__(self, inputs: dict[str, Any], start: str, end: str) -> None:
        self.start, self.end = start, end
        self.close = inputs["close"]
        self.spy_close = inputs["spy_close"]
        self.window_check = fp.validate_window_inputs(self.close, self.spy_close, start, end)
        spans, self.n_censored = fp.drop_file_end_rows(
            fp._normalise_spans(inputs["span_rows"]), end
        )
        self.spans = spans
        self.ends = rb_replay.span_end_dates(self.close, spans)
        flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
        self.successors = frozenset(s for s in self.ends.index if bool(flagged.get(s, False)))
        self.prices = {loss: rb_replay.exit_close(self.close, spans, loss) for loss in fp.LOSSES}
        self.symbols = np.array(self.close.columns)
        self.sets = random_plans.eligible_sets(
            self.close,
            inputs["eligible"],
            spans,
            lookback=fp.LOOKBACK,
            skip=fp.SKIP,
            start=start,
            end=end,
        )
        self.rule_steps = random_plans.rule_ranking_plan(self.sets, self.symbols, top_n=fp.TOP_N)
        reference = rb_replay.plan_trades(
            self.close,
            inputs["eligible"],
            spans,
            top_n=fp.TOP_N,
            lookback=fp.LOOKBACK,
            skip=fp.SKIP,
            dead_loss=1.0,
            start=start,
            end=end,
        )
        if reference != self.rule_steps:
            raise SystemExit("the rule's plan rebuilt here differs from the copied planner's")

    def book(self, steps: list, loss: float, cost: float, params: TaxParams) -> Any:
        """One finished ledger run of ``steps``."""
        return run_book(
            self.close,
            steps,
            self.ends,
            self.prices[loss],
            params=params,
            one_way_cost=cost,
            start=self.start,
            end=self.end,
            successors=self.successors,
            start_cash=fp.START_CASH,
        )

    def draw_cells(self, steps: list) -> dict[float, dict[str, Any]]:
        """The headline-cost, low-tax cell of ``steps`` at each loss reading."""
        return {
            loss: fp._cell(self.book(steps, loss, logic.COST, _tax_params()))
            for loss in logic.LOSSES
        }


def _check_first(
    prep: Prepared, saved_strategy: dict[str, Any], saved_spy: dict[str, Any]
) -> dict[str, Any]:
    """Rerun the rule's three headline cells and SPY; compare with the saved file."""
    computed = {
        fp.cell_key(loss, logic.COST, logic.TAX): cell
        for loss, cell in prep.draw_cells(prep.rule_steps).items()
    }
    spy_book = run_spy(
        prep.spy_close,
        params=_tax_params(),
        one_way_cost=logic.COST,
        start=prep.start,
        end=prep.end,
        start_cash=fp.START_CASH,
    )
    spy_cell = fp._cell(spy_book)
    computed_spy = {fp.spy_key(logic.COST, logic.TAX): spy_cell}
    diffs = logic.check_against_saved(computed, saved_strategy)
    diffs += logic.check_against_saved(computed_spy, saved_spy)
    return {
        "ok": not diffs,
        "differences": diffs,
        "tolerance_dollars": logic.HALF_CENT,
        "n_cells": len(computed) + len(computed_spy),
        "spy_cagr_after_tax": spy_cell["cagr_after_tax"],
    }


def _load_partial(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"R": [], "M": [], "sanity_dice_cagr": []}


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    saved_paths: dict[str, Path] | None = None,
    git_dirty: Callable[[], bool] | None = None,
) -> int:
    """Run one window: check first, then the draws of both arms, then write the results JSON.

    Everything except ``argv`` is for tests only. Exit codes: 0 done, 2 refused before any
    work, 3 the check against the saved numbers failed (no draw is run).
    """
    command = list(sys.argv if argv is None else ["run.py", *argv])
    args = build_parser().parse_args(argv)
    if args.release_id != PIN_RELEASE_ID or args.crsp_load != PIN_CRSP_LOAD:
        print("refused: the release and CRSP load must be the pinned 1 and 1", file=sys.stderr)
        return EXIT_REFUSED
    if args.draws < 1 or not 0 <= args.first_draw < 10_000 or args.sanity_draws < 0:
        print("refused: bad draw counts", file=sys.stderr)
        return EXIT_REFUSED
    dirty = git_dirty() if git_dirty is not None else (loader is None and fp.git_tree_dirty(ROOT))
    if dirty:
        print("refused: the tree has uncommitted changes to tracked files", file=sys.stderr)
        return EXIT_REFUSED
    out_path = args.out or STUDY_DIR / f"_{args.window}.json"
    if out_path.exists():
        print(f"refused: {out_path} already exists", file=sys.stderr)
        return EXIT_REFUSED
    saved_path = (saved_paths or SAVED_FILES)[args.window]
    saved_strategy, saved_spy = _saved_cells(saved_path)
    start, end = baseline_inputs.WINDOWS[args.window]
    inputs = (
        loader(args)
        if loader is not None
        else baseline_inputs.read_inputs(args.window, args.release_id, args.crsp_load)
    )
    drifted = baseline_inputs.drift_in(inputs)
    if drifted:
        print(f"refused: Yahoo drift in the pinned release: {drifted[:5]}", file=sys.stderr)
        return EXIT_REFUSED
    prep = Prepared(inputs, start, end)
    started = time.time()

    check = _check_first(prep, saved_strategy, saved_spy)
    print(f"check first: ok={check['ok']} ({time.time() - started:.0f}s)", file=sys.stderr)
    if not check["ok"]:
        print(json.dumps(check["differences"], indent=2), file=sys.stderr)
        return EXIT_CHECK_FAILED
    spy_cagr = check["spy_cagr_after_tax"]

    results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    partial_path = results_dir / f"{args.window}.partial.json"
    partial = _load_partial(partial_path)
    for arm in ARMS:
        have = {rec["seed"] for rec in partial[arm]}
        for draw in range(args.first_draw, args.first_draw + args.draws):
            seed = random_plans.draw_seed(args.window, arm, draw)
            if seed in have:
                continue
            t0 = time.time()
            if arm == "R":
                steps = random_plans.plan_fresh_dice(
                    prep.sets, prep.symbols, top_n=fp.TOP_N, seed=seed
                )
            else:
                steps = random_plans.plan_matched(
                    prep.sets, prep.symbols, prep.rule_steps, top_n=fp.TOP_N, seed=seed
                )
            partial[arm].append(logic.draw_record(seed, prep.draw_cells(steps)))
            write_json(partial_path, partial)
            print(f"arm {arm} draw {draw}: {time.time() - t0:.0f}s", file=sys.stderr)

    # Sanity: the dice against an equal hold of the whole eligible list, loss 0, no cost, no tax.
    free = {
        "all_eligible": prep.book(
            random_plans.plan_all_eligible(prep.sets, prep.symbols), 0.0, 0.0, _no_tax()
        ).cagr_after_tax
    }
    for draw in range(args.first_draw, args.first_draw + min(args.sanity_draws, args.draws)):
        if draw - args.first_draw < len(partial["sanity_dice_cagr"]):
            continue
        steps = random_plans.plan_fresh_dice(
            prep.sets,
            prep.symbols,
            top_n=fp.TOP_N,
            seed=random_plans.draw_seed(args.window, "R", draw),
        )
        partial["sanity_dice_cagr"].append(prep.book(steps, 0.0, 0.0, _no_tax()).cagr_after_tax)
        write_json(partial_path, partial)
    sanity = logic.sanity_block(free["all_eligible"], partial["sanity_dice_cagr"])

    rule_cells = {
        loss: saved_strategy[fp.cell_key(loss, logic.COST, logic.TAX)] for loss in logic.LOSSES
    }
    summary = {arm: logic.summarise_arm(partial[arm], spy_cagr, rule_cells) for arm in ARMS}
    payload = {
        "study": STUDY_ID,
        "window": args.window,
        "window_dates": [start, end],
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "git_head": git_sha(ROOT),
        "binding": binding(ROOT),
        "command": command,
        "first_draw": args.first_draw,
        "n_draws": args.draws,
        "saved_file": str(saved_path),
        "saved_file_sha256": _sha256(saved_path),
        "check_first": check,
        "sanity": sanity,
        "spy": {"cagr_after_tax": spy_cagr, "cell": fp.spy_key(logic.COST, logic.TAX)},
        "summary": summary,
        "draws": {arm: partial[arm] for arm in ARMS},
        "sanity_dice_cagr": partial["sanity_dice_cagr"],
        "window_check": prep.window_check,
        "n_rebalance_steps": len(prep.sets),
        "data_provenance": inputs.get("data_provenance"),
        "seconds": time.time() - started,
    }
    write_json(out_path, payload)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
