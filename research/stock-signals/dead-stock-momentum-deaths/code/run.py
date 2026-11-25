"""Driver for dead-stock-momentum-deaths: one window per run, check first, then measures."""

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

import dm_dead_stock_first_test as fp  # noqa: E402  (found on sys.path through bootstrap)
import dm_replay  # noqa: E402
import momentum_deaths as md  # noqa: E402
import momentum_inputs  # noqa: E402
import numpy as np  # noqa: E402

STUDY_DIR = study_dir(__file__)
STUDY_ID = "dead-stock-momentum-deaths"
SAVED_FILES: dict[str, Path] = {
    "early": STUDY_DIR.parent / "dead-stock-fresh-window" / "_fresh-A.json",
    "dev": STUDY_DIR.parent / "dead-stock-first-test" / "_dev-A.json",
    "holdout": STUDY_DIR.parent / "dead-stock-first-test" / "_holdout-A.json",
}
RANDOM_BASELINE_FILES: dict[str, Path] = {
    "early": STUDY_DIR.parent / "dead-stock-random-baseline" / "_early.json",
    "dev": STUDY_DIR.parent / "dead-stock-random-baseline" / "_dev.json",
    "holdout": STUDY_DIR.parent / "dead-stock-random-baseline" / "_holdout.json",
}
PIN_RELEASE_ID = 1
PIN_CRSP_LOAD = 1
EXIT_REFUSED = 2
EXIT_CHECK_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    """Command line: a registered window and the pinned data release and CRSP load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(momentum_inputs.WINDOWS), required=True)
    parser.add_argument("--release-id", type=int, required=True, help="pinned release (1)")
    parser.add_argument("--crsp-load", type=int, required=True, help="pinned CRSP load (1)")
    parser.add_argument(
        "--out", type=Path, default=None, help="results JSON (default: _<window>.json in study dir)"
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256() -> str:
    """One hash over the study's own code files and its frozen engine copies (tests excluded)."""
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((here / "frozen_copies.json").read_bytes())
    return digest.hexdigest()


def _read_held_at_death(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["result"]["sanity"]["held_at_death"])


def _read_random_baseline_median(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["summary"]["R"]["loss1"]["n_dead_exits"]["p50"])


class Prepared:
    """Window inputs normalised and validated once, ready for check-first and measures."""

    def __init__(self, inputs: dict[str, Any], start: str, end: str) -> None:
        self.start, self.end = start, end
        self.close = inputs["close"]
        self.spy_close = inputs["spy_close"]
        self.eligible = inputs["eligible"]
        self.span_rows = inputs["span_rows"]
        self.window_check = fp.validate_window_inputs(self.close, self.spy_close, start, end)
        spans, ends, successors, death_dates, n_censored = md.prepare_deaths(
            self.close, self.span_rows, end
        )
        self.spans = spans
        self.ends = ends
        self.successors = successors
        self.death_dates = death_dates
        self.n_censored = n_censored
        self.symbols = np.array(self.close.columns)
        self.sets = md.eligible_sets(
            self.close,
            self.eligible,
            self.spans,
            lookback=fp.LOOKBACK,
            skip=fp.SKIP,
            start=start,
            end=end,
        )
        self.rule_steps = md.rule_ranking_plan(self.sets, self.symbols, top_n=fp.TOP_N)


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    saved_paths: dict[str, Path] | None = None,
    random_baseline_paths: dict[str, Path] | None = None,
    git_dirty: Callable[[], bool] | None = None,
) -> int:
    """Run one window: check pins and tree, check plan against planner, compute and write JSON.

    Everything except ``argv`` is for tests only. Exit codes: 0 done, 2 refused before work,
    3 check-first plan comparison failed.
    """
    command = list(sys.argv if argv is None else ["run.py", *argv])
    args = build_parser().parse_args(argv)
    if args.release_id != PIN_RELEASE_ID or args.crsp_load != PIN_CRSP_LOAD:
        print("refused: the release and CRSP load must be the pinned 1 and 1", file=sys.stderr)
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
    rb_path = (random_baseline_paths or RANDOM_BASELINE_FILES)[args.window]
    held_at_death = _read_held_at_death(saved_path)
    rb_median = _read_random_baseline_median(rb_path)

    start, end = momentum_inputs.WINDOWS[args.window]
    started = time.time()
    inputs = (
        loader(args)
        if loader is not None
        else momentum_inputs.read_inputs(args.window, args.release_id, args.crsp_load)
    )
    drifted = momentum_inputs.drift_in(inputs)
    if drifted:
        print(f"refused: Yahoo drift in the pinned release: {drifted[:5]}", file=sys.stderr)
        return EXIT_REFUSED

    prep = Prepared(inputs, start, end)
    reference = dm_replay.plan_trades(
        prep.close,
        prep.eligible,
        prep.spans,
        top_n=fp.TOP_N,
        lookback=fp.LOOKBACK,
        skip=fp.SKIP,
        dead_loss=1.0,
        start=start,
        end=end,
    )
    if reference != prep.rule_steps:
        print(
            "check failed: rebuilt top-50 plan differs from dm_replay.plan_trades",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED

    measures = md.compute_measures(
        prep.close,
        prep.eligible,
        prep.span_rows,
        start=start,
        end=end,
        held_at_death=held_at_death,
        random_baseline_median=rb_median,
    )

    data_prov = inputs.get("data_provenance") or {}
    candidates_prov = data_prov.get("candidates") if isinstance(data_prov, dict) else None
    spy_prov = data_prov.get("spy") if isinstance(data_prov, dict) else None
    data_release = (candidates_prov or {}).get("data_release")
    bars_sha256 = (candidates_prov or {}).get("bars_sha256")

    provenance_block = {
        **binding(ROOT),
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "window": args.window,
        "window_dates": [start, end],
        "command": command,
        "code_sha256": _code_sha256(),
        "saved_file": str(saved_path),
        "saved_file_sha256": _sha256(saved_path),
        "random_baseline_file": str(rb_path),
        "random_baseline_file_sha256": _sha256(rb_path),
        "data_release": data_release,
        "bars_sha256": bars_sha256,
        "spy_bars_sha256": (spy_prov or {}).get("bars_sha256"),
        "data_provenance": data_prov,
    }

    payload = {
        "study": STUDY_ID,
        "window": args.window,
        "window_dates": [start, end],
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "git_head": git_sha(ROOT),
        "command": command,
        "provenance": provenance_block,
        "data_release": data_release,
        "bars_sha256": bars_sha256,
        "data_provenance": data_prov,
        "window_check": prep.window_check,
        "check_first": {
            "ok": True,
            "n_decision_days": len(prep.sets),
        },
        "n_file_end_censored": prep.n_censored,
        "n_decision_days": len(prep.sets),
        "measures": measures,
        "wall_seconds": time.time() - started,
    }
    write_json(out_path, payload)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
