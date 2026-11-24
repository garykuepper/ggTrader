"""Driver for the dead-stock first test: one window and one universe per run.

``--window dev`` writes ``results/dead_stock_first_test/dev-<universe>.json`` (or ``--out``).
``--window holdout`` writes only ``holdout-<universe>.json`` there. Before it reads anything it
claims the holdout with ``holdout-<universe>.claim`` (exclusive create) and refuses to run when
the claim or the result exists: the holdout is looked at once; only the owner deletes a claim.
Both windows are pinned in ``dead_stock_first_test.WINDOWS``; there is no way to pass dates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ggTrader.utils.binding import binding, git_sha
from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap, study_dir

ROOT = bootstrap(__file__)

import bars_reader  # noqa: E402  (found on sys.path through bootstrap)
import dead_stock_first_test as study  # noqa: E402
import universe_frames  # noqa: E402

BRIEF = "2026-11-23-dead-stock-first-test-brief.md"
DEFAULT_RESULTS_DIR = ROOT / study.RESULTS_FOLDER
SPY = "SPY"


def build_parser() -> argparse.ArgumentParser:
    """The command line: the window is a name, never a pair of dates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(study.WINDOWS), required=True)
    parser.add_argument("--universe", choices=["A", "B"], required=True)
    parser.add_argument("--release-id", type=int, required=True, help="published release, pinned")
    parser.add_argument("--crsp-load", type=int, default=1, help="raw CRSP load for eligibility")
    parser.add_argument("--out", type=Path, default=None, help="dev only: results JSON to write")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run only the false-death audit (metadata, no prices, never claims the holdout)",
    )
    return parser


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Read the release spans, the candidate closes and SPY from the read-only database.

    Two kinds of span rows are kept apart. ``release_spans`` is one row per release column
    (symbol, permno, kind, first and last date): eligibility and the one-company-one-column
    check read it. ``span_rows`` is what the close reader returns for the columns it read:
    the series that END inside the window (symbol, last_date, has_successor, total_loss), which
    the replay reads to find dead stocks and series that continue in a later CRSP spell.
    """
    from ggTrader.data.bars import connect_readonly

    start, end = study.WINDOWS[args.window]
    read_start = (pd.Timestamp(start) - pd.Timedelta(days=study.WARMUP_CALENDAR_DAYS)).strftime(
        "%Y-%m-%d"
    )
    # Eligibility looks at the previous month-end, so read fields from one month before.
    fields_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).replace(day=1)
    conn = connect_readonly()
    try:
        release_spans = bars_reader.read_release_spans(conn, args.release_id)
        fields = universe_frames.month_end_fields(
            conn, args.crsp_load, fields_start.strftime("%Y-%m-%d"), end
        )
    finally:
        conn.close()
    days = pd.bdate_range(start, end)
    ever = universe_frames.eligibility(
        fields, release_spans, universe=args.universe, daily_index=days
    )
    candidates = sorted(ever.columns[ever.to_numpy().any(axis=0)])
    del ever
    provenance: dict[str, Any] = {}
    close, span_rows = bars_reader.read_close_batched(
        candidates, read_start, end, release_id=args.release_id, provenance_out=provenance
    )
    spy_provenance: dict[str, Any] = {}
    spy_frame, _ = bars_reader.read_close_batched(
        [SPY], read_start, end, release_id=args.release_id, provenance_out=spy_provenance
    )
    if SPY not in spy_frame.columns:
        raise SystemExit(f"SPY has no bars in release {args.release_id} for {read_start}..{end}")
    # Same rule on the real trading days. Only the candidate columns are needed: a winner among
    # one company's columns is always a candidate, so dropping the others changes nothing.
    wanted = release_spans[release_spans["symbol"].isin(candidates)]
    window_days = _naive(close.index)
    window_days = window_days[window_days >= pd.Timestamp(start)]
    eligible = universe_frames.eligibility(
        fields, wanted, universe=args.universe, daily_index=window_days
    ).reindex(columns=close.columns, fill_value=False)
    conn = connect_readonly()
    try:
        audit = study.audit_false_deaths(
            conn,
            wanted,
            crsp_load=args.crsp_load,
            window_start=start,
            window_end=end,
            crsp_end=study.CRSP_END,
        )
    finally:
        conn.close()
    return {
        "false_death_audit": audit,
        "close": close,
        "spy_close": spy_frame[SPY],
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": release_spans,
        "data_provenance": {
            "candidates": provenance,
            "spy": spy_provenance,
            "n_candidate_columns": len(candidates),
        },
    }


def read_audit(args: argparse.Namespace) -> dict[str, Any]:
    """The false-death audit over every release span (metadata only; no prices are read)."""
    from ggTrader.data.bars import connect_readonly

    start, end = study.WINDOWS[args.window]
    conn = connect_readonly()
    try:
        release_spans = bars_reader.read_release_spans(conn, args.release_id)
        return study.audit_false_deaths(
            conn,
            release_spans,
            crsp_load=args.crsp_load,
            window_start=start,
            window_end=end,
            crsp_end=study.CRSP_END,
        )
    finally:
        conn.close()


def read_drift(args: argparse.Namespace) -> list[str]:
    """Drifted linked Yahoo columns of the pinned release (metadata only), before any claim."""
    from ggTrader.data.bars import connect_readonly

    conn = connect_readonly()
    try:
        return bars_reader.read_linked_drift(conn, args.release_id)
    finally:
        conn.close()


def drift_in(inputs: dict[str, Any]) -> list[str]:
    """Symbols the close reads reported as drifted (empty when none)."""
    found: set[str] = set()
    for part in (inputs.get("data_provenance") or {}).values():
        if isinstance(part, dict) and part.get("drift"):
            found.update(part.get("drift_symbols") or ["(drift reported without symbols)"])
    return sorted(found)


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index


def brief_commit() -> str | None:
    """The commit that last touched the frozen brief (``None`` outside git)."""
    path = study_dir(__file__) / BRIEF
    try:
        done = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    results_dir: Path | None = None,
    git_dirty: Callable[[], bool] | None = None,
    drift_preflight: Callable[[argparse.Namespace], list[str]] | None = None,
    audit_loader: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> int:
    """Run one window and universe and write its results JSON.

    Everything except ``argv`` is for tests only: there is no command-line or environment way
    to move the results folder, so the holdout guard cannot be bypassed from outside. With a
    ``loader`` given, the git check and the drift preflight are off unless passed too.
    """
    command = list(sys.argv if argv is None else ["run.py", *argv])
    args = build_parser().parse_args(argv)
    folder = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    start, end = study.WINDOWS[args.window]
    provenance_base = {
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
        "brief_commit": brief_commit(),
        "git_head": git_sha(ROOT),
        "window": {"start": start, "end": end},
        "window_name": args.window,
        "universe": args.universe,
        "crsp_end": study.CRSP_END,
        "command": command,
    }
    if args.audit_only:  # metadata only: never takes the claim, never reads prices
        audit = (audit_loader or read_audit)(args)
        out = study.audit_file(args.window, args.universe, folder)
        write_json(
            out,
            {
                "study": "dead-stock-first-test",
                "false_death_audit": audit,
                "provenance": provenance_base,
            },
        )
        print(f"wrote {out}")
        return 0
    target = study.resolve_output(args.window, args.universe, args.out, folder)
    study.refuse_existing_trades_file(args.window, args.universe, folder)
    if args.window == "holdout":
        dirty = git_dirty() if git_dirty else (loader is None and study.git_tree_dirty(ROOT))
        study.check_holdout_pins(args.release_id, args.crsp_load, git_dirty=bool(dirty))
        preflight = drift_preflight or (read_drift if loader is None else (lambda _a: []))
        drifted = preflight(args)
        if drifted:
            raise SystemExit(
                "holdout refused: the live Yahoo rows no longer match the pinned release "
                f"(drift in {len(drifted)} linked columns, e.g. {', '.join(drifted[:5])}); "
                "nothing was claimed"
            )
        study.claim_holdout(args.universe, folder, git_head=git_sha(ROOT), command=command)
    inputs = (loader or read_inputs)(args)
    drifted = drift_in(inputs)
    if args.window == "holdout" and drifted:  # the claim stays: the owner decides
        raise SystemExit(
            "holdout run stopped after the read: drift reported for "
            f"{', '.join(drifted[:5])}; no result was written and the claim is kept"
        )
    sink: dict[str, list] = {}
    result = study.run_study(
        inputs["close"],
        inputs["spy_close"],
        inputs["eligible"],
        inputs["span_rows"],
        start=start,
        end=end,
        release_spans=inputs.get("release_spans"),
        trade_sink=sink,
    )
    result["false_death_audit"] = inputs.get("false_death_audit")
    payload = {
        "study": "dead-stock-first-test",
        "result": result,
        "provenance": {
            **provenance_base,
            "data": inputs.get("data_provenance"),
            "drift_symbols": drifted,
            "binding": binding(ROOT),
        },
    }
    side = study.write_trades_csv(study.trades_file(args.window, args.universe, folder), sink)
    write_json(target, payload)  # after the side file, so a result implies its trades exist
    print(f"wrote {side} and {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
