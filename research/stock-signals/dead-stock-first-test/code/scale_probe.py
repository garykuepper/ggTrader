"""One-measurement scale probe: can the dead-stock test read every column it needs?

Each run reads one number of eligible columns in one way (``one_call`` or ``batched``) and
writes seconds, column and date counts and the process's peak memory, so the peak belongs to
that read alone. The columns are those with at least one eligible day in the window (the
warm-up year included), taken in symbol order. Run it through ``scripts/run_capped.sh``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ggTrader.utils.io import write_json
from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

import bars_reader  # noqa: E402  (found on sys.path through bootstrap)
import universe_frames  # noqa: E402

PEAK_LIMIT_MB = 2500.0
DECISION_RULE = (
    "ok means the process peak stayed at or under 2,500 MB. If the all-column read does not fit, "
    "the driver reads in date-bounded batches of columns and keeps only close."
)
DEFAULT_OUT = ROOT / "results" / "dead_stock_first_test" / "scale_probe.json"
STATUS_PATH = "/proc/self/status"


def read_status_text() -> str:
    """The text of ``/proc/self/status`` (empty where it does not exist)."""
    try:
        return Path(STATUS_PATH).read_text()
    except OSError:
        return ""


class PeakMemoryUnavailable(RuntimeError):
    """The process's peak memory could not be read, so no pass can be claimed."""


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (``VmHWM``).

    Raises:
        PeakMemoryUnavailable: The status file is unreadable or has no ``VmHWM`` line.
    """
    for line in read_status_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise PeakMemoryUnavailable(f"no VmHWM line readable from {STATUS_PATH}")


def candidate_columns(
    fields: pd.DataFrame, spans: pd.DataFrame, universe: str, start: str, end: str
) -> list[str]:
    """Release columns with at least one eligible day in the window, sorted by symbol."""
    days = pd.bdate_range(start, end)
    eligible = universe_frames.eligibility(fields, spans, universe=universe, daily_index=days)
    return sorted(eligible.columns[eligible.to_numpy().any(axis=0)])


def first_n(columns: list[str], n: int | str) -> list[str]:
    """The first ``n`` columns in symbol order, or every column for ``"all"``."""
    ordered = sorted(columns)
    return ordered if n == "all" else ordered[: int(n)]


def build_record(
    *,
    n_columns: int,
    n_dates: int,
    seconds: float,
    peak_mb: float | None,
    release_id: int,
    provenance: dict,
    measurement_error: str | None = None,
    **extra: object,
) -> dict:
    """The JSON record of one measurement; a missing peak is never ``ok``."""
    measured = peak_mb is not None
    return {
        "n_columns": n_columns,
        "n_dates": n_dates,
        "seconds": round(seconds, 2),
        "peak_memory_mb": round(peak_mb, 1) if measured else None,
        "ok": bool(measured and peak_mb <= PEAK_LIMIT_MB),
        "measurement_error": measurement_error,
        "peak_limit_mb": PEAK_LIMIT_MB,
        "release_id": release_id,
        "provenance": provenance,
        "decision_rule": DECISION_RULE,
        **extra,
    }


def safe_peak_mb() -> tuple[float | None, str | None]:
    """The peak in MB and no error, or ``None`` and the reason it could not be read."""
    try:
        return peak_memory_mb(), None
    except PeakMemoryUnavailable as error:
        return None, str(error)


def _n_value(text: str) -> int | str:
    """Argparse type for ``--n``: a positive integer or ``all``."""
    if text == "all":
        return "all"
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--n must be a positive integer or 'all': {text!r}"
        ) from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"--n must be at least 1: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The command line: one read per run; both window ends are pinned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=_n_value, required=True, help="columns to read, or 'all'")
    parser.add_argument("--mode", choices=["one_call", "batched"], required=True)
    parser.add_argument("--release-id", type=int, default=1)
    parser.add_argument("--crsp-load", type=int, default=1, help="raw CRSP load for eligibility")
    parser.add_argument("--universe", choices=["A", "B"], default="A")
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--batch", type=int, default=bars_reader.DEFAULT_BATCH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write")
    return parser


def pick_columns(args: argparse.Namespace) -> list[str]:
    """Read the release spans and month-end fields, and pick the columns to measure."""
    from ggTrader.data.bars import connect_readonly

    # Eligibility looks at the previous month-end, so read one month before the window.
    fields_start = (pd.Timestamp(args.start) - pd.DateOffset(months=1)).replace(day=1)
    conn = connect_readonly()
    try:
        spans = bars_reader.read_release_spans(conn, args.release_id)
        fields = universe_frames.month_end_fields(
            conn, args.crsp_load, fields_start.strftime("%Y-%m-%d"), args.end
        )
    finally:
        conn.close()
    columns = candidate_columns(fields, spans, args.universe, args.start, args.end)
    return first_n(columns, args.n)


def main(argv: list[str] | None = None) -> int:
    """Pick the columns, read them once, and write the measurement."""
    args = build_parser().parse_args(argv)
    meta = {
        "mode": args.mode,
        "n_requested": args.n,
        "universe": args.universe,
        "start": args.start,
        "end": args.end,
        "release_id": args.release_id,
        "crsp_load": args.crsp_load,
    }
    # First thing: a run killed or failed anywhere below can never leave an old result looking
    # current.
    write_json(args.out, {"status": "started", **meta})
    try:
        columns = pick_columns(args)
        setup_peak_mb, setup_error = safe_peak_mb()  # the picking step's own peak
        write_json(args.out, {"status": "started", "n_columns": len(columns), **meta})
        provenance: dict = {}
        batch = len(columns) if args.mode == "one_call" else args.batch
        began = time.perf_counter()
        close, _ = bars_reader.read_close_batched(
            columns,
            args.start,
            args.end,
            release_id=args.release_id,
            batch=max(batch, 1),
            provenance_out=provenance,
        )
        seconds = time.perf_counter() - began
        peak_mb, error = safe_peak_mb()
        extra = {k: v for k, v in meta.items() if k != "release_id"}
        record = build_record(
            n_columns=close.shape[1],
            n_dates=close.shape[0],
            seconds=seconds,
            peak_mb=peak_mb,
            release_id=args.release_id,
            provenance=provenance,
            measurement_error=error,
            status="finished",
            n_candidates_read=len(columns),
            setup_peak_memory_mb=None if setup_peak_mb is None else round(setup_peak_mb, 1),
            setup_measurement_error=setup_error,
            **extra,
        )
    except BaseException as error:
        write_json(
            args.out,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                **meta,
            },
        )
        raise
    write_json(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
