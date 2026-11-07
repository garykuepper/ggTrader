"""Peak RSS of the full ensemble grid (48 combinations x 605 symbols) through engine.run_signals.

Fix-up of the P4 chunker finding: ``chunker_experiment.py`` showed the engine kept every
finished chunk alive; ``engine/portfolio.py`` now releases each chunk after copying out what
``SimResult`` needs. This driver measures the result for one budget per process (a fresh
process each time, so ``ru_maxrss`` is that run's peak) and writes JSON.

Run each budget under a cgroup, for example::

    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \\
        .venv/bin/python chunker_memory.py --budget-gib 2 --out p2b_memory_2gib.json

Pass ``--reference p2b_memory_5gib.json`` to a later run to compare its equity hash with the
reference's (chunking must not change a number).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ggTrader.data.bars import load_bars
from ggTrader.engine import portfolio as pf
from ggTrader.engine.provenance import provenance
from ggTrader.lab.data import STOCK_BASE_CONFIG
from ggTrader.lab.strategies.v2.contract import tile_columns
from ggTrader.lab.strategies.v2.ensemble import Ensemble
from ggTrader.lab.universe import pit_mask

GIB = 1024**3
UNIVERSE_SPAN = ("2021-01-31", "2026-04-30")
BAR_WINDOW = ("2019-03-01", "2026-10-30")
#: Benchmark and sleeve ETFs: with the index members they make the 605 loaded symbols.
EXTRA_SYMBOLS = ["SPY", "TLT", "GLD", "PDBC", "BIL"]


def rss_mb() -> float:
    """Current resident set size in MB."""
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * 4096 / 1e6


def peak_mb() -> float:
    """Peak resident set size of this process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def trades_hash(trades: Any) -> str:
    """Hash of the numeric trade columns and the group labels (symbols may be tuples)."""
    numeric = ["col", "size", "entry_idx", "entry_price", "exit_idx", "exit_price", "pnl"]
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(trades[numeric].to_numpy(dtype="float64")).tobytes())
    digest.update("\x00".join(map(str, trades["group"])).encode())
    return digest.hexdigest()


def disable_release() -> None:
    """Re-create the pre-fix behaviour (keep every chunk alive) for a before/after control.

    Rewrites the source of ``_run_chunks``: drops the ``del``/``_release_memory`` lines and
    keeps the equity columns as views of the chunk frame, which is what the old code did.
    """
    source = inspect.getsource(pf._run_chunks)
    release = "        del pf, value, records\n        _release_memory()\n"
    copy = ".to_numpy().copy()"
    assert release in source and copy in source, "engine/portfolio.py changed; update this control"
    patched = source.replace(release, "").replace(copy, ".to_numpy()")
    namespace = dict(vars(pf))
    exec(patched, namespace)  # noqa: S102  (our own source, rewritten for the control arm)
    pf._run_chunks = namespace["_run_chunks"]


def measure(budget_gib: float) -> dict[str, Any]:
    """Build the full grid, run it at ``budget_gib``, return the measurements."""
    bars = load_bars(
        universe="sp500",
        universe_span=UNIVERSE_SPAN,
        extra_symbols=EXTRA_SYMBOLS,
        start=BAR_WINDOW[0],
        end=BAR_WINDOW[1],
    )
    out = Ensemble().build(bars)
    mask = pit_mask(bars.close, min_history_bars=400, on_stale="ignore")
    columns = out.entries.columns
    close_t, mask_t = tile_columns(bars.close, columns), tile_columns(mask, columns)
    rss_before = rss_mb()
    peak_before = peak_mb()
    started = time.time()
    result = pf.run_signals(
        out.entries,
        out.exits,
        close_t,
        dict(STOCK_BASE_CONFIG),
        groups=out.group_labels(),
        mask=mask_t,
        max_bytes=int(budget_gib * GIB),
    )
    seconds = time.time() - started
    equity = np.ascontiguousarray(result.value.to_numpy())
    n_bars, n_cols = len(bars.close), len(columns)
    return {
        "budget_gib": budget_gib,
        "symbols": bars.close.shape[1],
        "bars": n_bars,
        "grid_combos": len(result.value.columns),
        "signal_columns": n_cols,
        "n_chunks": result.n_chunks,
        "estimate_single_call_gib": round(pf.estimate_peak_bytes(n_bars, n_cols, 1) / GIB, 3),
        "rss_mb_before_run": round(rss_before),
        "peak_rss_mb_before_run": round(peak_before),
        "peak_rss_mb": round(peak_mb()),
        "peak_over_baseline_mb": round(peak_mb() - rss_before),
        "run_seconds": round(seconds, 1),
        "trades": int(len(result.trades)),
        "finite_equity": bool(np.isfinite(equity).all()),
        "equity_sha256": hashlib.sha256(equity.tobytes()).hexdigest(),
        "trades_sha256": trades_hash(result.trades),
        "tape_sha256": bars.provenance["snapshot"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    """Measure one budget and write ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget-gib", type=float, required=True, help="chunking budget in GiB")
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--no-release", action="store_true", help="control arm: pre-fix behaviour (chunks kept)"
    )
    parser.add_argument("--reference", type=Path, help="earlier JSON whose equity hash must match")
    args = parser.parse_args(argv)

    if args.no_release:
        disable_release()
    data = measure(args.budget_gib)
    data["released_chunks"] = not args.no_release
    if args.reference is not None:
        ref = json.loads(args.reference.read_text())["measurement"]
        data["equity_equals_reference"] = ref["equity_sha256"] == data["equity_sha256"]
        data["trades_equal_reference"] = ref["trades_sha256"] == data["trades_sha256"]
        data["reference"] = args.reference.name
    payload = {"measurement": data, "provenance": provenance(tape_sha256=data["tape_sha256"])}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
