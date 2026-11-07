"""P5 equity-level parity: the single-pass simulation against the legacy per-window simulations.

The walk-forward correctness proof that does not depend on any statistic definition. For one
strategy and window it checks, for every distinct window end of the folds (each ``train_end``,
each ``test_end`` and the live end):

* the equity curve of EVERY grid combination from the new single full-history simulation
  (``wfo_v2.scorers.SignalScorer``) equals, bit for bit, the equity of the legacy sweep
  (``lab.wfo._sweep_fold``) run on data truncated at that end, from the first loaded bar;
* the legacy per-simulation trade count (``pf.trades.count()``, the whole truncated simulation)
  equals the new count of entries before that end (so OD12's per-window count, entries inside
  the window, is the same quantity minus the entries before the window).

The anchor (first bar to ``train_end``), the train sweep and the live sweep are slices of these
equities, so equal equity at each end proves all of them.

Two phases so the legacy and the new simulation never share a process (memory): ``--phase new``
saves the new equities, ``--phase legacy`` re-simulates each end with the legacy code, compares
and checkpoints after every end (resumable).

Example:
    .venv/bin/python scripts/p5_equity_parity.py --phase new --run core17_ensemble_sp500
    .venv/bin/python scripts/p5_equity_parity.py --phase legacy --run core17_ensemble_sp500
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
OUT = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "p5-evidence"
GIB = 1024**3


def _trim() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def window_ends(index: pd.DatetimeIndex, eval_start: str, eval_end: str) -> list[pd.Timestamp]:
    """Distinct window ends (calendar) of the folds plus the live end, ascending."""
    from ggTrader.lab.wfo_v2.folds import generate_folds

    es, ee = pd.Timestamp(eval_start, tz="UTC"), pd.Timestamp(eval_end, tz="UTC")
    ends = {ee}
    for fold in generate_folds(es, ee):
        ends.update([fold.train_end, fold.test_end])
    return sorted(ends)


def load_inputs(spec: dict[str, Any]) -> tuple[Any, pd.DataFrame, Any, Any]:
    """Bars, PIT mask, config and the strategy class for a run spec (as the validation does)."""
    import p5_wfo_validation as v

    from ggTrader.data.bars import legacy_exclusive_end, load_bars
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo_v2.inputs import subset_bars, universe_mask

    cfg = LabConfig()
    es, ee = pd.Timestamp(v.EVAL_START), pd.Timestamp(spec["end"])
    warmup = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    bars = load_bars(
        universe="sp500",
        universe_span=(es, ee),
        extra_symbols=["SPY"],
        start=str((es - pd.Timedelta(days=warmup)).date()),
        end=legacy_exclusive_end(spec["end"]),
    )
    bars = subset_bars(bars, [s for s in bars.symbols if s != "SPY"])
    mask = universe_mask(bars.close, "sp500", min_history_bars=cfg.min_history_bars)
    return bars, mask, cfg, v


def phase_new(name: str, spec: dict[str, Any], max_gib: float) -> int:
    """Simulate once with the engine and save every combination's equity and entry bars."""
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.lab.wfo_v2 import inputs
    from ggTrader.lab.wfo_v2.scorers import SignalScorer

    bars, mask, _cfg, _v = load_inputs(spec)
    cls = REGISTRY_V2[spec["strategy"]]
    combos = cls.grid.combos()
    names = [inputs.combo_name(spec["strategy"], c) for c in combos]
    scorer = SignalScorer(
        cls(),
        bars,
        cls.grid,
        names,
        combos,
        mask,
        dict(STOCK_BASE_CONFIG),
        int(max_gib * GIB),
    )
    t0 = time.time()
    scorer.prepare([])
    seconds = time.time() - t0
    np.savez_compressed(
        OUT / f"_p5_equity_new_{name}.npz",
        value=scorer._val,
        index_ns=scorer.index.tz_convert(None).asi8,
        names=np.array(names),
        trades_group=scorer._trades["group"].to_numpy(),
        trades_entry=scorer._trades["entry_idx"].to_numpy(),
    )
    print(f"new simulation {seconds:.1f}s chunks={scorer.n_chunks} -> saved", flush=True)
    return 0


def phase_legacy(name: str, spec: dict[str, Any]) -> int:
    """Re-simulate each distinct end with the legacy sweep and compare with the saved equities."""
    from ggTrader.data.bars import to_legacy_frame
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import _sweep_fold
    from ggTrader.utils.results import write_result_json

    bars, mask, cfg, v = load_inputs(spec)
    saved = np.load(OUT / f"_p5_equity_new_{name}.npz")
    value, names = saved["value"], list(saved["names"])
    index = bars.close.index
    new_counts_entries = saved["trades_entry"]
    new_counts_group = saved["trades_group"]
    legacy_cls = STRATEGY_REGISTRY[spec["strategy"]]
    grid = build_grid(legacy_cls)
    if len(grid) != len(names):
        raise RuntimeError(f"grid sizes differ: legacy {len(grid)} vs new {len(names)}")
    frame = to_legacy_frame(bars)
    instance = legacy_cls(cfg)
    ends = window_ends(index, v.EVAL_START, spec["end"])
    target = OUT / f"_p5_equity_parity_{name}.json"
    record: dict[str, Any] = (
        json.loads(target.read_text()) if target.exists() else {"run": name, "ends": {}}
    )
    base = dict(STOCK_BASE_CONFIG)
    for end in ends:
        key = str(end.date())
        if key in record["ends"]:
            print(f"[skip] {key}", flush=True)
            continue
        t0 = time.time()
        results, all_eq = _sweep_fold(
            spec["strategy"], instance, frame, frame.index[0], end, base, grid, mask
        )
        n_bars = len(next(iter(all_eq.values())))
        by_combo: dict[str, Any] = {}
        identical = 0
        worst = 0.0
        trades_equal = 0
        legacy_trades = {r["combo"]: r["n_trades"] for r in results}
        for g, combo in enumerate(names):
            legacy_eq = all_eq[combo].to_numpy()
            new_eq = value[:n_bars, g]
            diff = float(np.max(np.abs(legacy_eq - new_eq))) if n_bars else 0.0
            same = bool(np.array_equal(legacy_eq, new_eq))
            identical += same
            worst = max(worst, diff)
            new_n = int(np.sum((new_counts_group == g) & (new_counts_entries < n_bars)))
            trades_equal += int(new_n == legacy_trades.get(combo))
            if not same or new_n != legacy_trades.get(combo):
                by_combo[combo] = {
                    "max_abs_diff": diff,
                    "new_trades": new_n,
                    "legacy_trades": legacy_trades.get(combo),
                }
        record["ends"][key] = {
            "bars": n_bars,
            "last_bar": str(index[n_bars - 1].date()),
            "combos": len(names),
            "bit_identical_combos": identical,
            "max_abs_diff_all_combos": worst,
            "trade_counts_equal_combos": trades_equal,
            "differences": by_combo,
            "seconds": round(time.time() - t0, 1),
        }
        del results, all_eq
        _trim()
        write_result_json(target, {k: v_ for k, v_ in record.items()})
        print(
            f"[{key}] bars={n_bars} bit-identical {identical}/{len(names)} "
            f"max|d|={worst:.3g} trades-equal {trades_equal}/{len(names)} "
            f"{record['ends'][key]['seconds']}s",
            flush=True,
        )
    ends_rec = record["ends"].values()
    record["summary"] = {
        "ends": len(record["ends"]),
        "all_combos_bit_identical": all(e["bit_identical_combos"] == e["combos"] for e in ends_rec),
        "all_trade_counts_equal": all(
            e["trade_counts_equal_combos"] == e["combos"] for e in ends_rec
        ),
        "max_abs_diff": max(e["max_abs_diff_all_combos"] for e in ends_rec),
    }
    write_result_json(target, record)
    print(json.dumps(record["summary"]), flush=True)
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description="P5 equity-level parity, legacy versus single pass.")
    p.add_argument("--phase", choices=("new", "legacy"), required=True)
    p.add_argument("--run", default="core17_ensemble_sp500")
    p.add_argument("--max-gib", type=float, default=3.0)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run one phase."""
    args = parse_args(argv)
    import p5_wfo_validation as v

    spec = v.ALL_RUNS[args.run]
    if args.phase == "new":
        return phase_new(args.run, spec, args.max_gib)
    return phase_legacy(args.run, spec)


if __name__ == "__main__":
    raise SystemExit(main())
