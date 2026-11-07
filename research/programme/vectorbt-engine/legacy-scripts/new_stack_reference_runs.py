"""P1 17-fold parity runs: legacy walk-forward code on the NEW stack (vectorbt-engine programme).

Re-runs the P0 reference runs (``core17_ensemble_sp500`` and its DSR-threshold-0 variant) with
the P1 source tree on the new interpreter (pandas 3, numpy 2.5, numba, vectorbt 1.2.0 + Rust),
reusing ``run_wfo_case`` of ``scripts/legacy_reference_runs.py`` unchanged, so the two sides differ
only by the stack and the documented P1 source edits. The vectorbt engine is forced through
``vbt.settings["engine"]`` and proven by counting engine resolutions and Rust kernel calls.

Reads only the frozen Parquet tape; database and network access are blocked. One JSON per run is
written into ``--out-dir`` (a run whose JSON already has ``"ok": true`` is skipped unless
``--force``).

Example:
    PYTHONPATH=<src copy> .venv/bin/python scripts/new_stack_reference_runs.py \\
        --src <src copy> --engine rust --runs core17_ensemble_sp500
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "p1-parity" / "runs17fold"
DEFAULT_RUNS = "core17_ensemble_sp500,core17_ensemble_sp500_dsr0"


def load_base() -> Any:
    """Import scripts/legacy_reference_runs.py by path (the P0 runner)."""
    spec = importlib.util.spec_from_file_location(
        "legacy_reference_runs", REPO_ROOT / "scripts" / "legacy_reference_runs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_reference_runs"] = mod
    spec.loader.exec_module(mod)
    return mod


def install_engine_counters() -> dict[str, collections.Counter]:
    """Count engine resolutions and Rust kernel calls (proof of which engine ran)."""
    import vectorbt as vbt  # noqa: PLC0415, TID251
    import vectorbt._engine as eng  # noqa: PLC0415, TID251

    counts: dict[str, collections.Counter] = {
        "resolve": collections.Counter(),
        "rust_kernels": collections.Counter(),
    }
    orig = eng.resolve_engine

    def counting_resolve(*args: Any, **kwargs: Any) -> str:
        out = orig(*args, **kwargs)
        counts["resolve"][out] += 1
        return out

    for name, mod in list(sys.modules.items()):
        if mod is not None and name.startswith("vectorbt") and hasattr(mod, "resolve_engine"):
            if getattr(mod, "resolve_engine") is orig:
                mod.resolve_engine = counting_resolve  # type: ignore[attr-defined]

    try:
        import vectorbt_rust  # noqa: PLC0415, TID251
    except ImportError:
        return counts
    for sub in (
        "portfolio",
        "records",
        "generic",
        "signals",
        "returns",
        "indicators",
        "labels",
        "base",
        "utils",
    ):
        mod = getattr(vectorbt_rust, sub, None)
        if mod is None:
            continue
        for attr in dir(mod):
            fn = getattr(mod, attr)
            if attr.endswith("_rs") and callable(fn):

                def make(f: Any, key: str) -> Any:
                    def wrapped(*a: Any, **k: Any) -> Any:
                        counts["rust_kernels"][key] += 1
                        return f(*a, **k)

                    return wrapped

                try:
                    setattr(mod, attr, make(fn, f"{sub}.{attr}"))
                except (AttributeError, TypeError):
                    counts["rust_kernels"]["__unpatchable__"] += 1
    del vbt
    return counts


def sha_array(values: Any) -> str:
    """sha256 of a float64 array's raw bytes."""
    import numpy as np  # noqa: PLC0415

    return hashlib.sha256(np.ascontiguousarray(values, dtype="float64").tobytes()).hexdigest()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="P1 17-fold parity runs on the new stack (resumable).")
    p.add_argument(
        "--src", required=True, help="source root containing ggTrader/ (a snapshot copy)"
    )
    p.add_argument("--engine", choices=["numba", "rust"], required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS, help="comma list of run names")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT), help="parent dir; <engine>/ is appended")
    p.add_argument("--snapshot", help="tape parquet (default: newest in MANIFEST.json)")
    p.add_argument("--force", action="store_true", help="redo runs that already have ok JSON")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the selected references on the new stack with the chosen engine."""
    args = parse_args(argv)
    base = load_base()
    base.bootstrap_legacy(Path(args.src))
    base.block_io()
    import vectorbt as vbt  # noqa: PLC0415, TID251

    vbt.settings["engine"] = args.engine
    counts = install_engine_counters()
    import vectorbt._engine as eng  # noqa: PLC0415, TID251

    helpers = base.import_snapshot_helpers()
    path, entry = base.pick_snapshot(args.snapshot)
    tape = base.Tape(path, entry, helpers)
    out_dir = Path(args.out_dir) / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in [r for r in args.runs.split(",") if r]:
        target = out_dir / f"_newstack_{args.engine}_{name}.json"
        if target.exists() and not args.force and json.loads(target.read_text()).get("ok"):
            print(f"[skip] {name}: ok JSON exists", flush=True)
            continue
        for c in counts.values():
            c.clear()
        t0 = time.time()
        try:
            result = base.run_wfo_case(name, dict(base.RUN_SPECS[name]), tape, out_dir)
            import pandas as pd  # noqa: PLC0415

            eq = pd.read_csv(out_dir / f"{name}_oos_equity.csv", index_col=0)["oos_equity"]
            result["oos_equity_sha256"] = sha_array(eq.to_numpy())
        except Exception as exc:  # noqa: BLE001 - recorded in the JSON
            failures += 1
            result = base._common_header(name, dict(base.RUN_SPECS[name]), tape)
            result.update(
                ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
            )
            print(f"[FAIL] {name}: {result['error']}", flush=True)
        import ggTrader  # noqa: PLC0415

        result.update(
            engine=f"new-stack-{args.engine}",
            engine_requested=args.engine,
            vbt_engine_setting=str(vbt.settings["engine"]),
            rust_available=bool(eng.is_rust_available()),
            engine_proof={
                "resolve_engine_results": dict(counts["resolve"]),
                "rust_kernel_calls_total": int(sum(counts["rust_kernels"].values())),
                "rust_kernel_calls_by_kernel": dict(counts["rust_kernels"]),
            },
            ggtrader_file=str(Path(ggTrader.__file__).resolve()),
            wall_seconds_total=round(time.time() - t0, 1),
        )
        target.write_text(json.dumps(base.jsonable(result), indent=2) + "\n")
        print(
            f"[done] {name} ok={result.get('ok')} {result['wall_seconds_total']}s -> {target}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
