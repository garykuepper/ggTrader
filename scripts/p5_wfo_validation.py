"""P5 validation: the vectorbt-engine walk-forward against the P0 legacy references.

Runs ``ggTrader.lab.wfo.run_wfo`` on the frozen tape with the configuration P0 used for its
legacy references (``scripts/legacy_reference_runs.py``: same windows, universe, grid, base
config, ``LabConfig()``), writes one result JSON per run (with provenance) and a comparison
JSON against the P0 reference: per-fold windows, winners, gates, anchor use, halt state, the
OOS equity (max absolute difference), OOS and SPY Sharpe/CAGR/max drawdown.

Resumable: a run whose result JSON already has ``"ok": true`` is skipped unless ``--force``.
Reads only the hashed Parquet snapshot; nothing touches the database or the network.

History: the evidence files with the suffix ``_legacydefs`` and ``_p5_diff_*`` were produced once
by an earlier version of this driver that could also score Sortino and CAGR with the legacy
forms (a diagnostic); that switch, and ``engine/legacy_defs.py``, were deleted afterwards (owner
decision 2026-11-06: vectorbt's definitions win). This version always uses vectorbt's.

Example:
    .venv/bin/python scripts/p5_wfo_validation.py --runs smoke2_ensemble_sp500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAMME = REPO_ROOT / "research" / "programme" / "vectorbt-engine"
REFERENCES = PROGRAMME / "legacy-references"
# Evidence is regenerated into a scratch folder; pass --out-dir to write under research/.
COMMITTED_OUT = PROGRAMME / "p5-evidence"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ggtrader_evidence" / "p5"
EVAL_START = "2021-01-31"
EVAL_END_17 = "2026-04-30"
EVAL_END_SMOKE = "2022-07-31"
VOTER_AND_OTHER = (
    "conviction_bb",
    "cross_asset_sleeve",
    "retail_attention",
    "xs_momentum",
    "bb_reversion",
    "rsi_reversion",
    "ema_cross",
    "macd_divergence",
    "volume_bb_reversion",
)
#: Relative tolerance for "identical within the ulp floor" (engine/stats documents 1e-12).
ULP_RTOL = 1e-12
ULP_ATOL = 1e-12
GIB = 1024**3

RUNS: dict[str, dict[str, Any]] = {
    "smoke2_ensemble_sp500": {"strategy": "ensemble", "end": EVAL_END_SMOKE, "universe": "sp500"},
    "core17_ensemble_sp500": {"strategy": "ensemble", "end": EVAL_END_17, "universe": "sp500"},
    "core17_ensemble_sp500_dsr0": {
        "strategy": "ensemble",
        "end": EVAL_END_17,
        "universe": "sp500",
        "kwargs": {"dsr_threshold": 0.0},
    },
    "ref_cross_asset_sleeve_17fold": {
        "strategy": "cross_asset_sleeve",
        "end": EVAL_END_17,
        "universe": "cross_asset",
    },
}
for _name in VOTER_AND_OTHER:
    RUNS[f"ref_{_name}"] = {
        "strategy": _name,
        "end": EVAL_END_SMOKE,
        "universe": "cross_asset" if _name == "cross_asset_sleeve" else "sp500",
    }
ALL_RUNS = dict(RUNS)


def _num(value: Any) -> float:
    return float("nan") if value is None else float(value)


def close_enough(a: Any, b: Any) -> bool:
    """Equal within the documented ulp floor; two missing values are equal."""
    x, y = _num(a), _num(b)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=ULP_RTOL, abs_tol=ULP_ATOL)


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# --------------------------------------------------------------------------- the run


def load_run_inputs(spec: dict[str, Any]) -> dict[str, Any]:
    """Bars, benchmark, PIT mask, strategy class/factory and config for a run spec (as P0)."""
    from ggTrader.data.bars import legacy_exclusive_end, load_bars
    from ggTrader.lab.strategies import REGISTRY
    from ggTrader.lab.strategies.cross_asset_sleeve import UNIVERSE as SLEEVE_UNIVERSE
    from ggTrader.lab.strategies.retail_attention import RetailAttention
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo.inputs import subset_bars, universe_mask

    cfg = LabConfig()
    es, ee = pd.Timestamp(EVAL_START), pd.Timestamp(spec["end"])
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())
    end_bar = legacy_exclusive_end(spec["end"])
    if spec["universe"] == "cross_asset":
        bars = load_bars(list(SLEEVE_UNIVERSE) + ["SPY"], start=data_start, end=end_bar)
    else:
        bars = load_bars(
            universe=spec["universe"],
            universe_span=(es, ee),
            extra_symbols=["SPY"],
            start=data_start,
            end=end_bar,
        )
    spy_close = bars.close["SPY"].dropna()
    keep = (
        list(SLEEVE_UNIVERSE)
        if spec["universe"] == "cross_asset"
        else [s for s in bars.symbols if s != "SPY"]
    )
    bars = subset_bars(bars, keep)
    if spec["universe"] == "cross_asset":
        mask = pd.DataFrame(True, index=bars.close.index, columns=bars.close.columns)
    else:
        mask = universe_mask(bars.close, spec["universe"], min_history_bars=cfg.min_history_bars)
    cls = REGISTRY[spec["strategy"]]
    factory: Any = cls
    if spec["strategy"] == "retail_attention":
        empty = pd.DataFrame(columns=["symbol", "date", "search_interest"])
        factory = lambda: RetailAttention(interest_loader=lambda *_a, **_k: empty)  # noqa: E731
    return {
        "cfg": cfg,
        "bars": bars,
        "spy_close": spy_close,
        "mask": mask,
        "cls": cls,
        "factory": factory,
        "data_start": data_start,
    }


def run_case(
    name: str,
    spec: dict[str, Any],
    max_bytes: int,
    out_dir: Path,
    stem: str,
    min_losing_days: int | None = None,
) -> dict[str, Any]:
    """One walk-forward run; returns the JSON-able record (without provenance)."""
    from ggTrader.engine import config as engine_config
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.wfo import result_payload, run_wfo

    t0 = time.time()
    guard = engine_config.MIN_LOSING_DAYS if min_losing_days is None else min_losing_days
    inp = load_run_inputs(spec)
    bars, grid = inp["bars"], inp["cls"].grid
    result = run_wfo(
        spec["strategy"],
        inp["factory"],
        inp["cfg"],
        bars,
        inp["spy_close"],
        eval_start=EVAL_START,
        eval_end=spec["end"],
        market="stock",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
        mask=inp["mask"],
        max_bytes=max_bytes,
        min_losing_days=guard,
        **spec.get("kwargs", {}),
    )
    if isinstance(result, str):
        raise RuntimeError(result)
    payload = result_payload(
        result,
        run=name,
        stem=stem,
        engine="vectorbt-1.2.0",
        spec={k: v for k, v in spec.items() if k != "kwargs"},
        run_wfo_kwargs={**spec.get("kwargs", {}), "min_losing_days": guard},
        definitions="vbt",
        grid_size=len(grid),
        symbols_loaded=len(bars.symbols),
        data_start=inp["data_start"],
        last_bar=str(bars.close.index[-1].date()),
        bars_summary=bars.summary(),
        tape=bars.provenance.get("snapshot"),
        ok=True,
        wall_seconds=round(time.time() - t0, 1),
    )
    result.oos_equity.rename("oos_equity").to_csv(out_dir / f"{stem}_oos_equity.csv", header=True)
    return payload


# --------------------------------------------------------------------------- comparison


def _read_equity(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.iloc[:, 0]


def compare_to_reference(stem: str, new: dict[str, Any], ref: dict[str, Any], out: Path) -> dict:
    """Fold-by-fold and aggregate comparison of a run with its P0 legacy reference."""
    rows_new, rows_ref = new["fold_table"], ref["fold_table"]
    report: dict[str, Any] = {
        "run": stem,
        "reference": ref["run"],
        "n_folds": {"new": len(rows_new), "reference": len(rows_ref)},
        "grid_size": {"new": new["grid_size"], "reference": ref["config"]["grid_size"]},
    }
    flags = ("ndh_passed", "dsr_passed", "gates_passed", "used_anchor", "halted", "halted_entering")
    numerics = ("ndh_density", "dsr_value", "is_sharpe", "oos_sharpe", "wfe", "train_score")
    folds: list[dict[str, Any]] = []
    first_real: list[dict[str, Any]] = []
    for rn, rr in zip(rows_new, rows_ref, strict=False):
        entry: dict[str, Any] = {"fold": rr["fold_num"]}
        entry["windows_equal"] = all(
            _ts(rn[k]) == _ts(rr[k]) for k in ("train_start", "train_end", "test_start", "test_end")
        )
        entry["winner_equal"] = rn["winner_combo"] == rr["winner_combo"]
        entry["flags_equal"] = {k: rn[k] == rr[k] for k in flags}
        diffs = {}
        for k in numerics:
            same = close_enough(rn[k], rr[k])
            diffs[k] = {"new": rn[k], "ref": rr[k], "equal_within_ulp_floor": same}
        entry["numerics"] = diffs
        bad = (
            not entry["windows_equal"]
            or not entry["winner_equal"]
            or not all(entry["flags_equal"].values())
            or not all(d["equal_within_ulp_floor"] for d in diffs.values())
        )
        entry["identical"] = not bad
        folds.append(entry)
        if bad and len(first_real) < 5:
            first_real.append(entry)
    report["folds"] = folds
    report["folds_identical"] = sum(1 for f in folds if f["identical"])
    report["winners_identical"] = sum(1 for f in folds if f["winner_equal"])
    report["gates_identical"] = sum(
        1
        for f in folds
        if all(f["flags_equal"][k] for k in ("ndh_passed", "dsr_passed", "gates_passed"))
    )
    report["anchor_identical"] = sum(1 for f in folds if f["flags_equal"]["used_anchor"])
    report["halt_state_identical"] = sum(
        1 for f in folds if f["flags_equal"]["halted"] and f["flags_equal"]["halted_entering"]
    )
    report["first_differences"] = first_real

    eq_ref = _read_equity(REFERENCES / f"{ref['run']}_oos_equity.csv")
    eq_new = _read_equity(out / f"{stem}_oos_equity.csv")
    same_index = eq_ref.index.equals(eq_new.index)
    if same_index and len(eq_ref):
        abs_diff = (eq_ref - eq_new).abs()
        rel = abs_diff / eq_ref.abs()
        report["oos_equity"] = {
            "same_index": True,
            "bars": int(len(eq_ref)),
            "max_abs_diff": float(abs_diff.max()),
            "max_rel_diff": float(rel.max()),
            "max_abs_diff_date": str(abs_diff.idxmax()),
            "identical_within_ulp_floor": bool(rel.max() <= 1e-12),
        }
    else:
        report["oos_equity"] = {
            "same_index": bool(same_index),
            "bars_new": int(len(eq_new)),
            "bars_reference": int(len(eq_ref)),
        }
    agg: dict[str, Any] = {}
    for key, label in (("sharpe", "sharpe"), ("cagr_pct", "cagr_pct"), ("max_drawdown_pct", "mdd")):
        a, b = new["diagnostics"]["oos_metrics"].get(key), (ref.get("oos") or {}).get(key)
        agg[f"oos_{label}"] = {"new": a, "ref": b, "equal_within_ulp_floor": close_enough(a, b)}
        a, b = new["diagnostics"]["spy_metrics"].get(key), (ref.get("spy_oos_index") or {}).get(key)
        agg[f"spy_{label}"] = {"new": a, "ref": b, "equal_within_ulp_floor": close_enough(a, b)}
    report["aggregate"] = agg
    live_new, live_ref = new["live_params"], ref["live_params"]
    report["live_params_combo_equal"] = live_new.get("combo") == live_ref.get("combo")
    report["gates"] = {"new": new["gates"], "reference": ref["gates"]}
    report["anchor_folds"] = {"new": new["anchor_folds"], "reference": ref["anchor_folds"]}
    report["skipped_folds_equal"] = len(new["skipped_folds"]) == len(ref["skipped_folds"])
    report["wall_seconds"] = {"new": new["wall_seconds"], "reference": ref["wall_seconds"]}
    report["peak_rss_mb"] = {
        "new": new["diagnostics"]["peak_rss_mb"],
        "reference": ref.get("peak_rss_mb"),
    }
    report["simulations"] = {
        "new": new["diagnostics"]["n_simulations"],
        "legacy_equivalent": new["diagnostics"]["legacy_equivalent_simulations"],
    }
    report["od12_ndh_flips_folds"] = new["diagnostics"]["od12_ndh_flips_folds"]
    report["verdict"] = (
        "IDENTICAL within the ulp floor"
        if (
            report["folds_identical"] == len(folds) == len(rows_ref)
            and report["oos_equity"].get("identical_within_ulp_floor", False)
            and all(v["equal_within_ulp_floor"] for v in agg.values())
            and report["live_params_combo_equal"]
        )
        else "DIFFERENCES - classify (see first_differences / oos_equity)"
    )
    return report


# --------------------------------------------------------------------------- main


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description="P5 walk-forward validation against P0 references.")
    p.add_argument("--runs", default="all", help="comma list of run names, or 'all'")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help=f"output folder (default: scratch {DEFAULT_OUT}; committed evidence: {COMMITTED_OUT})",
    )
    p.add_argument("--max-gib", type=float, default=3.0, help="engine chunking budget (GiB)")
    p.add_argument("--force", action="store_true", help="redo runs that already have ok JSON")
    p.add_argument(
        "--min-losing-days",
        type=int,
        default=None,
        help="Sortino degeneracy guard (default: engine.config.MIN_LOSING_DAYS; 0 = off)",
    )
    p.add_argument("--list", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the selected cases sequentially; failures are recorded in the JSON, not raised."""
    args = parse_args(argv)
    if args.list:
        for key in ALL_RUNS:
            print(key)
        return 0
    from ggTrader.engine import selfcheck
    from ggTrader.engine.provenance import provenance as build_provenance
    from ggTrader.utils.results import write_result_json

    names = list(RUNS) if args.runs == "all" else [r for r in args.runs.split(",") if r]
    unknown = [n for n in names if n not in ALL_RUNS]
    if unknown:
        print(f"unknown runs: {unknown}", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selfcheck.selfcheck()
    failures = 0
    for name in names:
        spec = ALL_RUNS[name]
        stem = name
        target = out / f"_p5_{stem}.json"
        if target.exists() and not args.force:
            if json.loads(target.read_text()).get("ok"):
                print(f"[skip] {stem}: ok JSON exists", flush=True)
                continue
        print(f"[run] {stem}", flush=True)
        t0 = time.time()
        try:
            payload = run_case(name, spec, int(args.max_gib * GIB), out, stem, args.min_losing_days)
        except Exception as exc:  # noqa: BLE001 - recorded, the chain continues
            failures += 1
            payload = {
                "run": name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_seconds": round(time.time() - t0, 1),
            }
            print(f"[FAIL] {stem}: {payload['error']}", flush=True)
        tape_sha = (payload.get("tape") or {}).get("sha256")
        write_result_json(
            target,
            payload,
            provenance=build_provenance(tape_sha256=tape_sha),
        )
        if not payload.get("ok"):
            continue
        saved = json.loads(target.read_text())
        ref_path = REFERENCES / f"_legacy_reference_{name}.json"
        prov = build_provenance(tape_sha256=tape_sha)
        if ref_path.exists():
            cmp = compare_to_reference(stem, saved, json.loads(ref_path.read_text()), out)
            write_result_json(out / f"_p5_compare_{stem}.json", cmp, provenance=prov)
            print(f"[compare to P0 reference] {stem}: {cmp['verdict']}", flush=True)
        print(f"[done] {stem} {payload['wall_seconds']}s -> {target}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
