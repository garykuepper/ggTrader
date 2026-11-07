"""Build the P5c deletion ledger (p5c-ledger.csv) from explicit decisions plus the import graph.

The decisions below are the only hand-written part. Evidence (who imports what) is computed from
``import_graph.py``; the closure is iterated so a script or test that imports an archived or
deleted file is itself archived, deleted or ported, never left dangling.

Usage: build_ledger.py [--out PATH]      (run from any directory; reads the working tree)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_graph as ig  # noqa: E402

ROOT = ig.ROOT
LAB = "src/ggTrader/lab/"
STRAT = LAB + "strategies/"

# --- src: deleted (reason) -------------------------------------------------------------------
SRC_DELETE: dict[str, str] = {
    LAB + "simulate.py": "hand-rolled simulator replaced by engine.portfolio (864/864 curves bit-identical, P5)",
    LAB + "sweep.py": "per-combo sweep loops, pit_entry_mask and --sweep replaced by one grouped engine simulation, lab.universe and wfo_v2.inputs",
    LAB + "harness.py": "non-WFO runs retired; ggt lab needs --wfo or --blend (spec 5.1: one-fold WFO)",
    LAB + "wfo.py": "replaced by lab/wfo_v2 (P5, P5b); breaker, folds, anchor, gates live there",
    LAB + "metrics.py": "curve_stats/benchmark replaced by wfo_v2.report.curve_summary over engine.stats; oracle copy moves to tests/",
    LAB + "kelly.py": "only importer is ensemble_kelly (closed NO-GO, not in the port set, OD11(b))",
    STRAT + "ic_weights.py": "only importer is ensemble_ic (closed NO-GO, not in the port set)",
    STRAT + "signals.py": "per-combo loops over hand-written indicators; the five voters are ported in strategies/v2/voter_strategies.py; mtf/overnight_gap/wfo_tournament not ported (OD11(b))",
    STRAT + "indicators.py": "hand-written pandas indicators replaced by engine.indicators presets; no kept importer remains",
    STRAT + "ensemble.py": "superseded by strategies/v2/ensemble.py (signals within class (ii), live golden 120/120)",
    STRAT + "ensemble_ic.py": "closed NO-GO, not in the port set (OD11(b)); reproducible at legacy tag",
    STRAT + "ensemble_kelly.py": "closed NO-GO, not in the port set (OD11(b))",
    STRAT + "conviction.py": "superseded by strategies/v2/conviction_bb.py (ensemble_conviction not ported)",
    STRAT + "cross_asset_sleeve.py": "superseded by strategies/v2/cross_asset_sleeve.py",
    STRAT + "momentum.py": "superseded by strategies/v2/xs_momentum.py; dual_momentum not ported",
    STRAT + "retail_attention.py": "superseded by strategies/v2/retail_attention.py",
    STRAT + "registry.py": "legacy registry helpers (build_strategy, apply_sector_constraints); v2/registry.py is the registry",
}
for name in (
    "commodity_trend congress_trades fomc_drift fx_hedge_overlay headline_sentiment idio_vol "
    "index_deletion insider_cluster leveraged_rotation leveraged_trend max_effect "
    "month_end_treasury pairs_stat_arb pead short_interest short_volume_ratio treasury_curve"
).split():
    SRC_DELETE[f"{STRAT}{name}.py"] = (
        "closed NO-GO / untested strategy not in the OD11(b) port set; kept reproducible at tag "
        "legacy-engine-vbt0285 (scripts/legacy_env.sh)"
    )

# --- src: ported / rewritten -----------------------------------------------------------------
SRC_PORT: dict[str, str] = {
    LAB + "blend.py": "drop _legacy_curves/engine flag; run each sleeve through wfo_v2; curve_stats -> curve_summary",
    LAB + "cli.py": "remove --engine/--sweep/plain run; --wfo/--blend on wfo_v2; explicit legacy-tag error for unported names",
    LAB + "replica.py": "build_signal_panel uses the v2 live ensemble (Ensemble.build) with the monthly PIT entry mask; equity-level check against captured legacy panel",
    LAB + "train_gate.py": "entry labels from the v2 live ensemble frames instead of EnsembleSignal._generate_signals",
    LAB + "significance.py": "curve_stats -> curve_summary (vectorbt definitions win)",
    LAB + "wfo_policy_synthetic.py": "imports WfoPolicy/WfoState/breaker from wfo_v2.breaker (same functions)",
    LAB + "strategy.py": "keep LabConfig only; Plan/SignalTargets/Strategy protocol belonged to the legacy classes",
    STRAT + "__init__.py": "no legacy classes left to import; package kept for strategies/vote.py and strategies/v2",
    STRAT + "v2/registry.py": "add LEGACY_ONLY_STRATEGIES (the 28 names) so the CLI can name the tag",
}

# --- scripts ---------------------------------------------------------------------------------
SCRIPT_ARCHIVE: dict[str, str] = {
    "ablation_voters.py": "closed voter ablation on legacy EnsembleSignal/lab.wfo",
    "anchor_fix_reproduction_wfo.py": "legacy WFO driver (lab.wfo/sweep/ensemble); numbers live in its research report",
    "audit_reruns_wfo.py": "legacy WFO driver for xs/dual momentum and fomc_drift reruns",
    "congress_trades_correlation_check.py": "closed NO-GO diversification check on unported congress_trades",
    "core_spy_blend.py": "legacy WFO + simulate driver; closed core+SPY blend study",
    "cross_asset_sleeve_wfo.py": "legacy WFO driver of the closed cross-asset study (sleeve is ported; the study is not rerun here)",
    "deployment_audit.py": "one-off capital-deployment audit on the legacy simulator",
    "fomc_clustered_event_study.py": "closed FOMC drift event study on unported fomc_drift/indicators",
    "fx_hedge_overlay_static_baseline_check.py": "closed NO-GO check of unported fx_hedge_overlay",
    "gate_value_ablation.py": "closed ML-gate ablation on legacy EnsembleSignal",
    "ic_kelly_rebaseline_wfo.py": "closed ensemble_ic/kelly rebaseline; imports anchor_fix_reproduction_wfo",
    "insider_cluster_correlation_check.py": "closed NO-GO check of unported insider_cluster_buy",
    "lever_diagnostic.py": "closed lever diagnostic on legacy wfo/simulate",
    "leveraged_rotation_research.py": "closed NO-GO leveraged rotation study (unported)",
    "leveraged_trend_research.py": "closed NO-GO leveraged trend study (unported); also a raw-vectorbt importer",
    "live_indicator_warmup.py": "one-off live-vs-backtest drift study on legacy EnsembleSignal/indicators",
    "max_effect_correlation_check.py": "closed NO-GO check of unported max_effect",
    "midcap_bias_calibration.py": "closed midcap survivorship calibration on legacy lab.wfo",
    "midcap_research.py": "closed MidCap 400 study on legacy lab.wfo",
    "ml_signal_screen.py": "closed ML pre-screen on legacy signals/indicators",
    "month_end_treasury_wfo.py": "closed A10 study on unported month_end_treasury",
    "new_stack_parity_compare.py": "P1 evidence: compares new-stack legacy-code runs with P0 references",
    "new_stack_reference_runs.py": "P1 parity driver running the legacy WFO code on the new stack; raw-vectorbt importer",
    "p4_strategy_parity.py": "P4 evidence driver: ported strategies vs legacy implementations (needs both)",
    "p5_equity_parity.py": "P5 evidence driver: single-pass simulation vs legacy lab.wfo._sweep_fold",
    "p5_tie_audit.py": "P5 evidence: tie audit built on p4_strategy_parity (archived with it)",
    "pairs_stat_arb_research.py": "closed NO-GO pairs study (unported)",
    "pead_correlation_check.py": "closed NO-GO check of unported pead",
    "position_sizing_regimes.py": "closed position-sizing study on legacy simulate; raw-vectorbt importer",
    "position_sizing_wfo.py": "closed position-sizing WFO on legacy lab.wfo/simulate; raw-vectorbt importer",
    "run_position_sizing_wfo_scheduled.sh": "launcher for position_sizing_wfo.py (archived with it)",
    "wfo_conviction_comparison.sh": "docker launcher of `ggt lab --strategy ensemble_conviction` (unported)",
}
SCRIPT_DELETE: dict[str, str] = {
    "update_universe_ccxt.py": "docstring says BROKEN (ImportError on load); crypto arc parked; recoverable at the tag",
    "update_venue_listings.py": "docstring says BROKEN (ImportError on load); crypto arc parked; recoverable at the tag",
}
SCRIPT_PORT: dict[str, str] = {
    "shadow_cross_asset.py": "ongoing static 80/20 shadow: simulate_weights/legacy strategy -> engine.run_weights + v2 CrossAssetSleeve (hand-computed test kept)",
    "production_replica_audit.py": "curve_stats -> curve_summary; replica signal panel now v2 ensemble",
    "wfo_policy_synthetic.py": "imports from wfo_v2.breaker",
    "engine_parity_evidence.py": "vectorbt-upgrade tape pass: indicators + voters sections kept (voter oracle moved to tests/oracles/legacy_voters.py); simulator section dropped (frozen in tests/engine/golden/legacy_simulate.npz); full P2 script archived as engine_parity_evidence_p2_full.py",
}

SCRIPT_KEEP_EXCEPTION: dict[str, str] = {
    "legacy_reference_runs.py": "documented exception (task decision e): runs under the legacy 0.28.5 interpreter against the tag extracted by legacy_env.sh; imports legacy modules by design and is the last file on the vectorbt allow-list",
}
SHELL_KEEP: dict[str, str] = {
    "run_tests.sh": "operating workflow (memory-capped test runner)",
    "legacy_env.sh": "operating workflow (rebuilds the legacy environment at the tag)",
    "paper_trade.sh": "operating workflow (paper trader cron entry; not touched)",
    "daily_pnl_report.sh": "operating workflow (daily report; not touched)",
    "run_capped.sh": "generic memory-capped runner, has tests",
    "guard_snapshot.sh": "investigation guard snapshot, no code dependency",
}

# --- tests -----------------------------------------------------------------------------------
TEST_DELETE: dict[str, str] = {}
for t in (
    "commodity_trend congress_trades_strategy conviction cross_asset_sleeve ensemble "
    "ensemble_conviction ensemble_ic ensemble_kelly fomc_drift fx_hedge_overlay harness "
    "harness_signals headline_sentiment_strategy ic_weights idio_vol index_deletion_strategy "
    "insider_cluster_strategy kelly leveraged_rotation leveraged_trend macd_signals max_effect "
    "metrics momentum month_end_treasury mtf_signals overnight_gap_signals pairs_stat_arb "
    "pead_strategy raw_values registry retail_attention_strategy reversion_signals "
    "sector_constraints short_interest_strategy short_volume_ratio_strategy signals simulate "
    "simulate_signals strength_indicators sweep treasury_curve validation_gate vol_target "
    "volume_bb_signals wfo"
).split():
    TEST_DELETE[f"tests/lab/test_{t}.py"] = "tests a deleted legacy module (see SRC_DELETE)"
TEST_DELETE["tests/lab/wfo_v2/test_legacy_equivalence.py"] = (
    "compared wfo_v2 with legacy lab.wfo/sweep; the proof is recorded in p5-evidence/ (P5-completion section 2)"
)
TEST_DELETE["tests/scripts/test_leveraged_rotation_research.py"] = "tests an archived script"

TEST_PORT: dict[str, str] = {
    "tests/engine/test_portfolio.py": "engine-vs-lab.simulate comparisons now read tests/engine/golden/legacy_simulate.npz (frozen legacy output)",
    "tests/lab/test_blend.py": "legacy path gone; vbt path and blend maths kept",
    "tests/lab/test_cli.py": "new parser (no --engine/--sweep)",
    "tests/lab/test_replica.py": "build_signal_panel on the v2 ensemble",
    "tests/lab/test_significance.py": "curve_summary instead of curve_stats",
    "tests/lab/test_wfo_policy_synthetic.py": "imports from wfo_v2.breaker",
    "tests/lab/strategies_v2/test_conviction_bb.py": "drop legacy-vs-new halves",
    "tests/lab/strategies_v2/test_ensemble.py": "drop legacy-vs-new halves",
    "tests/lab/strategies_v2/test_registry.py": "no legacy registry to compare with",
    "tests/lab/strategies_v2/test_voter_strategies.py": "drop legacy-vs-new halves",
    "tests/lab/strategies_v2/test_voters.py": "drop legacy-indicator comparisons",
    "tests/lab/strategies_v2/test_weights.py": "drop legacy-vs-new halves",
    "tests/lab/wfo_v2/helpers.py": "curve_stats oracle moves into tests/",
    "tests/lab/wfo_v2/test_logic_units.py": "drop legacy-comparison halves (docstring-marked in P5)",
    "tests/lab/wfo_v2/test_report_inputs_weights.py": "curve_stats oracle from tests/",
    "tests/paper/test_overlay.py": "drop legacy-overlay comparison",
    "tests/paper/test_paper_boundary.py": "banned-import list stays; deleted modules cannot be imported at all",
    "tests/test_pandas3_compat.py": "drop legacy strategy/indicator cases",
    "tests/scripts/test_driver_default_paths.py": "CHANGED_SCRIPTS lists archived drivers",
    "tests/scripts/test_check_vbt_engine.py": "move under tests/engine/ so the boundary allow-list reaches zero",
}


def _lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("rb"))
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "research/programme/vectorbt-engine/P5c-ledger.csv"))
    args = ap.parse_args()
    _, edges, _ = ig.build()
    rev: dict[str, list[str]] = {}
    for src, targets in edges.items():
        for t in targets:
            rev.setdefault(t, []).append(src)

    rows: list[dict[str, str]] = []

    def add(path: str, action: str, reason: str, evidence: str) -> None:
        rows.append(
            {
                "path": path,
                "action": action,
                "reason": reason,
                "evidence": evidence,
                "lines": str(_lines(ROOT / path)),
            }
        )

    gone = set(SRC_DELETE)
    keep_exc = {f"scripts/{n}" for n in SCRIPT_KEEP_EXCEPTION}
    arch = {f"scripts/{n}" for n in SCRIPT_ARCHIVE} | {f"scripts/{n}" for n in SCRIPT_DELETE}
    # closure: any other script that imports an archived/deleted script goes with it
    changed = True
    extra: dict[str, str] = {}
    while changed:
        changed = False
        for f, deps in edges.items():
            if f.startswith("scripts/") and f not in arch and f not in extra and f not in keep_exc:
                hit = sorted(d for d in deps if d in arch or d in gone)
                if hit and f not in {f"scripts/{n}" for n in SCRIPT_PORT}:
                    extra[f] = "imports " + ",".join(h.split("/")[-1] for h in hit)
                    arch.add(f)
                    changed = True

    def ev_src(p: str) -> str:
        users = sorted(rev.get(p, []))
        kept = [u for u in users if u not in gone and u not in arch]
        return f"{len(users)} importers; {len(kept)} outside the deleted/archived set: " + ", ".join(
            k.replace("src/ggTrader/", "") for k in kept[:8]
        )

    for p, why in sorted(SRC_DELETE.items()):
        add(p, "delete", why, ev_src(p))
    for p, why in sorted(SRC_PORT.items()):
        deps = sorted(d.split("/")[-1] for d in edges.get(p, set()) if d in gone)
        add(p, "port", why, "imports deleted: " + (", ".join(deps) or "none"))
    for n, why in sorted(SCRIPT_ARCHIVE.items()):
        add(f"scripts/{n}", "archive", why, "imports deleted/archived: " + ", ".join(
            sorted(d.split("/")[-1] for d in edges.get(f"scripts/{n}", set()) if d in gone or d in arch)) or "shell launcher")
    for p, why in sorted(extra.items()):
        add(p, "archive", "closure: " + why, why)
    for n, why in sorted(SCRIPT_KEEP_EXCEPTION.items()):
        add(f"scripts/{n}", "keep", why, "imports legacy modules by design (legacy interpreter only)")
    for n, why in sorted(SHELL_KEEP.items()):
        add(f"scripts/{n}", "keep", why, "shell script, outside the python import graph")
    for n, why in sorted(SCRIPT_DELETE.items()):
        add(f"scripts/{n}", "delete", why, "BROKEN per its own docstring; no importer")
    for n, why in sorted(SCRIPT_PORT.items()):
        deps = sorted(d.split("/")[-1] for d in edges.get(f"scripts/{n}", set()) if d in gone)
        add(f"scripts/{n}", "port", why, "imports deleted: " + (", ".join(deps) or "none"))
    for p, why in sorted(TEST_DELETE.items()):
        add(p, "delete", why, "imports deleted: " + ", ".join(
            sorted(d.split("/")[-1] for d in edges.get(p, set()) if d in gone or d in arch)))
    for p, why in sorted(TEST_PORT.items()):
        add(p, "port", why, "imports deleted: " + ", ".join(
            sorted(d.split("/")[-1] for d in edges.get(p, set()) if d in gone or d in arch)))

    listed = {r["path"] for r in rows}
    for f in sorted(edges):
        if f.startswith("scripts/") and f not in listed:
            add(f, "keep", "operating workflow / kept-module tool / evidence driver that runs on kept code", "no deleted or archived import")
    for f in sorted(edges):
        if f in listed or not f.startswith(("src/ggTrader/lab/", "src/ggTrader/data/", "tests/")):
            continue
        bad = sorted(d for d in edges[f] if d in gone or d in arch)
        if bad:
            add(f, "port", "still imports a deleted/archived file: fix", ", ".join(b.split("/")[-1] for b in bad))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "action", "reason", "evidence", "lines"])
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print(out, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
