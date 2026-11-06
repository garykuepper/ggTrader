"""Rebuild the Lens 2 verdict-audit JSON (classifications and sensitivity table) from inputs.

Reads committed files (the investigation inventory, the Lens 1/6/7/9 outputs and the raw
artifacts the inventory names) plus one gitignored artifact,
results/research/research_20260511_193048/wfo_stats_snapshot.json. On a fresh clone that file is
absent and the crypto-wfo-textbook-reset sensitivity row reads "listed path missing".
Writes only the paths given with --out-json and --md-dir.
Classification rules are the fixed text in RULES_TEXT; the verdict-type vocabularies, the
contamination-direction table, the 1.12-comparator pattern and the column-14 logic are module
constants below. Sensitivity columns follow
research/audits/root-cause-2026-11-04/2026-11-04-lens2-sensitivity-spec.md.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
#: Lens inputs (repo-relative), each at its own location in the root-cause audit folder.
INVENTORY_JSON = "research/audits/root-cause-2026-11-04/_investigation_inventory.json"
LENS1_JSON = "research/audits/root-cause-2026-11-04/_lens1_tape_state.json"
LENS6_JSON = "research/audits/root-cause-2026-11-04/_lens6_verdicts.json"
LENS7_JSON = "research/audits/root-cause-2026-11-04/_lens7_recompute.json"
LENS9_JSON = "research/audits/root-cause-2026-11-04/_lens9_power.json"
#: Screen results read directly by build() for the multiple-testing section.
FACTOR_SCREEN_JSON = "research/stock-signals/factor-screen/_factor_screen_results.json"
MOMENTUM_SCREEN_JSON = "research/stock-signals/momentum/_momentum_french_screen_results.json"
NET_ISSUANCE_JSON = "research/stock-signals/net-issuance/_net_issuance_screen_results.json"
#: Current location of each raw artifact the inventory names, keyed by file name. The inventory
#: is a frozen artifact and still lists the pre-2026-11-05 paths (before the move to research/),
#: so sens_rows_for() resolves each listed path to its study folder through this table.
ARTIFACT_LOCATIONS: dict[str, str] = {
    "_anchor_fix_reproduction_results.json": (
        "research/audits/rebaseline/_anchor_fix_reproduction_results.json"
    ),
    "_audit_reruns_20261019.json": "research/audits/production-replica/_audit_reruns_20261019.json",
    "_century_leverage_results.json": (
        "research/allocation/century-leverage/_century_leverage_results.json"
    ),
    "_core_spy_blend_20260924.json": (
        "research/allocation/core-spy-blend/_core_spy_blend_20260924.json"
    ),
    "_cross_asset_sleeve_holdout_results.json": (
        "research/allocation/cross-asset-sleeve/_cross_asset_sleeve_holdout_results.json"
    ),
    "_cross_asset_sleeve_results.json": (
        "research/allocation/cross-asset-sleeve/_cross_asset_sleeve_results.json"
    ),
    "_event_audit_readiness_decision.json": (
        "research/audits/event-timing/_event_audit_readiness_decision.json"
    ),
    "_event_evidence_export_file_search.json": (
        "research/audits/event-timing/_event_evidence_export_file_search.json"
    ),
    "_event_evidence_resolution_workbook.json": (
        "research/audits/event-timing/_event_evidence_resolution_workbook.json"
    ),
    "_factor_screen_results.json": (
        "research/stock-signals/factor-screen/_factor_screen_results.json"
    ),
    "_family_scope_extraction.json": (
        "research/stock-signals/feasibility/_family_scope_extraction.json"
    ),
    "_fomc_clustered_event_study_20261020.json": (
        "research/audits/production-replica/_fomc_clustered_event_study_20261020.json"
    ),
    "_free_data_feasibility_matrix.json": (
        "research/stock-signals/feasibility/_free_data_feasibility_matrix.json"
    ),
    "_ic_kelly_rebaseline_20260923.json": (
        "research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json"
    ),
    "_insider_acceptance_diagnostic.json": (
        "research/audits/event-timing/_insider_acceptance_diagnostic.json"
    ),
    "_insider_identity_inventory_resolution_results.json": (
        "research/audits/event-timing/_insider_identity_inventory_resolution_results.json"
    ),
    "_intraday_benchmark_verification.json": (
        "research/intraday/spy-momentum-feasibility/_intraday_benchmark_verification.json"
    ),
    "_live_indicator_warmup_results.json": (
        "research/paper-system/indicator-warmup/_live_indicator_warmup_results.json"
    ),
    "_momentum_french_screen_results.json": (
        "research/stock-signals/momentum/_momentum_french_screen_results.json"
    ),
    "_month_end_treasury_results.json": (
        "research/allocation/month-end-treasury/_month_end_treasury_results.json"
    ),
    "_net_issuance_screen_results.json": (
        "research/stock-signals/net-issuance/_net_issuance_screen_results.json"
    ),
    "_pit_rebaseline_20260923.json": "research/audits/rebaseline/_pit_rebaseline_20260923.json",
    "_position_sizing_wfo_results.json": (
        "research/paper-system/position-sizing/_position_sizing_wfo_results.json"
    ),
    "_production_replica_results.json": (
        "research/audits/production-replica/_production_replica_results.json"
    ),
    "_rebaseline_corrected_tape_20260822.json": (
        "research/audits/rebaseline/_rebaseline_corrected_tape_20260822.json"
    ),
    "_replica_parity_20261019.json": (
        "research/audits/production-replica/_replica_parity_20261019.json"
    ),
    "_research_process_audit_ledger.json": (
        "research/audits/research-process/_research_process_audit_ledger.json"
    ),
    "_static_leverage_aftertax_results.json": (
        "research/allocation/static-leverage/_static_leverage_aftertax_results.json"
    ),
    "_wfo_fold_attribution_results.json": (
        "research/programme/wfo-methods/_wfo_fold_attribution_results.json"
    ),
    "_wfo_policy_synthetic_results.json": (
        "research/programme/wfo-methods/_wfo_policy_synthetic_results.json"
    ),
    "ibit_overnight_split.json": (
        "research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json"
    ),
}
SPEC_COMMIT = "a5fe547b"

#: Fixed classification rules, verbatim as written before any row was classified.
RULES_TEXT = "\n\n".join(
    [
        (
            "Lens 2 classification rules, fixed 2026-11-04 before any row was classified. "
            "Inputs are the committed upstream lens outputs only; no result value chooses a "
            "class except through these rules."
        ),
        (
            "R1 Verdict type, from the inventory verdict text through a fixed vocabulary: "
            "NO (a measured test reported fail, no edge, underperformance, did not "
            "reproduce, or no alternative passed); GO (a measured test reported pass, "
            "validated, beat, keep, shipped or done on evidence, or a model claim "
            "supported); TEST-INCONCLUSIVE (the test's own verdict is inconclusive or "
            "invalid); STATUS (feasibility, design, intake, literature, untriaged, "
            "component, operations, data-fix and audit labels). Every STATUS and "
            "TEST-INCONCLUSIVE row is classed inconclusive; STATUS rows are counted "
            "separately because they are not edge tests."
        ),
        (
            "R2 Data condition (Lens 1). Each contamination item gets a direction from a "
            "fixed table with evidence: anchor_leak, pit_universe, universe survivorship, "
            "ML-gate training leakage and event pre-entry returns favour the strategy "
            "(look-ahead or survivorship); spy_duplicates is unclear (it deflated strategy "
            "and SPY Sharpe alike); dsr_units, day_shift and tape identity faults are "
            "unknown. The condition is met for a NO if the tape was clean or every item "
            "favours the strategy (it lost despite the help), and for a GO only if the tape "
            "was clean. Lens 1 'unknown' with no items, and crypto rows (Lens 8 owns them), "
            "do not meet it."
        ),
        (
            "R3 Code condition (Lens 6). Met if Lens 6 says yes; if Lens 6 has no row or is "
            "unverifiable but Lens 7 independently recomputed the tested statistic from "
            "frozen raw inputs; or if Lens 6 says partly and the mismatch is outside the "
            "verdict's declared scope, logged as a named exception. Not met if Lens 6 is "
            "partly (otherwise), no, unverifiable or absent."
        ),
        (
            "R4 Recompute condition (Lens 7). Met if Lens 7 match is yes, or the row has no "
            "raw artifact and its claim is prose only (flagged as such). Not met for no, "
            "partial or cannot_locate."
        ),
        (
            "R5 Power condition (Lens 9), NO rows only. The effect a NO is used to rule out "
            "is the row's own pre-registered economic bar in pp/yr (+1.0 for the replica, "
            "momentum replays and French screens); if none, the owner's 2x-SPY target gap "
            "over the row's window from Lens 9 (target_required_cagr_gap_pp). Met only if "
            "Lens 9's MDE at t >= 3 (80% power) is known and at or below that effect. An "
            "MDE that Lens 9 could not compute means power is not established."
        ),
        (
            "R6 Classes. NO rows: 'unreliable NO (flawed test)' if Lens 6 says no and the "
            "row's claim or headline names the later-voided 1.12 baseline as its "
            "comparator, or if Lens 7 says the numbers do not recompute; otherwise "
            "'reliable NO' if R2-R5 are all met; otherwise 'inconclusive', listing every "
            "unmet condition. GO rows: 'unreliable GO' if any contamination item favours "
            "the strategy, Lens 6 says no, or Lens 7 says no; otherwise 'reliable GO' if "
            "the tape was clean, R3 and R4 are met and the bar was pre-registered (Lens 7 "
            "status pre-registered or amended); otherwise 'inconclusive'."
        ),
        (
            "R7 'Rules out effects larger than' is Lens 9's MDE at t >= 3 in pp/yr where "
            "Lens 9 computed one, else 'unknown'. It is reported for every row, whatever "
            "its class, so that no NO is read as 'no edge of any size'."
        ),
        (
            "Logged exception E1 (made before classification): the production replica's "
            "Lens 6 'partly' (t-1 signal, t-close fill versus the live partial bar) is "
            "treated as outside the recorded verdict's scope, because closure.scope is "
            "'historical_daily_bar_model' and the parity amendment narrowed the question to "
            "a prior-completed-bar model. This exception changes no class: the replica "
            "fails R2 and R5 in any case."
        ),
    ]
)

#: Post hoc note added in review fix round 1 (2026-11-04); it changes no rule and no class.
RULES_KNOWN_LIMITATIONS = (
    "Asymmetry: R5 requires a NO to show power, while R6 asks a GO for no t-statistic, no power "
    "and no check beyond its own bar. A23, the cross-asset 80/20, its 2007-2011 holdout and the "
    "synthetic gate study each fail only the code check and would become 'reliable GO' if Lens 6 "
    "examined their code, despite having no t-statistic (and, for the 80/20, despite raw CAGR "
    "trailing SPY). Rule consequence: R6 needs Lens 6 'no' AND a named 1.12 comparator for "
    "'unreliable NO', so stocks-ensemble-kelly-2026-07 (Lens 6 'no', no named comparator) stays "
    "inconclusive."
)

#: STATUS rows are labels, not edge tests.
STATUS_NOTE = (
    "verdict_type STATUS rows (42) are status labels, not tests; they are classed inconclusive "
    "by rule R1. Count tests with verdict_type in {NO, GO, TEST-INCONCLUSIVE} (53 + 16 + 3 = 72); "
    "do not read class == 'inconclusive' over all 114 rows as 103 inconclusive tests."
)

inv: list[dict[str, Any]] = []
L1: dict[str, dict[str, Any]] = {}
L6: dict[str, dict[str, Any]] = {}
L7rc: dict[str, list[dict[str, Any]]] = {}
L7pr: dict[str, dict[str, Any]] = {}
L9: dict[str, dict[str, Any]] = {}


def load(p: str) -> Any:
    """Load a committed JSON file relative to the repository root."""
    return json.loads((ROOT / p).read_text())


def load_inputs() -> None:
    """Populate the module-level input tables from the committed lens outputs."""
    inv.extend(load(INVENTORY_JSON))
    L1.update({r["id"]: r for r in load(LENS1_JSON)})
    L6.update({r["id"]: r for r in load(LENS6_JSON)})
    l7 = load(LENS7_JSON)
    for r in l7["recompute"]:
        L7rc.setdefault(r["id"], []).append(r)
    L7pr.update({r["id"]: r for r in l7["preregistration"]})
    L9.update({r["id"]: r for r in load(LENS9_JSON)["rows"]})


# ---------------------------------------------------------------------------
# CLASSIFICATION RULES (fixed before any row was classified; see report)
# ---------------------------------------------------------------------------
NO_V = {
    "NO-GO",
    "closed NO-GO",
    "NO-GO stands",
    "NO-GO (PROVISIONAL)",
    "Soft NO-GO",
    "Historical NO-GO",
    "NO-GO as a return improvement",
    "clear NO-GO",
    "Rejected",
    "rejected; SP500 stays the core",
    "Rejected as a standalone strategy",
    "failed",
    "fails its bar",
    "loses to SPY",
    "no risk-adjusted edge",
    "matches SPY, no alpha",
    "no deployable edge",
    "did not reproduce",
    "Underperforms; Revert Queued",
    "recommend not queuing",
    "Inactive",
    "vacuous at this book size",
    "closed",
    "keep 30%",
    "closed in practice — do not build",
}
GO_V = {
    "GO",
    "GO for owner decision",
    "GO (shadow only)",
    "PASSED (weak)",
    "KEEP",
    "Validated",
    "Beat SPY in WFO",
    "Shipped",
    "Done",
    "Best = core+macd+vbb (5-voter, drop MTF)",
    "Live Core",
    "RECOMMENDED",
    "supported under this model",
    "modestly profitable",
    "directionally credible but not numerically trustworthy",
}
INC_V = {"Inconclusive", "INCONCLUSIVE", "invalid"}

DIRECTION = {
    "anchor_leak": "favours strategy",
    "pit_universe": "favours strategy",
    "universe_survivorship (2eafab1)": "favours strategy",
    "train_gate_membership_purge (2eafab1)": "favours strategy",
    "event_pre_entry_returns (7320c1d)": "favours strategy",
    "spy_duplicates": "unclear",
    "dsr_units": "unknown",
    "day_shift": "unknown",
    "tape_identity (PARA, unfixed)": "unknown",
    "tape_identity (2010-2020, unfixed)": "unknown",
    "tape_identity (unfixed)": "unknown",
}
DIRECTION_EVIDENCE = {
    "favours strategy": {
        "anchor_leak": "anchor fallback fitted on the full evaluation span, i.e. lookahead in the "
        "fallback parameters (Lens 1 fix timeline row 3; 2eafab1)",
        "pit_universe": "entries on non-member days (14.1% of core entries); core Sharpe "
        "0.9914 -> 0.5922 after c9d1abe (_rebaseline_corrected_tape_20260822.json core.sharpe "
        "vs _pit_rebaseline_20260923.json ensemble.sharpe)",
        "universe_survivorship (2eafab1)": "today's membership applied to the past (Lens 1)",
        "train_gate_membership_purge (2eafab1)": "missing purge gap / membership in gate training "
        "(label leakage; Lens 1 fix timeline row 3)",
        "event_pre_entry_returns (7320c1d)": "returns before entry were counted (not earnable)",
    },
    "unclear": "duplicate SPY rows deflated strategy and SPY Sharpe alike (Lens 1, Phase 0 "
    "hypotheses section)",
    "unknown": "dsr_units changes which parameters deploy (wfo.py:747-752), direction not "
    "measured; day_shift direction depends on each study's join (Lens 1 could not confirm); "
    "tape identity faults (PARA etc.) unmeasured",
}

# Rows whose NO was judged against a later-voided internal baseline (inventory claim/headline
# names the 1.12 / 0.84 / 1.14 baseline as the comparator). Text rule, not value rule.
VOID_BASELINE_RE = re.compile(r"1\.12 baseline|SP500 core Sharpe 1\.12|vs 1\.12")

# Logged exceptions to the mechanical code rule (judgment calls, stated in the report).
CODE_EXCEPTIONS = {
    "stocks-production-replica": "Lens 6 'partly': the timing mismatch (t-1 signal, t close "
    "fill vs live partial bar) is outside the verdict's declared scope "
    "(closure.scope = historical_daily_bar_model; parity amendment lines 38-42), so treated "
    "as irrelevant to the recorded verdict. Logged exception E1.",
}

# Effect each NO is used to rule out (pp/yr): own pre-registered economic bar, else owner
# target gap from Lens 9 (target_required_cagr_gap_pp). Fixed by text before values.
EFFECT_BAR = {
    "stocks-production-replica": (1.0, "pre-registered CAGR bar +1.0 pp/yr (brief §4)"),
    "stocks-production-replica-holdout-2010-2020": (1.0, "pre-registered CAGR bar +1.0 pp/yr"),
    "stocks-xs-momentum-corrected-replay": (1.0, "pre-registered CAGR bar +1.0 pp/yr"),
    "stocks-dual-momentum": (1.0, "pre-registered CAGR bar +1.0 pp/yr (same as xs_momentum)"),
    "stocks-factor-screen-h1-h4": (1.0, "pre-registered A2 bar +1.0 pp/yr"),
    "stocks-momentum-french-screen": (1.0, "pre-registered A2 bar +1.0 pp/yr"),
    "stocks-net-issuance-screen": (1.0, "pre-registered A2 bar +1.0 pp/yr"),
}


def l7_recompute_status(i, has_art):
    rows = L7rc.get(i, [])
    if not rows:
        return ("prose-only (flagged)" if not has_art else "not recomputed by Lens 7"), None
    m = rows[0]["match"]
    return m, rows[0]["source_of_check"]


def code_status(i):
    b = L6.get(i)
    rc = L7rc.get(i, [])
    indep = any(
        x["source_of_check"] == "recomputed_by_me"
        and "French" in x.get("evidence", "")
        or (x["source_of_check"] == "recomputed_by_me" and "frozen" in x.get("evidence", ""))
        for x in rc
    )
    if b is None or b["spec_matches_code"] == "unverifiable":
        if indep:
            return "statistic independently recomputed from raw data (Lens 7)", True
        return "unexamined (Lens 6 " + ("no row" if b is None else "unverifiable") + ")", False
    s = b["spec_matches_code"]
    if s == "yes":
        return "match (Lens 6 yes)", True
    if s == "no":
        return "mismatch (Lens 6 no)", False
    if i in CODE_EXCEPTIONS:
        return "partial, mismatch judged irrelevant (exception E1)", True
    return "partial (Lens 6 partly)", False


def data_status(i, vtype):
    a = L1.get(i, {})
    st = a.get("tape_state_at_run")
    items = a.get("contamination") or []
    dirs = {c: DIRECTION.get(c, "unknown") for c in items}
    if st == "clean":
        return "clean", dirs, "clean"
    if not items:
        if i.startswith("crypto-"):
            return "unknown (crypto; Lens 8 owns)", dirs, "unknown"
        return "unknown (no Lens 1 items)", dirs, "unknown"
    vals = set(dirs.values())
    if vals == {"favours strategy"}:
        return "contaminated, all items favour strategy", dirs, "favours"
    if "favours strategy" in vals:
        return "contaminated, mixed (some favour strategy, others unclear/unknown)", dirs, "mixed"
    return "contaminated, direction unclear/unknown", dirs, "unknown"


def verdict_type(v):
    if v in NO_V:
        return "NO"
    if v in GO_V:
        return "GO"
    if v in INC_V:
        return "TEST-INCONCLUSIVE"
    return "STATUS"


def classify(r):
    i, v = r["id"], r["verdict"]
    vt = verdict_type(v)
    has_art = bool(r["raw_artifact"])
    dlabel, dirs, dkey = data_status(i, vt)
    clabel, c_ok = code_status(i)
    rlabel, rsrc = l7_recompute_status(i, has_art)
    r_ok = rlabel in ("yes", "prose-only (flagged)")
    p9 = L9.get(i, {})
    mde = p9.get("mde_t3_pp_per_yr")
    mde2 = p9.get("mde_t2_pp_per_yr")
    eff = EFFECT_BAR.get(i)
    if eff is None and p9.get("target_required_cagr_gap_pp") is not None and mde is not None:
        eff = (
            p9["target_required_cagr_gap_pp"],
            "owner target gap (Lens 9 target_required_cagr_gap_pp)",
        )
    if mde is None:
        p_label, p_ok = "power not established (Lens 9: " + str(p9.get("status")) + ")", False
    elif eff is None:
        p_label, p_ok = f"MDE {mde:.2f} pp/yr; no effect size to compare", False
    else:
        p_ok = mde <= eff[0]
        p_label = (
            f"MDE(t>=3) {mde:.2f} pp/yr {'<=' if p_ok else '>'} effect ruled out "
            f"{eff[0]:.2f} pp/yr ({eff[1]})"
        )
    reasons = []
    cls = None
    if vt == "STATUS":
        cls = "inconclusive"
        reasons.append("status label, not an edge test (rule S)")
    elif vt == "TEST-INCONCLUSIVE":
        cls = "inconclusive"
        reasons.append("test's own verdict is inconclusive/invalid (rule I)")
    elif vt == "NO":
        flawed = []
        if L6.get(i, {}).get("spec_matches_code") == "no" and VOID_BASELINE_RE.search(
            (r.get("claim") or "") + " " + (r.get("headline_number") or "")
        ):
            flawed.append(
                "judged against the later-voided 1.12 baseline with pre-fix code (Lens 6 no)"
            )
        if rlabel == "no":
            flawed.append("numbers do not recompute (Lens 7 no)")
        if flawed:
            cls = "unreliable NO (flawed test)"
            reasons += flawed
        else:
            d_ok = dkey in ("clean", "favours")
            if d_ok and c_ok and r_ok and p_ok:
                cls = "reliable NO"
            else:
                cls = "inconclusive"
                if not d_ok:
                    reasons.append("data: " + dlabel)
                if not c_ok:
                    reasons.append("code: " + clabel)
                if not r_ok:
                    reasons.append("recompute: " + rlabel)
                if not p_ok:
                    reasons.append("power: " + p_label)
    else:  # GO
        flawed = []
        if dkey in ("favours", "mixed"):
            flawed.append(
                "contamination favouring the strategy: "
                + ", ".join(k for k, d in dirs.items() if d == "favours strategy")
            )
        if L6.get(i, {}).get("spec_matches_code") == "no":
            flawed.append("code did not match spec at run time or tested non-live code (Lens 6 no)")
        if rlabel == "no":
            flawed.append("numbers do not recompute (Lens 7 no)")
        if flawed:
            cls = "unreliable GO"
            reasons += flawed
        else:
            pre = L7pr.get(i, {}).get("status") in ("pre-registered", "amended")
            if dkey == "clean" and c_ok and r_ok and pre:
                cls = "reliable GO"
            else:
                cls = "inconclusive"
                if dkey != "clean":
                    reasons.append("data: " + dlabel)
                if not c_ok:
                    reasons.append("code: " + clabel)
                if not r_ok:
                    reasons.append("recompute: " + rlabel)
                if not pre:
                    reasons.append(
                        "bar not pre-registered (Lens 7: "
                        + str(L7pr.get(i, {}).get("status"))
                        + ")"
                    )
    if mde is not None:
        rules_out = round(mde, 2)
        rules_note = (
            f"MDE t>=3 {mde:.2f}; t>=2 {mde2:.2f} pp/yr (Lens 9, {p9.get('se_method', '')[:60]}...)"
        )
    else:
        rules_out = "unknown"
        rules_note = (
            "unknown: "
            + str(p9.get("status"))
            + (f" (missing {p9.get('missing_key')})" if p9.get("missing_key") else "")
        )
    # Bias direction text
    if dirs:
        bias = "; ".join(f"{k}: {d}" for k, d in dirs.items())
    else:
        bias = dlabel
    return {
        "id": i,
        "verdict": v,
        "verdict_type": vt,
        "class": cls,
        "rules_out_larger_than_pp_per_yr": rules_out,
        "rules_out_note": rules_note,
        "evidence": {
            "lens1": f"{L1.get(i, {}).get('tape_state_at_run')} "
            f"{L1.get(i, {}).get('contamination')}",
            "lens6": f"{L6[i]['spec_matches_code'] if i in L6 else 'no row'}",
            "lens7": f"prereg={L7pr.get(i, {}).get('status')}; recompute={rlabel}"
            + (f" ({rsrc})" if rsrc else ""),
            "lens9": f"{p9.get('status')}; mde_t3={mde}",
        },
        "data_condition": dlabel,
        "code_condition": clabel,
        "recompute_condition": rlabel,
        "power_condition": p_label,
        "bias_direction": bias,
        "note": "; ".join(reasons) if reasons else "all four conditions met",
    }


# ---------------------------------------------------------------------------
# SENSITIVITY TABLE (spec a5fe547b)
# ---------------------------------------------------------------------------
COLS = [
    "result_id",
    "window_used",
    "strategy_raw_cagr_pct",
    "spy_tr_cagr_pct",
    "raw_cagr_gap_pp",
    "ci95_gap",
    "t_stat",
    "verdict_t_ge_3",
    "verdict_t_ge_2",
    "after_tax_gap_pp",
    "power_note",
    "original_verdict_and_bar",
    "artifact_and_keys",
    "exposure_and_leverage",
    "se_method_and_quantity",
]
NRA = "no raw artifact"


def nc(missing):
    return f"not computable from artifact (missing: {missing})"


def f2(x):
    return f"{x:.2f}"


LENS1_BASIS = (
    "TR basis: Lens 1 finding, tape closes are yfinance auto_adjust=True "
    "(src/ggTrader/data/live/yfinance_loader.py:66); adjustment seams per Lens 1 §(b)"
)


def bar_text(i, verdict):
    p = L7pr.get(i, {})
    pb = p.get("pass_bar") or ""
    st = p.get("status")
    if pb:
        return (
            f"{verdict}; bar: {pb} (source: _lens7_recompute.json preregistration, status "
            f"{st}; brief {p.get('brief')})"
        )
    return (
        f"{verdict}; no pre-registered bar found (looked in: the study briefs, the "
        f"2026-09-10 audit, source report; Lens 7 status '{st}')"
    )


def stat_row(rid, window, c3, c4, c5, ann, t, after, bar, keys, expo, se_method, t_bar_ok=True):
    """Fill columns 6-9 and 11 from ann and t with one SE method."""
    if t is None:
        miss = nc("t-statistic")
        c6 = c7 = c8 = c9 = c11 = miss
    else:
        c7 = f2(t)
        c8 = "clears" if t >= 3 else "does not clear"
        c9 = "clears" if t >= 2 else "does not clear"
        if ann is None:
            c6 = nc("annualized mean excess")
            c11 = nc("annualized mean excess")
        elif t == 0 or math.isinf(t):
            c6 = c11 = f"not computable from artifact (t is {t})"
        else:
            se = ann / t
            se = abs(se)
            lo, hi = ann - 1.96 * se, ann + 1.96 * se
            c6 = f"[{lo:.2f}, {hi:.2f}] pp/yr on annualized mean daily excess vs SPY"
            c11 = f"MDE t>=3: {(3 + 0.8416) * se:.2f} pp/yr; t>=2: {(2 + 0.8416) * se:.2f} pp/yr"
    return dict(
        zip(COLS, [rid, window, c3, c4, c5, c6, c7, c8, c9, after, c11, bar, keys, expo, se_method])
    )


def sens_rows_for(r):
    i, art, v = r["id"], r["raw_artifact"], r.get("verdict") or "no verdict recorded in inventory"
    bar = bar_text(i, v)
    if not art:
        return [
            dict(zip(COLS, [i] + [NRA] * 10 + [bar, NRA, expo_from_sources(r), "none reported"]))
        ]
    loc = ARTIFACT_LOCATIONS.get(Path(art).name, art)
    p = ROOT / loc
    if not p.exists():
        m = f"no raw artifact (listed path missing: {loc})"
        return [dict(zip(COLS, [i] + [m] * 10 + [bar, m, expo_from_sources(r), "none reported"]))]
    d = json.loads(p.read_text())
    h = HANDLERS.get(loc)
    if h is None:
        return [generic_row(i, loc, d, bar)]
    return h(i, loc, d, bar, r)


#: Column 14 for rows without a raw artifact (spec section 2.2: "from the source report where it
#: states them"). Only the row's OWN inventory `sources[].quote` strings are searched, so a hit in
#: a shared file (changelog, roadmap) can never be attributed to another row. Review fix round 1
#: replaced an earlier regex over whole source files that mis-attributed rows.
EXPO_RE = re.compile(
    r"leverag|position[ _]size|position_pct|% of (?:cash|portfolio|book)|exposure|sizing|"
    r"\bmax[-_ ]positions\b|\b\d+(?:\.\d+)?x\b",
    re.I,
)


def expo_from_sources(r: dict[str, Any]) -> str:
    """Column 14 for a no-artifact row, quoted only from the row's own inventory sources."""
    for src in r.get("sources") or []:
        q = src.get("quote") or ""
        if EXPO_RE.search(q):
            return f'no raw artifact; source report states: "{q}" (source: {src.get("source")})'
    return "no raw artifact; not stated in source report (row's own inventory quotes searched)"


def generic_row(i, art, d, bar):
    win = None
    if isinstance(d, dict):
        for a, b in (("eval_start", "eval_end"), ("since", "through")):
            if a in d and b in d:
                win = (f"{d[a]}..{d[b]}", f"{a}; {b}")
        if win is None and isinstance(d.get("meta"), dict) and "eval_span" in d["meta"]:
            s = d["meta"]["eval_span"]
            win = (f"{s[0]}..{s[1]}", "meta.eval_span")
        if win is None and isinstance(d.get("window"), list):
            win = (f"{d['window'][0]}..{d['window'][1]}", "window")
    c2 = win[0] if win else nc("window")
    keys = f"{art}; " + (win[1] if win else "no window key") + "; (no CAGR, comparator or t keys)"
    miss3 = nc(
        "CAGR fields; artifact is a readiness/inventory/audit/diagnostic record, not a return test"
    )
    return dict(
        zip(
            COLS,
            [
                i,
                c2,
                miss3,
                nc("SPY series"),
                nc("CAGR fields"),
                nc("t-statistic"),
                nc("t-statistic"),
                nc("t-statistic"),
                nc("t-statistic"),
                nc("no after-tax fields").replace(
                    "(missing: no after-tax fields)", "(no after-tax fields)"
                ),
                nc("t-statistic"),
                bar,
                keys,
                "not recorded in artifact",
                "none reported",
            ],
        )
    )


AT_NONE = "not computable from artifact (no after-tax fields)"


def h_static(i, art, d, bar, r):
    arm = (
        "static 1.3x (70/30) band 8pp"
        if i == "stocks-a23-static-leverage"
        else "A24 rider: 60 SPY + 40 sleeve (10m SMA +/-2%)"
    )
    rows = []
    for w in ("main_real_sso", "holdout_synthetic"):
        for b in ("24/15", "40.8/23.8"):
            A = d["windows"][w][arm][b]
            S = d["windows"][w]["SPY buy-and-hold"][b]
            Aw = d["windows"][w][arm][b + " wash"]
            Sw = d["windows"][w]["SPY buy-and-hold"][b + " wash"]
            c3, c4 = A["pre_tax_cagr"] * 100, S["pre_tax_cagr"] * 100
            at = (A["after_tax_cagr"] - S["after_tax_cagr"]) * 100
            atw = (Aw["after_tax_cagr"] - Sw["after_tax_cagr"]) * 100
            keys = (
                f"{art}; windows.{w}.{arm}.{b}.{{window,pre_tax_cagr,after_tax_cagr}}; "
                f"windows.{w}.SPY buy-and-hold.{b}.{{pre_tax_cagr,after_tax_cagr}}; "
                f"windows.{w}.<arm|SPY>.{b} wash.after_tax_cagr; cost_bps_per_side; "
                f"beta_matched_pretax_cagr"
            )
            expo = (
                f"leverage from arm name '{arm}'; cost_bps_per_side={d['cost_bps_per_side']}; "
                f"note only: beta_matched_pretax_cagr={json.dumps(d['beta_matched_pretax_cagr'])}"
            )
            b12 = bar
            if i == "stocks-a24-sleeve-trend-rider":
                b12 = (
                    "original bar comparator: static book at the same average exposure, not SPY; "
                    + bar
                )
            rows.append(
                stat_row(
                    f"{i}/{w}/{b}",
                    f"{A['window'][0]}..{A['window'][1]}",
                    f2(c3),
                    f2(c4)
                    + " (TR basis: driver builds SPY total return from unadjusted closes plus "
                    "distributions, scripts/static_leverage_aftertax.py:93-94)",
                    f2(c3 - c4),
                    None,
                    None,
                    f"{b}: {at:.2f}; {b} wash (wash-sale sensitivity): {atw:.2f}",
                    b12,
                    keys,
                    expo,
                    "none reported",
                )
            )
    return rows


def h_century(i, art, d, bar, r):
    sc = "0.84%|24/15|wash=False|shiller"
    W = d["scenarios"][sc]["30"]["full"]
    parts = []
    for k in (
        "0.84%|24/15|wash=False|shiller",
        "0.84%|24/15|wash=True|shiller",
        "0.84%|40.8/23.8|wash=False|shiller",
        "0.84%|40.8/23.8|wash=True|shiller",
    ):
        g = (
            d["scenarios"][k]["30"]["full"]["after_tax_cagr"]
            - d["scenarios"][k]["0"]["full"]["after_tax_cagr"]
        ) * 100
        parts.append(f"{k}: {g:.2f}")
    after = (
        "comparator: 0% arm on the Ken French all-US market (provenance.market_proxy), "
        "not SPY; 30% minus 0% after-tax CAGR, full window: " + "; ".join(parts)
    )
    keys = (
        f"{art}; scenarios.<scenario>.30.full.{{window,after_tax_cagr}}; "
        f"scenarios.<scenario>.0.full.after_tax_cagr; provenance.market_proxy; "
        f"recommendation_rule.proposal"
    )
    return [
        stat_row(
            f"{i}/30/full",
            f"{W['window'][0]}..{W['window'][1]}",
            nc("pre-tax CAGR; scenarios.*.30.full holds after_tax_cagr only"),
            "comparator: 0% arm (Ken French value-weighted all-US market), not SPY; "
            + nc("pre-tax CAGR"),
            nc("pre-tax CAGR"),
            None,
            None,
            after,
            bar + " (selection: brief "
            "research/allocation/century-leverage/2026-10-23-century-leverage-brief.md §6, primary "
            "24/15, wash=False, 0.84% fee, Shiller distributions; arm 30 = the kept arm)",
            keys,
            "arm 30 = 30% SSO (2x fund) / 70% 1x market, band ±8 pp (brief); arm 0 = 1x "
            "buy-and-hold",
            "none reported",
        )
    ]


def h_core_spy(i, art, d, bar, r):
    A, S = d["live_k1.0"], d["spy"]
    keys = (
        f"{art}; meta.eval_start; meta.eval_end; live_k1.0.cagr_pct; spy.cagr_pct; "
        f"live_k1.0.mean_spy_weight; core_idle_mean_pct; meta.reserve; decision"
    )
    return [
        stat_row(
            f"{i}/live_k1.0",
            f"{d['meta']['eval_start']}..{d['meta']['eval_end']}",
            f2(A["cagr_pct"]),
            f2(S["cagr_pct"]) + " (" + LENS1_BASIS + ")",
            f2(A["cagr_pct"] - S["cagr_pct"]),
            None,
            None,
            AT_NONE,
            f"{r['verdict']}; bar: report §1 decision rule (k=1 Sharpe above SPY and MaxDD no "
            "worse; "
            "no t or CAGR bar), "
            "research/allocation/core-spy-blend/2026-09-24-core-spy-blend-keep.md; artifact "
            f"decision={d['decision']}; Lens 7: no earlier committed copy of the rule",
            keys,
            f"live_k1.0.mean_spy_weight={A['mean_spy_weight']:.4f}; core_idle_mean_pct="
            f"{d['core_idle_mean_pct']:.2f}; meta.reserve={d['meta']['reserve']}",
            "none reported",
        )
    ]


def h_reruns(i, art, d, bar, r):
    key = "xs_momentum" if i == "stocks-xs-momentum-corrected-replay" else "dual_momentum"
    S = d[key]
    E = S["excess_stats"]
    keys = (
        f"{art}; {key}.eval_start; {key}.eval_end; {key}.oos_metrics.cagr_pct; "
        f"{key}.oos_metrics.spy_sharpe; {key}.excess_stats.{{cagr_diff_pp,ann_excess_pct,t_stat,"
        f"n_obs,nw_lags,t_bar,cagr_bar_pp}}"
    )
    b = (
        f"{r['verdict']}; bar: t >= {E['t_bar']} and CAGR gap >= +{E['cagr_bar_pp']} pp/yr "
        f"({key}.excess_stats.t_bar, .cagr_bar_pp; brief "
        f"research/audits/production-replica/2026-10-19-production-replica-audit.md §6); artifact "
        f"{key}.verdict={S['verdict']}"
    )
    return [
        stat_row(
            f"{i}/{key}",
            f"{S['eval_start']}..{S['eval_end']}",
            f2(S["oos_metrics"]["cagr_pct"]),
            nc(f"SPY CAGR; only {key}.oos_metrics.spy_sharpe"),
            f2(E["cagr_diff_pp"]) + " (reported directly: excess_stats.cagr_diff_pp)",
            E["ann_excess_pct"],
            E["t_stat"],
            AT_NONE,
            b,
            keys,
            "not recorded in artifact",
            f"Newey-West, {E['nw_lags']} lags ({key}.excess_stats.nw_lags), daily OOS returns, "
            f"n_obs {E['n_obs']} (out-of-sample fold span); tested quantity: mean daily strategy "
            "minus "
            f"SPY return, annualized x252; SE = ann_excess_pct / t_stat",
        )
    ]


def h_netiss(i, art, d, bar, r):
    inp = d["inputs"]
    H = d["H1"]
    pre = "comparator: Fama-French market (Mkt-RF + RF), not SPY; "
    b = (
        f"{r['verdict']}; bar: A1 NW(12) t >= {inp['A1_t_bar']} and one-sided p <= "
        f"{inp['A1_p_bar']} "
        f"(single test) AND A2 >= +{inp['A2_bar_pp']} pp/yr (inputs.*; brief "
        "research/stock-signals/net-issuance/2026-10-23-net-issuance-screen.md); "
        "verdict fields H1.A1.pass="
        f"{H['A1']['pass']}, H1.A2.pass={H['A2']['pass']}, H1.overall_pass={H['overall_pass']}"
    )
    fw, aw = inp["full_window"], inp["A2_window"]
    mm = (
        f"not computable (window mismatch: {fw['start']}..{fw['end']} vs "
        f"{aw['start']}..{aw['end']})"
    )
    mm2 = (
        f"not computable (window mismatch: {aw['start']}..{aw['end']} vs "
        f"{fw['start']}..{fw['end']})"
    )
    se = (
        f"Newey-West, {inp['nw_lags']} lags (inputs.nw_lags), monthly, net of base cost "
        f"{inp['base_cost']}; tested quantity: mean monthly net excess of BIG NegNI vs market"
    )
    r1 = dict(
        zip(
            COLS,
            [
                f"{i}/H1/A1",
                f"{fw['start']}..{fw['end']}",
                nc("CAGR fields"),
                pre + nc("CAGR fields"),
                mm,
                nc("annualized mean excess"),
                f2(H["A1"]["t"]),
                "clears" if H["A1"]["t"] >= 3 else "does not clear",
                "clears" if H["A1"]["t"] >= 2 else "does not clear",
                AT_NONE,
                nc("annualized mean excess"),
                b,
                f"{art}; inputs.full_window.*; inputs.nw_lags; H1.A1.t; H1.A1.p",
                "not recorded in artifact",
                se,
            ],
        )
    )
    r2 = dict(
        zip(
            COLS,
            [
                f"{i}/H1/A2",
                f"{aw['start']}..{aw['end']}",
                nc("CAGR fields"),
                pre + nc("CAGR fields"),
                f2(H["A2"]["cagr_diff_pp"]),
                mm2,
                mm2,
                mm2,
                mm2,
                AT_NONE,
                mm2,
                b,
                f"{art}; inputs.A2_window.*; H1.A2.cagr_diff_pp",
                "not recorded in artifact",
                se,
            ],
        )
    )
    return [r1, r2]


def h_factor(i, art, d, bar, r):
    inp = d["inputs"]
    cw, aw = inp["common_window"], inp["A2_window"]
    pre = "comparator: Fama-French market (Mkt-RF + RF), not SPY; "
    mm = "not computable (window mismatch: common_window vs A2_window)"
    mm2 = "not computable (window mismatch: A2_window vs common_window)"
    rows = []
    for g, H in d["hypotheses"].items():
        b = (
            f"{r['verdict']}; bar: A1 Holm across four at one-sided alpha {inp['holm_alpha']} "
            f"(this hypothesis' Holm threshold {H['A1']['holm_threshold']}) AND A2 >= "
            f"+{inp['A2_bar_pp']} pp/yr (brief "
            "research/stock-signals/factor-screen/2026-10-20-factor-screen.md "
            "§6); "
            f"pass_holm={H['A1']['pass_holm']}, A2.pass={H['A2']['pass']}, "
            f"overall_pass={H['overall_pass']}"
        )
        se = (
            f"Newey-West, {inp['nw_lags']} lags (inputs.nw_lags), monthly, base cost; tested "
            f"quantity: mean monthly net excess vs Fama-French market"
        )
        t = H["A1"]["t"]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{g}/A1",
                        f"{cw['start']}..{cw['end']}",
                        nc("CAGR fields"),
                        pre + nc("CAGR fields"),
                        mm,
                        nc("annualized mean excess"),
                        f2(t),
                        "clears" if t >= 3 else "does not clear",
                        "clears" if t >= 2 else "does not clear",
                        AT_NONE,
                        nc("annualized mean excess"),
                        b,
                        f"{art}; inputs.common_window.*; inputs.nw_lags; "
                        f"inputs.holm_alpha; hypotheses.{g}.A1.{{t,p,holm_threshold,pass_holm}}",
                        "not recorded in artifact",
                        se,
                    ],
                )
            )
        )
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{g}/A2",
                        f"{aw['start']}..{aw['end']}",
                        nc("CAGR fields"),
                        pre + nc("CAGR fields"),
                        f2(H["A2"]["cagr_diff_pp"]),
                        mm2,
                        mm2,
                        mm2,
                        mm2,
                        AT_NONE,
                        mm2,
                        b,
                        f"{art}; inputs.A2_window.*; inputs.A2_bar_pp; "
                        f"hypotheses.{g}.A2.{{cagr_diff_pp,pass}}",
                        "not recorded in artifact",
                        se,
                    ],
                )
            )
        )
    return rows


def h_momentum(i, art, d, bar, r):
    rows = []
    w1, w2 = "1927-01..2026-04", "1994-01..2026-04"
    note = (
        " (window from brief research/stock-signals/momentum/2026-10-20-momentum-retest.md §5; not "
        "stored in artifact)"
    )
    pre = "comparator: Fama-French market (Mkt-RF + RF, brief §5), not SPY; "
    for g in ("big", "small"):
        G = d["groups"][g]
        b = (
            f"{r['verdict']}; bar: A1 Holm across two at family-wise one-sided p 0.00135 "
            f"(stronger p <= 0.000675, about t >= 3.2; other p <= 0.00135) AND A2 >= +1.0 pp/yr "
            f"(brief §3, §5); pass.{g}={d['pass'][g]}, holm.{g}={d['holm'][g]}"
        )
        se = (
            "Newey-West, 12 lags (NW_LAGS_MONTHLY, src/ggTrader/lab/momentum_screen.py:17), "
            "monthly, base cost; tested quantity: mean monthly net excess vs Fama-French market"
        )
        t = G["full_t"]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{g}/A1",
                        w1 + note,
                        nc("CAGR fields"),
                        pre + nc("CAGR fields"),
                        f"not computable (window mismatch: {w1} vs {w2})",
                        nc("annualized mean excess"),
                        f2(t),
                        "clears" if t >= 3 else "does not clear",
                        "clears" if t >= 2 else "does not clear",
                        AT_NONE,
                        nc("annualized mean excess"),
                        b,
                        f"{art}; groups.{g}.full_t; groups.{g}.full_p; pass.{g}; holm.{g}",
                        "not recorded in artifact",
                        se,
                    ],
                )
            )
        )
        mm = f"not computable (window mismatch: {w2} vs {w1})"
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{g}/A2",
                        w2 + note,
                        nc("CAGR fields"),
                        pre + nc("CAGR fields"),
                        f2(G["post_cagr_diff_pp"]),
                        mm,
                        mm,
                        mm,
                        mm,
                        AT_NONE,
                        mm,
                        b,
                        f"{art}; groups.{g}.post_cagr_diff_pp; pass.{g}",
                        "not recorded in artifact",
                        se,
                    ],
                )
            )
        )
    return rows


def h_pit(i, art, d, bar, r):
    s = {
        "stocks-sp500-core-pit-rebaseline": "ensemble",
        "stocks-ensemble-ic-pit": "ensemble_ic",
        "stocks-ensemble-kelly-pit": "ensemble_kelly",
        "stocks-live-ensemble-defaults-vs-pit-picks": "ensemble",
    }[i]
    S = d[s]
    miss = nc(f"SPY CAGR; only {s}.spy_sharpe")
    ver = S.get("verdict")
    b = bar
    if ver:
        b = (
            f"{r.get('verdict')}; code-encoded checks {s}.verdict.checks="
            f"{json.dumps(ver['checks'])}, go={ver['go']}; n_gate_pass={S['n_gate_pass']} "
            f"(pre-DSR-fix count); report next per column 12 precedence"
        )
    else:
        b = bar + f"; artifact n_gate_pass={S['n_gate_pass']} (pre-DSR-fix count)"
    return [
        dict(
            zip(
                COLS,
                [
                    f"{i}/{s}",
                    f"{S['eval_start']}..{S['eval_end']}",
                    f2(S["cagr_pct"]),
                    miss,
                    nc(f"SPY CAGR; only {s}.spy_sharpe"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    AT_NONE,
                    nc("t-statistic"),
                    b,
                    f"{art}; {s}.eval_start; {s}.eval_end; {s}.cagr_pct; {s}.spy_sharpe; "
                    f"{s}.verdict.*; {s}.n_gate_pass",
                    "not recorded in artifact",
                    "none reported",
                ],
            )
        )
    ]


def h_simple_core(i, art, d, bar, r, key, extra_expo=None):
    S = d[key]
    expo = "not recorded in artifact"
    if key == "blend":
        expo = (
            f"blend.overlay_params: target_vol={S['overlay_params']['target_vol']}, "
            f"window={S['overlay_params']['window']}, max_leverage="
            f"{S['overlay_params']['max_leverage']}"
        )
    spy = nc(f"SPY CAGR; only {key}.spy_sharpe") if "spy_sharpe" in S else nc("SPY series")
    return [
        dict(
            zip(
                COLS,
                [
                    f"{i}/{key}",
                    f"{S['eval_start']}..{S['eval_end']}",
                    f2(S["cagr_pct"]),
                    spy,
                    nc("SPY CAGR"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    AT_NONE,
                    nc("t-statistic"),
                    bar
                    + (
                        f"; artifact {key}.n_gate_pass={S['n_gate_pass']}"
                        if "n_gate_pass" in S
                        else ""
                    ),
                    f"{art}; {key}.eval_start; {key}.eval_end; {key}.cagr_pct; "
                    + (f"{key}.spy_sharpe" if "spy_sharpe" in S else "")
                    + ("; blend.overlay_params.*" if key == "blend" else ""),
                    expo,
                    "none reported",
                ],
            )
        )
    ]


def h_anchorfix(i, art, d, bar, r):
    return h_simple_core(i, art, d, bar, r, "core")


def h_rebase(i, art, d, bar, r):
    key = "blend" if i == "stocks-live-revert-blend-to-core" else "core"
    return h_simple_core(i, art, d, bar, r, key)


def h_ickelly(i, art, d, bar, r):
    rows = []
    for key in ("ensemble_ic", "ensemble_kelly"):
        S = d[key]
        b = (
            f"{r['verdict']}; code-encoded checks "
            f"{key}.verdict.checks={json.dumps(S['verdict']['checks'])}, "
            f"go={S['verdict']['go']}; selection: step 3 (claim names both variants)"
        )
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{key}",
                        f"{S['eval_start']}..{S['eval_end']}",
                        f2(S["cagr_pct"]),
                        nc(f"SPY CAGR; only {key}.spy_sharpe"),
                        nc("SPY CAGR"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        b,
                        f"{art}; {key}.eval_start; {key}.eval_end; {key}.cagr_pct; "
                        f"{key}.spy_sharpe; {key}.verdict.*",
                        "not recorded in artifact",
                        "none reported",
                    ],
                )
            )
        )
    return rows


def h_possize(i, art, d, bar, r):
    rows = []
    for k, S in enumerate(d["results"]):
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{S['name']}",
                        f"{d['meta']['eval_start']}..{d['meta']['eval_end']}",
                        f2(S["cagr_pct"]),
                        nc(
                            "numeric SPY CAGR key; SPY appears only inside the text field "
                            f"results[{k}].table"
                        ),
                        nc("SPY CAGR"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        bar
                        + "; selection: step 3 (one sub-row per results[i]); note: this artifact "
                        "is the "
                        "gated-WFO follow-up (window 2023-10-30..2026-04-30), not the "
                        "single-window run the "
                        "report's headline quotes (Lens 7 match: no)",
                        f"{art}; meta.eval_start; meta.eval_end; "
                        f"results[{k}].{{name,regime,slot_cap,cagr_pct}}",
                        f"results[{k}].regime={S['regime']}; "
                        f"results[{k}].slot_cap={S['slot_cap']}; "
                        f"meta.sleeve={d['meta']['sleeve']}; max leverage not recorded in artifact",
                        "none reported",
                    ],
                )
            )
        )
    return rows


def h_replica(i, art, d, bar, r):
    w = "holdout" if i.endswith("holdout-2010-2020") else "primary"
    W = d["windows"][w]
    A = W["arms"]["primary"]
    E = A["excess_vs_spy"]
    b = (
        f"{r['verdict']}; bar: NW(5) t >= 3.0 AND CAGR gap >= +1.0 pp/yr before tax, both "
        f"windows, primary arm; after-tax >= +1.0 pp/yr at 24/15 only if those pass (brief "
        "research/audits/production-replica/2026-10-19-production-replica-audit.md §4); artifact "
        "verdict.verdict="
        f"{d['verdict']['verdict']} (reasons {d['verdict']['reasons']}), closure.verdict="
        f"{d['closure']['verdict']} (stage {d['closure']['stage']}), windows.{w}.arms.primary."
        f"pretax_pass={A['pretax_pass']}"
    )
    keys = (
        f"{art}; windows.{w}.eval_start; windows.{w}.eval_end; "
        f"windows.{w}.arms.primary.stats.cagr_pct; "
        f"windows.{w}.arms.primary.spy_stats.cagr_pct; windows.{w}.arms.primary.excess_vs_spy."
        f"{{cagr_diff_pp,ann_excess_pct,t_stat,n_obs}}; windows.{w}.arms.primary.mean_exposure.*; "
        f"windows.{w}.arms.primary.config.risk.position_pct; "
        f"windows.{w}.arms.primary.config.reserve_pct; "
        f"windows.{w}.arms.primary.excess_vs_exposure_matched.*; windows.{w}.citable; "
        f"windows.{w}.stop_flags; closure.holdout; verdict.*; closure.*"
    )
    me = A["mean_exposure"]
    xm = A["excess_vs_exposure_matched"]
    expo = (
        f"mean_exposure stocks={me['stocks']:.4f}, spy={me['spy']:.4f}, cash={me['cash']:.4f}; "
        f"config.risk.position_pct={A['config']['risk']['position_pct']}; config.risk."
        f"max_positions={A['config']['risk']['max_positions']}; config.reserve_pct="
        f"{A['config']['reserve_pct']}; note only: excess_vs_exposure_matched cagr_diff_pp="
        f"{xm['cagr_diff_pp']:.2f}, t_stat={xm['t_stat']:.2f}"
    )
    se = (
        f"Newey-West, 5 lags (NW_LAGS, src/ggTrader/lab/significance.py:14), daily, n_obs "
        f"{E['n_obs']}; tested quantity: mean daily strategy minus SPY return, annualized x252; "
        f"SE = ann_excess_pct / t_stat"
    )
    if w == "holdout":
        se += (
            f"; NOT CITABLE: windows.holdout.citable={W['citable']}, windows.holdout.stop_flags="
            f'{W["stop_flags"]}, closure.holdout="{d["closure"]["holdout"]}"'
        )
    else:
        se += (
            "; tape note: the primary arm traded PARA 3 round trips (trade_counts) while the "
            "tape's "
            "PARA series is a different security (Lens 1); tainted_trades is empty"
        )
    return [
        stat_row(
            f"{i}/{w}",
            f"{W['eval_start']}..{W['eval_end']}",
            f2(A["stats"]["cagr_pct"]),
            f2(A["spy_stats"]["cagr_pct"]) + " (" + LENS1_BASIS + ")",
            f2(E["cagr_diff_pp"]),
            E["ann_excess_pct"],
            E["t_stat"],
            "not computable from artifact (no after-tax fields; closure.after_tax is a text "
            'string: "' + d["closure"]["after_tax"] + '")',
            b,
            keys,
            expo,
            se,
        )
    ]


def h_fomc(i, art, d, bar, r):
    t = d["net"]["t_stat"]
    return [
        dict(
            zip(
                COLS,
                [
                    f"{i}/net",
                    f"{d['eval_start']}..{d['eval_end']}",
                    nc("CAGR fields"),
                    nc("SPY series; comparator is non-event days of the same Treasury ETFs"),
                    nc("CAGR fields"),
                    nc("annualized mean excess; artifact gives net.mean_diff per event"),
                    f2(t),
                    "clears" if t >= 3 else "does not clear",
                    "clears" if t >= 2 else "does not clear",
                    AT_NONE,
                    nc("annualized mean excess"),
                    bar + "; selection: step 1 (bar names the net event t)",
                    f"{art}; eval_start; eval_end; n_event_dates; "
                    f"net.{{t_stat,mean_diff,n_event,n_nonevent}}",
                    "not recorded in artifact",
                    "artifact's own statistic: Welch t (src/ggTrader/lab/event_study.py:84-99) on "
                    "returns "
                    "averaged by date (day-clustered, "
                    "scripts/fomc_clustered_event_study.py:93-104), net of "
                    f"round-trip cost; n_event_dates {d['n_event_dates']}; tested quantity: mean "
                    "event-day minus "
                    "mean non-event-day return per event, not annualized, not vs SPY; "
                    "overlap/fold dependence: "
                    "events are distinct meeting dates after clustering (driver). Lens 9 reports "
                    "a labelled "
                    "DERIVED annualized MDE for this row; not used here (column 6 needs an "
                    "annualized mean "
                    "excess in the artifact)",
                ],
            )
        )
    ]


def h_monthend(i, art, d, bar, r):
    rows = []
    P = d["primary"]
    vb = (
        f"{r['verdict']}; bar: brief "
        "research/allocation/month-end-treasury/2026-09-24-month-end-treasury-sleeve.md "
        f"pass bar criteria 1-5 (Sharpe/return/yearly, post-2019 positive, WFO gates >= 12/17 "
        f"no halt stable >= 8/17, portfolio Sharpe > SPY with MaxDD no worse, criteria 1 and 4 "
        f"at 3 bp; no t bar); artifact verdict.checks={json.dumps(d['verdict']['checks'])}"
    )
    for w in ("pinned", "post_pub"):
        Wd = P[w]
        spy = Wd["bh"]["SPY"]["cagr_pct"]
        for c in ("1bp", "3bp", "5bp"):
            A = Wd[c]
            t = A["welch_t_invested_vs_other"]
            rows.append(
                dict(
                    zip(
                        COLS,
                        [
                            f"{i}/primary/{w}/{c}",
                            f"{Wd['start']}..{Wd['end']}",
                            f2(A["cagr_pct"]),
                            f2(spy) + " (" + LENS1_BASIS + ")",
                            f2(A["cagr_pct"] - spy),
                            nc("annualized mean excess; artifact gives per-day bp means"),
                            f2(t),
                            "clears" if t >= 3 else "does not clear",
                            "clears" if t >= 2 else "does not clear",
                            AT_NONE,
                            nc("annualized mean excess"),
                            vb + "; selection: step 3 for cost (criterion names no cost for the "
                            "primary rule)",
                            f"{art}; primary.{w}.start; primary.{w}.end; "
                            f"primary.{w}.{c}.{{cagr_pct,"
                            f"welch_t_invested_vs_other,exposure_pct}}; "
                            f"primary.{w}.bh.SPY.cagr_pct; meta.*",
                            f"primary.{w}.{c}.exposure_pct={A['exposure_pct']:.2f}; "
                            "meta.signal_position_size="
                            f"{d['meta']['signal_position_size']}; "
                            f"meta.cash_rate={d['meta']['cash_rate']}",
                            "artifact's own statistic: Welch t on daily IEF close-to-close "
                            "returns, invested "
                            "days vs other days (scripts/month_end_treasury_wfo.py:82-88), gross "
                            "of cost; "
                            "tested quantity: mean invested-day minus mean other-day return, per "
                            "day, not vs "
                            "SPY; overlap/fold dependence: not stated (invested days are three "
                            "consecutive "
                            "trading days a month)",
                        ],
                    )
                )
            )
    Wf = d["wfo"]
    rows.append(
        dict(
            zip(
                COLS,
                [
                    f"{i}/wfo",
                    f"{d['meta']['eval_start']}..{d['meta']['eval_end']} (meta; WFO OOS folds "
                    "start later, not stored under wfo)",
                    f2(Wf["oos"]["cagr_pct"]),
                    f2(Wf["spy"]["cagr_pct"]) + " (" + LENS1_BASIS + ")",
                    f2(Wf["oos"]["cagr_pct"] - Wf["spy"]["cagr_pct"]),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    AT_NONE,
                    nc("t-statistic"),
                    vb + "; criterion 3 window",
                    f"{art}; meta.eval_start; meta.eval_end; wfo.oos.cagr_pct; "
                    "wfo.spy.cagr_pct; wfo.{n_gate_pass,halted_any,n_used_anchor}",
                    f"meta.wfo_cost={d['meta']['wfo_cost']}; "
                    f"wfo.n_used_anchor={Wf['n_used_anchor']}; "
                    f"wfo.halted_any={Wf['halted_any']}; max leverage not recorded",
                    "none reported",
                ],
            )
        )
    )
    for c in ("1bp", "3bp"):
        Pp = d["portfolio"]["pinned"][c]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/portfolio/pinned/{c}",
                        f"{d['meta']['eval_start']}..{d['meta']['eval_end']} (meta; "
                        "portfolio block stores no window)",
                        f2(Pp["portfolio"]["cagr_pct"]),
                        f2(Pp["spy"]["cagr_pct"]) + " (" + LENS1_BASIS + ")",
                        f2(Pp["portfolio"]["cagr_pct"] - Pp["spy"]["cagr_pct"]),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        vb + "; criteria 4 and 5",
                        f"{art}; portfolio.pinned.{c}.portfolio.cagr_pct; "
                        f"portfolio.pinned.{c}.spy.cagr_pct",
                        "80% SPY fixed + 20% rotating SPY->IEF for the last 3 trading days "
                        "(driver docstring "
                        "scripts/month_end_treasury_wfo.py:10-11); weights not stored as artifact "
                        "fields",
                        "none reported",
                    ],
                )
            )
        )
    return rows


def h_xasset(i, art, d, bar, r):
    rows = []
    basis = f'TR basis from artifact field meta.prices = "{d["meta"]["prices"]}"'
    if i == "cross-asset-static-80-20-sleeve":
        subs = [
            ("pinned", "1bp"),
            ("pinned", "3bp"),
            ("long", "1bp"),
            ("long", "3bp"),
            ("y2022", "1bp"),
        ]
        arm = "static"
        b = (
            bar + "; selection: step 1 (brief "
            "research/allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve-brief.md "
            "pass bar names pinned, 2016-2026 and 2022 windows at the primary 1 bp, and pinned and "
            "2016-2026 at 3 bp); artifact verdict.static_checks="
            + json.dumps(d["verdict"]["static_checks"])
        )
    else:
        subs = [("pinned", "1bp")]
        arm = "trend"
        b = (
            "original bar comparator: the static 80/20 arm (Sharpe +0.05 or MaxDD +3 pts on the "
            "pinned window) plus WFO gates, not SPY; "
            + bar
            + "; artifact verdict.trend_checks="
            + json.dumps(d["verdict"]["trend_checks"])
        )
    for w, c in subs:
        Wd = d[f"primary:{w}"]
        A = Wd[c][arm]
        spy = Wd["bh"]["SPY"]["cagr_pct"]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{w}/{c}",
                        f"{Wd['start']}..{Wd['end']}",
                        f2(A["cagr_pct"]),
                        f2(spy) + f" ({basis})",
                        f2(A["cagr_pct"] - spy),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        b,
                        f"{art}; primary:{w}.start; primary:{w}.end; "
                        f"primary:{w}.{c}.{arm}.cagr_pct; "
                        f"primary:{w}.bh.SPY.cagr_pct; meta.prices; verdict.*",
                        f"arm '{arm}' (80% SPY + 20% TLT/GLD/PDBC per brief); "
                        f'meta.rebalance="{d["meta"]["rebalance"]}"; '
                        "leg weights not stored as fields",
                        "none reported",
                    ],
                )
            )
        )
    if arm == "trend":
        Wf = d["wfo"]
        ww = d["meta"]["windows"]["wfo_oos"]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/wfo",
                        f"{ww[0]}..{ww[1]}",
                        f2(Wf["oos"]["cagr_pct"]),
                        f2(Wf["spy"]["cagr_pct"]) + f" ({basis})",
                        f2(Wf["oos"]["cagr_pct"] - Wf["spy"]["cagr_pct"]),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        b,
                        f"{art}; meta.windows.wfo_oos; wfo.oos.cagr_pct; wfo.spy.cagr_pct; "
                        "wfo.{n_gate_pass,n_used_anchor,halted_any}",
                        f"meta.wfo_cost={d['meta']['wfo_cost']}; "
                        f"wfo.n_used_anchor={Wf['n_used_anchor']}; "
                        f"wfo.halted_any={Wf['halted_any']}; max_leverage 1.0 per brief (not an "
                        "artifact field)",
                        "none reported",
                    ],
                )
            )
        )
    return rows


def h_holdout(i, art, d, bar, r):
    rows = []
    for c in ("1bp", "3bp"):
        A = d[c]["static"]
        spy = d["bh"]["SPY"]["cagr_pct"]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{c}",
                        f"{d['window']['start']}..{d['window']['end']}",
                        f2(A["cagr_pct"]),
                        f2(spy) + f' (TR basis from artifact field prices = "{d["prices"]}")',
                        f2(A["cagr_pct"] - spy),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        nc("t-statistic"),
                        AT_NONE,
                        nc("t-statistic"),
                        bar
                        + "; selection: step 1 (checks 1-2 at the primary 1 bp, check 3 at 3 bp); "
                        "artifact "
                        f"verdict.checks={json.dumps(d['verdict']['checks'])}, "
                        f'verdict.reading="{d["verdict"]["reading"]}"',
                        f"{art}; window.*; {c}.static.cagr_pct; bh.SPY.cagr_pct; prices; verdict.*",
                        "static 80/20 (SPY + TLT/GLD/DBC per brief); weights not stored as fields",
                        "none reported",
                    ],
                )
            )
        )
    return rows


def h_ibit(i, art, d, bar, r):
    rows = []
    for s in ("intraday", "weeknight", "weekend"):
        t = d["t_stat"][s]
        rows.append(
            dict(
                zip(
                    COLS,
                    [
                        f"{i}/{s}",
                        f"{d['window'][0]}..{d['window'][1]}",
                        nc("CAGR fields"),
                        nc(
                            "comparator; artifact tests absolute IBIT session returns (no SPY, "
                            "BTC or ETH series)"
                        ),
                        nc("CAGR fields"),
                        nc("annualized mean excess; artifact gives mean_bp per session"),
                        f2(t),
                        "clears" if t >= 3 else "does not clear",
                        "clears" if t >= 2 else "does not clear",
                        AT_NONE,
                        nc("annualized mean excess"),
                        bar + "; selection: step 3 (one sub-row per session type with a t; "
                        "overnight_all has no t)",
                        f"{art}; window; mean_bp.{s}; t_stat.{s}; n_days; n_weeknight; n_weekend",
                        "not recorded in artifact",
                        "artifact's own statistic: mean/std*sqrt(n) per session type "
                        "(research/crypto/artifacts-2026-10-25-crypto-web-batch/"
                        "ibit_overnight_split.py:45-46), iid; "
                        "tested quantity: mean IBIT session return in bp, absolute; overlap/fold "
                        "dependence: not stated",
                    ],
                )
            )
        )
    return rows


def h_crypto_snap(i, art, d, bar, r):
    return [
        dict(
            zip(
                COLS,
                [
                    i,
                    nc(
                        "window (Lens 1 cites --days 1095 --end-date 2026-05-01 from run config, "
                        "not stored here)"
                    ),
                    nc("CAGR fields; per-symbol WFO fold stats only"),
                    nc("benchmark series"),
                    nc("CAGR fields"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    nc("t-statistic"),
                    AT_NONE,
                    nc("t-statistic"),
                    bar,
                    f"{art}; [i].{{symbol,strategy,exit,payload}} (231 rows)",
                    "not recorded in artifact",
                    "none reported",
                ],
            )
        )
    ]


HANDLERS = {
    "research/allocation/static-leverage/_static_leverage_aftertax_results.json": h_static,
    "research/allocation/century-leverage/_century_leverage_results.json": h_century,
    "research/allocation/core-spy-blend/_core_spy_blend_20260924.json": h_core_spy,
    "research/audits/production-replica/_audit_reruns_20261019.json": h_reruns,
    "research/stock-signals/net-issuance/_net_issuance_screen_results.json": h_netiss,
    "research/stock-signals/factor-screen/_factor_screen_results.json": h_factor,
    "research/stock-signals/momentum/_momentum_french_screen_results.json": h_momentum,
    "research/audits/rebaseline/_pit_rebaseline_20260923.json": h_pit,
    "research/audits/rebaseline/_anchor_fix_reproduction_results.json": h_anchorfix,
    "research/audits/rebaseline/_rebaseline_corrected_tape_20260822.json": h_rebase,
    "research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json": h_ickelly,
    "research/paper-system/position-sizing/_position_sizing_wfo_results.json": h_possize,
    "research/audits/production-replica/_production_replica_results.json": h_replica,
    "research/audits/production-replica/_fomc_clustered_event_study_20261020.json": h_fomc,
    "research/allocation/month-end-treasury/_month_end_treasury_results.json": h_monthend,
    "research/allocation/cross-asset-sleeve/_cross_asset_sleeve_results.json": h_xasset,
    "research/allocation/cross-asset-sleeve/_cross_asset_sleeve_holdout_results.json": h_holdout,
    "research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json": h_ibit,
    "results/research/research_20260511_193048/wfo_stats_snapshot.json": h_crypto_snap,
}

# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
FAMILY_RULES = [
    ("crypto", lambda i: i.startswith("crypto-")),
    ("momentum", lambda i: "momentum" in i or "wfo-tournament" in i),
    (
        "factor/anomaly screens",
        lambda i: any(k in i for k in ("factor-screen", "idio-vol", "max-effect", "net-issuance")),
    ),
    (
        "event and information signals",
        lambda i: any(
            k in i
            for k in (
                "pead",
                "insider-cluster",
                "congress",
                "short-interest",
                "short-volume",
                "index-deletion",
                "headline-sentiment",
                "july-research",
            )
        ),
    ),
    (
        "calendar and macro timing",
        lambda i: any(k in i for k in ("fomc", "month-end", "turn-of-month")),
    ),
    ("leveraged ETF timing", lambda i: "leveraged-" in i),
    (
        "macro approximations",
        lambda i: any(k in i for k in ("treasury-curve", "fx-hedge", "commodity-trend")),
    ),
    (
        "allocation and leverage",
        lambda i: any(
            k in i for k in ("a23", "a24", "century", "80-20", "sleeve-holdout", "sleeve-trend-arm")
        ),
    ),
    ("pairs", lambda i: "pairs" in i),
    ("gate-model study", lambda i: "wfo-policy" in i),
    ("ensemble/reversion and its levers", lambda i: i.startswith("stocks-")),
]


def family(i):
    for name, f in FAMILY_RULES:
        if f(i):
            return name
    return "other"


ALPHAS = {
    "t>=2 one-sided": 0.022750131948179195,
    "t>=3 one-sided": 0.0013498980316301035,
    "t>=2 two-sided": 0.04550026389635842,
    "t>=3 two-sided": 0.0026997960632601866,
}


def mt(n):
    return {
        k: {
            "expected_false_positives": round(n * a, 3),
            "p_at_least_one": round(1 - (1 - a) ** n, 3),
        }
        for k, a in ALPHAS.items()
    }


POST_FIX_GATED = {
    "stocks-xs-momentum-corrected-replay": "run_wfo rerun; n_gate_pass 0/17, anchor every fold",
    "stocks-dual-momentum": "run_wfo rerun; n_gate_pass 0/17",
    "cross-asset-fomc-drift-corrected-harness": "bar includes WFE >= 0.50 from the rerun WFO "
    "(_audit_reruns_20261019.json fomc_drift.wfo.aggregate_wfe)",
}

RESEARCHER_FREEDOM: list[dict[str, str]] = [
    {
        "date": "2026-06-17",
        "commit": "0db40de6",
        "change": "WFO windows 3yr/1yr -> 12mo/3mo",
        "direction": "more, shorter OOS folds (noisier per-fold Sharpe; feeds the "
        "2-negative-Sharpe "
        "breaker trigger)",
        "affects": "every run_wfo row from 2026-06-17",
    },
    {
        "date": "2026-06-17",
        "commit": "be405aae, 57bd66ba, af49c0e6, 2bea599c, fa8ec4b5",
        "change": "NDH + DSR gates, circuit breaker, shadow re-entry and anchor fallback "
        "introduced",
        "direction": "tighter (more anchor deployment)",
        "affects": "every run_wfo row",
    },
    {
        "date": "2026-06-24",
        "commit": "3cd98734",
        "change": "NDH neighbourhood Moore -> axis-aligned; regime params excluded",
        "direction": "looser (commit: 3-voter 0/17 -> 11/17 folds pass)",
        "affects": "stocks-voter-ablation-5-voter, stocks-three-pct-position-size, "
        "stocks-ensemble-ndh-baseline-1-12 and later ensemble WFO rows",
    },
    {
        "date": "2026-06-26",
        "commit": "431de4ec",
        "change": "NDH variance cap exempt at density 1.0",
        "direction": "looser (changelog: 13/17 -> 16/17), made the same day the 1.12 headline was "
        "recorded",
        "affects": "stocks-ensemble-ndh-baseline-1-12 and every later ensemble WFO row",
    },
    {
        "date": "2026-06-27",
        "commit": "d3d0c875",
        "change": "shadow re-entry 2 consecutive -> 2 of last 3 clean windows",
        "direction": "looser (commit: MidCap anchor folds 15 -> 10, gates 6 -> 11/17)",
        "affects": "stocks-midcap400-reversion, stocks-nasdaq100-single-core, "
        "stocks-equity-diversification-sleeves, stocks-three-sleeve-blend-1-14 and later run_wfo "
        "rows",
    },
    {
        "date": "2026-06-20 to 2026-08-19",
        "commit": "src/ggTrader/lab/cli.py:70 (--eval-end default None)",
        "change": "evaluation end not pinned (defaults to now) until the 2026-08-19 re-baseline "
        "pinned "
        "2021-01-31..2026-04-30",
        "direction": "window drift; benchmark SPY Sharpe moved 0.58 -> 0.78",
        "affects": "stocks-ensemble-ndh-baseline-1-12 (unrecoverable window) and the July 'to "
        "present' "
        "NO-GOs (Lens 1 §(c))",
    },
    {
        "date": "2026-07 (July studies)",
        "commit": "research/prompts/local-implementation-prompt-TEMPLATE.md",
        "change": "comparator for July studies = the prompt's 'Current baseline -- the bar to "
        "beat' "
        "(Lens 7), not a per-study pre-registered bar",
        "direction": "bar set by a moving internal baseline (1.12, later voided)",
        "affects": "July event-study and ETF WFO rows (Lens 7 status 'no brief found')",
    },
    {
        "date": "2026-07-25",
        "commit": "1b9b7f56",
        "change": "SPY index contamination collapsed (research side)",
        "direction": "benchmark Sharpe raised (0.58 contaminated -> ~0.78-0.91)",
        "affects": "all rows comparing to SPY Sharpe before 2026-07-25",
    },
    {
        "date": "2026-08-18",
        "commit": "2eafab18",
        "change": "anchor leak, PIT eligibility, survivorship fixes",
        "direction": "tighter (removes strategy-favouring leaks)",
        "affects": "rows run before 2026-08-18 (Lens 1 anchor_leak tag)",
    },
    {
        "date": "2026-09-23",
        "commit": "c9d1abe9",
        "change": "PIT membership for signal sweeps",
        "direction": "tighter (core 0.99 -> 0.59 Sharpe)",
        "affects": "rows tagged pit_universe",
    },
    {
        "date": "2026-09-24 to 2026-09-28",
        "commit": "briefs 2026-09-24-month-end, 2026-09-28-cross-asset",
        "change": "Sharpe-vs-SPY point-estimate bars with rf = 0 (meta.sharpe_basis)",
        "direction": "a bar with no significance threshold; a GO on Sharpe while CAGR trails SPY",
        "affects": "cross-asset-static-80-20-sleeve, cross-asset-sleeve-holdout-2007-2011, "
        "stocks-core-spy-blend-keep, cross-asset-month-end-treasury",
    },
    {
        "date": "2026-10-19",
        "commit": "09f9f401",
        "change": "DSR units fix",
        "direction": "much "
        "tighter (anchor deployed in 17/17 corrected momentum folds, "
        "research/programme/wfo-methods/_wfo_fold_attribution_results.json)",
        "affects": "every run_wfo row before 2026-10-19 (Lens 1 dsr_units tag, 51 rows)",
    },
    {
        "date": "2026-10-19",
        "commit": "3ae4a125; a89f31c2; 7320c1d6",
        "change": "NW t >= 3 and "
        "+1.0 pp/yr bars (significance.py:13-15); replica parity amendment approved 4 minutes "
        "after "
        "the 10/32 parity result; FOMC timing amendment",
        "direction": "stricter significance bar; amendments made after partial results (Lens 7)",
        "affects": "stocks-production-replica, "
        "stocks-production-replica-live-parity, stocks-xs-momentum-corrected-replay, "
        "stocks-dual-momentum, cross-asset-fomc-drift-corrected-harness",
    },
    {
        "date": "2026-10-20",
        "commit": "bf0e8279 (Lens 1)",
        "change": "FOMC re-analysed day-clustered after the first result",
        "direction": "conservative (net t 1.57 -> 0.98)",
        "affects": "cross-asset-fomc-drift-corrected-harness",
    },
    {
        "date": "2026-10-20 to 2026-10-21",
        "commit": "briefs 2026-10-20-momentum-retest, 2026-10-20-factor-screen",
        "change": "Holm families and post-1994 +1.0 pp/yr bar; factor hypotheses chosen after the "
        "momentum result (Lens 7); momentum brief committed with its results (pre-registered-late)",
        "direction": "strict; time order not provable for momentum",
        "affects": "stocks-momentum-french-screen, stocks-factor-screen-h1-h4",
    },
    {
        "date": "2026-10-23",
        "commit": "century rulings r1, r2",
        "change": "end date 2026-08-31 -> 2026-06-30 (data forced), reproduction gate",
        "direction": "neutral (made before results per ruling text)",
        "affects": "stocks-century-leverage",
    },
    {
        "date": "2026-05-11",
        "commit": "docs/changelog.md (MIN_TRADES_PER_TRAIN_FOLD 30 -> 19)",
        "change": "crypto trade-count gate lowered, pre-registered",
        "direction": "looser; result still empty",
        "affects": "crypto-wfo-textbook-reset",
    },
    {
        "date": "default",
        "commit": "src/ggTrader/lab/cli.py:97",
        "change": "--max-leverage default 2.0 vs production 1.0",
        "direction": "a blend quoted at the default overstates the deployed config",
        "affects": "blend rows; the artifacts that record it show 1.0 "
        "(_rebaseline_corrected_tape_20260822.json blend.overlay_params.max_leverage)",
    },
]


# Markdown fragments
def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def build(out_json: Path, md_dir: Path | None) -> dict[str, Any]:
    """Classify every verdict row, build the sensitivity table and write the outputs."""
    classes = [classify(r) for r in inv if r.get("verdict")]
    assert len(classes) == 114 and len({c["id"] for c in classes}) == 114

    sens = []
    for r in inv:
        sens.extend(sens_rows_for(r))

    for s in sens:
        assert list(s.keys()) == COLS, s

    tests = [c for c in classes if c["verdict_type"] in ("NO", "GO", "TEST-INCONCLUSIVE")]
    fam = {}
    for c in tests:
        fam.setdefault(family(c["id"]), []).append(c["id"])

    fd = load(FACTOR_SCREEN_JSON)
    md = load(MOMENTUM_SCREEN_JSON)
    nd = load(NET_ISSUANCE_JSON)
    n_tests = len(tests)
    n_hyp_extra = 3 + 1  # factor 4 hyps counted as 1 row (+3), momentum 2 groups as 1 row (+1)
    multiple_testing = {
        "counting_rule": "A test = one inventory row whose verdict is a measured test outcome "
        "(verdict_type NO, GO or TEST-INCONCLUSIVE under the fixed classification rules); rows "
        "with status labels are not tests. Registered multi-hypothesis rows add their extra "
        "hypotheses (factor screen 4 hypotheses = +3; momentum screen 2 groups = +1). In-WFO "
        "parameter search (e.g. 48 combos x 17 folds for the ensemble) is not counted as separate "
        "tests. Git cross-check: commit subjects containing 'NO-GO'.",
        "n_test_rows": n_tests,
        "n_test_rows_by_type": {
            t: sum(1 for c in tests if c["verdict_type"] == t)
            for t in ("NO", "GO", "TEST-INCONCLUSIVE")
        },
        "n_hypotheses_incl_registered_multi": n_tests + n_hyp_extra,
        "n_families": len(fam),
        "families": {k: {"n": len(v), "ids": v} for k, v in fam.items()},
        "git_commits_total": 1093,
        "git_commit_subjects_with_NO_GO": 32,
        "strategy_registry_entries_per_CLAUDE_md": 38,
        "chance_wins_if_all_independent_and_null": {
            "N_test_rows": mt(n_tests),
            "N_hypotheses": mt(n_tests + n_hyp_extra),
            "N_families": mt(len(fam)),
        },
        "holm_and_thresholds_used_by_studies": {
            "factor_screen": {
                "inputs.holm_alpha": fd["inputs"]["holm_alpha"],
                "per_rank_thresholds": {
                    g: fd["hypotheses"][g]["A1"]["holm_threshold"] for g in fd["hypotheses"]
                },
                "source": "research/stock-signals/factor-screen/_factor_screen_results.json",
            },
            "momentum_screen": {
                "family_wise_one_sided_p": 0.00135,
                "first_step": 0.000675,
                "second_step": 0.00135,
                "full_p": {g: md["groups"][g]["full_p"] for g in md["groups"]},
                "holm": md["holm"],
                "source": "brief research/stock-signals/momentum/2026-10-20-momentum-retest.md §5; "
                "_momentum_french_screen_results.json groups.*.full_p, holm.*",
            },
            "net_issuance": {
                "inputs.A1_p_bar": nd["inputs"]["A1_p_bar"],
                "A1_t_bar": nd["inputs"]["A1_t_bar"],
                "holm": "none (single registered hypothesis)",
            },
            "replica_and_reruns": {
                "T_BAR": 3.0,
                "CAGR_BAR_PP": 1.0,
                "source": "src/ggTrader/lab/significance.py:13-15",
                "holm": "none",
            },
            "fomc": {"bar": "net t >= 3.0 and WFE >= 0.50", "holm": "none"},
            "point_estimate_bars": "Most other GO/NO-GO bars (Sharpe > SPY, MaxDD no worse, beat a "
            "baseline Sharpe, WFO gate counts) carry no significance threshold at all",
        },
    }

    gated = []
    for r in inv:
        if not r.get("verdict") or verdict_type(r["verdict"]) == "STATUS":
            continue
        tags = set(L1.get(r["id"], {}).get("contamination") or [])
        # Lens 1 tags anchor_leak and dsr_units only on run_wfo results (fix timeline rows 3, 5)
        if tags & {"anchor_leak", "dsr_units"} or r["id"] in POST_FIX_GATED:
            gated.append(r["id"])
    gate_machinery = {
        "circuit_breaker": "check_circuit_breaker (src/ggTrader/lab/wfo.py:79-125): halt if the "
        "trailing 4-window average WFE < 0.25 or two consecutive negative OOS Sharpes; while "
        "halted "
        "or when NDH/DSR gates fail, the fold deploys the anchor (min-drawdown) parameters "
        "(wfo.py:747-752); re-entry needs 2 of the last 3 windows clean (NDH, DSR and WFE >= 0.5; "
        "wfo.py:128-165).",
        "gates": "NDH: density >= 0.85 of neighbours profitable and Sharpe dispersion <= 0.20 of "
        "peak "
        "unless density == 1.0 (gates.py:57-117); DSR >= 0.80 against expected max Sharpe over the "
        "grid (gates.py:190-215), corrected for annual/per-day units in 09f9f40.",
        "synthetic_halt_rates": {
            "P_halt_legacy_mu1.0_pgate1.0": 0.875,
            "P_halt_mu0": 0.984,
            "P_halt_mu1.5_pgate1.0": 0.729,
            "anchor_share_mu1.0_pgate1.0": 0.341,
            "anchor_share_mu1.0_pgate0.3": 0.868,
            "runs_ending_halted_mu1.0_pgate1.0": 0.496,
            "source": "research/programme/wfo-methods/2026-11-02-wfo-policy-synthetic.md §5.1-5.3; "
            "_wfo_policy_synthetic_results.json",
            "limits": "synthetic; real NDH/DSR not modelled; 2.0 OOS Sharpe s.d.",
        },
        "verdict_rows_resting_on_gated_wfo_rule": "test-type row (not STATUS) whose Lens 1 "
        "contamination includes anchor_leak or dsr_units (Lens 1 applies those tags only to "
        "run_wfo "
        "results), plus three post-fix reruns whose verdict uses run_wfo output: "
        + json.dumps(POST_FIX_GATED),
        "verdict_rows_resting_on_gated_wfo": gated,
        "simpler_test_would_differ_shown_from_artifact": [
            {
                "id": "stocks-momentum-french-screen",
                "finding": "A1 single-test p (big 0.001135 <= "
                "0.00135) would pass a non-Holm t>=3 bar; it does not pass Holm step 1 (0.000675; "
                "holm.big false). Overall "
                "verdict "
                "unchanged because A2 fails for both groups (+0.08 / +0.20 pp/yr vs +1.0).",
                "keys": "groups.big.full_p, groups.small.full_p, holm.*, "
                "groups.*.post_cagr_diff_pp",
            },
            {
                "id": "cross-asset-month-end-treasury",
                "finding": "WFO criterion 3 failed (halted_any "
                "true, n_used_anchor 8, top winner 5/17), but criterion 4 (portfolio, no WFO) "
                "also failed, "
                "so removing the WFO criterion leaves NO-GO.",
                "keys": "verdict.checks.*; wfo.*",
            },
            {
                "id": "cross-asset-sleeve-trend-arm",
                "finding": "c_no_regime_halt failed, but check a "
                "(Sharpe +0.05 or MaxDD +3 pts) also failed on the plain pinned comparison, so the "
                "verdict is unchanged without the WFO checks.",
                "keys": "verdict.trend_checks.*",
            },
            {
                "id": "stocks-xs-momentum-corrected-replay",
                "finding": "0/17 gate passes (anchor every "
                "fold), but the verdict rests on the NW t of OOS excess (0.76 < 3); whether a "
                "fixed-rule "
                "test would differ is unknown (no fixed-rule arm in the artifact).",
                "keys": "xs_momentum.n_gate_pass; excess_stats.t_stat",
            },
        ],
        "unknown": "All July 2026 gated WFO NO-GOs (no raw artifact) and the June ensemble rows: "
        "whether a simpler test would differ is unknown; no artifact holds a non-gated arm.",
    }

    out = {
        "spec_commit": SPEC_COMMIT,
        "classification_rules": RULES_TEXT,
        "classification_rules_known_limitations_post_hoc": RULES_KNOWN_LIMITATIONS,
        "status_note": STATUS_NOTE,
        "classifications": classes,
        "sensitivity_header": "Exploratory sensitivity table. Not a pass bar. Original "
        "pre-registered "
        f"bars decide verdicts. Spec commit: `{SPEC_COMMIT}`.",
        "sensitivity_columns": COLS,
        "sensitivity": sens,
        "multiple_testing": multiple_testing,
        "gate_machinery": gate_machinery,
        "researcher_freedom": RESEARCHER_FREEDOM,
    }
    out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if md_dir is None:
        return out
    md_dir.mkdir(parents=True, exist_ok=True)

    with open(md_dir / "sens.md", "w") as f:
        f.write(
            "| "
            + " | ".join(
                [
                    "id",
                    "window",
                    "strat CAGR %",
                    "SPY TR CAGR %",
                    "gap pp/yr",
                    "95% CI",
                    "t",
                    "t>=3",
                    "t>=2",
                    "after-tax gap",
                    "power note",
                    "original verdict and bar",
                    "artifact and keys",
                    "exposure / leverage",
                    "SE method",
                ]
            )
            + " |\n"
        )
        f.write("|" + "---|" * 15 + "\n")
        for s in sens:
            f.write("| " + " | ".join(esc(s[c]) for c in COLS) + " |\n")

    with open(md_dir / "class.md", "w") as f:
        f.write(
            "| id | verdict | type | class | rules out larger than (pp/yr, t>=3) | Lens 1 | Lens 6 "
            "| Lens 7 | Lens 9 | bias direction | why "
            "|\n|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for c in classes:
            e = c["evidence"]
            f.write(
                "| "
                + " | ".join(
                    esc(x)
                    for x in [
                        f"`{c['id']}`",
                        c["verdict"],
                        c["verdict_type"],
                        f"**{c['class']}**",
                        c["rules_out_larger_than_pp_per_yr"],
                        e["lens1"],
                        e["lens6"],
                        e["lens7"],
                        e["lens9"],
                        c["bias_direction"],
                        c["note"],
                    ]
                )
                + " |\n"
            )

    return out


def main() -> None:
    """Parse arguments and rebuild the Lens 2 outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-json", type=Path, required=True, help="JSON output path")
    parser.add_argument("--md-dir", type=Path, default=None, help="optional dir for table .md")
    args = parser.parse_args()
    load_inputs()
    out = build(args.out_json, args.md_dir)
    cls = out["classifications"]
    print(Counter(c["class"] for c in cls))
    print(Counter((c["verdict_type"], c["class"]) for c in cls))
    print("sensitivity rows", len(out["sensitivity"]))


if __name__ == "__main__":
    main()
