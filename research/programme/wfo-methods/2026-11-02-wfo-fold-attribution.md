# WFO Fold Attribution: what the saved walk-forward runs actually deployed

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Audience:** Principal Engineering Team & Quantitative Research Collaborators
**Status:** Documentary attribution of saved records. **Not a new strategy verdict.** No backtest, database query, data acquisition or order was run; no historical NO-GO changes.

Terms used: **WFO** = walk-forward optimization (train on a window, deploy on the next). **OOS** = out-of-sample. **WFE** = walk-forward efficiency, OOS Sharpe ÷ in-sample Sharpe. **NDH** = the plateau-density gate. **DSR** = deflated Sharpe ratio gate. **Anchor** = the defensive fallback parameter set the harness deploys when a fold fails its gates or the system is halted.

## 1. Executive Summary & Core Engine Audit

*Observation.* When a fold fails its gates, or the circuit breaker is halted, `run_wfo` deploys the anchor parameters instead of the training winner (`src/ggTrader/lab/wfo.py:731`). A WFO "OOS Sharpe" therefore measures the full gate-plus-fallback policy, not the training winner. This report counts, for every saved fold row that can be recovered from the repo, which of three things happened:

- `gate_fail`: gates failed, anchor deployed.
- `gates_pass_anchor`: gates passed but the anchor was still deployed. By `wfo.py:731` this implies the system was halted **entering** that fold.
- `gates_pass_winner`: gates passed and the training winner was deployed.

*Observation.* The 13 runs in 7 artifacts split into two groups by when each artifact entered Git relative to the DSR fix (`09f9f40`, 2026-10-19): ten were committed before it (old gate for certain) and three, all from `_audit_reruns_20261019.json`, after it (a corrected harness is documented for them in the [production-rule audit](../../audits/production-replica/2026-10-19-production-replica-audit-nogo.md); Git alone does not prove it). Three runs are dominated by anchor deployment: `xs_momentum` and `dual_momentum` (17/17 anchor, corrected harness documented) and `fomc_drift` (47/53 anchor, corrected harness documented); in the pre-fix group, PIT `ensemble_ic` deployed the anchor in 15/17 folds. Most other pre-fix runs deployed the winner in a majority of folds. Everything pre-fix reflects the density gate only, because the uncorrected DSR gate passed almost everything (`docs/changelog.md`, "Lab DSR correction").

*What this does not show.* It does not show what the undeployed winner would have earned, and it does not make any strategy succeed or fail. Already-closed verdicts stand; this report only states which part of the pipeline each saved run exercised.

## 2. Quantitative Performance Context: fold attribution

Counts are from `research/programme/wfo-methods/_wfo_fold_attribution_results.json`, produced by `scripts/wfo_fold_attribution.py` from existing artifacts (`runs[*].summary`). "Anchor" = `gate_fail + gates_pass_anchor`. "Regime" is inferred from Git commit history of the artifact (a commit time bounds, but does not prove, the gate version used by the run).

| Artifact : run | Folds | gate_fail | gates_pass_anchor | gates_pass_winner | Anchor folds | Regime |
|---|---:|---:|---:|---:|---:|---|
| `_rebaseline_corrected_tape_20260822.json` : core | 17 | 5 | 0 | 12 | 5 (29%) | pre-fix (committed before the fix) |
| `_anchor_fix_reproduction_results.json` : core | 17 | 5 | 0 | 12 | 5 (29%) | pre-fix (committed before the fix) |
| `_pit_rebaseline_20260923.json` : ensemble | 17 | 0 | 0 | 17 | 0 (0%) | pre-fix (committed before the fix) |
| `_pit_rebaseline_20260923.json` : ensemble_ic | 17 | 6 | 9 | 2 | 15 (88%) | pre-fix (committed before the fix) |
| `_pit_rebaseline_20260923.json` : ensemble_kelly | 17 | 1 | 6 | 10 | 7 (41%) | pre-fix (committed before the fix) |
| `_ic_kelly_rebaseline_20260923.json` : ensemble | 17 | 5 | 0 | 12 | 5 (29%) | pre-fix (committed before the fix) |
| `_ic_kelly_rebaseline_20260923.json` : ensemble_ic | 17 | 5 | 4 | 8 | 9 (53%) | pre-fix (committed before the fix) |
| `_ic_kelly_rebaseline_20260923.json` : ensemble_kelly | 17 | 1 | 3 | 13 | 4 (24%) | pre-fix (committed before the fix) |
| `_month_end_treasury_results.json` : wfo | 17 | 4 | 4 | 9 | 8 (47%) | pre-fix (committed before the fix) |
| `_cross_asset_sleeve_results.json` : wfo | 17 | 3 | 4 | 10 | 7 (41%) | pre-fix (committed before the fix) |
| `_audit_reruns_20261019.json` : xs_momentum | 17 | 17 | 0 | 0 | 17 (100%) | committed after fix; corrected harness documented |
| `_audit_reruns_20261019.json` : dual_momentum | 17 | 17 | 0 | 0 | 17 (100%) | committed after fix; corrected harness documented |
| `_audit_reruns_20261019.json` : fomc_drift | 53 | 42 | 5 | 6 | 47 (89%) | committed after fix; corrected harness documented |

*Verification (calculation).* The three-way counts for `core` (08-22), PIT `ensemble_ic` and `fomc_drift` were recomputed independently from the raw `fold_rows` and match. No row was an `anomaly` (no fold had both `gates_passed` and `used_anchor` false), and no run had a gap in `fold_num`. Two artifacts (`_cross_asset_sleeve_results.json`, `_month_end_treasury_results.json`) carry the post-fold `halted` field; in both, every `gates_pass_anchor` fold was preceded by a fold whose recorded `halted` was true, consistent with the halt-entering inference. For `cross_asset_sleeve` the recorded post-fold `halted` was true for folds 2–8; for `month_end_treasury` for folds 8–12.

*Observation (missing WFE).* Saved WFE is NaN in 4 folds: 3 in PIT `ensemble` and 1 in PIT `ensemble_ic`. WFE is `null` (the neutral window, in-sample Sharpe below the 0.4 floor) in 35 folds across 7 runs, including 15 of 53 in `fomc_drift` and 7 in PIT `ensemble_kelly` (`runs[*].summary.nan_wfe` / `none_wfe`). Because `check_circuit_breaker` filters only `None`, NaN values are the ones that can stay in the trailing window ([semantics](2026-11-02-wfo-fallback-semantics.md)); whether that changed any halt in these four folds is not established here.

*Unknown.* The two `core` rows are identical because they are two artifacts of the same configuration; they are not independent confirmations. The `has_halted_field` flag in the JSON shows which runs allow this cross-check.

## 3. Actionable Research Directions

None selected here. Bounded documentary and engineering follow-ups that this attribution supports:

1. Specify what WFE should mean on anchor folds (Task 2 of the plan; see [fallback semantics](2026-11-02-wfo-fallback-semantics.md)). Today it is the anchor's OOS Sharpe divided by the *winner's* in-sample Sharpe.
2. When a future brief proposes using WFO, require it to report the three-way fold split before interpreting OOS Sharpe.
3. Do not infer from the pre-fix gate-pass counts ("x/17") that parameter selection was validated; they reflect the density gate only.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

No arc is reopened. For orientation, these closed studies relied on runs with the following anchor shares (observation, not a verdict change): `xs_momentum` and `dual_momentum` replays (100% anchor); `ensemble_ic` PIT rebaseline (88%); `fomc_drift` WFO (89%). Their NO-GO status was decided by their registered bars; for momentum, the separately registered French long-history screen does not use this harness and also failed its bars ([momentum report](../../stock-signals/momentum/2026-10-20-momentum-french-screen-nogo.md)).

## 5. Operational Roadmap: Recommended First Action

**Keep this record as a reference and use it at gate G1.** Decide whether the audit report's WFO caveats ([audit §Verified concerns](../../audits/research-process/2026-11-01-research-process-audit.md)) need a dated addendum citing these counts. No production or paper-trading change follows from this report.

## 6. Contrarian Evaluation & Parked Research

*Contrarian question:* does an anchor-dominated run mean the strategy was never tested? Partly. It tested the gate/fallback policy; a high anchor share says the training winner was rarely deployed. That has two causes that the counts separate: gate failure (`gate_fail`) and deployment blocked by an existing halt after the gates passed (`gates_pass_anchor`; for example PIT `ensemble_ic` passed the gates in 11 of 17 folds yet deployed the anchor in 15). Infer gate passage only from the explicit gate counts. Neither cause says whether the undeployed winner would have lost or won. Unmeasurable from saved rows:

- The winner's OOS performance on anchor folds (not recorded).
- The halt state entering each fold (only the post-fold `halted` is stored, in 2 of 7 artifacts).
- Folds skipped for missing training metrics, which leave no row (`wfo.py` fold loop).
- Runs not saved as artifacts (database-only lab runs were not examined).

*Not done:* any rerun with the winner forced on. A preregistered synthetic characterization of the breaker bookkeeping has been run; see the [synthetic study](2026-11-02-wfo-policy-synthetic.md).
