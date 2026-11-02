# WFO Fallback and Halt Semantics: what each recorded quantity measures today

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** Historical legacy behavior is characterized below. Opt-in alternatives and additive reporting are available; owner adoption is pending. Passing the tests below shows the code does what this document says; it does not establish alpha or change any historical verdict.

Terms: **WFO** = walk-forward optimization. **IS / OOS** = in-sample / out-of-sample. **WFE** = walk-forward efficiency (OOS Sharpe ÷ IS Sharpe). **NDH** = plateau-density gate; **DSR** = deflated Sharpe ratio gate. **Anchor** = defensive fallback parameter set. **Winner** = the combination with the best training composite score. **Deployed policy** = whatever parameters the harness actually traded in the test window.

## 1. Three quantities that the harness mixes

| Quantity | Meaning | Where it is recorded |
|---|---|---|
| Winner performance | Did the parameters chosen in training generalize OOS? | **Not recorded** when the anchor is deployed; the winner is never run on the test window |
| Anchor performance | How did the defensive fallback do? | `oos_sharpe` on a fold with `used_anchor = True` |
| Deployed-policy performance | How did the complete gate + halt + fallback system do? | `oos_sharpe` on every fold; the stitched OOS curve in `oos_equity` |

*Observation.* `oos_sharpe` is always the **deployed** parameters' Sharpe: `test_metrics` is computed for `[deploy_params]` only (`wfo.py` fold loop, after line 731). The aggregate "OOS Aggregate: Sharpe …" therefore describes the deployed policy.

## 2. Findings, each pinned by a test

Each finding states observed behavior, the file reference, the characterization test, and an **owner decision** (intended behavior). Line numbers were verified 2026-11-02.

### F1. Anchor choice uses the halt state entering the fold; the recorded `halted` is the state after the fold

Implementation: additive reporting (`halted_entering`, `skipped_folds`) is implemented unconditionally; trading semantics remain legacy by default, and owner adoption of alternative semantics is pending.

- `wfo.py:731`: `if not gates_passed or wfo_state.halted:` uses the state **entering** the fold.
- `wfo.py:796`: the row stores `"halted": wfo_state.halted`, after `check_circuit_breaker` / `check_shadow_reentry` ran on this fold's results.
- Consequence: a row with `gates_passed = True`, `used_anchor = True` was halted entering; a row with `halted = True` may have deployed the winner. Attribution therefore classifies by `gates_passed` and `used_anchor`, not by `halted` ([attribution](2026-11-02-wfo-fold-attribution.md)).
- Test: `test_run_wfo_halted_field_is_post_fold_state`.
- **Owner decision:** keep both fields and rename, or store `halted_entering` explicitly. No effect on historical artifacts either way.

### F2. On anchor folds, WFE mixes two parameter sets

Option (b): implemented as opt-in WfoPolicy (default legacy); owner adoption pending. The exclusion affects breaker history only: shadow re-entry still receives computed WFE, since withholding it on every halted anchor fold would prevent re-entry and deadlock the halt.

- `wfo.py:767`: `wfe_val = compute_wfe(is_sharpe, oos_sharpe)` where `is_sharpe` is the **winner's** training Sharpe and `oos_sharpe` is the **deployed** (anchor) test Sharpe.
- That WFE feeds the circuit breaker history and `check_shadow_reentry`. A halt exit or entry can therefore depend on how the anchor did relative to the winner's in-sample Sharpe, not on whether the withheld winner recovered.
- Test: `test_run_wfo_wfe_on_anchor_fold_uses_winner_is_sharpe` (winner IS 2.0, anchor OOS 0.3 → WFE 0.15).
- **Owner decision:** (a) keep as a deployed-policy health measure; (b) compute WFE only on winner-deployed folds; (c) run a shadow evaluation of the winner on withheld folds to measure recovery. (c) changes what is computed and costs one extra test sweep per fold.

### F3. NaN WFE is not excluded from the breaker window

Implementation: implemented as opt-in `WfoPolicy(nan_wfe="exclude")` / `check_circuit_breaker(nan_policy="exclude")` (default legacy); owner adoption pending.

- `compute_wfe` returns `None` only for IS Sharpe below the 0.4 floor (`wfo.py:44`); a NaN OOS Sharpe gives NaN (`test_compute_wfe_returns_nan_when_oos_nan`).
- `check_circuit_breaker` filters only `None` (`wfo.py:91`). With history `[0.1, 0.1, 0.1, 0.1, nan]` and window 4 the trailing window contains the NaN, its mean is NaN, `NaN < 0.25` is false, and no chronic-decay halt fires (`test_circuit_breaker_nan_wfe_suppresses_chronic_decay`). With a trailing `None` instead, the four real values are used and a halt fires (`test_circuit_breaker_none_wfe_excluded_from_window`).
- Saved artifacts contain 4 NaN WFE folds (PIT `ensemble` 3, PIT `ensemble_ic` 1). Whether they changed a halt in those runs is **not established**; one reviewer reported no change in one PIT ensemble run, which is an unverified secondhand statement.
- **Owner decision:** treat NaN like `None` (excluded), like zero, or like a failure. Any change shifts halt timing in future runs and must be flag-guarded.

### F4. Non-finite OOS Sharpe values are dropped from the acute-failure window

- `wfo.py:769`: only finite OOS Sharpes are appended to `oos_sharpes`, so "two consecutive negative OOS Sharpe" can be satisfied by two negative folds that are not adjacent when a fold between them had no finite Sharpe.
- No dedicated test yet; it is a code reading, labeled **observation**, and low-priority.
- **Owner decision:** whether non-finite folds should break the "consecutive" run.

### F5. Folds without training metrics leave no row

Implementation: additive reporting (`halted_entering`, `skipped_folds`) is implemented unconditionally; trading semantics remain legacy by default, and owner adoption of alternative semantics is pending.

- `wfo.py:673`: `if not train_metrics: continue`. The fold is not recorded and does not enter `fold_winners`, the WFE history or the OOS curve.
- Consequence: fold counts and the fold-number sequence can understate attempts; the saved artifacts reviewed here show no gaps ([attribution](2026-11-02-wfo-fold-attribution.md) §2).
- Test: `test_run_wfo_skips_fold_without_train_metrics`.
- **Owner decision:** record a skipped-fold row with a reason, or keep silent skipping.

### F6. DSR trial count equals the parameter-grid size

- `wfo.py:718`: `n_trials=len(grid)`. Correlated variants count as independent trials in that fold, and this says nothing about the many studies tried before.
- Not pinned by a unit test (it would restate a constant); sensitivity belongs to the optional synthetic study (plan Task 6).
- **Owner decision:** whether the effective trial count should be reduced for correlated grids, and how to account for repeated research across strategies.

## 3. What this document does not decide

- It does not say which semantics are correct. Each "owner decision" line remains pending adoption of a policy.
- It does not re-score any strategy. A change would alter reproducibility of old runs; any change must be flag-guarded and each historical artifact must stay labeled as produced by the old policy.
- The sufficiency of these tests: they use scripted sweeps and gates (`_scripted_wfo` in `tests/lab/test_wfo.py`), so they verify state-transition bookkeeping, not statistical behavior. Three mutations of `wfo.py` (removing `or wfo_state.halted`, filtering NaN in the breaker, hard-coding the OOS Sharpe used in WFE) each made the matching test fail; the file was restored to its committed state afterwards.

## 4. Decision checklist for gate G2

- [ ] F1: store `halted_entering`? (yes/no)
- [ ] F2: anchor-fold WFE policy a / b / c
- [ ] F3: NaN WFE handling: exclude / zero / failure
- [ ] F4: should non-finite folds break "consecutive"?
- [ ] F5: record skipped folds?
- [ ] F6: effective trial-count policy
- [ ] Will the WFO harness be used by any future brief? (If not, F1–F6 stay documented only and the synthetic study, plan Task 6, is not started.)
