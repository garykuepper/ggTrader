# Step 1.5: Per-fold z-rank in within-combo param ranking

**Date:** 2026-05-08
**Owner:** garykuepper
**Predecessor:** `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md` (Step 1, NEUTRAL)

## Problem

Step 1 (PARAM_STABILITY_WEIGHT 0.3 → 0.7) was a no-op for selection: same picks at both weights, identical OOS values. The CV penalty `(1 − weight × CV)` is a uniform multiplicative scale — it changes the *magnitude* of cell scores but not their *relative ranking*, because high-IS-mean cells have low CV and are barely penalized while low-IS-mean cells have high CV and were already crushed at the baseline weight.

The within-combo Layer-1 ranking (`_weighted_robustness_series` in `src/ggTrader/core/wfo.py:235-293`) takes a fold-weighted **mean of raw IS Sharpe** across folds and picks the cell with the largest mean. This is dominated by single-fold outliers: a cell that's #1 in one fold (raw IS=3.0) and #50 in another (raw IS=0.5) averages to 1.75 — looking great in absolute terms, but its rank position bounces wildly across folds.

## Hypothesis

Replacing (or blending) the raw-mean ranking with a **mean-of-per-fold-z-scores** ranking will reward cells that are consistently above-median across folds and demote cells that are noisy single-fold outliers. Because z-scoring normalizes per-fold difficulty (a fold where everyone scores low has small std → big z's; a fold where everyone scores high has small std → big z's; both folds contribute equally to the rank), the cell that wins is the one with the most stable relative position, not the one with the biggest single-fold spike.

This directly addresses the failure mode Step 1 could not: changing the *winner identity*, not just its score.

## Diagnosis recap

From the Step 1 spec, per-fold IS spread on winning combos at PARAM_STABILITY_WEIGHT=0.3:

| Coin | n cells | typical max | p95 | median | std | max − median |
|---|---:|---:|---:|---:|---:|---:|
| TRX-USD | 144 | 1.0–2.0 | 1.0–1.7 | −0.4 to +0.3 | 0.99 | 0.7–2.5 |
| ETH-USD | 144 | 1.4–3.0 | 0.6–1.9 | −0.3 to +0.5 | 0.93 | 1.5–3.1 |
| DASH-USD | 192 | 2.2–3.0 | 1.6–2.0 | −0.3 to 0.06 | 0.92 | 2.1–3.2 |

The cell σ ≈ 1.0 across folds. Z-scoring within each fold gives the IS-best cell a z ≈ 2.5–3.0; a "consistently top-decile" cell sits at z ≈ 1.5 every fold. Under the raw-mean rule the IS-best cell wins with a single-fold spike; under z-rank the consistent top-decile cell wins with mean z ≈ 1.5 vs the spiky cell's mean z ≈ (3.0 + 0.0)/2 = 1.5 — and crucially, **with much lower variance across folds** (i.e. genuine edge).

## Change

Add a new config knob `PARAM_ZRANK_WEIGHT` (default `0.0` = current behavior) in `src/ggTrader/utils/run_config.py` next to `PARAM_STABILITY_WEIGHT`. Modify `_weighted_robustness_series` in `src/ggTrader/core/wfo.py:235-293` to compute a per-fold z-score column matrix and blend it into the final score:

```
raw_score   = sum_f w_f * IS[cell, f]                         (existing, weighted mean across folds)
z_score[f]  = (IS[:, f] - nanmean(IS[:, f])) / nanstd(IS[:, f])  per-fold standardization
zrank_score = sum_f w_f * z_score[cell, f]                    (weighted mean of per-fold z's)
final_score = (1 - alpha) * raw_score + alpha * zrank_score
```

with `alpha = PARAM_ZRANK_WEIGHT`. At `alpha=0` the function is bit-identical to current; at `alpha=1` it's pure z-rank; experiment value is `0.5` (50/50 blend) for first iteration.

**Edge cases:**

- **Degenerate fold** (`nanstd == 0` because all finite cells in a fold have the same IS): the per-fold z column is set to all zeros (every cell contributes 0 to the rank from that fold).
- **Fold with all-NaN cells**: already handled by the existing `finite_mask` machinery — both `raw_score` and `zrank_score` skip it.
- **Cell present in some folds, absent in others** (vectorized vs legacy WFO row-count drift): the existing `union_idx + reindex` machinery handles missing-key NaNs. The z-rank computation re-uses the same `mat` and `finite_mask`, so missing values contribute 0 weight to both raw and z-rank means.

**Why blend, not replace:** keeps a partial IS magnitude signal so a cell that's consistently top-rank but in absolute terms unprofitable doesn't win. With pure z-rank, a fold of universally negative IS (bear training window) would still rank the "least bad" cell positively. Blending with the raw mean caps that — a bad-magnitude cell can't win even if its rank is consistent.

## Implementation outline

1. **`src/ggTrader/utils/run_config.py`** — add the new knob:

   ```python
   # Per-fold z-rank blend weight: 0.0 = pure raw weighted mean (current),
   # 1.0 = pure mean-of-per-fold-z-scores, 0.5 = 50/50 blend.
   # Per-fold z-rank rewards cells that are *relatively* high in every
   # fold (stable rank position) over cells that spike high in one fold
   # and average elsewhere (noise-driven max-of-N selection bias).
   "PARAM_ZRANK_WEIGHT": 0.5,
   ```

2. **`src/ggTrader/core/wfo.py`** — in `_weighted_robustness_series`, after the existing `combined = np.where(den > 0.0, num / den, np.nan)` line, compute per-fold z-scores from `mat` and weighted-mean them across folds, then blend.

   ```python
   alpha = float((config or {}).get("PARAM_ZRANK_WEIGHT", 0.0))
   if alpha > 0.0:
       # Per-fold z: standardize each column over its finite cells.
       z_mat = np.full_like(mat, np.nan)
       for j in range(n_folds):
           col = mat[:, j]
           col_mask = np.isfinite(col)
           if col_mask.sum() < 2:  # need at least 2 cells for std
               continue
           col_finite = col[col_mask]
           col_mean = col_finite.mean()
           col_std = col_finite.std()
           if col_std == 0.0:
               continue  # degenerate fold; leave z's as NaN -> contributes 0
           z_mat[col_mask, j] = (col_finite - col_mean) / col_std
       z_finite = np.isfinite(z_mat)
       z_weighted_vals = np.where(z_finite, z_mat * wvec, 0.0)
       z_weighted_wts  = np.where(z_finite, wvec, 0.0)
       z_den = z_weighted_wts.sum(axis=1)
       z_num = z_weighted_vals.sum(axis=1)
       zrank = np.where(z_den > 0.0, z_num / z_den, np.nan)
       # Blend with raw weighted mean. Where one is NaN, fall back to the other.
       both_finite = np.isfinite(combined) & np.isfinite(zrank)
       combined = np.where(
           both_finite,
           (1.0 - alpha) * combined + alpha * zrank,
           np.where(np.isfinite(combined), combined, zrank),
       )
   ```

   The function signature must accept `config` so the alpha is reachable. Current signature is `(is_metrics_by_fold, weights)`; add a third optional `config: Optional[Dict[str, Any]] = None` kwarg with default `None` → preserves alpha=0 behavior at all existing call sites.

3. **Caller updates** — pass `config` from `core/orchestrator.py:976-480` and `wfo.py:476-480` (the two call sites of `_weighted_robustness_series`). The orchestrator already has `config` in scope; the wfo.py internal caller also has it via the function's closure.

4. **Cache key** — add `"PARAM_ZRANK_WEIGHT"` to `_WFO_RELEVANT_CONFIG_KEYS` in `src/ggTrader/data/cache/wfo_cache.py`. Even though the knob is post-hoc on cached IS metrics, including it in the cache key is conservative and follows the existing convention. The cost on the diagnostic 7-coin universe is one ~22 min re-run, which we're paying anyway.

5. **Test** — small-N micro-experiment (same 7-coin universe as Step 1). Run two configurations on the same universe:
   - **Run A** (`PARAM_ZRANK_WEIGHT = 0.5`, blend)
   - **Run B** (`PARAM_ZRANK_WEIGHT = 0.0`, baseline = current)

   Capture scorecards via `scripts/scorecard_step1.py` (already created in Step 1). Compare per-coin OOS deltas and check whether **picks change** (the key indicator that the new knob has any effect at all — Step 1's failure was diagnosed by inspecting picks, not just scorecard values).

## Scorecard (same as Step 1)

**Primary (B-criterion):** `n_oos_gt_0_30`, `median_fold_consistency`, `n_oos_gt_0`.
**Secondary:** `n_survivors`, `median_is_minus_oos_gap`.
**Sanity:** Phase 3 YTD return, Sharpe, MaxDD vs BTC B&H.
**Plus diagnostic:** for each survivor, did the chosen `(strategy, exit, params)` change vs Step 1 baseline? (Inspect `worker_*_results.json` `best_strategy / best_exit / best_params` per coin.) If picks unchanged, the experiment is a no-op like Step 1 and we move on to Step 2.

## Decision rule

Same B-criterion as Step 1, with one addition:

- **Hard precondition:** if `picks_changed_count == 0` across the 4 surviving coins, classify as **NO-OP** regardless of scorecard movement (rules out the Step-1 "uniform scaling" failure mode). Move to Step 2.
- **ADVANCE:** picks changed AND scorecard improved on ≥ 2 of 3 primary metrics. Promote to a 100-coin ratification run.
- **REVERT:** picks changed AND scorecard worsened on ≥ 2 of 3. Set `PARAM_ZRANK_WEIGHT` back to `0.0`; investigate why z-rank picked worse cells (most likely: pure rank ignores magnitude in a fold of universally negative IS).
- **NEUTRAL:** picks changed but scorecard mixed/flat. Try `alpha = 1.0` (pure z-rank) before moving to Step 2.
- **HARD FLOOR:** survivor count drops below 3 (was 4 in Step 1; one-coin slack is realistic). Back off and move on.

## Risks

- **Pure-rank insensitivity to magnitude:** a fold of universally negative IS will still produce z-scores ranging −2 to +2; the "least bad" cell wins from that fold's contribution. Blending with raw mean (alpha=0.5) caps this. If alpha=1.0 is tried in a follow-up, watch for cells that win on rank but have terminal absolute IS.
- **Degenerate folds in sparse-survivor coins** (DOGE/XMR with only 4 of 36 cells finite per fold): z-scoring 4 values gives unstable z's. The `col_mask.sum() < 2` guard skips those folds entirely, so the z-rank score for those coins comes from the few well-populated folds. Acceptable for now; the sparse-survivor problem is its own Step 3.
- **Ranking ties:** identical IS across cells in a fold → std=0 → fold contributes 0 z. If many folds are tied, the z-rank score collapses toward the raw mean, which is fine.

## What this experiment does *not* do

- Does not modify `PARAM_STABILITY_WEIGHT` (still `0.3` post-Step-1). The two knobs compose; this experiment isolates z-rank.
- Does not introduce OOS into Layer-1 ranking (that is Step 4 — schema-invasive).
- Does not change the cross-combo blend (Layer 2 is fine; it's already 70% OOS).
- Does not shrink the param grid (Step 2).

## Rollback

`PARAM_ZRANK_WEIGHT = 0.0` reproduces current behavior bit-for-bit. The new code path is gated by `if alpha > 0.0`. Reverting is a one-line config edit; the function signature change (added `config` kwarg with `None` default) is non-breaking for existing call sites.

## Multi-step plan reference (informational)

| # | Step | Cost | Expected impact | Status |
|---|---|---|---|---|
| 1 | `PARAM_STABILITY_WEIGHT` 0.3 → 0.7 | full WFO re-run | medium-high | NEUTRAL — uniform scale, no rank flip |
| **1.5** | **Per-fold z-rank blend (`PARAM_ZRANK_WEIGHT`)** | **full WFO re-run; ~30-line `wfo.py` edit** | **medium-high — directly re-ranks cells** | **this spec** |
| 2 | Shrink param grids (≈ halve combos) | full WFO re-run; grid edits | medium | pending |
| 3 | Address DOGE-style sparse-fold survivors (`MIN_CLOSED_TRADES_TRAIN` calibration per strategy) | analysis + tweak | medium | pending |
| 4 | Per-cell OOS in cache + Layer-1 OOS blend (`IS_OOS_PARAM_BLEND_ALPHA`) | invasive: schema migration + WFO core edit | high | pending |
| 5 | Universe pruning beyond current filters | cheap | low–medium | pending |

Each subsequent step gets its own spec, run, scorecard, and decision-rule check before the next one starts.

---

## Result

Run via small-N micro-experiment on the diagnostic 7-coin universe (BTC, ETH, TRX, DOGE, XMR, DASH, ADA), same shape as Step 1.

- **Run A** (`PARAM_ZRANK_WEIGHT=0.5`, blend): `results/research/research_20260508_232226/` (~38 min wall, slower than Step 1's runs due to server load)
- **Run B** (`PARAM_ZRANK_WEIGHT=0.0`, baseline): `results/research/research_20260509_102421/` (~47 min wall)

**Pick comparison (hard precondition):** picks changed on **0 of 4** surviving coins. Z-rank blend at α=0.5 produced bit-identical selection (same `(strategy, exit, params)` triples for TRX, DOGE, ETH, ADA in both runs). The hard-precondition fires: classify as **NO-OP** regardless of scorecard movement.

**Primary scorecard delta:**

| Metric | Before (α=0.0) | After (α=0.5) | Δ |
|---|---:|---:|---:|
| `n_oos_gt_0_30` | 1 | 1 | 0 |
| `median_fold_consistency` | 0.55 | 0.55 | 0 |
| `n_oos_gt_0` | 2 | 2 | 0 |

**Secondary:** `n_survivors` 4 → 4; `median_is_minus_oos_gap` 1.64 → 1.72 (slight rise — IS values displayed are now a 50/50 blend with z-rank means, scaled differently, so the gap metric is no longer apples-to-apples).

**Phase 3 sanity:** Total Return −21.45% (vs −21.78% baseline); Sharpe −1.14 (vs −1.17); MaxDD −35.7% both runs.

**Decision:** **NO-OP**.

**Mechanistic finding (different from Step 1):** Step 1 was a no-op because the CV penalty is a uniform multiplicative scale that doesn't flip rankings. Step 1.5 is a no-op for a *different* structural reason: when the IS-best cell is consistently the per-fold leader (which it is for all 4 surviving coins — TRX/DOGE/ETH/ADA), it is also consistently the highest-z cell. Both ranking schemes agree → no flip. Z-rank only diverges from raw mean when there are spiky single-fold outliers competing with consistent mid-rank cells; in this universe, the IS-leaders ARE the consistent leaders. They just happen not to generalize OOS.

**The deeper lesson:** Step 1 and Step 1.5 together demonstrate that **no IS-only re-ranking transform can fix the IS-OOS gap**. The problem isn't *how* we rank within IS — it's that IS leaders systematically don't generalize. The only fixes that bring new information into ranking are Step 4 (per-cell OOS in Layer 1 ranking; cache-schema invasive) and Step 2 (shrink the grid so there are fewer chances to find lucky overfit leaders). Step 4 is the principled fix; Step 2 is a cheaper proxy.

**Action:** `PARAM_ZRANK_WEIGHT` set to `0.0` in `src/ggTrader/utils/run_config.py` (post-Run-B revert). Live trader is unaffected because identical params would have been chosen at either α. Function code path remains in place (gated by `if alpha > 0.0`); the knob can be reused for future experiments without further code change. Tests stay in `tests/test_wfo_zrank_blend.py` as a regression net.

**Next step:** the natural next experiment is Step 2 (grid shrink) as a cheap, attributable test of the "fewer noise draws" hypothesis. Step 4 (per-cell OOS) is the principled fix but expensive (cache schema migration + larger code surface). Recommendation: do Step 2 first as a small-N micro-experiment, then if the Step 2 gain is partial, commit to Step 4.
