# Step 1: Tighten `PARAM_STABILITY_WEIGHT` to attack Layer-1 selection bias

**Date:** 2026-05-08
**Owner:** garykuepper
**Context:** First experiment in a multi-step plan to address WFO overfitting surfaced by the 2026-05-08 research run (median IS Sharpe ≈ 1.65 vs median OOS Sharpe ≈ 0.08 across 17 surviving coins).

## Problem

The 2026-05-08 research run produced 17 coins surviving the per-worker selection gates (robustness, consistency, history shrinkage). Of those, 11 entered the Phase 2/3 combined portfolio after the trade-frequency gate. Their per-coin IS robustness scores cluster around 1.5–2.3 while OOS robustness scores collapse to roughly 0–0.4, with negative OOS for several (CRV, NEAR, DASH, ADA). Phase 3 (YTD) returned 43.55% and beat BTC, but the IS↔OOS gap suggests the parameter selection process is picking IS noise rather than durable edges.

A diagnostic query of the WFO cache (`wfo_cache` table) confirms the source: per fold, across all 144 param cells in the winning combo, the maximum IS Sharpe sits 1.5–3.0 Sharpe units above the median, while the median itself is near zero. With cell-level σ ≈ 1.0 and ~140 cells, the expected max from noise alone is ~2.5σ above median — exactly what we observe. This is textbook max-of-N selection bias on noisy IS scores.

## Diagnosis (per-fold IS spread on winning combos)

| Coin | n cells | typical max | p95 | median | std | max − median |
|---|---:|---:|---:|---:|---:|---:|
| TRX-USD | 144 | 1.0–2.0 | 1.0–1.7 | −0.4 to +0.3 | 0.99 | 0.7–2.5 |
| ETH-USD | 144 | 1.4–3.0 | 0.6–1.9 | −0.3 to +0.5 | 0.93 | 1.5–3.1 |
| DASH-USD | 192 | 2.2–3.0 | 1.6–2.0 | −0.3 to 0.06 | 0.92 | 2.1–3.2 |
| XMR-USD | 24–108 | 1.4–2.9 | 1.0–2.8 | −0.7 to +0.8 | 1.07 | 1.2–3.6 |
| DOGE-USD | 4–36 | varies | varies | varies | varies | varies |

DOGE/XMR have a separate sparse-survivor problem (`MIN_CLOSED_TRADES_TRAIN` is filtering most cells), addressed in a later step.

## Existing mitigations (already in place)

- `PARAM_STABILITY_WEIGHT = 0.3` — CV-based fold-stability penalty
- `TRAIN_METRIC_NORMALIZE_ZSCORE = True`
- `OOS_ROBUSTNESS_BLEND_ALPHA = 0.70` — cross-combo blend already heavily OOS-weighted
- `OOS_STABILITY_WEIGHT = 0.3`
- `FOLD_CONSISTENCY_GATE_FLOOR = 0.25`, `MIN_FOLD_CONSISTENCY = 0.38`
- `MIN_VALID_TRAIN_FOLDS = 3`, `MIN_ROBUSTNESS_SCORE = 0.10`
- `HISTORY_SHRINKAGE_TARGET_YEARS = 3.0`
- `MIN_TRADES_PER_YEAR = 4` (Phase 3 contiguous-trade gate)
- Top-K fallback, bear-aware fold consistency

The IS≈1.65 / OOS≈0.08 gap persists *despite* this machinery, which points selection bias to Layer 1 (within-combo IS-only param-cell ranking — `_weighted_robustness_series` in `src/ggTrader/core/wfo.py:476-480`).

## Change

Increase `PARAM_STABILITY_WEIGHT` from `0.3` → `0.7` in `src/ggTrader/utils/run_config.py:208`.

The knob, defined in `_apply_stability_penalty` (`src/ggTrader/core/wfo.py:489-493`):

- For each param cell, compute coefficient of variation (CV = std/|mean|) of its IS metric across folds.
- Multiply the cell's IS robustness score by `(1 − weight × CV)`, floored at 0.
- Cells whose IS is high in one fold and low in another (curve-fit signature) get penalized harder at higher weights.

`0.7` was chosen over `1.0` because at `1.0` any cell with CV ≥ 1.0 is zeroed out, which can include genuinely volatile-but-real strategies. `0.7` is a meaningful step-change from `0.3` without being extreme; if `0.7` improves the scorecard we can advance to `0.85` in a follow-up.

## Implementation

```diff
# src/ggTrader/utils/run_config.py
- "PARAM_STABILITY_WEIGHT": 0.3,
+ "PARAM_STABILITY_WEIGHT": 0.7,
```

No source-code changes. The knob is already wired through.

## Cache invalidation and re-run

`PARAM_STABILITY_WEIGHT` is listed in `_WFO_RELEVANT_CONFIG_KEYS` (`src/ggTrader/data/cache/wfo_cache.py:33`), so all cached WFO entries miss on the new value. The next research run executes the full 6-fold WFO for every (symbol, combo). Yesterday's parallel run with warm OHLCV took 21 minutes; expect similar.

```
docker compose build ggtrader_live
docker compose run --rm ggtrader_live python -u ggt.py research --top 100 --days 1095 --end-date 2026-05-01
```

`docker compose build` is required because `src/` is not volume-mounted into the production image.

## Scorecard

Captured before and after the re-run.

**Primary metrics (B-criterion: OOS quality of survivors):**
- `n_coins_oos_gt_0_30` — count of survivors with `oos_robustness_score > 0.30`
- `median_fold_consistency` — across survivors
- `n_coins_oos_gt_0` — count of survivors with `oos_robustness_score > 0`

**Secondary metrics:**
- `n_coins_surviving_all_gates`
- `median_is_minus_oos_gap`

**Sanity:**
- Phase 3 YTD: total return, Sharpe, MaxDD, vs BTC B&H

The "before" baseline is the 2026-05-08 14:02 run (`results/research/research_20260508_135840/`).

## Decision rule

- **Advance to Step 2** if the scorecard improves on ≥ 2 of 3 primary metrics.
- **Revert** if the scorecard gets worse on ≥ 2 of 3 primary metrics.
- **Document and move on** if results are mixed or neutral; consider `PARAM_STABILITY_WEIGHT = 0.85` as the next attempt before moving to Step 2.
- **Hard floor:** if survivor count drops below 5, treat as a partial revert — back off to `0.5` and re-evaluate.

## Risks

- All currently-surviving coins might fail the tighter stability bar, leaving Phase 2/3 with too few coins. The top-K fallback and history-shrinkage cushions help but don't eliminate this risk.
- Genuinely high-CV edges (e.g. strategies that work in one regime and not another by design) get penalized along with the noise-driven outliers. Acceptable for now; revisit if specific coins we know to be real edges drop out.

## Rollback

Revert the single config line. The cache regenerates on the next run; old entries remain harmless because they're keyed by the now-superseded `0.7`. No data migration needed.

## What this experiment does *not* do

- Does not modify the within-combo ranking to incorporate OOS (that is Step 4 — schema-invasive).
- Does not shrink the param grid (Step 2).
- Does not address the DOGE/XMR sparse-fold problem from `MIN_CLOSED_TRADES_TRAIN` (Step 3).
- Does not re-tune cross-combo blend (`OOS_ROBUSTNESS_BLEND_ALPHA`); cross-combo gate is already at `0.70`.

These remain candidates for subsequent steps depending on Step 1's outcome.

## Multi-step plan reference (informational)

| # | Step | Cost | Expected impact | Status |
|---|---|---|---|---|
| 1 | `PARAM_STABILITY_WEIGHT` 0.3 → 0.7 | full WFO re-run; 1 config line | medium-high | **this spec — NEUTRAL** |
| 1.5 | Per-fold z-rank in `_weighted_robustness_series` (rank cells by mean z-score across folds, not raw weighted mean) | full WFO re-run; small `wfo.py` edit | medium-high (complements 1: catches positional noise where CV catches magnitude noise) | next — Step 1 was a no-op |
| 2 | Shrink param grids (≈ halve combos) | full WFO re-run; grid edits | medium | pending |
| 3 | Address DOGE-style sparse-fold survivors (`MIN_CLOSED_TRADES_TRAIN` calibration per strategy) | analysis + tweak | medium | pending |
| 4 | Per-cell OOS in cache + Layer-1 OOS blend (`IS_OOS_PARAM_BLEND_ALPHA`) | invasive: schema migration + WFO core edit | high | pending |
| 5 | Universe pruning beyond current filters | cheap | low–medium | pending |

Each subsequent step gets its own spec, run, scorecard, and decision-rule check before the next one starts.

---

## Result

Run via small-N micro-experiment (7 diagnostic coins: BTC, ETH, TRX, DOGE, XMR, DASH, ADA) instead of the originally-planned 100-coin run. The 100-coin run was bailed at ~10% progress after 51 minutes when it became clear it was on track for 5+ hours; the micro-experiment finished cleanly in ~50 minutes (Run A + Run B combined). New CLI flag `--symbols` was added in commit `02760a7` to support this.

- **Run A** (`PARAM_STABILITY_WEIGHT=0.7`): `results/research/research_20260508_175631/`
- **Run B** (`PARAM_STABILITY_WEIGHT=0.3`, baseline): `results/research/research_20260508_190409/`

**Primary scorecard delta:**

| Metric | Before (0.3) | After (0.7) | Δ |
|---|---:|---:|---:|
| `n_oos_gt_0_30` | 1 | 1 | 0 |
| `median_fold_consistency` | 0.55 | 0.55 | 0 |
| `n_oos_gt_0` | 2 | 2 | 0 |

**Secondary:** `n_survivors` 4 → 4; `median_is_minus_oos_gap` 1.64 → 1.36 (cosmetic — the gap shrinks because IS scores are scaled down, but the same cells still win).

**Phase 3 sanity:** Total Return −11.67% → −13.12% (within noise on a 4-coin equally-weighted portfolio); BTC B&H benchmark −23.82%.

**Verdict:** **NEUTRAL** by the spec rule (zero of three primary metrics moved). Per-coin OOS values are identical to 3 decimals because Run A and Run B picked the **exact same strategy + exit + parameters** for every surviving coin (TRX, DOGE, ETH, ADA), confirmed by inspecting `worker_*_results.json` in both runs.

**Mechanism (why Step 1 was a no-op):** the CV penalty `(1 − weight × CV)` in `_apply_stability_penalty` (`wfo.py:489-493`) is a uniform multiplicative scale. High-IS-mean cells tend to have low CV (consistently above-zero means → small CV), so they are barely penalized. Low-IS-mean cells have high CV (means near zero → CV explodes), so they are already crushed at the baseline weight. Increasing the weight from 0.3 to 0.7 squeezes both ends harder but does not flip the relative ranking — the IS-best cell stays IS-best, just with a smaller margin. Selection bias is structural; uniform CV scaling can't fix it.

**Implications for next steps:**

- Going to `0.85` is also pointless (same mechanism, just more of it). The spec's NEUTRAL fallback ("consider 0.85 next") is invalidated by the mechanistic finding.
- **Step 1.5 (per-fold z-rank)** is the right next experiment: it normalizes fold-difficulty and re-ranks cells by mean rank-position rather than raw IS mean, which directly *can* flip the winner.
- Step 2 (grid shrink) is the alternative; it reduces selection bias by reducing the number of noise draws.

**Action:** config left at `PARAM_STABILITY_WEIGHT=0.3` (post-Run-B revert). Live trader is unaffected because identical params would have been chosen at either weight. Step 1.5 spec drafted in `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md`.

**Side win:** the small-N approach (7 diagnostic coins, 7 workers, full uncached WFO) ran in ~22 min per iteration vs ~5 hours for the 100-coin equivalent. This is now the recommended pattern for tuning experiments — full 100-coin runs reserved for ratification only.
