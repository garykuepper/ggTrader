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
| 1 | `PARAM_STABILITY_WEIGHT` 0.3 → 0.7 | full WFO re-run; 1 config line | medium-high | **this spec** |
| 2 | Shrink param grids (≈ halve combos) | full WFO re-run; grid edits | medium | pending |
| 3 | Address DOGE-style sparse-fold survivors (`MIN_CLOSED_TRADES_TRAIN` calibration per strategy) | analysis + tweak | medium | pending |
| 4 | Per-cell OOS in cache + Layer-1 OOS blend (`IS_OOS_PARAM_BLEND_ALPHA`) | invasive: schema migration + WFO core edit | high | pending |
| 5 | Universe pruning beyond current filters | cheap | low–medium | pending |

Each subsequent step gets its own spec, run, scorecard, and decision-rule check before the next one starts.
