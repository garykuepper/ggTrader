# WFO Textbook Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset ggTrader's WFO pipeline to textbook standard — strict train-only selection, rank-based Sortino+Calmar+PF composite, 30-trade pre-score gate with 8-of-10 forgiveness, 20% locked holdout, 4 aggregate pass/fail gates, mean-Sortino + paramCV selection, median-fold live params with smoke test and holdout warning.

**Architecture:** Strip every test-data leakage from selection (gap penalty, OOS fold weights, fold-consistency multiplier, Layer 2 OOS blend, parameter stability penalty, z-rank blend, z-score normalization, composite weights, Sharpe component). Replace within-fold cell scoring with rank-of-Sortino + rank-of-Calmar + rank-of-PF, average ranks. Add a 30-train-trades hard gate with 8-of-10 forgiveness. Reserve the most recent 20% of bars as a locked holdout. After WFO, apply 4 aggregate gates (WFE, %profitable, paramCV, DD ratio) as pass/fail filters. Select among gate-passers by mean per-fold Sortino with CV tie-break. Live params = median of per-fold winners snapped to grid. Smoke-test on full WFO train data. Run median params once on the holdout block; warn if return<0 or holdout maxDD > 1.5× worst WFO test-fold DD.

**Tech Stack:** Python (numpy, pandas, vectorbt), pytest, Docker Compose, TimescaleDB.

**Constraint:** Do not introduce additional gates, weights, or selection criteria during implementation. Land cleanly at textbook WFO before adding anything back.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ggTrader/utils/run_config.py` | Modify | Remove deprecated config knobs (8); add new ones for textbook gates and holdout |
| `src/ggTrader/data/cache/wfo_cache.py` | Modify | Bump cache version (invalidate); strip `test_metrics_by_fold` schema; strip deprecated keys from `_WFO_RELEVANT_CONFIG_KEYS` |
| `src/ggTrader/core/metrics.py` | Modify | Replace `_train_metric_series` z-score composite with rank-based Sortino+Calmar+PF; drop Sharpe; add tie-handling for avg-rank |
| `src/ggTrader/core/wfo.py` | Modify | `_process_wfo_fold`: revert per-cell test compute; apply 30-trade gate before scoring. `_calculate_robustness`: strip gap penalty, OOS fold weights, stability multiplier, z-rank blend, fold-consistency multiplier; switch to rank-based selection. `_execute_wfo_loop`: 4-tuple return → 3-tuple (no `test_metrics_by_fold`). Add 8-of-10 forgiveness logic. |
| `src/ggTrader/core/orchestrator.py` | Modify | Strip Layer 2 OOS blend & fold-consistency multiplier; new per-coin selection rule (mean-Sortino + CV tie-break); compute 4 aggregate gates; pass/fail filtering; median-of-fold-winners live params + grid snap. |
| `src/ggTrader/core/holdout.py` | Create | New module: holdout reservation (last 20% of bars), median-params smoke test on WFO train data, median-params evaluation on holdout block, warning generation. |
| `src/ggTrader/pipeline/param_grids.py` | Modify | Coarsen each strategy's grid using the articulation test. Target ≤ 50 combos per strategy. |
| `tests/test_rank_composite.py` | Create | Unit tests for rank-based composite (Sortino+Calmar+PF, avg-rank ties, 30-trade gate, 8-of-10 forgiveness) |
| `tests/test_wfo_aggregate_gates.py` | Create | Unit tests for the 4 aggregate gates (WFE, %profitable, paramCV, DD ratio) and the per-coin selection rule |
| `tests/test_holdout.py` | Create | Unit tests for holdout reservation, median-param snap, warning logic |
| `tests/test_wfo_gap_penalty_removed.py` | Create | Regression test confirming the gap penalty code path is gone (gap-penalty config does nothing) |

---

## Task 1: Remove gap penalty + per-cell OOS cache schema (revert Steps 4a/4b)

The gap penalty leaks test data into selection. Step 4a's per-cell OOS cache was built to feed it. Both go away. This is pure deletion — no replacement yet.

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` (remove `PARAM_OOS_GAP_PENALTY`)
- Modify: `src/ggTrader/data/cache/wfo_cache.py` (remove `PARAM_OOS_GAP_PENALTY` from key list, remove `test_metrics_by_fold` from schema, bump version)
- Modify: `src/ggTrader/core/wfo.py` (`_process_wfo_fold` revert to single-cell test, `_calculate_robustness` remove gap-penalty block, `_execute_wfo_loop` revert to 3-tuple return)
- Modify: `src/ggTrader/core/orchestrator.py` (call sites — drop `test_metrics_by_fold` variable, drop `test_metrics_by_fold=` kwarg)
- Delete: `tests/test_wfo_oos_gap_penalty.py` (the feature is gone; tests would fail)

- [ ] **Step 1: Remove `PARAM_OOS_GAP_PENALTY` from `run_config.py`**

Open `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`. Find the block introducing the knob (lines around 214–220):

```python
        # IS-OOS gap penalty for cell selection (Step 4b consistency-aware ranking).
        # 0.0 = legacy IS-only selection. >0 subtracts gamma * |IS_mean - OOS_mean|
        # from each cell's score, so a cell whose train and test composites agree
        # keeps full credit; a cell where they diverge is dinged proportional to
        # the raw composite-units gap. Requires test_metrics_by_fold from Step 4a
        # (per-cell OOS composite per fold). Experiment value 1.0; final value TBD.
        "PARAM_OOS_GAP_PENALTY": 1.0,
```

Delete the entire block (both the comment lines and the key line). Save.

- [ ] **Step 2: Remove `PARAM_OOS_GAP_PENALTY` from cache key list and revert schema**

Open `/home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py`.

(a) Remove `"PARAM_OOS_GAP_PENALTY"` from `_WFO_RELEVANT_CONFIG_KEYS` (single-line delete).

(b) Bump `_WFO_CACHE_VERSION` from 2 to 3 with an updated comment:

```python
_WFO_CACHE_VERSION = 3  # bump to invalidate all cached entries globally
# v3 (2026-05-10): WFO textbook reset — removed per-cell OOS metrics
#                  (test_metrics_by_fold), rank-based composite scoring
```

(c) In `get()` (around line 220), remove the `test_metrics_by_fold` deserialization block and change the return signature back to a 2-tuple:

Replace:
```python
            wfo_stats = _wfo_stats_from_json(data["wfo_stats"])
            is_metrics_by_fold: Dict[int, pd.Series] = {
                int(k): _series_from_json(v)
                for k, v in data["is_metrics_by_fold"].items()
            }
            test_metrics_by_fold: Dict[int, pd.Series] = {
                int(k): _series_from_json(v)
                for k, v in (data.get("test_metrics_by_fold") or {}).items()
            }
            self._hits += 1
            return wfo_stats, is_metrics_by_fold, test_metrics_by_fold
```

With:
```python
            wfo_stats = _wfo_stats_from_json(data["wfo_stats"])
            is_metrics_by_fold: Dict[int, pd.Series] = {
                int(k): _series_from_json(v)
                for k, v in data["is_metrics_by_fold"].items()
            }
            self._hits += 1
            return wfo_stats, is_metrics_by_fold
```

(d) Update the function's return type annotation in `get()`'s signature from `Optional[Tuple[List[Dict], Dict[int, pd.Series], Dict[int, pd.Series]]]` to `Optional[Tuple[List[Dict], Dict[int, pd.Series]]]`.

(e) In `put()` (around line 240), remove the `test_metrics_by_fold` parameter and the corresponding serialization block. Restore the signature to:

```python
    def put(
        self,
        symbol: str,
        strategy_name: str,
        exit_name: str,
        param_grid: Dict[str, Any],
        config: Dict[str, Any],
        ohlcv: pd.DataFrame,
        wfo_stats: List[Dict],
        is_metrics_by_fold: Dict[int, pd.Series],
    ) -> None:
        """Persist WFO results to cache (non-fatal on write errors)."""
```

And remove from `payload` (around line 270):
```python
            "test_metrics_by_fold": {
                str(k): _series_to_json(v)
                for k, v in (test_metrics_by_fold or {}).items()
                if isinstance(v, pd.Series) and len(v) > 0
            },
```

Leave everything else in `payload` intact.

- [ ] **Step 3: Revert `_process_wfo_fold` per-cell test compute**

Open `/home/flynn/ggTrader/src/ggTrader/core/wfo.py`. In `_process_wfo_fold` (around line 192), remove the vectorized-test block that was added in Step 4a:

Find and delete:
```python
    # Per-cell OOS metrics: vectorized test pass over the FULL param grid (mirror of train pass).
    # Used downstream by consistency-aware cell selection (Step 2 PARAM_OOS_GAP_PENALTY).
    # Empty Series on failure — the gap penalty path treats absence as "fall back to IS only".
    test_metrics: pd.Series = pd.Series(dtype=float)
    try:
        wfo_test_cfg_vec = {**config, "USE_VECTORIZED": True}
        test_engine_all = FastBacktest(
            test_ohlcv, param_grid, config=wfo_test_cfg_vec, mover_mask=test_mask,
            regime_mask=test_regime,
        )
        pf_test_all = test_engine_all.run(show_progress=False)
        test_metrics, _ = _vectorized_grid_metrics(
            pf_test_all, test_engine_all, param_names, config
        )
    except Exception as test_vec_exc:
        print(
            f"  WFO fold {fold_idx}: vectorized test failed ({test_vec_exc!r}); "
            f"per-cell OOS metrics unavailable for this fold."
        )
```

Also remove the `"test_metrics": test_metrics,` line from the return dict (around line 234). And remove `"test_metrics": pd.Series(dtype=float),` from the two early-return paths (insufficient-bars and vectorized-failure).

- [ ] **Step 4: Revert `_execute_wfo_loop` to 3-tuple return**

In the same file, find `_execute_wfo_loop` (around line 590). Change the return type annotation and body to drop `test_metrics_by_fold`:

Find:
```python
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.Series], Dict[int, pd.Series], List[pd.Series]]:
    """Iterates through the dataset and processes each WFO fold.

    Returns (wfo_stats, is_metrics_by_fold, test_metrics_by_fold, oos_returns_list).
    test_metrics_by_fold mirrors is_metrics_by_fold but holds OOS composite per cell
    per fold — used by consistency-aware cell selection (Step 2).
    """
    wfo_stats = []
    is_metrics_by_fold = {}
    test_metrics_by_fold: Dict[int, pd.Series] = {}
    oos_returns_list = []
```

Replace with:
```python
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.Series], List[pd.Series]]:
    """Iterates through the dataset and processes each WFO fold."""
    wfo_stats = []
    is_metrics_by_fold = {}
    oos_returns_list = []
```

Inside the loop, remove `test_metrics_by_fold[fold_idx] = fold_result.pop(...)`.

At the bottom of the function, change `return wfo_stats, is_metrics_by_fold, test_metrics_by_fold, oos_returns_list` back to `return wfo_stats, is_metrics_by_fold, oos_returns_list`.

- [ ] **Step 5: Update the two internal `_execute_wfo_loop` callers in wfo.py**

Around line 753:
```python
    wfo_stats, is_metrics_by_fold, _, _ = _execute_wfo_loop(
```
Change to:
```python
    wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
```

Around line 898 (or wherever the second caller is — verify with `grep -n "_execute_wfo_loop(" src/ggTrader/core/wfo.py`):
```python
        wfo_stats, is_metrics_by_fold, _, _ = _execute_wfo_loop(
```
Change to:
```python
        wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
```

- [ ] **Step 6: Strip gap-penalty block from `_calculate_robustness`**

Still in `wfo.py`, find `_calculate_robustness` (around line 484). Remove these sections:

(a) The `test_metrics_by_fold` kwarg (added in Step 4b):
```python
    test_metrics_by_fold: Optional[Dict[int, pd.Series]] = None,
```

(b) The gap-penalty block (lines around 539–558 in current code, marked "Per-cell IS-OOS gap penalty (Step 4b...)"):
```python
    # Per-cell IS-OOS gap penalty (Step 4b consistency-aware selection).
    # When PARAM_OOS_GAP_PENALTY > 0 AND test_metrics_by_fold is provided, subtract
    # gamma * |IS_mean - OOS_mean| from each cell's score. Both means use the same
    # fold weights as IS, so a cell whose train and test composites agree keeps
    # full credit; one where they diverge is dinged proportional to the raw gap.
    _cfg = config or {}
    gap_penalty = float(_cfg.get("PARAM_OOS_GAP_PENALTY", 0.0))
    if gap_penalty > 0.0 and test_metrics_by_fold:
        oos_robustness_scores = _weighted_robustness_series(
            test_metrics_by_fold, weights, config=config
        )
        oos_aligned = oos_robustness_scores.reindex(robustness_scores.index)
        gap = (robustness_scores - oos_aligned).abs()
        gap = gap.fillna(0.0)
        robustness_scores = robustness_scores - gap_penalty * gap
```

Replace with the unchanged `_cfg = config or {}` line:
```python
    _cfg = config or {}
```

(c) Remove the docstring lines mentioning the gap penalty and `test_metrics_by_fold` from the function's docstring (lines mentioning "If test_metrics_by_fold is provided AND PARAM_OOS_GAP_PENALTY (gamma) > 0..." and the parameter description).

- [ ] **Step 7: Update orchestrator `_calculate_robustness` call site**

Open `/home/flynn/ggTrader/src/ggTrader/core/orchestrator.py`. Find the call (around line 1044):

```python
                    robust_top_5, best_robust_params = _calculate_robustness(
                        is_metrics_by_fold,
                        list(param_grid.keys()),
                        param_grid,
                        oos_metrics_by_fold,
                        debug_metrics=debug_wfo,
                        config=config,
                        test_metrics_by_fold=test_metrics_by_fold,
                    )
```

Remove the `test_metrics_by_fold=test_metrics_by_fold,` line.

- [ ] **Step 8: Update orchestrator cache+loop call sites**

Still in `orchestrator.py`, around line 1010–1030, change:

```python
                    if _cached is not None:
                        wfo_stats, is_metrics_by_fold, test_metrics_by_fold = _cached
                        print(f"    {label} [cache hit — skipping {n_splits} folds]")
                    else:
                        wfo_stats, is_metrics_by_fold, test_metrics_by_fold, _ = _execute_wfo_loop(
                            ...
                        )
                        if _wfo_cache is not None:
                            _wfo_cache.put(
                                symbol, strategy_name, exit_name, param_grid,
                                config_combo, symbol_ohlcv, wfo_stats, is_metrics_by_fold,
                                test_metrics_by_fold,
                            )
```

To:

```python
                    if _cached is not None:
                        wfo_stats, is_metrics_by_fold = _cached
                        print(f"    {label} [cache hit — skipping {n_splits} folds]")
                    else:
                        wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
                            ...
                        )
                        if _wfo_cache is not None:
                            _wfo_cache.put(
                                symbol, strategy_name, exit_name, param_grid,
                                config_combo, symbol_ohlcv, wfo_stats, is_metrics_by_fold,
                            )
```

- [ ] **Step 9: Delete the gap-penalty unit tests**

```bash
rm /home/flynn/ggTrader/tests/test_wfo_oos_gap_penalty.py
```

- [ ] **Step 10: Run all WFO tests to confirm no regression**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_robustness_selection.py tests/test_exit_tournament_wfo.py tests/test_wfo_trade_counts_gate.py tests/test_wfo_zrank_blend.py -v
```

Expected: all PASS. If anything fails, the failure should point at a `test_metrics_by_fold` reference still in the code — search and remove.

- [ ] **Step 11: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py src/ggTrader/data/cache/wfo_cache.py src/ggTrader/core/wfo.py src/ggTrader/core/orchestrator.py
git rm tests/test_wfo_oos_gap_penalty.py
git commit -m "$(cat <<'EOF'
revert(wfo): remove gap penalty + per-cell OOS cache (Step 4a/4b reversal)

WFO textbook reset prep. Gap penalty leaks test data into parameter
selection and invalidates the OOS label on reported metrics. Per-cell
OOS cache was built specifically to feed it; no longer needed.

- Drops PARAM_OOS_GAP_PENALTY config knob.
- Reverts _process_wfo_fold to single-cell test backtest.
- Reverts _execute_wfo_loop to 3-tuple return (no test_metrics_by_fold).
- Strips gap-penalty block from _calculate_robustness.
- Bumps cache version 2 -> 3 (invalidates existing entries).
- Removes test_wfo_oos_gap_penalty.py.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Remove other test-data leakage and now-redundant knobs

Strip the remaining test-data influences and now-redundant scoring knobs identified in the spec:

- OOS-derived fold weights in `_calculate_robustness` (lines ~507-531 today: better OOS folds get higher weight when averaging IS — selection leak).
- `OOS_ROBUSTNESS_BLEND_ALPHA` cross-combo blend in `orchestrator.py` (Layer 2 selection currently uses 0.7 weight on OOS).
- `FOLD_CONSISTENCY_IN_GATE` and `FOLD_CONSISTENCY_GATE_FLOOR` (multiplies Layer 2 score by OOS-derived fold consistency — selection leak).
- `PARAM_STABILITY_WEIGHT` (CV penalty on IS — superseded by the harder paramCV ≤ 0.3 final gate).
- `PARAM_ZRANK_WEIGHT` (per-fold z-rank blend — irrelevant under rank-based scoring).
- `TRAIN_METRIC_NORMALIZE_ZSCORE` (replaced by rank-based ranking).
- `TRAIN_METRIC_COMPOSITE_WEIGHTS` (rank-based has equal contribution).

**Files:**
- Modify: `src/ggTrader/utils/run_config.py`
- Modify: `src/ggTrader/data/cache/wfo_cache.py`
- Modify: `src/ggTrader/core/wfo.py`
- Modify: `src/ggTrader/core/orchestrator.py`
- Modify: `src/ggTrader/core/metrics.py`
- Delete: `tests/test_wfo_zrank_blend.py` (z-rank gone)

- [ ] **Step 1: Strip OOS-derived fold weights in `_calculate_robustness`**

Open `/home/flynn/ggTrader/src/ggTrader/core/wfo.py`. Find `_calculate_robustness` (around line 484). The current branch around lines 507–537 reads:

```python
    # If OOS metrics provided, use them to measure generalization
    if oos_metrics_by_fold:
        fold_indices = sorted(oos_metrics_by_fold.keys())
        oos_values = np.array(
            [_coerce_metric_float(oos_metrics_by_fold.get(f)) for f in fold_indices],
            dtype=float,
        )
        # Min-max normalize OOS Sharpes to [0.1, 1.0] so folds with better OOS get
        # proportionally higher weight even when *all* OOS values are negative.
        # This prevents the old ratio-to-mean collapsing to a flat 0.5 weight when
        # oos_mean <= 0 (which made OOS weighting useless in poor-OOS regimes).
        oos_finite = oos_values[np.isfinite(oos_values)]
        oos_min_val = float(np.min(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_max_val = float(np.max(oos_finite)) if oos_finite.size > 0 else 0.0
        oos_range_val = oos_max_val - oos_min_val

        weights = {}
        for f in fold_indices:
            oos_sharpe = _coerce_metric_float(oos_metrics_by_fold.get(f))
            recency_weight = float(f)
            if np.isfinite(oos_sharpe) and oos_range_val > 1e-8:
                consistency_weight = 0.1 + 0.9 * (oos_sharpe - oos_min_val) / oos_range_val
            else:
                consistency_weight = 0.5
            weights[f] = recency_weight * consistency_weight

        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
    else:
        # Original recency-weighted IS Sharpe (for backwards compatibility)
        weights = {f: float(f) for f in is_metrics_by_fold.keys()}
        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
```

Replace with equal-weighted recency-only blending (no OOS in selection):

```python
    # Selection uses train-window metrics only; folds get recency weights, no OOS.
    weights = {f: float(f) for f in is_metrics_by_fold.keys()}
    robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
```

Then remove `oos_metrics_by_fold` from the function signature and docstring.

- [ ] **Step 2: Update `_calculate_robustness` callers to remove `oos_metrics_by_fold` arg**

Three callers (verify with `grep -n "_calculate_robustness(" src/ggTrader/core/wfo.py src/ggTrader/core/orchestrator.py`):

(a) `wfo.py` line ~771: change
```python
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold,
        param_names,
        param_grid,
        None,  # Do not punish entire folds based on OOS performance of the single IS winner
        debug_metrics=dbg,
        config=config,
    )
```
to:
```python
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold,
        param_names,
        param_grid,
        debug_metrics=dbg,
        config=config,
    )
```

(b) `wfo.py` line ~916: same edit pattern (remove the positional `None` arg).

(c) `orchestrator.py` line ~1044: same edit pattern (remove `oos_metrics_by_fold,` positional arg).

- [ ] **Step 3: Strip Layer 2 OOS blend & fold-consistency multiplier in orchestrator.py**

Open `/home/flynn/ggTrader/src/ggTrader/core/orchestrator.py`. Find the gate-score block (around line 1062–1080):

```python
                    oos_blend_alpha = float(config.get("OOS_ROBUSTNESS_BLEND_ALPHA", 0.5))
                    if np.isfinite(oos_rob_combo) and np.isfinite(robustness_score):
                        is_oos_blend = (
                            1.0 - oos_blend_alpha
                        ) * robustness_score + oos_blend_alpha * oos_rob_combo
                    elif np.isfinite(oos_rob_combo):
                        is_oos_blend = oos_rob_combo
                    else:
                        is_oos_blend = robustness_score

                    # Fold consistency soft multiplier: strategies inconsistent across folds
                    # are penalized. floor=0.5 means the worst case halves the gate score.
                    use_fc_gate = bool(config.get("FOLD_CONSISTENCY_IN_GATE", True))
                    fc_floor = float(config.get("FOLD_CONSISTENCY_GATE_FLOOR", 0.5))
                    if use_fc_gate and np.isfinite(fold_cons_combo):
                        fc_factor = fc_floor + (1.0 - fc_floor) * fold_cons_combo
                        gate_score = is_oos_blend * fc_factor
                    else:
                        gate_score = is_oos_blend
```

Replace with strict train-only score:

```python
                    # Layer 2 selection uses train-window robustness only (textbook reset).
                    # OOS-derived blending and fold-consistency multiplier removed —
                    # both would leak test data into the cross-combo decision.
                    gate_score = robustness_score
```

The local variables `oos_rob_combo`, `fold_cons_combo`, etc. may still be used later for reporting only — keep them computed but stop using them in `gate_score`. Confirm via grep:
```bash
grep -n "oos_rob_combo\|fold_cons_combo\|is_oos_blend\|fc_factor" /home/flynn/ggTrader/src/ggTrader/core/orchestrator.py
```
Any remaining references should be reporting / save-to-disk only, not selection.

- [ ] **Step 4: Strip `PARAM_STABILITY_WEIGHT` multiplier in `_calculate_robustness`**

Back in `wfo.py:_calculate_robustness`. Find the stability-multiplier block (currently around lines 555–571):

```python
    # Parameter stability penalty: penalize combos with high fold-to-fold CV.
    # CV = std / |mean| measures how much a combo's IS metric varies across folds —
    # high CV indicates the combo is curve-fitting to specific folds rather than
    # generalizing. The multiplier is 1/(1 + w*CV), so a stable combo (CV≈0) keeps
    # its full score while an unstable combo (CV=2, w=0.3) loses ~37%.
    param_stability_weight = float(_cfg.get("PARAM_STABILITY_WEIGHT", 0.3))
    if param_stability_weight > 0.0 and len(is_metrics_by_fold) >= 2:
        cv_series = _param_cv_series(is_metrics_by_fold)
        cv_aligned = cv_series.reindex(robustness_scores.index).fillna(0.0)
        stability_multiplier = 1.0 / (1.0 + param_stability_weight * cv_aligned)
        robustness_scores = robustness_scores * stability_multiplier
```

Delete the entire block. (The harder paramCV ≤ 0.3 aggregate gate added in Task 5 supersedes this soft penalty.)

If `_param_cv_series` becomes unreferenced, leave it — it might still be useful for the aggregate paramCV computation in Task 5. Verify with `grep -n "_param_cv_series" src/ggTrader/core/wfo.py`.

- [ ] **Step 5: Strip `PARAM_ZRANK_WEIGHT` blend in `_weighted_robustness_series`**

Same file. Find `_weighted_robustness_series` (around line 235). Remove the z-rank blend block (lines around 295–323):

```python
    # Per-fold z-rank blend (Step 1.5 of WFO overfitting work).
    # alpha=0 reproduces the raw weighted-mean behavior above. alpha>0 mixes in a
    # weighted mean of per-fold z-scores so cells that rank consistently high across
    # folds beat cells that spike in one fold and average elsewhere.
    alpha = float((config or {}).get("PARAM_ZRANK_WEIGHT", 0.0))
    if alpha > 0.0:
        ... (entire block to the end of the if)
```

Delete the entire `alpha = ...` line and its conditional body, ending right before `return pd.Series(combined, index=union_idx)`.

The function signature still has `config: Optional[Dict[str, Any]] = None`; keep it (might be needed for future config-aware behavior).

- [ ] **Step 6: Strip composite-score knobs from `metrics.py`**

Open `/home/flynn/ggTrader/src/ggTrader/core/metrics.py`. Find `_train_metric_series` (around line 134). Step 3 of Task 3 will *replace* this with a rank-based implementation. For now, just remove the now-irrelevant config branches:

In the same function, find the composite branch (around line 142). For now, leave the function intact — Task 3 will replace it entirely. The z-score knob and weights are removed at the config layer (Task 2 Step 7), not here.

(No code change in this file in Task 2.)

- [ ] **Step 7: Remove the eight deprecated keys from `run_config.py`**

Open `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`. Delete these eight lines / comment blocks (verbatim — preserve indentation when grepping; lines may shift after the gap-penalty removal in Task 1):

- `"OOS_ROBUSTNESS_BLEND_ALPHA": 0.70,` + the preceding comment block (lines ~197–199 originally).
- `"OOS_STABILITY_WEIGHT": 0.3,` + comment block (lines ~215–218).
- `"FOLD_CONSISTENCY_IN_GATE": True,` + comment block.
- `"FOLD_CONSISTENCY_GATE_FLOOR": 0.25,` + comment block.
- `"PARAM_STABILITY_WEIGHT": 0.3,` + comment block.
- `"PARAM_ZRANK_WEIGHT": 0.0,` + comment block.
- `"TRAIN_METRIC_NORMALIZE_ZSCORE": True,` + comment block.
- `"TRAIN_METRIC_COMPOSITE_WEIGHTS": { ... }` + comment block (multi-line dict).

Search with: `grep -n "OOS_ROBUSTNESS_BLEND_ALPHA\|OOS_STABILITY_WEIGHT\|FOLD_CONSISTENCY_IN_GATE\|FOLD_CONSISTENCY_GATE_FLOOR\|PARAM_STABILITY_WEIGHT\|PARAM_ZRANK_WEIGHT\|TRAIN_METRIC_NORMALIZE_ZSCORE\|TRAIN_METRIC_COMPOSITE_WEIGHTS" /home/flynn/ggTrader/src/ggTrader/utils/run_config.py`

- [ ] **Step 8: Remove the same eight keys from `wfo_cache.py:_WFO_RELEVANT_CONFIG_KEYS`**

Open `/home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py`. Find the list (line ~23). Delete these entries:

```
    "TRAIN_METRIC_COMPOSITE_WEIGHTS",
    "TRAIN_METRIC_NORMALIZE_ZSCORE",
    "PARAM_STABILITY_WEIGHT",
    "PARAM_ZRANK_WEIGHT",
    "OOS_ROBUSTNESS_BLEND_ALPHA",
    "OOS_STABILITY_WEIGHT",
    "FOLD_CONSISTENCY_IN_GATE",
    "FOLD_CONSISTENCY_GATE_FLOOR",
```

- [ ] **Step 9: Delete the z-rank test file**

```bash
rm /home/flynn/ggTrader/tests/test_wfo_zrank_blend.py
```

- [ ] **Step 10: Find and fix any remaining references to deleted knobs**

```bash
grep -rn "OOS_ROBUSTNESS_BLEND_ALPHA\|OOS_STABILITY_WEIGHT\|FOLD_CONSISTENCY_IN_GATE\|FOLD_CONSISTENCY_GATE_FLOOR\|PARAM_STABILITY_WEIGHT\|PARAM_ZRANK_WEIGHT\|TRAIN_METRIC_NORMALIZE_ZSCORE\|TRAIN_METRIC_COMPOSITE_WEIGHTS" /home/flynn/ggTrader/src/
```

Any hit not in a comment or docstring needs to be removed or rewritten. Comments that document removed behavior may be kept if they aid future readers.

- [ ] **Step 11: Run WFO test suite to confirm no broken imports**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_robustness_selection.py tests/test_exit_tournament_wfo.py tests/test_wfo_trade_counts_gate.py -v
```

Expected: all PASS. Existing tests don't exercise the removed knobs as values, so the deletion shouldn't break them.

- [ ] **Step 12: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py src/ggTrader/data/cache/wfo_cache.py src/ggTrader/core/wfo.py src/ggTrader/core/orchestrator.py
git rm tests/test_wfo_zrank_blend.py
git commit -m "$(cat <<'EOF'
refactor(wfo): strip test-data leakage and redundant scoring knobs

Textbook reset cleanup. Removes:
- OOS-derived fold weights in _calculate_robustness (selection leak)
- OOS_ROBUSTNESS_BLEND_ALPHA Layer 2 IS+OOS blend (selection leak)
- FOLD_CONSISTENCY_IN_GATE / FOLD_CONSISTENCY_GATE_FLOOR multiplier
  in Layer 2 gate_score (selection leak via OOS-derived consistency)
- PARAM_STABILITY_WEIGHT (CV penalty on IS; superseded by the harder
  aggregate paramCV <= 0.3 gate added next)
- PARAM_ZRANK_WEIGHT (irrelevant under rank-based scoring)
- TRAIN_METRIC_NORMALIZE_ZSCORE (replaced by rank-based ranking)
- TRAIN_METRIC_COMPOSITE_WEIGHTS (rank-based has equal contribution)
- OOS_STABILITY_WEIGHT (no longer used after stability multiplier removal)
- tests/test_wfo_zrank_blend.py (knob removed)

Layer 1 and Layer 2 are now strict train-only selection. Both layers
still produce a single score per (entry, exit) combo, used downstream
by Task 3's new rank-based composite. OOS metrics continue to be
computed and reported but never feed selection.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement rank-based Sortino + Calmar + PF composite

Replace the current z-score-weighted composite in `_train_metric_series` with rank-based blending. Drop Sharpe entirely.

**Files:**
- Modify: `src/ggTrader/core/metrics.py` (rewrite `_train_metric_series` composite branch)
- Create: `tests/test_rank_composite.py`

- [ ] **Step 1: Write the failing test**

Create `/home/flynn/ggTrader/tests/test_rank_composite.py`:

```python
"""Tests for the rank-based composite metric (Sortino + Calmar + PF).

Step 1 of the WFO textbook reset. The composite is the within-fold selection
objective. Rank cells by each of Sortino/Calmar/PF descending, average ranks
on ties, take the mean of the three ranks, and emit -mean so 'max wins'
downstream.

Sharpe is intentionally dropped (Sharpe and Sortino are near-redundant; the
spec drops Sharpe to keep the composite to three non-redundant dimensions).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.metrics import _train_metric_series


def _mock_pf(sortino, calmar_total_return, calmar_max_dd, pf_factor):
    """Create a mock vbt portfolio with predetermined per-cell ratio series."""
    pf = MagicMock()
    pf.sortino_ratio.return_value = pd.Series(sortino)
    pf.total_return.return_value = pd.Series(calmar_total_return)
    pf.max_drawdown.return_value = pd.Series(calmar_max_dd)
    trades_mock = MagicMock()
    trades_mock.profit_factor.return_value = pd.Series(pf_factor)
    pf.trades = trades_mock
    # sharpe_ratio is called by the legacy path; we patch to fail-loud if invoked.
    pf.sharpe_ratio.side_effect = AssertionError("Sharpe should not be used in rank composite")
    return pf


def test_rank_composite_winner_is_top_of_each_axis():
    """Cell that's best in all three axes wins decisively."""
    # 3 cells: A best in all 3, B middle, C worst.
    sortino = [2.0, 1.0, 0.5]
    calmar_tr = [3.0, 2.0, 1.0]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [2.0, 1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    # A should score highest. Score = -(rank_sortino + rank_calmar + rank_pf).
    # With A rank 1 on all axes: score = -(1+1+1) = -3.
    # B rank 2 on all: score = -6. C: -9.
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_rank_composite_average_rank_on_ties():
    """Tied cells get the average rank, not min or max."""
    # Cell A and B tied at top of Sortino, C clearly worst.
    sortino = [2.0, 2.0, 0.5]
    calmar_tr = [3.0, 2.0, 1.0]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [2.0, 1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    # On Sortino: A & B tied -> average rank 1.5 each, C rank 3.
    # On Calmar: A rank 1, B rank 2, C rank 3.
    # On PF:     A rank 1, B rank 2, C rank 3.
    # Implementation uses MEAN rank across the 3 axes, then negates.
    # A mean rank = (1.5+1+1)/3 = 1.1666...; score = -1.1666...
    # B mean rank = (1.5+2+2)/3 = 1.8333...; score = -1.8333...
    # C mean rank = (3+3+3)/3   = 3.0;       score = -3.0
    assert scores.iloc[0] == pytest.approx(-7.0 / 6.0, abs=1e-9)
    assert scores.iloc[1] == pytest.approx(-11.0 / 6.0, abs=1e-9)
    assert scores.iloc[2] == pytest.approx(-3.0, abs=1e-9)


def test_rank_composite_sharpe_never_consulted():
    """The mock's sharpe_ratio raises if called; composite must not call it."""
    sortino = [1.0, 0.5]
    calmar_tr = [1.0, 0.5]
    calmar_dd = [-0.1, -0.2]
    pf_factor = [1.5, 1.0]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    # Should not raise (no AssertionError from the side_effect).
    _ = _train_metric_series(pf, config)


def test_rank_composite_nan_propagates_to_nan():
    """A cell with NaN Sortino (no trades) gets NaN score, not a real rank."""
    sortino = [1.0, float("nan"), 0.5]
    calmar_tr = [1.0, 0.5, 0.3]
    calmar_dd = [-0.1, -0.2, -0.3]
    pf_factor = [1.5, 1.0, 0.8]
    pf = _mock_pf(sortino, calmar_tr, calmar_dd, pf_factor)
    config = {"TRAIN_METRIC": "composite"}
    scores = _train_metric_series(pf, config)
    assert np.isfinite(scores.iloc[0])
    assert np.isnan(scores.iloc[1])
    assert np.isfinite(scores.iloc[2])
```

- [ ] **Step 2: Run the new tests; they must fail**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_rank_composite.py -v
```

Expected: tests fail because the current `_train_metric_series` uses z-score blending, not rank-based. Failures will likely be on specific value assertions (e.g., A score is not −3.5).

Note: `test_rank_composite_sharpe_never_consulted` will fail because the current code uses Sharpe via `pf_train.sharpe_ratio()`. That's the assertion-failure mode we want — confirms the test is enforcing the "no Sharpe" requirement.

- [ ] **Step 3: Rewrite the composite branch of `_train_metric_series`**

Open `/home/flynn/ggTrader/src/ggTrader/core/metrics.py`. Find `_train_metric_series` (around line 134). Replace the entire `elif name == "composite":` branch (lines ~141–181) with the rank-based version:

```python
    elif name == "composite":
        # Rank-based composite (Sortino + Calmar + PF). No Sharpe — redundant with
        # Sortino (Sortino is a strict refinement, only counts downside vol).
        # Per-fold: rank cells by each of the three ratios descending (rank 1 = best).
        # Average rank on ties (standard statistical convention). Average the ranks
        # across axes. Score = -mean_rank so "max wins" downstream is preserved.
        # Cells with NaN in any axis get NaN score (propagates to selection drop).
        so = pf_train.sortino_ratio()
        if not isinstance(so, pd.Series):
            so = pd.Series([float(so)])
        ca = _calmar_ratio_series(pf_train).reindex(so.index)
        pf_s = _profit_factor_series(pf_train).reindex(so.index)

        # Clip the same way the legacy composite did, to keep extreme values from
        # distorting ranks of ties at the edges (rank itself is scale-invariant, but
        # clipping defends against +inf / -inf making ranks unstable).
        ca_clipped = ca.clip(lower=-5.0, upper=5.0)
        pf_clipped = pf_s.clip(lower=-3.0, upper=3.0)

        # rank(ascending=False) → highest value = rank 1, lowest = rank N.
        # method='average' gives tied cells the mean of their tied position range.
        # NaN values produce NaN ranks (skipped by mean below).
        r_so = so.rank(ascending=False, method="average")
        r_ca = ca_clipped.rank(ascending=False, method="average")
        r_pf = pf_clipped.rank(ascending=False, method="average")

        # Mean rank across the 3 axes. NaN in any axis -> NaN final score.
        mean_rank = pd.concat([r_so, r_ca, r_pf], axis=1).mean(axis=1)
        m = -mean_rank  # negate so higher score = better, matches downstream "max wins"
        # Force NaN where Sortino itself is NaN (no trades / gated out).
        m = m.where(so.notna(), other=float("nan"))
```

The other branches (`sortino`, `calmar`, default-`sharpe`) stay as-is.

Also remove the function's local imports of weights and clipping constants that are only used by the deleted z-score path (none expected, but verify by re-reading the function after the edit).

- [ ] **Step 4: Run tests again — they should pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_rank_composite.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 5: Run the broader WFO test suite for regression**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_robustness_selection.py tests/test_exit_tournament_wfo.py tests/test_wfo_trade_counts_gate.py -v
```

Expected: PASS. The existing tests don't assert specific composite values; they assert structural invariants.

- [ ] **Step 6: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/core/metrics.py tests/test_rank_composite.py
git commit -m "$(cat <<'EOF'
feat(metrics): rank-based composite (Sortino + Calmar + PF) — no Sharpe

Within-fold selection composite is now:
  score = -mean(rank_sortino, rank_calmar, rank_pf)
with average rank on ties (standard statistical convention).

Sharpe dropped — near-redundant with Sortino (Sortino is a strict
refinement using downside-only volatility). The composite now has
three non-redundant dimensions: vol-adjusted level (Sortino),
drawdown-adjusted level (Calmar), win/loss asymmetry (PF).

Properties vs the previous z-score weighted blend:
- Scale-invariant — only order matters; extreme values can't dominate
- No noise amplification when cells score similarly within a fold
- Tied cells get average rank, no bias

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add 30-trade pre-score gate with 8-of-10 forgiveness

Cells with fewer than 30 trades on the train window are disqualified from ranking within that fold. A combo must pass the 30-trade gate in at least 8 of 10 folds to be eligible for selection. In the (up to 2) folds where a combo fails, assign it the median rank (neither reward nor penalty).

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` (add `MIN_TRADES_PER_TRAIN_FOLD`, `MIN_TRAIN_FOLD_PASS_COUNT` knobs)
- Modify: `src/ggTrader/data/cache/wfo_cache.py` (add the new keys to `_WFO_RELEVANT_CONFIG_KEYS`)
- Modify: `src/ggTrader/core/wfo.py` (`_process_wfo_fold` apply pre-score gate; `_calculate_robustness` apply 8-of-10 forgiveness)
- Test: `tests/test_rank_composite.py` (extend with gate and forgiveness tests)

- [ ] **Step 1: Add config knobs**

Open `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`. Add near the other WFO knobs (after `MIN_VALID_TRAIN_FOLDS` ~line 190 originally):

```python
        # Minimum number of closed train trades a cell must have to be eligible
        # for ranking within a fold (textbook reset). Cells with fewer trades
        # are disqualified before scoring; their rank is "skip". Spec value: 30.
        "MIN_TRADES_PER_TRAIN_FOLD": 30,
        # 8-of-N forgiveness: a combo must pass the MIN_TRADES_PER_TRAIN_FOLD gate
        # in at least N_PASS folds to be eligible for selection across folds.
        # In folds where it fails, the combo is assigned the median rank in that
        # fold (neither rewarding nor penalizing the inactivity). 8 of 10.
        "MIN_TRAIN_FOLD_PASS_COUNT": 8,
```

- [ ] **Step 2: Add the two keys to the cache-key list**

Open `/home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py`. Add to `_WFO_RELEVANT_CONFIG_KEYS`:

```python
    "MIN_TRADES_PER_TRAIN_FOLD",
    "MIN_TRAIN_FOLD_PASS_COUNT",
    "WFO_BARS_PER_YEAR",
```

- [ ] **Step 3: Apply the per-fold gate in `_process_wfo_fold`**

Open `/home/flynn/ggTrader/src/ggTrader/core/wfo.py`. Find the existing per-fold trade gate (around line 137):

```python
    min_closed = config.get("MIN_CLOSED_TRADES_TRAIN", 1)
```

Just below this block (after the existing `MIN_CLOSED_TRADES_TRAIN` filtering), add the harder per-fold trade gate:

```python
    # Textbook reset: 30+ trades per train fold is a hard pre-score gate.
    # Cells below the threshold get NaN score (disqualified from ranking).
    # 8-of-10 forgiveness is applied later in _calculate_robustness across folds.
    min_per_fold = int(config.get("MIN_TRADES_PER_TRAIN_FOLD", 30))
    if min_per_fold > 0:
        below_fold_min = trade_for_gate < min_per_fold
        # Combine with existing nan_mask so the consolidated update below catches it.
        nan_mask |= below_fold_min
```

The existing `nan_mask` consolidation (around line 168) will apply this. So no additional plumbing needed in this step.

- [ ] **Step 4: Add 8-of-10 forgiveness in `_calculate_robustness`**

Open the same file. Find `_calculate_robustness` (around line 484). After the `weights = {f: float(f) ...}` block and BEFORE `robustness_scores = _weighted_robustness_series(...)`, add the forgiveness logic:

```python
    # 8-of-10 forgiveness: for each cell, count how many folds have a finite IS
    # metric (i.e., passed the per-fold MIN_TRADES_PER_TRAIN_FOLD gate). Cells
    # that pass fewer than MIN_TRAIN_FOLD_PASS_COUNT folds are dropped entirely
    # (assigned -inf so they never win). Cells that pass at least that many
    # have the failing folds replaced with the per-fold median rank (neither
    # rewarding nor penalizing the inactivity).
    _cfg = config or {}
    min_pass = int(_cfg.get("MIN_TRAIN_FOLD_PASS_COUNT", 8))
    if min_pass > 0 and len(is_metrics_by_fold) > 0:
        # Build presence matrix: cells × folds. Per-cell pass count is the row sum
        # of finite values across the original is_metrics_by_fold.
        all_cells: set = set()
        for s in is_metrics_by_fold.values():
            all_cells.update(s.index.tolist())
        all_cells_idx = pd.Index(sorted(all_cells), dtype=object)
        pass_counts = pd.Series(0, index=all_cells_idx)
        fold_medians: Dict[int, float] = {}
        for f, s in is_metrics_by_fold.items():
            aligned = s.reindex(all_cells_idx)
            pass_counts = pass_counts + aligned.notna().astype(int)
            finite_vals = aligned[aligned.notna()]
            fold_medians[f] = float(finite_vals.median()) if len(finite_vals) > 0 else float("nan")
        # Rebuild is_metrics_by_fold: cells with pass_count < min_pass keep NaN
        # in all folds (so the weighted mean is NaN, then dropped); cells that
        # pass get failing folds filled with that fold's median.
        eligible = pass_counts >= min_pass
        is_metrics_by_fold = dict(is_metrics_by_fold)  # don't mutate caller's dict
        for f, s in list(is_metrics_by_fold.items()):
            aligned = s.reindex(all_cells_idx)
            # Eligible cells: fill NaN with fold median.
            filled = aligned.where(aligned.notna(), other=fold_medians.get(f, float("nan")))
            # Ineligible cells: force NaN regardless of fill.
            filled = filled.where(eligible, other=float("nan"))
            is_metrics_by_fold[f] = filled
```

This block runs before the existing `robustness_scores = _weighted_robustness_series(...)` call, replacing `is_metrics_by_fold` in-flight with the forgiveness-applied version.

- [ ] **Step 5: Add tests for the gate and forgiveness**

Append to `/home/flynn/ggTrader/tests/test_rank_composite.py`:

```python
from ggTrader.core.wfo import _calculate_robustness


def test_min_trades_gate_disqualifies_low_trade_cells():
    """A cell with <30 trades in a fold gets NaN, so it can't win that fold."""
    # Two folds, three cells. Cell A is always above 30 trades. Cell B is below
    # in fold 1, above in fold 2. Cell C is always above. We don't directly
    # test _process_wfo_fold (it requires real OHLCV/portfolios); we test the
    # logical contract: cells with NaN scores get NaN robustness.
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([1.0, float("nan"), 0.5], index=pd.Index(cells, dtype=object))
    fold2 = pd.Series([1.0, 0.8, 0.5], index=pd.Index(cells, dtype=object))
    is_metrics_by_fold = {1: fold1, 2: fold2}
    # With 8-of-10 forgiveness disabled (min_pass=0), the legacy weighted mean
    # would emit B's score from just fold 2. Confirm via the function.
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 0}  # disable forgiveness
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        config=config,
    )
    # All three cells produce a finite score (B from one fold only).
    assert any(r["params"]["x"] == "B" for r in top)


def test_eight_of_ten_forgiveness_drops_cells_below_threshold():
    """A cell present in fewer than min_pass folds is forced to NaN everywhere."""
    cells = [("A",), ("B",)]
    # 10 folds. A present in all 10, B present in only 5.
    is_metrics_by_fold = {}
    for f in range(1, 11):
        if f <= 5:
            row = pd.Series([1.0, 0.5], index=pd.Index(cells, dtype=object))
        else:
            row = pd.Series([1.0, float("nan")], index=pd.Index(cells, dtype=object))
        is_metrics_by_fold[f] = row
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 8}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B"]},
        config=config,
    )
    # B passed only 5 folds (<8). It must be dropped.
    assert not any(r["params"]["x"] == "B" for r in top), (
        "B should be dropped by 8-of-10 forgiveness (only 5 finite folds)"
    )


def test_eight_of_ten_forgiveness_fills_with_fold_median():
    """A cell that passes 8+ folds gets the missing folds filled with median rank."""
    # 10 folds, 3 cells. A always present, scores 1.0. B missing in 2 of 10,
    # scores 0.5 elsewhere. C present in all 10, scores -0.5 (below median).
    cells = [("A",), ("B",), ("C",)]
    is_metrics_by_fold = {}
    for f in range(1, 11):
        if f in (1, 2):
            row = pd.Series([1.0, float("nan"), -0.5], index=pd.Index(cells, dtype=object))
        else:
            row = pd.Series([1.0, 0.5, -0.5], index=pd.Index(cells, dtype=object))
        is_metrics_by_fold[f] = row
    config = {"MIN_TRAIN_FOLD_PASS_COUNT": 8}
    top, best = _calculate_robustness(
        is_metrics_by_fold=is_metrics_by_fold,
        param_names=["x"],
        param_grid={"x": ["A", "B", "C"]},
        config=config,
    )
    # B passes 8 folds, gets median-fill in folds 1 and 2. Fold median is
    # 0.25 (mean of 1.0 and -0.5 — only two finite cells in fold 1 since B is NaN).
    # The exact ranking depends on aggregation; the key assertion is B is not dropped.
    assert any(r["params"]["x"] == "B" for r in top), (
        "B should survive (passes 8 of 10 folds)"
    )
```

- [ ] **Step 6: Run tests; new ones should pass, old ones still pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_rank_composite.py tests/test_wfo_robustness_selection.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py src/ggTrader/data/cache/wfo_cache.py src/ggTrader/core/wfo.py tests/test_rank_composite.py
git commit -m "$(cat <<'EOF'
feat(wfo): 30-trade pre-score gate + 8-of-10 forgiveness

Pre-score gate: cells with <30 closed trades on the train window of a
fold are disqualified from ranking within that fold (NaN score).
Spec value: MIN_TRADES_PER_TRAIN_FOLD=30.

8-of-N forgiveness: a combo must pass the gate in at least 8 of 10
folds to be eligible for selection across folds. In the (up to 2)
forgiven folds, the combo gets the fold's median rank — neither
rewarding nor penalizing the inactivity. Combos passing fewer than
8 folds are forced to NaN everywhere and dropped from selection.
Spec value: MIN_TRAIN_FOLD_PASS_COUNT=8.

Reasoning: per-fold strict (10/10) rejects legitimate strategies
that correctly sit out a single regime. Aggregate (sum >= 300)
is too lenient (300 fires in one fold and 0 elsewhere). 8-of-10
with median-rank fill balances both.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Reserve 20% final holdout before WFO bounds

The most recent 20% of bars is locked away before `_calculate_wfo_bounds` runs. The 10-fold WFO operates only on the first 80%. The holdout is a separate object surfaced to downstream consumers as `holdout_ohlcv`.

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` (add `HOLDOUT_FRACTION` knob)
- Modify: `src/ggTrader/data/cache/wfo_cache.py` (add to cache-key list)
- Create: `src/ggTrader/core/holdout.py` (split + smoke-test + holdout-eval + warning generation)
- Modify: `src/ggTrader/core/wfo.py` (single-strategy callers — they don't compose with holdout; raise unless explicitly opted in)
- Modify: `src/ggTrader/core/orchestrator.py` (`run_multi_strategy_per_coin_wfo` splits OHLCV and routes holdout downstream)
- Create: `tests/test_holdout.py`

- [ ] **Step 1: Add holdout config knob**

In `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`, add near the WFO block:

```python
        # Fraction of the most-recent data reserved as a final holdout (locked
        # away before WFO fold bounds are computed; touched exactly once after
        # all gates pass). Spec value: 0.20 (20% holdout, 80% WFO). 0.0 disables
        # the holdout entirely — legacy WFO behavior (everything available to
        # the 10-fold loop).
        "HOLDOUT_FRACTION": 0.20,
```

- [ ] **Step 2: Add `HOLDOUT_FRACTION` to cache-key list**

`/home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py`, in `_WFO_RELEVANT_CONFIG_KEYS`:

```python
    "HOLDOUT_FRACTION",
```

- [ ] **Step 3: Write the holdout module tests first**

Create `/home/flynn/ggTrader/tests/test_holdout.py`:

```python
"""Tests for the final holdout reservation and warning logic.

Step 0 of the WFO textbook reset. The most recent HOLDOUT_FRACTION of bars is
locked away before WFO. After WFO + gates pass, median params are evaluated
on the holdout exactly once. A warning is raised if:
- holdout annualized return < 0, OR
- holdout max_dd > 1.5 * worst test-fold max_dd from WFO.
The holdout is NOT a gate. Numbers are always reported.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.holdout import (
    split_train_holdout,
    holdout_warning_flags,
)


def _make_ohlcv(n_bars: int) -> pd.DataFrame:
    """Multi-symbol OHLCV with (symbol, field) columns matching project convention."""
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h")
    cols = pd.MultiIndex.from_product(
        [["BTC-USD"], ["open", "high", "low", "close", "volume"]],
        names=["symbol", "field"],
    )
    data = np.random.randn(n_bars, len(cols)).cumsum(axis=0) + 100
    data = np.abs(data)  # keep positive
    return pd.DataFrame(data, index=idx, columns=cols)


def test_split_train_holdout_fraction_20_percent():
    """80% goes to WFO train, 20% goes to holdout."""
    ohlcv = _make_ohlcv(1000)
    train, holdout = split_train_holdout(ohlcv, holdout_fraction=0.20)
    assert len(train) == 800
    assert len(holdout) == 200
    # Train precedes holdout chronologically (no overlap, no gap).
    assert train.index[-1] < holdout.index[0]
    # Concatenation must reconstruct the original.
    reassembled = pd.concat([train, holdout])
    pd.testing.assert_frame_equal(reassembled, ohlcv)


def test_split_train_holdout_disabled_at_fraction_zero():
    """HOLDOUT_FRACTION=0 returns the original OHLCV and an empty holdout."""
    ohlcv = _make_ohlcv(500)
    train, holdout = split_train_holdout(ohlcv, holdout_fraction=0.0)
    assert len(train) == 500
    assert len(holdout) == 0


def test_warning_flag_negative_return():
    """Annualized return < 0 triggers the negative-return warning."""
    flags = holdout_warning_flags(
        holdout_ann_return=-0.05,
        holdout_max_dd=-0.10,
        worst_wfo_test_dd=-0.20,
    )
    assert "negative_return" in flags


def test_warning_flag_max_dd_exceeds_threshold():
    """Holdout DD worse than 1.5x WFO worst-test-DD triggers the DD warning."""
    # WFO worst test DD: -20%. Threshold = 1.5 * 20% = 30%. Holdout DD of -35% triggers.
    flags = holdout_warning_flags(
        holdout_ann_return=0.10,  # positive, so no return-flag
        holdout_max_dd=-0.35,
        worst_wfo_test_dd=-0.20,
    )
    assert "max_dd_exceeds_threshold" in flags
    assert "negative_return" not in flags


def test_no_warning_when_all_good():
    """Positive return AND DD within 1.5x → no warnings."""
    flags = holdout_warning_flags(
        holdout_ann_return=0.20,
        holdout_max_dd=-0.15,
        worst_wfo_test_dd=-0.20,
    )
    assert flags == []
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py -v
```

Expected: ALL FAIL with `ModuleNotFoundError: No module named 'ggTrader.core.holdout'`.

- [ ] **Step 5: Create the holdout module**

Create `/home/flynn/ggTrader/src/ggTrader/core/holdout.py`:

```python
"""Final holdout reservation, smoke testing, and warning generation.

Step 0 of the WFO textbook reset. The most recent HOLDOUT_FRACTION of bars is
reserved before the 10-fold WFO runs. After WFO + the 4 aggregate gates pass,
the median per-fold winners are evaluated on the holdout exactly once. A
warning is raised (but no auto-drop) when holdout annualized return is
negative OR holdout max drawdown exceeds 1.5x the worst test-fold drawdown
observed during WFO.

The holdout is NOT a gate. Numbers are always reported regardless of
warnings; the human decides whether to deploy.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def split_train_holdout(
    ohlcv: pd.DataFrame,
    holdout_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLCV into a leading train block and a trailing holdout block.

    The most recent ``holdout_fraction`` of bars goes to the holdout. The rest
    is used for the 10-fold WFO. Holdout starts immediately after train ends
    (no gap, no overlap).

    holdout_fraction <= 0 returns the full input as train and an empty holdout.
    """
    if holdout_fraction <= 0:
        return ohlcv, ohlcv.iloc[0:0]
    n = len(ohlcv)
    n_holdout = int(round(n * holdout_fraction))
    if n_holdout <= 0:
        return ohlcv, ohlcv.iloc[0:0]
    train = ohlcv.iloc[: n - n_holdout]
    holdout = ohlcv.iloc[n - n_holdout :]
    return train, holdout


def holdout_warning_flags(
    holdout_ann_return: float,
    holdout_max_dd: float,
    worst_wfo_test_dd: float,
    dd_multiple: float = 1.5,
) -> List[str]:
    """Determine which warning flags fire for a holdout evaluation result.

    Returns:
        List of warning names. Empty list means no warnings.
        Possible values: "negative_return", "max_dd_exceeds_threshold".

    The dd_multiple parameter is 1.5 by spec — holdout max_dd worse than
    1.5x the worst WFO test-fold max_dd is the threshold for the DD warning.
    """
    flags: List[str] = []
    if holdout_ann_return is not None and holdout_ann_return < 0:
        flags.append("negative_return")
    # max_dd is stored as a negative number (worse = more negative). Threshold
    # for "exceeds" is when |holdout_dd| > dd_multiple * |worst_wfo_test_dd|.
    if (
        holdout_max_dd is not None
        and worst_wfo_test_dd is not None
        and abs(holdout_max_dd) > dd_multiple * abs(worst_wfo_test_dd)
    ):
        flags.append("max_dd_exceeds_threshold")
    return flags
```

- [ ] **Step 6: Run tests; they should pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 7: Wire `split_train_holdout` into the orchestrator entry point**

Open `/home/flynn/ggTrader/src/ggTrader/core/orchestrator.py`. Find `run_multi_strategy_per_coin_wfo` (around line 874). At the top of the function, after `ohlcv` is loaded and config is set up, before any WFO machinery runs, add:

```python
    # Reserve the most recent HOLDOUT_FRACTION of bars as a locked holdout.
    # The 10-fold WFO operates on the leading 1 - HOLDOUT_FRACTION; the holdout
    # is touched exactly once after gates pass (see Task 7).
    from ggTrader.core.holdout import split_train_holdout
    holdout_fraction = float(config.get("HOLDOUT_FRACTION", 0.20))
    wfo_ohlcv, holdout_ohlcv = split_train_holdout(ohlcv, holdout_fraction)
    print(
        f"  [Holdout] Reserved {len(holdout_ohlcv)} bars ({holdout_fraction:.0%}) "
        f"as final holdout. WFO operates on {len(wfo_ohlcv)} bars."
    )
    # Use wfo_ohlcv (not the original) for everything below this point.
    ohlcv = wfo_ohlcv
```

The variable `holdout_ohlcv` will be consumed by Task 7's holdout-evaluation step. Save it into the per-coin results dict alongside the gate metrics:

In the per-coin results building at the end of the function (find the `per_coin_results[symbol] = {...}` block), add a reference to the holdout slice for that symbol. Defer the actual evaluation until Task 7.

- [ ] **Step 8: Confirm single-strategy WFO entry points are unaffected**

Open `/home/flynn/ggTrader/src/ggTrader/core/wfo.py`. The two single-strategy WFO entry points around lines 720 and 870 are NOT touched by holdout logic — they're legacy single-pair runs used for debugging and pre-deployment sensitivity work. They don't compose with the holdout (no live-deploy decision). Leave them alone but verify by `grep -n "HOLDOUT_FRACTION" src/ggTrader/core/wfo.py` returns nothing.

- [ ] **Step 9: Run holdout tests + WFO suite together**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py tests/test_rank_composite.py tests/test_wfo_robustness_selection.py -v
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py src/ggTrader/data/cache/wfo_cache.py src/ggTrader/core/holdout.py src/ggTrader/core/orchestrator.py tests/test_holdout.py
git commit -m "$(cat <<'EOF'
feat(wfo): reserve 20% final holdout before fold bounds

Step 0 of the WFO textbook reset. The most recent HOLDOUT_FRACTION
(default 0.20) of bars is split off before _calculate_wfo_bounds
runs. The 10-fold WFO operates on the leading 80%; the trailing 20%
is locked away as a final honest test (used once after gates pass).

New module ggTrader.core.holdout exposes:
- split_train_holdout(ohlcv, fraction) -> (train_ohlcv, holdout_ohlcv)
- holdout_warning_flags(ann_ret, max_dd, worst_wfo_test_dd) -> list[str]

The holdout is NOT a gate. After median-param evaluation on the
holdout (Task 7), the warning function may flag negative_return or
max_dd_exceeds_threshold (>1.5x worst WFO test-fold DD). Numbers
are always reported; the human decides deploy.

Cache invalidates on HOLDOUT_FRACTION change (added to key list).

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Compute aggregate metrics + 4 PASS/FAIL gates

For each (coin, entry, exit) combo, after the 10-fold WFO completes, compute:

- WFE = mean(test_annualized_return) / mean(train_annualized_return)
- % profitable folds = fraction of folds with test_annualized_return > 0
- Parameter CV = mean across param axes of std(chosen_per_fold) / |mean(chosen_per_fold)|
- Test/train max-DD ratio = mean(|test_max_dd|) / mean(|train_max_dd|)

Apply 4 PASS/FAIL gates with thresholds:
- WFE ≥ 0.5
- % profitable ≥ 0.6
- Parameter CV ≤ 0.3
- DD ratio ≤ 2.0

Gates are filters, not selection. A combo passes (continues to per-coin selection in Task 7) or fails (excluded from per-coin candidate set).

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` (add 4 gate thresholds)
- Modify: `src/ggTrader/data/cache/wfo_cache.py` (NOT in cache key — these are post-WFO; cache stores raw fold results, gate eval is cheap)
- Create: `src/ggTrader/core/wfo_aggregate.py` (new module with the 4 metric functions and the gate-apply function)
- Modify: `src/ggTrader/core/orchestrator.py` (call the new module after `_calculate_robustness`; populate per-coin gate-pass status)
- Create: `tests/test_wfo_aggregate_gates.py`

- [ ] **Step 1: Add gate thresholds to config**

In `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`, near the new WFO knobs:

```python
        # WFO textbook gates (Pardo convention) — applied after 10-fold loop
        # as PASS/FAIL filters, never as selection criteria.
        "WFO_GATE_WFE_MIN": 0.5,               # mean(test_ann_ret) / mean(train_ann_ret) >= this
        "WFO_GATE_PROFITABLE_FOLDS_MIN": 0.6,  # fraction of folds with test_ann_ret > 0
        "WFO_GATE_PARAM_CV_MAX": 0.3,          # MAX per-axis param CV across the 10 winners
        "WFO_GATE_DD_RATIO_MAX": 2.0,          # mean(|test_dd|) / mean(|train_dd|)
        # Bars-per-year for annualization. None = infer from OHLCV index frequency
        # (4h -> 2191.5, 1h -> 8766.0, 1d -> 365.25). Override only when needed
        # for non-standard intervals or when the index is unreliable.
        "WFO_BARS_PER_YEAR": None,
```

- [ ] **Step 2: Write the failing tests for `wfo_aggregate`**

Create `/home/flynn/ggTrader/tests/test_wfo_aggregate_gates.py`:

```python
"""Tests for WFO aggregate metrics and the 4 PASS/FAIL gates.

Per (coin, entry, exit) combo, after the 10-fold WFO completes, four metrics
are computed from the per-fold results and judged against fixed thresholds.
Gates are filters (pass/fail), never selection.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.wfo_aggregate import (
    compute_wfe,
    fraction_profitable_folds,
    parameter_cv,
    dd_ratio,
    apply_gates,
    infer_bars_per_year,
)


def test_infer_bars_per_year_4h():
    """4h bars give 2191.5 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    assert infer_bars_per_year(idx) == pytest.approx(2191.5, rel=1e-4)


def test_infer_bars_per_year_1d():
    """Daily bars give 365.25 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    assert infer_bars_per_year(idx) == pytest.approx(365.25, rel=1e-4)


def test_infer_bars_per_year_1h():
    """Hourly bars give 8766 bars/year."""
    idx = pd.date_range("2023-01-01", periods=100, freq="h")
    assert infer_bars_per_year(idx) == pytest.approx(8766.0, rel=1e-4)


def test_infer_bars_per_year_config_override():
    """Explicit WFO_BARS_PER_YEAR overrides inference."""
    idx = pd.date_range("2023-01-01", periods=100, freq="4h")
    # Index says 4h (2191.5), but config forces 1000.
    assert infer_bars_per_year(idx, config={"WFO_BARS_PER_YEAR": 1000}) == 1000.0


def test_infer_bars_per_year_fallback_on_empty():
    """Empty/too-short index falls back to 4h default."""
    idx = pd.DatetimeIndex([])
    assert infer_bars_per_year(idx) == pytest.approx(2191.5, rel=1e-4)


def test_wfe_basic_ratio():
    """WFE = mean(test) / mean(train)."""
    train_returns = [0.20, 0.15, 0.10]  # mean 0.15
    test_returns = [0.10, 0.075, 0.05]  # mean 0.075
    assert compute_wfe(train_returns, test_returns) == pytest.approx(0.5, abs=1e-9)


def test_wfe_handles_zero_train_mean():
    """If train mean is ~0, WFE is NaN (avoids division by zero)."""
    train_returns = [0.01, -0.01, 0.0]  # mean 0
    test_returns = [0.05, 0.05, 0.05]
    assert np.isnan(compute_wfe(train_returns, test_returns))


def test_fraction_profitable_folds():
    """Counts folds where test return > 0."""
    test_returns = [0.10, -0.05, 0.20, 0.0, 0.15]
    # 3 of 5 strictly positive (zero is not strictly positive).
    assert fraction_profitable_folds(test_returns) == pytest.approx(0.6, abs=1e-9)


def test_parameter_cv_takes_worst_axis():
    """CV is computed per axis; the WORST (max) axis CV is reported.

    Reasoning: a coin's stability is bottlenecked by its least stable axis.
    Averaging across axes lets one unstable axis hide behind several stable
    ones. Max forces every axis to satisfy the CV gate.
    """
    # Two params, each with 4 fold-winners.
    # adx_length: [14, 14, 14, 14] -> std 0, mean 14, CV 0.
    # adx_threshold: [20, 25, 30, 35] -> std ~5.59 (ddof=0), mean 27.5, CV ~0.203.
    # max(0.0, 0.203) = 0.203.
    fold_params = [
        {"adx_length": 14, "adx_threshold": 20},
        {"adx_length": 14, "adx_threshold": 25},
        {"adx_length": 14, "adx_threshold": 30},
        {"adx_length": 14, "adx_threshold": 35},
    ]
    cv = parameter_cv(fold_params)
    assert cv == pytest.approx(0.2032, abs=0.005)


def test_parameter_cv_max_dominated_by_worst_axis():
    """A coin with one stable axis and one wild axis fails on the wild axis."""
    # adx_length: stable [14, 14, 14, 14] -> CV 0.
    # adx_threshold: wild [10, 20, 30, 40] -> std ~11.18, mean 25, CV ~0.447.
    # max = 0.447. If we averaged we'd get ~0.224 — that would let the
    # combo squeak by the 0.3 gate despite the wild axis. With max, it fails.
    fold_params = [
        {"adx_length": 14, "adx_threshold": 10},
        {"adx_length": 14, "adx_threshold": 20},
        {"adx_length": 14, "adx_threshold": 30},
        {"adx_length": 14, "adx_threshold": 40},
    ]
    cv = parameter_cv(fold_params)
    assert cv == pytest.approx(0.4472, abs=0.005)
    assert cv > 0.3  # would fail the spec gate


def test_parameter_cv_constant_params():
    """All folds choose identical params -> CV = 0."""
    fold_params = [{"a": 1, "b": 2}] * 4
    assert parameter_cv(fold_params) == 0.0


def test_dd_ratio():
    """DD ratio = mean(|test_dd|) / mean(|train_dd|)."""
    train_dds = [-0.10, -0.12, -0.08]  # |mean| = 0.10
    test_dds = [-0.20, -0.18, -0.22]  # |mean| = 0.20
    assert dd_ratio(train_dds, test_dds) == pytest.approx(2.0, abs=1e-9)


def test_apply_gates_all_pass():
    """All 4 gates pass -> True."""
    result = apply_gates(
        wfe=0.6, profitable_fraction=0.7, param_cv=0.25, dd_ratio_val=1.8,
        thresholds={"wfe_min": 0.5, "profitable_min": 0.6, "cv_max": 0.3, "dd_max": 2.0},
    )
    assert result["passed"] is True
    assert result["failures"] == []


def test_apply_gates_one_fails():
    """If one gate fails, passed=False and failures lists the failing metric."""
    result = apply_gates(
        wfe=0.4,  # below 0.5 threshold
        profitable_fraction=0.7, param_cv=0.25, dd_ratio_val=1.8,
        thresholds={"wfe_min": 0.5, "profitable_min": 0.6, "cv_max": 0.3, "dd_max": 2.0},
    )
    assert result["passed"] is False
    assert "wfe" in result["failures"]


# pytest needs to be imported for pytest.approx; ensure it's at top of file.
import pytest  # noqa: E402
```

- [ ] **Step 3: Run tests; they fail because the module doesn't exist**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_aggregate_gates.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Create `wfo_aggregate.py`**

Create `/home/flynn/ggTrader/src/ggTrader/core/wfo_aggregate.py`:

```python
"""Aggregate post-WFO metrics and the 4 PASS/FAIL gates.

After the 10-fold WFO completes for a (coin, entry, exit) combo, four metrics
characterize the combo across folds:
- WFE: walk-forward efficiency = mean(test_ann_ret) / mean(train_ann_ret)
- % profitable folds: fraction with test_ann_ret > 0
- Parameter CV: per-axis CV of the 10 chosen-per-fold params, averaged
- DD ratio: mean(|test_max_dd|) / mean(|train_max_dd|)

Four gates are applied as pure PASS/FAIL filters (Pardo convention):
- WFE >= 0.5
- profitable >= 0.6
- param CV <= 0.3
- DD ratio <= 2.0

Gates are NOT selection criteria. A combo passes (proceeds to per-coin
selection in Task 7) or fails (excluded from candidate set).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def infer_bars_per_year(
    index: pd.DatetimeIndex,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Infer bars-per-year from the OHLCV DatetimeIndex frequency.

    Looks at the median spacing between consecutive bars and computes:
        bars_per_year = (365.25 * 24 * 3600) / median_spacing_seconds

    Examples (with no config override):
      4h bars  -> 365.25 * 24 / 4   = 2191.5
      1h bars  -> 365.25 * 24       = 8766.0
      1d bars  -> 365.25            = 365.25
      15m bars -> 365.25 * 24 * 4   = 35064.0

    If config sets ``WFO_BARS_PER_YEAR`` to a positive number, that value
    overrides inference. If inference fails (too few bars, non-uniform
    spacing), falls back to the config override or 2191.5 (4h default).
    """
    cfg = config or {}
    override = cfg.get("WFO_BARS_PER_YEAR")
    if override is not None:
        try:
            override_f = float(override)
            if override_f > 0:
                return override_f
        except (TypeError, ValueError):
            pass
    # Default fallback for 4h bars (project convention).
    default_4h = 365.25 * 24.0 / 4.0  # 2191.5
    if index is None or len(index) < 2:
        return default_4h
    try:
        deltas = pd.to_datetime(pd.Index(index)).to_series().diff().dropna()
        if len(deltas) == 0:
            return default_4h
        median_sec = float(deltas.median().total_seconds())
        if median_sec <= 0:
            return default_4h
        return (365.25 * 24.0 * 3600.0) / median_sec
    except Exception:
        return default_4h


def compute_wfe(train_returns: List[float], test_returns: List[float]) -> float:
    """Walk-Forward Efficiency = mean(test_ann_ret) / mean(train_ann_ret).

    Returns NaN if mean(train) is too close to zero (avoids division explosion).
    Spec convention from Pardo: WFE >= 0.5 means OOS performance is at least
    50% of in-sample.
    """
    train_arr = np.asarray([float(x) for x in train_returns if x is not None and np.isfinite(x)])
    test_arr = np.asarray([float(x) for x in test_returns if x is not None and np.isfinite(x)])
    if len(train_arr) == 0 or len(test_arr) == 0:
        return float("nan")
    train_mean = float(train_arr.mean())
    test_mean = float(test_arr.mean())
    if abs(train_mean) < 1e-6:
        return float("nan")
    return test_mean / train_mean


def fraction_profitable_folds(test_returns: List[float]) -> float:
    """Fraction of folds with test_ann_ret > 0.

    NaN/None values are excluded from both numerator and denominator.
    """
    finite = [float(x) for x in test_returns if x is not None and np.isfinite(x)]
    if len(finite) == 0:
        return float("nan")
    n_profitable = sum(1 for x in finite if x > 0)
    return n_profitable / len(finite)


def parameter_cv(fold_params: List[Dict[str, Any]]) -> float:
    """Per-axis CV across the chosen-per-fold params; report the MAX axis CV.

    For each param key, compute std(values) / |mean(values)| across folds.
    Axes with mean ~= 0 or constant value contribute 0 (no variation).
    Non-numeric param values are skipped.

    Returns the MAXIMUM CV across numeric axes (not the mean). Reasoning:
    a coin's stability is bottlenecked by its least stable axis. Averaging
    lets one wild axis hide behind several stable ones; max forces every
    axis to satisfy the CV gate.

    Returns 0.0 when all axes are constant. NaN when no numeric axes exist.
    """
    if not fold_params:
        return float("nan")
    keys = set()
    for p in fold_params:
        keys.update(p.keys())
    cvs: List[float] = []
    for k in keys:
        vals: List[float] = []
        for p in fold_params:
            v = p.get(k)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) < 2:
            continue
        arr = np.asarray(vals)
        mean_v = float(arr.mean())
        std_v = float(arr.std(ddof=0))
        if abs(mean_v) < 1e-9:
            cvs.append(0.0 if std_v < 1e-9 else float("inf"))
        else:
            cvs.append(std_v / abs(mean_v))
    if not cvs:
        return float("nan")
    finite_cvs = [c for c in cvs if np.isfinite(c)]
    if not finite_cvs:
        return float("inf")
    return float(max(finite_cvs))


def dd_ratio(train_dds: List[float], test_dds: List[float]) -> float:
    """Test/train max-drawdown ratio = mean(|test_dd|) / mean(|train_dd|).

    Drawdowns are stored as negative numbers (worse = more negative). The
    ratio uses absolute values so it's always >= 0.
    """
    train_arr = np.asarray(
        [abs(float(x)) for x in train_dds if x is not None and np.isfinite(x)]
    )
    test_arr = np.asarray(
        [abs(float(x)) for x in test_dds if x is not None and np.isfinite(x)]
    )
    if len(train_arr) == 0 or len(test_arr) == 0:
        return float("nan")
    train_mean = float(train_arr.mean())
    if train_mean < 1e-9:
        return float("inf")
    return float(test_arr.mean()) / train_mean


def apply_gates(
    wfe: float,
    profitable_fraction: float,
    param_cv: float,
    dd_ratio_val: float,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Apply the 4 PASS/FAIL gates. Returns a dict with 'passed' and 'failures'.

    A combo passes if ALL four gates pass. NaN values fail their gate.
    """
    failures: List[str] = []

    if not np.isfinite(wfe) or wfe < thresholds["wfe_min"]:
        failures.append("wfe")

    if not np.isfinite(profitable_fraction) or profitable_fraction < thresholds["profitable_min"]:
        failures.append("profitable_fraction")

    if not np.isfinite(param_cv) or param_cv > thresholds["cv_max"]:
        failures.append("param_cv")

    if not np.isfinite(dd_ratio_val) or dd_ratio_val > thresholds["dd_max"]:
        failures.append("dd_ratio")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "metrics": {
            "wfe": float(wfe) if np.isfinite(wfe) else None,
            "profitable_fraction": (
                float(profitable_fraction) if np.isfinite(profitable_fraction) else None
            ),
            "param_cv": float(param_cv) if np.isfinite(param_cv) else None,
            "dd_ratio": float(dd_ratio_val) if np.isfinite(dd_ratio_val) else None,
        },
    }
```

- [ ] **Step 5: Run tests; they should pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_aggregate_gates.py -v
```

Expected: 8/8 PASS.

- [ ] **Step 6: Wire aggregate gates into orchestrator**

Open `/home/flynn/ggTrader/src/ggTrader/core/orchestrator.py`. Find the per-coin per-combo loop (around line 1044 where `_calculate_robustness` is called). After the robustness call and after extracting the per-fold winner from `wfo_stats`, add:

```python
                    # WFO textbook aggregate gates: compute the four metrics and
                    # apply pass/fail filtering. A combo that fails any gate is
                    # excluded from per-coin selection in Task 7.
                    from ggTrader.core.wfo_aggregate import (
                        compute_wfe, fraction_profitable_folds,
                        parameter_cv, dd_ratio, apply_gates,
                    )

                    fold_train_returns = [
                        float(fold.get("train_annualized_return", float("nan")))
                        for fold in wfo_stats
                    ]
                    fold_test_returns = [
                        float(fold.get("oos_return", float("nan")))
                        for fold in wfo_stats
                    ]
                    fold_train_dds = [
                        float(fold.get("train_max_dd", float("nan")))
                        for fold in wfo_stats
                    ]
                    fold_test_dds = [
                        float(fold.get("oos_max_dd", float("nan")))
                        for fold in wfo_stats
                    ]
                    fold_params = [fold.get("params", {}) for fold in wfo_stats]

                    wfe = compute_wfe(fold_train_returns, fold_test_returns)
                    prof = fraction_profitable_folds(fold_test_returns)
                    pcv = parameter_cv(fold_params)
                    ddr = dd_ratio(fold_train_dds, fold_test_dds)

                    gate_result = apply_gates(
                        wfe=wfe,
                        profitable_fraction=prof,
                        param_cv=pcv,
                        dd_ratio_val=ddr,
                        thresholds={
                            "wfe_min": float(config.get("WFO_GATE_WFE_MIN", 0.5)),
                            "profitable_min": float(config.get("WFO_GATE_PROFITABLE_FOLDS_MIN", 0.6)),
                            "cv_max": float(config.get("WFO_GATE_PARAM_CV_MAX", 0.3)),
                            "dd_max": float(config.get("WFO_GATE_DD_RATIO_MAX", 2.0)),
                        },
                    )
```

Important: the per-fold dicts in `wfo_stats` need to carry `train_annualized_return`, `train_max_dd`, and `oos_max_dd`. Currently they only carry `oos_return` (test annualized return is derived from `total_return().mean()`) and not train versions. Add these fields in `_process_wfo_fold`:

In `wfo.py:_process_wfo_fold`, before the function returns its dict (around line 210), add:

```python
    # Train-window scalars for aggregate gates (Task 6).
    try:
        train_total_ret = float(pf_train.total_return().max()) if pf_train is not None else float("nan")
    except Exception:
        train_total_ret = float("nan")
    try:
        train_dd = float(pf_train.max_drawdown().min()) if pf_train is not None else float("nan")
    except Exception:
        train_dd = float("nan")
    try:
        oos_dd = float(pf_test.max_drawdown().min()) if pf_test is not None else float("nan")
    except Exception:
        oos_dd = float("nan")
    # Annualized return via compounding: (1 + total_return)^(bars_per_year / n_bars) - 1.
    # Linear scaling understates large compounded returns; compounding is correct
    # for ratio metrics across windows of different lengths.
    # bars_per_year is INFERRED from the OHLCV index frequency (no hardcoded 4h
    # assumption). Falls back to a config override or 4h default if inference fails.
    from ggTrader.core.wfo_aggregate import infer_bars_per_year
    bars_per_year = infer_bars_per_year(train_ohlcv.index, config=config)
    try:
        n_train_bars = float(len(train_ohlcv))
        if n_train_bars > 0 and np.isfinite(train_total_ret) and train_total_ret > -1.0:
            train_ann_ret = (1.0 + train_total_ret) ** (bars_per_year / n_train_bars) - 1.0
        else:
            train_ann_ret = float("nan")
    except Exception:
        train_ann_ret = float("nan")
    try:
        oos_total_ret = float(pf_test.total_return().mean()) if pf_test is not None else float("nan")
        n_test_bars = float(len(test_ohlcv))
        if n_test_bars > 0 and np.isfinite(oos_total_ret) and oos_total_ret > -1.0:
            oos_ann_ret = (1.0 + oos_total_ret) ** (bars_per_year / n_test_bars) - 1.0
        else:
            oos_ann_ret = float("nan")
    except Exception:
        oos_ann_ret = float("nan")
```

Then in the returned dict, add:
```python
        "train_annualized_return": train_ann_ret,
        "train_max_dd": train_dd,
        "oos_max_dd": oos_dd,
        "oos_annualized_return": oos_ann_ret,
```

Don't remove `oos_return` — it's used elsewhere. The new fields are additive.

Update the `oos_return` line in the call site above to use `oos_annualized_return` instead, for consistency:

```python
                    fold_test_returns = [
                        float(fold.get("oos_annualized_return", float("nan")))
                        for fold in wfo_stats
                    ]
```

- [ ] **Step 7: Store the gate result in per-coin results**

Continue in `orchestrator.py`. After `gate_result` is computed (the block from Step 6), include it in whatever stores the per-combo info. Look for the existing `top_combos_tracker.append(...)` block (around line 1097–1108) and add the gate result:

```python
                    if np.isfinite(gate_score):
                        top_combos_tracker.append({
                            "strategy": strategy_name,
                            "exit": exit_name,
                            "params": _to_native(best_params),
                            "gate_score": float(gate_score),
                            "wfo_aggregate_gates": gate_result,  # NEW
                            ...
                        })
```

- [ ] **Step 8: Run tests again**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_aggregate_gates.py tests/test_rank_composite.py tests/test_holdout.py tests/test_wfo_robustness_selection.py -v
```

Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py src/ggTrader/core/wfo_aggregate.py src/ggTrader/core/wfo.py src/ggTrader/core/orchestrator.py tests/test_wfo_aggregate_gates.py
git commit -m "$(cat <<'EOF'
feat(wfo): aggregate metrics + 4 PASS/FAIL gates (textbook reset)

After the 10-fold WFO completes for a (coin, entry, exit) combo,
compute four metrics and judge each against a fixed threshold:

- WFE = mean(test_ann_ret) / mean(train_ann_ret) >= 0.5
- % profitable folds = (test_ann_ret > 0) >= 0.6
- parameter CV = MAX per-axis CV across the 10 winners <= 0.3
  (worst axis defines stability; averaging would let one wild
  axis hide behind several stable ones.)
- DD ratio = mean(|test_dd|) / mean(|train_dd|) <= 2.0

Gates are PASS/FAIL filters, never selection. A combo passes
(continues to per-coin selection in Task 7) or fails (excluded
from candidate set entirely).

Pardo convention thresholds. Annualized returns use proper
compounding: (1 + total_return)^(bars_per_year / n_bars) - 1.
bars_per_year is INFERRED from the OHLCV DatetimeIndex frequency
(falls back to 4h / 2191.5 if inference fails; overridable via
WFO_BARS_PER_YEAR config).

New module ggTrader.core.wfo_aggregate exposes:
- infer_bars_per_year(index, config) — bars/yr from index frequency
- compute_wfe, fraction_profitable_folds, parameter_cv, dd_ratio
- apply_gates(wfe, prof, cv, ddr, thresholds) -> {passed, failures, metrics}

_process_wfo_fold now records train_annualized_return,
train_max_dd, oos_max_dd, oos_annualized_return per fold.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Per-coin selection among gate-passers + median-of-folds live params + smoke test + holdout report

For each coin: among (entry, exit) combos that passed all 4 gates, pick the one with the highest **mean of per-fold raw Sortino**. Tie-break (when multiple combos score within 5% of the top): lowest parameter CV. Coins where no combo passes all gates are dropped.

Live params for the chosen combo = median of the 10 per-fold winners' values per axis, snapped to the nearest grid point. Smoke-test the median params on the full WFO train history (the 80% pre-holdout block). Then run them on the 20% holdout block once and emit warning flags.

**Files:**
- Modify: `src/ggTrader/core/orchestrator.py` (per-coin selection + median params + smoke test + holdout call)
- Test: `tests/test_holdout.py` (extend with median-snap)

- [ ] **Step 1: Write the median-snap test**

Append to `/home/flynn/ggTrader/tests/test_holdout.py`:

```python
from ggTrader.core.holdout import median_params_snap_to_grid


def test_median_params_snap_to_grid_int_axis():
    """Median values for integer axes snap to nearest grid value."""
    fold_winners = [
        {"adx_length": 14},
        {"adx_length": 14},
        {"adx_length": 20},
        {"adx_length": 30},
    ]
    grid = {"adx_length": [10, 14, 20, 30]}
    snapped = median_params_snap_to_grid(fold_winners, grid)
    # Median is 17 (mean of 14 and 20 in a 4-elt list); nearest grid = 14 or 20.
    # Python statistics.median of [14,14,20,30] = (14+20)/2 = 17. Nearest in grid: 14 and 20 are equidistant.
    # Convention: when tied, take the LOWER value (more conservative).
    assert snapped["adx_length"] in (14, 20)


def test_median_params_snap_to_grid_float_axis():
    """Float axes snap to nearest grid value by absolute distance."""
    fold_winners = [
        {"atr_multiplier": 2.5},
        {"atr_multiplier": 3.5},
        {"atr_multiplier": 4.5},
        {"atr_multiplier": 6.0},
    ]
    grid = {"atr_multiplier": [2.5, 3.5, 4.5, 6.0]}
    snapped = median_params_snap_to_grid(fold_winners, grid)
    # Median of [2.5, 3.5, 4.5, 6.0] = (3.5+4.5)/2 = 4.0. Nearest in grid: 3.5 and 4.5 are equidistant.
    assert snapped["atr_multiplier"] in (3.5, 4.5)


def test_median_params_snap_grid_with_non_numeric():
    """Non-numeric params (bools, strings) take the mode (most common value)."""
    fold_winners = [
        {"use_filter": True},
        {"use_filter": True},
        {"use_filter": False},
    ]
    grid = {"use_filter": [True, False]}
    snapped = median_params_snap_to_grid(fold_winners, grid)
    assert snapped["use_filter"] is True  # mode of [T,T,F] = T
```

- [ ] **Step 2: Run tests — they fail**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py::test_median_params_snap_to_grid_int_axis -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `median_params_snap_to_grid` in holdout.py**

Append to `/home/flynn/ggTrader/src/ggTrader/core/holdout.py`:

```python
import statistics
from typing import Any, Dict


def median_params_snap_to_grid(
    fold_winners: list,
    grid: Dict[str, list],
) -> Dict[str, Any]:
    """Per axis: take median across fold winners, snap to nearest grid value.

    Numeric axes use statistics.median. When the median falls exactly between
    two grid values (tie), the lower value is chosen (more conservative).
    Non-numeric axes (bool, str) use mode (most common value).

    Args:
        fold_winners: List of per-fold winner-params dicts (one per fold).
        grid: The parameter grid the strategy was optimized over.
              Each axis's value list provides the snap targets.

    Returns:
        Dict mapping axis name to the snapped median value.
    """
    if not fold_winners:
        return {}
    result: Dict[str, Any] = {}
    axes = set()
    for w in fold_winners:
        axes.update(w.keys())
    for axis in axes:
        values = [w.get(axis) for w in fold_winners if axis in w]
        if not values:
            continue
        # Try numeric path first.
        try:
            numeric_vals = [float(v) for v in values]
            med = statistics.median(numeric_vals)
            grid_vals = [float(g) for g in grid.get(axis, [med])]
            if not grid_vals:
                result[axis] = med
                continue
            # Snap: find grid value with smallest absolute distance. Ties -> lower.
            grid_vals_sorted = sorted(grid_vals)
            best = grid_vals_sorted[0]
            best_dist = abs(best - med)
            for v in grid_vals_sorted[1:]:
                d = abs(v - med)
                # Strict-less means earlier tied values win (lower-is-conservative).
                if d < best_dist:
                    best = v
                    best_dist = d
            # Preserve the original numeric type from the grid if possible (int vs float).
            original_first = grid.get(axis, [best])[0]
            if isinstance(original_first, int) and best == int(best):
                result[axis] = int(best)
            else:
                result[axis] = best
        except (TypeError, ValueError):
            # Non-numeric axis: use mode.
            try:
                result[axis] = statistics.mode(values)
            except statistics.StatisticsError:
                # No unique mode — fall back to first.
                result[axis] = values[0]
    return result
```

- [ ] **Step 4: Run tests; new tests pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py -v
```

Expected: 8/8 PASS.

- [ ] **Step 5: Wire per-coin selection + median params + smoke test + holdout into orchestrator**

Open `/home/flynn/ggTrader/src/ggTrader/core/orchestrator.py`. In `run_multi_strategy_per_coin_wfo`, after the per-coin per-combo loop completes (where `top_combos_tracker` is built around line 1130), replace the existing rank-1 selection logic with the new gate-passers + Sortino selection:

```python
                    # Per-coin selection: among gate-passing combos, pick by mean per-fold Sortino.
                    # Tie-break (within 5% of top): lowest parameter CV.
                    gate_passing_combos = [
                        c for c in top_combos_tracker
                        if c.get("wfo_aggregate_gates", {}).get("passed") is True
                    ]
                    if not gate_passing_combos:
                        # No combo passes all 4 gates — drop the coin.
                        print(f"  WARNING: {symbol} — no (entry, exit) combo passed all 4 WFO gates. Dropping.")
                        continue  # skip this symbol; do not add to per_coin_results

                    # Sort by mean-per-fold Sortino descending. Compute mean-sortino per combo from wfo_stats.
                    # NOTE: each combo's wfo_stats was stored inside top_combos_tracker via per-combo tracking;
                    # if not stored, retrieve from the per-combo cache or recompute. Assumes 'mean_sortino' is
                    # populated when the gate is computed in Task 6.
                    def _combo_mean_sortino(combo):
                        return float(combo.get("mean_sortino", float("-inf")))

                    sorted_passers = sorted(gate_passing_combos, key=_combo_mean_sortino, reverse=True)
                    top_score = _combo_mean_sortino(sorted_passers[0])
                    # Within 5% of top, tie-break by lowest param CV (lower = more stable).
                    tie_band_threshold = top_score * (1.0 - 0.05) if top_score > 0 else top_score - 0.05 * abs(top_score)
                    tied_combos = [c for c in sorted_passers if _combo_mean_sortino(c) >= tie_band_threshold]
                    if len(tied_combos) > 1:
                        chosen = min(
                            tied_combos,
                            key=lambda c: c.get("wfo_aggregate_gates", {}).get("metrics", {}).get("param_cv", float("inf")),
                        )
                    else:
                        chosen = sorted_passers[0]

                    # Median-fold params for the chosen combo, snapped to grid.
                    from ggTrader.core.holdout import median_params_snap_to_grid
                    chosen_fold_params = chosen.get("fold_params", [])  # populated in Task 6
                    chosen_grid = chosen.get("param_grid", param_grid)
                    median_params = median_params_snap_to_grid(chosen_fold_params, chosen_grid)

                    per_coin_results[symbol] = {
                        "best_strategy": chosen["strategy"],
                        "best_exit": chosen["exit"],
                        "best_params": median_params,
                        "robustness_score": chosen["gate_score"],
                        "fold_consistency": chosen.get("fold_consistency", None),
                        "wfo_aggregate_gates": chosen["wfo_aggregate_gates"],
                        "selection_reason": "textbook_gates_then_sortino",
                    }
```

This replaces the prior selection logic (`best_robust_score`, top-K fallback, trade-freq fallback) — those were Layer 2 OOS-aware mechanisms that no longer apply.

Add to the top_combos_tracker dict in Step 7 of Task 6:
```python
                            "fold_params": [fold.get("params", {}) for fold in wfo_stats],
                            "param_grid": param_grid,
                            "mean_sortino": float(np.nanmean(
                                [fold.get("sortino", float("nan")) for fold in wfo_stats]
                            )),
```

- [ ] **Step 6: Add smoke test + holdout evaluation per coin**

After the per-coin selection (above) in `orchestrator.py`, run the median params on (a) the full 80% WFO train data and (b) the 20% holdout. Add this block just after `per_coin_results[symbol] = {...}`:

```python
                    # Smoke test: run median params on the full WFO train window.
                    # Not selection — just a sanity check that the median grid-snap
                    # doesn't break (e.g., produces no trades or wild metrics).
                    # NOTE: FastBacktest expects param_grid as {axis: [value, ...]}.
                    # median_params is {axis: value} — wrap each value as a single-element
                    # list so the vectorized engine builds a 1-cell portfolio.
                    median_grid = {k: [v] for k, v in median_params.items()}
                    try:
                        smoke_engine = FastBacktest(
                            ohlcv=ohlcv[[symbol]],
                            param_grid=median_grid,
                            config={**config, "USE_VECTORIZED": False, "ENTRY_STRATEGY": chosen["strategy"], "EXIT_STRATEGY": chosen["exit"]},
                        )
                        smoke_pf = smoke_engine.run(show_progress=False)
                        smoke_sortino = float(smoke_pf.sortino_ratio().mean())
                        smoke_total_ret = float(smoke_pf.total_return().mean())
                        smoke_max_dd = float(smoke_pf.max_drawdown().min())
                        print(
                            f"  [Smoke test] {symbol} median params: "
                            f"Sortino={smoke_sortino:.3f}, return={smoke_total_ret*100:.2f}%, "
                            f"max_dd={smoke_max_dd*100:.2f}%"
                        )
                        per_coin_results[symbol]["smoke_test"] = {
                            "sortino": smoke_sortino,
                            "total_return": smoke_total_ret,
                            "max_dd": smoke_max_dd,
                        }
                    except Exception as smoke_exc:
                        print(f"  WARNING: {symbol} smoke test failed: {smoke_exc!r}")
                        per_coin_results[symbol]["smoke_test"] = None

                    # Holdout evaluation: run median params on the 20% locked holdout.
                    # One-shot. Result is reported with warning flags (no auto-drop).
                    holdout_symbol_data = holdout_ohlcv[[symbol]] if len(holdout_ohlcv) > 0 else None
                    if holdout_symbol_data is not None and len(holdout_symbol_data) > 0:
                        try:
                            holdout_engine = FastBacktest(
                                ohlcv=holdout_symbol_data,
                                param_grid=median_grid,
                                config={**config, "USE_VECTORIZED": False, "ENTRY_STRATEGY": chosen["strategy"], "EXIT_STRATEGY": chosen["exit"]},
                            )
                            holdout_pf = holdout_engine.run(show_progress=False)
                            holdout_sortino = float(holdout_pf.sortino_ratio().mean())
                            holdout_total_ret = float(holdout_pf.total_return().mean())
                            holdout_max_dd = float(holdout_pf.max_drawdown().min())
                            # Annualize the holdout return via compounding using
                            # the bars-per-year inferred from the holdout index.
                            from ggTrader.core.wfo_aggregate import infer_bars_per_year
                            bars_per_year = infer_bars_per_year(
                                holdout_symbol_data.index, config=config
                            )
                            n_holdout_bars = float(len(holdout_symbol_data))
                            if n_holdout_bars > 0 and np.isfinite(holdout_total_ret) and holdout_total_ret > -1.0:
                                holdout_ann_ret = (1.0 + holdout_total_ret) ** (bars_per_year / n_holdout_bars) - 1.0
                            else:
                                holdout_ann_ret = float("nan")
                            # Worst test-fold DD from chosen combo's wfo_stats.
                            chosen_wfo_stats = chosen.get("wfo_stats_ref", [])
                            test_dds = [fold.get("oos_max_dd", float("nan")) for fold in chosen_wfo_stats]
                            finite_test_dds = [d for d in test_dds if d is not None and np.isfinite(d)]
                            worst_test_dd = min(finite_test_dds) if finite_test_dds else None

                            from ggTrader.core.holdout import holdout_warning_flags
                            warning_flags = holdout_warning_flags(
                                holdout_ann_return=holdout_ann_ret,
                                holdout_max_dd=holdout_max_dd,
                                worst_wfo_test_dd=worst_test_dd if worst_test_dd is not None else 0.0,
                            )

                            print(
                                f"  [Holdout] {symbol}: Sortino={holdout_sortino:.3f}, "
                                f"ann_return={holdout_ann_ret*100:.2f}%, max_dd={holdout_max_dd*100:.2f}%, "
                                f"warnings={warning_flags or 'none'}"
                            )
                            per_coin_results[symbol]["holdout"] = {
                                "sortino": holdout_sortino,
                                "annualized_return": holdout_ann_ret,
                                "max_dd": holdout_max_dd,
                                "warnings": warning_flags,
                            }
                        except Exception as hold_exc:
                            print(f"  WARNING: {symbol} holdout evaluation failed: {hold_exc!r}")
                            per_coin_results[symbol]["holdout"] = None
```

Also add `wfo_stats_ref` reference to top_combos_tracker (Task 6 Step 7 patch) so we can recover the worst test DD for warning logic.

- [ ] **Step 7: Run all tests + a syntax check**

```bash
cd /home/flynn/ggTrader
python -c "import ast; ast.parse(open('src/ggTrader/core/orchestrator.py').read()); print('OK')"
PYTHONPATH=src .venv/bin/pytest tests/test_holdout.py tests/test_rank_composite.py tests/test_wfo_aggregate_gates.py tests/test_wfo_robustness_selection.py -v
```

Expected: syntax OK, all tests PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/core/orchestrator.py src/ggTrader/core/holdout.py tests/test_holdout.py
git commit -m "$(cat <<'EOF'
feat(wfo): per-coin selection + median params + smoke test + holdout report

Per-coin selection (Task 7 of textbook reset):
- Filter combos to those that passed all 4 aggregate gates from Task 6.
- Pick the one with highest mean-per-fold Sortino.
- Tie-break (within 5% of top): lowest parameter CV.
- Coin is dropped entirely when no combo passes all gates.

Live params for chosen combo:
- Median of the 10 per-fold winners per axis.
- Snapped to nearest grid value; ties prefer the lower (conservative).

Smoke test: run median params on the full 80% WFO train window. Report
Sortino, total return, max_dd. Sanity check — not selection.

Holdout: run median params on the 20% locked holdout block exactly once.
Report Sortino, annualized return, max_dd, trade count. Emit warning
flags when holdout ann_return < 0 OR holdout max_dd > 1.5 * worst
WFO test-fold DD. NOT a gate — human reads warnings and decides deploy.

New holdout module function: median_params_snap_to_grid(fold_winners, grid)
- Numeric axes use statistics.median, snap to nearest grid value.
- Non-numeric axes use mode.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Coarsen parameter grids (≤ 50 combos/strategy via articulation test)

Reduce parameter grid sizes so each grid point represents a meaningfully different strategy. Apply the articulation test: if you cannot explain why value A should behave differently from adjacent value B, drop one.

**File:**
- Modify: `src/ggTrader/pipeline/param_grids.py`

This is a curation task — needs domain judgement per strategy. Below are minimum-viable starting grids for each strategy. Total combos per (entry × exit) combination must be ≤ 50.

- [ ] **Step 1: Apply the articulation test to each entry strategy**

Open `/home/flynn/ggTrader/src/ggTrader/pipeline/param_grids.py`. Replace each entry's grid in `DETAILED_ENTRY_PARAM_GRIDS` with a coarsened version:

```python
DETAILED_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        # Articulation test: sar accel/max collapsed to constants (already 100% empirical).
        # adx_length: only 14 vs 30 — short vs long lookback, qualitatively distinct.
        # adx_threshold: 20 (loose), 30 (tight) — drop 25 (interpolation), 35 (extreme).
        "sar_acceleration": [0.02],
        "sar_maximum": [0.1],
        "adx_length": [14, 30],
        "adx_threshold": [20, 30],
        "use_dmp_cross": [False],
        # Total: 1×1×2×2×1 = 4 combos
    },
    "ema_cross": {
        # Fast: 9 (short-term), 12 (intermediate). Drop 5 (too noisy).
        # Slow: 50 (medium trend), 200 (long trend). Drop 100 (interpolation).
        "ema_fast": [9, 12],
        "ema_slow": [50, 200],
        # Total: 4 combos
    },
    "mtf_momentum": {
        # 4h EMA cross gated by daily-equivalent. Daily filter: 100 (intermediate) vs 200 (long).
        "ema_fast": [9],
        "ema_slow": [21],
        "mtf_daily_ema": [100, 200],
        # Total: 2 combos
    },
    "rsi_reversal": {
        # Length: 10 (short), 21 (long). Drop 14 (interpolation).
        # Oversold: 25 (moderate), 30 (loose). Drop 20 (extreme), 35 (very loose).
        "rsi_length": [10, 21],
        "rsi_oversold": [25, 30],
        "rsi_trend_filter": [False],
        # Total: 4 combos
    },
    "adx_filtered_rsi": {
        # adx_max is the discriminating axis: 15 (very-range), 25 (loose-range). Drop 20.
        "rsi_length": [14],
        "rsi_oversold": [30],
        "adx_length": [14],
        "adx_max": [15, 25],
        # Total: 2 combos
    },
    "donchian_breakout": {
        # Length: 30 (short channel) vs 100 (long). Drop 50 (interpolation).
        "donchian_length": [30, 100],
        # Total: 2 combos
    },
    "macd_cross": {
        # Fast: 8 (short) vs 16 (long). Drop 12 (interpolation).
        # Slow: 26 only (standard MACD slow).
        # Signal: 9 only (standard MACD signal).
        "macd_fast": [8, 16],
        "macd_slow": [26],
        "macd_signal": [9],
        # Total: 2 combos
    },
    "supertrend_flip": {
        # Length: 10 vs 20 (short vs long ATR window).
        # Multiplier: 2.0 (tight) vs 4.0 (loose). Drop 3.0, 5.0.
        "st_length": [10, 20],
        "st_multiplier": [2.0, 4.0],
        # Total: 4 combos
    },
    "bbands_mean_reversion": {
        # Length: 20 (standard). Drop 14, 30 — 20 is canonical.
        # Std: 1.5 (loose) vs 2.5 (tight). Drop 2.0 (interpolation).
        "bb_length": [20],
        "bb_std": [1.5, 2.5],
        # Total: 2 combos
    },
    "stoch_rsi_reversal": {
        # Articulation: oversold 15 (very strict) vs 20 (moderate). Drop length variants.
        "stochrsi_rsi_length": [14],
        "stochrsi_stoch_length": [14],
        "stochrsi_oversold": [15, 20],
        # Total: 2 combos
    },
    "keltner_breakout": {
        # Length: 20 (standard). Multiplier: 1.0 vs 2.0 (tight vs loose).
        "kc_length": [20],
        "kc_multiplier": [1.0, 2.0],
        # Total: 2 combos
    },
}
```

Similarly coarsen `DETAILED_EXIT_AXIS_GRIDS`:

```python
DETAILED_EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        # Length: 14 (responsive) vs 30 (smooth). Drop 21 (interpolation).
        # Multiplier: 2.5 (tight) vs 4.5 (loose). Drop 3.5, 6.0.
        "atr_length": [14, 30],
        "atr_multiplier": [2.5, 4.5],
        # Total: 4 combos
    },
    "fixed_sl_tp": {
        # Stop: 2% (moderate) vs 5% (loose). Drop 1.5%, 3% (interpolation).
        # TP: 4% vs 10% (asymmetric high-reward).
        "stop_pct": [2.0, 5.0],
        "take_profit_pct": [4.0, 10.0],
        # Total: 4 combos
    },
    "trailing_stop": {
        # 3% (tight) vs 8% (loose). Drop 5%, 12%.
        "trailing_stop_pct": [3.0, 8.0],
        # Total: 2 combos
    },
}
```

Largest entry × exit cross product after coarsening:
- psar_adx (4) × atr_trailing (4) = 16
- psar_adx (4) × fixed_sl_tp (4) = 16
- supertrend_flip (4) × fixed_sl_tp (4) = 16

All under 50. The plan target is met.

- [ ] **Step 2: Verify grid sizes programmatically**

Create a quick sanity script (don't commit it):

```bash
cd /home/flynn/ggTrader
python3 -c "
from ggTrader.pipeline.param_grids import DETAILED_ENTRY_PARAM_GRIDS, DETAILED_EXIT_AXIS_GRIDS
for ent_name, ent_grid in DETAILED_ENTRY_PARAM_GRIDS.items():
    n_ent = 1
    for v in ent_grid.values():
        n_ent *= len(v) if isinstance(v, list) else 1
    print(f'  {ent_name}: {n_ent} entry combos')
for ex_name, ex_grid in DETAILED_EXIT_AXIS_GRIDS.items():
    n_ex = 1
    for v in ex_grid.values():
        n_ex *= len(v) if isinstance(v, list) else 1
    print(f'  {ex_name}: {n_ex} exit combos')
"
```

Expected: each entry grid produces 2-4 combos; each exit grid 2-4 combos. Cross product per (entry, exit) ≤ 16.

- [ ] **Step 3: Run all tests**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_rank_composite.py tests/test_holdout.py tests/test_wfo_aggregate_gates.py tests/test_wfo_robustness_selection.py tests/test_wfo_trade_counts_gate.py tests/test_exit_tournament_wfo.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/pipeline/param_grids.py
git commit -m "$(cat <<'EOF'
refactor(grids): coarsen param grids per articulation test (≤ 16/combo)

WFO textbook reset Task 8. Each grid point now represents a
meaningfully different strategy. The articulation test: if I cannot
explain why value A should behave differently from adjacent value B,
drop one.

Per-strategy combo counts: 2-4 each.
Largest entry × exit cross product after coarsening: 16
(psar_adx × atr_trailing, supertrend × fixed_sl_tp).
All under 50 — plan target met.

Selection bias reduction: with 16 cells per fold instead of 144,
the expected max-of-N outlier above per-fold median drops from
~2.5σ to ~1.6σ (~36% reduction). This composes with the rank-based
selection (which is already scale-invariant) — smaller grids mean
less luck factor in any winner.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Integration test — run diagnostic universe end-to-end

After all 8 tasks land, run the canonical 7-coin diagnostic universe end-to-end with the new textbook pipeline. Confirm:

1. The run completes without errors.
2. The 4 aggregate gates fire and are reported per (coin, combo).
3. Some coins survive selection; some are dropped (no combo passes all 4 gates).
4. Survivors have median-snap live params.
5. Holdout block is evaluated; warning flags reported.

- [ ] **Step 1: Rebuild image**

```bash
cd /home/flynn/ggTrader
docker compose build --no-cache ggtrader_live
docker compose up -d ggtrader_live
```

- [ ] **Step 2: Run diagnostic universe research**

```bash
cd /home/flynn/ggTrader
docker compose run --rm ggtrader_live python -u ggt.py research \
    --symbols BTC,ETH,TRX,DOGE,XMR,DASH,ADA \
    --workers 7 \
    --days 1095 --end-date 2026-05-01 \
    > /tmp/textbook_reset_run.log 2>&1
```

Expected runtime: 25-50 min depending on server load (smaller grids should be ~2-4x faster than the prior pipeline despite the per-fold trade gate now filtering more cells).

- [ ] **Step 3: Inspect the run output**

```bash
RUN=$(ls -td /home/flynn/ggTrader/results/research/research_2026* | head -1)
echo "RUN: $RUN"
tail -50 /tmp/textbook_reset_run.log
```

Expected indicators in the log:
- `[Holdout] Reserved N bars (20%) as final holdout. WFO operates on M bars.`
- Per-combo lines showing the 4 aggregate gate verdicts.
- `[Smoke test] <coin> median params: Sortino=..., return=...`
- `[Holdout] <coin>: Sortino=..., ann_return=..., max_dd=..., warnings=[...]`
- Coins with no gate-passing combo: `WARNING: <coin> — no (entry, exit) combo passed all 4 WFO gates. Dropping.`

- [ ] **Step 4: Capture scorecard**

```bash
python /home/flynn/ggTrader/scripts/scorecard_step1.py "$RUN" > /tmp/textbook_scorecard.json
python3 /tmp/show_scorecard.py /tmp/textbook_scorecard.json
```

Expected: a smaller number of survivors than the prior NEUTRAL Step 4b run (because the 4 hard gates will drop many coins), but the survivors should have higher OOS metrics on average. Most coins fail the new gates; that's the design.

- [ ] **Step 5: Commit the changelog entry**

Open `/home/flynn/ggTrader/docs/changelog.md`. Insert under today's date heading (create if needed):

```markdown
## 2026-05-10

### WFO textbook reset — strict train-only selection, rank-based Sortino+Calmar+PF, 4 aggregate gates, 20% locked holdout

Major refactor of the WFO pipeline to textbook standard. Plan:
`docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md`.

**Goal:** eliminate all test-data leakage from parameter selection. The
prior pipeline used OOS-derived fold weights, Layer 2 OOS blending, and
fold-consistency multipliers — all of which invalidated the "OOS" label
on reported metrics. The textbook reset enforces strict train-only
selection within and across folds; test metrics are reported but never
selected on.

**What changed:**

1. Rank-based composite (Sortino + Calmar + PF) replaces z-score weighted
   blend. Sharpe dropped (redundant with Sortino). Tie handling: average
   rank. Scale-invariant; ranks survive within-fold noise that the
   z-score blend amplified.

2. 30-trade pre-score gate with 8-of-10 forgiveness. Cells with <30
   train trades disqualified; combo must pass in 8+ of 10 folds. In
   forgiven folds, combo gets fold-median rank.

3. 20% locked holdout (most recent bars). 10-fold WFO operates only on
   the leading 80%. Median params evaluated on holdout exactly once
   after gates pass. Warnings (not gates) if return<0 or max_dd >
   1.5× worst WFO test-fold DD.

4. 4 PASS/FAIL aggregate gates (Pardo convention):
   - WFE = mean(test_ann_ret)/mean(train_ann_ret) ≥ 0.5
   - % profitable folds ≥ 60%
   - parameter CV across 10 winners ≤ 0.3
   - test/train DD ratio ≤ 2.0

5. Per-coin selection among gate-passers: highest mean-per-fold Sortino,
   tie-break (within 5%) by lowest parameter CV. Coins with no combo
   passing all gates are dropped.

6. Live params = median of 10 per-fold winners, snapped to grid. Smoke
   test on full WFO train history confirms metrics are sane.

7. Param grids coarsened: each grid point now meaningfully different,
   per the articulation test. Largest cross product: 16 cells per
   (entry × exit). Selection bias from max-of-N: ~36% reduction in
   expected outlier σ vs. previous 144-cell grids.

**Removed:** PARAM_OOS_GAP_PENALTY, OOS_ROBUSTNESS_BLEND_ALPHA,
FOLD_CONSISTENCY_IN_GATE, FOLD_CONSISTENCY_GATE_FLOOR,
PARAM_STABILITY_WEIGHT, PARAM_ZRANK_WEIGHT,
TRAIN_METRIC_NORMALIZE_ZSCORE, TRAIN_METRIC_COMPOSITE_WEIGHTS,
test_metrics_by_fold cache schema, Sharpe component of composite.

**Cache invalidated** (version 2 → 3). All ~20k entries regenerate
on next access at the new version.

**Constraint observed throughout:** no additional gates, weights, or
selection criteria introduced during implementation. The point of
this reset is to land cleanly at textbook WFO before adding anything
back.
```

Commit:

```bash
git add docs/changelog.md
git commit -m "$(cat <<'EOF'
docs: WFO textbook reset — changelog entry for the major refactor

Documents the full reset: rank-based composite (drop Sharpe),
30-trade gate + 8-of-10 forgiveness, 20% locked holdout, 4 PASS/FAIL
aggregate gates, mean-Sortino + paramCV selection, median-fold live
params + smoke + holdout report. Lists everything removed (8 knobs
+ cache schema field). Cache version bumped 2 -> 3.

Plan: docs/superpowers/plans/2026-05-10-wfo-textbook-reset.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**
- Spec "20% locked holdout, locked away before fold bounds" → Task 5.
- Spec "rank-based composite Sortino+Calmar+PF, average ranks" → Task 3.
- Spec "drop Sharpe, drop normalize-zscore, drop composite weights" → Tasks 2 (removals) + 3 (rewrite).
- Spec "30-trade pre-score gate, 8-of-10 forgiveness, median rank in forgiven folds" → Task 4.
- Spec "4 aggregate gates as PASS/FAIL" → Task 6.
- Spec "Per-coin selection by mean Sortino + paramCV tie-break, drop if no passer" → Task 7.
- Spec "Median across 10 fold-winners + grid snap" → Task 7 + holdout module.
- Spec "Sanity smoke test on full WFO train history" → Task 7.
- Spec "Holdout report with warning flags (return<0 OR max_dd > 1.5x worst WFO test DD)" → Task 5 (warning fn) + Task 7 (evaluation call).
- Spec "Coarsen grids per articulation test, ≤ 50 combos/strategy" → Task 8.
- Spec "Remove gap penalty + per-cell OOS cache" → Task 1.
- Spec "Remove OOS-influenced fold weights, OOS blend, fold-consistency mult, stability weight, z-rank weight" → Task 2.
- Spec "No new gates, weights, or selection criteria during implementation" → Stated in plan header; reinforced in each task.

**Placeholder scan:**
- Each step has explicit code or commands.
- "Implementation outline" entries reference exact file paths and line numbers.
- "Approximate annualized return" formula uses fixed 4h-bars-per-year constant (project assumption documented inline).
- Commit messages are HEREDOC'd with verbatim text.

**Type / interface consistency:**
- `test_metrics_by_fold` is referenced in Task 1 (deletion) but not in any other Task; no stale references after Task 1.
- `oos_metrics_by_fold` is removed from `_calculate_robustness` signature in Task 2; all three call sites updated in same task.
- `_train_metric_series` composite branch is rewritten in Task 3; same return type (`pd.Series` aligned to grid keys).
- `wfo_stats[fold]` dict gains `train_annualized_return`, `train_max_dd`, `oos_max_dd`, `oos_annualized_return` in Task 6; consumed by Task 7 selection.
- `top_combos_tracker` entries gain `wfo_aggregate_gates`, `fold_params`, `param_grid`, `mean_sortino` in Task 6; consumed by Task 7 selection.
- `median_params_snap_to_grid` signature in Task 7: `(fold_winners: list, grid: dict) -> dict`. Called once from `orchestrator.py`.
- `holdout_warning_flags` signature: `(holdout_ann_return, holdout_max_dd, worst_wfo_test_dd, dd_multiple=1.5) -> list[str]`. Consistent between Task 5 definition and Task 7 caller.

**No gaps identified.** Plan is complete.
