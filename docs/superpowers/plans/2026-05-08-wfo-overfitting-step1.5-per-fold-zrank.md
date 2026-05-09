# WFO Overfitting Step 1.5: Per-fold z-rank — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-fold z-score blend to `_weighted_robustness_series` so within-combo cell ranking can flip away from raw-mean single-fold outliers toward consistent rank-position cells, and validate via small-N A/B on the diagnostic 7-coin universe.

**Architecture:** New config knob `PARAM_ZRANK_WEIGHT` (default `0.5`). Function signature extended with optional `config` kwarg. When alpha > 0, computes per-fold z-scores from the existing `(n_keys × n_folds)` matrix, weighted-means them, blends with the existing raw weighted mean. Cache key includes the new knob.

**Tech Stack:** Python (numpy, pandas), pytest, Docker Compose, TimescaleDB.

**Spec:** `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `tests/test_wfo_zrank_blend.py` | Create | New unit test for the blend behavior of `_weighted_robustness_series` |
| `src/ggTrader/core/wfo.py` | Modify `_weighted_robustness_series` (lines 235-293) and its 2 call sites in `_calculate_robustness` (lines 476, 480) | Add `config` kwarg, compute z-rank, blend |
| `src/ggTrader/utils/run_config.py` | Add new line near 208 | Define `PARAM_ZRANK_WEIGHT: 0.5` default |
| `src/ggTrader/data/cache/wfo_cache.py:23-48` | Modify list | Add `"PARAM_ZRANK_WEIGHT"` to `_WFO_RELEVANT_CONFIG_KEYS` so cache invalidates on changes |
| `docs/changelog.md` | Modify (top) | Record Step 1.5 outcome |
| `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md` | Append `## Result` section | Lock outcome on the spec |

The scorecard tool (`scripts/scorecard_step1.py`) and pick-comparison helper (`/tmp/compare_picks.py`) are reused as-is from Step 1; no new tooling.

---

## Task 1: TDD — write failing test for z-rank blend

**Files:**
- Create: `tests/test_wfo_zrank_blend.py`

- [ ] **Step 1: Write the test file**

Create `/home/flynn/ggTrader/tests/test_wfo_zrank_blend.py` with:

```python
"""Tests for the per-fold z-rank blend in _weighted_robustness_series.

Step 1.5 of the WFO overfitting work. The change adds an optional ``config``
kwarg to the function; when ``PARAM_ZRANK_WEIGHT > 0`` the function blends
the existing raw weighted-mean-of-IS with a per-fold-z-score weighted mean.
At alpha=0 the function must reproduce its prior behavior bit-for-bit.

Synthetic 3-cell × 2-fold case:
    Cell A: IS = [4.0, 0.0]  → raw mean 2.0, single-fold spike
    Cell B: IS = [1.8, 1.8]  → raw mean 1.8, consistent
    Cell C: IS = [0.5, 0.5]  → raw mean 0.5, consistent low

Under raw mean (alpha=0): A wins.
Under per-fold z-rank (alpha=1): B wins (highest mean rank position).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.wfo import _weighted_robustness_series


def _make_fixture():
    """3 cells × 2 folds. Returns (is_metrics_by_fold, weights)."""
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([4.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    fold2 = pd.Series([0.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    return {1: fold1, 2: fold2}, {1: 1.0, 2: 1.0}


def test_alpha_zero_reproduces_raw_weighted_mean():
    """With PARAM_ZRANK_WEIGHT=0 (or no config), output equals raw weighted mean."""
    is_metrics, weights = _make_fixture()
    out_no_config = _weighted_robustness_series(is_metrics, weights)
    out_alpha0 = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 0.0}
    )
    pd.testing.assert_series_equal(out_no_config, out_alpha0)
    # And matches hand-computed weighted mean.
    expected = pd.Series([2.0, 1.8, 0.5], index=pd.Index([("A",), ("B",), ("C",)], dtype=object))
    pd.testing.assert_series_equal(out_no_config.sort_index(), expected.sort_index())


def test_alpha_one_picks_consistent_cell():
    """With PARAM_ZRANK_WEIGHT=1.0, B (consistent rank) outranks A (single-fold spike)."""
    is_metrics, weights = _make_fixture()
    out = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    out_sorted = out.sort_values(ascending=False)
    assert out_sorted.index[0] == ("B",), f"expected B to win at alpha=1, got order {list(out_sorted.index)}"
    assert out_sorted.index[-1] == ("C",), "expected C last (lowest in both folds)"


def test_alpha_half_blends_smoothly():
    """At alpha=0.5 the result is between raw and z-rank, not equal to either."""
    is_metrics, weights = _make_fixture()
    raw = _weighted_robustness_series(is_metrics, weights)
    zrank = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    blend = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 0.5}
    )
    # Per-cell: blend ≈ 0.5*raw + 0.5*zrank
    expected = 0.5 * raw + 0.5 * zrank
    pd.testing.assert_series_equal(blend.sort_index(), expected.sort_index())


def test_degenerate_fold_skipped_in_zrank():
    """A fold where every cell has the same IS contributes nothing to z-rank."""
    cells = [("A",), ("B",), ("C",)]
    fold1 = pd.Series([4.0, 1.8, 0.5], index=pd.Index(cells, dtype=object))
    # Every cell has identical IS in fold 2: std==0, contributes 0 to z-rank.
    fold2 = pd.Series([1.0, 1.0, 1.0], index=pd.Index(cells, dtype=object))
    is_metrics = {1: fold1, 2: fold2}
    weights = {1: 1.0, 2: 1.0}

    zrank_only = _weighted_robustness_series(
        is_metrics, weights, config={"PARAM_ZRANK_WEIGHT": 1.0}
    )
    # With fold 2 contributing 0, the z-rank is essentially fold-1's z divided by
    # weight sum that included fold 2 — but since fold 2's z is NaN (std==0),
    # it's excluded from the denominator too. Expected: pure fold-1 z-scores.
    fold1_vals = np.array([4.0, 1.8, 0.5])
    fold1_z = (fold1_vals - fold1_vals.mean()) / fold1_vals.std()
    expected = pd.Series(fold1_z, index=pd.Index(cells, dtype=object))
    pd.testing.assert_series_equal(zrank_only.sort_index(), expected.sort_index())


def test_signature_backward_compat():
    """Existing callers without config kwarg must still work."""
    is_metrics, weights = _make_fixture()
    # Positional-only call (current convention):
    out = _weighted_robustness_series(is_metrics, weights)
    assert len(out) == 3
    assert all(np.isfinite(out.to_numpy()))
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_zrank_blend.py -v
```

Expected:
- `test_alpha_zero_reproduces_raw_weighted_mean` — likely PASS already (no config = no z-rank, function ignores extra kwargs only if it accepts them; if it errors with `TypeError: unexpected keyword argument 'config'` then it fails on that line, which is the proper TDD signal).
- `test_alpha_one_picks_consistent_cell`, `test_alpha_half_blends_smoothly`, `test_degenerate_fold_skipped_in_zrank` — FAIL with `TypeError: _weighted_robustness_series() got an unexpected keyword argument 'config'`.
- `test_signature_backward_compat` — PASS.

If all tests pass without code changes, the test is wrong; stop and re-check.

---

## Task 2: Implement the z-rank blend

**Files:**
- Modify: `src/ggTrader/core/wfo.py:235-293` (function body), `src/ggTrader/core/wfo.py:476-480` (two call sites in `_calculate_robustness`).

- [ ] **Step 1: Edit the function**

Open `/home/flynn/ggTrader/src/ggTrader/core/wfo.py`. Replace the function header at line 235 with:

```python
def _weighted_robustness_series(
    is_metrics_by_fold: Dict[int, pd.Series],
    weights: Dict[int, float],
    config: Optional[Dict[str, Any]] = None,
) -> pd.Series:
```

After the existing line `combined = np.where(den > 0.0, num / den, np.nan)` (around line 291), add the z-rank blend block before the `return pd.Series(...)`:

```python
    # Per-fold z-rank blend (Step 1.5 of WFO overfitting work).
    # alpha=0 reproduces the raw weighted-mean behavior above. alpha>0 mixes in a
    # weighted mean of per-fold z-scores so cells that rank consistently high across
    # folds beat cells that spike in one fold and average elsewhere.
    alpha = float((config or {}).get("PARAM_ZRANK_WEIGHT", 0.0))
    if alpha > 0.0:
        z_mat = np.full_like(mat, np.nan)
        for j in range(n_folds):
            col = mat[:, j]
            col_mask = np.isfinite(col)
            if col_mask.sum() < 2:
                continue  # need >= 2 cells for std
            col_finite = col[col_mask]
            col_std = col_finite.std()
            if col_std == 0.0:
                continue  # degenerate fold: all cells equal, leave z's NaN
            col_mean = col_finite.mean()
            z_mat[col_mask, j] = (col_finite - col_mean) / col_std
        z_finite = np.isfinite(z_mat)
        z_weighted_vals = np.where(z_finite, z_mat * wvec, 0.0)
        z_weighted_wts = np.where(z_finite, wvec, 0.0)
        z_den = z_weighted_wts.sum(axis=1)
        z_num = z_weighted_vals.sum(axis=1)
        zrank = np.where(z_den > 0.0, z_num / z_den, np.nan)
        # Blend; if one side is NaN at a cell, fall back to the finite side.
        both_finite = np.isfinite(combined) & np.isfinite(zrank)
        combined = np.where(
            both_finite,
            (1.0 - alpha) * combined + alpha * zrank,
            np.where(np.isfinite(combined), combined, zrank),
        )
```

- [ ] **Step 2: Update both call sites in `_calculate_robustness`**

Lines 476 and 480 currently read:

```python
        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights)
```

Pass the existing local `config` variable through. Replace **both** occurrences with:

```python
        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
```

Verify with `grep`:

```bash
grep -n "_weighted_robustness_series(" /home/flynn/ggTrader/src/ggTrader/core/wfo.py
```

Expected output (4 lines: definition + 2 call sites + 1 in a docstring):
```
235:def _weighted_robustness_series(
381:    Uses the same flattened-tuple key scheme as _weighted_robustness_series.
476:        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
480:        robustness_scores = _weighted_robustness_series(is_metrics_by_fold, weights, config=config)
```

- [ ] **Step 3: Run the new tests to confirm they pass**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_zrank_blend.py -v
```

Expected: all 5 tests PASS. If any still fail, inspect the failure message and fix the implementation rather than the test.

- [ ] **Step 4: Run the existing WFO test suite for regression**

```bash
cd /home/flynn/ggTrader
PYTHONPATH=src .venv/bin/pytest tests/test_wfo_robustness_selection.py tests/test_exit_tournament_wfo.py tests/test_wfo_trade_counts_gate.py -v
```

Expected: all PASS. The new code path is gated by `if alpha > 0.0`; with the production default still at the old behavior (alpha=0.0 if we set the default to 0.0), existing tests aren't exercising the new path. If we set the default to 0.5, existing tests exercise the blend path — they should still pass because the scoring logic isn't asserted on specific numeric values, only on selection outcomes.

If a regression appears, revert the call-site changes first to isolate whether it's a function-internal bug or a config-default problem.

---

## Task 3: Add config default and cache key entry

**Files:**
- Modify: `src/ggTrader/utils/run_config.py:208` (add new line nearby)
- Modify: `src/ggTrader/data/cache/wfo_cache.py:23-48` (add to list)

- [ ] **Step 1: Add `PARAM_ZRANK_WEIGHT` to `run_config.py`**

Open `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`. After the existing `PARAM_STABILITY_WEIGHT` block (line 208), insert (preserving the surrounding indentation and trailing commas):

```python
        # Per-fold z-rank blend weight (Step 1.5). 0.0 = pure raw weighted mean
        # (legacy behavior), 1.0 = pure mean-of-per-fold-z-scores, 0.5 = 50/50.
        # Per-fold z-rank rewards cells that rank consistently high across folds
        # over cells that spike in one fold and average elsewhere (the max-of-N
        # selection bias signature). See _weighted_robustness_series in wfo.py.
        "PARAM_ZRANK_WEIGHT": 0.5,
```

The existing `PARAM_STABILITY_WEIGHT` line is the anchor. The new line goes immediately after its existing comment block and value.

- [ ] **Step 2: Add the knob to the cache key list**

Open `/home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py`. The `_WFO_RELEVANT_CONFIG_KEYS` list spans lines 23-48. Add `"PARAM_ZRANK_WEIGHT"` immediately after the existing `"PARAM_STABILITY_WEIGHT"` entry (line 33):

```python
    "TRAIN_METRIC_NORMALIZE_ZSCORE",
    "PARAM_STABILITY_WEIGHT",
    "PARAM_ZRANK_WEIGHT",
    "START_CASH",
```

Verify:

```bash
grep -n "PARAM_ZRANK_WEIGHT" /home/flynn/ggTrader/src/ggTrader/data/cache/wfo_cache.py /home/flynn/ggTrader/src/ggTrader/utils/run_config.py
```

Expected: 2 hits, one in each file.

- [ ] **Step 3: Commit code + tests + config + cache key together**

```bash
cd /home/flynn/ggTrader
git add tests/test_wfo_zrank_blend.py src/ggTrader/core/wfo.py src/ggTrader/utils/run_config.py src/ggTrader/data/cache/wfo_cache.py
git commit -m "$(cat <<'EOF'
feat(wfo): per-fold z-rank blend in within-combo cell ranking

Step 1.5 of WFO overfitting work. Step 1 (PARAM_STABILITY_WEIGHT bump)
was a no-op because the CV penalty is a uniform multiplicative scale
that does not flip rankings. This change directly re-ranks param cells
by blending the existing fold-weighted-mean-of-raw-IS with a new
fold-weighted-mean-of-per-fold-z-scores.

Per-fold z-scoring normalizes fold difficulty so the cell that wins is
the one with the most stable relative position across folds, not the
one with the biggest single-fold spike.

New config knob PARAM_ZRANK_WEIGHT (default 0.5 = 50/50 blend; 0.0 =
legacy bit-identical behavior; 1.0 = pure z-rank). Function signature
of _weighted_robustness_series extended with optional config kwarg
(non-breaking; default None preserves all existing call patterns).
Both call sites in _calculate_robustness pass config through.

Knob added to _WFO_RELEVANT_CONFIG_KEYS so cache invalidates correctly.

Tests cover: alpha=0 bit-equivalence, alpha=1 selection flip on
synthetic 3-cell case, alpha=0.5 linear blend, degenerate-fold (std=0)
exclusion, signature backward compat.

Spec: docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rebuild image + Run A on diagnostic universe

**Files:** none modified.

- [ ] **Step 1: Rebuild image with new code baked in**

```bash
cd /home/flynn/ggTrader
docker compose build --no-cache ggtrader_live && docker compose up -d ggtrader_live
docker compose ps ggtrader_live
```

Expected: build succeeds, container shows `Up <a few seconds>`.

- [ ] **Step 2: Run research on diagnostic universe at PARAM_ZRANK_WEIGHT=0.5**

```bash
cd /home/flynn/ggTrader
docker compose run --rm ggtrader_live python -u ggt.py research \
    --symbols BTC,ETH,TRX,DOGE,XMR,DASH,ADA \
    --workers 7 \
    --days 1095 --end-date 2026-05-01 \
    > /tmp/step15_run_a.log 2>&1
```

Expected runtime: ~22 minutes (matches Step 1 micro-experiment pace; full uncached WFO since cache invalidated by new knob).

After completion, capture the run directory:

```bash
NEW_RUN=$(ls -td /home/flynn/ggTrader/results/research/research_2026* | head -1)
echo "$NEW_RUN" | tee /tmp/step15_run_a_dir.txt
```

If the run fails (any non-zero exit), inspect `tail /tmp/step15_run_a.log` and worker logs in `$NEW_RUN/worker_*.log`. The most likely cause is a math edge case the unit tests didn't cover (e.g. all-NaN in a fold). Do not proceed until clean.

- [ ] **Step 3: Capture Run A scorecard**

```bash
NEW_RUN=$(cat /tmp/step15_run_a_dir.txt)
python /home/flynn/ggTrader/scripts/scorecard_step1.py "$NEW_RUN" > /tmp/scorecard_step15_a.json
python3 /tmp/show_scorecard.py /tmp/scorecard_step15_a.json
```

Expected: prints primary, secondary, phase3 sanity, and per-coin table. Save the entire output text to `/tmp/step15_run_a_summary.txt` for the changelog later.

---

## Task 5: Switch to alpha=0.0 + Run B baseline

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` — temporarily change `PARAM_ZRANK_WEIGHT` to `0.0`.

- [ ] **Step 1: Temporarily set the knob to 0.0**

Edit `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py`:

```diff
-        "PARAM_ZRANK_WEIGHT": 0.5,
+        "PARAM_ZRANK_WEIGHT": 0.0,
```

Do NOT commit yet — this is a test-state-only edit.

- [ ] **Step 2: Rebuild and run B**

```bash
cd /home/flynn/ggTrader
docker compose build --no-cache ggtrader_live && docker compose up -d ggtrader_live

docker compose run --rm ggtrader_live python -u ggt.py research \
    --symbols BTC,ETH,TRX,DOGE,XMR,DASH,ADA \
    --workers 7 \
    --days 1095 --end-date 2026-05-01 \
    > /tmp/step15_run_b.log 2>&1

NEW_RUN=$(ls -td /home/flynn/ggTrader/results/research/research_2026* | head -1)
echo "$NEW_RUN" | tee /tmp/step15_run_b_dir.txt
```

Expected: another ~22 minutes. Cache miss again because the knob value changed.

- [ ] **Step 3: Capture Run B scorecard**

```bash
NEW_RUN=$(cat /tmp/step15_run_b_dir.txt)
python /home/flynn/ggTrader/scripts/scorecard_step1.py "$NEW_RUN" > /tmp/scorecard_step15_b.json
python3 /tmp/show_scorecard.py /tmp/scorecard_step15_b.json
```

---

## Task 6: Compare picks and apply decision rule

**Files:** none modified — analysis only.

- [ ] **Step 1: Side-by-side scorecard diff**

```bash
echo "=== Run B baseline (PARAM_ZRANK_WEIGHT=0.0) ==="
python3 -c 'import json; d=json.load(open("/tmp/scorecard_step15_b.json")); print(json.dumps({"primary":d["primary"],"secondary":d["secondary"],"phase3_sanity":d["phase3_sanity"]}, indent=2))'
echo ""
echo "=== Run A experimental (PARAM_ZRANK_WEIGHT=0.5) ==="
python3 -c 'import json; d=json.load(open("/tmp/scorecard_step15_a.json")); print(json.dumps({"primary":d["primary"],"secondary":d["secondary"],"phase3_sanity":d["phase3_sanity"]}, indent=2))'
```

- [ ] **Step 2: Pick comparison (CRITICAL — same hard-precondition as Step 1)**

Edit `/tmp/compare_picks.py` (created in Step 1) — change the two run paths to the new ones:

```python
# Replace the existing two ("A (0.7)", path) and ("B (0.3)", path) entries with:
for label, path in [
    ("A (alpha=0.5)", open("/tmp/step15_run_a_dir.txt").read().strip()),
    ("B (alpha=0.0)", open("/tmp/step15_run_b_dir.txt").read().strip()),
]:
```

Run:

```bash
python3 /tmp/compare_picks.py
```

For each surviving coin, check whether the chosen `(strategy, exit, params)` differs between Run A and Run B. **Count the coins where picks changed.** Save this count.

- [ ] **Step 3: Classify outcome**

Per spec rule:

- **NO-OP (hard precondition failed):** if `picks_changed_count == 0` across all surviving coins. Z-rank blend made no difference at alpha=0.5; classify as NO-OP regardless of scorecard movement. Move to Step 2.
- **ADVANCE:** picks_changed_count ≥ 1 AND scorecard improved on ≥ 2 of 3 primary metrics (`n_oos_gt_0_30`, `median_fold_consistency`, `n_oos_gt_0`).
- **REVERT:** picks_changed_count ≥ 1 AND scorecard worsened on ≥ 2 of 3 primary metrics. Investigate before Step 2.
- **NEUTRAL:** picks_changed_count ≥ 1 but scorecard mixed/flat. Try `alpha = 1.0` in a follow-up before moving to Step 2.
- **HARD FLOOR:** survivor count drops below 3. Treat as REVERT.

Write the classification verbatim to `/tmp/step15_decision.txt`:

```bash
echo "ADVANCE" > /tmp/step15_decision.txt   # or NO-OP / REVERT / NEUTRAL / HARD_FLOOR
```

(Manual call; choose one.)

---

## Task 7: Set the final config value + record outcome

**Files:**
- Modify: `src/ggTrader/utils/run_config.py` — set `PARAM_ZRANK_WEIGHT` to the value matching the decision.
- Modify: `docs/changelog.md` (insert near top, under today's date).
- Modify: `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md` — append `## Result` section.

- [ ] **Step 1: Set the final config value based on decision**

Edit `src/ggTrader/utils/run_config.py:PARAM_ZRANK_WEIGHT`:

| Decision | Final value | Rationale |
|---|---|---|
| ADVANCE | `0.5` | the experimental value works; keep it |
| NEUTRAL | `0.0` | doesn't move the needle; revert to legacy |
| REVERT | `0.0` | actively worse; revert to legacy |
| HARD_FLOOR | `0.0` | survivor count crashed; revert to legacy |
| NO-OP | `0.0` | bit-identical at alpha=0.5; remove the meaningless default |

For example, if NEUTRAL/REVERT/NO-OP/HARD_FLOOR:

```diff
-        "PARAM_ZRANK_WEIGHT": 0.0,
+        "PARAM_ZRANK_WEIGHT": 0.0,
```

(Already at 0.0 from Run B; no edit needed.)

If ADVANCE, restore to 0.5:

```diff
-        "PARAM_ZRANK_WEIGHT": 0.0,
+        "PARAM_ZRANK_WEIGHT": 0.5,
```

- [ ] **Step 2: Append Result section to spec**

Append to `/home/flynn/ggTrader/docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md`:

```markdown

---

## Result

Run via small-N micro-experiment on the diagnostic 7-coin universe (BTC, ETH, TRX, DOGE, XMR, DASH, ADA), same shape as Step 1's experiment.

- **Run A** (`PARAM_ZRANK_WEIGHT=0.5`): `<run_a_dir>` (~22 min)
- **Run B** (`PARAM_ZRANK_WEIGHT=0.0`, baseline): `<run_b_dir>` (~22 min)

**Pick comparison (hard precondition):** picks changed on `<picks_changed_count>` of `<n_survivors>` surviving coins. <Sentence: "Z-rank blend changed selection — proceeding to scorecard analysis." OR "Identical picks — Step 1.5 is a NO-OP, same failure mode as Step 1.">

**Primary scorecard delta:**

| Metric | Before (alpha=0.0) | After (alpha=0.5) | Δ |
|---|---:|---:|---:|
| `n_oos_gt_0_30` | <b> | <a> | <Δ> |
| `median_fold_consistency` | <b> | <a> | <Δ> |
| `n_oos_gt_0` | <b> | <a> | <Δ> |

**Secondary:** `n_survivors` <b> → <a>; `median_is_minus_oos_gap` <b> → <a>.

**Phase 3 sanity:** Total Return <a>% (vs <b>% baseline); Sharpe <a>; MaxDD <a>%.

**Decision:** <ADVANCE | REVERT | NEUTRAL | HARD_FLOOR | NO-OP>.

<2–3 sentence interpretation: which lever moved or didn't, what to do next (Step 2 grid shrink, Step 4 per-cell OOS, or another alpha value).>

**Action:** `PARAM_ZRANK_WEIGHT` set to `<final_value>` in `src/ggTrader/utils/run_config.py`.
```

Replace `<…>` placeholders with values from the Run A and Run B scorecards. Use the same format and depth as Step 1's Result section.

- [ ] **Step 3: Insert changelog entry**

Open `/home/flynn/ggTrader/docs/changelog.md`. Below `## 2026-05-08` and above the existing first `### …` (which is currently the Step 1 Result entry), insert:

```markdown
### WFO overfitting Step 1.5: per-fold z-rank blend (PARAM_ZRANK_WEIGHT) — <DECISION>

Spec: `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md`. Plan: `docs/superpowers/plans/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md`.

**Hypothesis:** blending a per-fold-z-score weighted mean into the within-combo cell ranking will flip winners away from raw-mean single-fold outliers toward consistent rank-position cells. Step 1's CV penalty failed because it scaled cells uniformly without changing relative order; per-fold z-rank directly changes the order.

**Method:** small-N micro-experiment on the same 7 diagnostic coins as Step 1. Run A at `0.5` (50/50 blend), Run B at `0.0` (legacy bit-identical behavior). Total ~50 min.

**Pick comparison:** picks changed on `<n>` of `<m>` surviving coins. <Sentence on whether the new code path made any selection difference at all.>

**B-criterion scorecard:** [same table as spec result]

**Verdict:** `<DECISION>`. <One sentence interpretation.>

**Action:** `PARAM_ZRANK_WEIGHT` set to `<final_value>`. <If ADVANCE: "Promoting to a 100-coin ratification run before declaring victory." If NO-OP/NEUTRAL: "Moving to Step 2 (grid shrink)." If REVERT: "Investigating why z-rank picked worse cells before Step 2.">
```

Replace `<…>` placeholders with concrete values.

- [ ] **Step 4: Commit final config + spec result + changelog**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py docs/changelog.md docs/superpowers/specs/2026-05-08-wfo-overfitting-step1.5-per-fold-zrank.md
git commit -m "$(cat <<'EOF'
docs: WFO overfitting Step 1.5 result + final config

Per-fold z-rank blend (PARAM_ZRANK_WEIGHT) tested via micro-experiment
on the 7-coin diagnostic universe. Decision: <DECISION>. Picks changed
on <n>/<m> surviving coins. Final value <X>.

[2-3 sentence interpretation matching the Result section.]

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Replace `<DECISION>`, `<n>`, `<m>`, `<X>` and the interpretation with concrete values.

---

## Self-review

**Spec coverage:**
- Spec "Change" (new config knob, function signature change, blend formula) → Tasks 1–3.
- Spec "Edge cases" (degenerate fold, all-NaN, missing keys) → covered by Task 1 test (`test_degenerate_fold_skipped_in_zrank`) and Task 2 implementation (the `col_mask.sum() < 2` and `col_std == 0.0` guards).
- Spec "Implementation outline" (config kwarg, both call sites, cache key) → Tasks 2 and 3.
- Spec "Test" (small-N micro-experiment, same universe, scorecard, picks comparison) → Tasks 4–6.
- Spec "Decision rule" (NO-OP precondition + ADVANCE/REVERT/NEUTRAL/HARD FLOOR) → Task 6 Step 3.
- Spec "Risks" (pure-rank insensitivity to magnitude, sparse-survivor coins) → addressed by `alpha=0.5` blend default (mitigates pure-rank insensitivity) and `col_mask.sum() < 2` guard (handles sparse-fold coins). No dedicated task; documented in spec.
- Spec "Rollback" (alpha=0.0 reproduces current behavior) → Task 7 Step 1.

**Placeholder scan:**
- The `<…>` markers in Tasks 6 (decision file) and 7 (changelog/spec result) are explicit fill-in spots for run-time values, not unfinished tasks. Acceptable per the same convention as Step 1's plan.
- No "TBD" / "TODO" / "implement later" anywhere.
- Each code-changing step contains the actual code or diff.

**Type / interface consistency:**
- Function signature change (`config: Optional[Dict[str, Any]] = None`) is consistent across the definition (Task 2 Step 1), the test calls (Task 1 Step 1), and the call sites (Task 2 Step 2).
- Config key name `PARAM_ZRANK_WEIGHT` is consistent across `run_config.py`, `wfo.py`, `wfo_cache.py`, the test file, the spec, and this plan.
- Scorecard JSON keys (`primary.n_oos_gt_0_30`, etc.) match `scripts/scorecard_step1.py` from Step 1 — no new tooling, reuse confirmed.
