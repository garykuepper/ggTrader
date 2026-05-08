# WFO Overfitting Step 1: PARAM_STABILITY_WEIGHT 0.3 → 0.7 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether tightening the cross-fold CV stability penalty from `0.3` to `0.7` reduces selection bias enough to lift the surviving coins' OOS robustness scores above the B-criterion thresholds.

**Architecture:** Single-line config change in `src/ggTrader/utils/run_config.py`. Cache invalidates because `PARAM_STABILITY_WEIGHT` is in `_WFO_RELEVANT_CONFIG_KEYS` — full WFO re-run (~21 minutes). Scorecard captured before/after via direct queries against per-worker JSON outputs and the merged `run_results.json`. No source code changes.

**Tech Stack:** Python config (no logic), Docker Compose (rebuild + run), Postgres/TimescaleDB (cache + runs table), shell + Python for scorecard analysis.

**Spec:** `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `src/ggTrader/utils/run_config.py` | Modify line 208 | Bump `PARAM_STABILITY_WEIGHT` 0.3 → 0.7 |
| `scripts/scorecard_step1.py` | Create | Reusable scorecard generator (B-criterion + secondary metrics) for any research run directory |
| `docs/changelog.md` | Modify (top of file, under existing 2026-05-08 entries) | Record run + scorecard delta + decision-rule outcome |
| `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md` | Modify (append "## Result" section) | Lock in the experiment outcome on the spec itself |

The scorecard script is created (rather than a one-off shell pipeline) because Step 1.5, 2, 3, etc. will all need the same scorecard, and capturing it once as code makes step-to-step comparison reproducible.

---

## Task 1: Capture baseline scorecard from the existing run

The "before" baseline is the 2026-05-08 14:02 run at `results/research/research_20260508_135840/`. We need a reusable scorecard script that produces a deterministic JSON given a run directory, then we run it on the baseline and on the post-Step-1 run.

**Files:**
- Create: `scripts/scorecard_step1.py`
- Read-only inputs: `results/research/research_20260508_135840/worker_*_results.json`, `results/research/research_20260508_135840/phase_stats.json`

- [ ] **Step 1: Write the scorecard script**

Create `scripts/scorecard_step1.py` with the following content. The script reads worker JSONs to compute survivor-level OOS metrics (per-coin from Phase 1 selection), and `phase_stats.json` for Phase 2/3 portfolio-level sanity numbers. Output is JSON for diff-friendly comparison.

```python
#!/usr/bin/env python3
"""Compute WFO research scorecard for a given run directory.

Used to compare research outcomes step-to-step across the WFO overfitting
experiments. Reads per-worker JSON (Phase 1 selections) and phase_stats.json
(Phase 2/3 portfolio results).

Output: JSON with B-criterion primary metrics, secondary metrics, and
Phase 3 sanity numbers. Print to stdout; pipe to a file or jq as needed.

Usage:
    python scripts/scorecard_step1.py results/research/research_20260508_135840
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_per_coin_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load and merge per_coin_results from all worker JSONs in run_dir."""
    merged: dict[str, dict[str, Any]] = {}
    for worker_path in sorted(run_dir.glob("worker_*_results.json")):
        try:
            data = json.loads(worker_path.read_text())
        except Exception as exc:
            print(f"WARN: failed to read {worker_path.name}: {exc!r}", file=sys.stderr)
            continue
        per_coin = data.get("strategy_parameters", {}).get("per_coin", {}) or {}
        merged.update(per_coin)
    return merged


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def scorecard(run_dir: Path) -> dict[str, Any]:
    per_coin = load_per_coin_results(run_dir)
    rows = []
    for symbol, r in per_coin.items():
        is_score = safe_float(r.get("is_robustness_score"))
        oos_score = safe_float(r.get("oos_robustness_score"))
        cons = safe_float(r.get("fold_consistency"))
        gate = safe_float(r.get("robustness_score"))
        rows.append({
            "symbol": symbol,
            "is": is_score,
            "oos": oos_score,
            "consistency": cons,
            "gate": gate,
        })

    # B-criterion primary metrics
    oos_finite = [row["oos"] for row in rows if row["oos"] is not None]
    cons_finite = [row["consistency"] for row in rows if row["consistency"] is not None]
    n_oos_gt_30 = sum(1 for v in oos_finite if v > 0.30)
    n_oos_gt_0 = sum(1 for v in oos_finite if v > 0.0)
    median_cons = statistics.median(cons_finite) if cons_finite else None

    # Secondary metrics
    n_survivors = len(rows)
    gaps = [
        row["is"] - row["oos"]
        for row in rows
        if row["is"] is not None and row["oos"] is not None
    ]
    median_gap = statistics.median(gaps) if gaps else None

    # Phase 3 sanity from phase_stats.json (best-effort)
    phase_stats_path = run_dir / "phase_stats.json"
    phase3 = {}
    if phase_stats_path.exists():
        try:
            ps = json.loads(phase_stats_path.read_text())
            p3 = ps.get("phase_3_stats") or {}
            phase3 = {
                "total_return_pct": safe_float(p3.get("total_return_pct") or p3.get("total_return")),
                "cagr_pct": safe_float(p3.get("cagr_pct") or p3.get("cagr")),
                "sharpe": safe_float(p3.get("sharpe")),
                "max_drawdown_pct": safe_float(p3.get("max_drawdown_pct") or p3.get("max_drawdown")),
                "btc_cagr_pct": safe_float(p3.get("btc_cagr_pct")),
            }
        except Exception as exc:
            print(f"WARN: could not parse phase_stats.json: {exc!r}", file=sys.stderr)

    return {
        "run_dir": str(run_dir),
        "primary": {
            "n_oos_gt_0_30": n_oos_gt_30,
            "median_fold_consistency": median_cons,
            "n_oos_gt_0": n_oos_gt_0,
        },
        "secondary": {
            "n_survivors": n_survivors,
            "median_is_minus_oos_gap": median_gap,
        },
        "phase3_sanity": phase3,
        "per_coin": sorted(rows, key=lambda r: -(r["gate"] or 0)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    if not args.run_dir.is_dir():
        print(f"ERROR: not a directory: {args.run_dir}", file=sys.stderr)
        return 2
    print(json.dumps(scorecard(args.run_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the scorecard against the baseline run**

Run:
```bash
python /home/flynn/ggTrader/scripts/scorecard_step1.py \
    /home/flynn/ggTrader/results/research/research_20260508_135840 \
    > /tmp/scorecard_baseline.json
cat /tmp/scorecard_baseline.json | head -40
```

Expected output (top of JSON): a `primary` block with `n_oos_gt_0_30`, `median_fold_consistency`, `n_oos_gt_0`; a `secondary` block with `n_survivors` and `median_is_minus_oos_gap`; a `phase3_sanity` block; and a `per_coin` array sorted by gate score descending.

Sanity check the output reflects the data we've already eyeballed: ~17 survivors, TRX at the top with OOS≈0.76, multiple coins with negative OOS in the per_coin tail.

- [ ] **Step 3: Commit the scorecard script (no config change yet)**

```bash
cd /home/flynn/ggTrader
git add scripts/scorecard_step1.py
git commit -m "$(cat <<'EOF'
tools: add WFO research scorecard generator (Step 1 baseline)

Computes B-criterion primary metrics (n_oos_gt_0_30, median_fold_consistency,
n_oos_gt_0), secondary metrics, and Phase 3 portfolio sanity numbers from
a research run directory. Reusable across all WFO overfitting steps for
deterministic before/after comparison.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Apply the config change

**Files:**
- Modify: `src/ggTrader/utils/run_config.py:208`

- [ ] **Step 1: Edit the config line**

Open `/home/flynn/ggTrader/src/ggTrader/utils/run_config.py` and change line 208:

```diff
-        "PARAM_STABILITY_WEIGHT": 0.3,
+        "PARAM_STABILITY_WEIGHT": 0.7,
```

The surrounding comment block (lines 205–208) describes the knob and remains accurate.

- [ ] **Step 2: Verify the change**

Run:
```bash
grep -n "PARAM_STABILITY_WEIGHT" /home/flynn/ggTrader/src/ggTrader/utils/run_config.py
```

Expected output:
```
208:        "PARAM_STABILITY_WEIGHT": 0.7,
```

(The cache file at `src/ggTrader/data/cache/wfo_cache.py:33` also lists the key for cache-keying — that file is **not** modified.)

- [ ] **Step 3: Commit the config change**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py
git commit -m "$(cat <<'EOF'
config(wfo): bump PARAM_STABILITY_WEIGHT 0.3 -> 0.7 (overfitting Step 1)

Per-fold IS spread analysis on the 2026-05-08 research cache shows
max-of-N selection bias matching noise theory: max-IS cell is 1.5–3.0
Sharpe units above the per-fold median with cell sigma ~1.0 across
~144 cells. The current PARAM_STABILITY_WEIGHT=0.3 is not enough to
neutralize this; bump to 0.7 attacks high-CV outliers harder.

Cache invalidates (knob is in _WFO_RELEVANT_CONFIG_KEYS).

Spec: docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rebuild the production image and re-run research

**Files:** none modified — Docker build + research run.

- [ ] **Step 1: Rebuild the live image with the new config**

The `src/` directory is **not** volume-mounted in the production stack (see `/home/flynn/ggTrader/docker-compose.yaml`), so the config change requires an image rebuild before `docker compose run` can use it.

```bash
cd /home/flynn/ggTrader
docker compose build --no-cache ggtrader_live
```

Expected: build completes successfully (status `Successfully tagged ghcr.io/garykuepper/ggtrader:latest` or equivalent). No errors. Takes ~3–5 min depending on layer cache state.

- [ ] **Step 2: Restart the live trader on the new image**

```bash
cd /home/flynn/ggTrader
docker compose up -d ggtrader_live
```

Expected: `ggtrader_live` container restarts on the new image. Verify with `docker compose ps ggtrader_live` — `STATUS` should show `Up <a few seconds>`.

- [ ] **Step 3: Run the research pipeline**

This is the long-running step. The cache invalidates (because `PARAM_STABILITY_WEIGHT` is in the cache key) so the full WFO loop re-runs across all 100 coins / 5 workers in parallel.

```bash
cd /home/flynn/ggTrader
docker compose run --rm ggtrader_live python -u ggt.py research \
    --top 100 --days 1095 --end-date 2026-05-01 \
    2>&1 | tee /tmp/step1_research.log
```

Expected runtime: ~21 minutes (matches yesterday's parallel-warm-OHLCV run).

Expected output tail:
- `[merge] Merged N coins from 5 workers -> run_results.json` — N may be lower than 17 if the tighter stability bar drops more coins.
- `Phase 2 - full training/test range combined portfolio vs Buy & Hold:` block with `Total Return`, `CAGR`, `Sharpe Ratio`.
- `Phase 3 - YTD combined portfolio vs Buy & Hold:` block.
- `Research Pipeline complete.`

Note the new run directory printed near the start (`results/research/research_<timestamp>/`). Save it for the next task:

```bash
NEW_RUN=$(ls -td /home/flynn/ggTrader/results/research/research_* | head -1)
echo "$NEW_RUN" | tee /tmp/step1_run_dir.txt
```

If the run fails (any worker non-zero exit, or the pipeline raises), do NOT proceed — diagnose first. The most likely failure mode is "all coins dropped by gates in some worker," which the recently-shipped fix in `src/ggTrader/core/orchestrator.py:611` should handle gracefully (empty per_coin worker output, merger tolerates).

---

## Task 4: Capture the post-Step-1 scorecard and compare

**Files:**
- Read-only: the new run directory (path stored in `/tmp/step1_run_dir.txt`)

- [ ] **Step 1: Run the scorecard against the new run**

```bash
NEW_RUN=$(cat /tmp/step1_run_dir.txt)
python /home/flynn/ggTrader/scripts/scorecard_step1.py "$NEW_RUN" \
    > /tmp/scorecard_step1.json
```

- [ ] **Step 2: Diff primary metrics**

```bash
echo "=== BASELINE (PARAM_STABILITY_WEIGHT=0.3) ==="
jq '.primary, .secondary, .phase3_sanity' /tmp/scorecard_baseline.json
echo ""
echo "=== STEP 1 (PARAM_STABILITY_WEIGHT=0.7) ==="
jq '.primary, .secondary, .phase3_sanity' /tmp/scorecard_step1.json
```

Expected: a clean side-by-side of the four blocks. Inspect the deltas on:
1. `primary.n_oos_gt_0_30` — count of high-quality OOS coins. **Up = good.**
2. `primary.median_fold_consistency` — typical consistency among survivors. **Up = good.**
3. `primary.n_oos_gt_0` — count clearing zero-OOS bar. **Up = good.**

Secondary:
4. `secondary.n_survivors` — total post-gate survivor count. Floor: **5**. If < 5, treat as partial revert (proceed to next step's revert branch).
5. `secondary.median_is_minus_oos_gap` — symptom indicator. Should drop, but not the primary judge.

Sanity (Phase 3):
6. `phase3_sanity.total_return_pct`, `sharpe`, `max_drawdown_pct` — should not crater (e.g. >50% return drop or Sharpe falling below 0.5 would warrant a closer look).

- [ ] **Step 3: Apply the decision rule**

Per the spec, classify the outcome:

- **ADVANCE** (proceed to Step 2 / Step 1.5 next session): primary metrics improved on **≥ 2 of 3**. Continue to Task 5 to record the result.
- **REVERT**: primary metrics worsened on **≥ 2 of 3**. Continue to Task 5 to record, then revert the config in a follow-up commit.
- **NEUTRAL**: mixed or flat. Record as-is in Task 5; the spec calls out `0.85` as a reasonable next step before Step 2.
- **HARD FLOOR TRIPPED**: `n_survivors < 5`. Record, then revert to `0.5` in a follow-up (do not proceed at `0.7`).

Write the classification (one of `ADVANCE` / `REVERT` / `NEUTRAL` / `HARD_FLOOR`) to a small marker file for Task 5:

```bash
# Manual classification; choose one based on the diff above.
echo "ADVANCE" > /tmp/step1_decision.txt   # or REVERT / NEUTRAL / HARD_FLOOR
```

(This is a human judgement step — you read the diff and write the conclusion to the file. Keep it explicit so Task 5 records it verbatim.)

---

## Task 5: Record the result in changelog and append to spec

**Files:**
- Modify: `docs/changelog.md` (insert a new section at the top under `## 2026-05-08`, or under today's date if running on a later day)
- Modify: `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md` (append a `## Result` section at the very end)

- [ ] **Step 1: Add the changelog entry**

Open `/home/flynn/ggTrader/docs/changelog.md`. Below the existing `## 2026-05-08` heading and *above* the first existing `### …` section, insert a new entry. Use the actual numbers from `/tmp/scorecard_baseline.json` and `/tmp/scorecard_step1.json` and the classification from `/tmp/step1_decision.txt`.

Template (replace `<…>` placeholders with real values; keep the structure):

```markdown
### WFO overfitting Step 1: PARAM_STABILITY_WEIGHT 0.3 → 0.7

Spec: `docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md`. Plan: `docs/superpowers/plans/2026-05-08-wfo-overfitting-step1-param-stability.md`.

**Change:** `src/ggTrader/utils/run_config.py:208`. Single config-line bump. Cache invalidates; full WFO re-run.

**B-criterion scorecard (baseline → Step 1):**

| Metric | Baseline (0.3) | Step 1 (0.7) | Delta |
|---|---:|---:|---:|
| `n_oos_gt_0_30` | <baseline_n_oos_gt_0_30> | <step1_n_oos_gt_0_30> | <delta> |
| `median_fold_consistency` | <baseline_median_cons> | <step1_median_cons> | <delta> |
| `n_oos_gt_0` | <baseline_n_oos_gt_0> | <step1_n_oos_gt_0> | <delta> |

**Secondary:**

| Metric | Baseline | Step 1 | Delta |
|---|---:|---:|---:|
| `n_survivors` | <baseline_n_survivors> | <step1_n_survivors> | <delta> |
| `median_is_minus_oos_gap` | <baseline_gap> | <step1_gap> | <delta> |

**Phase 3 sanity:** Total Return <step1_total_return_pct>% (vs <baseline_total_return_pct>% baseline), Sharpe <step1_sharpe> (vs <baseline_sharpe>), MaxDD <step1_max_dd>% (vs <baseline_max_dd>%).

**Decision (per spec rule):** <ADVANCE | REVERT | NEUTRAL | HARD_FLOOR>. <one-sentence interpretation; e.g. "Two of three primary metrics improved; advancing to Step 2.">

Run: `<absolute path of new run directory>`.
```

- [ ] **Step 2: Append the same outcome to the spec**

Open `/home/flynn/ggTrader/docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md` and append at the very end (after the multi-step plan reference table):

```markdown

---

## Result

Run executed: `<absolute path of new run directory>` (took ~<minutes> min).

**Primary scorecard delta:**

| Metric | Before (0.3) | After (0.7) | Δ |
|---|---:|---:|---:|
| `n_oos_gt_0_30` | <baseline> | <step1> | <delta> |
| `median_fold_consistency` | <baseline> | <step1> | <delta> |
| `n_oos_gt_0` | <baseline> | <step1> | <delta> |

**Secondary:** `n_survivors` <baseline> → <step1>; `median_is_minus_oos_gap` <baseline> → <step1>.

**Phase 3 sanity:** Total Return <step1_total_return>% (vs <baseline_total_return>%); Sharpe <step1_sharpe> (vs <baseline_sharpe>); MaxDD <step1_max_dd>%.

**Decision:** <ADVANCE | REVERT | NEUTRAL | HARD_FLOOR>.

<2–3 sentence interpretation: which lever moved or didn't, what surprised, and what the next experiment should be (Step 1.5 z-rank, Step 2 grid shrink, or a different value of PARAM_STABILITY_WEIGHT). If REVERT or HARD_FLOOR, state the revert action.>
```

- [ ] **Step 3: Commit changelog + spec update together**

```bash
cd /home/flynn/ggTrader
git add docs/changelog.md docs/superpowers/specs/2026-05-08-wfo-overfitting-step1-param-stability.md
git commit -m "$(cat <<'EOF'
docs: WFO overfitting Step 1 results (PARAM_STABILITY_WEIGHT 0.3 -> 0.7)

Records the before/after B-criterion scorecard, Phase 3 sanity numbers,
and decision-rule outcome for the PARAM_STABILITY_WEIGHT bump experiment.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Conditional revert (only if decision was REVERT or HARD_FLOOR)

Skip this task entirely if the decision in Task 4 was `ADVANCE` or `NEUTRAL`.

**Files:**
- Modify: `src/ggTrader/utils/run_config.py:208`

- [ ] **Step 1: Revert the config**

If decision was **REVERT**, restore `PARAM_STABILITY_WEIGHT` to `0.3`.

If decision was **HARD_FLOOR** (survivor count < 5), set `PARAM_STABILITY_WEIGHT` to `0.5` instead — the spec calls this out as the back-off branch.

```diff
# REVERT branch:
-        "PARAM_STABILITY_WEIGHT": 0.7,
+        "PARAM_STABILITY_WEIGHT": 0.3,

# HARD_FLOOR branch:
-        "PARAM_STABILITY_WEIGHT": 0.7,
+        "PARAM_STABILITY_WEIGHT": 0.5,
```

- [ ] **Step 2: Verify**

```bash
grep -n "PARAM_STABILITY_WEIGHT" /home/flynn/ggTrader/src/ggTrader/utils/run_config.py
```

Expected: `208:        "PARAM_STABILITY_WEIGHT": 0.3,` or `208:        "PARAM_STABILITY_WEIGHT": 0.5,`.

- [ ] **Step 3: Commit the revert**

```bash
cd /home/flynn/ggTrader
git add src/ggTrader/utils/run_config.py
git commit -m "$(cat <<'EOF'
config(wfo): revert PARAM_STABILITY_WEIGHT to <0.3 or 0.5> (Step 1 outcome)

Step 1's tighter stability penalty triggered the <REVERT | HARD_FLOOR>
branch of the spec decision rule. See changelog and spec result section
for the scorecard.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

The live trader stays on the previously-built image until the next deploy — that's fine because the live trader auto-detects the latest research run, and the most recent run's per_coin params reflect the *previous* (good) configuration if we revert.

If at any point the live trader needs the reverted config in its image (e.g. if a 2nd research run is planned soon), repeat Task 3 Step 1 (`docker compose build --no-cache ggtrader_live && docker compose up -d ggtrader_live`).

---

## Self-review

**Spec coverage:**
- Spec "Change" section (single-line config bump) → Task 2.
- Spec "Cache invalidation and re-run" → Task 3 Step 3.
- Spec "Scorecard" definition → `scripts/scorecard_step1.py` in Task 1.
- Spec "Decision rule" → Task 4 Step 3 + Task 6 conditional.
- Spec "Rollback" → Task 6.
- Spec "Result" placeholder for outcome → Task 5 Step 2.

**Placeholder scan:** the changelog and spec-result templates contain `<…>` markers, but they are the spots an executor fills in *with real numbers from the scorecards*, not unfilled tasks. Acceptable.

**Type / interface consistency:** the scorecard JSON shape used in Task 4 (`primary.n_oos_gt_0_30`, `primary.median_fold_consistency`, `primary.n_oos_gt_0`, `secondary.n_survivors`, `secondary.median_is_minus_oos_gap`, `phase3_sanity.total_return_pct`, etc.) matches the keys produced by `scripts/scorecard_step1.py` in Task 1. Names are consistent across tasks.

**Spec requirements not in tasks:** none identified.
