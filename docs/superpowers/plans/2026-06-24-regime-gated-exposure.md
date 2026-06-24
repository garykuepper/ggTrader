# Regime-Gated Exposure Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat SPY outright (OOS CAGR > 13% AND Sharpe > 0.58, MaxDD ≥ -22%) by scaling the 5-voter ensemble's market exposure up in favorable regimes and down in adverse ones, via a lookahead-safe SPY trend+vol classifier feeding the existing per-bar size scalar in `simulate_signals`.

**Architecture:** A pure regime classifier (`regime.py`) turns SPY OHLCV-through-*t* into a per-bar exposure scalar. A diagnostic phase first *measures* whether exposure is the right lever (vs `min_agree`/vol-target) and whether its payoff is regime-conditional — a decision gate. If confirmed, the regime scalar composes (multiplies) with any existing vol scalar inside `simulate_signals`, and the regime-gated 5-voter is validated on the existing 17-fold WFO.

**Tech Stack:** Python 3.12, pandas, numpy, vectorbt, pytest. Native `.venv` for research (NOT Docker). Absolute imports from `ggTrader`.

## Global Constraints

- Run natively: `source .venv/bin/activate` before any python/pytest. (Docker is live-only.)
- Strict ruff lint must pass; vectorized pandas/numpy, no per-row Python loops in hot paths.
- Absolute imports from `ggTrader` (e.g. `from ggTrader.lab.regime import ...`).
- **Lookahead safety is mandatory:** every value at bar *t* uses only data through *t*. All scalars derived from rolling/expanding windows are `.shift(1)` lagged (follow `compute_vol_scalar`, `simulate.py:66-82`).
- Universe/WFO settings match the ablation: full SP500 (`DEFAULT_UNIVERSE`), `eval_start=2021-01-31`, rolling 12mo/3mo, 17 folds.
- Baselines reported in every comparison: SPY (CAGR 13.0% / Sharpe 0.58 / DD -22.1%) and static 5-voter (10.5% / 0.89 / -10.5%).
- Acceptance: OOS CAGR > 13.0% AND Sharpe > 0.58 AND MaxDD ≥ -22%.

---

### Task 1: Regime classifier (`regime.py`)

Pure, vectorized, lookahead-safe. Produces a discrete regime label (for analysis) and a continuous per-bar exposure scalar (for the lever). The shared core used by both the diagnostic (Task 3) and the lever (Task 4).

**Files:**
- Create: `src/ggTrader/lab/regime.py`
- Test: `tests/lab/test_regime.py`

**Interfaces:**
- Consumes: nothing (pure functions over a SPY close `pd.Series`).
- Produces:
  - `classify_regime(spy_close: pd.Series, ema_period: int = 200, vol_lookback: int = 20, vol_window: int = 252) -> pd.DataFrame` — returns a frame indexed like `spy_close` with columns `["trend", "vol_bucket", "label"]`. `trend` ∈ {"up","down"}; `vol_bucket` ∈ {"calm","normal","turbulent"}; `label` = `f"{trend}_{vol_bucket}"`. All values lagged 1 bar (computable at *t* from data ≤ *t*).
  - `compute_regime_scalar(spy_close: pd.Series, scalar_map: dict[str, float], ema_period: int = 200, vol_lookback: int = 20, vol_window: int = 252, default: float = 1.0) -> pd.Series` — maps each bar's `label` to its scalar via `scalar_map`, missing labels → `default`. Returns a `pd.Series` aligned to `spy_close.index`, `.fillna(default)`.

- [ ] **Step 1: Write the failing test for trend + lag**

```python
# tests/lab/test_regime.py
import numpy as np
import pandas as pd
import pytest
from ggTrader.lab.regime import classify_regime, compute_regime_scalar


def _spy(n=400, start=100.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(start * np.exp(np.cumsum(rets)), index=idx)


def test_trend_up_when_above_ema():
    spy = _spy(drift=0.002, seed=1)  # steady uptrend
    reg = classify_regime(spy, ema_period=50)
    # After warmup, an uptrending series sits above its EMA → trend "up"
    tail = reg["trend"].dropna().iloc[-50:]
    assert (tail == "up").mean() > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/lab/test_regime.py::test_trend_up_when_above_ema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.regime'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ggTrader/lab/regime.py
"""Lookahead-safe market-regime classifier (SPY trend + volatility)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_UP = "up"
TREND_DOWN = "down"
VOL_BUCKETS = ("calm", "normal", "turbulent")


def classify_regime(
    spy_close: pd.Series,
    ema_period: int = 200,
    vol_lookback: int = 20,
    vol_window: int = 252,
) -> pd.DataFrame:
    """SPY close -> per-bar regime (trend x vol_bucket). All values lagged 1 bar.

    trend: close vs its EMA(ema_period). vol_bucket: trailing realized vol
    (vol_lookback) bucketed by its own expanding 33/66 percentiles over the
    last vol_window bars (causal). Everything shifted 1 bar so the label at t
    uses only data through t-1's close -> usable to size the t entry.
    """
    ema = spy_close.ewm(span=ema_period, adjust=False).mean()
    trend = pd.Series(
        np.where(spy_close > ema, TREND_UP, TREND_DOWN), index=spy_close.index
    )

    rets = spy_close.pct_change(fill_method=None)
    realized = rets.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    lo = realized.rolling(vol_window, min_periods=vol_lookback).quantile(0.33)
    hi = realized.rolling(vol_window, min_periods=vol_lookback).quantile(0.66)
    vol_bucket = pd.Series("normal", index=spy_close.index, dtype=object)
    vol_bucket[realized <= lo] = "calm"
    vol_bucket[realized >= hi] = "turbulent"

    out = pd.DataFrame(
        {"trend": trend, "vol_bucket": vol_bucket}, index=spy_close.index
    ).shift(1)
    out["label"] = out["trend"].str.cat(out["vol_bucket"], sep="_")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/lab/test_regime.py::test_trend_up_when_above_ema -v`
Expected: PASS

- [ ] **Step 5: Write the lookahead-invariance test (the critical safety test)**

```python
def test_regime_is_lookahead_safe():
    """Appending future bars must not change any past label."""
    spy = _spy(n=400, seed=2)
    reg_full = classify_regime(spy, ema_period=50)
    reg_trunc = classify_regime(spy.iloc[:300], ema_period=50)
    # Past labels (well after warmup) must be identical whether or not the
    # future exists. Compare the overlap, skipping warmup NaNs.
    a = reg_full["label"].iloc[100:300]
    b = reg_trunc["label"].iloc[100:300]
    pd.testing.assert_series_equal(a, b)
```

- [ ] **Step 6: Run it — verify PASS** (the `.shift(1)` + causal rolling make this hold)

Run: `source .venv/bin/activate && pytest tests/lab/test_regime.py::test_regime_is_lookahead_safe -v`
Expected: PASS

- [ ] **Step 7: Write the scalar-mapping test**

```python
def test_compute_regime_scalar_maps_labels():
    spy = _spy(drift=0.002, seed=1)
    scalar_map = {
        "up_calm": 2.0, "up_normal": 1.5, "up_turbulent": 1.0,
        "down_calm": 1.0, "down_normal": 0.7, "down_turbulent": 0.5,
    }
    s = compute_regime_scalar(spy, scalar_map, ema_period=50, default=1.0)
    assert s.index.equals(spy.index)
    assert s.notna().all()                      # default fills warmup
    assert set(s.unique()).issubset(set(scalar_map.values()) | {1.0})
    # Uptrend series should spend most time at an up_* scalar (>= 1.0)
    assert (s.iloc[-50:] >= 1.0).mean() > 0.8
```

- [ ] **Step 8: Implement `compute_regime_scalar`, run test → PASS**

```python
def compute_regime_scalar(
    spy_close: pd.Series,
    scalar_map: dict[str, float],
    ema_period: int = 200,
    vol_lookback: int = 20,
    vol_window: int = 252,
    default: float = 1.0,
) -> pd.Series:
    """Per-bar exposure multiplier from regime label via scalar_map."""
    reg = classify_regime(spy_close, ema_period, vol_lookback, vol_window)
    return reg["label"].map(scalar_map).astype(float).fillna(default)
```

Run: `source .venv/bin/activate && pytest tests/lab/test_regime.py -v`
Expected: all PASS

- [ ] **Step 9: Lint + commit**

```bash
source .venv/bin/activate && ruff check src/ggTrader/lab/regime.py tests/lab/test_regime.py
git add src/ggTrader/lab/regime.py tests/lab/test_regime.py
git commit -m "feat(lab): lookahead-safe SPY trend+vol regime classifier"
```

---

### Task 2: Deployment audit script

Measure the static 5-voter's actual capital deployment across the WFO — the empirical test of the under-deployment hypothesis. One-off analysis script (pattern: `scripts/ablation_voters.py`).

**Files:**
- Create: `scripts/deployment_audit.py`

**Interfaces:**
- Consumes: `run_wfo` machinery / `simulate_signals` from `ggTrader.lab`; `EnsembleSignal` (default 5-voter).
- Produces: stdout report — mean % capital invested, mean/median concurrent-position count, idle-cash % — over the full eval window.

- [ ] **Step 1: Write the script**

Load the SP500 universe once (copy the loader block from `scripts/ablation_voters.py:main`). Build the 5-voter targets over the full eval window via `EnsembleSignal(cfg).sweep_signals([modal_combo], symbols, ohlcv)` using the live-recommended combo (`min_agree=3, min_agree_exit=2, bb_std=2.5, ema_fast=20, rsi_oversold=30`). Simulate with `simulate_signals`. From the resulting `vbt.Portfolio`, derive deployment:

```python
# scripts/deployment_audit.py — core measurement
# pf is the vbt Portfolio from a one-strategy simulate_signals call.
invested = pf.asset_value(group_by=False).sum(axis=1)   # $ in positions per bar
total = pf.value()                                       # total equity per bar
deploy_pct = (invested / total).clip(0, None)
n_positions = (pf.asset_value(group_by=False) > 0).sum(axis=1)
print(f"mean deployment: {deploy_pct.mean():.1%}")
print(f"idle cash:       {1 - deploy_pct.mean():.1%}")
print(f"concurrent positions: mean {n_positions.mean():.1f} / median {n_positions.median():.0f} / max {n_positions.max()}")
```

(If `simulate_signals` does not return `pf`, add an optional `return_pf=False` kwarg to it in this task — a 2-line change returning the portfolio when set — and have the audit call it with `return_pf=True`. Keep the default behavior unchanged.)

- [ ] **Step 2: Run it**

Run: `source .venv/bin/activate && python scripts/deployment_audit.py`
Expected: prints deployment/idle/position stats without error. Record the numbers — **if mean deployment is high (>80%) the under-deployment hypothesis is wrong** and Task 3's lever sweep must be read with that in mind.

- [ ] **Step 3: Commit**

```bash
source .venv/bin/activate && ruff check scripts/deployment_audit.py
git add scripts/deployment_audit.py src/ggTrader/lab/simulate.py
git commit -m "feat(lab): 5-voter deployment audit (under-deployment diagnostic)"
```

---

### Task 3: Lever sensitivity sweep + regime-conditioning — **DECISION GATE**

Compare exposure-scalar vs `min_agree` vs vol-target as CAGR levers, and check whether the winner's payoff is regime-conditional. Gates whether we build the regime exposure lever (Task 4) or redirect.

**Files:**
- Create: `scripts/lever_diagnostic.py`

**Interfaces:**
- Consumes: `run_wfo` (returns the formatted table; parse OOS line as in `ablation_voters.parse_table`), `compute_regime_scalar` (Task 1), `EnsembleSignal`.
- Produces: stdout table — per lever setting, OOS CAGR & Sharpe; plus a regime-label breakdown of OOS returns for the best exposure setting.

- [ ] **Step 1: Write the sweep**

Reuse `ablation_voters.parse_table` (import it) to read `run_wfo` output. Run the static 5-voter WFO under three lever families, each as a small set of `base_config` overrides:
  - **exposure scalar (flat):** set `SIGNAL_POSITION_SIZE` ∈ {0.02, 0.03, 0.04, 0.05} (raises per-entry size → more deployment/leverage).
  - **min_agree:** grid `min_agree` ∈ {2, 3} (looser entry → more trades).
  - **vol-target:** `vol_target` ∈ {0.15, 0.20, 0.25} with `vol_cap=2.0`.
Record OOS CAGR & Sharpe for each; print a per-lever CAGR-vs-Sharpe table. Use the 3-way parallel `ProcessPoolExecutor(fork)` pattern from `ablation_voters.py` (load OHLCV once, inherit copy-on-write).

- [ ] **Step 2: Add the regime-conditioning breakdown**

For the best exposure setting's OOS equity curve, join `classify_regime(spy_close)["label"]` and report mean daily OOS return per regime label:

```python
reg = classify_regime(spy_close)["label"].reindex(oos_returns.index).ffill()
by_regime = oos_returns.groupby(reg).agg(["mean", "count"])
print(by_regime)
```

This shows whether returns (and thus the right scalar) differ by regime — the justification for *gating* exposure by regime rather than applying it flat.

- [ ] **Step 3: Run it and record the verdict**

Run: `source .venv/bin/activate && python scripts/lever_diagnostic.py 2>&1 | tee lever_diagnostic.log`
Expected: prints per-lever frontiers + regime breakdown.
**GATE:** if exposure scaling is NOT the best CAGR-per-Sharpe lever, STOP and re-brainstorm before Task 4. Write the verdict (1 paragraph) into the design spec's status and into `docs/roadmap.md` research direction A.

- [ ] **Step 4: Commit**

```bash
source .venv/bin/activate && ruff check scripts/lever_diagnostic.py
git add scripts/lever_diagnostic.py docs/roadmap.md docs/superpowers/specs/2026-06-24-regime-gated-exposure-design.md
git commit -m "feat(lab): lever-selection diagnostic + regime-conditioning (decision gate)"
```

---

### Task 4: Integrate regime scalar into `simulate_signals`

Thread a regime exposure scalar through the simulator, composing (multiplying) with any vol scalar. Build only if Task 3 confirms exposure scaling.

**Files:**
- Modify: `src/ggTrader/lab/simulate.py:197-212` (the vol-targeting block)
- Test: `tests/lab/test_simulate_regime.py`

**Interfaces:**
- Consumes: `compute_regime_scalar` (Task 1).
- Produces: `simulate_signals` honors `base_config["regime_scalar"]` (a `pd.Series` aligned to the price index) — multiplies `base_size` per bar, composing with `vol_target` when both present.

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_simulate_regime.py
import numpy as np
import pandas as pd
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import SignalTargets  # adjust import to actual location


def _toy():
    idx = pd.date_range("2021-01-01", periods=30, freq="B", tz="UTC")
    px = pd.DataFrame({"AAA": np.linspace(100, 110, 30)}, index=idx)
    entries = pd.DataFrame(False, index=idx, columns=["AAA"]); entries.iloc[1] = True
    exits = pd.DataFrame(False, index=idx, columns=["AAA"]); exits.iloc[10] = True
    return px, SignalTargets(entries=entries, exits=exits)


def test_regime_scalar_scales_position_size():
    px, tgt = _toy()
    base = {"START_CASH": 1e5, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1D",
            "SIGNAL_POSITION_SIZE": 0.02}
    # 2x regime scalar should buy ~2x the position vs no scalar.
    scal = pd.Series(2.0, index=px.index)
    cfg2 = {**base, "regime_scalar": scal}
    _, eq_base, _ = simulate_signals({"e": tgt}, px, base)
    _, eq_scaled, _ = simulate_signals({"e": tgt}, px, cfg2)
    # Larger position -> larger absolute equity swing from entry to exit.
    swing_base = abs(eq_base["e"].iloc[10] - eq_base["e"].iloc[1])
    swing_scaled = abs(eq_scaled["e"].iloc[10] - eq_scaled["e"].iloc[1])
    assert swing_scaled > swing_base * 1.5
```

- [ ] **Step 2: Run test → FAIL** (regime_scalar ignored)

Run: `source .venv/bin/activate && pytest tests/lab/test_simulate_regime.py -v`
Expected: FAIL (assert: swings equal)

- [ ] **Step 3: Implement the compose logic**

Replace the vol-targeting block (`simulate.py:197-212`) with a combined scalar:

```python
    # --- Exposure scaling: vol-targeting x regime, composed multiplicatively ---
    base_size = float(base_config.get("SIGNAL_POSITION_SIZE", 0.02))
    combined = pd.Series(1.0, index=close.index)
    vol_target = base_config.get("vol_target")
    if vol_target is not None:
        vol_lookback = int(base_config.get("vol_lookback", 20))
        vol_cap = float(base_config.get("vol_cap", 2.0))
        vs = compute_vol_scalar(prices, float(vol_target), vol_lookback, vol_cap)
        combined = combined * vs.reindex(close.index).ffill().fillna(1.0)
    regime_scalar = base_config.get("regime_scalar")
    if regime_scalar is not None:
        combined = combined * regime_scalar.reindex(close.index).ffill().fillna(1.0)

    if vol_target is not None or regime_scalar is not None:
        size_param: float | pd.DataFrame = pd.DataFrame(
            {col: base_size * combined for col in close.columns}, index=close.index
        )
    else:
        size_param = base_size
```

- [ ] **Step 4: Run test → PASS**

Run: `source .venv/bin/activate && pytest tests/lab/test_simulate_regime.py -v`
Expected: PASS

- [ ] **Step 5: Regression — vol-targeting still works**

Run: `source .venv/bin/activate && pytest tests/lab -q`
Expected: all PASS (no behavior change when neither scalar is set; vol-target path preserved).

- [ ] **Step 6: Lint + commit**

```bash
source .venv/bin/activate && ruff check src/ggTrader/lab/simulate.py tests/lab/test_simulate_regime.py
git add src/ggTrader/lab/simulate.py tests/lab/test_simulate_regime.py
git commit -m "feat(lab): regime exposure scalar in simulate_signals (composes with vol-target)"
```

---

### Task 5: WFO validation of the regime-gated 5-voter

Run the regime-gated 5-voter through the 17-fold WFO and check the acceptance criteria. The regime→scalar map is the swept parameter; keep it small.

**Files:**
- Create: `scripts/validate_regime_gated.py`

**Interfaces:**
- Consumes: `run_wfo`, `compute_regime_scalar` (Task 1), regime scalar path in `simulate_signals` (Task 4), `EnsembleSignal`.
- Produces: stdout — per-fold WFO table + OOS aggregate vs SPY and static 5-voter, with an explicit PASS/FAIL on the three acceptance criteria.

- [ ] **Step 1: Write the validator**

Load universe once. Compute `regime_scalar = compute_regime_scalar(spy_close, scalar_map)` for a small set of candidate `scalar_map`s (start with one: `{up_calm:2.0, up_normal:1.5, up_turbulent:1.0, down_calm:1.0, down_normal:0.7, down_turbulent:0.5}`). Pass it via `base_config["regime_scalar"]` into `run_wfo` (thread `base_config` through unchanged — `run_wfo` already forwards `base_config` to `_sweep_fold` → `simulate_signals`). Parse the OOS line; print acceptance check:

```python
ok = (oos_cagr > 13.0) and (oos_sharpe > 0.58) and (oos_maxdd >= -22.0)
print(f"ACCEPTANCE: CAGR {oos_cagr:.1f}>13 & Sharpe {oos_sharpe:.2f}>0.58 "
      f"& MaxDD {oos_maxdd:.1f}>=-22  -> {'PASS' if ok else 'FAIL'}")
```

Verify `run_wfo`/`_sweep_fold` forward `base_config` to `simulate_signals` so `regime_scalar` reaches the sim; if the OOS-curve rebuild path (`wfo.py:530-534`) builds its own `sim_config`, ensure `regime_scalar` is included there too (it merges `base_config` + `stop_p`, so a Series value passes through — confirm in a dry run).

- [ ] **Step 2: Run validation**

Run: `source .venv/bin/activate && python scripts/validate_regime_gated.py 2>&1 | tee validate_regime_gated.log`
Expected: prints the WFO table + ACCEPTANCE line. If FAIL on Sharpe, narrow the scalar range (e.g. cap up_calm at 1.5) and rerun; if FAIL on CAGR, widen it (bounded by the DD criterion).

- [ ] **Step 3: Record results + update docs**

Write the validated numbers into `docs/roadmap.md` (research direction A → result) and the design spec status. If PASS, note the candidate live config; live deployment remains a separate user decision.

- [ ] **Step 4: Commit**

```bash
source .venv/bin/activate && ruff check scripts/validate_regime_gated.py
git add scripts/validate_regime_gated.py docs/roadmap.md docs/superpowers/specs/2026-06-24-regime-gated-exposure-design.md
git commit -m "feat(lab): WFO validation of regime-gated 5-voter vs SPY"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (regime classifier) → Task 1. ✅
- Component 2 (diagnostic: deployment audit + lever sweep + regime-conditioning + decision gate) → Tasks 2 & 3. ✅
- Component 3 (regime → exposure scalar via existing `simulate_signals` hook) → Task 4. ✅
- Component 4 (WFO validation against the objective) → Task 5. ✅
- Lookahead safety → Task 1 Steps 5-6 (invariance test) + Global Constraints. ✅
- Testing (unit/integration/validation) → Task 1 unit, Task 4 integration, Task 5 validation. ✅

**Placeholder scan:** No TBD/TODO; all code steps show real code; the scalar_map is a concrete starting value, swept in Task 5. ✅

**Type consistency:** `classify_regime` → DataFrame with `label`; `compute_regime_scalar` consumes that and returns a `pd.Series`; `simulate_signals` reads `base_config["regime_scalar"]` as a `pd.Series`. Names consistent across Tasks 1, 4, 5. ✅

**Note on `SignalTargets` import** (Task 4 test): adjust the import to its actual module (`ggTrader.lab.strategy` or `...simulate`) — confirm at implementation time via `grep -rn "class SignalTargets"`.
