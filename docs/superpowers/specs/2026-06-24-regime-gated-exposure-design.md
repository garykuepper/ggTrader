# Regime-Gated Exposure Scaling — Design Spec

**Date:** 2026-06-24
**Status:** Design (pending implementation plan)
**Author:** brainstormed with Claude

## Problem

The validated 5-voter ensemble (`core+macd+vbb`) beats SPY on two of three axes —
Sharpe **0.89 vs 0.58**, MaxDD **-10.5% vs -22.1%** — but **trails on CAGR
(10.5% vs 13.0%)**. It is a *defensive* edge. The goal is to **beat SPY
outright**: OOS CAGR > 13% **and** Sharpe > 0.58, with MaxDD allowed to widen up
to ~SPY's -22% (spend the Sharpe and drawdown cushion to close the CAGR gap).

## Root-cause hypothesis

A long-only reversion book that trails on CAGR is almost always
**under-deployed** — sitting in cash between setups. The simulator sizes each
entry at `SIGNAL_POSITION_SIZE` (default 2%) via `size_type="percent"`
(`simulate.py:198,226-232`), so the book only approaches full deployment when
~50 positions are held concurrently. If the 5-voter rarely does, idle cash caps
CAGR. The fix is then **exposure**, not signal logic — and crucially, more
exposure in *favorable* regimes only, so the Sharpe edge survives.

This hypothesis is **measured, not assumed** (Component 2). If a different lever
wins the diagnostic, we revisit before building Component 3.

## Objective (success criteria)

Validated on the existing 17-fold WFO (`run_wfo`, full 603-stock SP500,
rolling 12mo/3mo), the regime-gated 5-voter must achieve **all** of:

- OOS CAGR **> 13.0%** (beats SPY)
- OOS Sharpe **> 0.58** (beats SPY)
- OOS MaxDD **≥ -22%** (no worse than SPY)

compared against two baselines reported side by side: SPY, and the static
5-voter (CAGR 10.5% / Sharpe 0.89 / DD -10.5%).

## Components

### 1. Market regime classifier (shared core)

A lookahead-safe pure function: SPY OHLCV up to bar *t* → regime state.

- **Trend axis:** SPY close vs its 200-day EMA (above = uptrend, below =
  downtrend).
- **Volatility axis:** SPY trailing realized vol (e.g. 20-day) bucketed into
  calm / normal / turbulent via train-window percentiles.
- **Outputs:** (a) a discrete label for analysis/reporting, and (b) a continuous
  per-bar **exposure scalar** in [scalar_min, scalar_max].
- **Lookahead safety:** every value at *t* uses only SPY data through *t*; the
  vol-bucket percentile boundaries are fit on the WFO **train window only**.

New module: `src/ggTrader/lab/regime.py`. Pure, vectorized, unit-tested in
isolation (no DB, no network).

### 2. Diagnostic (lever selection — runs before Component 3 is built)

Goal: confirm the lever empirically rather than assume exposure scaling.

- **Deployment audit:** for the static 5-voter across the WFO, report avg %
  capital invested, mean/median concurrent-position count, and idle-cash %.
- **Lever sensitivity sweep:** independently vary (a) a flat exposure-scalar,
  (b) `min_agree`, (c) vol-target, and record OOS CAGR vs Sharpe for each.
  Output a CAGR-vs-Sharpe frontier per lever.
- **Regime-conditioning check:** does the winning lever's payoff depend on
  regime label? (Justifies *gating* it by regime vs applying it flat.)
- **Decision gate:** if exposure scaling is not the best CAGR-per-Sharpe lever,
  stop and re-brainstorm Component 3. Documented as a findings note.

Delivered as a one-off analysis script (pattern: `scripts/ablation_voters.py`,
3-way parallel, fork-inherited OHLCV) writing a findings table.

### 3. The lever: regime → exposure scalar

Assuming the diagnostic confirms exposure scaling:

- The classifier maps regime → position-size multiplier, e.g. ~0.5×
  (turbulent-downtrend) up to ~1.5–2.0× (calm-uptrend). The scalar **range** and
  bucket thresholds are **swept parameters fit per train-fold**, never on test.
- Feeds the **existing** `scalar` slot in `simulate_signals`
  (`simulate.py:204-211`) — `base_size * scalar` per bar. **No new simulation
  path.** If a vol-target scalar is also active, the two compose
  multiplicatively (documented; the diagnostic will show whether to combine).
- Leverage > 1× is permitted up to the swept `scalar_max`; the WFO MaxDD
  constraint (≥ -22%) is the guardrail that bounds how aggressive the sweep can
  go.

### 4. WFO validation

- Run the regime-gated 5-voter through `run_wfo`, with the regime→scalar mapping
  parameters selected on each fold's train window only.
- Report the per-fold table + OOS aggregate vs SPY and vs static 5-voter.
- **Ship only if** all three objective criteria are met out-of-sample. If CAGR
  clears 13% but Sharpe drops below 0.58, the scalar range was too aggressive —
  tighten and re-validate.

## Out of scope (YAGNI)

- Regime-adaptive `min_agree` (Lever B) — only revisited if the diagnostic
  picks it over exposure scaling.
- ML-driven regime detection — the SPY trend+vol classifier is deliberately
  simple to avoid overfitting a new model.
- Universe-breadth regime signal — deferred unless SPY trend+vol underperforms
  in validation.
- Live deployment — a separate decision after WFO validation, like every prior
  config change.

## Testing

- **Unit:** `regime.py` classifier — known SPY fixtures → expected labels and
  scalars; explicit lookahead assertions (value at *t* unchanged when future
  bars are appended).
- **Integration:** regime scalar threaded through `simulate_signals` produces
  the expected size scaling; static-vs-gated WFO runs on a small fixture
  universe.
- **Validation:** the 17-fold WFO objective check above is the acceptance test.

## Risks

- **Lookahead leakage** — the dominant risk; contained by pure ≤*t* classifier +
  per-train-fold parameter fitting + explicit lookahead unit tests.
- **Overfitting the scalar range** — mitigated by keeping the parameter count
  tiny (trend split + 3 vol buckets + scalar range) and validating OOS.
- **Diagnostic overturns the hypothesis** — handled by the Component 2 decision
  gate; not a failure, just a redirect.
