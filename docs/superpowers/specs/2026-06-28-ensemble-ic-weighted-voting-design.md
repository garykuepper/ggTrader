# Design: IC-Weighted Voting Ensemble (`ensemble_ic`) — v1

**Date:** 2026-06-28
**Status:** Approved (brainstorm) — pending spec review
**Source:** Rank 1 of `docs/research/2026-06-28-alpha-optimization-execution-plan.md`

## Goal

Replace the equal-weight 2-of-5 voting rule with a Spearman-rank Information
Coefficient (IC) weighted ensemble, where each voter's contribution is scaled by
its trailing cross-sectional predictive power. Ship as a **new sibling strategy**
(`ensemble_ic`) so the validated, going-live `ensemble` baseline is untouched.

This is the singular final equity-book experiment: if `ensemble_ic` cannot beat
the baseline OOS Sharpe of 1.12 after NDH + DSR, the equity selection book is
closed to further research.

## Non-Goals (deferred to v2 / out of scope)

- **Error-correlation clustering** (the 0.70 redundancy guard). Deferred: with only
  5 voters, normalized weights already down-weight a weak/correlated voter. Add
  only if the fit shows two voters dominating in lockstep.
- **Exit-side IC weighting.** The lever is entry-side; exits reuse the baseline
  path verbatim.
- **Per-fold fit-and-freeze in the WFO layer.** Rejected in favor of a causal
  trailing-window fit (see Design Decisions).

## Design Decisions (settled in brainstorm)

1. **Causal trailing-window IC**, recomputed quarterly inside the strategy, using
   only forward returns realized by the rebalance date. Pure function of past data
   → leak-safe by construction; mirrors the existing conviction-sizes pattern; no
   harness/WFO changes.
2. **Raw indicator values** feed the IC (not the existing `*_strength` functions,
   which are clipped to [0,1] and zero outside the active zone — that collapses the
   cross-section into tied zeros and yields a statistically weak IC).
3. **Mean of daily cross-sectional Spearman ICs** as the IC estimator (standard
   definition), not a single pooled correlation over all (raw, fwd) pairs —
   pooling lets cross-day level differences contaminate the rank.

## Architecture

Three focused units; the baseline `EnsembleSignal` is not modified.

### 1. `src/ggTrader/lab/strategies/indicators.py` — add 5 raw-value extractors

Add alongside the existing `*_signals` / `*_strength` functions. Each returns a
`(time × symbol)` float DataFrame of the **unclipped, point-in-time** raw value
used to rank names cross-sectionally:

| Function | Raw value |
|---|---|
| `rsi_raw(close, period)` | RSI level (0–100), **negated** so higher = more oversold = bullish |
| `bb_raw(close, period, std)` | `%b` = (close − lower) / (upper − lower), **negated** so deeper-below-lower ranks high |
| `ema_raw(close, fast, slow)` | signed `(ema_fast − ema_slow) / ema_slow` |
| `macd_raw(close, fast, slow, signal)` | MACD histogram (macd − signal line) |
| `vbb_raw(close, volume, period, std, vol_period)` | `%b` (as `bb_raw`) gated to bars with volume confirmation; **negated** |

**Directional convention:** every raw value is oriented so that a **higher value =
a more bullish / deeper-oversold reading**. This makes a *positive* IC mean "the
voter is predictive" uniformly across voters, so `max(0, IC_j)` prunes genuinely
unhelpful voters.

### 2. `src/ggTrader/lab/strategies/ic_weights.py` — new module (the heart)

Pure, independently testable functions:

- `forward_returns(close: DataFrame, horizon: int = 3) -> DataFrame`
  Forward return `close.shift(-horizon)/close - 1` per (date, symbol).

- `daily_cross_sectional_ic(raw: DataFrame, fwd: DataFrame) -> Series`
  Per-day Spearman rank correlation across symbols between `raw[d]` and `fwd[d]`.
  Returns a per-date IC series (NaN on days with < `min_names` valid pairs).

- `ic_weight_schedule(raw_by_voter: dict[str, DataFrame], close: DataFrame, *,
  lookback_months: int, horizon: int = 3, rebalance: str = "Q",
  min_names: int = 10) -> DataFrame`
  Returns a `(time × voter)` weight schedule, forward-filled between quarterly
  rebalance dates. At each rebalance date `t_k`:
  1. Trailing window `W = (t_k − lookback_months, t_k]`.
  2. **Leak guard:** drop the last `horizon` bars of `W` (their forward returns
     are not realized by `t_k`).
  3. For each voter `j`: `IC_j = mean over W of daily_cross_sectional_ic(raw_j, fwd)`.
  4. `w_j = max(0, IC_j) / Σ_k max(0, IC_k)`.
  5. **Degenerate guard:** if all `IC ≤ 0` (or denominator 0), fall back to equal
     weights `1/M`.
  6. **Warmup:** before the first rebalance date with a full trailing window,
     equal weights `1/M` (= baseline behavior).

### 3. `src/ggTrader/lab/strategies/ensemble_ic.py` — new `EnsembleICSignal`

`name = "ensemble_ic"`, `target_kind = "signals"`. Implements the `Strategy`
protocol (`select`, `to_targets`, `sweep_params`, `sweep_signals`).

Signal generation:
1. Build each voter's `ent_j` (reuse existing `*_signals`) and `raw_j` (new
   extractors), over the active voter set (default `FIVE_VOTERS`).
2. `weights = ic_weight_schedule(raw_by_voter, close, ...)` → `(time × voter)`.
3. `weighted_score[d, s] = Σ_j weights[d, j] · ent_j[d, s]`. Since `Σ_j w_j = 1`,
   `weighted_score ∈ [0, 1]` = fraction of IC-weight in agreement.
4. `entries = weighted_score ≥ consensus_threshold`.
5. **Exits:** identical to `EnsembleSignal` — RSI independent exit (when RSI
   active) OR consensus exit `≥ min_agree_exit`; same `td_stop` / `exits_enabled`
   handling. Reuse the baseline logic verbatim (extract a shared helper if needed).

## Parameters & Sweep

New swept axes (the only two):
- `consensus_threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`
- `ic_lookback_months ∈ {3, 6, 12}`

Fixed: `horizon = 3`, `rebalance = "Q"`, `voters = FIVE_VOTERS`. All indicator
params stay pinned at baseline values (same discipline as
`EnsembleSignal.sweep_params`). The DSR gate must count these two axes as the
search trials.

## Leak Safety

`entries[d]` depends only on:
- `weights[d]` — from the most recent quarterly fit `≤ d`, itself computed from
  data `≤ (fit_date − horizon)`, and
- `ent_j[d]` — already point-in-time.

Therefore `entries[≤ d]` is invariant to any data after `d`. Because the lab
`harness.leak_check` only covers `select()` (not `to_targets`), this invariance
gets a **dedicated unit test** (see Testing).

## Reduces-to-Baseline Property

With equal weights and `consensus_threshold = min_agree / M = 2/5 = 0.4`,
`ensemble_ic` reproduces the 2-of-5 `ensemble` entries exactly. This is the first
regression anchor.

## Testing (TDD)

`ic_weights` (unit):
- toy input → hand-computed daily IC, window-mean IC, and normalized weights;
- **truncation-invariance**: `weighted_score[≤ d]` unchanged when post-`d` rows are
  dropped (the critical leak test);
- all-non-positive IC → equal weights;
- warmup (before first full window) → equal weights;
- `< min_names` valid pairs on a day → that day's IC is NaN and excluded.

raw extractors (unit): directionality (higher = more bullish) + shape/NaN sanity.

`EnsembleICSignal` (unit):
- weighted thresholding on a synthetic frame → expected entries;
- **reduces-to-baseline** property (equal weights, threshold 0.4 == 2-of-5);
- exits identical to baseline on a shared synthetic frame.

integration:
- runs through `harness.walkforward` on a small universe; produces a summary row.

## Registration & Validation

- Register `ensemble_ic` in the strategy registry and CLI `--strategy` choices.
- Validate: `ggt lab --strategy ensemble_ic` WFO on SP500 through NDH + DSR.
- **Go/No-Go:** OOS Sharpe vs the 1.12 baseline, gate-honest. A pass must clear DSR
  accounting for the 2 new sweep axes. No live deploy regardless until it beats the
  live baseline with statistical significance (per report §5 discipline).

## Files Touched

| File | Change |
|---|---|
| `src/ggTrader/lab/strategies/indicators.py` | +5 raw-value extractors |
| `src/ggTrader/lab/strategies/ic_weights.py` | new module |
| `src/ggTrader/lab/strategies/ensemble_ic.py` | new `EnsembleICSignal` |
| `src/ggTrader/lab/strategies/__init__.py` | add `ensemble_ic` to `STRATEGY_REGISTRY` + exports |
| `src/ggTrader/lab/strategies/signals.py` | add `ensemble_ic` to `build_signal_strategy` map + `SIGNAL_STRATEGY_NAMES` (drives CLI `--strategy` choices and the `--wfo` gate) |
| `tests/` | unit + integration tests above |
