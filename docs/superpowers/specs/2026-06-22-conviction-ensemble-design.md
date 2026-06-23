# Conviction Ensemble Design

**Date:** 2026-06-22
**Status:** Spec
**Goal:** Replace the ensemble's flat 2% position sizing with conviction-weighted sizing (1%–4%) based on sub-signal strength at entry.

---

## Motivation

The current `EnsembleSignal` uses a fixed `SIGNAL_POSITION_SIZE = 0.02` for every entry regardless of signal quality. A barely-triggered 2-of-3 vote gets the same size as a 3-of-3 with deep oversold readings across all indicators. Conviction sizing allocates more capital to higher-confidence entries and less to marginal ones, which should improve risk-adjusted returns without changing the entry/exit logic.

## Design

### Conviction Scoring

Each sub-signal contributes a **strength score** (0–1) on bars where it fires an entry:

| Sub-signal | Strength metric | Formula | Interpretation |
|---|---|---|---|
| BB depth | Distance below lower band, normalized | `clamp((lower - price) / band_width, 0, 1)` | 0 = just touching band, 1 = one full band-width below |
| RSI depth | Distance below oversold threshold, normalized | `clamp((oversold - rsi) / oversold, 0, 1)` | 0 = at threshold, 1 = RSI at 0 |
| EMA gap | Fast-slow EMA separation, normalized | `clamp((ema_fast - ema_slow) / ema_slow, 0, 1)` | 0 = just crossed, 1 = large separation |

On each entry bar, only the **agreeing signals** (those that fired entry=True) contribute. The conviction score is the **mean of their strengths**.

### Size Mapping

```
size = min_size + conviction * (max_size - min_size)
```

Defaults: `min_size = 0.01` (1%), `max_size = 0.04` (4%).

A conviction of 0.0 (weakest possible agreeing signal) gets 1%. A conviction of 1.0 (maximum strength across all agreeing signals) gets 4%. The average entry should land near the current 2%.

### Risk Guardrail Compatibility

The paper trader's risk guardrails (3.3%/trade max, 5% concentration cap, 30 max positions) are enforced downstream in `trader.py` and are independent of the lab's sizing. The 4% max_size stays below the 5% concentration cap. The 3.3%/trade guardrail will clip any entry that exceeds it at execution time.

## Implementation

### New indicator functions (`indicators.py`)

Three new vectorized functions returning `(time x symbol)` float DataFrames:

- `bb_strength(close, period, std)` — returns depth below lower band normalized by band width, clipped [0, 1]. NaN where BB not yet computed.
- `rsi_strength(close, period, oversold)` — returns RSI depth below oversold normalized by oversold level, clipped [0, 1]. NaN where RSI above oversold.
- `ema_strength(close, ema_fast, ema_slow)` — returns EMA gap normalized by slow EMA, clipped [0, 1]. NaN where fast < slow.

### New strategy class (`strategies/ensemble.py`)

`EnsembleConvictionSignal` — same entry/exit logic as `EnsembleSignal`, plus:

- Calls the three strength functions
- On entry bars, masks each strength by whether that sub-signal fired
- Averages the masked strengths to produce a per-bar, per-symbol conviction score
- Maps conviction to size via `min_size + conviction * (max_size - min_size)`
- Returns `SignalTargets(entries, exits, sizes)` — the `sizes` DataFrame flows through `simulate_signals` which already handles it

Constructor adds `min_size: float = 0.01` and `max_size: float = 0.04` parameters. All other parameters inherited from `EnsembleSignal`.

### `sweep_params`

Same as `EnsembleSignal.sweep_params()` — `min_size` and `max_size` are fixed (not swept) initially. Can be added to the sweep grid later if results warrant.

### CLI registration

Register `ensemble_conviction` in the strategy loader so `ggt lab --strategy ensemble_conviction` works.

### Tests

- Unit tests for `bb_strength`, `rsi_strength`, `ema_strength` — verify range [0, 1], NaN handling, edge cases
- Unit test for `EnsembleConvictionSignal` — verify `sizes` is populated, range is [min_size, max_size], entries/exits match plain ensemble
- Integration test: run through `simulate_signals` and verify the portfolio sim completes with sizes

## Validation Plan

Run the same 17-fold SP500 WFO that validated the plain ensemble (2021-01 → 2026-04), comparing:

1. `ensemble` (flat 2%) — baseline, already validated at Sharpe 0.84
2. `ensemble_conviction` (1%–4% conviction sizing)

Key metrics: OOS Sharpe, CAGR, MaxDD, WFE. The conviction variant should improve Sharpe by allocating less to marginal entries and more to high-conviction ones. If Sharpe degrades, the sizing adds noise rather than signal.

## What This Does NOT Change

- Entry/exit logic — identical to current ensemble
- ML feature gate — remains a binary filter in the paper trader; not part of backtest conviction
- Vol targeting — orthogonal overlay; conviction sizes can be further scaled by vol targeting
- Risk guardrails — enforced at execution time, unchanged

## Future Extensions (Out of Scope)

- ML probability as a conviction input (requires vectorized feature extraction for backtest)
- Sweeping `min_size` / `max_size` in WFO
- Per-signal type weighting (e.g. BB 0.4, RSI 0.35, EMA 0.25)
