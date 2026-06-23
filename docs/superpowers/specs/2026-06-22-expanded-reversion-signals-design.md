# Expanded Reversion Signals — Design Spec

## Goal

Add three new reversion signal types to the ensemble, taking the voter pool from 3 (BB reversion, RSI reversion, EMA cross) to 6. More orthogonal voters improve signal diversity and smooth the equity curve — signal diversification is what drove the ensemble's Sharpe 0.84 vs SPY 0.59 in the first place. Each new signal is validated via an ML pre-screen (LightGBM precision check) before the expensive WFO run.

## Architecture

Approach A: individual standalone signal classes + expanded ensemble. Each new signal gets its own class in `signals.py` (same pattern as `BollingerReversionSignal`), indicator math in `indicators.py`, and is wired into the ensemble as an additional voter. No new abstractions — follows the existing codebase pattern exactly.

## New Signals

### 1. MACD Divergence

Detects momentum exhaustion: price makes a new low but MACD histogram doesn't (bullish divergence). Orthogonal to BB/RSI because it measures rate-of-change deceleration, not oversold levels.

- **Entry:** Price makes a lower low within a rolling `divergence_window`-bar window AND the MACD histogram makes a higher low over the same window (bullish divergence).
- **Exit:** MACD histogram crosses below zero (momentum reversal confirmed).
- **MACD computation:** Standard MACD: `macd_line = EMA(close, fast) - EMA(close, slow)`, `signal_line = EMA(macd_line, signal)`, `histogram = macd_line - signal_line`.
- **Divergence detection:** On each bar, compute `price_low = close.rolling(divergence_window).min()` and `hist_low = histogram.rolling(divergence_window).min()`. Entry fires when `close <= price_low` (price at a rolling low) AND `histogram > hist_low` (histogram NOT at its rolling low — i.e., momentum is higher than its recent trough even though price is not). Both conditions true on the same bar = bullish divergence.
- **Sweep params:** `macd_fast` [8, 12], `macd_slow` [21, 26], `macd_signal` [9], `divergence_window` [10, 20].
- **Indicator function:** `macd_signals(close, fast, slow, signal_period, divergence_window)` → `(entries, exits)`.

### 2. Volume-Confirmed BB Reversion

Same Bollinger Band reversion logic as `bb_reversion` but only fires when volume spikes above a multiple of its recent average. Filters out weak drift-below-band entries that tend to fail — a volume spike indicates capitulation, which is when mean reversion works best.

- **Entry:** Close crosses below lower Bollinger Band (same as `bb_reversion`) AND volume on the entry bar exceeds `vol_mult × SMA(volume, vol_period)`.
- **Exit:** Close crosses above the SMA (same as `bb_reversion`).
- **Data requirement:** Needs volume in addition to close. A new `extract_volume(data, symbols)` helper in `indicators.py` extracts the volume DataFrame from multi-level OHLCV data, matching the existing `extract_close` pattern.
- **Sweep params:** `bb_period` [15, 20], `bb_std` [2.0, 2.5], `vol_period` [20], `vol_mult` [1.5, 2.0, 2.5].
- **Indicator function:** `volume_bb_signals(close, volume, bb_period, bb_std, vol_period, vol_mult)` → `(entries, exits)`.

### 3. Multi-Timeframe Reversion

Weekly oversold confirming a daily entry. The weekly timeframe filters noise — weekly oversold means the move is structural, not just a one-day dip.

- **Entry:** Weekly RSI is below `weekly_rsi_oversold` AND daily close is below the daily lower Bollinger Band.
- **Exit:** Weekly RSI crosses above `weekly_rsi_exit` OR daily close crosses above the daily SMA.
- **Weekly resampling:** Resamples daily close to weekly internally via `close.resample('W').last()`, computes weekly RSI, then forward-fills the weekly RSI back to the daily index so the condition can be evaluated on every daily bar. No new data pipeline needed.
- **Sweep params:** `weekly_rsi_period` [7, 14], `weekly_rsi_oversold` [30, 35], `weekly_rsi_exit` [50, 55], `daily_bb_period` [15, 20], `daily_bb_std` [2.0, 2.5].
- **Indicator function:** `mtf_signals(close, weekly_rsi_period, weekly_rsi_oversold, weekly_rsi_exit, daily_bb_period, daily_bb_std)` → `(entries, exits)`.

## ML Pre-Screen

Before running WFO on the expanded ensemble, each new signal is screened for entry quality using a LightGBM classifier. This is a fast (5 min/signal) check that filters out signals whose entries are worse than random before committing to the expensive WFO run (hours).

### Process

1. Generate all entries for the signal across the full SP500 universe (2021-2026) using default sweep params.
2. For each entry bar, compute the same 10 OHLCV features the existing `FeatureGate` uses (returns at multiple windows, realized volatility, volume ratio, RSI, BB position).
3. Label: 5-day forward return > 0 = 1 (profitable), else 0.
4. Train a LightGBM classifier with 5-fold time-series CV (same methodology as the existing feature gate in `src/ggTrader/paper/feature_gate.py`).
5. Report: precision, recall, F1, sample count, and feature importances.

### Go/No-Go Threshold

| Precision | Verdict |
|---|---|
| < 0.50 | **Drop** — worse than coin flip, exclude from ensemble |
| 0.50 - 0.55 | **Borderline** — include in WFO but flag for review |
| > 0.55 | **Strong** — matches or exceeds current ensemble gate (0.585) |

### Implementation

A standalone script `scripts/ml_signal_screen.py` that:
- Accepts `--signal <name>` (any registered signal strategy name).
- Generates entries using the signal's default params over the full data range.
- Trains and evaluates the classifier.
- Prints a summary table and writes results to `results/ml_screen_<signal>_<timestamp>.json`.

## Ensemble Wiring

### Expanded Voting

`EnsembleSignal._generate_signals()` grows from 3 to 6 sub-signal calls:

```
bb_reversion  (existing)
rsi_reversion (existing)
ema_cross     (existing)
macd_divergence    (new)
volume_bb_reversion (new)
mtf_reversion      (new)
```

Vote counting stays the same: `entry_votes = sum(all 6 entry booleans)`, `entries = (entry_votes >= min_agree)`. Same for exits.

### Sweep Params

The ensemble's `sweep_params` adds the new signal parameters. `min_agree` range expands to `[2, 3, 4]` to accommodate 6 voters. The full grid will be large — WFO should use `--sweep-param` overrides to keep the grid manageable (same approach as the conviction comparison).

### Volume Data

The ensemble's `_generate_signals` method currently receives only `close`. With volume-confirmed BB, it also needs volume. The `to_targets` and `sweep_signals` methods already receive the full `data` DataFrame — extract volume alongside close at the ensemble level and pass it to `volume_bb_signals`.

### Conviction Extension

`EnsembleConvictionSignal` gets matching strength functions for the 3 new signals:
- `macd_strength(close, fast, slow, signal_period)` — normalized histogram magnitude.
- `volume_bb_strength(close, volume, bb_period, bb_std, vol_period)` — same as `bb_strength` but scaled by volume spike intensity.
- `mtf_strength(close, weekly_rsi_period, weekly_rsi_oversold, daily_bb_period, daily_bb_std)` — average of weekly RSI depth and daily BB depth.

## File Changes

### New indicator functions in `indicators.py`
- `extract_volume(data, symbols)` — companion to `extract_close` for volume data.
- `macd_signals(close, fast, slow, signal_period, divergence_window)` → `(entries, exits)`.
- `volume_bb_signals(close, volume, bb_period, bb_std, vol_period, vol_mult)` → `(entries, exits)`.
- `mtf_signals(close, weekly_rsi_period, weekly_rsi_oversold, weekly_rsi_exit, daily_bb_period, daily_bb_std)` → `(entries, exits)`.
- `macd_strength(close, fast, slow, signal_period)` → DataFrame [0, 1].
- `volume_bb_strength(close, volume, bb_period, bb_std, vol_period)` → DataFrame [0, 1].
- `mtf_strength(close, weekly_rsi_period, weekly_rsi_oversold, daily_bb_period, daily_bb_std)` → DataFrame [0, 1].

### New standalone signal classes in `signals.py`
- `MACDDivergenceSignal` — standalone MACD divergence strategy.
- `VolumeBBReversionSignal` — standalone volume-confirmed BB reversion.
- `MultiTimeframeReversionSignal` — standalone multi-timeframe reversion.

Each class follows the existing pattern: `name`, `target_kind = "signals"`, `__init__`, `sweep_params`, `select`, `to_targets`, `sweep_signals`.

### Modified `ensemble.py`
- `EnsembleSignal.__init__` gains params for the 3 new signals.
- `EnsembleSignal._generate_signals` calls all 6 sub-signals.
- `EnsembleSignal.sweep_params` includes new signal params, `min_agree` expanded to `[2, 3, 4]`.
- `EnsembleConvictionSignal` mirrors the changes with conviction strength functions.

### Registry updates
- `_build_signal_registry()` and `SIGNAL_STRATEGY_NAMES` in `signals.py` — add 3 new entries.
- `cls_map` in `cli.py` — add 3 new entries.

### New script
- `scripts/ml_signal_screen.py` — ML pre-screen for signal quality evaluation.

### Tests
- `tests/lab/test_macd_signals.py` — ~10 tests (indicator function + class + registration).
- `tests/lab/test_volume_bb_signals.py` — ~10 tests.
- `tests/lab/test_mtf_signals.py` — ~10 tests.
- `tests/lab/test_ml_screen.py` — ~5 tests (script logic, feature computation, threshold).
- `tests/lab/test_ensemble.py` — extend existing ensemble tests to cover 6-voter behavior.

Estimate: ~40 new tests total.

## Validation Plan

1. Implement each signal + tests (all pass).
2. Run ML pre-screen per signal. Drop any below 0.50 precision.
3. Wire survivors into ensemble.
4. Run WFO comparison: 3-voter ensemble (current) vs 6-voter ensemble (expanded) with reduced grid via `--sweep-param` overrides.
5. Compare Sharpe, CAGR, MaxDD. The expanded ensemble should match or beat Sharpe 0.84 with lower MaxDD due to increased signal diversity.

## Out of Scope

- Changing the paper trader's live ensemble (stays on 3-voter until WFO validates the expansion).
- Pluggable voter registry (YAGNI — hardcoded 6 voters is fine).
- Intraday signals (all signals use daily bars).
- New data sources (volume comes from the existing OHLCV pipeline; weekly bars are resampled from daily).
