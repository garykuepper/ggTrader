# Trailing Stop Exit Strategies — Design Spec

**Date:** 2026-06-17
**Approach:** Composable simulation-layer stops via vbt native `sl_stop` + `sl_trail`

## Overview

Add trailing stop exits as a composable layer on any signal strategy. Two
modes: fixed fractional trailing stop and ATR-adaptive trailing stop. Both
are implemented entirely at the simulation layer — strategies are unaware
of stops. Stops integrate with the parameter sweep tooling for grid search
over entry params + stop params simultaneously.

## Core Architecture

Trailing stops are a **simulation-layer concern**, not a strategy concern:

1. **Entry signals** come from existing strategies (ema_cross, wfo_tournament,
   future ones) — unchanged.
2. **Stop params** are specified via sweep CLI (`--sweep-param ts_stop=0.02,0.03`
   or `--sweep-param atr_mult=2.0,3.0`).
3. **`simulate_signals`** gains stop awareness: if stop params are present in
   `base_config`, it passes them to `vbt.Portfolio.from_signals` as `sl_stop`
   + `sl_trail=True`.
4. For **ATR-adaptive stops**, a pure vectorized function computes
   `atr_mult * ATR(period) / close` as a 2D DataFrame, passed as the per-bar
   `sl_stop` array.

Stops don't affect signal generation — `sweep_signals()` is untouched.

## Stop Parameter Types

### Fixed trailing stop

A constant fraction from peak equity per position.

- **Param:** `ts_stop` (float, e.g. 0.03 = 3% trailing stop)
- **vbt mapping:** `sl_stop=0.03, sl_trail=True`
- Passed directly as a scalar to `from_signals`.

### ATR-adaptive trailing stop

Volatility-scaled distance from peak, adapting per-bar per-symbol.

- **Params:** `atr_period` (int, default 14) + `atr_mult` (float, e.g. 2.0)
- **Computation:** `sl_stop[t, sym] = atr_mult * ATR(atr_period)[t, sym] / close[t, sym]`
- **vbt mapping:** `sl_stop=<2D DataFrame>, sl_trail=True`
- ATR computed via pandas rolling on true range: `max(high-low, |high-prev_close|, |low-prev_close|).rolling(period).mean()`
- Fully vectorized — one rolling call across all symbols.

### Mutual exclusivity

If a combo has both `ts_stop` and `atr_mult`, it is invalid and filtered by
the grid validator. `atr_mult` implies `sl_trail=True` and an ATR-based
`sl_stop`; `ts_stop` implies `sl_trail=True` and a fixed scalar `sl_stop`.
They cannot coexist.

### Where stop params come from

Stop params are NOT declared in strategy `sweep_params()`. They come from
the CLI only: `--sweep-param ts_stop=0.02,0.03,0.05` or
`--sweep-param atr_mult=1.5,2.0,3.0 --sweep-param atr_period=14,21`.
This keeps strategies decoupled from exit mechanics.

## Implementation Touch Points

### `src/ggTrader/lab/simulate.py` — Stop-aware simulation

The only file that talks to vbt. Changes to `simulate_signals`:

- Extract `ts_stop`, `atr_period`, `atr_mult` from `base_config` if present.
- If `ts_stop` is set: pass `sl_stop=ts_stop, sl_trail=True` to `from_signals`.
- If `atr_mult` is set: compute ATR stop array, pass `sl_stop=array, sl_trail=True`.
- If neither is set: current behavior (no stops).

New helper function:

```python
def compute_atr_stop(
    prices: pd.DataFrame, atr_period: int, atr_mult: float
) -> pd.DataFrame:
    """Vectorized ATR trailing stop as a fractional distance from close.

    Returns a (time x symbol) DataFrame of stop fractions suitable for
    vbt's sl_stop parameter.
    """
```

Uses true range: `max(high - low, |high - prev_close|, |low - prev_close|)`
rolled over `atr_period` bars. Divided by close to get a fractional distance.
Requires high/low/close columns. Currently `simulate_signals` receives
close-only prices. The sweep orchestrator (`run_sweep`) already has access
to the full OHLCV DataFrame. When ATR stop params are present, `run_sweep`
extracts high/low/close and passes them to `simulate_signals` via a new
optional `ohlcv` parameter. `simulate_signals` forwards this to
`compute_atr_stop`. When ATR params are absent, `ohlcv` is not passed and
the signature is backward-compatible.

### `src/ggTrader/lab/sweep.py` — Param splitting & grouped simulation

The sweep orchestrator splits each combo's params into:

- **Signal params** (e.g. `ema_fast`, `ema_slow`) → forwarded to `sweep_signals()`
- **Stop params** (`ts_stop`, `atr_period`, `atr_mult`) → merged into `base_config`

Since stop params vary per combo but `simulate_signals` takes one `base_config`,
combos are grouped by their stop config. One `simulate_signals` call per unique
stop-param set. Common cases:

- Sweeping entry params only (no stops) → 1 vbt call (current behavior).
- Sweeping entry params with one fixed stop → 1 vbt call.
- Sweeping entry params + multiple stop values → N vbt calls (one per
  unique stop config), each batching all entry combos that share that stop.

Known stop param names (constant set): `ts_stop`, `atr_period`, `atr_mult`.

### `src/ggTrader/lab/sweep.py` — Grid validation

`_is_valid_combo` gains one rule: reject combos containing both `ts_stop`
and `atr_mult`.

### Files NOT changed

- `strategies/signals.py` — sweep_signals unchanged
- `strategies/momentum.py` — unchanged
- `strategy.py` — protocol unchanged
- `persist.py` — unchanged (stop params are in the combo's `params` JSONB)
- `cli.py` — unchanged (stop params come via existing `--sweep-param`)
- `harness.py` — single-run walkforward unchanged

## Sweep Integration

### CLI examples

```bash
# EMA cross sweep + fixed trailing stop sweep
ggt lab --strategy ema_cross --sweep \
    --sweep-param ema_fast=5,10,20 --sweep-param ema_slow=50,100 \
    --sweep-param ts_stop=0.02,0.03,0.05

# EMA cross sweep + ATR-adaptive stop sweep
ggt lab --strategy ema_cross --sweep \
    --sweep-param ema_fast=10,20 --sweep-param ema_slow=50,100 \
    --sweep-param atr_mult=1.5,2.0,3.0 --sweep-param atr_period=14

# ATR stop only (one entry config, sweep exits)
ggt lab --strategy ema_cross --sweep \
    --sweep-param atr_mult=1.0,1.5,2.0,2.5,3.0
```

### Combo naming

Stop params appear in the combo name like any other param:
`ema_cross__atr_mult2.0_atr_period14_ema_fast10_ema_slow50`

### Output table

Unchanged format — same ranked Sharpe table. Stop params visible in the
combo name column.

## Performance

- **Fixed stops:** Zero overhead — vbt handles `sl_stop` scalar natively.
- **ATR stops:** One `rolling().mean()` call on the true range DataFrame
  (vectorized across all symbols). This adds O(T × S) pandas work, where
  T = bars and S = symbols. For 2660 bars × 563 symbols, this is ~1.5M
  float ops — negligible vs the vbt simulation.
- **Multiple stop configs:** One extra vbt call per unique stop config.
  For a typical sweep of 3 stop values × 6 entry combos = 3 vbt calls
  of 6 combos each, vs 1 call of 18 combos. vbt's per-call overhead
  is small; the simulation work is proportional to total columns, so
  total compute is the same.

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/ggTrader/lab/simulate.py` | Modify | Add stop params to `simulate_signals`, add `compute_atr_stop` helper |
| `src/ggTrader/lab/sweep.py` | Modify | Split combo params (signal vs stop), group by stop config, update `_is_valid_combo` |
| `tests/lab/test_simulate_signals.py` | Modify | Tests for fixed and ATR trailing stops in simulate_signals |
| `tests/lab/test_sweep.py` | Modify | Tests for param splitting, stop-config grouping, grid validation |
