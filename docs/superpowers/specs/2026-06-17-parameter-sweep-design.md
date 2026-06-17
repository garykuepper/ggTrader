# Parameter Sweep Tooling — Design Spec

**Date:** 2026-06-17
**Strategy:** vectorbt-native multi-param simulation (Approach B)

## Overview

Add parameter sweep capability to the lab bench. Given a strategy and a grid of
parameter values, generate signals for all combos simultaneously using vectorized
computation, simulate them in a single `vbt.Portfolio` call with `group_by`, and
persist ranked summary metrics to TimescaleDB.

## Core Concept

The sweep generates signals for **all param combos at once** using vectorized
EMA computation, stacks them into one wide DataFrame (combos × symbols), and runs
a single `vbt.Portfolio.from_signals` call with `group_by` to get per-combo
portfolio metrics.

- **Signal strategies** (ema_cross, wfo_tournament): signal generation is pure
  math on the close series — compute EMA crosses for every (fast, slow) pair
  simultaneously using broadcasting. Deduplicate unique span values across combos.
- **Weight strategies** (xs_momentum, dual_momentum): selection logic depends on
  params (lookback/skip change rankings), so selection runs per-combo. But the
  final simulation is batched into one `from_orders` call with combo-level grouping.
- The walk-forward loop stays: for each rebalance date, generate signals/weights
  for all combos, then simulate the full window in one vbt call.

## Parameter Grid & Strategy Interface

### Grid declaration

Each strategy declares sweepable parameters via a `sweep_params()` classmethod:

```python
@classmethod
def sweep_params(cls) -> dict[str, list]:
    return {
        "ema_fast": [5, 10, 20, 50],
        "ema_slow": [20, 30, 50, 100, 200],
    }
```

The sweep module computes the Cartesian product, filtering invalid combos
(e.g. `ema_fast >= ema_slow`). Users can override from CLI.

### Strategy construction

`build_signal_strategy` and `build_strategy` accept an optional `params: dict`
kwarg that gets unpacked into the constructor:

```python
build_signal_strategy("ema_cross", cfg, params={"ema_fast": 5, "ema_slow": 20})
```

### LabConfig params

`top_n`, `lookback`, `skip` can be included in the sweep grid via CLI. Each
unique LabConfig + strategy-params combo is one "combo" in the sweep.

### Combo naming

Each combo gets a deterministic label derived from strategy name + sorted param
key-value pairs: `ema_cross__f5_s20`. This is the group key in the vbt call and
the identifier in the DB.

## Vectorized Signal Generation

### New method: `sweep_signals()`

Signal strategies get a new method that takes the full param grid and returns
stacked entries/exits for all combos at once:

```python
def sweep_signals(
    self,
    combos: list[dict],
    plans_by_combo: dict[str, dict[Timestamp, Plan]],
    data: DataFrame,
) -> dict[str, SignalTargets]:
    """Generate entry/exit signals for ALL param combos in one vectorized pass."""
```

For **EmaCrossSignal**:

1. Extract close prices once for all symbols.
2. Compute `ewm(span=X)` for every unique fast/slow value (deduplicate spans
   across combos — if combos use fast=5,10,20 and slow=20,50,100, compute 6
   EMA series total, not per-combo).
3. For each (fast, slow) combo, crossover is `ema[fast] > ema[slow]` — pure
   boolean indexing, no loops.

For **WfoTournamentSignal**: the IS tournament still runs per-rebalance (it's
the selection step), but we can sweep over `is_fraction` and the candidate EMA
combo list. Signal generation after selection uses the same vectorized EMA cross.

### Weight strategies

No `sweep_signals` — they use existing `to_targets()` per combo, then all weight
matrices are stacked into one `from_orders` call with combo-level grouping.

### Single vbt call

All combos' entries/exits/close get stacked into a single wide DataFrame with
`(combo, symbol)` MultiIndex columns. One
`from_signals(group_by=combo_index, cash_sharing=True)` call produces per-combo
equity curves.

## Persistence & DB Schema

### New table: `lab_sweeps`

```sql
CREATE TABLE IF NOT EXISTS lab_sweeps (
    sweep_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    market TEXT NOT NULL,
    param_grid JSONB NOT NULL,
    n_combos INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### New table: `lab_sweep_combos`

```sql
CREATE TABLE IF NOT EXISTS lab_sweep_combos (
    sweep_id TEXT NOT NULL,
    combo_name TEXT NOT NULL,
    params JSONB NOT NULL,
    metrics JSONB,
    benchmark_metrics JSONB,
    diagnostics JSONB,
    PRIMARY KEY (sweep_id, combo_name)
);
```

### No per-combo time series

Full equity/returns per combo would bloat the DB (50 combos × 1200 days = 60K
rows per sweep). Only summary metrics are persisted per combo. To drill into a
specific combo's equity curve, re-run it as a single `ggt lab` run (cheap since
data is cached).

### Existing tables untouched

`lab_runs`, `lab_plans`, `lab_returns`, `lab_equity`, `lab_summary` stay as-is
for single-run workflows.

## CLI Interface

### Invocation

`--sweep` flag on the existing `ggt lab` command:

```bash
# Sweep with strategy defaults
ggt lab --strategy ema_cross --sweep

# Override specific param ranges
ggt lab --strategy ema_cross --sweep \
    --sweep-param ema_fast=5,10,20 \
    --sweep-param ema_slow=50,100,200

# Sweep LabConfig params too
ggt lab --strategy ema_cross --sweep \
    --sweep-param ema_fast=5,10,20 \
    --sweep-param top_n=20,50
```

Without `--sweep-param` overrides, uses the strategy's `sweep_params()` defaults.
All existing flags (`--eval-start`, `--eval-end`, `--market`, `--max-stocks`)
apply to every combo.

### Output

Ranked table to stdout after the vbt call:

```
Sweep complete: ema_cross | 12 combos | 2021-01-31 → 2026-06-17
sweep_id: sweep_ema_cross_a3f8c1d2

Rank  Combo               Sharpe   CAGR%   MaxDD%   Sortino  TotRet%
───────────────────────────────────────────────────────────────────────
  1   ema_cross__f5_s20     0.42    3.1%   -18.2%     0.61    12.4%
  2   ema_cross__f10_s30    0.38    2.7%   -21.0%     0.54    10.8%
  3   ema_cross__f10_s50    0.31    2.1%   -19.5%     0.43     8.3%
 ...
 12   ema_cross__f50_s200  -0.15   -1.2%   -34.1%    -0.20    -5.1%

SPY baseline: CAGR 18.2% | Sharpe 0.85 | MaxDD -24.5%
```

Sorted by Sharpe descending. SPY baseline anchors interpretation.

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/ggTrader/lab/sweep.py` | Create | Grid generation, vectorized sweep orchestration, CLI table output |
| `src/ggTrader/lab/persist.py` | Modify | Add `lab_sweeps` + `lab_sweep_combos` schema, start/finish/write helpers |
| `src/ggTrader/lab/strategies/signals.py` | Modify | Add `sweep_params()` + `sweep_signals()` to EmaCrossSignal and WfoTournamentSignal |
| `src/ggTrader/lab/strategies/momentum.py` | Modify | Add `sweep_params()` to CrossSectionalMomentum and DualMomentum |
| `src/ggTrader/lab/strategy.py` | Modify | Add `sweep_params` to Strategy protocol |
| `src/ggTrader/lab/cli.py` | Modify | Add `--sweep` and `--sweep-param` flags, wire to sweep module |
| `tests/lab/test_sweep.py` | Create | Unit tests for grid generation, combo naming, vectorized signal stacking, metrics extraction |
