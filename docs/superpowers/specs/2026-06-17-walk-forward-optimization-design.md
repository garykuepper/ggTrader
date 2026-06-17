# Walk-Forward Optimization for Signal Strategies — Design Spec

## Goal

Add honest out-of-sample (OOS) measurement to the signal strategy sweep path.
Currently `--sweep` runs all param combos over the full eval window — pure
in-sample. This feature adds `--wfo`, which splits the eval period into rolling
train/test folds, selects the best combo per train fold via composite scoring,
simulates only the winner on each test fold, and reports concatenated OOS
metrics.

## Architecture

A new `src/ggTrader/lab/wfo.py` module owns fold generation, composite train
scoring, OOS concatenation, and output formatting. It calls existing
infrastructure — `sweep.py` for grid building, `simulate_signals` for
simulation, `curve_stats` for metrics. No changes to simulate.py, sweep.py,
metrics.py, strategy.py, or persist.py.

CLI gets a `--wfo` flag (mutually exclusive with `--sweep`). Same
`--sweep-param` overrides work for both modes.

## Fold Generation

Rolling fixed-width windows. Each fold:

```
Fold N: Train [train_start, train_end)  Test [test_start, test_end)
        where test_start = train_end (no gap, no lookahead)
```

- **Train window:** 3 years (hardcoded; `TRAIN_YEARS = 3`)
- **Test window:** 1 year (hardcoded; `TEST_YEARS = 1`)
- Window slides forward by `TEST_YEARS` each step
- Stops when `test_end` would exceed `data_end`

With eval_start=2015 and eval_end=2026 → ~8 folds.

`generate_folds` takes `eval_start` and `eval_end` — the boundaries of the
evaluation period. The first train window starts at `eval_start`. OHLCV data
before `eval_start` (the warmup prefix loaded by the CLI) is available for
signal generation but is NOT part of any fold boundary.

```python
Fold = NamedTuple("Fold", [
    ("train_start", pd.Timestamp),
    ("train_end", pd.Timestamp),
    ("test_start", pd.Timestamp),
    ("test_end", pd.Timestamp),
])

def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_years: int = 3,
    test_years: int = 1,
) -> List[Fold]:
```

## Train-Fold Scoring

Each fold simulates all param combos on the train window, then ranks by a
composite metric.

### Composite Score

```
score = 0.5 * norm(sharpe) + 0.3 * norm(sortino) - 0.2 * norm(|max_drawdown|)
```

Where `norm` is min-max scaling across combos within the fold (best = 1.0,
worst = 0.0). This prevents scale differences from letting one metric dominate.

```python
def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
```

Single-combo degenerate case: all normalized values are 0.0 (min == max), score
is 0.0 for each component → score = 0.0. This is fine; the single combo wins
by default.

### Train-Fold Simulation

1. `strategy.sweep_signals(combos, symbols, ohlcv_train)` — generate signals
2. `simulate_signals(targets, prices_train, config)` — one grouped vbt call
3. `curve_stats(equity)` — score each combo over the train window
4. `composite_score` ranks, picks the winner

Stop params (ts_stop, atr_mult/atr_period) are handled identically to the
existing sweep path — `split_params` separates signal from stop params, combos
are grouped by stop config.

## OOS Test-Fold Simulation

For each fold, the train-fold winner is simulated on data up to `test_end`,
but only the test window `[test_start, test_end)` is scored.

Signal generation uses the full data range up to `test_end` so EMAs warm up
properly:

```
Data for signal generation:  [data_start ─────────────── test_end]
Equity scored:                                 [test_start ── test_end]
```

Per-fold equity is normalized for continuity:

- Fold 1: starts at `START_CASH`
- Fold N (N > 1): starts at the ending value of fold N-1's OOS equity

The concatenated OOS curve is scored with `curve_stats` for headline metrics.
SPY baseline is computed over the same concatenated OOS date range.

## CLI Integration

```bash
# WFO with default grid
ggt.py lab --strategy ema_cross --market sp500 --wfo

# WFO with custom stop params
ggt.py lab --strategy ema_cross --market sp500 --wfo \
  --sweep-param "atr_mult=1.5,2.0,3.0" --sweep-param "atr_period=14,20"
```

`--wfo` and `--sweep` are mutually exclusive. `--sweep-param` works identically
for both. Signal strategies only (ema_cross, wfo_tournament); weight strategies
raise an error if `--wfo` is used.

## Output Format

```
WFO: ema_cross | 96 combos x 8 folds | rolling 3yr/1yr

Fold  Train Window       Test Window        Winner                          Train    OOS
────────────────────────────────────────────────────────────────────────────────────────────
1     2015-01 → 2018-01  2018-01 → 2019-01  f10_s200_atr1.5_p14            0.85    0.41
2     2016-01 → 2019-01  2019-01 → 2020-01  f50_s100_atr1.5_p14            0.92    0.63
...
8     2022-01 → 2025-01  2025-01 → 2026-01  f20_s100_atr1.5_p14            0.78    0.55

OOS Aggregate: Sharpe 0.68 | CAGR 9.2% | MaxDD -18.3%
SPY baseline:  Sharpe 0.72 | CAGR 15.0% | MaxDD -24.5%
```

"Train" and "OOS" columns show the composite score for the winning combo on
each window.

## File Layout

### New Files

- `src/ggTrader/lab/wfo.py` (~200 lines)
- `tests/lab/test_wfo.py`

### Modified Files

- `src/ggTrader/lab/cli.py` — add `--wfo` flag, call `run_wfo`

### Unchanged

- `simulate.py`, `sweep.py`, `metrics.py`, `strategy.py`, `persist.py`

## Public API (wfo.py)

```python
Fold = NamedTuple("Fold", [
    ("train_start", pd.Timestamp),
    ("train_end", pd.Timestamp),
    ("test_start", pd.Timestamp),
    ("test_end", pd.Timestamp),
])

def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_years: int = 3,
    test_years: int = 1,
) -> List[Fold]:
    """Rolling fixed-width folds. Slides by test_years each step."""

def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
    """Min-max normalized composite: 0.5*sharpe + 0.3*sortino - 0.2*|maxdd|."""

def run_wfo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> str:
    """Main WFO entry point. Returns formatted results string."""

def format_wfo_table(
    fold_results: List[Dict[str, Any]],
    oos_metrics: Dict[str, float],
    spy_metrics: Dict[str, float],
    strategy_name: str,
    n_combos: int,
    n_folds: int,
) -> str:
    """Render per-fold + aggregate output table."""
```

## Tests

`tests/lab/test_wfo.py`:

- `test_generate_folds_count_and_boundaries` — correct fold dates, no overlap,
  test_start == train_end
- `test_generate_folds_short_data_returns_fewer` — graceful with < 4 years
- `test_composite_score_ranking` — known inputs produce expected ranking
- `test_composite_score_single_combo` — degenerate case
- `test_run_wfo_integration` — small synthetic data, verifies OOS curve
  continuity and output format

## Constraints

- No new dependencies
- No changes to existing simulation or sweep code
- Fully vectorized — each fold's train and test simulation uses the existing
  grouped vbt calls
- Trailing stops (ts_stop, atr_mult/atr_period) work identically to flat sweep
- Weight strategies are not supported (signal strategies only)
