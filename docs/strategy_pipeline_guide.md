# Strategy Pipeline Guide

This guide documents the comprehensive end-to-end pipeline for optimizing and validating trading strategies across multiple cryptocurrencies.

## Overview

The **Strategy Pipeline** automates the process of:

1. *(Optional)* Testing parameter sensitivity for each entry strategy (`--sensitivity`)
2. Running per-coin Walk-Forward Optimization (WFO) across all **seven** built-in entry strategies (see `ENTRY_REGISTRY` in `strategies.py`; default exit is **`atr_trailing`** — keep `atr_length` / `atr_multiplier` on each strategy grid)
3. Selecting the best strategy per coin based on robustness
4. Validating results on the full 3-year data range
5. Generating a comprehensive analysis report

This approach respects cryptocurrency volatility diversity by optimizing each coin independently, then combining them into a unified portfolio.

## Quick Start

Run the full pipeline (Phase 1 **omitted** by default; WFO uses **`COARSE_SENSITIVITY_PARAM_GRIDS`**):

```bash
python scripts/run_full_pipeline.py
```

Use **`--detailed-sensitivity`** to widen WFO to **`DETAILED_SENSITIVITY_PARAM_GRIDS`** (still no Phase 1).

To **run Phase 1** (coarse or detailed screen, then **`analyze_sensitivity_results`** narrowing before WFO):

```bash
python scripts/run_full_pipeline.py --sensitivity
# optional: add --detailed-sensitivity for the wide Phase 1 + WFO book
```

To **override the WFO train metric** without editing `run_config` (default is `composite`):

```bash
python scripts/run_full_pipeline.py --train-metric sharpe --max-symbols 5 --no-progress
# choices: sharpe | sortino | calmar | composite
```

Adjust the coarse/detailed dicts in [`src/ggTrader/pipeline/param_grids.py`](src/ggTrader/pipeline/param_grids.py) (`COARSE_ENTRY_PARAM_GRIDS`, `EXIT_AXIS_GRIDS`, etc.) to set the WFO search space when Phase 1 is off. Pipeline defaults (`EXIT_TOURNAMENT`, dates, fees, etc.) live in [`src/ggTrader/utils/run_config.py`](src/ggTrader/utils/run_config.py) (`full_pipeline_config()`), re-exported as `CONSTANTS` in `scripts/run_full_pipeline.py`.

Results are saved to `results/pipeline_<timestamp>/` including:
- `pipeline_report.md`: Comprehensive analysis
- `per_coin_strategy_selection.csv`: Best entry strategy and **exit** per coin
- `per_coin_final_stats.csv`: Performance metrics per coin
- `status.txt`: Timestamped progress log (updated throughout the run)

**Live status:** Open a **second terminal** at the repo root. The pipeline prints exact paths when it starts. Typical options:
- `ggtrader-pipeline-status --watch --interval 10` — reprints the latest `status.txt` every 10 seconds until `COMPLETE` or `FAILED` (default interval is 30 if omitted).
- `python scripts/pipeline_status.py --watch --interval 10` — same without console scripts.
- PowerShell: `Get-Content -Path 'results/pipeline_<timestamp>/status.txt' -Wait -Tail 40` — streams new lines as they are written.

To compare runs across code or config changes, maintain a high-level log in [**Pipeline run history**](pipeline_run_history.md) (manual table plus **default** automated bullets after each successful run; set `GGTRADER_APPEND_RUN_HISTORY=0` to disable when testing).

### Curated universes

Slice files under `data/` share the same JSON shape as `top_25_USD_2023-01-01_2025-12-31.json`. See [`data/universe_slices.md`](../data/universe_slices.md). Examples:

```bash
python scripts/run_full_pipeline.py --no-progress \
  --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --max-symbols 8
python scripts/run_full_pipeline.py --no-progress \
  --symbols-file data/alt_ranks_9_16_usd_2023-01-01_2025-12-31.json --max-symbols 8
```

### Exit tournament (CLI)

**Default** in `full_pipeline_config()` is **`EXIT_TOURNAMENT = ["atr_trailing"]`** (ATR-only). Opt in to both exits:

```bash
python scripts/run_full_pipeline.py --no-progress --max-symbols 5 --dual-exits
```

Explicit overrides (same as editing config):

```bash
python scripts/run_full_pipeline.py --no-progress --max-symbols 5 --exits atr_trailing
python scripts/run_full_pipeline.py --no-progress --max-symbols 5 --exits fixed_sl_tp
python scripts/run_full_pipeline.py --no-progress --max-symbols 5 --exits atr_trailing,fixed_sl_tp
```

Registered exits match `EXIT_REGISTRY` in `strategies.py` (`atr_trailing`, `fixed_sl_tp`).

### Hold-out validation (recommended practice)

The default pipeline runs WFO on rolling folds **and** reports a final backtest on the **full** `START_DATE`–`END_DATE` window. That final window overlaps information used inside folds, so treat headline portfolio metrics as **in-sample** unless you add a stricter protocol.

**Simple two-run hold-out:** (1) Set `END_DATE` in `full_pipeline_config()` to an earlier cutoff \(T\) (e.g. 12 months before your latest data). Run the full pipeline; record chosen strategies and params from `per_coin_strategy_selection.csv`. (2) Restore or extend `END_DATE` through the true end of history and run **final validation only** (or a dedicated script) with **fixed** params—no re-WFO on the tail—or run WFO only through \(T\) and backtest the hold-out segment separately. Document both runs in [Pipeline run history](pipeline_run_history.md).

### Recent validation (Phase 3B, same pipeline run)

After **Phase 2 (WFO)** on the full `START_DATE`–`END_DATE` history and **Phase 3** (full-range combined replay), you can run a **second combined backtest on a recent-only window** with **frozen** WFO params (no re-optimization). Metrics and equal-weight B&H on that window are written to **`pipeline_report.md`** under **Recent validation (frozen WFO params)**.

Configure in `full_pipeline_config()` with `RECENT_VALIDATION_START_DATE` / `RECENT_VALIDATION_END_DATE` (optional; end defaults to **now UTC**), or use CLI:

```bash
python scripts/run_full_pipeline.py --no-progress \
  --recent-validation-start 2025-06-01 --recent-validation-end 2025-12-31
```

- **`--recent-validation-ccxt-tail`**: after the last TimescaleDB bar, append OHLCV from Kraken via CCXT through the validation end (requires network; useful if the DB lags “today”).
- **Standalone rerun** (without repeating WFO): `python scripts/run_validation_backtest.py --run-results results/run_wfo_per_coin_multi_strategy_<ts>/run_results.json --validation-start YYYY-MM-DD [--validation-end DATE] [--ccxt-tail]`

## Pipeline Phases

### Phase 1: Sensitivity Analysis *(opt-in)*

**Not run by default.** Pass **`--sensitivity`** to enable. It screens each entry strategy on a **coarse** parameter grid (or **detailed** with **`--detailed-sensitivity`**). Phase 1 is only meant to **hint where parameters might live** (rough regions), not to find a global optimum. **Too many Cartesian combinations** (× many symbols) can stress memory and runtime; the default coarse book is capped around **~60–80 total grid points** across all strategies.

**Default pipeline:** **Walk-Forward Optimization** (Phase 2) searches **`COARSE_SENSITIVITY_PARAM_GRIDS`** (or detailed) **unchanged**. With **`--sensitivity`**, Phase 2 uses the **narrowed** grid from **`analyze_sensitivity_results`**.

**Why Phase 1 can be slow if grids are large**

- Every grid point is the **full Cartesian product** of all dimensions (e.g. 200+ combos × 3 strategies).
- **Default (fast path):** one **`FastBacktest` run per strategy** with **`USE_VECTORIZED=True`** and a full-grid precompute (indicator sharing + correct **entry × ATR** pairing). Controlled by **`USE_VECTORIZED_SENSITIVITY`** in pipeline config (default `True`). If vectorized sensitivity raises (alignment / memory), the orchestrator **falls back** to the older **chunked non-vectorized** path.
- **Fallback path:** chunked runs with `USE_VECTORIZED=False` and `PARAM_PRODUCT=False` (parallel lists per chunk).
- **`COARSE_SENSITIVITY_PARAM_GRIDS`** is the default WFO book; **`--detailed-sensitivity`** switches WFO (and Phase 1, if enabled) to the wide book.
- **`--sensitivity`:** run Phase 1, then WFO on **narrowed** grids; omit the flag when you already fixed ranges in the coarse/detailed dicts.

**Inputs** *(only when `--sensitivity`)*

- Configured symbols (from `data/top_25_USD_2023-01-01_2025-12-31.json`, truncated by `MAX_SYMBOLS`)
- Full OHLCV window from `CONSTANTS` (`START_DATE` / `END_DATE`)
- `COARSE_SENSITIVITY_PARAM_GRIDS` by default, or `DETAILED_SENSITIVITY_PARAM_GRIDS` with `--detailed-sensitivity`

**Process:**

- For each registered entry strategy (PSAR+ADX, EMA cross, RSI reversal, MACD cross, Bollinger mean-reversion, Donchian breakout, Supertrend flip):
  - Grid search on all parameter combinations for that strategy
  - Train-metric / closed-trade gating as configured (`MIN_CLOSED_TRADES_TRAIN`, etc.)
  - `analyze_sensitivity_results` narrows ranges for WFO

**Outputs:**
- Sensitivity results CSV: `sensitivity_results.csv`
- Narrowed parameter grids (top 20% by Sharpe ratio)
- Sensitivity plots (heatmaps, contour plots)

**Example Output:**
```
Analyzing parameter importance for psar_adx...
  sar_acceleration: 3 unique values -> 1 (top 20%): [0.02]
  sar_maximum: 3 unique values -> 1 (top 20%): [0.2]
  adx_length: 3 unique values -> 2 (top 20%): [14, 20]
  adx_threshold: 4 unique values -> 1 (top 20%): [25]
```

### Phase 2: Per-Coin Multi-Strategy WFO

Runs WFO for each coin across all strategies.

**Inputs:**
- Parameter grids: **coarse/detailed book unchanged** by default; **narrowed** by Phase 1 only if you passed **`--sensitivity`**
- Per-coin OHLCV data (single-symbol subsets)
- WFO configuration: 4 folds, 2:1 train/test ratio

**Process:**
- For each cryptocurrency:
  - For each strategy:
    - Run WFO with rolling time-series cross-validation
    - Train: optimize parameters on in-sample window
    - Test: evaluate on out-of-sample window
    - Compute robustness score (stability across folds)
  - Select best strategy based on highest robustness score

**Outputs:**
- Per-coin strategy selection: `per_coin_strategy_selection.csv`
- WFO statistics: `wfo_results.csv`
- Robustness rankings per strategy

**Example Output:**
```
--- Optimizing BTC ---
  Testing strategy: psar_adx
    psar_adx robustness: 0.8234
  Testing strategy: ema_cross
    ema_cross robustness: 0.7562
  Testing strategy: rsi_reversal
    rsi_reversal robustness: 0.6891
  ✓ BTC: Best Strategy = psar_adx, Robustness = 0.8234
```

### Phase 3: Final Validation Backtest

Runs a single backtest on the full 3-year range using WFO-selected parameters.

**Inputs:**
- Best strategy + params per coin from Phase 2
- Full 3-year OHLCV data

**Process:**
- For each coin, generate signals using its winning strategy + optimized parameters on full range
- Combine all configured symbols' entries/exits into a single portfolio
- Run `vbt.Portfolio.from_signals()` with shared capital (cash_sharing=True)
- Extract final metrics: total return %, **CAGR** (from calendar span of the close index), Sharpe, max drawdown, win rate, total trades
- Compute **benchmark**: equal-weight buy-and-hold (first-bar buy / last-bar sell per symbol) with the same `START_CASH`, `FEES`, `SLIPPAGE`, and bar frequency; report **excess CAGR** (strategy − benchmark)

**Outputs:**
- Per-coin final stats: `per_coin_final_stats.csv`
- Combined portfolio dashboard: `combined_portfolio_final_dashboard`
- Final performance metrics for the report

### Phase 4: Report Generation

Generates a comprehensive markdown report summarizing all results.

**Output:** `pipeline_report.md` with sections:

- **Executive Summary**: Combined portfolio Sharpe, total return, **CAGR**, max DD, and **equal-weight B&H benchmark** (+ excess CAGR)
- **Sensitivity Findings**: Parameter importance per strategy
- **Per-Coin Strategy Selection**: Best strategy per coin with robustness scores
- **Final Performance**: Per-coin metrics from full 3-year backtest
- **Combined Portfolio**: Aggregate metrics across all configured symbols
- **Methodology**: Description of workflow and validation approach

## Strategy Descriptions

### PSAR + ADX (Parabolic SAR + Average Directional Index)

**Entry Logic:**
- Price crosses above Parabolic SAR (reversal from downtrend to uptrend)
- ADX > threshold (confirming strong trend)
- Optional: DM+ > DM- (confirming bullish pressure)

**Parameters:**
- `sar_acceleration`: Initial SAR acceleration (0.02, 0.03)
- `sar_maximum`: Maximum SAR acceleration (0.2, 0.3)
- `adx_length`: Lookback period for ADX (10, 14, 20)
- `adx_threshold`: Minimum ADX to enter trade (15, 20, 25, 30)
- `use_dmp_cross`: Include directional indicator cross condition

**Best For:** Trending markets with clear reversals

### EMA Crossover

**Entry Logic:**
- Fast EMA crosses above slow EMA (bullish crossover)
- Simple trend-following mechanism

**Parameters:**
- `ema_fast`: Fast EMA period (5, 9, 12)
- `ema_slow`: Slow EMA period (21, 26, 34)

**Best For:** Smooth trending markets, less noisy

### RSI Reversal

**Entry Logic:**
- RSI drops below oversold threshold (30, 40)
- Captures mean-reversion at extremes

**Parameters:**
- `rsi_length`: Lookback period for RSI (7, 14, 21)
- `rsi_oversold`: Oversold threshold (20, 30, 40)

**Best For:** Mean-reverting / ranging markets

### Exit Strategies

#### ATR Trailing Stop

Tightens stop loss as price moves favorably, using Average True Range.

**Parameters:**
- `atr_length`: Lookback period for ATR (10, 14, 20)
- `atr_multiplier`: ATR multiple for stop distance (1.5, 2.0, 2.5, 3.0)

#### Fixed Stop/Take Profit

Static percentage-based stops.

**Parameters:**
- `stop_pct`: Stop loss percentage (1, 2, 3)
- `take_profit_pct`: Take profit percentage (3, 5, 8)

## Configuration

Edit `CONSTANTS` in `scripts/run_full_pipeline.py` to customize:

```python
CONSTANTS = {
    "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",  # Data source
    "MAX_SYMBOLS": 5,  # Debug preset (fast); full book: set 20 or `python scripts/run_full_pipeline.py --max-symbols 20`
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",  # Candle interval
    "START_CASH": 1000,  # Initial capital
    "PORTFOLIO_SHARE": 0.10,  # Max ~10% of portfolio per symbol (policy cap)
    "FEES": 0.004,  # Kraken-realistic; do not lower for optimistic backtests
    "SLIPPAGE": 0.003,  # Kraken-realistic slippage assumption
    "N_SPLITS": 4,  # Number of WFO folds
    "TEST_RATIO": 2,  # Train:test ratio (2:1)
    "MIN_TRADES": 0,  # Legacy — no longer used as primary filter (kept for backward compat)
    "MIN_CLOSED_TRADES_TRAIN": 1,  # Min completed round-trips on train window to be eligible
    "REJECT_OPEN_END_IF_CLOSED_LT": 0,  # Optional stricter tier (0 = off)
    "TRAIN_METRIC": "composite",  # sharpe | sortino | calmar | composite (default)
    "TRAIN_METRIC_COMPOSITE_WEIGHTS": {"sharpe": 0.35, "sortino": 0.35, "calmar": 0.30},
    "MAX_TRAIN_DRAWDOWN_PCT": None,  # e.g. 60 → drop combo if train max DD worse than -60%
    "CHUNK_SIZE": 500,  # Parameter combinations per chunk
}
```

### Debug vs full run

| Mode | Command / setting |
|------|-------------------|
| **Debug (default)** | `MAX_SYMBOLS: 5` in `CONSTANTS` — faster pipeline while tuning gates and reports |
| **Full universe** | `--max-symbols 20` or set `MAX_SYMBOLS` to `20` in `CONSTANTS` |
| **Default (no Phase 1)** | WFO uses merged grids from `param_grids.py` (see `build_wfo_superset_grids` + `EXIT_TOURNAMENT`) |
| **Wider WFO book (no Phase 1)** | `--detailed-sensitivity` — WFO uses `DETAILED_SENSITIVITY_PARAM_GRIDS` (much slower) |
| **Run Phase 1 + narrow for WFO** | `--sensitivity` (optional `--detailed-sensitivity` for wide screen) |
| **Disable vectorized sensitivity** | Set `USE_VECTORIZED_SENSITIVITY: False` in `CONSTANTS` (forces chunked fallback) |

### Cost and exit stress (research)

Defaults for `FEES` / `SLIPPAGE` are **Kraken-realistic**—do not reduce them to make equity curves look better. For stress testing, re-run with **higher** fees/slippage (e.g. +25%) via config override or notebook. Widen `atr_multiplier` in `EXIT_AXIS_GRIDS` / `DETAILED_EXIT_AXIS_GRIDS` (or derived `COARSE_SENSITIVITY_PARAM_GRIDS`) when testing wider stops.

### Trade-count filtering

The pipeline gates parameter combo selection on **completed round-trips** (`MIN_CLOSED_TRADES_TRAIN`),
not raw trade count. This targets the root failure mode: a strategy that **buys and never exits**
can produce an inflated Sharpe from a single unrealised drift — `pf.trades.count()` returns 0 for
such a path (open positions at period-end are excluded), so it is automatically disqualified.

| Key | Default | Effect |
|-----|---------|--------|
| `TRAIN_METRIC` | `composite` | Train-fold score for WFO: `composite` (weighted Sharpe + Sortino + Calmar-like), or `sharpe` / `sortino` / `calmar` alone. Tune weights via `TRAIN_METRIC_COMPOSITE_WEIGHTS`. |
| `MIN_CLOSED_TRADES_TRAIN` | `1` | Require ≥ N closed trades on the **train** window before a combo enters train-metric ranking. Set to `0` to disable all filtering. |
| `REJECT_OPEN_END_IF_CLOSED_LT` | `0` (off) | Stricter add-on: also NaN combos still in an open position on the last train bar **and** fewer than N closed trades. Useful when a single lucky round-trip followed by an open hold still inflates Sharpe. |
| `MIN_TRADES` | `0` | **Retired** as primary filter. Kept in config for backward compatibility; only has effect if you manually set it `> 0` and the old code path were re-enabled. |

**When every combo is rejected on a fold:** the log prints a line starting with `All param combos rejected after train gates` plus **Diagnostics**: train bar count and date range, min/max **closed** trade counts per combo (`pf.trades.count()`), how many combos meet `MIN_CLOSED_TRADES_TRAIN`, how many have a finite raw train metric before gating, whether the trade-count index matched the metric index, and a short **hypothesis** (`all_combos_below_MIN_CLOSED_TRADES_TRAIN`, drawdown/open-position gate, etc.). Train window size follows `N_SPLITS` and `TEST_RATIO` in `full_pipeline_config()` (roughly `train_len ≈ TEST_RATIO / (TEST_RATIO + N_SPLITS)` of the series). For extra per-fold **finite** counts after aggregation, set `WFO_DEBUG_METRICS: True` in `CONSTANTS` or run `python scripts/run_full_pipeline.py --wfo-debug-metrics`.

Edit **`COARSE_ENTRY_PARAM_GRIDS`** / **`DETAILED_ENTRY_PARAM_GRIDS`** and **`EXIT_AXIS_GRIDS`** / **`DETAILED_EXIT_AXIS_GRIDS`** in [`src/ggTrader/pipeline/param_grids.py`](src/ggTrader/pipeline/param_grids.py). Derived books **`COARSE_SENSITIVITY_PARAM_GRIDS`** (entry + ATR exit axes) are built in that module for backward compatibility. Choose which exits participate in per-coin WFO with **`CONSTANTS["EXIT_TOURNAMENT"]`** in `run_config` or the **`--exits`** CLI flag on `run_full_pipeline.py`.

After each coin in **Phase 3**, the console prints a line with **Best entry+exit**, **robustness**, **full-range win rate**, and **trade count**.

## How to Add a New Strategy

1. Create a strategy class in `src/ggTrader/indicators/strategies.py`:

```python
class MyStrategy:
    name = "my_strategy"
    param_schema = {
        "param1": [value1, value2],
        "param2": [value3, value4],
    }
    
    def compute_entries(self, precomputer, param_grid):
        """Return (entries_array, param_combos_list)"""
        # Your entry signal logic
        return entries, param_combos
    
    def compute_exits(self, entries, precomputer, param_grid, n_symbols):
        """Return (exits_array, stops_array, price_array)"""
        # Your exit signal logic
        return exits, stops, price_for_orders
```

2. Register in `ENTRY_REGISTRY` or `EXIT_REGISTRY`:

```python
ENTRY_REGISTRY["my_strategy"] = MyStrategy
```

3. Add the entry-only grid to **`COARSE_ENTRY_PARAM_GRIDS`** and **`DETAILED_ENTRY_PARAM_GRIDS`** in `param_grids.py` (keep coarse to a few levels per param). Exit axes stay in **`EXIT_AXIS_GRIDS`** / **`DETAILED_EXIT_AXIS_GRIDS`** per exit type.

4. Run the pipeline to test your new strategy alongside others!

## Interpreting Results

### Sensitivity Report

- **High variance parameters**: Most impactful to strategy performance
- **Narrow ranges**: Recommended values for WFO and final backtest

### Per-Coin Strategy Selection

- **Robustness Score**: Higher = more stable across WFO folds
- **Strategy Diversity**: Different coins may select different strategies

### Final Performance

- **Sharpe Ratio**: Risk-adjusted return. >1 is good, >2 is excellent
- **Max Drawdown**: Largest peak-to-trough decline (in %)
- **Win Rate**: % of trades that were profitable
- **Total Trades**: Diversification across many small trades better than few large

## Troubleshooting

**Q: Pipeline is slow**

A: Reduce `N_SPLITS`, `MAX_SYMBOLS`, or parameter grid sizes. Or increase `CHUNK_SIZE`.

**Q: Low Sharpe ratios**

A: Strategy may not fit this data. Try different parameter ranges or entry/exit strategies.

**Q: Different per-coin strategies**

A: Expected! Crypto volatility varies by coin. Diversity is healthy for risk management.

**Q: One coin dominates portfolio**

A: Adjust `PORTFOLIO_SHARE` to reduce individual position sizes, or use `USE_MOVERS` to filter to top movers only.

## See Also

- [Architecture Guide](architecture.md): Deep dive into technical implementation
- [Analysis Guide](analysis_guide.md): How to run individual sensitivity/WFO workflows
- [Installation](installation.md): Environment setup

---

*For questions or issues, see the [README](../README.md).*
