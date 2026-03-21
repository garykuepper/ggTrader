# Strategy Pipeline Guide

This guide documents the comprehensive end-to-end pipeline for optimizing and validating trading strategies across multiple cryptocurrencies.

## Overview

The **Strategy Pipeline** automates the process of:

1. Testing parameter sensitivity for each entry strategy
2. Running per-coin Walk-Forward Optimization (WFO) across all strategies
3. Selecting the best strategy per coin based on robustness
4. Validating results on the full 3-year data range
5. Generating a comprehensive analysis report

This approach respects cryptocurrency volatility diversity by optimizing each coin independently, then combining them into a unified portfolio.

## Quick Start

Run the full pipeline with:

```bash
python scripts/run_full_pipeline.py
```

Results are saved to `results/pipeline_<timestamp>/` including:
- `pipeline_report.md`: Comprehensive analysis
- `per_coin_strategy_selection.csv`: Best strategy per coin
- `per_coin_final_stats.csv`: Performance metrics per coin

## Pipeline Phases

### Phase 1: Sensitivity Analysis

Tests parameter sensitivity for each entry strategy across expanded ranges.

**Inputs:**
- Top 20 cryptocurrencies (from `data/top_25_USD_2023-01-01_2025-12-31.json`)
- Full 3-year OHLCV data (2023-01-01 to 2025-12-31)
- Expanded parameter grids (defined in `run_full_pipeline.py`)

**Process:**
- For each entry strategy (PSAR+ADX, EMA Crossover, RSI Reversal):
  - Run grid search on all parameter combinations
  - Evaluate Sharpe ratio as the optimization metric
  - Identify parameters with highest impact on performance

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

Runs WFO for each coin across all strategies using narrowed parameter ranges.

**Inputs:**
- Narrowed parameter grids from Phase 1
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
- Combine all 20 coins' entries/exits into a single portfolio
- Run `vbt.Portfolio.from_signals()` with shared capital (cash_sharing=True)
- Extract final metrics: return %, Sharpe, max drawdown, win rate, total trades

**Outputs:**
- Per-coin final stats: `per_coin_final_stats.csv`
- Combined portfolio dashboard: `combined_portfolio_final_dashboard`
- Final performance metrics for the report

### Phase 4: Report Generation

Generates a comprehensive markdown report summarizing all results.

**Output:** `pipeline_report.md` with sections:

- **Executive Summary**: Combined portfolio Sharpe, return, max DD
- **Sensitivity Findings**: Parameter importance per strategy
- **Per-Coin Strategy Selection**: Best strategy per coin with robustness scores
- **Final Performance**: Per-coin metrics from full 3-year backtest
- **Combined Portfolio**: Aggregate metrics across all 20 coins
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
    "MAX_SYMBOLS": 20,  # Number of coins to test
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",  # Candle interval
    "START_CASH": 1000,  # Initial capital
    "PORTFOLIO_SHARE": 0.10,  # Position size as % of capital
    "FEES": 0.004,  # Trading fees
    "SLIPPAGE": 0.003,  # Slippage per trade
    "N_SPLITS": 4,  # Number of WFO folds
    "TEST_RATIO": 2,  # Train:test ratio (2:1)
    "MIN_TRADES": 2,  # Minimum trades to accept result
    "CHUNK_SIZE": 500,  # Parameter combinations per chunk
}
```

Edit parameter grids in `SENSITIVITY_PARAM_GRIDS` to test different ranges:

```python
SENSITIVITY_PARAM_GRIDS = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.02, 0.03],
        "sar_maximum": [0.1, 0.2, 0.3],
        "adx_length": [10, 14, 20],
        "adx_threshold": [15, 20, 25, 30],
        "use_dmp_cross": [True, False],
    },
    # ... other strategies ...
}
```

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

3. Add to pipeline param grid in `run_full_pipeline.py`:

```python
SENSITIVITY_PARAM_GRIDS["my_strategy"] = {
    "param1": [...],
    "param2": [...],
}
```

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
