# ggTrader Architecture

This document provides a detailed technical overview of the components and data flow within the `ggTrader` project.

## 🏗️ Core Components

The project is structured into several modular layers, ensuring that logic for data handling, signal generation, and trade execution remains decoupled.

## Data Layer

- **Storage**: PostgreSQL (TimescaleDB) for OHLCV data, optimized for time-series.
- **Ingestion**: `KrakenPostgresIngestor` processes raw CSVs into Hypertable.
- **Access**: `KrakenHistoricalData` facade delegates to `KrakenPostgresReader`.
- **Caching**: `ResultDBManager` (PostgreSQL) stores backtest results, study caches, and optimization metadata.
- **Mover Mask**: `KrakenPostgresReader.get_daily_mover_mask()` builds a boolean DataFrame of daily top-N movers by notional volume in one SQL query.

## Core Engine

- **Backtesting**: `FastBacktest` is the **primary backtest engine**. It wraps `vectorbt.Portfolio.from_signals()` with proper position sizing (`max_position`), shared cash pool (`cash_sharing=True`), and optional dynamic mover masking.
- **Configuration**: `FastBacktest` accepts a `config` dict (CONSTANTS) for portfolio-level settings and a separate `params` dict for signal parameters. Signal parameters support list values for broadcasting parameter grids.
- **Broadcasting**: `SignalFactory` (vectorbt IndicatorFactory) enables running thousands of parameter combinations in a single vectorized operation.
- **Signals**: `signals.py` implements "Golden Source" logic (PSAR, ADX, ATR Trailing Stop) using Numba and VectorBT.

## Strategy Architecture

The project uses a **pluggable strategy framework** for flexible signal generation across multiple entry and exit strategies.

### Strategy Registry

- **Entry Strategies**: `ENTRY_REGISTRY` maps strategy names to classes:
  - `psar_adx`: Parabolic SAR + ADX momentum detection
  - `ema_cross`: EMA crossover (fast > slow)
  - `rsi_reversal`: RSI oversold reversal
  - `macd_cross`: MACD line crosses above signal
  - `bbands_mean_reversion`: Close crosses up through lower Bollinger band
  - `donchian_breakout`: Close breaks above prior Donchian upper band
  - `supertrend_flip`: Supertrend direction flips bullish
  - Custom strategies can be added to `ggTrader/indicators/strategies.py`

- **Exit Strategies**: `EXIT_REGISTRY` maps exit classes:
  - `atr_trailing`: ATR-based trailing stop
  - `fixed_sl_tp`: Fixed percentage stop/take-profit

### Indicator Pre-computation

`IndicatorPrecomputer` optimizes performance by:

- Pre-computing each indicator (PSAR, ADX, ATR, EMA, RSI, MACD, BBands, Donchian, Supertrend) once across full parameter ranges
- Caching results to avoid redundant calculations
- Enabling numpy broadcasting for efficient parameter grid evaluation

### Vectorized Signal Generation

`_generate_signals_vectorized()` in `FastBacktest` dispatches to the strategy registry:

1. Reads `config["ENTRY_STRATEGY"]` and `config["EXIT_STRATEGY"]`
2. Instantiates the strategy classes via `get_entry_strategy()` / `get_exit_strategy()`
3. Calls `compute_entries()` and `compute_exits()` which return numpy arrays
4. Wraps arrays in DataFrames with proper MultiIndex columns for VectorBT

### Per-Coin Walk-Forward Optimization

`run_wfo_per_coin_orchestrator()` extends WFO with per-coin independence:

1. Loads full 3-year OHLCV data once
2. For each symbol:
   - Runs standard WFO (rolling train/test folds) with the narrowed parameter grid
   - Selects best strategy based on robustness score (in-sample metric stability)
3. Combines results: uses the winning strategy + params for each coin
4. Runs final validation backtest on full 3-year range for each coin
5. Merges per-coin signals into single combined portfolio with shared cash

This approach respects the volatility diversity of individual cryptocurrencies while maintaining a unified portfolio structure.

### 📊 Optimization Model (WFO)

The system uses a **6-Fold Sliding Window** to ensure that all optimized parameters are robust across multiple market regimes.

```mermaid
gantt
    title Walk-Forward Folds (Sliding 3-Year Window)
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m
    
    section Fold 1
    Train (80%) :active, f1_train, 2023-01-01, 2025-06-01
    Test (20%)  :crit, f1_test, 2025-06-01, 2025-10-01
    
    section Fold 2
    Train (80%) :active, f2_train, 2023-03-01, 2025-08-01
    Test (20%)  :crit, f2_test, 2025-08-01, 2025-12-01
    
    section Fold 3
    Train (80%) :active, f3_train, 2023-05-01, 2025-10-01
    Test (20%)  :crit, f3_test, 2025-10-01, 2026-02-01
    
    section Fold 4
    Train (80%) :active, f4_train, 2023-07-01, 2025-12-01
    Test (20%)  :crit, f4_test, 2025-12-01, 2026-04-01

    section Fold 5
    Train (80%) :active, f5_train, 2023-09-01, 2026-02-01
    Test (20%)  :crit, f5_test, 2026-02-01, 2026-06-01

    section Fold 6
    Train (80%) :active, f6_train, 2023-11-01, 2026-04-01
    Test (20%)  :crit, f6_test, 2026-04-01, 2026-08-01
```

## Workflows

The system is controlled via the unified `ggt` CLI. For a detailed command reference, see [**Unified CLI Guide**](UNIFIED_PIPELINE.md).

1. **Research**: `python ggt.py research` — Orchestrates parallel WFO across the liquid universe.
2. **Backtest**: `python ggt.py backtest` — Replays results for validation.
3. **Database**: `python ggt.py db` — Manages TimescaleDB health and maintenance.
4. **Ingest**: `python ggt.py ingest` — Syncs historical data from Kraken.

For a concise WFO → backtest walkthrough, see [**Strategy Execution**](UNIFIED_PIPELINE.md).

## Legacy Modules

- **Trading Engine** (`trading.py`): Iterative day-by-day simulation. Retained for live trading execution only.
- **Archive**: All legacy standalone scripts have been moved to `docs/archive/` or deleted in favor of the unified `ggt` CLI.

## 🔄 Data Flow

```mermaid
graph TD
    A[Exchange API / Parquet] -->|Fetch| B(Data Layer)
    B -->|OHLCV Data| C["Strategy Registry<br/>Entry/Exit Classes"]
    B -->|Daily Mover Mask| D[FastBacktest Engine]
    C -->|IndicatorPrecomputer| E["Vectorized Signal Gen<br/>compute_entries/exits"]
    E -->|Entries / Exits| D
    D -->|vbt.Portfolio| F(Results Manager)
    F -->|Storage| G[TimescaleDB / results/ folder]
    H["Optimization Scripts<br/>Sensitivity/WFO/Pipeline"] -->|Loop| D
    G -->|Visualization| I[Jupyter Notebooks]
    J[Live Exchange] <-->|Rest/WS| K[ExecutionEngine]
    K -->|Load Params| G
    B -->|Live OHLCV| K
    K -->|Orders| J
```

## 📊 Result Management

Every run (backtest, sensitivity, or WFO) outputs results to a timestamped folder in the `results/` directory. This includes:

- **Trades**: Detailed log of every entry and exit.
- **Metrics**: Sharpe ratio, Max Drawdown, Profit Factor, etc.
- **Config**: A snapshot of the parameters used for that specific run.

## 🚀 Live Execution

- **`ExecutionEngine`**: Orchestrates live trading by fetching recent candles via `LiveExchangeLoader`, computing per-coin signals using WFO-optimized parameters, and managing Kraken orders.
- **Bot Persistence**: Tracks active positions in `data/active_positions.json` to handle process restarts.
- **Native Trailing Stop**: Leverages Kraken's `trailing-stop` order type for server-side risk management.

---
*Back to [README.md](../readme.md)*
