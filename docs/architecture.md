# ggTrader Architecture

This document provides a detailed technical overview of the components and data flow within the `ggTrader` project.

## 🏗️ Core Components

The project is structured into modular layers, ensuring that research and execution logic remain independent of the underlying asset class.

- **Data Layer**:
    - `TimescaleDBLoader`: Primary interface for reading historical OHLCV data.
    - `CachedExchangeLoader`: CCXT-based live data fetching with automatic DB caching (Crypto).
    - `CachedYFinanceLoader`: yfinance-based live data fetching with automatic DB caching (Stocks).
- **Strategy Layer**: Modular entry and exit strategies applied via `IndicatorPrecomputer` for high-performance vectorized backtesting.
- **WFO Engine**: Multi-fold Walk-Forward Optimization that selects robust parameters by blending In-Sample and Out-of-Sample performance.
- **Execution Engine**:
    - `BaseExecutionEngine`: Shared abstract base for risk management (Circuit Breaker), state persistence, and notification routing.
    - `CryptoExecutionEngine`: Specialized logic for Kraken/CCXT market and OCO orders.
    - `StockExecutionEngine`: Specialized logic for Alpaca limit orders and NYSE market hours.
- **Observability**: Real-time mirror to TimescaleDB for visualization in Grafana.

## Data Layer

- **Storage**: PostgreSQL (TimescaleDB) for OHLCV data, optimized for time-series.
- **Ingestion**: `KrakenPostgresIngestor` processes raw CSVs into Hypertable.
- **Access**: `KrakenHistoricalData` facade delegates to `KrakenPostgresReader`.
- **Caching**: `ResultDBManager` (PostgreSQL) stores backtest results, study caches, and optimization metadata.
- **Mover Mask**: `KrakenPostgresReader.get_daily_mover_mask()` builds a boolean DataFrame of daily top-N movers by notional volume in one SQL query.

## Core Engine

The `ggTrader.core` package is split into seven focused modules:

| Module | Responsibility |
| ------ | -------------- |
| `fast_backtest.py` | `FastBacktest` engine — wraps `vbt.Portfolio.from_signals()` with position sizing, shared cash, and mover masking |
| `orchestrator.py` | Public API (`run_backtest_orchestrator`, `run_frozen_params_combined_backtest`, `run_multi_strategy_per_coin_wfo`) + regime/allocation helpers |
| `orchestrator_utils.py` | Pure utility helpers (param coercion, ETA strings, logging) shared across orchestration layers |
| `sensitivity.py` | Grid search orchestration — vectorized and chunked paths |
| `wfo.py` | Walk-forward optimization loop (`run_wfo_orchestrator`, fold calculations, robustness scoring) |
| `metrics.py` | Train-metric extraction (Sharpe/Sortino/Calmar), trade-count gates, sensitivity result filtering |
| `regime_filtering.py` | Three-tier BTC-correlation regime filter (BTC regime, altcoin index, exempt) |
| `benchmarking.py` | Buy-and-hold benchmarks (BTC, S&P 500), CAGR helpers, SPY parquet cache |

- **Backtesting**: `FastBacktest` accepts a `config` dict (CONSTANTS) for portfolio-level settings and a separate `params` dict for signal parameters. Signal parameters support list values for broadcasting parameter grids.
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
  - `fixed_sl_tp`: Fixed percentage stop-loss / take-profit
  - `trailing_stop`: Percentage-based trailing stop

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

### Three-Tier Regime Filter

Before combining per-coin signals into the final portfolio, `run_frozen_params_combined_backtest()` applies a BTC-correlation regime filter that mutes entries during bear markets:

| Tier | Condition | Filter applied |
| ---- | --------- | -------------- |
| **BTC tier** | BTC return correlation ≥ `BTC_REGIME_FILTER_MIN_CORRELATION` (default 0.5) | Signal blocked when BTC EMA(50) < EMA(200) — golden cross filter |
| **Altcoin tier** | Correlation in `[ALTCOIN_REGIME_FILTER_CORR_MIN, btc_min)` (default 0.3–0.5) | Signal blocked when altcoin equal-weight index EMA(50) < EMA(200) |
| **Exempt tier** | Correlation < `ALTCOIN_REGIME_FILTER_CORR_MIN` | No filter — coin trades freely |

The filter compares a short EMA to the long EMA(200) rather than raw close price, which prevents single-candle spikes from flipping the regime signal. The short span is configured via `BTC_REGIME_FILTER_SHORT_EMA` (default `50`). Set to `None` to revert to the original `close > EMA(200)` behaviour.

The **altcoin index** is an equal-weighted, normalised price series built from all non-BTC symbols in the universe. Its EMA(50)/EMA(200) cross acts as a trend proxy for alt-correlated coins.

Correlations are computed over the full OHLCV date range using daily log-returns. Regime filtering is enabled by `BTC_REGIME_FILTER=True` in constants; it is disabled by default.

```python
_compute_btc_correlations()   # -> Dict[str, float]
_compute_btc_regime_mask()    # -> pd.Series[bool]  (True = bullish)
_compute_altcoin_index_mask() # -> pd.Series[bool]  (True = bullish)
_apply_tiered_regime_mask()   # -> filtered entries DataFrame
```

### Quality Gates (Anti-Overfitting)

After WFO, each coin passes through sequential gates before entering the combined portfolio:

| Gate | Config key | Default | Purpose |
| ---- | ---------- | ------- | ------- |
| Robustness floor | `MIN_ROBUSTNESS_SCORE` | 0.1 | Drop coins with very low OOS robustness |
| Fold consistency | `MIN_FOLD_CONSISTENCY` | 0.38 | At least 4 in 10 OOS folds must be profitable |
| Valid train folds | `MIN_VALID_TRAIN_FOLDS` | 3 | At least 3 of 6 folds must produce finite IS Sharpe |
| Strategy diversity | `MAX_COINS_PER_STRATEGY` | 10 | Prevent one entry strategy from dominating |

### 🛑 Risk Controls (Live)

Live trading incorporates additional real-time risk safeguards:

- **Daily Loss Circuit Breaker**: Halt all *new* entries for the day if the portfolio's intraday drawdown exceeds `DAILY_LOSS_LIMIT_PCT` (default 5%).
- **Regime Filter**: Blocks entries during sustained bear markets (BTC EMA-based).
- **Exchange Reconciliation**: Syncs local state with actual Kraken holdings on every heartbeat to detect server-side exits (TSL/OCO).

Coins that pass all gates but produce **0 regime-filtered trade signals** in the combined backtest have their OOS allocation weight zeroed out, so idle capital is redistributed to active coins.

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

### 📊 Optimization Model (Sliding WFO)

The system uses a **10-Fold Sliding Window** where each fold moves forward by the exact length of the test period (**Step = Test Length**). This ensures that every data point eventually serves as an "unseen" test bar, providing 10 granular OOS data points for robustness gating.

```mermaid
gantt
    title Walk-Forward Folds (10 Folds, Step = Test Length)
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m
    
    section Fold 1
    Train       :active, f1_tr, 2023-01-01, 2024-05-15
    Test        :crit, f1_ts, 2024-05-15, 2024-07-15
    
    section Fold 2
    Train       :active, f2_tr, 2023-03-01, 2024-07-15
    Test        :crit, f2_ts, 2024-07-15, 2024-09-15
    
    section Fold 3
    Train       :active, f3_tr, 2023-05-01, 2024-09-15
    Test        :crit, f3_ts, 2024-09-15, 2024-11-15
    
    section Fold 4
    Train       :active, f4_tr, 2023-07-01, 2024-11-15
    Test        :crit, f4_ts, 2024-11-15, 2025-01-15

    section Fold 5
    Train       :active, f5_tr, 2023-09-01, 2025-01-15
    Test        :crit, f5_ts, 2025-01-15, 2025-03-15

    section Fold 6
    Train       :active, f6_tr, 2023-11-01, 2025-03-15
    Test        :crit, f6_ts, 2025-03-15, 2025-05-15

    section Fold 7
    Train       :active, f7_tr, 2024-01-01, 2025-05-15
    Test        :crit, f7_ts, 2025-05-15, 2025-07-15

    section Fold 8
    Train       :active, f8_tr, 2024-03-01, 2025-07-15
    Test        :crit, f8_ts, 2025-07-15, 2025-09-15

    section Fold 9
    Train       :active, f9_tr, 2024-05-01, 2025-09-15
    Test        :crit, f9_ts, 2025-09-15, 2025-11-15

    section Fold 10
    Train       :active, f10_tr, 2024-07-01, 2025-11-15
    Test        :crit, f10_ts, 2025-11-15, 2026-01-15
```

## Workflows

The system is controlled via the unified `ggt` CLI. For a detailed command reference, see [**Unified CLI Guide**](unified_pipeline.md).

1. **Research**: `python ggt.py research` — Orchestrates parallel WFO across the liquid universe.
2. **Backtest**: `python ggt.py backtest` — Replays results for validation.
3. **Database**: `python ggt.py db` — Manages TimescaleDB health and maintenance.
4. **Ingest**: `python ggt.py ingest` — Syncs historical data from Kraken.

For a concise WFO → backtest walkthrough, see [**Strategy Execution**](unified_pipeline.md).

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
    K -->|Mirror| G
    G -->|Query| L[Grafana Dashboard]
```

## 📊 Result Management

Every run (backtest, sensitivity, or WFO) outputs results to a timestamped folder in the `results/` directory.

### Database Mirroring (Live)
In addition to CSV logs, the `ExecutionEngine` mirrors all live trading events to **TimescaleDB** in real-time. This enables high-performance monitoring via the **Grafana Dashboard**.
- **`orders`**: Every buy/sell request sent to exchange.
- **`trades`**: Completed round-trips with PnL.
- **`equity_curves`**: Periodic balance snapshots for the `LIVE` run_id.

## 🚀 Live Execution

- **`CryptoExecutionEngine`**: Orchestrates live crypto trading by fetching recent candles via `CachedExchangeLoader` and managing Kraken orders.
- **`StockExecutionEngine`**: Orchestrates live stock trading via `CachedYFinanceLoader` and Alpaca, respecting NYSE market hours.
- **Bot Persistence**: Tracks active positions and circuit breaker status in `data/active_positions.json` (Crypto) and `data/active_positions_stocks.json` (Stocks) to handle process restarts.
- **Native Exits**: Leverages exchange-native `trailing-stop` and `OCO` order types for server-side risk management.

---
*Back to [README.md](../README.md)*
