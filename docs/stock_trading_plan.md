# Stock Research & Trading Implementation Plan

This document describes how to extend ggTrader to research and trade US equities alongside its existing crypto capabilities. The goal is to reuse the asset-agnostic core (VectorBT backtesting, WFO, strategies, metrics, portfolio optimization) and add stock-specific modules for data, execution, regime filtering, and universe selection.

**Data Source:** yfinance (free, no API key, data back to 1980+, full market volume)  
**Broker:** Alpaca (execution only -- keys already configured in `.env`)  
**Universe:** S&P 500 stocks ranked by average daily dollar volume  
**Default Timeframe:** Daily (1d) bars  
**Scope:** Full pipeline -- research, production, and live trading  
**Execution Style:** Limit buy entries + trailing stop loss exits via Alpaca  

---

## Table of Contents

1. [Architecture Approach](#1-architecture-approach)
2. [File Map](#2-file-map)
3. [Step 1: Dependencies & Configuration](#step-1-dependencies--configuration)
4. [Step 2: Data Layer (YFinanceDataLoader)](#step-2-data-layer-yfinancedataloader)
5. [Step 3: Universe Selection](#step-3-universe-selection)
6. [Step 4: Regime Filtering](#step-4-regime-filtering)
7. [Step 5: Pipeline Integration (CLI + Setup)](#step-5-pipeline-integration-cli--setup)
8. [Step 6: Execution Engine (Multi-Asset Refactor)](#step-6-execution-engine-multi-asset-refactor)
9. [Step 7: Verification & Testing](#step-7-verification--testing)
10. [Key Design Decisions](#key-design-decisions)

---

## 1. Architecture Approach

**Parallel asset-class modules with shared Base logic.**

The crypto and stock domains differ in market hours, APIs, and regime proxies, but share core operational logic. We will use a shared **BaseExecutionEngine** to minimize duplication of risk and state management.

| Concern | Crypto | Stocks |
|---------|--------|--------|
| Market hours | 24/7 | 9:30 AM - 4:00 PM ET, holidays |
| Data API | CCXT (Kraken) | yfinance (daily OHLCV) |
| Execution API | CCXT market/OCO orders | Alpaca limit buy + trailing stop |
| Regime proxy | BTC EMA(50) vs EMA(200) | SPY EMA(50) vs EMA(200), VIX |
| Symbol format | `BTC-USD`, `ETH-USD` | `AAPL`, `MSFT` |
| Fees | 0.1% (Kraken) | 0% (Alpaca commission-free) |
| Polling cadence | Every 4h (aligned to candle close) | Once daily after market close |

**Shared Infrastructure:**
- `BaseDataLoader` for data (`src/ggTrader/data/core/base_loader.py`)
- `BaseExecutionEngine` for circuit breakers, state persistence, and notification routing.
- OHLCV MultiIndex DataFrames for all research/backtesting.
- TimescaleDB for shared OHLCV and results storage.

---

## 2. File Map

### New Files

| File | Purpose |
|------|---------|
| `src/ggTrader/data/live/yfinance_loader.py` | `YFinanceDataLoader(BaseDataLoader)` -- fetches daily OHLCV via yfinance |
| `src/ggTrader/data/live/cached_yfinance_loader.py` | `CachedYFinanceLoader` -- DB-first + yfinance fallback |
| `src/ggTrader/data/core/stock_constants.py` | Market hours constants, S&P 500 constituent list |
| `src/ggTrader/core/base_execution_engine.py` | **NEW BASE CLASS** -- Circuit breaker, state saving, notification logic |
| `src/ggTrader/core/stock_regime_filtering.py` | SPY EMA + VIX-based regime filters |
| `src/ggTrader/core/stock_execution_engine.py` | Alpaca-based live trading engine (inherits from Base) |
| `src/ggTrader/core/market_hours.py` | Market hours utilities via Alpaca clock API |
| `scripts/update_universe_stocks.py` | S&P 500 volume-ranked universe builder |

### Modified / Refactored Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `alpaca-py` to dependencies |
| `src/ggTrader/core/crypto_execution_engine.py` | **RENAME** from `execution_engine.py`. Refactored to inherit from Base. |
| `src/ggTrader/utils/run_config.py` | Add `stock_pipeline_config()` |
| `src/ggTrader/utils/setup.py` | Branch `load_data_and_setup()` on `config["ASSET_CLASS"]` |
| `src/ggTrader/cli/cmd_trade.py` | Add `--asset-class` flag, instantiate appropriate engine |
| `src/ggTrader/core/benchmarking.py` | Skip BTC benchmark when `ASSET_CLASS == "stocks"` |

---

## Step 1: Dependencies & Configuration

*(Same as original plan: add alpaca-py, Alpaca credentials helper, and stock pipeline config)*

---

## Step 2: Data Layer (YFinanceDataLoader)

*(Same as original plan: YFinanceDataLoader and CachedYFinanceLoader implementations)*

---

## Step 3: Universe Selection

*(Same as original plan: scripts/update_universe_stocks.py)*

---

## Step 4: Regime Filtering

*(Same as original plan: SPY + VIX regime filters)*

---

## Step 5: Pipeline Integration (CLI + Setup)

*(Same as original plan: --asset-class flags in CLI, setup.py branching)*

---

## Step 6: Execution Engine (Multi-Asset Refactor)

We will perform a surgical refactor to extract shared logic into a base class.

### 6a. BaseExecutionEngine

**File:** `src/ggTrader/core/base_execution_engine.py` (NEW)

Extract the following from the current `execution_engine.py`:
- `__init__`: Common config, logger, TradeTracker initialization, notifiers.
- `load_state` / `save_state`: Persistence logic (with asset-specific filename support).
- `_notify`: Common alert routing.
- **Circuit Breaker**: The intraday drawdown check logic.
- Abstract methods: `run_event_loop()`, `_execute_trade_logic()`, `_get_total_portfolio_usd()`.

### 6b. CryptoExecutionEngine

**File:** `src/ggTrader/core/crypto_execution_engine.py` (RENAME)

- Inherits from `BaseExecutionEngine`.
- Focuses exclusively on CCXT/Kraken order types (OCO, Trailing Stop offsets).
- Maintains the 4-hour polling loop.

### 6c. StockExecutionEngine

**File:** `src/ggTrader/core/stock_execution_engine.py` (NEW)

- Inherits from `BaseExecutionEngine`.
- Uses Alpaca `TradingClient` and `LimitOrderRequest`.
- Implements the daily evaluation loop (triggered after NYSE close).

---

## Step 7: Verification & Testing

*(Same as original plan + new tests for Base class and Crypto regression)*

---

## Key Design Decisions

### Shared Base Class vs. Parallel Modules?
The addition of the **Daily Loss Circuit Breaker** and **State Persistence** (both which are identical across assets) makes a shared Base class necessary to avoid bug-prone code duplication. However, the specific "order plumbing" (CCXT vs Alpaca) remains isolated in leaf classes.

### Separate State Files?
Yes. Crypto will continue to use `data/active_positions.json`. Stocks will use `data/active_positions_stocks.json`. This allows running both engines simultaneously on the same host without file locks or state contamination.

---

*Back to [README.md](../README.md)*
