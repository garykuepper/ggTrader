# ggTrader Live Trading Guide

This document explains the live execution pipeline for `ggTrader`, which enables automated trading on Kraken and Alpaca using optimized Walk-Forward Optimization (WFO) parameters.

## 🏗️ Architecture Overview

The live execution engine is designed for reliability, parameter stability, and minimal latency.

- **`CryptoExecutionEngine`**: Handles Kraken/CCXT management and persistent position tracking.
- **`StockExecutionEngine`**: Handles Alpaca management and respects NYSE market hours.
- **Native Orders**: Both engines prioritize exchange-side order management (TSL/OCO) to safeguard positions during local downtime.

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have set up your exchange credentials in the `.env` file at the project root:

```text
# Kraken (Crypto)
KRAKEN_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_api_private_key

# Alpaca (Stocks)
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
```

### 2. Prepare Optimized Parameters

The live bot requires a `run_results.json` file generated from a successful WFO run (e.g., `ggt research`).

### 3. Run a Dry Run

It is **highly recommended** to run the bot in dry-run mode to verify initial signals:

```bash
# Crypto Dry Run
python ggt.py trade --dry-run

# Stock Dry Run (Paper)
python ggt.py trade --asset-class stocks --paper --dry-run
```

### 4. Production Trading

```bash
# Crypto Live
python ggt.py trade --adaptive-sizing

# Stock Paper Trading
python ggt.py trade --asset-class stocks --paper
```

## ⚙️ How It Works

### Interval Alignment
- **Crypto**: Polls Kraken every 4 hours, aligned to UTC boundaries (00:00, 04:00, etc).
- **Stocks**: Evaluates signals once daily after the NYSE market close.

### Per-Coin Signals & Risk Controls
During every cycle:
1. The engine computes signals using WFO-optimized parameters.
2. **Regime Filter**: High-correlation assets are blocked during bear regimes (EMA-based).
3. **Circuit Breaker**: New entries are halted if the intraday drawdown exceeds `DAILY_LOSS_LIMIT_PCT` (default 5%).

## 📁 Persistence & State

Active positions, circuit breaker status, and start-of-day equity are saved to:
- **Crypto**: `data/active_positions.json`
- **Stocks**: `data/active_positions_stocks.json`

## 📊 Observability

### 1. Grafana Dashboard (Recommended)
Accessible at `http://localhost:3002`. Use the **Asset Run** dropdown to switch between `LIVE` (Crypto) and `LIVE_STOCKS`.

### 2. Terminal Dashboard (`ggt dashboard`)
View a local summary of trades and equity curves derived from CSV logs in `data/live/`.

---

## 🛠 Troubleshooting

### Bot Initialization Failures
- **Credential Check**: Ensure keys are set correctly in `.env`.
- **Database Connection**: Verify TimescaleDB is running (check `POSTGRES_CONNECTION_STRING`).
- **Results File**: Ensure the `run_results.json` path provided via `--results` actually exists.

### Research Workers Stalling
- A worker process likely crashed during Numba JIT compilation.
- **Fix**: Check individual worker logs in `results/research_{timestamp}/worker_N.log`.
- **Resources**: Reduce `--workers` if the system is low on RAM.

### Missing Data in Grafana
- **Run ID**: Ensure you've selected the correct ID in the dropdown.
- **Sync Status**: Run `python ggt.py db sync-live` to backfill historical CSV logs into the database.

---
*Back to [README.md](../README.md)*
