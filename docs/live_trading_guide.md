# ggTrader Live Trading Guide

This document explains the live execution pipeline for `ggTrader`, which enables automated trading on Kraken using optimized Walk-Forward Optimization (WFO) parameters.

## 🏗️ Architecture Overview

The live execution engine is designed for reliability, parameter stability, and minimal latency.

- **`ExecutionEngine`**: The core component that handles coin-specific signal generation, CCXT exchange management, and persistent position tracking.
- **`run_live_trader.py`**: An orchestration script providing a CLI entry point for starting the bot.
- **Kraken Native Orders**: The bot uses Kraken's built-in `trailing-stop` order types for robust exit management, reducing the need for continuous bot activity to manage open trades.

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have set up your exchange credentials in the `.env` file at the project root:

```text
KRAKEN_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_api_private_key
```

### 2. Prepare Optimized Parameters

The live bot requires a `run_results.json` file generated from a successful WFO run (e.g., `ggt research`).

### 3. Run a Dry Run

It is **highly recommended** to run the bot in dry-run mode for at least one 4h interval to verify initial signals:

```bash
python ggt.py trade --dry-run
```

### 4. Live Trading (Crypto)

To start real trading with adaptive position sizing and the daily loss circuit breaker:

```bash
python ggt.py trade --adaptive-sizing
```

### 5. Paper Trading (Stocks)

To start stock paper trading on Alpaca (uses NYSE market hours):

```bash
python ggt.py trade --asset-class stocks --paper
```

## ⚙️ How It Works

### Interval Alignment

The bot polls Kraken for new OHLCV data every 4 hours. It aligns itself with the standard candle closes (00:00, 04:00, 08:00... UTC) by calculating the seconds to the next boundary and sleeping until then.

### Per-Coin Signals & Regime Filters

During every poll cycle:

1. The engine identifies the optimal strategy for each coin based on the provided results file.
2. It generates a buy/sell signal for the latest completed candle.
3. **Regime Filter**: Coins with high BTC correlation are blocked if the overall market is in a bear regime (EMA-based).
4. **Circuit Breaker**: If your intraday portfolio value drops by more than `DAILY_LOSS_LIMIT_PCT` (default 5%), new entries are halted until the next day.

### Automated Exits (TSL/OCO)

Upon a successful entry fill:

1. The engine immediately places a **Trailing Stop Loss** or **OCO** order on Kraken.
2. Kraken takes over management of the exit, safeguarding the position even if the bot is temporarily disconnected.

## 📁 Persistence & State

Active positions, circuit breaker status, and start-of-day equity are saved to:
- **Crypto**: `data/active_positions.json`
- **Stocks**: `data/active_positions_stocks.json`

Upon restart, the engines reload their respective states to avoid double-entry or losing track of open trades.

## ⚠️ Important Considerations

- **Order Sizes**: Ensure your `--capital` allocation meets Kraken's minimum order size requirements for the selected symbols.
- **API Limits**: The engine uses conservative polling (once every 4 hours) to avoid rate-limiting issues.
- **Reliability**: While Kraken handles the TSL, the bot must be running during candle closes to detect new entries.

## 🐳 Docker Deployment

To run the live bot in a continuous container (recommended for production):

1. **Build and Start**:
   Update the `command` in `docker-compose.yaml` with your correct `--results` path, then run:

   ```bash
   docker compose up -d ggtrader_live
   ```

2. **Check Logs**:
   To monitor the 4h polling cycles:
 
   ```bash
   docker compose logs -f ggtrader_live
   ```

3. **Persistence**:
   The container mounts `./data` locally, ensuring `active_positions.json` persists even if the container is rebuilt or moved to another host.

## Performance Tracking & Observability

ggTrader provides two ways to monitor your live performance: a terminal-based dashboard and a real-time Grafana dashboard.

### 1. Grafana Dashboard (Recommended)

Accessible at `http://localhost:3002`, this provides:
- **Live Equity Curve**: Your total portfolio value updated every balance snapshot.
- **PnL per Trade**: Color-coded points (Gains vs. Losses) for every closed position.
- **Trade History**: A searchable table of your most recent fills.

The Grafana instance pulls directly from the **TimescaleDB** mirror where all live events are recorded in real-time.

### 2. Terminal Dashboard (`ggt dashboard`)

The live trader also logs every trade, balance snapshot, and round-trip P&L to local CSV files in `data/live/`. You can view a summary with:

```bash
# Print summary and generate interactive Plotly charts
python ggt.py dashboard

# Sync historical trades from Kraken first (recommended on first run)
python ggt.py dashboard --sync
```

### Data Synchronization

If you ever lose local state or want to backfill historical trades into the database for Grafana:

```bash
# Backfill CSV logs into the database mirror
python ggt.py db sync-live

# Re-sync CSV logs from Kraken trade history
python ggt.py dashboard --sync
```
