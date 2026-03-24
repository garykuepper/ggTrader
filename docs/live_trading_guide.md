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

The live bot requires a `run_results.json` file generated from a successful WFO run (e.g., `run_walk_forward_optimization.py` or `run_full_pipeline.py`).

### 3. Run a Dry Run

It is **highly recommended** to run the bot in dry-run mode for at least one 4h interval to verify initial signals:

```bash
python scripts/run_live_trader.py --results results/your_results_folder/run_results.json --dry-run
# or: ggtrader-live --results ... --dry-run
```

### 4. Live Trading

To start real trading with a specified capital per trade:

```bash
ggtrader-live --results results/your_results_folder/run_results.json --capital 25.0
```

## ⚙️ How It Works

### Interval Alignment

The bot polls Kraken for new OHLCV data every 4 hours. It aligns itself with the standard candle closes (00:00, 04:00, 08:00... UTC) by calculating the seconds to the next boundary and sleeping until then.

### Per-Coin Signals

During every poll cycle:

1. The engine identifies the optimal strategy for each coin (e.g., Ethereum uses `rsi_reversal` while Bitcoin uses `ema_cross`) based on the provided results file.
2. It generates a buy/sell signal for the latest completed candle.
3. If an entry signal is detected, the bot executes a **Market Buy** on Kraken.

### Automated Exits (TSL)

Upon a successful entry fill:

1. The engine immediately places a **Trailing Stop Loss** order on Kraken for the same amount.
2. The trailing setpoint is derived from the WFO-optimized `stop_pct`.
3. Kraken takes over management of the TSL, safeguarding the position even if the bot is temporarily disconnected.

## 📁 Persistence & State

Active positions and order IDs are saved to `data/active_positions.json`. Upon restart, the engine reloads this state to avoid double-entry or losing track of open trades.

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
