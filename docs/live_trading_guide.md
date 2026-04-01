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

## Performance Tracking & Dashboard

The live trader automatically logs every trade, balance snapshot, and round-trip P&L to local CSV files. This provides a local mirror of your Kraken trading activity with full fee tracking and performance metrics.

### How It Works

The `TradeTracker` class (`src/ggTrader/core/trade_tracker.py`) is integrated into the `ExecutionEngine` and records data at four points:

1. **Buy fills** — After every successful market buy, the fill price, amount, and Kraken fee are logged.
2. **Strategy-signal sells** — When a `fixed_sl_tp` exit signal fires and the bot executes a market sell, the round-trip P&L is computed and recorded.
3. **Reconciliation-detected closes** — When the bot detects that a Kraken-side trailing stop or OCO order has filled (position held locally but no longer on exchange), it fetches the exit order details and records the close.
4. **Balance snapshots** — Every 4h polling cycle, the bot snapshots total account value, free USD, and number of open positions.

### Data Files

All tracking data is stored in `data/live/` (git-ignored, never pushed to GitHub):

| File | Contents |
|------|----------|
| `data/live/trade_log.csv` | Every executed order (buys and sells) with price, amount, fee |
| `data/live/position_closes.csv` | Completed round-trips with entry/exit prices, gross/net P&L, fees, hold duration, exit reason |
| `data/live/balance_snapshots.csv` | Periodic account value snapshots (total USD, free USD, positions USD) |
| `data/live/dashboard/` | Generated HTML/PNG charts |

### Using the Dashboard

View your live trading performance with the `ggt dashboard` command:

```bash
# Print summary and generate interactive charts
ggt dashboard

# Sync historical trades from Kraken first (recommended on first run)
ggt dashboard --sync

# Sync trades from a specific start date
ggt dashboard --since 2025-06-01

# Text summary only, no charts
ggt dashboard --no-plots

# Save charts to a custom directory
ggt dashboard --output /path/to/charts
```

From Docker:

```bash
docker compose exec ggtrader_live python ggt.py dashboard --sync
```

### Console Output

The dashboard prints a formatted summary to the terminal:

```
====================================================
  ggTrader Live Performance Dashboard
====================================================
  Period:          2025-06-01 -> 2026-04-01
  Account Value:   $1,247.83

  --- P&L ---
  Gross P&L:       +$266.25
  Total Fees:      $18.42
  Net P&L:         +$247.83

  --- Trades ---
  Total Trades:    47
  Win Rate:        61.7% (29W / 18L)
  Avg Win:         +$14.22
  Avg Loss:        -$7.89
  Profit Factor:   2.31
  Best Trade:      +$42.10 (SOL-USD)
  Worst Trade:     -$19.33 (AVAX-USD)
====================================================
```

### Charts Generated

Interactive Plotly HTML charts (plus static PNGs if `kaleido` is installed):

- **Equity Curve** — Account value over time with starting balance reference line
- **P&L Per Trade** — Green/red bar chart of each closed trade
- **Cumulative P&L** — Running sum of net profit/loss
- **Cumulative Fees** — Total fees paid over time
- **P&L by Symbol** — Horizontal bar chart of net P&L grouped by coin
- **Summary Gauges** — Win rate, profit factor, total P&L, total fees

### Kraken Sync / Backfill

The `--sync` flag pulls your complete trade history from Kraken via the CCXT `fetch_my_trades` API. It deduplicates by order ID, so it's safe to run repeatedly. After syncing raw trades, it automatically pairs buys and sells (FIFO) to rebuild the `position_closes.csv` round-trip log.

This is useful for:
- **Initial setup** — Backfill trades that happened before the tracker was installed
- **Reconciliation** — Verify local records match Kraken's history
- **Recovery** — Rebuild local data if CSVs are lost
