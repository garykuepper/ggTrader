# ggTrader Unified Pipeline Guide

This document describes the end-to-end automated trading lifecycle of `ggTrader`, from symbol selection to live execution.

## 🏗️ The 4-Phase Lifecycle

The system is designed to run autonomously within Docker, performing periodic re-optimization to adapt to changing crypto market regimes.

### 1. Selection (Static)
The asset pool is defined in `data/combined_16_usd_2023-01-01_2025-12-31.json`. These 16 coins are selected based on:
- **High Volume**: Top tier liquidity on Kraken.
- **Historical Coverage**: Full 3-year history (2023-2025) available for robust training.

### 2. Re-Optimization (Every 30 Days)
The `auto_trader.py` orchestrator triggers a new **Walk-Forward Optimization (WFO)** run using a **3-year sliding window**.
- **Input**: Latest Kraken OHLCV data + 16-coin pool.
- **Process**: `run_full_pipeline.py` searches for the best strategy (RSI, EMA, PSAR, etc.) and parameters (ATR multiplier, etc.) for each coin independently.
- **Output**: `run_results.json` containing the "Best" configuration per coin.

### 3. Portfolio Analysis (Automated)
Immediately after WFO, the system runs `portfolio_analysis_standalone.py`.
- **Process**: It "simulates" the WFO signals against different allocation strategies (Equal Weight, Volatility Weighted, Robustness Weighted).
- **Selection**: The strategy with the highest **Sharpe Ratio** is automatically selected.
- **Output**: `portfolio_weights.json` containing the specific percentage allocation for each coin.

### 4. Live Execution (Heartbeat)
The `ExecutionEngine` runs in a 4-hour pollin loop, aligned with candle closes.
- **Signal Generation**: Uses the latest `run_results.json` to compute entries/exits for each coin.
- **Dynamic Sizing**: If an entry is detected, it reads `portfolio_weights.json` to determine the exact USD amount to risk (e.g., 8% of bankroll for BTC, 4% for a high-vol altcoin).
- **Order Execution**: Executes a Market Buy on Kraken and immediately places a native **Trailing Stop Loss (TSL)** for protection.

## 🐳 Docker Orchestration

The entire lifecycle is managed via `docker-compose.yaml`.

### Services
- **`ggtrader_db`**: TimescaleDB for high-speed OHLCV storage.
- **`ggtrader_live`**: The main bot running `auto_trader.py`.

### Persistence
- `/app/results`: Stores all reports, WFO results, and portfolio plots. These are mirrored to your host machine for review.
- `/app/data`: Stores `active_positions.json` to ensure the bot remembers its trades across restarts.

## 🛠️ Operational Commands

### View Live Heartbeat
```bash
docker compose logs -f ggtrader_live
```

### Review the Latest Performance Report
Check `results/pipeline_<timestamp>/pipeline_report.md` on your host machine.

### Review Portfolio Competition
Check `results/pipeline_<timestamp>/portfolio_analysis/comparison_stats.csv`.

---
*For technical details on individual scripts, see the [Architecture Guide](architecture.md) or [Live Trading Guide](live_trading_guide.md).*
