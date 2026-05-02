# ggTrader Project Context

This directory contains **ggTrader**, an algorithmic crypto trading bot designed for Kraken, utilizing a Walk-Forward Optimization (WFO) pipeline and a tiered regime filtering system.

## Core Architecture
- **Unified CLI**: All operations are routed through `ggt.py`.
- **Execution Engine**: The live trading bot (`ggt trade`) is a long-running process that manages its own lifecycle, including monthly recalibrations.
- **State Management**: Uses `data/live/` for trade logs, balance snapshots, and active positions.

## Monthly Recalibration (WFO)
- **Automation**: On the 1st of every month, the `ExecutionEngine` internally triggers a full research and production pipeline run.
- **Process**:
    1. **Phase 1 (WFO)**: Optimizes parameters for all symbols. Note: The Regime Filter is **NOT** applied during per-fold optimization.
    2. **Phase 2 (Validation)**: Validates selected parameters in a combined portfolio. The Regime Filter **IS** applied here.
    3. **Phase 3 (Recent Data)**: Evaluates performance on the most recent data (YTD) with filters active.
- **Reloading**: The bot reloads the new parameters automatically once Phase 2/3 complete. No container restart is required.

## Market Regime Filtering
The bot employs a tiered correlation-based filter to prevent trading in unfavorable market conditions:
- **High Correlation ($\ge 0.5$ with BTC)**: Controlled by the **BTC Regime** (BTC price vs its 100-day EMA).
- **Medium Correlation ($0.3$ to $0.5$ with BTC)**: Controlled by the **Altcoin Index Regime** (Index EMA).
- **Low Correlation ($< 0.3$ with BTC)**: Trades freely, ignoring global regime filters.
- **Report Status**: The daily PnL report shows the current status of these filters.

## Deployment Nuances
- **Docker**: The `ggtrader_live` container **does not** mount the `src/` directory.
- **Updates**: Code changes made on the host must be copied into the container via `docker cp` or applied via a container rebuild.
- **Logs**: Active logs are found in `logs/live_trader.log` inside the container and mapped to the host `logs/` directory.

## Daily Operations
- **PnL Reports**: Triggered at 08:00 AM local time via `scripts/daily_pnl_report.sh`.
- **Sync**: The report builder automatically syncs recent trade history from Kraken to ensure local CSVs are accurate.
