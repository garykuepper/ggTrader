# ggt Unified CLI Reference

This document provides a detailed command reference for the `ggt` unified interface, which orchestrates the entire trading lifecycle.

## 🚀 Commands Overview

| Command | Category | Description |
| :--- | :--- | :--- |
| `ggt research` | Optimization | Parallel WFO parameter search across a universe. |
| `ggt backtest` | Validation | Single-pass simulation using optimized parameters. |
| `ggt production` | Allocation | Competitive portfolio optimization and weight generation. |
| `ggt trade` | Execution | Long-running live trading heartbeat (Stocks/Crypto). |
| `ggt dashboard` | Monitoring | Terminal-based performance summary and Plotly charts. |
| `ggt db` | Maintenance | TimescaleDB administration (sync, clean, export). |
| `ggt ingest` | Data | Synchronize historical OHLCV data from exchanges. |
| `ggt status` | Status | Real-time progress monitoring for active research runs. |
| `ggt report` | Reporting | Regenerate Markdown reports from previous run results. |

---

## 🛠 Command Details

### 1. Research (`ggt research`)
Generates a volume-ranked universe and runs a **Grand Walk-Forward Optimization (WFO)**.

- **`--asset-class`**: Switch between `crypto` (default) and `stocks`.
- **`--top N`**: Select the top N assets by notional volume.
- **`--days N`**: Number of days of historical data to process (e.g., `1095` for 3 years).
- **`--workers N`**: Number of parallel CPU workers (default: 5).
- **`--end-date`**: Use a fixed end-date (YYYY-MM-DD) for reproducible research.

```bash
# Example: 3-year Stock research for top 25 symbols
python ggt.py research --asset-class stocks --top 25 --days 1095
```

### 2. Backtest (`ggt backtest`)
Replays trading logic against historical data to verify performance.

- **`--run-id`**: Path to a specific research result folder. Defaults to latest.
- **`--symbols`**: Overwrite the symbol set (comma-separated).

```bash
python ggt.py backtest --symbols BTC-USD,ETH-USD
```

### 3. Production (`ggt production`)
Promotes research results to live trading weights by running a "Tournament" between allocation models (Equal Weight, Kelly, etc).

```bash
python ggt.py production --asset-class crypto
```

### 4. Trade (`ggt trade`)
Starts the live execution engine.

- **`--asset-class`**: `crypto` (Kraken) or `stocks` (Alpaca).
- **`--paper`**: Use the exchange's paper/sandbox environment.
- **`--dry-run`**: Log all actions but do not place orders.
- **`--adaptive-sizing`**: Enable volatility-normalized position sizing.

```bash
# Example: Start stock paper trading
python ggt.py trade --asset-class stocks --paper
```

### 5. Database (`ggt db`)
Manages the TimescaleDB storage layer.

- **`sync-live`**: Mirror historical CSV logs from `data/live/` into the database for Grafana.
- **`diag`**: View table sizes and row counts.
- **`clean`**: Remove malformed or orphaned asset data.
- **`export`**: Generate a standard Postgres SQL dump.

---

## 🔧 Performance Tuning

### Parallelism
The default **5 workers** is optimized for a 16-core machine.
- **High-Memory (32GB+)**: Increase to 8–10 workers for faster results.
- **Low-Memory (8GB-16GB)**: Decrease to 2–3 workers to avoid GIL contention and OOM.

### Caching
`ggt` uses multi-layer caching to speed up repeated runs:
1. **Universe Cache**: Caches top-N assets for 24 hours.
2. **Indicator Cache**: `IndicatorPrecomputer` saves computed TA values across folds.
3. **WFO Result Cache**: Skips optimization if inputs haven't changed.

---
*For technical theory and data flow details, see the [Architecture Guide](architecture.md).*
