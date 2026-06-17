# CLI Reference

The `ggt` command-line tool is the entry point for all research and data operations. The codebase now contains three commands only.

---

## Command summary

| Command | Purpose |
|---|---|
| `ggt lab` | Run walk-forward optimization of a strategy |
| `ggt ingest` | Pull historical OHLCV from exchanges into TimescaleDB |
| `ggt db` | TimescaleDB diagnostics and administration |

---

## ggt lab

Run a strategy through monthly walk-forward folds over a historical dataset.

### Syntax

```bash
ggt lab --strategy <name> [options]
```

### Required arguments

| Flag | Choices | Description |
|---|---|---|
| `--strategy` | `wfo_tournament`, `xs_momentum`, `dual_momentum`, `ema_cross`, `wfo_tournament_signal` | Strategy to evaluate |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--market` | `equity` | Market type: `equity` or `crypto` |
| `--eval-start` | `2021-01-31` | Evaluation period start date (YYYY-MM-DD) |
| `--eval-end` | today | Evaluation period end date (YYYY-MM-DD) |
| `--top-n` | 50 | Universe size (number of stocks/coins) |
| `--lookback` | 252 | Momentum calculation window in trading days |
| `--skip` | 21 | Rebalance frequency in trading days (21 ≈ monthly) |
| `--max-stocks` | None | Cap universe at this count (optional, for diagnostic runs) |

### Examples

```bash
# Evaluate wfo_tournament on SP500 from 2021-01-31 to today
ggt lab --strategy wfo_tournament

# Evaluate ema_cross with a smaller lookback and skip
ggt lab --strategy ema_cross --lookback 126 --skip 10

# Evaluate xs_momentum on top 100 stocks over a specific date range
ggt lab --strategy xs_momentum --top-n 100 --eval-start 2023-01-01 --eval-end 2024-12-31

# Diagnostic run on a small universe
ggt lab --strategy dual_momentum --top-n 10 --max-stocks 5

# Evaluate a signal strategy
ggt lab --strategy ema_cross --market equity
```

### Output

Lab runs are **not** written to disk. Instead, results are persisted to TimescaleDB:

- **`lab_runs` table**: One row per invocation, storing strategy name, configuration (top_n, lookback, skip), execution timestamp, and aggregate metrics (mean return, Sharpe, Calmar, max drawdown, win rate).
- **`lab_periods` table**: One row per fold, storing per-fold metrics (start date, end date, monthly return, Sharpe, Calmar, max drawdown, number of trades).

To query results:

```sql
-- List recent runs
SELECT run_id, strategy, config, created_at, mean_return, sharpe FROM lab_runs ORDER BY created_at DESC LIMIT 10;

-- Examine folds for a run
SELECT * FROM lab_periods WHERE run_id = '<run_id>' ORDER BY fold_start;
```

---

## ggt ingest

Pull historical OHLCV (open/high/low/close/volume) data from exchanges and insert into TimescaleDB.

### Syntax

```bash
ggt ingest [options]
```

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--days` | 1 | Number of days of recent history to fetch |

### Examples

```bash
# Fetch the most recent 1 day
ggt ingest

# Fetch 180 days (6 months)
ggt ingest --days 180

# Fetch 1095 days (3 years)
ggt ingest --days 1095
```

### Notes

- Ingestion uses the `CachedExchangeLoader` (CCXT wrapper) to pull from the active exchange configured in `config.py`.
- Data is written to the TimescaleDB `ohlcv` table, keyed by `(timestamp, symbol, interval, venue)`.
- Duplicate bars are skipped; updates are written in-place.

---

## ggt db

TimescaleDB administration and diagnostics.

### Syntax

```bash
ggt db <subcommand> [options]
```

### Subcommands

| Subcommand | Purpose |
|---|---|
| `diag` | Print table sizes and row counts |
| `clean` | Remove malformed or orphaned rows |
| `truncate` | Drop specific tables (interactive prompt) |
| `compression` | Enable/disable TimescaleDB hypertable compression |
| `export` | PostgreSQL SQL dump to stdout |

### Examples

```bash
# Check database health
ggt db diag

# Clean up bad data
ggt db clean

# Enable compression for large tables
ggt db compression --enable

# Export database to SQL file
ggt db export > backup.sql

# Interactively drop tables
ggt db truncate
```

### Notes

- **diag**: Shows table structure and current row counts — useful for auditing before expensive operations.
- **clean**: Removes rows with NULL primary keys or foreign-key violations. Use before WFO runs to avoid schema conflicts.
- **compression**: TimescaleDB hypertables can be compressed to 10–50% of original size after data is considered "cold" (e.g., older than 30 days).
- **export**: Produces a full PostgreSQL dump; useful for backups or local replay.

---

## Docker usage

To run lab commands inside the Docker container (recommended for live environment consistency):

```bash
docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>
docker compose run --rm ggtrader_live python ggt.py ingest --days 180
docker compose run --rm ggtrader_live python ggt.py db diag
```

The container automatically uses `host.docker.internal:5433` for TimescaleDB connectivity on the host.

---

*See [Architecture](architecture.md) for how the lab works internally. For codebase guidelines, see [agents.md](../agents.md).*

*Back to [README.md](../README.md).*
