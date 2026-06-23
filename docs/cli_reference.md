# CLI Reference

The `ggt` command-line tool is the entry point for all research, paper trading, and data operations.

---

## Command summary

| Command | Purpose |
|---|---|
| `ggt lab` | Run walk-forward optimization of a strategy |
| `ggt paper` | Run paper trading cycle (daily signal generation → orders) |
| `ggt ingest` | Pull historical OHLCV from exchanges into TimescaleDB |
| `ggt db` | TimescaleDB diagnostics and administration |

---

## ggt lab

Run a strategy through monthly walk-forward folds over a historical dataset. Supports single runs, parameter sweeps, and full walk-forward optimization.

### Syntax

```bash
ggt lab --strategy <name> [options]
```

### Required arguments

| Flag | Description |
|---|---|
| `--strategy` | Strategy to evaluate (see choices below) |

### Strategy choices

**Weight strategies** (target_kind="weights"):

| Name | Description |
|---|---|
| `xs_momentum` | Cross-sectional momentum — rank by lookback return, equal-weight top N |
| `dual_momentum` | Dual momentum — drops negative-momentum picks to cash |

**Signal strategies** (target_kind="signals"):

| Name | Description |
|---|---|
| `ema_cross` | EMA crossover signals |
| `wfo_tournament` | 4-combo EMA tournament, picks best per fold |
| `bb_reversion` | Bollinger Band mean reversion |
| `rsi_reversion` | RSI mean reversion |
| `macd_divergence` | MACD bearish/bullish divergence |
| `volume_bb_reversion` | BB reversion confirmed by volume spike |
| `mtf_reversion` | Multi-timeframe (weekly RSI + daily BB) |
| `ensemble` | Majority-vote of up to 6 sub-signals (configurable `min_agree`) |
| `ensemble_conviction` | Ensemble with conviction-weighted position sizing |
| `conviction_bb` | Conviction-weighted BB sizing |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--market` | `equity` | Market type: `equity` or `crypto` |
| `--universe` | `sp500` | Stock universe: `sp500`, `nasdaq100`, or `russell2000` |
| `--eval-start` | `2021-01-31` | Evaluation period start date (YYYY-MM-DD) |
| `--eval-end` | today | Evaluation period end date (YYYY-MM-DD) |
| `--top-n` | 50 | Universe size (number of stocks/coins) |
| `--lookback` | 252 | Momentum calculation window in trading days |
| `--skip` | 21 | Rebalance frequency in trading days (21 ≈ monthly) |
| `--max-stocks` | None | Cap universe at this count (for diagnostic runs) |

### Mode flags (mutually exclusive)

| Flag | Description |
|---|---|
| `--sweep` | Run parameter sweep — grid search over `sweep_params()` |
| `--wfo` | Walk-forward optimization — rolling train/test folds with OOS scoring |

Neither flag = single walk-forward run with default parameters.

### Sweep customization

| Flag | Description |
|---|---|
| `--sweep-param` | Override sweep range. Repeatable. Format: `--sweep-param key=v1,v2,v3` |

### Examples

```bash
# Single walk-forward run with ensemble
ggt lab --strategy ensemble

# Ensemble on Nasdaq-100 universe
ggt lab --strategy ensemble --universe nasdaq100

# Parameter sweep over BB reversion
ggt lab --strategy bb_reversion --sweep

# Sweep with custom RSI thresholds
ggt lab --strategy rsi_reversion --sweep --sweep-param rsi_oversold=20,25,30,35

# Walk-forward optimization of ensemble
ggt lab --strategy ensemble --wfo

# Evaluate momentum on a specific date range
ggt lab --strategy xs_momentum --eval-start 2023-01-01 --eval-end 2024-12-31

# Diagnostic run on small universe
ggt lab --strategy dual_momentum --top-n 10 --max-stocks 5
```

### Output

Results are persisted to TimescaleDB:

- **`lab_runs` table**: One row per invocation — strategy, config, execution timestamp, aggregate metrics (mean return, Sharpe, Calmar, max drawdown, win rate).
- **`lab_periods` table**: One row per fold — start date, end date, monthly return, Sharpe, Calmar, max drawdown, number of trades.

To query results:

```sql
-- List recent runs
SELECT run_id, strategy, config, created_at, mean_return, sharpe
FROM lab_runs ORDER BY created_at DESC LIMIT 10;

-- Examine folds for a run
SELECT * FROM lab_periods WHERE run_id = '<run_id>' ORDER BY fold_start;
```

---

## ggt paper

Run one paper-trading cycle: generate ensemble signals from current market data, filter through the ML feature gate, apply risk guardrails, and execute orders on Alpaca paper.

### Syntax

```bash
ggt paper
```

Typically run via cron (1:30 PM PT, Mon–Fri). No arguments — configuration is via environment variables and the deployed model.

### Pipeline

1. **Signal generation** — ensemble strategy on latest OHLCV data
2. **ML feature gate** — LightGBM classifier filters low-confidence entries (precision < 0.50 → DROP)
3. **Risk guardrails** — max positions (30), max concentration (5%), daily loss halt (3%), drawdown halt (15%)
4. **Order execution** — Alpaca paper orders with DAY time-in-force
5. **Persistence** — trades and portfolio snapshots to TimescaleDB
6. **Notification** — Telegram alerts for trades and daily summary

---

## ggt ingest

Pull historical OHLCV data from exchanges and insert into TimescaleDB.

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
ggt ingest              # Fetch most recent 1 day
ggt ingest --days 180   # Fetch 6 months
ggt ingest --days 1095  # Fetch 3 years
```

---

## ggt db

TimescaleDB administration and diagnostics.

### Syntax

```bash
ggt db <subcommand>
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
ggt db diag
ggt db clean
ggt db compression --enable
ggt db export > backup.sql
```

---

## Scripts

### ML signal pre-screen

Evaluate a signal strategy's entry quality via LightGBM classification.

```bash
python scripts/ml_signal_screen.py --signal <name> [--start DATE] [--end DATE] [--universe UNIV]
```

| Flag | Default | Description |
|---|---|---|
| `--signal` | (required) | Signal strategy name to evaluate |
| `--start` | `2021-01-01` | Data start date |
| `--end` | today | Data end date |
| `--universe` | `sp500` | Stock universe |

Outputs precision, recall, F1, sample count, positive rate, verdict (DROP/BORDERLINE/STRONG), and top feature importances. Results saved to `results/ml_screen_<signal>_<timestamp>.json`.

---

## Docker usage

```bash
docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>
docker compose run --rm ggtrader_live python ggt.py paper
docker compose run --rm ggtrader_live python ggt.py ingest --days 180
docker compose run --rm ggtrader_live python ggt.py db diag
```

The container uses `host.docker.internal:5433` for TimescaleDB connectivity.

---

*See [Architecture](architecture.md) for internals. For codebase guidelines, see [agents.md](../agents.md).*

*Back to [README.md](../README.md).*
