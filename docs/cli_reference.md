# CLI Reference

The `ggt` command is the single entry point for research, backtesting, live trading, and maintenance.

## Commands at a glance

| Command | What it does |
|---|---|
| `ggt research` | Walk-forward optimization across the top-N most-traded assets |
| `ggt backtest` | Replay optimized parameters on historical data |
| `ggt production` | Compare allocation models (equal weight, Kelly, etc.) and pick a winner |
| `ggt trade` | Live trading heartbeat |
| `ggt signals` | Per-symbol diagnostic — show which signals would fire right now |
| `ggt dashboard` | Terminal summary of trades and equity |
| `ggt status` | Real-time progress of an in-flight research run |
| `ggt report` | Regenerate `research_report.md` from a saved run |
| `ggt pnl-daily` | Daily PnL report → Telegram + Discord |
| `ggt trade-report` | Summarize closed trades from `data/live/position_closes.csv` |
| `ggt repair` | Re-sync trade history from Kraken |
| `ggt db` | TimescaleDB administration |
| `ggt ingest` | Pull historical OHLCV from exchanges |
| `ggt cleanup` | Remove old results and logs |

## The big four

### `ggt research`
Generates a fresh universe (top-N by 30-day notional volume) and runs WFO across it in parallel.

| Flag | Default | Notes |
|---|---|---|
| `--top N` | 50 | Number of coins in the universe |
| `--days N` | 1095 | History window in days (1095 = 3 years) |
| `--workers N` | 5 | Parallel WFO workers; tune to CPU + RAM |
| `--end-date` | today | Pin a date for reproducibility |

```bash
# 3-year research, top 70 coins, 5 workers (default)
python ggt.py research --top 70

# Pinned end-date for reproducibility
python ggt.py research --top 50 --end-date 2025-12-31
```

### `ggt backtest`
Replays the latest research output as a combined-portfolio simulation.

```bash
python ggt.py backtest
python ggt.py backtest --symbols BTC-USD,ETH-USD     # subset
python ggt.py backtest --run-id results/research/research_20260507_120000   # specific run
```

### `ggt trade`
Long-running live execution loop. **Always start with `--dry-run` to inspect what it would do.**

| Flag | Default | Notes |
|---|---|---|
| `--paper` | off | Use the exchange's sandbox |
| `--dry-run` | off | Compute signals + sizes, place no orders |
| `--weighted-sizing` | **on by default** | Each coin gets a slice of capital based on its OOS robustness; capped at `MAX_COIN_ALLOCATION` |
| `--adaptive-sizing` | off | Volatility-normalised — wider stops mean smaller positions |
| `--capital N` | None | Fixed dollar amount per trade (overridden by either sizing flag) |
| `--min-trailing-stop-pct N` | 4.0 | Floor for trailing stop %, regardless of WFO param |
| `--min-atr-trailing-pct N` | 4.0 | Floor for ATR-derived stop % |
| `--portfolio-usd N` | live balance | Override exchange portfolio query (used with `--dry-run-sizing`) |

```bash
# Live trading (default sizing = weighted)
python ggt.py trade

# Inspect what would happen without placing orders
python ggt.py trade --dry-run
```

### `ggt signals`
Snapshot of the current bar across the live universe. Shows which symbols are in entry, in position, blocked by regime, etc. — useful for "why didn't I get a buy?"

```bash
python ggt.py signals
python ggt.py signals --firing-only          # only show entries triggering
python ggt.py signals --symbols BTC-USD,SOL-USD --verbose
```

## Maintenance

### `ggt db`
| Subcommand | Purpose |
|---|---|
| `sync-live` | Mirror CSV logs in `data/live/` into TimescaleDB so Grafana sees them |
| `diag` | Table sizes and row counts |
| `clean` | Drop malformed or orphaned rows |
| `export` | Postgres SQL dump |
| `purge-wfo-cache` | Clear cached WFO results (run after changing scoring config) |

### `ggt ingest`
Sync historical OHLCV from Kraken into TimescaleDB.

```bash
python ggt.py ingest --days 180
```

## Performance tuning

**Worker count.** `--workers 5` is the safe default (assumes ~16 GB RAM headroom). Bump higher only if RAM and cores allow — each worker holds a copy of the OHLCV slice it's testing.

**Caching.** Three layers, all on by default:
1. **Universe cache** — top-N for the day, keyed by `(asset_class, snapshot_date, top, window)`.
2. **Indicator cache** (`IndicatorPrecomputer`) — TA values reused across folds.
3. **WFO result cache** — skips re-running unchanged `(symbol, combo)` pairs in the `wfo_cache` TimescaleDB table.

Run `ggt db purge-wfo-cache` whenever you change the scoring config (composite weights, fold consistency, OOS alpha, N_SPLITS).

---
*See the [Architecture Guide](architecture.md) for how WFO and the regime filter actually work.*
