# CLI Reference

The `ggt` command is the single entry point for research, backtesting, live trading, and maintenance. If terms here are unfamiliar — Walk-Forward Optimization (WFO), Out-of-Sample (OOS), Open/High/Low/Close/Volume (OHLCV), Profit and Loss (PnL), CCXT (a multi-exchange Python library), Average True Range (ATR) — see the [Architecture Guide](architecture.md), which defines them up front.

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
| `ggt trade-report` | Summarize closed trades from the `trades` TimescaleDB table |
| `ggt repair` | Re-sync trade history from the active exchange |
| `ggt db` | TimescaleDB administration |
| `ggt ingest` | Pull historical OHLCV from exchanges |
| `ggt cleanup` | Remove old results and logs |

## The big four

### `ggt research`

Generates a fresh universe (top-N coins by recent volume on the active venue), then runs walk-forward optimization across that universe in parallel.

| Flag | Default | What it does |
|---|---|---|
| `--top N` | 50 | How many coins to include in the universe |
| `--days N` | 1095 | History window in days. 1095 = 3 years |
| `--workers N` | 5 | Number of parallel WFO worker processes. Each worker holds its share of the OHLCV slice in memory, so tune to available RAM |
| `--end-date YYYY-MM-DD` | today | Pin a date for reproducibility (matches a closure-doc baseline, for example) |
| `--symbols A,B,C` | — | Override the volume-based universe with an explicit comma-separated list |

```bash
# 3-year research, top 70 coins, default 5 workers
python ggt.py research --top 70

# Pinned end-date for reproducibility
python ggt.py research --top 50 --end-date 2025-12-31

# Diagnostic run on a small explicit symbol set
python ggt.py research --symbols BTC-USD,ETH-USD,DOGE-USD
```

### `ggt backtest`

Replays the latest research output as a combined-portfolio simulation (all chosen per-coin strategies trading together).

```bash
python ggt.py backtest                                              # latest research run
python ggt.py backtest --symbols BTC-USD,ETH-USD                    # subset
python ggt.py backtest --run-id results/research/research_20260507  # specific run
```

### `ggt trade`

Long-running live execution loop. **Always start with `--dry-run` to inspect what it would do.**

| Flag | Default | What it does |
|---|---|---|
| `--paper` | off | Use the exchange's sandbox if available |
| `--dry-run` | off | Compute signals + position sizes, place no orders, don't mutate persistent state |
| `--weighted-sizing` | **on by default** | Each coin's capital is proportional to its OOS robustness score, capped at `MAX_COIN_ALLOCATION` (default 25%). The "trust-the-research" mode |
| `--adaptive-sizing` | off | Kelly-style sizing. Volatility-normalized — wider stops mean smaller positions. Mutually exclusive with `--weighted-sizing` |
| `--capital N` | None | Fixed dollar amount per trade (only used if neither `--weighted-sizing` nor `--adaptive-sizing` is set) |
| `--max-position-pct N` | 0.15 | Adaptive sizing only. Per-coin position cap as a fraction of portfolio (e.g. 0.10 = max 10%) |
| `--min-trailing-stop-pct N` | 4.0 | Floor for trailing stop %, regardless of what the WFO chose. Keeps stops from being absurdly tight |
| `--min-atr-trailing-pct N` | 4.0 | Same floor, but applied to ATR-derived stops |
| `--portfolio-usd N` | live balance | Override exchange balance lookup (used with `--dry-run-sizing` for what-if analysis) |
| `--results PATH` | auto-latest | Point at a specific `run_results.json` instead of the most-recent research run |

```bash
# Live trading with weighted sizing (default)
python ggt.py trade

# Inspect what would happen without placing orders
python ggt.py trade --dry-run

# Adaptive sizing with 10% per-coin cap (Phase 1 venue validation)
python ggt.py trade --adaptive-sizing --max-position-pct 0.10
```

### `ggt signals`

Snapshot of the current 4-hour bar across the live universe. Shows which symbols are firing entries, which are already in position, and what the WFO-chosen parameters look like. Useful for answering "why didn't I get a buy this cycle?"

```bash
python ggt.py signals
python ggt.py signals --firing-only          # only show entries triggering
python ggt.py signals --symbols BTC-USD,SOL-USD --verbose
```

## Maintenance

### `ggt db`

| Subcommand | Purpose |
|---|---|
| `sync-live` | Mirror any rolling CSV logs in `data/live/` into TimescaleDB so Grafana stays current (mostly a no-op now that state is DB-backed) |
| `diag` | Table sizes and row counts |
| `clean` | Drop malformed or orphaned rows |
| `export` | Postgres SQL dump |
| `purge-wfo-cache` | Clear cached WFO results. Run after changing scoring config (composite weights, fold consistency thresholds, OOS alpha, `N_SPLITS`) — cached results were computed under the old settings |

### `ggt ingest`

Pull historical OHLCV from the active exchange into TimescaleDB.

```bash
python ggt.py ingest --days 180
```

For bulk Binance.US backfills (faster than CCXT pagination, especially for multi-year history), use `scripts/backfill_binanceus.py` or `scripts/backfill_binanceus_universe.py` (universe-driven) — they pull Binance's zipped CSV archives directly.

## Performance tuning

**Worker count.** `--workers 5` is the safe default (assumes ~16 GB RAM headroom). Bump higher only if RAM and cores allow — each worker holds a copy of the OHLCV slice it's testing.

**Caching.** Three layers, all on by default:

1. **Universe cache** — top-N coin selection for the day, keyed by `(asset_class, snapshot_date, top, window, venue)` in the `universe_cache` table.
2. **Indicator cache** (`IndicatorPrecomputer`) — technical-analysis values reused across folds within a single run.
3. **WFO result cache** — skips re-running unchanged `(symbol, strategy, exit, param_grid, config_hash, ohlcv_hash)` combos via the `wfo_cache` TimescaleDB table.

Run `ggt db purge-wfo-cache` whenever you change the scoring config (composite weights, fold consistency, OOS alpha, `N_SPLITS`).

---
*See the [Architecture Guide](architecture.md) for how WFO and the selection gates actually work.*
