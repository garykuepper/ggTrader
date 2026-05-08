# Live Trading Guide

How to take WFO output and put it on a real exchange. Read the [Architecture Guide](architecture.md) first if you want to know what's happening under the hood.

## What you need

- A successful research run (`results/research/research_<timestamp>/run_results.json` exists)
- Kraken API key + secret in `.env`
- TimescaleDB running and reachable
- Enough USD on Kraken to fund at least one position at the configured size

## Going live, step by step

### 1. Inspect with `--dry-run`

Always do this first. Dry-run computes signals and intended position sizes but places no orders.

```bash
python ggt.py trade --dry-run
```

Then run `ggt signals` in a second terminal to see exactly which coins would fire and which are blocked.

### 2. Pick a sizing mode

Two options:

- **Weighted (default)** — each coin's slice of capital is proportional to its OOS robustness from research, capped at `MAX_COIN_ALLOCATION` (default 25%). Coins with zero or negative robustness are skipped entirely. Use this when you trust the research.
- **Adaptive (`--adaptive-sizing`)** — Kelly-style. Each position is sized so a stop-out costs `TARGET_RISK_PCT` (default 1%) of portfolio value. Wider stops mean smaller positions. Use this when you want volatility to drive sizing instead of research scores.

You pick one or the other at startup; you can't mix.

### 3. Start the engine

```bash
python ggt.py trade
```

The bot runs as a long-lived loop, polling Kraken every 4 hours on UTC bar boundaries (00:00, 04:00, etc.).

## What the bot does each cycle

1. Fetches the latest OHLCV for every symbol in the live universe.
2. Computes signals using the WFO-optimized parameters from `run_results.json`.
3. Applies the regime filter (currently disabled by default — see Architecture).
4. Checks the daily-loss circuit breaker — if intraday drawdown > 5%, no new entries until the next day.
5. For triggering buys, sizes the position (weighted or adaptive) and places a limit-buy.
6. After fill, places a Kraken-native trailing-stop so the position is protected even if our process dies.
7. Reconciles local state with the exchange — picks up server-side stop fills that happened between heartbeats.

## State and persistence

- **`data/active_positions.json`** — current positions, circuit-breaker status, and start-of-day equity baseline. Survives restarts.
- **`data/live/`** — CSV logs (`trade_log.csv`, `position_closes.csv`, `balance_snapshots.csv`).
- **TimescaleDB** — orders, trades, and equity curves are mirrored in real time so Grafana stays current.

## Monitoring

**Grafana** at `http://localhost:3002` is the primary dashboard. The live run id is `LIVE`.

**Terminal**:

```bash
ggt dashboard           # quick summary
ggt trade-report        # closed-trade table from CSVs
ggt signals             # current bar, what's firing or blocked
```

**Daily PnL** — sent to Telegram + Discord at 08:00 local time via `scripts/daily_pnl_report.sh` (cron). Includes realized/unrealized PnL, open positions, circuit-breaker status, BTC regime, and Fear & Greed.

## Monthly recalibration

The live engine kicks off its own WFO research run on the **1st of each month at ~01:00 UTC**, then hot-reloads the new parameters into the running bot — no restart, no downtime. The new parameters take effect on the next bar.

## Troubleshooting

**Bot won't start**

- Check `.env` keys are correct.
- Verify the DB is reachable: `python ggt.py db diag`.
- Make sure the `run_results.json` path exists (use `--results PATH` to override).

**Research workers stall mid-run**

- Likely a Numba JIT crash in one worker. Check `results/research_<ts>/worker_N.log`.
- Lower `--workers` if you're tight on RAM.

**Grafana shows no data**

- Make sure you've selected the right run ID in the dropdown.
- Backfill CSV logs: `python ggt.py db sync-live`.

**Missing entry alerts on Telegram**

- Check `KRAKEN_KEY` / `TELEGRAM_BOT_TOKEN` are set.
- Look for `[entry-alert]` lines in the live log — the bot logs whenever it tries to send.

---
*Back to [README.md](../README.md).*
