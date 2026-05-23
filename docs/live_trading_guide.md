# Live Trading Guide

How to take WFO output and put it on a real exchange. Read the [Architecture Guide](architecture.md) first if you want to know what's happening under the hood.

## Contents

1. [What you need](#what-you-need)
2. [Going live, step by step](#going-live-step-by-step) — [Dry-run](#1-inspect-with---dry-run) · [Sizing mode](#2-pick-a-sizing-mode) · [Start the engine](#3-start-the-engine)
3. [What the bot does each cycle](#what-the-bot-does-each-cycle)
4. [State and persistence](#state-and-persistence)
5. [Monitoring](#monitoring)
6. [Monthly recalibration](#monthly-recalibration)
7. [Troubleshooting](#troubleshooting)

---

## What you need

- A successful research run (`results/research/research_<timestamp>/run_results.json` exists, with `strategy_parameters.per_coin` populated)
- API key + secret for the active venue (Binance.US or Kraken Pro) in `.env` — Binance.US is the lower-fee target and the current Phase 2 deployment goal
- TimescaleDB running and reachable
- Enough USD on the exchange to fund at least one position at the configured size

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

The bot runs as a long-lived loop, polling the active exchange every 4 hours on UTC bar boundaries (00:00, 04:00, etc.). Venue is selected by `EXCHANGE` in config (`binanceus` or `kraken`); the broker layer (`src/ggTrader/execution/`) wraps CCXT so the rest of the engine is venue-agnostic.

## What the bot does each cycle

1. Fetches the latest OHLCV for every symbol in the live universe.
2. Computes signals using the WFO-optimized parameters from `run_results.json`.
3. Applies the regime filter (currently disabled by default — see Architecture).
4. Checks the daily-loss circuit breaker — if intraday drawdown > 5%, no new entries until the next day.
5. For triggering buys, sizes the position (weighted or adaptive) and places a limit-buy.
6. After fill, places a venue-native trailing-stop (Kraken) or an OCO order (Binance.US) so the position is protected even if our process dies.
7. Reconciles local state with the exchange — picks up server-side stop fills that happened between heartbeats.

## State and persistence

- **`data/active_positions.json`** — current positions, circuit-breaker status, and start-of-day equity baseline. Survives restarts. *(Migration to DB as authoritative source is on the roadmap.)*
- **TimescaleDB** — orders (`orders`), trades (`trades`), balance snapshots (`live_balance_snapshots`), and position snapshots (`live_positions_snapshot`) are written in real time. This is the source of truth for everything except in-flight positions and circuit-breaker state.
- **`data/live/dashboard/`** — rendered HTML dashboards (`cumulative_pnl.html`, `equity_curve.html`, etc.). Read-only artifacts; safe to delete.

## Monitoring

**Grafana** at `http://localhost:3002` is the primary dashboard. The live run id is `LIVE`.

**Terminal**:

```bash
ggt dashboard           # quick summary
ggt trade-report        # closed-trade table from CSVs
ggt signals             # current bar, what's firing or blocked
```

**Daily PnL** — sent to Telegram + Discord at 06:00 local time via `scripts/daily_pnl_report.sh` (cron). Includes realized/unrealized PnL, open positions, circuit-breaker status, BTC regime, and Fear & Greed.

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
