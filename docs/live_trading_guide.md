# Live Trading Guide

How to take research output and put it on a real exchange. Read the [Architecture Guide](architecture.md) first if you want to know what's happening under the hood — it defines the terms used below (WFO = walk-forward optimization, OOS = out-of-sample, OCO = One-Cancels-Other, etc.).

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

- **A successful research run.** Either: (a) a recent row in the `runs` TimescaleDB table with `strategy_params.per_coin` populated, or (b) a `results/research/research_<timestamp>/run_results.json` on disk with the same structure. The trader auto-detects whichever is newer.
- **Exchange credentials.** An API (Application Programming Interface) key + secret for the active venue (Binance.US or Kraken Pro), stored in `.env`. Binance.US is the lower-fee venue and current deployment target.
- **TimescaleDB running and reachable** at the connection string in `.env`.
- **Enough USD on the exchange** to fund at least one position at the configured size. Binance.US minimum order sizes are typically $10–$25 per coin.

## Going live, step by step

### 1. Inspect with `--dry-run`

Always do this first. `--dry-run` mode computes signals and intended position sizes but places no orders. It also reads from a separate state record so it can't contaminate the live trader's circuit-breaker baseline.

```bash
python ggt.py trade --dry-run
```

Then in a second terminal:

```bash
python ggt.py signals
```

This snapshots the current 4-hour bar for every symbol in the live universe: which coins would fire an entry, which are blocked, which are already in a position.

### 2. Pick a sizing mode

Pick one — you can't mix:

- **Weighted (default)** — each coin's slice of capital is proportional to its OOS robustness score from research, capped at `MAX_COIN_ALLOCATION` (default 25%). Coins with zero or negative OOS robustness are skipped entirely (the allocator gives them 0% even if they survived the gates). Use this when you trust the research.
- **Adaptive** (`--adaptive-sizing`) — Kelly-criterion style. Each position is sized so a stop-out costs exactly `TARGET_RISK_PCT` (default 1%) of portfolio value. Wider stops mean smaller positions. Capped per-coin by `--max-position-pct` (default 15%). Use this when you want volatility, not research weights, to drive sizing.

### 3. Start the engine

```bash
python ggt.py trade
```

The bot runs as a long-lived loop, polling the active exchange every 5 minutes — but actual trade decisions are aligned to 4-hour UTC bar boundaries (00:00, 04:00, 08:00, …) because strategies operate on 4h candles.

Venue is selected by the `EXCHANGE` environment variable (`binanceus` or `kraken`). The broker layer in `src/ggTrader/execution/` wraps CCXT so the rest of the engine is venue-agnostic.

## What the bot does each cycle

Every poll (every 5 minutes):

1. **Reconcile positions** — compare local state to the exchange. If a stop-loss filled server-side while we were sleeping, pick it up.
2. **Check the circuit breaker** — if intraday Profit and Loss (PnL) drops more than `DAILY_LOSS_LIMIT_PCT` (default 5%), no new entries until tomorrow. Existing positions keep their stops.
3. **Snapshot balance and positions** into TimescaleDB so Grafana stays current.
4. **Fetch the latest OHLCV** for every symbol in the live universe.
5. **Compute signals** using the WFO-optimized parameters loaded at startup (one strategy + exit per coin).
6. **For each fired entry:**
   - Size the position (weighted or adaptive).
   - Place a market or limit buy via CCXT.
   - After the fill confirms, immediately place a **venue-native protection order**: a trailing-stop on Kraken, or an OCO order on Binance.US. Server-side so even if our process dies, the exchange still respects the stop.
7. **For each fired exit on an open position:** close out, log the trade.
8. **Poll open exit orders** to catch fills between cycles.

## State and persistence

- **TimescaleDB `system_state` table** (key = `live_trader_state`) — open positions, circuit-breaker status, last-check date, start-of-day equity baseline. This is the source of truth. Atomic upserts every cycle.
- **TimescaleDB rolling tables** — `orders`, `trades`, `live_balance_snapshots`, `live_positions_snapshot` are written in real time. Grafana reads from these.
- **`data/live/dashboard/`** — rendered HTML dashboards (`cumulative_pnl.html`, `equity_curve.html`, etc.). Read-only output artifacts; safe to delete and regenerate.

## Monitoring

**Grafana** at `http://localhost:3002` is the primary dashboard. Look for `run_id = LIVE`.

**Terminal** commands for quick checks:

```bash
ggt dashboard           # one-screen summary of balance / positions / 24h PnL
ggt trade-report        # closed-trade table
ggt signals             # current 4h bar — what's firing or in position right now
```

**Daily PnL report** — sent to Telegram + Discord at 06:00 local time via the `scripts/daily_pnl_report.sh` cron job. Includes realized + unrealized PnL, open positions, circuit-breaker status, BTC bull/bear regime context, BTC/ETH prices, and the Fear & Greed crypto sentiment index.

## Monthly recalibration

The live engine kicks off its own WFO research run on the **1st of each month at ~01:00 UTC**, then hot-reloads the new parameters into the running bot — no restart, no downtime. New parameters take effect on the next 4-hour bar.

## Troubleshooting

**Bot won't start**

- Check `.env` has the right exchange API keys (`BINANCE_API_LIVE_KEY` / `BINANCE_SECRET_LIVE_KEY` for Binance.US; `KRAKEN_KEY` / `KRAKEN_SECRET` for Kraken).
- Verify TimescaleDB is reachable: `python ggt.py db diag`.
- Make sure the `runs` table has a recent successful research row, or pass `--results /path/to/run_results.json` to point at a specific file.

**Circuit breaker triggered on startup**

- Symptom: log line `🛑 [CircuitBreaker] TRIGGERED: Loss -XX.XX% > 5.00%` immediately after `ggTrader Crypto Engine Started`.
- Cause: stale `daily_start_equity` in `system_state['live_trader_state']` (e.g. from a previous run with a different account balance).
- Fix: reset it.
  ```sql
  UPDATE system_state SET value = jsonb_set(value, '{daily_start_equity}', 'null') WHERE key = 'live_trader_state';
  UPDATE system_state SET value = jsonb_set(value, '{last_check_date}', 'null') WHERE key = 'live_trader_state';
  ```
  Then restart — the next cycle will capture the real current balance as the new baseline.

**Research workers stall mid-run**

- Likely a Numba just-in-time (JIT) compilation crash in one worker. Check `results/research/<run>/worker_N.log` for the traceback.
- If you're tight on RAM, lower `--workers` (default 5).

**Grafana shows no data**

- Make sure you've selected `run_id = LIVE` in the dropdown.
- If snapshots stopped writing for some reason, manually mirror CSV logs back to DB: `python ggt.py db sync-live`.

**Missing entry alerts on Telegram**

- Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in `.env`.
- Look for `[entry-alert]` lines in the live log — the bot logs every send attempt (success or failure).

---
*Back to [README.md](../README.md).*
