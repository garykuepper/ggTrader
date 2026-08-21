---
name: daily-trader-check
description: Check the ggTrader paper trader's health — container and image freshness, last cron run, portfolio value, open positions, recent orders, and data-tape sanity. Use when the user asks "is the trader running", "trader status", "daily check", "how's the trader doing", or similar health-check questions about the Alpaca paper-trading bot.
---

# Daily Trader Check

Produce a concise health report for the ggTrader **Alpaca paper trader**.
Run the steps in order, then render the summary block.

> **What this system is, as of 2026-08-21.** The live path is US equities on
> Alpaca paper, driven by `ggt paper --live` from cron (`45 12 * * 1-5`, via
> `scripts/paper_trade.sh`). **Crypto is parked by choice** — a stale crypto
> balance or an idle Kraken/Binance snapshot is expected, not a fault, so do
> not report it as one.

## 1. Container and image freshness

```bash
docker ps --filter name=ggtrader --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
docker image inspect ghcr.io/garykuepper/ggtrader:latest --format '{{.Created}}'
git -C /home/flynn/ggTrader log -1 --format='%h %ci %s'
```

The container is **not** bind-mounted to `src/` — it runs the code baked into
the image. **If the image predates the newest commit that touched
`src/`, the trader is running stale code.** This has bitten twice; call it
out prominently. The fix is `git push` → wait for
`.github/workflows/docker-build.yml` (~2.5 min) → `docker compose pull &&
docker compose up -d`.

## 2. Last cron run

```bash
ls -la ~/logs/paper_trade_*.log | tail -3
tail -40 ~/logs/paper_trade_$(date +%Y%m%d).log 2>/dev/null || echo "no run today yet"
```

Scan for `ERROR`, `CRITICAL`, `Traceback`, halt/gate messages, and whether
the run reported placing orders or was a dry run. A run that logs nothing
after "Starting paper trading run..." died mid-way.

## 3. Portfolio value and trend — query postgres MCP

```sql
SELECT run_date, portfolio_value, cash,
       (SELECT count(*) FROM jsonb_object_keys(positions)) AS n_positions
FROM paper_snapshots
ORDER BY run_date DESC
LIMIT 10;
```

Note `positions` is a JSONB **object keyed by symbol** (not an array), so
count its keys — `jsonb_array_length` silently returns nothing useful here.

Report the latest `portfolio_value`, the change vs the prior run date, and
cash as a share of the total. **Idle cash above ~50% is worth flagging** —
under-deployment has been a recurring finding.

## 4. Open positions

```sql
SELECT run_date, jsonb_pretty(positions)
FROM paper_snapshots
ORDER BY run_date DESC
LIMIT 1;
```

`positions` is JSONB. Extract symbols, quantities, and unrealized PnL if
present.

## 5. Recent orders

```sql
SELECT run_date, side, symbol, amount, order_id, created_at
FROM paper_trades
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

Report count, buy/sell split, and the symbols traded. Note `paper_trades`
records **submitted orders**, not closed round-trips — there is no realized
per-trade PnL column, so do not report a win/loss split from it. Derive PnL
from `paper_snapshots.portfolio_value` instead.

Cross-check against the broker with the Alpaca MCP (`get_account_info`,
`get_all_positions`, `get_orders`) when the DB and reality might disagree —
the DB records what the trader *thinks* it did.

## 6. Data tape sanity

The equity tape was corrupted once (every bar stamped one calendar day
early, fixed 2026-08-20 in `ef4e15f`). Cheap standing check:

```sql
SELECT to_char(timestamp, 'Dy') AS dow, count(*)
FROM ohlcv
WHERE venue = 'yfinance' AND interval = '1d'
GROUP BY 1 ORDER BY 2 DESC;
```

**Any Saturday or Sunday bar means the bug is back** — almost certainly a
stale image (step 1) writing through the old code path. Also worth checking
that the newest bar is recent:

```sql
SELECT max(timestamp) FROM ohlcv WHERE venue = 'yfinance' AND interval = '1d';
```

## Output format

Render this structure, filling in real values:

```
🟢/🔴 ggtrader_live: <up/down> (uptime: <Xh Ym>)
   Image: <created date> · HEAD: <sha> <date>  ⚠️ STALE IMAGE (if image older than last src/ commit)

📅 Last run: <date> — <ok / dry-run / errored>

💰 Portfolio: $<value> (Δ $<delta> / <pct>% vs <prior run date>)
   Cash: $<cash> (<pct>% idle)

📊 Open positions (<N>): <symbol list or "none">

📈 Orders, last 7d: <count> (<B> buy / <S> sell) — <symbols>

🗓️ Tape: <"clean, Mon-Fri only" or "🚨 N weekend bars — stale image?">  · newest bar <date>

⚠️ Issues: <"none" or 1-line summary of errors, halts, or stale-image warning>
```

Keep it tight — no extra prose unless something is wrong, then surface
specifics.
