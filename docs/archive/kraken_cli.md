# kraken-cli — Capabilities & Fit for ggTrader

Source: https://github.com/krakenfx/kraken-cli (MIT, Rust). Installed binary: `/home/flynn/.cargo/bin/kraken` (v0.3.2). Registered as MCP server `kraken` at user scope in `~/.claude.json`, so it's available in Claude Code sessions on Encom alongside `postgres`, `searxng`, `fetch`, `github`.

## Authentication

Set once in the shell (or in `auth` config) before authenticated commands:

```bash
export KRAKEN_API_KEY="..."
export KRAKEN_API_SECRET="..."
```

Public market data and `kraken paper` work with no credentials. Use `kraken setup` for a guided first-time config.

## What it CAN do for ggTrader

### Live ops / observability
- `kraken balance -o json` — cash balances across all assets.
- `kraken extended-balance -o json` — balance + credit + holds; richer than what ccxt surfaces.
- `kraken trade-balance -o json` — margin/equity for the trade account.
- `kraken open-orders -o json` / `kraken closed-orders -o json` — order state without writing a query.
- `kraken positions -o json` — open margin positions.
- `kraken ledgers` / `query-ledgers` — full audit trail (deposits, fees, rebates, transfers) for reconciliation.
- `kraken volume -o json` — 30-day volume + current fee tier (useful for fee-aware sizing).
- `kraken trades-history -o json` — fills history for PnL reconciliation against the local DB.
- `kraken status` — exchange/trading-mode status; cheap health check.

### Manual interventions
- `kraken order buy|sell PAIR VOL --type limit --price ...` — place spot orders by hand if the bot is paused.
- `kraken order cancel <txid>` — kill a stuck order without touching the Python loop.
- `kraken wallet-transfer` — move funds between spot/margin/futures wallets.
- `kraken withdraw` — guarded withdrawals (requires `--yes` to skip confirm; gated by API key permissions).

### Market data (no auth)
- `kraken ticker`, `orderbook`, `orderbook-grouped`, `spreads`, `trades`, `ohlc`, `pairs`, `assets`, `server-time`.
- `kraken ws` — WebSocket streaming subcommands (live tickers, books, trades) — could be useful for ad-hoc microstructure inspection.

### MCP integration (now active)
- Inside any Claude Code session on Encom: I can call `mcp__kraken__*` tools directly to inspect account state, balances, orders, market data — no shell roundtrip.
- Pairs naturally with the existing `postgres` MCP: pull a recent fill from the DB, cross-check it against `kraken trades-history` in one conversation.

### Paper trading
- `kraken paper` (spot) and `kraken futures paper` (perps) — live prices, simulated fills, no API keys, no money.
- Useful as a smoke test for manual hypotheses or for Claude-driven "agent" experiments without putting capital at risk.

### Earn / staking
- `kraken earn strategies --asset ETH` and friends — surface staking yields if we ever park idle quote currency.

### Futures (not used today, but available)
- `kraken futures order ...`, `kraken futures positions`, perpetual + fixed-date contracts up to 50x. Same JSON-first interface.

## What it CANNOT (or SHOULD NOT) do for ggTrader

### Not a hot-path replacement for ccxt
- ggTrader's live loop is Python + ccxt, vectorized, in-process. `kraken-cli` is a subprocess per call — the spawn overhead alone (tens of ms) makes it unsuitable for the trader's tick loop, signal evaluation, or the ExecutionEngine's order placement path.
- ccxt also gives us a unified async client and rate-limit handling integrated with the existing event loop. Don't refactor that to shell out.

### No backtesting / no historical bulk loader
- `ohlc` returns at most ~720 candles per call (Kraken API limit). Not a substitute for the historical OHLCV ingestion that feeds WFO. Keep the existing TimescaleDB pipeline.

### No portfolio analytics, no strategy primitives
- It's a thin, faithful API client. No PnL attribution, no regime detection, no walk-forward — those stay in `src/ggTrader/`.

### No shared rate-limit budget with the bot
- The CLI doesn't coordinate with the live trader's rate-limit accounting. Heavy ad-hoc CLI use during trading hours could nudge us into Kraken's per-key limits. Prefer read-only commands during active trading; do bulk pulls off-hours.

### Jurisdiction limits
- xStocks (`AAPLx`, `TSLAx`, `SPYx`, etc.) are **not available in the USA**. Forex and some futures availability vary. We are not trading these in ggTrader today.

### Withdrawals & destructive actions
- `withdraw`, `wallet-transfer`, `order cancel-all`, and any `--yes` flow are real-money operations. Treat the CLI like any production tool — confirm before invoking from an agent context.

### Not on the production container
- The binary is installed on the host (`/home/flynn/.cargo/bin/kraken`), not inside `ggtrader_live`. If we ever wanted the bot itself to call it (e.g., for a fee-tier check), we'd need to add it to the image — but ccxt already covers what we need there.

## Recommended usage pattern

1. **Daily ops** — Use `kraken balance`, `extended-balance`, `open-orders`, `trades-history` from the host shell or via the MCP for quick checks.
2. **Reconciliation** — Cross-check `kraken trades-history -o json` against the `trades` table in TimescaleDB after each live session.
3. **Incident response** — If the trader misbehaves, `kraken open-orders` then `kraken order cancel <txid>` is the fastest manual brake.
4. **Experiments** — Use `kraken paper` for one-off hypothesis tests that aren't worth a full WFO run.
5. **Stay out of the hot path** — The live trader keeps using ccxt. Don't add CLI subprocess calls to `ExecutionEngine` or strategy code.

## Files / config touched by install

- Binary: `/home/flynn/.cargo/bin/kraken`
- MCP registration: `/home/flynn/.claude.json` (user scope, server name `kraken`, command `kraken mcp`)
- Shell PATH: `$HOME/.cargo/bin` (already in `.bashrc` via cargo env, or `source $HOME/.cargo/env`)

## Verification

```bash
kraken --version          # 0.3.2
kraken status -o json     # {"status":"online", ...}
claude mcp list           # kraken: ... - ✓ Connected
```
