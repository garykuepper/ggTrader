# alpaca-cli — Capabilities & Fit for ggTrader Stock Implementation

Source: https://github.com/alpacahq/cli (MIT, Go, **alpha preview** v0.0.10). Installed binary: `~/.local/bin/alpaca`. **Does not** ship an MCP server (unlike `kraken-cli`), so it is not registered in `~/.claude.json`. It's a structured-output CLI for ops, scripting, and CI — pair it with the `alpaca-py` SDK that the planned stock pipeline already uses.

> **Stock implementation status:** Planned in `docs/stock_trading_plan.md`. yfinance for data, `alpaca-py` for execution, paper-first. Not yet built. This doc evaluates `alpaca-cli` against that plan, so all "fit" notes are forward-looking.

## Authentication

```bash
alpaca profile login              # OAuth, paper only (default)
alpaca profile login --api-key    # API keys, required for live
# or env vars (recommended for CI / non-interactive):
export ALPACA_API_KEY=PK...
export ALPACA_SECRET_KEY=...
```

Paper is the default account. Profiles stored at `~/.config/alpaca/profiles/` with 0600 perms. Our existing `.env` keys are `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`; the CLI uses different env names — either rename in a wrapper script or run `alpaca profile login --api-key` once.

> **Safety note from upstream:** "No confirmation prompts, no 'are you sure?' dialogs." `alpaca position close-all` and `alpaca order cancel-all` are immediate. Always start on a paper profile.

## What it CAN do for the stock pipeline

### Live ops / observability (post-launch)
- `alpaca account get` — equity, buying power, day-trade count.
- `alpaca account portfolio` — equity & PnL history (independent verification of our PnL reports).
- `alpaca account activity list` — fills, dividends, splits, fees, transfers (reconciliation against our `trades` table, just like `kraken trades-history` for crypto).
- `alpaca position list|get` — what's open, regardless of what `active_positions_stocks.json` thinks.
- `alpaca order list --status open|closed|all` — order state outside the bot.
- `alpaca clock` — is the market open right now? Cheap manual check; mirrors what `MarketHours` (planned in `core/market_hours.py`) does via SDK.
- `alpaca calendar` — trading-day calendar; useful for cron scheduling and avoiding half-days/holidays.

### Manual interventions
- `alpaca order submit --symbol AAPL --side buy --qty 10 --type limit --price 200` — manual entry if the bot is paused.
- `alpaca order cancel <id>` / `alpaca order cancel-all` — kill switch when the live trader misbehaves (paper first!).
- `alpaca position close <SYM>` / `alpaca position close-all` — flatten a single name or the whole book.
- `alpaca order replace` — adjust a stuck limit price without canceling/resubmitting.

### Market data sanity checks
- `alpaca data latest-quote --symbol AAPL` — confirm what `yfinance` says about the close (yfinance has occasional split/dividend lag; Alpaca data is authoritative for what we can actually trade against).
- `alpaca data multi-snapshots` — quick top-of-book across the whole universe.
- `alpaca data screener movers` / `most-actives` — ad-hoc inputs to a future "movers"-style universe expansion (mirrors the crypto `USE_MOVERS` flag).
- `alpaca data corporate-actions` — flag upcoming splits/dividends so we can skip names with imminent corp actions.
- `alpaca data news` — companion context for daily reports.

### Account configuration
- `alpaca account config get|set` — toggle DTBP check, fractional, no-shorting, etc. Worth scripting once at deploy time so live and paper accounts have identical risk settings.

### Scripting & CI
- `--csv`, `--jq` flags for piping into shell pipelines.
- `--schema` returns the JSON schema for any command — useful for codegen or for sanity-checking output drift between releases (this is alpha software).
- `alpaca api ...` — raw passthrough to any Alpaca REST endpoint, escape hatch when the typed commands lag the API.
- `alpaca doctor` — connectivity + auth diagnostics for the deploy host.

### Options & crypto perps (future)
- `alpaca option chain`, `alpaca option contracts`, `alpaca crypto-perp ...` — surfaces beyond what our `alpaca-py` integration is planned to use. Available if we ever extend.

## What it CANNOT (or SHOULD NOT) do for the stock pipeline

### Not a hot-path replacement for `alpaca-py`
The planned `StockExecutionEngine` (`docs/stock_trading_plan.md` §6b) calls `TradingClient.submit_order(LimitOrderRequest(...))` and `TrailingStopOrderRequest(...)` directly. That's the right answer:
- Subprocess-per-order would add 50–200ms of latency that we don't need, and it serializes when we want concurrency.
- The Python SDK gives us native typed requests, exceptions we can catch, and integration with our existing logger/metrics.
- Don't shell out from `StockExecutionEngine` to the CLI for any reason. If we need a feature the SDK lacks, prefer adding it in Python.

### Not a data source for research
- Our plan uses **yfinance** for OHLCV (1980+ history, full-market volume, no key, batch downloads). Alpaca free-tier data is **IEX-only (~2% volume)** and starts in **2016** — strictly worse for WFO.
- Don't switch the data loader to `alpaca-cli data bars` even though it works. Keep `YFinanceDataLoader` / `CachedYFinanceLoader` as designed.
- Exception: short-window sanity checks against yfinance for current-day closes (rare).

### No MCP server
- Unlike `kraken-cli mcp`, this binary has no MCP mode. I cannot call it from Claude Code as a tool. If we want agent integration, we either:
  1. Wrap it in a tiny MCP shim, or
  2. Just register the `alpaca-py` SDK calls behind our own MCP server, or
  3. Keep using it as a shell tool and live with that.

### Alpha software — expect breakage
- README explicitly says: "Commands, flags, and output formats may change or be removed without notice between releases. Do not depend on current behavior in production workflows."
- ⇒ Use it for **interactive ops**, not as a dependency of any cron/CI step that must survive upgrades. Pin the version (`alpaca update` is opt-in) if we do automate anything.

### Destructive by design
- No prompts on `position close-all` / `order cancel-all`. A typo or stale shell history can liquidate the book instantly. Mitigations:
  - Keep paper as the default profile (`alpaca profile login` does this).
  - Use distinct profiles for paper and live: `alpaca -p paper ...` vs `alpaca -p live ...`.
  - Never alias a destructive command. Never put one in a script without `--quiet` + explicit confirmation gate.

### Env-var name mismatch with our `.env`
- Our `.env` uses `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (alpaca-py convention). The CLI wants `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Either:
  - Run `alpaca profile login --api-key` once and let the profile handle it (recommended), or
  - Add a small shell wrapper that re-exports the names.

### Jurisdiction & account constraints
- xStocks / international assets aren't relevant; the equity universe we're targeting (S&P 500) is fine in the US. PDT rule still applies for sub-$25k accounts — the CLI doesn't check this; the API will reject.

## Recommended usage pattern

1. **Set up** — `alpaca profile login --api-key` against the paper account, then a separate `--profile live` against the live account once that's ready. Run `alpaca doctor` to confirm.
2. **Daily ops** — Use `alpaca account get`, `position list`, `order list`, `account activity list` from the host shell or scripts for quick checks alongside the existing dashboards.
3. **Reconciliation** — After each live session: `alpaca account activity list-by-type --type FILL -o json` cross-checked against the local `trades` table in TimescaleDB. Same role as `kraken trades-history` for the crypto side.
4. **Incident response** — Paper: `alpaca -p paper position close-all`. Live: never run close-all from muscle memory; cancel-then-close per name with explicit confirmation.
5. **Data sanity** — Spot-check yfinance closes vs `alpaca data latest-bar` only when something looks off in a backtest. Not a routine job.
6. **Stay out of `StockExecutionEngine`** — All bot-side execution flows through `alpaca-py`. The CLI is for humans (and a possible future MCP shim), not for the trader's hot path.
7. **Pin and treat as alpha** — If we automate anything against the CLI, pin the binary version and revisit on each upgrade. Prefer SDK calls for anything load-bearing.

## Crypto vs stocks parity (CLI-side)

| Concern | Crypto (Kraken) | Stocks (Alpaca) |
|---|---|---|
| CLI binary | `kraken` (Rust, v0.3.2) | `alpaca` (Go, v0.0.10 alpha) |
| MCP server | `kraken mcp` (registered) | None — shell only |
| Default safety | Confirm prompts, `--yes` to skip | **No prompts at all** |
| Paper trading | `kraken paper`, `kraken futures paper` | `alpaca profile login` defaults to paper |
| Reconciliation cmd | `kraken trades-history -o json` | `alpaca account activity list-by-type --type FILL -o json` |
| In hot path? | No — ccxt | No — `alpaca-py` |

## Files / config touched by install

- Binary: `~/.local/bin/alpaca` (extracted from `cli_0.0.10_linux_amd64.tar.gz`).
- No MCP registration (the CLI doesn't expose one).
- No env var changes — login when needed.

## Verification

```bash
~/.local/bin/alpaca --version       # 0.0.10
~/.local/bin/alpaca --help          # command list
# After login:
alpaca doctor                       # connectivity + auth diagnostics
alpaca clock -o json                # market clock (no auth required)
alpaca account get -o json          # auth required
```

## Open questions / next steps (only if/when we build the stock pipeline)

- Add `~/.local/bin` to `$PATH` (currently not in `.bashrc`) if we expect to use it interactively often.
- Decide whether to write a thin MCP shim around `alpaca` (low value vs. just exposing `alpaca-py` directly).
- Map the CLI's auth flow to a separate live profile and document the live-deploy runbook.
- Add `alpaca account activity` reconciliation to the planned daily PnL report once `StockExecutionEngine` is live.
