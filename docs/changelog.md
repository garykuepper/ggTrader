# Changelog

## 2026-04-10

### Fixed: dust positions inflating open position count

Exchange sells often leave sub-penny dust balances (e.g. 0.00001 ETH worth $0.003) due to lot-size rounding. These were picked up by `_reconcile_positions` as "untracked" positions and re-added to `active_positions.json`, inflating the open position count in reports.

- **Reconciliation** ([execution_engine.py](../src/ggTrader/core/execution_engine.py)): untracked exchange balances below `_DUST_THRESHOLD_USD` ($1.00) are now logged and skipped instead of being added as positions.
- **Report** ([pnl_report_builder.py](../src/ggTrader/utils/pnl_report_builder.py)): `_gather_report_data` filters out any existing positions with cost basis below $1.00 before counting or displaying them. This covers dust that was already in `active_positions.json` from prior reconciliations.

### Documentation reorganization

- Created [roadmap.md](roadmap.md) — themed roadmap covering extensions, research & strategy, infrastructure, and live trading hardening. Each item has a status tag and links to detailed specs where they exist. Includes a priority summary table ranked by impact x feasibility.
- Archived `future_tweaks_plan.md` to `archive/` — roadmap items migrated into the new roadmap; config experiment history and live config state preserved in the archive for reference.

## 2026-04-09

### Audit findings: silent data-loss bugs in the trade-recording layer

A thorough audit of how sell orders are calculated and how transactions are tracked uncovered five distinct bugs that caused trades executed on Kraken to never reach the local CSVs. The PnL math, fee aggregation, deposit-adjustment, and `_safe_extract_fill_price()` cascade were all correct — the bugs were in *which* trades made it into `position_closes.csv`, not in *how* the math was done once they were there.

When `ggt repair --dry-run` was run on the live data: **local `trade_log.csv` had 7 rows, Kraken had at least 50 trades — 43 trades silently dropped over weeks**.

After running the actual repair, the local CSV held 23 closed round trips with the real numbers: **5W / 18L, +$2.29 net PnL, profit factor 1.56, all-time win rate 21.74%**. The previously reported 0% win rate / -$123 loss / -2.32 Sharpe were all artifacts of corrupted local state.

### Fixed (execution_engine.py)

- **Bug A — Emergency rollback sell never recorded**: when OCO or TSL placement failed after a buy fill, the bot did an emergency `_execute_market_sell_order` to flatten the position and then deleted it from `active_positions`. `record_sell` was never called. Buy fee, sell fee, and PnL were all orphaned with no recovery path. Fixed via the new `_handle_emergency_rollback` method which routes through `_record_exit`.
- **Bug B — Reconcile-skip orphans the position**: when `_reconcile_positions` couldn't extract a fill price for a stale position, the code logged `SKIPPING CSV record` and **still** deleted the position from `active_positions`. The trade became permanently unrecoverable. Fixed: positions stay in `active_positions` with `pending_repair=True` flag for the next reconcile / repair cycle to retry.
- **Bug C — Untracked-position exits write half a record**: when the reconciler discovered a position on the exchange that wasn't in local state, it added it with `entry_price=None`. Subsequent `record_sell` calls landed in `trade_log.csv` but skipped `position_closes.csv` (the line-111 guard). Fixed via new `TradeTracker.find_open_buy_for(symbol)` which seeds entry data from the trade log via FIFO matching.
- **Bug D — `record_sell` used stored amount instead of actual filled amount**: all sell paths passed `pos.get("amount")` instead of reading the order's `filled` field. Partial fills or dust would record the wrong quantity. Fixed via the new `_safe_extract_filled_amount` helper, baked into `_record_exit`.
- **Bug E — Balance snapshot poisoned on fetch failure**: `_get_total_portfolio_usd()` returned `START_CASH` (default $1000) on any exchange exception, then the caller wrote that fake value to `balance_snapshots.csv`, poisoning every equity-based metric. Fixed: returns `Optional[float]`, both call sites handle `None` (snapshot writer skips, position sizer aborts the entry).

### Refactored

- **`_record_exit` helper**: extracted the "fetch order → extract price → record sell" sequence into a single method on `ExecutionEngine`. All four sell paths (strategy_signal, reconcile, poll, emergency rollback) now go through this one helper, so any future fix only needs to be applied in one place. The helper takes an `allow_ticker_fallback` flag to differentiate "fresh sells we just placed" (where ticker is fine) from "stale exits we discovered later" (where ticker would be wrong).
- **`_safe_extract_filled_amount`**: new module-level helper that mirrors `_safe_extract_fill_price` but for the order's filled quantity. Used by `_record_exit` to ensure the recorded amount is the real exchange fill, not the bot's pre-trade expectation.

### Added: `ggt pnl-daily` auto-syncs from Kraken before reading any CSV

The most important fix because it heals all five bugs after the fact even when other defenses miss something. At the very top of `run_pnl_daily`, the new `_autosync_from_kraken` helper:

1. Builds a Kraken CCXT client from `.env` credentials.
2. Calls `TradeTracker.sync_from_kraken` (idempotent, dedupes by `order_id`) which appends new trades to `trade_log.csv` and rebuilds `position_closes.csv` from scratch via FIFO matching.
3. On failure (e.g. network outage, missing API keys), warns and continues with stale CSV data — a `⚠️ STALE DATA` banner is added to the top of both the HTML Telegram message and the markdown report so the user immediately sees that the numbers may be behind reality.

A new `--no-sync` flag skips the sync for fast offline runs. Default behavior is to sync on every invocation.

### Added: `ggt repair` CLI command for manual recovery

New subcommand at [src/ggTrader/cli/cmd_repair.py](../src/ggTrader/cli/cmd_repair.py) that explicitly runs the same `sync_from_kraken` + `_rebuild_position_closes` flow as the auto-sync, separate from the daily report. Useful for one-off cleanup after a known incident, backfilling history, or auditing whether local state matches Kraken.

```bash
ggt repair                       # full sync, rebuild position_closes
ggt repair --since YYYY-MM-DD    # limit to recent trades
ggt repair --dry-run             # report the gap without writing
```

### Fixed: mixed-precision ISO8601 timestamps in CSVs

A latent bug surfaced after the first repair run because `position_closes.csv` ended up with mixed-precision timestamps (some rows had microseconds from `sync_from_kraken`'s `datetime` field, others didn't from the live writer). `pd.to_datetime` with no `format` arg crashed on the mix. Fixed: all 5 `pd.to_datetime` call sites in `live_metrics.py` and `pnl_report_builder.py` now pass `format="ISO8601"` which tolerates both precisions and any timezone representation (Z, +00:00, -0700). The `TradeTracker.get_*` accessors no longer auto-parse via `parse_dates` — callers parse explicitly when needed.

### Verification

- Local lint: all touched files pass ruff cleanly.
- `ggt repair --dry-run` against live data confirmed 43 missing trades (the smoking gun).
- `ggt repair` (live data) added all 43 trades and rebuilt position_closes.csv with 23 round trips.
- `ggt pnl-daily --no-notify --print` produced a clean report with the new metrics: `+0.95% trading return, Sharpe 1.85, profit factor 1.56`.
- Telegram test: HTML message rendered correctly with monospace tables, no STALE banner (sync succeeded).
- Live trader rebuilt and restarted: `[Reconcile] State verified: 3 position(s) match exchange`, no `SKIPPING CSV record` errors, polling cadence 300s.

## 2026-04-08

### Fixed: wrong stored entry_price from CCXT `order["average"]`

PENGU was reporting cost ≈ $54.79 in the daily PnL when the actual Kraken fill was $24.98. Root cause: `_safe_extract_fill_price()` in [execution_engine.py](../src/ggTrader/core/execution_engine.py) preferred `order["average"]` from CCXT, which on Kraken can return a stale or partial-fill snapshot value while `order["filled"]` reflects the full cumulative volume — yielding a bogus VWAP when stored as `entry_price` in `active_positions.json`.

For the PENGU order: stored `entry_price=0.013651` × `amount=4013.49 = $54.79`, but real `cost=$24.98`, real VWAP `0.006225`. A cross-check against Kraken via `fetch_order` confirmed the discrepancy and also surfaced a wrong BNB stored entry (`906.02` vs real `606.68`) — interesting because the BNB OCO levels on Kraken were placed correctly off the real price, so the stale value crept into `active_positions.json` separately from the OCO placement path.

**Fix**: prefer `order["cost"] / order["filled"]` as the primary fill-price source. CCXT's standardized `cost` is the total quote-currency value across all fills, so the ratio is always the true VWAP. `order["average"]` is now the second-choice fallback. Code change is the new branch at the top of `_safe_extract_fill_price()`.

Also patched the existing bad entries in `data/active_positions.json`:
- PENGU: `entry_price` 0.013651 → 0.006225
- BNB: `entry_price` 906.02 → 606.68

The PENGU trailing stop on Kraken (`OGRAXH-QLJ7U-57J5EX`) was sized off the wrong entry (`stop_pct=12.15%` derived from ATR vs the doubled price) and may need to be cancelled and re-placed. The BNB OCO was already at correct levels.

## 2026-04-07

### Fixed: phantom +72% return from manual deposits

The daily PnL report was reporting `Total return +72%` because it computed return as `(latest_balance / first_balance) - 1`, treating manual capital injections as trading profit. With $200 of fresh deposits (April 1) on a starting balance of ~$135, that math inflated apparent profit by exactly $200.

**Fix**: deposit-aware equity curve.

- New module [src/ggTrader/utils/kraken_ledger.py](../src/ggTrader/utils/kraken_ledger.py) — fetches deposit and withdrawal history via `exchange.fetch_deposits()` + `exchange.fetch_withdrawals()` (CCXT). Persists to `~/.cache/ggtrader/kraken_ledger.json` with a 1h TTL and incremental fetch (only pulls entries newer than the last cached one).
- New helper `apply_deposit_adjustment()` in [live_metrics.py](../src/ggTrader/utils/live_metrics.py) — subtracts cumulative net deposits made AFTER the first balance snapshot from each subsequent balance, isolating trading PnL from capital flows.
- The report now computes Sharpe / Sortino / Max DD / Calmar / total return on the **deposit-adjusted** equity, while still showing the **raw Kraken balance** in the snapshot row so the user sees both numbers correctly.

Result on this account: trading return flipped from `+72.34%` (wrong) to `-1.89%` (real, deposit-adjusted from `$134.72 → $132.18` over the live tracking window).

Initial implementation tried `fetch_ledger()` first but Kraken returns mostly `type='trade'` entries by default — the deposit/withdrawal entries needed exchange-specific filter params. Switched to `fetch_deposits()` + `fetch_withdrawals()` directly which is simpler and more reliable.

### Added: HTML-formatted Telegram messages

Telegram's Markdown parser doesn't render tables and is finicky with special characters, so the previous plain-text-only summary felt cramped. Switched to Telegram's `parse_mode=HTML` which supports `<b>`, `<i>`, `<pre>`, `<blockquote>`.

- New `build_daily_pnl_summary_html()` in [pnl_report_builder.py](../src/ggTrader/utils/pnl_report_builder.py) renders the report with bold section headers and `<pre>`-block monospace tables. Cells are HTML-escaped via a `_h()` helper to avoid injection from symbol names or alert messages.
- Helper functions `_render_kv_table()` (2-column key/value with right-padded keys) and `_render_table()` (n-column with header separator) produce ASCII table layouts that align cleanly inside Telegram's monospace `<pre>` rendering.
- Open positions table now fetches **current prices** via the public Kraken ticker and computes **unrealized PnL** per position plus a portfolio total — something the markdown report never showed.
- The CLI [cmd_pnl_daily.py](../src/ggTrader/cli/cmd_pnl_daily.py) dispatches by notifier type: Telegram gets the HTML summary inline + the markdown file as an attachment via `sendDocument`; Discord gets the full markdown.

### Fixed: scheduled cron run silently failed

The `daily_pnl_report.sh` cron wrapper used `docker exec -T` (the `-T` flag is only valid for `docker compose exec`, not plain `docker exec`). Cron fired at 8am as scheduled but the script crashed immediately with `unknown shorthand flag: 'T'`. Removed the flag and added an explicit `PATH` export so the script works under cron's minimal environment.

Also added the cron entry: `0 8 * * * /home/flynn/ggTrader/scripts/daily_pnl_report.sh` (daily at 8am America/Los_Angeles).

### Cleanup: archived corrupted CSV history

The legacy `position_closes.csv` and `trade_log.csv` were full of bogus PnL records from the entry-price storage bug fixed yesterday. Archived them to `data/live/archive_20260406_corrupted_pre_fix/` and started fresh with empty header-only files. Going forward all new trades use the corrected entry-price recording from the 2026-04-06 fixes.

`balance_snapshots.csv` was kept as-is — those are real Kraken balance fetches and the only data we can use to compute equity-curve metrics historically.

## 2026-04-06

### Investigation: bot was profitable, CSV PnL was wrong

Forensic investigation via [scripts/investigate_live_trades.py](../scripts/investigate_live_trades.py) revealed the live trader was actually **profitable on Kraken (+$9.88 realized over 5 days, 24 round trips, 5W/19L)** despite local CSV showing -$123.38 with 0W/13L. The 0% win rate and $130+ "loss" reported in [docs/future_tweaks_plan.md](future_tweaks_plan.md) were almost entirely **bookkeeping errors**, not real losses.

Note on balance growth: the account moved from $134 → $232 during this period, but ~$100 of that was a manual capital injection by the user to clear Kraken minimum order sizes — NOT trading gains. The actual trading PnL was a modest +$9.88.

### Fixed (execution_engine.py)

- **`entry_price` stored stale signal close**: `self.active_positions[symbol]["entry_price"]` was set to `sig["current_price"]` (the close of the most recent OHLCV bar, which can be hours stale and wildly different from the actual market fill price). Now stores the real `fill_price` from the buy order's `average` field. This was the root cause of the bogus -36% to -54% per-position losses.
- **Broken ternary in fill_price extraction**: `(order.get("average") or sig["current_price"] if "order" in dir() else sig["current_price"])` had unclear precedence and could fall back to a stale price. Replaced with a `_safe_extract_fill_price()` helper that cascades through `order["average"]` → trade-fill weighted average → limit price → fresh ticker.
- **Reconciliation `closed_order.get("average", 0)` fallback to zero**: Recording `exit_price=0` poisoned `position_closes.csv` with -100% loss records. Now uses the same safe extraction with a `fetch_my_trades` fallback, and **skips the record entirely** rather than writing bogus data if no valid price can be determined.
- **`fixed_sl_tp` strategy-signal exit had the same bug**: Used `sell_order.get("average") or sig["current_price"]`. Now uses safe extraction with ticker fallback.
- **NaN ATR stop logging**: When the ATR indicator returned NaN, the silent fallback logged `$nan` and hid the underlying data quality issue. Now explicitly detects NaN with `math.isnan()` and logs a warning.
- **`STRATEGY_MAP` missing new strategies**: The live trader's `STRATEGY_MAP` did not include `stoch_rsi_reversal` or `keltner_breakout`, so coins selecting those strategies were silently skipped. Added both.

### Added (execution_engine.py)

- **Periodic exit-order polling**: New `_poll_open_exit_orders()` method runs every 5 minutes between 4h candles via `POLL_INTERVAL_SEC=300`. When a TSL/OCO order's status flips to closed, the exit is recorded immediately instead of waiting up to 4 hours for the next reconciliation. This eliminates the `[Reconcile] not held on exchange` pattern that was producing late, often-bogus exit records.
- **`_safe_extract_fill_price()` helper**: Cascades through CCXT order fields (`average` → weighted trade fills → limit `price`) to robustly extract a meaningful fill price. Returns `None` if no valid price exists, so callers can refuse to record garbage.

### Added (daily PnL reporting)

- **`src/ggTrader/utils/live_metrics.py`** — Lightweight Sharpe / Sortino / Calmar / max-drawdown computation for live trade data (does not require VectorBT Portfolio objects, unlike the existing `core/metrics.py`).
- **`src/ggTrader/utils/notifier.py`** — `TelegramNotifier` and `DiscordNotifier` backends using webhook-style APIs (no `python-telegram-bot` or `discord.py` deps needed). `build_notifiers_from_env()` silently no-ops if env vars are missing.
- **`src/ggTrader/utils/pnl_report_builder.py`** — Daily markdown PnL report with snapshot, alerts, 24h activity, all-time health, open positions, and recent trades. Configurable alert thresholds (balance floor, consecutive losses, max drawdown).
- **`src/ggTrader/cli/cmd_pnl_daily.py`** — New `ggt pnl-daily` CLI command. Args: `--since`, `--until`, `--output`, `--no-notify`, `--print`.
- **`scripts/daily_pnl_report.sh`** — Cron wrapper. Schedule with `0 9 * * * /home/flynn/ggTrader/scripts/daily_pnl_report.sh` for a daily 9am report.
- **`scripts/investigate_live_trades.py`** — Forensic diagnostic that pulls Kraken history via CCXT and cross-references against local CSV. Read-only.
- **`.env.example`** — Added `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DISCORD_WEBHOOK_URL` placeholders.

### Why

Losses appeared catastrophic but Kraken showed otherwise. Root cause was a single line storing the signal price instead of the fill price as `entry_price`, which corrupted every position close. Combined with the periodic-polling fix and the safe exit-price extraction, future PnL records will reflect actual realized PnL. The daily report ensures we'll catch any real issue within 24h instead of discovering it days later.

## 2026-04-02

### Added
- **StochRSI Reversal strategy** (`stoch_rsi_reversal`) — 12 param combos. Faster-cycling mean-reversion entry that fires when StochRSI K-line bounces above oversold threshold. Selected by TRUMP-USD (robustness 0.50).
- **Keltner Channel Breakout strategy** (`keltner_breakout`) — 9 param combos. ATR-based volatility-adaptive channel breakout. Selected by TAO-USD (robustness 0.99, #2 overall).
- `compute_stochrsi()` and `compute_kc()` methods on `IndicatorPrecomputer`
- Tests for both new strategies in `test_vectorized_architecture.py`
- Local `.venv` environment for host-side development and testing
- [Future tweaks plan](future_tweaks_plan.md) documenting planned experiments
- Default `--end-date` for `ggt research` changed from hardcoded `2025-12-31` to today's date

### Tested & Reverted
- Composite metric weights biased toward Sortino/Calmar (0.30 each) — cut portfolio from 32→26 coins
- Fold consistency gate tightened (0.33 → 0.40) — too aggressive, removed viable coins
- OOS robustness blend alpha (0.65 → 0.70) — inconclusive when bundled with other changes

**Lesson learned**: Don't bundle config changes. Test one at a time.

### Research Results
- `research_20260402_123248` — 32 coins, WFO CAGR 15.46%, YTD -10.66% (vs BTC -20.92%)
- Deployed to live trader

## 2026-04-01

### Changed
- **N_SPLITS 6 → 8** — more OOS folds for robustness scoring
- **RSI grid narrowed** 72 → 24 combos — removed extreme lengths (5, 7, 28) and oversold levels (15, 40)
- **EMA grid narrowed** 25 → 9 combos — removed very fast/slow pairs overlapping with regime filter
- **Exit ranges widened** — removed 1% stop and 2% trailing (too tight for 4h crypto), added 5% stop, 10% TP, 12% trailing
- **Regime filter EMAs** 50/200 → 20/100 — ~2x faster regime detection, warmup bars 200 → 100
- **Added `bbands_mean_reversion`** to WFO tournament (was implemented but not in param grids)

### Research Results
- `research_20260401_225306` — 33 coins, WFO CAGR 15.15%, YTD 27.41%
- Massive improvement from prior run (CAGR -3.73% → +15.15%)
- Key driver: wider exit ranges stopped premature stop-outs

## 2026-03-29

### Changed
- Portfolio quality gates: OOS-weighted allocation, regime filter, diversity cap
- Lower `MIN_CLOSED_TRADES_TRAIN` from 5 to 3

### Research Results
- `research_20260329_120504` — 25 coins, WFO CAGR 13.15%, YTD -8.55%

## 2026-03-28

### Fixed
- SPY cache collision between phase-2 and phase-3 date ranges
- Regime mask shape mismatch with duplicate MultiIndex columns
- BTC regime filter EMA warmup and WFO fold consistency

### Changed
- Refactored `orchestrator.py` into focused sub-modules
- Pruned param grids based on 261 WFO selections across 13 runs
- Added altcoin index regime filter as mid-correlation tier

## 2026-03-27

### Added
- BTC regime filter (blocks long entries when BTC below EMA in bear markets)
- YTD dashboard plot and phase labels
- Executive summary CAGR emoji badges

### Fixed
- BTC regime filter DB fallback for workers without BTC in coin batch
- BTC regime filter timezone mismatch causing vectorized path crash
