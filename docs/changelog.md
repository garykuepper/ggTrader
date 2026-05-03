# Changelog

## 2026-05-03

### Hardening: asset-class-aware research/production discovery

Prevented cross-class contamination when crypto and stocks research runs coexist in `results/research/`. Previously, the live trader, backtest, and production recalibration all called `get_latest_research_run()` without any filter — on next restart (or monthly auto-recalibration), the crypto trader could auto-pick up a newer stocks run and try to trade `NVDA, TSLA, ...` on Kraken.

- **`run_results.json` now writes `asset_class` at the top level** ([results_manager.py](../src/ggTrader/utils/results_manager.py) `_build_output_structure`). Backwards-compatible: the legacy `configuration._raw_config.ASSET_CLASS` location is still populated and is used as a fallback by the discovery code.
- **`get_latest_research_run` and `get_latest_production_weights` accept an optional `asset_class` filter** ([state_manager.py](../src/ggTrader/utils/state_manager.py)). Reading order: top-level `asset_class` → `_raw_config.ASSET_CLASS` → defaults to `"crypto"` for truly legacy runs.
- **`validate_results_asset_class()` helper** rejects an explicit `--results PATH` if the file's class doesn't match `--asset-class`. Hard-fail with a clear remediation message.
- **`cmd_trade`, `cmd_backtest`, `cmd_production`** all now pass `asset_class` to discovery and validate explicit paths.
- **Tests**: 12 unit tests in `tests/test_state_manager.py` cover discovery filtering, legacy fallbacks, and validator behavior.

Verified against existing on-disk runs: `research_20260502_230057` and `research_20260503_085408` resolve to `"stocks"` via `_raw_config` fallback; `research_20260501_205849` resolves to `"crypto"`. No data migration needed.

### Fixed: stocks WFO research pipeline (four root-cause bugs)

The 2026-05-02 stocks research run reported `YTD CAGR -56% / Sharpe -2.14`, masking strong per-worker phase-1 results (+101%, +95%, +40% on three of five shards). Investigation found four independent bugs.

- **Phase 2/3 dispatch lost `--asset-class`** ([cmd_research.py](../src/ggTrader/cli/cmd_research.py) `final_cmd`). The post-merge replay defaulted to crypto, so stock symbols were looked up via `CachedExchangeLoader`. Most returned empty; `CVX-USD` and `CAT-USD` (real Kraken pairs for Convex Finance and Cat token) silently replayed against crypto OHLCV — that's where the misleading aggregate came from. Now passes `asset_class` through, and `run_walk_forward_optimization.py --asset-class` is `required=True` to prevent silent crypto fallback.
- **`TimescaleDBLoader.fetch_ohlcv` unconditionally appended `-USD`** ([timescaledb_loader.py:48](../src/ggTrader/data/historical/timescaledb_loader.py)). Correct for crypto, wrong for stocks (stored as bare ticker). Added `asset_class` kwarg; `CachedYFinanceLoader` and `load_hybrid_validation_ohlcv` now thread it through.
- **Post-portfolio zeroing crashed silently** ([orchestrator.py:522](../src/ggTrader/core/orchestrator.py)). `final_pf.trades.count()` returns a scalar when `cash_sharing=True` with single group — `.values` then raised `AttributeError`. Every worker swallowed it, so the second-pass zeroing of dead allocations never ran. Fixed via `count(group_by=False)` and narrowed the `except`.
- **Research report linked to non-existent `docs/UNIFIED_PIPELINE.md`** ([report_generator.py:407](../src/ggTrader/utils/report_generator.py)). Repointed to `docs/architecture.md` (now contains the fold structure and benchmarks).

**Deferred:** universe shrinkage 25 → 16 stocks. Many gates (`MIN_CLOSED_TRADES_TRAIN=3`) reject daily-bar stocks with too few trades per fold. Re-evaluate after the next clean run.

## 2026-05-02

### Added: Multi-Asset Trading Engine (Stocks & Crypto)

Refactored the core execution layer into a robust multi-asset architecture.

- **Shared Base Engine**: Introduced `BaseExecutionEngine` for unified risk management, state persistence, and notification routing.
- **Stock Extension**: Launched `StockExecutionEngine` with Alpaca integration (Paper/Live support) and NYSE market hours awareness.
- **Stock Data**: Implemented `YFinanceDataLoader` with free daily bars back to 1980 and automatic TimescaleDB caching.
- **Equity Risk Filters**: Added SPY EMA and VIX-based macro regime filters for stocks.
- **Stock Universe**: New `ggt research --asset-class stocks` workflow with automated S&P 500 volume-ranked selection.

### Added: Daily Loss Circuit Breaker

The system now provides automated intraday protection for both assets.
- **Auto-Halting**: Halts new entries if the portfolio's intraday drawdown exceeds a configurable limit (default: **5%**).
- **Start-of-Day Snapshots**: Automatically tracks starting equity at UTC midnight.
- **Persistence**: Circuit breaker state is preserved across bot restarts.
- **Real-time Alerts**: Notifies via Telegram/Discord instantly upon trigger.

### Added: Real-time Grafana Dashboard & DB Mirroring
... [Existing Grafana notes] ...

## 2026-05-01

### Monthly Recalibration: WFO Run Successful

The automated monthly recalibration triggered successfully at 01:06 AM UTC. The research pipeline processed the top 50 symbols by volume, and the production parameters were promoted to live trading at 02:27 AM UTC.

- **Results**: YTD Strategy CAGR of **48.46%** (vs BTC at **-21.34%**) in the validation window.
- **Bot State**: The `ggtrader_live` bot internally detected the new month and reloaded parameters without requiring a restart.
- **Active Symbols**: 29 optimized symbols are currently being monitored.

### Added: Market Regime status to Daily PnL Reports

The daily PnL report (sent via Telegram/Discord at 08:00 AM) now includes a "Market Regime" section. This clarifies why the bot may be sitting on the sidelines even when the broader altcoin market looks bullish.

- **Status Shown**: BTC Regime (bull/bear), Altcoin Regime (bull/bear), and current BTC price.
- **Visuals**: Employs 🟢/🔴 indicators for immediate readability.
- **Implementation**: Computes regime status on-the-fly using `ccxt` data and the same tiered filtering logic used by the live bot.

### Added: Project-specific `GEMINI.md`

Created a root-level `GEMINI.md` to document the internal automation, tiered regime filtering logic, and deployment nuances for future developer context.

## 2026-04-27

### Fixed: live exchange loader crash on duplicate Kraken bars

Kraken's OHLC endpoint occasionally repeats the in-progress partial bar's timestamp inside a single response. When the next-symbol fetch had a different index range, `pd.concat(all_dfs, axis=1)` in `LiveExchangeLoader.fetch_ohlcv` raised `InvalidIndexError: Reindexing only valid with uniquely valued Index objects`. Added a per-symbol `df.index.duplicated(keep="last")` dedupe before the horizontal concat. Live event-loop fetches happened to be insulated by the cached_loader's pre-existing dedupe; the bug surfaced via `--dry-run-sizing`. Regression test in `tests/test_exchange_loader_dedup.py`.

### Removed: dynamic weight-based sizing

The `weight is not None` branch in `_execute_trade_logic` and the `portfolio_weights` loader on `ExecutionEngine` are gone. Trade sizing is now binary: `ADAPTIVE_SIZING=True` → adaptive sizing, otherwise `CAPITAL_PER_TRADE`. The `--weights` CLI flag, `WEIGHTS_PATH` config key, and auto-detection of `portfolio_weights.json` at startup are removed. `scripts/auto_trader.py` no longer threads `WEIGHTS_PATH` either. `portfolio_optimizer.py` still writes the file for research/analysis use; the helper `state_manager.get_latest_production_weights` is left for that purpose.

### Added: trailing-stop floor (`MIN_TRAILING_STOP_PCT` / `MIN_ATR_TRAILING_PCT`)

WFO-optimised stop distances were occasionally tight enough (sub-2%) for normal 4h crypto noise to trigger trailing-stop exits within hours of entry at small losses. Both live stop placement (`execution_engine.py` `_execute_trade_logic`) and the pre-buy sizing estimator (`_estimate_stop_pct_for_sizing`) now clamp `stop_pct` upward to a configurable floor for `atr_trailing` and `trailing_stop` exits. Default 4.0% for both. Logged when the clamp fires. CLI: `--min-trailing-stop-pct` / `--min-atr-trailing-pct`.

### Added: `--dry-run-sizing` flag on `ggt trade`

Prints what each symbol's adaptive position size would be at the current bar/ATR (using live `_estimate_stop_pct_for_sizing` + `_compute_adaptive_position_usd`) and exits. Lets us verify the new `ADAPTIVE_SIZING` path without waiting for a live entry to fire. `--portfolio-usd` overrides the exchange query.

### Added: `ggt trade-report` CLI

Summarises closed trades from `data/live/position_closes.csv` by `exit_reason` / `symbol` / `week`. Replaces grepping logs to evaluate live performance after config changes.

## 2026-04-23

### Added: adaptive (volatility-normalized) position sizing — opt-in

Risk-parity-style sizing that targets a fixed fraction of portfolio-at-risk per entry rather than a fixed dollar or weight allocation. Higher-vol coins (wider ATR stops) get smaller positions, lower-vol coins get larger ones — drawdown on a single stop-out is bounded to `TARGET_RISK_PCT` of portfolio regardless of which coin triggers it.

- **Formula**: `position_usd = (portfolio * TARGET_RISK_PCT) / (stop_pct / 100)`, capped at `portfolio * MAX_POSITION_PCT`, skipped if below `MIN_POSITION_USD`.
- **Pre-buy stop estimation** ([execution_engine.py](../src/ggTrader/core/execution_engine.py) `_estimate_stop_pct_for_sizing`): mirrors the post-buy stop computation so sizing and eventual stop stay consistent. `atr_trailing` uses `sig["atr_value"] / sig["current_price"]`; `trailing_stop` / `fixed_sl_tp` use their WFO-fixed stop params. Floored at 0.5% to prevent blown positions when ATR is transiently tiny.
- **Min-size gate**: if the sized position falls below `MIN_POSITION_USD` the entry is skipped (not clamped up) — taking a position so small that Kraken fees + slippage dominate is worse than waiting for the next candle.
- **Override semantics**: when `ADAPTIVE_SIZING=True`, it overrides weight-based sizing. When off, existing behavior is preserved (`weight * portfolio` if weights loaded, else `CAPITAL_PER_TRADE`).
- **CLI** ([cmd_trade.py](../src/ggTrader/cli/cmd_trade.py)): `--adaptive-sizing` flag plus `--target-risk-pct` (default 0.01), `--max-position-pct` (default 0.15), `--min-position-usd` (default 15.0). Defaults OFF — opt-in per the "test one config change at a time" rule.

**To enable**: edit `docker-compose.yaml` to append `--adaptive-sizing` to the `command` line (e.g. `python -u ggt.py trade --adaptive-sizing`), then `docker compose build --no-cache && docker compose up -d`.

### Added: real-time trade fill alerts (Telegram)

Rich push notifications on every entry and exit fill — closes the last operational blind spot before the daily PnL report's 24h lag. Telegram creds (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) are already configured; no additional setup required.

- **Notifier wiring** ([execution_engine.py](../src/ggTrader/core/execution_engine.py)): `ExecutionEngine.__init__` now calls `build_notifiers_from_env()` and a new `_notify(msg)` method fans out to every configured backend, swallowing exceptions so Telegram outages can't crash the trade loop.
- **Entry alerts**: fired in `_execute_trade_logic` after the exit order (OCO/TSL) is successfully placed, gated on `symbol in self.active_positions` so emergency rollbacks never produce a ghost "BUY" alert. Format includes strategy, exit type, fill price, size, cost, stop %, open position count.
- **Exit alerts**: fired inside `_record_exit` right after `tracker.record_sell(...)` — a single hook covers all four sell paths (`strategy_signal`, `trailing_stop`, `oco_exit`, `emergency_rollback`). Format includes entry/exit prices, hold time, $/% PnL (fee-adjusted), ✅/❌ marker, exit reason, strategy.
- **Helpers**: module-level `_format_entry_alert`, `_format_exit_alert`, `_format_hold_time`. HTML-formatted for Telegram's rich rendering; all dynamic values are HTML-escaped to avoid parse errors from symbols or reasons.
- **Restart safety**: alerts only fire from `create_order` success paths and from the poll/reconcile exit-recording flow — never from state reload, so trader restarts don't spam the chat.

## 2026-04-19

### Fixed: dust positions in local state never cleaned up

The 2026-04-17 fix added a dust check to the `fixed_sl_tp` strategy exit path, but only ran when `sig["exit"]` was True. A dust position (BNB-USD with `entry_price=null`, `amount=1.1e-07`) survived a week because its drifting price never crossed the strategy's stop or take-profit level, so no exit signal ever fired.

- **Proactive dust cleanup in reconciliation** ([execution_engine.py](../src/ggTrader/core/execution_engine.py)): after the standard stale/untracked checks, iterate `active_positions` and remove any entry whose cost basis *and* current market value are both below `_DUST_THRESHOLD_USD`. Catches zombies regardless of whether an exit signal ever fires.

## 2026-04-17

### Fixed: zombie dust positions retry sell every loop

When a position's OCO/TSL closed on the exchange but left a dust amount (e.g. `1.1e-07 BNB`), the `fixed_sl_tp` strategy exit fired every 4h loop, tried to sell, Kraken rejected it ("amount must be greater than minimum"), and the position stayed in `active_positions.json` forever.

- **Dust check before sell** ([execution_engine.py](../src/ggTrader/core/execution_engine.py)): the `fixed_sl_tp` exit path now checks position value against `_DUST_THRESHOLD_USD` ($1.00) before attempting the sell. Dust positions are removed silently.

### Fixed: ATR trailing stop computed from stale backtest peak

`sig["stop_price"]` was the trailing stop from the backtest's historical position, which tracks a `peak` across many bars. For a new live entry, this peak can be much higher than the fill price, producing a stop above the current price (e.g. stop=$2974 at fill=$2358 for ETH-USD). The fallback used `atr_multiplier` (a multiplier like 3.0x) as a raw percentage, which is also wrong.

- **ATR value in signals** ([execution_engine.py](../src/ggTrader/core/execution_engine.py)): `_compute_latest_signals` now extracts the current ATR value from the precomputer and includes it as `atr_value` in the signal dict.
- **Proper stop computation**: the stop placement code now computes `fill_price - atr_multiplier * atr_value` to derive the trailing stop percentage from the actual fill price and current volatility, instead of using the backtest's stale peak-based stop. Falls back to `atr_multiplier` as a percentage only when ATR is NaN.

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
