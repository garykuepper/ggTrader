# Changelog

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
