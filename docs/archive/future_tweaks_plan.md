# Future Tweaks Plan

Last updated: 2026-04-06

## ⚠️ 2026-04-06 Update: Bookkeeping Bugs Found

A forensic investigation ([scripts/investigate_live_trades.py](../scripts/investigate_live_trades.py)) revealed that the apparent -$123 loss with 0% win rate was a **bookkeeping bug, not a real loss**. Kraken's actual realized PnL was **+$9.88** for the same period.

Root cause: `execution_engine.py` stored `sig["current_price"]` (a stale signal close) as `entry_price` in `active_positions.json` instead of the actual fill price. When positions closed, `position_closes.csv` was computed against this fake entry price, producing fake -36% to -54% per-position losses.

**The bot was modestly profitable** (+$9.88 realized over 5 days, 5W/19L). The account balance moved from $134 → $232 during this period, but ~$100 of that was a manual capital injection by the user to clear Kraken minimum order sizes — NOT trading gains.

Fixes deployed in commit 2026-04-06 (see [changelog.md](changelog.md)):

- Store real fill price as entry (root cause)
- Robust `_safe_extract_fill_price()` cascade for all order responses
- 5-minute interim polling of TSL/OCO orders so stops get recorded immediately, not 4h later
- Added missing `stoch_rsi_reversal` and `keltner_breakout` to live trader's `STRATEGY_MAP`
- New `ggt pnl-daily` command + cron + Telegram/Discord notifier so we catch real issues within 24h instead of days later

**Note on the $200 intervention floor**: It is no longer triggered by current data (real balance was always above it). Keeping the threshold but the alert is now driven by daily reports.

## Current Live Configuration

- **Research run**: `research_20260402_123248`
- **Date range**: 2023-04-03 → 2026-04-02 (3 years, 8 WFO folds)
- **Entry strategies**: 9 (original 7 + `stoch_rsi_reversal`, `keltner_breakout`)
- **Exit strategies**: 3 (`atr_trailing`, `fixed_sl_tp`, `trailing_stop`)
- **Coins selected**: 32 (29 after `MAX_COINS_PER_STRATEGY` gate)
- **WFO CAGR**: 15.46% | **YTD CAGR**: -10.66% (vs BTC -20.92%)

### New Strategies Added (2026-04-02)

| Strategy | Type | Combos | Selected By | Robustness | Notes |
| --- | --- | --- | --- | --- | --- |
| `stoch_rsi_reversal` | Mean-reversion | 12 | TRUMP-USD | 0.50 | Faster-cycling than raw RSI |
| `keltner_breakout` | Breakout | 9 | TAO-USD | 0.99 (#2 overall) | ATR-based, adapts to volatility |

### Config Tweaks Tested & Reverted

Tried these three changes together — result was worse (coins cut from 32→26, no YTD improvement). Reverted all. **Lesson: test config changes one at a time.**

- Composite weights biased toward Sortino/Calmar (0.25 each → 0.30/0.30)
- Fold consistency gate tightened (0.33 → 0.40)
- OOS robustness blend alpha increased (0.65 → 0.70)

---

## Observation Period

**Let the current configuration run for one week** (2026-04-02 → 2026-04-09) before making further changes.

**Early intervention trigger**: If account value dips below **$200**, re-evaluate immediately — don't wait for the week to end. Consider the N_SPLITS increase or other corrective action.

---

## Planned Experiments

### WFO Structure

#### Increase WFO Folds (N_SPLITS 8 → 10)

The jump from 6→8 folds was one of the key drivers behind the March improvement (CAGR -3.73% → +15.15%). More OOS data points give the robustness gate a stronger signal to distinguish genuine edge from overfitting.

Trade-offs (with 3 years of 4h data, ~6,570 bars, TEST_RATIO=3):

| N_SPLITS | Test Bars | Train Bars | Train Days | OOS Samples |
| --- | --- | --- | --- | --- |
| 8 (current) | ~597 | ~1,791 | ~299 days | 8 |
| **10 (proposed)** | ~505 | ~1,515 | ~253 days | 10 |
| 12 (aggressive) | ~438 | ~1,314 | ~219 days | 12 |

- 10 folds is the sweet spot — 2 more OOS samples without starving the training window
- 12 folds risks too-short training (~7 months), especially for slower strategies like `donchian_breakout` and `keltner_breakout`
- Compute time increases ~25% (linear with folds)

**When to try**: After the observation period, if performance is stable.

**How**:

```python
# src/ggTrader/utils/run_config.py
"N_SPLITS": 10,  # was 8
```

### Scoring & Selection

Config tweaks to how strategies are ranked and filtered. **Test these one at a time** with a research run between each to isolate effects.

#### Fold Consistency Gate (0.33 → 0.38)

Modest tightening. With 8 folds this still rounds to 3 profitable folds required. With 10 folds it would require 4 — a meaningful improvement.

#### OOS Robustness Blend Alpha (0.65 → 0.70)

Further favor out-of-sample signal over in-sample. The previous increase to 0.65 was part of the successful March tuning round.

#### Composite Metric Weights

Bias toward downside-risk metrics: Sortino (0.30) and Calmar (0.30), reduce Sharpe and ProfitFactor to 0.20 each. Rationale: crypto has fat tails, so drawdown-aware metrics are more predictive.

### Performance & Compute

#### Coarse Screening to Reduce WFO Runtime

Adding strategies increases WFO runtime linearly (7→9 entries was ~29% slower). If this becomes a problem as more strategies are added:

**Idea**: Run a coarse screening phase first — test each strategy with just 1 default param combo per coin. Only expand the full param grid for strategies that show signal (i.e., produce trades with finite Sharpe). Strategies that produce zero trades or -inf Sharpe on a coin get skipped entirely for that coin.

**Expected impact**: Could cut compute by 30-50% depending on how many strategy/coin pairs are duds. For example, `stoch_rsi_reversal` produced IS=-inf on many coins in the current run — those full grid expansions were wasted compute.

**Implementation**: Add a pre-filter step in `phase_1_per_coin_multi_strategy_wfo` that runs the coarse grid (already defined in `COARSE_ENTRY_PARAM_GRIDS`) before expanding to `DETAILED_ENTRY_PARAM_GRIDS`. Skip entries that show no signal.

### New Strategies

Ideas for future strategy additions. Lower priority — the current 9-strategy mix covers trend-following, mean-reversion, momentum, and breakout well.

#### Volume-Weighted Strategies

VWAP-based entries could add value if volume data quality improves. Currently skipped because crypto trades 24/7 with no natural session boundaries for VWAP anchoring.

#### Multi-Timeframe Confirmation

Enter on 4h signal only if daily trend aligns. Would reduce false entries in choppy markets. Requires architectural changes to support multiple timeframes in the precomputer.

#### Adaptive Position Sizing

Scale position size by inverse volatility (ATR-based) rather than fixed `PORTFOLIO_SHARE=0.10`. Higher-vol coins get smaller positions, lower-vol coins get larger. Reduces portfolio-level drawdown without changing entry/exit logic.

### Cashflow Ledger (deposits / withdrawals)

**Motivation**: TradeTracker only sees the Kraken balance snapshot — it has no concept of "deposit vs. PnL". On ~2026-04-03 the user manually added **$100** to clear Kraken minimum order sizes, and that injection is currently being counted as trading gains. This distorts:

- `balance_change_pct` / total return (currently shows +72.28%, real number is much lower once the deposit is netted out)
- Sharpe, Sortino, Calmar (computed from the contaminated equity curve)
- Max drawdown (deposit creates a fake equity step that hides the real DD)
- The 24h "balance change" line on any day a deposit lands

The unrealized PnL line on the daily report (added 2026-04-07) already tells the real story for open positions (-45.93% on $169.42 cost basis at time of writing), but the all-time health metrics are still misleading.

**Goal**: Track manual cashflows in a small ledger and net them out before computing any return/risk metric.

**Implementation sketch**:

1. **New file** `data/live/cashflows.csv` with columns `timestamp,amount_usd,note` (positive = deposit, negative = withdrawal). Append-only, human-readable.
2. **New CLI** in `src/ggTrader/cli/cmd_cashflow.py`:
   - `ggt cashflow add --amount 100 --date 2026-04-03 --note "Kraken min-order top-up"` — append a row
   - `ggt cashflow list` — print the ledger
   - `ggt cashflow remove <row_id>` — delete by index (in case of typos)
3. **Metrics integration** in `src/ggTrader/utils/live_metrics.py`:
   - New helper `load_cashflows(data_dir) -> pd.DataFrame`
   - Modify `equity_curve_from_balance()` to accept an optional `cashflows` arg and subtract the cumulative cashflow at each timestamp before returning the equity series. The result is "balance attributable to trading only".
   - All downstream metrics (`compute_sharpe_ratio`, `compute_sortino_ratio`, `compute_max_drawdown`, `compute_calmar_ratio`, daily returns) inherit the correction for free since they all consume the cleaned equity curve.
4. **Report integration** in `pnl_report_builder.py`:
   - Pass cashflows into `_gather_report_data` and on to `equity_curve_from_balance`
   - Add a "Net deposits" line to the Account Snapshot block (e.g. `Net deposits  +$100.00`) so the adjustment is visible, not hidden
   - The 24h `balance_change_pct` should also subtract any cashflows that landed inside the window
5. **Migration / seeding**: on first run after this ships, prompt-or-document the user to backfill the known 2026-04-03 +$100 entry. Just run `ggt cashflow add --amount 100 --date 2026-04-03 --note "Kraken min-order top-up"` once.

**Risks / things to get right**:

- **Tz handling**: cashflow timestamps must be UTC to match balance snapshots. The CLI should accept either `YYYY-MM-DD` (assume UTC midnight) or full ISO8601.
- **Don't double-count**: if the user later deposits via Kraken's UI and the bot reads the new balance on its next loop, that delta will appear as both a balance jump *and* a ledger entry. Document clearly that the ledger is the source of truth — anything missing from it gets attributed to trading.
- **Withdrawals**: same machinery, just negative amounts. Test at least one negative case.
- **Backfill** any historical injections discovered later — the ledger is append-anywhere by timestamp, not append-only.

**Docs to update when implemented**:

- `docs/changelog.md` — implementation entry
- `docs/live_trading_guide.md` — operator instructions for recording deposits
- This file — move to "Shipped" or delete

### Notifications & Alerting

#### Real-time Trade Fill Alerts (Telegram/Discord)

**Motivation**: The Kraken app already pings on order fills, but those notifications are context-free — no entry price, no strategy, no PnL, no exit reason. You can't tell from the Kraken alert whether a sell was profitable or which strategy fired it. The daily PnL report catches problems eventually but with up to 24h lag.

**Goal**: Push a rich Telegram/Discord message at each entry and exit so the full picture (entry, exit, hold time, $/% PnL, strategy, exit reason) is visible immediately.

**Scope (phase 1 — fills only)**:

- **BUY alert** at entry fill, formatted like:
  ```
  🟢 BUY  ETH-USD  @ $3,180.42
  strategy: psar_adx
  size: 0.062 ETH  (~$197)
  stop: $3,065  (-3.6%)  •  TP: $3,420  (+7.5%)
  open positions: 6/10
  ```
- **SELL alert** at exit fill, formatted like:
  ```
  🔴 SELL  ETH-USD  @ $3,254.10
  entry:   $3,180.42   (held 2d 4h)
  PnL:     +$4.57   (+2.32%)  ✅
  reason:  strategy_signal | stop_loss | take_profit | manual
  strategy: psar_adx
  open positions: 5/10
  ```
- ✅/❌ emoji on PnL so the notification preview tells the story without opening it.

**Implementation sketch** (`src/ggTrader/core/execution_engine.py`):

1. In `__init__`: `self.notifiers = build_notifiers_from_env()` — same pattern as the daily PnL command. Reuses existing `src/ggTrader/utils/notifier.py` (already supports Telegram + Discord, no new deps).
2. Add a small `self._notify(msg)` helper that loops `self.notifiers` and swallows exceptions so a Telegram outage cannot crash the trader loop.
3. Hook points (all data needed is already in scope at these lines):
   - **Entry fill**: after the successful `create_order` calls around `execution_engine.py:824` and `:870`. Format from the `pos` dict that's about to be written into `active_positions.json` (real fill price, stop, TP, strategy).
   - **Exit fill (strategy signal)**: at `execution_engine.py:1133`, where `exit_reason="strategy_signal"` is set. Compute realized PnL from `pos["entry_price"]` (now correct post-2026-04-06 fix) vs. fill price.
   - **Exit fill (stop/TP)**: in the TSL/OCO interim polling path (added 2026-04-06) — same formatting, with `reason` set from the order type that closed it.
4. Hold time: `datetime.now(tz) - pos["entry_time"]`, formatted as `Xd Yh` or `Yh Zm`.

**Phase 2 (optional, behind a separate env flag)**:

- **Regime-blocked entries**: alert when a signal fired but the bear-regime gate or `MAX_COINS_PER_STRATEGY` cap suppressed it. Useful to know "the bot saw something but didn't act." Hook point: inside `_execute_trade_logic` around `execution_engine.py:907`.
- **Daily running tally** appended to the end of any loop that produced trades: "Today: 3 closed, 2W/1L, +$8.40". Cheap because the daily PnL builder can already produce this.
- **Separate Telegram chat IDs** for entries vs. exits vs. blocked-signals so each channel can be muted independently. Would need `TELEGRAM_CHAT_ID_FILLS`, `TELEGRAM_CHAT_ID_SIGNALS` env vars and a small refactor of `build_notifiers_from_env`.

**Phase 3 (probably not worth it)**:

- Per-loop raw-signal alerts for every coin. On a 4h × 29-symbol cadence this is too noisy and the daily report already covers it.

**Risks / things to get right**:

- **Don't double-fire on restarts**: the 20:11–20:22 restart cycles today would have spammed entries if alerts triggered on `Loaded N active positions from state`. Only fire from the actual `create_order` success paths, never from state reload.
- **Rate limits**: Telegram allows ~30 msg/sec to different chats but only ~1/sec to the same chat. Fills are infrequent enough that this won't matter, but the `_notify` helper should still catch 429s and back off.
- **Stop/TP polling lag**: the 5-minute interim poll added 2026-04-06 means stop fills are detected within 5 min, not 4h. Alerts will inherit that latency — fine, but worth noting in the message ("detected at HH:MM, actual fill ~earlier").

**Docs to update when implemented**:

- `docs/changelog.md` — entry under the implementation date
- This file — move this section out of "Planned Experiments" into a "Shipped" note, or delete it
- `docs/live_trading_guide.md` — add the new env vars / behavior to the operator-facing docs
