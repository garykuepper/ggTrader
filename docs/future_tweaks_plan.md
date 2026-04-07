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
