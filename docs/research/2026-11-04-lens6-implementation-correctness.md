# Lens 6: Implementation correctness (does the code do what the spec says, and does live run what was tested?)

**Date:** 2026-11-04 · **Lens:** 6 of the root-cause investigation
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`) · **Type:** read-only
diagnosis, no fixes and no recommendations · **Raw artifacts:**
`docs/research/_lens6_replays.json` (hand replays), `docs/research/_lens6_verdicts.json`
(one verdict per inventory row)

Abbreviations, expanded once: WFO = walk-forward optimization (re-pick parameters on a
trailing window, test on the next unseen window); OOS = out-of-sample; PIT = point-in-time
(using only what was known on that date, e.g. index membership); IS = in-sample; DSR =
deflated Sharpe ratio; NAV = net asset value (account value); ET / PT = US Eastern / Pacific
time; BB = Bollinger Bands; RSI = relative strength index; EMA = exponential moving average;
MACD = moving-average convergence/divergence; vbb = volume-confirmed Bollinger voter.

## Summary for the owner

1. **The strategy arithmetic is right for the four rules I replayed.** I wrote four rules
   in my own plain code and ran them on real prices: the live `ensemble`, `bb_reversion`
   (both on 10 names over March to May 2025), 12-1 cross-sectional momentum (20 names,
   3 month-ends) and the month-end Treasury rule (6 months). All four produced exactly the
   same buy and sell dates as the repo code (0 mismatches). Where the written description
   is ambiguous, my hand code followed the code's reading: crossing events rather than
   levels ("touch the band", "RSI < 30"), the code's MACD divergence proxy, and its
   warm-up mask at row 55. So the replays test the arithmetic, not whether the code's
   interpretation of the specification is the intended one, and the coverage is small
   (10 names, 3 months for the daily signals).
2. **The live paper trader does not run the configuration that the backtests measured.**
   This is the main Lens 6 finding. The differences are design and configuration, not
   typos:
   - Live uses parameter values (`min_agree 2`, MACD divergence window 20, volume
     multiplier 2.0) that no walk-forward fold ever chose; two of them are not even in the
     search grid.
   - Live computes its signals at 12:45 PT from **today's unfinished bar** (price and
     volume 15 minutes before the close) and trades at once. The walk-forward backtests
     trade at the **same day's final close**, and the 2026-10-19 production replica trades
     **one day later** than the signal. Neither matches live. This is not new: the
     original replica brief assumed live uses the prior completed bar
     (`docs/research/briefs/2026-10-19-production-replica-audit.md:38-39`), and the
     project already conceded on 2026-10-19 that this assumption does not hold. The
     parity amendment says the loader "can include a same-session daily bar" and does
     not guarantee the prior completed close
     (`docs/research/briefs/2026-10-19-production-replica-parity-amendment.md:29-31`),
     and it narrows the replica's question to a prior-completed-bar model (lines
     38-42). Lens 6 confirms the amendment from the code, today's run log and the
     database.
   - Live holds at most 3 stocks, picked alphabetically when more signals fire, each sized
     at 3.3% of account value. The walk-forward backtests have no position cap and size
     each buy at 3% of the cash left, which shrinks as cash is used. The replica does model
     the cap, alphabetical fill and 3.3% sizing.
3. **Most published headline numbers were produced by code that later had bugs fixed.**
   18 rows were produced before the anchor-leak fix (2026-08-18) and/or the
   point-in-time membership fix (2026-09-23): 12 are marked "no" and 6 "partly" (the
   rule code is right; only the harness era is at issue). For those, the code at run
   time did not match its "honest out-of-sample" specification. (The verdicts file has
   14 "no" rows in all; the other 2 are the KEEP blend and the live-defaults row.)
4. **The paper snapshot's extra $320.24 is the code's cumulative dividend accrual, and
   it feeds the daily-loss halt.** Cause confirmed from code and database: the 19 rows of
   `paper_dividend_accruals` sum to $320.24, and the snapshot gap (value minus cash minus
   positions) is $320.24 on each day from 2026-10-30 to 11-04. Through October the gap
   was $155.17 to $163.08, about the $156.48 accrued before the 2026-10-30 batch. The
   accrual is not only cosmetic. `trader.py:1056` puts it into the value saved to the
   snapshot (`trader.py:1091`). The next run takes that snapshot as its day-start value
   (`trader.py:685-688`) and compares it with the broker's uncorrected value
   (`trader.py:588, 690`). So the 3% daily-loss halt now fires about $320 sooner than a
   true 3% loss: at about $102K, near 2.7%. The drawdown halt (`trader.py:670`) and
   position sizing use the broker value and are unaffected. Whether this is a defect,
   and whether Alpaca credited any of these dividends, is left to Lens 3.
5. **Not examined:** 33 of 90 rows are marked "unverifiable": 17 because I did not reach
   them in this pass, 8 because they have no code to compare (designs, decisions, data
   items), and 8 crypto rows whose code was removed on 2026-06-16 and whose run
   artifacts are mostly gone (listed under Rows not examined).

## Falsifiers

Written on 2026-11-04 before reading any strategy or paper-trading code (kept verbatim
from the scratch note made at the start; outcome added after the arrow).

- **F1 (Phase 0 #9: "live signals from a partial bar at 12:45 PT").** Overturned if the
  live signal runner drops the current-day bar before computing indicators, or if the
  backtest also signals on a price known at decision time and fills at that price.
  Confirmed if live appends today's intraday bar and signals on it while the backtest
  signals on the full daily close. → **Confirmed** (see Live path vs backtest, row 1).
- **F2 (Phase 0 #2: "max_positions 3, position_pct 0.033 caps the sleeve near 10%").**
  Overturned if paper code or `.env` give a different cap or size, or if the backtest
  uses the same cap. → **Confirmed for live** (`.env` MAX_POSITIONS=3,
  `src/ggTrader/paper/risk.py:17-18`). The WFO backtest has no cap; the replica has it.
- **F3 (live exit parity, 2026-09-28 fix).** Parity holds if the live exit rule equals the
  backtest exit rule (same RSI/BB exits, "first exit bar after entry"). Different if live
  acts only on today's bar or uses other exit parameters. → **Partly holds**: same exit
  logic and catch-up, but different parameters than the WFO picks and different fill
  timing.
- **F4 (dividend add-back near `src/ggTrader/paper/trader.py:1051-1056`).** Overturned if
  that code does not add dividends to snapshot value, or adds only dividends the broker
  did not already include in cash. → **Code adds the cumulative accrual to the saved
  snapshot value every run, and the DB gap equals the accrual total.** That saved value
  becomes the next day-start for the daily-loss halt. Whether the broker credited any of
  it cannot be tested from code (Lens 3).
- **F5 (ensemble spec vs code).** Code matches if voters, thresholds, min_agree, lookbacks
  and exit rules equal the written description; a mismatch in any rule means partly/no.
  → **Matches** (hand replay 0 mismatches), with three wording ambiguities noted.
- **F6 (backtest fills).** Phase 0 says the replica uses next-bar fills. Overturned if the
  simulator fills on the signal bar (lookahead-adjacent) or live fill timing differs from
  the assumed fill. → **Lab WFO fills on the signal bar's own close; replica fills one bar
  later at the close; live fills about 15 minutes before the close of the signal day.**
- **F7 (live params = PIT WFO picks).** The inventory row says live defaults were never
  selected. Overturned if the live values equal any fold's pick in
  `docs/research/_pit_rebaseline_20260923.json`. → **Not overturned**: no fold picked them.
- **F8 (crypto).** Crypto code was removed 2026-06-16; verifiable only if git history
  still holds the strategy code and the result artifacts. → **Code recoverable from git
  (commit 82931a40's parent), artifacts mostly absent**: crypto rows stay unverifiable,
  except one live-exit difference that is visible in the code itself.

## Steelman

The strongest case that the implementation is sound:

- **The signal arithmetic is right.** I wrote four rules again with textbook formulas
  in plain loops (not pandas rolling functions; RSI uses textbook Wilder seeding where
  the code uses a different seed). They reproduced the repo's entry and exit dates
  exactly on real prices: ensemble (live defaults, 10 names, 2025-03-01 to 2025-05-30),
  bb_reversion (same names and window), 12-1 momentum (20 names, 3 month-ends) and
  month-end Treasury (6 months). See `docs/research/_lens6_replays.json`. Limit: where
  the written rule is ambiguous (crossings rather than levels, the MACD divergence proxy,
  the row-55 warm-up mask), the hand code adopted the code's reading. These replays
  therefore test the arithmetic, not the interpretation of the specification.
- **The code is causal where it matters.** Indicators look only backward
  (`src/ggTrader/lab/strategies/indicators.py:67-149`); the weekly RSI is labelled at
  week end and forward-filled, so no same-week lookahead (`indicators.py:281-291`); IC
  weights drop forward returns that are not yet realized (`ic_weights.py:51-59`); Kelly
  sizing uses only trades closed before each entry (`src/ggTrader/lab/kelly.py:84-110`);
  momentum weights apply from the bar after the ranking date
  (`src/ggTrader/lab/strategies/momentum.py:62-73`); the WFO applies a daily PIT
  membership mask after every sweep (`src/ggTrader/lab/sweep.py:159-168`).
- **Live reuses the backtest's code.** Live imports `EnsembleSignal` itself
  (`src/ggTrader/paper/signal_runner.py:11, 85-98`), and the replica imports the live
  risk, sweep and stop functions (`src/ggTrader/lab/replica.py:14-27`), so neither can
  drift from the other by copy-paste.
- **Known live gaps were fixed in code:** missed exits are caught up
  (`src/ggTrader/paper/trader.py:104-130`), buys wait for sells to fill
  (`trader.py:870-876`), the 365-day warm-up removed indicator-seed drift
  (`signal_runner.py:60-66`).

**Does it hold up?** For the signal math, yes. It does **not** extend to "live runs what
was tested": the parameters, the bar used, the fill moment, the position cap and the
sizing base all differ between live and the WFO evidence, and the replica, the only test
built on live rules, uses a fill timing live does not have. The soundness of the code does
not make the headline backtest numbers describe the live book.

## Questions and answers

**Q1. Does each strategy's code match its written specification (entry/exit rules,
lookbacks, rebalance timing, PIT data use)?**

- `ensemble` (live): **Yes for rules.** Five voters (BB, RSI, EMA, MACD, vbb), enter on
  2 or more votes, exit when RSI crosses up through 50 or 2 or more exit votes
  (`src/ggTrader/lab/strategies/ensemble.py:167-209`), defaults as the replica brief
  lists (`ensemble.py:70-90`). Three wording gaps, none a bug: the changelog says "touch
  the lower band" and "RSI < 30" (levels) while code uses crossing events
  (`indicators.py:73-75, 130-132`); the Bollinger standard deviation convention is
  unstated (code uses sample std; population std moves 3 of about 19 entry dates in the
  replay window); "MACD divergence" is implemented as "close at its 20-bar low while the
  histogram is above its 20-bar low", bullish only (`indicators.py:182-212`), and its
  warm-up mask applies to the first 55 rows of the whole frame rather than to each
  symbol's first bars.
- Point-in-time: entries are masked from the first plan date a symbol was eligible
  (`indicators.py:30-64`, `ensemble.py:223-231`); exits are left unmasked on purpose.
- The other strategies: see the Per-strategy findings table. Unknown for 33 rows (reason
  and cheapest test under Rows not examined).

**Q2. Does a hand replay of a small window agree with the code?** **Yes, 4 of 4
strategies, 0 mismatches** (`docs/research/_lens6_replays.json`), on the arithmetic.
The hand code followed the code's reading of ambiguous wording, so this does not settle
the interpretation question in Q1. Coverage: 10 names over 3 months for the daily
signals. A synthetic check of
`simulate_signals` also confirmed the fill moment and sizing base (signal-bar close;
3% of remaining cash).

**Q3. Does the live path in `src/ggTrader/paper/` run the same logic the backtests
tested?** **No for the WFO evidence; partly for the replica.** Same signal code, but
different parameters, bar, fill moment, cap and sizing (table below). Unknown: how much
these differences change returns. Cheapest test: rebuild the 15:45 ET partial bar for the
last 30 sessions from free yfinance 5-minute data, recompute the live-default ensemble on
it and on the final close, and count differing buy/sell decisions (about 1 hour,
read-only).

**Q4. Rebalance timing.** Signal strategies decide daily; the lab fills on the signal
bar's close (`src/ggTrader/lab/simulate.py:277-290`, synthetic check); weight strategies
apply targets from the next bar (`momentum.py:62-73`); live decides and fills at 12:45 PT
on the same day. Answered.

**Q5. Crypto strategies.** Code is recoverable at the parent of commit 82931a40
(2026-06-16, "remove legacy trading/backtesting code"); most run artifacts are gone.
Classified "unverifiable, prose only", except the universal trailing stop (see Crypto).
Cheapest test to go further: check out 82931a40's parent in a scratch worktree and compare
`KeltnerBreakoutEntry` and `StochRsiReversalEntry` with their 2026-04 descriptions (about
1 hour; no data needed for a rules read).

**Q6. Phase 0 claims about code and live parity, checked.**

| Phase 0 claim | Result |
|---|---|
| Live signals come from a partial bar at 12:45 PT | Confirmed (code, run log, DB bar) |
| `max_positions 3`, `position_pct 0.033` | Confirmed (`.env`, `src/ggTrader/paper/risk.py:17-18`) |
| Snapshot includes about $320 of dividends added back (`trader.py:1050-1055`) | Cause confirmed (code + DB: gap = accrual total $320.24). Also feeds the next run's daily-loss day-start. Defect judgement and broker side left to Lens 3 |
| Replica uses next-bar fills | Confirmed. It is not what live does (live is same-day); the project already conceded this in the 2026-10-19 parity amendment (`docs/research/briefs/2026-10-19-production-replica-parity-amendment.md:29-42`) |
| 10/32 orders explained; no decision inputs archived | Not checked by Lens 6 (Lens 3) |
| Live ensemble, 3% sizing and blend were adopted on phantom numbers | Confirmed for the code side: those numbers came from code with the anchor and PIT defects |

## Per-strategy findings

| Strategy | Spec source | Code location | Rules compared | Match | Evidence |
|---|---|---|---|---|---|
| ensemble (live) | `docs/research/briefs/2026-10-19-production-replica-audit.md` §3; `docs/changelog.md` 2026-06-18, 06-23 | `src/ggTrader/lab/strategies/ensemble.py:67-209` | voters, vote thresholds, all lookbacks, exit rule, PIT mask | yes | Hand replay 0 mismatches; wording gaps listed in Q1 |
| bb_reversion | `docs/changelog.md` 2026-06-18 | `src/ggTrader/lab/strategies/signals.py:318-356`, `indicators.py:67-81` | BB 20/2.0 cross below, exit at mean | yes (rules) | Hand replay 0 mismatches; "touch" vs cross wording |
| rsi_reversion | `docs/changelog.md` 2026-06-18 | `indicators.py:118-138` | RSI 14, 30 entry, 50 exit | yes | Inside ensemble replay |
| volume_bb_reversion | `docs/changelog.md` 2026-06-23 | `indicators.py:237-263` | BB cross + volume > 2x 20-day mean | yes | Inside ensemble replay |
| ema_cross | `docs/changelog.md` 2026-06-16 era | `indicators.py:141-149` | 20/50 EMA cross | yes (rules) | Read; standard formula |
| macd_divergence | `docs/changelog.md` 2026-06-23 | `indicators.py:182-212` | divergence definition, exit | partly | Bullish only; loose divergence proxy; frame-level warm-up mask |
| mtf_reversion | `docs/changelog.md` 2026-06-23 | `indicators.py:281-324` | weekly RSI timing | yes (rules) | Week-end labels, forward-filled; no lookahead |
| ensemble_conviction | `docs/changelog.md` 2026-06-22 | `ensemble.py:280-441` | 1%-4% sizing by strength | yes | Read |
| ensemble_ic | `docs/research/2026-09-23-pit-rebaseline-core-nogo.md` | `src/ggTrader/lab/strategies/ic_weights.py:44-87` | IC leak guard | yes | Read; causal |
| ensemble_kelly | same | `src/ggTrader/lab/kelly.py:84-175` | Kelly causality | yes | Read; causal |
| xs_momentum / dual_momentum | `docs/research/2026-10-19-production-replica-audit-nogo.md` | `src/ggTrader/lab/strategies/momentum.py:13-85` | 12-1 score, top-N, next-bar targets, absolute filter | yes | Hand replay 3/3 month-ends identical |
| month_end_treasury | `docs/research/2026-09-24-month-end-treasury-sleeve-nogo.md` | `src/ggTrader/lab/strategies/month_end_treasury.py:80-114` | entry/exit calendar | yes | Hand replay 6/6 months |
| overnight_gap | `docs/roadmap.md` (strategy table) | `indicators.py:84-115` | fill moment | partly | Gap known at the open, filled at the close |
| pead / insider_cluster_buy / congress_trades / short_interest / short_volume_ratio / headline_sentiment / index_deletion_fade | their 2026-07 NO-GO reports under `docs/research/` | `src/ggTrader/lab/strategies/pead.py:45-90`, `insider_cluster.py:36-61`, `congress_trades.py:31-34`, `short_interest.py:40-59`, `short_volume_ratio.py:82-89`, `headline_sentiment.py:64-100`, `index_deletion.py:84-127` | PIT availability gates only | partly | Gates exist in current code; not replayed; event dates in July runs sat on the day-shifted tape (Lens 1) |
| production replica | `docs/research/briefs/2026-10-19-production-replica-audit.md` §3 | `src/ggTrader/lab/replica.py:35-290` | rules vs live | partly | Rules match live; timing does not, as the 2026-10-19 parity amendment already states (lines 29-31) |
| crypto entries/exits | `docs/archive/roadmap-pre-lab.md` | removed; 82931a40 parent | see Crypto | unverifiable / partly | Code in git only |

## Live path vs backtest

What each path does, with the evidence. "WFO" is the lab walk-forward (`run_wfo` +
`simulate_signals`); "replica" is `src/ggTrader/lab/replica.py`.

| # | Item | Live paper trader | WFO backtest | Replica |
|---|---|---|---|---|
| 1 | Bar used for today's signal | Today's **partial** bar at 12:45 PT: fetch end = today + 1 day (`src/ggTrader/data/live/yfinance_loader.py:51-52`; `signal_runner.py:71-77`); today's log requested `2026-08-11 -> 2026-11-05`; the DB holds a partial 2026-11-04 bar (HAS close 122.255, volume 978,471; AAPL volume 64.0M vs 89.0M later shown by yfinance). Code comments disagree: `trader.py:60-66` says "most recent completed daily bar", `trader.py:109-110` and `signal_runner.py:102-104` say partial | Final daily close | Prior day's final close (lag 1, `replica.py:178-179`) |
| 2 | Fill moment and price | Market order about 15:45 ET, same session | Signal bar's own close + 5 bp slippage (`simulate.py:277-290`, `src/ggTrader/lab/data.py:21-22`; synthetic check) | Next bar's close + 5 bp (`replica.py:250`) |
| 3 | Parameters | Class defaults: min_agree 2, min_agree_exit 2, divergence 20, vol_mult 2.0 (`signal_runner.py:85-86`, `ensemble.py:70-90`) | Re-picked per fold from a grid that holds min_agree 2-4, divergence 10 only, vol_mult 1.5 only (`ensemble.py:123-141`); all 17 PIT folds picked min_agree 3-4 | Live defaults (`replica.py:81`) |
| 4 | Position cap and order | 3 names (`.env`; `risk.py:69-71`), buy list sorted A to Z (`signal_runner.py:100`), loop stops at the cap (`trader.py:880-905`) | No cap | 3 names, A to Z (`replica.py:210-225`) |
| 5 | Size per buy | 3.3% of account value (`risk.py:18, 73-75, 119-135`) | 3% of remaining cash (`data.py:29`, `simulate.py:249, 282`; synthetic check: 300 then 291). `data.py:28` says this "matches" live; the sizing base differs | 3.3% of NAV (`replica.py:220`) |
| 6 | Eligibility history | min_history_bars 60 (`signal_runner.py:85`) | LabConfig default 400 (`src/ggTrader/lab/strategy.py:28`) unless overridden | 60 (`replica.py:81`) |
| 7 | Exits | Today's partial-bar exits plus catch-up of any exit since the last buy (`trader.py:104-130, 642-647`), sold at the catch-up day's price | First exit bar after entry, at that bar's close | Same catch-up rule (`replica.py:203-208`), at next close |
| 8 | Volume input to the vbb voter | Partial-day volume vs a 20-day mean of full days | Full-day volume | Full-day volume |
| 9 | SPY cash sweep | On (`.env` CASH_SWEEP_ENABLED=true; `trader.py:781-835, 937-972`) | Not modelled | Modelled with the live functions |
| 10 | Catastrophe stop -25% | On (`.env`; `trader.py:718-758`) | Not modelled | Modelled (`replica.py:200-202`) |
| 11 | Drawdown 15% and daily-loss 3% halts | On (`trader.py:655-693`, `risk.py:20-21`) | Not modelled | Modelled (`replica.py:185-191`) |
| 12 | ML entry gate | Off unless ML_GATE_ENABLED is truthy (`src/ggTrader/paper/feature_gate.py:22-26`); host `.env` does not set it | Not modelled | Not modelled |
| 13 | Untradable holdings (AVB) | Excluded from slots and sells (`trader.py:605-620`) | n/a | Not modelled |
| 14 | Dividends | Prices not adjusted live. The cumulative uncredited-dividend accrual is added to the value saved in the snapshot (`trader.py:1056, 1091`), and that snapshot becomes the next run's daily-loss day-start (`trader.py:685-690`). Tradable cash, sizing and the drawdown halt use the broker value | Adjusted closes (total return, per the replica brief §3) | Same as WFO |
| 15 | Trailing / ATR stops | None | Optional `atr_mult`/`ts_stop` in sweeps (`simulate.py:201-241`) | None |

**Other findings from the live path:**

- **Partial bars in the research tape are overwritten only by accident.** At the 12:45 PT
  run (20:45 UTC) the previous day's bar, stamped 00:00 UTC, is about 1.86 days old. That
  exceeds the 1.5-day freshness limit, so the symbol is judged stale
  (`src/ggTrader/data/live/cached_yfinance_loader.py:346-349`). But the incremental
  refetch starts at that bar's date plus one day, which is today, so on its own it never
  rewrites the previous day's partial bar. One batch start serves every stale symbol: it
  is the minimum over symbols (`cached_yfinance_loader.py:349, 371-377`). EA's last bar is
  2026-08-10, so every run's start is pulled back to 2026-08-11 (log line
  `1d 2026-08-11 -> 2026-11-05`), and that refetch overwrites the earlier partial bars.
  Spot check: AAPL and HAS bars for 2026-10-27 to 11-03 equal yfinance's final values.
  Without such a stale symbol, each 12:45 PT partial bar would stay in the tape (the
  benchmark refresh comments at `signal_runner.py:30-41` describe the same failure for
  SPY). Tape impact is Lens 1.
- **Dividend add-back: cause confirmed from two sources; defect judgement open.** Code:
  `corrected_value` = broker value + split delta + cumulative accrual
  (`trader.py:1045-1056`), saved as the snapshot value (`trader.py:1089-1094`). Database:
  `paper_dividend_accruals` holds 19 rows summing to $320.24. The gap between snapshot
  value and cash plus position values is $320.24 on 2026-10-30, 11-02, 11-03 and 11-04.
  From 2026-10-01 to 10-29 it ranges from $155.17 to $163.08 (21 days, most near $156.49),
  against $156.48 accrued before the 2026-10-30 batch. The remaining few-dollar daily
  wobble in October is not explained here. **Effect on a live control:** the next run
  takes `day_start = get_latest_snapshot()` (`trader.py:685-688`;
  `src/ggTrader/paper/persist.py:186-192`), which includes the accrual, and compares it
  with the broker's uncorrected `portfolio_value` (`trader.py:588, 690`). The 3%
  daily-loss halt therefore fires on about $320 less real loss. At the 2026-11-03 value
  of about $102,009 that is roughly 2.7% instead of 3%. The drawdown halt
  (`trader.py:670`) and position sizing use the broker value and are not affected.
  Whether this is a defect, and whether Alpaca credited these dividends, is left to
  Lens 3.
- **Stability-aware live parameters have no consumer.** `select_live_params`
  (`src/ggTrader/lab/wfo.py:922`) is never read by the live path, which always uses class
  defaults.

## Historical runs and code at run time

Headline rows produced before 2026-08-18 (anchor fix 2eafab1) or 2026-09-23 (PIT
membership fix c9d1abe) ran on code with known defects relative to an honest
out-of-sample specification, so they are marked "no" (or "partly" where the rule code is
right and only the harness era is the issue). Gate counts before 2026-10-19 also predate
the DSR units fix 09f9f40. The 2026-06-20 "validated" ensemble was the 3-voter version
(changelog 2026-06-18), not today's 5-voter default (`ensemble.py:37-39`). Row-by-row
verdicts are in `docs/research/_lens6_verdicts.json`.

## Crypto

- The legacy crypto engine and entry/exit classes were removed in commit 82931a40
  (2026-06-16); they can be read with `git show 82931a40^:<path>`.
- **Universal trailing stops: the code does what the inventory claim says, but live
  exits differ from the backtested exits (verdict: partly).** The removed live engine
  (82931a40^, src/ggTrader/core/crypto_execution_engine.py lines 713-731) places an
  exchange trailing stop on every buy and converts older WFO selections of `fixed_sl_tp`
  into trailing stops at placement, as the inventory row states. Two live-versus-backtest
  differences follow. First, coins whose WFO winner was `fixed_sl_tp` were backtested
  with a fixed stop and take-profit but traded live with a trailing stop. Second,
  `atr_trailing` exits are placed as a fixed trailing amount: the stop percent comes
  from the entry-time ATR and is turned into an absolute offset at placement
  (crypto_execution_engine.py:518-536, `trailingAmount`). The backtest moves the stop
  every bar as peak high minus multiplier x that bar's ATR, ratcheting up only. I
  verified this from git history: 82931a40^, src/ggTrader/indicators/signals.py:18-63,
  `_atr_trailing_stop_long_ohlc_touch_2d_numba`, called from `AtrTrailingExit`
  (82931a40^, src/ggTrader/indicators/strategies.py:1047 onward).
- All other crypto rows: unverifiable, prose only. The one raw artifact found
  (`results/research/research_20260511_193048/wfo_stats_snapshot.json`) was not opened by
  this lens.

## Rows not examined

33 rows are "unverifiable". 8 are crypto rows (see Crypto). 8 are stock rows with no
code to compare, or code that is gone or never traded: pre-FOMC intake rejection,
event-timing decision, forward-paper standard, two-symbol sweep design, paper-freeze
readiness, the OHLCV day-shift data item, retail_attention (never traded) and the
2026-05-02 legacy pipeline run. The remaining 17 were not examined in this pass because
the scope order (ensemble, live path, replays) used the time: conviction_bb, idio_vol,
max_effect, pairs_stat_arb, leveraged_rotation, leveraged_trend, wfo_tournament, equity
diversification sleeves (blend), ML entry gate study, DSR gate correction,
commodity_trend, fomc_drift (two rows), fx_hedge_overlay, treasury_curve and
cross_asset_sleeve (two rows). Cheapest test for each: read the
strategy file against its NO-GO report's method section and hand-replay one month on
three symbols with the same script pattern used here (about 20-40 minutes per strategy,
read-only).

## Verification

Commands run (all read-only; Python that executed repo code ran under
`scripts/run_capped.sh`, 6 GB cap, no 137 exits):

- Inventory selection: 144 rows loaded, 90 selected (lens 6 owner or live_deployed true).
- Hand replays: `scripts/run_capped.sh python <scratch>/replay.py <scratch>/replays_raw.json`
  → `ensemble match=True mismatches=0`, `ensemble (population std) match=False
  mismatches=3`, `bb_reversion match=True`, `xs_momentum match=True`,
  `month_end_treasury match=True`. Data: SELECT-only, read-only psycopg2 session on
  `ohlcv` (venue yfinance, 1d). I did not use `fetch_stock_ohlcv`, because it writes
  missing symbols back to the database (`src/ggTrader/lab/data.py:57-77`).
- Fill semantics: `scripts/run_capped.sh python <scratch>/fill_check.py` → buy A on the
  signal bar at 13.0065 (close 13 + 5 bp), size 300 (3% of 10,000); buy B next bar size
  291 (3% of 9,700 remaining cash).
- DB reads (SELECT only): recent AAPL/HAS/WLK bars; `paper_dividend_accruals` totals and
  daily totals; `paper_snapshots` value minus cash minus summed position market values
  for 2026-10-01 to 11-04.
- yfinance MCP read of AAPL 1-month daily history, to compare final values with the DB.
- Run log `paper_trade_20261104.log` (fetch window `2026-08-11 -> 2026-11-05`,
  `251 rows x 503 symbols`); crontab line `45 12 * * 1-5 .../paper_trade.sh`.
- `.env` read for four keys only (MAX_POSITIONS=3, CASH_SWEEP_ENABLED=true,
  CATASTROPHE_STOP_ENABLED=true; ML_GATE_ENABLED absent). The container's environment
  was not read; Lens 3 owns that.
- `git log`/`git show` on 82931a40 and the fix commits 2eafab1, c9d1abe, 09f9f40,
  fe5c4a7, 17b0971.
- Gate: `.venv/bin/python scripts/check_lens_report.py
  docs/research/2026-11-04-lens6-implementation-correctness.md` → exit 0.

Unverified: the size of the live-vs-backtest return effect; the container's runtime
environment; Alpaca's dividend records; the 17 rows listed above; crypto rules.

## Done checklist

- [x] ✅ Q1 code vs spec for `ensemble` (rules, lookbacks, timing, PIT): answered, matches
- [x] ✅ Q1 code vs spec for replayed strategies (bb_reversion, xs/dual momentum,
  month_end_treasury): answered, matches
- [x] ✅ Q1 code vs spec for voters, ensemble_ic, ensemble_kelly, ensemble_conviction,
  macd, mtf, overnight_gap: answered by reading (not replayed)
- [x] ✅ Q1 PIT gates of event strategies (pead, insider, congress, short interest, short
  volume, headline sentiment, index deletion): answered "partly" (gates present, not
  replayed)
- [ ] ❓ Q1 for 17 rows listed under Rows not examined: unknown (not reached;
  cheapest test given there)
- [x] ✅ Q2 hand replay for ensemble plus three other strategies: answered, 0 mismatches
- [x] ✅ Q3 live path equivalence: answered, not a match to the WFO evidence; partly to the
  replica (15-row table)
- [ ] ❓ Q3 size of the return effect of the live differences: unknown (no archived 12:45
  inputs; cheapest test: rebuild 15:45 ET bars from yfinance 5-minute data for 30
  sessions, about 1 hour)
- [x] ✅ Q4 rebalance and fill timing: answered
- [x] ✅ Q5 crypto: answered as unverifiable except the trailing-stop conversion
- [x] ✅ Q6 Phase 0 code/parity claims: four confirmed, two left to Lens 3
- [x] ✅ Dividend snapshot gap mechanism: answered from code and DB; broker side left to
  Lens 3
- [x] ✅ Verdicts written for all 90 rows (`docs/research/_lens6_verdicts.json`)
