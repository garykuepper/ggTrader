# Brief: does the strategy we actually run beat an appropriate passive alternative?

**Status:** pre-registered 2026-10-19, before any result existed. The owner approved
every *(planner's call)* below as written on 2026-10-19.

The decisions below come from the 2026-10-19 audit conversation, where the owner
adopted "Set B" plus four additions as the answers to round-1 Q1–Q5 and Q7. Items
marked *(planner's call)* were not discussed explicitly, and the owner should check
them before this brief is committed. Once it is committed, its numbers are the bar,
and changing them requires a new dated brief.

## 1. The question, and the decision it feeds

- **Question.** Does the paper trader *as deployed* (S&P 500 ensemble stock picks
  plus the SPY cash sweep) improve on an appropriate passive alternative? The
  improvement has to be statistically significant and has to survive tax.
- **Decision.** What happens to the stock-picking sleeve when the parity freeze lifts
  (~2026-12-29): keep it, shrink it, or retire it. This audit produces a
  recommendation only. Any live change is a separate owner decision, made through
  the `ggtrader-deploy` skill.
- **Real money is not affected.** The 2026-09-30 70/30 SPY/SSO decision stands; §6
  only puts bounds on its tax number.

## 2. Why the existing evidence can't answer it

These findings are from 2026-10-19 and were verified in code or JSON.

- **KEEP is a tie with SPY.** KEEP (2026-09-24) earned a CAGR of 12.963% against
  SPY's 12.959% (`_core_spy_blend_20260924.json`). Its Sharpe edge of 0.80 vs 0.78
  comes only from lower volatility.
- **KEEP tested different parameters.** It used walk-forward-picked parameters: in
  all 17 folds of `_pit_rebaseline_20260923.json`, the folds chose min_agree 3–4,
  divergence 10 and vol_mult 1.5. Live runs the `EnsembleSignal` defaults
  (`ensemble.py:70-84`): min_agree 2, divergence 20, vol_mult 2.0. min_agree 2 was
  in the grid and never won. Divergence 20 and vol_mult 2.0 are not in the grid at
  all.
- **Fill timing differs.** The backtest fills at the signal bar's own close
  (`simulate.py` `from_signals`, no shift). Live builds signals from the prior
  completed bar (`trader.py:60-66`) and fills at about 12:45 PT the next session.
- **Live caps positions and fills them alphabetically.** Live fills buy candidates
  A→Z until `MAX_POSITIONS=3` is reached (`signal_runner.py:100`,
  `trader.py:896-901`). The backtest has no cap.
- **The rest of the KEEP method is weak.** KEEP scaled the core's returns linearly
  instead of re-simulating, and it had no tax model. The deflated-Sharpe (DSR) gate
  is mis-scaled (`gates.py`) and passes almost everything, so no past "x/17 gates"
  count shows significance.

## 3. Design: the production replica

The replica is an event-loop backtest of the live trader's portfolio rules. It calls
the live decision functions themselves, so the rules cannot drift:
`cash_sweep.compute_sweep_buy`, `cash_sweep.compute_sweep_sell_for_funding`,
`risk.RiskGuard` and `catastrophe_stop.find_catastrophe_stops`.

- **Signals.** `EnsembleSignal` with the live defaults:
  - 5 voters, min_agree 2, min_agree_exit 2
  - BB 20/2.0; RSI 14/30 with exit at 50; EMA 20/50; MACD 12/26/9
  - divergence 20; volume 20/2.0
  - `LabConfig(min_history_bars=60)`

  Entries use a point-in-time S&P 500 mask built from monthly plans
  (`rebalance_dates` + `eligible_at`).
- **Timing.** Decisions for session *t* use signals from bar *t−1*. Fills happen at
  the close of *t*. *(Planner's call: there is no intraday data, so the close stands
  in for the ~15:45 ET market order.)*
- **Portfolio rules (live values):**
  - Each position is 3.3% of NAV, with at most 3 positions, filled A→Z, under a
    5% concentration cap.
  - A held name exits on an exit signal on any bar since its entry, matching live's
    catch-up behaviour.
  - Catastrophe stop at −25% against cost basis.
  - A 15% drawdown from the running peak halts all trading, sweep included. A 3%
    loss against the prior close halts trading for that day.
  - SPY sweep: 5% cash reserve, 8% buy trigger, $500 minimum order. Funding sells
    happen before buys.
  - Fractional shares.
- **Costs.**

  | | Stocks, per side | SPY, per side |
  |---|---|---|
  | Primary | 5 bp | 1 bp |
  | Sensitivity runs | 3 bp, 10 bp | 1 bp |

- **Cash.** In the primary arm, cash earns the 3-month T-bill rate: FRED `TB3MS`,
  forward-filled and compounded daily. That is generous to the strategy, since paper
  cash earns 0%. A 0%-cash arm runs as a sensitivity check.
- **Data.** The DB tape, via `load_ohlcv`. Adjusted closes give total return for
  stocks and SPY alike.
- **Windows (pinned):**
  - **Primary:** 2021-01-31 → 2026-04-30, the same span as KEEP and the PIT
    re-baseline.
  - **Holdout:** 2010-01-31 → 2020-12-31. As far as the record shows, no recorded
    parameter choice used it *(planner's call)*.
  - **Warm-up:** each window loads 420 calendar days before its start.
  - **Survivorship caveat for the holdout:** yfinance lacks most delisted names,
    which flatters a dip-buying strategy. A holdout *fail* is therefore conclusive,
    and a *pass* is weak evidence.
- **Benchmarks:**
  1. SPY buy-and-hold. This decides the verdict.
  2. An exposure-matched passive: each day, the replica's stock + SPY weight is held
     in SPY and the rest sits in cash at the same cash rate. This is diagnostic: it
     measures selection skill.
- **Diagnostic arm.** Signal lag 0, i.e. the old backtest's same-close fill. It
  measures what the one-day lag costs and is not a pass criterion.

## 4. Pre-registered pass bar

**GO** requires all of the following, on **both** windows, in the primary arm:

1. **Statistical.** The Newey-West (5 lags) t-stat of the mean daily return over SPY
   is at least **3.0**. *(Planner's call: 3.0 is roughly a one-sided 5% test
   Bonferroni-adjusted across the 44 studies on the roster. It is the "t≈2, raised
   for multiple testing" the owner agreed to.)*
2. **Economic, before tax.** CAGR minus SPY's CAGR is at least **+1.0 pt/yr**.
3. **Economic, after tax.** After-tax CAGR minus SPY's after-tax CAGR is at least
   **+1.0 pt/yr** at the 24%/15% bracket, with everything liquidated at the end
   (A23 conventions). This is computed only if 1 and 2 pass on both windows.

Anything else is a **NO-GO**. Item 3 is skipped when item 2 fails because tax can
only widen a pre-tax shortfall here: the sleeve realizes mostly short-term gains
(71% of live round trips were winners), while SPY defers its gains.

**Stop conditions.** No verdict is issued until each of these is resolved:
- The parity check explains fewer than 100% of live strategy orders.
- The replica traded within 5 bars of a price-tape artifact.
- Holdout coverage is under 70% of point-in-time member-days. In that case the
  holdout is **INCONCLUSIVE**, so GO is impossible.

## 5. Decision mapping (a recommendation to the owner)

*(Planner's call.)*

- **NO-GO.** Recommend retiring the stock sleeve at the freeze lift. The paper book
  becomes the SPY sweep alone, or the 80/20 shadow if that separately passes its own
  12-month T-bill-excess rule in `docs/next_steps.md`. The real-money default does
  not change.
- **GO.** Recommend keeping the sleeve at its current size for the next parity
  window. The 2026-09-30 rule stays: no stock picks with real money until a forward
  test confirms the result.

## 6. Secondary work in the same plan

- **Fix the DSR gate before any walk-forward runs.** Mark past gate counts void in
  the snapshot.
- **xs_momentum and dual_momentum.** Re-run on the corrected harness: pinned PIT
  sp500, 17 folds, through `run_core`. These are weights strategies, which the
  pre-08-18 cash-drag bug hurt. Bar: §4 items 1–2 against SPY on the out-of-sample
  daily returns.
- **fomc_drift on the corrected tape.** Universe `fomc_treasury`,
  2012-01-01 → 2026-04-30. The NO-GO stands unless the net-of-cost event t-stat is
  at least 3.0 **and** the walk-forward efficiency (WFE) is at least 0.50. Even
  passing would make it an overlay candidate that needs its own brief, not a
  deployable strategy.
- **A23 tax bounds.** Re-run the after-tax margin at today's bills (+2.5 pt) with
  SSO and BIL distributions taxed at the ordinary rate. The current model taxes them
  all at the long-term rate. This is information for the owner's pre-entry decision;
  the terms stay unchanged.

## 7. Out of scope

- **Anything live.** No change to `src/ggTrader/paper/`, `.env`, docker or the live
  container before ~2026-12-29. That includes the sweep's same-day sell/rebuy churn
  and a loader guard against split/dividend seams; both are follow-ups after the
  freeze.
- **idio_vol and PEAD re-runs.** They trail SPY by only 0.01–0.02 Sharpe, so they
  cannot reach §4 even if the bugs flipped them.
- **The voter-set re-ablation.** Worth running only if §4 returns GO.
- **The 80/20 on T-bill cash.** Already measured (`next_steps.md:215-219`), and it
  has its own pre-registered 12-month rule.
