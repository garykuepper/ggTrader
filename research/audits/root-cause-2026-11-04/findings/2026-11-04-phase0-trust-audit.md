# Phase 0 Trust Audit (stocks only)

**Date:** 2026-11-04 · **Author:** senior-quant Opus subagent (read-only), saved by the lead
**Status:** UNREVIEWED subagent output. Items marked "prose" were taken from reports, not
recomputed. The independent second-model review in Phase 3 of the
[investigation design](../../../../docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md)
has not happened yet. Crypto was not covered.

## Bottom line

No real stock-picking edge over SPY was found. The sound tests are mostly reliable NOs, and
the deployed design could not show an edge even if one existed. The live book is about 84%
SPY, 5% cash and 3 stocks at 3.3% each. The audited replica has the same shape (mean
exposure 9.4% stocks, 85.3% SPY). The early "beat SPY" numbers (1.12, 0.99, 16% CAGR) were
real bugs, since corrected. Reaching 2x SPY needs roughly +10 to 15 points a year of alpha,
or about 2x leverage. The best free-data long-only large-cap effects found after 1994 are
+0.1 to +0.8 points a year.

## Trust table

"Verified" = the auditor recomputed or read the number from the raw JSON or DB.

| Result | Claimed number | Classification | Reason | Verified how |
|---|---|---|---|---|
| Production-rule replica (fixed-default ensemble + SPY sweep) | 14.19% vs SPY 14.83%, NW t -0.69 | Reliable NO, high trust | Pinned window 2021-01-31 to 2026-04-30, PIT membership, 5 bp costs, next-bar fills, total-return SPY. Defaults tuned on the same window, which favours the strategy, and it still lost. | Verified `_production_replica_results.json` |
| Stock sleeve vs exposure-matched passive | +0.265 pp/yr, t 0.20 | Inconclusive about small edges | Sleeve is 9.4% of the book; implied alpha about +2%/yr at IR 0.09 needs centuries to detect | Verified, `excess_vs_exposure_matched` |
| Live parity | 10/32 orders explained | Trust the "UNVERIFIED" label | 19 manual liquidations, no archive of decision inputs | Prose (`_replica_parity_20261019.json` not opened) |
| Paper since 2026-06-24 | +1.39% vs SPY price +4.63% | Skeptical either way | 4.4 months of config churn; snapshot value includes about $320 of dividends the code adds back that the broker never credited (`paper/trader.py:1050-1055`) | Verified, `paper_snapshots` and `ohlcv` |
| Paper frozen window 09-30 to 11-04 | +1.96% vs SPY +0.2% | Noise | 7 round trips on 3 slots | Verified, DB `paper_trades` |
| Historical PIT core WFO | Sharpe 0.59, CAGR 3.1% vs SPY 0.78 / 13.0% | Reliable NO, medium trust | "17/17 gates" predates the DSR units fix (`09f9f40`, 10-19) so the count means nothing; low exposure makes CAGR uncomparable | Verified `_pit_rebaseline_20260923.json` |
| ensemble_ic / ensemble_kelly | 0.41 / 0.50 | Reliable NO | Same machinery, with regime halts | Verified, same JSON |
| Core+SPY blend "KEEP" | 0.80 vs 0.78 | Skeptical (superseded) | 0.02 Sharpe gap is noise; selected parameters; same-close fills | Verified `_core_spy_blend_20260924.json` |
| Old headlines 1.12 / 0.97 / 0.99 / "16.2% CAGR" | Beat SPY | Void | Built before fixes for tz day-shift, SPY duplicate rows, anchor leak, PIT leak; window drifted | Verified `_anchor_fix_reproduction_results.json`, `_rebaseline_corrected_tape_20260822.json` |
| xs_momentum (dual_momentum is the same portfolio) | +4.37 pp/yr, t 0.76 | Inconclusive | Fixed anchor parameters in 17/17 folds; tracking error 12.4%; one AI mega-cap regime | Verified `_audit_reruns_20261019.json` |
| French momentum screen | Post-1994 Big +0.08, Small +0.20 pp/yr | Reliable NO for long-only (proxy scope) | t 3.05/3.11 vs Holm threshold about 3.2; tests value-weighted top-30% portfolios, not a concentrated top-N rule | Verified `_momentum_french_screen_results.json` |
| Factor screen (profitability, value, investment, combo) | Best t 1.34; post-1994 <= +0.83 pp | Reliable NO (long-only big-cap scope) | 63 years survivorship-free data, Holm correction, cost haircuts | Verified `_factor_screen_results.json` |
| Net issuance | t 0.26, +0.48 pp | Reliable NO | Same method | Verified |
| FOMC drift | Clustered net t 0.98 | Reliable NO | First stats counted each day three times; corrected | Prose + snapshot |
| Month-end Treasury | Fails vs SPY funding leg | Reliable NO | Judged against the right comparator | Prose |
| Cross-asset 80/20 "shadow GO" | Sharpe 0.973 vs 0.896 | Skeptical for the owner's goal | CAGR 13.61% below SPY 14.70%; zero risk-free rate in Sharpe (BIL shows Sharpe 12); gold-driven | Verified `_cross_asset_sleeve_results.json` |
| A23 70/30 SPY/SSO | After-tax 12.0% vs 10.4%; +0.77 to 0.80 at higher funding | Trust, medium | Leveraged beta, not alpha; margin depends on funding cost | Verified `_static_leverage_aftertax_results.json` |
| Century leverage stress | 30% SSO: -91% drawdown 1929-32 | Trust, medium | Synthetic Shiller data | Prose |
| July event studies (PEAD, insider, congress, short interest, deletion, MAX) | Historical NO | Direction reliable, size not | Two known bugs favoured these strategies and they still lost | Prose + memory |
| Leveraged-ETF rotation / trend | NO | Reliable NO | Lost to holding the same leveraged ETF | Prose |
| Ensemble levers (ML gate, exits, voters) | Historical NO | Inconclusive | Judged against the phantom 1.12 baseline | Prose |
| Pairs / headline sentiment | NO / provisional | Inconclusive | Pairs has an open cache anomaly; sentiment had only 3 usable folds | Prose |

## Root causes, ranked

**The evidence is negative and the target is structurally hard**
1. A 2x-SPY goal cannot be met by any effect measured. Post-1994 net long-only excesses:
   +0.08 / +0.20 (momentum), +0.83 / -0.62 / 0.00 / -0.56 (factors), +0.48 (net issuance).
   Even momentum before 1994 was only +3.1 pp.
2. The deployed design caps the stock sleeve at about 10% (`max_positions 3`,
   `position_pct 0.033`). A sleeve beating SPY by 10% a year adds under 1 point to the whole
   account. "Marginal vs SPY" is built in, and tests do not find an edge inside that 10%.

**Method and process errors (mostly fixed, but they drove months of decisions)**
3. Results were built on contaminated machinery: five known bugs (bars stamped a day early,
   SPY duplicate rows, anchor leak `2eafab1`, PIT leak `c9d1abe`, DSR units `09f9f40`).
   Every gate count before 10-19 is void. The live ensemble, 3% sizing and the blend were
   adopted on phantoms (1.12 to 0.59).
4. WFO mostly tests the fallback parameters, not selection. After the DSR fix, passing a gate
   needs in-sample Sharpe near 2.5. Momentum used anchor parameters in 17/17 folds. The old
   circuit breaker halts 87.5% of runs even for a true Sharpe 1.0
   (`2026-11-02-wfo-policy-synthetic.md`). A WFO NO-GO is weak evidence of no edge.
5. Pass bars are unreachable on the project's own 5-year data (t of at least 3 needs IR near
   1.3). Only long-history French screens can discriminate, and they test proxy portfolios.
6. Two GO/KEEP calls used Sharpe with a zero risk-free rate, which does not match a
   raw-dollar goal (cross-asset 80/20, blend KEEP).
7. No true holdout. Designs were tuned on 2021-26 and "validated" on 2021-26. The 2010-2020
   holdout is unusable because of bad price data (TIE/MEE/CPWR).

**Implementation and live-parity errors**
8. Most paper shortfall is config churn, not signal. By era from the DB: 06-24 to 08-21 about
   60% idle cash, -1.9 pp vs SPY; 08-21 to 09-30 blend-to-core revert, MidCap orphans and
   manual liquidations, -2.6 pp; frozen window since 09-30, +1.4 pp (noise).
9. Live and backtest decisions cannot be reconciled: 10/32 orders explained, no decision
   inputs archived, live signals from a partial bar at 12:45 PT, AVB stuck in the book, paper
   value not the broker's figure.
10. The backtest ignores taxes. Live holds last about a week so every gain is short-term.
    (Inferred; not run.)

## What is done right

- Pinned eval windows, raw JSON next to each report, a pre-registered brief with a closure
  amendment (the stopped holdout was not relabelled clean).
- Fixed-rule replica: next-bar fills, costs, T-bill cash, total-return SPY, exposure-matched
  control.
- Bias fixes traced to commits: anchor, PIT, DSR units, event timing, day-clustered FOMC stats.
- Long-history survivorship-free French screens with Holm correction and a post-publication
  return bar.
- Each sleeve benchmarked against its funding leg.
- The 2026-11-01 self-audit already flags most of these concerns
  (`2026-11-01-research-process-audit.md`).

How large firms differ: IR is roughly IC x sqrt(independent bets). Three concurrent long-only
positions in crowded S&P 500 names on daily technical signals have a tiny IR ceiling whatever
the skill. Firms get edge from thousands of names, long-short, point-in-time fundamentals, a
proper security master, and cheap financing.

## Open unknowns only a re-run or more data could settle (none were run)

| Unknown | Rough cost | What would change the conclusion |
|---|---|---|
| Stock sleeve alone vs equal-weight S&P 500 (RSP) from replica sleeve P&L | 15-30 min (needs a sleeve return series; the curves CSV has only weights) | t > 2 would mean sizing, not selection, limits returns |
| After-tax replica with short-term lots (TaxLotBook) | 2-4 h | Only a strongly positive result would matter |
| French top-decile big-cap momentum, post-1994 net | 15 min | Above +2 pp narrows the momentum NO scope |
| Corrected-tape re-run of the July event studies | 1-3 h each | A sign flip to positive (unlikely) |
| 2010-2020 holdout on an identity-checked tape (free data) | Days | Positive and significant would reopen the stock model |
| 6+ months of clean frozen paper with an input archive | Calendar time | Settles parity; at a 10% sleeve it cannot show alpha |
| Actual SSO funding and tax-character records for A23 | 1 h | Margin above +1 pp or below 0 |

Unreconciled: snapshot value exceeds cash plus positions by $135-$320 every day from 09-15
to 11-04. Fits the dividend add-back but is not proven to be the only cause.
