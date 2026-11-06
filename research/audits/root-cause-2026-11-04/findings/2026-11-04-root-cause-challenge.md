# Independent challenge of the root-cause synthesis

**Date:** 2026-11-05 · **Branch:** `investigation/root-cause-2026-11-04` · **Type:** adversarial
review (Task 14 step 2) of `research/audits/root-cause-2026-11-04/2026-11-04-root-cause-investigation.md`. Diagnosis only:
this document makes no recommendation about trading, strategies, deployments, the paper account or
the account type, and no forecast.

**How this was done.** I did not see the author's reasoning and did not open the `.superpowers`
ledger. I checked claims against the raw artifacts (`docs/research/_*.json`), the lens JSONs (as
pointers, not as ground truth) and the database through a read-only session (SELECT only, 45 s
statement timeout; the helper named in the brief did not exist, so I wrote an equivalent one in the
session scratchpad). Short pandas computations ran under `scripts/run_capped.sh`. No backtest,
driver or other repository script was run apart from `scripts/check_lens_report.py`. No broker or
container was touched. Abbreviations follow the synthesis (SPY, TR, CAGR, pp/yr, WFO, OOS, PIT, MDE,
NW, SSO, A23, BTC, ETH, EOD). One added: MDY = SPDR S&P MidCap 400 ETF.

## 1. The opposite case, steelmanned

**(a) The synthesis under-credits real evidence.** The best-documented long-history test in the
project is the French momentum screen, and its full 1927-2026 result clears the conventional "t of
3" bar in both size groups: big winners t 3.05 (p 0.0011), small winners t 3.11 (p 0.0009),
size-matched t 3.72 and 5.18, CAPM alpha 3.4 and 5.7 %/yr
(`research/stock-signals/momentum/_momentum_french_screen_results.json`, `groups.*`). It was called NO-GO because a
Holm correction over two tests needed p below 0.000675 and because the registered bar was a
post-1994 CAGR gap. The synthesis reports only the pre-1994 sub-period. The S&P 500 momentum replay
(+4.37 pp/yr, Sharpe 0.87 against 0.78) ran the textbook 12-1 top-50 rule in every fold, so the
"fallback" it deployed is the canonical idea, not a substitute. The production replica's −0.64 pp/yr
is mostly its 5% cash reserve: against an exposure-matched SPY book it is +0.26 pp/yr, and in the
2010-2020 holdout +1.00 against SPY and +1.85 (t 1.57) exposure-matched
(`research/audits/production-replica/_production_replica_results.json`, `excess_vs_exposure_matched`). The FOMC Treasury
drift is real gross (day-clustered t 2.26). The NO comes from charging 10 bp per round trip, the
stock default, on Treasury ETFs. At 2 bp the net t is about 2.0. Four of five three-year crypto runs
beat ETH by 37 to 115 pp in total. In the only out-of-sample stretch, both runs beat BTC by 23 to
33 pp. The 80/20 cross-asset book delivered 93% of SPY's CAGR at 84% of its volatility and beat SPY
by 3.08 pp/yr in the 2007-2011 crisis holdout. MidCap reversion (15.0%/1.08 against MDY 9.1%/0.40)
was typed as a status label and so never counted as a GO.

**(b) The synthesis is too lenient.** The owner's question is about 2x SPY, and on that question
the record is more negative than "nothing reliably shows there is none" admits. The replica's
detectable effect is 4.24 pp/yr in 2021-2026 and 4.22 in 2010-2020. Both are below the target gaps
of those windows (9.57 and 6.14). The two windows gave −0.64 and +1.00. PARA cannot plausibly close
that distance. My inference: three long-only round trips capped at 5% of the account can cost at
most about 2.9 pp/yr over 5.25 years, so an ex-PARA replica would still be below about +2.3 pp/yr.
"0 reliable NO" is partly a product of rule R5, which tests a NO against its own +1.0 bar when one
exists and against the much easier 2x gap only when none exists. The headline count of "103
inconclusive" includes 42 rows that are not tests. The post-1994 French range is quoted as "+0.08 to
+0.83" when the registered results run from −0.62 to +0.83. The French screens are called the
"best-powered" tests, yet their indicative detectable effect after 1994 (about 4.4 pp/yr) is above
that window's 2.34 gap, so for the owner's target they are worse powered than the replica. The
artifact's own beta-matched figures show the leverage arms earning less than levered SPY at T-bill
funding, which is "exposure minus cost", yet 2.0x SSO is presented as the "measured maximum". The
paper shortfall is labelled "on the broker's own figures", but Alpaca's own end-of-day series gives
−4.57 pp, not −3.76.

**Which is stronger.** After checking, **(b) is stronger on the owner's question**: its points are
arithmetic on committed artifacts. Most (a) points are pre-publication, in-sample, Sharpe-not-CAGR,
low-beta or exposure effects. (a) succeeds in part. The long-history momentum premium is
under-reported. The replica's loss is mostly cash drag rather than stock selection. The FOMC NO
depends on a cost assumption. None of these moves a result toward the 2x target.

## 2. Claim audit table

"Syn" = `research/audits/root-cause-2026-11-04/2026-11-04-root-cause-investigation.md`. Result codes: **C** confirmed,
**C\*** confirmed with caveat, **W** wrong, **NC** not checked.

| # | Claim | Where | Checked against | Result | Discrepancy |
|---|---|---|---|---|---|
| 1 | 114 verdict rows: 0 reliable, 9 unreliable GO, 2 unreliable NO, 103 inconclusive | Syn:38-40 | `research/audits/root-cause-2026-11-04/_lens2_sensitivity.json` `classifications[].class` tally | C\* | Counts exact. But 42 of the 103 are STATUS labels and 3 are tests whose own verdict was inconclusive. Of 72 measured tests: 58 inconclusive (51 NO, 7 GO), 9 unreliable GO, 2 unreliable NO, 3 own-inconclusive |
| 2 | 2x target gap 9.57 pp/yr (2021-2026), 3.54 (2006-2026), 0.77 (1927-2026) | Syn:43-45 | `research/audits/root-cause-2026-11-04/_lens4_ceiling.json` `target.windows`; DB SPY closes 343.6255 (2021-01-29), 716.8133 (2026-04-30), 86.1470 (2006-06-20), 764.2000 (2026-09-29) | C | DB recompute: TR 108.60%, gap 9.56 (calendar-day years); 2006-2026 gap 3.54 |
| 3 | 2.0x SSO +4.15 pp/yr, 1.17x the 3.55 gap, −84.8% drawdown | Syn:47-48, 114-118 | `research/allocation/static-leverage/_static_leverage_aftertax_results.json` `windows.main_real_sso["static 2.0x (SSO) no rebalance"]["24/15"]`: pre_tax_cagr 0.1554, max_dd −0.8482; SPY TR 11.39 | C\* | 15.54 − 11.39 = 4.15. The arm's "pre-tax" CAGR is net of the annual tax paid from NAV, so 4.15 is a lower bound on a like-for-like pre-tax gap |
| 4 | 2.0x arm trailed SPY in 37.5% of 10-year windows | Syn:118 | same file, `rolling_10y_pretax_vs_spy` | C\* | 0.375 confirmed. Worst window ends 2009-03-09 and SPY's worst ends 2003-01-31, so the rolling series reaches into the synthetic pre-2006 stitch, not only 2006-2026 |
| 5 | Registered 1.3x arm +1.61, falling to +0.81 (+0.80 after tax) | Syn:119-120 | artifact 12.71 − 11.10; `research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.025.json` 11.91 − 11.10, after tax 11.2 − 10.4 | C\* | Numbers are on the artifact basis, while #3 is on the consistent basis. On the consistent basis they are +1.32 and +0.52 (Lens 4 table). The paragraph mixes bases |
| 6 | Momentum replay +4.37 pp/yr, 95% CI −7.07 to +16.05 | Syn:50, 121-123 | `research/audits/production-replica/_audit_reruns_20261019.json` `xs_momentum.excess_stats`: cagr_diff 4.367, ann_excess 4.486, t 0.761; SE 5.898 | C\* | The CI is on the arithmetic mean excess (4.49), not on the 4.37 CAGR gap |
| 7 | "Every measured stock-picking effect (post-1994: +0.08 to +0.83 pp/yr)" | Syn:54 (also 89-92, 353-354) | `research/stock-signals/factor-screen/_factor_screen_results.json` `H1-H4.A2`; `research/stock-signals/net-issuance/_net_issuance_screen_results.json` | W (at line 54) | Registered post-1994 figures run from −0.62 (H2) through −0.56 (H4) and −0.00 (H3) to +0.83. Lines 89-92 say "best", which is accurate |
| 8 | Replica lost 0.64 pp/yr, t −0.69 | Syn:84-85 | replica `windows.primary.arms.primary.excess_vs_spy` | C | −0.6423, t −0.6898 |
| 9 | Replica MDE 4.24 pp/yr | Syn:85-86, 240 | SE = 0.76198/0.68983 = 1.1046; × 3.8416 | C | 4.243 |
| 10 | Sleeve-scale MDE about 45 pp/yr | Syn:87-89 | 4.243 / `mean_exposure.stocks` 0.09370 | C | 45.29 |
| 11 | Replica 85.33% SPY, 9.37% stocks | Syn:405 | `mean_exposure` | C | 0.8533 / 0.0937 / cash 0.0530 |
| 12 | Gap would need about +102 pp/yr from the sleeve | Syn:363-365 | 9.57 / 0.0937 | C | 102.1 |
| 13 | Momentum pre-1994 t 3.56 (big); negative after 2013 | Syn:92-94 | `groups.big.diag` | C | t 3.556; 2013-2026 −0.97 (big), −3.34 (small) |
| 14 | Paper +1.07% vs SPY TR +4.84%, −3.76 pp "on the broker's own figures" | Syn:97-99 | `research/audits/root-cause-2026-11-04/_lens3_snapshot_gap.json` `era_returns_phase0_boundaries` (All) | W (label) | Numbers are the U basis (snapshot cash + qty × price at run time). Alpaca's own EOD equity gives −1.48% vs +3.09% = −4.57 pp (to 11-03 close) |
| 15 | Idle cash −2.46, stocks −0.88, sweep −0.15, other −0.29 | Syn:100-101 | same file, `decomposition_B_basis_estimate.fine` | C | Era cash-drag terms sum to about −2.44; labelled an estimate in the synthesis |
| 16 | 74 closed stock round trips, all short-term, median 25.5 days | Syn:378-379 | my SELECT on `paper_trades` (SPY excluded, FIFO by symbol) | C | 74 trips, median 25.5, max 92, all ≤ 365 days |
| 17 | Break-even +2.42 (24/15) / +5.05 (37/20); +1.0 becomes −1.08 / −2.55 | Syn:124-126, 380-381 | `research/audits/root-cause-2026-11-04/_lens4_ceiling.json` `after_tax.stylised_books` | C | 2.4244 / 5.0516; −1.0826 / −2.5525. Stylised model (all gains realised short-term yearly) |
| 18 | Core PIT Sharpe 0.59 vs SPY 0.78; 0.99 and 0.97 earlier | Syn:229-230, 434 | `research/audits/rebaseline/_pit_rebaseline_20260923.json`, `research/audits/rebaseline/_rebaseline_corrected_tape_20260822.json`, `research/audits/rebaseline/_anchor_fix_reproduction_results.json` | C | 0.5922 / 0.7792; 0.9914; 0.9712 |
| 19 | Every crypto portfolio lost to BTC over three years, by 5 to 44 pp/yr | Syn:142-144 | `research/audits/root-cause-2026-11-04/_lens8_tape_checks.json` `benchmark_comparison` | C\* | −5.31, −18.56, −18.63, −20.48; run 3 computed −43.9. The same runs beat ETH in 4 of 5 (by 37.5 to 115.4 pp total), which the synthesis omits |
| 20 | Headline 48.46% vs BTC −21.34%; beta about 0.26; t 1.51 | Syn:145-148 | same, 2026-05-01 YTD row | C\* | −21.34 is the report's own BTC figure; the tape gives −17.46 and yfinance −18.98. Beta 0.263-0.314; alpha t 1.51 |
| 21 | OOS stretch lost 9-19% while BTC fell 42% | Syn:148-149 | same, holdout split rows | C\* | −9.44 / −18.92 vs BTC −41.94 / −42.30, so it beat BTC by 23 to 33 pp in that stretch. The synthesis does not say so |
| 22 | Live crypto: 0 trades in the DB | Syn:150 | SELECT count(*) on `trades` / `orders` | C | 0 trades, 1 order |
| 23 | 19 `symbol_inception` rows rewritten at 2026-11-04 23:48 PST | Syn:521-523 | SELECT `symbol, recorded_at` | C | All 19 at 23:48:18.692 PST. No later `recorded_at` in the table (max is this time; 159 rows) |
| 24 | Each symbol's "full price history was re-downloaded and written over (up to 31,967 rows)" | Syn:523-527 | per-symbol row counts and `first_date` | C\* | 31,967 sums exactly. But CSRA, PARA, QRVO and SBNY keep 4,388 rows dated before their recorded first date (598, 492, 2,922 and 376), so 27,579 is the rewritten maximum. Those four series have level breaks at that date (×0.61, ×68.5, ×1.19, ×0.012) |
| 25 | PARA max close 3,935.11; factor 28.9; not a uniform rescaling | Syn:528-530 | SELECT on `ohlcv`; comparison with `ohlcv_yf_preshift_backup_20260820` | C\* | Confirmed. Against the backup, the change is piecewise constant (×0.0345 from 2021-02-12 to 2025-03-14, then 0.062, 0.443, 0.561, 1.0 from four 2025 dates). Daily returns change only on those boundary days |
| 26 | Five of the 19 are current members; ARES last traded 10-22; effect on live signals "not compared" | Syn:538-541 | SELECT on `paper_trades`; backup comparison | C\* | ARES bought 10-15 and sold 10-22. Through 2026-08-20, the bars of ARES, GDDY, LITE, VEEV and XYZ match the backup to within 0.35 pp of daily return (price ratio 0.9895-1.0001). Only post-08-20 bars remain uncompared |
| 27 | 81% chance of a false t-2 win across 72 tests; about 9% at t 3 | Syn:262-264 | 1 − (1 − p)^72, one-sided | C | 0.809 / 0.093 |
| 28 | "Two GO calls used Sharpe while raw CAGR trailed SPY" | Syn:267-269 | cross-asset and blend artifacts | W | Only the 80/20 trails (13.61 vs 14.70). Its holdout beat SPY (+3.08). The KEEP blend ties (12.963 vs 12.959). Lens 2 cites two *windows* for the 80/20 (−1.09, −1.07) |
| 29 | Momentum re-run used the fallback in 17 of 17 folds | Syn:242-243 | `xs_momentum.gate_pass_str` 0/17, `dsr_gate_passes` 0 | C\* | The anchor was 12-1 momentum, top 50 (Lens 4), so the fallback is the canonical idea |
| 30 | Circuit breaker halts 87.5% of runs at true Sharpe 1.0 | Syn:243-244 | `research/programme/wfo-methods/_wfo_policy_synthetic_results.json` `cells["1.0\|1.0\|legacy"]` | C\* | 0.875 is "ever halted at least once" with gates always passing. Halted-fold share is 0.34; runs ending halted 0.50 |
| 31 | Lights: 20 yellow, 5 red, 10 grey | Syn:467 | table recount | C\* | The count is exact. Row 7 should be red under the stated rule (see objection 9) |
| 32 | 33/72 one family; 44/51 large-cap; 65/72 long-only | Syn:273-274 | `research/audits/root-cause-2026-11-04/_lens5_coverage_matrix.json` rows | C | 65 long; 44 large of 51 single-stock; 33 is the Lens 2 family count (17 mean-reversion + 10 sizing/exit + ensemble rows) |
| 33 | "Every headline with a raw file recomputes" / "was checked" | Syn:65-66, 155-157 | `research/audits/root-cause-2026-11-04/2026-11-04-lens7-research-accuracy.md:108-117`; Syn:593-595 | C\* | 5 recomputed + 27 read = 32 of 41. Nine were never re-checked (Syn §7 says so), and two cannot be located (dual momentum, crypto textbook) |
| 34 | 44 orders at next open, about 47 bp | Syn:319 | `_lens3_snapshot_gap.json` `fills.next_open_vs_prior_close` | C | n 44, 47.4 bp |
| 35 | Frozen-era sleeve t 2.14; whole-period t 0.73 | Syn:497-498 | `detectable_effect.t_stats` | C | 2.139 / 0.734 |
| 36 | FOMC t fell from 1.57 to 0.98 | Syn:165 | `research/audits/production-replica/_fomc_clustered_event_study_20261020.json` | C\* | Net t 1.572 → 0.981. Gross clustered t 2.26 (significant). Net subtracts 10 bp per event |
| 37 | 43.7% of Binance.US 4-hour bars copy Kraken | Syn:337 | `_lens8_tape_checks.json` `tape.duplicates.cross_venue_clone` | C | 94,721 / 216,818 = 0.437 (count from Lens 8, ratio recomputed) |
| 38 | No weekend bars; one SPY row per date | Syn:153-154 | SELECT | C | SPY 6,751 rows = 6,751 distinct dates; 0 weekend rows for SPY, AAPL, MSFT, VZ, ARES and PARA |
| 39 | 53 NO rows: 2 unreliable, 51 inconclusive; power fails in 51 | Syn:128-133 | Lens 2 JSON `power_condition` tally | C | Power met only for FOMC-corrected and month-end |
| 40 | 5,595 daily moves of 40% or more in 2010-2020 | Syn:330-331 | not run (whole-table scan exceeds the timeout budget) | NC | |

**Tally (40 claims):** 20 confirmed, 16 confirmed with caveat, 3 wrong, 1 not checked.

## 3. Objections

Severity: **Critical** = changes a headline conclusion; **Important** = a reader would draw a
materially wrong inference; **Minor** = an accuracy or wording fix. I found nothing Critical. No
error I found reverses the synthesis's direction of travel.

### Important

1. **"103 inconclusive results" inflates the denominator.** The 42 STATUS rows are feasibility,
   design, literature, untriaged, operations and data-fix labels (Lens 2 R1), not tests. Line 41
   says "Inconclusive means the tests were too weak, too contaminated or too poorly recorded",
   which a plain reader applies to all 103. Of the 72 measured tests, 58 are inconclusive. The
   same R1 typing put measured claims under STATUS, for example MidCap 400 reversion ("Researching",
   15.0%/1.08 vs MDY 9.1%/0.40, `live_deployed: true` in
   `research/audits/root-cause-2026-11-04/_investigation_inventory.json`). Under R6 that row would be an unreliable GO. So
   the GO count is also understated by at least one in substance.

2. **On the owner's actual question, the headline is too symmetric.** "Nothing reliably shows
   there is none" (line 38) holds for the registered +1.0 bars. It does not hold for 2x SPY at
   whole-account level for this design. The replica was powered for that in both windows (MDE 4.24
   < 9.57; 4.22 < 6.14) and returned −0.64 and +1.00. The PARA bound in section 1(b) is my
   inference, not a measurement, and assumes the simulator enforces the configured long-only sizing.
   It suggests PARA cannot account for a 7-9 pp/yr miss. If PARA's tape added noise, removing it
   would also *lower* the replica's tracking error, and with it the MDE. The synthesis has the
   pieces (Q4 line 134) but makes the conclusion conditional on PARA without bounding PARA. The
   power rule (R5) uses the registered +1.0 bar where one exists and the 2x gap otherwise, so rows
   without a bar face an easier test than the replica. Line 89 calls the French screens "the
   best-powered tests". Measured against the target, the replica is better powered than the
   post-1994 French screens: indicative MDE about 4.4 (Lens 2 lines 63-71) against a 2.34 gap.

3. **The paper shortfall label is wrong, and the larger broker figure is omitted.** "−3.76 pp on
   the broker's own figures" (line 99) is the U basis, built from the trader's own snapshot at about
   12:45 PT. Alpaca's portfolio-history EOD equity, the broker's own series, gives −4.57 pp through
   the 11-03 close (Lens 3 table, line 194). Part of the 0.81 pp difference depends on an unproven
   hypothesis (AVB valued near zero at EOD). A reader of Q2 is not told that the broker's own record
   shows a deeper shortfall.

4. **The leverage paragraph mixes bases and omits evidence that supports "exposure, not
   skill".** Q3 sets +4.15 (consistent basis) next to +1.61 and +0.81 (artifact basis). On one
   basis the 1.3x arm is +1.32 and +0.52 (`research/audits/root-cause-2026-11-04/_lens4_ceiling.json`). Lens 4 corrected
   this mixing once (Syn:574-575), and the synthesis reintroduces it. The 37.5% rolling-window
   figure reaches into synthetic pre-2006 data. The artifact also holds
   `beta_matched_pretax_cagr` at the window's 1.64% T-bill rate: 13.94% for 1.3x and 15.83% for
   1.5x, against the arms' 12.71% and 13.76%. The SSO arms earned less than the same exposure at
   T-bill funding. That holds even after allowing about 0.3 pp for the tax-from-NAV basis. Calling
   the 2.0x arm "the highest figure that passes Lens 4's soundness tiers" frames the leverage grid's
   top end as a ceiling. It is the largest exposure tested, not a maximum.

5. **The ranking of root causes across groups is asserted, not derived.** Line 217 says the
   within-group rank is "by how much of the outcome the item can explain", and the summary's list
   gives the overall order. No item's share of the shortfall is measured: A1 says "unknown how
   much each bug contributed", and B1, B5 and C6 are "unknown". Cause 1 (the target is large) is a
   property of the target. Cause 3 (leaks that *favoured* the strategy) cannot by itself make
   results trail SPY. It explains false optimism and the choice of live design, not
   underperformance. Causes 1 and 2 overlap: with a 9% sleeve, construction rather than effect size
   is the binding limit. Read the list as the authors' judgement, not a measured ranking.

6. **Replica robustness and exposure-matched results are omitted.** The replica artifact has a
   same-day (lag 0) arm at −0.42 pp/yr, cost arms at 3 and 10 bp (−0.57 and −0.83), a zero-cash arm
   (−0.83) and an exposure-matched comparison: +0.26, t 0.20 in 2021-2026; +1.85, t 1.57 in the
   non-citable holdout. B1 says the replica does not model the same-day fill. The lag-0 arm is the
   nearest available approximation, and the result barely moves. The exposure-matched figures show
   that the −0.64 is mostly the 5% cash reserve, so the stock picks were neither clearly helpful nor
   harmful. Both facts are material to Q1 and Q2.

7. **The incident description is inaccurate for four symbols, and part of the "unknown" was
   checkable.** I verified the 19 rows at 23:48:18.692 PST and no later `recorded_at`. PARA's levels
   match the synthesis (max 3,935.11). Three corrections:
   - **Not every history was rewritten.** CSRA, PARA, QRVO and SBNY still hold 4,388 rows dated
     before their recorded first date, so line 523 overstates the rewrite. Their series carry level
     breaks at that date: PARA ×68.5 on 2021-02-12, SBNY ×0.012 on 2024-08-15, CSRA ×0.61, QRVO
     ×1.19. QRVO now has a single bar after the break.
   - **The PARA and SBNY breaks predate the incident.** The 2026-08-20 backup table holds
     identical pre-break rows, and SBNY matches it exactly. PARA's own break was about 29 times
     larger before the incident. The incident did not create these breaks, but the data still carry
     them.
   - **"Whether the rewrite changes any live signal was not compared" can be narrowed.**
     Read-only, against `ohlcv_yf_preshift_backup_20260820`, the five current members match to
     within 0.35 pp of daily return through 2026-08-20. Only bars after 08-20 remain unknown, and
     those include ARES's 10-15 to 10-22 trade.

   The incident is not understated in severity. It is imprecise in mechanism, and section 6's
   claims rest on nine citations to a gitignored ledger (`.gitignore:67`). A fresh reader cannot
   verify the owner decisions in section 8 or the 27,579 figure from the repository. I reproduced
   the 27,579 independently.

### Minor

8. **Wrong count of Sharpe-only GOs** (line 268). Only one GO (the 80/20) trailed SPY's CAGR. The
   KEEP blend ties it, and the 80/20 holdout beats it.
9. **Lights inconsistent with the stated rule.** Section 4 defines red as including "a claim Lens
   6 found the code does not support". Row 7 includes the anchor-fix reproduction, which has Lens 6
   `spec_matches_code: "no"` (`research/audits/root-cause-2026-11-04/_lens6_verdicts.json`), so it should be red or split.
   Row 13 bundles `stocks-ensemble-kelly-2026-07` (Lens 6 "no") under grey.
10. **Summary range omits negatives** (line 54; claim 7). Post-1994 registered results include
    −0.62 and −0.56, and post-2008 or post-2013 results go to −0.93, −0.97 and −3.34. That makes
    the evidence more negative than the summary shows.
11. **"Every headline with a raw file recomputes" (line 65) contradicts section 7** (9 of 41 not
    re-checked, plus cannot-locate rows).
12. **The momentum fallback framing** (A2, line 241). For the S&P 500 replay, the fallback deployed
    in 17 of 17 folds was 12-1 momentum, top 50. The +4.37 therefore measures plain momentum, not a
    stand-in.
13. **The full-century momentum test is not reported.** t 3.05 and 3.11 (both above 3), p 0.0011
    and 0.0009, fail Holm (0.000675) narrowly. Q1 reports only the pre-1994 sub-period t of 3.56,
    which reads as weaker evidence for the premium's long-run existence than the artifact shows.
14. **The FOMC NO depends on its cost assumption** (row 26). It charges 10 bp per round trip on
    Treasury ETFs (`research/allocation/fomc-drift/2026-07-20-fomc-drift-nogo.md:179-181`, the stock default). From
    the artifact's own SE, 2 bp would give net t of about 2.0 (my arithmetic). That is still below
    3, but "only the code check is missing" hides this dependence.
15. **The crypto framing omits relative results that cut the other way.** These are the ETH
    comparisons and the OOS stretch beating BTC by 23 to 33 pp. Low exposure (beta about 0.26 to
    0.42) is the likely explanation, but it is not stated. The BTC −21.34% is the report's own
    figure; the tape gives −17.5% and yfinance −19.0%.
16. **The 87.5% halt rate needs its definition**: "ever halted" in a model with gates always
    passing; 34% of folds are halted.
17. **Small internal inconsistencies.** The 2006-2026 gap is 3.54 (line 44) and 3.55 (line 116);
    these are two SPY sources, not explained. The 9.54 in C2 is the 2022-2026 OOS window, unlabelled
    there. The account start is "$100,000 funding" (line 304) and "about $102,000" (line 475).
18. **The limits are stated honestly, with one qualifier.** "No second model family" is stated
    twice (lines 75-77, 585-592), and section 7 lists partial recomputation and unexamined code.
    The 113-of-114 classifier reproduction checks how the rules were applied, not whether they are
    sound. Lens 2 itself records that the rules and classes arrived in one commit (Lens 2 lines
    1158-1166). This challenge is also a Claude agent, so it is not model-independent either.
19. **Owner constraints.** I found no recommendation, retirement or shutdown advice, account-type
    suggestion or banned phrase. The Phase 2 "Cost" column and "What would change this" column
    describe unknowns, not instructions.

## 4. Verdict

Most of the synthesis survives. 36 of 40 checked claims are confirmed, 16 of them with caveats;
3 are wrong and 1 was not checked. Every headline number I recomputed from raw artifacts or the
database matched. The errors are in framing:

- the "103 inconclusive" denominator includes non-tests;
- the paper shortfall is mislabelled as the broker's own figure, which is −4.57 rather than −3.76;
- the leverage bases are mixed;
- one count is wrong (two Sharpe-only GOs);
- one light breaks the stated rule;
- the incident description is imprecise.

The edits that would make it accurate:

- report the 72 measured tests separately from the 42 status rows;
- relabel −3.76 as the snapshot basis and give the Alpaca EOD −4.57 beside it;
- state the 1.3x figures on the same basis as the 2.0x figure, and cite the beta-matched figures;
- state that the replica was powered against the 2x gap in both windows, and bound PARA's possible
  effect;
- add the replica's lag-0, cost and exposure-matched arms;
- quote the full post-1994 range, including negatives, and the full-century momentum t;
- fix the Sharpe-only GO count and row 7's light;
- describe the four partially rewritten symbols and the backup comparison;
- label the cross-group ranking as judgement.

What remains genuinely uncertain: whether a stock-selection edge of 1 to 4 pp/yr exists (no test
here can see one); PARA's exact effect on the replica; how the live-versus-backtest differences
change returns; the post-08-20 effect of the rewrite on live-universe bars; and whether the
gate-heavy WFO machinery hid or manufactured any result.

## Verification

- Read-only SQL (SELECT, read-only session, 45 s timeout) on `symbol_inception`, `ohlcv` (19
  rewritten symbols, SPY), `ohlcv_yf_preshift_backup_20260820`, `paper_trades`, `trades` and
  `orders`. Scratch scripts are in the session scratchpad and are not committed.
- Artifact reads: replica, momentum, factor, net-issuance, PIT, corrected-tape, anchor-fix, core+SPY
  blend, static leverage (base, +1.5, +2.5), cross-asset (and holdout), month-end, FOMC clustered,
  audit re-runs, WFO policy synthetic and fold attribution, plus the Lens 2, 3, 4, 5, 6, 8 and 9
  JSONs as pointers.
- Gate: `.venv/bin/python scripts/check_lens_report.py --paths-only` on this file.
