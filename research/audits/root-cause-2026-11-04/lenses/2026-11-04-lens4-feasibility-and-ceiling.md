# Lens 4: target feasibility and measured ceiling

**Date:** 2026-11-04. **Branch:** `investigation/root-cause-2026-11-04`.
**Scope:** read-only diagnosis for the root-cause investigation
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`, section 5, Lens 4,
including the "Measured ceiling" bullet; target definition in section 1). No
recommendations, no strategies, no forecasts. Every figure below is a measured value from a
committed artifact, or a labelled DERIVED value computed here from such an artifact or from
free public data.
**Raw results:** `research/audits/root-cause-2026-11-04/lenses/_lens4_ceiling.json`.
**Upstream inputs (committed, reviewed):** inventory `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`;
Lens 2 `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` and
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md` under the fixed spec
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-sensitivity-spec.md`; Lens 1 `research/audits/root-cause-2026-11-04/lenses/_lens1_tape_state.json`;
Lens 3 `research/audits/root-cause-2026-11-04/lenses/_lens3_snapshot_gap.json`; Lens 5
`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md` and `research/audits/root-cause-2026-11-04/lenses/_lens5_coverage_matrix.json`;
Lens 6 `research/audits/root-cause-2026-11-04/lenses/_lens6_verdicts.json`; Lens 7 `research/audits/root-cause-2026-11-04/lenses/_lens7_recompute.json`;
Lens 9 `research/audits/root-cause-2026-11-04/lenses/_lens9_power.json` and `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens9-power-and-regime.md`.

Abbreviations, expanded once: SPY = SPDR S&P 500 ETF Trust; TR = total return (price plus
dividends reinvested); CAGR = compound annual growth rate; pp = percentage points; bp = basis
points (0.01%); CI = confidence interval; SE = standard error; NW = Newey-West (a standard error
that allows for autocorrelated returns); t = t-statistic; MDE = minimum detectable effect (the
smallest true effect a test finds with 80% probability); IR = information ratio (excess return
divided by tracking error); IC = information coefficient (correlation between forecasts and
outcomes); TE = tracking error; WFO = walk-forward optimization; OOS = out of sample; PIT =
point in time; SSO = ProShares Ultra S&P 500 (a 2x daily-leveraged S&P 500 fund); LETF =
leveraged exchange-traded fund; ETF = exchange-traded fund; TLT, IEF = iShares 20+ year and
7-10 year Treasury ETFs; GLD = SPDR Gold Shares; PDBC/DBC = commodity ETFs; FOMC = Federal
Open Market Committee; DSR = deflated Sharpe ratio; ST/LT = short-term/long-term capital gains
(held under / at least one year); A23 = the owner-adopted 70/30 SPY/SSO allocation study;
French = Kenneth French data library.

## Summary

1. **The target is a different number in every window.** Doubling SPY's total return needs
   a CAGR gap of **9.57 pp/yr over 2021-01-31..2026-04-30** (SPY +108.60%, target +217.21%),
   **3.54 pp/yr over 2006-06-21..2026-09-29**, **2.34 pp/yr over 1994-2026** and **0.77 pp/yr
   over 1927-2026** (French market). Because the target doubles a cumulative return, the
   longer the window the smaller the annual gap it implies. Phase 0's "+10 to +15 pp/yr needed"
   is close only for 5-year windows (9.5 to 9.8 pp/yr on 2021-2026 and 2022-2026), and Lens 9's
   2.8-year IBIT window (15.15 pp/yr, `research/audits/root-cause-2026-11-04/lenses/_lens9_power.json`) is the only window
   inside 10-15; every longer window is much smaller. Section "Target arithmetic".
2. **Tier A is empty.** Lens 2 classed 0 of 72 test rows reliable (0 reliable NO, 0 reliable
   GO) and 9 GO rows unreliable. Every ceiling figure below is Tier B at most, and all the
   early "beat SPY" headlines are Tier C.
3. **Measured ceiling by approach (Tier B unless stated; never mixing arithmetic and CAGR):**
   - SPY sweep sleeve and ensemble/reversion stock selection: **no Tier A/B figure.** Tier C
     reference: replica CAGR gap -0.64 pp/yr (2021-2026), holdout +1.00 (not citable).
   - Momentum, S&P 500 replay: **+4.37 pp/yr CAGR gap** (OOS 2022-2026), arithmetic CI
     [-7.07, 16.05]; 0.46 of the 9.54 pp/yr gap. French registered post-1994: +0.08 (Big),
     +0.20 (Small). French pre-1994 diagnostic: +3.12 (Big), **+4.77 (Small)**, 2.7 and 4.2
     times that window's 1.14 pp/yr gap; the 2013-2026 sub-period is -0.97 / -3.34.
   - Factor tilts post-1994: **+0.83** (0.35 of the 2.34 gap). Net issuance post-1994: **+0.48**.
   - Event/calendar: no Tier A/B figure versus SPY.
   - Cross-asset 80/20: **+3.08** only in the 2007-2011 crisis holdout (where SPY lost money
     and the 2x target is undefined in a useful sense); -1.07 over 2016-2026.
   - Static leverage (A23 artifact, 2006-2026, T-bill funding averaging 1.64% in the window):
     registered 1.3x arm **+1.61** pp/yr on the artifact's basis, falling to **+0.81 (+0.80
     after tax)** when funding is set 2.5 points higher (about today's 4.1% bills;
     `research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.025.json`). The 2.0x SSO arm measured
     **+4.44 pp/yr on the artifact's basis** (its SPY comparator, 11.10%, is NAV after the
     artifact's annual tax payments) and **+4.15 on one consistent basis** (15.54% against
     SPY's true total-return CAGR of 11.39%), which is **1.17 times** the 3.55 pp/yr gap on
     that basis. That 2.0x figure is a pick made after seeing the results (the best of five
     leverage arms in the artifact, not the registered 1.3x arm); it has no t-statistic and no
     confidence interval; it reaches Tier B only through a Lens 4 judgement exception (code not
     reviewed); it has no funding-sensitivity run, so it holds only at the window's 1.64%
     average funding; and it is leverage, meaning more market exposure and deeper drawdown
     (**-84.8%** against SPY's -55.3%), not stock-picking skill. It trailed SPY in 37.5% of
     rolling 10-year windows (worst -12.56 pp/yr, ending 2009-03-09), against 23.8% (worst
     -3.29) for the 1.3x arm. Over the synthetic 1993-2006 holdout the 2.0x arm is +1.28 on the
     artifact's basis (+0.96 against true SPY TR) against a 4.54 pp/yr gap. Century study
     (after tax only): 30% arm +1.24 (max drawdown -91.4%, ruin flag), 60% arm +2.13 (-95.7%);
     the 1x base arm itself fell -84.6% in the same study.
   - Crypto: pointer to Lens 8 (not in the repo at this commit).
4. **The only Tier B figures that reach the target gap of their own window are leverage
   (the post-hoc 2.0x SSO arm over 2006-2026, at that window's funding) and pre-1994 French
   momentum.** The century arms are after-tax only and the gap is pre-tax, so their ratio is
   not computed (basis mismatch). Leverage multiplies
   SPY's beta and drawdown; it is not stock-selection skill. Pre-1994 momentum is the
   pre-publication period and it reverses after 2013. No Tier A/B stock-selection figure after
   1994 reaches its window's gap.
5. **After tax (DERIVED):** all 74 closed paper stock round trips were short-term (median hold
   25.5 days, maximum 92; the 52 SPY sweep rows are outside this count), net realized gain $772.39. In a stylised book that realizes every
   gain short-term each year against SPY held and taxed once at the end, the book needs
   **+2.42 pp/yr pre-tax** (24/15) or **+5.05 pp/yr** (37/20) just to tie SPY after tax over
   2021-2026. A +4.37 pp/yr pre-tax edge keeps +1.48 (34%) at 24/15 and turns to -0.43 at
   37/20; a +1.0 edge turns negative at both.

## Falsifiers

Written before opening any ceiling figure. Phase 0 hypotheses under test
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-phase0-trust-audit.md`; spec section 4, hypothesis 7):

- **H4a, "the best long-only large-cap effects after 1994 are +0.1 to +0.8 pp/yr".**
  Overturned if any Tier A/B large-cap stock-selection row with a post-1994 window shows a
  point estimate outside that range: below +0.1 (then the range is too generous) or above
  +0.8 (then it is too pessimistic). Also overturned if the cited +0.1 / +0.8 do not come from
  artifact keys (prose only).
- **H4b, "+10 to +15 pp/yr is needed".** Overturned if the section 1 arithmetic on the
  windows actually used by the material results gives required gaps outside 10-15 pp/yr for
  most of those windows.
- **H4c, "a 2x-SPY goal is out of reach without leverage or a much larger effect".**
  Overturned if any Tier A/B unlevered row reaches the required gap of its own window.
  Confirmed in part if the only rows reaching their gap are levered or pre-publication.
- **Ceiling soundness.** The ceiling figure for an approach is overturned if its row fails
  the tier rules on re-reading Lens 1/2/6/7, or if the value does not recompute from the
  artifact key cited.
- **After-tax question ("do short holding periods turn a small edge negative?").** Answered
  "no" if the derived break-even pre-tax edge for a fully short-term book is at or below
  zero; "yes" if a +1 pp/yr pre-tax edge turns negative after tax at the 24/15 bracket.

Outcome (filled after the evidence): H4a confirmed on its own terms for the registered
French screens (post-1994 values +0.08 to +0.83, with H2/H3/H4 at -0.62 to -0.00), but the
range omits the 2021-2026 xs_momentum replay at +4.37 (Tier B, noise-dominated, MDE 22.66).
H4b overturned as a general statement: 9.54-9.77 pp/yr on the 2021-2026 and 2022-2026
windows, 3.5 to 6.5 pp/yr on 10-20 year windows, 0.77-2.46 on French long windows.
H4c confirmed in the form "only leverage or pre-publication momentum reached their window's
gap". After-tax question answered "yes" (break-even +2.42 pp/yr at 24/15).

## Steelman

The strongest evidence in the project's record that a large raw excess over SPY was
observed or is attainable, and how it survives the other lenses:

| claim | figure | source | survives? |
|---|---|---|---|
| Early "beat SPY" ensemble WFO (Sharpe 1.12, CAGR about 16-19%) | CAGR 16.3% (1.12 baseline) and 18.8% (2026-06-20 WFO) in prose; no raw artifact | `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` classifications for `stocks-ensemble-ndh-baseline-1-12`, `stocks-ensemble-wfo-validated-2026-06-20`, `stocks-three-sleeve-blend-1-14` | **No.** Lens 2 unreliable GO (anchor leak, PIT universe leak, SPY duplicates, day shift); no raw artifact; Lens 6 mismatch. Tier C. |
| ensemble_ic / ensemble_kelly pre-PIT | 33.95% / 38.24% CAGR | `research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json` via Lens 2 sub-rows | **No.** Same strategies with PIT membership: 3.33% / 6.28% (`research/audits/rebaseline/_pit_rebaseline_20260923.json`). The difference is the universe leak. Tier C. |
| Production replica holdout 2010-2020 | +1.00 pp/yr CAGR gap; exposure-matched +1.85 (t 1.57) | `research/audits/production-replica/_production_replica_results.json` `windows.holdout.arms.primary.excess_vs_spy`, `excess_vs_exposure_matched` | **Weakly.** Not citable (`windows.holdout.citable=False`, tainted trades); t 0.69; Lens 1 tape identity unknown. Tier C. |
| xs_momentum corrected replay | +4.37 pp/yr CAGR gap; arithmetic +4.49, t 0.76 | `research/audits/production-replica/_audit_reruns_20261019.json` `xs_momentum.excess_stats` | **As a point estimate, yes; as evidence, weak.** Clean tape, Lens 6 match, Lens 7 recompute yes: Tier B. CI [-7.07, 16.05]; MDE at t >= 3 is 22.66 pp/yr (Lens 9), so the test could not distinguish +4.37 from zero or from +15. |
| French momentum before 1994 | Big +3.12 pp/yr (t 3.56), Small +4.77 (t 3.38); size-matched full window +3.66 (t 3.72) / +5.14 (t 5.18) | `research/stock-signals/momentum/_momentum_french_screen_results.json` `groups.<g>.diag[0]`, `groups.<g>.size_matched` | **In its window, yes; later, no.** Clean French data, statistic recomputed (Lens 7): Tier B. The same groups are +0.78/+2.60 in 1994-2012 and -0.97/-3.34 in 2013-2026; registered post-1994 A2 +0.08/+0.20. The size-matched comparator is the same-size portfolio, not the market or SPY, so it is not a raw-over-SPY figure. |
| Static 2x leverage | 2.0x SSO +4.44 pp/yr over 2006-2026 on the artifact's basis (+4.15 against true SPY TR), above that window's 3.54-3.55 gap | `research/allocation/static-leverage/_static_leverage_aftertax_results.json` `windows.main_real_sso` | **Measured, but weakly placed.** Picked after the fact as the best of five leverage arms (registered arm: 1.3x, +1.61, and +0.81 at 2.5 points higher funding per `research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.025.json`); no t-statistic or CI; Tier B only by a Lens 4 judgement exception; no funding-sensitivity run, so conditional on the window's 1.64% average bills; maximum drawdown -84.8%; behind SPY in 37.5% of rolling 10-year windows; synthetic 1993-2006 holdout +1.28 (below its 4.54 gap). |
| Century leverage | 60% arm +2.13 pp/yr after tax over 100 years | `research/allocation/century-leverage/_century_leverage_results.json` `scenarios` | **Measured, Tier B (L4-X1).** After-tax only, so not comparable with the pre-tax 0.77 gap; ruin flag true for 30% and above; max drawdown -91.4% (30%) to -95.7% (60%), and -84.6% for the 1x base arm. |
| Cross-asset 80/20 in the 2007-2011 holdout | +3.08 pp/yr | `research/allocation/cross-asset-sleeve/_cross_asset_sleeve_holdout_results.json` | **Only in a crisis window.** -1.07 over 2016-2026, -1.09 over 2021-2026 (`research/allocation/cross-asset-sleeve/_cross_asset_sleeve_results.json`). |
| Month-end Treasury rule | Welch t 2.75 on invested days | `research/allocation/month-end-treasury/_month_end_treasury_results.json` | **Not as an SPY excess.** SPY-funded portfolio -0.08 pp/yr; Lens 1 DSR-units item of unknown direction. Tier C. |
| Crypto (Kraken live bot "profitable", funding carry +0.45%) | prose | inventory rows `crypto-kraken-live-bot-pnl`, `crypto-funding-carry-phase4` | **Not examined here.** No raw artifacts; benchmark is BTC/ETH, not SPY; Lens 8 owns it. |

Does the steelman hold up? A large raw excess over SPY was **observed** on sound-enough
evidence in exactly two ways: by holding more than 1x SPY (leverage), and by momentum before
its publication. Neither is stock-selection skill demonstrated after 1994. The project's
largest post-1994 stock-selection point estimate (+4.37) is Tier B but sits inside a CI
that includes zero and the full target.

## Target arithmetic

Definition (spec section 1): the target total return equals 2 x SPY's total return over the
same window. With R = SPY TR and Y = years: target CAGR = (1 + 2R)^(1/Y) - 1, SPY CAGR =
(1 + R)^(1/Y) - 1, required gap = target CAGR - SPY CAGR (a CAGR gap, pp/yr). Pre-tax.

Worked example, 2021-01-31..2026-04-30 (Lens 9; SPY closes 2021-01-29..2026-04-30, Y =
5.243): R = +108.60%; target = +217.21%; target CAGR = 3.1721^(1/5.243) - 1 = 24.63%; SPY
CAGR = 2.0860^(1/5.243) - 1 = 15.05%; **gap = 9.57 pp/yr**. Lens 4's independent recompute
with the same start-close convention: R = +108.60%, gap 9.56 (Y = 5.248 by calendar days).

| window (used by) | source | SPY / market TR | 2x target TR | SPY CAGR | target CAGR | gap pp/yr |
|---|---|---|---|---|---|---|
| 2021-01-31..2026-04-30 (replica, PIT core, blends, cross-asset pinned) | Lens 9 | +108.60% | +217.21% | 15.05% | 24.63% | **9.57** |
| 2022-01-31..2026-04-30 (xs_momentum OOS span) | Lens 9 / Lens 4 | +69.33% / +72.38% | +138.66% / +144.76% | 13.21 / 13.66 | 22.75 / 23.43 | **9.54 / 9.77** |
| 2010-01-31..2020-12-31 (replica holdout) | Lens 9 | +333.17% | +666.33% | 14.37 | 20.51 | 6.14 |
| 2012-01-01..2026-04-30 (FOMC) | Lens 9 | +635.17% | +1270.34% | 14.94 | 20.05 | 5.11 |
| 2016-01-04..2026-04-30 (cross-asset long) | Lens 4 | +317.06% | +634.12% | 14.83 | 21.29 | 6.46 |
| 2007-06-01..2011-02-28 (cross-asset holdout) | Lens 4 | -5.88% | -11.76% | -1.60 | -3.28 | -1.68 (degenerate) |
| 2006-06-21..2026-09-29 (A23 real SSO) | Lens 9 / Lens 4 | +785.02% / +791.58% | +1570.04% / +1583.16% | 11.35 / 11.39 | 14.90 / 14.94 | **3.54 / 3.55** |
| 1993-02-01..2006-06-20 (A23 synthetic holdout) | Lens 4 | +256.34% | +512.68% | 9.96 | 14.50 | 4.54 |
| 1993-02-01..2026-09-29 (A23 stitched) | Lens 4 | +3077.08% | +6154.16% | 10.82 | 13.07 | 2.25 |
| 1926-07..2026-06 (century study) | Lens 4, French market | +1,922,351% | x2 | 10.37 | 11.13 | 0.77 |
| 1927-01..2026-04 (momentum A1) | Lens 9 / Lens 4, French | +1,679,584% | x2 | 10.29 | 11.06 | 0.77 |
| 1927-01..1993-12 (momentum pre-1994 diagnostic) | Lens 4, French | +59,629% | x2 | 10.01 | 11.15 | 1.14 |
| 1963-07..1993-12 (factor pre-1994 diagnostic) | Lens 4, French | +2,174% | x2 | 10.79 | 13.25 | 2.46 |
| 1963-07..2026-04 (factor, net-issuance A1) | Lens 9 / Lens 4, French | +63,838% | x2 | 10.83 | 12.06 | 1.23 |
| 1994-01..2026-04 (factor/momentum/net-issuance A2) | Lens 9 / Lens 4, French | +2,712% | x2 | 10.87 | 13.21 | **2.34** |
| 2013-01..2026-04 (screens' latest sub-period) | Lens 4, French | +529% | x2 | 14.79 | 20.16 | 5.38 |
| 2024-01-11..2026-10-23 (IBIT session split; SPY for scale only, crypto benchmark is not SPY) | Lens 9 | +64.76% | +129.52% | 19.66 | 34.81 | 15.15 (2.78 years) |
| 2026-06-24..2026-11-03 (paper account, for scale only) | Lens 4 | +3.09% | +6.18% | 8.71 | 17.88 | 9.17 (0.36 years) |

Sources: Lens 9 rows from `research/audits/root-cause-2026-11-04/lenses/_lens9_power.json` `regime.targets_2x_spy_total_return`;
Lens 4 rows recomputed from yfinance SPY with `auto_adjust=True` (dividend-adjusted closes,
last close before the window start to last close on or before the end) and from the frozen
French `F-F_Research_Data_Factors` monthly file (Mkt-RF + RF), the same copy Lens 9 used.
Where Lens 9 and Lens 4 differ (2022 and 2006 starts), the difference is a one-day choice of
starting close; both are shown and the JSON carries both. The French rows are the market
portfolio, not SPY.

Plain reading: the same "2x SPY" goal is a 9.6 pp/yr gap over five years, a 3.5 pp/yr gap over
twenty years, and under 1 pp/yr over a century, because doubling a large cumulative return
adds less per year the longer the compounding runs.

## Soundness tiers

Written before building the ceiling table. Inputs are Lens 2's per-row conditions
(`research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` `classifications[].data_condition`,
`code_condition`, `recompute_condition`, `class`).

- **Tier A:** Lens 2 class `reliable` (reliable NO or reliable GO).
- **Tier B:** Lens 2 class is not `unreliable`; Lens 1 tape clean, or every contamination item
  favours the strategy; Lens 7 recompute yes; and Lens 6 code match yes, or Lens 2 rule R3 met
  by an independent recompute of the statistic, or code unverifiable but irrelevant to the
  figure (exception L4-X1).
- **Tier C:** everything else. Listed for reference, excluded from the ceiling.

Exceptions, stated explicitly:

- **L4-X1 (code unexamined, fixed-weight books).** `stocks-a23-static-leverage`,
  `stocks-century-leverage`, `cross-asset-static-80-20-sleeve` and
  `cross-asset-sleeve-holdout-2007-2011` have no Lens 6 row or an "unverifiable" one. They are
  fixed-weight rebalances with no signal, on clean tapes, with numbers recomputed by Lens 7.
  Lens 4 admits them to Tier B because a code review would have no signal timing or look-ahead
  to check. This is a Lens 4 judgement; a reader who rejects it moves them to Tier C, which
  empties the leverage and cross-asset ceilings.
- **L4-X2 (non-registered arms and sub-windows).** The A23 1.5x and 2.0x arms, the century 60%
  arm and the French pre-1994 sub-periods sit in Tier B artifacts but were not the registered
  test. They are flagged "non-registered"; taking the highest of several arms is a selected
  maximum, chosen after seeing the values.
- **Not excepted:** the production replica (Lens 1 tape identity, PARA, direction unknown) and
  the FOMC day-clustered study (Lens 6 unverifiable on an event-study harness with signal
  timing) stay in Tier C.

What Lens 2's result does to the tiers: Lens 2 found **0 reliable NO and 0 reliable GO** among
72 test rows, and **9 unreliable GO**. Tier A is therefore empty. Every number that once showed
the stock work beating SPY by several points a year is in Tier C. The ceiling is built from
Tier B only, and in most Tier B rows Lens 9 could not compute power, so a Tier B figure here is
"recomputes on a clean tape", not "statistically established".

## Ceiling table

One row per approach and window (several rows where the approach has registered and
diagnostic windows; values are never combined across rows). "Kind" is always the CAGR gap
(geometric) unless stated; the CI column, where present, is on the arithmetic mean excess
that the t-statistic tests, per the fixed spec, and is **not** an interval on the CAGR gap.
"x gap" is ceiling / required gap for the same window.

| approach (inventory ids) | best id | best value pp/yr (kind) | 95% CI | tier | window | gap | x gap | exposure / leverage | costs | regime | pre-tax | after-tax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY sweep sleeve (`stocks-core-spy-blend-keep`, `stocks-cash-sweep-live`, `stocks-two-symbol-sweep-design`) | `stocks-core-spy-blend-keep/live_k1.0` | 0.00 (CAGR gap) | not computable from artifact | **C** (unreliable GO) | 2021-01-31..2026-04-30 | 9.57 | 0.00 | SPY 87.4%, core idle 92.41% / 1.0 | per driver | 2021-2026 | 0.00 | not in artifact |
| Ensemble/reversion selection (`stocks-production-replica`, `stocks-production-replica-holdout-2010-2020`, `stocks-sp500-core-pit-rebaseline`, `stocks-ensemble-ic-pit`, `stocks-ensemble-kelly-pit`, and the pre-fix ensemble rows) | `stocks-production-replica/primary` | -0.64 (CAGR gap) | [-2.93, 1.40] on arithmetic -0.76, NW(5) | **C** (tape identity unknown) | 2021-01-31..2026-04-30 | 9.57 | -0.07 | stocks 9.37%, SPY 85.33%, cash 5.30% / 1.0 | stocks 5 bp, SPY 1 bp per side | 2021-2026 | -0.64 | not run |
| Momentum, S&P 500 replay (`stocks-xs-momentum-corrected-replay`, `stocks-dual-momentum`) | `stocks-xs-momentum-corrected-replay/xs_momentum` | **+4.37** (CAGR gap) | [-7.07, 16.05] on arithmetic +4.49, NW(5), t 0.76 | **B** | OOS 2022-01-31..2026-04-30 | 9.54 | **0.46** | top-N stocks (anchor top_n 50 in 17/17 folds) / not recorded | not recorded in artifact | 2022-2026 | +4.37 | not in artifact |
| Momentum, French registered (`stocks-momentum-french-screen`) | `.../small/A2` (Big +0.08) | +0.20 (CAGR gap vs French market) | not computable (window mismatch) | **B** | 1994-01..2026-04 | 2.34 | 0.08 | value-weighted winners / 1.0 | haircut 3%/yr Small, 1%/yr Big (brief) | post-publication | +0.20 | not in artifact |
| Momentum, French pre-1994 diagnostic | `.../small/diag 1927-01..1993-12` (Big +3.12, t 3.56) | **+4.77** (CAGR gap vs French market) | not computable from artifact | **B** (L4-X2) | 1927-01..1993-12 | 1.14 | **4.18** (Big 2.73) | winners / 1.0 | as above | pre-publication | +4.77 | not in artifact |
| Factor tilts (`stocks-factor-screen-h1-h4`) | `.../H1/A2` | +0.83 (CAGR gap vs French market) | not computable (window mismatch) | **B** | 1994-01..2026-04 | 2.34 | 0.35 | BIG HiOP / 1.0 | haircut 1%/yr | post-1994 | +0.83 | not in artifact |
| Net issuance (`stocks-net-issuance-screen`) | `.../H1/A2` | +0.48 (CAGR gap vs French market) | not computable (window mismatch) | **B** | 1994-01..2026-04 | 2.34 | 0.21 | BIG NegNI / 1.0 | haircut 1%/yr | post-1994 (2009-2026: -0.93) | +0.48 | not in artifact |
| Event/calendar (`cross-asset-month-end-treasury`, `cross-asset-fomc-drift-corrected-harness`, `cross-asset-fomc-drift-wfo`, `stocks-turn-of-month-rotation`, `stocks-pead`, `stocks-index-deletion-fade`, `stocks-insider-cluster-buy`, `stocks-congress-trades`) | `cross-asset-month-end-treasury/portfolio/pinned/1bp` | -0.08 (CAGR gap) | not computable from artifact | **C** | 2021-01-31..2026-04-30 | 9.57 | -0.01 | 80% SPY + 20% rotating SPY->IEF / 1.0 | 1 bp | 2021-2026 | -0.08 | not in artifact |
| Cross-asset 80/20 (`cross-asset-static-80-20-sleeve`, `cross-asset-sleeve-holdout-2007-2011`, `cross-asset-sleeve-trend-arm`, `cross-asset-treasury-curve`, `cross-asset-fx-hedge-overlay`, `cross-asset-commodity-trend`) | `cross-asset-sleeve-holdout-2007-2011/1bp` | +3.08 (CAGR gap) | not computable from artifact | **B** (L4-X1) | 2007-06-01..2011-02-28 | -1.68 (degenerate) | n/a | 80% SPY, 20% TLT/GLD/DBC / 1.0 | 1 bp | GFC holdout | +3.08 | not in artifact |
| Static leverage, A23 registered arm (`stocks-a23-static-leverage`) | `.../main_real_sso/static 1.3x` | +1.61 artifact basis; +1.32 consistent basis (CAGR gap) | not computable from artifact | **B** (L4-X1) | 2006-06-21..2026-09-29 | 3.54 / 3.55 | 0.45 / 0.37 | 70/30 SPY/SSO / 1.3x | 1 bp/side + SSO fee | 2006-2026 incl. 2008, 2020; **T-bill funding averaged 1.64%** | +1.61 (artifact "pre-tax" label) | 24/15 +1.60; 40.8/23.8 +1.55 |
| Static leverage 1.3x, funding +1.5 points (`research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.015.json`) | `.../static 1.3x` | +1.16 artifact basis; +0.87 consistent basis | not computable from artifact | **B** (L4-X1) | 2006-06-21..2026-09-29 | 3.54 / 3.55 | 0.33 / 0.24 | 70/30 / 1.3x | as above | **funding about 3.1% (1.64% + 1.5)** | +1.16 (artifact label) | 24/15 +1.13; 40.8/23.8 +1.10 |
| Static leverage 1.3x, funding +2.5 points (`research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.025.json`) | `.../static 1.3x` | +0.81 artifact basis; +0.52 consistent basis | not computable from artifact | **B** (L4-X1) | 2006-06-21..2026-09-29 | 3.54 / 3.55 | 0.23 / 0.15 | 70/30 / 1.3x | as above | **funding about 4.1% (1.64% + 2.5), near today's bills** | +0.81 (artifact label) | 24/15 +0.80; 40.8/23.8 +0.79 |
| Static leverage 1.5x | `.../static 1.5x (50/50)` | +2.66 artifact basis; +2.37 consistent basis | not computable from artifact | **B** (L4-X1, X2) | 2006-06-21..2026-09-29 | 3.54 / 3.55 | 0.75 / 0.67 | 50/50 SPY/SSO / 1.5x | as above | **funding 1.64% only; no funding-sensitivity run for this arm** | +2.66 (artifact label) | 24/15 +2.58; 40.8/23.8 +2.52 |
| Static leverage 2.0x (post-hoc pick, best of five leverage arms) | `.../static 2.0x (SSO) no rebalance` | **+4.44 artifact basis; +4.15 consistent basis** | not computable from artifact | **B** (L4-X1, X2) | 2006-06-21..2026-09-29 | 3.54 / 3.55 | **1.25 / 1.17** | 100% SSO / 2.0x | as above | **funding 1.64% only; no funding-sensitivity run for this arm**; max DD -84.8%; behind SPY in 37.5% of rolling 10-year windows | +4.44 (artifact label) | 24/15 +4.29; 40.8/23.8 +4.26 |
| Century leverage 30% (`stocks-century-leverage`) | `.../30/full` | +1.24 (after-tax CAGR gap vs French market) | not computable from artifact | **B** (L4-X1) | 1926-07-01..2026-06-30 | 0.77 (pre-tax) | not computed (basis mismatch: after-tax value, pre-tax gap) | 30% 2x fund / 1.30x | 0.84% fund fee, spread fitted to SSO over 2006-2026 | 1926-2026 incl. 1929-32; funding = French T-bill rate per day | not in artifact | 24/15 +1.24 |
| Century leverage 60% | `.../60/full` | +2.13 (after-tax CAGR gap) | not computable from artifact | **B** (L4-X1, X2) | 1926-07-01..2026-06-30 | 0.77 (pre-tax) | not computed (basis mismatch) | 60% 2x fund / 1.60x | as above | as above | not in artifact | 24/15 +2.13 |
| LETF timing (`stocks-leveraged-rotation`, `stocks-leveraged-trend`) | none | no raw artifact | n/a | **C** | n/a | n/a | n/a | 2x/3x ETFs | n/a | n/a | n/a | n/a |
| Crypto (`crypto-*` rows) | pointer | Lens 8 | n/a | Lens 8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Key paths: replica `windows.primary.arms.primary.excess_vs_spy.{cagr_diff_pp,ann_excess_pct,t_stat}`
and `mean_exposure.*`; xs_momentum `xs_momentum.excess_stats.*` in
`research/audits/production-replica/_audit_reruns_20261019.json`; French `groups.<g>.post_cagr_diff_pp`,
`groups.<g>.diag[0]`, `hypotheses.H1.A2.cagr_diff_pp`, `H1.A2.cagr_diff_pp`; A23
`windows.main_real_sso.<arm>.<bracket>.{pre_tax_cagr,after_tax_cagr,max_dd}`; century
`scenarios["0.84%|24/15|wash=False|shiller"].<arm>.full.{after_tax_cagr,max_drawdown,ruin}`;
cross-asset `1bp.static.cagr_pct`, `bh.SPY.cagr_pct`. CIs follow the fixed spec (SE =
annualized mean excess / t; CI = mean +/- 1.96 SE), recomputed here: replica SE = 0.7620 /
0.6898 = 1.1046, CI = -0.762 +/- 2.165; xs_momentum SE = 4.4856 / 0.7606 = 5.898, CI = 4.486
+/- 11.56.

**Basis of the A23 figures.** The A23 driver pays tax out of NAV every April
(`scripts/static_leverage_aftertax.py:177-178`), so the artifact's `pre_tax_cagr` is the
taxed-NAV path, and the SPY comparator inside the artifact changes with the bracket (11.10% at
24/15, 10.93% at 40.8/23.8). "Pre-tax" in the A23 rows is therefore the artifact's label, not
a tax-free return. Two bases are shown and never mixed in one number:

- *Artifact basis:* arm `pre_tax_cagr` minus the artifact's SPY `pre_tax_cagr`, same bracket
  (24/15). 2.0x: 15.54 - 11.10 = +4.44; 4.44 / 3.54 = 1.25.
- *Consistent basis (used for the ratio):* arm `pre_tax_cagr` (24/15) minus SPY's true
  total-return CAGR over the same window from yfinance (11.39%, Lens 4; the gap on this basis
  is 3.55). 2.0x: 15.54 - 11.39 = **+4.15; 4.15 / 3.55 = 1.17.** 1.3x: 12.71 - 11.39 = +1.32
  (0.37). 1.5x: 13.76 - 11.39 = +2.37 (0.67). The arm's own figure still carries the tax paid
  on its small distributions (the 40.8/23.8 bracket shows 15.45 against 15.54), so this basis
  understates the arm slightly; with Lens 9's 11.35% and 3.54 the 2.0x figure is +4.19 (1.18).
- Synthetic 1993-2006 holdout, 2.0x: 10.92 - 9.64 = +1.28 artifact basis; 10.92 - 9.96 = +0.96
  consistent basis; gap 4.54 (0.21 on the consistent basis).

**Funding.** The 2006-2026 window's T-bills averaged 1.64%
(`beta_matched_pretax_cagr.rf_ann` in `research/allocation/static-leverage/_static_leverage_aftertax_results.json`);
the A23 report treats funding as the headline condition
(`research/allocation/static-leverage/2026-09-30-static-fractional-leverage.md:20-25`, `:116-131`). The two
funding-sensitivity artifacts present, `research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.015.json`
and `research/allocation/static-leverage/_static_leverage_aftertax_rf_plus0.025.json`, contain only the SPY,
1.25x and 1.3x arms (verified by listing `windows.main_real_sso` keys). For the 1.3x arm they
give +1.16 and +0.81 on the artifact basis (after tax at 24/15: +1.13 and +0.80). **No funding
run exists for the 1.5x or 2.0x arms**, so their figures hold only at the window's 1.64%
average funding.

Conditions that apply to every leverage row, stated plainly: leverage multiplies SPY's beta
and its drawdown. The project's measured leverage results are what they are: A23 1.3x
returned +1.61 pp/yr (artifact basis) over 2006-2026 with a -66.4% maximum drawdown against
SPY's -55.3%; the 2.0x SSO arm returned +4.44 (artifact basis) with -84.8%; the 1993-2006
synthetic holdout gave the 2.0x arm +1.28 against a 4.54 pp/yr gap with -79.2%; the century
study's 30% arm had a -91.4% drawdown in 1929-32 with its ruin flag set, and its 1x base arm
fell -84.6% in the same study. The artifact's beta-matched figure (`beta_matched_pretax_cagr`:
1.3x cash-financed at 1.64% risk-free = 13.94%) is above the SSO book's 12.71%, so the SSO
route returned less than the beta it carried would have at that funding rate. The rolling
10-year table (`rolling_10y_pretax_vs_spy`) shows the 1.3x book behind SPY in 23.8% of 10-year
windows (worst -3.29 pp/yr, ending 2009-03-02) and the 2.0x arm behind SPY in 37.5% (worst
-12.56 pp/yr, ending 2009-03-09).

Highest figure that passes the tiers, versus the target gap: the 2.0x SSO arm, **+4.15 pp/yr
over 2006-2026 against true SPY total return (+4.44 on the artifact's own basis), 1.17 times
the 3.55 pp/yr gap.** In plain words: it was picked after seeing the results, as the best of
five leverage arms in the artifact, not the registered 1.3x arm; it has no t-statistic and no
confidence interval; it counts as Tier B only because Lens 4 judged that unreviewed code does
not matter for a fixed-weight book; it was measured only at the window's 1.64% average
funding; and it is leverage, meaning more market exposure and a -84.8% drawdown in the
window, not stock-picking skill. Beside it, the registered 1.3x arm measured +1.61 at the
window's funding and +0.80 after tax at funding 2.5 points higher. Highest Tier B
stock-selection figure: **pre-1994 French Small momentum, +4.77, 4.18 times its window's 1.14
gap** (a non-registered sub-period); after 1994 the highest registered Tier B stock-selection
figure is **+0.83** (factor H1), 0.35 of the 2.34 gap. On the 2021-2026 window that the paper
account and the replica live in, the highest Tier B figure is **xs_momentum +4.37**, 0.46 of
the 9.54 gap, with a CI spanning -7 to +16.

## Ruled-out table

Wording fixed in advance: "ruled out at the size the test could detect" (a reliable NO with
an MDE at or below the size in question), "not tested", "tested badly" (Lens 5 class or Lens
2 unreliable), "tested, inconclusive" (measured, neither ruled out nor sound enough).

| lever | status in this project | evidence |
|---|---|---|
| Breadth (more independent bets: more names, small/micro caps, more asset classes, faster cadence) | **Not ruled out at any size.** Micro caps, weekly cross-sectional rebalancing and international equity exposure: not tested. Small-cap stock selection and fundamentals at small-cap scale: tested badly. Cross-asset breadth and French Small screens: tested, inconclusive. | Lens 5 gap table (`research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens5-coverage-gaps.md`, section "Gap table"); Lens 2: no reliable NO in any of these rows. |
| Leverage (more than 1x SPY) | **Not ruled out; measured.** Static leverage tested on a clean tape with recomputed numbers but no t-statistic (Lens 2 inconclusive, power not established). LETF timing (rotation, trend): tested, inconclusive on a pre-fix tape with no raw artifact. | `research/allocation/static-leverage/_static_leverage_aftertax_results.json`; `research/allocation/century-leverage/_century_leverage_results.json`; Lens 2 classes for `stocks-a23-static-leverage`, `stocks-century-leverage`, `stocks-leveraged-rotation`, `stocks-leveraged-trend`. |
| Concentration (few names at large weights) | **Tested badly.** The only concentrated arm (xs_momentum top_n 10) was never deployed OOS; the live/replica book holds 3 names at about 3.3% each (9.37% total). | Lens 5 gap table, row "Concentration"; `research/audits/production-replica/_production_replica_results.json` `mean_exposure`. |
| Long-short / shorting | **Tested, inconclusive; not ruled out (no MDE computable).** Pairs and short-volume-ratio books on a pre-fix tape, no borrow-fee model, no raw artifacts. | Lens 5 gap table, row "Long-short"; Lens 2 classes for `stocks-pairs-stat-arb`, `stocks-short-volume-ratio`. |

No lever is "ruled out at the size the test could detect": that wording needs a reliable NO
with known power, and Lens 2 has none. The one area Lens 5 marks "evidence against at the size
the test could detect" is the day-clustered FOMC Treasury effect (a calendar effect, not one of
these four levers), and Lens 2 still classes it inconclusive on the code check.

## How large firms operate

Free sources only. Quotes are verbatim from the page as fetched on 2026-11-04 through a text
extractor, or marked "search snippet" where only the search-engine excerpt was read.
Fund-return claims are secondary-source claims, not audited figures.

- **Fundamental Law of Active Management.** IR is approximately IC x sqrt(breadth). "The
  information coefficient (IC) is the correlation of the manager's forecast to actual
  returns." "Breadth represents the number of trading opportunities an investor encounters per
  year." (https://www.omscs-notes.com/machine-learning-trading/fundamental-law-active-portfolio-management/).
  Breadth counts independent bets: "The second variable is breadth; the number of independent
  investment opportunities."
  (https://www.robeco.com/en-int/insights/2018/04/fundamental-law-of-active-management-shows-way-to-higher-information-ratio).
  A known limit: "IC and breadth are not generally independent and the information ratio may
  or may not increase with increasing breadth." (search snippet,
  https://www.newfrontieradvisors.com/media/1180/information-ratio-of-factor-based-alpha.pdf).
  Arithmetic on this project's numbers: the replica's 3-name sleeve rebalanced on daily signals
  and the 50-name monthly momentum book are low-breadth books by this definition; a small IC
  times a small sqrt(breadth) gives a small IR, and IR times TE gives the excess. At the
  replica's TE of 2.61%, a 9.57 pp/yr excess needs IR 3.7.
- **Capacity and size (secondary sources).** Medallion: "From 1994 through mid-2014, it
  averaged a 71.8% annual return, before fees." and "The fund has been closed to outside
  investors since 1993 and is available only to current and past employees and their
  families." (https://en.wikipedia.org/wiki/Renaissance_Technologies, citing Bloomberg).
  "despite the fact that Medallion reported annual net returns above 29 percent every year
  between 2010 and 2018, the fund's assets under management stayed at about $10 billion
  throughout that period." (search snippet,
  https://www.institutionalinvestor.com/article/2bswymr8cih3jeaslxc00/portfolio/famed-medallion-fund-stretches-explanation-to-the-limit-professor-claims).
  The same firm's outside funds differ: "RIEF has historically trailed the firm's better-known
  Medallion fund" (Wikipedia, as above). Tax: "Medallion primarily generates short-term
  profits that are subject to much higher taxes than long-term gains" (search snippet,
  https://www.accountingtoday.com/articles/renaissance-hedge-fund-ends-no-fee-investment-perk-after-irs-complaints).
- **Costs at scale.** "our trading cost estimates are many times smaller and our fund sizes are
  more than an order of magnitude larger than those claimed in the literature."
  (https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies;
  the page also says short-term reversal strategies "do not survive trading costs at
  reasonable size").
- **Financing and leverage.** "Prime brokers facilitate hedge fund leverage, primarily through
  loans secured by the long positions of their clients." (https://en.wikipedia.org/wiki/Prime_brokerage).
  The owner's broker: "In order to trade on margin or sell short, you must have $2,000 or more
  account equity." The same page states an annual margin rate of 5.00% (elite) / 6.50%
  (non-elite) and 2x overnight buying power
  (https://docs.alpaca.markets/us/docs/margin-and-short-selling).
- **Multi-strategy pod structure.** Millennium runs a "platform model of investing made up of
  approximately 280 investment teams" (https://en.wikipedia.org/wiki/Millennium_Management;
  secondary source).
- **Data and infrastructure.** Renaissance staff "tap data in its petabyte-scale data warehouse
  to assess statistical probabilities for the direction of securities prices in any given
  market." (Wikipedia, as above).

| constraint at large firms | applies to the owner's setup? (evidence) |
|---|---|
| Capacity: returns fall as assets grow (Medallion held near $10bn; AQR break-even sizes) | **No** at the owner's size: a paper account of about $102K (Lens 3 `era_boundaries`, first snapshot 102,459.43). The record has not used this: micro caps never tested, small-cap selection tested badly (Lens 5). |
| Breadth from thousands of independent bets | **Limited by the setup:** 3 positions at about 3.3% (replica `config.risk.max_positions=3`, `position_pct=0.033`), one daily decision at 12:45 PT, monthly rebalance for weights strategies (`src/ggTrader/lab/data.py:266-271` per Lens 5). |
| Cheap financing through prime brokers | **Not available in the same form:** retail margin at 5.00-6.50% a year and 2x overnight (Alpaca docs, above); the leverage tested in this project used SSO, a 2x fund, not margin (`research/allocation/static-leverage/_static_leverage_aftertax_results.json`). |
| Trading costs | Commission-free at Alpaca; measured paper fill cost +2.5 bp mean before the close for stocks, -1.9 bp for SPY (Lens 3 `fills`); backtests charged 5 bp stocks, 1 bp SPY (replica config). |
| Tax: short-term gains taxed at higher rates | **Applies:** taxable account only (owner instruction, `CLAUDE.md`); 74 of 74 closed paper round trips were short-term (section "After-tax short-holding check"). Medallion's own short-term profile is reported above. |
| Proprietary data and infrastructure | **Applies as a constraint:** free data only (owner instruction); analyst revisions, option chains and anomaly-demand data listed infeasible without paid feeds (inventory rows `stocks-analyst-revisions`, `stocks-option-iv-skew`, `stocks-anomaly-driven-demand`). |
| Pod structure with many independent teams and risk limits | **Does not apply in that form:** one researcher-operator; 72 measured tests across 11 families (Lens 2 `multiple_testing`). |
| Shorting and borrow | **Available but untested with costs:** Alpaca allows shorting from $2,000 equity; the two long-short books modelled no borrow fee (Lens 5). |

Small-account advantages that exist and whether they were used (from Lens 5's table): no
capacity limit (micro caps never tested; small caps tested badly), concentration (tested
badly), short horizons (weekly never tested; intraday prospective only), instruments a large
fund may avoid such as options (never tested), shorting single stocks (tested, inconclusive).

## After-tax short-holding check

Label: **DERIVED.** Inputs: `paper_trades` (SELECT only; `amount = filled_qty x filled_price`,
`src/ggTrader/paper/trader.py:201-202`), stock BUY/SELL rows paired first-in-first-out by
symbol into round trips; SPY cash-sweep rows excluded; four still-open names excluded. The
production replica artifact holds only `trade_counts` (symbol, side, count), no holding periods
or per-trade gains, so **no after-tax result is claimed for the replica**, and none for any
strategy without trade lists.

Scope: stock round trips only; the 52 SPY cash-sweep rows (20 buys, 32 sells) are excluded.
Paper round trips, 2026-06-24..2026-11-04: 74 closed (54 strategy sells, 15 manual legacy
liquidations, 4 orphan liquidations, 1 catastrophe stop; matches Lens 3 `trades_by_reason`).
Holding period mean 30.1 days, median 25.5, maximum 92: **100% short-term.** Gross gains
$5,130.93, gross losses -$4,358.53, net realized **$772.39** on $126,647.63 of cost. Dividends
not included (Lens 3: the broker credited none).

| bracket (ST / LT) | tax on $772.39 if all ST | after tax | share kept, as ST | share kept, had it been LT |
|---|---|---|---|---|
| 24% / 15% | $185.37 | $587.02 | 76.0% | 85.0% |
| 37% / 20% | $285.79 | $486.61 | 63.0% | 80.0% |
| 40.8% / 23.8% (A23 artifact brackets, with 3.8% net investment income tax) | $315.14 | $457.26 | 59.2% | 76.2% |

How much of a pre-tax edge survives (DERIVED, stylised): a book whose total return is
realized short-term each year (as the paper stock trades are) against SPY held for the whole
window and taxed once at the long-term rate. Assumptions: constant annual returns; losses
offset within the year; dividend taxes ignored for both (this flatters SPY's after-tax figure
slightly); not the fixed spec's column 10 (that column uses artifact after-tax fields only).

| window (SPY pre-tax CAGR) | bracket | SPY after-tax CAGR | pre-tax edge to tie SPY after tax | +1.0 pre-tax -> after-tax gap | +4.37 -> | +9.57 -> |
|---|---|---|---|---|---|---|
| 2021-01-31..2026-04-30 (15.05%) | 24/15 | 13.28% | **+2.42** | -1.08 | +1.48 (34%) | +5.43 (57%) |
| 2021-01-31..2026-04-30 | 37/20 | 12.67% | **+5.05** | -2.55 | -0.43 | +2.85 (30%) |
| 2021-01-31..2026-04-30 | 40.8/23.8 | 12.19% | +5.53 | -2.68 | -0.69 | +2.39 (25%) |
| 2006-06-21..2026-09-29 (11.35%) | 24/15 | 10.57% | +2.56 | -1.18 | +1.38 (32%) | +5.33 (56%) |
| 2006-06-21..2026-09-29 | 37/20 | 10.29% | +4.97 | -2.50 | -0.38 | +2.90 (30%) |
| 2006-06-21..2026-09-29 | 40.8/23.8 | 10.06% | +5.64 | -2.75 | -0.75 | +2.33 (24%) |

Reading: in this stylised book the tax cost falls on the whole return (the market part as well
as the edge), so a pre-tax edge below about +2.4 pp/yr (24/15) or about +5 pp/yr (37/20)
becomes a shortfall after tax. The paper account is not this book: about 85% of it is SPY
bought by the cash sweep, and the stock sleeve is about 9.4%. The SPY core is not simply held:
the sweep sold SPY 32 times and bought it 20 times between 2026-06-24 and 2026-11-04
(`paper_trades` rows with `reason = 'cash_sweep'`; Lens 3 `trades_by_reason`), so it also
realizes gains, all short-term so far given the account's age. Those 52 SPY sweep rows are
outside the 74-round-trip check above, and the account's total tax drag is not computed here. Cross-check against an
artifact that has after-tax fields: the A23 study, which mostly holds rather than trades,
loses almost nothing to tax (1.3x: +1.61 pre-tax, +1.60 after tax at 24/15;
`research/allocation/static-leverage/_static_leverage_aftertax_results.json`).

## Questions and answers

1. **Measured ceiling: what is the highest raw return over SPY that sound evidence supports,
   per approach, with CI and conditions?** Answered in the ceiling table. Tier A is empty.
   Tier B: momentum replay +4.37 pp/yr (CI on arithmetic excess [-7.07, 16.05]); French
   momentum +0.08/+0.20 post-1994 and +3.12/+4.77 pre-1994; factor +0.83; net issuance +0.48;
   cross-asset +3.08 only in 2007-2011; static leverage +1.61 (1.3x, registered; +0.81 at
   funding 2.5 points higher) to +4.44 (2.0x, post-hoc; +4.15 against true SPY TR) over
   2006-2026 at 1.64% average funding, with drawdowns to -84.8%; century leverage +1.24 to +2.13 after tax with
   drawdowns to -95.7%. SPY sweep, ensemble selection, event/calendar and LETF timing: no
   Tier A/B figure. CIs are "not computable from artifact" wherever the artifact has no
   t-statistic. Crypto: unknown here, owned by Lens 8; cheapest test: read the Lens 8 report
   when committed.
2. **What return over SPY does the target require, and how does it compare with each measured
   effect?** 9.57 pp/yr (2021-2026), 9.54 (2022-2026), 6.14 (2010-2020), 5.11 (2012-2026),
   3.54 (2006-2026), 2.34 (1994-2026), 0.77 (1927-2026); full table above. Tier B effects
   reach their own window's gap only for the post-hoc 2.0x SSO arm (1.17x on a consistent
   basis, 1.25x on the artifact's basis, at the window's 1.64% funding) and pre-1994 French
   momentum (2.7x, 4.2x); the century arms are after-tax against a pre-tax gap, so no ratio is
   computed. The registered 1.3x arm reaches 0.37-0.45 at the window's funding and 0.15-0.23
   with funding 2.5 points higher. Post-1994
   registered stock-selection effects reach 0.03 to 0.35 of the 2.34 gap; the 2022-2026
   momentum replay reaches 0.46 of 9.54.
3. **Which of breadth, leverage, concentration and long-short were ruled out, by what
   evidence?** None was ruled out at the size a test could detect. Breadth: not tested or
   tested badly in most forms. Leverage: measured, inconclusive. Concentration: tested badly.
   Long-short: tested, inconclusive. Ruled-out table above.
4. **How do large firms generate returns, and which constraints apply to the owner?**
   Breadth times skill (Fundamental Law), capacity limits, cheap financing, low costs at
   scale, proprietary data, multi-team platforms. Applies to the owner: tax on short-term
   gains, free data only, retail financing rates, low breadth by construction. Does not
   apply: capacity. Table above. Unverified: the Medallion size and tax claims rest on
   secondary sources and search snippets.
5. **Which small-account advantages exist, and which were never used?** No capacity limit
   (micro caps never tested, small caps tested badly), concentration (tested badly), short
   horizons (weekly never tested; intraday prospective), options (never tested), shorting
   (tested, inconclusive). From Lens 5.
6. **After tax, do short holding periods turn a small pre-tax edge negative?** Yes, in the
   DERIVED stylised book: break-even +2.42 pp/yr (24/15) and +5.05 (37/20) over 2021-2026;
   a +1.0 pp/yr edge becomes -1.08 / -2.55. The paper book's realized stock gains were 100%
   short-term. Unknown: the after-tax effect for the replica (no trade list); cheapest test:
   a replica re-run that writes per-trade lots (Phase 2, owner-gated).

## Verification

Commands run (all read-only; repo code only through the capped runner):

- `python3` key dumps of `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json` (114 classifications, 182
  sensitivity rows; class counts inconclusive 103, unreliable GO 9, unreliable NO 2),
  `research/audits/root-cause-2026-11-04/lenses/_lens9_power.json` (`regime.targets_2x_spy_total_return`),
  `research/audits/root-cause-2026-11-04/lenses/_lens3_snapshot_gap.json`, `research/audits/root-cause-2026-11-04/lenses/_lens5_coverage_matrix.json`,
  the inventory (144 rows), and the artifacts named in the ceiling table.
- Spot recomputes from raw artifacts: replica `excess_vs_spy` (cagr_diff -0.6423, ann_excess
  -0.7620, t -0.6898; CI [-2.93, 1.40]); xs_momentum (cagr_diff 4.3674, ann_excess 4.4856, t
  0.7606; CI [-7.07, 16.05]); cross-asset holdout 1.3429 - (-1.7351) = 3.0781; A23 arms
  12.71 - 11.10 = 1.61, 13.76 - 11.10 = 2.66, 15.54 - 11.10 = 4.44; century 10.713 - 9.473 =
  1.24, 11.602 - 9.473 = 2.13.
- `scripts/run_capped.sh .venv/bin/python -I <scratch>/l4/target.py`: yfinance SPY
  (`auto_adjust=True`) 1993-01-01..2026-11-04 and the frozen French factors file; output
  reproduces Lens 9's 2021 window (+108.60%) and adds the windows marked "Lens 4".
- `scripts/run_capped.sh .venv/bin/python <scratch>/db_ro_t.py` (read-only session,
  statement timeout): `paper_trades` schema and all 205 rows (2026-06-24..2026-11-04, 99
  BUY, 106 SELL; 52 SPY sweep rows).
- `python3 -I <scratch>/l4/trips.py` (FIFO pairing) and
  `scripts/run_capped.sh .venv/bin/python -I <scratch>/l4/aftertax.py` (derived tables).
- `python3 -I <scratch>/l4/build_json.py` writes `research/audits/root-cause-2026-11-04/lenses/_lens4_ceiling.json`.
- Fix round 1: listed `windows.main_real_sso` keys in both funding-sensitivity artifacts
  (SPY, 1.25x, 1.3x only); read `rolling_10y_pretax_vs_spy` (2.0x: share 0.375, worst
  -0.1256 ending 2009-03-09; 1.3x: 0.238, -0.0329, 2009-03-02); read the tax-payment step in
  `scripts/static_leverage_aftertax.py:177-178`; recomputed the consistent-basis gaps
  (15.54 - 11.39 = 4.15, 4.15 / 3.55 = 1.169; 12.71 - 11.39 = 1.32; 13.76 - 11.39 = 2.37;
  11.91 - 11.39 = 0.52; 10.92 - 9.96 = 0.96); median hold recomputed with
  `statistics.median` = 25.5 days.
- Web: fetched the Wikipedia pages for Renaissance Technologies, Millennium Management and
  Prime brokerage, the OMSCS notes, Robeco, AQR and Alpaca pages; search snippets for New
  Frontier, Institutional Investor and Accounting Today.
- Gate: `.venv/bin/python scripts/check_lens_report.py docs/research/2026-11-04-lens4-feasibility-and-ceiling.md`.

Not run: no backtest, WFO or replica re-run (Phase 2 is owner-gated); no write to the
database, `data/`, the container, the broker or the intraday observer.

Unverified or limited:
- The tier exceptions L4-X1 and L4-X2 are Lens 4 judgements.
- Search-snippet quotes (New Frontier, Institutional Investor, Accounting Today) were not
  read on the full page; WebFetch quotes pass through a text extractor.
- The paper round-trip pairing assumes each SELL closes the whole position (true for the
  strategy's exits and liquidations by design; not checked per order against the broker).
- The stylised after-tax book is a model; its break-even depends on the assumptions listed.
- Crypto ceiling: not examined; Lens 8 report not committed at this time.

## Done checklist

- ✅ Falsifiers written before the evidence for the Phase 0 ceiling hypotheses.
- ✅ Steelman with survival against Lenses 1, 2, 6, 7 and 9.
- ✅ Target arithmetic per window, with sources and the worked example.
- ✅ Soundness tiers written before the ceiling table, exceptions stated, effect of Lens 2's
  0 reliable / 9 unreliable GO stated.
- ✅ Ceiling table per approach with value kind, CI where computable, tier, window,
  exposure/leverage, costs, regime, pre-tax/after-tax and times-in-gap.
- ✅ Ruled-out table for breadth, leverage, concentration, long-short.
- ✅ How large firms operate, with URLs and quotes, and the constraint table.
- ✅ After-tax short-holding check (DERIVED) on paper trade lists; replica marked
  not claimable.
- ❓ Crypto ceiling: unknown (Lens 8 owns it; report not yet committed). Cheapest test: read
  the Lens 8 report.
- ❓ CI for leverage, cross-asset, French-screen and century rows: unknown (artifacts store
  no t-statistic or mean excess). Cheapest test: re-run the drivers with a stored monthly
  excess series (Phase 2, owner-gated).
- ❓ Replica after-tax: unknown (no per-trade lots in the artifact). Cheapest test: replica
  re-run writing lots (Phase 2, owner-gated).
