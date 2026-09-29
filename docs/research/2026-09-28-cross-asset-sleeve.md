# SPY + 20% Cross-Asset Sleeve (TLT/GLD/PDBC): Static Arm GO (Shadow Only), Trend Arm Adds Nothing

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-09-28
**Audience:** ggTrader owner & research collaborators

## 1. Executive Summary & Core Engine Audit

This candidate is `RESEARCH_SNAPSHOT.md` §6 Tier 1 #1. It holds 80% SPY and
20% split equally across long Treasuries (TLT), gold (GLD) and a broad
commodity fund (PDBC), rebalanced at every month-end close. Two arms were
tested exactly as the brief
(`docs/research/briefs/2026-09-28-cross-asset-sleeve.md`) fixed them before
any run:

- **Static:** the sleeve is always held.
- **Slow trend:** any leg whose trailing 252-day total return is ≤ 0 at the
  month-end close is swapped for BIL (T-bills).

Both arms are one `weights` strategy (`cross_asset_sleeve`), run through the
lab's own `select() → to_targets() → simulate_weights()` path. Plans are
decided at the month-end close and applied on the next bar.

**Verdict: the static arm is a GO under the pre-registered bar, which makes
it eligible for a paper-trading shadow and nothing more. The trend arm adds
nothing over static.**

- **Static passes all five criteria.** On the pinned window its Sharpe is 0.973
  vs SPY's 0.896, and on 2016–2026 it is 0.967 vs 0.872. MaxDD is −21.3% vs
  −24.5% and −28.4% vs −33.7%. In 2022 it lost 15.8% against SPY's 18.6%. The
  results are unchanged at 3 bp per side, and Sharpe still beats SPY with any
  one leg removed.
- **The pass is narrow, and the report should not be read as more than
  that.**
  - The gain is a volatility cut, not extra return. CAGR is *below* SPY on
    both windows (13.6% vs 14.7% pinned, 13.9% vs 15.0% long), and the book
    lagged SPY in 2021, 2023 and 2024.
  - On the pinned window the Sharpe gap is not statistically distinguishable
    from zero. A post-hoc block bootstrap (not pre-registered) gives a 90% CI
    of [−0.02, +0.18] for the gap, with P(book > SPY) = 0.90. On 2016–2026 the
    CI excludes zero: [+0.03, +0.16], P = 0.99.
  - Gold carries most of the pinned result. Drop GLD and the pinned Sharpe is
    0.909 vs SPY's 0.896, so criterion 5 passes by only 0.013.
- **The trend arm fails its bar on the pinned window.** Against static it
  gains only +0.007 Sharpe and 1.6 points of MaxDD, where the bar was ≥ 0.05
  Sharpe or ≥ 3 points of MaxDD. Its lookback WFO passed gates in 14/17 folds,
  and lookback 126 won 8/17. But the circuit breaker **halted the run in
  folds 2–8**, which fails the no-halt condition.
- **Against the live alternative:** on the WFO OOS span (2022-01-31 →
  2026-04-30), static scores Sharpe 0.87. That beats both `core + SPY sweep`
  (0.80, cited from `2026-09-24-core-spy-blend-keep.md` and not re-run) and
  SPY (0.79).

Already closed and not re-trodden here: month-end duration timing (A10),
slope/regime Treasury timing (A5), leveraged-ETF trend (`leveraged_trend_*`),
single-commodity cross-sectional trend (A3), and all equity
cross-sectional sleeves (`RESEARCH_SNAPSHOT.md` §2).

## 2. Quantitative Performance Context

Raw results are in `docs/research/_cross_asset_sleeve_results.json`, produced
by the driver `scripts/cross_asset_sleeve_wfo.py`.

- **Method.**
  - Prices are yfinance `auto_adjust` (total-return) closes from `ohlcv`
    (`venue='yfinance'`). The DB tape matched a fresh yfinance pull exactly
    for SPY, TLT, GLD and IEF over 2015–2026.
  - PDBC, BIL and IAU were added 2026-09-28 through `CachedYFinanceLoader` as
    naive 00:00 rows, the same convention as SPY.
  - Sharpe uses the lab's `curve_stats`: daily, rf = 0, √252.
  - Cost is 1 bp per side unless noted. Leverage is 1.0: weights sum to 1 and
    there is no margin.

| Configuration (pinned 2021-02-01 → 2026-04-30) | Sharpe | CAGR | MaxDD | Gates | Notes |
|---|---|---|---|---|---|
| **Static 80/20, 1 bp** | **0.973** | 13.6% | **−21.3%** | n/a | frozen rule |
| Static 80/20, 3 bp | 0.972 | 13.6% | −21.3% | n/a | monthly turnover is tiny |
| Trend 80/20 (252d), 1 bp | 0.980 | 13.5% | −19.6% | n/a | legs held: TLT 25%, GLD 75%, PDBC 70% of months |
| **SPY buy-and-hold** | 0.896 | 14.7% | −24.5% | n/a | primary bar |
| 60/40 SPY/IEF, monthly | 0.769 | 8.0% | −21.2% | n/a | reference line |
| TLT / GLD / PDBC / BIL buy-and-hold | −0.42 / 1.05 / 0.92 / n/m | −7.5% / 18.5% / 17.0% / 3.2% | −43.7% / −21.0% / −27.6% / −0.1% | n/a | BIL Sharpe is meaningless at rf = 0 |

| Configuration (2016-01-04 → 2026-04-30) | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| **Static 80/20, 1 bp** | **0.967** | 13.9% | **−28.4%** |
| Trend 80/20, 1 bp | 0.948 | 13.3% | −26.8% |
| **SPY buy-and-hold** | 0.872 | 15.0% | −33.7% |
| 60/40 SPY/IEF | 0.931 | 9.7% | −21.2% |
| TLT / GLD / PDBC buy-and-hold | 0.03 / 0.94 / 0.63 | −0.7% / 14.7% / 10.1% | −48.4% / −22.0% / −40.7% |

| Configuration (WFO OOS span 2022-01-31 → 2026-04-30, the snapshot's bar) | Sharpe | CAGR | MaxDD | Gates |
|---|---|---|---|---|
| **Static 80/20, frozen, 1 bp** | **0.87** | 12.4% | −20.4% | n/a |
| Trend 80/20, frozen 252d, 1 bp | 0.88 | 12.4% | −18.5% | n/a |
| Trend WFO, lookback {126, 189, 252}, 1 bp | 0.94 | 13.3% | −18.5% | 14/17, **halted folds 2–8**, anchor used in 7 |
| Deployed `core + SPY sweep` (cited) | 0.80 | 13.0% | −20.7% | n/a |
| **SPY buy-and-hold** | 0.79 | 13.2% | −22.1% | n/a |

- **Calendar 2022, the stress year.** Static −15.8%, trend −14.5%, 60/40
  −16.6%, SPY −18.6%. PDBC made +18.7%, which offset TLT's −29.4%.
- **Calendar years on the pinned window, static vs SPY:**

  | Year | Static | SPY |
  |---|---|---|
  | 2021 | 23.0% | 27.9% |
  | 2022 | −15.5% | −18.2% |
  | 2023 | 21.4% | 26.2% |
  | 2024 | 21.1% | 24.9% |
  | 2025 | 18.8% | 17.7% |
  | 2026 YTD | 7.6% | 5.7% |

### Pre-registered pass bar

| Criterion | Result | Pass |
|---|---|---|
| S1 Sharpe > SPY on pinned **and** 2016–26 | 0.973 > 0.896; 0.967 > 0.872 | ✓ |
| S2 MaxDD no worse on both | −21.3 vs −24.5; −28.4 vs −33.7 | ✓ |
| S3 2022 return no worse | −15.8% vs −18.6% | ✓ |
| S4 S1+S2 hold at 3 bp | 0.972 / −21.3; 0.967 / −28.4 | ✓ |
| S5 S1 holds with any leg removed | pinned: −TLT 1.072, −GLD **0.909**, −PDBC 0.923; long: 0.994 / 0.922 / 0.975 | ✓ (−GLD by 0.013) |
| T-a trend beats static by ≥ 0.05 Sharpe, or ≥ 3 pts MaxDD at no Sharpe loss | +0.007 Sharpe, +1.6 pts | ✗ |
| T-b WFO gates ≥ 12/17 | 14/17 | ✓ |
| T-c no regime halt | halted folds 2–8 | ✗ |
| T-d same lookback wins ≥ 8/17 | lookback 126: 8/17 (189: 7, 252: 2) | ✓ |

### Reported, not used to select

| Diagnostic | Pinned | 2016–2026 |
|---|---|---|
| Sleeve 10% (static / trend) Sharpe | 0.925 / 0.927 | 0.917 / 0.907 |
| Sleeve 30% (static / trend) Sharpe | 1.020 / 1.037 | 1.021 / 0.992 |
| Sleeve 30% static MaxDD | −19.7% | −25.7% |
| Sleeve-only (1/3 each) Sharpe / MaxDD | 0.85 / −21.1% | 0.85 / −21.1% |
| C4 pre-screen: SR_sleeve vs ρ(sleeve, SPY)·SR_SPY | 0.85 > 0.21 × 0.90 = 0.18 ✓ | 0.85 > 0.12 × 0.87 = 0.10 ✓ |
| ρ monthly, all months | 0.42 | 0.36 |
| ρ monthly, SPY's worst 10% of months | **0.84** (n = 7) | 0.40 (n = 13) |
| Mean return in SPY's worst 10% of months (sleeve / SPY) | −1.1% / −6.8% | −1.1% / −7.3% |
| Same, per leg (TLT / GLD / PDBC) | −4.4% / −1.0% / +2.1% | −1.2% / −0.3% / −1.7% |
| POST-HOC block bootstrap, Sharpe(book) − Sharpe(SPY), 90% CI | [−0.02, +0.18], P(>0) 0.90 | [+0.03, +0.16], P(>0) 0.99 |

The sleeve fell *with* SPY in its worst months on the pinned window
(ρ 0.84, driven by TLT in 2022), but it fell about a sixth as far. It works
as a shock absorber, not a hedge.

## 3. Actionable Research Directions

Ranked under current constraints. The live account is ~$100K paper on Alpaca
and fully invested via the SPY cash sweep. There is no rebalance-to-weights
path in the live trader.

### Rank 1: Paper-trading shadow of the static 80/20 book

**Mechanism.** Track the frozen static rule as a *shadow* book: 80% SPY and
6.67% each of TLT, GLD (or IAU) and PDBC, rebalanced at each month-end close.
Record it daily beside the live book and SPY. It needs no signal and no
parameters.

**Why it differs from rejected work.** A10 failed because its SPY-funded
overlay swapped one month-end premium for another (0.909 vs 0.896, worse
MaxDD). This sleeve holds different risk factors *all the time*: duration,
gold and commodities, with daily ρ to SPY of 0.12–0.21. It doesn't time a
calendar effect. It is also the first non-equity candidate to beat SPY on
both Sharpe and MaxDD in both windows.

**WFO framework.** None needed. The static arm has zero parameters, so a
walk-forward is degenerate. The shadow's job is forward, genuinely unseen
data, which this study doesn't have (§6).

**Payoff / Effort / Failure.**
- **Payoff:** unvalidated forward. Backtested, it gives about +0.08 to +0.10
  Sharpe and 3–5 points less MaxDD vs SPY, for about 1 point of CAGR given
  up.
- **Effort:** S–M. A shadow ledger could reuse the lab path. A live sleeve
  needs a new rebalance-to-target path in `paper/`, which is a separate,
  explicit ask.
- **Primary risk:** gold's 2024–26 run reverses. Without GLD, the pinned edge
  is 0.013 Sharpe.

### Rank 2: Pre-2016 holdout with DBC as the commodity leg

**Mechanism.** Re-run the frozen static rule on 2007–2015 with DBC in place
of PDBC (the two tracked identically where they overlap in the 2026-09-28
sanity check) and BIL from 2007. That period has not been seen by any
check.

**Why.** This is the only untouched out-of-sample data available (§6), and
it includes 2008 and the 2013 taper. Effort: S; it is a flag on the driver
plus a DBC backfill. It is reported, not a new selection.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Slow trend timing on the TLT/GLD/PDBC sleeve — REJECTED as an
improvement over static.**
- **Frozen 252d rule vs static:** +0.007 Sharpe and +1.6 pts MaxDD pinned;
  −0.019 Sharpe over 2016–2026.
- **Lookback WFO {126, 189, 252}:** OOS 0.94, but the circuit breaker halted
  folds 2–8 and the anchor fallback ran in 7 folds.
- **Why it fails:** the 30%-sleeve trend variant does no better. The gain
  comes from diversification, not timing, which matches the 2026-09-28
  sanity check. Don't re-propose trend timing on these legs with a different
  lookback.

**B. 60/40 SPY/IEF as an alternative — REJECTED.** Sharpe is 0.769 pinned vs
SPY's 0.896, and CAGR is 8.0%. Its 2016–2026 MaxDD is better (−21.2%), but
the Sharpe is below the 80/20 static.

## 5. Operational Roadmap: Recommended First Action

**Stand up a shadow ledger for the static 80/20 book and run it for the
~3 months the owner has before any real-money decision. Do not change the
live trader in this step.**

A GO under the brief means "eligible for a shadow", not "deploy". The edge
is small, and on the pinned window it is inside bootstrap noise. It is also
gold-concentrated, and it came from windows the sanity check had already
seen.

- **Before real money:** run Rank 2's pre-2016 holdout.
- **Deciding real money:** the explicit comparison is shadow vs SPY vs
  `core + SPY sweep` over the shadow period.
- **Live notes (owner decision, not done here):**
  - Use IAU over GLD (cheaper, identical research result: 1.06 vs 1.05 pinned
    buy-and-hold).
  - Gold ETFs are taxed as collectibles in a taxable account.
  - Use PDBC, not DBC (K-1).
  - SGOV is an alternative to BIL, but its history starts in 2020.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question: is this just "own some gold since 2021"?**

Partly, yes.
- Gold was the best asset in the pinned window (GLD Sharpe 1.05, +143%).
- Removing it leaves a book that ties SPY (0.909 vs 0.896).
- TLT was a drag: dropping it *raises* the pinned Sharpe to 1.07.

Three things argue against pure hindsight:
- The 2016–2026 window includes years before gold's run, and there the edge
  survives removing any leg (worst case 0.922 vs 0.872), with a CI that
  excludes zero.
- 2022 was the only full calendar year in which stocks and bonds fell
  together, and the book did better than SPY in it.
- The C4 pre-screen passes by a wide margin on both windows.

**What the evidence can't answer:**
- **No untouched holdout.** The 2026-09-28 sanity check had already looked at
  both windows before this brief was frozen, so "OOS" here means "frozen
  rule", not "unseen data". Rank 2 addresses this.
- **Multiple testing is light.** The pre-registered family is small: 2 arms,
  plus 3 lookbacks for the trend arm only. Across the research program,
  though, this is one of dozens of sleeve ideas tried (`RESEARCH_SNAPSHOT.md`
  §2), so the expected maximum Sharpe from noise alone is not trivial.

**Resolution:** one decisive experiment, the shadow plus the pre-2016
holdout, then deploy or close.

### Parked Direction: Live rebalance-to-target sleeve

The live trader has no path to hold a standing weights sleeve. It does
signal buys plus the SPY cash sweep, and the nearest existing code is the
blend-era `save_rebalance_state`. **Gate:** the shadow survives ~3 months
and the owner explicitly asks for a live change.

### Parked Direction: 30% sleeve

The 30% sleeve scored higher (1.02 static on both windows, MaxDD −19.7% /
−25.7%), but it was reported only, not selected. Picking it now would be
selection on the results. **Gate:** re-test as its own pre-registered brief
on the pre-2016 holdout.
