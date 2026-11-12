# Strategy categories: range of returns above or below the market

2026-11-12, evidence summary, no recommendation to trade. Every strategy, with its numbers, is in the [appendix](2026-11-12-strategy-categories-alpha-range-appendix.md).

## Read this first

- This report groups the project's research and shows return gaps against each study's benchmark, the comparison investment.
- A range shows how far a result could plausibly vary. A spread of results across studies is not an uncertainty band.
- True alpha here means return above simply holding the market, after costs and tax, out of sample, and statistically reliable.
- No category shows a gap above the market large enough and certain enough to count as proven alpha. The closest results are below, and none is proven.
- Many old numbers use the legacy engine and were not re-run.

## How to read a range

- Basis points (bps) measure return differences. 100 bps is one percentage point a year.
- A best estimate is the measured gap. A low-to-high band shows approximate uncertainty around it.
- A band crossing zero means we cannot tell whether the result is above or below its benchmark.
- The tax line is about +240 bps before tax. An audit estimated it as what a book selling within a year needs to tie SPY after tax over 2021–2026.
- Compare a pre-tax gap with the tax line with care. Most screens use a stand-in market return, not SPY, over 60 years, not 2021–2026. No band sits wholly above the line.
- The two leverage comparisons (century and static) are already after tax, so they are not held against the line.

| Category | Studies (with a number / total) | Spread of gaps, bps | Bands where available | Plain read |
|---|---:|---|---|---|
| Stock-selection screens: pick stocks by company or price traits | 19 / 24 | -1,810 to +618 | Of eight full-window bands, two sit above zero, one below and five cross zero | Mixed. The strongest are still uncertain. |
| Live design and its variants: change the stock portfolio and its rules | 8 / 23 | -1,092 to +590 | None | Mostly behind the market. |
| Allocation and leverage: mix assets or add market exposure | 10 / 12 | -1,280 to +160 | None | Mostly behind. Leverage was ahead after tax, with deeper drawdowns. |
| Event and calendar timing: trade around events or dates | 2 / 3 | -1,320 to -1,300 | None | Cash-heavy results, not comparable to always-invested SPY. |
| Crypto (parked): trade coins with different rules | 4 / 9 | -2,500 to +6,980 | None | The high result was a replay on the data used to pick the settings. |
| Intraday: record opportunities within a trading day | 0 / 2 | None yet | None | A forward-looking recorder with no result yet. |

## The results closest to the line

These screens assume a 1% annual cost and are before tax. They use a stand-in market return, not SPY. They are published portfolios run back over history, not tests on held-out data. Bands are approximate and not adjusted for trying eight screens.

| Screen | Full-window gap (band), bps | Since 1994 gap (band), bps | Why it remains uncertain |
|---|---|---|---|
| Accounting score | +406 (+47 to +765) | -109 (-669 to +450) | The estimate turns negative since 1994. |
| Expected investment growth | +208 (+56 to +361) | +264 (+58 to +469) | The only one with a band above zero in both windows. Its t-statistic of 2.728 is under the project's bar of 3, its inputs cannot yet be verified, and eight screens were tried. |
| R&D | +618 (about 0 to +1,237) | +324 (-248 to +897) | Exploratory, fixed before the calculation but not independently registered, and equal-weighted. Its worst fall was -79.32%, and it trailed badly in 2011–2025. |

Cross-sectional momentum, in a corrected replay against SPY over 2021–2026, was ahead by 4.37 percentage points a year (+437 bps). Its t-statistic of 0.76 gives a band of -711 to +1,584 bps, so the result cannot be told from luck.

## Category detail

**Stock-selection screens.** The eight long-window screens compare stock portfolios with a stand-in market return, not SPY. Their bands pair a compound-return gap with the t-statistic of mean monthly excess return, so they are approximate. A t-statistic near zero makes a very wide band by construction. The screens overlap and are not independent, and eight tries raise the chance that at least one looks good.

Monthly reversal (-204 bps) and low variance (-205 bps) trailed in both windows. The low-variance full-window band (-408 to -2) sits wholly below zero.

Long-history screens of momentum, net issuance and the factor screen show gaps of +0.08 to +0.83 percentage points a year after 1994, all below the project's bar of +1.0.

Ten older ideas trailed SPY by 70 to 1,810 bps, mostly in rolling out-of-sample tests on the legacy engine, with no band. They are insider and Congress trades, post-earnings drift, short interest and volume, the max effect, index-deletion fade, pairs, idiosyncratic volatility and a 50-symbol headline-sentiment pilot.

**The live design and its variants.** Eight rows give about five independent numbers: two pairs repeat the same run, and one is an archived run the project marks as invalid. Most compare with SPY on differing bases and use the legacy engine without a re-run.

The production replica re-runs the live rules on history. It returned 14.19% a year against 14.83% for SPY over 2021–2026, or -64 bps. Its t-statistic of -0.69 cannot be told from zero.

The MidCap 400 reversion result was +590 bps against MDY before tax. It is a June legacy result that predates the data and leak fixes, and it was not re-run. The source says it leaned on a fallback setting and applies a deduction for survivorship bias. The comparable S&P 500 version fell to a Sharpe ratio of 0.59, or 3.1% a year against SPY's 13.0%, once point-in-time membership was used.

**Allocation and leverage.** The century study's 30% SSO mix returned +124 bps after tax against unlevered SPY, but its worst fall was -91.4% versus -84.6%, and it crossed the study's ruin flag. The static study's 30% SSO mix returned +160 bps after tax against SPY. Its own funding-cost check cut the margin to +0.80 percentage points a year at about 4% bill rates.

Both are the same 30% SSO mix, so they are not two independent results. The static 70/30 SPY/SSO mix is the allocation the owner adopted on 2026-09-30. These gains reflect greater market exposure, not stock-picking skill, and no uncertainty bands are available.

The 80/20 cross-asset sleeve trailed SPY by 110 bps (13.6% against 14.7%) with a higher Sharpe ratio (0.973 against 0.896). It is in shadow, with no rating yet. Other comparisons use the legacy engine and were not re-run, and their benchmarks and tax treatment differ.

**Event and calendar timing.** The overnight gap and turn-of-month results were -1,320 and -1,300 bps against SPY before tax, using the legacy engine without a re-run. Both sit in cash most days, unlike always-invested SPY. The FOMC drift study has no gap here because its source says the comparison is inappropriate.

**Crypto (parked).** The +6,980 bps result against Bitcoin was a replay on the same data used to pick the settings, with a hindsight-selected basket on a pre-repair price tape, so it is not an out-of-sample result. The -2,500 bps comparison was also against Bitcoin. These legacy-engine results were not re-run, and their cost and tax treatment differ.

**Intraday.** Two studies exist. The historical feasibility screen is unresolved for lack of usable data and has no score. The forward-looking recorder has no performance result or benchmark comparison yet.

## What this report does not show

- It is not a recommendation to trade.
- It is not tax advice.
- Bands are approximate and unadjusted for multiple tries.
- Many older results use the legacy engine and were not re-run.
