# Momentum French screen: Phase 1 NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-10-20 (Pacific time)
**Audience:** Owner and quantitative research collaborators

## 1. Executive Summary & Core Engine Audit

**NO-GO. Momentum is closed and reported as NO-GO. No paid data.** Neither
company-size group passes: `pass.big = false`, `pass.small = false`, and `next`
records that exact closure in the [final results JSON](_momentum_french_screen_results.json).
The verdict follows the “Neither size passes” row of the
[registered brief §5](briefs/2026-10-20-momentum-retest.md), with the costs and
decisions in §11 unchanged. Phase 2 is not executed; Phases 2–5 are cancelled.
A pass would have authorized more research only, never deployment.

Big's full-window Newey-West t-statistic is **3.052434**, with one-sided
**p = 0.001134967** (`groups.big.full_t`, `groups.big.full_p`); Small's is
**3.107965**, **p = 0.000941902** (`groups.small.full_t`, `groups.small.full_p`).
Each group alone would clear t ≥ 3.0. The registered two-hypothesis Holm
correction decides significance test A1, however, and rejects neither
(`holm.big = false`, `holm.small = false`). The smallest p, Small's rounded
0.000942, exceeds the first Holm threshold, α/2 = 0.00135/2 = 0.000675
(thresholds: brief §5). Holm therefore stops before testing Big at the second
threshold; Big's standalone p does not rescue it.

Economic test A2 also fails independently of Holm. Since the registered
post-publication cutoff, Big's net compound annual growth rate (CAGR) difference
versus the market is **+0.08 percentage points per year (pt/yr)** and Small's
is **+0.20 pt/yr** (`groups.big.post_cagr_diff_pp`,
`groups.small.post_cagr_diff_pp`). Both are far below the registered +1.0 pt/yr
bar at base cost. Strong early returns do not establish a usable modern edge.

This is an independent-data historical screen, not a walk-forward out-of-sample
(OOS) test of our stock rules. The benchmark is the French market total return,
not the SPDR S&P 500 exchange-traded fund (SPY). It supplies no new SPY-relative
or after-tax implementation result. The
[production-replica audit](2026-10-19-production-replica-audit-nogo.md) remains
the separate closure of the deployed stock-picking construction. No validated
stock-trading candidate remains.

## 2. Quantitative Performance Context

All measured figures below come from the
[final JSON](_momentum_french_screen_results.json); each cell names its key.
Registered windows, costs and thresholds come from
[brief §§5 and 11](briefs/2026-10-20-momentum-retest.md), not estimated turnover.
The full window is 1927-01 through 2026-04; the post-publication window is
1994-01 through 2026-04. Newey-West uses 12 monthly lags. Base annual haircuts
are Big 1% and Small 3%; stress haircuts are Big 2% and Small 6%. The haircut is
subtracted monthly as annual cost divided by 12 before compounding.

Inputs are the frozen monthly value-weighted Big HiPRIOR and Small HiPRIOR
portfolios from Ken French's *6 Portfolios Formed on Size and Momentum*, and
*Fama/French 3 Factors*. Market total return is market-minus-risk-free plus
risk-free return. Big HiPRIOR is an analogue for large-cap winners, not an S&P
500 or top-50 replication. Source download times and checksums are recorded at
`provenance['6_Portfolios_ME_Prior_12_2_CSV.zip']` and
`provenance['F-F_Research_Data_Factors_CSV.zip']`. The archives and
[manifest](../../data/external/french/MANIFEST.json) were committed before the
screen in `357acf0`; the initial screen is `61ccf59`. Checksums and complete
registered monthly coverage were verified without downloading replacements.
Final-review diagnostics were regenerated offline from those same bytes;
the artifact retains its original `runtime_sec` as historical run metadata.

### Registered gates

| Group | Full-window t | One-sided p | A1: Holm rejection | A2: post-publication CAGR difference, base cost | Overall pass |
|---|---|---|---|---|---|
| Big | 3.052434 (`groups.big.full_t`) | 0.001134967 (`groups.big.full_p`) | false (`holm.big`) | +0.081863 pt/yr (`groups.big.post_cagr_diff_pp`), fails | false (`pass.big`) |
| Small | 3.107965 (`groups.small.full_t`) | 0.000941902 (`groups.small.full_p`) | false (`holm.small`) | +0.197551 pt/yr (`groups.small.post_cagr_diff_pp`), fails | false (`pass.small`) |

### Subperiod diagnostics at base cost

These are diagnostics, not additional pass tests. Dates are inclusive; the
final interval ends in April, not at year-end.

| Group and period | Inclusive dates | t | One-sided p | Net CAGR difference versus market |
|---|---|---|---|---|
| Big, 1927–1993 | 1927-01 (`groups.big.diag[0].start`) through 1993-12 (`groups.big.diag[0].end`) | 3.555825 (`groups.big.diag[0].t`) | 0.000188397 (`groups.big.diag[0].p`) | +3.120621 pt/yr (`groups.big.diag[0].cagr_diff_pp`) |
| Big, 1994–2012 | 1994-01 (`groups.big.diag[1].start`) through 2012-12 (`groups.big.diag[1].end`) | 0.543270 (`groups.big.diag[1].t`) | 0.293472 (`groups.big.diag[1].p`) | +0.783758 pt/yr (`groups.big.diag[1].cagr_diff_pp`) |
| Big, 2013–2026 | 2013-01 (`groups.big.diag[2].start`) through 2026-04 (`groups.big.diag[2].end`) | −0.556049 (`groups.big.diag[2].t`) | 0.710911 (`groups.big.diag[2].p`) | −0.970989 pt/yr (`groups.big.diag[2].cagr_diff_pp`) |
| Small, 1927–1993 | 1927-01 (`groups.small.diag[0].start`) through 1993-12 (`groups.small.diag[0].end`) | 3.376718 (`groups.small.diag[0].t`) | 0.000366781 (`groups.small.diag[0].p`) | +4.771228 pt/yr (`groups.small.diag[0].cagr_diff_pp`) |
| Small, 1994–2012 | 1994-01 (`groups.small.diag[1].start`) through 2012-12 (`groups.small.diag[1].end`) | 1.109303 (`groups.small.diag[1].t`) | 0.133650 (`groups.small.diag[1].p`) | +2.601949 pt/yr (`groups.small.diag[1].cagr_diff_pp`) |
| Small, 2013–2026 | 2013-01 (`groups.small.diag[2].start`) through 2026-04 (`groups.small.diag[2].end`) | −0.765980 (`groups.small.diag[2].t`) | 0.778156 (`groups.small.diag[2].p`) | −3.341792 pt/yr (`groups.small.diag[2].cagr_diff_pp`) |

The pattern is strong before 1994, smaller and statistically weak in 1994–2012,
and negative net since 2013. Small's middle-period gain is larger than Big's,
but its t-statistic is still weak; neither middle-period result establishes an
edge. The recent losses matter more to this decision than the early premium.
“Post-publication” names the brief's cutoff. A published-research claim that
publication caused decay is **unverified** here; these subperiods do not
identify a cause.

### Stress-cost diagnostics

| Group | Full-window t | One-sided p | Post-publication CAGR difference |
|---|---|---|---|
| Big | 1.555971 (`groups.big.stress.full_t`) | 0.059858 (`groups.big.stress.full_p`) | −1.014299 pt/yr (`groups.big.stress.post_cagr_diff_pp`) |
| Small | 0.929758 (`groups.small.stress.full_t`) | 0.176248 (`groups.small.stress.full_p`) | −3.066948 pt/yr (`groups.small.stress.post_cagr_diff_pp`) |

### Capital asset pricing model and size-matched diagnostics

The capital asset pricing model (CAPM) fits gross winners' returns less the
risk-free return to market returns less the risk-free return over the full
window. Alpha is the annualized arithmetic intercept; beta is market exposure.

| Group | Annual alpha | Beta |
|---|---|---|
| Big | 3.371359% (`groups.big.capm.alpha_ann_pct`) | 0.959956 (`groups.big.capm.beta`) |
| Small | 5.735061% (`groups.small.capm.alpha_ann_pct`) | 1.186636 (`groups.small.capm.beta`) |

The same-size comparator is the equal average of that size group's low,
middle and high momentum portfolio returns; each constituent portfolio is
value-weighted. It is not a capitalization-weighted reconstruction of the
entire size segment. These full-window comparisons are gross, with no cost
haircut.

| Group | t | One-sided p | CAGR difference versus same-size comparator |
|---|---|---|---|
| Big | 3.724889 (`groups.big.size_matched.t`) | 0.000097701 (`groups.big.size_matched.p`) | +3.663765 pt/yr (`groups.big.size_matched.cagr_diff_pp`) |
| Small | 5.183887 (`groups.small.size_matched.t`) | 0.000000109 (`groups.small.size_matched.p`) | +5.135458 pt/yr (`groups.small.size_matched.cagr_diff_pp`) |

**Diagnostic correction after the first run:** the size-matched diagnostic's
cost haircut was removed to match the brief's definition. The final review
also added the rebound and relative-drawdown diagnostics below to fulfill
brief §10. Their exact definitions were specified after the NO-GO result;
they were not pre-registered pass criteria. The optional `capm.alpha_t` was
removed because it treated estimated beta as fixed and was not a valid CAPM
intercept t-statistic. Alpha and beta point estimates are unchanged; no
replacement alpha significance test was built. The final JSON contains these
corrections. All original gate, subperiod and stress values reproduce exactly;
the base-cost A1/A2 calculations, costs, windows and pass logic are unchanged.
None of these diagnostic corrections overrides either failed gate.

### Worst rolling 12 months

These are gross relative wealth losses: compounded winners' wealth divided by
compounded market wealth, minus one. They are not subtraction of annual returns,
net-cost drawdowns or portfolio maximum drawdowns.

| Group | Start | End | Worst relative return |
|---|---|---|---|
| Big | 1932-06 (`groups.big.worst_12m.start`) | 1933-05 (`groups.big.worst_12m.end`) | −22.367595% (`groups.big.worst_12m.relative_pct`) |
| Small | 1937-03 (`groups.small.worst_12m.start`) | 1938-02 (`groups.small.worst_12m.end`) | −30.128742% (`groups.small.worst_12m.relative_pct`) |

### Rebound stress and relative drawdown

The rebound window includes the ten monthly returns from **2009-03 through
2009-12**. Relative return is compounded winners' wealth divided by compounded
French market wealth, minus one: Π(1 + r_win − c/12) / Π(1 + r_mkt) − 1.
Net uses the registered base annual haircut (Big 1%, Small 3%); gross uses zero.

| Group | Rebound net relative return | Rebound gross relative return |
|---|---|---|
| Big | −11.954055% (`groups.big.rebound_2009.net_pct`) | −11.240369% (`groups.big.rebound_2009.gross_pct`) |
| Small | −10.155327% (`groups.small.rebound_2009.net_pct`) | −7.956654% (`groups.small.rebound_2009.gross_pct`) |

Full-window relative drawdown uses that same wealth ratio at base cost over
**1927-01 through 2026-04**, initialized at one immediately before the first
return. It measures the largest peak-to-trough decline in the ratio, not the
portfolio's absolute drawdown. For tied peaks, use the last before the trough.
Recovery is the first month after the trough that regains the prior peak;
recovery months count elapsed calendar months from trough to recovery. Null
means no recovery by April 2026. These are French-market-relative diagnostics,
not SPY-relative diagnostics; neither series supplies a century of SPY returns.

| Group | Maximum relative drawdown | Peak | Trough | Recovery | Recovery months |
|---|---|---|---|---|---|
| Big | −31.688285% (`groups.big.relative_drawdown.max_drawdown_pct`) | 2008-06 (`groups.big.relative_drawdown.peak`) | 2025-08 (`groups.big.relative_drawdown.trough`) | null (`groups.big.relative_drawdown.recovery`) | null (`groups.big.relative_drawdown.recovery_months`) |
| Small | −56.848542% (`groups.small.relative_drawdown.max_drawdown_pct`) | 2006-03 (`groups.small.relative_drawdown.peak`) | 2025-07 (`groups.small.relative_drawdown.trough`) | null (`groups.small.relative_drawdown.recovery`) | null (`groups.small.relative_drawdown.recovery_months`) |

### Cost sensitivity: context, explicitly not a rescue

Derived from the base and stress rows, the post-publication CAGR difference
falls about **1.1 pt/yr for each extra 1%/yr of assumed cost**:

- Big: (+0.081863 − (−1.014299)) / (2 − 1) = **1.096161** pt/yr per
  cost percentage point (`groups.big.post_cagr_diff_pp`,
  `groups.big.stress.post_cagr_diff_pp`; cost assumptions: brief §5).
- Small: (+0.197551 − (−3.066948)) / (6 − 3) = **1.088166** in the same units
  (`groups.small.post_cagr_diff_pp`, `groups.small.stress.post_cagr_diff_pp`;
  cost assumptions: brief §5).

**Extrapolation, not a measured zero-cost run:** extending Big's base-to-stress
slope down to zero cost gives +0.081863 + 1.096161 × 1 = **+1.178024 pt/yr**,
roughly **+1.2 pt/yr** since 1994. This uses only
`groups.big.post_cagr_diff_pp`, `groups.big.stress.post_cagr_diff_pp` and the
registered costs. Compounding makes the relation nonlinear, so this is a local
linear extrapolation.

**Reasoning, not a measured after-tax result:** the registered base-cost screen
leaves almost no pre-tax margin and gives no evidence of the required
after-tax advantage. Aggregate monthly portfolios reveal neither turnover nor
tax lots; exact after-tax performance was not measured. This context is
not an additional gate or a replacement for the cancelled tax replay. The
verdict uses the registered costs, which are unchanged.

## 3. Actionable Research Directions

**No further momentum work.** The stock-trading path has no validated candidate.
Do not tune the screen, build its tax engine, buy data, or advance its paper or
real-money phases. No new strategy is proposed.

Per the 2026-10-20 research triage, the following remain **parked, all low
priority**, with no validated payoff claim: post-earnings announcement drift
(PEAD) needs public earnings-timing repair; insider trades and congressional
trades need filing-availability repair. These are data prerequisites, not
authorizations to resume the strategies. See §6.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**Momentum — REJECTED.** The approximately century-long French screen closes
the remaining research lead: `holm.big = false`, `holm.small = false`,
`pass.big = false`, `pass.small = false`. A2 fails for both groups regardless
of the multiple-testing result. `xs_momentum` and `dual_momentum` were the same
portfolio in the corrected replay, not independent supporting evidence
([brief §§2 and 4](briefs/2026-10-20-momentum-retest.md)). The French screen is
an idea-level test, not a rerun of those exact rules. Its closure does not
claim momentum never existed; it rejects further work under the registered bar.

**Deployed stock picks — CLOSED NO-GO.** The
[production-replica audit](2026-10-19-production-replica-audit-nogo.md) remains
the authority for that separate result. This report supplies no new live-parity
claim and does not rehabilitate the stopped older-price holdout.

## 5. Operational Roadmap: Recommended First Action

**Close momentum and retain the freeze.** The final gate text is:

> NO-GO. Momentum is closed and reported as NO-GO. No paid data. Neither size
> passes Phase 1 (`pass.big = false`, `pass.small = false`). Phase 2 is not
> executed; Phases 2–5 are cancelled. A pass would have authorized research only.

The next owner decision is unchanged: at the freeze lift, about 2026-12-29,
retire the stock picks so only the SPY sweep remains in the stock-trading path
([next steps](../next_steps.md), [brief §§1 and 8](briefs/2026-10-20-momentum-retest.md)).
This is a future owner action. Production remains frozen; this study changes
research code and documentation only. The owner's separate A23 allocation is unchanged.

## 6. Contrarian Evaluation & Parked Research

The strongest objection is that each size clears the standalone t bar, gross
CAPM and same-size diagnostics look stronger, and a cheaper implementation
could retain more of the premium. Those facts are reported in §2. They cannot
replace the pre-registered Holm family or the base costs after results are
known. Both groups independently fail the post-publication economic bar, and
the recent subperiod is negative net. No rescue experiment is authorized.

**Parked, low priority under the 2026-10-20 triage:** PEAD requires historical
public earnings-availability timing; insider and congressional signals require
historical filing-availability timing rather than the underlying transaction
date. Until those repairs exist and are audited, there is no clean timing
claim and no new performance verdict. Their historical reports remain caveated:
[PEAD](2026-07-17-pead-nogo.md),
[insider](2026-07-19-insider-cluster-buy-nogo.md),
[congressional](2026-07-19-congress-trades-nogo.md).

The screen does not measure turnover, investor taxes, sector concentration
or our exact trading rules. It reports the worst rolling relative windows,
the March–December 2009 rebound and full-window relative drawdown/recovery
against the French market. Published-research explanations,
including post-publication decay, remain **unverified**. None of these limits
changes the failed registered gates or reopens momentum.
