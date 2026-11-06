# Lens 7: Research Accuracy (Root-Cause Investigation)

**Date:** 2026-11-04 · **Branch:** `investigation/root-cause-2026-11-04` · **Status:**
diagnosis only, read-only. No recommendations. · **Raw artifact:**
[`research/audits/root-cause-2026-11-04/lenses/_lens7_recompute.json`](_lens7_recompute.json)

**Terms.** WFO = walk-forward optimization. OOS = out-of-sample. CAGR = compound annual
growth rate. NW t = Newey-West t-statistic (a t-stat that allows for correlated returns).
PIT = point-in-time index membership. DSR = deflated Sharpe ratio (a gate in the lab).
Holm = a step-down correction for testing several hypotheses at once. pp = percentage
points. "Pass bar" = the rule, written in advance, that decides GO or NO-GO.

Lens 7 asks three questions from the investigation spec
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`, Lens 7):
are the reported numbers true; do the academic sources say what the reports say; and was
each verdict reached under a pass bar set before the result was known.

## Summary

| Question | Short answer |
|---|---|
| Are the headline numbers true? | Yes for every headline that has a raw file. I recomputed five of them from raw data with my own code (production replica, its holdout, momentum, factor and net-issuance screens) and they match to the stored precision. I read 27 more directly from their files. One mismatch (position sizing) is a labelling problem: the report's number comes from a different, earlier run that has no raw file. |
| How much has no raw file? | 103 of 144 inventory rows (72%) are prose only. These include most of the "beat SPY" history (1.12, 1.14, 0.84 Sharpe), the July event-study NO-GOs, the live paper-versus-SPY figure, and all crypto results except one IBIT file. |
| Do the academic sources say what the reports say? | 11 of 14 citations checked against free sources say what the reports say. One (the "profitability and investment were introduced 2013 to 2015" framing) is only partly accurate. Two were not examined. |
| Were pass bars set first? | For the October studies, mostly yes: 18 rows pre-registered, 4 amended, 1 late (momentum). The former research briefs folder (dissolved in the 2026-11-05 reorganization; each brief now sits in its study folder) starts on 2026-09-24. The 2026-09-10 strategy audit (`083afc7d`) is the earliest committed source of per-study "Success Criteria". Nothing earlier than 2026-09-10 has a committed per-study bar. 48 verdict rows have no brief, including all July NO-GOs and every early "beat SPY" claim. |

Findings that matter for the owner's question (raw return of 2x SPY):

1. **The production-replica NO-GO is numerically true.** From the raw curves
   (`research/audits/production-replica/_production_replica_curves.csv`) I get CAGR 14.1871% against SPY's
   14.8294%. That is -0.6423 pp/yr with NW t -0.6898, the same as the report.
2. **The stopped holdout did produce a completed primary arm.** It shows CAGR 15.29%
   against SPY 14.29% (+1.00 pp/yr, NW t 0.69, 2,750 days). The closure amendment marks it
   non-citable, and it sits on tainted prices (`citable` false, `stop_flags`
   ["tainted_trade:primary"]). Only the four sensitivity arms (lag0, cost3, cost10, cash0)
   never ran. The wording "no holdout arm completed" appears in three places: the
   investigation spec section 4, the results JSON's own `audit_stop.holdout` string, and
   the FOMC amendment (`research/audits/production-replica/2026-10-19-production-replica-fomc-timing-amendment.md:16-17`).
   All three describe an earlier attempt stopped with SIGTERM. The Phase 0 audit does not
   use that wording; it calls the holdout "unusable because of bad price data"
   (`research/audits/root-cause-2026-11-04/findings/2026-11-04-phase0-trust-audit.md:71-72`). Even taken at face value, the
   completed arm fails the t ≥ 3 bar.
3. **The production-replica verdict was reached under an amended rule.** The original
   brief said no verdict could be issued while live parity was below 100% or while an
   artifact stop was open. The closure amendment, committed nine seconds before the results
   commit and quoting the results, allowed a NO-GO on the primary failure. The original
   bars already made GO impossible, so the amendment did not turn a pass into a fail. It
   did turn "no verdict" into "NO-GO".
4. **The momentum brief is self-attested, not provably pre-registered.** The brief, the
   plan and the results were all added in one commit (`bf0e8279`). The French archive was
   downloaded about 3 h 20 min before that commit. The two-part standard also replaced the
   2026-10-19 t ≥ 3 bar after the xs_momentum replay (t 0.76) was already known.
5. **Every early "beat SPY" headline has neither a raw file nor a pre-registered bar.** All of them predate the 2026-09-10 audit, the earliest committed per-study criteria.
   These are the 1.12 Sharpe baseline, the 1.14 three-sleeve blend, the 0.84 "first to
   beat SPY" figure and the 16.2% CAGR. Lenses 1 and 6 already void most of them for other
   reasons.

## Falsifiers

Written before I opened any evidence. These are the Phase 0 ratings this lens touches
(`research/audits/root-cause-2026-11-04/findings/2026-11-04-phase0-trust-audit.md`, trust table), and what would overturn
each one.

| Phase 0 rating touched | What would overturn it |
|---|---|
| Production replica "Reliable NO, high trust", verified | Raw curves give a CAGR difference or NW t that differs from 14.19/14.83/-0.69; or the pass bar was set or changed after the primary result in a way that changed the verdict's direction. |
| PIT core WFO "Reliable NO" 0.59 vs 0.78 | Artifact Sharpe or CAGR differs from 0.59/3.1, or the SPY comparator in the artifact differs. |
| French momentum "Reliable NO", t 3.05/3.11, post-1994 +0.08/+0.20 | Recompute from the frozen French files gives a different t or post-1994 CAGR difference; the Holm threshold is misapplied; or the bar was written after the result. |
| Factor screen "Reliable NO", best t 1.34, ≤ +0.83 pp | Recomputed H1/H4 differ; or the brief was approved after the results. |
| Net issuance "Reliable NO", t 0.26, +0.48 pp | Recomputed BIG NegNI statistics differ; or the bar was set late. |
| A23 "Trust, medium", 12.0% vs 10.4% | Artifact after-tax CAGR differs; or the +1.0 pp bar postdates the result. |
| Phase 0 hypothesis 4 (Holm t ≥ 3 bars are unreachable) | Not mine to settle (Lens 2 and Lens 9). Here it is touched only through when the bars were set. |
| Phase 0 hypothesis 7 (+0.1 to +0.8 pp effects; momentum counter-evidence t 3.72/5.18 size-matched, t 3.56 pre-1994) | The diagnostic numbers are not in the artifact, or they were pass criteria rather than diagnostics. |
| Rows Phase 0 marked "prose" (parity, FOMC, month-end, century, July studies) | A raw artifact exists and disagrees with the prose; or no artifact exists, so "prose" cannot be upgraded. |

Result: none of the numeric falsifiers fired. Pre-registration findings 3 and 4 above
qualify the production-replica and momentum ratings on process, not on numbers.

## Steelman

The strongest evidence that the reported numbers are accurate:

- **Exact independent reproduction.** I wrote my own French CSV parser, Newey-West
  estimator and CAGR code (scratchpad, not the repo's `src/ggTrader/lab/momentum_screen.py`).
  Run against the frozen archives in `data/external/french/`, it reproduces the momentum
  Big t (3.0524342824851454), the Small t (3.107964991876554), factor H1 t 0.12936, H4 t
  1.33885 and net-issuance t 0.25896, each to about 1e-12. The post-1994 CAGR differences
  match to 1e-12 as well. That rules out transcription errors and hidden parameter changes
  in these screens.
- **The replica curves CSV reproduces the summary JSON exactly** for all five primary arms
  and the holdout arm. For example, cost10 gives 13.9973% vs 14.8294% (-0.8321 pp, calendar-day
  CAGR) and t -0.8422; the JSON has -0.8320 pp and t -0.8422.
- **The reports flag their own post-hoc elements.** The momentum report says its rebound
  and drawdown diagnostics "were specified after the NO-GO result; they were not
  pre-registered pass criteria" (`research/stock-signals/momentum/2026-10-20-momentum-french-screen-nogo.md:125-128`).
  The closure amendment keeps the raw INCONCLUSIVE verdict instead of relabelling it.
- **The October briefs and their results are well ordered.** Factor screen: approved 03:26,
  results 03:32. Net issuance: approved 11:18, results 11:31. Century leverage: approved
  13:18, results 15:52. Cross-asset sleeve: brief 19:37, results 20:23. Month-end Treasury:
  brief 07:24, results 07:50. In each case the brief came first, and the gaps are short
  enough that the results were not available when the brief was written.
- **Most academic citations check out word for word** (see the academic claims section).

Does the steelman hold? For numbers that have a raw file, yes. It does not reach the 103
prose-only rows. It does not cure the pre-registration gap before September either.

## Recompute verification

The Haiku pass covered 41 rows. Its raw output was 31 yes, 2 partial, 1 no and
7 cannot_locate. My verification:

| Source of check | Rows | What it means |
|---|---|---|
| Recomputed by me from raw data or curves | 5 | `stocks-production-replica`, `stocks-production-replica-holdout-2010-2020`, `stocks-momentum-french-screen`, `stocks-factor-screen-h1-h4`, `stocks-net-issuance-screen`. For the holdout row, only its CAGR and t were recomputed from the curves; coverage and tainted counts were counted from JSON keys. |
| Read by me from the artifact key (spot-check of Haiku) | 27 | All other headline rows (A23, A24, century, PIT core, ic/kelly PIT, xs_momentum, dual momentum, parity, cross-asset static/trend/holdout, month-end, FOMC, KEEP blend, pre-PIT ic/kelly, anchor-fix, corrected-tape, WFO policy), plus all 10 non-yes rows |
| Haiku output, not re-checked | 9 | Mostly event-audit and process rows with count headlines (see JSON) |
| Prose only (no raw artifact) | 103 | Not recomputable |

Of the 31 rows Haiku marked yes, I re-checked 22 (71%). The required minimum was 10%.

**The non-yes rows, resolved:**

| Row | Haiku | My finding |
|---|---|---|
| `stocks-position-sizing-regimes` | no | Confirmed mismatch, with a cause. The headline (0.70 / 4.97% / -7.64%) is the single ungated window in the report's table (`research/paper-system/position-sizing/2026-08-19-position-sizing-regimes.md:95`) and has no raw file. The JSON is a different run: the 6-fold gated WFO, 2023-10-30 to 2026-04-30 (regime_p 0.525; f_cap16 0.574; f_cap30 0.474; SPY 1.03). The report's addendum only says this run was "scheduled". Its results were never written into any report. |
| `stocks-ic-kelly-rebaseline-pre-pit` | partial | Matches: 0.614 / 33.95% / -14.67%. The SPY CAGR and drawdown in the headline come from `research/allocation/core-spy-blend/_core_spy_blend_20260924.json` (12.96 / -22.09), same window. |
| `crypto-ibit-overnight-session` | partial | Overnight +186.1% and intraday -42.5% match. The post-NGHT split (-9% / +23%) is prose only (`research/WEB_RESEARCH_CANDIDATES.md:1343-1345`). |
| `stocks-dual-momentum` | cannot_locate | The artifact confirms output identical to xs_momentum. The "+12.2% minimum momentum" figure is changelog prose only (`docs/changelog.md:1304`). |
| `stocks-event-audit-readiness-decision` | cannot_locate | 62,373 / 762 / 2026-07-17 is a database snapshot quoted in a brief. Prose only; not re-queried. |
| `stocks-event-timing-audit-decision` | cannot_locate | No number. PARK confirmed (`research/audits/event-timing/2026-10-23-event-timing-audit-decision.md:267`). |
| `stocks-family-scope-review` | cannot_locate | The table is in the report (`research/stock-signals/feasibility/2026-11-02-family-scope-review.md:31-37`), not in the JSON. |
| `stocks-intraday-feasibility` | cannot_locate | The numbers come from a paper. I verified them against the published paper (see the academic claims section). |
| `crypto-web-research-batch-2026-10-25` | cannot_locate | No headline number. |
| `crypto-wfo-textbook-reset` | cannot_locate | The snapshot has no fee-scenario or pass field. "35-36 validated, 0 after costs" is changelog prose (`docs/changelog.md:1667-1700`). |

**Mismatches found:** one real mismatch (position sizing: the reported number is from a
different run than its cited artifact). There are no numerical disagreements between any
artifact and the report that cites it. Two smaller notes:

- The PIT core artifact (`research/audits/rebaseline/_pit_rebaseline_20260923.json`) and the pre-PIT
  ic/kelly artifact (`research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json`) store SPY CAGR
  13.0% and MaxDD -22.1% only as text. Each has the line "SPY baseline: Sharpe 0.78 | CAGR
  13.0% | MaxDD -22.1%" three times in its `table` strings and no numeric key. Only
  `spy_sharpe` is numeric. The figures match; the same values are numeric keys in
  `research/allocation/core-spy-blend/_core_spy_blend_20260924.json` (12.96 / -22.09).
- The holdout's completed primary arm (finding 2) is a record-accuracy issue. The text
  "no holdout portfolio arms completed"
  (`research/audits/production-replica/2026-10-19-production-replica-fomc-timing-amendment.md:17`) and
  the JSON's `audit_stop.holdout` string describe the earlier SIGTERM-stopped attempt.
  `windows.holdout.arms` holds the completed `primary` arm. The JSON also has a `closure`
  block: verdict NO-GO, stage pretax_fail, holdout INCONCLUSIVE, live replication
  UNVERIFIED.

**Figures in this report and how each was checked:**

| Figure | Check |
|---|---|
| Replica 14.1871 / 14.8294 / -0.6423 / t -0.6898; holdout 15.29 / 14.29 / +1.00 / t 0.69 | Recomputed (curves CSV) |
| Momentum t 3.052 / 3.108, post-1994 +0.082 / +0.198; factor H1 / H4; NI t 0.259 / +0.482 | Recomputed (French archives) |
| A23 after-tax 12.00 vs 10.40, holdout 9.53 vs 8.89 | Read from artifact |
| KEEP CAGR 12.963 vs 12.959 | Read from artifact |
| Cross-asset static CAGR 13.61 vs SPY 14.70 | Read from artifact |
| FOMC net t 1.57 (unclustered) / 0.98 (clustered), WFE 0.24 | Read from artifact |

**Prose-only rows relevant to the owner's question** (no raw file, so they cannot be
verified):

- The "beat SPY" history: `stocks-ensemble-ndh-baseline-1-12` (Sharpe 1.12, CAGR 16.3%),
  `stocks-three-sleeve-blend-1-14` (1.14), `stocks-ensemble-wfo-validated-2026-06-20`
  (0.84 vs 0.59, CAGR 18.8%), `stocks-three-pct-position-size` (16.2% / 1.09),
  `stocks-bb-reversion`, `stocks-ensemble-ic-2026-06-28` and `stocks-midcap400-reversion`.
- All July event-study and sleeve NO-GOs: PEAD, insider cluster, congress, short interest,
  short volume, index deletion, MAX, pairs, headline sentiment, leveraged rotation and
  trend, commodity trend, FOMC WFO, FX hedge, Treasury curve.
- `stocks-live-paper-performance-vs-spy` (+0.41% vs SPY +4.35%) and
  `stocks-wfo-tournament-sp500-monthly`.
- `stocks-xs-momentum-2026-06` (+126.0% vs SPY +104.5% total return, with a lower Sharpe).
  This is the only early prose headline with raw return above SPY.
- Every crypto result except IBIT: the edge search (+13.8% vs BTC +38.8%), the textbook
  reset, the Kraken live PnL and funding carry.

## Academic claims

Network access worked in this session. Each claim was checked against a free source.
Quotes are from the fetched pages.

| Claim (where cited) | Source and quote | Verdict |
|---|---|---|
| French BIG = above the NYSE median; momentum uses prior 2-12 return at the 30/70 NYSE percentiles (`research/stock-signals/factor-screen/2026-10-21-factor-screen-nogo.md:28`) | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_6_port_form_sz_pr_12_2.html: "The monthly size breakpoint is the median NYSE market equity. The monthly prior (2-12) return breakpoints are the 30th and 70th NYSE percentiles." | Verified. The "~1,000+ names" count is not on the page. |
| French NI portfolios (`research/stock-signals/net-issuance/2026-10-23-net-issuance-screen.md:33`) | https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/tw_5_ports_me_NI.html: "NI for June of year t is the change in the natural log of split-adjusted shares outstanding ... The NI breakpoints are NYSE quintiles." | Verified |
| Momentum post-publication cutoff 1994-01 (`research/stock-signals/momentum/2026-10-20-momentum-retest.md:116`) | Jegadeesh and Titman, *Journal of Finance* 48(1), 65-91, March 1993 (https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html) | Verified as a date. The brief never names the paper. The report itself leaves "decay caused by publication" unverified. |
| "Profitability and investment formally introduced circa 2013-2015 (Novy-Marx 2013; Fama and French 2015)" (`research/stock-signals/factor-screen/2026-10-21-factor-screen-nogo.md:30`) | Both papers exist as cited. But Fama-French 2015 (https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002323) itself cites earlier evidence: "(See also Haugen and Baker, 1996, ... Titman et al., 2004, Fama and French, 2006 ...)". Novy-Marx measures *gross*, not robust, profitability. | **Partly accurate.** The publication date is earlier than stated. The verdict is unaffected because A2 starts in 1994. |
| Pontiff and Woodgate (2008) (`research/stock-signals/net-issuance/2026-10-23-net-issuance-screen.md:24`) | Abstract: "Post-1970, share issuance exhibits a strong cross-sectional ability to predict stock returns ... We estimate the issuance relation pre-1970 and find no statistically significant predictive ability." (https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01335.x; publisher page returned 403, abstract taken from the search index and RePEc) | Verified. The brief omits the post-1970 qualifier; the screen starts in 1963-07. |
| McLean and Pontiff (2016) | https://gwern.net/doc/economics/2016-mclean.pdf: "Portfolio returns are 26% lower out-of-sample and 58% lower post-publication." | Verified |
| Hou, Xue and Zhang (2020) | https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html: "65% of the 452 anomalies ... cannot clear ... t-value of 1.96 ... 2.78 ... raises the failure rate to 82%." | Verified |
| Hartley and Schwarz (2019), month-end Treasury (`research/allocation/month-end-treasury/2026-09-24-month-end-treasury-sleeve.md:62`) | https://www.kristaschwarz.com/EOM.pdf: "annualized Sharpe ratio of around 1"; "data from January 1990 to the end of 2018"; "around 20 basis points per month at the 10-year maturity". | Verified |
| Bianchi and Goldberg, leveraged ETFs (`research/allocation/static-leverage/2026-09-30-static-fractional-leverage-brief.md:60-66`) | https://arxiv.org/html/2604.27287v1: index "earning a cumulative return of 0.076%"; "SSO and UPRO were down, earning -11.09% and -28.24%"; Table 3 covariance -2.242 / -5.643. | Verified |
| Gao, Han, Li and Zhou (2018), Table 6: 6.67% and Sharpe 1.08 (`research/intraday/spy-momentum-feasibility/2026-10-24-intraday-trading-feasibility.md:24`) | *JFE* 129 (2018) 394-414, Table 6, row η(r1): "6.67∗∗∗ ... 1.08"; text: "resulting in a Sharpe ratio of 1.08". | Verified |
| SSO expense ratio 0.88% gross / 0.84% net | https://www.proshares.com/our-etfs/leveraged-and-inverse/sso: "Gross Expense Ratio 0.88%", "Net Expense Ratio 0.84%" | Verified |
| Shiller dividends end 2026-06 (century ruling r2) | Frozen copy of `ie_data.xls` in the repo's data folder: last non-blank D is 2026.06 | Verified (local copy, not re-fetched) |
| Smirnov and Dapporto / Cooper (Kelly context in the A23 brief) | — | Not examined (context only; sets no bar) |
| Baltussen et al. (2021) | — | Not examined (no verdict rests on it) |

No citation was found to misstate a number. The one partly accurate claim affects only how
the factor screen describes its window, not its verdict.

## Pre-registration audit

Method. For each brief and each results file I ran
`git log --follow --format='%h %ad %s' --date=iso`, then compared the brief's first and
approval commits against the first commit of the results JSON or report. I read each
brief's pass-bar text. Results: 114 inventory rows have a verdict string. Of those, 18 are
pre-registered, 4 amended, 1 pre-registered-late, 48 have no brief, and 43 are not
applicable (status, feasibility or audit labels rather than pass/fail results). The full
list is in the JSON under `preregistration`.

**Amended or late (the cases that matter):**

| Row | Status | Timeline (2026, PT) | Effect |
|---|---|---|---|
| `stocks-production-replica` | amended | Brief 10-19 15:56 (`9ee18ea3`). Parity result 16:20 (`77014bb2`). **Parity amendment 16:24** (`a89f31c2`). FOMC stop fired. **FOMC timing amendment 17:22** (`7320c1d6`). **Closure amendment 19:05:53** (`80981e25`). Results commit 19:06:02 (`8bc365ee`). | The original text said "No verdict is issued until" parity reaches 100% and the artifact stop and coverage floor resolve (`research/audits/production-replica/2026-10-19-production-replica-audit.md` §4). The closure amendment quotes the primary and holdout numbers, so it was written after them. It allows NO-GO on primary failure. Under the original bars GO was already impossible, so the amendment changes "no verdict" into NO-GO, never a pass into a fail. The raw verdict stays INCONCLUSIVE/stopped. |
| `stocks-production-replica-live-parity` | amended | Parity result 16:20, amendment 16:24 | The study was narrowed to a daily-bar model after parity failed. Exact replication stays UNVERIFIED. |
| `stocks-production-replica-holdout-2010-2020` | pre-registered | — | The stop fired as registered. The completed arm (finding 2) is preserved and non-citable. |
| `stocks-momentum-french-screen` | pre-registered-late | Brief, plan and results all first appear in `bf0e8279` (10-20 22:40). Archive downloaded 19:20 PT. | Self-attested "before any result existed" cannot be checked in git. The two-part standard replaced the 10-19 bar after xs_momentum (t 0.76) was known. Post-hoc diagnostics are disclosed. |
| `stocks-century-leverage` | amended (before results) | Approved 13:18; rulings r1 15:14 and r2 15:19; results 15:52 | The end date moved to 2026-06-30 because Shiller dividends end in June. Both rulings came before the results. |
| `stocks-event-timing-audit-decision` | amended (before decision) | Date-only insider amendment 10-22 08:17; decision 10-23 09:04 | Audit predicates, not a return bar. |

**Pre-registered and clean:** factor screen, net issuance, A23 (the brief discloses that
pre-tax sanity windows were already seen), A24, the cross-asset static and trend arms, the
cross-asset holdout, month-end Treasury, xs_momentum and dual momentum, the FOMC corrected
harness, WFO policy synthetic (`edc4d79`, 11 minutes before results), the event-audit
rows, ensemble_ic under PIT, and the pre-PIT ic/kelly run (both use the 2026-09-10 audit
criteria at `research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md:197`).
Small caveats:

- A23: the 2026-10-19 addendum puts the margin at today's bills at +0.80 pp/yr (+0.77
  under ordinary-income distributions), below the registered +1.0. The GO was kept as an
  owner allocation decision, not re-judged against the bar.
- ensemble_ic: the 09-10 text reads "MaxDD ≤ -10%", which literally points the wrong way.
  The 09-23 report adds "above the control". IC fails every reading.
- ensemble_kelly: the 2026-09-10 audit gives no Kelly-specific criteria. The 09-23 report applies the IC criteria to Kelly. For that reason `stocks-ensemble-kelly-pit` is "no brief found". The combined pre-PIT row `stocks-ic-kelly-rebaseline-pre-pit` is "pre-registered" for its IC half only, and its JSON note says the Kelly half has no bar of its own.
- FOMC: the clustered re-analysis came after the first result and lowered net t from 1.57
  to 0.98. That is the conservative direction, and both fail the registered bar of net
  t ≥ 3 and WFE ≥ 0.50 (WFE was 0.24).

**No brief found (48 verdict rows).** The former research briefs folder (dissolved in the 2026-11-05 reorganization) begins
2026-09-24 (first file `664cadee`). The earliest committed per-study criteria are the
"Success Criteria" items in the 2026-09-10 strategy audit
(`research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md:183-212`,
`083afc7d`, 2026-09-10 22:08). Those cover xs_momentum/dual_momentum, cross-asset trend,
constrained ensemble_ic, pre-FOMC drift and retail attention. Nothing committed before
2026-09-10 sets a per-study bar. Rows without a bar:

- **July studies** (PEAD, insider, congress, short interest, short volume, index deletion,
  MAX, pairs, headline sentiment, leveraged rotation and trend, commodity trend, FOMC WFO,
  FX hedge, Treasury curve): the only committed bar was the implementation prompt's
  "Current baseline — the bar to beat" (`research/prompts/local-implementation-prompt-TEMPLATE.md`,
  added 2026-07-16 in `46f9ff83`). It held the then-current baseline numbers, which
  included the 1.12 Sharpe later shown to be leak-inflated. None of these reports states a
  per-study pass bar.
- **Every early positive claim**: 1.12 "Validated", 1.14 blend "GO", 0.84 "Validated",
  BB reversion "beat SPY", NASDAQ core rejection, and the 5-voter ablation "best". None has
  a committed bar before its result.
- **2026-09-23/24**: the PIT core comparison was a plain comparison with SPY, not a bar.
  The KEEP blend rule is stated as pre-registered in the report
  (`research/allocation/core-spy-blend/2026-09-24-core-spy-blend-keep.md:29`), but report and JSON share one
  commit, so the order cannot be checked.
- **Crypto**: the edge search, the textbook reset, funding carry and the Kraken PnL.

The "stopped holdout" is recorded honestly: the raw verdict, stop flags and tainted trades
are all kept in `research/audits/production-replica/_production_replica_results.json`.

## Questions and answers

**Q1. Recompute every headline number from its raw results file and flag mismatches.**
Answered for all 41 rows that have a raw file (5 recomputed, 27 read and checked, 9 taken
from Haiku without a re-check). There is one mismatch, position sizing: the report's
number has no artifact, and the artifact's numbers were never reported. There are no
numerical disagreements. Partly unknown: 103 prose-only rows cannot be recomputed. The
cheapest test for the most decision-relevant ones is a pinned-window re-run per row, which
belongs to the Phase 2 list. For the July NO-GOs, re-running one driver with
`--eval-start 2021-01-31 --eval-end 2026-04-30` and writing JSON would cost about 30-60
minutes each.

**Q2. Do the academic claims relied on say what the reports say?** Answered: 11 of 14
verified, 1 partly accurate (the profitability/investment publication framing), and 2 not
examined (Kelly-context citations and Baltussen; no verdict depends on them). No cited
number was misquoted.

**Q3. Was each verdict reached under the pre-registered bar or one chosen after the
result?**
- From 2026-09-10 onward (the strategy audit's Success Criteria, then the briefs folder from 2026-09-24): mostly yes.
- Production replica: the verdict rests on an amendment written after the results. Its
  direction matches the original bars, but it replaced "no verdict" with NO-GO.
- Momentum: pre-registration is self-attested only, and the standard itself was rewritten
  after a related result.
- Before 2026-09-10: unknown-to-negative. Nothing committed earlier sets a per-study bar.
  48 verdict rows have no brief, including all early "beat SPY" claims and all July
  NO-GOs. The cheapest test is a
  search of private session transcripts for each brief's text and timestamp. Those are
  outside the repo and were not searched here.

**Steelman question (does the result under review hold up?).** The numbers do. The
evidence for an edge in the October screens is reproduced exactly. The momentum Big and
Small t-stats (3.05 and 3.11) exceed a single-test t ≥ 3 but fail the registered Holm step
(Small p 0.00094 > 0.000675). Both also fail the post-1994 +1.0 pp/yr economic bar by a
wide margin (+0.08 and +0.20 pp). The strongest raw-return evidence in the reproduced set
is the replica's tainted holdout arm (+1.00 pp/yr, t 0.69). It is non-citable, and even
at face value it is not significant.

## Verification

Commands run, all read-only:

- `git log --follow --format='%h %ad %s' --date=iso -- <brief>` for all 25 briefs, plus
  `--diff-filter=A` first-add dates for every inventory `source_report` and `raw_artifact`.
  Key outputs are quoted in the pre-registration tables above.
- `git log --since=2026-10-19T12:00 --until=2026-10-24 --name-status` to order the
  October brief, amendment and result commits.
- My own scripts, run with `.venv/bin/python -I` on scratchpad copies: `recompute_replica.py`
  on `research/audits/production-replica/_production_replica_curves.csv` (output: "primary primary n 1317 ...
  CAGR365 14.1871 vs 14.8294 NWt5 -0.6898"; "holdout primary n 2750 ... CAGR365 15.2857
  vs 14.2855 NWt5 0.6884"). The script's "CAGR365" label is a misnomer: it uses
  365.25-day years (calendar days / 365.25), which is the only convention that reproduces
  the stored figures. Holdout coverage (87.39%) and the tainted counts (21 matches, 11
  pairs) were counted from JSON keys, not recomputed from raw data. Only CAGR and t came
  from the curves., and `recompute_french.py` on unzipped frozen archives (output:
  "MOM big (3.0524342824851454, 0.0818628053081305, 1192)"; "H4 50/50 (1.338846128355209,
  -0.5592...)"; "NI H1 BIG NegNI (0.2589571335822779, 0.4822...)").
- `python3` JSON reads of each artifact key cited in `_lens7_recompute.json`.
- Shiller check: pandas read of the frozen `ie_data.xls`; the last dividend is in 2026.06.
- Web: `mcp__fetch__fetch_txt` on the French detail pages and arXiv; WebFetch on arXiv
  HTML and ProShares; PDFs of Gao et al. and Hartley-Schwarz fetched and extracted with
  `pdftotext`; SearXNG for the Pontiff-Woodgate, McLean-Pontiff, Hou-Xue-Zhang,
  Novy-Marx, Fama-French 2015 and Jegadeesh-Titman abstracts.
- Gate: `.venv/bin/python scripts/check_lens_report.py docs/research/2026-11-04-lens7-research-accuracy.md`
  (exit 0 before commit).

No database queries were run. No repository code was executed, so `scripts/run_capped.sh`
was not needed. No tests were run.

**What remains unverified:**

- The 9 Haiku rows I did not re-check.
- The 103 prose-only rows.
- A23, century and cross-asset values were read, not re-simulated.
- The event-audit database counts.
- Whether any pre-September study had a brief outside the repo.
- Two citations not examined.
- The Pontiff-Woodgate and Hou-Xue-Zhang publisher pages returned 403. Their abstracts
  came from search indexes and RePEc.

## Done checklist

- ✅ Recompute headline numbers from raw files and flag mismatches: 41 rows (5 recomputed,
  27 checked, 9 Haiku-only), 1 mismatch explained.
- ❓ Prose-only rows: unknown. 103 rows have no raw file; the cheapest test is a pinned
  re-run per row in Phase 2.
- ✅ Academic claims: 11 verified, 1 partly accurate, with URLs.
- ❓ Two minor citations: not examined (time); they set no bar.
- ✅ Pre-registration for October studies: classified, with commits.
- ❓ Pre-registration before 2026-09-10: unknown. No committed per-study bar exists in
  the repo before the 2026-09-10 audit; the cheapest test is a search of session
  transcripts outside the repo.
- ✅ Production-replica closure, parity and FOMC amendments and the stopped holdout:
  surfaced and timed.
- ✅ Falsifiers written before the evidence; steelman answered.
