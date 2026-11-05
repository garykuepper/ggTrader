# Lens 9: statistical power and regime

**Date:** 2026-11-04. **Branch:** `investigation/root-cause-2026-11-04`.
**Scope:** read-only diagnosis for the root-cause investigation
(`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md`, section 5, Lens 9).
No recommendations, fixes or deployments.
**Raw results:** `docs/research/_lens9_power.json` (all 144 inventory ids, once each).
**Binding method:** `docs/research/2026-11-04-lens2-sensitivity-spec.md` sections 4 (column 11)
and 9 (field map). Lens 2 and Lens 9 use the same standard error per row.

Abbreviations, expanded once: SE = standard error; MDE = minimum detectable effect (the
smallest true effect a test finds with 80% probability); NW = Newey-West (a standard error
that allows for autocorrelated returns); IR = information ratio (annualized excess return
divided by tracking error); TE = tracking error (annualized standard deviation of the return
difference against the benchmark); WFO = walk-forward optimization; WFE = walk-forward
efficiency; OOS = out of sample; PIT = point in time; TR = total return (dividends
reinvested); CAGR = compound annual growth rate; pp = percentage points; bp = basis points
(0.01%); FOMC = Federal Open Market Committee; IBIT = iShares Bitcoin Trust ETF; T-bill = US
Treasury bill; Holm = the Holm step-down correction for testing several hypotheses at once.

## Summary

1. **Only 7 of 144 inventory rows have the inputs for an MDE; 34 have an artifact that lacks
   them, and 103 have no raw artifact.** The seven: the production replica (primary window and
   the non-citable 2010-2020 holdout), the xs_momentum replay, dual_momentum (identical
   numbers to xs_momentum, so not independent), the day-clustered FOMC event study, and the
   IBIT session split (two inventory rows, one artifact). The long-history factor, momentum
   and net-issuance screens store a NW t-statistic but no mean excess, so their MDE in pp/yr
   is not computable from the artifact (spec field map). Most WFO artifacts (PIT core, core
   plus SPY blend, cross-asset sleeve, month-end Treasury, A23 static leverage) store no
   t-statistic at all.
2. **The detectable effect depends on how far the book strays from SPY, not only on the
   five-year sample.** At t >= 3 and 80% power:
   - production replica, 2021-01-31..2026-04-30: **MDE 4.24 pp/yr** (t >= 2: 3.14), required
     IR 1.62, because the book is about 85% SPY (TE 2.61%);
   - xs_momentum OOS: **MDE 22.66 pp/yr** (t >= 2: 16.76), required IR 1.82, TE 12.44%;
   - FOMC, net, day-clustered: **30.1 bp per event**, about 2.37 pp/yr in a labelled
     annualization.
3. **Against the owner's target (raw TR = 2 x SPY's over the same window):** over
   2021-01-31..2026-04-30 SPY returned +108.60% (yfinance, dividend-adjusted), so the target
   is +217.21%, a CAGR of 24.63% against SPY's 15.05%, a **required gap of 9.57 pp/yr**. The
   replica, its holdout and the FOMC study could have detected a target-sized effect at t >= 3.
   The xs_momentum (and dual_momentum) replay could not, even at t >= 2. Over the
   replica's stock exposure (9.37% of the book), a 9.57 pp/yr account-level gap would need
   roughly +102 pp/yr from the stock sleeve (arithmetic illustration, labelled).
4. **Phase 0's power claims, tested.** "t >= 3 needs IR near 1.3" holds for a test with 50%
   power over 5.24 years (3 / sqrt(5.24) = 1.31). At 80% power the requirement is 1.68
   (independent returns) or 1.62 (the replica's own NW SE). "Pass bars unreachable on five
   years of data" holds for effects of the size the long-history screens measured (about
   -0.6 to +0.8 pp/yr after 1994): power below 12% at 2 pp/yr even in the low-TE replica.
   It does not hold for a target-sized effect in a low-TE book.
5. **Regime.** The 2021-01-31..2026-04-30 window contains **one** bear market (SPY -24.50%,
   2022-01-03 to 2022-10-12, recovered 2023-12-13) and one -18.76% decline in 2025. SPY CAGR
   was 15.05%, against a Fama-French market CAGR of 10.29% over 1927-2026. T-bill yields
   went from about 0% to 5% and back to about 4%. **37 inventory rows rest only on 2020-2026
   data**, among them the replica primary, the PIT core, ensemble_ic and ensemble_kelly,
   xs_momentum, the core plus SPY blend and month-end Treasury. The long-history screens
   cover 7 to 10 bear episodes. Their subperiod diagnostics show that the measured premia
   are absent or negative in every post-1994 subperiod, not just in 2021-2026.

## Falsifiers

Written before any MDE or regime statistic was computed. They apply to the two Phase 0
hypotheses on power (`docs/research/2026-11-04-phase0-trust-audit.md:67-68`, and
hypothesis 4 of the spec's section 4).

- **H-IR ("t >= 3 needs IR near 1.3").** Overturned if the IR implied by the artifacts' own
  SE at t = 3 over the 2021-01-31..2026-04-30 window is far from 1.3 (outside about 1.1 to
  1.5), or if the NW SE differs from the independent-returns SE by more than about 20%.
  Partly overturned if the 1.3 figure only holds at 50% power and the 80%-power figure is
  materially higher.
- **H-unreachable ("pass bars unreachable on five years of data").** Overturned if, for a
  material 2021-2026 test, the t >= 3 MDE is at or below the effects the project has
  actually measured on long histories (about +0.1 to +0.8 pp/yr after 1994, per Phase 0
  hypothesis 7) or at or below about 2 pp/yr. Overturned for the owner's question if
  every material 2021-2026 test could detect a target-sized (2 x SPY) effect. Confirmed if
  the MDEs are several times the measured long-history effects.
- **H-single-regime (spec question 2).** "Conclusions depend on 2021-2026 being one regime"
  is overturned if the 2021-2026 verdict rows are also supported by longer or earlier
  windows in their own artifacts, or if the window contains several independent bear
  episodes and a long-run-typical return. Confirmed if most 2021-2026 verdicts have no
  other window and the window holds one bear episode with above-average returns.

## Steelman

The strongest evidence that the tests had adequate power:

- **The replica could see a target-sized edge.** Its primary-window MDE is 4.24 pp/yr at
  t >= 3, under half the 9.57 pp/yr the 2 x SPY target needs over the same window. The
  noncentrality at 9.57 pp/yr is 8.7 SE (power about 100%). At +5 pp/yr power is 94%. The
  replica's NO is therefore strong evidence against an edge of target size *in that book*
  (`windows.primary.arms.primary.excess_vs_spy.*` in
  `docs/research/_production_replica_results.json`). The 2010-2020 holdout had the same
  MDE (4.22 pp/yr) on a second, independent decade. That run is not citable because of
  tainted trades.
- **The long-history screens had long samples.** The factor screen's A1 test uses 754
  months (`inputs.common_window.n_months` in `docs/research/_factor_screen_results.json`),
  and momentum A1 runs 1927-01..2026-04. Momentum's observed t (3.05 big, 3.11 small) is
  0.79 and 0.81 of the t needed for 80% power at t >= 3, so those tests could resolve
  effects close to the size they found. The factor hypotheses' t of 0.13 to 1.34 are 3% to
  35% of that requirement, so their premia are small relative to the noise, not hidden by
  a short sample.
- **FOMC had 113 distinct event dates.** The net MDE of 30.1 bp per event is about four
  times the observed 7.7 bp. The test could find an effect a few times larger than the
  one measured.

**Does it hold up?** Partly. For a target-sized edge in a book that tracks SPY closely,
yes. For small edges (1 to 2 pp/yr), no: the replica has 2% to 12% power at t >= 3. For any
book with momentum-like tracking error, no: xs_momentum could not detect a target-sized
effect (8% power at t >= 3, 35% at t >= 2). The required effect also shrinks with window
length (the 2 x TR target is 9.57 pp/yr over 5.24 years, but 0.77 pp/yr over 1927-2026 on
the French market), so long-sample power does not carry over to the owner's horizon.

## Method

- **SE and MDE** (spec section 4, column 11): `SE = ann_excess / t` from the artifact, one
  method per row; `MDE(bar) = (bar + 0.8416) x SE` for bar = 3 and 2. **Required IR** =
  MDE / artifact `tracking_error_pct` (only where the artifact stores it). TE is never
  used as the SE.
- **Units.** The MDE is on the arithmetic annualized mean excess. The owner's target is a
  CAGR (geometric) gap. Arithmetic excess is roughly the CAGR gap plus half the difference
  in variances, so comparing the two is approximate (label: estimate).
- **Rows whose artifact stores only t** (factor, momentum, net issuance): no MDE in pp/yr.
  Lens 9 reports `t / (3 + 0.8416)`, the observed effect as a fraction of the t >= 3 MDE.
  That needs no mean and no tracking error.
- **FOMC and IBIT** test per-event or per-session means, not SPY excess. The per-event or
  per-session MDE is primary. Annualized figures multiply by events or sessions per year
  (count / years in the artifact window) and are labelled derivations.
- **SPY TR** from yfinance `auto_adjust=True` closes (dividend-adjusted), downloaded
  2026-11-04, not from the DB tape, which has adjustment seams (Lens 1:
  `docs/research/2026-11-04-lens1-backtest-validity.md`). The replica artifact's own SPY CAGR
  for the same window is 14.83% (DB tape), against 15.05% from yfinance.
- **French market** = `Mkt-RF + RF` from the frozen archive
  `F-F_Research_Data_Factors_CSV.zip` under `data/external/french/` (read only), parsed with
  `src/ggTrader/lab/french_data.py`.

## MDE table (all 144 inventory rows)

Exploratory. Not a pass bar; the original pre-registered bars decide verdicts. Spec rules:
"no raw artifact" when the inventory `raw_artifact` is null; "not computable from
artifact" with the missing key otherwise. Computed SEs are in pp/yr of annualized mean
excess unless the units are stated.

| # | id | status | window | n | SE method | SE (pp/yr) | MDE t>=3 (pp/yr) | MDE t>=2 (pp/yr) | required IR t>=3 | missing key or note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `stocks-a23-static-leverage` | not computable from artifact | 2006-06-21 → 2026-09-29 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (windows.{w}.{arm}.{b} carry CAGR fields only; spec appendix). |
| 2 | `stocks-a24-sleeve-trend-rider` | not computable from artifact | 2006-06-21 → 2026-09-29 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (windows.{w}.{arm}.{b} carry CAGR fields only; spec appendix). |
| 3 | `stocks-accruals-feasibility` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 4 | `stocks-after-tax-lot-engine` | no raw artifact | 2021-01-31..2026-04-30 (planned B0 kill-only replay; not run) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 5 | `stocks-analyst-revisions` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 6 | `stocks-anomaly-driven-demand` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 7 | `stocks-avb-eqr-merger-holding` | no raw artifact | 2026-03-30 → 2026-09-25 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 8 | `stocks-bb-reversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 9 | `stocks-cash-sweep-live` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 10 | `stocks-catastrophe-stop` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 11 | `stocks-century-leverage` | not computable from artifact | 1926-07-01 → 2026-06-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (scenarios.* carry CAGR, drawdown and recovery fields only). |
| 12 | `stocks-congress-trades` | no raw artifact | 2021-01–2026-04 (matched, 17 folds); 2015-06–2026-07 (long, 40 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 13 | `stocks-conviction-bb` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 14 | `stocks-core-spy-blend-keep` | not computable from artifact | 2021-01-31 → 2026-04-30 (WFO OOS span 2022-01 → 2026-04) | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (live_k1.0 has CAGR/Sharpe only; spec appendix). |
| 15 | `stocks-daily-threshold-tilt` | no raw artifact | Pinned 2021-02 → 2026-04; Long 2011-03 → 2026-04 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 16 | `stocks-dsr-gate-correction` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 17 | `stocks-dual-momentum` | computed | 2021-01-31..2026-04-30 | n_obs_daily 1064, n_folds 17 | NW(5), daily OOS | 5.90 | 22.66 | 16.76 | 1.82 | observed t 0.76, ann. excess +4.49 pp/yr, TE 12.44%; identical to xs_momentum, not independent |
| 18 | `stocks-ema-cross` | no raw artifact | 2021-2026 (as stated; no explicit end date) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 19 | `stocks-ensemble-conviction` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 20 | `stocks-ensemble-ic-2026-06-28` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 21 | `stocks-ensemble-ic-pit` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: SPY CAGR and t-statistic (only {s}.spy_sharpe; spec appendix). |
| 22 | `stocks-ensemble-kelly-2026-07` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 23 | `stocks-ensemble-kelly-pit` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: SPY CAGR and t-statistic (only {s}.spy_sharpe; spec appendix). |
| 24 | `stocks-ensemble-ndh-baseline-1-12` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 25 | `stocks-ensemble-paper-deployment` | no raw artifact | 2021-01 → 2026-04 (17-fold rolling 12mo/3mo) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 26 | `stocks-ensemble-wfo-validated-2026-06-20` | no raw artifact | 2021-01 → 2026-04 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 27 | `stocks-equity-diversification-sleeves` | no raw artifact | 2021-01 → 2026-06 (standalone sleeves, fixed-param) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 28 | `stocks-equity-pre-fomc-drift-intake` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 29 | `stocks-event-audit-readiness-decision` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 30 | `stocks-event-evidence-resolution-workbook` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 31 | `stocks-event-timing-audit-decision` | not computable from artifact | registered assessment ranges, not a run: PEAD 2002–2025; insider 2015–2025 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 32 | `stocks-exit-rule-sweep` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 33 | `stocks-factor-screen-h1-h4` | not computable from artifact | A1 1963-07..2026-04; A2 1994-01..2026-04 | A1_n_months 754, A2_n_months 388 | NW(12), monthly (t only) | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: annualized mean excess (hypotheses.{g}.A1 holds only t, p, Holm fields; spec appendix). Observed t: H1 0.13 (= 0.03 of the t>=3 MDE); H2 1.01 (= 0.26 of the t>=3 MDE); H3 0.66 (= 0.17 of the t>=3 MDE); H4 1.34 (= 0.35 of the t>=3 MDE) |
| 34 | `stocks-family-scope-review` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 35 | `stocks-form-144-inversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 36 | `stocks-forward-paper-standard` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 37 | `stocks-free-data-stock-feasibility` | not computable from artifact | B0 replay on clean tape 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 38 | `stocks-headline-sentiment` | no raw artifact | 2024-07-21 to 2026-07-21 (pilot sample backfill window as stated) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 39 | `stocks-ic-kelly-rebaseline-pre-pit` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 40 | `stocks-idio-vol` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 41 | `stocks-index-deletion-fade` | no raw artifact | 2015-01-01 → end not stated (42 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 42 | `stocks-insider-acceptance-time-diagnostic` | not computable from artifact | 2015–2025 (insider); 2026 through April 30 diagnostic only | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 43 | `stocks-insider-cluster-buy` | no raw artifact | 2021-01–2026-04 (matched, 17 folds); 2015-06–2026-07 (long, 40 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 44 | `stocks-insider-identity-inventory` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 45 | `stocks-institutional-rebalancing-tilt` | no raw artifact | Pinned 2021-02 → 2026-04; long 2011-03 → 2026-04 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 46 | `stocks-intraday-feasibility` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 47 | `stocks-intraday-replication-source-audit` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 48 | `stocks-july-research-program-summary` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 49 | `stocks-leveraged-rotation` | no raw artifact | 2019-01-01 → end not stated (also 2010-06-30 variant) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 50 | `stocks-leveraged-trend` | no raw artifact | eval_start=2019-01-01 (end not stated) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 51 | `stocks-live-ensemble-defaults-vs-pit-picks` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: SPY CAGR and t-statistic (only {s}.spy_sharpe; spec appendix). |
| 52 | `stocks-live-exit-parity` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 53 | `stocks-live-paper-performance-vs-spy` | no raw artifact | 2021-01 → 2026-04 (primary window) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 54 | `stocks-live-revert-blend-to-core` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 55 | `stocks-live-signal-lookback-365d` | not computable from artifact | 2026-03-30 → 2026-09-25 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (warm-up parity record, not a return test). |
| 56 | `stocks-macd-divergence` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 57 | `stocks-max-effect` | no raw artifact | 2015-01-01 → present | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 58 | `stocks-midcap400-reversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 59 | `stocks-ml-entry-gate-falsified` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 60 | `stocks-ml-feature-gate-deploy` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 61 | `stocks-momentum-french-screen` | not computable from artifact | A1 1927-01..2026-04; A2 1994-01..2026-04 (windows from brief docs/research/briefs/2026-10-20-momentum-retest.md §5; not stored in artifact) | not in artifact | NW(12), monthly (t only) | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: annualized mean excess and n_months (groups.{g} holds full_t, full_p, post_cagr_diff_pp; spec appendix). Observed t: big 3.05 (= 0.79 of the t>=3 MDE); small 3.11 (= 0.81 of the t>=3 MDE) |
| 62 | `stocks-mtf-reversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 63 | `stocks-nasdaq100-single-core` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 64 | `stocks-net-issuance-screen` | not computable from artifact | A1 1963-07..2026-04; A2 1994-01..2026-04 | A1_n_months 754, A2_n_months 388 | NW(12), monthly (t only) | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: annualized mean excess (H1.A1 holds t only). Observed t: H1 (BIG NegNI) 0.26 (= 0.07 of the t>=3 MDE) |
| 65 | `stocks-ohlcv-day-shift` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 66 | `stocks-option-iv-skew` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 67 | `stocks-overnight-gap` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 68 | `stocks-pairs-stat-arb` | no raw artifact | 2015-01-01 through present (42 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 69 | `stocks-paper-cron-1245` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 70 | `stocks-paper-freeze-readiness` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 71 | `stocks-paper-sell-fill-wait` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 72 | `stocks-pead` | no raw artifact | 2021-01-31..2026-04-30 (matched, 17 folds); long window 2015-06–2026-07 (40 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 73 | `stocks-position-sizing-regimes` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 74 | `stocks-production-replica` | computed | 2021-01-31..2026-04-30 | n_obs_daily 1317 | NW(5), daily | 1.10 | 4.24 | 3.14 | 1.62 | observed t -0.69, ann. excess -0.76 pp/yr, TE 2.61% |
| 75 | `stocks-production-replica-holdout-2010-2020` | computed | 2010-01-31..2020-12-31 | n_obs_daily 2750 | NW(5), daily | 1.10 | 4.22 | 3.12 | 1.11 | observed t 0.69, ann. excess +0.76 pp/yr, TE 3.80%; holdout not citable (tainted trades) |
| 76 | `stocks-production-replica-live-parity` | not computable from artifact | 2026-09-29..2026-10-19 (parity since/through) | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (live-parity record, not a return test). |
| 77 | `stocks-research-process-audit` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (artifact is a readiness, inventory, provenance or audit record, not an excess-return test). |
| 78 | `stocks-retail-attention` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 79 | `stocks-rsi-reversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 80 | `stocks-short-interest` | no raw artifact | 2020-07-01 through present (20 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 81 | `stocks-short-volume-ratio` | no raw artifact | 2018-09-01 → end not stated (27 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 82 | `stocks-six-indicator-ensemble` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 83 | `stocks-sp500-core-anchor-fix-reproduction` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 84 | `stocks-sp500-core-corrected-tape-rebaseline` | not computable from artifact | 2021-01-31..2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 85 | `stocks-sp500-core-pit-rebaseline` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: SPY CAGR and t-statistic (only {s}.spy_sharpe; spec appendix). |
| 86 | `stocks-sp500-equity-research-doc-trials` | no raw artifact | 2016–2026 (as stated in archive review) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 87 | `stocks-split-dividend-corrections` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 88 | `stocks-spy-benchmark-pinned-window` | no raw artifact | 2021-01-31..2026-04-30 (pinned 17-fold window) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 89 | `stocks-spy-duplicate-rows` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 90 | `stocks-stability-aware-live-params` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 91 | `stocks-three-pct-position-size` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 92 | `stocks-three-sleeve-blend-1-14` | no raw artifact | 2021-01-31..2026-04-30 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 93 | `stocks-three-sleeve-blend-reproduction` | no raw artifact | 2021-01-31 → 2026-04-30 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 94 | `stocks-trailing-stops` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 95 | `stocks-turn-of-month-rotation` | no raw artifact | Pinned 2021-02 → 2026-04; Long 2011-03 → 2026-04 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 96 | `stocks-two-symbol-sweep-design` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 97 | `stocks-vix-entry-throttling` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 98 | `stocks-vol-managed-spy` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 99 | `stocks-volume-bb-reversion` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 100 | `stocks-voter-ablation-5-voter` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 101 | `stocks-wfo-anchor-leakage-fix` | no raw artifact | 2021-01-31 → 2026-08-18 (run described as 2021-01-31 → present) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 102 | `stocks-wfo-fallback-semantics` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 103 | `stocks-wfo-fold-attribution` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (fold attribution record, not a return test). |
| 104 | `stocks-wfo-policy-synthetic` | not computable from artifact | not recorded | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (synthetic gate-policy simulation; its power-type output is cells.{mu}|{p_gate}|{policy}.ever_halted, not an MDE). |
| 105 | `stocks-wfo-research-run-2026-05-02` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 106 | `stocks-wfo-tournament-sp500-monthly` | no raw artifact | 2021-02 → 2026-05 (64 OOS months) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 107 | `stocks-xs-momentum-2026-06` | no raw artifact | 2021-02 → 2026-05 (64 OOS months) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 108 | `stocks-xs-momentum-corrected-replay` | computed | 2021-01-31..2026-04-30 | n_obs_daily 1064, n_folds 17 | NW(5), daily OOS | 5.90 | 22.66 | 16.76 | 1.82 | observed t 0.76, ann. excess +4.49 pp/yr, TE 12.44% |
| 109 | `cross-asset-commodity-trend` | no raw artifact | 2013-01-01 → end not stated (50 folds) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 110 | `cross-asset-etf-implied-fx-expectations` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 111 | `cross-asset-fomc-drift-corrected-harness` | computed | 2012-01-01..2026-04-30 | n_event_dates 113, n_nonevent 3487 | Welch t, day-clustered, per event (net) | 7.83 bp per event | 30.1 bp/event; derived 2.37 pp/yr | 22.3 bp/event; derived 1.75 pp/yr | not computable (no tracking error) | annualized = per-event x 7.886 events/yr (labelled derivation); observed net t 0.98 |
| 112 | `cross-asset-fomc-drift-wfo` | no raw artifact | eval_start 2012-01-01 (end not stated in this report) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 113 | `cross-asset-fx-hedge-overlay` | no raw artifact | Standalone 2016-01 → 2026-07 (38 folds); OOS 2017-01-02 → 2026-06-30 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 114 | `cross-asset-month-end-treasury` | not computable from artifact | 2021-01-31 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (primary.* carry CAGR/Sharpe/vol/hit-rate only). |
| 115 | `cross-asset-on-deck-candidate-feasibility` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 116 | `cross-asset-sleeve-holdout-2007-2011` | not computable from artifact | 2007-06-01 → 2011-02-28 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic. |
| 117 | `cross-asset-sleeve-trend-arm` | not computable from artifact | pinned 2021-02-01 → 2026-04-30; 2016-01-04 → 2026-04-30 | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (primary:{w}.{cost}.{arm} carry CAGR/Sharpe/yearly returns only). |
| 118 | `cross-asset-static-80-20-sleeve` | not computable from artifact | 2021-02-01 → 2026-04-30 (report table heading; raw JSON meta says 2021-01-31 → 2026-04-30) | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (primary:{w}.{cost}.{arm} carry CAGR/Sharpe/yearly returns only). |
| 119 | `cross-asset-treasury-curve` | no raw artifact | 2014-01-01 → 2026-06-30 | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 120 | `crypto-binance-us-live-trader` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 121 | `crypto-config-tweaks-reverted-2026-04-02` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 122 | `crypto-defi-tokenized-equity-arb` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 123 | `crypto-diversity-cap-disabled-2026-05-07` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 124 | `crypto-edge-search-2026-06-08` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 125 | `crypto-execution-venue-fees` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 126 | `crypto-funding-carry-phase4` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 127 | `crypto-ibit-overnight-session` | computed (sub-rows) | 2024-01-11..2026-10-23 | n_days 699, n_weeknight 543, n_weekend 155 | iid t per session (own statistic) | see note | see note | see note | not computable (no tracking error) | intraday: SE 7.5 bp, MDE t>=3 28.9 bp/session (derived 73 pp/yr), t>=2 21.4 bp (derived 54 pp/yr); weeknight: SE 8.4 bp, MDE t>=3 32.1 bp/session (derived 63 pp/yr), t>=2 23.7 bp (derived 46 pp/yr); weekend: SE 27.5 bp, MDE t>=3 105.8 bp/session (derived 59 pp/yr), t>=2 78.2 bp (derived 44 pp/yr). Absolute session returns, no benchmark. |
| 128 | `crypto-kraken-live-bot-pnl` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 129 | `crypto-live-ohlcv-tz-bug` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 130 | `crypto-perp-funding-carry-candidate` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 131 | `crypto-pipeline-runs-2026-03` | no raw artifact | symbols-file names 2023-01-01 → 2025-12-31; recent-validation windows vary (e.g. 2025-06-01 → 2025-12-30) | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 132 | `crypto-regime-filters` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 133 | `crypto-spot-etf-flow-persistence` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 134 | `crypto-strategy-usage-analyzer-2026-05-07` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 135 | `crypto-trade-frequency-gate-2026-05-07` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 136 | `crypto-tsmom-btc` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 137 | `crypto-universal-trailing-stops` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 138 | `crypto-venue-migration-path-d` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 139 | `crypto-web-research-batch-2026-10-25` | computed (sub-rows) | 2024-01-11..2026-10-23 | n_days 699, n_weeknight 543, n_weekend 155 | iid t per session (own statistic) | see note | see note | see note | not computable (no tracking error) | intraday: SE 7.5 bp, MDE t>=3 28.9 bp/session (derived 73 pp/yr), t>=2 21.4 bp (derived 54 pp/yr); weeknight: SE 8.4 bp, MDE t>=3 32.1 bp/session (derived 63 pp/yr), t>=2 23.7 bp (derived 46 pp/yr); weekend: SE 27.5 bp, MDE t>=3 105.8 bp/session (derived 59 pp/yr), t>=2 78.2 bp (derived 44 pp/yr). Absolute session returns, no benchmark. |
| 140 | `crypto-wfo-overfitting-param-stability` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 141 | `crypto-wfo-recalibration-2026-05-01` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 142 | `crypto-wfo-research-snapshots-kraken` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |
| 143 | `crypto-wfo-textbook-reset` | not computable from artifact | --days 1095 --end-date 2026-05-01 (start derived) | not in artifact | none reported | not computable from artifact | not computable from artifact | not computable from artifact | not computable from artifact | missing: t-statistic (per-symbol WFO fold stats; no excess-vs-benchmark statistic). |
| 144 | `crypto-wfo-tournament-robustness` | no raw artifact | not recorded | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact |

Totals: **7 computed** (5 in pp/yr, 2 in per-session sub-rows), **34 not computable from
artifact**, **103 no raw artifact**.

Row notes:
- Replica rows: `windows.{w}.arms.primary.excess_vs_spy.{ann_excess_pct,t_stat,n_obs,tracking_error_pct}`;
  NW lag count from `NW_LAGS` (`src/ggTrader/lab/significance.py:14`). The holdout row is
  computed for power only. Its result is not citable: `verdict.reasons =
  ['holdout:tainted_trade:primary']` and `closure.holdout = 'INCONCLUSIVE'`.
- xs_momentum and dual_momentum: `{key}.excess_stats.*` in
  `docs/research/_audit_reruns_20261019.json`. The 1,064 daily returns are the OOS fold span
  (first test fold 2022-01-31). Lens 7 found the dual_momentum block identical to the
  xs_momentum block (`docs/research/_lens7_recompute.json`).
- FOMC: `net.mean_diff` / `net.t_stat` (Welch t on day-averaged returns,
  `src/ggTrader/lab/event_study.py:84-99`; clustering in
  `scripts/fomc_clustered_event_study.py:93-104`). The bar is net event t >= 3.0 and WFE >= 0.50
  (`docs/research/briefs/2026-10-19-production-replica-audit.md:149-152`). The observed net
  effect is 7.68 bp per event.
- IBIT: t = mean / sd x sqrt(n), independent returns
  (`docs/research/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.py:45-46`). There
  is no pre-registered bar. The verdict names "overnight", but `overnight_all` has no t, so
  one sub-row is reported per session type (spec 2.1, step 3).
- Momentum bar: Holm across two hypotheses; the stronger needs about t >= 3.2
  (`docs/research/briefs/2026-10-20-momentum-retest.md:112-115`). Observed t as a fraction
  of the MDE at that bar: big 0.76, small 0.77.

## Ceiling comparison

Statement, not recommendation. Target per spec section 1: strategy raw TR = 2 x SPY raw TR
over the same window; target CAGR = (1 + 2R)^(1/years) - 1, with R = SPY TR.

**Typical window, 2021-01-31..2026-04-30** (SPY closes 2021-01-29 to 2026-04-30, 5.243
years): SPY TR R = +108.60%. Target TR = 2 x 108.60 = +217.21%. Target CAGR =
3.1721^(1/5.243) - 1 = 24.63%. SPY CAGR = 2.0860^(1/5.243) - 1 = 15.05%. **Required gap =
9.57 pp/yr.**

| Window (rows) | Benchmark TR | Benchmark CAGR | Target CAGR | Required gap pp/yr |
|---|---|---|---|---|
| 2021-01-31..2026-04-30 (replica, PIT core, blends, month-end) | SPY +108.60% | 15.05% | 24.63% | 9.57 |
| 2022-01-31..2026-04-30 (xs_momentum OOS span) | SPY +69.33% | 13.21% | 22.75% | 9.54 |
| 2010-01-31..2020-12-31 (replica holdout) | SPY +333.17% | 14.37% | 20.51% | 6.14 |
| 2012-01-01..2026-04-30 (FOMC) | SPY +635.17% | 14.94% | 20.05% | 5.11 |
| 2006-06-21..2026-09-29 (A23, A24) | SPY +785.02% | 11.35% | 14.90% | 3.54 |
| 2024-01-11..2026-10-23 (IBIT; SPY shown for scale only) | SPY +64.76% | 19.66% | 34.81% | 15.15 |
| 1994-01..2026-04 (factor/momentum A2) | French market | 10.87% | 13.21% | 2.34 |
| 1963-07..2026-04 (factor, net-issuance A1) | French market | 10.83% | 12.06% | 1.23 |
| 1927-01..2026-04 (momentum A1) | French market | 10.29% | 11.06% | 0.77 |

Source: `regime.summary` and `regime.targets_2x_spy_total_return` in
`docs/research/_lens9_power.json`.

| Test | MDE t>=3 / t>=2 (pp/yr) | Required gap | Could detect target-sized effect? | Measured effect |
|---|---|---|---|---|
| Production replica, primary | 4.24 / 3.14 | 9.57 | yes, at both bars | -0.76 pp/yr arithmetic, CAGR gap -0.64 |
| Replica holdout (not citable) | 4.22 / 3.12 | 6.14 | yes, at both bars | +0.76 pp/yr arithmetic, CAGR gap +1.00 |
| xs_momentum OOS (dual_momentum identical) | 22.66 / 16.76 | 9.54 | **no**, at either bar | +4.49 pp/yr arithmetic, CAGR gap +4.37 |
| FOMC net, day-clustered (derived annualization) | 2.37 / 1.75 | 5.11 | yes, if the effect is read as a sleeve's event-day excess | +7.68 bp/event, about +0.61 pp/yr derived |
| IBIT sessions (no benchmark) | 59-73 / 44-54 derived | 15.15 (SPY, scale only) | **no** | weeknight +13.8 bp/session, t 1.65 |
| Factor H1-H4, momentum, net issuance | not computable from artifact | 0.77-2.34 | unknown in pp/yr | A2 CAGR diffs -0.62 to +0.83 (factor), +0.08 / +0.20 (momentum), +0.48 (net issuance) |
| PIT core, ensemble_ic/kelly, core plus SPY blend, cross-asset sleeve, month-end, A23/A24, century leverage | not computable from artifact (no t) | 3.54-9.57 | unknown | see Lens 2 sensitivity table |

What the table says:
- **Could not detect a target-sized effect:** xs_momentum and dual_momentum (MDE 2.4 times
  the required gap at t >= 3, 1.8 times at t >= 2) and the IBIT split. For every
  2021-2026 WFO artifact without a t (PIT core, ensemble_ic, ensemble_kelly, core plus SPY
  blend, month-end Treasury, cross-asset sleeve) the answer is unknown from the artifact.
- **Could detect one:** the replica (both windows) and FOMC, because the replica tracks SPY
  closely (85% SPY, 9.4% stocks; `windows.primary.arms.primary.mean_exposure`) and FOMC
  has 113 independent events. The replica's low TE is also why a target-sized effect is
  implausible there: the 9.4% stock sleeve would need about +102 pp/yr of its own.
- **Measured effects against MDEs:** every measured effect with a computable MDE is below
  its t >= 2 MDE. The largest, xs_momentum's +4.49 pp/yr, is 0.27 of its t >= 2 MDE
  (16.76). The long-history windows need the smallest target gaps (0.77 to 2.34 pp/yr),
  and their measured post-1994 effects (-0.62 to +0.83 pp/yr) are below those gaps by
  arithmetic alone, before any test.

## Years to detect

Independent-returns approximation, kept separate from the per-row MDEs because it uses TE
as sigma: years = ((bar + 0.8416) x sigma / X)^2. Full grid in
`years_to_detect_iid_scenarios` in `docs/research/_lens9_power.json`.

| sigma (source) | X = 1 pp/yr | X = 2 | X = 5 | X = 10 |
|---|---|---|---|---|
| 2.61%, replica TE (t>=2 / t>=3) | 55 / 101 | 14 / 25 | 2.2 / 4.0 | 0.6 / 1.0 |
| 12.44%, xs_momentum TE | 1,250 / 2,285 | 313 / 571 | 50 / 91 | 12.5 / 23 |
| 17.14%, live sleeve 06-24..11-04 (Lens 3) | 2,373 / 4,337 | 593 / 1,084 | 95 / 174 | 24 / 43 |
| 22.44%, live sleeve frozen 09-30..11-04 (Lens 3) | 4,067 / 7,433 | 1,017 / 1,858 | 163 / 297 | 41 / 74 |
| 2.59%, account level frozen (Lens 3) | 54 / 99 | 14 / 25 | 2.2 / 4.0 | 0.5 / 1.0 |

**Cross-check against Lens 3** (`detectable_effect.table` in
`docs/research/_lens3_snapshot_gap.json`): Lens 3 uses z = 1.96 + 0.8416 = 2.8016. For the
frozen sleeve at X = 10 pp/yr: 252 x (2.8016 x 0.22442 / 0.10)^2 = 9,962 days = 39.5
years. Lens 9's grid recomputes 39.5 years with the same z (`years_lens3_z2.8016`).
Lens 3's figure reproduces. With the spec's t >= 2 bar (2 + 0.8416 = 2.8416) it is 40.7
years.

## Regime dependence

**What 2021-01-31..2026-04-30 contains** (SPY TR, yfinance; `regime.spy_yfinance` in the JSON):

| Year | SPY TR |
|---|---|
| 2021 (from 01-29) | +30.05% |
| 2022 | -18.18% |
| 2023 | +26.18% |
| 2024 | +24.89% |
| 2025 | +17.72% |
| 2026 (to 04-30) | +5.68% |

- Negative full calendar years: 1 of 4. Negative quarters: 6 of 22 (2022 Q1-Q3, 2023 Q3,
  2025 Q1, 2026 Q1; the first and last quarters are partial). Negative months: 21 of 64.
- Max drawdown: **-24.50%**, peak 2022-01-03, trough 2022-10-12, recovered 2023-12-13.
  Next largest: -18.76%, 2025-02-19 to 2025-04-08, recovered 2025-06-26. That is **one
  bear episode** of 20% or more, plus one near-bear. Annualized SPY volatility: 16.95%.
- Rate regime (13-week T-bill, `^IRX` annual means): 0.03% (2021), 2.00% (2022), 5.05%
  (2023), 4.95% (2024), 4.05% (2025), 3.70% (2026 through 11-03). One full tightening
  cycle from zero, then a partial easing.
- Dispersion proxy (spread between the best and worst of nine SPDR sector ETFs per calendar
  year; a sector-level proxy, not stock-level dispersion): 36 pp (2021), **101 pp (2022, the
  highest in 2007-2026)**, 63 pp (2023), 30 pp (2024), **23 pp (2025, the lowest in
  2007-2026)**. The 2007-2019 range was 24-56 pp. The window holds both extremes.
- Return level: SPY CAGR 15.05% (French market 13.52% over 2021-02..2026-04), against a
  French market CAGR of 10.29% over 1927-2026 and 10.87% over 1994-2026.

**Long-history comparison** (French market, monthly, peak-to-recovery episodes with a
drawdown of 20% or more; declines nested inside a longer episode, such as 1937-38 inside
1929-45, are not counted separately):

| Window | Bear episodes (troughs) | Negative calendar years | Market CAGR |
|---|---|---|---|
| 1927-01..2026-04 | 10 (1932, 1947, 1962, 1970, 1974, 1987, 2002, 2009, 2020, 2022) | 24 of 99 (1927-2025) | 10.29% |
| 1963-07..2026-04 | 7 | not computed | 10.83% |
| 1994-01..2026-04 | 4 (2002, 2009, 2020, 2022) | 7 of 32 (1994-2025) | 10.87% |
| 2021-02..2026-04 | 1 (2022) | 1 of 4 full years | 13.52% |

Quarterly: 125 of 396 French quarters (1927-2025) were negative (32%), against 6 of 22 in
the primary window (27%). The 2021-2026 window is not unusually free of down years. It has
one bear episode, an above-average return level, and a rate shock.

**Which verdict rows rest only on 2020-2026 data.** Classification rule (in the JSON): the
earliest year in the Lens 1 `eval_window` and in the row's artifact window. 37 rows fall in
2020-2026 only, and 78 rows have no recorded window. The verdict-bearing ones:
`stocks-production-replica` (citable primary only; the holdout is not citable),
`stocks-sp500-core-pit-rebaseline`, `stocks-ensemble-ic-pit`, `stocks-ensemble-kelly-pit`,
`stocks-xs-momentum-corrected-replay`, `stocks-xs-momentum-2026-06`, `stocks-dual-momentum`,
`stocks-core-spy-blend-keep`, `stocks-live-revert-blend-to-core`,
`stocks-sp500-core-corrected-tape-rebaseline`, `stocks-sp500-core-anchor-fix-reproduction`,
`stocks-three-sleeve-blend-1-14`, `stocks-ensemble-wfo-validated-2026-06-20`,
`stocks-equity-diversification-sleeves`, `stocks-wfo-tournament-sp500-monthly`,
`stocks-position-sizing-regimes`, `stocks-short-interest` (from 2020-07),
`stocks-headline-sentiment`, `stocks-ema-cross`, `cross-asset-month-end-treasury`,
`cross-asset-static-80-20-sleeve` (verdict window 2021-2026; the artifact also holds a
2016-2026 window, and a 2007-2011 holdout row exists), `crypto-ibit-overnight-session` and
`crypto-wfo-textbook-reset`. The full list is in `regime.summary.rows_resting_only_on_2020_2026`.
Rows with longer windows: A23/A24 (2006-2026), century leverage (1926-2026), the factor,
momentum and net-issuance screens, FOMC (2012-2026), and the cross-asset trend arm
(2016-2026).

**Do the long-history screens depend on 2021-2026?** No. Their weakness spans all recent
subperiods. Momentum big: t 3.56 (1927-1993), 0.54 (1994-2012), -0.56 (2013-2026); small:
3.38, 1.11, -0.77 (`groups.{g}.diag` in
`docs/research/_momentum_french_screen_results.json`). Factor H2 and H4 t: 2.00 / 3.04 in
1963-1993, then about 0 in 1994-2012 and -0.56 to 0.04 in 2013-2026
(`hypotheses.{g}.diagnostics.subperiods`). Net issuance: 0.12, 0.90 (1994-2008), -1.24
(2009-2026). The decline is post-1994 and post-2008, not specific to the 2021-2026 bull run.

## Questions and answers

1. **With the data actually available, what size of edge could each test have detected?**
   Answered for the 7 computable rows (table above). Low-TE replica: 4.2 pp/yr at t >= 3
   (3.1 at t >= 2) on both windows. Momentum-like TE: 22.7 pp/yr (16.8). FOMC: 30 bp per
   event (about 2.4 pp/yr, derived). IBIT sessions: 29-106 bp per session. For 34 rows it is
   **unknown** from the artifact. The reason per row is in the table: mostly a missing
   t-statistic, or, for the French screens, a missing mean excess. **Cheapest test:** for
   the French screens, recompute the mean monthly net excess from the frozen French archives
   already on disk (pure arithmetic, minutes, no backtest). For the 2021-2026 WFO artifacts,
   apply `excess_stats` (`src/ggTrader/lab/significance.py`) to the OOS equity curves. These
   are not saved, so this needs a re-run, which is a Phase 2 item needing owner approval
   (about 10-40 min per strategy, going by the `elapsed_sec` fields in the PIT and
   audit-rerun artifacts).
   For 103 rows there is no raw artifact. They stay unknown unless re-run.
2. **How much do conclusions depend on 2021-2026 being a single (largely bull) regime?**
   Partly answered. 37 inventory rows, including most of the stock-selection and WFO
   verdicts, rest only on 2020-2026 data. That window holds one bear episode, an
   above-average SPY return (15.05% CAGR) and a rate shock from 0% to 5%. For these rows,
   a different regime could give a different result, and nothing in their artifacts tests
   that. The one cross-decade check, the replica holdout 2010-2020, ran but is not citable.
   The long-history screens do not depend on this window: their post-1994 subperiods are
   flat or negative. **Unknown:** whether the 2021-2026 stock and WFO verdicts would hold in
   another regime. **Cheapest test:** the clean 2010-2020 holdout for the replica (already
   on the Phase 2 list in spec section 6, hours). A cheaper partial check: the same rows'
   per-fold returns split by 2022 versus the other years, where `fold_rows` exist (minutes,
   read-only; folds are not independent).
3. **Phase 0 power claims.** "t >= 3 needs IR near 1.3": confirmed at 50% power (1.31). At
   80% power it is 1.62 to 1.68, so the claim understates the requirement. "Unreachable on
   five years of data": confirmed for effects of the size the project measured on long
   histories, refuted for target-sized effects in a low-TE book (replica MDE 4.24 pp/yr
   against a 9.57 pp/yr target). The falsifier set before the evidence ("MDE at or below
   about 2 pp/yr") was not met by any computed 2021-2026 row.

## Verification

Commands (scratch scripts are in the session scratchpad under `l9/`. All Python ran through
`scripts/run_capped.sh .venv/bin/python -I`, with the 6 GB cap and no swap):

- `l9/keys.py` and inline `python3 -c` key dumps of all 33 distinct artifacts. Only
  `_production_replica_results.json`, `_audit_reruns_20261019.json`,
  `_fomc_clustered_event_study_20261020.json` and `ibit_overnight_split.json` hold both a
  mean and a t. The factor, momentum and net-issuance artifacts hold t only. The
  remaining 26 artifacts hold no t.
- `l9/regime.py`: yfinance download, `auto_adjust=True`, of SPY, `^IRX` and nine sector
  ETFs, 2005-01-03 to 2026-11-03. Output: SPY TR 2021-01-29..2026-04-30 = +108.603%. Its
  French section was superseded (it divided the parser's decimal returns by 100 again) and
  those fields are dropped from the JSON.
- `l9/french.py`: frozen `F-F_Research_Data_Factors_CSV.zip`, 1,202 months 1926-07..2026-08,
  read only. Output: 10 bear episodes, 24 of 99 negative years.
- `l9/build.py`: writes `docs/research/_lens9_power.json`. Asserts 144 rows and 144
  unique ids. Status counts: computed 7, not computable 34, no raw artifact 103.
- Hand check of one MDE: replica SE = -0.76198 / -0.68983 = 1.1046. MDE = 3.8416 x 1.1046 =
  4.243 pp/yr. The independent-returns SE = 2.6146 / sqrt(1317/252) = 1.144, so NW and
  independent agree within 4%.
- Lens 3 cross-check reproduced: 39.5 years.
- Gate: `.venv/bin/python scripts/check_lens_report.py docs/research/2026-11-04-lens9-power-and-regime.md`.
- No DB query, no Alpaca call, no repo code edits, nothing under `data/` written.

**Unverified:**
- Whether the TE behind each MDE would stay the same for a strategy with a larger true
  edge. MDEs are conditional on the realized noise of the book that was tested.
- The arithmetic-versus-CAGR conversion is approximate. For high-volatility books the two
  can differ materially.
- The dispersion figure is a sector-ETF proxy, not stock-level cross-sectional dispersion.
- The classification of 2020-2026-only rows uses Lens 1's `eval_window` strings, which are
  missing for 78 rows. Rows with "end not stated" were classified by start year.
- The yfinance SPY series was not compared day by day with a second source in this lens.
  Lens 3 found yfinance and Alpaca IEX within 0.05 pp per era.

## Done checklist

- ✅ Falsifiers written before evidence, for both Phase 0 power claims and the regime question
- ✅ Steelman, with whether it holds up
- ✅ MDE table covering all 144 inventory ids (7 computed, 34 not computable with the missing key, 103 no raw artifact)
- ✅ Ceiling comparison with target arithmetic from yfinance SPY TR
- ✅ Years-to-detect grid; one Lens 3 number cross-checked (39.5 years)
- ✅ Regime description of 2021-2026, long-history comparison, list of rows resting only on 2020-2026
- ✅ Spec question 1 (detectable edge per test): answered for 7 rows
- ❓ Spec question 1 for 34 artifact rows: unknown (missing t or mean excess; cheapest test given)
- ❓ Spec question 2 (regime dependence) for the 2021-2026 stock and WFO verdicts: unknown whether they hold in another regime (the only cross-decade run is not citable; cheapest test given)
- ✅ Spec question 2 for the long-history screens: not dependent on 2021-2026 (subperiod evidence)
