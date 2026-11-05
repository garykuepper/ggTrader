# Production-rule daily-bar audit: NO-GO; historical live replication unverified

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-10-19
**Audience:** Flynn and research collaborators
**Brief:** [original pre-registration](briefs/2026-10-19-production-replica-audit.md),
with approved [daily-bar scope](briefs/2026-10-19-production-replica-parity-amendment.md),
[event-timing correction](briefs/2026-10-19-production-replica-fomc-timing-amendment.md),
and [primary-failure closure](briefs/2026-10-19-production-replica-closure-amendment.md).

## 1. Executive Summary & Core Engine Audit

**NO-GO for the specified historical daily-bar model.** On the primary window
the portfolio earned **14.19% CAGR versus SPY's 14.83%**, an excess of **−0.64
percentage points/year**, with Newey-West **t = −0.69**. The registered bars are
t ≥ 3.0 and excess CAGR ≥ +1.0 point/year; both fail. The raw driver remains
`INCONCLUSIVE/stopped` because its holdout triggered a data stop. Flynn approved
closing the model negatively on its primary failure while preserving that stop.
Source: [`_production_replica_results.json`](_production_replica_results.json),
`closure`, `verdict`, and `windows.primary.arms.primary.{stats,spy_stats,excess_vs_spy}`.

**Exact replication of historical deployed decisions remains UNVERIFIED.** The
parity check matched **10/32 orders (31.25%)**. Its unchanged denominator includes
**19 manual liquidations**; another sale reproduces with the former indicator
lookback. ALL and ARES lack recoverable decision-time yfinance inputs. Plausible
partial-bar substitutions are diagnostics, not successful parity. The approved
daily-bar study therefore answers a narrower model question and cannot measure
the edge of the unreconstructed live path. Sources:
[`_replica_parity_20261019.json`](_replica_parity_20261019.json),
`explained_orders`, `strategy_orders`, `explained_pct`, `notes`, `investigation`;
the [parity amendment](briefs/2026-10-19-production-replica-parity-amendment.md).

The replica uses fixed default ensemble votes, prior-completed-bar signals,
portfolio-value sizing, a position cap, catch-up exits, catastrophe stops and the
SPY cash sweep. It calls only the permitted pure production decision functions.
It is a historical fixed-rule replay, not a fresh forward validation or a
walk-forward parameter selection. The older KEEP construction used selected
parameters and same-close fills; this report supersedes that decision for this
model. Momentum and FOMC were rerun separately on the corrected harness and
remain NO-GO. After-tax replay of this stock model was not triggered.

## 2. Quantitative Performance Context

### Primary: 2021-01-31 → 2026-04-30

The engine retains a preceding lag/valuation bar. Stock fills cost 5 bp/side,
with 3 and 10 bp sensitivities; sweep fills cost 1 bp/side. Cash accrues cached
TB3MS. SPY buy-and-hold is the normalized adjusted-close curve; isolated benchmark
entry/liquidation fees are not separately deducted. Constants and windows are
fixed by the [brief](briefs/2026-10-19-production-replica-audit.md) and driver.

| Arm | CAGR | Sharpe | Max drawdown | Excess CAGR over SPY, pp/year | NW t |
|---|---:|---:|---:|---:|---:|
| **Primary: lag 1, 5 bp, T-bills** | **14.19%** | **0.925** | **−21.44%** | **−0.642** | **−0.690** |
| Lag 0 diagnostic | 14.41% | 0.939 | −21.46% | −0.416 | −0.565 |
| Stock cost 3 bp | 14.26% | 0.929 | −21.41% | −0.566 | −0.629 |
| Stock cost 10 bp | 14.00% | 0.914 | −21.54% | −0.832 | −0.842 |
| Cash at zero | 14.00% | 0.914 | −21.48% | −0.833 | −0.841 |
| **SPY buy-and-hold** | **14.83%** | **0.903** | **−24.50%** | — | — |

All table numbers: `_production_replica_results.json`,
`windows.primary.arms.<primary|lag0|cost3|cost10|cash0>.stats`,
`.excess_vs_spy`, and `windows.primary.arms.primary.spy_stats`. Sharpe is the
lab's zero-risk-free-rate statistic, not evidence of significant stock alpha.
The registered test uses daily excess returns over SPY, with the prescribed
Newey-West lags. No confidence interval or unregistered resampling rescue was
used. Curves: [`_production_replica_curves.csv`](_production_replica_curves.csv).

The lag-0 minus lag-1 CAGR difference is **0.227 point/year**, calculated from
`windows.primary.arms.{lag0,primary}.stats.cagr_pct`. Even same-close fills fail.
Against the exposure-matched passive, the primary adds only **0.265 point/year**
with **t = 0.202** (`primary.excess_vs_exposure_matched`). Selection skill is not
established. Mean exposures were **9.37% stocks, 85.33% SPY, 5.30% cash**
(`primary.mean_exposure`). Its lower drawdown and slightly higher raw Sharpe do
not satisfy the return and significance bars.

The primary recorded **83 halted days**, **998 days with more candidates than
slots**, and **854 fills** (`primary.halted_days`, `.truncated_days`,
`.trade_count`). Truncation counts days, not rejected orders. The negative
information ratio supplies no finite positive-effect time-to-significance
estimate: `primary.years_for_t` is JSON null, representing infinity.

### Holdout: 2010-01-31 → 2020-12-31 — INCONCLUSIVE

Coverage was **87.39% of member-days**, above the registered **70%** floor;
primary coverage was **98.60%**. Source: `windows.<holdout|primary>.mean_coverage`
and `.coverage_member_days`. Yearly coverage is retained under `.coverage`.
Coverage measures available prices, not correct security identity. Missing
delisted names still create survivorship risk; a pass would be weak evidence.

The holdout primary arm produced **21 trade/artifact matches**, representing
**11 distinct (symbol, trade-date) pairs** across TIE, MEE and RHT. Source:
`windows.holdout.arms.primary.tainted_trades`; distinct count is the set of its
`(symbol,date)` fields. The raw stop is retained under `windows.holdout.stop_flags`.
Only this arm ran; the other sensitivities were not run. No holdout performance
figure is offered as investment evidence. Its **67 halted days** and **2,021
truncated days** are operational diagnostics from this tainted run only
(`windows.holdout.arms.primary.{halted_days,truncated_days}`).

The read-only review found incompatible price levels in TIE/MEE and suspicious
stale, low-volume CPWR prices. Mixed security identities are an inference, not a
verified mapping. CPWR illustrates a limitation: fresh Yahoo can agree with the
same wrong identity. RHT may be a genuine acquisition jump: IBM announced its
Red Hat agreement before the 2018-10-29 artifact bar (the trade was 2018-10-30).
Those dates are recorded in `.tainted_trades`; the announcement is
independently documented in [IBM's release](https://uk.newsroom.ibm.com/2018-10-28-IBM-To-Acquire-Red-Hat-Completely-Changing-The-Cloud-Landscape-And-Becoming-Worlds-1-Hybrid-Cloud-Provider);
it does not independently validate each cached price. Review findings and their
status are recorded in the [closure amendment](briefs/2026-10-19-production-replica-closure-amendment.md).

The registered detector treats missing fresh comparisons as artifacts, checks
only trade proximity, shares Yahoo as a comparison source, and misses changes
across missing-price gaps. It does not certify positions held through bad marks.
No names or bars were removed, no prices repaired, and no holdout stop cleared.

### Corrected-harness secondary studies

| Study | Excess CAGR vs SPY, pp/year | Statistical test | Other registered bar | Verdict |
|---|---:|---:|---:|---|
| xs_momentum | +4.367 | NW t 0.761 | t ≥ 3 and excess ≥ +1 | NO-GO stands |
| dual_momentum | +4.367 | NW t 0.761 | t ≥ 3 and excess ≥ +1 | NO-GO stands |
| FOMC primary pair | Not the decision metric | Net Welch t 1.572 | WFE 0.244 vs ≥ 0.50 | NO-GO stands |

Source: [`_audit_reruns_20261019.json`](_audit_reruns_20261019.json),
`<xs_momentum|dual_momentum>.excess_stats`, `.verdict`, `.fold_rows`,
`fomc_drift.primary_event_study.net.t_stat`, `.wfo.aggregate_wfe`, `.verdict`.
Momentum uses the pinned primary evaluation window through `run_core`; its
stitched OOS curve starts after the first training year. Both runs used anchors
in **17/17 folds**, with **0/17 corrected gate passes** (`fold_rows.used_anchor`,
`n_folds`, `n_gate_pass`, `dsr_gate_passes`). Their identical aggregate outcomes
are not independent confirmation of an edge. `run_core` retains its existing
CLI-equivalent warm-up calculation; the replica and FOMC driver use the plan's
fixed warm-up. That difference is not an optimized parameter.

FOMC uses its registered event window and primary parameter pair. The unchanged
semantics regression now excludes the return before the entry close. Gross
Welch **t = 3.619** falls to **1.572** after the registered round-trip cost;
other pairs are context only (`fomc_drift.primary_event_study.{gross,net}.t_stat`,
`context_pairs`). Prior FOMC event statistics are superseded, not repaired by
selecting a more attractive pair.

### A23 distribution-tax information

The default main-window run exactly reproduced the committed result rows.
At the registered higher-funding scenario, all-LT distribution treatment gives
**11.20%** after-tax CAGR for the static allocation versus SPY **10.40%**,
a **+0.80 point/year** margin. All-ordinary SSO/BIL treatment gives **11.17%**
versus **10.40%**, or **+0.77 point/year**. Both miss A23's original +1.0 bar.
At the upper modeled bracket, the margin changes from **+0.79 to +0.72 point**.
Sources: `_static_leverage_aftertax_rf_plus0.025.json` and
[`_static_leverage_aftertax_dist_bounds.json`](_static_leverage_aftertax_dist_bounds.json),
`windows.main_real_sso['static 1.3x (70/30) band 8pp'][<24/15|40.8/23.8>].after_tax_cagr`
minus `['SPY buy-and-hold'][same bracket].after_tax_cagr`; new artifact `dist_tax`.
These are modeled endpoints, not reconstructed annual tax character. This
static arm holds no BIL, so its change comes from SSO. The higher funding rate
is a registered sensitivity, not a new observation of today's bill rate.
The offset is specified by the plan's Task 9 run command rather than a dedicated
field in the bounds artifact; `calibration.residual_ann_mean` corroborates the
shift. The LT and ordinary artifacts have different download vintages, recorded
by `calibration.overlap`. Calibration changes the pre-inception synthetic series,
outside this real-SSO window. Exact default main-row reproduction and identical
SPY results supply checks against drift in the reported comparison.
The separate A23 owner allocation and exit terms remain unchanged. See its
[dated addendum](2026-09-30-static-fractional-leverage.md).

## 3. Actionable Research Directions

1. **Owner decision at the freeze lift:** retire the stock-picking sleeve for
   lack of validated incremental return under this model. Follow brief §5:
   SPY sweep alone, or the separate shadow only if it passes its own forward
   rule. This is a recommendation; no live change is authorized by this report.
2. **Use the A23 bounds as pre-entry information.** Its allocation decision,
   annual review date and band rules are separate from stock-sleeve evidence.
   Distribution character is a sensitivity assumption, not an individualized
   tax determination or a new entry trigger.
3. **Require data identity and immutable inputs before another stock audit.**
   Research-only security identifiers, independent price checks and preserved
   signal inputs would make a future audit falsifiable. Any production loader
   or trading change requires separate authorization after the freeze. No new
   strategy search or post-hoc holdout repair is proposed here.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**Daily-bar ensemble plus SPY sweep — NO-GO.** Primary return and statistical
bars fail, including sensitivities. Smaller drawdown alone does not pass.

**Historical live parity — UNVERIFIED.** Manual cleanups and legacy behavior
explain part of the mismatch; missing exact inputs prevent full reconstruction.
The parity denominator and failure evidence remain intact.

**Momentum and FOMC corrected-harness reruns — NO-GO stands.** Economic momentum
outperformance without the required statistical evidence is insufficient.
FOMC fails both net-event significance and walk-forward efficiency.

**DSR and event timing — CORRECTED methodology, not alpha.** Commit `09f9f40`
converts annualized Sharpe to per-period units and corrects the kurtosis term.
Past gate counts are void as evidence of DSR validation. Commit `7320c1d`
corrects event-return attribution after the registered test exposed the error.
The original failure log is preserved by `audit_stop` in the replica JSON.

The old [anchor reproduction](2026-08-19-anchor-fix-reproduction.md) receives a
dated correction: the blend window was pinned, so eval-end drift does not
explain its drop. Attribution to the earlier harness fixes remains inferred.
The [KEEP report](2026-09-24-core-spy-blend-keep.md) receives a scope correction;
its selected-parameter, same-close construction is not this fixed-rule replay.

## 5. Operational Roadmap: Recommended First Action

**Leave production frozen and put the stock-sleeve retirement recommendation
before Flynn for the freeze lift.** The owner separately decides any live action.
The approved closure records three distinct conclusions: model NO-GO, holdout
INCONCLUSIVE, exact deployed-trader replication UNVERIFIED.

Alphabetical candidate ordering favors earlier tickers whenever slots are
scarce; this is reproduced rather than improved. While the drawdown/daily-loss
halt is active, all trading stops, including catastrophe stops and the sweep,
except forced delisting sales. Positions keep marking to market, so the halt is
not a drawdown cap. Delisting liquidation uses the last valid mark before the
guard check. Funding sells precede stock buys and sweep buys; the code permits
same-session sweep sell/rebuy churn, but this audit did not measure its frequency.
Cash conservation and
missing-SPY behavior are tested. Point-in-time entry eligibility comes from the
specified monthly as-of plans, with no claim of daily constituent refresh.
Sources: `src/ggTrader/lab/replica.py`, `tests/lab/test_replica.py`, the registered
portfolio rules, and primary operational fields cited above.

Final verification: `scripts/run_tests.sh -q` passed **1,101 tests**, with
**30 deselected** and one existing websockets deprecation warning (local log
`logs/production_replica_final_tests_20261019.log`). Changed and tracked Python
files pass Ruff and formatting checks. Whole-directory `ruff check .` and
`ruff format --check .` still fail only on pre-existing, untracked skill-workspace
artifacts: three long lines in `dollar_carry_sanity.py`, and formatting in that
file and `sell_in_may_check.py`. Those workspaces were neither edited nor staged.
Historical report bodies were verified unchanged beneath their appended dated
corrections. The original brief and frozen live/config paths are unchanged.

## 6. Contrarian Evaluation & Parked Research

The strongest objection is that live may have acted on partial bars and differs
from this model. That objection is valid: this is why historical live replication
remains unverified. Lag 0 is a sensitivity, not a reconstruction of those inputs.
It supplies no basis for converting the model's failure into a live GO.

Another objection is that better holdout data could change its result. It could,
but it cannot make this primary satisfy both registered bars. The approved
closure avoids selecting a cleaner-looking subset after seeing results and
preserves the unresolved evidence for future data-quality work.

After-tax replay of the failed stock model, voter re-ablation and live fixes are
parked. No real-money stock sleeve is supported by this audit. The A23 owner
allocation and the separate cross-asset shadow retain their own rules.

## Correction (2026-10-20)

This dated addendum records corrections resulting from the pre-registered
momentum re-test specification (`docs/research/briefs/2026-10-20-momentum-retest.md`,
§4, items 1–4) and the Task 1 day-clustered event study. The historical report
body above remains intact.

1. **`dual_momentum` is identical to `xs_momentum` on this replay (§4, item 1).**
   In the 2026-10-19 corrected-harness replay (`_audit_reruns_20261019.json`),
   at `top_n = 50` its negative-momentum filter never removed a pick. Consequently,
   `dual_momentum` produced returns and metrics identical to `xs_momentum`
   (excess CAGR +4.367 percentage points/year over SPY, Newey-West t = 0.761).
   It is not an independent test or separate confirmation of an edge.

2. **Deflated Sharpe Ratio (DSR) gate power limits (§4, item 2).**
   With one-year training folds and a 12-combo parameter grid, the corrected DSR
   gate (`09f9f40`) requires an in-sample Sharpe ratio of roughly 2.5 to pass.
   As a result, walk-forward out-of-sample (OOS) results predominantly reflect the
   per-fold anchor fallback (17/17 folds used the anchor). The gate rejections
   stand; the low pass counts reflect low statistical power of the test under
   short training windows, not proof that there is no underlying edge.

3. **Federal Open Market Committee (FOMC) day-clustered event study (§4, item 3).**
   The unclustered statistics in §2 above (gross Welch t = 3.619, net Welch t = 1.572;
   `fomc_drift.primary_event_study.{gross,net}.t_stat` in `_audit_reruns_20261019.json`)
   counted each meeting day three times (`n_event` = 339) across the three-ETF
   Treasury universe (TLT, IEF, EDV), ignoring cross-sectional correlation on
   shared announcement dates.
   Task 1 recomputed the event study with returns clustered by meeting date over
   the registered window (2012-01-01 to 2026-04-30) for `pair: entry_1_hold_0`.
   Source: `docs/research/_fomc_clustered_event_study_20261020.json`:
   - `n_event_dates`: 113
   - `gross.t_stat`: 2.258 (`gross.p_value` = 0.0258)
   - `net.t_stat`: 0.981 (`net.p_value` = 0.3288; `significant` = false)
   - `unclustered_reference`: gross t = 3.619, net t = 1.572 (`n_event` = 339)
   The day-clustered net event t-statistic falls to 0.981, confirming that after
   transaction costs and clustering by event date, the FOMC drift effect is not
   statistically significant on this Treasury universe.

4. **Prompt templates and momentum re-test (§4, item 4).**
   Prompt templates have been regenerated following `.claude/skills/research-snapshot/SKILL.md`
   to remove superseded claims (such as historical KEEP 0.80 vs 0.78, void "gates 17/17",
   or 1.12 / 16.3%). Momentum is subject to a pre-registered two-part re-test
   protocol detailed in `docs/research/briefs/2026-10-20-momentum-retest.md`.
