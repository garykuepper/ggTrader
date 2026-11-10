# Objectives and Benchmarks for ggTrader: Consolidated Research Review

**Classification:** Internal Quantitative Research & Engineering Strategy  
**Date:** 2026-11-10  
**Status:** Synthesis and proposals; no benchmark, hurdle, strategy test, or production change approved.  
**Scope:** Four owner-supplied AI research reports. The synthesis claim groups were checked against project records and primary sources; see the [claim verification ledger](2026-11-10-claim-verification.md). This is not a new backtest or a comprehensive audit of every input-report claim.

## 1. Executive Summary & Core Engine Audit

The owner seeks long-term wealth growth, with systematic stock selection, opportunities to hold substantial winners, controlled losses, and possibly resilience during market declines. These aims need explicit trade-offs: a higher risk-adjusted return does not necessarily mean more ending wealth; a strategy that loses less in a bear market does not necessarily earn a positive return during it; a stop does not guarantee its trigger price. No single result demonstrates that all aims have been met.

**Proposed, not adopted:** assess after-cost, estimated after-tax portfolio wealth against an investable passive alternative under prespecified risk constraints. Retain SPY total return as a visible comparator. Separately measure stock-sleeve contribution and defensive behavior against suitable counterfactuals. Pick the primary benchmark, economic hurdle, risk budget, and tax scenarios before viewing the new evaluation results. The owner has not selected those terms.

The historical production-rule replica remains **NO-GO under its registered return and significance rules**; a later audit found it inconclusive for a modest edge. It is a daily-bar model, not a verified reconstruction of the paper trader. Exact live parity is unverified. The pre-2026-11-06 numbers below use the legacy vectorbt engine and are not automatically validated on the Rust engine. Source: [replica report](../../audits/production-replica/2026-10-19-production-replica-audit-nogo.md), [research snapshot](../../RESEARCH_SNAPSHOT.md), and [root-cause executive summary](../../audits/root-cause-2026-11-04/2026-11-05-executive-summary.md).

The momentum, classic factor, and net-issuance screens are closed NO-GO on their registered terms. Historical leveraged rotation also closed NO-GO; the static cross-asset 80/20 result is a **shadow-only GO under its own bar**, not proven timing skill. Do not rebrand an old test as a new signal. See the [research snapshot](../../RESEARCH_SNAPSHOT.md) before proposing a test.

## 2. Quantitative Performance Context

| Historical daily-bar replica, primary window | CAGR | Sharpe | Maximum drawdown | Sizing / exposure |
|---|---:|---:|---:|---|
| Fixed-rule replica, SPY sweep | 14.19% | 0.925 | −21.44% | Mean 9.37% stocks, 85.33% SPY, 5.30% cash |
| SPY buy-and-hold | 14.83% | 0.903 | −24.50% | Fully invested comparator |

The replica's excess CAGR was −0.642 percentage points/year and Newey–West t was −0.690. Sharpe uses the historical **zero risk-free rate**; its small apparent difference does not establish a meaningful advantage. Gate pass rate is not applicable to this fixed-rule replica. The baseline comparison and its limitations are in the [replica report](../../audits/production-replica/2026-10-19-production-replica-audit-nogo.md); do not transplant numbers from a different window or engine.

The audit estimated the whole-account test could detect approximately 4.2 percentage points/year under its assumptions and conditional on the PARA data issue being immaterial; it could not reliably resolve the smaller registered effect. This is an audit estimate, not proof of any positive edge. The project’s prior NO-GO and the audit's assessment of limited power can both be true.

## 3. Actionable Research Directions

These are proposed planning or feasibility tasks, **not registered experiments**. No return series has been opened or computed for this synthesis. A rank reflects decision value and feasibility, not a measured probability of success or expected profit.

### 3.1 Check whether historical attribution is possible

**Question:** Can existing frozen replica outputs reconstruct the daily active positions, cash, cash flows, dividends, transaction costs, and decision times needed to replace active stock exposure with SPY at the same times and capital allocation?

**First action:** inventory the actual pinned JSON, curves, code, and any retained trade/position records, and report whether each field is present, reconstructible, or absent. Do not calculate or inspect new performance as part of this feasibility step. Check source pins, historical engine, corporate actions, the PARA caveat, and whether original decision inputs exist. Do not equate a modeled replica with live parity. If essential records are missing, report attribution as unresolved; designing prospective collection is a separate proposal.

If feasible, write an attribution brief **before calculation** defining a daily self-financing counterfactual, idle-cash treatment, decision and execution lags, distributions, costs, taxes, and rebalancing. Separate stock selection, cash/exposure timing, and implementation cost; document interactions and avoid claiming an order-independent decomposition without proving it. Compare the same dates and risk exposures. Report uncertainty and inputs, not a reconstructed live result. This analysis would not reopen the historical verdict.

### 3.2 Specify a benchmark framework

- **Whole portfolio:** SPY total return including reinvested dividends, plus a prespecified investable passive alternative with comparable target exposure or risk and symmetric cost, tax, funding, and cash-flow assumptions.
- **Stock sleeve:** the SPY replacement counterfactual above, with explicitly defined capital base and treatment of idle cash. Report factor/sector tilts as diagnostics; an exposure match alone does not isolate stock-selection skill.
- **Defensive objective:** compare wealth, drawdown, conditional returns, recovery time, and opportunity cost against both SPY and an investable passive mix. A risk-matching rule must use only lagged information, with estimation window, frequency, caps, financing and costs fixed before evaluation. A full-sample matched mix is descriptive, not a forward-investable benchmark.
- **Leveraged or multi-asset construction:** use an appropriate additional leverage/beta-aware counterfactual and disclose path dependence and costs; beta matching does not erase other risk differences.

The owner must choose horizon, tolerable drawdown and benchmark underperformance, and the return they would exchange for reduced risk. A single historical maximum drawdown is an uncertain, sample-specific observation, not a guarantee.

### 3.3 Preserve the next stock research and paper-evidence plans

The current worklist calls for [accruals feasibility](../../stock-signals/feasibility/2026-11-02-accruals-feasibility.md) and [paper parity and forward evidence](../../paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md). Accruals is **unresolved**: usage terms, accounting revisions, cell metadata, benchmark and costs remain to be settled before registration; stock-level point-in-time reconstruction is separate and unproven. The [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) provide free submissions and standardized XBRL data, not a turnkey survivorship-safe point-in-time trading dataset.

A prospective exit ablation or buffered-rank idea is a hypothesis only. Before selecting it, compare exact entry, universe, lookback, exit and cost rules against prior studies; specify execution, gaps, re-entry, corporate actions, and what happens to released cash. Hold the entry and sizing rules fixed when isolating exit effects. The SEC notes that a stop-market trigger is not a guaranteed execution price, while a stop-limit may not execute ([SEC investor bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15)). Daily OHLC alone may not determine intraday event ordering. A failed specification closes that test, not the stock research programme.

## 4. Completed Findings and Corrections

1. **Original replica NO-GO stays intact.** Changing the question or the benchmark later cannot turn an earlier registered result into a pass. Previous return and significance bars and execution assumptions remain part of the evidence record.
2. **No proof that cash caused the historical shortfall.** One supplied report computes 5.30% × 14.83% assuming zero cash yield and claims this explains the replica gap. The primary replica credited T-bill returns; its cash-at-zero diagnostic had 14.00% CAGR versus 14.19% in the primary. That diagnostic is not a full attribution either. Use a dated, self-financing counterfactual rather than that shortcut.
3. **Fix the information-ratio arithmetic.** At 1 percentage point annual *arithmetic* active return and 4%–6% tracking error, illustrative information ratio is about 0.17–0.25; 2 points implies about 0.33–0.50. Do not convert CAGR differences mechanically.
4. **Do not adopt speculative tax thresholds.** One report's table credits liquidation loss benefits without a reproducible lot-level ledger or demonstrated ability to use losses; its numerical break-even is unverified. Model basis, distributions, realized gains/losses, wash sales, carryforwards, tax funding and liquidation consistently under labeled hypothetical scenarios. [IRS Publication 550](https://www.irs.gov/publications/p550) supplies general rules; actual treatment depends on facts and law. No personal tax rate is assumed.
5. **Correct unverifiable citations and sample dates.** The first supplied Markdown report contains unresolved internal citation markers. Another attributes the original Deflated Sharpe Ratio to an unrelated 2026 arXiv paper; the [original Bailey–López de Prado paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) is the appropriate reference. SPY began in 1993, so pre-inception analysis needs an explicitly identified index or proxy ([State Street](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy)).
6. **Do not infer a universal return ceiling or failure probability.** In the [S&P DJI year-end 2025 SPIVA report](https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2025.pdf), 85.59% of all large-cap funds underperformed the S&P 500 over ten years, 89.93% over fifteen, and 92.89% over twenty. These are historical fund-category comparisons, not the probability of failure for this project. A preliminary source check incorrectly assigned 92.89% to fifteen years; the primary table resolves the error.

## 5. Operational Roadmap: Recommended First Action

**Inventory the replica artifacts for attribution feasibility without running a return test.** The output should list each necessary input, its source, hash and availability, and whether any reconstruction would rely on assumptions. Then decide whether to draft a preregistered attribution brief or a prospective evidence-collection plan. Do not run, select thresholds, change a research verdict, change production settings, or modify historical artifacts as a consequence of this report. Keep the existing paper-parity and accruals-feasibility work visible in [next steps](../../../docs/next_steps.md).

### Evidence and power

A useful diagnostic compares paired strategy-minus-benchmark arithmetic returns. Under independent, normally distributed observations, stable known variance, and a normal critical-value approximation, an approximate two-sided 5% test with 80% power requires `n ≈ 7.84 × (tracking error / annual mean active return)²` independent years. For an illustrative 1% annual mean excess and 2% annual tracking error, that is approximately **31.4 independent years**, before practical complications. It is not a CAGR formula or a guarantee. Overlapping folds, autocorrelation, changing regimes, estimated variance, multiple testing, and selection can substantially alter power; more correlated trades do not create independent years. A result near zero can be inconclusive, not an automatic fail. Use preregistered materiality and confidence intervals with a pass/fail/inconclusive rule.

Sharpe is estimated with uncertainty and serial correlation can distort naïve annualization; see [Lo, The Statistics of Sharpe Ratios](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios). Pre-tax, after-tax without liquidation, and after-tax liquidation wealth should all be reported using consistent assumptions. Do not assert that 0.5, 1, or 2 percentage points per year is attainable because it is economically attractive.

## 6. Contrarian Evaluation & Parked Research

The strongest objection to benchmark work is that a cleaner metric cannot manufacture a stock-selection edge, and short samples might remain inconclusive even with perfect accounting. Conversely, absence of a detectable edge is not proof that none exists. Feasibility and transparent attribution are useful because they establish what the existing evidence can and cannot answer; new strategies still need their own registered tests.

Managed futures, shorting, and options may have different return and crisis profiles, but free point-in-time data, financing, borrow, execution and tax modeling, and implementation are not established for this project. Treat them as research candidates, not assumed executable solutions. A long-only trend or cash rule may lose less without earning positive crisis returns, and may lag in reversals. Check previously closed strategy specifications before any new experiment.

## Source provenance and limits

Selected primary-source PDFs, original URLs, hashes, and download limitations are in the [source archive](sources/README.md).

The four owner-supplied reports are external AI-generated inputs, not primary evidence or independently verified results. Their originals are T3 conversation attachments, not repository files, and may not survive moving the checkout. SHA-256 identifies the exact source if retained elsewhere. No source attachment was copied into the repository.

| Supplied document | SHA-256 |
|---|---|
| `deep-research-report(4).md` | `0252b429323ed405d793c5cb5ec866906924587bc23f69b5732c466be266f119` |
| `ggTrader_ A Realistic Objective, Benchmark Framework, and Resear.md` | `4c43a2b8952f0a8d2d087f0ac2c257b00401d0b8284a7697b1bfdd465ea6b060` |
| `ggTrader Objectives_ Benchmarks_ and Testable Hurdles.md` | `a05ab1c1f5d0bec656a89c546a45a9e6e559c13ecc09118e90d22093b5dff2df` |
| `Systematic Trading Objectives and Benchmarks.pdf` | `0e49bf0b1ce5b35b2778eff624764d9595f2ac4908d33288cb76f89c2061409b` |

The [claim verification ledger](2026-11-10-claim-verification.md) records source checks, mathematical derivations, proposals, and remaining gaps. It did **not** validate every citation, tax calculation, study claim, or strategy ranking in the four documents. The primary-source research assistance was same-model-family and the lead checked the SPIVA table; it was not an independent cross-family audit. No new research result, raw market data, or trade was generated.
