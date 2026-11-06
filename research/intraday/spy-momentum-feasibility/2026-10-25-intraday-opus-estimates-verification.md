# Verification of Opus's intraday estimates

**2026-10-25 — documents and calculations only.** The [Opus review](2026-10-25-intraday-overall-plan-opus-review.md) correctly identifies planning questions, but its approximate numbers are conditional arithmetic, not measured strategy evidence. No protocol or gate changes are approved here. [Raw inputs, formulas, software versions, source links, hashes and executable reproduction](_intraday_opus_estimates_verification_2026-10-25.json) accompany this note.

## F2: reproduce the arithmetic; qualify the inputs

Use Opus's **assumed** `N = 2,670`, whole-calendar daily standard deviation `σ = 25 bp`, 252 sessions/year, trades on 50% of sessions, 2 bp per round trip and zero cash yield. “Whole calendar” here means all eligible full sessions, including zero-return flat sessions; it does not mean civil days. The trade fraction is a frequency assumption, not continuous capital exposure.

The IID standard error is `25/√2670 = 0.483821 bp/session`. Therefore `mean/SE = 3` needs **1.451462 bp/session net of costs, before tax**: 2.902924 bp net/trade or **4.902924 bp gross/trade**. The annual arithmetic mean is 3.657684%. Opus's approximately 1.5 / 3 / 5 bp figures are verified conditional arithmetic.

The public final Gao PDF matches the existing [full-text artifact](_intraday_full_text_verification_2026-10-25.json): SHA-256 `f689eefedd3c9de229dc3837fcf7a00ab383e9078783b84762a2f015e20d5fb2`. It is the **2018 final publication with a 1993–2013 sample**, rather than proof of what a pre-2016 manuscript contained. Table 6, printed p404 / PDF page 11, reports annualized last-half-hour always-long volatility **6.21%** and long/short timing volatility 6.19%. Using the standard `√252` volatility conversion, the former implies **39.1193 bp/session**. The reported 6.67% timing mean is **arithmetic annualization**, 2.646825 bp/session, for the original gross long/short rule; it is not a long-only net CAGR. See [Gao et al., Table 6 and footnote 11](https://www.dropbox.com/s/wk5ap8hnvgdyudx/MarketIntradayMomentum.pdf?raw=1).

That table does **not identify 25 bp whole-calendar long-only volatility**. For trade indicator `I`, last-half-hour return `R` and trade fraction `q`,

`Var(I·R) = q·E[R²|I=1] − q²·E[R|I=1]²`.

Conditional moments and frequency are needed. At assumed `q=0.5`, equal conditional second moments and negligible means, a rough published-scale proxy is `39.1193·√0.5 = 27.6615 bp`; 25 bp is about 9.6% lower. That proxy also rests on assumptions. Neither value establishes the later window's volatility or covariance. **Classification: published scale verified; exact 25 bp and 50% frequency unsupported as measurements.**

## A1 power is different from reaching t = 3 on average

Under a normal known-variance planning model, `power ≈ Φ(μ√N/σ − 3)`, so the minimum detectable positive mean at power `p` is `(3 + Φ⁻¹(p))σ/√N`.

| A1 planning target | Net bp/session | Net bp/trade | Gross bp/trade |
|---|---:|---:|---:|
| Nominal mean/SE = 3; approximately 50% power | 1.451462 | 2.902924 | 4.902924 |
| 80% power | 1.858656 | 3.717311 | 5.717311 |
| 90% power | 2.071503 | 4.143006 | 6.143006 |

A finite-sample IID Gaussian, estimated-variance cross-check using noncentral Student t with 2,669 degrees of freedom and rejection at `t ≥ 3` gives **1.858863** and **2.071889 bp/session** at 80% and 90%. This verifies model-conditional MDEs, not power of the proposed HAC gate. [Power definitions](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.solve_power.html); [noncentral t definition and survival function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html).

The draft gate uses five-lag Bartlett Newey–West. Its uncertainty depends on serial covariance, the lag truncation and correction, rather than daily SD alone. For a mean, long-run variance is `γ₀ + 2Σγₖ`; persistent positive correlation raises the MDE. For illustration, stationary AR(1) correlation 0.2 gives a long-run SD multiplier 1.2247, while a five-lag Bartlett truncation gives about 1.1815 before its correction. These are external assumptions, not fitted observations. Heteroskedasticity, tails, regime shifts and estimated HAC uncertainty further limit the reference calculations. They give **no joint A1-and-A2 pass probability**. [Statsmodels HAC assumptions and parameters](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

## A2: 3% after tax does not generally invert to a daily mean

For equal positive full-year gains under the draft's year-end 24% debit, `R_pre = 0.03/0.76 = 3.947368%`. Dividing by 252 gives the **linear approximation** 1.566416 bp/session: 3.132832 bp net/trade, 5.132832 bp gross/trade at 50% frequency. Under a different, constant-daily-compounding construction, the daily rate is `(1+R_pre)^(1/252)−1 = 1.536408 bp`. This explains Opus's rounded comparison without asserting equivalence between A1 and A2.

The actual [draft tax ledger](2026-10-24-intraday-feasibility.md) multiplies annual wealth factors `1 + R_y − 0.24·max(R_y,0)`, taxes the terminal partial year and annualizes by elapsed time. Variable returns, losing years, compounding and tax debits prevent a general `pretax CAGR = after-tax CAGR/0.76` identity. A2 therefore cannot be reduced to one exact unconditional daily mean without a return-path model.

## F1: “under two years” needs a strong benchmark assumption

For illustrative smooth growth, sleeve CAGR 3% and benchmark CAGR `b` on the **same after-tax basis**, relative wealth at year `t` is `(1.03/(1+b))^t`. The 15% shortfall crossing is `ln(0.85)/ln(1.03/(1+b))` when `b>3%`.

| Assumed benchmark CAGR | Smooth crossing, years | First check, quarter-aligned inception |
|---|---:|---:|
| 5% | 8.451 | 8.50 years |
| 8% | 3.429 | 3.50 years |
| 10% | 2.472 | 2.50 years |
| 12% | 1.940 | 2.00 years |
| 15% | 1.475 | 1.50 years |

A continuous crossing before two years requires `b > 11.7192%`; a breach by the aligned 1.75-year check requires `b ≥ 13.0236%`. Real quarter-end timing and return paths matter; terminal CAGR alone cannot establish the first breach. These benchmark CAGRs are **illustrative external assumptions**, not forecasts or historical measurements. Opus's “in most rising markets” probability claim is **unsupported**. The mismatch between standalone and SPY-relative hurdles remains real; this calculation changes neither standard. [Existing forward standard](../../programme/evidence-standard/2026-10-23-forward-paper-standard.md).

## F9: approximately 100 pages describes regular-session page equivalents

For the assumed 2,670 dates and one symbol:

| Grid on those dates | Maximum minute bars | 10,000-row page equivalents, rounded up |
|---|---:|---:|
| Regular, 09:30–16:00 | 1,041,300 | **105** |
| Extended, 04:00–20:00 | 2,563,200 | **257** |
| 24-hour bound per assumed date | 3,844,800 | **385** |

[NYSE Arca hours](https://www.nyse.com/markets/hours-calendars) support the first two grids. A continuous unfiltered request need not return regular hours only. Additional half-days, anchor dates or overnight dates require separate sizing. [Alpaca's bars API](https://docs.alpaca.markets/us/reference/stockbars) permits 10,000 rows, defaults to 1,000, can underfill pages and requires following `next_page_token`. Thus 105 is a packed-page equivalent, **not a verified request count or upper bound on requests**; daily partitioning alone can require thousands of calls. Auction pages are separate. The estimate does not prove sampling has no quality-control value.

An additional **offline cross-check**, using installed `pandas_market_calendars 5.5.0`, `NYSE`, gives 2,700 sessions from **2016-01-05 through 2026-09-30**: **2,679 full 390-minute sessions and 21 early closes**. Starting January 4 adds one full session. At `N=2679`, the conditional t=3 mean is 1.449022 bp, barely different. These are library schedule counts, **not official vendor-calendar verification or validated data eligibility**. The JSON records reproducible schedule hashes. [Library schedule documentation](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html).

## F10: a running count documents reuse; it does not control error

The normal one-sided tail at 3 is 0.001349898. With independent valid null tests, the probability of at least one rejection is `1−(1−α)^m`: **1.3417% for 10 tests**, **12.6355% for 100**. These are hypothetical family sizes, not a count of tests already run. Without independence, the union bound is `min(1,mα)` if each marginal level is valid; adaptive reuse can invalidate those marginal levels. The count and frozen diagnostics improve transparency but do not themselves correct multiplicity or restore an untouched holdout. No new threshold is proposed. [Primary multiple-testing documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

Verification used the existing project interpreter, calculation assertions and the public PDF held only at `/tmp/intraday_opus_gao_final_2026-10-25.pdf`. No new dependencies, credentials, authenticated requests, candidate returns, orders or code changes; no test suite was needed for these documents. Official calendar/eligibility, candidate covariance and actual joint-gate power remain unverified.
