# Brief: intraday feasibility screen — SPY first-half-hour → last-half-hour momentum

**Status:** DRAFT r6, not approved. No intraday prices, signals or strategy returns have been inspected or computed. The daily SPY benchmark alone has been reproduced with a saved artifact. Proposed gates and the window still need owner approval. r5 adopted the owner’s long-only decision; r6 corrects source, tax and execution claims after the [verification review](../2026-10-24-intraday-trading-feasibility.md). Nothing here authorizes a probe or trading.

**Terms used below:**

- *PDT* (pattern day trader) was the FINRA rule requiring $25,000 equity to day trade. FINRA's real-time intraday margin standard replaced it effective 2026-06-04; firms needing more time may phase in until 2027-10-20 ([Alpaca's summary](https://docs.alpaca.markets/us/docs/the-intraday-margin-rule)). Alpaca says it implemented on the effective date, but its house margin terms remain the operative constraint.
- *IEX* is a single exchange; *SIP* is the consolidated tape of all US exchanges.
- *NW t* is the Newey-West t-statistic: mean daily return divided by a standard error that allows serial correlation and changing variance.
- *bps* is a basis point, 0.01%. *Round trip* is one entry plus one exit.
- *Auction close* is the official 16:00 ET closing-auction price, as opposed to the last minute-bar close.

## 1. The question, and the decision it feeds

- **Question.** Does the long-only variant of the Gao–Han–Li–Zhou rule (buy SPY after a positive first half-hour return, otherwise stay flat) earn a net-of-cost, tax-policy-adjusted return worth trading, after the original sample and first public posting?
- **Decision fed.** Whether ggTrader opens an intraday research track at all (minute-bar data pipeline, intraday harness). A pass authorizes more research only. A fail closes this idea and the track's first candidate, and says nothing about intraday trading as a category.
- **Owner context.** The account is taxable only. Alpaca confirms that it removed PDT restrictions in production on June 4, 2026. Live-account provenance, buying power and settlement permissions still need account-specific confirmation before any shadow design (§11). The main constraints are costs, taxes and edge.

## 2. Why: verified evidence and its limits

- **Original idea:** the [author-affiliated Gao publication record](https://www.researchwithrutgers.org/en/publications/market-intraday-momentum/) and a full June 2017 manuscript confirm the prior-close anchor, 1993–2013 sample and final-half-hour long/short strategy. The final 2018 published full text remains inaccessible. This long-only, minute-bar rule is a deliberate variant; no published result validates it directly.
- **Publication label:** 2016 is after the 2014 SSRN posting and original sample, but before the final 2018 journal publication. Use “post-original-sample / post-first-public-release,” not “post-journal-publication,” for the full window.
- **Replication evidence:** the [source audit](../2026-10-24-intraday-replication-source-audit.md) confirms several reported results but finds signal, leverage and cost mismatches. One SPY replication’s Sharpe-significance code has inconsistent annualization; another describes strong early results and weak later results. The Substack test does not clearly include the prior-close anchor. None decisively tests this rule.
- **Mechanism:** short-gamma hedging is supported as a possible intraday-momentum mechanism. Its main source uses a different predictor; a causal explanation of recent decay by rising 0-DTE volume is not established. No gamma filter is registered.
- **Research prior:** cost sensitivity and regime dependence justify skepticism. They do not establish either a profitable long-only edge or a foregone NO-GO. Existing daily-bar NO-GO findings remain closed; see the [research roster](../RESEARCH_SNAPSHOT.md).

## 3. Data — free only, and a staged probe comes first

The lab is daily-bar only. `ohlcv` holds no intraday equity data, and `docs/roadmap.md` records free intraday history as the blocker for earlier intraday ideas. Paid data is off the table.

**Documented limits (checked 2026-10-24 on Alpaca's "About Market Data API" page).** Historical stock data on both the Basic (free) and Algo Trader Plus plans is listed as "Since 2016". Basic's historical limitation is "latest 15 minutes", so older SIP data should be queryable for free. Basic allows 200 historical calls a minute. The pending probe will test entitlement and coverage for these credentials.

**Step 0 requires separate authorization for its coverage and price-quality stages.** Record everything in `docs/research/_intraday_probe.json`, with no secrets logged. Stop at the first failing stage.

**Stage 0A — coverage and structure (timestamps, counts, volumes, status codes, errors only; no prices inspected):**

1. **Entitlement:** one old-date SPY 1-minute request on `feed=sip` and one on `feed=iex`.
2. **Coverage:** request a tiny interval in 2014, in 2015, and in the first weeks of 2016 for each feed. Record the earliest date returned within the sampled intervals; this is not proof of the archive’s absolute first date. Classify any failure as permission error, throttling, pagination mistake, transient failure, or a genuine coverage boundary; an error response alone does not prove history is missing.
3. **Bar timestamp semantics:** confirm from response structure that a bar is stamped at the start of its minute (a bar stamped 15:30 covers 15:30:00 up to but not including 15:31:00), per Alpaca's FAQ.
4. **Auction data:** test whether the [SIP-only historical-auctions endpoint](https://docs.alpaca.markets/us/reference/stockauctions-1) returns SPY records over the same dates. Record the earliest sampled observation and exchange identifiers; the official primary-listing closing auction must be distinguished from other venues’ auctions before endpoint selection.
5. **Calendar:** pull official session opens and closes (including early closes) from Alpaca's calendar endpoint, and freeze its version and the `America/New_York` time-zone conversion.
6. **Completeness sample:** bar counts against the calendar-derived minute grid on the 20 sample dates, and volumes by feed.

**Stage 0B — price quality (needs its own owner approval; runs only if 0A passes):** check that prices are positive and finite, IEX-versus-SIP endpoint differences, and close reconciliation (§9). It does not compute morning returns, afternoon returns, strategy sides or P&L. The 20 sample dates are fixed by a deterministic calendar-only rule before any price is seen: the first full regular session of each of 20 evenly spaced months across the window, so no session is picked for looking clean.

**Stage 0C — bulk pull:** only after 0A and 0B pass. Record rate limits and wall-clock time.

**Provenance.** Every cached pull records request time, feed, adjustment mode, endpoint, parameters, response status, pagination completion, earliest returned timestamp, and a content hash, because Alpaca can revise a minute bar for late trades. Retries and corrected pagination to recover the provider's original records are allowed; interpolation, selective deletion and replacement are not.

**Feed precedence.** SIP is the primary research feed when it covers the full window. IEX is a sensitivity feed only, unless the owner approves a separate IEX-based brief.

**Stop rule.** If the chosen feed cannot supply gap-free 1-minute SPY history from the registered start (§6), this study ends **UNRESOLVED (data)**, not NO-GO: a missing archive is not evidence against the effect. No shortened window after the probe, no interpolation, no substitute ticker, no paid feed.

## 4. Hypothesis: a family of ONE

**H1:** the long-only variant earns a positive net return worth trading. On each full regular session, at 15:30 ET, go long SPY if `r_first > 0`; otherwise stay flat. Exit at the close. A1 and A2 apply only to this long-only rule.

**Price conventions (frozen before any return is viewed; the probe may choose between the two marked alternatives, by availability only):**

| Endpoint | Price used |
|---|---|
| Previous close, `P_close(t−1)` | Auction close if the auctions endpoint covers the full window; otherwise the close of the 15:59 bar. One global choice, not per session. |
| 10:00 signal price | Close of the 09:59 bar (the last trade before 10:00) |
| 15:30 entry price | Open of the 15:30 bar (the first trade at or after 15:30) |
| Exit price | Same convention as the previous close (auction close, or the close of the 15:59 bar) |

- **Exit mechanism.** If the auction close is used, the intended exit is a market-on-close (`cls`) order submitted after the entry fill is confirmed, before Alpaca’s 15:50 cutoff. Its [order matrix](https://docs.alpaca.markets/us/docs/orders-at-alpaca) does **not** support fractional `cls` orders. Whole-equity fractional sizing is therefore an idealized research ceiling, not a directly executable auction design. A later shadow brief must specify whole-share sizing with residual cash, or separately register another exit convention, and handle partial fills and rejected exits. If only the 15:59 proxy is available, label the result a **proxy feasibility screen**; it is not an observed auction execution.
- Prices are `adjustment=raw`. Dividend-adjusting would change the overnight signal on ex-dividend dates.
- **Prior-close anchor.** The first test session needs the previous session's close. That close (2015-12-31) is a signal input only; no pre-2016 observation enters a test or performance series. If it is unavailable, the first test session is the next day. A half-day's actual close is the previous close for the next full session.
- `r_first = P_0959 / P_close(t−1) − 1`; `r_last = P_exit / P_1530 − 1`.
- **Flat rule:** every day with `r_first <= 0` is flat (no trade, no cost). Omitting the source rule’s short leg reflects the owner’s trading intent. Report negative- and zero-signal day counts separately.
- Skip half-day sessions as trading days, taken from the official calendar.
- No filters, volatility conditioning, or parameter search. The paper's conditioning variants are not tested here (§7). Choosing this candidate after other screens is not covered by any error-rate correction.
- This is a bar-based implementation of the paper's idea, not an exact replication; the paper's trade filters and adjustment conventions have not been mapped line by line.

## 5. Cost and tax model *(proposed)*

- **Cost.** Charge **1.0 bp per side** (2.0 bps per round trip) in the base case and **2.0 bps per side** in the stress case, as a first-order subtraction from the daily return. This stands in for spread, slippage and close-order uncertainty. This is a proposed policy haircut, not a measured execution cost or a claim of conservatism. Paper-account fills are not a cost estimate.
- **Long-only sizing.** Use at most 100% of then-current sleeve equity, without borrowing or short positions. Charge costs only on traded days.
- **Tax algorithm.** The strategy sleeve is compounded on 100% of then-current equity. At each calendar year-end, debit **24%** of that year's positive net realized strategy P&L from sleeve wealth. A zero or negative year earns **no tax benefit**, and a loss does not reduce tax in any later year. The debit is taken from sleeve equity on the last session of the calendar year, so it reduces the next trade's sizing base. The partial final year (2026-01 → 2026-09) accrues its modeled liability at the terminal valuation, so it is not accidentally tax-free. This omits loss carryforwards, wash-sale basis adjustments, state taxes, other liabilities and interactions with other SPY holdings. The omissions can work in different directions, so the model is not uniformly conservative. It is a **tax-policy hurdle, not a tax-return simulation**; 24% is a project convention, not the owner’s actual bracket. [IRS capital-gain rules](https://www.irs.gov/taxtopics/tc409) and [wash-sale rules](https://www.irs.gov/publications/p550) were read; a personal lot-level tax estimate remains open.
- **Cash.** The sleeve earns zero on cash outside the 30-minute holding window. Starting capital, valuation timestamps and external-flow policy are identical for the sleeve and the benchmark; strategy taxes are paid from sleeve wealth.

## 6. Tests *(proposed)*

Window: **2016-01-04 → 2026-09-30**, the first session of Alpaca's documented history through the last full month. The paper's 1993–2013 sample is excluded from every test. This start is proposed *before* any intraday data or strategy returns are inspected and is justified by the documented data limit, not by a result. If the probe shows undocumented 2014–2015 coverage, that does not move the window; changing it needs a new dated brief.

Daily strategy return `s_t = side_t × r_last,t − cost_t`, where `side_t = 1` if `r_first > 0`, otherwise 0. `cost_t` is the round-trip cost on traded days and zero on flat days.

- **A1 — profitability of the fixed rule.** One-sided NW t on the mean of `s_t` from an intercept-only regression, Bartlett weights, **exactly 5 lags** (a lag is one eligible full session), statsmodels HAC with its small-sample correction setting and software version recorded before the run. All non-positive-signal (flat) days stay in the series as zero returns. One test, so no Holm correction. **Pass: unrounded t ≥ 3.0.** The 0.00135 one-sided tail is the standard-normal reference at 3.0, not an exact finite-sample guarantee. If the variance estimate is zero or non-finite, the result is statistically unresolved, not an infinite t. NW lags 0, 10 and 20 are diagnostics and cannot change the verdict.

  A1 is a profitability gate, not a test of predictive dependence: `E[d·R] = Cov(d, R) + E[d]·E[R]`, where `d` is the long-only indicator, so the rule can have a positive mean from exposure to unconditional afternoon drift even if the morning sign adds no information. The attribution diagnostics in §7 address that and are non-gating. If the owner wants a formal predictive-dependence gate instead, that must be chosen now, before any result, and not added later.
- **A2 — standalone economics *(proposed; replaces r3's SPY-displacement gate, see §11.3)*.** The tax-adjusted sleeve (§5 ledger: 100% of then-current equity in the 30-minute window, idle cash at zero, annual 24% debit) must earn **an after-tax-policy CAGR of at least +3.0% a year** over the window. This is a proposed standalone hurdle for further research, not a derived economic minimum or a claim that the sleeve can replace SPY. The SPY total-return comparison (distributions reinvested, no interim or terminal tax) is reported as a **non-gating displacement diagnostic**.
- **Verdict.** H1 passes only if A1 **and** A2 pass at base cost. The stress cost is reported but cannot rescue a failure or fail a pass by itself.

If A1 passes but A2 fails, the rule meets the profitability test but fails the proposed economic hurdle for this candidate.

## 7. Diagnostics (never pass criteria)

- Subperiods: 2016–2019, 2020–2023, 2024–2026-09. NW t (lag 5), mean bps/day, hit rate in each.
- **Attribution (non-gating):** a regression of `r_last` on the long-only indicator `1[r_first > 0]` with an intercept, and an always-long last-half-hour comparator, to separate predictive information from afternoon drift.
- Break-even one-way cost in bps: total gross return divided by twice the number of round trips, not a calendar-day mean halved, because flat days exist.
- Primary long-only performance, reported both conditional on trading and as a whole-calendar sleeve with flat days retained; negative- and zero-signal day counts. No short-only or combined long/short candidate is tested.
- NW lags 0, 10 and 20.
- The paper's volatility, volume and macro-news conditioning is **not** tested: those labels have no operational definition here, and using full-day information unavailable at 15:30 would not be a prospective filter. A later brief may register them with frozen formulas.
- Worst month, longest losing streak, and maximum drawdown of the 100%-allocated sleeve.
- IEX-versus-SIP rerun if both feeds are available. SIP is the primary research feed whenever it is available for the full window; IEX is a robustness check only.
- Auction close versus the 15:59 bar close versus the raw daily close, as a convention-sensitivity table, including the distribution of |15:59 close − daily close| in bps.
- **Displacement comparison (non-gating):** sleeve versus SPY total return with distributions reinvested. Report the gap without assuming its sign. Use a distribution-reinvested benchmark aligned to the strategy’s valuation timestamps; published price-only comparisons are not substitutes.
- **Ex-dividend sensitivity:** the primary result uses raw prices; also report it with SPY's ex-dividend sessions set flat, because ex-dividend overnight repricing can change the sign of `r_first`. No fixed dividend-size estimate is assumed.

## 8. What happens next

| Result | Next step |
|---|---|
| Probe stop rule fires | Close **UNRESOLVED (data)**; record what the free feeds can and cannot supply. |
| A1 or A2 fails | Close **NO-GO**; publish the failed gate(s) and diagnostics. The verdict applies to this candidate; it does not decide other intraday research or existing paper observation. |
| A1 and A2 pass | Write a separate fixed-rules brief for a forward paper shadow (intraday orders in the Alpaca paper account, separate from the frozen production strategy) under the [approved forward-paper standard](2026-10-23-forward-paper-standard.md). No capital, no production change. |

## 9. Data-quality stops

Stop before any verdict on the following. Do not drop, fill or patch observations; report the defect and leave the verdict unresolved.

- **Missing minutes.** Build the expected minute grid for each full session from the official calendar. Every absent minute must be explained by a documented exchange or market-wide halt (with the date and duration verified from an exchange source) or the study fails this measurement requirement and stops. Alpaca generates no bar for a minute without qualifying trades, so an absence can reflect trade coverage rather than a corrupt file; the all-minutes requirement is kept as policy and is not relaxed after gaps are seen. A session is eligible only if every endpoint price in §4 can be built exactly under the frozen convention.
- Duplicate timestamps, and non-positive or non-finite prices.
- **Close reconciliation.** Record the auction close, the 15:59 bar close and the raw daily close for each session. A difference is fatal only if it cannot be explained by the published construction rules, trade conditions, a corporate action or a documented vendor revision. There is no blanket 1 bp rule.
- **Calendar.** An unflagged half-day.
- **Corporate actions.** Ordinary ex-dividend overnight repricing is part of the raw price move and is *not* an intraday basis change. Only a true intraday discontinuity or an unresolved vendor adjustment change is fatal. Ex-dividend sessions are never dropped.

## 10. Scope and boundaries

- **This task:** this brief only. No bars pulled, no probe run, no returns computed before owner approval.
- **Later approved implementation:** new code only under `src/ggTrader/lab/` and `scripts/`, with raw results to `docs/research/_intraday_momentum_results.json`. Cache pulled bars under `data/external/` (never delete there); the pull must be resumable and write the §3 provenance manifest.
- **Untouchable:** `src/ggTrader/paper/`, `src/ggTrader/data/live/`, `.env`, Docker files and the production cron. No paid dataset or subscription. No parameter search or post-result hypothesis swap.
- **Parked, not registered:** the Zarattini–Aziz–Barbon SPY noise-band rule (SSRN 4824172) and the “stocks in play” opening-range breakout (SSRN 4729284). Their original full papers have now been read. They permit shorting and leverage; the single-stock study uses CRSP and IQFeed. A free, complete historical universe has not been demonstrated here. Each needs a separate long-only, unleveraged specification before any test. Also parked: single-name intraday reversal, and any leveraged-ETF intraday rule.

## 11. Decisions, source corrections and remaining conditions

1. **Candidate:** the owner confirmed no planned shorting on 2026-10-24. Long-only is the sole primary rule; short-only and combined long/short variants are not tested. This decision does not validate its edge.
2. **Window (proposed):** 2016-01-04 → 2026-09-30, based on Alpaca’s documented history. The earlier 2014 option is unresolved pending the owner’s choice. A probe must not silently shorten or move a window after inspecting results.
3. **A2 (proposed):** +3.0% standalone after-policy CAGR remains a policy judgment requiring approval. It is not derived from A1’s t threshold or from the long/short paper. The earlier 4.5 bp/day edge estimate and 3.7 bp/day statistical floor were not verified and are withdrawn from the justification. “The old SPY-displacement gate could never pass” was too strong; it was demanding, not mathematically impossible.
4. **Benchmark artifact:** the adjusted-close proxy was reproduced and saved to [JSON](../_intraday_benchmark_verification.json), including dates, provider/version, raw daily inputs and hash. These close-to-close figures exclude the first session’s return. A future comparison must align timestamps and reconstruct or verify distributions; this artifact is not an intraday result.
5. **A1 choice:** profitability remains proposed. It does not prove that the morning signal adds information beyond afternoon drift. A formal dependence gate must be chosen before any returns, rather than added after a result.
6. **Cost and tax:** 1 bp per side, 2 bp stress and the annual 24% policy ledger remain assumptions. Actual slippage and combined SPY wash-sale/lot effects have not been measured. IRS rules do not justify treating every profitable day as a separately taxed item or the simplified model as uniformly conservative.
7. **Alpaca implementation:** [Alpaca’s announcement](https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/) confirms removal of PDT in production on June 4. The [FINRA notice](https://www.finra.org/rules-guidance/notices/26-10) was read directly. An industry phase-in does not imply that Alpaca has delayed implementation.
8. **Connected account:** read-only checks return multiplier cap “1”, fractional trading enabled, shorting not disabled, and no reported account/trading block or suspension. The tool response does not identify paper/live environment; earlier context identified paper. This is not live-account permission verification. No settings changed. Long-only enforcement and operational buying power belong to the later shadow design.
9. **Order compatibility:** fractional `cls` is unsupported in the documented matrix. A whole-share auction exit or a different registered exit must be designed and tested before a shadow. An idealized fractional backtest cannot certify execution readiness.
10. **Evidence corrections:** the original QQQ paper is SSRN **4416622**, not 4898296. The noise-band 6.2% baseline is not a fixed final-half-hour rule. The [replication audit](../2026-10-24-intraday-replication-source-audit.md) documents source access levels, the Sharpe unit error, mixed regimes and the distinction between zero passing and zero profitable ORB cells. These corrections remove overstatements; they do not approve a different hypothesis or gate.
11. **Remaining documentary gaps:** final Gao published-version details, Delgado/Demirev full methods, registration chronology and reproduction of outside raw returns. Unsupported precise cost and 0-DTE claims are excluded. Full archive access, data quality, actual costs and the strategy edge require empirical work that has not started.

**Approval boundary:** the owner requested source verification before proceeding. This revision records that review; it does not approve Stage 0A, a window, A1/A2, bulk pulls, strategy returns or a paper shadow. Keep those steps pending. The production freeze and existing paper observation remain in force.
