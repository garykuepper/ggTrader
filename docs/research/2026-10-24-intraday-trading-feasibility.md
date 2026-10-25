# Intraday algorithmic trading: verified feasibility review

**Date:** 2026-10-24. **Revision:** source-verification pass, superseding the earlier draft's unverified synthesis.
**Scope:** taxable account, Alpaca, long-only, no borrowing, free data. No intraday bars or strategy returns have been obtained. No orders, settings, deployment or production changes were made.

**Conclusion:** Alpaca's former PDT trade-count and $25,000 restrictions are removed. The literature supplies reasons to investigate intraday strategies, but it does not validate the owner's proposed long-only rule. Several earlier negative conclusions were overstated. Research remains **UNRESOLVED**, with the proposed experiment still a draft.

Evidence labels distinguish **full primary text**, **primary abstract**, **author account/code**, and **unverified**. Reading a published result confirms what an author claims; it does not reproduce the result. Detailed replication checks and access failures are in the [source audit](2026-10-24-intraday-replication-source-audit.md); document fingerprints, the arithmetic self-check and selected non-identifying account flags are in the [verification artifact](_intraday_source_verification.json).

## 1. Day-trading permissions and account limits

**Confirmed from the regulator:** FINRA Notice 26-10 replaces the PDT designation, trade-count criteria and $25,000 minimum. Effective date: June 4, 2026; transition deadline for firms: October 20, 2027. [FINRA notice](https://www.finra.org/rules-guidance/notices/26-10).

**Confirmed from the broker:** Alpaca states that it implemented the replacement in production on June 4 for both Trading API and Broker API users. The industry transition period is therefore not evidence that Alpaca is still enforcing PDT. Its orders remain subject to available buying power, intraday risk checks and house restrictions. [Alpaca implementation announcement](https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/).

Alpaca's unleveraged-account FAQ permits repeated entries and exits, subject to sufficient equity. Margin-enabled accounts can avoid T+1 settlement delays without borrowing. Its $2,000 threshold concerns margin/short access; it is not a replacement PDT trade-count threshold. This documentation does not establish every settlement permission for an individual account below $2,000. [Unleveraged-account FAQ](https://docs.alpaca.markets/us/docs/understanding-the-new-intraday-margin-rule), [margin rules](https://docs.alpaca.markets/us/docs/margin-and-short-selling).

**Read-only account check:** the connected Alpaca account reports `max_margin_multiplier="1"`, `no_shorting=false`, `fractional_trading=true`, `closing_transactions_only=false`, and no reported trading/account block or user suspension. No setting was changed. The tool response does not identify the API environment, so this check does not independently certify the live account; the earlier conversation identified the connection as paper. Live-account provenance and funding/settlement permissions remain open. No order was placed to test permissions.

## 2. What the primary strategy sources actually say

### Gao–Han–Li–Zhou: first half-hour to last half-hour

**Primary abstract and June 2017 working paper read.** The original SPY sample is 1993–2013. The morning predictor includes the previous close through 10:00, not just 09:30–10:00. The sign strategy trades the final half-hour in both directions; a non-positive morning signal shorts. The working paper's Table 6 reports annualized arithmetic mean 6.67% and Sharpe 1.08 before costs. Those are historical long/short figures, not a CAGR or a long-only result. The source's own recursive OOS tests occur within its historical sample; they are not post-publication validation. [Author-affiliated publication record](https://www.researchwithrutgers.org/en/publications/market-intraday-momentum/), [June 2017 manuscript, §§2/4 and Table 6](https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/ee7dac49-530b-4950-b5d0-e0b5eee08f2e.pdf).

The publisher's final 2018 full text was inaccessible. The available manuscript establishes the anchor and direction, but exact final-version filters and price adjustments are not fully reconciled. Our minute-bar, long-only implementation must be called a variant, not an exact replication.

### Zarattini–Aziz–Barbon: SPY noise bands

**Full author-hosted paper read, version September 22, 2025.** Table 3 distinguishes three strategies: opposite-band stop (6.2% annualized, Sharpe 0.61); current-band/VWAP stop (9.7%, 1.24); and dynamic sizing (19.6%, 1.33). The last uses up to 4x leverage. All permit long and short trades. The 6.2% baseline was incorrectly described in the earlier draft as an unconditional 15:30–16:00 rule; it is a noise-band strategy. The study window is May 2007–April 2024. Its cost model includes per-share commission and assumed slippage; the authors' small execution experiment is not an Alpaca cost measurement. [Paper, §§3/4.6 and Tables 1–3](https://concretumgroup.com/wp-content/uploads/2026/02/Beat-the-Market.pdf).

None of these published numbers can be transferred to a long-only, 1x implementation. The revised manuscript also contains later performance updates, making paper-version and first-publication dates important when labeling OOS evidence.

### QQQ opening-range strategy and single-stock “stocks in play”

**Full original QQQ paper read.** Its correct identifier is **SSRN 4416622**, replacing the erroneous 4898296 citation. The January 2016–February 17, 2023 result uses both directions, up to 4x leverage, commission and zero modeled slippage. Its entry is the second five-minute candle's open based on the first candle's direction; this is not a requirement to wait for a later range break. [Original QQQ paper](https://concretumgroup.com/wp-content/uploads/2026/02/Can-Day-Trading-Really-Be-Profitable.pdf). The source audit retains the quoted performance and implementation details.

The separate **SSRN 4729284** single-stock study uses CRSP for a universe including delisted stocks and IQFeed for minute prices; it ranks eligible names by opening relative volume. Its headline portfolio includes shorting and allows leverage. A free, complete historical universe and execution tape have not been demonstrated here. The earlier categorical claim that Alpaca can never reconstruct the universe was stronger than the evidence; the defensible statement is that we have not established such coverage. [Full original stocks-in-play paper, §§2–3](https://concretumgroup.com/wp-content/uploads/2026/02/A-Profitable-Day-Trading-Strategy-For-The-U.S.-Equity-Market.pdf).

## 3. What the negative replications do and do not establish

**Paz Serpa, SSRN 7290621:** the author's paper confirms May 1, 2024–March 31, 2026 return 9.4% versus 29.5% SPY and Sharpe 0.39. This is the leveraged, long/short noise-band rule. The benchmark omits dividends; its start is post-original-sample, not strictly post-May-10 publication. Crucially, the published code mixes annualized Sharpe with a daily standard error. Its claim of overwhelmingly significant decay is not accepted. The arithmetic derivation and limitations are in the [source audit](2026-10-24-intraday-replication-source-audit.md#statistical-code-defect-sharpe-annualization-and-standard-error-use-different-units). [Author's full paper source](https://raw.githubusercontent.com/PazSheimy/spy-intraday-momentum-oos/main/paper/Out-of-Sample_Intraday_Momentum_SPY_PazSerpa_2026.tex).

**Delgado, SSRN 7323419:** only the indexed primary abstract was accessible. Its pooled Sharpe 0.35 conceals a strong initial period and a weak later period; the author explicitly rejects an immediate-decay interpretation. Full methods remain unresolved. [Primary abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7323419).

**García Arano's Substack:** the full article reports negative net results, but describes a 09:30–10:00 signal and misstates the original sample. Its prior-close implementation is not established. It cannot decisively refute our proposed signal. [Primary article](https://deadsignalslab.substack.com/p/replication-intraday-momentum-eight).

**Fetna futures ORB study, SSRN 7428398:** author report and registration text confirm zero of 225 cells pass its joint bar. That does **not** mean zero profitable cells: 39 have positive after-cost confirmation P&L and 138 are underpowered. The report acknowledges roll-day/session issues. This is a different market and protocol. [Author's full technical report](https://raw.githubusercontent.com/mulhamfetna/trading-strategy-finder/main/subprojects/Parametric-Indicators/shareable/orb_study_bundle/docs/WS-ORB-REPORT.md).

**Demirev, SSRN 5325376:** primary text remains inaccessible. The secondary-only QQQ replication figure has been excluded from the decision evidence. Access failures are documented, not interpreted as proof of fabrication.

**Corrected synthesis:** cost sensitivity and regime instability are credible concerns. “Every replication proves immediate decay” and “all ORB variants lose” are unsupported. No cited replication establishes profitability or failure for the owner's exact long-only, unleveraged rule.

## 4. Mechanism and publication-decay context

**Full primary text read:** Baltussen et al. (2021) links intraday momentum to short-gamma hedging across futures through 2020. Its main predictor is the return from previous close to 15:30, differing from Gao's morning-only predictor. It supports a possible mechanism, not the earlier claim that growing 0-DTE volume necessarily caused post-2013 decay. That causal claim and the reviewer's precise option-volume percentages are removed. [Author-hosted published paper](https://academicweb.nd.edu/~zda/intramom.pdf).

**Primary abstract read:** Rosa (2022) reports disappearance of unconditional OOS predictability and signal-strength regimes for an overnight-to-final-half-hour futures rule. Full methods were not accessible; it is related evidence, not a test of our exact SPY rule. [Publisher abstract and data statement](https://onlinelibrary.wiley.com/doi/10.1002/fut.22375).

McLean–Pontiff's published paper reports average post-publication decay across cross-sectional predictors. It provides a skeptical prior, not a numerical decay forecast for intraday SPY. [Published paper, mirrored full text](https://gwern.net/doc/economics/2016-mclean.pdf).

## 5. Data and execution feasibility

Alpaca's Basic plan lists history since 2016, 200 historical requests/minute and real-time IEX coverage. Its FAQ explicitly permits historical SIP queries when `end` is at least 15 minutes old. Documentation verifies the offering, not this account's actual entitlement or completeness. [Data plans](https://docs.alpaca.markets/us/docs/about-market-data-api), [data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq).

Minute bars are stamped at the interval's start. Some trade conditions affect minute and daily aggregates differently; missing minute bars can reflect absent qualifying trades. A 15:59 close need not equal the auction or daily close. The historical auctions endpoint exists and is SIP-only; its full-window coverage is untested. The unverified claim that IEX starts only in 2020 is removed. [Historical auctions](https://docs.alpaca.markets/us/reference/stockauctions-1).

**Execution correction:** Alpaca documents `cls` closing-auction orders and the 15:50 cutoff, but its fractional-order matrix does not support `cls`. Fractional whole-equity sizing and an auction exit cannot both be assumed executable. A later execution design must specify whole-share auction orders and residual cash, or separately register a fractional exit convention. The historical idealized screen is not forward-trading approval. [Order types and time-in-force matrix](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

One basis point per side remains an explicit policy assumption. Neither paper fills nor a quoted spread validate realized costs. Broker eligibility, partial fills, exit failures, latency and actual slippage remain empirical questions.

## 6. Tax and economic-hurdle corrections

The IRS taxes **net** short-term capital gains at ordinary graduated rates. The earlier phrase “taxes every winning day” was incorrect. Loss netting and carryforwards matter. SPY buy-and-hold can also owe current tax on distributions; a benchmark with no interim or terminal tax is a research diagnostic, not the owner's tax outcome. [IRS Topic 409](https://www.irs.gov/taxtopics/tc409).

Wash-sale rules can defer losses after substantially identical replacement purchases within 30 days before or after a loss sale, with basis adjustments. Repeated SPY trading and a concurrent SPY investment book require coordinated lot accounting. Being flat overnight does not exempt the activity. A daily bot does not automatically receive trader tax status or a section 475 election. [IRS Publication 550, Wash Sales](https://www.irs.gov/publications/p550), [IRS Topic 429](https://www.irs.gov/taxtopics/tc429).

The brief's annual 24% positive-P&L haircut is a **policy scenario**, not a personal tax estimate. Omitting loss carryforwards can be harsher, while omitting wash-sale deferrals, state tax and other liabilities can be optimistic. It is not uniformly conservative.

**Benchmark reproduced with a saved artifact:** Yahoo adjusted-close CAGR is 13.7421% for 2014-01-02 close to 2026-09-30 close, and 15.0624% for 2016-01-04 close to the same end. Formula, provider/version, exact daily inputs and hash are in [_intraday_benchmark_verification.json](_intraday_benchmark_verification.json). This is a provider-adjusted total-return proxy, not an independently reconstructed dividend ledger. These close-to-close conventions exclude the start session's return; the future strategy benchmark must align its timestamps.

The old SPY-plus-one-point gate is demanding, but “could never pass” was unjustified. The reviewer's 4.5 bp/day edge and 3.7 bp/day statistical floor are not adopted. The observed working-paper mean is a different historical long/short quantity. The proposed **+3% standalone after-policy CAGR** remains an owner policy decision; its value is not derived from a t ≥ 3 requirement, and its suitability for long-only is unproven.

## 7. Research decision and remaining gaps

The [long-only brief](briefs/2026-10-24-intraday-feasibility.md) remains **DRAFT r6**. Its candidate buys SPY at 15:30 after a positive prior-close-to-09:59 signal, exits at the close, and otherwise stays flat, with no borrowing. No return result exists against SPY or any acceptance gate. This remains the first candidate for simplicity and a single liquid symbol, not because the evidence proves it has the best expected payoff. Noise bands and equity ORB require separate long-only specifications before testing.

**Closed documentary gaps:** FINRA/Alpaca implementation, historical SIP policy, timestamp construction, auction endpoint existence, order-type incompatibility, IRS rules, original strategy identities, the misidentified SPY baseline, wrong QQQ citation, and the replication inference error.

**Still unresolved:** final Gao published-version details; Delgado/Demirev full methods; replication raw-data reproduction and registration chronology; credential-specific archive/auction coverage and quality; measured execution costs and mechanics; live-account provenance/settlement permissions; personal taxes and combined SPY lot effects; and the long-only rule's OOS edge.

Owner choices on the window, A1 gate meaning and A2 policy remain open. No verification finding silently approves those choices. Stage 0A is still pending, followed by separately approved price-quality work. **Do not proceed to bulk data, return tests or execution while the required design and data gates remain open.** A missing archive is UNRESOLVED (data); a later strategy gate failure is NO-GO for that candidate. Existing paper observation and the production freeze continue. Prior daily-bar NO-GO studies remain closed; see [current worklist](../next_steps.md) and [research roster](RESEARCH_SNAPSHOT.md).
