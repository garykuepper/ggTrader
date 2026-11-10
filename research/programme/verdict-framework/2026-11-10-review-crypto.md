# Proposed annotations: remaining non-stock/non-allocation inventory

**Batch:** five crypto records, plus one status-only exclusion  
**Date:** 2026-11-10  
**Scope:** Five reviewed crypto records and one status-only exclusion. The canonical inventory and register now reflect this review; it authorizes no rerun or revision of original verdicts.

## Review notes

### `lens2-crypto-edge-search-2026-06-08`

- **Original verdict:** no deployable edge
- **Protocol result:** UNRESOLVED
- **Proposed evidence/action:** UNRESOLVED / CLOSE THIS SPECIFICATION
- **Original source:** [docs/archive/edge_search_report_2026-06-08.md](../../../docs/archive/edge_search_report_2026-06-08.md)
- **Audit:** [Lens 2 verdict audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md); [Lens 8 crypto audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Rationale:** The source reports the top-11 Binance.US run had zero full-cascade passers; the all-51 Binance.US run had two marginal passers and +13.8% CAGR versus BTC +38.8%; and the Kraken top-100 run returned -10.0%. The report says no passer had a clean, trade-rich positive holdout and closes the cheap gate, universe, and venue/fee levers. Preserve the original “no deployable edge” verdict and this bounded scope. Lens 8 says the Binance.US tape includes 44% Kraken copies, flags survivorship, and notes the headline venue flip is confounded by universe, tape, selected coins, and fees. The underlying raw results are absent from the repository and Lens 2 reports data unknown, no raw artifact, and no pre-registered brief. These prevent independent validation of the inference and registered protocol result; they do not establish that every crypto strategy lacks an edge.
- **Uncertainty:** Reported values are sourced to the 2026-06-08 report and Lens 8 reanalysis; no raw results were independently recomputed. Binance.US copied-bar provenance, survivorship and confounded venue comparison limit causal interpretation. The source tested long-only spot WFO in the listed universes and venues; it did not test directional shorting, broader crypto strategies or a longer history.

### `lens2-crypto-funding-carry-phase4`

- **Original verdict:** directionally credible but not numerically trustworthy
- **Protocol result:** UNRESOLVED
- **Proposed evidence/action:** UNRESOLVED / HOLD / PARK THIS VERSION
- **Original source:** [docs/archive/phase4_funding_carry.md](../../../docs/archive/phase4_funding_carry.md)
- **Audit:** [Lens 2 verdict audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md); [Lens 8 crypto audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Rationale:** The Phase 4 report describes a delta-neutral long-spot/short-PF_XBTUSD funding-carry test. Its real-funding run reports +0.45%, Sharpe 4.44 and 8 trades over 2025-05-15–2026-05-15; the report itself limits the basis proxy as not quotable and notes zero trade-event overlap between the real-funding and proxy runs. Lens 8 confirms the results are prose-only, with inputs and legacy code noted as existing, but no raw result artifact is available; Lens 2 found no brief or registered bar. Preserve the original cautious verdict. Numerical evidence and the registered protocol determination cannot be independently checked without a raw result and a registered brief; unknown power alone is not the reason for UNRESOLVED.
- **Uncertainty:** No results artifact or pre-registration was located. The report’s input dataset and legacy code are described in the source and Lens 8, but this annotation does not access the database or reconstruct the run. One year of real funding data and eight trades are thin; the basis proxy differs in trade timing and is not a valid substitute.

### `lens2-crypto-ibit-overnight-session`

- **Original verdict:** closed in practice — do not build
- **Protocol result:** UNRESOLVED
- **Proposed evidence/action:** DESCRIPTIVE / CLOSE THIS SPECIFICATION
- **Original source:** [research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json](../../../research/crypto/artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json)
- **Audit:** [Lens 2 verdict audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md); [Lens 8 crypto audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Rationale:** The original finding is a descriptive split of IBIT ETF session returns, not a registered strategy test. The JSON artifact reports separate intraday, weeknight and weekend statistics; Lens 2 notes no benchmark series, no registered bar, and that `overnight_all` lacks a t-statistic, so it selected the three available sub-rows for partial recomputation. Lens 8 says the crypto Sharpe annualization issue does not affect these total-return results. Preserve the source’s “closed in practice — do not build” action. Because there is no registered decision rule and the artifact’s selected statistic does not match a single declared verdict arm, protocol determination is unresolved; the descriptive split does not establish a deployable edge.
- **Uncertainty:** This artifact concerns IBIT (a US-listed ETF), despite its crypto inventory grouping. It tests absolute session returns, not excess returns versus a benchmark. The audit’s iid t-statistics and post-hoc selection are descriptive; the window, session counts and means do not resolve whether the pattern persists or is tradeable.

### `lens2-crypto-kraken-live-bot-pnl`

- **Original verdict:** modestly profitable
- **Protocol result:** UNRESOLVED
- **Proposed evidence/action:** UNRESOLVED / HOLD / PARK THIS VERSION
- **Original source:** [research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Audit:** [Lens 2 verdict audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md); [Lens 8 crypto audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Rationale:** Lens 8 traces the claim to +$9.88 realized over about five Kraken trading days and notes an approximately $100 mid-window capital injection, with Kraken-era trade records and the investigation script missing. Its live evidence table has zero rows in the `trades` and `equity_curves` tables and identifies no checkable performance artifact. Lens 2 records no raw artifact, unverified code, and no registered brief. Preserve the original “modestly profitable” claim as the historical verdict while marking evidence and protocol unresolved; the dollar gain cannot be verified or interpreted as a return without capital-flow and trade records. The old engine’s removal is recorded history, not an action recommendation about the project.
- **Uncertainty:** Trade records, the source P&L script, starting/ending capital and full external cash-flow-adjusted return are unavailable. The capital injection makes the dollar figure unsuitable as a performance rate; no inference about profitability follows.

### `lens2-crypto-wfo-textbook-reset`

- **Original verdict:** closed
- **Protocol result:** UNRESOLVED
- **Proposed evidence/action:** UNRESOLVED / CLOSE THIS SPECIFICATION
- **Original source:** [docs/superpowers/closures/2026-05-12-wfo-textbook-reset-and-venue-migration.md](../../../docs/superpowers/closures/2026-05-12-wfo-textbook-reset-and-venue-migration.md)
- **Audit:** [Lens 2 verdict audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md); [Lens 8 crypto audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens8-crypto.md)
- **Rationale:** The closure report records a same-universe, same-grid fee sweep with 36, 14, 13 and 0 aggregate-gate passers from frictionless through Kraken taker costs, and states that zero coins formed a portfolio in every variant. It also distinguishes the separate venue decision that selected 22 BTC/ETH/DOGE combos and deferred 14 TRX combos. Lens 8 found the inventory’s cited raw artifact points to the frictionless run rather than the realistic-fee run; the latter’s gate count is reconstructed from worker logs, and no portfolio formed even at zero cost. Lens 2 reports no registered brief, no excess-vs-benchmark statistic, and unverified code. Preserve the original closed result and the narrow WFO/fee-sweep scope. The mispointed artifact, incomplete durable evidence and missing registered brief prevent an independent determination of the affected protocol and portfolio claims.
- **Uncertainty:** The closure report is detailed, but the realistic-fee raw snapshot was not preserved at the cited location; Lens 8 says the stated zero-pass count is available in worker logs. Do not conflate the aggregate gate-pass counts with deployable portfolios or with the separate venue/migration decision. Crypto data provenance and survivorship concerns remain under Lens 8; no new computation was done.

### Prompt-kit review: excluded status-only item

The [prompt-kit review](../prompt-kit-review/2026-11-05-research-prompt-kit-review.md) is a recommendation about research prompts, not a strategy verdict. Its adoption status remains in that source and it is not counted as a test.

Independent Sol review checked figures, clarified that the funding-carry caution applies to basis-proxy B, and identified prompt-kit as status-only rather than a strategy verdict. Same-family review; no returns were calculated.
