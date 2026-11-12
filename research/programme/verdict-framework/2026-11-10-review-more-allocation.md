# Allocation and cross-asset verdict annotations

Reviewed source-linked annotations for 9 inventory entries, reflected in the canonical inventory and register. Identities use the Lens 2 classification IDs where no independently registered specification/stage identity was available.

## `lens2-stocks-core-spy-blend-keep` — KEEP

- **Identity:** study `core-spy-blend`; specification `stocks-core-spy-blend-keep`; stage `historical-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** Historical scope is the roughly 8% stock exposure tested in the 2026-09-24 construction. Sharpe 0.80 vs 0.78 and equal 12.96% CAGR; report says live size is unvalidated. Lens 2 finds DSR-unit contamination of unknown direction, code/spec mismatch and no MDE. Preserve KEEP only for tested construction; inference and transfer to live sizing are unresolved.
- **Sources:** `research/allocation/core-spy-blend/2026-09-24-core-spy-blend-keep.md`; `research/allocation/core-spy-blend/_core_spy_blend_20260924.json`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-stocks-equity-diversification-sleeves` — failed

- **Identity:** study `equity-diversification-sleeves`; specification `stocks-equity-diversification-sleeves`; stage `historical-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** Lens 2 records SPY duplicates, anchor leak, PIT-universe contamination, no raw artifact, no brief, prose-only recomputation and no power estimate. Contamination includes strategy-favoring components; no valid return inference or protocol determination can be recovered. Preserve “failed” as historical wording, without converting it into evidence against a valid registered benefit.
- **Sources:** `docs/roadmap.md`; `research/allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve-brief.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-stocks-live-revert-blend-to-core` — Underperforms; Revert Queued

- **Identity:** study `live-revert-blend-to-core`; specification `stocks-live-revert-blend-to-core`; stage `historical-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** Lens 2 finds PIT-universe and DSR-unit contamination (direction mixed/unknown), no found preregistration and no measurable power; Lens 6 code match is affirmative. Preserve the documented underperformance/revert-queued record, but contaminated inputs and unknown inferential precision leave performance determination unresolved.
- **Sources:** `research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`; `research/audits/root-cause-2026-11-04/lenses/_lens1_tape_state.json`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-stocks-position-sizing-regimes` — Inconclusive

- **Identity:** study `position-sizing-regimes`; specification `stocks-position-sizing-regimes`; stage `historical-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** The report headline is a single ungated window, not the gated WFO represented in the JSON; Lens 7 documents that mismatch and that the gated results were not incorporated into the report. Lens 2 also identifies PIT-universe and DSR-unit contamination and no t-statistic. Preserve the original Inconclusive verdict and its limited scope; neither the headline nor the separate JSON supports a generalized sizing conclusion.
- **Sources:** `research/paper-system/position-sizing/2026-08-19-position-sizing-regimes.md`; `research/paper-system/position-sizing/_position_sizing_wfo_results.json`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens7-research-accuracy.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-stocks-three-sleeve-blend-1-14` — GO

- **Identity:** study `three-sleeve-blend`; specification `stocks-three-sleeve-blend-1-14`; stage `historical-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** No raw artifact; Lens 2 finds SPY duplicates, anchor leak, PIT-universe contamination, DSR units, code mismatch and prose-only recomputation; Lens 7 finds no brief. The reported 1.14 result cannot support a valid determination. Preserve GO as historical wording; the later reproduction is a distinct record.
- **Sources:** `docs/roadmap.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens7-research-accuracy.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-cross-asset-fomc-drift-corrected-harness` — NO-GO stands

- **Identity:** study `fomc-drift`; specification `cross-asset-fomc-drift-corrected-harness`; stage `registered-net-event-and-wfe-gates`.
- **Evidence:** INCONCLUSIVE; **protocol:** NOT MET; **action:** CLOSE THIS SPECIFICATION.
- **Rationale:** The registered corrected-harness rule requires net event t >= 3.0 and WFE >= 0.50. The clustered artifact reports net t=0.981, and the separate audit reruns artifact reports WFE=0.244, so both registered conditions fail and the original NO-GO stands for this corrected event-study specification. Lens 6 did not examine code, so the evidence label remains INCONCLUSIVE and the result does not generalize to cross-asset allocation. The prior MDE comparison is removed.
- **Sources:** `research/audits/production-replica/2026-10-19-production-replica-audit.md`; `research/audits/production-replica/2026-10-19-production-replica-fomc-timing-amendment.md`; `research/audits/production-replica/2026-10-19-production-replica-audit-nogo.md`; `research/audits/production-replica/_fomc_clustered_event_study_20261020.json`; `research/audits/production-replica/_audit_reruns_20261019.json`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-cross-asset-fomc-drift-wfo` — NO-GO

- **Identity:** study `fomc-drift`; specification `cross-asset-fomc-drift-wfo`; stage `historical-wfo-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** Lens 2 finds day shift, SPY duplicates, anchor leak, DSR units and pre-entry event returns; no raw artifact or found brief and recomputation is prose-only. Some known issues favor the strategy; historical NO-GO cannot be treated as a valid registered determination. Preserve its exact verdict and mark evidence/protocol unresolved.
- **Sources:** `research/allocation/fomc-drift/2026-07-20-fomc-drift-nogo.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens1-backtest-validity.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-cross-asset-month-end-treasury` — NO-GO

- **Identity:** study `month-end-treasury`; specification `cross-asset-month-end-treasury`; stage `registered-five-criterion-study`.
- **Evidence:** UNRESOLVED; **protocol:** NOT MET; **action:** CLOSE THIS SPECIFICATION.
- **Rationale:** The registered five-criterion study fails its WFO and portfolio criteria. Independently, the portfolio arm reports Sharpe 0.909 versus SPY 0.896 but MaxDD -24.9% versus -24.5%, so the registered requirement that drawdown be no worse fails. The artifact records the NO-GO checks; preserve protocol NOT MET. Lens 2 flags DSR-unit contamination of unknown direction, so edge inference remains UNRESOLVED.
- **Sources:** `research/allocation/month-end-treasury/2026-09-24-month-end-treasury-sleeve.md`; `research/allocation/month-end-treasury/2026-09-24-month-end-treasury-sleeve-nogo.md`; `research/allocation/month-end-treasury/_month_end_treasury_results.json`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.

## `lens2-cross-asset-treasury-curve` — NO-GO

- **Identity:** study `treasury-curve`; specification `cross-asset-treasury-curve`; stage `historical-verdict-classification`.
- **Evidence:** UNRESOLVED; **protocol:** UNRESOLVED; **action:** HOLD / PARK THIS VERSION.
- **Rationale:** Lens 2 reports no raw artifact, no found pre-registered brief, SPY duplicates/anchor leak/DSR contamination, and prose-only recomputation. With inputs and registered decision rule unavailable, validity and determination cannot be established. Preserve NO-GO as historical disposition only.
- **Sources:** `research/allocation/treasury-curve/2026-07-20-treasury-curve-nogo.md`; `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`; `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`.


Independent Sol reviewed and rechecked this batch after corrections; final FOMC WFE is attributed to the separate audit reruns artifact. Review was same-family; no returns were computed.
