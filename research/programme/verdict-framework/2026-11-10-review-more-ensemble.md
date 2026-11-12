# Supplemental verdict annotations: ensemble lineage

**Date:** 2026-11-10  
**Scope:** seven `lens2-stocks-ensemble-*` entries, the pre-PIT IC/Kelly rebaseline, and the five-voter ablation.  
**Interpretation:** Original decisions and Lens 2 classifications are preserved. Evidence is marked `UNRESOLVED` where contamination or source limitations prevent a clean inference. Unknown power alone is not treated as a fault. Actions are record-level dispositions and do not authorize trading or deployment.

## `lens2-stocks-ensemble-conviction`

- Original verdict: **Soft NO-GO**
- Lens 2 classification: **inconclusive**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'Soft NO-GO' (inconclusive); this supplemental record preserves that historical decision without granting authorization. The source roster only preserves snapshot/lineage claims, with no primary run report or raw artifact available in this checkout; the source audit additionally records duplicate SPY rows (direction unclear), anchor leakage and non-PIT membership (both favor strategy), and a DSR-unit fault (direction unknown). Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/RESEARCH_SNAPSHOT.md`

## `lens2-stocks-ensemble-ic-2026-06-28`

- Original verdict: **NO-GO**
- Lens 2 classification: **unreliable NO (flawed test)**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'NO-GO' (unreliable NO (flawed test)); this supplemental record preserves that historical decision without granting authorization. The June result is compared with the later-voided 1.12 baseline and Lens 6 found a code/spec mismatch. The source audit records duplicate SPY rows (direction unclear), anchor leakage and non-PIT membership (favor strategy), and DSR-unit contamination (direction unknown); the inventory records no raw run artifact; the available historical result summary is in the relocated initial-research-plan report. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/programme/initial-research-plan/2026-06-28-alpha-optimization-execution-plan.md`, `research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`

## `lens2-stocks-ensemble-ic-pit`

- Original verdict: **closed NO-GO**
- Lens 2 classification: **inconclusive**
- Evidence: **UNRESOLVED**; action: **CLOSE THIS SPECIFICATION**.
- Rationale: Lens 2 records 'closed NO-GO' (inconclusive); this supplemental record preserves that historical decision without granting authorization. This PIT run is closed NO-GO in the original source. Lens 1 identifies DSR-unit contamination; its effect direction is unknown. Lens 9 could not compute the requested MDE because the stored artifact lacks SPY CAGR and t-statistic; unknown power is not itself a validity fault. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/audits/rebaseline/2026-09-23-pit-rebaseline-core-nogo.md`, `research/audits/rebaseline/_pit_rebaseline_20260923.json`, `research/RESEARCH_SNAPSHOT.md`

## `lens2-stocks-ensemble-kelly-2026-07`

- Original verdict: **NO-GO**
- Lens 2 classification: **inconclusive**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'NO-GO' (inconclusive); this supplemental record preserves that historical decision without granting authorization. The July finding is preserved as a historical NO-GO, but it was compared with the later-voided 1.12 baseline, has no raw run artifact in the source record, and Lens 6 found a code/spec mismatch. Tape contamination includes anchor leakage and non-PIT membership (favor strategy), duplicate SPY rows (unclear), and DSR units (unknown). Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/audits/comprehensive-review/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`, `docs/superpowers/specs/2026-07-06-kelly-position-sizing-design.md`

## `lens2-stocks-ensemble-kelly-pit`

- Original verdict: **closed NO-GO**
- Lens 2 classification: **inconclusive**
- Evidence: **UNRESOLVED**; action: **CLOSE THIS SPECIFICATION**.
- Rationale: Lens 2 records 'closed NO-GO' (inconclusive); this supplemental record preserves that historical decision without granting authorization. This PIT run is closed NO-GO in the original source. Lens 1 identifies DSR-unit contamination; its effect direction is unknown. Lens 9 could not compute the requested MDE because the stored artifact lacks SPY CAGR and t-statistic; unknown power is not itself a validity fault. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/audits/rebaseline/2026-09-23-pit-rebaseline-core-nogo.md`, `research/audits/rebaseline/_pit_rebaseline_20260923.json`, `research/RESEARCH_SNAPSHOT.md`

## `lens2-stocks-ensemble-ndh-baseline-1-12`

- Original verdict: **Validated**
- Lens 2 classification: **unreliable GO**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'Validated' (unreliable GO); this supplemental record preserves that historical decision without granting authorization. The headline 1.12 Sharpe did not reproduce and its original evaluation window is unrecoverable. Lens 1 records duplicate SPY rows (direction unclear), anchor leakage and non-PIT membership (favor strategy), plus DSR-unit contamination (direction unknown); Lens 6 found code/spec mismatch. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/RESEARCH_SNAPSHOT.md`, `docs/changelog.md`, `docs/roadmap.md`

## `lens2-stocks-ensemble-wfo-validated-2026-06-20`

- Original verdict: **Validated**
- Lens 2 classification: **unreliable GO**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'Validated' (unreliable GO); this supplemental record preserves that historical decision without granting authorization. This early GO is preserved as a historical verdict, but the evidence is contaminated: duplicate SPY rows (direction unclear), anchor leakage and non-PIT membership (favor strategy), and DSR-unit fault (direction unknown). Lens 6 found code/spec mismatch; no raw artifact is available. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/RESEARCH_SNAPSHOT.md`, `docs/changelog.md`, `docs/archive/roadmap-pre-lab.md`

## `lens2-stocks-ic-kelly-rebaseline-pre-pit`

- Original verdict: **invalid**
- Lens 2 classification: **inconclusive**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'invalid' (inconclusive); this supplemental record preserves that historical decision without granting authorization. The original report calls this run invalid and says neither variant is closed. The WFO omitted point-in-time membership; Kelly’s headline CAGR is driven by a post-delisting SIVB bar ($0.0013 to $0.11). The corresponding IC signature was presumed to be the same artifact but was not measured per day. DSR-unit fault direction is also unknown; the original report and raw artifact are available at their relocated research/audits/rebaseline paths. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `research/audits/rebaseline/2026-09-23-ic-kelly-rebaseline-invalid.md`, `research/audits/rebaseline/_ic_kelly_rebaseline_20260923.json`, `docs/superpowers/specs/2026-06-28-ensemble-ic-weighted-voting-design.md`, `docs/superpowers/specs/2026-07-06-kelly-position-sizing-design.md`

## `lens2-stocks-voter-ablation-5-voter`

- Original verdict: **Best = core+macd+vbb (5-voter, drop MTF)**
- Lens 2 classification: **unreliable GO**
- Evidence: **UNRESOLVED**; action: **HOLD / PARK THIS VERSION**.
- Rationale: Lens 2 records 'Best = core+macd+vbb (5-voter, drop MTF)' (unreliable GO); this supplemental record preserves that historical decision without granting authorization. The recorded best configuration is preserved as a historical GO, but Lens 1 identifies duplicate SPY rows (direction unclear), anchor leakage and non-PIT membership (favor strategy), and DSR-unit contamination (direction unknown); Lens 6 found code/spec mismatch. The inventory points to an external memory report and records no raw artifact; the available in-repository snapshot and changelog provide historical corroboration, with primary traceability limited. Evidence remains UNRESOLVED because the reported contamination/code/source limitations do not establish a clean result. Unknown power alone is not treated as a fault.
- Sources: `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens2-verdict-audit.md`, `research/audits/root-cause-2026-11-04/lenses/_lens2_sensitivity.json`, `research/audits/root-cause-2026-11-04/records/_investigation_inventory.json`, `docs/project_snapshot_2026-06-28.md`, `docs/changelog.md`, `research/RESEARCH_SNAPSHOT.md`, `research/audits/root-cause-2026-11-04/lenses/2026-11-04-lens6-implementation-correctness.md`

Independent Sol review and recheck identified and corrected source lineage, data-fault and scope issues. Same-family review; no returns were calculated.
