# Proposed stock-screen verdict batch — 2026-11-10

These are reviewed annotations for three Lens-2 inventory rows. **Protocol: NOT MET** means the stated advancement gates failed, not that the tests were necessarily improperly conducted. **Evidence: INCONCLUSIVE** for factor and net issuance reflects unresolved power; momentum is **UNRESOLVED** for unverified prior registration of its prospective claim; it is not a reversal of the original **NO-GO**. **Action: CLOSE THIS SPECIFICATION** preserves each original decision, with no approval for additional research inferred. No result establishes that all smaller benefits are absent.

| Inventory row | Registered result | Principal qualification |
| --- | --- | --- |
| `lens2-stocks-momentum-french-screen` | BIG and SMALL each miss Holm A1; smallest p = 0.000941902 versus first threshold 0.000675. A2 is +0.081863 / +0.197551 pp/yr versus +1.0, 1994-01–2026-04. NO-GO. | Brief and results first appear in the same commit; the audit calls prior registration unverified. Power not established from artifact. |
| `lens2-stocks-factor-screen-h1-h4` | All fail A1 and A2. H4 first Holm p = 0.090310 versus 0.0003375; A2 H1 +0.826675, H2 −0.620700, H3 −0.000702, H4 −0.559214 pp/yr, all below +1.0. NO-GO. | Power not established from artifact; factor development overlaps historical data. |
| `lens2-stocks-net-issuance-screen` | BIG NegNI fails A1 (t = 0.26, p = 0.398; required t ≥ 3, p ≤ 0.00135) and A2 (+0.48 versus +1.0 pp/yr). NO-GO. | Power not established from artifact; other issuance cells are diagnostics, not substitutes for H1. |

Independent Sol review checked the cited figures and recommended the momentum timing qualification; the lead confirmed the original reports. Same-family review. No returns were recomputed.

All compare French value-weighted portfolios with French market total return, not SPY, and the fixed haircuts are assumptions, not measured execution or taxes. None evaluates an exact point-in-time S&P 500 implementation. Lens 9 cannot compute the effect-size minimum detectable from these artifacts because annualized mean excess is missing (momentum also lacks its month counts in the artifact); this is uncertainty, not a finding of zero power. The JSON cites each original brief, report, raw results, and the relevant audit records. Statistical thresholds and diagnostics have not been revised retrospectively.
