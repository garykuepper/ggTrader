# Family-Scope Review: do the closed stock-screen NO-GOs apply to the pending feasibility rows?

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** Documentary review of **definitions only**. It uses no returns, Sharpe ratios, correlations or result files. It changes no recorded verdict, recommends no candidate, and authorizes no test. Machine-readable fields: `family_scope_review` on rows S02–S08 and S13 of [`_free_data_feasibility_matrix.json`](_free_data_feasibility_matrix.json). Extracted definitions: [`_family_scope_extraction.json`](_family_scope_extraction.json).

Terms: **NO-GO** = a specified test failed its registered bar. **Part A** = the long free-history test in the [forward-paper standard](briefs/2026-10-23-forward-paper-standard.md). **Sort variable** = the characteristic used to form portfolios. **Sibling construction** = a portfolio built on a different variable that targets the same economic channel.

## 1. Question and rule

[`RESEARCH_INDEX.md`](RESEARCH_INDEX.md) says a NO-GO "applies to that test and its data, not automatically to every related idea." Eight feasibility rows overlap closed families. For each, did the closed report's registered hypotheses or scope statement plausibly cover it?

Outcome classes, fixed before the reviews:

- `closure_applies_inferred`: same sort variable or portfolio as a closed registered hypothesis, a combination or near-identical construction of it, or covered by the closed report's own scope statement.
- `distinct_mechanism`: different sort variable **and** economic channel, neither a combination nor a sibling construction.
- `undetermined`: anything else, or insufficient definitions. A shared theme alone never yields `closure_applies_inferred`.

Reconciliation rule: if the two independent reviewers disagree, the outcome is `undetermined`.

## 2. Method

1. DeepSeek Flash extracted the registered definitions and scope statements of the three closed screens (momentum, the four-factor screen, net issuance) with verbatim quotes. All 27 quotes were machine-checked against their cited lines.
2. Codex Sol and Claude Sonnet each classified the eight rows independently from the definitions. Neither used returns or result files.
3. The evidence quotes cited below were machine-checked against the files; all matched.

## 3. Results

| Row | Family | Sol | Sonnet | Outcome |
|---|---|---|---|---|
| S02 | Long-term reversal | distinct_mechanism | undetermined | **undetermined** (disagreement) |
| S03 | Low beta / variance / residual variance | undetermined | undetermined | **undetermined** |
| S04 | Accruals | distinct_mechanism | distinct_mechanism | **distinct_mechanism** (caveat below) |
| S05 | Valuation variants (E/P, CF/P, D/P) | undetermined | undetermined | **undetermined** |
| S06 | Industry momentum | undetermined | undetermined | **undetermined** |
| S07 | Quality (QMJ) | undetermined | undetermined | **undetermined** |
| S08 | q-factor ROE / expected growth | undetermined | undetermined | **undetermined** |
| S13 | Factor momentum | undetermined | undetermined | **undetermined** |

No row came out `closure_applies_inferred`. The rationale and verified evidence for each row are in its `family_scope_review` matrix field.

## 4. Observations that matter

- **S04's `distinct_mechanism` is conditional.** The project files give no accruals construction beyond the row's mechanism text, so the call rests on the sort variable and channel differing from every registered hypothesis (operating profitability, book-to-market, asset growth, prior 12–2 month return, net share issues). It does not show that an accruals screen would pass or that its data are available point-in-time.
- **The project's own net-issuance brief leaned toward overlap for S05 and S06.** It excluded industry momentum and valuation siblings from its scope, and says that exclusion "is a scope choice, not a new measured NO-GO." That is a project judgment, not a registered scope statement, so it was recorded as evidence and not treated as closure.
- **Seven of eight rows are `undetermined`, mostly because their construction is not defined in the project files.** What would settle each (definitions needed, not results): the intended low-risk sort and portfolio (S03); each valuation portfolio's numerator and thresholds (S05); industry lookback, ranking rule and weighting (S06); the QMJ component weights and selected decile (S07); the construction of the listed q-factor benchmark portfolios (S08); the underlying portfolios and allocation rule (S13); and for S02 whether the 13–60 month loser sort is treated as a sibling of book-to-market.

## 5. What this review does not do

- It does not lift or extend any recorded NO-GO, and it does not decide that any row should be tested.
- It does not use evidence from results. A related theme can still be economically linked to a closed result; the review only asks whether the registered definitions cover it.
- It is not an error-control argument: choosing among rows after seeing closed results has no cross-study error control, as the standard states.

## 6. Options for the direction decision (G3), none recommended

These are choices the owner can make; this review does not rank them.

1. Treat S04 as eligible for Part A registration on its own merits, subject to the point-in-time data questions already listed in the [feasibility inventory](2026-11-02-free-data-stock-feasibility.md).
2. Commission definitional work to settle one or more `undetermined` rows (the definitions listed in §4), without running any test.
3. Leave all rows as they are and take no new stock study.
