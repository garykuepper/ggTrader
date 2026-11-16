# r9 amendment: no GO/NO-GO label

2026-11-15, written after the single r9 score. It changes no number, threshold, rule or data.

## What changed

- **The label is gone.** r9 copied r8's wording that a score turns into "GO" or "NO-GO". The owner
  had already retired new GO/NO-GO determinations
  ([research guide](../../programme/RESEARCH_GUIDE.md), "Decisions"). The lead missed this when
  drafting r9, and the owner pointed it out after the score.
- **How the result is reported instead.**
  - The two preset thresholds stay exactly as registered.
  - The result states whether each threshold was met, the evidence and its uncertainty.
  - Any action is a separate owner decision, not part of the study.
- **The scorer's output is left as written.** Its write-once files
  (`_intraday_momentum_r9_results.json` and `.md`) print the word "NO-GO" because the code
  followed r9's text. They are hash-bound and not edited. Read that word as "neither preset
  threshold was met".
- **The r9 protocol file is left as is.** It is bound by hash
  (`222521ce…593ed3`) to the code and the score, and this note is its correction.

## Result in the agreed form

| Preset threshold | Registered bar | Measured | Met? |
|---|---|---|---|
| A1: statistical test on the average daily result after base costs (Newey-West t, 5 lags, one-sided) | t of 3.0 or more | t = −3.48 | No |
| A2: yearly growth after the illustrative 24% tax ledger | +3.0% a year or more | −3.68% a year | No |

Neither threshold was met. The measured average is below zero, not just short of the bar.
