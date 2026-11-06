# Intraday r8 data-gap investigation

Owner accepted all three planning decisions: investigate before repair/scoring,
evaluate free authoritative alternatives before use, and cap targeted HTTP GET
attempts at 30 including pages and retries. This is investigation authorization,
not permission to amend r8, replace inputs or score.

## Tasks and ownership

1. Sonnet: static collector/pagination diagnosis. Completed: no demonstrated
   collector defect; vendor comparison required. Its interpretation of the
   dividend coverage gate is provisional pending Opus review.
2. Parent: authenticated Alpaca comparisons. Completed seven GETs, no retries:
   exact August 2019 month (two pages), exact corporate actions (one page),
   narrow affected day, two missing dividend months, one present control month.
   Compare hashes, counts and timestamps only; compute no signals or returns.
   Raw evidence: `../_intraday_r8_gap_investigation.json`.
3. Flash: extract SPY identity, dates and dividend amounts from the specified
   downloaded State Street workbook only. No browsing, credentials, code changes
   or inference. Parent independently checks the extracted records.
4. Opus: independent protocol/source compatibility assessment. Distinguish
   performance gates, mandatory diagnostic readiness and implementation-added
   coverage checks. Give a concrete remediation proposal and approval boundary;
   do not implement it or downgrade missingness rules.
5. Parent: synthesize verified findings into one report with JSON evidence and
   update the next work item. Stop before remediation or performance analysis.

## Source and request controls

Parent alone handles credentials and network requests. Children receive local
files; no child network access is authorized. Existing dataset, manifest, protocol,
validation and score claims are preserved. No orders or production changes.

Seven Alpaca GET attempts plus one direct issuer workbook GET have completed.
Public discovery used one search, two page reads and one workbook click (the
click failed on unsupported XLSX content type). Conservatively reserve five
additional slots for these discovery operations: 13 of 30 slots allocated.
Search-service internal requests are not observable; no exact network-level
count is claimed for that service. No further network access is planned.

## Completion criteria

- Reproduce or explain each reported absence without choosing strategy outcomes.
- Verify issuer evidence and distinguish vendor omission from absent distribution.
- Explain which remediation would require a protocol or source amendment.
- Preserve unresolved findings; do not call a data failure a strategy NO-GO.
- Record delegate identifiers and results in the final investigation report.


## Completion

All five tasks completed. Final findings and delegate IDs are in
[the investigation report](2026-10-25-intraday-gap-investigation.md).
Ten Alpaca GETs and one workbook GET, no retries, plus seven reserved public
lookup slots: 18 of 30 allocated. No repair, scoring or further request planned.
