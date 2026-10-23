# One insider predecessor candidate: reviewed acquisition handoff

Owner instructed **Execute** on the connected-metadata plan. Task 1's read-only
query narrowed the 14 original-submission-date candidates to one stored issuer /
owner / filing-date match: `0001225208-18-003052`, CIK 1133421, owner 1539147,
filed 2018-02-15. It also occurs in the accepted cached-inventory candidate list.
This is identification evidence, not a proved original/amendment lot linkage.

The single frozen URL is
`https://www.sec.gov/Archives/edgar/data/1133421/000122520818003052/0001225208-18-003052.txt`.
The [request JSON](../_insider_predecessor_acquisition.json) binds the accepted
workbook and newly recorded DB query snapshot. No other candidate or primary
XML fallback is included. If the full response lacks usable XML/header evidence,
retain it and the unresolved linkage rather than silently expanding collection.

Runner `/tmp/insider_predecessor_batch.py` adapts the reviewed two-charter runner:
parent is `data/external/event_timing_audit/cost-charters-h1kbl362/`, with 81 SEC,
zero Yahoo and 105.01228246744722 seconds. One nominal / four maximum attempts
projects to 82 nominal / 85 maximum cumulative. The global six-hour deadline is
inherited, not estimated afresh. Default dry run and help issue no HTTP and leave
parents unchanged. Ruff passes. An initial dry run caught an incorrect derived
ceiling assertion; it was corrected to 85 before review or any request.

Before HTTP, fresh scoped review must clear URL/source bindings, the runner and
authority. Claim `insider-predecessor-20261022-claim.json` is exclusive and persists
on interruption. The runner copies immutable raw files and cumulative checkpoint
into a distinct child, records parent metadata/all-file hashes, code/spec binding
and checkpoints responses. It validates every response entry's own path/hash,
including earlier retry bodies, and the unchanged first 81 reservations.

Prior documentation 403 and consumed claims remain recorded. Do not retry the
blocked URL or reset budgets. First new 403/429 stops all calls; max five SEC
requests/sec, initial plus three counted retries, no redirects or automatic
restart. Contact remains process-only; no `.env`, email or secret output.

After collection, verify the issuer/owner, accession, raw header dates and lots
against amendment `0001225208-18-004211`. Identify exact replaced lots only when
raw statements support them; retain the amendment's own public eligibility time.
Never backdate a correction or convert unknown linkage into standalone usable
lots. Output `/tmp/insider-predecessor-batch-result.json` and source-bound evidence
in the handoff. No return test, detector run, population expansion, DB write or
production change is authorized. The wider audit remains unfinished.
