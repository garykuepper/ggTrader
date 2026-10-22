# Limited restart: three missing operational documents

**Authorization:** The owner instructed “Continue” on 2026-10-22 after being
told that the next bounded collection required an explicit restart decision.
This is interpreted as authorization for this three-document batch only. It
does not reopen the whole audit, documentary search, or a repeated restart.
Execution waits for the scoped input/accounting review below.

**Execution complete:** Independent review cleared the plan and runner with
zero blockers (`/tmp/insider-three-document-restart-review.md`). The parent then
executed this single authorized batch: all three documents returned HTTP 200
in three attempts, extending the collection ledger from 50 to 53. No new
403/429 occurred. This authorization is consumed; it permits no further requests.
See the implementation handoff for the offline corroboration results.

## Reason and exact scope

The reviewed offline date-only amendment can corroborate 21 cached filing-date
pairs. Two in-window filing records still need three bodies:

1. Northrop `0001628280-25-006862`: full submission/header; its XML is cached.
2. Costco `0000909832-26-000039`: ownership XML and full submission/header.

The runner contains the three exact canonical SEC URLs. Group order uses the
registered accession SHA-256 ordering, with XML before its header inside a group.
These are operational examples, not a replacement or expansion of A9 sampling.
The selected Northrop record after the cutoff remains recorded and unfetched.
Northrop's inventory discrepancy and other evidence gaps remain unresolved.

## Single restart and stop history

The earlier documentation search reported HTTP 403; its stop remains part of
the record. This explicit new instruction permits one limited restart of SEC
document requests with the already-authorized process User-Agent. Do not alter
the old checkpoint, erase the reported stop, retry its blocked documentation
URL, change routing, or bypass access controls. The first new 403/429 stops the
entire batch immediately, without retry or an automatic further restart.

All other original A11 limits remain. Each missing URL has at most its initial
attempt plus three retries, so at most 12 additional SEC attempts are possible.
No Yahoo request, database access, research search, return, price or strategy
calculation is included. Contact authorization persists; no new email is needed.

## Accounting and cache plan

- Source: `data/external/event_timing_audit/network-smoke-5b1q16b6/`.
- Verify its 50 existing reservations, zero Yahoo attempts and all raw hashes.
- Create a distinct child registration containing copies of the immutable raw
  responses, manifest and complete accounting state. Do not copy/resume old
  smoke progress or change the parent directory.
- Record parent paths, file hashes, request list and current Git revision.
  Child reservations extend the same prefix, starting at 50, rather than a
  zero counter. This is one accounting lineage: do not add parent 50 to child
  cumulative totals when reporting spend.
- The batch's cumulative SEC ceiling is 62, within the original 20,000 ceiling.
  Carry the prior elapsed runtime forward under the existing six-hour limit;
  no new six-hour allocation is created. A cache directory cannot clear a stop
  or restore budget. The prior reported research failure remains separately
  identified; transport totals describe this collection lineage, not an invented
  exact total of all research-tool HTTP subrequests.
- Preserve response bytes, errors and checkpoints. Write partial outcome JSON
  to `/tmp/insider-three-document-restart-result.json` and the child cache.
- Assert that every parent file is unchanged and the child's first 50 request
  reservations match the original. No full runner or canonical audit artifact
  is produced.
- An exclusive persistent batch claim prevents rerunning this authorization
  against a second child cache. If interrupted, preserve the claim and child
  accounting for explicit recovery; do not delete the claim or start over.

## Execution and checks

Candidate runner: `/tmp/insider_three_document_restart.py`. Default execution
validates and prints the plan without network activity or writes. `--execute`
requires the authorized contact in that process environment. A fresh Codex
reviewer checks the runner and this scoped plan before execution.

Afterward, verify actual HTTP outcomes and new cache hashes. Reparse only these
filings against the inherited raw inventory and compare original/date-only
clocks offline. Missing or contradictory support remains UNVERIFIED. Do not
repair dates or infer continuous identities from successful downloads.

This is not a paid-data decision, feasibility PASS, backtest authorization,
population acquisition, or production change.
