# Intraday preflight: delegation and review

Date: 2026-10-25. Status: documentation/design review complete; implementation and authenticated probes not started. The [r7 brief](2026-10-24-intraday-feasibility.md) remains unapproved. This note does not approve its window or gates.

## Completed delegation

**Antigravity / Gemini 3.8 Flash High** extracted six named facts from four fixed Alpaca documentation pages. Codex independently checked all values. The [verification artifact](_intraday_flash_doc_check_2026-10-25.json) records the task ID, source links, values and one qualification: bar fields depend on trade-condition eligibility. Finding the documented start year, endpoint or cutoff does not demonstrate account entitlement, historical completeness or executable fills.

**Codex / GPT-6.1-Sol** reviewed r7, the feasibility report and source audit without editing files or querying accounts/market data. Its principal finding is that sampling must support a bounded conclusion: twenty clean sessions cannot establish gap-free full-window coverage. The probe also needs explicit date selection, retry/pagination limits, outcome categories and a price-free artifact contract.

## Work packages and ownership

1. **Codex parent: finalize the Stage 0A specification.** Set the diagnostic sample range independently of the unapproved strategy window; specify the items below. Deliverable: a mechanically checkable probe contract. No credentials or price inspection is needed for this design work.
2. **Codex implementer: build Stage 0A after implementation authorization.** Use the smallest research-only implementation and mocked tests. No strategy registration, simulation integration, database writes or paper-trader imports. Ownership stays in the existing research/script/test directories; no production files, `.env`, Docker or cron changes.
3. **Gemini Flash: mechanical verification only.** Given fixed files/fixtures and expected outputs, check exact fields, artifact schema or named links. It must not choose dates based on prices, interpret eligibility failures, write the monetary/statistical logic, or issue readiness conclusions. Codex verifies every result before use.
4. **Codex parent: run and review Stage 0A only after its authorization.** Its artifact must support every conclusion. Empty or failed requests are not archive-boundary proof. IEX limitations do not automatically fail primary SIP coverage.
5. **Codex: separately approved Stage 0B.** Check the frozen sample's price quality and explain each reconciliation defect, with no signals or P&L. Preserve Stage 0A evidence and separately hash any revised/refetched vendor observations.
6. **Owner and Codex: freeze the strategy protocol before bulk data/returns.** The evaluation window, A1 meaning, A2 policy and executable sizing/exit convention remain open. Review findings cannot silently approve them.

## Required specification clarifications

- **Dates:** declare an inclusive diagnostic month range. For M months, select indices `floor(j*(M-1)/19)` for `j=0..19`; require M >= 20. Select each month's first official 09:30–16:00 New York session, freeze the calendar/date manifest before any bar request, and never replace a session because its data are missing. This sampling range does not approve the strategy evaluation window.
- **Sentinels:** name exact short intervals in 2014, 2015 and early 2016, plus the required prior-close input interval. Record the earliest observation only within those intervals. Empty pre-2016 responses do not fail a proposed 2016 study.
- **Pagination/recovery:** specify finite attempts/timeouts and rate pacing before implementation. Follow tokens to completion, detect repeated tokens and preserve counts/hash history. Never classify a partially fetched response as a complete empty interval. Honor documented rate-limit headers without an unbounded loop.
- **Outcomes:** distinguish nonempty/empty successes, authentication/entitlement errors, invalid requests, exhausted throttling/transient failures, schema failures and incomplete pagination. Retain failures; do not interpolate or substitute observations.
- **Timestamps:** use documented interval semantics plus checks for alignment, duplicates, order and calendar-grid gaps. Response timestamps by themselves cannot establish the vendor's labeling convention.
- **Auctions:** preserve venue/event/time metadata. Primary-listing closing-auction selection needs a documented mapping before any price endpoint is selected. Do not choose an auction by price, volume or proximity to another close. Full-window availability remains a later validation task.
- **Stage 0A artifact:** version/stage, code and brief revisions, diagnostic range, frozen calendar/sample manifest and hash, request parameters, sanitized status/retry/pagination records, response hashes, counts and missing timestamps. Persist no OHLC prices, credentials, headers or raw error bodies. Raw-response hashes may be computed without retaining the payload.
- **Stage 0B defects:** require specific reason codes and evidence for differences. A generic citation to aggregation rules cannot excuse every discrepancy. Unknown differences stay unresolved; no tolerance is invented after seeing prices.

The six official-document claims are verified; calendar response fields and auction event-schema/venue mapping still need endpoint-specific checks while finalizing the contract. A missing answer is recorded as missing rather than delegated to Flash for an inference.

## Checks required of implementation

Mocked checks should cover deterministic month selection, DST and half-days; multi-page and repeated-token handling; bounded 429/5xx recovery; empty versus failed responses; retained missing bars; distinct auction venues/event types; and the absence of prices/secrets anywhere in a Stage 0A artifact. Stage 0B checks must reject invalid prices and retain unexplained differences as unresolved. Run all pytest work through `scripts/run_tests.sh`.

## Next action and boundary

Finalize the small Stage 0A contract, then implement/review it before seeking approval to execute the concrete metadata probe. The separate execution approval comes from the [brief's §3/§10 boundary](2026-10-24-intraday-feasibility.md), not from a need to re-approve delegation. No additional broad literature search is required for that preflight work. No return test, broker change or production change follows from this note.
