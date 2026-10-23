# Connected Metadata and Evidence Recovery Plan

> **For agentic workers:** Use superpowers:executing-plans for Codex execution.
> Track the checkboxes below. This replaces the completed charter-acquisition
> slice; do not repeat that acquisition or its reviews.

**Goal:** Recover usable audit inputs through the working read-only connection
and existing research artifacts, then measure the registered gates they support.

**Architecture:** Reuse `mcp__postgres__query`, reviewed SEC caches, the accepted
workbook and existing audit/reconstruction code. Preserve connector results as
source-bound metadata snapshots. Make no new ingestion framework or parallel audit.

**Tech stack:** Read-only PostgreSQL, existing Python audit and tests, immutable
JSON/raw evidence and Markdown reporting.

**Spec:** [Approved audit](../../research/briefs/2026-10-21-event-timing-audit.md),
[source manifest](../../research/briefs/2026-10-21-event-timing-audit-sources.md),
[date-only amendment](../../research/briefs/2026-10-22-event-timing-source-conflict-proposal.md),
[accepted workbook](../../research/2026-10-22-event-evidence-resolution-workbook.md).

## Starting point

- Read-only connector works. Schema and two venue-pinned daily-bar counts are
  [recorded](../../research/_event_audit_connected_metadata.json). The earlier
  failed direct-process lookup is not a database-access blocker.
- Two COST charters and their comparison passed independent review. Do not
  chase more charter exhibits unless a specific mapping assertion needs them.
- Latest SEC lineage: 81 attempts, zero Yahoo, 105.01228246744722 seconds;
  raw cache `data/external/event_timing_audit/cost-charters-h1kbl362/`.
- No qualifying mapping or detector export was found in the three recorded
  search roots. Different names/locations and existing DB metadata remain to check.
- Present decision remains no Phase 2 return test; source feasibility and the
  registered final audit remain unfinished. This plan does not relax that gate.

## Global constraints

- Codex owns SQL, interpretation, reconstruction, code and reviews. Flash may
  only list specified paths or extract exact fields/links, with parent verification.
- This request creates a plan; no new queries, acquisition or tests execute here.
- During execution use read-only metadata queries only: no price values, EPS
  values, return/P&L paths, production state, DB writes or detector reruns.
  Dates, identifiers, counts and missing-field indicators are sufficient.
- No `.env` read, credential request or alternate connection is needed for the
  working connector. No production/container/configuration edits or deployment.
- No external request in the first two tasks. A later exact acquisition needs
  a reviewed request list, source/spec hashes, fresh exclusive claim, inherited
  accounting and authority/stop clearance. Do not reuse consumed claims.
- SEC ≤5/sec, initial plus three retries, ≤20,000 SEC / ≤2,000 Yahoo attempts,
  six-hour cumulative deadline; first new 403/429 stops calls. Preserve old stops.
- Vendor search spent/remaining time is unknown; reconcile before further R1–R9
  searching. No fresh two-hour allocation, purchase, trial, account or outreach.
- Preserve 2026-04-30 cutoff, original conflicts/quarantine and the separate uniform
  date-only convention. Preserve A6/A7/A9 denominators, failed samples and all
  136 insider prefix checkpoints. No one-day NOC inventory tolerance.

## Review focus

Check zero-bar pairs, venue mixing, current versus dated identity, source-bound
screening versus absent flags, and amendment predecessor candidates versus proved
links. Keep insider member-months separate from PEAD member-quarters. Diagnostics
cannot pass population gates or certify a qualifying five-year window.

## Task 1: Recover database metadata and identify exact source dependencies

**Files:** Create `docs/research/_event_audit_metadata_inputs.json`; append the
findings to `docs/research/event-timing-audit-implementation.md`.
**Consumes:** Working connector and actual database schema.
**Produces:** Query/source receipts, inventory counts and exact candidate IDs.

- [ ] Query `information_schema` for tables/views and relevant column names
  describing security identities, historical symbols, validity intervals,
  corporate actions, detector runs or flags. Record the searched schemas and
  results. Name matches alone do not establish usable provenance or coverage.
- [ ] Inspect candidate schemas before selecting data. Read only identifiers,
  dates, lineage references, counts and missing-field indicators. If no candidate
  exists, record the bounded search; do not infer universal absence.
- [ ] Produce pinned annual metadata counts for `form4_transactions` and
  `earnings_surprise` through the cutoff. Retain malformed/unknown dates and
  anomalous transaction/filing relationships in separate counts. These are stored
  inventories, not verified source coverage or tradeable event counts.
- [ ] Identify the 2018 amendment predecessor using this metadata-only query:

```sql
SELECT DISTINCT accession_number, issuer_cik, insider_cik, filing_date
FROM public.form4_transactions
WHERE issuer_cik = 1133421
  AND insider_cik = 1539147
  AND filing_date = DATE '2018-02-15'
ORDER BY accession_number;
```

- [ ] Compare any returned accession with the 14 cached-inventory candidates.
  Stored owner/date matches narrow acquisition; they do not prove lot replacement.
  If none matches, retain the candidate list and its uncertainty.
- [ ] Record exact SQL, parameters, query timestamp, connector identity, raw result
  and hash. No secrets. Do not invent an existing-runner adapter or canonical
  status from a custom metadata record.

**Done when:** We know which useful inputs actually exist in the connected DB,
which exact accession(s) need raw verification, and which metadata sources are absent.

## Task 2: Recover source-bound identity and existing detector evidence

**Files:** Update metadata-input JSON and handoff; preserve accepted source artifacts.
**Consumes:** Task 1 locations/lineage and the workbook's detector contract.
**Produces:** Supported identity intervals/export receipts, or named missing inputs.

- [ ] Trace existing detector outputs from their producing script/report and
  recorded source locations. Search only project/research locations implicated
  by those references; no private-vault or arbitrary whole-host scan.
- [ ] If a legacy artifact exists, determine what dates/securities were actually
  screened and which flags were retained. Export existing metadata only when
  its lineage supports the required representation. Do not rerun detection or
  synthesize exhaustive clean intervals from incomplete legacy results.
- [ ] Validate an export's source bytes, screened intervals and flag correspondence
  using the existing ingestion contract; shape-valid JSON is insufficient.
- [ ] Inspect historical mapping candidates for issuer/class/ticker/vendor IDs,
  explicit validity intervals and supporting records. Annual observations and
  current symbol/CIK matches remain diagnostic. Identify any COST effective-date
  evidence actually needed by that map; do not let it block unrelated work.
- [ ] If a required source is unavailable, record its exact fields, historical
  scope and source/location requirement. The owner should receive one concrete
  missing-input request only if local/connected recovery is exhausted.

**Done when:** Each mapping/detector input is either source-supported and usable
or has a precise missing-source dependency. No clean-tape or identity PASS is inferred.

## Task 3: Make metadata evaluation reliable, then test supported inputs

**Files:** Conditional change in `src/ggTrader/lab/event_timing_audit.py` and
`tests/lab/test_event_timing_audit.py`; changelog if runtime logic changes.
**Consumes:** Reviewed event/security/date pairs and Task 2 evidence.
**Produces:** Correct venue-pinned counts and per-event supported predicates.

- [ ] Before using `check_tape_integrity_batch()`, add a failing multi-venue fixture:
  one yfinance daily bar plus a same-date other-venue bar must count as one;
  an other-venue-only row must count as zero. Keep missing/duplicate cases distinct.
- [ ] Reproduce the failure, pin the shared equity count query to
  `venue='yfinance'`, and preserve existing NYSE/session and detector checks.
  Read all callers first; change no unrelated runtime behavior.
- [ ] Run capped focused tests via `scripts/run_tests.sh` and Ruff check/format
  on changed Python files; obtain one fresh scoped code review before relying on it.
- [ ] Use the accepted paired LEFT JOIN query/parameters for supported exact event
  pairs. Preserve zero counts, unknown mappings, missing tradeable dates, existing
  detector flags and scope gaps. Do not query price/outcome values.
- [ ] Replay only source-supported insider inputs. Verify predecessor links from
  raw disclosures at their own public times; exclude unresolved amendments.
  Population prefix checks stay UNVERIFIED until complete required inputs exist.

**Done when:** Metadata/replay results are reproducible for their stated scope;
missing population or source evidence is explicit rather than silently omitted.

## Task 4: Advance the remaining branches only when the inputs justify it

**Consumes:** Tasks 1–3, accepted source-route evidence and fixed denominators.
**Produces:** Measured supported gates and one consolidated unresolved-input list.

- [ ] Freeze the smallest raw-request union needed for confirmed predecessor or
  independent inventory corroboration. Cost nominal/worst attempts, overlaps,
  runtime/deadline and recovery before acquisition review. Do not download the
  whole population or refresh NOC pages merely because endpoints are reachable.
- [ ] Keep NOC completeness uncleared until independently supported; a corroborated
  row and disjoint pages cannot prove that no filings were omitted.
- [ ] For PEAD, inspect any newly located provenance metadata for original
  release/issuer/fiscal matching, actual-EPS basis and strictly pre-announcement
  consensus timestamps. No EPS values or return tests are needed to establish
  whether the lineage exists. If absent, preserve that exact dependency.
- [ ] Reconcile vendor-search accounting before proposing one targeted question
  within the existing R1–R9 manifest. Do not repeat the whole vendor search.
- [ ] Compute registered annual coverage only when full applicable denominators
  and supported numerators exist. A6: ≥95% identity/retrieval/mapped prices and
  both unexplained-missingness ratios ≤2%; A7: every mandatory annual check plus
  ≥5 contiguous complete qualifying years. 2026 is diagnostic only.
- [ ] Update the existing workbook/handoff and next one or two worklist actions.
  Publish canonical audit outputs only through the existing completed bounded-run
  contract; offline snapshots cannot bypass it. Keep source UNRESOLVED separate
  from the present no-Phase-2 decision. Commit coherent verified work; no deployment.

## Completion criteria

The next work is complete when database metadata is measured, the predecessor
candidate set is narrowed, mapping/detector sources have supported dispositions,
and any evaluated gates have raw receipts. If a necessary input is unavailable,
return the exact missing source/fields/date scope and one actionable dependency.
Do not claim all five gaps are resolved or start another planning/review cycle
merely because those inputs are still missing.
