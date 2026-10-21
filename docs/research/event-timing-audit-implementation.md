# Event-timing audit implementation handoff — 2026-10-21

This is a code handoff, not the registered audit report or a feasibility result.
The [approved brief](briefs/2026-10-21-event-timing-audit.md) and
[source manifest](briefs/2026-10-21-event-timing-audit-sources.md) remain unchanged.
No collection, full audit, database access or return calculation was performed
for this rework. Tests use synthetic, checksummed responses and metadata.

## Operation and evidence

`scripts/event_timing_audit.py` delegates to the single parser/main in
`ggTrader.lab.event_timing_audit`. It is offline by default. `--help` describes
`--smoke`, `--resume`, `--evidence`, `--detector-flags`, `--read-db`, `--network`
and `--output`. An offline smoke invocation is:

```bash
.venv/bin/python scripts/event_timing_audit.py --smoke --output /tmp/event_timing_audit_smoke.json
```

Smoke samples have at most two items per stratum, cannot qualify a year or
emit measured coverage PASS, and must write JSON/Markdown in `/tmp`.
Incomplete evidence cannot overwrite either registered final artifact.
`--read-db` enables explicit metadata/count/date SELECTs in read-only
transactions with a statement timeout. Connection settings come exclusively
from exported environment variables; the audit does not load `.env` or log a
connection string. No prices are selected.

**Network collection remains blocked pending code review and an authorized,
configured contact.** `--network` is a separate, explicit activation. The
User-Agent comes from exported `SEC_USER_AGENT` or `SEC_CONTACT_EMAIL`; neither
is retrieved from another file. Each HTTP attempt, including retries, reserves
its cumulative budget before sending. The first 403/429 stops both providers.
Runtime accumulates across resumes. Raw responses, including HTTP error bodies,
are immutable and checksummed; cache corruption and lost accounting fail closed.
Legacy cache files are preserved and are usable only after checksum validation.

The evidence input is JSON with `schema: "event-timing-evidence-v1"` and lists:

- `identities`: dated `symbol`, `cik`, `security_id`, `price_symbol`, `valid_from`,
  `valid_to`. Every membership interval needs supported historical identity.
  Today's SEC ticker map supplies discovery candidates only.
- `inventories`: issuer `cik`, `start`, `end`, `request_succeeded`, `complete`,
  and full `filings`, including 4/A. Verified-empty insider months count;
  empty PEAD inventories do not.
- `filings`: accession, form, filing date, API/header acceptance readings,
  raw completeness, all owners, and individual non-derivative `lots`.
  The collector preserves XML lot fingerprints and all owner/plan states.
- `amendments`: accession and evidenced `replaces_lot_ids`. Ambiguous linkage
  stays unresolved; later corrections never rewrite an earlier public ledger.
- `pead_fetches`: symbol, request success/completeness, and raw earnings records.
  `pead_events` adds the issuer/fiscal match, original-release readings,
  `actual_eps` and `consensus` lineage. Provenance records require fiscal period,
  issuer, value, basis, currency and split basis. The two bases must reconcile;
  consensus also needs a snapshot
  strictly before the original announcement. Publication metadata, 8-K receipt,
  fresh-fetch agreement and supplied surprise percentages do not prove lineage.
- `release_candidates`: cached filing index/header and exhibit observations.
  Publisher metadata is not automatically certified as the original release.
- `classifications`: security, status, effective date, SEC evidence form/link.
  These facts are frozen before selection. Index removal alone stays UNKNOWN;
  Form 25/15 and merger 8-K candidates require semantic/effective-date validation.
- `coverage_causes`: security/date intervals and supported `listing_inception`,
  `delisting_merger`, or `halt` reasons. Missing sessions are counted by cause;
  unsupported causes remain unexplained, and administrative censoring stays separate.
- `routes`: only the registered R1–R9. Each assessment cites checksummed public
  documentation, retrieval dates, statements and supported/contradictory/unknown
  dimensions. Record the exact history interval, delisted/acquired inclusion and
  requirements covered. A contradiction limited to an exact-time representation
  does not exclude a permitted fallback/mixed representation. Price versioning
  is not required. No narrative vendor verdict is promoted automatically.

Every factual record has `source_refs` matching URL/SHA-256 entries in the raw
cache MANIFEST. Hashes establish byte integrity, not the truth of a curator's
semantic annotation. Historical mapping, original-release meaning, amendment
links and vendor entitlement claims still need evidence review.

Detector input uses `event-timing-detector-flags-v1`, with `screened` intervals
(symbol/start/end/source_ref) and `flags` (symbol/date). It must be an existing
flags-only export, not a newly computed detector. Absent, malformed or
out-of-scope screening is UNVERIFIED. Earlier tape is not certified by a later
screening interval.

Progress binds the original input, membership, brief and source-manifest hashes.
`--resume` checks those bindings and reuses validated responses/accepted evidence;
request budgets and external-stop state never reset. Evidence checkpoints are
batched to avoid rewriting the growing population after every filing; individual
HTTP reservations and response manifests are durable immediately. Replayed work
uses validated cache bytes. Raw caches and earlier untracked smoke outputs are
not removed.

## Review findings 1–17

1. **Coverage constants:** replaced with dated identity/inventory unit records,
   independent attempted/resolved/retrieved/stored counts and both missingness ratios.
2. **Population/event units:** coverage uses the full PIT population independently
   of reconciliation samples; insider units are reconstructed clusters. Unknown
   populations have null official event denominators plus observed lower bounds.
3. **Stored defects:** separate Phase 2 diagnostics; raw lot preservation,
   amendment links and boundary reconstruction drive raw-source predicates.
4. **Acquisition/provenance:** counted SEC pagination, headers, ownership XML,
   filing indexes/exhibits and the installed yfinance calendar scrape protocol;
   dated identity and PEAD lineage ingestion; separate reconciled clocks.
5. **Prefix invariance:** 14-day/three-owner reconstruction with existing purchase
   and plan exclusions, every registered NYSE month-end close, full ledger versus
   independently truncated inputs, per-year results and explicit unavailable inputs.
6. **Membership:** incumbents plus every change within each month/quarter;
   unchanged periods and removals inside a period count.
7. **A10:** all relevant fixed routes, documentary dimensions, representations,
   supported history intersections and requirement-complete combinations are assessed.
   Missing documentation cannot establish NO VERIFIED PATH.
8. **Cutoff/censoring:** execution-date cutoff and 2026 YTD diagnostics;
   63-session and 126/189/252-calendar-day completeness are diagnostics only.
9. **Quarantine:** required-date failures precede grouping; raw anomalies and
   unassignable stored/raw records remain visible and cannot enter reconstruction.
10. **Sampling:** union accession frame includes stored/enumerated 4/A;
    frozen classification, 25/12/30/70 quotas, canonical hash order, shortfalls,
    failed requests and no replacement.
11. **Detector evidence:** explicit availability/interval checks; every known
    in-scope event is checked, with bar existence separate from screening.
12. **Resume/runtime:** durable request reservations, initial attempt plus three
    retries, cumulative six-hour deadline, progress/input bindings and partial reports.
13. **Recovery:** malformed/lost accounting and manifests fail closed;
    response hashes, atomic writes and immutable bytes prevent partial-cache reuse.
14. **Secrets:** connection strings are omitted entirely; no `.env` loader.
15. **Reproduction:** selected keys, shortfalls, per-event findings, raw lot IDs,
    references, SQL/parameters/query time, annual matrix and Markdown persist.
16. **Tests:** real orchestration, raw-versus-stored recovery, missing/failed
    sources, cache/security, prefix, denominator, lineage and resume fixtures.
17. **Eligible years:** missing years break runs; 2026 is excluded; longest run
    and most-recent tie-break are explicit.

These are implementation dispositions, not assertions that real data passes.

## Open definitions and full-run readiness

- Quarantined/absent disclosures can make the true cluster count unknowable.
  The tool does not invent a cluster denominator from transaction counts.
- Ambiguous amendment targets and re-signalling after a correction require
  clarification/evidenced linkage. No role, size, private-purchase or verified
  non-plan gate is added. P is not certified as exclusively open-market; ABSENT
  is not PRESENT-FALSE. Related-party independence remains a diagnostic.
- The 63-session entry-inclusion convention is unspecified. Both endpoint
  conventions are reported, without affecting qualification.
- The exact A6 event-year/membership anchor is not explicit when stated, public
  and execution dates cross a year or membership boundary. Disagreeing anchors,
  period-membership differences and unknown execution dates leave affected years
  unresolved. Provisional assignments cannot qualify such a population.
- A10's “any unresolved relevant evidence” and §5's single documented paid route
  need reconciliation before a candidate path with other unknown routes could
  receive PAID. The implementation retains UNRESOLVED in that case.
- The registered vendor routes do not automatically establish historical
  identity, actual-EPS lineage, or measured A6 coverage.

The independently reviewed vendor evidence at
`/tmp/event-timing-vendor-evidence-review.md` supersedes the original research
narrative as evidence. Neither document is loaded as runtime findings, and no
paid route or qualifying window is established by this implementation work.

**Full run is not ready for authorization:** agy code review is still required;
no SEC contact is authorized/configured; real historical identities, complete
inventories, classification/release/amendment annotations, consensus/actual
lineage and validated detector evidence remain to be supplied/collected and
reviewed. Network behavior has fixture coverage, not endpoint validation. The
full-population runtime and collection budget have not been benchmarked. Existing
NO-GOs and production freezes remain unchanged.

## Verification record

- Final owned-suite run: `scripts/run_tests.sh tests/lab/test_event_timing_audit.py -q`
  — 79 passed. Includes final cache recovery, explained missingness and PEAD
  issuer/value/basis/date lineage regressions.
- One broader run: `scripts/run_tests.sh tests/lab -q` — 868 passed, 29 deselected,
  one existing `websockets.legacy` deprecation warning. This preceded the final
  audit-only recovery/lineage regressions; the complete owned suite was rerun
  after those fixes. No additional broad-suite run was made.
- Owned Python files pass Ruff checks and format checks; script `--help` passes.
- Offline smoke and offline resume smoke: zero external requests, no database,
  at most two selected items per stratum, zero matrix PASS results. Output is
  `/tmp/event_timing_audit_codex_smoke.json` with adjacent Markdown; retained
  smoke progress is `/tmp/event-timing-offline-smoke-7lftlqlb/`.

No network/full-population collection, real DB query or endpoint validation was
performed. The smoke outputs are partial diagnostics, not canonical artifacts.
