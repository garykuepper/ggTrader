# DRAFT amendment: insider acceptance-timestamp source

**Status: DRAFT — owner approval required; not implemented in the audit runner.**
Prepared 2026-10-23 under [resolution plan Task 1](../../superpowers/plans/2026-10-23-resolve-event-audit.md).
This proposes a change to [audit brief §8 A2/A8](2026-10-21-event-timing-audit.md),
not a reinterpretation of its existing approval. The
[approved date-only amendment](2026-10-22-event-timing-source-conflict-proposal.md)
continues to govern its separate representation.

## Reproduced evidence

The [raw diagnostic artifact](../_insider_acceptance_diagnostic.json) records every
cached API/header pair, issuer CIK, accession, raw-response URL, cache filename and
SHA-256. It reads the existing `insider-predecessor-bnz1mn22/MANIFEST.json`, SHA-256
`07d63f70d44280066c71aa099b27f8309cbf272c9dc965eba988b7dbbe39c83e`, under
`data/external/event_timing_audit/`. Source bytes and registration are unchanged.
The script verifies each selected raw response against its manifest hash, joins on
issuer/accession, checks header identity, and rejects absent or contradictory API
matches. It does not request data or connect to the database.

| Diagnostic on all cached matching Form 4/4-A pairs | Count |
|---|---:|
| Unique API/header pairs | 25 |
| Header interpreted in America/New_York: within 06:00–22:00 ET and same calendar date as both filing-date fields | 25/25 |
| API interpreted as its explicit UTC value, then converted to New York: outside those hours or not on the filing date | 9/25 |
| API UTC wall clock minus naive header wall clock equals twice the absolute New York UTC offset | 25/25 |
| DST header observations: wall-clock difference +8 hours | 15 |
| Standard-time header observations: wall-clock difference +10 hours | 10 |
| API UTC reading falls outside 06:00–22:00 ET | 9 |
| API UTC reading gives a different New York filing date | 1 |

The last two conditions overlap: they produce nine inconsistent pairs, not ten.
The supplied filing dates agree in all 25 pairs. The filing-hours and same-day
tests are the hypothesis specified by Task 1; the authoritative documentation
check below remains UNVERIFIED.

The subtraction needs an explicit convention. The **wall-clock** difference is
`API_UTC_without_timezone - header_without_timezone`: +8 or +10 hours. After
localizing the header to New York, the **instant** difference is +4 or +5 hours.
Both differences are retained in the artifact. For example, accession
`0001209191-24-002111` has header `20240123185549`, filing date `2024-01-23`, and
API `2024-01-24T04:55:49.000Z`. The API's UTC interpretation is
`2024-01-23T23:55:49-05:00`, outside the stated filing hours; the header interpreted
locally is `2024-01-23T18:55:49-05:00`.

This is a diagnostic of these 25 cached pairs, not the A9 sample quota of 25 per
year, a complete filing inventory, or proof about every API historical vintage.
The offset fingerprint supports the proposed source preference; it does not
establish which upstream transformation produced the discrepancy.

Reproduce from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/insider_acceptance_diagnostic.py --help
PYTHONPATH=src .venv/bin/python scripts/insider_acceptance_diagnostic.py \
  > docs/research/_insider_acceptance_diagnostic.json
scripts/run_tests.sh tests/lab/test_event_acceptance_diagnostic.py -q
.venv/bin/ruff check src/ggTrader/lab/event_acceptance_diagnostic.py \
  scripts/insider_acceptance_diagnostic.py tests/lab/test_event_acceptance_diagnostic.py
.venv/bin/ruff format --check src/ggTrader/lab/event_acceptance_diagnostic.py \
  scripts/insider_acceptance_diagnostic.py tests/lab/test_event_acceptance_diagnostic.py
```

The optional `--cache-dir` accepts another preserved cache directory with its own
manifest. The script reads only submissions API metadata and full-submission
headers. It has no return, price-path or P&L calculation.

## SEC documentation citation — UNVERIFIED

`SEC_CONTACT_EMAIL` was unset in the execution environment when checked on
2026-10-23. Under the task's explicit instruction, no SEC network request was
made and no secret or contact value was printed. **No live SEC documentation was
retrieved.** This part of Task 1 is therefore UNVERIFIED, not a verified citation.

The [SEC EDGAR Filer Manual landing page](https://www.sec.gov/submit-filings/filer-support-resources/edgar-filer-manual)
is an unverified documentation lead, not evidence retrieved in this task. The
required authoritative passages must establish EDGAR's 06:00–22:00 ET operation
and the filing-date treatment applicable specifically to Forms 4/4-A, including
the distinction from the ordinary 17:30 cutoff and any historical exceptions.
The same-day predicate used above follows the registered finding; it must not
be generalized to all filing forms. No new collection is authorized by this
draft. This missing citation is a blocker for the owner memo.

## Proposed rule, effective only after explicit owner approval

1. Assess a uniformly applied `insider_header_acceptance` representation for the
   entire registered insider population. Use the matching full-submission header
   `ACCEPTANCE-DATETIME`, interpreted in `America/New_York` with DST, as the
   acceptance-based signal-availability clock. Match issuer CIK and accession,
   and retain the raw acceptance string and filing date separately.
2. Preserve the API `acceptanceDateTime` unchanged as a **known-defective
   diagnostic for the cached pairs**. Record both raw readings, source references,
   offsets and original conflict results. Do not arithmetically repair the API
   value, use it as a fallback clock, or widen the existing one-minute tolerance.
   Broader API-vintage behavior remains unverified.
3. For this proposed representation only, an API/header disagreement would remain
   an explicitly recorded source defect rather than independently invalidating a
   supported header clock. Missing, malformed, mismatched-identity or conflicting
   header evidence remains UNVERIFIED or quarantined as applicable. Other A8
   quarantines remain binding. Do not choose a representation by event, company,
   year or performance.
4. Keep A1's first regular-session close at least 30 minutes after the supported
   availability time, its secondary open diagnostic, and the existing NYSE
   holiday/early-close treatment. Store observed timestamps and inferred execution
   times separately. This proposal changes the timestamp source, not those rules.
5. Acceptance is distinct from public dissemination. This proposal supplies no
   observed dissemination timestamp or supported dissemination bound; those
   properties remain UNVERIFIED. An owner's approval authorizes a source
   convention, not certification of historical public availability.
6. Preserve the original `insider_timestamp` result and quarantine for comparison.
   Independently assess every A6/A7 identity, inventory, timing, reconstruction,
   tape and five-complete-year requirement for the proposed representation.
   The 25-pair diagnostic alone passes none of those annual gates.

## Explicit effect on the approved date-only amendment

Approval of this draft would add a separately assessed header-clock candidate;
it would **not rescind or silently replace** `insider_filing_date_only`. That
approved candidate still requires exact agreement between the API filing-date
field and header filing-date field for the same issuer/accession, trades at the
first NYSE close strictly after that corroborated date, applies no second
30-minute buffer, and retains every original receipt conflict. Its precision is
`date_only`; it remains distinct from observed acceptance timing.

The two conventions can imply different execution dates: a sufficiently early
supported header time can permit that session's close, while the approved
date-only convention always waits for a later session. No return calculation or
selection between conventions has been performed. If this draft is rejected or
left pending, the runner's existing timestamp quarantine and approved date-only
logic continue unchanged. Any future implementation needs its own scoped
authorization; this task changes neither runner nor protocol behavior.

## Owner gate and unresolved evidence

The owner may approve or reject this source amendment. Until then it is DRAFT.
The SEC documentation citation is UNVERIFIED. Identity, complete inventories,
historical dissemination, annual coverage, reconstruction and detector/tape
evidence remain independent requirements; this draft does not resolve them.
Any additional evidence need belongs in the decision memo under the stop rule.
