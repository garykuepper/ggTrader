# Intraday observer readiness — in progress

The real-source samples pass. The recorder remains off: operational rehearsal,
final code checks and independent Opus review are still pending. No live study
has been initialized and no services have been installed.

## What the source checks established

Parent performed 12 of the 20 allowed direct HTTP GET attempts, with automatic
redirects/retries disabled, plus one Alpaca documentation-connector discovery.
Exact original bodies, parameters, counts, timestamps and hashes are retained in
[the artifact folder](artifacts-2026-10-26-intraday-readiness/requests.json).
The [machine-readable preflight](_intraday_observer_readiness_preflight.json)
records the observations and limitations. No strategy signals or returns were
computed from these samples.

- Both completed sessions, October 23 and 26, 2026, returned 390 regular-session
  minute bars, no pagination and valid required 09:59, 15:30 and 15:59 records.
  The immediately preceding October 22 close and July 2 early-close anchor also
  passed. These small samples do not certify future coverage or revisions.
- A five-minute-old SIP request returned HTTP 403 with the subscription restriction;
  a sixteen-minute-old request returned HTTP 200. This agrees with Alpaca's
  [documented historical restriction](https://docs.alpaca.markets/us/docs/market-data-faq):
  the end parameter must be at least fifteen minutes old without a subscription.
  Register 900 seconds as the source restriction. The after-hours probes measure
  entitlement behavior, not exact intraday publication latency or a delivery guarantee.
- The official paper calendar agrees with offline NYSE calendar 5.5.0 through the
  proposed span, including early closes and the next session after the endpoint.
  An illustrative October 26 activation would start October 29, checkpoint November
  25, 2026, and finish November 2, 2027. These are examples, not activated dates;
  freeze the actual dates only in the final approved package.
- The host reports NTP synchronized and user lingering enabled. The largest sampled
  one-page archive is 42,243 bytes: approximately 10.65 MB for 252 such archives,
  before inputs, ledger records, metadata and filesystem overhead. Full-state size
  and runtime still require the synthetic operational check.
- Normal inputs and archive need two single-page GETs; bootstrap and recovery can
  add one each. Three attempts for all four would use the full twelve-attempt cap.
  Pagination or unusual retries can exhaust it and produce faults. The cap stays
  unchanged. Successful sampled requests completed in under a second.

## Remaining source limits

The previously parent-verified issuer workbook remains preserved under its original
hash with 43 SPY distribution rows. This does not establish future completeness.
USD currency convention and blank capital-gain semantics remain unresolved; the
non-gating benchmark stays provisional and must never be reported as complete on
those inputs. This does not prevent collecting the hypothetical journal, whose
primary rule does not use distributions. Complete terminal comparison still needs
verified semantics and an issuer publication covering the full future window.

Two public lookup attempts failed (one redirect and one missing bars-reference
route); both count toward the budget. The canonical market-data guide and FAQ
were fetched successfully. An issuer-page lookup returned an empty HTTP 301 and
was not followed. No support inquiry, paid feed or alternate source was used.

## Pending integration

Sonnet owns operational tests, representative-year timing and any necessary
recorder/wrapper changes. Hermes with Kimi supplied draft standalone unit files
and a runbook, with installation deliberately reserved. Parent must reconcile the
wrapper interface and code bindings, verify final tests and hashes, and obtain
Opus review of the concrete package. A manifest-bound live readiness attestation
must not be generated while those checks remain pending.
