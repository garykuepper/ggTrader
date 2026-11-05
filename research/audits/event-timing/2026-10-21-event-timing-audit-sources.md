# Source-search manifest: event-timing audit

**Status:** approved by the owner 2026-10-21 together with
[the audit brief](2026-10-21-event-timing-audit.md) (§8 A10). It is fixed by the senior quant, and the
builder may not add, omit or substitute routes.

Every route below is a **candidate to assess**, not a claim that it qualifies. Product names
come from general knowledge and must be confirmed from the vendor's own public documentation;
a name or document that can't be found is UNRESOLVED, not an exclusion. Only authoritative evidence that contradicts a requirement excludes a route.

## Required evidence per route

A route *addresses* a requirement only if public vendor documentation (cite URL and retrieval
date) states all four:

1. **Fields:** the specific fields that resolve the blocking requirement.
2. **History:** coverage start date, with delisted and acquired companies included.
3. **Point-in-time:** values available as of each historical date (as-of or snapshot timestamps),
   not restated or revised later.
4. **Access:** licensable by an individual, non-academic subscriber, with a stated or quotable
   price.

A route is a documented exclusion only if its documentation contradicts any of the four. If
the documentation is silent on any of them, the route is UNRESOLVED for that requirement.
Point-in-time evidence (item 3) applies to signal inputs. Execution prices may be corrected
without versioning, provided delisted securities are included (brief §8 A10).

## Routes, in search order

| # | Blocking requirement | Candidate route (to verify) |
|---|---|---|
| R1 | PEAD pre-announcement consensus history | LSEG I/B/E/S historical estimates |
| R2 | PEAD pre-announcement consensus history | Zacks historical consensus estimates |
| R3 | PEAD pre-announcement consensus history | S&P Capital IQ estimates history |
| R4 | PEAD announcement timestamps | Wall Street Horizon earnings-date history |
| R5 | Survivorship-free daily prices, delisted names included | CRSP daily stock file |
| R6 | Survivorship-free daily prices, delisted names included | Sharadar Equity Prices (Nasdaq Data Link) |
| R7 | Survivorship-free daily prices, delisted names included | Norgate Data US stocks, delisted included |
| R8 | Insider filings with acceptance times and all owners | Sharadar insider holdings (Nasdaq Data Link) |
| R9 | Insider filings with acceptance times and all owners | SEC EDGAR itself (free; re-parse) |

R9 is the free baseline. It resolves only its own documented filing requirements, not historical identity or prices. Statuses are assessed per signal, using only the relevant routes (brief §8 A10).

## Limits and completion

- Use public documentation pages only: no sales calls, trials, or account creation.
- At most 2 hours in total across all routes.
- The search is *complete* when every route has either a documented pass for each of the four
  evidence items, or a documented exclusion citing the contradicting page. A route still
  silent on any item when the time limit is reached leaves that requirement UNRESOLVED.
- A **mixed route** (for example free SEC timestamps plus paid consensus) qualifies only if
  each blocking requirement is covered by some route over a common contiguous window of at
  least 5 complete calendar years (brief §8 A7).
