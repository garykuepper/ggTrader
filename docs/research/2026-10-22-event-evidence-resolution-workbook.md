# Event evidence resolution workbook — corrected offline disposition

Both insider representations and PEAD remain **UNRESOLVED**. This workbook replaces the rejected draft; it is a documentary evidence ledger, not the registered final audit matrix. The production freeze and every fixed A6/A7/A9 rule remain in force. No new external, database, detector, production or strategy work was performed.

Authority: [binding plan](../superpowers/plans/2026-10-22-event-evidence-resolution.md), [audit A1–A12](briefs/2026-10-21-event-timing-audit.md), [R1–R9 source manifest](briefs/2026-10-21-event-timing-audit-sources.md), [approved date-only amendment](briefs/2026-10-22-event-timing-source-conflict-proposal.md), [latest handoff](event-timing-audit-implementation.md), and [reviewed acquisition results](_insider_identity_inventory_resolution_results.json). The [machine companion](_event_evidence_resolution_workbook.json) preserves exact URL/file/hash tuples, raw locators, all cached ownership metadata, separate timing predicates, annual counts and all 136 checkpoints.

## Accounting and scope

The immutable cache is `data/external/event_timing_audit/identity-inventory-we_55vr8/`. Rehashed **79/79** raw responses; the checkpoint retains **79 SEC attempts, zero Yahoo attempts, 104.3013715306297 cumulative seconds**. The prefix is 53 + 26, not two independent totals. The 26-URL authorization is consumed. Limits remain 20,000 SEC / 2,000 Yahoo attempts, four attempts per URL including initial, ≤5 SEC requests/sec and six cumulative hours. No new request is authorized by these arithmetic balances.

Vendor documentation spent and remaining hours are **UNVERIFIED** under the two-hour cumulative ceiling. The original claimed 50 minutes is unverified; the retained independent review adds under four minutes (2026-10-21 21:45:47–21:49:13 UTC), without establishing a reconciled allowance. Its references to 70 remaining minutes depend on that unverified claim. The review reported no exposed 403/429; it records bot interstitials, JavaScript shells and a CRSP 401 separately. The later field-semantics report records a **SEC documentation** 403 at `https://www.sec.gov/edgar/sec-api-documentation`, not demonstrated third-party vendor blocking. Preserve this STOP and consumed bounded restarts separately; do not retry that URL or infer new search allowance.

## 1. Historical identity and legal class

All 24 annual report cover/symbol extracts were checked against their exact manifest sources. These are dated observations, **not annual or continuous identity PASS**. Filing date, fiscal end and any legal effective date are distinct; effective dates remain unknown. Newer NOC covers say “Common Stock”; $1 par is located separately in Item 5 (2020–2025) or the financial-position common-stock row (2019). Earlier NOC covers include $1 par. Exact extracts and locators are in `annual_identity_observations`.

| Issuer | Fiscal end | Filing date | Accession | Dated class observation |
|---|---|---|---|---|
| NOC | 2025-12-31 | 2026-01-27 | `0001133421-26-000003` | Common Stock; $1 par in separately located statement |
| NOC | 2024-12-31 | 2025-01-30 | `0001133421-25-000006` | Common Stock; $1 par in separately located statement |
| NOC | 2023-12-31 | 2024-01-25 | `0001133421-24-000006` | Common Stock; $1 par in separately located statement |
| NOC | 2022-12-31 | 2023-01-26 | `0001133421-23-000006` | Common Stock; $1 par in separately located statement |
| NOC | 2021-12-31 | 2022-01-27 | `0001133421-22-000004` | Common Stock; $1 par in separately located statement |
| NOC | 2020-12-31 | 2021-01-28 | `0001133421-21-000006` | Common Stock; $1 par in separately located statement |
| NOC | 2019-12-31 | 2020-01-30 | `0001133421-20-000006` | Common Stock; $1 par in separately located statement |
| NOC | 2018-12-31 | 2019-01-31 | `0001133421-19-000007` | Common Stock; $1 par in separately located statement |
| NOC | 2017-12-31 | 2018-01-29 | `0001133421-18-000005` | Common Stock; $1 par in separately located statement |
| NOC | 2016-12-31 | 2017-01-30 | `0001133421-17-000008` | Common Stock; $1 par in separately located statement |
| NOC | 2015-12-31 | 2016-02-01 | `0001133421-16-000065` | Common Stock; $1 par in separately located statement |
| NOC | 2014-12-31 | 2015-02-02 | `0001133421-15-000008` | Common Stock; $1 par in separately located statement |
| COST | 2025-08-31 | 2025-10-08 | `0000909832-25-000101` | Common Stock; $.005 par in cover table |
| COST | 2024-09-01 | 2024-10-09 | `0000909832-24-000049` | Common Stock; $.005 par in cover table |
| COST | 2023-09-03 | 2023-10-11 | `0000909832-23-000042` | Common Stock; $.005 par in cover table |
| COST | 2022-08-28 | 2022-10-05 | `0000909832-22-000021` | Common Stock; $.005 par in cover table |
| COST | 2021-08-29 | 2021-10-06 | `0000909832-21-000014` | Common Stock; $.01 par in cover table |
| COST | 2020-08-30 | 2020-10-07 | `0000909832-20-000017` | Common Stock; $.01 par in cover table |
| COST | 2019-09-01 | 2019-10-11 | `0000909832-19-000019` | Common Stock; $.01 par in cover table |
| COST | 2018-09-02 | 2018-10-26 | `0000909832-18-000013` | Common Stock; $.01 par in cover table |
| COST | 2017-09-03 | 2017-10-18 | `0000909832-17-000014` | Common Stock; $.01 par in cover table |
| COST | 2016-08-28 | 2016-10-12 | `0000909832-16-000032` | Common Stock; $.005 par in cover table |
| COST | 2015-08-30 | 2015-10-14 | `0000909832-15-000014` | Common Stock; $.005 par in cover table |
| COST | 2014-08-31 | 2014-10-15 | `0000909832-14-000021` | Common Stock; $.005 par in cover table |

COST covers report $.005 (2014–2016), $.01 (2017–2021), then $.005 (2022–2025). The 2017/2022 balance-sheet and combined dividends/other equity rows show $2 million common-stock/APIC transfers in opposite directions. This extends beyond cover typography; it does **not** establish a legal conversion, effective date, or rule out broader clerical/accounting explanations. The 2019 Exhibit 3.1 caption cites a 2019 quarter, but its actual href targets the 2015 charter; neither establishes a separate 2019 charter.

The exact two-charter subpackage and actual Exhibit 4.8 targets appear below. Their contents are uncached. Continuous issuer identity, legal-class continuity, ticker validity and vendor-price identity remain separate **UNVERIFIED** dependencies for COST, NOC and the full PIT population. No current ticker map or survivor-only substitute is accepted.

## 2. NOC inventory incident

`page_inventory` reproduces page 001’s **2,000 rows**, one out-of-range accession `0001225208-17-005319`, descriptor end **2017-03-01**, API/header filing date **2017-03-02**, API acceptance `2017-03-03T02:23:24.000Z`, header acceptance `20170302162324`. Page 002 has **214 rows**, 2001-02-01–2002-06-17, within its declared end 2002-06-22. Recent/page001/page002 accession sets have zero overlaps. All bytes/rows are retained.

The strict descriptor predicate remains **FAIL**. Existing `enumerate_sec()` rejects the page rather than certifying it. Missing-page acquisition is closed; completeness is not. Corroboration establishes this row, and disjointness establishes no overlaps; neither detects omitted filings. No authoritative descriptor-semantics document is present in the 79-response manifest. The rejected field-semantics narrative cannot supply it. Cause remains unknown.

Dependency D2 names the exact root/page URLs for a possible separately recorded source-vintage comparison, but a refresh alone cannot establish completeness. Independent interval-wide corroboration still needs a frozen source list and cost. No one-day tolerance or descriptor waiver is proposed. Any future changed interpretation requires explicit approval plus a completeness assessment across all pages, count mismatches, gaps/overlaps and boundary errors in both directions; the present failed gate remains unchanged.

## 3. Mapped metadata and detector evidence

A bounded schema-string search, `rg -l --hidden --glob '!*.pyc' 'event-timing-detector-flags-v1' docs results data/external/event_timing_audit/identity-inventory-we_55vr8`, located only the two rejected workbook files at correction start. No qualifying export was located **in these searched paths**. This is not exhaustive filesystem absence or a clean-tape finding.

`load_existing_artifact_flags()` requires `schema`, `screened` and `flags`; screened rows contain `symbol/start/end/source_ref`, flags contain `symbol/date`. Runner ingestion additionally validates the referenced bytes, schema, exact screened interval and corresponding in-interval flags. A shape-valid JSON file alone is insufficient. An existing source-bound export is required; no detector may be rerun.

One paired, zero-preserving metadata SELECT replaces the three defective proposals. It is not executed. Parameters below are executable psycopg2-style parameters, and the statement starts with SELECT for `ReadOnlyDb`. The illustrative security IDs deliberately remain UNVERIFIED and block audit use until mapped to reviewed historical securities. Bare NOC/COST are the equity-cache symbols. For PEAD the lower horizon is 2002-01-01; insider uses 2015-01-01; cutoff is 2026-04-30.

```sql
SELECT e.event_id, e.security_id, e.price_symbol, e.session_date,
       count(o.timestamp)::bigint AS bar_count
FROM jsonb_to_recordset(%s::jsonb)
     AS e(event_id text, security_id text, price_symbol text, session_date date)
LEFT JOIN ohlcv AS o
  ON o.symbol = e.price_symbol AND o.venue = 'yfinance' AND o.interval = '1d'
 AND o.timestamp >= e.session_date::timestamp
 AND o.timestamp < e.session_date::timestamp + INTERVAL '1 day'
WHERE e.session_date BETWEEN %s::date AND %s::date
GROUP BY e.event_id, e.security_id, e.price_symbol, e.session_date
ORDER BY e.event_id, e.security_id, e.price_symbol, e.session_date
```

```python
params = (
    '[{"event_id":"date-only-fixture-NOC-2025-02-20","security_id":"UNVERIFIED:NOC:1133421","price_symbol":"NOC","session_date":"2025-02-21"},{"event_id":"date-only-fixture-COST-2026-04-01","security_id":"UNVERIFIED:COST:909832","price_symbol":"COST","session_date":"2026-04-02"}]',
    "2015-01-01",
    "2026-04-30",
)
```

The controller must validate the paired input manifest before execution, retain unavailable date/mapping exceptions, and check each date against NYSE sessions. Every requested pair returns a count including zero; zero is missing, >1 is multiplicity. No unbounded extents or HAVING-only missing-date test remains. For forward-window diagnostics use explicit expected NYSE event/security/session pairs, preserve right censoring, and do not turn forward completeness into eligibility. Counts cannot prove historical security linkage. Missing detector scope remains UNVERIFIED.

Existing `check_tape_integrity_batch()` omits venue from its count although OHLCV’s PK includes venue and the stock loader uses `yfinance`. Before that runtime path is relied on, the controller needs a separate multi-venue failing fixture, smallest shared query fix, capped targeted tests and review. This workbook makes no runtime change.

## 4. Ownership reconstruction, timing and qualification

The current cache contains **24 ownership accessions / 52 non-derivative lots**: 23 retained smoke selections plus the separately acquired inventory incident. The unavailable 24th smoke selection `0001628280-26-037192` (2026-05-21) remains retained, after cutoff, with no XML/header and no replacement. None of these operational selections substitutes for A9’s population frame or 25/year quota. All 52 cached lots contain share-count fields; transaction-price and dollar-size values were not inspected, and no size/role filter was selected. The machine ledger records every cached owner/role, lot ordinal/fingerprint, class, direct/indirect holding and nature, derivative/holding metadata, missing fields and footnotes. Each cached accession has one owner; multi-owner handling is code capability, not demonstrated sample coverage. Related-entity independence remains unknown. No cached non-derivative lot has code P, so no purchase-cluster or private-P classification is demonstrated.

| Timing scope | Original timestamp namespace | `insider_filing_date_only` | Dissemination |
|---|---|---|---|
| 23 cached retained smoke selections | 23 receipt comparisons FAIL; quarantine retained; timing UNVERIFIED | 23 convention-only PASS; exact API/header filing dates, next NYSE close | UNVERIFIED, availability null |
| Missing selected 2026-05-21 accession | UNVERIFIED raw pair; retained shortfall/after cutoff | UNVERIFIED, no inferred usable event | UNVERIFIED |
| Separate 2017-03-02 inventory incident | Receipt comparison FAIL; quarantine retained | UNVERIFIED under existing corroboration replay: strict inventory rejects page; directly observed filing dates agree | UNVERIFIED |

`timing_observations` retains both raw receipt readings, failed comparisons, corroboration sources, quarantines and inferred execution separately. The separate amendment applies uniformly, uses no extra 30-minute buffer, and does not certify publication. All other A8 exclusions and annual gates remain required. There is no blanket R9/timing PASS.

Amendment **`0001225208-18-004211`** was filed **2018-02-26**, header acceptance **20180226174855**, original-submission date **2018-02-15**. Footnote 2 corrects the number of shares surrendered for tax; a transaction-code change is unproved. Owner is **1539147**. The original is uncached. Existing `cluster_snapshot()` was exercised offline with unresolved amendment lots: it records `amendment linkage unresolved: 0001225208-18-004211` and excludes them. They are never standalone usable entries or retroactive replacements. `predecessor_identification` lists all **14** same-issuer/date Form 4 candidates with exact canonical full-submission URLs; these are candidates, not asserted predecessors. D4 costs that identification package.

The corrected full-submission bindings are:

- [0001628280-22-028934](https://www.sec.gov/Archives/edgar/data/1133421/000162828022028934/0001628280-22-028934.txt): `sec_e2eb2eb2c49e369a9ddbcbc7261873473fe97f4cd1dccb8e8ecdec11d96f3222_3aec9aaabd62aecee7981bbfe2cdde00c92a273db5086dff0424b58452dd896d.raw`, SHA-256 `3aec9aaabd62aecee7981bbfe2cdde00c92a273db5086dff0424b58452dd896d`.
- [0001225208-18-004211](https://www.sec.gov/Archives/edgar/data/1133421/000122520818004211/0001225208-18-004211.txt): `sec_e2cd8182197047e068709fa77a4d0a06082b545c965c9cdbf72b7379dd25e0b1_0415762b37a539e6e2cc6b8f7463abcb56d5bc15ac2b50a2273418860dff7c04.raw`, SHA-256 `0415762b37a539e6e2cc6b8f7463abcb56d5bc15ac2b50a2273418860dff7c04`.

Collision grouping uses the actual stored PK `(accession_number, insider_cik, transaction_date, transaction_code, shares)`. In `22-028934`, **11 lots** include groups of **six and two**, **eight involved lots**, and **six potential excess rows** the key cannot distinguish. All 11 are S; this does not demonstrate lost purchase-cluster events. COST `16-146844` has two F groups of two, two further potential excess rows. Across the 52 cached lots, four collision groups and zero exact-duplicate candidates were reproduced. No stored DB rows were inspected; no measured loss is asserted. Raw ordinal/fingerprint retention addresses reconstructability separately from a future stored-parser build requirement.

Plan inventory is **18 ABSENT / 5 PRESENT-FALSE / 1 PRESENT-TRUE** at filing level. ABSENT includes February 2023 and remains unknown participation. The retained primary review identifies **April 1, 2023 compliance** and separately February 27, 2023 rule effectiveness; neither sample absence nor these dates establish schema introduction history. Authoritative versioned schema evidence is required for that claim; no NLP implementation is prescribed.

Raw-date lag diagnostic via existing `filing_lags()` covers 52 lots; only the two amendment lots exceed two NYSE sessions (eight each). This is neither a legal late-filing conclusion nor clearance of original timing quarantines. Existing `load_boundaries()` inspected 51 convention-compliant cached copies at **136 month ends × 3 hold settings = 408 diagnostic rows**, finding **6 nonempty omission rows**. This cannot exclude missing disclosures in an incomplete cache. Complete carry-in, 14-day cluster lookback, 30-day load pad, prior-year late filings/amendments and explicit predecessor links remain required.

A9 stays frozen: ascending SHA-256(`20261021|canonical_key`), no replacement; insider 25 filings/year (2015–2026), key accession; PEAD 12 events/year (2002–2026), key `symbol|earnings_date`; re-fetch 30 verified delisted/acquired plus 70 other ever-members only after classification. Smaller strata are censused and shortfalls/failures retained.

All **136** NYSE month-end closes (2015-01–2026-04) are preserved in `prefix_checkpoints`, with independently labelled original/date-only UNVERIFIED predicates. Existing helper’s unavailable-input branches generated them; no prefix equality was verified. No passing empty reconstruction is inferred. PEAD does not inherit this insider-specific requirement.

## Annual population denominators and fixed gates

Reproduced through existing `build_universe_denominators()` / `membership_units()` using membership SHA-256 **`b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472`**. Insider: **68,806 member-months / 761 symbols**. PEAD: **49,542 member-quarters / 1,013 symbols**, including partial 2026-Q2. NOC+COST’s **272 insider units** describe diagnostic denominator scope, not verified retrieval coverage. Current-symbology membership labels do not by themselves certify historical ticker/price identity.

| Year | Insider member-months | PEAD member-quarters | Qualification treatment |
|---|---:|---:|---|
| 2002 | — | 2001 | Numerators/mandatory annual checks UNVERIFIED |
| 2003 | — | 1985 | Numerators/mandatory annual checks UNVERIFIED |
| 2004 | — | 1994 | Numerators/mandatory annual checks UNVERIFIED |
| 2005 | — | 1999 | Numerators/mandatory annual checks UNVERIFIED |
| 2006 | — | 2020 | Numerators/mandatory annual checks UNVERIFIED |
| 2007 | — | 2025 | Numerators/mandatory annual checks UNVERIFIED |
| 2008 | — | 2021 | Numerators/mandatory annual checks UNVERIFIED |
| 2009 | — | 2022 | Numerators/mandatory annual checks UNVERIFIED |
| 2010 | — | 2007 | Numerators/mandatory annual checks UNVERIFIED |
| 2011 | — | 2004 | Numerators/mandatory annual checks UNVERIFIED |
| 2012 | — | 2004 | Numerators/mandatory annual checks UNVERIFIED |
| 2013 | — | 2006 | Numerators/mandatory annual checks UNVERIFIED |
| 2014 | — | 2006 | Numerators/mandatory annual checks UNVERIFIED |
| 2015 | 6034 | 2030 | Numerators/mandatory annual checks UNVERIFIED |
| 2016 | 6087 | 2046 | Numerators/mandatory annual checks UNVERIFIED |
| 2017 | 6096 | 2051 | Numerators/mandatory annual checks UNVERIFIED |
| 2018 | 6088 | 2045 | Numerators/mandatory annual checks UNVERIFIED |
| 2019 | 6087 | 2048 | Numerators/mandatory annual checks UNVERIFIED |
| 2020 | 6080 | 2040 | Numerators/mandatory annual checks UNVERIFIED |
| 2021 | 6081 | 2041 | Numerators/mandatory annual checks UNVERIFIED |
| 2022 | 6067 | 2038 | Numerators/mandatory annual checks UNVERIFIED |
| 2023 | 6055 | 2031 | Numerators/mandatory annual checks UNVERIFIED |
| 2024 | 6055 | 2032 | Numerators/mandatory annual checks UNVERIFIED |
| 2025 | 6057 | 2033 | Numerators/mandatory annual checks UNVERIFIED |
| 2026 | 2019 | 1013 | Diagnostic only; PEAD Q2 partial |

PEAD 2026 diagnostics are **509** member-quarters in Q1 and **504** in partial Q2 through April 30; neither contributes a complete qualifying year.

A6 remains identity ≥95%, successful complete retrieval ≥95%, correctly mapped event prices ≥95%, and **both** unexplained-unit/event missingness ratios ≤2%. Failed, quarantined, unknown-trade-date and administratively censored events stay in denominators. Insider events are clusters; PEAD events are security/fiscal-period earnings records. Attempted/resolved/fetched/stored counts are distinct and unavailable here. A7 additionally requires every applicable mandatory annual predicate and ≥5 contiguous complete qualifying years; 2026 never qualifies. No such window is established. This workbook does not replace the registered matrix or invent a new feasibility status.

## 5. PEAD lineage and retained vendor evidence

The stored schema is `symbol, earnings_date, eps_estimate, eps_reported, surprise_pct`. The field is **eps_reported**, not eps_actual. Contract inspection supplies no announcement-time, consensus-snapshot/revision, actual-EPS source receipt, issuer/fiscal-period, currency, GAAP/adjusted or split-basis lineage. No stored EPS values or current row counts were read. This contract gap is not a measured zero-provenance rate for all records.

Original release/issuer/fiscal matching, actual-EPS provenance and consensus provenance remain three separate UNVERIFIED requirements. Actual EPS needs source receipts plus fiscal/basis/currency/split correspondence. Consensus requires an historically available snapshot **strictly before the original announcement**, with matching basis and period; later acceptance or revised estimates cannot satisfy it. Scaling, zero/negative estimates, duplicates and missing values need source metadata; no price or outcome work is authorized. A3’s filing regimes and date-only fallback do not prove consensus vintage.

The following four-item matrix is rebuilt from the retained independent evidence review, **not fresh web verification**. `vendor_routes` retains each primary URL, retrieval date **2026-10-21**, precise review-section locator and successful-open versus indexed/limited/failed status. Primary vendor bytes are not in the 79-file SEC manifest; URLs alone are not treated as retrieved contents. All end-to-end route dispositions remain UNRESOLVED.

| Route | Fields | History / delisted-acquired | PIT | Individual access / price |
|---|---|---|---|---|
| [R1 LSEG I/B/E/S](https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf) | Consensus/PIT snapshots documented; exact pre-release EPS/fiscal fields unresolved. | Product starts differ: 1976/1987 brochure vs PIT 1980 page; selected-package delisted/acquired scope unresolved. | Daily PIT dates/timestamps documented for PIT package; event-level strict pre-release selection and common window unresolved. | Individual eligibility and matched price unresolved. |
| [R2 Zacks](https://zacksdata.com/consensus/faq/) | Consensus history and export capability documented separately; EEH schema/basis unresolved. | Annual 1979, quarterly 1982; DBCMHIST delisted/acquired since 1999-12-17 is product-specific. | DBCMHIST bias-control claim does not establish EEH or historical-export snapshot contract. | Individual historical entitlement/current matched price unresolved; institutional restriction concerns individual analyst estimates, not all consensus. |
| [R3 S&P Capital IQ](https://www.spglobal.com/market-intelligence/en/solutions/capital-iq-estimates) | Snapshot spEffectiveDate/spToDate and consensus documented; EPS schema/basis unresolved. | Ordinary history 1999 North America/1997 international; package delisted/acquired scope unresolved. | Two-hour Snapshot changes since August 2016; not PIT since ordinary-series inception. | Individual eligibility/current price unresolved. |
| [R4 Wall Street Horizon](https://www.wallstreethorizon.com/upload/WSHEclassesandfieldsforIBAPI2022-12-23.pdf) | wshe_ed schedule confirmation differs from wshe_eps report-announcement datetime/release purl; issuer/fiscal-period original-release matching unresolved. | General 2006 history varies by dataset; release-specific and delisted/acquired history unresolved. | Schedule snapshots do not establish original-release historical timing. | Individual entitlement and price unresolved. |
| [R5 CRSP](https://indexes.morningstar.com/research-data-products/crsp-us-stock-databases) | Daily prices, corporate actions, PERMNO/PERMCO and delisting fields documented at product level. | Active/inactive and exchange history: NYSE 1925-12-31, NYSE American 1962-07-02, Nasdaq 1972-12-14. | Historical execution-price versioning not required by A10; dated mapping and tape checks remain required. | Individual eligibility and price unresolved. No academic-only or natural-person prohibition established. |
| [R6 Sharadar SEP](https://sharadar.com/docs/stocks) | Price/date fields, SEP alias and security metadata documented. | December 1997, active/delisted; acquired retention not separately explicit in examined schema. | Execution-price versions not required; dated mapping/coverage/tape still unverified. | Direct personal Prices: 5y $9/mo or $99/y; 10y $19/mo or $199/y; full $39/mo or $299/y. Not transferred to Nasdaq entitlement. |
| [R7 Norgate](https://norgatedata.com/stockmarketpackages.php) | Daily price exports documented; dated ticker-CIK mapping unresolved. | Platinum listed/delisted/previously listed OTC to 1990; Diamond to 1950; essentially complete late 1992 onward for included types, less complete earlier. Acquired target delisted, survivor continues. | Execution-price versions not required; corrections allowed. | Natural-person license. Platinum $630/y, Diamond $787.50/y (USD); Windows updater/export constraint, not Linux exclusion. |
| [R8 Sharadar SF2](https://sharadar.com/docs/insiders) | Insiders/SF2: filing date, roles, codes, shares, security title, direct/indirect and row number. Published direct schema lacks intraday acceptance. | January 2008; explicit delisted/acquired retention unresolved. | Exact timestamp standalone unsupported; permitted date-only/mixed route remains unresolved for raw/version/amendment/all-owner/prefix requirements. | Direct personal Investors 5y $19/mo, 10y $29/mo, full $39/mo; Nasdaq quote unresolved. |
| [R9 SEC EDGAR](https://www.sec.gov/about/webmaster-frequently-asked-questions) | Headers, repeated owners, amendments, transaction/security fields supported; actual reconstruction not thereby passed. | Electronic ownership filing mandatory 2003-06-30; complete per-issuer/month historical inventories and dated mapping unresolved. | Receipt differs from dissemination. Cached original conflicts retained; amended convention assessed separately; dissemination bound unresolved. | Free individual public access documented; does not resolve prices, identity or consensus. |

Exact retained primary references are in the machine matrix and [retained independent review](/tmp/event-timing-vendor-evidence-review.md); no unvisited documentation lead is upgraded to primary evidence. For signal inputs PIT evidence remains mandatory. Historical versioning of execution prices is **not** an added gate. R5 has no supported academic-only exclusion; R8 begins January 2008, not 2003; R4 schedule histories do not establish original releases; R7’s 1950 history is Diamond-qualified. Direct Sharadar terms/prices do not automatically transfer to Nasdaq.

Candidate mixed routes remain R9 plus eligible R5/R6/R7 prices for insider, R8 permitted date-only or R8/R9 supplementation where demonstrated, and R1/R2/R3 consensus plus verified original release/actual EPS and eligible prices for PEAD. No component overlap establishes a common requirement-complete five-full-year window. None is documented universally ruled out, so NO VERIFIED PATH is not supported.

## Consolidated dependency and acquisition handoff

Only the following **two exact charter URLs** form the bounded initial document subpackage, with actual href provenance in `charter_link_observations` (relative raw links resolved against the source URL; http/https archival spelling is preserved separately):

- [costex3110q21515.htm](https://www.sec.gov/Archives/edgar/data/909832/000090983215000003/costex3110q21515.htm): 2015 charter incorporated by both 2016 and 2017 annual reports; inspect common-stock/par-value provisions and effective-date history.
- [costex3110k082822.htm](https://www.sec.gov/Archives/edgar/data/909832/000090983222000021/costex3110k082822.htm): 2022 charter filed as Exhibit 3.1; compare operative common-stock terms and effective-date clauses.

Cost: two nominal SEC attempts, eight maximum including retries; inherited cumulative totals project to **81 nominal / 87 maximum**. Nominal and worst-case runtime are unknown; six seconds was unsupported. The global six-hour cap is not a runtime estimate. These charter requests are a proposal, not a restart, a complete acquisition package, or proof they will resolve legal continuity.

Conditional only, after reading those charters:

- [costex3110q21620.htm](https://www.sec.gov/Archives/edgar/data/909832/000090983220000004/costex3110q21620.htm): Intermediate operative charter if two-document comparison leaves gap.
- [costex4810k083020.htm](https://www.sec.gov/Archives/edgar/data/909832/000090983220000017/costex4810k083020.htm): 2020 Exhibit 4.8 description of common stock, actual linked target.
- [costex4810k082822.htm](https://www.sec.gov/Archives/edgar/data/909832/000090983222000021/costex4810k082822.htm): 2022 Exhibit 4.8 description of common stock, conditional support.

| Dependency / order | Exact missing input or action | Cost / stop disposition |
|---|---|---|
| D0-clearance / 0 | Controller reviews concrete scope against consumed authorization, prior SEC documentation stop and cumulative ledger; no restart occurs here. | No external calls. Required before any acquisition. |
| D1-charters / 1 | Two exact charter leads; operative common-stock/par clauses, amendment/effective dates. NOC equivalent legal-action evidence remains separately unavailable. | 2 nominal / 8 maximum SEC attempts; cumulative 81 / 87. Nominal/worst runtime unavailable; six-hour cumulative hard cap retained. |
| D2-inventory / 1 | Authoritative descriptor-semantics content not in 79-response cache. Independent interval-wide filing/index corroboration would need a frozen exact URL inventory and dated scope; not supplied. Root/page refresh URLs below are independent source-vintage comparisons only, cannot prove completeness alone. | Exact independent corroboration set and nominal/worst requests/runtime unavailable; stop before collection. Three named NOC root/page refresh URLs would cost 3 nominal / 12 maximum attempts, but are not queued or a completeness resolution. |
| D3-mapping / 2 | Reviewed identities with historical issuer, ticker/class and price-security validity start/end plus source receipts for all PIT units; vendor/security identifiers and exact event/date pairs. | Export/source/location, size and runtime unavailable; mapping cannot be synthesized from annual observations. |
| D4-predecessor / 2 | Identify original Form 4 for owner 1539147 among 14 same-date issuer candidates; then explicit replaces_lot_ids and source receipts. | 14 nominal full submissions / 56 with full retries / at most 112 including primary fallback retries. Runtime unknown; owner/lot linkage not guaranteed. |
| D5-exports / 3 | Existing detector flags-only export and raw referenced bytes; query proposal plus metadata event-date manifest. No known flags table is assumed. | One batched metadata SELECT proposed, zero executed; export source/location/size, nominal and worst runtime unavailable. No detector run allowed. |
| D6-population / 4 | Cost full PIT issuer discovery, inventories, fixed A9 selections, raw disclosures/carry-in before any expansion. Two-issuer diagnostic below is not a population cost or verified retrieval numerator. | Two-issuer full submissions: 1,544 nominal / 6,176 full retries / at most 12,352 with fallback. Full-population requests/runtime and carry-in costs unknown. |
| D7-vendors / 1 | Reconcile vendor search ledger and prior documentation stop; matched four-item evidence per route and common five-complete-year intersection for relevant mixed routes. | Spent/remaining hours and request/runtime costs unknown; cumulative ceiling remains two hours. |
| D8-pead / 4 | Frozen classification manifest before 30 delisted/acquired + 70 other A9 re-fetch selections; original release/issuer/fiscal match, actual EPS receipt/basis/currency/split basis, consensus pre-release vintages and complete member-quarter retrieval inventory. | Exact URLs/export source/fields lineage and nominal/worst runtime/requests unavailable; cannot collect or claim completed sample. |

Current cached-page recount, **filing-date scope only**, preserves NOC **848** rows (820 Form 4, 28 Form 4/A) and COST **720** (661 / 59). Annual observed counts and exact URL-generation/filter rules appear in `two_issuer_observed_footprint`. Missing full submissions now number 836 NOC + 708 COST = **1544** nominal full requests; 6176 with full retries, at most 12352 if every full response also needs separately retried primary fallback. This excludes carry-in and mapping/completeness costs, is not authorized collection, and is **not** a full-population cost forecast or a conclusion that budgets suffice/exceed limits.

The 14 predecessor candidates overlap the two-issuer footprint; do not add these component estimates as disjoint batches. Cost a union of exact uncached URLs before reserving a consolidated batch.

Before any cleared dependent work, hand off exact ordered URLs/event-security-date pairs, evidence/spec/source hashes, the immutable manifest/checkpoint/registration hashes, consumed claim identity and a new exclusive reviewed claim. Inherit 79/0/104.3013715306297 and STOP history without double-counting prefixes. Checkpoint response/claim outcomes and failures; a new directory is not a budget or stop reset. First new 403/429 stops calls. Missing nominal/worst costs or required inputs halt the affected acquisition proposal; no broad population batch is queued.

## Correction mapping and verification limits

| Review finding | Correction |
|---|---|
| 1 | Separate original/date-only/dissemination table and per-record sources; original quarantine retained. |
| 2 | Correct amendment dates/footnote; predecessor unknown; existing exclusion behavior reproduced. |
| 3 | Both corrected complete URL/file/hash bindings above; all workbook raw references matched against manifest. |
| 4 | Vendor spent/remaining null; vendor review and later SEC documentation STOP separated. |
| 5 | One executable paired LEFT JOIN SELECT with exact parameters, zero counts, yfinance/1d/horizon; venue omission conditional runtime follow-up. |
| 6 | Separate insider/PEAD denominators, annual counts and membership hash; prefix only insider. |
| 7 | Full cached owner/role/class/holding/lot metadata; duplicate, late/carry-in/boundary dependencies; separate release/actual/consensus requirements; eps_reported corrected. |
| 8 | Strict inventory failure retained; no tolerance or waiver promoted to completeness. |
| 9 | Correct annual accessions; dated extracts/locators; no annual continuity PASS or inferred legal explanation. |
| 10 | All nine four-item route receipts rebuilt from retained primary references with retrieval provenance and common-window gaps. |
| 11 | Sample checkbox absence distinct from compliance/effective dates and unknown schema introduction/participation. |
| 12 | 11-lot example, 6+2 groups, eight involved/six potential excess; all S, no measured DB or purchase-cluster loss. |
| 13 | Correct linked Exhibit 4.8 targets; bounded charter costs plus predecessor/mapping/export/vendor/population dependencies; no invented runtime. |
| 14 | Bounded schema-string search; byte/raw-locator/type/link checks only; custom workbook not formally schema-validated. |

Verification: JSON parses; required record fields/types checked; all raw source URL/cache/hash tuples checked against the manifest and bytes; all annual cover/symbol excerpts located in raw documents; relevant charter hrefs and amendment/collision locators inspected; counts reproduced with existing pure code; relative links and whitespace checked. Cache files were hashed before/after. No matching formal workbook validator was identified, so **no formal schema validation is claimed**. The registered final audit JSON/report remain untouched, and no new manual status substitutes for them.

Not checked: external services, database contents, price values/returns/P&L, detector execution or broad tests. Remaining risk: mandatory identity, inventory, reconstruction, tape and PEAD provenance inputs are still missing; neither signal has an established qualifying window.

## Retrieval-provenance review follow-up

Independent rereview cleared findings 1–9 and 11–14 at workbook scope, but found
remaining receipt-label errors under finding 10. Each vendor source now carries
its actual retained-review retrieval kind, exact receipt excerpt and receipt hash.
The ten indexed-only LSEG/Zacks/S&P/WSH/CRSP/Sharadar/Norgate sources are no
longer called successful page opens. Sharadar Insiders documentation and its
matched subscription schedule retain the successful-open evidence from the
review, while Nasdaq SF2 keeps indexed text distinct from its live JavaScript
shell. Bot interstitials, the CRSP 401 and old SEC PDF 404 are separately labeled.
These are retained receipts, not newly retrieved primary bytes or present-day
product verification. Route statuses, audit gates and request authority are
unchanged. Fresh independent Codex review is clean at this scope: all 38
source entries (37 unique URLs) across R1–R9 match their retained receipt
excerpts, hashes, retrieval kinds, dates, titles and sections. Prior findings
1–9 and 11–14 were cleared in the preceding review; finding 10 is now resolved.
This accepts the offline disposition workbook, not acquisition clearance or
completion of the registered feasibility audit.
