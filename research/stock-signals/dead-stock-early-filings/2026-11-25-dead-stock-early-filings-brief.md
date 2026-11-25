# Brief: Dead Stock Early Filings

**Status:** FROZEN 2026-11-25, approved by the owner ("merge and push first then perform subagent", executing the plan `docs/superpowers/plans/2026-11-25-early-window-filings-and-rescore.md`). Frozen before any code of this study exists; later changes are dated amendments at the bottom, never silent edits.
**Study:** stock-signals/dead-stock-early-filings (kind: audit), written 2026-11-25.
**Decision fed:** whether a later re-score of the first test's momentum rule on 2000 to 2004 may treat the stocks that stopped trading as paid their last price (the best dead-stock reading) rather than guessing between three readings. The owner makes that decision. There is no pass bar and no GO/NO-GO label. No strategy is run here.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these values or say which line it departs from.

- eval_start: 2000-01-03
- eval_end: 2004-12-31
- max_leverage: 1.0
- benchmark: none (this study classifies stocks from filings; no account is simulated, no SPY comparison)
- costs: 0 bps per side (no trades are simulated)
- family_size: 1
- seed: 20261125 (used only to choose the pilot sample of section 1)

## 1. Question and fixed measures

**Question, in plain words.** The first test's momentum rule held stocks that later stopped trading, and what holders of such stocks received is unknown for 2000 to 2004 (a search of the SEC's filings was made for 2005 to 2024 only). For every stock in the first test's list that stopped trading from 2000 to 2004, what do the SEC's free filings say about how it ended?

**The roster.** Every stock of the first test's universe A (common stock on the NYSE or Nasdaq, price at least $5, among the 1,000 largest by company value on the previous month-end) that was usable on a month-end decision day (eligible, a finite momentum score, series not already ended) and whose price history ended in the holding month that opens on that decision day, with no successor series and not a row cut off at the end of the CRSP file; the same definition as the momentum-deaths study (`research/stock-signals/dead-stock-momentum-deaths`), which found 251 such deaths in this window (33 among the rule's 50 picks). One row per death. The roster check: the count must equal 251 and the number held by the rule 33; if not, the run stops and the report explains.

**Company names.** From the McGill table in our database (`kaggle_chars.mcgill`, column `comp_name`, with `permno` and `exchcd`; coverage 2000-01-31 to 2023-12-29), by the stock's `permno` on or before the last trading day. The raw CRSP table has no name column. A stock with no name is `name_unknown`.

**The search.** For each named stock, in the SEC's quarterly master index files (EDGAR `full-index/<year>/QTR<q>/master.zip`) for the quarters that cover the period from 30 days before its last trading day to 180 days after it: find filings whose company name equals the stock's name after the cleaning rule below (upper case, `&` to `AND`, punctuation removed, then dropping the words INC, CORP, CORPORATION, CO, COMPANY, LTD, LP, LLC, PLC, THE, and trailing HOLDINGS and GROUP only when another word remains; as in the data-integrity study's `sec_name_index.find_filers`). Also try the former names listed in the filing headers of candidates (confidence tier below). Forms read: 8-K and 8-K/A (current reports), Form 25, 25/A, 25-NSE and 25-NSE/A (notices of removal from an exchange), and any filing form that is 8-A12B/8-A12G and similar listings (ignored).

**What is read in a filing.** For each candidate 8-K, the item list from the filing header. Before 23 August 2004 the header has lines `ITEM INFORMATION:` followed by a label; from that date the items are numbered. Registered mapping (the pilot confirms the exact label text on real filings; any correction is a dated amendment before the full run):

| Era | Acquisition-type | Bankruptcy | Delisting notice |
|---|---|---|---|
| Before 2004-08-23 (labels) | Item 1 "Changes in control of registrant"; Item 2 "Acquisition or disposition of assets" | Item 3 "Bankruptcy or receivership" | none (use Form 25) |
| From 2004-08-23 (numbers) | 2.01, 5.01 | 1.03 | 3.01 |

An 8-K whose items are only others (for example "Other events", 8.01, financial statements) is `other`. Form 25 or 25-NSE filed for the company in the period counts as a delisting notice.

**Labels** (one per stock, the first that applies in this order): `bankruptcy` (a bankruptcy item); `acquisition_type` (an acquisition-type item); `delisting_notice_only` (a Form 25 notice and no 8-K item of the first two kinds); `kept_filing` (the company filed any form more than 180 days after the stock's last trading day: the price series ended without the company ending, so this is a data break and not a death); `no_event_found` (matched a company, no filing of the kinds above); `no_company_match` (named, no company found); `name_unknown` (no name).

**Confidence tiers** (one per stock with a company): `checked` (the filing's header names the same company by CIK that the master index gave, and the form is the one named); `name_only` (matched by cleaned name alone); `former_name` (matched through a former name in a filing header). A label from a `former_name` match is never promoted to `checked`.

**Suggested loss.** 0.0 for `acquisition_type`, otherwise no figure. This is the earlier study's convention: it means "treat the last price as what a holder received" as a scenario, not a payment amount. No payment amount or form (cash, stock) is read.

**Planned output.** A per-stock table (`_early_labels.csv`: column, permno, symbol, last trading day, company name, name source, exchange, rank band, held by rule, label, confidence, evidence string (form, date, items), suggested loss) and a results JSON with counts by label, by year, by exchange, by rank band and held-by-rule (all 251, and the 33), and the number of SEC requests made and served from the cache.

**Pilot.** Before the full run the controller runs a pilot on 20 roster rows chosen with seed 20261125 (the first 20 of a seeded shuffle of the roster ordered by permno). The pilot's purpose is to check that EDGAR holds the filings of these years, that the header text matches the mapping, and that names match. If fewer than half of the pilot stocks that have a name are found on EDGAR at all, or the header parser misses the items on filings opened by hand, the work stops and the owner is asked before the full run.

**Not in this study:** payment amounts, filing text beyond headers, any return or comparison with SPY, the 2005 to 2024 windows, stocks outside universe A.

## 2. Data and freeze before calculation

- **Prices, list, eligibility:** the first test's inputs, read only: release 1 and CRSP load 1, universe A. Every output records the release and price hashes. The engine files are copies of the first test's with the prefix `ef_`, checked against the originals by hash.
- **Filings:** the SEC's public archive (free). Every request carries a contact line the owner keeps in `~/.config/ggtrader/sec_contact`; it is never written into any file of this study. At most 5 requests a second; every answer cached under `/media/thesix/ggTrader-data/sec-early-window/` outside the repository; "not found" answers are recorded and not repeated.
- **Freeze:** this brief is committed before any code. Filings are outcome information: they say how a stock ended and are used only to describe it.

## 3. Registered model

None. The measures are counts and labels. The roster uses only what is known on each decision day to decide which stocks were usable; the death and the filings are the outcome being described.

## 4. Portfolio and accounting

None. No account, cost or tax is simulated.

## 5. Frozen output plan

The per-stock table and results JSON of section 1, and a short report in plain words: counts by label and confidence tier, by year and by exchange, the 33 stocks the rule held, what the labels do and do not mean, the number of stocks with no information, and the pilot's findings. Each count in the report is traceable to the table.

## 6. Decision rule

None. The owner's four plain rules apply.

## 7. What this cannot tell you, and the owner's checks

- **A label is the SEC's classification of a filing, not a payment.** `acquisition_type` does not say that holders received cash, or the last price, or anything in particular. Stock-for-stock and cash-below-market deals are possible.
- **Name matching can fail.** A renamed company, a successor that carries a different name, or a name our data spells differently is missed or mismatched. The confidence tiers and the `no_company_match` count show how much.
- **Before the SEC's August 2004 item renumbering the labels are text, not numbers.** A wording difference could hide an acquisition; the pilot and the hand checks of section 1 test this.
- **A stock that kept filing is not a death.** `kept_filing` marks price-data breaks; those stocks are not classified as acquisitions.
- **Small samples:** 251 stocks over five years; the counts by year and exchange are small.

**Execution boundary.** New study code goes in this study's `code/` folder; nothing in `src/` is touched, vectorbt is not imported, the paper trader stays stopped and the database is read only. SEC requests are made only by the controller. One heavy job at a time through the capped-run wrapper.

## Amendments

None yet.

**2026-11-25, first note (after the review of the parser, before any SEC request; no data seen).** The item mapping of section 1 is clarified in three ways. (1) Item *numbers* (2.01, 5.01, 1.03, 3.01) count only in a filing dated on or after 2004-08-23, the day the SEC's renumbered item list took effect; a bare number in an earlier filing, or in a filing with no readable date, is `other`. (2) Item *text labels* count in every era, because the SEC's header labels are text in both eras (the pilot confirms the exact wording on real filings): "Changes in control of registrant" and "Acquisition or disposition of assets" (also "Completion of acquisition or disposition of assets") are acquisition-type; "Bankruptcy or receivership" is bankruptcy; "Notice of delisting or failure to satisfy a continued listing rule or standard; transfer of listing" is a delisting notice. Labels are matched by their whole normalised text, never by a number or word found inside a longer description. (3) A number counts only when it stands as an item declaration (the whole value is an item code or a list of item codes), never when it appears inside a description such as "Other events (agreement section 2.01)".

**2026-11-25, second note (after the review of the classifier and driver, before any SEC request; no data seen).** Four rulings that make the search fail loudly or say so instead of quietly returning "nothing found", and settle a conflict between the label order of section 1 and the plan's rule that a company that kept filing is never classed as an acquisition.
1. **Two new labels.** `ambiguous_match`: two or more different companies (CIKs) in the index have the stock's cleaned name; no event is attributed to any of them and no figure is suggested. `evidence_unresolved`: a filing the classification needs could not be used (its header could not be read, was not available, or names a different company (CIK) or a different form than the index row); the evidence string says which. Neither label carries a suggested loss.
2. **Label order is now:** `ambiguous_match`, `bankruptcy`, `evidence_unresolved`, `kept_filing`, `acquisition_type`, `delisting_notice_only`, `no_event_found`, with `no_company_match` and `name_unknown` for stocks without a company. A resolved bankruptcy item outranks everything but an ambiguous match. A company that filed any form more than 180 days after the stock's last trading day is `kept_filing` even when an acquisition-type notice exists; the evidence string records the notice so the report can count them. This is the cautious choice: a real buy-out whose company keeps filing (for example because of public debt) gets no suggested figure instead of a possibly wrong one.
3. **The company is bound by its CIK.** Once the stock's name is matched to a CIK, only that CIK's filings are used, found in every quarter by CIK, whatever name they are indexed under; a header whose CIK or form differs from its index row is `evidence_unresolved`. The second pass that looked up other companies under a former name from a header is dropped: it could attach another company's event. The tier `former_name` stays defined but is unused in this run; a stock whose name changed between the filings and our price data is `no_company_match`, which the report counts and discusses (conservative: no figure).
4. **A missing quarterly index stops the run** (a not-found answer for a required index quarter, cached or not, is an error naming the quarter, never an empty search). A not-found answer for one filing header makes that stock `evidence_unresolved`.

**2026-11-25, third note (after the pilot, before any full run).** The pilot (20 stocks, the seeded sample of section 1; kept as `_pilot1.json` and `_pilot1_labels.csv`) found a company for only 6 of the 20 stocks. The cause is the company-name rule of section 1: the roster names come from McGill and are short forms ("MIDAMERICAN ENRG", "WACHOVIA CP NC", "NATL PROCESSING"), 77 of the 251 are cut off at 16 characters, and the exact-equality rule of section 1 is much stricter than the rule the earlier 2005 to 2024 study (`dead-stock-data-integrity`, "v1.1") used successfully. The pilot is the plan's stop point for this; I decided to repair the matching rule and rerun the (cached) pilot instead of stopping, because nothing else in the method failed. The outcomes of the 20 pilot stocks were seen before this amendment; the new abbreviations below were chosen from the roster's company names only, never from an SEC result. Matching rule from now on (same cleaning on roster and filer names):
1. **The earlier study's v1.1 rule**, copied: capitals, punctuation to spaces, whole-word short forms spelled out by that study's table (CP, CORPORATION, INCORPORATED, INDS, INTL, HLDGS, GRP, TECH, TECHS, SYS, PHARMACEUTICAL, DRG, STRS, AMERN, NATL, PHARMA, PHARMS, NAT, ELEC, MED, CONTL, EXPLOR, RES, PETE, COMM, SOLU, SOL, PAYMNT, SOUTHN, FINL, INS), a trailing EDGAR slash tag (`/DE/`, `/NEW`) removed, a trailing share-class marker removed, the fixed plural-to-singular folding table, then trailing corporate-form words dropped (last word never dropped); equal cleaned names match.
2. **The 16-character cut-off rule:** a roster name of exactly 16 characters may be cut off, so it matches a filer whose untrimmed cleaned name starts with the roster's untrimmed cleaned name.
3. **Added short forms**, spelled out on both sides: ENRG to ENERGY, CPTL to CAPITAL, GP to GROUP, SEMICOND to SEMICONDUCTOR, MANG and MGMT to MANAGEMENT, BROS to BROTHERS, ASSUR to ASSURANCE, FIN to FINANCIAL, 1ST to FIRST, HOLD to HOLDINGS, COR to CORP, PAC to PACIFIC, ENTERTAIN to ENTERTAINMENT, TECHNOL to TECHNOLOGY, COMMS to COMMUNICATIONS, NTWK to NETWORK, NTWKS to NETWORKS, SOFTWAR to SOFTWARE, DEVELOPME to DEVELOPMENT, PHARMACEUTI to PHARMACEUTICALS, LAB to LABORATORIES.
4. **A trailing state code on the roster name** (a last word that is one of the 50 US state postal codes, when the name has at least three words after cleaning, for example "WACHOVIA CP NC") is dropped on the roster side only.
5. **The guards stay:** two or more different companies (CIKs) matching one roster stock make it `ambiguous_match`; the evidence stays bound to the CIK; a name that matches nothing is `no_company_match`. A match by rules 2 to 4 is tier `name_only`, never `checked`.
If the rerun pilot still finds a company for fewer than half of the named stocks, the work stops and the owner is asked.

**2026-11-25, fourth note (after the review of the matching repair, before the rerun pilot).** The short form LAB to LABORATORIES, added in the third note, is withdrawn: combined with the earlier table's plural folding it gives LAB and LABS different spellings and could miss a match the earlier rule found. The earlier study's rule is kept for those words as it was. One roster stock (OPTICAL COAT LAB) is expected to go unmatched because of this; the report counts it as `no_company_match`. All other rules of the third note stand.

**2026-11-25, fifth note (after the rerun pilot, before the full run).** The rerun pilot found a company for 16 of the 20 stocks, so the plan's stop point is passed. Reading the cached indexes behind its `kept_filing` labels (names, forms and dates only, no payout) showed that most of them were false: the filings counted as "kept filing" were ownership filings made BY OTHER PARTIES that the SEC's index files under the subject company (Schedule 13G and 13G/A, Forms 4 and 4/A: for example NetScreen, Dallas Semiconductor and Security Capital Group), and one match was a different company created years later that reuses the name (Columbia Energy LLC, first filing 2004). Two rules are corrected for the full run:
1. **Kept filing** now counts only periodic reports filed by the company itself more than 180 days after the stock's last trading day: Forms 10-K, 10-K405, 10-KSB, 10-Q, 10-QSB and their amendments (`/A`) and the late-filing notices NT 10-K and NT 10-Q (and their variants). Ownership forms (Schedule 13D and 13G and amendments, Forms 3, 4, 5, 144), SEC letters (UPLOAD, CORRESP), deregistration forms (15 and its variants), listing forms, registration statements and current reports (8-K) do not count. The precedence of section 1's second note is unchanged: such a periodic-report filer is still `kept_filing`, ahead of `acquisition_type` and `delisting_notice_only`, with any acquisition notice recorded in the evidence. A company that went private with public debt and kept filing 10-Qs (Dole Food) therefore stays `kept_filing`, which gives it no suggested figure (the cautious choice of that note).
2. **A company must exist before it can be the listed company.** A candidate CIK counts as a match only if the index holds a filing by it (any form) dated on or before the stock's last trading day. The index range read is extended back to 1999 Q1 so that a stock that died in early 2000 and last filed in 1999 is not dropped for this reason. A name match with no earlier filing is not a match (it is as if no company had been found, `no_company_match`).

