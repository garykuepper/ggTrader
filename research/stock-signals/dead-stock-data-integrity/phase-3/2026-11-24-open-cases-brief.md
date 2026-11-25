# Phase 3 brief: the 39 dead stocks that still have no evidence

**Date written:** 2026-11-24, before any Phase 3 code exists or runs. Nothing below was chosen after seeing a Phase 3 result.
**Study:** `research/stock-signals/dead-stock-data-integrity` (closed on 2026-11-24; this addendum reopens it for new files only).
**Why:** Phase 2 found SEC filing evidence for 102 of the 141 dead stocks the first test held. 39 have none. This phase tries once more, with a different way of finding the company, and writes down its rules first. Last time the second name rule (v1.1) was written after the first run showed its misses; this time the rules are fixed before the run.

## Words used (each said once, here)

- **SEC:** the US Securities and Exchange Commission, which keeps the public filing database (EDGAR). **Filer:** a company (or person) that sends documents to it. Each has a number, its **CIK**.
- **Quarterly index:** the SEC's list, one per quarter, of every filing made: who filed, which form, on what date. We already downloaded the 80 lists for 2005 Q2 to 2025 Q1 (the last one ends 2025-03-31) when we searched for Form 25 notices (the form an exchange files when a stock stops trading). They list every form type, not only Form 25. **Roster:** our list of the 141 dead stocks the first test held, with the company name we have for each (from the McGill data set, cut to 16 characters) and the stock's last trading day (`exit_date` in `_payouts.csv`). **PERMNO:** the university price database's number for one stock.
- **8-K:** the form a company files when something important happens (a sale of the company, a bankruptcy). Its "items" say what: 1.03 bankruptcy; 2.01, 3.03 or 5.01 a purchase, merger or change of control; 3.01 a delisting notice.
- **10-K and 10-Q:** the annual and the quarterly report a listed company must file. A trailing `/A` marks an amendment of one.
- **Deal forms:** `DEFM14A` (a merger vote notice to shareholders), `SC TO-T` (a tender offer: someone offers to buy shares), `SC 14D9` (the target's reply to a tender offer), `SC 13E3` (a going-private deal), and their `/A` amendments.
- **Suggested loss:** the share of its last price we assume a holder loses. 0.0 for an acquisition is the brief's assumed scenario, not a measured payment. For a bankruptcy the figure (0.30 NYSE or American stock, 0.55 Nasdaq stock) is taken from published research, not measured here; the table calls that basis `literature_fill`.

## What this phase may use

- Only the 39 rows of `_payouts.csv` whose `evidence_class` is `no_filing_match` (26), `name_unknown` (8) or `no_8k_found` (5). The other 102 rows keep every field value unchanged.
- Only free data: the cached quarterly indexes (**no** new download for them; if any of the 80 is missing or unreadable the run stops, it never searches fewer quarters), the SEC's own submission pages for matched filers (new downloads, at most 4 a second, by the controller only), and our own database tables for the one name probe in rule 7 (read only).
- Nothing is chosen from memory: a company name, a CIK or a deal is accepted only if a file we hold or an SEC page we download states it. No ticker, CUSIP or fuzzy matching.

## The rules (rule v2)

**Dates.** All day counts are calendar days, both ends included, counted from the stock's `exit_date` in `_payouts.csv` (nothing else defines "last trading day"). "A filing dated X" means the `Date Filed` of its line in the quarterly index.

1. **Name and activity on the same line (stocks of class `no_filing_match`, 26 rows).** A filer is a candidate for a stock when at least one index line of that filer, dated from 365 days before to 180 days after `exit_date`, carries a name equal to the stock's roster name under rule v1 (the Phase 2 name-cleaning function, called the same way, including the 16-character cut-off rule and the fixed abbreviation table). Only if rule v1 leaves **no** candidate does rule v1.1 run, over the same lines and the same dates (the same fallback order Phase 2 used: v1.1 never runs when v1 found a candidate). Exactly one candidate CIK: a match. Two or more different CIKs under the rule that ran: `ambiguous_match`, no pick. None under either rule: `no_filing_match`. Each match records which name rule found it (`v1` or `v1.1`) in the column `name_rule`; the overall `match_rule_version` is `v2` either way, and v1.1-only matches are counted apart. The name shown for a match is the name on the earliest qualifying line.
2. **The five `no_8k_found` stocks keep their Phase 2 match.** They already have a Form 25 notice and a matched CIK. Phase 3 does not rematch them; their `match_confidence` and `match_rule_version` stay as in `_payouts.csv` (two are `confirmed`). Only rule 5 (deal forms) may change their class.
3. **Still filing (applies to rule 1 matches only).** If the matched CIK has a `10-K`, `10-K/A`, `10-Q` or `10-Q/A` line in the quarterly indexes dated from 181 days after `exit_date` up to the last index day (2025-03-31), the match is rejected: class `no_filing_match`, `match_confidence` and `match_rule_version` left blank, `reason = company_still_filing: <CIK>`. This is an identity rejection, not proof: it can wrongly reject the true issuer, so every rejection is listed by name in the report. A stock whose `exit_date` + 181 days is after 2025-03-31 cannot be checked: the match is kept and the row carries `still_filing_check = incomplete`; all other matched rows carry `still_filing_check = done`. The test uses only the cached quarterly indexes, never the submission pages.
4. **8-K evidence (rule 1 matches that were not rejected).** The same evidence table, window and download client as Phase 2: any item 1.03 gives `bankruptcy_filed`; else any of 2.01, 3.03, 5.01 gives `acquired_or_merged`; else 3.01 gives `delisted_other`; else `no_8k_found`. If the 8-K list could not be fully read (Phase 2's status other than `ok` or `no_older_pages_needed`), the class stays `no_8k_found`, `reason = events_incomplete`, and rule 5 is not used for the class (the deal-form lines are still listed in `deal_form_accessions`).
5. **Deal forms (rule 4 gave `no_8k_found` with a complete 8-K list, and the five Phase 2 `no_8k_found` stocks).** If the matched filer has a deal form line dated from 365 days before to 30 days after `exit_date`, the class is the new `deal_forms_only`. It means only this: a listed form was filed by that filer in that window. Which stock, security or deal the form concerns is **not** read or checked. It **never sets a loss figure** (`suggested_loss` empty, basis `none`).
6. **Confidence.** Every rule 1 match has `match_confidence = name_only` and `match_rule_version = v2`, never `confirmed`: nothing besides the name ties the filer to our stock. Loss figures come from the existing `payout_table.suggested_loss` unchanged.
7. **The 8 unnamed stocks.** The probe is exactly the name lookup the roster already used (the McGill load 2 names and `ref.symbol_map` of release 1, by PERMNO and by date, in `roster.py`), run once more for the 8 stocks, read only. No new join, no use of the SEC index to name a stock (it has no key linking a PERMNO to a name). The result is written down. Whatever it returns, the class stays `name_unknown` unless a name appears, in which case that stock is treated as a rule 1 stock with the name's 16-character status taken from the source.

**Known limit, stated before the run.** A different filer with the same cleaned name that was alive near the exit and has no 10-K/10-Q after it (a private or foreign filer, a person, a fund) can still match. No issuer or security check is required and none will be added after seeing candidates. The name pairs printed in the report are the reader's check.

**Output.** `_payouts_v2.csv` and `_payouts_v2.json` hold all 141 rows in the order of `_payouts.csv`, with its 21 columns, then the added columns `phase` (1 or 3), `name_rule`, `reason`, `still_filing_check`, `deal_form_accessions`. For the 102 evidence rows every original field value is unchanged (the file is written fresh, so bytes may differ in line endings only; a test compares field values). `_payouts.csv` and `_payouts.json` are never written by Phase 3.

## What gets reported

- Counts by class before and after, and every move between classes.
- Every v2 match as a pair (our roster name, the SEC filer name), so the owner can read them.
- `deal_forms_only` reported apart from the acquisition evidence, never added to it.
- The statement that the first test's best case (every dead stock paid its last price) cannot change, because all 102 already sit at 0.0.
- Limits: identity is by name only; a deal form is an announcement, not a payment; every `company_still_filing` rejection printed by name.

## Not done in this phase

No strategy is rerun. No ticker or CUSIP lookup (EDGAR has no free CUSIP search). No file listed in `phase_provenance()` of the first two phases is edited.
