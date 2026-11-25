# Phase 3 brief: the 39 dead stocks that still have no evidence

**Date written:** 2026-11-24, before any Phase 3 code exists or runs. Nothing below was chosen after seeing a Phase 3 result.
**Study:** `research/stock-signals/dead-stock-data-integrity` (closed on 2026-11-24; this addendum reopens it for new files only).
**Why:** Phase 2 found SEC filing evidence for 102 of the 141 dead stocks the first test held. 39 have none. This phase tries once more, with a different way of finding the company, and writes down its rules first. Last time the second name rule (v1.1) was written after the first run showed its misses; this time the rules are fixed before the run.

## Words used (each said once, here)

- **Filer:** a company (or person) that sends documents to the SEC. Each has a number, its **CIK**.
- **Quarterly index:** the SEC's list, one per quarter, of every filing made: who filed, which form, on what date. We already downloaded the 80 lists for 2005 Q2 to 2025 Q1 when we searched for Form 25 notices (the form an exchange files when a stock stops trading). They list every form type, not only Form 25.
- **8-K:** the form a company files when something important happens (a sale of the company, a bankruptcy). Its "items" say what: 1.03 bankruptcy; 2.01, 3.03 or 5.01 a purchase, merger or change of control; 3.01 a delisting notice.
- **Deal forms:** `DEFM14A` (a merger vote notice to shareholders), `SC TO-T` (a tender offer: someone offers to buy shares), `SC 14D9` (the target's reply to a tender offer), `SC 13E3` (a going-private deal).

## What this phase may use

- Only the 39 rows of `_payouts.csv` whose `evidence_class` is `no_filing_match` (26), `name_unknown` (8) or `no_8k_found` (5). The 102 rows with evidence are copied through unchanged, byte for byte.
- Only free data: the cached quarterly indexes (no new download for them) and the SEC's own submission pages for matched filers (new downloads, at most 4 a second, by the controller only).
- Nothing is chosen from memory: a company name, a CIK or a deal is accepted only if a file we hold or an SEC page we download states it.

## The rules (rule v2)

1. **Name match.** A filer is a candidate for a stock when the filer's name in the index equals the stock's roster name under rule v1 or under rule v1.1 (the same name-cleaning functions Phase 2 used, called the same way, including the 16-character cut-off rule and the fixed abbreviation table). No other loose matching. No ticker, CUSIP or fuzzy matching.
2. **Alive then.** A candidate must have at least one filing, of any form, dated from 365 days before to 180 days after the stock's last trading day. Exactly one such filer: a match. Two or more different CIKs: `ambiguous_match`, no pick. None: `no_filing_match`.
3. **Still filing.** A matched filer with a `10-K`, `10-K/A`, `10-Q` or `10-Q/A` dated more than 180 days after the last trading day did not stop existing: the match is rejected, class `no_filing_match`, reason `company_still_filing`. (A filing exactly 180 days later does not reject.)
4. **8-K evidence.** For a match, the same evidence table, window and `SecClient` as Phase 2: any item 1.03 gives `bankruptcy_filed`; else any of 2.01, 3.03, 5.01 gives `acquired_or_merged`; else 3.01 gives `delisted_other`; else `no_8k_found`. An 8-K list that cannot be fully read keeps the Phase 2 status labels and is counted, never dropped.
5. **Deal forms.** If rule 4 gives `no_8k_found`, any deal form (the list above, including `/A` amendments of `SC TO-T`, `SC 14D9` and `SC 13E3`) filed by the matched filer from 365 days before to 30 days after the last trading day gives the new class `deal_forms_only`: a merger vote or tender offer was announced, completion is not shown. **It never sets a loss figure** (`suggested_loss` empty, basis `none`). Day offsets are inclusive: day −365 and day +30 count, −366 and +31 do not.
6. **Confidence.** Every Phase 3 match has `match_confidence = name_only` and `match_rule_version = v2`. None is ever `confirmed`, because no Form 25 notice stands behind it. Loss figures come from the existing `payout_table.suggested_loss` unchanged (0.0 for `acquired_or_merged` with basis `filing_evidence`; 0.30 for a bankruptcy of an NYSE or American stock and 0.55 for a Nasdaq stock, basis `literature_fill`).
7. **The 8 unnamed stocks.** One measured probe of free name sources: the McGill names by permno, `ref.symbol_map`, and the SEC index. A name is accepted only if a source states it. The probe's result is written down even if it finds nothing (the expected result).

## What gets reported

- Counts by class before and after, and every move between classes.
- Every v2 match as a pair (our roster name, the SEC filer name), so the owner can read them.
- `deal_forms_only` reported apart from the acquisition evidence, never added to it.
- The statement that the first test's best case (every dead stock paid its last price) cannot change, because all 102 already sit at 0.0.
- Limits: identity is by name only; a deal form is an announcement, not a payment.

## Not done in this phase

No strategy is rerun. No ticker or CUSIP lookup (EDGAR has no free CUSIP search). No file listed in `phase_provenance()` of the first two phases is edited.
