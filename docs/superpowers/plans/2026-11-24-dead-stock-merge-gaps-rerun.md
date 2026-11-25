# Dead-Stock Merge, Open Cases and Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the data-integrity study on `main`, try to find SEC evidence for the 39 dead stocks that still have none, then rerun the first test with what we know about each dead stock instead of one guess for all of them.

**Architecture:** Three pieces in order. (A) Merge and push the finished branch. (B) "Phase 3" inside the closed data-integrity study, new files only: look the 39 open stocks up by company name in the SEC quarterly indexes we already downloaded (every filer, every form type), then read their 8-K filings and merger/tender-offer forms. It writes a second table, `_payouts_v2.csv`, and leaves the first table untouched. (C) A new study, `dead-stock-payout-rerun`, that replays the first test's exact rule, windows, costs and taxes (universe A only) and changes one thing: a dead stock with filing evidence is sold at its last price (or at the literature loss for a bankruptcy); every other dead stock keeps the old 1.0 / 0.5 / 0.0 readings.

**Tech Stack:** Python 3.12, pandas, the study's existing `SecClient` and filing readers, the first test's replay planner and tax-lot ledger (copied byte for byte), pytest through `scripts/run_tests.sh`.

**Spec:** No separate design doc. Two frozen briefs stand in for one: Task 2 writes the Phase 3 matching rules, Task 6 writes the rerun's registered plan. Questions come from the data-integrity report, [2026-11-24-dead-stock-payout-evidence.md](../../../research/stock-signals/dead-stock-data-integrity/phase-2/2026-11-24-dead-stock-payout-evidence.md), and the owner's "yes to all 3" of 2026-11-24 (merge and push; rerun the first test with per-stock outcomes; fill more gaps).

## In plain words, for the owner

- **Order is changed from your list:** merge first, then fill gaps, then rerun, because the rerun should use the best table we can build.
- **What the rerun can and cannot tell you.** 102 of the 141 dead stocks already sit at the 0.0 reading, which is the old best case. So the best case cannot change. What changes is the worst case: today it assumes all 141 lost everything; after the rerun it assumes only the stocks with no evidence did. It shrinks the range, it does not move the top of it.
- **It is a re-score, not a fresh test.** Both windows (2005–2019 and 2020–2024) were already looked at for this rule, and the data file ends 2024-12-31, so there is no unseen period. The report will say so on its first page. The first test's one-run lock files are not touched.
- **Universe B (the $1 stress list) is not rerun.** The payout table covers only the main list's 141 stocks.
- **Choices I made that you can overrule:** (1) the rerun copies the first test's engine files instead of moving them into a shared folder, so the closed first study and its fingerprints are not disturbed; (2) the headline uses only the 88 stocks whose notice was read and checked, and the name-only matches go in a labelled second set; (3) merger-proxy and tender-offer forms are treated as a weaker kind of evidence and never set a loss figure; (4) nothing is deleted after the merge, including the branch.

## Global Constraints

- Free data only (owner, 2026-10-23). SEC EDGAR is free and public.
- The first test's claim files `results/dead_stock_first_test/holdout-A.claim` and `holdout-B.claim` and its result files are never written, moved or deleted. The closed study `research/stock-signals/dead-stock-first-test/` is read as data only (its `_dev-A.json`, `_holdout-A.json`). No code is imported from any study's `code/` folder (study isolation rule); the four files Task 6 copies are copies.
- Pins: `release_id = 1`, CRSP load `1`, McGill load `2`, evaluation end `2024-12-31`.
- SEC fair access: at most 4 requests a second (the limit is 10); every request carries a User-Agent that names who is asking. The owner's contact line reaches the code only through the `SEC_USER_AGENT` environment variable (exported by `~/.bashrc`, saved in `~/.config/ggtrader/sec_contact`). It is never typed into a command, a repo file, a brief or a delegate prompt. To check it is set, run `test -n "$SEC_USER_AGENT" && echo set` and print nothing else.
- Only the controller (the session running this plan) makes SEC downloads or reads the database for real. Builders and reviewers write and test code against fixtures and make no network requests. Raw downloads go to `/media/thesix/ggTrader-data/sec-delistings/` (existing cache).
- Tests run only as `scripts/run_tests.sh <small targeted paths>`; heavy jobs only through `scripts/run_capped.sh`; one heavy job at a time (exit 75 means busy, not failed). The `--full` gate runs once, in Task 1, and `--studies` once, in Task 9. Put the "Rules for builders and delegates" block of `AGENTS.md` section 3 in every brief.
- Plain words in every owner-facing text; terms such as 8-K, CIK, Form 25 or bps get one sentence of meaning where first used.
- Nothing is merged, pushed or deployed except Task 1 (the owner's yes of 2026-11-24 covers only `dead-stock-data-integrity` into `main`). The trader stays stopped. Memory and OOM slice settings are not changed.
- Ruff must pass (`line-length = 100`). Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Files named in the data-integrity study's `phase_provenance()` code list (`sec_client.py`, `sec_filings.py`, `sec_pilot.py`, `payout_table.py`, `roster.py`, the two `_lib` readers, `utils/study.py`, `utils/io.py`) are **not edited**; Phase 3 adds new files only, so the Phase 1 and Phase 2 result files keep verifying.

## Delegation and usage (CLAUDE.md, 2026-11-08 and 2026-11-24)

Owner's usage screenshot of 2026-11-24: Codex weekly 84% left, Claude session 68%, Claude weekly 25% left (the scarce meter), Claude Fable weekly 80%, Antigravity Gemini 5-hour 86%, OpenCode 28%. So: building goes to Antigravity Gemini 4 Neon (Argon and Astra stay critical-only), every review goes to Codex Sol, and Claude Sonnet only writes briefs and reports and runs the real jobs. Opus is used once, at the end, and skipped if Claude weekly is under about 15% then (the owner decides). Pauses with a usage report: after Task 1, after Task 5, after Task 9.

| Task | Builder | Task reviewer |
| :--- | :--- | :--- |
| 1 merge, push | controller (Sonnet), no builder | none; the gate is the test run |
| 2 Phase 3 brief | controller (Sonnet) | Codex Sol reads the brief |
| 3 name index | Antigravity Gemini 4 Neon | Codex Sol |
| 4 open-case evidence | Antigravity Gemini 4 Neon | Codex Sol |
| 5 Phase 3 driver, run, report | Neon builds the driver; controller runs it and writes the report | Codex Sol; Neon third read |
| 6 rerun study scaffold, brief | controller (Sonnet) | Codex Sol |
| 7 per-stock prices, rerun grid | Antigravity Gemini 4 Neon | Codex Sol |
| 8 rerun driver, run, report | Neon builds the driver; controller runs it and writes the report | Codex Sol; Neon third read; Claude Opus once at the end |
| 9 close-out | controller (Sonnet) | Codex Sol final read of both studies |

A reviewer never reviews its own model family; the controller re-reads every diff and reruns every check itself and writes a ledger in `.superpowers/sdd/2026-11-24-dead-stock-merge-gaps-rerun/` (git-ignored, kept).

## Review Focus

The failure modes a straight reading would miss, most likely first. Each has a test in the task named.

1. **A different, living company with the same name** (name-only matching has no Form 25 to confirm identity): if the matched filer keeps filing 10-K or 10-Q reports more than 180 days after the stock's last trade, it is rejected as `company_still_filing`. Task 4.
2. **Two filers with the same name inside the window** (parent and subsidiary, old and new company): `ambiguous_match`, never a pick. Task 4.
3. **A dead stock whose price column is a later spell (`PERMNO-123~2`) or a Yahoo ticker** must pick up the right payout row, and a stock must never be pinned through another stock's row. Task 7.
4. **A held dead stock missing from the payout table** (the held set differs a little between cells because the cash path differs): it falls back to the reading, and the count and the names are printed. Task 7 and Task 8.
5. **A cached archive missing, cut short or over the size limit**: the name search stops loudly naming the quarter instead of searching less and reporting fewer matches. Task 3.

Plus one whole-study anchor: with no stock pinned, the rerun engine must reproduce the first test's saved final values to the cent (dev and holdout, middle cost, low tax, readings 1.0 and 0.0). Task 8.

## File Structure

| Path | One job |
| :--- | :--- |
| `research/stock-signals/dead-stock-data-integrity/phase-3/` | Phase 3 brief addendum, results, report. |
| `.../dead-stock-data-integrity/code/sec_name_index.py` (new) | Reads the cached quarterly archives of all filers and form types; finds filers by name; lists a filer's forms. |
| `.../code/open_cases.py` (new) | v2 matching, deal-form evidence, the merged 141-row table. |
| `.../code/open_cases_run.py` (new) | Driver: `--cache-dir`, `--out`, `--force`; writes `_payouts_v2.json` and `.csv`. |
| `.../code/tests/test_sec_name_index.py`, `test_open_cases.py`, `test_open_cases_run.py` (new) | Tests. |
| `research/stock-signals/dead-stock-payout-rerun/` (new study) | Brief, report, results, `study.toml`, `PIN.json`. |
| `.../code/replay.py`, `ledger_run.py`, `bars_reader.py`, `universe_frames.py`, `dead_stock_first_test.py` | **Byte copies** of the first test's files, never edited (Task 6). |
| `.../code/payout_prices.py` (new) | Which stocks are pinned, and the per-stock sale-price frame. |
| `.../code/payout_rerun.py` (new) | The rerun grid, the anchor check, the result JSON. |
| `.../code/run.py` (new) | Driver with `--window`, `--release-id`, `--crsp-load`, `--payouts`, `--out`. |

---

### Task 1: Merge and push the data-integrity branch

**Files:** none edited. Branch `dead-stock-data-integrity` (HEAD 65e56f4d, clean) into `main`.

**Interfaces:** Produces: `main` containing the study, pushed to `origin/main`; a new branch `dead-stock-gaps-and-rerun` cut from it for Tasks 2–9.

- [ ] **Step 1: Confirm the state.** Run `git status -sb`, `git rev-parse --short HEAD`, `git merge-base --is-ancestor main dead-stock-data-integrity && echo ff`. Expected: clean tree, `65e56f4d`, `ff`.
- [ ] **Step 2: Check no private contact line is in what will be pushed.** Run `git grep -n -i "gmail\.com" -- research/stock-signals/dead-stock-data-integrity docs/superpowers/plans docs/changelog.md AGENTS.md`. Expected: no output.
- [ ] **Step 3: Run the merge gate once:** `scripts/run_tests.sh --full` (about 7 minutes, 2 GiB or more). Expected: exit 0. If it fails, read which tests: a failure that also fails on `main` at 0b284cc5 (for example the missing real-tape file `ohlcv_20261106.parquet`) is reported to the owner, not fixed here; any other failure stops the merge.
- [ ] **Step 4: Fast-forward merge and push.** `git switch main && git merge --ff-only dead-stock-data-integrity && git push origin main`. Expected: push accepted; `git status -sb` shows `main` level with `origin/main`.
- [ ] **Step 5: Cut the working branch.** `git switch -c dead-stock-gaps-and-rerun`. Keep `dead-stock-data-integrity` (the owner decides on deleting it).
- [ ] **Step 6: Usage pause.** Report to the owner what the gate cost and ask for current meters before Task 2.

---

### Task 2: Freeze the Phase 3 rules (brief addendum)

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/phase-3/2026-11-24-open-cases-brief.md`
- Modify: `research/stock-signals/dead-stock-data-integrity/2026-11-24-dead-stock-data-integrity-brief.md` (one dated line pointing to the addendum; the frozen text above it is not edited)

**Interfaces:** Produces: the constants Tasks 3–5 implement. Written and committed **before any Phase 3 code runs**, because the v1.1 rule last time was written after seeing the first run and had to be disclosed.

- [ ] **Step 1: Write the addendum with exactly these rules.**
  - *Who:* the 39 rows of `_payouts.csv` whose `evidence_class` is `no_filing_match` (26), `name_unknown` (8) or `no_8k_found` (5). The 102 rows with evidence are copied through unchanged.
  - *Name index:* the 80 quarterly archives already in the cache (2005 Q2 to 2025 Q1), all filers and all form types, **no new download for them**. A filer is a candidate when its index name equals the stock's roster name under rule v1 or rule v1.1 (the same functions Phase 2 used). Passing candidates: at least one filing dated from 365 days before to 180 days after the stock's last trading day. Exactly one passing filer: a match; two or more: `ambiguous_match`; none: `no_filing_match`.
  - *Still filing:* a matched filer with a 10-K, 10-K/A, 10-Q or 10-Q/A dated more than 180 days after the last trade is rejected as `company_still_filing` (class `no_filing_match`).
  - *Evidence:* for a match, the same 8-K evidence table as Phase 2 (item 1.03 bankruptcy_filed; else 2.01/3.03/5.01 acquired_or_merged; else 3.01 delisted_other; else no_8k_found), same window and same `SecClient`. If still `no_8k_found`, deal forms `DEFM14A`, `SC TO-T`, `SC TO-T/A`, `SC 14D9`, `SC 14D9/A`, `SC 13E3`, `SC 13E3/A` filed by the matched filer from 365 days before to 30 days after the last trade give class `deal_forms_only`: a merger vote or tender offer was announced, completion not shown; **no suggested loss**.
  - *Confidence:* every Phase 3 match has `match_confidence = name_only` and `match_rule_version = v2`; never `confirmed` (no Form 25 notice stands behind it). Loss figures follow `payout_table.suggested_loss` unchanged.
  - *Unnamed stocks (8):* one measured probe of free name sources (McGill by permno, `ref.symbol_map`, the SEC index). A name is accepted only if a source states it; nothing is entered from memory. Expected result: still unnamed. The probe's result is recorded either way.
  - *Reporting:* counts by class before and after, every Phase 3 match listed by name pair for the owner to read, `deal_forms_only` reported apart, and the statement that the best case of the first test cannot change.
- [ ] **Step 2: Read it for plain words and for any rule that could be tuned after a result;** Codex Sol reads it cold. Fix and commit: `git add research/stock-signals/dead-stock-data-integrity && git commit -m "research: freeze Phase 3 open-case rules before any run"`.

---

### Task 3: Name index over the cached SEC archives

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/sec_name_index.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_name_index.py` (`pytestmark = pytest.mark.study`)

**Interfaces:**
- Consumes: `SecClient` (`sec_client.py`; the builder reads how it caches and how `sec_pilot.py` builds the quarterly archive URLs and the quarter window), `sec_master_index.parse_listing_archive` only as the format model (it keeps listing forms only and is not edited), `sec_filings.normalize_name` and the v1.1 name function.
- Produces:
  - `@dataclass(frozen=True) class IndexedFiling: cik: int; name: str; form: str; filed: date`
  - `class MissingArchive(RuntimeError)` naming the quarter.
  - `def parse_all_forms(content: bytes) -> list[IndexedFiling]` (same header check and size limits as the listing parser; keeps every form type).
  - `def find_filers(archives: Iterable[tuple[str, bytes]], wanted: Mapping[str, str]) -> dict[int, set[str]]`: pass 1; `wanted` maps a normalized name to the roster key that asked; returns CIK to the roster keys whose name it matched. Streams: never holds more than one parsed archive.
  - `def filings_of(archives: Iterable[tuple[str, bytes]], ciks: set[int]) -> dict[int, list[IndexedFiling]]`: pass 2; every filing of those CIKs, sorted by date.
  - `def cached_archives(client: SecClient, quarters: Sequence[str]) -> Iterator[tuple[str, bytes]]`: reads the cache only; raises `MissingArchive` for a missing or cut-short body and never downloads.

- [ ] **Step 1: Write the failing tests** (fixtures are small hand-built archives in the same format; builders use no network):
  - `test_parse_all_forms_keeps_10k_8k_and_defm14a`: a three-line archive returns three filings with the right form and date.
  - `test_find_filers_keeps_only_wanted_names`: ten unrelated names and one wanted one return exactly that CIK.
  - `test_filings_of_returns_all_form_types_sorted`: a wanted CIK with `10-K`, `8-K` and `DEFM14A` lines in two archives returns all three in date order.
  - `test_missing_archive_fails_loudly`: `cached_archives` on an empty cache raises `MissingArchive` whose message contains the quarter; the `open_counter` fixture shows zero network opens.
  - `test_oversized_or_truncated_archive_refused`: a body over the listing parser's size limit and a body with no valid header each raise.
  - `test_wrong_header_is_not_parsed_as_data`: an HTML error page saved as a body raises instead of returning an empty list.
- [ ] **Step 2: Run, expect failure:** `scripts/run_tests.sh research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_name_index.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the functions above in `sec_name_index.py`. Two passes over the archives so memory stays small (80 archives of up to 30 MB are never held together). Reuse the first-pass parse for the second by re-reading the cached bytes.
- [ ] **Step 4: Run the tests, expect PASS;** `ruff check` and `ruff format --check` on both files.
- [ ] **Step 5: Commit:** `git add research/stock-signals/dead-stock-data-integrity/code/sec_name_index.py research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_name_index.py && git commit -m "research: name index over cached SEC quarterly archives"`.
- [ ] **Step 6: Codex Sol review** of the diff with the Review Focus item 5 named; fix findings; re-run the tests.

---

### Task 4: Open-case matching and evidence (rule v2)

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/open_cases.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_open_cases.py`

**Interfaces:**
- Consumes: Task 3's `find_filers`, `filings_of`, `IndexedFiling`; `sec_filings.fetch_events_with_status`, `extract_8k_events`, `evidence_class`; `payout_table.suggested_loss`, `loss_decision`, `EVIDENCE_CLASSES`. Row dicts have the columns of `_payouts.csv` (`permno,symbol,window,universe,exit_date,last_close,stock_name,exchcd,price_label,evidence_class,match_confidence,match_rule_version,form25_accession,...,suggested_loss,basis,loss_note,cross_check`).
- Produces:
  - Constants `DEAL_FORMS: frozenset[str]`, `DEAL_DAYS_BEFORE = 365`, `DEAL_DAYS_AFTER = 30`, `CANDIDATE_DAYS_BEFORE = 365`, `CANDIDATE_DAYS_AFTER = 180`, `STILL_FILING_DAYS = 180`, `PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})`, `PHASE3_CLASSES = EVIDENCE_CLASSES + ("deal_forms_only",)`.
  - `def match_open_case(row: dict, filers: Mapping[int, list[IndexedFiling]], names: Mapping[int, set[str]]) -> tuple[str, int | None, str]`: returns `(status, cik, reason)`, status one of `matched`, `ambiguous_match`, `no_filing_match`.
  - `def classify_open_case(row: dict, cik: int, filings: list[IndexedFiling], events: list, events_status: str) -> dict`: the updated row (class, `match_confidence="name_only"`, `match_rule_version="v2"`, `suggested_loss`, `basis`, `loss_note`, `deal_form_accessions`).
  - `def merge_open_cases(phase1_rows: list[dict], phase3_rows: dict[str, dict]) -> list[dict]`: all 141 rows in the original order; asserts every row that had evidence in Phase 1 is byte-identical; adds a column `phase` (`1` or `3`).
  - `def phase3_coverage(before: list[dict], after: list[dict]) -> dict`: counts by class before and after, moves between classes, `deal_forms_only` apart, a list of every v2 match as (roster name, matched filer name).

- [ ] **Step 1: Write the failing tests:**
  - `test_company_still_filing_after_death_is_rejected`: a 10-K dated last trade + 181 days gives status `no_filing_match`, reason `company_still_filing`; + 180 days does not reject.
  - `test_two_live_filers_same_name_is_ambiguous`: two CIKs with the same name and a filing inside the window give `ambiguous_match` and no cik.
  - `test_filer_with_no_filing_in_window_is_not_a_candidate`: filings only 400 days before the last trade give `no_filing_match`.
  - `test_deal_form_window_edges`: a `DEFM14A` at last trade − 365 counts, − 366 does not, + 30 counts, + 31 does not.
  - `test_deal_forms_only_gets_no_loss`: 8-K path gives `no_8k_found`, a `SC TO-T` is present: class `deal_forms_only`, `suggested_loss` empty, `basis == "none"`.
  - `test_item_103_beats_deal_forms`: items 1.03 and a `DEFM14A` give `bankruptcy_filed` with 0.30 for `exchcd` 1 and 0.55 for `exchcd` 3.
  - `test_acquisition_item_gives_zero_loss_name_only`: item 2.01 gives `acquired_or_merged`, loss 0.0, basis `filing_evidence`, confidence `name_only`, rule `v2`.
  - `test_v2_match_is_never_confirmed`: no Phase 3 row ever has `match_confidence == "confirmed"`.
  - `test_merge_keeps_evidenced_rows_unchanged`: a 3-row fixture (one with evidence, one open that gains evidence, one open that stays open) returns the first byte-identical and `phase` values `1`, `3`, `3`; changing an evidenced row in `phase3_rows` raises.
  - `test_unavailable_events_are_counted_not_dropped`: `events_status != "ok"` leaves the class `no_8k_found` with the reason recorded, as Phase 2 does.
- [ ] **Step 2: Run, expect failure** (`scripts/run_tests.sh .../test_open_cases.py`).
- [ ] **Step 3: Implement** the functions above in `open_cases.py`. Match rule exactly as the Task 2 addendum; use `payout_table` for losses and do not restate its numbers.
- [ ] **Step 4: Run the tests, expect PASS;** ruff both files.
- [ ] **Step 5: Commit:** `git add research/stock-signals/dead-stock-data-integrity/code/open_cases.py research/stock-signals/dead-stock-data-integrity/code/tests/test_open_cases.py && git commit -m "research: open-case matching and evidence, rule v2"`.
- [ ] **Step 6: Codex Sol review** with Review Focus items 1 and 2 named; fix; re-run.

---

### Task 5: Phase 3 driver, real run, report

**Files:**
- Create: `.../code/open_cases_run.py`, `.../code/tests/test_open_cases_run.py`, `.../phase-3/2026-11-24-dead-stock-open-cases.md`
- Create (by the run): `.../_payouts_v2.json`, `.../_payouts_v2.csv`, `.../_form25_matches` is **not** touched
- Modify: study `README.md`, `study.toml` (`reports`, `results`), `_hand_checks.txt` (section 14), `docs/changelog.md`

**Interfaces:**
- Consumes: Task 3 and 4 functions; `_payouts.csv` and `_roster.csv` of this study (their SHA-256 are recorded and checked); `sec_pilot.phase_provenance` is **called** for the provenance base and extended with the new files' hashes.
- Produces: `main(argv: Sequence[str] | None = None) -> int` with `--cache-dir`, `--out`, `--force`; exit 2 when a precondition fails (missing archive, `SEC_USER_AGENT` unset, dirty tree outside the output names, output exists without `--force`).

- [ ] **Step 1: Write the failing tests** (fixtures, no network): `test_driver_refuses_missing_archive` (exit 2, names the quarter); `test_driver_refuses_unset_user_agent` (exit 2, message never contains a value); `test_driver_writes_141_rows_and_provenance` (rows count 141, `provenance` has git head, code hashes of the three new files, SHA-256 of `_payouts.csv` and `_roster.csv`, the Task 2 constants); `test_driver_never_writes_phase1_files` (`_payouts.csv`, `_payouts.json` hash unchanged after a run); `test_download_counts_are_reported_by_kind` (archives 0 downloads, submissions pages counted).
- [ ] **Step 2: Run, expect failure. Step 3: implement `open_cases_run.py`** (streams archives from the cache, builds `wanted` from the 39 open rows, fetches 8-K evidence only for matched CIKs through the existing client at 4 requests a second, writes JSON and CSV atomically with `ggTrader.utils.io`). **Step 4: tests PASS; ruff.** **Step 5: commit** (`git add ... && git commit -m "research: Phase 3 open-case driver"`).
- [ ] **Step 6: Codex Sol review; fix.**
- [ ] **Step 7: The controller runs the real job.** Check `test -n "$SEC_USER_AGENT" && echo set`. Run: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/open_cases_run.py --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force`. Expected: exit 0, 141 rows, 102 unchanged, archive downloads 0, new downloads limited to submission pages for matched CIKs.
- [ ] **Step 8: Controller hand-checks:** read every v2 match as a name pair and three filings of the largest-changing match; record in `_hand_checks.txt` section 14 and in the report; spot check three `deal_forms_only` rows against the filing index pages.
- [ ] **Step 9: Write the Phase 3 report** (plain words, table of classes before and after, the v2 match pairs, the `deal_forms_only` list apart, what the 8 unnamed probe found, the statement that the best case cannot change, limits: name-only identity, merger forms are announcements). Lint with `scripts/check_reading_edition.py` if present. Commit result files and report.
- [ ] **Step 10: Reviews:** Codex Sol on the report against the result files; Neon third read of the numbers. Fix; commit.
- [ ] **Step 11: Usage pause.** Report meters and the Phase 3 counts to the owner; the rerun (Task 6 on) uses `_payouts_v2.csv` whatever it shows.

---

### Task 6: Rerun study scaffold, byte copies and registered plan

**Files:**
- Create: `research/stock-signals/dead-stock-payout-rerun/` via `scripts/study.py new stock-signals dead-stock-payout-rerun --kind screen --placement 9`
- Create: `.../code/replay.py`, `ledger_run.py`, `bars_reader.py`, `universe_frames.py`, `dead_stock_first_test.py` (byte copies from `research/stock-signals/dead-stock-first-test/code/`), `.../code/frozen_copies.json`, `.../code/tests/test_frozen_copies.py`
- Create: `.../2026-11-24-dead-stock-payout-rerun-brief.md`

**Interfaces:** Produces: the engine modules importable by bare name in the new study (the first study's own pattern), and the frozen brief that Tasks 7–8 implement. The new study imports no code from the first study.

- [ ] **Step 1: Scaffold,** then replace the scaffold's placeholder module and keep `code/run.py` for Task 8.
- [ ] **Step 2: Copy** the five files with `cp`, write `frozen_copies.json` as `{filename: sha256}` from the copies.
- [ ] **Step 3: Failing test first** (`test_frozen_copies.py`): `test_copies_match_recorded_hashes` (each file's SHA-256 equals the JSON entry) and `test_copies_match_first_study_when_present` (same hash as the first study's file; skipped if that folder is absent). Run: expect PASS for the first, and FAIL for none if copied correctly; run once before copying to see it fail.
- [ ] **Step 4: Write the brief,** committed before any rerun code exists. It states: the question; that the rule, the universe A eligibility, the windows (2005-01-03 to 2019-12-31 and 2020-01-02 to 2024-12-31), top 50, costs 6.4, 18.3, 40 bps and both tax sets are exactly those of the first test's brief; that this is a **re-score of windows both already seen and not a fresh test**; the one change (pinned dead stocks sold at their table loss); the two evidence sets (`confirmed`: Phase 1 rows with `match_confidence == "confirmed"` and a suggested loss, the headline; `all`: every row of `_payouts_v2.csv` with a suggested loss, the sensitivity); the readings for every unpinned dead stock (1.0, 0.5, 0.0); the grid (2 sets × 3 readings × 3 costs × 2 taxes = 36 cells plus 6 SPY per window); the three numbers the report leads with, all at 18.3 bps and low tax, headline set: **worst case** (unpinned stocks at 1.0), **middle** (0.5), **best case** (0.0), each as a gap to SPY in points a year, beside the first test's old range (dev -16.4 to -2.0, holdout -7.3 to +1.6); no GO or NO-GO label; the anchor check; the SHA-256 of `_payouts_v2.csv`; that `deal_forms_only` rows are not pinned; universe B is not rerun; the first test's claim files are not touched.
- [ ] **Step 5: `scripts/study.py check`** on the new folder passes (card, brief, driver rules once Task 8 adds `run.py`; before that, only check what exists and note the rest). Commit: `git add research/stock-signals/dead-stock-payout-rerun && git commit -m "research: payout rerun scaffold, frozen engine copies and registered plan"`.
- [ ] **Step 6: Codex Sol reads the brief cold** for plain words, for any choice that could still be made after a result, and for whether the headline is fixed. Fix; commit.

---

### Task 7: Per-stock sale prices and the rerun grid

**Files:**
- Create: `.../dead-stock-payout-rerun/code/payout_prices.py`, `.../code/payout_rerun.py`
- Test: `.../code/tests/test_payout_prices.py`, `.../code/tests/test_payout_rerun.py`

**Interfaces:**
- Consumes: the copied `replay.exit_close(close: pd.DataFrame, spans: pd.DataFrame, loss: float) -> pd.DataFrame` (the close frame with each ending stock's last bar replaced by its sale price; successor spells are not written down), `ledger_run.run_book`, `run_spy`, `TaxParams`, and the helpers of `dead_stock_first_test.py` (`validate_window_inputs`, `yearly_turnover`, `month_end_series`, `_cell`, `coverage_block`, `decomposition_for_loss`, `curve_summary`-based stats). The builder reads how `run_study` (lines 801–993 of the copy) wires them and **does not edit the copies**.
- Produces:
  - `READINGS: tuple[float, ...] = (1.0, 0.5, 0.0)`, `EVIDENCE_SETS = ("confirmed", "all", "none")` (`none` is only for the anchor).
  - `def base_symbol(column: str) -> str`: strips a `~k` spell suffix.
  - `def pinned_losses(rows: Sequence[Mapping[str, str]], evidence_set: str) -> dict[str, float]`: base symbol to suggested loss; `confirmed` needs `match_confidence == "confirmed"` and a numeric `suggested_loss`; `all` needs only a numeric `suggested_loss`; `none` returns `{}`. A symbol with two rows with different losses raises.
  - `def exit_prices(close: pd.DataFrame, spans: pd.DataFrame, reading: float, pinned: Mapping[str, float]) -> pd.DataFrame`: `exit_close(close, spans, reading)` with the final-bar value of every pinned column taken from `exit_close(close, spans, that column's pinned loss)`. Only a dying column's last bar may differ from the reading's frame.
  - `def unpinned_held_dead(trades: Iterable, pinned: Mapping[str, float]) -> list[str]`: base symbols of held dead stocks with no pinned loss.
  - `def cell_key(evidence_set: str, reading: float, cost: float, tax: str) -> str` and `def run_rerun(inputs: dict, rows: Sequence[Mapping[str, str]], *, window: str) -> dict`: runs the grid for universe A, returns cells, SPY cells, a coverage block per cell group (pinned and unpinned dead stocks held, and the unpinned names), and the anchor block.

- [ ] **Step 1: Write the failing tests** (small synthetic frames, five stocks, three of which die; no database):
  - `test_exit_prices_pins_only_the_last_bar_of_a_pinned_dying_stock`: pinned stock `AAA` (loss 0.0) with reading 1.0: its last bar equals its close, every other cell equals the reading-1.0 frame.
  - `test_exit_prices_unpinned_stock_follows_reading`: `BBB` unpinned, reading 0.5: last bar is half the close.
  - `test_later_spell_column_maps_to_base_symbol`: column `PERMNO-5~2` picks up the row for `PERMNO-5`, and a living column with a similar name is untouched.
  - `test_successor_spell_is_not_written_down_even_if_pinned_loss_is_positive`: a `has_successor` span keeps its last price, as in `exit_close`.
  - `test_pinned_losses_confirmed_excludes_name_only` and `test_pinned_losses_all_includes_name_only_but_not_blank_loss` (a `deal_forms_only` row with empty `suggested_loss` is never pinned).
  - `test_conflicting_duplicate_symbol_raises`.
  - `test_unpinned_held_dead_lists_missing_symbol` (Review Focus 4).
  - `test_empty_pin_set_reproduces_scalar_exit_close` (the frame equals `exit_close(close, spans, reading)` exactly).
  - `test_run_rerun_cell_keys_and_counts` (a tiny two-month run returns 36 strategy keys and 6 SPY keys, and the coverage block names the unpinned stocks).
  - `test_rerun_never_writes_the_claim_folder`: `run_rerun` and the driver helpers touch no path under `results/dead_stock_first_test`.
- [ ] **Step 2: Run, expect failure.** `scripts/run_tests.sh research/stock-signals/dead-stock-payout-rerun/code/tests/test_payout_prices.py research/stock-signals/dead-stock-payout-rerun/code/tests/test_payout_rerun.py`.
- [ ] **Step 3: Implement** `payout_prices.py` (splice per distinct pinned loss value; do not loop over rows for the price frame) and `payout_rerun.py` (reuse the copied helpers; the anchor block runs `none` at readings 1.0 and 0.0, 18.3 bps, low tax, and records the final values beside the first test's saved ones for the caller to compare).
- [ ] **Step 4: Tests PASS; ruff both files.**
- [ ] **Step 5: Commit:** `git add research/stock-signals/dead-stock-payout-rerun/code && git commit -m "research: per-stock sale prices and the rerun grid"`.
- [ ] **Step 6: Codex Sol review** with Review Focus items 3 and 4 named; fix; re-run. The controller reads `exit_prices` itself line by line, because it moves money in the replay.

---

### Task 8: Rerun driver, real run, report

**Files:**
- Create: `.../dead-stock-payout-rerun/code/run.py`, `.../code/tests/test_run.py`, `.../2026-11-24-dead-stock-payout-rerun.md`
- Create (by the run): `_rerun-dev-A.json`, `_rerun-holdout-A.json`, and the trade side files under `results/dead_stock_payout_rerun/` (not tracked by git)
- Modify: `study.toml`, `README.md`, `docs/changelog.md`

**Interfaces:**
- Consumes: Task 7's `run_rerun`; `ggTrader.utils.binding.binding`, `git_sha`; `ggTrader.utils.io.write_json`; `bootstrap` and `study_dir` from `ggTrader.utils.study`; the first test's saved `_dev-A.json` and `_holdout-A.json` as data for the anchor.
- Produces: `main(argv=None, *, loader=None) -> int`; flags `--window {dev,holdout}`, `--release-id`, `--crsp-load` (default 1), `--payouts PATH` (default the integrity study's `_payouts_v2.csv`), `--out PATH`, `--force`. Reads close frames and spans through the copied `bars_reader` and `universe_frames` exactly as the first test's `run.py` does (the builder reads `read_inputs`, lines 52–130 there, and the pre-run pin checks `check_holdout_pins`, drift preflight and clean-tree rule); it needs the release pinned to 1, load 1, a clean tree except the output names, no Yahoo drift, and the `_payouts_v2.csv` SHA-256 from the brief. It claims nothing and writes no claim file.

- [ ] **Step 1: Write the failing tests:** `test_refuses_wrong_release_or_dirty_tree` (exit 2); `test_refuses_payouts_with_a_different_hash` (exit 2); `test_refuses_existing_output_without_force`; `test_output_has_provenance_anchor_and_both_evidence_sets` (fixture loader); `test_driver_has_pinned_eval_end` (the study check's rule); `test_no_write_under_first_test_results_folder`.
- [ ] **Step 2: Run, expect failure. Step 3: implement `run.py`. Step 4: tests PASS; ruff; `scripts/study.py check research/stock-signals/dead-stock-payout-rerun`.** **Step 5: commit** (`git add ... && git commit -m "research: payout rerun driver"`). **Step 6: Codex Sol review; fix.**
- [ ] **Step 7: The controller runs development first,** through the guard: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --window dev --release-id 1 --crsp-load 1 --out research/stock-signals/dead-stock-payout-rerun/_rerun-dev-A.json --force`. Expected: exit 0. **Anchor:** the `none`-set final values at 18.3 bps, low tax equal the first test's saved cells to the cent (reading 1.0: 27,108; reading 0.0: 243,212, rounded as in its report). A mismatch stops the task: find the cause before running the holdout, never adjust a tolerance.
- [ ] **Step 8: Run the holdout** the same way with `--window holdout` (anchor: 128,547 and 193,187). Then read both result files: counts of pinned and unpinned dead stocks held in every cell group, the unpinned names, and the three lead numbers of the brief.
- [ ] **Step 9: Write the report** (plain words; first page says: re-score of seen windows, not a fresh test; best case unchanged by construction; the three gaps against the old range for the headline set; the sensitivity set beside it; the unpinned names; sources and limits: loss 0.0 is the brief's assumed scenario for acquisitions, stock-fund returns of the 14 name-only leads, cash-path differences, universe B not rerun; no GO/NO-GO label). Lint with `scripts/check_reading_edition.py` if present. Commit result files and report.
- [ ] **Step 10: Reviews:** Codex Sol against the result files and the brief (did the report change any rule after seeing numbers?); Neon third read of every figure; Claude Opus once, numbers and conclusions only, skipped if the owner says Claude weekly is too low. Fix; commit.

---

### Task 9: Close-out

**Files:** `study.toml` of both studies, `PIN.json` of both, `docs/changelog.md`, `docs/next_steps.md` if it lists this work, `research/RESEARCH_INDEX.md` if the study tool requires a row, `research/RESEARCH_SNAPSHOT.md` (regenerate with the `research-snapshot` skill, never by hand), the memory note, the ledger.

- [ ] **Step 1:** `scripts/study.py check` on both studies; `scripts/study.py close` for both (reopen the data-integrity card first if the tool requires it), then `.venv/bin/python scripts/verify_pins.py`. Expected: FAIL=0 and a reproducibility line for each.
- [ ] **Step 2:** `scripts/run_tests.sh --studies` once. Expected: exit 0, collected count at or above `STUDIES_MIN_COLLECTED`.
- [ ] **Step 3:** Changelog entry for Phase 3 and the rerun in the style of the 2026-11-24 entries (what, numbers, files, "nothing in `src/` changed; the trader stays stopped"). Regenerate the research snapshot. Update the memory note and `MEMORY.md` line (both exist: edit, do not duplicate).
- [ ] **Step 4:** `git grep -n -i "gmail\.com"` over the new files returns nothing. Commit.
- [ ] **Step 5:** Final Codex Sol read of both studies together (whole-branch), then the plain-words hand-off to the owner: results, usage, rulings, and the two questions left (merge and push this branch, and whether to extend the evidence search beyond names, which would need a source we do not have). Nothing is merged or pushed.

---

## Self-review

- **Coverage:** merge/push (Task 1), fill gaps (Tasks 2–5), rerun with per-stock outcomes (Tasks 6–8), close-out (Task 9). The three owner decisions each map to tasks.
- **Placeholders:** none; every constant, window and key is fixed in the plan or in the briefs written before code runs. The only things left to the builders are function bodies and reading how the existing client and the first test's `run_study` and `read_inputs` are wired, which the Interfaces blocks point at.
- **Types:** `IndexedFiling`, `find_filers`, `filings_of` (Task 3) are what Task 4 consumes; `pinned_losses`, `exit_prices`, `unpinned_held_dead`, `run_rerun` (Task 7) are what Task 8 calls; `EVIDENCE_SETS` names match the brief's `confirmed` and `all`.
- **Proportion:** about 300 lines against two ledger-and-SEC studies of roughly 4,000 lines of existing code; code blocks are signatures and test names only.
