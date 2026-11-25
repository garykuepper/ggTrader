# Early-Window Filings and Re-score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out from the SEC's free filings what the stocks that stopped trading in 2000 to 2004 were paid, then re-score the first test's momentum rule and the random-pick baseline on 2000 to 2004 with those findings, so that "dead stocks pay nothing" stops being a guess for the window where the rule looked worst.

**Architecture:** Two new closed studies, each a small audit or re-score with its own frozen brief, hash-checked copies of the first test's engine files, injectable drivers and synthetic-data tests (the pattern of `dead-stock-random-baseline` and `dead-stock-momentum-deaths`). Phase 1 classifies every stock in the first test's list that stopped trading inside 2000 to 2004 (251 stock deaths, 33 of them held by the rule) from EDGAR filings. Phase 2 reruns the rule and the 100+100 random draws of the random-baseline study on 2000 to 2004 with the classified stocks sold at their last price and the rest at the three readings.

**Tech Stack:** Python 3.12 in `.venv`, pandas/numpy, the project's `SecClient` pattern (cached, throttled, contact line from a user file), the read-only database (published release 1, CRSP load 1), `scripts/study.py`, `scripts/run_capped.sh`.

**Spec:** There is no separate spec. The plan implements the "free data / dead stocks" gaps of the owner's goal as recorded in: `research/stock-signals/dead-stock-momentum-deaths/2026-11-25-dead-stock-momentum-deaths.md` (section 4: no filings were searched for 2000 to 2004), `research/stock-signals/dead-stock-random-baseline/2026-11-25-dead-stock-random-baseline.md` (section 4: what dead stocks paid is unknown for 2000 to 2004), and `research/stock-signals/dead-stock-data-integrity/phase-2/2026-11-24-dead-stock-payout-evidence.md` (the method for 2005 to 2024, whose code is the model).

## Global Constraints

- Free data only. The owner refuses any purchase or subscription (EDGAR is free).
- The paper trader stays stopped. Nothing under `src/` is touched. The database is read only.
- Nothing is merged, pushed or deployed without the owner's say. Memory and OOM slice settings are unchanged.
- The owner's four plain rules apply to new work (test only on data the strategy never saw; use only what was known on each day and include stocks that later died; subtract costs and taxes and compare with SPY over the same dates; paper trade first). No pass bar and no GO/NO-GO labels. Talk in plain words with every term explained.
- The 2000 to 2004 window has been seen by the momentum rule. Phase 2 is therefore a re-score of a seen window and the report says so. Filings are outcome information: they set a sale price and never choose a stock.
- The owner's SEC contact line lives in `~/.config/ggtrader/sec_contact`. It is sent with every SEC request and is never written into any file, brief, command line, log or delegate prompt. Never put the owner's email anywhere.
- Builders and reviewers make no network requests and never run a driver against the real database or the SEC; the controller alone makes real runs and the SEC downloads, one heavy job at a time through `scripts/run_capped.sh` (set `GGT_RUN_MAX_SECONDS` above the expected run time). Tests run only as `scripts/run_tests.sh <small targeted paths>`.
- Be polite to the SEC: at most 5 requests a second, every response cached on the main thesix drive under `/media/thesix/ggTrader-data/sec-early-window/` (outside the repo, never deleted), a request the SEC answers "not found" is recorded and not repeated.
- Study rules: new code in the study's `code/` folder; engine files copied with a new prefix and checked by hash (a test); drivers call `bootstrap(__file__)`, refuse a dirty tracked tree, refuse to overwrite an existing result, write a `provenance` block, load in tests by `ggTrader.utils.study.load_driver`; every study ends with a frozen brief, a report in plain words, `PIN.json`, a changelog entry and a snapshot row.
- Delegation (owner's 2026-11-08 and 2026-11-24 preferences, sized to the quota the owner showed on 2026-11-25: Claude weekly 18% left, resets in about 1 day 19 hours; Codex weekly 82%; Gemini 100%): builders are Antigravity `gemini-4-neon`; reviewers are Codex `gpt-6.1-sol`; Opus only for one read of a final report if the owner says quota allows; the controller (Sonnet) writes briefs, makes the copies, runs the real jobs, writes the reports and re-verifies every slice. Report usage after each phase so the owner can stop early.

## Review Focus

1. **Old 8-K item numbers.** Before 23 August 2004 an acquisition is Item 2 (acquisition or disposition of assets), a change of control Item 1, a bankruptcy Item 3; afterwards 2.01, 5.01 and 1.03. A parser written for the new numbers calls every 2000 to 2003 acquisition "no event". Expected: both numbering schemes map to the same labels, and a test per scheme.
2. **A company that changed its name.** The name in our price data at the last trading day may differ from the name EDGAR indexes the filing under (the filer's former names are in the filing header). Expected: a match through a former name counts, is labelled lower confidence, and a near miss is never silently accepted.
3. **Last price is not the payment.** Stock-for-stock deals, cash deals below the last price, and a company that kept filing after our price series ended (a data break, not a death) all break "last price in full". Expected: the labels say "acquisition-type notice found", never "paid in full"; a company that kept filing for more than 180 days after the last day is labelled "kept filing" and is not classified as an acquisition.
4. **Outcome information leaking into choices.** A payout label used to decide which stocks are bought would be look-ahead. Expected: the re-score changes only the sale price of a stock that already died; the plan, the draws and their seeds are identical to the baseline's, and a test proves the plans are byte-identical with and without labels.
5. **The dice must get the same labels as the rule.** Labelling only the stocks the rule held would favour the rule against chance. Expected: the roster is every death in the list that window, and the same label table prices the rule's and the dice's dead stocks.

---

## Phase 1: what the SEC says about the 2000 to 2004 deaths

New study `research/stock-signals/dead-stock-early-filings` (audit, placement row 9, copy prefix `ef_`).

### Task 1: Frozen brief and scaffold

**Files:**
- Create via `scripts/study.py new stock-signals dead-stock-early-filings --kind audit --placement 9`: study folder, `study.toml`, README, brief.
- Create: copies `code/ef_*.py` of the first test's five engine files plus `code/frozen_copies.json` and `code/tests/test_frozen_copies.py` (same pattern, prefix `ef_`, import lines only).

**Interfaces:**
- Produces: the registered definitions every later task uses. Section 1 of the brief fixes: the roster (every stock in universe A, usable on a decision day, whose series ends inside 2000-01-03 to 2004-12-31, not a successor, not a file-end row: 251 expected, a check against `_early.json` of the momentum-deaths study); the filing window (30 days before the last trading day to 180 days after, as in the earlier study); the old and new item-number tables of Review Focus 1; the labels (`acquisition_type`, `bankruptcy`, `delisting_notice_only`, `kept_filing`, `no_event_found`, `no_company_match`, `name_unknown`) and the confidence tiers (`checked`, `name_only`, `former_name`); the planned output (a per-stock table and counts by label, window year and exchange).

- [ ] **Step 1: Scaffold and write the brief.** Run the `study.py new` command above; fill section 0 (eval 2000-01-03 to 2004-12-31, costs 0 bps, family size 1, seed none); write sections 1 to 7 as listed in Interfaces; state that nothing about payouts is measured and that a label is the SEC's classification, not a payment.
- [ ] **Step 2: Make the engine copies** with a script that applies only the import-line substitutions, writes `frozen_copies.json` with the original and copy hashes, and copies `tests/test_frozen_copies.py` from the momentum-deaths study with the prefix changed.
- [ ] **Step 3: Verify.** Run `.venv/bin/python scripts/study.py check research/stock-signals/dead-stock-early-filings` (expect `OK`) and `scripts/run_tests.sh research/stock-signals/dead-stock-early-filings/code/tests/test_frozen_copies.py` (expect all pass). Commit.

### Task 2: Death roster

**Files:**
- Create: `code/early_roster.py`, `code/early_inputs.py` (copy of `momentum_inputs.py` of the momentum-deaths study with `ef_` imports, window `early` only), tests `code/tests/test_early_roster.py`.

**Interfaces:**
- Consumes: `ef_replay.span_end_dates`, `ef_dead_stock_first_test.drop_file_end_rows` and `_normalise_spans`, `ef_universe_frames`.
- Produces: `build_death_roster(close: pd.DataFrame, eligible: pd.DataFrame, span_rows: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame` with one row per death: `column`, `permno`, `symbol`, `last_bar` (Timestamp), `decision_date`, `rank_band` (`"1-50"`, `"51-150"`, `"151-500"`, `"501+"` by momentum rank on the decision day that opens the death month), `held_by_rule` (bool: in the top 50 then). A `permno` comes from the release spans; a column with a `~k` suffix keeps its base symbol.

- [ ] **Step 1: Write failing tests.** Names and assertions: `test_roster_counts_the_deaths_in_a_synthetic_panel` (hand-made panel with 3 deaths, 1 successor, 1 file-end row: roster has exactly the 3 deaths); `test_rank_band_and_held_flag_follow_the_decision_day_ranking`; `test_roster_equals_the_momentum_deaths_definition` (on the momentum-deaths synthetic panel, the number of rows equals `A1_totals.deaths` of `momentum_deaths.compute_measures` run on the same panel, using a copy of that function's death definition, not an import).
- [ ] **Step 2: Run them; expect FAIL** (`scripts/run_tests.sh research/stock-signals/dead-stock-early-filings/code/tests/test_early_roster.py`).
- [ ] **Step 3: Implement** `build_death_roster` by reusing the decision-day and holding-month logic of `momentum_deaths.py` (copy the needed functions under this study's names; nothing imported across studies).
- [ ] **Step 4: Run; expect PASS. Lint and commit.**

### Task 3: Company names for the roster

**Files:**
- Create: `code/early_names.py`, tests `code/tests/test_early_names.py`.

**Interfaces:**
- Consumes: the roster of Task 2; the read-only connection of `ef_bars_reader`/`ggTrader.data.bars.connect_readonly`.
- Produces: `attach_names(conn, roster: pd.DataFrame) -> pd.DataFrame` adding `company_name` (str or None), `name_source` (`"mcgill"` or `"crsp"` or `""`), `exchcd`. Names come first from the McGill table (`kaggle_chars.mcgill_of(2)`, which covers 2000 to 2023) and, where absent, from the raw CRSP table's name field if it has one (the task reports which exists; if neither has a name the stock is `name_unknown`).

- [ ] **Step 1: Tests** with a fake connection returning small frames: McGill name preferred; CRSP fallback; no name gives None; the query uses the permno and the last-bar date and never a column of the other tables (assert on the SQL text, as the earlier roster tests do).
- [ ] **Step 2: Run; FAIL. Step 3: implement. Step 4: run; PASS; commit.**

### Task 4: SEC fetch, index search and filing-header parser

**Files:**
- Create: `code/ef_sec_client.py` (copy of `dead-stock-data-integrity/code/sec_client.py`, prefix only, with the cache folder default `/media/thesix/ggTrader-data/sec-early-window/` and the throttle set to 5 requests a second), `code/early_sec.py`, tests `code/tests/test_early_sec.py` and `code/tests/test_ef_sec_client.py` (copy of the client's tests adapted).

**Interfaces:**
- Produces: `quarters_for(last_bar: pd.Timestamp) -> list[tuple[int, int]]` (the quarters covering 30 days before to 180 days after); `find_filings(index_rows, company_name: str, former_names: Sequence[str]) -> list[Filing]` over the quarterly master index (form type, company name, CIK, date, file path), name-cleaned the way `sec_name_index.find_filers` cleans it; `parse_8k_items(header_text: str) -> frozenset[str]` returning item labels from either the old header lines (`ITEM INFORMATION:` followed by a label such as "Acquisition or disposition of assets") or the new numbered items; `label_items(items: frozenset[str]) -> str` mapping to `acquisition_type`, `bankruptcy`, `delisting_notice_only` or `other` using the tables frozen in the brief.

- [ ] **Step 1: Failing tests.** `test_old_header_acquisition_item_is_acquisition_type`, `test_old_header_bankruptcy_item_is_bankruptcy`, `test_new_numbered_items_2_01_and_5_01_are_acquisition_type`, `test_1_03_is_bankruptcy`, `test_unknown_items_are_other`, `test_quarters_for_a_december_last_bar_spans_the_next_quarters`, `test_name_cleaning_matches_inc_corp_and_punctuation_variants`, `test_former_name_match_is_marked_lower_confidence`, `test_a_company_that_kept_filing_after_180_days_is_marked_kept_filing`, plus the client tests (cache hit makes no request; 404 recorded and not repeated; throttle; the contact line is read from the environment path and never logged).
- [ ] **Step 2: Run; expect FAIL. Step 3: Implement** the signatures above (reuse the earlier module's name cleaning by copying it). **Step 4: Run; PASS. Commit.**

### Task 5: Classification and driver

**Files:**
- Create: `code/early_classify.py`, `code/run.py`, tests `code/tests/test_early_classify.py`, `code/tests/test_run.py`.

**Interfaces:**
- Consumes: Tasks 2 to 4.
- Produces: `classify_roster(roster, names, sec_lookup) -> pd.DataFrame` with `label`, `confidence`, `evidence` (a short string naming the form, date and items), and `suggested_loss` (0.0 for `acquisition_type`, otherwise empty: no figure); a driver `run.py` with `--mode {pilot,full}`, `--pilot-size N` (default 20, the first N roster rows by a fixed seed 20261125), `--release-id`, `--crsp-load` (both 1), refusing a dirty tree and an existing output, taking an injectable `sec_client`, `loader`, `git_dirty`, and writing `_pilot.json` or `_early.json` plus `_early_labels.csv` (one row per death) with the provenance block. Every SEC request goes through the cached client; a resume reads the cache.

- [ ] **Step 1: Failing tests:** classification of a hand-made roster with a fake SEC lookup covering every label and confidence tier; `test_label_counts_sum_to_the_roster_size`; `test_no_contact_line_in_the_output_file` (the string from the environment is absent from the JSON, the CSV and the log); driver end-to-end on synthetic inputs (pilot picks the seeded rows, refuses overwrite and a dirty tree).
- [ ] **Step 2: Run; FAIL. Step 3: implement. Step 4: Run; PASS. Lint. Commit.**
- [ ] **Step 5: Independent review** (Codex Sol, code only, no network): against the brief and the five Review Focus lines; fix Critical and Important findings; re-review only if Critical.

### Task 6: Pilot, then the full run (controller)

- [ ] **Step 1: Pilot.** `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-early-filings/code/run.py --mode pilot --release-id 1 --crsp-load 1`. Read the pilot's label counts and ten hand-opened filings. **Stop and ask the owner** if fewer than half of the pilot's stocks with a company name found any company on EDGAR, or if the header parser misses the items on the hand-opened filings: the method then needs another plan before 251 downloads.
- [ ] **Step 2: Full run** with `--mode full`. Expect about a thousand cached requests (23 quarterly indexes plus a header per candidate filing); the run resumes from the cache after an interruption.
- [ ] **Step 3: Check the numbers** (roster 251 and the 33 held by the rule matches the momentum-deaths result), write the report in plain words (counts by label and year, confidence tiers, what is not known), a Codex Sol check of every number against the CSV and JSON, fix, `study.py close` with the usual backup of the report, fill `PIN.json`, changelog entry, snapshot row, memory note, commit. Report usage so far to the owner. **Stop point: the owner decides whether Phase 2 goes ahead.**

---

## Phase 2: re-score 2000 to 2004 with the labels

New study `research/stock-signals/dead-stock-early-payout-rerun` (screen, placement row 9, copy prefix `er_`; a re-score of a seen window).

### Task 7: Frozen brief, copies and the exit-price table

**Files:**
- Create via `scripts/study.py new`: study folder; copies `code/er_*.py` of the five engine files from the **random-baseline** study (they include its one recorded ledger change), `code/frozen_copies.json`, `code/tests/test_frozen_copies.py`; `code/er_random_plans.py` (copy of the random-baseline `random_plans.py` with its imports renamed, hash-recorded the same way); `code/exit_prices.py`; `code/tests/test_exit_prices.py`.

**Interfaces:**
- Consumes: `_early_labels.csv` of Phase 1; `er_replay.exit_close(close, spans, loss)`.
- Produces: `labelled_exit_prices(close: pd.DataFrame, spans: pd.DataFrame, labels: Mapping[str, float | None], reading: float) -> pd.DataFrame`: closes where a stock labelled `acquisition_type` (suggested loss 0.0) is sold at its last price and every other dying stock at last price times (1 minus `reading`), `reading` in 1.0, 0.5, 0.0. The brief fixes: the cells (the rule, fresh draws, matched draws; three readings for the unlabelled stocks; 18.3 bps; low tax), the draws (the random-baseline study's seeds and counts: 100 per arm), the check first (with an empty label table the rule's three headline cells and SPY reproduce `_fresh-A.json`, and the first 5 draws reproduce the random-baseline study's `_early.json` draws to the cent), the output (the rule's gap and place among the draws at each reading with labels, next to the unlabelled numbers), and the plain statement that the rule and the dice share one label table.

- [ ] **Step 1: Write the brief and the copies** as in Task 1. **Step 2: Failing tests:** `test_labelled_stock_sold_at_last_price_unlabelled_at_reading`, `test_reading_applies_only_to_unlabelled_dying_stocks`, `test_no_labels_equals_exit_close` (byte-identical frames), `test_successor_stocks_unchanged`, `test_labels_never_change_the_plan` (the plan from `er_random_plans` is identical with and without labels, Review Focus 4). **Step 3: Run; FAIL. Step 4: Implement. Step 5: PASS. Commit.**

### Task 8: Driver

**Files:**
- Create: `code/run.py`, `code/rerun_summary.py`, tests `code/tests/test_run.py`, `code/tests/test_rerun_summary.py`.

**Interfaces:**
- Produces: `run.py --release-id 1 --crsp-load 1 [--draws N]` writing `_early_rescored.json` (per reading, per arm: every draw's gap to SPY, spread percentiles, the rule's place, plus the unlabelled-readings table); the identity and resume binding of the random-baseline driver (window, data, saved files, code hash, commit, lock, draw set); refuses if the check first fails. Summary functions copied from `dead_stock_random_baseline.py` under this study's names.

- [ ] **Step 1: Failing tests** adapted from the random-baseline driver's (check-first failure runs no draw; resume refusals; drift refusal), plus `test_labelled_and_unlabelled_results_are_both_written`. **Step 2 to 4: red, implement, green.** **Step 5: Sol review** of code against the brief (no real runs). Fix. Commit.

### Task 9: Real run, report, close (controller)

- [ ] **Step 1: Run** with `GGT_RUN_MAX_SECONDS=14400` through `scripts/run_capped.sh` on a clean tree (about 20 minutes plus the check). **Step 2: Report** in plain words: the rule's gap to SPY at each reading with labels, where it falls among the dice (with the same labels), how many of its 33 deaths and the dice's deaths were labelled, and what remains unknown (stocks with no figure, payment form, 1999 to 2004 filings the search could not read). A Codex Sol numbers check. **Step 3: Close** the study, pin, changelog, snapshot, memory, studies tier (`scripts/run_tests.sh --studies`), commit. Report total usage.

---

## Phase 3 and 4: decision points, not scheduled

- **Phase 3 (only if the owner asks):** classify every death in 2005 to 2019 and 2020 to 2024 (785 stocks) with the same machinery, whose 80 quarterly indexes are already cached, and re-score the random-baseline study's draws with the labels, so the rule and chance are compared with the same payout evidence in all three windows.
- **Phase 4 (only if the owner picks a different direction):** a slower-hold or different-kind rule study as a new frozen brief from the consensus report's steps 4 to 7; its own plan. Not planned here.

## Self-review

- **Spec coverage:** each stated gap has a task: no filings for 2000 to 2004 (Tasks 1 to 6); the rule versus chance under the same payout evidence (Tasks 7 to 9, Review Focus 5); the first-test best-reading question (Task 9 report). The owner's rules appear in Global Constraints.
- **Type consistency:** `build_death_roster` output columns feed `attach_names` and `classify_roster`; `labels: Mapping[str, float | None]` in Task 7 is built from the `suggested_loss` column of Task 5's CSV keyed by the roster's `column`.
- **Proportion:** the plan is about twice the length of the brief it will freeze; code bodies are left to the builder except the two frozen tables (item numbers, labels), which live in the brief.
