# Company-Measure Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, once, whether any of the loaded McGill company measures ranks next-month stock returns well enough to beat SPY after costs and taxes, with stocks that later died included.

**Architecture:** A new study folder `research/stock-signals/company-measures/` reuses the first test's engine as hash-checked `cm_` copies (the pattern `early-decades` used) and adds one new piece: a function that turns a McGill measure into a monthly stock ranking using only rows dated on or before each decision date. Task 1 is read-only and decides what can be tested; nothing is scored until the owner approves a written registration (Task 2).

**Tech Stack:** Python, pandas, `kaggle_chars.mcgill_of(2)` and `kaggle_crsp.daily_of(1)` (read-only), `load_bars(source="combined")`, `scripts/study.py`, `scripts/run_tests.sh`, `scripts/run_capped.sh`.

**Spec:** Agenda choice 4 in `research/programme/next-research-agenda-2026-11-26.md`; windows in `research/programme/window-use-ledger.md`; pattern from `research/stock-signals/early-decades/` (report `2026-11-28-early-decades.md`).

## Global Constraints

- Free data only, taxable account, information known on each date, stocks that later died stay in. Paper trader stays stopped; no orders.
- Pin `--eval-start` and `--eval-end` on every cited result; save raw JSON beside the report; report every tested measure and setting.
- Costs 0.183% per purchase or sale with 0.5% as a second reading; tax 24% on gains held a year or less, 15% on longer; dead-stock readings 0.0, 0.5 and 1.0. Compare with SPY on the same dates.
- McGill covers large stocks only, 2000-01-31 to 2023-12-29, and those years overlap windows already spent by other rules. The study is exploratory and must not call the period untouched. Do not import the Gu-Kelly-Xiu file (needs its own backup gate and owner approval).
- Plain words in every owner-facing text, define each term where it first appears. No GO/NO-GO labels. Do not cite the retired old evidence rules.
- Heavy jobs only through `scripts/run_capped.sh`, one at a time, overnight. Tests only through `scripts/run_tests.sh` with targeted paths. Nothing merged or pushed without the owner's say.

## Review Focus

- A measure built from yearly accounting numbers that was not yet public on its `date` (lookahead). Expected: the availability check reports when each measure's value changes through the year, and any measure that changes before it could have been public is dropped or lagged before registration.
- `date` versus `ret_eom` misalignment: the target return must be the return after the decision month. Expected: a check proves `stock_exret` covers the month after `date`.
- Stocks with missing measure values. Expected: they are skipped for that month, never filled, and never ranked on zero.
- Stocks that stop trading while held. Expected: sold on the last valid bar under the three loss readings, as in the first test.
- Trying many measures makes a lucky winner likely. Expected: the registration states the exact number tried, and the report shows all of them with that count next to every headline.

---

### Task 1: Scaffold the study and check what the data can support (read-only)

**Files:**
- Create: `research/stock-signals/company-measures/` via `scripts/study.py new stock-signals company-measures --kind screen --placement 9`
- Create: `code/cm_availability.py`, test `code/tests/test_cm_availability.py`
- Output: `_availability.json` (with a provenance block)

**Interfaces:**
- Produces: `load_mcgill_frame(load_id: int, columns: list[str]) -> pd.DataFrame` (read-only query through `kaggle_chars.mcgill_of`); `measure_coverage(frame: pd.DataFrame, measures: list[str]) -> pd.DataFrame` (rows years, columns measures, value = share of rows not null); `change_month_profile(frame: pd.DataFrame, measure: str) -> pd.Series` (index month 1–12, value = share of a stock's value changes that happen in that month); `timing_check(frame: pd.DataFrame) -> dict` with keys `rows`, `pairs`, `target_is_next_month_share` (share of consecutive-month pairs where a row's `stock_exret` equals the NEXT row's `ret_1_0` minus the row's `rf`, within 0.002; checked by hand on 2026-11-28 at 264,392 of 264,392 pairs, while `ret_eom` is only a month-end label and is not the target's date); `link_rate(mcgill_permnos: set[int], crsp_permnos: set[int]) -> float`.

- [ ] **Step 1: Write failing tests on toy frames:** `test_measure_coverage_counts_nulls`, `test_change_month_profile_flags_june_clustering` (a measure that only changes each June gives month 6 share 1.0), `test_timing_check_flags_same_month_target` (a row whose `stock_exret` equals its own month's return instead of the next month's gives `target_is_next_month_share` below 1.0), `test_link_rate_counts_unmatched`.
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/company-measures/code/tests/test_cm_availability.py`; expect FAIL (names undefined).
- [ ] **Step 3:** Implement the four functions and a `main()` that writes `_availability.json`: per-measure first year with at least 90% coverage, the change-month profile for every accounting-style measure, the timing result, the link rate to CRSP, the number of permnos whose last CRSP date is before 2023-12-29 (dead stocks), and the monthly count of stocks.
- [ ] **Step 4:** Rerun the tests; expect PASS. Run `main()` once through `scripts/run_capped.sh` (read-only, a few minutes, daytime is fine if no other heavy job runs).
- [ ] **Step 5:** Commit code and `_availability.json`; show the owner the table of usable measures, in plain words.

### Task 2: Owner decisions and the written registration

**Files:**
- Create: `research/stock-signals/company-measures/2026-11-28-company-measures-brief.md` (from `research/TEMPLATE-brief.md`), test `code/tests/test_registration.py`
- Modify: `research/programme/window-use-ledger.md` (add the decision row, not a score)

**Interfaces:** Produces the registered values later tasks read: `measures` (names and direction, high or low), `eval_start`, `eval_end`, `family_size`.

- [ ] **Step 1: Owner decisions (ask in plain words, with a recommendation):** (a) which measures, using only those that passed Task 1's timing and coverage checks; (b) the dates, proposed as the McGill span after a 12-month warm-up; (c) the selection rule, proposed as the first test's: top 50 of the loaded large stocks each month, equal weight, next-session trade; (d) whether Gu-Kelly-Xiu import is wanted later. Record the ledger decision that the span is exploratory.
- [ ] **Step 2: Write `test_registration_family_size_matches_measures`:** parses the brief; asserts `family_size == len(measures) * 2 * 3` and that `eval_start`/`eval_end` equal the pinned values.
- [ ] **Step 3:** Write the brief (rule, measures, count tried, dates, costs, tax, dead-stock readings, comparison, limits); run the test; PASS.
- [ ] **Step 4:** Commit. The commit hash is the proof the plan came before scoring.

### Task 3: Engine copy and the measure-to-ranking function

**Files:**
- Create: `code/cm_*.py` copies of the first test's engine (source `research/stock-signals/early-decades/code/ed_*.py`, renamed) with `frozen_copies.json`; `code/cm_signals.py`; tests `code/tests/test_cm_copies.py`, `code/tests/test_cm_signals.py`

**Interfaces:**
- Consumes: Task 2 `measures`.
- Produces: `rank_scores(frame: pd.DataFrame, measure: str, higher_is_better: bool, decision_date: pd.Timestamp) -> pd.Series` (index permno, value rank score; only rows with `date <= decision_date`, latest row per stock, missing values dropped); `scores_by_month(frame, measure, higher_is_better, decision_dates) -> pd.DataFrame` (dates by permno).

- [ ] **Step 1: Write failing tests:** `test_cm_copies_match_sources_by_hash`; `test_rank_scores_ignore_rows_after_decision_date` (a row dated after the decision date changes nothing); `test_rank_scores_drop_missing_not_zero`; `test_rank_scores_direction` (lower-is-better reverses order); `test_stale_row_over_one_month_old_is_dropped` (a stock whose latest row is older than 35 days is not ranked).
- [ ] **Step 2:** Run them; expect FAIL.
- [ ] **Step 3:** Copy the engine with hashes recorded; implement `cm_signals.py`.
- [ ] **Step 4:** Run the tests, `ruff check`, and `scripts/study.py check research/stock-signals/company-measures`; expect PASS.
- [ ] **Step 5:** Commit.

### Task 4: Driver with a one-shot lock, regression guard, then the overnight run

**Files:**
- Create: `code/cm_run.py`, test `code/tests/test_cm_run.py`

**Interfaces:**
- Consumes: Tasks 2 and 3. Produces: `_screen.json` (one cell per measure × cost × loss reading, with a provenance block) and `_screen.claim`.

- [ ] **Step 1: Write failing tests:** `test_claim_blocks_second_run`; `test_refuses_dirty_tree`; `test_refuses_dates_not_equal_to_registration`; `test_every_registered_measure_has_six_cells`.
- [ ] **Step 2:** Implement `main()` with `--eval-start`, `--eval-end`, `--out`; write the claim after the clean-tree check and before reading data; one claim covers the whole registered family.
- [ ] **Step 3 (regression guard):** Run the driver's engine on 2020-01-02 to 2024-12-31 with the first test's own momentum picks (not a McGill measure) and require all saved first-test cells to match exactly, as `early-decades` did. Run overnight through `scripts/run_capped.sh`. Stop and show the owner any difference.
- [ ] **Step 4:** Only if the guard matches, run the registered screen overnight, serially, through `scripts/run_capped.sh`. Verify every cell is present and the saved dates equal the registration.
- [ ] **Step 5:** Commit code; raw JSON and the claim stay git-ignored.

### Task 5: Report, records, reviews

**Files:** `research/stock-signals/company-measures/2026-MM-DD-company-measures.md` (template `research/TEMPLATE-research-report.md`), `research/programme/window-use-ledger.md`, `docs/changelog.md`, `docs/next_steps.md`, roster via the `research-snapshot` skill.

- [ ] **Step 1:** Report every measure and every setting against SPY over identical dates, with the number of measures tried beside each headline, the data limits (large stocks only, overlap with spent windows) and a plain "no label".
- [ ] **Step 2:** Update ledger, roster, next steps, changelog.
- [ ] **Step 3:** Two independent reviews from different vendors (Gemini 4 Neon and Claude Opus when the builder is Codex), each checking every figure against the raw JSON; fix findings; have both re-read the final text.
- [ ] **Step 4:** Close the study with `scripts/study.py close` (verdict UNRESOLVED), commit, and ask the owner before any merge or push.

## Self-Review

- Coverage: availability and timing (Task 1), ledger decision and count of tries (2), no-lookahead ranking (3), one-shot run with guard (4), report and review (5).
- Open risk: McGill is large stocks only and overlaps spent windows, so even a clean win is a lead for a later-data check (agenda choice 5), not a result to trade on.
