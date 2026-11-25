# Dead Stock Fresh Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the first test's exact 12-1 momentum rule, once, on five years it has never seen (2000-01-03 to 2004-12-31), and report the gap to SPY after costs and taxes at every loss, cost and tax setting.

**Architecture:** A new study `dead-stock-fresh-window` holds private `fp_`-prefixed copies of the first test's engine files (the study tool forbids two studies defining the same module name, so a prefix is the only way to copy). A thin driver reads the window from the database, runs the unchanged `run_study`, and writes one result file. Before the fresh run the driver must first re-run the already-seen 2020-2024 window and reproduce the first test's saved numbers to the cent (the "anchor"); the fresh window is then claimed and run once.

**Tech Stack:** Python, pandas, the first test's tax-lot ledger, PostgreSQL read-only (`connect_readonly`), `scripts/study.py`, pytest via `scripts/run_tests.sh`.

**Spec:** the frozen first-test brief `research/stock-signals/dead-stock-first-test/2026-11-23-dead-stock-first-test-brief.md` (rule, grid, accounting) and the new brief written in Task 1. Pattern to follow: `research/stock-signals/dead-stock-payout-rerun/code/` (`run.py`, `frozen_copies.json`, `tests/test_frozen_copies.py`, `tests/test_run.py`).

## Global Constraints

- Rule, universe A, grid and accounting are the first test's, unchanged: 12-1 momentum (`lookback=252, skip=21`), top 50, equal weight, universe A (common stock `shrcd` 10/11, NYSE/Nasdaq `exchcd` 1/3, previous month-end price at least $5, market-value rank 1,000 or better), loss 1.0/0.5/0.0, cost 6.4/18.3/40 bps per side, tax low (24%/15%) and high (35%/20%), start cash 100,000, SPY through the same ledger.
- Pinned inputs: `--release-id 1`, `--crsp-load 1`. The fresh window refuses any other value, a dirty tracked tree, or Yahoo data drift.
- No tuning, no payout evidence (the three loss readings only), no universe B, no new parameters.
- Window dates are registered before any return is computed: fresh 2000-01-03 to 2004-12-31; anchor 2020-01-02 to 2024-12-31 (the first test's holdout, already seen, used only to prove the code reproduces the first test).
- Study rules: `ggTrader.utils.study.bootstrap(__file__)` in the driver, no hand-rolled paths, no importing another study's `code/`, no duplicate module names across studies, a `MODULE_CALLS`-style allowlist test must still pass, pandas-3 `pct_change(fill_method=None)`.
- Tests only as `scripts/run_tests.sh <small targeted paths>`. Builders never run the real database, never touch the network, never run a whole tier. The controller alone runs the real anchor and fresh runs, one heavy job at a time.
- Never put the owner's email in any file, brief, command or prompt. No merge, push or deploy without the owner's say. The paper trader stays stopped. Memory and OOM slice settings unchanged.
- Report in plain words, no unexplained jargon; no GO/NO-GO label and no pass bar (owner rules reset 2026-11-20).
- Builder: Antigravity `gemini-4-neon`. Reviewer: Codex `gpt-6.1-sol` (a Gemini build is reviewed by Sol). One Opus read of the final report.

## Review Focus

- **SPY starts 2000-01-03 in our data.** A window that starts earlier cannot be benchmarked: `validate_window_inputs` (already in the copy) must reject it, and the driver must not offer a window starting before 2000-01-03. Test: a fake SPY series starting 2000-01-04 against start 2000-01-03 still passes the 7-day edge rule, one starting 2000-01-12 raises (Task 3).
- **Anchor mismatch must stop the fresh run.** Test: an anchor file with one cell off by one cent makes `fresh` exit 3 and write nothing, claim nothing (Task 5).
- **The fresh window can be run once.** Test: a second `fresh` call, and a call after a crash that left the claim file, both exit with the refusal naming the file (Task 5).
- **Outputs must never land inside the first test's or the rerun's folders**, nor over their files. Test in Task 5.
- **The dot-com bust deaths are bankruptcies more than buy-outs.** The first test's sentence "flat or rising price in the last month, so loss 1.0 is a worst-case bound" must not be reused. The report states what this window's price-path block shows (Task 8); a test in Task 4 checks the summary carries the block through unchanged.
- **Costs before 2001 were higher** (fractions, wide spreads). The 40 bps column is the relevant reading; the report says so and picks no cell afterwards.

## File Structure

Study folder `research/stock-signals/dead-stock-fresh-window/`:

| File | Responsibility |
|---|---|
| `2026-11-24-dead-stock-fresh-window-brief.md` | Frozen brief (Task 1). |
| `study.toml`, `README.md` | Study card (placement rule 9, kind `screen`). |
| `code/fp_replay.py`, `fp_ledger_run.py`, `fp_bars_reader.py`, `fp_universe_frames.py`, `fp_dead_stock_first_test.py` | Copies of the first test's engine files; only import lines change. |
| `code/frozen_copies.json` | Original and copy hashes and the exact import substitutions. |
| `code/fresh_inputs.py` | Window constants and `read_inputs` / drift check for any window. |
| `code/fresh_compare.py` | Anchor comparison and the gap summary. |
| `code/run.py` | Driver: `--window {anchor,fresh}`. |
| `code/tests/` | One test file per module above plus `test_frozen_copies.py`. |
| `_anchor-A.json`, `_fresh-A.json`, `_run_notes.txt` | Results (Tasks 6 and 7). |
| `2026-11-24-dead-stock-fresh-window.md` | Report (Task 8). |

---

### Task 1: Scaffold the study and freeze the brief

**Files:**
- Create: `research/stock-signals/dead-stock-fresh-window/` via the scaffold tool, then the brief `2026-11-24-dead-stock-fresh-window-brief.md`, `study.toml`.

**Interfaces:**
- Produces: the registered window and arm constants every later task reads: `FRESH = ("2000-01-03", "2004-12-31")`, `ANCHOR = ("2020-01-02", "2024-12-31")`.

- [ ] **Step 1:** Controller runs `scripts/study.py new stock-signals dead-stock-fresh-window --kind screen --placement 9`, removes the scaffold's placeholder module, keeps `code/run.py` and `code/tests/`.
- [ ] **Step 2:** Controller writes the brief from `research/TEMPLATE-brief.md`. It states: the question (does the first test's exact rule beat SPY after costs and taxes on 2000-2004, a window never run); the registered parameters line list (`eval_start: 2000-01-03`, `eval_end: 2004-12-31`, `max_leverage: 1.0`, benchmark SPY same ledger, costs 6.4/18.3/40 bps, `family_size: 1`, `seed: none`); the headline cell (loss 1.0, 18.3 bps, low tax, as in the first test) with every cell shown and none picked; the anchor rule; run-once rule; what it cannot tell (one five-year window and one market regime, the dot-com bust; SPY not in our data before 2000-01-03 so 1995-1999 is left out; costs before decimal pricing were higher so the 40 bps column is the relevant one; no filing evidence for 1999-2004 deaths so only the three readings; the momentum idea itself was examined on other data in earlier studies, so the idea is not unseen, only this exact stock-level rule and universe; payout of dying stocks unknown). It states what is NOT in this study: universe B, tuning, payout evidence.
- [ ] **Step 3:** Run `scripts/study.py check research/stock-signals/dead-stock-fresh-window`; expected: only complaints about missing code. Commit brief and card: `git commit -m "research: dead-stock fresh window brief (frozen before any code)"`.
- [ ] **Step 4:** Sol reads the brief once (fresh context, brief and the first-test brief only) for wording that overclaims or hides a limit. Controller fixes findings and commits before Task 2.

### Task 2: Frozen copies of the engine files

**Files:**
- Create: `code/fp_replay.py`, `fp_ledger_run.py`, `fp_bars_reader.py`, `fp_universe_frames.py`, `fp_dead_stock_first_test.py`, `code/frozen_copies.json`, `code/tests/test_frozen_copies.py`.

**Interfaces:**
- Consumes: the five originals in `research/stock-signals/dead-stock-first-test/code/`.
- Produces: importable modules `fp_*` with the same public names as the originals (`run_study`, `exit_close`, `run_book`, `run_spy`, `read_release_spans`, `read_close_batched`, `month_end_fields`, `eligibility`, `TaxParams`, `check_holdout_pins`, `claim_holdout`, `git_tree_dirty`, `write_trades_csv`).

- [ ] **Step 1:** Test `test_copy_differs_only_by_recorded_substitutions`: for each entry of `frozen_copies.json`, apply its substitutions to the original's text and assert the result equals the copy's text byte for byte, and that the recorded `original_sha256` equals the original file's hash and `copy_sha256` the copy's hash. Test `test_substitutions_are_import_lines_only`: every recorded substitution's source line begins `from ` and ends in `import (` or `import ...`. Model on `dead-stock-payout-rerun/code/tests/test_frozen_copies.py`.
- [ ] **Step 2:** Run it; expected FAIL (no copies).
- [ ] **Step 3:** Create the copies exactly as `dead-stock-payout-rerun/code/` did (`from replay import` becomes `from fp_replay import`, and so on for the three importing files); write `frozen_copies.json` with the same layout. Read the originals' hashes with `sha256sum`; do not edit the originals.
- [ ] **Step 4:** Run the test file with `scripts/run_tests.sh research/stock-signals/dead-stock-fresh-window/code/tests/test_frozen_copies.py`; expected PASS. Run `scripts/study.py check` on the study; expected no "module defined more than once".
- [ ] **Step 5:** Commit `research: fresh-window frozen engine copies`.

### Task 3: Window constants and input reading

**Files:**
- Create: `code/fresh_inputs.py`, `code/tests/test_fresh_inputs.py`.

**Interfaces:**
- Consumes: `fp_bars_reader`, `fp_universe_frames`, `fp_dead_stock_first_test as study` (`audit_false_deaths`, `WARMUP_CALENDAR_DAYS`, `CRSP_END`).
- Produces:
  - `WINDOWS: dict[str, tuple[str, str]] = {"anchor": ("2020-01-02", "2024-12-31"), "fresh": ("2000-01-03", "2004-12-31")}`
  - `SPY_FIRST_DAY = "2000-01-03"`
  - `read_inputs(window: str, release_id: int, crsp_load: int) -> dict[str, Any]` returning the same keys as the first test's `run.py read_inputs` (`close`, `spy_close`, `eligible`, `span_rows`, `release_spans`, `false_death_audit`, `data_provenance`), universe A only.
  - `read_drift(release_id: int) -> list[str]` and `drift_in(inputs: dict[str, Any]) -> list[str]` as in the first test's `run.py` (adapted copies; the first test's `run.py` is not importable).

- [ ] **Step 1:** Tests: `test_windows_are_registered_values` (the two tuples above, exactly); `test_fresh_window_not_before_spy_first_day` (`WINDOWS["fresh"][0] >= SPY_FIRST_DAY`); `test_unknown_window_raises` (`read_inputs("dev", 1, 1)` raises `ValueError` naming the window before any connection is opened, checked by monkeypatching `connect_readonly` to raise `AssertionError`); `test_spy_missing_before_start_is_refused` (build the fake frames, call `study.validate_window_inputs` with SPY starting 2000-01-12 and start 2000-01-03, expect `ValueError` containing "starts late").
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. `read_inputs` is the first test's `run.py read_inputs` with `args` replaced by the three parameters and `universe` fixed to `"A"`; use the `fp_` modules. `read_drift` / `drift_in`: copy the first test's logic (read its `run.py` lines 130-170) into this file.
- [ ] **Step 4:** Run `scripts/run_tests.sh research/stock-signals/dead-stock-fresh-window/code/tests/test_fresh_inputs.py`; expected PASS.
- [ ] **Step 5:** Commit `research: fresh-window inputs`.

### Task 4: Anchor comparison and gap summary

**Files:**
- Create: `code/fresh_compare.py`, `code/tests/test_fresh_compare.py`.

**Interfaces:**
- Produces:
  - `compare_to_saved(result: dict[str, Any], saved: dict[str, Any]) -> list[str]`: for every strategy cell (18) and SPY cell (6) of `saved["result"]`, compares `after_tax_final` of `result` to the saved unrounded value to within half a cent; returns one message per difference, and a message for any missing or extra cell key. Empty list means reproduced.
  - `gap_table(result: dict[str, Any]) -> list[dict[str, Any]]`: one row per strategy cell with `cell`, `loss`, `cost_bps`, `tax`, `cagr_gap_pts` (rule minus SPY, percentage points a year), `drawdown_gap_pts`, `sharpe_gap`.
  - `headline(result: dict[str, Any]) -> dict[str, float]`: keys `worst`, `middle`, `best` = `cagr_gap_pts` at cost 18.3, tax low, loss 1.0 / 0.5 / 0.0.
  - `price_path_block(result: dict[str, Any]) -> Any`: the result's dead-stock price-path block, returned unchanged.

- [ ] **Step 1:** Tests: `test_identical_results_compare_clean` (a result compared to itself gives `[]`); `test_one_cent_difference_is_reported` (change one cell's `after_tax_final` by 0.01, expect exactly one message naming that cell); `test_missing_cell_is_reported`; `test_headline_picks_the_registered_cells` (three known cells give the three numbers); `test_gap_table_has_18_rows_and_sign` (rule above SPY gives a positive `cagr_gap_pts`); `test_price_path_block_is_passed_through_unchanged`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement. Cell keys come from `fp_dead_stock_first_test.cell_key` and `spy_key`; the cell field names (`cagr`, `after_tax_final`, `max_drawdown`, `sharpe`) are read from `fp_dead_stock_first_test._cell`. The rerun's `payout_rerun.py` anchor code in `dead-stock-payout-rerun/code/` shows the comparison idiom; copy the idea, not the module.
- [ ] **Step 4:** Run the targeted test file; expected PASS.
- [ ] **Step 5:** Commit `research: fresh-window anchor comparison and gap summary`.

### Task 5: The driver

**Files:**
- Create: `code/run.py` (replace the scaffold's), `code/tests/test_run.py`.

**Interfaces:**
- Consumes: `fresh_inputs.read_inputs/read_drift/drift_in/WINDOWS`, `fresh_compare.compare_to_saved/headline`, `fp_dead_stock_first_test` (`run_study`, `claim_holdout`, `check_holdout_pins`, `git_tree_dirty`, `write_trades_csv`), `ggTrader.utils.study.bootstrap`, `ggTrader.utils.io`/`binding` for the provenance block as the rerun's `run.py` does.
- Produces: `main(argv=None, *, loader=None, results_dir=None, study_dir_=None, git_dirty=None, drift_preflight=None) -> int` (everything after `argv` is for tests); CLI `run.py --window {anchor,fresh} --release-id N [--crsp-load N]`; exit codes 0 ok, 2 refusal, 3 anchor mismatch. Writes `_anchor-A.json` or `_fresh-A.json` in the study folder and the trades side file in `results/dead_stock_fresh_window/`. The saved first-test file read for the anchor is `research/stock-signals/dead-stock-first-test/_holdout-A.json`.

Rules the driver enforces:
1. `anchor`: no claim, may repeat; runs window 2020-01-02..2024-12-31, compares to the saved first-test holdout with `compare_to_saved`; on mismatch writes nothing and exits 3; on a match writes `_anchor-A.json` with `anchor_ok: true` and the comparison.
2. `fresh`: refuses (exit 2) unless release 1, load 1, clean tracked tree, `_anchor-A.json` exists with `anchor_ok: true`, **and** its stored commit equals the current `git_head` of the code files (the code that passed the anchor is the code that runs the fresh window; any commit touching `code/` after the anchor invalidates it); refuses when `_fresh-A.json` or the claim file `fresh-window.claim` exists; refuses on Yahoo drift before and after the read (claim kept after the read, as the first test did); claims before reading any fresh data.
3. Output paths are fixed; refuse any path resolving inside `dead-stock-first-test/`, `dead-stock-payout-rerun/` or `dead-stock-data-integrity/`.
4. The result file carries `headline` (worst/middle/best), `gap_table`, the full `run_study` result, `window`, `release_id`, `crsp_load`, `git_head`, `binding`, `command`.

- [ ] **Step 1:** Tests (with an injected `loader` returning small synthetic frames and a `study_dir_` in `tmp_path`, as in `dead-stock-payout-rerun/code/tests/test_run.py`): `test_anchor_match_writes_anchor_file`; `test_anchor_mismatch_exits_3_and_writes_nothing`; `test_fresh_refused_without_anchor_file`; `test_fresh_refused_when_anchor_not_ok`; `test_fresh_refused_when_code_changed_since_anchor`; `test_fresh_refused_on_wrong_release_or_load`; `test_fresh_refused_on_dirty_tree`; `test_fresh_writes_result_and_claim_once`; `test_second_fresh_refused_naming_file`; `test_claim_left_after_crash_blocks_fresh` (loader raises, claim file remains, next call refuses); `test_fresh_drift_after_read_keeps_claim_and_writes_no_result`; `test_output_inside_first_test_folder_refused`; `test_help_exits_zero`.
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/dead-stock-fresh-window/code/tests/test_run.py`; expected FAIL.
- [ ] **Step 3:** Implement `main`. Pattern: the rerun's `run.py` for refusals and provenance, the first test's `run.py main` for claim ordering. Pin `git_head` by `git log -1 --format=%H -- <study>/code` so the anchor check ties to the code folder only.
- [ ] **Step 4:** Targeted tests PASS; then `scripts/study.py check research/stock-signals/dead-stock-fresh-window` passes; `scripts/run_tests.sh tests/scripts/test_scripts_have_help.py` passes.
- [ ] **Step 5:** Commit `research: fresh-window driver`.

### Task 6: Anchor run on the real database (controller)

- [ ] **Step 1:** Tree clean at the commit that holds all code. Run `.venv/bin/python research/stock-signals/dead-stock-fresh-window/code/run.py --window anchor --release-id 1 --crsp-load 1` under `scripts/run_capped.sh` (one heavy job at a time; expect a few minutes).
- [ ] **Step 2:** Expected: exit 0, `_anchor-A.json` with `anchor_ok: true`. Read every strategy and SPY cell against the first test's `_holdout-A.json` myself. If it exits 3, stop, report the cells that differ, and do not go on.
- [ ] **Step 3:** Write `_run_notes.txt` (commit, time, memory, command), commit `research: fresh-window anchor run reproduces the first test`.
- [ ] **Step 4:** Sol review (fresh context: diff of Tasks 2-5, the brief, the anchor file). Fix Critical and Important findings, rerun the anchor if any file under `code/` changed.

### Task 7: The fresh run (controller, after the owner's go-ahead)

**Pause:** the owner reads the frozen brief and says go. This run uses up the window; no repeat.

- [ ] **Step 1:** Confirm tree clean, anchor file present and its commit equals the code folder's last commit.
- [ ] **Step 2:** Run `... run.py --window fresh --release-id 1 --crsp-load 1` once, under the capped runner.
- [ ] **Step 3:** Check by hand: 18 strategy cells and 6 SPY cells; SPY zero-cost check passes; `headline` numbers match a recomputation from the stored cells; first and last trading days; dead-stock count by loss; price-path block. Append to `_run_notes.txt`. Commit `research: fresh-window result (run once)`.
- [ ] **Step 4:** If the run refused or failed after the claim, do not remove the claim; report to the owner, who decides.

### Task 8: Report and close

**Files:**
- Create: `2026-11-24-dead-stock-fresh-window.md` (from `research/TEMPLATE-research-report.md`), `README.md`; update `study.toml` (`status = "closed"`, `verdict = "UNRESOLVED"`, results list).

- [ ] **Step 1:** Write the report in plain words: the one-line answer; the headline cell and the worst/middle/best gap; all 18 cells beside SPY; risk (deepest fall, Sharpe) beside SPY; the price-path block for this window and what it does and does not say about buy-outs against failures; how this window compares to the two seen windows (their ranges are in the first test and rerun reports); the six limits in the brief; the statement that the window is now used and a second run would not be fresh.
- [ ] **Step 2:** Sol checks every number in the report against `_fresh-A.json` (fresh context, report plus JSON only). One Opus read for overclaiming and hidden limits. Fix findings.
- [ ] **Step 3:** `scripts/study.py close` (move any stale `PIN.json` first and add the report with `--input` as in the rerun's close; back up the report before using `--force`). Run `scripts/verify_pins.py`; update the expected pin count in `tests/scripts/test_verify_pins.py` (29).
- [ ] **Step 4:** Commit `research: fresh-window report and pin`.

### Task 9: Docs, gate, hand-off

- [ ] **Step 1:** Add a `docs/changelog.md` entry; an incremental update to `research/RESEARCH_SNAPSHOT.md` (one roster row, one gap); a memory note and `MEMORY.md` line.
- [ ] **Step 2:** `scripts/run_tests.sh --studies`; expected all pass (the study tier collects the new tests). Controller runs it alone.
- [ ] **Step 3:** Whole-branch Sol review; fix findings.
- [ ] **Step 4:** Hand-off message to the owner in plain words: result, usage (which models ran how often), rulings made with cost if wrong, decisions needed (merge and push; whether to extend with filing evidence for 1999-2004 deaths; whether to build the 1995-1999 part with a French-market benchmark). Nothing merged or pushed.

## Self-review

- **Spec coverage:** question and window (Task 1, 7), unchanged rule and grid (Task 2 copies), anchor (Tasks 4-6), run once (Task 5), limits and plain report (Tasks 1, 8), close and pin (Task 8), docs (Task 9).
- **Types:** `read_inputs(window, release_id, crsp_load)`, `compare_to_saved`, `headline`, `main(...)` names are used the same in Tasks 3-5.
- **Proportion:** only the driver rules and the window values are spelled out; bodies are left to the builder, with the rerun's `run.py` and the first test's `run.py` named as the models.
