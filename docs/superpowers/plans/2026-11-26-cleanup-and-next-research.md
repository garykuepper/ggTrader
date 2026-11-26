# Clean-up and Next Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tidy the repository, database host and documents after the database consolidation, then write a short, reviewed agenda of the stock-selection studies worth running on the data already held.

**Architecture:** Part A is housekeeping plus three documents (a dated data inventory, a refreshed next-steps page, a refreshed research roster). Part B is two research documents: a ledger of which date windows earlier studies already used, and the agenda built on that ledger. Both parts end with a cross-vendor review. No new data loads, no new database writes, no study is run.

**Tech Stack:** git, `docker exec ggtrader_db psql` (SELECT only), Markdown, `scripts/run_tests.sh`, `ruff`.

**Spec:** [docs/next_steps.md](../../next_steps.md) (goal and standing rules), [database consolidation design](../specs/2026-11-25-database-consolidation-design.md), [Kaggle import runbook](../../kaggle_import_runbook.md), and the closed dead-stock studies under `research/stock-signals/dead-stock-*`. The owner's instruction (2026-11-26): "write a plan to perform clean up and draft up the next steps we want to research into or test strategies. seems like we got all the data we could get at this point."

## Global Constraints

- Production paper trader and its cron stay stopped; send no orders; never touch paper state.
- Never delete database backup copies, the `db-dumps` folder (including `attempts/`), `ggtraderdb-backup-*`, `ggtraderdb-dump-*`, `ohlcv_yf_preshift_backup_20260820`, `data/` or paper `*_backup_*` tables. Ask the owner first.
- Delete a branch only with `git branch -d` (merged only), never `-D`. Never push or merge without the owner's say; the owner said "push" for the 2026-11-26 work only.
- Talk to the owner in plain words; define each technical term in the same sentence. Never cite the old evidence rules (Part A, Holm and Newey-West bars, the excess-return threshold, B0/B1, six-month clean wait) as requirements. No new GO/NO-GO labels; report evidence and uncertainty.
- Every result cited pins both `--eval-start` and `--eval-end`; raw JSON sits next to the report. Subtract trading costs and taxes and compare with SPY on the same dates; taxable account only (no IRA suggestions).
- Never recommend retiring the sleeve, ending paper trading or shutting down the project.
- Tests only through `scripts/run_tests.sh`; one heavy job at a time; ruff clean.
- Quota: Claude weekly pool is nearly out (7% left on 2026-11-26). Builders: Codex (Terra or Sol); reviewers: Gemini Neon (Antigravity) and Codex Sol. No Astra or Opus without the owner's approval each time.
- Delegate prompts carry no private account records and no secrets.

## Review Focus

1. A task deletes or "cleans" something that is a backup copy (the 11 GB `db-dumps/attempts` folder looks like scratch but is not ours to remove). Expected: it stays; the plan lists it as an owner question.
2. A candidate study quietly reuses a date window already spent as development, holdout or fresh test. Expected: the window ledger names every spent window and the agenda marks each candidate with the window it may use.
3. A figure in the inventory or agenda is copied from memory or an old report. Expected: each figure re-measured with a SELECT the same day, or cited to its report with a locator.
4. Jargon reaches the owner (bps, t-statistic, holdout, WFO) without a same-sentence definition. Expected: a plain-words lint pass over both research documents.
5. The agenda ranks a study as promising without saying what result would change a decision. Expected: every candidate carries "what we would do differently if it works / if it does not".

---

### Task 1: Local housekeeping

**Files:** none changed in the repo.

**Interfaces:** Produces: a clean local state that Task 4 can describe (five merged branches gone, no scratch containers, scratch files removed).

- [ ] **Step 1: Confirm each branch is merged and has no unique commits.** Run `git branch --merged main` and `git branch --no-merged main`. Expected: the five names `database-consolidation`, `database-consolidation-design`, `db-dump-retry-and-coverage-exemption`, `dead-stock-early-filings`, `price-tables-protection-and-sources-view` appear under merged, and the unmerged list is empty.
- [ ] **Step 2: Delete them with `git branch -d <name>` one at a time.** Expected: each prints "Deleted branch". If any refuses, stop and report it; do not use `-D`.
- [ ] **Step 3: Confirm no scratch containers or temp files remain.** Run `docker ps -a --format '{{.Names}}' | grep -i scratch` (expect no output) and list `/tmp/claude-1000/-home-flynn-ggTrader/79fe0979-9c64-4a7d-8304-625abc97b77d/scratchpad/`; remove only files this session created (`deep_verify.*`, `full_gate_final.log`, `trig_test.sql`, `freeze_yahoo.py`) once their results are recorded in the changelog.
- [ ] **Step 4: Report, do not remove:** the 11 GB `/media/thesix/ggTrader-data/db-dumps/attempts/` folder (superseded dumps from the first restore drill). Put this question to the owner in Task 8: keep as a spare backup or remove.

### Task 2: Backup timer check and optional fresh dump

**Files:** none changed.

**Interfaces:** Consumes: `ggtrader-db-dump.timer` (Sundays 04:30, user service). Produces: a recorded yes/no that the weekly dump ran and the newest dump covers the SEC tables.

- [ ] **Step 1: Before the first Sunday run (2026-11-29 04:30), ask the owner whether to take a dump now.** Run `systemctl --user list-timers ggtrader-db-dump.timer` to show the next run. If yes: `scripts/run_capped.sh .venv/bin/python scripts/db_dump.py --out-dir /media/thesix/ggTrader-data/db-dumps/` (about 12 minutes), then the drill `scripts/run_capped.sh .venv/bin/python scripts/db_restore_drill.py` per the runbook. Expected: exit 0, `drill_report_<date>.json` has `"ok": true`.
- [ ] **Step 2: After 2026-11-29 04:30, verify the weekly dump.** Run `systemctl --user status ggtrader-db-dump.service` and list `db-dumps/`. Expected: a new `ggtrader_20261129.dump`, a matching line in `SHA256SUMS`, a `counts_20261129.json`, and no `.mismatch` file. If a mismatch exists, the script already retried once; investigate what wrote to the database during the dump.

### Task 3: Data-in-hand inventory

**Files:**
- Create: `docs/database-inventory-2026-11-26.md` (the 2026-11-23 inventory stays archived and untouched).

**Interfaces:** Produces: one page every later task and every study links for "what data exists and how to read it".

- [ ] **Step 1: Re-measure, SELECT only.** For each schema (`kaggle_crsp`, `kaggle_chars`, `market`, `sec`, `ref`, `universe`, `public.ohlcv` equity rows with `venue='yfinance'` and `interval='1d'`, `market.yahoo_daily_frozen_20261126`) record row count, date range, stored size (use `hypertable_size()` for hypertables), load number, coverage label and the sanctioned reader (`kaggle_crsp.daily_of`, `kaggle_chars.mcgill_of`, `sec.*_of`, `market.prices_daily_sources`, `load_bars()`). Save the raw query output next to the page as `docs/database-inventory-2026-11-26-measurements.txt`.
- [ ] **Step 2: Write the page** with four sections in plain words: what is in the database (one table: name, what it holds, rows, dates, coverage, how to read); what is downloaded but not loaded (Gu-Kelly-Xiu characteristics, insider trades, SEC company facts, SPY holdings, Quandl, Stooq, the small sets, Fama-French and Shiller series, the two intraday SPY sets), each with size and the reason it is not loaded; what is known to be unavailable for free (delisting returns, real bid-ask spreads, 1999-and-earlier volume in CRSP); and the known limits: Yahoo rows are overwritten in place (frozen copy exists), the two-load cap on saved responses, plain (not always-on) edit triggers on the three compressed price tables.
- [ ] **Step 3: Cross-check.** A second agent (Codex Sol) re-runs every query behind a figure and lists any mismatch. Expected: zero mismatches, or each one fixed in the page.
- [ ] **Step 4: Commit** `docs: dated database inventory 2026-11-26` with the measurements file.

### Task 4: Refresh the next-steps page, roster and memory

**Files:**
- Modify: `docs/next_steps.md` (status table and "To resume"), `docs/changelog.md` (one entry), `research/RESEARCH_SNAPSHOT.md` (regenerated), `README.md` and `docs/architecture.md` only where they name the database tables or the import stages.
- Modify (outside the repo): `/home/flynn/.claude/projects/-home-flynn-ggTrader/memory/MEMORY.md` and `project_database_consolidation_2026-11-26.md`.

**Interfaces:** Consumes: Task 1 and Task 3 results. Produces: a next-steps page whose table is true on 2026-11-26.

- [ ] **Step 1: Audit for stale statements.** Run `grep -n "2026-11-12\|paused\|clean slate" docs/next_steps.md` and list every row that is no longer true (the "paused as of 2026-11-12" line, the data rows, the dead-stock studies merged 2026-11-24 to 2026-11-25, the consolidation, the frozen Yahoo copy).
- [ ] **Step 2: Edit the table:** add rows for "Done: SEC data, labels and price-table protection loaded and pushed (2026-11-26)", "Done: dead-stock studies" (link each report), "Waiting on you" for the three open decisions from Task 8, and keep the intraday checkpoint row (slot 20, 2026-11-30) with a note that supervision is still paused. Keep the two unsent drafts and the AVB items as listed. Do not add retirement language.
- [ ] **Step 3: Regenerate the roster** with the `research-snapshot` skill (studies closed since 2026-11-20 are missing). Expected: the dead-stock studies appear with their reports.
- [ ] **Step 4: Check the repo.** `.venv/bin/ruff check .` clean, `scripts/run_tests.sh --fast` passes, `.venv/bin/python scripts/verify_pins.py` exits 0.
- [ ] **Step 5: Update memory:** one line in `MEMORY.md` for the inventory and window ledger once they exist; mark the consolidation note's open items that Task 8 closes.
- [ ] **Step 6: Commit** `docs: refresh next steps, roster and changelog after the consolidation`.

### Task 5: Window-use ledger

**Files:**
- Create: `research/programme/window-use-ledger.md`.

**Interfaces:** Produces: for each stock-level study, the date windows it used and how (development, holdout, fresh), plus a list of windows still unused. Task 6 reads it.

- [ ] **Step 1: Collect.** From each report under `research/stock-signals/` (`dead-stock-first-test`, `dead-stock-fresh-window`, `dead-stock-random-baseline`, `dead-stock-momentum-deaths`, `dead-stock-payout-rerun`, `dead-stock-early-filings`, `factor-screen`, `net-issuance`, `momentum`, and the other folders there) extract: question, data source, date windows, whether each window was development, holdout or fresh, and whether the owner's holdout claim files exist. Known starting points to verify, not to trust: development 2005-2019, holdout 2020-2024 (run once for each universe), fresh 2000-2004 (run once), all on the combined CRSP + Yahoo tape.
- [ ] **Step 2: Write the ledger** as one table (study, window, role, spent yes/no, source report with section locator) and a short plain-words paragraph: what "spent" means (a window that already shaped or scored a rule cannot give a clean test of a new rule tuned after seeing it).
- [ ] **Step 3: List windows not used by any stock-level rule.** Candidates to check, not assume: CRSP 1968-1999 (price, return and shares only; no volume, no delisting returns); Yahoo 2025-01 to 2026-11 (survivors only). State for each what it can and cannot show.
- [ ] **Step 4: Cross-check every figure and locator** with Codex Sol against the reports; fix mismatches. Expected: zero unsupported rows.
- [ ] **Step 5: Commit** `research: window-use ledger for the stock-level studies`.

### Task 6: Next research agenda

**Files:**
- Create: `research/programme/next-research-agenda-2026-11-26.md`.

**Interfaces:** Consumes: Task 3 inventory and Task 5 ledger. Produces: the shortlist the owner chooses from in Task 8.

- [ ] **Step 1: Write the agenda**, plain words, neutral voice, summary first. For each candidate give: the question in one sentence; why it follows from evidence already in the repo (cite the report); data it needs and whether it is loaded; the date window it may use (from the ledger) and how it would be pinned; the cost and tax treatment; what we would do differently if it works and if it does not; effort and which lane runs it. Use these candidates, in this order, and add others only with a cited reason:
  1. **Untouched decades:** run any one pre-registered, CRSP-only rule on CRSP 1968-1999, dead stocks included, with the owner's stated dead-stock reading assumption. Needs nothing new.
  2. **Slow it down:** the first test lost 3.4 to 5.6 points a year to costs and taxes; the same rule with yearly rebalancing and one-year holds (long-term gains rate). Window per ledger.
  3. **Death-aware momentum:** the momentum-deaths study found dying picks had 20% one-day jumps and a flat last month. This was seen in data already used, so test it only on an untouched window.
  4. **McGill screen:** `kaggle_chars.mcgill` (large stocks, 2000-2023) is loaded and has never been read; ask which of its measures rank next-month returns after costs and taxes. State how many measures are tried, because trying many raises the chance of a lucky one.
  5. **Forward check:** any rule that survives 1-4 gets one look at Yahoo 2025-01 to 2026-11, stating plainly that survivors-only data cannot show deaths.
  6. **Costs that match the stock:** needs volume (Quandl and Stooq, downloaded, not loaded). Only worth loading if 1-4 show something.
  7. **Merger-document search for the 2000-2004 deaths:** the re-score with current labels would change almost nothing (1 acquisition among 33 held deaths), so list it as available but not recommended.
  Add a parked list (PEAD, insider-trade studies, crypto, the intraday observer, paper parity) without recommending or ending any of them.
- [ ] **Step 2: Lint the language.** Run `.venv/bin/python scripts/check_reading_edition.py research/programme/next-research-agenda-2026-11-26.md` if present in the repo; otherwise re-read for undefined terms. Expected: zero findings.
- [ ] **Step 3: Commit** `research: next research agenda (draft for owner and review)`.

### Task 7: Independent review

**Files:** read-only.

- [ ] **Step 1: Two reviewers from different vendors**, Gemini Neon (Antigravity) and Codex Sol, each reads the inventory, ledger and agenda. Brief: verify every figure and locator against files and a read-only database; find reused windows, claims that cannot be tested with the data held, jargon, missing "what would we do differently" lines. Neither is told what the other found.
- [ ] **Step 2: Check each finding against the files yourself,** fix the confirmed ones, rerun `scripts/run_tests.sh --fast` and ruff, and commit `docs: review fixes for the inventory, ledger and agenda`.

### Task 8: Owner decisions

**Files:** none changed until the owner answers.

- [ ] **Step 1: Present in plain words:** (a) keep or remove the 11 GB `attempts/` folder; (b) take a fresh dump now or wait for Sunday (Task 2); (c) which one to three agenda candidates to run first, with the recommended order 1, 2, 3; (d) whether to push this work to `origin`.
- [ ] **Step 2: For each chosen candidate** write its question into a brief and scaffold it with `scripts/study.py new stock-signals <slug> --kind <kind> --placement <row>` per `research/README.md`; do not run it until the owner approves the brief. Record the choice in `docs/next_steps.md` and memory.

---

## Self-review notes

- Spec coverage: clean-up (Tasks 1, 2, 4), data picture (Task 3), research drafting (Tasks 5, 6), review (Task 7), owner gates (Task 8). The owner's "we have all the data" is reflected: no task loads data; Quandl/Stooq volume appears only as a conditional later step.
- Owner-gate items (backup folder, dump, push, which study) are not decided by this plan.
- Proportion: documents and checks only; no code is written, so no function bodies appear.
