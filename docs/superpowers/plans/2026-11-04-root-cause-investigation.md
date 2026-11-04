# Root-Cause Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed, plain-language diagnosis of why ggTrader's stock and crypto work has not beaten SPY by much, including the measured ceiling, without changing any live state.

**Architecture:** Two small tools first (a memory-capped runner and a report checker), then a committed inventory, then nine read-mostly lenses run by Claude subagents (Sonnet orchestrates, Opus judges, Haiku extracts), an owner-approved re-run phase, and an Opus synthesis challenged by a fresh Opus agent. Every lens report must pass the checker before the lead accepts it.

**Tech Stack:** Python 3.12 (repo `.venv`), pytest via `scripts/run_tests.sh` only, `systemd-run --user --scope`, Claude subagents (`claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5`), markdown reports under `docs/research/`.

**Spec:** `docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` (approved 2026-11-04). Starting evidence: `docs/research/2026-11-04-phase0-trust-audit.md` (unreviewed).

## Global Constraints

- Diagnosis only: no new strategies, fixes, deployments, or parameter changes. Lenses report what was ruled out and by what evidence.
- Target (spec section 1): raw total return, SPY +14% means target +28%; pre-tax primary, after-tax shown; drawdown reported, not pass/fail.
- No change to the frozen paper trader, `ggtrader_live`, cron, orders, or `data/`; no database writes; Alpaca reads only (positions, orders, account activities), never order/cancel/config endpoints.
- Off-limits: timers `ggtrader-intraday-observer-{recover,inputs,decide,archive}`, worktree `/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a`, and the observer archive. Reading is allowed.
- Every re-run: `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`, at most 2 in parallel, pinned `--eval-start`/`--eval-end`, `--max-leverage` matching the config under test, raw JSON beside the report. Exit 137 is reported, never retried with a higher cap.
- Tests only via `scripts/run_tests.sh`. Python scripts need `main()`, `argparse`, `--help`, absolute `ggTrader.` imports, type hints, ruff-clean (line length 100).
- No recommendation to retire the sleeve, end paper trading, or shut the project down; no IRA advice; taxable only; no paid data. Report NO-GO plainly.
- Models: Sonnet orchestrates; Opus for trust/verdict judgments, implementation review, root-cause ranking, synthesis; Haiku for extraction, inventories, listing. Opus checks at least 10% of each Haiku batch (minimum 5 items) plus every figure that reaches a report. No second model family reviews; the synthesis says so.
- Private account records stay out of subagent prompts. Expand abbreviations (WFO, OOS, IR, PIT) on first use.
- Commit after each task. Do not push. Add `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` to commit messages.

## Review Focus

Failure modes the spec implies that no lens would catch on its own, most likely first. Each has an owning test or gate.

1. A report cites a file path, commit, or number that does not exist or does not match (hallucinated evidence). Owned by Task 1 checker (path existence) and Task 4 (Haiku figures rechecked).
2. A report omits its falsifiers, steelman, or done checklist, or marks a question done without an answer. Owned by Task 1 checker (required headings, no empty sections).
3. A report slides into recommendations or retirement advice. Owned by Task 1 checker (banned-phrase scan).
4. A re-run hits the memory cap and is silently retried or lost. Owned by Task 2 (runner reports 137 and refuses retry).
5. A subagent touches live state (observer timers/worktree, paper trader, `data/`). Owned by Task 3 (before/after guard snapshot) and re-run at every checkpoint.

---

### Task 1: Lens report checker

**Files:**
- Create: `scripts/check_lens_report.py`
- Test: `tests/scripts/test_check_lens_report.py`

**Interfaces:**
- Produces: `check_report(path: Path, repo_root: Path) -> list[str]` returns a list of problems (empty means pass). CLI: `python scripts/check_lens_report.py <report.md> [--repo-root PATH] [--paths-only]` exits 0 on pass, 1 on any problem, prints one problem per line. `--paths-only` skips the heading rules and checks only cited paths and banned phrases (used for the synthesis).

Required headings in every lens report (exact text, level 2): `## Falsifiers`, `## Steelman`, `## Questions and answers`, `## Verification`, `## Done checklist`. A report fails if any heading is missing or its section is empty, if any backticked repo-relative path (`docs/...`, `src/...`, `scripts/...`, `tests/...`, `results/...`) does not exist under the repo root, or if it contains a banned phrase (case-insensitive): `retire the sleeve`, `shut down the project`, `end paper trading`, `should deploy`, `we recommend deploying`, `open an IRA`.

- [ ] **Step 1: Write failing tests** in `tests/scripts/test_check_lens_report.py` using `tmp_path`: `test_passes_complete_report`, `test_fails_missing_steelman_heading` (asserts a problem mentioning `Steelman`), `test_fails_empty_section`, `test_fails_nonexistent_cited_path` (asserts the missing path appears in the problem text), `test_fails_banned_phrase_retire_the_sleeve`, `test_ignores_paths_in_urls`, `test_paths_only_skips_heading_rules`.
- [ ] **Step 2: Run to confirm failure.** Run: `scripts/run_tests.sh tests/scripts/test_check_lens_report.py -q`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `check_report` and `main` in `scripts/check_lens_report.py`; one regex for backticked repo paths, one pass over `## ` headings for section bodies. Skip paths containing `://` and paths with glob characters.
- [ ] **Step 4: Run tests.** Same command. Expected: all PASS. Run `.venv/bin/ruff check scripts/check_lens_report.py tests/scripts/test_check_lens_report.py` and `ruff format` on both. Expected: clean.
- [ ] **Step 5: Verify `--help`.** Run: `.venv/bin/python scripts/check_lens_report.py --help`. Expected: exit 0 with usage text.
- [ ] **Step 6: Commit** `scripts/check_lens_report.py` and its test: `feat(scripts): add lens report checker for investigation`.

---

### Task 2: Memory-capped runner

**Files:**
- Create: `scripts/run_capped.sh`
- Test: `tests/scripts/test_run_capped.py`

**Interfaces:**
- Produces: `scripts/run_capped.sh [--help] <command...>`; env `GGT_RUN_MEM_MAX` (default `6G`). Runs the command in `systemd-run --user --scope -q --collect -p MemoryMax=$GGT_RUN_MEM_MAX -p MemorySwapMax=0` using the repo `.venv` on PATH. Exit 137 prints `KILLED AT MEMORY CAP (<value>): do not retry with a higher cap; report it` to stderr and exits 137. Never retries. Modelled on `scripts/run_tests.sh`.

- [ ] **Step 1: Write failing tests** in `tests/scripts/test_run_capped.py` (skip if `systemd-run` is absent): `test_runs_command_and_returns_its_exit_code` (`run_capped.sh true` is 0, `false` is 1), `test_help_exits_zero`, `test_allocation_over_cap_exits_137_and_prints_message` (`GGT_RUN_MEM_MAX=64M`, command allocates 512 MB with `python -c "x = bytearray(512*1024*1024)"`; assert returncode 137 and the stderr message).
- [ ] **Step 2: Run to confirm failure.** `scripts/run_tests.sh tests/scripts/test_run_capped.py -q`. Expected: FAIL (script missing).
- [ ] **Step 3: Implement** `scripts/run_capped.sh` (bash, `set -euo pipefail`, `--help` text, message on 137).
- [ ] **Step 4: Run tests.** Expected: PASS. Run `shellcheck scripts/run_capped.sh` if installed.
- [ ] **Step 5: Commit** `feat(scripts): add memory-capped runner for investigation re-runs`.

---

### Task 3: Guard snapshot for off-limits state

**Files:**
- Create: `scripts/guard_snapshot.sh`

**Interfaces:**
- Produces: `scripts/guard_snapshot.sh <outfile>` writes: HEAD and `git status --short` of the intraday worktree; `systemctl --user list-timers --all` lines matching `ggtrader`; `docker ps --format '{{.Names}} {{.Image}} {{.Status}}'` for `ggtrader_live` (read-only, ignore if docker is unavailable); `git -C /home/flynn/ggTrader status --short data/` and `git log -1 --format=%H`. Timestamps and "next run" columns are stripped so two snapshots diff cleanly. `--help` supported.

- [ ] **Step 1: Implement** the script (bash, read-only commands only).
- [ ] **Step 2: Verify determinism.** Run twice into two files: `scripts/guard_snapshot.sh /tmp/claude-1000/-home-flynn-ggTrader/47577f2e-9398-4897-a14d-c38613eb3a47/scratchpad/g1.txt` then `.../g2.txt`; `diff g1.txt g2.txt`. Expected: no output.
- [ ] **Step 3: Record the baseline** to `docs/research/_investigation_guard_baseline_2026-11-04.txt`.
- [ ] **Step 4: Commit** `feat(scripts): add guard snapshot for investigation off-limits state`.

---

### Task 4: Inventory (Phase 1a)

**Files:**
- Create: `docs/research/2026-11-04-investigation-inventory.md`
- Create: `docs/research/_investigation_inventory.json`

**Interfaces:**
- Consumes: `docs/research/RESEARCH_SNAPSHOT.md`, `docs/research/RESEARCH_INDEX.md`, `docs/roadmap.md`, `docs/changelog.md`, `docs/research/*.md`, `docs/research/_*.json`, the Phase 0 audit, the auto-memory index at `/home/flynn/.claude/projects/-home-flynn-ggTrader/memory/MEMORY.md` (including the 2026-10-20 triage note), and crypto-era docs (roadmap, changelog, `docs/archive/`).
- Produces: JSON list of rows `{id, name, market ("stocks"|"crypto"|"cross-asset"), claim, headline_number, verdict, source_report, raw_artifact (path or null), live_deployed (bool), lens_owners (list of lens numbers)}`. "Material" means the row carries a headline number, a verdict, or a live deployment decision. Later tasks measure coverage against `id`s here.

- [ ] **Step 1: Haiku extraction.** Dispatch Haiku agents (batches of about 10 reports) to emit rows from the sources above with exact quotes and paths. Prompt: read-only, extract only, mark unknown fields `null`, never infer a number.
- [ ] **Step 2: Opus classification.** One Opus agent merges duplicates, assigns `market`, `live_deployed`, and `lens_owners`, and decides which rows are material.
- [ ] **Step 3: Verify.** An Opus agent re-opens at least 10% of rows (minimum 10) against their source and reports mismatches; fix them. Confirm every row has a `source_report` that exists (`python -c` loop over the JSON). Expected: zero missing.
- [ ] **Step 4: Render** the markdown inventory (table sorted by market then id) from the JSON; state the row count and how many have `raw_artifact: null`.
- [ ] **Step 5: Commit** both files: `docs(research): add investigation inventory`.

---

### Task 5: Lens 2 sensitivity-table specification

**Files:**
- Create: `docs/research/2026-11-04-lens2-sensitivity-spec.md`

**Interfaces:**
- Produces: a fixed specification Lens 2 must follow. Columns: result id, raw CAGR gap vs SPY total return in pp/yr, 95% confidence interval, t-statistic, verdict at t of at least 3, verdict at t of at least 2, power note. States: exploratory only, no verdict changes on the looser bar alone, SPY comparator is total return, and the windows used. Must be committed before any Lens 2 work opens a result.

- [ ] **Step 1: Write the spec** (Opus), including how each column is computed from a results JSON and what to write when a JSON lacks a field ("not computable from artifact").
- [ ] **Step 2: Commit before Task 9 starts**: `docs(research): fix lens 2 sensitivity table spec`.

---

### Task 6: Lenses 6 and 1 (implementation correctness, backtest validity)

**Files:**
- Create: `docs/research/2026-11-04-lens6-implementation-correctness.md`
- Create: `docs/research/2026-11-04-lens1-backtest-validity.md`
- Create: `docs/research/_lens6_replays.json`

**Interfaces:**
- Consumes: inventory (Task 4); `src/ggTrader/lab/strategies/`, `src/ggTrader/lab/` (harness, wfo, gates, simulate, metrics), `src/ggTrader/paper/`, the research briefs, git history.
- Produces: per-inventory-id verdicts `{id, spec_matches_code: yes|no|partly|unverifiable, tape_state_at_run: clean|contaminated|unknown}` used by Lens 2 and Lens 7.

- [ ] **Step 1: Lens 6, Opus per strategy group.** Start with `ensemble`, then strategies behind headline numbers. Re-derive the spec from the research brief, compare entry/exit rules, lookbacks, rebalance timing, point-in-time data use. Include crypto strategies from git history before the 2026-06-16 removal; mark gone items "unverifiable".
- [ ] **Step 2: Hand-replay.** For `ensemble` and two other headline strategies, replay a few symbols over a few weeks by hand and compare with code output (read-only; use the memory-capped runner if executing code). Record in `_lens6_replays.json` with symbols, dates, hand result, code result.
- [ ] **Step 3: Lens 6 live path.** Compare `src/ggTrader/paper/` logic with the backtested logic (partial-bar signals, position caps, sweep sleeve, exits).
- [ ] **Step 4: Lens 1, Opus + Haiku.** Haiku lists for each inventory id the run date and window; Opus maps these to the five known fixes (`2eafab1`, `c9d1abe`, `09f9f40`, tz day-shift, SPY duplicates) and judges remaining contamination, benchmark, universe, fills, costs, yfinance adjustment and dividend handling. State falsifiers first.
- [ ] **Step 5: Gate.** Run `.venv/bin/python scripts/check_lens_report.py <report>` on both reports. Expected: exit 0.
- [ ] **Step 6: Commit** the three files: `docs(research): add lens 6 and lens 1 reports`.

---

### Task 7: Lens 7 (research accuracy)

**Files:**
- Create: `docs/research/2026-11-04-lens7-research-accuracy.md`
- Create: `docs/research/_lens7_recompute.json`

**Interfaces:**
- Consumes: inventory rows, raw results JSONs, lens 6 verdicts.
- Produces: `{id, reported, recomputed, match: bool, note}` per row; academic-claim checks; pre-registration status per verdict (the only lens asking this).

- [ ] **Step 1: Haiku recompute.** For each inventory row with `raw_artifact`, Haiku recomputes the headline number from the JSON (read-only, quick scripts in the scratchpad) and records it. Rows without an artifact are marked "prose only".
- [ ] **Step 2: Opus verification.** Opus re-checks at least 10% of rows (minimum 5) and every mismatch and every figure destined for a report.
- [ ] **Step 3: Academic claims.** Opus checks the cited factor, momentum and leverage papers say what the reports claim (free sources only, cite URLs).
- [ ] **Step 4: Pre-registration audit.** For each verdict, was the pass bar set before the result? Cite the brief commit or date.
- [ ] **Step 5: Gate** with `check_lens_report.py`; **Step 6: Commit** `docs(research): add lens 7 report`.

---

### Task 8: Lens 3 (live versus backtest parity)

**Files:**
- Create: `docs/research/2026-11-04-lens3-live-parity.md`
- Create: `docs/research/_lens3_snapshot_gap.json`

**Interfaces:**
- Consumes: DB tables `paper_trades`, `paper_snapshots`, `ohlcv` (SELECT only), Alpaca account-activity reads (DIV), `src/ggTrader/paper/trader.py:1051-1056`, `docs/research/2026-11-02-paper-freeze-readiness.md`.
- Produces: era attribution of the paper shortfall; verdict on the snapshot gap with two independent sources; supplies evidence to the existing paper-parity plan without replacing it.

- [ ] **Step 1: Era attribution (Opus).** Recompute return versus SPY total return per configuration era from the DB and compare with the Phase 0 figures (-1.9, -2.6, +1.4 pp).
- [ ] **Step 2: Snapshot gap.** Compare daily `value - (cash + positions)` since 2026-08-20 against Alpaca DIV activities; record both series in `_lens3_snapshot_gap.json`. "Confirmed" requires two sources agreeing. If confirmed, tell the owner in chat at once; change nothing.
- [ ] **Step 3: Unmodelled behaviour.** List what live does that tests did not (partial-bar signals at 12:45 PT, caps, sweep, orphans, manual liquidations, dividends, fills before close). Estimate days a frozen sleeve of about 10% needs to detect a given effect size.
- [ ] **Step 4: Gate and commit** `docs(research): add lens 3 report`.

---

### Task 9: Lens 2 (verdict audit)

**Files:**
- Create: `docs/research/2026-11-04-lens2-verdict-audit.md`
- Create: `docs/research/_lens2_sensitivity.json`

**Interfaces:**
- Consumes: Task 5 spec, inventory, Task 6 and Task 7 outputs.
- Produces: per-id classification `reliable NO | unreliable NO (flawed test) | inconclusive | reliable GO | unreliable GO`; the sensitivity table per Task 5; multiple-testing count; gate-machinery and researcher-freedom findings.

- [ ] **Step 1: Classify** every material NO/GO row (Opus), using Lens 6 and Lens 1 flags. Resolve the replica conflict: Phase 0 says "Reliable NO", the results JSON says `INCONCLUSIVE` (`docs/research/_production_replica_results.json`).
- [ ] **Step 2: Sensitivity table** exactly as specified; mark columns "not computable" where the artifact lacks fields; no verdict changes on the looser bar.
- [ ] **Step 3: Count tests** across the project and state what multiple testing implies for a "win". Assess whether the circuit breaker and gates (`2026-11-02-wfo-policy-synthetic.md`) rejected candidates a simpler test would keep, and how much design freedom shaped verdicts.
- [ ] **Step 4: Gate and commit** `docs(research): add lens 2 report`.

---

### Task 10: Lenses 4, 5, 9 (feasibility and ceiling, coverage, power and regime)

**Files:**
- Create: `docs/research/2026-11-04-lens4-feasibility-and-ceiling.md`
- Create: `docs/research/2026-11-04-lens5-coverage-gaps.md`
- Create: `docs/research/2026-11-04-lens9-power-and-regime.md`

**Interfaces:**
- Consumes: inventory, Lens 2 classifications (for Lens 4 ceiling), `WEB_RESEARCH_CANDIDATES.md`, intraday readiness documents (read-only).
- Produces: the measured ceiling table (approach, highest sound raw return over SPY, confidence interval, conditions: exposure, leverage, regime, costs, tax); the ruled-out list for breadth/leverage/concentration/long-short with evidence; coverage list (never tested versus tested badly); detectable-effect sizes per test; regime dependence.

- [ ] **Step 1: Lens 9 first (Opus).** Compute the minimum detectable effect for each material test from sample length and tracking error; state single-regime dependence of 2021 to 2026.
- [ ] **Step 2: Lens 4 (Opus).** Apply the section 1 definition to produce the measured ceiling table from existing sound results only; cite free sources for how large systematic firms operate; run the after-tax short-holding check.
- [ ] **Step 3: Lens 5 (Opus + Haiku).** Haiku lists tested universes, horizons and instruments; Opus identifies untested areas and hypothesis sourcing and crowding. Treat intraday as prospective evidence in progress.
- [ ] **Step 4: Gate and commit** `docs(research): add lens 4, 5 and 9 reports`.

---

### Task 11: Lens 8 (crypto)

**Files:**
- Create: `docs/research/2026-11-04-lens8-crypto.md`
- Create: `docs/research/_lens8_tape_checks.json`

**Interfaces:**
- Consumes: crypto docs and git history before the 2026-06-16 removal, `docs/archive/`, the 2026-06-08 edge-search report, `docs/research/2026-11-01-research-process-audit.md`, TimescaleDB `ohlcv` (SELECT only, about 648 `-USD` symbols).
- Produces: what was tried (universe, fees, venue, benchmark); whether anything beat buy-and-hold BTC or ETH; fee soundness; untested scope (shorting, perpetuals, funding carry); tape integrity checks (duplicates, timestamp shift, gaps, symbol normalization, survivorship of coins); the annualization question; live evidence available.

- [ ] **Step 1: Haiku history pull.** Haiku lists every crypto study with its source and numbers from docs and git history.
- [ ] **Step 2: Opus tape checks** with read-only SELECTs (duplicate rows per symbol/day, timestamp-hour distribution, gaps, delisted coins missing); save counts to `_lens8_tape_checks.json`.
- [ ] **Step 3: Opus judgment** on the benchmark question, BTC beta, fee tiers, and the 252-day versus 365-day annualization in `src/ggTrader/lab/metrics.py`.
- [ ] **Step 4: Gate and commit** `docs(research): add lens 8 report`.

---

### Task 12: Phase 1 checkpoint and re-run proposal

**Files:**
- Create: `docs/research/2026-11-04-phase1-checkpoint.md`

**Interfaces:**
- Consumes: all lens reports, the guard baseline (Task 3).
- Produces: coverage against the inventory (every id classified or "unverifiable, prose only"); each lens "done" or listing its unknowns; the proposed Phase 2 re-run list with cost, memory expectation, and which verdict each would change; a note whether Phase 2 can be skipped; usage report.

- [ ] **Step 1: Guard check.** Run `scripts/guard_snapshot.sh` to a scratch file and diff against the baseline. Expected: no difference. Any difference is stopped and reported to the owner before continuing.
- [ ] **Step 2: Coverage script.** `python -c` loop confirming each inventory id appears in at least one lens report. Expected: none missing, or listed as unverifiable.
- [ ] **Step 3: Compile** the re-run proposal (Opus) and report the usage so far.
- [ ] **Step 4: Commit** the checkpoint. **Stop and ask the owner to approve the re-run list.**

---

### Task 13: Phase 2 re-runs (only for items the owner approves)

**Files:**
- Create: `docs/research/2026-11-04-phase2-rerun-<slug>.md` and `docs/research/_phase2_<slug>_results.json` per approved item.

**Interfaces:**
- Consumes: `scripts/run_capped.sh`; existing drivers in `scripts/*_wfo.py` and the factor/momentum screen scripts; pinned windows from the original runs.
- Produces: raw JSON plus a report per item. Candidates already flagged: stock sleeve versus equal-weight S&P 500 (15 to 30 min; needs a sleeve return series), French top-decile big-cap momentum after 1994 (15 min).

- [ ] **Step 1: For each approved item**, write a one-paragraph pre-registration (question, window, comparator, what would change which verdict) and commit it before running.
- [ ] **Step 2: Run** via `scripts/run_capped.sh <command>` with pinned `--eval-start`/`--eval-end` and the correct `--max-leverage`; at most 2 in parallel. Capture JSON beside the report.
- [ ] **Step 3: On exit 137**, record it as a finding; do not rerun with a higher cap.
- [ ] **Step 4: Gate** each report with `check_lens_report.py`, run the guard diff, and commit `docs(research): add phase 2 re-run <slug>`.

---

### Task 14: Synthesis, challenge, and owner summary

**Files:**
- Create: `docs/research/2026-11-04-root-cause-investigation.md`
- Create: `docs/research/2026-11-04-root-cause-challenge.md`
- Modify: `docs/changelog.md`, `docs/research/RESEARCH_SNAPSHOT.md` (via the `research-snapshot` skill)

**Interfaces:**
- Consumes: all lens reports and Phase 2 results.
- Produces: the synthesis in plain language, containing a one-page summary; a traffic-light trust table (one sentence per row, "what would change this" column); the ranked root causes in the three groups (method and process, implementation and live parity, evidence negative or target structurally hard) with citations; the measured ceiling; what is sound; what is unknown; the fixed 5 to 7 owner questions with answers; an explicit statement that no second model family reviewed it.

- [ ] **Step 1: Opus writes** the synthesis, then runs `.venv/bin/python scripts/check_lens_report.py --paths-only docs/research/2026-11-04-root-cause-investigation.md`. Expected: exit 0.
- [ ] **Step 2: Fresh Opus challenger** receives only the raw results, the inventory and the claims (not the reasoning), is told to argue the opposite case, and writes `...-root-cause-challenge.md`. Lead resolves each objection in the synthesis or records it as unresolved.
- [ ] **Step 3: Final guard diff** against the baseline. Expected: no difference.
- [ ] **Step 4: Update** `docs/changelog.md` and regenerate the snapshot with the `research-snapshot` skill. Do not change `docs/next_steps.md` without the owner.
- [ ] **Step 5: Commit** `docs(research): add root-cause investigation synthesis and challenge`. Report quota used and present the one-page summary to the owner.

---

## Self-Review

- **Spec coverage:** section 1 target, Task 10 (ceiling) and Task 5; anti-goals, Task 1 banned phrases; section 3 constraints, Global Constraints plus Tasks 2 and 3; section 4 hypotheses, Tasks 6 to 9; Lenses 1 to 9, Tasks 6 to 11; inventory, Task 4; sensitivity spec before Lens 2, Task 5; sequencing and early exit, Task 12; deliverables, Task 14; success criteria, Tasks 12 and 14.
- **Interface consistency:** the `--paths-only` flag used in Task 14 is defined in Task 1's CLI and tested there.
- **Order:** Tasks 1 to 3 (tools), 4, 5 in parallel with 6 (after 4), 7 after 6, 8 and 10 and 11 in parallel with 6, 9 after 5, 6 and 7, 12 after all lens tasks, 13 after owner approval, 14 last.
