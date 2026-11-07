# Study isolation: corrections to PLAN.md

Base 46d8ff75, worktree /home/flynn/ggTrader-vbt12. All checks were static (git, grep, ast, wc), except for the C19
scratch-copy simulation, which ran outside the repo. None of the oracle numbers was re-run. Bottom line: the plan's
core safety argument holds. Nothing that stays in src imports anything that moves, the oracles cannot move on import
grounds, the hash-pin mechanism works from history, and the live intraday recorder cannot see the repo. The plan is
**not executable as written**, though:
- Two slices turn the tests red the moment they land (S5(d), and every study-test move before a root conftest exists).
- Several of the plan's own counts and stated reasons are wrong.
- The test edit list is badly incomplete.

## 1. Claim verdicts

| ID | Final | Corrected statement / evidence (only where not "holds") |
|---|---|---|
| C01 | holds | Nothing that stays in src (or ggt.py, the Docker checks or p5_wfo_validation) imports a moving module. This covers 512 import statements plus dynamic imports. Wording: the moving set is 23 files (22 lab modules + utils/returns), not 22. significance and fred_data move to utils, not out of src. |
| C02 | holds | `provenance._IGNORED_KEYS` (provenance.py:72-81) ignores git_sha, git_dirty, lock_sha256, strategy_*, created_utc and rust_calls. The pyproject diff adds markers only, and uv.lock is unchanged. **Oracle wording fix:** "field-equal to the guarded JSON" is true only for the driver route (`p5_wfo_validation.py --runs core17_ensemble_sp500`), with provenance, wall_seconds, diagnostics.timings_s, diagnostics.peak_rss_mb and tape.path (absolute path) excluded. The CLI route differs from the driver by design in 5 fields (P10-completion.md:50). Compare a CLI run with a base-commit CLI run instead. |
| C03 | holds | tests/lab/test_a23_rebalance.py:18 runs `from scripts import static_leverage_aftertax as a23`. load_prices is used at :168 and :248, _trade_to at :174 and run at :196. Once S5(d) moves the script, collection of all 14 tests fails. PLAN.md:37 says "none" and :176-178 keeps the file unedited. **Both are wrong.** |
| C04 | holds | lens2_build.py imports only the standard library (lines 14-20). It mentions significance.py only in prose (1303, 1885, 1902, 2025, 2030). The real significance importers are momentum_screen.py:9, scripts/production_replica_audit.py:20, tests/lab/test_significance.py:9, tests/scripts/test_production_replica_audit.py:10, and the frozen, already-broken legacy-scripts/audit_reruns_wfo.py:44 (leave it alone). That is **two families, not three.** The lazy curve_summary import is still required. |
| C05 | **wrong** | There are **22 pins in 6 files: 15 MATCH and 7 differ.** The plan says 21 and 6. The missing pin is tests/lab/test_intraday_observer.py (c33f1c09) in _intraday_observer_build_verification.json, which resolves to blob 4ab01d1a. Its row should read "1 MATCH, 2 HISTORY". PLAN.md:88, :289, :344 and :365 are all wrong. |
| C06 | holds | Every differing pin resolves to the same-path blob: event_timing_audit 740781a6, production_replica_audit 8bc365ee, intraday_probe c20be227 (3 files), observer and test 4ab01d1a. Recomputing the activation code_sha256 at 280d4a0 gives f0320a3e; at 46d8ff75 it gives 33c2b9d5. Additions: the feasibility code_sha256 f0b85690 resolves at **6e89281c**, which the plan does not record. In a shallow clone none of these resolve (verified with a depth-1 clone). verify_pins must check `--is-shallow-repository` and skip a history commit where the old path is the deletion. |
| C07 | holds | test_recorded_hashes.py checks only the docs/superpowers plan digest. No test anywhere holds the sibling or loader pin prefixes. The CODEMOVED do_not_edit sha test is the only guard. |
| C08 | holds | The 4 siblings have no `__file__`, parents or ROOT, and their imports are absolute to modules that stay. Addendum the plan lacks: the audit needs **3** import rewrites (:26, :2169, :2690) plus the ROOT change (:43). Audit, sources and reconstruction must stay in the same directory (`with_name` hashing at :1545). From code/, the repo root is `parents[4]`. The tests carry 7 string monkeypatch targets (see C13). |
| C09 | holds | The installed units point only at releases/280d4a0…, runtimes/366f9247… and the credentials file (2 APCA keys, no repo path). The runtime venv has stale traces (pyvenv.cfg, direct_url.json, the bin/ggt shebang) but the editable .pth is absent, so nothing runs from the repo. Do not touch the release, the runtime or the credentials. |
| C10 | holds | tax_lots has no importer outside its own test. paper/ has no lot code; the "lot" hits are "slot" and "wash out". The A23 Book lives in a23_rebalance. The moved test must be rewired. |
| C11 | holds | Delete event_study and utils/returns, and all of tests/lab/test_event_study.py. In test_pandas3_compat.py, remove imports :17 and :19 plus TestPaddedPctChange (4) and TestEventStudyStack (2), keeping 3 tests. Correction: "no importer anywhere" is false. Three frozen research files import it (legacy-scripts/audit_reruns_wfo.py, fomc_clustered_event_study.py, p1-parity/p1_parity_driver.py:358). All three are already unrunnable at base. Leave them untouched and say so. docs/changelog.md:121-125 is stale about padded_pct_change users. |
| C12 | partly | **Two** skills call the script, and S5(e) fixes only one. daily-trader-check/SKILL.md:139 runs it with `2>/dev/null \| tail -5`, so the failure is **silent**. Nothing in the repo or in Hermes jobs.json schedules it, and results/shadow_cross_asset.csv was last written 2026-10-18, so it runs on demand. The crontab was not read. If the script moves, `REPO_ROOT=parents[1]` would send the CSV into research/. |
| C13 | holds | The edit list misses test_import_is_inert (:281-293, a `python -I` subprocess that inserts ROOT/src). It also misses the 7 string patch targets in test_event_timing_audit (:580, 603, 810, 1295, 1792, 1814, 1816) and 5 function-local `import ggTrader.lab.event_timing_*` statements (:1936, 2375, 2589, 2696, 3093). No other test file has dotted-string targets. |
| C14 | **wrong** | The test names 26 scripts (25 .py + 1 .sh). 13 are live, and 13 are checked through the archive fallback. **All 26 cases run real assertions**, and none is vacuous. Two more tests check the lens2_build artifact map and the fold-attribution sources. "Half of it already checks nothing" (PLAN.md:198, :356) is false. Deleting the file removes the only guard on driver evidence paths. |
| C15 | partly | 32 of the 48 trimmed test_paper_boundary cases are true duplicates. The other 16 (the legacy lab-module ban) duplicate nothing. Five of the six banned modules are already deleted, so an import of them fails anyway. Only `ggTrader.lab.strategy` would slip through. Low value, but the plan must state the loss or keep a one-line ban. |
| C16 | holds | src is 26,343 lines in 107 files. 12,426 lines move or are deleted, which with ~200 utils lines gives ~14,100. scripts/ goes 12,320 − 5,089 − 3,203 = 4,028. File counts: 21 modules (23 with significance and fred_data), **23 moved scripts, not 24**. Assumes D2 = move tax_lots (+583 if reversed) and D6 = delete tools (scripts would be ~7,230 without it). scripts/ excludes the new study.py and verify_pins.py, so it is a floor. |
| C17 | partly | Line and case arithmetic holds: 19,384 lines and 1,569 cases left. **The FAST tier prediction does not.** Recomputed per test it is **~1,215–1,276 cases and ~105–120 s** of test time (~100–115 s wall after the sleep patch), against the plan's ~1,160 cases in 70–100 s. The plan subtracted whole files that stay partly in FAST: universe synthetic 13.8 s, causality synthetic 11.3 s, leak_golden non-memory 8.0 s. The S1 gate of "FAST <= 100 s" is at risk, and nothing has been measured yet. |
| C18 | holds | All 166 research .py files sit under vectorbt-engine (161) or `<cat>/artifacts-*` (5). A planted-file experiment confirms the narrowed excludes and `force-exclude`. Refinement: ruff's `*` crosses `/`, so any `artifacts-*` segment at any depth below research/ is silently excluded, including inside code/. The replacement test must enforce an allow-list. |
| C19 | partly | Study tests don't see the real .env **only if** a repo-root conftest.py carries `_no_dotenv` and Agg. Simulated with a sentinel .env: it does not leak with the root conftest and does leak without it. importlib mode works on the real tests (228/230 and 246/253; every failure was a scratch artifact). Corrections: (1) `--tier`, `pytest_addoption` and `pytest_configure` must also move to the root conftest. Today `--tier studies research/...` fails with "unrecognized arguments". (2) Moved code reads os.environ directly (SEC_*, APCA_*, POSTGRES_CONNECTION_STRING), not only through utils.config. That is unchanged from today and is not a .env leak, but the claim's wording is wrong. Subprocess tests are outside the patch. |
| C20 | partly | Packaging holds. `where=["src"]` never sees research/. The Docker build is editable, and the trader's 74-module import closure contains no moved module. testpaths=['tests'] already exists and no workflow runs a bare pytest. Overstatements: (a) the image already carries scripts/ and tests/ (not in .dockerignore), so it becomes study-free only because files leave those directories; (b) `pytest tests` does **not** run research/**/code/tests, so a STUDIES job is mandatory; (c) the in-progress ci.yml runs only `--tier fast` on PRs; (d) lint **does** change: CI will lint code/ and _lib/, and `bootstrap(__file__)` before the imports raises E402. |

Count: 11 hold, 6 partly, 2 wrong (C05, C14). No claim was dropped.

## 2. Completeness gaps, ranked, with the PLAN.md amendment

1. **BLOCKER: no root conftest yet, and no gate requiring one.** `/home/flynn/ggTrader-vbt12/conftest.py` does
   not exist. `_no_dotenv`, `--tier`/`--update-golden`/`--golden-reason` and `pytest_configure` are still in tests/conftest.py.
   *Amend S1 exit, and S5/S7/S8 entry:* "repo-root conftest.py holds `_no_dotenv`, Agg, `pytest_addoption(--tier)`
   and `pytest_configure(apply_tier)`, plus a FAST sentinel test showing that a test under research/ never sees a tmp
   .env's keys and that `run_tests.sh --studies research/...` parses."
2. **BLOCKER: S5(d) breaks test_a23_rebalance (14 tests, including the tax-advisory tests).** *Amend PLAN.md:37, :176-178 and S5(d):*
   move `load_prices`/`run` (and their frozen-input loader) into `research/allocation/_lib/a23_backtest.py`. Keep
   the 12 pure tests in tests/lab. Move the 2 frozen-parity tests (they need data/external/a23_inputs) to static-leverage
   code/tests, all in the same commit. Name test_a23_rebalance in the S5(d) exit.
3. **MAJOR: cross-study import.** test_century_leverage.py:19 imports the static-leverage driver, which violates the plan's own
   "no study imports another study's code/" rule. *Amend S5(d):* the `_lib/a23_backtest.py` from gap 2 removes this edge too.
   Sequence (d) after the decision. century_leverage_a23_x keeps its `git show` history read unchanged.
4. **MAJOR: `.parent.parent` roots are missed, and their only tests are deleted first.** lens2_build.py:22 and
   wfo_fold_attribution.py:16 would resolve ROOT into the study folder with no failing test. *Amend S1:* do not delete
   test_driver_default_paths outright. Port its two inventory tests to root-cause/code/tests and wfo-methods/code/tests
   before S5. Widen the root sweep regex to `parents\[|\.parent\.parent|find_project_root|sys\.path\.insert`. Fix the
   rationale at :198 and :356.
5. **MAJOR: test edit list incomplete.** Add the C13 items, the `parents[2]` script paths
   (test_momentum_screen:276, test_intraday_probe:432, test_intraday_momentum_data:17), the cwd-relative `scripts/...` (test_shadow_cross_asset:9,
   test_static_leverage_aftertax:122) and every `from scripts import` (production_replica_audit:11, root_cause_figures:11-12,
   static_leverage_aftertax ×5, test_replica:203). *Amend the S5/S7/S8 exit* to a mechanical sweep that must return zero hits:
   `git grep -nE 'ggTrader\.lab\.(<moved>)|scripts/(<moved>)\.py|from scripts import (<moved>)'` over tests,
   scripts, research/**/code, .claude and current docs. Rewrite string patches as object patches.
6. **MAJOR: the `--help` guard silently drops 13 drivers.** The S3 test globs only `code/run*.py`. *Amend S3:* cover every
   `research/*/*/code/*.py` (and root-cause code) that has a `__main__` guard, carry DELEGATES forward, and assert
   that the before and after script count balances (49 − 4 deleted = stayed + moved).
7. **MAJOR: S8 observer re-release path.** The runtime venv has no working ggTrader install. *Amend S8:* the wrappers stay
   self-bootstrapping with stdlib-only sys.path setup to code/ and intraday/_lib. Add a `python -I` import test. Keep the
   existing runbook commands unchanged (they describe the 280d4a0 layout) and add a separate "releases cut after <S8 sha>" section.
8. **MAJOR: REPO_ROOT can resolve to the wrong checkout.** The main .venv's editable .pth points at /home/flynn/ggTrader/src.
   *Amend §3 and S0:* `bootstrap(__file__)` returns the root found by walking up from the study file, and moved
   code uses that. S0 and the experiments assert `ggTrader.__file__` lies under the worktree under test, or use that worktree's own venv.
9. **MAJOR: STUDIES gate missing, and plan and in-progress CI disagree.** The in-progress `--studies` is `-m study` inside tests/,
   which collects 0 once tests move. PRs run only fast, and there is no STUDIES job. test_replica (which exercises paper internals) leaves
   the deploy gate. *Amend §4 and S1:* add a STUDIES CI job (triggered on src/**, required on PRs, command contains "tests"
   for test_upgrade_tooling_fixups). `--studies` targets the explicit `research/*/*/code/tests` and `research/*/_lib/tests`
   paths with a minimum-collected guard. Add STUDIES (or at least test_replica) to the ggtrader-deploy pre-flight. Reconcile
   PLAN.md:206-212 (FULL required on PRs) with the S1 work in progress, in writing, before S1 closes.
10. **MAJOR: daily-trader-check not updated.** *Amend S5(e):* edit both skills in the same commit, or keep
    shadow_cross_asset.py in scripts/ as an ops tool. It reads live paper_snapshots, so the second option is recommended. If it moves,
    its CSV default becomes `REPO_ROOT/results/`.
11. **MAJOR: docs updated before the code moves.** S6 precedes S7 and S8, yet architecture.md and cli_reference.md name the event-timing
    and intraday paths. *Amend:* move S6 after S8, or give S7 and S8 their own doc steps. Limit the path check to AGENTS.md,
    README.md, architecture.md, cli_reference.md, next_steps.md and .claude/skills (history docs are excluded). Rebase on the
    tier agent's AGENTS.md, README.md and architecture.md edits first.
12. **MAJOR: FAST gate mis-sized.** *Amend D4, S0 and S1:* quote ~1,200–1,280 cases and ~100–120 s. Make S0 measure on an idle
    host with `--durations=25`. Set the S1 gate from that measurement, or mark the synthetic parity halves slow.
13. **MINOR: pin count and commits.** *Amend :84, :88, :289, :344 and :365:* change 21/6 to 22/7 and 21/21 to 22/22. Add the test file to the S8
    CODEMOVED rows. Record 6e89281c for the feasibility map, and name intraday_momentum.CODE_FILES (:65-70) among the S8 edits.
14. **MINOR: unstated coverage loss.** Keep a single non-parametrised ban on paper importing `ggTrader.lab.strategy` plus the
    deleted names. Add `research/*/*/code` and `research/*/_lib` to the TestNoBarePctChangeCalls scan roots.
15. **MINOR: lint and collection traps.** Add the per-file ignore `"research/**/code/*.py" = ["E402"]`, or a noqa policy.
    Never pass `research` as a whole to pytest: it collects 10 frozen test_*.py files that import deleted modules. Extend
    basename uniqueness to the repo-root names (`scripts`, `tests`, `data` and `conftest`).
16. **MINOR: plan inaccuracies.** Correct the significance row (C04). Leave the trader.py:81 and :1099 comments that cite the deleted
    migrate script alone (editing them touches the trader and the image), and resolve them through CODEMOVED. Reword risk 4: the
    ruff-fix hook hardcodes REPO=/home/flynn/ggTrader, so it does not run in the vbt12 worktree. Fix the stale pyproject testpaths comment
    in S1.
17. **MINOR: rollback incomplete.** Add a per-slice stop rule: if O or the 120/120 golden moves, revert and stop. Add a
    `git status --porcelain research data results` must-be-clean check after each slice, to catch untracked output from a mis-rooted
    driver. Push to main once, at S9: docker-build.yml rebuilds ghcr :latest on every push with no path filter, and the trader cron must be confirmed off.
18. **MINOR: S2 oracle exemption is wrong.** S2 deletes and moves files under src/ggTrader/lab, and the plan's own rule requires O
    whenever src/{engine,lab,data,paper} changes. *Amend S2, S5, S7 and S8:* run O (cheap insurance, even if C01 says it cannot move).
    Classify the unclassified items: `src/ggTrader/__init__.py` (stays), tests/tiers.py and test_tiers.py (the in-progress S1 agent
    contradicts "no central table"; pick one), and test_frozen_inputs.py (assign a tier).

## 3. Slice verdicts

| Slice | Verdict | Exact conditions |
|---|---|---|
| S0 | go-with-changes | Run O using the driver form with the named exclusion set (C02). Assert `ggTrader.__file__` lies in the tested worktree (gap 8). Measure FAST per test on an idle host (gap 12). Experiments pass explicit research paths only (gap 15). |
| S1 | no-go-until-fixed | Root conftest with `_no_dotenv`, Agg, `--tier` and `pytest_configure`, plus the sentinel test (gap 1). Do not delete test_driver_default_paths before porting its two inventory tests (gap 4). Keep the lab.strategy ban (gap 14). STUDIES job and `--studies` paths reconciled with the plan (gap 9). FAST gate taken from the S0 measurement (gap 12). |
| S2 | go-with-changes | test_pandas3_compat: drop imports :17 and :19 and both classes (6 tests) (C11). Move the significance row to two families (C04), and make the curve_summary and get_engine imports lazy. Run O (gap 18). Run serially after S1, because S2 shares test files with it. |
| S3 | go-with-changes | 22/22 pins. verify_pins checks for a shallow clone and skips deletion commits (C05, C06). Ruff allow-list test with an `artifacts-*` depth rule (C18). E402 per-file ignore. The help/AST test covers every code/*.py with a `__main__` guard plus a count balance (gap 6). |
| S4 | go | It touches no moved code. `study check` must enforce the cross-study rule, which is only satisfiable after gap 3. |
| S5 | no-go-until-fixed | (d) needs the a23 `_lib` split in the same commit (gaps 2 and 3). (e) updates both skills or keeps the script in place (gap 10). (b) and (f) need the ported inventory tests and `.parent.parent` fixes (gap 4). Every sub-slice passes the mechanical sweep (gap 5). STUDIES CI must exist first (gap 9). Sub-slices are not truly parallel: run (d) alone. |
| S6 | no-go-until-fixed | Must run after S8, or be split per slice (gap 11). Path check restricted to current docs. Rebase on the other agent's doc edits. |
| S7 | go-with-changes | Three audit import rewrites, the ROOT change to `parents[4]`/bootstrap, and audit, sources and reconstruction co-located (C08). Object-based patches for the 7 string targets and the 5 local imports rewritten (C13). CODEMOVED do_not_edit sha test present and green. Verify the 4 siblings with `sha256sum` after `git mv`. |
| S8 | no-go-until-fixed | Self-bootstrapping wrappers with a `python -I` test, and the runbook commands left unchanged (gap 7). test_import_is_inert rewritten without weakening it (C13). Test-file CODEMOVED row and both observer and feasibility CODE_FILES (gap 13). Read-only checks of the release, timers and credentials (C09). O re-run. |
| S9 | go-with-changes | Use the corrected figures: src ~14,100, scripts ≥4,028 (23 moved), FAST ~1,200+ cases. Push once, with the trader cron confirmed off (gap 17). |

## 4. Unverified or dropped, stated plainly

- No oracle was re-run. O (0.391 / 3.50% / 1/17), dsr0 (0.637 / 3.41%) and the 120/120 golden are argued unchanged from import
  structure (C01, C02) only. Numeric equality is unproven until S0 and S9 run them.
- FAST timing is a recomputation from loaded-host durations.json and per_test.json. It was not measured.
- `uv build --wheel` and the Docker context-size check were not run (static `find_packages` only).
- The user crontab was not read; one checker ran a stray `crontab -l`, which showed only the header. Whether anything outside Hermes and the repo
  schedules shadow_cross_asset is unknown. Hermes jobs.json was read and contains no such job.
- The post-move `sha256sum` of the 4 event-timing siblings cannot be checked until S7 exists.
- Whether any subprocess test spawns a config-reaching script (form4_backfill, shadow) was not checked.
- The ~200 new utils lines is an estimate of code not yet written.
- No claim was dropped. All 20 claims were checked by 2–3 independent reports.

## 5. Amendments to apply to PLAN.md, in order

1. §1a: lab/a23_rebalance row → importer tests/lab/test_a23_rebalance.py:18. Add `research/allocation/_lib/a23_backtest.py`
   (load_prices, run, frozen loader). Fix the significance row to two families and remove lens2_build. Add `src/ggTrader/__init__.py`.
   Moved-script count 22/24 → 23.
2. §2: pins 21/6 → 22/7 (:84, :88, :344, :365). Add the test_intraday_observer row (HISTORY 4ab01d1a) and feasibility 6e89281c.
   verify_pins: shallow-clone check, deletion-commit skip.
3. §3: bootstrap returns the root walked up from the study file. Root sweep regex includes `.parent.parent` and `sys.path.insert`. Add the
   mechanical zero-hit grep. String patches become object patches. Observer wrappers stay self-bootstrapping.
4. §4: root conftest contents (gap 1). Replace the "half checks nothing" rationale (:198, :356) and port the 2 inventory tests.
   Restore a single lab.strategy ban. Extend the pct_change scan roots. FAST ≈1,200–1,280 cases and ~100–120 s. STUDIES job with explicit
   paths and a minimum-collected guard. test_replica in the deploy pre-flight. Reconcile with tests/tiers.py.
5. §5: E402 per-file ignore. Ruff allow-list with an `artifacts-*` depth rule. Help/AST test over all code/*.py with a count balance.
   Basename uniqueness includes repo-root names.
6. §6 O definition: driver-form field compare with the named exclusions. CLI compared with a base-commit CLI run.
7. §6 slice table: O required in S2, S5, S7 and S8. S1 exit adds the root conftest and sentinel. S3 exit 22/22. S5(d) a23 split in one
   commit and run alone. S5(e) both skills (or keep the script). S6 moved after S8 with a restricted path check. S7/S8 edit lists
   per section 2 items 5 and 7. Per-slice stop rule and clean `git status` check. A single push at S9 with the cron confirmed off.
8. §7: reword risk 4 (the hook does not run in vbt12). Add risks: wrong-checkout REPO_ROOT, observer re-release, STUDIES gap during
   S5–S8. Mark Hermes jobs as verified safe. Note the trader.py comments as left alone.
