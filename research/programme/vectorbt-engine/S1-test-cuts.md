# S1: test tiers, dead test files, CI split

Slice S1 of the study-isolation plan (branch `refactor/study-isolation`, from `46d8ff75`). Tests, scripts and CI only: nothing under `src/` changed, so no engine number can move (the 17-fold core17 run was repeated anyway, see section 6). Raw numbers: `p11-evidence/S1-*.json`.

Terms: **FAST CORE** = the default `scripts/run_tests.sh` run. **FULL** = every test except `integration` (the merge, deploy and vectorbt-upgrade gate). **Tier** = a named set of tests chosen by pytest markers.

## 1. Mechanism (one, stated plainly)

Membership of a test in a tier is decided by **per-file and per-test markers** (`pytestmark = pytest.mark.<name>` or a decorator), not by a central table of test names. `tests/tiers.py` holds only the other half: which markers each tier leaves out (`FAST_EXCLUDED`, `TIER_MARKEXPR`), so `scripts/run_tests.sh`, `--tier` in pytest and CI read one definition. `tests/test_tiers.py` fails if the markers registered in `pyproject.toml` and `FAST_EXCLUDED` differ. `tests/test_frozen_inputs.py` carries no marker: it is in the FAST CORE (it costs seconds and is what keeps the golden inputs from silently falling back to CPU-dependent generation).

| Tier | Selects | Command |
|---|---|---|
| FAST CORE | not `integration`, `slow`, `numba`, `tripwire`, `study`, `tooling` | `scripts/run_tests.sh` (no arguments) or `--fast [pytest args]` |
| FULL | not `integration` (also what a bare `pytest` runs) | `scripts/run_tests.sh --full [pytest args]` |
| STUDIES | `study`; `tests` plus every existing `research/*/*/code/tests` and `research/*/_lib/tests` folder, named explicitly, `--import-mode=importlib` when any exist; collecting nothing is an error | `scripts/run_tests.sh --studies [pytest args]` |

Any other argument list is passed to pytest unchanged and runs every test in the paths named (the old behaviour, so `scripts/run_tests.sh tests/engine -q` in the vbt-upgrade runbook is unchanged). The cgroup cap (8 GB, no swap) applies to all of them. New switches: `GGT_TEST_NO_CGROUP=1` (CI runners have no user systemd) and `GGT_TEST_DRY_RUN=1` (print the command; used by `tests/test_root_conftest.py`).

**Root conftest.** `conftest.py` at the repo root (moved with `git mv` from `tests/conftest.py`) holds the `_no_dotenv` safety fixture, the Agg backend, `--tier`, `--update-golden`, `--golden-reason` and `pytest_configure`. A test collected from anywhere in the repo, including `research/**/code/tests` later, therefore cannot read the real `.env`. `tests/test_root_conftest.py` proves it with a throwaway test file collected from outside `tests/`, shows the same file leaks the key when the root conftest does not apply, and checks that the `--studies` command line parses. The bare `research` folder must never be passed to pytest (10 frozen `test_*.py` files there import deleted modules).

Markers (`pyproject.toml`, markers only; `uv.lock` untouched):
- `slow`: `paper/test_live_golden` (file), `engine/test_check_vbt_engine_script` (file), `lab/wfo/test_cli_engine` (file), `test_scripts_have_help::test_script_help_exits_zero_with_usage` (42 subprocesses), `engine/test_selfcheck` (2 subprocess tests), `engine/test_golden` (the two golden compares), `lab/test_universe` (real-tape parity and the synthetic parity half), `lab/wfo/test_causality` (real-tape and synthetic halves), `lab/wfo/test_leak_golden_memory` (the two memory-budget tests that run a WFO and the four non-memory tests).
- `numba`: the `[numba]` ids of the `engine` fixture, of `test_provenance::test_requested_and_effective_engine`, and `test_golden::test_numba_and_rust_agree_on_every_scenario`, `test_selfcheck::test_numba_never_calls_rust`, `test_selfcheck::test_both_engines_simulate_the_same_value`.
- `tripwire`: `engine/test_warnings`. `test_defaults` and `test_tripwires` stay in the FAST CORE (they are in the mutation-proven set).
- `tooling`: `test_check_lens_report`, `test_check_reading_edition`, `test_vbt_latest`, `test_evidence_script_defaults`, `test_upgrade_tooling`, `test_upgrade_tooling_fixups`.
- `study`: the 21 R4 study files that move to `research/` in S5 (they stay in `tests/` for now).

## 2. Measured result

| | Before (`46d8ff75`) | After |
|---|---:|---:|
| FULL: passed / integration-deselected | 2,659 / 12 | **2,337 / 12** |
| FULL: seconds | 721 | **620** |
| FAST CORE: cases | none (every run was full) | **1,316** (1,033 deselected) |
| FAST CORE: seconds | none | **93.1 and 97.5** (two runs, host load average about 2.4 from a concurrent read-only verification workflow) |

FULL count check: 2,659 minus 345 removed ids plus 23 added ids is 2,337 exactly (`p11-evidence/S1-test-counts.json` lists removed and added ids by file). The 345 removed are the cuts of section 3 plus two renamed ids; the 23 added are the new tier and conftest tests (15), the study-code `pct_change` scan (1), the single legacy-import ban (1), the renamed real-tape parity test (1), and five cases of the intraday grid (5).

**FAST gate decision (recorded).** The first FAST measurement, with only the plan's markers, was **1,340 cases, 125.7 s** (the plan had said about 1,160 cases and 70-100 s; the independent check expected 1,200-1,280 and 100-120 s). Rather than relax the target, the three heaviest synthetic halves were marked `slow`: `test_universe::test_synthetic_parity_with_eligible_at` (about 14 s), `test_causality` synthetic panel (about 11 s) and the four non-memory tests of `test_leak_golden_memory` (about 8 s). That brought it to 93-97 s. The protection they gave stays in FULL, and in the FAST CORE the mutation-proven toy oracle, legacy-logic units, p5b fixes, causality teeth test, strategy lookahead test and the p5b perturbation tests remain. The arithmetic memory test (`test_every_planned_chunk_of_the_full_17_fold_run_fits_its_budget`, 0 s) also stays FAST on purpose: it is the OOM guard for the 17-fold run.

## 3. Cuts and what each loses

| Cut | Cases | Lines | Protection lost |
|---|---:|---:|---|
| Deleted `tests/scripts/test_research_reorg.py`, `test_research_retarget.py` | -155 | -2,565 | A safe re-run of two finished one-off tools (`scripts/research_reorg.py`, `scripts/research_retarget.py`; nothing imports the tests; the tools stay in `scripts/`, now untested, see section 7). |
| Deleted `tests/lab/test_strategy.py` | -2 | -16 | `LabConfig` default values (`top_n=50`, ...). The module is still used by `lab/data.py`, `lab/cli.py`, `lab/blend.py`. |
| Deleted `tests/lab/test_midcap_universe.py` | -2 | -14 | A 380-420 member-count check of the parked midcap400 snapshot and "midcap400 is a CLI choice". |
| Deleted `tests/lab/test_backfill_universe.py` | -1 | -16 | The same member-count check through a script that was executed at import. |
| `tests/paper/test_paper_boundary.py` from 50 to 3 cases | -47 | -15 | The per-file repeats of "no vectorbt import" and "no `.vbt` accessor" (32 cases; `tests/engine/test_boundary.py` scans every file under `src/` and ruff TID251 bans the import; checked by planting `import vectorbt` and `.vbt.rsi()` in `paper/risk.py`: the engine test fails). The 16 per-file legacy-import cases are replaced by one test over all files, so the rule is kept and only the per-file failure ids are lost. |
| `tests/engine/test_upgrade_tooling.py` 16 to 7, `_fixups.py` 17 to 9 | -17 | -180 | Runbook-wording pins (sections present in the skill, command names, "names a prefix glob still match something"), Dependabot ignore list, workflow cron text. Kept: workflow never merges, least token scope, no secret in the job that runs the new release, no credential persistence, every action pinned by SHA, Docker base pinned by digest, a bare pytest not collecting `research/`, workflow pytest lines naming `tests`, evidence script default to scratch, new-preset-has-a-test, and the two "not relevant" runbook-table guards (the skill cites them). |
| `tests/lab/test_intraday_momentum.py` constant-series grid 120 to 6 | -114 | +12 | None: one property, every case under 5 ms; the six cases use every length, level and lag of the old grid. |
| `test_universe` real-tape parity sample 150 to 20 random dates | 0 | 0 | Fewer random days (77 dates instead of 202; the 57 constituent-change neighbourhood dates stay; `scripts/p3_pit_parity.py` and the synthetic parity are unchanged). Saves about 125 s of FULL. |
| Patched `time.sleep`/`time.time` in `test_trader.py::test_log_pending_order_stamps_today_et_not_as_of` | 0 | +6 | None (15 s to milliseconds, same pattern as the sibling poll tests). |

**Kept, against the plan.** `tests/scripts/test_driver_default_paths.py` is **not** deleted (the independent check showed all 26 cases are real assertions and two more tests guard the lens2 and fold-attribution paths). It stays in the FAST CORE and is a port target in S5 (to the root-cause and wfo-methods studies). The paired "second independent" engine tests, the toy oracle, legacy-logic tests and `legacy_simulate.npz` are untouched.

## 4. Other changes

- `tests/test_pandas3_compat.py`: a new test scans `research/*/*/code` and `research/*/_lib` for bare `pct_change()` (passes trivially while they do not exist).
- `ruff.toml`: per-file ignore `"research/**/code/*.py" = ["E402"]` (study code calls its bootstrap before the ggTrader imports). `research/` stays excluded from lint; the study-code lint scan is S3.
- CI (`.github/workflows/ci.yml`): job `test` (kept as the job id, so a required check named "test" still works) = engine check + lint + FAST CORE, on pull requests and pushes to main; job `studies` = `scripts/run_tests.sh --studies` on pull requests and pushes to main; job `full` = engine check + the whole suite on pushes to main, manual runs and a weekly schedule (Mondays 07:15 UTC). Actions pinned as before; `pyproject.toml` changes only in the markers list. This replaces the plan's "FULL required on PRs": the independent check required lint + FAST + STUDIES on PRs and FULL at merge to main. The CI runner has no tape, so the live golden and the real-tape oracles still skip there: the merge and deploy gate is `scripts/run_tests.sh --full` run locally.
- Docs: `AGENTS.md` (testing paragraph; also the conftest sentence), `README.md` (file table row), `docs/architecture.md` (the stale `test_paper_boundary` reference and a test-tier paragraph), `.claude/skills/ggtrader-deploy` and `.claude/skills/vbt-upgrade` (the gate is `--full`; `test_replica` named in the deploy pre-flight), `docs/changelog.md`.

## 5. Oracles

See `p11-evidence/S1-oracles.json`: FULL green (2,337 passed), FAST CORE green twice, `ruff check .` and `ruff format --check .` clean in the worktree and in a fresh `git clone --local` of the commit, core17 reproduced through `scripts/p5_wfo_validation.py` with 0 differing fields and `ggTrader.__file__` under the worktree `src/`.

## 6. Stop rule

Not triggered: core17 (Sharpe 0.391, CAGR 3.50%, gates 1/17) equals the guarded evidence.

## 7. Unresolved

- `scripts/research_reorg.py` and `scripts/research_retarget.py` (about 2,900 lines) are now untested and have no user; deleting them (plan D6) is not part of this slice.
- FAST is 93-97 s on a loaded host; it was 125 s before the three synthetic halves were moved to FULL. An idle-host measurement should replace these two numbers.
- The independent check's STUDIES gate is satisfied by `tests` (21 study files marked `study`) until S5 moves them; after S5 the explicit `research/*/*/code/tests` paths supply the tests and pytest exit 5 guards against collecting none.
