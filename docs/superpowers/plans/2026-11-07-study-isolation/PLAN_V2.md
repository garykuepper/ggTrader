# Study isolation: executable move plan, v2 (PROPOSAL, not approved)

Tree `/home/flynn/ggTrader-vbt12`, branch `refactor/study-isolation`, base `46d8ff75`. Static analysis only (git, grep,
ast, wc); nothing in the repo was edited or run. Line counts are `wc -l` at `46d8ff75`. Effort is ESTIMATED agent-hours.
Inputs: PLAN.md, CORRECTIONS.md (all of section 2 and amendments 1-8 of section 5 are applied below, not referenced),
R1-R6, the engine spec. The machine-readable slices are in `slices.json` (same folder). Nothing is merged or pushed
without the owner; the one push to main happens after S9 and the owner's yes.

**Headline.** `src/ggTrader` 26,343 -> about **14,100** lines. `scripts/` 12,320 -> at least **4,138** (at least 4,028 if
the owner overrides lead call L3 and `shadow_cross_asset.py` moves too). The default test command goes from 2,659 cases
(~13 min) to a FAST tier of about **1,200-1,280 cases, ~100-120 s** (recomputed, not measured; S0 measures). No engine
number may change: oracle **O** (section 6) is run in S0, S2, every S5 sub-slice, S7, S8 and S9.

## 0. Changes from PLAN.md

1. Pins: **22 pins in 6 files, 15 match, 7 differ** (was 21/6). The missed pin is `tests/lab/test_intraday_observer.py`
   in `_intraday_observer_build_verification.json` (HISTORY blob `4ab01d1a`). Feasibility `code_sha256` resolves at
   **`6e89281c`**. `verify_pins` refuses a shallow clone and skips the deletion commit when it walks history.
2. `significance` serves **two** families (stock-signals via `momentum_screen.py:9`; audits via
   `production_replica_audit.py:20`), not three: `lens2_build.py` only names it in prose.
3. `a23_rebalance` row: `tests/lab/test_a23_rebalance.py:18` imports the static-leverage driver. New
   `research/allocation/_lib/a23_backtest.py` (`load_prices`, `run`, the constants they read). S5(d) is that split, done in
   **one commit, run alone**. It also removes the cross-study edge `test_century_leverage.py:19`.
4. Moved scripts: **23** (was "24"); lead call L3 keeps `shadow_cross_asset.py` in `scripts/`, so 22 actually move.
5. `test_driver_default_paths` is **kept** (all 26 cases are real); its two inventory tests are ported to their studies,
   and it resolves moved drivers through `CODEMOVED.csv`. The "half of it checks nothing" claim is withdrawn.
6. The root `conftest.py` (`_no_dotenv`, Agg, `pytest_addoption(--tier, --update-golden, --golden-reason)`,
   `pytest_configure(apply_tier)`) and a sentinel test are an **S1 exit condition**, checked by S0 before any move.
7. `bootstrap(__file__)` returns the repo root found by walking up from the study file; moved code never uses
   `utils.paths.REPO_ROOT` (the main venv's editable install can point at another checkout).
8. Zero-hit sweep (patterns in section 3) over tests, scripts, src, research code, `.claude`, CI files; current docs join
   it in S6. It now also matches `from ggTrader.lab import <moved>` (12 such lines at base; NEW).
9. String monkeypatch targets become object patches; the observer wrappers stay self-bootstrapping (stdlib only).
10. Help/AST test covers **every** research `code/*.py` with a `__main__` guard, plus a script-count balance (49).
11. Ruff: per-file ignore `E402` for `research/**/code/*.py`; an allow-list test for the `artifacts-*` exclude (ruff's
    `*` crosses `/`); basename uniqueness includes repo-root names (`scripts`, `tests`, `data`, `conftest`, ...).
12. STUDIES tier = explicit `research/*/*/code/tests` and `research/*/_lib/tests` paths with a minimum-collected guard;
    a STUDIES CI job (full history, `verify_pins`) from S3; `--studies` added to the deploy pre-flight in S5(g).
13. Slice order: S0, S2, S3, S4, S5a-b-c-d-f-g (serial), S7, S8, **S6 after S8**, S9. S6's path check covers only current
    docs (AGENTS.md, README.md, architecture, cli_reference, next_steps, `.claude/skills`), against an S0 baseline.
14. Per-slice **stop rule** (O or the 120/120 golden moves: revert, stop, report) and a clean
    `git status --porcelain research data results` after every slice.
15. O is the **driver-form field compare** with five named exclusions (section 6), not the CLI route.
16. O is also required in S2 (it touches `src/ggTrader/lab`). FAST figures corrected (item headline).
17. NEW (not in CORRECTIONS): `scripts/century_leverage_a23_x.py` replays commit `53b739c`'s A23 script, which imports
    `ggTrader.lab.fred_data`; S2's move breaks that replay unless the tool aliases the old name (`# history-read`).
18. NEW: wfo-methods would hold two files named `wfo_policy_synthetic.py` (module and driver); drivers are renamed.
19. NEW: two `_lib` tests exercise a study's driver (`test_momentum_screen.py:270-299`, `test_intraday_probe.py:427-446`);
    they move to that study's `code/tests`, so a library test never depends on a study.
20. NEW: S1 tagged `test_shadow_cross_asset.py` as `study`; it stays in `tests/` (L3) and is retagged `tooling`.
21. NEW: the docs path check fails at base already (placeholders such as `research/<category>/<study>/`); S6 checks
    "no new problem versus the S0 baseline", not "zero problems".
22. NEW: gate Y of `_century_leverage_a23_gate.json` (driver + `--frozen-inputs`, output sha256 `645d4ee2...`) is the
    S5(d) oracle for the A23 split.
23. Experiments: no `git worktree add` (it writes repo metadata); scratch clones only, moved into S3 (shallow clone),
    S7 step 0 and S8 step 0. Risk 4 reworded (the ruff hook never runs in vbt12). Hermes `jobs.json` verified safe.

## 1. Final classification

Labels as in PLAN.md: ENGINE stays in `src`; SHARED-UTIL -> `src/ggTrader/utils/`; FAMILY-LIB ->
`research/<category>/_lib/`; STUDY -> `research/<category>/<study>/code/`; DELETE (row in `research/CODEMOVED.csv`).

### 1a. `src/ggTrader` (107 files)

| Module(s) | Lines | Label | Destination / note | Importers that must change |
|---|---:|---|---|---|
| `__init__.py`, `engine/*`, `lab/wfo/*`, `lab/strategies/*`, `paper/*`, `cli/*`, `data/*` (incl. do-not-edit `cached_yfinance_loader.py`, dormant `data/core/constants.py`, D7) | - | ENGINE | stay | none |
| `lab/__init__`, `strategy`, `data` (incl. `eligible_at`), `universe`, `persist`, `negative_cache`, `gates`, `cli`, `allocation`, `blend`, `google_trends_data` | - | ENGINE | stay | none |
| `lab/a23_rebalance.py` | 319 | ENGINE (tax) | stays: the lot book behind the monthly `scripts/a23_rebalance_check.py` | `century_leverage` keeps importing `_trade_to`; `tests/lab/test_a23_rebalance.py:18` loses its driver import in S5(d) |
| `lab/significance.py` | 90 | SHARED-UTIL | `utils/significance.py`; `curve_summary` import made lazy | `momentum_screen.py:9`, `production_replica_audit.py:20`, `test_significance.py:9`, `test_production_replica_audit.py:10`. Frozen `legacy-scripts/audit_reruns_wfo.py:44` stays broken (as at base) |
| `lab/fred_data.py` | 145 | SHARED-UTIL | `utils/fred_data.py`; `get_engine` lazy | `static_leverage_aftertax.py:28`, `production_replica_audit.py:17`, `century_leverage_a23_x.py:19` (+ alias, item 17), `test_fred_data.py`, `test_century_leverage.py:18` |
| `utils/config,db_engine,paths,results,__init__` | 201 | ENGINE | stay; `paths` gains `REPO_ROOT` (src use only), delete dead `config.load_symbols_from_json` (`config.py:11`) | none |
| `utils/returns.py`, `lab/event_study.py` | 155 | DELETE | three frozen research files import `event_study` and are unrunnable already; left alone | `test_pandas3_compat.py` (imports :17, :19; classes `TestPaddedPctChange` 4, `TestEventStudyStack` 2), `test_event_study.py` deleted |
| `lab/french_data.py`, `momentum_screen.py` | 429 | FAMILY-LIB | `research/stock-signals/_lib/` | 3 drivers, `factor_screen`, `net_issuance_screen`, 4 tests |
| `lab/intraday_probe.py` | 579 | FAMILY-LIB, pinned | `research/intraday/_lib/` | 3 modules, 2 drivers, 4 tests |
| `lab/factor_screen.py`, `net_issuance_screen.py` | 352 | STUDY | `stock-signals/{factor-screen,net-issuance}/code/` | drivers, tests |
| `lab/century_leverage.py` | 271 | STUDY | `allocation/century-leverage/code/` | 2 drivers, test |
| `lab/replica.py` | 325 | STUDY | `audits/production-replica/code/` (allow-listed `ggTrader.paper` imports) | 2 drivers, test (`test_replica.py:203`) |
| `lab/tax_lots.py` | 583 | STUDY (D2) | `paper-system/tax-lots/code/` | test |
| `lab/wfo_attribution.py`, `wfo_policy_synthetic.py` | 183 | STUDY | `programme/wfo-methods/code/` | 2 drivers (renamed, L4), 2 tests |
| `lab/event_timing_{audit,sources,reconstruction}.py`, `event_acceptance_diagnostic.py`, `form4_data.py`, `earnings_surprise_data.py` | 5,268 | STUDY, pinned | `audits/event-timing/code/` | 4 drivers, 4 tests |
| `lab/intraday_momentum.py`, `intraday_momentum_data.py` | 1,651 | STUDY, pinned | `intraday/spy-momentum-feasibility/code/` | 2 drivers, 2 tests |
| `lab/intraday_observer.py` | 2,630 | STUDY, pinned, running | `intraday/spy-momentum-prospective-observer/code/` | 2 wrappers, test |

Moved or deleted: 12,426 lines in 21 files (19 lab modules move to research; `event_study` and `utils/returns` are
deleted). `significance` and `fred_data` (the other 2 of C01's 23) only change path inside src. New `utils` helpers ~200.
**src after ~14,100** (assumes D2; +583 if reversed).

### 1b. `scripts/` (49 tracked files)

| Label | Scripts (destination) |
|---|---|
| ENGINE/OPS, stay | `run_tests.sh`, `run_capped.sh`, `paper_trade.sh`, `daily_pnl_report.sh`, `legacy_env.sh`, `check_vbt_engine.py`, `vbt_latest.py`, `make_tape_snapshot.py`, `freeze_test_inputs.py`, `equity_backfill.py`, `google_trends_backfill.py`, `fetch_midcap400_snapshot.py`, `a23_rebalance_check.py`, **`shadow_cross_asset.py` (L3)** |
| ENGINE-ORACLE, stay | `p7_live_golden.py`, `p5_wfo_validation.py`, `engine_parity_evidence.py`, `p3_pit_parity.py`, `legacy_reference_runs.py`, `legacy_reference_compare.py`, `p5b_guard_compare.py` |
| TOOLING, stay / new | `check_lens_report.py`, `check_reading_edition.py`; new `verify_pins.py` (S3), `study.py` (S4) |
| STUDY, move (22) | century-leverage `code/`: `run.py` (<- `century_leverage.py`), `century_leverage_a23_x.py`, `century_leverage_calibrate.py`; static-leverage `code/run.py` (<- `static_leverage_aftertax.py`, minus the `_lib` split); `factor_french_screen.py`, `momentum_french_screen.py`, `net_issuance_screen.py` -> `<study>/code/run.py`; production-replica `code/run.py` (<- `production_replica_audit.py`), `replica_parity_check.py`; wfo-methods `code/run_fold_attribution.py`, `code/run_policy_synthetic.py` (L4); root-cause `code/`: `lens2_build.py`, `root_cause_figures.py`, `guard_snapshot.sh`; event-timing `code/run.py` (<- `event_timing_audit.py`), `insider_acceptance_diagnostic.py`, `form4_backfill.py`, `earnings_surprise_backfill.py`; feasibility `code/run.py` (<- `intraday_momentum_research.py`), `intraday_coverage_probe.py`; observer `code/intraday_observe.py`, `intraday_observer_job.py` |
| DELETE (D6) | `research_reorg.py`, `research_retarget.py`, `migrate_paper_dates_20260822.py`, `one_off/restate_mnst_snapshots_20260911.py` (3,203 lines). `trader.py:81` and `:1099` keep their comments citing the migrate script (editing them touches the image); CODEMOVED resolves them |

Moved 4,979 lines (5,089 with shadow), deleted 3,203. **scripts after >= 4,138** (a floor: it excludes `verify_pins.py`
and `study.py`). Balance: 49 at base = files still in `scripts/` (minus the two new ones) + CODEMOVED rows from `scripts/`.

## 2. Hash-bound evidence

| Binding file | Code pins | Today | After | Slice |
|---|---|---|---|---|
| `audits/event-timing/_event_evidence_resolution_workbook.json` | loader, `event_timing_sources`, `_reconstruction`, `form4_data`, `earnings_surprise_data` MATCH; `event_timing_audit` blob `740781a6` | 5 MATCH, 1 HISTORY | loader MATCH; 4 siblings MOVED-MATCH (byte-identical, do-not-edit); audit HISTORY | S7 |
| `audits/event-timing/_event_audit_detector_provenance.json` | `scripts/production_replica_audit.py` blob `8bc365ee` | HISTORY | HISTORY | S5(g) |
| `intraday/.../_intraday_observer_activation_proposal.json`, `_release_verification.json`, `artifacts-2026-10-26-intraday-readiness/unactivated-proposal-2026-10-26.json` | observer, `intraday_observe.py`, `intraday_observer_job.py` MATCH; probe blob `c20be227` | 3 MATCH + 1 HISTORY each | all HISTORY (reviewed commit `280d4a0`) | S8 |
| `intraday/.../_intraday_observer_build_verification.json` | observe MATCH; observer **and `tests/lab/test_intraday_observer.py`** blob `4ab01d1a` | 1 MATCH, 2 HISTORY | 3 HISTORY | S8 |
| `_intraday_observer_rehearsal.json`, `_activation_verification.json` (canonical `CODE_FILES` hash; `f0320a3e` at `280d4a0`), `_readiness_preflight.json`, feasibility `_intraday_momentum_validation*.json` (`code_sha256` `f0b85690` at **`6e89281c`**) | hashes of `{path: sha}` maps | verify at the recorded commit | unchanged (recomputed from `git show <commit>:<old path>`) | S8 |
| Engine provenance blocks | commit, lock, strategy files | strategies stay | unaffected | - |

**22 per-file pins, 15 MATCH, 7 HISTORY; every one of the 7 resolves to a historical blob of the same path.** Expected
`verify_pins` totals: S3-S5 `MATCH=15 MOVED-MATCH=0 HISTORY-MATCH=7`; after S7 `11/4/7`; after S8 `1/4/17`; FAIL 0 always.
The docs plan digest checked by `tests/scripts/test_recorded_hashes.py` is separate and unchanged.

Mechanism:
1. `research/CODEMOVED.csv`, append-only: `old_path,new_path,kind{moved,moved+edited,deleted},base_commit,move_commit,`
   `sha256_before,sha256_after,do_not_edit{yes,no},note`. One row per path under `src/`, `scripts/` or `tests/` that existed
   at `46d8ff75` and was moved or deleted by this programme (S3 backfills S1/S2's rows). A derived new file (the A23
   `_lib`) is named in its parent's `note`. `research/README.md` names it next to `MOVED.csv` (S6).
2. `scripts/verify_pins.py` (S3): walks every JSON under `research/`, collects `(path, sha256)` pairs and canonical maps;
   resolves MATCH, MOVED-MATCH (CODEMOVED `new_path`), HISTORY-MATCH(commit) (`git log --format=%H -- <old>`, skipping
   commits where the path is deleted, then `git show <c>:<old>`), else FAIL. Exits 2 with "shallow clone" when
   `git rev-parse --is-shallow-repository` is true. `--expect MATCH=a,MOVED-MATCH=b,HISTORY-MATCH=c,FAIL=0` exits 1 on any
   deviation; `--summary` prints one line.
3. FAST test: every CODEMOVED row with `do_not_edit=yes` has `sha256(new_path) == sha256_after` (the only guard on the 4
   event-timing siblings; C07).
4. No stubs or copies at old paths. Frozen reports keep their old paths; CODEMOVED resolves them like `MOVED.csv`.

**Observer.** Units run `releases/280d4a0.../scripts/intraday_observer_job.py` with their own runtime and a credentials
file; no unit names the repo, and the runtime has no editable ggTrader. Nothing touches the release, runtime,
credentials or timers (S8 checks them read-only against S0's record). The repo copy is edited as in S8; `deploy/`
templates and the existing runbook commands stay (they describe `280d4a0`); a new runbook section covers releases cut
after the S8 commit.

**Event timing.** Audit, sources and reconstruction stay in one directory (`with_name` hashing, `event_timing_audit.py:1545`).
Edits: imports `:26`, `:2169`, `:2690`; `ROOT` (`:43`) becomes `repo_root(__file__)` (name kept: the test patches it,
`test_event_timing_audit.py:2378`); `event_acceptance_diagnostic.py:13,:16`. Old checkpoints do not resume (accepted).

## 3. Import and path rewrites

- **`ggTrader.utils.study` (S2).** `repo_root(file)`: first ancestor of `file` holding `pyproject.toml` and `.git`.
  `bootstrap(file) -> Path`: nearest ancestor named `code` or `_lib`; for `code` puts `<study>/code` and
  `<category>/_lib` (if present) at the front of `sys.path`, for `_lib` puts `_lib`; idempotent; returns
  `repo_root(file)` (gap 8). `study_dir(file)`: the folder above `code`. `load_driver(path)`: `spec_from_file_location`,
  name `study_<category>_<study>_<stem>`. src never calls them (import-rule test).
- **Drivers** (`run.py`, named drivers) start with `from ggTrader.utils.study import bootstrap` / `ROOT = bootstrap(__file__)`,
  keep argparse, `main()`, guard, side-effect-free `--help`; never imported by name; tests use `load_driver`.
  `code/tests/conftest.py` and `_lib/tests/conftest.py` call `bootstrap(__file__)`.
- **Intraday family (L5)** stays ggTrader-free: roots are `Path(__file__).resolve().parents[N]` with a `# root-ok`
  comment; wrappers set `sys.path[:0] = [code dir, research/intraday/_lib]` with stdlib only (no `src`).
- **Library basenames** stay repo-unique; a FAST layout test rejects duplicates across `research/*/*/code` and
  `research/*/_lib` (excluding `run.py`, `conftest.py`, `test_*.py`) and any shadowing of stdlib, installed top-level
  modules or repo-root names (`scripts`, `tests`, `data`, `src`, `docs`, `research`, `conftest`). Studies never import
  another study's `code/`; a second category promotes to `src/ggTrader/utils` with its tests in the same commit.
- **Rewrite patterns.** `from ggTrader.lab.X import` / `from ggTrader.lab import X` / `import ggTrader.lab.X as Y` ->
  `from X import` / `import X` / `import X as Y`; `from scripts import D` / `scripts/D.py` / `parents[2] / "scripts"` ->
  `load_driver(<code dir> / "<new name>.py")`; string patches `monkeypatch.setattr("ggTrader.lab.M.time.sleep", f)` ->
  `monkeypatch.setattr(M.time, "sleep", f)` (same object is patched). Root lines (`parents[N]`, `.parent.parent`,
  `find_project_root`, `sys.path.insert/append`) -> `bootstrap`/`repo_root`/`study_dir`.
- **Zero-hit sweep** (per slice, cumulative names; exact commands in `slices.json`): `git grep -nE` for
  `ggTrader\.lab\.(<L>)`, `from ggTrader\.lab import (<L>)`, `lab/(<L>)\.py`, `scripts/(<S>)\.py`,
  `from scripts(\.| import )(<S>)` over `src scripts tests ggt.py pyproject.toml ruff.toml Dockerfile .github .claude`
  and `research/*/*/code/**`, `research/*/_lib/**`; S6 adds AGENTS.md, README.md, `docs/{architecture,cli_reference,next_steps}.md`.
  Exempt (L6): `src/ggTrader/paper/trader.py` (comments), `lens2_build.py` (frozen prose that renders a published lens),
  `tests/scripts/test_driver_default_paths.py` (historical names resolved via CODEMOVED), lines marked `# history-read`
  (A23 replay), `deploy/` (describes the running release), and history docs (changelog, archive, superpowers).
  A FAST test (`tests/scripts/test_codemoved_references.py`, S3) does the same from CODEMOVED rows, so CI enforces it.
- **Root sweep:** `parents\[[0-9]+\]|\.parent\.parent|find_project_root|sys\.path\.(insert|append)|sys\.path\[` over research
  code returns only `# root-ok` lines.
- **NEW: A23 replay alias.** `century_leverage_a23_x.py` gets `sys.modules.setdefault("ggTrader.lab.fred_data",
  ggTrader.utils.fred_data)` before `exec` (`# history-read`); the replayed `53b739c` logic is unchanged.
- **pyproject.** `where=["src"]`, `testpaths=["tests"]`, `pythonpath=["."]`, `addopts` unchanged; markers from S1; fix
  the `scipy` (S2, S5a) and `beautifulsoup4` (S7) comments. No dependency moves (would change `uv.lock`).
- **Root conftest (S1, verified in S0).** `_no_dotenv`, `matplotlib.use("Agg")`, `pytest_addoption` (`--tier`,
  `--update-golden`, `--golden-reason`), `pytest_configure(apply_tier)`; S3 adds `--min-collected`.
  Moved code still reads `os.environ` directly (SEC_*, APCA_*, POSTGRES_CONNECTION_STRING); unchanged from today.
- **ruff (S3).** `extend-exclude = ["research/programme/vectorbt-engine", "research/*/artifacts-*"]`, `force-exclude`
  kept, per-file ignore `"research/**/code/*.py" = ["E402"]`. Because `*` crosses `/`, the layout test forbids an
  `artifacts-` segment under any `code/` or `_lib/`. Boundary test scans research code and `_lib`.
- **Docker.** `research/` is in `.dockerignore`; `scripts/` and `tests/` are not, so the image loses study code only
  because it leaves those folders. The trader's import closure (74 modules) holds no moved module.

## 4. Tests

**Tiers (L1).** `tests/tiers.py` (S1) stays the one tier -> marker-expression table; membership is a marker in each file.
FAST = `tests/` minus `integration slow numba tripwire study tooling`. FULL = `pytest --tier full tests` **then** the research
half. STUDIES = `pytest --tier studies tests` (exit 5 "nothing collected" accepted) **then** the research half. Research half
= `pytest --import-mode=importlib -m "not integration" --min-collected N <existing research/*/*/code/tests and
research/*/_lib/tests dirs>` (`N = tests/tiers.py:STUDIES_MIN_COLLECTED`, raised to the measured count by every moving slice).
Never pass `research` whole (10 frozen `test_*.py` import deleted modules). `test_frozen_inputs.py`: FAST unless S0
measures it >= 3 s. After S8 a FAST test asserts no `tests/` file carries the `study` marker.

**Where tests go.** `_lib/tests`: `test_french_data`, `test_momentum_screen` (minus its driver test), `test_intraday_probe`
(minus its help test), new `research/allocation/_lib/tests/test_a23_backtest_parity.py` (the 2 frozen-parity tests of
`test_a23_rebalance`, L2). Study `code/tests`: factor-screen, net-issuance, momentum (driver test), century-leverage,
static-leverage, wfo-methods (2 + ported `test_fold_attribution_sources_exist`), tax-lots, root-cause (figures + ported
`test_lens2_resolves_every_inventory_artifact`), production-replica (2), event-timing (4), feasibility (2 + coverage-probe
help test), observer. **Stay in `tests/`:** the 12 pure tests of `test_a23_rebalance`, `test_a23_rebalance_cli`,
`test_google_trends_data`, `test_shadow_cross_asset` (retagged `tooling`), `tests/utils/test_significance`,
`tests/utils/test_fred_data`. `test_event_study` is deleted.

**Cuts (S1) and their losses, corrected.** `test_research_reorg`/`_retarget` go with the tools; `test_strategy`,
`test_midcap_universe`, `test_backfill_universe` go (LabConfig pins, midcap count check). `test_driver_default_paths` is
**not** cut (restored in S2 if S1 removed it). `test_paper_boundary` keeps one non-parametrised ban on `ggTrader.lab.strategy`
and the deleted names; the 32 per-file vectorbt/accessor repeats are true duplicates of `engine/test_boundary` + TID251.
`test_universe` 150 -> 20 dates, sleep patch at `test_trader.py:653`, upgrade-tooling trims, intraday grid 120 -> 6: as PLAN.md.

**CI (L12, reconciled with S1).** `test` (FAST + lint + engine check) on PRs and pushes; `full` on pushes to main, weekly,
manual; **`studies`** (S3: `fetch-depth: 0`, `verify_pins.py`, STUDIES tier) on PRs, pushes and weekly, no path filter
(a path-filtered required check stays pending). Required on PRs: `test`, `studies`. Every workflow `pytest` line names a
`tests` path (`test_upgrade_tooling_fixups`). CI cannot run tape oracles (tape git-ignored), and cannot run at all until the
owner allows a push (L13), so every slice gate is local.
**Merge/deploy checklist.** Local `--full` with the tape present, `--studies`, `verify_pins.py`, O. Before a trader deploy
(`ggtrader-deploy`, from S5g): `--full`, `--studies` (it holds `test_replica`, which exercises `paper` internals),
`p7_live_golden.py` 120/120, image import check.

**Counts after.** `tests/` ~19,400 lines, ~1,570 cases; FAST ~1,200-1,280 cases, ~100-120 s of test time (~100-115 s
wall after the sleep patch); the S1 FAST gate is set from S0's idle-host measurement. Research test code ~11,000 lines.

## 5. Study scaffold (S4)

`scripts/study.py` (argparse, `--repo-root` for scratch use, no DB, no network; later `ggt study`): `new <category> <slug>
--kind {screen,wfo,event,audit,prospective} --placement <1-9>` (placement row must map to the category, slug unique);
`check <path>` (naming, `study.toml` schema via `tomllib`, brief fields, `code/run.py` AST rules, boundaries incl. the
**cross-study import rule**, unique basenames, provenance block, `PIN.json` hashes, report paths via
`check_lens_report --paths-only`); `repro <path> [--print-only]` (scratch clone at `git_sha`, never `git worktree add`;
`uv.lock` and tape hashes; recorded argv; `engine.provenance.compare` + numeric diff; legacy PIN prints `legacy_env.sh`);
`close <path> --verdict {GO,NO-GO,UNRESOLVED,METHODS}` (writes `PIN.json` and a report skeleton rendered from the existing
`research/TEMPLATE-research-report.md`, which stays byte-identical, L19). New file `research/TEMPLATE-brief.md` (the only
new non-code research document; in the F allow-list). Tree, `study.toml` and `PIN.json` schema as PLAN.md section 5.
Closed studies get no retrofitted `study.toml`; the paper-import allow-list is a constant in the layout test (L18).
Helpers: `utils/io.py`, `manifest.py`, `binding.py` (vectorbt-free git/lock fields; a test asserts the same `git_sha`
and `lock_sha256` as `engine.provenance`), `paths.REPO_ROOT` + `EVAL_START_17`/`EVAL_END_17`, `study.py`;
`significance.holm()`. Pinned or closed code is not migrated to them. HAC (D5): keep both forms, document each study's.

## 6. Oracle, invariants and slices

**O (full commands in `slices.json` "oracle").** (a) `ggTrader.__file__` lies in the tested checkout. (b)
`scripts/run_capped.sh .venv/bin/python scripts/p5_wfo_validation.py --runs core17_ensemble_sp500,core17_ensemble_sp500_dsr0
--out-dir <scratch> --force`. (c) Field compare of `_p5_<run>.json` with
`research/programme/vectorbt-engine/p5-evidence/guarded/` after removing exactly `provenance`, `wall_seconds`,
`diagnostics.timings_s`, `diagnostics.peak_rss_mb`, `tape.path`; expected core17 Sharpe 0.391 / CAGR 3.50% / gates 1/17,
dsr0 0.637 / 3.41% / 17/17. (d) `cmp` of both `*_oos_equity.csv`. `_p5_compare_*` files are derived and not compared.
(e) `p7_live_golden.py --out <scratch>` exit 0 with `sessions_equal == 120`, and `scripts/run_tests.sh --full
tests/paper/test_live_golden.py` with 2 passed, 0 skipped. (f) `scripts/run_tests.sh --full` and `--studies` green.
(g) ruff check + format in a fresh clone of HEAD. (h) F and a clean `git status`. The CLI route is not in O (it differs from
the driver in 5 fields by design; `tests/lab/wfo/test_cli_engine` covers it in FULL) (L15).

**F (research documents byte-identical).** `git diff --no-renames --name-status 46d8ff75 HEAD -- research` lists only A under
`research/*/*/code/`, `research/*/_lib/`, `research/CODEMOVED.csv`, `research/TEMPLATE-brief.md`, the four S1 evidence files (A: `research/programme/vectorbt-engine/S1-test-cuts.md`, `.../p11-evidence/S1-oracles.json`, `S1-test-counts.json`, `S1-timing.json`; added by commits 50b67e4b and dda77c35, allow-listed by exact path in the S0 round-2 amendment), and M of `research/README.md`.

**Per-slice rules.** One slice at a time, its own commits on the branch: `git mv` first, edits second, so renames show.
Checks run after the slice's commits. Every slice: its `mechanical_checks`, FAST (wall <= 1.10 x S0's), FULL, STUDIES,
F, `test -z "$(git status --porcelain --untracked-files=all research data results)"`, and from S3 `verify_pins --expect`.
**Stop rule:** if O or the 120/120 golden moves, `git revert --no-edit <slice commits>`, stop and report; never fix forward
without the lead. Any other check failing twice after a fix attempt: stop and report. Never force-push, never push.
Models (AGENTS.md): Sonnet builds; Haiku inventories (checked by a stronger model); Opus reviews S3 rules, S5d, S7, S8 and
the final diff in a fresh context. Ask the owner for quota headroom first; report usage after each slice.

| Slice | Content | Depends | Oracle | Effort |
|---|---|---|---|---:|
| S0 Baseline | verify S1 exit (root conftest, sentinel, boundary ban); env check; O at branch base; FAST/FULL timings on an idle host; gates X and Y of the A23 gate; doc path-check, unit-file and release baselines | S1 | O | 2 |
| S2 Dead code + utils | delete `event_study`, `utils/returns`, `load_symbols_from_json`, 4 tools (D6); `significance`, `fred_data` -> utils (lazy imports); A23 replay alias; new utils + tests; restore `test_driver_default_paths` | S0 | O | 3.5 |
| S3 Plumbing | CODEMOVED (+backfill), `verify_pins`, ruff, layout/help/boundary/pct_change/do-not-edit/codemoved tests, studies runner + `--min-collected`, CI `studies` job, spec addendum, shadow retag | S2 | none | 6 |
| S4 Scaffold | `study.py`, `TEMPLATE-brief.md`, tests | S3 | none | 6 |
| S5a stock-signals | `_lib` (french, momentum_screen) + factor-screen, momentum, net-issuance | S4 | O | 1.5 |
| S5b wfo-methods | 2 modules, 2 drivers (renamed), 2 tests + ported inventory test | S5a | O | 1 |
| S5c tax-lots | module + test | S5b | O | 0.5 |
| S5d A23 split (alone) | `_lib/a23_backtest.py`, century-leverage, static-leverage, test_a23_rebalance split, gate Y | S5c | O | 2.5 |
| S5f root-cause tools | `lens2_build`, `root_cause_figures`, `guard_snapshot.sh`, test + ported inventory test | S5d | O | 1.5 |
| S5g production-replica | `replica`, 2 drivers, 2 tests; deploy pre-flight gets `--studies` | S5f | O | 1.5 |
| S7 Event timing | 6 modules, 4 drivers, 4 tests; 4 byte-identical siblings | S5g | O | 4 |
| S8 Intraday | probe `_lib`, feasibility, observer; CODE_FILES; wrappers; runbook section | S7 | O | 5.5 |
| S6 Docs, skills | current docs, research/README, skills, changelog; path check vs S0 | S8 | none | 3.5 |
| S9 Final | fresh-context Opus review, O, timings, line counts, owner summary; one push after the owner's yes | S6 | O | 2 |

Total ~41 agent-hours (ESTIMATE). Strictly serial (the S5 sub-slices share CODEMOVED, `tests/tiers.py` and the sweep).

## 7. Risks, owner decisions, lead calls, limits

**Risks.** (1) Silent coverage loss: FAST stops running the golden and tape oracles; mitigated by `--full`/`--studies` in
the deploy and upgrade skills, the reminder line, the checklist. (2) Study tests escaping the `.env` guard: root conftest
gated by S0. (3) Lint over/under-exclusion: directory excludes + allow-list test. (4) Pins: the Claude ruff hook hardcodes
`REPO=/home/flynn/ggTrader`, so it never runs in vbt12; after merge it could reformat a do-not-edit file in the main
checkout; the FAST sha test catches it (history still verifies). (5) **Wrong-checkout root:** the main venv's editable
install points at `/home/flynn/ggTrader/src`; O's env check and `bootstrap`'s walk-up prevent a silent cross-checkout run.
(6) **Observer re-release:** a release cut from the new layout needs edited unit templates; documented, owner-gated.
(7) **STUDIES gap during S5-S8:** not-yet-moved study tests run in FULL; moved ones in the research half; the minimum
guard stops a silent zero. (8) Rot of closed study code after an upgrade: PIN + legacy tag; weekly STUDIES shows it.
(9) A23 replay (item 17) breaks unnoticed without the S2 alias and gate-X check. Hermes `jobs.json` holds no job that
runs a moved script (verified); the crontab was not read.

**Owner decisions (plain words; default first).** D1 study code next to its report in `code/` (or a separate `studies/`).
D2 keep the A23 lot book in base code, move the unused tax-lot library to its study. D3 move the recorder's repo copy
(the running one keeps its own frozen copy). D4 default test command = fast set, ~1.5-2 min; full set required before
merge or deploy. D5 keep both significance forms. D6 delete finished tools. D7 keep crypto constants and Tiingo loader.
D8 keep `_<slug>_results.json`. D9 allow `code/`, `code/tests/` and one `_lib` per topic. **D10 (new)** keep the shadow
cross-asset report in `scripts/` as a daily tool (default) or move it to its study.

**LEAD CALLS (choices CORRECTIONS left open; reason after the colon).**
- L1 `tests/tiers.py` (S1) is the one tier table, membership is a marker in each file: CI and `run_tests.sh` already read it.
- L2 A23 `_lib` gets `load_prices` (`:47-132`), `run` (`:136-226`) and the constants they read, importing `Book, Lot,
  _trade_to, COST_BPS` into its own namespace; parity tests go to `research/allocation/_lib/tests/`: the frozen test patches
  `a23._trade_to`, and a test belongs next to the module it imports.
- L3 `shadow_cross_asset.py` stays in `scripts/`, its test retagged `tooling`: it reads live `paper_snapshots` and
  `daily-trader-check` runs it with stderr hidden, so a move would fail silently (gap 10's recommended option).
- L4 wfo-methods drivers become `run_fold_attribution.py` and `run_policy_synthetic.py`: a module already owns the name.
- L5 the intraday family stays ggTrader-free (stdlib roots marked `# root-ok`): a future release venv has no ggTrader.
- L6/L7 sweep exemptions as in section 3; current docs join the sweep only in S6: docs move last (gap 11).
- L8 keep `test_driver_default_paths`; `_script_path` tries live path, CODEMOVED, then the archive: all 26 cases are real.
- L9 a test that runs a study's driver lives in that study: otherwise a `_lib` test depends on a study.
- L10 CODEMOVED covers src, scripts and tests: the observer test is itself pinned.
- L11 two pytest invocations per tier plus `--min-collected`: importlib mode stays off `tests/` (only proven on studies).
- L12 the `studies` CI job has no path filter: a path-filtered required check never reports on unrelated PRs.
- L13 every gate is local until the owner allows a push: no branch push without the owner.
- L14 experiments run in scratch clones inside S3/S7/S8: `git worktree add` writes repo metadata; verify_pins exists only from S3.
- L15 O is the driver route; an S0 mismatch stops the programme: re-baselining would hide a moved number.
- L16 FAST wall gate = 1.10 x S0's idle-host time: the plan's 100 s figure was never measured.
- L17 do-not-edit = CODEMOVED flag + FAST sha test: no other test holds those 4 pins (C07).
- L18 the paper-import allow-list is a constant in the layout test: closed studies get no `study.toml` retrofit.
- L19 closing skeleton is rendered from the existing report template: keeps F to one new research document.
- L20 S5d must reproduce S0's gate-Y hash (`645d4ee2...` if S0 reproduced it): proves the split moved no A23 number.
- L21 `test_import_is_inert` also asserts `ggTrader` is absent: the rewrite must not weaken it.
- L22 `--studies` joins the deploy pre-flight in S5g: that is when `test_replica` leaves `tests/`.

**Not verified.** No oracle or test was run (O, dsr0, golden, timings, gates X/Y argued from structure); `--import-mode=importlib`
on the full real study set; `uv build` / Docker context size; the crontab; whether a subprocess test spawns a
config-reaching script; exact `utils` line count; whether `static_leverage_aftertax.py --frozen-inputs` reproduces gate Y
at base (S0 decides); whether S1 lands the root conftest as specified (S0 checks).

## 8. Self-check (facts in slices.json, re-verified 2026-11-07; read-only, scratch writes only)

| # | Fact / command | Method | Result |
|---|---|---|---|
| 1 | All 202 command strings (13 oracle + 189 slice checks) | `bash -n -c` on each | 0 syntax errors |
| 2 | O compare program | `ast.parse`; run on scratch copies of the guarded JSONs; then with only excluded fields changed; then `gates` changed | parses; FIELD-EQUAL 0.3905 / 3.501 / 1/17 and 0.6374 / 3.406 / 17/17, rc 0; excluded-only change rc 0; real change rc 1 |
| 3 | Golden check command | run against committed `p7-evidence/p7_live_golden.json` | rc 0 (120/120) |
| 4 | Shallow / full clone of the worktree via `file://` | `git clone --depth 1` and full clone into scratch | shallow reports `true`, branch `refactor/study-isolation`; full clone checks out HEAD |
| 5 | Ruff stdin exit codes used in S3 | bad source on `scripts/zz_probe.py`; same on an `artifacts-*` path | rc 1 and rc 0 |
| 6 | F check | at HEAD (= base + S1 evidence) and on a synthetic diff | passes; synthetic flags exactly the `M report.md` and `A new.md` rows |
| 7 | Script balance | `git ls-files scripts` count; awk on a synthetic CODEMOVED | 49; awk counts only `scripts/` rows |
| 8 | Name sweeps | S2, S5d and S6 patterns run against `46d8ff75` | hits are exactly the files each slice edits (S6: 63 files incl. architecture, cli_reference, pyproject) |
| 9 | Line numbers | `git show 46d8ff75:<file> \| sed -n` | `test_a23_rebalance.py:18`, `config.py:11`, `century_leverage_a23_x.py:19/38/45/47/72`, `significance.py:11/76`, `lens2_build.py:22`, `root_cause_figures.py:20`, `trader.py:81`, `pyproject.toml:23/39`, `event_timing_audit.py:26/43/1545/2169/2690`, `intraday_observer.py:32/46/47`, `intraday_momentum.py:65-70`, `test_intraday_observer.py:18/248/283/1869/1891/2298/2315`, the 7 string patches and 5 local imports of `test_event_timing_audit.py`: all as stated. Corrected during this pass: `load_prices` ends `:132`, `run` is `:136-226`, the driver tests start at `test_momentum_screen.py:270` and `test_intraday_probe.py:427` |
| 10 | Paths | `ls` / `git ls-tree` | guarded JSON + CSVs, `p7_live_golden.json`, `scripts/{p5_wfo_validation,p7_live_golden}.py`, `run_capped.sh`, `data/external/a23_inputs/` (5 files), the A23 gate JSON (runs.Y sha `645d4ee2...`), observer release at HEAD `280d4a0` and clean, 8 unit files with no repo path: all exist as stated |
| 11 | NEW alias need (item 17) | `git show 53b739c:scripts/static_leverage_aftertax.py` | line 26 `from ggTrader.lab.fred_data import load_fred_series`: confirmed |
| 12 | S1 state (in progress) | `git status` of the worktree | root `conftest.py` absent so far, `tests/test_root_conftest.py` untracked, `test_driver_default_paths.py` staged for deletion: S0's checks and S2 step 1 cover both outcomes |

Not checked here: the commands that need the tape, the venv runs of tests, `verify_pins.py`/`study.py` (not written yet).
