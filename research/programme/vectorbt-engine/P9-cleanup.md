# P9 cleanup: last items of the vectorbt engine rebuild

**Date:** 2026-11-06. **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed). Terms: **AST** = abstract syntax tree (the script text is parsed, never run). **Tape** = the frozen daily-bars file. **Scratch** = a folder under the system temp directory.

Nothing was run against the broker, the database, Docker, cron or Telegram; no `.env` was read; `data/` and the research reports were not touched (only this note and one line in `README.md` of this folder). Every script check below was by AST first; `--help` was run only on scripts the AST check had cleared, with an `env -i` style environment, a scratch working directory and a 20 s timeout.

## 1. What changed

| Item | Result |
|---|---|
| `engine.stats._std` warning | `warnings.catch_warnings` plus `np.errstate` around `np.nanstd` (the warning is not controlled by `errstate`). Same NaN, other values bit-identical (`np.array_equal` against `np.nanstd`). Tests in `tests/engine/test_stats.py` run under `filterwarnings("error")`; the filter that hid the warning in `test_tripwires.py` is removed. |
| Changelog | P8b entry pasted from `P8b-fixups.md`; the P5c entry already existed; a P9 entry added. |
| Evidence-script defaults | Six scripts default to `<tmp>/ggtrader_evidence/...`: `p5_wfo_validation.py` (`--out-dir`), `p7_live_golden.py`, `p3_pit_parity.py`, `legacy_reference_compare.py`, `engine_parity_evidence.py` (already scratch; the unused `OUT_DIR` constant removed) and `legacy_reference_runs.py` (`--out-dir`). CLI otherwise unchanged. `tests/scripts/test_evidence_script_defaults.py` (AST) pins it. The `vbt-upgrade` skill text was updated. Not changed: the study drivers (`factor_french_screen`, `momentum_french_screen`, `net_issuance_screen`, `production_replica_audit`, `wfo_fold_attribution`, `wfo_policy_synthetic`, `static_leverage_aftertax`, `intraday_coverage_probe`, `lens2_build`) still default into their own `research/<study>/` folder; they reproduce a report's `_results.json` and `tests/scripts/test_driver_default_paths.py` pins those paths. |
| Scripts without `--help` | Re-derived by AST scan of `scripts/*.py` and `scripts/one_off/*.py`: the 14 of P5c-completion plus the one-off `restate_mnst_snapshots_20260911.py`. **Edited (parser, `main(argv)`, guard; description from the module docstring; no new arguments):** `earnings_surprise_backfill`, `fetch_midcap400_snapshot`, `fomc_calendar_backfill`, `form4_backfill`, `google_trends_backfill`, `headline_sentiment_backfill`, `house_ptr_backfill`, `short_interest_backfill`, `short_volume_backfill`, `one_off/restate_mnst_snapshots_20260911`, and `mcp_server` (parser runs before the optional `mcp` import so `--help` works where `mcp` is not installed; `mcp` is not a project dependency). **Not edited, because they already hand over to a library parser that parses first:** `event_timing_audit`, `insider_acceptance_diagnostic`, `intraday_observe`, `intraday_observer_job` (the test follows the hand-over and checks the library function). **Cannot be made safe: none.** |
| Script test | `tests/scripts/test_scripts_have_help.py`: 48 AST tests (parser, main guard, no module-level work beyond constants and paths, `parse_args` before any other call in the entry function) and 48 subprocess `--help` tests (exit 0, `usage` in stdout, nothing written to the working directory); about 45 s. |
| Flattening | `lab/strategies/v2/*` to `lab/strategies/*` (10 modules; the package `__init__` is the old `v2/__init__` with a merged docstring; no name clash: `strategies/` held only `vote.py`), `lab/wfo_v2/` to `lab/wfo/` (no legacy `lab/wfo.py` remained), `tests/lab/strategies_v2` to `tests/lab/strategies`, `tests/lab/wfo_v2` to `tests/lab/wfo` (goldens move with them). All imports in `src/`, `tests/`, `scripts/`, `AGENTS.md`, `README.md`, `docs/architecture.md`, `docs/cli_reference.md` and `.claude/skills` rewritten. Kept: `REGISTRY_V2`, `LEGACY_ONLY_STRATEGIES`, the provenance contract version and baseline epoch (not touched). `blend.py` lost its `V2Result` / `run_wfo_v2` aliases. `test_leak.py` now scans every module of `lab/strategies/`. `test_paper_boundary.py` dropped `ggTrader.lab.strategies.ensemble` from the banned legacy list (that name now is the engine ensemble). Historical prose (changelog entries, earlier completion notes, ledger, archived scripts, evidence scripts) keeps the old names; a one-line "renamed" note sits in `README.md` of this folder. |
| DSR tests in `tests/engine/test_stats.py` | `ggTrader.lab.gates.dsr_check` still exists, so they stay as they were. |
| Spec | Appendix E of `docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md` has a dated entry for P5c, P8b and P9. |

## 2. Verification

| Step | Tests (`scripts/run_tests.sh tests/`, 8 GB cap, no swap) |
|---|---|
| Before the renames (after items 1 to 4) | 2,709 passed, 19 deselected, 11 min 50 s (P5c ended at 2,602 before P8b added its tripwires and this note added the script tests) |
| After `strategies/v2` flattening | 2,709 passed |
| After `wfo_v2` rename | 2,709 passed |
| Final (after docstring and changelog edits) | 2,709 passed, 19 deselected, 12 min 40 s |

`ruff check .` and `ruff format --check .` clean (307 files); `ggt.py lab --help`, `paper --help`, `db --help` exit 0.

## 3. Unfinished or left for the owner

* `REGISTRY_V2` keeps its name (a symbol, not a module path); renaming it touches 23 references and the docs. Say if you want it.
* `scripts/mcp_server.py` is stale: it lists `STRATEGY_REGISTRY` (deleted), points its cache at the production checkout (`/home/flynn/ggTrader/scripts/mcp_cache.json`) and `mcp` is not recorded in `pyproject.toml`. It now has `--help`; whether to fix or archive it is a separate decision.
* The nine study drivers named in section 1 still write into `research/` by default (by design: they reproduce committed `_results.json` files).
* Everything from P1 to P9 is still uncommitted; the tag restores only the pre-migration code.
