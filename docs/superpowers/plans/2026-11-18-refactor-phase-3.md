# Refactor phase 3 (research engine: `run_signals` helpers, `run_wfo` records and helpers) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two largest research-engine functions readable without changing what either one
computes, raises, prints or returns. `run_signals` (`engine/portfolio.py`) gets three helpers.
`run_wfo` (`lab/wfo/run.py`) gets typed carriers and three helpers and keeps its public signature.
No research number may move by one bit.

**Architecture:** Tests come first, as in Phase 0-2. Today's behavior is recorded as goldens over
small synthetic inputs, written to run in the fast core and offline. These goldens pin what the
existing tests and the oracle drivers cannot see: the order in which errors win, the exact error
text, the 28 fold keys in order with their value types, the order of scorer calls, the verbose
output, and the opt-in branches the oracles never take (halt masks, intraday stops, policy
options, skipped folds). Each extraction is then one small commit that must reproduce those goldens
exactly (in an exact-float mode on this host), the existing tests, and the oracle results.

**Tech Stack:** Python 3.12 (uv-managed `.venv`), pytest through `scripts/run_tests.sh`, ruff,
pandas 3, numpy, vectorbt 1.2.0 with the Rust simulator (reached only through `ggTrader.engine`).

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md),
section 5.3 and the phase 3 row of section 7. The Phase 0-2 plan
([2026-11-17-refactor-phase-0-2.md](2026-11-17-refactor-phase-0-2.md)) defines the tools this plan
reuses by reference: the worktree, `offline.sh`, the oracle check, the test-outcome check, the
verify-pins check and the comparers. Task numbers continue from that plan (it ends at Task 14).

**Revision 1 (2026-11-18)** is the first draft. Every line number below was read on 2026-11-18
from the worktree `/home/flynn/ggTrader-refactor` (branch `refactor/phase-0-2`, HEAD `a270fc5a`,
plus the uncommitted Task 7 edit to `run.py` that removes the `names` parameter of
`_select_live_params`). Other builders were editing a few files in that worktree while this was
written, and Phase 0-2's Task 14 will rewrite annotations in `run.py` and `scorers.py`. Treat every
line number as approximate and re-grep the anchor string given beside it (Task 15 Step 4 records
the new numbers).

## Scope

**In this plan:** design section 5.3 only.
- `engine/portfolio.py`: `_halt_positions`, `_validate_intraday_stops` and `_reject_overlaps`,
  extracted from `run_signals` with no change to its signature or results.
- `lab/wfo/run.py` and `lab/wfo/scorers.py`: `SignalSimOptions`, `ScoringInputs`, `FoldRecord`,
  `_score_fold`, `_stitch_oos` and `_build_diagnostics`, with no change to the signature or
  result of `run_wfo`. `WfoResult.fold_results` stays a list of dicts.
- The characterization tests that make those changes provable (Tasks 16 to 19).
- An optional, owner-gated last task that replaces the string return with an exception (Task 29).

**Left out on purpose:**
- Phases 4 to 6 (paper trader, plumbing, tidy-ups).
- Merging the two overlap loops of `run_signals` into one. They differ in the column map and the
  message, and the design does not ask for it.
- The long ensemble WFO inputs (`mini_wfo.json`, the 17-fold tape run): those stay with the
  existing slow tests and the oracles.
- Deleting the `_make_scorer` wrapper. Two existing tests import it (Task 23).
- Enabling `ARG001`/`ARG002` or `PLR0913`. The design turns those on later; this plan only makes
  sure its own new code adds no unused arguments and no function of more than eight parameters.
- Any edit under `research/` and any edit to the oracle and legacy scripts (list in the Phase 0-2
  plan, "Left out on purpose").

## Owner decisions this plan needs before execution

A default is not an approval. Nothing in this plan runs until the owner says so, as in Phase 0-2.

- **O-P3-1, the string return of `run_wfo` (Task 29 only).** `run_wfo` returns a message string
  instead of a result when the evaluation window holds no fold (under 15 months). Three source
  callers and two tests rely on that. Of the frozen callers only the oracle driver and one DSR evidence
  script can still run, and neither depends on the string; the ten frozen scripts that check for
  it cannot import today.
  Task 29 would change it to an exception. *Default: skip Task 29.* The reasons and
  the exact callers are in the task.
- **O-P3-2, quota (Phases B and C).** The agent cannot see plan usage. The owner is asked for
  current headroom before Task 15 (see Execution roles). If it is unknown, run only Phase A
  (Tasks 15 to 19) and report.

## Global Constraints

> **Heavy-job rules (added 2026-11-18 after the Encom out-of-memory incident; implemented by
> [2026-11-18-test-run-safety.md](2026-11-18-test-run-safety.md)).** Only how ggTrader is tested
> and run is affected. Until branch `chore/test-run-safety` is merged into the refactor branch, these hold by discipline (once merged, the scripts enforce the first two with a lock):
> - **One heavy job at a time:** one pytest run or one oracle driver, never two. Check
>   `ps -eo args | grep -E "pytest|run_tests|p5_wfo|p7_live"` before starting one.
> - **Builders run only small targeted test paths** through `./offline.sh scripts/run_tests.sh <paths>`.
>   They never run `tests/paper/`, a whole tier or an oracle driver unless their brief names it; the
>   orchestrator runs those gates, serially. A builder that sees exit 137 or 143 stops and reports.
> - **Two builders never run tests at the same time.** Builders that edit files in parallel must not
>   run tests in parallel: stagger them, or give one the test run.
> - Use `GGT_TEST_MEM_MAX=3G` on every run (a healthy suite peaks well under 1 GB).
> Once that plan is built, the scripts enforce the first rule with a lock and refuse a run when free
> memory is below the cap plus 4G.

Everything in the Phase 0-2 plan's "Global Constraints" applies and is not repeated here:
offline by construction (`offline.sh`), tests only through `scripts/run_tests.sh`, no real
connections, no edit under `research/`, ruff clean, commits end with the attribution line.
What Phase 3 adds or changes:

- **Starting point.** Branch `refactor/phase-3` is created from the head of `refactor/phase-0-2`
  after that plan's Phase 2 gate has passed. Phase 0-2's Tasks 7 and 14 edit `run.py`, and Task 14
  rewrites annotations in both files, so starting earlier would only create conflicts. Never
  commit to `main`, push or merge without the owner's say-so.
- **Two baselines.**
  - *Oracle baseline:* `/home/flynn/ggTrader-refactor-evidence/baseline` (recorded from the untouched Phase 0-2
    `BASE`; 39 files under `p5/` plus `p7.json`). It is re-recorded only if the base commit
    changes. Phase 0-2's gate already proved its code left these results identical, so Phase 3
    is held to the original numbers, not to a new baseline.
  - *Test-outcome baseline:* recorded in Task 15 on the Phase 0-2 head as
    `p3base-core.xml` and `p3base-studies.xml`. Phase 0-2 adds tests, so the original files are
    too small a reference for this phase.
- **Public surface is frozen.** `run_wfo`'s parameter names, order, kinds and defaults;
  `run_signals`'s; the keys and key order of every fold dict and of `diagnostics`; the exceptions
  and messages; the printed output; and the call order into the scorer. Task 18 pins each of
  these, so a change shows up as a failing test, not as a reviewer's opinion.
- **Names that must stay importable.** Two different things check this, and neither is a
  complete inventory of frozen research:
  - `tests/utils/test_study_surface.py` scans only `research/*/*/code/**/*.py` and
    `research/*/_lib/**/*.py` (`tests/study_surface.py:24-25`) and checks imported *names*, not
    `module.attr` uses. Of the engine and WFO names, it sees `run_signals` (from
    `research/intraday/_lib/tests/test_halt_intervals.py`), the `ggTrader.lab.wfo.breaker` names
    (`WfoPolicy`, `WfoState`, `check_circuit_breaker`, `check_shadow_reentry`, `compute_wfe`) and
    `ggTrader.lab.wfo.report.curve_summary`. It does not scan `research/programme/vectorbt-engine/`
    (the engine evidence, the profiling script and the legacy scripts), which is where most
    `run_wfo` and engine consumers are.
  - Task 18's new `tests/lab/wfo/test_frozen_evidence_seams.py` covers the frozen evidence that
    still imports today, without requiring the broken ones to resolve. Measured 2026-11-18 (every
    `ggTrader` import of each file tried): the files whose imports **all resolve** are
    `p10-evidence/dsr_one_combo.py` (uses `run.run_wfo`, patches `run.run_dsr`, reads
    `gates.run_dsr` and `gates.MIN_DSR_BARS`, and the fold keys `gates_passed` and `dsr_value`), `p8-evidence/probes/golden_dump.py`
    (`engine.config.set_engine`), and the two D-examples `d1_wfo_splitter.py` and
    `d5_full_history_wfo.py` (`generate_folds`). The test asserts exactly those names and
    attributes, and that replacing `run.run_dsr` with a recorder is honored by `run_wfo`.
    The files that **already fail to import today**, and are therefore not asserted (they stay
    broken whatever this plan does): `p2b-evidence/window_convention_check.py`
    (`ggTrader.lab.metrics`), `p4-evidence/chunker_memory.py` and `chunker_experiment.py`
    (`ggTrader.lab.strategies.v2.*`; note they also use `portfolio._run_chunks`,
    `portfolio.estimate_peak_bytes` and `portfolio.run_signals`, none of which this plan touches),
    `legacy-scripts/engine_parity_evidence_p2_full.py` and `p4_strategy_parity.py`,
    `D-examples/d4_prefix_causality.py` (`ggTrader.lab.metrics`), `profiling/prof_wfo.py`
    (`lab.strategies.STRATEGY_REGISTRY`, `lab.sweep`), `S3-evidence/e2e_wfo.py`, `e2e_sweep.py`,
    `sweep_timing.py`, `p1-parity/p1_parity_driver.py`, `F-checks/f_real_checks.py`, and all 18
    `legacy-scripts/*` files that call `run_wfo` (each imports at least one removed module such
    as `ggTrader.lab.sweep` or `EnsembleSignal`).
  The names themselves:
  - from `ggTrader.engine.portfolio`: `run_signals` (imported by
    `research/intraday/_lib/tests/test_halt_intervals.py`), `run_weights`, `SimResult`,
    `DEFAULT_MAX_BYTES`, `TIME_STOP_WARNING`, `estimate_peak_bytes`, `_plan_chunks`
    (`tests/lab/wfo/test_leak_golden_memory.py`), `_validate_stop_bars`;
  - from `ggTrader.lab.wfo`: `run_wfo`, `WfoResult`, `WfoPolicy`, `WfoState`, `Fold`,
    `generate_folds`, `check_circuit_breaker`, `check_shadow_reentry`, `compute_wfe`,
    `result_payload`, `format_wfo_table`, `combo_name`, `composite_score`, `pick_anchor`,
    `AnchorSet`; and `ggTrader.lab.wfo.report.curve_summary`;
  - from `ggTrader.lab.wfo.run`: `_make_scorer` (two tests import it), `_halt_mask_hash`,
    `_select_live_params`, `CFG_DEFAULT_KEYS`, and the *module-level names* `run_dsr`, `run_ndh`,
    `SignalScorer`, `WeightScorer`, `composite_score`, `pick_anchor`. The last group is not an
    import list. `tests/lab/wfo/test_p5b_review_fixes.py:340` and the frozen study
    `research/programme/vectorbt-engine/p10-evidence/dsr_one_combo.py:67` replace `run.run_dsr`
    on the module, and the Task 18 pins replace `run.SignalScorer`. That only works if every new
    helper stays in `run.py` and calls these names through the module namespace at call time,
    never through a default argument, a dict built at import time or another module.
  - `lab/wfo/__init__.py` and `run.__all__` are not widened: the new classes are internal.
- **Pin goldens live outside `tests/engine/golden/`.** `tests/engine/test_golden.py::
  test_goldens_are_changelogged` globs `tests/engine/golden/*.json` and requires a changelog entry
  for each. The new goldens go to `tests/engine/characterization/` and
  `tests/lab/wfo/characterization/`, and are regenerated with an environment variable (same
  pattern as the transcript goldens), not with `--update-golden`.
- **Floats in goldens.** A golden stores floats at full precision (`repr`). The default
  comparison uses `rtol=1e-9`, the same tolerance as the existing engine goldens, so CI on
  another CPU does not fail on a last-bit difference. `GGT_CHARACTERIZATION_EXACT=1` compares
  every float bit for bit. Every refactor task runs the exact mode on this host, because the
  refactor runs on the same machine, library versions and engine as the golden.
- **Heavy runs one at a time.** Two builders may work in two worktrees at once, but the oracle
  check and the `--full` gate are run by the orchestrator one after the other. Two 6 GB cgroup
  runs together are how the 2026-09-16 host wedge started.
- **Test time budget.** `slow` means a single test of 3 s or more and leaves the fast core. Each
  new pin test must stay under 3 s (`--durations=10` shows it), or be marked `slow` and listed in
  the task's hand-back note. The helper-level tests of Tasks 20 to 27 must stay in the fast core.

**The pin check** (new; used by Tasks 19 to 28):
```
./offline.sh env GGT_CHARACTERIZATION_EXACT=1 scripts/run_tests.sh \
  tests/test_characterization.py \
  tests/engine/test_run_signals_characterization.py \
  tests/lab/wfo/test_run_wfo_characterization.py
./offline.sh env GGT_CHARACTERIZATION_EXACT=1 PYTHONHASHSEED=1 scripts/run_tests.sh <the same three files>
./offline.sh env GGT_CHARACTERIZATION_EXACT=1 PYTHONHASHSEED=2 scripts/run_tests.sh <the same three files>
```
All three must pass with no golden regenerated. (`offline.sh` scrubs the environment, so the
variables are set after it with `env`.)

**The neighbourhood run** (used by every code task):
`./offline.sh scripts/run_tests.sh tests/engine tests/lab tests/utils`. Path arguments run every
test in those folders, including the `slow` and `numba` ones, and it includes the study-surface
and import-direction tests.

**The task gate** (every code task, Tasks 20 to 27): ruff check and format check clean
(`.venv/bin/ruff check . && .venv/bin/ruff format --check .`), the pin check, the neighbourhood
run, then the oracle check of the Phase 0-2 plan with the task's label, held against
`/home/flynn/ggTrader-refactor-evidence/baseline`. What the oracle covers: both drivers run the default paths of
`run_signals` and `run_wfo` on the real tape, for signal and weight strategies, and the p5 result
JSON carries every fold row and the diagnostics except the timing fields. What it cannot see: key
order (the comparer reads parsed JSON), the opt-in paths, verbose output, error precedence,
warnings. Those are the pins' job.

**The phase gate:** the Phase 0-2 test-outcome check against `p3base-*.xml` with the Task 16 to
18 and 20 to 27 tests named in `--must-pass`, the oracle check (label `phase3`), the verify-pins
check with the counts in `.github/workflows/ci.yml` as Phase 0-2 left them, and both reviews.

## Deviations from the design

Each was checked against the code. None changes the design's intent.

1. **There are 28 fold keys, not 27.** Design sections 2 and 5.3 say 27. The dict built at
   `run.py:383-416` has 28 (counted by script and by hand): `fold_num`, `train_start`,
   `train_end`, `test_start`, `test_end`, `winner_combo`, `winner_params`, `train_score`,
   `oos_score`, `ndh_passed`, `dsr_passed`, `gates_passed`, `ndh_density`, `dsr_value`, `wfe`,
   `is_sharpe`, `oos_sharpe`, `halted`, `halted_entering`, `used_anchor`, `anchor_combo`,
   `anchor_params`, `anchor_reason`, `deployed_combo`, `halt_cleared`,
   `winner_trades_in_window`, `winner_trades_whole_simulation`, `od12_ndh_flip`. `FoldRecord` has
   28 fields, and Task 18 hard-codes the list so a regenerated golden cannot drop one.
2. **The scorer constructors keep their signatures.** The design says `ScoringInputs` is one
   object "both scorers take". About 20 call sites in `tests/lab/wfo/` build `SignalScorer(...)`
   or `WeightScorer(...)` with nine positional arguments (`test_toy_known_answers.py`,
   `test_p5b_review_fixes.py`, `test_causality.py`, `test_report_inputs_weights.py`), and two
   tests call `_make_scorer` positionally. Rewriting them would put the pins' own tests into the
   change. Both scorers therefore gain a classmethod `from_inputs(inputs, ...)` and keep
   `__init__` unchanged. If the owner would rather have the constructors take the object, it is
   a mechanical follow-up of about 22 test edits.
3. **Three small private carriers are added.** The design names three helpers. `_score_fold`
   needs a context (the per-run constants) and an outcome (what the loop needs back), and
   `_build_diagnostics` would otherwise take 13 parameters. `_FoldContext`, `_FoldOutcome` and
   `_Tally` are private to `run.py` and keep every function at eight parameters or fewer, which
   the design's later `PLR0913` rule wants.
4. **`_stitch_oos` is the final stitching only.** The per-fold part of the stitching (turning a
   test window into a curve scaled to the running value, `run.py:344-361`) uses state that changes
   fold by fold. It moves into `_score_fold`, and `_stitch_oos` takes the final concat and the
   SPY comparison (`run.py:427-444`).
5. **`_validate_intraday_stops` must return a value.** The design lists it as a check. The block
   at `portfolio.py:211-239` also assigns `stop_columns`, which the `simulate` closure reads, so
   the helper returns it. The `gaps` dictionary, which `simulate` fills per chunk, stays in
   `run_signals`.
6. **The existing "engine goldens" are real but incomplete.** The design's roadmap row says
   "engine goldens; oracles identical". `tests/engine/test_golden.py:38-45` compares equity at
   `rtol=1e-9` and also trade counts and entry and exit indices, on a trailing-stop panel, a
   masked 7-bar-time-stop plus take-profit panel and a monthly-weights panel, plus a 2-fold
   mini WFO (winners, test trades, Sharpes, drawdown); `mini_wfo.json` pins 7 of the 28 fold
   keys. What none of them reaches: halt masks and intraday-stop inputs, the order in which
   errors win, exact message text, warning attribution, scorer call order, key order, value
   types, skipped folds and the `policy` options. Tasks 16 to 19 add those, not what the
   existing goldens already cover.

## Review Focus

1. **A pin that cannot fail.** Check the mutation step of Task 19: the pins must fail when error
   precedence, a message, the warning's attribution, a fold key's position, the `halted_entering`
   read, or the `anchor` assignment of a skipped fold is changed.
2. **Moved code that is not verbatim.** For each extraction the diff must show the block as moved
   (`git diff --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`), and the
   reviewer lists every changed line that is not moved.
3. **Order of operations.** `run_signals` raises in a fixed order, and `run_wfo` reads and updates
   state in a fixed order (the numbered lists in Tasks 20 and 25). The risk table below names
   each ordering that is easy to disturb.
4. **A moved warning.** `warnings.warn(..., stacklevel=2)` at `portfolio.py:241` must stay in
   `run_signals`.
5. **Aliasing and types in `FoldRecord`.** `as_dict()` must return the same objects (no deep
   copy) and the same numpy or builtin types.
6. **Module-level seams.** `run.run_dsr`, `run.run_ndh` and `run.SignalScorer` are still looked up
   by name at call time.
7. **Whether the golden shows what its case claims.** An Opus reader goes through each case once.

## Risks found in the code, and the check that catches each

| # | Risk | Where | Check |
|---|---|---|---|
| 1 | Which error wins changes when validations move. Today: `_reject_unsupported`, `order_sequence`, `stop_exit_price`, `stop_entry_price`, shapes, halt mask, intraday config, then the time-stop's own `ValueError`, then `_stop_arguments` | `portfolio.py:171-251` | Task 17 cases `precedence_*` (two invalid inputs at once, each pair) |
| 2 | The time-stop warning is attributed to the caller through `stacklevel=2`. Moving it into a helper attributes it to `portfolio.py` | `portfolio.py:241-245` | Task 17 `test_td_stop_warning_is_attributed_to_the_caller` (the warning's file is the test file) |
| 3 | `stop_columns` is assigned in the intraday block and read inside the `simulate` closure; `gaps` is filled inside that closure once per chunk and read after the run | `portfolio.py:210-211,269-278,310-321` | Chunked cases (`max_bytes=1`) for compact stop prices and for a gap in the second chunk; ruff `F821` and `F841` |
| 4 | Halt overlap is checked before the missing-bar overlap, and the first overlapping trade in `result.trades` order (by `col`, then `id`) is the one reported | `portfolio.py:298-321` | Task 17 `halt_and_gap_both_overlap`, `two_halted_positions_first_by_column` |
| 5 | An input frame is changed in place by a helper (`entries`, `exits`, `close`, `mask`, `halt_mask`, stop frames) | helpers receive caller frames | Task 17 `test_inputs_are_not_mutated` |
| 6 | `dataclasses.asdict` deep-copies nested dicts and Timestamps; wrapping a value in `bool()`, `float()` or `int()` changes numpy types | `FoldRecord.as_dict` | Task 24 `test_as_dict_returns_the_same_objects`; the Task 18 golden tags every non-builtin type (`numpy.float64`, `pandas.Timestamp`) |
| 7 | Key order of the fold dict (28) and of `diagnostics` (26): the oracle comparer reads parsed JSON and cannot see a reorder | `run.py:383-416,468-516` | Task 18 `test_fold_keys_are_the_28_in_order`, `test_diagnostics_keys_are_the_26_in_order` (hard-coded in the test, not read from the golden) |
| 8 | The state's `wfe_history` and `oos_sharpes` lists are appended to in place, then the breaker returns a copy. A helper that appends to a copy and passes the original on changes the breaker's input | `run.py:368-381` | Task 18 `scripted_halt_and_reentry` plus Task 25 `test_score_fold_appends_to_the_state_it_was_given` |
| 9 | `anchor` is assigned at the top of every fold, including a skipped one, and the last value goes to the table | `run.py:298,310-313,455-457` | Task 18 `scripted_skipped_fold` with the skipped fold last |
| 10 | The verbose header line is printed with `end=""` before the skip check, so a skipped fold leaves an unterminated line | `run.py:299-313` | Task 18 `toy_verbose` and `scripted_skipped_fold` (exact stdout) |
| 11 | Module-level lookup of `run_dsr`, `run_ndh`, `SignalScorer`, `WeightScorer` (see Global Constraints) | all of `run.py` | existing `test_a_default_run_hands_the_default_threshold_to_the_gate`; Task 18 `test_scorer_seam_is_live` and `test_gate_seam_is_live` |
| 12 | Float operation order: `oos_running_value * (eq_test / eq_test.iloc[0])`, `start_cash * (spy_oos / spy_oos.iloc[0])`. Writing `a * b / c` differently changes the last bit | `run.py:353,438` | exact-float pins and the byte comparison of the 13 `*_oos_equity.csv` files |
| 13 | `n_folds` counts skipped folds, so `legacy_equivalent_simulations` and `n_folds` in the diagnostics include them | `run.py:462,500` | `scripted_skipped_fold` golden |
| 14 | `diagnostics` reads `base_config.get("stop_entry_price")` from the caller's dict, not the normalized copy. They are equal (normalization only changes `FREQ`), but a helper that takes the wrong one by mistake would also pass today's tests | `run.py:485` | `toy_intraday_halt_mask` runs with `stop_entry_price` set |
| 15 | `order_sequence or ("auto" if intraday_stops else "default") if kind == "signals" else "auto"` parses as `(a or b) if c else d`. Tidying it changes weight runs | `run.py:475-479` | copy verbatim; `weights_xs_momentum` golden |
| 16 | `strategy.defaults` is assigned on the instance a factory returns, which may be shared | `run.py:244-246` | not moved by this plan; a reviewer confirms the block is untouched |
| 17 | New tests slower than 3 s fall out of the fast core, and the pin silently stops running in CI | all new tests | `--durations=10` in each pin task; Task 19 checks the fast core collects every pin |
| 18 | Golden floats differ by a last bit on another CPU (numpy SIMD) | CI only | default `rtol=1e-9`; exact mode only on this host |

## Order of work and what can run in parallel

```
15 setup ─ 16 helper ─┬─ 17 pin run_signals ─┐
                      └─ 18 pin run_wfo ─────┴─ 19 pin gate ─┬─ 20 ─ 21 ─ 22 ──────────────┐
                                                             └─ 23 ─ 24 ─ 25 ─ 26 ─ 27 ────┴─ 28 ─ (29) ─ gate
```
- **Parallel:** Tasks 17 and 18 (different test files and golden folders, both need Task 16);
  and the engine chain (Tasks 20 to 22, only `engine/portfolio.py` and
  `tests/engine/test_run_signals_helpers.py`) with the WFO chain (Tasks 23 to 27, only
  `lab/wfo/scorers.py`, `lab/wfo/run.py`, `lab/wfo/fold_record.py` and two new test files). The
  chains share no file, and the two packages do not import each other's new names. Each chain
  gets its own worktree (Task 19 Step 5).
- **Serial:** everything inside a chain (the same function or file), Task 16 before 17 and 18,
  Task 19 before either chain, Task 28 after both, Task 29 last.
- Oracle and `--full` runs are never concurrent (Global Constraints).

---

### Task 15: Phase 3 branch, baselines and line anchors

**Files:** none changed. Creates a branch and records measurements.

- [ ] **Step 1: Preconditions.** The owner has approved execution. Phase 0-2's Phase 2 gate has
  passed and is recorded. The owner has been asked for quota headroom (Execution roles).
- [ ] **Step 2: Branch.**
  ```
  cd /home/flynn/ggTrader-refactor
  git status --short            # must be empty
  git switch -c refactor/phase-3
  git rev-parse HEAD            # record as BASE3 in the hand-back notes
  .venv/bin/python -c "import ggTrader; print(ggTrader.__file__)"   # under /home/flynn/ggTrader-refactor/src/
  ```
- [ ] **Step 3: Check the oracle baseline still stands.**
  `ls /home/flynn/ggTrader-refactor-evidence/baseline/p5 | wc -l` prints 39 and `ls /home/flynn/ggTrader-refactor-evidence/baseline`
  shows `p5` and `p7.json`. If the base commit changed since it was recorded, re-record it
  from the new base before anything else (Phase 0-2 Task 0 Step 8).
- [ ] **Step 4: Re-measure every anchor** and write the new numbers into the hand-back notes:
  ```
  grep -n "^def run_signals\|halt_positions = {}\|entries = _as_bool(entries)\|gaps: dict\|stop_columns = None\|warnings.warn(\|raw_close = \|result = _run_chunks\|if halt_positions:\|    if gaps:\|    return result" src/ggTrader/engine/portfolio.py
  grep -n "^def \|^class \|started = time.time()\|for i, fold in enumerate\|fold_results.append\|nan_metrics = \|live = _select_live_params\|diagnostics = {\|return WfoResult" src/ggTrader/lab/wfo/run.py
  grep -n "^class \|def __init__\|^def check_combination_order" src/ggTrader/lab/wfo/scorers.py
  ```
  Expected, as of 2026-11-18: `run_signals` at `portfolio.py:122-322`; `run_wfo` at
  `run.py:169-524`; `_make_scorer` at `:134-166`; the fold dict at `:383-416`; the diagnostics
  dict at `:468-516`; `SignalScorer.__init__` at `scorers.py:181`; `WeightScorer.__init__` at
  `:343`. A different structure (not just shifted numbers) is a finding to report before any
  task starts.
- [ ] **Step 5: Test-outcome baseline** on the Phase 0-2 head:
  ```
  ./offline.sh scripts/run_tests.sh --full -rA --junitxml=/home/flynn/ggTrader-refactor-evidence/p3base-core.xml tests
  ./offline.sh scripts/run_tests.sh --studies -rA --junitxml=/home/flynn/ggTrader-refactor-evidence/p3base-studies.xml
  ```
  Both exit 0. Check that both XML files exist and that the study file holds at least 777 nodes
  (`tests/tiers.py` `STUDIES_MIN_COLLECTED`).
- [ ] **Step 6: Oracle check of the starting point** (label `p3base`) against
  `/home/flynn/ggTrader-refactor-evidence/baseline`. It must exit 0. If it does not, Phase 0-2 changed a result;
  stop and report.
- [ ] **Step 7:** No commit (nothing changed).

### Task 16: Characterization helper (`tests/characterization.py`)

**Files:**
- Create: `tests/characterization.py`
- Test: `tests/test_characterization.py`

A plain module like `tests/netguard.py` and `tests/study_surface.py`, imported as
`from tests.characterization import ...`. It lets a test record a value tree as canonical JSON
that keeps key order and value types, and compare it to a stored file.

**Interfaces:**
- `RTOL = 1e-9` (the tolerance of `tests/engine/test_golden.py`).
- `encode(value: object) -> object`, a JSON-able tree. Rules (any other type raises `TypeError`
  naming the type, so no object address can reach a golden):
  - `None`, `bool`, `int` and `str` of exactly the builtin type: as they are;
  - builtin `float`: a JSON number when finite, otherwise `{"$float": "NaN"}`,
    `{"$float": "Infinity"}` or `{"$float": "-Infinity"}`;
  - `numpy.generic` (for example `numpy.float64`, `numpy.int64`, `numpy.bool_`):
    `{"$type": "numpy.float64", "v": <encode(value.item())>}`;
  - `pandas.Timestamp`: `{"$type": "pandas.Timestamp", "v": "<isoformat with tz>"}`; `NaT`:
    `{"$type": "pandas.NaT"}`;
  - `dict` of exactly the builtin type: `{"$dict": [[<encode key>, <encode value>], ...]}` in
    insertion order. This is what makes a key reorder visible;
  - `list`: a list; `tuple`: `{"$tuple": [...]}`; `set` and `frozenset`:
    `{"$set": [...]}` sorted by their canonical text;
  - `pandas.Series`: `{"$series": {"name": ..., "dtype": "float64", "index": [...], "values": [...]}}`;
  - `pandas.DataFrame`: `{"$frame": {"columns": [...], "dtypes": [...], "index": [...], "rows": [[...]]}}`
    (labels encoded with the same rules, so tuples and Timestamps are allowed);
  - `numpy.ndarray`: `{"$array": {"dtype": ..., "values": [...]}}`.
- `dumps(encoded: object) -> str`: `json.dumps(encoded, indent=1, ensure_ascii=False) + "\n"`.
- `compare(actual: object, golden: object, *, rtol: float, exact: bool) -> list[str]`: walks two
  encoded trees and returns the paths that differ (`cases.toy_default.fold_results[1].$dict[5]`
  style). Structure, strings, ints, bools and type tags are compared exactly. Floats: exactly when
  `exact`, else `math.isclose(rel_tol=rtol, abs_tol=0.0)`. An int against a float is a difference.
- `check_golden(path: Path, actual: object) -> None`: encodes, then
  - with `GGT_REGEN_CHARACTERIZATION=1`: writes the file, then `pytest.fail("regenerated <path>;
    review the diff and rerun")`;
  - with the file missing: `pytest.fail` naming the regeneration command;
  - otherwise compares (exact when `GGT_CHARACTERIZATION_EXACT=1`) and, on a difference, fails
    with the differing paths and the first 60 lines of a unified diff of the two canonical texts.

- [ ] **Step 1: Write the failing tests** (`tests/test_characterization.py`):
  - `test_encode_preserves_dict_order`: two dicts with the same items in a different order encode
    differently.
  - `test_encode_tags_numpy_and_pandas_types`: `np.float64(1.5)` and `1.5` encode differently;
    `np.bool_(True)` and `True` likewise; a tz-aware Timestamp keeps its offset.
  - `test_encode_tags_non_finite_floats`: `nan`, `inf`, `-inf`, and a `np.float64("nan")` inside
    the tag.
  - `test_encode_rejects_unknown_types`: an arbitrary object and an `OrderedDict` raise
    `TypeError`.
  - `test_compare_is_exact_on_structure_and_tolerant_on_floats`: a one-ulp float passes at
    `exact=False` and fails at `exact=True`; an int against a float fails in both.
  - `test_compare_reports_a_path`: `{"a": [1, 2]}` against `{"a": [1, 3]}` names `a[1]`.
  - `test_check_golden_regenerate_then_pass`: in `tmp_path`, with the regeneration variable set the
    call writes the file and fails; without it the same value passes; a changed value fails with
    the path in the message.
  - `test_frame_and_series_round_trip_dtypes`: `float64` and `int64` columns keep their dtype
    names.
- [ ] **Step 2:** `./offline.sh scripts/run_tests.sh tests/test_characterization.py`. Expected:
  FAIL (module missing).
- [ ] **Step 3:** Implement `tests/characterization.py` with module and function docstrings.
- [ ] **Step 4:** Run the same command. Expected: PASS, under 3 s per test.
- [ ] **Step 5:** `.venv/bin/ruff check tests/characterization.py tests/test_characterization.py`
  and `ruff format --check` on both.
- [ ] **Step 6:** Commit: `test: characterization helper that records value trees with key order and types`.

### Task 17: Pin `run_signals` (and `_validate_stop_bars` as seen through it)

**Files:**
- Create: `tests/engine/test_run_signals_characterization.py`
- Create: `tests/engine/characterization/run_signals.json`

**What already pins `run_signals`, so it is not repeated here** (all in `tests/engine/`):

| Behavior | Existing test |
|---|---|
| default-path equity bit-identical to the legacy simulator; returns; conviction sizes; ATR stop | `test_portfolio.py:70-125` (5 extras, `GOLDEN`) |
| chunking changes no number; one group over budget warns; chunk release | `test_portfolio.py:194-258` |
| legacy config keys and shape mismatches raise | `test_portfolio.py:275-286,388-410` |
| mask ANDed before the time stop | `test_portfolio.py:296-358` |
| halt: refusal on an open position, entry-bar and last-bar inclusion, unknown state, compact mask in a later chunk, missing symbol, clear mask equals no mask | `test_intraday_stop_inputs.py:70-150` |
| intraday stops, OHLC validity, missing bars, compact stop prices, `stop_entry_price`, order sequences | `test_intraday_stop_inputs.py:153-905` (61 call sites; messages matched by regex) |
| time-stop warning present or absent; invalid limits | `test_time_stop_entry_reuse.py:11-33` |
| golden panels on both engines | `test_golden.py`, `golden/engine_scenarios.json` |

**What is not pinned today, and this task adds:** the exact message text (the existing tests use
regexes); which error wins when two inputs are invalid at once; the warning's attribution;
whether inputs are left unchanged; the full result tables and dtypes for the halt and intraday
paths; which trade is reported when several overlap; halt against missing-bar precedence.

**Interfaces:** the module defines `CASES: dict[str, Callable[[], dict]]` in definition order.
Each case builds tiny inputs (at most 8 business days and 3 columns, closed-form prices, no
randomness) and returns `_record(call)`:
`{"outcome": "result", "result": {"engine", "n_chunks", "value", "returns", "trades", "n_trades"},
"warnings": [...]}` or `{"outcome": "raises", "error": {"type": <class name>, "message": <exact
text>}, "warnings": [...]}`. Warnings are recorded as `{"category", "message", "from_caller"}`
under `warnings.catch_warnings(record=True)` with `simplefilter("always")`; `from_caller` is true
when the warning's file is the test module. A fixture sets the Rust engine and restores it
(`tests/engine/conftest.py` already restores it); no `numba` case, so every case stays in the fast
core.

Cases, grouped (each name is a key in the golden):
- **Config order:** `unsupported_key`, `bad_order_sequence`, `bad_stop_exit_price`,
  `bad_stop_entry_price`, `shape_mismatch_entries`.
- **Precedence (risk 1).** Every case makes BOTH of its inputs invalid, so the message in the
  golden shows which check ran first. The set must cover every adjacent pair of checks in the
  order list of Task 20 (steps 1 to 14) and every adjacent pair inside the intraday block of
  Task 21 (checks a to h) and inside `_halt_positions`; the table after this list maps each
  frozen ordering to its case(s), and `test_every_frozen_ordering_has_a_case` fails if a row
  names a case that is not in `CASES`. Cases:
  - *Config order (steps 1 to 5):* `precedence_unsupported_key_over_order_sequence`,
    `precedence_order_sequence_over_stop_exit_price`,
    `precedence_stop_exit_price_over_stop_entry_price`,
    `precedence_stop_entry_price_over_shapes`,
    `precedence_entries_shape_over_exits_shape` (both frames mismatched: the message names
    `entries`).
  - *Shapes before halt (steps 5, 6):* `precedence_shapes_over_halt_mask`.
  - *Inside `_halt_positions`:* `precedence_halt_bad_frame_over_missing_symbol` (a NaN state and a
    missing symbol on a MultiIndex `close`: the first check wins),
    the missing-symbol and flat-columns errors are not a pair (they need different `close`
    shapes), so each keeps its own single case in the halt-mask group below.
  - *Halt before intraday (steps 6, 9):* `precedence_halt_mask_over_intraday_config`.
  - *Inside the intraday block (checks a to h, one case per adjacent pair):*
    `precedence_td_stop_over_atr_without_trailing` (`td_stop` set and `atr_mult` without
    `ts_stop`: the `td_stop` message), `precedence_atr_without_trailing_over_no_price_stop`
    (`atr_mult` set, no `ts_stop`, no `tp_stop`: the ATR message),
    `precedence_trailing_over_target_both_invalid` (`ts_stop=nan` and `tp_stop=0`: the message
    names `ts_stop`; reversing the loop would name `tp_stop`), and the same pair with
    `ts_stop="x"` and `tp_stop=inf`,
    `precedence_invalid_target_over_missing_high_low` (`tp_stop=0` with `high=None`),
    `precedence_missing_high_low_over_stop_columns_mismatch` (`high=None` and a mismatched
    `stop_open` column set on a flat `close`).
  - *Intraday before time stop and arguments (steps 9, 10, 11):*
    `precedence_intraday_config_over_time_stop_limit` (`stop_open` given and `td_stop=1.9`, an
    invalid limit: the intraday message `td_stop is unreliable ...` wins over the limit error),
    `precedence_time_stop_limit_over_atr_arguments` (`td_stop=-1` with `atr_mult=2.0` and
    `atr_period=0`, no `stop_open`),
    `precedence_atr_shapes_over_size_shape` (ATR `high` of the wrong shape and a `size` frame of
    the wrong shape: the ATR message).
  - *Per chunk before overlaps (steps 12 to 14):* `precedence_stop_bars_invalid_price_over_entry_on_missing_bar`
    and `precedence_stop_bars_entry_on_missing_bar_over_invalid_ohlc` (the order inside
    `_validate_stop_bars`, which the extractions do not move but whose reach they could change),
    `precedence_intraday_chunk_check_over_halt_overlap` (an invalid OHLC value in chunk 1 and a
    halted position: the OHLC error wins), `precedence_halt_overlap_in_chunk_two_over_gap_in_chunk_one`
    (the halt message wins although the gap is found first).

  | Frozen ordering (Task 20 step or Task 21 check) | Case |
  |---|---|
  | 1 over 2 over 3 over 4 over 5 | the first five `precedence_*` of the config group |
  | 5 over 6 | `precedence_shapes_over_halt_mask` |
  | inside 6 | `precedence_halt_bad_frame_over_missing_symbol` |
  | 6 over 9 | `precedence_halt_mask_over_intraday_config` |
  | b over c; c over d | `precedence_td_stop_over_atr_without_trailing`; `precedence_atr_without_trailing_over_no_price_stop` |
  | e: `ts_stop` over `tp_stop` | `precedence_trailing_over_target_both_invalid` (both variants) |
  | e over f; f over g | `precedence_invalid_target_over_missing_high_low`; `precedence_missing_high_low_over_stop_columns_mismatch` |
  | 9 over 10 | `precedence_intraday_config_over_time_stop_limit` |
  | 10 over 11 | `precedence_time_stop_limit_over_atr_arguments` |
  | 11 (ATR shapes) over size shape | `precedence_atr_shapes_over_size_shape` |
  | 12 over 13 | `precedence_intraday_chunk_check_over_halt_overlap` |
  | 13 over 14 | `precedence_halt_overlap_in_chunk_two_over_gap_in_chunk_one`, `halt_and_gap_both_overlap` |

  The author checks each case in a scratch run before the golden is written: it must raise the
  message the table claims, and swapping the two inputs' validity (one valid, one invalid) must
  give the other message, so a case that always raises the same text is a finding. Ordering
  `g over h` (stop columns versus per-frame index) raises the same message in both branches, so
  no case can tell them apart; the plan records that as a known blind spot and does not claim
  to freeze it.
- **Halt mask:** `halt_mask_not_a_frame`, `halt_mask_index_mismatch`, `halt_mask_duplicate_columns`,
  `halt_mask_nan_state`, `halt_mask_int_dtype`, `halt_mask_grid_aligned_no_overlap` (result),
  `halt_mask_compact_multiindex_no_overlap` (result), `halt_mask_missing_symbol`,
  `halt_mask_columns_differ_flat_close`, `halt_overlap_on_entry_bar`, `halt_overlap_on_exit_bar`,
  `halt_overlap_in_second_chunk` (`max_bytes=1`, MultiIndex, two groups),
  `two_halted_positions_first_by_column`, `halt_flagged_but_flat` (result), `all_false_halt_mask_equals_none`
  (record both results in one case and assert equality inside the case builder).
- **Intraday stops:** `intraday_with_time_stop`, `intraday_atr_without_trailing_stop`,
  `intraday_without_any_price_stop`, `intraday_invalid_trailing_fraction` (parametrized values
  `nan`, `0`, `1.0`, `"x"` as separate cases), `intraday_invalid_target_fraction` (`0`, `-1`,
  `inf`), `intraday_missing_high_low`, `intraday_stop_columns_mismatch_flat_close`,
  `intraday_frame_index_mismatch`, `intraday_compact_prices_multiindex` (result),
  `intraday_compact_prices_missing_symbol`, `intraday_gap_in_second_chunk`, `intraday_entry_on_missing_bar`,
  `intraday_invalid_ohlc`, `intraday_trailing_stop_result` (result, a stop fires),
  `intraday_take_profit_only_result` (result), `intraday_chunked_equals_unchunked` (both results
  in one case, equality asserted).
- **Halt and gap together:** `halt_and_gap_both_overlap` (risk 4: the halt message wins),
  `gap_only_overlap`.
- **Time stop and defaults:** `time_stop_result` (result, warns), `plain_result` (no warning).

Separate tests in the module (not golden-driven):
- `test_run_signals_case_matches_golden[<name>]`: `check_golden`-style comparison per case against
  `tests/engine/characterization/run_signals.json` (the file holds `{"cases": {name: record}}`).
- `test_every_case_has_a_golden_entry` and `test_no_orphan_golden_entries`.
- `test_case_shows_what_its_name_claims[<name>]`: one assertion per case that the record shows
  what the name says (`raises` with a message that names the right thing, or `result` with at
  least one trade, or the equality flag set). A regenerated golden of empty results fails here.
- `test_td_stop_warning_is_attributed_to_the_caller`: calling `run_signals(..., td_stop=2)`
  from the test yields one `UserWarning` whose `filename` is the test file (risk 2).
- `test_inputs_are_not_mutated`: for one halt run and one intraday run, deep-copy every input
  frame and mapping before the call, and assert frame equality, identical `index` and `columns`
  objects' values, and equal `base_config` afterwards (risk 5).
- `test_every_frozen_ordering_has_a_case`: the ordering table above is written as a literal in the
  test module; every case name in it is a key of `CASES` and of the golden, and each such case's
  record is `outcome == "raises"` (except the two `*_result` cases that are not in the table).
- `test_run_signals_signature_is_frozen`: `inspect.signature(run_signals)` parameter names, kinds
  and defaults equal a literal in the test (annotations are not compared; Task 14 rewrites them).
- `test_helper_names_do_not_exist_yet` is NOT written: the pins must pass both before and after
  Tasks 20 to 22.

- [ ] **Step 1:** Write the module and the cases. Read `tests/engine/test_intraday_stop_inputs.py`
  for the working input shapes (closed-form `close`, `high`, `low`; MultiIndex columns
  `("first", "A")`) and reuse its values, citing the test in a comment per case.
- [ ] **Step 2:** `./offline.sh env GGT_REGEN_CHARACTERIZATION=1 scripts/run_tests.sh tests/engine/test_run_signals_characterization.py`
  (expected to fail, writing the golden), then without the variable. Expected: PASS.
- [ ] **Step 3: Read the golden.** The author reads every case once and confirms each message is
  the exact string in `portfolio.py` (copy-compare, not regex) and each `result` case holds at
  least one trade where the name claims one. Any case that raises something the name does not
  expect is a finding, not a golden: fix the inputs.
- [ ] **Step 4:** `./offline.sh env GGT_CHARACTERIZATION_EXACT=1 scripts/run_tests.sh tests/engine/test_run_signals_characterization.py --durations=10`.
  Expected: PASS, every test under 3 s. Then the second and third process with
  `PYTHONHASHSEED=1` and `2`.
- [ ] **Step 5:** ruff check and format check on the new file. Commit:
  `test: pin run_signals error order, messages, warning attribution and halt and intraday results`.

### Task 18: Pin `run_wfo`

**Files:**
- Create: `tests/lab/wfo/test_run_wfo_characterization.py`
- Create: `tests/lab/wfo/characterization/run_wfo.json`
- Create: `tests/lab/wfo/test_frozen_evidence_seams.py` (not golden-driven; fast core)

**What already pins `run_wfo`, so it is not repeated here:**

| Behavior | Existing test |
|---|---|
| fold windows, winners, gates, anchor, breaker against a closed-form toy | `test_toy_known_answers.py:79-186` |
| OOS curve continuity; live params; table text; `diagnostics` simulation counts | `test_toy_known_answers.py:186-211` |
| no valid folds returns the string | `test_toy_known_answers.py:214-231` |
| list grid and legacy frame inputs; legacy class refused | `test_toy_known_answers.py:233-285` |
| intraday, halt-mask, order-sequence and stop-fill options reach the scorer and the diagnostics; weight strategies refuse them | `test_toy_known_answers.py:286-667` |
| every fold records its anchor and why it was used | `test_p5b_review_fixes.py:172-189` |
| thresholds, guards, no-anchor cash curve, mask required | `test_p5b_review_fixes.py:135,327-350,415-446` |
| no lookahead, memory budget, mini 3-fold golden (7 of the 28 keys; slow) | `test_causality.py`, `test_leak_golden_memory.py` |
| weight WFO end to end | `test_report_inputs_weights.py:219` |

**What is not pinned today, and this task adds:** the 28 fold keys, their order and their value
types; `skipped_folds`; the halt, shadow re-entry and OD12 values across a run; NaN and `None`
handling in fold rows; the `policy` options at run level (no test passes `policy=` to `run_wfo`);
the order and arguments of every scorer call; the verbose output; the `diagnostics` key order;
the shape of the NaN metric dicts; identity of the parameter dicts.

**Interfaces:**
- `ScriptedScorer`, a subclass of `SignalScorer` whose `__init__(*args, **kwargs)` ignores its
  arguments and whose `from_inputs(cls, *args, **kwargs)` classmethod returns `cls()`, so the
  seam works whether `run_wfo` builds a scorer by `SignalScorer(...)` today or by
  `SignalScorer.from_inputs(...)` after Task 23. It has the attributes `run_wfo` reads (`kind`,
  `n_simulations`, `n_chunks`, `timings`) and implements `prepare`, `metrics`, `returns`,
  `equity`, `window_dates` and `trade_counts` from a per-scenario script keyed by
  `(req.kind, req.fold)`. Every call is appended to `calls` as
  `(method, kind, fold, start, end, extra)` and that list goes into the golden.
  `stop_settings` is deliberately absent (`getattr(scorer, "stop_settings", None)` is a branch).
- Scenarios are patched in with `monkeypatch.setattr(run, "SignalScorer", <subclass>)`.
- `SCENARIOS: dict[str, Callable[[], dict]]`; each returns
  `{"fold_results": [...], "skipped_folds": [...], "live_params": ..., "table": str,
  "oos_equity": <series>, "diagnostics": <dict with peak_rss_mb and the values of timings_s
  replaced by their types>, "stdout": str, "calls": [...] (scripted only)}`. Everything goes
  through `encode`, so every fold row's keys, order and value types are in the golden.

Scenarios:
- **Real engine, on the closed-form toy** (`tests/lab/wfo/helpers.py`: `ToyHold`, `toy_bars`,
  `TOY_CONFIG`; windows `2020-06-01` to `2022-03-01`, 3 folds): `toy_default`,
  `toy_entries_inside_train` (delays 330, 400, 480), `toy_open_gates` (same grid with
  `ndh_threshold=0.0, dsr_threshold=0.0`, so a winner is deployed), `toy_never_trades` (delays
  1200, 1250, 1290: no anchor, flat curve), `toy_policy_exclude`
  (`WfoPolicy(nan_wfe="exclude", anchor_fold_wfe="exclude_from_breaker")` on
  `toy_entries_inside_train`), `toy_intraday_halt_mask` (`intraday_stops=True`, a `ts_stop` and an
  all-false `halt_mask`, `stop_entry_price` set in the config; model on
  `test_toy_known_answers.py:467`), `toy_verbose` (`verbose=True` under `capsys`, so the fold lines
  and the table are in `stdout`), `toy_no_valid_folds` (the string, 2021-01-01 to 2021-06-01),
  `weights_xs_momentum` (as `test_report_inputs_weights.py:219`; if it takes 3 s or more, mark its
  test `slow` and say so in the hand-back).
- **Scripted scorer** (strategy `ToyHold`, so names and combos are real; rows are built to be
  accepted by `composite_score`, `run_ndh` and `pick_anchor`, which the author reads in
  `lab/wfo/select.py`, `gates.py` and `anchor.py`):
  - `scripted_skipped_fold`: the middle fold's train metrics are empty, then the *last* fold's are
    too (two scenarios, or one with both). Shows `skipped_folds` (`fold_num`, the four dates,
    `reason`), the unterminated verbose header, `n_folds` counting skipped folds and the anchor
    of the last fold reaching the table (risks 9, 10, 13).
  - `scripted_halt_and_reentry`: needs **at least four processed folds**, so it does not use the
    standard 3-fold toy window. Fold 1 and fold 2 return a negative OOS Sharpe (the acute-failure
    rule halts after fold 2: `check_circuit_breaker` looks at the last two `oos_sharpes`). Folds 3
    and 4 each run shadow re-entry as a clean window (NDH passed, DSR passed, WFE at or above 0.5,
    positive OOS Sharpe so the breaker does not trip again), and two clean windows clear the halt
    after fold 4 (`clean_required=2` of the last 3). A fifth fold shows the state after the clear.
    The window is `eval_start="2020-06-01"`, `eval_end="2022-09-01"`, which `generate_folds`
    turns into 5 folds (cursors 2020-06-01, 2020-09-01, 2020-12-01, 2021-03-01, 2021-06-01; each
    needs `cursor + 15 months <= eval_end`). The toy bars (1,300 business days from 2019-03-01)
    cover it; the scripted scorer never reads them. Shows `halted`, `halted_entering`,
    `halt_cleared` (true only in fold 4), `anchor_reason="halted"` in folds 3 and 4 and the
    growth of the WFE history (risk 8). `test_scenario_shows_what_its_name_claims` asserts
    `len(fold_results) >= 4`.
  - `scripted_nan_and_none_values`: a winner Sharpe below the floor (`wfe` is `None`), a NaN
    in-sample Sharpe (`wfe` is NaN), a test row without `sharpe` (`oos_sharpe` NaN).
  - `scripted_no_anchor_and_empty_curves`: no anchor (`deploy_idx` is `None`) with `window_dates`
    present, then empty; and an empty test equity (`len(eq_test) == 0`).
  - `scripted_od12_flip_gate_double`: the real gate cannot flip on ordinary inputs.
    `extract_grid_arrays` (`lab/wfo/gates.py:79`) clamps the trade count to at least 1 and divides
    the window return by it, and `ndh_check` (`lab/gates.py:94`) uses only the *sign* of the
    expectancy, with the Sharpe grid (which the count does not touch) for density and variance.
    A different count therefore cannot change the verdict. So this scenario replaces `run.run_ndh`
    with a documented double (a function in the test module whose docstring says it is not the
    real gate) that returns `NdhOutcome(passed=True, ...)` when every row's `n_trades` is below
    100 and `NdhOutcome(passed=False, ...)` otherwise; the script gives per-window counts below
    100 and cumulative counts of 100 or more. Then `od12_ndh_flip` is true, the fold number is in
    `od12_ndh_flips_folds`, and the double is called twice per fold. The golden records that the
    scenario uses a double (`"gate": "double"`).
  - `scripted_od12_no_flip_real_gate`: the same differing counts (for example 0 per window and 7
    cumulative) with the real `run_ndh`: `od12_ndh_flip` is false in every fold and
    `od12_ndh_flips_folds` is empty. This pins the real behavior the double does not show.
  - `scripted_empty_test_metrics`: `test_metrics` empty while `deploy_idx` is not `None`: no curve,
    `oos_score` 0.0, `oos_sharpe` NaN.

Separate tests (not golden-driven, so a regenerated golden cannot weaken them):
- `FOLD_KEYS_28` and `DIAGNOSTICS_KEYS_26` are tuples written out in the test module (the lists in
  Deviation 1 and below). `test_fold_keys_are_the_28_in_order` checks `list(row)` of every fold
  row of `toy_default`, `toy_open_gates` and `scripted_halt_and_reentry` against it.
  `test_diagnostics_keys_are_the_26_in_order` does the same for `diagnostics`:
  `engine, definitions, strategy_kind, intraday_stops, halt_mask_supplied, halt_mask_sha256,
  order_sequence, stop_exit_price, stop_entry_price, stop_settings, time_stop_warning, n_combos,
  n_symbols, n_bars, n_folds, n_simulations, n_chunks, legacy_equivalent_simulations,
  od12_ndh_flips_folds, timings_s, peak_rss_mb, max_bytes, oos_metrics, spy_metrics, pit_masked,
  guards`.
- `test_run_wfo_signature_is_frozen`: names, kinds and defaults of all 23 parameters against a
  literal (`policy` default equals `WfoPolicy()`; `max_bytes` equals
  `engine_portfolio.DEFAULT_MAX_BYTES`; `min_losing_days` equals `engine_config.MIN_LOSING_DAYS`).
- `test_scorer_seam_is_live`: after `monkeypatch.setattr(run, "SignalScorer", ...)`,
  `run_wfo` really used it (the script was consulted). `test_gate_seam_is_live`: replace
  `run.run_dsr` and `run.run_ndh` with recorders that call the real ones; `run_dsr` is called with
  `(returns, n_trials, threshold)` positionally once per fold, `run_ndh` twice per fold (risk 11).
- `test_frozen_evidence_seams_still_resolve` (new file `tests/lab/wfo/test_frozen_evidence_seams.py`,
  see Global Constraints): the focused import and patch-seam checks for the frozen evidence that
  can still import today.
- `test_winner_and_anchor_params_are_the_scorers_own_dicts`: with the scripted scorer, a fold's
  `winner_params` and `anchor_params` are the very dict objects the script supplied (`is`).
- `test_every_scenario_has_a_golden_entry`, `test_no_orphan_golden_entries`,
  `test_scenario_shows_what_its_name_claims[<name>]` (for example `toy_open_gates` has
  `used_anchor` false in fold 1; `scripted_skipped_fold` has a non-empty `skipped_folds`;
  `scripted_halt_and_reentry` has at least one fold with `halt_cleared` true).

- [ ] **Step 1:** Write the module, the script helper and the scenarios.
- [ ] **Step 2:** Regenerate once
  (`./offline.sh env GGT_REGEN_CHARACTERIZATION=1 scripts/run_tests.sh tests/lab/wfo/test_run_wfo_characterization.py`,
  expected to fail), then rerun without the variable. Expected: PASS.
- [ ] **Step 3: Read the golden.** Confirm: each fold row lists 28 keys in the order above; types
  show `numpy.float64` or builtin `float` exactly where the code produces them today (do not
  "fix" a numpy value into a float in the golden: the golden records today's types); the volatile
  fields (`peak_rss_mb`, the values of `timings_s`) hold types, not numbers; no absolute path or
  timestamp of the run appears.
- [ ] **Step 4:** Run with `GGT_CHARACTERIZATION_EXACT=1 --durations=10` in three processes
  (`PYTHONHASHSEED` unset, 1, 2). Every test passes and none takes 3 s or more, except a
  `slow`-marked weights test, which then runs through the full tier.
- [ ] **Step 5:** ruff check and format check. Commit:
  `test: pin run_wfo fold records, scorer call order, verbose output and branch outcomes`.

### Task 19: Pin gate, mutation check and build worktrees

**Files:** none committed, apart from the two worktrees.

- [ ] **Step 1: The pin check** (Global Constraints) on the untouched code. All three processes
  pass.
- [ ] **Step 2: The fast core collects the pins.** `run_tests.sh` selects the fast tier only for
  no arguments or `--fast`; with a path argument it collects everything in the path, so the two
  collections below are different sets. Compare node ids, not counts:
  ```
  ./offline.sh scripts/run_tests.sh --collect-only -q <the pin files and test_frozen_evidence_seams.py> | grep '::' | sort > /home/flynn/ggTrader-refactor-evidence/pins-all.txt
  ./offline.sh scripts/run_tests.sh --fast --collect-only -q | grep -E 'characterization|test_frozen_evidence_seams' | grep '::' | sort > /home/flynn/ggTrader-refactor-evidence/pins-fast.txt
  diff /home/flynn/ggTrader-refactor-evidence/pins-all.txt /home/flynn/ggTrader-refactor-evidence/pins-fast.txt
  ```
  The diff must list only node ids that carry `@pytest.mark.slow` in the source (check each with
  `grep -n "mark.slow"`), and those are named in the hand-back. Any other missing node id means a
  pin left the fast core.
- [ ] **Step 3: Mutation check (not committed).** Make each change, rerun the two pin files with
  `GGT_CHARACTERIZATION_EXACT=1`, confirm at least one failure, revert:
  - `portfolio.py`: swap the order of the halt-mask validation and the `stop_entry_price` check;
  - `portfolio.py`: change one character of the `halt during an open position` message;
  - `portfolio.py`: make the halt overlap loop run after the missing-bar loop;
  - `portfolio.py`: add `stacklevel=1` to the time-stop warning;
  - `run.py`: swap the keys `halted` and `halted_entering` in the fold dict;
  - `run.py`: read `halted_entering = wfo_state.halted` after the breaker call;
  - `run.py`: move `anchor = pick_anchor(...)` below the skip check;
  - `run.py`: change `int(whole[winner_idx])` to `whole[winner_idx]` in the fold dict
    (`winner_trades_whole_simulation`). The value is equal either way, so the observable is the
    **type tag**: the golden records `builtin int`, the mutant produces `numpy.int64`
    (`whole` comes from `np.bincount`), and the pin's type tags fail while the oracle's JSON
    would not;
  - `run.py`: change `oos_running_value * (eq_test / eq_test.iloc[0])` to
    `eq_test * (oos_running_value / eq_test.iloc[0])`. Observable only if the two orders round
    differently on a pinned scenario, in exact mode, in the OOS equity of fold 2 or later;
  - `run.py`: change `end=""` in the fold header to `end="\n"`.

  **Rule for a mutant that survives.** Do not force it. Run the mutant through every pin and the
  oracle check; if all outputs are identical, record the evidence in the hand-back (what the
  mutation changes inside the function and why no public output differs), drop it from this
  list, and say which test could in principle see it. The mutation `float(normalized.iloc[-1])`
  to `normalized.iloc[-1]` was removed from this list for that reason: it changes an internal
  accumulator from a Python float to a `numpy.float64`, which is a `float` subclass with equal
  arithmetic, and no returned value stores it.

  Afterwards `git status` shows `portfolio.py` and `run.py` clean.
- [ ] **Step 4: Opus read of both goldens** (a fresh context with only the two golden files, the
  two test modules and the claim "each case shows what its name says"). It reports any case that
  would pass for a wrong reason.
- [ ] **Step 5: Two build worktrees** from the pin commit (merge both back in Task 28):
  ```
  git -C /home/flynn/ggTrader-refactor worktree add /home/flynn/ggTrader-refactor-p3-engine -b refactor/phase-3-engine refactor/phase-3
  git -C /home/flynn/ggTrader-refactor worktree add /home/flynn/ggTrader-refactor-p3-wfo    -b refactor/phase-3-wfo    refactor/phase-3
  ```
  In each: `uv sync --frozen --extra test --extra dev`, copy `offline.sh` from the main worktree
  and keep it out of commits. In a linked worktree `.git` is a file, not a directory, so
  `.git/info/exclude` does not exist there; the exclude file git reads is
  `git rev-parse --git-path info/exclude` (run inside the worktree). That path is in the shared
  repository metadata (`/home/flynn/ggTrader/.git/info/exclude`), so one line
  (`offline.sh`) covers every worktree of this repository and is also visible to the main
  checkout. If the owner does not want a shared exclude, skip it and stage by explicit path
  (`git add <files>`, never `git add -A`) instead. (Phase 0-2 Task 0 Step 1b says
  `.git/info/exclude` for the same file; the same correction applies there.) Then copy the frozen tape
  (`cp /home/flynn/ggTrader-refactor/data/snapshots/ohlcv_20261106.parquet data/snapshots/`, hash
  against `data/snapshots/MANIFEST.json`), and check
  `./offline.sh .venv/bin/python -c "from ggTrader.utils.config import get_db_connection_string as g; g()"`
  raises `ValueError: Database configuration not found`.
- [ ] **Step 6:** Report usage to the owner (Phase A done).

---

### Task 20: Extract `_halt_positions` (worktree `...-p3-engine`)

**Files:**
- Modify: `src/ggTrader/engine/portfolio.py`
- Create: `tests/engine/test_run_signals_helpers.py` (grows over Tasks 20 to 22)

**Current range:** `portfolio.py:182-203`, from `halt_positions = {}` to the end of the `for column
in np.unique(halt_columns)` loop. It sits after `_check_shapes(close, entries=entries,
exits=exits)` (`:181`) and before `entries = _as_bool(entries)` (`:204`).

**New signature:**
```python
def _halt_positions(
    halt_mask: pd.DataFrame | None, close: pd.DataFrame
) -> tuple[dict[int, np.ndarray], np.ndarray]:
```
Docstring (Google style, one summary line then Args, Returns, Raises): validates `halt_mask`
against `close` and indexes its flagged bars. Returns `(positions, columns)`, where `positions`
maps a halt-mask column position to the sorted bar positions flagged in it (columns with no flag
are absent), and `columns[c]` is the halt-mask column that serves column `c` of `close`
(`np.arange` when the frames share columns; the symbol lookup for a MultiIndex `close`). With no
mask it returns `({}, np.empty(0, dtype=np.intp))`. Raises `ValueError` with the three messages
unchanged: `halt_mask must have matching dates and known boolean states`, `halt_mask must cover
every symbol`, `halt_mask must match close columns`.

**Order of operations that must be preserved** (the whole `run_signals` order, from `:171`):
1. `_reject_unsupported`; 2. `order_sequence` check; 3. `stop_exit_price` check;
4. `stop_entry_price` default and check; 5. `_check_shapes(close, entries=, exits=)`;
6. **`_halt_positions(halt_mask, close)`**, called with the caller's `close`, before the forward
fill; 7. `_as_bool(entries)`, `_as_bool(exits)`; 8. the mask AND; 9. the intraday block;
10. the time-stop warning and `sig.time_stop`; 11. `raw_close`, `close.ffill()`,
`_stop_arguments`, size checks; 12. `_run_chunks` (the `simulate` closure runs
`_validate_stop_bars` per chunk); 13. the halt overlap check; 14. the missing-bar overlap check.
`halt_columns` was undefined when no mask was given and only used under `if halt_positions:`;
returning an empty array is equivalent.

**Stays importable:** everything in the Global Constraints list. The helper is new and private.

- [ ] **Step 1: Failing unit tests** in `tests/engine/test_run_signals_helpers.py`:
  - `test_halt_positions_none_returns_empty`
  - `test_halt_positions_grid_aligned_maps_columns_to_themselves`
  - `test_halt_positions_compact_mask_maps_symbols_of_a_multiindex_close`
  - `test_halt_positions_only_lists_columns_with_a_flag`
  - `test_halt_positions_error_messages_are_unchanged` (the three messages, exact)
- [ ] **Step 2:** Run the file. Expected: FAIL (`_halt_positions` missing).
- [ ] **Step 3:** Move the block into `_halt_positions` and call it at the old place:
  `halt_positions, halt_columns = _halt_positions(halt_mask, close)`.
- [ ] **Step 4:** The unit tests and the pin check pass. Show the move:
  `git diff --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`; every changed
  line that is not moved is listed in the commit body.
- [ ] **Step 5:** The task gate, with oracle label `t20`. Expected: identical.
- [ ] **Step 6:** Commit: `refactor(engine): extract _halt_positions from run_signals`.

### Task 21: Extract `_validate_intraday_stops`

**Files:** `src/ggTrader/engine/portfolio.py`, `tests/engine/test_run_signals_helpers.py`.

**Current range:** `portfolio.py:211-239`: `stop_columns = None`, then `if stop_open is not None:`
through the last `raise ValueError("stop prices must have the same index and columns as close")`.
`gaps` (`:210`) and `raw_close` (`:247`) stay in `run_signals`.

**New signature:**
```python
def _validate_intraday_stops(
    stop_open: pd.DataFrame | None,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
    close: pd.DataFrame,
    base_config: Mapping[str, Any],
    td_stop: int | None,
) -> np.ndarray | None:
```
Docstring: checks the configuration and shape of an intraday-stop request and returns the
position array that maps each column of `close` to its column in `stop_open` when the stop frames
carry one column per symbol, or `None` when they share `close`'s columns or no intraday stops
were asked for. Raises `ValueError` (list the messages).

**Order inside the helper, unchanged:** (a) return `None` if `stop_open is None`; (b) `td_stop` not
`None`; (c) ATR stop without a trailing stop; (d) no `ts_stop` and no `tp_stop`; (e) per key
`ts_stop` then `tp_stop`: skip `None`, `float()` conversion, finite positive and (for `ts_stop`)
below 1; (f) `high` or `low` is `None`; (g) `stop_open.columns` differ from `close.columns`:
non-MultiIndex or non-unique raises, else the indexer, and a missing symbol raises; (h) each of
`stop_open`, `high`, `low` must match `close`'s index and `stop_open`'s columns.

The helper is called where the block was (after the mask AND, before the time-stop warning). The
`warnings.warn(TIME_STOP_WARNING, UserWarning, stacklevel=2)` call and `sig.time_stop` stay in
`run_signals` (risk 2). The per-chunk `_validate_stop_bars` call stays inside `simulate`.
Ruff `F821` and `F841` confirm that no local of the moved block (`key`, `value`, `fraction`,
`frame`) is read later.

- [ ] **Step 1: Failing unit tests:** `test_validate_intraday_stops_none_when_not_requested`,
  `test_validate_intraday_stops_returns_none_for_shared_columns`,
  `test_validate_intraday_stops_returns_indexer_for_compact_prices`, and one test per check
  (b) to (h), each with the exact message.
- [ ] **Step 2:** FAIL, then **Step 3:** move the block, **Step 4:** unit tests and pin check pass
  (the pins `precedence_halt_mask_over_intraday_config`,
  `precedence_intraday_config_over_time_stop_limit`, `intraday_*` and the warning-attribution test
  are the ones that bite), with the moved-code diff as in Task 20.
- [ ] **Step 5:** The task gate, oracle label `t21`.
- [ ] **Step 6:** Commit: `refactor(engine): extract _validate_intraday_stops from run_signals`.

### Task 22: Extract `_reject_overlaps`

**Files:** `src/ggTrader/engine/portfolio.py`, `tests/engine/test_run_signals_helpers.py`.

**Current range:** `portfolio.py:298-321`, the two loops after `result = _run_chunks(...)`
(`:297`) and before `return result`.

**New signature:**
```python
def _reject_overlaps(
    result: SimResult,
    close: pd.DataFrame,
    *,
    halt_positions: Mapping[int, np.ndarray],
    halt_columns: np.ndarray,
    gaps: Mapping[int, np.ndarray],
) -> None:
```
Docstring: raises `ValueError` for the first trade (in `result.trades` order: by column, then id)
that overlaps a flagged halt bar, then, if there is none, for the first that overlaps a missing
price bar. A trade overlaps when a flagged bar lies at or after its entry bar and at or before its
exit bar. `close` is the forward-filled frame, used only to format the message.

Keep both loops verbatim: the halt loop maps the trade's column through `halt_columns`, the gap
loop uses the column directly, and the messages differ (`halt during an open position at
{date}, {column}` and `missing price bar during an open position at {date}, {column}`). The halt
loop runs first and only `if halt_positions`; the gap loop only `if gaps`. Do not merge them
(Scope).

- [ ] **Step 1: Failing unit tests:** `test_reject_overlaps_noop_without_flags`,
  `test_reject_overlaps_halt_before_gap`, `test_reject_overlaps_reports_first_trade_by_column`,
  `test_reject_overlaps_maps_columns_through_halt_columns`,
  `test_reject_overlaps_message_formats_timestamp_and_symbol_label` (a MultiIndex close; exact
  text). Build `SimResult` by hand or from a tiny `run_signals` call.
- [ ] **Step 2 to 4:** FAIL, move, PASS, with the moved-code diff.
- [ ] **Step 5:** The task gate, oracle label `t22`.
- [ ] **Step 6:** Commit: `refactor(engine): extract _reject_overlaps from run_signals`.
- [ ] **Step 7:** Re-measure `run_signals`. Before: 201 lines (`:122-322`, of which the docstring
  and signature are 49 and the body 152). The three blocks were 22, 29 and 24 lines, and three
  call lines replace them, so the function should now be about 130 lines and the body about 80.
  Record both in the hand-back. A longer result means a block was copied rather than moved.

### Task 23: `ScoringInputs` and `SignalSimOptions` (worktree `...-p3-wfo`)

**Files:**
- Modify: `src/ggTrader/lab/wfo/scorers.py`, `src/ggTrader/lab/wfo/run.py`
- Test: `tests/lab/wfo/test_scorer_inputs.py`

**Interfaces (in `scorers.py`, next to the scorers that consume them):**
```python
@dataclass(frozen=True, eq=False)
class ScoringInputs:
    """The nine values every scorer is built from."""
    strategy: Any
    bars: Bars
    grid: ParamGrid
    names: Sequence[str]
    params: Sequence[Mapping[str, Any]]   # _make_scorer calls this argument `combos`
    mask: pd.DataFrame | None
    base_config: Mapping[str, Any]
    max_bytes: int
    min_losing_days: int = engine_config.MIN_LOSING_DAYS

    def as_args(self) -> tuple[Any, ...]: ...   # the nine, in the constructors' positional order

@dataclass(frozen=True, eq=False)
class SignalSimOptions:
    """Execution options only a signal strategy can honor."""
    intraday_stops: bool = False
    halt_mask: pd.DataFrame | None = None
    order_sequence: str | None = None
    stop_exit_price: str | None = None

    def require_signal_strategy(self) -> None: ...
    def as_kwargs(self) -> dict[str, Any]: ...   # the four, in this order
```
`eq=False` because a frame in a field makes the generated `__eq__` raise. Neither class copies or
validates anything: the scorers keep taking their own copies (`list(names)`, `dict(base_config)`).

- `require_signal_strategy` raises, in this order, with these exact messages: `intraday stops
  require a signal strategy` when `self.intraday_stops` is truthy (truthiness, not `is True`, as
  today); `halt_mask requires a signal strategy` when `halt_mask is not None`; `order_sequence
  requires a signal strategy` and `stop_exit_price requires a signal strategy` likewise.
- `SignalScorer.from_inputs(cls, inputs, options=SignalSimOptions())` returns
  `cls(*inputs.as_args(), **options.as_kwargs())`; `WeightScorer.from_inputs(cls, inputs)` returns
  `cls(*inputs.as_args())`. Both `__init__` signatures stay as they are (Deviation 2). Neither
  calls `prepare`.
- In `run.py`, a new private `_scorer_for(strategy, inputs, options) -> Scorer`: for
  `strategy.kind == "signals"` it returns `SignalScorer.from_inputs(inputs, options)`; otherwise
  `options.require_signal_strategy()` and `WeightScorer.from_inputs(inputs)`. The class names are
  looked up in the module at call time (risk 11).
- `_make_scorer` keeps its exact 13-parameter signature (positional order `strategy, bars, grid,
  names, combos, mask, base_config, max_bytes, min_losing_days, intraday_stops, order_sequence,
  stop_exit_price, halt_mask`) and becomes a wrapper that builds the two objects and calls
  `_scorer_for`. Its docstring says it exists for callers of the old signature.
- `run_wfo` builds one `ScoringInputs` and one `SignalSimOptions` from its arguments and calls
  `_scorer_for`. Nothing else in `run_wfo` changes in this task. In particular the blocks at
  `run.py:242-254` (inputs, `strategy.defaults`, mask) stay exactly as they are (risk 16).

- [ ] **Step 1: Failing tests** (`tests/lab/wfo/test_scorer_inputs.py`):
  - `test_scoring_inputs_as_args_matches_the_constructor_order`
  - `test_from_inputs_builds_the_same_scorer_as_the_constructor` (compare the fields of both)
  - `test_from_inputs_does_not_prepare`
  - `test_signal_options_default_as_kwargs`
  - `test_require_signal_strategy_messages_and_order` (all four set at once raises the first;
    each alone raises its own)
  - `test_scorer_for_weights_rejects_signal_options`, `test_scorer_for_signals_accepts_them`
  - `test_make_scorer_wrapper_keeps_its_signature` (names and defaults against a literal)
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** the new tests, the two existing
  `_make_scorer` tests (`test_toy_known_answers.py:622-667`) and the pin check pass.
- [ ] **Step 5:** The task gate, oracle label `t23`.
- [ ] **Step 6:** Commit: `refactor(wfo): ScoringInputs and SignalSimOptions replace nine and four repeated arguments`.

### Task 24: `FoldRecord`

**Files:**
- Create: `src/ggTrader/lab/wfo/fold_record.py`
- Modify: `src/ggTrader/lab/wfo/run.py` (imports `FoldRecord`, so `run.FoldRecord` resolves)
- Test: `tests/lab/wfo/test_fold_record.py`

**Interfaces:**
```python
FOLD_KEYS: tuple[str, ...]   # the 28 names, in today's order (Deviation 1)

@dataclass(frozen=True)
class FoldRecord:
    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    winner_combo: str
    winner_params: dict[str, Any]
    train_score: float
    oos_score: float
    ndh_passed: bool
    dsr_passed: bool
    gates_passed: bool
    ndh_density: float
    dsr_value: float
    wfe: float | None
    is_sharpe: float
    oos_sharpe: float
    halted: bool
    halted_entering: bool
    used_anchor: bool
    anchor_combo: str
    anchor_params: dict[str, Any]
    anchor_reason: str | None
    deployed_combo: str
    halt_cleared: bool
    winner_trades_in_window: int
    winner_trades_whole_simulation: int
    od12_ndh_flip: bool

    def as_dict(self) -> dict[str, Any]:
        """The legacy fold dict: the same 28 keys in the same order, the same objects."""
        return {name: getattr(self, name) for name in FOLD_KEYS}
```
The annotations document intent. They do **not** coerce: today some of these values are numpy
scalars, and the record must hold exactly what the loop computes (risk 6). The class docstring
carries the comment now at `run.py:405-406` about the anchor fields (review MAJOR-3). Do not use
`dataclasses.asdict`, `dataclasses.replace` or any copy. The record's `as_dict()` is called once
per fold and the resulting dict is the one appended to `fold_results`, as the literal was.

- [ ] **Step 1: Failing tests:**
  - `test_fold_keys_are_the_28_in_todays_order` (a literal in the test)
  - `test_fields_equal_fold_keys`
  - `test_as_dict_key_order`
  - `test_as_dict_returns_the_same_objects` (`rec.as_dict()["winner_params"] is rec.winner_params`,
    same for the Timestamps and a numpy scalar)
  - `test_as_dict_returns_a_new_dict_each_call`
  - `test_record_is_frozen`
- [ ] **Step 2:** FAIL. **Step 3:** add the module, replace the dict literal at `run.py:383-416`
  with `fold_results.append(FoldRecord(fold_num=i + 1, ...).as_dict())`. The values are the same
  expressions, in the same statement order; only the container changes.
- [ ] **Step 4:** New tests and the pin check pass (the golden's type tags are the guard).
- [ ] **Step 5:** The task gate, oracle label `t24`; additionally compare the 13
  `*_oos_equity.csv` byte for byte (the comparer already does).
- [ ] **Step 6:** Commit: `refactor(wfo): FoldRecord holds the 28 fold values and gives the same dict`.

### Task 25: `_score_fold`

**Files:** `src/ggTrader/lab/wfo/run.py`; test `tests/lab/wfo/test_run_wfo_helpers.py` (grows over
Tasks 25 to 27).

**Current range:** the loop body at `run.py:293-425` (`for i, fold in enumerate(folds):` to the
closing verbose `print`), except the three accumulator updates that stay in the loop.

**Interfaces (private, in `run.py`):**
```python
@dataclass(frozen=True, eq=False)
class _FoldContext:
    scorer: Scorer
    strategy_name: str
    names: Sequence[str]
    name_to_idx: Mapping[str, int]
    combos: Sequence[Mapping[str, Any]]
    ndh_threshold: float
    dsr_threshold: float
    policy: WfoPolicy
    n_folds: int
    verbose: bool

class _FoldOutcome(NamedTuple):
    anchor: AnchorSet
    record: FoldRecord | None          # None when the train window had no metrics
    skipped: dict[str, Any] | None     # the skipped_folds entry when record is None
    winner: dict[str, Any] | None
    oos_curve: pd.Series | None
    running_value: float
    ran_test: bool
    od12_flip: bool
    state: WfoState

def _score_fold(
    ctx: _FoldContext, i: int, fold: Fold, state: WfoState, running_value: float
) -> _FoldOutcome:
```
Docstring requirements: say that it prints the two verbose fragments itself (the interleaving with
the scorer calls is part of the behavior), that it appends to `state.wfe_history` and
`state.oos_sharpes` **in place** and returns the breaker's copy, and that a skipped fold returns
the state and running value it was given.

**Order of operations inside `_score_fold`, unchanged** (the line numbers are today's):
1. build the three `WindowReq` (anchor, train, test) (`:295-297`);
2. `anchor = pick_anchor(scorer.metrics(anchor_req))` (`:298`);
3. verbose header with `end=""` and `flush=True` (`:299-307`);
4. `train_metrics = scorer.metrics(train_req)`; if empty, return the skipped outcome
   (`{"fold_num": i + 1, **fold._asdict(), "reason": "no_train_metrics"}`) with `anchor` set
   (`:308-313`);
5. `scores`, `best_idx`, `winner`, `winner_idx` (`:315-318`);
6. `ndh = run_ndh(train_metrics, combos, best_idx, ndh_threshold)`, then
   `run_dsr(scorer.returns(train_req, winner_idx), len(combos), dsr_threshold)` (`:320-324`);
7. `whole = scorer.trade_counts(train_req, cumulative=True)`, `legacy_metrics`, the **second**
   `run_ndh`, `od12_flip` (`:328-333`);
8. `halted_entering = state.halted`; the anchor override and `anchor_reason` (`:335-342`);
9. the test window: `deploy_idx`, `scorer.metrics(test_req, [deploy_idx])`, `oos_score`,
   `scorer.equity`, the running-value scaling `oos_running_value * (eq_test / eq_test.iloc[0])`
   (risk 12), or the flat curve through `scorer.window_dates` when `deploy_idx is None`
   (`:344-361`);
10. `is_sharpe`, `oos_sharpe`, `wfe_val` (`:363-368`);
11. the two in-place appends, then the breaker or shadow re-entry (`:369-381`);
12. build the `FoldRecord` with `halted=<new state>.halted` and `halted_entering` read in step 8;
13. the verbose result line (`:418-425`).

**The loop after the change:**
```python
for i, fold in enumerate(folds):
    out = _score_fold(ctx, i, fold, wfo_state, oos_running_value)
    anchor = out.anchor
    if out.record is None:
        skipped_folds.append(out.skipped)
        continue
    n_test_runs += out.ran_test
    if out.od12_flip:
        od12_flips.append(i + 1)
    if out.oos_curve is not None:
        oos_curves.append(out.oos_curve)
    oos_running_value = out.running_value
    wfo_state = out.state
    fold_results.append(out.record.as_dict())
    fold_winners.append(out.winner)
```
`n_test_runs += out.ran_test` adds a bool to an int: write `n_test_runs += int(out.ran_test)`.
`run_dsr`, `run_ndh`, `composite_score`, `best_index`, `pick_anchor`, `compute_wfe`,
`check_circuit_breaker` and `check_shadow_reentry` stay module-level names in `run.py`, called by
bare name (risk 11). `fold_winners` and `fold_results` keep today's order (a skipped fold appends
to neither).

- [ ] **Step 1: Failing tests** (`tests/lab/wfo/test_run_wfo_helpers.py`):
  - `test_score_fold_skipped_returns_state_and_running_value_unchanged`
  - `test_score_fold_appends_to_the_state_it_was_given` (risk 8)
  - `test_score_fold_calls_the_scorer_in_the_pinned_order` (uses the Task 18 `ScriptedScorer`
    transcript)
  - `test_score_fold_verbose_off_prints_nothing`
  - `test_score_fold_flat_curve_when_no_anchor`
- [ ] **Step 2:** FAIL. **Step 3:** implement and call it. **Step 4:** new tests and the pin check
  pass (every scripted scenario and the verbose pin bite here).
- [ ] **Step 5:** The task gate, oracle label `t25`.
- [ ] **Step 6:** Commit: `refactor(wfo): score one fold in _score_fold`.

### Task 26: `_stitch_oos`

**Files:** `src/ggTrader/lab/wfo/run.py`, `tests/lab/wfo/test_run_wfo_helpers.py`.

**Current range:** `run.py:427-444`, from `nan_metrics = {...}` to the `else:` branch that builds
the empty result.

**New signature:**
```python
def _stitch_oos(
    oos_curves: Sequence[pd.Series], spy_close: pd.Series, start_cash: float
) -> tuple[pd.Series, dict[str, float], dict[str, float]]:
```
Returns `(oos_equity, oos_metrics, spy_metrics)`. Docstring: concatenates the per-fold curves,
keeps the last of any duplicated date, scores the result with `curve_summary`, and scores the SPY
close over the same dates scaled to `start_cash`; with no curve, or fewer than two SPY dates, the
affected metrics are a three-key NaN dict (`sharpe`, `cagr_pct`, `max_drawdown_pct`), **not** the
six-key `curve_summary` shape. Each NaN dict is built fresh (`dict(nan_metrics)` twice, as today),
never a shared module constant. The expressions
`oos_equity[~oos_equity.index.duplicated(keep="last")]`,
`spy_close.reindex(oos_equity.index).ffill().dropna()` and
`start_cash * (spy_oos / spy_oos.iloc[0])` are copied unchanged (risk 12).

- [ ] **Step 1: Failing tests:** `test_stitch_oos_no_curves_gives_empty_float_series_and_nan_dicts`,
  `test_stitch_oos_keeps_last_of_duplicated_dates`, `test_stitch_oos_nan_dicts_have_three_keys`,
  `test_stitch_oos_nan_dicts_are_not_shared`, `test_stitch_oos_spy_with_one_date_gives_nan_spy`,
  `test_stitch_oos_scales_spy_to_start_cash`.
- [ ] **Step 2 to 4:** FAIL, implement, PASS with the pin check.
- [ ] **Step 5:** The task gate, oracle label `t26`.
- [ ] **Step 6:** Commit: `refactor(wfo): stitch the out-of-sample curve in _stitch_oos`.

### Task 27: `_build_diagnostics`

**Files:** `src/ggTrader/lab/wfo/run.py`, `tests/lab/wfo/test_run_wfo_helpers.py`.

**Current range:** `run.py:462-516`, from `n_folds = len(folds)` to the closing brace of the
`diagnostics` dict.

**Interfaces:**
```python
class _Tally(NamedTuple):
    n_folds: int
    n_test_runs: int
    od12_flips: list[int]
    oos_metrics: dict[str, float]
    spy_metrics: dict[str, float]

def _build_diagnostics(
    scorer: Scorer,
    inputs: ScoringInputs,
    options: SignalSimOptions,
    tally: _Tally,
    *,
    started: float,
    fold_started: float,
) -> dict[str, Any]:
```
The dict is the 26-key dict of Task 18, in that order, built with the same expressions, copied
verbatim, including the two conditional expressions whose precedence is easy to "fix" (risk 15):
`order_sequence or ("auto" if intraday_stops else "default") if strategy.kind == "signals" else
"auto"` and the `stop_exit_price` one. Mapping of old names to new: `strategy` is
`inputs.strategy`, `bars` is `inputs.bars`, `combos` is `inputs.params`, `pit` is `inputs.mask`,
`base_config` is `inputs.base_config` (equal to the caller's for every key read, risk 14),
`max_bytes` and `min_losing_days` come from `inputs`, and the four options from `options`.
`price_stop_active` and `stop_settings = getattr(scorer, "stop_settings", None)` move with it.
`started` and `fold_started` stay as the two `time.time()` readings taken where they are today
(`run.py:233,291`); `timings_s` keeps `scorer.timings` keys first, then `folds_s`, then
`total_s`. `peak_rss_mb` stays `round(_peak_rss_mb())`.

- [ ] **Step 1: Failing tests:** `test_build_diagnostics_has_the_26_keys_in_order`,
  `test_build_diagnostics_legacy_equivalent_simulations_counts_skipped_folds`,
  `test_build_diagnostics_weights_run_has_no_stop_settings`,
  `test_build_diagnostics_order_sequence_precedence` (the four combinations of kind and
  `intraday_stops`, plus an explicit `order_sequence`), `test_build_diagnostics_timings_order`.
- [ ] **Step 2 to 4:** FAIL, implement, PASS with the pin check.
- [ ] **Step 5:** The task gate, oracle label `t27`.
- [ ] **Step 6:** Commit: `refactor(wfo): build the run diagnostics in _build_diagnostics`.
- [ ] **Step 7:** Re-measure `run_wfo`. Its body should now read in three screens: setup (inputs,
  scorer, requests), the fold loop above, then stitch, live pick, table, diagnostics and the
  return. Record its length and parameter count (23, unchanged) in the hand-back.

### Task 28: Integration, docs and changelog

**Files:**
- Modify: `docs/architecture.md` (the walk-forward and engine sections: name the new records and
  helpers, point to `lab/wfo/fold_record.py`)
- Modify: `docs/changelog.md` (one entry: the three `run_signals` helpers, the new `run_wfo`
  pieces, the new characterization tests, no result changed, the 28-key correction)

- [ ] **Step 1: Merge both branches** into `refactor/phase-3` in the main worktree
  (`git merge --no-ff refactor/phase-3-engine refactor/phase-3-wfo`, or one at a time). The
  chains share no file, so the merge is conflict-free; a conflict means a chain edited a file it
  does not own, which is a finding to report.
- [ ] **Step 2:** The pin check, then the neighbourhood run, on the merged tree.
- [ ] **Step 3: Docs.** Edit the two files. No code changes.
- [ ] **Step 4:** `./offline.sh scripts/run_tests.sh` (fast core) and the ruff checks pass.
- [ ] **Step 5:** Commit: `docs: refactor phase 3 (research engine helpers and fold records)`.

### Task 29 (OPTIONAL, gated on O-P3-1): replace the string return with an exception

**Do not start without a recorded owner answer. Default: skip.**

**What the string return is today:** `run_wfo` returns
`"WFO: <strategy> | no valid folds (need >= 15 months)"` when `generate_folds` yields no fold
(`run.py:236-240`). That needs an evaluation window under 15 months. Nothing computed has been
done by then, so raising there loses nothing.

**Every caller.** Inventory command (direct, aliased and module-qualified calls; the helper
`run_wfo_case` of `scripts/legacy_reference_runs.py` is a different function and is not a caller):
`grep -rnE "(^|[^_[:alnum:]])(wrun\.|run\.)?run_wfo\(" src scripts research tests --include=*.py`.
Each import-resolution claim below was measured on 2026-11-18 by trying every `ggTrader` import of
the file (`from x import y` resolved by `hasattr` or by importing `x.y`).

*A. Editable callers that depend on the string:*

| Caller | What it does with a string |
|---|---|
| `src/ggTrader/lab/blend.py:101-116` | `isinstance(result, WfoResult)` else `SystemExit("blend sleeve ...: WFO produced no valid folds (...)")` |
| `src/ggTrader/lab/wfo/cli_support.py:118-133` (`run_vbt_wfo`, return type `WfoResult \| str`) | prints it and returns it |
| `src/ggTrader/lab/cli.py:187` | `outcome if isinstance(outcome, str) else outcome.table` (changes only if `run_vbt_wfo` stops returning a string) |
| `tests/lab/wfo/test_toy_known_answers.py:214-231` | asserts the string |
| `tests/lab/test_blend.py:127-140` | a fake `run_wfo` returns a string and expects `SystemExit` |

All other callers in `tests/` (`test_p5b_review_fixes.py`, `test_report_inputs_weights.py`,
`test_leak_golden_memory.py`, `test_causality.py` and the rest of `test_toy_known_answers.py`)
use the result's attributes directly and would already fail on a string.

*B. Never-edit callers that can run today:*

| Caller | What it does with a string |
|---|---|
| `scripts/p5_wfo_validation.py:219-220` (oracle driver) | `isinstance(result, str)` then `RuntimeError(result)`; its windows (2021-01-31 to 2022-07-31 and to 2026-04-30) yield 2 and 17 folds, so the branch is unreachable; with an exception it would fail with the same text instead |
| `scripts/legacy_reference_runs.py:483` | runs on the legacy tag's package, not this `run_wfo`; unaffected |
| `research/programme/vectorbt-engine/p10-evidence/dsr_one_combo.py:68` | **does not check for the string**: it reads `res.fold_results` and the keys `gates_passed` and `dsr_value` of each row directly (so it also depends on those fold keys), and a string would already crash it. All its imports resolve; its window comes from `--eval-start/--eval-end`, so it would raise instead of crashing with `AttributeError` for a window under 15 months. No dead branch |

*C. Frozen callers that cannot run today for other reasons* (each imports a module that no longer
exists, such as `ggTrader.lab.sweep`, `ggTrader.lab.strategies.STRATEGY_REGISTRY`,
`EnsembleSignal`, `ggTrader.lab.metrics`, `ggTrader.lab.simulate` or `wfo._sweep_fold`), so
Task 29 changes nothing for them whatever it does:
- *With an `isinstance` check on the result* (a dead branch either way):
  `evidence/vbt-spec/S3-evidence/e2e_wfo.py:150-154`, `p1-parity/p1_parity_driver.py:341`,
  `legacy-scripts/congress_trades_correlation_check.py`, `insider_cluster_correlation_check.py`,
  `max_effect_correlation_check.py`, `pead_correlation_check.py`,
  `fx_hedge_overlay_static_baseline_check.py`, `leveraged_trend_research.py`,
  `pairs_stat_arb_research.py` and `leveraged_rotation_research.py` (`:65`, `if not
  isinstance(result, WfoResult): return f"{universe}: {result}"`, which is the one place a
  string is formatted into a report).
- *Without a check, using the result directly or as the legacy table string (already incompatible
  with today's `WfoResult` return):* `legacy-scripts/ablation_voters.py` (`result.table`),
  `anchor_fix_reproduction_wfo.py` (`table = run_wfo(...)` at `:266`), `position_sizing_wfo.py`
  (`table = run_wfo(...)` at `:291`), `audit_reruns_wfo.py`, `core_spy_blend.py`,
  `cross_asset_sleeve_wfo.py`, `lever_diagnostic.py`, `midcap_bias_calibration.py`,
  `midcap_research.py`, `month_end_treasury_wfo.py`, and `profiling/prof_wfo.py:38`, which ignores
  the return value but accepts any `--eval-end` (`:8`), so its default window (2 folds) does not
  prove every invocation has folds; it cannot import today, so this is moot.

**Correction of the first draft.** It said four frozen scripts keep a string check. That is wrong:
the DSR script and the profiling script ignore the return. The frozen scripts with a string check
are all in group C, which cannot run. The only runnable caller with a dead branch is the oracle
driver in group B, which is never edited.

**Verdict.** Not blocked. Nothing that can run today and records a result depends on the string:
the runnable never-edit callers (group B) either never reach it or already fail on it. But the
public contract changes, group C keeps ten dead `isinstance` branches, and the gain is one early
return. **Recommendation: skip this task** unless the owner decides otherwise.

- [ ] **Step 1 (Haiku inventory, Opus check):** rerun the inventory command and the import-resolution
  scan, and confirm the three groups above are complete and each file is in the right group. Any
  caller found outside the lists is a finding before anything is edited. A caller that moves from
  group C to "can run" (because a removed module returned) moves the verdict.
- [ ] **Step 2: Failing tests:** `test_run_wfo_raises_no_valid_folds_with_the_old_message`
  (`NoValidFoldsError`, a `ValueError` subclass, message equal to today's string),
  `test_run_vbt_wfo_still_prints_and_returns_the_message` (so `lab/cli.py` is unchanged),
  `test_blend_exits_with_the_same_message`. Update the two existing tests above.
- [ ] **Step 3:** Add `NoValidFoldsError` to `run.py`; `run_wfo` raises it; the return annotation
  becomes `WfoResult`; the docstring's Returns becomes Raises. `blend.py` catches it and raises
  the same `SystemExit` text; `run_vbt_wfo` catches it, prints, and returns the message.
- [ ] **Step 4:** The task gate (oracle label `t29`). The pins' `toy_no_valid_folds` scenario now
  records the exception; regenerate that one entry, review the diff, and say so in the commit.
- [ ] **Step 5:** Commit: `refactor(wfo): no valid folds raises NoValidFoldsError instead of returning a string`.

**Phase 3 gate and hand-back:**
- The test-outcome check passes against `p3base-core.xml,p3base-studies.xml`, with the tests of
  Tasks 16 to 18 and 20 to 28 named in `--must-pass`.
- The oracle check (label `phase3`) exits 0 against `/home/flynn/ggTrader-refactor-evidence/baseline`.
- The pin check passes in exact mode in three processes.
- The verify-pins check passes with the counts in `.github/workflows/ci.yml` as Phase 0-2 left
  them (none of `portfolio.py`, `run.py` or `scorers.py` is a pinned file; the 2026-11-11
  evidence JSON `research/programme/_intraday_stop_ohlc_check_20261111.json` records a
  `validator_file_sha256` for `portfolio.py` that matches no commit in the file's history, and
  `verify_pins.py` does not read it).
- `ruff check .` and `ruff format --check .` are clean.
- Codex Sol and Claude Opus each review the whole branch in a fresh context. Sol also checks this
  plan against design section 5.3 and the 28-key correction.
- Report to the owner with: the BASE3 and head commits, line counts of `run_signals` and
  `run_wfo` before and after, and usage. Merge only on their say-so.

## Execution roles (`CLAUDE.md`, delegation preference of 2026-11-08)

**Quota first.** The agent cannot see plan usage. Before Task 15 ask the owner for current
headroom and size the run to it. If it is unknown, run Phase A only. The phases and the number of
subagent runs expected:

| Phase | Tasks | Builder runs | Reviewer runs |
|---|---|---|---|
| A: pins | 15 to 19 | 3 | 2 Sol, 1 Opus (goldens) |
| B: engine chain | 20 to 22 | 1 | 1 Opus |
| C: WFO chain | 23 to 28 | 2 | 1 Opus |
| optional | 29 | 1 Haiku inventory | 1 Opus |
| branch gate | | | 1 Sol, 1 Opus |

Phases B and C can run together if headroom allows; otherwise B first, since it is the smaller.
Usage is reported after each phase so the owner can stop early.

- **Orchestrator:** Claude Sonnet. It re-verifies every task itself: it reads the diff (with
  `--color-moved`), reruns the task gate, and does not accept a builder's claim unchecked.
- **Builders:**
  - Claude Sonnet: Tasks 17 and 18 (they need judgment about what a case must show).
  - Codex Terra (or Luna): Task 16 and Tasks 20 to 28.
  - Claude Haiku: Task 29's caller inventory and the line-anchor re-measurement of Task 15, with
    every figure checked by a stronger model.
- **Reviewers, never from the builder's own family:**
  - Codex Sol reviews Claude-built work (Tasks 17 and 18).
  - Claude Opus reviews Codex-built work (Tasks 16 and 20 to 28), reads both goldens (Task 19
    Step 4), and takes the Task 29 recommendation.
  - The whole branch gets both Sol and Opus, each in a fresh context given only the evidence and
    the claims. The pins are written by a different model family from the one that refactors under
    them, which is the independence this phase relies on.
- The lead remains responsible for the final call.
