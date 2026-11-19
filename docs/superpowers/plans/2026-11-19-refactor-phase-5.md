# Refactor phase 5 (split `PaperTrader._run` into step objects, migrate the trader tests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 826-line `PaperTrader._run` into a short orchestrator that calls one step object at
a time (`PreflightGate`, `PortfolioView`, `RiskAssessor`, planning functions, `OrderExecutor` /
`DryRunExecutor`, `CorporateActions`, `RunReporter`), move the trader's legacy tests off string-patched
persist names, and retire the two temporary devices the design names (the frozen copy `_run_legacy`, and
the string-patch adapter `ModuleLedger`, the latter by an owner-gated task), without changing one order,
one Telegram line, one SQL statement, one log line, one clock read position or one price row.

**Architecture:** Same method as Phases 0-4, with one addition. Safety nets first (Tasks 46-51): the
103 transcripts and the SQL golden stay byte-identical and are joined by named tests for the seven rules
of design 5.1 (and the characterizations they miss), a frozen copy of today's `_run` (`_run_legacy`,
hash-pinned), a seeded differential test (old run against new run, transcript plus log records) and
**immutable recorded per-seed digests that outlive the copy**, and a ratchet on `_run`'s size. The nets are
split into an **immutable behavioral set** (rules, generator, digests, seam contract; hash-recorded) and a
**mutable structural set** (ratchet, differential, frozen-copy check; reviewed, not hash-recorded, retired or
edited by Tasks 57-74). Then the pieces are built inert (Tasks 52-54), the bodies of the order and
corporate-action methods are moved behind one-line forwards (Tasks 55-56), and the regions of `_run` are cut
out front to back, one commit each (Tasks 57-63), every commit against all the nets. Legacy tests are
migrated last, in eight batches that keep every one of the 277 node ids (Tasks 65-73); only then is the
frozen copy deleted (Task 74), the adapter retired if the owner approves (Task 75).

**Tech Stack:** Python 3.12 (uv-managed `.venv`), pytest through `scripts/run_tests.sh`, ruff, frozen
dataclasses, `typing.Protocol`, `typing.TypedDict`; no new dependency.

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md),
sections 3 (principles), 5.1 (the run), 6 (safety and verification) and 7 (the phase 5 row). Reused by
reference from earlier plans: the offline wrapper, the oracle and test-outcome checks and the hash
records ([Phase 0-2](2026-11-17-refactor-phase-0-2.md)); the record, ledger and config classes this
phase builds on ([Phase 4](2026-11-18-refactor-phase-4.md), report `task-44-report.md` in
`.superpowers/sdd/2026-11-18-refactor-phase-4/`). Task numbers continue from Phase 4 (it ended at 45).

**Revision 1 (2026-11-19).** Every line number was read on 2026-11-19 from
`/home/flynn/ggTrader-refactor`, branch `refactor/phase-4`, code head `73c35417` (Phases 3 and 4
integrated locally), `src/ggTrader/paper/trader.py` 1,600 lines. Treat line numbers as approximate and
re-grep the quoted anchor string; Task 46 records the numbers again.

**Revision 2 (2026-11-19, after independent Sol and Opus reviews, both "not ready").** Corrections, each
recorded in `.superpowers/sdd/2026-11-19-refactor-phase-5/revision-report.md`: the slot-arithmetic example
(three held, one stopped, one ordinary sell permits **two** buys; the double count would permit three);
a working partial sell raises in reconciliation, a terminal unfilled sell blocks buys; an invalid exit still
allows a reserve-restoring sweep SELL, an unfilled exit leaves the sweep BUY enabled; a sell's
`market_value` is read **after** the order is submitted and never in a dry run; the nets no longer need a
temporary `xfail`, and are split into immutable and mutable sets with explicit pre- and post-retirement
gates; the gate wrappers are written out (env controls after `offline.sh`, the p5 oracle at 6G only with the
owner's recorded approval); logs are pinned at DEBUG over a named logger set; the differential test is
honest about the helpers `_run_legacy` shares with the new code; a kth-lock-failure wrapper is specified;
every mutation is mapped to a test that reaches it; all 277 baseline node ids (not 197 functions) are
collected; the Telegram dual-base slice and the `ledger.py` docstring were already done in `1c7a3138` and
`612e169f`; the reviewer-family rule is applied; an autouse guard for unstubbed persist names is kept; a
guard test keeps `InMemoryLedger` out of production code; the executors reach `submit`, `poll` and
`reconcile` through the seams; planners are `advance()`-style so PEP 479 cannot change an exception type;
the interim flow state is defined; `RunHalted` becomes a returned value; and retiring `ModuleLedger` is
specified as a concrete owner-gated task (Task 75), with the phase reported **incomplete against the
design** until it is done. `BASE5` is `73c35417`, not the plan commit.

**Revision 3 (2026-11-19, after the second Sol and Opus reviews), recorded in
`.superpowers/sdd/2026-11-19-refactor-phase-5/revision-2-report.md`.** Phase-gate hash checks split into
files that existed at `BASE5` (compared to `BASE5`) and new immutable files (compared to the Task 51
hashes), and the gate tools, conftests, tiers, both runner scripts, both comparers and `offline.sh` are
hash-pinned (`$E/p5-tools.sha256`); Task 74 retires every removed `legacy` parametrized node by name and
defines the pre- and post-retirement must-pass files exactly; Task 75 names every frozen file it edits,
keeps the per-seed digest values and amends only the digest file's runner-hash provenance, and retires the
`ModuleLedger` baseline nodes through a filtered baseline, not an edit of the comparator; `Executor.submit`
returns the broker order id and the **flow** builds `SubmittedOrder` after the old counter updates and the
post-submit `market_value` read, never in a dry run; `after_orders` receives every order, sells included,
and `unfilled_exit_errors` receives the raw broker positions; standalone scripts get the repository root on
`PYTHONPATH` and their own `blocked_connections()` per process, and run under offline, cap and
serialization; the p5 oracle cap is exactly the approved 6G (no override), the p7 driver is bounded, and the
tests run even while the oracle half is pending; the generator reaches late-stage planner failures after an
earlier submission (mutation SM16) and a `membership_source` failure inside the signal `try` (SM17); planners
are **explicit iterator classes**, not generators, because the real `find_catastrophe_stops` turns a
`StopIteration` inside its own generator expression into `RuntimeError` today and a heuristic adapter would
change that; digests are recorded from the Task 49 commit that holds the runner (not `BASE5`) after a
pre-record review gate; `run_with` moves to a hash-pinned file and `impl.py` holds only `IMPLS`; buys keep
their order and cross-sleeve duplicates, only sells are sorted unique; the 600-seed checks share one cached
pass per session; amendment and verdict authority follow the family rule; shared constants and helpers get a
non-circular home and new modules never import `trader`; the slot-arithmetic explanation is corrected; and
the final gate results are written after the gate, in Task 78.

## Scope

**In this plan** (design 5.1 and the phase 5 row of section 7):
1. The step objects of the table in design 5.1, in the modules listed under "Target structure".
2. The named test for each of the seven rules of design 5.1, run against both old and new code.
3. The seeded comparison of the old run with the new one ("planning equivalence"), and the recorded digests
   that replace it once the old run is deleted.
4. Migration of `tests/paper/test_trader.py` from 239 persist-name string patches to an
   `InMemoryLedger` rig, keeping all 277 node ids (197 test functions; the rest are parametrized cases).
5. Deleting `_run_legacy` and the implementation switch, then ruff complexity rules for the survivors,
   docs and one changelog entry.
6. **Owner-gated (Task 75):** retiring `ModuleLedger` and migrating the transcript harness to an injected
   ledger, as design section 7 says ("delete the adapter and `_run_legacy`").

**What Phase 5 does NOT do:**
- It does not touch `src/ggTrader/paper/persist.py` (no SQL, no transaction mode, no bind dictionary:
  Task 45 of Phase 4 stays skipped), `ledger.py` (its stale docstring was already corrected in `612e169f`;
  Task 52 no longer edits it; Task 75, if approved, deletes `ModuleLedger` from it), `risk.py`,
  `cash_sweep.py`, `catastrophe_stop.py`, `gates.py`, `split_check.py`, `dividend_check.py`,
  `untradable.py`, `notifier.py`, `alpaca_broker.py`, `signal_runner.py`, `overlay.py`, `cli/main.py`,
  `data/`, `lab/`, `engine/`, anything under `research/`, or the oracle and legacy scripts.
- It does not change `ggt paper` output: `run()` still returns `dict`, with the three shapes and key
  orders pinned by Phase 4 (`tests/paper/test_phase4_pins.py`).
- It does not change a Telegram message. The "total" percentage against both bases (owner decision 4 of
  2026-11-18) **is already done** (`1c7a3138`, `feat(paper): show total P&L against both reference
  values`, with its goldens regenerated and reviewed: 56 transcripts changed and `total_pnl_two_bases`
  added). Phase 5 **preserves** it: the reporter passes the same `starting_capital` keyword to
  `daily_summary`, and `notifier.py` is untouched (Task 63 pins both).
- It does not fix quirks found on the way (listed under "Preserved quirks"); it records them.
- It does not add a runtime refusal of `InMemoryLedger` in a live run: that would change behavior and is an
  owner decision (O-P5-5). Phase 5 adds a static guard instead (Task 50).
- It does not run `ggt paper --live`, touch cron, the `ggtrader_live` container or deploy. The trader
  stays off.
- It does not merge, push or deploy. Branch `refactor/phase-5` ends at the Task 78 hand-back.

## Owner decisions and open questions needed before execution

A default is not an approval. Nothing here runs until the owner says so.

- **O-P5-1, retiring the string-patch adapter (`ModuleLedger`).** Design section 7 says Phase 5 deletes it,
  so **keeping it leaves Phase 5 incomplete against the approved design**, and no report may call the phase
  complete while it stays. The transcript harness (`tests/paper/transcript/harness.py`, hash-pinned,
  immutable) patches the 25 persist names in the `ggTrader.paper.trader` namespace and relies on
  `ModuleLedger` to make that reach the trader; `tests/paper/test_phase4_pins.py` (immutable) asserts the
  names stay bound in the trader module. Retiring the adapter therefore needs the owner's written approval
  of edits to a **closed list of frozen files** (Task 75's table: the harness, the Phase 4 pins, this phase's
  digest runner, the seam contract, the digest file's runner-hash provenance field and, if they patch persist
  names, the rule tests and their support file, plus two Phase 4 ledger tests), each proven by the 103
  goldens staying byte-identical and the 600 per-seed digests staying unchanged, and a filtered baseline for
  the `ModuleLedger` tests that disappear. Task 75 specifies this concretely. *Recommended: approve Task 75 after Task 74 passes.
  Until the owner decides, the hand-back reports Phase 5 as "complete except adapter retirement (O-P5-1)".*
- **O-P5-2, `RunContext` and the clock.** Design 5.1 says the context holds `now_pt` and "the clock is
  read once". Today `now_pt` is read after signal generation (minutes after the run starts), at the
  late-run gate, and the cutoff is 12:55 PT for a 12:45 PT cron. Reading it once at the top would let a run
  that began on time but finished its signals late slip past the gate. *Default: `today_et` is read once, at
  the freshness gate where it is read now; `now_pt` is read at the late-run gate; no other clock read
  moves (Deviation 1).*
- **O-P5-3, quota and caps.** The owner reported ample Claude and Codex quota on 2026-11-19 (recorded;
  the agent cannot see usage, so usage is still reported after each phase). Caps: every pytest run
  `GGT_TEST_MEM_MAX=3G`, serialized. **The p5 oracle driver peaked at 3,916 MB at the baseline
  (`peak_rss_mb` in `$E/baseline/p5/*.json`), so it cannot pass under 3G; it needs the 6G cap
  (`run_capped.sh` default), and 6G was approved only for the Phase 3 and 4 final runs. Phase 5 needs its own
  recorded approval.** Until the owner writes it (the orchestrator then creates
  `$E/p5/owner-approved-6G.txt` quoting it, first line exactly `cap=6G`), `gate_oracles.sh` refuses to run the
  p5 driver, the oracle half of every gate is reported "not run: awaiting O-P5-3", and no phase gate is
  declared passed. **The tests half still runs and is still compared** (the wrappers of Task 46 Step 3 do not
  stop at the refusal): a gate is then reported "tests half passed, oracle half pending", which is not a pass.
  The p5 cap is the literal `6G`, the approved figure, with no environment override and no other value; the
  p7 driver is bounded at 3G and 900 s and needs no approval. A 137 or 143 exit stops the task and is
  reported, never retried with a higher cap. The owner may also drop the per-task oracle (Tasks 55-63, 74
  and 75) for Phase 5 by saying so; the phase gates keep it.
- **O-P5-4, what "kept seams" mean.** Eight private `PaperTrader` methods are assigned or patched by tests
  (`_submit_order`, `_poll_orders`, `_reconcile_order_intents`, `_reconcile_pending_orders`,
  `_missed_exits`, `_compute_split_corrections`, `_accrue_dividends`, plus `_risk`, `_dry_run` and
  `_paper_lock` attributes). *Default: they stay as one-line forwards after the bodies move (Seam rule
  S1); retiring them is a separate owner-approved slice.*
- **O-P5-5, runtime refusal of `InMemoryLedger` in a live run.** Phase 4's hand-back noted that a live
  `PaperTrader` can be handed an `InMemoryLedger` and would then trade without durable state. A runtime
  refusal changes behavior (a live construction that works today would raise), so Phase 5 does not add
  one. *Default: only the static guard of Task 50 (production code never imports or injects it).*

## Global Constraints

> **Heavy-job rules (AGENTS.md, 2026-11-18).** One heavy job at a time, ever: one pytest run or one oracle
> driver. Check `ps -eo args | grep -E "pytest|run_tests|p5_wfo|p7_live"` before starting one. A run that
> prints `waiting for another heavy ggTrader job` waits; nobody kills another run or bypasses
> `scripts/run_tests.sh` and `scripts/run_capped.sh`. Exit 75 is busy or short of memory, 137 or 143 is a
> hang: stop, report, never raise a cap. Builders run only small targeted paths. **Every delegated brief
> contains this block verbatim and adds: "You must not run `gate_oracles.sh`, `scripts/p5_wfo_validation.py`,
> `scripts/p7_live_golden.py`, `tests/paper/` as a directory, a whole tier or `--full`; the orchestrator runs
> those." (A Phase 4 builder ran the oracles anyway; it is recorded as a boundary violation.)** **Every
> standalone script this plan adds (the Task 49 log investigation and the digest recorder included) is a heavy
> job too: it runs only as `./offline.sh env PYTHONPATH=<repository root> ... scripts/run_capped.sh <python>
> <script>` (see "Standalone scripts" below), capped, serialized, one process at a time.**

Everything in the Phase 0-2 and Phase 4 plans' Global Constraints applies and is not repeated: offline by
construction (`./offline.sh`, scrubbed environment, no `.env`, no database address), tests only through
`scripts/run_tests.sh`, never `ggt paper --live`, no edit under `research/`, `ohlcv.timestamp` stays naive
UTC, ruff clean, commits end with the attribution line from the session instructions, synthetic data only in
delegate prompts and goldens. Phase 5 adds:

- **`offline.sh` scrubs the environment (`env -i`).** A control variable written before it is dropped. Every
  control goes **after** it: `./offline.sh env GGT_TEST_MEM_MAX=3G scripts/run_tests.sh ...`,
  `./offline.sh env GGT_RUN_MEM_MAX=6G GGT_RUN_MAX_SECONDS=1800 scripts/run_capped.sh ...`, and the dry-run
  print is `./offline.sh env GGT_TEST_DRY_RUN=1 GGT_TEST_MEM_MAX=3G scripts/run_tests.sh <paths> -m "not slow"`.
- **Never real network or database, in any test or helper.** New tests build `PaperTrader` only with the
  harness fakes (`FakeBroker`, `RecordingNotifier`) or `MagicMock`, and only `InMemoryLedger` or the
  harness's wrapped persist names. They never construct `AlpacaBroker`, `TelegramNotifier`, `PgPaperLedger`
  or call `run_paper_trading`. The connection guard (net N) stays on; a new AST test (Task 50) forbids the
  new modules from importing `psycopg2`, `sqlalchemy`, `requests`, `httpx`, `alpaca*`,
  `ggTrader.paper.persist`, `ggTrader.paper.alpaca_broker`, `ggTrader.paper.notifier`,
  `ggTrader.paper.signal_runner` or `ggTrader.lab`.
- **Everything is bounded, and the fast core stays fast.** Fixed seed counts, never read from the
  environment: `FAST_SEEDS = 100` (the first 100 seeds) and `FULL_SEEDS = 600`. **Each seed is run once per
  pytest session and summarized once.** `runner.summarize_seed(seed)` runs the **new** implementation under
  `blocked_connections()` (a session-scoped fixture is set up *before* the function-scoped autouse guard, so
  it brings its own) and returns a small immutable `SeedSummary`: the digest, the set of witness names
  reached, the list of safety-invariant violations (a)-(g), and the harness error text or `None`. It never
  keeps the `Recorded` transcript, so memory stays flat. Two session fixtures in `test_run_digests.py` fill
  one shared cache keyed by seed: `fast_summaries` (seeds 0-99) and `full_summaries` (seeds 0-599, computing
  only the seeds the cache lacks). The harness-error, witness, digest and safety tests are **assertions over
  that cache**, not separate runs; the 600-seed tests are `@pytest.mark.slow` and
  `@pytest.mark.timeout_seconds(600)` and the first of them pays for the pass. The differential
  (`test_run_differential.py`, two implementations) has its own 100-seed pass and is the only other
  seed loop. Budgets (one table, used by every task): the Phase 5 fast tests together run in at most 60 s;
  the whole 600-seed pass in at most 300 s measured (once, not per test); every other test keeps the default
  180 s limit; every pytest run keeps `GGT_TEST_MAX_SECONDS` at its default. A budget breach is fixed by
  making the generator or the runner cheaper, never by lowering a seed count. No loop in a test depends on a
  wall clock; the harness's fake clock and its runaway guard (`MAX_POLLS = 200`) stay on. Child processes
  spawned by a test (the determinism test) run only `build_case` (no trader, no broker, no network) with the
  environment they inherit from `offline.sh` plus `PYTHONHASHSEED` and `PYTHONPATH`.
- **Standalone scripts** (the Task 49 log investigation and the digest recorder; they live outside the
  repository). Run from `$E`, they cannot import `tests.paper...` unless the repository root is on the
  import path, and they run outside pytest, so the autouse connection guard is not there. Every such script
  (1) is started as `./offline.sh env PYTHONPATH=/home/flynn/ggTrader-refactor PYTHONHASHSEED=<n>
  GGT_RUN_MEM_MAX=3G GGT_RUN_MAX_SECONDS=1800 scripts/run_capped.sh .venv/bin/python <script>` (controls after
  `offline.sh`; one process per command, never two at once), (2) checks at its top that `ggTrader.__file__`
  and `tests.paper.equivalence.runner.__file__` are under the worktree it was told to use, and (3) wraps the
  work of **each process** in `tests.netguard.blocked_connections()` (the guard does not cover child
  processes, so a script spawns none) and exits non-zero if the guard recorded an attempt. Its own sha256 is
  written next to it.
- **Two sets of pins, with different rules (Task 51 records them).**
  - **Immutable behavioral set**, hash-recorded in `$E/p5-behavior.sha256` at Task 51: the 103 goldens in
    `tests/paper/golden/trader_transcripts/`, `tests/paper/transcript/harness.py`,
    `tests/paper/transcript/scenarios.py`, `tests/paper/test_trader_transcripts.py`,
    `tests/paper/golden/persist_sql.json`, `tests/paper/test_persist_sql_golden.py`,
    `tests/paper/test_phase4_pins.py`, `tests/lab/test_fetch_stock_ohlcv_chain.py`,
    `src/ggTrader/paper/persist.py`, and the files Tasks 47-50 add that describe behavior and do not mention
    the frozen copy: `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py`,
    `tests/paper/equivalence/run_with.py` (the `run_with` helper; Task 47), `tests/paper/test_run_with.py`,
    `tests/paper/equivalence/__init__.py`,
    `tests/paper/equivalence/generator.py`, `witnesses.py`, `runner.py`, `tests/paper/test_run_digests.py`,
    `tests/paper/golden/run_equivalence_digests.json` and `tests/paper/test_seam_contract.py`. Phase 5 never
    regenerates a golden or a digest (no regeneration switch exists in any new test; a digest is recorded
    once, by a script that lives outside the repository, and is never re-recorded silently). A change to one
    of these needs a written amendment note (authority below) and a new hash record line.
    **Two lists are kept apart, because a git diff against `BASE5` cannot describe a file that did not exist
    then:** the **pre-existing** immutable paths (every file above that exists at `73c35417`: the 103 goldens,
    `harness.py`, `scenarios.py`, `test_trader_transcripts.py`, `persist_sql.json`,
    `test_persist_sql_golden.py`, `test_phase4_pins.py`, `tests/lab/test_fetch_stock_ohlcv_chain.py`,
    `persist.py`) are checked with `git diff --stat 73c35417 -- <paths>` printing nothing; the **new**
    immutable files (everything Tasks 47-50 add above) are checked with `sha256sum -c
    $E/p5-behavior.sha256`, recorded at Task 51. A gate runs both and no other substitute.
  - **Mutable structural set**, reviewed per task and **not** hash-recorded:
    `tests/paper/test_run_structure.py` (the ratchet constants), `tests/paper/test_run_differential.py`,
    `tests/paper/equivalence/impl.py`, `tests/paper/test_run_legacy_frozen.py` and
    `tests/paper/golden/run_legacy.sha256`. **`impl.py` holds exactly one assignment, `IMPLS`** (plus its
    docstring and the `Literal` alias); it holds no function, so a mutable file never defines behavior. Tasks
    57-63 edit the ratchet; Task 74 deletes the differential, the frozen-copy check and its hash, and changes
    the one `IMPLS` line to `("new",)`, and checks that `git diff` of `impl.py` is exactly that line.
    Nothing in the gates requires a mutable file after its retirement task (see "The task gate").
  - **Gate tools are pinned too** (`$E/p5-tools.sha256`, recorded in Task 46 Step 5, checked by `sha256sum -c`
    and by `git diff --stat 73c35417 -- <repository-resident tools>` at every task gate and phase gate): `scripts/run_tests.sh`,
    `scripts/run_capped.sh`, `scripts/_heavy_job.sh`, `scripts/compare_test_outcomes.py`,
    `scripts/compare_evidence_json.py`, `scripts/verify_pins.py`, `scripts/p5_wfo_validation.py`,
    `scripts/p7_live_golden.py`, the repository-root `conftest.py`, `tests/paper/conftest.py`,
    `tests/tiers.py`, `tests/netguard.py`, `offline.sh`, `.github/workflows/ci.yml`, `pyproject.toml`,
    and (outside the repository) the four `$E/p5/gate_*.sh` wrappers and `$E/p5/pins-*.txt`. A change to any
    of them is an amendment, not a task edit; no Phase 5 task edits one.
  - **Amendment authority (the family rule applies to amendments and to verdicts).** The pins record one
    file `$E/p5/amendments.md`. An amendment to a **Claude-built** file (Tasks 47, 48, 50 and their
    helpers, Task 76) is approved by **Sol**; to a **Codex-built** file (Task 49 and everything Terra builds)
    by **Opus**; to a file of mixed or unrecorded authorship (the Phase 3-4 harness, `test_phase4_pins.py`) by
    **both**, each reading the whole amendment. **Opus's verdicts are evidence-based judgments**
    (CLAUDE.md: trust and verdict calls, audit conclusions, protocol decisions, final synthesis) made on gate
    results, mutation matrices and Codex-built code; **Opus does not certify a Claude-built test**, and no
    verdict of this plan lets it. The family that wrote a fix never approves it.
- **The per-run settings snapshot is preserved.** `run()` builds `TraderConfig.from_env(...)` once, at its
  first line, and stores it as `self._cfg` (Phase 4, Deviations 3 and 4). `RunContext.config` is that
  object. No new module reads the environment or calls `cash_sweep.sweep_enabled`, `sweep_symbol`,
  `reserve_pct`, `min_clip_usd`, `buy_trigger_pct`, `catastrophe_stop.catastrophe_stop_enabled`,
  `catastrophe_stop_pct` or `untradable.configured_untradable` (an AST test, Task 50). A malformed float
  keeps its fail time: the reserve ends the run where it did, the other two sweep floats fail soft inside
  the sweep-buy `try`, and the stop threshold error is raised before the lock.
- **Transaction mode and SQL do not change.** `git diff 73c35417 -- src/ggTrader/paper/persist.py
  src/ggTrader/paper/ledger.py` is empty at every gate before Task 75 (`ledger.py` changes only in Task 75).
- **Log records are part of the contract, pinned at DEBUG.** Moved code logs through
  `logging.getLogger(TRADER_LOGGER_NAME)` (`ggTrader.paper.trader`, a constant added in Task 52), so
  `caplog` filters and `test_swallowed_exceptions_are_logged.py` keep working. The differential and digest
  tests capture the named logger set `ggTrader.paper.trader`, `ggTrader.paper.cash_sweep`,
  `ggTrader.paper.untradable` and `ggTrader.paper.notifier` (the four modules of `paper/` that call
  `getLogger` and run during a trader run; `ReadOnlyLedger` logs through the trader's logger) at level
  DEBUG, one tuple per record: `(transcript calls so far, logger name, levelno, record.getMessage(),
  exception type name or None)`, in emission order, so both the records and their position between calls are
  pinned. Any record from another `ggTrader*` logger fails the test. What is nondeterministic (object
  addresses, set order, exception text carrying either) is decided in Task 49 after an investigation, before
  any digest is recorded (Task 49 Step 3a).
- **Layer rules.** The new modules sit in `paper/`; they import nothing from `lab` except what `paper/`
  imports today (`STOCK_BASE_CONFIG`, in `overlay.py`, which Phase 5 does not touch). `tests/utils/test_import_rules.py`
  and `tests/paper/test_paper_boundary.py` stay green without changes.
- **Shared constants and helpers have a non-circular home; no new module imports `trader`.** `trader.py`
  imports every new module, so a new module that imports `ggTrader.paper.trader` (or
  `from ggTrader.paper import trader`) would be circular, and a function-local import would hide it.
  What the new modules need from `trader.py` moves **down**, to modules that import nothing from the step
  objects: the constants `MAX_SIGNAL_AGE_DAYS` (4), `STALE_PENDING_DAYS` (5) and `SPLIT_LOOKBACK_DAYS` (14) go
  to `constants.py` beside `TRADER_LOGGER_NAME`, and the two helpers `as_date(value)` and
  `account_read_halt(notifier, exc)` go to the new `run_helpers.py` (imports: `constants`, `ports`, `logging`;
  it logs through `logging.getLogger(TRADER_LOGGER_NAME)`). `trader.py` keeps its old names bound to them
  (`_MAX_SIGNAL_AGE_DAYS = constants.MAX_SIGNAL_AGE_DAYS`, `_STALE_PENDING_DAYS`, `_SPLIT_LOOKBACK_DAYS`,
  `_as_date = run_helpers.as_date`, `_account_read_halt = run_helpers.account_read_halt`, kept for
  `run_paper_trading`, for `persist.py`'s docstring reference and for any patch). Because a patch of a
  `trader` alias would **not** reach the new modules, Task 50 asserts that no test and no harness file
  patches or assigns `ggTrader.paper.trader.` + one of those five names (today none does). Everything
  else the new modules need is either already outside `trader.py` (`gates.py`: `SignalGateError`,
  `late_run_reason`; `ports.py`; `models.py`; `config.py`; `risk.py`; `cash_sweep`; `catastrophe_stop`;
  `untradable`; `split_check`; `dividend_check`; `ledger.py`) or **injected**: the signal-generation and clock
  functions through `RunHooks`, the private methods through `TraderSeams`, and `LIVE_MEMBERSHIP_SPEC`,
  which stays in `trader.py` and is passed in as the `membership` argument. Task 50's scanner fails any new
  module that imports `ggTrader.paper.trader` in any form.
- **Names that must stay importable and patchable** (Task 50 turns this into a test that scans the
  tests for dotted strings): everything Phase 4 lists, plus `ggTrader.paper.trader.find_split_applied_symbols`,
  `catastrophe_stop`, `cash_sweep` (module bindings patched by attribute), `time`, `datetime`, `uuid4`,
  `_today_et`, `_now_pt`, `generate_core_signals`, `refresh_benchmark_tape`, `live_membership_source`,
  `require_live_membership_spec`, the 26 persist names, `AlpacaBroker`, `TelegramNotifier`,
  `DryRunNotifier`, `PaperTrader`, the five moved names of "Shared constants and helpers" (as aliases) and
  the eight private seams above. Extracted code calls
  `catastrophe_stop.<fn>` and `cash_sweep.<fn>` through the module attribute, never `from ... import`.
- **Exceptions keep their types, exactly where they are raised today.** Two facts about today's code, both
  to be measured in Task 48 and not assumed: a `StopIteration` raised by a collaborator **in a plain
  statement of `_run`** (an empty `sleeves` mapping through `next(iter(...))`, an exhausted
  `Mock(side_effect=[...])` on `catastrophe_stop.unrealized_pct` in the unassessable scan or in the per-stop
  call) reaches the caller of `run()` as a `StopIteration`; but the same exhausted mock called **inside
  `catastrophe_stop.find_catastrophe_stops`, whose body is `sorted(<generator expression>)`**, reaches it as
  `RuntimeError("generator raised StopIteration")` with the `StopIteration` as `__cause__` (PEP 479, in the
  collaborator's own frame). A planner that is a generator, behind an adapter that re-raises "a
  `RuntimeError` whose cause is a `StopIteration`", cannot tell its own frame's conversion from the
  collaborator's, and would turn the second case into a `StopIteration`. So **planners are explicit iterator
  classes**: plain classes whose `advance()` is an ordinary method with no `yield`, which hold their cursor
  and counters as attributes and let every exception propagate untouched. The orchestrator loops
  `while (d := planner.advance()) is not None`, never `for`, and `OrderFlow.run`, `PreflightGate.check` and
  every function that calls a collaborator are plain functions or methods. Task 50 forbids `yield`,
  `yield from` and generator expressions in `planning.py`, `order_flow.py`, `preflight.py`, `portfolio.py` and
  `risk_assessor.py`, and allows in `reporter.py` only the generator expressions `_run` has in region R7
  (three at `73c35417`, moved verbatim), so no new frame can convert an exception.

## Review Focus

The inputs and conditions the spec implies and the early tasks do not exercise; each has a test in the task
named.

1. **A planner failing in the middle of a stage.** Today the buy loop evaluates `check_concentration`
   after the previous buy was submitted, so an exception there leaves earlier orders live. Planners are
   therefore lazy and consumed one decision at a time by the orchestrator, never lists computed up front.
   Test: Task 54 `test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed`, and the
   seeded differential over mid-stage raises (Task 49).
2. **The late-run gate and the clock.** A run whose signals take minutes must still be gated by the time
   after the signals. Test: Task 48 `test_hook_call_order_for_a_live_run` (literal event order of hook calls
   against broker calls) and `test_late_run_gate_reads_the_clock_after_signal_generation`. The transcripts
   cannot show this: the harness pins `_today_et` and `_now_pt` as plain lambdas and does not record them.
3. **An environment change in the middle of a run** (a setting edited while the run is in flight). Test:
   Task 50 `test_env_changed_after_the_first_broker_call_is_ignored_by_every_stage`.
4. **Malformed sleeves and signals** (an empty `sleeves` mapping, which ends in `StopIteration` today; a
   buy signal for the sweep symbol; the same symbol bought by two sleeves, which submits two BUYs today).
   Tests: Task 53 `test_empty_sleeves_ends_the_run_with_stop_iteration_after_the_fallback_notice`, Task 54
   `test_a_symbol_bought_by_two_sleeves_is_planned_twice_as_today`. These are recorded as candidate findings
   in the hand-back, not changed.
5. **The lock or the broker failing on order k of n** (a submit that raises after `k-1` orders went out; a
   `lock.verify` failure on the third order; a dry-run executor that is handed a broker that would submit).
   Tests: Task 49 `test_safety_invariants_*` (the generator injects the kth lock failure through its own
   wrapper, because the harness's `FakeRunLock` fails every call or none) and Task 55
   `test_dry_run_executor_never_reaches_submit`.
6. **A position record missing a field the run reads late.** A sell's `market_value` is read only after the
   order was submitted, and never in a dry run. Tests: Task 48
   `test_live_exit_reads_market_value_only_after_the_order_is_submitted` and the dry-run twin.
7. **A planner failing late, after an earlier order went out.** Three today-real shapes: a later catastrophe
   stop whose position record lacks `qty` (`KeyError('qty')` after the first stop was submitted); an exit
   validation that finds a position without `qty` after stop orders went out (the validation reads `qty`
   before any ordinary exit, but after the stops); and a later sleeve's buy whose sleeve is missing from the
   open-position counts (`KeyError` at `sleeve_open_count[universe]` after an earlier sleeve's buy was
   submitted). A planner that is drained into a list before the first submission moves all three raises in
   front of the first order. Tests: Task 48 `test_a_later_stop_without_qty_raises_after_the_first_stop_order_went_out`,
   `test_exit_validation_without_qty_raises_after_the_stop_orders`,
   `test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy`; generator witnesses W16-W18
   (Task 49) and mutation SM16.
8. **`live_membership_source` failing inside the signal `try`.** Its call is an argument of the
   `generate_core_signals(...)` call, so a failure (an unselected or unreadable membership source) is caught by
   the `except Exception` of the signals block: the owner gets `Paper trading failed: signal generation
   error` plus the exception text, then the exception propagates, and `generate_core_signals` is never
   called. A gate that evaluated the source before the `try` would raise with no alert. Tests: Task 48
   `test_a_membership_source_failure_sends_the_signal_generation_error_alert`, Task 57
   `test_membership_source_failure_is_inside_the_signal_try`, mutation SM17.

## Preserved quirks (recorded, not fixed)

Each is pinned by a transcript or a named test, and carried through every task unchanged.
1. The late-run return sits inside `if not self._dry_run:`; a dry run past the cutoff warns, sends a
   "DRY RUN late-run notice" and carries on (`late_run_dry`).
2. A malformed `SWEEP_CASH_RESERVE_PCT` ends the run after the peak was written and after catastrophe-stop
   orders were submitted (`sweep_reserve_invalid_ends_run`).
3. The pending-order alert can say "pending" for an order that reconciliation has since filled.
4. A configured-untradable holding is announced as "broker tradable=false".
5. `_submit_order` called directly on a read-only trader with a hand-injected lock can submit without a
   durable intent (Phase 4 hand-back); an ordinary run never takes that path.
6. A buy signal for the sweep symbol is an ordinary candidate; the same symbol in two sleeves is bought
   twice (Review Focus 4).
7. The hard stops (`peak_unreadable`, `invalid_equity_halt`, `last_equity_missing_halt` and the other
   three) send `🛑 HALTED:` outside any `try`, so a failing notifier raises through them; the two loss-halt
   notices are inside a `try`.
8. **An invalid exit quantity blocks strategy buys and the sweep BUY, but not a sweep SELL.** `all_buys`
   is emptied before the sweep funding estimate, so the estimate is zero, and a reserve-restoring sweep SELL
   still goes out when cash is below the reserve
   (`test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy[...-1000.0-10.0]`: cash 1,000 sells 10 SPY).
9. **An exit that is submitted but not fully filled clears the strategy buys only.** The sweep BUY stays
   enabled (its condition is `sweep_on and not loss_halts and not invalid_exit_symbols`).
10. **A sell that is still working at the poll deadline does not reach the "not fully filled" message.** The
    intent reconcile that follows the sell poll raises `RuntimeError: Unresolved working orders block
    submissions` and the run ends (`sell_partial_fill_working`); only a *terminal* unfilled or partly filled
    sell (canceled, rejected, expired, or canceled after a partial fill) settles, clears the buys and adds the
    error (`sell_partial_fill_canceled`, `sell_rejected_by_broker`,
    `test_exit_must_fill_to_free_a_position_slot`).
11. **A sell's book amount (`strategy_positions[symbol]["market_value"]`) is read after the order was
    submitted** (and after the symbol was booked), never before and never in a dry run. A position record
    without `market_value` therefore raises `KeyError` *after* a live order went out (Task 48 characterizes
    it; the planners must not read the field earlier).
12. **A `StopIteration` from `catastrophe_stop.unrealized_pct` keeps its type in two places and becomes
    `RuntimeError` in the third.** In the unassessable scan and in the per-stop percentage call it reaches
    the caller as `StopIteration`; inside `find_catastrophe_stops` (a `sorted` over a generator expression) it
    reaches the caller as `RuntimeError("generator raised StopIteration")` (Task 48 measures all three).
13. **Late failures leave earlier orders live** (Review Focus 7): a later stop or sleeve whose position or
    count record is incomplete raises after earlier orders were submitted, with no cancellation attempted.

## The nets, by name

| Net | What it proves | Where |
|---|---|---|
| **T** transcripts | 103 scenarios of `PaperTrader.run`: every broker, persist, notifier, signal, membership, tape, lock (`lock.enter`, `lock.verify` with no client id, `lock.exit`) and clock (`clock.time`, `clock.monotonic`, `clock.sleep`) call, arguments deep-copied; the Telegram text, the result and the raise. It does **not** record `_today_et`, `_now_pt` or `uuid4` calls (pinned lambdas; ids show only inside the order and persist calls) | `tests/paper/test_trader_transcripts.py`, 103 goldens (immutable) |
| **L** SQL golden | every SQL text and bind dictionary of the 31 persist functions | `tests/paper/test_persist_sql_golden.py`, `persist_sql.json` (immutable) |
| **K** shape pins | result key order, tuple sites, environment timing, poll fallback | `tests/paper/test_phase4_pins.py` (immutable) |
| **C** chain, **G** live golden, **N** guard, **S** surface, **I** imports, **V** pins | as in Phase 4 | unchanged |
| **R** rules (new, Task 48) | one named test per rule of design 5.1, each also citing the existing tests that pin it, run against `_run` and `_run_legacy`; hook-order events; the characterizations the transcripts miss | `tests/paper/test_run_rules.py` (immutable) |
| **D** digests (new, Task 49) | the new run of 100 seeds (fast) and 600 seeds (slow), each seed run once per session and summarized (digest, witnesses reached, safety violations), against recorded digests of transcript plus log records taken from the untouched code (recorded at the Task 49 commit that holds the runner); safety invariants and witnesses over the same cached pass | `tests/paper/test_run_digests.py`, `equivalence/` (including `run_with.py`), `golden/run_equivalence_digests.json` (immutable) |
| **E** differential (new, Task 47/49) | new run equals old run for 100 seeds (transcript plus log records). **Blind to helpers the two share** (see "Standard mutations"). Needs `run_with` from the immutable `equivalence/run_with.py`; the mutable `impl.py` only says which implementations exist | `tests/paper/test_run_differential.py` (mutable; deleted in Task 74) |
| **Q** seams and structure (new, Task 50) | legacy patch seams still intercept; imports and environment discipline of new modules; `InMemoryLedger` out of production code; ratchet on `_run` | `tests/paper/test_seam_contract.py` (immutable), `tests/paper/test_run_structure.py` (mutable) |
| **F** frozen copy (new, Task 47) | `_run_legacy` source hash | `tests/paper/test_run_legacy_frozen.py` (mutable; deleted in Task 74) |
| **O** oracle, **P** outcomes | as in Phase 0-2; baselines under `$E` | `gate_*.sh` in `$E/p5/` |

`$E` is `/home/flynn/ggTrader-refactor-evidence`. `T` below is shorthand for
`./offline.sh env GGT_TEST_MEM_MAX=3G scripts/run_tests.sh`.

**Why a frozen copy and digests both.** The copy (`_run_legacy`) localizes a divergence to a step: the two
transcripts are diffed. It cannot outlive Task 74, and **it is not independent of the code it checks**: it
calls the private methods (`_submit_order`, `_poll_orders`, `_reconcile_*`, `_missed_exits`,
`_compute_split_corrections`, `_accrue_dividends`) and the ledger wrapper that the new code shares, and from
Task 55 those are forwards into the new executors. A mutation in a shared helper moves both runs the same
way and the differential passes. The 600 recorded digests are taken from the untouched code in Task 49,
before any extraction, so they guard shared helpers too and keep guarding after the copy is deleted; the
transcripts and targeted unit tests do the rest. Each mutation below says which nets must catch it.

## Target structure and file ownership

New modules under `src/ggTrader/paper/` (each module docstring lists its public names, as Phase 4 did):

| Module | Holds | Built in | Cut from `trader.py` (anchors at `73c35417`) |
|---|---|---|---|
| `seams.py` | `RunHooks`, `TraderSeams` | 52 | none (new) |
| `models.py` (edit) | `IntentKind`, `OrderIntent`, `Skip`, `InvalidExit`, `Decision`, `StrategySignals`, `ExecutionBook`, `RiskVerdict`, `RunContext`, `FlowState` | 52 | none |
| `planning.py` | `Planner`, `strategy_signals`, `assess_stops`, `plan_stops`, `validate_exits`, `plan_exits`, `plan_sweep_sell`, `plan_buys`, `plan_sweep_buy` | 53, 54 | decisions inside 1053-1397 |
| `execution.py` | `Executor` protocol, `OrderExecutor`, `DryRunExecutor` | 55, 60-62 | `_submit_order` 193, `_reconcile_order_intents` 224, `_flag_if_stale` 333, `_reconcile_pending_orders` 361, `_poll_orders` 428 |
| `corporate_actions.py` | `CorporateActions`, `missed_exits` | 56 | `_missed_exits` 305, `_compute_split_corrections` 483, `_accrue_dividends` 607 |
| `preflight.py` | `PreflightGate`, `PreflightResult`, `Halted` | 57 | `_run` 744-855 |
| `portfolio.py` | `PortfolioView` | 58 | `_run` 857-953 |
| `risk_assessor.py` | `RiskAssessor` | 59 | `_run` 955-1034 |
| `order_flow.py` | `OrderFlow`, `FlowOutcome` (the staged plan-then-execute glue) | 60-62 | `_run` 1048-1397 |
| `reporter.py` | `RunReporter` | 63 | `_run` 1036-1046, 1399-1553 |
| `constants.py` (edit) | `TRADER_LOGGER_NAME`, `MAX_SIGNAL_AGE_DAYS`, `STALE_PENDING_DAYS`, `SPLIT_LOOKBACK_DAYS` | 52 | module constants 83, 89, 113 (trader keeps aliases) |
| `run_helpers.py` | `as_date`, `account_read_halt` (non-circular home; see Global Constraints) | 52 | `_as_date` 126, `_account_read_halt` 131 (trader keeps aliases) |

Rows of design 5.1 not in the table: `TraderConfig` and `RunResult` exist since Phase 4 (`config.py`,
`models.py`) and are only consumed here; `RunContext` is in `models.py`. `ledger.py` is not in the table:
Phase 5 does not edit it before Task 75.

`_run` regions (re-grep each quoted anchor): R1 schema init and reconcile 732-742; R2 preflight 744-855
(clock gate 744, signals try 761, sleeve log loop 774, late gate 791, tape refresh 799, fallback notice 810,
signal derivation 816, freshness gate 832); R3 portfolio 857-953; R4 risk 955-1034; R5 prior snapshot
1036-1046; R6a stops, exit validation, buy limits 1048-1181; R6b sweep sell, sells, sell settlement
1183-1301; R6c buys, settlement, sweep buy 1302-1397; R7 reporting 1399-1553.

**File ownership** (a task edits only the files in its own Files list; nothing else changes):

| Path | Edited by |
|---|---|
| `src/ggTrader/paper/trader.py` | 47, 52, 55-63, 74, 75, 76 |
| `src/ggTrader/paper/models.py`, `constants.py`, `seams.py`, `run_helpers.py` | 52 (`models.py` also 53-54 only for a field a planner needs, reported) |
| `src/ggTrader/paper/ledger.py` | 75 only (delete `ModuleLedger`) |
| `planning.py` | 53, 54 (then only by 60-62 for a wiring fix, reported) |
| `execution.py` | 55, 60, 61, 62 |
| `corporate_actions.py` / `preflight.py` / `portfolio.py` / `risk_assessor.py` / `reporter.py` | 56 / 57 / 58 / 59 / 63 |
| `order_flow.py` | 60, 61, 62 |
| `tests/paper/test_run_legacy_frozen.py`, `tests/paper/golden/run_legacy.sha256`, `tests/paper/equivalence/impl.py` (only `IMPLS`), `tests/paper/test_run_differential.py` (first version) | 47 (the differential edited by 49; all deleted or reduced in 74) |
| `tests/paper/equivalence/__init__.py`, `tests/paper/equivalence/run_with.py`, `tests/paper/test_run_with.py` | 47 (immutable from Task 51) |
| `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py` | 48 |
| `tests/paper/equivalence/generator.py`, `witnesses.py`, `runner.py`, `tests/paper/test_run_digests.py`, `tests/paper/golden/run_equivalence_digests.json`, `tests/paper/test_run_differential.py` (differential cases) | 49 |
| `tests/paper/test_seam_contract.py`, `tests/paper/test_run_structure.py` | 50, then the ratchet constant by 57-63, 74 |
| `tests/paper/test_<module>.py` (one per new module) | the task that builds the module |
| `tests/paper/test_trader.py`, `tests/paper/support.py` | 65-73 |
| `AGENTS.md` (`CLAUDE.md` is a symlink to it) | 73 (the persist-stub guidance) |
| `tests/paper/transcript/` harness files, `tests/paper/test_phase4_pins.py`, `tests/paper/test_trader_ledger.py`, `tests/paper/test_ledger.py` | 75 only, and only with the owner's approval |
| `docs/architecture.md`, `docs/changelog.md`, `ruff.toml` | 76 |

## Interfaces (signatures only; the bodies are the builder's)

```python
# paper/seams.py -- every field resolves its target in the ggTrader.paper.trader namespace at CALL time
@dataclass(frozen=True)
class RunHooks:
    today_et: Callable[[], date]; now_pt: Callable[[], datetime]; utcnow: Callable[[], datetime]
    new_uuid: Callable[[], UUID]; clock_time: Callable[[], float]; sleep: Callable[[float], None]
    generate_signals: Callable[..., SignalBundle]; membership_source: Callable[[str | None], object]
    refresh_tape: Callable[[], list[str]]
    find_split_applied: Callable[[dict, list, dict, date], tuple[set[str], set[str]]]

@dataclass(frozen=True)
class TraderSeams:   # the private PaperTrader methods, looked up on the instance at call time
    submit_order: Callable[[str, str, float, str | None], str]
    poll_orders: Callable[[list[SubmittedOrder]], dict[str, OrderStatus]]
    reconcile_intents: Callable[[], set[str]];  reconcile_pending: Callable[[], set[str]]
    missed_exits: Callable[[dict, dict[str, str], set[str]], list[str]]
    split_corrections: Callable[[dict, date, date], dict[str, float]]
    accrue_dividends: Callable[[dict], dict]

# paper/models.py (additions)
class IntentKind(StrEnum): STOP; EXIT; SWEEP_SELL; BUY; SWEEP_BUY
@dataclass(frozen=True, slots=True)
class OrderIntent:
    kind: IntentKind; side: Literal["BUY", "SELL"]; symbol: str
    size: float                  # qty for a SELL, notional for a BUY: the 2nd argument of _submit_order
    reason: str | None
    book_amount: float | None    # SubmittedOrder.amount when it is known BEFORE the order (sweep sell: the sized
                                 # notional; buy and sweep buy: the notional). None for STOP and EXIT: the FLOW
                                 # reads strategy_positions[symbol]["market_value"] AFTER `executor.submit` returned
                                 # AND after it updated the old counters (stopped / executed_sells / regular_sell_count),
                                 # from the mapping it handed the planner, and never in a dry run (Quirk 11).
    sleeve: str | None = None; unrealized_pct: float | None = None; stop_threshold: float | None = None
@dataclass(frozen=True, slots=True)
class Skip: level: int; message: str          # message already formatted; logged on TRADER_LOGGER_NAME
@dataclass(frozen=True, slots=True)
class InvalidExit: symbol: str; error: str    # the orchestrator appends the error and blocks buys
Decision = OrderIntent | Skip | InvalidExit
@dataclass(frozen=True)
class StrategySignals:
    buys: tuple[tuple[str, str], ...]   # (symbol, sleeve) in the ORDER of `all_buys` today: sleeve by sleeve, each
                                        # sleeve's own order, NOT sorted, NOT deduplicated (the same symbol in two
                                        # sleeves appears twice; Quirk 6). The tuple holds exactly the list's items.
    sells: tuple[str, ...]              # `sorted(set(all_sells))`: the ONLY sorted-unique field
    last_exit: Mapping[str, str]; as_of: str; weights: Mapping[str, float]; scale: float
@dataclass
class ExecutionBook:   # mutable on purpose: the run's accumulator; its lists are returned as they are
    buys: list[str]; sells: list[str]; errors: list[str]
    orders: list[SubmittedOrder]; filled: dict[str, OrderStatus]
@dataclass(frozen=True)
class RiskVerdict:
    hard_stop: str | None; buy_halts: tuple[str, ...]
    def with_halt(self, reason: str) -> RiskVerdict: ...
@dataclass(frozen=True)
class RunContext: today_et: date; config: TraderConfig; pending_symbols: frozenset[str]

# paper/planning.py -- pure; no logging, no broker, no notifier, no ledger, no environment
class Planner(Protocol):
    def advance(self) -> Decision | None: ...    # the next decision, or None when the stage is exhausted
# Every planner is an EXPLICIT ITERATOR CLASS (`StopPlanner`, `ExitValidationPlanner`, `ExitPlanner`,
# `BuyPlanner`): an ordinary class whose `advance()` is a plain method with no `yield`, holding its cursor
# and counters as attributes. Nothing wraps or inspects an exception: whatever a collaborator raises
# (`StopIteration`, `KeyError`, a `RuntimeError` produced by a collaborator's own generator expression)
# leaves `advance()` as it left the collaborator. After the stage is exhausted `advance()` keeps returning
# `None`. The orchestrator never uses `for` on a planner; it loops `while (d := planner.advance()) is not None`.
# The planning functions named below as returning `Planner` are the constructors of these classes
# (`plan_stops(...) -> StopPlanner`), kept as functions so the call sites read as before.
class RiskLimits(Protocol):      # RiskGuard satisfies it; used read-only
    def max_new_positions(self, current_count: int) -> int: ...
    def sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]: ...
    def sleeve_position_notional(self, portfolio_value: float, sleeve_weight: float, scale: float) -> float: ...
    def check_concentration(self, symbol: str, positions: dict[str, dict], portfolio_value: float,
                            prospective_notional: float = 0.0) -> bool: ...
def strategy_signals(bundle: SignalBundle) -> StrategySignals   # PLAIN function: StopIteration passes through
def assess_stops(corrected_positions: Mapping[str, Position]) -> frozenset[str]   # PLAIN; unassessable
def plan_stops(*, assessable: Mapping[str, Position], strategy_positions: Mapping[str, Position],
               corrected_positions: Mapping[str, Position], threshold: float,
               pending_symbols: frozenset[str]) -> Planner
def validate_exits(*, sells: Iterable[str], strategy_positions: Mapping[str, Position],
                   invalid_symbols: frozenset[str]) -> Planner                    # InvalidExit only
def plan_exits(*, sells: Iterable[str], strategy_positions: Mapping[str, Position],
               invalid_symbols: frozenset[str], pending_symbols: frozenset[str]) -> Planner
def plan_sweep_sell(*, cfg: TraderConfig, account: Account, positions: Mapping[str, Position],
                    strategy_positions: Mapping[str, Position], sells: Iterable[str],
                    pending_symbols: frozenset[str], risk: RiskLimits,
                    buys_by_sleeve: Mapping[str, list[str]], notional_by_sleeve: Mapping[str, float],
                    slot_caps: Mapping[str, int], portfolio_value: float) -> OrderIntent | None   # PLAIN
def plan_buys(*, buys_by_sleeve: Mapping[str, list[str]], notional_by_sleeve: Mapping[str, float],
              slot_caps: Mapping[str, int], strategy_positions: Mapping[str, Position],
              corrected_positions: Mapping[str, Position], stopped: frozenset[str],
              pending_symbols: frozenset[str], slots_available: int, max_positions: int,
              risk: RiskLimits, portfolio_value: float, open_by_sleeve: Mapping[str, int]) -> Planner
def plan_sweep_buy(*, cfg: TraderConfig, mid_account: Account) -> OrderIntent | None  # PLAIN; may raise: caller's try

# paper/execution.py
class Executor(Protocol):
    submits_orders: bool            # True on OrderExecutor, False on DryRunExecutor: the flow branches on THIS,
                                    # never on the value `submit` returned (a seam may return None or a mock)
    def recover_intents(self) -> None: ...
    def submit(self, intent: OrderIntent) -> str | None: ...   # the BROKER ORDER ID the seam returned; None in a dry
                                                               # run. It builds no SubmittedOrder and reads no position
    def announce(self, intent: OrderIntent) -> None: ...      # the message sent after the intent is booked
    def poll(self, orders: Sequence[SubmittedOrder]) -> dict[str, OrderStatus]: ...
    def after_sells(self, filled: Mapping[str, OrderStatus]) -> None: ...   # `filled` = what `poll` of the sells returned
    def after_orders(self, orders: Sequence[SubmittedOrder]) -> None: ...   # EVERY order submitted this run, sells included
    def unfilled_exit_errors(self, orders: Sequence[SubmittedOrder], filled: Mapping[str, OrderStatus],
                             broker_positions: Mapping[str, Position]) -> list[str]: ...
    # `broker_positions` is the RAW `positions` mapping of `broker.get_positions()` (today's
    # `positions[symbol]["qty"]` is the expected quantity), NOT `strategy_positions` and not the corrected view.
class OrderExecutor:                       # live; holds the bodies moved in Task 55
    def __init__(self, *, broker: Broker, notifier: Notifier, ledger: PaperLedger, hooks: RunHooks,
                 lock: Callable[[], RunLock | None], read_only: bool, seams: TraderSeams) -> None
    # (a) the RUN-PATH methods of the Executor protocol. They reach the work only through the seams, so a
    #     test assignment or class patch of a private PaperTrader method intercepts the new run exactly as it
    #     intercepts the old one (S7):
    #       recover_intents -> seams.reconcile_intents(); submit -> seams.submit_order(side, symbol, size, reason)
    #       and returns its return value (the order id) untouched; poll -> seams.poll_orders(list(orders));
    #       after_sells -> seams.reconcile_intents() when `filled` is non-empty (today `if filled_orders and not
    #       self._dry_run`); after_orders -> seams.reconcile_intents() when `orders` is non-empty (today
    #       `if pending_orders and not self._dry_run`; `orders` includes the sells); the sweep buy's
    #       reconcile (live branch only, unconditional there) is `recover_intents()`, which is the same seam call.
    # (b) the MOVED BODIES, called only by the PaperTrader forwards (S1) and by tests:
    def submit_order(self, side: str, symbol: str, amount: float, reason: str | None) -> str
    def poll_orders(self, orders: list[SubmittedOrder]) -> dict[str, OrderStatus]
    def reconcile_order_intents(self) -> set[str]
    def reconcile_pending_orders(self) -> set[str]
    def flag_if_stale(self, pending_order: dict, now: datetime) -> None
class DryRunExecutor(OrderExecutor): ...  # overrides only: pending-load failure, recover_intents (no-op),
                                            # submit (None, never reaches the seam or the broker), announce,
                                            # after_sells, after_orders (no-ops), unfilled_exit_errors ([])
                                            # `poll` still goes through seams.poll_orders: today's dry run calls
                                            # `_poll_orders` with an empty list, twice, and tests count those calls

# paper/corporate_actions.py
class CorporateActions:
    def __init__(self, *, broker: Broker, ledger: PaperLedger, hooks: RunHooks) -> None
    def split_corrections(self, positions: dict, split_since: date, today: date) -> dict[str, float]
    def accrue_dividends(self, positions: dict) -> dict
def missed_exits(ledger: PaperLedger, positions: dict, last_exit: dict[str, str], already: set[str]) -> list[str]

# paper/preflight.py -- a returned value, not an exception (Deviation 2)
@dataclass(frozen=True)
class Halted: reason: str; result: RunResult          # reason: "market_closed", "late_run" or "stale_signals"
@dataclass(frozen=True)
class PreflightResult: bundle: SignalBundle; signals: StrategySignals; today_et: date
class PreflightGate:
    def __init__(self, *, broker: Broker, notifier: Notifier, hooks: RunHooks, membership: str | None,
                 read_only: bool, dry_run: bool, skipped_writes: list[str]) -> None
    def check(self) -> PreflightResult | Halted           # the old errors (signal gate, StopIteration) still raise
    # `hooks.membership_source(membership)` is evaluated INSIDE the signal `try`, as the argument of
    # `hooks.generate_signals(persist_tape=..., membership=...)`, so its failure sends the "signal generation
    # error" alert and re-raises, and `generate_signals` is never reached (Review Focus 8).

# paper/portfolio.py
@dataclass(frozen=True)
class PortfolioView:
    account: Account; positions: Mapping[str, Position]; strategy_positions: Mapping[str, Position]
    corrected_positions: Mapping[str, Position]; split_corrections: Mapping[str, float]
    sweep_on: bool; sweep_symbol: str | None; untradable_seen: frozenset[str]
    buy_candidates: tuple[tuple[str, str], ...]; sells: tuple[str, ...]   # sells include missed exits
    dividends: Mapping[str, object]
    @property
    def portfolio_value(self) -> float: ...
    @classmethod
    def load(cls, *, broker: Broker, notifier: Notifier, seams: TraderSeams, ctx: RunContext,
             signals: StrategySignals) -> PortfolioView          # performs the reads in today's order

# paper/risk_assessor.py
class RiskAssessor:
    def __init__(self, *, risk: RiskGuard, ledger: PaperLedger, notifier: Notifier) -> None
    def assess(self, ctx: RunContext, view: PortfolioView) -> RiskVerdict   # sends today's notices, in order

# paper/order_flow.py
@dataclass(frozen=True)
class FlowOutcome: book: ExecutionBook; buy_halts: tuple[str, ...]; invalid_exit_symbols: frozenset[str]
class OrderFlow:
    def __init__(self, *, executor: Executor, broker: Broker, notifier: Notifier, risk: RiskGuard,
                 ctx: RunContext, stop_threshold: float | None) -> None
    def run(self, view: PortfolioView, signals: StrategySignals, verdict: RiskVerdict) -> FlowOutcome

# paper/reporter.py
class RunReporter:
    def __init__(self, *, broker: Broker, notifier: Notifier, ledger: PaperLedger, risk: RiskGuard,
                 skipped_writes: list[str]) -> None
    def previous_value(self, today_et: date) -> float | None
    def finish(self, ctx: RunContext, view: PortfolioView, signals: StrategySignals,
               outcome: FlowOutcome, previous_value: float | None) -> RunResult
```

**The final `_run`** (about 40 lines) is: read `cfg`; init schema (the `try` that only logs); live-only
`seams.reconcile_intents()`; `pending = seams.reconcile_pending()`; preflight (`isinstance(pre, Halted)`
returns `pre.result.as_dict()`); build `RunContext`; `PortfolioView.load`; `RiskAssessor.assess` (a
`hard_stop` returns `RunResult([], [], [reason]).as_dict()`); `RunReporter.previous_value`; `OrderFlow.run`;
`RunReporter.finish`; return `.as_dict()`. `run()` keeps its 22 lines (snapshot, stop-threshold alert, lock).

## Interim flow state and bridge locals (Tasks 57-62)

Between Task 57 and Task 63 `_run` is part step objects, part inline code on **bridge locals**. Each wiring
task ends with the inline remainder fed from the new object by explicit rebinding lines, named below, and
the next task deletes the lines it consumes. Nothing else crosses a cut.

| Cut (task) | The inline remainder still reads | Rebound from |
|---|---|---|
| 57 gate | `blend`, `signals` (dict: `buys`, `sells`, `as_of`), `all_buys`, `weights`, `scale`, `last_exit`, `today_et` | `PreflightResult.bundle`, `.signals`, `.today_et` |
| 58 portfolio | `account`, `positions`, `portfolio_value`, `sweep_on`, `sweep_sym`, `strategy_positions`, `corrected_positions`, `split_corrections`, `untradable_seen`, `all_buys` (after the untradable filter), `signals["sells"]` (with the missed exits), `dividend_info` | `PortfolioView` fields |
| 59 risk | `loss_halts` (a **new list** copied from `verdict.buy_halts`), `all_buys = []` when `loss_halts` is non-empty (line 1022 stays inline), the notices are sent inside `assess` | `RiskVerdict` |
| 60 stage I | the **`FlowState`** below | `OrderFlow` stage I |
| 61 stage II | the same `FlowState`, with the sell settlement added | `OrderFlow` stage II |
| 62 stage III | `FlowState` is final; only reporting remains inline | `OrderFlow` stage III |
| 63 | `FlowOutcome(book, buy_halts, invalid_exit_symbols)` | replaces `FlowState` |

```python
@dataclass
class FlowState:        # mutable, internal to order_flow.py; introduced by Task 60, replaced by FlowOutcome in 63
    strategy_positions: dict[str, Position]     # AFTER the stop removal (a new dict, as today)
    stopped: set[str]                           # catastrophe-stopped symbols this run
    invalid_exit_symbols: set[str]
    all_buys: list[tuple[str, str]]             # emptied by an unassessable stop, an invalid exit or a loss halt
    loss_halts: list[str]                       # the verdict's halts plus an appended unassessable-stop reason
    slot_caps: dict[str, int]; sleeve_open_count: dict[str, int]
    buys_by_sleeve: dict[str, list[str]]        # cleared (not emptied-and-rebuilt) when an exit is not fully filled
    sleeve_notional_by_universe: dict[str, float]
    book: ExecutionBook                         # executed buys and sells, errors, submitted orders, filled statuses
    regular_sell_count: int = 0                 # counts ordinary exits only: the slot arithmetic of rule 2
    buys_attempted: int = 0                     # counts yielded buys, live and dry
```

Stage I (Task 60) fills everything except the last two counters and `book.filled`; stage II (61) adds
`regular_sell_count`, `book.filled` and the `errors` of unfilled exits and may `clear()` `buys_by_sleeve`;
stage III (62) adds `buys_attempted` and the sweep buy's order. `loss_halts` is the same list object the
sweep conditions and, in Task 63, the reporter's `buy_halts + errors` read; `errors` is the book's own list,
mutated by the reporter after the summary, so the result's `loss_halts + errors` is evaluated after the
snapshot step, as today.

## Seam rules (the string-patch contract, in force until the owner retires the seams)

- **S1. Private methods stay as forwards.** After a body moves, `PaperTrader._submit_order`,
  `_poll_orders`, `_reconcile_order_intents`, `_reconcile_pending_orders`, `_flag_if_stale`,
  `_missed_exits`, `_compute_split_corrections` and `_accrue_dividends` are one- or two-line forwards to
  the new class's **moved-body** methods, with the same signature. New components reach them only through
  `TraderSeams`, whose lambdas look the method up on the instance at call time, so an assignment on the
  instance (`trader._submit_order = MagicMock(...)`) or a patch on the class
  (`patch.object(PaperTrader, "_poll_orders", spy)`) wins, as today.
- **S2. Module-level collaborators go through `RunHooks`.** `generate_core_signals`,
  `live_membership_source`, `refresh_benchmark_tape`, `_today_et`, `_now_pt`, `uuid4`, `time`,
  `datetime.now(timezone.utc)` and `find_split_applied_symbols` are named in `trader.py` only; the
  `RunHooks` built in `trader.py` wraps each in a lambda that reads the module global when called. New
  modules call `hooks.<x>()`; an AST test (Task 50) fails if a new module imports `time` or `uuid`, calls
  `.now(`/`.today(`, or imports `signal_runner`.
- **S3. Persist names go through the ledger.** New components take a `PaperLedger`; the trader hands them
  `self._ledger` (a `ModuleLedger` or `ReadOnlyLedger` by default, an `InMemoryLedger` in migrated tests).
- **S4. Late-bound mutable attributes.** Components that need `self._risk`, `self._paper_lock`,
  `self._dry_run`, `self._read_only` or `self._broker` receive them when `_run` starts (the risk guard and
  the executor choice) or as a getter (`lock=lambda: self._paper_lock`), never at construction, because
  tests assign them after `PaperTrader(...)` (`test_trader.py:1965, 2119, 3509, 3864`).
- **S5. Executor choice at run time.** `DryRunExecutor` is used when `cfg.dry_run`, else `OrderExecutor`;
  `_run` decides from the snapshot, not from `__init__`.
- **S6. Shared list identity.** `PaperTrader.skipped_writes` is one list shared with `ReadOnlyLedger`; the
  gate appends `"refresh_benchmark_tape"` to the same object and the result copies it (`list(...)`), as now.
  `RunResult.buys` and `.sells` are `ExecutionBook`'s own lists (Phase 4 `as_dict()` identity).
- **S7. The executors reach `submit`, `poll` and `reconcile` through the seams.** `OrderExecutor.submit` calls
  `seams.submit_order`, `poll` calls `seams.poll_orders`, `recover_intents`, `after_sells` and `after_orders`
  call `seams.reconcile_intents`: the seam lambda finds `trader._submit_order` (a `MagicMock` in `_make_trader`,
  assigned after construction), whose default is the forward of S1, whose body is the moved-body method. The
  names differ at each hop (`submit` / `submit_order` / `_submit_order`), so there is no recursion; Task 55
  tests the whole chain and a recursion guard. `DryRunExecutor.submit` never calls the seam, so a mock
  `_submit_order` is never reached in a dry run, as today. The run makes the same seam calls, in the same
  order and number, in dry and live runs (`_poll_orders([])` twice in a dry run). **Who builds what:**
  `Executor.submit` returns only the order id (a `str`; `None` in a dry run). The **flow** keeps the order of
  today's statements: submit, then the old counters and sets (`stopped`, `executed_sells`,
  `regular_sell_count`, `executed_buys`, `buys_attempted`, `sleeve_open_count`), then **the post-submit
  `strategy_positions[symbol]["market_value"]` read for a stop or an exit**, then `SubmittedOrder(...)` appended
  to the book, then the announcement. In a dry run (`executor.submits_orders` is False) the flow updates the
  same counters and announces, but builds no `SubmittedOrder` and reads no `market_value`. A seam that returns
  `None` or a mock in a live run is still a live order.
- **S8. Exceptions keep their types** (Global Constraints): planners are explicit iterator classes advanced
  with `advance()`, never generators and never iterated with `for`; `strategy_signals`, `assess_stops`,
  `plan_sweep_sell`, `plan_sweep_buy`, `OrderFlow.run` and `PreflightGate.check` are plain functions or
  methods. No adapter, no `try` and no `except` anywhere in the planners touches `StopIteration` or
  `RuntimeError`.

## Deviations from the design

1. **`RunContext` carries `today_et`, `config` and `pending_symbols`, not `now_pt`.** `now_pt` is read at the
   late-run gate through `hooks.now_pt()` (O-P5-2). `today_et` is read once at the freshness gate, as
   today; `_submit_order` still reads `hooks.today_et()` per order (Phase 4 tests and
   `TestRunDateIsSessionDateNotAsOf` patch it between calls). Task 48 pins the literal order of hook calls.
2. **`PreflightGate.check()` returns `PreflightResult | Halted`; it does not raise `RunHalted`** (design 5.1
   says "raises `RunHalted(reason, result)`"). A raised `RunHalted(Exception)` could be swallowed by any
   broad `except Exception` between its raise and its catch, or caught by a test's `pytest.raises(Exception)`;
   a returned value cannot. The reasons and results are the design's. The check also returns the derived
   signals and `today_et`, not the bare `SignalBundle`; the signal derivation (lines 816-830) sits between the
   fallback notice and the freshness loop inside the gate, where it is today. This is a stated deviation from
   the approved wording; the owner sees it in the hand-back.
3. **Two modules the design does not name:** `seams.py` (the late-binding records of S1 and S2) and
   `order_flow.py` (`OrderFlow`, the staged glue). The glue is necessary: the order is plan, execute, plan,
   execute (a stop sell frees a slot before the sweep funding is estimated; the sweep sell is computed from
   the post-stop position set), so the planners cannot return one list. Without a home for it `_run` would
   stay near 200 lines.
4. **Planners are lazy `Planner` objects** (explicit iterator classes) advanced one decision at a time, not
   functions returning lists (Review Focus 1) and not generators (S8). They are still pure.
5. **`ModuleLedger` stays until Task 75** (O-P5-1), and so do the eight private seams (O-P5-4). Without
   Task 75 the phase is incomplete against design section 7, and every report says so.
6. **Test migration scope.** "Migrate the trader tests to the in-memory ledger" means the 239 persist-name
   patches of `test_trader.py`. The 137 `generate_core_signals` patches, the clock patches and the private
   seams stay: they are the hooks and seams above.
7. **Phase 4 Deviations 1, 3, 4, 7 and 11 are carried unchanged**: transaction shape, run-time-only
   `TraderConfig` read at the top of `run()` with stored parse errors, frozen `SubmittedOrder`,
   `ReadOnlyLedger` treating every mutator as a write.

## Risks found in the code, and the check that catches each

| # | Risk | Where | Check |
|---|---|---|---|
| 1 | The harness and about 440 tests patch names in the trader namespace; a moved body that imports the name itself bypasses the patch and runs real code (network, database, clock) | S1, S2 | Task 50 `test_every_patch_target_still_resolves_in_the_trader_namespace` (scans the tests); T through the harness; N |
| 2 | Reading `now_pt` before signal generation weakens the late-run gate | `trader.py:791` | Task 48 event-order test; mutation SM9 |
| 3 | Planning everything, then executing, changes behavior when a stage raises midway or when a stop sell must precede the sweep estimate | `trader.py:1070-1336` | lazy planners (Review Focus 1); D and E over mid-stage raises; SM11 |
| 4 | `compute_sweep_sell_for_funding(...)` evaluates `account["cash"]` before `cfg.reserve_pct()`; a malformed reserve raises after live stop orders went out | `trader.py:1202-1208` | Task 54 keeps keyword-argument order; transcript `sweep_reserve_invalid_ends_run`; a live seed with a stop and a bad reserve |
| 5 | Which notices are inside a `try` and which are not (quirk 7) | R2, R4, R7 | T goldens (`failure_notice_fails`, `loss_halt_notice_fails_with_pending`); D with scripted notifier failures |
| 6 | Log records are not in the transcripts | everywhere | D and E compare the DEBUG record tuples; Task 50 logger test |
| 7 | `ruff --fix` (PostToolUse hook, pre-commit) deletes imports only tests patch, or only `_run_legacy` still needs | `trader.py:13-75` | Task 50 patch-target test; every kept name keeps its `# noqa: F401 - <why>` |
| 8 | `trader._dry_run = X` and `trader._paper_lock = MagicMock()` are assigned after construction | tests | S4, S5; Task 50 `test_dry_run_assigned_after_construction_selects_the_dry_executor` |
| 9 | Set iteration order leaks into messages or the result | stop and untradable code | sorted everywhere today; D runs under two `PYTHONHASHSEED`s |
| 10 | Aliasing: `ExecutionBook.buys` and `.sells` are returned as they are; `skipped_writes` is shared | `trader.py:1547-1553` | Task 63 `test_result_lists_are_the_books_own_lists` |
| 11 | `DryRunExecutor` must never submit; `read_only` must never write | executors | Task 55 test with a broker and a seam whose submit raise `AssertionError`; D invariants (a) and (f) |
| 12 | The recorded digests become wrong if the generator or the runner changes | `tests/paper/equivalence/` | Task 49 pins `generator.py`, `witnesses.py` and `runner.py` hashes inside the digest file; no recording switch is committed |
| 13 | A test conversion that keeps its name but loses its teeth | Tasks 65-73 | assertion-count guard (never fewer assertions than at `73c35417`), a reach-checked mutation per batch, Opus review |
| 14 | A new `trader.py` persist import without a stub | AGENTS.md | the autouse guard of Task 73 (raises on any unstubbed persist name and fails at teardown); Task 50 lists the 26 names |
| 15 | Unused `_run_legacy` helpers or stale imports after Task 74 | `trader.py` | Task 74 removes only names `trader.py` no longer uses and that no test or harness patches (Task 50 list) |
| 16 | A planner reads a position field before the order that needs it (`market_value` after submit, never in a dry run) | planners | Task 48 characterizations; Task 53 `Mapping`-subclass lookup counts |
| 17 | A mutation in a helper that old and new runs share passes the differential | executors, `__init__`, ledger | "Standard mutations" table names the nets per site; D and T carry the shared sites |
| 18 | A `StopIteration` inside a planner changes type, or a collaborator's own `RuntimeError` (the one `find_catastrophe_stops` produces from its generator expression) is mistaken for one and rewritten | planners | explicit iterator classes, no adapter (Global Constraints); Task 48 measures all three call sites; Task 53 tests each; Task 50 forbids `yield` and generator expressions in the new modules |
| 19 | The new run bypasses `trader._submit_order = MagicMock(...)` and calls the real ledger and lock | executors | S7; Task 55 chain tests; Task 50 seam-interception tests |
| 20 | A planner drained before the first submission moves a late failure in front of the first order (Review Focus 7) | planners, flow | Task 48 late-failure tests; W16-W18; SM16 |
| 21 | `membership_source` evaluated outside the signal `try` loses the alert (Review Focus 8) | gate | Task 48 and Task 57 tests; SM17 |
| 22 | A gate tool, conftest, tier file or wrapper edited to make a gate pass | `scripts/`, `conftest.py`, `$E/p5/` | `$E/p5-tools.sha256` and `git diff 73c35417` on the repository-resident tools at every gate |
| 23 | A new module imports `trader` (circular) or copies a constant the trader also holds | new modules | "Shared constants and helpers" in Global Constraints; Task 50 scanner |

## Order of work

```
46 setup ─ 47 frozen copy ─ 48 rules ─ 49 equivalence ─ 50 contracts ─ 51 nets gate (phase A)
  ─ 52 seams+records ─ 53 planning I ─ 54 planning II ─ 55 executors move ─ 56 corporate actions move (phase B-C)
  ─ 57 preflight ─ 58 portfolio ─ 59 risk ─ 60 flow I ─ 61 flow II ─ 62 flow III ─ 63 reporter+final _run ─ 64 wiring gate (phase D)
  ─ 65 rig+pilot ─ 66..73 batches M1..M8 ─ 74 delete the frozen copy (phase E)
  ─ 75 retire ModuleLedger [owner-gated, O-P5-1] ─ 76 lint+docs ─ 77 integration gate ─ 78 reviews and hand-back (phase F)
```
Everything is serial (one worktree, one heavy job at a time; this is what Phase 4 did). Inert tasks 52-54
and 56 could be built in parallel by different builders in separate worktrees, but their test runs queue behind
the one lock, so the plan does not count on it. Oracle and `--full` runs are never concurrent with anything.

**`_run` size ratchet** (`tests/paper/test_run_structure.py`, constant `RUN_MAX_LINES`; each wiring task first
lowers it, sees the test fail, then cuts). Starting value 830; after each task set the bound to the measured
line count rounded up to the next 5. Expected: 57 ≈ 740, 58 ≈ 650, 59 ≈ 575, 60 ≈ 470, 61 ≈ 380, 62 ≈ 300,
63 ≤ 45, and Task 74 keeps 45. `run()` keeps ≤ 25 lines.

## Standard mutations (SM)

Applied by the orchestrator, never committed, each reverted with `git checkout -- <file>`; the builder
re-locates the site in whichever module holds it (the mapping lives in `$E/p5/mutation-sites.md`, updated by
every wiring task).

**Step 0 of every mutation matrix is a reach map.** Before a mutation is trusted, the orchestrator applies a
*tripwire* at its site (raise a `BaseException` subclass `SMReached`, which no `except Exception` can swallow)
and runs the candidate tests; the tests that fail with `SMReached` are exactly the tests that exercise the
path, and the real mutation is then required to fail **at least one of those**, not a test chosen by name. The
reach map is written to `$E/p5/mutation-reach-<label>.md`. A test that never reaches a site cannot be cited
for it (a live sell test cannot detect a change to the read-only wrapper, a dry-run test cannot detect a
change to submission).

"Site" says where the mutated line lives: **N** = code only the new run executes (planners, `OrderFlow`,
`PreflightGate`, `RiskAssessor`, `RunReporter`, and, before the wiring tasks, the inline lines of `_run`), so
the differential **must** fail; **S** = a helper the frozen copy also executes (the private methods and, from
Task 55, the executors behind them; `PaperTrader.__init__`; `ReadOnlyLedger`), so the differential is **blind
by design** and the digests, the transcripts and the named test must fail instead.

| SM | Mutation | Site | Must fail (each verified by the reach map) | Differential |
|---|---|---|---|---|
| 1 | drop the `stopped_symbols` guard in the buy loop | N | rule 1 tests; `test_stopped_symbol_not_rebought_by_coincident_buy_signal`; D | must fail |
| 2 | slots use `len(executed_sells)` instead of the regular-sell count | N | rule 2 (three held, one stopped, one ordinary sell: three buys instead of two); `catastrophe_stop_at_position_cap` golden (two buys instead of one) and `test_slot_math_counts_stop_exactly_once_at_position_cap`; D | must fail |
| 3 | drop the clearing of the buys when an exit did not fully fill | N | rule 3 terminal cases; `test_exit_must_fill_to_free_a_position_slot`; `sell_partial_fill_canceled`, `sell_rejected_by_broker`; D | must fail |
| 4a | drop `if invalid_exit_symbols: all_buys = []` | N | rule 4 buys case; `test_invalid_exit_quantity_blocks_buys_but_allows_other_exits`; `strategy_exit_invalid_qty`; D | must fail |
| 4b | drop `not invalid_exit_symbols` from the sweep-buy condition | N | rule 4 sweep-buy case; `test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy[...-50000.0-None]`; `strategy_exit_invalid_qty`; D | must fail |
| 5 | submit the split-corrected quantity | N | rule 5; `test_sell_submits_brokers_uncorrected_qty_not_split_adjusted`; `split_correction_held`; D | must fail |
| 6 | do not wrap the ledger read-only (writes happen) | **S** (`__init__`) | rule 6; `dry_run_read_only` golden; the read-only batch (M6); D. Live-sell tests (`TestSellExits`) cannot reach it | blind |
| 7 | swallow the submit exception and continue | **S** (`submit_order`) | rule 7a; `submit_error` golden; `test_order_failure_stops_later_buys`; `test_run_stops_after_ambiguous_order_failure`; D | blind |
| 8 | drop the intent reconcile after the buys | N | rule 7b (rejected-buy case); `plain_buys_and_sells`, `sweep_buy_after_strategy_orders`; D | must fail |
| 9 | read `now_pt` before generating signals | N | event-order tests only (the harness pins both clocks and does not record them) | not applicable |
| 10 | poll sleep 1.0 to 0.5 | **S** (`poll_orders`) | T (`clock.sleep` calls); `test_poll_orders_reads_the_clock_through_the_hooks_and_sleeps_one_second`; D | blind |
| 11 | compute the sweep sell before the stops are executed | N | rule `test_stops_are_executed_before_the_sweep_sell_is_sized` (Task 48); D (witness W15); T if a golden reaches it | must fail |
| 12 | read `CASH_SWEEP_ENABLED` inside the run instead of the snapshot | N | env test only (the harness pins the environment) | not applicable |
| 13a | the read-only tape-skip log message goes to a different logger | N (`PreflightGate`) | logger test; D (log tuples) | must fail |
| 13b | `ReadOnlyLedger`'s skip message goes to a different logger | **S** | logger test; D | blind |
| 14 | trade alert shows the intended amount for a filled order | N (`RunReporter`) | T; D | must fail |
| 15 | drop `if loss_halts: all_buys = []` (a loss halt no longer clears the buys) | N | `drawdown_halt`, `daily_loss_halt` goldens; `test_loss_limit_allows_protective_exit_but_blocks_new_buys`; D | must fail |
| 16 | **materialize a planner before submission**: evaluate every decision of a stage (the stop loop, the exits, the buys) before its first order is submitted, then submit. In `OrderFlow`: drain `advance()` into a list first. In the inline `_run` (before Task 60): a pre-pass that performs the same reads (`strategy_positions[symbol]["qty"]`, `sleeve_open_count[universe]`) over every candidate ahead of the first `_submit_order` | N (inline stage before Task 60; `OrderFlow` after) | Task 48 `test_a_later_stop_without_qty_raises_after_the_first_stop_order_went_out`, `test_exit_validation_without_qty_raises_after_the_stop_orders`, `test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy` (the raise moves in front of the first order, so the call sequence differs); Task 54 `test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed`; witnesses W16-W18 and D | must fail |
| 17 | evaluate `hooks.membership_source(...)` before the signal `try` (or outside it) | N (`PreflightGate`) | Task 48 `test_a_membership_source_failure_sends_the_signal_generation_error_alert`; Task 57 `test_membership_source_failure_is_inside_the_signal_try`; D (witness W19: a membership-source failure) | must fail |

## The task gate

Every code task, Tasks 52-63, 74 and 75 (shorthand `T` above; every pytest command below is one bounded
run, and the exact spelling of `-m "not slow"` is confirmed in Task 46 Step 3 with the dry-run print):

**Pin lists** (written in Task 46 Step 3, one path per line, read by every gate; a path appears only from the
task that creates it, and **no gate names a deleted or a frozen mutable file after its retirement task**):
- `$E/p5/pins-pre.txt`: `test_run_rules.py`, `test_run_with.py`, `test_run_digests.py`, `test_run_differential.py`,
  `test_run_legacy_frozen.py`, `test_seam_contract.py`, `test_run_structure.py`, `test_persist_sql_golden.py`,
  `test_phase4_pins.py` (all under `tests/paper/`). Used by Tasks 47-73.
- `$E/p5/pins-post.txt`: the same minus `test_run_differential.py` and `test_run_legacy_frozen.py`. Used from
  Task 74 on (Task 74 collects with **both**, see there).
- `$E/p5/units.txt`: the unit-test files of the new modules (`test_seams.py`, `test_planning.py`,
  `test_execution.py`, ...), each appended by the task that creates it.

1. `.venv/bin/ruff check . && .venv/bin/ruff format --check .` clean.
2. **T and the hash records:** `T tests/paper/test_trader_transcripts.py`, no golden regenerated, then
   `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-tools.sha256` (and, from Task 51,
   `$E/p5-behavior.sha256`) prints OK for every line, and the **two-list check of Global Constraints**:
   (a) `git diff --stat 73c35417 -- <pre-existing immutable paths> <repository-resident tools>` prints nothing
   (the paths: `src/ggTrader/paper/persist.py src/ggTrader/paper/ledger.py tests/paper/transcript
   tests/paper/golden/trader_transcripts tests/paper/golden/persist_sql.json
   tests/paper/test_trader_transcripts.py tests/paper/test_persist_sql_golden.py tests/paper/test_phase4_pins.py
   tests/lab/test_fetch_stock_ohlcv_chain.py`, plus the tool list of Global Constraints) and (b) the new
   immutable files are checked **only** by the `$E/p5-behavior.sha256` hashes (a diff against `BASE5` would
   list them as added and is not used for them). Task 75 amends the `ledger.py`, `transcript`,
   `test_phase4_pins.py` and the behavioral-set entries it names, with the owner's approval, recorded in
   `$E/p5/amendments.md`.
3. **The pins:** `T $(cat $E/p5/pins-pre.txt) -m "not slow"` (Tasks 47-73) or `pins-post.txt` (Task 74 on).
   Tasks 55-63 and 74 also run the 600-seed digest test: `T tests/paper/test_run_digests.py -m slow`.
4. **The neighbourhood:** `T tests/paper tests/utils -m "not slow"` (the paper tests run under the guard N;
   `tests/utils` holds S and I). The live golden G and the other `slow` tests run at the phase gates.
5. The task's own unit tests (`$E/p5/units.txt` entries it added, named in the task) pass; its mutations
   fail the nets of the SM table, with the reach map written first.
6. **The oracle O** (Tasks 55-63, 74 and 75 only; label `t<task number>`): `$E/p5/gate_oracles.sh t<NN>`, then
   `PART=oracle $E/p5/gate_compare.sh t<NN>` (the oracle half only; the tests half was step 3 and 4 above),
   exit 0 against `$E/baseline`. The p5 driver runs only with the owner's
   recorded 6G approval (O-P5-3) and at exactly 6G; without it this step is reported "not run" and the task is
   marked "tests half passed, oracle half pending", never gate-complete, and **steps 1-5 and 7 still run and
   still count**. The oracle drivers do not import `paper.trader`, so here it is mainly an import-time and
   shared-module slip detector; the owner may drop it per task by saying so.
7. **Orchestrator re-verification:** it reads the diff with `git diff --color-moved=dimmed-zebra BASE..HEAD`,
   confirms every moved block is a pure move, reruns steps 1-4 itself, and does not pass on a builder's claim.
   A task that moved a body shows the moved block unchanged apart from `self.` to the new class's names.

Where a task below says "the task gate" it means steps 1-5 and 7, plus step 6 when step 6 lists the task. Tasks
47-50 (tests only) run the steps with only the pin files that exist by then; the test-migration tasks 65-73 run
steps 1-4 and 7, with the Task 65-73 protocol replacing step 5.

**The phase gates** (Tasks 51, 64, 77), run by the orchestrator one job at a time: the whole fast core (`T`),
the `--full` tier and `--studies` (`$E/p5/gate_run.sh LABEL`: oracle half, then two junit files, 3G tests; an oracle
refusal for want of the 6G approval is recorded and the tests half still runs), then
`PHASE=pre|post $E/p5/gate_compare.sh LABEL` (oracle compare against `$E/baseline`; test-outcome compare against
`$E/p5-base-core.xml,$E/p5-base-studies.xml` with `$E/p5/must-pass-<pre|post>.txt`), `scripts/verify_pins.py`
with the counts in `.github/workflows/ci.yml`, the mutation matrix, the hash records (the two-list check and
`$E/p5-tools.sha256`), and the reviews of the roles table. A phase gate is **passed** only when both halves
passed; "tests half passed, oracle half pending" is reported as exactly that. **Pre and post retirement:** gates up to Task 73 use `pre`; Task 74 and later use `post`. The two files
are defined exactly, and Task 74 produces `post` from `pre` by an explicit, checked list:
- `must-pass-pre.txt` = `$E/safety-nodes.txt` + the two live-golden node ids + the 277 `test_trader.py`
  node ids + every node of the new behavioral and structural tests as collected with `--collect-only -q`
  (fast and slow) at the Task 51 head, **including** every `[...legacy...]` parametrized node of the rules and
  seam tests and every node of the differential and frozen-copy files.
- `$E/p5/retired-nodes.txt` (Task 74) = **every node that exists in `pre` and no longer exists after Task 74**:
  (i) every node of `test_run_differential.py` and `test_run_legacy_frozen.py` (deleted files), (ii) every
  parametrized node of `test_run_rules.py`, `test_seam_contract.py` and any other file collected over `IMPLS`
  whose id carries the `legacy` implementation, (iii) any further node `--collect-only` shows as gone, each
  with a one-line reason. It never contains a node of the baseline (`p5-base-*.xml`) and never a `new` node.
- `must-pass-post.txt` = `must-pass-pre.txt` minus `retired-nodes.txt` plus the new nodes Task 74 adds (its two
  structure tests). A node of `pre` that is neither retired nor in `post` stops the gate.
Both files contain the 277 `test_trader.py` node ids and the two live-golden nodes. `gate_compare.sh` takes
`PHASE=pre|post` and is run against the matching file only.

## Tasks

### Task 46: Branch, baselines, gate wrappers, hash records and anchors

**Files:** none under `src/` or `tests/`. Evidence under `$E` (not in git).

- [ ] **Step 1: Preconditions.** The owner has approved execution; O-P5-1 to O-P5-5 are answered in writing
  (a default is not an answer; O-P5-3's 6G approval is recorded in `$E/p5/owner-approved-6G.txt`, first line
  exactly `cap=6G`, then the owner's words, or the oracle steps below are reported "not run" and the tests half still runs); the plan commit is the head of `refactor/phase-4`;
  `git status --short` is empty; no heavy job runs.
- [ ] **Step 2: Branch and `BASE5`.** **`BASE5` is `73c35417`** (the code head of Phases 3 and 4 integrated),
  not whatever commit the plan sits on. Check `git rev-parse 73c35417` resolves, and
  `git diff --stat 73c35417 HEAD -- . ':!docs' ':!.superpowers'` is empty (the plan commit `af6b8e89` and any
  later commit change documents only); if it is not empty, stop. `git switch -c refactor/phase-5` in
  `/home/flynn/ggTrader-refactor` (the worktree Phases 3 and 4 used; no second worktree). Record
  `BASE5=73c35417...` (full hash) and the head the branch started from in
  `.superpowers/sdd/2026-11-19-refactor-phase-5/progress.md`. Every `git diff BASE5` in this plan means
  `git diff 73c35417`. Check `.venv/bin/python -c "import ggTrader; print(ggTrader.__file__)"` names this
  worktree, `offline.sh` is present and excluded, the frozen tape's hash matches `data/snapshots/MANIFEST.json`,
  and `get_db_connection_string()` raises `ValueError: Database configuration not found`.
- [ ] **Step 3: Write the four gate wrappers, the pin lists and confirm the command spelling.** The Phase 0-2
  scripts in `$E` have no environment controls and `gate_run.sh` has no Phase 4 copy, so Phase 5 writes all
  four **new** under `$E/p5/` (`chmod +x`), modelled on `$E/p4/gate_*.sh`; the label namespace is `p5-<label>`;
  every control follows `offline.sh`. Keep a `diff` of each against its model in `$E/p5/*.diff`.

  `$E/p5/gate_oracles.sh LABEL` (**the p5 cap is the literal `6G` and cannot be changed from the
  environment; the p7 driver is the literal 3G and 900 s; the refusal is a recorded status, not a stop of the
  whole gate**):
  ```bash
  #!/usr/bin/env bash
  set -u
  E=/home/flynn/ggTrader-refactor-evidence
  L=p5-${1:?label required}
  WT=/home/flynn/ggTrader-refactor
  P5_MEM=6G            # the one approved figure (O-P5-3); the p5 driver peaked at 3,916 MB, so 3G cannot pass
  P7_MEM=3G; P7_SECONDS=900
  cd "$WT" || exit 1
  mkdir "$E/$L" || exit 1
  if [[ "$(head -n1 "$E/p5/owner-approved-6G.txt" 2>/dev/null)" != "cap=$P5_MEM" ]]; then
      echo "p5 oracle not run: no recorded owner approval of cap=$P5_MEM (O-P5-3)" >&2
      printf 'p5 not-run\n' > "$E/$L-oracle-status.txt"
      rc=2
  else
      ./offline.sh env GGT_RUN_MEM_MAX="$P5_MEM" GGT_RUN_MAX_SECONDS=1800 scripts/run_capped.sh \
          .venv/bin/python scripts/p5_wfo_validation.py --runs all --force --out-dir "$E/$L/p5" \
          > "$E/$L-p5.log" 2>&1
      rc=$?
      printf 'p5 exit=%s\n' "$rc" > "$E/$L-oracle-status.txt"
      # a real failure (a cap kill 137, a time stop 143, a busy 75) stops the oracle half here and is reported
      (( rc == 0 )) || exit "$rc"
  fi
  # the p7 driver is bounded and needs no approval, so it runs even when p5 was refused
  ./offline.sh env GGT_RUN_MEM_MAX="$P7_MEM" GGT_RUN_MAX_SECONDS="$P7_SECONDS" scripts/run_capped.sh \
      .venv/bin/python scripts/p7_live_golden.py --out "$E/$L/p7.json" > "$E/$L-p7.log" 2>&1
  rc7=$?
  printf 'p7 exit=%s\n' "$rc7" >> "$E/$L-oracle-status.txt"
  (( rc == 0 )) || exit "$rc"          # 2 = p5 not run for want of the approval
  exit "$rc7"
  ```
  `$E/p5/gate_tests.sh LABEL`: the same shape with `./offline.sh env GGT_TEST_MEM_MAX=3G scripts/run_tests.sh
  --full -rA --junitxml="$E/$L-core.xml" tests > "$E/$L-core.log" 2>&1`, status `core exit=N`, then
  `--studies -rA --junitxml="$E/$L-studies.xml"`, status `studies exit=N` appended. Both pytest runs are at 3G;
  `GGT_TEST_MAX_SECONDS` stays at its default.

  `$E/p5/gate_compare.sh LABEL` with `PART=oracle|tests|all` (default `all`) and `PHASE=pre|post` (needed for
  `tests` and `all`): it checks the status files the part needs (`$'p5 exit=0\np7 exit=0'` for the oracle half;
  `$'core exit=0\nstudies exit=0'` for the tests half; the `-run.txt` marker `run-done` for `all`). With
  `PART=all` it compares the tests half even when the oracle status says `p5 not-run`, and then exits 3 with
  the line `tests half compared; oracle half not run` (never 0); a failed tests half exits 1. It accepts
  `BASELINE_SUFFIX` (default empty) and, only when that is `-t75`, compares against
  `$E/p5-base-{core,studies}-t75.xml`; it refuses any other value, and refuses `-t75` unless
  `$E/p5/retired-baseline-nodes.txt` and `$E/p5/amendments.md` exist (Task 75 Step 5). Then
  `./offline.sh .venv/bin/python scripts/compare_evidence_json.py "$E/baseline" "$E/$L"` (oracle parts) and
  `./offline.sh .venv/bin/python scripts/compare_test_outcomes.py "$E/p5-base-core.xml,$E/p5-base-studies.xml"
  "$E/$L-core.xml,$E/$L-studies.xml" --must-pass-file "$E/p5/must-pass-$PHASE.txt"` (tests parts), writing
  `$E/$L-compare-status.txt` with one exit line each.

  `$E/p5/gate_run.sh LABEL`: validates the label (`^[a-zA-Z0-9_]+$`, and no existing `$E/p5-$LABEL-*` file or
  `$E/p5-$LABEL` directory), runs `gate_oracles.sh` and records its exit status, then **runs `gate_tests.sh`
  whatever the oracle status was** except 75, 137 or 143 (busy, cap kill, time stop: the heavy-job rules say
  stop and report), and writes `oracle=<rc>` and `tests=<rc>` and `run-done` to `$E/p5-$LABEL-run.txt`. An
  oracle refusal (exit 2) therefore leaves a complete tests half and a recorded "oracle not run"; the gate is
  then reported "tests half passed, oracle half pending" and is not a pass.

  Pin lists: write `$E/p5/pins-pre.txt`, `pins-post.txt` and an empty `units.txt` as defined under "The task
  gate". Command spelling: print what the wrapper would run, **with the controls after `offline.sh`**:
  `./offline.sh env GGT_TEST_DRY_RUN=1 GGT_TEST_MEM_MAX=3G scripts/run_tests.sh tests/paper/test_trader_transcripts.py
  -m "not slow"` and the same with `-m slow`, and write the printed command lines to `$E/p5/commands.txt`. (A
  path run passes its arguments to pytest unchanged and applies no tier marker, so `-m "not slow"` is what
  keeps the 600-seed tests out; if the print shows otherwise, the gate uses the printed spelling and the plan's
  "not slow" means "what the print shows keeps `slow` out".)
- [ ] **Step 4: Baselines, built from `73c35417`.** Not copied or renamed from any earlier label. The
  test-outcome baseline is a fresh run at the `BASE5` tree: `$E/p5/gate_tests.sh base` (both exit 0; the live
  golden passes, not skips) yielding `$E/p5-base-core.xml` and `$E/p5-base-studies.xml`. As a cross-check only,
  its node set and statuses equal `$E/merge-core.xml` and `$E/merge-studies.xml` (the recorded merge run); a
  difference stops the task. The oracle comparison target stays `$E/baseline`; BASE5's own oracle evidence may
  reuse `$E/merge/` only if `$E/merge-oracle-status.txt` is `p5 exit=0 / p7 exit=0`, its compare file shows
  exit 0 and its log names `73c35417`; otherwise run `$E/p5/gate_oracles.sh base` (6G, with the approval
  file).
- [ ] **Step 5: Hash records.** `$E/p5-fixed.sha256` = `$E/p4-fixed.sha256` (105 lines: 103 goldens,
  `harness.py`, `persist_sql.json`) plus the hashes of `tests/paper/test_persist_sql_golden.py`,
  `tests/paper/test_phase4_pins.py` and `src/ggTrader/paper/persist.py`; `$E/p5-pins.sha256` =
  `$E/p4-pins.sha256` (3 lines). `sha256sum -c` prints OK for all 111. **`$E/p5-tools.sha256`** records the
  hash of every gate tool listed under "Gate tools are pinned too" in Global Constraints: `scripts/run_tests.sh`,
  `scripts/run_capped.sh`, `scripts/_heavy_job.sh`, `scripts/compare_test_outcomes.py`,
  `scripts/compare_evidence_json.py`, `scripts/verify_pins.py`, `scripts/p5_wfo_validation.py`,
  `scripts/p7_live_golden.py`, `conftest.py`, `tests/paper/conftest.py`, `tests/tiers.py`, `tests/netguard.py`,
  `offline.sh`, `.github/workflows/ci.yml`, `pyproject.toml`, `$E/p5/gate_oracles.sh`, `gate_tests.sh`,
  `gate_compare.sh`, `gate_run.sh`, and (after Step 6 writes them) `$E/p5/pins-pre.txt`, `pins-post.txt`. It is
  built from the tree at `73c35417` (check `git diff --stat 73c35417 -- <repository-resident tools>` prints
  nothing first); `sha256sum -c` prints OK for every line.
- [ ] **Step 6: Node lists.** From `$E/p5-base-core.xml`, with the node-id construction of
  `scripts/compare_test_outcomes.py` (`outcomes()`, loaded by file path in a one-off, not imported as a
  package): `$E/p5/trader-node-ids.txt` = every node of `tests/paper/test_trader.py` (**277 nodes**: 197 test
  functions expanded by their parametrizations; the plan's earlier "197" counted functions); `$E/p5/batches.json`
  = a partition of those 277 ids into the batches M0 (the pilot, Task 65) and M1-M8, each node in exactly one
  batch, the batch sizes summing to 277 (the function counts per batch in Tasks 65-73 are re-measured here);
  `$E/p5/must-pass-pre.txt` = `$E/safety-nodes.txt` + the two live-golden node ids + the 277 ids (Tasks 47-51
  append the nodes of the new behavioral and structural tests as they appear, collected with
  `--collect-only -q` for the fast and for the slow collection, `legacy` parametrized nodes included, as
  defined under "The task gate"); `must-pass-post.txt` is produced by Task 74 (see there).
- [ ] **Step 7: Anchors (Haiku inventory, every figure checked by Sonnet).** Re-measure the region table
  of "Target structure", the ten `RunResult` sites, every `self._dry_run` and `self._read_only` read, the 28
  `self._ledger` calls, the `book_amount` read sites (`["market_value"]` of a sell, after `_submit_order`),
  and the counts quoted here (4,224 lines, 197 test functions, 277 nodes, 439 dotted patches of which 239
  name a persist function, 211 of them inside tests and 28 in the autouse fixtures and `_intent_store`, and
  137 name `generate_core_signals`; 103 goldens; the persist-name string patches outside `test_trader.py`, for
  Task 75), the generator expressions inside `_run` (three, all in R7, with their lines), and the module
  constants and helpers that move (lines 83, 89, 113, 126, 131). Write them to `$E/p5/anchors-trader.txt`.
- [ ] **Step 8:** Report usage to the owner.

### Task 47: Freeze a copy of today's `_run`, and the implementation switch (builder: Sonnet; reviewer: Sol)

**Files:**
- Modify: `src/ggTrader/paper/trader.py` (add `_run_legacy`, a byte-for-byte copy of `_run`, directly under it;
  its docstring says "frozen oracle copy of `_run` at 73c35417; deleted in Task 74; never edited")
- Create: `tests/paper/test_run_legacy_frozen.py`, `tests/paper/golden/run_legacy.sha256`,
  `tests/paper/equivalence/__init__.py` (empty), `tests/paper/equivalence/run_with.py`,
  `tests/paper/equivalence/impl.py`, `tests/paper/test_run_differential.py`

**Interfaces.** The behavior and the one switch are in **different files** so that the file Task 74 edits
never defines behavior:
- `tests/paper/equivalence/run_with.py` (**immutable behavioral set**, hash-recorded at Task 51; builder
  Sonnet, reviewer Sol): `def run_with(impl: str, scenario: Scenario) -> dict`, the harness's `run_scenario`
  with `PaperTrader._run` replaced by `PaperTrader._run_legacy` inside a `patch.object` context for
  `"legacy"` (the harness is not edited). It reads `impl.IMPLS` **at call time** (`from tests.paper.equivalence import impl`) and **raises `ValueError`
  for an `impl` not in `IMPLS`**, so after Task 74 the `legacy` branch is unreachable by construction, not by
  convention. It also exports `Impl = Literal["new", "legacy"]` for typing.
- `tests/paper/equivalence/impl.py` (**mutable set**): its whole code is `IMPLS: tuple[str, ...] = ("new",
  "legacy")` plus a docstring. No function, no import of the trader. Task 74 changes that one line and
  checks that `git diff` of the file is exactly that line.
- `test_run_differential.py` and `test_run_legacy_frozen.py` are **mutable-set** files (Global Constraints):
  Task 74 retires them. Every immutable test imports `run_with` from `run_with.py` and `IMPLS` from `impl.py`.

- [ ] **Step 1: Failing tests.** `test_run_legacy_exists_and_is_a_copy` (`PaperTrader._run_legacy` is a
  function with the same parameters as `_run`), `test_run_legacy_source_is_frozen` (`sha256` of the dedented
  source of `_run_legacy`, with its docstring line removed and the name on its `def` line normalized to
  `_run`, equals the constant in `run_legacy.sha256`, which is the hash of the same normalization of `_run`
  at `BASE5`, computed with `git show 73c35417:src/ggTrader/paper/trader.py`) in `test_run_legacy_frozen.py`;
  `test_both_implementations_give_identical_transcripts_for_every_golden_scenario` (for all 103
  `SCENARIOS`, `run_with("new", s) == run_with("legacy", s)`; trivially true now) in
  `test_run_differential.py`; `test_run_with_rejects_an_implementation_not_in_impls` (patch `IMPLS` to
  `("new",)`: `run_with("legacy", ...)` raises `ValueError`) and `test_impl_py_holds_only_impls` (an AST scan of
  `impl.py` finds one assignment and no function) in `test_run_legacy_frozen.py`'s sibling
  `tests/paper/test_run_with.py` (**immutable**, hash-recorded; add it to the behavioral set list in Task 51).
  FAIL: `_run_legacy` is missing.
- [ ] **Step 2:** Implement: copy the method; compute and record the hash.
- [ ] **Step 3:** The new tests pass; the 103 transcripts (T) are byte-identical; `git diff 73c35417 --
  src/ggTrader/paper/trader.py` shows only the added method.
- [ ] **Step 4: Mutation check (not committed).** Edit one character inside `_run_legacy`: the frozen test
  fails; revert. Edit one character inside `_run`: the 103-scenario switch test still passes only if no
  scenario reaches it; record which (a hit is the point of Task 49).
- [ ] **Step 5:** The task gate. Commit: `test: freeze a copy of PaperTrader._run as the oracle for phase 5`.

### Task 48: Named tests for the seven rules, characterizations, hook order, run against both implementations (builder: Sonnet; reviewer: Sol)

**Files:**
- Create: `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py` (a scenario builder over
  `dataclasses.replace(SCENARIOS[...], ...)`; scenarios and goldens are **not** added: `scenarios.py` is
  immutable). Both are **immutable-behavioral-set** files from Task 51.

Each rule test is parametrized over `IMPLS` (Task 47; the file imports it and asserts
`("legacy" in IMPLS) == hasattr(PaperTrader, "_run_legacy")`, so Task 74's change is a one-line edit of the
mutable `impl.py`, not of this file). It builds its input from a named existing scenario
(`plain_buys_and_sells`, `catastrophe_stop`, `catastrophe_stop_at_position_cap`, `sell_partial_fill_working`,
`sell_partial_fill_canceled`, `sell_rejected_by_broker`, `strategy_exit_invalid_qty`, `split_correction_held`,
`dry_run_read_only`, `submit_error`, `sweep_funding_sell`, `sweep_buy_after_strategy_orders`), cites it and
the existing tests that pin the same behavior in the docstring, and asserts on the transcript's calls, never
on a golden file. Targets are `broker.submit_sell`, `broker.submit_buy`, `persist.create_order_intent`,
`persist.settle_order_intent`, `notifier.send` and `lock.verify` (which carries **no client id**).

- [ ] **Step 1: Failing tests, written against the old code (they fail only because the module is empty).**
  - `test_rule1_catastrophe_stopped_symbol_leaves_at_once_and_is_not_sold_twice_or_rebought`: the stopped
    symbol X also has a strategy sell and a buy signal; exactly one `submit_sell` for X (reason
    `catastrophe_stop`), no `submit_buy` for X, and the stop's freed slot is counted once. Pinned today by
    `TestCatastropheStopEnabled.test_stopped_symbol_not_also_sold_by_coincident_strategy_exit` and
    `test_stopped_symbol_not_rebought_by_coincident_buy_signal` (`test_trader.py:2694, 2708`).
  - `test_rule2_slot_arithmetic_counts_regular_sells_not_all_sells`: **`MAX_POSITIONS=3`, three held (one
    catastrophe-stopped, one with an ordinary exit signal that fully fills, one flat), three buy signals, one
    sleeve: exactly TWO buys are submitted.** The arithmetic, with `max_new_positions(n) = max(0,
    MAX_POSITIONS - n)` where `n` is a **held count** and the result is a **number of free slots**:
    `strategy_positions` already lost the stopped symbol, so `len(strategy_positions) = 2`; one ordinary sell
    (`regular_sell_count = 1`); held count `2 - 1 = 1`; free slots `max_new_positions(1) = 3 - 1 = 2`; two buys.
    Counting the stop sell a second time through `len(executed_sells) = 2` would give a held count of
    `2 - 2 = 0`, hence `max_new_positions(0) = 3` free slots, and permit **three** buys (one over the cap
    once the one flat holding is counted). A second case is the existing at-the-cap shape
    (`catastrophe_stop_at_position_cap`: three held, one stopped, no ordinary sell: `len(strategy_positions) =
    2`, held count `2 - 0 = 2`, one free slot, exactly one buy; the double count would give held count
    `2 - 1 = 1`, two free slots, two buys). Pinned today by
    `test_slot_math_counts_stop_exactly_once_at_position_cap` (`test_trader.py:2770`).
  - `test_rule3_a_terminal_sell_that_does_not_fully_fill_blocks_all_strategy_buys` (cases: canceled after a
    partial fill = `sell_partial_fill_canceled`; rejected = `sell_rejected_by_broker`; expired; `filled` with a
    short quantity): zero `submit_buy`, the error text `SELL X not fully filled (<status>); new strategy buys
    blocked` is in the result's `errors`, the intent reached `settle_order_intent`. With the sweep on, the
    sweep BUY is **still submitted** (Quirk 9). Pinned by `test_exit_must_fill_to_free_a_position_slot`
    (`test_trader.py:3785`). **And, as a separate test,**
    `test_rule3_a_sell_still_working_at_the_poll_deadline_raises_in_reconciliation_instead`
    (`sell_partial_fill_working`): the run raises `RuntimeError: Unresolved working orders block submissions`
    after the "Order recovery/pending orders failed" notice, there is no `settle_order_intent`, no `submit_buy`,
    and the result has no "not fully filled" error because there is no result (Quirk 10).
  - `test_rule4_an_invalid_exit_quantity_blocks_strategy_buys_and_the_sweep_buy`: qty `0`, `None`, NaN,
    negative, on a stop and on a strategy exit; zero `submit_buy` of any kind (no strategy buy, no sweep BUY).
    **`test_rule4_an_invalid_exit_does_not_block_a_reserve_restoring_sweep_sell`** (Quirk 8): with cash below
    the reserve the sweep SELL goes out (cash 1,000: SELL SPY qty 10), with cash 5,000 or 50,000 no sweep order
    goes out. Pinned by `test_invalid_exit_does_not_sell_sweep_to_fund_blocked_buy` (its three parameter sets,
    `test_trader.py:3901`) and `strategy_exit_invalid_qty`.
  - `test_rule5_orders_use_the_brokers_uncorrected_quantity`: with a split correction on record, the
    `submit_sell` quantity equals the broker's raw `qty`; the corrected figures appear only in the daily
    summary and in the concentration check. Pinned by
    `TestSellExits.test_sell_submits_brokers_uncorrected_qty_not_split_adjusted` (`test_trader.py:306`).
  - `test_rule6_a_read_only_run_writes_nothing_and_records_what_it_skipped` (`dry_run_read_only`): no
    `persist.*` call that is a member of `WRITE_METHODS`, `skipped_writes` equals the golden's list,
    `result["skipped_writes"]` too; and `test_rule6_the_same_run_live_does_write` (the control that makes the
    first test detect a missing wrapper).
  - `test_rule7a_an_exception_on_submit_ends_the_run_after_the_intent_is_saved` (`submit_error`): the order of
    calls is `create_order_intent`, `lock.verify`, `submit_*` (raises), then the failure notice naming the stage
    (`Order <client id> failed at submit; investigation required`), and no later `create_order_intent` or
    `submit_*`; the run raises. Pinned by `test_order_failure_stops_later_buys` and
    `test_run_stops_after_ambiguous_order_failure` (`test_trader.py:2411, 3701`).
  - `test_rule7b_a_rejected_order_settles_normally_and_other_exits_stay_eligible` (cases: a rejected exit,
    pinned by `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit` at `test_trader.py:3752`; and a
    rejected **buy**, which settles at the reconcile after the buys): status `rejected` reaches
    `settle_order_intent`, the second exit is still submitted.
- [ ] **Step 2: Characterizations the transcripts and the existing tests miss** (each measured on the untouched
  code and written as the literal it measured; a builder who cannot reproduce the expectation below reports it
  rather than adjusting it):
  - `test_live_exit_reads_market_value_only_after_the_order_is_submitted`: a held position without a
    `market_value` key and an exit signal, live: `broker.submit_sell` is called once and `attach_broker_order`
    ran, **then** the run raises `KeyError('market_value')` (Quirk 11); no poll, no stop or sell notice after the
    submit. The same for a catastrophe stop (the stop's `qty` is read before the submit, its `market_value`
    after). `test_dry_run_exit_never_reads_market_value`: the same inputs in a dry run complete, with the DRY
    RUN sell (and stop) notice and no raise. This is what forbids a planner from reading `market_value` into an
    eager `OrderIntent` field. A position without `qty` raises `KeyError('qty')` in the validation loop before
    any order (`test_missing_qty_raises_before_any_order`).
  - `test_unrealized_pct_call_sequence_is_pinned`: with a recording wrapper on
    `catastrophe_stop.unrealized_pct` (patched as `ggTrader.paper.trader.catastrophe_stop.unrealized_pct`, the
    way `test_trader.py:3810` does), three positions (one healthy, one breaching, one breaching with a pending
    order) produce the literal sequence measured on the untouched code: one call per held position in
    dictionary order (the unassessable scan), one per assessable position (inside `find_catastrophe_stops`), then
    one per breaching symbol that passed the quantity and pending checks, in that order. A second case replaces
    the wrapper with `Mock(side_effect=[...])` of exactly that length: it must pass. **Then three
    "one value short" cases, one per call site, each measured on the untouched code (Quirk 12):** short in the
    unassessable scan: `StopIteration` reaches the caller of `run()`; short in the call **inside
    `find_catastrophe_stops`** (a `sorted` over a generator expression): `RuntimeError` with message
    `generator raised StopIteration` and a `StopIteration` as `__cause__` reaches the caller; short in the
    per-stop percentage call: `StopIteration` reaches the caller. The test asserts the exception type, the
    message and the cause chain literally, so an implementation that "fixes" the middle case into a
    `StopIteration` (or the outer two into `RuntimeError`) fails. A fourth case patches
    `catastrophe_stop.find_catastrophe_stops` itself with a generator function whose body raises
    `StopIteration` (a nested generator frame): the caller gets `RuntimeError` with that cause, in both
    implementations. These four cases are what forbids a planner adapter that rewrites a
    `RuntimeError` whose cause is a `StopIteration`.
  - **Late planner failures after an earlier order (Review Focus 7; the raise must stay behind the first
    order).** `test_a_later_stop_without_qty_raises_after_the_first_stop_order_went_out`: two breaching
    positions, the second (in sorted symbol order) without a `qty` key, live: `submit_sell` was called once for the
    first, its intent was saved and attached, its stop notice was sent, **then** `KeyError('qty')` reaches the
    caller; no second order. (A dry run: the first stop's DRY RUN notice is sent, then the same `KeyError`.)
    `test_exit_validation_without_qty_raises_after_the_stop_orders`: a stop that submits, then an exit signal on a
    held position without `qty`: the run raises `KeyError('qty')` after the stop order, with no ordinary exit
    submitted. `test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy`: two sleeves,
    the second not in `weights`, both with buy candidates: the first sleeve's buy was submitted, then
    `KeyError` for the second sleeve's name reaches the caller. Each is measured on the untouched code and
    the literal call sequence is written down; a builder who cannot reproduce a shape reports it.
  - `test_a_membership_source_failure_sends_the_signal_generation_error_alert` (Review Focus 8): the patched
    `live_membership_source` raises `ValueError("boom")`: the notifier receives `Paper trading failed: signal
    generation error\nboom`, the run raises that `ValueError`, `generate_core_signals` was never called, and no
    broker order, persist write or refresh follows. A `SignalGateError` from `generate_core_signals` keeps its own
    `🛑 Signal gate` notice (existing `signal_gate_error` and `signal_generation_error` goldens). In a read-only dry run, the same.
  - `test_stops_are_executed_before_the_sweep_sell_is_sized` (SM11): a stop and a funding sweep sell in one run
    (`catastrophe_stop` plus `sweep_funding_sell` environments): the stop SELL's `submit_sell` precedes the SPY
    SELL, and the SPY quantity equals the one sized from the post-stop positions.
- [ ] **Step 3: Hook-order tests** (an event-logging copy of the environment builder in
  `tests/paper/test_trader_ledger.py::_trader_environment`, copied, not imported, so that file stays
  untouched; the harness does **not** record `_today_et`, `_now_pt` or `uuid4`, so these events come from this
  copy's own wrappers): `test_hook_call_order_for_a_live_run` pins, as a literal list, the order of
  `membership_source`, `generate_signals`, `now_pt`, `refresh_tape`, `today_et` and `new_uuid` against the
  broker calls for `plain_buys_and_sells` (the builder measures it on the untouched code and writes the
  literal, with the counts: `today_et` once at the freshness gate plus once per submitted order, `new_uuid`
  once per submitted order), and `test_late_run_gate_reads_the_clock_after_signal_generation` (`now_pt` is
  called after `generate_signals` returned, in every implementation).
- [ ] **Step 4: Run:** all pass on both implementations.
- [ ] **Step 5: Mutation matrix (not committed), reach map first.** Apply SM 1-9, 11, 15, 16 and 17 (SM 6, 7
  are shared sites; the others sit in `_run`; SM16 and SM17 are written against the inline stage and the
  inline signal `try`) per the SM table: each fails a rule or characterization test that the reach map showed
  reaches it (two cases for SM4, SM8's buy case for 7b), SM9 fails the event tests, SM16 the three
  late-failure tests, SM17 the membership-alert test. A mutation that passes means the test is rewritten, not
  the mutation.
- [ ] **Step 6:** `-p no:randomly`, in the neighbourhood, the task gate (pins-pre files that exist). Commit:
  `test: name the seven order-flow rules of design 5.1, pin the hook call order and the late reads`.

### Task 49: Seeded differential, recorded digests and safety invariants (builder: Terra; reviewer: Opus)

**Files:**
- Create: `tests/paper/equivalence/generator.py`, `witnesses.py`, `runner.py`, `tests/paper/test_run_digests.py`
  (all **immutable-set**), and, in a **second commit**, `tests/paper/golden/run_equivalence_digests.json`
  (immutable-set); add the differential cases to `tests/paper/test_run_differential.py` (**mutable-set**,
  deleted in Task 74). `run_with` is **not** built here: the differential, the runner's `"legacy"` branch and
  every rule test import it from `equivalence/run_with.py` (Task 47).

**Interfaces:**
```python
def build_case(seed: int) -> Case                            # deterministic from random.Random(seed)
@dataclass(frozen=True)
class Case: scenario: Scenario; lock_fails_on: int | None   # k: the kth lock.verify raises (None: never)
def run_recorded(impl: str, case: Case) -> Recorded          # transcript, log records, skipped writes
@dataclass(frozen=True)
class Recorded:
    transcript: dict
    logs: tuple[tuple[int, str, int, str, str | None], ...]  # (calls so far, logger, levelno, message, exc type)
def digest(rec: Recorded) -> str                             # sha256 of to_json(transcript) + the log tuples
@dataclass(frozen=True)
class SeedSummary:                                           # all a test needs; the transcript is NOT kept
    seed: int; digest: str; witnesses: frozenset[str]
    violations: tuple[str, ...]                              # safety invariants (a)-(g) that failed, with detail
    error: str | None                                        # a harness error (never expected), else None
def summarize_seed(seed: int) -> SeedSummary                 # ONE run of the NEW implementation, under
                                                             # tests.netguard.blocked_connections(); derives the
                                                             # digest, the witnesses and the violations from it
FAST_SEEDS = 100; FULL_SEEDS = 600                           # constants, not read from the environment
WITNESSES: dict[str, Callable[[dict], bool]]                 # one predicate per rule precondition
```

`run_recorded` is **its own copy of the harness's `_run_scenario`** (it imports `_Run`, `FakeRunLock`,
`persist_function_names`, `_pinned_datetime`, `_gate_error_copy`, `normalize` and `to_json` from
`tests.paper.transcript.harness` and never edits the harness) with three additions: (1) DEBUG capture of the
named logger set of Global Constraints, each logger's level, handlers and `propagate` restored afterwards (the
repository's autouse fixture restores only the `ggTrader` logger); a record's first field is the number of
transcript calls made so far, so the interleaving of records and calls is pinned; a record from any other
`ggTrader*` logger fails; (2) **`KthFailLock`**, a `FakeRunLock` subclass that counts `verify()` calls and
raises the scripted exception on call number `lock_fails_on` only, recording it exactly as the harness lock
does and replacing `run.lock` (the harness's `FakeRunLock` takes one `failure` and applies it to every call, and
`Seq` is a persist-reply type the lock never reads, so the harness alone cannot fail the kth verify); (3)
`impl="legacy"` goes through `run_with.run_with` (Task 47), not through code of its own. A submit failing on
order k needs no wrapper: the scenario's `replies["submit_buy"]` is an ordered list whose last entry repeats.

The generator samples (bounded ranges, valid harness inputs only): mode (live 60%, read-only dry 25%,
dry with writes 10%, dry-run-with-late-clock 5%); the sweep on or off; the stop on or off with a threshold in
-0.10..-0.40; `MAX_POSITIONS` in {2, 3, 5}; one or two sleeves with weights and a scale; 0-6 held positions
(quantities normal, 0, negative, NaN, `None`; cost bases that breach or do not breach the stop; a sweep
position); buy and sell signals; `last_exit` entries with `get_last_buy_dates` set so some exits are missed;
untradable symbols from the environment and from the broker; legacy pending orders; split and dividend
evidence; account cash and `last_equity` (halts); the persisted peak; per-symbol status sequences (filled,
partial then filled, working, rejected, canceled, expired); a submit failure on order k; a `lock.verify`
failure on order k (the `Case`); scripted notifier failures; a clock failure; a late `now_pt`; a stale
`as_of`; the overlay fallback flag; **a failing `live_membership_source`**; and **the late-stage planner
failures of Review Focus 7**, each drawn on its own with a fixed probability so each is reached more than
five times in 600 seeds: (L1) two or more breaching positions of which a **later one (in sorted symbol order)
has no `qty` key**, so the run raises `KeyError` after the first stop order went out; (L2) a stop that
submits followed by an **exit signal on a held position without `qty`**, so the exit validation raises after the
stop order; (L3) **two sleeves with the second absent from `weights`** and buy candidates in both, so the
second sleeve's first reachable buy raises `KeyError` after the first sleeve's buy went out. **Buys keep the order and the
duplicates the trader keeps (`StrategySignals.buys`): within one sleeve the generator draws buy symbols
without replacement (unique), and the same symbol appears in two sleeves only in the deliberate duplicate
seeds (W20).** That uniqueness is what makes safety invariant (e) true on the untouched code; without it a
repeated symbol inside one sleeve is bought twice today (the held-check reads a mapping the buys never
update) and invariant (e) would fail for a reason that is not a bug. Read replies not drawn come from
`scenarios.PERSIST_DEFAULTS` (imported, not edited).

- [ ] **Step 1: Failing tests** (`tests/paper/test_run_digests.py` unless named otherwise). **One cache serves
  every 600-seed check** (Global Constraints, "Everything is bounded"): `fast_summaries` and `full_summaries` are
  session-scoped fixtures filling one dict seed -> `SeedSummary` by `summarize_seed`; each fixture wraps its
  work in `blocked_connections()` because a session fixture is set up before the autouse function guard. The
  tests below are assertions over the cache; the ones over `full_summaries` are `slow` and
  `timeout_seconds(600)`.
  `test_generator_is_deterministic` (same seed, same case, in a child process under two `PYTHONHASHSEED`s; the
  child runs only `build_case` and prints a hash of the case, with the environment of Global Constraints),
  `test_every_generated_case_runs_without_a_harness_error` (over `fast_summaries`: no summary has an `error`) and
  `..._600` (slow, `full_summaries`),
  `test_every_witness_is_reached_at_least_once_in_100_seeds` and
  `test_every_witness_is_reached_at_least_five_times_in_600_seeds` (slow) (witnesses W1-W20: stopped symbol
  also buy-signalled; stop plus regular sell with slot-limited buys; live sell not fully filled with buys
  signalled; invalid exit quantity with buys and the sweep on; held split correction on a sold symbol;
  read-only run with a non-empty skip list; submit raised after at least one order; rejected exit then another
  exit; unassessable stop; loss halt with exits; sweep sell and sweep buy in one run; live late-run gate; a
  pending symbol skipped; two sleeves at their caps; a catastrophe stop and a funding sweep sell in one run;
  **W16 a later stop without `qty` raising after the first stop order; W17 an exit validation without `qty`
  raising after a stop order; W18 a later sleeve missing from the open counts raising after an earlier
  sleeve's buy; W19 a `live_membership_source` failure ending in the signal-generation-error alert; W20 the
  same symbol bought by two sleeves**),
  `test_new_run_matches_the_recorded_digests_for_100_seeds` and
  `test_new_run_matches_the_recorded_digests_for_600_seeds` (slow; compare each cached digest with the file),
  `test_digest_file_pins_the_generator_and_the_runner` (the file stores the sha256 of `generator.py`,
  `witnesses.py` and `runner.py`, the production base `73c35417` and the `recorder_commit`; a changed file
  fails),
  `test_safety_invariants_for_100_seeds` and `..._for_600_seeds` (slow; assertions that `violations` is empty
  over the cache; the invariants are computed inside `summarize_seed`): (a) a dry run has no
  `submit_*` and no `create_order_intent`, `attach_broker_order` or `settle_order_intent`; (b) every
  `submit_*` is preceded in order by `create_order_intent` and a `lock.verify` and followed by
  `attach_broker_order` unless it raised (the lock call carries no client id, so the pairing is by position);
  (c) after a raised `submit_*` or a raised `lock.verify` there is no later `create_order_intent` or
  `submit_*`; (d) once a `HALTED buys` notice is sent no `submit_buy` follows; (e) in a **single-sleeve** seed no
  symbol except the sweep symbol is submitted twice (valid because the generator draws unique buys within a
  sleeve), and in a two-sleeve seed a symbol is submitted twice only if it is a buy candidate of both sleeves;
  (f) a read-only run has no `persist.*` call that is a member of `WRITE_METHODS`; (g) every `submit_buy`
  notional is positive and finite, every `submit_sell` quantity is positive and finite. Also
  `test_summarize_seed_keeps_no_transcript` (a `SeedSummary` holds no dict or list beyond the witness set and the
  violation tuple; its `pickle` size is under 2 KB) and `test_the_cache_runs_each_seed_once` (a counting wrapper
  on `run_recorded` sees one call per seed after both fixtures were used). In `test_run_differential.py`:
  `test_new_run_equals_old_run_for_100_seeds` (`run_recorded("new", c) == run_recorded("legacy", c)`, compared as
  whole `Recorded` values; on a failure print the first differing call index and log record).
  Run them: all fail (no generator).
- [ ] **Step 2:** Implement the generator, witnesses and runner; run the tests on the untouched code (`_run`
  equals `_run_legacy`, so the equality test passes trivially; its power is shown in Step 4). Digest-file tests
  fail until Step 7 (the file does not exist yet): that is the only expected failure.
- [ ] **Step 3a: Investigate the log content before any digest exists.** Standalone script
  (`$E/p5/investigate_logs.py`, outside the repository, with the Global Constraints "Standalone scripts"
  rules): run all 600 seeds on the **legacy** implementation with **raw** messages, in **four separate capped
  processes**, one at a time (`PYTHONHASHSEED` 1 then 2, each twice), each started as `./offline.sh env
  PYTHONPATH=/home/flynn/ggTrader-refactor PYTHONHASHSEED=<n> GGT_RUN_MEM_MAX=3G GGT_RUN_MAX_SECONDS=1800
  scripts/run_capped.sh .venv/bin/python $E/p5/investigate_logs.py --out $E/p5/logs-<n>-<k>.json`, each wrapping
  its work in `blocked_connections()`; then diff the four result sets. Classify every differing record: (i) an
  object address or set rendering inside text -> a masking rule in `runner.normalize_log`, listed with the
  record, its source line and the reason; (ii) a set whose order leaks into a message where today's code does not
  sort it -> a candidate finding for the hand-back (not fixed), and a sorted rendering in the digest only; (iii)
  anything else -> stop and report. Write the decisions to `$E/p5/log-normalization.md`; **Opus (reviewing
  Codex's work) approves it before Step 3b.** Decide there, too, whether exception text of caught errors stays
  in the message (it does unless it carried an address).
- [ ] **Step 3b: Provisional digests (uncommitted).** With the generator, witnesses and runner as they stand,
  record a **provisional** `tests/paper/golden/run_equivalence_digests.json` in the working tree (not
  committed) with the recorder of Step 7 pointed at the working tree (`--worktree /home/flynn/ggTrader-refactor`,
  the same offline, `PYTHONPATH`, cap, `blocked_connections()` and serialization wrapper as Step 7), so Step 4 can
  exercise the digest tests.
- [ ] **Step 4: Mutation matrix (not committed), reach map first.** Apply SM 1-8, 10, 11, 13a, 13b, 14, 15, 16
  and 17 at their sites in `_run` and the shared helpers. **N sites** (1-5, 8, 11, 13a, 14, 15, 16, 17): the
  differential test and the digest tests both fail. **S sites** (6, 7, 10, 13b): the digest tests fail and so
  does the named rule or unit test; the differential is *expected to pass* (both runs share the helper), and
  the matrix records that as the reason the digests exist. SM16 must be caught by the witnesses W16-W18 seeds
  and SM17 by W19. Every mutation run is one capped, serialized `T` command: the fast differential and fast
  digest tests for an N site, the slow digest tests (one 600-seed pass, at most 300 s) for an S site or for a
  mutation the fast tests miss; never two at once. A mutation that survives 100 seeds but not 600 is noted, and the generator (not the seed
  count) is improved until none does; revert each. **Any change to the generator, witnesses or runner sends
  the work back to Step 3a's normalization check and Step 3b**; the matrix is rerun on the final candidate.
- [ ] **Step 5: Pre-record review gate (before anything is committed or recorded for good).** On the **final
  candidate** (matrix clean, normalization approved), **Opus** (reviewing Codex's work; fresh context; given
  `generator.py`, `witnesses.py`, `runner.py`, `test_run_digests.py`, `$E/p5/log-normalization.md`, the
  witness list and the matrix result, and the claim "each test fails for the wrong behavior and passes for
  today's; each witness is reachable and counts what it names") reports every finding. Findings are fixed by
  Terra and the review repeats until there are none. The orchestrator then writes the sha256 of the three
  files to `$E/p5/review-gate.txt`. **From here the three files are frozen:** an edit after this point
  restarts at Step 3a. (Task 51's reads are confirmations of these hashes, never the first look.)
- [ ] **Step 6: Commit A (the recorder commit).** Commit everything except the digest file (the provisional
  copy is removed from the tree): `test: seeded equivalence of the trader run and safety invariants`. This commit
  is **not a gate point** (the digest-file tests fail at it by absence). Check
  `git diff 73c35417 A -- src` shows exactly the `_run_legacy` addition (and its docstring) and nothing else, and
  that the frozen-copy hash test passes.
- [ ] **Step 7: Record the digests once, from commit A, not from `BASE5`.** `BASE5` has no runner, so a
  scratch checkout of `73c35417` cannot record anything; the **recorder commit is A**. A one-off script outside the
  repository (`$E/p5/record_digests.py`, its own sha256 written next to it, taking `--worktree` and `--out`) is run
  against a scratch checkout of A (`git worktree add $E/p5/record-A A`; it is removed afterwards) as
  `./offline.sh env PYTHONPATH=$E/p5/record-A:$E/p5/record-A/src PYTHONHASHSEED=<n> GGT_RUN_MEM_MAX=3G
  GGT_RUN_MAX_SECONDS=1800 scripts/run_capped.sh .venv/bin/python $E/p5/record_digests.py --worktree
  $E/p5/record-A --out $E/p5/digests-<n>.json` (capped, scrubbed, one heavy job, **`blocked_connections()` around
  the whole process**). The script first asserts that `ggTrader.__file__` and `tests.paper.equivalence.runner.__file__`
  lie under `--worktree` (if the editable install of the `.venv` wins over `PYTHONPATH` it stops and reports,
  never records from the wrong tree) and that the source of `PaperTrader._run` hashes to `run_legacy.sha256`. It calls
  `summarize_seed(s).digest` for seeds 0-599 and writes `generator_sha256`, `witnesses_sha256`, `runner_sha256`,
  `production_base: 73c35417`, `recorder_commit: <A full hash>`, the seed count and the normalization version, and
  writes `$E/p5/digests-map.sha256` (the sha256 of the canonical JSON of the `digests` object alone, which Task 75
  uses to prove an amendment changed no digest).
  **No recording switch is committed** and the file is never re-recorded to make a test pass: a digest mismatch is
  investigated as a behavior change first, and only an amendment (authority: Opus, Codex-built; reason in
  `$E/p5/amendments.md`) may re-record, again from a scratch checkout of the recorder commit (commit A, or the
  amended generator's commit with the same two prechecks), never from the working tree and never from
  `73c35417`. Run under `PYTHONHASHSEED` 1 and 2: the two files are byte-identical, and their `digests` mapping and
  three stored file hashes equal the provisional file of Step 3b (only `recorder_commit` may differ; any other
  difference stops the task).
- [ ] **Step 8: Commit B** adds `tests/paper/golden/run_equivalence_digests.json`: `test: recorded per-seed
  digests of the untouched trader run`. Run `T tests/paper/test_run_digests.py tests/paper/test_run_differential.py
  -m "not slow"` and `T tests/paper/test_run_digests.py -m slow` and record the wall times in
  `$E/p5/timings.md` against the budgets (the fast Phase 5 tests together at most 60 s; the one 600-seed pass at
  most 300 s). The task gate. Task 51 records the immutable hashes at the head of this commit.

### Task 50: Seam, import and structure contracts (builder: Sonnet; reviewer: Sol)

**Files:** Create `tests/paper/test_seam_contract.py` (**immutable set**) and `tests/paper/test_run_structure.py`
(**mutable set**: it holds the ratchet).

**Every test below passes on today's code. There is no `xfail`, no skip and no deferred failure**: the outcome
comparator treats a new xfail as a breach, and an expected-failure marker is how a gate goes green on a test
that proves nothing. Where a rule concerns modules that do not exist yet, the test discovers them
(`src/ggTrader/paper/*.py` minus the frozen list `EXISTING_MODULES` of the modules present at `BASE5`), so it
checks nothing today and everything from Task 52 on; a **meta-test feeds the scanner synthetic sources** to
prove each rule detects its violation, so a vacuous pass is not a silent one.

- [ ] **Step 1: Tests.**
  - `test_every_patch_target_still_resolves_in_the_trader_namespace`: scans `tests/` and
    `tests/paper/transcript/harness.py` for `"ggTrader.paper.trader.X"` strings and for the names the harness
    patches, asserts each `X` (first path segment) is an attribute of `ggTrader.paper.trader`.
  - `test_legacy_seams_intercept_the_run` (each case drives a minimal scenario, for every implementation in
    `IMPLS`): an instance assignment of `_submit_order`, `_reconcile_pending_orders`,
    `_compute_split_corrections`; a class patch of `_poll_orders`; replacement of `_risk` and of `_risk.cfg`;
    `_dry_run` assigned after construction selects the dry path (and no `submit_*` is called);
    `_paper_lock` assigned a `MagicMock` is the one verified; a string patch of `find_split_applied_symbols`,
    `generate_core_signals`, `_today_et`, `_now_pt`, `uuid4`, `refresh_benchmark_tape` and of one persist name
    after construction each win.
  - `test_moved_code_logs_on_the_trader_logger`: `trader._log.name == "ggTrader.paper.trader"` (the literal), and
    every `logging.getLogger` call in a discovered new module passes `TRADER_LOGGER_NAME` (Task 52 adds the
    constant and its own equality test).
  - `test_new_modules_import_only_what_they_may` (discovered modules): forbidden imports of Global Constraints;
    no `import time`, no `import uuid`, no `.now(`, `.today(`, `os.environ`, `getenv`, no call to the
    environment accessors of "per-run settings snapshot"; no `from ggTrader.paper.cash_sweep import` or
    `from ggTrader.paper.catastrophe_stop import` (module attribute access only); no class deriving from
    `Exception` or `BaseException` in `preflight.py` (Deviation 2); **no import of `ggTrader.paper.trader` in
    any form** (`import ggTrader.paper.trader`, `from ggTrader.paper import trader`, `from ggTrader.paper.trader
    import ...`, a relative form, or `importlib` of that name; function-local imports included: the scan walks
    every `Import`/`ImportFrom` node); **no `yield`, `yield from` or generator expression** in `planning.py`,
    `order_flow.py`, `preflight.py`, `portfolio.py` and `risk_assessor.py`, and in `reporter.py` only the
    generator expressions `_run` has in region R7 (three at `73c35417`, listed in `GENEXP_ALLOWED` with the line
    they were moved from); `ruff`'s unused-import rule cannot catch these, so the scan does.
  - `test_moved_constants_and_helpers_have_one_home`: `constants.py` defines `MAX_SIGNAL_AGE_DAYS`,
    `STALE_PENDING_DAYS`, `SPLIT_LOOKBACK_DAYS` and `run_helpers.py` defines `as_date` and `account_read_halt`
    (a half that checks nothing until Task 52 creates the module and is proved on synthetic sources, like the
    scanner; no `xfail`); `trader.py` binds the five old names to them from Task 52 on; **and no file under `tests/`
    patches or assigns `ggTrader.paper.trader.` followed by one of `_MAX_SIGNAL_AGE_DAYS`,
    `_STALE_PENDING_DAYS`, `_SPLIT_LOOKBACK_DAYS`, `_as_date`, `_account_read_halt`** (a patch of an alias would
    not reach the new modules; today none does).
  - `test_the_new_module_scanner_detects_each_violation`: the same scanner over synthetic sources.
  - `test_env_changed_after_the_first_broker_call_is_ignored_by_every_stage`: a broker whose first
    `get_account` mutates `SWEEP_SYMBOL`, `UNTRADABLE_SYMBOLS`, `CATASTROPHE_STOP_PCT`, `SWEEP_MIN_CLIP_USD`,
    `SWEEP_BUY_TRIGGER_PCT`, `SWEEP_CASH_RESERVE_PCT`, `CASH_SWEEP_ENABLED`; the transcript equals the run
    without the mutation (sweep and stop scenarios).
  - `test_trader_persist_names_unchanged`: the 26 names of `ledger.TRADER_NAMES` are all bound in the trader
    module (Task 75, if approved, amends this with the owner's approval).
  - **`test_in_memory_ledger_is_never_reachable_from_production_code`** (O-P5-5, static only): an AST scan of
    `src/ggTrader/**/*.py`, `scripts/*.py` and `ggt.py` finds the name `InMemoryLedger` only inside
    `src/ggTrader/paper/ledger.py` (its definition and docstring): no import, no reference, and no
    `ledger=` argument in `run_paper_trading`, `cli/main.py` or any production entry point. A runtime refusal
    would change behavior and is not added.
  - `test_run_stays_an_orchestrator` (in `test_run_structure.py`): `RUN_MAX_LINES = 830` today; `run()` at most
    25 lines; both measured with `ast` on `PaperTrader.run` and `PaperTrader._run`. A comment states the
    ratchet protocol.
- [ ] **Step 2:** Run; confirm every test passes on today's code.
- [ ] **Step 3: Mutation check (not committed).** The synthetic-source meta-test also shows the scanner flags
  `from ggTrader.paper import trader` inside a function, a `yield` and a generator expression in `planning.py`.
  Delete the `find_split_applied_symbols` import: patch-target test fails. Make `_run` read `os.environ` once: the env test and the existing `test_trader_py_reads_no_environment`
  fail. Add `from ggTrader.paper.ledger import InMemoryLedger` to `cli/main.py`: the guard test fails. Revert.
- [ ] **Step 4:** The task gate. Commit: `test: contracts for the legacy seams, new-module imports and run size`.

### Task 51: Phase A gate and hash records of the new pins

**Files:** none committed; updates `$E`.

- [ ] **Step 1: Cross-process determinism.** The pin files of Tasks 47-50 under `PYTHONHASHSEED` unset, 1 and 2, all pass.
- [ ] **Step 2: Fast core collects them.** `T --fast --collect-only -q` lists every node of the new files except the
  `slow` digest tests, which `T --full --collect-only -q` lists (two files of node ids, diffed, as in Phase 4 Task 33).
  Append the new node ids to `$E/p5/must-pass-pre.txt`, **every `legacy` parametrized node included**; Task 74's
  `retired-nodes.txt` is built from this list.
- [ ] **Step 3: The whole mutation matrix** (SM 1-17), reach map first, on the untouched-but-for-mutation `_run`
  and helpers: each fails the nets of the table (S sites: the differential is expected to pass, and the matrix says
  so). Record the result in `$E/p5/mutation-matrix-A.md`.
- [ ] **Step 4: Phase gate** (fast core, `--full`, `--studies`, O, P) through `$E/p5/gate_run.sh phaseA` and
  `PHASE=pre $E/p5/gate_compare.sh phaseA`: all pass, the oracle only with the owner's recorded approval (else
  report "oracle not run" and leave the gate open); no `src/` change beyond `_run_legacy`.
- [ ] **Step 5: Family-rule reads (fresh contexts).** **Sol** (reviewing the Claude-built Tasks 47, 48, 50) is
  given only `test_run_rules.py`, `rules_support.py`, `run_with.py`, `test_run_with.py`, `test_seam_contract.py`
  and the claim "each test fails for the wrong behavior and passes for today's"; Sol also reads the Task 49
  files as context but **does not certify them**. **Opus** (reviewing the Codex-built Task 49) **confirms** that
  the sha256 of `generator.py`, `witnesses.py` and `runner.py` equal the ones recorded at the Task 49 pre-record
  review gate (`$E/p5/review-gate.txt`) and reads the digest tests with the same claim; it does not read the
  Claude-built rules tests. Each reports any test that would pass for a wrong reason. Fixes to Claude-built
  files are made before the hash record by Claude and approved by Sol; **a finding in a Task 49 file is an
  amendment** (authority Opus) that restarts Task 49 at Step 3a and re-records from the new recorder commit.
- [ ] **Step 6: Record** `$E/p5-behavior.sha256` (the **new** immutable files of Global Constraints: the Task 47
  to 50 files above, `run_with.py`, `test_run_with.py`, `equivalence/__init__.py` and the digest file; the
  pre-existing immutable paths are covered by `p5-fixed.sha256` and the `git diff` against `BASE5`, not by this
  record) and `$E/p5/mutable-files.txt` (the mutable set, names only). The behavioral files are immutable from
  here. Check the digest file's three stored hashes against the files and `recorder_commit` is an ancestor of
  the head (`git merge-base --is-ancestor`).
- [ ] **Step 7:** Report usage (Phase A done).

### Task 52: Seams, records and the logger constant (builder: Terra; reviewer: Opus)

**Files:**
- Create: `src/ggTrader/paper/seams.py`, `src/ggTrader/paper/run_helpers.py`, `tests/paper/test_seams.py`,
  `tests/paper/test_run_helpers.py`
- Modify: `src/ggTrader/paper/models.py` (the records of "Interfaces", including `FlowState`),
  `src/ggTrader/paper/constants.py` (`TRADER_LOGGER_NAME = "ggTrader.paper.trader"`, `MAX_SIGNAL_AGE_DAYS = 4`,
  `STALE_PENDING_DAYS = 5`, `SPLIT_LOOKBACK_DAYS = 14`: the values `trader.py` holds at 113, 89, 83),
  `src/ggTrader/paper/trader.py` (add `_hooks()` and `_seams()` building the two records with late-binding
  lambdas, call them nowhere yet; rebind `_MAX_SIGNAL_AGE_DAYS`, `_STALE_PENDING_DAYS`, `_SPLIT_LOOKBACK_DAYS`,
  `_as_date` and `_account_read_halt` to the new homes, deleting their bodies; nothing else in `trader.py`
  changes, and `_run` and `_run_legacy` read the same names as before), `tests/paper/test_models.py`.
  `run_helpers.py` holds `as_date(value)` and `account_read_halt(notifier, exc)`, moved unchanged apart from
  the logger (`logging.getLogger(TRADER_LOGGER_NAME)`); its imports are `logging`, `datetime`, `constants` and
  `ports` only, never `trader`. **`ledger.py` is not edited**: its stale docstring
  was already corrected in `612e169f`, and a gate step checks `git diff 73c35417 -- src/ggTrader/paper/ledger.py`
  is empty.

- [ ] **Step 1: Failing tests** (`test_seams.py`, `test_models.py`): `test_hooks_resolve_the_trader_namespace_at_call_time` (patch
  `ggTrader.paper.trader.generate_core_signals` after `_hooks()` was built; calling `hooks.generate_signals` reaches the patch),
  the same for the other nine fields, `test_seams_resolve_instance_attributes_at_call_time` (assign
  `trader._submit_order` after `_seams()`; the seam calls it), `test_default_hooks_read_nothing_at_construction`
  (building them calls no clock, uuid or signal function), `test_trader_logger_constant_names_the_trader_logger`
  (`constants.TRADER_LOGGER_NAME == trader._log.name`), `test_order_intent_is_frozen_and_slotted`,
  `test_order_intent_book_amount_is_optional`, `test_skip_and_invalid_exit_are_frozen`,
  `test_execution_book_lists_are_not_shared_between_books`,
  `test_risk_verdict_with_halt_returns_a_new_verdict_and_keeps_order`, `test_run_context_is_frozen`,
  `test_strategy_signals_fields_are_immutable_views`, `test_flow_state_lists_are_not_shared_between_states`,
  `test_strategy_signals_keeps_buy_order_and_cross_sleeve_duplicates` (a hand-built `StrategySignals` keeps the
  tuple as given; the sorted-unique rule is documented on `sells` only), and in `test_run_helpers.py`:
  `test_account_read_halt_logs_on_the_trader_logger_and_survives_a_failing_notifier` (the two records and the
  message of today's `_account_read_halt`), `test_as_date_accepts_datetime_and_date`,
  `test_trader_aliases_are_the_new_homes` (`trader._as_date is run_helpers.as_date`, the three constants equal).
  FAIL: modules and names are missing.
- [ ] **Step 2:** Implement. `RunHooks.utcnow` is `lambda: datetime.now(timezone.utc)` reading the trader's
  `datetime` name; `clock_time` and `sleep` read the trader's `time`. No behavior of `_run` changes; `_run_legacy` and `_run` still use their old code.
- [ ] **Step 3:** New tests pass; the logger test of Task 50 now discovers `seams.py` and passes.
- [ ] **Step 4:** The task gate (T byte-identical, F unchanged, `ledger.py` and `persist.py` diffs empty). Commit:
  `refactor(paper): late-binding hooks and seams, run records, logger constant`.

### Task 53: Pure planning I: signals, stops, exit validation, exits (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/planning.py` (this task's functions), `tests/paper/test_planning.py`
(append it to `$E/p5/units.txt`).

The bodies are the code now at: signal derivation `trader.py:816-830`; the unassessable scan 1071-1077 and
the stop loop 1095-1138 (the `find_catastrophe_stops` call stays inside `plan_stops`, via the module attribute, and runs
on the **first `advance()`**, after the orchestrator sent the unassessable notices); the exit validation
loop 1147-1160; the regular-sell loop 1249-1268. Log texts become `Skip` messages character for character,
`Skip.level` is `logging.INFO` or `logging.WARNING` as in the source line. **Reads keep their stage:** the stop
planner reads `strategy_positions[symbol]["qty"]` before the order and `corrected_positions[symbol]` for the
percentage (one `catastrophe_stop.unrealized_pct` call per processed stop, after the quantity and pending checks);
an exit planner reads `["qty"]` before the order; **no planner reads `["market_value"]`** (the intent's
`book_amount` is `None`; the flow reads it after the submit, Quirk 11).

- [ ] **Step 1: Failing tests** (plain data, no mocks, no notifier, no broker):
  - `test_strategy_signals_merges_sleeves_in_order_and_sorts_sells` (**buys keep the order of `all_buys` and the
    duplicates across sleeves; only the sells are `sorted(set(...))`**: a symbol in two sleeves appears twice, in
    sleeve order, and an unsorted sleeve stays unsorted), `test_strategy_signals_takes_as_of_from_the_first_sleeve`,
    `test_empty_sleeves_ends_the_run_with_stop_iteration_after_the_fallback_notice` (the plain function raises
    `StopIteration`, as `next(iter({}))` does today; Review Focus 4).
  - `test_assess_stops_flags_none_and_invalid_positions` (a position whose `unrealized_pct` is `None` or raises `ValueError`),
    `test_assess_stops_calls_unrealized_pct_once_per_position_in_dictionary_order`.
  - `test_plan_stops_yields_one_intent_per_breaching_symbol_with_qty_and_reason` (kind `STOP`, `size` is the
    strategy quantity, `book_amount` is `None`, `reason` is `catastrophe_stop.CATASTROPHE_STOP_REASON`,
    `unrealized_pct` and `stop_threshold` carried),
    `test_plan_stops_yields_invalid_exit_for_a_bad_quantity_and_goes_on` (rule 4),
    `test_plan_stops_skips_a_pending_symbol_with_the_old_log_text` (the `Skip` message equals
    `"Skipping catastrophe stop X: order already pending"` and its level is INFO),
    `test_plan_stops_calls_the_catastrophe_module_attribute` (patching `catastrophe_stop.find_catastrophe_stops` changes the result),
    `test_plan_stops_unrealized_pct_call_count_and_order_match_the_old_loop` (the sequence of Task 48's
    `test_unrealized_pct_call_sequence_is_pinned`).
  - `test_validate_exits_yields_invalid_exit_once_per_symbol_and_ignores_pending` (a pending order does not excuse an
    invalid quantity), `test_validate_exits_ignores_unheld_and_already_invalid_symbols`.
  - `test_plan_exits_skips_pending_symbols_with_the_old_log_text`, `test_plan_exits_intent_size_is_the_raw_quantity_and_book_amount_is_none`
    (rule 5, Quirk 11), `test_plan_exits_ignores_invalid_symbols`,
    `test_no_planner_reads_market_value` (a `Mapping` subclass raises on any `["market_value"]` lookup).
  - `test_planners_are_lazy_and_do_not_mutate_their_inputs` (inputs are `MappingProxyType`s and frozen sets; one
    `advance()` does not evaluate the next decision: a `Mapping` subclass counts lookups).
  - **Exception types are untouched (Quirk 12; the three call sites of Task 48):**
    `test_a_stop_iteration_from_the_per_stop_percentage_call_surfaces_as_stop_iteration` (a
    `Mock(side_effect=[...])` on `catastrophe_stop.unrealized_pct`, one value short at the per-stop call:
    `advance()` raises that same `StopIteration` object);
    `test_a_collaborators_own_runtime_error_passes_through_advance_unchanged` (the patched
    `catastrophe_stop.find_catastrophe_stops` is a generator function whose body raises `StopIteration`, and, in
    a second case, the real function with `unrealized_pct` one value short inside its `sorted(...)`: `advance()`
    raises the **`RuntimeError` object with message `generator raised StopIteration` and its `StopIteration`
    `__cause__`**, never a `StopIteration`); `test_a_key_error_passes_through_advance_unchanged` (a later
    position without `qty`); `test_a_normally_finished_planner_returns_none`;
    `test_advance_after_exhaustion_keeps_returning_none`; `test_planners_are_explicit_classes` (no planner
    constructor returns a generator: `inspect.isgenerator` false, no `yield` in the module source).
  FAIL: ImportError.
- [ ] **Step 2:** Implement; planners are **explicit iterator classes** (`StopPlanner`, `ExitValidationPlanner`,
  `ExitPlanner`; Interfaces), each `advance()` a plain method with a cursor attribute, no `yield`, no
  `try`/`except` around a collaborator call; no logging, no notifier, no broker, no ledger.
- [ ] **Step 3:** New tests pass. `_run` is **not** wired yet; the task gate (T byte-identical). Commit:
  `refactor(paper): pure planning for signals, stops and exits`.

### Task 54: Pure planning II: sweep sell, buys, sweep buy (builder: Terra; reviewer: Opus)

**Files:** Modify `src/ggTrader/paper/planning.py`; Modify `tests/paper/test_planning.py`.

Bodies at `trader.py:1171-1181` (limits, which stay in `OrderFlow` and are passed in), 1183-1236 (sweep sell:
the keyword arguments of `compute_sweep_sell_for_funding(...)` keep today's order, so `account["cash"]` is
read before `cfg.reserve_pct()` raises), 1302-1351 (buys), 1367-1396 (sweep-buy sizing). The sweep sell reads
the **sweep position's** `market_value` and `current_price` before the order, as today (it is the sizing input,
not a book amount); its intent's `book_amount` is the sized notional.

- [ ] **Step 1: Failing tests:**
  - `test_plan_sweep_sell_returns_none_when_cash_covers_the_buys`, `test_plan_sweep_sell_sizes_qty_from_price_and_rounds_to_four_places`,
    `test_plan_sweep_sell_returns_none_for_a_zero_price_or_zero_quantity`,
    `test_plan_sweep_sell_counts_only_sells_not_pending_for_the_slot_estimate`,
    `test_plan_sweep_sell_still_restores_the_reserve_when_all_buys_were_cleared` (empty `buys_by_sleeve`, cash below
    the reserve: a SELL; Quirk 8),
    `test_plan_sweep_sell_reraises_the_stored_reserve_error_after_reading_the_account_cash` (the error is
    `ValueError`; a missing `cash` key raises `KeyError` first).
  - `test_plan_buys_follows_held_stopped_pending_slots_sleeve_cap_concentration_in_that_order` (one candidate hitting each
    check in turn; the `Skip` messages equal the old log lines, including `Max positions reached (N), skipping X`),
    `test_plan_buys_breaks_at_the_slot_limit_per_sleeve_and_goes_on_to_the_next_sleeve`,
    `test_plan_buys_counts_each_yielded_intent_toward_slots_and_sleeve_caps`,
    `test_plan_buys_never_rebuys_a_stopped_symbol` (rule 1),
    `test_plan_buys_slot_arithmetic_is_the_callers_slots_available` (rule 2: the planner takes `slots_available`, the orchestrator computes it from regular sells),
    `test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed` (a concentration check that raises on the
    second symbol is not called after one `advance()`; Review Focus 1),
    `test_a_stop_iteration_from_check_concentration_surfaces_as_stop_iteration` (`BuyPlanner` is an explicit
    class too: `advance()` raises the same object; the old `for` loop over a list did not convert either),
    `test_a_symbol_bought_by_two_sleeves_is_planned_twice_as_today`, `test_a_buy_signal_for_the_sweep_symbol_is_an_ordinary_candidate`,
    `test_plan_buys_raises_key_error_for_a_sleeve_missing_from_the_open_counts_at_the_old_point` (`open_by_sleeve[universe]` is read after
    the held, stopped, pending and slot checks, as `sleeve_open_count[universe]` is today).
  - `test_plan_sweep_buy_matches_compute_sweep_buy_for_valid_settings`, `test_plan_sweep_buy_reraises_each_stored_parse_error_in_its_old_order`
    (`reserve_pct`, then `min_clip_usd`, then `buy_trigger_pct`), `test_plan_sweep_buy_returns_none_when_the_action_is_not_a_buy`.
- [ ] **Step 2:** Implement (`BuyPlanner`, an explicit iterator class with the sleeve cursor, the symbol cursor
  and `buys_attempted`-style counters as attributes); **Step 3:** pass; `_run` still unwired. The task gate. Commit:
  `refactor(paper): pure planning for the sweep sell, buys and the sweep buy`.

### Task 55: Move the order bodies into `OrderExecutor` / `DryRunExecutor` (builder: Terra; reviewer: Opus)

**Files:**
- Create: `src/ggTrader/paper/execution.py`, `tests/paper/test_execution.py` (append to `units.txt`)
- Modify: `src/ggTrader/paper/trader.py`: the five methods become forwards (S1); `_run` and `_run_legacy` are untouched

The bodies are `_submit_order` (193), `_reconcile_order_intents` (224), `_flag_if_stale` (333),
`_reconcile_pending_orders` (361), `_poll_orders` (428), moved unchanged except: `self._ledger` is the
injected ledger, `self._notifier` and `self._broker` likewise, `uuid4()` is `hooks.new_uuid()`, `_today_et()` is
`hooks.today_et()`, `time.time()`/`time.sleep()` are `hooks.clock_time()`/`hooks.sleep()`, `datetime.now(timezone.utc)` is
`hooks.utcnow()`, the lock is `self._lock()` read at the moment of the call, and the `self._dry_run` branch of
`_reconcile_pending_orders` becomes the overridable `_pending_load_failed(exc) -> set[str]` (live: send the
notice and raise; dry: warn and return `set()`). `PaperTrader._executor_for_call()` builds
`DryRunExecutor` when `self._dry_run`, else `OrderExecutor`, from the current attributes (S4, S5), with
`seams=self._seams()`. The executor's run-path methods (`submit`, `poll`, `recover_intents`, `after_*`) are
added now and reach the work through the seams (S7); nothing calls them yet. **`submit` returns the
broker order id the seam returned, as a bare value; it does not build a `SubmittedOrder`, update a counter or
read a position** (the flow does, in Tasks 60-62: S7, "Who builds what"). `after_sells(filled)` reconciles when
`filled` is non-empty; `after_orders(orders)` reconciles when `orders` is non-empty and receives **every**
order of the run, sells included; `unfilled_exit_errors(orders, filled, broker_positions)` takes the raw
broker positions (today's `positions[symbol]["qty"]`); `submits_orders` is a class attribute.

Note for the differential (Tasks 49 and 55 on): after this task `_run_legacy` calls the same moved bodies through
the forwards, so the differential no longer checks them; the digests, the transcripts and this task's unit
tests do.

- [ ] **Step 1: Failing tests** (`test_execution.py`, fake ports; each moved method keeps the characterization of its old tests):
  `test_submit_order_saves_the_intent_then_verifies_the_lock_then_submits_then_attaches`,
  `test_submit_order_names_the_failing_stage_in_the_notice_and_reraises` (four stages),
  `test_submit_order_rejects_a_side_other_than_buy_or_sell`, `test_submit_order_reads_the_run_date_per_call_from_the_hook`,
  `test_poll_orders_reads_the_clock_through_the_hooks_and_sleeps_one_second`, `test_poll_orders_of_no_orders_does_not_touch_the_broker`,
  `test_poll_fallback_record_matches_the_old_dictionary` (the `notional` is the amount only for a BUY),
  `test_reconcile_order_intents_has_each_validation_branch` (identity, amount, fill, status, working),
  `test_reconcile_pending_orders_read_only_returns_symbols_without_a_broker_call`,
  `test_reconcile_pending_orders_load_failure_live_notifies_and_raises`, `..._dry_warns_and_returns_empty`,
  `test_flag_if_stale_is_idempotent`,
  `test_executor_uses_the_hooks_not_the_modules` (patching `ggTrader.paper.trader.uuid4` after construction changes the client id),
  **the seam chain (S7):** `test_executor_submit_routes_through_the_instance_seam` (a `MagicMock` assigned to
  `trader._submit_order` after construction is the callee of `OrderExecutor.submit`, with the intent's side,
  symbol, size and reason), `test_executor_poll_routes_through_the_class_patched_seam`,
  `test_executor_reconcile_routes_through_the_instance_seam`, `test_the_default_seam_chain_reaches_the_moved_body_without_recursion`
  (`submit` -> `trader._submit_order` -> forward -> `submit_order`, exactly one body execution),
  **the dry path:** `test_dry_run_executor_never_reaches_submit` (a seam whose `submit_order` raises
  `AssertionError` and a broker whose `submit_buy` and `submit_sell` raise `AssertionError`; `DryRunExecutor.submit`
  returns `None`, and a full dry scenario run through `PaperTrader` with `trader._submit_order = MagicMock()`
  and `trader._dry_run = True` assigned after construction ends with that mock never called),
  `test_dry_run_executor_still_polls_through_the_seam_with_an_empty_list`,
  **the run-path contract of Tasks 60-62:** `test_executor_submit_returns_the_seam_value_untouched` (a seam
  returning `"oid-1"`, `None` or a `MagicMock`: `OrderExecutor.submit` returns exactly that object, and builds
  nothing), `test_executors_declare_submits_orders` (`OrderExecutor.submits_orders is True`,
  `DryRunExecutor.submits_orders is False`), `test_after_orders_reconciles_for_a_list_holding_only_sells` (the
  old `if pending_orders and not self._dry_run`; an empty list does not reconcile), `test_after_sells_reconciles_only_for_a_non_empty_filled_mapping`,
  `test_dry_run_after_hooks_do_nothing`, `test_unfilled_exit_errors_use_the_raw_broker_positions` (a quantity
  that differs between the raw position and a split-corrected one decides the error; sweep sells are skipped; the
  texts equal `SELL X not fully filled (<status>); new strategy buys blocked`, `unknown` for a missing status),
  `test_unfilled_exit_errors_is_empty_in_a_dry_run`.
  FAIL: ImportError.
- [ ] **Step 2:** Implement and cut the five bodies from `trader.py` (their forwards: `return self._executor_for_call().<method>(...)`
  with the same signature and docstring first line); `trader.py` keeps its imports that tests patch (`# noqa: F401 - patch target`).
- [ ] **Step 3:** New tests pass; the 103 transcripts, `test_phase4_pins.py` (it calls `_poll_orders` directly
  and spies on the class) and the whole `tests/paper` neighbourhood stay green with no edit.
- [ ] **Step 4: Mutations (reach map first).** SM 7 and SM 10 at their new sites (shared helpers: the differential
  is expected blind; the transcripts, the digests and the named rule or unit tests fail), and a mutated stage
  order in the moved `submit_order` (`lock` before `intent`) fails `test_rule7a...`, the transcripts and the
  unit test. Revert. (SM 14 is the reporter's and belongs to Task 63.)
- [ ] **Step 5:** The task gate; Opus reviews the moved blocks with `--color-moved`. Commit:
  `refactor(paper): OrderExecutor and DryRunExecutor hold the order bodies`.

### Task 56: Move split, dividend and missed-exit logic into `CorporateActions` (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/corporate_actions.py`, `tests/paper/test_corporate_actions.py` (append to
`units.txt`); Modify `src/ggTrader/paper/trader.py` (three forwards).

Bodies: `_compute_split_corrections` (483-605), `_accrue_dividends` (607-703), `_missed_exits` (305-331), moved
unchanged except the ledger and broker are injected and `find_split_applied_symbols` is `hooks.find_split_applied`.

- [ ] **Step 1: Failing tests:** `test_split_corrections_fail_soft_at_every_ledger_read` (five cases, each logs the old warning and degrades as before),
  `test_persisted_state_for_an_unheld_symbol_is_cleared`, `test_applied_split_clears_the_persisted_row`,
  `test_dividend_accrual_returns_the_partial_result_on_each_failure_point` (`total`, `new_accruals`, `skipped`),
  `test_missed_exits_orders_candidates_and_survives_a_lookup_failure`,
  `test_split_check_reaches_the_patched_trader_name` (a string patch of `ggTrader.paper.trader.find_split_applied_symbols` wins).
- [ ] **Step 2-3:** Implement, cut, forwards; **Step 4:** mutation (shared helper: reach map first; the differential is
  blind to it): drop a `clear_split_correction` call; T, the digests that reach it and the unit test fail; revert.
- [ ] **Step 5:** The task gate. Commit: `refactor(paper): CorporateActions and missed_exits hold the split and dividend bodies`.

### Task 57: Wire `PreflightGate` (region R2) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/preflight.py`, `tests/paper/test_preflight.py` (append to `units.txt`); Modify
`trader.py` (`_run` lines 744-855 replaced by `PreflightGate(...).check()` with an `isinstance(pre, Halted)` early
return, the old locals rebound per the bridge table); lower `RUN_MAX_LINES`.

The gate keeps this order exactly: clock gate (fail open on a clock error, `Market closed` notice and
`Halted("market_closed", RunResult([], [], []))`), the signals `try` (**`hooks.membership_source(membership)` is
evaluated inside it, as an argument of `hooks.generate_signals(persist_tape=not read_only, membership=...)`**;
`SignalGateError` notice and re-raise, any other exception, a membership-source failure included, notice
`Paper trading failed: signal generation error\n<exc>` and re-raise), the per-sleeve info log and membership warning, the late-run gate
(`hooks.now_pt()`; live: notice and `Halted("late_run", RunResult([], [], [], gate="late_run"))`; dry: warning, DRY RUN
notice, continue), the read-only skip (`skipped_writes.append`, the info log) or `hooks.refresh_tape()` with its
warning, the overlay fallback notice, the signal derivation (`strategy_signals`, a plain function), the freshness
loop (`hooks.today_et()` read once; stale notice and `Halted("stale_signals", RunResult([], [], []))`).

- [ ] **Step 1: Failing tests:** `test_ratchet` (lower the bound, see it fail), `test_market_closed_returns_halted_with_the_old_notice_and_result`,
  `test_check_never_raises_for_a_halt` (the three reasons return `Halted`), `test_clock_error_fails_open`,
  `test_signal_gate_error_notifies_and_reraises`, `test_signal_error_notifies_and_reraises`,
  `test_late_run_live_halts_with_gate_late_run`, `test_late_run_dry_warns_and_continues` (quirk 1),
  `test_read_only_appends_to_the_shared_skipped_list_and_does_not_refresh`, `test_live_refreshes_and_warns_when_nothing_refreshed`,
  `test_fallback_notice_precedes_the_freshness_check`, `test_stale_unparseable_and_future_as_of_halt_with_the_old_texts` (three cases),
  `test_the_clock_is_read_after_signals_and_today_after_the_late_gate`, `test_empty_sleeves_raise_stop_iteration_through_check`,
  `test_membership_source_failure_is_inside_the_signal_try` (a hook whose `membership_source` raises: the notice
  above is sent once, `generate_signals` is never called, the same exception propagates; a `SignalGateError` from
  `generate_signals` keeps its own notice), `test_gate_uses_the_shared_constants_not_trader` (the freshness
  limit is `constants.MAX_SIGNAL_AGE_DAYS`; the module imports `gates`, `constants`, `models`, `ports`, `signals`
  types and nothing from `trader`).
- [ ] **Step 2-3:** Implement; cut; SM9 is re-located here (mutation: read `hooks.now_pt()` first; the event-order tests
  fail, the transcripts cannot); SM13a (the read-only skip message to another logger; the differential must fail);
  **SM17** (evaluate `hooks.membership_source(...)` before the `try`; the membership-alert tests and the W19 seeds fail).
- [ ] **Step 4:** The task gate (D under both hash seeds). Commit: `refactor(paper): PreflightGate takes the market, signal, late-run and freshness gates`.

### Task 58: Wire `PortfolioView.load` (region R3) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/portfolio.py`, `tests/paper/test_portfolio.py` (append to `units.txt`); Modify `trader.py` (`_run`
857-953 replaced; `RunContext` is built here, after the gate and the pending reconcile); lower the ratchet.

`load` performs, in this order: `get_account` (failure: `_account_read_halt` notice and re-raise), `get_positions`
(failure: the two-line log, notice inside its own `try`, re-raise), the strategy-position set (the sweep symbol removed when the sweep is on),
`resolve_untradable` with `cfg.untradable`, the frozen-holdings notice, the untradable-candidate notice, the buy filter, split corrections
(through `seams.split_corrections`), `apply_corrections_to_positions`, missed exits (through `seams.missed_exits`, merged into the sells,
logged), dividends (through `seams.accrue_dividends`). The view holds copies, so a later stop removal in the flow does not change it.

- [ ] **Step 1: Failing tests:** `test_load_reads_in_todays_order` (the recorded call sequence), `test_account_failure_halts_with_the_old_notice`,
  `test_position_failure_halts_even_if_the_notice_fails`, `test_sweep_position_is_invisible_to_strategy_positions`,
  `test_untradable_holdings_are_frozen_and_their_buy_signal_dropped`, `test_blocked_buy_candidates_are_announced`,
  `test_orders_use_uncorrected_positions_and_reports_use_corrected` (rule 5), `test_missed_exits_are_merged_into_the_sells_sorted`,
  `test_the_view_is_frozen_and_shares_no_mutable_state_with_the_broker_reply`, `test_load_uses_the_per_run_config_not_the_environment`.
- [ ] **Step 2-4:** Implement, cut, ratchet; SM12 mutation (read the environment in `load`): the env test fails. The task gate. Commit:
  `refactor(paper): PortfolioView.load reads the account, positions, splits and dividends`.

### Task 59: Wire `RiskAssessor` (region R4) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/risk_assessor.py`, `tests/paper/test_risk_assessor.py` (append to `units.txt`); Modify `trader.py`
(`_run` 955-1034, the `hard_stop` branch returns `RunResult([], [], [reason]).as_dict()`; the bridge copies
`verdict.buy_halts` into a new `loss_halts` list and clears `all_buys` when it is non-empty); ratchet.

`assess` keeps this order: persisted peak (read failure: warning, notice, hard stop; `update_peak` `ValueError`: notice,
hard stop), `check_drawdown_halt` (invalid value: notice, hard stop; else a buy halt), `update_peak` and the peak write when no halt
(write failure: warning, notice, hard stop), `last_equity` checks (missing, invalid: notice, hard stop), `check_daily_loss` with the
split note, then the halted-buys notice and the pending-orders notice, each inside its own `try`. The hard stops send their notice
outside any `try` (quirk 7). `RiskVerdict.with_halt` is what `OrderFlow` uses for an unassessable stop.

- [ ] **Step 1: Failing tests:** one per exit of the list above (nine), `test_split_note_names_symbols_sorted_even_with_non_string_keys`,
  `test_peak_is_saved_only_when_no_loss_halt`, `test_loss_halt_notices_fail_soft_and_the_pending_notice_still_goes`,
  `test_hard_stop_notice_failure_raises_through` (quirk 7), `test_assess_uses_the_guard_assigned_after_construction`.
- [ ] **Step 2-4:** Implement, cut, ratchet; SM15 (drop the clearing of the buys when `loss_halts` is non-empty): the
  loss-halt goldens and `test_loss_limit_allows_protective_exit_but_blocks_new_buys` fail, and so does the
  differential. Gate 1-6. Commit:
  `refactor(paper): RiskAssessor seeds the peak and applies the drawdown and daily-loss halts`.

### Task 60: Wire `OrderFlow` stage I: stops, exit validation, buy limits (region R6a) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/order_flow.py`, `tests/paper/test_order_flow.py` (append to `units.txt`); Modify `execution.py`
(intent-level API: `recover_intents`, `submit`, `announce` for stops, `DryRunExecutor` overrides), `trader.py` (`_run` 1048-1181 replaced by
`OrderFlow(...).run_stage_one(...)`, which returns the **`FlowState`** of "Interim flow state"; the rest of `_run` still runs inline on its
fields); ratchet.

The stage: `assess_stops`; with unassessable symbols, `verdict.with_halt(reason)`, `all_buys = []`, the alert and the pending notice
(each in a `try`); `plan_stops` advanced decision by decision (`Skip` is logged; `InvalidExit` appends the error and the symbol;
an `OrderIntent` goes to `executor.submit`, which returns **only the broker order id** (`None` in a dry run); **then**
the flow does what `_run` does now, in this order: `stopped.add`, `executed_sells.append`; **then, live only
(`executor.submits_orders`), it reads `strategy_positions[symbol]["market_value"]` (Quirk 11) and builds
`SubmittedOrder(order_id, "SELL", symbol, market_value, reason)` and appends it to the book's orders**; then
`announce`. A dry run updates the same sets, builds no `SubmittedOrder`, reads no `market_value`, and announces); the stopped symbols are removed from the open positions;
`validate_exits`; if any symbol is invalid the buy candidates are emptied; then the limits (`sleeve_slot_caps`,
`sleeve_open_count`, `buys_by_sleeve`, `sleeve_position_notional`, in this order, each evaluated here as today).

- [ ] **Step 1: Failing tests:** `test_stop_stage_submits_before_it_plans_the_next_symbol` (the order of fake calls interleaves),
  `test_a_stop_sell_is_booked_then_announced_in_that_order` (live and dry), `test_unassessable_blocks_buys_and_adds_a_halt_without_blocking_assessable_stops`,
  `test_invalid_stop_quantity_blocks_buys_not_other_stops` (rule 4), `test_stopped_symbols_leave_the_open_positions_at_once` (rule 1),
  `test_exit_validation_runs_after_the_stops_and_before_the_limits`, `test_limits_are_computed_in_the_old_order_and_raise_where_they_raised`,
  `test_a_stop_sells_book_amount_is_read_after_the_submit_and_never_in_a_dry_run` (Quirk 11),
  `test_the_flow_builds_the_submitted_order_from_the_returned_id_after_the_counters` (a fake executor records
  the sequence submit -> `stopped`/`executed_sells` visible -> `market_value` lookup -> order appended ->
  announce; with a `Mapping` subclass that counts lookups on `strategy_positions`; a seam returning `None` in a live
  run still books an order with `order_id=None`),
  `test_flow_state_after_stage_one_holds_every_field_the_inline_remainder_reads` (the fields of the interim table).
- [ ] **Step 2-4:** Implement; ratchet; SM 1, 4a, 11, 16: the rule tests and D (and the differential, N sites) fail; SM16 is caught by `test_a_later_stop_without_qty_raises_after_the_first_stop_order_went_out` and `test_exit_validation_without_qty_raises_after_the_stop_orders` (Task 48), now through the flow. The task gate. Commit:
  `refactor(paper): OrderFlow runs the catastrophe-stop stage and the exit validation`.

### Task 61: Wire `OrderFlow` stage II: sweep sell, sells, settlement of the sells (region R6b) (builder: Terra; reviewer: Opus)

**Files:** Modify `order_flow.py`, `execution.py` (`after_sells`, `unfilled_exit_errors`, the sweep-sell and sell announcements),
`trader.py` (`_run` 1183-1301 gone), ratchet, tests.

The stage: when the sweep is on and there is no halt, `plan_sweep_sell` (the anticipated-sell and slot estimate use the open positions after the stops;
`cfg.reserve_pct()` raises here, outside any `try`) and its execution (`executor.submit` returns the id; the flow builds `SubmittedOrder(oid, "SELL", sweep_sym, sell_action.notional, SWEEP_TRADE_REASON)` and appends it; no counter, no `market_value` read); `plan_exits` advanced lazily (after each submit the flow does `executed_sells.append` and `regular_sell_count += 1`, **then, live only,** reads `strategy_positions[symbol]["market_value"]` and builds and appends the `SubmittedOrder`; a dry run counts and announces only; `regular_sell_count` counts regular sells only);
`executor.poll` of the sells (through the seam, dry runs included; the list is the book's orders whose `side == "SELL"`); `after_sells(filled)` with the mapping `poll` returned; `unfilled_exit_errors(book.orders, filled, view.positions)` (live only; **`view.positions` is the raw broker mapping, not `strategy_positions` and not the corrected view**): each error is appended and the buys cleared (rule 3).
A sell still working at the deadline never reaches `unfilled_exit_errors`: `after_sells` raises first (Quirk 10).

- [ ] **Step 1: Failing tests:** `test_sweep_sell_is_planned_after_the_stops_were_executed` (SM 11 target), `test_a_bad_reserve_raises_after_the_stop_orders_went_out`
  (quirk 2), `test_regular_sell_count_excludes_stop_and_sweep_sells` (rule 2), `test_only_sells_are_polled_before_the_buys`,
  `test_a_terminal_unfilled_exit_blocks_all_buys_and_names_the_status` (rule 3, three statuses),
  `test_a_working_exit_raises_in_reconciliation_before_any_error_is_built` (Quirk 10),
  `test_an_unfilled_exit_leaves_the_sweep_buy_enabled` (Quirk 9),
  `test_an_invalid_exit_still_allows_a_reserve_restoring_sweep_sell` (Quirk 8),
  `test_dry_run_polls_an_empty_list_and_reconciles_nothing`,
  `test_unfilled_exit_errors_are_given_the_raw_broker_positions` (a split-corrected quantity differs from the raw
  one; the raw one decides), `test_the_flow_builds_each_sell_order_after_its_counters_and_its_market_value_read`.
- [ ] **Step 2-4:** Implement; ratchet; SM 2, 3, 8-adjacent, 11, 16 at N sites (D and the differential fail) and 7, 10 at S sites (D and the unit tests fail, the differential does not). The task gate. Commit:
  `refactor(paper): OrderFlow runs the sweep sell, the sells and their settlement`.

### Task 62: Wire `OrderFlow` stage III: buys, settlement, sweep buy (region R6c) (builder: Terra; reviewer: Opus)

**Files:** Modify `order_flow.py`, `execution.py`, `trader.py` (`_run` 1302-1397 gone), ratchet, tests.

The stage: `slots_available = risk.max_new_positions(len(open_positions) - regular_sell_count)` computed here (`open_positions` is the post-stop
`strategy_positions`; `max_new_positions(n)` takes a **held count** and returns **free slots**, see Task 48 rule 2); `plan_buys` advanced lazily (after each submit the flow does
`executed_buys.append`, `buys_attempted += 1`, `sleeve_open_count[universe] += 1`, then, live only, builds and appends `SubmittedOrder(oid, "BUY", symbol, sleeve_notional, None)`); the
poll of the remaining orders (`[o for o in book.orders if o.order_id not in filled]`); `after_orders(book.orders)` with **every order of the run, the sells included** (reconciles if the list is non-empty and the run is live); the sweep buy only when the sweep is on, there is no halt and no invalid exit symbol (rule 4), with the broker
account read, `plan_sweep_buy` and the old fail-soft `try` (`Cash sweep buy sizing failed (non-fatal): %s`); live: `executor.submit` returns the id, the flow builds `SubmittedOrder(oid, "BUY", sweep_sym, notional, SWEEP_TRADE_REASON)`, polls it (`filled.update(executor.poll([order]))`), **then** appends it to the book, then reconciles unconditionally (`executor.recover_intents()`, the same seam call).

- [ ] **Step 1: Failing tests:** `test_buy_count_and_sleeve_open_count_follow_yielded_intents`, `test_a_halted_run_submits_no_buys_and_no_sweep_buy`,
  `test_invalid_exit_blocks_the_sweep_buy` (rule 4), `test_sweep_buy_sizing_failure_logs_and_buys_nothing` (each stored parse error, and a broker error),
  `test_live_sweep_buy_is_polled_appended_and_reconciled_in_that_order`, `test_dry_sweep_buy_only_announces`,
  `test_a_submit_exception_ends_the_run_after_the_intent_is_saved` (rule 7a; no later order), `test_rejected_order_reaches_normal_settlement` (rule 7b),
  `test_three_held_one_stopped_one_ordinary_sell_permits_two_buys` (rule 2, the corrected example: held count `2 - 1 = 1`, two free slots),
  `test_after_orders_receives_every_order_including_the_sells` (a run with only sells still reconciles once; an empty list does not),
  `test_a_later_sleeve_missing_from_the_open_counts_raises_after_an_earlier_buy` (the Task 48 shape, now through the flow).
- [ ] **Step 2-4:** Implement; ratchet; SM 1-5, 8, 16 (N sites: D and the differential fail), SM 7 (S site). The task gate. Commit:
  `refactor(paper): OrderFlow runs the buys, their settlement and the sweep buy`.

### Task 63: Wire `RunReporter` and finish `_run` (regions R5, R7) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/reporter.py`, `tests/paper/test_reporter.py` (append to `units.txt`); Modify `trader.py` (`_run` becomes the
orchestrator described under "Interfaces"; the bridge locals and `FlowState` give way to `FlowOutcome`); ratchet to the measured size (≤ 45 lines).

`previous_value` is the prior-snapshot read with its two guards (R5). `finish` keeps this order: one `trade_alert` per submitted order (filled amount when filled),
the "N order(s) failed" message, the post-trade account and positions read (failure: log, notice inside a `try` whose text depends on dry or live, re-raise), the
corrected value (with the invalid-value error appended to `book.errors`), daily and unrealized P&L, the starting-capital read (failure appends its error), total P&L, the risk
line, `daily_summary(... errors=buy_halts + errors ...)`, the risk message, the snapshot write (failure appends its error and sends its notice inside a `try`), and the `RunResult`
(`buy_halts + errors` evaluated after the snapshot step; `untradable=sorted(...)`; `skipped_writes=list(skipped_writes)`). **The "total" percentage against both bases already
exists (`1c7a3138`) and is preserved:** `daily_summary` keeps every keyword it has today, `starting_capital=starting_capital` among them, and `notifier.py` is untouched.

- [ ] **Step 1: Failing tests:** `test_one_trade_alert_per_order_with_filled_or_intended_amount`, `test_post_trade_read_failure_text_depends_on_dry_or_live` (two cases),
  `test_invalid_post_trade_value_is_reported_and_no_snapshot_is_written`, `test_pnl_is_unavailable_if_any_unrealized_value_is_invalid`,
  `test_starting_snapshot_failure_keeps_the_summary_and_attempts_the_snapshot`, `test_summary_errors_exclude_the_snapshot_failure_but_the_result_includes_it`,
  `test_daily_summary_receives_starting_capital_and_every_keyword_it_receives_today` (the exact keyword set of `trader.py:1513-1525`; the dual-base total is computed in
  `notifier.py`, so a dropped `starting_capital` would silently remove the second base), `test_total_pnl_two_bases_transcript_is_unchanged` (the `total_pnl_two_bases` golden through T),
  `test_result_lists_are_the_books_own_lists` (risk 10), `test_final_run_is_an_orchestrator` (the ratchet, ≤ 45 lines), `test_run_py_unchanged_in_shape`.
- [ ] **Step 2-4:** Implement; the bridge locals go; SM 14 (N site: T, D and the differential fail) and a mutation that appends the snapshot error before the summary. The task gate. Commit:
  `refactor(paper): RunReporter reports the run; _run is an orchestrator`.

### Task 64: Wiring gate (phase D)

- [ ] **Step 1:** Mutation matrix SM 1-17 on the final tree, reach map first: each fails the nets of its row, recorded in `$E/p5/mutation-matrix-D.md` (S sites: differential blind, digests and named tests fail); a mutation no net catches is a finding and is fixed by adding a test through an **amendment note whose approver is the family that did not write the new test** (Global Constraints, "Amendment authority": Sol for a Claude-built test, Opus for a Codex-built one); a new test never goes into an immutable file without that note and a new hash line.
- [ ] **Step 2:** The phase gate (`gate_run.sh phaseD`, `PHASE=pre gate_compare.sh phaseD`, `verify_pins.py`); the oracle only with the owner's recorded 6G approval (the tests half runs and is compared either way; without the approval the gate is "tests half passed, oracle half pending").
- [ ] **Step 3: The records, in the two lists of Global Constraints.** (a) **Pre-existing immutable files and tools against `BASE5`:** `git diff --stat 73c35417 -- src/ggTrader/paper/persist.py src/ggTrader/paper/ledger.py tests/paper/transcript tests/paper/golden/trader_transcripts tests/paper/golden/persist_sql.json tests/paper/test_trader_transcripts.py tests/paper/test_persist_sql_golden.py tests/paper/test_phase4_pins.py tests/lab/test_fetch_stock_ohlcv_chain.py scripts/run_tests.sh scripts/run_capped.sh scripts/_heavy_job.sh scripts/compare_test_outcomes.py scripts/compare_evidence_json.py scripts/verify_pins.py scripts/p5_wfo_validation.py scripts/p7_live_golden.py conftest.py tests/paper/conftest.py tests/tiers.py tests/netguard.py offline.sh .github/workflows/ci.yml pyproject.toml` prints nothing; (b) **new immutable files against the Task 51 hashes**, not against `BASE5` (a diff against `BASE5` shows them as added): `sha256sum -c $E/p5-behavior.sha256` all OK, and the digest file's stored generator, witnesses and runner hashes equal the files; (c) **every pin and tool record:** `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-tools.sha256` all OK (the wrapper scripts and the pin lists of `$E/p5/` are in `p5-tools.sha256`, so a wrapper edited to make a gate pass fails here); (d) the amendment notes, each listed with its approver, for any line that changed.
- [ ] **Step 4: Reviews (the family rule).** **Opus** reviews Tasks 52-63 as one body (Codex-built; fresh context, given the diff, the rules, and the claim "behavior is unchanged"). **Sol** reviews the Claude-built pins (Tasks 47, 48, 50), and the decision to keep the seams and the `Halted` deviation (a design judgment, on evidence). **The family rule is symmetric:** Opus never reviews a Claude-built file and Sol never certifies a Codex-built one, though each may read the other's slice as context. Where this plan says "Opus judges", it means the verdict calls CLAUDE.md gives it (trust, audit conclusion, protocol decision, synthesis) over **evidence and Codex-built code**; a verdict about a Claude-built test needs Sol's review as its basis. Findings are fixed by the family that did not review the fixed code and approved by the reviewer of the other family, before Task 65.
- [ ] **Step 5:** Report usage (Phase D done).

### Task 65: The test rig and one pilot conversion (builder: Terra; reviewer: Opus)

**Files:** Create `tests/paper/support.py`; Modify `tests/paper/test_trader.py` (the pilot only: `TestSellExits`, lines 274-327; batch M0).

`support.py` holds: `class SpyLedger(InMemoryLedger)` recording every public call as `(name, args, kwargs)` in `calls(name=None)`;
`make_trader(...)` (the current `_make_trader` with `ledger=SpyLedger()` injected and the private `_submit_order` seam kept as today);
`seed_snapshots`, `seed_peak`, `seed_pending_orders` helpers; `PERSIST_NAMES` (the 26); and `persist_guard()`, the context manager of Task 73's
guard (built here, installed there). The conversion protocol (the same for Tasks 66-73):

1. Keep every node id and every parametrization id: **the 277 nodes of `$E/p5/trader-node-ids.txt`**, not the 197 test functions. The node-id comparison is per batch (`$E/p5/batches.json`).
2. Replace each `patch("ggTrader.paper.trader.<persist name>")` by seeding or spying through the rig; assertions on a persist mock become assertions on `ledger.calls(...)` or ledger state, one for one.
3. Leave non-ledger patches and the private seams alone.
4. The autouse persist stubs of the file stay until Task 73; a converted test injects its own ledger and is unaffected by them.
5. No new `skip`, `xfail` or weakened assertion; the number of assertions (`assert`, `.assert_*`, `pytest.raises`) per test function is not lower than at `73c35417` (a parametrized function's count applies to each of its nodes).
6. **Mutations are chosen by what reaches the batch, not by name.** Before converting, apply each candidate SM of the batch's row to the *unconverted* batch (the base-kill run) and record which of the batch's nodes fail (`$E/p5/mutation-kill-<batch>.md`); a candidate that no node of the batch kills is dropped for that batch and noted. After conversion the same mutation must fail at least one node of the batch. A mutation that needs a read-only run cannot be assigned to a live-run batch.

- [ ] **Step 1: The assertion-count guard.** It is a one-off outside the repository (`$E/p5/assertion_counts.py`, written by Haiku
  and checked by Sonnet, not committed): it prints, per test function, the number of `assert` statements, `.assert_*` calls and `pytest.raises`
  at a git ref. Record the `73c35417` counts for the whole file in `$E/p5/assertions-base5.json` before any conversion.
- [ ] **Step 2:** Convert `TestSellExits` (3 test functions, 4 persist patches, the class-level `get_latest_snapshot`, `log_snapshot`, `log_trade`, `init_paper_schema`); **Step 3:** `T tests/paper/test_trader.py -k TestSellExits` passes.
- [ ] **Step 4: Mutation, reach-checked.** The pilot is three **live** runs with `_submit_order` mocked and no read-only trader, so **SM6 cannot fail it** (it is a read-only-wrapper
  mutation; it belongs to M6). The pilot's mutations are SM5 (submit the split-corrected quantity: `test_sell_submits_brokers_uncorrected_qty_not_split_adjusted`) and SM-held (drop the
  `symbol not in strategy_positions` guard of the exit planner: `test_skips_sell_if_not_holding`), both confirmed by the base-kill run first, and a ledger-path mutation of the
  converted test: remove the snapshot write from the reporter and see the converted test's `ledger.calls("log_snapshot")` assertion fail (the rig must give the converted tests a ledger assertion they did not have as a mock).
- [ ] **Step 5:** Assertion counts not lower; outcome check on the file; the task gate. Commit: `test: in-memory ledger rig and the first converted trader test class`.

### Tasks 66-73: Convert the rest of `test_trader.py`, batch by batch (builder: Terra for each; reviewer: Opus for each)

Each task follows Task 65's protocol exactly: **Step 1** the `73c35417` assertion counts for the batch and the base-kill run of its candidate mutations; **Step 2** convert; **Step 3**
`T tests/paper/test_trader.py -k "<the batch's names>"` passes and then `T tests/paper/test_trader.py` passes in full (never
`tests/paper/` as a directory, which the orchestrator runs), with the batch's node ids all passing; **Step 4** the batch's reach-checked mutations fail it; **Step 5** assertion counts not lower, no new skip, the task gate (`pins-pre`); **Step 6** Opus reads the diff of the batch in a fresh context with the claim "each converted test still proves what it proved, and would fail on the mutations the base-kill run found"; **Step 7** commit
`test: convert trader tests <batch> to the in-memory ledger`.

| Task | Batch | Names (line range at `73c35417`) | Test functions | Persist patches | Candidate mutations (kept only if the base-kill run shows the batch reaches them) |
|---|---|---|---|---|---|
| 66 | M1 | TestMissedExitCatchUp, TestUntradableHoldings, TestMaxPositionsEnv, TestBenchmarkTapeKeepalive, TestBuyEntries, TestNotifications, TestErrorHandling, TestFillLogging (335-722) | 26 | 28 | SM 5, 14; the missed-exit and untradable sites |
| 67 | M2 | TestRunDateIsSessionDateNotAsOf, TestReconciliation, TestDailyPnl (725-1218) | 16 | 37 | SM 8, 14, a mutation that stamps `as_of` instead of the ET date |
| 68 | M3 | TestDividendAccrual, TestDryRun, TestRunPaperTrading, TestPendingOrderDedup, TestMarketGates, TestPersistedDrawdown (1221-1920) | 38 | 54 | SM 6 (`TestDryRun`), 9 (`TestMarketGates`: it reads the clock only if the batch's tests pin the late gate, which the base-kill run decides), a mutation that skips the peak save |
| 69 | M4 | the module functions from `test_loss_limit_allows_protective_exit_but_blocks_new_buys` to `test_invalid_risk_state_still_blocks_all_orders` (1923-2285) | 19 | 21 | SM 4a, 15 (`test_loss_limit_allows_protective_exit_but_blocks_new_buys`, `test_invalid_stop_position_*`) |
| 70 | M5 | TestStalePendingOrders, TestOrderErrorAlerting, TestCashSweepFlagOff, TestCashSweepEnabled, TestCatastropheStopFlagOff, TestCatastropheStopEnabled (2288-2835) | 23 | 30 | SM 1 (`test_stopped_symbol_not_rebought_by_coincident_buy_signal`, line 2708), 2 (`test_slot_math_counts_stop_exactly_once_at_position_cap`, line 2770), 7 (`test_order_failure_stops_later_buys`, line 2411), 11, 12 |
| 71 | M6 | TestFailClosed, TestUntradableAvb, TestReadOnlyDryRun, TestDryRunNotifier, TestRunPaperTradingReadOnly (2854-3263) | 29 | 2 | SM 6, 13a, 13b (only the read-only classes reach them) |
| 72 | M7 | TestLiveMembershipGate and the functions from `test_pending_order_read_failure_stops_live_run` to `test_loss_halt_stops_after_ambiguous_exit_submission` (3269-3749) | 21 | 17 | SM 7 (`test_run_stops_after_ambiguous_order_failure`, line 3701), 8 |
| 73 | M8 | the functions from `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit` to the end, `TestUnappliedSplitAndDailyLoss` included (3752-4224); then replace the seven persist autouse fixtures and `_intent_store` by the guard below | 22 | 18 | SM 3 (`test_exit_must_fill_to_free_a_position_slot`, line 3785), 4a and 4b (lines 3858, 3901, 3935, 4105), 7, 8 |

The per-class test and patch counts were measured on 2026-11-19 with `ast` and a regex over `test_trader.py` at `73c35417`; with Task 65's `TestSellExits` (3 test functions, 4 patches)
they sum to 197 test functions and 211 patches, and the other 28 persist patches sit in the autouse fixtures and `_intent_store` (Task 73). **The 197 functions are 277 collected nodes**
(parametrization): Task 46 Step 6 measures nodes per batch and each batch task checks its own. After Task 73:
no string patch of a persist name remains in `test_trader.py`, the other 200 dotted patches remain, and the file's outcome check against `p5-base-*` shows all 277 node ids passed.

**Task 73 adds the unstubbed-persist guard and keeps it (it is not deleted with the stubs).** Today the seven autouse fixtures exist because an unstubbed persist call reaches the database
(the 2026-09-22 hang came from one). Deleting them with nothing in their place would let a new test, or a new `trader.py` persist call, escape to the real pool. So Task 73 replaces them by **one** autouse
fixture, `_unstubbed_persist_guard`, in `test_trader.py`, built on `support.persist_guard()`: it patches every one of the 26 persist names in the `ggTrader.paper.trader` namespace (the list is
`ledger.TRADER_NAMES`) with a function that records the call and raises `UnstubbedPersistCall(AssertionError)`, and **its teardown fails the test if anything was recorded, even when the
trader swallowed the exception** (the connection guard N works the same way). A test that needs a name gives itself a ledger, or patches that name inside its own context (a test-level patch wins).
A test that goes through `run_paper_trading` (which builds its own trader with the default ledger) patches the names it needs explicitly; each such test is listed in the fixture's docstring.
`tests/paper/test_persist_guard.py` (new) proves it: an unstubbed call raises, a swallowed call still fails the teardown, an allowed patch wins. **Task 73 also edits `AGENTS.md`**
(`CLAUDE.md` is a symlink to it): the sentence "Any new `trader.py` persist import needs a stub in `tests/paper/test_trader.py`'s autouse fixtures" becomes "`tests/paper/test_trader.py` has an
autouse guard (`_unstubbed_persist_guard`) that fails any test which reaches a persist function without its own ledger or patch, even if the trader swallows the error; give a test an
`InMemoryLedger` or `SpyLedger` (`tests/paper/support.py`) instead of patching persist names." (If Task 75 is approved it amends the sentence again.)

### Task 74: Delete the frozen copy; the digests keep guarding (builder: Terra; reviewer: Opus)

**Files:** Modify `src/ggTrader/paper/trader.py` (delete `_run_legacy`, then every import and name only it used and no test or harness patches, from the Task 50 list; keep each name that is a patch target with its `# noqa: F401 - patch target`),
`tests/paper/equivalence/impl.py` (**only** the `IMPLS` line becomes `IMPLS: tuple[str, ...] = ("new",)`),
`tests/paper/test_run_structure.py` (keeps the ratchet; gains the two tests below). **Delete** `tests/paper/test_run_differential.py`, `tests/paper/test_run_legacy_frozen.py` and
`tests/paper/golden/run_legacy.sha256` (all **mutable-set** files). **Not edited:** every immutable-set file, `run_with.py`, `test_run_with.py`, `test_run_rules.py`, `test_seam_contract.py`, `test_run_digests.py` and the digest file included. The rule tests and the seam tests parametrize over `impl.IMPLS`, so they lose their `legacy` cases with no edit; `run_with("legacy", ...)` now raises `ValueError` (Task 47), and the `"legacy"` branch of `runner.run_recorded` stays unreachable and unedited (no remaining test calls it).

- [ ] **Step 1: Failing tests:** `test_trader_has_no_run_legacy` and `test_every_noqa_f401_in_trader_names_its_reason` (every remaining `# noqa: F401` in `trader.py` is followed by `- <reason>`), both in `test_run_structure.py`; the unused-import check itself is `ruff check` (the gate's step 1).
- [ ] **Step 2: Before.** Record **every** collected node of the pin files and the digest tests, fast and slow, with their outcomes: `T $(cat $E/p5/pins-pre.txt) -m "not slow" -rA --junitxml=$E/t74-before-fast.xml`, `T tests/paper/test_run_digests.py -m slow -rA --junitxml=$E/t74-before-slow.xml`, and `T $(cat $E/p5/pins-pre.txt) tests/paper/test_run_with.py tests/paper/test_run_digests.py --collect-only -q` for the slow set too (saved as `$E/p5/t74-collect-before.txt`, one node per line, both markers). The digest file is hashed (`$E/p5/t74-digest-before.sha256`).
- [ ] **Step 3: Delete, then After.** Delete the copy, the two test files, the hash file; change the one `IMPLS` line; switch the gates to `pins-post.txt` (which `test_run_with.py` is part of in both lists). Rerun the same three commands into `$E/p5/t74-collect-after.txt`, `$E/t74-after-fast.xml`, `$E/t74-after-slow.xml`. Then:
  - **Checks on `impl.py`:** `git diff -U0 -- tests/paper/equivalence/impl.py` shows exactly one removed and one added line, the `IMPLS` assignment; no other file under `tests/paper/equivalence/` differs from its Task 51 hash.
  - **The retired nodes, by name.** `$E/p5/retired-nodes.txt` is **computed, then reviewed**: `comm -23 t74-collect-before.txt t74-collect-after.txt` (nodes that existed and are gone), and each line is classified in the file as (i) a node of a deleted file (`test_run_differential.py`, `test_run_legacy_frozen.py`, **every** node of both, fast and slow collections), (ii) a **`legacy`-parametrized node of a surviving file**: every id of `test_run_rules.py`, `test_seam_contract.py` and any other immutable file parametrized over `IMPLS` that carries the `legacy` implementation (these exist in `pre` and must-pass-pre and disappear without an edit of those files, because `IMPLS` lost its second member; their `new` twins keep their ids), or (iii) anything else, which stops the task (an unexplained disappearance is a lost test, not a retirement). The `new`-implementation nodes' ids are checked to be **unchanged** (`comm -13 collect-before collect-after` lists only the two new structure tests).
  - **Added nodes:** `comm -13` shows exactly the two new tests of Step 1 and nothing else.
  - **The digest nodes** (every node of `test_run_digests.py`, fast and slow) have identical ids and outcomes before and after (`diff` of the two junit summaries filtered to that file), and the digest file hash equals `t74-digest-before.sha256` and the Task 49 record.
- [ ] **Step 4: Must-pass files, exactly.** `$E/p5/must-pass-post.txt` = `must-pass-pre.txt` minus `retired-nodes.txt` plus the two new tests; the orchestrator checks (a) every node in `retired-nodes.txt` is in `must-pass-pre.txt`, (b) none is in the baseline `p5-base-*.xml` (the baseline predates every one of them, so no baseline node is retired by this task), (c) every other node of `pre` is in `post` (`comm -23 pre-sorted post-sorted` equals `retired-nodes.txt`), (d) `post` still holds the 277 `test_trader.py` ids and both live-golden nodes. From here the gates use `PHASE=post`. `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-tools.sha256 $E/p5-behavior.sha256` still all OK (no immutable file changed); the 103 transcripts and the rule tests pass.
- [ ] **Step 5:** The task gate (`pins-post`). Commit: `refactor(paper): delete the frozen oracle copy of _run; recorded digests keep guarding`.

### Task 75 (OWNER-GATED, O-P5-1): retire `ModuleLedger` and the string-patch harness

**Why it is a task of this plan.** Design section 7 says "delete the adapter and `_run_legacy`". Without this task the phase leaves the adapter in place and **must be reported as incomplete against the
design**; with it, Phase 5 matches the design. It runs only after the owner approves in writing, because it edits files the owner asked to keep immutable and one hash-recorded digest file. Builder: Terra; reviewers: Opus for the Codex-built code, Sol for any Claude-built part (the amendment authority of Global Constraints applies to each file below: files of mixed or unrecorded authorship need **both**); orchestrator re-verification as in the task gate. If the owner declines or defers,
the plan stops at Task 74 and Tasks 76-78 report "complete except adapter retirement (O-P5-1)".

**The closed list of frozen files this task edits (the owner's approval names exactly these; the Step 0 inventory may remove a name from the list, never add one: a further file found is a stop and a new approval).**

| File | Set | Edit | Proof it changes no recorded behavior |
|---|---|---|---|
| `tests/paper/transcript/harness.py` | pre-existing immutable (`p5-fixed.sha256`) | the persist-patch loop and the `paper_run_lock` replacement become the injection of Step 1 | 103 goldens byte-identical |
| `tests/paper/test_phase4_pins.py` | pre-existing immutable (`p5-fixed.sha256`) | the lines asserting the persist names stay bound in `trader` (about 106-107 and 127) | the rest of the file passes unchanged |
| `tests/paper/equivalence/runner.py` | new immutable (`p5-behavior.sha256`) | takes the same injection instead of `persist_function_names()` | `test_run_digests.py` (fast and slow) passes against the digest file's **per-seed digests, unchanged** |
| `tests/paper/golden/run_equivalence_digests.json` | new immutable (`p5-behavior.sha256`) | **only** `runner_sha256` (the runner it records the pin of) and one appended `amendments` entry (date, reason, the new `runner_sha256`, the commit); `generator_sha256`, `witnesses_sha256`, `production_base`, `recorder_commit` and the whole `digests` mapping are not touched | the `digests` object's canonical-JSON sha256 equals the one recorded before the edit (`$E/p5/digests-map.sha256`, taken at Task 49 Step 7); no digest is recomputed or re-recorded |
| `tests/paper/test_seam_contract.py` | new immutable | `test_trader_persist_names_unchanged` is removed or reduced to the ledger-protocol equality, and the `test_legacy_seams_intercept_the_run` case "a string patch of one persist name wins" becomes "an injected ledger's method wins"; every other case unchanged | the file's other nodes keep ids and pass |
| `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py` | new immutable | **only if** their hook-order environment copy patches persist names (the Step 0 inventory decides); then the same injection | the rule tests keep their ids and pass on the unchanged goldens |
| `tests/paper/test_trader_ledger.py` | Phase 4 test | its persist-name patching and `TRADER_NAMES` uses | the injected-ledger tests keep ids and pass |
| `tests/paper/test_ledger.py` | Phase 4 test | the five `test_module_ledger_*` tests and `test_trader_names_are_the_26_names_trader_imports` are removed with `ModuleLedger`; `test_module_exposes_its_public_names` drops the name `ModuleLedger` | see "Baseline node retirement" below |

Also recorded: `$E/p5-fixed.sha256` (the `harness.py` and `test_phase4_pins.py` lines) and `$E/p5-behavior.sha256` (the `runner.py`, `test_seam_contract.py`, `test_run_rules.py`/`rules_support.py` if edited, and digest-file lines) are re-recorded **after** each amendment is approved, each re-recorded line listed in `$E/p5/amendments.md` with the approver. Files **not** frozen and edited here: `src/ggTrader/paper/trader.py`, `src/ggTrader/paper/ledger.py`, `tests/paper/test_trader.py` (the explicit persist-name patches Task 73's guard docstring lists for `run_paper_trading` tests), `tests/paper/support.py`, `tests/paper/transcript/injected.py` and `tests/paper/test_trader_transcripts_injected.py` (new).

- [ ] **Step 0: Owner approval, recorded** in `$E/p5/amendments.md`, quoting the owner and naming each row of the table above. Inventory (Haiku, checked by Sonnet): every `ModuleLedger`, `TRADER_NAMES`, `persist_function_names` and `sys.modules` reference; every persist-name string patch outside `test_trader.py`; the harness persist loop; the `test_phase4_pins.py` lines asserting the names bound; `tests/paper/test_ledger.py`'s `ModuleLedger` tests; every other user of `TRADER_NAMES` (it is deleted with `ModuleLedger` unless the inventory finds a user, in which case it is kept and the amendment note says why). Write `$E/p5/digests-map.sha256` (if not already there from Task 49) and `$E/p5/retired-baseline-nodes.txt` (below) here, before any edit.
- [ ] **Step 1: Build the injected harness beside the old one.** New file `tests/paper/transcript/injected.py` (the harness is not edited yet): `HarnessLedger`, a `PaperLedger` whose every method calls `run.persist_wrapper(name)` (so the recorded `persist.<name>` entries are unchanged) and whose `paper_run_lock` is `run.paper_run_lock`; `run_scenario_injected(scenario)`, a copy of `_run_scenario` that passes `ledger=HarnessLedger(run)` and patches only the non-persist names. New test `tests/paper/test_trader_transcripts_injected.py`: for all 103 scenarios the injected transcript's `to_json` equals the same golden file, byte for byte. Both harnesses pass: this is the proof that retiring the patching changes no recorded call. Not a gate point on its own beyond that.
- [ ] **Step 2: Amend the frozen files of the table, one commit each**, in the order harness, runner, digest-file provenance, pins, seam contract, rules (if listed), the Phase 4 tests; each with its approver recorded and its proof run: after the **runner** commit, `T tests/paper/test_run_digests.py -m "not slow"` and `-m slow` pass; after the **digest-file** commit (`runner_sha256` and the `amendments` entry only) `test_digest_file_pins_the_generator_and_the_runner` passes and the `digests` map hash equals `digests-map.sha256`. The 103 goldens, `scenarios.py` and `test_trader_transcripts.py` stay byte-identical (their `p5-fixed.sha256` lines are unchanged; only the lines of the edited files are re-recorded).
- [ ] **Step 3: Migrate the remaining string patches** outside `test_trader.py` (`test_trader_ledger.py` and any other the inventory found) and the explicit persist-name patches of the `run_paper_trading` tests inside `test_trader.py` to an injected `InMemoryLedger` or `SpyLedger`; the autouse guard of Task 73 now patches the default ledger's factory instead of the 26 names (`ggTrader.paper.trader.PgPaperLedger` -> a factory whose every method records and raises `UnstubbedPersistCall`). All 277 `test_trader.py` node ids still pass.
- [ ] **Step 4: Retire.** `PaperTrader.__init__` builds `PgPaperLedger()` as its default (it is stateless and reads the engine at each call, so construction opens no connection: `persist.py:135-150`); `ModuleLedger` leaves `ledger.py`, its tests leave `tests/paper/test_ledger.py`; the 26 persist imports and the `sys.modules` lookup leave `trader.py`; the constructor docstring and the module docstring of `ledger.py` stop describing a module lookup. `git grep -n ModuleLedger src tests scripts` finds nothing outside history and the changelog.
- [ ] **Step 5: Baseline node retirement, narrowly scoped (the comparator is not edited).** `compare_test_outcomes.py` treats a baseline node missing from the new run as a breach ("missing from new") and a smaller node count as a breach, so the removed `ModuleLedger` tests would fail the gate; the script and `gate_compare.sh`'s defaults stay as hash-pinned. Instead, **only for gates from this task on**, `PHASE=post BASELINE_SUFFIX=-t75 $E/p5/gate_compare.sh LABEL` compares against `$E/p5-base-core-t75.xml` and `$E/p5-base-studies-t75.xml`: copies of the baseline junit files with exactly the `<testcase>` elements named in `$E/p5/retired-baseline-nodes.txt` removed, made by a one-off `$E/p5/filter_baseline.py` (outside the repository, sha256 recorded). Rules: every listed node is in `tests/paper/test_ledger.py` and tests only `ModuleLedger` or the "26 names are bound in the trader" check (at most the six nodes of the table above; the Step 0 inventory fixes the exact ids); a node that was merely edited stays in the baseline and must pass; the file, the filter script, the two filtered XMLs and the owner's approval are listed in `amendments.md`; `gate_compare.sh` refuses `BASELINE_SUFFIX` unless that file and the amendment note exist; `must-pass-post.txt` loses the same nodes (they appear in neither baseline nor new run). The unfiltered baseline remains the comparison target for every gate before this task and is kept.
- [ ] **Step 6:** The task gate and a **phase-style gate** (fast core, `--full`, `--studies`, oracle with approval at exactly 6G; the tests half runs either way): the 103 goldens byte-identical, the digest tests (fast and slow) unchanged against the unchanged per-seed digests, the SQL golden and `persist.py` untouched, `ledger.py`'s diff exactly the deletion of `ModuleLedger` (and `TRADER_NAMES` if deleted), every changed frozen-file hash listed in `amendments.md`. Commit per step; the last: `refactor(paper): retire ModuleLedger; the transcript harness injects its ledger`.

### Task 76: Lint survivors, docs, changelog (builder: Sonnet; reviewer: Sol)

**Files:** Modify `ruff.toml`, `docs/architecture.md` (the paper-trader section: the nine step objects, the data flow of one run, which seams stay and why, the rules table, `Halted`), `docs/changelog.md` (one entry: what moved, "no order, SQL statement, Telegram line or log record changed", the nets added, the preserved quirks, the migrated tests, the guard; whether the adapter was retired; merged with Phase 3's and 4's entries; **no mention of a `ledger.py` docstring edit: there was none in Phase 5**); `docs/cli_reference.md` is untouched (output did not change).

- [ ] **Step 1:** Run `ruff check --select C901,PLR0912,PLR0915,PLR0913 src/ggTrader/paper` with design 5.4's limits (`max-complexity = 15`, `max-args = 8`); list every survivor.
- [ ] **Step 2:** Enable the four rules for `src/ggTrader/paper` with a per-file ignore for each survivor, each with a one-line reason (a deliberate wide constructor, a long moved body whose only change was a move). Nothing outside `paper/` is switched on here (design 5.4 enables the rest later).
- [ ] **Step 3:** Docs, **without any gate result.** Task 76 runs *before* the final gate (Task 77) and the final reviews (Task 78), so it cannot truthfully hold their results. Remove the stale "verification status" paragraph Phase 4 left and put in its place one line: "Phase 5 verification: recorded in Task 78's commit `docs: record phase 5 verification`" (the commit does not exist yet; the architecture and changelog entries say what was built and what was preserved, not what passed). The changelog entry also states whether the adapter was retired.
- [ ] **Step 4:** The task gate. Commit: `docs: refactor phase 5 (the trader run split into step objects)`.

### Task 77: Integration gate (orchestrator)

- [ ] **Step 1:** `git log --oneline refactor/phase-5 --not refactor/phase-4` lists every task; `git status` clean.
- [ ] **Step 2:** The phase gate on the final head, one job at a time: fast core; `--full` and `--studies` (`$E/p5/gate_run.sh phase5`: oracle half at the 6G cap only with the owner's recorded approval, tests at 3G);
  `PHASE=post $E/p5/gate_compare.sh phase5`: the outcome check against `p5-base-*.xml` with `must-pass-post.txt` (every Phase 5 pin that still exists, **all 277 `test_trader.py` node ids** and both live-golden nodes); `scripts/verify_pins.py` with the counts of `.github/workflows/ci.yml`. Without the owner's 6G approval the oracle half is reported "not run" and the gate is **not** declared passed.
- [ ] **Step 3:** The records: `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-behavior.sha256` all OK (or the amendment notes of Task 75, each listed); `git diff --stat 73c35417 -- src/ggTrader/paper/persist.py` empty; `git diff --stat 73c35417 -- src/ggTrader/paper/ledger.py` **empty if Task 75 did not run, exactly the `ModuleLedger` deletion if it did**; `git diff --stat 73c35417 -- tests/paper/golden/trader_transcripts tests/paper/golden/persist_sql.json` empty; the digest file's hash unchanged.
- [ ] **Step 4:** `grep -n "os.environ\|getenv\|_write\b" src/ggTrader/paper/trader.py` prints nothing; `_run` is at most 45 lines and `run` at most 25 (the ratchet); every new module passes the discovered-module scan of Task 50.
- [ ] **Step 5:** The final mutation matrix (SM 1-17), reach map first, on the final head. **Step 6:** report usage. (This task records its results in `$E` only; the documents that quote them are written by Task 78 Step 5, after the reviews.)

### Task 78: Independent reviews and the hand-back

- [ ] **Step 1: Sol** (fresh context) reviews the **Claude-built** work (Tasks 47, 48, 50, 76, and the orchestrator's fixes): given the branch diff of those tasks, the seven rules, the safety-invariant list, the immutable list and the claim "no behavior changed"; not given the builders' reports. Sol may read the rest of the branch as context but does not certify the Codex-built code (family rule). Findings are checked by the orchestrator against the code before they are acted on.
- [ ] **Step 2: Opus** (fresh context) reviews the **Codex-built** work (Tasks 49, 52-75 as built by Terra): same evidence plus the SM matrix results and reach maps; asked to find a mutation or an input class the nets miss. Opus does not certify the Claude-built pins (that is Sol's, Step 1). Opus renders the **equivalence verdict** and the judgment on the deviations that touch live safety (Deviation 1, the lazy planners, the `Halted` return) as the critical judgments CLAUDE.md assigns it, **on evidence and on the Codex-built code**; where a verdict leans on a Claude-built test, it quotes Sol's Step 1 finding for that test and does not re-review it. The hand-back states which reviewer certified which slice.
- [ ] **Step 3:** Each finding is fixed in its own commit, with its test, and the affected gate steps rerun; a finding that is a candidate behavior change (the quirks) is **not** fixed and goes to the hand-back. A fix is reviewed by the family that did not write it.
- [ ] **Step 4: Record the verification, after the reviews (a documentation-only commit).** Only now, with the gate (Task 77) and both reviews (Steps 1-2) done and their findings fixed, write the final gate results into the documents Task 76 left a pointer in: `docs/architecture.md` and `docs/changelog.md` get the "verification status" paragraph (gate labels, pass/pending halves, node counts, the 103 goldens and 600 digests unchanged, the mutation matrix result, who reviewed which slice, the owner decisions still open, whether Task 75 ran). The commit changes documents only (`git diff --stat <head before> HEAD -- . ':!docs'` prints nothing), so no gate reruns except `ruff` and the hash records (`$E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-tools.sha256 $E/p5-behavior.sha256` still OK). Commit: `docs: record phase 5 verification`. If a gate half is still pending (the oracle awaiting O-P5-3), the paragraph says exactly that and the phase is **not** reported verified.
- [ ] **Step 5: Hand-back** to the owner: branch `refactor/phase-5` and its head, the gate results as recorded in Step 4 (and any half "not run"), the anchors before and after (`_run` 826 lines to at most 45; 239 persist patches to 0), the preserved quirks (1-13) as candidate findings, the deviations (notably `Halted` in place of a raised `RunHalted`), the owner decisions still open (O-P5-1, O-P5-3, O-P5-4, O-P5-5), and usage. **Say plainly whether Task 75 ran: if it did not, the phase is "complete except adapter retirement" and not complete against design section 7.** The Telegram dual-base total is already merged on the base (`1c7a3138`) and untouched. **No merge, push or deploy; the trader stays off.**

## Execution roles (`CLAUDE.md`, delegation preference of 2026-11-08)

**Quota first.** The owner reported ample Claude and Codex quota on 2026-11-19; the agent cannot see usage, so usage is reported after each phase and the owner can stop early.
Phases and expected runs:

| Phase | Tasks | Builder runs | Reviewer runs |
|---|---|---|---|
| A: nets | 46-51 | 3 Sonnet (47, 48, 50), 1 Terra (49), Haiku inventories | 2 Sol (47-48-50, Task 51 read), 1 Opus (49 pre-record gate, normalization approval, Task 51 hash confirmation) |
| B-C: inert pieces and moved bodies | 52-56 | 5 Terra | 5 Opus |
| D: wiring | 57-63 | 7 Terra | 7 Opus, then 1 Opus (Terra's 52-63) and 1 Sol (Sonnet's pins) at Task 64 |
| E: tests | 65-74 | 10 Terra | 10 Opus |
| E2: adapter (owner-gated) | 75 | 1 Terra | 1 Opus (+ Sol for any Claude-built part) |
| F: docs, gate | 76-78 | 1 Sonnet | 1 Sol, final Sol and Opus (each on the other family's slice) |

- **Orchestrator: Claude Sonnet.** It re-verifies every task itself (reads the diff with `--color-moved`, reruns the task gate, runs the mutations, accepts no builder claim unchecked) and runs the oracle and `--full` runs one at a time.
- **The family rule, as `CLAUDE.md` states it: a reviewer never reviews its own model family's build.** Claude output (Sonnet, Haiku) is reviewed by **Sol**; Codex output (Terra, Luna) is reviewed by **Opus**. Applied here: the pins and rules (Tasks 47, 48, 50) and docs (76) are Sonnet-built and reviewed by **Sol**; the generator, digests (Task 49), the refactoring under them (52-63), the test migration (65-74) and Task 75 are Terra-built and reviewed by **Opus**. Opus therefore does not read the Claude-built rules tests, and Sol does not read the Codex-built generator. A fix is attributed to the family that wrote it and reviewed by the other. **Amendments follow the same rule** (Global Constraints, "Amendment authority"): Sol approves an amendment to a Claude-built file, Opus one to a Codex-built file, both one to a file of mixed or unrecorded authorship; Opus's verdicts are evidence-based and never a certification of a Claude-built test. "Anything headed for a merge gets both" is met at Tasks 64 and 78 by **each reviewer certifying the other family's slice**, both reading the whole diff as context. Whole-branch reviews are given only the evidence and the claims; a review of a family's own build says so.
- **Haiku** takes the inventories (Task 46 anchors and node lists, per-node assertion counts, the mutation-site map, the Task 75 inventory); a stronger model samples each batch and checks every figure that reaches a report.
- **Briefs** carry the task text, the file paths, the heavy-job block of Global Constraints (with the explicit ban on the oracle scripts), and no private account records.
- The lead remains responsible for the final call.
