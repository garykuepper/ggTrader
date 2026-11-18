# ggTrader refactor: consolidated design

**Date:** 2026-11-17
**Status:** Owner decisions 1 to 5 recorded 2026-11-17 (section 8). Execution has not been
approved. No code has changed.
**Scope:** `src/ggTrader/` (22,680 lines in 105 files) and `scripts/` (about 7,400 lines).
Study code under `research/` is out of scope: each study is bound to its results by a code hash.
**Implementation plan for phases 0 to 2:** [2026-11-17-refactor-phase-0-2.md](../plans/2026-11-17-refactor-phase-0-2.md).
Phases 3 to 6 get their own plans once phase 0 has landed.

This document replaces the separate code review and the four review reports. Section 9
records what each review contributed and which of the original proposals the reviews
overturned.

Terms used throughout:

- **Transcript test**: a test that records, in order, every call a piece of code makes to the
  outside world (broker, database, Telegram) and compares that record with a saved copy.
- **Oracle drivers**: `scripts/p5_wfo_validation.py` (the walk-forward reference runs) and
  `scripts/p7_live_golden.py` (the live signal reference over 120 sessions). Their output is
  the measuring stick for research numbers.
- **Pin**: a recorded SHA-256 of a source file that a piece of evidence depends on, checked by
  `scripts/verify_pins.py`.
- **WFO**: walk-forward optimisation (train on one window, test on the next, roll forward).

---

## 1. Goal

Make the code clear, concise and maintainable. Use object-oriented design where it models real
state and behavior, and keep plain functions where they are the better design. Nothing the
trader trades and no research number may change, except through a change the owner approves
by name.

## 2. Current state

- **The code base has two halves.**
  - **The newer half already has the target shape.** It covers the membership database
    (`data/membership*.py`, `data/universe_db.py`), `engine/`, `lab/wfo/` and
    `lab/strategies/`, and uses frozen dataclasses, protocols, injected clocks and stores, and
    a typed error hierarchy.
  - **The older half is long procedural code that passes plain dicts and tuples.** It covers
    `paper/trader.py`, `paper/persist.py`, `lab/data.py`, `data/live/` and `cli/cmd_db.py`.
- **Style passes; structure does not.** The configured ruff rules and formatting are clean.
  Wider rules find 47 functions with too many arguments, 26 that are too complex, 169 `print`
  calls and 58 broad `except` clauses in `src/`.
- **The largest problems:**
  - `PaperTrader._run` (`paper/trader.py:730-1558`) is one 829-line method with about 20 jobs
    and nine early returns in three result shapes.
  - The trader imports 26 persistence functions by name, and `tests/paper/test_trader.py`
    (4,220 lines) replaces them by string 449 times.
  - `run_wfo` (`lab/wfo/run.py:169-524`) takes 23 parameters, builds each fold as a 27-key
    dict, and returns either a result or a string.
  - `fetch_stock_ohlcv` (`lab/data.py:58-185`) chains four data sources through copy-pasted
    blocks.
- **Defects found along the way:**
  - **Dead code:** `PaperTrader._order_failed` (`trader.py:697-707`) is never called, so its
    alert (`:1422-1427`) cannot fire. The docstring in `paper/untradable.py:10-12` still
    promises a skip-and-continue behavior that does not exist.
  - **Unguarded wipe commands:** `ggt db truncate` empties the price table and `ggt db clean`
    deletes rows, both without confirmation (`cli/cmd_db.py:97-115`).
  - **No logging setup:** nothing configures Python logging, so the trader's INFO lines,
    including the dry-run "intended order" lines, are probably dropped.
  - **Possible wrong percentage:** `paper/notifier.py:151` divides total profit by the day's
    starting value, not by starting capital.

## 3. Design principles

1. **Use a class where something has state, more than one implementation, or a caller that
   should not care which implementation it gets.** In this code base that means brokers,
   notifiers, the paper ledger, order execution and run reporting.
2. **Keep pure calculations as functions over frozen inputs.** This applies to sizing, gates,
   risk thresholds, statistics, parsing and order planning. They can then be tested with plain
   data and no mocks.
3. **Use typed records instead of dicts and tuples at boundaries.** Frozen dataclasses or
   `TypedDict`s, each with an `as_dict()` that reproduces today's keys wherever a record is
   stored or compared.
4. **Every move leaves the old name importable.** Studies and tests import `src` names, so a
   moved name stays reachable as a re-export or alias.
5. **Prove before changing.** Each refactor slice is preceded by a test that captures today's
   behavior, and must reproduce that behavior exactly.

## 4. Fixed constraints

- **Engine boundary:** only `src/ggTrader/engine/` imports vectorbt.
- **Strategy contract:** `name`, `kind`, `ParamGrid` and `build(bars, grid)`, with the registry
  built by scanning `lab/strategies/`.
- **Import direction:** nothing in `src/` imports a study, and `utils` holds no domain logic.
- **Production status of `paper/`:** it places real orders with `--live`. The trader stays off
  throughout; cron returns only on the owner's word.
- **Price table:** `ohlcv.timestamp` is naive UTC, and no change may introduce a tz-aware
  write.
- **Frozen names.** Study-imported `src` names stay importable. There are 53 distinct pairs in
  166 study files, among them:
  - `lab.persist.get_engine` and the private `_sanitize`;
  - `lab.data`, `lab.a23_rebalance` (`COST_BPS`, `Book`, `Lot`, `_trade_to`) and
    `lab.wfo.breaker`;
  - `paper.cash_sweep`, `.catastrophe_stop`, `.risk`, `.live_bars` and `.live_signals`
    (`MIN_HISTORY_BARS`, `live_ensemble_frames`);
  - `ggTrader.utils.*`.
- **Untouched files.** The measuring sticks and standard-library-only scripts are never
  edited: `scripts/p5_wfo_validation.py`, `p7_live_golden.py`, `legacy_reference_runs.py`,
  `legacy_reference_compare.py`, `verify_pins.py`, `verdict_register.py`, `study.py` and
  `vbt_latest.py`.
  - `legacy_reference_runs.py` executes `scripts/make_tape_snapshot.py` by path on top of the
    tagged `legacy-engine-vbt0285` package, which has no `utils/manifest.py`. That script's
    own hash function therefore stays.
- **Owner policy (CLAUDE.md):** no paid data, never delete under `data/`, and no
  recommendations to retire the trader, the sleeve or the project.

## 5. Target architecture

### 5.1 The paper trader run (phase 5)

`PaperTrader.run` becomes about 40 lines that call these steps in today's order. The order
itself is behavior: the sweep sell comes before strategy sells, and buys wait for sells to
fill.

| Object | Kind | Owns | From today's lines |
|---|---|---|---|
| `TraderConfig` | frozen dataclass, `from_env()` read once | risk limits, sweep, stop, untradable set, dry-run and read-only flags | env reads at `:712-714`, `:881-882`, `:898`, `:1209`, `:1371-1373` |
| `RunContext` | frozen dataclass | `today_et`, `now_pt`, `config`, pending symbols; the clock is read once | `:106`, `:111` |
| `PreflightGate` | class; returns `SignalBundle` or raises `RunHalted(reason, result)` | market open, signals, late run, freshness | `:747-854` |
| `PortfolioView` | frozen dataclass, built by `PortfolioView.load(...)` | account, raw, strategy and corrected positions, splits, sweep position, untradable set, dividends | `:856-953` |
| `RiskAssessor` | class around `RiskGuard`; returns `RiskVerdict(hard_stop, buy_halts)` | peak seed, drawdown halt, daily-loss halt | `:959-1034` |
| Planning | **pure functions** in `paper/planning.py`: `plan_stops`, `plan_exits`, `plan_sweep_sell`, `plan_buys` | decide what to trade; return `OrderIntent` lists and skip reasons | decisions inside `:1070-1336` |
| `OrderExecutor` / `DryRunExecutor` | classes holding broker, ledger, lock and notifier | submit, wait and reconcile; the dry-run variant removes five `if self._dry_run:` branches | `:181`, `:212`, `:349`, `:417` |
| `CorporateActions` | class | split corrections and dividend accruals, moved as they are | `:474-695` |
| `RunReporter` | class | alerts, corrected value, profit figures, daily summary, snapshot | `:1403-1550` |
| `RunResult` | frozen dataclass with `as_dict()` | `buys`, `sells`, `errors`, `untradable`, `skipped_writes`, `gate`; today's keys, so `cli/main.py` is unchanged | `:758`, `:794`, `:854`, `:965`, `:1552-1558` |

Each of these rules is written as a comment in today's code. Each must become a named test
that still passes after the split:

- A catastrophe-stopped symbol leaves the strategy positions at once, and is not sold twice or
  re-bought in the same run.
- Slot arithmetic uses the count of regular sells, not all sells.
- A sell that does not fully fill blocks all buys.
- An invalid exit quantity blocks buys and the sweep buy.
- Orders use the broker's uncorrected quantity; split-corrected positions are for reporting and
  concentration only.
- A read-only run writes nothing and records what it skipped.
- An exception raised on submit ends the run after the intent is saved. An order the broker
  later reports as `rejected` goes through normal settlement and can leave other exits
  eligible.

### 5.2 Interfaces and typed records for the paper trader (phase 4)

- **`PaperLedger` protocol.** It covers the run lock, order intents, legacy pending orders,
  snapshots, peak value, split and dividend state, and trade dates. It has three
  implementations:
  - `PgPaperLedger`: today's SQL, statement for statement, always run inside
    `engine.begin()`, and with explicit parameter dicts in place of `locals()`.
  - `ReadOnlyLedger(inner)`: passes reads through and records each write by name; it replaces
    `PaperTrader._write`.
  - `InMemoryLedger`: the test double.

  The module-level functions in `paper/persist.py` stay as one-line delegations, because
  studies and tests import them. Until the trader tests are migrated, a `ModuleLedger` adapter
  looks each function up on the `trader` module at call time, so every existing string patch
  keeps working.
- **`Broker` and `Notifier` protocols** go in `paper/ports.py`. The one-method `Notifier` in
  `data/membership_weekly.py` stays separate, because `data` must not import `paper`.
- **Records** go in `paper/models.py`: `Account`, `Position`, `OrderStatus`, `MarketClock`,
  `SubmittedOrder`, `SleeveSignals`, `SignalBundle` and `RunResult`.
  - `SubmittedOrder` replaces the five-item tuple read by index; it is the first and
    lowest-risk step.
  - `Position` starts as a `TypedDict`, which changes nothing at run time, because snapshots
    store positions as JSON and the split, dividend and study helpers take mappings.
- **`OhlcvFetchChain`** goes in `data/live/`, built from a primary source, gap fillers, a cache
  and an optional negative cache. It replaces the three copy-pasted blocks in
  `fetch_stock_ohlcv`. Two cache routes must stay distinct:
  - a cache that constructs but returns nothing keeps its writer attached, so gap fills are
    still saved;
  - a cache whose constructor or fetch fails is dropped, so nothing is saved.

  `CachedYFinanceLoader` gains a public `persist()`. `lab.data.fetch_stock_ohlcv` stays as a
  thin wrapper with the same signature and result. `paper/` then imports its data path from
  `data/live`, not `lab`.

### 5.3 The research engine (phase 3)

- **`run_wfo`** keeps its public signature, because three production callers and several
  studies read its output by key. It gains these pieces:
  - `SignalSimOptions`, which carries the four errors now at `run.py:158-165`;
  - `ScoringInputs`, a single object both scorers take in place of nine repeated arguments;
  - `FoldRecord`, a frozen dataclass whose `as_dict()` gives the same 27 keys in the same
    order;
  - the helpers `_score_fold`, `_stitch_oos` and `_build_diagnostics`.

  `WfoResult.fold_results` stays a list of dicts. Replacing the string return with an
  exception is a separate, optional slice. This stays function-based on purpose: a class
  would only wrap one call.
- **`run_signals`** (`engine/portfolio.py:122-322`) gets three helpers extracted with no
  change to its signature: `_halt_positions`, `_validate_intraday_stops` and
  `_reject_overlaps`.

### 5.4 Shared plumbing (phase 2)

- **Database engine:** `utils/db.py` owns the cached engine and its cache variable, moved
  unchanged from `lab/persist.py`.
  - `lab.persist.get_engine` and `paper.persist._get_engine` stay as aliases, and setting
    `lab.persist.get_engine` on the module still takes effect.
  - The three database-address converters are not merged. `universe_db.to_psycopg_url`
    replaces only the first match, while `data/bars.py:185` and
    `cached_yfinance_loader.py:55` replace every match.
- **Paper constants:** `paper/constants.py` holds `TERMINAL_STATUSES` and the time zones.
  `TERMINAL_STATUSES` stays a tuple in today's order, and the old module names stay bound.
- **Logging:** `utils/logging_setup.configure()` is called once from the `ggt` CLI.
  - It attaches one handler to the `ggTrader` logger only, with `propagate = False`, and the
    handler looks up `sys.stderr` at each emit.
  - Third-party loggers are left alone. An invalid level falls back to INFO with a warning.
  - It makes no claim that logs are free of secrets. Today urllib3's retry warnings can
    include the Telegram URL with its token, and redacting that is a separate follow-up.
- **Scripts:** the three relative `sys.path.insert(0, "src")` lines are deleted. Two
  file-hash duplicates in `src/` delegate to `utils.manifest.sha256_file`.
  - Each script keeps its own `Path(__file__).parents[1]` root, because `utils.paths.REPO_ROOT`
    follows the installed package and can point at another checkout.
  - The `canonical_hash` and `git_sha` duplicates are not merged, because they behave
    differently.
- **Lint:** modern annotations (ruff `UP006`, `UP035`, `UP045`), plus bugbear and
  try-except-pass rules, are switched on for `src/` and the non-excluded `scripts/`, with
  per-file ignores for `research/`, `tests/` and the untouched scripts.
  - The flagged `zip` calls get an explicit `strict=False`, which keeps today's behavior.
  - Later rules, each switched on only when its hits are zero or listed as exceptions:
    - `ARG001`/`ARG002` (unused arguments) after the Phase 3 clean-up;
    - `G004` (f-string in a logging call) after logging is in place;
    - `ANN001`/`ANN201` (missing annotations) and `D100`-`D104` (docstrings, Google
      convention, skipping `D107`) after Phase 4;
    - `PLR0913` (`max-args = 8`) after Phases 3 to 5;
    - `C901`/`PLR0912`/`PLR0915` (`max-complexity = 15`) after Phase 5, listing the survivors;
    - `T201` (`print`) for `src/ggTrader/` only, exempting the CLI modules;
    - `DTZ` (naive datetimes) for `src/ggTrader/` once `cmd_db.py:173` is fixed, with one
      documented exception in the tape writer.
    - Not enabled: `BLE001` (45 of its 58 hits in `trader.py` are deliberate fail-soft
      boundaries) and `PLR2004` (too noisy for numeric code). The import-direction rule stays a
      test, because ruff cannot express it.

### 5.5 Clean-ups (phase 1)

- **Untradable branch:** delete the dead skip-and-continue code and fix the
  `paper/untradable.py` docstring (owner decision 1). `is_untradable_error` stays, because
  tests use it.
- **Config keys:** keep `STOCK_BASE_CONFIG` and `CRYPTO_BASE_CONFIG` exactly as they are,
  because command paths copy them into run records. Document their unread keys instead, and
  remove only the unused `names` parameter of `_select_live_params`.
- **Stale text:** fix `docs/architecture.md:246-299`, which describes a `Strategy` protocol
  that no longer exists, plus stale comments and help text.
- **Wipe commands:** `ggt db truncate` requires `--yes-wipe-ohlcv` and `ggt db clean` requires
  `--yes` (owner decision 2). Without the flag, the command prints what would be deleted and
  exits with code 2 before any database connection.

### 5.6 Left alone on purpose

- `engine/` as modules of functions.
- `lab/strategies/`: the contract, `ParamGrid` and the registry are already a good, small
  design.
- The `lab/wfo/` package layout; only `run.py` needs work.
- The membership family, which is the model for everything else.
- The pure paper modules (`cash_sweep`, `catastrophe_stop`, `gates`, `split_check`,
  `dividend_check`, `untradable` and `RiskGuard`). Studies import their signatures.
- The long "why" comments in `trader.py`; they record incidents and must move with their
  code.
- The four hand-rolled repo roots in `src/`. They are not equal to `utils.paths` outside an
  editable install.
- The `ggTrader` package name, the 193-line flat argparse builder, and `scripts/study.py`.

## 6. Safety and verification

How each phase proves it changed nothing. These are the outcome of the reviews; the plan's
global constraints implement them.

1. **Offline by construction.** The real boundary is that no process can build a database
   address or hold a service key. `utils/config.py` raises when no `DB_*` or
   `POSTGRES_CONNECTION_STRING` is set, and there is no default address.
   - Work happens in a separate worktree with no `.env`.
   - Every command runs through `offline.sh`, which scrubs the environment and keeps only
     `HOME`, `PATH`, the locale, the time zone and `XDG_RUNTIME_DIR`. The last is needed for
     the `systemd-run` memory cap in `run_tests.sh`; it is not a credential.
   - Task 0 checks that building a database address fails.
   - A process-level network block is not available on this host: unprivileged user
     namespaces are restricted, and `IPAddressDeny` has no effect on user scopes.
2. **An in-process guard as a second layer.** It blocks Python socket connects, hostname
   lookups, `psycopg2.connect` and curl_cffi's `Curl.perform`. It records each attempt and
   fails the test on exit, so a caller that swallows the error still fails.
3. **Trader transcripts: 86 scenarios recorded before any trader code moves.** (The first 38
   were extended after an independent review found that regressions could pass; the added
   scenarios cover startup recovery, legacy pending orders, polling time, notifier failures,
   and the branches that phases 1 and 2 edit.) Every later
   trader slice must reproduce them byte for byte.
   - Patches go only on the `trader` module's names. The harness never patches the global
     `time.time`, the pattern behind the 2026-09-22 hang.
   - Collaborators are faked:
     - the broker gives per-call replies, holds orders that exist before the run, and keeps
       one status cursor per order;
     - the run lock is a context manager with `verify()`;
     - the membership source is a serializable token;
     - an in-memory intent store stands in for the intent ledger;
     - a recording notifier runs the real message formatting.
   - Nothing varies between runs: uuids and the clock are pinned, arguments are deep-copied at
     call time, non-finite numbers are tagged, and two hash seeds are checked in child
     processes.
   - A mutation check proves the recordings can fail.
4. **A price-loader characterization test** with all four data sources faked, run under the
   guard.
5. **Oracle comparison against a baseline from the untouched base commit.** The comparison is
   not against "IDENTICAL" verdicts: 16 of the 26 committed p5 comparison reports say
   `DIFFERENCES - classify`, the accepted outcome of the vectorbt rebuild, and they must stay
   exactly as they are.
   - The runs require exit code 0, the 13 run results with `ok: true`, every comparison report
     and equity file, and p7's 120 sessions.
   - The comparison uses exact types and ignores only named timing and provenance fields.
   - At about 100 seconds for p5 and 31 for p7, it runs after every task that changes code.
6. **Test-outcome comparison.** Each node's result is compared with the baseline's JUnit
   record:
   - every test that passed still passes;
   - every skip is a known skip with the same reason;
   - the real-tape and live-golden tests pass rather than skip.

   The frozen price tape is copied into the worktree from `/home/flynn/ggTrader-vbt12`; its
   hash matches the manifest. The main checkout does not have it today, so those tests
   currently skip there.
7. **Study-surface and import-direction tests.**
   - Every `ggTrader` name a study imports still resolves.
   - `utils`, `data` and `engine` do not import `lab`, `paper` or `cli`, and `lab` does not
     import `paper`. Today's four exceptions are listed explicitly.
8. **Pins.** `verify_pins.py` passes with the counts in `.github/workflows/ci.yml:147`. The
   one `MATCH` pin, `cached_yfinance_loader.py`, changes with the lint task, which updates the
   CI counts in the same commit.

## 7. Phased roadmap

| Phase | Content | Proof | Plan |
|---|---|---|---|
| 0 | Safety nets: worktree and offline wrapper, comparers and baseline, guard, study-surface and import tests, trader transcripts, loader characterization | baselines recorded; new tests pass on today's code | phase 0-2 plan, tasks 0-5 |
| 1 | Dead branch, unused parameter, stale text, wipe flags | transcripts, oracles, test outcomes | tasks 6-9 |
| 2 | Engine cache home, paper constants, scripts, logging, lint rules | same, plus pins | tasks 10-14 |
| 3 | `run_signals` helpers; `run_wfo` dataclasses and helpers; optional exception return | engine goldens; oracles identical | later plan |
| 4 | `SubmittedOrder`, `RunResult`, `PaperLedger`, protocols and `TypedDict`s, `OhlcvFetchChain`, `TraderConfig` | transcripts; characterization; live golden | later plan |
| 5 | Split `PaperTrader._run` one step object at a time; migrate the trader tests to the in-memory ledger; delete the adapter and `_run_legacy` | transcripts; a seeded property test of planning against the old method | later plan |
| 6 | Optional tidy-ups: membership CLI dispatch, weekly-job helpers, `verdict_register` validator class, persist transactions | existing tests | later plan |

Phases 0 to 2 are small and mostly mechanical. Phase 5 is the expensive one. Its largest item
is rewriting the 4,220-line trader test file, which can be spread out because the adapter keeps
the old tests valid meanwhile.

## 8. Owner decisions

A default is not an approval. Tasks 6 and 9 wait for a recorded answer.

**Recorded 2026-11-17 (owner):** 1 yes (delete the dead branch); 2 yes (wipe flags);
3 keep the unused writers; 4 both (clarified 2026-11-18): the Telegram total line shows the percentage
against the day's starting value and against starting capital, a separate small change
outside phases 0 to 2; 5 not yet: no execution until the owner says so.

1. **Dead untradable branch.** Delete it? The trader keeps behaving exactly as now: an
   exception raised on submit ends the run. *Recommended: yes.*
2. **Wipe commands.** Require an explicit flag? This is a deliberate behavior change.
   *Recommended: yes.*
3. **Unused database writers** (`log_trade`, `log_pending_order`, `clear_pending_order`, the
   lab plan and sweep writers): keep for now, or delete in a later slice? *Recommended: keep.*
4. **Telegram "total" percentage.** Should it be measured against the day's starting value
   (today) or against starting capital? Answered "both" on 2026-11-18. Not part of phases 0 to 2: it changes the message text, so it
   needs its own slice with the transcript goldens regenerated and reviewed, after phase 0.
5. **Execution.** Delegation under `CLAUDE.md` roles. Builders: Sonnet for the transcript and
   loader tests, then Terra, Luna or Sonnet. Reviewers: Sol and Opus, never reviewing their
   own model family's work. Both review the whole branch before merge. The owner reported
   ample quota on 2026-11-17.

## 9. Review history

| Review | Model | Main contribution |
|---|---|---|
| Code review | Claude Fable | Findings, the target designs in section 5, and the original phased plan |
| Plan review 1 | Claude Fable | The guard missed psycopg2 and curl_cffi; the frozen tape was missing; a failed oracle run could pass; the worktree had no pytest; oracle and legacy scripts were being edited; a CI pin count would have broken |
| Plan review 2 | Gemini 4 Argon | The run lock is a context manager; the membership object must be serializable; notifier arguments must be recorded; one order-status cursor per order; `peak_rss_mb` appears at the top level of the compare files; two modules had no logger. About half its claims were wrong and were not applied: calls the trader does not make, and tasks that do not exist |
| Plan review 3 | GPT-6.2 Astra | Compare against a base-commit baseline because 16 reports say `DIFFERENCES`; credentials, not markers, are the boundary; the guard must fail on swallowed attempts; per-node test outcomes; per-call broker replies and expected outcomes; the converters differ; the legacy runner breaks on a hash alias; keep the config dicts |

**Original proposals the reviews overturned:**
- `utils/significance.py` does not import the engine; it imports `curve_summary`, and stays
  the one layer exception.
- `utils/db_engine.py` has no URL rewrite and stays as it is.
- The URL converters are not merged.
- `canonical_hash` and `git_sha` are not merged, and scripts keep their own repo roots.
- The config dict keys are kept.
- The four `src/` repo roots stay.
- The oracle check compares against a baseline, not against "IDENTICAL".
- Logging touches only the `ggTrader` logger.

The full review reports were working files in the session scratchpad and are not kept. This
section is their record.
