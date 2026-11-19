# Refactor phase 4 (paper trader interfaces and typed records, the price-loader chain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the paper trader typed records and swappable interfaces, and give the price-loader
fallback one tested home, without changing one order, one Telegram line, one SQL statement or one
price row. `PaperTrader._run` is not split here (that is Phase 5); this phase builds every part
Phase 5 needs and proves each one inert.

**Architecture:** Same method as Phases 0-3: the safety nets are written first (Tasks 31-33),
then each slice is one small commit that must reproduce the nets exactly. The nets already in the
worktree are the 96 transcript goldens, the 28-test characterization of `fetch_stock_ohlcv`, the
live golden, the connection guard, the study-surface test and the import-direction rules. Phase 4
adds two nets they cannot supply: a recording-engine golden of every SQL statement and bind
dictionary in `paper/persist.py`, and pins for the tuple-index reads, the result-dict key order
and the environment-read timing. The order of work is: `SubmittedOrder`, `RunResult`, records and
ports, the ledger (protocol, three implementations, the string-patch adapter, `PgPaperLedger`),
the trader's switch to the ledger, the fetch chain, and `TraderConfig`.

**Tech Stack:** Python 3.12 (uv-managed `.venv`), pytest through `scripts/run_tests.sh`, ruff,
SQLAlchemy 2 (`text()`, `Engine.connect()`), psycopg2, pandas 3, `typing.Protocol` and
`TypedDict`.

**Spec:** [docs/superpowers/specs/2026-11-17-refactor-design.md](../specs/2026-11-17-refactor-design.md),
sections 3 (principles), 5.2 (interfaces and typed records), 6 (verification) and the phase 4 row
of section 7. The tools this plan reuses by reference are defined in the Phase 0-2 plan
([2026-11-17-refactor-phase-0-2.md](2026-11-17-refactor-phase-0-2.md)): the worktree and
`offline.sh`, the oracle check, the test-outcome check (two JUnit files per side), the verify-pins
check and the comparers. Phase 3 ([2026-11-18-refactor-phase-3.md](2026-11-18-refactor-phase-3.md))
is a sibling: it touches `engine/` and `lab/wfo/` only, so the two phases do not share a source
file (they share `docs/changelog.md` and `docs/architecture.md`; Task 44 resolves that). Task
numbers continue from Phase 3 (it ends at Task 29).

**Revision 1 (2026-11-18)** is the first draft. Every line number was read on 2026-11-18 from the
worktree `/home/flynn/ggTrader-refactor` (branch `refactor/phase-0-2`, HEAD `c509d7c1`, Phases 0
and 1 committed; Phase 2 Tasks 10-14 were mid-edit and uncommitted: `utils/db.py`,
`utils/logging_setup.py`, edits to `paper/persist.py:13`, `lab/persist.py`, `data/live/yfinance_loader.py`
and `tiingo_loader.py`). Phase 2 will move lines in `trader.py` (Task 11 constants, Task 14
annotations and one `except` at `trader.py:436`) and in `persist.py` (Tasks 10, 11, 14). Treat
every line number as approximate and re-grep the anchor string given beside it; Task 30 Step 4
records the new numbers.

**Revision 2 (2026-11-18)** applies the independent Codex Sol review; each item was checked against
the worktree before it was applied. Blocking: `persist_function_names()` returns 26 names including
`paper_run_lock` (Task 32 asserts 26, and separately that the ordinary wrapping covers the other 25);
the SQL reference golden is immutable and compared through one explicit `conn`-dropping normalization
(Task 38); the chain-test hash gets one authorized, mechanical re-record in Task 40 (the hash count is
100: 96 goldens plus `harness.py`, `scenarios.py`, `test_trader_transcripts.py` and the chain
test); the parallel worktrees are created after Task 36's commit; the in-memory-ledger test keeps the
non-ledger collaborator patches and the Phase 5 claim is narrowed; `init_paper_schema` has eleven
statements. Important: modeled recorder events and execution-failure cases for every mutator,
variadic-notifier signature handling, the per-run settings snapshot documented as an intended
deviation, Task 45 enumerated, logger identity preserved. Minor: counts and anchors corrected, `--fast`
collection check, Deviation 2 reworded. The evidence directory is `/home/flynn/ggTrader-refactor-evidence`
(the earlier `/tmp` copy was deleted by a reboot), with the gate scripts copied for the Phase 4 worktree.

## Scope

**In this plan:** design sections 5.2 and the phase 4 row, in this order of work.
1. `SubmittedOrder` replacing the five-item tuple read by index (`paper/models.py`, `trader.py`).
2. `RunResult` with `as_dict()` (`models.py`, ten return sites in `trader.py`).
3. Records `Account`, `Position`, `OrderStatus`, `MarketClock`, `SleeveSignals`, `SignalBundle`
   and the `Broker` and `Notifier` protocols (`paper/models.py`, `paper/ports.py`).
4. The `PaperLedger` protocol; `InMemoryLedger`, `ReadOnlyLedger`, `ModuleLedger`
   (`paper/ledger.py`); `PgPaperLedger` with the module-level functions of `paper/persist.py` kept
   as one-line delegations; the trader's switch from `_write` and 26 imported functions to
   `self._ledger`.
5. `OhlcvFetchChain` and `fetch_stock_ohlcv` in `data/live/fetch_chain.py`; a public
   `CachedYFinanceLoader.persist()`; `lab.data.fetch_stock_ohlcv` as a thin wrapper; `paper/`
   importing its fetch path from `data/live`.
6. `TraderConfig` (`paper/config.py`) replacing the environment reads in `trader.py`.
7. The characterization tests that make those changes provable (Tasks 31-33), the docs (Task 44)
   and one optional, owner-gated task (Task 45).

**What Phase 4 does NOT do** (Phase 5 or later, or never):
- **It does not split `PaperTrader._run`** (`trader.py:716-1536`, 820 lines). No `PreflightGate`,
  `PortfolioView`, `RiskAssessor`, `planning.py`, `OrderExecutor`, `CorporateActions` or
  `RunReporter`. The ten `return` sites keep their places; they only wrap a `RunResult`.
- It does not rename or move any `PaperTrader` method: tests assign and patch `_submit_order`,
  `_reconcile_pending_orders` and `_compute_split_corrections` (`tests/paper/test_trader.py:2044,
  2222, 4005, 4194`), `_risk` (`:389-543`) and `_dry_run` (`:1965`). `_run(self, *, stop_threshold)`
  keeps its signature.
- It does not migrate `tests/paper/test_trader.py` (4,220 lines, about 440 string patches: 438 dotted `ggTrader.paper.trader.*` literals) to the
  in-memory ledger, and it does not delete the string-patch adapter (Phase 5).
- It does not change the SQL, the transaction shape or the bind parameters of any persist
  function (Deviation 1). The one exception is the stray `conn` key of `create_order_intent`
  (Deviation 5).
- It does not touch `RiskGuard`, `config_from_env()` or `RiskConfig` (read at construction time,
  Deviation 3), the pure paper modules (`cash_sweep`, `catastrophe_stop`, `gates`, `split_check`,
  `dividend_check`, `untradable`), `lab/negative_cache.py`, anything under `research/`, or the
  oracle and legacy scripts. It does not change what `ggt paper` prints (`cli/main.py:75-100`
  reads `result["buys"]`, `["sells"]`, `.get("untradable")`, `.get("skipped_writes")`,
  `["errors"]`).
- It does not make any `ggt paper --live` call, touch cron or the `ggtrader_live` container, or
  deploy. Code reaches the container only through the image rebuild described in
  `ggtrader-deploy`; the trader stays off and cron returns only on the owner's word.

**What must be true when Phase 4 ends, so that Phase 5 can start** (Task 44 checks each line):
1. `trader.py` reaches the database only through `self._ledger`; it has no `_write`, no call to an
   imported persist function and no `os.environ` access (`grep -n "os.environ\|getenv"
   src/ggTrader/paper/trader.py` prints nothing). It still binds the 26 persist names at module
   level (the adapter and the harness need them).
2. `PaperTrader.__init__` takes `broker: Broker`, `notifier: Notifier` and a keyword-only
   `ledger: PaperLedger | None = None`; Phase 5 tests can build a trader on `InMemoryLedger` and need no
   persist patch. This is **ledger-independent testing only**: the signal, membership, benchmark-refresh,
   date, uuid and clock collaborators are still patched in the trader module (the harness does that, and
   Phase 5 may inject them later); no patch-free whole-trader harness is claimed.
3. Every pending order is a `SubmittedOrder`; every result is built as a `RunResult` and returned
   as `.as_dict()` (`run()` still returns `dict`).
4. The run-time settings are one frozen `TraderConfig` held as `self._cfg` for the whole run.
5. `models.py` holds all eight records, `ports.py` the two protocols, `ledger.py` the protocol and
   its three implementations. The module docstrings say which names are public.
6. The transcript harness and goldens are byte-identical to Task 30's record (apart from the
   scenarios Task 32 added), `paper/` imports nothing from `lab.data` but `STOCK_BASE_CONFIG`, and
   `fetch_stock_ohlcv` has one implementation.
7. `docs/architecture.md` names these pieces, and the anchors for `_run` (line range, the ten
   return sites, the local names the steps share) are re-measured in the Task 44 hand-back.

## Owner decisions this plan needs before execution

A default is not an approval. Nothing in this plan runs until the owner says so.

- **O-P4-1, persist transactions (Task 45 only).** The design says `PgPaperLedger` runs "always
  inside `engine.begin()`". Eleven of the fifteen write functions today use `engine.connect()` plus an explicit
  `conn.commit()`; four use `begin()`. Switching changes what a recorded session looks like
  (`COMMIT` instead of `ROLLBACK` on `log_trade`'s early return, and the connect-based mutators' tests
  would have to be rewritten: 38 test functions in `test_paper_persist.py` configure
  `.connect.return_value`, 12 of which exercise those mutators). The design's own roadmap lists "persist transactions"
  under Phase 6. *Default: skip Task 45; Phase 4 keeps each function's transaction shape.*
- **O-P4-2, quota.** The agent cannot see plan usage. Before Task 30 the owner is asked for
  current headroom. If it is unknown, run Phase A (Tasks 30-33) and Phase B (Tasks 34-36) only and
  report. Usage is reported after each phase (Execution roles).

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

Everything in the Phase 0-2 plan's "Global Constraints" applies and is not repeated: offline by
construction (`offline.sh`), tests only through `scripts/run_tests.sh`, no real connection, never
`ggt paper --live`, never read `.env`, no edit under `research/`, `ohlcv.timestamp` stays naive
UTC, ruff clean, commits end with the attribution line from the session instructions. Phase 3's
conventions are reused: goldens regenerate only with an environment variable, never silently, and
every extraction shows its diff with `--color-moved`. What Phase 4 adds:

- **Starting point.** Branch `refactor/phase-4` is created from the head of `refactor/phase-0-2`
  after Phase 2's gate has passed (recorded as `BASE4`), in its own worktree
  `/home/flynn/ggTrader-refactor-p4` (Phase 3 uses the main worktree). It is independent of Phase 3:
  neither branch needs the other. Never commit to `main`, push or merge without the owner's word.
  The worktree is built as in Phase 0-2 Task 0 and Phase 3 Task 19 Step 5: `uv sync --frozen
  --extra test --extra dev`, `offline.sh` copied and excluded, the frozen tape copied and its hash
  checked against `data/snapshots/MANIFEST.json`, and the `get_db_connection_string()` check that
  must raise `ValueError: Database configuration not found`.
- **Evidence directory and gate scripts.** Every recorded artifact lives in
  `/home/flynn/ggTrader-refactor-evidence` (written `$E` below), not in `/tmp` (a reboot deleted the
  earlier copy). It holds the oracle baseline `$E/baseline/` (`p5/` and `p7.json`), the Phase 0-2
  test-outcome baselines `$E/baseline-core.xml` and `$E/baseline-studies.xml`, and the scripts
  `gate_oracles.sh LABEL`, `gate_tests.sh LABEL` (two junit files per run, `$E/LABEL-core.xml` and
  `$E/LABEL-studies.xml`), `gate_compare.sh LABEL` and `gate_run.sh LABEL`. Those scripts `cd` into
  `/home/flynn/ggTrader-refactor`, the worktree Phase 3 uses, so Phase 4 does not call them directly:
  Task 30 Step 3 makes copies in `$E/p4/` (`gate_oracles.sh`, `gate_tests.sh`, `gate_compare.sh`)
  whose only changes are the `cd` target (a `WT` variable: the Phase 4 worktree, or a chain's build
  worktree), the label prefix `p4-` on every output file, and in `gate_compare.sh` the test baseline
  `$E/p4base-core.xml,$E/p4base-studies.xml` plus a `MUST_PASS` environment variable appended to
  `--must-pass`. The oracle half keeps comparing against `$E/baseline`. Wherever this plan says
  "the oracle check, label X" it means `$E/p4/gate_oracles.sh p4-X` then the oracle half of
  `$E/p4/gate_compare.sh p4-X` (exit 0); "the test-outcome check" means `$E/p4/gate_tests.sh LABEL` then
  the outcome half of `gate_compare.sh LABEL`.
- **Two baselines.** *Oracle baseline:* `$E/baseline` (recorded from the untouched Phase 0-2 `BASE`;
  Phase 0-2's gate proved it unchanged). *Test-outcome baseline:* `$E/p4base-core.xml` and
  `$E/p4base-studies.xml`, recorded in Task 30 on `BASE4`.
- **Hash records** (Task 30 Step 5; all under `$E`, all checked by gate step 2). `p4-fixed.sha256` holds
  the 96 transcript goldens and `harness.py` (97 lines) and is **never re-recorded**; Task 33 appends
  `persist_sql.json` and the goldens Task 32 added, and nothing else changes it. `p4-pins.sha256` holds
  `scenarios.py`, `test_trader_transcripts.py` and `tests/lab/test_fetch_stock_ohlcv_chain.py` (3 lines);
  Task 33 re-records the first two (Task 32 edits `scenarios.py` additively) and Task 40 re-records
  the third (the mechanical rename of Deviation 8), each with its diff shown. Total at Task 30: 100 lines
  (96 goldens plus 4 Python files).
- **The nets, by name** (used by every task below; the exact commands are in "The task gate"):

| Net | What it proves | Where |
|---|---|---|
| **T** transcripts | 96 scenarios of `PaperTrader.run`: every broker, persist, notifier, signal, lock, clock and uuid call with deep-copied arguments, the rendered Telegram text, the result dict and the raise | `tests/paper/test_trader_transcripts.py`, `tests/paper/golden/trader_transcripts/` (96 files), harness in `tests/paper/transcript/` |
| **C** chain characterization | 28 tests of `fetch_stock_ohlcv`: every provider request, every persisted frame and its order, every printed line, the two cache routes | `tests/lab/test_fetch_stock_ohlcv_chain.py` |
| **G** live golden | the live signal path over 120 sessions on the frozen tape (needs `data/snapshots/ohlcv_20261106.parquet`); it patches `signal_runner.fetch_stock_ohlcv` | `tests/paper/test_live_golden.py` |
| **N** connection guard | any socket, psycopg2, hostname or curl_cffi attempt fails the test, even if the code swallows it | `tests/netguard.py`, `tests/paper/conftest.py`, applied by `test_fetch_stock_ohlcv_chain.py` |
| **S** study surface | every `ggTrader` name a study imports still resolves (53 pairs) | `tests/utils/test_study_surface.py` |
| **I** import rules | layer directions (`utils`, `data`, `engine` import none of `lab`, `paper`, `cli`; `lab` imports no `paper`) with one exception (`utils/significance.py:49`) | `tests/utils/test_import_rules.py` |
| **O** oracle check | `p5` and `p7` results against `$E/baseline` | Phase 0-2 Global Constraints |
| **P** outcome check | per-node test outcomes against `p4base-*.xml` | Phase 0-2 Global Constraints |
| **V** verify-pins | pinned-file hashes, counts in `.github/workflows/ci.yml:147` | `scripts/verify_pins.py` |
| **L** SQL golden (new, Task 31) | every SQL text, bind dictionary, commit, rollback and close of the 31 persist functions | `tests/paper/test_persist_sql_golden.py`, `tests/paper/golden/persist_sql.json` |
| **K** shape pins (new, Task 32) | result-dict key order, every tuple-index site, environment-read timing | `tests/paper/test_phase4_pins.py` and added scenarios |

- **The transcript harness is part of the contract.** It patches only names in the
  `ggTrader.paper.trader` module namespace (`tests/paper/transcript/harness.py`, `_run_scenario`,
  and `persist_function_names()` at `harness.py:618`). The exact list, as of 2026-11-18:
  - *The 26 names `persist_function_names()` returns, found by inspection*: every callable in
    `vars(trader_module)` whose `__module__ == "ggTrader.paper.persist"`. That is the 25 below plus
    `paper_run_lock`. The ordinary-wrapper loop skips `paper_run_lock` (`harness.py:755-756`); the 25
    are each replaced by a wrapper: `attach_broker_order`,
    `clear_pending_order`, `clear_split_correction`, `create_order_intent`,
    `get_accrued_dividend_keys`, `get_earliest_snapshot`, `get_last_buy_dates`,
    `get_latest_snapshot`, `get_open_split_states`, `get_peak_value`, `get_pending_orders`,
    `get_snapshot_history`, `get_total_dividend_accrual`, `get_trade_history_dates`,
    `get_unresolved_order_intents`, `init_paper_schema`, `log_dividend_accrual`,
    `log_pending_order`, `log_snapshot`, `log_trade`, `mark_pending_order_stale`,
    `save_peak_value`, `save_split_correction`, `settle_legacy_pending_order`,
    `settle_order_intent`. Each replacement is a wrapper whose `__name__` is the function's name
    (because `_write` records `fn.__name__`, `trader.py:174`).
  - *The run lock*: `paper_run_lock` (the 26th name), replaced separately by a context manager
    yielding one `FakeRunLock`.
  - *Eight other names*: `generate_core_signals`, `refresh_benchmark_tape`,
    `live_membership_source`, `_today_et`, `_now_pt`, `uuid4`, `time` (a fake clock object) and
    `datetime` (a pinned subclass).
  - *Not patched*: `require_live_membership_spec` (the real live-membership guard runs),
    `AlpacaBroker`, `TelegramNotifier` (the harness passes its own fakes into the constructor).
  - *Also relied on*: `PaperTrader(broker, notifier, dry_run=, read_only=, membership=)`,
    `trader.skipped_writes` (read after the run), `trader.run()`, and `ENV_VARS`
    (`harness.py:~38`), kept whole by `test_env_vars_cover_every_variable_the_paper_package_reads`
    (`test_trader_transcripts.py:1252`, a grep of `paper/*.py` for variable names).

  **Rule for every task below that moves or renames anything in the trader module:** if the
  harness needs a change, make it in the same commit and show the 96 goldens byte-identical with
  `sha256sum -c $E/p4-fixed.sha256`. The expected harness diff for Tasks
  34-43 is *none*; a needed harness change is reported as a finding before it is made.
- **Names that must stay importable** (S and the existing tests check these):
  - `ggTrader.paper.persist`: all 31 public functions with unchanged signatures, `PaperRunLock`,
    and the module name `_get_engine` (17 dotted-string patches plus one `monkeypatch.setattr` (`tests/paper/test_paper_persist.py:655`), read at
    call time). The study `research/.../S3-evidence/paper_persist_ro_smoke.py` imports the module.
  - `ggTrader.paper.trader`: `PaperTrader`, `run_paper_trading`, `LIVE_MEMBERSHIP_SPEC`, the 26
    persist names, `generate_core_signals`, `refresh_benchmark_tape`, `live_membership_source`,
    `require_live_membership_spec`, `_today_et`, `_now_pt`, `uuid4`, `time`, `datetime`, `cash_sweep`,
    `catastrophe_stop`, `AlpacaBroker`, `TelegramNotifier`, `DryRunNotifier` (tests patch every one
    of these by dotted string; the count of such patches is about 440: 438 dotted literals in `tests/paper/test_trader.py`).
  - `ggTrader.paper.signal_runner.fetch_stock_ohlcv` and `ggTrader.paper.overlay.fetch_stock_ohlcv`
    as module attributes called by global name: `tests/paper/test_live_golden.py:83` and
    `scripts/p7_live_golden.py:81` (never edited) patch the first, and about 30 tests in
    `test_signal_runner.py` and `test_trader.py:3326-3399` patch it.
  - `ggTrader.lab.data.fetch_stock_ohlcv` (same signature and result), `STOCK_BASE_CONFIG`,
    `CRYPTO_BASE_CONFIG`, `load_ohlcv`, `collapse_daily_duplicates`, `equity_universe_between`,
    `rebalance_dates`, `eligible_at` (about 30 study files import `lab.data`).
  - `ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader._cache_to_db` (stays the
    implementation; `persist()` delegates; `tests/data/test_cached_yfinance_loader.py` patches it
    on instances at `:141-269`).
- **No new module-level state.** The only module-level objects Phase 4 adds are constants and one
  lazily created, stateless `PgPaperLedger()` instance in `persist.py`. The engine is looked up
  through `_get_engine()` at each call, never cached by a ledger object.
- **Layer rules** (design section 6.7). `data/live/fetch_chain.py` imports nothing from `lab`,
  `paper` or `cli`; the layer exceptions list (`LAYER_EXCEPTIONS` in
  `tests/utils/test_import_rules.py`, one entry after Phase 2: `utils/significance.py:49`) neither
  grows nor shrinks. Task 42 adds one rule (below); no rule is weakened.
- **Heavy runs one at a time.** Two builders may work in two worktrees at once; the oracle check
  and the `--full` gate are run by the orchestrator one after the other (the 2026-09-16 host wedge).
- **Synthetic data only.** Delegate prompts, scenarios and goldens contain no private account
  records; scenario figures are the existing synthetic ones.

**The task gate** (every code task, Tasks 34-43):
1. `.venv/bin/ruff check . && .venv/bin/ruff format --check .` clean.
2. Net T and the hash records: `./offline.sh scripts/run_tests.sh tests/paper/test_trader_transcripts.py`
   passes with no golden regenerated, then `sha256sum -c $E/p4-fixed.sha256 $E/p4-pins.sha256` prints
   OK for every line (the 96 goldens and `harness.py` are in the first file and never change), and
   `git diff --stat -- tests/paper/transcript/harness.py tests/paper/golden` is empty (apart from
   Task 32's additions, which Task 33 appends to the hash records).
3. The neighbourhood run: `./offline.sh scripts/run_tests.sh tests/paper tests/lab tests/data tests/utils`.
   Path arguments run every test in those folders, including S and I, and the paper tests under N.
4. The pin check of this phase: `./offline.sh scripts/run_tests.sh tests/paper/test_persist_sql_golden.py
   tests/paper/test_phase4_pins.py` (L and K).
5. Net G: `./offline.sh scripts/run_tests.sh tests/paper/test_live_golden.py` (passes, never skips:
   the tape is in the worktree). Net C: `tests/lab/test_fetch_stock_ohlcv_chain.py`, included in
   step 3 and named in each fetch task.
6. O, with the task's label (the copies in `$E/p4/`, Global Constraints), exit 0 against `$E/baseline`. It is cheap
   (about 130 s) and the oracle drivers import `paper.signal_runner`, `paper.overlay`,
   `paper.live_signals` and `lab.data`, so an import-time slip in any task shows here.
7. V when the task edits `cached_yfinance_loader.py` (Task 40), and at the phase gate.

**The phase gate:** P against `p4base-*.xml` with the tests of Tasks 31, 32, 34-43 named in
`--must-pass` (and Task 45's if it ran); O (label `phase4`); V with the counts in
`.github/workflows/ci.yml` as Phase 2 left them (updated in Task 40's commit only if the
classification of `cached_yfinance_loader.py` changed); ruff clean; both hash records all OK;
both reviews.

## Deviations from the design

Each was checked against the code. None changes the design's intent.

1. **`PgPaperLedger` keeps each function's transaction shape** (design 5.2: "always run inside
   `engine.begin()`"). Evidence: of the 15 writing functions, `create_order_intent`,
   `attach_broker_order`, `settle_order_intent` and `settle_legacy_pending_order` use
   `engine.begin()` (`persist.py:121,159,214,375`); `init_paper_schema`, `log_trade`,
   `log_pending_order`, `clear_pending_order`, `mark_pending_order_stale`, `log_snapshot`,
   `save_peak_value`, `save_rebalance_state`, `log_dividend_accrual`, `save_split_correction` and
   `clear_split_correction` use `engine.connect()` and an explicit `conn.commit()`; none forgets the
   commit. `log_trade` returns early without committing when the trade already exists
   (`persist.py:339-340`), which under `begin()` would become a `COMMIT`. The run lock
   (`paper_run_lock`) needs a plain `connect()` (session advisory lock). In
   `tests/paper/test_paper_persist.py`, 38 test functions configure `.connect.return_value` (many are read
   mocks), and 12 of them exercise the connect-based mutators. The design's roadmap
   (section 7, Phase 6) already lists "persist transactions". Phase 4 therefore records and keeps the
   shape (Task 31 records `begin`, `commit`, `rollback`, `close` events); Task 45 is the optional
   conversion.
2. **`PgPaperLedger` lives in `paper/persist.py`, not a new module.** The design is silent on the
   location. Keeping it there is the simplest way to preserve two things, not the only way: the
   17 dotted-string patches plus one `monkeypatch.setattr` (`tests/paper/test_paper_persist.py:655`) of `ggTrader.paper.persist._get_engine` resolve the name in that module at call time, and the
   harness finds the persist functions by `__module__ == "ggTrader.paper.persist"` (`harness.py:618`). A
   class in another module would need an engine accessor that looks `persist._get_engine` up at call
   time, and delegations still defined in `persist.py`; Phase 4 does the simpler thing.
3. **`TraderConfig` holds the run-time settings only; the risk limits stay `RiskConfig`.** The
   design's table cites the environment reads at `:712-714, :881-882, :898, :1209, :1371-1373`
   (the stop, the sweep and the untradable set), not `config_from_env()` at the constructor
   (`trader.py:160`). Tests read `MAX_POSITIONS` at construction time and replace `trader._risk.cfg`
   (`test_trader.py:386-426`, `3015-3020`), so that read keeps its time and place. `from_env()` is
   called once at the top of `run()` (not in `__init__`), because tests set the variables after
   constructing the trader (`test_trader.py:1957-1959, 2229`), and the transcript harness sets them
   before construction; both work only if the read is at run time. The `dry_run` and `read_only`
   flags (constructor arguments, not environment) are carried in the record.
4. **Run-time settings become one per-run snapshot (an intended deviation: read timing moves), and
   three float settings keep their fail time.** Today the reads are spread through the run: the stop
   settings at the start of `run()` (before the lock, `trader.py:698-706`), `sweep_enabled()` and
   `sweep_symbol()` after the account and positions are loaded (`:867-868`), `configured_untradable()`
   after that (`:884`), `reserve_pct()` twice (`:1194` and `:1356`), and the two other floats inside the
   sweep-buy `try` (`:1357-1358`). With one snapshot at the top of `run()` the sweep and untradable reads
   move **before** the lock, the market gate, signal generation and the account reads, and
   `reserve_pct()` is read once. The only observable difference is if the process environment changes
   during a run, which nothing in the code does; every early exit still returns the same result,
   because a malformed float is stored, not raised, and `configured_untradable()` and `sweep_symbol()`
   never raise. Task 43 tests the early exits and repeated runs and does not claim the read timing is
   unchanged. The fail time is kept: `SWEEP_CASH_RESERVE_PCT` is parsed outside any
   `try` (`trader.py:1194`, an error ends the run), but `SWEEP_MIN_CLIP_USD` and
   `SWEEP_BUY_TRIGGER_PCT` are parsed inside the sweep-buy `try` (`trader.py:1351-1362`, an error
   is logged and the buy skipped). Reading all of them once at the top would turn the second kind
   into a fatal error at the start of the run. `TraderConfig` therefore stores a parse error and
   re-raises it at the call site (Task 43). The design does not mention this; it is a consequence of
   "read once".
5. **`create_order_intent` passed `locals()` to SQL**, which includes the open connection under
   the key `conn` (`persist.py:127`, evaluated inside `with ... as conn`). SQLAlchemy ignores the
   extra key. The explicit dictionary (design 5.2) drops it, so the recorded bind dictionary of that
   one function changes by that one key; the compared SQL text and the other seven keys are
   identical. Task 31 records the stray key on purpose; the reference golden is never regenerated,
   and Task 38 compares the new code against it through one explicit normalization that drops `conn`.
6. **The negative cache is injected into `OhlcvFetchChain` as a protocol, not imported.**
   `lab/negative_cache.py:19` imports `ggTrader.lab.persist`, and `data/` must not import `lab`
   (rule I). The lab wrapper supplies an adapter that looks the module functions up at call time,
   which is also what `tests/lab/test_fetch_stock_ohlcv_chain.py` patches.
7. **`SubmittedOrder` is a frozen dataclass, not a `NamedTuple`.** A `NamedTuple` would still unpack
   by position, which is the habit being removed (`trader.py:1268, 1388`). No test constructs the
   old tuples or calls `_poll_orders` directly (checked: no match in `tests/`), so nothing needs the
   tuple behavior.
8. **`CachedYFinanceLoader.persist()` costs three mechanical test edits.** The chain calls the
   public method, so `tests/lab/test_fetch_stock_ohlcv_chain.py` (the `_cache_to_db` mocks),
   `tests/paper/test_signal_runner.py:342` and the characterization helper rename one token. Task 40
   does the rename before the chain is built, so Task 42 is held to an unedited net.
9. **The design counts 86 transcript scenarios; there are 96** (nine added after review and one for Telegram,
   `tests/paper/transcript/scenarios.py:24`, `EXPECTED_SCENARIO_COUNT = 96`). Task 32 adds a few
   more; Task 33 re-records the count and the pin hashes (the 96 original goldens stay in `p4-fixed.sha256`).
10. **`paper/` keeps one import from `lab.data`.** The design says `paper/` imports "its data path"
    from `data/live`, not `lab`. `overlay.py:18` also imports the config constant `STOCK_BASE_CONFIG`
    from `lab.data`; that is not a data path and stays. Task 42's rule allows exactly that name.
11. **`ReadOnlyLedger` treats every mutating method as a write**, not only the six that `_write`
    wraps today (`init_paper_schema`, `clear_split_correction`, `log_dividend_accrual`,
    `save_peak_value`, `save_split_correction`, `log_snapshot`). The other mutators are unreachable
    in a read-only run (the intent calls need `not self._dry_run`, `_reconcile_pending_orders`
    returns before booking when read-only, `trader.py:372-375`), and a read-only run must never write.
    Task 39's tests show both facts.

## Review Focus

1. **A pin that cannot fail.** Task 33's mutation step must make the SQL golden, the shape pins
   and the transcripts fail for: one character of any SQL text, a bind key renamed, a bind value
   cast, `commit` dropped from one function, `begin()` swapped for `connect()` in one, a tuple
   index read changed at each of the ten sites, a result key reordered, and a float setting read at
   the wrong time.
2. **The string-patch contract.** The about 440 patches in `test_trader.py` and 96 goldens must pass
   with no edit. A name patched *after* the trader is constructed (many tests) must still win:
   `ModuleLedger` resolves the name on the `trader` module at each call. Task 37 tests this for all
   26 names, including a patch applied between two calls.
3. **Read-only and exception behavior.** `ReadOnlyLedger` must record exactly the names the six
   `_write` sites record today (the `dry_run_read_only` golden lists them), and must not swallow or
   wrap a ledger exception: the order recovery paths (`_reconcile_order_intents`, `_submit_order`'s
   `stage` labels) depend on which call raised.
4. **The fetch chain's differences from itself.** The three copy-pasted blocks differ in when
   they announce, when they construct, which errors they swallow and what they print (table in
   Task 41). A tidy loop would merge them. The 28 tests and Task 41's own cases pin each.
5. **Environment timing.** Which setting fails when (see Deviation 4); that the snapshot is taken at the
   top of `run()` and not at construction; that an early exit (market closed, stale signal, lock failure)
   with a malformed sweep float still returns its normal result; and that a second `run()` on the same
   trader after an environment change reads the new values.

## Risks found in the code, and the check that catches each

| # | Risk | Where | Check |
|---|---|---|---|
| 1 | The harness finds persist functions by `__module__`; a delegating function defined elsewhere, or a name no longer bound in `trader.py`, makes it patch nothing, and real persist code runs | `harness.py` `persist_function_names()` | Task 32 `test_harness_finds_the_26_persist_names` and `test_harness_wraps_25_names_and_replaces_the_lock_separately`; net T (every `persist.*` call would vanish from the goldens); the guard N and the missing database settings stop any real call |
| 2 | After Task 39 the 26 imported names are referenced only through `getattr`, so ruff `F401` autofix (Phase 2 Task 14 ran it) deletes them | `trader.py:18-45` | each import line keeps `# noqa: F401 - looked up by ModuleLedger at call time`; Task 39 `test_trader_module_binds_the_26_ledger_names`; the neighbourhood run |
| 3 | `locals()` bind dictionary carries `conn` | `persist.py:127` | L records it; Task 38 compares the bind dictionary restricted to the names in the SQL text and asserts the removed key is exactly `conn` |
| 4 | Transaction shape: `connect()` with no commit on an early return rolls back; `begin()` would commit | `persist.py:339-340` and 14 others | L records `begin`, `commit`, `rollback`, `close` per case; Task 38 keeps the shape |
| 5 | Tests swap `persist._get_engine`; an engine cached by a ledger object, or a second import path, ignores the swap | 17 dotted-string patches plus one `monkeypatch.setattr` in `test_paper_persist.py` | `PgPaperLedger` calls the module global `_get_engine()` at each call; the existing tests; Task 38 `test_engine_is_looked_up_at_call_time` |
| 6 | Validation runs before the engine is touched (`settle_order_intent`, `settle_legacy_pending_order` raise `ValueError("Invalid terminal fill")` with no connection) | `persist.py:194-213, 362-374` | L cases with an empty event list |
| 7 | `skipped_writes` identity: the trader appends `"refresh_benchmark_tape"` itself (`trader.py:788`) and returns `list(self.skipped_writes)` (`:1535`); a ledger holding a copy loses its entries | `trader.py:167,788,1535` | `ReadOnlyLedger(inner, skipped)` shares the trader's list; `dry_run_read_only` golden lists all four entries |
| 8 | Result dict shapes: three shapes, ten return sites; `"untradable"` and `"skipped_writes"` are absent from nine of them, `"gate"` present in one; `None` versus empty list | `trader.py:744,780,840,950,957,971,983,993,998,1530` | K `test_result_key_order_per_shape`; transcripts `result` fields; `cli/main.py` tests (`test_cli_paper.py`) |
| 9 | Tuple-index reads: ten sites, one per field, and a silent swap of `item[2]` and `item[1]` still type-checks | `trader.py:438,462-465,1260,1268,1339,1388` | Task 32 mutates each site and needs a transcript failure; sites with none get a scenario |
| 10 | Environment timing and fail mode (Deviations 3 and 4): the sweep and untradable reads move earlier (a per-run snapshot), `reserve_pct()` is read twice today and once after, `catastrophe_stop_pct()` only when enabled | `trader.py:698-706, 867-884, 1194, 1353-1359` | Task 32 adds scenarios for each invalid float; Task 43's tests for early exits and repeated runs; Task 43 does not claim unchanged read timing |
| 11 | Mutable defaults and aliasing: `RunResult` list fields passed through (the trader's `executed_buys` list is returned as is, `errors` is a fresh concatenation, `skipped_writes` a copy); a defensive copy would change object identity but not content | `trader.py:1531-1535` | `as_dict()` passes the same objects (K test); no dataclass field uses a mutable default |
| 12 | tz-aware datetimes into `ohlcv.timestamp` (a whole-tape corruption in 2026-08) | `cached_yfinance_loader.py:215-226` | the chain hands frames to `persist()` unchanged (the 28 tests compare index and tz with `assert_frame_equal`); `persist()` is a one-line delegation to the unchanged `_cache_to_db`; Task 40 test checks the frame object is forwarded untouched |
| 13 | Unresolved-intent recovery: `attach_broker_order` and `settle_order_intent` raising inside `_reconcile_order_intents` must send the failure notice and re-raise; `_submit_order` names the failing stage | `trader.py:180-290` | the 13 `recovery_*` goldens, `attach_failure`, `intent_create_failure`, `intent_settle_failure`, `lock_verify_failure`; Task 39 leaves each call at its position |
| 14 | The chain's cache routes: constructs-but-empty keeps the writer; constructor or fetch raising drops it | `lab/data.py:108-118` (cache construction and failure; the negative cache is `:92-106`) | C tests `test_cache_empty_falls_back_to_yfinance_and_keeps_writer`, `test_cache_constructor_fails_...`, `test_cache_fetch_raises_falls_back_and_clears_writer`; Task 41's own cases |
| 15 | Lazy imports: `cached_yfinance_loader` pulls in psycopg2 and numpy; the original imports it inside a `try`, so an import error becomes "DB cache unavailable" | `lab/data.py:110-112` | factories import inside their body; Task 41 case where the factory raises `ImportError` |
| 16 | `signal_runner.fetch_stock_ohlcv` is bound by name and patched by the oracle driver | `signal_runner.py:32,182` | name kept, imported from `data.live.fetch_chain`; G and p7 |
| 17 | `cached_yfinance_loader.py` is a pinned file | `.github/workflows/ci.yml:147` | V before and after Task 40 |
| 18 | Logger identity: the read-only skip message is logged by the trader's logger (`ggTrader.paper.trader`); a ledger-module logger would change filtering and routing, and transcripts do not record logs | `trader.py:176` | `ReadOnlyLedger` takes the logger as an argument and the trader passes `_log`; Task 37 and Task 39 caplog tests check the logger name and text |

## Order of work and what can run in parallel

```
30 setup ─ 31 pin SQL ─┬─ 32 pin shapes ─ 33 pin gate ─ 34 SubmittedOrder ─ 35 RunResult ─ 36 records+ports ─┬─ 37 ledger classes ─ 38 PgPaperLedger ─┬─ 39 trader on ledger ─ 43 TraderConfig ─┐
                       │                                                                                    └─ 40 persist() ─ 41 chain ─ 42 wrapper ──┘                                     ├─ 44 docs ─ (45) ─ gate
```
- **Parallel:** Tasks 31 and 32 (different test files; 32 does not need 31). After Task 36, the
  ledger chain (37, 38: only `paper/ledger.py`, `paper/persist.py` and their tests) and the fetch
  chain (40, 41, 42: `data/live/*`, `lab/data.py`, `paper/signal_runner.py`, `paper/overlay.py`, the
  chain tests, `tests/utils/test_import_rules.py`) share no source file and no test file; each gets its
  own worktree from the Task 36 commit (Task 36 Step 7). The trader chain waits at Task 39.
- **Serial:** everything that edits `trader.py` or `models.py` (34, 35, 36, 39, 43: they share
  hunks around `_run`, so no two run at once); Task 33 before 34; Task 39 after 38; Task 43 after
  39; Task 44 after both chains merge; Task 45 last.
- Oracle and `--full` runs are never concurrent.

---

### Task 30: Phase 4 branch, baselines, goldens hash and line anchors

**Files:** none changed. Creates a branch and records measurements.

- [ ] **Step 1: Preconditions.** The owner has approved execution, Phase 2's gate has passed and is
  recorded, and the owner has been asked for quota headroom (O-P4-2).
- [ ] **Step 2: Branch and worktree.**
  ```
  git -C /home/flynn/ggTrader-refactor worktree add /home/flynn/ggTrader-refactor-p4 -b refactor/phase-4 refactor/phase-0-2
  cd /home/flynn/ggTrader-refactor-p4 && uv sync --frozen --extra test --extra dev
  git rev-parse HEAD    # record as BASE4
  .venv/bin/python -c "import ggTrader; print(ggTrader.__file__)"   # under /home/flynn/ggTrader-refactor-p4/src/
  ```
  Then copy `offline.sh` (add its name to `.git/info/exclude`), copy the frozen tape and compare its
  hash with `data/snapshots/MANIFEST.json`, and run the `get_db_connection_string()` check (it must
  raise `ValueError: Database configuration not found`). `git status --short` must be empty.
- [ ] **Step 3: Gate-script copies and baselines.** First make the Phase 4 copies: `mkdir -p $E/p4`,
  copy `gate_oracles.sh`, `gate_tests.sh` and `gate_compare.sh` from `$E`, and change in each only the
  `cd /home/flynn/ggTrader-refactor` line (to `cd ${WT:-/home/flynn/ggTrader-refactor-p4}`), the label
  (`L=p4-$1`, so every output file is `$E/p4-<label>*`), and in `gate_compare.sh` the baseline files
  (`$E/p4base-core.xml,$E/p4base-studies.xml`) and the `--must-pass` list (the two live-golden node ids
  already there, then `$MUST_PASS`). `diff` each copy against its original and keep the three diffs in
  the hand-back. Then, on `BASE4`: `$E/p4/gate_tests.sh p4base` (writes `$E/p4-p4base-core.xml` and
  `$E/p4-p4base-studies.xml`; rename them to `$E/p4base-core.xml` and `$E/p4base-studies.xml`; both exit 0;
  the study file holds at least 777 nodes; `tests/paper/test_live_golden.py` passes, not skips), then
  `$E/p4/gate_oracles.sh p4base` and its comparison against `$E/baseline` (exit 0). A failure means
  Phase 2 changed a result: stop and report.
- [ ] **Step 4: Re-measure every anchor** and write the numbers in the hand-back:
  ```
  grep -n "^def \|^class \|^    def \|_LOCK_KEY\|_TERMINAL_STATUSES\|locals()\|\.begin()\|\.connect()\|conn.commit()" src/ggTrader/paper/persist.py
  grep -n "^class \|    def \|self\._write\|pending_orders\|item\[\|o\[[0-9]\]\|return {\|os.environ\|cash_sweep\.\(sweep\|reserve\|min_clip\|buy_trigger\)\|catastrophe_stop\.catastrophe_stop\|configured_untradable\|config_from_env" src/ggTrader/paper/trader.py
  grep -n "^def \|print(\|_cache_to_db\|loader = None\|plain = \|tiingo" src/ggTrader/lab/data.py
  grep -rn "_cache_to_db\|fetch_stock_ohlcv" src/ggTrader/paper src/ggTrader/data
  ```
  Expected as of 2026-11-18: `persist.py` 31 public functions (`init_paper_schema:98` to
  `clear_split_correction:773`), one `locals()` (`:127`); `trader.py` class `PaperTrader:130`,
  `_write:170`, `_submit_order:180`, `_reconcile_order_intents:211`, `_poll_orders:415`, `run:695`,
  `_run:716-1536`, `run_paper_trading:1539`; ten `return {` sites (`:744,780,840,950,957,971,983,
  993,998,1530`); `lab/data.py` `fetch_stock_ohlcv:67-195`. A different *structure* (not shifted
  numbers) is a finding to report before any task starts.
- [ ] **Step 5: The hash records** (Global Constraints, "Hash records").
  ```
  ls tests/paper/golden/trader_transcripts/*.json | wc -l          # 96
  sha256sum tests/paper/golden/trader_transcripts/*.json tests/paper/transcript/harness.py \
    > $E/p4-fixed.sha256                                            # 97 lines
  sha256sum tests/paper/transcript/scenarios.py tests/paper/test_trader_transcripts.py \
    tests/lab/test_fetch_stock_ohlcv_chain.py > $E/p4-pins.sha256   # 3 lines; 100 in all
  ```
  (96 JSON files plus four Python files make 100; the builder confirms the two counts.)
  and record `./offline.sh .venv/bin/python -c "from tests.paper.transcript.harness import persist_function_names as f; print(len(f()), f())"`
  (expected: **26** names: the 25 listed in Global Constraints plus `paper_run_lock`, which the
  patching loop skips and replaces separately).
- [ ] **Step 6: Haiku inventories (every figure checked by a stronger model).** (a) every test that
  patches or asserts on `_cache_to_db` (expected: `tests/lab/test_fetch_stock_ohlcv_chain.py:64-65,
  165,190,259,333,391,401`, `tests/paper/test_signal_runner.py:342`, and the instance patches in
  `tests/data/test_cached_yfinance_loader.py` that stay); (b) every reader of the `run()` result
  keys in `src/` and `scripts/` (expected: `cli/main.py` only); (c) every tuple-index read of a
  pending order in `trader.py` (the list in risk 9). Differences from the expectations are findings.
- [ ] **Step 7:** No commit.

### Task 31: Pin every SQL statement of `paper/persist.py` (net L)

**Files:**
- Create: `tests/paper/sql_recorder.py`
- Create: `tests/paper/test_persist_sql_golden.py`
- Create: `tests/paper/golden/persist_sql.json`

**What already pins `persist.py`:** `tests/paper/test_paper_persist.py` (mock `connect()`, a few
SQLite runs of `log_snapshot`, `save_peak_value`, rebalance state) checks table names and a few
values. Nothing records a full SQL text, a full bind dictionary, or the commit and rollback events,
so a refactor could change a clause and stay green. The transcript harness patches the functions out
entirely.

**Interfaces:** `sql_recorder.py` holds:
- `RecordingEngine(script: list[object])` with `connect()` and `begin()` returning context managers
  over one `RecordingConnection`. `RecordingConnection.execute(statement, params=None)` appends
  `{"op": "execute", "sql": <statement.text exactly>, "params": <encoded copy of params or null>}` and
  returns the next scripted `FakeResult` (or raises the next scripted exception). `FakeResult`
  supports exactly the calls `persist.py` makes: `.mappings()` (returns itself), `.first()`, `.all()`,
  `.scalar_one()`, and iteration over mapping rows. `commit()` appends `{"op": "commit"}`. The
  context-manager exits append `{"op": "close", "committed": <bool>}` for `connect()` (`committed` is
  true when a `commit` event was recorded in that session), and `{"op": "commit"}` or `{"op": "rollback"}`
  for `begin()` (by whether an exception is propagating). The first event of a session is
  `{"op": "connect"}` or `{"op": "begin"}`.
  **These are modeled context events, not SQLAlchemy's real transaction semantics.** They record what
  the code asked of the engine. A real 2.0 `connect()` block that ends without `commit()` is rolled back
  by SQLAlchemy when the connection closes; the recorder cannot show that, so `close` with
  `committed: false` is the marker for "this session ended without a commit" and the golden's
  `test_case_shows_what_its_name_claims` entries treat it as that and no more. What the real engine does
  with it is covered by SQLAlchemy and by the SQLite runs already in `test_paper_persist.py`.
- `encode_params(params)`: a dict becomes `{key: encode(value)}` with keys in **sorted** order. Bind
  dictionary insertion order is **not** part of the contract (SQLAlchemy binds by name), so the golden
  will not notice a reordered dictionary; this is deliberate and stated in the module docstring. The
  value type is recorded for non-JSON types (`{"$type": "date", "v": "2026-06-20"}`,
  `{"$type": "Connection"}` for an object that is a `RecordingConnection`, so the stray `conn` key
  of `create_order_intent` is visible); floats keep `repr`; `NaN` and infinities are tagged.
- `run_case(fn, args, kwargs, script)`: patches `ggTrader.paper.persist._get_engine` to return the
  `RecordingEngine`, calls the function, and returns `{"events": [...], "returned": encode(value),
  "raised": "<Type>: <message>" | null}`.

**Cases** (module-level `CASES: dict[str, Callable[[], dict]]`, one entry per name; synthetic
values only). At least one case per public function (31), plus the branches below. The golden
stores `{"cases": {name: record}}` through `json.dumps(..., indent=1, sort_keys=False)`.
- `init_paper_schema`: the **eleven** non-empty statements that `_SCHEMA.strip().split(";")` yields
  (`persist.py:99-104`), each compared as full ordered text: `CREATE TABLE ... paper_order_intents`,
  `CREATE TABLE ... paper_trades`, `ALTER TABLE paper_trades ADD COLUMN ... reason`,
  `CREATE TABLE ... paper_snapshots`, `CREATE TABLE ... paper_pending_orders`,
  `ALTER TABLE paper_pending_orders` (the multi-line `flagged_stale` statement),
  `ALTER TABLE paper_pending_orders ADD COLUMN ... reason`, `CREATE TABLE ... paper_rebalance_state`,
  `CREATE TABLE ... paper_risk_state`, `CREATE TABLE ... paper_dividend_accruals`,
  `CREATE TABLE ... paper_split_state`, then one `commit`; the test asserts `len(execute events) == 11`
  (counted on 2026-11-18 by running the split on the source text).
- `create_order_intent` happy path (records the stray `conn` key, Deviation 5).
- `attach_broker_order`: missing intent; conflicting broker id; state not `intent`; legacy pending
  row present (`Order has both pending and intent representations`); success (update).
- `settle_order_intent`: each invalid-fill input (`None`, `nan`, negative qty, unknown status,
  filled with zero qty, zero price, non-finite price) with **no events at all** (risk 6);
  missing intent; already resolved and identical (returns, no write); resolved and conflicting;
  conflicting historical trade; fill with no prior trade (insert then update); zero fill
  (update only).
- `settle_legacy_pending_order`: invalid fill (no events); missing row; intent exists; conflicts;
  insert path; delete-only path.
- `log_trade`: new trade (lock, select, insert, **commit**); identical existing trade (lock,
  select, **close with no commit**); conflicting trade (raises); `reason` omitted and given.
- `paper_run_lock`: acquired then released; not acquired (raises `Paper run already locked`);
  verification false (raises `Paper run lock lost` and still unlocks); and `PaperRunLock.verify`.
- **Execution-failure cases for every mutator** (the transcripts replace the persist functions
  entirely, so a moved method that swallowed a failed `execute` would otherwise be invisible). For each
  of the 15 mutators (`init_paper_schema`, `create_order_intent`, `attach_broker_order`,
  `settle_order_intent`, `settle_legacy_pending_order`, `log_trade`, `log_pending_order`,
  `clear_pending_order`, `mark_pending_order_stale`, `log_snapshot`, `save_peak_value`,
  `save_rebalance_state`, `log_dividend_accrual`, `save_split_correction`, `clear_split_correction`)
  a case `<name>__execute_fails` scripts a `RuntimeError("db down")` at the first `execute`, and for the
  multi-statement ones a second case at the last write (`init_paper_schema` at statement 6;
  `log_trade`, `settle_order_intent` and `settle_legacy_pending_order` at the insert; `attach_broker_order`
  at the update). Each asserts `raised == "RuntimeError: db down"`, no `commit` event after the failing
  `execute`, and `committed: false` (or a `rollback`) at the end. `test_every_mutator_has_an_execution_failure_case`
  fails if a name in the test module's own literal list of the 15 mutators lacks one (Task 37's
  `WRITE_METHODS` does not exist yet; Task 37 later checks its set against this golden).
- Writes with explicit commit: `log_pending_order`, `clear_pending_order`, `mark_pending_order_stale`,
  `log_snapshot` (with a non-finite position value, recording the `json.dumps(json_safe(...),
  allow_nan=False)` string), `save_peak_value`, `save_rebalance_state`, `log_dividend_accrual`,
  `save_split_correction`, `clear_split_correction`.
- Reads: `get_unresolved_order_intents`, `get_pending_orders`, `get_latest_snapshot` (with and
  without `before=`, row and no row), `get_earliest_snapshot`, `get_latest_snapshot_positions`
  (dict value, JSON string value, no row), `get_latest_snapshot_run_date`, `get_peak_value`,
  `get_rebalance_state` (dict and string weights), `get_snapshot_history`,
  `get_trade_history_dates`, `get_last_buy_dates`, `get_accrued_dividend_keys`,
  `get_total_dividend_accrual`, `get_open_split_corrections`, `get_open_split_states`.
- Test functions: `test_persist_sql_case_matches_golden[<name>]` (compares the encoded record to the
  golden, with a unified diff on failure; regenerated only with
  `GGT_REGEN_PERSIST_SQL=1`, which writes the file and fails, same pattern as the transcripts);
  `test_every_case_has_a_golden_entry`, `test_no_orphan_entries`;
  `test_every_public_persist_function_has_a_case` (compares `inspect`-found public functions of
  `ggTrader.paper.persist` against the case names, so a new function cannot go unpinned);
  `test_persist_signatures_are_frozen` (names, kinds, defaults and return annotations of the 31
  functions against a literal in the test; used again after Task 38);
  `test_case_shows_what_its_name_claims[<name>]` (for example the identical-trade case has no
  `commit` event and the new-trade case has exactly one; every invalid-fill case has `events == []`).

- [ ] **Step 1:** Write the recorder, the cases and the tests.
- [ ] **Step 2:** `./offline.sh env GGT_REGEN_PERSIST_SQL=1 scripts/run_tests.sh tests/paper/test_persist_sql_golden.py`
  (expected to fail, writing the golden), then without the variable. Expected: PASS, every test under
  3 s, no `RealConnectionAttempted` (the engine is fake; the guard in `tests/paper/conftest.py`
  applies).
- [ ] **Step 3: Read the golden.** The author reads every case once against `persist.py` and
  confirms the SQL text is the exact string in the source (copy-compare, not regex), the event
  order is the one in the code, and the `create_order_intent` case shows the `conn` key. A case that
  raises something its name does not expect is a finding, not a golden.
- [ ] **Step 4:** Run the file in two more processes (`PYTHONHASHSEED=1`, `2`) to catch set-order
  dependence (`get_accrued_dividend_keys` returns a set; the recorder sorts it). Commit:
  `test: pin every SQL statement, bind dictionary and transaction event of paper/persist.py`.

### Task 32: Pin result key order, tuple-index sites and environment timing (net K)

**Files:**
- Create: `tests/paper/test_phase4_pins.py`
- Modify (additive only): `tests/paper/transcript/scenarios.py` (new scenarios), and
  `tests/paper/golden/trader_transcripts/<new>.json`. `EXPECTED_SCENARIO_COUNT` in `scenarios.py:24`
  rises by the number added. No existing scenario, golden or harness line changes.

**What already pins this, so it is not repeated:** the 96 goldens pin the `result` dict content
(`to_json` sorts keys, so **not** the key order) and every call. The scenarios already cover the
three result shapes' content, the `poll_*` family and `invalid_catastrophe_threshold`.

**Tests:**
- `test_result_key_order_per_shape`: runs today's scenarios `market_closed`, `late_run_live`,
  `drawdown_halt` and `plain_buys_and_sells` through `run_scenario` and asserts
  `list(t["result"])` equals the literals `["buys","sells","errors"]`,
  `["buys","sells","errors","gate"]`, `["buys","sells","errors"]` and
  `["buys","sells","errors","untradable","skipped_writes"]`; and that the nine early returns
  never carry `untradable` or `skipped_writes` (loop over the scenarios named in the ten-site table
  of risk 8: `market_closed`, `late_run_live`, `stale_as_of`, `peak_unreadable`, `peak_invalid`,
  `invalid_equity_halt`, `peak_save_failure`, `last_equity_missing_halt`, `last_equity_invalid_halt`).
- `test_harness_finds_the_26_persist_names`: `persist_function_names()` returns exactly the 26 names
  of Global Constraints (sorted, `paper_run_lock` included, as it does on unchanged code), and
  `paper_run_lock.__module__ == "ggTrader.paper.persist"`; and, separately,
  `test_harness_wraps_25_names_and_replaces_the_lock_separately`, which runs one scenario with a
  recorder on `patch.object` and asserts the ordinary wrapping covers the 25 names other than
  `paper_run_lock`, which gets the lock context manager (`harness.py:755-760`) (risk 1).
- New scenarios (each with a docstring stating its terminal outcome, like the existing ones; inputs
  copied from the nearest existing scenario and cited):
  - `sweep_min_clip_invalid_fail_soft`: `CASH_SWEEP_ENABLED=true`, `SWEEP_MIN_CLIP_USD=abc`;
    sweep-buy sizing raises inside its `try`, the run logs and continues, no sweep buy
    (`trader.py:1351-1362`). Based on `sweep_sizing_failure`.
  - `sweep_buy_trigger_invalid_fail_soft`: same with `SWEEP_BUY_TRIGGER_PCT=abc`.
  - `sweep_reserve_invalid_ends_run`: `SWEEP_CASH_RESERVE_PCT=abc` with a day that reaches the
    funding-sell computation (`trader.py:1189-1195`); the run raises
    `ValueError: could not convert string to float: 'abc'`.
  - `sweep_reserve_invalid_but_sweep_off`: the same variable with the sweep off; the run succeeds
    (nothing reads it).
  - `untradable_symbols_from_env`: `UNTRADABLE_SYMBOLS=" aapl, ,Msft "` with both held; both frozen,
    notice text recorded (`trader.py:881-893`).
  - `catastrophe_enabled_pct_blank`: `CATASTROPHE_STOP_ENABLED=true`, `CATASTROPHE_STOP_PCT=""` uses
    the default -25%.
  - `pending_order_all_five_fields`: a live day with a strategy sell, a catastrophe-stop sell, a sweep
    sell and a buy submitted in one run, then a `get_order` failure on the final lookup for one of
    them (`poll_final_lookup_fails` shows the shape), so that the fallback dictionary of
    `_poll_orders` (`trader.py:460-469`) is built from `item[2]`, `item[1]` and `item[3]` for a SELL
    and for a BUY (the `notional` is `item[3]` only for a BUY).
  Add only the scenarios whose behavior no existing golden shows; the builder checks each name
  against the 96 and records the decision in the commit message.
- Each new scenario gets a `test_golden_shows_scenario_behavior` entry like the others.

- [ ] **Step 1:** Write the tests and scenarios.
- [ ] **Step 2:** Regenerate once (`GGT_REGEN_TRANSCRIPTS=1`, expected to fail) and rerun; the only
  new or changed files under `tests/paper/golden/` are the added scenarios' goldens (`git status`
  confirms; the 96 existing files are untouched). Run the determinism test across two hash seeds
  (the existing `test_transcripts_stable_across_hash_seeds`).
- [ ] **Step 3: Tuple-index mutation check (not committed).** For each of the ten sites of risk 9,
  change the index or attribute read (for example `item[2]` to `item[1]` at `trader.py:462`, `o[1]`
  to `o[2]` at `:1260`, `o[0]` to `o[1]` at `:1339`, the `reason` position in the unpack at `:1268`),
  rerun `tests/paper/test_trader_transcripts.py`, and require at least one failure per site. A site
  with no failure gets a scenario (this is what `pending_order_all_five_fields` is for); revert after
  each. `git status` shows `trader.py` clean afterwards.
- [ ] **Step 4:** Run the file with `-p no:randomly` and in the neighbourhood run; commit:
  `test: pin result key order, pending-order tuple sites and sweep and stop setting behavior`.

### Task 33: Pin gate and mutation check

**Files:** none committed; updates the hash records under `$E`.

- [ ] **Step 1: The pin check** of Task 31 and 32, on untouched code, in three processes
  (`PYTHONHASHSEED` unset, 1, 2). All pass.
- [ ] **Step 2: Fast core collects the pins.**
  `./offline.sh scripts/run_tests.sh --fast --collect-only -q 2>&1 | grep "persist_sql_golden\|phase4_pins" | sort > fast.ids`
  and the same with the two files named directly
  (`./offline.sh scripts/run_tests.sh --collect-only -q tests/paper/test_persist_sql_golden.py tests/paper/test_phase4_pins.py | grep "::" | sort > direct.ids`);
  the two lists of node ids are identical, so the fast core collects every test of both files and none
  is `slow`.
- [ ] **Step 3: Mutation check (not committed).** Make each change, rerun both pin files and the
  transcripts, confirm at least one failure, revert:
  - `persist.py`: change one character in the `settle_order_intent` UPDATE text; rename one bind key
    in `log_snapshot`; cast `pv` with `float()` in `log_snapshot`; delete `conn.commit()` in
    `save_peak_value`; replace `engine.begin()` with `engine.connect()` in `attach_broker_order`;
    add `conn.commit()` to the early-return branch of `log_trade`.
  - `trader.py`: swap the keys `untradable` and `skipped_writes` in the final result; add
    `"untradable"` to the `market_closed` return; read `SWEEP_MIN_CLIP_USD` outside the `try`.
  Afterwards `git status` shows both files clean.
- [ ] **Step 4: Opus read of the new goldens** (fresh context, given only `persist_sql.json`,
  `persist.py`, the new scenario goldens and the claim "each case shows what its name says"). It
  reports any case that would pass for a wrong reason.
- [ ] **Step 5: Update the hash records.** Append to `$E/p4-fixed.sha256` the hashes of
  `tests/paper/golden/persist_sql.json` and of every golden Task 32 added (the reference golden is now
  immutable, Task 38); re-record the `scenarios.py` and `test_trader_transcripts.py` lines of
  `$E/p4-pins.sha256` (Task 32 edited `scenarios.py` additively); leave the chain-test line alone.
  `git diff` of `scenarios.py` against the Task 30 commit shows additions only. The build worktrees are
  created in Task 36 Step 7, after that task is committed.
- [ ] **Step 6:** Report usage to the owner (Phase A done).

---

### Task 34: `SubmittedOrder` replaces the five-item tuple (trader chain)

**Files:**
- Create: `src/ggTrader/paper/models.py` (module docstring: "Typed records of the paper trader;
  `TypedDict`s describe mappings that are stored as JSON, dataclasses describe values the trader
  builds.")
- Modify: `src/ggTrader/paper/trader.py` (sites below)
- Test: `tests/paper/test_models.py`

**Current sites (re-grep `pending_orders`, `item\[`, `o\[`):** the annotation of `_poll_orders`
(`:415-417`, `list[tuple[str, str, str, float, str | None]]`); inside it `remaining = {item[0]: item
...}` (`:438`) and the fallback dictionary (`:456-469`: `item[2]`, `item[1]`, `item[3]`); in `_run`
the annotation (`:1038`), the four `append((oid, ...))` sites (`:1113`, `:1215`, `:1251`, `:1335`),
the sweep buy tuple (`:1372-1378`), the filters `o[1] == "SELL"` (`:1260`) and `o[0] not in
filled_orders` (`:1339`), and the two unpacking loops (`:1268`, `:1388`).

**Interfaces:**
```python
@dataclass(frozen=True, slots=True)
class SubmittedOrder:
    """An order handed to the broker this run, kept until it settles."""
    order_id: str
    side: Literal["BUY", "SELL"]
    symbol: str
    amount: float          # BUY: intended notional; SELL: the position's market value at submit time
    reason: str | None     # None for a strategy order; cash_sweep.SWEEP_TRADE_REASON or
                           # catastrophe_stop.CATASTROPHE_STOP_REASON otherwise
```
`_poll_orders(self, orders: list[SubmittedOrder]) -> dict[str, dict]`. The meaning of `amount` is
the comment text now at `trader.py:1036-1037`, moved to the dataclass.

- [ ] **Step 1: Failing tests** (`tests/paper/test_models.py`): `test_submitted_order_is_frozen`,
  `test_submitted_order_fields_in_order` (`dataclasses.fields` names, in this order),
  `test_submitted_order_equality_and_hash`, `test_submitted_order_has_no_tuple_behavior`
  (`len(order)` raises `TypeError`; this documents Deviation 7).
- [ ] **Step 2:** FAIL, then **Step 3:** create the dataclass, replace every site above with
  `SubmittedOrder(...)` and attribute reads. The loops become `for order in pending_orders:` with
  `order.order_id`, `order.side`, `order.symbol`, `order.amount`, `order.reason`. No other line of
  `_run` changes; local variable names used afterwards (`oid`, `symbol`, `amount`) are re-bound from
  the attributes in the same statement.
- [ ] **Step 4:** the new tests pass; `git diff --color-moved` shows only these sites; Step 3 of
  Task 32 (the mutation check) is re-run on the new attribute reads (swap `.symbol` and `.side` at
  each) and still fails the transcripts.
- [ ] **Step 5:** The task gate, oracle label `t34`.
- [ ] **Step 6:** Commit: `refactor(paper): SubmittedOrder replaces the five-item pending-order tuple`.

### Task 35: `RunResult` with `as_dict()` (trader chain)

**Files:**
- Modify: `src/ggTrader/paper/models.py`, `src/ggTrader/paper/trader.py` (ten return sites)
- Test: `tests/paper/test_models.py`

**Current sites:** `trader.py:744, 780, 840, 950, 957, 971, 983, 993, 998` (early returns) and
`:1530-1536` (the final result). `run()` (`:695`) returns `_run`'s value.

**Interfaces:**
```python
@dataclass(frozen=True)
class RunResult:
    buys: list[str]
    sells: list[str]
    errors: list[str]
    gate: str | None = None
    untradable: list[str] | None = None
    skipped_writes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Today's keys, in today's order; a field that is None is absent."""
```
`as_dict()` returns `buys`, `sells`, `errors`, then `gate` if not `None`, then `untradable` and
`skipped_writes` if not `None`, passing the *same list objects* (no copy; risk 11). Eight early
returns become `RunResult([], [], [reason]).as_dict()` (or `[]` for the three with no error),
`:780` adds `gate="late_run"`, the last becomes `RunResult(executed_buys, executed_sells,
loss_halts + errors, untradable=sorted(untradable_seen), skipped_writes=list(self.skipped_writes)).as_dict()`.
`run()` and `_run()` keep `-> dict`.

- [ ] **Step 1: Failing tests:** `test_run_result_as_dict_key_order` (the four shapes of
  `test_result_key_order_per_shape` built by hand), `test_run_result_omits_none_fields`,
  `test_empty_untradable_is_present_not_omitted` (`untradable=[]` appears), `test_as_dict_returns_the_same_objects`
  (`result.as_dict()["buys"] is result.buys`), `test_run_result_is_frozen`, and
  `test_cli_consumes_every_shape` (feeds each shape to the printing code of `ggt paper` through
  `CliRunner` with `run_paper_trading` patched, as `tests/paper/test_cli_paper.py` does, and checks
  no `KeyError`).
- [ ] **Step 2 to 4:** FAIL, implement, replace the ten sites, PASS. K's `test_result_key_order_per_shape`
  and every transcript `result` field must pass unchanged.
- [ ] **Step 5:** The task gate, oracle label `t35`.
- [ ] **Step 6:** Commit: `refactor(paper): RunResult builds the run result; run() still returns its dict`.

### Task 36: Records and ports (`models.py`, `ports.py`) (trader chain)

**Files:**
- Modify: `src/ggTrader/paper/models.py`
- Create: `src/ggTrader/paper/ports.py`
- Modify: `src/ggTrader/paper/trader.py` (annotations only: `__init__`, `_run` locals where the
  record is obvious, `_poll_orders` returns)
- Test: `tests/paper/test_models.py`, `tests/paper/test_ports.py`

**Interfaces (all in `models.py` unless stated):**
- `class Account(TypedDict, total=False)`: `cash: float`, `portfolio_value: float`,
  `last_equity: float | None`, `buying_power: float`, `multiplier: float` (the keys of
  `AlpacaBroker.get_account`, `alpaca_broker.py:36-48`; `total=False` because
  `tests/paper/test_trader.py:234-239` builds accounts without `multiplier`).
- `class Position(TypedDict, total=False)`: `qty`, `market_value`, `current_price`, `avg_entry`,
  `cost_basis: float`; `unrealized_pl`, `unrealized_plpc`, `change_today: float | None` (the keys at
  `alpaca_broker.py:50-65`). A `TypedDict`, not a dataclass (design 5.2): snapshots store positions as
  JSON, and `split_check`, `dividend_check`, `catastrophe_stop` and the studies take plain mappings.
  Those modules' annotations are not changed.
- `class OrderStatus(TypedDict, total=False)`: `id`, `client_order_id`, `symbol`, `side`, `status: str`;
  `qty`, `notional`, `filled_qty`, `filled_avg_price: float | None` (`alpaca_broker.py:105-118`).
- `class MarketClock(TypedDict, total=False)`: `is_open: bool`, `next_open`, `next_close: str`.
  (`total=False`: the trader reads it with `.get` and fails open, `trader.py:735-740`.)
- `class SleeveSignals(TypedDict, total=False)`: `buys: list[str]`, `sells: list[str]`,
  `as_of: str`, `universe_size: int`, `membership_gate: dict[str, Any]`, `last_exit: dict[str, str]`,
  `engine: dict[str, Any]`, `bars: dict[str, Any]` (the return of `signal_runner.generate_signals`,
  `signal_runner.py:~230-262`).
- `class SignalBundle(TypedDict, total=False)`: `sleeves: dict[str, SleeveSignals]`,
  `weights: dict[str, float]`, `scale: float`, `fallback_used: bool` (the return of
  `generate_core_signals`; the builder greps its `return {` for any further key and adds it).
- In `ports.py`: `@runtime_checkable class Broker(Protocol)` with the 11 public methods the trader
  calls: `get_account`, `get_positions`, `get_untradable`, `submit_buy`, `submit_sell`, `get_order`,
  `get_order_by_client_id`, `get_clock`, `get_split_evidence`, `get_split_corrections`,
  `get_dividend_corrections` (the public methods of `AlpacaBroker`; its `_normalize_order` and `_get_*`
  helpers are private and excluded; the builder re-counts and lists them in the docstring). `@runtime_checkable
  class Notifier(Protocol)` with `send(message: str) -> bool`, `trade_alert(side, symbol, amount, order_id,
  qty=None, price=None, status=None) -> bool` and `daily_summary(...)` with the signature of
  `TelegramNotifier.daily_summary` (`notifier.py:68`). The one-method `Notifier` in
  `data/membership_weekly.py:113` is not touched and not imported (`data` must not import `paper`).
- `PaperTrader.__init__` is annotated `broker: Broker`, `notifier: Notifier`. No runtime change.

- [ ] **Step 1: Failing tests:** `test_alpaca_broker_satisfies_broker_protocol` (exact equality of the
  method names and `inspect.signature` parameter names and defaults of `Broker` with `AlpacaBroker`'s
  public methods, so a drift fails; does not instantiate `AlpacaBroker`, which needs credentials),
  `test_fake_broker_of_the_transcript_harness_satisfies_broker_protocol` (the same exact equality for
  `tests.paper.transcript.harness.FakeBroker`, whose methods are concrete; it sits next to the existing
  `test_fake_broker_implements_every_public_alpaca_broker_method`),
  `test_notifier_protocols_match_telegram_and_dry_run_notifier`: signature equality for the **concrete**
  implementations (`TelegramNotifier`; `DryRunNotifier`, whose `trade_alert` and `daily_summary` end in
  `**_`/`*_` catch-alls, is compared by the parameters it names plus acceptance of the protocol's call
  shape), and for the harness `RecordingNotifier`, whose `trade_alert(self, *args, **kwargs)` and
  `daily_summary(self, *args, **kwargs)` are variadic recording wrappers (`harness.py:397-407`), the test
  checks **argument acceptance** only: it binds the `TelegramNotifier` signature's full argument set
  (every parameter name, positionally and by keyword) to the wrapper with `inspect.Signature.bind` and
  it succeeds. No harness edit is needed or allowed,
  `test_position_keys_match_alpaca_broker` (reads `get_positions` source keys through a stub client),
  `test_typed_dict_keys_are_the_documented_ones` for each record.
- [ ] **Step 2 to 4:** FAIL, implement, PASS. The module docstrings state the public names.
- [ ] **Step 5:** The task gate, oracle label `t36`.
- [ ] **Step 6:** Commit: `feat(paper): typed records and Broker and Notifier protocols`.
- [ ] **Step 7: Build the parallel worktrees from that commit** (not before: Task 37 needs
  `models.py`, which exists only after Step 6). `git rev-parse HEAD` is recorded as `BASE36`, then:
  ```
  git -C /home/flynn/ggTrader-refactor-p4 worktree add /home/flynn/ggTrader-refactor-p4-ledger -b refactor/phase-4-ledger BASE36
  git -C /home/flynn/ggTrader-refactor-p4 worktree add /home/flynn/ggTrader-refactor-p4-fetch  -b refactor/phase-4-fetch  BASE36
  ```
  In each: `uv sync --frozen --extra test --extra dev`, `offline.sh` copied and excluded, the frozen
  tape copied and hash-checked, the database-address check as in Task 30 Step 2, and
  `git log -1 --format=%H` equal to `BASE36`; `ls src/ggTrader/paper/models.py src/ggTrader/paper/ports.py`
  succeeds. A chain's gate runs the `$E/p4/` copies with `WT=<its worktree>`.
- [ ] **Step 8:** Report usage to the owner (Phase B done).

---

### Task 37: `PaperLedger` protocol and its three implementations (ledger worktree)

**Files:**
- Create: `src/ggTrader/paper/ledger.py`
- Test: `tests/paper/test_ledger.py`

No existing file is edited, so nothing can change behavior; Task 39 wires it in.

**Interfaces:**
- `class RunLock(Protocol)`: `def verify(self) -> None`.
- `@runtime_checkable class PaperLedger(Protocol)`: **one method per persist function, same name,
  same parameters, same return annotation** (the 31 in the Task 31 signature literal;
  `paper_run_lock(self) -> AbstractContextManager[RunLock]`). Same names on purpose: `ModuleLedger`
  and the skipped-write log are then name-for-name with today (Deviation 11 note, risk 7).
  `positions` parameters are `dict[str, Position]`.
- `WRITE_METHODS: frozenset[str]`: `init_paper_schema`, `create_order_intent`, `attach_broker_order`,
  `settle_order_intent`, `settle_legacy_pending_order`, `log_trade`, `log_pending_order`,
  `clear_pending_order`, `mark_pending_order_stale`, `log_snapshot`, `save_peak_value`,
  `save_rebalance_state`, `log_dividend_accrual`, `save_split_correction`, `clear_split_correction`
  (15; the three `get_*` and `paper_run_lock` are not writes). The builder derives it from the
  Task 31 case names that have a commit or insert event and checks it equals this list.
- `TRADER_NAMES: tuple[str, ...]`: the 26 names `trader.py` imports (Global Constraints).
- `class ModuleLedger`: `__init__(self, namespace: ModuleType, fallback: ModuleType)`; `__getattr__(name)`
  returns `getattr(namespace, name)` for a name in `TRADER_NAMES`, else `getattr(fallback, name)`, **at
  each access** (no caching). `fallback` is the `persist` module, which serves the five names the
  trader module does not bind (`get_latest_snapshot_positions`, `get_latest_snapshot_run_date`,
  `get_open_split_corrections`, `get_rebalance_state`, `save_rebalance_state`). An unknown name raises
  `AttributeError`.
- `class ReadOnlyLedger`: `__init__(self, inner: PaperLedger, skipped: list[str], log: logging.Logger)`;
  a name in `WRITE_METHODS` appends the name to `skipped`, logs `"read-only: skipped write %s"` at INFO
  through the `log` it was given (text as `trader.py:176`), and returns `None` without calling `inner`.
  **Logger identity is preserved**: the trader passes its own module logger (`_log`, name
  `ggTrader.paper.trader`), so filtering and routing configured by logger name (Phase 2 Task 13 attaches
  its handler to the `ggTrader` logger and tests or operators may filter by module) see the same logger as
  today; every other name is returned
  from `inner` unchanged. It never catches an exception.
- `class InMemoryLedger`: a test double holding dicts and lists. The intent lifecycle (create,
  attach, get unresolved, settle) raises the same `ValueError` messages as the SQL versions
  (`Missing intent or conflicting broker order`, `Invalid intent state`, `Invalid terminal fill`,
  `Conflicting settlement`, `Order has both pending and intent representations`); reads return what
  was seeded or written; `paper_run_lock()` yields an object whose `verify()` records its calls.
  It is modeled on the harness's `IntentStore` and `tests/paper/test_trader.py:154-200`
  `_intent_store`. It does **not** claim SQL parity for validation (Deviation note in the docstring).

- [ ] **Step 1: Failing tests** (`tests/paper/test_ledger.py`):
  - `test_protocol_names_equal_persist_function_names` (sorted equality with the 31 public persist
    functions; `inspect.signature` equality per method against `PgPaperLedger` is added in Task 38).
  - `test_module_ledger_resolves_the_trader_module_at_call_time`: a stand-in namespace; patch a name
    after constructing the ledger, call, and see the patched function used; patch again between two
    calls (risk: string patches applied late).
  - `test_module_ledger_covers_all_26_trader_names_and_falls_back_for_the_other_five`.
  - `test_module_ledger_unknown_name_raises_attribute_error`.
  - `test_read_only_ledger_skips_every_write_and_records_names_in_order`: 15 names, none reaches the
    inner ledger; the shared `skipped` list is the one passed in (identity).
  - `test_read_only_ledger_passes_reads_through_and_never_swallows`: a read returns the inner value; an
    inner exception from a read propagates unchanged.
  - `test_read_only_ledger_logs_through_the_logger_it_was_given` (caplog: the record's logger name is the injected logger's, `ggTrader.paper.trader` when built by the trader, and the text is unchanged).
  - `test_in_memory_ledger_intent_round_trip_and_error_messages`, `test_in_memory_ledger_lock_records_verify`.
  - `test_write_methods_set_matches_the_sql_golden` (every name in `WRITE_METHODS` has a `commit`
    or `insert`-bearing event in `persist_sql.json`, and every other name has none, using the Task 31
    golden as the independent source).
- [ ] **Step 2 to 4:** FAIL, implement with module and Google docstrings, PASS.
- [ ] **Step 5:** The task gate (the fast subset: ruff, the new tests, neighbourhood run; transcripts
  are unaffected because nothing imports `ledger.py` yet; oracle label `t37`).
- [ ] **Step 6:** Commit: `feat(paper): PaperLedger protocol with in-memory, read-only and module-lookup implementations`.

### Task 38: `PgPaperLedger` and one-line delegations in `persist.py` (ledger worktree)

**Files:**
- Modify: `src/ggTrader/paper/persist.py`
- Test: `tests/paper/test_persist_ledger.py`. `tests/paper/golden/persist_sql.json` is **not edited or
  regenerated** in this task: it is the immutable pre-extraction reference recorded from today's
  functions in Task 31 (hash in `$E/p4-fixed.sha256`), and it keeps the stray `conn` key of Deviation 5.

**Current range:** the 31 functions, `PaperRunLock` and `_lock_order_and_reject_pending`
(`persist.py:98-783`; `_SCHEMA` at `:16-95`, `_LOCK_KEY` at `:107`, `_TERMINAL_STATUSES` at `:108`).

**Design of the change:**
- `class PgPaperLedger` in `persist.py`; each method body is the current function body **moved
  verbatim**: same SQL text, same bind dictionaries, same `engine.connect()` or `engine.begin()` (Deviation 1),
  same `conn.commit()`, same validation order (the two `settle_*` validators run before the engine is
  touched), same early returns, same exceptions. It calls the module global `_get_engine()` inside the
  method (risk 5). `paper_run_lock` keeps `@contextmanager`; `PaperRunLock`, `_lock_order_and_reject_pending`,
  `_SCHEMA`, `_LOCK_KEY`, `_TERMINAL_STATUSES` stay module-level, where tests and Phase 2's Task 11
  put them.
- `create_order_intent` replaces `locals()` with an explicit dictionary of its seven named
  parameters (`client_order_id`, `run_date`, `side`, `symbol`, `amount`, `amount_kind`, `reason`; the SQL
  binds exactly these seven; the old `locals()` dictionary also carried `conn`).
- The 31 module-level functions become one-line delegations with **unchanged signatures, defaults,
  keyword-only markers, annotations and docstrings**, for example
  `def create_order_intent(client_order_id, run_date, side, symbol, amount, amount_kind, reason) -> None: return _pg().create_order_intent(client_order_id, run_date, side, symbol, amount, amount_kind, reason)`
  (written with full annotations). `_pg()` returns a module-level, lazily created, stateless
  `PgPaperLedger()`. `paper_run_lock()` stays a `@contextmanager` function that does
  `with _pg().paper_run_lock() as lock: yield lock`; its `__module__` is still `ggTrader.paper.persist`
  (risk 1). The function objects stay defined in `persist.py` (the harness inspects `__module__`).

- [ ] **Step 1: Failing tests** (`tests/paper/test_persist_ledger.py`):
  - `test_pg_paper_ledger_satisfies_paper_ledger` (method names and `inspect.signature` per method
    against the protocol of Task 37).
  - `test_module_functions_delegate_and_keep_their_signatures` (Task 31's
    `test_persist_signatures_are_frozen` runs unchanged and passes; each delegation's `__module__` is
    `ggTrader.paper.persist`; `inspect.getsource` of each delegation is at most 3 lines of body).
  - `test_engine_is_looked_up_at_call_time`: patch `persist._get_engine` between two calls and see each
    engine used.
  - `test_sql_recording_identical_to_the_pre_extraction_reference`: **the statement-for-statement
    proof.** For every case of Task 31, run the case through *both* the module-level delegation and the
    matching `PgPaperLedger` method on a fresh `RecordingEngine` and assert (a) the two event lists are
    equal, and (b) each equals `normalize_reference(golden_entry)`, where `normalize_reference` is the
    **one** explicit normalization in the test module: it drops the key `conn` from the bind dictionary
    of every `execute` event of the `create_order_intent` case and does nothing else (its docstring cites
    Deviation 5). Every other case is compared untouched, and the events of the new code carry no `conn`
    key. A case where the delegation and the method disagree, or the reference differs outside that one
    key, fails.
  - `test_reference_golden_still_records_the_stray_conn_key`: the unedited `persist_sql.json` has a
    `conn` key (type `Connection`) in the `create_order_intent` case, so the reference cannot be
    regenerated away, and `normalize_reference` is shown to change exactly that case (every other
    entry compares equal before and after it).
- [ ] **Step 2:** FAIL, then **Step 3:** move the bodies and write the delegations. Show the move with
  `git diff --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`; every changed line
  that is not moved (the `self` parameter, the class line, the explicit dictionary, the delegation bodies) is
  listed in the commit body.
- [ ] **Step 4:** Do **not** regenerate the golden (`GGT_REGEN_PERSIST_SQL` stays unset);
  `git diff --stat -- tests/paper/golden/persist_sql.json` is empty and `sha256sum -c $E/p4-fixed.sha256`
  is all OK.
- [ ] **Step 5:** The task gate (existing `test_paper_persist.py`, including its 17 dotted-string
  `_get_engine` patches and one `monkeypatch.setattr` (`:655`) and the SQLite runs, passes with no
  edit; oracle label `t38`).
- [ ] **Step 6:** Commit: `refactor(paper): PgPaperLedger holds the persist SQL; module functions delegate`.

### Task 39: The trader works through `self._ledger` (trader worktree, after Tasks 37 and 38)

**Files:**
- Modify: `src/ggTrader/paper/trader.py`
- Test: `tests/paper/test_trader_ledger.py`

First merge `refactor/phase-4-ledger` into `refactor/phase-4` (the chain touches `ledger.py`,
`persist.py` and their tests only; a conflict is a finding).

**Current sites (re-grep `_write(`, the persist call names, `paper_run_lock`):** `_write` at
`:170-178`; its call sites `init_paper_schema` (`:718`), `clear_split_correction` (`:523, 577`),
`save_split_correction` (`:590`), `log_dividend_accrual` (`:657`), `save_peak_value` (`:978`),
`log_snapshot` (`:1514`); direct calls `create_order_intent`, `attach_broker_order` (`:188, 202, 268`),
`settle_order_intent` (`:270`), `get_unresolved_order_intents` (`:214`), `get_pending_orders`
(`:215, 362`), `settle_legacy_pending_order` (`:396`), `mark_pending_order_stale` (`:340`),
`get_last_buy_dates` (`:310`), `get_open_split_states` (`:513`), `get_snapshot_history` (`:546, 622`),
`get_trade_history_dates` (`:554`), `get_total_dividend_accrual` (`:616, 688`),
`get_accrued_dividend_keys` (`:643`), `get_peak_value` (`:945`), `get_latest_snapshot` (`:1023`),
`get_earliest_snapshot` (`:1480`), and `paper_run_lock()` (`:709`).

**Change:**
- Keep the `from ggTrader.paper.persist import (...)` block and add to each name's line
  `# noqa: F401 - looked up by ModuleLedger at call time` (the three that already carry a
  `noqa` keep a combined comment). Add `import sys` and `from ggTrader.paper import persist as _persist`
  and `from ggTrader.paper.ledger import ModuleLedger, PaperLedger, ReadOnlyLedger`.
- `PaperTrader.__init__` gains a keyword-only `ledger: PaperLedger | None = None`. It builds
  `base = ledger if ledger is not None else ModuleLedger(sys.modules[__name__], _persist)`,
  then `self._ledger = ReadOnlyLedger(base, self.skipped_writes, _log) if self._read_only else base`.
  `self.skipped_writes` is created before it (it is today, `:167`), so the list the trader returns and
  the one the ledger fills are one object (risk 7). **Delete `_write`.**
- Every `self._write(fn, *a)` becomes `self._ledger.fn(*a)`; every direct call becomes
  `self._ledger.<same name>(...)` **at the same position** inside its existing `try`/`except` (risk 13:
  stage labels and notices depend on which call raised); `with paper_run_lock() as lock:` becomes
  `with self._ledger.paper_run_lock() as lock:`. `self._paper_lock.verify()` stays.
- `refresh_benchmark_tape` is not a ledger call and keeps its manual `skipped_writes.append`
  (`:787-789`); `generate_core_signals`, `live_membership_source`, `_today_et` etc. are untouched.

- [ ] **Step 1: Failing tests** (`tests/paper/test_trader_ledger.py`):
  - `test_trader_module_binds_the_26_ledger_names` (each is an attribute of the trader module and
    `__module__ == "ggTrader.paper.persist"`; guards risk 2).
  - `test_trader_has_no_write_method_and_no_direct_persist_call` (AST of `trader.py`: no `_write`
    attribute, and no `Call` whose function is a bare name in the 26).
  - `test_string_patch_after_construction_still_wins` (constructs a trader, then
    `patch("ggTrader.paper.trader.get_peak_value", return_value=123.0)`; a run reads 123.0).
  - `test_read_only_run_records_the_same_skipped_names_as_before`: the `dry_run_read_only` scenario's
    `skipped_writes` list equals the literal in its golden.
  - `test_read_only_skip_is_logged_by_the_trader_logger`: a read-only run's `"read-only: skipped write ..."`
    records come from the logger named `ggTrader.paper.trader` (caplog), with today's text.
  - `test_read_only_ledger_is_never_asked_to_run_an_intent_call`: in every read-only scenario the
    transcript contains no `persist.create_order_intent`, `attach_broker_order`, `settle_*`,
    `log_trade`, `log_pending_order`, `clear_pending_order` or `mark_pending_order_stale` (Deviation 11).
  - `test_trader_runs_on_an_in_memory_ledger`: `PaperTrader(broker, notifier, ledger=InMemoryLedger(...))`
    runs the data of `plain_buys_and_sells` (from the scenario module) and its result dict equals that
    golden's `result`, with the settled intent rows present in the ledger. **Injecting a ledger injects
    nothing else**: the signals (`generate_core_signals`), the membership source, the benchmark refresh,
    `_today_et`/`_now_pt` (the scenario dates are fixed in June), `uuid4`, `time` and `datetime` are still
    patched in the trader module exactly as `_run_scenario` does (`harness.py:759-770`). The test builds
    them with the harness's own fakes (`FakeBroker`, `RecordingNotifier`, `FakeTime`, the pinned
    `datetime`, the signal and membership replacements; read-only use of the harness's names, no harness
    edit) and patches **no persist function**. What it proves is ledger-independent testing: the persist
    patches are no longer needed, not that the whole trader runs without a patch.
- [ ] **Step 2 to 4:** FAIL, implement, PASS. The about 440 existing string patches pass unchanged (the
  neighbourhood run), as do all 96 goldens byte for byte.
- [ ] **Step 5: Mutation check (not committed):** in `ModuleLedger.__getattr__`, cache the looked-up
  function; confirm `test_string_patch_after_construction_still_wins` and a transcript fail; revert.
- [ ] **Step 6:** The task gate, oracle label `t39`.
- [ ] **Step 7:** Commit: `refactor(paper): the trader reaches the database only through a PaperLedger`.

---

### Task 40: A public `CachedYFinanceLoader.persist()` (fetch worktree)

**Files:**
- Modify: `src/ggTrader/data/live/cached_yfinance_loader.py`
- Modify: `src/ggTrader/lab/data.py` (the two `loader._cache_to_db(...)` calls, `:~134, ~163`)
- Modify: `src/ggTrader/paper/signal_runner.py` (`:89`, `CachedYFinanceLoader()._cache_to_db(df, "1d")`)
- Modify (one token each, listed): `tests/lab/test_fetch_stock_ohlcv_chain.py` (`_cache_to_db` to
  `persist` at `:64,65,118,165,190,259,333,391,401`), `tests/paper/test_signal_runner.py:342`
- Test: `tests/data/test_cached_loader_persist.py`

**Interfaces:** `def persist(self, df: pd.DataFrame, interval: str) -> None`, docstring "Write a
fetched OHLCV frame to the `ohlcv` table (naive UTC timestamps, upsert by timestamp, symbol, interval
and venue); failures are logged and swallowed, as `_cache_to_db` always did." Body:
`self._cache_to_db(df, interval)`. `_cache_to_db` stays the implementation and keeps every line
(the tz-stripping comment at `:215-226` is untouched; risk 12).

- [ ] **Step 1: V before:** `./offline.sh .venv/bin/python scripts/verify_pins.py --expect <counts of ci.yml:147>`
  exits 0; record the listing.
- [ ] **Step 2: Failing tests:** `test_persist_forwards_the_frame_and_interval_untouched` (patch
  `_cache_to_db` on an instance; the exact frame object, tz-aware UTC index, and `"1d"` arrive),
  `test_persist_swallows_nothing_new` (an exception from `_cache_to_db` propagates; `_cache_to_db`
  itself still logs and swallows driver errors, shown with a fake `_connect`), `test_persist_exists_on_the_class`.
- [ ] **Step 3:** FAIL, add the method, change the three source call sites and the listed test tokens.
  The test edits are a **mechanical rename only**; the commit body lists each changed line and the
  reviewer confirms each differs by that token.
- [ ] **Step 3b: The one authorized hash change.** `tests/lab/test_fetch_stock_ohlcv_chain.py` is hashed
  in `$E/p4-pins.sha256` (Task 30 Step 5), so the rename makes the gate's hash check fail for that one
  line. Re-record **only that line** (`sha256sum tests/lab/test_fetch_stock_ohlcv_chain.py` replacing
  its entry), after `git diff` of the file shows exclusively the nine `_cache_to_db` to `persist` token
  changes. The 96 goldens and `harness.py` in `$E/p4-fixed.sha256` and the other two lines of
  `$E/p4-pins.sha256` are untouched, and `sha256sum -c` on both files must still be all OK afterwards.
- [ ] **Step 4:** Net C (28 tests, now through `persist`), `tests/paper/test_signal_runner.py`,
  `tests/data/test_cached_yfinance_loader.py` pass; the task gate; V after: counts unchanged, or the
  new counts written to `.github/workflows/ci.yml:147` in this commit with a changelog line (Phase 0-2
  Task 14 did this for the one `MATCH` pin).
- [ ] **Step 5:** Oracle label `t40`. Commit: `refactor(data): a public persist() on CachedYFinanceLoader; callers stop reaching for the private writer`.

### Task 41: `OhlcvFetchChain` and `fetch_stock_ohlcv` in `data/live` (fetch worktree)

**Files:**
- Create: `src/ggTrader/data/live/fetch_chain.py`
- Test: `tests/data/test_fetch_chain.py`

Nothing is rewired yet; Task 42 does that, so net C is unaffected here.

**The three blocks differ (this table is the contract):**

| | Primary fallback | yfinance gap | Tiingo gap |
|---|---|---|---|
| runs when | cache route left `df` empty | symbols missing and not skipped | symbols still missing and not skipped |
| announces | none | `"  [data] fetching {n} symbols missing from cache..."` **before** constructing | `"  [data] trying Tiingo for {n} symbols yfinance missed..."` **after** constructing |
| provider construction | `YFinanceDataLoader()` stored and reused for the gap | reuses it, else constructs | `TiingoDataLoader()` |
| requests | all tickers (skipped names included) | missing only | still-missing only |
| fetch errors | propagate; empty result raises `ValueError("yfinance returned no data for the requested universe")` | propagate | caught, `"  [data] Tiingo fallback unavailable: {exc!r}"` (the whole block, construction and concat included, is inside the `try`) |
| persist | none | `loader._cache_to_db` inside `try`, `"  [data] failed to persist gap fetch: {exc!r}"` | same, `"  [data] failed to persist Tiingo data: {exc!r}"` |
| merge | | `pd.concat([df, extra], axis=1)` then `sort_index(axis=1, inplace=True)` | same |

The cache route: `CachedYFinanceLoader()` then `fetch_ohlcv(tickers, interval, start_date=, end_date=,
limit=None)`; an exception in either clears the writer and prints
`"  [data] DB cache unavailable ({exc!r}); falling back to yfinance only"`; an empty result keeps the writer
(risk 14). The negative cache: load before anything is fetched (`"  [data] negative cache: skipping {n}
known no-data symbols"` only if non-empty; on any exception `"  [data] negative cache unavailable
({exc!r}); fetching all"` and an empty skip set); record after the gaps (`"  [data] negative cache:
recorded {n} no-data symbols"`, or `"  [data] negative cache write failed ({exc!r})"`), computed from the
symbols neither provider supplied and not skipped. The closing steps: window trim (`df.index >= start_ts`,
`<= end_ts`), coverage filter, `"  [data] {rows} rows x {n} symbols ({interval})"`, `df.columns.names =
["symbol", "field"]`.

**Interfaces** (`fetch_chain.py`; no import of `lab`, `paper` or `cli`):
```python
class OhlcvProvider(Protocol):
    def fetch_ohlcv(self, symbols: list[str], interval: str, start_date=..., end_date=...) -> pd.DataFrame: ...
class CacheWriter(Protocol):
    def persist(self, df: pd.DataFrame, interval: str) -> None: ...
class NegativeCache(Protocol):
    def load_skip(self, interval: str) -> set[str]: ...
    def record(self, symbols: list[str], interval: str) -> None: ...
@dataclass(frozen=True)
class GapFiller:
    make: Callable[[], OhlcvProvider]
    intro: str                  # format string with {n}
    announce_before_make: bool  # True for the yfinance gap, False for Tiingo
    persist_error: str          # label after "failed to persist "
    swallow_errors: str | None  # None: errors propagate; else the label after "[data] " (Tiingo)
class OhlcvFetchChain:
    def __init__(self, *, primary: Callable[[], OhlcvProvider], fillers: Sequence[GapFiller],
                 cache: Callable[[], OhlcvProvider] | None = None,
                 negative_cache: NegativeCache | None = None) -> None
    def fetch(self, symbols: list[str], start: str, end: str | None = None, interval: str = "1d",
              min_coverage: float = 0.0) -> pd.DataFrame
def fetch_stock_ohlcv(symbols: list[str], start: str, end: str | None = None, interval: str = "1d",
                      use_db_cache: bool = True, min_coverage: float = 0.0,
                      negative_cache: NegativeCache | None = None) -> pd.DataFrame
```
The cache is a provider that also has `persist`. The primary fallback's loader instance is the first
filler's provider as well (the "reuse" row). `fetch_stock_ohlcv` builds the default chain with
**factories that resolve the classes at call time from their modules**
(`from ggTrader.data.live import cached_yfinance_loader as m; return m.CachedYFinanceLoader()`),
because the characterization patches the module attributes and the original imports inside a `try`
(risk 15). The ticker normalization (`normalize_yf_ticker` from `ggTrader.data.core.index_constituents`,
already a `data` module), the `Timestamp(..., tz="UTC")` bounds and the sorted-set dedupe move with it.
All progress text stays `print` with the exact strings above (Phase 2's logging change does not touch
them).

- [ ] **Step 1: Failing tests** (`tests/data/test_fetch_chain.py`; fakes only; the guard applied; made-up
  tickers): one test per row and cell of the table above (each asserting requests, persisted frames in
  order, printed lines and the exception), plus `test_cache_empty_keeps_writer`,
  `test_cache_constructor_error_drops_writer`, `test_cache_fetch_error_drops_writer`,
  `test_factory_raising_import_error_is_cache_unavailable`, `test_chain_has_no_lab_or_paper_import`
  (AST), `test_negative_cache_is_optional_and_never_touched_when_absent`,
  `test_default_factories_resolve_classes_at_call_time` (patch the module attribute after building the
  chain; the patched class is used), `test_persisted_frames_keep_their_utc_index` (risk 12).
- [ ] **Step 2 to 4:** FAIL, implement (a straight extraction of `lab/data.py:67-195`, with the
  repeated blocks as one loop over `fillers` and the table's flags for the differences), PASS.
- [ ] **Step 5:** ruff; the task gate steps 1-3 (net C is untouched and green); Commit:
  `feat(data): OhlcvFetchChain holds the price-loader fallback, not yet wired`.

### Task 42: `lab.data.fetch_stock_ohlcv` as a wrapper; `paper/` imports from `data/live` (fetch worktree)

**Files:**
- Modify: `src/ggTrader/lab/data.py` (the body of `fetch_stock_ohlcv`, `:67-195`)
- Modify: `src/ggTrader/paper/signal_runner.py:32`, `src/ggTrader/paper/overlay.py:18`
- Modify: `tests/utils/test_import_rules.py` (one new test)
- Modify (docstring only): `src/ggTrader/paper/live_bars.py:5,32`

**Change:**
- `lab/data.py` keeps `fetch_stock_ohlcv(symbols, start, end=None, interval="1d", use_db_cache=True,
  min_coverage=0.0, use_negative_cache=False) -> pd.DataFrame` with its docstring. The body builds a
  `NegativeCache` adapter when `use_negative_cache` is true (its `load_skip` and `record` do
  `from ggTrader.lab import negative_cache` and call `negative_cache.load_skip_symbols(interval)` and
  `negative_cache.record_no_data(symbols, interval)` **at call time**, inside the chain's `try`, as the
  original imports did) and returns `fetch_chain.fetch_stock_ohlcv(...)`. The module's other names
  (`STOCK_BASE_CONFIG`, `load_ohlcv`, and so on) are untouched.
- `signal_runner.py`: `from ggTrader.data.live.fetch_chain import fetch_stock_ohlcv` (same bound name,
  same call `fetch_stock_ohlcv(symbols, start=..., end=...)` at `:182`). `overlay.py`: the same for
  `fetch_stock_ohlcv`, keeping `from ggTrader.lab.data import STOCK_BASE_CONFIG` (Deviation 10).
- New rule in `tests/utils/test_import_rules.py`: `test_paper_imports_only_the_config_constant_from_lab_data`:
  over `src/ggTrader/paper/*.py` (AST, all import forms of `ggTrader.lab.data`), the set of imported names
  is a subset of `{"STOCK_BASE_CONFIG"}`, and no paper file imports `ggTrader.lab.negative_cache`. It
  fails on today's tree (`signal_runner.py:32`, `overlay.py:18`) and passes after this task. `LAYER_RULES`
  and `LAYER_EXCEPTIONS` are unchanged; `data/live/fetch_chain.py` is covered by the existing `data`
  rule, which Task 41's AST test also checks directly.

- [ ] **Step 1:** Add the new rule and see it fail; record the two hits.
- [ ] **Step 2:** Make the three edits. Net C (28 tests) passes **with no edit** since Task 40;
  `tests/lab/test_data.py`, `tests/paper/test_signal_runner.py`, `tests/paper/test_trader.py` (its five
  `signal_runner.fetch_stock_ohlcv` patches) and net G (which patches the same name) pass.
- [ ] **Step 3: Mutation check (not committed):** make the wrapper resolve `negative_cache` functions at
  import time, or construct `CachedYFinanceLoader` at import, or merge the two gap blocks' messages;
  confirm net C fails each; revert.
- [ ] **Step 4:** The task gate (V: nothing pinned changes), oracle label `t42`.
- [ ] **Step 5:** Commit: `refactor(data): the price-loader fallback lives in data/live; lab.data and paper use it`.

### Task 43: `TraderConfig` (trader worktree, after Task 39 and the fetch merge)

**Files:**
- Create: `src/ggTrader/paper/config.py`
- Modify: `src/ggTrader/paper/trader.py`
- Test: `tests/paper/test_trader_config.py`

Merge `refactor/phase-4-fetch` into `refactor/phase-4` first.

**Current environment reads in `trader.py` (re-grep `cash_sweep.`, `catastrophe_stop.catastrophe`,
`configured_untradable`, `config_from_env`):** `catastrophe_stop.catastrophe_stop_enabled()` and
`catastrophe_stop_pct()` (`run`, `:698-706`); `cash_sweep.sweep_enabled()` and `sweep_symbol()`
(`:867-868`); `configured_untradable()` (`:884`); `cash_sweep.reserve_pct()` (`:1194` and again `:1356`);
`cash_sweep.min_clip_usd()` and `cash_sweep.buy_trigger_pct()` (`:1357-1358`); and `config_from_env()`
in `__init__` (`:160`, **stays**, Deviation 3). The accessors live in `cash_sweep.py:44-80`,
`catastrophe_stop.py:38-52`, `untradable.py:38-43` and are the single place the variable names are
written; `TraderConfig` calls them and adds no new `os.environ` access (the harness `ENV_VARS` list and
its grep test stay whole).

**Interfaces** (`paper/config.py`):
```python
@dataclass(frozen=True)
class TraderConfig:
    dry_run: bool
    read_only: bool
    stop_enabled: bool
    stop_threshold: float | None            # None when disabled or invalid
    stop_threshold_error: ValueError | None
    sweep_enabled: bool
    sweep_symbol: str | None                # None when the sweep is off
    untradable: frozenset[str]
    sweep_reserve_pct: float | None
    sweep_min_clip_usd: float | None
    sweep_buy_trigger_pct: float | None
    sweep_parse_errors: Mapping[str, ValueError]  # keyed by accessor name, empty when all parsed
    @classmethod
    def from_env(cls, *, dry_run: bool, read_only: bool) -> TraderConfig
    def reserve_pct(self) -> float            # re-raises the stored error, else returns the value
    def min_clip_usd(self) -> float
    def buy_trigger_pct(self) -> float
```
`from_env` never raises for a malformed float: it stores the `ValueError` the accessor raised (the same
object, so the message `could not convert string to float: 'abc'` is the same). It reads the stop
threshold only when the stop is enabled (as `run` does today), and the sweep symbol and the three floats
only when the sweep is enabled (Deviation 4: nothing reads them otherwise; scenario
`sweep_reserve_invalid_but_sweep_off`). **This is a per-run snapshot**: the sweep and untradable values
are read at the top of `run()`, earlier than today (Deviation 4), not at their old call sites; only the
point where a stored float error is raised (or swallowed) stays where it was.

**Change in `trader.py`:** `run()` builds `cfg = TraderConfig.from_env(dry_run=self._dry_run,
read_only=self._read_only)` as its first statement and stores it as `self._cfg`; if `cfg.stop_enabled`
and `cfg.stop_threshold_error` it sends `<b>🛑 HALTED:</b> {exc}` in the same `try` structure and raises
the stored error (text and order as `:698-706`); `threshold = cfg.stop_threshold` is passed to `_run`
(signature unchanged). In `_run`: `sweep_on = self._cfg.sweep_enabled`, `sweep_sym = self._cfg.sweep_symbol`,
`configured_untradable()` becomes `self._cfg.untradable`, and the three float reads become
`self._cfg.reserve_pct()`, `min_clip_usd()`, `buy_trigger_pct()` **at the same call sites** (the last two
stay inside the sweep-buy `try`, so a stored error is raised and swallowed exactly where today's parse
error was; the values themselves come from the snapshot taken at the top of `run()`). The `cash_sweep` module stays imported for `compute_*`, `SweepAction` and
`SWEEP_TRADE_REASON`. After this task `grep -n "os.environ\|getenv" src/ggTrader/paper/trader.py` is empty.

- [ ] **Step 1: Failing tests:** `test_from_env_defaults` (no variable set), `test_from_env_reads_each_setting`,
  `test_invalid_stop_pct_is_stored_not_raised`, `test_invalid_sweep_floats_are_stored_not_raised`,
  `test_accessors_re_raise_the_stored_error_with_the_original_message`, `test_sweep_off_reads_no_float`
  (a bad `SWEEP_MIN_CLIP_USD` with the sweep off yields no stored error), `test_config_is_frozen`,
  `test_from_env_adds_no_new_environment_variable` (greps `config.py` for `os.environ`: none),
  `test_trader_py_reads_no_environment` (AST of `trader.py`),
  `test_env_is_read_at_run_time_not_construction` (construct, then `monkeypatch.setenv`, then run: the
  setting applies; the stop-threshold alert and raise keep their text),
  `test_early_exits_with_a_malformed_sweep_float_return_their_normal_result` (market closed, stale
  `as_of` with `SWEEP_CASH_RESERVE_PCT=abc` and the sweep on: same results and same Telegram text as
  with the variable unset, nothing raised; for `lock_verify_failure`, preserve the existing reserve
  parse error before the lock check, with that environment added through `Scenario.env`),
  `test_two_runs_on_one_trader_each_take_their_own_snapshot` (run, change the environment, run again:
  the second run uses the new values, and one run never mixes two values),
  `test_catastrophe_stop_error_is_still_raised_before_the_lock` (a bad `CATASTROPHE_STOP_PCT` with the stop
  on alerts and raises with a live trader before `paper_run_lock` is entered).
- [ ] **Step 2 to 4:** FAIL, implement, PASS. The scenarios added in Task 32 (`sweep_min_clip_invalid_fail_soft`,
  `sweep_buy_trigger_invalid_fail_soft`, `sweep_reserve_invalid_ends_run`, `sweep_reserve_invalid_but_sweep_off`,
  `untradable_symbols_from_env`, `catastrophe_enabled_pct_blank`, `invalid_catastrophe_threshold`) and the
  existing 96 pass byte-identical; `ENV_VARS` and its grep test are unchanged.
- [ ] **Step 5: Mutation check (not committed):** make `from_env` raise on a bad `SWEEP_MIN_CLIP_USD`;
  confirm `sweep_min_clip_invalid_fail_soft` fails; read `reserve_pct()` once at the top; confirm a
  transcript fails; revert.
- [ ] **Step 6:** The task gate, oracle label `t43`.
- [ ] **Step 7:** Commit: `refactor(paper): TraderConfig holds the run-time settings, one snapshot per run`.

### Task 44: Integration, docs and the Phase 5 hand-off

**Files:**
- Modify: `docs/architecture.md` (the paper-trader and data sections: name `models.py`, `ports.py`,
  `ledger.py`, `config.py`, `data/live/fetch_chain.py`, the ledger implementations and the string-patch
  adapter, and say which is temporary)
- Modify: `docs/changelog.md` (one entry: what moved, "no order, SQL statement, Telegram line or price
  row changed", the 96 plus added scenarios, the SQL golden, the one dropped bind key, the three renamed
  test tokens; the entry is merged with Phase 3's if both landed)
- Modify: `docs/cli_reference.md` only if `ggt paper` output changed (it must not)

- [ ] **Step 1: Merge** the three chains into `refactor/phase-4` (ledger and fetch chains were merged at
  Tasks 39 and 43; confirm `git log --oneline refactor/phase-4 --not refactor/phase-0-2` lists every task).
- [ ] **Step 2: Check each line of "What must be true when Phase 4 ends"** with its command (the greps
  for `_write`, `os.environ`, tuple indexing, the `lab.data` import rule; `git diff --stat BASE4 --
  tests/paper/transcript tests/paper/golden/trader_transcripts` shows only Task 32's additions;
  `sha256sum -c` on the hash file OK).
- [ ] **Step 3: Re-measure the Phase 5 anchors** and write them in the hand-back: `_run`'s range, the
  ten `RunResult` sites, the 26 ledger call sites, the `if self._dry_run:` branches of `_run`
  (stop, sweep sell, sell, buy and sweep-buy blocks), the length and parameter count of `run`, `_run`, `__init__`.
- [ ] **Step 4:** `./offline.sh scripts/run_tests.sh` (fast core) and the ruff checks pass. Docs only;
  no code changes.
- [ ] **Step 5:** Commit: `docs: refactor phase 4 (paper trader interfaces and records, price-loader chain)`.

### Task 45 (OPTIONAL, gated on O-P4-1): write functions run inside `engine.begin()`

**Do not start without a recorded owner answer. Default: skip.**

**What it is:** the eleven `connect()` plus `commit()` write functions become `with
_get_engine().begin() as conn:` with the explicit `conn.commit()` removed. They are exactly:
`init_paper_schema`, `log_trade`, `log_pending_order`, `clear_pending_order`, `mark_pending_order_stale`,
`log_snapshot`, `save_peak_value`, `save_rebalance_state`, `log_dividend_accrual`,
`save_split_correction`, `clear_split_correction`. The four functions already on `begin()`, every read
and `paper_run_lock` are not touched. What changes in a recorded session: `begin` then `commit`
replaces `connect` then `commit` then `close`; on an exception a `rollback` replaces a bare `close`;
`log_trade`'s early return commits an empty transaction. What it costs: only the tests of those eleven
mutators. Of the 38 test functions in `tests/paper/test_paper_persist.py` that configure
`.connect.return_value` (most are read mocks), the twelve that exercise these mutators are the ones to
change: `test_legacy_trade_checks_existing_rows_under_lock` (the builder confirms which mutator it calls),
`test_creates_tables`, `test_inserts_trade_row`, `test_inserts_snapshot_row`,
`test_save_rebalance_state_inserts_row`, `test_save_peak_value_upserts_row`,
`test_mark_pending_order_stale_updates_row`, `test_creates_dividend_accrual_table`,
`test_log_dividend_accrual_inserts_row`, `test_creates_split_state_table`,
`test_save_split_correction_upserts` and `test_clear_split_correction_deletes_row` (list counted on
2026-11-18 by the AST of that file; re-derive it in the task). The read mocks and the SQLite runs are
left alone.

- [ ] **Step 1: Failing tests.** The reference golden `persist_sql.json` stays immutable. Add a second
  file, `tests/paper/golden/persist_sql_begin.json`, recorded by hand from the reference by a reviewed
  transformation in the test module (`to_begin_shape(reference_entry)`), and compare the new code to it,
  for **every case of the eleven mutators**, enumerated: (a) success (`begin`, the same `execute` events,
  `commit`); (b) every early-return and conflict path, i.e. `log_trade`'s identical-trade early return
  (now ends with `commit` and no insert), its conflicting-trade `ValueError` (ends with `rollback`), its
  new-trade insert, and its `reason` omitted and given variants; (c) the `<name>__execute_fails` case of
  Task 31 for each of the eleven and its second failure point (`init_paper_schema` at statement 6, `log_trade`
  at the insert), each ending with `rollback` and the exception propagated; (d) `init_paper_schema`'s eleven
  statements in order. A test asserts the **SQL texts and bind dictionaries of every case are equal to the
  reference**, so only the transaction events differ. Then change only the twelve tests above to mock
  `.begin()`. All of these fail on the current code.
- [ ] **Step 2:** Make the change in `PgPaperLedger` only; the delegations are untouched.
- [ ] **Step 3:** The task gate, oracle label `t45`; Commit: `refactor(paper): ledger writes run inside engine.begin()`.

**Phase 4 gate and hand-back:**
- The test-outcome check passes against `p4base-core.xml,p4base-studies.xml`, with the tests of Tasks
  31, 32, 34-43 named in `--must-pass`, and the real-tape and live-golden tests passing, not skipping.
- The oracle check (label `phase4`) exits 0 against `$E/baseline`.
- `sha256sum -c $E/p4-fixed.sha256 $E/p4-pins.sha256` is all OK (the 96 original goldens and
  `harness.py` unchanged since Task 30).
- The pin check (L and K) passes in three processes; V passes with the counts in `.github/workflows/ci.yml`.
- `ruff check .` and `ruff format --check .` are clean; `tests/utils/test_import_rules.py` shows the same
  one layer exception and the new `paper` rule green.
- Codex Sol and Claude Opus each review the whole branch in a fresh context, given only the evidence and
  the claims. Sol also checks this plan against design 5.2 and the eleven deviations. Opus takes the
  verdicts on Deviations 1, 3 and 4 and on the SQL equivalence.
- Report to the owner with: `BASE4` and the head commit, the line counts of `trader.py`, `persist.py` and
  `lab/data.py` before and after, the Phase 5 anchors, and usage. Merge only on their say-so. Nothing
  is deployed: the image rebuild and `docker compose pull` are not part of this phase, and the trader
  stays off.

## Execution roles (`CLAUDE.md`, delegation preference of 2026-11-08)

**Quota first.** The agent cannot see plan usage. Before Task 30 ask the owner for current headroom and
size the run to it. If it is unknown, run Phase A and Phase B only and report. The phases and the number
of subagent runs expected:

| Phase | Tasks | Builder runs | Reviewer runs |
|---|---|---|---|
| A: pins | 30 to 33 | 3 (31, 32 Sonnet; 30 and 33 orchestrator, Haiku inventory) | 2 Sol, 1 Opus (goldens) |
| B: records | 34 to 36 | 3 Terra | 1 Opus (all three) |
| C: ledger | 37, 38, 39 | 3 Terra | 1 Opus (38 SQL equivalence; 39 trader) |
| D: fetch | 40 to 42 | 3 Terra or Luna | 1 Opus |
| E: config and docs | 43, 44 | 1 Terra, 1 Sonnet | 1 Opus, 1 Sol (docs) |
| optional | 45 | 1 Terra | 1 Opus |
| branch gate | | | 1 Sol, 1 Opus |

Phases C and D can run together (two worktrees) if headroom allows; otherwise C first, since the trader
chain waits for it. Usage is reported after each phase so the owner can stop early.

- **Orchestrator:** Claude Sonnet. It re-verifies every task itself: it reads the diff (with
  `--color-moved`), reruns the task gate and the hash checks, and accepts no builder claim
  unchecked. It runs the oracle and `--full` runs one at a time.
- **Builders:**
  - Claude Sonnet: Tasks 31 and 32 (they need judgment about what each case or scenario must show).
  - Codex Terra (or Luna, or Claude Sonnet): Tasks 34 to 43 and 45.
  - Claude Haiku: Task 30's line anchors and inventories, with every figure checked by a stronger model.
- **Reviewers, never from the builder's own family:**
  - Codex Sol reviews Claude-built work (Tasks 31, 32, 44) and the docs.
  - Claude Opus reviews Codex-built work (Tasks 34 to 43), reads the new goldens (Task 33 Step 4), and
    takes the critical judgments: the SQL-equivalence verdict, the transaction-shape deviation, the
    environment-timing deviation, and the final synthesis.
  - The whole branch gets both Sol and Opus, each in a fresh context given only the evidence and the
    claims. The pins (Tasks 31 and 32) are written by a different family from the one that refactors
    under them (Terra, Codex), which is the independence this phase relies on.
- Delegate prompts carry the task text and the file paths only; no private account records.
- The lead remains responsible for the final call.
