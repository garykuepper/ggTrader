# Refactor phase 5 (split `PaperTrader._run` into step objects, migrate the trader tests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 826-line `PaperTrader._run` into a short orchestrator that calls one step object at
a time (`PreflightGate`, `PortfolioView`, `RiskAssessor`, planning functions, `OrderExecutor` /
`DryRunExecutor`, `CorporateActions`, `RunReporter`), and move the trader's legacy tests off string-patched
persist names, without changing one order, one Telegram line, one SQL statement, one log line, one
clock read position or one price row.

**Architecture:** Same method as Phases 0-4, with one addition. Safety nets first (Tasks 46-51): the
103 transcripts and the SQL golden stay byte-identical and are joined by named tests for the seven rules
of design 5.1, a frozen copy of today's `_run` (`_run_legacy`, hash-pinned), a seeded differential test
(old run against new run, transcript plus log records, plus recorded per-seed digests that outlive the
copy) and a ratchet on `_run`'s size. Then the pieces are built inert (Tasks 52-54), the bodies of the
order and corporate-action methods are moved behind one-line forwards (Tasks 55-56), and the regions of
`_run` are cut out front to back, one commit each (Tasks 57-63), every commit against all the nets.
Legacy tests are migrated last, in eight batches that keep every node id (Tasks 65-73); only then is the
frozen copy deleted (Task 74).

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

## Scope

**In this plan** (design 5.1 and the phase 5 row of section 7):
1. The step objects of the table in design 5.1, in the modules listed under "Target structure".
2. The named test for each of the seven rules of design 5.1, run against both old and new code.
3. The seeded comparison of the old run with the new one ("planning equivalence").
4. Migration of `tests/paper/test_trader.py` from 239 persist-name string patches to an
   `InMemoryLedger` rig, keeping all 197 node ids.
5. Deleting `_run_legacy` and the implementation switch, then ruff complexity rules for the survivors,
   docs and one changelog entry.

**What Phase 5 does NOT do:**
- It does not touch `src/ggTrader/paper/persist.py` (no SQL, no transaction mode, no bind dictionary:
  Task 45 of Phase 4 stays skipped), `risk.py`, `cash_sweep.py`, `catastrophe_stop.py`, `gates.py`,
  `split_check.py`, `dividend_check.py`, `untradable.py`, `notifier.py`, `alpaca_broker.py`,
  `signal_runner.py`, `overlay.py`, `cli/main.py`, `data/`, `lab/`, `engine/`, anything under
  `research/`, or the oracle and legacy scripts. The only edit in `ledger.py` is one docstring line.
- It does not change `ggt paper` output: `run()` still returns `dict`, with the three shapes and key
  orders pinned by Phase 4 (`tests/paper/test_phase4_pins.py`).
- It does not change a Telegram message. Owner decision 4 of 2026-11-18 (the "total" percentage against
  both bases) is a separate slice with regenerated goldens, after this phase's gate.
- It does not fix quirks found on the way (listed under "Preserved quirks"); it records them.
- It does not run `ggt paper --live`, touch cron, the `ggtrader_live` container or deploy. The trader
  stays off.
- It does not merge, push or deploy. Branch `refactor/phase-5` ends at the Task 78 hand-back.

## Owner decisions and open questions needed before execution

A default is not an approval. Nothing here runs until the owner says so.

- **O-P5-1, the string-patch adapter (`ModuleLedger`).** Design section 7 says Phase 5 deletes it. The
  transcript harness (`tests/paper/transcript/harness.py`, hash-pinned, immutable) patches the 25 persist
  names in the `ggTrader.paper.trader` namespace and relies on `ModuleLedger` to make that reach the
  trader. Deleting the adapter needs either a harness change (it would break the immutability the owner
  asked for) or a test-only copy of the adapter. *Default: keep `ModuleLedger` as the transcript seam,
  delete everything else the design lists, and do not run Task 75.*
- **O-P5-2, `RunContext` and the clock.** Design 5.1 says the context holds `now_pt` and "the clock is
  read once". Today `now_pt` is read after signal generation (minutes after the run starts), at the
  late-run gate, and the cutoff is 12:55 PT for a 12:45 PT cron. Reading it once at the top would let a run
  that began on time but finished its signals late slip past the gate. *Default: `today_et` is read once, at
  the freshness gate where it is read now; `now_pt` is read at the late-run gate; no other clock read
  moves (Deviation 1).*
- **O-P5-3, quota and caps.** The owner reported ample Claude and Codex quota on 2026-11-19 (recorded;
  the agent cannot see usage, so usage is still reported after each phase). Caps: every pytest run
  `GGT_TEST_MEM_MAX=3G`, serialized; the p5 oracle driver only, `GGT_RUN_MEM_MAX=6G`, which the owner
  approved after a 3G run failed (`run_capped.sh` default). Nothing else exceeds 3G. A 137 or 143 exit
  stops the task and is reported, never retried with a higher cap.
- **O-P5-4, what "kept seams" mean.** Eight private `PaperTrader` methods are assigned or patched by tests
  (`_submit_order`, `_poll_orders`, `_reconcile_order_intents`, `_reconcile_pending_orders`,
  `_missed_exits`, `_compute_split_corrections`, `_accrue_dividends`, plus `_risk`, `_dry_run` and
  `_paper_lock` attributes). *Default: they stay as one-line forwards after the bodies move (Seam rule
  S1); retiring them is a separate owner-approved slice.*

## Global Constraints

> **Heavy-job rules (AGENTS.md, 2026-11-18).** One heavy job at a time, ever: one pytest run or one oracle
> driver. Check `ps -eo args | grep -E "pytest|run_tests|p5_wfo|p7_live"` before starting one. A run that
> prints `waiting for another heavy ggTrader job` waits; nobody kills another run or bypasses
> `scripts/run_tests.sh` and `scripts/run_capped.sh`. Exit 75 is busy or short of memory, 137 or 143 is a
> hang: stop, report, never raise a cap. Builders run only small targeted paths. **Every delegated brief
> contains this block verbatim and adds: "You must not run `gate_oracles.sh`, `scripts/p5_wfo_validation.py`,
> `scripts/p7_live_golden.py`, `tests/paper/` as a directory, a whole tier or `--full`; the orchestrator runs
> those." (A Phase 4 builder ran the oracles anyway; it is recorded as a boundary violation.)**

Everything in the Phase 0-2 and Phase 4 plans' Global Constraints applies and is not repeated: offline by
construction (`./offline.sh`, scrubbed environment, no `.env`, no database address), tests only through
`scripts/run_tests.sh`, never `ggt paper --live`, no edit under `research/`, `ohlcv.timestamp` stays naive
UTC, ruff clean, commits end with the attribution line from the session instructions, synthetic data only in
delegate prompts and goldens. Phase 5 adds:

- **Never real network or database, in any test or helper.** New tests build `PaperTrader` only with the
  harness fakes (`FakeBroker`, `RecordingNotifier`) or `MagicMock`, and only `InMemoryLedger` or the
  harness's wrapped persist names. They never construct `AlpacaBroker`, `TelegramNotifier`, `PgPaperLedger`
  or call `run_paper_trading`. The connection guard (net N) stays on; a new AST test (Task 50) forbids the
  new modules from importing `psycopg2`, `sqlalchemy`, `requests`, `httpx`, `alpaca*`,
  `ggTrader.paper.persist`, `ggTrader.paper.alpaca_broker`, `ggTrader.paper.notifier`,
  `ggTrader.paper.signal_runner` or `ggTrader.lab`.
- **Everything is bounded.** The equivalence tests use fixed seed counts (100 in the fast core, 600 in the
  `slow` tier, never read from the environment), every test has the default 180 s timeout, the 600-seed test
  carries `@pytest.mark.timeout_seconds(600)`, and every pytest run keeps `GGT_TEST_MAX_SECONDS` at its
  default. No loop in a test depends on a wall clock; the harness's fake clock and its runaway guard
  (`MAX_POLLS = 200`) stay on.
- **Immutable until the owner says otherwise** (hash records, Task 46; every task gate checks them):
  the 103 goldens in `tests/paper/golden/trader_transcripts/`, `tests/paper/transcript/harness.py`,
  `tests/paper/transcript/scenarios.py`, `tests/paper/test_trader_transcripts.py`,
  `tests/paper/golden/persist_sql.json`, `tests/paper/test_persist_sql_golden.py`,
  `tests/paper/test_phase4_pins.py`, `tests/lab/test_fetch_stock_ohlcv_chain.py` and
  `src/ggTrader/paper/persist.py`. Phase 5 never regenerates a golden (no regeneration switch exists in
  any new test). The pins Phase 5 adds (Tasks 47-50) become immutable at Task 51; a change to one needs a
  written amendment note approved by Opus.
- **The per-run settings snapshot is preserved.** `run()` builds `TraderConfig.from_env(...)` once, at its
  first line, and stores it as `self._cfg` (Phase 4, Deviations 3 and 4). `RunContext.config` is that
  object. No new module reads the environment or calls `cash_sweep.sweep_enabled`, `sweep_symbol`,
  `reserve_pct`, `min_clip_usd`, `buy_trigger_pct`, `catastrophe_stop.catastrophe_stop_enabled`,
  `catastrophe_stop_pct` or `untradable.configured_untradable` (an AST test, Task 50). A malformed float
  keeps its fail time: the reserve ends the run where it did, the other two sweep floats fail soft inside
  the sweep-buy `try`, and the stop threshold error is raised before the lock.
- **Transaction mode and SQL do not change.** `git diff BASE5 -- src/ggTrader/paper/persist.py` is empty
  at every gate; `ledger.py` differs by the one docstring line of Task 52.
- **Log records are part of the contract.** Moved code logs through
  `logging.getLogger(TRADER_LOGGER_NAME)` (`ggTrader.paper.trader`, a constant added in Task 52), so
  `caplog` filters and `test_swallowed_exceptions_are_logged.py` keep working; the differential test
  compares (logger, level, message) per record.
- **Layer rules.** The new modules sit in `paper/`; they import nothing from `lab` except what `paper/`
  imports today (`STOCK_BASE_CONFIG`, in `overlay.py`, which Phase 5 does not touch). `tests/utils/test_import_rules.py`
  and `tests/paper/test_paper_boundary.py` stay green without changes.
- **Names that must stay importable and patchable** (Task 50 turns this into a test that scans the
  tests for dotted strings): everything Phase 4 lists, plus `ggTrader.paper.trader.find_split_applied_symbols`,
  `catastrophe_stop`, `cash_sweep` (module bindings patched by attribute), `time`, `datetime`, `uuid4`,
  `_today_et`, `_now_pt`, `generate_core_signals`, `refresh_benchmark_tape`, `live_membership_source`,
  `require_live_membership_spec`, the 26 persist names, `AlpacaBroker`, `TelegramNotifier`,
  `DryRunNotifier`, `PaperTrader` and the eight private seams above. Extracted code calls
  `catastrophe_stop.<fn>` and `cash_sweep.<fn>` through the module attribute, never `from ... import`.

## Review Focus

The inputs and conditions the spec implies and the early tasks do not exercise; each has a test in the task
named.

1. **A planner failing in the middle of a stage.** Today the buy loop evaluates `check_concentration`
   after the previous buy was submitted, so an exception there leaves earlier orders live. Planners are
   therefore lazy generators consumed by the orchestrator, never lists computed up front. Test: Task 54
   `test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed`, and the seeded
   differential over mid-stage raises (Task 49).
2. **The late-run gate and the clock.** A run whose signals take minutes must still be gated by the time
   after the signals. Test: Task 48 `test_hook_call_order_for_a_live_run` (literal event order of hook calls
   against broker calls) and `test_late_run_gate_reads_the_clock_after_signal_generation`.
3. **An environment change in the middle of a run** (a setting edited while the run is in flight). Test:
   Task 50 `test_env_changed_after_the_first_broker_call_is_ignored_by_every_stage`.
4. **Malformed sleeves and signals** (an empty `sleeves` mapping, which ends in `StopIteration` today; a
   buy signal for the sweep symbol; the same symbol bought by two sleeves, which submits two BUYs today).
   Tests: Task 53 `test_empty_sleeves_ends_the_run_with_stop_iteration_after_the_fallback_notice`, Task 54
   `test_a_symbol_bought_by_two_sleeves_is_planned_twice_as_today`. These are recorded as candidate findings
   in the hand-back, not changed.
5. **The lock or the broker failing on order k of n** (a submit that raises after `k-1` orders went out; a
   `lock.verify` failure on the third order; a dry-run executor that is handed a broker that would submit).
   Tests: Task 49 `test_safety_invariants_over_all_seeds` and Task 55 `test_dry_run_executor_never_touches_a_submitting_broker`.

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

## The nets, by name

| Net | What it proves | Where |
|---|---|---|
| **T** transcripts | 103 scenarios of `PaperTrader.run`: every broker, persist, notifier, signal, lock, clock, uuid call (arguments deep-copied), the Telegram text, the result and the raise | `tests/paper/test_trader_transcripts.py`, 103 goldens (immutable) |
| **L** SQL golden | every SQL text and bind dictionary of the 31 persist functions | `tests/paper/test_persist_sql_golden.py`, `persist_sql.json` (immutable) |
| **K** shape pins | result key order, tuple sites, environment timing, poll fallback | `tests/paper/test_phase4_pins.py` (immutable) |
| **C** chain, **G** live golden, **N** guard, **S** surface, **I** imports, **V** pins | as in Phase 4 | unchanged |
| **R** rules (new, Task 48) | one named test per rule of design 5.1, each run against `_run` and `_run_legacy`; hook-order events | `tests/paper/test_run_rules.py` |
| **E** equivalence (new, Task 49) | 100 seeded scenarios: new run equals old run (transcript and log records); 600 seeds equal recorded digests; safety invariants over all seeds | `tests/paper/test_run_equivalence.py`, `tests/paper/equivalence/`, `tests/paper/golden/run_equivalence_digests.json` |
| **Q** seams and structure (new, Task 50) | legacy patch seams still intercept; imports and environment discipline of new modules; ratchet on `_run` | `tests/paper/test_seam_contract.py`, `tests/paper/test_run_structure.py` |
| **F** frozen copy (new, Task 47) | `_run_legacy` source hash | `tests/paper/test_run_legacy_frozen.py` |
| **O** oracle, **P** outcomes | as in Phase 0-2; baselines under `$E` | `gate_*.sh` copies in `$E/p5/` |

`$E` is `/home/flynn/ggTrader-refactor-evidence`. `T` below is shorthand for
`./offline.sh env GGT_TEST_MEM_MAX=3G scripts/run_tests.sh`.

**Why a frozen copy and digests both.** The copy (`_run_legacy`) localizes a divergence to a step: the two
transcripts are diffed. It cannot outlive Task 74. The 600 recorded digests are taken from the untouched
code in Task 49, before any extraction, so they keep guarding after the copy is deleted.

## Target structure and file ownership

New modules under `src/ggTrader/paper/` (each module docstring lists its public names, as Phase 4 did):

| Module | Holds | Built in | Cut from `trader.py` (anchors at `73c35417`) |
|---|---|---|---|
| `seams.py` | `RunHooks`, `TraderSeams` | 52 | none (new) |
| `models.py` (edit) | `IntentKind`, `OrderIntent`, `Skip`, `InvalidExit`, `Decision`, `StrategySignals`, `ExecutionBook`, `RiskVerdict`, `RunContext` | 52 | none |
| `planning.py` | `strategy_signals`, `assess_stops`, `plan_stops`, `validate_exits`, `plan_exits`, `plan_sweep_sell`, `plan_buys`, `plan_sweep_buy` (pure, lazy) | 53, 54 | decisions inside 1053-1397 |
| `execution.py` | `Executor` protocol, `OrderExecutor`, `DryRunExecutor` | 55, 60-62 | `_submit_order` 193, `_reconcile_order_intents` 224, `_flag_if_stale` 333, `_reconcile_pending_orders` 361, `_poll_orders` 428 |
| `corporate_actions.py` | `CorporateActions`, `missed_exits` | 56 | `_missed_exits` 305, `_compute_split_corrections` 483, `_accrue_dividends` 607 |
| `preflight.py` | `PreflightGate`, `PreflightResult`, `RunHalted` | 57 | `_run` 744-855 |
| `portfolio.py` | `PortfolioView` | 58 | `_run` 857-953 |
| `risk_assessor.py` | `RiskAssessor` | 59 | `_run` 955-1034 |
| `order_flow.py` | `OrderFlow`, `FlowOutcome` (the staged plan-then-execute glue) | 60-62 | `_run` 1048-1397 |
| `reporter.py` | `RunReporter` | 63 | `_run` 1036-1046, 1399-1553 |
| `constants.py` (edit) | `TRADER_LOGGER_NAME` | 52 | none |

Rows of design 5.1 not in the table: `TraderConfig` and `RunResult` exist since Phase 4 (`config.py`,
`models.py`) and are only consumed here; `RunContext` is in `models.py`.

`_run` regions (re-grep each quoted anchor): R1 schema init and reconcile 732-742; R2 preflight 744-855
(clock gate 744, signals try 761, sleeve log loop 774, late gate 791, tape refresh 799, fallback notice 810,
signal derivation 816, freshness gate 832); R3 portfolio 857-953; R4 risk 955-1034; R5 prior snapshot
1036-1046; R6a stops, exit validation, buy limits 1048-1181; R6b sweep sell, sells, sell settlement
1183-1301; R6c buys, settlement, sweep buy 1302-1397; R7 reporting 1399-1553.

**File ownership** (a task edits only the files in its own Files list; nothing else changes):

| Path | Edited by |
|---|---|
| `src/ggTrader/paper/trader.py` | 47, 52, 55-63, 74, 76 |
| `src/ggTrader/paper/models.py`, `constants.py`, `seams.py`, `ledger.py` (docstring only) | 52 |
| `planning.py` | 53, 54 (then only by 60-62 for a wiring fix, reported) |
| `execution.py` | 55, 60, 61, 62 |
| `corporate_actions.py` / `preflight.py` / `portfolio.py` / `risk_assessor.py` / `reporter.py` | 56 / 57 / 58 / 59 / 63 |
| `order_flow.py` | 60, 61, 62 |
| `tests/paper/test_run_legacy_frozen.py`, `tests/paper/golden/run_legacy.sha256` | 47 (deleted in 74) |
| `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py` | 48 |
| `tests/paper/equivalence/`, `tests/paper/test_run_equivalence.py`, `tests/paper/golden/run_equivalence_digests.json` | 49 (edited by 74) |
| `tests/paper/test_seam_contract.py`, `tests/paper/test_run_structure.py` | 50, then the ratchet constant by 57-63, 74 |
| `tests/paper/test_<module>.py` (one per new module) | the task that builds the module |
| `tests/paper/test_trader.py`, `tests/paper/support.py` | 65-73 |
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
    book_amount: float           # becomes SubmittedOrder.amount (position market value, or notional)
    sleeve: str | None = None; unrealized_pct: float | None = None; stop_threshold: float | None = None
@dataclass(frozen=True, slots=True)
class Skip: level: int; message: str          # message already formatted; logged on TRADER_LOGGER_NAME
@dataclass(frozen=True, slots=True)
class InvalidExit: symbol: str; error: str    # the orchestrator appends the error and blocks buys
Decision = OrderIntent | Skip | InvalidExit
@dataclass(frozen=True)
class StrategySignals:
    buys: tuple[tuple[str, str], ...]; sells: tuple[str, ...]      # (symbol, sleeve); sorted unique
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

# paper/planning.py -- pure; every planner returns an Iterator[Decision] (lazy) unless noted
class RiskLimits(Protocol):      # RiskGuard satisfies it; used read-only
    def max_new_positions(self, current_count: int) -> int: ...
    def sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]: ...
    def sleeve_position_notional(self, portfolio_value: float, sleeve_weight: float, scale: float) -> float: ...
    def check_concentration(self, symbol: str, positions: dict[str, dict], portfolio_value: float,
                            prospective_notional: float = 0.0) -> bool: ...
def strategy_signals(bundle: SignalBundle) -> StrategySignals
def assess_stops(corrected_positions: Mapping[str, Position]) -> frozenset[str]     # unassessable
def plan_stops(*, assessable: Mapping[str, Position], strategy_positions: Mapping[str, Position],
               corrected_positions: Mapping[str, Position], threshold: float,
               pending_symbols: frozenset[str]) -> Iterator[Decision]
def validate_exits(*, sells: Iterable[str], strategy_positions: Mapping[str, Position],
                   invalid_symbols: frozenset[str]) -> Iterator[Decision]            # InvalidExit only
def plan_exits(*, sells: Iterable[str], strategy_positions: Mapping[str, Position],
               invalid_symbols: frozenset[str], pending_symbols: frozenset[str]) -> Iterator[Decision]
def plan_sweep_sell(*, cfg: TraderConfig, account: Account, positions: Mapping[str, Position],
                    strategy_positions: Mapping[str, Position], sells: Iterable[str],
                    pending_symbols: frozenset[str], risk: RiskLimits,
                    buys_by_sleeve: Mapping[str, list[str]], notional_by_sleeve: Mapping[str, float],
                    slot_caps: Mapping[str, int], portfolio_value: float) -> OrderIntent | None
def plan_buys(*, buys_by_sleeve: Mapping[str, list[str]], notional_by_sleeve: Mapping[str, float],
              slot_caps: Mapping[str, int], strategy_positions: Mapping[str, Position],
              corrected_positions: Mapping[str, Position], stopped: frozenset[str],
              pending_symbols: frozenset[str], slots_available: int, max_positions: int,
              risk: RiskLimits, portfolio_value: float, open_by_sleeve: Mapping[str, int]
              ) -> Iterator[Decision]
def plan_sweep_buy(*, cfg: TraderConfig, mid_account: Account) -> OrderIntent | None  # may raise: caller's try

# paper/execution.py
class Executor(Protocol):
    def recover_intents(self) -> None: ...
    def submit(self, intent: OrderIntent) -> SubmittedOrder | None: ...        # None in a dry run
    def announce(self, intent: OrderIntent) -> None: ...      # the message sent after the intent is booked
    def poll(self, orders: Sequence[SubmittedOrder]) -> dict[str, OrderStatus]: ...
    def after_sells(self, filled: Mapping[str, OrderStatus]) -> None: ...
    def after_orders(self, orders: Sequence[SubmittedOrder]) -> None: ...
    def unfilled_exit_errors(self, orders: Sequence[SubmittedOrder], filled: Mapping[str, OrderStatus],
                             positions: Mapping[str, Position]) -> list[str]: ...
class OrderExecutor:                       # live; holds the bodies moved in Task 55
    def __init__(self, *, broker: Broker, notifier: Notifier, ledger: PaperLedger, hooks: RunHooks,
                 lock: Callable[[], RunLock | None], read_only: bool,
                 seams: TraderSeams | None = None) -> None
    def submit_order(self, side: str, symbol: str, amount: float, reason: str | None) -> str
    def poll_orders(self, orders: list[SubmittedOrder]) -> dict[str, OrderStatus]
    def reconcile_order_intents(self) -> set[str]
    def reconcile_pending_orders(self) -> set[str]
    def flag_if_stale(self, pending_order: dict, now: datetime) -> None
class DryRunExecutor(OrderExecutor): ...  # overrides only: pending-load failure, recover_intents, submit,
                                            # announce, after_sells, after_orders, unfilled_exit_errors

# paper/corporate_actions.py
class CorporateActions:
    def __init__(self, *, broker: Broker, ledger: PaperLedger, hooks: RunHooks) -> None
    def split_corrections(self, positions: dict, split_since: date, today: date) -> dict[str, float]
    def accrue_dividends(self, positions: dict) -> dict
def missed_exits(ledger: PaperLedger, positions: dict, last_exit: dict[str, str], already: set[str]) -> list[str]

# paper/preflight.py
class RunHalted(Exception): reason: str; result: RunResult
@dataclass(frozen=True)
class PreflightResult: bundle: SignalBundle; signals: StrategySignals; today_et: date
class PreflightGate:
    def __init__(self, *, broker: Broker, notifier: Notifier, hooks: RunHooks, membership: str | None,
                 read_only: bool, dry_run: bool, skipped_writes: list[str]) -> None
    def check(self) -> PreflightResult                    # raises RunHalted(reason, result) or the old errors

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
`seams.reconcile_intents()`; `pending = seams.reconcile_pending()`; preflight (`except RunHalted` returns its
result `.as_dict()`); build `RunContext`; `PortfolioView.load`; `RiskAssessor.assess` (a `hard_stop` returns
`RunResult([], [], [reason]).as_dict()`); `RunReporter.previous_value`; `OrderFlow.run`; `RunReporter.finish`;
return `.as_dict()`. `run()` keeps its 22 lines (snapshot, stop-threshold alert, lock).

## Seam rules (the string-patch contract, in force until the owner retires the seams)

- **S1. Private methods stay as forwards.** After a body moves, `PaperTrader._submit_order`,
  `_poll_orders`, `_reconcile_order_intents`, `_reconcile_pending_orders`, `_flag_if_stale`,
  `_missed_exits`, `_compute_split_corrections` and `_accrue_dividends` are one- or two-line forwards to
  the new class, with the same signature. New components reach them only through `TraderSeams`, whose
  lambdas look the method up on the instance at call time, so an assignment on the instance
  (`trader._submit_order = MagicMock(...)`) or a patch on the class (`patch.object(PaperTrader,
  "_poll_orders", spy)`) wins, as today.
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

## Deviations from the design

1. **`RunContext` carries `today_et`, `config` and `pending_symbols`, not `now_pt`.** `now_pt` is read at the
   late-run gate through `hooks.now_pt()` (O-P5-2). `today_et` is read once at the freshness gate, as
   today; `_submit_order` still reads `hooks.today_et()` per order (Phase 4 tests and
   `TestRunDateIsSessionDateNotAsOf` patch it between calls). Task 48 pins the literal order of hook calls.
2. **`PreflightGate.check()` returns a `PreflightResult` (bundle, derived signals, `today_et`), not the bare
   `SignalBundle`.** The signal derivation (lines 816-830) sits between the fallback notice and the
   freshness loop inside the gate, where it is today.
3. **Two modules the design does not name:** `seams.py` (the late-binding records of S1 and S2) and
   `order_flow.py` (`OrderFlow`, the staged glue). The glue is necessary: the order is plan, execute, plan,
   execute (a stop sell frees a slot before the sweep funding is estimated; the sweep sell is computed from
   the post-stop position set), so the planners cannot return one list. Without a home for it `_run` would
   stay near 200 lines.
4. **Planners are lazy generators**, not functions returning lists (Review Focus 1). They are still pure.
5. **`ModuleLedger` stays** (O-P5-1), and so do the eight private seams (O-P5-4).
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
| 3 | Planning everything, then executing, changes behavior when a stage raises midway or when a stop sell must precede the sweep estimate | `trader.py:1070-1336` | lazy planners (Review Focus 1); E over mid-stage raises; SM11 |
| 4 | `compute_sweep_sell_for_funding(...)` evaluates `account["cash"]` before `cfg.reserve_pct()`; a malformed reserve raises after live stop orders went out | `trader.py:1202-1208` | Task 54 keeps keyword-argument order; transcript `sweep_reserve_invalid_ends_run`; a live seed with a stop and a bad reserve |
| 5 | Which notices are inside a `try` and which are not (quirk 7) | R2, R4, R7 | T goldens (`failure_notice_fails`, `loss_halt_notice_fails_with_pending`); E with scripted notifier failures |
| 6 | Log records are not in the transcripts | everywhere | E compares (logger, level, message) between old and new; Task 50 logger-name test |
| 7 | `ruff --fix` (PostToolUse hook, pre-commit) deletes imports only tests patch, or only `_run_legacy` still needs | `trader.py:13-75` | Task 50 patch-target test; every kept name keeps its `# noqa: F401 - <why>` |
| 8 | `trader._dry_run = X` and `trader._paper_lock = MagicMock()` are assigned after construction | tests | S4, S5; Task 50 `test_dry_run_assigned_after_construction_selects_the_dry_executor` |
| 9 | Set iteration order leaks into messages or the result | stop and untradable code | sorted everywhere today; E runs under two `PYTHONHASHSEED`s |
| 10 | Aliasing: `ExecutionBook.buys` and `.sells` are returned as they are; `skipped_writes` is shared | `trader.py:1547-1553` | Task 63 `test_result_lists_are_the_books_own_lists` |
| 11 | `DryRunExecutor` must never submit; `read_only` must never write | executors | Task 55 test with a broker whose submit methods raise `AssertionError`; E invariants (a) and (f) |
| 12 | The recorded digests become wrong if the generator changes | `tests/paper/equivalence/generator.py` | Task 49 pins the generator hash inside the digest file |
| 13 | A test conversion that keeps its name but loses its teeth | Tasks 65-73 | assertion-count guard (never fewer assertions than at `BASE5`), mutation per batch, Opus review |
| 14 | A new `trader.py` persist import without a stub in the legacy autouse fixtures | AGENTS.md | none added; Task 50 test lists the trader's persist names (26) |
| 15 | Unused `_run_legacy` helpers or stale imports after Task 74 | `trader.py` | Task 74 removes only names `trader.py` no longer uses and that no test or harness patches (Task 50 list) |

## Order of work

```
46 setup ─ 47 frozen copy ─ 48 rules ─ 49 equivalence ─ 50 contracts ─ 51 nets gate (phase A)
  ─ 52 seams+records ─ 53 planning I ─ 54 planning II ─ 55 executors move ─ 56 corporate actions move (phase B-C)
  ─ 57 preflight ─ 58 portfolio ─ 59 risk ─ 60 flow I ─ 61 flow II ─ 62 flow III ─ 63 reporter+final _run ─ 64 wiring gate (phase D)
  ─ 65 rig+pilot ─ 66..73 batches M1..M8 ─ 74 delete the frozen copy (phase E)
  ─ [75 optional] ─ 76 lint+docs ─ 77 integration gate ─ 78 reviews and hand-back (phase F)
```
Everything is serial (one worktree, one heavy job at a time; this is what Phase 4 did). Inert tasks 52-54
and 56 could be built in parallel by different builders in separate worktrees, but their test runs queue behind
the one lock, so the plan does not count on it. Oracle and `--full` runs are never concurrent with anything.

**`_run` size ratchet** (`tests/paper/test_run_structure.py`, constant `RUN_MAX_LINES`; each wiring task first
lowers it, sees the test fail, then cuts). Starting value 830; after each task set the bound to the measured
line count rounded up to the next 5. Expected: 57 ≈ 740, 58 ≈ 650, 59 ≈ 575, 60 ≈ 470, 61 ≈ 380, 62 ≈ 300,
63 ≤ 45, and Task 74 keeps 45. `run()` keeps ≤ 25 lines.

**Standard mutations (SM)**, applied by the orchestrator, never committed, each reverted with `git
checkout -- <file>`; the builder re-locates the site in whichever module holds it (the mapping lives in
`$E/p5/mutation-sites.md`, updated by every wiring task). Each SM must fail at least one named rule test
**and** at least one transcript or equivalence test (SM9 and SM12 fail the rule tests and the event test
instead, because the harness pins both clocks and the environment).

| SM | Mutation | Must fail |
|---|---|---|
| 1 | drop the `stopped_symbols` guard in the buy loop | rule 1 |
| 2 | slots use `len(executed_sells)` instead of the regular-sell count | rule 2 |
| 3 | drop the clearing of the buys when an exit did not fully fill | rule 3 |
| 4 | drop `if invalid_exit_symbols: all_buys = []`, and separately drop `not invalid_exit_symbols` from the sweep-buy condition | rule 4 (two cases) |
| 5 | submit the split-corrected quantity | rule 5 |
| 6 | do not wrap the ledger read-only (writes happen) | rule 6 |
| 7 | swallow the submit exception and continue | rule 7a |
| 8 | drop the intent reconcile after the buys | rule 7b |
| 9 | read `now_pt` before generating signals | event-order test |
| 10 | poll sleep 1.0 to 0.5 | T (`clock.sleep`) |
| 11 | compute the sweep sell before the stops are executed | T, E |
| 12 | read `CASH_SWEEP_ENABLED` inside the run instead of the snapshot | env test |
| 13 | read-only skip message to a different logger | E, Task 50 logger test |
| 14 | trade alert shows the intended amount for a filled order | T |

## The task gate

Every code task, Tasks 52-63 and 74 (shorthand `T` above; every pytest command below is one bounded
run, and the exact spelling of `-m "not slow"` is confirmed in Task 46 Step 3 with `GGT_TEST_DRY_RUN=1`):
1. `.venv/bin/ruff check . && .venv/bin/ruff format --check .` clean.
2. **T and the hash records:** `T tests/paper/test_trader_transcripts.py`, no golden regenerated, then
   `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256` prints OK for every line, and
   `git diff --stat BASE5 -- <the immutable paths of Global Constraints>` prints nothing.
3. **The pins:** `T tests/paper/test_run_rules.py tests/paper/test_run_equivalence.py
   tests/paper/test_seam_contract.py tests/paper/test_run_structure.py tests/paper/test_run_legacy_frozen.py
   tests/paper/test_persist_sql_golden.py tests/paper/test_phase4_pins.py -m "not slow"`. Tasks 55-63 and 74
   also run the 600-seed digest test: `T tests/paper/test_run_equivalence.py -m slow`.
4. **The neighbourhood:** `T tests/paper tests/utils -m "not slow"` (the paper tests run under the guard N;
   `tests/utils` holds S and I). The live golden G and the other `slow` tests run at the phase gates.
5. The task's own new tests, named in the task, pass; the task's mutations fail the nets they must.
6. **The oracle O** (Tasks 55-63 and 74 only; label `t<task number>`): `$E/p5/gate_oracles.sh p5-t<NN>` and
   the oracle half of `gate_compare.sh`, exit 0 against `$E/baseline`, p5 under the 6G cap of O-P5-3. This is
   the spec's rule (design 6.5: after every task that changes code). The oracle drivers do not import
   `paper.trader`, so here it is mainly an import-time and shared-module slip detector; the owner may drop
   it for Phase 5 by saying so.
7. **Orchestrator re-verification:** it reads the diff with `git diff --color-moved=dimmed-zebra BASE..HEAD`,
   confirms every moved block is a pure move, reruns steps 1-4 itself, and does not pass on a builder's claim.
   A task that moved a body shows the moved block unchanged apart from `self.` to the new class's names.

Where a task below says "the task gate" it means steps 1-5 and 7, plus step 6 when step 6 lists the task. Tasks 47-50 (tests only)
run the steps with only the pin files that exist by then; the test-migration tasks 65-73 run steps 1-4 and 7, with the Task 65-73 protocol replacing step 5.

**The phase gates** (Tasks 51, 64, 77), run by the orchestrator one job at a time: the whole fast core
(`T`), the `--full` tier and `--studies` (`$E/p5/gate_tests.sh LABEL`, two junit files), the oracle check
(`$E/p5/gate_oracles.sh p5-LABEL`, p5 under the 6G cap approved in O-P5-3, p7, compared with `$E/baseline`,
exit 0), the test-outcome check against `p5base-*.xml` with the Phase 5 pins in `--must-pass`, `scripts/verify_pins.py` with the counts in
`.github/workflows/ci.yml`, the mutation matrix, both hash records, and the reviews of the roles table.

## Tasks

### Task 46: Branch, baselines, hash records and anchors

**Files:** none under `src/` or `tests/`. Evidence under `$E` (not in git).

- [ ] **Step 1: Preconditions.** The owner has approved execution; O-P5-1 to O-P5-4 are answered or their
  defaults accepted in writing; the plan commit is the head of `refactor/phase-4`; `git status --short` is
  empty; no heavy job runs.
- [ ] **Step 2: Branch.** `git switch -c refactor/phase-5` in `/home/flynn/ggTrader-refactor` (the worktree
  Phases 3 and 4 used; no second worktree). Record `BASE5=$(git rev-parse HEAD)` in
  `.superpowers/sdd/2026-11-19-refactor-phase-5/progress.md`. Check `.venv/bin/python -c "import ggTrader;
  print(ggTrader.__file__)"` names this worktree, `offline.sh` is present and excluded, the frozen tape's hash
  matches `data/snapshots/MANIFEST.json`, and `get_db_connection_string()` raises
  `ValueError: Database configuration not found`.
- [ ] **Step 3: Gate scripts and command spelling.** `mkdir -p $E/p5`; copy `gate_oracles.sh`, `gate_tests.sh`,
  `gate_compare.sh` from `$E` and change only the label prefix (`p5-`), the test baseline (`$E/p5base-core.xml,
  $E/p5base-studies.xml`) and a `MUST_PASS` argument appended to `--must-pass`; keep the three `diff`s. Print
  the exact pytest commands the wrapper would run for a path run with and without `-m "not slow"` using
  `GGT_TEST_DRY_RUN=1 ./offline.sh env GGT_TEST_MEM_MAX=3G scripts/run_tests.sh <paths> -m "not slow"` and write
  the confirmed spelling to `$E/p5/commands.txt`; if the wrapper rejects `-m`, the gate uses the spelling the
  dry run shows, and the plan's "not slow" means "the fast-tier markers".
- [ ] **Step 4: Baselines.** The test-outcome baseline is recorded fresh at `BASE5`: `$E/p5/gate_tests.sh
  p5base`, renamed to `$E/p5base-core.xml` and `p5base-studies.xml` (both exit 0; the live golden passes,
  not skips). The oracle baseline stays `$E/baseline`; for `BASE5` reuse `$E/merge/` only if
  `$E/merge-oracle-status.txt` is `p5 exit=0 / p7 exit=0`, the compare file shows exit 0, and the log names
  `73c35417`; otherwise run `$E/p5/gate_oracles.sh p5base` (6G cap).
- [ ] **Step 5: Hash records.** `$E/p5-fixed.sha256` = `$E/p4-fixed.sha256` (105 lines: 103 goldens,
  `harness.py`, `persist_sql.json`) plus the hashes of `tests/paper/test_persist_sql_golden.py`,
  `tests/paper/test_phase4_pins.py` and `src/ggTrader/paper/persist.py`; `$E/p5-pins.sha256` =
  `$E/p4-pins.sha256` (3 lines). `sha256sum -c` prints OK for all 111.
- [ ] **Step 6: Anchors (Haiku inventory, every figure checked by Sonnet).** Re-measure the region table
  of "Target structure", the ten `RunResult` sites, every `self._dry_run` and `self._read_only` read, the 28
  `self._ledger` calls, and the counts quoted here (4,224 lines, 197 tests, 439 dotted patches of which 239
  name a persist function, 211 of them inside tests and 28 in the autouse fixtures and `_intent_store`, and
  137 name `generate_core_signals`; 103 goldens). Write them to `$E/p5/anchors-trader.txt`.
- [ ] **Step 7:** Report usage to the owner.

### Task 47: Freeze a copy of today's `_run`, and the implementation switch (builder: Sonnet; reviewer: Sol)

**Files:**
- Modify: `src/ggTrader/paper/trader.py` (add `_run_legacy`, a byte-for-byte copy of `_run`, directly under it;
  its docstring says "frozen oracle copy of `_run` at 73c35417; deleted in Task 74; never edited")
- Create: `tests/paper/test_run_legacy_frozen.py`, `tests/paper/golden/run_legacy.sha256`,
  `tests/paper/equivalence/__init__.py`, `tests/paper/equivalence/impl.py`

**Interfaces:** `tests/paper/equivalence/impl.py` produces
`def run_with(impl: Literal["new", "legacy"], scenario: Scenario) -> dict` (the harness's
`run_scenario`, with `PaperTrader._run` replaced by `PaperTrader._run_legacy` inside a
`patch.object` context for `"legacy"`; the harness is not edited) and
`IMPLS: tuple[str, ...] = ("new", "legacy")`.

- [ ] **Step 1: Failing tests.** `test_run_legacy_exists_and_is_a_copy` (`PaperTrader._run_legacy` is a function with the same
  parameters as `_run`), `test_run_legacy_source_is_frozen` (`sha256` of the dedented source of
  `_run_legacy`, with its docstring line removed and the name on its `def` line normalized to `_run`, equals the
  constant in `run_legacy.sha256`, which is the hash of the same normalization of `_run` at `BASE5`, computed
  with `git show BASE5:src/ggTrader/paper/trader.py`),
  `test_both_implementations_give_identical_transcripts_for_every_golden_scenario` (for all 103
  `SCENARIOS`, `run_with("new", s) == run_with("legacy", s)`; trivially true now, the test that stays true).
  FAIL: `_run_legacy` is missing.
- [ ] **Step 2:** Implement: copy the method; compute and record the hash.
- [ ] **Step 3:** The new tests pass; the 103 transcripts (T) are byte-identical; `git diff BASE5 --
  src/ggTrader/paper/trader.py` shows only the added method.
- [ ] **Step 4: Mutation check (not committed).** Edit one character inside `_run_legacy`: the frozen test
  fails; revert. Edit one character inside `_run`: the 103-scenario switch test still passes only if no
  scenario reaches it; record which (a hit is the point of Task 49).
- [ ] **Step 5:** The task gate. Commit: `test: freeze a copy of PaperTrader._run as the oracle for phase 5`.

### Task 48: Named tests for the seven rules, hook order, run against both implementations (builder: Sonnet; reviewer: Sol)

**Files:**
- Create: `tests/paper/test_run_rules.py`, `tests/paper/rules_support.py` (a scenario builder over
  `dataclasses.replace(SCENARIOS[...], ...)`; scenarios and goldens are **not** added: `scenarios.py` is
  immutable)

Each test is parametrized over `IMPLS` (Task 47), builds its input from a named existing scenario
(`plain_buys_and_sells`, `catastrophe_stop`, `catastrophe_stop_at_position_cap`, `sell_partial_fill_working`,
`strategy_exit_invalid_qty`, `split_correction_held`, `dry_run_read_only`, `submit_error`,
`sell_rejected_by_broker`, `sweep_buy_after_strategy_orders`), cites it in the docstring and asserts on the
transcript's calls, never on a golden file.

- [ ] **Step 1: Failing tests, written against the old code (they fail only because the module is empty).**
  The rules and their assertions (target names are `broker.submit_sell`, `broker.submit_buy`,
  `persist.create_order_intent`, `notifier.send`):
  - `test_rule1_catastrophe_stopped_symbol_leaves_at_once_and_is_not_sold_twice_or_rebought`: the stopped symbol
    X also has a strategy sell and a buy signal; exactly one `submit_sell` for X (reason
    `catastrophe_stop`), no `submit_buy` for X, and X is absent from the slot count.
  - `test_rule2_slot_arithmetic_counts_regular_sells_not_all_sells`: `MAX_POSITIONS=3`, three held, one stopped
    and one regular sell, three buy signals: exactly one buy is submitted (counting stop sells too would allow two).
  - `test_rule3_a_sell_that_does_not_fully_fill_blocks_all_buys` (cases: partial-then-working, canceled,
    rejected): zero `submit_buy`, the error text `SELL X not fully filled (<status>); new strategy buys
    blocked` is in the result's `errors`.
  - `test_rule4_an_invalid_exit_quantity_blocks_buys` and
    `test_rule4_an_invalid_exit_quantity_blocks_the_sweep_buy_and_the_sweep_sell`: qty `0`, `None`, NaN, negative,
    on a stop and on a strategy exit; zero buys; with the sweep on, no sweep order either way.
  - `test_rule5_orders_use_the_brokers_uncorrected_quantity`: with a split correction on record, the
    `submit_sell` quantity equals the broker's raw `qty`; the corrected figures appear only in the daily
    summary and in the concentration check.
  - `test_rule6_a_read_only_run_writes_nothing_and_records_what_it_skipped`: no `persist.*` call that is a member of
    `WRITE_METHODS`, `skipped_writes` equals the golden's list, `result["skipped_writes"]` too.
  - `test_rule7a_an_exception_on_submit_ends_the_run_after_the_intent_is_saved`: the order of calls is
    `create_order_intent`, `lock.verify`, `submit_*` (raises), then the failure notice naming the stage, and no
    later `create_order_intent` or `submit_*`; the run raises.
  - `test_rule7b_a_rejected_order_settles_normally_and_other_exits_stay_eligible`: status `rejected` reaches
    `settle_order_intent`, the second exit is still submitted.
- [ ] **Step 2: Hook-order tests** (an event-logging copy of the environment builder in
  `tests/paper/test_trader_ledger.py::_trader_environment`, copied, not imported, so that file stays
  untouched): `test_hook_call_order_for_a_live_run` pins, as a literal list, the order of `membership_source`,
  `generate_signals`, `now_pt`, `refresh_tape`, `today_et` against the broker calls for
  `plain_buys_and_sells` (the builder measures it on the untouched code and writes the literal, with the count
  of `today_et` and `new_uuid` calls: one per submitted order plus one), and
  `test_late_run_gate_reads_the_clock_after_signal_generation` (`now_pt` is called after `generate_signals`
  returned, in every implementation).
- [ ] **Step 3:** Run: all pass on both implementations.
- [ ] **Step 4: Mutation matrix (not committed).** SM 1-9 in `_run` (the table above), run the file: each SM
  fails the rule test it names (two cases for SM4), SM9 fails the event tests. A mutation that passes means
  the test is rewritten, not the mutation.
- [ ] **Step 5:** `-p no:randomly`, in the neighbourhood, the task gate. Commit:
  `test: name the seven order-flow rules of design 5.1 and pin the hook call order`.

### Task 49: Seeded differential, recorded digests and safety invariants (builder: Terra; reviewer: Opus)

**Files:**
- Create: `tests/paper/equivalence/generator.py`, `tests/paper/equivalence/witnesses.py`,
  `tests/paper/equivalence/compare.py`, `tests/paper/test_run_equivalence.py`,
  `tests/paper/golden/run_equivalence_digests.json`

**Interfaces:**
```python
def build_scenario(seed: int) -> Scenario                    # deterministic from random.Random(seed)
def run_recorded(impl: str, scenario: Scenario) -> Recorded  # transcript, log records, skipped writes
@dataclass(frozen=True)
class Recorded: transcript: dict; logs: tuple[tuple[str, int, str], ...]   # (logger, level, message)
def digest(rec: Recorded) -> str                             # sha256 of to_json(transcript) + the log tuples
FAST_SEEDS = 100; FULL_SEEDS = 600                           # constants, not read from the environment
WITNESSES: dict[str, Callable[[dict], bool]]                 # one predicate per rule precondition
```
The generator samples (bounded ranges, valid harness inputs only): mode (live 60%, read-only dry 25%,
dry with writes 10%, dry-run-with-late-clock 5%); the sweep on or off; the stop on or off with a threshold in
-0.10..-0.40; `MAX_POSITIONS` in {2, 3, 5}; one or two sleeves with weights and a scale; 0-6 held positions
(quantities normal, 0, negative, NaN, `None`; cost bases that breach or do not breach the stop; a sweep
position); buy and sell signals; `last_exit` entries with `get_last_buy_dates` set so some exits are missed;
untradable symbols from the environment and from the broker; legacy pending orders; split and dividend
evidence; account cash and `last_equity` (halts); the persisted peak; per-symbol status sequences (filled,
partial then filled, working, rejected, canceled, expired); a submit failure on order k; a `lock.verify` failure
on order k; scripted notifier failures; a clock failure; a late `now_pt`; a stale `as_of`; the overlay fallback
flag. Read replies not drawn come from `scenarios.PERSIST_DEFAULTS` (imported, not edited).

- [ ] **Step 1: Failing tests** (`tests/paper/test_run_equivalence.py`):
  `test_generator_is_deterministic` (same seed, same scenario, in a child process under two `PYTHONHASHSEED`s),
  `test_every_generated_scenario_runs_without_a_harness_error` (all 600 seeds on the legacy
  implementation),
  `test_every_witness_is_reached_at_least_five_times` (witnesses W1-W14: stopped symbol also buy-signalled;
  stop plus regular sell with slot-limited buys; live sell not fully filled with buys signalled; invalid exit
  quantity with buys and the sweep on; held split correction on a sold symbol; read-only run with a non-empty
  skip list; submit raised after at least one order; rejected exit then another exit; unassessable stop; loss
  halt with exits; sweep sell and sweep buy in one run; live late-run gate; a pending symbol skipped; two sleeves at
  their caps),
  `test_new_run_equals_old_run_for_100_seeds` (`run_recorded("new", s) == run_recorded("legacy", s)`, compared
  as whole `Recorded` values, on a failure print the first differing call index and log record),
  `test_new_run_matches_the_recorded_digests` (`@pytest.mark.slow`, `@pytest.mark.timeout_seconds(600)`, 600
  seeds against the digest file, the new run only),
  `test_digest_file_pins_the_generator` (the file stores `sha256(generator.py)`; a changed generator fails),
  `test_safety_invariants_over_all_seeds` (on the new run of all 600 seeds): (a) a dry run has no
  `submit_*` and no `create_order_intent`, `attach_broker_order` or `settle_order_intent`; (b) every
  `submit_*` is preceded in order by `create_order_intent` and `lock.verify` for the same client id and
  followed by `attach_broker_order` unless it raised; (c) after a raised `submit_*` there is no later
  `create_order_intent` or `submit_*`; (d) once a `HALTED buys` notice is sent no `submit_buy` follows;
  (e) in a single-sleeve seed no symbol except the sweep symbol is submitted twice; (f) a read-only run has no
  `persist.*` call that is a member of `WRITE_METHODS`; (g) every `submit_buy` notional is positive and
  finite, every `submit_sell` quantity is positive and finite.
  Run them: all fail (no generator).
- [ ] **Step 2:** Implement the generator, witnesses, compare module; run the tests on the untouched code
  (`_run` equals `_run_legacy`, so the equality test passes trivially; its power is shown in Step 4).
- [ ] **Step 3: Record the digests once, from the untouched code** with a one-off `python -c` run, through
  `./offline.sh env GGT_RUN_MEM_MAX=3G scripts/run_capped.sh .venv/bin/python -c ...` (capped, scrubbed, one heavy
  job), that calls `digest(run_recorded("new", build_scenario(s)))` for seeds 0-599 and writes the file with
  `generator_sha256`; **no recording switch is committed** (a test that can write its own reference is how a
  golden gets regenerated by accident). Run twice under two hash seeds: identical files.
- [ ] **Step 4: Mutation matrix (not committed).** Apply SM 1-8, 10, 11, 13 and 14 to `_run` only (never to
  `_run_legacy`): `test_new_run_equals_old_run_for_100_seeds` and the digest test both fail for each; a
  mutation that survives 100 seeds but not 600 is noted (the fast test's seed count is raised in the same
  task until none does, within the 20-second budget); revert.
- [ ] **Step 5:** `T tests/paper/test_run_equivalence.py` within budget (100-seed tests under 60 s, the slow test
  under 300 s); the task gate. Commit: `test: seeded equivalence of the trader run, digests and safety
  invariants`.

### Task 50: Seam, import and structure contracts (builder: Sonnet; reviewer: Sol)

**Files:** Create `tests/paper/test_seam_contract.py`, `tests/paper/test_run_structure.py`.

- [ ] **Step 1: Failing or vacuous-today tests** (those marked * pass on today's code, and must keep passing; the rest fail until their task):
  - `test_every_patch_target_still_resolves_in_the_trader_namespace`*: scans `tests/` and
    `tests/paper/transcript/harness.py` for `"ggTrader.paper.trader.X"` strings and for the names the harness
    patches, asserts each `X` (first path segment) is an attribute of `ggTrader.paper.trader`.
  - `test_legacy_seams_intercept_in_the_new_run`* (each case drives a minimal scenario): an instance
    assignment of `_submit_order`, `_reconcile_pending_orders`, `_compute_split_corrections`; a class patch of
    `_poll_orders`; replacement of `_risk` and of `_risk.cfg`; `_dry_run` assigned after construction selects
    the dry path (and no `submit_*` is called); `_paper_lock` assigned a `MagicMock` is the one verified; a
    string patch of `find_split_applied_symbols`, `generate_core_signals`, `_today_et`, `_now_pt`, `uuid4`,
    `refresh_benchmark_tape` and of one persist name after construction each win.
  - `test_moved_code_logs_on_the_trader_logger` (fails until Task 52): `constants.TRADER_LOGGER_NAME ==
    "ggTrader.paper.trader"`, and every `logging.getLogger` call in a file of `NEW_MODULES` uses that constant.
  - `test_new_modules_import_only_what_they_may` (the list `NEW_MODULES` is empty today and each task appends its
    module): forbidden imports of Global Constraints; no `import time`, no `import uuid`, no `.now(`, `.today(`,
    `os.environ`, `getenv`, no call to the environment accessors of "per-run settings snapshot"; no
    `from ggTrader.paper.cash_sweep import` or `from ggTrader.paper.catastrophe_stop import` (module attribute
    access only).
  - `test_env_changed_after_the_first_broker_call_is_ignored_by_every_stage`*: a broker whose first
    `get_account` mutates `SWEEP_SYMBOL`, `UNTRADABLE_SYMBOLS`, `CATASTROPHE_STOP_PCT`, `SWEEP_MIN_CLIP_USD`,
    `SWEEP_BUY_TRIGGER_PCT`, `SWEEP_CASH_RESERVE_PCT`, `CASH_SWEEP_ENABLED`; the transcript equals the run
    without the mutation (sweep and stop scenarios).
  - `test_run_stays_an_orchestrator`: `RUN_MAX_LINES = 830` today; `run()` at most 25 lines; both measured
    with `ast` on `PaperTrader.run` and `PaperTrader._run`. A comment states the ratchet protocol.
  - `test_trader_persist_names_unchanged`*: the 26 names of `ledger.TRADER_NAMES` are all bound in the trader
    module.
- [ ] **Step 2:** Run; confirm the starred tests pass and the others fail for the stated reason.
- [ ] **Step 3: Mutation check (not committed).** Delete the `find_split_applied_symbols` import: patch-target
  test fails. Make `_run` read `os.environ` once: the env test and the existing `test_trader_py_reads_no_environment`
  fail. Revert.
- [ ] **Step 4:** The task gate (the one not-yet-satisfied test, the logger constant, is `xfail(strict=True)`
  with a reason naming Task 52, so the gate is green; Task 52 removes the marker in its own commit).
  Commit: `test: contracts for the legacy seams, new-module imports and run size`.

### Task 51: Phase A gate and hash record of the new pins

**Files:** none committed; updates `$E`.

- [ ] **Step 1: Cross-process determinism.** The pin files of Tasks 47-50 under `PYTHONHASHSEED` unset, 1 and 2, all pass.
- [ ] **Step 2: Fast core collects them.** `T --fast --collect-only -q` lists every node of the new files except the
  `slow` digest test, which `T --full --collect-only -q` lists (two files of node ids, diffed, as in Phase 4 Task 33).
- [ ] **Step 3: The whole mutation matrix** (SM 1-14) on the untouched-but-for-mutation `_run`: each fails exactly the nets of the table. Record
  the result in `$E/p5/mutation-matrix-A.md`.
- [ ] **Step 4: Phase gate** (fast core, `--full`, `--studies`, O, P): all pass; the oracle at the 6G cap; no `src/` change
  beyond `_run_legacy`.
- [ ] **Step 5: Opus read** (fresh context; given only `test_run_rules.py`, `rules_support.py`, the generator
  and witnesses, and the claim "each test fails for the wrong behavior and passes for today's"): reports any test
  that would pass for a wrong reason; fixes are made before the hash record.
- [ ] **Step 6: Record** `$E/p5-added.sha256` (every new test file, `rules_support.py`, the equivalence package,
  the digest file, `run_legacy.sha256`); from here on they are immutable (Global Constraints).
- [ ] **Step 7:** Report usage (Phase A done).

### Task 52: Seams, records and the logger constant (builder: Terra; reviewer: Opus)

**Files:**
- Create: `src/ggTrader/paper/seams.py`, `tests/paper/test_seams.py`
- Modify: `src/ggTrader/paper/models.py` (the records of "Interfaces"), `src/ggTrader/paper/constants.py`
  (`TRADER_LOGGER_NAME = "ggTrader.paper.trader"`), `src/ggTrader/paper/trader.py` (add `_hooks()` and
  `_seams()` building the two records with late-binding lambdas, call them nowhere yet), `src/ggTrader/paper/ledger.py`
  (docstring line "Nothing imports this module yet..." corrected; nothing else), `tests/paper/test_models.py`

- [ ] **Step 1: Failing tests** (`test_seams.py`, `test_models.py`): `test_hooks_resolve_the_trader_namespace_at_call_time` (patch
  `ggTrader.paper.trader.generate_core_signals` after `_hooks()` was built; calling `hooks.generate_signals` reaches the patch),
  the same for the other nine fields, `test_seams_resolve_instance_attributes_at_call_time` (assign
  `trader._submit_order` after `_seams()`; the seam calls it), `test_default_hooks_read_nothing_at_construction`
  (building them calls no clock, uuid or signal function), `test_order_intent_is_frozen_and_slotted`,
  `test_skip_and_invalid_exit_are_frozen`, `test_execution_book_lists_are_not_shared_between_books`,
  `test_risk_verdict_with_halt_returns_a_new_verdict_and_keeps_order`, `test_run_context_is_frozen`,
  `test_strategy_signals_fields_are_immutable_views`. FAIL: modules and names are missing.
- [ ] **Step 2:** Implement. `RunHooks.utcnow` is `lambda: datetime.now(timezone.utc)` reading the trader's
  `datetime` name; `clock_time` and `sleep` read the trader's `time`. No behavior of `_run` changes; `_run_legacy` and `_run` still use their old code.
- [ ] **Step 3:** New tests pass; `test_trader_py_logger_name...` of Task 50 lists no module yet and passes.
- [ ] **Step 4:** The task gate (T byte-identical, F unchanged). Commit: `refactor(paper): late-binding hooks and seams, run records, logger constant`.

### Task 53: Pure planning I: signals, stops, exit validation, exits (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/planning.py` (this task's functions), `tests/paper/test_planning.py`;
append the module to `NEW_MODULES` (Task 50).

The bodies are the code now at: signal derivation `trader.py:816-830`; the unassessable scan 1071-1077 and
the stop loop 1095-1138 (the `find_catastrophe_stops` call stays inside `plan_stops`, via the module attribute); the exit validation
loop 1147-1160; the regular-sell loop 1249-1268. Log texts become `Skip` messages character for character,
`Skip.level` is `logging.INFO` or `logging.WARNING` as in the source line.

- [ ] **Step 1: Failing tests** (plain data, no mocks, no notifier, no broker):
  - `test_strategy_signals_merges_sleeves_in_order_and_sorts_sells`, `test_strategy_signals_takes_as_of_from_the_first_sleeve`,
    `test_empty_sleeves_ends_the_run_with_stop_iteration_after_the_fallback_notice` (the function raises
    `StopIteration`, as `next(iter({}))` does today; Review Focus 4).
  - `test_assess_stops_flags_none_and_invalid_positions` (a position whose `unrealized_pct` is `None` or raises `ValueError`).
  - `test_plan_stops_yields_one_intent_per_breaching_symbol_with_qty_and_reason` (kind `STOP`, `size` is the
    strategy quantity, `book_amount` is the strategy market value, `reason` is `catastrophe_stop.CATASTROPHE_STOP_REASON`,
    `unrealized_pct` and `stop_threshold` carried),
    `test_plan_stops_yields_invalid_exit_for_a_bad_quantity_and_goes_on` (rule 4),
    `test_plan_stops_skips_a_pending_symbol_with_the_old_log_text` (the `Skip` message equals
    `"Skipping catastrophe stop X: order already pending"` and its level is INFO),
    `test_plan_stops_calls_the_catastrophe_module_attribute` (patching `catastrophe_stop.find_catastrophe_stops` changes the result).
  - `test_validate_exits_yields_invalid_exit_once_per_symbol_and_ignores_pending` (a pending order does not excuse an
    invalid quantity), `test_validate_exits_ignores_unheld_and_already_invalid_symbols`.
  - `test_plan_exits_skips_pending_symbols_with_the_old_log_text`, `test_plan_exits_intent_size_is_the_raw_quantity_book_amount_the_market_value`
    (rule 5), `test_plan_exits_ignores_invalid_symbols`.
  - `test_planners_are_lazy_and_do_not_mutate_their_inputs` (inputs are `MappingProxyType`s and frozen sets; consuming one decision does not
    evaluate the next: a `Mapping` subclass counts lookups).
  FAIL: ImportError.
- [ ] **Step 2:** Implement; the planners are generators; no logging, no notifier, no broker, no ledger.
- [ ] **Step 3:** New tests pass. `_run` is **not** wired yet; the task gate (T byte-identical). Commit:
  `refactor(paper): pure planning for signals, stops and exits`.

### Task 54: Pure planning II: sweep sell, buys, sweep buy (builder: Terra; reviewer: Opus)

**Files:** Modify `src/ggTrader/paper/planning.py`; Modify `tests/paper/test_planning.py`.

Bodies at `trader.py:1171-1181` (limits, which stay in `OrderFlow` and are passed in), 1183-1236 (sweep sell:
the keyword arguments of `compute_sweep_sell_for_funding(...)` keep today's order, so `account["cash"]` is
read before `cfg.reserve_pct()` raises), 1302-1351 (buys), 1367-1396 (sweep-buy sizing).

- [ ] **Step 1: Failing tests:**
  - `test_plan_sweep_sell_returns_none_when_cash_covers_the_buys`, `test_plan_sweep_sell_sizes_qty_from_price_and_rounds_to_four_places`,
    `test_plan_sweep_sell_returns_none_for_a_zero_price_or_zero_quantity`,
    `test_plan_sweep_sell_counts_only_sells_not_pending_for_the_slot_estimate`,
    `test_plan_sweep_sell_reraises_the_stored_reserve_error_after_reading_the_account_cash` (the error is
    `ValueError`; a missing `cash` key raises `KeyError` first).
  - `test_plan_buys_follows_held_stopped_pending_slots_sleeve_cap_concentration_in_that_order` (one candidate hitting each
    check in turn; the `Skip` messages equal the old log lines, including `Max positions reached (N), skipping X`),
    `test_plan_buys_breaks_at_the_slot_limit_per_sleeve_and_goes_on_to_the_next_sleeve`,
    `test_plan_buys_counts_each_yielded_intent_toward_slots_and_sleeve_caps`,
    `test_plan_buys_never_rebuys_a_stopped_symbol` (rule 1),
    `test_plan_buys_slot_arithmetic_is_the_callers_slots_available` (rule 2: the planner takes `slots_available`, the orchestrator computes it from regular sells),
    `test_plan_buys_evaluates_each_symbol_only_after_the_previous_intent_was_consumed` (a concentration check that raises on the
    second symbol is not called after one `next()`; Review Focus 1),
    `test_a_symbol_bought_by_two_sleeves_is_planned_twice_as_today`, `test_a_buy_signal_for_the_sweep_symbol_is_an_ordinary_candidate`,
    `test_plan_buys_raises_key_error_for_a_sleeve_missing_from_the_open_counts_at_the_old_point` (`open_by_sleeve[universe]` is read after
    the held, stopped, pending and slot checks, as `sleeve_open_count[universe]` is today).
  - `test_plan_sweep_buy_matches_compute_sweep_buy_for_valid_settings`, `test_plan_sweep_buy_reraises_each_stored_parse_error_in_its_old_order`
    (`reserve_pct`, then `min_clip_usd`, then `buy_trigger_pct`), `test_plan_sweep_buy_returns_none_when_the_action_is_not_a_buy`.
- [ ] **Step 2:** Implement; **Step 3:** pass; `_run` still unwired. The task gate. Commit:
  `refactor(paper): pure planning for the sweep sell, buys and the sweep buy`.

### Task 55: Move the order bodies into `OrderExecutor` / `DryRunExecutor` (builder: Terra; reviewer: Opus)

**Files:**
- Create: `src/ggTrader/paper/execution.py`, `tests/paper/test_execution.py`
- Modify: `src/ggTrader/paper/trader.py`: the five methods become forwards (S1); `_run` and `_run_legacy` are untouched; append the module to `NEW_MODULES`

The bodies are `_submit_order` (193), `_reconcile_order_intents` (224), `_flag_if_stale` (333),
`_reconcile_pending_orders` (361), `_poll_orders` (428), moved unchanged except: `self._ledger` is the
injected ledger, `self._notifier` and `self._broker` likewise, `uuid4()` is `hooks.new_uuid()`, `_today_et()` is
`hooks.today_et()`, `time.time()`/`time.sleep()` are `hooks.clock_time()`/`hooks.sleep()`, `datetime.now(timezone.utc)` is
`hooks.utcnow()`, the lock is `self._lock()` read at the moment of the call, and the `self._dry_run` branch of
`_reconcile_pending_orders` becomes the overridable `_pending_load_failed(exc) -> set[str]` (live: send the
notice and raise; dry: warn and return `set()`). `PaperTrader._executor_for_call()` builds
`DryRunExecutor` when `self._dry_run`, else `OrderExecutor`, from the current attributes (S4, S5).

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
  `test_dry_run_executor_never_touches_a_submitting_broker` (a broker whose `submit_buy`/`submit_sell` raise
  `AssertionError`, a full dry scenario runs to the end),
  `test_executor_uses_the_hooks_not_the_modules` (patching `ggTrader.paper.trader.uuid4` after construction changes the client id).
  FAIL: ImportError.
- [ ] **Step 2:** Implement and cut the five bodies from `trader.py` (their forwards: `return self._executor_for_call().<method>(...)`
  with the same signature and docstring first line); `trader.py` keeps its imports that tests patch (`# noqa: F401 - patch target`).
- [ ] **Step 3:** New tests pass; the 103 transcripts, `test_phase4_pins.py` (it calls `_poll_orders` directly
  and spies on the class) and the whole `tests/paper` neighbourhood stay green with no edit.
- [ ] **Step 4: Mutations.** SM 10 and 14 at their new sites: fail T; a mutated moved `submit_order` stage order fails
  `test_rule7a...` and T. Revert.
- [ ] **Step 5:** The task gate; Opus reviews the moved blocks with `--color-moved`. Commit:
  `refactor(paper): OrderExecutor and DryRunExecutor hold the order bodies`.

### Task 56: Move split, dividend and missed-exit logic into `CorporateActions` (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/corporate_actions.py`, `tests/paper/test_corporate_actions.py`; Modify
`src/ggTrader/paper/trader.py` (three forwards); append to `NEW_MODULES`.

Bodies: `_compute_split_corrections` (483-605), `_accrue_dividends` (607-703), `_missed_exits` (305-331), moved
unchanged except the ledger and broker are injected and `find_split_applied_symbols` is `hooks.find_split_applied`.

- [ ] **Step 1: Failing tests:** `test_split_corrections_fail_soft_at_every_ledger_read` (five cases, each logs the old warning and degrades as before),
  `test_persisted_state_for_an_unheld_symbol_is_cleared`, `test_applied_split_clears_the_persisted_row`,
  `test_dividend_accrual_returns_the_partial_result_on_each_failure_point` (`total`, `new_accruals`, `skipped`),
  `test_missed_exits_orders_candidates_and_survives_a_lookup_failure`,
  `test_split_check_reaches_the_patched_trader_name` (a string patch of `ggTrader.paper.trader.find_split_applied_symbols` wins).
- [ ] **Step 2-3:** Implement, cut, forwards; **Step 4:** SM mutation: drop a `clear_split_correction` call; T and the unit test fail; revert.
- [ ] **Step 5:** The task gate. Commit: `refactor(paper): CorporateActions and missed_exits hold the split and dividend bodies`.

### Task 57: Wire `PreflightGate` (region R2) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/preflight.py`, `tests/paper/test_preflight.py`; Modify `trader.py` (`_run`
lines 744-855 replaced by `PreflightGate(...).check()` inside `try/except RunHalted`, the old locals rebound for the
inline code that remains); lower `RUN_MAX_LINES`; append to `NEW_MODULES`.

The gate keeps this order exactly: clock gate (fail open on a clock error, `Market closed` notice and
`RunHalted(RunResult([], [], []))`), the signals `try` (`SignalGateError` notice and re-raise, any other
exception notice and re-raise), the per-sleeve info log and membership warning, the late-run gate
(`hooks.now_pt()`; live: notice and `RunHalted(gate="late_run")`; dry: warning, DRY RUN notice, continue), the
read-only skip (`skipped_writes.append`, the info log) or `hooks.refresh_tape()` with its warning, the overlay
fallback notice, the signal derivation, the freshness loop (`hooks.today_et()` read once; stale notice and
`RunHalted(RunResult([], [], []))`).

- [ ] **Step 1: Failing tests:** `test_ratchet` (lower the bound, see it fail), `test_market_closed_halts_with_the_old_notice_and_result`,
  `test_clock_error_fails_open`, `test_signal_gate_error_notifies_and_reraises`, `test_signal_error_notifies_and_reraises`,
  `test_late_run_live_halts_with_gate_late_run`, `test_late_run_dry_warns_and_continues` (quirk 1),
  `test_read_only_appends_to_the_shared_skipped_list_and_does_not_refresh`, `test_live_refreshes_and_warns_when_nothing_refreshed`,
  `test_fallback_notice_precedes_the_freshness_check`, `test_stale_unparseable_and_future_as_of_halt_with_the_old_texts` (three cases),
  `test_the_clock_is_read_after_signals_and_today_after_the_late_gate`, `test_empty_sleeves...` (via the gate, `StopIteration`).
- [ ] **Step 2-3:** Implement; cut; the SM table row for SM9 is re-located here (mutation: read `hooks.now_pt()` first).
- [ ] **Step 4:** The task gate (E under both hash seeds). Commit: `refactor(paper): PreflightGate takes the market, signal, late-run and freshness gates`.

### Task 58: Wire `PortfolioView.load` (region R3) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/portfolio.py`, `tests/paper/test_portfolio.py`; Modify `trader.py` (`_run`
857-953 replaced; `RunContext` is built here, after the gate and the pending reconcile); lower the ratchet; append to `NEW_MODULES`.

`load` performs, in this order: `get_account` (failure: `_account_read_halt` notice and re-raise), `get_positions`
(failure: the two-line log, notice inside its own `try`, re-raise), the strategy-position set (the sweep symbol removed when the sweep is on),
`resolve_untradable` with `cfg.untradable`, the frozen-holdings notice, the untradable-candidate notice, the buy filter, split corrections
(through `seams.split_corrections`), `apply_corrections_to_positions`, missed exits (through `seams.missed_exits`, merged into the sells,
logged), dividends (through `seams.accrue_dividends`).

- [ ] **Step 1: Failing tests:** `test_load_reads_in_todays_order` (the recorded call sequence), `test_account_failure_halts_with_the_old_notice`,
  `test_position_failure_halts_even_if_the_notice_fails`, `test_sweep_position_is_invisible_to_strategy_positions`,
  `test_untradable_holdings_are_frozen_and_their_buy_signal_dropped`, `test_blocked_buy_candidates_are_announced`,
  `test_orders_use_uncorrected_positions_and_reports_use_corrected` (rule 5), `test_missed_exits_are_merged_into_the_sells_sorted`,
  `test_the_view_is_frozen_and_shares_no_mutable_state_with_the_broker_reply`, `test_load_uses_the_per_run_config_not_the_environment`.
- [ ] **Step 2-4:** Implement, cut, ratchet; SM12 mutation (read the environment in `load`): the env test fails. The task gate. Commit:
  `refactor(paper): PortfolioView.load reads the account, positions, splits and dividends`.

### Task 59: Wire `RiskAssessor` (region R4) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/risk_assessor.py`, `tests/paper/test_risk_assessor.py`; Modify `trader.py`
(`_run` 955-1034, the `hard_stop` branch returns `RunResult([], [], [reason]).as_dict()`); ratchet; `NEW_MODULES`.

`assess` keeps this order: persisted peak (read failure: warning, notice, hard stop; `update_peak` `ValueError`: notice,
hard stop), `check_drawdown_halt` (invalid value: notice, hard stop; else a buy halt), `update_peak` and the peak write when no halt
(write failure: warning, notice, hard stop), `last_equity` checks (missing, invalid: notice, hard stop), `check_daily_loss` with the
split note, then the halted-buys notice and the pending-orders notice, each inside its own `try`. The hard stops send their notice
outside any `try` (quirk 7). `RiskVerdict.with_halt` is what `OrderFlow` uses for an unassessable stop.

- [ ] **Step 1: Failing tests:** one per exit of the list above (nine), `test_split_note_names_symbols_sorted_even_with_non_string_keys`,
  `test_peak_is_saved_only_when_no_loss_halt`, `test_loss_halt_notices_fail_soft_and_the_pending_notice_still_goes`,
  `test_hard_stop_notice_failure_raises_through` (quirk 7), `test_assess_uses_the_guard_assigned_after_construction`.
- [ ] **Step 2-4:** Implement, cut, ratchet; mutations SM 4-adjacent: drop the `all_buys = []` equivalent (`buy_halts` ignored): rule tests fail. Gate 1-6. Commit:
  `refactor(paper): RiskAssessor seeds the peak and applies the drawdown and daily-loss halts`.

### Task 60: Wire `OrderFlow` stage I: stops, exit validation, buy limits (region R6a) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/order_flow.py`, `tests/paper/test_order_flow.py`; Modify `execution.py` (intent-level API:
`recover_intents`, `submit`, `announce` for stops, `DryRunExecutor` overrides), `trader.py` (`_run` 1048-1181 replaced by
`OrderFlow(...).run(...)`, which for now returns after this stage's code while the rest of `_run` still runs inline on its bridge locals); ratchet; `NEW_MODULES`.

The stage: `assess_stops`; with unassessable symbols, `verdict.with_halt(reason)`, `all_buys = []`, the alert and the pending notice
(each in a `try`); `plan_stops` consumed decision by decision (`Skip` is logged; `InvalidExit` appends the error and the symbol;
an `OrderIntent` is submitted, then `stopped`, the sell list, the pending list, then `announce`); the stopped symbols are removed from the open positions;
`validate_exits`; if any symbol is invalid the buy candidates are emptied; then the limits (`sleeve_slot_caps`,
`sleeve_open_count`, `buys_by_sleeve`, `sleeve_position_notional`, in this order, each evaluated here as today).

- [ ] **Step 1: Failing tests:** `test_stop_stage_submits_before_it_plans_the_next_symbol` (the order of fake calls interleaves),
  `test_a_stop_sell_is_booked_then_announced_in_that_order` (live and dry), `test_unassessable_blocks_buys_and_adds_a_halt_without_blocking_assessable_stops`,
  `test_invalid_stop_quantity_blocks_buys_not_other_stops` (rule 4), `test_stopped_symbols_leave_the_open_positions_at_once` (rule 1),
  `test_exit_validation_runs_after_the_stops_and_before_the_limits`, `test_limits_are_computed_in_the_old_order_and_raise_where_they_raised`.
- [ ] **Step 2-4:** Implement; ratchet; SM 1, 4, 11: the rule tests and E fail. The task gate. Commit:
  `refactor(paper): OrderFlow runs the catastrophe-stop stage and the exit validation`.

### Task 61: Wire `OrderFlow` stage II: sweep sell, sells, settlement of the sells (region R6b) (builder: Terra; reviewer: Opus)

**Files:** Modify `order_flow.py`, `execution.py` (`after_sells`, `unfilled_exit_errors`, the sweep-sell and sell announcements),
`trader.py` (`_run` 1183-1301 gone), ratchet, tests.

The stage: when the sweep is on and there is no halt, `plan_sweep_sell` (the anticipated-sell and slot estimate use the open positions after the stops;
`cfg.reserve_pct()` raises here, outside any `try`) and its execution; `plan_exits` consumed lazily (`regular_sell_count` counts regular sells only); `executor.poll`
of the sells; `after_sells(filled)`; `unfilled_exit_errors` (live only): each error is appended and the buys cleared (rule 3).

- [ ] **Step 1: Failing tests:** `test_sweep_sell_is_planned_after_the_stops_were_executed` (SM 11 target), `test_a_bad_reserve_raises_after_the_stop_orders_went_out`
  (quirk 2), `test_regular_sell_count_excludes_stop_and_sweep_sells` (rule 2), `test_only_sells_are_polled_before_the_buys`,
  `test_an_exit_that_is_not_fully_filled_blocks_all_buys_and_names_the_status` (rule 3, three statuses), `test_dry_run_polls_an_empty_list_and_reconciles_nothing`.
- [ ] **Step 2-4:** Implement; ratchet; SM 2, 3, 7, 8, 10, 11. The task gate. Commit:
  `refactor(paper): OrderFlow runs the sweep sell, the sells and their settlement`.

### Task 62: Wire `OrderFlow` stage III: buys, settlement, sweep buy (region R6c) (builder: Terra; reviewer: Opus)

**Files:** Modify `order_flow.py`, `execution.py`, `trader.py` (`_run` 1302-1397 gone), ratchet, tests.

The stage: `slots_available = risk.max_new_positions(len(open_positions) - regular_sell_count)` computed here; `plan_buys` consumed lazily; the
poll of the remaining orders; `after_orders`; the sweep buy only when the sweep is on, there is no halt and no invalid exit symbol (rule 4), with the broker
account read, `plan_sweep_buy` and the old fail-soft `try` (`Cash sweep buy sizing failed (non-fatal): %s`); live: submit, poll, append, reconcile unconditionally.

- [ ] **Step 1: Failing tests:** `test_buy_count_and_sleeve_open_count_follow_yielded_intents`, `test_a_halted_run_submits_no_buys_and_no_sweep_buy`,
  `test_invalid_exit_blocks_the_sweep_buy` (rule 4), `test_sweep_buy_sizing_failure_logs_and_buys_nothing` (each stored parse error, and a broker error),
  `test_live_sweep_buy_is_polled_appended_and_reconciled_in_that_order`, `test_dry_sweep_buy_only_announces`,
  `test_a_submit_exception_ends_the_run_after_the_intent_is_saved` (rule 7a; no later order), `test_rejected_order_reaches_normal_settlement` (rule 7b).
- [ ] **Step 2-4:** Implement; ratchet; SM 1-8. The task gate. Commit:
  `refactor(paper): OrderFlow runs the buys, their settlement and the sweep buy`.

### Task 63: Wire `RunReporter` and finish `_run` (regions R5, R7) (builder: Terra; reviewer: Opus)

**Files:** Create `src/ggTrader/paper/reporter.py`, `tests/paper/test_reporter.py`; Modify `trader.py` (`_run` becomes the
orchestrator described under "Interfaces"; the bridge locals are deleted); ratchet to the measured size (≤ 45 lines); `NEW_MODULES`.

`previous_value` is the prior-snapshot read with its two guards (R5). `finish` keeps this order: one `trade_alert` per submitted order (filled amount when filled),
the "N order(s) failed" message, the post-trade account and positions read (failure: log, notice inside a `try` whose text depends on dry or live, re-raise), the
corrected value (with the invalid-value error appended to `book.errors`), daily and unrealized P&L, the starting-capital read (failure appends its error), total P&L, the risk
line, `daily_summary(... errors=buy_halts + errors ...)`, the risk message, the snapshot write (failure appends its error and sends its notice inside a `try`), and the `RunResult`
(`buy_halts + errors` evaluated after the snapshot step; `untradable=sorted(...)`; `skipped_writes=list(skipped_writes)`).

- [ ] **Step 1: Failing tests:** `test_one_trade_alert_per_order_with_filled_or_intended_amount`, `test_post_trade_read_failure_text_depends_on_dry_or_live` (two cases),
  `test_invalid_post_trade_value_is_reported_and_no_snapshot_is_written`, `test_pnl_is_unavailable_if_any_unrealized_value_is_invalid`,
  `test_starting_snapshot_failure_keeps_the_summary_and_attempts_the_snapshot`, `test_summary_errors_exclude_the_snapshot_failure_but_the_result_includes_it`,
  `test_result_lists_are_the_books_own_lists` (risk 10), `test_final_run_is_an_orchestrator` (the ratchet, ≤ 45 lines), `test_run_py_unchanged_in_shape`.
- [ ] **Step 2-4:** Implement; the bridge locals go; SM 14 and a mutation that appends the snapshot error before the summary. The task gate. Commit:
  `refactor(paper): RunReporter reports the run; _run is an orchestrator`.

### Task 64: Wiring gate (phase D)

- [ ] **Step 1:** Mutation matrix SM 1-14 on the final tree: each fails the nets of its row, recorded in `$E/p5/mutation-matrix-D.md`; a mutation no net catches is a finding and is fixed (a test is added, in an amendment note approved by Opus).
- [ ] **Step 2:** The phase gate (fast core, `--full`, `--studies`, oracle at the 6G cap, outcome check, `verify_pins.py`).
- [ ] **Step 3:** `git diff --stat BASE5 -- <immutable paths>` empty; the added pins match `p5-added.sha256` (or the amendment notes).
- [ ] **Step 4: Reviews.** Opus reviews Tasks 52-63 as one body (fresh context, given the diff, the rules, and the claim "behavior is unchanged"); Sol reviews the pins
  and generator (Claude-written, Task 47, 48, 50) and the decision to keep the seams. Findings are fixed before Task 65.
- [ ] **Step 5:** Report usage (Phase D done).

### Task 65: The test rig and one pilot conversion (builder: Terra; reviewer: Opus)

**Files:** Create `tests/paper/support.py`; Modify `tests/paper/test_trader.py` (the pilot only: `TestSellExits`, lines 274-327).

`support.py` holds: `class SpyLedger(InMemoryLedger)` recording every public call as `(name, args, kwargs)` in `calls(name=None)`;
`make_trader(...)` (the current `_make_trader` with `ledger=SpyLedger()` injected and the private `_submit_order` seam kept as today);
`seed_snapshots`, `seed_peak`, `seed_pending_orders` helpers; and `PERSIST_NAMES` (the 26). The conversion protocol (the same for Tasks 66-73):

1. Keep every node id and every parametrization id.
2. Replace each `patch("ggTrader.paper.trader.<persist name>")` by seeding or spying through the rig; assertions on a persist mock become assertions on `ledger.calls(...)` or ledger state, one for one.
3. Leave non-ledger patches and the private seams alone.
4. The autouse persist stubs of the file stay until Task 73; a converted test injects its own ledger and is unaffected by them.
5. No new `skip`, `xfail` or weakened assertion; the number of assertions (`assert`, `.assert_*`, `pytest.raises`) per node is not lower than at `BASE5`.

- [ ] **Step 1: The assertion-count guard.** It is a one-off outside the repository (`$E/p5/assertion_counts.py`, written by Haiku
  and checked by Sonnet, not committed): it prints, per test node, the number of `assert` statements, `.assert_*` calls and `pytest.raises`
  at a git ref. Record the `BASE5` counts for the whole file in `$E/p5/assertions-base5.json` before any conversion.
- [ ] **Step 2:** Convert `TestSellExits` (3 tests, 4 persist patches); **Step 3:** `T tests/paper/test_trader.py -k TestSellExits` passes.
- [ ] **Step 4: Mutation.** SM 5 and SM 6 fail the converted tests (they must still have teeth).
- [ ] **Step 5:** Assertion counts not lower; outcome check on the file; the task gate. Commit: `test: in-memory ledger rig and the first converted trader test class`.

### Tasks 66-73: Convert the rest of `test_trader.py`, batch by batch (builder: Terra for each; reviewer: Opus for each)

Each task follows Task 65's protocol exactly: **Step 1** the `BASE5` assertion counts for the batch; **Step 2** convert; **Step 3**
`T tests/paper/test_trader.py -k "<the batch's names>"` passes and then `T tests/paper/test_trader.py` passes in full (never
`tests/paper/` as a directory, which the orchestrator runs); **Step 4** the batch's mutations fail it; **Step 5** assertion counts not lower, no new skip, the task gate;
**Step 6** Opus reads the diff of the batch in a fresh context with the claim "each converted test still proves what it proved, and would fail on the listed mutations"; **Step 7** commit
`test: convert trader tests <batch> to the in-memory ledger`.

| Task | Batch | Names (line range at `73c35417`) | Tests | Persist patches | Mutations that must fail it |
|---|---|---|---|---|---|
| 66 | M1 | TestMissedExitCatchUp, TestUntradableHoldings, TestMaxPositionsEnv, TestBenchmarkTapeKeepalive, TestBuyEntries, TestNotifications, TestErrorHandling, TestFillLogging (335-722) | 26 | 28 | SM 2, 5, 14 |
| 67 | M2 | TestRunDateIsSessionDateNotAsOf, TestReconciliation, TestDailyPnl (725-1218) | 16 | 37 | SM 8, 14, a mutation that stamps `as_of` instead of the ET date |
| 68 | M3 | TestDividendAccrual, TestDryRun, TestRunPaperTrading, TestPendingOrderDedup, TestMarketGates, TestPersistedDrawdown (1221-1920) | 38 | 54 | SM 6, 9, a mutation that skips the peak save |
| 69 | M4 | the module functions from `test_loss_limit_allows_protective_exit_but_blocks_new_buys` to `test_invalid_risk_state_still_blocks_all_orders` (1923-2285) | 19 | 21 | SM 4, 7 |
| 70 | M5 | TestStalePendingOrders, TestOrderErrorAlerting, TestCashSweepFlagOff, TestCashSweepEnabled, TestCatastropheStopFlagOff, TestCatastropheStopEnabled (2288-2835) | 23 | 30 | SM 1, 4, 11, 12 |
| 71 | M6 | TestFailClosed, TestUntradableAvb, TestReadOnlyDryRun, TestDryRunNotifier, TestRunPaperTradingReadOnly (2854-3263) | 29 | 2 | SM 6, 13 |
| 72 | M7 | TestLiveMembershipGate and the functions from `test_pending_order_read_failure_stops_live_run` to `test_loss_halt_stops_after_ambiguous_exit_submission` (3269-3749) | 21 | 17 | SM 7, 8 |
| 73 | M8 | the functions from `test_loss_halt_reconciles_rejected_exit_and_submits_other_exit` to the end, `TestUnappliedSplitAndDailyLoss` included (3752-4224); then delete the seven persist autouse fixtures (`_stub_intent_db`, `_stub_write_path_db`, `_stub_pending_order_db`, `_stub_risk_state_db`, `_stub_dividend_db`, `_stub_split_trade_db`, `_stub_last_buy_dates`) and `_intent_store` once nothing uses them. A fixture stays, with a comment naming the test, if a test that cannot take an injected ledger still needs it (a test that goes through `run_paper_trading`, which builds its own trader) | 22 | 18 | SM 3, 4, 7, 8 |

The per-class test and patch counts were measured on 2026-11-19 with `ast` and a regex over `test_trader.py` at `73c35417`; with Task 65's `TestSellExits` (3 tests, 4 patches)
they sum to 197 tests and 211 patches, and the other 28 persist patches sit in the autouse fixtures and `_intent_store` (Task 73). Task 46 re-measures the table. After Task 73:
no string patch of a persist name remains in `test_trader.py`, the other 200 dotted patches remain, and the file's outcome check against `p5base` shows all 197 node ids passed.

### Task 74: Delete the frozen copy; the digests keep guarding (builder: Terra; reviewer: Opus)

**Files:** Modify `src/ggTrader/paper/trader.py` (delete `_run_legacy`, then every import and name only it used and no test or harness patches, from the Task 50 list; keep each name that is a patch target with its `# noqa: F401 - patch target`),
`tests/paper/equivalence/impl.py` and `tests/paper/test_run_equivalence.py` (the old-versus-new tests are replaced by the digest test at 600 seeds in the `slow` tier plus a 100-seed digest test in the fast core; the invariants stay),
`tests/paper/test_run_legacy_frozen.py` and `tests/paper/golden/run_legacy.sha256` (deleted), `tests/paper/test_run_rules.py` (`IMPLS` becomes `("new",)`), `tests/paper/test_run_structure.py`.

- [ ] **Step 1: Failing tests:** `test_trader_has_no_run_legacy`, `test_digests_still_match_for_the_first_100_seeds_in_the_fast_core`, `test_no_unused_import_in_trader_except_patch_targets` (ruff `F401` clean without a blanket `noqa`).
- [ ] **Step 2:** Delete; run the digest test **before** and **after** deleting: identical result. **Step 3:** the digest file is byte-identical to its Task 49 hash; the 103 transcripts and the rule tests pass.
- [ ] **Step 4:** The task gate. Commit: `refactor(paper): delete the frozen oracle copy of _run; recorded digests keep guarding`.

### Task 75 (OPTIONAL, gated on O-P5-1): retire `ModuleLedger`

Only if the owner approves a harness change or a test-side adapter. It would need a new harness whose 103 goldens stay byte-identical, which is a Phase 5.1 plan of its own. **Default: skipped, not part of this plan's gate.**

### Task 76: Lint survivors, docs, changelog (builder: Sonnet; reviewer: Sol)

**Files:** Modify `ruff.toml`, `docs/architecture.md` (the paper-trader section: the nine step objects, the data flow of one run, which seams stay and why, the rules table), `docs/changelog.md` (one entry: what moved, "no order, SQL statement, Telegram line or log record changed", the nets added, the preserved quirks, the one docstring edit in `ledger.py`, the migrated tests; merged with Phase 3's and 4's entries); `docs/cli_reference.md` is untouched (output did not change).

- [ ] **Step 1:** Run `ruff check --select C901,PLR0912,PLR0915,PLR0913 src/ggTrader/paper` with design 5.4's limits (`max-complexity = 15`, `max-args = 8`); list every survivor.
- [ ] **Step 2:** Enable the four rules for `src/ggTrader/paper` with a per-file ignore for each survivor, each with a one-line reason (a deliberate wide constructor, a long moved body whose only change was a move). Nothing outside `paper/` is switched on here (design 5.4 enables the rest later).
- [ ] **Step 3:** Docs. Remove the stale "verification status" paragraph Phase 4 left and replace it with the recorded gate results of Task 77 (written after it, in the same commit as the final evidence references).
- [ ] **Step 4:** The task gate. Commit: `docs: refactor phase 5 (the trader run split into step objects)`.

### Task 77: Integration gate (orchestrator)

- [ ] **Step 1:** `git log --oneline refactor/phase-5 --not refactor/phase-4` lists every task; `git status` clean.
- [ ] **Step 2:** The phase gate on the final head, one job at a time: fast core; `--full` and `--studies` (`$E/p5/gate_run.sh phase5`: oracle at the 6G cap, tests at 3G);
  the outcome check against `p5base-*.xml` with `--must-pass` listing every Phase 5 pin, all 197 `test_trader.py` ids and both live-golden nodes; `scripts/verify_pins.py` with the counts of `.github/workflows/ci.yml`.
- [ ] **Step 3:** The records: `sha256sum -c $E/p5-fixed.sha256 $E/p5-pins.sha256 $E/p5-added.sha256` all OK; `git diff --stat BASE5 -- src/ggTrader/paper/persist.py` empty; `git diff BASE5 -- src/ggTrader/paper/ledger.py` is the one docstring hunk; `git diff --stat BASE5 -- tests/paper/transcript tests/paper/golden/trader_transcripts tests/paper/golden/persist_sql.json` empty.
- [ ] **Step 4:** `grep -n "os.environ\|getenv\|_write\b" src/ggTrader/paper/trader.py` prints nothing; `_run` is at most 45 lines and `run` at most 25 (the ratchet); every new module is listed in `NEW_MODULES`.
- [ ] **Step 5:** The final mutation matrix (SM 1-14) on the final head. **Step 6:** report usage.

### Task 78: Independent reviews and the hand-back

- [ ] **Step 1: Sol** (fresh context; given the branch diff, the seven rules, the safety-invariant list, the immutable list and the claim "no behavior changed"; not given the builders' reports). Findings are checked by the orchestrator against the code before they are acted on.
- [ ] **Step 2: Opus** (fresh context; same evidence plus the SM matrix results; asked to find a mutation or an input class the nets miss). Opus alone judges the equivalence verdict and the two deviations that touch live safety (Deviation 1, the lazy planners).
- [ ] **Step 3:** Each finding is fixed in its own commit, with its test, and the affected gate steps rerun; a finding that is a candidate behavior change (the quirks) is **not** fixed and goes to the hand-back.
- [ ] **Step 4: Hand-back** to the owner: branch `refactor/phase-5` and its head, the gate results, the anchors before and after (`_run` 826 lines to at most 45; 239 persist patches to 0), the preserved quirks as candidate findings, the owner decisions still open (O-P5-1, O-P5-4, and the Telegram percentage slice), and usage. **No merge, push or deploy; the trader stays off.**

## Execution roles (`CLAUDE.md`, delegation preference of 2026-11-08)

**Quota first.** The owner reported ample Claude and Codex quota on 2026-11-19; the agent cannot see usage, so usage is reported after each phase and the owner can stop early.
Phases and expected runs:

| Phase | Tasks | Builder runs | Reviewer runs |
|---|---|---|---|
| A: nets | 46-51 | 3 Sonnet (47, 48, 50), 1 Terra (49), Haiku inventories | 2 Sol (47, 48-50), 1 Opus (49, Step 5 read) |
| B-C: inert pieces and moved bodies | 52-56 | 5 Terra | 5 Opus |
| D: wiring | 57-63 | 7 Terra | 7 Opus, then 1 Opus and 1 Sol at Task 64 |
| E: tests | 65-74 | 10 Terra | 10 Opus |
| F: docs, gate | 76-78 | 1 Sonnet | 1 Sol, final Sol and Opus |

- **Orchestrator: Claude Sonnet.** It re-verifies every task itself (reads the diff with `--color-moved`, reruns the task gate, runs the mutations, accepts no builder claim unchecked) and runs the oracle and `--full` runs one at a time.
- **The family rule.** The pins (Tasks 47, 48, 50) are Claude-built and reviewed by Sol; the refactoring under them (Tasks 52-63) is Codex-built and reviewed by Opus, so the family that writes the nets is not the family that moves code under them. The generator and digests (Task 49) are Codex-built and reviewed by Opus. The test migration (65-73) is Codex-built and reviewed by Opus. Whole-branch reviews (64 for the code, 78) use both Sol and Opus, each in a fresh context given only the evidence and the claims; a review of a family's own build says so.
- **Haiku** takes the inventories (Task 46 anchors, per-node assertion counts, the mutation-site map); a stronger model samples each batch and checks every figure that reaches a report.
- **Briefs** carry the task text, the file paths, the heavy-job block of Global Constraints (with the explicit ban on the oracle scripts), and no private account records.
- The lead remains responsible for the final call.
