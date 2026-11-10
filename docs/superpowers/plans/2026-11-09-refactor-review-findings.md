# Refactoring review findings — 2026-11-09

Baseline: `main` at `4a0daaa034138c43d4d16631e1fdf6946fbaf583`; untracked `data/external/event_timing_audit/` and the plan preserved. Ruff check and format check passed (362 files). Fast tier passed: 1,954 passed, 7 skipped, 409 deselected; skips include absent frozen tape. Full tier not run. Reviews are selective static analysis, not a full safety certification. Three fresh-context Sol reviews; same model family as the lead. Model usage is not visible to the lead.

## Verified code paths and priorities

1. **Paper pending-order lookup fails open** — `src/ggTrader/paper/trader.py:231` catches the read failure and returns an empty set; later order paths use that set to suppress conflicting orders. Prioritize fail-closed behavior and a run-level broker submission test.
2. **Accepted-order recovery gap** — `trader.py:1041` keeps a broker-accepted order in memory until persistence at `trader.py:1134`; a crash can leave an unrecorded working order. Recovery and idempotency design requires broker contract and persistence analysis before selecting a fix. Do not claim exactly-once behavior without proving it.
3. **WFO shadow reentry uses deployed anchor metrics** — `src/ggTrader/lab/wfo/run.py:283` selects the anchor when halted, then computes WFE from winner training and deployed test Sharpe at line 307 and passes it to reentry. Add a run-level divergent winner/anchor regression and assess impacts on pinned evidence before changing behavior.
4. **Cache write failure can return stale data** — `src/ggTrader/data/live/cached_yfinance_loader.py:274` uses autocommit and logs write failures; its DB-first return at line 395 can omit fresh downloaded bars. Specify transactional and caller failure semantics after examining consumers.
5. **Backfill refresh race** — `scripts/backfill_missing_members.py:335` updates by key against a backup taken from an earlier read. A concurrent correction could be overwritten without being in the backup. Investigate conditional updates or locking and test the race without real DB writes.
6. **Blend ignores sweep overrides** — `src/ggTrader/lab/cli.py:116` accepts `--sweep-param`, but only the WFO path passes it onward. Reject the incompatible combination before running, subject to a focused test.
7. **Unknown membership comparison** — `src/ggTrader/engine/provenance.py:239` adds a note without a non-comparable flag. Existing tests encode this behavior; changing it is a policy decision, not a routine refactor. Assess both-missing as well as one-missing provenance and callers.
8. **Test runner 137 guidance contradicts current instructions** — `scripts/run_tests.sh:188` suggests raising the cap or fixing a leak; AGENTS.md requires investigating hangs and clock, network, and database interactions first. Correct diagnostic and associated tests.

The backfill and live cache differ in safeguards and timestamp treatment; a shared helper is not yet justified without a contract analysis. No historical result, DB, broker, timer or production code has been changed. Priorities are provisional; review of existing tests and reachability is required in each slice plan.

## Slice update

- Pending-order lookup: live failure now aborts before submissions; dry run warns and continues. Fresh-context Sol review found no actionable gaps. Lead reran `scripts/run_tests.sh tests/paper/test_trader.py`: 138 passed. Builder fast tier: 1,957 passed, 7 skipped; full tier exited successfully; study tier: 777 passed, 2 deselected. Frozen-tape-dependent checks remain unavailable; no real-tape oracle claim is made. Same-family review limitation applies.
- WFO reentry: the mixed winner-training/deployed-test WFE is established in `tests/lab/wfo/test_toy_known_answers.py` and the preregistered synthetic methodology. An alternative winner-shadow metric would be a policy change, with potentially different halt dates and research outcomes; its impact remains unmeasured. Preserve legacy and existing evidence. A separately registered policy comparison and pinned-tape oracle are needed before proposing any change; the frozen tape was absent during baseline.
