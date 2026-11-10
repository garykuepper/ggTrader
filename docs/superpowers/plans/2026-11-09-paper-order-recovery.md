# Paper Order Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and reconcile every paper order across process interruptions without silently submitting a duplicate or recording a fill twice.

**Architecture:** Persist a unique submission intent before every broker call and associate it with a stable client order ID. Reconcile unresolved intents by that ID before new submissions; an ambiguous response blocks new orders and requires operator investigation, rather than assuming a missing broker result proves no order exists. Settle confirmed fills and resolve intents atomically in the database; preserve the existing pending-order path for orders predating the migration.

**Tech Stack:** Python, Alpaca-py, PostgreSQL/TimescaleDB, SQLAlchemy, pytest via `scripts/run_tests.sh`.

**Spec:** This plan and the accepted-order recovery finding in `docs/superpowers/plans/2026-11-09-refactor-review-findings.md`. Alpaca documents [lookup by client order ID](https://docs.alpaca.markets/us/reference/getorderbyclientorderid); no assumption that a 404 immediately proves non-acceptance is used here.

## Global Constraints

- This is paper-account production code. The trader remains stopped; no real broker submissions, production DB writes, timer changes, deploy, merge, or cron restart are authorized by this plan.
- Per the owner’s explicit ChatGPT-only direction in this conversation (overriding the usual cross-family review assignment for this work), use ChatGPT models only: Terra or Luna builds, fresh-context Sol reviews each slice; the lead independently inspects diffs and runs checks. Reviews are same-family, not cross-family independent. Owner reports ample quota; exact usage is unavailable to the agent.
- Preserve all existing `paper_pending_orders`, trade records, original run dates, reasons, and existing account data. Do not delete `data/`. Schema migration must be additive, idempotent, and tested on an isolated test database or transactional fake, never the production database.
- Existing frozen evidence and WFO behavior are untouched. Tests must not read the real `.env`, call a real broker, or connect to a production database. Unit tests use fakes; explicitly marked integration tests may use only a disposable PostgreSQL instance with a dedicated test-only DSN and an explicit identity guard that rejects the production host and database. If unavailable, report PostgreSQL transaction, DDL and lock behavior unverified. Use `scripts/run_tests.sh`, never bare pytest. Full tier and Ruff required before a merge.
- Use a stable randomly generated UUID client ID persisted before submission; never generate a replacement ID for an unresolved intent. The ID is scoped to one actual intent, not a date/symbol key; legitimate later trades can have different IDs. The intent row is retained after resolution for audit.
- Never retry an ambiguous submission automatically, even after a lookup returns not found. Preserve it as unresolved, notify, and block new submissions until manually resolved by an explicitly authorized future procedure. Do not invent that procedure as part of this plan.
- No claim of exactly-once broker execution or perfect fencing: hold a PostgreSQL session advisory lock for each live run. A connection can fail after its last ownership check; a stale process could still call Alpaca. Every replacement run must reconcile all committed unresolved intents and abort all new submissions while any remains unresolved, including known working orders. This depends on committed intents being visible to the replacement and all submitters following this protocol; manual or external broker orders are outside its guarantee.

## Review Focus

1. Broker accepts then connection drops: retain intent, recover by client ID, never resubmit (Task 3).
2. Crash before broker call or lookup returns 404/timeout: treat as ambiguous, alert and block, including across restart (Task 3).
3. Fill booking interrupted between insert and resolution: rollback both; replay records one fill (Task 2).
4. Two live runs overlap or a database lock is lost: second runner stops on unresolved intent even if lock loss occurs between ownership check and submission; stale-run fencing is not guaranteed (Task 3).
5. Existing pending rows, partial fills, terminal cancellations and read-only dry runs retain correct accounting and never submit (Tasks 2–3).

## File map

- Modify `src/ggTrader/paper/persist.py`: additive intent schema, lock, and transactional settlement; retain legacy pending APIs.
- Modify `src/ggTrader/paper/alpaca_broker.py`: carry client ID to market requests and normalize client-ID lookup.
- Modify `src/ggTrader/paper/trader.py`: acquire lock, reconcile before submissions, route all five order paths through the durable intent seam.
- Modify `tests/paper/test_paper_persist.py`, `tests/paper/test_alpaca_broker.py`, `tests/paper/test_trader.py`: focused tests with fake broker and isolated storage; update trader autouse stubs for new persist imports.
- Modify `docs/architecture.md`, `docs/cli_reference.md`, `docs/changelog.md`: describe recovery, unresolved state and operational limitations. No new command to clear ambiguity in this scope.

### Task 1: Broker client identity

**Files:** `src/ggTrader/paper/alpaca_broker.py`, `tests/paper/test_alpaca_broker.py`.

**Interfaces:** `submit_buy(symbol: str, notional: float, client_order_id: str) -> str`; `submit_sell(symbol: str, qty: float, client_order_id: str) -> str`; `get_order_by_client_id(client_order_id: str) -> dict`. The lookup returns the same normalized shape as `get_order` and propagates all errors, including not found.

- [ ] Add adapter tests asserting both request types pass `client_order_id`, preserve side, quantity/notional and DAY time-in-force, and return the broker ID; assert lookup uses the SDK's `get_order_by_client_id` and normalizes client ID, broker ID, side, symbol, requested quantity/notional, status, filled quantity and average price, preserving missing fill values as `None`. Assert lookup exceptions propagate.
- [ ] Run `scripts/run_tests.sh tests/paper/test_alpaca_broker.py`; expect the new tests to fail before implementation.
- [ ] Implement those signatures and share order normalization with `get_order`; do not add retry logic. Update existing adapter tests for the required argument.
- [ ] Run the same test command; expect all adapter tests to pass. Obtain fresh Sol review of the adapter diff before proceeding.

### Task 2: Durable intent and atomic settlement

**Files:** `src/ggTrader/paper/persist.py`, `tests/paper/test_paper_persist.py`.

**Interfaces:** `create_order_intent(client_order_id: str, run_date: str, side: str, symbol: str, amount: float, amount_kind: str, reason: str | None) -> None`; `get_unresolved_order_intents() -> list[dict]`; `attach_broker_order(client_order_id: str, order_id: str) -> None`; `settle_order_intent(client_order_id: str, order_id: str, filled_qty: float, filled_avg_price: float | None, terminal_status: str) -> None`. `amount_kind` is `notional` for buys or `qty` for sells; preserve the broker quantity and price as settlement evidence and derive the dollar value only after validation. A context manager `paper_run_lock()` acquires and holds a dedicated PostgreSQL advisory lock on one connection, fails if unavailable, and raises on lock-connection failure.

- [ ] Add an isolated-database test proving repeated schema initialization preserves legacy rows; intent creation commits before return; duplicate client IDs fail; unresolved queries include new and attached intents; attach is idempotent for the same ID and rejects conflicting broker IDs.
- [ ] Add atomic settlement tests: filled and partially-filled-then-canceled orders insert one trade for actual filled quantity × average price, preserve original run date/reason, and mark resolved in the same transaction. Inject failure after trade insertion and assert both writes roll back; retry and assert exactly one trade. A zero-fill terminal order resolves without a trade. Assert missing, non-finite or negative quantities, positive quantity with missing/non-finite/non-positive price, and `filled` with zero quantity all fail closed without resolving; compare quantity and price on replay, not just their product. Repeated settlement with the same identity does not duplicate a trade; conflicting settlement fails.
- [ ] Add lock tests with two independent isolated connections: second acquisition fails while first holds it, succeeds after release; a lost lock connection is detected before another submission. Test without production DB access.
- [ ] Run `scripts/run_tests.sh tests/paper/test_paper_persist.py`; expect the new tests to fail before implementation.
- [ ] Add a `paper_order_intents` table with primary key `client_order_id`, immutable payload and run date, nullable unique `broker_order_id`, state (`intent`, `submitted`, `resolved`), terminal status and timestamps. Enforce side, amount-kind, positive amount and state constraints. Implement the interfaces using transactions and row locks. Atomically check an existing trade by broker ID before insert; if historical duplicates or conflicting values exist, fail closed without editing them. Do not add an unverified unique constraint to historical `paper_trades`.
- [ ] Implement `paper_run_lock()` with a dedicated connection and transaction/session lock lifecycle; verify lock ownership immediately before every submission using the same live connection. A lock-loss error aborts the live run.
- [ ] Run the same tests; expect all to pass. Fresh Sol reviews migration, transaction boundaries, and lock behavior.

### Task 3: Submission and reconciliation integration

**Files:** `src/ggTrader/paper/trader.py`, `tests/paper/test_trader.py`, `docs/architecture.md`, `docs/cli_reference.md`, `docs/changelog.md`.

**Interfaces:** `PaperTrader._submit_order(side: str, symbol: str, amount: float, reason: str | None) -> str` creates the intent, checks lock ownership, submits with its ID, and attaches the broker ID before returning. `PaperTrader._reconcile_order_intents() -> set[str]` looks up each unresolved intent by client ID, validates returned client ID and broker ID, case-normalized symbol and side, and requested quantity or notional against the immutable intent using decimal precision (quantity within 0.000000001 shares; notional within $0.01); attaches the broker ID and reconciles broker status. Missing or non-finite values, wrong identity, and amount outside tolerance abort reconciliation. These are internal seams; existing CLI remains unchanged.

- [ ] Add run-level tests using an in-memory transactional fake and fake broker: acceptance followed by exception; process interruption after acceptance but before attach; restart finds the same client ID, submits nothing again, and tracks or settles it; lookup 404/timeout/wrong client ID or mismatched symbol, side or requested amount blocks all new submissions and leaves intent unresolved. Verify intent is committed before each broker call.
- [ ] Add tests for interruption after intent creation but before submission, failed intent write, failed broker-ID attach, active and terminal statuses, partial fill then cancellation, and settlement failure. Assert no second broker submission and no duplicate ledger entry after restart. Confirm both ordinary orders, catastrophe stop, and both cash-sweep directions use the seam.
- [ ] Add tests for two overlapping runs, lost lock before a later submission, and loss after the ownership check but before broker invocation: the replacement sees the committed unresolved intent and submits nothing, while the test records that the stale process cannot be fenced from Alpaca. Add legacy pending order reconciliation and dry-run/read-only operation with no writes or submissions. Preserve current fail-closed pending-order lookup.
- [ ] Run `scripts/run_tests.sh tests/paper/test_trader.py`; expect new tests to fail before implementation.
- [ ] Acquire the lock before live reconciliation and hold it through final persistence. Route all five direct submission sites through `_submit_order`; create intent before each submit, attach immediately after broker acceptance. Any submission exception or attach failure leaves an unresolved intent and aborts further submissions.
- [ ] Reconcile intents before any new orders. Working or partial statuses retain the unresolved intent and block all new submissions for the run. Filled and terminal unfilled/partially-filled statuses settle atomically with actual filled value. Unknown, mismatched, absent or error responses notify and abort the run. Use the same atomic settlement path for orders filled during the current run. New intent-backed orders must never be written to `paper_pending_orders`; disable that path for them. Retain legacy pending reconciliation and ensure its fill insert plus removal is also atomic and replay-safe. If an order appears in both tables, abort and alert without settling either representation until investigated; test this on both recovery pass orderings and after interruption.
- [ ] Update the three documentation files with the failure and recovery behavior, the manual-investigation requirement, schema/lock implications and the stopped-trader status. Do not document an unimplemented reset command.
- [ ] Run `scripts/run_tests.sh tests/paper/test_trader.py tests/paper/test_paper_persist.py tests/paper/test_alpaca_broker.py`; expect all to pass. Obtain fresh-context Sol review of the integrated diff and resolve substantive findings. Lead inspects every submission site and reruns the targeted tests.

### Task 4: Verification and handoff

**Files:** all changed files; `docs/superpowers/plans/2026-11-09-refactor-review-findings.md`.

- [ ] Run the disposable-PostgreSQL integration checks with the guarded test-only DSN; if unavailable, explicitly report DDL, rollback, and advisory-lock semantics as unverified. Run `scripts/run_tests.sh --full`, `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .`; record results, failures and skips. Run relevant paper golden checks if available and mark missing frozen-tape evidence explicitly rather than treating skips as a pass.
- [ ] Lead checks the complete diff, migration on an isolated pre-existing schema, tests and reviewer findings; document residual risks, same-family review limitation and any unavailable evidence in the ledger.
- [ ] Report the reviewable result and verification. Do not merge, push, deploy, migrate the production DB, or re-enable the trader without separate authorization.
