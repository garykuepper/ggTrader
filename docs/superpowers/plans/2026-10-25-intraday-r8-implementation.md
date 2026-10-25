# Intraday r8 Mocked Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. All task steps use the approved protocol; no second owner check is needed for this already-authorized implementation phase.

**Goal:** Build the frozen research collector, signal-blind validator and single-shot scorer without fetching or scoring candidate data.

**Architecture:** One data/transport module, one pure validation/scoring module, a thin standalone CLI and focused synthetic tests. Reuse existing safe helpers where appropriate; no lab registry, DB or production integration.

**Tech Stack:** Python, requests, pandas/numpy, statsmodels HAC, existing yfinance only for free ancillary corporate-action/benchmark inputs if needed. New dependencies recorded before installation.

**Spec:** docs/research/briefs/2026-10-25-intraday-feasibility-r8.md

## Global Constraints

- Approval 1 only: synthetic/mocked implementation; no real market-data, credentials, signals/returns, orders or production changes.
- SPY only, no borrowing/shorting. Registered scored window 2016-01-05–2026-09-30; 2016-01-04 anchor. Prior close uses immediately preceding official session's last minute including early-close anchors.
- Profitability A1: one-sided intercept-only NW Bartlett 5 lags, use_correction=True, unrounded t>=3; A2 standalone after-tax-policy CAGR>=3%/yr. Costs1bp/side base2bp/side stress; annual24%positiveP&L debit, no loss benefit/carryforward, terminal partial-year tax; zero idle cash. Raw-price proxy, no auctions/IEX.
- Validator never computes signal/return. All four required endpoint valuations on every full session; missing/invalid endpoints, provenance/calendar faults and split/vendor-adjustment changes yield UNRESOLVED. No silent zeroing, patching, quota or rolling start. Interior gaps diagnostic.
- All pytest via scripts/run_tests.sh unchanged memory cap; Ruff gates. Frozen Stage0A module/artifact, src/ggTrader/paper/, loaders, DB, Docker, cron untouched.

## Review Focus

- Hash tampering/stale validation must prevent scoring; code/protocol/data identities must match and actual inputs must be rehashed.
- Early-close anchors and DST must not become wrong-date or wrong-minute fills.
- Resume after failure must not mix vendor snapshots, silently drop failed chunks or mark partial pagination complete.
- Multi-year tax compounding must use after-debit sizing and terminal tax, not sum daily tax haircuts.
- Calendar/ancillary-source gaps must not create a benchmark or verdict from incomplete inputs; no real data in tests.

### Task 1: Build the isolated research pipeline

**Files:**
- Create src/ggTrader/lab/intraday_momentum.py (protocol/data validation, pure calculations and result artifacts).
- Create src/ggTrader/lab/intraday_momentum_data.py (bounded resumable collection and provenance).
- Create scripts/intraday_momentum_research.py (thin CLI; help/default never access credentials/network).
- Create tests/lab/test_intraday_momentum.py and tests/lab/test_intraday_momentum_data.py.
- Modify pyproject.toml only to record statsmodels if required. No unrelated dependency updates.

**Interfaces:** Define the smallest explicit versioned dataset, PASS validation and score-result structures in core. The collector returns/stores only that schema; the scorer consumes the exact validated data plus matching protocol/code/version identities. Specify public function signatures in a short module docstring or implementation report; no plugin/framework or registry.

- [ ] Original strict test-first ordering was missed (documented process deviation); fixes used red-first regression tests. Behavior coverage completed: hand-computed signals/costs/annual taxes, known statsmodels NW reference, flat days, equal signal zero, invalid variance, half-day/DST anchors, missing any endpoint regardless signal, malformed calendar, hashes tampered after PASS, one-score refusal, metadata privacy, resumable pagination/repeated tokens/retries/deferred, help without credentials/network.
- [x] Implement pure validator/scorer and collection integration to the approved spec. Scoring is callable only against verified PASS data with matching identities. Keep a common date/price basis for strategy, always-long and distribution-reinvested SPY diagnostic; record all assumptions and raw JSON.
- [x] Implement bounded GET-only future collection with inherited credentials (no dotenv/production imports), fixed inputs, per-page response hashes, sanitized errors, complete pagination, finite retries/timeouts/rate spacing and resumable atomic checkpoints. No tokens/credentials/error bodies in logs. Persist prices only in future approved stage, never in this task's real run. No overwrites of frozen snapshots/results.
- [x] Keep collector input manifest precise: official calendar, raw SIP minute bars, raw daily closes, and the minimal free dividend/ex-date/split history needed for mandatory diagnostics. No network now; test source adapters through mocks. A later data approval reviews this manifest. No unsupported empty corporate-action history accepted as verified absence.
- [x] Register any required dependency before using the existing venv for installation. Run targeted capped tests, Ruff and --help; self-review. Do not run full suite (parent does once at final gate), real collector, real scorer, network examples or broker helpers.
- [x] Commit only task-owned files and write report with interfaces, source manifest, code/runtime versions, commands/outcomes, review concerns. Parent owns docs and final reviews.

Parent then dispatches an independent Opus task review and whole-branch review, resolves findings, runs full capped suite and updates docs. End with the concrete future data/conditional-scoring request; do not execute approval 2.
