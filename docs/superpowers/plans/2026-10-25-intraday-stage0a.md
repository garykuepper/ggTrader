# Intraday Stage 0A implementation

Spec: research/intraday/spy-momentum-feasibility/2026-10-25-intraday-stage0a-contract.md

## Global Constraints

Stage 0A only. No Stage 0B, bulk pulls, signals, returns, orders, production changes, credentials in artifacts or price fields. The contract is authoritative. No new dependencies. All pytest runs use scripts/run_tests.sh with its unchanged memory cap.

### Task 1: Implement and verify the metadata probe

- [x] Read the spec above and implement all its fixed request, whitelist, bounded retry, pagination, atomic checkpoint and conclusion requirements.
- [x] Own only src/ggTrader/lab/intraday_probe.py, scripts/intraday_coverage_probe.py, tests/lab/test_intraday_probe.py.
- [x] Use mocked tests first for the nontrivial paths named by the spec; never network or read .env. Run targeted capped tests and Ruff check/format for the three files.
- [x] Ensure --help works without credentials, network or artifact writes. Keep wrapper thin and core logic in src.
- [x] Self-review, commit only your three files, and write a report with tests and remaining concerns.

Parent then obtains independent task and whole-branch reviews, runs required checks, executes the authorized metadata-only probe, and records results and documentation. No push or merge.
