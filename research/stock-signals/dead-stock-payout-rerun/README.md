# dead-stock-payout-rerun

**Question:** If the dead stocks the first test held that have filing evidence of an acquisition are sold at their last price, how far behind or ahead of SPY is the same 12-1 momentum rule after costs and taxes, with the other dead stocks kept at the three loss readings?

**Kind:** screen. **Status:** closed (2026-11-24, UNRESOLVED; no GO or NO-GO label; see the report). **Category:** stock-signals (placement row 9). A re-score of two windows the first test had already seen, not a fresh test.

## Where things are

- Brief (registered plan, frozen before any rerun code): [2026-11-24-dead-stock-payout-rerun-brief.md](2026-11-24-dead-stock-payout-rerun-brief.md)
- Report: [2026-11-24-dead-stock-payout-rerun.md](2026-11-24-dead-stock-payout-rerun.md)
- Results: `_rerun-dev-A.json`, `_rerun-holdout-A.json`; timing note `_run_notes.txt`; every trade in `results/dead_stock_payout_rerun/` (not tracked by git); pin: [PIN.json](PIN.json)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py); [code/payout_prices.py](code/payout_prices.py) and [code/payout_rerun.py](code/payout_rerun.py) are the new logic; `code/ft_*.py` are copies of the first test's engine files, checked by [code/tests/test_frozen_copies.py](code/tests/test_frozen_copies.py) against [code/frozen_copies.json](code/frozen_copies.json))
- Inputs it reads: `research/stock-signals/dead-stock-data-integrity/_payouts_v2.csv` (fingerprint checked) and the first test's `_dev-A.json` and `_holdout-A.json`. It never writes to the first test's folders and never touches its lock files.

## How to reproduce

Run from the repository root with a clean tracked tree, one after the other, through the capped-run wrapper (the development run took about 25 minutes; the holdout run was not timed; read-only database access):

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --window dev --release-id 1 --crsp-load 1
    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --window holdout --release-id 1 --crsp-load 1

Each command refuses to start if an output exists (add `--force`), and stops before publishing if the reproduction check against the first test's saved cells or the best-case identity check fails.
