# dead-stock-random-baseline

**Question:** Is the first test's momentum rule behind SPY because of how it picks stocks, or would 50 stocks drawn by chance from the same list, with the same costs and taxes, trail SPY by about as much?

**Kind:** screen. **Status:** closed (2026-11-25, UNRESOLVED; no GO or NO-GO label; see the report). **Category:** stock-signals (placement row 9). In short: chance does not give one answer. The rule's picks were worse than chance in 2000 to 2004, worse at the two lower dead-stock readings and about level at the best in 2005 to 2019, and better in 2020 to 2024; in 2005 to 2024 almost every chance portfolio also trailed SPY.

## Where things are

- Brief (registered plan, frozen before any code, with four dated notes): [2026-11-25-dead-stock-random-baseline-brief.md](2026-11-25-dead-stock-random-baseline-brief.md)
- Report: [2026-11-25-dead-stock-random-baseline.md](2026-11-25-dead-stock-random-baseline.md)
- Results: `_early.json` (2000 to 2004), `_dev.json` (2005 to 2019), `_holdout.json` (2020 to 2024): every draw, the check against the first test's saved numbers, the sanity block and the summary. Resumable partial files are in `results/dead_stock_random_baseline/` (not tracked by git).
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py); [code/random_plans.py](code/random_plans.py) draws the portfolios; [code/dead_stock_random_baseline.py](code/dead_stock_random_baseline.py) holds the summary logic; `code/rb_*.py` are copies of the first test's engine files, recorded with hashes in [code/frozen_copies.json](code/frozen_copies.json) and checked by a test; one copy has a recorded one-line change to a rounding check).
- Inputs it reads: the published combined price release 1 and raw CRSP load 1 of our database (read only) and the saved results of the first test and the fresh-window test (for the check). It never writes to another study's folder.

## How to reproduce

From the repository root with a clean tracked tree, one window at a time, through the capped-run wrapper (the 2005 to 2019 window takes about 3 hours and needs `GGT_RUN_MAX_SECONDS=14400`):

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-random-baseline/code/run.py --window early --release-id 1 --crsp-load 1
    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-random-baseline/code/run.py --window holdout --release-id 1 --crsp-load 1
    GGT_RUN_MAX_SECONDS=14400 scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-random-baseline/code/run.py --window dev --release-id 1 --crsp-load 1

Each run first reproduces the rule's saved headline results to half a cent and refuses to draw if they differ. The seeds are fixed by a rule in the brief, so a rerun gives the same draws.
