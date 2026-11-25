# dead-stock-fresh-window

**Question:** Does the first test's fixed 12-1 momentum rule, with dead stocks included, end ahead of SPY after costs and taxes on 2000 to 2004, a window it has never been run on?

**Kind:** screen. **Status:** closed (2026-11-24, UNRESOLVED; no GO or NO-GO label; see the report). **Category:** stock-signals (placement row 9). The rule trailed SPY at all 18 settings; the window is used (a second run would not be fresh).

## Where things are

- Brief (registered plan, frozen before any code): [2026-11-24-dead-stock-fresh-window-brief.md](2026-11-24-dead-stock-fresh-window-brief.md)
- Report: [2026-11-24-dead-stock-fresh-window.md](2026-11-24-dead-stock-fresh-window.md)
- Results: `_anchor-A.json` (the check: 2020 to 2024 reproduces the first test's saved cells exactly), `_fresh-A.json` (the one fresh run); timing and check notes `_run_notes.txt`; every trade in `results/dead_stock_fresh_window/` (not tracked by git, with the run-once claim file); pin: [PIN.json](PIN.json)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py); [code/fresh_inputs.py](code/fresh_inputs.py) reads a window; [code/fresh_compare.py](code/fresh_compare.py) holds the check and the gap table; `code/fp_*.py` are copies of the first test's engine files with only three import lines changed, recorded with hashes in [code/frozen_copies.json](code/frozen_copies.json) and checked by a test).
- Inputs it reads: the published combined price release 1 and raw CRSP load 1 of our database (read only), and the first test's `_holdout-A.json` (for the check). It never writes to the first test's folder or any other study's.

## How to reproduce

Run from the repository root with a clean tracked tree, one after the other, through the capped-run wrapper (the check took about 10 minutes, the fresh run about 12):

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-fresh-window/code/run.py --window anchor --release-id 1 --crsp-load 1
    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-fresh-window/code/run.py --window fresh --release-id 1 --crsp-load 1

The fresh command refuses unless the check passed on exactly the same code, and refuses when a run-once claim exists in `results/dead_stock_fresh_window/` or a result file exists, so it cannot be repeated in this checkout.

To replay the fresh run in a scratch clone, use `scripts/study.py repro research/stock-signals/dead-stock-fresh-window` (it clones the pinned commit 3277570e, which holds the passing anchor file and the frozen code but no fresh result, and runs the pinned fresh command; add `--print-only` to see the plan). A replay is not a new test: the window is used.
