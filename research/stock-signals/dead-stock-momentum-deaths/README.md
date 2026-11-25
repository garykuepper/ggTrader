# dead-stock-momentum-deaths

**Question:** Why did the first test's momentum rule hold two to four times as many stocks on the day they stopped trading as 50 randomly drawn stocks did?

**Kind:** audit. **Status:** closed (2026-11-25, UNRESOLVED; no GO or NO-GO label; see the report). **Category:** stock-signals (placement row 9). In short: strong risers stop trading 2.5 to 3.6 times as often as the average stock on the list; the dying picks were mostly stocks with a one-day jump of 20 percent or more in the scoring year and a flat last month. That fits buy-outs but the study cannot tell them from other jumps.

## Where things are

- Brief (frozen before any code, one dated amendment): [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md)
- Report: [2026-11-25-dead-stock-momentum-deaths.md](2026-11-25-dead-stock-momentum-deaths.md)
- Results: `_early.json` (2000 to 2004), `_dev.json` (2005 to 2019), `_holdout.json` (2020 to 2024)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py); [code/momentum_deaths.py](code/momentum_deaths.py) holds the measures; [code/momentum_inputs.py](code/momentum_inputs.py) reads a window; `code/dm_*.py` are copies of the first test's engine files with only import lines changed, recorded with hashes in [code/frozen_copies.json](code/frozen_copies.json) and checked by a test).
- Inputs it reads: the published combined price release 1 and raw CRSP load 1 of our database (read only), and the saved result files of the first test, the fresh-window test and the random-baseline study (read only).

## How to reproduce

From the repository root with a clean tracked tree, through the capped-run wrapper (about 2 to 4 minutes a window):

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-momentum-deaths/code/run.py --window early --release-id 1 --crsp-load 1

Use `--window dev` or `--window holdout` for the other two. Each run first checks that its top 50 equals the copied planner's on every decision day and stops if not.
