# dead-stock-fresh-window

**Question:** Does the first test's fixed 12-1 momentum rule, with dead stocks included, end ahead of SPY after costs and taxes on 2000 to 2004, a window it has never been run on?

**Kind:** screen. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-24-dead-stock-fresh-window-brief.md](2026-11-24-dead-stock-fresh-window-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py),)

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-fresh-window/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
