# dead-stock-first-test

**Question:** Does a fixed 12-1 momentum rule over liquid NYSE and Nasdaq stocks, dead ones included, beat SPY after trading costs and taxes, at dead-stock losses of 1.0, 0.5 and 0.0?

**Kind:** screen. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-23-dead-stock-first-test-brief.md](2026-11-23-dead-stock-first-test-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/dead_stock_first_test.py](code/dead_stock_first_test.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-first-test/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
