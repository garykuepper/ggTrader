# dead-stock-early-filings

**Question:** TODO: state the one question this study answers, in one sentence.

**Kind:** audit. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-25-dead-stock-early-filings-brief.md](2026-11-25-dead-stock-early-filings-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/dead_stock_early_filings.py](code/dead_stock_early_filings.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-early-filings/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
