# dead-stock-momentum-deaths

**Question:** TODO: state the one question this study answers, in one sentence.

**Kind:** audit. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-25-dead-stock-momentum-deaths-brief.md](2026-11-25-dead-stock-momentum-deaths-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/dead_stock_momentum_deaths.py](code/dead_stock_momentum_deaths.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-momentum-deaths/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
