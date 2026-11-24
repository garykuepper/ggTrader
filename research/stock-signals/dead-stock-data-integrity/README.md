# dead-stock-data-integrity

**Question:** For the dead stocks the first test held, what do our CRSP rows and SEC filings say about why they stopped trading, and how clean are the price rows around them?

**Kind:** audit. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-24-dead-stock-data-integrity-brief.md](2026-11-24-dead-stock-data-integrity-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/dead_stock_data_integrity.py](code/dead_stock_data_integrity.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
