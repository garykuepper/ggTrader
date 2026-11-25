# dead-stock-payout-rerun

**Question:** TODO: state the one question this study answers, in one sentence.

**Kind:** screen. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-24-dead-stock-payout-rerun-brief.md](2026-11-24-dead-stock-payout-rerun-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/dead_stock_payout_rerun.py](code/dead_stock_payout_rerun.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
