# early-decades

**Question:** TODO: state the one question this study answers, in one sentence.

**Kind:** screen. **Status:** open. **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-27-early-decades-brief.md](2026-11-27-early-decades-brief.md)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py), module [code/early_decades.py](code/early_decades.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/early-decades/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).

The early-window driver requires a matching regression output before it will read the early
window. After the owner authorizes the one-time runs, pass `--window early --arm A` (or B or C),
`--eval-start 1969-02-03 --eval-end 1998-07-31`, `--regression-proof` pointing to the matching
2020–2024 JSON, and a distinct `--out` path. Use `scripts/run_capped.sh` and run each arm
serially. Each arm claims its window before data are read; a claim remains after a failed run
and requires the owner to decide what to do. Each output contains six cells (two costs by three
loss readings), for 18 across all three arms.
