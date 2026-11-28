# early-decades

**Question:** How do the three fixed recent-winner rules compare with the market after costs and taxes?

**Kind:** screen. **Status:** scored once on 2026-11-28 (report below; no GO or NO-GO label; the window is spent). Reviewed by Gemini 4 Neon and Claude Opus (figures checked against the raw results; wording fixes applied). **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-27-early-decades-brief.md](2026-11-27-early-decades-brief.md)
- Report: [2026-11-28-early-decades.md](2026-11-28-early-decades.md); raw results `_early-A.json`, `_early-B.json`, `_early-C.json` and the regression proof `_regression.json` (git-ignored, kept in this folder)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/ed_run.py](code/ed_run.py))

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/early-decades/code/ed_run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).

The early-window driver requires a matching regression output before it will read the early
window. After the owner authorizes the one-time runs, pass `--window early --arm A` (or B or C),
`--eval-start 1969-02-03 --eval-end 1998-07-31`, `--regression-proof` pointing to the matching
2020–2024 JSON, and a distinct `--out` path. Use `scripts/run_capped.sh` and run each arm
serially. Each arm claims its window before data are read; a claim remains after a failed run
and requires the owner to decide what to do. Each output contains six cells (two costs by three
loss readings), for 18 across all three arms.

Before claiming, the driver checks that the checkout is clean and the linked Yahoo data have
not drifted. Claims are `research/stock-signals/early-decades/_early-A.claim`, `_early-B.claim`
and `_early-C.claim`; each records the time, commit, command and output path. If a crash occurred
before any scoring, only the owner may decide to remove the affected claim file after checking
that no result was produced. Keep a claim when scoring may have occurred.

The regression proof can be written to `research/stock-signals/early-decades/_regression.json`
and supplied with `--regression-proof` for each arm. Put each arm's `--out` at
`research/stock-signals/early-decades/_early-A.json`, `_early-B.json` or `_early-C.json`.
These files are ignored by Git and must not be committed: the proof is tied to the code commit.
The preflight checks changes to tracked files; it permits untracked output files. A result marked
`loading` or `metrics_pending` is a checkpoint, not a completed result, and its claim remains.
