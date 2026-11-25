# dead-stock-early-filings

**Question:** For every stock in the first test's list that stopped trading from 2000 to 2004, what do the SEC's free filings say about how it ended?

**Kind:** audit. **Status:** closed (2026-11-25, UNRESOLVED; no GO or NO-GO label; see the report). **Category:** stock-signals (placement row 9). In short: of 251 stocks, 8 receive an acquisition-type or bankruptcy label (two more have an acquisition notice but stay labelled kept filing) and 18 a removal notice only; for the 33 the rule held, 1 has an acquisition-type filing. The search read the company's own 8-K and Form 25 filings, which for these years say little; no payment amount was read.

## Where things are

- Brief (frozen before any code, five dated notes): [2026-11-25-dead-stock-early-filings-brief.md](2026-11-25-dead-stock-early-filings-brief.md)
- Report: [2026-11-25-dead-stock-early-filings.md](2026-11-25-dead-stock-early-filings.md)
- Results: `_early.json` and `_early_labels.csv` (the full run); `_pilot*.json` and `_pilot*_labels.csv` (three pilot versions kept as evidence of the repairs)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (driver [code/run.py](code/run.py); roster, names, SEC wrapper and header parser, name matching and classification modules; `code/ef_*.py` are copies of the first test's engine files and of the earlier SEC client, hash-recorded in `frozen_copies.json` and `frozen_sec_copies.json` and checked by tests).
- Inputs it reads: release 1 and raw CRSP load 1 and the McGill table of our database (read only), and the SEC's public quarterly index and filing files (cached under `/media/thesix/ggTrader-data/sec-early-window/`, outside the repository).

## How to reproduce

From the repository root with a clean tracked tree and the contact line in `~/.config/ggtrader/sec_contact` (answers are cached, so a rerun makes few or no new requests):

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-early-filings/code/run.py --mode full --release-id 1 --crsp-load 1
