# dead-stock-data-integrity

**Question:** For the dead stocks the first test held, what do our CRSP rows and SEC filings say about why they stopped trading, and how clean are the price rows around them?

**Kind:** audit. **Status:** closed (2026-11-24, UNRESOLVED; see the reports). **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-24-dead-stock-data-integrity-brief.md](2026-11-24-dead-stock-data-integrity-brief.md)
- Reports: [Phase 1 findings, our own price rows](2026-11-24-dead-stock-data-integrity.md) and [payout evidence from the SEC's filings](phase-2/2026-11-24-dead-stock-payout-evidence.md) (kept in `phase-2/` because the study tool allows one plainly named dated report at the top)
- Results: `_integrity.json`, `_roster.csv`, `_form25_matches.json`, `_payouts.json`, `_payouts.csv`; hand checks: [_hand_checks.txt](_hand_checks.txt); pin: [PIN.json](PIN.json) (pins the three results `_integrity.json`, `_form25_matches.json` and `_payouts.json`, and the roster and `_payouts.csv` as inputs; see the payout report, section 10)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (Phase 1 driver [code/run.py](code/run.py), module [code/dead_stock_data_integrity.py](code/dead_stock_data_integrity.py); SEC phases `.venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase form25|payouts`, which need network access and a contact line in the `SEC_USER_AGENT` environment variable)

## How to reproduce

Run the driver from the repository root:

    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --help

When the study closes, `PIN.json` records the commit, lock hash, data hashes and the exact
command, and the study tool re-runs it in a scratch clone (the repro subcommand).
