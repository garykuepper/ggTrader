# dead-stock-data-integrity

**Question:** For the dead stocks the first test held, what do our CRSP rows and SEC filings say about why they stopped trading, and how clean are the price rows around them?

**Kind:** audit. **Status:** closed (2026-11-24, UNRESOLVED; see the reports). **Category:** stock-signals (placement row 9).

## Where things are

- Brief (registered plan): [2026-11-24-dead-stock-data-integrity-brief.md](2026-11-24-dead-stock-data-integrity-brief.md)
- Reports: [Phase 1 findings, our own price rows](2026-11-24-dead-stock-data-integrity.md) and [payout evidence from the SEC's filings](phase-2/2026-11-24-dead-stock-payout-evidence.md) (kept in `phase-2/` because the study tool allows one plainly named dated report at the top)
- Phase 3 (open cases; rules frozen first in [phase-3/2026-11-24-open-cases-brief.md](phase-3/2026-11-24-open-cases-brief.md), report [phase-3/2026-11-24-dead-stock-open-cases.md](phase-3/2026-11-24-dead-stock-open-cases.md)): `_payouts_v2.csv` and `_payouts_v2.json`, from `code/open_cases_run.py`
- Results: `_integrity.json`, `_roster.csv`, `_form25_matches.json`, `_payouts.json`, `_payouts.csv`; hand checks: [_hand_checks.txt](_hand_checks.txt); pin: [PIN.json](PIN.json) (pins the three results `_integrity.json`, `_form25_matches.json` and `_payouts.json`, and the roster and `_payouts.csv` as inputs; see the payout report, section 11)
- Study card: [study.toml](study.toml)
- Code: [code/](code/) (Phase 1 driver [code/run.py](code/run.py), module [code/dead_stock_data_integrity.py](code/dead_stock_data_integrity.py); SEC phases `.venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase form25|payouts`, which need network access and a contact line in the `SEC_USER_AGENT` environment variable)

## How to reproduce

Run these from the repository root, in this order. Prerequisites: release 1, CRSP load 1 and McGill load 2 in the `ggtrader` database; the contact line the SEC requires in the environment variable `SEC_USER_AGENT` (never written to a file); the saved SEC files in `/media/thesix/ggTrader-data/sec-delistings` (without them everything is downloaded again, about 360 requests; the Form 25 phase needs network access and `SEC_USER_AGENT` even with the saved folder, because six "not found" notices (HTTP 404) are never saved and are requested again; the payouts phase uses saved copies except for any filing list that was not saved). Each command needs `--force` because the result files already exist.

    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --phase integrity --release-id 1 --crsp-load 1 --mcgill-load 2 --force
    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase form25 --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/sec_pilot.py --phase payouts --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force
    .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/open_cases_run.py --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity --force

`PIN.json` records the commit, code and data hashes and the first command; the study tool's repro subcommand re-runs only that first command, so the two SEC phases are reproduced with the commands above.
