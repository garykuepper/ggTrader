# Archived legacy source (P10)

Source modules and their tests that left `src/ggTrader` and `tests/` on 2026-11-06 because nothing in the kept code imports them any more, but research reports cite them by file name. They are **not importable** from here (no package, outside `testpaths`) and are kept as the record. The same files also exist at git tag `legacy-engine-vbt0285`. Do not edit the copies.

Ledger with the import-graph evidence: [../P10-ledger.csv](../P10-ledger.csv). Completion note: [../P10-completion.md](../P10-completion.md).

| File (relative to this folder) | Why it is here |
|---|---|
| `src/ggTrader/paper/feature_gate.py` | The machine-learning entry gate (LightGBM P(up)). Falsified NO-GO 2026-06-28 and off by default; its hand-written RSI, Bollinger and ATR were the last hand-rolled indicators on the live signal path. The import and the `gate` hook are removed from `paper/signal_runner.py` and `paper/trader.py`. `models/ensemble_gate.joblib` (the model file it loaded) was not touched. |
| `src/ggTrader/lab/train_gate.py` | Trainer of that gate. |
| `src/ggTrader/lab/regime.py` | Regime-gated exposure helpers (EMA, rolling volatility, quantiles); only its own test imported it. |
| `src/ggTrader/lab/headline_sentiment_data.py`, `short_interest_data.py`, `short_volume_data.py`, `house_ptr_data.py`, `fomc_calendar.py` | Data loaders of strategies that now exist only at the legacy tag (`headline_sentiment`, `short_interest`, `short_volume_ratio`, `congress_trades`, `fomc_drift`). Their database tables were not touched. Their backfill drivers are in `../legacy-scripts/`. |
| `tests/lab/test_regime.py`, `test_train_gate.py`, `test_ml_screen.py`, `test_headline_sentiment_data.py`, `test_short_interest_data.py`, `test_short_volume_data.py`, `test_house_ptr_data.py`, `test_fomc_calendar.py`, `tests/paper/test_feature_gate.py` | The tests of the modules above. |

Deleted outright (no research report cites them; they stay in git history and at the tag): `src/ggTrader/data/historical/postgres_ingestor.py` and `src/ggTrader/cli/cmd_ingest.py` (the `ggt ingest` stub).

Kept on purpose, although the audit listed them: `form4_data.py` and `earnings_surprise_data.py` (their tables are read by `lab/event_timing_audit.py`, which the owner has not decided), `fred_data.py`, `google_trends_data.py`.
