# A1 review fixes

Both loaders require the database drive path and check estimated space there; McGill checks each batch. CRSP estimates 160 bytes per row from Parquet metadata, and McGill uses an estimated 4096 bytes per row. The CLI requires a backup directory or an explicit opt-out, refused for database `ggtrader` before connecting. Backups under 1 MiB are rejected.

Verification: Ruff check and format passed on touched Python files. Named non-integration tests: 16 passed, 49 deselected. Scratch database integration tests: 49 passed, 8 deselected. Tests cover path selection, batch interruption and resume, backup size, and CLI safeguards. McGill's per-row estimate remains an estimate, not a measured database size.
