# CLAUDE.md

## Project

ggTrader is an algorithmic crypto trading bot. It runs walk-forward optimization (WFO) research, then deploys optimized parameters to a live Kraken/Binance.US trader via Docker.

## Documentation Rules

When making changes to the codebase, keep these docs updated:

- **docs/changelog.md** — Add an entry whenever strategies, config values, param grids, or infrastructure are changed. Include what changed, why, and research results if a run was done.
- **docs/future_tweaks_plan.md** — Update the "Current Live Configuration" section when new research params go live. Add new experiment ideas as they come up. Remove or mark completed experiments that have been tested.

## Development Guidelines

- **Config changes**: Test one at a time with a research run between each. Don't bundle multiple config tweaks — it makes it impossible to attribute results.
- **New strategies**: Can be bundled together since they're purely additive and don't affect existing strategy scoring.
- **Research runs**: Run inside Docker (`docker compose run --rm ggtrader_live python -u ggt.py research`). The DB uses `host.docker.internal` which only resolves inside containers.
- **Live trader**: Always rebuild before restarting (`docker compose build --no-cache && docker compose up -d`). The trader auto-detects the latest research params.
- **Code Updates**: The `src/` directory is **not** volume-mounted in production. To apply code changes without a full rebuild/restart, use `docker cp src/ggTrader/path/to/file.py ggtrader_live:/app/src/ggTrader/path/to/file.py`.
- **WFO cache**: Run `ggt db purge-wfo-cache` when changing scoring config (composite weights, fold consistency, OOS alpha, N_SPLITS) — cached results use old settings. The cache lives in the TimescaleDB `wfo_cache` table (migrated from `results/wfo_cache/` JSON files).

## Core Systems

- **Monthly Recalibration**: The `ExecutionEngine` handles its own WFO run internally on the 1st of each month (~01:00 AM). It reloads the new parameters automatically once complete.
- **2-Tier BTC Regime Filter** (`src/ggTrader/core/regime_filtering.py`, applied in `orchestrator._apply_tiered_regime_mask`): Coins with `corr_BTC ≥ LEADER_CORR_THRESHOLD` (default 0.7) only fire entries when `close > EMA(EMA_WARMUP_BARS=100)`; below the threshold they trade freely. `BTC_REGIME_FILTER_SHORT_EMA=None` means `close > long_EMA` (no short-EMA cross). Off by default (`BTC_REGIME_FILTER=False`). Live engine and research orchestrator both default the missing-symbol corr to **1.0** (conservative — gate by default).
- **PNL Reporting**: Daily reports are sent at 06:00 AM local time. They include a "Market Regime" status compute on-the-fly from live `ccxt` data.
