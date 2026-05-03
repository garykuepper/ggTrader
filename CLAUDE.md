# CLAUDE.md

## Project

ggTrader is an algorithmic crypto trading bot. It runs walk-forward optimization (WFO) research, then deploys optimized parameters to a live Kraken trader via Docker.

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
- **WFO cache**: Clear `results/wfo_cache/` when changing scoring config (composite weights, fold consistency, OOS alpha, N_SPLITS) — cached results use old settings.

## Core Systems

- **Monthly Recalibration**: The `ExecutionEngine` handles its own WFO run internally on the 1st of each month (~01:00 AM). It reloads the new parameters automatically once complete.
- **Tiered Regime Filter**: Entries are gated by correlation with BTC:
    - High ($\ge 0.5$): BTC Regime (`BTC_REGIME_FILTER_SHORT_EMA` vs 200-day EMA; default short = 20).
    - Medium ($0.3-0.5$): Altcoin Index.
    - Low ($< 0.3$): Free trading.
- **PNL Reporting**: Daily reports are sent at 08:00 AM local time. They include a "Market Regime" status compute on-the-fly from live `ccxt` data.
