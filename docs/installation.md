# Installation

ggTrader needs three things to run: Python 3.10+, a TimescaleDB instance, and (for live trading) exchange API keys.

## 1. Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the project and the `ggt` CLI. Verify with:

```bash
python ggt.py --help
```

## 2. TimescaleDB

TimescaleDB is PostgreSQL with a time-series extension. Easiest path is Docker:

```bash
docker run -d \
  --name ggtrader_db \
  -p 5433:5432 \
  -e POSTGRES_USER=ggtrader \
  -e POSTGRES_PASSWORD=ggtrader \
  -e POSTGRES_DB=ggtrader \
  timescale/timescaledb:latest-pg16
```

(The `5433:5432` mapping keeps it off the default Postgres port if you have one running locally.)

## 3. Environment

Copy `.env.example` to `.env` and fill in:

```text
# Database
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader

# Exchange (set one or both; EXCHANGE config var selects the active venue)
# Binance.US is the lower-fee target (0.04% RT vs Kraken Pro 0.50–0.80% RT).
BINANCEUS_KEY=...
BINANCEUS_SECRET=...
KRAKEN_KEY=...
KRAKEN_SECRET=...

# Telegram + Discord (optional, for alerts)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

## 4. First data sync

```bash
# Verify DB connectivity
python ggt.py db diag

# Pull recent OHLCV for the live universe
python ggt.py ingest --days 180

# Optional: enable Timescale compression to shrink old data
python ggt.py db compression --enable
```

## 5. Sanity check

Run a research job on a small universe to confirm the pipeline works end-to-end:

```bash
python ggt.py research --top 10 --workers 2
```

If it finishes and writes a `research_report.md` to `results/research/research_<timestamp>/`, you're set.

## Docker (optional)

For running the live bot in a container while keeping the DB on the host, use the bundled `docker-compose.yaml`:

```bash
docker compose build --no-cache
docker compose up -d
```

The compose file uses `host.docker.internal` to reach the host's TimescaleDB, so make sure your `.env` `DB_HOST` matches.

## What's next

- [CLI Reference](cli_reference.md) — every command and flag
- [Live Trading Guide](live_trading_guide.md) — going from research output to live orders
- [Architecture](architecture.md) — what each piece does

---
*Back to [README.md](../README.md).*
