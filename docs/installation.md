# Installation

ggTrader is a research-first lab for walk-forward optimization of trading strategies. It requires Python 3.10+, a TimescaleDB instance, and (optionally) exchange API keys for data ingestion.

## 1. Python dependencies

Clone the repository and install the package:

```bash
git clone https://github.com/garykuepper/ggTrader.git
cd ggTrader
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify the installation:

```bash
ggt --help
```

This should show the three main commands: `lab`, `ingest`, `db`.

## 2. TimescaleDB

TimescaleDB is PostgreSQL with a time-series extension. The easiest path is Docker:

```bash
docker run -d \
  --name ggtrader_db \
  -p 5433:5432 \
  -e POSTGRES_USER=ggtrader \
  -e POSTGRES_PASSWORD=ggtrader \
  -e POSTGRES_DB=ggtrader \
  timescale/timescaledb:latest-pg16
```

The mapping `5433:5432` keeps it off the default Postgres port.

## 3. Environment

Copy `.env.example` to `.env` and fill in:

```text
# Database
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader

# Exchange API keys (optional, only needed for ggt ingest)
BINANCEUS_KEY=...
BINANCEUS_SECRET=...
KRAKEN_KEY=...
KRAKEN_SECRET=...

# Alerts (optional)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

For lab research (which uses yfinance for equities), exchange credentials are **not required**.

## 4. First run

Verify database connectivity:

```bash
ggt db diag
```

This should print table information if the database is reachable.

Run a lab strategy on a small dataset to confirm the pipeline works:

```bash
ggt lab --strategy wfo_tournament --top-n 10 --eval-start 2024-01-01 --eval-end 2024-12-31
```

If it completes and prints a run ID, you're set.

## 5. Docker (optional)

For containerized research (recommended in production), use the bundled `docker-compose.yaml`:

```bash
docker compose build --no-cache
docker compose up -d
```

Then run lab commands inside the container:

```bash
docker compose run --rm ggtrader_live python ggt.py lab --strategy wfo_tournament
```

The container automatically uses `host.docker.internal:5433` to reach the host's TimescaleDB.

## What's next

- [CLI Reference](cli_reference.md) — all commands and flags
- [Architecture](architecture.md) — how the lab works internally
- [agents.md](../agents.md) — coding standards and development guidelines

---

*Back to [README.md](../README.md).*
