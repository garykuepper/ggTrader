# Installation Guide

Welcome! ggTrader is a research platform (or "lab") designed to test and tune trading strategies. It uses a method called **Walk-Forward Optimization**—which simulates trading by constantly adjusting strategy rules using past data and testing them on new data.

To run ggTrader, you will need:
1. **Python 3.10+**: The programming language used to run the software.
2. **TimescaleDB**: A specialized database built for handling time-series data (like price history over time).
3. **Exchange API Keys** (Optional): Digital credentials that allow ggTrader to download market data from exchanges like Kraken or Binance.

---

## 1. Setting Up Python

To start, you need to download the ggTrader source code and install its Python dependencies. We recommend using a **virtual environment** (`venv`), which is a private space on your computer that keeps these files separate from your other projects.

Run the following commands in your terminal:

```bash
# 1. Download the repository from GitHub
git clone https://github.com/garykuepper/ggTrader.git

# 2. Enter the project folder
cd ggTrader

# 3. Install uv (https://docs.astral.sh/uv/) if you do not have it, then create the
#    virtual environment ".venv" from the committed lock file. This installs the exact
#    versions that CI and the Docker image use (Python 3.12, pandas 3, numpy 2.5,
#    vectorbt 1.2.0 with its Rust engine).
uv sync --frozen --extra test --extra dev

# 4. Activate the virtual environment
# (On macOS/Linux use this command. On Windows, use: .venv\Scripts\activate)
source .venv/bin/activate
```

`--frozen` makes uv fail instead of re-resolving if `uv.lock` is out of date with
`pyproject.toml`; after editing a dependency, run `uv lock` and commit both files. Check that
vectorbt really runs on its Rust engine (the numba fallback costs 20-30 s of compile time per
fresh process) with `python scripts/check_vbt_engine.py`.

Verify that the command-line interface (CLI) is working by running:

```bash
ggt --help
```

You should see a list of the three main tools: `lab` (for simulations), `ingest` (for downloading data), and `db` (for database management).

---

## 2. Setting Up TimescaleDB

TimescaleDB is a popular database for storing price history. The simplest way to run it is through **Docker**, a tool that runs software in self-contained packages ("containers") without messing up your computer's settings.

If you have Docker installed, you can start the database with this command:

```bash
docker run -d \
  --name ggtrader_db \
  -p 5433:5432 \
  -e POSTGRES_USER=ggtrader \
  -e POSTGRES_PASSWORD=ggtrader \
  -e POSTGRES_DB=ggtrader \
  timescale/timescaledb:latest-pg16
```

> [!NOTE]
> The setting `-p 5433:5432` maps the database to port `5433` on your computer. This prevents conflicts if you already have standard PostgreSQL running on port `5432`.

---

## 3. Configuring the Environment

Settings and credentials (like passwords or API keys) are stored in a hidden file named `.env`. 

Copy the template file `.env.example` to a new file named `.env`:

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in the details:

```text
# Database connection settings
POSTGRES_CONNECTION_STRING=postgresql+psycopg2://ggtrader:ggtrader@localhost:5433/ggtrader

# Exchange API keys (only needed by archived crypto tooling; the equity research and paper path do not use them)
BINANCEUS_KEY=...
BINANCEUS_SECRET=...
KRAKEN_KEY=...
KRAKEN_SECRET=...

# Chat alerts (Optional: sends notifications to your phone/chat apps)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

> [!TIP]
> If you are just doing basic research (which downloads free stock data from Yahoo Finance), you **do not** need to provide exchange API keys.

---

## 4. Confirming Everything Works (First Run)

Let's test if ggTrader can connect to your database:

```bash
ggt db diag
```

If it successfully connects, it will print out table sizes and row counts (which will be empty at first).

Next, run a quick simulation to confirm that the entire backtesting pipeline works:

```bash
ggt lab --strategy ensemble --wfo --max-stocks 10 --eval-start 2021-01-31 --eval-end 2022-07-31
```

If the command finishes without errors and prints out a "run ID" and summary table, your installation is complete!

---

## 5. Running with Docker (Optional)

If you prefer running everything in containerized environments, you can use the provided `docker-compose.yaml` file:

```bash
# Build the ggTrader container image
docker compose build --no-cache

# Start the database and services in the background
docker compose up -d
```

To run simulations inside the Docker container, prefix your commands like this:

```bash
docker compose run --rm ggtrader_live python ggt.py lab --strategy ensemble --wfo --eval-end 2022-07-31 --max-stocks 40
```

The container is pre-configured to reach the database running on your host machine automatically.

---

## What's Next?

- [CLI Reference](cli_reference.md) — How to use all the different command-line flags.
- [Architecture Guide](architecture.md) — How the system works under the hood.
- [Developer Guidelines](../AGENTS.md) — Coding standards for modifying the project.

---

*Back to [README.md](../README.md).*
