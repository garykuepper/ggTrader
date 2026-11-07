# Slim Python 3.12, pinned by digest (multi-arch index resolved 2026-11-06 from Docker Hub;
# Dependabot proposes digest bumps within 3.12 only, see .github/dependabot.yml).
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# uv installs from the committed uv.lock, so the image carries exactly the versions CI
# and the research venv test. Before 2026-11 this ran `pip install -e ".[test]"` after
# `COPY . .`, which re-resolved every unpinned package on every source change.
# Pinned to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Dependencies first (cached until pyproject.toml or uv.lock change), then the source.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra test

COPY . .
# Editable project install on top of the locked dependency set.
RUN uv sync --frozen --extra test

# Fail the build, not the 12:45 live run, if a dependency breaks the trader's
# import chain (2026-09-23: unpinned plotly 7 broke `import vectorbt`). The trader's signal
# path imports ggTrader.engine, the validated v2 ensemble and the paper modules, so import the
# whole of it, and make sure the CLI entry point still loads.
RUN python -c "import ggTrader.paper.trader, ggTrader.paper.signal_runner, ggTrader.paper.overlay" \
    && python -u ggt.py paper --help > /dev/null

# The wheel set must be self-consistent, and the simulator must really run on the Rust
# engine: `engine="auto"` silently falls back to numba (+25 s JIT per fresh container)
# when vectorbt-rust is missing or minor-mismatched. The trader has no fallback engine, so the
# same probe runs as `python -m ggTrader.engine.selfcheck` (what the trader calls at start-up):
# it counts calls into the Rust simulator around one tiny trailing-stop `from_signals`.
RUN uv pip check \
    && python scripts/check_vbt_engine.py --engine rust \
    && python -m ggTrader.engine.selfcheck --engine rust

# Create the data directory for persistence
RUN mkdir -p /app/data

# Run as a non-root user. uid 1000 matches the host user (flynn) so files written
# to the mounted ./results, ./data, ./logs volumes are owned by the host user
# rather than root. chown covers /app so the editable install and the runtime
# .cache/indicators directory remain writable under the non-root user.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
# Same Rust checks as the non-root user the cron run uses (cache and permission paths differ).
RUN python scripts/check_vbt_engine.py --engine rust \
    && python -m ggTrader.engine.selfcheck --engine rust \
    && python -c "import ggTrader.paper.trader"

# Default: show help
CMD ["python", "-u", "ggt.py", "--help"]
