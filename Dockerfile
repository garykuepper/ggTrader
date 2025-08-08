dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Set a non-root user
ARG APP_USER=appuser
ARG APP_HOME=/app

# Install OS packages required by pandas, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app dir and user
RUN useradd -m -d ${APP_HOME} ${APP_USER}
WORKDIR ${APP_HOME}

# Install Python dependencies (use the slim service requirements to keep image small)
# If you prefer to use your project-wide requirements.txt, copy that instead.
COPY requirements-live.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (adjust path if your script lives in a subfolder)
COPY . .

# Ensure state dir exists and is writable
RUN mkdir -p ${APP_HOME}/data && chown -R ${APP_USER}:${APP_USER} ${APP_HOME}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Drop privileges
USER ${APP_USER}

# Optional: healthcheck that pings an internal HTTP endpoint if you add one
# For now, we just check the process exists via PID 1 (handled by compose restart policy).
# HEALTHCHECK NONE

# Entrypoint: run the live trader
# Expects env vars for API keys and Mongo/Matrix in the environment or .env file
CMD ["python", "live_trader.py"]
