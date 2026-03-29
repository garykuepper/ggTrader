# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# We'll install in editable mode later if we copy the whole src
COPY pyproject.toml .
# We need to create a dummy src to satisfy setuptools during initial pip install if needed,
# but it's easier to just copy everything.
COPY . .

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

# Create the data directory for persistence
RUN mkdir -p /app/data

# Default command: Start Live Exec Engine natively with Unbuffered output for log tailing
CMD ["python", "-u", "ggt.py", "trade"]
