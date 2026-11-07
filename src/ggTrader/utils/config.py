import os

from dotenv import load_dotenv

from ggTrader.utils.paths import find_project_root


def _load_env() -> None:
    """Load environment variables from .env file in project root."""
    project_root = find_project_root()
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)


def get_db_connection_string() -> str:
    """
    Get the PostgreSQL database connection string.
    Checks for individual DB_HOST, DB_USER, etc. or falls back to POSTGRES_CONNECTION_STRING.
    """
    _load_env()

    # Priority 1: Individual environment variables (common in Docker)
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    if all([db_user, db_pass, db_host, db_name]):
        return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    # Priority 2: Full connection string
    conn_str = os.getenv("POSTGRES_CONNECTION_STRING")

    if not conn_str:
        raise ValueError(
            "Database configuration not found. Please set DB_HOST/USER/PASS/NAME "
            "or POSTGRES_CONNECTION_STRING in your .env or environment."
        )

    # Auto-fix for host.docker.internal when running on the host machine
    if "host.docker.internal" in conn_str:
        import socket

        try:
            socket.gethostbyname("host.docker.internal")
        except socket.gaierror:
            # Fallback to localhost if host.docker.internal doesn't resolve
            conn_str = conn_str.replace("host.docker.internal", "localhost")

    return conn_str
