import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from ggTrader.utils.paths import find_project_root


def load_symbols_from_json(file_path: str) -> Optional[List[str]]:
    """Loads symbols from a JSON file (list of strings or list of objects with 'symbol' key)."""
    if not Path(file_path).exists():
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return None

            symbols = []
            for item in data:
                if isinstance(item, str):
                    symbols.append(item)
                elif isinstance(item, dict) and "symbol" in item:
                    symbols.append(item["symbol"])
            return symbols
    except Exception as e:
        print(f"Error loading symbols from {file_path}: {e}")
        return None


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

    return conn_str


def get_alpaca_credentials(paper: bool = True) -> dict:
    """Return Alpaca API key, secret, and base URL from .env."""
    _load_env()
    if paper:
        return {
            "key_id": os.getenv("APCA_API_KEY_ID"),
            "secret_key": os.getenv("APCA_API_SECRET_KEY"),
            "base_url": os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        }
    return {
        "key_id": os.getenv("APCA_API_LIVE_KEY_ID"),
        "secret_key": os.getenv("APCA_API_LIVE_SECRET_KEY"),
        "base_url": "https://api.alpaca.markets",
    }
