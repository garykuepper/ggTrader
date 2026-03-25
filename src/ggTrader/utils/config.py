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
    Get the PostgreSQL database connection string from environment variables.

    Returns:
        str: Database connection string

    Raises:
        ValueError: If POSTGRES_CONNECTION_STRING is not set in .env
    """
    _load_env()

    conn_str = os.getenv("POSTGRES_CONNECTION_STRING")

    if not conn_str:
        raise ValueError(
            "POSTGRES_CONNECTION_STRING not found in environment variables. "
            "Please ensure it is set in the .env file at the project root."
        )

    return conn_str
