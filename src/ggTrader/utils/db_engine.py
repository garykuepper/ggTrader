"""Shared SQLAlchemy engine creation for CLI tools and scripts."""

from __future__ import annotations

import sys
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from ggTrader.utils.config import get_db_connection_string


def create_db_engine(connection_string: Optional[str] = None) -> Engine:
    """Build a SQLAlchemy engine using env config or an explicit URL."""
    if connection_string is None:
        connection_string = get_db_connection_string()
    return create_engine(connection_string)


def create_db_engine_or_exit(connection_string: Optional[str] = None) -> Engine:
    """Like ``create_db_engine`` but print and ``sys.exit(1)`` on failure (CLI scripts)."""
    try:
        return create_db_engine(connection_string)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)
