"""Shared cached SQLAlchemy engine."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from ggTrader.utils.config import get_db_connection_string

_ENGINE = None


def get_engine() -> Engine:
    """Return the cached engine, creating it on first use."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_db_connection_string())
    return _ENGINE
