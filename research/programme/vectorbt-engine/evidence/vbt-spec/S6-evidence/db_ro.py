"""Read-only DB helper for S6 (never writes; session forced read-only)."""
import psycopg2
from ggTrader.utils.config import get_db_connection_string

def ro_conn():
    cs = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
    c = psycopg2.connect(cs)
    c.set_session(readonly=True, autocommit=True)
    return c
