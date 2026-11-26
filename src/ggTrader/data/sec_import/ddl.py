"""Apply the consolidation schema after the Kaggle schema."""

from psycopg2.extensions import connection as PgConnection

from ggTrader.utils.paths import REPO_ROOT

COVERAGE_VALUES = (
    "includes_dead_stocks",
    "survivors_only",
    "partial_dead_stocks",
    "large_stocks_only",
    "unaudited_sample",
    "not_a_stock_source",
)


def apply_ddl(conn: PgConnection) -> None:
    """Apply the consolidation schema and backfill coverage in one transaction."""
    sql = (REPO_ROOT / "sql" / "consolidation_schema.sql").read_text(encoding="utf-8")
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '15s'")  # do not queue behind open readers
            cur.execute(sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
