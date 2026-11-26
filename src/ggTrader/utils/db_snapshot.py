"""Read table counts and selected content digests from a database."""

from typing import Any

from psycopg2 import sql

DIGEST_TABLES: tuple[str, ...] = (
    "ref.loads",
    "ref.releases",
    "universe.loads",
    "universe.membership",
    "public.paper_trades",
    "public.lab_runs",
)


def table_counts(conn: Any) -> dict[str, int]:
    """Count base tables, excluding PostgreSQL and Timescale internal schemas."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        )
        tables = cur.fetchall()
        counts = {}
        for schema, table in tables:
            if schema in {"pg_catalog", "information_schema"} or schema.startswith("_timescaledb_"):
                continue
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            counts[f"{schema}.{table}"] = cur.fetchone()[0]
        return counts


def table_digests(conn: Any) -> dict[str, str]:
    """Hash the ordered text representations of the selected tables."""
    with conn.cursor() as cur:
        digests = {}
        for name in DIGEST_TABLES:
            schema, table = name.split(".")
            cur.execute(
                sql.SQL(
                    "SELECT md5(coalesce(string_agg(t::text, '|' ORDER BY t::text), "
                    "'')) FROM {}.{} AS t"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            digests[name] = cur.fetchone()[0]
        return digests


def snapshot(conn: Any) -> dict:
    """Capture counts and digests."""
    return {"counts": table_counts(conn), "digests": table_digests(conn)}
